import time, os
import numpy as np
import logging
from types import MethodType
import torch as th
from torch import einsum
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
import torch
import math
from typing import Optional, Sequence
from openfold.utils.rigid_utils import Rigid

from qdiff.quant_layer import QuantModule, UniformAffineQuantizer, StraightThrough
from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, ResBlock, TimestepBlock, checkpoint
from ldm.modules.diffusionmodules.openaimodel import QKMatMul, SMVMatMul
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.attention import exists, default

from ddim.models.diffusion import ResnetBlock, AttnBlock, nonlinearity

from torch.nn import TransformerEncoderLayer, TransformerEncoder
from model.quant_score_network import Embedder
from model.quant_ipa_pytorch import InvariantPointAttention, permute_final_dims, flatten_final_dims, StructureModuleTransition, EdgeTransition, TorsionAngles
from data import utils as du

logger = logging.getLogger(__name__)

def multi_head_self_attn_forward(self, x, xk, xv, attn_mask=None,
                           key_padding_mask=None,
                           need_weights=False):         # getInpOut also goes here
    dropout_mhsa = 0.0
    h = self.num_heads
    
    # print('self._qkv_same_embed_dim:', self._qkv_same_embed_dim)  # True
    if not self._qkv_same_embed_dim:
        q = F.linear(x, self.w_quantizer_q(self.q_proj_weight))
        k = F.linear(xk, self.w_quantizer_k(self.k_proj_weight), self.bias_k)
        v = F.linear(xv, self.w_quantizer_v(self.v_proj_weight), self.bias_v)
    else:
        # print(x.shape, self.in_proj_weight.shape, self.w_quantizer_qkv(self.in_proj_weight).shape, self.in_proj_bias.shape, self.w_quantizer_qkv(self.in_proj_bias).shape)
        # qkv = F.linear(x, self.w_quantizer_qkv(self.in_proj_weight), self.in_proj_bias)
        qkv = self.in_proj(x)
        if attn_mask is not None:
            if len(x.shape) == len(attn_mask.shape):
                qkv *= (attn_mask[:x.shape[0], :, 0][... , None])
            else:
                qkv *= attn_mask[... , None]
        q, k, v = qkv.chunk(3, dim=-1)
    # print(q)    
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    # if attn_mask is not None:
    #     if attn_mask.dtype == torch.uint8:
    #         warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
    #         attn_mask = attn_mask.to(torch.bool)
    #     else:
    #         assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
    #             f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
    #     # ensure attn_mask's dim is 3
    #     if attn_mask.dim() == 2:
    #         correct_2d_size = (tgt_len, src_len)
    #         if attn_mask.shape != correct_2d_size:
    #             raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
    #         attn_mask = attn_mask.unsqueeze(0)
    #     elif attn_mask.dim() == 3:
    #         correct_3d_size = (bsz * num_heads, tgt_len, src_len)
    #         if attn_mask.shape != correct_3d_size:
    #             raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
    #     else:
    #         raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # merge key padding and attention masks

    # if key_padding_mask is not None:
    #     (bsz, src_len) = key_padding_mask.shape
    #     key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
    #         expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1, src_len)
    #     if attn_mask is None:
    #         attn_mask = key_padding_mask
    #     elif attn_mask.dtype == torch.bool:
    #         attn_mask = attn_mask.logical_or(key_padding_mask)
    #     else:
    #         attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # if attn_mask is not None and attn_mask.dtype == torch.bool:
    #     new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
    #     new_attn_mask.masked_fill_(attn_mask, float("-inf"))
    #     attn_mask = new_attn_mask
    
    self.scale = 1.0 / math.sqrt(q.shape[-1])
    if self.use_act_quant:
        # print('---------- tel goes here -----------')
        q_scaled = self.act_quantizer_q(q) * self.scale
        k = self.act_quantizer_k(k)
        # sim = einsum('b i d, b j d -> b i j', quant_q, quant_k) * self.scale
    else:
        q_scaled = q * self.scale   # fix BUG
        # sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    # fix BUG, no need of badbmm, ref. https://pytorch.org/docs/2.0/generated/torch.baddbmm.html#torch.baddbmm
    # if attn_mask is not None:       
    #     attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
    # else:
    #     attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    
    attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    # print(attn_output_weights)
    # add masked for filled residue, no real attention exist
    if attn_mask is not None:        # fill with -inf
        if not attn_mask.shape == attn_output_weights.shape:
            attn_mask = torch.tile(attn_mask[:,:, None] * attn_mask[:, None, :], (h, 1, 1))

        attn_output_weights = attn_output_weights.masked_fill(attn_mask==0, -10000.)
    # exit(0)
    # if exists(attn_mask):
    #     attn_mask = rearrange(attn_mask, 'b ... -> b (...)')
    #     max_neg_value = -th.finfo(sim.dtype).max
    #     attn_mask = repeat(attn_mask, 'b j -> (b h) () j', h=h)
    #     sim.masked_fill_(~attn_mask, max_neg_value)
    
    # attention, what we cannot get enough of
    attn = attn_output_weights.softmax(dim=-1)
    # print(attn[0][0], attn[-1][-1])

    if self.use_act_quant:
        out = torch.bmm(self.act_quantizer_w(attn), self.act_quantizer_v(v))
    else:
        out = torch.bmm(attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    # if dropout_mhsa > 0.0:
    #     out = nn.Dropout(out, p = dropout_mhsa)
    output = self.out_proj(out)
    # output = F.linear(out, self.out_proj.weight, self.out_proj.bias)
    
    if attn_mask is not None:
        # output = output.masked_fill(torch.tile(attn_mask[:output.shape[0], :, 0][..., None], (1, 1, output.shape[-1])) == 0, 0)
        output *= attn_mask[:output.shape[0], :, 0][... , None]
         
    # print('output:', output[0])
    # print('============goes self-defined forward?????? yes====================')
    # print(output)
    return output   # , None

class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    """
    def __init__(self, act_quant_params: dict = {}):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer

        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.activation_function = StraightThrough()

        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)

class QuantEmbedder(BaseQuantBlock):
    def __init__(self, emb: Embedder, act_quant_params: dict = {}, weight_quant_params: dict={}, sm_abit: int = 8):
        super().__init__(act_quant_params)
        self.name = 'quantemb'
        self._model_conf = emb._model_conf
        self._embed_conf = emb._embed_conf
        self.node_embedder = emb.node_embedder
        self.edge_embedder = emb.edge_embedder
        self.timestep_embedder = emb.timestep_embedder
        self.index_embedder = emb.index_embedder
        self._cross_concat = emb._cross_concat
        self.use_weight_quant = True
        self.use_act_quant = False
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        # self.disable_act_quant = True
        
    def forward(self,
            seq_idx,
            t,
            fixed_mask,
            self_conditioning_ca,):
        # print(self.name)
        num_batch, num_res = seq_idx.shape
        node_feats = []
        fixed_mask = fixed_mask[..., None]

        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1))

        prot_t_embed = torch.cat([prot_t_embed, fixed_mask], dim=-1)

        node_feats = [prot_t_embed]
        pair_feats = [self._cross_concat(prot_t_embed, num_batch, num_res)]

        # Positional index features.
        node_feats.append(self.index_embedder(seq_idx))

        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]
        rel_seq_offset = rel_seq_offset.reshape([num_batch, num_res**2])    # (1, seq**2)
        pair_feats.append(self.index_embedder(rel_seq_offset))      # (seq,seq, )

        # Self-conditioning distogram.
        if self._embed_conf.embed_self_conditioning:
            sc_dgram = du.calc_distogram(
                self_conditioning_ca,
                self._embed_conf.min_bin,
                self._embed_conf.max_bin,
                self._embed_conf.num_bins,
            )
            pair_feats.append(sc_dgram.reshape([num_batch, num_res**2, -1]))

        split_n = [tmp.shape[-1] for tmp in node_feats]     # [33, 32]
        split_e = [tmp.shape[-1] for tmp in pair_feats]     # [66, 32, 22]

        # add split optimization, to compare results
        # node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        node_embed = self.node_embedder[0](torch.cat(node_feats, dim=-1).float(), split = split_n)
        for layer in self.node_embedder[1:]:
            node_embed = layer(node_embed)
                
        # edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = self.edge_embedder[0](torch.cat(pair_feats, dim=-1).float(), split = split_e)
        for layer in self.edge_embedder[1:]:
            edge_embed = layer(edge_embed)
        
        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])

        return node_embed, edge_embed
        
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)

class QuantStructureModuleTransition(BaseQuantBlock):
    def __init__(self, smt: StructureModuleTransition, act_quant_params: dict = {}, weight_quant_params: dict={}, sm_abit: int = 8):
        super().__init__(act_quant_params)
        self.name = 'quantsmt'
        self.c = smt.c
        self.linear_1 = smt.linear_1
        self.linear_2 = smt.linear_2
        self.linear_3 = smt.linear_3
        self.relu = smt.relu
        self.ln = smt.ln    
        self.use_weight_quant = True
        self.use_act_quant = False
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        # self.disable_act_quant = True
        
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # print(self.name)
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        s = self.ln(s)
        return s    
        
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)

class QuantEdgeTransition(BaseQuantBlock):
    def __init__(self, et: EdgeTransition, act_quant_params: dict = {}, weight_quant_params: dict={}, sm_abit: int = 8):
        super().__init__(act_quant_params)
        self.name = 'quantet'
        self.initial_embed = et.initial_embed
        self.trunk = et.trunk
        self.final_layer = et.final_layer
        self.layer_norm = et.layer_norm
        self.use_weight_quant = True
        self.use_act_quant = False
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        # self.disable_act_quant = True
        
    def forward(self, node_embed, edge_embed):
        # print(self.name)
        # tag = time.time()
        # if self.use_act_quant:
        #     folder = os.path.join('dist', 'int')
        #     os.makedirs(folder, exist_ok=True)
        # else:
        #     folder = os.path.join('dist', 'fp')
        #     os.makedirs(folder, exist_ok=True)
        # name = os.path.join(folder, f'input_node_{tag}.npy')
        # np.save(name, node_embed.detach().cpu().numpy())
        
        node_embed = self.initial_embed(node_embed)
        
        # name = os.path.join(folder, f'input_edge_bias_{tag}.npy')
        # np.save(name, node_embed.detach().cpu().numpy())
        # name = os.path.join(folder, f'input_edge_{tag}.npy')
        # np.save(name, edge_embed.detach().cpu().numpy())
        
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat([
            torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
        ], axis=-1)
        
        split = [edge_embed.shape[-1], edge_bias.shape[-1]]
        # print('edge_embed:', edge_embed.shape)  # torch.Size([2, 512, 512, 128])
        # print('edge_bias:', edge_bias.shape)        #  torch.Size([2, 512, 512, 256])
        edge_embed = torch.cat(
            [edge_embed, edge_bias], axis=-1).reshape(
                batch_size * num_res**2, -1)
        # print('new edge_embed:', edge_embed.shape)
        
        # edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        orig_embed = edge_embed
        edge_embed = self.trunk[0](edge_embed, split=split)
        # name = os.path.join(folder, f'edge_0_{tag}.npy')
        # np.save(name, edge_embed.detach().cpu().numpy())
        
        
        for idx, layer in enumerate(self.trunk[1:]):
            edge_embed = layer(edge_embed)
            # name = os.path.join(folder, f'edge_{idx+1}_{tag}.npy')
            # np.save(name, edge_embed.detach().cpu().numpy())
            
        edge_embed = self.final_layer(edge_embed + orig_embed)
        # name = os.path.join(folder, f'edge_final_{tag}.npy')
        # np.save(name, edge_embed.detach().cpu().numpy())
        
        edge_embed = self.layer_norm(edge_embed)
        
        # name = os.path.join(folder, f'after_norm_edge_final_{tag}.npy')
        # np.save(name, edge_embed.detach().cpu().numpy())
        
        edge_embed = edge_embed.reshape(
            batch_size, num_res, num_res, -1
        )
        return edge_embed
        
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)
                
class QuantTorsionAngles(BaseQuantBlock):
    def __init__(self, ta: TorsionAngles, act_quant_params: dict = {}, weight_quant_params: dict={}, sm_abit: int = 8):
        super().__init__(act_quant_params)
        self.name = 'quantta'
        self.c = ta.c
        self.eps = ta.eps
        self.num_torsions = ta.num_torsions
        self.linear_1 = ta.linear_1
        self.linear_2 = ta.linear_2
        self.linear_final = ta.linear_final
        self.relu = ta.relu
        self.use_weight_quant = True
        self.use_act_quant = False
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        # self.disable_act_quant = True
        
    def forward(self, s: torch.Tensor):
        # print(self.name)
        # tag = time.time()
        # if self.use_act_quant:
        #     folder = os.path.join('dist-torsion', 'int')
        #     os.makedirs(folder, exist_ok=True)
        # else:
        #     folder = os.path.join('dist-torsion', 'fp')
        #     os.makedirs(folder, exist_ok=True)
        # name = os.path.join(folder, f's_initial_{tag}.npy')
        # np.save(name, s.detach().cpu().numpy())

        s_initial = s
        s = self.linear_1(s)
        
        # name = os.path.join(folder, f's_linear1_{tag}.npy')
        # np.save(name, s.detach().cpu().numpy())
        
        s = self.relu(s)
        
        # name = os.path.join(folder, f's_relu1_{tag}.npy')
        # np.save(name, s.detach().cpu().numpy())
        
        s = self.linear_2(s)
        
        # name = os.path.join(folder, f's_linear2_{tag}.npy')
        # np.save(name, s.detach().cpu().numpy())
        
        s = s + s_initial
        unnormalized_s = self.linear_final(s)
        
        # name = os.path.join(folder, f's_unnor_{tag}.npy')
        # np.save(name, unnormalized_s.detach().cpu().numpy())
        
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(unnormalized_s ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        normalized_s = unnormalized_s / norm_denom
        
        # name = os.path.join(folder, f's_nor_{tag}.npy')
        # np.save(name, normalized_s.detach().cpu().numpy())

        return unnormalized_s, normalized_s  
        
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)
        
class QuantIPA(BaseQuantBlock): 
    def __init__(self, ipa: InvariantPointAttention, act_quant_params: dict = {}, weight_quant_params: dict={}, sm_abit: int = 8):
        super().__init__(act_quant_params)
        self.name = 'quantipa'
        self.use_weight_quant = True
        self.use_act_quant = False
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        # self.disable_act_quant = True
        self._ipa_conf = ipa._ipa_conf
        self.c_s = ipa.c_s
        self.c_z = ipa.c_z
        self.c_hidden = ipa.c_hidden
        self.no_heads = ipa.no_heads
        self.no_qk_points = ipa.no_qk_points
        self.no_v_points = ipa.no_v_points
        self.inf = ipa.inf
        self.eps = ipa.eps
        self.linear_q = ipa.linear_q
        self.linear_kv = ipa.linear_kv
        self.linear_q_points = ipa.linear_q_points
        self.linear_kv_points = ipa.linear_kv_points
        self.linear_b = ipa.linear_b
        self.down_z = ipa.down_z
        self.head_weights = ipa.head_weights
        self.linear_out = ipa.linear_out            # split dim
        self.softmax = ipa.softmax
        self.softplus = ipa.softplus
        self.linear_rbf = ipa.linear_rbf
        self.split = 0
        
        self.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
        self.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
        # self.act_quantizer_ptdisp = UniformAffineQuantizer(**act_quant_params)
        act_quant_params_aupd = act_quant_params.copy()
        act_quant_params_aupd['n_bits'] = sm_abit
        act_quant_params_aupd['always_zero'] = True
        self.act_quantizer_aupd = UniformAffineQuantizer(**act_quant_params_aupd)
        self.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)
        self.act_quantizer_vpts = UniformAffineQuantizer(**act_quant_params)
        # self.act_quantizer_opt = UniformAffineQuantizer(**act_quant_params)
        self.act_quantizer_pairz = UniformAffineQuantizer(**act_quant_params)
        
        self.item_1 = torch.nn.Parameter(torch.tensor(math.sqrt(1.0 / 3)))         # BUG miss math.sqrt
        self.item_2 = torch.nn.Parameter(torch.tensor(math.sqrt(1.0 / 3)))
        self.item_3 = torch.nn.Parameter(torch.tensor(math.sqrt(1.0 / 3)))
        
    def forward(self, s: torch.Tensor,
            z: torch.Tensor,  #z: Optional[torch.Tensor],
            r: Rigid,
            mask: torch.Tensor,
            _offload_inference: bool = False,
            _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
            ) -> torch.Tensor:
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]
        # print(self.name)
        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        # tag = time.time()
        # if self.use_act_quant or self.use_weight_quant:
        #     folder = os.path.join('dist-ipa3-w', 'int')
        #     os.makedirs(folder, exist_ok=True)
        # else:
        #     folder = os.path.join('dist-ipa3-w', 'fp')
        #     os.makedirs(folder, exist_ok=True)
        # name = os.path.join(folder, f'z_initial_{tag}.npy')
        # np.save(name, z[0].detach().cpu().numpy())
        
        edge_mask = mask[..., None] * mask[..., None, :]
        q = self.linear_q(s) * mask[..., None]
        kv = self.linear_kv(s) * mask[..., None]
        # print('======goes ipa============')
        # print('q, kv: ', torch.isnan(q).any(), torch.isnan(kv).any())

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        # print(q.shape)  # torch.Size([1, 100, 8, 256])
        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))
        # print(q.shape, kv.shape)    # torch.Size([2, 512, 8, 256]) torch.Size([2, 512, 8, 512])

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s) * mask[..., None]

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        # print('q_pts 1:', torch.isnan(q_pts).any())   
        # print('r: ', r)
        q_pts = r[..., None].apply(q_pts)
        # print('q_pts 2:', torch.isnan(q_pts).any())      # nan
        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )
        # print(q_pts.shape)  # torch.Size([1, 100, 8, 8, 3])
        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s) * mask[..., None]

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )
        # print(k_pts.shape, v_pts.shape) # torch.Size([1, 100, 8, 8, 3]) torch.Size([1, 100, 8, 12, 3])

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0]) * edge_mask[..., None]
        
        
        # name = os.path.join(folder, f'q_{tag}.npy')
        # np.save(name, q.detach().cpu().numpy())
        # name = os.path.join(folder, f'k_{tag}.npy')
        # np.save(name, k.detach().cpu().numpy())
        # name = os.path.join(folder, f'v_{tag}.npy')
        # np.save(name, v.detach().cpu().numpy())
        # name = os.path.join(folder, f'q_pt_{tag}.npy')
        # np.save(name, q_pts.detach().cpu().numpy())
        # name = os.path.join(folder, f'k_pt_{tag}.npy')
        # np.save(name, k_pts.detach().cpu().numpy())
        # name = os.path.join(folder, f'v_pt_{tag}.npy')
        # np.save(name, v_pts.detach().cpu().numpy())
        # name = os.path.join(folder, f'b_{tag}.npy')
        # np.save(name, b.detach().cpu().numpy())
        
        
        if(_offload_inference):
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        # TODO add if else act_quant to q, k
        if not self.use_act_quant:
            a = torch.matmul(
                permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
                permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
            )
        else:
            a = torch.matmul(
                permute_final_dims(self.act_quantizer_q(q), (1, 0, 2)),  # [*, H, N_res, C_hidden]
                permute_final_dims(self.act_quantizer_k(k), (1, 2, 0)),  # [*, H, C_hidden, N_res]
            )
            
        a *= math.sqrt(1.0 / self.c_hidden) * self.item_1
        a += (self.item_2 * permute_final_dims(b, (2, 0, 1)))
        # print('a', a.shape) # torch.Size([1, 8, 100, 100])
        # exit(0)

        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        # TODO add if else act_quant to pt_displacement, UNDO
        # if not self.use_act_quant:
        #     pt_att = pt_displacement ** 2
        # else:
        #     # print('-------- ipa forward use act quantizer ---------')
        #     pt_att = self.act_quantizer_ptdisp(pt_displacement) ** 2
        pt_att = pt_displacement ** 2
        
        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        # print(pt_att.shape) # torch.Size([1, 100, 100, 8, 8])
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        # head_weights = head_weights * math.sqrt(
        #     1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        # )
        head_weights = head_weights * math.sqrt(
            1.0 / (self.no_qk_points * 9.0 / 2)
        ) * self.item_3
        pt_att = pt_att * head_weights
        # print(head_weights.shape) # torch.Size([1, 1, 1, 8, 1])

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        # print('pt_att', pt_att.shape)   # torch.Size([1, 8, 100, 100])
        
        a = a + pt_att 
        a = a + square_mask.unsqueeze(-3)
        
        # name = os.path.join(folder, f'a_bef_sm_{tag}.npy')
        # np.save(name, a.detach().cpu().numpy())
        
        a = self.softmax(a)
        # print('a', a.shape)     # torch.Size([1, 8, 100, 100])
        
        # name = os.path.join(folder, f'a_aft_sm_{tag}.npy')
        # np.save(name, a.detach().cpu().numpy())
        
        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        # TODO add if else act_quant to a, v, a needs sm_abits quant
        if not self.use_act_quant:
            o = torch.matmul(
                a, v.transpose(-2, -3).to(dtype=a.dtype)
            ).transpose(-2, -3)
        else:
            o = torch.matmul(
                self.act_quantizer_aupd(a), self.act_quantizer_v(v).transpose(-2, -3).to(dtype=a.dtype)
            ).transpose(-2, -3)
        # print('o', o.shape)     # torch.Size([1, 100, 8, 256])
        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v] 
        # # TODO add if else act_quant to a, v_pts
        if not self.use_act_quant:
            o_pt = torch.sum(
                (
                    a[..., None, :, :, None]
                    * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
                ),
                dim=-2,
            )
        else:
            o_pt = torch.sum(
                (
                    self.act_quantizer_aupd(a)[..., None, :, :, None]
                    * permute_final_dims(self.act_quantizer_vpts(v_pts), (1, 3, 0, 2))[..., None, :, :]
                ),
                dim=-2,
            )
        # print(a[..., None, :, :, None].shape, permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :].shape)
        # torch.Size([1, 8, 1, 100, 100, 1]) torch.Size([1, 8, 3, 1, 100, 12])
        # print(o_pt.shape)   # torch.Size([1, 8, 3, 100, 12])
        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)
        # print(o_pt.shape)   # torch.Size([1, 100, 8, 12, 3])

        # [*, N_res, H * P_v]
        # TODO add if else act_quant to o_pt, UNDO
        # if not self.use_act_quant:
        #     o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        # else:
        #     o_pt_dists = torch.sqrt(torch.sum(self.act_quantizer_opt(o_pt) ** 2, dim=-1) + self.eps)
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        # print(o_pt_dists.shape)     # torch.Size([1, 100, 8, 12])
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if(_offload_inference):
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z // 4]
        pair_z = self.down_z(z[0]).to(dtype=a.dtype) * edge_mask[..., None]
        # TODO add if else act_quant to a, pair_z
        if not self.use_act_quant:
            o_pair = torch.matmul(a.transpose(-2, -3), pair_z)
        else:
            o_pair = torch.matmul(self.act_quantizer_aupd(a).transpose(-2, -3), self.act_quantizer_pairz(pair_z))
        # print(pair_z.shape, o_pair.shape)       # torch.Size([1, 100, 100, 32]) torch.Size([1, 100, 8, 32])
        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]
        
        # name = os.path.join(folder, f'o_{tag}.npy')
        # np.save(name, o.detach().cpu().numpy())
        # name = os.path.join(folder, f'o_pt_{tag}.npy')
        # np.save(name, o_pt.detach().cpu().numpy())
        # name = os.path.join(folder, f'o_pt_norm_{tag}.npy')
        # np.save(name, o_pt_norm_feats.detach().cpu().numpy())
        # name = os.path.join(folder, f'a_pair_{tag}.npy')
        # np.save(name, o_pair.detach().cpu().numpy())
        
        
        # print(torch.cat(
        #         o_feats, dim=-1
        #     ).shape)        # torch.Size([1, 100, 2688])

        # [*, N_res, C_s]
        split = [tmp.shape[-1] for tmp in o_feats]      # [2048, 96, 96, 96, 96, 256]
        # print(split)
        # exit(0)
        s = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            ).to(dtype=z[0].dtype), split = split
            ) * mask[..., None]

        # print('==========finish ipa===============')
        # name = os.path.join(folder, f's_{tag}.npy')
        # np.save(name, s.detach().cpu().numpy())
        return s    
        
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)
        
                        
class QuantTransformerEncoderLayer(BaseQuantBlock):
    def __init__(self, enc: TransformerEncoderLayer, act_quant_params: dict = {}, weight_quant_params: dict={},
            sm_abit: int = 8):
        super().__init__(act_quant_params)
        self.use_weight_quant = True
        self.use_act_quant = False
        self.name = 'quanttel'
        self.self_attn = enc.self_attn
                                
        self.linear1 = enc.linear1
        self.dropout = enc.dropout

        self.linear2 = enc.linear2
        self.norm_first = enc.norm_first
        self.norm1 = enc.norm1
        self.norm2 = enc.norm2
        self.dropout1 = enc.dropout1
        self.dropout2 = enc.dropout2
        self.activation = enc.activation
        self.activation_relu_or_gelu = enc.activation_relu_or_gelu
        
        # logger.info(f"quant attn matmul")
        self.self_attn.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
        self.self_attn.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
        self.self_attn.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)
        # self.self_attn.act_quantizer_out = UniformAffineQuantizer(**act_quant_params)   # No need, '(out_proj): NonDynamicallyQuantizableLinear' is a nn.Linear module, act_quantizer is defined

        act_quant_params_w = act_quant_params.copy()
        act_quant_params_w['n_bits'] = sm_abit
        act_quant_params_w['always_zero'] = True
        self.self_attn.act_quantizer_w = UniformAffineQuantizer(**act_quant_params_w)
        
        self.self_attn.forward = MethodType(multi_head_self_attn_forward, self.self_attn)
        self.self_attn.use_act_quant = False
        self.self_attn.use_weight_quant = True
        
        self._qkv_same_embed_dim = self.self_attn._qkv_same_embed_dim
        # assert self._qkv_same_embed_dim == True
        if not self._qkv_same_embed_dim:        # TODO
            self.self_attn.w_quantizer_q = UniformAffineQuantizer(**weight_quant_params)
            self.self_attn.w_quantizer_k = UniformAffineQuantizer(**weight_quant_params)
            self.self_attn.w_quantizer_v = UniformAffineQuantizer(**weight_quant_params)
        else:
            # self.self_attn.w_quantizer_qkv = UniformAffineQuantizer(**weight_quant_params)
            # print(self.self_attn.in_proj_weight.shape)
            self.self_attn.in_proj = torch.nn.Linear(self.self_attn.in_proj_weight.shape[0], self.self_attn.in_proj_weight.shape[1])
            self.self_attn.in_proj.weight = torch.nn.Parameter(self.self_attn.in_proj_weight)
            self.self_attn.in_proj.bias = torch.nn.Parameter(self.self_attn.in_proj_bias)
            setattr(self.self_attn, 'in_proj', QuantModule(
                    self.self_attn.in_proj, weight_quant_params, act_quant_params))
        # self.self_attn.w_quantizer_out = UniformAffineQuantizer(**weight_quant_params)


    def forward(self, src, src_mask: Optional[torch.Tensor],
                src_key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
                # src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # print(self.name)
        # print('======goes tel==========')
        # tag = time.time()
        # if self.use_act_quant or self.use_weight_quant:
        #     folder = os.path.join('dist-1tel-fix-bug', 'int')
        #     os.makedirs(folder, exist_ok=True)
        # else:
        #     folder = os.path.join('dist-1tel-fix-bug', 'fp')
        #     os.makedirs(folder, exist_ok=True)
        # name = os.path.join(folder, f'src_{tag}.npy')
        # np.save(name, src.detach().cpu().numpy())
        # name = os.path.join(folder, f'src_mask_{tag}.npy')
        # np.save(name, src_mask.detach().cpu().numpy())
        # print(self.norm_first)    # False
        x = src
        if len(src_mask.shape) == len(x.shape):
            if self.norm_first:         
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask) * src_mask[:x.shape[0], :, 0][..., None] 
                x = x + self._ff_block(self.norm2(x), src_mask) * src_mask[:x.shape[0], :, 0][..., None] 
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask) * src_mask[:x.shape[0], :, 0][..., None])
                # name = os.path.join(folder, f'x_sa_{tag}.npy')
                # np.save(name, x.detach().cpu().numpy())
                x = self.norm2(x + self._ff_block(x, src_mask) * src_mask[:x.shape[0], :, 0][..., None])
                # name = os.path.join(folder, f'x_ff_{tag}.npy')
                # np.save(name, x.detach().cpu().numpy())
        
        else:
            if self.norm_first:         
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask) * src_mask[..., None]
                x = x + self._ff_block(self.norm2(x), src_mask) * src_mask[..., None]
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask) * src_mask[..., None])
                # name = os.path.join(folder, f'x_sa_{tag}.npy')
                # np.save(name, x.detach().cpu().numpy())
                x = self.norm2(x + self._ff_block(x, src_mask) * src_mask[..., None])
                # name = os.path.join(folder, f'x_ff_{tag}.npy')
                # np.save(name, x.detach().cpu().numpy())
        # print(x.shape, src_mask.shape)
        
        if len(src_mask.shape) == len(x.shape):
            x *=  src_mask[:x.shape[0], :, 0][..., None]         # Fix bug
        else:
            x *= src_mask[..., None]
        # name = os.path.join(folder, f'x_out_{tag}.npy')
        # np.save(name, x.detach().cpu().numpy())
        return x
    
    def _sa_block(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], 
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)      # [0]  self impl only return one variable
        return self.dropout1(x)
    
    def _ff_block(self, x: torch.Tensor, src_mask) -> torch.Tensor:
        if len(src_mask.shape) == len(x.shape):
            x = self.linear2(self.dropout(self.activation(self.linear1(x)) * src_mask[:x.shape[0], :, 0][..., None]))
        else:
            x = self.linear2(self.dropout(self.activation(self.linear1(x)) * src_mask[..., None]))
        return self.dropout2(x)
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.self_attn.use_act_quant = act_quant
        self.self_attn.use_weight_quant = weight_quant

        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)
                        
class QuantTransformerEncoder(BaseQuantBlock):
    def __init__(self, teall: TransformerEncoder, act_quant_params: dict = {}, weight_quant_params: dict={},
            sm_abit: int = 8):
        super().__init__(act_quant_params)
        self.use_weight_quant = True
        self.use_act_quant = False
        self.name = 'quantteall'
        self.num_layers = teall.num_layers
        self.layers = teall.layers
        self.sm_abit = sm_abit
        # for idx in range(len(self.layers)):
        #     setattr(module, str(idx), QuantTransformerEncoderLayer(child_module,
        #                 act_quant_params, weight_quant_params, sm_abit=self.sm_abit))
 
    def forward(self, src, mask: Optional[torch.Tensor],
                src_key_padding_mask: Optional[torch.Tensor] = None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output
     
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_act_quant = act_quant
        self.use_weight_quant = weight_quant

        for m in self.layers():
            m.set_quant_state(weight_quant, act_quant)   
            
            
class QuantResBlock(BaseQuantBlock, TimestepBlock):
    def __init__(
        self, res: ResBlock, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.channels = res.channels
        self.emb_channels = res.emb_channels
        self.dropout = res.dropout
        self.out_channels = res.out_channels
        self.use_conv = res.use_conv
        self.use_checkpoint = res.use_checkpoint
        self.use_scale_shift_norm = res.use_scale_shift_norm

        self.in_layers = res.in_layers

        self.updown = res.updown

        self.h_upd = res.h_upd
        self.x_upd = res.x_upd

        self.emb_layers = res.emb_layers
        self.out_layers = res.out_layers

        self.skip_connection = res.skip_connection

    def forward(self, x, emb=None, split=0):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if split != 0 and self.skip_connection.split == 0:
            return checkpoint(
                self._forward, (x, emb, split), self.parameters(), self.use_checkpoint
            )
        return checkpoint(
                self._forward, (x, emb), self.parameters(), self.use_checkpoint
            )  

    def _forward(self, x, emb, split=0):
        # print(f"x shape {x.shape} emb shape {emb.shape}")
        if emb is None:
            assert(len(x) == 2)
            x, emb = x
        assert x.shape[2] == x.shape[3]

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        if split != 0:
            return self.skip_connection(x, split=split) + h
        return self.skip_connection(x) + h


class QuantQKMatMul(BaseQuantBlock):
    def __init__(
        self, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.scale = None
        self.use_act_quant = False
        self.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
        self.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
        
    def forward(self, q, k):
        if self.use_act_quant:
            quant_q = self.act_quantizer_q(q * self.scale)
            quant_k = self.act_quantizer_k(k * self.scale)
            weight = th.einsum(
                "bct,bcs->bts", quant_q, quant_k
            ) 
        else:
            weight = th.einsum(
                "bct,bcs->bts", q * self.scale, k * self.scale
            )
        return weight

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_act_quant = act_quant


class QuantSMVMatMul(BaseQuantBlock):
    def __init__(
        self, act_quant_params: dict = {}, sm_abit=8):
        super().__init__(act_quant_params)
        self.use_act_quant = False
        self.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)
        act_quant_params_w = act_quant_params.copy()
        act_quant_params_w['n_bits'] = sm_abit
        act_quant_params_w['symmetric'] = False
        act_quant_params_w['always_zero'] = True
        self.act_quantizer_w = UniformAffineQuantizer(**act_quant_params_w)
        
    def forward(self, weight, v):
        if self.use_act_quant:
            a = th.einsum("bts,bcs->bct", self.act_quantizer_w(weight), self.act_quantizer_v(v))
        else:
            a = th.einsum("bts,bcs->bct", weight, v)
        return a

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_act_quant = act_quant


class QuantAttentionBlock(BaseQuantBlock):
    def __init__(
        self, attn: AttentionBlock, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.channels = attn.channels
        self.num_heads = attn.num_heads
        self.use_checkpoint = attn.use_checkpoint
        self.norm = attn.norm
        self.qkv = attn.qkv
        
        self.attention = attn.attention

        self.proj_out = attn.proj_out

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

    
def cross_attn_forward(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    if self.use_act_quant:
        quant_q = self.act_quantizer_q(q)
        quant_k = self.act_quantizer_k(k)
        sim = einsum('b i d, b j d -> b i j', quant_q, quant_k) * self.scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    if exists(mask):
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -th.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)

    if self.use_act_quant:
        out = einsum('b i j, b j d -> b i d', self.act_quantizer_w(attn), self.act_quantizer_v(v))
    else:
        out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)


class QuantBasicTransformerBlock(BaseQuantBlock):
    def __init__(
        self, tran: BasicTransformerBlock, act_quant_params: dict = {}, 
        sm_abit: int = 8):
        super().__init__(act_quant_params)
        self.attn1 = tran.attn1
        self.ff = tran.ff
        self.attn2 = tran.attn2
        
        self.norm1 = tran.norm1
        self.norm2 = tran.norm2
        self.norm3 = tran.norm3
        self.checkpoint = tran.checkpoint
        # self.checkpoint = False

        # logger.info(f"quant attn matmul")
        self.attn1.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
        self.attn1.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
        self.attn1.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)

        self.attn2.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
        self.attn2.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
        self.attn2.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)
        
        act_quant_params_w = act_quant_params.copy()
        act_quant_params_w['n_bits'] = sm_abit
        act_quant_params_w['always_zero'] = True
        self.attn1.act_quantizer_w = UniformAffineQuantizer(**act_quant_params_w)
        self.attn2.act_quantizer_w = UniformAffineQuantizer(**act_quant_params_w)

        self.attn1.forward = MethodType(cross_attn_forward, self.attn1)
        self.attn2.forward = MethodType(cross_attn_forward, self.attn2)
        self.attn1.use_act_quant = False
        self.attn2.use_act_quant = False

    def forward(self, x, context=None):
        # print(f"x shape {x.shape} context shape {context.shape}")
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        if context is None:
            assert(len(x) == 2)
            x, context = x

        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.attn1.use_act_quant = act_quant
        self.attn2.use_act_quant = act_quant

        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)


# the two classes below are for DDIM CIFAR
class QuantResnetBlock(BaseQuantBlock):
    def __init__(
        self, res: ResnetBlock, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.in_channels = res.in_channels
        self.out_channels = res.out_channels
        self.use_conv_shortcut = res.use_conv_shortcut

        self.norm1 = res.norm1
        self.conv1 = res.conv1
        self.temb_proj = res.temb_proj
        self.norm2 = res.norm2
        self.dropout = res.dropout
        self.conv2 = res.conv2
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = res.conv_shortcut
            else:
                self.nin_shortcut = res.nin_shortcut


    def forward(self, x, temb=None, split=0):
        if temb is None:
            assert(len(x) == 2)
            x, temb = x

        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x, split=split)
        out = x + h
        return out


class QuantAttnBlock(BaseQuantBlock):
    def __init__(
        self, attn: AttnBlock, act_quant_params: dict = {}, sm_abit=8):
        super().__init__(act_quant_params)
        self.in_channels = attn.in_channels

        self.norm = attn.norm
        self.q = attn.q
        self.k = attn.k
        self.v = attn.v
        self.proj_out = attn.proj_out

        self.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
        self.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
        self.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)
        
        act_quant_params_w = act_quant_params.copy()
        act_quant_params_w['n_bits'] = sm_abit
        self.act_quantizer_w = UniformAffineQuantizer(**act_quant_params_w)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        if self.use_act_quant:
            q = self.act_quantizer_q(q)
            k = self.act_quantizer_k(k)
        w_ = th.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        if self.use_act_quant:
            v = self.act_quantizer_v(v)
            w_ = self.act_quantizer_w(w_)
        h_ = th.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        
        out = x + h_
        return out


def get_specials(quant_act=False):
    specials = {
        ResBlock: QuantResBlock,
        BasicTransformerBlock: QuantBasicTransformerBlock,
        ResnetBlock: QuantResnetBlock,
        AttnBlock: QuantAttnBlock,
        InvariantPointAttention: QuantIPA,
        TransformerEncoderLayer: QuantTransformerEncoderLayer,
        StructureModuleTransition: QuantStructureModuleTransition,
        EdgeTransition: QuantEdgeTransition,
        TorsionAngles: QuantTorsionAngles,
        Embedder: QuantEmbedder,
    }
    if quant_act:
        specials[QKMatMul] = QuantQKMatMul
        specials[SMVMatMul] = QuantSMVMatMul
    else:
        specials[AttentionBlock] = QuantAttentionBlock
    return specials
