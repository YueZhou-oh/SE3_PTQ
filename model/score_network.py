"""Score network module."""
import torch
import math
from torch import nn
from torch.nn import functional as F
from data import utils as du
from data import all_atom
from model import ipa_pytorch
import functools as fn

Tensor = torch.Tensor


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class Embedder(nn.Module):

    def __init__(self, model_conf):
        super(Embedder, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed

        # Time step embedding
        index_embed_size = self._embed_conf.index_embed_size
        t_embed_size = index_embed_size     # 32
        node_embed_dims = t_embed_size + 1  # 33
        edge_in = (t_embed_size + 1) * 2    # 66

        # Sequence index embedding
        node_embed_dims += index_embed_size # 65
        edge_in += index_embed_size         # 98

        node_embed_size = self._model_conf.node_embed_size      # 256
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        if self._embed_conf.embed_self_conditioning:
            edge_in += self._embed_conf.num_bins            # 120
        edge_embed_size = self._model_conf.edge_embed_size  # 128
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=self._embed_conf.index_embed_size
        )
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=self._embed_conf.index_embed_size
        )

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res**2, -1])

    def forward(
            self,
            *,
            seq_idx,
            t,
            fixed_mask,
            self_conditioning_ca,
        ):
        """Embeds a set of inputs

        Args:
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.
            self_conditioning_ca: [..., N, 3] Ca positions of self-conditioning
                input.

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        # print(seq_idx.shape)
        num_batch, num_res = seq_idx.shape
        node_feats = []
        # print(t,num_batch,num_res)      # t decrease from 1 to min_t during inference
        # print(self.timestep_embedder(t))        # time_embed_dim = index_embed_dim = 32, defined in base.yaml
        # exit(0)
        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]
        # print(fixed_mask.shape) #torch.Size([1, 100, 1])
        # print(t.shape)
        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1))
        # print(prot_t_embed.shape)   # torch.Size([1, 100, 32])
        prot_t_embed = torch.cat([prot_t_embed, fixed_mask], dim=-1)
        # print(prot_t_embed.shape)     # torch.Size([1, 100, 33])
        node_feats = [prot_t_embed]
        pair_feats = [self._cross_concat(prot_t_embed, num_batch, num_res)]
        # print(pair_feats[0].shape)      # torch.Size([1, 10000, 66])

        # Positional index features.
        node_feats.append(self.index_embedder(seq_idx))
        # print(self.index_embedder(seq_idx).shape) # torch.Size([1, 100, 32])
        # exit(0)
        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]
        rel_seq_offset = rel_seq_offset.reshape([num_batch, num_res**2])    # (1, seq**2)
        pair_feats.append(self.index_embedder(rel_seq_offset))      # (seq,seq, )
        # print(pair_feats[1].shape)  # torch.Size([1, 10000, 32])

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
        # print(split_n, split_e)
        # exit(0)
        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])
        # print(node_embed.shape)
        # print(edge_embed.shape)
        # exit(0)
        return node_embed, edge_embed


class ScoreNetwork(nn.Module):

    def __init__(self, model_conf, diffuser):
        super(ScoreNetwork, self).__init__()
        self._model_conf = model_conf

        self.embedding_layer = Embedder(model_conf)
        self.diffuser = diffuser
        self.score_model = ipa_pytorch.IpaScore(model_conf, diffuser)

    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def forward(self, input_feats):
        """Forward computes the reverse diffusion conditionals p(X^t|X^{t+1})
        for each item in the batch

        Args:
            X: the noised samples from the noising process, of shape [Batch, N, D].
                Where the T time steps are t=1,...,T (i.e. not including the un-noised X^0)
            input aatype: torch.Size([42, 109])
            input seq_idx: torch.Size([42, 109])
            input chain_idx: torch.Size([42, 109])
            input residx_atom14_to_atom37: torch.Size([42, 109, 14])
            input residue_index: torch.Size([42, 109])
            input res_mask: torch.Size([42, 109])
            input atom37_pos: torch.Size([42, 109, 37, 3])
            input atom37_mask: torch.Size([42, 109, 37])
            input atom14_pos: torch.Size([42, 109, 14, 3])
            input rigidgroups_0: torch.Size([42, 109, 8, 4, 4])
            input torsion_angles_sin_cos: torch.Size([42, 109, 7, 2])
            input fixed_mask: torch.Size([42, 109])
            input sc_ca_t: torch.Size([42, 109, 3])
            input trans_score: torch.Size([42, 109, 3])
            input rot_score: torch.Size([42, 109, 3])
            input t: torch.Size([42])
            input rot_score_scaling: torch.Size([42])
            input trans_score_scaling: torch.Size([42])
            input rigids_0: torch.Size([42, 109, 7])
            input rigids_t: torch.Size([42, 109, 7])
        Returns:
            model_out: dictionary of model outputs.
            output psi: torch.Size([42, 109, 2])
            output rot_score: torch.Size([42, 109, 3])
            output trans_score: torch.Size([42, 109, 3])
            output rigids: torch.Size([42, 109, 7])
            output atom37: torch.Size([42, 109, 37, 3])
            output atom14: torch.Size([42, 109, 14, 3])
        """
        # print(input_feats['seq_idx'][2])    # range(x)
        # # print(input_feats['aatype'][1])     # amino acid type
        # # print(input_feats['residue_index'][1])  # seq_idx+1
        # print(input_feats['sc_ca_t'][2])        # all zero
        # # print(input_feats['torsion_angles_sin_cos'][1])
        # print(input_feats['t'])   # random(0,1)
        # print(input_feats['res_mask'][2])
        # print(input_feats['fixed_mask'][2])
        # # print(input_feats['rigids_t'][1])   # generated through (multiple noisy process/one step through DDIM theory) on gt rigid
        # exit(0)

        # Frames as [batch, res, 7] tensors.
        bb_mask = input_feats['res_mask'].type(torch.float32)  # [B, N]
        fixed_mask = input_feats['fixed_mask'].type(torch.float32)
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]

        # Initial embeddings of positonal and relative indices.
        init_node_embed, init_edge_embed = self.embedding_layer(
            seq_idx=input_feats['seq_idx'],
            t=input_feats['t'],
            fixed_mask=fixed_mask,
            self_conditioning_ca=input_feats['sc_ca_t'],
        )
        edge_embed = init_edge_embed * edge_mask[..., None] # [B, N, N, D_edge]
        node_embed = init_node_embed * bb_mask[..., None]   # [B, N, D_node]

        # Run main network
        model_out = self.score_model(node_embed, edge_embed, input_feats)

        # Psi angle prediction
        gt_psi = input_feats['torsion_angles_sin_cos'][..., 2, :]
        psi_pred = self._apply_mask(
            model_out['psi'], gt_psi, 1 - fixed_mask[..., None])

        pred_out = {
            'psi': psi_pred,
            'rot_score': model_out['rot_score'],
            'trans_score': model_out['trans_score'],
        }
        rigids_pred = model_out['final_rigids']
        pred_out['rigids'] = rigids_pred.to_tensor_7()
        bb_representations = all_atom.compute_backbone(rigids_pred, psi_pred)
        pred_out['atom37'] = bb_representations[0].to(rigids_pred.device)
        pred_out['atom14'] = bb_representations[-1].to(rigids_pred.device)
        return pred_out
