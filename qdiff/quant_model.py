import logging
import torch.nn as nn
import copy
from model.quant_ipa_pytorch import TorsionAngles, EdgeTransition, StructureModuleTransition
from model.quant_score_network import Embedder
from qdiff.quant_block import get_specials, BaseQuantBlock
from qdiff.quant_block import QuantBasicTransformerBlock, QuantResBlock
from qdiff.quant_block import QuantQKMatMul, QuantSMVMatMul, QuantBasicTransformerBlock, QuantAttnBlock
from qdiff.quant_layer import QuantModule, StraightThrough
from ldm.modules.attention import BasicTransformerBlock
from qdiff.quant_block import QuantEmbedder, QuantIPA, QuantTransformerEncoderLayer, QuantStructureModuleTransition, QuantEdgeTransition, QuantTorsionAngles
from torch.nn import TransformerEncoderLayer
# logger = logging.getLogger(__name__)


class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, logger=logging.getLogger(__name__), **kwargs, ):
        super().__init__()
        self.model = model
        self.sm_abit = kwargs.get('sm_abit', 8)
        # self.in_channels = model.in_channels
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        self.specials = get_specials(act_quant_params['leaf_param'])
        self.logger = logger
        # self.relu_modules_dict = {}
        # self.relu_modules = [Embedder, StructureModuleTransition, EdgeTransition, TorsionAngles]
        # self.relu_modules_dict['node_embedder'] = ['2', '4']
        # self.relu_modules_dict['edge_embedder'] = ['2', '4']
        # self.relu_modules_dict['torsion_pred'] = ['linear_2']
        # for j in range(4):
        #     self.relu_modules_dict[f'node_transition_{j}'] = ['linear_2', 'linear_3']
        #     self.relu_modules_dict[f'edge_transition_{j}'] = ['2', 'final_layer']
            
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        self.quant_block_refactor(self.model, weight_quant_params, act_quant_params)
        # self.relu_modules = ['node_embedder', 'edge_embedder', 'node_transition_0', 'edge_transition_0', 
        #                      'node_transition_1', 'edge_transition_1', 'node_transition_2', 'edge_transition_2',
        #                      'node_transition_3', 'torsion_pred',]
        # exit(0)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, prev_name = None):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantModule
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            # if isinstance(child_module, nn.ReLU):
            #     print(module, child_module)
            # print(name)
            # print(name, type(module) in self.relu_modules)
            # print('wwwwww', name, type(module))
            # if isinstance(module, Embedder):
            #     print('xxxxxxx embedder')
            if isinstance(module, TorsionAngles) and name == 'linear_3':
                continue
            elif isinstance(child_module, StraightThrough) or name == 'linear_rbf':
                continue
            elif isinstance(module, TorsionAngles) and name == 'linear_2':
                # print(name)
                tmp_act_quant_params = copy.deepcopy(act_quant_params)
                tmp_act_quant_params['symmetric'] = False
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, tmp_act_quant_params, self.logger))
            
            elif isinstance(module, StructureModuleTransition) and (name == 'linear_2' or name == 'linear_3'):
                # print(name)
                tmp_act_quant_params = copy.deepcopy(act_quant_params)
                tmp_act_quant_params['symmetric'] = False
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, tmp_act_quant_params, self.logger))
            
            elif (prev_name == 'node_embedder' or prev_name == 'edge_embedder') and (name == '2' or name == '4'):
                # print(name)
                tmp_act_quant_params = copy.deepcopy(act_quant_params)
                tmp_act_quant_params['symmetric'] = False
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, tmp_act_quant_params, self.logger))
            
            elif prev_name == 'trunk' and name == '2':      # esge_transition trunk
                # print(name)
                tmp_act_quant_params = copy.deepcopy(act_quant_params)
                tmp_act_quant_params['symmetric'] = False
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, tmp_act_quant_params, self.logger))
            
            elif isinstance(module, EdgeTransition) and name == 'final_layer':
                # print(name)
                tmp_act_quant_params = copy.deepcopy(act_quant_params)
                tmp_act_quant_params['symmetric'] = False
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, tmp_act_quant_params, self.logger))
            
            elif isinstance(module, TransformerEncoderLayer) and name == 'linear2':
                # print('.......', name)
                tmp_act_quant_params = copy.deepcopy(act_quant_params)
                tmp_act_quant_params['symmetric'] = False
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, tmp_act_quant_params, self.logger))
                
            elif isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)) and name != 'linear_rbf': # nn.Conv1d
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, act_quant_params, self.logger))
                prev_quantmodule = getattr(module, name)
                # print(name, prev_name)
                # print(module)
                # print('----', child_module)
                # print('====', prev_quantmodule)
            else:
                # print('-----refactor')
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params, name)

    def quant_block_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        block_names = [QuantBasicTransformerBlock, QuantAttnBlock, ]
        # se3 block
        block_names2 = [QuantEmbedder, QuantIPA, QuantTransformerEncoderLayer, QuantStructureModuleTransition, QuantEdgeTransition, QuantTorsionAngles]
        # , QuantIpaScore, QuantMultiheadAttention, QuantEmbedder, QuantIPA, QuantTransformerEncoder, QuantStructureModuleTransition, QuantEdgeTransition, QuantTorsionAngles
        for name, child_module in module.named_children():
            # print(child_module, type(child_module))
            if type(child_module) in self.specials:
                if self.specials[type(child_module)] in block_names:
                    setattr(module, name, self.specials[type(child_module)](child_module,
                        act_quant_params, sm_abit=self.sm_abit))
                # elif self.specials[type(child_module)] == QuantSMVMatMul:
                #     setattr(module, name, self.specials[type(child_module)](
                #         act_quant_params, sm_abit=self.sm_abit))
                # elif self.specials[type(child_module)] == QuantQKMatMul:
                #     setattr(module, name, self.specials[type(child_module)](
                #         act_quant_params))
                elif self.specials[type(child_module)] in block_names2:
                    setattr(module, name, self.specials[type(child_module)](child_module,
                        act_quant_params, weight_quant_params, sm_abit=self.sm_abit))
                else:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params, sm_abit=self.sm_abit))
            else:
                self.quant_block_refactor(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, x):
        return self.model(x)    # , timesteps, context)
    
    
    # TODO
    def set_running_stat(self, running_stat: bool, sm_only=False):
        for m in self.model.modules():
            if isinstance(m, QuantBasicTransformerBlock):
                if sm_only:
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
                else:
                    m.attn1.act_quantizer_q.running_stat = running_stat
                    m.attn1.act_quantizer_k.running_stat = running_stat
                    m.attn1.act_quantizer_v.running_stat = running_stat
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_q.running_stat = running_stat
                    m.attn2.act_quantizer_k.running_stat = running_stat
                    m.attn2.act_quantizer_v.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
            if isinstance(m, QuantModule) and not sm_only:
                m.set_running_stat(running_stat)

    # TODO
    def set_grad_ckpt(self, grad_ckpt: bool):
        for name, m in self.model.named_modules():
            if isinstance(m, (QuantBasicTransformerBlock, BasicTransformerBlock)):
                # logger.info(name)
                m.checkpoint = grad_ckpt
            # elif isinstance(m, QuantResBlock):
                # logger.info(name)
                # m.use_checkpoint = grad_ckpt

