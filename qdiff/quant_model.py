import logging
import torch.nn as nn
from model.ipa_pytorch import TorsionAngles
from qdiff.quant_block import get_specials, BaseQuantBlock
from qdiff.quant_block import QuantBasicTransformerBlock, QuantResBlock
from qdiff.quant_block import QuantQKMatMul, QuantSMVMatMul, QuantBasicTransformerBlock, QuantAttnBlock
from qdiff.quant_layer import QuantModule, StraightThrough
from ldm.modules.attention import BasicTransformerBlock
from qdiff.quant_block import QuantEmbedder, QuantIPA, QuantTransformerEncoderLayer, QuantStructureModuleTransition, QuantEdgeTransition, QuantTorsionAngles

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
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        self.quant_block_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantModule
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            # print(isinstance(child_module, StraightThrough) or name is 'linear_rbf')
            if isinstance(module, TorsionAngles) and name is 'linear_3':
                continue
            elif isinstance(child_module, StraightThrough) or name is 'linear_rbf':
                continue
            elif isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)) and name is not 'linear_rbf': # nn.Conv1d
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, act_quant_params, self.logger))
                prev_quantmodule = getattr(module, name)
            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def quant_block_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        block_names = [QuantBasicTransformerBlock, QuantAttnBlock, ]
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

