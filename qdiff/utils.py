import logging
from typing import Union
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from qdiff.quant_layer import QuantModule, UniformAffineQuantizer
from qdiff.quant_block import BaseQuantBlock
from qdiff.quant_model import QuantModel
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
from openfold.utils.rigid_utils import Rigid, Rotation

# logger = logging.getLogger(__name__)


def save_inp_oup_data(cali_data, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock], #: torch.Tensor,
                      asym: bool = False, act_quant: bool = False, batch_size: int = 32, keep_gpu: bool = False, #True,
                      cond: bool = False, is_sm: bool = False, logger = logging.getLogger(__name__)):
    """
    Save input data and output data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param asym: if Ture, save quantized input and full precision output
    :param act_quant: use activation quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :param cond: conditional generation or not
    :param is_sm: avoid OOM when caching n^2 attention matrix when n is large
    :return: input and output data
    """
    device = next(model.parameters()).device
    get_inp_out = GetLayerInpOut(model, layer, device=device, asym=asym, act_quant=act_quant)
    cached_batches = []
    cached_inps, cached_outs = None, None
    torch.cuda.empty_cache()


    # if not cond:
    #     cali_xs, cali_ts = cali_data
    # else:
    #     cali_xs, cali_ts, cali_conds = cali_data

    # print(cali_xs.shape, cali_ts.shape) # torch.Size([5120, 3, 32, 32]) torch.Size([5120])
    
    # if is_sm:
    #     logger.info("Checking if attention is too large...")
    #     if not cond:
    #         test_inp, test_out = get_inp_out(
    #             cali_xs[:1].to(device), 
    #             cali_ts[:1].to(device)
    #         )
    #     else:
    #         test_inp, test_out = get_inp_out(
    #             cali_xs[:1].to(device), 
    #             cali_ts[:1].to(device),
    #             cali_conds[:1].to(device)
    #         )
            
    #     is_sm = False
    #     if (isinstance(test_inp, tuple) and test_inp[0].shape[1] == test_inp[0].shape[2]):
    #         logger.info(f"test_inp shape: {test_inp[0].shape}, {test_inp[1].shape}")
    #         if test_inp[0].shape[1] == 4096:
    #             is_sm = True
    #     if test_out.shape[1] == test_out.shape[2]:
    #         logger.info(f"test_out shape: {test_out.shape}")
    #         if test_out.shape[1] == 4096:
    #             is_sm = True
            
    #     if is_sm:
    #         logger.info("Confirmed. Trading speed for memory when caching attn matrix calibration data")
    #         inds = np.random.choice(cali_xs.size(0), cali_xs.size(0) // 2, replace=False)
    #     else:
    #         logger.info("Nope. Using normal caching method")
    
    
    num = int(cali_data['seq_idx'].size(0) / batch_size)
    # if is_sm:
    #     num //= 2
    l_in_0, l_in_1, l_in, l_out = 0, 0, 0, 0
     
    for i in trange(num):
        # if not cond:
        batch_data = {}
        for key in cali_data.keys():
            batch_data[key] = cali_data[key][i * batch_size:(i + 1) * batch_size].to(device)
        
        # print('batch_data:', batch_data)
        node_mask = batch_data['res_mask'].type(torch.float32)
        src_key_padding_mask = 1 - node_mask
        
        cur_inp, cur_out = get_inp_out(batch_data) 
        # print('cur_inp:', len(cur_inp))    
        # print('cur_inp: ', cur_inp[0][1], cur_inp[1][1], cur_inp[3][1])
        # print('cur_out: ', cur_out[1])      # nan
        # print(cur_out)
        # exit(0)
        # cached_inps, cached_outs = get_inp_out(batch_data) 
        
        # for idx in range(len(cur_inp)):
        #     print('self, in ', idx, 'shape: ', cur_inp[idx].shape)
        # print('self, out shape: ', cur_out.shape)
        
        #) #if not is_sm else get_inp_out(
            #     cali_xs[inds[i * batch_size:(i + 1) * batch_size]].to(device), 
            #     cali_ts[inds[i * batch_size:(i + 1) * batch_size]].to(device)
            # )
            # print(layer)
            # print(cur_inp.shape, cur_out.shape) # torch.Size([8, 128]) torch.Size([8, 512])
            # exit(0)
        # else:
        #     cur_inp, cur_out = get_inp_out(
        #         cali_xs[i * batch_size:(i + 1) * batch_size].to(device), 
        #         cali_ts[i * batch_size:(i + 1) * batch_size].to(device),
        #         cali_conds[i * batch_size:(i + 1) * batch_size].to(device)
        #     ) if not is_sm else get_inp_out(
        #         cali_xs[inds[i * batch_size:(i + 1) * batch_size]].to(device), 
        #         cali_ts[inds[i * batch_size:(i + 1) * batch_size]].to(device),
        #         cali_conds[inds[i * batch_size:(i + 1) * batch_size]].to(device)
        #     )

        # cached_batches is not used, copy into cpu
        if isinstance(layer, BaseQuantBlock) and layer.name == 'quantipa':
            cur_node, cur_edge, cur_rigids, cur_mask = cur_inp
            # print(len(cur_node), len(cur_edge), cur_rigids, len(cur_mask))
            # exit(0)
            if not is_sm:
                # cached_batches.append(((cur_node, cur_edge, cur_rigids, cur_mask), cur_out))
                cached_batches.append(((cur_node.cpu(), cur_edge.cpu(),
                                        # Rigid(Rotation(cur_rigids._rots._quats.cpu()), 
                                        Rigid(cur_rigids._rots.to(device='cpu', dtype=torch.float32), cur_rigids._trans.cpu()), 
                                        cur_mask.cpu()), cur_out.cpu()))
        elif isinstance(layer, BaseQuantBlock) and layer.name == 'quanttel':    # transformerencoderlayer
            cached_batches.append(((cur_inp.cpu(), src_key_padding_mask.cpu()), cur_out.cpu()))       # src_key_padding_mask=1 - node_mask
        elif isinstance(layer, BaseQuantBlock) and layer.name == 'quantet':
            node, edge = cur_inp
            cached_batches.append(((node.cpu(), edge.cpu()), cur_out.cpu()))
        elif isinstance(layer, BaseQuantBlock) and layer.name == 'quantta':
            unnor, nor = cur_out
            cached_batches.append((cur_inp.cpu(), (unnor.cpu(), nor.cpu())))
        elif isinstance(layer, BaseQuantBlock) and layer.name == 'quantemb':
            # cur_node, cur_edge, cur_rigids, cur_mask = cur_inp
            # exit(0)
            # cur_node, cur_edge, cur_rigids, cur_mask = cur_inp
            orig = (batch_data['seq_idx'].detach().cpu(), batch_data['t'].detach().cpu(), batch_data['fixed_mask'].type(torch.float32).detach().cpu(), batch_data['sc_ca_t'].detach().cpu(), )
            node, edge = cur_out
            cached_batches.append((orig, (node.cpu(), edge.cpu())))
            # exit(0)
        else:   # transformerencoderlayer
            if not is_sm:
                cached_batches.append((cur_inp.cpu(), cur_out.cpu()))
                # cached_batches.append((cur_inp, cur_out))
            else:
                if cached_inps is None:
                    l_in = cur_inp.shape[0] * num
                    cached_inps = torch.zeros(l_in, *cur_inp.shape[1:])
                cached_inps.index_copy_(0, torch.arange(i * cur_inp.shape[0], (i + 1) * cur_inp.shape[0]), cur_inp.cpu())
        
        
        # if is_sm:
        #     if cached_outs is None:
        #         l_out = cur_out.shape[0] * num
        #         cached_outs = torch.zeros(l_out, *cur_out.shape[1:])
        #     cached_outs.index_copy_(0, torch.arange(i * cur_out.shape[0], (i + 1) * cur_out.shape[0]), cur_out.cpu())

    if not is_sm:
        if isinstance(cached_batches[0][0], tuple) and layer.name == 'quantipa':
            rigids = []
            for tmp in cached_batches:
                rigids.append(tmp[0][2])
            
            cached_inps = [
                torch.cat([x[0][0] for x in cached_batches]), 
                torch.cat([x[0][1] for x in cached_batches]),
                rigids,        # Rigid
                torch.cat([x[0][3] for x in cached_batches]),
            ]
            cached_outs = torch.cat([x[1] for x in cached_batches])
        elif isinstance(cached_batches[0][0], tuple) and (layer.name == 'quanttel' or layer.name == 'quantet'):
            cached_inps = [
                torch.cat([x[0][0] for x in cached_batches]), 
                torch.cat([x[0][1] for x in cached_batches]),
            ]
            cached_outs = torch.cat([x[1] for x in cached_batches])
        elif isinstance(cached_batches[0][0], tuple) and layer.name == 'quantemb':
            # print('here')
            cached_inps = [
                torch.cat([x[0][0] for x in cached_batches]), 
                torch.cat([x[0][1] for x in cached_batches]),
                torch.cat([x[0][2] for x in cached_batches]),
                torch.cat([x[0][3] for x in cached_batches]),
            ]
            cached_outs = [
                torch.cat([x[1][0] for x in cached_batches]), 
                torch.cat([x[1][1] for x in cached_batches]),
            ]
        elif isinstance(cached_batches[0][1], tuple) and layer.name == 'quantta':
            cached_inps = torch.cat([x[0] for x in cached_batches])
            cached_outs = [
                torch.cat([x[1][0] for x in cached_batches]), 
                torch.cat([x[1][1] for x in cached_batches]),
            ]
        else:
            cached_inps = torch.cat([x[0] for x in cached_batches])
            cached_outs = torch.cat([x[1] for x in cached_batches])
    
    if isinstance(cached_inps, list):
        for idx in range(len(cached_inps)):
            try:
                logger.info(f"in {idx} shape: {cached_inps[idx].shape}")
            except:
                logger.info(f'in {idx}')
                continue
    else:
        logger.info(f"in shape: {cached_inps.shape}")
    
    if isinstance(cached_outs, list):
        for idx in range(len(cached_outs)):
            try:
                logger.info(f"out {idx} shape: {cached_outs[idx].shape}")
            except:
                logger.info(f'out {idx}')
                continue
    else:
        logger.info(f"out shape: {cached_outs.shape}")

    
    torch.cuda.empty_cache()
    if keep_gpu:
        if isinstance(cached_inps, list):
            for idx in range(len(cached_inps)):
                cached_inps[idx] = cached_inps[idx].to(device)
        else:
            cached_inps = cached_inps.to(device)
        cached_outs = cached_outs.to(device)
    # print(cached_inps.shape, cached_outs.shape) # torch.Size([5120, 128]) torch.Size([5120, 512])
    # exit(0)
    return cached_inps, cached_outs


def save_grad_data(model: QuantModel, layer: Union[QuantModule, BaseQuantBlock], cali_data: torch.Tensor,
                   damping: float = 1., act_quant: bool = False, batch_size: int = 32,
                   keep_gpu: bool = True):
    """
    Save gradient data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param damping: damping the second-order gradient by adding some constant in the FIM diagonal
    :param act_quant: use activation quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: gradient data
    """
    device = next(model.parameters()).device
    get_grad = GetLayerGrad(model, layer, device, act_quant=act_quant)
    cached_batches = []
    torch.cuda.empty_cache()

    for i in range(int(cali_data.size(0) / batch_size)):
        cur_grad = get_grad(cali_data[i * batch_size:(i + 1) * batch_size])
        cached_batches.append(cur_grad.cpu())

    cached_grads = torch.cat([x for x in cached_batches])
    cached_grads = cached_grads.abs() + 1.0
    # scaling to make sure its mean is 1
    # cached_grads = cached_grads * torch.sqrt(cached_grads.numel() / cached_grads.pow(2).sum())
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_grads = cached_grads.to(device)
    return cached_grads


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class GetLayerInpOut:
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, asym: bool = False, act_quant: bool = False):
        self.model = model
        self.layer = layer
        self.asym = asym
        self.device = device
        self.act_quant = act_quant
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)

    def __call__(self, input_feats):
        self.model.eval()
        self.model.set_quant_state(False, False)

        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            try:
                _ = self.model(input_feats)
            except StopForwardException:
                pass

            if self.asym:
                # Recalculate input with network quantized
                self.data_saver.store_output = False
                self.model.set_quant_state(weight_quant=True, act_quant=self.act_quant)
                try:
                    _ = self.model(input_feats)
                except StopForwardException:
                    pass
                self.data_saver.store_output = True

        handle.remove()

        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()

        # if len(self.data_saver.input_store) > 1 and torch.is_tensor(self.data_saver.input_store[1]):
        #     return (self.data_saver.input_store[0].detach(),  
        #         self.data_saver.input_store[1].detach()), self.data_saver.output_store.detach()
        # else:
        #     return self.data_saver.input_store[0].detach(), self.data_saver.output_store.detach()
        # print(self.data_saver.input_store[2]._rots.shape) # torch.Size([8, 485])
        # exit(0)
        # if self.layer.name == 'quanttel':
        #     print(f'getinpout: {len(self.data_saver.input_store)}, shape: {self.data_saver.input_store[0].shape}')

        if isinstance(self.layer, BaseQuantBlock) and len(self.data_saver.input_store) > 1 and self.layer.name == 'quantipa':
            return (self.data_saver.input_store[0].detach(),  
                self.data_saver.input_store[1].detach(), self.data_saver.input_store[2].stop_rot_gradient(), #.detach(),  stop gradient
                self.data_saver.input_store[3].detach()), self.data_saver.output_store.detach()
        
        elif isinstance(self.layer, BaseQuantBlock) and len(self.data_saver.input_store) > 1 and self.layer.name == 'quantet':
            return (self.data_saver.input_store[0].detach(), self.data_saver.input_store[1].detach(), ), self.data_saver.output_store.detach()
        # if len(self.data_saver.input_store) > 1 and self.layer.name == 'quanttel':
        #     return (self.data_saver.input_store[0].detach(), self.data_saver.input_store[1].detach(), 
        #         self.data_saver.input_store[2].detach()), self.data_saver.output_store.detach()
        elif isinstance(self.layer, BaseQuantBlock) and len(self.data_saver.output_store) > 1 and self.layer.name == 'quantemb':
            return None, (self.data_saver.output_store[0].detach(),self.data_saver.output_store[1].detach())
        elif isinstance(self.layer, BaseQuantBlock) and len(self.data_saver.output_store) > 1 and self.layer.name == 'quantta':
            return self.data_saver.input_store[0].detach(), (self.data_saver.output_store[0].detach(),self.data_saver.output_store[1].detach())
        else:
            return self.data_saver.input_store[0].detach(), self.data_saver.output_store.detach()


class GradSaverHook:
    def __init__(self, store_grad=True):
        self.store_grad = store_grad
        self.stop_backward = False
        self.grad_out = None

    def __call__(self, module, grad_input, grad_output):
        if self.store_grad:
            self.grad_out = grad_output[0]
        if self.stop_backward:
            raise StopForwardException


class GetLayerGrad:
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, act_quant: bool = False):
        self.model = model
        self.layer = layer
        self.device = device
        self.act_quant = act_quant
        self.data_saver = GradSaverHook(True)

    def __call__(self, model_input):
        """
        Compute the gradients of block output, note that we compute the
        gradient by calculating the KL loss between fp model and quant model

        :param model_input: calibration data samples
        :return: gradients
        """
        self.model.eval()

        handle = self.layer.register_backward_hook(self.data_saver)
        with torch.enable_grad():
            try:
                self.model.zero_grad()
                inputs = model_input.to(self.device)
                self.model.set_quant_state(False, False)
                out_fp = self.model(inputs)
                quantize_model_till(self.model, self.layer, self.act_quant)
                out_q = self.model(inputs)
                loss = F.kl_div(F.log_softmax(out_q, dim=1), F.softmax(out_fp, dim=1), reduction='batchmean')
                loss.backward()
            except StopForwardException:
                pass

        handle.remove()
        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()
        return self.data_saver.grad_out.data


def quantize_model_till(model: QuantModule, layer: Union[QuantModule, BaseQuantBlock], act_quant: bool = False):
    """
    We assumes modules are correctly ordered, holds for all models considered
    :param model: quantized_model
    :param layer: a block or a single layer.
    """
    model.set_quant_state(False, False)
    for name, module in model.named_modules():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.set_quant_state(True, act_quant)
        if module == layer:
            break


def get_train_samples(args, sample_data, custom_steps=None):
    print(sample_data.keys())
    num_samples, num_st = args.cali_n, args.cali_st
    custom_steps = args.custom_steps if custom_steps is None else custom_steps
    if num_st == 1:
        xs = sample_data[:num_samples]
        ts = (torch.ones(num_samples) * 800)
    else:
        # get the real number of timesteps (especially for DDIM)
        nsteps = len(sample_data["ts"])
        assert(nsteps >= custom_steps)
        timesteps = list(range(0, nsteps, nsteps//num_st))
        logger.info(f'Selected {len(timesteps)} steps from {nsteps} sampling steps')
        xs_lst = [sample_data["xs"][i][:num_samples] for i in timesteps]
        ts_lst = [sample_data["ts"][i][:num_samples] for i in timesteps]
        if args.cond:
            xs_lst += xs_lst
            ts_lst += ts_lst
            conds_lst = [sample_data["cs"][i][:num_samples] for i in timesteps] + [sample_data["ucs"][i][:num_samples] for i in timesteps]
        xs = torch.cat(xs_lst, dim=0)
        ts = torch.cat(ts_lst, dim=0)
        if args.cond:
            conds = torch.cat(conds_lst, dim=0)
            return xs, ts, conds
    return xs, ts


def convert_adaround(model):
    for name, module in model.named_children():
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                # logger.info('Ignore reconstruction of layer {}'.format(name))
                continue
            else:
                # logger.info('Change layer {} to adaround'.format(name))
                module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode='learned_hard_sigmoid',
                                                   weight_tensor=module.org_weight.data)
        elif isinstance(module, BaseQuantBlock):
            if module.ignore_reconstruction is True:
                # logger.info('Ignore reconstruction of block {}'.format(name))
                continue
            else:
                # logger.info('Change block {} to adaround'.format(name))
                for name, sub_module in module.named_modules():
                    if isinstance(sub_module, QuantModule):
                        if sub_module.split != 0:
                            # print(f"split {name}")
                            if len(sub_module.split) == 6:
                                sub_module.weight_quantizer = AdaRoundQuantizer(uaq=sub_module.weight_quantizer, round_mode='learned_hard_sigmoid',
                                                                        weight_tensor=sub_module.org_weight.data[:, :sub_module.split[0], ...])
                                sub_module.weight_quantizer_0 = AdaRoundQuantizer(uaq=sub_module.weight_quantizer_0, round_mode='learned_hard_sigmoid',
                                                                        weight_tensor=sub_module.org_weight.data[:, sub_module.split[0]:sub_module.split[0]+sub_module.split[1], ...])
                                sub_module.weight_quantizer_1 = AdaRoundQuantizer(uaq=sub_module.weight_quantizer_1, round_mode='learned_hard_sigmoid',
                                                                        weight_tensor=sub_module.org_weight.data[:, sub_module.split[0]+sub_module.split[1]:sub_module.split[0]+sub_module.split[1]+sub_module.split[2], ...])
                                sub_module.weight_quantizer_2 = AdaRoundQuantizer(uaq=sub_module.weight_quantizer_2, round_mode='learned_hard_sigmoid',
                                                                            weight_tensor=sub_module.org_weight.data[:, sub_module.split[0]+sub_module.split[1]+sub_module.split[2]:sub_module.split[0]+sub_module.split[1]+sub_module.split[2]+sub_module.split[3], ...])
                                sub_module.weight_quantizer_3 = AdaRoundQuantizer(uaq=sub_module.weight_quantizer_3, round_mode='learned_hard_sigmoid',
                                                                            weight_tensor=sub_module.org_weight.data[:, sub_module.split[0]+sub_module.split[1]+sub_module.split[2]+sub_module.split[3]:sub_module.split[0]+sub_module.split[1]+sub_module.split[2]+sub_module.split[3]+sub_module.split[4], ...])
                                sub_module.weight_quantizer_4 = AdaRoundQuantizer(uaq=sub_module.weight_quantizer_4, round_mode='learned_hard_sigmoid',
                                                                            weight_tensor=sub_module.org_weight.data[:, sub_module.split[0]+sub_module.split[1]+sub_module.split[2]+sub_module.split[3]+sub_module.split[4]:, ...])
                            elif len(sub_module.split) == 3:
                                # print('===edge split =====')
                                sub_module.weight_quantizer = AdaRoundQuantizer(uaq=sub_module.weight_quantizer, round_mode='learned_hard_sigmoid',
                                                                            weight_tensor=sub_module.org_weight.data[:, :sub_module.split[0], ...])
                                sub_module.weight_quantizer_0 = AdaRoundQuantizer(uaq=sub_module.weight_quantizer_0, round_mode='learned_hard_sigmoid',
                                                                            weight_tensor=sub_module.org_weight.data[:, sub_module.split[0]:sub_module.split[0]+sub_module.split[1], ...])
                                sub_module.weight_quantizer_1 = AdaRoundQuantizer(uaq=sub_module.weight_quantizer_1, round_mode='learned_hard_sigmoid',
                                                                            weight_tensor=sub_module.org_weight.data[:, sub_module.split[0]+sub_module.split[1]:, ...])
                            elif len(sub_module.split) == 2:
                                # print('===node split =====')
                                sub_module.weight_quantizer = AdaRoundQuantizer(uaq=sub_module.weight_quantizer, round_mode='learned_hard_sigmoid',
                                                                            weight_tensor=sub_module.org_weight.data[:, :sub_module.split[0], ...])
                                sub_module.weight_quantizer_0 = AdaRoundQuantizer(uaq=sub_module.weight_quantizer_0, round_mode='learned_hard_sigmoid',
                                                                            weight_tensor=sub_module.org_weight.data[:, sub_module.split[0]:, ...])
                        else:
                            sub_module.weight_quantizer = AdaRoundQuantizer(uaq=sub_module.weight_quantizer, round_mode='learned_hard_sigmoid',
                                                                    weight_tensor=sub_module.org_weight.data)
        else:
            convert_adaround(module)


def resume_cali_model(qnn, ckpt_path, cali_data, quant_act=False, act_quant_mode='qdiff', cond=False):
    print("Loading quantized model checkpoint")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    print("Initializing weight quantization parameters")
    qnn.set_quant_state(True, False)
    _ = qnn(cali_data)
    
    # print('=========1==========', qnn.model)
    # if not cond:
    #     cali_xs, cali_ts = cali_data
    #     _ = qnn(cali_xs[:1].cuda(), cali_ts[:1].cuda())
    # else:
    #     cali_xs, cali_ts, cali_cs = cali_data
    #     _ = qnn(cali_xs[:1].cuda(), cali_ts[:1].cuda(), cali_cs[:1].cuda())
    
    # change weight quantizer from uniform to adaround
    convert_adaround(qnn)       # only weight quantizer are defined as AdaRoundQuantizer
    
    
    # print('=========2==========', qnn.model)
    for m in qnn.model.modules():   # only weight quantizer data are reinitialized
        if isinstance(m, AdaRoundQuantizer):
            m.zero_point = nn.Parameter(m.zero_point)
            m.delta = nn.Parameter(m.delta)
    
    # remove act_quantizer states for now
    keys = [key for key in ckpt.keys() if "act" in key]     #  delete all existing act quantizer
    # print('act keys: ', keys)       # blank
    for key in keys:
        del ckpt[key]
    qnn.load_state_dict(ckpt, strict=(act_quant_mode=='qdiff'))
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    
    # print('=========3==========', qnn.model)        
    for m in qnn.model.modules():                   # change trainable w quantizer to frozen/untrainable
        if isinstance(m, AdaRoundQuantizer):
            zero_data = m.zero_point.data
            delattr(m, "zero_point")
            m.zero_point = zero_data

            delta_data = m.delta.data
            delattr(m, "delta")
            m.delta = delta_data

    if quant_act:           
        print("Initializing act quantization parameters")
        qnn.set_quant_state(True, True)                    
        _ = qnn(cali_data)                           # initialize activation AdaRoundQuantizer
        # if not cond:
        #     _ = qnn(cali_xs[:1].cuda(), cali_ts[:1].cuda())
        # else:
        #     _ = qnn(cali_xs[:1].cuda(), cali_ts[:1].cuda(), cali_cs[:1].cuda())
        print("Loading quantized model checkpoint again")
        
        # print('=========4==========', qnn.model)
        for m in qnn.model.modules():                   # defined trainable parameter
            if isinstance(m, AdaRoundQuantizer):
                # print('========== here 1=============')     # no
                m.zero_point = nn.Parameter(m.zero_point)
                m.delta = nn.Parameter(m.delta)
            elif isinstance(m, UniformAffineQuantizer):
                if m.zero_point is not None:
                    if not torch.is_tensor(m.zero_point):
                        m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                    else:
                        m.zero_point = nn.Parameter(m.zero_point)
                    
        ckpt = torch.load(ckpt_path, map_location='cpu')
        qnn.load_state_dict(ckpt)
        qnn.set_quant_state(weight_quant=True, act_quant=True)
        
        # print('=========5==========', qnn.model)
        for m in qnn.model.modules():           # initialize activation quantizer data
            if isinstance(m, AdaRoundQuantizer):
                # print('========== here 2=============') no
                # print(m)  no
                zero_data = m.zero_point.data
                delattr(m, "zero_point")
                m.zero_point = zero_data

                delta_data = m.delta.data
                delattr(m, "delta")
                m.delta = delta_data
            elif isinstance(m, UniformAffineQuantizer):
                if m.zero_point is not None:
                    zero_data = m.zero_point.item()
                    delattr(m, "zero_point")
                    assert(int(zero_data) == zero_data)
                    m.zero_point = int(zero_data)
                    
                    