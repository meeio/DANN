import logging

import numpy as np
import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.utils as utils


def anpai(tom, use_gpu, need_logging=True):
    """decide Cpu Or Gpu (cog) for list of tensor and module
    
    Arguments:
        tom {list} -- list of tensor and module
        params {bool} -- indicate use gpu or not.

    Returns:
        list -- some as `tom` but in different divece.
    """
    # init a list to store result
    handle = list()
    
    if not isinstance(tom, (list, tuple)):
        l = list()
        l.append(tom)
        tom = l
    
    def __handle_module(module):
        # use data parallel
        dpm =  nn.DataParallel(module)
        
        handle.append(dpm)
        # info gpu used
        name = module.__class__.__name__
        info = "A >%s< object sent to Multi-GPU wits ids:" % name
        for i in dpm.device_ids:
            info += str(i)
        info += '.'
        if need_logging:
            logging.info(info)

    def __handle_tensor(tensor, d):
        handle.append(tensor.to(d))
        name = tensor.__class__.__name__
        if need_logging :
            if str(d.type) == 'cpu':
                logging.info("Has no gpu or not use, %s will sent to CPU." % name)
            else:
                logging.info("A >%s< object sent to GPU." % name)

    device = torch.device("cpu")
    # When use_gpu and has gpu to use
    if use_gpu and cuda.is_available():

        # init a default device when has gpu
        device = torch.device("cuda:0")
 
        for i in tom:
            i.cuda()
            # When usable gpu more then one
            if cuda.device_count() > 1:
                # for Module object use parallel
                if isinstance(i, nn.Module):
                    __handle_module(i)
                # for tnesor object just send to default gpu
                elif isinstance(i, torch.Tensor):
                    __handle_tensor(i,device)
                else:
                    if need_logging:
                        logging.error(i.__class__.__name__ + 'not spuuort')

            # When only one gpu can be used
            else:
                __handle_tensor(i,device)
    # use CPU
    else:
        for i in tom:
            target = i
            if isinstance(i, nn.DataParallel):
                for j in i.children():
                    target = j
                    __handle_tensor(target,device)
            else:
                __handle_tensor(target, device)
    

    return handle[0] if len(handle) == 1 else handle