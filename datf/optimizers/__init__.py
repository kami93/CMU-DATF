import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .torch_optims import adam, sgd

__all__ = {
    "adam": adam,
    "sgd": sgd
}

def build_optimizer( **kwargs):
    
    cfg=kwargs.get('cfg')
    model = kwargs.get('model')
    if isinstance(cfg.optimizer_name, list):
        if isinstance(model, list):
            optim_list=[]
            for i, optim in enumerate(cfg.optimizer_name):
                optimizer = __all__[optim](model_instance=model[i], **cfg)
                optim_list.append(optimizer)
            optimizer = optim_list
        else:
            optim_list=[]
            for optim in cfg.optimizer_name:
                optimizer = __all__[optim](model_instance=model, **cfg)
                optim_list.append(optimizer)
            optimizer = optim_list
    else:
        try:
            optimizer = __all__[cfg.optimizer_name](model_instance=model, **cfg)
        except Exception as e:
            print("[ERROR] Check if its a optimizer list/ Model list or not")
    return optimizer