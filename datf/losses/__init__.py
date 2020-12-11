
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import Log_determinant, Interpolated_Ploss, MSE_Ploss 

__all__ = {
    "MSE": MSE_Ploss,
    "mseloss": MSE_Ploss,
    "Log_determinant": Log_determinant,
    "Interpolated_Ploss": Interpolated_Ploss,
}


def build_criterion( **kwargs):
    cfg=kwargs.get('cfg')
    if hasattr(cfg, "ploss_criterion") and cfg.ploss_criterion:
        loss_fn = __all__[cfg.ploss_criterion]
    else:
        loss_fn = None
    return loss_fn 