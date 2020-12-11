import torch
import torch.nn as nn
import torch.nn.functional as F
from .argoverse.argoverse import ArgoverseDataset, argoverse_collate
from .argoverse.argoverse_r2p2 import ArgoverseDataset_R2P2, argoverse_collate_R2P2
from .nuscenes.nuscenes import NuscenesDataset, nuscenes_collate
from .nuscenes.nuscenes_r2p2 import NuscenesDataset_R2P2, nuscenes_collate_R2P2
from .carla.carla import CarlaDataset, carla_collate
import os

__all__ = {
    "argoverse": ArgoverseDataset,
    "argoverse_r2p2": ArgoverseDataset_R2P2,
    "nuscenes": NuscenesDataset,
    "nuscenes_r2p2": NuscenesDataset_R2P2,
    "carla": CarlaDataset
}

__allcollate__ = {
    "argoverse": argoverse_collate,
    "argoverse_r2p2": argoverse_collate_R2P2,
    "nuscenes": nuscenes_collate,
    "nuscenes_r2p2": nuscenes_collate_R2P2,
    "carla": carla_collate
}

def build_dataset( **kwargs):
    cfg=kwargs.get('cfg')
    train=kwargs.get('train', cfg.train if hasattr(cfg, "train") else False )
    if hasattr(cfg, "test_set"):
        if cfg.test_set:
            print("[LOG] Working on given test set (Test dataset)")
            dataset = __all__[cfg.dataset](cfg, 'test', map_version=cfg.map_version, sampling_rate=cfg.sampling_rate)
        else:
            print("[LOG] Working on given test/val set")
            dataset = __all__[cfg.dataset](cfg, 'val', map_version=cfg.map_version, sampling_rate=cfg.sampling_rate)
        collate_fn = __allcollate__[cfg.dataset]
        return dataset, None , collate_fn 

    if train:
        print("[LOG] Working on train and val set")
        dataset = __all__[cfg.dataset](cfg, 'train', map_version=cfg.map_version, sampling_rate=cfg.sampling_rate)
        val_dataset = __all__[cfg.dataset](cfg, 'val', map_version=cfg.map_version, sampling_rate=cfg.sampling_rate)
        collate_fn = __allcollate__[cfg.dataset]
    else:
        print("[LOG] Wokring on test and val set")
        val_dataset = __all__[cfg.dataset](cfg, 'val', map_version=cfg.map_version, sampling_rate=cfg.sampling_rate)
        dataset = __all__[cfg.dataset](cfg, 'test', map_version=cfg.map_version, sampling_rate=cfg.sampling_rate)
        collate_fn = __allcollate__[cfg.dataset]
    return dataset, val_dataset, collate_fn 