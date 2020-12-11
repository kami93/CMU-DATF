import os
import argparse

import torch
from torch.utils.data import DataLoader
from datf.models import build_model
from datf.datasets import build_dataset
from configs.config_args import parse_train_configs
from datf.utils.trainer import ModelTrainer
from datf.optimizers import build_optimizer
from datf.losses import build_criterion

def train(args):
    val_dataset = True
    device = args.device
    
    model = build_model(cfg=cfg)
    if hasattr(cfg, "ckpt") and len(cfg.ckpt):
        model.load_params_from_file(filename=cfg.ckpt)
        print("[LOG] Loaded checkpoint")
        
    if isinstance(model, list):
        for m in model:
            m = m.to(device)

    ploss_criterion = None
    if hasattr(cfg, "ploss_criterion"):
        ploss_criterion = build_criterion(cfg=cfg)

    # Send model to Device:
    train_dataset, val_dataset, collate_fn = build_dataset(cfg=cfg)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                collate_fn=lambda x: collate_fn(x), num_workers=args.num_workers)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,	
                                collate_fn=lambda x: collate_fn(x), num_workers=1)	
    print(f'Train Examples: {len(train_dataset)} | Valid Examples: {len(val_dataset) if val_dataset else "None" }')

    optimizer_list = build_optimizer(cfg=cfg, model=model.model)
    trainer = ModelTrainer( model, train_loader, valid_loader, optimizer_list, exp_path = args.exp_path, \
        cfg=cfg, device=device, ploss_criterion=ploss_criterion)
    trainer.train(cfg.num_epochs)



if __name__ == "__main__":
    cfg = parse_train_configs()
    train(cfg)
