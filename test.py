import os
import argparse

import torch
from torch.utils.data import DataLoader
from datf.models import build_model
from datf.datasets import build_dataset
from configs.config_args import parse_train_configs
from datf.utils.tester import ModelTester
from datf.optimizers import build_optimizer
from datf.losses import build_criterion

def test(args):
    val_dataset = True
    device = args.device
    scene_channels = 5 if args.map_version == '2.1' else 3
    nfuture = int(3 * args.sampling_rate)

    model = build_model(cfg=cfg)
    if hasattr(cfg, "test_ckpt") and cfg.test_ckpt and len(cfg.test_ckpt):
        model.load_params_from_file(filename=cfg.test_ckpt)
        print("[LOG] Loaded checkpoint")
    model.cuda()

    ploss_type = None 
    if hasattr(cfg, "ploss"):
        ploss_criterion = build_criterion(cfg=cfg)
        ploss_criterion = ploss_criterion.to(device) if ploss_criterion else None

    # Send model to Device:
    model = model.to(device)
    test_dataset, _, collate_fn = build_dataset(cfg=cfg, train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                collate_fn=lambda x: collate_fn(x, test_set= args.test_set if hasattr(args, "test_set") else False), num_workers=args.num_workers)
    print(f'Test Examples: {len(test_dataset)} ')

    tester = ModelTester( model, test_loader, exp_path = args.exp_path, \
        cfg=cfg, device=device, ploss_criterion=ploss_criterion)
    tester.run()

if __name__ == "__main__":
    cfg = parse_train_configs()
    test(cfg)
