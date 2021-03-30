import os, sys
import argparse
import random
import logging
import datetime
import pathlib

import torch
import numpy as np

from torch.utils.data import DataLoader

def get_model(args):
    if args.model_type == 'CAM':
        from Proposed.models import CAM
        from Proposed.utils import ModelTrainer
        from common.model_utils import MSEloss

        model = CAM(motion_features=args.motion_features,
                    rnn_layers=args.rnn_layers,
                    rnn_dropout=args.rnn_dropout)
        context_map_size = None

    elif args.model_type == 'CAM_NFDecoder':	
        from Proposed.models import CAM_NFDecoder
        from Proposed.utils import ModelTrainer

        model = CAM_NFDecoder(velocity_const=args.velocity_const,
                              motion_features=args.motion_features,
                              rnn_layers=args.rnn_layers,
                              rnn_dropout=args.rnn_dropout,
                              detach_output=bool(args.detach_output))
        context_map_size = None

    elif args.model_type == 'Scene_CAM_NFDecoder':
        from Proposed.models import Scene_CAM_NFDecoder
        from Proposed.utils import ModelTrainer

        model = Scene_CAM_NFDecoder(scene_distance=args.scene_distance,
                                    velocity_const=args.velocity_const,
                                    motion_features=args.motion_features,
                                    rnn_layers=args.rnn_layers,
                                    rnn_dropout=args.rnn_dropout,
                                    detach_output=bool(args.detach_output))
        context_map_size = (64, 64)

    elif args.model_type == 'Global_Scene_CAM_NFDecoder':
        from Proposed.models import Global_Scene_CAM_NFDecoder
        from Proposed.utils import ModelTrainer

        model = Global_Scene_CAM_NFDecoder(scene_distance=args.scene_distance,
                                           velocity_const=args.velocity_const,
                                           motion_features=args.motion_features,
                                           rnn_layers=args.rnn_layers,
                                           rnn_dropout=args.rnn_dropout,
                                           detach_output=bool(args.detach_output))
        context_map_size = (64, 64)
    
    elif args.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
        from Proposed.models import AttGlobal_Scene_CAM_NFDecoder
        from Proposed.utils import ModelTrainer

        model = AttGlobal_Scene_CAM_NFDecoder(scene_distance=args.scene_distance,
                                              velocity_const=args.velocity_const,
                                              motion_features=args.motion_features,
                                              rnn_layers=args.rnn_layers,
                                              rnn_dropout=args.rnn_dropout,
                                              detach_output=bool(args.detach_output))
        context_map_size = (64, 64)

    else:
        raise ValueError("Unknown model type {:s}.".format(args.model_type))

    return model, context_map_size

def train(args):
    # Initialize logger.
    exp_path = pathlib.Path(args.exp_path).joinpath(args.tag + '_' + datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('_%d_%B__%H_%M_'))
    if not exp_path.is_dir():
        exp_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(exp_path.joinpath('training.log')))
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt='%Y-%m-%d %H:%M:%S'))
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("Experiment Path {:s}.".format(str(exp_path)))
    
    # Set random seeds.
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    logger.info("Set random seed {:d}.".format(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get Model
    model, context_map_size = get_model(args)
    logger.info("Loaded Model {:s}.".format(str(model)))
    logger.info("Context Map:  {:s}.".format("N/A" if context_map_size is None else str(context_map_size)))
    
    if "CAM" in args.model_type:
        if "NFDecoder" in args.model_type:
            if args.ploss_type == 'map':
                from common.model_utils import InterpolatedPloss
                ploss_criterion = InterpolatedPloss(scene_distance=args.scene_distance)
                prior_map_size = (100, 100)
            
            elif args.ploss_type == 'logistic':
                from common.model_utils import InterpolatedPloss
                ploss_criterion = InterpolatedPloss(scene_distance=args.scene_distance)
            
            else:
                from common.model_utils import MSEloss
                ploss_criterion = MSEloss()
        
        else:
            from common.model_utils import MSEloss
            ploss_criterion = MSEloss()

    logger.info("Prior Map: {:s}.".format("N/A" if prior_map_size is None else str(prior_map_size)))
    logger.info("Loss Type: {:s}.".format(str(ploss_criterion)))

    # Send model & ploss to device:
    model = model.to(device)
    ploss_criterion = ploss_criterion.to(device)

    if args.dataset == 'nuscenes':
        from dataset.nuscenes import NuscenesDataset, nuscenes_collate
        logger.info("Loading nuScenes dataset.")
        train_dataset = NuscenesDataset('./data/Preprocessed/nuScenes', args.train_partition, logger=logger, sampling_rate=args.sampling_rate,
                                        sample_stride=args.sample_stride, context_map_size=context_map_size, prior_map_size=prior_map_size, vis_map_size=None,
                                        max_distance=args.scene_distance, cache_file=args.train_cache, multi_agent=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=nuscenes_collate, num_workers=args.num_workers)
        
        valid_dataset = None
        valid_loader = None
        if args.val_partition is not None:
            valid_dataset = NuscenesDataset('./data/Preprocessed/nuScenes', args.val_partition, logger=logger, sampling_rate=args.sampling_rate,
                                            sample_stride=args.sample_stride, context_map_size=context_map_size, prior_map_size=prior_map_size, vis_map_size=None,
                                            max_distance=args.scene_distance, cache_file=args.val_cache, multi_agent=True)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=nuscenes_collate, num_workers=args.num_workers)


    elif args.dataset == 'argoverse':
        from dataset.argoverse import ArgoverseDataset, argoverse_collate
        logger.info("Loading Argoverse dataset.")
        train_dataset = ArgoverseDataset('./data/Preprocessed/Argoverse', args.train_partition, logger=logger, sampling_rate=args.sampling_rate,
                                        context_map_size=context_map_size, prior_map_size=prior_map_size, vis_map_size=None,
                                        max_distance=args.scene_distance, cache_file=args.train_cache, multi_agent=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=argoverse_collate, num_workers=args.num_workers)
        
        valid_dataset = None
        valid_loader = None
        if args.val_partition is not None:
            valid_dataset = ArgoverseDataset('./data/Preprocessed/Argoverse', args.val_partition, logger=logger, sampling_rate=args.sampling_rate,
                                            context_map_size=context_map_size, prior_map_size=prior_map_size, vis_map_size=None,
                                            max_distance=args.scene_distance, cache_file=args.val_cache, multi_agent=True)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=argoverse_collate, num_workers=args.num_workers)

    else:
        raise ValueError("Unknown Dataset {:s}.".format(args.dataset))

    logger.info(f'Train Examples: {len(train_dataset)}')
    if valid_dataset is not None:
        logger.info(f'Valid Examples: {len(valid_dataset)}')
    else:
        logger.info('Valid Examples: N/A')

    # Model optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.l2_reg)

    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9, weight_decay=args.l2_reg)
    
    else:
        raise ValueError("Unknown Optimizer {:s}.".format(args.optimizer))

    # Training Runner
    if "CAM" in args.model_type:
        from Proposed.utils import ModelTrainer
        trainer = ModelTrainer(model,
                               train_loader,
                               valid_loader,
                               ploss_criterion,
                               optimizer,
                               device,
                               exp_path,
                               logger,
                               args)

    trainer.train(args.num_epochs)


def test(args):
    # Initialize logger.
    test_path = pathlib.Path(args.test_path).joinpath(args.tag + '_' + datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('_%d_%B__%H_%M_'))
    if not test_path.is_dir():
        test_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(test_path.joinpath('training.log')))
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt='%Y-%m-%d %H:%M:%S'))
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("Test Path {:s}.".format(str(test_path)))
    
    # Set random seeds.
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    logger.info("Set random seed {:d}.".format(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get Model
    model, context_map_size = get_model(args)
    logger.info("Loaded Model {:s}.".format(str(model)))
    logger.info("Context Map:  {:s}.".format("N/A" if context_map_size is None else str(context_map_size)))
    
    ckpt = args.test_ckpt
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state'], strict=True)

    # Send model to Device:
    model = model.to(device)

    vis_map_size = (224, 224)
    if args.dataset == 'nuscenes':
        from dataset.nuscenes import NuscenesDataset, nuscenes_collate
        logger.info("Loading nuScenes dataset.")
        test_dataset = NuscenesDataset('./data/Preprocessed/nuScenes', args.test_partition, logger=logger, sampling_rate=args.sampling_rate,
                                       sample_stride=args.sample_stride, context_map_size=context_map_size, prior_map_size=None, vis_map_size=vis_map_size,
                                       max_distance=args.scene_distance, cache_file=args.test_cache, multi_agent=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=nuscenes_collate, num_workers=args.num_workers) 

    elif args.dataset == 'argoverse':
        from dataset.argoverse import ArgoverseDataset, argoverse_collate
        logger.info("Loading Argoverse dataset.")
        test_dataset = ArgoverseDataset('./data/Preprocessed/Argoverse', args.test_partition, logger=logger, sampling_rate=args.sampling_rate,
                                        context_map_size=context_map_size, prior_map_size=None, vis_map_size=vis_map_size,
                                        max_distance=args.scene_distance, cache_file=args.test_cache, multi_agent=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=argoverse_collate, num_workers=args.num_workers)

    else:
        raise ValueError("Unknown dataset name {:s}.".format(args.dataset))

    logger.info(f'Test Examples: {len(test_dataset)}')

    # Training Runner
    if "CAM" in args.model_type:
        from Proposed.utils import ModelTest
        tester = ModelTest(model,
                            test_loader,
                            device,
                            test_path,
                            logger,
                            args)

    tester.run(args.test_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Random Seed
    parser.add_argument('--seed', type=int, default=88245, help='Random seed.')

    # Training Options
    parser.add_argument('--tag', type=str, help="Add a tag to the saved folder")
    parser.add_argument('--exp_path', type=str, default='./experiment', help='Experient Directory')
    parser.add_argument('--restore_path', type=str, help='Restore model training from this tag.')
    parser.add_argument('--restore_epoch', type=int, help='Restore model training from this epoch.')
    parser.add_argument('--restore_optimizer', type=int, default=0, help='Restore the optimizer states along with the model weights.')

    # Optimization Parameters
    parser.add_argument('--optimizer', type=str, default='adam', help="Optimizer")
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of epochs for training the model")
    parser.add_argument('--init_lr', type=float, default=1e-4, help='Initial Learning Rate')
    parser.add_argument('--lr_decay', type=int, default=1, help="Wheter to perform LR decay on plateau.")
    parser.add_argument('--decay_factor', type=float, default=0.5, help="LR decay factor.")
    parser.add_argument('--decay_patience', type=int, default=5, help="LR decay patience.")
    parser.add_argument('--num_decays', type=int, default=3, help="Number of LR decays before the training halt.")
    parser.add_argument('--l2_reg', type=float, default=1e-4, help="L2 Deacy factor for weight regularization.")
    
    parser.add_argument('--batch_size', type=int, help='Batch size')

    # Model type
    # TODO: Organize MATF, Desire, CSP, R2P2, SimpleEncoderDecoder
    parser.add_argument('--model_type', type=str, default='AttGlobal_Scene_CAM_NFDecoder', help="AttGlobal_Scene_CAM_NFDecoder | Global_Scene_CAM_NFDecoder | Scene_CAM_NFDecoder | CAM_NFDecoder | CAM")
    
    # Hardware Parameters
    parser.add_argument('--num_workers', type=int, default=20, help="")
    parser.add_argument('--gpu_id', type=str, default='0', help="GPU IDs for model running")

    # Dataset Parameters
    parser.add_argument('--dataset', type=str, default='nuscenes', help="argoverse | nuscenes")
    parser.add_argument('--train_partition', type=str, help="Data partition to perform train")
    parser.add_argument('--train_cache', default=None, help="")
    parser.add_argument('--val_partition', type=str, help="Data partition to perform train")
    parser.add_argument('--val_cache', default=None, help="")
    parser.add_argument('--sample_stride', type=int, default=1, help="Stride between reference frames in a single scene (for nuScenes)")
    parser.add_argument('--scene_distance', type=float, default=56.0, help="Physical length in meters that the context map represents.")

    # Trajectory Parameters
    parser.add_argument('--sampling_rate', type=int, default=2, help="Sampling Rate for Encoding/Decoding sequences") # Hz | 10 frames per sec % sampling_interval=5 => 2 Hz

    # Common parameters for RNN-based trajectory encoder.
    # (Not used for R2P2 and Desire)
    parser.add_argument('--motion_features', type=int, default=128, help="Agent Embedding Dimension")
    parser.add_argument('--rnn_layers', type=int, default=1, help="")
    parser.add_argument('--rnn_dropout', type=float, default=0.0, help="")

    # Common parameters for generative models.
    parser.add_argument('--num_candidates', type=int, default=12, help="Number of trajectory candidates to generate.")

    # Parameters for Normalizing Flow Models (X_CAM_NFDecoders and R2P2).
    parser.add_argument('--beta', type=float, default=0.1, help="Ploss beta parameter")
    parser.add_argument('--velocity_const', type=float, default=0.5, help="Constant multiplied to dx in verlet integration")
    parser.add_argument('--ploss_type', type=str, default='map', help="Ploss Type, \"mseloss | logistic | map\"")
    parser.add_argument('--detach_output', default=0, type=int)

    """ TODO: Organize MATF, Desire, CSP, R2P2, SimpleEncoderDecoder
    ## Parameters for MATF
    parser.add_argument('--scene_dropout', type=float, help="Dropout rate for the encoded scene.")
    parser.add_argument('--scene_encoder', type=str, help="ShallowCNN | ResNet")
    parser.add_argument('--freeze_resnet', type=bool, help="Whether to freeze ResNet weights.")

    # Parameters for Convoultional Social Pooling-based models (MATF and CSP).
    parser.add_argument('--pooling_size', type=int, default=30, help="Map grid H and W dimension")

    # Parameters for GAN Models (MATF)
    # It first starts with gan weight = 0.1 and when the training epoch reaches 20, gan weight becomes 0.5 and so on.
    parser.add_argument('--noise_dim', type=int, default=16, help="")
    parser.add_argument('--gan_weight', type=float, default=[0.5, 0.7, 1, 1.5, 2.0, 2.5], help="Adversarial Training Alpha")
    parser.add_argument('--gan_weight_schedule', type=float, default=[20, 30, 40, 50, 65, 200], help="Decaying Gan Weight by Epoch")
    parser.add_argument('--disc_hidden', type=int, default=512, help="")  
    parser.add_argument('--disc_dropout', type=float, default=0.5, help="") 
    """

    # Model Testing Parameters
    parser.add_argument('--test_partition', type=str,
                        help="Data partition to perform test")
    parser.add_argument('--test_cache', type=str, help="")
    parser.add_argument('--test_path', default='./tests/', type=str, help="Test output dir")
    parser.add_argument('--test_ckpt', default=None, help="Model Checkpoint for test")
    parser.add_argument('--test_epochs', type=int, default=10, help='Number of test trials to calculate std.')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.test_ckpt is not None:
        test(args)
    else:
        train(args)
