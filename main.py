import os
import argparse

import torch
from torch.utils.data import DataLoader

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_channels = 5 if args.map_version == '2.1' else 3
    nfuture = int(3 * args.sampling_rate)

    if args.model_type == 'SimpleEncDec':
        from MATF.models import SimpleEncoderDecoder
        from MATF.utils import ModelTrainer

        model = SimpleEncoderDecoder(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=nfuture,
                                    lstm_layers=args.lstm_layers, lstm_dropout=args.lstm_dropout, noise_dim=args.noise_dim)

        use_scene = False
        scene_size = None
        ploss_type = None

    elif args.model_type == 'SocialPooling':

        from MATF.models import SocialPooling
        from MATF.utils import ModelTrainer

        model = SocialPooling(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=nfuture,
                              lstm_layers=args.lstm_layers, lstm_dropout=args.lstm_dropout, noise_dim=args.noise_dim,
                              pooling_size=args.pooling_size)

        use_scene = False
        scene_size = None
        ploss_type = None

    elif args.model_type == 'MATF':

        from MATF.models import MATF
        from MATF.utils import ModelTrainer

        model = MATF(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=nfuture,
                     lstm_layers=args.lstm_layers, lstm_dropout=args.lstm_dropout, noise_dim=args.noise_dim,
                     pooling_size=args.pooling_size,
                     encoder_type=args.scene_encoder, scene_channels=scene_channels, 
                     scene_dropout=args.scene_dropout, freeze_resnet=args.freeze_resnet)

        use_scene = True
        scene_size = (60, 60)
        ploss_type = None

    elif args.model_type == 'MATF_GAN':

        from MATF_GAN.models import MATF_Gen, MATF_Disc
        from MATF_GAN.utils import ModelTrainer

        model = MATF_Gen(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=nfuture,
                        lstm_layers=args.lstm_layers, lstm_dropout=args.lstm_dropout, noise_dim=args.noise_dim,
                        pooling_size=args.pooling_size, encoder_type=args.scene_encoder, scene_channels=scene_channels, 
                        scene_dropout=args.scene_dropout, freeze_resnet=args.freeze_resnet)

        discriminator = MATF_Disc(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=nfuture,
                                  lstm_layers=args.lstm_layers, lstm_dropout=args.lstm_dropout, noise_dim=args.noise_dim,
                                  pooling_size=args.pooling_size, encoder_type=args.scene_encoder, scene_channels=scene_channels, 
                                  scene_dropout=args.scene_dropout, freeze_resnet=args.freeze_resnet, 
                                  disc_hidden=args.disc_hidden, disc_dropout=args.disc_dropout)

        use_scene = True
        scene_size = (60, 60)
        ploss_type = None

    elif args.model_type == 'R2P2_SimpleRNN':

        from R2P2_MA.models import R2P2_SimpleRNN
        from R2P2_MA.utils import ModelTrainer
        model = R2P2_SimpleRNN(velocity_const=args.velocity_const, num_candidates=args.num_candidates, decoding_steps=nfuture)

        use_scene = False
        scene_size = None
        ploss_type = args.ploss_type

        if ploss_type == 'mseloss':
            from R2P2_MA.model_utils import MSE_Ploss
            ploss_criterion = MSE_Ploss()
        else:
            from R2P2_MA.model_utils import Interpolated_Ploss
            ploss_criterion = Interpolated_Ploss()

    elif args.model_type == 'R2P2_RNN':

        from R2P2_MA.models import R2P2_RNN
        from R2P2_MA.utils import ModelTrainer
        
        model = R2P2_RNN(scene_channels=scene_channels, velocity_const=args.velocity_const, num_candidates=args.num_candidates, decoding_steps=nfuture)

        use_scene = True
        scene_size = (64, 64)
        ploss_type = args.ploss_type

        if ploss_type == 'mseloss':
            from R2P2_MA.model_utils import MSE_Ploss
            ploss_criterion = MSE_Ploss()
        else:
            from R2P2_MA.model_utils import Interpolated_Ploss
            ploss_criterion = Interpolated_Ploss()

    elif args.model_type == "Desire":
        from Desire.models import DESIRE_SGM, DESIRE_IOC
        from Desire.utils import ModelTrainer

        model = DESIRE_SGM(decoding_steps=nfuture, num_candidates=args.num_candidates)
        ioc = DESIRE_IOC(in_channels=scene_channels, decoding_steps=nfuture)

        use_scene = True
        scene_size = (64, 64)
        ploss_type = args.ploss_type

    elif args.model_type == 'CAM':

        from Proposed.models import CAM
        from Proposed.utils import ModelTrainer

        model = CAM(device=device, embedding_dim=args.agent_embed_dim, nfuture=nfuture, att_dropout=args.att_dropout)

        use_scene = False
        scene_size = (64, 64)
        ploss_type = None

    elif args.model_type == 'Scene_CAM':	
        from Proposed.models import Scene_CAM	
        from Proposed.utils import ModelTrainer	
        model = Scene_CAM(device=device, embedding_dim=args.agent_embed_dim, nfuture=nfuture, att_dropout=args.att_dropout)	
        use_scene = True	
        scene_size = (64, 64)	
        ploss_type = None	


    elif args.model_type == 'CAM_NFDecoder': 

        from Proposed.models import CAM_NFDecoder
        from Proposed.utils import ModelTrainer

        model = CAM_NFDecoder(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=nfuture, att_dropout=args.att_dropout,
                            velocity_const=args.velocity_const, num_candidates=args.num_candidates, decoding_steps=nfuture)

        use_scene = True
        scene_size = (64, 64)
        ploss_type = args.ploss_type

        if ploss_type == 'mseloss':
            from R2P2_MA.model_utils import MSE_Ploss
            ploss_criterion = MSE_Ploss()
        else:
            from R2P2_MA.model_utils import Interpolated_Ploss
            ploss_criterion = Interpolated_Ploss()

    elif args.model_type == 'Scene_CAM_NFDecoder':

        from Proposed.models import Scene_CAM_NFDecoder
        from Proposed.utils import ModelTrainer

        model = Scene_CAM_NFDecoder(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=nfuture, att_dropout=args.att_dropout,
                            velocity_const=args.velocity_const, num_candidates=args.num_candidates, decoding_steps=nfuture)

        use_scene = True
        scene_size = (64, 64)
        ploss_type = args.ploss_type

        if ploss_type == 'mseloss':
            from R2P2_MA.model_utils import MSE_Ploss
            ploss_criterion = MSE_Ploss()

        else:
            from R2P2_MA.model_utils import Interpolated_Ploss
            ploss_criterion = Interpolated_Ploss()

    elif args.model_type == 'Global_Scene_CAM_NFDecoder' or args.model_type == 'AttGlobal_Scene_CAM_NFDecoder':

        from Proposed.models import Global_Scene_CAM_NFDecoder
        from Proposed.utils import ModelTrainer

        if args.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
            crossmodal_attention = True
        else:
            crossmodal_attention = False
        model = Global_Scene_CAM_NFDecoder(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=nfuture, att_dropout=args.att_dropout,
                            velocity_const=args.velocity_const, num_candidates=args.num_candidates, decoding_steps=nfuture, att=crossmodal_attention)

        use_scene = True
        scene_size = (64, 64)
        ploss_type = args.ploss_type

        if ploss_type == 'mseloss':
            from R2P2_MA.model_utils import MSE_Ploss
            ploss_criterion = MSE_Ploss()
        else:
            from R2P2_MA.model_utils import Interpolated_Ploss
            ploss_criterion = Interpolated_Ploss()

    else:
        raise ValueError("Unknown model type {:s}.".format(args.model_type))

    # Send model to Device:
    model = model.to(device)

    if args.dataset == 'argoverse' and args.model_type not in ["R2P2_SimpleRNN", "R2P2_RNN"]:

        from dataset.argoverse import ArgoverseDataset, argoverse_collate

        train_dataset = ArgoverseDataset('train', map_version=args.map_version, sampling_rate=args.sampling_rate, sample_stride=args.sample_stride,
                                         use_scene=use_scene, scene_size=scene_size, ploss_type=ploss_type, num_workers=args.num_workers, cache_file=args.train_cache, multi_agent=args.multi_agent)
        valid_dataset = ArgoverseDataset('val', map_version=args.map_version, sampling_rate=args.sampling_rate, sample_stride=args.sample_stride,
                                         use_scene=use_scene, scene_size=scene_size, ploss_type=ploss_type, num_workers=args.num_workers, cache_file=args.val_cache, multi_agent=args.multi_agent)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                collate_fn=lambda x: argoverse_collate(x), num_workers=args.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                collate_fn=lambda x: argoverse_collate(x), num_workers=1)

    elif args.dataset == 'argoverse' and args.model_type in ["R2P2_SimpleRNN", "R2P2_RNN"]:	
        
        from dataset.argoverse_r2p2 import ArgoverseDataset_R2P2, argoverse_collate_R2P2	
        from dataset.argoverse import ArgoverseDataset, argoverse_collate	
        train_dataset = ArgoverseDataset_R2P2('train', map_version=args.map_version, sampling_rate=args.sampling_rate, sample_stride=args.sample_stride,	
                                         use_scene=use_scene, scene_size=scene_size, ploss_type=ploss_type, num_workers=args.num_workers, cache_file=args.train_cache)	
        valid_dataset = ArgoverseDataset('val', map_version=args.map_version, sampling_rate=args.sampling_rate, sample_stride=args.sample_stride,	
                                         use_scene=use_scene, scene_size=scene_size, ploss_type=ploss_type, num_workers=args.num_workers, cache_file=args.val_cache)	
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,	
                                collate_fn=lambda x: argoverse_collate_R2P2(x), num_workers=args.num_workers)	
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,	
                                collate_fn=lambda x: argoverse_collate(x), num_workers=1)	
    
    elif args.dataset == 'nuscenes' and args.model_type not in ["R2P2_SimpleRNN", "R2P2_RNN"]:

        from dataset.nuscenes import NuscenesDataset, nuscenes_collate

        train_dataset = NuscenesDataset('train', map_version=args.map_version, sampling_rate=args.sampling_rate, sample_stride=args.sample_stride,
                                        use_scene=use_scene, scene_size=scene_size, ploss_type=ploss_type, num_workers=args.num_workers, cache_file=args.train_cache, multi_agent=args.multi_agent)
        valid_dataset = NuscenesDataset('val', map_version=args.map_version, sampling_rate=args.sampling_rate, sample_stride=args.sample_stride,
                                        use_scene=use_scene, scene_size=scene_size, ploss_type=ploss_type, num_workers=args.num_workers, cache_file=args.val_cache, multi_agent=args.multi_agent)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                  collate_fn=lambda x: nuscenes_collate(x), num_workers=args.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                  collate_fn=lambda x: nuscenes_collate(x), num_workers=1)
    
    elif args.dataset == 'nuscenes' and args.model_type in ["R2P2_SimpleRNN", "R2P2_RNN"]:	

        from dataset.nuscenes_r2p2 import NuscenesDataset_R2P2, nuscenes_collate_R2P2	
        from dataset.nuscenes import NuscenesDataset, nuscenes_collate	

        train_dataset = NuscenesDataset_R2P2('train', map_version   =args.map_version, sampling_rate=args.sampling_rate, sample_stride=args.sample_stride,	
                                        use_scene=use_scene, scene_size=scene_size, ploss_type=ploss_type, num_workers=args.num_workers, cache_file=args.train_cache, multi_agent=args.multi_agent)
        valid_dataset = NuscenesDataset('val', map_version=args.map_version, sampling_rate=args.sampling_rate, sample_stride=args.sample_stride,	
                                        use_scene=use_scene, scene_size=scene_size, ploss_type=ploss_type, num_workers=args.num_workers, cache_file=args.val_cache, multi_agent=args.multi_agent)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,	
                                  collate_fn=lambda x: nuscenes_collate_R2P2(x), num_workers=args.num_workers)	
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,	
                                  collate_fn=lambda x: nuscenes_collate(x), num_workers=1)

    elif args.dataset == 'carla':
        from dataset.carla import CarlaDataset, carla_collate

        train_dataset = CarlaDataset('train', map_version=args.map_version, sampling_rate=args.sampling_rate, sample_stride=args.sample_stride,
                                        use_scene=use_scene, scene_size=scene_size, ploss_type=ploss_type, num_workers=args.num_workers, cache_file=args.train_cache, multi_agent=args.multi_agent)
        valid_dataset = CarlaDataset('val', map_version=args.map_version, sampling_rate=args.sampling_rate, sample_stride=args.sample_stride,
                                        use_scene=use_scene, scene_size=scene_size, ploss_type=ploss_type, num_workers=args.num_workers, cache_file=args.val_cache, multi_agent=args.multi_agent)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                  collate_fn=lambda x: carla_collate(x), num_workers=args.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                  collate_fn=lambda x: carla_collate(x), num_workers=1)


    else:
        raise ValueError("Unknown dataset name {:s}.".format(args.dataset))

    print(f'Train Examples: {len(train_dataset)} | Valid Examples: {len(valid_dataset)}')

    # Model optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)

    # Trainer
    exp_path = args.exp_path

    # Training Runner
    if args.model_type in ["R2P2_SimpleRNN", "R2P2_RNN"] or "NFDecoder" in args.model_type:
        ploss_criterion = ploss_criterion.to(device)

        trainer = ModelTrainer(model, train_loader, valid_loader, optimizer, exp_path, args, device, ploss_criterion)

    elif args.model_type == "MATF_GAN":
        discriminator = discriminator.to(device)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, weight_decay=1e-4)

        trainer = ModelTrainer(model, train_loader, valid_loader, optimizer, exp_path, args, device, discriminator, optimizer_d)

    elif args.model_type == "Desire":

        ioc = ioc.to(device)
        optimizer_ioc = torch.optim.Adam(ioc.parameters(), lr=args.learning_rate, weight_decay=1e-4)

        trainer = ModelTrainer(model, train_loader, valid_loader, optimizer, exp_path, args, device, ioc, optimizer_ioc)

    else:

        trainer = ModelTrainer(model, train_loader, valid_loader, optimizer, exp_path, args, device)

    trainer.train(args.num_epochs)


def test(args):	
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_channels = 5 if args.map_version == '2.1' else 3
    nfuture = int(3 * args.sampling_rate)

    if args.model_type == 'SimpleEncDec':
        from MATF.models import SimpleEncoderDecoder
        from MATF.utils import ModelTest
        model = SimpleEncoderDecoder(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=nfuture,	
                                    lstm_layers=args.lstm_layers, lstm_dropout=args.lstm_dropout, noise_dim=args.noise_dim)	
        use_scene = False	
        scene_size = None	
        ploss_type = None	
    
    elif args.model_type == 'SocialPooling':

        from MATF.models import SocialPooling
        from MATF.utils import ModelTest

        model = SocialPooling(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=nfuture,
                              lstm_layers=args.lstm_layers, lstm_dropout=args.lstm_dropout, noise_dim=args.noise_dim,
                              pooling_size=args.pooling_size)
        use_scene = False
        scene_size = None
        ploss_type = None
		
    elif args.model_type == 'MATF':

        from MATF.models import MATF
        from MATF.utils import ModelTest

        model = MATF(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=nfuture,
                     lstm_layers=args.lstm_layers, lstm_dropout=args.lstm_dropout, noise_dim=args.noise_dim,
                     pooling_size=args.pooling_size,
                     encoder_type=args.scene_encoder, scene_channels=scene_channels, 
                     scene_dropout=args.scene_dropout, freeze_resnet=args.freeze_resnet)
        use_scene = True
        scene_size = (60, 60)
        ploss_type = None

    elif args.model_type == 'MATF_GAN':

        from MATF_GAN.models import MATF_Gen, MATF_Disc
        from MATF_GAN.utils import ModelTest

        model = MATF_Gen(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=nfuture,
                        lstm_layers=args.lstm_layers, lstm_dropout=args.lstm_dropout, noise_dim=args.noise_dim,
                        pooling_size=args.pooling_size, encoder_type=args.scene_encoder, scene_channels=scene_channels, 
                        scene_dropout=args.scene_dropout, freeze_resnet=args.freeze_resnet)

        use_scene = True
        scene_size = (60, 60)
        ploss_type = None

    elif args.model_type == 'R2P2_SimpleRNN':

        from R2P2_MA.models import R2P2_SimpleRNN
        from R2P2_MA.utils import ModelTest
        model = R2P2_SimpleRNN(velocity_const=args.velocity_const, num_candidates=args.num_candidates, decoding_steps=nfuture)

        use_scene = False
        scene_size = None
        ploss_type = args.ploss_type

        if ploss_type == 'mseloss':
            from R2P2_MA.model_utils import MSE_Ploss
            ploss_criterion = MSE_Ploss()
        else:
            from R2P2_MA.model_utils import Interpolated_Ploss
            ploss_criterion = Interpolated_Ploss()

    elif args.model_type == 'R2P2_RNN':

        from R2P2_MA.models import R2P2_RNN
        from R2P2_MA.utils import ModelTest
        model = R2P2_RNN(scene_channels=scene_channels, velocity_const=args.velocity_const, num_candidates=args.num_candidates, decoding_steps=nfuture)

        use_scene = True
        scene_size = (64, 64)
        ploss_type = args.ploss_type

        if ploss_type == 'mseloss':
            from R2P2_MA.model_utils import MSE_Ploss
            ploss_criterion = MSE_Ploss()
        else:
            from R2P2_MA.model_utils import Interpolated_Ploss
            ploss_criterion = Interpolated_Ploss()

    elif args.model_type == "Desire":
        from Desire.models import DESIRE_SGM, DESIRE_IOC
        from Desire.utils import ModelTest
        model = DESIRE_SGM(decoding_steps=nfuture, num_candidates=args.num_candidates)
        ioc = DESIRE_IOC(in_channels=scene_channels, decoding_steps=nfuture)

        use_scene = True
        scene_size = (64, 64)
        ploss_type = args.ploss_type
		
    elif args.model_type == 'CAM':

        from Proposed.models import CAM
        from Proposed.utils import ModelTest

        model = CAM(device=device, embedding_dim=args.agent_embed_dim, nfuture=nfuture, att_dropout=args.att_dropout)

        use_scene = False
        scene_size = (64, 64)
        ploss_type = None
        
    elif args.model_type == 'Scene_CAM':

        from Proposed.models import Scene_CAM
        from Proposed.utils import ModelTrainer

        model = Scene_CAM(device=device, embedding_dim=args.agent_embed_dim, nfuture=nfuture, att_dropout=args.att_dropout)
                                                                                                                 
        use_scene = True
        scene_size = (64, 64)
        ploss_type = None

    elif args.model_type == 'CAM_NFDecoder': 

        from Proposed.models import CAM_NFDecoder
        from Proposed.utils import ModelTest

        model = CAM_NFDecoder(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=nfuture, att_dropout=args.att_dropout,
                            velocity_const=args.velocity_const, num_candidates=args.num_candidates, decoding_steps=nfuture)

        use_scene = True
        scene_size = (64, 64)
        ploss_type = args.ploss_type

        if ploss_type == 'mseloss':
            from R2P2_MA.model_utils import MSE_Ploss
            ploss_criterion = MSE_Ploss()
        else:
            from R2P2_MA.model_utils import Interpolated_Ploss
            ploss_criterion = Interpolated_Ploss()

    elif args.model_type == 'Scene_CAM_NFDecoder':

        from Proposed.models import Scene_CAM_NFDecoder
        from Proposed.utils import ModelTest

        model = Scene_CAM_NFDecoder(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=nfuture, att_dropout=args.att_dropout,
                            velocity_const=args.velocity_const, num_candidates=args.num_candidates, decoding_steps=nfuture)

        use_scene = True
        scene_size = (64, 64)
        ploss_type = args.ploss_type

        if ploss_type == 'mseloss':
            from R2P2_MA.model_utils import MSE_Ploss
            ploss_criterion = MSE_Ploss()
        else:
            from R2P2_MA.model_utils import Interpolated_Ploss
            ploss_criterion = Interpolated_Ploss()

    elif args.model_type == 'Global_Scene_CAM_NFDecoder' or args.model_type == 'AttGlobal_Scene_CAM_NFDecoder':

        from Proposed.models import Global_Scene_CAM_NFDecoder
        from Proposed.utils import ModelTest

        if args.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
            crossmodal_attention = True

        else:
            crossmodal_attention = False

        model = Global_Scene_CAM_NFDecoder(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=nfuture, att_dropout=args.att_dropout,
                            velocity_const=args.velocity_const, num_candidates=args.num_candidates, decoding_steps=nfuture, att=crossmodal_attention)

        use_scene = True
        scene_size = (64, 64)
        ploss_type = args.ploss_type

        if ploss_type == 'mseloss':
            from R2P2_MA.model_utils import MSE_Ploss
            ploss_criterion = MSE_Ploss()
        else:
            from R2P2_MA.model_utils import Interpolated_Ploss
            ploss_criterion = Interpolated_Ploss()

    else:
        raise ValueError("Unknown model type {:s}.".format(args.model_type))

    # Send model to Device:
    model = model.to(device)

    if args.dataset == 'argoverse':
        from dataset.argoverse import ArgoverseDataset, argoverse_collate

        dataset = ArgoverseDataset(args.test_partition, map_version=args.map_version, sampling_rate=args.sampling_rate, sample_stride=args.sample_stride,
                                        use_scene=use_scene, scene_size=scene_size, ploss_type=ploss_type, num_workers=args.num_workers, cache_file=args.test_cache, multi_agent=args.multi_agent)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                collate_fn=lambda x: argoverse_collate(x), num_workers=1)

    elif args.dataset == 'nuscenes':
        from dataset.nuscenes import NuscenesDataset, nuscenes_collate

        dataset = NuscenesDataset(args.test_partition, map_version=args.map_version, sampling_rate=args.sampling_rate, sample_stride=args.sample_stride,
                                        use_scene=use_scene, scene_size=scene_size, ploss_type=ploss_type, num_workers=args.num_workers, cache_file=args.test_cache, multi_agent=args.multi_agent)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                collate_fn=lambda x: nuscenes_collate(x), num_workers=1)
    elif args.dataset == 'carla':
        from dataset.carla import CarlaDataset, carla_collate

        dataset = CarlaDataset(args.test_partition, map_version=args.map_version, sampling_rate=args.sampling_rate, sample_stride=args.sample_stride,
                                        use_scene=use_scene, scene_size=scene_size, ploss_type=ploss_type, num_workers=args.num_workers, cache_file=args.test_cache, multi_agent=args.multi_agent)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                 collate_fn=lambda x: carla_collate(x), num_workers=1)

    else:

        raise ValueError("Unknown dataset name {:s}.".format(args.dataset))

    print(f'Test Examples: {len(dataset)}')
    if not os.path.isdir(args.test_dir):
        os.mkdir(args.test_dir)

    if args.model_type in ["R2P2_SimpleRNN", "R2P2_RNN"] or "NFDecoder" in args.model_type:
        ploss_criterion = ploss_criterion.to(device)
        tester = ModelTest(model, data_loader, args, device, ploss_criterion)

    elif args.model_type == "MATF_GAN":

        tester = ModelTest(model, data_loader, args, device)

    elif args.model_type == "Desire":
        ioc = ioc.to(device)
        
        tester = ModelTest(model, ioc, data_loader, args, device)

    else:	
        tester = ModelTest(model, data_loader, args, device)

    tester.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training Tag
    parser.add_argument('--tag', type=str, help="Add a tag to the saved folder")
    parser.add_argument('--exp_path', type=str, default='./experiment', help='Experient Directory')

    # Model type
    parser.add_argument('--model_type', type=str, default='SimpleEncDec', help="SimpleEncDec | SocialPooling | MATF | MATF_GAN | CAM | MACAM | R2P2_RNN | R2P2_SimpleRNN | Desire")
    
    # Hardware Parameters
    parser.add_argument('--num_workers', type=int, default=20, help="")
    parser.add_argument('--gpu_devices', type=str, default='0', help="GPU IDs for model running")

    # Dataset Parameters
    parser.add_argument('--dataset', type=str, default='argoverse', help="argoverse | nuscenes | carla")
    parser.add_argument('--train_cache', default=None, help="")
    parser.add_argument('--val_cache', default=None, help="")

    # Episode sampling parameters
    parser.add_argument('--sample_stride', type=int, default=1, help="Stride between reference frames in a single episode")

    # Trajectory Parameters
    parser.add_argument('--sampling_rate', type=int, default=2, help="Sampling Rate for Encoding/Decoding sequences") # Hz | 10 frames per sec % sampling_interval=5 => 2 Hz

    # Scene Context Parameters
    parser.add_argument('--map_version', type=str, default='2.0', help="Map version")
    ## Only used for MATFs
    parser.add_argument('--scene_dropout', type=float, default=0.5, help="")
    parser.add_argument('--scene_encoder', type=str, default='ShallowCNN', help="ShallowCNN | ResNet")
    parser.add_argument('--freeze_resnet', type=bool, default=True, help="")

    # Agent Encoding
    # (Not used for R2P2 and Desire)
    parser.add_argument('--agent_embed_dim', type=int, default=128, help="Agent Embedding Dimension")
    parser.add_argument('--lstm_layers', type=int, default=1, help="")
    parser.add_argument('--lstm_dropout', type=float, default=0.3, help="")

    # the number of candidate futures in generative models
    parser.add_argument('--num_candidates', type=int, default=12, help="Number of trajectory candidates sampled")
    
    # CSP Models
    parser.add_argument('--pooling_size', type=int, default=30, help="Map grid H and W dimension")

    # Attention Models
    parser.add_argument('--att_dropout', type=float, default=0.1, help="")

    # Normalizing Flow Models
    parser.add_argument('--multi_agent', type=int, default=1, help="Enables multi-agent setting for dataset")
    parser.add_argument('--beta', type=float, default=0.1, help="Ploss beta parameter")
    parser.add_argument('--velocity_const', type=float, default=0.5, help="Constant multiplied to dx in verlet integration")
    parser.add_argument('--ploss_type', type=str, default='map', help="Ploss Type, \"mseloss | logistic | map\"")

    # GAN Models
    # It first starts with gan weight = 0.1 and when the training epoch reaches 20, gan weight becomes 0.5 and so on.
    parser.add_argument('--noise_dim', type=int, default=16, help="")
    parser.add_argument('--gan_weight', type=float, default=[0.5, 0.7, 1, 1.5, 2.0, 2.5], help="Adversarial Training Alpha")
    parser.add_argument('--gan_weight_schedule', type=float, default=[20, 30, 40, 50, 65, 200], help="Decaying Gan Weight by Epoch")
    parser.add_argument('--disc_hidden', type=int, default=512, help="")  
    parser.add_argument('--disc_dropout', type=float, default=0.5, help="") 

    # Optimization Parameters
    parser.add_argument('--optimizer', type=str, default='adam', help="Optimizer")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training the model")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--load_ckpt', default=None, help='Load Model Checkpoint')
    parser.add_argument('--start_epoch', type=int, default=1, help='Resume Model Training')
    
    # Model Testing Parameters
    parser.add_argument('--test_partition', type=str,
                        default='test_obs',
                        help="Data partition to perform test")
    parser.add_argument('--test_cache', type=str, help="")
    parser.add_argument('--test_dir', type=str, help="Test output dir")
    parser.add_argument('--test_ckpt', default=None, help="Model Checkpoint for test")
    parser.add_argument('--test_times', type=int, default=10, help='Number of test trials to calculate std.')
    parser.add_argument('--test_render', type=int, default=1, help='Whether to render the outputs as figure')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    if args.test_ckpt is not None:
        test(args)
    else:
        train(args)
