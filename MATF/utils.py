import os
import sys
import time
import numpy as np
import datetime

import pickle as pkl

import matplotlib.pyplot as plt
import cv2
import torch

from tqdm import tqdm

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions.multivariate_normal import MultivariateNormal

from multiprocessing import Pool

import logging


class ModelTrainer:

    def __init__(self, model, train_loader, valid_loader, optimizer, exp_path, args, device):

        self.exp_path = os.path.join(exp_path, args.tag + '_' + datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-4))).strftime('_%d_%B__%H_%M_'))
        if not os.path.exists(self.exp_path):
            os.mkdir(self.exp_path)

        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.exp_path, 'training.log'))
        sh = logging.StreamHandler(sys.stdout)
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt='%Y-%m-%d %H:%M:%S'))
        sh.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        self.logger.info(f'Current Exp Path: {self.exp_path}')

        self.writter = SummaryWriter(os.path.join(self.exp_path, 'logs'))

        self.model_type = args.model_type
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.device = device

        self.decoding_steps = int(3 * args.sampling_rate)
        self.encoding_steps = int(2 * args.sampling_rate)

        if self.model_type == 'MATF':
            self.map_version = args.map_version
        else:
            self.map_version = ''
        
        self.optimizer = optimizer    
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=(1/2), verbose=True, patience=3)

        if args.load_ckpt is not None:
            self.load_checkpoint(args.load_ckpt)

        # Other Parameters
        self.best_valid_ade = 1e9
        self.best_valid_fde = 1e9
        self.start_epoch = args.start_epoch


    def train(self, num_epochs):
        self.logger.info('Model Type: '+str(self.model_type))
        self.logger.info('TRAINING .....')

        for epoch in tqdm(range(self.start_epoch, self.start_epoch + num_epochs)):
            self.logger.info("==========================================================================================")

            train_loss, train_ades, train_fdes = self.train_single_epoch()
            valid_ades, valid_fdes, scheduler_metric = self.inference()

            ## unwrapping ADEs/FDEs
            train_minade2, train_avgade2, train_minade3, train_avgade3 = train_ades
            train_minfde2, train_avgfde2, train_minfde3, train_avgfde3 = train_fdes

            valid_minade2, valid_avgade2, valid_minade3, valid_avgade3 = valid_ades
            valid_minfde2, valid_avgfde2, valid_minfde3, valid_avgfde3 = valid_fdes

            self.best_valid_ade = min(*valid_ades, self.best_valid_ade)
            self.best_valid_fde = min(*valid_fdes, self.best_valid_fde)
            self.scheduler.step(scheduler_metric)

            logging_msg1 = (
                f'| Epoch: {epoch:02} | Train Loss: {train_loss:0.6f} '
                f'| Train minADE[2/3]: {train_minade2:0.4f} / {train_minade3:0.4f} | Train minFDE[2/3]: {train_minfde2:0.4f} / {train_minfde3:0.4f} '
                f'| Train avgADE[2/3]: {train_avgade2:0.4f} / {train_avgade3:0.4f} | Train avgFDE[2/3]: {train_avgfde2:0.4f} / {train_avgfde3:0.4f}'
            )

            logging_msg2 = (
                f'| Epoch: {epoch:02} | ' #Valid Loss: {valid_loss:0.6f} '
                f'| Valid minADE[2/3]: {valid_minade2:0.4f} / {valid_minade3:0.4f} | Valid minFDE[2/3]: {valid_minfde2:0.4f} /{valid_minfde3:0.4f} '
                f'| Valid avgADE[2/3]: {valid_avgade2:0.4f} / {valid_avgade3:0.4f} | Valid avgFDE[2/3]: {valid_avgfde2:0.4f} /{valid_avgfde3:0.4f} '
                f'| Scheduler Metric: {scheduler_metric:0.4f} | Learning Rate: {self.get_lr():g}\n'
            )

            self.logger.info("------------------------------------------------------------------------------------------")
            self.logger.info(logging_msg1)
            self.logger.info(logging_msg2)

            self.save_checkpoint(epoch, ade=valid_avgade3, fde=valid_avgfde3)

            # Log values to Tensorboard
            self.writter.add_scalar('data/Train_Loss', train_loss, epoch)
            self.writter.add_scalar('data/Learning_Rate', self.get_lr(), epoch)

            self.writter.add_scalar('data/Train_minADE2', train_minade2, epoch)
            self.writter.add_scalar('data/Train_minFDE2', train_minfde2, epoch)
            self.writter.add_scalar('data/Train_minADE3', train_minade3, epoch)
            self.writter.add_scalar('data/Train_minFDE3', train_minfde3, epoch)

            self.writter.add_scalar('data/Train_avgADE2', train_avgade2, epoch)
            self.writter.add_scalar('data/Train_avgFDE2', train_avgfde2, epoch)
            self.writter.add_scalar('data/Train_avgADE3', train_avgade3, epoch)
            self.writter.add_scalar('data/Train_avgFDE3', train_avgfde3, epoch)

            self.writter.add_scalar('data/Scheduler_Metric', scheduler_metric, epoch)
            # self.writter.add_scalar('data/Valid_Loss', valid_loss, epoch)

            self.writter.add_scalar('data/Valid_minADE2', valid_minade2, epoch)
            self.writter.add_scalar('data/Valid_minFDE2', valid_minfde2, epoch)
            self.writter.add_scalar('data/Valid_minADE3', valid_minade3, epoch)
            self.writter.add_scalar('data/Valid_minFDE3', valid_minfde3, epoch)

            self.writter.add_scalar('data/Valid_avgADE2', valid_avgade2, epoch)
            self.writter.add_scalar('data/Valid_avgFDE2', valid_avgfde2, epoch)
            self.writter.add_scalar('data/Valid_avgADE3', valid_avgade3, epoch)
            self.writter.add_scalar('data/Valid_avgFDE3', valid_avgfde3, epoch)

        self.writter.close()
        self.logger.info("Training Complete! ")
        self.logger.info(f'| Best Valid ADE: {self.best_valid_ade} | Best Valid FDE: {self.best_valid_fde} |')


    def train_single_epoch(self):
        """Trains the model for a single round."""

        self.model.train()
        epoch_loss = 0.0

        epoch_minade2, epoch_avgade2 = 0.0, 0.0
        epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
        epoch_minade3, epoch_avgade3 = 0.0, 0.0
        epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
    
        epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

        H = W = 60
        if '2.' in self.map_version:
            with torch.no_grad():
                coordinate_2d = np.indices((H, W))
                coordinate = np.ravel_multi_index(coordinate_2d, dims=(H, W))
                coordinate = torch.FloatTensor(coordinate)
                coordinate = coordinate.reshape((1, 1, H, W))

                coordinate_std, coordinate_mean = torch.std_mean(coordinate)
                coordinate = (coordinate - coordinate_mean) / coordinate_std

                distance_2d = coordinate_2d - np.array([(H-1)/2, (H-1)/2]).reshape((2, 1, 1))
                distance = np.sqrt((distance_2d ** 2).sum(axis=0))
                distance = torch.FloatTensor(distance)
                distance = distance.reshape((1, 1, H, W))

                distance_std, distance_mean = torch.std_mean(distance)
                distance = (distance - distance_mean) / distance_std
            
            coordinate = coordinate.to(self.device)
            distance = distance.to(self.device)

        for b, batch in enumerate(self.train_loader):

            print("Working on batch {:d}/{:d}".format(b+1, len(self.train_loader)), end='\r')

            self.optimizer.zero_grad()

            scene_images, log_prior, \
            future_agent_masks, \
            num_past_agents,   past_agents_traj,   past_agents_traj_len,   past_agents_traj_len_idx, \
            num_future_agents, future_agents_traj, future_agents_traj_len, future_agents_traj_len_idx, \
            two_mask, three_mask, \
            decode_start_vel, decode_start_pos, \
            scene_id = batch

            # Detect dynamic batch size
            batch_size = scene_images.size(0)
            num_agents = future_agents_traj.size(0)

            if '2.' in self.map_version:
                coordinate_batch = coordinate.expand(batch_size, -1, -1, -1)
                distance_batch = distance.expand(batch_size, -1, -1, -1)
                scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)

            elif self.map_version == '1.3':
                scene_images = scene_images.to(self.device)

            num_past_agents = num_past_agents.to(self.device)
            past_agents_traj = past_agents_traj.to(self.device)
            past_agents_traj_len = past_agents_traj_len.to(self.device)

            future_agents_traj = future_agents_traj.to(self.device)
            future_agents_traj_len = future_agents_traj_len.to(self.device)
            future_agents_traj_len_idx = future_agents_traj_len_idx.to(self.device)

            two_mask = two_mask.to(self.device)
            three_mask = three_mask.to(self.device)

            num_future_agents = num_future_agents.to(self.device)
            decode_start_vel = decode_start_vel.to(self.device)
            decode_start_pos = decode_start_pos.to(self.device)
            
            episode_idx = torch.arange(len(num_past_agents), device=self.device).repeat_interleave(num_past_agents)

            # Inference
            if 'SimpleEncDec' == self.model_type:
                predicted_trajs = self.model(past_agents_traj, past_agents_traj_len, future_agent_masks,
                                            decode_start_vel, decode_start_pos)

            elif 'SocialPooling' == self.model_type:
                predicted_trajs = self.model(past_agents_traj, past_agents_traj_len, episode_idx, future_agent_masks,
                                            decode_start_vel, decode_start_pos)             

            elif 'MATF' in self.model_type:
                stochastic = False                
                predicted_trajs = self.model(past_agents_traj, past_agents_traj_len, episode_idx, future_agent_masks,
                                            decode_start_vel, decode_start_pos, 
                                            scene_images, stochastic)

            else:
                raise ValueError("Unknown model type {:s}.".format(self.model_type))

            agent_time_index = torch.arange(num_agents, device=self.device).repeat_interleave(future_agents_traj_len)
            time_normalizer = future_agents_traj_len.float().repeat_interleave(future_agents_traj_len)

            error = future_agents_traj - predicted_trajs # A x Td x 2
            batch_loss = (error ** 2).sum(dim=-1) # x**2 + y**2 | A x Td 
            batch_loss = batch_loss[agent_time_index, future_agents_traj_len_idx] / time_normalizer
            batch_loss = batch_loss.sum() / (num_agents * 2.0)

            with torch.no_grad():

                # Two-Errors
                sq_error2 = (error[two_mask, :int(self.decoding_steps*2/3), :] ** 2).sum(2).sqrt() # A X Td X 2 >> A X Td
                sq_error3 = (error[three_mask, :, :] ** 2).sum(2).sqrt()

                num_agents2 = sq_error2.size(0)
                num_agents3 = sq_error3.size(0)

                ade2 = sq_error2.mean(dim=-1) # A X T >> A
                fde2 = sq_error2[... , -1]
                ade3 = sq_error3.mean(dim=-1)
                fde3 = sq_error3[... , -1]
                
                # 2-Loss
                batch_minade2 = ade2.mean() # A >> 1
                batch_minfde2 = fde2.mean()
                batch_avgade2 = ade2.mean()
                batch_avgfde2 = fde2.mean()
                
                # 3-Loss
                batch_minade3 = ade3.mean()
                batch_minfde3 = fde3.mean()
                batch_avgade3 = ade3.mean()
                batch_avgfde3 = fde3.mean()


            # Loss backward
            batch_loss.backward()
            self.optimizer.step()

            epoch_loss += batch_loss.item()

            epoch_minade2 += batch_minade2.item() * num_agents2
            epoch_minfde2 += batch_minfde2.item() * num_agents2
            epoch_avgade2 += batch_avgade2.item() * num_agents2
            epoch_avgfde2 += batch_avgfde2.item() * num_agents2

            epoch_minade3 += batch_minade3.item() * num_agents3
            epoch_minfde3 += batch_minfde3.item() * num_agents3
            epoch_avgade3 += batch_avgade3.item() * num_agents3
            epoch_avgfde3 += batch_avgfde3.item() * num_agents3

            epoch_agents += len(future_agents_traj_len)
            epoch_agents2 += num_agents2
            epoch_agents3 += num_agents3


        epoch_loss /= epoch_agents

        # 2-Loss
        epoch_minade2 /= epoch_agents2
        epoch_avgade2 /= epoch_agents2
        epoch_minfde2 /= epoch_agents2
        epoch_avgfde2 /= epoch_agents2

        # 3-Loss
        epoch_minade3 /= epoch_agents3
        epoch_avgade3 /= epoch_agents3
        epoch_minfde3 /= epoch_agents3
        epoch_avgfde3 /= epoch_agents3

        epoch_ades = ( epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3 )
        epoch_fdes = ( epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3 )

        return epoch_loss, epoch_ades, epoch_fdes


    def inference(self):
        self.model.eval()  # Set model to evaluate mode.
        
        with torch.no_grad():
            epoch_loss = 0.0
            epoch_minade2, epoch_avgade2 = 0.0, 0.0
            epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
            epoch_minade3, epoch_avgade3 = 0.0, 0.0
            epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
        
            epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

            H = W = 60
            if '2.' in self.map_version:
                coordinate_2d = np.indices((H, W))
                coordinate = np.ravel_multi_index(coordinate_2d, dims=(H, W))
                coordinate = torch.FloatTensor(coordinate)
                coordinate = coordinate.reshape((1, 1, H, W))

                coordinate_std, coordinate_mean = torch.std_mean(coordinate)
                coordinate = (coordinate - coordinate_mean) / coordinate_std

                distance_2d = coordinate_2d - np.array([H/2 - 0.5, W/2 - 0.5]).reshape((2, 1, 1))
                distance = np.sqrt((distance_2d ** 2).sum(axis=0))
                distance = torch.FloatTensor(distance)
                distance = distance.reshape((1, 1, H, W))

                distance_std, distance_mean = torch.std_mean(distance)
                distance = (distance - distance_mean) / distance_std
                
                coordinate = coordinate.to(self.device)
                distance = distance.to(self.device)

            for b, batch in enumerate(self.valid_loader):

                print("Working on batch {:d}/{:d}".format(b+1, len(self.valid_loader)), end='\r')

                scene_images, log_prior, \
                future_agent_masks, \
                num_past_agents,   past_agents_traj,   past_agents_traj_len,   past_agents_traj_len_idx, \
                num_future_agents, future_agents_traj, future_agents_traj_len, future_agents_traj_len_idx, \
                two_mask, three_mask, \
                decode_start_vel, decode_start_pos, \
                scene_id = batch
                
                # Detect dynamic batch size
                batch_size = scene_images.size(0)
                num_agents = future_agents_traj.size(0)

                if '2.' in self.map_version:
                    coordinate_batch = coordinate.expand(batch_size, -1, -1, -1)
                    distance_batch = distance.expand(batch_size, -1, -1, -1)
                    scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)

                elif self.map_version == '1.3':
                    scene_images = scene_images.to(self.device)

                # Upload to GPU
                num_past_agents = num_past_agents.to(self.device)
                past_agents_traj = past_agents_traj.to(self.device)
                past_agents_traj_len = past_agents_traj_len.to(self.device)

                future_agents_traj = future_agents_traj.to(self.device)
                future_agents_traj_len = future_agents_traj_len.to(self.device)
                future_agents_traj_len_idx = future_agents_traj_len_idx.to(self.device)

                two_mask = two_mask.to(self.device)
                three_mask = three_mask.to(self.device)

                num_future_agents = num_future_agents.to(self.device)
                decode_start_vel = decode_start_vel.to(self.device)
                decode_start_pos = decode_start_pos.to(self.device)
                
                episode_idx = torch.arange(len(num_past_agents), device=self.device).repeat_interleave(num_past_agents)

                # Inference
                if 'SimpleEncDec' == self.model_type:
                    predicted_trajs = self.model(past_agents_traj, past_agents_traj_len, future_agent_masks,
                                                decode_start_vel, decode_start_pos)

                elif 'SocialPooling' == self.model_type:
                    predicted_trajs = self.model(past_agents_traj, past_agents_traj_len, episode_idx, future_agent_masks,
                                                decode_start_vel, decode_start_pos)

                elif 'MATF' in self.model_type:
                    stochastic = False
                    predicted_trajs = self.model(past_agents_traj, past_agents_traj_len, episode_idx, future_agent_masks,
                                                decode_start_vel, decode_start_pos, 
                                                scene_images, stochastic)

                else:
                    raise ValueError("Unknown model type {:s}.".format(self.model_type))

                agent_time_index = torch.arange(num_agents, device=self.device).repeat_interleave(future_agents_traj_len)
                time_normalizer = future_agents_traj_len.float().repeat_interleave(future_agents_traj_len)

                error = future_agents_traj - predicted_trajs # A x Td x 2
                batch_loss = (error ** 2).sum(dim=-1) # x**2 + y**2 | A x Td 
                batch_loss = batch_loss[agent_time_index, future_agents_traj_len_idx] / time_normalizer
                batch_loss = batch_loss.sum() / (num_agents * 2.0)

                sq_error2 = (error[two_mask, :int(self.decoding_steps*2/3), :] ** 2).sum(2).sqrt()
                sq_error3 = (error[three_mask,:, :] ** 2).sum(2).sqrt()

                num_agents2 = sq_error2.size(0)
                num_agents3 = sq_error3.size(0)

                ade2 = sq_error2.mean(dim=-1) # A X T >> A
                fde2 = sq_error2[... , -1]
                ade3 = sq_error3.mean(dim=-1)
                fde3 = sq_error3[... , -1]
                
                # 2-Loss
                batch_minade2 = ade2.mean() # A >> 1
                batch_minfde2 = fde2.mean()
                batch_avgade2 = ade2.mean()
                batch_avgfde2 = fde2.mean()
                
                # 3-Loss
                batch_minade3 = ade3.mean()
                batch_minfde3 = fde3.mean()
                batch_avgade3 = ade3.mean()
                batch_avgfde3 = fde3.mean()

                epoch_loss += batch_loss.item()

                epoch_minade2 += batch_minade2.item() * num_agents2
                epoch_avgade2 += batch_avgade2.item() * num_agents2
                epoch_minfde2 += batch_minfde2.item() * num_agents2
                epoch_avgfde2 += batch_avgfde2.item() * num_agents2

                epoch_minade3 += batch_minade3.item() * num_agents3
                epoch_avgade3 += batch_avgade3.item() * num_agents3
                epoch_minfde3 += batch_minfde3.item() * num_agents3
                epoch_avgfde3 += batch_avgfde3.item() * num_agents3

                epoch_agents += len(future_agents_traj_len)
                epoch_agents2 += num_agents2
                epoch_agents3 += num_agents3


        epoch_minade2 /= epoch_agents2
        epoch_avgade2 /= epoch_agents2
        epoch_minfde2 /= epoch_agents2
        epoch_avgfde2 /= epoch_agents2

        epoch_minade3 /= epoch_agents3
        epoch_avgade3 /= epoch_agents3
        epoch_minfde3 /= epoch_agents3
        epoch_avgfde3 /= epoch_agents3

        epoch_ades = ( epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3 )
        epoch_fdes = ( epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3 )

        scheduler_metric = epoch_avgade3 + epoch_avgfde3 

        return epoch_ades, epoch_fdes, scheduler_metric


    def get_lr(self):
        """Returns Learning Rate of the Optimizer."""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def save_checkpoint(self, epoch, ade, fde):
        """Saves experiment checkpoint.
        Saved state consits of epoch, model state, optimizer state, current
        learning rate and experiment path.
        """

        state_dict = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'learning_rate': self.get_lr(),
            'exp_path': self.exp_path,
            'val_minade3': ade,
            'val_minfde3': fde
        }

        save_path = "{}/ck_{}_{:0.4f}_{:0.4f}.pth.tar".format(self.exp_path, epoch, ade, fde)
        torch.save(state_dict, save_path)

    def load_checkpoint(self, ckpt):
        self.logger.info(f"Loading checkpoint from {ckpt}")
        checkpoint = torch.load(ckpt)
        result = self.model.load_state_dict(checkpoint['model_state'], strict=False)
        self.logger.info(result)
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.start_epoch = checkpoint['epoch']


class ModelTest:
    def __init__(self, model, data_loader, args, device, ploss_criterion=None):
        self.model = model
        self.data_loader = data_loader

        self.num_candidates = args.num_candidates

        self.decoding_steps = int(3 *  args.sampling_rate)

        self.model_type = args.model_type
        
        if self.model_type == 'MATF':
            self.map_version = args.map_version
        else:
            self.map_version = ''

        self.device = device
        self.out_dir = args.test_dir
        self.render = args.test_render

        self.test_times = args.test_times
                
        if args.dataset == "argoverse":
            _data_dir = './data/argoverse'
            self.map_file = lambda scene_id: [os.path.join(_data_dir, x[0], x[1], x[2], 'map/v1.3', x[3] ) + '.png' for x in scene_id]

        elif args.dataset == "nuscenes":
            _data_dir = './data/nuscenes'
            self.map_file = lambda scene_id: [os.path.join(_data_dir, x[0], x[1], x[2], 'map/v1.3', x[3]) + '.pkl' for x in scene_id]

        elif args.dataset == "carla":
            _data_dir = './data/carla'
            self.map_file = lambda scene_id: [os.path.join(_data_dir, x[0], x[1], x[2], 'map/v1.3', x[3] ) + '.pkl' for x in scene_id]

        self.load_checkpoint(args.test_ckpt)


    def load_checkpoint(self, ckpt):
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state'], strict=False)

    def run(self):
        print('Starting model test.....')    
        self.model.eval()  # Set model to evaluate mode.

        list_minade2, list_avgade2 = [], []
        list_minfde2, list_avgfde2 = [], []
        list_minade3, list_avgade3 = [], []
        list_minfde3, list_avgfde3 = [], []

        list_dao = []
        list_dac = []

        for test_time in range(self.test_times):
            epoch_minade2, epoch_avgade2 = 0.0, 0.0
            epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
            epoch_minade3, epoch_avgade3 = 0.0, 0.0
            epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
        
            epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

            epoch_dao = 0.0
            epoch_dac = 0.0
            dao_agents = 0.0
            dac_agents = 0.0

            H = W = 60
            with torch.no_grad():
                if '2.' in self.map_version:
                    coordinate_2d = np.indices((H, W))
                    coordinate = np.ravel_multi_index(coordinate_2d, dims=(H, W))
                    coordinate = torch.FloatTensor(coordinate)
                    coordinate = coordinate.reshape((1, 1, H, W))

                    coordinate_std, coordinate_mean = torch.std_mean(coordinate)
                    coordinate = (coordinate - coordinate_mean) / coordinate_std

                    distance_2d = coordinate_2d - np.array([H/2 - 0.5, W/2 - 0.5]).reshape((2, 1, 1))
                    distance = np.sqrt((distance_2d ** 2).sum(axis=0))
                    distance = torch.FloatTensor(distance)
                    distance = distance.reshape((1, 1, H, W))

                    distance_std, distance_mean = torch.std_mean(distance)
                    distance = (distance - distance_mean) / distance_std
                    
                    coordinate = coordinate.to(self.device)
                    distance = distance.to(self.device)

                for b, batch in enumerate(self.data_loader):

                    scene_images, log_prior, \
                    future_agent_masks, \
                    num_past_agents,   past_agents_traj,   past_agents_traj_len,   past_agents_traj_len_idx, \
                    num_future_agents, future_agents_traj, future_agents_traj_len, future_agents_traj_len_idx, \
                    two_mask, three_mask, \
                    decode_start_vel, decode_start_pos, \
                    scene_id = batch
                    
                    # Detect dynamic batch size
                    batch_size = scene_images.size(0)
                    num_agents = future_agents_traj.size(0)

                    if '2.' in self.map_version:
                        coordinate_batch = coordinate.expand(batch_size, -1, -1, -1)
                        distance_batch = distance.expand(batch_size, -1, -1, -1)
                        scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)

                    elif self.map_version == '1.3':
                        scene_images = scene_images.to(self.device)


                    # Upload to GPU
                    num_past_agents = num_past_agents.to(self.device)
                    past_agents_traj = past_agents_traj.to(self.device)
                    past_agents_traj_len = past_agents_traj_len.to(self.device)

                    future_agents_traj = future_agents_traj.to(self.device)
                    future_agents_traj_len = future_agents_traj_len.to(self.device)
                    future_agents_traj_len_idx = future_agents_traj_len_idx.to(self.device)

                    two_mask = two_mask.to(self.device)
                    three_mask = three_mask.to(self.device)

                    num_future_agents = num_future_agents.to(self.device)
                    decode_start_vel = decode_start_vel.to(self.device)
                    decode_start_pos = decode_start_pos.to(self.device)

                    episode_idx = torch.arange(len(num_past_agents), device=self.device).repeat_interleave(num_past_agents)

                    # Inference
                    if 'SimpleEncDec' == self.model_type:
                        predicted_trajs = self.model(past_agents_traj, past_agents_traj_len, future_agent_masks,
                                                    decode_start_vel, decode_start_pos)

                    elif 'SocialPooling' == self.model_type:
                        predicted_trajs = self.model(past_agents_traj, past_agents_traj_len, episode_idx, future_agent_masks,
                                                    decode_start_vel, decode_start_pos)

                    elif 'MATF' in self.model_type:
                        stochastic = False
                        predicted_trajs = self.model(past_agents_traj, past_agents_traj_len, episode_idx, future_agent_masks,
                                                    decode_start_vel, decode_start_pos,
                                                    scene_images, stochastic)
                    else:
                        raise ValueError("Unknown model type {:s}.".format(self.model_type))
                    
                    ## Deterministic model returns as: Na x Td x C

                    agent_time_index = torch.arange(num_agents, device=self.device).repeat_interleave(future_agents_traj_len)
                    time_normalizer = future_agents_traj_len.float().repeat_interleave(future_agents_traj_len)

                    error = future_agents_traj - predicted_trajs # A x Td x 2

                    sq_error2 = (error[two_mask, :int(self.decoding_steps*2/3), :] ** 2).sum(2).sqrt()
                    sq_error3 = (error[three_mask,:, :] ** 2).sum(2).sqrt()

                    num_agents2 = sq_error2.size(0)
                    num_agents3 = sq_error3.size(0)

                    ade2 = sq_error2.mean(dim=-1) # A X T >> A
                    fde2 = sq_error2[... , -1]
                    ade3 = sq_error3.mean(dim=-1)
                    fde3 = sq_error3[... , -1]
                    
                    # 2-Loss
                    batch_minade2 = ade2.mean() # A >> 1
                    batch_minfde2 = fde2.mean()
                    batch_avgade2 = ade2.mean()
                    batch_avgfde2 = fde2.mean()
                    
                    # 3-Loss
                    batch_minade3 = ade3.mean()
                    batch_minfde3 = fde3.mean()
                    batch_avgade3 = ade3.mean()
                    batch_avgfde3 = fde3.mean()

                    print("Working on test {:d}/{:d}, batch {:d}/{:d}... ".format(test_time+1, self.test_times, b+1, len(self.data_loader)), end='\r')

                    epoch_minade2 += batch_minade2.item() * num_agents2
                    epoch_avgade2 += batch_avgade2.item() * num_agents2
                    epoch_minfde2 += batch_minfde2.item() * num_agents2
                    epoch_avgfde2 += batch_avgfde2.item() * num_agents2

                    epoch_minade3 += batch_minade3.item() * num_agents3
                    epoch_avgade3 += batch_avgade3.item() * num_agents3
                    epoch_minfde3 += batch_minfde3.item() * num_agents3
                    epoch_avgfde3 += batch_avgfde3.item() * num_agents3

                    epoch_agents += len(future_agents_traj_len)
                    epoch_agents2 += num_agents2
                    epoch_agents3 += num_agents3

                    map_files = self.map_file(scene_id)
                    output_files = [self.out_dir + '/' + x[2] + '.jpg' for x in scene_id]

                    cum_num_future_agents = [0] + torch.cumsum(num_future_agents, dim=0).tolist()

                    predicted_trajs = predicted_trajs.unsqueeze(1).cpu().numpy()

                    past_agents_traj = past_agents_traj.cpu().numpy()
                    past_agents_traj_len = past_agents_traj_len.cpu().numpy()

                    future_agents_traj = future_agents_traj.cpu().numpy()
                    future_agents_traj_len = future_agents_traj_len.cpu().numpy()

                    for i in range(batch_size):
                        candidate_i   = predicted_trajs[       cum_num_future_agents[i]:cum_num_future_agents[i+1]]
                        future_traj_i = future_agents_traj[    cum_num_future_agents[i]:cum_num_future_agents[i+1]]
                        future_lens_i = future_agents_traj_len[cum_num_future_agents[i]:cum_num_future_agents[i+1]]
                        past_traj_i   = past_agents_traj[      cum_num_future_agents[i]:cum_num_future_agents[i+1]]
                        past_lens_i   = past_agents_traj_len[  cum_num_future_agents[i]:cum_num_future_agents[i+1]]

                        map_file_i = map_files[i]
                        output_file_i = output_files[i]
                        
                        dao_i, dao_mask_i = self.dao(candidate_i, map_file_i)
                        dac_i, dac_mask_i = self.dac(candidate_i, map_file_i)

                        epoch_dao += dao_i.sum()
                        dao_agents += dao_mask_i.sum()

                        epoch_dac += dac_i.sum()
                        dac_agents += dac_mask_i.sum()

            # 2-Loss
            list_minade2.append(epoch_minade2 / epoch_agents2)
            list_avgade2.append(epoch_avgade2 / epoch_agents2)
            list_minfde2.append(epoch_minfde2 / epoch_agents2)
            list_avgfde2.append(epoch_avgfde2 / epoch_agents2)

            # 3-Loss
            list_minade3.append(epoch_minade3 / epoch_agents3)
            list_avgade3.append(epoch_avgade3 / epoch_agents3)
            list_minfde3.append(epoch_minfde3 / epoch_agents3)
            list_avgfde3.append(epoch_avgfde3 / epoch_agents3)

            list_dao.append(epoch_dao / dao_agents)
            list_dac.append(epoch_dac / dac_agents)
        
        test_minade2 = [np.mean(list_minade2), np.std(list_minade2)]
        test_avgade2 = [np.mean(list_avgade2), np.std(list_avgade2)]
        test_minfde2 = [np.mean(list_minfde2), np.std(list_minfde2)]
        test_avgfde2 = [np.mean(list_avgfde2), np.std(list_avgfde2)]

        test_minade3 = [np.mean(list_minade3), np.std(list_minade3)]
        test_avgade3 = [np.mean(list_avgade3), np.std(list_avgade3)]
        test_minfde3 = [np.mean(list_minfde3), np.std(list_minfde3)]
        test_avgfde3 = [np.mean(list_avgfde3), np.std(list_avgfde3)]

        test_dao = [np.mean(list_dao), np.std(list_dao)]
        test_dac = [np.mean(list_dac), np.std(list_dac)]

        test_ades = ( test_minade2, test_avgade2, test_minade3, test_avgade3 )
        test_fdes = ( test_minfde2, test_avgfde2, test_minfde3, test_avgfde3 )

        print("--Final Performane Report--")
        print("minADE3: {:.5f}±{:.5f}, minFDE3: {:.5f}±{:.5f}".format(test_minade3[0], test_minade3[1], test_minfde3[0], test_minfde3[1]))
        print("avgADE3: {:.5f}±{:.5f}, avgFDE3: {:.5f}±{:.5f}".format(test_avgade3[0], test_avgade3[1], test_avgfde3[0], test_avgfde3[1]))
        print("DAO: {:.5f}±{:.5f}, DAC: {:.5f}±{:.5f}".format(test_dao[0] * 10000.0, test_dao[1] * 10000.0, test_dac[0], test_dac[1]))
        with open(self.out_dir + '/metric.pkl', 'wb') as f:
            pkl.dump({"ADEs": test_ades,
                      "FDEs": test_fdes,
                      "DAO": test_dao,
                      "DAC": test_dac}, f)

    @staticmethod
    def dac(gen_trajs, map_file):
        if '.png' in map_file:
            map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)

        elif '.pkl' in map_file:
            with open(map_file, 'rb') as pnt:
                map_array = pkl.load(pnt)

        da_mask = np.any(map_array > 0, axis=-1)

        num_agents, num_candidates, decoding_timesteps = gen_trajs.shape[:3]
        dac = []

        gen_trajs = ((gen_trajs + 56) * 2).astype(np.int64)

        stay_in_da_count = [0 for i in range(num_agents)]
        for k in range(num_candidates):
            gen_trajs_k = gen_trajs[:, k]

            stay_in_da = [True for i in range(num_agents)]

            oom_mask = np.any( np.logical_or(gen_trajs_k >= 224, gen_trajs_k < 0), axis=-1 )
            diregard_mask = oom_mask.sum(axis=-1) > 2
            for t in range(decoding_timesteps):
                gen_trajs_kt = gen_trajs_k[:, t]
                oom_mask_t = oom_mask[:, t]
                x, y = gen_trajs_kt.T

                lin_xy = (x*224+y)
                lin_xy[oom_mask_t] = -1
                for i in range(num_agents):
                    xi, yi = x[i], y[i]
                    _lin_xy = lin_xy.tolist()
                    lin_xyi = _lin_xy.pop(i)

                    if diregard_mask[i]:
                        continue

                    if oom_mask_t[i]:
                        continue

                    if not da_mask[yi, xi] or (lin_xyi in _lin_xy):
                        stay_in_da[i] = False
            
            for i in range(num_agents):
                if stay_in_da[i]:
                    stay_in_da_count[i] += 1
        
        for i in range(num_agents):
            if diregard_mask[i]:
                dac.append(0.0)
            else:
                dac.append(stay_in_da_count[i] / num_candidates)
        
        dac_mask = np.logical_not(diregard_mask)

        return np.array(dac), dac_mask

    @staticmethod
    def dao(gen_trajs, map_file):
        if '.png' in map_file:
            map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)

        elif '.pkl' in map_file:
            with open(map_file, 'rb') as pnt:
                map_array = pkl.load(pnt)

        da_mask = np.any(map_array > 0, axis=-1)

        num_agents, num_candidates, decoding_timesteps = gen_trajs.shape[:3]
        dao = [0 for i in range(num_agents)]

        occupied = [[] for i in range(num_agents)]

        gen_trajs = ((gen_trajs + 56) * 2).astype(np.int64)

        for k in range(num_candidates):
            gen_trajs_k = gen_trajs[:, k]

            oom_mask = np.any( np.logical_or(gen_trajs_k >= 224, gen_trajs_k < 0), axis=-1 )
            diregard_mask = oom_mask.sum(axis=-1) > 2

            for t in range(decoding_timesteps):
                gen_trajs_kt = gen_trajs_k[:, t]
                oom_mask_t = oom_mask[:, t]
                x, y = gen_trajs_kt.T

                lin_xy = (x*224+y)
                lin_xy[oom_mask_t] = -1
                for i in range(num_agents):
                    xi, yi = x[i], y[i]
                    _lin_xy = lin_xy.tolist()
                    lin_xyi = _lin_xy.pop(i)

                    if diregard_mask[i]:
                        continue

                    if oom_mask_t[i]:
                        continue

                    if lin_xyi in occupied[i]:
                        continue

                    if da_mask[yi, xi] and (lin_xyi not in _lin_xy):
                        occupied[i].append(lin_xyi)
                        dao[i] += 1

        for i in range(num_agents):
            if diregard_mask[i]:
                dao[i] = 0.0
            else:
                dao[i] /= da_mask.sum()

        dao_mask = np.logical_not(diregard_mask)
        
        return np.array(dao), dao_mask

    @staticmethod
    def write_img_output(gen_trajs, src_trajs, src_lens, tgt_trajs, tgt_lens, map_file, output_file):
        """abcd"""
        if '.png' in map_file:
            map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)
            map_array = cv2.cvtColor(map_array, cv2.COLOR_BGR2RGB)

        elif '.pkl' in map_file:
            with open(map_file, 'rb') as pnt:
                map_array = pkl.load(pnt)

        H, W = map_array.shape[:2]
        fig = plt.figure(figsize=(float(H) / float(80), float(W) / float(80)),
                        facecolor='k', dpi=80)

        ax = plt.axes()
        ax.imshow(map_array, extent=[-56, 56, 56, -56])
        ax.set_aspect('equal')
        ax.set_xlim([-56, 56])
        ax.set_ylim([-56, 56])

        plt.gca().invert_yaxis()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                            hspace = 0, wspace = 0)
        plt.margins(0,0)
        
        num_agents, num_candidates = gen_trajs.shape[:2]
        for k in range(num_candidates):
            gen_trajs_k = gen_trajs[:, k]

            x_pts_k = []
            y_pts_k = []
            for i in range(num_agents):
                gen_traj_ki = gen_trajs_k[i]
                tgt_len_i = tgt_lens[i]
                x_pts_k.extend(gen_traj_ki[:tgt_len_i, 0])
                y_pts_k.extend(gen_traj_ki[:tgt_len_i, 1])

            ax.scatter(x_pts_k, y_pts_k, s=0.5, marker='o')

        x_pts = []
        y_pts = []
        for i in range(num_agents):
                src_traj_i = src_trajs[i]
                src_len_i = src_lens[i]
                x_pts.extend(src_traj_i[:src_len_i, 0])
                y_pts.extend(src_traj_i[:src_len_i, 1])

        ax.scatter(x_pts, y_pts, s=0.5, marker='x')

        x_pts = []
        y_pts = []
        for i in range(num_agents):
                tgt_traj_i = tgt_trajs[i]
                tgt_len_i = tgt_lens[i]
                x_pts.extend(tgt_traj_i[:tgt_len_i, 0])
                y_pts.extend(tgt_traj_i[:tgt_len_i, 1])

        ax.scatter(x_pts, y_pts, s=0.5, marker='o')

        fig.canvas.draw()
        buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buffer = buffer.reshape((H, W, 3))

        buffer = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_file, buffer)
        ax.clear()
        plt.close(fig)