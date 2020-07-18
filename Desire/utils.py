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

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from multiprocessing import Pool

import logging

import pdb

def KLDLoss(mean, std):
    var = std ** 2
    KLD = -0.5 * torch.sum(1 + torch.log(var) - mean ** 2 - var, dim=1)
    return KLD

class ModelTrainer:
    def __init__(self, model, train_loader, valid_loader, optimizer_SGM, exp_path, args, device, IOC, optimizer_IOC):

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
        self.ioc = IOC
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.optimizer_SGM = optimizer_SGM
        self.optimizer_IOC = optimizer_IOC

        self.scheduler_SGM = ReduceLROnPlateau(self.optimizer_SGM, factor=(1/2), verbose=True, patience=3)
        self.scheduler_IOC = ReduceLROnPlateau(self.optimizer_IOC, factor=(1/2), verbose=True, patience=3)

        self.device = device

        self.decoding_steps = int(3 * args.sampling_rate)
        self.encoding_steps = int(2 * args.sampling_rate)

        self.map_version = args.map_version

        self.num_candidates = args.num_candidates

        self.start_epoch = args.start_epoch
        
        if args.load_ckpt:
            self.load_checkpoint(args.load_ckpt)

    def train(self, num_epochs):
        self.logger.info('Model Type: '+str(self.model_type))
        self.logger.info('TRAINING .....')

        for epoch in tqdm(range(self.start_epoch, self.start_epoch + num_epochs)):
            self.logger.info("==========================================================================================")
            
            train_loss, train_sgmloss, train_iocloss, train_ades, train_fdes = self.train_single_epoch()
            valid_loss, valid_sgmloss, valid_iocloss, valid_ades, valid_fdes, scheduler_metric = self.inference()
            self.scheduler_SGM.step(scheduler_metric)
            self.scheduler_IOC.step(scheduler_metric)

            train_minade2, train_avgade2, train_bestade2, train_randade2, train_minade3, train_avgade3, train_bestade3, train_randade3 = train_ades
            train_minfde2, train_avgfde2, train_bestfde2, train_randfde2, train_minfde3, train_avgfde3, train_bestfde3, train_randfde3 = train_fdes

            valid_minade2, valid_avgade2, valid_bestade2, valid_randade2, valid_minade3, valid_avgade3, valid_bestade3, valid_randade3 = valid_ades
            valid_minfde2, valid_avgfde2, valid_bestfde2, valid_randfde2, valid_minfde3, valid_avgfde3, valid_bestfde3, valid_randfde3 = valid_fdes

            logging_msg1 = (
                f'| Epoch: {epoch:02} | Train SGMLoss: {train_sgmloss:0.6f} | Train IOCLoss: {train_iocloss:0.6f} '
                f'| Train minADE[2/3]: {train_minade2:0.4f} / {train_minade3:0.4f} | Train minFDE[2/3]: {train_minfde2:0.4f} / {train_minfde3:0.4f} '
                f'| Train avgADE[2/3]: {train_avgade2:0.4f} / {train_avgade3:0.4f} | Train avgFDE[2/3]: {train_avgfde2:0.4f} / {train_avgfde3:0.4f} '
                f'| Train bestADE[2/3]: {train_bestade2:0.4f} / {train_bestade3:0.4f} | Train bestFDE[2/3]: {train_bestfde2:0.4f} / {train_bestfde3:0.4f} '
                f'| Train randADE[2/3]: {train_randade2:0.4f} / {train_randade3:0.4f} | Train randFDE[2/3]: {train_randfde2:0.4f} / {train_randfde3:0.4f}'
            )

            logging_msg2 = (
                f'| Epoch: {epoch:02} | Valid SGMLoss: {valid_sgmloss:0.6f} | Valid IOCLoss: {valid_iocloss:0.6f} '
                f'| Valid minADE[2/3]: {valid_minade2:0.4f} / {valid_minade3:0.4f} | Valid minFDE[2/3]: {valid_minfde2:0.4f} / {valid_minfde3:0.4f} '
                f'| Valid avgADE[2/3]: {valid_avgade2:0.4f} / {valid_avgade3:0.4f} | Valid avgFDE[2/3]: {valid_avgfde2:0.4f} / {valid_avgfde3:0.4f} '
                f'| Valid bestADE[2/3]: {valid_bestade2:0.4f} / {valid_bestade3:0.4f} | Valid bestFDE[2/3]: {valid_bestfde2:0.4f} / {valid_bestfde3:0.4f} '
                f'| Valid randADE[2/3]: {valid_randade2:0.4f} / {valid_randade3:0.4f} | Valid randFDE[2/3]: {valid_randfde2:0.4f} / {valid_randfde3:0.4f} '
                f'| Scheduler Metric: {scheduler_metric:0.4f} | Learning Rate: {self.get_lr():g}\n'
            )

            self.logger.info("------------------------------------------------------------------------------------------")
            self.logger.info(logging_msg1)
            self.logger.info(logging_msg2)

            self.save_checkpoint(epoch, sgmloss=valid_sgmloss, iocloss=valid_iocloss, ade=valid_minade3, fde=valid_minfde3)

            self.writter.add_scalar('data/Train_Loss', train_loss, epoch)
            self.writter.add_scalar('data/Train_SGMLoss', train_sgmloss, epoch)
            self.writter.add_scalar('data/Train_IOCLoss', train_iocloss, epoch)
            self.writter.add_scalar('data/Learning_Rate', self.get_lr(), epoch)

            self.writter.add_scalar('data/Train_minADE2', train_minade2, epoch)
            self.writter.add_scalar('data/Train_minFDE2', train_minfde2, epoch)
            self.writter.add_scalar('data/Train_minADE3', train_minade3, epoch)
            self.writter.add_scalar('data/Train_minFDE3', train_minfde3, epoch)

            self.writter.add_scalar('data/Train_avgADE2', train_avgade2, epoch)
            self.writter.add_scalar('data/Train_avgFDE2', train_avgfde2, epoch)
            self.writter.add_scalar('data/Train_avgADE3', train_avgade3, epoch)
            self.writter.add_scalar('data/Train_avgFDE3', train_avgfde3, epoch)

            self.writter.add_scalar('data/Train_bestADE2', train_bestade2, epoch)
            self.writter.add_scalar('data/Train_bestFDE2', train_bestfde2, epoch)
            self.writter.add_scalar('data/Train_bestADE3', train_bestade3, epoch)
            self.writter.add_scalar('data/Train_bestFDE3', train_bestfde3, epoch)

            self.writter.add_scalar('data/Train_randADE2', train_randade2, epoch)
            self.writter.add_scalar('data/Train_randFDE2', train_randfde2, epoch)
            self.writter.add_scalar('data/Train_randADE3', train_randade3, epoch)
            self.writter.add_scalar('data/Train_randFDE3', train_randfde3, epoch)

            self.writter.add_scalar('data/Scheduler_Metric', scheduler_metric, epoch)

            self.writter.add_scalar('data/Valid_Loss', valid_loss, epoch)
            self.writter.add_scalar('data/Valid_SGMLoss', valid_sgmloss, epoch)
            self.writter.add_scalar('data/Valid_IOCLoss', valid_iocloss, epoch)

            self.writter.add_scalar('data/Valid_minADE2', valid_minade2, epoch)
            self.writter.add_scalar('data/Valid_minFDE2', valid_minfde2, epoch)
            self.writter.add_scalar('data/Valid_minADE3', valid_minade3, epoch)
            self.writter.add_scalar('data/Valid_minFDE3', valid_minfde3, epoch)

            self.writter.add_scalar('data/Valid_avgADE2', valid_avgade2, epoch)
            self.writter.add_scalar('data/Valid_avgFDE2', valid_avgfde2, epoch)
            self.writter.add_scalar('data/Valid_avgADE3', valid_avgade3, epoch)
            self.writter.add_scalar('data/Valid_avgFDE3', valid_avgfde3, epoch)

            self.writter.add_scalar('data/Valid_bestADE2', valid_bestade2, epoch)
            self.writter.add_scalar('data/Valid_bestFDE2', valid_bestfde2, epoch)
            self.writter.add_scalar('data/Valid_bestADE3', valid_bestade3, epoch)
            self.writter.add_scalar('data/Valid_bestFDE3', valid_bestfde3, epoch)

            self.writter.add_scalar('data/Valid_randADE2', valid_randade2, epoch)
            self.writter.add_scalar('data/Valid_randFDE2', valid_randfde2, epoch)
            self.writter.add_scalar('data/Valid_randADE3', valid_randade3, epoch)
            self.writter.add_scalar('data/Valid_randFDE3', valid_randfde3, epoch)
            
        self.writter.close()
        self.logger.info("Training Complete! ")

    def train_single_epoch(self):
        """Trains the model for a single round."""
        self.model.train()
        epoch_sgmloss, epoch_iocloss = 0.0, 0.0
        epoch_minade2, epoch_avgade2 = 0.0, 0.0
        epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
        epoch_bestade2, epoch_bestfde2 = 0.0, 0.0
        epoch_randade2, epoch_randfde2 = 0.0, 0.0

        epoch_minade3, epoch_avgade3 = 0.0, 0.0
        epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
        epoch_bestade3, epoch_bestfde3 = 0.0, 0.0
        epoch_randade3, epoch_randfde3 = 0.0, 0.0
        
        epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

        H = W = 64
        if '2.' in self.map_version:
            with torch.no_grad():
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

        for b, batch in enumerate(self.train_loader):
            self.optimizer_SGM.zero_grad()
            self.optimizer_IOC.zero_grad()
            scene_images, log_prior, \
            agent_masks, \
            num_src_trajs, src_trajs, src_lens, src_len_idx, \
            num_tgt_trajs, tgt_trajs, tgt_lens, tgt_len_idx, \
            tgt_two_mask, tgt_three_mask, \
            decode_start_vel, decode_start_pos, scene_id = batch

            # Detect dynamic batch size
            batch_size = scene_images.size(0)

            if '2.' in self.map_version:
                coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                distance_batch = distance.repeat(batch_size, 1, 1, 1)
                scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)
            
            else:
                scene_images = scene_images.to(self.device)

            src_trajs = src_trajs.to(self.device)
            src_lens = src_lens.to(self.device)

            num_tgt_trajs = num_tgt_trajs.to(self.device)
            tgt_trajs = tgt_trajs.to(self.device)
            tgt_lens = tgt_lens.to(self.device)
            tgt_len_idx = tgt_len_idx.to(self.device)
            tgt_two_mask = tgt_two_mask.to(self.device)
            tgt_three_mask = tgt_three_mask.to(self.device)

            decode_start_pos = decode_start_pos.to(self.device)

            # mask out the non-decoding agents
            src_trajs = src_trajs[agent_masks]
            src_lens = src_lens[agent_masks]
            decode_start_pos = decode_start_pos[agent_masks]

            # zero-center the trajectories
            src_trajs_normalized = src_trajs - decode_start_pos.unsqueeze(1)

            y_rel, y_, Hx, mu, sigma, _ = self.model(src_trajs_normalized,
                                                     src_lens,
                                                     tgt_trajs,
                                                     tgt_lens,
                                                     decode_start_pos)
            
            score, y_delta, _ = self.ioc(y_rel,
                                         y_,
                                         Hx,
                                         scene_images,
                                         num_tgt_trajs)

            y_ = y_.permute(1, 2, 0, 3)
            y_delta = y_delta.permute(1, 2, 0, 3)

            num_agents = y_.size(0)
            num_candidates = y_.size(1)
            
            # SGM KLD Loss
            KLD = KLDLoss(mu[0], sigma[0])
            mKLD = KLD.mean()

            # SGM Recon Loss (Three masked)
            recon = tgt_trajs.detach().unsqueeze(1) - y_ # T x Na x Nc x 2
            SE_recon = (recon ** 2).sum(dim=-1)
            mSE_recon = SE_recon[tgt_three_mask].mean()

            batch_sgmloss = mKLD + mSE_recon
            batch_sgmloss.backward()
            self.optimizer_SGM.step()

            # IOC Cross Entropy Loss
            with torch.no_grad():
                max_error, _ = SE_recon.detach().max(dim=-1)
                log_qdist = F.log_softmax(-max_error, dim=-1)

            pdist = F.softmax(score.sum(dim=0).squeeze(-1), dim=-1)
            MeanCE = -(pdist[tgt_three_mask] * log_qdist[tgt_three_mask]).sum(-1).mean()

            # IOC Regression Error
            gen_trajs = y_.detach() + y_delta
            regress = tgt_trajs.detach().unsqueeze(1) - gen_trajs
            SE_regress = (regress ** 2).sum(dim=-1)
            mSE_regress = SE_regress[tgt_three_mask].mean()

            batch_iocloss = MeanCE + mSE_regress
            batch_iocloss.backward()
            self.optimizer_IOC.step()

            with torch.no_grad():
                batch_loss = batch_iocloss + batch_sgmloss
                best_candidate = pdist.argmax(dim=-1)
                random_candidate = torch.randint_like(best_candidate, num_candidates)

                # ade2 and fde2
                se2 = SE_regress[tgt_two_mask, :, :int(self.decoding_steps*2/3)]
                num_agents2 = se2.size(0)
                de2 = se2.sqrt()
                
                ade2 = de2.mean(dim=-1)
                fde2 = de2[..., -1]

                agent_idx2 = torch.arange(num_agents2, device=self.device)
                best_candidate2 = best_candidate[tgt_two_mask]
                random_candidate2 = random_candidate[tgt_two_mask]

                minade2, _ = ade2.min(dim=-1)
                minfde2, _ = fde2.min(dim=-1)
                avgade2 = ade2.mean(dim=-1)
                avgfde2 = fde2.mean(dim=-1)
                bestade2 = ade2[agent_idx2, best_candidate2]
                bestfde2 = fde2[agent_idx2, best_candidate2]
                randade2 = ade2[agent_idx2, random_candidate2]
                randfde2 = fde2[agent_idx2, random_candidate2]

                batch_minade2 = minade2.mean()
                batch_minfde2 = minfde2.mean()
                batch_avgade2 = avgade2.mean()
                batch_avgfde2 = avgfde2.mean()
                batch_bestade2 = bestade2.mean()
                batch_bestfde2 = bestfde2.mean()
                batch_randade2 = randade2.mean()
                batch_randfde2 = randfde2.mean()

                # ade3 and fde3
                se3 = SE_regress[tgt_three_mask, :, :]
                num_agents3 = se3.size(0)
                de3 = se3.sqrt()
                
                ade3 = de3.mean(dim=-1)
                fde3 = de3[..., -1]

                agent_idx3 = torch.arange(num_agents3, device=self.device)
                best_candidate3 = best_candidate[tgt_three_mask]
                random_candidate3 = random_candidate[tgt_three_mask]

                minade3, _ = ade3.min(dim=-1)
                minfde3, _ = fde3.min(dim=-1)
                avgade3 = ade3.mean(dim=-1)
                avgfde3 = fde3.mean(dim=-1)
                bestade3 = ade3[agent_idx3, best_candidate3]
                bestfde3 = fde3[agent_idx3, best_candidate3]
                randade3 = ade3[agent_idx3, random_candidate3]
                randfde3 = fde3[agent_idx3, random_candidate3]

                batch_minade3 = minade3.mean()
                batch_minfde3 = minfde3.mean()
                batch_avgade3 = avgade3.mean()
                batch_avgfde3 = avgfde3.mean()
                batch_bestade3 = bestade3.mean()
                batch_bestfde3 = bestfde3.mean()
                batch_randade3 = randade3.mean()
                batch_randfde3 = randfde3.mean()

            print("Working on train batch {:d}/{:d}... ".format(b+1, len(self.train_loader)) +
                  "batch_loss: {:.2f}, sgmloss: {:.2f}, iocloss: {:g}, ".format(batch_loss.item(), batch_sgmloss.item(), batch_iocloss.item()) +
                  "minFDE3: {:.2f}, avgFDE3: {:.2f}, bestFDE3 {:.2f}, randomFDE3 {:.2f}".format(batch_minfde3.item(), batch_avgfde3.item(), batch_bestfde3.item(), batch_randfde3.item()), end='\r')

            epoch_iocloss += batch_iocloss.item() * num_agents
            epoch_sgmloss += batch_sgmloss.item() * num_agents

            epoch_minade2 += batch_minade2.item() * num_agents2
            epoch_avgade2 += batch_avgade2.item() * num_agents2
            epoch_minfde2 += batch_minfde2.item() * num_agents2
            epoch_avgfde2 += batch_avgfde2.item() * num_agents2
            epoch_bestade2 += batch_bestade2.item() * num_agents2
            epoch_bestfde2 += batch_bestfde2.item() * num_agents2
            epoch_randade2 += batch_randade2.item() * num_agents2
            epoch_randfde2 += batch_randfde2.item() * num_agents2

            epoch_minade3 += batch_minade3.item() * num_agents3
            epoch_avgade3 += batch_avgade3.item() * num_agents3
            epoch_minfde3 += batch_minfde3.item() * num_agents3
            epoch_avgfde3 += batch_avgfde3.item() * num_agents3
            epoch_bestade3 += batch_bestade3.item() * num_agents3
            epoch_bestfde3 += batch_bestfde3.item() * num_agents3
            epoch_randade3 += batch_randade3.item() * num_agents3
            epoch_randfde3 += batch_randfde3.item() * num_agents3
            
            epoch_agents += num_agents
            epoch_agents2 += num_agents2
            epoch_agents3 += num_agents3

        epoch_iocloss /= epoch_agents
        epoch_sgmloss /= epoch_agents
        epoch_loss = epoch_sgmloss + epoch_iocloss

        epoch_minade2 /= epoch_agents2
        epoch_avgade2 /= epoch_agents2
        epoch_minfde2 /= epoch_agents2
        epoch_avgfde2 /= epoch_agents2
        epoch_bestade2 /= epoch_agents2
        epoch_bestfde2 /= epoch_agents2
        epoch_randade2 /= epoch_agents2
        epoch_randfde2 /= epoch_agents2

        epoch_minade3 /= epoch_agents3
        epoch_avgade3 /= epoch_agents3
        epoch_minfde3 /= epoch_agents3
        epoch_avgfde3 /= epoch_agents3
        epoch_bestade3 /= epoch_agents3
        epoch_bestfde3 /= epoch_agents3
        epoch_randade3 /= epoch_agents3
        epoch_randfde3 /= epoch_agents3

        epoch_ades = [epoch_minade2, epoch_avgade2, epoch_bestade2, epoch_randade2, epoch_minade3, epoch_avgade3, epoch_bestade3, epoch_randade3]
        epoch_fdes = [epoch_minfde2, epoch_avgfde2, epoch_bestfde2, epoch_randfde2, epoch_minfde3, epoch_avgfde3, epoch_bestfde3, epoch_randfde3]

        self.optimizer_SGM.zero_grad()
        self.optimizer_IOC.zero_grad()

        return epoch_loss, epoch_sgmloss, epoch_iocloss, epoch_ades, epoch_fdes

    def inference(self):
        self.model.eval()  # Set model to evaluate mode.
        epoch_sgmloss, epoch_iocloss = 0.0, 0.0
        epoch_minade2, epoch_avgade2 = 0.0, 0.0
        epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
        epoch_bestade2, epoch_bestfde2 = 0.0, 0.0
        epoch_randade2, epoch_randfde2 = 0.0, 0.0

        epoch_minade3, epoch_avgade3 = 0.0, 0.0
        epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
        epoch_bestade3, epoch_bestfde3 = 0.0, 0.0
        epoch_randade3, epoch_randfde3 = 0.0, 0.0
        
        epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

        H = W = 64
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
            
            for b, batch in enumerate(self.valid_loader):
                # #TODO: Only for testing remove this part!
                # if b > 50:
                #     break
                scene_images, log_prior, \
                agent_masks, \
                num_src_trajs, src_trajs, src_lens, src_len_idx, \
                num_tgt_trajs, tgt_trajs, tgt_lens, tgt_len_idx, \
                tgt_two_mask, tgt_three_mask, \
                decode_start_vel, decode_start_pos, scene_id = batch

                # Detect dynamic batch size
                batch_size = scene_images.size(0)

                if '2.' in self.map_version:
                    coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                    distance_batch = distance.repeat(batch_size, 1, 1, 1)
                    scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)
                else:
                    scene_images = scene_images.to(self.device)
                
                src_trajs = src_trajs.to(self.device)
                src_lens = src_lens.to(self.device)

                num_tgt_trajs = num_tgt_trajs.to(self.device)
                tgt_trajs = tgt_trajs.to(self.device)
                tgt_lens = tgt_lens.to(self.device)
                tgt_len_idx = tgt_len_idx.to(self.device)
                tgt_two_mask = tgt_two_mask.to(self.device)
                tgt_three_mask = tgt_three_mask.to(self.device)
                
                decode_start_pos = decode_start_pos.to(self.device)

                # mask out the non-decoding agents
                src_trajs = src_trajs[agent_masks]
                src_lens = src_lens[agent_masks]
                decode_start_pos = decode_start_pos[agent_masks]

                # zero-center the trajectories
                src_trajs_normalized = src_trajs - decode_start_pos.unsqueeze(1)

                y_rel, y_, Hx, _ = self.model.inference(src_trajs_normalized,
                                                        src_lens,
                                                        decode_start_pos)

            
                score, y_delta, _ = self.ioc(y_rel,
                                             y_,
                                             Hx,
                                             scene_images,
                                             num_tgt_trajs)
                y_ = y_.permute(1, 2, 0, 3)
                y_delta = y_delta.permute(1, 2, 0, 3)

                num_agents = y_.size(0)
                num_candidates = y_.size(1)

                # SGM KLD Loss
                # KLD = KLDLoss(mu[0], sigma[0])
                # mKLD = KLD.mean()
                mKLD = 0.0

                # SGM Recon Loss (Three masked)
                recon = tgt_trajs.detach().unsqueeze(1) - y_ # T x Na x Nc x 2
                SE_recon = (recon ** 2).sum(dim=-1)
                mSE_recon = SE_recon[tgt_three_mask].mean()

                batch_sgmloss = mKLD + mSE_recon

                # IOC Cross Entropy Loss
                max_error, _ = SE_recon.detach().max(dim=-1)
                log_qdist = F.log_softmax(-max_error, dim=-1)

                pdist = F.softmax(score.sum(dim=0).squeeze(-1), dim=-1)
                MeanCE = -(pdist[tgt_three_mask] * log_qdist[tgt_three_mask]).sum(-1).mean()

                # IOC Regression Error
                gen_trajs = y_.detach() + y_delta
                regress = tgt_trajs.detach().unsqueeze(1) - gen_trajs
                SE_regress = (regress ** 2).sum(dim=-1)
                mSE_regress = SE_regress[tgt_three_mask].mean()

                batch_iocloss = MeanCE + mSE_regress

                batch_loss = batch_iocloss + batch_sgmloss
                best_candidate = pdist.argmax(dim=-1)
                random_candidate = torch.randint_like(best_candidate, num_candidates)
                
                # ade2 and fde2
                se2 = SE_regress[tgt_two_mask, :, :int(self.decoding_steps*2/3)]
                num_agents2 = se2.size(0)
                de2 = se2.sqrt()
                
                ade2 = de2.mean(dim=-1)
                fde2 = de2[..., -1]

                agent_idx2 = torch.arange(num_agents2, device=self.device)
                best_candidate2 = best_candidate[tgt_two_mask]
                random_candidate2 = random_candidate[tgt_two_mask]

                minade2, _ = ade2.min(dim=-1)
                minfde2, _ = fde2.min(dim=-1)
                avgade2 = ade2.mean(dim=-1)
                avgfde2 = fde2.mean(dim=-1)
                bestade2 = ade2[agent_idx2, best_candidate2]
                bestfde2 = fde2[agent_idx2, best_candidate2]
                randade2 = ade2[agent_idx2, random_candidate2]
                randfde2 = fde2[agent_idx2, random_candidate2]

                batch_minade2 = minade2.mean()
                batch_minfde2 = minfde2.mean()
                batch_avgade2 = avgade2.mean()
                batch_avgfde2 = avgfde2.mean()
                batch_bestade2 = bestade2.mean()
                batch_bestfde2 = bestfde2.mean()
                batch_randade2 = randade2.mean()
                batch_randfde2 = randfde2.mean()

                # ade3 and fde3
                se3 = SE_regress[tgt_three_mask, :, :]
                num_agents3 = se3.size(0)
                de3 = se3.sqrt()
                
                ade3 = de3.mean(dim=-1)
                fde3 = de3[..., -1]

                agent_idx3 = torch.arange(num_agents3, device=self.device)
                best_candidate3 = best_candidate[tgt_three_mask]
                random_candidate3 = random_candidate[tgt_three_mask]

                minade3, _ = ade3.min(dim=-1)
                minfde3, _ = fde3.min(dim=-1)
                avgade3 = ade3.mean(dim=-1)
                avgfde3 = fde3.mean(dim=-1)
                bestade3 = ade3[agent_idx3, best_candidate3]
                bestfde3 = fde3[agent_idx3, best_candidate3]
                randade3 = ade3[agent_idx3, random_candidate3]
                randfde3 = fde3[agent_idx3, random_candidate3]

                batch_minade3 = minade3.mean()
                batch_minfde3 = minfde3.mean()
                batch_avgade3 = avgade3.mean()
                batch_avgfde3 = avgfde3.mean()
                batch_bestade3 = bestade3.mean()
                batch_bestfde3 = bestfde3.mean()
                batch_randade3 = randade3.mean()
                batch_randfde3 = randfde3.mean()
                
                print("Working on valid batch {:d}/{:d}... ".format(b+1, len(self.valid_loader)) +
                      "batch_loss: {:.2f}, sgmloss: {:.2f}, iocloss: {:g}, ".format(batch_loss.item(), batch_sgmloss.item(), batch_iocloss.item()) +
                      "minFDE3: {:.2f}, avgFDE3: {:.2f}, bestFDE3 {:.2f}, randomFDE3 {:.2f}".format(batch_minfde3.item(), batch_avgfde3.item(), batch_bestfde3.item(), batch_randfde3.item()), end='\r')
            
                epoch_iocloss += batch_iocloss.item() * num_agents
                epoch_sgmloss += batch_sgmloss.item() * num_agents

                epoch_minade2 += batch_minade2.item() * num_agents2
                epoch_avgade2 += batch_avgade2.item() * num_agents2
                epoch_minfde2 += batch_minfde2.item() * num_agents2
                epoch_avgfde2 += batch_avgfde2.item() * num_agents2
                epoch_bestade2 += batch_bestade2.item() * num_agents2
                epoch_bestfde2 += batch_bestfde2.item() * num_agents2
                epoch_randade2 += batch_randade2.item() * num_agents2
                epoch_randfde2 += batch_randfde2.item() * num_agents2

                epoch_minade3 += batch_minade3.item() * num_agents3
                epoch_avgade3 += batch_avgade3.item() * num_agents3
                epoch_minfde3 += batch_minfde3.item() * num_agents3
                epoch_avgfde3 += batch_avgfde3.item() * num_agents3
                epoch_bestade3 += batch_bestade3.item() * num_agents3
                epoch_bestfde3 += batch_bestfde3.item() * num_agents3
                epoch_randade3 += batch_randade3.item() * num_agents3
                epoch_randfde3 += batch_randfde3.item() * num_agents3
                
                epoch_agents += num_agents
                epoch_agents2 += num_agents2
                epoch_agents3 += num_agents3

        epoch_iocloss /= epoch_agents
        epoch_sgmloss /= epoch_agents
        epoch_loss = epoch_sgmloss + epoch_iocloss

        epoch_minade2 /= epoch_agents2
        epoch_avgade2 /= epoch_agents2
        epoch_minfde2 /= epoch_agents2
        epoch_avgfde2 /= epoch_agents2
        epoch_bestade2 /= epoch_agents2
        epoch_bestfde2 /= epoch_agents2
        epoch_randade2 /= epoch_agents2
        epoch_randfde2 /= epoch_agents2

        epoch_minade3 /= epoch_agents3
        epoch_avgade3 /= epoch_agents3
        epoch_minfde3 /= epoch_agents3
        epoch_avgfde3 /= epoch_agents3
        epoch_bestade3 /= epoch_agents3
        epoch_bestfde3 /= epoch_agents3
        epoch_randade3 /= epoch_agents3
        epoch_randfde3 /= epoch_agents3

        epoch_ades = [epoch_minade2, epoch_avgade2, epoch_bestade2, epoch_randade2, epoch_minade3, epoch_avgade3, epoch_bestade3, epoch_randade3]
        epoch_fdes = [epoch_minfde2, epoch_avgfde2, epoch_bestfde2, epoch_randfde2, epoch_minfde3, epoch_avgfde3, epoch_bestfde3, epoch_randfde3]

        scheduler_metric = epoch_loss

        return epoch_loss, epoch_sgmloss, epoch_iocloss, epoch_ades, epoch_fdes, scheduler_metric

    def get_lr(self):
        """Returns Learning Rate of the Optimizer."""
        for param_group in self.optimizer_SGM.param_groups:
            return param_group['lr']

    def save_checkpoint(self, epoch, sgmloss, iocloss, ade, fde):
        """Saves experiment checkpoint.
        Saved state consits of epoch, model state, optimizer state, current
        learning rate and experiment path.
        """

        state_dict = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'sgm_optimizer': self.optimizer_SGM.state_dict(),
            'ioc_optimizer': self.optimizer_IOC.state_dict(),
            'learning_rate': self.get_lr(),
            'exp_path': self.exp_path,
            'val_iocloss': iocloss,
            'val_sgmloss': sgmloss,
            'val_minade3': ade,
            'val_minfde3': fde,
        }
        
        save_path = "{}/ck_{}_{:0.4f}_{:0.4f}_{:0.4f}_{:0.4f}.pth.tar".format(self.exp_path, epoch, sgmloss, iocloss, ade, fde)
        torch.save(state_dict, save_path)

    def load_checkpoint(self, ckpt):
        self.logger.info(f"Loading checkpoint from {ckpt}")
        checkpoint = torch.load(ckpt)
        result1 = self.model.load_state_dict(checkpoint['model_state'], strict=False)
        result2 = self.optimizer_SGM.load_state_dict(checkpoint['sgm_optimizer'])
        result3 = self.optimizer_IOC.load_state_dict(checkpoint['ioc_optimizer'])
        self.start_epoch = checkpoint['epoch']
        self.logger.info(result1)
        self.logger.info(result2)
        self.logger.info(result3)

class ModelTest:
    def __init__(self, model, IOC, data_loader,args, device):
        self.model = model
        self.data_loader = data_loader
        self.ioc = IOC

        if args.model_type == 'Desire':
            self.map_version = args.map_version
        else:
            self.map_version = ''

        self.device = device
        
        self.decoding_steps = 4 if args.dataset == 'carla' else int(3 * args.sampling_rate)
        self.encoding_steps = 2 if args.dataset == 'carla' else int(2 * args.sampling_rate)
        
        self.num_candidates = args.num_candidates

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
        result = self.model.load_state_dict(checkpoint['model_state'], strict=False)
        self.start_epoch = checkpoint['epoch']

    def run(self):
        print('Starting model test.....')
        self.model.eval()

        list_minade2, list_avgade2, list_bestade2, list_randade2 = [], [], [], []
        list_minfde2, list_avgfde2, list_bestfde2, list_randfde2 = [], [], [], []

        list_minade3, list_avgade3, list_bestade3, list_randade3 = [], [], [], []
        list_minfde3, list_avgfde3, list_bestfde3, list_randfde3 = [], [], [], []

        list_dao = []
        list_dac = []

        for test_time in range(self.test_times):

            epoch_minade2, epoch_avgade2 = 0.0, 0.0
            epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
            epoch_bestade2, epoch_bestfde2 = 0.0, 0.0
            epoch_randade2, epoch_randfde2 = 0.0, 0.0

            epoch_minade3, epoch_avgade3 = 0.0, 0.0
            epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
            epoch_bestade3, epoch_bestfde3 = 0.0, 0.0
            epoch_randade3, epoch_randfde3 = 0.0, 0.0

            epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

            epoch_dao = 0.0
            epoch_dac = 0.0
            dao_agents = 0.0
            dac_agents = 0.0

            H = W = 64
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
                    agent_masks, \
                    num_src_trajs, src_trajs, src_lens, src_len_idx, \
                    num_tgt_trajs, tgt_trajs, tgt_lens, tgt_len_idx, \
                    tgt_two_mask, tgt_three_mask, \
                    decode_start_vel, decode_start_pos, scene_id = batch

                    # Detect dynamic batch size
                    batch_size = scene_images.size(0)

                    if '2.' in self.map_version:
                        coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                        distance_batch = distance.repeat(batch_size, 1, 1, 1)
                        scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)
                    else:
                        scene_images = scene_images.to(self.device)
                
                    src_trajs = src_trajs.to(self.device)
                    src_lens = src_lens.to(self.device)

                    num_tgt_trajs = num_tgt_trajs.to(self.device)
                    tgt_trajs = tgt_trajs.to(self.device)
                    tgt_lens = tgt_lens.to(self.device)
                    tgt_len_idx = tgt_len_idx.to(self.device)
                    tgt_two_mask = tgt_two_mask.to(self.device)
                    tgt_three_mask = tgt_three_mask.to(self.device)
                    
                    decode_start_pos = decode_start_pos.to(self.device)

                    # mask out the non-decoding agents
                    src_trajs = src_trajs[agent_masks]
                    src_lens = src_lens[agent_masks]
                    decode_start_pos = decode_start_pos[agent_masks]

                    # zero-center the trajectories
                    src_trajs_normalized = src_trajs - decode_start_pos.unsqueeze(1)

                    y_rel, y_, Hx, _ = self.model.inference(src_trajs_normalized,
                                                            src_lens,
                                                            decode_start_pos)
                    
                    score, y_delta, _ = self.ioc(y_rel,
                                                    y_,
                                                    Hx,
                                                    scene_images,
                                                    num_tgt_trajs)

                    ## y_rel: relative distance between each decoding timesteps [Td x N x Nc x 2]
                    y_ = y_.permute(1, 2, 0, 3)
                    y_delta = y_delta.permute(1, 2, 0, 3)
                    ## y_rel: relative distance between each decoding timesteps [Na x Nc x Td x 2]

                    num_agents = y_.size(0)
                    num_candidates = y_.size(1)

                    # SGM KLD Loss
                    # KLD = KLDLoss(mu[0], sigma[0])
                    # mKLD = KLD.mean()
                    mKLD = 0.0

                    # SGM Recon Loss (Three masked)
                    recon = tgt_trajs.detach().unsqueeze(1) - y_ # T x Na x Nc x 2
                    SE_recon = (recon ** 2).sum(dim=-1)
                    mSE_recon = SE_recon[tgt_three_mask].mean()

                    batch_sgmloss = mKLD + mSE_recon

                    # IOC Cross Entropy Loss
                    max_error, _ = SE_recon.detach().max(dim=-1)
                    log_qdist = F.log_softmax(-max_error, dim=-1)

                    pdist = F.softmax(score.sum(dim=0).squeeze(-1), dim=-1)
                    MeanCE = -(pdist[tgt_three_mask] * log_qdist[tgt_three_mask]).sum(-1).mean()

                    # IOC Regression Error
                    gen_trajs = y_.detach() + y_delta
                    regress = tgt_trajs.detach().unsqueeze(1) - gen_trajs
                    SE_regress = (regress ** 2).sum(dim=-1)
                    mSE_regress = SE_regress[tgt_three_mask].mean()

                    batch_iocloss = MeanCE + mSE_regress

                    batch_loss = batch_iocloss + batch_sgmloss
                    best_candidate = pdist.argmax(dim=-1)

                    random_candidate = torch.randint_like(best_candidate, num_candidates)

                    # ade2 and fde2
                    se2 = SE_regress[tgt_two_mask, :, :int(self.decoding_steps*2/3)]
                    num_agents2 = se2.size(0)
                    de2 = se2.sqrt()
                    
                    ade2 = de2.mean(dim=-1)
                    fde2 = de2[..., -1]

                    agent_idx2 = torch.arange(num_agents2, device=self.device)
                    best_candidate2 = best_candidate[tgt_two_mask]
                    random_candidate2 = random_candidate[tgt_two_mask]

                    minade2, _ = ade2.min(dim=-1)
                    minfde2, _ = fde2.min(dim=-1)
                    avgade2 = ade2.mean(dim=-1)
                    avgfde2 = fde2.mean(dim=-1)
                    bestade2 = ade2[agent_idx2, best_candidate2]
                    bestfde2 = fde2[agent_idx2, best_candidate2]
                    randade2 = ade2[agent_idx2, random_candidate2]
                    randfde2 = fde2[agent_idx2, random_candidate2]

                    batch_minade2 = minade2.mean()
                    batch_minfde2 = minfde2.mean()
                    batch_avgade2 = avgade2.mean()
                    batch_avgfde2 = avgfde2.mean()
                    batch_bestade2 = bestade2.mean()
                    batch_bestfde2 = bestfde2.mean()
                    batch_randade2 = randade2.mean()
                    batch_randfde2 = randfde2.mean()

                    # ade3 and fde3
                    se3 = SE_regress[tgt_three_mask, :, :]
                    num_agents3 = se3.size(0)
                    de3 = se3.sqrt()
                    
                    ade3 = de3.mean(dim=-1)
                    fde3 = de3[..., -1]

                    agent_idx3 = torch.arange(num_agents3, device=self.device)
                    best_candidate3 = best_candidate[tgt_three_mask]
                    random_candidate3 = random_candidate[tgt_three_mask]

                    minade3, _ = ade3.min(dim=-1)
                    minfde3, _ = fde3.min(dim=-1)
                    avgade3 = ade3.mean(dim=-1)
                    avgfde3 = fde3.mean(dim=-1)
                    bestade3 = ade3[agent_idx3, best_candidate3]
                    bestfde3 = fde3[agent_idx3, best_candidate3]
                    randade3 = ade3[agent_idx3, random_candidate3]
                    randfde3 = fde3[agent_idx3, random_candidate3]

                    batch_minade3 = minade3.mean()
                    batch_minfde3 = minfde3.mean()
                    batch_avgade3 = avgade3.mean()
                    batch_avgfde3 = avgfde3.mean()
                    batch_bestade3 = bestade3.mean()
                    batch_bestfde3 = bestfde3.mean()
                    batch_randade3 = randade3.mean()
                    batch_randfde3 = randfde3.mean()

                    print("Working on test {:d}/{:d}, batch {:d}/{:d}... ".format(test_time+1, self.test_times, b+1, len(self.data_loader)), end='\r')

                    epoch_minade2 += batch_minade2.item() * num_agents2
                    epoch_avgade2 += batch_avgade2.item() * num_agents2
                    epoch_minfde2 += batch_minfde2.item() * num_agents2
                    epoch_avgfde2 += batch_avgfde2.item() * num_agents2
                    epoch_bestade2 += batch_bestade2.item() * num_agents2
                    epoch_bestfde2 += batch_bestfde2.item() * num_agents2
                    epoch_randade2 += batch_randade2.item() * num_agents2
                    epoch_randfde2 += batch_randfde2.item() * num_agents2

                    epoch_minade3 += batch_minade3.item() * num_agents3
                    epoch_avgade3 += batch_avgade3.item() * num_agents3
                    epoch_minfde3 += batch_minfde3.item() * num_agents3
                    epoch_avgfde3 += batch_avgfde3.item() * num_agents3
                    epoch_bestade3 += batch_bestade3.item() * num_agents3
                    epoch_bestfde3 += batch_bestfde3.item() * num_agents3
                    epoch_randade3 += batch_randade3.item() * num_agents3
                    epoch_randfde3 += batch_randfde3.item() * num_agents3
                    
                    epoch_agents += num_agents
                    epoch_agents2 += num_agents2
                    epoch_agents3 += num_agents3

                    map_files = self.map_file(scene_id)
                    output_files = [self.out_dir + '/' + x[2] + '.jpg' for x in scene_id]

                    cum_num_tgt_trajs = [0] + torch.cumsum(num_tgt_trajs, dim=0).tolist()

                    tgt_three_mask = tgt_three_mask.cpu().numpy()

                    gen_trajs = gen_trajs.cpu().numpy()

                    src_trajs = src_trajs.cpu().numpy()
                    src_lens = src_lens.cpu().numpy()

                    tgt_trajs = tgt_trajs.cpu().numpy()
                    tgt_lens = tgt_lens.cpu().numpy()

                    zero_ind = np.nonzero(tgt_three_mask == 0)[0]
                    zero_ind -= np.arange(len(zero_ind))
                    
                    gen_trajs = np.insert(gen_trajs, zero_ind, 0, axis=0)
                    tgt_trajs = np.insert(tgt_trajs, zero_ind, 0, axis=0)
                    src_trajs = np.insert(src_trajs, zero_ind, 0, axis=0)

                    for i in range(batch_size):
                        candidate_i = gen_trajs[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]
                        tgt_traj_i = tgt_trajs[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]
                        tgt_lens_i = tgt_lens[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]
                        src_traj_i = src_trajs[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]
                        src_lens_i = src_lens[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]
                        map_file_i = map_files[i]
                        output_file_i = output_files[i]

                        candidate_i = candidate_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]]
                        tgt_traj_i = tgt_traj_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]]
                        src_traj_i = src_traj_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]]
            
                        dao_i, dao_mask_i = self.dao(candidate_i, map_file_i)
                        dac_i, dac_mask_i = self.dac(candidate_i, map_file_i)

                        epoch_dao += dao_i.sum()
                        dao_agents += dao_mask_i.sum()

                        epoch_dac += dac_i.sum()
                        dac_agents += dac_mask_i.sum()

            # 2-Loss
            list_minade2.append(epoch_minade2 / epoch_agents2)
            list_avgade2.append(epoch_avgade2 / epoch_agents2)
            list_bestade2.append(epoch_bestade2 / epoch_agents2)
            list_randade2.append(epoch_randade2 / epoch_agents2)

            list_minfde2.append(epoch_minfde2 / epoch_agents2)
            list_avgfde2.append(epoch_avgfde2 / epoch_agents2)
            list_bestfde2.append(epoch_bestfde2 / epoch_agents2)
            list_randfde2.append(epoch_randfde2 / epoch_agents2)

            # 3-Loss
            list_minade3.append(epoch_minade3 / epoch_agents3)
            list_avgade3.append(epoch_avgade3 / epoch_agents3)
            list_bestade3.append(epoch_bestade3 / epoch_agents3)
            list_randade3.append(epoch_randade3 / epoch_agents3)

            list_minfde3.append(epoch_minfde3 / epoch_agents3)
            list_avgfde3.append(epoch_avgfde3 / epoch_agents3)
            list_bestfde3.append(epoch_bestfde3 / epoch_agents3)
            list_randfde3.append(epoch_randfde3 / epoch_agents3)

            list_dao.append(epoch_dao / dao_agents)
            list_dac.append(epoch_dac / dac_agents)

        test_minade2 = [np.mean(list_minade2), np.std(list_minade2)]
        test_avgade2 = [np.mean(list_avgade2), np.std(list_avgade2)]
        test_bestade2 = [np.mean(list_bestade2), np.std(list_bestade2)]
        test_randade2 = [np.mean(list_randade2), np.std(list_randade2)]

        test_minfde2 = [np.mean(list_minfde2), np.std(list_minfde2)]
        test_avgfde2 = [np.mean(list_avgfde2), np.std(list_avgfde2)]
        test_bestfde2 = [np.mean(list_bestfde2), np.std(list_bestfde2)]
        test_randfde2 = [np.mean(list_randfde2), np.std(list_randfde2)]

        test_minade3 = [np.mean(list_minade3), np.std(list_minade3)]
        test_avgade3 = [np.mean(list_avgade3), np.std(list_avgade3)]
        test_bestade3 = [np.mean(list_bestade3), np.std(list_bestade3)]
        test_randade3 = [np.mean(list_randade3), np.std(list_randade3)]

        test_minfde3 = [np.mean(list_minfde3), np.std(list_minfde3)]
        test_avgfde3 = [np.mean(list_avgfde3), np.std(list_avgfde3)]
        test_bestfde3 = [np.mean(list_bestfde3), np.std(list_bestfde3)]
        test_randfde3 = [np.mean(list_randfde3), np.std(list_randfde3)]

        test_dao = [np.mean(list_dao), np.std(list_dao)]
        test_dac = [np.mean(list_dac), np.std(list_dac)]

        test_ades = ( test_minade2, test_avgade2, test_bestade2, test_randade2, test_minade3, test_avgade3, test_bestade3, test_randade3 )
        test_fdes = ( test_minfde2, test_avgfde2, test_bestfde2, test_randfde2, test_minfde3, test_avgfde3, test_bestfde3, test_randfde3 )

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

        ax.scatter(x_pts, y_pts, s=2.0, marker='x')

        x_pts = []
        y_pts = []
        for i in range(num_agents):
                tgt_traj_i = tgt_trajs[i]
                tgt_len_i = tgt_lens[i]
                x_pts.extend(tgt_traj_i[:tgt_len_i, 0])
                y_pts.extend(tgt_traj_i[:tgt_len_i, 1])

        ax.scatter(x_pts, y_pts, s=2.0, marker='o')

        fig.canvas.draw()
        buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buffer = buffer.reshape((H, W, 3))

        buffer = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_file, buffer)
        ax.clear()
        plt.close(fig)