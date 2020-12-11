
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

checkNone = lambda x, alt: alt if not x else x


def log_determinant(sigma):
    det = sigma[:, :, 0, 0] * sigma[:, :, 1, 1] - sigma[:, :, 0, 1] ** 2
    logdet = torch.log(det + 1e-9)

    return logdet

def KLDLoss(mean, std):
    var = std ** 2
    KLD = -0.5 * torch.sum(1 + torch.log(var) - mean ** 2 - var, dim=1)
    return KLD

class ModelTrainer:

    def __init__(self, model, train_loader, valid_loader, optimizer, exp_path, **kwargs):
        
        # Parameters set
        device = kwargs.get("device", "cpu")
        args = kwargs.get("cfg", None)
        self.cfg = args
        discriminator = kwargs.get("discriminator", None)
        optimizer_d = kwargs.get("optimizer_d", None)
        self.generative = kwargs.get("generative", self.cfg.generative if hasattr(self.cfg, "generative") else None)
        self.ploss_criterion = kwargs.get("ploss_criterion", self.cfg.ploss_criterion if hasattr(self.cfg, "ploss_criterion") else None)
        self.ploss_type = kwargs.get("ploss_criterion", self.cfg.ploss_type)
        self.decoding_steps = int(3 * args.sampling_rate)
        self.encoding_steps = int(2 * args.sampling_rate)
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Timestamp and path set
        self.exp_path = os.path.join(exp_path, str(args.TAG) + '_' + datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-4))).strftime('_%d_%B__%H_%M_'))
        if not os.path.exists(self.exp_path):
            os.mkdir(self.exp_path)

        # Logger
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

        self.device = device
        self.ploss_criterion = self.ploss_criterion().to(self.device) if self.ploss_criterion else None 
        
        # Optimizer
        if isinstance(optimizer, list):
            if self.generative:
                optimizer_d = optimizer[1] # Overwriting
                optimizer = optimizer[0]
        # Model
        if isinstance(model.model, list):
            if self.generative and not discriminator:
                discriminator = model.model[1]
                discriminator = discriminator.to(self.device)
                model = model.model[0]
            elif not self.generative: 
                self.ioc = model.model[1]
                self.ioc.to(self.device)
                model = model.model[0]
        else:
            model= model.model
            
        self.model_name = args.model_name
        self.model = model
        self.model.to(self.device)

        self.discriminator = discriminator
        self.optimizer_D = optimizer_d
        self.optimizer = optimizer

        if isinstance(self.optimizer, list):     
            self.scheduler = [None]*len(self.optimizer)
            for i in range(len(self.optimizer)):
                self.scheduler[i] = ReduceLROnPlateau(self.optimizer[i], factor=(1/2), verbose=True, patience=3)
        else:
            self.scheduler = ReduceLROnPlateau(self.optimizer, factor=(1/2), verbose=True, patience=3)
        
        if hasattr(args, "map_version"):
            if hasattr(args, "use_scene") and args.use_scene:
                self.map_version = args.map_version
            else:
                self.map_version = "1.3"    
        else:
            print("[LOG] Default map version 2.0")
            self.map_version = '2.0'

        if hasattr(args, "beta"):
            self.beta = args.beta 
        
        if args.load_ckpt is not None:
            self.load_checkpoint(args.load_ckpt)

        # Gan training parameters
        self.best_valid_ade = 1e9
        self.best_valid_fde = 1e9
        self.start_epoch = args.start_epoch
        self.gan_weight = args.gan_weight
        self.gan_weight_schedule = args.gan_weight_schedule
        self.adversarial_loss = torch.nn.BCELoss()
        self.num_candidates_train = 1
        self.num_candidates = checkNone(args.num_candidates, None)

        if self.model_name in ["R2P2_SimpleRNN", "R2P2_RNN"] or 'NFDecoder' in self.model_name:
            self.flow_based_decoder = True
            self.num_candidates = args.num_candidates
        else:
            self.flow_based_decoder = False
            self.num_candidates = 1

    def train(self, num_epochs):
        self.logger.info('Model Type: '+str(self.model_name))
        self.logger.info('TRAINING .....')

        for epoch in tqdm(range(self.start_epoch, self.start_epoch + num_epochs)):
            self.logger.info("==========================================================================================")

            if self.model_name in ["SimpleEncoderDecoder", "SocialPooling", "MATF"]:
                train_loss, train_ades, train_fdes = self.train_single_epoch_matf()
                valid_ades, valid_fdes, scheduler_metric = self.inference_matf()
            elif self.model_name  in ["MATF_GAN"]:
                train_loss, train_qloss, train_ploss, train_ades, train_fdes, scheduler_metric = self.train_single_epoch_gan(epoch)
                valid_loss, valid_g_loss, valid_d_loss, valid_ades, valid_fdes, scheduler_metric = self.inference_gan()
            elif self.model_name  in ["R2P2_SimpleRNN", "R2P2_RNN"]:
                train_loss, train_qloss, train_ploss, train_ades, train_fdes = self.train_single_epoch_r2p2()
                valid_loss, valid_qloss, valid_ploss, valid_ades, valid_fdes, scheduler_metric = self.inference_r2p2()
            elif self.model_name.lower() in ["desire"]:
                train_loss, train_sgmloss, train_iocloss, train_ades, train_fdes = self.train_single_epoch_desire()
                valid_loss, valid_sgmloss, valid_iocloss, valid_ades, valid_fdes, scheduler_metric = self.inference_desire()
            elif self.model_name in ["CAM", "CAM_NFDecoder", "Scene_CAM_NFDecoder", "Global_Scene_CAM_NFDecoder", "AttGlobal_Scene_CAM_NFDecoder", "Scene_CAM_NFDecoder"]:
                train_loss, train_qloss, train_ploss, train_ades, train_fdes = self.train_single_epoch_proposed()
                valid_loss, valid_qloss, valid_ploss, valid_ades, valid_fdes, scheduler_metric = self.inference_propsed()
            if self.model_name.lower() in ["desire"]:
                train_minade2, train_avgade2, train_bestade2, train_randade2, train_minade3, train_avgade3, train_bestade3, train_randade3 = train_ades
                train_minfde2, train_avgfde2, train_bestfde2, train_randfde2, train_minfde3, train_avgfde3, train_bestfde3, train_randfde3 = train_fdes

                valid_minade2, valid_avgade2, valid_bestade2, valid_randade2, valid_minade3, valid_avgade3, valid_bestade3, valid_randade3 = valid_ades
                valid_minfde2, valid_avgfde2, valid_bestfde2, valid_randfde2, valid_minfde3, valid_avgfde3, valid_bestfde3, valid_randfde3 = valid_fdes
            else:
                ## unwrapping ADEs/FDEs
                train_minade2, train_avgade2, train_minade3, train_avgade3 = train_ades
                train_minfde2, train_avgfde2, train_minfde3, train_avgfde3 = train_fdes

                valid_minade2, valid_avgade2, valid_minade3, valid_avgade3 = valid_ades
                valid_minfde2, valid_avgfde2, valid_minfde3, valid_avgfde3 = valid_fdes

            if self.flow_based_decoder:
                self.best_valid_ade = min(valid_avgade3, self.best_valid_ade)
                self.best_valid_fde = min(valid_avgfde3, self.best_valid_fde)
            else:
                self.best_valid_ade = min(*valid_ades, self.best_valid_ade)
                self.best_valid_fde = min(*valid_fdes, self.best_valid_fde)
            
            if isinstance(self.scheduler, list):
                for scheduler in self.scheduler:
                    scheduler.step(scheduler_metric)
            else:
                self.scheduler.step(scheduler_metric)

            if self.model_name.lower() in ["desire"]:
                # For longer message, it would be better to use this kind of thing.
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
                    f'| Scheduler Metric: {scheduler_metric:0.4f} | Learning Rate: {self.get_desire_lr():g}\n'
                )
            else:
                # For longer message, it would be better to use this kind of thing.
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

            
            if self.flow_based_decoder:
                self.save_checkpoint_r2p2(epoch, qloss=valid_qloss, ploss=valid_ploss, ade=valid_minade3, fde=valid_minfde3)
            else:
                self.save_checkpoint(epoch, ade=valid_avgade3, fde=valid_avgfde3)

            # Log values to Tensorboard

            if self.flow_based_decoder:    
                self.writter.add_scalar('data/Train_QLoss', train_qloss, epoch)
                self.writter.add_scalar('data/Train_PLoss', train_ploss, epoch)

            self.writter.add_scalar('data/Train_Loss', train_loss, epoch)

            if self.model_name.lower() in ["desire"]:
                self.writter.add_scalar('data/Learning_Rate', self.get_desire_lr(), epoch)
            else:
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

            if self.flow_based_decoder:
                self.writter.add_scalar('data/Valid_Loss', valid_loss, epoch)
                self.writter.add_scalar('data/Valid_QLoss', valid_qloss, epoch)
                self.writter.add_scalar('data/Valid_PLoss', valid_ploss, epoch)

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


    # Desire Train and inference 
    def train_single_epoch_desire(self):
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
            for i in range(len(self.optimizer)):
                self.optimizer[i].zero_grad()
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
            # self.optimizer_SGM.step()

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
            # self.optimizer_IOC.step()
            for i in range(len(self.optimizer)):
                self.optimizer[i].step()

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

        for i in range(len(self.optimizer)):
            self.optimizer[i].zero_grad()

        return epoch_loss, epoch_sgmloss, epoch_iocloss, epoch_ades, epoch_fdes

    def inference_desire(self):
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

    # MATF train and inference
    def train_single_epoch_matf(self):
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
            if 'SimpleEncoderDecoder' == self.model_name:
                predicted_trajs = self.model(past_agents_traj, past_agents_traj_len, future_agent_masks,
                                            decode_start_vel, decode_start_pos)

            elif 'SocialPooling' == self.model_name:
                predicted_trajs = self.model(past_agents_traj, past_agents_traj_len, episode_idx, future_agent_masks,
                                            decode_start_vel, decode_start_pos)             

            elif 'MATF' in self.model_name:
                stochastic = False                
                predicted_trajs = self.model(past_agents_traj, past_agents_traj_len, episode_idx, future_agent_masks,
                                            decode_start_vel, decode_start_pos, 
                                            scene_images, stochastic)

            else:
                raise ValueError("Unknown model type {:s}.".format(self.model_name))

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

                ## ## TODO check the reshape purpose once again, not sure about this part.
                ## sq_error = sq_error.reshape((-1))

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

            ## TODO check the reason why here is future while in GAN version if accepts past.
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

    def inference_matf(self):
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
                if 'SimpleEncoderDecoder' == self.model_name:
                    predicted_trajs = self.model(past_agents_traj, past_agents_traj_len, future_agent_masks,
                                                decode_start_vel, decode_start_pos)

                elif 'SocialPooling' == self.model_name:
                    predicted_trajs = self.model(past_agents_traj, past_agents_traj_len, episode_idx, future_agent_masks,
                                                decode_start_vel, decode_start_pos)

                elif 'MATF' in self.model_name:
                    stochastic = False
                    predicted_trajs = self.model(past_agents_traj, past_agents_traj_len, episode_idx, future_agent_masks,
                                                decode_start_vel, decode_start_pos, 
                                                scene_images, stochastic)

                else:
                    raise ValueError("Unknown model type {:s}.".format(self.model_name))

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

        # scheduler_metric = epoch_loss
        scheduler_metric = epoch_avgade3 + epoch_avgfde3 

        return epoch_ades, epoch_fdes, scheduler_metric

    # MATF GaN training and inference
    def train_single_epoch_gan(self, epoch):
        """Trains discriminator and generator for a single round."""

        self.model.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
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

        for i, e in enumerate(self.gan_weight_schedule):
            if epoch <= e:
                gan_weight = self.gan_weight[i]
                break

        for b, batch in enumerate(self.train_loader):
            print("Working on batch {:d}/{:d}".format(b+1, len(self.train_loader)), end='\r')
            self.optimizer.zero_grad()
            self.optimizer_D.zero_grad()

            scene_images, log_prior, \
            future_agent_masks, \
            num_past_agents,   past_agents_traj,   past_agents_traj_len,   past_agents_traj_len_idx, \
            num_future_agents, future_agents_traj, future_agents_traj_len, future_agents_traj_len_idx, \
            two_mask, three_mask, \
            decode_start_vel, decode_start_pos, \
            scene_id = batch

            # Detect dynamic batch size
            batch_size = scene_images.size(0)
            total_past_agent = past_agents_traj.size(0)
            total_future_agent = future_agents_traj.size(0)

            if '2.' in self.map_version:
                coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                distance_batch = distance.repeat(batch_size, 1, 1, 1)
                scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)

            elif self.map_version == '1.3':
                scene_images = scene_images.to(self.device)

            num_past_agents = num_past_agents.to(self.device)
            past_agents_traj = past_agents_traj.to(self.device)
            past_agents_traj_len = past_agents_traj_len.to(self.device)
            past_agents_traj_len_idx = past_agents_traj_len_idx.to(self.device)

            future_agent_masks = future_agent_masks.to(self.device)
            future_agents_traj = future_agents_traj.to(self.device)
            future_agents_traj_len = future_agents_traj_len.to(self.device)
            future_agents_traj_len_idx = future_agents_traj_len_idx.to(self.device)

            two_mask = two_mask.to(self.device)
            three_mask = three_mask.to(self.device)

            num_future_agents = num_future_agents.to(self.device)
            decode_start_vel = decode_start_vel.to(self.device)
            decode_start_pos = decode_start_pos.to(self.device)

            episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_past_agents)

            # Generator
            if 'GAN' in self.model_name:
                    stochastic = True
                    predicted_trajs = self.model(past_agents_traj, past_agents_traj_len, episode_idx, future_agent_masks,
                                                decode_start_vel, decode_start_pos, scene_images, stochastic, self.num_candidates_train)
                    
                    predicted_trajs_ = predicted_trajs.reshape(total_future_agent, self.num_candidates_train, self.decoding_steps, 2)
            else:
                raise ValueError("Unknown model type {:s}.".format(self.model_name))
            
            # Length for the concatnated trajectory
            agents_total_traj_len = past_agents_traj_len.clone()
            agents_total_traj_len[future_agent_masks] += future_agents_traj_len
            agents_total_traj_len_ = agents_total_traj_len.repeat_interleave(self.num_candidates_train)

            # Fill past trajectories
            past_agent_idx = torch.arange(total_past_agent, device=self.device)
            past_agent_idx_repeat = past_agent_idx.repeat_interleave(past_agents_traj_len)
            real_trajs = torch.zeros((total_past_agent, self.encoding_steps+self.decoding_steps, 2), device=self.device)
            real_trajs[past_agent_idx_repeat, past_agents_traj_len_idx] = past_agents_traj[past_agent_idx_repeat, past_agents_traj_len_idx]
            
            # Copy and repeat the past trajectories
            fake_trajs = real_trajs.clone().unsqueeze(0).repeat(self.num_candidates_train, 1, 1, 1)

            # Fill real future
            decoding_agent_idx = past_agent_idx[future_agent_masks]
            decoding_agent_idx_repeat = decoding_agent_idx.repeat_interleave(future_agents_traj_len)
            future_agent_idx = torch.arange(total_future_agent, device=self.device)
            future_agent_idx_repeat = future_agent_idx.repeat_interleave(future_agents_traj_len)
            
            shifted_future_agents_traj_len_idx = future_agents_traj_len_idx + past_agents_traj_len[future_agent_masks].repeat_interleave(future_agents_traj_len, 0)
            
            real_trajs[decoding_agent_idx_repeat, shifted_future_agents_traj_len_idx] = future_agents_traj[future_agent_idx_repeat, future_agents_traj_len_idx]

            # Fill Fake future
            fake_trajs[:, decoding_agent_idx_repeat, shifted_future_agents_traj_len_idx] = predicted_trajs_[future_agent_idx_repeat, :, future_agents_traj_len_idx].transpose(0, 1)
            fake_trajs = fake_trajs.reshape(total_past_agent*self.num_candidates_train, self.encoding_steps+self.decoding_steps, 2)

            # Calculate discriminator score
            true_score = self.discriminator(real_trajs, agents_total_traj_len, episode_idx, future_agent_masks, decode_start_pos, scene_images, 1)  # [num_agents X 1]
            
            num_past_agents_ = num_past_agents.repeat(self.num_candidates_train)
            batch_size_ = len(num_past_agents_)
            episode_idx_ = torch.arange(batch_size_, device=self.device).repeat_interleave(num_past_agents_, 0)
            future_agent_masks_ = future_agent_masks.repeat(self.num_candidates_train)
            decode_start_pos_ = decode_start_pos.repeat(self.num_candidates_train, 1)
            
            fake_score = self.discriminator(fake_trajs, agents_total_traj_len_, episode_idx_, future_agent_masks_, decode_start_pos_, scene_images, self.num_candidates_train)  # [num_agents X 1]

            ### Train Generator (i.e. MATF decoder)
            self.model.require_grad = True
            self.discriminator.require_grad = False

            error = future_agents_traj.unsqueeze(1) - predicted_trajs_ # Na x Nc x T x 2
            error = (error ** 2).sum(dim=-1) # Na x Nc x T
            rs_error = error.sqrt()

            # Two-Errors
            rs_error3 = rs_error[three_mask] # Na x Nc x T
            rs_error2 = rs_error[two_mask, :, :int(self.decoding_steps*2/3)]

            agent_time_index = torch.arange(total_future_agent, device=self.device).repeat_interleave(future_agents_traj_len)
            time_normalizer = future_agents_traj_len.float().repeat_interleave(future_agents_traj_len)

            batch_loss = error[agent_time_index, :, future_agents_traj_len_idx] / time_normalizer.unsqueeze(1)
            batch_loss = batch_loss.sum() / (total_future_agent * 2.0)

            num_agents2 = rs_error2.size(0)
            num_agents3 = rs_error3.size(0)

            ade2 = rs_error2.mean(-1) #  A X candi X T >> A X candi
            fde2 = rs_error2[..., -1]

            minade2, _ = ade2.min(dim=-1) # A X candi >> A
            avgade2 = ade2.mean(dim=-1)
            minfde2, _ = fde2.min(dim=-1)
            avgfde2 = fde2.mean(dim=-1)

            batch_minade2 = minade2.mean() # A >> 1
            batch_minfde2 = minfde2.mean()
            batch_avgade2 = avgade2.mean()
            batch_avgfde2 = avgfde2.mean()

            ade3 = rs_error3.mean(-1)
            fde3 = rs_error3[..., -1]

            minade3, _ = ade3.min(dim=-1)
            avgade3 = ade3.mean(dim=-1)
            minfde3, _ = fde3.min(dim=-1)
            avgfde3 = fde3.mean(dim=-1)

            batch_minade3 = minade3.mean()
            batch_minfde3 = minfde3.mean()
            batch_avgade3 = avgade3.mean()
            batch_avgfde3 = avgfde3.mean()

            batch_adversarial_loss = self.adversarial_loss(fake_score, torch.ones_like(fake_score))
            batch_g_loss = batch_loss + (batch_adversarial_loss * gan_weight)

            # Loss backward
            batch_g_loss.backward(retain_graph=True)
            self.optimizer.step()

            ### Train Discriminator
            self.discriminator.require_grad = True
            self.model.require_grad = False

            real_loss = self.adversarial_loss(true_score, torch.ones_like(true_score))
            fake_loss = self.adversarial_loss(fake_score, torch.zeros_like(fake_score))
            batch_d_loss = gan_weight*(real_loss + fake_loss)

            batch_d_loss.backward()
            self.optimizer_D.step()

            epoch_g_loss += batch_g_loss.item()
            epoch_d_loss += batch_d_loss.item()
            
            epoch_minade2 += batch_minade2.item() * num_agents2
            epoch_minfde2 += batch_minfde2.item() * num_agents2
            epoch_avgade2 += batch_avgade2.item() * num_agents2
            epoch_avgfde2 += batch_avgfde2.item() * num_agents2

            epoch_minade3 += batch_minade3.item() * num_agents3
            epoch_minfde3 += batch_minfde3.item() * num_agents3
            epoch_avgade3 += batch_avgade3.item() * num_agents3
            epoch_avgfde3 += batch_avgfde3.item() * num_agents3

            epoch_agents += total_future_agent
            epoch_agents2 += num_agents2
            epoch_agents3 += num_agents3

        epoch_g_loss /= epoch_agents
        epoch_d_loss /= epoch_agents

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

        return epoch_g_loss+epoch_d_loss, epoch_g_loss, epoch_d_loss, epoch_ades, epoch_fdes

    def inference_gan(self, epoch):
        self.model.eval()  # Set model to evaluate mode.
        
        with torch.no_grad():
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
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

                distance_2d = coordinate_2d - np.array([(H-1)/2, (H-1)/2]).reshape((2, 1, 1))
                distance = np.sqrt((distance_2d ** 2).sum(axis=0))
                distance = torch.FloatTensor(distance)
                distance = distance.reshape((1, 1, H, W))

                distance_std, distance_mean = torch.std_mean(distance)
                distance = (distance - distance_mean) / distance_std
                
                coordinate = coordinate.to(self.device)
                distance = distance.to(self.device)

            for i, e in enumerate(self.gan_weight_schedule):
                if epoch <= e:
                    gan_weight = self.gan_weight[i]
                    break

            for b, batch in enumerate(self.valid_loader):
                print("Working on batch {:d}/{:d}".format(b+1, len(self.valid_loader)), end='\r')
                self.optimizer.zero_grad()
                self.optimizer_D.zero_grad()

                scene_images, log_prior, \
                future_agent_masks, \
                num_past_agents,   past_agents_traj,   past_agents_traj_len,   past_agents_traj_len_idx, \
                num_future_agents, future_agents_traj, future_agents_traj_len, future_agents_traj_len_idx, \
                two_mask, three_mask, \
                decode_start_vel, decode_start_pos, \
                scene_id = batch

                # Detect dynamic batch size
                batch_size = scene_images.size(0)
                total_past_agent = past_agents_traj.size(0)
                total_future_agent = future_agents_traj.size(0)

                if '2.' in self.map_version:
                    coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                    distance_batch = distance.repeat(batch_size, 1, 1, 1)
                    scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)

                elif self.map_version == '1.3':
                    scene_images = scene_images.to(self.device)

                num_past_agents = num_past_agents.to(self.device)
                past_agents_traj = past_agents_traj.to(self.device)
                past_agents_traj_len = past_agents_traj_len.to(self.device)
                past_agents_traj_len_idx = past_agents_traj_len_idx.to(self.device)

                future_agent_masks = future_agent_masks.to(self.device)
                future_agents_traj = future_agents_traj.to(self.device)
                future_agents_traj_len = future_agents_traj_len.to(self.device)
                future_agents_traj_len_idx = future_agents_traj_len_idx.to(self.device)

                two_mask = two_mask.to(self.device)
                three_mask = three_mask.to(self.device)

                num_future_agents = num_future_agents.to(self.device)
                decode_start_vel = decode_start_vel.to(self.device)
                decode_start_pos = decode_start_pos.to(self.device)

                episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_past_agents)

                if 'GAN' in self.model_name:
                    stochastic = True
                    predicted_trajs = self.model(past_agents_traj, past_agents_traj_len, episode_idx, future_agent_masks,
                                                decode_start_vel, decode_start_pos, scene_images, stochastic, self.num_candidates)
                    predicted_trajs_ = predicted_trajs.reshape(total_future_agent, self.num_candidates, self.decoding_steps, 2)
                
                else:
                    raise ValueError("Unknown model type {:s}.".format(self.model_name))
                
                # Length for the concatnated trajectory
                agents_total_traj_len = past_agents_traj_len.clone()
                agents_total_traj_len[future_agent_masks] += future_agents_traj_len
                agents_total_traj_len_ = agents_total_traj_len.repeat_interleave(self.num_candidates)

                # Fill past trajectories
                past_agent_idx = torch.arange(total_past_agent, device=self.device)
                past_agent_idx_repeat = past_agent_idx.repeat_interleave(past_agents_traj_len)
                real_trajs = torch.zeros((total_past_agent, self.encoding_steps+self.decoding_steps, 2), device=self.device)
                real_trajs[past_agent_idx_repeat, past_agents_traj_len_idx] = past_agents_traj[past_agent_idx_repeat, past_agents_traj_len_idx]
                
                # Copy and repeat the past trajectories
                fake_trajs = real_trajs.clone().unsqueeze(0).repeat(self.num_candidates, 1, 1, 1)

                # Fill real future
                decoding_agent_idx = past_agent_idx[future_agent_masks]
                decoding_agent_idx_repeat = decoding_agent_idx.repeat_interleave(future_agents_traj_len)
                future_agent_idx = torch.arange(total_future_agent, device=self.device)
                future_agent_idx_repeat = future_agent_idx.repeat_interleave(future_agents_traj_len)
                
                shifted_future_agents_traj_len_idx = future_agents_traj_len_idx + past_agents_traj_len[future_agent_masks].repeat_interleave(future_agents_traj_len, 0)
                
                real_trajs[decoding_agent_idx_repeat, shifted_future_agents_traj_len_idx] = future_agents_traj[future_agent_idx_repeat, future_agents_traj_len_idx]

                # Fill Fake future
                fake_trajs[:, decoding_agent_idx_repeat, shifted_future_agents_traj_len_idx] = predicted_trajs_[future_agent_idx_repeat, :, future_agents_traj_len_idx].transpose(0, 1)
                fake_trajs = fake_trajs.reshape(total_past_agent*self.num_candidates, self.encoding_steps+self.decoding_steps, 2)

                # Calculate discriminator score
                true_score = self.discriminator(real_trajs, agents_total_traj_len, episode_idx, future_agent_masks, decode_start_pos, scene_images, 1)  # [num_agents X 1]
                
                num_past_agents_ = num_past_agents.repeat(self.num_candidates)
                batch_size_ = len(num_past_agents_)
                episode_idx_ = torch.arange(batch_size_, device=self.device).repeat_interleave(num_past_agents_, 0)
                future_agent_masks_ = future_agent_masks.repeat(self.num_candidates)
                decode_start_pos_ = decode_start_pos.repeat(self.num_candidates, 1)

                fake_score = self.discriminator(fake_trajs, agents_total_traj_len_, episode_idx_, future_agent_masks_, decode_start_pos_, scene_images, self.num_candidates)  # [num_agents X 1]

                ### Evaluate Generator (i.e. MATF decoder)
                error = future_agents_traj.unsqueeze(1) - predicted_trajs_ # Na x Nc x T x 2
                error = (error ** 2).sum(dim=-1) # Na x Nc x T
                rs_error = error.sqrt()

                # Two-Errors
                rs_error3 = rs_error[three_mask] # Na x Nc x T
                rs_error2 = rs_error[two_mask, :, :int(self.decoding_steps*2/3)]

                agent_time_index = torch.arange(total_future_agent, device=self.device).repeat_interleave(future_agents_traj_len)
                time_normalizer = future_agents_traj_len.float().repeat_interleave(future_agents_traj_len)

                batch_loss = error[agent_time_index, :, future_agents_traj_len_idx] / time_normalizer.unsqueeze(1)
                batch_loss = batch_loss.sum() / (total_future_agent * 2.0)

                num_agents2 = rs_error2.size(0)
                num_agents3 = rs_error3.size(0)

                ade2 = rs_error2.mean(-1) #  A X candi X T >> A X candi
                fde2 = rs_error2[..., -1]

                minade2, _ = ade2.min(dim=-1) # A X candi >> A
                avgade2 = ade2.mean(dim=-1)
                minfde2, _ = fde2.min(dim=-1)
                avgfde2 = fde2.mean(dim=-1)

                batch_minade2 = minade2.mean() # A >> 1
                batch_minfde2 = minfde2.mean()
                batch_avgade2 = avgade2.mean()
                batch_avgfde2 = avgfde2.mean()

                ade3 = rs_error3.mean(-1)
                fde3 = rs_error3[..., -1]

                minade3, _ = ade3.min(dim=-1)
                avgade3 = ade3.mean(dim=-1)
                minfde3, _ = fde3.min(dim=-1)
                avgfde3 = fde3.mean(dim=-1)

                batch_minade3 = minade3.mean()
                batch_minfde3 = minfde3.mean()
                batch_avgade3 = avgade3.mean()
                batch_avgfde3 = avgfde3.mean()

                batch_adversarial_loss = self.adversarial_loss(fake_score, torch.ones_like(fake_score))
                batch_g_loss = batch_loss + (batch_adversarial_loss * gan_weight)

                # Evaluate D loss
                real_loss = self.adversarial_loss(true_score, torch.ones_like(true_score))
                fake_loss = self.adversarial_loss(fake_score, torch.zeros_like(fake_score))
                batch_d_loss = gan_weight*(real_loss + fake_loss)

                epoch_g_loss += batch_g_loss.item()
                epoch_d_loss += batch_d_loss.item()
                
                epoch_minade2 += batch_minade2.item() * num_agents2
                epoch_minfde2 += batch_minfde2.item() * num_agents2
                epoch_avgade2 += batch_avgade2.item() * num_agents2
                epoch_avgfde2 += batch_avgfde2.item() * num_agents2

                epoch_minade3 += batch_minade3.item() * num_agents3
                epoch_minfde3 += batch_minfde3.item() * num_agents3
                epoch_avgade3 += batch_avgade3.item() * num_agents3
                epoch_avgfde3 += batch_avgfde3.item() * num_agents3

                epoch_agents += total_future_agent
                epoch_agents2 += num_agents2
                epoch_agents3 += num_agents3

            epoch_g_loss /= epoch_agents
            epoch_d_loss /= epoch_agents

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

            scheduler_metric = epoch_avgade3 + epoch_avgfde3 

            return epoch_g_loss+epoch_d_loss, epoch_g_loss, epoch_d_loss, epoch_ades, epoch_fdes, scheduler_metric

    # Proposed model training and inference
    def train_single_epoch_proposed(self):
        """Trains the model for a single round."""
        self.model.train()
        epoch_loss = 0.0
        epoch_qloss = 0.0
        epoch_ploss = 0.0
        epoch_minade2, epoch_avgade2 = 0.0, 0.0
        epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
        epoch_minade3, epoch_avgade3 = 0.0, 0.0
        epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
        epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

        H = W = 64
        if self.map_version == '2.0':
            """ Make position & distance embeddings for map v2.0"""
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

        c1 = -self.decoding_steps * np.log(2 * np.pi)
        for b, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            scene_images, log_prior, \
            future_agent_masks, \
            num_past_agents,   past_agents_traj,   past_agents_traj_len,   past_agents_traj_len_idx, \
            num_future_agents, future_agents_traj, future_agents_traj_len, future_agents_traj_len_idx, \
            two_mask, three_mask, \
            decode_start_vel, decode_start_pos, \
            scene_id = batch

            # Detect dynamic sizes
            batch_size = scene_images.size(0)
            # num_encoding_agents = past_agents_traj.size(0)
            # num_decoding_agents = future_agents_traj.size(0)
            num_three_agents = torch.sum(three_mask)

            if self.map_version == '2.0':
                coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                distance_batch = distance.repeat(batch_size, 1, 1, 1)
                
                scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)
            
            past_agents_traj = past_agents_traj.to(self.device)
            past_agents_traj_len = past_agents_traj_len.to(self.device)

            future_agents_traj = future_agents_traj.to(self.device)[three_mask]
            future_agents_traj_len = future_agents_traj_len.to(self.device)[three_mask]

            num_future_agents = num_future_agents.to(self.device)
            episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_future_agents)[three_mask]

            future_agent_masks = future_agent_masks.to(self.device)
            agent_tgt_three_mask = torch.zeros_like(future_agent_masks)
            agent_masks_idx = torch.arange(len(future_agent_masks), device=self.device)[future_agent_masks][three_mask]
            agent_tgt_three_mask[agent_masks_idx] = True

            decode_start_vel = decode_start_vel.to(self.device)[agent_tgt_three_mask]
            decode_start_pos = decode_start_pos.to(self.device)[agent_tgt_three_mask]

            log_prior = log_prior.to(self.device)

            if self.flow_based_decoder:
                # Normalizing Flow (q loss)
                # z_: A X Td X 2
                # mu_: A X Td X 2
                # sigma_: A X Td X 2 X 2
                # Generate perturbation
                perterb = torch.normal(mean=0.0, std=np.sqrt(0.001), size=future_agents_traj.shape, device=self.device)
                
                if self.model_name == 'R2P2_SimpleRNN':
                    z_, mu_, sigma_, motion_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, decode_start_vel, decode_start_pos)

                elif self.model_name == 'R2P2_RNN':
                    z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, episode_idx, decode_start_vel, decode_start_pos, scene_images)

                elif self.model_name == 'CAM_NFDecoder':
                    z_, mu_, sigma_, motion_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_past_agents)

                elif self.model_name == 'Scene_CAM_NFDecoder':
                    z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)

                elif self.model_name == 'Global_Scene_CAM_NFDecoder':
                    z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)

                elif self.model_name == 'AttGlobal_Scene_CAM_NFDecoder':
                    z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)

                z_ = z_.reshape((num_three_agents, -1)) # A X (Td*2)
                log_q0 = c1 - 0.5 * ((z_ ** 2).sum(dim=1))

                logdet_sigma = log_determinant(sigma_)

                log_qpi = log_q0 - logdet_sigma.sum(dim=1)
                qloss = -log_qpi
                batch_qloss = qloss.mean()
                
                # Prior Loss (p loss)
                if self.model_name == 'R2P2_SimpleRNN':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, decode_start_vel, decode_start_pos, motion_encoded=True)

                elif self.model_name == 'R2P2_RNN':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, episode_idx, decode_start_vel, decode_start_pos, scene_encoding_, motion_encoded=True, scene_encoded=True)

                elif self.model_name == 'CAM_NFDecoder':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_past_agents, agent_encoded=True)

                elif self.model_name == 'Scene_CAM_NFDecoder':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_encoding_, agent_encoded=True, scene_encoded=True)

                elif self.model_name == 'Global_Scene_CAM_NFDecoder':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_encoding_, agent_encoded=True, scene_encoded=True)

                elif self.model_name == 'AttGlobal_Scene_CAM_NFDecoder':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_encoding_, agent_encoded=True, scene_encoded=True)

                if self.beta != 0.0:
                    if self.cfg.ploss_type == 'mseloss':
                        ploss = self.ploss_criterion(gen_trajs, past_agents_traj)
                    else:
                        ploss = self.ploss_criterion(episode_idx, gen_trajs, log_prior, -15.0)

                else:
                    ploss = torch.zeros(size=(1,), device=self.device)

                batch_ploss = ploss.mean()
                batch_loss = batch_qloss + self.beta * batch_ploss

                epoch_ploss += batch_ploss.item() * batch_size
                epoch_qloss += batch_qloss.item() * batch_size

            else:

                if 'CAM' == self.model_name:
                    gen_trajs = self.model(past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_past_agents)

                gen_trajs = gen_trajs.reshape(num_three_agents, self.num_candidates, self.decoding_steps, 2)


            rs_error3 = ((gen_trajs - future_agents_traj.unsqueeze(1)) ** 2).sum(dim=-1).sqrt_() # A X candi X T X 2 >> A X candi X T
            rs_error2 = rs_error3[..., :int(self.decoding_steps*2/3)]
            
            num_agents = gen_trajs.size(0)
            num_agents2 = rs_error2.size(0)
            num_agents3 = rs_error3.size(0)

            ade2 = rs_error2.mean(-1) #  A X candi X T >> A X candi
            fde2 = rs_error2[..., -1]

            minade2, _ = ade2.min(dim=-1) # A X candi >> A
            avgade2 = ade2.mean(dim=-1)
            minfde2, _ = fde2.min(dim=-1)
            avgfde2 = fde2.mean(dim=-1)

            batch_minade2 = minade2.mean() # A >> 1
            batch_minfde2 = minfde2.mean()
            batch_avgade2 = avgade2.mean()
            batch_avgfde2 = avgfde2.mean()

            ade3 = rs_error3.mean(-1)
            fde3 = rs_error3[..., -1]

            minade3, _ = ade3.min(dim=-1)
            avgade3 = ade3.mean(dim=-1)
            minfde3, _ = fde3.min(dim=-1)
            avgfde3 = fde3.mean(dim=-1)

            batch_minade3 = minade3.mean()
            batch_minfde3 = minfde3.mean()
            batch_avgade3 = avgade3.mean()
            batch_avgfde3 = avgfde3.mean()

            if self.flow_based_decoder is not True:
                batch_loss = batch_minade3
                epoch_loss += batch_loss.item()
                batch_qloss = torch.zeros(1)
                batch_ploss = torch.zeros(1)

            # Loss backward
            batch_loss.backward()
            self.optimizer.step()

            print("Working on train batch {:d}/{:d}... ".format(b+1, len(self.train_loader)) +
            "batch_loss: {:.2f}, qloss: {:.2f}, ploss: {:g}, ".format(batch_loss.item(), batch_qloss.item(), batch_ploss.item()) +
            "minFDE3: {:.2f}, avgFDE3: {:.2f}".format(batch_minfde3.item(), batch_avgfde3.item()), end='\r')

            epoch_minade2 += batch_minade2.item() * num_agents2
            epoch_avgade2 += batch_avgade2.item() * num_agents2
            epoch_minfde2 += batch_minfde2.item() * num_agents2
            epoch_avgfde2 += batch_avgfde2.item() * num_agents2
            epoch_minade3 += batch_minade3.item() * num_agents3
            epoch_avgade3 += batch_avgade3.item() * num_agents3
            epoch_minfde3 += batch_minfde3.item() * num_agents3
            epoch_avgfde3 += batch_avgfde3.item() * num_agents3

            epoch_agents += num_agents
            epoch_agents2 += num_agents2
            epoch_agents3 += num_agents3


        if self.flow_based_decoder:
            epoch_ploss /= epoch_agents
            epoch_qloss /= epoch_agents
            epoch_loss = epoch_qloss + self.beta * epoch_ploss
        else:
            epoch_loss /= epoch_agents

        epoch_minade2 /= epoch_agents2
        epoch_avgade2 /= epoch_agents2
        epoch_minfde2 /= epoch_agents2
        epoch_avgfde2 /= epoch_agents2
        epoch_minade3 /= epoch_agents3
        epoch_avgade3 /= epoch_agents3
        epoch_minfde3 /= epoch_agents3
        epoch_avgfde3 /= epoch_agents3

        epoch_ades = [epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3]
        epoch_fdes = [epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3]

        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        return epoch_loss, epoch_qloss, epoch_ploss, epoch_ades, epoch_fdes

    def inference_propsed(self):
        self.model.eval()  # Set model to evaluate mode.
        
        epoch_loss = 0.0
        epoch_qloss = 0.0
        epoch_ploss = 0.0
        epoch_minade2, epoch_avgade2 = 0.0, 0.0
        epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
        epoch_minade3, epoch_avgade3 = 0.0, 0.0
        epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
        epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

        H = W = 64
        with torch.no_grad():
            if self.map_version == '2.0':
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
            
            c1 = -self.decoding_steps * np.log(2 * np.pi)
            for b, batch in enumerate(self.valid_loader):

                scene_images, log_prior, \
                future_agent_masks, \
                num_past_agents,   past_agents_traj,   past_agents_traj_len,   past_agents_traj_len_idx, \
                num_future_agents, future_agents_traj, future_agents_traj_len, future_agents_traj_len_idx, \
                two_mask, three_mask, \
                decode_start_vel, decode_start_pos, \
                scene_id = batch

                # Detect dynamic batch size
                batch_size = scene_images.size(0)
                three_mask = three_mask.to(self.device)
                # num_encoding_agents = past_agents_traj.size(0)
                # num_decoding_agents = future_agents_traj.size(0)
                num_three_agents = torch.sum(three_mask)

                if self.map_version == '2.0':
                    coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                    distance_batch = distance.repeat(batch_size, 1, 1, 1)
                    scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)
                elif self.map_version == '1.3':
                    scene_images = scene_images.to(self.device)

                past_agents_traj = past_agents_traj.to(self.device)
                past_agents_traj_len = past_agents_traj_len.to(self.device)

                future_agents_traj = future_agents_traj.to(self.device)[three_mask]
                future_agents_traj_len = future_agents_traj_len.to(self.device)[three_mask]

                num_future_agents = num_future_agents.to(self.device)
                # past_agents_traj = num_future_agents
                episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_future_agents)[three_mask]

                future_agent_masks = future_agent_masks.to(self.device)
                agent_tgt_three_mask = torch.zeros_like(future_agent_masks)
                agent_masks_idx = torch.arange(len(future_agent_masks), device=self.device)[future_agent_masks][three_mask]
                agent_tgt_three_mask[agent_masks_idx] = True

                decode_start_vel = decode_start_vel.to(self.device)[agent_tgt_three_mask]
                decode_start_pos = decode_start_pos.to(self.device)[agent_tgt_three_mask]

                log_prior = log_prior.to(self.device)

                if self.flow_based_decoder:
                    # Normalizing Flow (q loss)
                    # z: A X Td X 2
                    # mu: A X Td X 2
                    # sigma: A X Td X 2 X 2
                    # Generate perturbation
                    perterb = torch.normal(mean=0.0, std=np.sqrt(0.001), size=future_agents_traj.shape, device=self.device)
                    
                    if self.model_name == 'R2P2_SimpleRNN':
                        z_, mu_, sigma_, motion_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, decode_start_vel, decode_start_pos)

                    elif self.model_name == 'R2P2_RNN':
                        z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, episode_idx, decode_start_vel, decode_start_pos, scene_images)

                    elif self.model_name == 'CAM_NFDecoder':
                        z_, mu_, sigma_, motion_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_past_agents)

                    elif self.model_name == 'Scene_CAM_NFDecoder':
                        z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)

                    elif self.model_name == 'Global_Scene_CAM_NFDecoder':
                        z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)

                    elif self.model_name == 'AttGlobal_Scene_CAM_NFDecoder':
                        z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)

                    z_ = z_.reshape((num_three_agents, -1)) # A X (Td*2)
                    log_q0 = c1 - 0.5 * ((z_ ** 2).sum(dim=1))

                    logdet_sigma = log_determinant(sigma_)

                    log_qpi = log_q0 - logdet_sigma.sum(dim=1)
                    qloss = -log_qpi
                    batch_qloss = qloss.mean()

                    # Prior Loss (p loss)
                    if self.model_name == 'R2P2_SimpleRNN':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, decode_start_vel, decode_start_pos, motion_encoded=True)

                    elif self.model_name == 'R2P2_RNN':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, episode_idx, decode_start_vel, decode_start_pos, scene_encoding_, motion_encoded=True, scene_encoded=True)

                    elif self.model_name == 'CAM_NFDecoder':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_past_agents, agent_encoded=True)

                    elif self.model_name == 'Scene_CAM_NFDecoder':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_encoding_, agent_encoded=True, scene_encoded=True)

                    elif self.model_name == 'Global_Scene_CAM_NFDecoder':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_encoding_, agent_encoded=True, scene_encoded=True)

                    elif self.model_name == 'AttGlobal_Scene_CAM_NFDecoder':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_encoding_, agent_encoded=True, scene_encoded=True)

                    if self.beta != 0.0:
                        if self.cfg.ploss_type == 'mseloss':
                            ploss = self.ploss_criterion(gen_trajs, past_agents_traj)
                        else:
                            ploss = self.ploss_criterion(episode_idx, gen_trajs, log_prior, -15.0)

                    else:
                        ploss = torch.zeros(size=(1,), device=self.device)

                    batch_ploss = ploss.mean()
                    batch_loss = batch_qloss + self.beta * batch_ploss

                    epoch_ploss += batch_ploss.item() * batch_size
                    epoch_qloss += batch_qloss.item() * batch_size   

                else:

                    if 'CAM' == self.model_name:
                        gen_trajs = self.model(past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_past_agents)

                    gen_trajs = gen_trajs.reshape(num_three_agents, self.num_candidates, self.decoding_steps, 2)


                rs_error3 = ((gen_trajs - future_agents_traj.unsqueeze(1)) ** 2).sum(dim=-1).sqrt_()
                rs_error2 = rs_error3[..., :int(self.decoding_steps*2/3)]
                
                num_agents = gen_trajs.size(0)
                num_agents2 = rs_error2.size(0)
                num_agents3 = rs_error3.size(0)

                ade2 = rs_error2.mean(-1)
                fde2 = rs_error2[..., -1]

                minade2, _ = ade2.min(dim=-1)
                avgade2 = ade2.mean(dim=-1)
                minfde2, _ = fde2.min(dim=-1)
                avgfde2 = fde2.mean(dim=-1)

                batch_minade2 = minade2.mean()
                batch_minfde2 = minfde2.mean()
                batch_avgade2 = avgade2.mean()
                batch_avgfde2 = avgfde2.mean()

                ade3 = rs_error3.mean(-1)
                fde3 = rs_error3[..., -1]

                minade3, _ = ade3.min(dim=-1)
                avgade3 = ade3.mean(dim=-1)
                minfde3, _ = fde3.min(dim=-1)
                avgfde3 = fde3.mean(dim=-1)

                batch_minade3 = minade3.mean()
                batch_minfde3 = minfde3.mean()
                batch_avgade3 = avgade3.mean()
                batch_avgfde3 = avgfde3.mean()

                if self.flow_based_decoder is not True:
                    batch_loss = batch_minade3
                    epoch_loss += batch_loss.item()
                    batch_qloss = torch.zeros(1)
                    batch_ploss = torch.zeros(1)

                print("Working on valid batch {:d}/{:d}... ".format(b+1, len(self.valid_loader)) +
                "batch_loss: {:.2f}, qloss: {:.2f}, ploss: {:g}, ".format(batch_loss.item(), batch_qloss.item(), batch_ploss.item()) +
                "minFDE3: {:.2f}, avgFDE3: {:.2f}".format(batch_minfde3.item(), batch_avgfde3.item()), end='\r')

                epoch_ploss += batch_ploss.item() * batch_size
                epoch_qloss += batch_qloss.item() * batch_size
                epoch_minade2 += batch_minade2.item() * num_agents2
                epoch_avgade2 += batch_avgade2.item() * num_agents2
                epoch_minfde2 += batch_minfde2.item() * num_agents2
                epoch_avgfde2 += batch_avgfde2.item() * num_agents2
                epoch_minade3 += batch_minade3.item() * num_agents3
                epoch_avgade3 += batch_avgade3.item() * num_agents3
                epoch_minfde3 += batch_minfde3.item() * num_agents3
                epoch_avgfde3 += batch_avgfde3.item() * num_agents3

                epoch_agents += num_agents
                epoch_agents2 += num_agents2
                epoch_agents3 += num_agents3
        
        if self.flow_based_decoder:
            epoch_ploss /= epoch_agents
            epoch_qloss /= epoch_agents
            epoch_loss = epoch_qloss + self.beta * epoch_ploss
        else:
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

        scheduler_metric = epoch_avgade3 + epoch_avgfde3 

        return epoch_loss, epoch_qloss, epoch_ploss, epoch_ades, epoch_fdes, scheduler_metric

    # R2P2 Trian inference
    def train_single_epoch_r2p2(self):
        """Trains the model for a single round."""
        self.model.train()
        epoch_loss = 0.0
        epoch_qloss = 0.0
        epoch_ploss = 0.0
        epoch_minade2, epoch_avgade2 = 0.0, 0.0
        epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
        epoch_minade3, epoch_avgade3 = 0.0, 0.0
        epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
        epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

        H = W = 64
        if '2.' in self.map_version:
            """ Make position & distance embeddings for map v2.x"""
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

        c1 = -self.decoding_steps * np.log(2 * np.pi)
        for b, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            scene_images, log_prior, \
            agent_masks, \
            num_src_trajs, src_trajs, src_lens, src_len_idx, \
            num_tgt_trajs, tgt_trajs, tgt_lens, tgt_len_idx, \
            tgt_two_mask, tgt_three_mask, \
            decode_start_vel, decode_start_pos, scene_id = batch

            # Detect dynamic sizes
            batch_size = scene_images.size(0)

            if '2.' in self.map_version:
                coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                distance_batch = distance.repeat(batch_size, 1, 1, 1)
                scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)
            scene_images = scene_images.to(self.device)
            
            src_trajs = src_trajs.to(self.device)[agent_masks][tgt_three_mask]
            tgt_trajs = tgt_trajs.to(self.device)[tgt_three_mask]
            
            decode_start_vel = decode_start_vel.to(self.device)[agent_masks][tgt_three_mask]
            decode_start_pos = decode_start_pos.to(self.device)[agent_masks][tgt_three_mask]

            num_tgt_trajs = num_tgt_trajs.to(self.device)

            log_prior = log_prior.to(self.device)

            # Total number of three-masked agents in this batch
            with torch.no_grad():
                episode_idx = torch.arange(len(num_tgt_trajs), device=self.device).repeat_interleave(num_tgt_trajs)
                episode_idx = episode_idx[tgt_three_mask]
                total_three_agents = episode_idx.size(0)
            # Normalizing Flow (q loss)
            # z_: Na X (Td*2)
            # mu_: Na X Td X 2
            # sigma_: Na X Td X 2 X 2

            # Generate perturbation
            perterb = torch.normal(mean=0.0, std=np.sqrt(0.001), size=tgt_trajs.shape, device=self.device)

            if self.model_name == 'R2P2_SimpleRNN':
                z_, mu_, sigma_, motion_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, episode_idx, decode_start_vel, decode_start_pos)
            elif self.model_name == 'R2P2_RNN':
                z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, episode_idx, decode_start_vel, decode_start_pos, scene_images)

            log_q0 = c1 - 0.5 * ((z_ ** 2).sum(dim=1))

            logdet_sigma = log_determinant(sigma_)

            log_qpi = log_q0 - logdet_sigma.sum(dim=1)
            qloss = -log_qpi
            batch_qloss = qloss.mean()
            
            # Prior Loss (p loss)
            if self.model_name == 'R2P2_SimpleRNN':
                gen_trajs, z, mu, sigma = self.model(motion_encoding_, episode_idx, decode_start_vel, decode_start_pos, motion_encoded=True)
            elif self.model_name == 'R2P2_RNN':
                gen_trajs, z, mu, sigma = self.model(motion_encoding_, episode_idx, decode_start_vel, decode_start_pos, scene_encoding_, motion_encoded=True, scene_encoded=True)

            if self.beta != 0.0:
                if self.cfg.ploss_type == 'mseloss':
                    ploss = self.ploss_criterion(gen_trajs, tgt_trajs)
                else:
                    ploss = self.ploss_criterion(episode_idx, gen_trajs, log_prior, log_prior.min())
            
            else:
                ploss = torch.zeros(size=(1,), device=self.device)

            batch_ploss = ploss.mean()
            batch_loss = batch_qloss + self.beta * batch_ploss
            batch_loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                rs_error3 = ((gen_trajs - tgt_trajs.unsqueeze(1)) ** 2).sum(dim=-1).sqrt_() # A X candi X T X 2 >> A X candi X T
                rs_error2 = rs_error3[..., :int(self.decoding_steps*2/3)]
                
                num_agents = gen_trajs.size(0)
                num_agents2 = rs_error2.size(0)
                num_agents3 = rs_error3.size(0)

                ade2 = rs_error2.mean(-1) #  A X candi X T >> A X candi
                fde2 = rs_error2[..., -1]

                minade2, _ = ade2.min(dim=-1) # A X candi >> A
                avgade2 = ade2.mean(dim=-1)
                minfde2, _ = fde2.min(dim=-1)
                avgfde2 = fde2.mean(dim=-1)

                batch_minade2 = minade2.mean() # A >> 1
                batch_minfde2 = minfde2.mean()
                batch_avgade2 = avgade2.mean()
                batch_avgfde2 = avgfde2.mean()

                ade3 = rs_error3.mean(-1)
                fde3 = rs_error3[..., -1]

                minade3, _ = ade3.min(dim=-1)
                avgade3 = ade3.mean(dim=-1)
                minfde3, _ = fde3.min(dim=-1)
                avgfde3 = fde3.mean(dim=-1)

                batch_minade3 = minade3.mean()
                batch_minfde3 = minfde3.mean()
                batch_avgade3 = avgade3.mean()
                batch_avgfde3 = avgfde3.mean()

            print("Working on train batch {:d}/{:d}... ".format(b+1, len(self.train_loader)) +
            "batch_loss: {:.2f}, qloss: {:.2f}, ploss: {:g}, ".format(batch_loss.item(), batch_qloss.item(), batch_ploss.item()) +
            "minFDE3: {:.2f}, avgFDE3: {:.2f}".format(batch_minfde3.item(), batch_avgfde3.item()), end='\r')

            epoch_ploss += batch_ploss.item() * batch_size
            epoch_qloss += batch_qloss.item() * batch_size
            epoch_minade2 += batch_minade2.item() * num_agents2
            epoch_avgade2 += batch_avgade2.item() * num_agents2
            epoch_minfde2 += batch_minfde2.item() * num_agents2
            epoch_avgfde2 += batch_avgfde2.item() * num_agents2
            epoch_minade3 += batch_minade3.item() * num_agents3
            epoch_avgade3 += batch_avgade3.item() * num_agents3
            epoch_minfde3 += batch_minfde3.item() * num_agents3
            epoch_avgfde3 += batch_avgfde3.item() * num_agents3

            epoch_agents += num_agents
            epoch_agents2 += num_agents2
            epoch_agents3 += num_agents3

        epoch_ploss /= epoch_agents
        epoch_qloss /= epoch_agents
        epoch_loss = epoch_qloss + self.beta * epoch_ploss
        epoch_minade2 /= epoch_agents2
        epoch_avgade2 /= epoch_agents2
        epoch_minfde2 /= epoch_agents2
        epoch_avgfde2 /= epoch_agents2
        epoch_minade3 /= epoch_agents3
        epoch_avgade3 /= epoch_agents3
        epoch_minfde3 /= epoch_agents3
        epoch_avgfde3 /= epoch_agents3

        epoch_ades = [epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3]
        epoch_fdes = [epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3]

        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        return epoch_loss, epoch_qloss, epoch_ploss, epoch_ades, epoch_fdes

    def inference_r2p2(self):
        self.model.eval()  # Set model to evaluate mode.
        
        epoch_loss = 0.0
        epoch_qloss = 0.0
        epoch_ploss = 0.0
        epoch_minade2, epoch_avgade2 = 0.0, 0.0
        epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
        epoch_minade3, epoch_avgade3 = 0.0, 0.0
        epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
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

                distance_2d = coordinate_2d - np.array([(H-1)/2, (H-1)/2]).reshape((2, 1, 1))
                distance = np.sqrt((distance_2d ** 2).sum(axis=0))
                distance = torch.FloatTensor(distance)
                distance = distance.reshape((1, 1, H, W))

                distance_std, distance_mean = torch.std_mean(distance)
                distance = (distance - distance_mean) / distance_std
            
                coordinate = coordinate.to(self.device)
                distance = distance.to(self.device)
            
            c1 = -self.decoding_steps * np.log(2 * np.pi)
            for b, batch in enumerate(self.valid_loader):
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
                
                src_trajs = src_trajs.to(self.device)[agent_masks][tgt_three_mask]
                tgt_trajs = tgt_trajs.to(self.device)[tgt_three_mask]
                
                decode_start_vel = decode_start_vel.to(self.device)[agent_masks][tgt_three_mask]
                decode_start_pos = decode_start_pos.to(self.device)[agent_masks][tgt_three_mask]
                
                num_tgt_trajs = num_tgt_trajs.to(self.device)
                
                log_prior = log_prior.to(self.device)

                # Total number of three-masked agents in this batch
                with torch.no_grad():
                    episode_idx = torch.arange(len(num_tgt_trajs), device=self.device).repeat_interleave(num_tgt_trajs)
                    episode_idx = episode_idx[tgt_three_mask]
                    total_three_agents = episode_idx.size(0)
                # Normalizing Flow (q loss)
                # z: Na X (Td*2)
                # mu: Na X Td X 2
                # sigma: Na X Td X 2 X 2

                # Generate perturbation
                perterb = torch.normal(mean=0.0, std=np.sqrt(0.001), size=tgt_trajs.shape, device=self.device)
                
                if self.model_name == 'R2P2_SimpleRNN':
                    z_, mu_, sigma_, motion_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, episode_idx, decode_start_vel, decode_start_pos)
                elif self.model_name == 'R2P2_RNN':
                    z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, episode_idx, decode_start_vel, decode_start_pos, scene_images)

                log_q0 = c1 - 0.5 * ((z_ ** 2).sum(dim=1))

                logdet_sigma = log_determinant(sigma_)

                log_qpi = log_q0 - logdet_sigma.sum(dim=1)
                qloss = -log_qpi
                batch_qloss = qloss.mean()

                # Prior Loss (p loss)
                if self.model_name == 'R2P2_SimpleRNN':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, episode_idx, decode_start_vel, decode_start_pos, motion_encoded=True)
                elif self.model_name == 'R2P2_RNN':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, episode_idx, decode_start_vel, decode_start_pos, scene_encoding_, motion_encoded=True, scene_encoded=True)

                if self.beta != 0.0:
                    if self.cfg.ploss_type == 'mseloss':
                        ploss = self.ploss_criterion(gen_trajs, tgt_trajs)
                    else:
                        ploss = self.ploss_criterion(episode_idx, gen_trajs, log_prior, log_prior.min())

                else:
                    ploss = torch.zeros(size=(1,), device=self.device)

                batch_ploss = ploss.mean()
                batch_loss = batch_qloss + self.beta * batch_ploss
                                
                rs_error3 = ((gen_trajs - tgt_trajs.unsqueeze(1)) ** 2).sum(dim=-1).sqrt_()
                rs_error2 = rs_error3[..., :int(self.decoding_steps*2/3)]
                
                num_agents = gen_trajs.size(0)
                num_agents2 = rs_error2.size(0)
                num_agents3 = rs_error3.size(0)

                ade2 = rs_error2.mean(-1)
                fde2 = rs_error2[..., -1]

                minade2, _ = ade2.min(dim=-1)
                avgade2 = ade2.mean(dim=-1)
                minfde2, _ = fde2.min(dim=-1)
                avgfde2 = fde2.mean(dim=-1)

                batch_minade2 = minade2.mean()
                batch_minfde2 = minfde2.mean()
                batch_avgade2 = avgade2.mean()
                batch_avgfde2 = avgfde2.mean()

                ade3 = rs_error3.mean(-1)
                fde3 = rs_error3[..., -1]

                minade3, _ = ade3.min(dim=-1)
                avgade3 = ade3.mean(dim=-1)
                minfde3, _ = fde3.min(dim=-1)
                avgfde3 = fde3.mean(dim=-1)

                batch_minade3 = minade3.mean()
                batch_minfde3 = minfde3.mean()
                batch_avgade3 = avgade3.mean()
                batch_avgfde3 = avgfde3.mean()

                print("Working on valid batch {:d}/{:d}... ".format(b+1, len(self.valid_loader)) +
                "batch_loss: {:.2f}, qloss: {:.2f}, ploss: {:g}, ".format(batch_loss.item(), batch_qloss.item(), batch_ploss.item()) +
                "minFDE3: {:.2f}, avgFDE3: {:.2f}".format(batch_minfde3.item(), batch_avgfde3.item()), end='\r')

                epoch_ploss += batch_ploss.item() * batch_size
                epoch_qloss += batch_qloss.item() * batch_size
                epoch_minade2 += batch_minade2.item() * num_agents2
                epoch_avgade2 += batch_avgade2.item() * num_agents2
                epoch_minfde2 += batch_minfde2.item() * num_agents2
                epoch_avgfde2 += batch_avgfde2.item() * num_agents2
                epoch_minade3 += batch_minade3.item() * num_agents3
                epoch_avgade3 += batch_avgade3.item() * num_agents3
                epoch_minfde3 += batch_minfde3.item() * num_agents3
                epoch_avgfde3 += batch_avgfde3.item() * num_agents3

                epoch_agents += num_agents
                epoch_agents2 += num_agents2
                epoch_agents3 += num_agents3
        
        epoch_ploss /= epoch_agents
        epoch_qloss /= epoch_agents
        epoch_loss = epoch_qloss + self.beta * epoch_ploss
        epoch_minade2 /= epoch_agents2
        epoch_avgade2 /= epoch_agents2
        epoch_minfde2 /= epoch_agents2
        epoch_avgfde2 /= epoch_agents2
        epoch_minade3 /= epoch_agents3
        epoch_avgade3 /= epoch_agents3
        epoch_minfde3 /= epoch_agents3
        epoch_avgfde3 /= epoch_agents3

        epoch_ades = [epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3]
        epoch_fdes = [epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3]

        scheduler_metric = epoch_loss
        torch.cuda.empty_cache()

        return epoch_loss, epoch_qloss, epoch_ploss, epoch_ades, epoch_fdes, scheduler_metric

    def get_lr(self):
        """Returns Learning Rate of the Optimizer."""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
            
    def get_desire_lr(self):
        """Returns Learning Rate of the Optimizer."""
        for param_group in self.optimizer[1].param_groups:
            return param_group['lr']

    def get_D_lr(self):
        for param_group in self.optimizer_D.param_groups:
            return param_group['lr']

    def save_checkpoint_r2p2(self, epoch, qloss, ploss, ade, fde):
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
            'val_ploss': ploss,
            'val_qloss': qloss,
            'val_ade': ade,
            'val_fde': fde,
        }

        save_path = "{}/ck_{}_{:0.4f}_{:0.4f}_{:0.4f}_{:0.4f}.pth.tar".format(self.exp_path, epoch, qloss, ploss, ade, fde)
        torch.save(state_dict, save_path)


    def save_checkpoint(self, epoch, ade, fde):
        """Saves experiment checkpoint.
        Saved state consits of epoch, model state, optimizer state, current
        learning rate and experiment path.
        """

        model_dict = self.model.state_dict() if not isinstance(self.model, list) else [model.state_dict() for model in self.model]
        optimizer_dict = self.optimizer.state_dict() if not isinstance(self.optimizer, list) else [optimizer.state_dict() for optimizer in self.optimizer] 
        lr = self.get_desire_lr() if self.model_name.lower() in ["desire"] else self.get_lr()

        state_dict = {
            'epoch': epoch,
            'model_state': model_dict ,
            'optimizer': optimizer_dict,
            'learning_rate': lr,
            'exp_path': self.exp_path,
            # 'val_ade': ade,
            # 'val_fde': fde,
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