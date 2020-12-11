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



class ModelTester: # GAN
    # Codes were matched based on the ModelTrainer argument order
    def __init__(self, model, data_loader,  **kwargs):
        args = kwargs.get("cfg", None)
        ploss_criterion = kwargs.get("ploss_criterion", None)
        device = kwargs.get("device", "cpu")
        self.generative = kwargs.get("generative", False)
        self.model = model
        self.data_loader = data_loader
        self.datetime = datetime.datetime.now().strftime('_%d_%B__%H_%M_')
        self.exp_path = os.path.join(args.exp_path, str(args.TAG) + '_' + self.datetime)
        if not os.path.exists(self.exp_path):
            os.mkdir(self.exp_path)

        self.beta = args.beta
        self.num_candidates = args.num_candidates
        if self.generative:
            self.decoding_steps = 4 if args.dataset == 'carla' else int(3 * args.sampling_rate)
            self.encoding_steps = 2 if args.dataset == 'carla' else int(2 * args.sampling_rate)
        else:
            self.decoding_steps = int(3 *  args.sampling_rate)

        self.model_name = args.model_name
        
        if self.generative:
            self.map_version = ''
        else:
            if self.model_name == "MATF":
                self.map_version = args.map_version

            if self.model_name in ['R2P2_SimpleRNN', 'CAM', 'CAM_NFDecoder', 'SimpleEncoderDecoder']:
                self.map_version = None
            

        if self.model_name in ["R2P2_SimpleRNN", "R2P2_RNN", "CAM_NFDecoder", "Scene_CAM_NFDecoder", "Global_Scene_CAM_NFDecoder", "AttGlobal_Scene_CAM_NFDecoder"]:
            self.flow_based_decoder = True
            self.num_candidates = args.num_candidates
        else:
            self.flow_based_decoder = False
            self.num_candidates = 1


        self.device = device
        self.render = args.test_render
                
        if args.dataset == "argoverse":
            _data_dir = '/data/datasets/datf/fl/home/spalab/argoverse_shpark/argoverse-forecasting-from-forecasting'
            self.map_file = lambda scene_id: [os.path.join(_data_dir, x[0], x[1], x[2], 'map/v1.3', x[3] ) + '.png' for x in scene_id]

        elif args.dataset == "nuscenes":
            _data_dir = '/home/spalab/nuscenes_shpark/'
            self.map_file = lambda scene_id: [os.path.join(_data_dir, x[0], x[1], x[2], 'map/v1.3', x[3]) + '.pkl' for x in scene_id]

        elif args.dataset == "carla":
            _data_dir = '/home/spalab/carla_shpark/'
            self.map_file = lambda scene_id: [os.path.join(_data_dir, x[0], x[1], x[2], 'map/v1.3', x[3] ) + '.pkl' for x in scene_id]

        # self.load_checkpoint(args.test_ckpt)

    def load_checkpoint(self, ckpt):
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state'], strict=False)

        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.start_epoch = checkpoint['epoch']
    
    def run(self):
        if self.model_name in ["SimpleEncoderDecoder", "SocialPooling", "MATF"]:
            return self.run_matf()
        elif self.model_name  in ["MATF_GAN"]:
            return self.run_gan()
        elif self.model_name  in ["R2P2_SimpleRNN", "R2P2_RNN"]:
            return self.run_r2p2()
        elif self.model_name in ["Desire"]:
            return self.run_desire()
        elif self.model_name in ["CAM", "CAM_NFDecoder", "Scene_CAM_NFDecoder", "Global_Scene_CAM_NFDecoder", "AttGlobal_Scene_CAM_NFDecoder", "Scene_CAM_NFDecoder"]:
            return self.run_proposed()
    
    def run_matf(self):
        print('Starting model test.....')    
        self.model.eval()  # Set model to evaluate mode.

        with torch.no_grad():
            epoch_loss = 0.0
            epoch_minade2, epoch_avgade2 = 0.0, 0.0
            epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
            epoch_minade3, epoch_avgade3 = 0.0, 0.0
            epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
        
            epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

            epoch_dao = 0.0
            epoch_dac = 0.0
            dao_agents = 0.0
            dac_agents = 0.0

            pool = Pool(processes=5)

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

                print("Working on test batch {:d}/{:d}... ".format(b+1, len(self.data_loader)) +
                "batch_loss: {:.2f}, ".format(batch_loss.item()) +
                "minFDE3: {:.2f}, avgFDE3: {:.2f}".format(batch_minfde3.item(), batch_avgfde3.item()), end='\r')

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
                    
                    # dao_i, dao_mask_i = self.dao(candidate_i, map_file_i)
                    # dac_i, dac_mask_i = self.dac(candidate_i, map_file_i)

                    # epoch_dao += dao_i.sum()
                    # dao_agents += dao_mask_i.sum()

                    # epoch_dac += dac_i.sum()
                    # dac_agents += dac_mask_i.sum()

                    # if self.render:
                    #     pool.apply(self.write_img_output, 
                    #                (candidate_i, past_traj_i, past_lens_i, future_traj_i, future_lens_i, map_file_i, output_file_i)
                    #     )
            
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

        # epoch_dao /= dao_agents
        # epoch_dac /= dac_agents
        
        print("--Final Performane Report--")
        print("minADE2: {:.5f}, minFDE2: {:.5f}, avgADE2: {:.5f}, avgFDE2: {:.5f}".format(epoch_minade2, epoch_minfde2, epoch_avgade2, epoch_avgfde2))
        print("minADE3: {:.5f}, minFDE3: {:.5f}, avgADE3: {:.5f}, avgFDE3: {:.5f}".format(epoch_minade3, epoch_minfde3, epoch_avgade3, epoch_avgfde3))
        # print("DAO: {:.5f}e-5, DAC: {:.5f}".format(epoch_dao * 10000.0, epoch_dac))
        with open(self.out_dir + '/metric.pkl', 'wb') as f:
            pkl.dump({"ADEs": epoch_ades,
                      "FDEs": epoch_fdes,
                    #   "DAO": epoch_dao,
                    #   "DAC": epoch_dac}, f)
                        "DAC": epoch_dac}, f)

    def run_desire(self):
        self.model.eval()
        epoch_minade2, epoch_avgade2 = 0.0, 0.0
        epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
        epoch_bestade2, epoch_bestfde2 = 0.0, 0.0
        epoch_randade2, epoch_randfde2 = 0.0, 0.0

        epoch_minade3, epoch_avgade3 = 0.0, 0.0
        epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
        epoch_bestade3, epoch_bestfde3 = 0.0, 0.0
        epoch_randade3, epoch_randfde3 = 0.0, 0.0
        
        epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

        pool = Pool(5)

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

                print("Working on test batch {:d}/{:d}... ".format(b+1, len(self.data_loader)) +
                      "minFDE3: {:.2f}, avgFDE3: {:.2f}, bestFDE3 {:.2f}, randomFDE3 {:.2f}".format(batch_minfde3.item(), batch_avgfde3.item(), batch_bestfde3.item(), batch_randfde3.item()), end='\r')

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

                # map_files = ['/home/spalab/argoverse_shpark/argoverse-forecasting-from-forecasting/' + x[0] + '/' + x[1] + '/' + x[2] + '/map/v1.3/' + x[3] + '.png' for x in scene_id]
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

                    if self.render:
                        pool.apply(self.write_img_output, (candidate_i, src_traj_i, src_lens_i, tgt_traj_i, tgt_lens_i, map_file_i, output_file_i))

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

        epoch_dao /= dao_agents
        epoch_dac /= dac_agents

        epoch_ades = [epoch_minade2, epoch_avgade2, epoch_bestade2, epoch_randade2, epoch_minade3, epoch_avgade3, epoch_bestade3, epoch_randade3]
        epoch_fdes = [epoch_minfde2, epoch_avgfde2, epoch_bestfde2, epoch_randfde2, epoch_minfde3, epoch_avgfde3, epoch_bestfde3, epoch_randfde3]

        print("--Final Performane Report--")
        print("minADE2: {:.5f}, minFDE2: {:.5f}, avgADE2: {:.5f}, avgFDE2: {:.5f}, bestADE2: {:.5f}, bestFDE2: {:.5f}, randADE2: {:.5f}, randFDE2: {:.5f}".format(epoch_minade2, epoch_minfde2, epoch_avgade2, epoch_avgfde2, epoch_bestade2, epoch_bestfde2, epoch_randade2, epoch_randfde2))
        print("minADE3: {:.5f}, minFDE3: {:.5f}, avgADE3: {:.5f}, avgFDE3: {:.5f}, bestADE3: {:.5f}, bestFDE3: {:.5f}, randADE3: {:.5f}, randFDE3: {:.5f}".format(epoch_minade3, epoch_minfde3, epoch_avgade3, epoch_avgfde3, epoch_bestade3, epoch_bestfde3, epoch_randade3, epoch_randfde3))
        print("DAO: {:.5f}e-5, DAC: {:.5f}".format(epoch_dao * 10000.0, epoch_dac))
        with open(self.out_dir + '/metric.pkl', 'wb') as f:
            pkl.dump({"ADEs": epoch_ades,
                      "FDEs": epoch_fdes,
                      "DAO": epoch_dao,
                      "DAC": epoch_dac}, f)
           
    def run_gan(self):
        print('Starting model test.....')
        self.model.eval()  # Set model to evaluate mode.
        
        with torch.no_grad():
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_adv_loss = 0.0
            epoch_minade2, epoch_avgade2 = 0.0, 0.0
            epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
            epoch_minade3, epoch_avgade3 = 0.0, 0.0
            epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
            epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

            epoch_dao = 0.0
            epoch_dac = 0.0
            dao_agents = 0.0
            dac_agents = 0.0

            pool = Pool(5)

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

            for b, batch in enumerate(self.data_loader):
                # if b > 5:
                #     break

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
                if 'GAN' in self.model_type:
                    stochastic = True
                    predicted_trajs = self.model(past_agents_traj, past_agents_traj_len, episode_idx, future_agent_masks,
                                                decode_start_vel, decode_start_pos, scene_images, stochastic, self.num_candidates)
                    predicted_trajs_ = predicted_trajs.reshape(total_future_agent, self.num_candidates, self.decoding_steps, 2)
                
                else:
                    raise ValueError("Unknown model type {:s}.".format(self.model_type))
                
 
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

                print("Working on test batch {:d}/{:d}... ".format(b+1, len(self.data_loader)) +
                "batch_loss: {:.2f}, ".format(batch_loss.item()) +
                "minFDE3: {:.2f}, avgFDE3: {:.2f}".format(batch_minfde3.item(), batch_avgfde3.item()), end='\r')

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

                torch.cuda.empty_cache()
                
                map_files = self.map_file(scene_id)
                output_files = [self.out_dir + '/' + x[2] + '.jpg' for x in scene_id]

                cum_num_future_agents = [0] + torch.cumsum(num_future_agents, dim=0).tolist()

                predicted_trajs_ = predicted_trajs_.cpu().numpy()

                past_agents_traj = past_agents_traj.cpu().numpy()
                past_agents_traj_len = past_agents_traj_len.cpu().numpy()

                future_agents_traj = future_agents_traj.cpu().numpy()
                future_agents_traj_len = future_agents_traj_len.cpu().numpy()

                for i in range(batch_size):
                    candidate_i   = predicted_trajs_[    cum_num_future_agents[i]:cum_num_future_agents[i+1]]
                    future_traj_i = future_agents_traj[cum_num_future_agents[i]:cum_num_future_agents[i+1]]
                    future_lens_i = future_agents_traj_len[ cum_num_future_agents[i]:cum_num_future_agents[i+1]]
                    past_traj_i   = past_agents_traj[  cum_num_future_agents[i]:cum_num_future_agents[i+1]]
                    past_lens_i   = past_agents_traj_len[   cum_num_future_agents[i]:cum_num_future_agents[i+1]]

                    map_file_i = map_files[i]
                    output_file_i = output_files[i]

                    dao_i, dao_mask_i = self.dao(candidate_i, map_file_i)
                    dac_i, dac_mask_i = self.dac(candidate_i, map_file_i)

                    epoch_dao += dao_i.sum()
                    dao_agents += dao_mask_i.sum()

                    epoch_dac += dac_i.sum()
                    dac_agents += dac_mask_i.sum()

                    if self.render:
                        pool.apply(self.write_img_output, 
                                   (candidate_i, past_traj_i, past_lens_i, future_traj_i, future_lens_i, map_file_i, output_file_i)
                        )

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

            epoch_dao /= dao_agents
            epoch_dac /= dac_agents

            epoch_ades = ( epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3 )
            epoch_fdes = ( epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3 )

            # return epoch_g_loss+epoch_d_loss, epoch_g_loss, epoch_d_loss, epoch_ades, epoch_fdes, scheduler_metric


        print("--Final Performane Report--")
        print("minADE2: {:.5f}, minFDE2: {:.5f}, avgADE2: {:.5f}, avgFDE2: {:.5f}".format(epoch_minade2, epoch_minfde2, epoch_avgade2, epoch_avgfde2))
        print("minADE3: {:.5f}, minFDE3: {:.5f}, avgADE3: {:.5f}, avgFDE3: {:.5f}".format(epoch_minade3, epoch_minfde3, epoch_avgade3, epoch_avgfde3))
        print("DAO: {:.5f}e-5, DAC: {:.5f}".format(epoch_dao * 10000.0, epoch_dac))
        with open(self.out_dir + '/metric.pkl', 'wb') as f:
            pkl.dump({"ADEs": epoch_ades,
                      "FDEs": epoch_fdes,
                      "DAO": epoch_dao,
                      "DAC": epoch_dac}, f)
            
    def run_proposed(self):
        print('Starting model test.....')
        self.model.eval()  # Set model to evaluate mode.
        
        epoch_loss = 0.0
        epoch_qloss = 0.0
        epoch_ploss = 0.0
        epoch_minade2, epoch_avgade2 = 0.0, 0.0
        epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
        epoch_minade3, epoch_avgade3 = 0.0, 0.0
        epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
        epoch_minmsd, epoch_avgmsd = 0.0, 0.0
        epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

        epoch_dao = 0.0
        epoch_dac = 0.0
        dao_agents = 0.0
        dac_agents = 0.0

        pool = Pool(5)

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
            for b, batch in enumerate(self.data_loader):
                
                # if b > 5:
                #     break

                scene_images, log_prior, \
                agent_masks, \
                num_src_trajs, src_trajs, src_lens, src_len_idx, \
                num_tgt_trajs, tgt_trajs, tgt_lens, tgt_len_idx, \
                tgt_two_mask, tgt_three_mask, \
                decode_start_vel, decode_start_pos, scene_id = batch

                # Detect dynamic batch size
                batch_size = scene_images.size(0)
                # num_encoding_agents = src_trajs.size(0)
                # num_decoding_agents = tgt_trajs.size(0)
                num_three_agents = torch.sum(tgt_three_mask)

                if self.map_version == '2.0':
                    coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                    distance_batch = distance.repeat(batch_size, 1, 1, 1)
                    scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)
                
                src_trajs = src_trajs.to(self.device)
                src_lens = src_lens.to(self.device)

                tgt_trajs = tgt_trajs.to(self.device)[tgt_three_mask]
                tgt_lens = tgt_lens.to(self.device)[tgt_three_mask]

                num_tgt_trajs = num_tgt_trajs.to(self.device)
                episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_tgt_trajs)[tgt_three_mask]

                agent_masks = agent_masks.to(self.device)
                agent_tgt_three_mask = torch.zeros_like(agent_masks)
                agent_masks_idx = torch.arange(len(agent_masks), device=self.device)[agent_masks][tgt_three_mask]
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
                    perterb = torch.normal(mean=0.0, std=np.sqrt(0.001), size=tgt_trajs.shape, device=self.device)
                    
                    if self.model_type == 'R2P2_SimpleRNN':
                        z_, mu_, sigma_, motion_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, decode_start_vel, decode_start_pos)

                    elif self.model_type == 'R2P2_RNN':
                        z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, episode_idx, decode_start_vel, decode_start_pos, scene_images)

                    elif self.model_type == 'CAM_NFDecoder':
                        z_, mu_, sigma_, motion_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, src_lens, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_src_trajs)

                    elif self.model_type == 'Scene_CAM_NFDecoder':
                        z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, src_lens, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_src_trajs, scene_images)

                    elif self.model_type == 'Global_Scene_CAM_NFDecoder':
                        z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, src_lens, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_src_trajs, scene_images)

                    elif self.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
                        z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, src_lens, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_src_trajs, scene_images)

                    z_ = z_.reshape((num_three_agents, -1)) # A X (Td*2)
                    log_q0 = c1 - 0.5 * ((z_ ** 2).sum(dim=1))

                    logdet_sigma = log_determinant(sigma_)

                    log_qpi = log_q0 - logdet_sigma.sum(dim=1)
                    qloss = -log_qpi
                    batch_qloss = qloss.mean()

                    # Prior Loss (p loss)
                    if self.model_type == 'R2P2_SimpleRNN':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, decode_start_vel, decode_start_pos, motion_encoded=True)

                    elif self.model_type == 'R2P2_RNN':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, episode_idx, decode_start_vel, decode_start_pos, scene_encoding_, motion_encoded=True, scene_encoded=True)

                    elif self.model_type == 'CAM_NFDecoder':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, src_lens, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_src_trajs, agent_encoded=True)

                    elif self.model_type == 'Scene_CAM_NFDecoder':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, src_lens, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_src_trajs, scene_encoding_, agent_encoded=True, scene_encoded=True)

                    elif self.model_type == 'Global_Scene_CAM_NFDecoder':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, src_lens, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_src_trajs, scene_encoding_, agent_encoded=True, scene_encoded=True)

                    elif self.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, src_lens, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_src_trajs, scene_encoding_, agent_encoded=True, scene_encoded=True)
                    if self.beta != 0.0:
                        if self.ploss_type == 'mseloss':
                            ploss = self.ploss_criterion(gen_trajs, tgt_trajs)
                        else:
                            ploss = self.ploss_criterion(episode_idx, gen_trajs, log_prior, -15.0)

                    else:
                        ploss = torch.zeros(size=(1,), device=self.device)
                    batch_ploss = ploss.mean()
                    batch_loss = batch_qloss + self.beta * batch_ploss

                    epoch_ploss += batch_ploss.item() * batch_size
                    epoch_qloss += batch_qloss.item() * batch_size   

                else:

                    if 'CAM' == self.model_type:
                        gen_trajs = self.model(src_trajs, src_lens, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_src_trajs)                                                    

                    gen_trajs = gen_trajs.reshape(num_three_agents, self.num_candidates, self.decoding_steps, 2)


                rs_error3 = ((gen_trajs - tgt_trajs.unsqueeze(1)) ** 2).sum(dim=-1).sqrt_()
                rs_error2 = rs_error3[..., :int(self.decoding_steps*2/3)]
                
                
                diff = gen_trajs - tgt_trajs.unsqueeze(1)
                msd_error = (diff[:,:,:,0] ** 2 + diff[:,:,:,1] ** 2)
                
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
                
                
                msd = msd_error.mean(-1)
                minmsd, _ = msd.min(dim=-1)
                avgmsd = msd.mean(dim=-1)
                batch_minmsd = minmsd.mean()
                batch_avgmsd = avgmsd.mean()


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

                print("Working on test batch {:d}/{:d}... ".format(b+1, len(self.data_loader)) +
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
                

                epoch_minmsd += batch_minmsd.item() * num_agents3
                epoch_avgmsd += batch_avgmsd.item() * num_agents3

                epoch_agents += num_agents
                epoch_agents2 += num_agents2
                epoch_agents3 += num_agents3
                
                map_files = self.map_file(scene_id)
                output_files = [self.out_dir + '/' + x[2] + '_' + x[3] + '.jpg' for x in scene_id]

                cum_num_tgt_trajs = [0] + torch.cumsum(num_tgt_trajs, dim=0).tolist()
                cum_num_src_trajs = [0] + torch.cumsum(num_src_trajs, dim=0).tolist()

                ## Currently, for CAM model, gen_traj and src_traj does not shown well.

                src_trajs = src_trajs.cpu().numpy()
                src_lens = src_lens.cpu().numpy()

                tgt_trajs = tgt_trajs.cpu().numpy()
                tgt_lens = tgt_lens.cpu().numpy()

                zero_ind = np.nonzero(tgt_three_mask.numpy() == 0)[0]
                zero_ind -= np.arange(len(zero_ind))

                tgt_three_mask = tgt_three_mask.numpy()
                agent_tgt_three_mask = agent_tgt_three_mask.cpu().numpy()

                gen_trajs = gen_trajs.cpu().numpy()

                src_mask = agent_tgt_three_mask

                gen_trajs = np.insert(gen_trajs, zero_ind, 0, axis=0)

                tgt_trajs = np.insert(tgt_trajs, zero_ind, 0, axis=0)
                tgt_lens = np.insert(tgt_lens, zero_ind, 0, axis=0)

                for i in range(batch_size):
                    candidate_i = gen_trajs[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]
                    tgt_traj_i = tgt_trajs[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]
                    tgt_lens_i = tgt_lens[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]

                    src_traj_i = src_trajs[cum_num_src_trajs[i]:cum_num_src_trajs[i+1]]
                    src_lens_i = src_lens[cum_num_src_trajs[i]:cum_num_src_trajs[i+1]]
                    map_file_i = map_files[i]
                    output_file_i = output_files[i]
                    
                    candidate_i = candidate_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]]
                    tgt_traj_i = tgt_traj_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]]
                    tgt_lens_i = tgt_lens_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]]

                    src_traj_i = src_traj_i[agent_tgt_three_mask[cum_num_src_trajs[i]:cum_num_src_trajs[i+1]]]
                    src_lens_i = src_lens_i[agent_tgt_three_mask[cum_num_src_trajs[i]:cum_num_src_trajs[i+1]]]

                    # dao_i, dao_mask_i = self.dao(candidate_i, map_file_i)
                    # dac_i, dac_mask_i = self.dac(candidate_i, map_file_i)

                    # epoch_dao += dao_i.sum()
                    # dao_agents += dao_mask_i.sum()

                    # epoch_dac += dac_i.sum()
                    # dac_agents += dac_mask_i.sum()

                    # if self.render:
                    #     pool.apply(self.write_img_output, (candidate_i, src_traj_i, src_lens_i, tgt_traj_i, tgt_lens_i, map_file_i, output_file_i))
        
                # break

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

        epoch_minmsd /= epoch_agents3
        epoch_avgmsd /= epoch_agents3

        # epoch_dao /= dao_agents
        # epoch_dac /= dac_agents

        epoch_ades = ( epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3 )
        epoch_fdes = ( epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3 )

        print("--Final Performane Report--")
        print("minADE2: {:.5f}, minFDE2: {:.5f}, avgADE2: {:.5f}, avgFDE2: {:.5f}".format(epoch_minade2, epoch_minfde2, epoch_avgade2, epoch_avgfde2))
        print("minADE3: {:.5f}, minFDE3: {:.5f}, avgADE3: {:.5f}, avgFDE3: {:.5f}".format(epoch_minade3, epoch_minfde3, epoch_avgade3, epoch_avgfde3))
        print("minMSD: {:.5f}, avgMSD: {:5f}".format(epoch_minmsd, epoch_avgmsd))
        print("QLoss: {:.5f}, PLoss: {:5f}".format(epoch_qloss, epoch_ploss))
        # print("DAO: {:.5f}e-5, DAC: {:.5f}".format(epoch_dao * 10000.0, epoch_dac))
        with open(self.out_dir + '/metric.pkl', 'wb') as f:
            pkl.dump({"ADEs": epoch_ades,
                      "FDEs": epoch_fdes,
                      "Qloss": epoch_qloss,
                      "Ploss": epoch_ploss, 
                    #   "DAO": epoch_dao,
                      }, f)
 
    def run_r2p2(self):
        print('Starting model test.....')
        self.model.eval()

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
            
            epoch_loss = 0.0
            epoch_qloss = 0.0
            epoch_ploss = 0.0
            epoch_minade2, epoch_avgade2 = 0.0, 0.0
            epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
            epoch_minade3, epoch_avgade3 = 0.0, 0.0
            epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
            epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0
            
            epoch_dao = 0.0
            epoch_dac = 0.0
            dao_agents = 0.0
            dac_agents = 0.0

            pool = Pool(5)

            c1 = -self.decoding_steps * np.log(2 * np.pi)
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
                
                src_trajs = src_trajs.to(self.device)[agent_masks][tgt_three_mask]
                tgt_trajs = tgt_trajs.to(self.device)[tgt_three_mask]
                
                decode_start_vel = decode_start_vel.to(self.device)[agent_masks][tgt_three_mask]
                decode_start_pos = decode_start_pos.to(self.device)[agent_masks][tgt_three_mask]

                num_tgt_trajs = num_tgt_trajs.to(self.device)
                
                log_prior = log_prior.to(self.device)

                # Total number of three-masked agents in this batch
                episode_idx = torch.arange(len(num_tgt_trajs), device=self.device).repeat_interleave(num_tgt_trajs)
                episode_idx = episode_idx[tgt_three_mask]
                total_three_agents = episode_idx.size(0)

                # Normalizing Flow (q loss)
                # z: Na X (Td*2)
                # mu: Na X Td X 2
                # sigma: Na X Td X 2 X 2

                # Generate perturbation
                perterb = torch.normal(mean=0.0, std=np.sqrt(0.001), size=tgt_trajs.shape, device=self.device)

                if self.model_type == 'R2P2_SimpleRNN':
                    z_, mu_, sigma_, motion_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, episode_idx, decode_start_vel, decode_start_pos)
                elif self.model_type == 'R2P2_RNN':
                    z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, episode_idx, decode_start_vel, decode_start_pos, scene_images)

                log_q0 = c1 - 0.5 * ((z_ ** 2).sum(dim=1))

                logdet_sigma = log_determinant(sigma_)

                log_qpi = log_q0 - logdet_sigma.sum(dim=1)
                qloss = -log_qpi
                batch_qloss = qloss.mean()

                # Prior Loss (p loss)
                if self.model_type == 'R2P2_SimpleRNN':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, episode_idx, decode_start_vel, decode_start_pos, motion_encoded=True)
                elif self.model_type == 'R2P2_RNN':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, episode_idx, decode_start_vel, decode_start_pos, scene_encoding_, motion_encoded=True, scene_encoded=True)

                if self.beta != 0.0:
                    if self.ploss_type == 'mseloss':
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

                print("Working on testing batch {:d}/{:d}... ".format(b+1, len(self.data_loader)) +
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

                # partition, sub_partition, scenario, frame = scene_id
                # map_files = ['/home/spalab/argoverse_shpark/argoverse-forecasting-from-forecasting/' + x[0] + '/' + x[1] + '/' + x[2] + '/map/v1.3/' + x[3] + '.png' for x in zip(partition, sub_partition, scenario, frame)]
                # output_files = [self.out_dir + '/' + x + '.jpg' for x in scenario]

                tgt_three_mask = tgt_three_mask.cpu().numpy()

                map_files = self.map_file(scene_id)
                output_files = [self.out_dir + '/' + x[2] + '.jpg' for x in scene_id]

                cum_num_tgt_trajs = [0] + torch.cumsum(num_tgt_trajs, dim=0).tolist()

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

                    if self.render:
                        pool.apply(self.write_img_output, (candidate_i, src_traj_i, src_lens_i, tgt_traj_i, tgt_lens_i, map_file_i, output_file_i))
            
            
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

        epoch_dao /= dao_agents
        epoch_dac /= dac_agents

        epoch_ades = [epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3]
        epoch_fdes = [epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3]

        print("--Final Performane Report--")
        print("minADE2: {:.5f}, minFDE2: {:.5f}, avgADE2: {:.5f}, avgFDE2: {:.5f}".format(epoch_minade2, epoch_minfde2, epoch_avgade2, epoch_avgfde2))
        print("minADE3: {:.5f}, minFDE3: {:.5f}, avgADE3: {:.5f}, avgFDE3: {:.5f}".format(epoch_minade3, epoch_minfde3, epoch_avgade3, epoch_avgfde3))
        print("QLoss: {:.5f}, PLoss: {:5f}".format(epoch_qloss, epoch_ploss))
        print("DAO: {:.5f}e-5, DAC: {:.5f}".format(epoch_dao * 10000.0, epoch_dac))
        
        with open(self.out_dir + '/metric.pkl', 'wb') as f:
            pkl.dump({"ADEs": epoch_ades,
                      "FDEs": epoch_fdes,
                      "Qloss": epoch_qloss,
                      "Ploss": epoch_ploss,
                      "DAO": epoch_dao,
                      "DAC": epoch_dac}, f)


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
        dac_mask = []
        for i in range(num_agents):
            gen_trajs_i = gen_trajs[i]
            oom_mask = np.any(np.abs(gen_trajs_i) > 56, axis=-1)
            oom_ratio = oom_mask.sum(axis=-1) / decoding_timesteps
            if oom_ratio.mean() >= 0.5:
                dac.append(0.0)
                dac_mask.append(False)
            
            else:
                inmap_mask = np.logical_not(oom_mask)
                count = 0
                for k in range(num_candidates):
                    gen_trajs_ik = gen_trajs_i[k]
                    gen_trajs_ik = gen_trajs_ik[inmap_mask[k]]

                    x, y = ((gen_trajs_ik + 56) * 2).astype(np.int64).T
                    da_ik = np.all(da_mask[x, y])
                    if da_ik:
                        count += 1
                
                dac.append(count / num_candidates)
                dac_mask.append(True)

        return np.array(dac), np.array(dac_mask)

    @staticmethod
    def dao(gen_trajs, map_file):
        if '.png' in map_file:
            map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)

        elif '.pkl' in map_file:
            with open(map_file, 'rb') as pnt:
                map_array = pkl.load(pnt)

        da_mask = np.any(map_array > 0, axis=-1)

        num_agents, num_candidates, decoding_timesteps = gen_trajs.shape[:3]
        
        dao = []
        dao_mask = []
        for i in range(num_agents):
            gen_trajs_i = gen_trajs[i]
            oom_mask = np.any(np.abs(gen_trajs_i) > 56, axis=-1)
            oom_ratio = oom_mask.sum(axis=-1) / decoding_timesteps
            if oom_ratio.mean() >= 0.5:
                dao.append(0.0)
                dao_mask.append(False)
            
            else:
                inmap_mask = np.logical_not(oom_mask)

                gen_trajs_i = gen_trajs_i[inmap_mask]
                multi_idx = ((gen_trajs_i + 56) * 2).astype(np.int64).T

                linear_idx = np.ravel_multi_index(multi_idx, (224, 224))
                linear_idx = np.unique(linear_idx)

                x, y = np.unravel_index(linear_idx, (224, 224))
                da = da_mask[x, y]

                dao.append(da.sum() / da_mask.sum())
                dao_mask.append(True)

        return np.array(dao), np.array(dao_mask)

    @staticmethod
    def write_img_output(gen_trajs, src_trajs, src_lens, tgt_trajs, tgt_lens, map_file, output_file):
        """abcd"""
        if '.png' in map_file:
            map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)
            map_array = cv2.cvtColor(map_array, cv2.COLOR_BGR2RGB)

        elif '.pkl' in map_file:
            with open(map_file, 'rb') as pnt:
                map_array = pkl.load(pnt)

        # with open(map_file, 'rb') as f:
        #     map_array = pkl.load(f)

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