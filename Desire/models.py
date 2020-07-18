import time

import torch
import torch.nn as nn

from Desire.model_utils import *

import pdb

class DESIRE_SGM(nn.Module):
    def __init__(self,
                 decoding_steps=6,
                 num_candidates=12):
        super(DESIRE_SGM, self).__init__()

        self.CVAE = CVAE(decoding_steps, num_candidates)

        self.decoding_steps = decoding_steps
        self.num_candidates = num_candidates

    def forward(self,
                src_trajs,
                src_lens,
                tgt_trajs,
                tgt_lens,
                decode_start_pos):
        """
        Na: Total # of agents in a batch of episodes
        Nc: # of decoding candidates
        Te: encoding timestaps
        Td: decoding timesteps

        inputs
        src_trajs: [Na x Te x 2]
        src_lens: [Na]
        tgt_trajs: [Na x Td x 2]
        tgt_lens: [Na]
        decode_start_pos: [Na]

        outputs
        y_rel: relative distance between each decoding timesteps [Td x N x Nc x 2]
        y_: final decoding trajectory [Td x N x Nc x 2]
        Hx: Trajectory encodings for each agent [1 x Na x 48]
        mu: latent mean for each agent [1 x Na x 48]
        sigma: latent covariance for each agent [1 x Na x 48]
        z: latent samples for each agent [1 x Na x 48]
        """
        total_agents = src_trajs.size(0)

        y_rel, mu, sigma, z, Hx = self.CVAE(src_trajs, src_lens, tgt_trajs, tgt_lens) # y_rel: 30 X N x Nc x 2
        y_ = decode_start_pos.reshape(1, total_agents, 1, 2) + torch.cumsum(y_rel, dim=0)

        return y_rel, y_, Hx, mu, sigma, z

    def inference(self,
                  src_trajs,
                  src_lens,
                  decode_start_pos):
        """
        Na: Total # of agents in a batch of episodes
        Nc: # of decoding candidates
        Te: encoding timestaps
        Td: decoding timesteps

        inputs
        src_trajs: [Na x Te x 2]
        src_lens: [Na]
        decode_start_pos: [Na]

        outputs
        y_rel: relative distance between each decoding timesteps [Td x N x Nc x 2]
        y_: final decoding trajectory [Td x N x Nc x 2]
        Hx: Trajectory encodings for each agent [1 x Na x 48]
        z: latent samples for each agent [1 x Na x 48]
        """
        
        total_agents = src_trajs.size(0)

        y_rel, z, Hx = self.CVAE.inference(src_trajs, src_lens) # y_rel: 30 X N x Nc x 2
        y_ = decode_start_pos.reshape(1, total_agents, 1, 2) + torch.cumsum(y_rel, dim=0)
        
        return y_rel, y_, Hx, z
    
class DESIRE_IOC(nn.Module):
    def __init__(self,
                 in_channels=3,
                 decoding_steps=6):

        super(DESIRE_IOC, self).__init__()
        self.IOC = IOC(in_channels, decoding_steps)

    def forward(self,
                velocity,
                position,
                Hx,
                scene,
                num_tgt_trajs,
                scene_encoded=False):
        """
        Na: Total # of agents in a batch of episodes
        Nb: batch size (# episodes)
        Nc: # of decoding candidates
        Ce: scene encdoing channels
        Td: decoding timesteps

        inputs
        velocity: [Td x Na x Nc x 2]
        position: [Td x Na x Nc x 2]
        Hx: [1 x Na x 48]
        scene: [Nb x Ce x (Height / 2) x (Width / 2)] if scene_encoded else [Nb x 3 x Height x Width]
        num_tgt_trajs: [Nb]

        outputs
        scores: [Td x Na x Nc x 1]
        y_delta: [Td x Na x Nc x 2]
        scene_feature: [Nb x Ce x (Height / 2) x (Width / 2)]
        """
        # SGM and IOC are two separate networks.
        # Therefore velocity, position, and Hx need be detached
        # so that the graph for the two networks are separated.
        velocity_ = velocity.detach()
        position_ = position.detach()
        Hx_ = Hx.detach()

        score, y_delta, scene_feature = self.IOC(velocity=velocity_,
                                                 position=position_,
                                                 Hx=Hx_,
                                                 scene=scene,
                                                 num_tgt_trajs=num_tgt_trajs,
                                                 scene_encoded=scene_encoded)
        
        return score, y_delta, scene_feature