
import torch
import torch.nn as nn
from R2P2_MA.model_utils import *

import pdb

class R2P2_CNN(nn.Module):
    """
    R2P2 CNN Model
    """
    def __init__(self, in_channels=3):
        super(R2P2_CNN, self).__init__()
        self.conv_modules = nn.ModuleDict({
            'conv1': nn.Conv2d(in_channels, 32, 3, padding=0, dilation=1),
            'conv2': nn.Conv2d(32, 32, 3, padding=0, dilation=1),
            'conv3': nn.Conv2d(32, 32, 3, padding=0, dilation=1),
            'conv4': nn.Conv2d(32, 32, 3, padding=0, dilation=1),
            'conv5': nn.Conv2d(32, 32, 3, padding=2, dilation=2),
            'conv6': nn.Conv2d(32, 32, 3, padding=4, dilation=4),
            'conv7': nn.Conv2d(32, 32, 3, padding=8, dilation=8),
            'conv8': nn.Conv2d(32, 32, 3, padding=4, dilation=4),
            'conv9': nn.Conv2d(32, 32, 3, padding=2, dilation=2),
            'conv10': nn.Conv2d(32, 32, 3, padding=0, dilation=1),
            'conv11': nn.Conv2d(32, 32, 3, padding=0, dilation=1),
            'conv12': nn.Conv2d(32, 32, 3, padding=0, dilation=1),
            'conv13': nn.Conv2d(32, 6, 1)
          })
        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='bilinear')
        self.activations = nn.ModuleDict({
              'softplus': nn.Softplus(),
              'tanh': nn.Tanh()
          })

    def forward(self, x):
        '''
        input shape
        x: B X Ci X 64 X 64

        ouput shape
        final_output: B X 6 X 100 X 100
        hidden: [
            B X 32 X 62 X 62,
            B X 32 X 60 X 60,
            B X 32 X 58 X 58,
            B X 32 X 56 X 56,
            B X 32 X 56 X 56,
            B X 32 X 56 X 56,
            B X 32 X 56 X 56,
            B X 32 X 56 X 56,
            B X 32 X 56 X 56,
            B X 32 X 54 X 54,
            B X 32 X 52 X 52,
            B X 32 X 50 X 50,
            B X 6 X 50 X 50,
            B X 6 X 100 X 100
          ] Conv1~Conv13 intermediate states
        '''

        conv_modules = self.conv_modules
        activations = self.activations

        hidden = []

        # Conv 1~10 with softplus
        for i in range(1, 11):
            x = conv_modules['conv{:d}'.format(i)](x)
            x = activations['softplus'](x)
            hidden.append(x)

        # Conv 11~12 with tanh
        for i in range(11, 13):
            x = conv_modules['conv{:d}'.format(i)](x)
            x = activations['tanh'](x)
            hidden.append(x)
          
        # Conv 13 (Linear 1X1)
        x = conv_modules['conv13'](x)
        hidden.append(x)

        # Upsample
        final_output = self.upsample(x)
        hidden.append(final_output)

        return final_output, hidden


class R2P2_SimpleRNN(nn.Module):
    """
    R2P2 Simple RNN Model
    """
    def __init__(self,
                velocity_const,
                num_candidates=12,
                decoding_steps=6,
                ):
        super(R2P2_SimpleRNN, self).__init__()

        self.motion_encoder = MotionEncoder()
        self.dynamic_decoder = DynamicDecoder(decoding_steps=decoding_steps, velocity_const=velocity_const)
        self.num_candidates = num_candidates
        self.decoding_steps = decoding_steps

        self.mlp = nn.Sequential(
        nn.Linear(150, 50),
        nn.Softplus(),
        nn.Linear(50, 50),
        nn.Softplus(),
        )

    def forward(self, src_trajs_or_src_encoding, episode_idx, decode_start_vel, decode_start_pos, motion_encoded=False):
        """
        Na: total # of agents
        Nc: # candidates
        Te: encoding timestaps
        Td: decoding timesteps
        D: past motion encoding dimension

        inputs
        src_trajs_or_src_encoding: Past observation or its encoding,
          [Na X Te X 2] or [Na X D]
        episode_idx: Agent-episode matching index, [Na]
        decode_start_vel: Agents' velocity at present, [Na X 2]
        decode_start_pos: Agents' position at present, [Na X 2]
        
        outputs
        x: Predicting trajectory for the agents, [Na X Nc X Td X 2]
        z: Latent states for the agents, [Na X Nc X (Td*2)]
        mu: Predicting mu for the agents, [Na X Nc X Td X 2]
        sigma: Predicting sigma for the agenets, [Na X Nc X Td X 2 X 2]
        """
        total_agents = src_trajs_or_src_encoding.size(0)
        device = src_trajs_or_src_encoding.device	

        if motion_encoded:
            motion_encoding = src_trajs_or_src_encoding	
        else:	
            src_trajs = src_trajs_or_src_encoding.transpose(1, 0)	
            motion_encoding = self.motion_encoder(src_trajs)	

        context_encoding = self.mlp(motion_encoding)	
        x = []	
        mu = []	
        sigma = []	
        z = torch.normal(mean=0.0, std=1.0, size=(total_agents*self.num_candidates, self.decoding_steps*2), device=device)	
        	
        context_encoding = context_encoding.repeat_interleave(self.num_candidates, dim=0)	
        decode_start_vel = decode_start_vel.repeat_interleave(self.num_candidates, dim=0)	
        decode_start_pos = decode_start_pos.repeat_interleave(self.num_candidates, dim=0)	

        x_flat = torch.zeros_like(z)	
        x_prev = decode_start_pos	
        dx = decode_start_vel	
        h = None	
        for i in range(self.decoding_steps):	
            z_t = z[:, i*2:(i+1)*2]	
            x_flat[:, i*2:(i+1)*2] = x_prev	
            x_t, mu_t, sigma_t, h = self.dynamic_decoder(z_t, x_flat, h, context_encoding, dx, x_prev)	

            x.append(x_t)	
            mu.append(mu_t)	
            sigma.append(sigma_t)	

            dx = x_t - x_prev	
            x_prev = x_t	
            x_flat = x_flat.clone()	
            
        x = torch.stack(x, dim=1).reshape(total_agents, self.num_candidates, self.decoding_steps, 2) # x: Na X Nc X Td X 2	
        z = z.reshape(total_agents, self.num_candidates, self.decoding_steps*2)	
        mu = torch.stack(mu, dim=1).reshape(total_agents, self.num_candidates, self.decoding_steps, 2) # mu: Na X Nc X Td X 2	
        sigma = torch.stack(sigma, dim=1).reshape(total_agents, self.num_candidates, self.decoding_steps, 2, 2) # sigma: Na X Nc X Td X 2 X 2	
        return x, z, mu, sigma	
    
    def infer(self, tgt_trajs, src_trajs, episode_idx, decode_start_vel, decode_start_pos):
        """
        Na: total # of agents
        Nb: batch size (# episodes)
        Te: encoding timestaps
        Td: decoding timesteps
        D: past motion encoding dimension

        inputs
        tgt_trajs: Ground truth future, [Na X Td X 2]
        src_trajs: Past observation, [Na X Te X 2]
        episode_idx: Agent-episode matching index, [Na]
        decode_start_vel: Agents' velocity at present, [Na X 2]
        decode_start_pos: Agents' position at present, [Na X 2]
        
        outputs
        z: Latent state for the agents, [Na X (Td*2)]
        mu: mu for the agents, [Na X Td X 2]
        sigma: sigma for the agents, [Na X Td X 2 X 2]
        motion_encoding: past motion encoding, [Na X D]
        """
        src_trajs = src_trajs.transpose(1, 0) # [Te X A X 2]
        motion_encoding = self.motion_encoder(src_trajs) # (A X D)
        
        # Expand motion encdoing for unrollig time
        context_encoding = self.mlp(motion_encoding)
        context_encoding = context_encoding.unsqueeze(dim=1)
        context_encoding = context_encoding.expand(-1, self.decoding_steps, -1) # [A X Td X 150]
 
        z, mu, sigma = self.dynamic_decoder.infer(tgt_trajs, context_encoding, decode_start_vel, decode_start_pos)
        return z, mu, sigma, motion_encoding


class R2P2_RNN(nn.Module):
    """
    R2P2_RNN Model
    """
    def __init__(self,
                 scene_channels,
                 velocity_const,
                 num_candidates=12,
                 decoding_steps=6
                ):

        super(R2P2_RNN, self).__init__()

        self.cnn_model = R2P2_CNN(in_channels=scene_channels)
        self.motion_encoder = MotionEncoder()
        self.context_fusion = ContextFusion()
        self.dynamic_decoder = DynamicDecoder(decoding_steps=decoding_steps, velocity_const=velocity_const)

        self.interpolator = Bilinear_Interpolation()
        self.num_candidates = num_candidates
        self.decoding_steps = decoding_steps

    def forward(self, src_trajs_or_src_encoding, episode_idx, decode_start_vel, decode_start_pos, scene_or_scene_encoding, motion_encoded=False, scene_encoded=False):
        """
        Na: total # of agents
        Nb: batch size (# episodes)
        Nc: # candidates
        Ci: scene input channels
        Ce: scene encdoing channels
        Te: encoding timestaps
        Td: decoding timesteps
        D: past motion encoding dimension

        inputs
        src_trajs_or_src_encoding: Past observation or its encoding,
          [Na X Te X 2] or [Na X D]
        episode_idx: Agent-episode matching index, [Na]
        decode_start_vel: Agents' velocity at present, [Na X 2]
        decode_start_pos: Agents' position at present, [Na X 2]
        scene: Scene context or its encoding,
          [Nb X Ci X H X W] or [Nb X Ce X H X W]
        
        outputs
        x: Predicting trajectory for the agents, [Na X Nc X Td X 2]
        z: Latent states for the agents, [Na X Nc X (Td*2)]
        mu: Predicting mu for the agents, [Na X Nc X Td X 2]
        sigma: Predicting sigma for the agenets, [Na X Nc X Td X 2 X 2]
        """

        total_agents = src_trajs_or_src_encoding.size(0)
        device = src_trajs_or_src_encoding.device

        if scene_encoded:
            scene_encoding = scene_or_scene_encoding
        else:
            scene_encoding, _ = self.cnn_model(scene_or_scene_encoding)
        
        if motion_encoded:
            motion_encoding = src_trajs_or_src_encoding
        else:
            src_trajs = src_trajs_or_src_encoding.transpose(1, 0)
            motion_encoding = self.motion_encoder(src_trajs)

        x = []
        mu = []
        sigma = []

        # Make latent states and repeat inputs for the # of candidates
        # z: [(Na*Nc), (Td*2)]
        # episode_idx: [Na*Nc]
        # motion_encoding: [(Na*Nc) X D]
        # decode_start_vel: [(Na*Nc) X 2]
        # decode_start_pos: [(Na*Nc) X 2]
        z = torch.normal(mean=0.0, std=1.0, size=(total_agents*self.num_candidates, self.decoding_steps*2), device=device)
        episode_idx = episode_idx.repeat_interleave(self.num_candidates)
        motion_encoding = motion_encoding.repeat_interleave(self.num_candidates, dim=0)
        decode_start_vel = decode_start_vel.repeat_interleave(self.num_candidates, dim=0)
        decode_start_pos = decode_start_pos.repeat_interleave(self.num_candidates, dim=0)

        x_flat = torch.zeros_like(z)
        x_prev = decode_start_pos
        dx = decode_start_vel
        h = None

        for i in range(self.decoding_steps):
            z_t = z[:, i*2:(i+1)*2]
            x_flat[:, i*2:(i+1)*2] = x_prev
            interpolated_feature, _ = self.interpolator(episode_idx, x_prev, scene_encoding, 0.0) # [(Na*Nc) X Ce]

            context_encoding, _ = self.context_fusion(motion_encoding, interpolated_feature) # [Na X (Ce+D)]

            x_t, mu_t, sigma_t, h = self.dynamic_decoder(z_t, x_flat, h, context_encoding, dx, x_prev)

            x.append(x_t)
            mu.append(mu_t)
            sigma.append(sigma_t)

            dx = x_t - x_prev
            x_prev = x_t
            x_flat = x_flat.clone()

        x = torch.stack(x, dim=1).reshape(total_agents, self.num_candidates, self.decoding_steps, 2) # x: [Na X Nc X Td X 2]
        z = z.reshape(total_agents, self.num_candidates, self.decoding_steps*2) # z: [Na X Nc X (Td*2)]
        mu = torch.stack(mu, dim=1).reshape(total_agents, self.num_candidates, self.decoding_steps, 2) # mu: [Na X Nc X Td X 2]
        sigma = torch.stack(sigma, dim=1).reshape(total_agents, self.num_candidates, self.decoding_steps, 2, 2) # sigma: [Na X Nc X Td X 2 X 2]

        return x, z, mu, sigma
    	
    def infer(self, tgt_trajs, src_trajs, episode_idx, decode_start_vel, decode_start_pos, scene):
        """
        Na: total # of agents
        Nb: batch size (# episodes)
        Ci: scene input channels
        Ce: scene encdoing channels
        Te: encoding timestaps
        Td: decoding timesteps
        D: past motion encoding dimension

        inputs
        tgt_trajs: Ground truth future, [Na X Td X 2]
        src_trajs: Past observation, [Na X Te X 2]
        episode_idx: Agent-episode matching index, [Na]
        decode_start_vel: Agents' velocity at present, [Na X 2]
        decode_start_pos: Agents' position at present, [Na X 2]
        scene: Scene context, [Nb X Ci X H X W]
        
        outputs
        z: Latent state for the agents, [Na X (Td*2)]
        mu: mu for the agents, [Na X Td X 2]
        sigma: sigma for the agents, [Na X Td X 2 X 2]
        motion_encoding: past motion encoding, [Na X D]
        scene_encoding: scene enocding, [B X Ce X H x W]
        """
        total_agents = tgt_trajs.size(0)
        decoding_timesteps = tgt_trajs.size(1)

        scene_encoding, _ = self.cnn_model(scene) # [B X Ce X H x W]
        src_trajs = src_trajs.transpose(1, 0) # transpose to time-major format
        motion_encoding = self.motion_encoder(src_trajs) # [Na X D]
        	
        init_pos = decode_start_pos.unsqueeze(1) # Initial position, [Na X 1 X 2]
        prev_pos = tgt_trajs[:, :-1, :] # Future positions, [Na X (Td-1) X 2] 	
        interp_pos = torch.cat((init_pos, prev_pos), dim=1) # [Na X Td X 2]

        # Merge agent & time dimensions then generate duplicated episode idx
        # batch_idx_array = [0,0,..,0,1,1,...,1,2,2,...,2,...,Na-1,Na-1,...,Na-1]
        interp_pos = interp_pos.reshape(-1, 2)
        episode_idx = episode_idx.repeat_interleave(decoding_timesteps) # [Na*Td]
        
        interpolated_feature, _ = self.interpolator(episode_idx, interp_pos, scene_encoding, 0.0) # [(Na*Td) X Ce]	
        interpolated_feature = interpolated_feature.reshape(total_agents, decoding_timesteps, -1) # [Na X Td X Ce]
        
        # Expand motion encdoing for unrollig time	
        motion_encoding_ = motion_encoding.unsqueeze(dim=1)	
        motion_encoding_ = motion_encoding_.expand(-1, self.decoding_steps, -1) # [A X Td X 150]	
        	
        context_encoding, _  = self.context_fusion(motion_encoding_, interpolated_feature) # [A X Td X 50]	
        	
        z, mu, sigma = self.dynamic_decoder.infer(tgt_trajs, context_encoding, decode_start_vel, decode_start_pos)	
        return z, mu, sigma, motion_encoding, scene_encoding