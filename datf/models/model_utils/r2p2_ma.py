import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class Interpolated_Ploss(nn.Module):
    """
    Interpolated PLoss
    """
    def __init__(self, scene_size=100):
        super(Interpolated_Ploss, self).__init__()
        self.interpolator = Bilinear_Interpolation(scene_size=scene_size)
    
    def forward(self, episode_idx, pred_traj, log_prior, oom_val=-15.0):
        total_agents = pred_traj.size(0)
        num_candidates = pred_traj.size(1)
        decoding_steps = pred_traj.size(2)
        
        log_prior = log_prior.unsqueeze(dim=1) # Add dummy channel dimension

        # Merge agent-candidate-time dimensions then repeat episode_idx
        pred_traj = pred_traj.reshape(total_agents*num_candidates*decoding_steps, 2)
        episode_idx = episode_idx.repeat_interleave(num_candidates).repeat_interleave(decoding_steps)

        log_prior_interp, _ = self.interpolator(episode_idx, pred_traj, log_prior, oom_val)
        
        ploss = -log_prior_interp.squeeze().reshape(total_agents, num_candidates, decoding_steps)
        ploss = ploss.sum(2).mean(1)

        return ploss
        
class MSE_Ploss(nn.Module):
    """
    MSE_Ploss
    """
    def __init__(self):
        super(MSE_Ploss, self).__init__()
    
    def forward(self, pred_traj, tgt_traj):
        '''
        pred_traj: [N x C x T x 2]
        tgt_traj: [N x T x 2]
        '''
        if len(pred_traj.size()) != 4:
            raise ValueError()
        if len(tgt_traj.size()) != 3:
            raise ValueError()

        ploss = ((pred_traj - tgt_traj.unsqueeze(1)) ** 2).sum((2,3)).mean(1)
        
        return ploss

class Bilinear_Interpolation(nn.Module):
    def __init__(self, scene_size=100):
        super(Bilinear_Interpolation, self).__init__()
        self.scene_size = 100

    def forward(self, episode_idx, positions, grid_map, padding_val=0.0):
        """
        Na: Total # of agents (possibly repeated by timesteps)
        Nb: batch size (# episodes)
        Ce: scene encdoing channels

        inputs
        episode_idx: Agent-episode matching index, [Na]
        positions : Agents' location in ego coordinate system, [Na X 2]
        grid_map: 2D feature map to perform bilinear interpolation, [Nb X Ce X 100 X 100]
        padding_val: The grid_map's edge is padded by this value. This handles "out-of-map" positions.

        outputs
        map_interp: Bilinear interpolated grid_map, [Na X Ce]
        positions_map: Agents' location in map coordinate system, [Na X 2]
        """
        # Detect total agents
        total_agents = positions.size(0)

        # Pad the feature_map with padding_val
        pad = (1, 1, 1, 1)
        grid_map_ = F.pad(grid_map, pad, mode='constant', value=padding_val) # [Na X Ce X 102 X 102]

        # Coordinate transfrom to the map coordinate system
        positions_map = (positions + 56.0)/112.0 * 100.0 + 1.0
        
        x, y = torch.split(positions_map, 1, dim=-1)

        # Qunatize x and y
        floor_positions_map = torch.floor(positions_map)
        ceil_positions_map = torch.ceil(positions_map)

        # Clamp by range [0, 101]
        floor_positions_map = torch.clamp(floor_positions_map, 0, 101)
        ceil_positions_map = torch.clamp(ceil_positions_map, 0, 101)

        x1, y1 = torch.split(floor_positions_map, 1, dim=-1)
        x2, y2 = torch.split(ceil_positions_map, 1, dim=-1)

        # Make integers for indexing
        x1_int = x1.long().squeeze_()
        x2_int = x2.long().squeeze_()
        y1_int = y1.long().squeeze_()
        y2_int = y2.long().squeeze_()

        # Get 2x2 grids around (x, y) positions
        q11 = grid_map_[episode_idx, :, y1_int, x1_int]
        q12 = grid_map_[episode_idx, :, y1_int, x2_int]
        q21 = grid_map_[episode_idx, :, y2_int, x1_int]
        q22 = grid_map_[episode_idx, :, y2_int, x2_int]
        
        # Perform bilinear interpolation
        map_interp = (q11 * ((x2 - x) * (y2 - y)) +
                              q21 * ((x - x1) * (y2 - y)) +
                              q12 * ((x2 - x) * (y - y1)) +
                              q22 * ((x - x1) * (y - y1))
                              ) # Na X Ce
        
        return map_interp, positions_map

class MotionEncoder(nn.Module):
  """
  Motion Encoder for R2P2 RNN
  """
  def __init__(self, hidden_size=150):
    super(MotionEncoder, self).__init__()
    self.gru = nn.GRU(input_size=2, hidden_size=hidden_size, num_layers=1)
  
  def forward(self, x):
    '''
    input shape
    x: Te X A X 2
    
    ouput shape
    motion_encoding: A X 150
    '''
    motion_encoding, _ = self.gru(x)
    motion_encoding = motion_encoding[-1] # Need the last one

    return motion_encoding

class ContextFusion(nn.Module):
  """
  Context Fusion Network for R2P2 RNN
  """
  def __init__(self, encoding_size=50, hidden_size=150, scene_size=100, scene_channels=6):
    """
    Default keyword params are from the original settings in R2P2 paper.
    encoding_size: size of final encoding vector to return.
    hidden_size: hidden state size for the motion states encoder (GRU cell).
    scene_size: width and height of the scene encoding (Currently int value since we assume retangular shape).
    scene_channels: # channels of the scene encoding.
    """

    super(ContextFusion, self).__init__()


    self.encoding_size = encoding_size
    self.mlp = nn.Sequential(
        nn.Linear(hidden_size + scene_channels, encoding_size),
        nn.Softplus(),
        nn.Linear(encoding_size, encoding_size),
        nn.Softplus(),
      )
  
  def forward(self, motion_encoding, scene):
    '''
    input shape
    motion_encoding: Na X (Td) X 150
    scene: Na X (Td) X 6
    
    ouput shape
    final_output: Na X (Td) X 50
    '''
    # hidden state returns (for debuggings)
    hidden = []
    hidden.append(motion_encoding)

    # concat the scene & motion encodings
    concat_encoding = torch.cat((motion_encoding, scene), dim=-1)
    hidden.append(concat_encoding)

    # 2-layer MLP
    final_output = self.mlp(concat_encoding)
    hidden.append(final_output)

    return final_output, hidden

class DynamicDecoder(nn.Module):
    """
    Dynamic Decoder for R2P2 RNN
    """
    def __init__(self, hidden_size=150, context_dim=50, decoding_steps=6, velocity_const=0.5):
        super(DynamicDecoder, self).__init__()
        self.velocity_const = velocity_const
        self.decoding_steps = decoding_steps
        self.gru = nn.GRU(input_size=decoding_steps*2, hidden_size=hidden_size, num_layers=1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size+context_dim, 50),
            nn.Softplus(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 6) # DON'T USE ACTIVATION AT THE TOP MOST LAYER
        )
        
    def infer(self, x, static, init_velocity, init_position):
        """
        Na: total # of agents
        Td: decoding timesteps
        Ds: static encoding dimension
        Dg: GRU hidden dimension

        inputs
        x: Ground truth trajectory, [Na X Td X 2]
        static: Static encoding, [Na X Td X Ds]
        init_velocity: Agents initial velocity, [Na X 2]
        init_position: Agents initial position, [Na X 2]

        outputs
        z: Inference for agents latent states, [Na X (Td*2)]
        mu: Predicted agents mu, [Na x Td x 2]
        sigma: Predicted agents sigma, [Na X Td x 2 X 2]
        """
        # Detect sizes
        total_agents = x.size(0)
        decoding_steps = x.size(1)

        # Build the state differences for each timestep
        dx = x[:, 1:, :] - x[:, :-1, :] # Na X (Td-1) X 2
        dx = torch.cat((init_velocity.unsqueeze(1), dx), dim=1) # Na X Td X 2
        
        # Build the previous states for each timestep
        x_prev = x[:, :-1, :] # Na X (Td-1) X 2
        x_prev = torch.cat((init_position.unsqueeze(1), x_prev), dim=1) # Na X Td X 2

        # Build the flattend & zero padded previous states for GRU input
        x_flat = x_prev.reshape((total_agents, -1)) # Na X Td X 2 >> Na X (Td*2)
        x_flat = x_flat.unsqueeze(0).repeat(self.decoding_steps, 1, 1) # Td X Na X (Td*2)
        for i in range(decoding_steps):
            x_flat[i, :, (i+1)*2:] = 0.0

        # Unroll a step
        dynamic_encoding, _ = self.gru(x_flat) # dynamic_encoding: Td X Na X Dg
        dynamic_encoding = dynamic_encoding.transpose(1, 0) # dynamic_encoding: Na X Td X Dg
      
        # Concat the dynamic and static encodings
        dynamic_static = torch.cat((dynamic_encoding, static), dim=-1) # Na X Td X (Ds + Dg)

        # 2-layer MLP
        output = self.mlp(dynamic_static) # Na X Td X 6
        mu_hat = output[:, :, :2] # [Na X Td X 2]
        sigma_hat = output[:, :, 2:].reshape((total_agents, decoding_steps, 2, 2)) # [Na X Td X 2 X 2]

        # verlet integration
        mu = x_prev + self.velocity_const * dx + mu_hat

        # Calculate the matrix exponential for
        # [[a, b],
        #  [b, d]]
        b = sigma_hat[:, :, 0, 1] + sigma_hat[:, :, 1, 0]
        apd_2 = sigma_hat[:, :, 0, 0] + sigma_hat[:, :, 1, 1] # (a+d) / 2
        amd_2 = sigma_hat[:, :, 0, 0] - sigma_hat[:, :, 1, 1] # (a-d) / 2
        delta = torch.sqrt(amd_2 ** 2 + b ** 2)
        sinh = torch.sinh(delta)
        cosh = torch.cosh(delta)

        sigma = torch.zeros_like(sigma_hat)

        tmp1 = sinh / delta
        tmp2 = amd_2 / tmp1

        sigma[:, :, 0, 0] = cosh + tmp2
        sigma[:, :, 0, 1] = b * tmp1
        sigma[:, :, 1, 0] = sigma[:, :, 0, 1]
        sigma[:, :, 1, 1] = cosh - tmp2
        sigma *= torch.exp(apd_2).unsqueeze(-1).unsqueeze(-1)

        # solve  Z = inv(sigma) * (X-mu)
        X_mu = (x - mu).unsqueeze(-1)
        z, _ = X_mu.solve(sigma)
        z = z.reshape(total_agents, self.decoding_steps*2) # Na X (T*2)

        return z, mu, sigma

    def forward(self, z, x_flat, h, static, dx, x_prev):
        """
        Na: total # of agents (Possibly repeated by # candidates)
        Td: decoding timesteps
        Ds: static encoding dimension
        Dg: GRU hidden dimension

        inputs
        z: Latent states, [Na x 2]
        x_flat: Previous positions, [Na x (Td*2)]
        h: GRU hidden state flow, [1 X Na X Dg]
        static: Static encoding, [Na X Ds]
        dx: Agents velocity, [Na X 2]
        x_prev, Agents position, [Na x 2]

        outputs
        x: Predicted agents position, [Na X 2]
        mu: Predicted agents mu, [Na x 2]
        sigma: Predicted agents sigma, [Na X 2 X 2]
        """
        # Detect dynamic batch size
        total_agents = static.size(0)
        
        # Unroll a step
        dynamic_encoding, h = self.gru(x_flat.unsqueeze(0), h)
        dynamic_encoding = dynamic_encoding[-1] # Need the last one

        # Concat the dynamic and static encodings
        dynamic_static = torch.cat((dynamic_encoding, static), dim=-1) # [Na X (Dg+Ds)]
        
        # 2-layer MLP
        output = self.mlp(dynamic_static)# [Na X 6]
        mu_hat = output[:, :2] # [Na X 2]
        sigma_hat = output[:, 2:].reshape((total_agents, 2, 2)) # [Na X 2 X 2]

        # verlet integration
        mu = x_prev + self.velocity_const * dx + mu_hat

        # Calculate the matrix exponential for
        # [[a, b],
        #  [b, d]]
        b = sigma_hat[:, 0, 1] + sigma_hat[:, 1, 0]
        apd_2 = sigma_hat[:, 0, 0] + sigma_hat[:, 1, 1] # (a+d) / 2
        amd_2 = sigma_hat[:, 0, 0] - sigma_hat[:, 1, 1] # (a-d) / 2
        delta = torch.sqrt(amd_2 ** 2 + b ** 2)
        sinh = torch.sinh(delta)
        cosh = torch.cosh(delta)

        sigma = torch.zeros_like(sigma_hat)

        tmp1 = sinh / delta
        tmp2 = amd_2 / tmp1

        sigma[:, 0, 0] = cosh + tmp2
        sigma[:, 0, 1] = b * tmp1
        sigma[:, 1, 0] = sigma[:, 0, 1]
        sigma[:, 1, 1] = cosh - tmp2
        sigma *= torch.exp(apd_2).unsqueeze(-1).unsqueeze(-1)

        x = sigma.matmul(z.unsqueeze(-1)).squeeze(-1) + mu

        return x, mu, sigma, h