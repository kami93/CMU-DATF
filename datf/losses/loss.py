

import torch
import torch.nn as nn
import torch.nn.functional as F


class Log_determinant(nn.Module):
    def __init__(self):
        super(Log_determinant, self).__init__()

    def forward(self, sigma):
        det = sigma[:, :, 0, 0] * sigma[:, :, 1, 1] - sigma[:, :, 0, 1] ** 2
        logdet = torch.log(det + 1e-9)
        return logdet


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