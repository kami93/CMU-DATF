import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pdb


class IOC(nn.Module):
    def __init__(self, in_channels=3, decoding_steps=30):
        super(IOC, self).__init__()
        self.CNN = CNN(in_channels=in_channels)
        self.POOL = ScenePooling(scene_size=32)
        self.SCF = SceneContextFusion()
        self.GRU = nn.GRU(input_size=96,
                          hidden_size=48,
                          num_layers=1)
        self.score_fc = nn.Sequential(nn.Linear(48, 1), nn.ReLU())
        self.refine_fc = nn.Linear(48, 2*decoding_steps)
        self.decoding_steps = decoding_steps

    def forward(self, velocity, position, Hx, scene, num_tgt_trajs, scene_encoded=False):
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
        delta_Y: [Td x Na x Nc x 2]
        scene_features: [Nb x Ce x (Height / 2) x (Width / 2)]
        """
        batch_size = num_tgt_trajs.size(0)
        decoding_steps = velocity.size(0)
        num_agents = velocity.size(1)
        num_candidates = velocity.size(2)
        device = velocity.device

        episode_idx = torch.arange(batch_size, device=device).repeat_interleave(num_tgt_trajs).repeat_interleave(num_candidates)
        Hx = Hx.unsqueeze(2).expand(-1, -1, num_candidates, -1) # 1 x Na x Nc x 48

        # Scene feature pooling
        if scene_encoded:
            scene_feature = scene
        else:
            scene_feature = self.CNN(scene) # Nb x Ce x (Height/2) x (Width/2)

        position_flat = position.reshape(decoding_steps, num_agents*num_candidates, 2)
        pooled_feature_flat, _ = self.POOL(episode_idx, position_flat, scene_feature, 0.0) # [Td x (Na*Nc) x Ce]
        pooled_feature = pooled_feature_flat.reshape(decoding_steps, num_agents, num_candidates, -1) # [Td x Na x Nc x Ce]

        h_ = Hx
        scores = []
        for t_ in range(decoding_steps):
            scf_out = self.SCF(h_[0],
                               velocity[t_],
                               position[t_],
                               pooled_feature[t_],
                               num_tgt_trajs)

            scf_flat = scf_out.reshape(1, num_agents*num_candidates, 96)
            h_flat = h_.reshape(1, num_agents*num_candidates, 48)

            _, h_ = self.GRU(scf_flat, h_flat)
            h_ = h_.reshape(1, num_agents, num_candidates, 48)
            score = self.score_fc(h_)
            scores.append(score)

        scores = torch.cat(scores, dim=0)
        delta_Y = self.refine_fc(h_[0])
        delta_Y = delta_Y.reshape(num_agents, num_candidates, decoding_steps, 2)
        delta_Y = delta_Y.permute(2, 0, 1, 3)
        
        return scores, delta_Y, scene_feature

class ScenePooling(nn.Module):
    def __init__(self, scene_size=32):
        super(ScenePooling, self).__init__()
        self.scene_size = 32

    def forward(self, episode_idx, trajectory, feature_map, oom_val):
        """
        Na: Total # of agents (possibly repeated by # candidates)
        Nb: batch size (# episodes)
        Ce: scene encdoing channels
        Td: decoding timesteps
        
        inputs
        episode_idx: Episode index for each agent, [Na]
        trajectory : Agents' Trajectory [Td X Na X 2]
        feature_map: [Nb X Ce X 32 X 32]
        oom_val: padding value of out-of-map, [1]

        outputs
        local_featrue: [Td x Na x Ce]
        trajectory_mapCS: [Td x Na x 2]
        """
        # Detect trajectory length
        traj_len = trajectory.size(0)
        # Detect batch_size
        num_agents = trajectory.size(1)

        # Pad the feature_map with oom_val
        pad = (1, 1, 1, 1)
        feature_map_padded = F.pad(feature_map, pad, mode='constant', value=oom_val) # [A X Ce X 32 X 32]

        # Change to map CS
        trajectory_mapCS = (trajectory + 56.0) * self.scene_size / 112.0 + 1.0

        # Merge Time-Agents dimensions
        trajectory_mapCS_tb = trajectory_mapCS.reshape(-1, 2) # [A*Td, 2]

        # Qunatize x and y
        floor_mapCS_tb = torch.floor(trajectory_mapCS_tb)

        # Clamp by range [0, 101]
        floor_mapCS_tb = torch.clamp(floor_mapCS_tb, 0, self.scene_size+1)
        x = floor_mapCS_tb[:, 0:1]
        y = floor_mapCS_tb[:, 1:2]

        # Make integers for indexing
        x_int = x.long().squeeze()
        y_int = y.long().squeeze()

        # Generate duplicated batch indexes for prediction length
        # batch_idx_array = [0,1,...,A-1,0,1,...,A-1,...,0,1,...,A-1]
        # of length (Td * A)
        batch_idx_array = episode_idx.repeat(traj_len)

        # Pool the encodings at (x, y)s
        local_feature_tb = feature_map_padded[batch_idx_array, :, y_int, x_int]

        local_featrue = local_feature_tb.reshape((traj_len, num_agents, -1))

        return local_featrue, trajectory_mapCS

class SceneContextFusion(nn.Module):
    def __init__(self, num_wedges=6, num_rings=6, hidden_state=48, vel_fc=16,
                       rmax=4, rmin=0.5):
        super(SceneContextFusion, self).__init__()
        
        self.num_wedges = num_wedges
        self.num_rings = num_rings
        self.hidden_dim = hidden_state

        self.fc_vel = nn.Sequential(nn.Linear(2, vel_fc),
                                    nn.ReLU())
        self.fc_sp = nn.Sequential(nn.Linear(self.num_wedges*self.num_rings*self.hidden_dim, self.hidden_dim),
                                   nn.ReLU())

        self.rmax = rmax
        self.rmin = rmin
        self.rmax_by_rmin = np.log(self.rmax / self.rmin) / self.num_rings

    def forward(self, hidden, velocity, position, scene, num_tgt_trajs):
        """
        Na: Total # of agents in a batch of episodes
        Nb: batch size (# episodes)
        Nc: # of decoding candidates
        Ce: scene encdoing channels
        
        inputs
        hidden: Trajectory encodings for each agents [Na x Nc x 48]
        velocity: [Na x Nc x 2]
        position: [Na x Nc x 2]
        scene: [Na x Nc x Ce]
        num_tgt_trajs: [Nb]

        outputs
        SCF tensor: [Na x Nc x (Ce+16+48)]
        """
        num_agents = velocity.size(0)
        num_candidates = velocity.size(1)

        vel_encoding = self.fc_vel(velocity)
        social_pooling = self.social_pooling(hidden, position, num_tgt_trajs) # [Na x Nc x 6 x 6 x 48]
        sp_encoding = self.fc_sp(social_pooling.reshape(num_agents, num_candidates, -1)) # [Na x Nc x 48]

        return torch.cat((scene, vel_encoding, sp_encoding), dim=-1)

    def social_pooling(self, hidden, position, num_tgt_trajs):
        """
        Na: Total # of agents in a batch of episodes
        Nb: batch size (# episodes)
        Nc: # of decoding candidates
        Ce: scene encdoing channels
        
        inputs
        hidden: Trajectory encodings for each agents [Na x Nc x 48]
        position: [Na x Nc x 2]
        num_tgt_trajs: [Nb]

        outputs
        sp_tensors: [Na x Nc x 6 x 6 x 48]
        """
        import torch_scatter as ts
        
        batch_size = num_tgt_trajs.size(0)
        num_agents = hidden.size(0)
        num_candidates = hidden.size(1)
        hidden_size = hidden.size(2)

        cumsum_tgt_trajs = num_tgt_trajs.cumsum(dim=0)
        cumsum_tgt_trajs = torch.cat([cumsum_tgt_trajs.new_zeros([1]), cumsum_tgt_trajs])

        global_surround_idx_list = []
        candidate_idx_list = []
        linearized_idx_list = []
        for i in range(batch_size):
            # Ni: # of agents in this episode
            position_i = position[cumsum_tgt_trajs[i]:cumsum_tgt_trajs[i+1]] # [Ni x Nc X 2]

            # Calculate distance between all agents within each episode and each candidate.
            # As a result, the candidates are like different universes of the future.
            # Only those agents in the same episode and the same universe (candidate) would
            # affect each other.
            x_diff = position_i[:, None, :, 0] - position_i[None, :, :, 0]
            y_diff = position_i[:, None, :, 1] - position_i[None, :, :, 1]
            distance = torch.sqrt(x_diff ** 2 + y_diff ** 2) # [Ni x Ni x Nc]

            # Calculate the ring idx using the distances.
            # Ring idx is assigned with respect to the log-polar rings around each agent.
            ring_idx = (torch.log(distance / self.rmin + 1e-6) / self.rmax_by_rmin).floor().long()
            mask = ring_idx.ge(0) & ring_idx.lt(self.num_rings) # 0 <= ring_idx < self.num_rings
            
            ego_idx, surround_idx, candidate_idx = torch.nonzero(mask, as_tuple=True)

            num_pooling = len(ego_idx)
            if num_pooling:
                """
                Calculate the full linearized idx if there exist any surround agent
                within the rings around each agent.
                """
                # Save global surround_idx in total agents in whole batch
                global_surround_idx = surround_idx + cumsum_tgt_trajs[i]
                global_surround_idx_list.append(global_surround_idx)

                # Save candidate_idx
                candidate_idx_list.append(candidate_idx)

                # Gather ring_idx of interesting ego, surrounds, and candidates
                surround_ring_idx = ring_idx[ego_idx, surround_idx, candidate_idx]

                # Calculate the wedge idx.
                # Wedge idx is assigned by discretized angles between each agent and its surrounding agents.
                theta = torch.atan2(y_diff[ego_idx, surround_idx, candidate_idx], x_diff[ego_idx, surround_idx, candidate_idx])
                surround_wedge_idx = ((theta + np.pi - 1e-6) / np.pi / 2.0 * self.num_wedges).floor().long()
                
                # Linearied idx
                global_ego_idx = ego_idx + cumsum_tgt_trajs[i] # global ego_idx
                linearized_idx = global_ego_idx * (self.num_rings * self.num_wedges * num_candidates) + candidate_idx * (self.num_rings * self.num_wedges) + surround_ring_idx * self.num_rings + surround_wedge_idx
                linearized_idx_list.append(linearized_idx)
        
        if len(global_surround_idx_list):
            # Np: # of pooling agents
            global_surround_idx_list = torch.cat(global_surround_idx_list, dim=0)
            candidate_idx_list = torch.cat(candidate_idx_list, dim=0)
            linearized_idx_list = torch.cat(linearized_idx_list, dim=0)

            # Gather encodings of interesting surrounds and candidates
            hidden_pooling = hidden[global_surround_idx_list, candidate_idx_list].transpose(0, 1) # [48 x Np]
            
            # Perform pytorch scatter mean pooling
            sp_tensors = ts.scatter_mean(hidden_pooling, linearized_idx_list, dim_size=num_agents * num_candidates * self.num_rings * self.num_wedges)
            sp_tensors = sp_tensors.reshape(hidden_size, num_agents, num_candidates, self.num_rings, self.num_wedges)
            sp_tensors = sp_tensors.permute(1, 2, 3, 4, 0) # [Na x Nc x 6 x 6 x 48]
        else:
            sp_tensors = hidden.new_full((num_agents, num_candidates, self.num_rings, self.num_wedges, hidden_size), 0.0)

        return sp_tensors
    
class CNN(nn.Module):
    def __init__(self, in_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=16,
                      stride=2,
                      kernel_size=5,
                      padding=2),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      padding=2),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=5,
                      padding=2),
            nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        return x

class CVAE(nn.Module):
    def __init__(self, decoding_steps=30, num_candidates=6):
        super(CVAE, self).__init__()
        self.RNNEncoder1 = RNNEncoder1()
        self.RNNEncoder2 = RNNEncoder2()
        self.QDistribution = QDistribution()
        self.SampleReconstruction = SampleReconstruction(decoding_steps)
        self.decoding_steps = decoding_steps
        self.num_candidates = num_candidates

    def forward(self, x, x_len, y, y_len):
        """
        Na: Total # of agents in a batch of episodes
        Nb: batch size (# episodes)
        Nc: # of decoding candidates
        Te: encoding timesteps
        Td: decoding timestaps
        
        inputs
        x: [Na x Te x 2]
        x_len: [Na]
        y: [Na x Td x 2]
        y_len: [Na]

        outputs
        recon_y: [Td X Na x Nc x 2]
        mu: [1 x Na x 48]
        sigma: [1 x Na x 48]
        z: [Na x Nc x 48]
        Hx: [1 x Na x 48]
        """
        num_agents = x.size(0)
        device = x.device

        Hx = self.RNNEncoder1(x, x_len) # Hx: 1 x Na x 48
        Hy = self.RNNEncoder2(y, y_len) # Hy: 1 x Na x 48

        mu, sigma = self.QDistribution(Hx, Hy) # mu, sigma: (1 x Na x 48)

        eps = torch.randn([num_agents, self.num_candidates, 48], device=device)
        z = sigma.transpose(0, 1) * eps + mu.transpose(0, 1) # z: (Na x Nc x 48)

        recon_y = self.SampleReconstruction(Hx.transpose(0, 1), z) # Td X Na x Nc x 2

        return recon_y, mu, sigma, z, Hx

    def inference(self, x, x_len):
        """
        Na: Total # of agents in a batch of episodes
        Nb: batch size (# episodes)
        Nc: # of decoding candidates
        Te: encoding timesteps
        Td: decoding timestaps
        
        inputs
        x: [Na x Te x 2]
        x_len: [Na]

        outputs
        recon_y: [Td X Na x Nc x 2]
        z: [Na x Nc x 48]
        Hx: [1 x Na x 48]
        """
        num_agents = x.size(0)
        device = x.device

        Hx = self.RNNEncoder1(x, x_len)

        z = torch.randn([num_agents, self.num_candidates, 48], device=device)
        
        recon_y = self.SampleReconstruction(Hx.transpose(0, 1), z)
        
        return recon_y, z, Hx

class RNNEncoder1(nn.Module):
    def __init__(self):
        # hidden_size, n_layers=20, dropout=0
        super(RNNEncoder1, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=2,
                      out_channels=16,
                      kernel_size=3,
                      padding=1),
            nn.ReLU())
        self.gru = nn.GRU(input_size=16,
                          hidden_size=48,
                          num_layers=1)

    def forward(self, X, X_len):
        """
        X: Padded Observation Seqeunce (B X T X 2)
        X_len: True Observation Lengths
        """
        X = X.transpose(-1, -2) # Convert to (B X 2 X T)
        tX = self.conv1d(X)
        tX = tX.permute(2, 0, 1) # Convert to (T X B X C)
        tX_packed = nn.utils.rnn.pack_padded_sequence(tX, X_len, enforce_sorted=False)
        _, hidden = self.gru(tX_packed)
        
        return hidden

class RNNEncoder2(nn.Module):
    def __init__(self):
        super(RNNEncoder2, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=2,
                       out_channels=16,
                       kernel_size=1,
                       padding=0),
            nn.ReLU())

        self.gru = nn.GRU(input_size=16,
                          hidden_size=48,
                          num_layers=1)

    def forward(self, Y, Y_len):
        """
        Y: Padded Target Seqeunce (B X T X 2)
        Y_len: True Target Lengths
        """
        X = Y.transpose(-1, -2) # Convert to (B X 2 X T)
        tY = self.conv1d(X)
        tY = tY.permute(2, 0, 1) # Convert to (T X B X C)
        tY_packed = nn.utils.rnn.pack_padded_sequence(tY, Y_len, enforce_sorted=False)
        _, hidden = self.gru(tY_packed)
        
        return hidden

class QDistribution(nn.Module):
    def __init__(self):
        super(QDistribution, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU())
        self.fc_mu = nn.Linear(48, 48)
        self.fc_sigma = nn.Linear(48, 48)

    def forward(self, Hx, Hy):
        concat = torch.cat([Hx, Hy], dim=-1)
        fc1 = self.fc1(concat)
        
        mu = self.fc_mu(fc1)
        sigma = 0.5 * torch.exp(self.fc_sigma(fc1))

        return mu, sigma

class SampleReconstruction(nn.Module):
    def __init__(self, decoding_steps=30):
        super(SampleReconstruction, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(48, 48),
            nn.Softmax(dim=-1))
        self.RNNDecoder1 = RNNDecoder1()
        self.decoding_steps = decoding_steps

    def forward(self, Hx, z):
        """
        Hx: Source trajectory encdoing of shape Na x 1 x 48
        z: Random Noise of shape Na x Nc x 48

        y_: Candidate generations of shape Td X Na x Nc x 2
        """
        batch_size = z.size(0)
        num_candidates = z.size(1)

        beta = self.fc(z)
        Xz = Hx * beta
        Xz = Xz.reshape(1, batch_size*num_candidates, -1)
        Xz = Xz.repeat(self.decoding_steps, 1, 1)
        y_ = self.RNNDecoder1(Xz)

        y_ = y_.reshape(self.decoding_steps, batch_size, num_candidates, -1)
        return y_

class RNNDecoder1(nn.Module):
    def __init__(self):
        super(RNNDecoder1, self).__init__()
        self.gru = nn.GRU(input_size=48,
                          hidden_size=48,
                          num_layers=1)
        self.fc = nn.Linear(48, 2)

    def forward(self, Xz):
        HXz, _ = self.gru(Xz)
        Y_ = self.fc(HXz)
        return Y_