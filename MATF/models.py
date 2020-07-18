""" Code for the main model variants. """

import torch
import torch.nn as nn
from MATF.model_utils import *


class SimpleEncoderDecoder(nn.Module):
    """
    Simple EncoderDecoder model
    """
    def __init__(self, device, agent_embed_dim, nfuture, lstm_layers, lstm_dropout, noise_dim=16):

        super(SimpleEncoderDecoder, self).__init__()

        self.device = device
        self.num_layers = lstm_layers
        self.agent_embed_dim = agent_embed_dim
        self.noise_dim = noise_dim

        self.agent_encoder = AgentEncoderLSTM(device=device, embedding_dim=agent_embed_dim,
                                              h_dim=agent_embed_dim, num_layers=lstm_layers, dropout=lstm_dropout)
                                            
        self.agent_decoder = AgentDecoderLSTM( # decoder has noise_dim more dimension than encoder due to GAN pretraining
                                            device=device, seq_len=nfuture, embedding_dim=agent_embed_dim + noise_dim,
                                            h_dim=agent_embed_dim + noise_dim, num_layers=lstm_layers, dropout=lstm_dropout
                                            )

    def encoder(self, past_agents_traj, past_agents_traj_len, future_agent_masks):
        # Encode Scene and Past Agent Paths
        past_agents_traj = past_agents_traj.permute(1, 0, 2)  # [B X T X D] -> [T X B X D]

        agent_lstm_encodings = self.agent_encoder(past_agents_traj, past_agents_traj_len).squeeze(0) # [B X H]

        filtered_agent_lstm_encodings = agent_lstm_encodings[future_agent_masks, :] 

        return filtered_agent_lstm_encodings

    def decoder(self, agent_encodings, decode_start_vel, decode_start_pos, stochastic=False):

        total_agent = agent_encodings.shape[0]
        if stochastic:
            noise = torch.randn((total_agent, self.noise_dim), device=self.device)
        else:
            noise = torch.zeros((total_agent, self.noise_dim), device=self.device)

        fused_noise_encodings = torch.cat((agent_encodings, noise), dim=1)
        decoder_h = fused_noise_encodings.unsqueeze(0)

        predicted_trajs, final_decoder_h = self.agent_decoder(last_pos_rel=decode_start_vel,
                                                              hidden_state=decoder_h,
                                                              start_pos=decode_start_pos)
        predicted_trajs = predicted_trajs.permute(1, 0, 2) # [B X L X 2]

        return predicted_trajs

    def forward(self, past_agents_traj, past_agents_traj_len, future_agent_masks, decode_start_vel, decode_start_pos):
        
        agent_encodings = self.encoder(past_agents_traj, past_agents_traj_len, future_agent_masks)
        decode = self.decoder(agent_encodings, decode_start_vel[future_agent_masks], decode_start_pos[future_agent_masks])

        return decode


class SocialPooling(SimpleEncoderDecoder):
    """
    Social Pooling Model
    """
    def __init__(self, device, agent_embed_dim, nfuture, lstm_layers, lstm_dropout, noise_dim, pooling_size):

        super(SocialPooling, self).__init__(device, agent_embed_dim, nfuture, lstm_layers, lstm_dropout, noise_dim)

        self.pooling_size = pooling_size
        self.spatial_encode_agent=SpatialEncodeAgent(device=device, pooling_size=pooling_size)
        self.spatial_pooling_net=AgentsMapFusion(in_channels=agent_embed_dim, out_channels=agent_embed_dim)
        self.spatial_fetch_agent=SpatialFetchAgent(device=device)

    def encoder(self, past_agents_traj, past_agents_traj_len, decode_start_pos, episode_idx, future_agent_masks):

        past_agents_traj = past_agents_traj.permute(1, 0, 2)  # [B X T X D] -> [T X B X D]
        agent_lstm_encodings = self.agent_encoder(past_agents_traj, past_agents_traj_len).squeeze(0) # [B X H]

        discrete_idx = ((decode_start_pos + 56.0) / 112.0 * self.pooling_size).floor().long()
        discrete_idx = torch.clamp(discrete_idx, 0, self.pooling_size-1)
        x, y = discrete_idx[:, 0], discrete_idx[:, 1]
        encode_coordinates = episode_idx*self.pooling_size*self.pooling_size + y*self.pooling_size + x

        batch_size = episode_idx[-1] + 1

        spatial_encoded_agents = self.spatial_encode_agent(batch_size, encode_coordinates, agent_lstm_encodings)

        # Do social pooling on the pooled agents map.
        fused_grid = self.spatial_pooling_net(spatial_encoded_agents)

        # Gather future decoding agents
        future_agent_encodings = agent_lstm_encodings[future_agent_masks, :]
        fetching_coordinates = encode_coordinates[future_agent_masks]

        # Fetch fused agents states back w.r.t. coordinates from fused map grid:
        fused_agent_encodings = self.spatial_fetch_agent(fused_grid, future_agent_encodings, fetching_coordinates)

        return fused_agent_encodings 

    def forward(self, past_agents_traj, past_agents_traj_len, episode_idx, future_agent_masks, decode_start_vel, decode_start_pos):
        
        agent_encodings = self.encoder(past_agents_traj, past_agents_traj_len, decode_start_pos, episode_idx, future_agent_masks)
        decode = self.decoder(agent_encodings, decode_start_vel[future_agent_masks], decode_start_pos[future_agent_masks])

        return decode

class MATF(SocialPooling):
    """
    Multi Agent Tensor Fusion
    """
    def __init__(self, device, agent_embed_dim, nfuture, 
                 lstm_layers, lstm_dropout, noise_dim, pooling_size, encoder_type, 
                 scene_channels, scene_dropout, freeze_resnet=True):

        super(MATF, self).__init__(device, agent_embed_dim, nfuture, lstm_layers, lstm_dropout, noise_dim, pooling_size)
        if encoder_type == 'ResNet':
            if scene_channels != 3:
                raise ValueError("ResNet only supports RGB input.")
            self.scene_encoder = ResnetShallow(dropout=scene_dropout)
            for param in self.scene_encoder.trunk.parameters():
                if freeze_resnet:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        elif encoder_type == 'ShallowCNN':
            self.scene_encoder = ShallowCNN(in_channels=scene_channels, dropout=scene_dropout)

        else:
            raise ValueError("Invalid Map Encoder Type")

        self.spatial_pooling_net=AgentsMapFusion(in_channels=(agent_embed_dim + 32), out_channels=agent_embed_dim)


    def forward(self, past_agents_traj, past_agents_traj_len, episode_idx, future_agent_masks, decode_start_vel, decode_start_pos, scene_images, stochastic=False):

        agent_encodings = self.encoder(past_agents_traj, past_agents_traj_len, decode_start_pos, episode_idx, future_agent_masks, scene_images)
        decode = self.decoder(agent_encodings, decode_start_vel[future_agent_masks], decode_start_pos[future_agent_masks], stochastic)

        return decode

    def encoder(self, past_agents_traj, past_agents_traj_len, decode_start_pos, episode_idx, future_agent_masks, scene_images):

        # Encode Scene
        scene_encodings = self.scene_encoder(scene_images)

        # Encode Agent Motion
        past_agents_traj = past_agents_traj.permute(1, 0, 2)  # [B X T X D] -> [T X B X D]
        agent_lstm_encodings = self.agent_encoder(past_agents_traj, past_agents_traj_len).squeeze(0) # [B X H]

        discrete_idx = ((decode_start_pos + 56.0) / 112.0 * self.pooling_size).floor().long()
        discrete_idx = torch.clamp(discrete_idx, 0, self.pooling_size-1)
        x, y = discrete_idx[:, 0], discrete_idx[:, 1]
        encode_coordinates = episode_idx*self.pooling_size*self.pooling_size + y*self.pooling_size + x

        batch_size = scene_encodings.size(0)

        # Spatial Encode Agents
        spatial_encoded_agents = self.spatial_encode_agent(batch_size, encode_coordinates, agent_lstm_encodings)
        
        # Tensor Fusion
        tensor_fusion = torch.cat((scene_encodings, spatial_encoded_agents), 1)
        fused_grid = self.spatial_pooling_net(tensor_fusion)
        
        # Gather future decoding agents
        future_agent_encodings = agent_lstm_encodings[future_agent_masks, :]
        fetching_coordinates = encode_coordinates[future_agent_masks]

        # Fetch fused agents states back w.r.t. coordinates from fused map grid:
        fused_agent_encodings = self.spatial_fetch_agent(fused_grid, future_agent_encodings, fetching_coordinates)
        
        return fused_agent_encodings 

