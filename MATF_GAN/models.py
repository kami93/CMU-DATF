""" Code for the main model variants. """

import torch
import torch.nn as nn
from MATF.models import MATF
from MATF_GAN.model_utils import Classifier

class MATF_Gen(MATF):
    """
    Multi Agent Tensor Fusion
    """
    def __init__(self, device, agent_embed_dim, nfuture, 
                 lstm_layers, lstm_dropout, noise_dim, pooling_size, encoder_type, 
                 scene_channels, scene_dropout, freeze_resnet=True):

        super(MATF_Gen, self).__init__(device, agent_embed_dim, nfuture,
                 lstm_layers, lstm_dropout, noise_dim, pooling_size, encoder_type, 
                 scene_channels, scene_dropout, freeze_resnet)


    def forward(self, past_agents_traj, past_agents_traj_len, episode_idx, future_agent_masks, decode_start_vel, decode_start_pos, scene_images, stochastic=False, num_candidate=1):

        agent_encodings = self.encoder(past_agents_traj, past_agents_traj_len, decode_start_pos, episode_idx, future_agent_masks, scene_images)
        decode = self.decoder(agent_encodings, decode_start_vel[future_agent_masks], decode_start_pos[future_agent_masks], stochastic, num_candidate)

        return decode

    def decoder(self, agent_encodings, decode_start_vel, decode_start_pos, stochastic=False, num_candidate=1):

        total_agent = agent_encodings.shape[0]
        if stochastic:
            noise = torch.randn((total_agent*num_candidate, self.noise_dim), device=self.device)
        else:
            noise = torch.zeros((total_agent*num_candidate, self.noise_dim), device=self.device)

        if num_candidate > 1:
            agent_encodings = agent_encodings.repeat_interleave(num_candidate, dim=0)
            decode_start_vel = decode_start_vel.repeat_interleave(num_candidate, dim=0)
            decode_start_pos = decode_start_pos.repeat_interleave(num_candidate, dim=0)
        fused_noise_encodings = torch.cat((agent_encodings, noise), dim=1)
        decoder_h = fused_noise_encodings.unsqueeze(0)

        predicted_trajs, final_decoder_h = self.agent_decoder(last_pos_rel=decode_start_vel,
                                                              hidden_state=decoder_h,
                                                              start_pos=decode_start_pos)

        predicted_trajs = predicted_trajs.permute(1, 0, 2) # [B*Nc X T X 2]

        return predicted_trajs


class MATF_Disc(MATF):
    """
    Multi Agent Tensor Fusion Discriminator Network
    """
    def __init__(self, device, agent_embed_dim, nfuture, 
                 lstm_layers, lstm_dropout, noise_dim, pooling_size, encoder_type, 
                 scene_channels, scene_dropout, freeze_resnet, disc_hidden, disc_dropout):

        super(MATF_Disc, self).__init__(device, agent_embed_dim, nfuture, 
                                        lstm_layers, lstm_dropout, noise_dim, pooling_size, encoder_type, 
                                        scene_channels, scene_dropout, freeze_resnet)

        self.classifier = Classifier(device, agent_embed_dim, classifier_hidden=disc_hidden, dropout=disc_dropout)

    def encoder(self, agents_traj, agents_traj_len, decode_start_pos, episode_idx, future_agent_masks, scene_images, num_candidate=1):

        # Encode Scene and (Past+Future) Agent Paths
        scene_encodings = self.scene_encoder(scene_images).repeat(num_candidate, 1, 1, 1)

        agents_traj = agents_traj.permute(1, 0, 2)  # [B X T X D] -> [T X B X D]
        agent_lstm_encodings = self.agent_encoder(agents_traj, agents_traj_len).squeeze(0) # [B X H]

        discrete_idx = ((decode_start_pos + 56.0) / 112.0 * self.pooling_size).floor().long()
        discrete_idx = torch.clamp(discrete_idx, 0, self.pooling_size-1)
        x, y = discrete_idx[:, 0], discrete_idx[:, 1]
        encode_coordinates = episode_idx*self.pooling_size*self.pooling_size + y*self.pooling_size + x

        batch_size = scene_encodings.size(0) # batch_size repeated by num_candidate

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

    def decoder(self, agent_encodings):

        discriminator_score = self.classifier(agent_encodings)

        return discriminator_score

    def forward(self, agents_traj, agents_traj_len, episode_idx, future_agent_masks, decode_start_pos, scene_images, num_candidate=1):

        agent_encodings = self.encoder(agents_traj, agents_traj_len, decode_start_pos, episode_idx, future_agent_masks, scene_images, num_candidate)

        decode = self.decoder(agent_encodings)

        return decode

