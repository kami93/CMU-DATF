""" Code for all the model submodules part
    of various model architecures. """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch_scatter as ts

class AgentEncoderLSTM(nn.Module):
    def __init__(self, device, embedding_dim=32, h_dim=32, num_layers=1, dropout=0.3):
        super(AgentEncoderLSTM, self).__init__()

        self.device = device
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.num_layers = num_layers

        if self.num_layers > 1:
            self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout) 
        else:
            self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers) 

        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def init_hidden(self, total_agents):
        # h_0, c_0 of shape (num_layers, batch, hidden_size)
        return (
            torch.zeros(self.num_layers, total_agents, self.h_dim, device=self.device),
            torch.zeros(self.num_layers, total_agents, self.h_dim, device=self.device)
        )

    def forward(self, obs_traj, src_lens):
        total_agents = obs_traj.size(1)
        hidden = self.init_hidden(total_agents)

        # Convert to relative, as Social GAN do
        rel_curr_ped_seq = torch.zeros_like(obs_traj)
        rel_curr_ped_seq[1:, :, :] = obs_traj[1:, :, :] - obs_traj[:-1, :, :]

        # Trajectory Encoding
        obs_traj_embedding = self.spatial_embedding(rel_curr_ped_seq.reshape(-1, 2))
        obs_traj_embedding = obs_traj_embedding.reshape(-1, total_agents, self.embedding_dim)

        obs_traj_embedding = nn.utils.rnn.pack_padded_sequence(obs_traj_embedding, src_lens, enforce_sorted=False)
        output, (hidden_final, cell_final) = self.encoder(obs_traj_embedding, hidden)
        
        if self.num_layers>1:
            hidden_final = hidden_final[0]

        return hidden_final


class AgentDecoderLSTM(nn.Module):

    def __init__(self, device, seq_len, embedding_dim=32, h_dim=32, num_layers=1, dropout=0.0):
        super(AgentDecoderLSTM, self).__init__()

        self.seq_len = seq_len
        self.device = device
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.num_layers = num_layers

        if self.num_layers > 1:
            self.decoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        else:
            self.decoder = nn.LSTM(embedding_dim, h_dim, num_layers)  

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def relative_to_abs(self, rel_traj, start_pos=None):
        """
        Inputs:
        - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
        - start_pos: pytorch tensor of shape (batch, 2)
        Outputs:
        - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
        """
        if start_pos is None:
            start_pos = torch.zeros_like(rel_traj[0])

        rel_traj = rel_traj.permute(1, 0, 2)
        displacement = torch.cumsum(rel_traj, dim=1)

        start_pos = torch.unsqueeze(start_pos, dim=1)
        abs_traj = displacement + start_pos

        return abs_traj.permute(1, 0, 2)

    def forward(self, last_pos_rel, hidden_state, start_pos=None):
        """
        Inputs:
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        total_agents = last_pos_rel.size(0)
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.reshape(1, total_agents, self.embedding_dim)

        if self.num_layers > 1:
            zero_hidden_states = torch.zeros((self.num_layers-1), hidden_state.shape[1], hidden_state.shape[2], device=self.device)
            decoder_h = torch.cat((hidden_state, zero_hidden_states), dim=0)
            decoder_c = torch.zeros_like(decoder_h)
            state_tuple = (decoder_h, decoder_c)
        else:
            decoder_c = torch.zeros_like(hidden_state)
            state_tuple = (hidden_state, decoder_c)

        predicted_rel_pos_list = []
        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            predicted_rel_pos = self.hidden2pos(output.reshape(total_agents, self.h_dim))
            predicted_rel_pos_list.append(predicted_rel_pos) # [B X 2]

            decoder_input = self.spatial_embedding(predicted_rel_pos)
            decoder_input = decoder_input.reshape(1, total_agents, self.embedding_dim)            

        predicted_rel_pos_result = torch.stack(predicted_rel_pos_list, dim=0) # [L X B X 2]

        return self.relative_to_abs(predicted_rel_pos_result, start_pos), state_tuple[0]


class AgentsMapFusion(nn.Module):

    def __init__(self, in_channels=32, out_channels=32):
        super(AgentsMapFusion, self).__init__()

        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, n_filters=out_channels,
                                          k_size=3, stride=1, padding=1, dilation=1)
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv2 = conv2DBatchNormRelu(in_channels=out_channels, n_filters=out_channels,
                                          k_size=3, stride=1, padding=1, dilation=1)
        self.pool2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv3 = conv2DBatchNormRelu(in_channels=out_channels, n_filters=out_channels,
                                          k_size=4, stride=1, padding=1, dilation=1)

        self.deconv2 = deconv2DBatchNormRelu(in_channels=out_channels, n_filters=out_channels,
                                              k_size=4, stride=2, padding=1)

    def forward(self, input_tensor):
        conv1 = self.conv1.forward(input_tensor)
        conv2 = self.conv2.forward(self.pool1.forward(conv1))
        conv3 = self.conv3.forward(self.pool2.forward(conv2))

        up2 = self.deconv2.forward(conv2)
        up3 = F.interpolate(conv3, scale_factor=5)

        features = conv1 + up2 + up3
        return features

class SpatialEncodeAgent(nn.Module):

    def __init__(self, device, pooling_size):
        super(SpatialEncodeAgent, self).__init__()
        self.device = device
        self.pooling_size = pooling_size

    def forward(self, batch_size, encode_coordinates, agent_encodings):
        channel = agent_encodings.shape[-1]
        pool_vector = agent_encodings.transpose(1, 0) # [C X D]

        init_map_ts = torch.zeros((channel, batch_size*self.pooling_size*self.pooling_size), device=self.device) # [C X B*H*W]
        out, _ = ts.scatter_min(src=pool_vector, index=encode_coordinates, out=init_map_ts) # [C X B*H*W]
        out, _ = ts.scatter_max(src=pool_vector, index=encode_coordinates, out=out) # [C X B*H*W]

        out = out.reshape((channel, batch_size, self.pooling_size, self.pooling_size)) # [C X B X H X W]
        out = out.permute((1, 0, 2, 3)) # [B X C X H X W]

        return out

class SpatialFetchAgent(nn.Module):

    def __init__(self, device):
        super(SpatialFetchAgent, self).__init__()
        self.device = device

    def forward(self, fused_grid, agent_encodings, fetch_coordinates):
        # Rearange the fused grid so that linearized index may be used.
        batch, channel, map_h, map_w = fused_grid.shape
        fused_grid = fused_grid.permute((0, 2, 3, 1)) # B x H x W x C
        fused_grid = fused_grid.reshape((batch*map_h*map_w, channel))

        fused_encodings = fused_grid[fetch_coordinates]
        final_encoding = fused_encodings + agent_encodings

        return final_encoding



# class conv2DBatchNormRelu(nn.Module):
#     def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1):
#         super(conv2DBatchNormRelu, self).__init__()

#         self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
#                                                 padding=padding, stride=stride, bias=bias, dilation=dilation),
#                                       nn.BatchNorm2d(int(n_filters)),
#                                       nn.ReLU(inplace=True))

#     def forward(self, inputs):
#         outputs = self.cbr_unit(inputs)
#         return outputs


class conv2DRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1):
        super(conv2DRelu, self).__init__()

        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias, dilation=dilation),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class deconv2DRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs

class ResnetShallow(nn.Module):

    def __init__(self, dropout=0.5):  # Output Size: 30 * 30
        super(ResnetShallow, self).__init__()

        self.trunk = models.resnet18(pretrained=True)

        self.upscale3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), )

        self.upscale4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 7, stride=4, padding=3, output_padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), )

        self.shrink = conv2DBatchNormRelu(in_channels=384, n_filters=32,
                                          k_size=1, stride=1, padding=0)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, image):
        x = self.trunk.conv1(image)
        x = self.trunk.bn1(x)
        x = self.trunk.relu(x)
        x = self.trunk.maxpool(x)

        x = self.trunk.layer1(x)
        x2 = self.trunk.layer2(x)  # /8 the size
        x3 = self.trunk.layer3(x2)  # 16
        x4 = self.trunk.layer4(x3)  # 32

        x3u = self.upscale3(x3)
        x4u = self.upscale4(x4)

        xall = torch.cat((x2, x3u, x4u), dim=1)
        xall = F.interpolate(xall, size=(30, 30))
        final = self.shrink(xall)

        output = self.dropout(final)

        return output

class ShallowCNN(nn.Module):

    def __init__(self, in_channels, dropout=0.5):  # Output Size: 30 * 30
        super(ShallowCNN, self).__init__()

        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, n_filters=16,
                                          k_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=16, n_filters=16,
                                          k_size=4, stride=1, padding=2, dilation=1)
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv3 = conv2DBatchNormRelu(in_channels=16, n_filters=32,
                                          k_size=5, stride=1, padding=2, dilation=1)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, image):

        x = self.conv1(image)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)

        output = self.dropout(x)

        return output

class TestCNN(nn.Module):
    def __init__(self, in_channels):  # Output Size: 30 * 30
        super(TestCNN, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, n_filters=16,
                                          k_size=3, stride=1, padding=1, dilation=1)

    def forward(self, image):
        output = self.conv1(image)

        return output

    def pause_stats_update(self):
        for instance in self.modules():
            if isinstance(instance, MyBatchNorm2d):
                instance.pause_stats_update()

    def resume_stats_update(self):
        for instance in self.modules():
            if isinstance(instance, MyBatchNorm2d):
                instance.resume_stats_update()

class conv2DBatchNormRelu(nn.Module):
    """ conv2DBatchNormRelu v2 with pause/resume stats update function.
    """
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()
        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias, dilation=dilation),
                                      MyBatchNorm2d(int(n_filters)),
                                      nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
    
    def pause_stats_update(self):
        for instance in self.modules():
            if isinstance(instance, MyBatchNorm2d):
                instance.pause_stats_update()

    def resume_stats_update(self):
        for instance in self.modules():
            if isinstance(instance, MyBatchNorm2d):
                instance.resume_stats_update()

class MyBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(MyBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.stats_update = True

    def pause_stats_update(self):
        self.stats_update = False

    def resume_stats_update(self):
        self.stats_update = True
    
    def forward(self, input):
        if self.training and not self.stats_update:
            self._check_input_dim(input)
            return F.batch_norm(
                input,
                None,
                None,
                self.weight, self.bias, self.training, 0.0, self.eps)

        else:
            return super(MyBatchNorm2d, self).forward(input)