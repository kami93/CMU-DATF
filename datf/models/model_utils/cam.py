""" Code for all the model submodules part
    of various model architecures. """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()

        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias, dilation=dilation),
                                      nn.BatchNorm2d(int(n_filters)),
                                      nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):

        d_k = k.size(-1)
        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(d_k)
        # bs x n_head x seq_len x d_k @ # bs x n_head x d_k x seq_len
        # => bs x n_head x seq_len x seq_len

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9) # 1 x seq_len x seq_len

        attn = self.dropout(self.softmax(attn))
        output = torch.matmul(attn, v)
        # bs x n_head x seq_len x seq_len @ bs x n_head x seq_len x d_k
        # => bs x n_head x seq_len x d_k

        return output, attn


class SelfAttention(nn.Module):
    ''' Multi-Head Attention module ''' 
    def __init__(self, d_model, d_k, d_v, n_head=1, dropout=0.1):
        super(SelfAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.Qw = nn.Linear(d_model, n_head * d_k)
        self.Kw = nn.Linear(d_model, n_head * d_k)
        self.Vw = nn.Linear(d_model, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):
        
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1) # sz_b = 1

        q = self.Qw(q).view(sz_b, len_q, n_head, d_k) # bs x seq_len x n_head x d_k
        k = self.Kw(k).view(sz_b, len_k, n_head, d_k)
        v = self.Vw(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # bs x n_head x seq_len x d_k
        
        if mask is not None: # bs x eq_len x seq_len
            mask = mask.unsqueeze(1)   # For head axis broadcasting => bs x 1 x seq_len x seq_len

        q, attn = self.attention(q, k, v, mask=mask) # bs x n_head x seq_len x d_k

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1) 
        # bs x seq_len x n_head x d_k 
        # => bs x seq_len x d_model
        output = self.dropout(self.fc(q))

        return output #, attn


class CrossModalAttention(nn.Module):
    """
    Crossmodal Attention Module from Show, Attend, and Tell
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim, att=True):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(CrossModalAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
        self.att = att

    def forward(self, map_features, traj_encoding, episode_idx):
        """
        Forward propagation.
        :param map_features: encoded images, a tensor of dimension (agent_size, num_pixels, attention_dim)
        :param traj_encoding: previous decoder output, a tensor of dimension (agent_size, attention_dim)
        :return: attention weighted map encoding, weights
        """
        if self.att:
            att1 = self.encoder_att(map_features)  # (agent_size, num_pixels, attention_dim)
            att2 = self.decoder_att(traj_encoding)  # (agent_size, attention_dim)
            att = self.full_att(self.relu(att1[episode_idx].add_(att2.unsqueeze_(1))))  # (agent_size, num_pixels)
            
            alpha = self.softmax(att)  # (agent_size, num_pixels)
        else:
            alpha = torch.empty((episode_idx.size(0),map_features.size(1), 1), device=map_features.device).fill_(1/map_features.size(1))

        # att1: (agent_size, num_pixels, map_feat_dim) -> (agent_size, num_pixels, attention_dim) 
        # att2: (agent_size, num_pixels, traj_encoding_dim) -> (agent_size, attention_dim) 
        # att: (agent_size, num_pixels, attention_dim) + (agent_size, 1, attention_dim) -> (agent_size, num_pixels)
        # alpha: (agent_size, num_pixels)

        # alpha = torch.ones_like(alpha)
        attention_weighted_encoding = (map_features[episode_idx].mul_(alpha)).sum(dim=1)
        # (agent_size, num_pixels, encoder_dim) * (agent_size, num_pixels, 1) 
        # => (agent_size, num_pixels, encoder_dim)
        # => (agent_size, encoder_dim)

        return attention_weighted_encoding, alpha



class R2P2_Ploss(nn.Module):
    """
    R2P2 Ploss
    Interpolated Prior Loss
    """
    def __init__(self, interpolator):
        super(R2P2_Ploss, self).__init__()
        self.interpolator = interpolator
    
    def forward(self, episode_idx, pred_traj, log_prior, oom_val=-300.0):
        log_prior = log_prior.unsqueeze(dim=1)
        log_prior_bt, coordinates = self.interpolator(episode_idx, pred_traj, log_prior, oom_val)
        
        ploss = -log_prior_bt.squeeze()

        return ploss, coordinates
        

class Bilinear_Interpolation(nn.Module):
    def __init__(self, scene_size=100):
        super(Bilinear_Interpolation, self).__init__()
        self.scene_size = scene_size

    def forward(self, episode_idx, sequence, feature_map, oom_val):
        """
        inputs
        episode_idx: [A]
        sequence : [A X Td X 2]
        feature_map: [B X Ce X 100 X 100]
        oom_val: padding value
        outputs
        local_featrue_bt: [A X Td X Ce]
        sequence_mapCS: [A X Td X 2]
        """
        # Detect total agents
        total_agents = sequence.size(0)
        # Detect sequence length
        seq_len = sequence.size(1)

        if feature_map.device != sequence.device:
          feature_map = feature_map.to(sequence.device)

        # Pad the feature_map with oom_val
        pad = (1, 1, 1, 1)
        feature_map_padded = F.pad(feature_map, pad, mode='constant', value=oom_val) # [A X Ce X 102 X 102]

        # Change to map CS
        sequence_mapCS = (sequence + 56.0) / 112.0 * 100.0 + 1.0

        # Merge Agents-Time dimensions
        sequence_mapCS_bt = sequence_mapCS.reshape(-1, 2) # [A*Td, 2]
        x = sequence_mapCS_bt[:, 0:1] # [A*Td, 1]
        y = sequence_mapCS_bt[:, 1:] # [A*Td, 1]

        # Qunatize x and y
        floor_mapCS_bt = torch.floor(sequence_mapCS_bt)
        ceil_mapCS_bt = torch.ceil(sequence_mapCS_bt)

        # Clamp by range [0, 101]
        floor_mapCS_bt = torch.clamp(floor_mapCS_bt, 0, 101)
        ceil_mapCS_bt = torch.clamp(ceil_mapCS_bt, 0, 101)
        x1 = floor_mapCS_bt[:, 0:1]
        y1 = floor_mapCS_bt[:, 1:]
        x2 = ceil_mapCS_bt[:, 0:1]
        y2 = ceil_mapCS_bt[:, 1:]

        # Make integers for indexing
        x1_int = x1.long().squeeze()
        x2_int = x2.long().squeeze()
        y1_int = y1.long().squeeze()
        y2_int = y2.long().squeeze()

        # Generate duplicated batch indexes for prediction length
        # batch_idx_array = [0,0,..,0,1,1,...,1,A-1,A-1,...,A-1]
        # of length (Td * A)
        batch_idx_array = episode_idx.repeat_interleave(seq_len)

        # Get the four quadrants around (x, y)
        q11 = feature_map_padded[batch_idx_array, :, y1_int, x1_int]
        q12 = feature_map_padded[batch_idx_array, :, y1_int, x2_int]
        q21 = feature_map_padded[batch_idx_array, :, y2_int, x1_int]
        q22 = feature_map_padded[batch_idx_array, :, y2_int, x2_int]
        
        # Perform bilinear interpolation
        local_featrue_flat = (q11 * ((x2 - x) * (y2 - y)) +
                              q21 * ((x - x1) * (y2 - y)) +
                              q12 * ((x2 - x) * (y - y1)) +
                              q22 * ((x - x1) * (y - y1))
                              ) # (A*Td) X Ce
        
        local_featrue_bt = local_featrue_flat.reshape((total_agents, seq_len, -1))

        return local_featrue_bt, sequence_mapCS


class NewModelShallowCNN(nn.Module):

    def __init__(self, dropout=0.5, size=100):  # Output Size: 30 * 30
        super(NewModelShallowCNN, self).__init__()

        self.conv1 = conv2DBatchNormRelu(in_channels=3, n_filters=16, k_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=16, n_filters=16, k_size=3, stride=1, padding=1, dilation=1)
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv3 = conv2DBatchNormRelu(in_channels=16, n_filters=32, k_size=5, stride=1, padding=2, dilation=1)
        self.conv4 = conv2DBatchNormRelu(in_channels=32, n_filters=6, k_size=1, stride=1, padding=0, dilation=1)

        self.dropout = nn.Dropout(p=dropout)
        self.upsample = nn.Upsample(size=size, mode='bilinear')

    def forward(self, image, size=60):

        x = self.conv1(image) # 64 >> 64
        x = self.conv2(x) # 64 >> 64
        x = self.pool1(x) # 64 >> 32
        x = self.conv3(x) # 32 >> 32
        local_ = self.conv4(x) # 32 >> 32
        global_ = self.dropout(x)

        local_ = self.upsample(local_)

        return local_, global_


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
    motion_encoding: A X (Td) X 150
    scene: A X (Td) X 6
    
    ouput shape
    final_output: A X (Td) X 50
    '''
    # Detect dynamic batch size
    batch_size = scene.size(0)
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
        # self.lstm = nn.LSTM(decoding_steps*2, hidden_size, 1) 
        self.gru = nn.GRU(input_size=decoding_steps*2, hidden_size=hidden_size, num_layers=1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size+context_dim, 50),
            nn.Softplus(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 6) # DON'T USE ACTIVATION AT THE TOP MOST LAYER
        )
        
    def infer(self, x, past_encoding, init_velocity, init_position):
        '''
        input shape
        x: A X Td X 2 // future trj coordindates
        past_encoding: A X Td X 50 // past trj encoding
        init_velocity: A X 2
        init_position: A X 2
        
        output shape
        z : A X Td X 2
        mu: A X Td X 2
        sigma: A X Td X 2 X 2
        '''
        
        # Detect dynamic batch size
        batch_size = x.size(0)
        
        # Detect prediction length
        T = x.size(1) # T: Prediction Time Length

        # Build the state differences for each timestep
        dx = x[:, 1:, :] - x[:, :-1, :] # B X (T-1) X 2
        dx = torch.cat((init_velocity.unsqueeze(1), dx), dim=1) # B X T X 2
        
        # Build the previous states for each timestep
        x_prev = x[:, :-1, :] # B X (T-1) X 2
        x_prev = torch.cat((init_position.unsqueeze(1), x_prev), dim=1) # B X T X 2

        # Build the flattend & zero padded previous states for GRU input
        x_flat = x_prev.reshape((batch_size, -1)) # B X T X 2 >> B X (T*2)
        x_flat = x_flat.unsqueeze(0).repeat(self.decoding_steps, 1, 1) # T X B X (T*2)
        for i in range(T):
            x_flat[i, :, (i+1)*2:] = 0.0

        # Unroll a step
        dynamic_encoding, _ = self.gru(x_flat) # dynamic_encoding: T X B X 150
        dynamic_encoding = dynamic_encoding.transpose(1, 0) # dynamic_encoding: B X T X 150
      
        # Concat the dynamic and past_encoding encodings
        dynamic_static = torch.cat((dynamic_encoding, past_encoding), dim=-1) # B X T X 200

        # 2-layer MLP
        output = self.mlp(dynamic_static) # B X T X 6
        mu_hat = output[:, :, :2] # [B X T X 2]
        sigma_hat = output[:, :, 2:].reshape((batch_size, T, 2, 2)) # [B X T X 2 X 2]

        # verlet integration
        mu = x_prev + self.velocity_const * dx + mu_hat

        # Calculate the matrix exponential
        # sigma_sym = sigma_hat + sigma_hat.transpose(-2, -1) # Make a symmetric
        
        # "Batched symeig and qr are very slow on GPU"
        # https://github.com/pytorch/pytorch/issues/22573
        # device = sigma_sym.device # Detect the sigma tensor device
        # sigma_sym = sigma_sym.cpu() # eig decomposition is faster in CPU
        # e, v = torch.symeig(sigma_sym, eigenvectors=True)

        # # Convert back to gpu tensors
        # e = e.to(device) # B X T X 2
        # v = v.to(device) # B X T X 2 X 2

        # vt = v.transpose(-2, -1)
        # sigma = torch.matmul(v * torch.exp(e).unsqueeze(-2), vt) # B X T X 2 X 2

        # New matrix Exponential
        # sigma_sym = [[a, b], [b, d]]
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
        X_mu = (x - mu).unsqueeze(-1) # B X T X 2 X 1
        z, _ = X_mu.solve(sigma) # B X T X 2 X 1
        z = z.squeeze(-1) # B X T X 2

        return z, mu, sigma

    def forward(self, z, x_flat, h, past_encoding, dx, x_prev):
        '''
        input shape
        z: A X 2
        x_flat: A X 60
        h: 1 X A X 150
        past_encoding: A X 50
        dx: A X 2
        x_prev: A X 2
        
        ouput shape
        x : A X 2
        mu: A X 2
        sigma: A X 2 X 2
        '''

        # Detect dynamic batch size
        batch_size = past_encoding.size(0)
        
        # Unroll a step
        dynamic_encoding, h = self.gru(x_flat.unsqueeze(0), h)
        dynamic_encoding = dynamic_encoding[-1] # Need the last one

        # Concat the dynamic and static encodings
        dynamic_static = torch.cat((dynamic_encoding, past_encoding), dim=-1) # B X 200
        
        # 2-layer MLP
        output = self.mlp(dynamic_static)# B X 6
        mu_hat = output[:, :2] # B X 2
        sigma_hat = output[:, 2:].reshape((batch_size, 2, 2)) # B X 2 X 2

        # verlet integration
        mu = x_prev + self.velocity_const * dx + mu_hat

        # # matrix exponential
        # sigma_sym = sigma_hat + sigma_hat.transpose(-2, -1) # Make a symmetric
        
        # # "Batched symeig and qr are very slow on GPU"
        # # https://github.com/pytorch/pytorch/issues/22573
        # device = sigma_sym.device # Detect device before conversion
        # sigma_sym = sigma_sym.cpu() # Eigen decomposition is faster in CPU
        # e, v = torch.symeig(sigma_sym, eigenvectors=True)

        # # Convert back to gpu tensors
        # e = e.to(device)
        # v = v.to(device)

        # vt = v.transpose(-2, -1)
        # sigma = torch.matmul(v * torch.exp(e).unsqueeze(-2), vt) # B X 2 X 2
        
        # New matrix Exponential
        # sigma_sym = [[a, b], [b, d]]
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



class CrossModalDynamicDecoder(nn.Module):
    """
    Dynamic Decoder for R2P2 RNN
    """
    def __init__(self, image_dim=32, hidden_size=150, context_dim=50, decoding_steps=6, velocity_const=0.5, att=True):
        super(CrossModalDynamicDecoder, self).__init__()
        self.velocity_const = velocity_const
        self.decoding_steps = decoding_steps
        # self.lstm = nn.LSTM(decoding_steps*2, hidden_size, 1)        
        self.gru = nn.GRU(input_size=decoding_steps*2, hidden_size=hidden_size, num_layers=1)        
        self.crossmodal_attention = CrossModalAttention(encoder_dim=image_dim, decoder_dim=hidden_size, attention_dim=hidden_size, att=att) # map2agent attention
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size+context_dim+image_dim, 50),
            nn.Softplus(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 6) # DON'T USE ACTIVATION AT THE TOP MOST LAYER
        )
        
    def infer(self, x, static, init_velocity, init_position, global_scene_encoding, episode_idx):
        '''
        input shape
        x: A X Td X 2 // future trj coordindates
        static: A X Td X 50 // past trj encoding
        init_velocity: A X 2
        init_position: A X 2
        
        output shape
        z : A X Td X 2
        mu: A X Td X 2
        sigma: A X Td X 2 X 2
        '''
        
        # Detect dynamic batch size
        batch_size = x.size(0)
        
        # Detect prediction length
        T = x.size(1) # T: Prediction Time Length

        # Build the state differences for each timestep
        dx = x[:, 1:, :] - x[:, :-1, :] # B X (T-1) X 2
        dx = torch.cat((init_velocity.unsqueeze(1), dx), dim=1) # B X T X 2
        
        # Build the previous states for each timestep
        x_prev = x[:, :-1, :] # B X (T-1) X 2
        x_prev = torch.cat((init_position.unsqueeze(1), x_prev), dim=1) # B X T X 2

        # Build the flattend & zero padded previous states for GRU input
        x_flat = x_prev.reshape((batch_size, -1)) # B X T X 2 >> B X (T*2)
        x_flat = x_flat.unsqueeze(0).repeat(self.decoding_steps, 1, 1) # T X B X (T*2)
        for i in range(T):
            x_flat[i, :, (i+1)*2:] = 0.0

        # Get attn for all timesteps and agents
        dynamic_encoding, _ = self.gru(x_flat) # dynamic_encoding: T X B X 150
        dynamic_encoding = dynamic_encoding.transpose(0, 1)
        dynamic_encoding_ = dynamic_encoding.reshape(batch_size * T, -1)
        episode_idx_ = episode_idx.repeat_interleave(T)

        att_scenes, alpha = self.crossmodal_attention(global_scene_encoding, dynamic_encoding_, episode_idx_)
        att_scenes = att_scenes.reshape(batch_size, T, -1)

        # Concat the dynamic and static encodings
        dynamic_static = torch.cat((dynamic_encoding, static, att_scenes), dim=-1) # B X T X 200

        # 2-layer MLP
        output = self.mlp(dynamic_static) # B X T X 6
        mu_hat = output[:, :, :2] # [B X T X 2]
        sigma_hat = output[:, :, 2:].reshape((batch_size, T, 2, 2)) # [B X T X 2 X 2]

        # verlet integration
        mu = x_prev + self.velocity_const * dx + mu_hat

        # Calculate the matrix exponential
        # sigma_sym = sigma_hat + sigma_hat.transpose(-2, -1) # Make a symmetric
        
        # "Batched symeig and qr are very slow on GPU"
        # https://github.com/pytorch/pytorch/issues/22573
        # device = sigma_sym.device # Detect the sigma tensor device
        # sigma_sym = sigma_sym.cpu() # eig decomposition is faster in CPU
        # e, v = torch.symeig(sigma_sym, eigenvectors=True)

        # # Convert back to gpu tensors
        # e = e.to(device) # B X T X 2
        # v = v.to(device) # B X T X 2 X 2

        # vt = v.transpose(-2, -1)
        # sigma = torch.matmul(v * torch.exp(e).unsqueeze(-2), vt) # B X T X 2 X 2

        # New matrix Exponential
        # sigma_sym = [[a, b], [b, d]]
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
        X_mu = (x - mu).unsqueeze(-1) # B X T X 2 X 1
        z, _ = X_mu.solve(sigma) # B X T X 2 X 1
        z = z.squeeze(-1) # B X T X 2

        return z, mu, sigma

    def forward(self, z, x_flat, h, static, dx, x_prev, global_scene_encoding, episode_idx):
        '''
        input shape
        z: A X 2
        x_flat: A X 60
        h: 1 X A X 150
        static: A X 50
        dx: A X 2
        x_prev: A X 2
        
        ouput shape
        x : A X 2
        mu: A X 2
        sigma: A X 2 X 2
        '''

        # Detect dynamic batch size
        batch_size = static.size(0)
        
        # Unroll a step
        dynamic_encoding, h = self.gru(x_flat.unsqueeze(0), h)
        dynamic_encoding = dynamic_encoding[-1] # Need the last one

        att_scene, alpha = self.crossmodal_attention(global_scene_encoding, dynamic_encoding, episode_idx)

        # Concat the dynamic and static encodings
        dynamic_static = torch.cat((dynamic_encoding, static, att_scene), dim=-1) # B X 200
        
        # 2-layer MLP
        output = self.mlp(dynamic_static)# B X 6
        mu_hat = output[:, :2] # B X 2
        sigma_hat = output[:, 2:].reshape((batch_size, 2, 2)) # B X 2 X 2

        # verlet integration
        mu = x_prev + self.velocity_const * dx + mu_hat

        # # matrix exponential
        # sigma_sym = sigma_hat + sigma_hat.transpose(-2, -1) # Make a symmetric
        
        # # "Batched symeig and qr are very slow on GPU"
        # # https://github.com/pytorch/pytorch/issues/22573
        # device = sigma_sym.device # Detect device before conversion
        # sigma_sym = sigma_sym.cpu() # Eigen decomposition is faster in CPU
        # e, v = torch.symeig(sigma_sym, eigenvectors=True)

        # # Convert back to gpu tensors
        # e = e.to(device)
        # v = v.to(device)

        # vt = v.transpose(-2, -1)
        # sigma = torch.matmul(v * torch.exp(e).unsqueeze(-2), vt) # B X 2 X 2
        
        # New matrix Exponential
        # sigma_sym = [[a, b], [b, d]]
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
