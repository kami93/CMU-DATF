""" Code for the main model variants. """

import torch
import torch.nn as nn
from .matf import *
from .model_utils.cam import *



# CAM
class CAM(SimpleEncoderDecoder):
    """
    Cross-agent Attention Model
    """
    def __init__(self, cfg, device, **kwargs):
        embedding_dim = kwargs.get( "embedding_dim",cfg.agent_embed_dim )
        nfuture = kwargs.get( "nfuture", cfg.nfuture )
        att_dropout = kwargs.get( "att_dropout", cfg.att_dropout )
        lstm_layers = kwargs.get( "lstm_layers", cfg.lstm_layers ) # 1
        lstm_dropout = kwargs.get( "lstm_dropout", cfg.lstm_dropout ) # 0.1
        self.device = device
        super(CAM, self).__init__(cfg, device=device, agent_embed_dim=embedding_dim, nfuture=nfuture, \
                lstm_layers=lstm_layers, lstm_dropout=lstm_dropout)
        
        self.self_attention = SelfAttention(d_model=embedding_dim, d_k=embedding_dim, \
                d_v=embedding_dim, n_head=1, dropout=att_dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)

    def crossagent_attention_block(self, agent_lstm_encodings, num_past_agents):

        ############# Cross-agent Interaction Module ############

        # Mask agents in different scenes
        trj_num = agent_lstm_encodings.size(1) # number of traj
        batch_mask = torch.zeros((trj_num, trj_num), device=self.device) # trj_num x trj_num

        blocks = [torch.ones((i, i), device=self.device) for i in num_past_agents]

        start_i = torch.zeros_like(num_past_agents[0])
        end_i = torch.zeros_like(num_past_agents[0])
        for end_i, block in zip(torch.cumsum(num_past_agents, 0), blocks):
            batch_mask[start_i:end_i, start_i:end_i] = block
            start_i = end_i
        batch_mask = batch_mask.unsqueeze(0) # 1 x trj_num x trj_num

        residual = agent_lstm_encodings # trj_num x embed
        agent_embed = self.layer_norm(agent_lstm_encodings) # T x trj_num x embed
        agent_attended_agents = self.self_attention(agent_embed, agent_embed, agent_embed, batch_mask) # trj_num x embed

        agent_attended_agents += residual # trj_num x embed

        return agent_attended_agents

    def encoder(self, past_agents_traj, past_agents_traj_len, future_agent_masks, num_past_agents):
        # Encode Scene and Past Agent Paths
        past_agents_traj = past_agents_traj.permute(1, 0, 2)  # [B X T X D] -> [T X B X D]
        agent_lstm_encodings = self.agent_encoder(past_agents_traj, past_agents_traj_len).squeeze(0) # [B X H]
        agent_lstm_encodings = agent_lstm_encodings.unsqueeze(0) # 1 x trj_num x embed

        agent_attended_agents = self.crossagent_attention_block(agent_lstm_encodings, num_past_agents)

        agent_attended_agents = agent_attended_agents.squeeze(0) # trj_num x embed
        filtered_agent_attended_agents = agent_attended_agents[future_agent_masks, :]

        return filtered_agent_attended_agents

    def forward(self, past_agents_traj, past_agents_traj_len, future_agent_masks, decode_start_vel, decode_start_pos, num_past_agents):
        
        agent_encodings = self.encoder(past_agents_traj, past_agents_traj_len, future_agent_masks, num_past_agents)
        decode = self.decoder(agent_encodings, decode_start_vel, decode_start_pos)

        return decode


# CAM + NF
class CAM_NFDecoder(CAM):
    """
    Model2: Cross-agent Attention & Normalizing Flow Decoder
    """
    def __init__(self, cfg, device, **kwargs):
        # Settings
        agent_embed_dim =  kwargs.get("agent_embed_dim", cfg.agent_embed_dim)
        velocity_const = kwargs.get("velocity_const", cfg.velocity_const) 
        nfuture = kwargs.get("nfuture", cfg.nfuture) 
        num_candidates = kwargs.get("num_candidates", cfg.num_candidates) 
        decoding_steps = kwargs.get("decoding_steps", cfg.decoding_steps)
        att_dropout = kwargs.get("att_dropout", cfg.att_dropout)
        self.cfg = cfg 
        
        self.device = device if not self.cfg else self.cfg.device

        super(CAM_NFDecoder, self).__init__(cfg=cfg, device=device)
        # super(CAM_NFDecoder, self).__init__(device, agent_embed_dim, nfuture, att_dropout)

        # self.self_attention = SelfAttention(d_model=agent_embed_dim, d_k=agent_embed_dim, d_v=agent_embed_dim, n_head=1, dropout=att_dropout)
        # self.layer_norm = nn.LayerNorm(agent_embed_dim, eps=1e-6)

        self.dynamic_decoder = DynamicDecoder(decoding_steps=decoding_steps, velocity_const=velocity_const)
        self.decoding_steps = decoding_steps    
        self.num_candidates = num_candidates
        self.mlp = nn.Sequential(
        nn.Linear(agent_embed_dim, 50),
        nn.Softplus(),
        nn.Linear(50, 50),
        nn.Softplus(),
        )

    def forward(self, src_trajs_or_src_encoding, src_lens, future_agent_masks, decode_start_vel, decode_start_pos, num_past_agents, agent_encoded=False):
        """	
        input shape	
        src_trajs_or_src_encoding:	
          A x Te x 2 if src_trajs	
        src_lens: A
        future_agent_masks: A
        decode_start_vel: A X 2	
        decode_start_pos: A X 2	
        output shape	
        x: A X Td X 2	
        mu: A X Td X 2	
        sigma: A X Td X 2 X 2	
        """	
        num_agents = src_trajs_or_src_encoding.size(0)	
        device = src_trajs_or_src_encoding.device	

        if agent_encoded:
            agent_encodings = src_trajs_or_src_encoding # (Ad*num_cand X Dim)
        else:
            agent_encodings = self.encoder(src_trajs_or_src_encoding, src_lens, future_agent_masks, num_past_agents) # (Ad X Dim)
        context_encoding = self.mlp(agent_encodings)	

        x = []
        mu = []
        sigma = []        
        z = torch.normal(mean=0.0, std=1.0, size=(num_agents*self.num_candidates, self.decoding_steps*2), device=device)	

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

        x = torch.stack(x, dim=1).reshape(num_agents, self.num_candidates, self.decoding_steps, 2) # x: Na X Nc X Td X 2	
        z = z.reshape(num_agents, self.num_candidates, self.decoding_steps*2)	
        mu = torch.stack(mu, dim=1).reshape(num_agents, self.num_candidates, self.decoding_steps, 2) # mu: Na X Nc X Td X 2	
        sigma = torch.stack(sigma, dim=1).reshape(num_agents, self.num_candidates, self.decoding_steps, 2, 2) # sigma: Na X Nc X Td X 2 X 2	

        return x, z, mu, sigma	

    def infer(self, tgt_trajs, src_trajs, src_lens, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_past_agents):
        
        """
        input shape
        tgt_trajs: Ad X Td X 2
        src_trajs: Ae X Te X 2
        src_lens: Ae
        agent_tgt_three_mask: Ae
        decode_start_vel: Ad X 2
        decode_start_pos: Ad X 2
        num_past_agents: B // sums up to Ae

        output shape
        z: Ad X Td X 2
        mu: Ad X Td X 2
        sigma: Ad X Td X 2 X 2
        agent_encodings_: Ad X Dim
        """

        agent_encodings_ = self.encoder(src_trajs, src_lens, agent_tgt_three_mask, num_past_agents) # (B X Dim)
        
        # Repeat motion encdoing for unrollig time
        agent_encodings = agent_encodings_.unsqueeze(dim=1) # (B X 1 X Dim)
        agent_encodings = agent_encodings.expand(-1, self.decoding_steps, -1) # [B X T X Dim]
        context_encoding = self.mlp(agent_encodings)	

        z, mu, sigma = self.dynamic_decoder.infer(tgt_trajs, context_encoding, decode_start_vel, decode_start_pos)
        return z, mu, sigma, agent_encodings_


# LocalScene + CAM + NF
class Scene_CAM_NFDecoder(CAM):
    """
    Model2: Cross-agent Attention & Normalizing Flow Decoder
    """
    def __init__(self, cfg, device=None, **kwargs):
        # Settings
        agent_embed_dim =  kwargs.get("agent_embed_dim", cfg.agent_embed_dim)
        velocity_const = kwargs.get("velocity_const", cfg.velocity_const) 
        nfuture = kwargs.get("nfuture", cfg.nfuture) 
        num_candidates = kwargs.get("num_candidates", cfg.num_candidates) 
        att_dropout = kwargs.get("att_dropout", cfg.att_dropout) 
        decoding_steps = kwargs.get("decoding_steps", cfg.decoding_steps)
        self.cfg = cfg 
        self.device = device if not self.cfg else self.cfg.device

        super(Scene_CAM_NFDecoder, self).__init__(cfg=cfg, device=device)

        # self.self_attention = SelfAttention(d_model=agent_embed_dim, d_k=agent_embed_dim, d_v=agent_embed_dim, n_head=1, dropout=att_dropout)
        # self.layer_norm = nn.LayerNorm(agent_embed_dim, eps=1e-6)

        self.cnn_model = NewModelShallowCNN()
        self.context_fusion = ContextFusion(hidden_size=agent_embed_dim)
        self.dynamic_decoder = DynamicDecoder(decoding_steps=decoding_steps, velocity_const=velocity_const)

        self.interpolator = Bilinear_Interpolation()
        self.decoding_steps = decoding_steps    
        self.num_candidates = num_candidates

    def forward(self, src_trajs_or_src_encoding, src_lens, future_agent_masks, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_or_scene_encoding, agent_encoded=False, scene_encoded=False):
        """	
        input shape	
        src_trajs_or_src_encoding:	
          A x Te x 2 if src_trajs	
        src_lens: A
        future_agent_masks: A
        decode_start_vel: A X 2	
        decode_start_pos: A X 2	
        output shape	
        x: A X Td X 2	
        mu: A X Td X 2	
        sigma: A X Td X 2 X 2	
        """	
        num_agents = src_trajs_or_src_encoding.size(0)	
        device = src_trajs_or_src_encoding.device	

        if scene_encoded:	
            scene_encoding = scene_or_scene_encoding	
        else:	
            scene_encoding, _ = self.cnn_model(scene_or_scene_encoding)	

        if agent_encoded:
            agent_encodings = src_trajs_or_src_encoding # (Ad*num_cand X Dim)
        else:
            agent_encodings = self.encoder(src_trajs_or_src_encoding, src_lens, future_agent_masks, num_past_agents) # (Ad X Dim)

        x = []
        mu = []
        sigma = []
        z = torch.normal(mean=0.0, std=1.0, size=(num_agents*self.num_candidates, self.decoding_steps*2), device=device)	

        episode_idx = episode_idx.repeat_interleave(self.num_candidates)
        agent_encodings = agent_encodings.repeat_interleave(self.num_candidates, dim=0)	
        decode_start_vel = decode_start_vel.repeat_interleave(self.num_candidates, dim=0)	
        decode_start_pos = decode_start_pos.repeat_interleave(self.num_candidates, dim=0)	

        x_flat = torch.zeros_like(z)
        x_prev = decode_start_pos
        dx = decode_start_vel
        h = None
        for i in range(self.decoding_steps):
            z_t = z[:, i*2:(i+1)*2]
            x_flat[:, i*2:(i+1)*2] = x_prev

            interpolated_feature, _ = self.interpolator(episode_idx, x_prev.unsqueeze(-2), scene_encoding, 0.0) # [A X 6]	
            interpolated_feature = interpolated_feature.squeeze(-2)	
            context_encoding, _ = self.context_fusion(agent_encodings, interpolated_feature) # [A X 50]	

            x_t, mu_t, sigma_t, h = self.dynamic_decoder(z_t, x_flat, h, context_encoding, dx, x_prev)	

            x.append(x_t)
            mu.append(mu_t)
            sigma.append(sigma_t)

            dx = x_t - x_prev
            x_prev = x_t
            x_flat = x_flat.clone()

        x = torch.stack(x, dim=1).reshape(num_agents, self.num_candidates, self.decoding_steps, 2) # x: Na X Nc X Td X 2	
        z = z.reshape(num_agents, self.num_candidates, self.decoding_steps*2)	
        mu = torch.stack(mu, dim=1).reshape(num_agents, self.num_candidates, self.decoding_steps, 2) # mu: Na X Nc X Td X 2	
        sigma = torch.stack(sigma, dim=1).reshape(num_agents, self.num_candidates, self.decoding_steps, 2, 2) # sigma: Na X Nc X Td X 2 X 2	

        return x, z, mu, sigma

    def infer(self, tgt_trajs, src_trajs, src_lens, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene):
        
        """
        input shape
        tgt_trajs: Ad X Td X 2
        src_trajs: Ae X Te X 2
        src_lens: Ae
        agent_tgt_three_mask: Ae
        episode_idx: A	
        decode_start_vel: Ad X 2
        decode_start_pos: Ad X 2
        num_past_agents: B // sums up to Ae
        scene: B X Ci X H X W	

        output shape
        z: Ad X Td X 2
        mu: Ad X Td X 2
        sigma: Ad X Td X 2 X 2
        agent_encodings_: Ad X Dim
        """
        scene_encoding_, _ = self.cnn_model(scene)

        agent_encodings_ = self.encoder(src_trajs, src_lens, agent_tgt_three_mask, num_past_agents) # (B X Dim)
        
        init_loc = decode_start_pos.unsqueeze(1) # [A X 1 X 2] Initial location	
        prev_locs = tgt_trajs[:, :-1, :] # [A X (Td -1) X 2] Unrolling positions	
        interp_locs = torch.cat((init_loc, prev_locs), dim=1) # [A X Td X 2]	
        interpolated_feature, _ = self.interpolator(episode_idx, interp_locs, scene_encoding_, 0.0) # [A X Td X Ce]	

        # Repeat motion encdoing for unrollig time
        agent_encodings = agent_encodings_.unsqueeze(dim=1) # (B X 1 X Dim)
        agent_encodings = agent_encodings.expand(-1, self.decoding_steps, -1) # [B X T X Dim]

        context_encoding, _  = self.context_fusion(agent_encodings, interpolated_feature) # [A X Td X 50]	

        z, mu, sigma = self.dynamic_decoder.infer(tgt_trajs, context_encoding, decode_start_vel, decode_start_pos)
        return z, mu, sigma, agent_encodings_, scene_encoding_



# GlobalScene + LocalScene + CAM + NF & AttGlobalScene + LocalScene + CAM + NF
class Global_Scene_CAM_NFDecoder(CAM):
    """
    Model2: Cross-agent Attention & Normalizing Flow Decoder
    """
    def __init__(self, cfg, device=None, **kwargs):
        # Settings
        agent_embed_dim =  kwargs.get("agent_embed_dim", cfg.agent_embed_dim)
        velocity_const = kwargs.get("velocity_const", cfg.velocity_const) 
        nfuture = kwargs.get("nfuture", cfg.nfuture) 
        num_candidates = kwargs.get("num_candidates", cfg.num_candidates) 
        att_dropout = kwargs.get("att_dropout", cfg.att_dropout) 
        decoding_steps = kwargs.get("decoding_steps", cfg.decoding_steps)
        att = kwargs.get("att", cfg.att)
        self.cfg = cfg 
        self.device = device if not self.cfg else self.cfg.device
        super(Global_Scene_CAM_NFDecoder, self).__init__( cfg, device=device, **kwargs)

        # self.self_attention = SelfAttention(d_model=agent_embed_dim, d_k=agent_embed_dim, d_v=agent_embed_dim, n_head=1, dropout=att_dropout)
        # self.layer_norm = nn.LayerNorm(agent_embed_dim, eps=1e-6)

        self.cnn_model = NewModelShallowCNN()
        self.context_fusion = ContextFusion(hidden_size=agent_embed_dim)
        self.crossmodal_dynamic_decoder = CrossModalDynamicDecoder(decoding_steps=decoding_steps, velocity_const=velocity_const, att=att)

        self.interpolator = Bilinear_Interpolation()
        self.decoding_steps = decoding_steps    
        self.num_candidates = num_candidates

    def forward(self, src_trajs_or_src_encoding, src_lens, future_agent_masks, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_or_scene_encoding, agent_encoded=False, scene_encoded=False):
        """
        input shape
        src_trajs_or_src_encoding:
          A x Te x 2 if src_trajs
        src_lens: A
        future_agent_masks: A
        decode_start_vel: A X 2
        decode_start_pos: A X 2
        output shape
        x: A X Td X 2
        mu: A X Td X 2
        sigma: A X Td X 2 X 2
        """
        num_agents = src_trajs_or_src_encoding.size(0)	
        batch_size = num_past_agents.size(0)
        device = src_trajs_or_src_encoding.device	

        if scene_encoded:	
            (local_scene_encoding_, global_scene_encoding_) = scene_or_scene_encoding	
        else:	
            local_scene_encoding_, global_scene_encoding_ = self.cnn_model(scene_or_scene_encoding)	
            global_scene_encoding_ = global_scene_encoding_.permute(0,2,3,1) # B x C x H x W >> B x H x W x C
            channel_dim = global_scene_encoding_.size(-1)
            global_scene_encoding_ = global_scene_encoding_.reshape((batch_size, -1, channel_dim)) # H, W are flattened 

        if agent_encoded:
            agent_encodings = src_trajs_or_src_encoding # (Ad*num_cand X Dim)
        else:
            agent_encodings = self.encoder(src_trajs_or_src_encoding, src_lens, future_agent_masks, num_past_agents) # (Ad X Dim)

        x = []
        mu = []
        sigma = []
        z = torch.normal(mean=0.0, std=1.0, size=(num_agents*self.num_candidates, self.decoding_steps*2), device=device)	

        episode_idx = episode_idx.repeat_interleave(self.num_candidates)
        agent_encodings = agent_encodings.repeat_interleave(self.num_candidates, dim=0)	
        decode_start_vel = decode_start_vel.repeat_interleave(self.num_candidates, dim=0)	
        decode_start_pos = decode_start_pos.repeat_interleave(self.num_candidates, dim=0)	

        x_flat = torch.zeros_like(z)
        x_prev = decode_start_pos
        dx = decode_start_vel
        h = None
        for i in range(self.decoding_steps):
            z_t = z[:, i*2:(i+1)*2]
            x_flat[:, i*2:(i+1)*2] = x_prev

            interpolated_feature, _ = self.interpolator(episode_idx, x_prev.unsqueeze(-2), local_scene_encoding_, 0.0) # [A X 6]	
            interpolated_feature = interpolated_feature.squeeze(-2)	
            context_encoding, _ = self.context_fusion(agent_encodings, interpolated_feature) # [A X 50]	

            x_t, mu_t, sigma_t, h = self.crossmodal_dynamic_decoder(z_t, x_flat, h, context_encoding, dx, x_prev, global_scene_encoding_, episode_idx)

            x.append(x_t)
            mu.append(mu_t)
            sigma.append(sigma_t)

            dx = x_t - x_prev
            x_prev = x_t
            x_flat = x_flat.clone()

        x = torch.stack(x, dim=1).reshape(num_agents, self.num_candidates, self.decoding_steps, 2) # x: Na X Nc X Td X 2	
        z = z.reshape(num_agents, self.num_candidates, self.decoding_steps*2)	
        mu = torch.stack(mu, dim=1).reshape(num_agents, self.num_candidates, self.decoding_steps, 2) # mu: Na X Nc X Td X 2	
        sigma = torch.stack(sigma, dim=1).reshape(num_agents, self.num_candidates, self.decoding_steps, 2, 2) # sigma: Na X Nc X Td X 2 X 2	

        return x, z, mu, sigma	

    def infer(self, tgt_trajs, src_trajs, src_lens, future_agent_masks, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene):
        
        """
        input shape
        tgt_trajs: Ad X Td X 2
        src_trajs: Ae X Te X 2
        src_lens: Ae
        future_agent_masks: Ae
        episode_idx: A	
        decode_start_vel: Ad X 2
        decode_start_pos: Ad X 2
        num_past_agents: B // sums up to Ae
        scene: B X Ci X H X W	

        output shape
        z: Ad X Td X 2
        mu: Ad X Td X 2
        sigma: Ad X Td X 2 X 2
        agent_encodings_: Ad X Dim
        """

        batch_size = num_past_agents.size(0)

        local_scene_encoding_, global_scene_encoding_ = self.cnn_model(scene)	
        agent_encodings_ = self.encoder(src_trajs, src_lens, future_agent_masks, num_past_agents) # (B X Dim)
        
        init_loc = decode_start_pos.unsqueeze(1) # [A X 1 X 2] Initial location	
        prev_locs = tgt_trajs[:, :-1, :] # [A X (Td -1) X 2] Unrolling positions	
        interp_locs = torch.cat((init_loc, prev_locs), dim=1) # [A X Td X 2]	
        interpolated_feature, _ = self.interpolator(episode_idx, interp_locs, local_scene_encoding_, 0.0) # [A X Td X Ce]	

        # Repeat motion encdoing for unrollig time
        agent_encodings = agent_encodings_.unsqueeze(dim=1) # (B X 1 X Dim)
        agent_encodings = agent_encodings.expand(-1, self.decoding_steps, -1) # [B X T X Dim]

        context_encoding, _  = self.context_fusion(agent_encodings, interpolated_feature) # [A X Td X 50]	

        global_scene_encoding_ = global_scene_encoding_.permute(0,2,3,1) # B x C x H x W >> B x H x W x C
        channel_dim = global_scene_encoding_.size(-1)
        global_scene_encoding_ = global_scene_encoding_.reshape((batch_size, -1, channel_dim)) # H, W are flattened 

        z, mu, sigma = self.crossmodal_dynamic_decoder.infer(tgt_trajs, context_encoding, decode_start_vel, decode_start_pos, global_scene_encoding_, episode_idx)
        return z, mu, sigma, agent_encodings_, (local_scene_encoding_, global_scene_encoding_)

