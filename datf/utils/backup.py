
    # def train_single_epoch(self, epoch):
    #     """Trains the model for a single round."""

    #     self.model.train()
    #     epoch_loss = 0.0
    #     epoch_d_loss = 0.0

    #     if self.flow_based_decoder:
    #         epoch_qloss = 0.0
    #         epoch_ploss = 0.0

    #     epoch_minade2, epoch_avgade2 = 0.0, 0.0
    #     epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
    #     epoch_minade3, epoch_avgade3 = 0.0, 0.0
    #     epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
    #     epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

    #     H = W = 60
    #     if '2.' in self.map_version:
    #         with torch.no_grad():
    #             coordinate_2d = np.indices((H, W))
    #             coordinate = np.ravel_multi_index(coordinate_2d, dims=(H, W))
    #             coordinate = torch.FloatTensor(coordinate)
    #             coordinate = coordinate.reshape((1, 1, H, W))

    #             coordinate_std, coordinate_mean = torch.std_mean(coordinate)
    #             coordinate = (coordinate - coordinate_mean) / coordinate_std

    #             distance_2d = coordinate_2d - np.array([(H-1)/2, (H-1)/2]).reshape((2, 1, 1))
    #             distance = np.sqrt((distance_2d ** 2).sum(axis=0))
    #             distance = torch.FloatTensor(distance)
    #             distance = distance.reshape((1, 1, H, W))

    #             distance_std, distance_mean = torch.std_mean(distance)
    #             distance = (distance - distance_mean) / distance_std
            
    #         coordinate = coordinate.to(self.device)
    #         distance = distance.to(self.device)
    #     if self.flow_based_decoder:
    #         c1 = -self.decoding_steps * np.log(2 * np.pi)

    #     if self.generative:
    #         for i, e in enumerate(self.gan_weight_schedule):
    #             if epoch <= e:
    #                 gan_weight = self.gan_weight[i]
    #                 break

    #     torch.autograd.set_detect_anomaly(True)
    #     for b, batch in enumerate(self.train_loader):

    #         print("Working on batch {:d}/{:d}".format(b+1, len(self.train_loader)), end='\r')

    #         self.optimizer.zero_grad()
    #         if self.generative: self.optimizer_D.zero_grad()

    #         scene_images, log_prior, \
    #         future_agent_masks, \
    #         num_past_agents,   past_agents_traj,   past_agents_traj_len,   past_agents_traj_len_idx, \
    #         num_future_agents, future_agents_traj, future_agents_traj_len, future_agents_traj_len_idx, \
    #         two_mask, three_mask, \
    #         decode_start_vel, decode_start_pos, \
    #         scene_id = batch

    #         agent_masks = future_agent_masks 
    #         num_src_trajs = num_past_agents 
    #         src_trajs = past_agents_traj
    #         src_lens = past_agents_traj_len
    #         src_len_idx = past_agents_traj_len_idx 
    #         num_tgt_trajs = num_future_agents  
    #         tgt_trajs = future_agents_traj
    #         tgt_lens = future_agents_traj_len
    #         tgt_len_idx = future_agents_traj_len_idx 
    #         tgt_two_mask = two_mask
    #         tgt_three_mask = three_mask

    #         # Detect dynamic batch size
    #         batch_size = scene_images.size(0)
    #         num_agents = future_agents_traj.size(0)
    #         num_three_agents = torch.sum(three_mask)

    #         # generative
    #         total_past_agent = past_agents_traj.size(0)
    #         total_future_agent = future_agents_traj.size(0)

    #         if '2.' in self.map_version:
    #             coordinate_batch = coordinate.expand(batch_size, -1, -1, -1)
    #             distance_batch = distance.expand(batch_size, -1, -1, -1)
    #             scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)

    #         elif self.map_version == '1.3':
    #             scene_images = scene_images.to(self.device)

    #         num_past_agents = num_past_agents.to(self.device)
    #         past_agents_traj = past_agents_traj.to(self.device)
    #         past_agents_traj_len = past_agents_traj_len.to(self.device)
    #         past_agents_traj_len_idx = past_agents_traj_len_idx.to(self.device)

    #         future_agent_masks = future_agent_masks.to(self.device)
    #         future_agents_traj = future_agents_traj.to(self.device)
    #         future_agents_traj_len = future_agents_traj_len.to(self.device)
    #         future_agents_traj_len_idx = future_agents_traj_len_idx.to(self.device)

    #         two_mask = two_mask.to(self.device)
    #         three_mask = three_mask.to(self.device)
            
    #         num_future_agents = num_future_agents.to(self.device)
    #         decode_start_vel = decode_start_vel.to(self.device)
    #         decode_start_pos = decode_start_pos.to(self.device)

    #         agent_tgt_three_mask = torch.zeros_like(future_agent_masks)
    #         agent_masks_idx = torch.arange(len(future_agent_masks), device=self.device)[future_agent_masks][three_mask]
    #         agent_tgt_three_mask[agent_masks_idx] = True
            
            

    #         if self.flow_based_decoder:
    #             # Normalizing Flow (q loss)
    #             # z_: A X Td X 2
    #             # mu_: A X Td X 2
    #             # sigma_: A X Td X 2 X 2
    #             # Generate perturbation

    #             # R2p2-ma
    #             src_trajs = src_trajs.to(self.device)[agent_masks][tgt_three_mask]
    #             tgt_trajs = tgt_trajs.to(self.device)[tgt_three_mask]

    #             decode_start_vel = decode_start_vel.to(self.device)[tgt_three_mask]
    #             decode_start_pos = decode_start_pos.to(self.device)[tgt_three_mask]

    #             # Total number of three-masked agents in this batch
    #             with torch.no_grad():
    #                 num_tgt_trajs = num_tgt_trajs.to(self.device)
    #                 episode_idx = torch.arange(len(num_tgt_trajs), device=self.device).repeat_interleave(num_tgt_trajs)
    #                 episode_idx = episode_idx[tgt_three_mask]
    #                 total_three_agents = episode_idx.size(0)

    #             perterb = torch.normal(mean=0.0, std=np.sqrt(0.001), size=future_agents_traj.shape, device=self.device)
    #             log_prior = log_prior.to(self.device)

    #             if self.model_name == 'R2P2_SimpleRNN':
    #                 z_, mu_, sigma_, motion_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, decode_start_vel, decode_start_pos)

    #             elif self.model_name == 'R2P2_RNN':
    #                 # input_dict={
    #                 #     future_agents_traj+perterb, 
    #                 #     past_agents_traj, 
    #                 #     episode_idx, 
    #                 #     decode_start_vel, 
    #                 #     decode_start_pos, 
    #                 #     scene_images
    #                 # }
    #                 # z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(**input_dict)
    #                 z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, episode_idx, decode_start_vel, decode_start_pos, scene_images)

    #             elif self.model_name == 'CAM_NFDecoder':
    #                 z_, mu_, sigma_, motion_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_past_agents)

    #             elif self.model_name == 'Scene_CAM_NFDecoder':
    #                 z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)

    #             elif self.model_name == 'Global_Scene_CAM_NFDecoder':
    #                 z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)

    #             elif self.model_name == 'AttGlobal_Scene_CAM_NFDecoder':
    #                 z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)

    #             z_ = z_.reshape((num_three_agents, -1)) # A X (Td*2)
    #             log_q0 = c1 - 0.5 * ((z_ ** 2).sum(dim=1))

    #             logdet_sigma = log_determinant(sigma_)

    #             log_qpi = log_q0 - logdet_sigma.sum(dim=1)
    #             qloss = -log_qpi
    #             batch_qloss = qloss.mean()
                
    #             # Prior Loss (p loss)
    #             if self.model_name == 'R2P2_SimpleRNN':
    #                 gen_trajs, z, mu, sigma = self.model(motion_encoding_, decode_start_vel, decode_start_pos, motion_encoded=True)

    #             elif self.model_name == 'R2P2_RNN':
    #                 gen_trajs, z, mu, sigma = self.model(motion_encoding_, episode_idx, decode_start_vel, decode_start_pos, scene_encoding_, motion_encoded=True, scene_encoded=True)

    #             elif self.model_name == 'CAM_NFDecoder':
    #                 gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_past_agents, agent_encoded=True)

    #             elif self.model_name == 'Scene_CAM_NFDecoder':
    #                 gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_encoding_, agent_encoded=True, scene_encoded=True)

    #             elif self.model_name == 'Global_Scene_CAM_NFDecoder':
    #                 gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_encoding_, agent_encoded=True, scene_encoded=True)

    #             elif self.model_name == 'AttGlobal_Scene_CAM_NFDecoder':
    #                 gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_encoding_, agent_encoded=True, scene_encoded=True)

    #             if self.beta != 0.0:
    #                 if self.ploss_type == 'mseloss':
    #                     ploss = self.ploss_criterion(gen_trajs, past_agents_traj)
    #                 else:
    #                     ploss = self.ploss_criterion(episode_idx, gen_trajs, log_prior, -15.0)

    #             else:
    #                 ploss = torch.zeros(size=(1,), device=self.device)

    #             batch_ploss = ploss.mean()
    #             batch_loss = batch_qloss + self.beta * batch_ploss

    #             epoch_ploss += batch_ploss.item() * batch_size
    #             epoch_qloss += batch_qloss.item() * batch_size
            
    #         episode_idx = torch.arange(len(num_past_agents), device=self.device).repeat_interleave(num_past_agents)

    #         # Inference
    #         # TODO: Remove dependency of names on decoding, add a parameter
    #         if 'SimpleEncoderDecoder' == self.model_name:
    #             # import pdb; pdb.set_trace()
    #             # try:
    #             input_dict={
    #                 "past_agents_traj": past_agents_traj,
    #                 "past_agents_traj_len": past_agents_traj_len,
    #                 "future_agent_masks": future_agent_masks,
    #                 "decode_start_vel": decode_start_vel,
    #                 "decode_start_pos": decode_start_pos
    #             }

    #             predicted_trajs = self.model(**input_dict)

    #         elif 'SocialPooling' == self.model_name:
    #             input_dict={
    #                 "past_agents_traj":past_agents_traj, 
    #                 "past_agents_traj_len":past_agents_traj_len, 
    #                 "episode_idx":episode_idx, 
    #                 "future_agent_masks":future_agent_masks,
    #                 "decode_start_vel":decode_start_vel, 
    #                 "decode_start_pos":decode_start_pos
    #                     }   
    #             predicted_trajs = self.model(input_dict)             

    #         elif 'GAN' in self.model_name:
    #             stochastic = True
    #             input_dict={
    #                 "past_agents_traj":past_agents_traj, 
    #                 "past_agents_traj_len":past_agents_traj_len, 
    #                 "episode_idx":episode_idx, 
    #                 "future_agent_masks":future_agent_masks,
    #                 "decode_start_vel":decode_start_vel, 
    #                 "decode_start_pos":decode_start_pos,
    #                 "scene_images":scene_images, 
    #                 "stochastic":stochastic,
    #                 "num_candidate": self.num_candidates_train
    #             }
    #             predicted_trajs = self.model(**input_dict)
    #             predicted_trajs_ = predicted_trajs.reshape(total_future_agent, self.num_candidates_train, self.decoding_steps, 2)
            

    #         elif 'MATF' in self.model_name:
    #             stochastic = False             
    #             input_dict={
    #                 "past_agents_traj":past_agents_traj, 
    #                 "past_agents_traj_len":past_agents_traj_len, 
    #                 "episode_idx":episode_idx, 
    #                 "future_agent_masks":future_agent_masks,
    #                 "decode_start_vel":decode_start_vel, 
    #                 "decode_start_pos":decode_start_pos,
    #                 "scene_images":scene_images, 
    #                 "stochastic":stochastic
    #                     }   
    #             predicted_trajs = self.model(input_dict)
            
            
    #         elif 'CAM' == self.model_name:
    #             input_dict={
    #                 "past_agents_traj":past_agents_traj, 
    #                 "past_agents_traj_len":past_agents_traj_len, 
    #                 "agent_tgt_three_mask":agent_tgt_three_mask, 
    #                 "future_agent_masks":future_agent_masks,
    #                 "decode_start_vel":decode_start_vel, 
    #                 "decode_start_pos":decode_start_pos,
    #                 "num_past_agents":num_past_agents
    #                     }   
    #             gen_trajs = self.model(input_dict)
    #             gen_trajs = gen_trajs.reshape(num_three_agents, self.num_candidates, self.decoding_steps, 2)
            
    #         agent_time_index = torch.arange(num_agents, device=self.device).repeat_interleave(future_agents_traj_len)
    #         time_normalizer = future_agents_traj_len.float().repeat_interleave(future_agents_traj_len)

    #         error = future_agents_traj - predicted_trajs # A x Td x 2
    #         batch_loss = (error ** 2).sum(dim=-1) # x**2 + y**2 | A x Td 
    #         batch_loss = batch_loss[agent_time_index, future_agents_traj_len_idx] / time_normalizer
    #         batch_loss = batch_loss.sum() / (num_agents * 2.0)

    #         if self.flow_based_decoder:
    #             rs_error3 = ((gen_trajs - future_agents_traj.unsqueeze(1)) ** 2).sum(dim=-1).sqrt_() # A X candi X T X 2 >> A X candi X T
    #             rs_error2 = rs_error3[..., :int(self.decoding_steps*2/3)]
                
    #             num_agents = gen_trajs.size(0)
    #             num_agents2 = rs_error2.size(0)
    #             num_agents3 = rs_error3.size(0)

    #             ade2 = rs_error2.mean(-1) #  A X candi X T >> A X candi
    #             fde2 = rs_error2[..., -1]

    #             minade2, _ = ade2.min(dim=-1) # A X candi >> A
    #             avgade2 = ade2.mean(dim=-1)
    #             minfde2, _ = fde2.min(dim=-1)
    #             avgfde2 = fde2.mean(dim=-1)

    #             batch_minade2 = minade2.mean() # A >> 1
    #             batch_minfde2 = minfde2.mean()
    #             batch_avgade2 = avgade2.mean()
    #             batch_avgfde2 = avgfde2.mean()

    #             ade3 = rs_error3.mean(-1)
    #             fde3 = rs_error3[..., -1]

    #             minade3, _ = ade3.min(dim=-1)
    #             avgade3 = ade3.mean(dim=-1)
    #             minfde3, _ = fde3.min(dim=-1)
    #             avgfde3 = fde3.mean(dim=-1)

    #             batch_minade3 = minade3.mean()
    #             batch_minfde3 = minfde3.mean()
    #             batch_avgade3 = avgade3.mean()
    #             batch_avgfde3 = avgfde3.mean()

    #             if 'CAM' == self.model_name:
    #                 batch_loss = batch_minade3
    #                 epoch_loss += batch_loss.item()
    #                 batch_qloss = torch.zeros(1)
    #                 batch_ploss = torch.zeros(1)

    #             # Loss backward
    #             batch_loss.backward()
    #             self.optimizer.step()

    #             print("Working on train batch {:d}/{:d}... ".format(b+1, len(self.train_loader)) +
    #             "batch_loss: {:.2f}, qloss: {:.2f}, ploss: {:g}, ".format(batch_loss.item(), batch_qloss.item(), batch_ploss.item()) +
    #             "minFDE3: {:.2f}, avgFDE3: {:.2f}".format(batch_minfde3.item(), batch_avgfde3.item()), end='\r')

    #             epoch_minade2 += batch_minade2.item() * num_agents2
    #             epoch_avgade2 += batch_avgade2.item() * num_agents2
    #             epoch_minfde2 += batch_minfde2.item() * num_agents2
    #             epoch_avgfde2 += batch_avgfde2.item() * num_agents2
    #             epoch_minade3 += batch_minade3.item() * num_agents3
    #             epoch_avgade3 += batch_avgade3.item() * num_agents3
    #             epoch_minfde3 += batch_minfde3.item() * num_agents3
    #             epoch_avgfde3 += batch_avgfde3.item() * num_agents3

    #             epoch_agents += num_agents
    #             epoch_agents2 += num_agents2
    #             epoch_agents3 += num_agents3

    #         if self.generative:
    #             # Length for the concatnated trajectory
    #             agents_total_traj_len = past_agents_traj_len.clone()
    #             agents_total_traj_len[future_agent_masks] += future_agents_traj_len
    #             agents_total_traj_len_ = agents_total_traj_len.repeat_interleave(self.num_candidates_train)

    #             # Fill past trajectories
    #             past_agent_idx = torch.arange(total_past_agent, device=self.device)
    #             past_agent_idx_repeat = past_agent_idx.repeat_interleave(past_agents_traj_len)
    #             real_trajs = torch.zeros((total_past_agent, self.encoding_steps+self.decoding_steps, 2), device=self.device)
    #             real_trajs[past_agent_idx_repeat, past_agents_traj_len_idx] = past_agents_traj[past_agent_idx_repeat, past_agents_traj_len_idx]
                
    #             # Copy and repeat the past trajectories
    #             fake_trajs = real_trajs.clone().unsqueeze(0).repeat(self.num_candidates_train, 1, 1, 1)

    #             # Fill real future
    #             decoding_agent_idx = past_agent_idx[future_agent_masks]
    #             decoding_agent_idx_repeat = decoding_agent_idx.repeat_interleave(future_agents_traj_len)
    #             future_agent_idx = torch.arange(total_future_agent, device=self.device)
    #             future_agent_idx_repeat = future_agent_idx.repeat_interleave(future_agents_traj_len)
                
    #             shifted_future_agents_traj_len_idx = future_agents_traj_len_idx + past_agents_traj_len[future_agent_masks].repeat_interleave(future_agents_traj_len, 0)
                
    #             real_trajs[decoding_agent_idx_repeat, shifted_future_agents_traj_len_idx] = future_agents_traj[future_agent_idx_repeat, future_agents_traj_len_idx]

    #             # Fill Fake future
    #             fake_trajs[:, decoding_agent_idx_repeat, shifted_future_agents_traj_len_idx] = predicted_trajs_[future_agent_idx_repeat, :, future_agents_traj_len_idx].transpose(0, 1)
    #             fake_trajs = fake_trajs.reshape(total_past_agent*self.num_candidates_train, self.encoding_steps+self.decoding_steps, 2)

    #             # Calculate discriminator score
    #             true_score = self.discriminator(real_trajs, agents_total_traj_len, episode_idx, future_agent_masks, decode_start_pos, scene_images, 1)  # [num_agents X 1]
                
    #             num_past_agents_ = num_past_agents.repeat(self.num_candidates_train)
    #             batch_size_ = len(num_past_agents_)
    #             episode_idx_ = torch.arange(batch_size_, device=self.device).repeat_interleave(num_past_agents_, 0)
    #             future_agent_masks_ = future_agent_masks.repeat(self.num_candidates_train)
    #             decode_start_pos_ = decode_start_pos.repeat(self.num_candidates_train, 1)
                
    #             fake_score = self.discriminator(fake_trajs, agents_total_traj_len_, episode_idx_, future_agent_masks_, decode_start_pos_, scene_images, self.num_candidates_train)  # [num_agents X 1]

    #             ### Train Generator (i.e. MATF decoder)
    #             self.model.require_grad = True
    #             self.discriminator.require_grad = False

    #             error = future_agents_traj.unsqueeze(1) - predicted_trajs_ # Na x Nc x T x 2
    #             error = (error ** 2).sum(dim=-1) # Na x Nc x T
    #             rs_error = error.sqrt()

    #             # Two-Errors
    #             rs_error3 = rs_error[three_mask] # Na x Nc x T
    #             rs_error2 = rs_error[two_mask, :, :int(self.decoding_steps*2/3)]

    #             agent_time_index = torch.arange(total_future_agent, device=self.device).repeat_interleave(future_agents_traj_len)
    #             time_normalizer = future_agents_traj_len.float().repeat_interleave(future_agents_traj_len)

    #             batch_loss = error[agent_time_index, :, future_agents_traj_len_idx] / time_normalizer.unsqueeze(1)
    #             batch_loss = batch_loss.sum() / (total_future_agent * 2.0)

    #             num_agents2 = rs_error2.size(0)
    #             num_agents3 = rs_error3.size(0)

    #             ade2 = rs_error2.mean(-1) #  A X candi X T >> A X candi
    #             fde2 = rs_error2[..., -1]

    #             minade2, _ = ade2.min(dim=-1) # A X candi >> A
    #             avgade2 = ade2.mean(dim=-1)
    #             minfde2, _ = fde2.min(dim=-1)
    #             avgfde2 = fde2.mean(dim=-1)

    #             batch_minade2 = minade2.mean() # A >> 1
    #             batch_minfde2 = minfde2.mean()
    #             batch_avgade2 = avgade2.mean()
    #             batch_avgfde2 = avgfde2.mean()

    #             ade3 = rs_error3.mean(-1)
    #             fde3 = rs_error3[..., -1]

    #             minade3, _ = ade3.min(dim=-1)
    #             avgade3 = ade3.mean(dim=-1)
    #             minfde3, _ = fde3.min(dim=-1)
    #             avgfde3 = fde3.mean(dim=-1)

    #             batch_minade3 = minade3.mean()
    #             batch_minfde3 = minfde3.mean()
    #             batch_avgade3 = avgade3.mean()
    #             batch_avgfde3 = avgfde3.mean()

    #             batch_adversarial_loss = self.adversarial_loss(fake_score, torch.ones_like(fake_score))
    #             batch_g_loss = batch_loss + (batch_adversarial_loss * gan_weight)

    #             # Loss backward
    #             batch_g_loss.backward(retain_graph=True)
    #             self.optimizer.step()

    #             ### Train Discriminator
    #             self.discriminator.require_grad = True
    #             self.model.require_grad = False

    #             real_loss = self.adversarial_loss(true_score, torch.ones_like(true_score))
    #             fake_loss = self.adversarial_loss(fake_score, torch.zeros_like(fake_score))
    #             batch_d_loss = gan_weight*(real_loss + fake_loss)

    #             batch_d_loss.backward()
    #             self.optimizer_D.step()

    #             epoch_g_loss += batch_g_loss.item()
    #             epoch_d_loss += batch_d_loss.item()
                
    #             epoch_minade2 += batch_minade2.item() * num_agents2
    #             epoch_minfde2 += batch_minfde2.item() * num_agents2
    #             epoch_avgade2 += batch_avgade2.item() * num_agents2
    #             epoch_avgfde2 += batch_avgfde2.item() * num_agents2

    #             epoch_minade3 += batch_minade3.item() * num_agents3
    #             epoch_minfde3 += batch_minfde3.item() * num_agents3
    #             epoch_avgade3 += batch_avgade3.item() * num_agents3
    #             epoch_avgfde3 += batch_avgfde3.item() * num_agents3

    #             epoch_agents += total_future_agent
    #             epoch_agents2 += num_agents2
    #             epoch_agents3 += num_agents3

    #         else:
    #             with torch.no_grad():

    #                 # Two-Errors
    #                 sq_error2 = (error[two_mask, :int(self.decoding_steps*2/3), :] ** 2).sum(2).sqrt() # A X Td X 2 >> A X Td
    #                 sq_error3 = (error[three_mask, :, :] ** 2).sum(2).sqrt()

    #                 ## ## TODO check the reshape purpose once again, not sure about this part.
    #                 ## sq_error = sq_error.reshape((-1))

    #                 num_agents2 = sq_error2.size(0)
    #                 num_agents3 = sq_error3.size(0)

    #                 ade2 = sq_error2.mean(dim=-1) # A X T >> A
    #                 fde2 = sq_error2[... , -1]
    #                 ade3 = sq_error3.mean(dim=-1)
    #                 fde3 = sq_error3[... , -1]
                    
    #                 # 2-Loss
    #                 batch_minade2 = ade2.mean() # A >> 1
    #                 batch_minfde2 = fde2.mean()
    #                 batch_avgade2 = ade2.mean()
    #                 batch_avgfde2 = fde2.mean()
                    
    #                 # 3-Loss
    #                 batch_minade3 = ade3.mean()
    #                 batch_minfde3 = fde3.mean()
    #                 batch_avgade3 = ade3.mean()
    #                 batch_avgfde3 = fde3.mean()


    #             # Loss backward
    #             batch_loss.backward()
    #             self.optimizer.step()

    #             epoch_loss += batch_loss.item()

    #             epoch_minade2 += batch_minade2.item() * num_agents2
    #             epoch_minfde2 += batch_minfde2.item() * num_agents2
    #             epoch_avgade2 += batch_avgade2.item() * num_agents2
    #             epoch_avgfde2 += batch_avgfde2.item() * num_agents2

    #             epoch_minade3 += batch_minade3.item() * num_agents3
    #             epoch_minfde3 += batch_minfde3.item() * num_agents3
    #             epoch_avgade3 += batch_avgade3.item() * num_agents3
    #             epoch_avgfde3 += batch_avgfde3.item() * num_agents3

    #             ## TODO check the reason why here is future while in GAN version if accepts past.
    #             epoch_agents += len(future_agents_traj_len)
    #             epoch_agents2 += num_agents2
    #             epoch_agents3 += num_agents3


    #     if self.generative:
    #         epoch_g_loss /= epoch_agents
    #         epoch_d_loss /= epoch_agents
    #         epoch_loss = epoch_g_loss+epoch_d_loss
    #     elif self.flow_based_decoder:
    #         epoch_ploss /= epoch_agents
    #         epoch_qloss /= epoch_agents
    #         epoch_loss = epoch_qloss + self.beta * epoch_ploss
    #     else:
    #         epoch_loss /= epoch_agents

    #     # 2-Loss
    #     epoch_minade2 /= epoch_agents2
    #     epoch_avgade2 /= epoch_agents2
    #     epoch_minfde2 /= epoch_agents2
    #     epoch_avgfde2 /= epoch_agents2

    #     # 3-Loss
    #     epoch_minade3 /= epoch_agents3
    #     epoch_avgade3 /= epoch_agents3
    #     epoch_minfde3 /= epoch_agents3
    #     epoch_avgfde3 /= epoch_agents3

    #     epoch_ades = ( epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3 )
    #     epoch_fdes = ( epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3 )

    #     scheduler_metric = epoch_avgade3 + epoch_avgfde3 
    #     self.optimizer.zero_grad()
    #     torch.cuda.empty_cache()
    #     if self.generative:
    #         return epoch_loss, epoch_g_loss, epoch_d_loss, epoch_ades, epoch_fdes
    #     elif self.flow_based_decoder:
    #         return epoch_loss, epoch_qloss, epoch_ploss, epoch_ades, epoch_fdes
    #     else:
    #         return epoch_loss, epoch_ades, epoch_fdes

    def inference(self):
        self.model.eval()  # Set model to evaluate mode.
        
        with torch.no_grad():
            epoch_loss = 0.0
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_minade2, epoch_avgade2 = 0.0, 0.0
            epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
            epoch_minade3, epoch_avgade3 = 0.0, 0.0
            epoch_minfde3, epoch_avgfde3 = 0.0, 0.0

            if self.flow_based_decoder:
                epoch_qloss = 0.0
                epoch_ploss = 0.0
        
            epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

            H = W = 60
            if '2.' in self.map_version:
                coordinate_2d = np.indices((H, W))
                coordinate = np.ravel_multi_index(coordinate_2d, dims=(H, W))
                coordinate = torch.FloatTensor(coordinate)
                coordinate = coordinate.reshape((1, 1, H, W))

                coordinate_std, coordinate_mean = torch.std_mean(coordinate)
                coordinate = (coordinate - coordinate_mean) / coordinate_std

                distance_2d = coordinate_2d - np.array([H/2 - 0.5, W/2 - 0.5]).reshape((2, 1, 1))
                distance = np.sqrt((distance_2d ** 2).sum(axis=0))
                distance = torch.FloatTensor(distance)
                distance = distance.reshape((1, 1, H, W))

                distance_std, distance_mean = torch.std_mean(distance)
                distance = (distance - distance_mean) / distance_std
                
                coordinate = coordinate.to(self.device)
                distance = distance.to(self.device)

            if self.generative:
                for i, e in enumerate(self.gan_weight_schedule):
                    if epoch <= e:
                        gan_weight = self.gan_weight[i]
                        break
            if self.flow_based_decoder:
                c1 = -self.decoding_steps * np.log(2 * np.pi)

            for b, batch in enumerate(self.valid_loader):

                print("Working on batch {:d}/{:d}".format(b+1, len(self.valid_loader)), end='\r')
                if self.generative:
                    self.optimizer.zero_grad()
                    self.optimizer_D.zero_grad()

                scene_images, log_prior, \
                future_agent_masks, \
                num_past_agents,   past_agents_traj,   past_agents_traj_len,   past_agents_traj_len_idx, \
                num_future_agents, future_agents_traj, future_agents_traj_len, future_agents_traj_len_idx, \
                two_mask, three_mask, \
                decode_start_vel, decode_start_pos, \
                scene_id = batch
                
                # Detect dynamic batch size
                batch_size = scene_images.size(0)
                num_agents = future_agents_traj.size(0)
                                # batch_size = scene_images.size(0)
                total_past_agent = past_agents_traj.size(0)
                total_future_agent = future_agents_traj.size(0)
                num_three_agents = torch.sum(three_mask)
                
                # if '2.' in self.map_version:
                #     coordinate_batch = coordinate.expand(batch_size, -1, -1, -1)
                #     distance_batch = distance.expand(batch_size, -1, -1, -1)
                #     scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)

                # elif self.map_version == '1.3':
                #     scene_images = scene_images.to(self.device)

                if '2.' in self.map_version:
                    coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                    distance_batch = distance.repeat(batch_size, 1, 1, 1)
                    scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)

                elif self.map_version == '1.3':
                    scene_images = scene_images.to(self.device)

                # Upload to GPU
                num_past_agents = num_past_agents.to(self.device)
                past_agents_traj = past_agents_traj.to(self.device)
                past_agents_traj_len = past_agents_traj_len.to(self.device)
                past_agents_traj_len_idx = past_agents_traj_len_idx.to(self.device)

                future_agent_masks = future_agent_masks.to(self.device)
                future_agents_traj = future_agents_traj.to(self.device)
                future_agents_traj_len = future_agents_traj_len.to(self.device)
                future_agents_traj_len_idx = future_agents_traj_len_idx.to(self.device)

                num_future_agents = num_future_agents.to(self.device)
                two_mask = two_mask.to(self.device)
                
                if self.flow_based_decoder:
                    three_mask = three_mask.to(self.device)
                    future_agents_traj = future_agents_traj.to(self.device)[three_mask]
                    future_agents_traj_len = future_agents_traj_len.to(self.device)[three_mask]

                    decode_start_vel = decode_start_vel.to(self.device)[agent_tgt_three_mask]
                    decode_start_pos = decode_start_pos.to(self.device)[agent_tgt_three_mask]
                    log_prior = log_prior.to(self.device)
                else:
                    decode_start_vel = decode_start_vel.to(self.device)
                    decode_start_pos = decode_start_pos.to(self.device)
                
                import pdb; pdb.set_trace()
                if self.generative:
                    episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_past_agents)
                if self.flow_based_decoder:
                    episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_future_agents)[three_mask]
                else:
                    episode_idx = torch.arange(len(num_past_agents), device=self.device).repeat_interleave(num_past_agents)

                # Inference
                if 'SimpleEncoderDecoder' == self.model_name:
                    input_dict={
                        "past_agents_traj": past_agents_traj,
                        "past_agents_traj_len": past_agents_traj_len,
                        "future_agent_masks": future_agent_masks,
                        "decode_start_vel": decode_start_vel,
                        "decode_start_pos": decode_start_pos
                    }
                    predicted_trajs = self.model(input_dict)

                elif 'SocialPooling' == self.model_name:
                    input_dict={
                        "past_agents_traj":past_agents_traj, 
                        "past_agents_traj_len":past_agents_traj_len, 
                        "episode_idx":episode_idx, 
                        "future_agent_masks":future_agent_masks,
                        "decode_start_vel":decode_start_vel, 
                        "decode_start_pos":decode_start_pos
                            }   
                    predicted_trajs = self.model(**input_dict)  

                elif 'MATF' in self.model_name:
                    stochastic = False
                    input_dict={
                        "past_agents_traj":past_agents_traj, 
                        "past_agents_traj_len":past_agents_traj_len, 
                        "episode_idx":episode_idx, 
                        "future_agent_masks":future_agent_masks,
                        "decode_start_vel":decode_start_vel, 
                        "decode_start_pos":decode_start_pos,
                        "scene_images":scene_images, 
                        "stochastic":stochastic
                            }   
                    predicted_trajs = self.model(**input_dict)

                elif 'GAN' in self.model_name:
                    stochastic = True
                    input_dict={
                        "past_agents_traj":past_agents_traj, 
                        "past_agents_traj_len":past_agents_traj_len, 
                        "episode_idx":episode_idx, 
                        "future_agent_masks":future_agent_masks,
                        "decode_start_vel":decode_start_vel, 
                        "decode_start_pos":decode_start_pos,
                        "scene_images":scene_images, 
                        "stochastic":stochastic,
                        "num_candidates_train": self.num_candidates_train
                    }
                    import pdb; pdb.set_trace()
                    predicted_trajs = self.model(**input_dict)
                    predicted_trajs_ = predicted_trajs.reshape(total_future_agent, self.num_candidates, self.decoding_steps, 2)
                
                # if 'CAM' == self.model_name:
                #     gen_trajs = self.model(past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_past_agents)
                #     gen_trajs = gen_trajs.reshape(num_three_agents, self.num_candidates, self.decoding_steps, 2)
                #     rs_error3 = ((gen_trajs - future_agents_traj.unsqueeze(1)) ** 2).sum(dim=-1).sqrt_()
                #     rs_error2 = rs_error3[..., :int(self.decoding_steps*2/3)]
                    
                #     num_agents = gen_trajs.size(0)
                #     num_agents2 = rs_error2.size(0)
                #     num_agents3 = rs_error3.size(0)

                #     ade2 = rs_error2.mean(-1)
                #     fde2 = rs_error2[..., -1]

                #     minade2, _ = ade2.min(dim=-1)
                #     avgade2 = ade2.mean(dim=-1)
                #     minfde2, _ = fde2.min(dim=-1)
                #     avgfde2 = fde2.mean(dim=-1)

                #     batch_minade2 = minade2.mean()
                #     batch_minfde2 = minfde2.mean()
                #     batch_avgade2 = avgade2.mean()
                #     batch_avgfde2 = avgfde2.mean()

                #     ade3 = rs_error3.mean(-1)
                #     fde3 = rs_error3[..., -1]

                #     minade3, _ = ade3.min(dim=-1)
                #     avgade3 = ade3.mean(dim=-1)
                #     minfde3, _ = fde3.min(dim=-1)
                #     avgfde3 = fde3.mean(dim=-1)

                #     batch_minade3 = minade3.mean()
                #     batch_minfde3 = minfde3.mean()
                #     batch_avgade3 = avgade3.mean()
                #     batch_avgfde3 = avgfde3.mean()

                #     if self.flow_based_decoder is not True:
                #         batch_loss = batch_minade3
                #         epoch_loss += batch_loss.item()
                #         batch_qloss = torch.zeros(1)
                #         batch_ploss = torch.zeros(1)

                #     print("Working on valid batch {:d}/{:d}... ".format(b+1, len(self.valid_loader)) +
                #     "batch_loss: {:.2f}, qloss: {:.2f}, ploss: {:g}, ".format(batch_loss.item(), batch_qloss.item(), batch_ploss.item()) +
                #     "minFDE3: {:.2f}, avgFDE3: {:.2f}".format(batch_minfde3.item(), batch_avgfde3.item()), end='\r')

                #     epoch_ploss += batch_ploss.item() * batch_size
                #     epoch_qloss += batch_qloss.item() * batch_size
                #     epoch_minade2 += batch_minade2.item() * num_agents2
                #     epoch_avgade2 += batch_avgade2.item() * num_agents2
                #     epoch_minfde2 += batch_minfde2.item() * num_agents2
                #     epoch_avgfde2 += batch_avgfde2.item() * num_agents2
                #     epoch_minade3 += batch_minade3.item() * num_agents3
                #     epoch_avgade3 += batch_avgade3.item() * num_agents3
                #     epoch_minfde3 += batch_minfde3.item() * num_agents3
                #     epoch_avgfde3 += batch_avgfde3.item() * num_agents3

                #     epoch_agents += num_agents
                #     epoch_agents2 += num_agents2
                #     epoch_agents3 += num_agents3
                else:
                    raise ValueError("Unknown model type {:s}.".format(self.model_name))

                if self.flow_based_decoder:
                    # Normalizing Flow (q loss)
                    # z: A X Td X 2
                    # mu: A X Td X 2
                    # sigma: A X Td X 2 X 2
                    # Generate perturbation
                    perterb = torch.normal(mean=0.0, std=np.sqrt(0.001), size=future_agents_traj.shape, device=self.device)
                    
                    if self.model_name == 'R2P2_SimpleRNN':
                        z_, mu_, sigma_, motion_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, decode_start_vel, decode_start_pos)
                    elif self.model_name == 'R2P2_RNN':
                        z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, episode_idx, decode_start_vel, decode_start_pos, scene_images)
                    elif self.model_name == 'CAM_NFDecoder':
                        z_, mu_, sigma_, motion_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_past_agents)
                    elif self.model_name == 'Scene_CAM_NFDecoder':
                        z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)
                    elif self.model_name == 'Global_Scene_CAM_NFDecoder':
                        z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)
                    elif self.model_name == 'AttGlobal_Scene_CAM_NFDecoder':
                        z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)
                    z_ = z_.reshape((num_three_agents, -1)) # A X (Td*2)
                    log_q0 = c1 - 0.5 * ((z_ ** 2).sum(dim=1))
                    logdet_sigma = log_determinant(sigma_)
                    log_qpi = log_q0 - logdet_sigma.sum(dim=1)
                    qloss = -log_qpi
                    batch_qloss = qloss.mean()

                    # Prior Loss (p loss)
                    if self.model_name == 'R2P2_SimpleRNN':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, decode_start_vel, decode_start_pos, motion_encoded=True)
                    elif self.model_name == 'R2P2_RNN':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, episode_idx, decode_start_vel, decode_start_pos, scene_encoding_, motion_encoded=True, scene_encoded=True)
                    elif self.model_name == 'CAM_NFDecoder':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_past_agents, agent_encoded=True)
                    elif self.model_name == 'Scene_CAM_NFDecoder':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_encoding_, agent_encoded=True, scene_encoded=True)
                    elif self.model_name == 'Global_Scene_CAM_NFDecoder':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_encoding_, agent_encoded=True, scene_encoded=True)
                    elif self.model_name == 'AttGlobal_Scene_CAM_NFDecoder':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_encoding_, agent_encoded=True, scene_encoded=True)
                    if self.beta != 0.0:
                        if self.ploss_type == 'mseloss':
                            ploss = self.ploss_criterion(gen_trajs, past_agents_traj)
                        else:
                            ploss = self.ploss_criterion(episode_idx, gen_trajs, log_prior, -15.0)
                    else:
                        ploss = torch.zeros(size=(1,), device=self.device)
                    batch_ploss = ploss.mean()
                    batch_loss = batch_qloss + self.beta * batch_ploss
                    epoch_ploss += batch_ploss.item() * batch_size
                    epoch_qloss += batch_qloss.item() * batch_size  

                    rs_error3 = ((gen_trajs - future_agents_traj.unsqueeze(1)) ** 2).sum(dim=-1).sqrt_()
                    rs_error2 = rs_error3[..., :int(self.decoding_steps*2/3)]
                    
                    num_agents = gen_trajs.size(0)
                    num_agents2 = rs_error2.size(0)
                    num_agents3 = rs_error3.size(0)

                    ade2 = rs_error2.mean(-1)
                    fde2 = rs_error2[..., -1]
                    minade2, _ = ade2.min(dim=-1)
                    avgade2 = ade2.mean(dim=-1)
                    minfde2, _ = fde2.min(dim=-1)
                    avgfde2 = fde2.mean(dim=-1)
                    batch_minade2 = minade2.mean()
                    batch_minfde2 = minfde2.mean()
                    batch_avgade2 = avgade2.mean()
                    batch_avgfde2 = avgfde2.mean()
                    ade3 = rs_error3.mean(-1)
                    fde3 = rs_error3[..., -1]
                    minade3, _ = ade3.min(dim=-1)
                    avgade3 = ade3.mean(dim=-1)
                    minfde3, _ = fde3.min(dim=-1)
                    avgfde3 = fde3.mean(dim=-1)
                    batch_minade3 = minade3.mean()
                    batch_minfde3 = minfde3.mean()
                    batch_avgade3 = avgade3.mean()
                    batch_avgfde3 = avgfde3.mean()

                    # Loss backward
                    batch_loss.backward()
                    self.optimizer.step()

                    print("Working on valid batch {:d}/{:d}... ".format(b+1, len(self.valid_loader)) +
                    "batch_loss: {:.2f}, qloss: {:.2f}, ploss: {:g}, ".format(batch_loss.item(), batch_qloss.item(), batch_ploss.item()) +
                    "minFDE3: {:.2f}, avgFDE3: {:.2f}".format(batch_minfde3.item(), batch_avgfde3.item()), end='\r')
                    epoch_ploss += batch_ploss.item() * batch_size
                    epoch_qloss += batch_qloss.item() * batch_size
                    epoch_minade2 += batch_minade2.item() * num_agents2
                    epoch_avgade2 += batch_avgade2.item() * num_agents2
                    epoch_minfde2 += batch_minfde2.item() * num_agents2
                    epoch_avgfde2 += batch_avgfde2.item() * num_agents2
                    epoch_minade3 += batch_minade3.item() * num_agents3
                    epoch_avgade3 += batch_avgade3.item() * num_agents3
                    epoch_minfde3 += batch_minfde3.item() * num_agents3
                    epoch_avgfde3 += batch_avgfde3.item() * num_agents3
                    epoch_agents += num_agents
                    epoch_agents2 += num_agents2
                    epoch_agents3 += num_agents3
        
                elif self.generative:
                    # Length for the concatnated trajectory
                    agents_total_traj_len = past_agents_traj_len.clone()
                    agents_total_traj_len[future_agent_masks] += future_agents_traj_len
                    agents_total_traj_len_ = agents_total_traj_len.repeat_interleave(self.num_candidates)

                    # Fill past trajectories
                    past_agent_idx = torch.arange(total_past_agent, device=self.device)
                    past_agent_idx_repeat = past_agent_idx.repeat_interleave(past_agents_traj_len)
                    real_trajs = torch.zeros((total_past_agent, self.encoding_steps+self.decoding_steps, 2), device=self.device)
                    real_trajs[past_agent_idx_repeat, past_agents_traj_len_idx] = past_agents_traj[past_agent_idx_repeat, past_agents_traj_len_idx]
                    
                    # Copy and repeat the past trajectories
                    fake_trajs = real_trajs.clone().unsqueeze(0).repeat(self.num_candidates, 1, 1, 1)

                    # Fill real future
                    decoding_agent_idx = past_agent_idx[future_agent_masks]
                    decoding_agent_idx_repeat = decoding_agent_idx.repeat_interleave(future_agents_traj_len)
                    future_agent_idx = torch.arange(total_future_agent, device=self.device)
                    future_agent_idx_repeat = future_agent_idx.repeat_interleave(future_agents_traj_len)
                    
                    shifted_future_agents_traj_len_idx = future_agents_traj_len_idx + past_agents_traj_len[future_agent_masks].repeat_interleave(future_agents_traj_len, 0)
                    
                    real_trajs[decoding_agent_idx_repeat, shifted_future_agents_traj_len_idx] = future_agents_traj[future_agent_idx_repeat, future_agents_traj_len_idx]

                    # Fill Fake future
                    fake_trajs[:, decoding_agent_idx_repeat, shifted_future_agents_traj_len_idx] = predicted_trajs_[future_agent_idx_repeat, :, future_agents_traj_len_idx].transpose(0, 1)
                    fake_trajs = fake_trajs.reshape(total_past_agent*self.num_candidates, self.encoding_steps+self.decoding_steps, 2)

                    # Calculate discriminator score
                    true_score = self.discriminator(real_trajs, agents_total_traj_len, episode_idx, future_agent_masks, decode_start_pos, scene_images, 1)  # [num_agents X 1]
                    
                    num_past_agents_ = num_past_agents.repeat(self.num_candidates)
                    batch_size_ = len(num_past_agents_)
                    episode_idx_ = torch.arange(batch_size_, device=self.device).repeat_interleave(num_past_agents_, 0)
                    future_agent_masks_ = future_agent_masks.repeat(self.num_candidates)
                    decode_start_pos_ = decode_start_pos.repeat(self.num_candidates, 1)

                    fake_score = self.discriminator(fake_trajs, agents_total_traj_len_, episode_idx_, future_agent_masks_, decode_start_pos_, scene_images, self.num_candidates)  # [num_agents X 1]

                    ### Evaluate Generator (i.e. MATF decoder)
                    error = future_agents_traj.unsqueeze(1) - predicted_trajs_ # Na x Nc x T x 2
                    error = (error ** 2).sum(dim=-1) # Na x Nc x T
                    rs_error = error.sqrt()

                    # Two-Errors
                    rs_error3 = rs_error[three_mask] # Na x Nc x T
                    rs_error2 = rs_error[two_mask, :, :int(self.decoding_steps*2/3)]

                    agent_time_index = torch.arange(total_future_agent, device=self.device).repeat_interleave(future_agents_traj_len)
                    time_normalizer = future_agents_traj_len.float().repeat_interleave(future_agents_traj_len)

                    batch_loss = error[agent_time_index, :, future_agents_traj_len_idx] / time_normalizer.unsqueeze(1)
                    batch_loss = batch_loss.sum() / (total_future_agent * 2.0)

                    num_agents2 = rs_error2.size(0)
                    num_agents3 = rs_error3.size(0)

                    ade2 = rs_error2.mean(-1) #  A X candi X T >> A X candi
                    fde2 = rs_error2[..., -1]

                    minade2, _ = ade2.min(dim=-1) # A X candi >> A
                    avgade2 = ade2.mean(dim=-1)
                    minfde2, _ = fde2.min(dim=-1)
                    avgfde2 = fde2.mean(dim=-1)

                    batch_minade2 = minade2.mean() # A >> 1
                    batch_minfde2 = minfde2.mean()
                    batch_avgade2 = avgade2.mean()
                    batch_avgfde2 = avgfde2.mean()

                    ade3 = rs_error3.mean(-1)
                    fde3 = rs_error3[..., -1]

                    minade3, _ = ade3.min(dim=-1)
                    avgade3 = ade3.mean(dim=-1)
                    minfde3, _ = fde3.min(dim=-1)
                    avgfde3 = fde3.mean(dim=-1)

                    batch_minade3 = minade3.mean()
                    batch_minfde3 = minfde3.mean()
                    batch_avgade3 = avgade3.mean()
                    batch_avgfde3 = avgfde3.mean()

                    batch_adversarial_loss = self.adversarial_loss(fake_score, torch.ones_like(fake_score))
                    batch_g_loss = batch_loss + (batch_adversarial_loss * gan_weight)

                    # Evaluate D loss
                    real_loss = self.adversarial_loss(true_score, torch.ones_like(true_score))
                    fake_loss = self.adversarial_loss(fake_score, torch.zeros_like(fake_score))
                    batch_d_loss = gan_weight*(real_loss + fake_loss)

                    epoch_g_loss += batch_g_loss.item()
                    epoch_d_loss += batch_d_loss.item()
                    
                    epoch_minade2 += batch_minade2.item() * num_agents2
                    epoch_minfde2 += batch_minfde2.item() * num_agents2
                    epoch_avgade2 += batch_avgade2.item() * num_agents2
                    epoch_avgfde2 += batch_avgfde2.item() * num_agents2

                    epoch_minade3 += batch_minade3.item() * num_agents3
                    epoch_minfde3 += batch_minfde3.item() * num_agents3
                    epoch_avgade3 += batch_avgade3.item() * num_agents3
                    epoch_avgfde3 += batch_avgfde3.item() * num_agents3

                    epoch_agents += total_future_agent
                    epoch_agents2 += num_agents2
                    epoch_agents3 += num_agents3

                else:
                    if 'CAM' == self.model_name:
                        gen_trajs = self.model(past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_past_agents)
                        gen_trajs = gen_trajs.reshape(num_three_agents, self.num_candidates, self.decoding_steps, 2)
                        rs_error3 = ((gen_trajs - future_agents_traj.unsqueeze(1)) ** 2).sum(dim=-1).sqrt_() # A X candi X T X 2 >> A X candi X T
                        rs_error2 = rs_error3[..., :int(self.decoding_steps*2/3)]
                        
                        num_agents = gen_trajs.size(0)
                        num_agents2 = rs_error2.size(0)
                        num_agents3 = rs_error3.size(0)

                        ade2 = rs_error2.mean(-1) #  A X candi X T >> A X candi
                        fde2 = rs_error2[..., -1]

                        minade2, _ = ade2.min(dim=-1) # A X candi >> A
                        avgade2 = ade2.mean(dim=-1)
                        minfde2, _ = fde2.min(dim=-1)
                        avgfde2 = fde2.mean(dim=-1)

                        batch_minade2 = minade2.mean() # A >> 1
                        batch_minfde2 = minfde2.mean()
                        batch_avgade2 = avgade2.mean()
                        batch_avgfde2 = avgfde2.mean()

                        ade3 = rs_error3.mean(-1)
                        fde3 = rs_error3[..., -1]

                        minade3, _ = ade3.min(dim=-1)
                        avgade3 = ade3.mean(dim=-1)
                        minfde3, _ = fde3.min(dim=-1)
                        avgfde3 = fde3.mean(dim=-1)

                        batch_minade3 = minade3.mean()
                        batch_minfde3 = minfde3.mean()
                        batch_avgade3 = avgade3.mean()
                        batch_avgfde3 = avgfde3.mean()

                        if self.flow_based_decoder is not True:
                            batch_loss = batch_minade3
                            epoch_loss += batch_loss.item()
                            batch_qloss = torch.zeros(1)
                            batch_ploss = torch.zeros(1)

                        # Loss backward
                        batch_loss.backward()
                        self.optimizer.step()

                        print("Working on train batch {:d}/{:d}... ".format(b+1, len(self.train_loader)) +
                        "batch_loss: {:.2f}, qloss: {:.2f}, ploss: {:g}, ".format(batch_loss.item(), batch_qloss.item(), batch_ploss.item()) +
                        "minFDE3: {:.2f}, avgFDE3: {:.2f}".format(batch_minfde3.item(), batch_avgfde3.item()), end='\r')

                        epoch_minade2 += batch_minade2.item() * num_agents2
                        epoch_avgade2 += batch_avgade2.item() * num_agents2
                        epoch_minfde2 += batch_minfde2.item() * num_agents2
                        epoch_avgfde2 += batch_avgfde2.item() * num_agents2
                        epoch_minade3 += batch_minade3.item() * num_agents3
                        epoch_avgade3 += batch_avgade3.item() * num_agents3
                        epoch_minfde3 += batch_minfde3.item() * num_agents3
                        epoch_avgfde3 += batch_avgfde3.item() * num_agents3

                        epoch_agents += num_agents
                        epoch_agents2 += num_agents2
                        epoch_agents3 += num_agents3
                        break

                    agent_time_index = torch.arange(num_agents, device=self.device).repeat_interleave(future_agents_traj_len)
                    time_normalizer = future_agents_traj_len.float().repeat_interleave(future_agents_traj_len)

                    error = future_agents_traj - predicted_trajs # A x Td x 2
                    batch_loss = (error ** 2).sum(dim=-1) # x**2 + y**2 | A x Td 
                    batch_loss = batch_loss[agent_time_index, future_agents_traj_len_idx] / time_normalizer
                    batch_loss = batch_loss.sum() / (num_agents * 2.0)

                    sq_error2 = (error[two_mask, :int(self.decoding_steps*2/3), :] ** 2).sum(2).sqrt()
                    sq_error3 = (error[three_mask,:, :] ** 2).sum(2).sqrt()

                    num_agents2 = sq_error2.size(0)
                    num_agents3 = sq_error3.size(0)

                    ade2 = sq_error2.mean(dim=-1) # A X T >> A
                    fde2 = sq_error2[... , -1]
                    ade3 = sq_error3.mean(dim=-1)
                    fde3 = sq_error3[... , -1]
                    
                    # 2-Loss
                    batch_minade2 = ade2.mean() # A >> 1
                    batch_minfde2 = fde2.mean()
                    batch_avgade2 = ade2.mean()
                    batch_avgfde2 = fde2.mean()
                    
                    # 3-Loss
                    batch_minade3 = ade3.mean()
                    batch_minfde3 = fde3.mean()
                    batch_avgade3 = ade3.mean()
                    batch_avgfde3 = fde3.mean()

                    epoch_loss += batch_loss.item()

                    epoch_minade2 += batch_minade2.item() * num_agents2
                    epoch_avgade2 += batch_avgade2.item() * num_agents2
                    epoch_minfde2 += batch_minfde2.item() * num_agents2
                    epoch_avgfde2 += batch_avgfde2.item() * num_agents2

                    epoch_minade3 += batch_minade3.item() * num_agents3
                    epoch_avgade3 += batch_avgade3.item() * num_agents3
                    epoch_minfde3 += batch_minfde3.item() * num_agents3
                    epoch_avgfde3 += batch_avgfde3.item() * num_agents3

                    epoch_agents += len(future_agents_traj_len)
                    epoch_agents2 += num_agents2
                    epoch_agents3 += num_agents3
            
        if self.flow_based_decoder:
            epoch_ploss /= epoch_agents
            epoch_qloss /= epoch_agents
            epoch_loss = epoch_qloss + self.beta * epoch_ploss
            
            # 2-Loss
            epoch_minade2 /= epoch_agents2
            epoch_avgade2 /= epoch_agents2
            epoch_minfde2 /= epoch_agents2
            epoch_avgfde2 /= epoch_agents2
            
            # 3-Loss
            epoch_minade3 /= epoch_agents3
            epoch_avgade3 /= epoch_agents3
            epoch_minfde3 /= epoch_agents3
            epoch_avgfde3 /= epoch_agents3
            epoch_ades = ( epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3 )
            epoch_fdes = ( epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3 )
            
            scheduler_metric = epoch_avgade3 + epoch_avgfde3 
            return epoch_loss, epoch_qloss, epoch_ploss, epoch_ades, epoch_fdes, scheduler_metric

        if self.generative:
            epoch_minade2 /= epoch_agents2
            epoch_avgade2 /= epoch_agents2
            epoch_minfde2 /= epoch_agents2
            epoch_avgfde2 /= epoch_agents2
            epoch_minade3 /= epoch_agents3
            epoch_avgade3 /= epoch_agents3
            epoch_minfde3 /= epoch_agents3
            epoch_avgfde3 /= epoch_agents3

            epoch_ades = [epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3]
            epoch_fdes = [epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3]

            self.optimizer.zero_grad()
            torch.cuda.empty_cache()
            return epoch_loss, epoch_qloss, epoch_ploss, epoch_ades, epoch_fdes

        else:
            epoch_g_loss /= epoch_agents
            epoch_d_loss /= epoch_agents

            # 2-Loss
            epoch_minade2 /= epoch_agents2
            epoch_avgade2 /= epoch_agents2
            epoch_minfde2 /= epoch_agents2
            epoch_avgfde2 /= epoch_agents2

            # 3-Loss
            epoch_minade3 /= epoch_agents3
            epoch_avgade3 /= epoch_agents3
            epoch_minfde3 /= epoch_agents3
            epoch_avgfde3 /= epoch_agents3
            
            epoch_ades = ( epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3 )
            epoch_fdes = ( epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3 )

            scheduler_metric = epoch_avgade3 + epoch_avgfde3 
            return epoch_ades, epoch_fdes, scheduler_metric




















# TESTER

    def run(self):
        print('Starting model test.....')
        self.model.eval()  # Set model to evaluate mode.
        
        with torch.no_grad():

            if self.generative:
                epoch_g_loss = 0.0
                epoch_d_loss = 0.0
                epoch_adv_loss = 0.0
            else:
                epoch_loss = 0.0
                epoch_qloss = 0.0
                epoch_ploss = 0.0
                epoch_minmsd, epoch_avgmsd = 0.0, 0.0
            
            epoch_minade2, epoch_avgade2 = 0.0, 0.0
            epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
            epoch_minade3, epoch_avgade3 = 0.0, 0.0
            epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
            epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

            epoch_dao = 0.0
            epoch_dac = 0.0
            dao_agents = 0.0
            dac_agents = 0.0

            # pool = Pool(5)
            pool = Pool(processes=5)

            H = W = 60
            if self.map_version and '2.' in self.map_version:
                coordinate_2d = np.indices((H, W))
                coordinate = np.ravel_multi_index(coordinate_2d, dims=(H, W))
                coordinate = torch.FloatTensor(coordinate)
                coordinate = coordinate.reshape((1, 1, H, W))

                coordinate_std, coordinate_mean = torch.std_mean(coordinate)
                coordinate = (coordinate - coordinate_mean) / coordinate_std

                if self.generative:
                    distance_2d = coordinate_2d - np.array([(H-1)/2, (H-1)/2]).reshape((2, 1, 1))
                else:
                    distance_2d = coordinate_2d - np.array([H/2 - 0.5, W/2 - 0.5]).reshape((2, 1, 1))
                distance = np.sqrt((distance_2d ** 2).sum(axis=0))
                distance = torch.FloatTensor(distance)
                distance = distance.reshape((1, 1, H, W))

                distance_std, distance_mean = torch.std_mean(distance)
                distance = (distance - distance_mean) / distance_std
                
                coordinate = coordinate.to(self.device)
                distance = distance.to(self.device)

            c1 = -self.decoding_steps * np.log(2 * np.pi)

            if self.flow_based_decoder:
                epoch_dao = 0.0
                epoch_dac = 0.0
                dao_agents = 0.0
                dac_agents = 0.0

                pool = Pool(5)

                H = W = 64
                with torch.no_grad():
                    if self.map_version == '2.0':
                        coordinate_2d = np.indices((H, W))
                        coordinate = np.ravel_multi_index(coordinate_2d, dims=(H, W))
                        coordinate = torch.FloatTensor(coordinate)
                        coordinate = coordinate.reshape((1, 1, H, W))

                        coordinate_std, coordinate_mean = torch.std_mean(coordinate)
                        coordinate = (coordinate - coordinate_mean) / coordinate_std

                        distance_2d = coordinate_2d - np.array([(H-1)/2, (H-1)/2]).reshape((2, 1, 1))
                        distance = np.sqrt((distance_2d ** 2).sum(axis=0))
                        distance = torch.FloatTensor(distance)
                        distance = distance.reshape((1, 1, H, W))

                        distance_std, distance_mean = torch.std_mean(distance)
                        distance = (distance - distance_mean) / distance_std
                    
                        coordinate = coordinate.to(self.device)
                        distance = distance.to(self.device)
                    
                    c1 = -self.decoding_steps * np.log(2 * np.pi)
                    for b, batch in enumerate(self.data_loader):
                        
                        # if b > 5:
                        #     break

                        scene_images, log_prior, \
                        agent_masks, \
                        num_src_trajs, src_trajs, src_lens, src_len_idx, \
                        num_tgt_trajs, tgt_trajs, tgt_lens, tgt_len_idx, \
                        tgt_two_mask, tgt_three_mask, \
                        decode_start_vel, decode_start_pos, scene_id = batch

                        # Detect dynamic batch size
                        batch_size = scene_images.size(0)
                        # num_encoding_agents = src_trajs.size(0)
                        # num_decoding_agents = tgt_trajs.size(0)
                        num_three_agents = torch.sum(tgt_three_mask)

                        if self.map_version == '2.0':
                            coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                            distance_batch = distance.repeat(batch_size, 1, 1, 1)
                            scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)
                        
                        src_trajs = src_trajs.to(self.device)
                        src_lens = src_lens.to(self.device)

                        tgt_trajs = tgt_trajs.to(self.device)[tgt_three_mask]
                        tgt_lens = tgt_lens.to(self.device)[tgt_three_mask]

                        num_tgt_trajs = num_tgt_trajs.to(self.device)
                        episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_tgt_trajs)[tgt_three_mask]

                        agent_masks = agent_masks.to(self.device)
                        agent_tgt_three_mask = torch.zeros_like(agent_masks)
                        agent_masks_idx = torch.arange(len(agent_masks), device=self.device)[agent_masks][tgt_three_mask]
                        agent_tgt_three_mask[agent_masks_idx] = True

                        decode_start_vel = decode_start_vel.to(self.device)[agent_tgt_three_mask]
                        decode_start_pos = decode_start_pos.to(self.device)[agent_tgt_three_mask]

                        log_prior = log_prior.to(self.device)

                        if self.flow_based_decoder:
                            # Normalizing Flow (q loss)
                            # z: A X Td X 2
                            # mu: A X Td X 2
                            # sigma: A X Td X 2 X 2
                            # Generate perturbation
                            perterb = torch.normal(mean=0.0, std=np.sqrt(0.001), size=tgt_trajs.shape, device=self.device)
                            
                            if self.model_name == 'R2P2_SimpleRNN':
                                z_, mu_, sigma_, motion_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, decode_start_vel, decode_start_pos)

                            elif self.model_name == 'R2P2_RNN':
                                z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, episode_idx, decode_start_vel, decode_start_pos, scene_images)

                            elif self.model_name == 'CAM_NFDecoder':
                                z_, mu_, sigma_, motion_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, src_lens, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_src_trajs)

                            elif self.model_name == 'Scene_CAM_NFDecoder':
                                z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, src_lens, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_src_trajs, scene_images)

                            elif self.model_name == 'Global_Scene_CAM_NFDecoder':
                                z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, src_lens, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_src_trajs, scene_images)

                            elif self.model_name == 'AttGlobal_Scene_CAM_NFDecoder':
                                z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, src_lens, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_src_trajs, scene_images)

                            z_ = z_.reshape((num_three_agents, -1)) # A X (Td*2)
                            log_q0 = c1 - 0.5 * ((z_ ** 2).sum(dim=1))

                            logdet_sigma = log_determinant(sigma_)

                            log_qpi = log_q0 - logdet_sigma.sum(dim=1)
                            qloss = -log_qpi
                            batch_qloss = qloss.mean()

                            # Prior Loss (p loss)
                            if self.model_name == 'R2P2_SimpleRNN':
                                gen_trajs, z, mu, sigma = self.model(motion_encoding_, decode_start_vel, decode_start_pos, motion_encoded=True)

                            elif self.model_name == 'R2P2_RNN':
                                gen_trajs, z, mu, sigma = self.model(motion_encoding_, episode_idx, decode_start_vel, decode_start_pos, scene_encoding_, motion_encoded=True, scene_encoded=True)

                            elif self.model_name == 'CAM_NFDecoder':
                                gen_trajs, z, mu, sigma = self.model(motion_encoding_, src_lens, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_src_trajs, agent_encoded=True)

                            elif self.model_name == 'Scene_CAM_NFDecoder':
                                gen_trajs, z, mu, sigma = self.model(motion_encoding_, src_lens, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_src_trajs, scene_encoding_, agent_encoded=True, scene_encoded=True)

                            elif self.model_name == 'Global_Scene_CAM_NFDecoder':
                                gen_trajs, z, mu, sigma = self.model(motion_encoding_, src_lens, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_src_trajs, scene_encoding_, agent_encoded=True, scene_encoded=True)

                            elif self.model_name == 'AttGlobal_Scene_CAM_NFDecoder':
                                gen_trajs, z, mu, sigma = self.model(motion_encoding_, src_lens, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_src_trajs, scene_encoding_, agent_encoded=True, scene_encoded=True)

                            # if self.beta != 0.0:
                            #     if self.ploss_criterion == 'mseloss':
                            #         # Simple Gaussian Prior
                            #         error = gen_trajs - tgt_trajs.unsqueeze(1)
                            #         ploss = (error ** 2).sum((2,3)).mean(1)

                            #     else:
                            #         # NF based prior
                            #         # Calculate P(x)
                            #         episode_idx = episode_idx.repeat_interleave(self.num_candidates)
                            #         ploss, coord = self.ploss_criterion(episode_idx, gen_trajs.reshape(num_three_agents*self.num_candidates, self.decoding_steps, 2), log_prior, -15.0)
                            #         ploss = ploss.reshape(num_three_agents, self.num_candidates, self.decoding_steps)
                            #         ploss = ploss.sum(2).mean(1)

                            # else:
                            #     ploss = torch.zeros(size=(1,), device=self.device)

                            if self.beta != 0.0:
                                if self.ploss_type == 'mseloss':
                                    ploss = self.ploss_criterion(gen_trajs, tgt_trajs)
                                else:
                                    ploss = self.ploss_criterion(episode_idx, gen_trajs, log_prior, -15.0)

                            else:
                                ploss = torch.zeros(size=(1,), device=self.device)
                            batch_ploss = ploss.mean()
                            batch_loss = batch_qloss + self.beta * batch_ploss

                            epoch_ploss += batch_ploss.item() * batch_size
                            epoch_qloss += batch_qloss.item() * batch_size   

                        else:

                            if 'CAM' == self.model_name:
                                gen_trajs = self.model(src_trajs, src_lens, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_src_trajs)                                                    

                            gen_trajs = gen_trajs.reshape(num_three_agents, self.num_candidates, self.decoding_steps, 2)


                        rs_error3 = ((gen_trajs - tgt_trajs.unsqueeze(1)) ** 2).sum(dim=-1).sqrt_()
                        rs_error2 = rs_error3[..., :int(self.decoding_steps*2/3)]
                        
                        
                        diff = gen_trajs - tgt_trajs.unsqueeze(1)
                        msd_error = (diff[:,:,:,0] ** 2 + diff[:,:,:,1] ** 2)
                        
                        num_agents = gen_trajs.size(0)
                        num_agents2 = rs_error2.size(0)
                        num_agents3 = rs_error3.size(0)

                        ade2 = rs_error2.mean(-1)
                        fde2 = rs_error2[..., -1]

                        minade2, _ = ade2.min(dim=-1)
                        avgade2 = ade2.mean(dim=-1)
                        minfde2, _ = fde2.min(dim=-1)
                        avgfde2 = fde2.mean(dim=-1)

                        batch_minade2 = minade2.mean()
                        batch_minfde2 = minfde2.mean()
                        batch_avgade2 = avgade2.mean()
                        batch_avgfde2 = avgfde2.mean()

                        ade3 = rs_error3.mean(-1)
                        fde3 = rs_error3[..., -1]
                        
                        
                        msd = msd_error.mean(-1)
                        minmsd, _ = msd.min(dim=-1)
                        avgmsd = msd.mean(dim=-1)
                        batch_minmsd = minmsd.mean()
                        batch_avgmsd = avgmsd.mean()


                        minade3, _ = ade3.min(dim=-1)
                        avgade3 = ade3.mean(dim=-1)
                        minfde3, _ = fde3.min(dim=-1)
                        avgfde3 = fde3.mean(dim=-1)

                        batch_minade3 = minade3.mean()
                        batch_minfde3 = minfde3.mean()
                        batch_avgade3 = avgade3.mean()
                        batch_avgfde3 = avgfde3.mean()

                        if self.flow_based_decoder is not True:
                            batch_loss = batch_minade3
                            epoch_loss += batch_loss.item()
                            batch_qloss = torch.zeros(1)
                            batch_ploss = torch.zeros(1)

                        print("Working on test batch {:d}/{:d}... ".format(b+1, len(self.data_loader)) +
                        "batch_loss: {:.2f}, qloss: {:.2f}, ploss: {:g}, ".format(batch_loss.item(), batch_qloss.item(), batch_ploss.item()) +
                        "minFDE3: {:.2f}, avgFDE3: {:.2f}".format(batch_minfde3.item(), batch_avgfde3.item()), end='\r')

                        epoch_ploss += batch_ploss.item() * batch_size
                        epoch_qloss += batch_qloss.item() * batch_size
                        epoch_minade2 += batch_minade2.item() * num_agents2
                        epoch_avgade2 += batch_avgade2.item() * num_agents2
                        epoch_minfde2 += batch_minfde2.item() * num_agents2
                        epoch_avgfde2 += batch_avgfde2.item() * num_agents2
                        epoch_minade3 += batch_minade3.item() * num_agents3
                        epoch_avgade3 += batch_avgade3.item() * num_agents3
                        epoch_minfde3 += batch_minfde3.item() * num_agents3
                        epoch_avgfde3 += batch_avgfde3.item() * num_agents3
                        

                        epoch_minmsd += batch_minmsd.item() * num_agents3
                        epoch_avgmsd += batch_avgmsd.item() * num_agents3

                        epoch_agents += num_agents
                        epoch_agents2 += num_agents2
                        epoch_agents3 += num_agents3
                        
                        map_files = self.map_file(scene_id)
                        folder_path = os.path.join(self.exp_path, "images")
                        if not os.path.exists(folder_path): os.makedirs(folder_path)
                        output_files = [ folder_path + '/' + x[2] + '_' + x[3] + '.jpg' for x in scene_id]

                        cum_num_tgt_trajs = [0] + torch.cumsum(num_tgt_trajs, dim=0).tolist()
                        cum_num_src_trajs = [0] + torch.cumsum(num_src_trajs, dim=0).tolist()

                        ## Currently, for CAM model, gen_traj and src_traj does not shown well.

                        src_trajs = src_trajs.cpu().numpy()
                        src_lens = src_lens.cpu().numpy()

                        tgt_trajs = tgt_trajs.cpu().numpy()
                        tgt_lens = tgt_lens.cpu().numpy()

                        zero_ind = np.nonzero(tgt_three_mask.numpy() == 0)[0]
                        zero_ind -= np.arange(len(zero_ind))

                        tgt_three_mask = tgt_three_mask.numpy()
                        agent_tgt_three_mask = agent_tgt_three_mask.cpu().numpy()

                        gen_trajs = gen_trajs.cpu().numpy()

                        src_mask = agent_tgt_three_mask

                        gen_trajs = np.insert(gen_trajs, zero_ind, 0, axis=0)

                        tgt_trajs = np.insert(tgt_trajs, zero_ind, 0, axis=0)
                        tgt_lens = np.insert(tgt_lens, zero_ind, 0, axis=0)

                        for i in range(batch_size):
                            candidate_i = gen_trajs[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]
                            tgt_traj_i = tgt_trajs[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]
                            tgt_lens_i = tgt_lens[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]

                            src_traj_i = src_trajs[cum_num_src_trajs[i]:cum_num_src_trajs[i+1]]
                            src_lens_i = src_lens[cum_num_src_trajs[i]:cum_num_src_trajs[i+1]]
                            map_file_i = map_files[i]
                            output_file_i = output_files[i]
                            
                            candidate_i = candidate_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]]
                            tgt_traj_i = tgt_traj_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]]
                            tgt_lens_i = tgt_lens_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]]

                            src_traj_i = src_traj_i[agent_tgt_three_mask[cum_num_src_trajs[i]:cum_num_src_trajs[i+1]]]
                            src_lens_i = src_lens_i[agent_tgt_three_mask[cum_num_src_trajs[i]:cum_num_src_trajs[i+1]]]

                            # dao_i, dao_mask_i = self.dao(candidate_i, map_file_i)
                            # dac_i, dac_mask_i = self.dac(candidate_i, map_file_i)

                            # epoch_dao += dao_i.sum()
                            # dao_agents += dao_mask_i.sum()

                            # epoch_dac += dac_i.sum()
                            # dac_agents += dac_mask_i.sum()

                            # if self.render:
                            #     pool.apply(self.write_img_output, (candidate_i, src_traj_i, src_lens_i, tgt_traj_i, tgt_lens_i, map_file_i, output_file_i))
                
                        # break

                if self.flow_based_decoder:
                    epoch_ploss /= epoch_agents
                    epoch_qloss /= epoch_agents
                    epoch_loss = epoch_qloss + self.beta * epoch_ploss
                else:
                    epoch_loss /= epoch_agents

                # 2-Loss
                epoch_minade2 /= epoch_agents2
                epoch_avgade2 /= epoch_agents2
                epoch_minfde2 /= epoch_agents2
                epoch_avgfde2 /= epoch_agents2

                # 3-Loss
                epoch_minade3 /= epoch_agents3
                epoch_avgade3 /= epoch_agents3
                epoch_minfde3 /= epoch_agents3
                epoch_avgfde3 /= epoch_agents3

                epoch_minmsd /= epoch_agents3
                epoch_avgmsd /= epoch_agents3

                # epoch_dao /= dao_agents
                # epoch_dac /= dac_agents

                epoch_ades = ( epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3 )
                epoch_fdes = ( epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3 )

                print("--Final Performane Report--")
                print("minADE2: {:.5f}, minFDE2: {:.5f}, avgADE2: {:.5f}, avgFDE2: {:.5f}".format(epoch_minade2, epoch_minfde2, epoch_avgade2, epoch_avgfde2))
                print("minADE3: {:.5f}, minFDE3: {:.5f}, avgADE3: {:.5f}, avgFDE3: {:.5f}".format(epoch_minade3, epoch_minfde3, epoch_avgade3, epoch_avgfde3))
                print("minMSD: {:.5f}, avgMSD: {:5f}".format(epoch_minmsd, epoch_avgmsd))
                print("QLoss: {:.5f}, PLoss: {:5f}".format(epoch_qloss, epoch_ploss))
                # print("DAO: {:.5f}e-5, DAC: {:.5f}".format(epoch_dao * 10000.0, epoch_dac))
                with open(self.exp_path + f'/test_metrics_{self.datetime}.pkl', 'wb') as f:
                    pkl.dump({"ADEs": epoch_ades,
                            "FDEs": epoch_fdes,
                            "Qloss": epoch_qloss,
                            "Ploss": epoch_ploss, 
                            #   "DAO": epoch_dao,
                            }, f)
                return 

            for b, batch in enumerate(self.data_loader):
                # if b > 5:
                #     break

                if self.generative:
                    scene_images, log_prior, \
                    future_agent_masks, \
                    num_past_agents,   past_agents_traj,   past_agents_traj_len,   past_agents_traj_len_idx, \
                    num_future_agents, future_agents_traj, future_agents_traj_len, future_agents_traj_len_idx, \
                    two_mask, three_mask, \
                    decode_start_vel, decode_start_pos, \
                    scene_id = batch

                elif self.flow_based_decoder:
                    scene_images, log_prior, \
                    agent_masks, \
                    num_src_trajs, src_trajs, src_lens, src_len_idx, \
                    num_tgt_trajs, tgt_trajs, tgt_lens, tgt_len_idx, \
                    tgt_two_mask, tgt_three_mask, \
                    decode_start_vel, decode_start_pos, scene_id = batch
                    
                else:
                    scene_images, log_prior, \
                    future_agent_masks, \
                    num_past_agents,   past_agents_traj,   past_agents_traj_len,   past_agents_traj_len_idx, \
                    num_future_agents, future_agents_traj, future_agents_traj_len, future_agents_traj_len_idx, \
                    two_mask, three_mask, \
                    decode_start_vel, decode_start_pos, \
                    scene_id = batch

                # Detect dynamic batch size
                batch_size = scene_images.size(0)
                if self.generative:
                    total_past_agent = past_agents_traj.size(0)
                    total_future_agent = future_agents_traj.size(0)
                else:
                    num_agents = future_agents_traj.size(0)

                if self.generative:
                    if '2.' in self.map_version:
                        coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                        distance_batch = distance.repeat(batch_size, 1, 1, 1)
                        scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)

                    elif self.map_version == '1.3':
                        scene_images = scene_images.to(self.device)

                elif self.map_version:
                    if '2.' in self.map_version:
                        coordinate_batch = coordinate.expand(batch_size, -1, -1, -1)
                        distance_batch = distance.expand(batch_size, -1, -1, -1)
                        scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)

                    elif self.map_version == '1.3':
                        scene_images = scene_images.to(self.device)

                if self.generative:
                    num_past_agents = num_past_agents.to(self.device)
                    past_agents_traj = past_agents_traj.to(self.device)
                    past_agents_traj_len = past_agents_traj_len.to(self.device)
                    past_agents_traj_len_idx = past_agents_traj_len_idx.to(self.device)

                    future_agent_masks = future_agent_masks.to(self.device)
                    future_agents_traj = future_agents_traj.to(self.device)
                    future_agents_traj_len = future_agents_traj_len.to(self.device)
                    future_agents_traj_len_idx = future_agents_traj_len_idx.to(self.device)

                    two_mask = two_mask.to(self.device)
                    three_mask = three_mask.to(self.device)

                    num_future_agents = num_future_agents.to(self.device)
                    decode_start_vel = decode_start_vel.to(self.device)
                    decode_start_pos = decode_start_pos.to(self.device)

                    episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_past_agents)
                    if 'GAN' in self.model_name:
                        stochastic = True
                        predicted_trajs = self.model(past_agents_traj, past_agents_traj_len, episode_idx, future_agent_masks,
                                                    decode_start_vel, decode_start_pos, scene_images, stochastic, self.num_candidates)
                        predicted_trajs_ = predicted_trajs.reshape(total_future_agent, self.num_candidates, self.decoding_steps, 2)
                    
                    else:
                        raise ValueError("Unknown model type {:s}.".format(self.model_name))
                    
    
                    ### Evaluate Generator (i.e. MATF decoder)
                    error = future_agents_traj.unsqueeze(1) - predicted_trajs_ # Na x Nc x T x 2
                    error = (error ** 2).sum(dim=-1) # Na x Nc x T
                    rs_error = error.sqrt()
                    # Two-Errors
                    rs_error3 = rs_error[three_mask] # Na x Nc x T
                    rs_error2 = rs_error[two_mask, :, :int(self.decoding_steps*2/3)]

                    agent_time_index = torch.arange(total_future_agent, device=self.device).repeat_interleave(future_agents_traj_len)
                    time_normalizer = future_agents_traj_len.float().repeat_interleave(future_agents_traj_len)

                    batch_loss = error[agent_time_index, :, future_agents_traj_len_idx] / time_normalizer.unsqueeze(1)
                    batch_loss = batch_loss.sum() / (total_future_agent * 2.0)

                    num_agents2 = rs_error2.size(0)
                    num_agents3 = rs_error3.size(0)

                    ade2 = rs_error2.mean(-1) #  A X candi X T >> A X candi
                    fde2 = rs_error2[..., -1]

                    minade2, _ = ade2.min(dim=-1) # A X candi >> A
                    avgade2 = ade2.mean(dim=-1)
                    minfde2, _ = fde2.min(dim=-1)
                    avgfde2 = fde2.mean(dim=-1)

                    batch_minade2 = minade2.mean() # A >> 1
                    batch_minfde2 = minfde2.mean()
                    batch_avgade2 = avgade2.mean()
                    batch_avgfde2 = avgfde2.mean()

                    ade3 = rs_error3.mean(-1)
                    fde3 = rs_error3[..., -1]

                    minade3, _ = ade3.min(dim=-1)
                    avgade3 = ade3.mean(dim=-1)
                    minfde3, _ = fde3.min(dim=-1)
                    avgfde3 = fde3.mean(dim=-1)

                    batch_minade3 = minade3.mean()
                    batch_minfde3 = minfde3.mean()
                    batch_avgade3 = avgade3.mean()
                    batch_avgfde3 = avgfde3.mean()

                    print("Working on test batch {:d}/{:d}... ".format(b+1, len(self.data_loader)) +
                    "batch_loss: {:.2f}, ".format(batch_loss.item()) +
                    "minFDE3: {:.2f}, avgFDE3: {:.2f}".format(batch_minfde3.item(), batch_avgfde3.item()), end='\r')

                    epoch_minade2 += batch_minade2.item() * num_agents2
                    epoch_minfde2 += batch_minfde2.item() * num_agents2
                    epoch_avgade2 += batch_avgade2.item() * num_agents2
                    epoch_avgfde2 += batch_avgfde2.item() * num_agents2

                    epoch_minade3 += batch_minade3.item() * num_agents3
                    epoch_minfde3 += batch_minfde3.item() * num_agents3
                    epoch_avgade3 += batch_avgade3.item() * num_agents3
                    epoch_avgfde3 += batch_avgfde3.item() * num_agents3

                    epoch_agents += total_future_agent
                    epoch_agents2 += num_agents2
                    epoch_agents3 += num_agents3

                    torch.cuda.empty_cache()
                    
                    map_files = self.map_file(scene_id)
                    output_files = [self.exp_path + '/' + x[2] + '.jpg' for x in scene_id]

                    cum_num_future_agents = [0] + torch.cumsum(num_future_agents, dim=0).tolist()

                    predicted_trajs_ = predicted_trajs_.cpu().numpy()

                    past_agents_traj = past_agents_traj.cpu().numpy()
                    past_agents_traj_len = past_agents_traj_len.cpu().numpy()

                    future_agents_traj = future_agents_traj.cpu().numpy()
                    future_agents_traj_len = future_agents_traj_len.cpu().numpy()

                    for i in range(batch_size):
                        candidate_i   = predicted_trajs_[    cum_num_future_agents[i]:cum_num_future_agents[i+1]]
                        future_traj_i = future_agents_traj[cum_num_future_agents[i]:cum_num_future_agents[i+1]]
                        future_lens_i = future_agents_traj_len[ cum_num_future_agents[i]:cum_num_future_agents[i+1]]
                        past_traj_i   = past_agents_traj[  cum_num_future_agents[i]:cum_num_future_agents[i+1]]
                        past_lens_i   = past_agents_traj_len[   cum_num_future_agents[i]:cum_num_future_agents[i+1]]

                        map_file_i = map_files[i]
                        output_file_i = output_files[i]

                        dao_i, dao_mask_i = self.dao(candidate_i, map_file_i)
                        dac_i, dac_mask_i = self.dac(candidate_i, map_file_i)

                        epoch_dao += dao_i.sum()
                        dao_agents += dao_mask_i.sum()

                        epoch_dac += dac_i.sum()
                        dac_agents += dac_mask_i.sum()

                        if self.render:
                            pool.apply(self.write_img_output, 
                                    (candidate_i, past_traj_i, past_lens_i, future_traj_i, future_lens_i, map_file_i, output_file_i)
                            )
                else:
                    # Upload to GPU
                    num_past_agents = num_past_agents.to(self.device)
                    past_agents_traj = past_agents_traj.to(self.device)
                    past_agents_traj_len = past_agents_traj_len.to(self.device)

                    future_agents_traj = future_agents_traj.to(self.device)
                    future_agents_traj_len = future_agents_traj_len.to(self.device)
                    future_agents_traj_len_idx = future_agents_traj_len_idx.to(self.device)

                    two_mask = two_mask.to(self.device)
                    three_mask = three_mask.to(self.device)

                    num_future_agents = num_future_agents.to(self.device)
                    decode_start_vel = decode_start_vel.to(self.device)
                    decode_start_pos = decode_start_pos.to(self.device)

                    episode_idx = torch.arange(len(num_past_agents), device=self.device).repeat_interleave(num_past_agents)

                    # Inference
                    if 'SimpleEncoderDecoder' == self.model_name:
                        input_dict={
                            "past_agents_traj": past_agents_traj,
                            "past_agents_traj_len": past_agents_traj_len,
                            "future_agent_masks": future_agent_masks,
                            "decode_start_vel": decode_start_vel,
                            "decode_start_pos": decode_start_pos
                        }   
                        predicted_trajs = self.model(input_dict)

                    elif 'SocialPooling' == self.model_name:
                        predicted_trajs = self.model(past_agents_traj, past_agents_traj_len, episode_idx, future_agent_masks,
                                                    decode_start_vel, decode_start_pos)

                    elif 'MATF' in self.model_name:
                        stochastic = False
                        input_dict={
                            "past_agents_traj":past_agents_traj, 
                            "past_agents_traj_len":past_agents_traj_len, 
                            "episode_idx":episode_idx, 
                            "future_agent_masks":future_agent_masks,
                            "decode_start_vel":decode_start_vel, 
                            "decode_start_pos":decode_start_pos,
                            "scene_images":scene_images, 
                            "stochastic":stochastic
                        }
                        predicted_trajs = self.model(input_dict)
                    else:
                        raise ValueError("Unknown model type {:s}.".format(self.model_name))
                    
                    ## Deterministic model returns as: Na x Td x C

                    agent_time_index = torch.arange(num_agents, device=self.device).repeat_interleave(future_agents_traj_len)
                    time_normalizer = future_agents_traj_len.float().repeat_interleave(future_agents_traj_len)

                    error = future_agents_traj - predicted_trajs # A x Td x 2
                    batch_loss = (error ** 2).sum(dim=-1) # x**2 + y**2 | A x Td 
                    batch_loss = batch_loss[agent_time_index, future_agents_traj_len_idx] / time_normalizer
                    batch_loss = batch_loss.sum() / (num_agents * 2.0)

                    sq_error2 = (error[two_mask, :int(self.decoding_steps*2/3), :] ** 2).sum(2).sqrt()
                    sq_error3 = (error[three_mask,:, :] ** 2).sum(2).sqrt()

                    num_agents2 = sq_error2.size(0)
                    num_agents3 = sq_error3.size(0)

                    ade2 = sq_error2.mean(dim=-1) # A X T >> A
                    fde2 = sq_error2[... , -1]
                    ade3 = sq_error3.mean(dim=-1)
                    fde3 = sq_error3[... , -1]
                    
                    # 2-Loss
                    batch_minade2 = ade2.mean() # A >> 1
                    batch_minfde2 = fde2.mean()
                    batch_avgade2 = ade2.mean()
                    batch_avgfde2 = fde2.mean()
                    
                    # 3-Loss
                    batch_minade3 = ade3.mean()
                    batch_minfde3 = fde3.mean()
                    batch_avgade3 = ade3.mean()
                    batch_avgfde3 = fde3.mean()

                    print("Working on test batch {:d}/{:d}... ".format(b+1, len(self.data_loader)) +
                    "batch_loss: {:.2f}, ".format(batch_loss.item()) +
                    "minFDE3: {:.2f}, avgFDE3: {:.2f}".format(batch_minfde3.item(), batch_avgfde3.item()), end='\r')

                    epoch_loss += batch_loss.item()

                    epoch_minade2 += batch_minade2.item() * num_agents2
                    epoch_avgade2 += batch_avgade2.item() * num_agents2
                    epoch_minfde2 += batch_minfde2.item() * num_agents2
                    epoch_avgfde2 += batch_avgfde2.item() * num_agents2

                    epoch_minade3 += batch_minade3.item() * num_agents3
                    epoch_avgade3 += batch_avgade3.item() * num_agents3
                    epoch_minfde3 += batch_minfde3.item() * num_agents3
                    epoch_avgfde3 += batch_avgfde3.item() * num_agents3

                    epoch_agents += len(future_agents_traj_len)
                    epoch_agents2 += num_agents2
                    epoch_agents3 += num_agents3

                    map_files = self.map_file(scene_id)
                    output_files = [self.exp_path + '/' + x[2] + '.jpg' for x in scene_id]

                    cum_num_future_agents = [0] + torch.cumsum(num_future_agents, dim=0).tolist()

                    predicted_trajs = predicted_trajs.unsqueeze(1).cpu().numpy()

                    past_agents_traj = past_agents_traj.cpu().numpy()
                    past_agents_traj_len = past_agents_traj_len.cpu().numpy()

                    future_agents_traj = future_agents_traj.cpu().numpy()
                    future_agents_traj_len = future_agents_traj_len.cpu().numpy()

                    for i in range(batch_size):
                        candidate_i   = predicted_trajs[       cum_num_future_agents[i]:cum_num_future_agents[i+1]]
                        future_traj_i = future_agents_traj[    cum_num_future_agents[i]:cum_num_future_agents[i+1]]
                        future_lens_i = future_agents_traj_len[cum_num_future_agents[i]:cum_num_future_agents[i+1]]
                        past_traj_i   = past_agents_traj[      cum_num_future_agents[i]:cum_num_future_agents[i+1]]
                        past_lens_i   = past_agents_traj_len[  cum_num_future_agents[i]:cum_num_future_agents[i+1]]

                        map_file_i = map_files[i]
                        output_file_i = output_files[i]
                        
                        # dao_i, dao_mask_i = self.dao(candidate_i, map_file_i)
                        # dac_i, dac_mask_i = self.dac(candidate_i, map_file_i)

                        # epoch_dao += dao_i.sum()
                        # dao_agents += dao_mask_i.sum()

                        # epoch_dac += dac_i.sum()
                        # dac_agents += dac_mask_i.sum()

                        # if self.render:
                        #     pool.apply(self.write_img_output, 
                        #                (candidate_i, past_traj_i, past_lens_i, future_traj_i, future_lens_i, map_file_i, output_file_i)
                        #     )


            # 2-Loss
            epoch_minade2 /= epoch_agents2
            epoch_avgade2 /= epoch_agents2
            epoch_minfde2 /= epoch_agents2
            epoch_avgfde2 /= epoch_agents2

            # 3-Loss
            epoch_minade3 /= epoch_agents3
            epoch_avgade3 /= epoch_agents3
            epoch_minfde3 /= epoch_agents3
            epoch_avgfde3 /= epoch_agents3

            epoch_dao /= max(1, dao_agents)
            epoch_dac /= max(1, dac_agents)

            epoch_ades = ( epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3 )
            epoch_fdes = ( epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3 )

            # return epoch_g_loss+epoch_d_loss, epoch_g_loss, epoch_d_loss, epoch_ades, epoch_fdes, scheduler_metric


        print("--Final Performane Report--")
        print("minADE2: {:.5f}, minFDE2: {:.5f}, avgADE2: {:.5f}, avgFDE2: {:.5f}".format(epoch_minade2, epoch_minfde2, epoch_avgade2, epoch_avgfde2))
        print("minADE3: {:.5f}, minFDE3: {:.5f}, avgADE3: {:.5f}, avgFDE3: {:.5f}".format(epoch_minade3, epoch_minfde3, epoch_avgade3, epoch_avgfde3))
        
        if self.generative:
            print("DAO: {:.5f}e-5, DAC: {:.5f}".format(epoch_dao * 10000.0, epoch_dac))
            with open(self.exp_path + f'/test_metric_{self.datetime}.pkl', 'wb') as f:
                pkl.dump({"ADEs": epoch_ades,
                         "FDEs": epoch_fdes,
                          "DAO": epoch_dao,
                          "DAC": epoch_dac}, f)