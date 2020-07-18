import os
import pickle
import multiprocessing as mp

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

_data_dir = './data/argoverse'

class ParallelSim(object):
    def __init__(self, processes):
        self.pool = mp.Pool(processes=processes)
        self.total_processes = 0
        self.completed_processes = 0
        self.results = []

    def add(self, func, args):
        self.pool.apply_async(func=func, args=args, callback=self.complete)
        self.total_processes += 1

    def complete(self, result_tuple):
        result, flag = result_tuple
        if flag:
            self.results.append(result)
        self.completed_processes += 1
        print('-- processed {:d}/{:d}'.format(self.completed_processes,
                                              self.total_processes), end='\r')

    def run(self):
        self.pool.close()
        self.pool.join()

    def get_results(self):
        return self.results

def argoverse_collate(batch, test_set=False):
    # batch_i:
    # 1. past_agents_traj : (Num obv agents in batch_i X 20 X 2)
    # 2. past_agents_traj_len : (Num obv agents in batch_i, )
    # 3. future_agents_traj : (Num pred agents in batch_i X 20 X 2)
    # 4. future_agents_traj_len : (Num pred agents in batch_i, )
    # 5. future_agent_masks : (Num obv agents in batch_i)
    # 6. decode_rel_pos: (Num pred agents in batch_i X 2)
    # 7. decode_start_pos: (Num pred agents in batch_i X 2)
    # 8. map_image : (3 X 224 X 224)
    # 9. scene ID: (string)
    # Typically, Num obv agents in batch_i < Num pred agents in batch_i ##

    batch_size = len(batch)

    if test_set:
        past_agents_traj, past_agents_traj_len, future_agent_masks, decode_start_vel, decode_start_pos, map_image, prior, scene_id = list(zip(*batch))

    else:
        past_agents_traj, past_agents_traj_len, future_agents_traj, future_agents_traj_len, future_agent_masks, decode_start_vel, decode_start_pos, map_image, prior, scene_id = list(zip(*batch))
        
        # Future agent trajectory
        num_future_agents = np.array([len(x) for x in future_agents_traj])
        future_agents_traj = np.concatenate(future_agents_traj, axis=0)
        future_agents_traj_len = np.concatenate(future_agents_traj_len, axis=0)
        
        future_agents_three_idx = future_agents_traj.shape[1]
        future_agents_two_idx = int(future_agents_three_idx * 2 // 3)

        future_agents_three_mask = future_agents_traj_len >= future_agents_three_idx
        future_agents_two_mask = future_agents_traj_len >= future_agents_two_idx
        
        future_agents_traj_len_idx = []
        for traj_len in future_agents_traj_len:
            future_agents_traj_len_idx.extend(list(range(traj_len)))

        # Convert to Tensor
        num_future_agents = torch.LongTensor(num_future_agents)
        future_agents_traj = torch.FloatTensor(future_agents_traj)
        future_agents_traj_len = torch.LongTensor(future_agents_traj_len)

        future_agents_three_mask = torch.BoolTensor(future_agents_three_mask)
        future_agents_two_mask = torch.BoolTensor(future_agents_two_mask)

        future_agents_traj_len_idx = torch.LongTensor(future_agents_traj_len_idx)


    # Past agent trajectory
    num_past_agents = np.array([len(x) for x in past_agents_traj])
    past_agents_traj = np.concatenate(past_agents_traj, axis=0)
    past_agents_traj_len = np.concatenate(past_agents_traj_len, axis=0)
    past_agents_traj_len_idx = []
    for traj_len in past_agents_traj_len:
        past_agents_traj_len_idx.extend(list(range(traj_len)))

    # Convert to Tensor
    num_past_agents = torch.LongTensor(num_past_agents)
    past_agents_traj = torch.FloatTensor(past_agents_traj)
    past_agents_traj_len = torch.LongTensor(past_agents_traj_len)
    past_agents_traj_len_idx = torch.LongTensor(past_agents_traj_len_idx)


    # Future agent mask
    future_agent_masks = np.concatenate(future_agent_masks, axis=0)
    future_agent_masks = torch.BoolTensor(future_agent_masks)

    # decode start vel & pos
    decode_start_vel = np.concatenate(decode_start_vel, axis=0)
    decode_start_pos = np.concatenate(decode_start_pos, axis=0)
    decode_start_vel = torch.FloatTensor(decode_start_vel)
    decode_start_pos = torch.FloatTensor(decode_start_pos)

    map_image = torch.stack(map_image, dim=0)
    prior = torch.stack(prior, dim=0)

    scene_id = np.array(scene_id)

    data = (
        map_image, prior, 
        future_agent_masks, 
        num_past_agents, past_agents_traj, past_agents_traj_len, past_agents_traj_len_idx, 
        num_future_agents, future_agents_traj, future_agents_traj_len, future_agents_traj_len_idx, 
        future_agents_two_mask, future_agents_three_mask,
        decode_start_vel, decode_start_pos, 
        scene_id
    )

    return data

class ArgoverseDataset(Dataset):
    def __init__(self, data_partition, map_version, sampling_rate, sample_stride=3,
                use_scene=True, scene_size=(64, 64), ploss_type=None, intrinsic_rate=10,
                max_distance=56, num_workers=None, cache_file=None, multi_agent=True):
        """
        data_dir: Dataset root directory
        data_parititon: Dataset Parition (train | val | test_obs)
        map_version: Map data version (1.3 | 2.0)
        sampling_rate: Physical sampling rate of processed trajectory (Hz)
        intrinsic_rate: Physical sampling rate of raw trajectory (Hz, eg., Argo:10, Nuscene:2)
        sample_stride: The interval between the reference frames in a single episode
        min_past_obv_len: Minimum length of the agent's past trajectory to encode
        min_future_obv_len: Minimum length of the agent's past trajectory to decode
        min_future_pred_len: Minimum length of the agent's future trajectory to decode
        max_distance: Maximum physical distance from the ROI center to an agent's current position
        multi_agent: Boolean flag for including multiple agent setting
        """
        super(ArgoverseDataset, self).__init__()
        self.data_dir = _data_dir
        self.data_partition = data_partition

        if num_workers:
            self.num_workers = num_workers
        else:
            self.num_workers = mp.cpu_count()

        # Sampling Interval = "intrinsic sampling rate" / sampling rate
        self.intrinsic_rate = intrinsic_rate
        if intrinsic_rate % sampling_rate:
            raise ValueError("Intrinsic sampling rate must be evenly divisble by sampling rate.\n Intrinsic SR: {:d}, Given SR: {:d}".format(10, sampling_rate))
        self.sampling_interval = int(self.intrinsic_rate // sampling_rate)

        ## TODO: Caluclate MAX_OBSV_LEN and MAX_PRED_LEN dynamically
        self.max_obsv_len = int(self.intrinsic_rate * 2 // self.sampling_interval)
        self.max_pred_len = int(self.intrinsic_rate * 3 // self.sampling_interval)
        
        self.sample_stride = sample_stride
        self.min_past_obv_len = self.sampling_interval + 1	
        self.min_future_obv_len = int(1 * self.intrinsic_rate)	
        self.min_future_pred_len = int(1.5 * self.intrinsic_rate)	
        self.max_distance = max_distance

        self.use_scene = use_scene
        self.scene_size = scene_size

        self.multi_agent = multi_agent

        if map_version=='1.3' or map_version=='2.0':
            self.map_version = map_version
        else:
            raise("Invalid map: v1.3 | v2.0 are valid")

        if map_version == '1.3':
            self.img_transform = transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225]),
                                                     transforms.Resize(self.scene_size)])
        elif map_version == '2.0':
            self.img_transform = transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize([23.0582], [27.3226])])

        self.ploss_type = ploss_type
        if ploss_type == 'map':
            self.p_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize([23.0582], [27.3226]),
                                                   transforms.Lambda(lambda x: F.log_softmax(x.reshape(-1), dim=0).reshape(x.shape[1:]))]
                                                 )
        # Extract Data:
        if cache_file is None:
            cache_dir = "./argoverse_{}_cache.pkl".format(self.data_partition)

            if os.path.isfile(cache_dir):
                self.load_cache(cache_dir)
            else:
                self.get_data(save_cache_dir=cache_dir)

        else:
            if os.path.isfile(cache_file):
                self.load_cache(cache_file)
            else:
                self.get_data(save_cache_dir=cache_file)

    def __getitem__(self, idx):
        # Create one past list and future list with all the
        past_agents_traj = self.past_agents_traj_list[idx]
        past_agents_traj_len = self.past_agents_traj_len_list[idx]
        future_agents_traj = self.future_agents_traj_list[idx]
        future_agents_traj_len = self.future_agents_traj_len_list[idx]
        future_agent_masks = self.future_agent_masks_list[idx]
        decode_start_vel = self.decode_start_vel[idx]
        decode_start_pos = self.decode_start_pos[idx]
        scene_id = self.scene_id[idx]

        if self.use_scene:
            if self.map_version == '1.3':
                map_file = scene_id[3] + '.png'
            elif self.map_version == '2.0':
                map_file = scene_id[3] + '.pkl'
            img_path = os.path.join(self.data_dir, scene_id[0], scene_id[1], scene_id[2], 'map', 'v{:s}'.format(self.map_version), map_file)

            if self.map_version == '1.3':
                raw_map_image = Image.open(img_path)
            elif self.map_version == '2.0':
                with open(img_path, 'rb') as f:
                    raw_image = pickle.load(f)
                raw_map_image = raw_image.astype(np.float32)
                raw_map_image = cv2.resize(raw_image.astype(np.float32), dsize=self.scene_size, interpolation=cv2.INTER_LINEAR)

            map_image = self.img_transform(raw_map_image)
        
        else:
            map_image = torch.FloatTensor([0.0])

        if self.ploss_type == "map":
            if self.map_version != '2.0' or not self.use_scene:
                map_file = scene_id[3] + '.pkl'
                img_path = os.path.join(self.data_dir, scene_id[0], scene_id[1], scene_id[2], 'map', 'v2.0', map_file)
                with open(img_path, 'rb') as f:
                    raw_image = pickle.load(f)

            raw_map_image = cv2.resize(raw_image.astype(np.float32), dsize=(100, 100), interpolation=cv2.INTER_LINEAR)
            raw_map_image[raw_map_image < 0] = 0 # Uniform on drivable area
            raw_map_image = raw_map_image.max() - raw_map_image # Invert values so that non-drivable area has smaller values
            prior = self.p_transform(raw_map_image)
                
        elif self.ploss_type == "logistic":
            prior_path = os.path.join(self.data_dir, 'logistic_prior', scene_id[0], scene_id[2], scene_id[3] + '.pkl')
            with open(prior_path, 'rb') as f:
                prior = pickle.load(f)
            prior = torch.FloatTensor(prior)

        else:
            prior = torch.FloatTensor([0.0])
        
        if 'test' in self.data_partition:
          episode = (past_agents_traj, past_agents_traj_len, future_agent_masks, decode_start_vel, decode_start_pos, map_image, prior, scene_id)
        else:
          episode = (past_agents_traj, past_agents_traj_len, future_agents_traj, future_agents_traj_len, future_agent_masks, decode_start_vel, decode_start_pos, map_image, prior, scene_id)

        return episode

    def __len__(self):
        return len(self.scene_id)

    def load_cache(self, cache_dir):
        with open(cache_dir, 'rb') as f:
            results = pickle.load(f)

        self.past_agents_traj_list, self.past_agents_traj_len_list,\
        self.future_agents_traj_list, self.future_agents_traj_len_list,\
        self.future_agent_masks_list,\
        self.decode_start_pos, self.decode_start_vel, self.scene_id = list(zip(*results))

    def get_data(self, save_cache_dir=None):
        partition_dir = os.path.join(self.data_dir, self.data_partition)
        print(f'Extracting data from: {partition_dir}')

        sub_partitions = os.listdir(partition_dir)
        sub_partitions.sort()
        path_lists = []

        for sub_partition in sub_partitions: 
            print(f'Sub-partition: {sub_partition}')
            path_lists.extend(self.extract_directory(sub_partition))

        runner = ParallelSim(processes=self.num_workers)
        option_list = [self.min_past_obv_len, self.min_future_obv_len, self.min_future_pred_len, self.max_distance, self.sampling_interval, self.max_obsv_len, self.max_pred_len, self.multi_agent]

        for path_list in path_lists:
            runner.add(self.extract_submodule_multicore, (path_list, option_list))

        runner.run()
        results = runner.get_results()

        if save_cache_dir is not None:
            with open(save_cache_dir, 'wb') as f:
                pickle.dump(results, f) 

        self.past_agents_traj_list, self.past_agents_traj_len_list,\
        self.future_agents_traj_list, self.future_agents_traj_len_list,\
        self.future_agent_masks_list,\
        self.decode_start_pos, self.decode_start_vel, self.scene_id = list(zip(*results))

        print('\nExtraction Compltete!\n')

    def extract_directory(self, sub_partition):        
        work_dir = os.path.join(self.data_dir, self.data_partition, sub_partition)
        episodes = os.listdir(work_dir)

        episodes.sort(key=lambda x: int(x[-8:], 16))
        path_lists = []

        num_episodes = len(episodes)
        for i, episode in enumerate(episodes): 
            observation_dir = os.path.join(work_dir, episode, 'observation')

            reference_frames = os.listdir(observation_dir)
            reference_frames.sort()

            for j in range(len(reference_frames)):
                path_lists.append((self.data_dir, self.data_partition, sub_partition, episode, reference_frames[j].replace('.pkl', '')))
            print('Counting episodes {:d}/{:d}'.format(i, num_episodes), end='\r')

        return path_lists

    @staticmethod
    def extract_submodule_multicore(path_list, options_list):
        data_dir, data_partition, sub_partition, episode, reference_frame = path_list
        min_past_obv_len, min_future_obv_len, min_future_pred_len, max_distance, sampling_interval, max_obsv_len, max_pred_len, is_MA = options_list

        def get_agent_ids(dataframe):
            """
            Returns:
                    List of past agent ids: List of agent ids that are to be considered for the encoding phase.
                    Future agent ids mask: A mask which dentoes if an agent in past agent ids list is to be considered
                                        during decoding phase.
            """
            # Select past agent ids for the encoding phase.
            # obsolute classes (previously used in Argo Tracking): (dataframe['class'] == 'VEHICLE') | (dataframe['class'] == 'LARGE_VEHICLE') | (dataframe['class'] == 'PEDESTRIAN') | (dataframe['class'] == 'ON_ROAD_OBSTACLE') | (dataframe['class'] == 'BICYCLE')
            if is_MA:
                past_df = dataframe.loc[((dataframe['class'] == 'AGENT') | (dataframe['class'] == 'AV') | (dataframe['class'] == 'OTHERS'))
                                & (dataframe['observation_length']>=min_past_obv_len)]

            else:
                past_df = dataframe.loc[(dataframe['class'] == 'AGENT')]

            past_agent_ids = past_df['track_id'].unique()

            # Check if the encoding trajectories have their current position in the region of interest.
            updated_past_agent_ids = []
            for agent_id in past_agent_ids:
                last_pos = past_df[past_df['track_id'] == agent_id].iloc[-1][['X','Y']].to_numpy()
                if np.any(np.abs(last_pos) > max_distance):
                    pass
                else:
                    updated_past_agent_ids.append(agent_id)

            updated_past_agent_ids = np.array(updated_past_agent_ids)

            # Select future agent ids for the decoding phase.
            if is_MA:
                future_df = dataframe.loc[(dataframe['class'] == 'AGENT')]
                # future_df = dataframe.loc[((dataframe['class'] == 'AGENT') | (dataframe['class'] == 'AV') | (dataframe['class'] == 'OTHERS'))
                #                 & (dataframe['observation_length']>=min_past_obv_len)]

            else:
                future_df = dataframe.loc[(dataframe['class'] == 'AGENT')]

            future_agent_ids = future_df['track_id'].unique()

            # Create a mask corresponding to the past_agent_ids list where the value '1' in mask denotes
            # that agent is to be considered while decoding and 0 denotes otherwise.
            future_agent_ids_mask = np.isin(updated_past_agent_ids, future_agent_ids)

            return updated_past_agent_ids, future_agent_ids_mask

        def extract_trajectory_info(obv_df, pred_df, past_agent_ids, future_agent_ids_mask):
            """
            Extracts the past and future trajectories of the agents as well as the encode and decode
            coordinates.
            """
            past_traj_list = []
            past_traj_len_list = []

            future_traj_list = []
            future_traj_len_list = []

            decode_start_pos_list = []
            decode_start_vel_list = []
            for agent_id in past_agent_ids:
                mask = obv_df['track_id'] == agent_id
                
                ### TODO: Modify the extraction routine so that it supports different sampling intervals.
                past_agent_traj = obv_df[mask][['X', 'Y']].to_numpy().astype(np.float32) # raw past traj
                past_agent_traj = past_agent_traj[-1::-sampling_interval, :][::-1, :] # sampled past traj according to sampling interval
                decode_start_pos = past_agent_traj[-1]
                decode_start_vel = (past_agent_traj[-1] - past_agent_traj[-2]) # decode velocity

                obsv_len = past_agent_traj.shape[0]
                obsv_pad = max_obsv_len - obsv_len

                if obsv_pad: # equiv. to if obsv_pad != 0
                    past_agent_traj = np.pad(past_agent_traj, ((0, obsv_pad), (0, 0)), mode='constant')
                
                past_traj_list.append(past_agent_traj)
                past_traj_len_list.append(obsv_len)
                decode_start_pos_list.append(decode_start_pos)
                decode_start_vel_list.append(decode_start_vel)

            for agent_id in past_agent_ids[future_agent_ids_mask]:
                mask = pred_df['track_id'] == agent_id
                future_agent_traj = pred_df[mask][['X', 'Y']].to_numpy().astype(np.float32) # raw future traj
                future_agent_traj = future_agent_traj[sampling_interval-1::sampling_interval] # sampled future traj according to sampling interval

                pred_len = future_agent_traj.shape[0]
                pred_pad = max_pred_len - pred_len

                if pred_pad:
                    future_agent_traj = np.pad(future_agent_traj, ((0, pred_pad), (0, 0)), mode='constant')
                
                future_traj_list.append(future_agent_traj)
                future_traj_len_list.append(pred_len)

            past_traj_list = np.array(past_traj_list)
            past_traj_len_list = np.array(past_traj_len_list)

            future_traj_list = np.array(future_traj_list)
            future_traj_len_list = np.array(future_traj_len_list)

            decode_start_pos_list = np.array(decode_start_pos_list)
            decode_start_vel_list = np.array(decode_start_vel_list)

            return past_traj_list, past_traj_len_list, future_traj_list, future_traj_len_list, decode_start_pos_list, decode_start_vel_list

        observation_file = os.path.join(data_dir, data_partition, sub_partition, episode, 'observation', reference_frame + '.pkl')

        with open(observation_file, 'rb') as f:
            observation_df = pickle.load(f)

        prediction_df = None
        if 'test' not in data_partition:
          prediction_file = os.path.join(data_dir, data_partition, sub_partition, episode, 'prediction', reference_frame + '.pkl')
          with open(prediction_file, 'rb') as f:
              prediction_df = pickle.load(f)

        past_agent_ids, future_agent_ids_mask = get_agent_ids(observation_df)

        past_traj = None
        past_traj_len = None
        future_traj = None
        future_traj_len = None
        decode_start_pos = None
        decode_start_vel = None
        condition = bool(future_agent_ids_mask.sum())

        if condition:
            past_traj, past_traj_len, future_traj, future_traj_len, decode_start_pos, decode_start_vel = extract_trajectory_info(observation_df, prediction_df, past_agent_ids, future_agent_ids_mask)
        else:
            pass

        scene_id = [data_partition, sub_partition, episode, reference_frame]
        return (past_traj, past_traj_len, future_traj, future_traj_len, future_agent_ids_mask, decode_start_pos, decode_start_vel, scene_id), condition