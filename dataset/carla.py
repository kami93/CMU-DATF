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
import glob 
import pdb 
from tqdm import tqdm
import sys
_data_dir = './data/carla'
PLOT = 0

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

def carla_collate(batch, test_set=False):
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
        future_agents_three_mask = future_agents_traj_len >=0
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

class CarlaDataset(Dataset):
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
        super(CarlaDataset, self).__init__()
        self.data_dir = _data_dir
        self.data_partition = data_partition

        if num_workers:
            self.num_workers = num_workers
        else:
            self.num_workers = 1

        # Sampling Interval = "intrinsic sampling rate" / sampling rate
        self.intrinsic_rate = intrinsic_rate
        if intrinsic_rate % sampling_rate:
            raise ValueError("Intrinsic sampling rate must be evenly divisble by sampling rate.\n Intrinsic SR: {:d}, Given SR: {:d}".format(10, sampling_rate))
        self.sampling_interval = int(self.intrinsic_rate // sampling_rate)

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
            cache_dir = "./carla_n_{}_cache.pkl".format(self.data_partition)
            
            if os.path.isfile(cache_dir):
                self.load_cache(cache_dir)
                
            else:
                self.extract_carla_data(save_cache_dir=cache_dir)

        else:
            if os.path.isfile(cache_file):
                self.load_cache(cache_file)
            else:
                self.extract_carla_data(save_cache_dir=cache_file)



    def __len__(self):
        return len(self.scene_id)

    def load_cache(self, cache_dir):
        with open(cache_dir, 'rb') as f:
            results = pickle.load(f)
        self.past_agents_traj_list, self.past_agents_traj_len_list,\
        self.future_agents_traj_list, self.future_agents_traj_len_list,\
        self.future_agent_masks_list,\
        self.decode_start_pos, self.decode_start_vel, self.scene_id = list(zip(*results))

    def __getitem__(self, idx):
        # Create one past list and future list with all the
        
        past_agents_traj = self.past_agents_traj_list[idx]
        past_agents_traj_len = self.past_agents_traj_len_list[idx]
        future_agents_traj = self.future_agents_traj_list[idx]
        future_agents_traj_len = self.future_agents_traj_len_list[idx]
        future_agent_masks = self.future_agent_masks_list[idx]
        # encode_coordinates = self.encode_coordinates[idx]
        decode_start_vel = self.decode_start_vel[idx]
        decode_start_pos = self.decode_start_pos[idx]

        scene_id = self.scene_id[idx]

        if self.use_scene:
            if self.map_version == '1.3':
                map_file = scene_id[3] + '.png'
                img_path = os.path.join(self.data_dir, scene_id[0], scene_id[1], 'map', 'v{:s}'.format(self.map_version), map_file)

                raw_map_image = Image.open(img_path)

            elif self.map_version == '2.0':
                map_file = scene_id[3] + '.pkl'
                img_path = os.path.join(self.data_dir, scene_id[0], scene_id[1], 'map', 'v{:s}'.format(self.map_version), map_file)

                try:
                    with open(str(img_path), 'rb') as f:
                        raw_image = pickle.load(f)
                    
                except Exception as e:
                    print(e)
                    with open("skipped_pkl_train.txt", "a") as f:
                        f.write(str(scene_id)+ " broken \n")
                    print("No image for ", idx, " scene id ", scene_id, " loc: ", img_path)
                    sys.stdout.flush()
                    return

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

    def extract_directory(self, sub_partition):        
        if 'town' in sub_partition:
            path_lists = glob.glob(os.path.join(self.data_dir, self.data_partition, sub_partition, 'traj', '*'))
        else:
            print("No towns found")

        return path_lists

    @staticmethod
    def extract_carla_from_path( path, sampling_interval, scene_size):

        data2 = None
        # Read arrays from pkl files.
        try:
            with open(str(path), 'rb') as f:
                data2 = pickle.load(f)
        except Exception as e:
            print(e)
            print("Error loading, stopping everything ...")
        
        ## Coordinate Sequences ##
        # surround agents
        future_agents_traj = data2["agent_futures"]
        past_agents_traj = data2["agent_pasts"]
        
        # Data clean fix (ignore this)
        for i in range(len(past_agents_traj)):
            center_point = ((past_agents_traj[i][-2]+future_agents_traj[i][0])/2)
            past_agents_traj[i][-1] = center_point

        # Compensate little deviations (~1e-13) in the ego's current locations.
        # So the locations are always (0, 0) exactly.
        ego_current = past_agents_traj[0, -1, :].copy()
        past_agents_traj[:, :, :] -= ego_current
        future_agents_traj[:, :, :] -= ego_current
        scene_size = [-56, 56]
        # Scaling trajectory from (-200,200) to (scene_size[0], scene_size[1])
        future_agents_traj = future_agents_traj*(scene_size[1] - scene_size[0])/400
        past_agents_traj = past_agents_traj*(scene_size[1] - scene_size[0])/400

        filename = path[len(os.path.dirname(path))+1:-4]
        scene_id = [ 'train', 'town01','traj', filename]

        # Filter the agents whose locations at t=0, are out of ROI
        # ROI === [(-56, 56), (-56, 56)]
        agents_current = past_agents_traj[ :, -1, :]
        oom_mask = np.all(np.abs(agents_current) < np.abs(scene_size[0]), axis=-1)
        num_agents = oom_mask.sum(axis=-1)
        past_agents_traj_filt = past_agents_traj[oom_mask]
        future_agents_traj_filt = future_agents_traj[oom_mask]
        
        # Apply sampling intervals
        past_agents_traj_filt = past_agents_traj_filt[:, -1::-sampling_interval][:, ::-1]
        future_agents_traj_filt = future_agents_traj_filt[:, sampling_interval-1::sampling_interval]

        # Make encode_coordinates, decode_rel_pos, decode_start_pos
        agents_current = past_agents_traj_filt[:, -1, :]
        decode_start_pos = past_agents_traj_filt[:, -1, :]
        decode_start_vel = past_agents_traj_filt[:, -1, :] - past_agents_traj_filt[:, -2, :]

        # Set dtypes float64 to float32
        past_agents_traj_filt = past_agents_traj_filt.astype(np.float32)
        future_agents_traj_filt = future_agents_traj_filt.astype(np.float32)
        decode_start_pos = decode_start_pos.astype(np.float32)
        decode_start_vel = decode_start_vel.astype(np.float32)

        past_agents_traj_len = np.full((num_agents, ), int(20//sampling_interval), np.int64)
        
        future_agents_traj_len = np.full((num_agents, ), int(30//sampling_interval), np.int64)
        future_agent_masks = np.full((num_agents, ), True, np.bool)
        condition =1 
        
        if PLOT: 
            from matplotlib import pyplot as plt
            import matplotlib
            pdb.set_trace()
            matplotlib.use("Agg") 
            plt.figure(figsize=(10,10))
            plt.scatter(*future_agents_traj_filt.T, color="green", alpha=0.4)
            plt.scatter(*past_agents_traj_filt.T, marker="+", color="red", alpha=0.4)
            plt.scatter(*past_agents_traj_filt[:, -1, :].T, marker="+", color="black", alpha=0.4)
            cwd = os.getcwd()
            plt.savefig(os.path.join(cwd, "plots", filename))

        
        return (past_agents_traj_filt, past_agents_traj_len, future_agents_traj_filt, future_agents_traj_len, future_agent_masks, decode_start_pos, decode_start_vel, scene_id), condition 

    def extract_carla_data(self, save_cache_dir=None):
        
        if not self.scene_size:
            self.scene_size = (60,60)

        partition_dir = os.path.join(self.data_dir, self.data_partition)
        print(f'Extracting data from: {partition_dir}')

        sub_partitions = os.listdir(partition_dir)
        sub_partitions.sort()
        path_lists = []

        for sub_partition in sub_partitions: 
            
            traj_files =  glob.glob(os.path.join( partition_dir, sub_partition, 'traj', '*'))
            map_files =  glob.glob(os.path.join( partition_dir, sub_partition, 'map', 'v2.0', '*'))
            traj_names = [each[len(os.path.dirname(each))+1:] for each in traj_files]
            map_names = [each[len(os.path.dirname(each))+1:] for each in map_files]
            common_elements = list(set(traj_names).intersection(set(map_names)))
            try:
                root = os.path.dirname(traj_files[0])
            except Exception as e:
                print(e)
                pdb.set_trace()
            
            common_traj_file = [os.path.join(root, each) for each in common_elements]
            path_lists.extend( common_traj_file )

        runner = ParallelSim(self.num_workers)
        results = []
        sampling_interval = self.sampling_interval

        for i,path_ in enumerate(tqdm(path_lists)):
            runner.add(self.extract_carla_from_path, (path_,sampling_interval, self.scene_size))
        
        runner.run()
        results = runner.get_results()

        if save_cache_dir is not None:
            with open(save_cache_dir, 'wb') as f:
                pickle.dump(results, f) 
        self.past_agents_traj_list, self.past_agents_traj_len_list,\
        self.future_agents_traj_list, self.future_agents_traj_len_list,\
        self.future_agent_masks_list,\
        self.decode_start_pos, self.decode_start_vel, self.scene_id = list(zip(*results))
        print('Extraction Compltete!\n')


