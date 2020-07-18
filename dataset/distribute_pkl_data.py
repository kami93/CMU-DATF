import pathlib
import numpy as np
from tqdm import tqdm
import pdb
import pickle as pkl
import pdb
import glob
import os
import h5py
import shutil
import multiprocessing as mp
import sys 
from datetime import datetime

CROSS_CHECK_MAP_WITH_TRAJ = 1
INCLUDE_PCL_VOXEL = 0
WRITE_DEBUG = 1
WORK_MAP_OR_TRAJ = 0 # map==0, traj==1 
REMOVE_FILES_BADMAP_NOMAP = 0

train_scene_path = os.path.join('/mnt/sdc1/manoj/carla_new/pcd_data/train/')  ##TODO: Relative
train_traj_path = os.path.join('/mnt/sdc1/manoj/carla_new/train/')  ##TODO: Relative
map_files = sorted(glob.glob(os.path.join(train_scene_path,'*.h5')))
traj_files =  sorted(glob.glob(os.path.join(train_traj_path,'*.h5')))
l_files_count = len(map_files)
t_files_count = len(traj_files)
print("Found {:d} map files, {:d} traj files".format(l_files_count, t_files_count))


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


def distribute_map_files(each_h5, k):
    h5_name = each_h5[len(os.path.dirname(each_h5))+1:]
    try:
        all_data = h5py.File(each_h5)
    except:
        if WRITE_DEBUG:
            date_str = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
            with open("skipped.txt", "a") as file:
                file.write(date_str + "  " + each_h5+ " broken \n")
        print("H5 broken exitting")
        sys.stdout.flush()
        return 0,1

    filenames = all_data['filenames']
    seg_images = all_data['seg_images']

    if INCLUDE_PCL_VOXEL:
        pcds = all_data['pcd']
        pcd_nums = all_data['pcd_nums']
        max_points = max(pcd_nums)
        points = np.zeros((len(pcd_nums), max_points, 3))
        for j in range(len(pcd_nums)):
            pcd = pcds[str(j)]
            pdb.set_trace()
            points[j, :pcd_nums[i], :] = pcd
    
    map_file = os.path.join(train_scene_path, h5_name)
    if not os.path.exists(map_file):
        if WRITE_DEBUG:
            with open("skipped.txt", "a") as file:
                file.write(date_str + "  " + traj_file+ "\n")
        print("H5 doesnt exists ",traj_file )
        sys.stdout.flush()
        return 0,1

    pbar = tqdm(filenames)
    for i, filename in enumerate(pbar):
        pbar.set_description(str(k)+" "+filename)

        save_file_seg = pathlib.Path("/mnt/sdc1/shpark/carla_shpark_pkl/train/town01/").joinpath('map', 'v2.0', filename+'.pkl')  ##TODO: Relative
        save_file_seg.parent.mkdir(parents=True, exist_ok=True)
        if not save_file_seg.exists():
            h5_path = pathlib.Path("/mnt/sdc1/shpark/carla_shpark_pkl/train/town01/").joinpath('map', 'v2.0', filename+'.pkl')  ##TODO: Relative
            seg_image = seg_images[i]
            with save_file_seg.open('wb') as f:
                pkl.dump(seg_image, f)
        
        if INCLUDE_PCL_VOXEL:
            save_file_pcd = pathlib.Path("/mnt/sdc1/shpark/carla_shpark_pkl/train/town01/").joinpath('voxel', filename+'.pkl')  ##TODO: Relative
            save_file_pcd.parent.mkdir(parents=True, exist_ok=True)
            if not save_file_pcd.exists():
                pcd = pcds[str(i)][...]
                with save_file_pcd.open('wb') as f:
                    pkl.dump(pcd, f)
    return 1,1


def distribute_traj_files(each_h5, k):
    
    h5_name = each_h5[len(os.path.dirname(each_h5))+1:]
    pbar.set_description("Process %s" % (h5_name))
    
    try:
        all_data = h5py.File(each_h5)
    except:
        with open("skipped.txt", "a") as file:
            file.write(each_h5+ " broken \n")
        print("H5 broken exitting")
        sys.stdout.flush()
        return 0,1

    agent_pasts = all_data['agent_pasts']
    agent_futures = all_data['agent_futures']
    filenames = all_data['filenames']
    overhead_image = all_data['overhead_image']
    
    traj_file = os.path.join(train_traj_path, h5_name)
    if not os.path.exists(traj_file):
        date_str = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        with open("skipped.txt", "a") as file:
            file.write(traj_file+ "\n")
        print("H5 doesnt exists ",traj_file )
        sys.stdout.flush()
        return 0,1

    for i, filename in enumerate(filenames):
        pbar.set_description(str(k)+" "+filename)

        save_file_seg = pathlib.Path("/mnt/sdc1/shpark/carla_shpark_pkl/train/town01/").joinpath('map', 'v2.0', filename+'.pkl')  ##TODO: Relative
        save_file_seg.parent.mkdir(parents=True, exist_ok=True)
        save_file_traj = pathlib.Path("/mnt/sdc1/shpark/carla_shpark_pkl/train/town01/").joinpath('traj', filename+'.pkl') ##TODO: Relative
        save_file_traj.parent.mkdir(parents=True, exist_ok=True)
            
        if CROSS_CHECK_MAP_WITH_TRAJ:
            # Check if map data is present
            if(os.path.exists(str(save_file_seg))):
                # read file
                with open(str(save_file_seg), 'rb') as f:
                    data = pkl.load(f)
                if len(list(data.keys())) > 0:
                    if WRITE_DEBUG:
                        with open("GoodMaps.txt", "a") as file:
                            file.write(str(save_file_seg)+ "\n")
                    pass
                else:
                    if WRITE_DEBUG:
                        with open("BadMaps.txt", "a") as file:
                            file.write(str(save_file_seg)+ "\n")
                    print("BadMap ",save_file_seg )
                    # Check if traj present on BADMAP and NOMAP
                    if os.path.exists(str(save_file_traj)):
                        if REMOVE_FILES_BADMAP_NOMAP:
                            os.remove(str(save_file_traj))
                            with open("BadMaps.txt", "a") as file:
                                file.write(str(save_file_traj)+ " removed \n")
                    else:
                        if WRITE_DEBUG:
                            with open("BadMaps.txt", "a") as file:
                                file.write(str(save_file_seg)+ " already removed \n")
                        print("Traj file corresponding to badmap/nomap already removed ",save_file_seg )
                    sys.stdout.flush()
                    return 0,1
            else:
                if WRITE_DEBUG:
                    with open("NoPathsMap.txt", "a") as file:
                        file.write(str(save_file_seg)+ "\n")
                print("No such path present ",save_file_seg )

                # Check if traj present on BADMAP and NOMAP
                if os.path.exists(str(save_file_traj)):
                    if REMOVE_FILES_BADMAP_NOMAP:
                        os.remove(str(save_file_traj))
                    pass
                else:
                    if WRITE_DEBUG:
                        with open("BadMaps.txt", "a") as file:
                            file.write(str(save_file_seg)+ " already removed \n")
                    print("Traj file corresponding to badmap/nomap already removed ",save_file_seg )
                sys.stdout.flush()
                return 0,1

        if not save_file_traj.exists():
            h5_path = save_file_traj
            # seg_image = overhead_image[i]
            traj_data = {}
            traj_data["agent_pasts"] = agent_pasts[i]
            traj_data["agent_futures"] = agent_futures[i][:, :30, :]
            with save_file_traj.open('wb') as f:
                pkl.dump(traj_data, f)
    return 1,1

if __name__ == "__main__":
    runner = ParallelSim(processes=20) # self.num_workers
    results = []
    
    if WORK_MAP_OR_TRAJ:
        files = traj_files.copy()
        for i,path_ in enumerate(tqdm(traj_files)):
            runner.add(distribute_traj_files, (path_, i))
            # results.append(distribute_traj_files(path_)) # debugging purpose
        runner.run()
        results = runner.get_results()

    else:
        files = map_files.copy()
        for i,path_ in enumerate(tqdm(map_files)):
            runner.add(distribute_map_files, (path_, i))
            # results.append(distribute_map_files(path_)) # debugging purpose
        runner.run()
        results = runner.get_results()
    
    if WRITE_DEBUG:
        date_str = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        with open("distribute_results.txt", "a") as file:
            file.write(date_str + "\n")
            for i,each in enumerate(files):
                file.write(each +" "+str(results[i])+ "\n")
            




