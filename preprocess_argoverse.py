from pathlib import Path as P
import argparse
import random
from typing import Dict, List, Tuple, Optional, Union
from multiprocessing import Pool, cpu_count
import warnings

from scipy.ndimage.morphology import distance_transform_edt
from compress_pickle import dump, load
import numpy as np
import cv2

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader

warnings.filterwarnings("ignore") # ignore pandas copy warning.
REFERENCE_FRAME = 19
SEED = 88245

INPUT = './data/Argoverse/'
OUTPUT = './data/Preprocessed/Argoverse/'

FRAC_TRAIN_VAL = 0.3

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_command_line_args():
  parser = argparse.ArgumentParser(description='Arguments')
  parser.add_argument('--input', type=str, help='')
  parser.add_argument('--output', type=str, help='')
  parser.add_argument('--trajectories', type=str2bool, default=True, help='')
  parser.add_argument('--maps', type=str2bool, default=True, help='')
  parser.add_argument('--seed', type=int, help='The random seed for the train/train_val split.')

  args, _ = parser.parse_known_args()

  input_path = INPUT
  if args.input is not None:
    input_path = args.input
  
  output_path = OUTPUT
  if args.output is not None:
    output_path = args.output
  
  do_trajectories = True
  if args.trajectories is not None:
    do_trajectories = args.trajectories
  
  do_maps = True
  if args.maps is not None:
    do_maps = args.maps
  
  seed = SEED
  if args.seed is not None:
    seed = args.seed

  return input_path, output_path, do_trajectories, do_maps, seed

class Counting_Callback():
  '''
  Multiprocessing Counting Helper.
  '''
  def __init__(self, task_name, num_data):
    self.results = []
    self.count = 0
    self.task_name = task_name
    self.num_data = num_data

  def __call__(self, res):
    self.count += 1

    if res is not None:
      self.results.append(res)

    if self.count == self.num_data:
      print("Working on {:s}... {:.2f}% Done.".format(self.task_name, self.count/self.num_data*100))
    else:
      print("Working on {:s}... {:.2f}% Done.".format(self.task_name, self.count/self.num_data*100), end="\r")
  
  def get_results(self):
    return self.results

def generate_trajectories(df, obsv_path, pred_path=None):
  """
  Generate Trajectories w/ reference frame set to 19 (the current timestep).
  """
  # Assign Frames to Timestamps
  ts_list = df['TIMESTAMP'].unique()
  ts_list.sort()

  ts_mask = []
  frames = []
  for i, ts in enumerate(ts_list):
    ts_mask.append(df['TIMESTAMP'] == ts)
    frames.append(i)

  df.loc[:, 'FRAME'] = np.select(ts_mask, frames)

  # Filter TRACK_IDs that do not exist at reference_frame.
  present_agents = df[df.FRAME == REFERENCE_FRAME]
  present_mask = np.isin(df["TRACK_ID"].to_numpy(), present_agents["TRACK_ID"].to_numpy())
  df = df[present_mask]

  track_masks = []
  observation_timelens = []
  observation_curvelens = []
  observation_curvatures = []
  prediction_timelens = []
  prediction_curvelens = []
  prediction_curvatures = []
  for track_id in df["TRACK_ID"].unique():
    # Get trajectories corresponding to a track_id
    track_mask = (df.TRACK_ID==track_id)
    track_masks.append(track_mask)

    track_df = df[track_mask]
    
    track_frames = track_df["FRAME"].to_numpy() # All frame indices except those for missing frames.
    start_frame = track_frames[0]
    end_frame = track_frames[-1]
    for earliest_frame in range(REFERENCE_FRAME, start_frame-1, -1):
      """
      Find the earliest frame within the longest continuous
      observation sequence that contains reference_frame.
      """
      if earliest_frame not in track_frames:
        earliest_frame += 1
        break
      
    obsv_timelen = REFERENCE_FRAME - earliest_frame + 1
    
    obsv_track_df = track_df[(track_df.FRAME>=earliest_frame) & (track_df.FRAME<=REFERENCE_FRAME)]
    obsv_XY = obsv_track_df[['X', 'Y']].to_numpy()
    
    if len(obsv_XY.shape) == 1:
      obsv_XY = np.expand_dims(obsv_XY, axis=0)

    obsv_td = np.diff(obsv_XY, axis=0)
    obsv_curvlen = np.linalg.norm(obsv_td, axis=1).sum()

    obsv_err = obsv_XY[-1] - obsv_XY[0]
    obsv_disp = np.sqrt(obsv_err.dot(obsv_err))

    if obsv_disp != 0.0:
      obsv_curvature = obsv_curvlen / obsv_disp
    
    else:
      obsv_curvature = float("inf")

    observation_timelens.append(obsv_timelen)
    observation_curvelens.append(obsv_curvlen)
    observation_curvatures.append(obsv_curvature)

    pred_timelen = 0
    pred_curvlen = None
    pred_curvature = None
    if pred_path is not None:
      if REFERENCE_FRAME != end_frame:
        for latest_frame in range(REFERENCE_FRAME+1, end_frame+1):
          """
          Find the latest frame within the longest continuous
          prediction sequence right next to the reference frame.
          """
          if latest_frame not in track_frames:
            latest_frame -= 1
            break
    
        pred_timelen = latest_frame - REFERENCE_FRAME
        if pred_timelen != 0:
          pred_track_df = track_df[(track_df.FRAME>REFERENCE_FRAME) & (track_df.FRAME<=latest_frame)]
          pred_XY = pred_track_df[['X', 'Y']].to_numpy()
        
          if len(pred_XY.shape) == 1:
            pred_XY = np.expand_dims(pred_XY, axis=0)

          pred_td = np.diff(pred_XY, axis=0)
          pred_curvlen = np.linalg.norm(pred_td, axis=1).sum()

          pred_err = pred_XY[-1] - pred_XY[0]
          pred_disp = np.sqrt(pred_err.dot(pred_err))

          if pred_disp != 0.0:
            pred_curvature = pred_curvlen / pred_disp
          
          else:
            pred_curvature = float("inf")

    prediction_timelens.append(pred_timelen)
    prediction_curvelens.append(pred_curvlen)
    prediction_curvatures.append(pred_curvature)

  df.loc[:, 'OBSERVATION_TIMELEN'] = np.select(track_masks, observation_timelens)
  df.loc[:, 'OBSERVATION_CURVELEN'] = np.select(track_masks, observation_curvelens)
  df.loc[:, 'OBSERVATION_CURVATURE'] = np.select(track_masks, observation_curvatures)

  df.loc[:, 'PREDICTION_TIMELEN'] = np.select(track_masks, prediction_timelens)
  df.loc[:, 'PREDICTION_CURVELEN'] = np.select(track_masks, prediction_curvelens)
  df.loc[:, 'PREDICTION_CURVATURE'] = np.select(track_masks, prediction_curvatures)

  filter_condition = None
  for track_id in df["TRACK_ID"].unique():
    """
    Process observation & prediction trajectories with missing frames by...
    Filtering observation sequences with frames earlier than earliest_frame_idx.
    Filtering prediction sequences with frames later than latest_frame_idx.
    """
    track_mask = df["TRACK_ID"] == track_id
    observation_length = df[track_mask]['OBSERVATION_TIMELEN'].iloc[0]
    prediction_length = df[track_mask]['PREDICTION_TIMELEN'].iloc[0]
    
    track_condition = track_mask & (df.FRAME > REFERENCE_FRAME - observation_length)
    if pred_path is not None:
      track_condition = track_condition & (df.FRAME < REFERENCE_FRAME + prediction_length + 1)
    
    if filter_condition is not None:
      filter_condition = filter_condition | track_condition
    else:
      filter_condition = track_condition

  df = df[filter_condition]

  # Add X_CITY & Y_CITY features.
  XY_CITY = df[['X', 'Y']].to_numpy()
  df.loc[:, 'X_CITY'], df.loc[:, 'Y_CITY'] = XY_CITY[:, 0], XY_CITY[:, 1]

  # Center X_CIYU & Y_CITY to make X & Y.
  agent_df = df[df.OBJECT_TYPE=='AGENT'] # Agent of Interest (AoI)
  translation = agent_df[agent_df.FRAME == REFERENCE_FRAME][["X_CITY", "Y_CITY"]].to_numpy()

  # XY = agent_df[["X", "Y"]].to_numpy()
  # curve_length = agent_df["OBSERVATION_CURVELEN"].to_numpy()[0]
  
  # sin = cos = None
  # if curve_length > 0.1:
  #   XY_curr = XY[REFERENCE_FRAME] # AoI at the reference_frame
  #   for i in range(1, REFERENCE_FRAME+1):
  #     XY_prev = XY[REFERENCE_FRAME-1]

  #     error = XY_curr-XY_prev
  #     z_err = np.sqrt(error.dot(error))

  #     if z_err > 0.1:
  #       x_err, y_err = error
  #       sin = y_err / z_err
  #       cos = x_err / z_err
  
  # if sin is None:
  #   sin = 0.0
  #   cos = 1.0
  # rotation = np.array([[cos, sin], [-sin, cos]])
  # XY = rotation.dot((XY - np.expand_dims(XY_curr, axis=0)).T).T

  XY_center = XY_CITY - translation
  df.loc[:, 'X'], df.loc[:, 'Y'] = XY_center[:, 0], XY_center[:, 1]

  # Partition the observation and prediction.
  observation = df[df["FRAME"] <= REFERENCE_FRAME]
  dump(observation, obsv_path)

  if pred_path is not None:
    prediction = df[df["FRAME"] > REFERENCE_FRAME]
    dump(prediction, pred_path)

def main():
  input_root, output_root, do_trajectories, do_maps, seed = get_command_line_args()
  print("Preprocessing Script for Argoverse Dataset.")
  print("Trajectories: {:s}, Maps: {:s}, Random Seed: {:d}".format("Y" if do_trajectories else "N", "Y" if do_maps else "N", seed))
  np.random.seed(seed)
  random.seed(seed)
  
  if do_trajectories:
    train_all = None
    for partition in ['train', 'train_val', 'val', 'test_obs']:
      print("Start Processing {:s} set.".format(partition))
      if 'train' in partition:
        if train_all is None:
          partition_path = P(input_root).joinpath('train', 'data')
          afl = ArgoverseForecastingLoader(partition_path)
          seq_list = afl.seq_list
          seq_list.sort()
          np.random.shuffle(seq_list)
          train_all = seq_list

        if partition == 'train':
          seq_list = train_all[int(len(train_all) * FRAC_TRAIN_VAL):]
        
        if partition == 'train_val':
          seq_list = train_all[:int(len(train_all) * FRAC_TRAIN_VAL)]

      else:
        partition_path = P(input_root).joinpath(partition, 'data')
        afl = ArgoverseForecastingLoader(partition_path)
        seq_list = afl.seq_list
        
      pool = Pool(cpu_count())
      callback = Counting_Callback(task_name="Trajectories", num_data=len(seq_list))
      for seq in seq_list:
          scn_code = int(seq.stem)

          afl.get(seq)
          df = afl.seq_df
          
          obsv_path = P(output_root).joinpath(partition, 'observation', '{:06d}-{:03d}.pkl'.format(scn_code, REFERENCE_FRAME))
          obsv_path.parent.mkdir(parents=True, exist_ok=True)

          pred_path = None
          if partition is not 'test_obs':
              pred_path = P(output_root).joinpath(partition, 'prediction', '{:06d}-{:03d}.pkl'.format(scn_code, REFERENCE_FRAME))
              pred_path.parent.mkdir(parents=True, exist_ok=True)
        
          # generate_trajectories(df.copy(), obsv_path, pred_path)
          pool.apply_async(generate_trajectories, (df.copy(), obsv_path, pred_path), callback=callback)
    
      pool.close()
      pool.join()
    
    # Create train_all set using symbolic link.
    print("Making symlinks to form train_all split... ", end="", flush=True)
    trainall_dirname = 'train_all'
    trainall_obsv_path = P(output_root).joinpath('{:s}/observation'.format(trainall_dirname))
    trainall_obsv_path.mkdir(parents=True, exist_ok=True)
    trainall_pred_path = P(output_root).joinpath('{:s}/prediction'.format(trainall_dirname))
    trainall_pred_path.mkdir(parents=True, exist_ok=True)

    train_path = P(output_root).joinpath('train')
    train_obsv_pkl = list(train_path.glob('observation/*.pkl'))
    train_pred_pkl = list(train_path.glob('prediction/*.pkl'))

    trainval_path = P(output_root).joinpath('train_val')
    trainval_obsv_pkl = list(trainval_path.glob('observation/*.pkl'))
    trainval_pred_pkl = list(trainval_path.glob('prediction/*.pkl'))

    obsv_pkl_list = train_obsv_pkl + trainval_obsv_pkl
    pred_pkl_list = train_pred_pkl + trainval_pred_pkl
    for obsv_pkl, pred_pkl in zip(obsv_pkl_list, pred_pkl_list):
      obsv_filename, obsv_split = obsv_pkl.name, obsv_pkl.parent.parent.stem
      pred_filename, pred_split = pred_pkl.name, pred_pkl.parent.parent.stem
      
      obsv_relpath = P('../../{:s}/observation/'.format(obsv_split)).joinpath(obsv_filename)
      obsv_link = trainall_obsv_path.joinpath(obsv_filename)
      obsv_link.symlink_to(obsv_relpath)
      
      pred_relpath = P('../../{:s}/prediction/'.format(pred_split)).joinpath(pred_filename)
      pred_link = trainall_pred_path.joinpath(pred_filename)
      pred_link.symlink_to(pred_relpath)
    print(" Done.")

  if do_maps:
    am = ArgoverseMap()
    for city_name in ["MIA", "PIT"]:
        print("Generating maps for {:s}.".format(city_name))

        mask_path = P(output_root).joinpath('raw_map', '{:s}_mask.pkl'.format(city_name))
        dt_path = P(output_root).joinpath('raw_map', '{:s}_dt.pkl'.format(city_name))
        mask_vis_path = P(output_root).joinpath('raw_map_visualization', '{:s}_mask_vis.png'.format(city_name))
        dt_vis_path = P(output_root).joinpath('raw_map_visualization', '{:s}_dt_vis.png'.format(city_name))
        mask_vis_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        map_mask, image_to_city = am.get_rasterized_driveable_area(city_name)

        print("Calculating Signed Distance Transform... ", end="", flush=True)
        image = map_mask.astype(np.int32)
        invert_image = 1-image
        dt = np.where(invert_image, -distance_transform_edt(invert_image), distance_transform_edt(image))
        print("Done.")

        print("Saving Results... ", end="", flush=True)
        dump({'map': map_mask, 'image_to_city': image_to_city}, mask_path)
        dump({'map': dt, 'image_to_city': image_to_city}, dt_path)
        
        mask_vis = (map_mask*255).astype(np.uint8)

        dt_max = dt.max()
        dt_min = dt.min()
        dt_vis = ((dt - dt_min)/(dt_max - dt_min)*255).astype(np.uint8)

        cv2.imwrite(str(mask_vis_path), mask_vis)
        cv2.imwrite(str(dt_vis_path), dt_vis)
        print("Done. Saved {:s}, {:s}, {:s}, and {:s}.".format(str(mask_path), str(mask_vis_path), str(dt_path), str(dt_vis_path)))

if __name__ == '__main__':
  main()