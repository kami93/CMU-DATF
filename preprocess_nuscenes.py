from pathlib import Path as P
import os, argparse
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple, Optional, Union

import pandas as pd
import numpy as np
import numpy.ma as ma
import cv2

from compress_pickle import dump, load

from pykalman import KalmanFilter

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.splits import create_splits_scenes

from pyquaternion import Quaternion

from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box

from scipy.ndimage.morphology import distance_transform_edt

INPUT = './data/nuScenes/trainval_meta/'
OUTPUT = './data/Preprocessed/nuScenes/'

NUM_IN_TRAIN_VAL = 200

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
  parser.add_argument('--maps', type=str2bool, default=True, help="")

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

  return input_path, output_path, do_trajectories, do_maps

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
  
  def set_num_data(self, num_data):
    self.num_data = num_data

  def get_results(self):
    return self.results

class annotation_dict(dict):
  def __init__(self):
    super(annotation_dict, self).__init__()
    self.update({'CITY_NAME':[],
                 'SAMPLE_TOKEN':[],
                 'INSTANCE_TOKEN':[],
                 'OBJECT_CATEGORY':[],
                 'OBJECT_ATTRIBUTE':[],
                 'TIMESTAMP':[],
                 'FRAME':[],
                 'X_CITY':[], 'Y_CITY':[], 'Z_CITY':[],
                 'QW':[], 'QX':[], 'QY':[], 'QZ':[]})
  
  def append(self,
             city_name,
             sample_token,
             instance_token,
             object_category,
             object_attribute,
             timestamp,
             frame,
             translation,
             rotation):
    """
    Append annotations to annotation_dict.
    """
    self.get("CITY_NAME").append(city_name)
    self.get("SAMPLE_TOKEN").append(sample_token)
    self.get("INSTANCE_TOKEN").append(instance_token)
    self.get("OBJECT_CATEGORY").append(object_category)
    self.get("OBJECT_ATTRIBUTE").append(object_attribute)
    self.get("TIMESTAMP").append(timestamp)
    self.get("FRAME").append(frame)
    
    for key, val in zip(['X_CITY', 'Y_CITY', 'Z_CITY', 'QW', 'QX', 'QY', 'QZ'],
                        translation+rotation):
      self.get(key).append(val)

def generate_trajectories(df, ref_frame, obsv_path, pred_path=None):
  """
  Generate Trajectories with reference frame (the current timestep).
  """
  df = df[(ref_frame-3 <= df.FRAME) & (df.FRAME <= ref_frame+6)]

  # Filter INSTANCE_TOKENs that do not exist at reference_frame.
  present_agents = df[df.FRAME == ref_frame]
  present_mask = np.isin(df["INSTANCE_TOKEN"].to_numpy(), present_agents["INSTANCE_TOKEN"].to_numpy())
  df = df[present_mask]

  instance_masks = []
  observation_timelens = []
  observation_curvelens = []
  observation_curvatures = []
  prediction_timelens = []
  prediction_curvelens = []
  prediction_curvatures = []
  for token in df["INSTANCE_TOKEN"].unique():
    # Get trajectories corresponding to a track_id
    instance_mask = (df.INSTANCE_TOKEN==token)
    instance_masks.append(instance_mask)

    instance_df = df[instance_mask]

    instance_frames = instance_df["FRAME"].to_numpy() # All frame indices except those for missing frames.
    start_frame = instance_frames[0]
    end_frame = instance_frames[-1]
    
    for earliest_frame in range(ref_frame, start_frame-1, -1):
      """
      Find the earliest frame within the longest continuous
      observation sequence that contains reference_frame.
      """
      if earliest_frame not in instance_frames:
        earliest_frame += 1
        break
      
    obsv_timelen = ref_frame - earliest_frame + 1
  
    obsv_instance_df = instance_df[(instance_df.FRAME>=earliest_frame) & (instance_df.FRAME<=ref_frame)]
    obsv_XYZ = obsv_instance_df[['X_CITY', 'Y_CITY', 'Z_CITY']].to_numpy()
    
    if len(obsv_XYZ.shape) == 1:
      obsv_XYZ = np.expand_dims(obsv_XYZ, axis=0)

    obsv_td = np.diff(obsv_XYZ, axis=0)
    obsv_curvlen = np.linalg.norm(obsv_td, axis=1).sum()

    obsv_err = obsv_XYZ[-1] - obsv_XYZ[0]
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
        if ref_frame != end_frame:
          for latest_frame in range(ref_frame+1, end_frame+1):
            """
            Find the latest frame within the longest continuous
            prediction sequence right next to the reference frame.
            """
            if latest_frame not in instance_frames:
              latest_frame -= 1
              break

          pred_timelen = latest_frame - ref_frame
          if pred_timelen != 0:
            pred_instance_df = instance_df[(instance_df.FRAME>ref_frame) & (instance_df.FRAME<=latest_frame)]
            pred_XYZ = pred_instance_df[['X_CITY', 'Y_CITY', 'Z_CITY']].to_numpy()

            if len(pred_XYZ.shape) == 1:
              pred_XYZ = np.expand_dims(pred_XYZ, axis=0)

            pred_td = np.diff(pred_XYZ, axis=0)
            pred_curvlen = np.linalg.norm(pred_td, axis=1).sum()

            pred_err = pred_XYZ[-1] - pred_XYZ[0]
            pred_disp = np.sqrt(pred_err.dot(pred_err))

            if pred_disp != 0.0:
              pred_curvature = pred_curvlen / pred_disp
            
            else:
              pred_curvature = float("inf")

    prediction_timelens.append(pred_timelen)
    prediction_curvelens.append(pred_curvlen)
    prediction_curvatures.append(pred_curvature)

  df.loc[:, 'OBSERVATION_TIMELEN'] = np.select(instance_masks, observation_timelens)
  df.loc[:, 'OBSERVATION_CURVELEN'] = np.select(instance_masks, observation_curvelens)
  df.loc[:, 'OBSERVATION_CURVATURE'] = np.select(instance_masks, observation_curvatures)

  df.loc[:, 'PREDICTION_TIMELEN'] = np.select(instance_masks, prediction_timelens)
  df.loc[:, 'PREDICTION_CURVELEN'] = np.select(instance_masks, prediction_curvelens)
  df.loc[:, 'PREDICTION_CURVATURE'] = np.select(instance_masks, prediction_curvatures)
  
  filter_condition = None
  for token in df["INSTANCE_TOKEN"].unique():
    """
    Process observation & prediction trajectories with missing frames by...
    Filtering observation sequences with frames earlier than earliest_frame_idx.
    Filtering prediction sequences with frames later than latest_frame_idx.
    """
    instance_mask = df["INSTANCE_TOKEN"] == token
    observation_length = df[instance_mask]['OBSERVATION_TIMELEN'].iloc[0]
    prediction_length = df[instance_mask]['PREDICTION_TIMELEN'].iloc[0]
    
    instance_condition = instance_mask & (df.FRAME > ref_frame - observation_length)
    if pred_path is not None:
      instance_condition = instance_condition & (df.FRAME < ref_frame + prediction_length + 1)
    
    if filter_condition is not None:
      filter_condition = filter_condition | instance_condition
    else:
      filter_condition = instance_condition

  df = df[filter_condition]

  agent_df = df[df.OBJECT_ATTRIBUTE=='ego'] # Agent of Interest (AoI)
  translation = agent_df[agent_df.FRAME == ref_frame][["X_CITY", "Y_CITY", "Z_CITY"]].to_numpy()
  
  # rotation = agent_df[agent_df.FRAME == ref_frame][['QW','QX','QY','QZ']].to_numpy().squeeze()
  # rot_matrix = Quaternion(rotation).rotation_matrix
  # invrot_matrix = rot_matrix.T

  # Normalize this split to ego coordinate sys.
  # XYZ_normal = invrot_matrix.dot(XYZ.T-translation)
  
  XYZ_CITY = df[["X_CITY", "Y_CITY", "Z_CITY"]].to_numpy()
  XYZ_center = XYZ_CITY - translation
  df.loc[:, "X"], df.loc[:, "Y"], df.loc[:, "Z"] = XYZ_center[:, 0], XYZ_center[:, 1], XYZ_center[:, 2]

  # Partition the observation and prediction.
  observation = df[df["FRAME"] <= ref_frame]
  dump(observation, obsv_path)

  if pred_path is not None:
    prediction = df[df["FRAME"] > ref_frame]
    dump(prediction, pred_path)

def kalman_smoother(df, scene_id):
  df = df.sort_values(by=['FRAME'])
  df.loc[:, "OCCLUSION"] = int(0) # Wheter the occlusions is dealt with.
  df.loc[:, "KALMAN"] = int(0) # Wheter the Kalman Smoothing is done.
  token_list = df["INSTANCE_TOKEN"].unique()

  output_df_rows = []
  for instance_token in token_list:
    instance_df = df[df["INSTANCE_TOKEN"] == instance_token].copy()
    
    category = instance_df["OBJECT_CATEGORY"].iloc[0]
    if ("vehicle" not in category) and ("human" not in category):
      # Do not include this instance
      # if category does not belong to vehicles or humans.
      continue

    else:
      raw_frames = instance_df['FRAME'].to_numpy()
      raw_frames_diff = np.diff(raw_frames)

      initial_raw_frame = raw_frames[0]
      last_raw_frame = raw_frames[-1]
      raw_frames_shifted = raw_frames - initial_raw_frame
      len_raw_trajectory = last_raw_frame - initial_raw_frame + 1

      # Detect static agents with the attribute condition set as
      # vehicle.parked, vehicle.stopped, pedestrian.sitting_lying_down,
      # cycle.without_rider, or pedestrian.standing for the entire sequence of frames.
      static_frame = [np.any([cond in attribute for cond in ['stopped', 'parked', 'sitting_lying_down', 'standing', 'without_rider']]) for attribute in instance_df.OBJECT_ATTRIBUTE]
      trajectory_is_static = np.all(static_frame)

      if trajectory_is_static:
        # No smoothing is performed for static agents.
        
        # Check if there exists any occlussion for this static trajectory.
        occlusion_mask = np.ones(len_raw_trajectory, dtype="int")
        occlusion_mask[raw_frames_shifted] = 0
        
        has_occlusion = 1 in occlusion_mask
        if has_occlusion:
          repeat_counts = raw_frames_diff.tolist() + [1]

          # Repeat instance_df rows at occluded points.
          instance_df_rep = pd.DataFrame(np.repeat(instance_df.values, repeat_counts, axis=0))
          instance_df_rep.columns = instance_df.columns

          instance_df_rep.loc[:, 'OCCLUSION'] = occlusion_mask
          instance_df_rep.loc[:, 'FRAME'] = [x for x in range(initial_raw_frame, last_raw_frame+1)]

          output_df_rows.append(instance_df_rep)
        
        else:
          # instance_df is direct-copied if trajectory_is_static and not trajectory_has_occlusion.
          output_df_rows.append(instance_df)

      else:
        # Kalamn Smoothing for non-static agents.
          
        # A trajectory is splitted at points where occlusions persisted for more than 3 frames.
        split_points = (raw_frames_diff > 3).nonzero()[0]
        if len(split_points) > 0:
          split_points += 1

        frames_split = np.split(raw_frames, split_points)
        for frames in frames_split:
          frames_df = instance_df[np.isin(instance_df.FRAME, frames)].copy()

          initial_frame = frames[0]
          last_frame = frames[-1]
        
          len_trajectory = last_frame - initial_frame + 1
          if len_trajectory < 3:
            # Do not perform Kalman smoothing for this split.
            # (the sequence legnth is too short)
            output_df_rows.append(frames_df)
            continue
          
          frames_df.loc[:, 'KALMAN'] = int(1)
          trajectory = frames_df[['X_CITY','Y_CITY','Z_CITY']].to_numpy()

          translation = np.expand_dims(trajectory[0], axis=1)
          rotation = frames_df[frames_df.FRAME == initial_frame][['QW','QX','QY','QZ']].to_numpy().squeeze()

          rot_matrix = Quaternion(rotation).rotation_matrix
          invrot_matrix = rot_matrix.T

          # Normalize this split to ego coordinate sys.
          trajectory_ego = invrot_matrix.dot(trajectory.T-translation)

          # Calculate initial states for Kalman smoother.
          initial_position = trajectory_ego[:, 0]
          initial_velocity = (trajectory_ego[:, 1] - trajectory_ego[:, 0]) / (frames[1] - frames[0]) * 2.0
          
          kf_initial_state = np.concatenate([initial_position, initial_velocity], axis=0)
          kf_initial_state_cov = 0.1 * np.eye(6)

          kf_initial_transition = np.array([[1.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.5, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.5],
                                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
          kf_initial_transition_cov = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]) * np.eye(6)

          kf_initial_observation = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
          kf_initial_observation_cov = 0.5 * np.eye(3)

          kf = KalmanFilter(transition_matrices=kf_initial_transition, observation_matrices=kf_initial_observation,
                            transition_covariance=kf_initial_transition_cov, observation_covariance=kf_initial_observation_cov,
                            initial_state_mean=kf_initial_state, initial_state_covariance=kf_initial_state_cov,
                            em_vars=['transition_covariance', 'observation_covariance', 'initial_state_mean', 'initial_state_covariance'])

          # Prepare a masked array for Kalman Smoothing.
          frames_shifted = frames - initial_frame

          array = np.zeros([len_trajectory, 3])
          array[frames_shifted, :] = trajectory_ego.T

          occlusion_mask = np.ones(len_trajectory, dtype="int")
          occlusion_mask[frames_shifted] = 0
          
          masked_array = ma.array(array,
                                  mask=np.repeat(occlusion_mask.reshape((-1, 1)), 3, axis=1),
                                  copy=True)
          
          # EM fit the KF initial states
          kf = kf.em(masked_array, n_iter=5)

          smoothed_state, _ = kf.smooth(masked_array)
          smoothed_trajectory_ego = smoothed_state[:,:3]
          smoothed_frames = [x for x in range(initial_frame, last_frame+1)]

          # Denormalize this split to global coordinate sys
          smoothed_trajectory = rot_matrix.dot(smoothed_trajectory_ego.T) + translation
          smoothed_trajectory = smoothed_trajectory.T # Dim X Time to Time X Dim
          
          has_occlusion = 1 in occlusion_mask
          if has_occlusion:
            # Replace X_CITY,Y_CITY,Z_CITY with the smoothed trajectory.
            # With potentially appending occluded points.
            repeat_counts = np.diff(frames).tolist() + [1]
            
            # Repeat frames_df rows at occluded points,
            # then repalce their values with the smoothed ones. 
            frames_df_rep = pd.DataFrame(np.repeat(frames_df.values, repeat_counts, axis=0))
            frames_df_rep.columns = frames_df.columns

            frames_df_rep[['X_CITY','Y_CITY','Z_CITY']] = smoothed_trajectory.astype("float")
            frames_df_rep['OCCLUSION'] = occlusion_mask
            frames_df_rep['FRAME'] = smoothed_frames
      
            output_df_rows.append(frames_df_rep)
          
          else:
            frames_df[['X_CITY','Y_CITY','Z_CITY']] = smoothed_trajectory.astype("float")

            output_df_rows.append(frames_df)

  smoothed_df = pd.concat(output_df_rows)
  
  dtypes = {"CITY_NAME": "str",
            "SAMPLE_TOKEN": "str",
            "INSTANCE_TOKEN": "str",
            "OBJECT_CATEGORY": "str",
            "OBJECT_ATTRIBUTE": "str",
            "TIMESTAMP": "int64",
            "FRAME": "int32",
            "X_CITY": "float64",
            "Y_CITY": "float64",
            "Z_CITY": "float64",
            "QW": "float64",
            "QX": "float64",
            "QY": "float64",
            "QZ": "float64",
            "OCCLUSION": "int32",
            "KALMAN": "int32"}

  smoothed_df = smoothed_df.astype(dtypes)
  smoothed_df = smoothed_df.sort_values(by=['FRAME'])

  return smoothed_df, scene_id

def get_patch(patch_box: Tuple[float, float, float, float],
              patch_angle: float = 0.0) -> Polygon:
    """
    Convert patch_box to shapely Polygon coordinates.
    :param patch_box: Patch box defined as [x_center, y_center, height, width].
    :param patch_angle: Patch orientation in degrees.
    :return: Box Polygon for patch_box.
    """
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

    return patch

def mask_for_polygons(polygons: MultiPolygon, mask: np.ndarray) -> np.ndarray:
    """
    Convert a polygon or multipolygon list to an image mask ndarray.
    :param polygons: List of Shapely polygons to be converted to numpy array.
    :param mask: Canvas where mask will be generated.
    :return: Numpy ndarray polygon mask.
    """
    if not polygons:
        return mask

    def int_coords(x):
        # function to round and convert to int
        return np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]
    cv2.fillPoly(mask, exteriors, 1)
    cv2.fillPoly(mask, interiors, 0)
    return mask

def get_drivable_area_mask(nusc_map: NuScenesMap, scale_h: int = 2, scale_w: int = 2) -> Tuple[np.ndarray, List[float]]:
    if nusc_map.map_name == 'singapore-onenorth':
        map_dims = [1586, 2026]
    elif nusc_map.map_name == 'singapore-hollandvillage':
        map_dims = [2810, 3000]
    elif nusc_map.map_name == 'singapore-queenstown':
        map_dims = [3230, 3688]
    elif nusc_map.map_name == 'boston-seaport':
        map_dims = [2980, 2120]
    else:
        raise Exception('Error: Invalid map!')

    patch_box = [map_dims[0] / 2, map_dims[1] / 2, map_dims[1], map_dims[0]]
    
    patch = get_patch(patch_box, 0.0)
    
    map_scale = 2
    
    canvas_size = (patch_box[2]*scale_h, patch_box[3]*scale_w)

    map_mask = np.zeros(canvas_size, np.uint8)
    records = nusc_map.drivable_area
    for record in records:
        polygons = [nusc_map.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]
        
        for polygon in polygons:
            new_polygon = polygon.intersection(patch)
            new_polygon = affinity.scale(new_polygon, xfact=map_scale, yfact=map_scale, origin=(0, 0))
            if not new_polygon.is_empty:
                if new_polygon.geom_type is 'Polygon':
                    new_polygon = MultiPolygon([new_polygon])
                
                map_mask = mask_for_polygons(new_polygon, map_mask)
    
    return map_mask

def main():
  input_root, output_root, do_trajectories, do_maps = get_command_line_args()
  print("Preprocessing Script for nuScenes Dataset.")
  print("Trajectories: {}, Maps: {}".format("Y" if do_trajectories else "N", "Y" if do_maps else "N"))

  if do_trajectories:
    nusc = NuScenes(version='v1.0-trainval', dataroot=input_root)
    
    name2ind = {} # Maps "scene-name" to nusc.scene(list) index.
    for ind, member in enumerate(nusc.scene):
      name2ind[member['name']] = ind

    token2attr = {} # Maps attribute_token to attribute string.
    for attribute in nusc.attribute:
      token2attr[attribute['token']] = attribute['name']
    
    splits = create_splits_scenes()

  if do_maps:
    from nuscenes.map_expansion.map_api import NuScenesMap

    city_list = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']
    for city_name in city_list:
      scale_h = scale_w = 2
      print("Generating maps for {:s}.".format(city_name))

      mask_path = P(output_root).joinpath('raw_map', '{:s}_mask.pkl'.format(city_name))
      dt_path = P(output_root).joinpath('raw_map', '{:s}_dt.pkl'.format(city_name))
      mask_vis_path = P(output_root).joinpath('raw_map_visualization', '{:s}_mask_vis.png'.format(city_name))
      dt_vis_path = P(output_root).joinpath('raw_map_visualization', '{:s}_dt_vis.png'.format(city_name))
      mask_vis_path.parent.mkdir(parents=True, exist_ok=True)
      mask_path.parent.mkdir(parents=True, exist_ok=True)

      nusc_map = NuScenesMap(input_root, city_name)

      print("Calculating a map mask with scale_h: {:d}, scale_w: {:d}... ".format(scale_h, scale_w), end="", flush=True)
      map_mask = get_drivable_area_mask(nusc_map, scale_h=2, scale_w=2)
      print("Done.")

      print("Calculating Signed Distance Transform... ", end="", flush=True)
      image = map_mask.astype(np.int32)
      invert_image = 1-image
      dt = np.where(invert_image, -distance_transform_edt(invert_image), distance_transform_edt(image))
      print("Done.")

      print("Saving Results... ", end="", flush=True)
      dump({'map': map_mask, 'scale_h': 2, 'scale_w': 2}, mask_path)
      dump({'map': dt, 'scale_h': 2, 'scale_w': 2}, dt_path)

      mask_vis = (map_mask*255).astype(np.uint8)

      dt_max = dt.max()
      dt_min = dt.min()
      dt_vis = ((dt - dt_min)/(dt_max - dt_min)*255).astype(np.uint8)

      cv2.imwrite(str(mask_vis_path), mask_vis)
      cv2.imwrite(str(dt_vis_path), dt_vis)
      print("Done. Saved {:s}, {:s}, {:s}, and {:s}.".format(str(mask_path), str(mask_vis_path), str(dt_path), str(dt_vis_path)))

  if do_trajectories:
    for partition in ['train', 'train_val', 'val']:
      print("Generating Trajectories for {:s} set.".format(partition))
      
      if 'train' in partition:
        scene_list = splits['train']
        if partition == "train":
          scene_list = scene_list[NUM_IN_TRAIN_VAL:]
          
        if partition == "train_val":
          scene_list = scene_list[:NUM_IN_TRAIN_VAL]

      else:
        scene_list = splits['val']

      pool = Pool(cpu_count())
      callback = Counting_Callback(task_name="Trajectory Imputation & Smoothing", num_data=len(scene_list))
      for name in scene_list:  
        """
        Generate a raw DataFrame object for each scene_name.
        Filter object categories other than "human" and "vehicle".
        Perform Kalman Smoothing and/or rule-based Imputation.
        """
        
        ind = name2ind[name]

        scene = nusc.scene[ind]

        log = nusc.get('log', scene['log_token'])
        location = log['location']

        data_dict = annotation_dict()

        sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        frame = 0
        passed_last = False
        while not passed_last:
          sample_data = nusc.get("sample", sample_token)
          timestamp = sample_data["timestamp"]

          # Gather pose token from LiDAR since it has timestamp synced with sample_data.
          lidar_data = nusc.get("sample_data", sample_data["data"]["LIDAR_TOP"])
          ego_pose_token = lidar_data['ego_pose_token']

          ego_pose_data = nusc.get("ego_pose", ego_pose_token)
          
          ego_translation = ego_pose_data["translation"]
          ego_rotation = ego_pose_data["rotation"]
          
          # Append Ego Motion Data
          data_dict.append(location,
                            sample_token,
                            '00000000000000000000000000000000',
                            'vehicle.ego',
                            'ego',
                            timestamp,
                            frame,
                            ego_translation,
                            ego_rotation)

          for anns_token in sample_data['anns']:
            anns_data = nusc.get("sample_annotation", anns_token)

            instance_token = anns_data['instance_token']
            instance_category = anns_data['category_name']
            
            instance_attributes = [token2attr[token] for token in anns_data['attribute_tokens']]
            instance_attributes = ", ".join(instance_attributes)

            instance_translation = anns_data["translation"]
            instance_rotation = anns_data["rotation"]

            # Append Instance Motion Data
            data_dict.append(location,
                              sample_token,
                              instance_token,
                              instance_category,
                              instance_attributes,
                              timestamp,
                              frame,
                              instance_translation,
                              instance_rotation)
          # goto next sample
          if sample_token == last_sample_token or len(sample_data['next']) == 0:
            passed_last = True
          
          else:
            sample_token = sample_data['next']
            frame += 1

        df = pd.DataFrame.from_dict(data_dict) # Generate a DataFrame
        pool.apply_async(kalman_smoother,
                        (df.copy(), name),
                        callback=callback) # Perform Kalman Smoothing
        
      pool.close()
      pool.join()

      # Get Kalman Smoothed results and sort w.r.t. scene_anme.
      smoothed_trajectories = callback.get_results()
      smoothed_trajectories.sort(key=lambda x: x[1])

      pool = Pool(cpu_count())
      callback = Counting_Callback(task_name="Trajectory Chopping & Sample Generation", num_data=float('inf'))
      num_data = 0
      for df, scene_name in smoothed_trajectories:
        """
        Chop a smoothed DataFrame into multiple samples (~33 samples per scene)
        such that each sample spans 5 seconds where the reference frame is set at the 2 second's frame.

        Then, split the sample to obsv (0~2 seconds) and pred (2~5 seconds) files.
        """
        scn_code = int(scene_name.split('-')[-1])
        
        frames = df.FRAME.to_list()
        initial_frame = frames[0]
        last_frame = frames[-1]

        for ref_frame in range(initial_frame+3, last_frame-5):
          obsv_path = P(output_root).joinpath(partition, 'observation', '{:04d}-{:03d}.pkl'.format(scn_code, ref_frame))
          obsv_path.parent.mkdir(parents=True, exist_ok=True)

          pred_path = P(output_root).joinpath(partition, 'prediction', '{:04d}-{:03d}.pkl'.format(scn_code, ref_frame))
          pred_path.parent.mkdir(parents=True, exist_ok=True)
          
          pool.apply_async(generate_trajectories, (df.copy(), ref_frame, obsv_path, pred_path), callback=callback)
          # generate_trajectories(df.copy(), ref_frame, obsv_path, pred_path)
          num_data += 1
      
      callback.set_num_data(num_data)
      pool.close()
      pool.join()

      print("Saved {:d} {:s} samples at {:s}.".format(num_data, partition, str(P(output_root).joinpath(partition))))
    
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

if __name__ == '__main__':
  main()