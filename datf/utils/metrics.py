import os
import sys
import time
import numpy as np
import datetime

import pickle as pkl

import matplotlib.pyplot as plt
import cv2
import torch
import pdb
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

import logging

from multiprocessing import Pool

def dac(gen_trajs, map_file):
    if '.png' in map_file:
        map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)

    elif '.pkl' in map_file:
        with open(map_file, 'rb') as pnt:
            map_array = pkl.load(pnt)

    da_mask = np.any(map_array > 0, axis=-1)

    num_agents, num_candidates, decoding_timesteps = gen_trajs.shape[:3]
    dac = []

    gen_trajs = ((gen_trajs + 56) * 2).astype(np.int64)

    stay_in_da_count = [0 for i in range(num_agents)]
    for k in range(num_candidates):
        gen_trajs_k = gen_trajs[:, k]

        stay_in_da = [True for i in range(num_agents)]

        oom_mask = np.any( np.logical_or(gen_trajs_k >= 224, gen_trajs_k < 0), axis=-1 )
        diregard_mask = oom_mask.sum(axis=-1) > 2
        for t in range(decoding_timesteps):
            gen_trajs_kt = gen_trajs_k[:, t]
            oom_mask_t = oom_mask[:, t]
            x, y = gen_trajs_kt.T

            lin_xy = (x*224+y)
            lin_xy[oom_mask_t] = -1
            for i in range(num_agents):
                xi, yi = x[i], y[i]
                _lin_xy = lin_xy.tolist()
                lin_xyi = _lin_xy.pop(i)

                if diregard_mask[i]:
                    continue

                if oom_mask_t[i]:
                    continue

                if not da_mask[yi, xi] or (lin_xyi in _lin_xy):
                    stay_in_da[i] = False
        
        for i in range(num_agents):
            if stay_in_da[i]:
                stay_in_da_count[i] += 1
    
    for i in range(num_agents):
        if diregard_mask[i]:
            dac.append(0.0)
        else:
            dac.append(stay_in_da_count[i] / num_candidates)
    
    dac_mask = np.logical_not(diregard_mask)

    return np.array(dac), dac_mask


def dao(gen_trajs, map_file):
    if '.png' in map_file:
        map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)

    elif '.pkl' in map_file:
        with open(map_file, 'rb') as pnt:
            map_array = pkl.load(pnt)

    da_mask = np.any(map_array > 0, axis=-1)

    num_agents, num_candidates, decoding_timesteps = gen_trajs.shape[:3]
    dao = [0 for i in range(num_agents)]

    occupied = [[] for i in range(num_agents)]

    gen_trajs = ((gen_trajs + 56) * 2).astype(np.int64)

    for k in range(num_candidates):
        gen_trajs_k = gen_trajs[:, k]

        oom_mask = np.any( np.logical_or(gen_trajs_k >= 224, gen_trajs_k < 0), axis=-1 )
        diregard_mask = oom_mask.sum(axis=-1) > 2

        for t in range(decoding_timesteps):
            gen_trajs_kt = gen_trajs_k[:, t]
            oom_mask_t = oom_mask[:, t]
            x, y = gen_trajs_kt.T

            lin_xy = (x*224+y)
            lin_xy[oom_mask_t] = -1
            for i in range(num_agents):
                xi, yi = x[i], y[i]
                _lin_xy = lin_xy.tolist()
                lin_xyi = _lin_xy.pop(i)

                if diregard_mask[i]:
                    continue

                if oom_mask_t[i]:
                    continue

                if lin_xyi in occupied[i]:
                    continue

                if da_mask[yi, xi] and (lin_xyi not in _lin_xy):
                    occupied[i].append(lin_xyi)
                    dao[i] += 1

    for i in range(num_agents):
        if diregard_mask[i]:
            dao[i] = 0.0
        else:
            dao[i] /= da_mask.sum()

    dao_mask = np.logical_not(diregard_mask)
    
    return np.array(dao), dao_mask

def write_img_output(gen_trajs, src_trajs, src_lens, tgt_trajs, tgt_lens, map_file, output_file):
    """abcd"""
    if '.png' in map_file:
        map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)
        map_array = cv2.cvtColor(map_array, cv2.COLOR_BGR2RGB)

    elif '.pkl' in map_file:
        with open(map_file, 'rb') as pnt:
            map_array = pkl.load(pnt)

    H, W = map_array.shape[:2]
    fig = plt.figure(figsize=(float(H) / float(80), float(W) / float(80)),
                    facecolor='k', dpi=80)

    ax = plt.axes()
    ax.imshow(map_array, extent=[-56, 56, 56, -56])
    ax.set_aspect('equal')
    ax.set_xlim([-56, 56])
    ax.set_ylim([-56, 56])

    plt.gca().invert_yaxis()
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
    plt.margins(0,0)
    
    num_tgt_agents, num_candidates = gen_trajs.shape[:2]
    num_src_agents = len(src_trajs)

    for k in range(num_candidates):
        gen_trajs_k = gen_trajs[:, k]

        x_pts_k = []
        y_pts_k = []
        for i in range(num_tgt_agents):
            gen_traj_ki = gen_trajs_k[i]
            tgt_len_i = tgt_lens[i]
            x_pts_k.extend(gen_traj_ki[:tgt_len_i, 0])
            y_pts_k.extend(gen_traj_ki[:tgt_len_i, 1])

        ax.scatter(x_pts_k, y_pts_k, s=0.5, marker='o', c='b')
    
    x_pts = []
    y_pts = []
    for i in range(num_src_agents):
            src_traj_i = src_trajs[i]
            src_len_i = src_lens[i]
            x_pts.extend(src_traj_i[:src_len_i, 0])
            y_pts.extend(src_traj_i[:src_len_i, 1])

    ax.scatter(x_pts, y_pts, s=2.0, marker='x', c='g')

    x_pts = []
    y_pts = []
    for i in range(num_tgt_agents):
            tgt_traj_i = tgt_trajs[i]
            tgt_len_i = tgt_lens[i]
            x_pts.extend(tgt_traj_i[:tgt_len_i, 0])
            y_pts.extend(tgt_traj_i[:tgt_len_i, 1])

    ax.scatter(x_pts, y_pts, s=2.0, marker='o', c='r')

    fig.canvas.draw()
    buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buffer = buffer.reshape((H, W, 3))

    buffer = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_file, buffer)
    ax.clear()
    plt.close(fig)