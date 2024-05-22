# -------------------------------------------------------------------
# Copyright (C) 2021 Carnegie Mellon University, Airlab
# Author: Huai Yu (huaiy@andrew.cmu.edu)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------
import csv
import random
import open3d as o3
import os
import glob
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torchvision import transforms
# import visibility_old as visibility
import visibility as visibility
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False
from PIL import Image
from tqdm import tqdm
import pykitti
import time

import pandas as pd

from camera_model import CameraModel
from models.CMRNet.CMRNet import CMRNet
from quaternion_distances import quaternion_distance
from utils import (invert_pose, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat, to_rotation_matrix, pose_error, count_parameters)
from argoverse.data_loading.synchronization_database import SynchronizationDB
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat
from argoverse.utils.calibration import (
    point_cloud_to_homogeneous,
    get_calibration_config,
    project_lidar_to_img,
    project_lidar_to_img_motion_compensated,
)

# ex = Experiment("2D-3D-pose-tracking")
# ex.captured_out_filter = apply_backspaces_and_linefeeds

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # noinspection PyUnusedLocal
# # @ex.config
# def config():
#     dataset = 'kitti'
#     data_folder = '/data/cky/KITTI/sequences'
#     test_sequence = 0
#     use_prev_output = True
#     occlusion_kernel = 5
#     occlusion_threshold = 3.0
#     network = 'PWC_f1'
#     norm = 'bn'
#     show = False
#     use_reflectance = False
#     weight = './checkpoints/iter1.tar'
#     save_name = None
#     save_log = False
#     maps_file = 'map-00_0.1_0-4541.pcd'
#     multi_GPU = False
#
#
# # weights = [
# #     './checkpoints/kitti/00/checkpoint_r5.00_t1.00_e99_0.139.tar',
# # ]
#
# weights = [
#     './checkpoints/iter1.tar',
#     './checkpoints/iter2.tar',
#     './checkpoints/iter3.tar'
# ]


# @ex.capture
def load_map(map_file):
    downpcd = o3.io.read_point_cloud(map_file)
    voxelized = np.asarray(downpcd.points, dtype=np.float).T
    return voxelized



# @ex.capture
def crop_local_map(vox_map, city_to_egovehicle_se3):
    local_map = vox_map.copy()
    local_map = city_to_egovehicle_se3.inverse_transform_point_cloud(local_map.T)

    indexes = local_map[:, 1] > -25
    indexes = indexes & (local_map[:, 1] < 25.)
    indexes = indexes & (local_map[:, 0] > -10.)
    indexes = indexes & (local_map[:, 0] < 100.)

    local_map = local_map[indexes]

    return local_map


# @ex.capture
def depth_generation(pc, calib_data, device, real_shape, occlusion_threshold, occlusion_kernel):
    uv, uv_cam, valid_pts_bool, camera_config = \
        project_lidar_to_img(pc, copy.deepcopy(calib_data), "ring_front_center", True)

    intrinsic = camera_config.intrinsic
    cam_params = [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]]
    cam_params = torch.tensor(cam_params, dtype=torch.float).to(device)
    depth = uv_cam[2, :]
    uv = np.round(uv[valid_pts_bool]).astype(np.int32)
    uv = torch.tensor(uv, dtype=torch.int32).to(device)
    depth = depth[valid_pts_bool]
    depth = torch.tensor(depth, dtype=torch.float).to(device)
    mask_u = uv[:, 0] < 960
    mask_v = uv[:, 1] < 600
    # mask_u = uv[:, 0] < 1280
    # mask_v = uv[:, 1] < 384
    mask_uv = mask_u * mask_v
    uv = uv[mask_uv, :]
    depth = depth[mask_uv]
    indexes = torch.ones(depth.shape[0]).to(device)

    depth_img = torch.zeros(real_shape[:2], dtype=torch.float).to(device)
    depth_img += 1000.
    idx_img = (-1) * torch.ones(real_shape[:2], dtype=torch.float).to(device)
    indexes = indexes.float()
    depth_img, idx_img = visibility.depth_image(uv, depth, indexes,
                                                depth_img, idx_img, uv.shape[0],
                                                real_shape[1], real_shape[0])
    depth_img[depth_img == 1000.] = 0.

    deoccl_index_img = (-1) * torch.ones(real_shape[:2], dtype=torch.float).to(device)
    depth_img_no_occlusion = torch.zeros_like(depth_img).to(device)
    depth_img_no_occlusion, _ = visibility.visibility2(depth_img, cam_params, idx_img,
                                                         depth_img_no_occlusion,
                                                         deoccl_index_img,
                                                         real_shape[1],
                                                         real_shape[0],
                                                         occlusion_threshold,
                                                         occlusion_kernel)
    depth_img_no_occlusion /= 100.
    depth_img_no_occlusion = depth_img_no_occlusion.unsqueeze(0)

    return depth_img_no_occlusion, cam_params


# @ex.capture
def custom_transform(rgb):
    to_tensor = transforms.ToTensor()
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    rgb = to_tensor(rgb)
    rgb = normalization(rgb)
    return rgb


# @ex.automain
def main(_config, device):
    print(_config)
    if _config['weight'] is None:
        _config['weight'] = weights

    img_shape = (384, 1280)
    occlusion_threshold, occlusion_kernel = 3., 5

    maps_file = os.path.join(_config["data_folder"], _config['test_sequence'], "map_0.1.pcd")

    print(f'load pointclouds from {maps_file}')
    vox_map = load_map(maps_file)
    print(f'load pointclouds finished! {vox_map.shape[1]} points')

    if _config['network'].startswith('PWC'):
        feat = 1
        md = 4
        split = _config['network'].split('_')
        for item in split[1:]:
            if item.startswith('f'):
                feat = int(item[-1])
            elif item.startswith('md'):
                md = int(item[2:])
        assert 0 < feat < 7, "Feature Number from PWC have to be between 1 and 6"
        assert 0 < md, "md must be positive"
        model = CMRNet(img_shape, use_feat_from=feat, md=md,
                       use_reflectance=_config['use_reflectance'])
    else:
        raise TypeError("Network unknown")
    if torch.cuda.device_count() > 1 and _config['multi_GPU']:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    print(f"Loading weights from {_config['weight']}")
    checkpoint = torch.load(_config['weight'], map_location='cuda')
    saved_state_dict = checkpoint['state_dict']

    # if model trained using single GPU, then here run with multi-GPUs, can do this.
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in saved_state_dict.items():
        if 'module' in k:
            k = k[7:]
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Parameter Count: %d" % count_parameters(model))

    _config['occlusion_threshold'] = checkpoint['config']['occlusion_threshold']
    _config['occlusion_kernel'] = checkpoint['config']['occlusion_kernel']

    sdb = SynchronizationDB(_config["data_folder"], collect_single_log_id=_config['test_sequence'])
    calib_fpath = os.path.join(_config["data_folder"], _config['test_sequence'], "vehicle_calibration_info.json")
    calib_data = read_json_file(calib_fpath)
    img_paths = sorted(glob.glob(os.path.join(_config["data_folder"], _config['test_sequence'], "ring_front_center", "*.jpg")))

    if _config['save_log']:
        log_file = f'./my_runs/CMRNet_Argoverse1_2.csv'
        log_file_f = open(log_file, 'w')
        log_file = csv.writer(log_file_f)
        header = [f'timestamp', f'x', f'y', f'z',
                  f'qx', f'qy', f'qz', f'qw']
        log_file.writerow(header)

    est_rot = []
    est_trans = []
    errors_rot = []
    errors_trans = []
    k = 0
    # tracking start here:tqdm
    print('Start tracking using CMRNet...')
    end = time.time()
    for idx in range(len(img_paths) - 1 - k):
        idx = idx + k
        cur_img_fpath = img_paths[idx]
        cur_img_timestamp_ns = cur_img_fpath.split("/")[-1].split(".")[0].split("_")[-1]
        cur = Image.open(cur_img_fpath)
        cur = custom_transform(cur)
        cur = cur.to(device)
        real_shape = [cur.shape[1] // 2, cur.shape[2] // 2, cur.shape[0]]

        pose_fpath = os.path.join(_config["data_folder"], _config['test_sequence'], "poses", f"city_SE3_egovehicle_{cur_img_timestamp_ns}.json")
        pose_data = read_json_file(pose_fpath)
        GT_R = np.array(pose_data["rotation"])
        GT_T = np.array(pose_data["translation"])
        if idx - k == 0:
            initial_R = GT_R
            initial_T = GT_T
            est_rot.append(torch.tensor(GT_R).to(device))
            est_trans.append(torch.tensor(GT_T).to(device))
        else:
            initial_R = est_rot[idx - k].cpu().detach().numpy()
            initial_T = est_trans[idx - k].cpu().detach().numpy()
        city_to_egovehicle_se3 = SE3(rotation=quat2rotmat(initial_R), translation=initial_T)

        RT = to_rotation_matrix(torch.tensor(initial_R, device=device),
                                torch.tensor(initial_T, device=device))
        local_map = crop_local_map(vox_map, city_to_egovehicle_se3)
        pc = point_cloud_to_homogeneous(copy.deepcopy(local_map)).T
        sparse, cam_params = depth_generation(pc, calib_data, device, real_shape, occlusion_threshold, occlusion_kernel)

        # img_name = f'./my_runs/ini_overlay/{cur_img_timestamp_ns}_ini.png'
        # out0 = overlay_imgs(cur, sparse.unsqueeze(0), img_name)

        ## downsample and crop
        downsample = transforms.Resize([600, 960], interpolation=Image.NEAREST)
        cur = downsample(cur)
        x = (cur.shape[1] - 384) // 2
        cur = cur[:, x:x + 384, :]
        cur = cur.unsqueeze(0)
        sparse = sparse[:, x:x + 384, :]
        sparse = sparse.unsqueeze(0)

        shape_pad = [0, 0, 0, 0]
        shape_pad[1] = (img_shape[1] - real_shape[1])
        cur = F.pad(cur, shape_pad)
        sparse = F.pad(sparse, shape_pad)

        img_name = f'./my_runs/ini_overlay/{cur_img_timestamp_ns}_ini.png'
        out0 = overlay_imgs(cur[0, ...], sparse, img_name)

        with torch.no_grad():
            # Run the network
            d_T_predicted, d_R_predicted = model(cur, sparse)

            d_R_predicted = quat2mat(d_R_predicted[0])
            d_T_predicted = tvector2mat(d_T_predicted[0])
            d_RT_predicted = torch.mm(d_T_predicted, d_R_predicted)

        # delta_RT = torch.mm(torch.mm(velo2cam2.inverse(), d_RT_predicted), velo2cam2)
        # update_RT = torch.mm(RT, delta_RT)
        update_RT = torch.mm(RT, d_RT_predicted)


        predicted_R = quaternion_from_matrix(update_RT)
        predicted_T = update_RT[:3, 3]
        est_rot.append(predicted_R)
        est_trans.append(predicted_T)

        # calculate error
        err_rot, err_trans = pose_error(predicted_R, predicted_T, torch.tensor(GT_R).to(device), torch.tensor(GT_T).to(device))
        errors_rot.append(err_rot.item())
        errors_trans.append(err_trans.item())
        predicted_T = predicted_T.cpu().numpy()
        predicted_R = predicted_R.cpu().numpy()
        # err_rot = err_rot.cpu().numpy()
        # err_trans = err_trans.cpu().numpy()
        # in_err_rot = in_err_rot.cpu().numpy()
        # in_err_trans = in_err_trans.cpu().numpy()
        if _config['save_log']:
            log_string = [cur_img_timestamp_ns, str(predicted_T[0]), str(predicted_T[1]), str(predicted_T[2]),
                          str(predicted_R[1]), str(predicted_R[2]), str(predicted_R[3]), str(predicted_R[0])]
                          # ,str(in_err_rot[0]), str(in_err_trans), str(err_rot[0]), str(err_trans)]
            log_file.writerow(log_string)

        print(idx, ": ", np.mean(errors_trans), np.mean(errors_rot),
              np.std(errors_trans), np.std(errors_rot), (time.time()-end)/(idx+1))

    errors_rot = torch.tensor(errors_rot)
    errors_trans = torch.tensor(errors_trans)
    print(f'mean rotation error: {errors_rot.mean():.4f}, translation error: {errors_trans.mean():.4f}')


if __name__ == '__main__':
    _config = {
        "dataset": 'kitti',
        "data_folder": '/data/cky/Argoverse/argoverse-tracking/train1',
        "test_sequence": "25952736-2595-2595-2595-225953853440",
        "use_prev_output": True,
        "occlusion_kernel": 5,
        "occlusion_threshold": 3.0,
        "network": 'PWC_f1',
        "norm": 'bn',
        "show": False,
        "use_reflectance": False,
        # "weight": './checkpoints/checkpoint_r5.00_t1.00_e95_0.128.tar',
        "weight": './checkpoints/iter1.tar',
        "save_name": None,
        "save_log": True,
        "maps_file": None,
        "multi_GPU": False,
    }


    # weights = [
    #     './checkpoints/kitti/00/checkpoint_r5.00_t1.00_e99_0.139.tar',
    # ]

    weights = [
        './checkpoints/iter1.tar',
        './checkpoints/iter2.tar',
        './checkpoints/iter3.tar'
    ]

    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.set_device(1)

    main(_config, device=device)