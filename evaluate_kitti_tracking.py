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
    voxelized = torch.tensor(downpcd.points, dtype=torch.float)
    voxelized = torch.cat((voxelized, torch.ones([voxelized.shape[0], 1], dtype=torch.float)), 1)
    voxelized = voxelized.t()
    voxelized = voxelized.to('cuda')
    return voxelized



# @ex.capture
def crop_local_map(PC_map, pose, velo2cam):
    """
    Crop the local point clouds around given camera pose and transform the local points to camera frame
    Args:
        PC_map (torch.tensor): 4XN, global point cloud map
        pose (torch.tensor): camera pose in global point cloud
        velo2cam (torch.tensor): the extrinsic between LiDAR and camera
    return:
        torch.tensor: 4xM, local point clouds around the position
    """
    local_map = PC_map.clone()
    pose = pose.inverse()  # pose in map, so transform points to local camera frame should be inverse
    local_map = torch.mm(pose, local_map).t()
    indexes = local_map[:, 1] > -25.
    indexes = indexes & (local_map[:, 1] < 25.)
    indexes = indexes & (local_map[:, 0] > -10.)
    indexes = indexes & (local_map[:, 0] < 100.)
    local_map = local_map[indexes]

    local_map = torch.mm(velo2cam, local_map.t())
    local_map = local_map[[2, 0, 1, 3], :]

    return local_map


# @ex.capture
def load_GT_poses(dataset_dir, seq_id='00'):
    all_files = []
    GTs_R = []
    GTs_T = []
    df_locations = pd.read_csv(os.path.join(dataset_dir, seq_id, 'poses.csv'), sep=',', dtype={'timestamp': str})
    for index, row in df_locations.iterrows():
        # if not os.path.exists(os.path.join(dataset_dir, seq_id, 'image_2', str(row['timestamp']) + '.png')):
        if not os.path.exists(f"{dataset_dir}/{seq_id}/image_2/{int(row['timestamp']):06d}.png"):
            continue
        all_files.append(f"{int(row['timestamp']):06d}")
        GT_T = torch.tensor([float(row['x']), float(row['y']), float(row['z'])])
        GT_R = torch.tensor([float(row['qw']), float(row['qx']), float(row['qy']), float(row['qz'])])
        GTs_R.append(GT_R)
        GTs_T.append(GT_T)

    return all_files, GTs_R, GTs_T


# @ex.capture
def get_calib_kitti(sequence):
    if sequence in [0, 1, 2]:
        return torch.tensor([718.856, 718.856, 607.1928, 185.2157])
    elif sequence == 3:
        return torch.tensor([721.5377, 721.5377, 609.5593, 172.854])
    elif sequence in [4, 5, 6, 7, 8, 9, 10]:
        return torch.tensor([707.0912, 707.0912, 601.8873, 183.1104])
    else:
        raise TypeError("Sequence Not Available")


# @ex.capture
def depth_generation(local_map, image_size, sequence, occu_thre, occu_kernel):
    cam_params = get_calib_kitti(int(sequence)).cuda()
    cam_model = CameraModel()
    cam_model.focal_length = cam_params[:2]
    cam_model.principal_point = cam_params[2:]
    uv, depth, points, refl = cam_model.project_pytorch(local_map, image_size)
    uv = uv.t().int().contiguous()
    depth_img = torch.zeros(image_size[:2], device='cuda', dtype=torch.float)
    depth_img += 1000.
    idx_img = (-1) * torch.ones(image_size[:2], device='cuda', dtype=torch.float)
    indexes = torch.ones(depth.shape[0], device='cuda').float()
    depth_img, idx_img = visibility.depth_image(uv, depth, indexes, depth_img, idx_img, uv.shape[0], image_size[1], image_size[0])
    depth_img[depth_img == 1000.] = 0.

    deoccl_index_img = (-1) * torch.ones(image_size[:2], dtype=torch.float, device="cuda")
    projected_points = torch.zeros_like(depth_img, device='cuda')
    projected_points, _ = visibility.visibility2(depth_img, cam_params, idx_img, projected_points, deoccl_index_img, depth_img.shape[1],
                                              depth_img.shape[0], occu_thre, occu_kernel)
    projected_points /= 100.
    projected_points = projected_points.unsqueeze(0)

    return projected_points


# @ex.capture
def custom_transform(rgb):
    to_tensor = transforms.ToTensor()
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    rgb = to_tensor(rgb)
    rgb = normalization(rgb)
    return rgb


# @ex.automain
def main(_config, seed, device):
    print(_config)
    # if _config['weight'] is not None:
    #     weights = _config['weight']
    if _config['weight'] is None:
        _config['weight'] = weights

    img_shape = (384, 1280)

    maps_file = 'map-00_0.1_0-4541.pcd'
    if _config['maps_file'] is not None:
        maps_file = _config['maps_file']

    print(f'load pointclouds from {maps_file}')
    vox_map = load_map(maps_file)
    print(f'load pointclouds finished! {vox_map.shape[1]} points')

    _config['test_sequence'] = f"{_config['test_sequence']:02d}"
    kitti_folder = os.path.split(_config["data_folder"][:-1])[0]

    kitti = pykitti.odometry(kitti_folder, _config['test_sequence'])
    velo2cam2 = kitti.calib.T_cam2_velo
    velo2cam2 = torch.from_numpy(velo2cam2).float().to('cuda')

    print('load ground truth poses')
    all_files, GTs_R, GTs_T = load_GT_poses(_config["data_folder"], _config['test_sequence'])
    print(len(all_files))

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

    # model.load_state_dict(saved_state_dict)
    model.eval()
    print("Parameter Count: %d" % count_parameters(model))

    _config['occlusion_threshold'] = checkpoint['config']['occlusion_threshold']
    _config['occlusion_kernel'] = checkpoint['config']['occlusion_kernel']

    if _config['save_log']:
        # log_file = f'./my_runs/track_seq{_config["test_sequence"]}.csv'
        log_file = f'./my_runs/CMRNet_KITTI10_2.csv'
        log_file_f = open(log_file, 'w')
        log_file = csv.writer(log_file_f)
        header = [f'timestamp', f'x', f'y', f'z',
                  f'qx', f'qy', f'qz', f'qw']
                  # ,f'in_err_r', f'in_err_t', f'err_r', f'err_t']
        log_file.writerow(header)

    est_rot = []
    est_trans = []
    errors_rot = []
    errors_trans = []
    # tracking start here:tqdm
    print('Start tracking using CMRNet...')
    end = time.time()
    for idx in range(len(all_files)):  #
        # pose initialization
        if idx == 0:
            inital_R = GTs_R[idx].to(device)
            initial_T = GTs_T[idx].to(device)
        else:
            inital_R = est_rot[idx - 1]
            initial_T = est_trans[idx - 1]

        RT = to_rotation_matrix(inital_R, initial_T)
        local_map = crop_local_map(vox_map, RT, velo2cam2)  # pointclouds under the coordinate system of inaccurate pose

        # project local map to generate depth image
        img_name = os.path.join(_config["data_folder"], _config["test_sequence"], 'image_2', all_files[idx] + '.png')
        img = Image.open(img_name)
        rgb = custom_transform(img)
        real_shape = [rgb.shape[1], rgb.shape[2], rgb.shape[0]]
        projected_points = depth_generation(local_map, real_shape, _config["test_sequence"],
                                            _config['occlusion_threshold'], _config['occlusion_kernel'])

        lidar_input = []
        rgb_input = []
        shape_pad = [0, 0, 0, 0]
        shape_pad[3] = (img_shape[0] - real_shape[0])
        shape_pad[1] = (img_shape[1] - real_shape[1])

        rgb = rgb.cuda()
        rgb = F.pad(rgb, shape_pad)
        projected_points = F.pad(projected_points, shape_pad)
        rgb_input.append(rgb)
        lidar_input.append(projected_points)
        lidar_input = torch.stack(lidar_input)
        rgb_input = torch.stack(rgb_input)

        if _config['show']:
            img_name = f'./my_runs/est_overlay/{all_files[idx]}_ini.png'
            out0 = overlay_imgs(rgb, lidar_input, img_name)
        rgb = rgb_input.to(device)
        lidar = lidar_input.to(device)

        with torch.no_grad():
            # Run the network
            d_T_predicted, d_R_predicted = model(rgb, lidar)

            d_R_predicted = quat2mat(d_R_predicted[0])
            d_T_predicted = tvector2mat(d_T_predicted[0])
            d_RT_predicted = torch.mm(d_T_predicted, d_R_predicted)

        delta_RT = torch.mm(torch.mm(velo2cam2.inverse(), d_RT_predicted), velo2cam2)
        update_RT = torch.mm(RT, delta_RT)

        if _config['show']:
            rotated_point_cloud = rotate_forward(local_map, d_RT_predicted)
            projected_points = depth_generation(rotated_point_cloud, real_shape, _config["test_sequence"],
                                                _config['occlusion_threshold'], _config['occlusion_kernel'])
            lidar = F.pad(projected_points, shape_pad)
            lidar = lidar.unsqueeze(0)
            img_name = f'./my_runs/est_overlay/{all_files[idx]}_est.png'
            out0 = overlay_imgs(rgb[0], lidar, img_name)

        predicted_R = quaternion_from_matrix(update_RT)
        predicted_T = update_RT[:3, 3]
        est_rot.append(predicted_R)
        est_trans.append(predicted_T)

        # calculate error
        GT_rot = GTs_R[idx].to(device)
        GT_trans = GTs_T[idx].to(device)
        # in_err_rot, in_err_trans = pose_error(inital_R, initial_T, GT_rot, GT_trans)
        err_rot, err_trans = pose_error(predicted_R, predicted_T, GT_rot, GT_trans)
        errors_rot.append(err_rot.item())
        errors_trans.append(err_trans.item())
        predicted_T = predicted_T.cpu().numpy()
        predicted_R = predicted_R.cpu().numpy()
        # err_rot = err_rot.cpu().numpy()
        # err_trans = err_trans.cpu().numpy()
        # in_err_rot = in_err_rot.cpu().numpy()
        # in_err_trans = in_err_trans.cpu().numpy()
        if _config['save_log']:
            log_string = [all_files[idx], str(predicted_T[0]), str(predicted_T[1]), str(predicted_T[2]),
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
        "data_folder": '/data/cky/KITTI/sequences',
        "test_sequence": 10,
        "use_prev_output": True,
        "occlusion_kernel": 5,
        "occlusion_threshold": 3.0,
        "network": 'PWC_f1',
        "norm": 'bn',
        "show": False,
        "use_reflectance": False,
        "weight": './checkpoints/checkpoint_r5.00_t1.00_e95_0.128.tar',
        # "weight": './checkpoints/iter1.tar',
        "save_name": None,
        "save_log": True,
        "maps_file": '/data/cky/KITTI/sequences/10/map-10.pcd',
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

    main(_config, seed=1234, device=device)