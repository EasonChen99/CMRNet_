import csv
import random
import open3d as o3
import cv2
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
SETTINGS.CONFIG.READ_ONLYargs = False
from PIL import Image
from tqdm import tqdm
import pykitti

import pandas as pd

from camera_model import CameraModel
from models.CMRNet.CMRNet import CMRNet
from quaternion_distances import quaternion_distance
from utils import (invert_pose, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat, to_rotation_matrix, pose_error, mat2xyzrpy)

# flow estimation
from flow_model.core.raft import RAFT
from flow_model.core.utils.utils import InputPadder
from flow_viz import flow_to_image

import argparse
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform

# ex = Experiment("2D-3D-pose-tracking")
# ex.captured_out_filter = apply_backspaces_and_linefeeds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# @ex.capture
def load_map(map_file):
    downpcd = o3.io.read_point_cloud(map_file)
    voxelized = torch.tensor(np.asarray(downpcd.points), dtype=torch.float)
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
    if sequence == 0:
        return torch.tensor([718.856, 718.856, 607.1928, 185.2157])
    elif sequence == 3:
        return torch.tensor([721.5377, 721.5377, 609.5593, 172.854])
    elif sequence in [5, 6, 7, 8, 9]:
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
    depth_img = visibility.depth_image(uv, depth, depth_img, uv.shape[0], image_size[1], image_size[0])
    depth_img[depth_img == 1000.] = 0.

    projected_points = torch.zeros_like(depth_img, device='cuda')
    projected_points = visibility.visibility2(depth_img, cam_params, projected_points, depth_img.shape[1],
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
def demo(args):
    print("-"*80)
    print(args)
    print("-"*80)
    img_shape = (384, 1280)

    maps_file = 'map-00_0.1_0-4541.pcd'
    if args.maps_file is not None:
        maps_file = args.maps_file

    print(f'load pointclouds from {maps_file}')
    vox_map = load_map(maps_file)
    print(f'load pointclouds finished! {vox_map.shape[1]} points')

    args.test_sequence = f"{args.test_sequence:02d}"
    kitti_folder = os.path.split(args.data_folder[:-1])[0]

    kitti = pykitti.odometry(kitti_folder, args.test_sequence)
    velo2cam2 = kitti.calib.T_cam2_velo
    velo2cam2 = torch.from_numpy(velo2cam2).float().to('cuda')

    print('load ground truth poses')
    all_files, GTs_R, GTs_T = load_GT_poses(args.data_folder, args.test_sequence)
    print(len(all_files))

    # load localization model
    if args.network.startswith('PWC'):
        feat = 1
        md = 4
        split = args.network.split('_')
        for item in split[1:]:
            if item.startswith('f'):
                feat = int(item[-1])
            elif item.startswith('md'):
                md = int(item[2:])
        assert 0 < feat < 7, "Feature Number from PWC have to be between 1 and 6"
        assert 0 < md, "md must be positive"
        model = CMRNet(img_shape, use_feat_from=feat, md=md,
                       use_reflectance=args.use_reflectance)
    else:
        raise TypeError("Network unknown")
    if torch.cuda.device_count() > 1 and args.multi_GPU:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    print(f"Loading weights from {args.weight}")
    checkpoint = torch.load(args.weight, map_location='cuda')
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


    # load flow estimation model
    if args.use_flow:
        model_flow = torch.nn.DataParallel(RAFT(args))
        model_flow.load_state_dict(torch.load(args.model))
        model_flow = model_flow.module
        model_flow.to(device)
        model_flow.eval()


    args.occlusion_threshold = checkpoint['config']['occlusion_threshold']
    args.occlusion_kernel = checkpoint['config']['occlusion_kernel']

    if args.save_log:
        log_file = f'./my_runs/track_seq{args.test_sequence}.csv'
        log_file_f = open(log_file, 'w')
        log_file = csv.writer(log_file_f)
        header = [f'timestamp', f'x', f'y', f'z',
                  f'qx', f'qy', f'qz', f'qw',
                  f'in_err_r', f'in_err_t', f'err_r', f'err_t']
        log_file.writerow(header)

    len_trajMap = 1000
    trajMap = np.zeros((len_trajMap, len_trajMap, 3), dtype=np.uint8)

    est_rot = []
    est_trans = []
    errors_rot = []
    errors_trans = []
    if args.save_fig:
        pre_warp_img = np.zeros((376, 1241), dtype=np.uint8)
    print('Start tracking using CMRNet...')
    for idx in tqdm(range(len(all_files))):
        # pose initialization
        if idx == 0:
            inital_R = GTs_R[idx].to(device)
            initial_T = GTs_T[idx].to(device)
        else:
            inital_R = est_rot[idx - 1]
            initial_T = est_trans[idx - 1]


        # VO
        img_name = os.path.join(args.data_folder, args.test_sequence, 'image_2', all_files[idx] + '.png')
        img = Image.open(img_name)
        if idx > 0:
            pre_img_name = os.path.join(args.data_folder, args.test_sequence, 'image_2', all_files[idx - 1] + '.png')
            pre_img = Image.open(pre_img_name)

            gray_cur_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
            gray_pre_img = cv2.cvtColor(np.asarray(pre_img), cv2.COLOR_RGB2GRAY)

            # flow estimation
            if args.use_flow:
                # flow = cv2.calcOpticalFlowFarneback(gray_pre_img, gray_cur_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                img_torch = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float()[None].to(device)
                pre_img_torch = torch.from_numpy(np.asarray(pre_img)).permute(2, 0, 1).float()[None].to(device)
                padder = InputPadder(img_torch.shape)
                img_torch, pre_img_torch = padder.pad(img_torch, pre_img_torch)
                flow_low, flow_up = model_flow(pre_img_torch, img_torch, iters=20, test_mode=True)
                flow = flow_up[0].permute(1, 2, 0).cpu().detach().numpy()

                if args.save_fig:
                    dst_path = f"./my_runs/flow"
                    if not os.path.exists(dst_path):
                        os.mkdir(dst_path)
                    flow_img = flow_to_image(flow)
                    cv2.imwrite(f"{dst_path}/{idx:06d}.png", flow_img)


                if args.save_fig:
                    dst_path = f"./my_runs/warp"
                    if not os.path.exists(dst_path):
                        os.mkdir(dst_path)
                    warp_img = np.zeros((376, 1241), dtype=np.uint8)
                    if idx == 1:
                        pre_warp_img = gray_pre_img
                    for i in range(flow.shape[0]):
                        for j in range(flow.shape[1]):
                            if i + flow[i, j, 1] >= 0 and i + flow[i, j, 1] < flow.shape[0]:
                                if j + flow[i, j, 0] >= 0 and j + flow[i, j, 0] < flow.shape[1]:
                                        warp_img[int(i + flow[i, j, 1]), int(j + flow[i, j, 0])] = pre_warp_img[i, j]
                    pre_warp_img = warp_img
                    cv2.imwrite(f"{dst_path}/{idx:06d}.png", warp_img)

            # feature matching
            if not args.use_flow and args.use_match:
                # create ORB features
                orb = cv2.ORB_create(nfeatures=6000)

                # find the keypoints and descriptors with ORB
                kp1, des1 = orb.detectAndCompute(gray_pre_img, None)
                kp2, des2 = orb.detectAndCompute(gray_cur_img, None)

                # use brute-force matcher
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

                # Match ORB descriptors
                matches = bf.match(des1, des2)

                # Sort the matched keypoints in the order of matching distance
                # so the best matches came to the front
                matches = sorted(matches, key=lambda x: x.distance)

                # if args.save_fig:
                #     dst_path = f"./my_runs/featurematching"
                #     if not os.path.exists(dst_path):
                #         os.mkdir(dst_path)
                #     img_matching = cv2.drawMatches(gray_pre_img, kp1, gray_cur_img, kp2, matches[0:100], None)
                #     cv2.imwrite(f"{dst_path}/{idx:06d}.png", img_matching)

                pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

                # compute essential matrix
                cam_params = get_calib_kitti(int(args.test_sequence)).numpy()
                K = np.array([[cam_params[0], 0., cam_params[2]],
                              [0., cam_params[1], cam_params[3]],
                              [0., 0., 1.]])
                E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1)
                pts1 = pts1[mask.ravel() == 1]
                pts2 = pts2[mask.ravel() == 1]
                _, R, t, mask = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K)

                # get camera motion
                R = R.transpose()
                t = -np.matmul(R, t)
                # import sys
                # sys.exit()

            if args.use_flow or args.use_match:
                VO_R = np.eye(4)
                VO_R[:3, :3] = R
                VO_t = np.eye(4)
                VO_t[:3, 3] = t.transpose()[0, [2, 1, 0]]
                VO_pose = np.matmul(VO_R, VO_t)
                VO_pose = torch.tensor(VO_pose, device=device).float()
            else:
                VO_pose = torch.eye(4, device=device).float()


        RT = to_rotation_matrix(inital_R, initial_T)
        if idx > 0:
            RT = torch.mm(RT, VO_pose)
        local_map = crop_local_map(vox_map, RT, velo2cam2)  # pointclouds under the coordinate system of inaccurate pose

        # project local map to generate depth image
        rgb = custom_transform(img)
        real_shape = [rgb.shape[1], rgb.shape[2], rgb.shape[0]]
        projected_points = depth_generation(local_map, real_shape, args.test_sequence,
                                            args.occlusion_threshold, args.occlusion_kernel)

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

        if args.save_fig:
            dst_path = f"./my_runs/ini_overlay"
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            img_name = f'{dst_path}/{all_files[idx]}_ini.png'
            out0 = overlay_imgs(rgb, lidar_input, img_name)
        rgb = rgb_input.to(device)
        lidar = lidar_input.to(device)

        with torch.no_grad():
            # Run the network
            d_T_predicted, d_R_predicted = model.forward(rgb, lidar)

            d_R_predicted = quat2mat(d_R_predicted[0])
            d_T_predicted = tvector2mat(d_T_predicted[0])
            d_RT_predicted = torch.mm(d_T_predicted, d_R_predicted)

        delta_RT = torch.mm(torch.mm(velo2cam2.inverse(), d_RT_predicted), velo2cam2)
        update_RT = torch.mm(RT, delta_RT)
        # update_RT = RT

        if args.save_fig:
            dst_path = f"./my_runs/pred_overlay"
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            rotated_point_cloud = rotate_forward(local_map, d_RT_predicted)
            projected_points = depth_generation(rotated_point_cloud, real_shape, args.test_sequence,
                                                args.occlusion_threshold, args.occlusion_kernel)
            lidar = F.pad(projected_points, shape_pad)
            lidar = lidar.unsqueeze(0)
            img_name = f'{dst_path}/{all_files[idx]}_est.png'
            out0 = overlay_imgs(rgb[0], lidar, img_name)

        predicted_R = quaternion_from_matrix(update_RT)
        predicted_T = update_RT[:3, 3]
        est_rot.append(predicted_R)
        est_trans.append(predicted_T)

        # calculate error
        GT_rot = GTs_R[idx].to(device)
        GT_trans = GTs_T[idx].to(device)
        in_err_rot, in_err_trans = pose_error(inital_R, initial_T, GT_rot, GT_trans)
        err_rot, err_trans = pose_error(predicted_R, predicted_T, GT_rot, GT_trans)
        errors_rot.append(err_rot)
        errors_trans.append(err_trans)
        predicted_T = predicted_T.cpu().numpy()
        predicted_R = predicted_R.cpu().numpy()
        err_rot = err_rot.cpu().numpy()
        err_trans = err_trans.cpu().numpy()
        in_err_rot = in_err_rot.cpu().numpy()
        in_err_trans = in_err_trans.cpu().numpy()
        if args.save_log is True:
            log_string = [all_files[idx], str(predicted_T[0]), str(predicted_T[1]), str(predicted_T[2]),
                          str(predicted_R[1]), str(predicted_R[2]), str(predicted_R[3]), str(predicted_R[0]),
                          str(in_err_rot[0]), str(in_err_trans), str(err_rot[0]), str(err_trans)]
            log_file.writerow(log_string)

        offset_draw = (int(len_trajMap/2))
        cv2.circle(trajMap, (int(predicted_T[0])+offset_draw, int(predicted_T[1])+offset_draw), 1, (255,0,0), 2)

        cv2.imwrite('./my_runs/trajMap_2.png', trajMap)

    errors_rot = torch.tensor(errors_rot)
    errors_trans = torch.tensor(errors_trans)
    print(f'mean rotation error: {errors_rot.mean():.4f}, translation error: {errors_trans.mean():.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # CMRNet
    parser.add_argument('--dataset', default="kitti")
    parser.add_argument('--data_folder', default='/data/cky/KITTI/sequences/')
    parser.add_argument('--test_sequence', default=0)
    parser.add_argument('--occlusion_kernel', default=5)
    parser.add_argument('--occlusion_threshold', default=3.0)
    parser.add_argument('--network', default='PWC_f1')
    parser.add_argument('--norm', default='bn')
    parser.add_argument('--use_reflectance', action='store_true')
    parser.add_argument('--weight', default='./checkpoints/checkpoint_r5.00_t1.00_e95_0.128.tar')
    parser.add_argument('--maps_file', default='map-00_0.1_0-4541.pcd')
    parser.add_argument('--save_name', default=None)
    parser.add_argument('--multi_GPU', type=bool, default=True)
    # RAFT
    parser.add_argument('--model', default="/home/cky/2D3DRegistration/CMRNet/flow_model/models/raft-kitti.pth", help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    # config
    parser.add_argument('--use_prev_output', action='store_true')
    parser.add_argument('--use_flow', action='store_true')
    parser.add_argument('--use_match', action='store_true')
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--save_fig', action='store_true')
    args = parser.parse_args()

    demo(args)
