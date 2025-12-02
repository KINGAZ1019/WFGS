#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple

import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from scene.gaussian_model import BasicPointCloud
import torch
import imageio
import math
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation
import random

class SpectrumInfo(NamedTuple):
    R: np.array           # A NumPy array representing the rotation matrix of the camera.
    T_rx: np.array        # A NumPy array representing the translation vector of the camera.
    T_tx: np.array
    spectrum: np.array    # A NumPy array containing image data.
    spectrum_path: str    # A string representing the file path to the image.
    spectrum_name: str    # A string representing the name of the image.
    width: int     
    height: int
    freq: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_spectrums: list
    test_spectrums: list
    nerf_normalization: dict
    ply_path: str


def split_dataset_train_v2(datadir, train_path, test_path, ratio=0.8):
    # here the ratio is the ratio of training set
    # train and test set ratios are fixed to 0.7 and 0.3
    llffhold_t = 8

    spectrum_dir = os.path.join(datadir, 'spectrum')
    spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
    image_names = [x.split('.')[0] for x in spt_names]

    len_image = len(image_names)

    # Set the seed for reproducibility
    random.seed(1994)
    np.random.seed(1994)

    test_index = np.arange(int(len_image))[:: llffhold_t]
    train_index_raw = np.array([j for j in np.arange(int(len_image)) if (j not in test_index)])
    train_len = len(train_index_raw)

    number_train = int(train_len * ratio)
    train_index = np.random.choice(train_index_raw, number_train, replace=False)

    print("\n Ratio in Train set: {}....  Train set: {}....  Test set: {}....\n".format(ratio, number_train, len(test_index)))

    train_image = [image_names[idx] for idx in train_index]
    test_image = [image_names[idx] for idx in test_index]

    np.savetxt(train_path, train_index, fmt='%s')
    np.savetxt(test_path,  test_index,  fmt='%s')


def readSpectrumImage(data_dir_path):
    data_infos = []

    tx_pos_path = os.path.join(data_dir_path, 'tx_pos.csv')
    tx_pos = pd.read_csv(tx_pos_path).values               # (N, 3)
    # tx_pos = torch.tensor(tx_pos, dtype=torch.float32)   # torch.Size([N, 3]), N is N is 6123, torch.Size([6123, 3])

    gateway_pos_path = os.path.join(data_dir_path, 'gateway_info.yml')
    spectrum_dir     = os.path.join(data_dir_path, 'spectrum')
    spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
    # 读取频率信息
    freq_path = os.path.join(data_dir_path, 'freq.txt')
    try:
        freq = np.loadtxt(freq_path)
        MAX_FREQ = 94.0
        # normalization
        freq_norm = freq / MAX_FREQ
        print(f"Loaded frequencies from {freq_path}. Range: {freq.min()} - {freq.max()} GHz")

    except Exception as e:
        print(f"ERROR: Failed to load or process '{freq_path}'.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)  # 退出程序

    with open(gateway_pos_path) as f_loader:
        gateway_info = yaml.safe_load(f_loader)
        
         # [5.0, 0.26, 0]
        gateway_pos = gateway_info['gateway1']['position']
        gateway_quaternion = gateway_info['gateway1']['orientation']

    for image_idx, image_name in enumerate(spt_names):

        qvec = np.array(gateway_quaternion)
        # torch.Size([3, 3])
        rotation_matrix = torch.from_numpy(Rotation.from_quat(qvec).as_matrix()).float()

        tvec_rx = torch.from_numpy(np.array(gateway_pos)).float()         # torch.Size([3])

        tvec_tx = torch.from_numpy(np.array(tx_pos[image_idx])).float()   # torch.Size([3])

        # extr.name: '00001.jpg', os.path.basename(extr.name): '00001.jpg'
        image_path = os.path.join(spectrum_dir, os.path.basename(image_name))

        # os.path.basename(image_path): '00001.jpg', image_name: '00001'
        image_name_t = os.path.basename(image_path).split(".")[0]

        # The imread function handles opening and closing the file internally, 
        #       so once the function call is complete, the file is automatically closed.
        image = imageio.imread(image_path).astype(np.float32) / 255.0

        height = image.shape[0]
        width  = image.shape[1]

        resized_image = torch.from_numpy(np.array(image)).float()   # torch.Size([90, 360])

        # resized_image = resized_image.unsqueeze(dim=-1).permute(2, 0, 1).repeat(3, 1, 1)
        freq = float(freq_norm[image_idx % len(freq_norm)])

        spec_info = SpectrumInfo(R=rotation_matrix, 
                                #  T_rx=tvec_tx,
                                #  T_tx=tvec_rx,
                                 T_rx=tvec_rx,
                                 T_tx=tvec_tx,
                                 spectrum=resized_image, 
                                 spectrum_path=image_path, 
                                 spectrum_name=image_name_t,
                                 height=height,
                                 width=width,
                                 freq = freq)
        
        data_infos.append(spec_info)

    sys.stdout.write('\n')

    return data_infos


def getNorm_3d(specs_info, scale):

    # scale = 2.0
    # len(cam_centers): 301, each one is an tensor of torch.Size([3])
    def get_center_and_diag(gatewa_pos_t, cam_center):

        gatewa_pos_t = gatewa_pos_t.unsqueeze(1)  # torch.Size([3, 1])

        # torch.Size([3, 6124])
        cam_center = torch.stack(cam_center, dim=1)

        dists = torch.norm(cam_center - gatewa_pos_t, dim=0)
        radius = torch.max(dists) * scale

        deviations = cam_center - gatewa_pos_t

        # Clone to avoid modifying original
        positive_deviations = deviations.clone()
        negative_deviations = deviations.clone()

        positive_deviations[positive_deviations < 0] = 0
        negative_deviations[negative_deviations > 0] = 0

        max_positive = positive_deviations.max(dim=1).values
        max_negative = negative_deviations.min(dim=1).values.abs()

        epsilon = 1e-6
        max_positive[max_positive < epsilon] = 1.0
        max_negative[max_negative < epsilon] = 1.0

        return {"max_positive": max_positive * scale, "max_negative": max_negative * scale}, radius.item()
    
    cam_centers = []
    gatewa_pos  = specs_info[0].T_rx   # gateway location, torch.Size([3])
    # gatewa_pos  = specs_info[0].T_tx   # gateway location, torch.Size([3])


    for cam in specs_info:
        cam_centers.append(cam.T_tx)  # TX location, torch.Size([3])
        # cam_centers.append(cam.T_rx)  # TX location, torch.Size([3])

    # diagonal: {'max_positive': tensor([1.2000, 0.2820, 2.0412]), 
    #            'max_negative': tensor([7.0716, 1.3428, 1.2000])}

    #           {'max_positive': tensor([2.0000, 0.4700, 3.4020]), 
    #            'max_negative': tensor([11.7860,  2.2380,  2.0000])}

    # radius: 7.3305511474609375   12.217584609985352
    diagonal, radius = get_center_and_diag(gatewa_pos, cam_centers)

    translate = -gatewa_pos   # tensor([-5.0000, -0.2600, -0.0000])  

    return {"translate": translate, "radius": radius, "extent": diagonal}


def obtain_train_test_idx(args_model, len_list):

    path     = args_model.source_path
    llffhold = args_model.llffhold

    llffhold_flag = args_model.llffhold_flag

    train_index = os.path.join(path, args_model.train_index_path)    # train_index_knn.txt
    test_index  = os.path.join(path, args_model.test_index_path)     # test_index_knn.txt

    if llffhold_flag:
        
        print("\nUSING LLFFHOLD INDEX FILE\n")
        # the index start from 0
        i_test = np.arange(int(len_list))[:: llffhold]
        i_train = np.array([j for j in np.arange(int(len_list)) if (j not in i_test)])

    elif "knn" in train_index:
        print("\nUSING KNN INDEX FILE\n")
        # the index start from 0
        i_train = np.loadtxt(train_index, dtype=int)
        i_test  = np.loadtxt(test_index,  dtype=int)
        
    else:
        print("\nUSING RANDOM INDEX FILE\n")
        # the index start from 1, if no - 1, since 00001.png is the first image
        i_train = np.loadtxt(train_index, dtype=int)
        i_test  = np.loadtxt(test_index,  dtype=int)

    return i_train, i_test


def readRFSceneInfo(args_model):

    path         = args_model.source_path
    eval         = args_model.eval
    camera_scale     = args_model.camera_scale
    voxel_size_scale = args_model.voxel_size_scale

    ratio_train = args_model.ratio_train

    spectrums_infos_unsorted = readSpectrumImage(path)

    train_index_path = os.path.join(path, args_model.train_index_path)    
    test_index_path  = os.path.join(path, args_model.test_index_path)  

    split_dataset_train_v2(path, train_index_path, test_index_path, ratio=ratio_train)

    i_train, i_test = obtain_train_test_idx(args_model, len(spectrums_infos_unsorted))
    
    spectrums_infos = sorted(spectrums_infos_unsorted.copy(), key = lambda x : int(x.spectrum_name))

    if eval:
        train_infos = [spectrums_infos[idx] for idx in i_train]
        test_infos  = [spectrums_infos[idx] for idx in i_test]

    else:
        train_infos = spectrums_infos
        test_infos = []

    nerf_normalization = getNorm_3d(spectrums_infos, camera_scale)

    ply_path = os.path.join(path, "points3D.ply")
    if (not os.path.exists(ply_path)) or (args_model.gene_init_point):

        receiver_pos = spectrums_infos[0].T_rx.numpy()

        avg_freq_ghz = np.mean([s.freq for s in spectrums_infos])
        avg_freq = avg_freq_ghz * 1e9

        cube_size = round((3.00e8 / avg_freq) * voxel_size_scale, 2)

        num_pos = init_ply_v2(ply_path, receiver_pos, nerf_normalization["extent"], cube_size)

        print(f"\nRandomlize point coluds. Cube size: {cube_size} meters, Number of points: {num_pos}\n")

    try:
        pcd = fetch_init_ply(ply_path)

    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd, 
                           train_spectrums=train_infos,
                           test_spectrums=test_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    
    return scene_info


def fetch_init_ply(path):
    
    plydata = PlyData.read(path)

    vertices = plydata['vertex']
    # np.vstack 将 x, y, z 三个 (N,) 的数组垂直堆叠成 (3, N)
    # .T 转置操作将其变为 (N, 3)，符合 PyTorch 的标准输入格式
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    # 在该项目中是全零填充的占位数据，仅为了符合 PLY 文件格式标准，不参与核心的射频场训练或渲染计算
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    # 不需要点云中的颜色信息
    return BasicPointCloud(points=positions, attris=None, normals=normals)

# 这一部分点云的生成需要更改（下列两个函数）
# 根据物理计算（波长和场景大小）生成了一个规则的网格点云来初始化场景
# 这是一个“体素化空间采样”的过程
def init_ply_v2(ply_path, receiver_pos, camera_extent, cube_size):

    dtype = [('x', 'f4'),  ('y', 'f4'),  ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')
            ]
    xyz = generate_cube_coordinates(receiver_pos, camera_extent, cube_size)

    #  Initializes an array of normals with the same shape as xyz but filled with zeros
    normals = np.zeros_like(xyz)

    # Creates an empty structured array for the elements of the PLY file, with the specified dtype.
    elements = np.empty(xyz.shape[0], dtype=dtype)

    attributes = np.concatenate((xyz, normals), axis=1)

    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')

    ply_data = PlyData([vertex_element])

    ply_data.write(ply_path)

    return xyz.shape[0]


def generate_cube_coordinates(receiver_pos, camera_extent, cube_size):
    # Define the 3D space boundaries
    x_min = receiver_pos[0] - camera_extent["max_negative"][0].item()
    x_max = receiver_pos[0] + camera_extent["max_positive"][0].item()

    y_min = receiver_pos[1] - camera_extent["max_negative"][1].item()
    y_max = receiver_pos[1] + camera_extent["max_positive"][1].item()
    
    z_min = receiver_pos[2] - camera_extent["max_negative"][2].item()
    z_max = receiver_pos[2] + camera_extent["max_positive"][2].item()

    num_cubes_x = int(np.ceil((x_max - x_min) / cube_size))
    num_cubes_y = int(np.ceil((y_max - y_min) / cube_size))
    num_cubes_z = int(np.ceil((z_max - z_min) / cube_size))

    x_coords = np.linspace(x_min, x_max, num_cubes_x) if num_cubes_x > 1 else np.array([(x_min + x_max) / 2])
    y_coords = np.linspace(y_min, y_max, num_cubes_y) if num_cubes_y > 1 else np.array([(y_min + y_max) / 2])
    z_coords = np.linspace(z_min, z_max, num_cubes_z) if num_cubes_z > 1 else np.array([(z_min + z_max) / 2])

    # Create a grid of points
    x_grid, y_grid, z_grid = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    # [N, 3]
    cube_points = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T

    return cube_points




