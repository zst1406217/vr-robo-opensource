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

import torch
from scene import Scene
import os
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
from scene.app_model import AppModel
import copy
from collections import deque
from scene.cameras import MiniCam
import time
from utils.graphics_utils import getProjectionMatrix
from einops import einsum
from pytorch3d.transforms import matrix_to_quaternion
import einops
from e3nn import o3


def render_set(gaussians, pipeline, background):
    QW, QX, QY, QZ = 0.95804870795258756, -0.28124543695451032, 0.04713193995873477, 0.028675034757964676
    TX, TY, TZ = -2.5344067727108173, -2.6614219158234897, 3.6379965648091055

    QW, QX, QY, QZ = 0.5, -0.5, 0.5, -0.5
    TX, TY, TZ = -2, 0.6, 0.5
    
    T_sim = np.eye(4)
    T_sim[:3, :3] = o3d.geometry.TriangleMesh.get_rotation_matrix_from_quaternion([QW, QX, QY, QZ])
    T_sim[:3, 3] = [TX, TY, TZ]

    T_sim=np.linalg.inv(T_sim)

    width = 320
    height = 180
    fovx = 1.5701
    fovy = 1.0260
    znear = 0.01
    zfar = 100.0

    # world_view_transform = torch.tensor(T_colmap, dtype=torch.float, device="cuda").transpose(0,1)
    world_view_transform = torch.tensor(T_sim, dtype=torch.float, device="cuda").transpose(0,1)
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1).cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    
    custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
    for i in range(10):
        torch.cuda.synchronize(); t0 = time.time()
        render_pkg = render(custom_cam, gaussians, pipeline, background)
        torch.cuda.synchronize(); t1 = time.time()
        print(f"Render time: {t1-t0:.4f} seconds")

        rendering = render_pkg["render"]
    torchvision.utils.save_image(rendering, "./test.png")


def render_sets(dataset : ModelParams, pipeline : PipelineParams):

    with torch.no_grad():
        gaussians_env = GaussianModel(3)
        gaussians_red = GaussianModel(3)
        gaussians_green = GaussianModel(3)
        gaussians_blue = GaussianModel(3)


        gaussians_env.load_ply(os.path.join("output/env",
                                            "point_cloud.ply"))
        gaussians_red.load_ply(os.path.join("output/red",
                                            "point_cloud.ply"))
        gaussians_green.load_ply(os.path.join("output/green",
                                            "point_cloud.ply"))
        gaussians_blue.load_ply(os.path.join("output/blue",
                                            "point_cloud.ply"))

        # env
        T_env = np.array([[0.469, -0.032, 0.082, -1.595],
                [-0.079, -0.352, 0.313, -1.047],
                [0.039, -0.322, -0.351, 1.341],
                [0.000, 0.000, 0.000, 1.000]])
        scale_env = 0.477609

        # red
        bounding_box_red = np.array([
            [0.993446648121, -0.030691670254, 0.110096260905, 0.776266396046],
            [-0.002712320071, 0.956668972969, 0.291165977716, 1.652374744415],
            [-0.114261977375, -0.289556652308, 0.950316369534, 1.803330898285],
            [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
        ])
        width_red = np.array([0.87486160, 0.86398125, 1.08366120])
        T_red = np.array([[-0.008, 0.247, -0.077, -0.261],
                [0.257, 0.001, -0.024, -0.159],
                [-0.023, -0.077, -0.246, 0.721],
                [0.000, 0.000, 0.000, 1.000]])
        scale_red = 0.258537
        
        # green
        bounding_box_green = np.array([
            [0.927658736706, -0.328061193228, 0.178395494819, 0.267540454865],
            [0.351592063904, 0.928269267082, -0.121237739921, -0.996843576431],
            [-0.125825762749, 0.175189733505, 0.976460814476, 3.083600521088],
            [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
        ])
        width_green = np.array([1.10369134, 1.12264895, 1.52788544])
        T_green = np.array([[0.055, -0.175, -0.033, -0.085],
                    [-0.175, -0.060, 0.025, -0.091],
                    [-0.034, 0.023, -0.182, 0.735],
                    [0.000, 0.000, 0.000, 1.000]])
        scale_green = 0.186556
        
        # blue
        bounding_box_blue = np.array([
            [0.669211506844, -0.716110467911, 0.198347613215, -0.207296669483],
            [0.724392294884, 0.688192307949, 0.040584608912, 1.151019573212],
            [-0.165564328432, 0.116521820426, 0.979291141033, 3.180432558060],
            [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
        ])
        width_blue = np.array([0.89889467, 0.90830016, 1.14793587])
        T_blue = np.array([[-0.166, -0.177, 0.042, 0.035],
                [-0.175, 0.171, 0.030, -0.326],
                [-0.051, -0.010, -0.241, 0.908],
                [0.000, 0.000, 0.000, 1.000]])
        scale_blue = 0.246493
        
        gaussians_red = filter_gaussians_within_bounding_box(gaussians_red, bounding_box_red, width_red)
        gaussians_green = filter_gaussians_within_bounding_box(gaussians_green, bounding_box_green, width_green)
        gaussians_blue = filter_gaussians_within_bounding_box(gaussians_blue, bounding_box_blue, width_blue)
        
        gaussians_env=transform_gaussians(gaussians_env, T_env, scale_env)
        gaussians_red=transform_gaussians(gaussians_red, T_red, scale_red)
        gaussians_green=transform_gaussians(gaussians_green, T_green, scale_green)
        gaussians_blue=transform_gaussians(gaussians_blue, T_blue, scale_blue)

        red_XYZ=[-0.2, 0.6, 0.01]
        gaussians_red._xyz += torch.tensor(red_XYZ, device="cuda")
        green_XYZ=[-0.5, 0.6, 0.0]
        gaussians_green._xyz += torch.tensor(green_XYZ, device="cuda")
        blue_XYZ=[-0.5, 0.2, 0.0]
        gaussians_blue._xyz += torch.tensor(blue_XYZ, device="cuda")

        start_indices = [
            0,
            gaussians_env._xyz.shape[0],
            gaussians_env._xyz.shape[0] + gaussians_red._xyz.shape[0],
            gaussians_env._xyz.shape[0] + gaussians_red._xyz.shape[0] + gaussians_green._xyz.shape[0]
        ]

        gaussians_env._xyz = torch.cat([gaussians_env._xyz, gaussians_red._xyz, gaussians_green._xyz, gaussians_blue._xyz], dim=0)
        gaussians_env._features_dc = torch.cat([gaussians_env._features_dc, gaussians_red._features_dc, gaussians_green._features_dc, gaussians_blue._features_dc], dim=0)
        gaussians_env._features_rest = torch.cat([gaussians_env._features_rest, gaussians_red._features_rest, gaussians_green._features_rest, gaussians_blue._features_rest], dim=0)
        gaussians_env._scaling = torch.cat([gaussians_env._scaling, gaussians_red._scaling, gaussians_green._scaling, gaussians_blue._scaling], dim=0)
        gaussians_env._rotation = torch.cat([gaussians_env._rotation, gaussians_red._rotation, gaussians_green._rotation, gaussians_blue._rotation], dim=0)
        gaussians_env._opacity = torch.cat([gaussians_env._opacity, gaussians_red._opacity, gaussians_green._opacity, gaussians_blue._opacity], dim=0)
        
        print(start_indices)

        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(gaussians_env, pipeline, background)


def transform_gaussians(gaussians, T, scale):
    ones = torch.ones((gaussians._xyz.shape[0], 1), device="cuda")
    xyz_h = torch.cat([gaussians._xyz, ones], dim=1)
    transformed_xyz_h = torch.matmul(torch.tensor(T, device="cuda").float(), xyz_h.transpose(0, 1)).transpose(0, 1)
    gaussians._xyz = transformed_xyz_h[:, :3]
    gaussians._scaling += np.log(scale)
    rotation_matrix = gaussians.get_rotation_matrix()
    new_rotation = torch.matmul(torch.tensor(T[:3, :3]/scale, device="cuda").float(), rotation_matrix)
    gaussians._rotation = matrix_to_quaternion(new_rotation)
    shs_feat = gaussians._features_rest.cpu().double()
    shs_feat = transform_shs(shs_feat, T[:3, :3]/scale)
    gaussians._features_rest = shs_feat.float().cuda()
    return gaussians


def transform_shs(shs_feat, rotation_matrix):

    ## rotate shs
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) # switch axes: yzx -> xyz
    permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(torch.from_numpy(permuted_rotation_matrix))
    
    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2])

    #rotation of the shs features
    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einsum(
            D_1,
            one_degree_shs,
            "... i j, ... j -> ... i",
        )
    one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 0:3] = one_degree_shs

    two_degree_shs = shs_feat[:, 3:8]
    two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    two_degree_shs = einsum(
            D_2,
            two_degree_shs,
            "... i j, ... j -> ... i",
        )
    two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 3:8] = two_degree_shs

    three_degree_shs = shs_feat[:, 8:15]
    three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    three_degree_shs = einsum(
            D_3,
            three_degree_shs,
            "... i j, ... j -> ... i",
        )
    three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 8:15] = three_degree_shs

    return shs_feat


def filter_gaussians_within_bounding_box(gaussians, bounding_box, width):
    bounding_box = np.linalg.inv(bounding_box)

    # Extract points within the bounding box
    ones = torch.ones((gaussians._xyz.shape[0], 1), device="cuda")
    xyz_h = torch.cat([gaussians._xyz, ones], dim=1)
    transformed_xyz_h = torch.matmul(torch.tensor(bounding_box, device="cuda").float(), xyz_h.transpose(0, 1)).transpose(0, 1)
    transformed_xyz = transformed_xyz_h[:, :3]

    mask = (
        (transformed_xyz[:, 0] >= -width[0]/2) & (transformed_xyz[:, 0] <= width[0]/2) &
        (transformed_xyz[:, 1] >= -width[1]/2) & (transformed_xyz[:, 1] <= width[1]/2) &
        (transformed_xyz[:, 2] >= -width[2]/2) & (transformed_xyz[:, 2] <= width[2]/2)
    )

    gaussians._xyz = gaussians._xyz[mask]
    gaussians._features_dc = gaussians._features_dc[mask]
    gaussians._features_rest = gaussians._features_rest[mask]
    gaussians._scaling = gaussians._scaling[mask]
    gaussians._rotation = gaussians._rotation[mask]
    gaussians._opacity = gaussians._opacity[mask]

    return gaussians


if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--use_depth_filter", action="store_true")

    args = get_combined_args(parser)
    print("Rendering")

    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    render_sets(model.extract(args), pipeline.extract(args))