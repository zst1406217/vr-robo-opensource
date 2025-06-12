from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import rpyc
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
from PIL import Image

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import ObservationTermCfg

import socket
import pickle
from multiprocessing import Process, Queue
import numpy as np
import os
import atexit
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import threading
import timm
import torchvision

HEAD_POS = [0.332, 0.0, 0.00]

@torch.jit.script
def euler_to_quaternion(euler_angles):
    cy = torch.cos(euler_angles[:, 2] * 0.5)
    sy = torch.sin(euler_angles[:, 2] * 0.5)
    cp = torch.cos(euler_angles[:, 1] * 0.5)
    sp = torch.sin(euler_angles[:, 1] * 0.5)
    cr = torch.cos(euler_angles[:, 0] * 0.5)
    sr = torch.sin(euler_angles[:, 0] * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack((w, x, y, z), dim=-1)

class GSServer:
    def __init__(self, host='localhost', port=12345, time_out=10):
        self.host = host
        self.port = port
        self.time_out = time_out
        # self.data = None
        # self.last_data = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()
        
    def init_data(self, env_num):
        self.data = np.zeros((env_num, 3*180*320))
        self.last_data = np.zeros((env_num, 3*180*320))
        self.latency = np.random.randint(0, 2, size=(env_num,1))
        self.env_num = env_num

    def receive_data(self, host='localhost', port=12345):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        s.listen(1)
        conn, addr = s.accept()
        conn.settimeout(self.time_out)  # Set timeout for receiving data
        data = b""
        try:
            while True:
                packet = conn.recv(40960000)
                if not packet:
                    break
                data += packet
        except socket.timeout:
            print("No new tensor received for 10 seconds, terminating connection.")
        finally:
            conn.close()
        pickle_data = pickle.loads(data)
        return pickle_data

    def start(self):
        atexit.register(self.close)
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        
    def close(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def run(self):
        while self.running:
            data = self.receive_data(self.host, self.port)
            with self.lock:
                self.last_data = self.data
                self.data = data

    def get_data(self):
        with self.lock:
            is_start = (self.last_data == 0).all(axis=1).reshape(-1, 1)
            latency = self.latency
            return (latency * self.last_data + (1 - latency) * self.data)*(1-is_start)+self.data*is_start
        
    def reset(self, env_ids):
        if env_ids is None:
            return
        else:
            env_ids=env_ids.cpu().numpy()
            self.data[env_ids] = np.zeros((len(env_ids), 3*180*320))
            self.last_data[env_ids] = np.zeros((len(env_ids), 3*180*320))
            self.latency[env_ids] = np.random.randint(0, 2, size=(len(env_ids), 1))
    
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv
import threading
import random

def base_lin_vel_zero(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.zeros(env.num_envs, 3, device=env.device)

def base_pos_z_e(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_e = asset.data.root_pos_w - env.scene.env_origins
    return pos_e[:, 2].unsqueeze(-1)

def standing_velocity_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    commands=env.command_manager.get_command(command_name)
    commands[:, :2] *= (torch.norm(commands[:, :2], dim=1) > 0.1).unsqueeze(1)
    commands[:, 2] *= (torch.abs(commands[:, 2]) > 0.1)
    return commands

def zero_velocity_commands(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    commands=torch.zeros(env.num_envs, 3, device=env.device)
    return commands

def goal_pos(env: ManagerBasedEnv, base_height, command_name, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The goal position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    rgb_commands=env.command_manager.get_command(command_name)
    goal_red = env.scene["cone_red"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_green = env.scene["cone_green"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_blue = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_tensor = torch.stack([goal_red, goal_green, goal_blue], dim=1)
    goal_tensor[:, :, 2] += base_height
    selected_goal = goal_tensor * rgb_commands.unsqueeze(2)
    pos_goal = selected_goal.sum(dim=1)
    
    return pos_goal

def goal_pos_multi(env: ManagerBasedEnv, base_height, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The goal position in the environment frame."""
    goal_red = env.scene["cone_red"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_green = env.scene["cone_green"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_blue = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_tensor = torch.stack([goal_red, goal_green, goal_blue], dim=1)
    goal_tensor[:, :, 2] += base_height
    pos_goal = goal_tensor.reshape(env.num_envs, -1)
    
    return pos_goal

def head_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pos_r = asset.data.root_pos_w - env.scene.env_origins
    quat_r = asset.data.root_quat_w
    head_pos=torch.tensor(HEAD_POS, device=env.device).repeat(env.num_envs, 1)
    head_pos_r = pos_r + math_utils.quat_apply(quat_r, head_pos)
    return head_pos_r

class gs_image_feature(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)
        self.conn = rpyc.connect('localhost', 18861)
        self.image_server = GSServer()
        self.image_server.start()
        self.image_server.init_data(env.num_envs)
        # self.cnn_model = getattr(models, "resnet18")(pretrained=True)
        # self.cnn_model.fc = nn.Identity()  # Remove the final fully connected layer
        
        # self.cnn_model = getattr(models, "mobilenet_v3_small")(pretrained=True)
        # self.cnn_model.avgpool = nn.AdaptiveMaxPool2d(1)  # Remove the final average pooling layer
        # self.cnn_model.classifier = nn.Identity()  # Remove the final fully connected layer

        self.cnn_model=timm.create_model('vit_tiny_patch16_224', pretrained=True)
        self.cnn_model.head = nn.Identity()  # Remove the final fully connected layer
        
        self.cnn_model.to(env.device)
        self.cnn_model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0))], p=0.8),
            transforms.RandomApply([transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.1)], p=0.1),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000]),
        ])
        self.camera_pos_noise_scale = torch.tensor([0.01, 0.01, 0.01], device=env.device)
        self.camera_rot_noise_scale = torch.tensor([1., 1., 2.], device=env.device)
        print("GS Server Initialized")
        self.save_count=0
        
    def reset(self, env_ids: torch.Tensor | None = None):
        self.image_server.reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        camera_pos: list = None,
        camera_rot: list = None,
        asset_offset_pos: list = None,
        asset_offset_rot: list = None,
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        pos_r = asset.data.root_pos_w - env.scene.env_origins
        quat_r = asset.data.root_quat_w
        camera_pos=torch.tensor(camera_pos, device=env.device).repeat(env.num_envs, 1)
        camera_pos += (2*torch.rand_like(camera_pos)-1) * self.camera_pos_noise_scale
        camera_rot=torch.tensor(camera_rot, device=env.device).repeat(env.num_envs, 1)
        camera_rot += (2*torch.rand_like(camera_rot)-1) * self.camera_rot_noise_scale
        camera_rot = torch.deg2rad(camera_rot)
        camera_rot = euler_to_quaternion(camera_rot)
        
        asset_offset_pos=torch.tensor(asset_offset_pos, device=env.device).repeat(env.num_envs, 1)
        asset_offset_rot=torch.tensor(asset_offset_rot, device=env.device).repeat(env.num_envs, 1)
        camera_pos_r = pos_r + math_utils.quat_apply(quat_r, camera_pos) - asset_offset_pos
        camera_rot_r = math_utils.quat_mul(quat_r, camera_rot)
        camera_rot_r = math_utils.convert_camera_frame_orientation_convention(camera_rot_r, "world", "ros")
        # TODO: add asset_offset_rot
        # red_cone=torch.tensor([3.2-0.53, -0.43, -0.03+0.36], device=env.device) - asset_offset_pos
        # blue_cone=torch.tensor([3.2-1.6, -1.59, -0.03+0.03], device=env.device) - asset_offset_pos
        # green_cone=torch.tensor([3.2-1.8, -1.8, -0.03+0.03], device=env.device) - asset_offset_pos
        red_cone = env.scene["cone_red"].data.object_pos_w.squeeze(1) - env.scene.env_origins - asset_offset_pos
        green_cone = env.scene["cone_green"].data.object_pos_w.squeeze(1) - env.scene.env_origins - asset_offset_pos
        blue_cone = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - env.scene.env_origins - asset_offset_pos
        self.conn.root.render(camera_pos_r, camera_rot_r, red_cone, green_cone, blue_cone)
        images=torch.tensor(self.image_server.get_data()).float().to(env.device)
        # print(images.shape)
        images=images.reshape(env.num_envs, 3, 180, 320)
        if images is not None:
            ndarr = (images[0]).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            concat_im = Image.fromarray(ndarr)
            concat_im.save('obs_test.png')
            # concat_im.save('./obs_record/obs_test'+str(self.save_count)+'.png')
            # self.save_count+=1
        
        images=self.preprocess(images/255)
        with torch.inference_mode():
            image_feature = self.cnn_model(images)
        # print(image_feature.shape)

        return image_feature
    
    
class gs_image_ImageNet(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)
        self.conn = rpyc.connect('localhost', 18861)
        self.image_server = GSServer()
        self.image_server.start()
        self.image_server.init_data(env.num_envs)
        # self.cnn_model = getattr(models, "resnet18")(pretrained=True)
        # self.cnn_model.fc = nn.Identity()  # Remove the final fully connected layer
        # self.cnn_model = getattr(models, "mobilenet_v3_small")(pretrained=True)
        self.cnn_model=timm.create_model('vit_tiny_patch16_224', pretrained=True)
        # self.cnn_model.avgpool = nn.Identity()  # Remove the final average pooling layer
        # self.cnn_model.avgpool = nn.AdaptiveMaxPool2d(1)  # Remove the final average pooling layer
        # self.cnn_model.classifier = nn.Identity()  # Remove the final fully connected layer
        self.cnn_model.head = nn.Identity()  # Remove the final fully connected layer
        self.cnn_model.to(env.device)
        self.cnn_model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0))], p=0.8),
            transforms.RandomApply([transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.1)], p=0.1),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000]),
        ])
        self.camera_pos_noise_scale = torch.tensor([0.01, 0.01, 0.01], device=env.device)
        self.camera_rot_noise_scale = torch.tensor([1., 1., 2.], device=env.device)
        print("GS Server Initialized")
        self.save_count=0
        self.imagenet_dataset = ImageNetDataset()
        
    def reset(self, env_ids: torch.Tensor | None = None):
        self.image_server.reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        camera_pos: list = None,
        camera_rot: list = None,
        asset_offset_pos: list = None,
        asset_offset_rot: list = None,
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        pos_r = asset.data.root_pos_w - env.scene.env_origins
        quat_r = asset.data.root_quat_w
        camera_pos=torch.tensor(camera_pos, device=env.device).repeat(env.num_envs, 1)
        camera_pos += (2*torch.rand_like(camera_pos)-1) * self.camera_pos_noise_scale
        camera_rot=torch.tensor(camera_rot, device=env.device).repeat(env.num_envs, 1)
        camera_rot += (2*torch.rand_like(camera_rot)-1) * self.camera_rot_noise_scale
        camera_rot = torch.deg2rad(camera_rot)
        camera_rot = euler_to_quaternion(camera_rot)
        
        asset_offset_pos=torch.tensor(asset_offset_pos, device=env.device).repeat(env.num_envs, 1)
        asset_offset_rot=torch.tensor(asset_offset_rot, device=env.device).repeat(env.num_envs, 1)
        camera_pos_r = pos_r + math_utils.quat_apply(quat_r, camera_pos) - asset_offset_pos
        camera_rot_r = math_utils.quat_mul(quat_r, camera_rot)
        camera_rot_r = math_utils.convert_camera_frame_orientation_convention(camera_rot_r, "world", "ros")
        # TODO: add asset_offset_rot
        # red_cone=torch.tensor([3.2-0.53, -0.43, -0.03+0.36], device=env.device) - asset_offset_pos
        # blue_cone=torch.tensor([3.2-1.6, -1.59, -0.03+0.03], device=env.device) - asset_offset_pos
        # green_cone=torch.tensor([3.2-1.8, -1.8, -0.03+0.03], device=env.device) - asset_offset_pos
        red_cone = env.scene["cone_red"].data.object_pos_w.squeeze(1) - env.scene.env_origins - asset_offset_pos
        green_cone = env.scene["cone_green"].data.object_pos_w.squeeze(1) - env.scene.env_origins - asset_offset_pos
        blue_cone = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - env.scene.env_origins - asset_offset_pos
        
        self.conn.root.render(camera_pos_r, camera_rot_r, red_cone, green_cone, blue_cone)
        images=torch.tensor(self.image_server.get_data()).float().to(env.device)
        # print(images.shape)
        images=images.reshape(env.num_envs, 3, 180, 320)
        
        for i in range(env.num_envs):
            image=images[i].permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # --------------------------
            # Red cone mask
            # Red is split in the HSV space near the edges, so we use two ranges.
            lower_red1 = np.array([0, 70, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 70, 50])
            upper_red2 = np.array([180, 255, 255])

            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            # --------------------------
            # Blue cone mask
            lower_blue = np.array([100, 150, 0])
            upper_blue = np.array([140, 255, 255])
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

            # --------------------------
            # Green cone mask
            lower_green = np.array([40, 70, 70])
            upper_green = np.array([80, 255, 255])
            mask_green = cv2.inRange(hsv, lower_green, upper_green)

            # Optional: refine the masks using morphological operations
            kernel = np.ones((3, 3), np.uint8)
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=2)
            mask_red = cv2.dilate(mask_red, kernel, iterations=1)

            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=2)
            mask_blue = cv2.dilate(mask_blue, kernel, iterations=1)

            mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=2)
            mask_green = cv2.dilate(mask_green, kernel, iterations=1)
            
            mask=(mask_red+mask_blue+mask_green)>0
            imagenet_data = self.imagenet_dataset.random_image()
            # print(imagenet_data.shape)
            
            new_image = image*mask[:, :, np.newaxis] + imagenet_data*(1-mask[:, :, np.newaxis])
            new_image = new_image[..., [2, 1, 0]]
            images[i] = torch.tensor(new_image).permute(2, 0, 1).to(env.device)
            
        if images is not None:
            ndarr = (images[0]).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            concat_im = Image.fromarray(ndarr)
            concat_im.save('obs_test.png')
            # concat_im.save('./obs_record/obs_test'+str(self.save_count)+'.png')
            # self.save_count+=1
        
        images=self.preprocess(images/255)
        with torch.inference_mode():
            image_feature = self.cnn_model(images)
        # print(image_feature.shape)

        return image_feature


class gs_image_int(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)
        self.conn = rpyc.connect('localhost', 18861)
        self.image_server = GSServer()
        self.image_server.start()
        self.image_server.init_data(env.num_envs)
        self.camera_pos_noise_scale = torch.tensor([0.01, 0.01, 0.01], device=env.device)
        self.camera_rot_noise_scale = torch.tensor([1., 1., 2.], device=env.device)
        self.preprocess = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0))], p=0.8),
        ])
        print("GS Server Initialized")

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        camera_pos: list = None,
        camera_rot: list = None,
        asset_offset_pos: list = None,
        asset_offset_rot: list = None,
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        pos_r = asset.data.root_pos_w - env.scene.env_origins
        quat_r = asset.data.root_quat_w
        camera_pos=torch.tensor(camera_pos, device=env.device).repeat(env.num_envs, 1)
        camera_pos += (2*torch.rand_like(camera_pos)-1) * self.camera_pos_noise_scale
        camera_rot=torch.tensor(camera_rot, device=env.device).repeat(env.num_envs, 1)
        camera_rot += (2*torch.rand_like(camera_rot)-1) * self.camera_rot_noise_scale
        camera_rot = torch.deg2rad(camera_rot)
        camera_rot = euler_to_quaternion(camera_rot)
        
        asset_offset_pos=torch.tensor(asset_offset_pos, device=env.device).repeat(env.num_envs, 1)
        asset_offset_rot=torch.tensor(asset_offset_rot, device=env.device).repeat(env.num_envs, 1)
        camera_pos_r = pos_r + math_utils.quat_apply(quat_r, camera_pos) - asset_offset_pos
        camera_rot_r = math_utils.quat_mul(quat_r, camera_rot)
        camera_rot_r = math_utils.convert_camera_frame_orientation_convention(camera_rot_r, "world", "ros")
        # TODO: add asset_offset_rot
        self.conn.root.render(camera_pos_r, camera_rot_r)
        images=torch.tensor(self.image_server.get_data()).to(env.device)
        images=images.reshape(env.num_envs, 3, 180, 320)
        images=self.preprocess(images)
        
        if images is not None:
            ndarr = (images[0]).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            # print(ndarr)
            concat_im = Image.fromarray(ndarr)
            concat_im.save('obs_test.png')   

        images=images.reshape(env.num_envs, -1).to(torch.uint8)

        return images


def rgb_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    commands=env.command_manager.get_command(command_name)
    return commands


cnn_model=timm.create_model('vit_tiny_patch16_224', pretrained=True)
cnn_model.head = nn.Identity()  # Remove the final fully connected layer
cnn_model.cuda()
cnn_model.eval()

import cv2
from PIL import Image
import numpy as np

def custom_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    global cnn_model
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    preprocess = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0))], p=0.8),
        transforms.RandomApply([transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.1)], p=0.1),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000]),
    ])

    # obtain the input image
    images = sensor.data.output[data_type]

    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # rgb/depth image normalization
    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0

    images[images == float("inf")] = 5
    images[images > 5] = 5
    images[images < 0] = 0
    images=images/5
    images=images.permute(0, 3, 1, 2)
    torch.set_printoptions(profile="full")
    ndarr = (images[0]*255).permute(1,2,0).to("cpu", torch.uint8).numpy()
    cv2.imwrite('obs_test.png', ndarr)
    # im=Image.fromarray(ndarr)
    # im.save('obs_test.png')

    image_feature = images.reshape(env.num_envs, -1)

    # images=preprocess(images/255)
    # with torch.inference_mode():
    #     image_feature = cnn_model(images)

    return image_feature

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir="/home/zhust/data/ILSVRC2012_img_val", transform=torchvision.transforms.Resize((320, 180))):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.JPEG')]
        print("ImageNet Dataset Initialized")
        print("Total images: ", len(self.image_files))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image
    
    def random_image(self):
        idx = random.randint(0, len(self.image_files) - 1)
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.resize(image, (320, 180))

        return image