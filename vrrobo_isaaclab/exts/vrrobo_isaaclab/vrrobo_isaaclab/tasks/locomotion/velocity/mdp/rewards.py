from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.assets import Articulation
from collections.abc import Sequence
import omni.isaac.lab.utils.math as math_utils

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import RewardTermCfg

HEAD_POS = [0.332, 0.0, 0.00]

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # reward = torch.sum(torch.clip((last_air_time - threshold),max=0) * first_contact, dim=1)
    reward = torch.sum((last_air_time - 0.3) * first_contact, dim=1)+10*torch.sum(torch.clip((threshold-last_air_time) * first_contact,max=0),dim=1)

    # no reward for zero command
    # reward *= torch.logical_or(torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1, torch.abs(env.command_manager.get_command(command_name)[:, 2]) > 0.1)
    return reward

def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def dof_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    dof_error = torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return dof_error

def dof_error_named(env: ManagerBasedRLEnv, joint_names: str | Sequence[str], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ Reward for a given named joints """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, joint_names = asset.find_joints(joint_names)
    dof_error = torch.sum(torch.square(asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]), dim=1)
    return dof_error

def stand_still(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    dof_error = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    
    return dof_error * (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1)\
                     * (torch.abs(env.command_manager.get_command(command_name)[:, 2]) < 0.1)
                    
def stop_dof_vel(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # Penalize dof velocities at zero commands
    return torch.sum(torch.abs(asset.data.joint_vel), dim=1) \
        * (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1) \
        * (torch.abs(env.command_manager.get_command(command_name)[:, 2] < 0.1))
        
def stop_base_lin_vel(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # Penalize dof velocities at zero commands
    return torch.sum(torch.abs(asset.data.root_lin_vel_b), dim=1) \
        * (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1) \
        * (torch.abs(env.command_manager.get_command(command_name)[:, 2] < 0.1))
        
def stop_base_ang_vel(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # Penalize dof velocities at zero commands
    return torch.abs(asset.data.root_ang_vel_b[:, 2]) \
        * (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1) \
        * (torch.abs(env.command_manager.get_command(command_name)[:, 2] < 0.1))

def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque * asset.data.joint_vel), dim=1)

def foot_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    forces = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    # compute the penalty
    forces_reward = torch.square(forces[:,0]+forces[:,2]-forces[:,1]-forces[:,3])
    return forces_reward

def joint_pos_limits_count(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_upper_limits = torch.sum(asset.data.joint_pos > asset.data.soft_joint_pos_limits[..., 1], dim=1)
    out_of_lower_limits = torch.sum(asset.data.joint_pos < asset.data.soft_joint_pos_limits[..., 0], dim=1)
    return out_of_lower_limits+out_of_upper_limits

def reach_goal(env: ManagerBasedRLEnv, base_height, command_name, threshold, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pos_r = asset.data.root_pos_w - env.scene.env_origins
    quat_r = asset.data.root_quat_w

    head_pos=torch.tensor(HEAD_POS, device=env.device).repeat(env.num_envs, 1)
    head_pos_r = pos_r + math_utils.quat_apply(quat_r, head_pos)
    
    rgb_commands=env.command_manager.get_command(command_name)
    goal_red = env.scene["cone_red"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_green = env.scene["cone_green"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_blue = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_tensor = torch.stack([goal_red, goal_green, goal_blue], dim=1)
    goal_tensor[:, :, 2] += base_height
    
    selected_goal = goal_tensor * rgb_commands.unsqueeze(2)
    pos_goal = selected_goal.sum(dim=1)
    
    reward = (torch.norm(head_pos_r - pos_goal, dim=1)<threshold).float()
    
    return reward

class goal_dis(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        self.dis=torch.zeros(env.num_envs, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, base_height, command_name, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        # root_env = asset.data.root_pos_w - env.scene.env_origins
        pos_r = asset.data.root_pos_w - env.scene.env_origins
        quat_r = asset.data.root_quat_w
        
        head_pos=torch.tensor(HEAD_POS, device=env.device).repeat(env.num_envs, 1)
        head_pos_r = pos_r + math_utils.quat_apply(quat_r, head_pos)
        rgb_commands=env.command_manager.get_command(command_name)
        goal_red = env.scene["cone_red"].data.object_pos_w.squeeze(1) - env.scene.env_origins
        goal_green = env.scene["cone_green"].data.object_pos_w.squeeze(1) - env.scene.env_origins
        goal_blue = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - env.scene.env_origins
        goal_tensor = torch.stack([goal_red, goal_green, goal_blue], dim=1)
        goal_tensor[:, :, 2] += base_height
        selected_goal = goal_tensor * rgb_commands.unsqueeze(2)
        pos_goal = selected_goal.sum(dim=1)
        dis=torch.norm(head_pos_r - pos_goal, dim=1)
        start_id=torch.where(self.dis==0.)
        reward = self.dis-dis
        reward[start_id]=0.
        self.dis=dis
        return reward*(dis>0.25)
    
    def reset(self, env_ids: torch.Tensor):
        self.dis[env_ids]=0.
        
        
class goal_dis_z(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        self.dis=torch.zeros(env.num_envs, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, base_height, command_name, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        root_env = asset.data.root_pos_w - env.scene.env_origins
        
        rgb_commands=env.command_manager.get_command(command_name)
        goal_red = env.scene["cone_red"].data.object_pos_w.squeeze(1) - env.scene.env_origins
        goal_green = env.scene["cone_green"].data.object_pos_w.squeeze(1) - env.scene.env_origins
        goal_blue = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - env.scene.env_origins
        goal_tensor = torch.stack([goal_red, goal_green, goal_blue], dim=1)
        goal_tensor[:, :, 2] += base_height
        selected_goal = goal_tensor * rgb_commands.unsqueeze(2)
        pos_goal = selected_goal.sum(dim=1)
        dis=torch.abs(root_env[:, 2] - pos_goal[:, 2])
        start_id=torch.where(self.dis==0.)
        reward = self.dis-dis
        reward[start_id]=0.
        self.dis=dis
        return reward
    
    def reset(self, env_ids: torch.Tensor):
        self.dis[env_ids]=0.        
        
def goal_heading_l1(env: ManagerBasedRLEnv, base_height, command_name, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    
    root_env = asset.data.root_pos_w - env.scene.env_origins
    rgb_commands=env.command_manager.get_command(command_name)
    goal_red = env.scene["cone_red"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_green = env.scene["cone_green"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_blue = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_tensor = torch.stack([goal_red, goal_green, goal_blue], dim=1)
    goal_tensor[:, :, 2] += base_height
    selected_goal = goal_tensor * rgb_commands.unsqueeze(2)
    pos_goal = selected_goal.sum(dim=1)
    
    root_quat = asset.data.root_quat_w

    # Compute direction to goal in yaw
    direction_to_goal = pos_goal[:, :2] - root_env[:, :2]
    goal_yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])

    # Convert quaternion to yaw
    w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    current_yaw = torch.atan2(t3, t4)

    # Calculate yaw error
    yaw_error = torch.abs(goal_yaw - current_yaw)
    yaw_error = torch.where(yaw_error > torch.pi, 2 * torch.pi - yaw_error, yaw_error)

    # Convert yaw error to linear reward
    linear_reward = 2 * yaw_error / torch.pi
    
    pos_r = asset.data.root_pos_w - env.scene.env_origins
    quat_r = asset.data.root_quat_w
    head_pos=torch.tensor(HEAD_POS, device=env.device).repeat(env.num_envs, 1)
    head_pos_r = pos_r + math_utils.quat_apply(quat_r, head_pos)
    dis=torch.norm(head_pos_r - pos_goal, dim=1)
    
    return 1 - linear_reward*(dis>0.25)

def goal_heading_cos(env: ManagerBasedRLEnv, goal_xyz, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_env = asset.data.root_pos_w - env.scene.env_origins
    goal_xyz = torch.tensor(goal_xyz, device=env.device).repeat(env.num_envs, 1)
    root_quat = asset.data.root_quat_w
    direction_to_goal = goal_xyz[:, :2] - root_env[:, :2]
    direction_to_goal = direction_to_goal / torch.norm(direction_to_goal, dim=1, keepdim=True)
    # Convert quaternion to Euler angles
    w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    # Calculate the yaw
    forward_direction = torch.stack([torch.cos(yaw_z), torch.sin(yaw_z)], dim=1)
    cos_theta = torch.sum(direction_to_goal * forward_direction, dim=1)
    # print(cos_theta)

    return cos_theta

def stuck(env: ManagerBasedRLEnv, threshold, goal_xyz, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_lin_vel = asset.data.root_lin_vel_b
    # root_env = asset.data.root_pos_w - env.scene.env_origins
    goal_xyz = torch.tensor(goal_xyz, device=env.device).repeat(env.num_envs, 1)
    pos_r = asset.data.root_pos_w - env.scene.env_origins
    quat_r = asset.data.root_quat_w
    head_pos=torch.tensor(HEAD_POS, device=env.device).repeat(env.num_envs, 1)
    head_pos_r = pos_r + math_utils.quat_apply(quat_r, head_pos)
    
    dis=torch.norm(head_pos_r - goal_xyz, dim=1)
    return (torch.norm(root_lin_vel[:, :2], dim=1) < threshold)*(dis>0.25)

def stand_still_at_goal(env: ManagerBasedRLEnv, base_height, command_name, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # root_env = asset.data.root_pos_w - env.scene.env_origins
    rgb_commands=env.command_manager.get_command(command_name)
    goal_red = env.scene["cone_red"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_green = env.scene["cone_green"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_blue = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_tensor = torch.stack([goal_red, goal_green, goal_blue], dim=1)
    goal_tensor[:, :, 2] += base_height
    selected_goal = goal_tensor * rgb_commands.unsqueeze(2)
    pos_goal = selected_goal.sum(dim=1)
    
    pos_r = asset.data.root_pos_w - env.scene.env_origins
    quat_r = asset.data.root_quat_w
    
    head_pos=torch.tensor(HEAD_POS, device=env.device).repeat(env.num_envs, 1)
    head_pos_r = pos_r + math_utils.quat_apply(quat_r, head_pos)
    dis=torch.norm(head_pos_r - pos_goal, dim=1)
    return (dis<0.25)*(1/(torch.norm(env.action_manager.get_term("joint_pos").velocity_command, dim=1)+0.4))


def penalty_wrong_goal(env: ManagerBasedRLEnv, base_height, command_name, threshold, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pos_r = asset.data.root_pos_w - env.scene.env_origins
    quat_r = asset.data.root_quat_w
    head_pos = torch.tensor(HEAD_POS, device=env.device).repeat(env.num_envs, 1)
    head_pos_r = pos_r + math_utils.quat_apply(quat_r, head_pos)
    
    rgb_commands = env.command_manager.get_command(command_name)
    goal_red = env.scene["cone_red"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_green = env.scene["cone_green"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_blue = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_red[:, 2] += base_height
    goal_green[:, 2] += base_height
    goal_blue[:, 2] += base_height

    # Calculate distances to each goal
    dist_red = torch.norm(head_pos_r - goal_red, dim=1)
    dist_green = torch.norm(head_pos_r - goal_green, dim=1)
    dist_blue = torch.norm(head_pos_r - goal_blue, dim=1)

    # print(dist_red, dist_green, dist_blue)
    
    # Determine if the robot is at the wrong goal
    wrong_goal_penalty = torch.zeros(env.num_envs, device=env.device)
    wrong_goal_penalty += (rgb_commands[:, 0] == 0) * (dist_red < threshold).float()
    wrong_goal_penalty += (rgb_commands[:, 1] == 0) * (dist_green < threshold).float()
    wrong_goal_penalty += (rgb_commands[:, 2] == 0) * (dist_blue < threshold).float()
    # print(wrong_goal_penalty)
    
    return wrong_goal_penalty  # Apply a penalty of -10 for being at the wrong goal


def track_lin_vel_xy_exp_command(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.action_manager.get_term("joint_pos").velocity_command[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp_command(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.action_manager.get_term("joint_pos").velocity_command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)
