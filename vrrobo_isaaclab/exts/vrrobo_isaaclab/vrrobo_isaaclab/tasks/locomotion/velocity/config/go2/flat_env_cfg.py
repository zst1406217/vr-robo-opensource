# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass
from vrrobo_isaaclab.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
# import omni.isaac.lab.terrains as terrain_gen
import vrrobo_isaaclab.terrains as terrain_gen

from vrrobo_isaaclab.assets import UNITREE_GO2_CFG  # isort: skip

@configclass
class UnitreeGo2FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        self.scene.num_envs = 4096
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        self.scene.terrain.slope_threshold=2
        # self.scene.terrain.terrain_generator.sub_terrains = {
        #     "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #         proportion=1.0, noise_range=(0.01, 0.1), noise_step=0.01, border_width=0.
        #     ),
        # }
        self.scene.terrain.terrain_generator.horizontal_scale = 0.05
        self.scene.terrain.terrain_generator.sub_terrains = {
            "perlin_terrain": terrain_gen.HfPerlinTerrainCfg(
                horizontal_scale=0.05, frequency=10, zScale=0.0
            ),
        }

        # reduce action scale
        self.actions.joint_pos.scale = 0.5

        # event
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        # self.rewards.feet_air_time.weight = 0.1
        self.rewards.undesired_contacts = None
        # self.rewards.dof_torques_l2.weight = -0.0002
        # self.rewards.track_lin_vel_xy_exp.weight = 5.0
        # self.rewards.track_ang_vel_z_exp.weight = 1.0
        # self.rewards.dof_acc_l2.weight = 0.

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
        # self.terminations.base_contact=None
        
        # no curriculum
        self.curriculum.terrain_levels = None


@configclass
class UnitreeGo2FlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.terrain.max_init_terrain_level=10
