# -*- coding: utf-8 -*-
# Copyright (c) 2024 Galaxea

"""Configuration for the Galaxea R1 robot (production date: 0604)
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab_assets import ISAACLAB_ASSETS_DATA_DIR


##
# Configuration
##

GALAXEA_R1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"exts/vrrobo_isaaclab/robot_data/Galaxea/r1_v2_1_0_dummy_wheel.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "left_arm_joint1": 0.0,
            "left_arm_joint2": 2.0,
            "left_arm_joint3": 1.57,
            "left_arm_joint4": 0.0,
            "left_arm_joint5": 1.57,
            "left_arm_joint6": 0.0,
            "left_gripper_finger_joint1": 0.03,
            "left_gripper_finger_joint2": 0.03,
            "right_arm_joint1": 0.0,
            "right_arm_joint2": 2.0,
            "right_arm_joint3": -1.57,
            "right_arm_joint4": 0.0,
            "right_arm_joint5": -1.57,
            "right_arm_joint6": 0.0,
            "right_gripper_finger_joint1": 0.03,
            "right_gripper_finger_joint2": 0.03,
            "torso_joint1": 0.0,
            "torso_joint2": 0.0,
            "torso_joint3": 0.0,
            "torso_joint4": 0.0,
            "steer_motor_joint1": 0.0,
            "steer_motor_joint2": 0.0,
            "steer_motor_joint3": 0.0,
            "wheel_motor_joint1": 0.0,
            "wheel_motor_joint2": 0.0,
            "wheel_motor_joint3": 0.0,
        },
        joint_vel={".*": 0.0}   
    ),
    actuators={
        "r1_arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_joint[1-5]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "r1_eefs": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_joint6"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "r1_grippers": ImplicitActuatorCfg(
            joint_names_expr=[".*_gripper_finger_joint.*"],
            effort_limit=200.0,
            velocity_limit=0.25,
            stiffness=1e6,  # 1e7,
            damping=1e4,  # 1e5,
        ),
        "r1_torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_joint.*"],
            effort_limit=1000.0,
            velocity_limit=30.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "r1_steering": ImplicitActuatorCfg(
            joint_names_expr=["steer_motor_joint.*"],
            effort_limit=1000.0,
            velocity_limit=30.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "r1_wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_motor_joint.*"],
            effort_limit=1000.0,
            velocity_limit=30.0,
            stiffness=800.0,
            damping=40000.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
GALAXEA_R1_FIXBASE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"exts/vrrobo_isaaclab/robot_data/Galaxea/r1_v2_1_0_fixbase.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "left_arm_joint1": 0.0,
            "left_arm_joint2": 2.0,
            "left_arm_joint3": -1.57,
            "left_arm_joint4": 0.0,
            "left_arm_joint5": 1.57,
            "left_arm_joint6": 0.0,
            "left_gripper_finger_joint1": 0.03,
            "left_gripper_finger_joint2": 0.03,
            "right_arm_joint1": 0.0,
            "right_arm_joint2": 2.0,
            "right_arm_joint3": -1.57,
            "right_arm_joint4": 0.0,
            "right_arm_joint5": -1.57,
            "right_arm_joint6": 0.0,
            "right_gripper_finger_joint1": 0.03,
            "right_gripper_finger_joint2": 0.03,
            "torso_joint1": 0.0,
            "torso_joint2": 0.0,
            "torso_joint3": 0.0,
            "torso_joint4": 0.0,

        }    
    ),
    actuators={
        "r1_arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_joint[1-5]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "r1_eefs": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_joint6"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "r1_grippers": ImplicitActuatorCfg(
            joint_names_expr=[".*_gripper_finger_joint.*"],
            effort_limit=200.0,
            velocity_limit=0.25,
            stiffness=1e6,  # 1e7,
            damping=1e4,  # 1e5,
        ),
        "r1_torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_joint.*"],
            effort_limit=1000.0,
            velocity_limit=30.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
GALAXEA_R1_HIGH_PD_CFG = GALAXEA_R1_CFG.copy()
GALAXEA_R1_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = False
GALAXEA_R1_HIGH_PD_CFG.actuators["r1_arms"].stiffness = 400.0
GALAXEA_R1_HIGH_PD_CFG.actuators["r1_arms"].damping = 80.0
GALAXEA_R1_HIGH_PD_CFG.actuators["r1_eefs"].stiffness = 1000.0
GALAXEA_R1_HIGH_PD_CFG.actuators["r1_eefs"].damping = 200.0
GALAXEA_R1_HIGH_PD_CFG.actuators["r1_torso"].stiffness = 100000000.0
GALAXEA_R1_HIGH_PD_CFG.actuators["r1_torso"].damping = 2000.0
GALAXEA_R1_HIGH_PD_CFG.actuators["r1_steering"].stiffness = 10000.0
GALAXEA_R1_HIGH_PD_CFG.actuators["r1_steering"].damping = 2000.0
GALAXEA_R1_HIGH_PD_CFG.actuators["r1_wheels"].stiffness = 10000.0
GALAXEA_R1_HIGH_PD_CFG.actuators["r1_wheels"].damping = 1000000

GALAXEA_R1_HIGH_PD_GRIPPER_CFG = GALAXEA_R1_HIGH_PD_CFG.copy()
GALAXEA_R1_HIGH_PD_GRIPPER_CFG.actuators["r1_grippers"].stiffness = 1e3
GALAXEA_R1_HIGH_PD_GRIPPER_CFG.actuators["r1_grippers"].damping = 1e2

GALAXEA_R1_FIXBASE_HIGH_PD_CFG = GALAXEA_R1_FIXBASE_CFG.copy()
GALAXEA_R1_FIXBASE_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = False
GALAXEA_R1_FIXBASE_HIGH_PD_CFG.actuators["r1_arms"].stiffness = 400.0
GALAXEA_R1_FIXBASE_HIGH_PD_CFG.actuators["r1_arms"].damping = 80.0
GALAXEA_R1_FIXBASE_HIGH_PD_CFG.actuators["r1_eefs"].stiffness = 1000.0
GALAXEA_R1_FIXBASE_HIGH_PD_CFG.actuators["r1_eefs"].damping = 200.0
GALAXEA_R1_FIXBASE_HIGH_PD_CFG.actuators["r1_torso"].stiffness = 10000000000.0
GALAXEA_R1_FIXBASE_HIGH_PD_CFG.actuators["r1_torso"].damping = 2000.0

GALAXEA_R1_IK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"exts/vrrobo_isaaclab/robot_data/Galaxea/r1_v2_1_0_wheel_moved.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "left_arm_joint1": 0.0,
            "left_arm_joint2": 2.0,
            "left_arm_joint3": -1.57,
            "left_arm_joint4": 0.0,
            "left_arm_joint5": 1.57,
            "left_arm_joint6": 0.0,
            # "left_gripper_axis1": 0.03,
            # "left_gripper_axis2": 0.03,
            "right_arm_joint1": 0.0,
            "right_arm_joint2": 2.0,
            "right_arm_joint3": -1.57,
            "right_arm_joint4": 0.0,
            "right_arm_joint5": -1.57,
            "right_arm_joint6": 0.0,
            # "right_gripper_axis1": 0.03,
            # "right_gripper_axis2": 0.03,
        },
    ),
    actuators={
        "r1_arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_joint[1-5]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "r1_eefs": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_joint6"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        # "r1_grippers": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_gripper_axis.*"],
        #     effort_limit=200.0,
        #     velocity_limit=0.25,
        #     stiffness=1e6,  # 1e7,
        #     damping=1e4,  # 1e5,
        # ),
    },
    soft_joint_pos_limit_factor=1.0,
)

GALAXEA_R1_IK_HIGH_PD_CFG = GALAXEA_R1_IK_CFG.copy()
GALAXEA_R1_IK_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = False
GALAXEA_R1_IK_HIGH_PD_CFG.actuators["r1_arms"].stiffness = 400.0
GALAXEA_R1_IK_HIGH_PD_CFG.actuators["r1_arms"].damping = 80.0
GALAXEA_R1_IK_HIGH_PD_CFG.actuators["r1_eefs"].stiffness = 1000.0
GALAXEA_R1_IK_HIGH_PD_CFG.actuators["r1_eefs"].damping = 200.0


GALAXEA_R1_BASE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"exts/vrrobo_isaaclab/robot_data/Galaxea/fixbase.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    
    init_state=ArticulationCfg.InitialStateCfg(
        # pos= (-1.4,-1.5,0.06),
        pos= (0, 0, 0.045),
        # pos= (-1.4,-4.3874,0.06),
        # pos= (-1.4,1.3874,0.06),
        
        # rot=(0.3826834,0, 0, 0.9238795,),
        # rot=(0.9102341,0, 0, 0.414094,),
        joint_pos={
            ##########################
            # base
            "dummy_base_prismatic_x_joint": 0.0,
            "dummy_base_prismatic_y_joint": 0.0,
            "dummy_base_revolute_z_joint": 0.0,
            ##########################
            # other
            "left_arm_joint1": 0.0,
            "left_arm_joint2": 2.0,
            "left_arm_joint3": -1.57,
            "left_arm_joint4": 0.0,
            "left_arm_joint5": 1.57,
            "left_arm_joint6": 0.0,
            # "left_gripper_axis1": 0.03,
            # "left_gripper_axis2": 0.03,
            "right_arm_joint1": 0.0,
            "right_arm_joint2": 2.0,
            "right_arm_joint3": -1.57,
            "right_arm_joint4": 0.0,
            "right_arm_joint5": -1.57,
            "right_arm_joint6": 0.0,
            # "right_gripper_axis1": 0.03,
            # "right_gripper_axis2": 0.03,
        },
        joint_vel={".*": 0.0}
    ),
    actuators={
        ###################################
        "r1_base": ImplicitActuatorCfg(
            joint_names_expr=["dummy_base_.*"],
            velocity_limit=100.0,
            effort_limit=1000.0,
            stiffness=0.0,
            damping=1e5,
        ),
        ###################################
        "r1_arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_joint[1-5]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "r1_eefs": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_joint6"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        # "r1_grippers": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_gripper_axis.*"],
        #     effort_limit=200.0,
        #     velocity_limit=0.25,
        #     stiffness=100,  # 1e7,
        #     damping=10,  # 1e5,
        # ),
        "r1_torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_joint.*"],
            effort_limit=1000.0,
            velocity_limit=30.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

GALAXEA_R1_BASE_CFG = GALAXEA_R1_BASE_CFG.copy()
GALAXEA_R1_BASE_CFG.spawn.rigid_props.disable_gravity = False
GALAXEA_R1_BASE_CFG.actuators["r1_arms"].stiffness = 400.0
GALAXEA_R1_BASE_CFG.actuators["r1_arms"].damping = 80.0
GALAXEA_R1_BASE_CFG.actuators["r1_eefs"].stiffness = 1000.0
GALAXEA_R1_BASE_CFG.actuators["r1_eefs"].damping = 200.0
GALAXEA_R1_BASE_CFG.actuators["r1_torso"].stiffness = 10000000000.0
GALAXEA_R1_BASE_CFG.actuators["r1_torso"].damping = 2000.0
# GALAXEA_R1_BASE_CFG.actuators["r1_steering"].stiffness = 10000.0
# GALAXEA_R1_BASE_CFG.actuators["r1_steering"].damping = 2000.0
# GALAXEA_R1_BASE_CFG.actuators["r1_wheels"].stiffness = 10000.0
# GALAXEA_R1_BASE_CFG.actuators["r1_wheels"].damping = 2000.0


GALAXEA_CAMERA_CFG = CameraCfg(
    prim_path="/World/envs/env_.*/Camera",  # should be replaced with the actual parent frame
    update_period=1 / 60.0,  # 30 Hz
    height=240,
    width=320,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=12,
        focus_distance=100.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 100),
    ),
    offset=CameraCfg.OffsetCfg(  # offset from the parent frame
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        convention="ros",
    ),
)
