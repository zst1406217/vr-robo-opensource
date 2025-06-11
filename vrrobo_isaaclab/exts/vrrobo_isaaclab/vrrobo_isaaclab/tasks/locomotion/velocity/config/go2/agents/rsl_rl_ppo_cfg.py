# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from vrrobo_isaaclab.wrapper.rl_cfg import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticCNNCfg,
    RslRlPpoAlgorithmCfg,
    RslRlPpoAmpCfg,
)

@configclass
class UnitreeGo2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    device="cuda:0"
    num_steps_per_env = 24
    max_iterations = 80000
    save_interval = 2000
    experiment_name = "unitree_go2_rough"
    empirical_normalization = False
    resume = True
    load_run = "2024-12-29_23-55-29"
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticRecurrent",
        init_noise_std=0.75,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    amp = RslRlPpoAmpCfg(
        hidden_dims=[1024, 512],
        learning_rate=1e-3,
        file_path="./exts/mpc_data/mpc_data_go2.npy",
        gradient_penalty=1,
        reward_weight=0.025,
    )

@configclass
class UnitreeGo2FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    device="cuda:0"
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 1000
    experiment_name = "unitree_go2_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticRecurrent",
        init_noise_std=0.75,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    amp = RslRlPpoAmpCfg(
        hidden_dims=[1024, 512],
        learning_rate=1e-3,
        file_path="./exts/mpc_data/mpc_data_go2.npy",
        gradient_penalty=1,
        reward_weight=0.025,
    )
    
@configclass
class UnitreeGo2GSFixPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    device="cuda:0"
    num_steps_per_env = 40
    max_iterations = 8000
    save_interval = 100
    experiment_name = "unitree_go2_gsfix"
    empirical_normalization = False
    resume = False
    load_run = "2025-05-15_12-00-43"
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticRecurrent",
        init_noise_std=0.9,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=2,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1,
    )
    amp = RslRlPpoAmpCfg(
        hidden_dims=[1024, 512],
        learning_rate=1e-3,
        file_path="./exts/mpc_data/mpc_data_go2.npy",
        gradient_penalty=1,
        reward_weight=0.0,
    )
    
@configclass
class UnitreeGo2GSFtPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    device="cuda:0"
    num_steps_per_env = 40
    max_iterations = 80000
    save_interval = 100
    experiment_name = "unitree_go2_gsft"
    empirical_normalization = False
    resume = True
    load_run = "2025-01-12_20-58-01"
    # policy = RslRlPpoActorCriticCNNCfg(
    #     class_name="ActorCriticCNNRecurrent",
    #     init_noise_std=2.0,
    #     actor_hidden_dims=[512, 256, 128],
    #     critic_hidden_dims=[512, 256, 128],
    #     activation="elu",
    #     cnn_type="mobilenet_v3_small",
    #     pretrain=True,
    #     image_size=[1, 180, 320],
    #     num_cnn_features=576,
    # )
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticRecurrent",
        init_noise_std=2.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=2,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1,
    )
    amp = RslRlPpoAmpCfg(
        hidden_dims=[1024, 512],
        learning_rate=1e-3,
        file_path="./exts/mpc_data/mpc_data_go2.npy",
        gradient_penalty=1,
        reward_weight=0.0,
    )