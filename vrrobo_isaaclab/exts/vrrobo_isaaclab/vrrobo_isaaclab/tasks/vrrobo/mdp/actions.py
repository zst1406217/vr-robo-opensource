from . import JointPositionAction
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
import torch
from omni.isaac.lab.envs import ManagerBasedEnv
import os

class VelocityCommandAction(JointPositionAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""
    """The configuration of the action term."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        
        policy_cfg = {
            "class_name": "ActorCriticRecurrent",
            "init_noise_std": 0.75,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        }
        
        actor_critic_class = eval(policy_cfg["class_name"])  # ActorCritic
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            48, 235, 12, **policy_cfg
        ).to(self._env.device)
        policy_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.policy_dir))
        actor_critic.load_state_dict(torch.load(policy_dir)["model_state_dict"])
        actor_critic.eval()
        self.policy = actor_critic.act_inference
        self.velocity_range=torch.tensor(cfg.velocity_range).cuda()
        self.velocity_command = torch.zeros([self.num_envs, 3]).cuda()
        self.uplevel_frequency = cfg.uplevel_frequency
        self.lowlevel_counter = 0
        self.tanh = torch.nn.Tanh()

    def process_actions(self, actions: torch.Tensor):
        self.velocity_command = self.tanh(actions)*self.velocity_range
    
    def apply_actions(self):
        if self.lowlevel_counter % 4 == 0:
            obs = self.get_observations()
            with torch.inference_mode():
                obs[:, 9:12] = self.velocity_command
                joint_positions = self.policy(obs)
            super().process_actions(joint_positions)
        self.lowlevel_counter += 1
        self.lowlevel_counter %= 4
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)
        
    def get_observations(self):
        """Returns the current observations of the environment."""
        obs = self._env.observation_manager.compute_group("locomotion")
        return obs
    
    @property
    def action_dim(self) -> int:
        return 3