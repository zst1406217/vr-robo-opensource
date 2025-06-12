from dataclasses import MISSING

from omni.isaac.lab.controllers import DifferentialIKControllerCfg
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs.mdp import JointPositionActionCfg, JointPositionAction
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from vrrobo_isaaclab.wrapper.rl_cfg import (
    RslRlPpoActorCriticCfg,
)
import torch
from . import actions

@configclass
class VelocityCommandActionCfg(JointPositionActionCfg):
    """Configuration for the velocity command action term.

    See :class:`VelocityCommandAction` for more details.
    """

    class_type: type[ActionTerm] = actions.VelocityCommandAction

    use_default_offset: bool = True
    
    velocity_range: list = MISSING
    
    policy_dir: str = MISSING
    
    uplevel_frequency: int = MISSING