"""
NWM - Negative Weight Mapping
=============================

A Reinforcement Learning framework that uses persistent potential fields
to guide agent exploration through attractive (success) and repulsive (failure) forces.

Quick Start
-----------
>>> import gymnasium as gym
>>> from nwm import NWM, NWMConfig
>>>
>>> env = gym.make("CartPole-v1")
>>> agent = NWM(
...     state_dim=env.observation_space.shape[0],
...     num_actions=env.action_space.n
... )
>>>
>>> state, _ = env.reset()
>>> action = agent.select_action(state)
>>> next_state, reward, done, truncated, _ = env.step(action)
>>> agent.step(state, action, reward, next_state, done or truncated)

Main Components
---------------
- NWM: The main RL agent using potential fields
- NWMConfig: Configuration dataclass for agent parameters
- PersistentCentroid: Memory unit storing state-action experiences
- PersistentPotentialField: Spatial memory structure with force calculations
"""

__version__ = "1.0.1"
__author__ = "CusterMustOfficial"

from nwm.agents.nwm_agent import NWMAgent as NWM
from nwm.core.centroid import PersistentCentroid
from nwm.core.potential_field import PersistentPotentialField
from nwm.utils.config import NWMConfig

__all__ = [
    "NWM",
    "NWMConfig",
    "PersistentCentroid",
    "PersistentPotentialField",
]
