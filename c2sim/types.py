# imports
from chex import dataclass
import chex
from jax import jit, vmap, Array
import jax.numpy as jnp
from typing import Any, Callable, List, Tuple, Dict, Optional


# dataclasses
@dataclass
class Args:  # for behavior tree
    status: Any
    action: Any
    obs: Any
    child: int
    env_info: Any
    agent_info: Any


@dataclass
class Status:  # for behavior tree
    SUCCESS: int = 1
    FAILURE: int = -1

@dataclass
class EnvInfo:  # for atomics (who am i, what is this world?)
    num_allies: Array
    num_enemies: Array
    num_agents: Array
    num_own_features: Array
    num_types: Array
    map_width: Array
    map_height: Array
    time_per_step: Array
    world_steps_per_env_step: Array
    terrain_raster: Array

@dataclass
class AgentInfo:
    agent_id: Array
    velocity: Array
    sight_range: Array
    attack_range: Array
    is_ally: Array

# types
NodeFunc = Callable[[Any], Status]
