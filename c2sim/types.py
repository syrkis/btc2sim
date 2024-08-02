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
    info: Any

@dataclass
class Status:  # for behavior tree
    SUCCESS: int = 1
    FAILURE: int = -1

@dataclass
class Info:  # for atomics (who am i, what is this world?)
    agent_id: int
    num_allies: int
    num_enemies: int
    num_agents: int
    num_own_features: int
    num_types: int
    # enemy_features: jnp.ndarray  # hardcode
    # ally_features: jnp.ndarray  # hardcode
    velocity: float
    sight_range: float
    attack_range: float
    is_ally: bool
    map_width: int
    map_height: int
    time_per_step: float
    world_steps_per_env_step: int
    terrain_raster: jnp.ndarray

# types
NodeFunc = Callable[[Any], Status]
