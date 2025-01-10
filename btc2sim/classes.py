# imports
#from chex import dataclass
import chex
from jax import jit, vmap, Array
import jax.numpy as jnp
from typing import Any, Callable, List, Tuple, Dict, Optional
#from parabellum import tps
from flax.struct import dataclass


@dataclass
class Terrain:
    building: chex.Array
    water: chex.Array
    forest: chex.Array
    basemap: chex.Array

    def test(self):
        print("banana")
    
    def __getitem__(self, index):  # to allow slicing operations
        return Terrain(
            building=self.building[index],
            water=self.water[index],
            forest=self.forest[index],
            basemap=self.basemap[index],
        )


# dataclasses
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
    terrain: Terrain

@dataclass
class AgentInfo:
    agent_id: Array
    velocity: Array
    sight_range: Array
    attack_range: Array
    is_ally: Array
    direction_map: Array

@dataclass
class Info:
    env: EnvInfo
    agent: AgentInfo

# types
NodeFunc = Callable[[Any], Status]


