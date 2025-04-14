# imports
# from chex import dataclass
from chex import dataclass
from jax import Array
import parabellum as pb
from typing import Any, Callable


# dataclasses
@dataclass
class Parent:  # for behavior tree
    SEQUENCE: int = 1
    NONE: int = 0
    FALLBACK: int = -1


@dataclass
class Behavior:
    atomics_id: Array
    parents: Array
    predecessors: Array
    passings: Array


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
    terrain: pb.types.Terrain


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
