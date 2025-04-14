# imports
# from chex import dataclass
from chex import dataclass
from dataclasses import field
from jaxtyping import Array
import jax.numpy as jnp
import parabellum as pb
from typing import Any, Callable
import parabellum as pb


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

    def __len__(self):
        return self.atomics_id.shape[0]


@dataclass
class State:
    status: Array
    action: pb.types.Action

    def __add__(self, other):
        return State(status=self.status + other.status, action=self.action + other.action)


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
