# imports
from chex import dataclass
from dataclasses import field
from jaxtyping import Array
import parabellum as pb
import jax.numpy as jnp


# dataclasses
@dataclass
class Status:
    status: Array = field(default_factory=lambda: jnp.array(True))  # status
    active: Array = field(default_factory=lambda: jnp.array(False))  # active

    @property
    def success(self):
        return self.status

    @property
    def failure(self):
        return ~self.status


@dataclass
class Behavior:
    idx: Array
    parent: Array
    prev: Array
    skip: Array


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
