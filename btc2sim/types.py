# imports
from chex import dataclass
from jaxtyping import Array
import parabellum as pb


# dataclasses
@dataclass
class Behavior:
    idx: Array
    parent: Array
    prev: Array
    skip: Array

    def __len__(self):
        return self.idx.shape[0]


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
