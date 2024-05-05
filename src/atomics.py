# atomics.py
#   c2sim bt molecules (complex functions)
# by: Noah Syrkis

# imports
import jax
import jax.numpy as jnp
from jax import random, lax
from jaxmarl import make
import numpy as np
from functools import partial

from .utils import Status, dir_to_idx, idx_to_dir, STAND

# constants
SUCCESS, FAILURE, RUNNING = Status.SUCCESS, Status.FAILURE, Status.RUNNING
ATOMICS = [
    "attack",
    "move",
    "stand",
    "in_region",
    "in_sight",
    "in_reach",
    "is_armed",
    "is_dying",
]
FF_DICT = {
    ("enemy", "friend"): ("enemy", lambda env: env.num_allies),
    ("enemy", "foe"): ("ally", lambda _: 0),
    ("ally", "foe"): ("enemy", lambda env: env.num_allies - 1),
    ("ally", "friend"): ("ally", lambda _: 0),
}


"""
TODO: the ids of allies and enemies are super arbitrary.
Maybe we should have the agent index agents by distance?
"""


# helpers
@partial(jax.jit, static_argnums=(1, 2))
def process_obs(obs, agent, env):
    n, k = env.num_agents, 10  # len(env.own_features)
    is_ally = agent.startswith("ally")
    order = jnp.where(is_ally, 1, -1)
    self_obs = obs[-k:]
    others_obs = obs[:-k].reshape(n - 1, -1)
    idx = env.num_allies - int(is_ally)
    return self_obs, others_obs, idx


def agent_info_fn(state, _, agent, env):
    agent_id = env.agent_ids[agent]
    agent_type = state.unit_types[agent_id]
    sight_range = env.unit_type_sight_ranges[agent_type]
    attack_range = env.unit_type_attack_ranges[agent_type]
    return sight_range, attack_range


# actions
def attack(target):  # TODO: attack closest if no target
    if "_" in target:
        target = int(target.split("_")[-1])

        def attack_fn(state, obs, agent, env):
            is_ally = agent.startswith("ally")
            self_obs, others_obs, idx = process_obs(obs, agent, env)
            sight_range, attack_range = agent_info_fn(state, obs, agent, env)
            target_idx = jnp.where(is_ally, target + idx, target)
            target_obs = others_obs[target_idx]
            dist = jnp.linalg.norm(target_obs[1:3] - self_obs[1:3])
            status = jnp.where(dist < (attack_range / sight_range), RUNNING, FAILURE)
            action = jnp.where(status != FAILURE, STAND, target + 5)
            return (status, action)
    else:

        def attack_fn(state, obs, agent, env):
            # attack closest
            return (RUNNING, 5)

    return attack_fn


def move(direction):
    if direction == "center":
        mat_to_dir = jnp.array([[1, 3], [0, 2]])

        def center_fn(state, obs, agent, env):
            agent_id = env.agent_ids[agent]
            self_pos = obs[-len(env.own_features) :][1:3] * 32 - 16
            dimension = jnp.argmax(jnp.abs(self_pos))
            direction = jnp.where(self_pos[dimension] > 0, 1, 0)
            action = mat_to_dir[dimension, direction]
            # move on dimension with higest absolute value
            is_alive = state.unit_health[agent_id] > 0
            action = jnp.where(is_alive, action, STAND)
            return (RUNNING, action)

        return center_fn

    return lambda *_: (RUNNING, dir_to_idx[direction])


def stand(*_):
    return (RUNNING, STAND)


# location conditions
def in_region(x, y=None):  # only applies to self
    y = x if y is None else y  # in_region center instead of in_region center center
    dir2int = {"north": 1, "south": -1, "west": -1, "east": 1, "center": 0}

    def in_region_fn(state, obs, agent, env):
        self_pos = obs[-len(env.own_features) :][1:3]
        # confirm pos ranges from -1 to 1 (might be from 0 to 1)
        row = jnp.where(self_pos[0] > 2 / 3, 1, jnp.where(self_pos[0] < 1 / 3, -1, 0))
        col = jnp.where(self_pos[1] > 2 / 3, 1, jnp.where(self_pos[1] < 1 / 3, -1, 0))
        flag = jnp.logical_and(row == dir2int[x], col == dir2int[y])
        return jnp.where(flag, SUCCESS, FAILURE)

    return in_region_fn


def in_sight(target, d):  # is unit x in direction y?
    n = int(target.split("_")[-1]) if "_" in target else -1

    def in_sight_fn(state, obs, agent, env):
        team, offset_fn = FF_DICT[(agent.split("_")[0], target.split("_")[0])]
        offset = offset_fn(env)
        _, others_obs, _ = process_obs(obs, agent, env)
        target_pos = others_obs[n + offset][1:3]
        status = jnp.where(d in ["east", "west"], target_pos[1] > 0, target_pos[0] > 0)
        return jnp.where(status, SUCCESS, FAILURE)

    return in_sight_fn


def in_reach(other_agent):  # in shooting range
    if "_" in other_agent:
        n = int(other_agent.split("_")[-1]) if "_" in other_agent else -1

        def in_reach_fn(state, obs, self_agent, env):
            team, offset_fn = FF_DICT[
                (self_agent.split("_")[0], other_agent.split("_")[0])
            ]
            self_obs, others_obs, _ = process_obs(obs, self_agent, env)
            other_obs = others_obs[n + offset_fn(env)]
            alive = other_obs[0] > 0
            dist = jnp.linalg.norm(other_obs[1:3])
            sight_range, attack_range = agent_info_fn(state, obs, self_agent, env)
            flag = jnp.logical_and(attack_range / sight_range > dist, alive)
            return jnp.where(flag, SUCCESS, FAILURE)
    else:

        def in_reach_fn(state, obs, self_agent, env):  # if any is in reach
            # ALMOST DONE
            is_ally = self_agent.startswith("ally")
            self_obs, others_obs, _ = process_obs(obs, self_agent, env)
            n_targets = jnp.where(is_ally, env.num_enemies, env.num_allies)
            m = jnp.where(is_ally, env.num_allies, env.num_enemies) - 1
            alive = (others_obs.T[0] > 0)[m : m + n_targets]
            dist = jnp.linalg.norm(others_obs.T[1:3], axis=0)[m : m + n_targets]
            sight_range, attack_range = agent_info_fn(state, obs, self_agent, env)
            flag = (jnp.logical_and(attack_range / sight_range > dist, alive)).any()
            return jnp.where(flag, SUCCESS, FAILURE)

    return in_reach_fn


# other conditions
def is_armed(agent):
    agent = -1 if agent == "self" else int(agent.split("_")[-1])

    @partial(jax.jit, static_argnums=(2, 3))
    def is_armed_fn(state, obs, self_agent, env):
        others_obs = obs[: -len(env.own_features)].reshape(env.num_agents - 1, -1)
        other_obs = others_obs[agent]
        return jnp.where(other_obs[-1] > 0, SUCCESS, FAILURE)

    return is_armed_fn


def is_dying(agent):
    agent = -1 if agent == "self" else int(agent.split("_")[-1])

    @partial(jax.jit, static_argnums=(2, 3))
    def aux(state, obs, self_agent, env):
        others_obs = obs[: -len(env.own_features)].reshape(env.num_agents - 1, -1)
        other_obs = others_obs[agent]
        return jnp.where(other_obs[-len(env.own_features)] < 0.25, SUCCESS, FAILURE)

    return aux


def main():
    rng = random.PRNGKey(0)
    env = make("SMAX")
    obs, state = env.reset(rng)
    args = (state, obs["ally_0"], "ally_0", env)
    # region test
    print(in_region("west", "center")(*args))
