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
ATOMICS = ["attack", "move", "region", "locate", "shootable"]

"""
TODO: the ids of allies and enemies are super arbitrary.
Maybe we should have the agent index agents by distance?
"""


# atomic functions
def attack(agent):  # move in a random direction
    def aux(state, obs, self_agent, env):
        # return failure if observed health is 0
        return (RUNNING, int(agent) + 5)

    return aux


def move(direction):
    return lambda *_: (RUNNING, dir_to_idx[direction])


def region(x, y):
    dir2int = {"north": 0, "south": 2, "west": 0, "east": 2, "center": 1}

    def aux(state, obs, agent, env):
        self_pos = obs[-len(env.own_features) :][1:3]
        # confirm pos ranges from -1 to 1 (might be from 0 to 1)
        row = jnp.where(self_pos[0] > 2 / 3, 1, jnp.where(self_pos[0] < 1 / 3, -1, 0))
        col = jnp.where(self_pos[1] > 2 / 3, 1, jnp.where(self_pos[1] < 1 / 3, -1, 0))
        flag = jnp.logical_and(row == dir2int[x], col == dir2int[y])
        return jnp.where(flag, SUCCESS, FAILURE)

    return aux


def locatable(other_agent, direction):  # is unit x in direction y?
    other_agent = int(other_agent.split("_")[-1])
    verts = ["north", "south"]

    def aux(state, obs, self_agent, env):
        is_ally = self_agent.startswith("ally")
        order = jnp.where(is_ally, 1, -1)
        idx = jnp.where(is_ally, other_agent, env.num_agents - 1 - other_agent)
        s_pos = obs[-len(env.own_features) :][1:3]  # self pos
        o_pos = obs[: -len(env.own_features)].reshape(env.num_agents - 1, -1)[idx][1:3]
        flag = jnp.where(direction in verts, o_pos[0] > s_pos[1], o_pos[1] > s_pos[2])
        return jnp.where(flag, SUCCESS, FAILURE)

    return aux


def shootable(other_agent):  # in shooting range
    def aux(state, obs, self_agent, env):
        # self and other obs
        self_obs = obs[-len(env.own_features) :]
        other_obs = obs[: -len(env.own_features)].reshape(env.num_agents - 1, -1)
        rel_pos = other_obs[:, 1:3] - self_obs[1:3]
        dist = jnp.linalg.norm(rel_pos, axis=1)
        return jnp.where(dist[other_agent] < 0.5, SUCCESS, FAILURE)

    return aux


# atomics
def armed(agent):
    agent = -1 if agent == "self" else int(agent.split("_")[-1])

    def aux(state, obs, self_agent, env):
        others_obs = obs[: -len(env.own_features)].reshape(env.num_agents - 1, -1)
        other_obs = others_obs[agent]
        return jnp.where(other_obs[-1] > 0, SUCCESS, FAILURE)

    return aux


def dying(agent):
    agent = -1 if agent == "self" else int(agent.split("_")[-1])

    def aux(state, obs, self_agent, env):
        others_obs = obs[: -len(env.own_features)].reshape(env.num_agents - 1, -1)
        other_obs = others_obs[agent]
        return jnp.where(other_obs[-len(env.own_features)] < 0.25, SUCCESS, FAILURE)

    return aux


def centralize(state, obs, agent, env):
    dim_dir_matrix = jnp.array([[2, 1], [0, 3]])
    self_obs = obs[-len(env.own_features) :]
    self_pos = self_obs[1:3] - 0.75
    dimension = jnp.where(jnp.abs(self_pos[0]) > jnp.abs(self_pos[1]), 0, 1)
    direction = jnp.where(self_pos[dimension] > 0, 0, 1)
    action = dim_dir_matrix[dimension, direction]
    return (RUNNING, action)


def main():
    rng = jax.random.PRNGKey(1)
    env = make("SMAX", num_allies=10, num_enemies=10, walls_cause_death=False)
    obs, state = env.reset(rng)

    # test region detection
    top_north_fn = region("north", "center")
    print(top_north_fn(obs["ally_0"], env))
