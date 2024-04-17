# quarks.py
#   atomic c2sim bt functions
# by: Noah Syrkis

# imports
import jax
import jax.numpy as jnp
from jax import random, lax
from jaxmarl import make

from functools import partial

from .utils import Status, dir_to_idx, idx_to_dir

# constants
SUCCESS, FAILURE, RUNNING = Status.SUCCESS, Status.FAILURE, Status.RUNNING


"""
TODO: the ids of allies and enemies are super arbitrary.
Maybe we should have the agent index agents by distance?
"""


# actions
def action_fn(action):  # move in a random direction
    return lambda *_: (SUCCESS, action)


# helpers
@partial(jax.vmap, in_axes=(0, None))
def parse_unit_obs(obs, env):
    hp, pos_x, pos_y, last_action, weapon_cd = obs[:5]
    return hp, (pos_x, pos_y), last_action, weapon_cd


# conditions
def sight_fn(direction, other_agent):  # are there any enemies to direction?
    def aux(obs, self_agent, env):
        # self and other obs
        self_obs = obs[-len(env.own_features) :]
        other_obs = obs[: -len(env.own_features)].reshape(env.num_agents - 1, -1)
        rel_pos = other_obs[:, 1:3] - self_obs[1:3]
        column = jnp.where(dir_to_idx[direction] < 2, rel_pos[:, 1], rel_pos[:, 0])
        sight = jnp.where(dir_to_idx[direction] % 2 == 0, column > 0, column < 0)
        # TODO: logical and that is has health > 0
        return jnp.where(sight[other_agent], SUCCESS, FAILURE)

    return aux


""" def reminisce_fn(action):  # only last actions of others are in obs
    return lambda obs, *_: obs[-1] == action """


# decorators
def negate(state, _):
    if state == RUNNING:
        return RUNNING
    return SUCCESS if state == FAILURE else FAILURE


def main():
    rng = jax.random.PRNGKey(1)
    env = make("SMAX", num_allies=1, num_enemies=10)
    obs, state = env.reset(rng)
    sight = sight_fn("north", 1)
    print(sight(obs["ally_0"], "ally_0", env))
