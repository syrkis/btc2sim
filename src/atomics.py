# atomics.py
#   atomic c2sim bt functions
# by: Noah Syrkis

# imports
import jax
import jax.numpy as jnp
from jax import random, lax
from jaxmarl import make

from .utils import Status

# constants
SUCCESS, FAILURE, RUNNING = Status.SUCCESS, Status.FAILURE, Status.RUNNING


# functions
def see_fn(obs, agent, env):
    other_obs = obs[: -len(env.own_features)].reshape(env.num_agents - 1, -1)
    split_idx = env.num_allies - (jnp.where(agent.startswith("ally"), 1, 0))
    mask = jnp.arange(env.num_agents - 1) < split_idx
    return other_obs, mask


# atomics
def enemy_found(_, obs, agent, env):  # see's for a given AGENT
    other_obs, mask = see_fn(obs, agent, env)
    mask = jnp.where(agent.startswith("ally"), ~mask, mask)
    return jnp.where(jnp.absolute(other_obs[mask].sum()) > 0, SUCCESS, FAILURE)


def find_enemy(rng, _, __, ___):  # walk around randomly to find enemy
    # just chose a random direction to move in for now
    return RUNNING, random.randint(rng, (1,), 0, 5)[0]


def attack_enemy(rng, obs, agent, env):  # attack random enemy in range
    other_obs, mask = see_fn(obs, agent, env)
    mask = jnp.where(agent.startswith("ally"), ~mask, mask[::-1])
    in_sight = jnp.absolute(other_obs[mask]).any(axis=1)
    # probs = jnp.zeros(in_sight.size).at[in_sight].set(1)  # probability of attacking
    idxs = jnp.where(in_sight, size=in_sight.size)[0]
    probs = jnp.put(jnp.zeros(in_sight.size), idxs, 1, inplace=False)
    probs = jnp.where(probs.sum() > 0, probs / probs.sum(), probs)
    probs = jnp.concatenate((probs, (1 - probs.sum()).reshape(1)))
    actions = jnp.arange(probs.size).at[-1].set(-1)
    act = random.choice(rng, actions, p=probs)
    return SUCCESS, act


def main():
    env = make("SMAX", num_allies=2, num_enemies=5)
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    out = attack_enemy(rng, obs["enemy_0"], "enemy_0", env)
