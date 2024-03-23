# atomics.py
#   atomic c2sim bt functions
# by: Noah Syrkis

# imports
import jax
import jax.numpy as jnp
from jaxmarl import make

from .utils import Status


# functions
def see_fn(obs, agent, env):
    self_obs = obs[-len(env.own_features) :]  # self obs is 10 long, other's is 11
    other_obs = obs[: -len(env.own_features)].reshape(env.num_agents - 1, -1)
    split_idx = env.num_allies - (1 if agent.startswith("ally") else 0)
    return self_obs, other_obs[:split_idx], other_obs[split_idx:]


# atomics
def enemy_found(_, obs, agent, env):  # see's for a given AGENT
    _, allies_obs, enemies_obs = see_fn(obs, agent, env)
    targets = enemies_obs if agent.startswith("ally") else allies_obs
    return jnp.where(jnp.absolute(targets.sum()) > 0, Status.SUCCESS, Status.FAILURE)


def find_enemy(rng, *_):  # find random enemy
    return Status.RUNNING, jax.random.randint(rng, (1,), 0, 5)[0]


def attack_enemy(rng, obs, agent, env):  # attack random enemy in range
    _, allies_obs, enemies_obs = see_fn(obs, agent, env)
    targets = enemies_obs if agent.startswith("ally") else allies_obs
    in_range = jnp.absolute(targets).sum(axis=1) > 0  # if any enemy is in range
    action = jax.random.choice(rng, in_range.nonzero()[0]) + 5
    return Status.SUCCESS, action


def main():
    env = make("SMAX", num_allies=2, num_enemies=5)
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    print(enemy_found(rng, obs["ally_0"], "ally_0", env))
