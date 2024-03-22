# atomics.py
#   atomic c2sim bt functions
# by: Noah Syrkis

# imports
import jax
import jax.numpy as jnp
from jaxmarl import make

from utils import Status


# functions
def see_fn(env, obs, agent):
    self_obs = obs[-env.own_features :]  # self obs is 10 long, other's is 11
    other_obs = obs[: -env.own_features].reshape(env.num_agents - 1, -1)
    split_idx = env.num_allies - (1 if agent.startswith("ally") else 0)
    return self_obs, other_obs[:split_idx], other_obs[split_idx:]


# atomics
def enemy_found(rng, env, obs, agent):  # see's for a given AGENT
    self_obs, allies_obs, enemies_obs = see_fn(env, obs, agent)
    if agent.startswith("ally"):
        return Status.SUCCESS if enemies_obs.sum() > 0 else Status.FAILURE
    if agent.startswith("enemy"):
        return Status.SUCCESS if allies_obs.sum() > 0 else Status.FAILURE


def find_enemy(rng, env, obs, agent):  # random movement (or stop) action
    return Status.RUNNING, jax.random.randint(rng, (1,), 0, 5)


def attack_enemy(rng, env, obs, agent):  # attack random enemy in range
    self_obs, allies_obs, enemies_obs = see_fn(env, obs, agent)
    in_range = enemies_obs.sum(axis=1) > 0  # if any enemy is in range
    return Status.SUCCESS, jax.random.choice(rng, in_range.nonzero()[0]) + 5


def main():
    env = make("SMAX", num_allies=2, num_enemies=5)
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    see_fn(env, obs["enemy_0"], "enemy_0")
    print(find_enemy(rng, env, obs["enemy_0"], "enemy_0"))
    print(len(env.own_features))


if __name__ == "__main__":
    main()
