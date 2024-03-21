# atomics.py
#   atomic c2sim bt functions
# by: Noah Syrkis

# imports
import jax


# atomics
def enemy_found(rng, env, obs):  # see's for a given agent
    allies = obs[: (env.num_allies - 1) * 11].reshape(env.num_allies - 1, -1)
    enemies = obs[(env.num_allies - 1) * 11 : -10].reshape(env.num_enemies, -1)
    own = obs[-10:]
    return enemies.sum() > 0


def find_enemy(rng, env, obs):  # random movement (or stop) action
    return env, obs, rng.randint(0, 5)


def attack_enemy(rng, env, obs):  # attack random enemy in range
    enemies = obs[(env.num_allies - 1) * 11 : -10].reshape(env.num_enemies, -1)
    in_range = enemies.sum(axis=1) > 0
    return env, obs, rng.choose(in_range.nonzero()[0]) + 5
