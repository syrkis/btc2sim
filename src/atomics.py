# atomics.py
#   c2sim bt molecules (complex functions)
# by: Noah Syrkis

# imports
import jax
import jax.numpy as jnp
from jax import random, lax
from jaxmarl import make
from .utils import Status, dir_to_idx, idx_to_dir


# constants
SUCCESS, FAILURE, RUNNING = Status.SUCCESS, Status.FAILURE, Status.RUNNING


# constants
def am_armed(obs, self_agent, env):  # or is my weapon in cooldown?
    return jnp.where(obs[-1] > 0, SUCCESS, FAILURE)


def am_exiled(obs, self_agent, env):  # or is the enemy too far away?
    self_pos = obs[-len(env.own_features) : -1] - 16  # 16 is map size (get from env)
    return jnp.where(jnp.linalg.norm(self_pos) < 10, SUCCESS, FAILURE)


def am_dying(obs, agent, env):  # is my health below a certain threshold?
    thresh = 0.25 * env.unit_type_health[env.unit_type[agent]]
    return jnp.where(obs[-len(env.own_features)] < thresh, SUCCESS, FAILURE)


def enemy_found(obs, agent, env):
    _, _, other_team = see_teams(obs, agent, env)
    return jnp.where(jnp.sum(other_team[:, 0]) == 0, FAILURE, SUCCESS)


def find_enemy(obs, agent, env):
    self_obs = obs[-len(env.own_features) : -1]
    self_pos = self_obs[1:3] - 0.5
    dimension = jnp.abs(self_pos.argmax())
    direction = jnp.sign(self_pos[dimension])
    return (RUNNING, (2 * dimension + (jnp.sign(direction) + 1) // 2).astype(jnp.int32))


def attack_enemy(obs, agent, env):
    self_obs, my_team, other_team = see_teams(obs, agent, env)
    potential_targets = other_team[:, 0] > 0
    target = jnp.concatenate([jnp.where(potential_targets)[0] + 5, jnp.array([-1])])[0]
    return (RUNNING, target)


def see_teams(obs, agent, env):
    self_obs = obs[-len(env.own_features) :]
    other_obs = obs[: -len(env.own_features)].reshape(env.num_agents - 1, -1)
    idx = jnp.where(agent.startswith("ally"), env.num_allies - 1, env.num_enemies - 1)
    my_team = other_obs[:idx]
    other_team = other_obs[idx:]
    return self_obs, my_team, other_team


def main():
    rng = jax.random.PRNGKey(1)
    env = make("SMAX", num_allies=10, num_enemies=10)
    obs, state = env.reset(rng)
    args = (obs["ally_0"], "ally_0", env)
    # print(attack_enemy(*args))
    print(find_enemy(*args))
    print(enemy_found(*args))
