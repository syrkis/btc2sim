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


# # Actions

# ## tested

# atomic functions
def attack(agent):  # move in a random direction
    return lambda *_: (RUNNING, int(agent) + 5)


def move(direction):
    return lambda *_: (RUNNING, dir_to_idx[direction])


<<<<<<< HEAD
def region(x, y):
    dir2int = {"north": 0, "south": 2, "west": 0, "east": 2, "center": 1}

    def aux(obs, env):
        self_obs = obs[-len(env.own_features) :]
        self_pos = self_obs[1:3]
        # confirm pos ranges from -1 to 1 (might be from 0 to 1)
        row = jnp.where(self_pos[0] > 1 / 3, 1, jnp.where(self_pos[0] < -1 / 3, -1, 0))
        col = jnp.where(self_pos[1] > 1 / 3, 1, jnp.where(self_pos[1] < -1 / 3, -1, 0))
        return jnp.where(row == dir2int[x] and col == dir2int[y], SUCCESS, FAILURE)

    return aux


=======
def move_to_center(state, obs, agent, env):  # move to the center
    dim_dir_matrix = jnp.array([[2, 1], [0, 3]])
    self_obs = obs[-len(env.own_features) :]
    self_pos = self_obs[1:3] - 0.75
    dimension = jnp.where(jnp.abs(self_pos[0]) > jnp.abs(self_pos[1]), 0, 1)
    direction = jnp.where(self_pos[dimension] > 0, 0, 1)
    action = dim_dir_matrix[dimension, direction]
    return (RUNNING, action)


def attack_enemy(state, obs, agent, env):
    self_obs, my_team, other_team = see_teams(obs, agent, env)
    potential_targets = (other_team[:, 0] > 0).astype(jnp.int32)
    order = jnp.where(agent.startswith("ally"), 1, -1)
    potential_targets = potential_targets[::order]
    potential_targets = jnp.concatenate([potential_targets, jnp.array([1])])
    target = jnp.argmax(potential_targets)
    target = jnp.where(target == other_team.shape[0], 5, target + 5)  # do nothing if no target in sight 
    return RUNNING, target


# ## to test

# +
def go_to_target(target):
    dim_dir_matrix = jnp.array([[1, 2], [0, 3]])
    xy_axis = jnp.where(self_pos[0] > self_pos[1], 0, 1)
    xminusy_axis = jnp.where(self_pos[0] > -self_pos[1], 0, 1)
    return dim_dir_matrix[xy_axis, xminusy_axis]

def move_to_enemy_in_sight():
    self_obs, my_team, other_team = see_teams(obs, agent, env)
    potential_targets = (other_team[:, 0] > 0).astype(jnp.int32)
    order = jnp.where(agent.startswith("ally"), 1, -1)
    potential_targets = potential_targets[::order]
    potential_targets = jnp.concatenate([potential_targets, jnp.array([1])])
    target_idx = jnp.argmax(potential_targets)
    target = jnp.where(target_idx == other_team.shape[0], self_obs[1:3], other_team[target_idx][1:3]) 
    action = go_to_target(target)
    action = jnp.where(target_idx == other_team.shape[0], 5, action)  # do nothing if no target in sight 
    return (RUNNING, action)


# -

# # Conditions

# ## Tested

# +
def see_teams(obs, agent, env):
    self_obs = obs[-len(env.own_features) :]
    other_obs = obs[: -len(env.own_features)].reshape(env.num_agents - 1, -1)
    idx = jnp.where(agent.startswith("ally"), env.num_allies - 1, env.num_enemies - 1)
    order = jnp.where(agent.startswith("ally"), 1, -1)
    my_team = other_obs[:idx][::order]
    other_team = other_obs[idx:][::order]
    return self_obs, my_team, other_team

def enemy_in_sight(state, obs, agent, env):
    _, _, other_team = see_teams(obs, agent, env)
    other_pos = other_team[:, 1:3]
    dists = jnp.linalg.norm(other_pos, axis=1)
    in_sight = jnp.where(dists < 1., True, False)
    other_health = other_team[:, 0]
    targets = jnp.where(jnp.logical_and(in_sight, other_health > 0), True, False)
    status = jnp.where(jnp.any(targets > 0), SUCCESS, FAILURE)
    return status

def enemy_in_range(state, obs, agent, env):
    self_obs, my_team, other_team = see_teams(obs, agent, env)
    self_pos = self_obs[1:3]
    other_pos = other_team[:, 1:3]
    agent_id = env.agent_ids[agent]
    unit_type = state.unit_types[agent_id]
    sight_range = env.unit_type_sight_ranges[unit_type] 
    attack_range = env.unit_type_attack_ranges[unit_type] 
    # dist to others
    dists = jnp.linalg.norm(other_pos, axis=1)
    in_range = jnp.where(dists < attack_range/sight_range, True, False)
    _, _, other_team = see_teams(obs, agent, env)
    other_health = other_team[:, 0]
    targets = jnp.where(jnp.logical_and(in_range, other_health > 0), True, False)
    status = jnp.where(jnp.any(targets > 0), SUCCESS, FAILURE)
    return status


# -

def am_dying(state, obs, agent, env):  # is my health below a certain threshold?
    thresh = 0.33 
    return jnp.where(obs[-len(env.own_features)] < thresh, SUCCESS, FAILURE)


# +
def quarter(obs, env):
    quarter_matrix = jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]) 
                             # [["NW", "N", "NE"], ["W", "C", "E"], ["SW", "S", "SE"]])
    self_obs = obs[-len(env.own_features) :]
    self_pos = self_obs[1:3]
    is_west = jnp.where(self_pos[0] < 1./3, True, False)
    is_east = jnp.where(self_pos[0] > 2./3, True, False)
    is_south = jnp.where(self_pos[1] < 1./3, True, False)
    is_north = jnp.where(self_pos[1] > 2./3, True, False)
    vertical = jnp.where(is_north, 0, jnp.where(is_south, 2, 1))
    horizontal = jnp.where(is_west, 0, jnp.where(is_east, 2, 1))
    return quarter_matrix[vertical, horizontal]

def am_east(state, obs, self_agent, env):
    return jnp.where(quarter(obs, env) == 5, SUCCESS, FAILURE)

def am_west(state, obs, self_agent, env):
    return jnp.where(quarter(obs, env) == 3, SUCCESS, FAILURE)

def am_north(state, obs, self_agent, env):
    return jnp.where(quarter(obs, env) == 1, SUCCESS, FAILURE)

def am_south(state, obs, self_agent, env):
    return jnp.where(quarter(obs, env) == 7, SUCCESS, FAILURE)

def am_north_west(state, obs, self_agent, env):
    return jnp.where(quarter(obs, env) == 0, SUCCESS, FAILURE)

def am_south_west(state, obs, self_agent, env):
    return jnp.where(quarter(obs, env) == 6, SUCCESS, FAILURE)

def am_north_east(state, obs, self_agent, env):
    return jnp.where(quarter(obs, env) == 2, SUCCESS, FAILURE)

def am_south_east(state, obs, self_agent, env):
    return jnp.where(quarter(obs, env) == 8, SUCCESS, FAILURE)

def am_center(state, obs, self_agent, env):
    return jnp.where(quarter(obs, env) == 4, SUCCESS, FAILURE)


# -

# ## Untested / in progress

>>>>>>> 57830a75e304a9238cd81880fda2cb081d3d98fc
def locate(other_agent, direction):  # is unit x in direction y?
    def aux(state, obs, self_agent, env):
        # self and other obs
        self_obs = obs[-len(env.own_features) :]
        other_obs = obs[: -len(env.own_features)].reshape(env.num_agents - 1, -1)
        rel_pos = other_obs[:, 1:3] - self_obs[1:3]
        column = jnp.where(dir_to_idx[direction] < 2, rel_pos[:, 1], rel_pos[:, 0])
        sight = jnp.where(dir_to_idx[direction] % 2 == 0, column > 0, column < 0)
        # TODO: logical and that is has health > 0
        return jnp.where(sight[int(other_agent)], SUCCESS, FAILURE)

    return aux


def shootable(other_gent):  # is unit x in direction y?
    def aux(obs, self_agent, env):
        # self and other obs
        self_obs = obs[-len(env.own_features) :]
        other_obs = obs[: -len(env.own_features)].reshape(env.num_agents - 1, -1)
        rel_pos = other_obs[:, 1:3] - self_obs[1:3]
        dist = jnp.linalg.norm(rel_pos, axis=1)
        return jnp.where(dist[other_agent] < 0.5, SUCCESS, FAILURE)

    return aux


# atomics
def am_armed(state, obs, self_agent, env):  # or is my weapon in cooldown?
    return jnp.where(obs[-1] > 0, SUCCESS, FAILURE)


def am_exiled(state, obs, self_agent, env):  # or is the enemy too far away?
    self_pos = obs[-len(env.own_features) : -1] - 16  # 16 is map size (get from env)
    return jnp.where(jnp.linalg.norm(self_pos) < 10, SUCCESS, FAILURE)


<<<<<<< HEAD
def am_dying(state, obs, agent, env):  # is my health below a certain threshold?
    thresh = 0.25 * env.unit_type_health[state.unit_types[env.agent_ids[agent]]]
    return jnp.where(obs[-len(env.own_features)] < thresh, SUCCESS, FAILURE)


def enemy_found(state, obs, agent, env):
    self_obs, my_team, other_team = see_teams(obs, agent, env)
    self_pos = self_obs[1:3]
    other_pos = other_team[:, 1:3]
    agent_id = env.agent_ids[agent]
    unit_type = state.unit_types[agent_id]
    sight_range = env.unit_type_sight_ranges[unit_type] / 32.0
    # dist to others
    dists = jnp.linalg.norm(other_pos - self_pos, axis=1)
    in_range = jnp.where(dists < sight_range, True, False)
    _, _, other_team = see_teams(obs, agent, env)
    other_health = other_team[:, 0]
    targets = jnp.where(jnp.logical_and(in_range, other_health > 0), True, False)
    status = jnp.where(jnp.any(targets > 0), SUCCESS, FAILURE)
    return status


def find_enemy(state, obs, agent, env):
    dim_dir_matrix = jnp.array([[2, 1], [0, 3]])
    self_obs = obs[-len(env.own_features) :]
    self_pos = self_obs[1:3] - 0.75
    dimension = jnp.where(jnp.abs(self_pos[0]) > jnp.abs(self_pos[1]), 0, 1)
    direction = jnp.where(self_pos[dimension] > 0, 0, 1)
    action = dim_dir_matrix[dimension, direction]
    return (RUNNING, action)


def attack_enemy(state, obs, agent, env):
    self_obs, my_team, other_team = see_teams(obs, agent, env)
    potential_targets = (other_team[:, 0] > 0).astype(jnp.int32)
    potential_targets = jnp.concatenate([potential_targets, jnp.array([1])])
    target = jnp.argmax(potential_targets)
    target = jnp.where(target == other_team.shape[0], STAND, target + 5)
    return (RUNNING, target)


def see_teams(obs, agent, env):
    self_obs = obs[-len(env.own_features) :]
    other_obs = obs[: -len(env.own_features)].reshape(env.num_agents - 1, -1)
    idx = jnp.where(agent.startswith("ally"), env.num_allies - 1, env.num_enemies - 1)
    order = jnp.where(agent.startswith("ally"), 1, -1)
    my_team = other_obs[:idx][::order]
    other_team = other_obs[idx:][::order]
    return self_obs, my_team, other_team


def main():
    rng = jax.random.PRNGKey(1)
    env = make("SMAX", num_allies=10, num_enemies=10, walls_cause_death=False)
    obs, state = env.reset(rng)
    top_north_fn = region("north", "center")
    print(top_north_fn(obs["ally_0"], env))
=======
# # Main

def main():
    rng = jax.random.PRNGKey(1)
    env = make("SMAX", num_allies=2, num_enemies=3)
    obs, state = env.reset(rng)
    args = (state, obs["ally_0"], "ally_0", env)
    # print(attack_enemy(*args))
    print(enemy_found(*args))


if __name__ == "__main__":
    main()


>>>>>>> 57830a75e304a9238cd81880fda2cb081d3d98fc
