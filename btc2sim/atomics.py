# # imports

import jax
from jax import random
import jax.numpy as jnp
from functools import partial
from .utils import NONE, STAND, MOVE, ATTACK, Action, Stand_action
from .classes import Status
from jax import vmap

# # Miscaleanous

# ## Constants

unit_types = ["any", "soldier", "sniper", "swat", "turret", "drone", "civilian"]
target_types = {
    "soldier": 0,
    "sniper": 1,
    "swat": 2,
    "turret": 3,
    "drone": 4,
    "civilian": 5,
    "any": None,
}

# constants
SUCCESS, FAILURE = Status.SUCCESS, Status.FAILURE


# + active=""
# atomic :
#     | move  # DONE
#     | attack
#     | stand  # DONE 
#     | follow_map
#     | in_sight  # DONE 
#     | in_reach  # DONE 
#     | is_dying
#     | is_armed
#     | is_flock
#     | is_type 
#     | has_obstacle
#     | is_in_forest
# -

# ## auxiliary functions

def in_sight_units(env, scenario, state, rng, agent_id, target_foe, targeted_types, rejected_units_value):
    unit_type = scenario.unit_type[agent_id]
    unit_team = scenario.unit_team[agent_id]
    dist_matrix = compute_distance(agent_id, state, rejected_units_value)  # distance on the whole map
    concerned_units = jnp.where(target_foe, scenario.unit_team != unit_team, scenario.unit_team == unit_team)
    dist_matrix = jnp.where(concerned_units, dist_matrix, rejected_units_value)  # concerned team
    dist_matrix = jnp.where(targeted_types[scenario.unit_type], dist_matrix, rejected_units_value)  # concerned type
    dist_matrix = jnp.where(dist_matrix <= env.unit_type_sight_ranges[unit_type], dist_matrix, rejected_units_value)  # in sight distance
    in_sight = vmap(has_line_of_sight, in_axes=(None, 0, None, None))(state.unit_positions[agent_id], state.unit_positions, env, scenario)
    dist_matrix = jnp.where(in_sight, dist_matrix, rejected_units_value)  # in line of sight (no obstacle)
    return dist_matrix


def has_line_of_sight(source, target, env, scenario):  
    # suppose that the target is in sight_range of source, otherwise the line of sight might miss some cells
    obstacles = (scenario.terrain.building + scenario.terrain.water)
    current_line_of_sight = source[:, jnp.newaxis] * (1-env.line_of_sight) + env.line_of_sight * target[:, jnp.newaxis]
    cells = jnp.array(current_line_of_sight, dtype=jnp.int32)
    in_sight = obstacles[cells[0], cells[1]].sum() == 0
    return in_sight


def compute_distance(agent_id, state, rejected_units_value=jnp.inf):
    dist_matrix = jnp.linalg.norm(state.unit_positions[agent_id]-state.unit_positions, axis=-1)
    dist_matrix = dist_matrix.at[agent_id].set(rejected_units_value)
    return dist_matrix


# # Actions

# ## Stand

def stand(env, scenario, state, rng, agent_id):
    return (SUCCESS, Stand_action)


# ## Move

def move(direction, qualifier, target, *units):  # TODO the units types 
    assert target in ["foe", "friend"]
    assert qualifier in ["closest", "furthest", "strongest", "weakest"]

    use_health = qualifier in ["strongest", "weakest"]
    use_min = qualifier in ["closest", "weakest"]
    rejected_units_value = jnp.inf if use_min else -jnp.inf
    if len(units) == 0 or units[0] == "any":
        targeted_types = [1] * 6
    else:
        targeted_types = [0] * 6
        for unit in units:
            assert unit in unit_types
            for unit in units:
                targeted_types[target_types[unit]] = 1
    targeted_types = jnp.array(targeted_types)

    target_foe = target == "foe"
    move_toward = direction == "toward"

    def atomic_fn(env, scenario, state, rng, agent_id):
        dist_matrix = in_sight_units(env, scenario, state, rng, agent_id, target_foe, targeted_types, rejected_units_value)
        target_id = jnp.where(use_min, jnp.argmin(dist_matrix), jnp.argmax(dist_matrix))
        flag = jnp.where(dist_matrix[target_id] != rejected_units_value, SUCCESS, FAILURE)
        delta = state.unit_positions[target_id] - state.unit_positions[agent_id]
        delta = jnp.where(move_toward, delta, -delta)
        distance = jnp.linalg.norm(delta)
        velocity = env.unit_type_velocities[scenario.unit_type[agent_id]]
        return flag, Action(kind=jnp.where(flag == SUCCESS, MOVE, NONE), value=jnp.where(distance<=velocity, delta, velocity*delta/distance))
        
    return atomic_fn


def success_action(env, scenario, state, rng, agent_id):
    unit_type = scenario.unit_type[agent_id]
    velocity = env.unit_type_velocities[unit_type]
    return SUCCESS, Action(kind=MOVE, value=jnp.array([0., velocity]))


def failure_action(env, scenario, state, rng, agent_id):
    unit_type = scenario.unit_type[agent_id]
    velocity = env.unit_type_velocities[unit_type]
    return FAILURE, Action(kind=NONE, value=jnp.array([0., -velocity]))


# # Conditions

# ## in sight

def in_sight(target, *units):  # is unit x in direction y?
    assert target in ["foe", "friend"]
    target_foe = target == "foe"
    if len(units) == 0 or units[0] == "any":
        targeted_types = [1] * 6
    else:
        targeted_types = [0] * 6
        for unit in units:
            assert unit in unit_types
            for unit in units:
                targeted_types[target_types[unit]] = 1
    targeted_types = jnp.array(targeted_types)
    rejected_units_value = jnp.inf
    
    def in_sight_fn(env, scenario, state, rng, agent_id):
        dist_matrix = in_sight_units(env, scenario, state, rng, agent_id, target_foe, targeted_types, rejected_units_value)
        flag = jnp.where(jnp.any(dist_matrix < rejected_units_value), SUCCESS, FAILURE)
        return flag

    return in_sight_fn


# ## in reach

# +
def in_reach_units(env, scenario, state, rng, agent_id, target_foe, targeted_types, rejected_units_value):
    unit_type = scenario.unit_type[agent_id]
    unit_team = scenario.unit_team[agent_id]
    dist_matrix = compute_distance(agent_id, state, rejected_units_value)  # distance on the whole map
    concerned_units = jnp.where(target_foe, scenario.unit_team != unit_team, scenario.unit_team == unit_team)
    dist_matrix = jnp.where(concerned_units, dist_matrix, rejected_units_value)  # concerned team
    dist_matrix = jnp.where(targeted_types[scenario.unit_type], dist_matrix, rejected_units_value)  # concerned type
    dist_matrix = jnp.where(dist_matrix <= env.unit_type_attack_ranges[unit_type], dist_matrix, rejected_units_value)  # in sight distance
    in_sight = vmap(has_line_of_sight, in_axes=(None, 0, None, None))(state.unit_positions[agent_id], state.unit_positions, env, scenario)
    dist_matrix = jnp.where(in_sight, dist_matrix, rejected_units_value)  # in line of sight (no obstacle)
    return dist_matrix

def in_reach(target, *units):  # is unit x in direction y?
    assert target in ["foe", "friend"]
    target_foe = target == "foe"
    if len(units) == 0 or units[0] == "any":
        targeted_types = [1] * 6
    else:
        targeted_types = [0] * 6
        for unit in units:
            assert unit in unit_types
            for unit in units:
                targeted_types[target_types[unit]] = 1
    targeted_types = jnp.array(targeted_types)
    rejected_units_value = jnp.inf
    
    def in_reach_fn(env, scenario, state, rng, agent_id):
        dist_matrix = in_reach_units(env, scenario, state, rng, agent_id, target_foe, targeted_types, rejected_units_value)
        flag = jnp.where(jnp.any(dist_matrix < rejected_units_value), SUCCESS, FAILURE)
        return flag

    return in_reach_fn
