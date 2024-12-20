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

unit_types = ["any", "spearmen", "archer", "cavalry", "balista", "dragon", "civilian"]
target_types = {
    "spearmen": 0,
    "archer": 1,
    "cavalry": 2,
    "balista": 3,
    "dragon": 4,
    "civilian": 5,
    "any": None,
}

# constants
SUCCESS, FAILURE = Status.SUCCESS, Status.FAILURE


# + active=""
# atomic :
#     | move  # DONE 
#     | attack  # DONE
#     | stand  # DONE 
#     | follow_map # DONE 
#     | in_sight  # DONE 
#     | in_reach  # DONE 
#     | is_dying # DONE
#     | is_armed  # DONE
#     | is_flock
#     | is_type  # DONE 
#     | is_in_forest  # DONE 
# -

# ## auxiliary functions

def compute_distance(agent_id, state, rejected_units_value=jnp.inf):
    dist_matrix = jnp.linalg.norm(state.unit_positions[agent_id]-state.unit_positions, axis=-1)
    dist_matrix = dist_matrix.at[agent_id].set(rejected_units_value)
    return dist_matrix


def has_line_of_sight(obstacles, source, target, env):  
    # suppose that the target position is in sight_range of source, otherwise the line of sight might miss some cells
    current_line_of_sight = source[:, jnp.newaxis] * (1-env.line_of_sight) + env.line_of_sight * target[:, jnp.newaxis]
    cells = jnp.array(current_line_of_sight, dtype=jnp.int32)
    in_sight = obstacles[cells[0], cells[1]].sum() == 0
    return in_sight


# # Actions

# ## Stand

def stand(env, scenario, state, rng, agent_id):
    return (SUCCESS, Stand_action)


# ## Attack

def attack(qualifier, *units):  # TODO: attack closest if no target
    assert qualifier in ["closest", "farthest", "strongest", "weakest", "random"]    
    use_health = qualifier in ["strongest", "weakest"]
    use_min = qualifier in ["closest", "weakest"] 
    use_random = qualifier == "random"
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

    def aux(env, scenario, state, rng, agent_id):
        dist_matrix = in_reach_units_factory("them_from_me", True)(env, scenario, state, rng, agent_id, targeted_types, rejected_units_value)
        if use_random:
            value = jnp.where(dist_matrix != rejected_units_value, 1, 0)
            value += random.uniform(rng, value.shape)*0.5
            target_id = jnp.argmax(value)
        elif use_health:
            health = jnp.where(dist_matrix != rejected_units_value, state.unit_health, rejected_units_value)
            health += random.uniform(rng, health.shape)*0.5
            target_id = jnp.argmin(health) if use_min else jnp.argmax(health)
        else:
            target_id = jnp.argmin(dist_matrix) if use_min else jnp.argmax(dist_matrix)
        flag = jnp.where(dist_matrix[target_id] != rejected_units_value, SUCCESS, FAILURE)
        flag = jnp.where(state.unit_cooldowns[agent_id] <= 0, flag, FAILURE)  # only attack if not in cooldown 
        return flag, Action(kind=jnp.where(flag == SUCCESS, ATTACK, NONE), value=jnp.array([target_id, 0], dtype=jnp.float32))  # the second paramter of the value is not used at the moment

    return aux 


# ## Move

def move(direction, qualifier, target, *units):  # TODO the units types 
    assert target in ["foe", "friend"]
    assert qualifier in ["closest", "farthest", "strongest", "weakest", "random"]

    use_health = qualifier in ["strongest", "weakest"]
    use_min = qualifier in ["closest", "weakest"]
    use_random = qualifier == "random"
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
        dist_matrix = in_sight_units_factory(target_foe)(env, scenario, state, rng, agent_id, targeted_types, rejected_units_value)
        if use_random:
            value = jnp.where(dist_matrix != rejected_units_value, 1, 0)
            value += random.uniform(rng, value.shape)*0.5
            target_id = jnp.argmax(value)
        elif use_health:
            health = jnp.where(dist_matrix != rejected_units_value, state.unit_health, rejected_units_value)
            health += random.uniform(rng, health.shape)*0.5
            target_id = jnp.argmin(health) if use_min else jnp.argmax(health)
        else:
            target_id = jnp.argmin(dist_matrix) if use_min else jnp.argmax(dist_matrix)
        
        delta = state.unit_positions[target_id] - state.unit_positions[agent_id]
        delta = delta if move_toward else -delta
        norm = jnp.linalg.norm(delta)
        velocity = env.unit_type_velocities[scenario.unit_types[agent_id]]
        delta = jnp.where(norm<=velocity, delta, velocity*delta/norm)
        obstacles = (scenario.terrain.building + scenario.terrain.water)  # cannot cross building and water 
        can_move_toward_closest_target = has_line_of_sight(obstacles, state.unit_positions[agent_id], state.unit_positions[agent_id]+delta, env)
        flag = jnp.logical_and(can_move_toward_closest_target, dist_matrix[target_id] != rejected_units_value)
        flag = jnp.where(flag, SUCCESS, FAILURE)
        return flag, Action(kind=jnp.where(flag == SUCCESS, MOVE, NONE), value=jnp.where(flag == SUCCESS, delta, jnp.zeros(2)))
        
    return atomic_fn


# ## Follow map

def follow_map(sense, distance=None):
    assert sense in ["toward", "away_from"]
    assert distance in [None, "low", "middle", "high"]
    
    toward = sense == "toward"
    n_direction = 8  # number of direction arround the unit (2pi/n_direction)
    n_step_size = 4  # number of steps in the direction up to the unit's velocity (should be at least equal to the max velocity so that it check every cells 
    def aux(env, scenario, state, rng, agent_id):  
        candidates = jnp.array([[0,0]] + [ [step_size/n_step_size*jnp.cos(2*jnp.pi*theta/n_direction), step_size/n_step_size*jnp.sin(2*jnp.pi*theta/n_direction)] for theta in jnp.arange(n_direction) for step_size in jnp.arange(1, n_step_size+1)])
        candidates *= env.unit_type_velocities[scenario.unit_types[agent_id]]
        candidates_idx = jnp.array(state.unit_positions[agent_id] + candidates, dtype=jnp.uint32)
        candidates_idx = jnp.clip(candidates_idx, 0, env.size-1)
        distances = scenario.distance_map[scenario.unit_target_position_id[agent_id]][candidates_idx[:,0], candidates_idx[:,1]]
        distances += random.uniform(rng, distances.shape, minval=0.0, maxval=scenario.movement_randomness)  # to resolve tighs and give a more organic vibe 
        obstacles = (scenario.terrain.building + scenario.terrain.water)  # cannot walk through building and water
        in_sight = vmap(has_line_of_sight, in_axes=(None, None, 0, None))(obstacles, state.unit_positions[agent_id], state.unit_positions[agent_id] + candidates, env)
        distances = jnp.where(in_sight, distances, env.size**2)  # in sight positions
        if not toward:
            distances = jnp.where(distances >= env.size**2, -1, distances)
            if distance is None:
                d = jnp.inf
            elif distance == "low":
                d = env.unit_type_sight_ranges[scenario.unit_types[agent_id]]/4
            elif distance == "middle":
                d = env.unit_type_sight_ranges[scenario.unit_types[agent_id]]/2
            else:
                d = env.unit_type_sight_ranges[scenario.unit_types[agent_id]]
            flag = jnp.where(distances[0] < d, SUCCESS, FAILURE)  # continue to move if not far enough
        else:
            if distance is None:
                d = 0.
            elif distance == "low":
                d = env.unit_type_sight_ranges[scenario.unit_types[agent_id]]/4
            elif distance == "middle":
                d = env.unit_type_sight_ranges[scenario.unit_types[agent_id]]/2
            else:
                d = env.unit_type_sight_ranges[scenario.unit_types[agent_id]]
            flag = jnp.where(distances[0] > d, SUCCESS, FAILURE)  # continue to move if not close enough
        return flag, Action(kind=jnp.where(flag==SUCCESS, MOVE, NONE), value=candidates[jnp.argmin(distances) if toward else jnp.argmax(distances)])
    return aux 


# ## Debug actions

def success_action(env, scenario, state, rng, agent_id):
    unit_types = scenario.unit_types[agent_id]
    velocity = env.unit_type_velocities[unit_types]
    return SUCCESS, Action(kind=MOVE, value=jnp.array([0., velocity]))


def failure_action(env, scenario, state, rng, agent_id):
    unit_types = scenario.unit_types[agent_id]
    velocity = env.unit_type_velocities[unit_types]
    return FAILURE, Action(kind=NONE, value=jnp.array([0., -velocity]))


# # Conditions

# ## in sight

# + active=""
# # compute dist_matrix on the fly 
# def in_sight_units_factory(target_foe):
#     def in_sight_units(env, scenario, state, rng, agent_id, targeted_types, rejected_units_value):
#         unit_types = scenario.unit_types[agent_id]
#         unit_team = scenario.unit_team[agent_id]
#         dist_matrix = compute_distance(agent_id, state, rejected_units_value)  # distance on the whole map
#         concerned_units = scenario.unit_team != scenario.unit_team[agent_id] if target_foe else scenario.unit_team == scenario.unit_team[agent_id]
#         dist_matrix = jnp.where(concerned_units, dist_matrix, rejected_units_value)  # concerned team
#         dist_matrix = jnp.where(targeted_types[scenario.unit_types], dist_matrix, rejected_units_value)  # concerned type
#         dist_matrix = jnp.where(dist_matrix <= env.unit_type_sight_ranges[unit_types], dist_matrix, rejected_units_value)  # in sight distance
#         dist_matrix = jnp.where(state.unit_health > 0, dist_matrix, rejected_units_value)  # alive units
#         obstacles = (scenario.terrain.building + scenario.terrain.forest)  # cannot see through building and forest
#         in_sight = vmap(has_line_of_sight, in_axes=(None, None, 0, None))(obstacles, state.unit_positions[agent_id], state.unit_positions, env)
#         dist_matrix = jnp.where(in_sight, dist_matrix, rejected_units_value)  # in line of sight (no obstacle)
#         return dist_matrix
#     return in_sight_units
# -

# use precomputed dist_matrix
def in_sight_units_factory(target_foe):
    def in_sight_units(env, scenario, state, rng, agent_id, targeted_types, rejected_units_value):
        dist_matrix = state.unit_in_sight_distance[agent_id]
        dist_matrix = jnp.where(dist_matrix == jnp.inf, rejected_units_value, dist_matrix)  # in the step function we set jnp.inf when not alive or not in sight 
        concerned_units = scenario.unit_team != scenario.unit_team[agent_id] if target_foe else scenario.unit_team == scenario.unit_team[agent_id]
        dist_matrix = jnp.where(concerned_units, dist_matrix, rejected_units_value)  # concerned team
        dist_matrix = jnp.where(targeted_types[scenario.unit_types], dist_matrix, rejected_units_value)  # concerned type
        return dist_matrix
    return in_sight_units


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
        dist_matrix = in_sight_units_factory(target_foe)(env, scenario, state, rng, agent_id, targeted_types, rejected_units_value)
        flag = jnp.where(jnp.any(dist_matrix < rejected_units_value), SUCCESS, FAILURE)
        return flag

    return in_sight_fn


# ## in reach

# + active=""
# # compute dist_matrix on the fly 
# def in_reach_units_factory(source, target_foe, time_delay=0):
#     def in_reach_units(env, scenario, state, rng, agent_id, targeted_types, rejected_units_value):
#         unit_type = scenario.unit_types[agent_id]
#         unit_team = scenario.unit_team[agent_id]
#         dist_matrix = compute_distance(agent_id, state, rejected_units_value)  # distance on the whole map
#         concerned_units = jnp.where(target_foe, scenario.unit_team != unit_team, scenario.unit_team == unit_team)
#         dist_matrix = jnp.where(concerned_units, dist_matrix, rejected_units_value)  # concerned team
#         dist_matrix = jnp.where(targeted_types[scenario.unit_types], dist_matrix, rejected_units_value)  # concerned type
#         if source == "them_from_me":
#             dist_matrix = jnp.where(dist_matrix <= env.unit_type_attack_ranges[unit_type], dist_matrix, rejected_units_value)  # in attack range
#         else:
#             dist_matrix = jnp.where(dist_matrix <= (env.unit_type_attack_ranges[scenario.unit_types] + env.unit_type_velocities[scenario.unit_types]*time_delay), dist_matrix, rejected_units_value)  # in attack range
#         dist_matrix = jnp.where(state.unit_health > 0, dist_matrix, rejected_units_value)  # alive units
#         obstacles = (scenario.terrain.building + scenario.terrain.forest)  # cannot see through buildings and forest
#         in_sight = vmap(has_line_of_sight, in_axes=(None, None, 0, None))(obstacles, state.unit_positions[agent_id], state.unit_positions, env)
#         dist_matrix = jnp.where(in_sight, dist_matrix, rejected_units_value)  # in line of sight (no obstacle)
#         return dist_matrix
#     return in_reach_units
# -

# use precomputed dist_matrix
def in_reach_units_factory(source, target_foe, time_delay=0):
    def in_reach_units(env, scenario, state, rng, agent_id, targeted_types, rejected_units_value):
        dist_matrix = state.unit_in_sight_distance[agent_id]
        dist_matrix = jnp.where(dist_matrix == jnp.inf, rejected_units_value, dist_matrix)  # in the step function we set jnp.inf when not alive or not in sight 
        concerned_units = scenario.unit_team != scenario.unit_team[agent_id] if target_foe else scenario.unit_team == scenario.unit_team[agent_id]
        dist_matrix = jnp.where(concerned_units, dist_matrix, rejected_units_value)  # concerned team
        dist_matrix = jnp.where(targeted_types[scenario.unit_types], dist_matrix, rejected_units_value)  # concerned type
        if source == "them_from_me":
            dist_matrix = jnp.where(dist_matrix <= env.unit_type_attack_ranges[scenario.unit_types[agent_id]], dist_matrix, rejected_units_value)  # in attack range
        else:
            dist_matrix = jnp.where(dist_matrix <= (env.unit_type_attack_ranges[scenario.unit_types] + env.unit_type_velocities[scenario.unit_types]*time_delay), dist_matrix, rejected_units_value)  # in attack range
        return dist_matrix
    return in_reach_units


def in_reach(target, source, time, *units):  # is unit x in direction y?
    assert target in ["foe", "friend"]
    assert time in ["now", "low", "middle", "high"]
    assert source in ["them_from_me", "me_from_them"]
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
    time_delay = {"now": 0, "low": 1, "middle": 2, "high": 3}[time]
    
    def aux(env, scenario, state, rng, agent_id):
        dist_matrix = (in_reach_units_factory(source, target_foe, time_delay))(env, scenario, state, rng, agent_id, targeted_types, rejected_units_value)
        flag = jnp.where(jnp.any(dist_matrix < rejected_units_value), SUCCESS, FAILURE)
        return flag
    return aux


# ## is type

# ## is type
def is_type(negation, unit):
    assert unit in unit_types
    assert negation in ["a", "not_a"]
    target_type = target_types[unit]
    true_condition = SUCCESS if negation == "a" else FAILURE
    false_condition = FAILURE if negation == "a" else SUCCESS

    def aux(env, scenario, state, rng, agent_id):
        return jnp.where(scenario.unit_types[agent_id] == target_type, true_condition, false_condition)

    return aux


# ## is in forest

def is_in_forest(env, scenario, state, rng, agent_id):
    pos = state.unit_positions[agent_id].astype(jnp.uint32)
    return jnp.where(scenario.terrain.forest[pos[0], pos[1]], SUCCESS, FAILURE)


# ## is armed

# ## is armed
def is_armed(agent, *units):
    on_self = agent == "self"
    target_foe = agent == "foe"  # used only if not on_self
    rejected_units_value = jnp.inf
    if len(units) == 0 or units[0] == "any":
        targeted_types = [1] * 6
    else:
        targeted_types = [0] * 6
        for unit in units:
            assert unit in unit_types
            for unit in units:
                targeted_types[target_types[unit]] = 1
    targeted_types = jnp.array(targeted_types)
    if on_self:
        def aux(env, scenario, state, rng, agent_id):
            return jnp.where(state.unit_cooldowns[agent_id] <= 0, SUCCESS, FAILURE)
        return aux 
    else:
        def is_armed_fn(env, scenario, state, rng, agent_id):
            dist_matrix = in_sight_units_factory(target_foe)(env, scenario, state, rng, agent_id, targeted_types, rejected_units_value)
            cooldown = jnp.where(dist_matrix != rejected_units_value, state.unit_cooldowns, 0)
            return jnp.where(jnp.logical_and(jnp.any(dist_matrix < rejected_units_value), jnp.max(cooldown) <= 0), SUCCESS, FAILURE)
    
        return is_armed_fn


# ## is dying

# ## Is dying
def is_dying(agent, hp_level, *units):
    assert hp_level in ["low", "middle", "high"]
    on_self = agent == "self"
    target_foe = agent == "foe"  # used only if not on_self
    threshold = {"low": 0.25, "middle": 0.5, "high": 0.75}[hp_level]
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
    if on_self:
        def aux(env, scenario, state, rng, agent_id):
            return jnp.where(state.unit_health[agent_id]/env.unit_type_health[scenario.unit_types[agent_id]] <= threshold, SUCCESS, FAILURE)
        return aux 
    else:
        def aux(env, scenario, state, rng, agent_id):
            dist_matrix = in_sight_units_factory(target_foe)(env, scenario, state, rng, agent_id, targeted_types, rejected_units_value)
            health = jnp.where(dist_matrix != rejected_units_value, state.unit_health/env.unit_type_health[scenario.unit_types], 1)
            return jnp.where(jnp.logical_and(jnp.any(dist_matrix < rejected_units_value), jnp.min(health) <= threshold), SUCCESS, FAILURE)
        return aux

    return aux
