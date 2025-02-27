# # # imports

# import jax
# from jax import random
# import jax.numpy as jnp
# from functools import partial
# from .utils import NONE, STAND, MOVE, ATTACK, Action, Stand_action
# from .classes import Status
# from jax import vmap

# # # Miscaleanous

# # ## Constants

# unit_types = ["any", "spearmen", "archer", "cavalry", "balista", "dragon", "civilian"]
# target_types = {
#     "spearmen": 0,
#     "archer": 1,
#     "cavalry": 2,
#     "balista": 3,
#     "dragon": 4,
#     "civilian": 5,
#     "any": None,
# }

# # constants
# SUCCESS, FAILURE = Status.SUCCESS, Status.FAILURE


# # + active=""
# # atomic :
# #     | move  # DONE
# #     | attack  # DONE
# #     | stand  # DONE
# #     | follow_map # DONE
# #     | in_sight  # DONE
# #     | in_reach  # DONE
# #     | is_dying # DONE
# #     | is_armed  # DONE
# #     | is_flock
# #     | is_type  # DONE
# #     | is_in_forest  # DONE
# # -

# # ## auxiliary functions

<<<<<<< HEAD
# # +
# <<<<<<< HEAD
# def self_obs_fn(obs, info):
#     return obs[-10:]


# def other_obs_fn(obs, info):
#     return obs[:-10].reshape(-1, 13)
=======
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
>>>>>>> b8225c3118f3d5bf4a9a92a54b8a9fd515f5d608


# def process_obs(obs, info):
#     return self_obs_fn(obs, info), other_obs_fn(obs, info)
# =======
# def has_line_of_sight(obstacles, source, target, env):
#     # suppose that the target position is in sight_range of source, otherwise the line of sight might miss some cells
#     current_line_of_sight = source[:, jnp.newaxis] * (1-env.line_of_sight) + env.line_of_sight * target[:, jnp.newaxis]
#     cells = jnp.array(current_line_of_sight, dtype=jnp.int32)
#     in_sight = obstacles[cells[0], cells[1]].sum() == 0
#     return in_sight

<<<<<<< HEAD
# def compute_distance(agent_id, state, rejected_units_value=jnp.inf):
#     dist_matrix = jnp.linalg.norm(state.unit_positions[agent_id]-state.unit_positions, axis=-1)
#     dist_matrix = dist_matrix.at[agent_id].set(rejected_units_value)
#     return dist_matrix
# >>>>>>> d64dc2087f355f738b9dfa406730622c9c9943dd
=======
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
>>>>>>> b8225c3118f3d5bf4a9a92a54b8a9fd515f5d608


# # -

# <<<<<<< HEAD

# @partial(jax.vmap, in_axes=(None, None, 0, 0))
# def inter_fn(pos, new_pos, obs, obs_end):
#     d1 = jnp.cross(obs - pos, new_pos - pos)
#     d2 = jnp.cross(obs_end - pos, new_pos - pos)
#     d3 = jnp.cross(pos - obs, obs_end - obs)
#     d4 = jnp.cross(new_pos - obs, obs_end - obs)
#     return (d1 * d2 <= 0) & (d3 * d4 <= 0)
# =======
# # # Actions

<<<<<<< HEAD
# # ## Stand

# def stand(env, scenario, state, rng, agent_id):
#     return (SUCCESS, Stand_action)
# >>>>>>> d64dc2087f355f738b9dfa406730622c9c9943dd
=======
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
>>>>>>> b8225c3118f3d5bf4a9a92a54b8a9fd515f5d608


# # ## Attack

<<<<<<< HEAD
# def attack(qualifier, *units):  # TODO: attack closest if no target
# <<<<<<< HEAD
#     assert qualifier in ["closest", "furthest", "strongest", "weakest"]
#     use_health = qualifier in ["strongest", "weakest"]
#     use_min = qualifier in ["closest", "weakest"]
# =======
#     assert qualifier in ["closest", "farthest", "strongest", "weakest", "random"]
#     use_health = qualifier in ["strongest", "weakest"]
#     use_min = qualifier in ["closest", "weakest"]
#     use_random = qualifier == "random"
#     rejected_units_value = jnp.inf if use_min else -jnp.inf
#     if len(units) == 0 or units[0] == "any":
#         targeted_types = [1] * 6
#     else:
#         targeted_types = [0] * 6
#         for unit in units:
#             assert unit in unit_types
#             for unit in units:
#                 targeted_types[target_types[unit]] = 1
#     targeted_types = jnp.array(targeted_types)

#     def aux(env, scenario, state, rng, agent_id):
#         dist_matrix = in_reach_units_factory("them_from_me")(env, scenario, state, rng, agent_id, True, targeted_types, rejected_units_value)
#         if use_random:
#             value = jnp.where(dist_matrix != rejected_units_value, 1, 0)
#             value += random.uniform(rng, value.shape)*0.5
#             target_id = jnp.argmax(value)
#         elif use_health:
#             health = jnp.where(dist_matrix != rejected_units_value, state.unit_health, rejected_units_value)
#             health += random.uniform(rng, health.shape)*0.5
#             target_id = jnp.argmin(health) if use_min else jnp.argmax(health)
#         else:
#             target_id = jnp.argmin(dist_matrix) if use_min else jnp.argmax(dist_matrix)
#         flag = jnp.where(dist_matrix[target_id] != rejected_units_value, SUCCESS, FAILURE)
#         flag = jnp.where(state.unit_cooldowns[agent_id] <= 0, flag, FAILURE)  # only attack if not in cooldown
#         return flag, Action(kind=jnp.where(flag == SUCCESS, ATTACK, NONE), value=jnp.array([target_id, 0], dtype=jnp.float32))  # the second paramter of the value is not used at the moment

#     return aux
=======
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
>>>>>>> b8225c3118f3d5bf4a9a92a54b8a9fd515f5d608


# # ## Move

# def move(direction, qualifier, target, *units):  # TODO the units types
#     assert target in ["foe", "friend"]
#     assert qualifier in ["closest", "farthest", "strongest", "weakest", "random"]

#     use_health = qualifier in ["strongest", "weakest"]
#     use_min = qualifier in ["closest", "weakest"]
#     use_random = qualifier == "random"
#     rejected_units_value = jnp.inf if use_min else -jnp.inf

# >>>>>>> d64dc2087f355f738b9dfa406730622c9c9943dd
#     if len(units) == 0 or units[0] == "any":
#         targeted_types = [1] * 6
#     else:
#         targeted_types = [0] * 6
#         for unit in units:
#             assert unit in unit_types
#             for unit in units:
#                 targeted_types[target_types[unit]] = 1
#     targeted_types = jnp.array(targeted_types)
# <<<<<<< HEAD

#     def attack_fn(obs, info, rng):
#         fill = jnp.where(use_min, jnp.inf, -jnp.inf)
#         self_obs, others_obs = process_obs(obs, info)
#         n = jnp.where(
#             info.agent.is_ally, info.env.num_enemies, info.env.num_allies
#         )  # number of foes
#         m = (
#             jnp.where(info.agent.is_ally, info.env.num_allies, info.env.num_enemies) - 1
#         )  # number of allies
#         alive = others_obs.T[0] > 0
#         is_enemies = jnp.arange(alive.size) >= (alive.size - n)
#         alive = jnp.logical_and(alive, is_enemies)
#         is_unit_types = others_obs.T[-6:].T.dot(targeted_types)
#         alive = jnp.logical_and(alive, is_unit_types)  # was
#         dist = jnp.linalg.norm(others_obs.T[1:3], axis=0)
#         in_reach = jnp.logical_and(
#             info.agent.attack_range / info.agent.sight_range > dist, alive
#         )

#         health = others_obs.T[0]
#         dist = jnp.where(in_reach, jnp.where(use_health, health, dist), fill)
#         targ = jax.lax.cond(use_min, jnp.argmin, jnp.argmax, dist)
#         alive_and_not_in_cooldown = jnp.logical_and(in_reach.any(), self_obs[3] <= 0)
#         flag = jnp.where(alive_and_not_in_cooldown, SUCCESS, FAILURE)
#         action = jnp.where(alive_and_not_in_cooldown, targ + 5 - m, NONE)
#         return (flag, action)

#     return attack_fn


# # ## Move
# def move(direction, qualifier=None, target=None, *units):
#     if direction in ["toward", "away_from"]:  # target = another agent
#         assert target in ["foe", "friend"]
#         assert qualifier in ["closest", "furthest", "strongest", "weakest"]

#         use_health = qualifier in ["strongest", "weakest"]
#         use_min = qualifier in ["closest", "weakest"]
#         if len(units) == 0 or units[0] == "any":
#             targeted_types = [1] * 6
#         else:
#             targeted_types = [0] * 6
#             for unit in units:
#                 assert unit in unit_types
#                 for unit in units:
#                     targeted_types[target_types[unit]] = 1
#         targeted_types = jnp.array(targeted_types)

#         target_foe = target == "foe"
#         move_toward = direction == "toward"

#         def move_fn(obs, info, rng):
#             fill = jnp.where(use_min, jnp.inf, -jnp.inf)
#             n = jnp.where(
#                 info.agent.is_ally, info.env.num_enemies, info.env.num_allies
#             )  # number of foes
#             self_obs, others_obs = process_obs(obs, info)
#             alive = (
#                 others_obs.T[0] > 0
#             )  # takes health and in_sight into consideration as health = 0 if not in sight
#             target_team = jnp.where(
#                 target_foe,
#                 jnp.arange(alive.size) >= (alive.size - n),
#                 jnp.arange(alive.size) < (alive.size - n),
#             )
#             alive = jnp.logical_and(alive, target_team)
#             is_unit_types = others_obs.T[-6:].T.dot(targeted_types)
#             alive = jnp.logical_and(alive, is_unit_types)
#             dists = jnp.linalg.norm(others_obs.T[1:3], axis=0)
#             health = others_obs.T[0]
#             dists = jnp.where(alive, jnp.where(use_health, health, dists), fill)
#             targ = jax.lax.cond(use_min, jnp.argmin, jnp.argmax, dists)
#             x = others_obs[targ][1]
#             y = others_obs[targ][2]
#             SE = x > y
#             NE = x > -y
#             E = x > 0
#             N = y > 0
#             action = jnp.where(SE, jnp.where(NE, 1, 2), jnp.where(NE, 0, 3))
#             action = jnp.where(move_toward, action, (action + 2) % 4)

#             sub_action = jnp.where(
#                 N,
#                 jnp.where(
#                     E, jnp.where(action == 1, 0, 1), jnp.where(action == 3, 0, 3)
#                 ),
#                 jnp.where(
#                     E, jnp.where(action == 1, 2, 1), jnp.where(action == 3, 2, 3)
#                 ),
#             )
#             sub_action = jnp.where(move_toward, sub_action, (sub_action + 2) % 4)

#             flag = jnp.where(alive.any(), SUCCESS, FAILURE)

#             pos = self_obs[1:3] * jnp.array([info.env.map_width, info.env.map_height])
#             vec_direction = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]])[action]
#             new_pos = (
#                 pos
#                 + jnp.array(vec_direction)
#                 * info.agent.velocity
#                 * info.env.time_per_step
#                 * info.env.world_steps_per_env_step
#             )
#             clash = raster_crossing(pos, new_pos, info)

#             action = jnp.where(
#                 clash, sub_action, action
#             )  # test suboptimal action if there is collision with the environment

#             vec_direction = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]])[action]
#             new_pos = (
#                 pos
#                 + jnp.array(vec_direction)
#                 * info.agent.velocity
#                 * info.env.time_per_step
#                 * info.env.world_steps_per_env_step
#             )
#             clash = raster_crossing(pos, new_pos, info)
#             flag = jnp.where(clash, FAILURE, flag)
#             action = jnp.where(clash, NONE, action)
#             action = jnp.where(alive.any(), action, NONE)
#             return (flag, action)

#         return move_fn
#     else:  # target = direction or region
#         if direction == "center":
#             mat_to_dir = jnp.array([[1, 3], [0, 2]])

#             def center_fn(obs, info, rng):
#                 self_obs, others_obs = process_obs(obs, info)
#                 self_pos = self_obs[1:3] * 32 - 16
#                 dimension = jnp.argmax(jnp.abs(self_pos))
#                 direction = jnp.where(self_pos[dimension] > 0, 1, 0)
#                 action = mat_to_dir[dimension, direction]
#                 # move on dimension with higest absolute value
#                 # agent_id = env.agent_ids[agent]
#                 # is_alive = state.unit_health[agent_id] > 0
#                 # action = jnp.where(is_alive, action, STAND)
#                 return (SUCCESS, action)

#             return center_fn
#         else:
#             vec_direction = jnp.array(
#                 {"north": [0, 1], "east": [1, 0], "south": [0, -1], "west": [-1, 0]}[
#                     direction
#                 ]
#             )

#             def move_fn_alt(obs, info, rng):
#                 self_obs, _ = process_obs(obs, info)
#                 pos = self_obs[1:3] * jnp.array(
#                     [info.env.map_width, info.env.map_height]
#                 )
#                 new_pos = (
#                     pos
#                     + jnp.array(vec_direction)
#                     * info.agent.velocity
#                     * info.env.time_per_step
#                     * info.env.world_steps_per_env_step
#                 )
#                 clash = raster_crossing(pos, new_pos, info)
#                 flag = jnp.where(clash, FAILURE, SUCCESS)
#                 motion = jnp.where(clash, NONE, dir_to_idx[direction])
#                 return (flag, motion)

#             return move_fn_alt


# # + active=""
# # def follow_map(obs, info, rng):  # given an already computed gradient
# #     self_obs, _ = process_obs(obs, info)
# #     pos = jnp.clip(jnp.array(self_obs[1:3] * jnp.array([info.env.map_width, info.env.map_height]), dtype=jnp.int32), 0, jnp.array([info.env.map_width-1, info.env.map_height-1]))
# #     return (SUCCESS, info.agent.direction_map[pos[0], pos[1]])
# # -
# =======

#     target_foe = target == "foe"
#     move_toward = direction == "toward"

<<<<<<< HEAD
#     def atomic_fn(env, scenario, state, rng, agent_id):
#         dist_matrix = in_sight_units(env, scenario, state, rng, agent_id, target_foe, targeted_types, rejected_units_value)
#         if use_random:
#             value = jnp.where(dist_matrix != rejected_units_value, 1, 0)
#             value += random.uniform(rng, value.shape)*0.5
#             target_id = jnp.argmax(value)
#         elif use_health:
#             health = jnp.where(dist_matrix != rejected_units_value, state.unit_health, rejected_units_value)
#             health += random.uniform(rng, health.shape)*0.5
#             target_id = jnp.argmin(health) if use_min else jnp.argmax(health)
#         else:
#             target_id = jnp.argmin(dist_matrix) if use_min else jnp.argmax(dist_matrix)
#         flag = jnp.where(dist_matrix[target_id] != rejected_units_value, SUCCESS, FAILURE)
#         delta = state.unit_positions[target_id] - state.unit_positions[agent_id]
#         delta = delta if move_toward else -delta
#         distance = jnp.linalg.norm(delta)
#         velocity = env.unit_type_velocities[scenario.unit_types[agent_id]]
#         return flag, Action(kind=jnp.where(flag == SUCCESS, MOVE, NONE), value=jnp.where(distance<=velocity, delta, velocity*delta/distance))

#     return atomic_fn


# # ## Follow map
# >>>>>>> d64dc2087f355f738b9dfa406730622c9c9943dd


# def follow_map(sense):
#     assert sense in ["toward", "away_from"]
# <<<<<<< HEAD
#     if sense == "toward":

#         def aux(obs, info, rng):  # given the distances to the goal
#             self_obs, _ = process_obs(obs, info)
#             pos = jnp.clip(
#                 jnp.array(
#                     self_obs[1:3]
#                     * jnp.array([info.env.map_width, info.env.map_height]),
#                     dtype=jnp.int32,
#                 ),
#                 0,
#                 jnp.array([info.env.map_width - 1, info.env.map_height - 1]),
#             )
#             current_distance = info.agent.direction_map[pos[0], pos[1]]
#             north_distance = jnp.where(
#                 pos[1] + 1 < info.env.map_height,
#                 info.agent.direction_map[pos[0], pos[1] + 1],
#                 jnp.inf,
#             )
#             south_distance = jnp.where(
#                 pos[1] - 1 >= 0, info.agent.direction_map[pos[0], pos[1] - 1], jnp.inf
#             )
#             east_distance = jnp.where(
#                 pos[0] + 1 < info.env.map_width,
#                 info.agent.direction_map[pos[0] + 1, pos[1]],
#                 jnp.inf,
#             )
#             west_distance = jnp.where(
#                 pos[0] - 1 >= 0, info.agent.direction_map[pos[0] - 1, pos[1]], jnp.inf
#             )
#             distances = jnp.array(
#                 [
#                     north_distance,
#                     east_distance,
#                     south_distance,
#                     west_distance,
#                     current_distance,
#                 ]
#             )
#             action = jnp.where(
#                 jnp.min(distances) == jnp.max(distances),
#                 4,
#                 jnp.arange(5)[
#                     jnp.argmin(
#                         distances + random.uniform(rng, (5,), minval=0.0, maxval=0.5)
#                     )
#                 ],
#             )  # stand if the map is uniform
#             flag = jnp.where(jnp.min(distances) == jnp.max(distances), FAILURE, SUCCESS)
#             return (flag, action)  # actions [0,1,2,3,4] == [↑, →, ↓, ←, ∅]
=======
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
>>>>>>> b8225c3118f3d5bf4a9a92a54b8a9fd515f5d608

#         return aux
#     else:  # away from

#         def aux(obs, info, rng):  # given the distances to the goal
#             self_obs, _ = process_obs(obs, info)
#             pos = jnp.clip(
#                 jnp.array(
#                     self_obs[1:3]
#                     * jnp.array([info.env.map_width, info.env.map_height]),
#                     dtype=jnp.int32,
#                 ),
#                 0,
#                 jnp.array([info.env.map_width - 1, info.env.map_height - 1]),
#             )
#             current_distance = info.agent.direction_map[pos[0], pos[1]]
#             north_distance = jnp.where(
#                 pos[1] + 1 < info.env.map_height,
#                 info.agent.direction_map[pos[0], pos[1] + 1],
#                 jnp.inf,
#             )
#             south_distance = jnp.where(
#                 pos[1] - 1 >= 0, info.agent.direction_map[pos[0], pos[1] - 1], jnp.inf
#             )
#             east_distance = jnp.where(
#                 pos[0] + 1 < info.env.map_width,
#                 info.agent.direction_map[pos[0] + 1, pos[1]],
#                 jnp.inf,
#             )
#             west_distance = jnp.where(
#                 pos[0] - 1 >= 0, info.agent.direction_map[pos[0] - 1, pos[1]], jnp.inf
#             )
#             distances = jnp.array(
#                 [
#                     north_distance,
#                     east_distance,
#                     south_distance,
#                     west_distance,
#                     current_distance,
#                 ]
#             )
#             action = jnp.where(
#                 jnp.min(distances) == jnp.max(distances),
#                 4,
#                 jnp.arange(5)[
#                     jnp.argmax(
#                         distances + random.uniform(rng, (5,), minval=0.0, maxval=0.5)
#                     )
#                 ],
#             )  # stand if the map is uniform
#             flag = jnp.where(jnp.min(distances) == jnp.max(distances), FAILURE, SUCCESS)
#             return (flag, action)  # actions [0,1,2,3,4] == [↑, →, ↓, ←, ∅]

#         return aux
# =======
#     toward = sense == "toward"
#     n_direction = 8  # number of direction arround the unit (2pi/n_direction)
#     n_step_size = 4  # number of steps in the direction up to the unit's velocity (should be at least equal to the max velocity so that it check every cells
#     def aux(env, scenario, state, rng, agent_id):
#         candidates = jnp.array([[0,0]] + [ [step_size/n_step_size*jnp.cos(2*jnp.pi*theta/n_direction), step_size/n_step_size*jnp.sin(2*jnp.pi*theta/n_direction)] for theta in jnp.arange(n_direction) for step_size in jnp.arange(1, n_step_size+1)])
#         candidates *= env.unit_type_velocities[scenario.unit_types[agent_id]]
#         candidates_idx = jnp.array(state.unit_positions[agent_id] + candidates, dtype=jnp.uint32)
#         candidates_idx = jnp.clip(candidates_idx, 0, env.size-1)
#         distances = scenario.distance_map[scenario.unit_target_position_id[agent_id]][candidates_idx[:,0], candidates_idx[:,1]]
#         distances += random.uniform(rng, distances.shape, minval=0.0, maxval=scenario.movement_randomness)  # to resolve tighs and give a more organic vibe
#         obstacles = (scenario.terrain.building + scenario.terrain.water)  # cannot walk through building and water
#         in_sight = vmap(has_line_of_sight, in_axes=(None, None, 0, None))(obstacles, state.unit_positions[agent_id], state.unit_positions[agent_id] + candidates, env)
#         distances = jnp.where(in_sight, distances, env.size**2)  # in sight positions
#         if not toward:
#             distances = jnp.where(distances >= env.size**2, -1, distances)
#         return SUCCESS, Action(kind=MOVE, value=candidates[jnp.argmin(distances) if toward else jnp.argmax(distances)])
#     return aux
# >>>>>>> d64dc2087f355f738b9dfa406730622c9c9943dd


# # ## Debug actions

# def success_action(env, scenario, state, rng, agent_id):
#     unit_types = scenario.unit_types[agent_id]
#     velocity = env.unit_type_velocities[unit_types]
#     return SUCCESS, Action(kind=MOVE, value=jnp.array([0., velocity]))


<<<<<<< HEAD
# def failure_action(env, scenario, state, rng, agent_id):
#     unit_types = scenario.unit_types[agent_id]
#     velocity = env.unit_type_velocities[unit_types]
#     return FAILURE, Action(kind=NONE, value=jnp.array([0., -velocity]))
=======
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
>>>>>>> b8225c3118f3d5bf4a9a92a54b8a9fd515f5d608


# # # Conditions

<<<<<<< HEAD
# # ## in sight
=======
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
>>>>>>> b8225c3118f3d5bf4a9a92a54b8a9fd515f5d608

# # +
# def in_sight_units(env, scenario, state, rng, agent_id, target_foe, targeted_types, rejected_units_value):
#     unit_types = scenario.unit_types[agent_id]
#     unit_team = scenario.unit_team[agent_id]
#     dist_matrix = compute_distance(agent_id, state, rejected_units_value)  # distance on the whole map
#     concerned_units = jnp.where(target_foe, scenario.unit_team != unit_team, scenario.unit_team == unit_team)
#     dist_matrix = jnp.where(concerned_units, dist_matrix, rejected_units_value)  # concerned team
#     dist_matrix = jnp.where(targeted_types[scenario.unit_types], dist_matrix, rejected_units_value)  # concerned type
#     dist_matrix = jnp.where(dist_matrix <= env.unit_type_sight_ranges[unit_types], dist_matrix, rejected_units_value)  # in sight distance
#     dist_matrix = jnp.where(state.unit_health > 0, dist_matrix, rejected_units_value)  # alive units
#     obstacles = (scenario.terrain.building + scenario.terrain.forest)  # cannot see through building and forest
#     in_sight = vmap(has_line_of_sight, in_axes=(None, None, 0, None))(obstacles, state.unit_positions[agent_id], state.unit_positions, env)
#     dist_matrix = jnp.where(in_sight, dist_matrix, rejected_units_value)  # in line of sight (no obstacle)
#     return dist_matrix

# def in_sight(target, *units):  # is unit x in direction y?
#     assert target in ["foe", "friend"]
#     target_foe = target == "foe"
#     if len(units) == 0 or units[0] == "any":
#         targeted_types = [1] * 6
#     else:
#         targeted_types = [0] * 6
#         for unit in units:
#             assert unit in unit_types
#             for unit in units:
#                 targeted_types[target_types[unit]] = 1
#     targeted_types = jnp.array(targeted_types)
# <<<<<<< HEAD

#     def in_sight_fn(obs, info, rng):
#         n = jnp.where(
#             info.agent.is_ally, info.env.num_enemies, info.env.num_allies
#         )  # number of foes
#         self_obs, others_obs = process_obs(obs, info)
#         alive = others_obs.T[0] > 0
#         target_team = jnp.where(
#             target_foe,
#             jnp.arange(alive.size) >= (alive.size - n),
#             jnp.arange(alive.size) < (alive.size - n),
#         )
#         alive = jnp.logical_and(alive, target_team)
#         is_unit_types = others_obs.T[-6:].T.dot(targeted_types)
#         alive = jnp.logical_and(alive, is_unit_types)
#         enemies_flag = alive.any()
#         return jnp.where(enemies_flag, SUCCESS, FAILURE)
# =======
#     rejected_units_value = jnp.inf

#     def in_sight_fn(env, scenario, state, rng, agent_id):
#         dist_matrix = in_sight_units(obstacles, env, scenario, state, rng, agent_id, target_foe, targeted_types, rejected_units_value)
#         flag = jnp.where(jnp.any(dist_matrix < rejected_units_value), SUCCESS, FAILURE)
#         return flag
# >>>>>>> d64dc2087f355f738b9dfa406730622c9c9943dd

#     return in_sight_fn


# # -

# # ## in reach

# # +
# def in_reach_units_factory(source, time_delay=0):
#     def in_reach_units(env, scenario, state, rng, agent_id, target_foe, targeted_types, rejected_units_value):
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

# def in_reach(target, source, time, *units):  # is unit x in direction y?
#     assert target in ["foe", "friend"]
#     assert time in ["now", "low", "middle", "high"]
#     assert source in ["them_from_me", "me_from_them"]
#     target_foe = target == "foe"
#     if len(units) == 0 or units[0] == "any":
#         targeted_types = [1] * 6
#     else:
#         targeted_types = [0] * 6
#         for unit in units:
#             assert unit in unit_types
#             for unit in units:
#                 targeted_types[target_types[unit]] = 1
#     targeted_types = jnp.array(targeted_types)
# <<<<<<< HEAD

#     def in_reach_fn(obs, info, rng):
#         n = jnp.where(
#             info.agent.is_ally, info.env.num_enemies, info.env.num_allies
#         )  # number of foes
#         self_obs, others_obs = process_obs(obs, info)
#         alive = others_obs.T[0] > 0
#         target_team = jnp.where(
#             on_foe,
#             jnp.arange(alive.size) >= (alive.size - n),
#             jnp.arange(alive.size) < (alive.size - n),
#         )
#         alive = jnp.logical_and(alive, target_team)
#         is_unit_types = others_obs.T[-6:].T.dot(targeted_types)
#         alive = jnp.logical_and(alive, is_unit_types)
#         dist = jnp.linalg.norm(others_obs.T[1:3], axis=0)
#         in_range = info.agent.attack_range / info.agent.sight_range > dist
#         flag = (jnp.logical_and(in_range, alive)).any()
#         return jnp.where(flag, SUCCESS, FAILURE)

#     return in_reach_fn


# # ## is armed
# def is_armed(agent):
#     on_self = agent == "self"
#     on_foe = agent == "foe"  # used only if not on_self

#     def is_armed_fn(obs, info, rng):
#         self_obs, others_obs = process_obs(obs, info)
#         alive = others_obs.T[0] > 0
#         n = jnp.where(
#             info.agent.is_ally, info.env.num_enemies, info.env.num_allies
#         )  # number of foes
#         target_team = jnp.where(
#             on_foe,
#             jnp.arange(alive.size) >= (alive.size - n),
#             jnp.arange(alive.size) < (alive.size - n),
#         )
#         alive = jnp.logical_and(alive, target_team)
#         other_cooldown = jnp.where(alive, others_obs.T[6], -jnp.inf)
#         other_check = jnp.where(jnp.max(other_cooldown) <= 0, SUCCESS, FAILURE)  # type: ignore
#         self_check = jnp.where(self_obs[3] <= 0, SUCCESS, FAILURE)
#         return jnp.where(on_self, self_check, other_check)

#     return is_armed_fn


# # ## Is dying
# def is_dying(agent, hp_level):
#     assert hp_level in ["low", "middle", "high"]
#     on_self = agent == "self"
#     on_foe = agent == "foe"  # used only if not on_self
#     threshold = {"low": 0.25, "middle": 0.5, "high": 0.75}[hp_level]

#     def aux(obs, info, rng):
#         self_obs, others_obs = process_obs(obs, info)
#         alive = others_obs.T[0] > 0
#         n = jnp.where(
#             info.agent.is_ally, info.env.num_enemies, info.env.num_allies
#         )  # number of foes
#         target_team = jnp.where(
#             on_foe,
#             jnp.arange(alive.size) >= (alive.size - n),
#             jnp.arange(alive.size) < (alive.size - n),
#         )
#         alive = jnp.logical_and(alive, target_team)
#         other_health = jnp.where(alive, others_obs.T[0], jnp.inf)
#         other_check = jnp.where(jnp.min(other_health) < threshold, SUCCESS, FAILURE)  # type: ignore
#         self_check = jnp.where(self_obs[0] < threshold, SUCCESS, FAILURE)
#         return jnp.where(on_self, self_check, other_check)

# =======
#     rejected_units_value = jnp.inf
#     time_delay = {"now": 0, "low": 1, "middle": 2, "high": 3}[time]

#     def aux(env, scenario, state, rng, agent_id):
#         dist_matrix = (in_reach_units_factory(source, time_delay))(env, scenario, state, rng, agent_id, target_foe, targeted_types, rejected_units_value)
#         flag = jnp.where(jnp.any(dist_matrix < rejected_units_value), SUCCESS, FAILURE)
#         return flag
# >>>>>>> d64dc2087f355f738b9dfa406730622c9c9943dd
#     return aux


# # -

# # ## is type

# # ## is type
# def is_type(negation, unit):
#     assert unit in unit_types
#     assert negation in ["a", "not_a"]
#     target_type = target_types[unit]
#     true_condition = SUCCESS if negation == "a" else FAILURE
#     false_condition = FAILURE if negation == "a" else SUCCESS

#     def aux(env, scenario, state, rng, agent_id):
#         return jnp.where(scenario.unit_types[agent_id] == target_type, true_condition, false_condition)

#     return aux


# # ## is in forest

# def is_in_forest(env, scenario, state, rng, agent_id):
#     pos = state.unit_positions[agent_id].astype(jnp.uint32)
#     return jnp.where(scenario.terrain.forest[pos[0], pos[1]], SUCCESS, FAILURE)


# # ## is armed

# # ## is armed
# def is_armed(agent, *units):
#     on_self = agent == "self"
#     target_foe = agent == "foe"  # used only if not on_self
#     rejected_units_value = jnp.inf
#     if len(units) == 0 or units[0] == "any":
#         targeted_types = [1] * 6
#     else:
# <<<<<<< HEAD
#         assert direction in ["north", "west", "east", "south"]
#         is_north = SUCCESS if direction == "north" else FAILURE
#         is_west = SUCCESS if direction == "west" else FAILURE
#         is_east = SUCCESS if direction == "east" else FAILURE
#         is_south = SUCCESS if direction == "south" else FAILURE

#         def is_flock_fn_alt(obs, info, rng):
#             self_obs, others_obs = process_obs(obs, info)
#             alive = others_obs.T[0] > 0
#             n = jnp.where(
#                 info.agent.is_ally, info.env.num_enemies, info.env.num_allies
#             )  # number of enemies
#             target_team = jnp.where(
#                 on_foe,
#                 jnp.arange(alive.size) >= (alive.size - n),
#                 jnp.arange(alive.size) < (alive.size - n),
#             )
#             alive = jnp.logical_and(alive, target_team)
#             x = jnp.mean(jnp.where(alive, others_obs.T[1], 0))  # type: ignore
#             y = jnp.mean(jnp.where(alive, others_obs.T[2], 0))  # type: ignore
#             SE = x > y
#             NE = x > -y
#             status = jnp.where(
#                 SE, jnp.where(NE, is_east, is_south), jnp.where(NE, is_north, is_west)
#             )
#             return jnp.where(alive.any(), status, FAILURE)

#         return is_flock_fn_alt


# # ## has_obstacle or out of bound
# def raster_crossing(pos, new_pos, info):
#     mask = info.env.terrain.building + info.env.terrain.water
#     out_of_map = jnp.logical_or(jnp.min(new_pos) < 0, jnp.max(new_pos) >= mask.shape[0])
#     pos, new_pos = pos.astype(jnp.int32), new_pos.astype(jnp.int32)
#     minimum = jnp.minimum(pos, new_pos)
#     maximum = jnp.maximum(pos, new_pos)
#     mask = jnp.where(jnp.arange(mask.shape[0]) >= minimum[0], mask.T, 0).T  # type: ignore
#     mask = jnp.where(jnp.arange(mask.shape[0]) <= maximum[0], mask.T, 0).T
#     mask = jnp.where(jnp.arange(mask.shape[1]) >= minimum[1], mask, 0)
#     mask = jnp.where(jnp.arange(mask.shape[1]) <= maximum[1], mask, 0)
#     return jnp.logical_or(jnp.any(mask), out_of_map)
# =======
#         targeted_types = [0] * 6
#         for unit in units:
#             assert unit in unit_types
#             for unit in units:
#                 targeted_types[target_types[unit]] = 1
#     targeted_types = jnp.array(targeted_types)
#     if on_self:
#         def aux(env, scenario, state, rng, agent_id):
#             return jnp.where(state.unit_cooldowns[agent_id] <= 0, SUCCESS, FAILURE)
#         return aux
#     else:
#         def is_armed_fn(env, scenario, state, rng, agent_id):
#             dist_matrix = in_sight_units(env, scenario, state, rng, agent_id, target_foe, targeted_types, rejected_units_value)
#             cooldown = jnp.where(dist_matrix != rejected_units_value, state.unit_cooldowns, 0)
#             return jnp.where(jnp.logical_and(jnp.any(dist_matrix < rejected_units_value), jnp.max(cooldown) <= 0), SUCCESS, FAILURE)

#         return is_armed_fn


# # ## is dying
# >>>>>>> d64dc2087f355f738b9dfa406730622c9c9943dd

# # ## Is dying
# def is_dying(agent, hp_level, *units):
#     assert hp_level in ["low", "middle", "high"]
#     on_self = agent == "self"
#     target_foe = agent == "foe"  # used only if not on_self
#     threshold = {"low": 0.25, "middle": 0.5, "high": 0.75}[hp_level]
#     if len(units) == 0 or units[0] == "any":
#         targeted_types = [1] * 6
#     else:
#         targeted_types = [0] * 6
#         for unit in units:
#             assert unit in unit_types
#             for unit in units:
#                 targeted_types[target_types[unit]] = 1
#     targeted_types = jnp.array(targeted_types)
#     rejected_units_value = jnp.inf
#     if on_self:
#         def aux(env, scenario, state, rng, agent_id):
#             return jnp.where(state.unit_health[agent_id]/env.unit_type_health[scenario.unit_types[agent_id]] <= threshold, SUCCESS, FAILURE)
#         return aux
#     else:
#         def aux(env, scenario, state, rng, agent_id):
#             dist_matrix = in_sight_units(env, scenario, state, rng, agent_id, target_foe, targeted_types, rejected_units_value)
#             health = jnp.where(dist_matrix != rejected_units_value, state.unit_health/env.unit_type_health[scenario.unit_types], 1)
#             return jnp.where(jnp.logical_and(jnp.any(dist_matrix < rejected_units_value), jnp.min(health) <= threshold), SUCCESS, FAILURE)
#         return aux

#     return aux
