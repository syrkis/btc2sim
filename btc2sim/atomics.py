# atomics.py
# by: Noah Syrkis

# imports
import jax
from jax import random
import jax.numpy as jnp
from functools import partial

from .utils import dir_to_idx, STAND, NONE
from .classes import Status


# constants
unit_types = ["any", "soldier", "sniper", "swat", "tank", "turret", "drone", "civilian"]
directions = ["north", "west", "center", "east", "south"]
target_types = {
    "soldier": -6,
    "sniper": -5,
    "swat": -4,
    "turret": -3,
    "drone": -2,
    "civilian": -1,
    "any": None,
}

d2i = {
    ("north", "west"): (1, -1),
    ("north", "north"): (1, 0),
    ("north", "east"): (1, 1),
    ("west", "west"): (0, -1),
    ("center", "center"): (0, 0),
    ("east", "east"): (0, 1),
    ("south", "west"): (-1, -1),
    ("south", "south"): (-1, 0),
    ("south", "east"): (-1, 1),
}


# constants
SUCCESS, FAILURE = Status.SUCCESS, Status.FAILURE
ATOMICS = [
    "attack",
    "move",
    "stand",
    "in_region",
    "in_sight",
    "in_reach",
    "is_armed",
    "is_dying",
]


def self_type_fn(obs, info):
    return jnp.argmax(self_obs_fn(obs, info)[-info.env.num_types :])


# +
def self_obs_fn(obs, info):
    return obs[-10:]
    
def other_obs_fn(obs, info):
    return obs[:-10].reshape(-1, 13)

def process_obs(obs, info):
    return self_obs_fn(obs, info), other_obs_fn(obs, info)


# -

@partial(jax.vmap, in_axes=(None, None, 0, 0))
def inter_fn(pos, new_pos, obs, obs_end):
    d1 = jnp.cross(obs - pos, new_pos - pos)
    d2 = jnp.cross(obs_end - pos, new_pos - pos)
    d3 = jnp.cross(pos - obs, obs_end - obs)
    d4 = jnp.cross(new_pos - obs, obs_end - obs)
    return (d1 * d2 <= 0) & (d3 * d4 <= 0)


# Attacks
def attack(qualifier, *units):  # TODO: attack closest if no target
    assert qualifier in ["closest", "furthest", "strongest", "weakest"]    
    use_health = qualifier in ["strongest", "weakest"]
    use_min = qualifier in ["closest", "weakest"]   
    if len(units) == 0 or units[0] == "any":
        targeted_types = [1] * 6
    else:
        targeted_types = [0] * 6
        for unit in units:
            assert unit in unit_types
            for unit in units:
                targeted_types[target_types[unit]] = 1
    targeted_types = jnp.array(targeted_types)
    
    def attack_fn(obs, info, rng):
        fill = jnp.where(use_min, jnp.inf, -jnp.inf)
        self_obs, others_obs = process_obs(obs, info)
        n = jnp.where(
            info.agent.is_ally, info.env.num_enemies, info.env.num_allies
        )  # number of foes
        m = (
            jnp.where(info.agent.is_ally, info.env.num_allies, info.env.num_enemies) - 1
        )  # number of allies
        alive = others_obs.T[0] > 0
        is_enemies = jnp.arange(alive.size) >= (alive.size - n)
        alive = jnp.logical_and(alive, is_enemies)
        is_unit_types = others_obs.T[-6:].T.dot(targeted_types)
        alive = jnp.logical_and(alive, is_unit_types) # was
        dist = jnp.linalg.norm(others_obs.T[1:3], axis=0)
        in_reach = jnp.logical_and(
            info.agent.attack_range / info.agent.sight_range > dist, alive
        )

        health = others_obs.T[0]
        dist = jnp.where(in_reach, jnp.where(use_health, health, dist), fill)
        targ = jax.lax.cond(use_min, jnp.argmin, jnp.argmax, dist)
        alive_and_not_in_cooldown = jnp.logical_and(in_reach.any(), self_obs[3] <= 0)
        flag = jnp.where(alive_and_not_in_cooldown, SUCCESS, FAILURE)
        action = jnp.where(alive_and_not_in_cooldown, targ + 5 - m, NONE)
        return (flag, action)

    return attack_fn


# ## Move
def move(direction, qualifier=None, target=None, *units):
    if direction in ["toward", "away_from"]:  # target = another agent
        assert target in ["foe", "friend"]
        assert qualifier in ["closest", "furthest", "strongest", "weakest"]


        use_health = qualifier in ["strongest", "weakest"]
        use_min = qualifier in ["closest", "weakest"]
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

        def move_fn(obs, info, rng):
            fill = jnp.where(use_min, jnp.inf, -jnp.inf)
            n = jnp.where(
                info.agent.is_ally, info.env.num_enemies, info.env.num_allies
            )  # number of foes
            self_obs, others_obs = process_obs(obs, info)
            alive = (
                others_obs.T[0] > 0
            )  # takes health and in_sight into consideration as health = 0 if not in sight
            target_team = jnp.where(
                target_foe,
                jnp.arange(alive.size) >= (alive.size - n),
                jnp.arange(alive.size) < (alive.size - n),
            )
            alive = jnp.logical_and(alive, target_team)
            is_unit_types = others_obs.T[-6:].T.dot(targeted_types)
            alive = jnp.logical_and(alive, is_unit_types)
            dists = jnp.linalg.norm(others_obs.T[1:3], axis=0)
            health = others_obs.T[0]
            dists = jnp.where(alive, jnp.where(use_health, health, dists), fill)
            targ = jax.lax.cond(use_min, jnp.argmin, jnp.argmax, dists)
            x = others_obs[targ][1]
            y = others_obs[targ][2]
            SE = x > y
            NE = x > -y
            action = jnp.where(SE, jnp.where(NE, 1, 2), jnp.where(NE, 0, 3))
            action = jnp.where(move_toward, action, (action + 2) % 4)
            flag = jnp.where(alive.any(), SUCCESS, FAILURE)
            action = jnp.where(alive.any(), action, NONE)
            pos = self_obs[1:3] * jnp.array(
                [info.env.map_width, info.env.map_height]
            )
            vec_direction = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]])[action]
            new_pos = (
                pos
                + jnp.array(vec_direction)
                * info.agent.velocity
                * info.env.time_per_step
                * info.env.world_steps_per_env_step
            )
            clash = raster_crossing(pos, new_pos, info)
            flag = jnp.where(clash, FAILURE, flag)
            action = jnp.where(clash, NONE, action)
            return (flag, action)

        return move_fn
    else:  # target = direction or region
        if direction == "center":
            mat_to_dir = jnp.array([[1, 3], [0, 2]])

            def center_fn(obs, info, rng):
                self_obs, others_obs = process_obs(obs, info)
                self_pos = self_obs[1:3] * 32 - 16
                dimension = jnp.argmax(jnp.abs(self_pos))
                direction = jnp.where(self_pos[dimension] > 0, 1, 0)
                action = mat_to_dir[dimension, direction]
                # move on dimension with higest absolute value
                # agent_id = env.agent_ids[agent]
                # is_alive = state.unit_health[agent_id] > 0
                # action = jnp.where(is_alive, action, STAND)
                return (SUCCESS, action)

            return center_fn
        else:
            vec_direction = jnp.array(
                {"north": [0, 1], "east": [1, 0], "south": [0, -1], "west": [-1, 0]}[
                    direction
                ]
            )

            def move_fn_alt(obs, info, rng):
                self_obs, _ = process_obs(obs, info)
                pos = self_obs[1:3] * jnp.array(
                    [info.env.map_width, info.env.map_height]
                )
                new_pos = (
                    pos
                    + jnp.array(vec_direction)
                    * info.agent.velocity
                    * info.env.time_per_step
                    * info.env.world_steps_per_env_step
                )
                clash = raster_crossing(pos, new_pos, info)
                flag = jnp.where(clash, FAILURE, SUCCESS)
                motion = jnp.where(clash, NONE, dir_to_idx[direction])
                return (flag, motion)

            return move_fn_alt


# + active=""
# def follow_map(obs, info, rng):  # given an already computed gradient
#     self_obs, _ = process_obs(obs, info)
#     pos = jnp.clip(jnp.array(self_obs[1:3] * jnp.array([info.env.map_width, info.env.map_height]), dtype=jnp.int32), 0, jnp.array([info.env.map_width-1, info.env.map_height-1])) 
#     return (SUCCESS, info.agent.direction_map[pos[0], pos[1]])
# -

def follow_map(obs, info, rng):  # given the distances to the goal
    self_obs, _ = process_obs(obs, info)
    pos = jnp.clip(jnp.array(self_obs[1:3] * jnp.array([info.env.map_width, info.env.map_height]), dtype=jnp.int32), 0, jnp.array([info.env.map_width-1, info.env.map_height-1])) 
    current_distance = info.agent.direction_map[pos[0], pos[1]]
    north_distance = jnp.where(pos[1]+1<info.env.map_height, info.agent.direction_map[pos[0], pos[1]+1], jnp.inf)
    south_distance = jnp.where(pos[1]-1>=0, info.agent.direction_map[pos[0], pos[1]-1], jnp.inf)
    east_distance = jnp.where(pos[0]+1<info.env.map_width, info.agent.direction_map[pos[0]+1, pos[1]], jnp.inf)
    west_distance = jnp.where(pos[0]-1>=0, info.agent.direction_map[pos[0]-1, pos[1]], jnp.inf)
    distances = jnp.array([north_distance, east_distance, south_distance, west_distance, current_distance]) 
    action = jnp.where(jnp.min(distances) == jnp.max(distances), 4, jnp.arange(5)[jnp.argmin(distances + random.uniform(rng, (5,), minval=0.0, maxval=0.5))])  # stand if the map is uniform 
    flag = jnp.where(jnp.min(distances) == jnp.max(distances), FAILURE, SUCCESS)
    return (flag, action)  # actions [0,1,2,3,4] == [↑, →, ↓, ←, ∅]


# ## Stand
def stand(obs, info, rng):
    return (SUCCESS, STAND)


# # conditions
# ## Regions location
def in_region(x, y=None):  # only applies to self
    y = x if y is None else y  # in_region center instead of in_region center center
    target_row, target_col = d2i[(x, y)]

    def in_region_fn(obs, info, rng):
        self_pos = obs[-info.env.num_own_features :][1:3]
        col = jnp.where(self_pos[0] > 2 / 3, 1, jnp.where(self_pos[0] < 1 / 3, -1, 0))
        row = jnp.where(self_pos[1] > 2 / 3, 1, jnp.where(self_pos[1] < 1 / 3, -1, 0))
        flag = jnp.logical_and(row == target_row, col == target_col)
        return jnp.where(flag, SUCCESS, FAILURE)

    return in_region_fn


# In sight
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
    
    def in_sight_fn(obs, info, rng):
        n = jnp.where(
            info.agent.is_ally, info.env.num_enemies, info.env.num_allies
        )  # number of foes
        self_obs, others_obs = process_obs(obs, info)
        alive = others_obs.T[0] > 0
        target_team = jnp.where(
            target_foe,
            jnp.arange(alive.size) >= (alive.size - n),
            jnp.arange(alive.size) < (alive.size - n),
        )
        alive = jnp.logical_and(alive, target_team)
        is_unit_types = others_obs.T[-6:].T.dot(targeted_types)
        alive = jnp.logical_and(alive, is_unit_types)
        enemies_flag = alive.any()
        return jnp.where(enemies_flag, SUCCESS, FAILURE)

    return in_sight_fn


# ## In reach
def in_reach(other_agent, *units):  # in shooting range
    assert other_agent in ["foe", "friend"]
    on_foe = other_agent == "foe"
    if len(units) == 0 or units[0] == "any":
        targeted_types = [1] * 6
    else:
        targeted_types = [0] * 6
        for unit in units:
            assert unit in unit_types
            for unit in units:
                targeted_types[target_types[unit]] = 1
    targeted_types = jnp.array(targeted_types)

    def in_reach_fn(obs, info, rng):
        n = jnp.where(
            info.agent.is_ally, info.env.num_enemies, info.env.num_allies
        )  # number of foes
        self_obs, others_obs = process_obs(obs, info)
        alive = others_obs.T[0] > 0
        target_team = jnp.where(
            on_foe,
            jnp.arange(alive.size) >= (alive.size - n),
            jnp.arange(alive.size) < (alive.size - n),
        )
        alive = jnp.logical_and(alive, target_team)
        is_unit_types = others_obs.T[-6:].T.dot(targeted_types)
        alive = jnp.logical_and(alive, is_unit_types)
        dist = jnp.linalg.norm(others_obs.T[1:3], axis=0)
        in_range = info.agent.attack_range / info.agent.sight_range > dist
        flag = (jnp.logical_and(in_range, alive)).any()
        return jnp.where(flag, SUCCESS, FAILURE)

    return in_reach_fn


# ## is armed
def is_armed(agent):
    on_self = agent == "self"
    on_foe = agent == "foe"  # used only if not on_self

    def is_armed_fn(obs, info, rng):
        self_obs, others_obs = process_obs(obs, info)
        alive = others_obs.T[0] > 0
        n = jnp.where(
            info.agent.is_ally, info.env.num_enemies, info.env.num_allies
        )  # number of foes
        target_team = jnp.where(
            on_foe,
            jnp.arange(alive.size) >= (alive.size - n),
            jnp.arange(alive.size) < (alive.size - n),
        )
        alive = jnp.logical_and(alive, target_team)
        other_cooldown = jnp.where(alive, others_obs.T[6], -jnp.inf)
        other_check = jnp.where(jnp.max(other_cooldown) <= 0, SUCCESS, FAILURE)
        self_check = jnp.where(self_obs[3] <= 0, SUCCESS, FAILURE)
        return jnp.where(on_self, self_check, other_check)

    return is_armed_fn


# ## Is dying
def is_dying(agent, hp_level):
    assert hp_level in ["low", "middle", "high"]
    on_self = agent == "self"
    on_foe = agent == "foe"  # used only if not on_self
    threshold = {"low": 0.25, "middle": 0.5, "high": 0.75}[hp_level]

    def aux(obs, info, rng):
        self_obs, others_obs = process_obs(obs, info)
        alive = others_obs.T[0] > 0
        n = jnp.where(
            info.agent.is_ally, info.env.num_enemies, info.env.num_allies
        )  # number of foes
        target_team = jnp.where(
            on_foe,
            jnp.arange(alive.size) >= (alive.size - n),
            jnp.arange(alive.size) < (alive.size - n),
        )
        alive = jnp.logical_and(alive, target_team)
        other_health = jnp.where(alive, others_obs.T[0], jnp.inf)
        other_check = jnp.where(jnp.min(other_health) < threshold, SUCCESS, FAILURE)
        self_check = jnp.where(self_obs[0] < threshold, SUCCESS, FAILURE)
        return jnp.where(on_self, self_check, other_check)

    return aux


# ## is type
def is_type(negation, unit):
    assert unit in ["marine", "marauder", "stalker", "zealot", "zergling", "hydralisk"]
    assert negation in ["a", "not_a"]
    target_type = target_types[unit]
    true_condition = SUCCESS if negation == "a" else FAILURE
    false_condition = FAILURE if negation == "a" else SUCCESS

    def aux(obs, info, rng):
        return jnp.where(obs[target_type] == 1, true_condition, false_condition)

    return aux


# ## in flock
def is_flock(team, direction):
    on_foe = team == "foe"
    if direction == "center":

        def is_flock_fn(obs, info, rng):
            self_obs, others_obs = process_obs(obs, info)
            alive = others_obs.T[0] > 0
            n = jnp.where(
                info.agent.is_ally, info.env.num_enemies, info.env.num_allies
            )  # number of enemies
            target_team = jnp.where(
                on_foe,
                jnp.arange(alive.size) >= (alive.size - n),
                jnp.arange(alive.size) < (alive.size - n),
            )
            alive = jnp.logical_and(alive, target_team)
            x = jnp.where(alive, others_obs.T[1], 0)
            y = jnp.where(alive, others_obs.T[2], 0)
            SE = x > y
            NE = x > (-1 * y)
            SW = x < (-1 * y)
            NW = x < y
            N = jnp.logical_and(NE, NW).any()
            E = jnp.logical_and(NE, SE).any()
            S = jnp.logical_and(SE, SW).any()
            W = jnp.logical_and(SW, NW).any()
            status = jnp.logical_and(jnp.logical_and(N, S), jnp.logical_and(E, W))
            return jnp.where(alive.any(), status, FAILURE)

        return is_flock_fn
    else:
        assert direction in ["north", "west", "east", "south"]
        is_north = SUCCESS if direction == "north" else FAILURE
        is_west = SUCCESS if direction == "west" else FAILURE
        is_east = SUCCESS if direction == "east" else FAILURE
        is_south = SUCCESS if direction == "south" else FAILURE

        def is_flock_fn_alt(obs, info, rng):
            self_obs, others_obs = process_obs(obs, info)
            alive = others_obs.T[0] > 0
            n = jnp.where(
                info.agent.is_ally, info.env.num_enemies, info.env.num_allies
            )  # number of enemies
            target_team = jnp.where(
                on_foe,
                jnp.arange(alive.size) >= (alive.size - n),
                jnp.arange(alive.size) < (alive.size - n),
            )
            alive = jnp.logical_and(alive, target_team)
            x = jnp.mean(jnp.where(alive, others_obs.T[1], 0))
            y = jnp.mean(jnp.where(alive, others_obs.T[2], 0))
            SE = x > y
            NE = x > -y
            status = jnp.where(
                SE, jnp.where(NE, is_east, is_south), jnp.where(NE, is_north, is_west)
            )
            return jnp.where(alive.any(), status, FAILURE)

        return is_flock_fn_alt


# ## has_obstacle or out of bound
def raster_crossing(pos, new_pos, info):
    mask = info.env.terrain.building + info.env.terrain.water
    out_of_map = jnp.logical_or(jnp.min(new_pos) < 0,  jnp.max(new_pos) >= mask.shape[0])
    pos, new_pos = pos.astype(jnp.int32), new_pos.astype(jnp.int32)
    minimum = jnp.minimum(pos, new_pos)
    maximum = jnp.maximum(pos, new_pos)
    mask = jnp.where(jnp.arange(mask.shape[0]) >= minimum[0], mask.T, 0).T
    mask = jnp.where(jnp.arange(mask.shape[0]) <= maximum[0], mask.T, 0).T
    mask = jnp.where(jnp.arange(mask.shape[1]) >= minimum[1], mask, 0)
    mask = jnp.where(jnp.arange(mask.shape[1]) <= maximum[1], mask, 0)
    return jnp.logical_or(jnp.any(mask), out_of_map)


def has_obstacle(direction):
    assert direction in directions

    vec_direction = jnp.array(
        {"north": [0, 1], "east": [1, 0], "south": [0, -1], "west": [-1, 0]}[direction]
    )

    def has_obstacle_fn(obs, info, rng):
        self_obs, _ = process_obs(obs, info)
        pos = self_obs[1:3] * jnp.array([info.env.map_width, info.env.map_height])
        new_pos = (
            pos
            + jnp.array(vec_direction)
            * info.agent.velocity
            * info.env.time_per_step
            * info.env.world_steps_per_env_step
        )
        clash = raster_crossing(pos, new_pos, info)
        return jnp.where(clash, SUCCESS, FAILURE)

    return has_obstacle_fn


def is_in_forest(obs, info, rng):
    self_obs, _ = process_obs(obs, info)
    pos = self_obs[1:3] * jnp.array([info.env.map_width, info.env.map_height])
    pos = pos.astype(jnp.int32)
    return jnp.where(info.env.terrain.forest[pos[0], pos[1]], SUCCESS, FAILURE)
