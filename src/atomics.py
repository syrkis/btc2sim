# atomics.py
#   c2sim bt molecules (complex functions)
# by: Noah Syrkis

# # imports

import jax
import jax.numpy as jnp
from jax import random, lax
from jaxmarl import make
import numpy as np
from functools import partial

from .utils import Status, dir_to_idx, idx_to_dir, STAND

# constants
SUCCESS, FAILURE, RUNNING = Status.SUCCESS, Status.FAILURE, Status.RUNNING
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
FF_DICT = {
    ("enemy", "friend"): ("enemy", lambda env: env.num_allies),
    ("enemy", "foe"): ("ally", lambda _: 0),
    ("ally", "foe"): ("enemy", lambda env: env.num_allies - 1),
    ("ally", "friend"): ("ally", lambda _: 0),
}


"""
TODO: the ids of allies and enemies are super arbitrary.
Maybe we should have the agent index agents by distance?
"""


# # helpers


def process_obs(obs, env):
    n, k = env.num_agents, len(env.own_features)
    self_obs = obs[-k:]
    others_obs = obs[:-k].reshape(n - 1, -1)
    return self_obs, others_obs

# + active=""
# def agent_info_fn(state, _, agent, env):
#     agent_id = env.agent_ids[agent]
#     agent_type = state.unit_types[agent_id]
#     sight_range = env.unit_type_sight_ranges[agent_type]
#     attack_range = env.unit_type_attack_ranges[agent_type]
#     return sight_range, attack_range
# -


# # actions

# ## Attacks


def attack(qualifier, unit="any"):  # TODO: attack closest if no target
    assert qualifier in ["closest", "furthest", "strongest", "weakest"]
    assert unit in [
        "any",
        "marine",
        "marauder",
        "stalker",
        "zealot",
        "zergling",
        "hydralisk",
    ]
    use_health = qualifier in ["strongest", "weakest"]
    use_min = qualifier in ["closest", "weakest"]
    use_unit_type = unit != "any"
    target_type = {
            "marine": -6,
            "marauder": -5,
            "stalker": -4,
            "zealot": -3,
            "zergling": -2,
            "hydralisk": -1,
            "any": None,
        }[unit]


    def attack_fn(state, obs, sight_range, attack_range, is_ally, env):
        fill = jnp.where(use_min, jnp.inf, -jnp.inf)
        self_obs, others_obs = process_obs(obs, env)
        n = jnp.where(is_ally, env.num_enemies, env.num_allies)  # number of foes
        m = jnp.where(is_ally, env.num_allies, env.num_enemies) - 1  # number of allies
        alive = others_obs.T[0] > 0
        is_enemies = jnp.arange(alive.size) >= (alive.size - n)
        is_unit_types = jnp.where(use_unit_type, others_obs.T[target_type], 1)
        alive = jnp.logical_and(jnp.logical_and(alive, is_enemies), is_unit_types)
        dist = jnp.linalg.norm(others_obs.T[1:3], axis=0)
        in_reach = jnp.logical_and(attack_range / sight_range > dist, alive)

        health = others_obs.T[0]
        dist = jnp.where(in_reach, jnp.where(use_health, health, dist), fill)
        targ = jnp.where(use_min, jnp.argmin(dist), jnp.argmax(dist))
        alive_and_not_in_cooldown = jnp.logical_and(in_reach.any(), self_obs[3] <= 0)
        flag = jnp.where(alive_and_not_in_cooldown, RUNNING, FAILURE) 
        action = jnp.where(alive_and_not_in_cooldown, targ + 5 - m, STAND)
        return (flag, action)

    return attack_fn


# ## Move


def move(direction, qualifier=None, target=None, unit="any"):
    if direction in ["toward", "away_from"]:  # target = another agent
        assert target in ["foe", "friend"]
        assert qualifier in ["closest", "furthest", "strongest", "weakest"]
        assert unit in [
            "any",
            "marine",
            "marauder",
            "stalker",
            "zealot",
            "zergling",
            "hydralisk",
        ]

        use_health = qualifier in ["strongest", "weakest"]
        use_min = qualifier in ["closest", "weakest"]
        use_unit_type = unit != "any"
        target_type = {
            "marine": -6,
            "marauder": -5,
            "stalker": -4,
            "zealot": -3,
            "zergling": -2,
            "hydralisk": -1,
            "any": None,
        }[unit]
        
        target_foe = target == "foe"
        move_toward = direction == "toward"

        def move_fn(state, obs, sight_range, attack_range, is_ally, env):
            fill = jnp.where(use_min, jnp.inf, -jnp.inf)
            n = jnp.where(is_ally, env.num_enemies, env.num_allies)  # number of foes
            self_obs, others_obs = process_obs(obs, env)
            alive = (
                others_obs.T[0] > 0
            )  # takes health and in_sight into consideration as health = 0 if not in sight
            target_team = jnp.where(
                target_foe,
                jnp.arange(alive.size) >= (alive.size - n),
                jnp.arange(alive.size) < (alive.size - n),
            )
            is_unit_types = jnp.where(use_unit_type, others_obs.T[target_type], 1)
            alive = jnp.logical_and(jnp.logical_and(alive, target_team), is_unit_types)
            dists = jnp.linalg.norm(others_obs.T[1:3], axis=0)
            health = others_obs.T[0]
            dists = jnp.where(alive, jnp.where(use_health, health, dists), fill)
            targ = jnp.where(use_min, jnp.argmin(dists), jnp.argmax(dists))
            x = others_obs[targ][1]
            y = others_obs[targ][2]
            SE = x > y
            NE = x > -y
            action = jnp.where(SE, jnp.where(NE, 1, 2), jnp.where(NE, 0, 3))
            action = jnp.where(move_toward, action, (action + 2) % 4)
            flag = jnp.where(alive.any(), RUNNING, FAILURE)
            action = jnp.where(alive.any(), action, STAND)
            return (flag, action)

        return move_fn
    else:  # target = direction or region
        if direction == "center":
            mat_to_dir = jnp.array([[1, 3], [0, 2]])

            def center_fn(state, obs, _, __, ___, env):
                self_pos = obs[-len(env.own_features) :][1:3] * 32 - 16
                dimension = jnp.argmax(jnp.abs(self_pos))
                direction = jnp.where(self_pos[dimension] > 0, 1, 0)
                action = mat_to_dir[dimension, direction]
                # move on dimension with higest absolute value
                #agent_id = env.agent_ids[agent]
                #is_alive = state.unit_health[agent_id] > 0
                #action = jnp.where(is_alive, action, STAND)
                return (RUNNING, action)

            return center_fn

        return lambda *_: (RUNNING, dir_to_idx[direction])


# ## Stand


def stand(*_):
    return (RUNNING, STAND)


# # conditions
# ## Regions location


def in_region(x, y=None):  # only applies to self
    y = x if y is None else y  # in_region center instead of in_region center center
    target_row, target_col = {
        ("north", "west"): (1, -1),
        ("north", "north"): (1, 0),
        ("north", "east"): (1, 1),
        ("west", "west"): (0, -1),
        ("center", "center"): (0, 0),
        ("east", "east"): (0, 1),
        ("south", "west"): (-1, -1),
        ("south", "south"): (-1, 0),
        ("south", "east"): (-1, 1),
    }[(x, y)]

    def in_region_fn(state, obs, _, __, ___, env):
        self_pos = obs[-len(env.own_features) :][1:3]
        col = jnp.where(self_pos[0] > 2 / 3, 1, jnp.where(self_pos[0] < 1 / 3, -1, 0))
        row = jnp.where(self_pos[1] > 2 / 3, 1, jnp.where(self_pos[1] < 1 / 3, -1, 0))
        flag = jnp.logical_and(row == target_row, col == target_col)
        return jnp.where(flag, SUCCESS, FAILURE)

    return in_region_fn


# ## In sight


def in_sight(target, unit="any"):  # is unit x in direction y?
    assert target in ["foe", "friend"]
    use_unit_type = unit != "any"
    target_type = {
            "marine": -6,
            "marauder": -5,
            "stalker": -4,
            "zealot": -3,
            "zergling": -2,
            "hydralisk": -1,
            "any": None,
        }[unit]

    target_foe = target == "foe"

    def in_sight_fn(state, obs, sight_range, attack_range, is_ally, env):
        n = jnp.where(is_ally, env.num_enemies, env.num_allies)  # number of foes
        self_obs, others_obs = process_obs(obs, env)
        alive = others_obs.T[0] > 0
        target_team = jnp.where(
            target_foe,
            jnp.arange(alive.size) >= (alive.size - n),
            jnp.arange(alive.size) < (alive.size - n),
        )
        is_unit_types = jnp.where(use_unit_type, others_obs.T[target_type], 1)
        alive = jnp.logical_and(jnp.logical_and(alive, target_team), is_unit_types)
        enemies_flag = alive.any()
        return jnp.where(enemies_flag, SUCCESS, FAILURE)

    return in_sight_fn


# ## In reach


def in_reach(other_agent, unit="any"):  # in shooting range
    assert other_agent in ["foe", "friend"]
    on_foe = other_agent == "foe"
    use_unit_type = unit != "any"
    target_type = {
            "marine": -6,
            "marauder": -5,
            "stalker": -4,
            "zealot": -3,
            "zergling": -2,
            "hydralisk": -1,
            "any": None,
        }[unit]

    def in_reach_fn(state, obs, sight_range, attack_range, is_ally, env):  # if any is in reach
        n = jnp.where(is_ally, env.num_enemies, env.num_allies)  # number of foes
        self_obs, others_obs = process_obs(obs, env)
        alive = others_obs.T[0] > 0
        target_team = jnp.where(
            on_foe,
            jnp.arange(alive.size) >= (alive.size - n),
            jnp.arange(alive.size) < (alive.size - n),
        )
        is_unit_types = jnp.where(use_unit_type, others_obs.T[target_type], 1)
        alive = jnp.logical_and(jnp.logical_and(alive, target_team), is_unit_types)
        dist = jnp.linalg.norm(others_obs.T[1:3], axis=0)
        in_range = attack_range / sight_range > dist
        flag = (jnp.logical_and(in_range, alive)).any()
        return jnp.where(flag, SUCCESS, FAILURE)

    return in_reach_fn


# ## is armed


def is_armed(agent):
    on_self = agent == "self"
    on_foe = agent == "foe"  # used only if not on_self

    def is_armed_fn(state, obs, sight_range, attack_range, is_ally, env):
        self_obs, others_obs = process_obs(obs, env)
        alive = others_obs.T[0] > 0
        n = jnp.where(is_ally, env.num_enemies, env.num_allies)  # number of foes
        target_team = jnp.where(
            on_foe,
            jnp.arange(alive.size) >= (alive.size - n),
            jnp.arange(alive.size) < (alive.size - n),
        )
        alive = jnp.logical_and(alive, target_team)
        other_cooldown = jnp.where(alive, others_obs.T[4], -jnp.inf)
        other_check = jnp.where(jnp.max(other_cooldown) > 0, SUCCESS, FAILURE)
        self_check = jnp.where(self_obs[3] <= 0, SUCCESS, FAILURE)
        return jnp.where(on_self, self_check, other_check)

    return is_armed_fn


# ## Is dying


def is_dying(agent, hp_level):
    assert hp_level in ["low", "middle", "high"]
    on_self = agent == "self"
    on_foe = agent == "foe"  # used only if not on_self
    threshold = {"low": 0.25, "middle": 0.5, "high": 0.75}[hp_level]

    def aux(state, obs, sight_range, attack_range, is_ally, env):
        self_obs, others_obs = process_obs(obs, env)
        alive = others_obs.T[0] > 0
        n = jnp.where(is_ally, env.num_enemies, env.num_allies)  # number of foes
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
    target_type = {
            "marine": -6,
            "marauder": -5,
            "stalker": -4,
            "zealot": -3,
            "zergling": -2,
            "hydralisk": -1,
    }[unit]
    true_condition = SUCCESS if negation == "a" else FAILURE
    false_condition = FAILURE if negation == "a" else SUCCESS

    def aux(_, obs, __, ___, ____, _____):
        return jnp.where(obs[target_type] == 1, true_condition, false_condition)

    return aux

# ## in flock


def is_flock(team, direction):
    on_foe = team == "foe"
    if direction == "center":

        def is_flock_fn(state, obs, sight_range, attack_range, is_ally, env):
            self_obs, others_obs = process_obs(obs, env)
            alive = others_obs.T[0] > 0
            n = jnp.where(is_ally, env.num_enemies, env.num_allies)  # number of enemies
            target_team = jnp.where(
                on_foe,
                jnp.arange(alive.size) >= (alive.size - n),
                jnp.arange(alive.size) < (alive.size - n),
            )
            alive = jnp.logical_and(alive, target_team)
            x = jnp.where(alive, others_obs.T[1], 0)
            y = jnp.where(alive, others_obs.T[2], 0)
            SE = x > y
            NE = x > -y
            SW = x < -y
            NW = x < y
            N = jnp.logical_and(NE, NW).any()
            E = jnp.logical_and(NE, SE).any()
            S = jnp.logical_and(SE, SW).any()
            W = jnp.logical_and(SW, NW).any()
            status = jnp.logical_and(jnp.logical_and(N, S), jnp.logical_and(E, W))
            return jnp.where(alive.any(), status, FAILURE)
    else:
        assert direction in ["north", "west", "east", "south"]
        is_north = SUCCESS if direction == "north" else FAILURE
        is_west = SUCCESS if direction == "west" else FAILURE
        is_east = SUCCESS if direction == "east" else FAILURE
        is_south = SUCCESS if direction == "south" else FAILURE

        def is_flock_fn(state, obs, sight_range, attack_range, is_ally, env):
            self_obs, others_obs = process_obs(obs, env)
            alive = others_obs.T[0] > 0
            n = jnp.where(is_ally, env.num_enemies, env.num_allies)  # number of enemies
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

    return is_flock_fn


# # Main


def main():
    rng = random.PRNGKey(0)
    env = make("SMAX")
    obs, state = env.reset(rng)
    args = (state, obs["ally_0"], "ally_0", env)
    # region test
    print(in_region("west", "center")(*args))
