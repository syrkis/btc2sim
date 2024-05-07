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

@partial(jax.jit, static_argnums=(1, 2))
def process_obs(obs, agent, env):
    n, k = env.num_agents, 10  # len(env.own_features)
    is_ally = agent.startswith("ally")
    order = jnp.where(is_ally, 1, -1)
    self_obs = obs[-k:]
    others_obs = obs[:-k].reshape(n - 1, -1)
    idx = env.num_allies - int(is_ally)
    return self_obs, others_obs, idx


def agent_info_fn(state, _, agent, env):
    agent_id = env.agent_ids[agent]
    agent_type = state.unit_types[agent_id]
    sight_range = env.unit_type_sight_ranges[agent_type]
    attack_range = env.unit_type_attack_ranges[agent_type]
    return sight_range, attack_range


# # actions

# ## Attacks 

def attack(target):  # TODO: attack closest if no target
    if target not in ["closest", "furthest", "weakest", "strongest"]:
        target_id = int(target.split("_")[-1])

        def attack_fn(state, obs, agent, env):
            is_ally = agent.startswith("ally")
            self_obs, others_obs, idx = process_obs(obs, agent, env)
            sight_range, attack_range = agent_info_fn(state, obs, agent, env)
            target_idx = jnp.where(is_ally, target_id + idx, target_id)
            target_obs = others_obs[target_idx]
            dist = jnp.linalg.norm(target_obs[1:3] - self_obs[1:3])
            status = jnp.where(dist < (attack_range / sight_range), RUNNING, FAILURE)
            action = jnp.where(status != FAILURE, STAND, target_id + 5)
            return (status, action)
            
    else: 
        qualifier = target
        assert (qualifier in ["closest", "furthest", "strongest", "weakest"])
        use_health = qualifier in ["strongest", "weakest"]
        use_min = qualifier in ["closest", "weakest"]
        
        def attack_fn(state, obs, agent, env):
            fill = jnp.where(use_min, jnp.inf, -jnp.inf)
            is_ally = agent.startswith("ally")
            self_obs, others_obs, idx = process_obs(obs, agent, env)
            n = jnp.where(is_ally, env.num_enemies, env.num_allies)  # number of foes 
            m = jnp.where(is_ally, env.num_allies, env.num_enemies) - 1  # number of allies
            alive = others_obs.T[0] > 0
            is_enemies = jnp.arange(alive.size) >= (alive.size - n)
            alive = jnp.logical_and(alive, is_enemies)
            dist = jnp.linalg.norm(others_obs.T[1:3], axis=0)
            sight_range, attack_range = agent_info_fn(state, obs, agent, env)
            in_reach = jnp.logical_and(attack_range / sight_range > dist, alive)
            
            health = others_obs.T[0]
            dist = jnp.where(in_reach, jnp.where(use_health, health, dist), fill)
            targ = jnp.where(use_min, jnp.argmin(dist), jnp.argmax(dist)) 
            flag = jnp.where( in_reach.any(), RUNNING, FAILURE)
            return (flag, targ + 5 - m)

    return attack_fn


# ## Move

def move(direction, qualifier=None, target=None):
    if direction in ["toward", "away_from"]:  # target = another agent
        assert (target in ["foe", "friend"])
        assert (qualifier in ["closest", "furthest", "strongest", "weakest"])
        use_health = qualifier in ["strongest", "weakest"]
        use_min = qualifier in ["closest", "weakest"]
        target_foe = target=="foe"
        move_toward = direction == "toward"
        def move_fn(state, obs, agent, env):
            is_ally = agent.startswith("ally")
            fill = jnp.where(use_min, jnp.inf, -jnp.inf)
            n = jnp.where(is_ally, env.num_enemies, env.num_allies)  # number of foes 
            self_obs, others_obs, _ = process_obs(obs, agent, env)
            sight_range, attack_range = agent_info_fn(state, obs, agent, env)
            alive = others_obs.T[0] > 0  # takes health and in_sight into consideration as health = 0 if not in sight 
            target_team = jnp.where(target_foe, jnp.arange(alive.size) >= (alive.size - n), jnp.arange(alive.size) < (alive.size - n))
            alive = jnp.logical_and(alive, target_team)
            dists = jnp.linalg.norm(others_obs.T[1:3], axis=0)
            health = others_obs.T[0]
            dists = jnp.where(alive, jnp.where(use_health, health, dists), fill)
            targ = jnp.where(use_min, jnp.argmin(dists), jnp.argmax(dists))
            x = others_obs[targ][1]
            y = others_obs[targ][2]
            SE = x > y
            NE = x > -y 
            action = jnp.where(SE, jnp.where(NE, 1, 2), jnp.where(NE, 0, 3))
            action = jnp.where(move_toward, action, (action+2)%4) 
            flag = jnp.where(alive.any(), RUNNING, FAILURE)
            return (flag, action)
        return move_fn
    else:  # target = direction or region
        if direction == "center":
            mat_to_dir = jnp.array([[1, 3], [0, 2]])
            
            def center_fn(state, obs, agent, env):
                agent_id = env.agent_ids[agent]
                self_pos = obs[-len(env.own_features) :][1:3] * 32 - 16
                dimension = jnp.argmax(jnp.abs(self_pos))
                direction = jnp.where(self_pos[dimension] > 0, 1, 0)
                action = mat_to_dir[dimension, direction]
                # move on dimension with higest absolute value
                is_alive = state.unit_health[agent_id] > 0
                action = jnp.where(is_alive, action, STAND)
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
    dir2int = {"north": 1, "south": -1, "west": -1, "east": 1, "center": 0}

    def in_region_fn(state, obs, agent, env):
        self_pos = obs[-len(env.own_features) :][1:3]
        # confirm pos ranges from -1 to 1 (might be from 0 to 1)
        row = jnp.where(self_pos[0] > 2 / 3, 1, jnp.where(self_pos[0] < 1 / 3, -1, 0))
        col = jnp.where(self_pos[1] > 2 / 3, 1, jnp.where(self_pos[1] < 1 / 3, -1, 0))
        flag = jnp.logical_and(row == dir2int[x], col == dir2int[y])
        return jnp.where(flag, SUCCESS, FAILURE)

    return in_region_fn


# ## In sight

def in_sight(target, d=None):  # is unit x in direction y?
    if d is None:  # any direction
        if "_" in  target:  # specified target
            # TODO
            in_sight_fn = stand
        else:  # any target  
            target_foe = target == "foe"
            def in_sight_fn(state, obs, agent, env):
                is_ally = agent.startswith("ally")
                n = jnp.where(is_ally, env.num_enemies, env.num_allies)  # number of foes 
                self_obs, others_obs, _ = process_obs(obs, agent, env)
                sight_range, attack_range = agent_info_fn(state, obs, agent, env)
                alive = others_obs.T[0] > 0
                target_team = jnp.where(target_foe, jnp.arange(alive.size) >= (alive.size - n), jnp.arange(alive.size) < (alive.size - n))
                alive = jnp.logical_and(alive, target_team)
                enemies_flag = alive.any()
                return jnp.where(enemies_flag, SUCCESS, FAILURE)               
    else:
        n = int(target.split("_")[-1]) if "_" in target else -1

        def in_sight_fn(state, obs, agent, env):
            team, offset_fn = FF_DICT[(agent.split("_")[0], target.split("_")[0])]
            offset = offset_fn(env)
            _, others_obs, _ = process_obs(obs, agent, env)
            target_pos = others_obs[n + offset][1:3]
            status = jnp.where(d in ["east", "west"], target_pos[1] > 0, target_pos[0] > 0)
            return jnp.where(status, SUCCESS, FAILURE)
    
    return in_sight_fn


# ## In reach

def in_reach(other_agent):  # in shooting range
    if "_" in other_agent:  # specific target 
        n = int(other_agent.split("_")[-1])
        def in_reach_fn(state, obs, self_agent, env):
            team, offset_fn = FF_DICT[
                (self_agent.split("_")[0], other_agent.split("_")[0])
            ]
            self_obs, others_obs, _ = process_obs(obs, self_agent, env)
            other_obs = others_obs[n + offset_fn(env)]
            alive = other_obs[0] > 0
            dist = jnp.linalg.norm(other_obs[1:3])
            sight_range, attack_range = agent_info_fn(state, obs, self_agent, env)
            flag = jnp.logical_and(attack_range / sight_range > dist, alive)
            return jnp.where(flag, SUCCESS, FAILURE)
    else:  # ["foe", "friend"]
        assert (other_agent in ["foe", "friend"])
        on_foe = other_agent == "foe"
        def in_reach_fn(state, obs, self_agent, env):  # if any is in reach
            is_ally = self_agent.startswith("ally")
            n = jnp.where(is_ally, env.num_enemies, env.num_allies)  # number of foes 
            self_obs, others_obs, _ = process_obs(obs, self_agent, env)
            alive = others_obs.T[0] > 0
            target_team = jnp.where( on_foe, jnp.arange(alive.size) >= (alive.size - n), jnp.arange(alive.size) < (alive.size - n))
            alive = jnp.logical_and(alive, target_team)
            dist = jnp.linalg.norm(others_obs.T[1:3], axis=0)
            sight_range, attack_range = agent_info_fn(state, obs, self_agent, env)
            in_range = attack_range / sight_range > dist
            flag = (jnp.logical_and(in_range, alive)).any()
            return jnp.where(flag, SUCCESS, FAILURE)

    return in_reach_fn


# ## is armed

def is_armed(agent):
    agent = -1 if agent == "self" else int(agent.split("_")[-1])
    
    @partial(jax.jit, static_argnums=(2, 3))
    def is_armed_fn(state, obs, self_agent, env):
        others_obs = obs[: -len(env.own_features)].reshape(env.num_agents - 1, -1)
        other_obs = others_obs[agent]
        return jnp.where(other_obs[-1] > 0, SUCCESS, FAILURE)

    return is_armed_fn


# ## Is dying

def is_dying(agent, hp_level):
    assert hp_level in ["low", "middle", "high"]
    on_self = agent == "self"
    on_foe = agent == "foe"  # used only if not on_self
    threshold = {"low": 0.25, "middle": 0.5, "high": 0.75}[hp_level]
       
    @partial(jax.jit, static_argnums=(2, 3))
    def aux(state, obs, self_agent, env):
        self_obs, others_obs, _ = process_obs(obs, self_agent, env)
        alive = others_obs.T[0] > 0
        is_ally = self_agent.startswith("ally")
        n = jnp.where(is_ally, env.num_enemies, env.num_allies)  # number of foes 
        target_team = jnp.where(on_foe, jnp.arange(alive.size) >= (alive.size - n), jnp.arange(alive.size) < (alive.size - n))
        alive = jnp.logical_and(alive, target_team)
        other_health = jnp.where(alive, others_obs.T[0], jnp.inf)
        other_check = jnp.where(jnp.min(other_health) < threshold, SUCCESS, FAILURE)
        self_check = jnp.where(self_obs[0] < threshold, SUCCESS, FAILURE)
        return jnp.where(on_self, self_check, other_check)

    return aux


# # Main

def main():
    rng = random.PRNGKey(0)
    env = make("SMAX")
    obs, state = env.reset(rng)
    args = (state, obs["ally_0"], "ally_0", env)
    # region test
    print(in_region("west", "center")(*args))
