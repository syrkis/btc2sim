# %%
# agent.py
#  functions for agents
# by: Noah Syrkis

# %% imports
import jax.numpy as jnp
from jax import random, vmap, jit
import btc2sim


# %% functions
def info_fn(env, parallel_envs):
    env_info = env_info_fn(env)
    agent_info = agent_info_fn(env)


def env_info_fn(env):
    return btc2sim.classes.EnvInfo(
        num_agents=env.num_agents,
        num_allies=env.num_allies,
        num_enemies=env.num_enemies,
        num_types=jnp.array(len(env.unit_type_names)),
        num_own_features=jnp.array(len(env.own_features)),
        world_steps_per_env_step=env.world_steps_per_env_step,
        time_per_step=env.time_per_step,
        # map info
        map_width=env.map_width,
        map_height=env.map_height,
        terrain=env.terrain,  # 2D array of terrain types
    )


def agent_info_fn(env, direction_maps={}):
    agent_info = {
        a: btc2sim.classes.AgentInfo(
            agent_id=env.agent_ids[a],
            velocity=env.unit_type_velocities[env.scenario.unit_types[env.agent_ids[a]]],
            sight_range=env.unit_type_sight_ranges[env.scenario.unit_types[env.agent_ids[a]]],
            attack_range=env.unit_type_attack_ranges[env.scenario.unit_types[env.agent_ids[a]]],
            is_ally=a.startswith("ally"),
            direction_map=jnp.ones(env.terrain.building.shape, dtype=jnp.int32) if a not in direction_maps else direction_maps[a],
        )
        for a in env.agents
    }
    return agent_info

def agent_info_fn_to_vmap(env, direction_maps={}):
    agent_info = btc2sim.classes.AgentInfo(
            agent_id=jnp.array([env.agent_ids[a] for a in env.agents]),
            velocity=jnp.array([env.unit_type_velocities[env.scenario.unit_types[env.agent_ids[a]]] for a in env.agents]),
            sight_range=jnp.array([env.unit_type_sight_ranges[env.scenario.unit_types[env.agent_ids[a]]] for a in env.agents]),
            attack_range=jnp.array([env.unit_type_attack_ranges[env.scenario.unit_types[env.agent_ids[a]]] for a in env.agents]),
            is_ally=jnp.array([a.startswith("ally") for a in env.agents]),
            direction_map=jnp.array([jnp.ones(env.terrain.building.shape, dtype=jnp.int32) if a not in direction_maps else jnp.array(direction_maps[a]) for a in env.agents]),
        )
    return agent_info

def split_agent_info(agent_info, indices):
    new_agent_info = btc2sim.classes.AgentInfo(
        agent_id=agent_info.agent_id[indices],
        velocity=agent_info.velocity[indices],
        sight_range=agent_info.sight_range[indices],
        attack_range=agent_info.attack_range[indices],
        is_ally=agent_info.is_ally[indices],
        direction_map=agent_info.direction_map[indices],
    )
    return new_agent_info
