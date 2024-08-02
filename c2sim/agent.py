# agent.py
#  functions for agents
# by: Noah Syrkis

# imports
import jax
import jax.numpy as jnp
from jax import random, vmap, jit
import c2sim


# functions
def info_fn(env):
    return c2sim.types.Info(
        env=env_info_fn(env),
        agent=agent_info_fn(env)
    )

def env_info_fn(env):
    return c2sim.types.EnvInfo(
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
        terrain_raster = env.terrain_raster, # 2D array of terrain types
    )

def agent_info_fn(env):
    return c2sim.types.AgentInfo(
        agent_id=jnp.array([env.agent_ids[agent] for agent in env.agent_ids]),
        velocity=jnp.array([env.unit_type_velocities[env.agent_ids[agent]] for agent in env.agent_ids]),
        sight_range=jnp.array([env.unit_type_sight_ranges[env.agent_ids[agent]] for agent in env.agent_ids]),
        attack_range=jnp.array([env.unit_type_attack_ranges[env.agent_ids[agent]] for agent in env.agent_ids]),
        is_ally=jnp.array([agent.startswith('ally') for agent in env.agent_ids])
    )
