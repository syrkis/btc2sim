# agent.py
#  functions for agents
# by: Noah Syrkis

# imports
import jax
import jax.numpy as jnp
from jax import random, vmap, jit
import c2sim

# functions
def info_fn(agent, env):
    return c2sim.types.Info(
        agent_id=env.agent_ids[agent],
        num_agents=env.num_agents,
        num_allies=env.num_allies,
        num_enemies=env.num_enemies,
        num_own_features=len(env.own_features),
        num_types=len(env.unit_type_names),
        velocity=env.unit_type_velocities[env.agent_ids[agent]],
        sight_range=env.unit_type_sight_ranges[env.agent_ids[agent]],
        attack_range=env.unit_type_attack_ranges[env.agent_ids[agent]],
        is_ally=agent.startswith('ally'),
        map_width=env.map_width,
        map_height=env.map_height,
        world_steps_per_env_step=env.world_steps_per_env_step,
        time_per_step=env.time_per_step,
        terrain_raster = env.terrain_raster,
    )
