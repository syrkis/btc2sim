# gps.py
#   calculates a navigation path from everywhere on terrain to target
# by: Noah Syrkis


# Imports
import jax.numpy as jnp
from parabellum.types import Obs, Scene
from jax import jit, debug, lax


# %% Globals
kernel = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).reshape((1, 1, 3, 3))


# @cache
def step_fn(carry, step):
    front, distance_map, obstacles, step_num = carry
    expanded = lax.conv(front[None, None], kernel, (1, 1), "SAME")[0, 0]
    new_front = (expanded > 0) & (distance_map == -1) & (~obstacles)
    distance_map = jnp.where(new_front, step_num, distance_map)
    return (new_front, distance_map), None


@jit
def gps_fn(scene: Scene, target):
    front = jnp.zeros_like(scene.terrain.building).at[*jnp.int32(target)].set(1)
    df = jnp.where(target, 0, -1)
    init = (front, df, scene.terrain.building)
    *(front, df), _ = lax.scan(step_fn, init, jnp.arange(100))
