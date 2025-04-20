# gps.py
#   calculates a navigation path from everywhere on terrain to target
# by: Noah Syrkis


# Imports
import jax.numpy as jnp
from parabellum.types import Scene
from jax import jit, debug, lax


# %% Globals
kernel = jnp.int16(jnp.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).reshape((1, 1, 3, 3)))


def gps_fn(scene, obs):
    df = df_fn(scene, jnp.int32(obs.target))
    coord = jnp.int32(obs.coords[0])
    # debug.breakpoint()
    return jnp.ones(2)


@jit
def df_fn(scene: Scene, target):
    def step_fn(carry, step):
        front, df = carry
        expanded = lax.conv(front[None, None, ...], kernel, (1, 1), "SAME").squeeze()
        front = (expanded > 0) & (df == -1) & (~scene.terrain.building)
        df = jnp.where(front, step, df)
        return (front, df), None

    front = jnp.zeros_like(scene.terrain.building).at[*target].set(1)
    front, df = lax.scan(step_fn, (front, jnp.where(front, 0, -1)), jnp.arange(front.shape[0]))[0]
    return df
