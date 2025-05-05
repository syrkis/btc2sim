# gps.py
#   calculates a navigation path from everywhere on terrain to target
# by: Noah Syrkis


# Imports
import jax.numpy as jnp
import jax.lax as lax
from jax import vmap, random
import equinox as eqx
from functools import partial
from parabellum.types import Scene
from btc2sim.types import GPS
from typing import Tuple
from jaxtyping import Array
import matplotlib.pyplot as plt


kernel = jnp.array([[jnp.sqrt(2), 1, jnp.sqrt(2)], [1, 0, 1], [jnp.sqrt(2), 1, jnp.sqrt(2)]]).reshape((1, 1, 3, 3))


def gps_fn(scene: Scene, marks) -> GPS:
    # marks = random.randint(rng, (n, 2), 0, scene.terrain.building.shape[0])
    df, dy, dx = vmap(partial(grad_fn, scene))(marks)
    # targets = random.randint(rng, (scene.unit_types.size,), 0, n)
    return GPS(marks=marks, df=df, dy=dy, dx=dx)


@eqx.filter_jit
def grad_fn(scene: Scene, target):
    def step_fn(carry, step):
        front, df = carry
        front = (lax.conv(front, kernel, (1, 1), "SAME") > 0) * (df[None, None, ...] == front.size) * mask
        df = jnp.where(front, step, df).squeeze()
        return (front, df), None

    mask = jnp.float32(jnp.abs(scene.terrain.building - 1))
    front = jnp.zeros(scene.terrain.building.shape).at[*target].set(1)[None, None, ...]
    df = jnp.where(front, 0, front.size).squeeze()
    steps = jnp.arange(scene.terrain.building.shape[0] * 2)
    front, df = lax.scan(step_fn, (front, df), steps)[0]
    dy, dx = jnp.gradient(df * ~(target == 0).all())  # 0,0 is an INVLAID target
    return df, dy, dx


def plot_gps(gps):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, int(gps.df.shape[0]), figsize=(20, 20))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(gps.df[idx].clip(0, gps.df.shape[1] * 2), cmap="twilight")
        ax.scatter(*gps.marks[idx][::-1])  # very sus that i need to flip order
        ax.axis("off")
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()
