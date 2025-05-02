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


kernel = jnp.int16(jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).reshape((1, 1, 3, 3)))


def gps_fn(scene: Scene, n, rng) -> Tuple[GPS, Array]:
    # marks = random.randint(rng, (n, 2), 0, scene.terrain.building.shape[0])
    marks = jnp.array([[27, i * 10] for i in range(n)])
    df, dy, dx = vmap(partial(grad_fn, scene))(marks)
    targets = random.randint(rng, (scene.unit_types.size,), 0, n)
    return GPS(marks=marks, df=df, dy=dy, dx=dx), targets


@eqx.filter_jit
def grad_fn(scene: Scene, target) -> Tuple[Array, Array, Array]:
    def step_fn(carry, step):
        front, df = carry
        expanded = lax.conv(front[None, None, ...], kernel, (1, 1), "SAME").squeeze()
        front = (expanded > 0) & (df == -1) & (~scene.terrain.building)
        df = jnp.where(front, step, df)
        return (front, df), None

    front = jnp.zeros_like(scene.terrain.building).at[*target].set(1)
    front, df = lax.scan(step_fn, (front, jnp.where(front, 0, -1)), jnp.arange(front.shape[0] * 2))[0]
    df, dy, dx = jnp.where(df == -1, df.max(), df), *jnp.gradient(df)

    return df, dy, dx


def plot_gps(gps):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, int(gps.df.shape[0]), figsize=(20, 20))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(gps.df[idx].clip(0, gps.df.shape[1] * 2), cmap="twilight_shifted")
        ax.scatter(*gps.marks[idx][::-1])  # very sus that i need to flip order
        ax.axis("off")
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()
