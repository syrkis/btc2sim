# %% Imports
from functools import partial
from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
import parabellum as pb
from jax import lax, random, vmap
from jaxtyping import Array
from parabellum.types import Scene

import btc2sim as b2s
from btc2sim.types import GPS
from omegaconf import OmegaConf
import seaborn as sns
import matplotlib.pyplot as plt


# %% Globals
cfg = OmegaConf.load("conf.yaml")
rng, key = random.split(random.PRNGKey(0))
env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)
scene.terrain.building = b2s.utils.scene_fn(scene.terrain.building)[4::10, 4::10]
kernel = jnp.float32(jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).reshape((1, 1, 3, 3)))


# %% Functions
def gps_fn(scene: Scene, n, rng) -> Tuple[GPS, Array]:
    marks = random.randint(rng, (n, 2), 0, scene.terrain.building.shape[0])
    df, dy, dx = vmap(partial(grad_fn, scene))(marks)
    targets = random.randint(rng, (scene.unit_types.size,), 0, n)
    return GPS(marks=marks, df=df, dy=dy, dx=dx), targets


@eqx.filter_jit
def grad_fn(scene: Scene, target):
    def step_fn(carry, step):
        front, df = carry
        front = (lax.conv(front, kernel, (1, 1), "SAME") > 0) * (df[None, None, ...] == front.size) * mask
        df = jnp.where(front, step, df).squeeze()
        return (front, df), None

    front = jnp.zeros(scene.terrain.building.shape).at[*target].set(1)[None, None, ...]
    df = jnp.where(front, 0, front.size).squeeze()
    steps = jnp.arange(scene.terrain.building.shape[0] * 2)
    front, df = lax.scan(step_fn, (front, df), steps)[0]
    return df, *jnp.gradient(df)


gps, target = gps_fn(scene, 10, rng)


def plot_df(df):
    sns.heatmap(df.squeeze().clip(0, df.shape[0] * 2), cmap="twilight", cbar=False)
    plt.axis("off")
    plt.gca().set_aspect("equal")


plot_df(gps.df[0])
# plot_df(gps.df[0])
