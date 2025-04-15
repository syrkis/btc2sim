# %% main.py
#   btc2sim main file
# by: Noah Syrkis

# Imports
import parabellum as pb
from jax import random, vmap, tree, lax
from einops import repeat
import jax.numpy as jnp
import numpy as np
from PIL import Image
import btc2sim as b2s
from omegaconf import OmegaConf


# %% Constants
bt = b2s.bts.txt2bts(open("bts.txt", "r").readline())
cfg = OmegaConf.load("conf.yaml")

rng, key = random.split(random.PRNGKey(0))
env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)

behavior = tree.map(lambda x: repeat(x, "h -> agents h", agents=env.num_units), bt)
action_fn = vmap(b2s.bts.action_fn, in_axes=(0, None, None, None, 0, 0))


# Functions
def step(carry, rng):
    obs, state = carry
    rngs = random.split(rng, (2, env.num_units))
    action, stats = tree.map(jnp.squeeze, action_fn(rngs[0], env, scene, state, obs, behavior))
    obs, state = env.step(rngs[1], scene, state, action)
    return (obs, state), (state, stats)


def anim(scene, seq, scale=2):  # animate positions TODO: remove dead units
    pos = seq.unit_position.astype(int)
    cord = jnp.concat((jnp.arange(pos.shape[0]).repeat(pos.shape[1])[..., None], pos.reshape(-1, 2)), axis=1).T
    idxs = cord[:, seq.unit_health.flatten().astype(bool) > 0]
    imgs = np.array(repeat(scene.terrain.building, "... -> a ...", a=len(pos)).at[*idxs].set(1))
    imgs = [Image.fromarray(img).resize(np.array(img.shape[:2]) * scale, Image.NEAREST) for img in imgs * 255]  # type: ignore
    imgs[0].save("output.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)


# Environment
rng, key = random.split(random.PRNGKey(0))
rngs = random.split(rng, 100)
obs, state = env.reset(key, scene)
state, (seq, stats) = lax.scan(step, (obs, state), rngs)
anim(scene, seq)
