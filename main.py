# %% main.py
#   btc2sim main file
# by: Noah Syrkis

# Imports
import parabellum as pb
from jax import random, vmap, tree, lax, debug
from einops import repeat
import jax.numpy as jnp
import numpy as np
from PIL import Image
from lark import Lark
import btc2sim as b2s
from omegaconf import OmegaConf


# %% Constants
grammar = Lark(open("grammar.lark", "r").read())
bt = b2s.bts.txt2bts(open("bts.txt", "r").readline())

n_steps = 100
rng, key = random.split(random.PRNGKey(0))
cfg = OmegaConf.load("conf.yaml")
env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)


behavior = tree.map(lambda x: repeat(x, "h -> agents h", agents=env.num_units), bt)
action_fn = vmap(b2s.act.make_action_fn(b2s.dsl.all_vars, env.num_units), in_axes=(None, None, 0, None, None, 0, 0))


# Functions
def step(carry, rng):
    obs, state = carry
    action_key, step_key = random.split(rng, (2, env.num_units))
    action = action_fn(env, scene, action_key, state, obs, behavior, jnp.arange(env.num_units))
    obs, state = env.step(step_key, scene, state, action)
    return (obs, state), state


def anim(scene, seq, scale=2):  # animate positions TODO: remove dead units
    pos = seq.unit_position.astype(int)
    cord = jnp.concat((jnp.arange(pos.shape[0]).repeat(pos.shape[1])[..., None], pos.reshape(-1, 2)), axis=1).T
    idxs = cord[:, seq.unit_health.flatten().astype(bool) > 0]
    imgs = np.array(repeat(scene.terrain.building, "... -> a ...", a=n_steps).at[*idxs].set(1)).astype(np.uint8) * 255
    imgs = [Image.fromarray(img).resize(np.array(img.shape[:2]) * scale, Image.NEAREST) for img in imgs]  # type: ignore
    imgs[0].save("output.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)


# Environment
rng, key = random.split(random.PRNGKey(0))
rngs = random.split(rng, 100)
obs, state = env.reset(key, scene)
state, seq = lax.scan(step, (obs, state), rngs)
anim(scene, seq)  # np.array(seq.unit_position).astype(int), width=scene.terrain.building.shape[0])
