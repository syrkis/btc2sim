# %% main.py
#   btc2sim main file
# by: Noah Syrkis

# Imports
import sys

import jax.numpy as jnp
import numpy as np
import parabellum as pb
from einops import repeat
from jax import lax, random, tree, vmap, debug
from jax_tqdm import scan_tqdm
from omegaconf import OmegaConf
from PIL import Image

import btc2sim as b2s

# %% Constants
cfg = OmegaConf.load("conf.yaml")
rng, key = random.split(random.PRNGKey(0))
env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)
scene.terrain.building = b2s.utils.scene_fn(scene.terrain.building)
gps, targets = b2s.gps.gps_fn(scene, 6, key)
rngs = random.split(rng, cfg.steps)
obs, state = env.reset(key, scene)

bt = b2s.dsl.txt2bts(open("bts.txt", "r").readline())
bt = tree.map(lambda x: repeat(x, "h -> agents h", agents=env.num_units), bt)
action_fn = vmap(b2s.act.action_fn, in_axes=(0, 0, 0, None, None, None, 0))


# %% Functions
@scan_tqdm(n=cfg.steps)
def step_fn(carry, input):
    step, rng = input
    obs, state = carry
    rngs = random.split(rng, (2, env.num_units))
    action = action_fn(rngs[0], obs, bt, env, scene, gps, targets)
    # debug.breakpoint()
    obs, state = env.step(rngs[1], scene, state, action)
    return (obs, state), state


def anim(scene, seq, scale=4):  # animate positions TODO: remove dead units
    pos = seq.coords.astype(int)
    cord = jnp.concat((jnp.arange(pos.shape[0]).repeat(pos.shape[1])[..., None], pos.reshape(-1, 2)), axis=1).T
    idxs = cord[:, seq.health.flatten().astype(bool) > 0]
    mask = scene.terrain.building.at[*jnp.int32(gps.marks.T)].set(1)
    imgs = np.array(repeat(mask, "... -> a ...", a=len(pos)).at[*idxs].set(1))
    imgs = [Image.fromarray(img).resize(np.array(img.shape[:2]) * scale, Image.NEAREST) for img in imgs * 255]  # type: ignore
    imgs[0].save("output.gif", save_all=True, append_images=imgs[1:], duration=10, loop=0)


state, seq = lax.scan(step_fn, (obs, state), (jnp.arange(cfg.steps), rngs))
anim(scene, seq)
