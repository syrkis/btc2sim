# main.py
import parabellum as pb
from jax import random, lax
import jax.numpy as jnp
import numpy as np
from PIL import Image


# Functions
def step(state, rng):
    moving = random.normal(rng, (env.cfg.num_agents, 2))  # <- ADD BT BASED ACTION HERE
    action = pb.env.Action(health=None, moving=moving)
    obs, state = env.step(rng, state, action)
    return state, state


def anim(seq, scale=8, width=10):  # animate positions
    idxs = jnp.concat((jnp.arange(seq.shape[0]).repeat(seq.shape[1])[..., None], seq.reshape(-1, 2)), axis=1).T  #
    imgs = np.array(jnp.zeros((seq.shape[0], width, width)).at[*idxs].set(255)).astype(np.uint8)  # setting color
    imgs = [Image.fromarray(img).resize(np.array(img.shape[:2]) * scale, Image.NEAREST) for img in imgs]  # type: ignore
    imgs[0].save("output.gif", save_all=True, append_images=imgs[1:], duration=100, loop=0)


# Environment
env = pb.env.Env(cfg=(cfg := pb.env.Conf()))
rng, key = random.split(random.PRNGKey(0))
obs, state = env.reset(key)
rngs = random.split(rng, 100)
state, seq = lax.scan(step, state, rngs)
anim(seq.unit_position.astype(int), width=env.cfg.size, scale=8)
