# main.py
import parabellum as pb
from jax import random, lax, vmap
import jax.numpy as jnp
import numpy as np
from PIL import Image
import btc2sim


# %% Constants
bt = """S(A (stand) :: A (stand))"""
max_leafs = 10
env = pb.env.Env(cfg=(cfg := pb.env.Conf()))
action_fn = btc2sim.atomics.get_action_factory(btc2sim.dsl.all_variants, cfg.num_agents, max_leafs)
vaction_fn = vmap(action_fn, in_axes=(None, None, None, 0, 0, 0))
behavior = btc2sim.bt.txt2array(bt, max_leafs)


# Functions
def step(rng, state):
    action_key, step_key = random.split(rng, (2, env.cfg.num_agents))
    action = vaction_fn(env, env.cfg, state, action_key, behavior, jnp.arange(env.cfg.num_agents))
    obs, state = env.step(step_key, state, action)
    return state, state


def anim(seq, scale=8, width=10):  # animate positions
    idxs = jnp.concat((jnp.arange(seq.shape[0]).repeat(seq.shape[1])[..., None], seq.reshape(-1, 2)), axis=1).T  #
    imgs = np.array(jnp.zeros((seq.shape[0], width, width)).at[*idxs].set(255)).astype(np.uint8)  # setting color
    imgs = [Image.fromarray(img).resize(np.array(img.shape[:2]) * scale, Image.NEAREST) for img in imgs]  # type: ignore
    imgs[0].save("output.gif", save_all=True, append_images=imgs[1:], duration=100, loop=0)


# Environment

rng, key = random.split(random.PRNGKey(0))
rngs = random.split(rng, 100)
obs, state = env.reset(key)
# state, seq = lax.scan(step, state, rngs)
# anim(seq.unit_position.astype(int), width=env.cfg.size, scale=8)
