# main.py
import parabellum as pb
from jax import random, vmap, lax, tree
from einops import repeat
import jax.numpy as jnp
import numpy as np
from PIL import Image
import btc2sim as b2s


# %% Constants
bt, n = """S(A (stand) :: A (stand))""", 10
env = pb.env.Env(cfg=(cfg := pb.env.Conf()))
action_fn = vmap(b2s.make_action_fn(b2s.dsl.all_vars, cfg.num_agents, n), in_axes=(None, 0, None, None, 0, 0))
behavior = tree.map(lambda x: repeat(x, "h -> agents h", agents=env.cfg.num_agents), b2s.bt.txt2array(bt, n))


# Functions
def step(carry, rng):
    obs, state = carry
    action_key, step_key = random.split(rng, (2, env.cfg.num_agents))
    action = action_fn(env, action_key, state, obs, behavior, jnp.arange(env.cfg.num_agents))
    obs, state = env.step(step_key, state, action)
    return state, (obs, state)


def anim(seq, scale=8, width=10):  # animate positions
    idxs = jnp.concat((jnp.arange(seq.shape[0]).repeat(seq.shape[1])[..., None], seq.reshape(-1, 2)), axis=1).T
    imgs = np.array(jnp.zeros((seq.shape[0], width, width)).at[*idxs].set(255)).astype(np.uint8)  # setting color
    imgs = [Image.fromarray(img).resize(np.array(img.shape[:2]) * scale, Image.NEAREST) for img in imgs]  # type: ignore
    imgs[0].save("output.gif", save_all=True, append_images=imgs[1:], duration=100, loop=0)


# Environment
rng, key = random.split(random.PRNGKey(0))
rngs = random.split(rng, 100)
obs, state = env.reset(key)
state, seq = lax.scan(step, (obs, state), rngs)
# anim(seq.unit_position.astype(int), width=env.cfg.size, scale=8)
