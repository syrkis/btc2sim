# utils.py
#    c2sim utility functions
# by: Noah Syrkis

# imports
import os
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image


# functions
def render(env, state_seq, scale=2):
    imgs = []
    for i, state in enumerate(state_seq):
        unit_pos = (state.state.unit_positions * scale).astype(int)
        img      = jnp.zeros((env.map_height * scale, env.map_width * scale, 3))
        img      = img.at[unit_pos[:, 0], unit_pos[:, 1], :].set(255).transpose((1, 0, 2))
        imgs    += [(img * 255).astype(np.uint8)]
    
    imgs = [Image.fromarray(np.array(img)) for img in imgs]
    imgs[0].save("docs/smax.gif", save_all=True, append_images=imgs[1:], loop=0, duration=len(imgs) // 24)