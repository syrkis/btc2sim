# utils.py
#    c2sim utility functions
# by: Noah Syrkis

# imports
import jax.numpy as jnp


# default action
NORTH, EAST, SOUTH, WEST, STAND, NONE = jnp.array(0), jnp.array(1), jnp.array(2), jnp.array(3), jnp.array(4), jnp.array(-1)


# dicts
dir_to_idx = {"north": 0, "east": 1, "south": 2, "west": 3}
idx_to_dir = {0: "north", 1: "east", 2: "south", 3: "west"}
