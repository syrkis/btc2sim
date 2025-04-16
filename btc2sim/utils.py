# utils.py
#    c2sim utility functions
# by: Noah Syrkis

# imports
import jax.numpy as jnp
from jax import tree
import equinox as eqx
from typing import Tuple, Callable
from btc2sim.types import Behavior
from parabellum.types import Scene, State, Obs
from parabellum.env import Env
