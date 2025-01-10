# imports
from chex import dataclass
import chex
from jax import jit, vmap, Array
import jax.numpy as jnp
from typing import Any, Callable, List, Tuple, Dict, Optional


# dataclasses
@dataclass
class Status:  # for behavior tree
    SUCCESS: int = 1
    FAILURE: int = -1

# types
NodeFunc = Callable[[Any], Status]


