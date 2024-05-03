# utils.py
#    c2sim utility functions
# by: Noah Syrkis

# imports
import os
import jax
import jax.numpy as jnp
import chex

import numpy as np
from PIL import Image
import yaml
import argparse
from typing import Any, Callable, List, Tuple, Dict


# dataclasses
@chex.dataclass
class Status:  # for behavior tree
    SUCCESS: int = 1
    RUNNING: int = 0  # we might not need running, since we always have a return action
    FAILURE: int = -1


# default behavior tree
DEFAULT_BT = """
S (
    F (
        C ( in_region west center ) |>
        A ( move west )
    ) |>
    A ( attack foe_0 )
)
"""


# default action
STAND = 4  # do nothing


# types
NodeFunc = Callable[[Any], Status]

# dicts
dir_to_idx = {"north": 0, "east": 1, "south": 2, "west": 3}
idx_to_dir = {0: "north", 1: "east", 2: "south", 3: "west"}


# functions
def parse_args():
    parser = argparse.ArgumentParser(description="c2sim")
    # specify which script in src to run
    parser.add_argument("--script", type=str, default="main", help="script to run")
    return parser.parse_args()
