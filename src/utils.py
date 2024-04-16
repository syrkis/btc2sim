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
    FAILURE: int = 0
    RUNNING: int = -1  # we might not need running, since we always have a return action


# types
NodeFunc = Callable[[Any], Status]

# dicts
dir_to_idx = {"north": 1, "south": 2, "east": 3, "west": 4, "stop": 0}
idx_to_dir = {1: "north", 2: "south", 3: "east", 4: "west", 0: "stop"}


# functions
def parse_args():
    parser = argparse.ArgumentParser(description="c2sim")
    # specify which script in src to run
    parser.add_argument("--script", type=str, default="main", help="script to run")
    return parser.parse_args()
