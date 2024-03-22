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
    SUCCESS: int = 0
    FAILURE: int = 1
    RUNNING: int = 2


# types
NodeFunc = Callable[[Any], Status]


# functions
def parse_args():
    parser = argparse.ArgumentParser(description="c2sim")
    # specify which script in src to run
    parser.add_argument("--script", type=str, default="main", help="script to run")
    return parser.parse_args()
