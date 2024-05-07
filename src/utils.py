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


# scenarios
scenarios = [
    "3m",
    "2s3z",
    "25m",
    "3s5z",
    "8m",
    "5m_vs_6m",
    "10m_vs_11m",
    "27m_vs_30m",
    "3s5z_vs_3s6z",
    "3s_vs_5z",
    "6h_vs_8z",
]


# dataclasses
@chex.dataclass
class Status:  # for behavior tree
    SUCCESS: int = 1
    RUNNING: int = 0  # we might not need running, since we always have a return action
    FAILURE: int = -1


# default behavior tree
DEFAULT_BT = """
F (
    A ( move toward closest foe) ::
    S (
        C ( in_region east) ::
        F (
            S (
                C ( is_flock friend south) ::
                A ( move north)
            ) ::
            S (
                C ( is_flock friend north) ::
                A ( move south)
            ) ::
            A ( move toward closest friend)
        )
    ) ::
    S (
        C ( in_region north east) ::
        A ( move west)
    ) ::
    S (
        C ( in_region north) ::
        A ( move west)
    ) ::
    S (
        C ( in_region south) ::
        A ( move west)
    ) ::
    S (
        C ( in_region south east) ::
        A ( move west)
    )
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
