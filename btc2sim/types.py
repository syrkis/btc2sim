# imports
from chex import dataclass
from dataclasses import field
from jaxtyping import Array
import jax.numpy as jnp


# dataclasses
@dataclass
class Status:
    status: Array = field(default_factory=lambda: jnp.array(True))  # status
    active: Array = field(default_factory=lambda: jnp.array(False))  # active

    @property
    def success(self):
        return self.status

    @property
    def failure(self):
        return ~self.status


@dataclass
class Behavior:  # there will be one per unit (called wihth differnt obs)
    idx: Array
    parent: Array
    prev: Array
    skip: Array


@dataclass
class Compass:  # groups can have targets
    marks: Array
    df: Array
    dy: Array
    dx: Array


@dataclass
class Battalion:
    units: Array  # bool array in batalion else 0
    target: Array  # 0 to 6
    bt_idx: Array  # 0 to num bts
