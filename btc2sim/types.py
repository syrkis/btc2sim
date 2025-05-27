# imports
from chex import dataclass
from dataclasses import field
from jaxtyping import Array, Int32, Bool, Float32
import jax.numpy as jnp


# dataclasses
@dataclass
class Plan:
    units: Bool  # one hot of what units are in
    coord: Float32
    child: Int32
    btidx: Int32
    done: Bool
    move: Bool  # or kill

    @property
    def kill(self):
        return ~self.move

    @property
    def active(self):
        return ~self.done


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
