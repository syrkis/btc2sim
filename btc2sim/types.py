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
class Behavior:
    idx: Array
    parent: Array
    prev: Array
    skip: Array


@dataclass
class GPS:
    # marks: Array
    dy: Array
    dx: Array


@dataclass
class Info:
    targets: Array
