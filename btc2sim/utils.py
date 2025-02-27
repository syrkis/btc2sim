# utils.py
#    c2sim utility functions
# by: Noah Syrkis

# imports
import jax.numpy as jnp
from flax.struct import dataclass
import chex


@dataclass
class Action:
    kind: chex.Array
    value: chex.Array
    
    def __getitem__(self, index):  # to allow slicing operations
        return Action(
            kind=self.kind[index],
            value=self.value[index],
        )
        
    def set_item(self, index, new_value):
        # Perform an in-place update to kind and value at the specified index
        return Action( 
            kind = self.kind.at[index].set(new_value.kind),
            value = self.value.at[index].set(new_value.value)
        )

    @classmethod
    def from_shape(cls, shape, dtype=jnp.float32):
        # Create an instance with empty arrays of the specified shape
        return cls(
            kind=jnp.zeros(shape, dtype=dtype),
            value=jnp.zeros(shape+(2,), dtype=dtype)
        )

    def conditional_action(condition, action_if_true, action_if_false):
        return Action(
            kind=jnp.where(condition, action_if_true.kind, action_if_false.kind),
            value=jnp.where(condition, action_if_true.value, action_if_false.value)
        )


NONE, STAND, MOVE, ATTACK = jnp.array(-1), jnp.array(0), jnp.array(1), jnp.array(2)

None_action = Action(NONE, jnp.zeros((2,)))
Stand_action = Action(STAND, jnp.zeros((2,)))


