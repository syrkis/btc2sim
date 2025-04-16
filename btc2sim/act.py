# act.py
#   returns an action function that applies a plan
# by: Noah Syrkis


# imports
import jax.numpy as jnp
import parabellum as pb
from jaxtyping import Array
from parabellum.env import Env
from parabellum.types import Obs, Scene, State


# %% Functions
def move_fns(rng: Array, env: Env, scene: Scene, state: State, obs: Obs):
    direction = obs.unit_pos[0] - state.mark_position
    coords = direction / jnp.linalg.norm(direction)
    action = pb.types.Action(kinds=jnp.ones(coords.shape[0]) == 1, coord=coords)
    return jnp.ones(action.kinds.size) == 1, action


def stand_fns(rng: Array, env: Env, scene: Scene, state: State, obs: Obs):
    action = pb.types.Action(kinds=jnp.ones((1,)) == 1, coord=jnp.zeros((1, 2)))
    return jnp.array((True,)), action
