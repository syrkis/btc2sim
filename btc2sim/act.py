# act.py
#   returns an action function that applies a plan
# by: Noah Syrkis


# imports
import equinox as eqx
import jax.numpy as jnp
import parabellum as pb
from flax.struct import dataclass
from jax import lax, tree, debug
from jaxtyping import Array
from parabellum.env import Env
from parabellum.types import Obs, Scene, State
from typing import Tuple

from btc2sim.bts import Parent
from btc2sim.types import BehaviorArray


# %%
@dataclass
class Status:  # for behavior tree
    SUCCESS: int = 1
    FAILURE: int = -1
    NONE: int = 0


# returns dir for all 6 pieces
def move_fns(rng: Array, env: Env, scene: Scene, state: State, obs: Obs):
    direction = obs.unit_pos[0] - state.mark_position
    coords = direction / jnp.linalg.norm(direction)
    action = pb.types.Action(kinds=jnp.ones(coords.shape[0]), coord=coords)
    return jnp.ones(action.kinds.size), action


# stands
def stand_fns(rng: Array, env: Env, scene: Scene, state: State, obs: Obs):
    action = pb.types.Action(kinds=jnp.ones((1,)), coord=jnp.zeros((1, 2)))
    return jnp.ones(1), action


@eqx.filter_jit
def fmap(fns, rng: Array, env: Env, scene: Scene, state: State, obs: Obs):
    args = env, scene, state, obs
    status, action = zip(*(f(rng, *args) for f, rng in zip(fns, jnp.split(rng, len(fns)))))
    return jnp.concatenate(status), tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *action)


def atomic_fns(rng: Array, env: Env, scene: Scene, state: State, obs: Obs):
    fns = (stand_fns, move_fns)
    args = env, scene, state, obs
    status, action = fmap(fns, rng, *args)
    return status, action


def action_fn(rng, env, scene, state, obs, behavior: BehaviorArray):  # for one agent
    status, actions = atomic_fns(rng, env, scene, state, obs)
    output, stats = lax.scan(eval_bt, (False, pb.types.Action()), (status, actions, behavior))
    return output[1], stats


def eval_bt(carry, inputs: Tuple[Array, pb.types.Action, BehaviorArray]):  # depth first search leafs
    (status, actions, behavior), (status, action) = inputs, carry
    debug.breakpoint()
    action = tree.map(jnp.int32, pb.types.Action(kinds=jnp.zeros(1), coord=jnp.array([0, 1])))
    return (True, action), jnp.ones(10)

    need_action = jnp.logical_or(status != Status.SUCCESS, jnp.all(action.coord == 0))
    s_condition = jnp.logical_and(behavior.predecessors == Parent.SEQUENCE, status != Status.FAILURE)
    f_condition = jnp.logical_and(behavior.predecessors == Parent.FALLBACK, status != Status.SUCCESS)

    is_valid = jnp.logical_or(jnp.logical_or(s_condition, f_condition), behavior.predecessors == Parent.NONE)
    is_valid = jnp.logical_and(is_valid, behavior.atomics_id != -1)  # not an empty leaf (from fixed size)
    condition = jnp.logical_and(jnp.logical_and(need_action, is_valid), passing <= 0)
    passing_if_FAILURE_in_sequence = jnp.logical_and(
        behavior.parents == Parent.SEQUENCE, behavior.status[behavior.atomics_id] == Status.FAILURE
    )
    passing_if_SUCCESS_in_failure = jnp.logical_and(
        behavior.parents == Parent.FALLBACK, behavior.status[behavior.atomics_id] == Status.SUCCESS
    )
    if_passing = jnp.logical_and(
        condition, jnp.logical_or(passing_if_FAILURE_in_sequence, passing_if_SUCCESS_in_failure)
    )
    passing = jnp.where(if_passing, behavior.passings, passing - 1)
    status = jnp.where(condition, behavior.status[behavior.atomics_id], status)  # status
    action = behavior.action[behavior.atomics_id].where(condition, action)  # action
    return (True, action), jnp.ones(10)
