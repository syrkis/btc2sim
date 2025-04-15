# act.py
#   returns an action function that applies a plan
# by: Noah Syrkis


# imports
import equinox as eqx
import jax.numpy as jnp
import parabellum as pb
from jax import lax, tree, debug
from jaxtyping import Array
from parabellum.env import Env
from parabellum.types import Obs, Scene, State
from typing import Tuple

from btc2sim.bts import Parent
from btc2sim.types import BehaviorArray


S, F, N = Parent.SEQUENCE, Parent.FALLBACK, Parent.NONE


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
    output, stats = lax.scan(eval_bt, (False, pb.types.Action(), 0), (status, actions, behavior))
    return output[1], stats


def eval_bt(carry, inputs: Tuple[Array, pb.types.Action, BehaviorArray]):  # depth first
    fn_status, fn_action, behavior = inputs  # load atomic in focus
    status, action, passing = carry  # passing on bt eval state

    search = status != 0 | (action.coord == 0).all()  # have we found yet?
    active = (status != (behavior.pred != S)) | (~behavior.pred == -1)  # is this fn in bt?

    status = lax.cond(search & active & (passing <= 0), lambda: fn_status, lambda: status)  #  update status
    action = lax.cond(search & active & (passing <= 0), lambda: fn_action, lambda: action)  # update action

    flag = (behavior.parent == S & status == 0) | (behavior.parent == F & status == 1)  # passing flag
    passing = (passing - 1 if flag else behavior.passing) if search & active & passing <= 0 else passing  # update
    return (status, action, passing), flag
