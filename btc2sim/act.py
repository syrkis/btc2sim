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
from parabellum.types import Action


S, F, N = Parent.SEQUENCE, Parent.FALLBACK, Parent.NONE


# returns dir for all 6 pieces
def move_fns(rng: Array, env: Env, scene: Scene, state: State, obs: Obs):
    direction = obs.unit_pos[0] - state.mark_position
    coords = direction / jnp.linalg.norm(direction)
    action = pb.types.Action(kinds=jnp.ones(coords.shape[0]) == 1, coord=coords)
    return jnp.ones(action.kinds.size) == 1, action


# stands
def stand_fns(rng: Array, env: Env, scene: Scene, state: State, obs: Obs):
    action = pb.types.Action(kinds=jnp.ones((1,)) == 1, coord=jnp.zeros((1, 2)))
    return jnp.array((True,)), action


@eqx.filter_jit
def fmap(fns, rng: Array, env: Env, scene: Scene, state: State, obs: Obs):
    args = env, scene, state, obs
    status, action = zip(*(f(rng, *args) for f, rng in zip(fns, jnp.split(rng, len(fns)))))
    return jnp.concatenate(status), tree.map(lambda *xs: jnp.concatenate(xs), *action)


def atomic_fns(rng: Array, env: Env, scene: Scene, state: State, obs: Obs):
    fns = (stand_fns, move_fns)
    args = env, scene, state, obs
    status, action = fmap(fns, rng, *args)
    return status, action


def action_fn(rng, env, scene, state, obs, behavior: BehaviorArray):  # for one agent
    status, action = atomic_fns(rng, env, scene, state, obs)
    init = (jnp.array((True,)), Action(), jnp.zeros(1))
    (status, action, passing), stats = lax.scan(eval_bt, init, (status, action, behavior))
    return action, stats


def eval_bt(carry: Tuple[Array, Action, Array], input: Tuple[Array, Action, BehaviorArray]):
    # load atomics and bt status
    fn_status, fn_action, behavior = input
    status, action, passing = carry

    # boolean flags
    search = status != 1 | (action.coord == 0).all()
    active = (status != (behavior.pred != S)) | (behavior.pred == -1)

    # (potentially) update action and status
    status = jnp.where(search & active & (passing <= 0), fn_status, status)
    action = tree.map(lambda x, y: jnp.where(search & active & (passing <= 0), x, y), fn_action, action)

    # (potentially) upate passing variable
    flag = ((behavior.parent == S) & (status == 0)) | ((behavior.parent == F) & (status == 1))
    passing = jnp.where(search & active & (passing <= 0), jnp.where(flag, passing - 1, behavior.passing), passing)
    return (status, action, passing), flag
