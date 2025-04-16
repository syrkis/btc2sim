# act.py
#   returns an action function that applies a plan
# by: Noah Syrkis


# imports
import jax.numpy as jnp
import parabellum as pb
from jaxtyping import Array
from parabellum.env import Env
from parabellum.types import Obs, Scene, State
from typing import Tuple
import equinox as eqx
from jax import lax, tree
from parabellum.types import Action
from btc2sim.types import Behavior, Parent


# %% Globals
S, F, N = Parent.SEQUENCE, Parent.FALLBACK, Parent.NONE


# %% Behavior functions
@eqx.filter_jit
def fmap(fns, rng: Array, env: Env, scene: Scene, state: State, obs: Obs):
    args = env, scene, state, obs
    status, action = zip(*(f(rng, *args) for f, rng in zip(fns, jnp.split(rng, len(fns)))))
    return jnp.concatenate(status), tree.map(lambda *xs: jnp.concatenate(xs), *action)


def leafs_fns(rng: Array, env: Env, scene: Scene, state: State, obs: Obs):
    fns = (stand_fns, move_fns)
    args = env, scene, state, obs
    status, action = fmap(fns, rng, *args)
    return status, action


def action_fn(rng, env, scene, state, obs, behavior: Behavior):  # for one agent
    status, action = leafs_fns(rng, env, scene, state, obs)
    init = (jnp.array((True,)), Action(), jnp.zeros(1))
    (status, action, passing), stats = lax.scan(bt_fn, init, (status, action, behavior))
    return action, stats


def bt_fn(carry: Tuple[Array, Action, Array], input: Tuple[Array, Action, Behavior]):
    fn_status, fn_action, behavior = input  # load atomics and bt status
    status, action, passing = carry

    search = status != 1 | (action.coord == 0).all()  # boolean flags
    active = (status != (behavior.prevs != S)) | (behavior.prevs == -1)

    status = jnp.where(search & active & (passing <= 0), fn_status, status)  # (potentially) update action
    action = tree.map(lambda x, y: jnp.where(search & active & (passing <= 0), x, y), fn_action, action)  # don't skip

    flag = ((behavior.parent == S) & (status == 0)) | ((behavior.parent == F) & (status == 1))  # update passing
    passing = jnp.where(search & active & (passing <= 0), jnp.where(flag, passing - 1, behavior.skips), passing)
    return (status, action, passing), flag


# %% Atomics
def move_fns(rng: Array, env: Env, scene: Scene, state: State, obs: Obs):
    direction = obs.unit_pos[0] - state.mark_position
    coords = direction / jnp.linalg.norm(direction)
    action = pb.types.Action(kinds=jnp.ones(coords.shape[0]) == 1, coord=coords)
    return jnp.ones(action.kinds.size) == 1, action


def stand_fns(rng: Array, env: Env, scene: Scene, state: State, obs: Obs):
    action = pb.types.Action(kinds=jnp.ones((1,)) == 1, coord=jnp.zeros((1, 2)))
    return jnp.array((True,)), action
