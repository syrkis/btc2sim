# %%
from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
from jax import lax, tree
from jaxtyping import Array
from parabellum.types import Obs, Scene, State, Action
from parabellum.env import Env

from btc2sim.act import move_fns, stand_fns
from btc2sim.types import BehaviorArray, Parent


# %% Globals
S, F, N = Parent.SEQUENCE, Parent.FALLBACK, Parent.NONE


# %% functions
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


def action_fn(rng, env, scene, state, obs, behavior: BehaviorArray):  # for one agent
    status, action = leafs_fns(rng, env, scene, state, obs)
    init = (jnp.array((True,)), Action(), jnp.zeros(1))
    (status, action, passing), stats = lax.scan(bt_fn, init, (status, action, behavior))
    return action, stats


def bt_fn(carry: Tuple[Array, Action, Array], input: Tuple[Array, Action, BehaviorArray]):
    fn_status, fn_action, behavior = input  # load atomics and bt status
    status, action, passing = carry

    search = status != 1 | (action.coord == 0).all()  # boolean flags
    active = (status != (behavior.pred != S)) | (behavior.pred == -1)

    status = jnp.where(search & active & (passing <= 0), fn_status, status)  # (potentially) update action
    action = tree.map(lambda x, y: jnp.where(search & active & (passing <= 0), x, y), fn_action, action)  # don't skip

    flag = ((behavior.parent == S) & (status == 0)) | ((behavior.parent == F) & (status == 1))  # update passing
    passing = jnp.where(search & active & (passing <= 0), jnp.where(flag, passing - 1, behavior.passing), passing)
    return (status, action, passing), flag
