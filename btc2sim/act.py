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
from jax import lax, tree, debug, random
from parabellum.types import Action
from btc2sim.types import Behavior


# %% Globals
S, F = 1, 0


# %% Behavior functions
@eqx.filter_jit
def fmap(fns, rng: Array, env: Env, scene: Scene, state: State, obs: Obs):
    *args, rngs = env, scene, state, obs, random.split(rng, len(fns))
    status, action = zip(*(f(rng, *args) for f, rng in zip(fns, rngs)))
    return jnp.concatenate(status), tree.map(lambda *xs: jnp.concatenate(xs), *action)


def leafs_fns(rng: Array, env: Env, scene: Scene, state: State, obs: Obs):
    fns = (is_alive, move_fns, stand_fns)  # it is important that this is run alphabetacally
    args = env, scene, state, obs
    status, action = fmap(fns, rng, *args)
    return status, action


def action_fn(rng, env, scene, state, obs, behavior: Behavior):  # for one agent
    fn_status, fn_action = leafs_fns(rng, env, scene, state, obs)
    init = (jnp.array((True,)), Action(), jnp.zeros(1))
    (status, action, passing), stats = lax.scan(bt_fn, init, (fn_status, fn_action, behavior))
    return action, stats


def bt_fn(carry: Tuple[Array, Action, Array], input: Tuple[Array, Action, Behavior]):  # this is wrong
    fn_status, fn_action, behavior = input  # load atomics and bt status
    # debug.breakpoint()

    status, action, passing = carry
    # debug.breakpoint()

    search = status != 1 | (action.coord == 0).all()  # boolean flags
    # debug.breakpoint()

    active = (status != (behavior.prev != S)) | (behavior.prev == -1)
    # debug.breakpoint()

    status = jnp.where(search & active & (passing <= 0), fn_status, status)  # (potentially) update action
    # debug.breakpoint()

    action = tree.map(lambda x, y: jnp.where(search & active & (passing <= 0), x, y), fn_action, action)  # don't skip
    # debug.breakpoint()

    flag = ((behavior.parent == S) & (status == 0)) | ((behavior.parent == F) & (status == 1))  # update passing
    # debug.breakpoint()

    passing = jnp.where(search & active & (passing <= 0), jnp.where(flag, passing - 1, behavior.skip), passing)
    debug.breakpoint()

    return (status, action, passing), flag


# %% Atomics
def move_fns(rng: Array, env: Env, scene: Scene, state: State, obs: Obs):
    direction = obs.unit_pos[0] - state.mark_position
    coords = direction / jnp.linalg.norm(direction, axis=-1)[..., None]  # used for moving from (-) and to (+)
    action = pb.types.Action(shoot=jnp.ones(coords.shape[0] * 2) == 0, coord=jnp.concat((-coords, coords)))
    return jnp.ones(action.shoot.size) == 1, action


def stand_fns(rng: Array, env: Env, scene: Scene, state: State, obs: Obs):
    action = pb.types.Action(shoot=jnp.ones((1,)) == 0, coord=jnp.zeros((1, 2)))
    return jnp.array((True,)), action


def is_alive(rng: Array, env: Env, scene: Scene, state: State, obs: Obs):
    action = pb.types.Action(shoot=jnp.ones((1,)) == 0, coord=jnp.zeros((1, 2)))
    return (obs.unit_health[0] > 0).reshape((1,)), action
