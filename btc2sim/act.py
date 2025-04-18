# act.py
#   returns an action function that applies a plan
# by: Noah Syrkis


# imports
import jax.numpy as jnp
import parabellum as pb
from jaxtyping import Array
from parabellum.env import Env
from parabellum.types import Obs, Scene
from typing import Tuple
import equinox as eqx
from jax import lax, tree, debug, random
from parabellum.types import Action
from btc2sim.types import Behavior, Status


# %% Globals
S, F = 1, 0


# %% Behavior functions
@eqx.filter_jit
def fmap(fns, rng: Array, obs: Obs, env: Env, scene: Scene):
    *args, rngs = obs, env, scene, random.split(rng, len(fns))
    status, action = zip(*(f(rng, *args) for f, rng in zip(fns, rngs)))
    status = tree.map(lambda *xs: jnp.stack(xs), *status)
    action = tree.map(lambda *xs: jnp.stack(xs), *action)
    return status, action


def leafs_fns(rng: Array, env: Env, scene: Scene, obs: Obs) -> Tuple[Status, Action]:
    fns = (alive_fn, move_fn, stand_fn)  # it is important that this is run alphabetacally
    args = obs, env, scene
    status, action = fmap(fns, rng, *args)
    return status, action


def action_fn(rng, obs: Obs, behavior: Behavior, env: Env, scene: Scene):  # for one agent
    fn_status, fn_action = leafs_fns(rng, env, scene, obs)
    init = (Status(), Action(), jnp.array(0))
    xs = fn_status, fn_action, behavior, jnp.arange(behavior.idx.shape[0])
    (status, action, passing), stats = lax.scan(bt_fn, init, xs)
    return action, stats


def bt_fn(carry: Tuple[Status, Action, Array], input: Tuple[Status, Action, Behavior, Array]):  # this is wrong
    fn_status, fn_action, behavior, step = input  # load atomics and bt status
    status, action, passing = carry
    # debug.breakpoint()

    searching = status.failure | (action.coord == 0).all()
    validated = (behavior.prev_sequence & status.success) | (behavior.prev_fallback & status.failure) | step == 0
    debug.breakpoint()

    status.status = jnp.where(searching & validated & (passing <= 0), fn_status.status, status.status)
    action = tree.map(lambda x, y: jnp.where(searching & validated & (passing <= 0), x, y), fn_action, action)
    # debug.breakpoint()

    flag = (behavior.parent_sequence & status.failure) | (behavior.parent_fallback & status.success)  # update passing
    passing = jnp.where(searching & validated & (passing <= 0), jnp.where(flag, passing - 1, behavior.skip), passing)
    # debug.breakpoint()

    return (status, action, passing), flag


# %% Atomics
def move_fn(rng: Array, obs: Obs, env: Env, scene: Scene):
    direction = obs.coords[0] - obs.target
    coords = direction / jnp.linalg.norm(direction)  # used for moving from (-) and to (+)
    action = pb.types.Action(shoot=jnp.array(False), coord=coords)
    status = Status(status=jnp.array(True))
    return status, action


def stand_fn(rng: Array, obs: Obs, env: Env, scene: Scene):
    action = pb.types.Action(shoot=jnp.array(False), coord=jnp.zeros(2))
    status = Status(status=jnp.array(True))
    return status, action


def alive_fn(rng: Array, obs: Obs, env: Env, scene: Scene):
    action = pb.types.Action(shoot=jnp.array(False), coord=jnp.zeros(2))
    status = Status(status=(obs.health[0] > 0))
    return status, action
