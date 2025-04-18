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
from btc2sim.types import Behavior


# %% Globals
S, F = 1, 0


# %% Behavior functions
@eqx.filter_jit
def fmap(fns, rng: Array, obs: Obs, env: Env, scene: Scene):
    *args, rngs = obs, env, scene, random.split(rng, len(fns))
    status, action = zip(*(f(rng, *args) for f, rng in zip(fns, rngs)))
    return jnp.stack(status), tree.map(lambda *xs: jnp.stack(xs), *action)


def leafs_fns(rng: Array, env: Env, scene: Scene, obs: Obs):
    fns = (alive_fn, move_fn, stand_fn)  # it is important that this is run alphabetacally
    args = obs, env, scene
    status, action = fmap(fns, rng, *args)
    return status, action


def action_fn(rng, obs: Obs, behavior: Behavior, env: Env, scene: Scene):  # for one agent
    fn_status, fn_action = leafs_fns(rng, env, scene, obs)
    init = (jnp.array(False), Action(), jnp.array(0))
    (status, action, passing), stats = lax.scan(bt_fn, init, (fn_status, fn_action, behavior))
    return action, stats


def bt_fn(carry: Tuple[Array, Action, Array], input: Tuple[Array, Action, Behavior]):  # this is wrong
    fn_status, fn_action, behavior = input  # load atomics and bt status
    status, action, passing = carry
    debug.breakpoint()

    search = ~status | (action.coord == 0).all()  # status is false and still standing
    active = status != behavior.fallback | (behavior.prev == 0)  # maybe mistake here
    # debug.breakpoint()

    status = jnp.where(search & active & (passing <= 0), fn_status, status)  # (potentially) update action
    action = tree.map(lambda x, y: jnp.where(search & active & (passing <= 0), x, y), fn_action, action)  # don't skip
    # debug.breakpoint()

    flag = ((behavior.parent == S) & (status == 0)) | ((behavior.parent == F) & (status == 1))  # update passing
    passing = jnp.where(search & active & (passing <= 0), jnp.where(flag, passing - 1, behavior.skip), passing)
    # debug.breakpoint()

    return (status, action, passing), flag


# %% Atomics
def move_fn(rng: Array, obs: Obs, env: Env, scene: Scene):
    direction = obs.coords[0] - obs.target
    coords = direction / jnp.linalg.norm(direction)  # used for moving from (-) and to (+)
    action = pb.types.Action(shoot=jnp.array(False), coord=coords)
    return jnp.array(True), action


def stand_fn(rng: Array, obs: Obs, env: Env, scene: Scene):
    action = pb.types.Action(shoot=jnp.array(False), coord=jnp.zeros(2))
    return jnp.array(True), action


def alive_fn(rng: Array, obs: Obs, env: Env, scene: Scene):
    action = pb.types.Action(shoot=jnp.array(False), coord=jnp.zeros(2))
    return obs.health[0] > 0, action
