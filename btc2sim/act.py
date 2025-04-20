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
from btc2sim.gps import gps_fn


# %% Behavior functions
@eqx.filter_jit
def fmap(fns, rng: Array, obs: Obs, env: Env, scene: Scene, bt: Behavior):
    *args, rngs = obs, env, scene, random.split(rng, len(fns))
    status, action = zip(*(f(rng, *args) for f, rng in zip(fns, rngs)))
    status = tree.map(lambda *xs: jnp.stack(xs).take(bt.idx), *status)
    action = tree.map(lambda *xs: jnp.stack(xs).take(bt.idx), *action)
    return status, action


def leafs_fns(rng: Array, env: Env, scene: Scene, obs: Obs, bt: Behavior) -> Tuple[Status, Action]:
    fns = (alive_fn, move_fn, stand_fn)  # it is important that this is run alphabetacally
    args = obs, env, scene
    status, action = fmap(fns, rng, *args, bt)
    return status, action


def action_fn(rng, obs: Obs, bt: Behavior, env: Env, scene: Scene):  # for one agent
    atom_status, atom_action = leafs_fns(rng, env, scene, obs, bt)
    init = (Status(), Action(), jnp.array(0))
    xs = atom_status, atom_action, bt, jnp.arange(atom_action.shoot.size)
    (_, action, _), _ = lax.scan(bt_fn, init, xs)
    return action


def bt_fn(carry: Tuple[Status, Action, Array], input: Tuple[Status, Action, Behavior, Array]):  # this is wrong
    atom_status, atom_action, bt, idx = input  # load atomics and bt status
    prev_status, prev_action, passing = carry

    search = prev_status.failure | ((prev_action.coord == 0).all() & prev_action.shoot)  # almost certainly right
    checks = (bt.prev & ~prev_status.failure) | (~bt.prev & ~prev_status.success) | (idx == 0)  # probably right

    status = Status(status=jnp.where(search & checks & (passing <= 0), atom_status.status, prev_status.status))
    action = tree.map(lambda x, y: jnp.where(search & checks & (passing <= 0), x, y), atom_action, prev_action)
    # debug.breakpoint()

    flag = (bt.parent & status.failure) | (~bt.parent & status.success)  # update passing
    passing = jnp.where(search & checks & (passing <= 0), jnp.where(flag, passing - 1, bt.skip), passing)

    return (status, action, passing), flag


# %% Atomics
def move_fn(rng: Array, obs: Obs, env: Env, scene: Scene):
    coord = gps_fn(scene, obs)
    action = Action(coord=coord, shoot=jnp.array(False))
    return Status(status=jnp.array(True)), action


def stand_fn(rng: Array, obs: Obs, env: Env, scene: Scene):
    status = Status(status=jnp.array(True))
    return status, Action(coord=jnp.array([0, 0]), shoot=jnp.array(False))


def alive_fn(rng: Array, obs: Obs, env: Env, scene: Scene):
    status = Status(status=(obs.health[0] > 0))
    return status, Action()


def foe_in_sight(rng: Array, obs: Obs, env: Env, scene: Scene):
    status = Status(status=(obs.health[1] > 0))
    return status, Action()


def foe_in_range(rng: Array, obs: Obs, env: Env, scene: Scene):
    status = Status(status=(obs.health[1] > 0) & (jnp.linalg.norm(obs.target - obs.coords[0]) < 1))
    return status, Action()


def friend_in_range(rng: Array, obs: Obs, env: Env, scene: Scene):
    status = Status(status=(obs.health[2] > 0) & (jnp.linalg.norm(obs.target - obs.coords[0]) < 1))
    return status, Action()


def friend_in_sight(rng: Array, obs: Obs, env: Env, scene: Scene):
    status = Status(status=(obs.health[2] > 0))
    return status, Action()


def shoot_nearest(rng: Array, obs: Obs, env: Env, scene: Scene):
    status = Status(status=(obs.health[0] > 0))
    return status, Action()
