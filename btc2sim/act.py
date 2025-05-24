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
from btc2sim.types import Behavior, Status, Compass


# %% Behavior Treefunctions
@eqx.filter_jit
def fmap(fns, rng: Array, obs: Obs, env: Env, scene: Scene, gps: Compass, target: Array, bt: Behavior):
    *args, rngs = obs, env, scene, gps, target, random.split(rng, len(fns))
    status, action = zip(*(f(rng, *args) for f, rng in zip(fns, rngs)))
    select = lambda *xs: jnp.stack(xs).take(bt.idx, axis=0)  # noqa
    debug.breakpoint()
    return tree.map(select, *status), tree.map(select, *action)


def leafs_fns(rng: Array, obs: Obs, env: Env, scene: Scene, gps: Compass, target: Array, bt: Behavior):
    args = obs, env, scene, gps, target
    status, action = fmap(fns, rng, *args, bt)
    return status, action


def action_fn(rng, obs: Obs, bt: Behavior, env: Env, scene: Scene, gps: Compass, target: Array):
    atom_status, atom_action = leafs_fns(rng, obs, env, scene, gps, target, bt)
    init = (Status(), Action(), jnp.array(0))
    xs = atom_status, atom_action, bt, jnp.arange(atom_action.shoot.size)
    (_, action, _), flag = lax.scan(bt_fn, init, xs)
    return action


def bt_fn(carry: Tuple[Status, Action, Array], input: Tuple[Status, Action, Behavior, Array]):  # this is wrong
    atom_status, atom_action, bt, idx = input  # load atomics and bt status
    prev_status, prev_action, passing = carry

    search = prev_status.failure | ((prev_action.coord == 0).all() & prev_action.shoot)  # almost certainly right
    checks = (bt.prev & ~prev_status.failure) | (~bt.prev & ~prev_status.success) | (idx == 0)  # probably right

    status = Status(status=jnp.where(search & checks & (passing <= 0), atom_status.status, prev_status.status))
    action = tree.map(lambda x, y: jnp.where(search & checks & (passing <= 0), x, y), atom_action, prev_action)

    flag = (bt.parent & status.failure) | (~bt.parent & status.success)  # update passing
    passing = jnp.where(search & checks & (passing <= 0), jnp.where(flag, passing - 1, bt.skip), passing)

    return (status, action, passing), flag


###################################################################################
# %% Atomics ######################################################################
###################################################################################
def move_fn(rng: Array, obs: Obs, env: Env, scene: Scene, gps: Compass, target: Array):
    pos = jnp.int32(obs.coords[0])
    coord = -jnp.array((gps.dy[target][*pos], gps.dx[target][*pos]))
    action = Action(coord=coord, shoot=jnp.array(False))
    return Status(status=jnp.array(True)), action


def stand_fn(rng: Array, obs: Obs, env: Env, scene: Scene, gps: Compass, targets: Array):
    status = Status(status=jnp.array(True))
    return status, Action(coord=jnp.zeros(2), shoot=jnp.array(False))


def alive_fn(rng: Array, obs: Obs, env: Env, scene: Scene, gps: Compass, targets: Array):
    status = Status(status=(obs.health[0] > 0))
    return status, Action()


def shoot_random_fn(rng: Array, obs: Obs, env: Env, scene: Scene, gps: Compass, targets: Array):
    status = Status(status=jnp.array(True))
    action = Action(coord=random.normal(rng, (2,)))
    return status, action


def shoot_closest_fn(rng: Array, obs: Obs, env: Env, scene: Scene, gps: Compass, targets: Array):
    status = Status(status=jnp.array(True))
    action = Action(coord=random.normal(rng, (2,)))
    return status, action


def foe_in_sight_fn(rng: Array, obs: Obs, env: Env, scene: Scene, gps: Compass, targets: Array):
    status = Status(status=jnp.array(True))
    return status, Action()


def foe_in_range_fn(rng: Array, obs: Obs, env: Env, scene: Scene, gps: Compass, targets: Array):
    status = Status(status=jnp.array(True))
    return status, Action()


def friend_in_sight_fn(rng: Array, obs: Obs, env: Env, scene: Scene, gps: Compass, targets: Array):
    status = Status(status=jnp.array(True))
    return status, Action()


def friend_in_range_fn(rng: Array, obs: Obs, env: Env, scene: Scene, gps: Compass, targets: Array):
    status = Status(status=jnp.array(True))
    return status, Action()


# %% Grammar stuff
tuples = sorted(
    [
        (("stand",), stand_fn),
        (("is_alive",), alive_fn),
        (("move", "target"), move_fn),
        (("in_sight", "ally"), friend_in_sight_fn),
        (("in_range", "ally"), friend_in_range_fn),
        (("in_sight", "enemy"), foe_in_sight_fn),
        (("in_range", "enemy"), foe_in_range_fn),
        (("shoot", "closest"), shoot_closest_fn),
        (("shoot", "random"), shoot_random_fn),
    ],
    key=lambda x: x[0],
)
atomics = tuple((a[0] for a in tuples))
fns = tuple((a[1] for a in tuples))
