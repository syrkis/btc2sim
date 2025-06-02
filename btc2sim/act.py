# act.py
#   returns an action function that applies a plan
# by: Noah Syrkis


# imports
import jax.numpy as jnp
from jaxtyping import Array
from parabellum.env import Env
from parabellum.types import Obs, Scene
from typing import Tuple
import equinox as eqx
from jax import lax, tree, debug, random
from parabellum.types import Action
from btc2sim.types import Behavior, Status, Compass


# %% Globals
STAND = Action(types=jnp.array(1), coord=jnp.zeros(2))
NONE = Action(types=jnp.array(0), coord=jnp.zeros(2))


# %% Behavior Treefunctions
@eqx.filter_jit
def fmap(fns, rng: Array, obs: Obs, gps: Compass, target: Array, bt: Behavior):
    status, action = zip(*(f(rng, obs, gps, target) for f, rng in zip(fns, random.split(rng, len(fns)))))
    select = lambda *xs: jnp.stack(xs).take(bt.idx, axis=0)  # noqa # .take() is reodering the bt to leaf order
    return tree.map(select, *status), tree.map(select, *action)


def action_fn(rng, obs: Obs, bt: Behavior, env: Env, scene: Scene, gps: Compass, target: Array):
    atom_status, atom_action = fmap(fns, rng, obs, gps, target, bt)
    init = (Status(), NONE, jnp.array(0))
    xs = atom_status, atom_action, bt, jnp.arange(atom_action.shoot.size)
    (_, action, _), flag = lax.scan(bt_fn, init, xs)
    return action


def bt_fn(carry: Tuple[Status, Action, Array], input: Tuple[Status, Action, Behavior, Array]):  # this is wrong
    atom_status, atom_action, bt, idx = input  # load atomics and bt status
    prev_status, prev_action, passing = carry

    search = prev_status.failure | prev_action.invalid  # almost certainly right
    checks = (bt.prev & ~prev_status.failure) | (~bt.prev & ~prev_status.success) | (idx == 0)  # probably right

    status = Status(status=jnp.where(search & checks & (passing <= 0), atom_status.status, prev_status.status))
    action = tree.map(lambda x, y: jnp.where(search & checks & (passing <= 0), x, y), atom_action, prev_action)

    flag = (bt.parent & status.failure) | (~bt.parent & status.success)  # update passing
    passing = jnp.where(search & checks & (passing <= 0), jnp.where(flag, passing - 1, bt.skip), passing)
    # debug.breakpoint()
    return (status, action, passing), flag


###################################################################################
# %% Actions ######################################################################
###################################################################################
def stand_fn(rng: Array, obs: Obs, gps: Compass, targets: Array):
    status = Status(status=jnp.array(True))
    return status, STAND


def move_fn(rng: Array, obs: Obs, gps: Compass, target: Array):
    pos = jnp.int32(obs.coord[0])
    coord = -jnp.array((gps.dy[target][*pos], gps.dx[target][*pos])) * obs.speed[0]
    action = Action(coord=coord, types=jnp.array(1))
    return Status(status=jnp.array(True)), action


def attack_fn(rng: Array, obs: Obs, gps: Compass, targets: Array):
    p = ((obs.type - obs.type[0]) % 3 == 2) & (obs.team != obs.team[0]) & (obs.hp > 0)
    idx = random.choice(rng, a=jnp.arange(obs.type.size), p=p)
    # debug.breakpoint()
    status = Status(status=jnp.array(True))
    action = Action(coord=obs.coord[idx], types=jnp.array(2))
    return status, action


# def shoot_closest_fn(rng: Array, obs: Obs, gps: Compass, targets: Array):  # preferably target rock paper sissor dynamic
# status = Status(status=jnp.array(True))
# action = Action(coord=obs.coord[(obs.coord**2).sum(axis=1).argmin()], shoot=jnp.array(True))
# return status, action


###################################################################################
# %% Conditions ###################################################################
###################################################################################
def alive_fn(rng: Array, obs: Obs, gps: Compass, targets: Array):
    status = Status(status=(obs.hp[0] > 0))
    return status, NONE


def enemy_in_sight_fn(rng: Array, obs: Obs, gps: Compass, targets: Array):
    return Status(status=obs.enemy.any() > 0), NONE


def enemy_in_reach_fn(rng: Array, obs: Obs, gps: Compass, targets: Array):
    status = Status(status=(obs.enemy * (obs.dist < obs.reach[0])).sum() > 0)
    return status, NONE


def ally_in_sight_fn(rng: Array, obs: Obs, gps: Compass, targets: Array):
    return Status(status=obs.ally.any() > 0), NONE


def ally_in_reach_fn(rng: Array, obs: Obs, gps: Compass, targets: Array):
    status = Status(status=(obs.ally * obs.dist < obs.reach[0]).sum() > 0)
    return status, NONE


###################################################################################
# %% Grammar ######################################################################
###################################################################################
tuples = sorted(
    [
        (("stand",), stand_fn),
        (("is_alive",), alive_fn),
        (("move", "target"), move_fn),
        (("in_sight", "ally"), ally_in_sight_fn),
        (("in_range", "ally"), ally_in_reach_fn),
        (("in_sight", "enemy"), enemy_in_sight_fn),
        (("in_range", "enemy"), enemy_in_reach_fn),
        # (("shoot", "closest"), shoot_closest_fn),
        (("shoot", "random"), attack_fn),
    ],
    key=lambda x: x[0],
)
a2i = {a[0]: idx for idx, a in enumerate(tuples)}
fns = tuple((a[1] for a in tuples))
