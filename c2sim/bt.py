# bt.py
#   behavior tree code
# by: Noah Syrkis

# imports
import jax
import jax.numpy as jnp
from jax import jit, vmap, Array
from chex import dataclass
import chex
from jaxmarl import make

import os
from functools import partial
from typing import Any, Callable, List, Tuple, Dict, Optional

from c2sim.utils import Status, NodeFunc as NF, STAND, a2i, i2a
import c2sim.atomics as atomics

# constants
ATOMICS = {fn: getattr(atomics, fn) for fn in dir(atomics) if not fn.startswith("_")}
SUCCESS, FAILURE, RUNNING = Status.SUCCESS, Status.FAILURE, Status.RUNNING


# dataclasses
@dataclass
class Args:
    status: Array
    action: Array
    child: int
    state: Array
    obs: Array
    agent: int

# functions
def tree_fn(children, env, kind):

    def cond_fn(args):
        flag = jnp.where(kind == 'sequence', args.status == RUNNING, args.status != SUCCESS)
        return jnp.logical_and(flag, args.action == STAND)

    def body_fn(args):
        node_status, node_action = children(args.child)(args.state, args.obs, args.agent, env)
        child = args.child + 1
        status = jax.lax.select(node_status == RUNNING, RUNNING, args.status)
        action = jax.lax.select(node_status == RUNNING, node_action, args.action)
        args = Args(status=status, action=action, child=child, state=args.state, obs=args.obs, agent=args.agent)
        return args

    def tick(state, obs, agent, env):
        status = jnp.where(kind == 'sequence', SUCCESS, FAILURE)
        args = Args(status=status, action=STAND, child=0, state=state, obs=obs, agent=a2i(agent))
        args = jax.lax.while_loop(cond_fn, body_fn, args)
        return args.status, args.action

    return tick
