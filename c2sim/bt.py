# bt.py
#   behavior tree code
# by: Noah Syrkis

# %% imports
import jax
import jax.numpy as jnp
from jax import jit, vmap, Array, tree_util
from chex import dataclass
import chex
from jaxmarl import make

import os
from functools import partial
from typing import Any, Callable, List, Tuple, Dict, Optional

import c2sim
from c2sim.types import Status, NodeFunc as NF
from c2sim.utils import STAND
import c2sim.atomics as atomics

# constants
ATOMICS = {fn: getattr(atomics, fn) for fn in dir(atomics) if not fn.startswith("_")}
SUCCESS, FAILURE = Status.SUCCESS, Status.FAILURE


# dataclasses
@dataclass
class Args:
    status: Array
    action: Array
    obs: Array
    child: int
    info: Any

# functions
def tree_fn(children, kind):
    start_status = jnp.where(kind == 'sequence', SUCCESS, FAILURE)

    def cond_fn(args):  # conditions under which we continue
        cond = jnp.where(kind == 'sequence', SUCCESS, FAILURE)
        flag = jnp.logical_and(args.status == cond, args.action == STAND)
        return jnp.logical_and(flag, args.child < len(children))

    def body_fn(args):
        child_status, child_action = jax.lax.switch(args.child, children, *(args.obs, args.info))  # make info
        args = Args(status=child_status, action=child_action, obs=args.obs, child=args.child + 1, info=args.info)
        return args

    def tick(obs, idx, env_info, agent_info):  # idx is to get info from batch dict
        info = c2sim.types.Info(env=env_info, agent=tree_util.tree_map(lambda x: x[idx], agent_info))
        args = Args(status=start_status, action=STAND, obs=obs, child=0, info=info)
        args = jax.lax.while_loop(cond_fn, body_fn, args)  # While we haven't found action action continue through children'
        return args.status, args.action

    return tick


def leaf_fn(func):

    def tick(obs, idx, env_info, agent_info):
        info = c2sim.types.Info(env=env_info, agent=tree_util.tree_map(lambda x: x[idx], agent_info))
        return func(obs, info)

    return tick


def seed_fn(seed: dict):
    # grows a tree from a seed
    assert seed[0] in ["sequence", "fallback", "condition", "action"]
    if seed[0] in ["sequence", "fallback"]:
        children = [seed_fn(child) for child in seed[1]]
        return tree_fn(children, seed[0])
    else:  #  seed[0] in ['condition', 'action']:
        _, func, args = seed[0], seed[1][0], seed[1][1]
        args = [args] if isinstance(args, str) else args
        return ATOMICS[func] if len(args) == 0 else leaf_fn(ATOMICS[func](*args))
