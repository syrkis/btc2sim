# %%
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

import btc2sim
from btc2sim.classes import Status, NodeFunc as NF
from btc2sim.utils import STAND, NONE
import btc2sim.atomics as atomics

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
    rng: Array


# functions
def tree_fn(children, kind):
    start_status = jnp.where(kind == "sequence", SUCCESS, FAILURE)

    def cond_fn(args):  # conditions under which we continue
        cond = jnp.where(kind == "sequence", SUCCESS, FAILURE)
        flag = jnp.logical_and(args.status == cond, args.action == NONE)
        return jnp.logical_and(flag, args.child < len(children))

    def body_fn(args):
        child_status, child_action = jax.lax.switch(
            args.child, children, *(args.obs, args.info.env, args.info.agent, args.rng)
        )  # make info
        args = Args(
            status=child_status,
            action=child_action,
            obs=args.obs,
            child=args.child + 1,
            info=args.info,
            rng=args.rng,
        )
        return args

    def tick(obs, env_info, agent_info, rng):  # idx is to get info from batch dict
        info = btc2sim.classes.Info(env=env_info, agent=agent_info)
        args = Args(status=start_status, action=NONE, obs=obs, child=0, info=info, rng=rng)
        args = jax.lax.while_loop(
            cond_fn, body_fn, args
        )  # While we haven't found action action continue through children'
        return args.status, args.action
    return tick


def leaf_fn(func, kind):
    def tick(obs, env_info, agent_info, rng):
        info = btc2sim.classes.Info(env=env_info, agent=agent_info)
        return func(obs, info, rng)
    if kind == "action":
        return tick
    else:
        return lambda *args: (tick(*args), NONE)

def seed_fn(seed: dict):
    # grows a tree from a seed
    assert seed[0] in ["sequence", "fallback", "condition", "action"]
    if seed[0] in ["sequence", "fallback"]:
        children = [seed_fn(child) for child in seed[1]]
        tree = tree_fn(children, seed[0])
    else:  #  seed[0] in ['condition', 'action']:
        _, func, args = seed[0], seed[1][0], seed[1][1]
        args = [args] if isinstance(args, str) else args
        
        if len(args) == 0:
            tree = leaf_fn(ATOMICS[func], seed[0])
        else:   
            tree = leaf_fn(ATOMICS[func](*args), seed[0]) 
    def catch_none_action(*args):
        status, action = tree(children, seed[0])(*args)
        return  status, jnp.where(action==None, STAND, action)
    return catch_none_action


# %% [markdown]
# # Dynamic programming computation of the atomics 

# %%
def leaf_fn_dp(atomics_bank, func_name, args, kind):
    key = (func_name,) + tuple(args) 
    if key not in atomics_bank:
        func = ATOMICS[func_name] if len(args) == 0 else ATOMICS[func_name](*args)
        if kind == "action":
            def tick(obs, env_info, agent_info, rng):
                info = btc2sim.classes.Info(env=env_info, agent=agent_info)
                return func(obs, info, rng)
            atomics_bank[key] = tick 
        else:
            def tick(obs, env_info, agent_info, rng):
                info = btc2sim.classes.Info(env=env_info, agent=agent_info)
                return func(obs, info, rng)
            atomics_bank[key] = lambda *args: (tick(*args), NONE)
    return atomics_bank[key]


def seed_fn_dp(atomics_bank, seed: dict):
    # grows a tree from a seed
    assert seed[0] in ["sequence", "fallback", "condition", "action"]
    if seed[0] in ["sequence", "fallback"]:
        children = [seed_fn_dp(atomics_bank, child) for child in seed[1]]
        tree = tree_fn(children, seed[0])
    else:  #  seed[0] in ['condition', 'action']:
        _, func, args = seed[0], seed[1][0], seed[1][1]
        args = [args] if isinstance(args, str) else args
        tree = leaf_fn_dp(atomics_bank, func, args, seed[0])
    def catch_none_action(*args):
        status, action = tree(*args)
        return  status, jnp.where(action == NONE, STAND, action)
    return catch_none_action
