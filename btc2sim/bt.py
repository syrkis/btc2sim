# %%
# bt.py
#   behavior tree code
# by: Noah Syrkis

# %%
import jax
import jax.numpy as jnp
from jax import jit, vmap, tree_util
from flax.struct import dataclass
import chex

import os
from functools import partial
from typing import Any, Callable, List, Tuple, Dict, Optional

import btc2sim
from btc2sim.classes import Status, NodeFunc as NF
from btc2sim.utils import STAND, NONE, Action, None_action, Stand_action
import btc2sim.atomics as atomics

# %% [markdown]
# # constants

# %%
ATOMICS = {fn: getattr(atomics, fn) for fn in dir(atomics) if not fn.startswith("_")}
SUCCESS, FAILURE = Status.SUCCESS, Status.FAILURE


# %% [markdown]
# # dataclasses

# %%
@dataclass
class Args:
    status: chex.Array
    action: Action
    child: int
    env: Any
    scenario: Any
    state: Any
    rng: chex.Array
    agent_id: int


# %% [markdown]
# # functions

# %% imports
def tree_fn(children, kind):
    start_status = jnp.where(kind == "sequence", SUCCESS, FAILURE)

    def cond_fn(args):  # conditions under which we continue
        cond = jnp.where(kind == "sequence", SUCCESS, FAILURE)
        flag = jnp.logical_and(args.status == cond, args.action.kind == NONE)  # atomics that returns FAILURE must also return NONE
        return jnp.logical_and(flag, args.child < len(children))

    def body_fn(args):
        child_status, child_action = jax.lax.switch(
            args.child, children, *(args.env, args.scenario, args.state, args.rng, args.agent_id)
        )  # make info
        args = Args(
            status=child_status,
            action=child_action,
            child=args.child + 1,
            env=args.env,
            scenario=args.scenario,
            state=args.state,
            rng=args.rng,
            agent_id=args.agent_id,
        )
        return args

    def tick(env, scenario, state, rng, agent_id):  # idx is to get info from batch dict
        args = Args(status=start_status, action=None_action, child=0, env=env, scenario=scenario, state=state, rng=rng, agent_id=agent_id)
        args = jax.lax.while_loop(
            cond_fn, body_fn, args
        )  # While we haven't found action action continue through children'
        return args.status, args.action
    return tick

def leaf_fn(func, kind):
    if kind == "action":
        return func
    else:
        return lambda *args: (func(*args), None_action)

def seed_fn(seed: dict, final=True):
    # grows a tree from a seed
    assert seed[0] in ["sequence", "fallback", "condition", "action"]
    if seed[0] in ["sequence", "fallback"]:
        children = [seed_fn(child, False) for child in seed[1]]
        tree = tree_fn(children, seed[0])
    else:  #  seed[0] in ['condition', 'action']:
        _, func, args = seed[0], seed[1][0], seed[1][1]
        args = [args] if isinstance(args, str) else args
        
        if len(args) == 0:
            tree = leaf_fn(ATOMICS[func], seed[0])
        else:   
            tree = leaf_fn(ATOMICS[func](*args), seed[0]) 
    if final:
        def catch_none_action(*args):
            status, action = tree(*args)
            return  status, Action.conditional_action(jnp.logical_or(action.kind == NONE, status == FAILURE), Stand_action, action)
        return catch_none_action
    else:
        return tree


# %% [markdown]
# # Dynamic programming computation of the atomics 

# %%
def leaf_fn_dp(atomics_bank, func_name, args, kind):
    key = (func_name,) + tuple(args) 
    if key not in atomics_bank:
        func = ATOMICS[func_name] if len(args) == 0 else ATOMICS[func_name](*args)
        if kind == "action":
            atomics_bank[key] = func
        else:
            atomics_bank[key] = lambda *args: (func(*args), None_action)
    return atomics_bank[key]


def seed_fn_dp(atomics_bank, seed: dict, final=False):
    # grows a tree from a seed
    assert seed[0] in ["sequence", "fallback", "condition", "action"]
    if seed[0] in ["sequence", "fallback"]:
        children = [seed_fn_dp(atomics_bank, child) for child in seed[1]]
        tree = tree_fn(children, seed[0])
    else:  #  seed[0] in ['condition', 'action']:
        _, func, args = seed[0], seed[1][0], seed[1][1]
        args = [args] if isinstance(args, str) else args
        tree = leaf_fn_dp(atomics_bank, func, args, seed[0])
    if final:
        def catch_none_action(*args):
            status, action = tree(*args)
            return  status, Action.conditional_action(jnp.logical_or(action.kind == NONE, status == FAILURE), Stand_action, action)
        return catch_none_action
    else:
        return tree

# %%
