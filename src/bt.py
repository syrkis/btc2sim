# bt.py
#   behavior tree code
# by: Noah Syrkis

# imports
import jax
import jax.numpy as jnp
from jax import jit, vmap
import chex
from jaxmarl import make

import os
from functools import partial
from typing import Any, Callable, List, Tuple, Dict

from src.utils import Status, NodeFunc as NF, STAND, DEFAULT_BT
import src.atomics as atomics
from .bank import grammar_fn, parse_fn, dict_fn

# constants
ATOMIC_FNS = {fn: getattr(atomics, fn) for fn in dir(atomics) if not fn.startswith("_")}
S, F, R = Status.SUCCESS, Status.FAILURE, Status.RUNNING
PARENT_DIR = os.path.dirname(os.path.dirname(__file__))


# functions
def tree_fn(children: List[NF], node_kind: bool) -> NF:  # sequence / fallback
    def tick(state, obs: jnp.array, agent, env) -> Status:
        status, action, on = (S if node_kind else F, STAND, True)  # on=need act?
        for child in children:  # loop through all children
            ns, na = child(state, obs, agent, env)  # new state and action
            flag = jnp.logical_and(jnp.where(node_kind, ns != S, ns != F), on)
            # on, status, action = jnp.where(flag, (0, ns, na), (on, status, action))
            on = jnp.where(flag, 0, on)
            status = jnp.where(flag, ns, status)
            action = jnp.where(flag, na, action)
        return status, action

    return tick


def atomic_fn(fn: Callable, dec_fn: Callable = None) -> NF:
    def tick(state, obs: jnp.array, agent, env) -> Status:
        args = (state, obs, agent, env)
        fn_res = fn(*args)
        res = dec_fn(*fn_res) if dec_fn is not None else fn_res
        out = res if isinstance(res, tuple) else (res, STAND)
        return out

    return tick


def make_bt(env, tree) -> NF:
    def make_node(node: dict) -> NF:
        if node[0] in ["sequence", "fallback"]:
            children = [make_node(child) for child in node[1]]
            return tree_fn(children, node[0] == "sequence")
        if node[0] in ["condition", "action"]:
            _, func, args = node[0], node[1][0], node[1][1]
            args = [args] if isinstance(args, str) else args
            fn = ATOMIC_FNS[func] if len(args) == 0 else ATOMIC_FNS[func](*args)
            return atomic_fn(fn)
        if node[0] == "decorator":
            dec_fn = ATOMIC_FNS[node[1][0]]
            subtree = make_node(node[1][1])
            return atomic_fn(subtree, dec_fn)
        raise ValueError(f"Invalid node type: {node}")

    return partial(make_node(tree), env=env)  # partial to pass env to all nodes


def main():
    bt_str = DEFAULT_BT
    tree = dict_fn(grammar_fn().parse(bt_str))
    env = make("SMAX", num_allies=2, num_enemies=2)
    bt = make_bt(env, tree)
    rng = jax.random.PRNGKey(1)
    obs, state = env.reset(rng)
    acts = {a: bt(state, obs[a], a)[1] for idx, a in enumerate(env.agents)}
