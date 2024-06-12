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

from src.utils import Status, NodeFunc as NF, STAND
import src.atomics as atomics

# constants
ATOMIC_FNS = {fn: getattr(atomics, fn) for fn in dir(atomics) if not fn.startswith("_")}
S, F, R = Status.SUCCESS, Status.FAILURE, Status.RUNNING
PARENT_DIR = os.path.dirname(os.path.dirname(__file__))


# functions
def tree_fn(children: List[NF], node_kind: bool) -> NF:  # sequence / fallback
    # run parallel trees
    def tick(state, obs: jnp.array, sight_range, attack_range, is_ally, env) -> Status:
        status, action, on, selected_node_id = (S if node_kind else F, STAND, True, -1)  # on=need act?
        for child in children:  # loop through all children
            ns, na, node_id = child(state, obs, sight_range, attack_range, is_ally, env)  # new state and action
            flag = jnp.logical_and(jnp.where(node_kind, ns != S, ns != F), on)
            on = jnp.where(flag, 0, on)
            status = jnp.where(flag, ns, status)
            action = jnp.where(flag, na, action)
            selected_node_id = jnp.where(flag, node_id, selected_node_id)
        return status, action, selected_node_id

    return tick


def atomic_fn(fn: Callable, node_id, dec_fn: Callable = None) -> NF:
    def tick(state, obs: jnp.array, sight_range, attack_range, is_ally, env) -> Status:
        args = (state, obs, sight_range, attack_range, is_ally, env)
        fn_res = fn(*args)
        res = dec_fn(*fn_res) if dec_fn is not None else fn_res
        state, action = res if isinstance(res, tuple) else (res, STAND)
        return state, action, node_id

    return tick


def make_bt(tree) -> NF:
    node_id = 0
    
    def make_node(node: dict, node_id: int) -> NF:
        if node[0] in ["sequence", "fallback"]:
            node_id += 1
            children = []
            for child in node[1]:
                child_node, node_id = make_node(child, node_id)
                children.append(child_node)
            return tree_fn(children, node[0] == "sequence"), node_id
        if node[0] in ["condition", "action"]:
            _, func, args = node[0], node[1][0], node[1][1]
            args = [args] if isinstance(args, str) else args
            fn = ATOMIC_FNS[func] if len(args) == 0 else ATOMIC_FNS[func](*args)
            return atomic_fn(fn, node_id), node_id+1
        if node[0] == "decorator":
            dec_fn = ATOMIC_FNS[node[1][0]]
            subtree, node_id = make_node(node[1][1], node_id)
            return atomic_fn(subtree, node_id, dec_fn), node_id+1
        raise ValueError(f"Invalid node type: {node}")
    tree, _ = make_node(tree, node_id)
    return partial(tree)  # partial to pass env to all nodes


def main():
    pass
