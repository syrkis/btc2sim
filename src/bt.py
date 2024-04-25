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
from .bank import grammar_fn, parse_fn, dict_fn

# constants
ATOMIC_FNS = {fn: getattr(atomics, fn) for fn in dir(atomics) if not fn.startswith("_")}
S, F, R = Status.SUCCESS, Status.FAILURE, Status.RUNNING
PARENT_DIR = os.path.dirname(os.path.dirname(__file__))


# functions
def tree_fn(children: List[NF], seq: bool) -> NF:  # sequence / fallback (selector)
    def tick(state, obs: jnp.array, agent, env) -> Status:
        stt, act, active = S if seq else F, STAND, True
        for child in children:  # loop through all children
            n_stt, n_act = child(state, obs, agent, env)
            node = jnp.logical_and(jnp.where(seq, n_stt != S, n_stt != F), act == STAND)
            cond = jnp.logical_and(node, active)
            active = jnp.where(cond, False, active)
            stt, act = jnp.where(cond, n_stt, stt), jnp.where(cond, n_act, act)
        return stt, act

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
            args = ()
            if isinstance(node, str):
                fn = ATOMIC_FNS[node]
            else:
                fn = ATOMIC_FNS[node[1][0]](*node[1][1:])
            return atomic_fn(fn)
        if node[0] == "decorator":
            dec_fn = ATOMIC_FNS[node[1][0]]
            subtree = make_node(node[1][1])
            return atomic_fn(subtree, dec_fn)
        raise ValueError(f"Invalid node type: {node[0]}")

    return partial(make_node(tree), env=env)  # partial to pass env to all nodes


def main():
    string = "S ( F ( C ( enemy_found ) :: A ( find_enemy )) :: A ( attack_enemy ))"
    tree = dict_fn(grammar_fn().parse(string))
    rng = jax.random.PRNGKey(1)
    env = make("SMAX", num_allies=10, num_enemies=10)
    bt = make_bt(env, tree)
    obs, state = env.reset(rng)
    acts = {a: bt(state, obs[a], a)[1] for idx, a in enumerate(env.agents)}
    for k, v in acts.items():
        print(f"{k}: {v}")
