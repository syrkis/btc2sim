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
import yaml
from functools import partial
from typing import Any, Callable, List, Tuple, Dict

from .utils import Status, NodeFunc
import src.atomics as atomics

# constants
ATOMIC_FNS = {fn: getattr(atomics, fn) for fn in dir(atomics) if not fn.startswith("_")}
SUCCESS, FAILURE, RUNNING = Status.SUCCESS, Status.FAILURE, Status.RUNNING


# functions (kind is static)
def tree(children: List[NodeFunc], kind: str) -> NodeFunc:
    def tick(rng, obs: jnp.array, agent, env) -> Status:
        ret_state, ret_action = RUNNING, -1
        for child in children:  # loop through all children
            state, action = child(rng, obs, agent, env)  # if action >= 0 keep
            sequence_failure = jnp.logical_and(kind == "sequence", state != SUCCESS)
            fallback_success = jnp.logical_and(kind == "fallback", state != FAILURE)
            node_cond = jnp.logical_or(sequence_failure, fallback_success)
            return_cond = jnp.logical_and(node_cond, ret_action == -1)
            # update action and rat state if return condition is met (no action yet)
            ret_state = jnp.where(return_cond, state, ret_state)
            action = jnp.where(return_cond, action, ret_action)
        return ret_state, action

    return tick


def leaf(fn: Callable) -> NodeFunc:
    def tick(rng, obs: jnp.array, agent, env) -> Status:
        response = fn(rng, obs, agent, env)  # returns (status, and possibly action)
        return response if isinstance(response, tuple) else (response, -1)

    return tick


def make_bt(env, fname: str) -> NodeFunc:
    with open(fname, "r") as f:
        bt_dict = yaml.safe_load(f)[0]

    def make_node(node: dict) -> NodeFunc:
        if node["type"] in ["sequence", "fallback"]:
            children = [make_node(child) for child in node["children"]]
            return tree(children, node["type"])
        if node["type"] in ["condition", "action"]:
            fn = ATOMIC_FNS.get(node["fn"], lambda _: (Status.FAILURE, None))
            return leaf(fn)
        raise ValueError(f"Invalid node type: {node['type']}")

    return partial(make_node(bt_dict), env=env)  # partial to pass env to all nodes


def main():
    fname = os.path.dirname(os.path.dirname(__file__)) + "/bt_bank.yaml"
    rng = jax.random.PRNGKey(0)
    env = make("SMAX", num_allies=10, num_enemies=10)
    bt = make_bt(env, fname)
    obs, state = env.reset(rng)
    acts = {a: bt(rng, obs[a], a) for a in env.agents}
    for k, v in acts.items():
        print(f"{k}: {v}")
