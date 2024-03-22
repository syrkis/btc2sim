# bt.py
#   behavior tree code
# by: Noah Syrkis

# imports
import jax
import jax.numpy as jnp
from jax import jit, vmap
import chex
from jaxmarl import make

from typing import Any, Callable, List, Tuple, Dict
import yaml
import os
from functools import partial

from .utils import Status, NodeFunc
import src.atomics as atomics


# functions
def tree(children: List[NodeFunc], kind: str) -> NodeFunc:
    def tick(rng, obs: jnp.array, agent, env) -> Status:
        for child in children:
            status, action = child(rng, obs, agent, env)
            if kind == "fallback" and status == Status.SUCCESS:
                return status, action
            if kind == "sequence" and status == Status.FAILURE:
                return status, action
        return (Status.FAILURE if kind == "fallback" else Status.SUCCESS), action

    return tick


def leaf(fn: Callable) -> NodeFunc:
    def tick(rng, obs: jnp.array, agent, env) -> Status:
        response = fn(rng, obs, agent, env)  # returns (status, and possibly action)
        return response if isinstance(response, tuple) else (response, None)

    return tick


def make_bt(env, fname) -> NodeFunc:
    with open(fname, "r") as f:
        bt_dict = yaml.safe_load(f)

    def make_node(node: dict) -> NodeFunc:
        if node["type"] in ["sequence", "fallback"]:
            children = [make_node(child) for child in node["children"]]
            return tree(children, node["type"])
        if node["type"] in ["condition", "action"]:
            fn = eval(f"globals()['atomics'].{node['fn']}")
            return leaf(fn)

    return partial(make_node(bt_dict), env=env)  # partial to pass env to all nodes


def main():
    fname = os.path.dirname(os.path.dirname(__file__)) + "/data/bt.yaml"
    rng = jax.random.PRNGKey(0)
    env = make("SMAX", num_allies=10, num_enemies=10)
    bt = make_bt(env, fname)
    obs, state = env.reset(rng)
    acts = {a: bt(rng, obs[a], a) for a in env.agents}
    print(acts)
