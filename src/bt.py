# bt.py
#   behavior tree code
# by: Noah Syrkis

# imports
import jax
import jax.numpy as jnp
import chex
from jaxmarl import make

from typing import Any, Callable, List, Tuple, Dict
import yaml
import os

import atomics
from utils import Status


# types
NodeFunc = Callable[[Any], Status]


# functions
def tree(children: List[NodeFunc], kind: str) -> NodeFunc:
    def tick(rng, env, obs, agent) -> Status:
        for child in children:
            status, action = child(rng, env, obs, agent)

            if kind == "fallback" and status == Status.SUCCESS:
                return status, action
            if kind == "sequence" and status != Status.SUCCESS:
                return status, action

        if kind == "fallback":
            return Status.RUNNING, action
        if kind == "sequence":
            return Status.FAILURE, action

    return tick


def leaf(fn: Callable) -> NodeFunc:
    def tick(rng, env, obs, agent) -> Status:
        out = fn(rng, env, obs, agent)
        return out if isinstance(out, tuple) else (out, None)

    return tick


def make_bt(fname) -> NodeFunc:
    with open(fname, "r") as f:
        bt_dict = yaml.safe_load(f)

    def make_node(node: dict) -> NodeFunc:
        if node["type"] in ["sequence", "fallback"]:
            children = [make_node(child) for child in node["children"]]
            return tree(children, node["type"])
        if node["type"] in ["condition", "action"]:
            fn = eval(f"globals()['atomics'].{node['fn']}")
            return leaf(fn)

    return make_node(bt_dict)


def main():
    fname = os.path.dirname(os.path.dirname(__file__)) + "/data/bt.yaml"
    bt = make_bt(fname)
    rng = jax.random.PRNGKey(0)
    env = make("SMAX", num_allies=10, num_enemies=10)
    obs, state = env.reset(rng)
    for i in range(100):
        actions = {agent: bt(rng, env, obs[agent], agent) for agent in obs}
        obs, state = env.step(rng, state, actions)


if __name__ == "__main__":
    main()
