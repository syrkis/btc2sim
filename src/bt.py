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


# dataclasses
@chex.dataclass
class Status:
    SUCCESS: int = 0
    FAILURE: int = 1
    RUNNING: int = 2


# types
NodeFunc = Callable[[Any], Status]


# functions
def sequence(children: List[NodeFunc]) -> NodeFunc:
    def tick(rng, env, obs: Any) -> Status:
        for child in children:  # iterate over children (which are nodes themselves)
            status = child(rng, env, obs)
            if status != Status.SUCCESS:
                return status
        return Status.SUCCESS

    return tick


def fallback(children: List[NodeFunc]) -> NodeFunc:
    def tick(rng, env, obs: Any) -> Status:
        for child in children:
            status = child(rng, env, obs)
            if status == Status.SUCCESS:
                return status
        return Status.FAILURE

    return tick


def tree(children: List[NodeFunc], kind: str) -> NodeFunc:
    def tick(rng, env, obs) -> Status:
        for child in children:
            status = child(rng, env, obs)
            fallback = kind == "fallback" and status == Status.SUCCESS
            sequence = kind == "sequence" and status != Status.SUCCESS
            if fallback or sequence:
                return status
        return Status.FAILURE if kind == "fallback" else Status.sequence


def leaf(fn: Callable) -> NodeFunc:
    def tick(rng, env: Any, obs) -> Status:
        return fn(rng, env, obs)

    return tick


def make_bt(fname) -> NodeFunc:
    with open(fname, "r") as f:
        bt = yaml.safe_load(f)

    def make_node(node: dict) -> NodeFunc:
        if node["type"] in ["sequence", "fallback"]:
            children = [make_node(child) for child in node["children"]]
            return tree(children, node["type"])
        if node["type"] == "leaf":
            fn = eval(f"globals()['atomics'].{node['fn']}")
            return leaf(fn)

    return make_node(bt)


def main():
    fname = os.path.dirname(os.path.dirname(__file__)) + "/data/bt.yaml"
    bt = make_bt(fname)
    rng = jax.random.PRNGKey(0)
    env = make("SMAX", num_allies=2, num_enemies=10)
    obs, state = env.reset(rng)
    bt(rng, env, obs["ally_0"])
    # print(atomics.enemy_found(env, obs["ally_0"]))


if __name__ == "__main__":
    main()
