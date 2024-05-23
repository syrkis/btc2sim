# bank.py
#   c2sim bt bank
# by: Noah Syrkis

# imports
from lark import Lark
import yaml
import json
import os
from jax import vmap, jit

from .bt import make_bt


# functions
def grammar_fn():
    with open("grammar.lark", "r") as f:
        return Lark(f.read(), start="node")


def parse_fn(string):
    return grammar_fn().parse(string)


def dict_fn(tree):
    if tree.data.title() in [
        "String",
        "Direction",
        "Foe",
        "Friend",
        "Qualifier",
        "Sense",
        "Hp_Level",
        "Self",
        "Unit",
    ]:
        return tree.children[0].lower()
    elif tree.data.title() == "Node":
        return dict_fn(tree.children[0])
    elif tree.data.title() == "Nodes":
        return [dict_fn(child) for child in tree.children]
    elif tree.data.title() in ["Atomic"]:
        return dict_fn(tree.children[0])
    elif tree.data.title() in ["Action", "Condition"]:
        return (tree.data.title().lower(), dict_fn(tree.children[0]))
    else:  # Sequence or Fallback or Decorator
        key = tree.data.title().lower()
        value = [dict_fn(child) for child in tree.children]
        return key, value[0] if len(value) == 1 else value


def load_trees(env):
    # bank dir is in the data folder of the parant of this very file
    with open("data/bank.yaml", "r") as f:
        bank = yaml.safe_load(f)
    # replace tree with parsed tree
    for idx, tree in enumerate(bank):
        bt = make_bt(env, dict_fn(parse_fn(tree["tree"])))
        # bt = vmap(bt, in_axes=(0, 0, None))
        # bt = jit(bt, static_argnums=(2))
        bank[idx]["tree"] = bt
    return bank


def main():
    trees = load_trees()
    print(trees)


if __name__ == "__main__":
    main()
