# bank.py
#   c2sim bt bank
# by: Noah Syrkis

# imports
from lark import Lark
import yaml
import json

from .atomics import ATOMICS


# functions
def grammar_fn():
    with open("grammar.lark", "r") as f:
        return Lark(f.read(), start="node")


def parse_fn(string):
    return grammar_fn().parse(string)


def dict_fn(tree):
    if tree.data.title() in ["String", "Direction", "Foe", "Friend", "Qualifier"]:
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


def main():
    bt_str = """
    S (
        F (
            C ( in_region east center ) |> 
            A ( move north ) ) |>
        A ( attack foe_0 ) |>
        A ( stand ) |>
        C ( in_sight foe_0 ) |>
        C ( in_reach foe_1 ) |>
        C ( in_region east center ) |>
        C ( is_dying self ) |>
        C ( is_armed friend_3 )
    )
    """
    tree = parse_fn(bt_str)
    dict_tree = dict_fn(tree)
    print(dict_tree)
    exit()
    json_tree = json.dumps(dict_tree, indent=2)
    print(dict_tree)


if __name__ == "__main__":
    main()
