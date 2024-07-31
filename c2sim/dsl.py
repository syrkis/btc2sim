# %% Imports
from lark import Lark
import os, sys
import c2sim
import jax
from functools import partial


# %% Constants
qualifiers = [ "String", "Direction", "Foe", "Friend", "Qualifier", "Sense", "Hp_Level", "Self", "Unit", "Negation", "Any"]
grammar = Lark(open("grammar.lark", "r"), start="start")


# %% Functions
def read(string):
    return grammar.parse(string)

def parse(lark_tree) -> dict:
    match lark_tree.data.title():
        case title if title in qualifiers:
            return lark_tree.children[0].lower()
        case "Node" | "Atomic":
            return parse(lark_tree.children[0])
        case "Nodes":
            return [parse(child) for child in lark_tree.children] # type: ignore
        case "Action" | "Condition":
            return (lark_tree.data.title().lower(), parse(lark_tree.children[0])) # type: ignore
        case _:  # Sequence or Fallback or Decorator
            value = [parse(child) for child in lark_tree.children]
            return lark_tree.data.title().lower(), value[0] if len(value) == 1 else value # type: ignore

def grow(seed: dict, env):
    if seed[0] in ["sequence", "fallback"]:
        children = knot([grow(child, env) for child in seed[1]])  # this should be a function that takes a number
        return c2sim.bt.tree_fn(children, env, seed[0])
    else:  #  seed[0] in ['condition', 'action']:
        _, func, args = seed[0], seed[1][0], seed[1][1]
        args = [args] if isinstance(args, str) else args
        return c2sim.bt.ATOMICS[func] if len(args) == 0 else c2sim.bt.ATOMICS[func](*args)

def knot(children):  # <- this could be a way to implement a switch
    def knot_fn(i, *args):
        fn = jax.lax.switch(i, *children)
        return fn(*args)
    return knot_fn
