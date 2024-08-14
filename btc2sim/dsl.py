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
