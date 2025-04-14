# %% dsl.py
#   btc2sim dsl stuff
# by: Noah Syrkis

# Imports
from lark import Lark
from itertools import product

# Constants
grammar = Lark(open("grammar.lark", "r").read())
k = "QUALIFIER SENSE DIRECTION UNIT_TYPE SOURCE STEP THRESH MARGIN FRIEND FOE".lower().split()
v = map(lambda x: x.pattern.value.strip("()?:").split("|"), filter(lambda x: x.name.lower() in k, grammar.terminals))
t = {k: v for k, v in zip(k, v)}

# Helpers
find_bt_var = lambda x, subsets, variants: variants[subsets.index(x)]  # noqa
set_default = lambda bt_txt, default: f"F ( S( C (in_sight foe any) :: {bt_txt}) :: {default})"  # noqa
set_product = lambda sets: [" ".join(a) for a in product(*tuple(vals for vals in sets))]  # noqa


# %% Globals  # <- maybe put into grammar
actions = {
    "attack": set_product([["attack"], t["qualifier"], t["unit_type"] + ["any"]]),
    "move": set_product([["move"], t["sense"], t["qualifier"], t["friend"] + t["foe"], t["unit_type"] + ["any"]]),
    "stand": ["stand"],
    "follow_map": set_product([["follow_map"], t["sense"], t["margin"]]),
    "heal": set_product([["heal"], t["qualifier"], t["unit_type"] + ["any"]]),
}

conditions = {
    "in_sight": set_product([["in_sight"], t["friend"] + t["foe"], t["unit_type"] + ["any"]]),
    "in_reach": set_product([["in_reach"], t["friend"] + t["foe"], t["source"], t["step"], t["unit_type"] + ["any"]]),
    "is_type": set_product([["is_type"], t["unit_type"]]),
    "is_dying": set_product([["is_dying"], ["self", "foe", "friend"], t["thresh"]]),
    "is_in_forest": ["is_in_forest"],
}

atomics = {"A": actions, "C": conditions}
all_vars = [elm for row in atomics["A"].values() for elm in row] + [elm for row in atomics["C"].values() for elm in row]
