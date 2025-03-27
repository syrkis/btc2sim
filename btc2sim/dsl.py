# %%
from lark import Lark
from itertools import product


# %%
def set_product(sets):
    return [" ".join(a) for a in product(*tuple(vals for vals in sets))]


# %%
qualifiers = ["closest", "farthest", "weakest", "strongest", "random"]
senses = ["toward", "away_from"]
directions = ["north", "south", "east", "west"]
unit_types = ["spearmen", "archer", "cavalry", "healer", "grenadier"]
source = ["me_from_them", "them_from_me"]
steps = ["0", "1", "2", "3"]
any_ = ["any"]
thresholds = ["25%", "50%", "75%"]
margins = ["0%", "25%", "50%", "100%"]
unit_any = unit_types + any_

actions = {
    "attack": set_product([["attack"], qualifiers, unit_any]),
    "move": set_product([["move"], senses, qualifiers, ["foe", "friend"], unit_any]),
    "stand": ["stand"],
    "follow_map": set_product([["follow_map"], senses, margins]),
    "heal": set_product([["heal"], qualifiers, unit_any]),
    # "debug": set_product([["debug"], directions]),
}

conditions = {
    "in_sight": set_product([["in_sight"], ["foe", "friend"], unit_any]),
    "in_reach": set_product([["in_reach"], ["foe", "friend"], source, steps, unit_any]),
    "is_type": set_product([["is_type"], unit_types]),
    "is_dying": set_product([["is_dying"], ["self", "foe", "friend"], thresholds]),
    "is_in_forest": ["is_in_forest"],
}

atomics = {"A": actions, "C": conditions}
grammar = Lark(open("grammar.lark", "r").read())

# %%
all_vars = []
for atomic in atomics.values():
    for variants in atomic.values():
        all_vars += variants


# %%
def txt2expr(txt):
    return grammar.parse(txt)


# %%
def choose_targets(bt_txt, targets):
    targets_txt = "any" if len(targets) == 0 else " or ".join([t for t in targets])
    return bt_txt.replace("any", targets_txt)


def compute_all_variants(bt_txt, unit_types):
    if "any" in bt_txt:
        subsets = [{}] + [set([unit_type]) for unit_type in unit_types]
        variants = [choose_targets(bt_txt, subset) for subset in subsets]
        return subsets, variants
    else:
        return [None], [bt_txt]


def find_bt_variant(x, subsets, variants):
    assert x in subsets
    return variants[subsets.index(x)]


def set_default(bt_txt, default):
    return f"F ( S( C (in_sight foe any) :: {bt_txt}) :: {default})"
