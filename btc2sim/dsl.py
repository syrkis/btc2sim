# %% Imports
from lark import Lark
import os, sys
import btc2sim
import jax
from functools import partial


# %%
qualifiers = [
    "String",
    "Direction",
    "Foe",
    "Friend",
    "Qualifier",
    "Sense",
    "Self",
    "Unit",
    "Negation",
    "Any",
    "Source",
    "Time",
    "Intensity",
]

# %% Constants
grammar_txt = """
?start: node

%import common.WS
%ignore WS

nodes : node ("::" node | "|>" node)*
node  :
    | sequence
    | fallback
    | action
    | condition

sequence  : "S" "(" nodes ")"
fallback  : "F" "(" nodes ")"
action    : "A" "(" atomic ")"
condition : "C" "(" atomic ")"

atomic :
    | move
    | attack
    | stand
    | follow_map
    | in_sight
    | in_reach
    | is_dying
    | is_armed
    | is_flock
    | is_type 
    | is_in_forest
    | success_action
    | failure_action
    

move      : "move" (direction | sense qualifier (foe | friend) (unit ("or" unit)* |any)?)
attack    : "attack" qualifier (unit ("or" unit)* |any)?
stand     : "stand"
in_sight  : "in_sight" (foe | friend) (unit ("or" unit)* |any)?
in_reach  : "in_reach" (foe | friend) source time (unit ("or" unit)* |any)?
is_dying  : "is_dying" (self | foe | friend) intensity
is_armed  : "is_armed" (self | foe | friend)
is_flock  : "is_flock" (foe | friend) direction
is_type   : "is_type" negation unit
follow_map : "follow_map" sense intensity?
is_in_forest : "is_in_forest" 
success_action : "success_action" 
failure_action: "failure_action"

sense     : /toward|away_from/
direction : /north|east|south|west|center/
foe       : /foe/
friend    : /friend/
qualifier : /strongest|weakest|closest|farthest|random/
intensity : /low|middle|high/
self      : /self/
unit      : /spearmen|archer|cavalry|balista|dragon|civilian/
any       : /any/
negation  : /a|not_a/
source    : /them_from_me|me_from_them/
time      : /now|low|middle|high/
"""

grammar = Lark(grammar_txt, start="start")


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
            return [parse(child) for child in lark_tree.children]  # type: ignore
        case "Action" | "Condition":
            return (lark_tree.data.title().lower(), parse(lark_tree.children[0]))  # type: ignore
        case _:  # Sequence or Fallback or Decorator
            value = [parse(child) for child in lark_tree.children]
            return lark_tree.data.title().lower(), value[0] if len(
                value
            ) == 1 else value  # type: ignore
