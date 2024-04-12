# bank.py
#   c2sim bt bank
# by: Noah Syrkis

# imports
from lark import Lark
import yaml

# load lark grammar from ../grammar.lark
with open("grammar.lark", "r") as f:
    grammar = f.read()
    parser = Lark(grammar, start="tree")


example = "tree ( sequence ( atomic ( noah ), atomic ( syrkis)))"

print(parser.parse(example))  # tree { sequence { } }
