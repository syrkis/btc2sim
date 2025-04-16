# utils.py
#    c2sim utility functions
# by: Noah Syrkis

# imports
from parsimonious.grammar import Grammar
from itertools import product

# grammar stuff
with open("grammar.peg", "r") as f:
    grammar = Grammar(f.read())
    pieces = [m.literal for m in grammar["piece"].members]  # type: ignore
    directions = [m.literal for m in grammar["direction"].members]  # type: ignore
    move_fns = [("stand",)] + [("move", *comb) for comb in list(product(directions, pieces))]
    cond_fns = [("is_alive",)]
    i2v = sorted(move_fns + cond_fns)
    t2i = {var: i for i, var in enumerate(i2v)}
