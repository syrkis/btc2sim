# __init__.py
#   c2sim package
# by: Noah Syrkis

# imports
from .smax import bullet_fn
from .plot import plot_fn
from .bt import make_bt
from .utils import parse_args
from .bank import grammar_fn, parse_fn, dict_fn

import src.bt as bt
import src.atomics as atomics
import src.smax as smax
import src.bank as bank

# scripts
scripts = {"bt": bt.main, "atomics": atomics.main, "smax": smax.main, "bank": bank.main}

# exports
__all__ = [
    "bullet_fn",
    "plot_fn",
    "make_bt",
    "parse_args",
    "scripts",
    "grammar_fn",
    "parse_fn",
    "dict_fn",
]
