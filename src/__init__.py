from .smax import bullet_fn
from .plot import plot_fn
from .bt import make_bt
from .utils import parse_args

import src.bt as bt
import src.atomics as atomics
import src.smax as smax

scripts = {"bt": bt.main, "atomics": atomics.main, "smax": smax.main}

__all__ = ["bullet_fn", "plot_fn", "make_bt", "parse_args", "scripts"]
