# test.py
#   c2sim tests
# by: Noah Syrkis

# imports
import src.atomics as A
import yaml
from jax import random, vmap, jit
from jax import numpy as jnp
from tqdm import tqdm
from functools import partial
from jaxmarl import make
from src import parse_args, scripts, make_bt, plot_fn, grammar_fn, parse_fn, dict_fn


# test atomics
def atomics_test():
    env = make("SMAX")
    obs, state = env.reset(random.PRNGKey(0))
    print(A.move("north")())  # move test
    print(A.attack(0)())  # attack test
    print(A.region("north", "west")(state, obs["ally_0"], "ally_0", env))  # region test
    print(A.locate("target_1", "east")(state, obs["ally_0"], "ally_0", env))  # locate


def main():
    atomics_test()


if __name__ == "__main__":
    main()
