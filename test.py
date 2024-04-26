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
    args = (state, obs["enemy_0"], "enemy_0", env)
    print(A.in_sight("friend_0", "west")(*args))  # region test


"""     
    print(A.in_region("west", "center")(*args))  # region test
    print(A.attack("target_0")(*args))  # attack test
    print(A.move("north")(*args))  # move test
    print(A.locatable("target_1", "east")(*args))  # locate
"""


def main():
    atomics_test()


if __name__ == "__main__":
    main()
