# main.py
#   c2sim main
# by: Noah Syrkis

# imports
import yaml
from jax import random, vmap, jit
from jax import numpy as jnp
from tqdm import tqdm
from functools import partial
from jaxmarl import make
from src import parse_args, scripts, make_bt, plot_fn, grammar_fn, parse_fn, dict_fn
from src.utils import Status, STAND, DEFAULT_BT


# constants

with open("config.yaml", "r") as f:
    conf = yaml.safe_load(f)
    n_envs = conf["n_envs"]
    n_steps = conf["n_steps"]
    n_allies = conf["n_allies"]
    n_enemies = conf["n_enemies"]


# trajectory functions
def step_fn(btv, rng, old_state_v, obs_v, env):  # take a step in the env
    rng, step_rng = random.split(rng)
    step_keys = random.split(step_rng, n_envs)
    acts = {a: btv(old_state_v, obs_v[a], a)[1] for i, a in enumerate(env.agents)}
    obs_v, state_v, reward_v, done, info = vmap(env.step)(step_keys, old_state_v, acts)
    return obs_v, (btv, rng, state_v), (step_keys, old_state_v, acts), reward_v


def traj_fn(btv, rng, env, state_seq, reward_seq):  # take n_steps in m env
    rng, reset_rng = random.split(random.PRNGKey(0))  # split rng for reset and step
    reset_keys = random.split(reset_rng, n_envs)  # split reset rng for n_envs
    obs_v, state_v = vmap(env.reset)(reset_keys)  # initiate envs
    traj_state = (btv, rng, state_v)  # initial state for step_fn
    for _ in tqdm(range(n_steps)):  # take n steps in env and append to lists
        obs_v, traj_state, state_v, reward_v = step_fn(*traj_state, obs_v, env)
        state_seq.append(state_v)
        reward_seq.append(reward_v)
    return state_seq, reward_seq


def metric_fn(env, state_seq, reward_seq):  # calculate metrics
    final_ally_health = state_seq[-1][1].unit_health[:, :n_allies].sum(axis=1)
    final_enemy_health = state_seq[-1][1].unit_health[:, n_allies:].sum(axis=1)
    return final_ally_health[:, None], final_enemy_health[:, None]


def main():
    args = parse_args()

    if args.script in scripts:
        scripts[args.script]()

    if args.script == "main":
        tree = dict_fn(grammar_fn().parse(DEFAULT_BT))
        env = make("SMAX")
        btv = vmap(make_bt(env, tree), in_axes=(0, 0, None))
        rng = random.PRNGKey(0)
        seq = traj_fn(btv, rng, env, [], [])  # seq[0][0][2] is the first action dict
        plot_fn(env, seq[0], seq[1], expand=False)


if __name__ == "__main__":
    main()
