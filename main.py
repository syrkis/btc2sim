# main.py
#   c2sim main
# by: Noah Syrkis

# imports
import yaml
from jax import random, vmap
from jax import numpy as jnp
from tqdm import tqdm
from functools import partial
from jaxmarl import make
from src import parse_args, scripts, make_bt


# constants
with open("config.yaml", "r") as f:
    conf = yaml.safe_load(f)
    n_envs = conf["n_envs"]
    n_steps = conf["n_steps"]
    n_allies = conf["n_allies"]
    n_enemies = conf["n_enemies"]


# trajectory functions
def step_fn(bt, rng, old_state_v, obs_v, env):  # take a step in the env
    rng, act_rng, step_rng = random.split(rng, 3)
    act_keys = random.split(act_rng, env.num_agents * n_envs).reshape(-1, n_envs, 2)
    step_keys = random.split(step_rng, n_envs)
    acts = {a: bt(act_keys[i], obs_v[a], a) for i, a in enumerate(env.agents)}  # para
    obs_v, state_v, reward_v, _, _ = vmap(env.step)(step_keys, old_state_v, acts)
    print(acts)
    exit()
    return obs_v, (bt, rng, state_v), (step_keys, old_state_v, acts), reward_v


def traj_fn(bt, rng, env, state_seq, reward_seq):  # take n_steps in m env
    rng, reset_rng = random.split(random.PRNGKey(0))  # split rng for reset and step
    reset_keys = random.split(reset_rng, n_envs)  # split reset rng for n_envs
    step = partial(step_fn, env=env)  # partial step function
    obs_v, state_v = vmap(env.reset)(reset_keys)  # initiate envs
    traj_state = (bt, rng, state_v)  # initial state for step_fn
    for _ in tqdm(range(n_steps)):  # take n steps in env and append to lists
        obs_v, traj_state, state_v, reward_v = step(*traj_state, obs_v)  # take step
        state_seq, reward_seq = state_seq + [state_v], reward_seq + [reward_v]
    return state_seq, reward_seq


# main
def main():
    args = parse_args()
    if args.script in scripts:
        scripts[args.script]()
    else:  # run this main
        env = make("SMAX", num_allies=2, num_enemies=5)
        rng = random.PRNGKey(0)
        btv = vmap(make_bt(env, "bt_bank.yaml"), in_axes=(0, 0, None))
        seq = traj_fn(btv, rng, env, [], [])


if __name__ == "__main__":
    main()
