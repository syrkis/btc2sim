# main.py
#   c2sim main
# by: Noah Syrkis

# %% imports
import parabellum as pb

import yaml
from tqdm import tqdm
from functools import partial

from jax import random, vmap, jit
from jax import numpy as jnp
from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario as n2s
from lark import Lark
from time import time
import c2sim

# %% constants
with open("config.yaml", "r") as f:
    conf = yaml.safe_load(f)
n_envs = conf["n_envs"]
n_trees = conf["n_trees"]
n_steps = conf["n_steps"]
with open("grammar.lark", "r") as f:
    grammar = f.read()


# %% trajectory functions
def step_fn(
    bts, enemy_btv, old_state_v, steps_rng, obs_v, env, n_envs, n_trees
):  # take a step in the env
    times = {}
    times["SMAX_acts"] = time()
    bts_idxs = jnp.arange(n_envs * n_trees) // n_envs
    acts, nodes_id = {}, {}
    for i, a in enumerate(env.agents):
        if i < env.num_allies:
            action, node_id = bts(old_state_v, bts_idxs, obs_v[a], a, env)
        else:
            _, action, node_id = enemy_btv(old_state_v, obs_v[a], a, env)
        acts[a] = action
        nodes_id[a] = node_id

    times["SMAX_step"] = time()
    steps_rng = jnp.array([steps_rng for _ in range(n_envs * n_trees)])
    obs_v, state_v, reward_v, done, info = vmap(env.step)(steps_rng, old_state_v, acts)
    times["void2"] = time()
    return (
        obs_v,
        (bts, enemy_btv, state_v),
        (steps_rng, old_state_v, acts),
        (reward_v, times, nodes_id),
    )


def traj_fn(
    reset_rng, steps_rng, btv, enemy_btv, env, n_envs, n_trees, n_steps
):  # take n_steps in m env
    state_seq, reward_seq, obs_seq = [], [], []
    obs_v, state_v = vmap(env.reset)(reset_rng)  # initiate envs
    traj_state = (btv, enemy_btv, state_v)  # initial state for step_fn
    for i in range(n_steps):  # take n steps in env and append to lists
        obs_v, traj_state, state_v, reward_v = step_fn(
            *traj_state, steps_rng[i], obs_v, env, n_envs, n_trees
        )
        state_seq.append(state_v)
        reward_seq.append(reward_v)
        obs_seq.append(obs_v)
    return state_seq, reward_seq, obs_seq


def trees_fn(bts, use_jit=True):
    def bts_fn(env_state, idx, obs, agent, env):
        state, action, selected_node_id = Status.FAILURE, STAND, -1
        for i, bt in enumerate(bts):
            tree_state, tree_action, node_id = bt["tree"](env_state, obs, agent, env)
            state = jnp.where(idx == i, tree_state, state)
            action = jnp.where(idx == i, tree_action, action)
            selected_node_id = jnp.where(idx == i, node_id, selected_node_id)
        return action, selected_node_id

    bts_fn = vmap(bts_fn, in_axes=(0, 0, 0, None, None), out_axes=(0, 0))
    if use_jit:
        bts_fn = jit(bts_fn, static_argnums=(3, 4))
    return bts_fn


# @partial(jit, static_argnums=(1, 2, 3))
def run_fn(
    reset_rng, steps_rng, bts, enemy_bts, envs, n_envs, n_trees, n_steps, verbose=True
):
    # rngs = random.split(rng, len(envs))
    seqs = []
    for i, env in (
        tqdm(enumerate(envs), total=len(envs)) if verbose else enumerate(envs)
    ):
        seqs.append(
            traj_fn(
                reset_rng, steps_rng, bts, enemy_bts[i], env, n_envs, n_trees, n_steps
            )
        )
    return seqs


def tree2smaxbt(bt):
    return make_bt(dict_fn(Lark(grammar, start="node").parse(bt)))


def main():
    args = parse_args()

    if args.script in scripts:
        scripts[args.script]()

    else:
        bts = trees_fn(load_trees())
        envs = tuple([make("SMAX", scenario=n2s(s)) for s in ["c2sim"]])
        bt = "F(A ( attack closest) :: A (stand))"
        enemy_bts = [
            vmap(tree2smaxbt(bt), in_axes=(0, 0, None, None), out_axes=(0, 0, 0))
            for _ in range(len(scenarios))
        ]  # you can choose a different ennemy for each scenario
        enemy_bts = [jit(bt, static_argnums=(2, 3)) for bt in enemy_bts]
        rng = random.PRNGKey(0)
        reset_rng, steps_rng = random.split(rng)
        reset_rng = random.split(
            reset_rng, n_envs
        )  # the reset only depends to the number of // evaluations
        reset_rng = jnp.concatenate(
            [reset_rng for _ in range(n_trees)]
        )  # copy the reset rng for each // BTs
        steps_rng = random.split(
            steps_rng, n_steps
        )  # the steps are the same for all evaluations so that the order of the BTs does not influence the result
        seqs = run_fn(
            reset_rng, steps_rng, bts, enemy_bts, envs, n_envs, n_trees, n_steps
        )
        print(len(seqs))
        # plot_fn(env, seq[0], seq[1], expand=True)


# + active=""
# if __name__ == "__main__":
#     main()
# -
