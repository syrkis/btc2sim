# %% Imports
import os
from tqdm import tqdm
from functools import partial

# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=5'  # this for runnning on multiple devices with pmap

import jax
import jax.numpy as jnp
from jax import random, vmap, jit, pmap, tree_util

import parabellum as pb
import c2sim


# %% Constants
places = ['Vesterbro, København, Denmark', 'Nørrebro, København, Denmark']
bt_strs = ["A ( move north )", "A ( move south )"]
parallel_envs = 2

# %% Environment
def envs_fn(places):  # use switch to select place when running (combine with fori_loop on rngs and idxs)
    maps = list(map(lambda place: pb.terrain_fn(place, 100), places))
    scenes = list(map(lambda mask: pb.make_scenario(terrain_raster=mask[0], place='mask'), maps))  # this is wrong but close to right, could fix quickly.
    envs = list(map(lambda scene: pb.Environment(scene), scenes))
    return envs

# %% Behavior Tree
def bts_fn(bt_strs):  # <- use switch to select bt when running (combine with fori_loop on rngs and idxs)
    dsl_trees = [c2sim.dsl.parse(c2sim.dsl.read(bt_str)) for bt_str in bt_strs]
    bts = [vmap(c2sim.bt.seed_fn(dsl_tree), in_axes=(0, 0, None, None)) for dsl_tree in dsl_trees]
    return bts


# %%
def batchify(obs):
    all_obs = [v for k, v in obs.items() if k != 'world_state']
    batched_obs = jnp.stack(all_obs).reshape((len(obs) - 1) * parallel_envs, -1)
    return batched_obs

# %%
def unbatchify(batched_acts, idxs, obs):
    acts = {a: batched_acts[idxs == i] for i, a in zip(jnp.arange(idxs.max()+1), obs.keys()) if a != 'world_state'}
    return acts


# %%
def bt_fn(bt, batched_obs, env_info, agent_info):  # take actions for all agents in parallel
    batched_acts = bt(batched_obs, idxs, env_info, agent_info)[1].astype(jnp.int32)
    return batched_acts
    # acts = unbatchify(batched_acts, idxs, obs)
    # return acts



# %%
envs, bts = envs_fn(places), bts_fn(bt_strs)
bt_fns = [partial(bt_fn, bt) for bt in bts]
rngs = random.split(random.PRNGKey(0), len(envs) * parallel_envs).reshape(len(envs), parallel_envs, 2)

# %%
fns = [env.step for env in envs]
for rng, env in zip(rngs, envs):  # <- replace with fori_loop and switch
    idxs = jnp.repeat(jnp.arange(env.num_agents), parallel_envs)
    env_info = c2sim.info.env_info_fn(env)
    agent_info = c2sim.info.agent_info_fn(env)
    obs, state = vmap(env.reset)(rng)
    for i in range(100):  # <- replace with scan though rngs
        batched_obs = batchify(obs)
        batched_acts = jax.lax.map(lambda i: jax.lax.switch(i, bt_fns, batched_obs, env_info, agent_info), jnp.arange(len(bts)))
        acts = vmap(unbatchify, in_axes=(0, None, None))(batched_acts, idxs, obs)
        # obs, state, *_ = jax.lax.map(vmap(env.step))(rng, state, acts)
        print(batched_acts, acts)
        exit()
        # for bt in bts:  # replace with fori_loop and switch
            # obs, state = for_fn(bt, env, rng, obs, state, env_info, agent_info)
            # bt_fn(bt, env, rng, obs, state, env_info, agent_info)
        break
    break

# %%
