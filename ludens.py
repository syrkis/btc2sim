# %% Imports
import os

from jax._src import tree_util
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=5'
import c2sim
import parabellum as pb
import jax
import jax.numpy as jnp
from jax import random, vmap, jit, pmap
from tqdm import tqdm

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
def acts_fn(obs, env_info, agent_info):  # take actions for all agents in parallel
    idxs = jnp.repeat(jnp.arange(env.num_agents), parallel_envs)  # this can be done outside of the function only once
    batched_obs = batchify(obs)
    batched_acts = bt(batched_obs, idxs, env_info, agent_info)[1].astype(jnp.int32)  # first value is the state, second is the action
    acts = unbatchify(batched_acts, idxs, obs)
    return acts


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
envs, bts = envs_fn(places), bts_fn(bt_strs)
rngs = random.split(random.PRNGKey(0), len(envs) * parallel_envs).reshape(len(envs), parallel_envs, 2)

# %%
for rng, env in zip(rngs, envs):  # <- replace with fori_loop and switch
    obs, state = vmap(env.reset)(rng)
    env_info = c2sim.info.env_info_fn(env)
    agent_info = c2sim.info.agent_info_fn(env)
    for j, bt in enumerate(bts):  # <- replace with fori_loop and switch (use pmap)
        for _ in range(100):  # <- replace with scan though rngs
            acts = acts_fn(obs, env_info, agent_info)
            obs, state, _, _, _= vmap(env.step)(rng, state, acts)
        break
    break

# %%
