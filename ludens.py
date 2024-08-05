# %% Imports
import c2sim
import parabellum as pb
import jax
import jax.numpy as jnp
from jax import random, vmap, jit
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
    bts = [vmap(c2sim.bt.seed_fn(dsl_tree), in_axes=(0, None, None)) for dsl_tree in dsl_trees]
    return bts

# %%
def acts_fn(obs):  # take actions for all agents in parallel
    batch_obs = batchify(obs)

# %%
def batchify(obs):
    all_obs = [v for k, v in obs.items() if k != 'world_state']
    batched_obs = jnp.stack(all_obs).reshape((len(obs) - 1) * n_envs, -1)
    return batched_obs

# %%
def unbatchify(batched_obs, obs):
    keys = [k for k in obs.keys() if k != 'world_state']
    unbatched_obs = {k: batched_obs[i * parallel_envs: (i + 1) * parallel_envs] for i, k in enumerate(keys)}
    return unbatched_obs


# %%
envs, bts = envs_fn(places), bts_fn(bt_strs)
rngs = random.split(random.PRNGKey(0), len(envs) * len(bts) * parallel_envs).reshape(len(envs), len(bts), parallel_envs, 2)

# %%
for i, env in enumerate(envs):  # <- replace with fori_loop and switch
    env_info = c2sim.agent.env_info_fn(env)

    for j, bt in enumerate(bts):  # <- replace with fori_loop and switch
        rng = rngs[i, j]
        obs, state = vmap(env.reset)(rng)
        for i in range(100):  # <- replace with scan
            acts = {a: bt(obs[a], env_info, c2sim.agent.agent_info_fn(env, a))[1] for a in env.agents} # <- replace with batching
            obs, state, _, _, _= vmap(env.step)(rng, state, acts)
            break
        break
    break
