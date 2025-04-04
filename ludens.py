# %% Imports
from functools import partial

# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=5'  # this for runnning on multiple devices with pmap

# %%
import jax
import jax.numpy as jnp
from jax import random, vmap, jit, tree_util
from einops import repeat

import parabellum as pb
import btc2sim


# %% Constants
places = ["Vesterbro, København, Denmark", "Nørrebro, København, Denmark"]
bt_strs = ["A ( move north )", "A ( move south )"]
n_seeds = 4  # 4 random seeds (parallel starting positions)
n_scene = len(places)  # run with 2 different places
n_model = len(bt_strs)  # run with 2 different models
n_total = n_seeds * n_scene * n_model
n_steps = 100
print(f"Total number of runs: {n_total}")

# %%
scenario_kwargs = {
    'unit_starting_sectors': jnp.array([[0.,0.,0.2,0.2], [0.8,0.8,0.2,0.2]]), 
    'allies_type': 0, 
    'n_allies': 10, 
    'enemies_type': 0,
    'n_enemies': 10,
}


# %% Environment
def envs_fn(places):  # use switch to select place when running (combine with fori_loop on rngs and idxs)
    #maps = list(map(lambda place: pb.map.get_raster(place, 100), places))
    maps = list(map(lambda place: (jnp.zeros((100, 100)), None) , places))
    scenes = list(map(lambda mask: pb.make_scenario(place='Vesterbro, København, Denmark', size=100, **scenario_kwargs), maps))  # this is wrong but close to right, could fix quickly.
    envs = list(map(lambda scene: pb.Environment(scene), scenes))
    return envs


# %% Behavior Tree
def bts_fn(bt_strs):  # <- use switch to select bt when running (combine with fori_loop on rngs and idxs)
    dsl_trees = [btc2sim.dsl.parse(btc2sim.dsl.read(bt_str)) for bt_str in bt_strs]
    bts = [
        vmap(btc2sim.bt.seed_fn(dsl_tree), in_axes=(0, None, None))
        for dsl_tree in dsl_trees
    ]
    return bts


# %%
def bt_fn(bt, obs, env_info, agent_info):  # take actions for all agents in parallel
    acts = tree_util.tree_map(lambda x, i: bt(x, env_info, i)[1], obs, agent_info)  # not working as the bt need the rng now
    return acts


# %%
rng = random.PRNGKey(0)
envs, bts = envs_fn(places), bts_fn(bt_strs)
idxs = tuple([jnp.repeat(jnp.arange(env.num_agents), n_seeds) for env in envs])
bt_fns = tuple([partial(bt_fn, bt) for bt in bts])
env_infos = tuple([btc2sim.info.env_info_fn(env) for env in envs])
agent_infos = tuple([btc2sim.info.agent_info_fn(env) for env in envs])
rngs = repeat(
    random.split(rng, n_scene * n_seeds * n_steps).reshape(n_scene, n_steps, n_seeds, 2),
    "n s b d -> n s x b d",
    x=len(bts),
)


# %%
# bt_fns and env are static arguments
# @partial(jit, static_argnums=(5, 6))
def step_fn(idx, env_info, agent_info, env, bt_fns):
    def step(carry, rng):
        obs, state = carry
        acts = jax.lax.map(
            lambda i: jax.lax.switch(
                i, bt_fns, tree_util.tree_map(lambda x: x[i], obs), env_info, agent_info
            ),
            jnp.arange(n_model),
        )
        obs, state, rewards, dones, infos = vmap(vmap(env.step))(rng, state, acts)
        return (obs, state), acts
    return step


# %%
for rng, env, idx, env_info, agent_info in zip(
    rngs, envs, idxs, env_infos, agent_infos
):  # <- replace with fori_loop
    step = jit(step_fn(idx, env_info, agent_info, env, bt_fns))
    obs, state = vmap(vmap(env.reset))(rng[0])
    acts = jnp.zeros((n_seeds * n_scene * n_model, env.num_agents), dtype=jnp.int32)
    state_seq = jax.lax.scan(step, (obs, state), rng[1:])
    break

# %%
