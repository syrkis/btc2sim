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
n_envs = 10

# %% Environment
def envs_fn(places):  # use switch to select place when running (combine with fori_loop on rngs and idxs)
    maps = list(map(lambda place: pb.terrain_fn(place, 100), places))
    scenes = list(map(lambda mask: pb.make_scenario(terrain_raster=mask[0], place='mask'), maps))  # this is wrong but close to right, could fix quickly.
    envs = list(map(lambda scene: pb.Environment(scene), scenes))
    return envs

# %% Behavior Tree
def bts_fn(bt_strs):  # <- use switch to select bt when running (combine with fori_loop on rngs and idxs)
    dsl_trees = [c2sim.dsl.parse(c2sim.dsl.read(bt_str)) for bt_str in bt_strs]
    bts = [vmap(c2sim.bt.seed_fn(dsl_tree)) for dsl_tree in dsl_trees]
    return bts

# %% Agent
def main_fn(envs, bts):
    envs = envs_fn(places)
    bts = bts_fn(bt_strs)
    rngs = random.split(random.PRNGKey(0), n_envs + 1)
    obs, state = vmap(env.reset)(rngs[1:])


# %% Initialization
rngs = random.split(random.PRNGKey(0), n_envs + 1)
obs, state = vmap(env.reset)(rngs[1:])
agent_info, env_info = c2sim.agent.agent_info_fn(env, n_envs), c2sim.agent.env_info_fn(env)
print(agent_info, env_info)
exit()

# %% TODO: use multiple trees

# %% Run


def acts_fn(obs):  # take actions for all agents in parallel
    batch_obs = batchify(obs)
    idxs = jnp.arange(batch_obs.shape[0]) % (scene.num_allies + scene.num_enemies)
    batch_act = bt(batch_obs, agent_info, idxs.T)[1]
    print(batch_act)
    exit()
    # return unbatchify(batch_act)

def batchify(obs):
    all_obs = [v for k, v in obs.items() if k != 'world_state']
    batched_obs = jnp.stack(all_obs).reshape((len(obs) - 1) * n_envs, -1)
    return batched_obs

def unbatchif(batched_obs, obs):
    return obs

state_seq = []
for i in tqdm(range(100)):
    action = acts_fn(obs)  # bt returns state and action, but we only want action
    rngs = random.split(rngs[0], n_envs + 1)
    state_seq += [((rngs[1:], state, action))]  # save state for visualization
    obs, state, reward, done, info = vmap(env.step)(rngs[1:], state, action)

# %%
vis = pb.Visualizer(env, state_seq, skin=pb.Skin(maskmap=mask))
vis.animate()
