# %% Imports
import c2sim
import parabellum as pb
import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from tqdm import tqdm

# %% Constants
place = 'Vesterbro, København, Denmark'
n_envs = 10

# %% Environment
mask, _ = pb.terrain_fn(place, 100)
scene = pb.make_scenario(place, mask)
env = pb.Environment(scene)

# %% Initialization
rngs = random.split(random.PRNGKey(0), n_envs + 1)
obs, state = vmap(env.reset)(rngs[1:])
info = c2sim.agent.info_fn(env)

# %% TODO: use multiple trees
bt_str = "A ( move north )"
dsl_tree = c2sim.dsl.parse(c2sim.dsl.read(bt_str))
bt = vmap(c2sim.bt.seed_fn(dsl_tree, env, info))

# %% Run
def action_fn(obs):
    batch_obs = batchify(obs)
    idxs = jnp.arange(batch_obs.shape[0]) % (scene.num_allies + scene.num_enemies)
    batch_act = bt(batch_obs, idxs.T)[1]
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
    action = action_fn(obs)  # bt returns state and action, but we only want action
    rngs = random.split(rngs[0], n_envs + 1)
    state_seq += [((rngs[1:], state, action))]  # save state for visualization
    obs, state, reward, done, info = vmap(env.step)(rngs[1:], state, action)

# %%
vis = pb.Visualizer(env, state_seq, skin=pb.Skin(maskmap=mask))
vis.animate()
