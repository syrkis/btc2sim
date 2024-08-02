# %% Imports
import c2sim
import parabellum as pb
import jax
import jax.numpy as jnp
from tqdm import tqdm
from jax import random, vmap, jit

# %% Constants
place = 'Vesterbro, København, Denmark'
n_envs = 10

# %% Environment
mask, photo = pb.terrain_fn(place, 100)
scen = pb.make_scenario(place, mask)
env = pb.Environment(scen)

# %%
bt_str = "A ( move north )"
dsl_tree = c2sim.dsl.parse(c2sim.dsl.read(bt_str))
bt = vmap(c2sim.bt.seed_fn(dsl_tree, env), in_axes=(0, None))

# %% Initialization
rngs = random.split(random.PRNGKey(0), n_envs + 1)
obs, state = vmap(env.reset)(rngs[1:])
infos = {agent: c2sim.agent.info_fn(agent, env) for agent in env.agent_ids}

# %% Run
state_seq = []
for i in tqdm(range(100)):
    actions = {agent: bt(obs[agent], infos[agent])[1] for agent in env.agents}  # bt returns state and action, but we only want action
    rngs = random.split(rngs[0], n_envs + 1)
    state_seq += [((rngs[1:], state, actions))]
    obs, state, reward, done, info = vmap(env.step)(rngs[1:], state, actions)

# %%
vis = pb.Visualizer(env, state_seq, skin=pb.Skin(maskmap=mask))
vis.animate()
