# %% Imports
import c2sim
import parabellum as pb
import jax
import jax.numpy as jnp
from jax import random, vmap, jit

# %% Constants
place = 'Vesterbro, Copenhagen, Denmark'

# %% Environment
mask = pb.terrain_fn(place, 1000)
scen = pb.make_scenario(place, mask)
env = pb.Environment(scen)

# %%
bt_str = "F ( A ( attack weakest) |> A ( move toward weakest foe) |> A ( move center) )"
dsl_tree = c2sim.dsl.parse(c2sim.dsl.read(bt_str))
bt = c2sim.dsl.grow(dsl_tree, env)

# %% Initialization
rng = random.PRNGKey(0)
obs, state = env.reset(rng)
arg_fn = lambda agent: (state, obs[agent], agent, env)

# %% Run
bt(*arg_fn('ally_0'))  # type: ignore
