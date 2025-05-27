# %% main.py
#   btc2sim main file
# by: Noah Syrkis

# Imports
import jax.numpy as jnp
import parabellum as pb
from jax import debug, lax, random, tree, vmap
from jax_tqdm import scan_tqdm
from omegaconf import DictConfig

import btc2sim as b2s


# %% Config #####################################################
num_sim = 9
loc = dict(place="Tietgenkollegiet, Copenhagen, Denmark", size=100)
red = dict(plane=1, soldier=1)
blue = dict(plane=1, soldier=1)
cfg = DictConfig(dict(steps=100, knn=4, blue=blue, red=red) | loc)


# %% Behavior trees ( in range should be in reach )
bt_strs = """
F ( S ( C in_range enemy |> A shoot closest ) |> A move target )
"""

# %% Constants
env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)
scene.terrain.building = b2s.utils.scene_fn(scene.terrain.building)
bts = b2s.dsl.bts_fn(bt_strs)
action_fn = vmap(b2s.act.action_fn, in_axes=(0, 0, 0, None, None, None, 0))

rng, key = random.split(random.PRNGKey(0))
marks = jnp.int32(random.uniform(rng, (2, 2), minval=0, maxval=100))
targets = random.randint(rng, (env.num_units,), 0, marks.shape[0])
gps = b2s.gps.gps_fn(scene, marks)  # 6, key)


# %% Functions
@scan_tqdm(n=cfg.steps)
def step_fn(carry, input):
    (_, rng), (obs, state) = input, carry
    rngs = random.split(rng, env.num_units)
    behavior = plan_fn(rng, None, state)
    action = action_fn(rngs, obs, behavior, env, scene, gps, targets)
    obs, state = env.step(rng, scene, state, action)
    return (obs, state), state


def plan_fn(rng, plan, state):  # TODO: Focus on this for now. Currently broken
    idxs = random.randint(rng, (env.num_units,), 0, bts.idx.shape[0])  # random bt idxs for units
    behavior = tree.map(lambda x: jnp.take(x, idxs, axis=0), bts)  # behavior
    return behavior  # TODO: Maybe add target to behavior


# maybe vmap here
@vmap
def traj_fn(obs, state, rngs):
    state, seq = lax.scan(step_fn, (obs, state), (jnp.arange(cfg.steps), rngs))
    return state, seq


obs, state = vmap(env.reset, in_axes=(0, None))(random.split(key, num_sim), scene)
rngs = random.split(rng, (num_sim, cfg.steps))
state, seq = traj_fn(obs, state, rngs)
b2s.utils.svgs_fn(scene, seq)
