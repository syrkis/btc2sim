# %% main.py
#   btc2sim main file
# by: Noah Syrkis

# Imports
import jax.numpy as jnp
import parabellum as pb
from jax import debug, lax, random, tree, vmap
from jax_tqdm import scan_tqdm
from omegaconf import OmegaConf

import btc2sim as b2s

# %% Behavior trees
bt_strs = """
F ( S ( C in_range enemy |> A shoot closest ) |> A move target )
"""
# A move target

# %% Constants
cfg = OmegaConf.load("conf.yaml")
rng, key = random.split(random.PRNGKey(0))
env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)
scene.terrain.building = b2s.utils.scene_fn(scene.terrain.building)
marks = jnp.int32(random.uniform(rng, (2, 2), minval=0, maxval=100))
targets = random.randint(rng, (env.num_units,), 0, marks.shape[0])
gps = b2s.gps.gps_fn(scene, marks)  # 6, key)
rngs = random.split(rng, cfg.steps)
obs, state = env.reset(key, scene)


# %% Functions
@scan_tqdm(n=cfg.steps)
def step_fn(carry, input):
    (_, rng), (obs, state) = input, carry
    rngs = random.split(rng, env.num_units)
    behavior = plan_fn(rng, None, state)
    action = action_fn(rngs, obs, behavior, env, scene, gps, targets)
    obs, state = env.step(rng, scene, state, action)
    # debug.breakpoint()
    return (obs, state), state


def plan_fn(rng, plan, state):  # TODO: Focus on this for now. Currently broken
    idxs = random.randint(rng, (env.num_units,), 0, bts.idx.shape[0])  # random bt idxs for units
    behavior = tree.map(lambda x: jnp.take(x, idxs, axis=0), bts)  # behavior
    return behavior  # TODO: Maybe add target to behavior


action_fn = vmap(b2s.act.action_fn, in_axes=(0, 0, 0, None, None, None, 0))
bts = b2s.dsl.bts_fn(bt_strs)
state, seq = lax.scan(step_fn, (obs, state), (jnp.arange(cfg.steps), rngs))
b2s.utils.svg_fn(scene, seq)
# debug.breakpoint()

# exit()
# debug.breakpoint()
# behavior, _ = plan_fn(None, state)
