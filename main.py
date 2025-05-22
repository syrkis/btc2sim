# %% main.py
#   btc2sim main file
# by: Noah Syrkis

# Imports
import esch
import jax.numpy as jnp
import numpy as np
import parabellum as pb
from einops import rearrange
from jax import debug, lax, random, tree, vmap
from jax_tqdm import scan_tqdm
from omegaconf import OmegaConf

import btc2sim as b2s

# %% Constants
cfg = OmegaConf.load("conf.yaml")
rng, key = random.split(random.PRNGKey(0))
env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)
scene.terrain.building = b2s.utils.scene_fn(scene.terrain.building)
marks = jnp.int32(random.uniform(rng, (23, 2), minval=0, maxval=100))
targets = random.randint(rng, (env.num_units,), 0, marks.shape[0])
gps = b2s.gps.gps_fn(scene, marks)  # 6, key)
rngs = random.split(rng, cfg.steps)
obs, state = env.reset(key, scene)

# bts = b2s.dsl.txt2bts(open("bts.txt", "r").readline())
bts = b2s.dsl.file2bts("bts.txt")
action_fn = vmap(b2s.act.action_fn, in_axes=(0, 0, 0, None, None, None, 0))


# %% Functions
def plan_fn(plan, state):  # plan
    target = random.randint(rng, (env.num_units,), 0, marks.shape[0])  # random targets
    idxs = random.randint(rng, (env.num_units,), 0, 2)  # random bt idxs for units
    behavior = tree.map(lambda x: jnp.take(x, idxs, axis=0), bts)  # behavior
    return behavior, target  # TODO: Maybe add target to behavior


@scan_tqdm(n=cfg.steps)
def step_fn(carry, input):
    (_, rng), (obs, state) = input, carry
    # behavior, target = plan_fn(None, state)
    rngs = random.split(rng, env.num_units)
    action = action_fn(rngs, obs, behavior, env, scene, gps, targets)
    obs, state = env.step(rng, scene, state, action)
    return (obs, state), state


def svg_fn(scene, seq):
    dwg = esch.init(100, 100)
    esch.grid_fn(np.array(scene.terrain.building).T, dwg)
    arr = np.array(rearrange(seq.coords[:, :, ::-1], "time unit coord -> unit coord time"), dtype=np.float32)
    esch.anim_sims_fn(arr, dwg, fps=24)
    esch.save(dwg, "/Users/nobr/desk/s3/btc2sim/test.svg")


behavior, target = plan_fn(None, state)
state, seq = lax.scan(step_fn, (obs, state), (jnp.arange(cfg.steps), rngs))
svg_fn(scene, seq)
