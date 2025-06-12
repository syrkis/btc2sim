# %% main.py
#   btc2sim main file
# by: Noah Syrkis

# Imports
import jax.numpy as jnp
import parabellum as pb
from jax import lax, random, tree, vmap
from typing import Tuple
from jaxtyping import Array
from functools import partial
import aic2sim as a2s


# %% Config #####################################################
num_sim = 4
cfg = pb.types.Config()


# %% Behavior trees ( in range should be in reach )
with open("data/bts.txt", "r") as f:
    bts_str: str = f.read().strip()

with open("data/plan.txt", "r") as f:
    pln_str: str = f.read().strip().split("---")[0].strip()

with open("data/prompt.txt", "r") as f:
    llm_str: str = f.read().strip()


# %% Constants
env = pb.env.Env(cfg=cfg)
bts = a2s.dsl.bts_fn(bts_str)
action_fn = vmap(a2s.act.action_fn, in_axes=(0, 0, 0, None, None, None, 0))

rng, key = random.split(random.PRNGKey(0))
marks = jnp.int32(random.uniform(rng, (1, 2), minval=0, maxval=cfg.size))
targets = random.randint(rng, (cfg.length,), 0, marks.shape[0])
gps = a2s.gps.gps_fn(cfg, marks)  # 6, key)


# %% Functions
def step_fn(env: pb.env.Env, cfg: pb.types.Config, carry: Tuple[pb.types.Obs, pb.types.State], rng: Array):
    obs, state = carry
    rngs = random.split(rng, cfg.length)
    behavior = a2s.act.plan_fn(rng, bts, plan, state)  # perhaps only update plan every m steps
    action = action_fn(rngs, obs, behavior, env, gps, targets)
    obs, state = env.step(rng, cfg, state, action)
    return (obs, state), (state, action)


# %%
plan = tree.map(lambda *x: jnp.stack(x), *tuple(map(partial(a2s.lxm.str_to_plan, pln_str, cfg), (-1, 1))))  # type: ignore
rngs = random.split(rng, (num_sim, cfg.steps))
obs, state = vmap(env.reset, in_axes=(0, None))(random.split(key, num_sim), cfg)
state, (seq, action) = lax.scan(vmap(partial(step_fn, env, cfg)), (obs, state), rngs)
pb.utils.svg_fn(cfg, tree.map(lambda x: x[0], seq), tree.map(lambda x: x[0], action), fps=10)
