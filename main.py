# %% main.py
#   btc2sim main file
# by: Noah Syrkis

# Imports
import jax.numpy as jnp
import parabellum as pb
from jax import debug, lax, random, tree, vmap
import equinox as eqx
from jax_tqdm import scan_tqdm
from jaxtyping import Array
from omegaconf import DictConfig
from functools import partial

import btc2sim as b2s

# %% Config #####################################################
num_sim = 4
loc = dict(place="Palazzo della Civiltà Italiana, Rome, Italy", size=64)
red = dict(infantry=6, armor=6, airplane=6)
blue = dict(infantry=6, armor=6, airplane=6)
cfg = DictConfig(dict(steps=300, knn=4, blue=blue, red=red) | loc)


# %% Behavior trees ( in range should be in reach )
bt_strs = """
F ( S ( C in_range enemy |> A shoot random ) |> A move target )
"""


# %% Constants
env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)
bts = b2s.dsl.bts_fn(bt_strs)
action_fn = vmap(b2s.act.action_fn, in_axes=(0, 0, 0, None, None, None, 0))

rng, key = random.split(random.PRNGKey(0))
marks = jnp.int32(random.uniform(rng, (6, 2), minval=0, maxval=cfg.size))
targets = random.randint(rng, (env.num_units,), 0, marks.shape[0])
gps = b2s.gps.gps_fn(scene, marks)  # 6, key)


# %% Functions
@scan_tqdm(n=cfg.steps)
def step_fn(carry, input):
    (_, rng), (obs, state) = input, carry
    rngs = random.split(rng, env.num_units)
    behavior = plan_fn(rng, plan, state, scene)  # perhaps only update plan every m steps
    action = action_fn(rngs, obs, behavior, env, scene, gps, targets)
    obs, state = env.step(rng, scene, state, action)
    return (obs, state), state


@eqx.filter_jit
def plan_fn(rng: Array, plan: b2s.types.Plan, state: pb.types.State, scene: pb.types.Scene):  # TODO: Focus
    def move(step):  # all units in focus within 10 meters of target position
        return ((jnp.linalg.norm(state.coords - step.coord) * step.units) < 10).all()

    def kill(step):  # all enemies dead within 10 meters of target
        return ((jnp.linalg.norm(state.coords - step.coord) * ~step.units * (state.health == 0)) < 10).any()

    def aux(plan: b2s.types.Plan):
        idx = lax.map(lambda step: lax.cond(step.move, move, kill, step), plan)
        idxs = plan.btidx[idx.argmin()] * plan.units[idx.argmin()]
        return idxs

    idxs = lax.map(aux, plan).sum(0)  # mapping across teams (2 for now, but supports any number)
    return tree.map(lambda x: jnp.take(x, idxs, axis=0), bts)  # behavior


# maybe vmap here
def traj_fn(obs, state, rngs):
    state, seq = lax.scan(step_fn, (obs, state), (jnp.arange(cfg.steps), rngs))
    return state, seq


dot_str = """
digraph G {
    A [alpha move knight scout]
    B [bravo move queen scout]
    C [alpha attack king scout]

    A -> C
    B -> C
}
"""

plan = tree.map(lambda *x: jnp.stack(x), *tuple(map(partial(b2s.lxm.str_to_plan, dot_str, scene), (-1, 1))))
obs, state = vmap(env.reset, in_axes=(0, None))(random.split(key, num_sim), scene)
rngs = random.split(rng, (num_sim, cfg.steps))
state, seq = vmap(traj_fn)(obs, state, rngs)
b2s.utils.svgs_fn(scene, seq)  # tree.map(lambda x: x[0], seq))
