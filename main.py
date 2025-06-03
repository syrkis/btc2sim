# %% main.py
#   btc2sim main file
# by: Noah Syrkis

# Imports
import jax.numpy as jnp
import parabellum as pb
from jax import lax, random, tree, vmap
from jax_tqdm import scan_tqdm
from omegaconf import DictConfig
from functools import partial
import btc2sim as b2s


# %% Config #####################################################
num_sim = 4
loc = dict(place="Palazzo della Civiltà Italiana, Rome, Italy", size=64)
red = dict(infantry=2, armor=0, airplane=0)
blue = dict(infantry=2, armor=0, airplane=0)
cfg = DictConfig(dict(steps=100, knn=4, blue=blue, red=red) | loc)


# %% Behavior trees ( in range should be in reach )
bt_strs = """
F ( S ( C in_range enemy |> A shoot random ) |> A move target )
"""

# A move target

# %% Constants
env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)
bts = b2s.dsl.bts_fn(bt_strs)
action_fn = vmap(b2s.act.action_fn, in_axes=(0, 0, 0, None, None, None, 0))

rng, key = random.split(random.PRNGKey(0))
marks = jnp.int32(random.uniform(rng, (1, 2), minval=0, maxval=cfg.size))
targets = random.randint(rng, (env.num_units,), 0, marks.shape[0])
gps = b2s.gps.gps_fn(scene, marks)  # 6, key)


# %% Functions
def step_fn(carry, input):
    (_, rng), (obs, state) = input, carry
    rngs = random.split(rng, env.num_units)
    behavior = b2s.lxm.plan_fn(rng, bts, plan, state, scene)  # perhaps only update plan every m steps
    action = action_fn(rngs, obs, behavior, env, scene, gps, targets)
    obs, state = env.step(rng, scene, state, action)
    return (obs, state), (state, action)


# maybe vmap here
def traj_fn(obs, state, rngs):
    step = scan_tqdm(cfg.steps)(step_fn)
    return lax.scan(step, (obs, state), (jnp.arange(cfg.steps), rngs))


dot_str = """
digraph G {
    A [alpha move knight scout]
    B [bravo move queen scout]
    C [alpha attack king scout]

    A -> C
    B -> C
}
"""

plan = tree.map(lambda *x: jnp.stack(x), *tuple(map(partial(b2s.lxm.str_to_plan, dot_str, scene), (-1, 1))))  # type: ignore
obs, state = vmap(env.reset, in_axes=(0, None))(random.split(key, num_sim), scene)
rngs = random.split(rng, (num_sim, cfg.steps))
state, (seq, action) = vmap(traj_fn)(obs, state, rngs)
pb.utils.svg_fn(scene, tree.map(lambda x: x[0], seq), tree.map(lambda x: x[0], action), fps=10)
