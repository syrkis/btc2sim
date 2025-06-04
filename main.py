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
import aic2sim as a2s
# from ollama import chat
# import ollama


# evaluate plan
def obs_fn(scene: pb.types.Scene, state: pb.types.State, marks):  # obs function for lxm (NOT units)
    return f"raster_map: {scene.terrain.building}\nunit_coord: {state.coord}\nunit_teams: {scene.unit_teams}"


# %%
# model = "hive"
# print(ollama.show(model))
# stream = chat(
# model=model,
# messages=[{"role": "user", "content": "what are you?"}],
# stream=True,
# )

# for chunk in stream:
# print(chunk["message"]["content"], end="", flush=True)
#
# exit()

# %% Config #####################################################
num_sim = 4
loc = dict(place="Palazzo della Civiltà Italiana, Rome, Italy", size=64)
red = dict(infantry=6, armor=6, airplane=6)
blue = dict(infantry=6, armor=6, airplane=6)
cfg = DictConfig(dict(steps=100, knn=4, blue=blue, red=red) | loc)


# %% Behavior trees ( in range should be in reach )
with open("data/roe.txt", "r") as f:
    roe_str = f.read().strip()


with open("data/dot.txt", "r") as f:
    dot_str = f.read().strip().split("---")[0].strip()


# %% Constants
env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)
bts = a2s.dsl.bts_fn(roe_str)
action_fn = vmap(a2s.act.action_fn, in_axes=(0, 0, 0, None, None, None, 0))

rng, key = random.split(random.PRNGKey(0))
marks = jnp.int32(random.uniform(rng, (1, 2), minval=0, maxval=cfg.size))
targets = random.randint(rng, (env.num_units,), 0, marks.shape[0])
gps = a2s.gps.gps_fn(scene, marks)  # 6, key)


# %% Functions
def step_fn(carry, input):
    (obs, state), (_, rng) = carry, input
    rngs = random.split(rng, env.num_units)
    behavior = a2s.lxm.plan_fn(rng, bts, plan, state, scene)  # perhaps only update plan every m steps
    action = action_fn(rngs, obs, behavior, env, scene, gps, targets)
    obs, state = env.step(rng, scene, state, action)
    return (obs, state), (state, action)


# maybe vmap here
def traj_fn(obs, state, rngs):
    step = scan_tqdm(cfg.steps)(step_fn)
    return lax.scan(step, (obs, state), (jnp.arange(cfg.steps), rngs))


def log_fn(seq):
    # print(tree.map(jnp.shape, seq))
    print(tree.map(lambda x: lax.map(jnp.shape, x), seq))
    # run = aim.Run()
    pass


plan = tree.map(lambda *x: jnp.stack(x), *tuple(map(partial(a2s.lxm.str_to_plan, dot_str, scene), (-1, 1))))  # type: ignore
obs, state = vmap(env.reset, in_axes=(0, None))(random.split(key, num_sim), scene)
print(obs_fn(scene, state, marks))
# rngs = random.split(rng, (num_sim, cfg.steps))
# state, (seq, action) = vmap(traj_fn)(obs, state, rngs)
# log_fn(seq)
# pb.utils.svg_fn(scene, tree.map(lambda x: x[0], seq), tree.map(lambda x: x[0], action), fps=10)
