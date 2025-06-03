# %% api.py
#   This file contains the FastAPI server that serves the Parabellum environment.
# by: Noah Syrkis

# Imports
import uuid
from dataclasses import asdict, dataclass
from functools import partial
from typing import List

import btc2sim as b2s
import cv2
import jax.numpy as jnp
import numpy as np
import parabellum as pb
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from jax import random, tree, vmap
from jaxtyping import Array
from omegaconf import DictConfig


# %% Types
@dataclass
class Step:
    rng: Array
    obs: pb.types.Obs
    state: pb.types.State
    action: pb.types.Action | None


@dataclass
class Game:
    rng: Array
    env: pb.env.Env
    scene: pb.env.Scene
    step_fn: pb.env.step_fn
    gps: b2s.types.Compass
    step_seq: List[Step]


# Configure CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

bt_strs = """
F ( S ( C in_range enemy |> A shoot random ) |> A move target )
"""

# %% Globals
games = {}
sleep_time = 0.1
n_steps = 100

# Config
loc = dict(place="Palazzo della Civiltà Italiana, Rome, Italy", size=64)
red = dict(infantry=2, armor=0, airplane=0)
blue = dict(infantry=2, armor=0, airplane=0)
cfg = DictConfig(dict(steps=100, knn=4, blue=blue, red=red) | loc)

env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)
rng, key = random.split(random.PRNGKey(0))
bts = b2s.dsl.bts_fn(bt_strs)
action_fn = vmap(b2s.act.action_fn, in_axes=(0, 0, 0, None, None, None, 0))
targets = jnp.int32(jnp.arange(6).repeat(env.num_units // 6)).flatten()


def step_fn(rng: Array, obs: pb.types.Obs, state: pb.types.State, plan: b2s.types.Plan, gps: b2s.types.Compass):
    rngs = random.split(rng, env.num_units)
    behavior = b2s.lxm.plan_fn(rng, bts, plan, state, scene)  # perhaps only update plan every m steps
    action = action_fn(rngs, obs, behavior, env, scene, gps, targets)
    obs, state = env.step(rng, scene, state, action)
    return (obs, state), (state, action)


# %% End points
@app.get("/init/{place}")
def init(place: str):  # should inlcude settings from frontend
    game_id = str(uuid.uuid4())
    step = partial(step_fn, env, scene)
    rng = random.PRNGKey(0)
    gps = tree.map(jnp.zeros_like, b2s.gps.gps_fn(scene, jnp.int32(jnp.zeros((6, 2)))))
    games[game_id] = Game(rng, env, scene, step, gps, [])  # <- state_seq list
    terrain = cv2.resize(np.array(scene.terrain.building), dsize=(100, 100)).tolist()
    teams = scene.unit_teams.tolist()
    marks = {k: v for k, v in zip(b2s.utils.int_to_chess, gps.marks.tolist())}
    return {"game_id": game_id, "terrain": terrain, "size": cfg.size, "teams": teams, "marks": marks}


@app.get("/reset/{game_id}")
def reset(game_id: str):
    rng, key = random.split(games[game_id].rng[-1])
    obs, state = games[game_id].env.reset(rng=key, scene=games[game_id].scene)
    games[game_id].step_seq.append(Step(rng, obs, state, None))
    games[game_id].rng.append(rng)
    return {"state": asdict(tree.map(lambda x: x.tolist(), state)) | {"step": len(games[game_id].step_seq)}}


@app.get("/step/{game_id}")
def step(game_id: str):
    rng, key = random.split(games[game_id].step_seq[-1].rng)
    obs, state = games[game_id].step_seq[-1].obs, games[game_id].step_seq[-1].state
    gps = games[game_id].gps
    action, obs, state = games[game_id].step_fn(obs, state, gps, targets, key)
    games[game_id].step_seq.append(Step(rng, obs, state, action))
    return {"state": asdict(tree.map(lambda x: x.tolist(), state)) | {"step": len(games[game_id].step_seq)}}


@app.post("/close/{game_id}")
async def close(game_id: str):
    del games[game_id]


@app.post("/marks/{game_id}")
async def marks(game_id: str, marks: list = Body(...)):
    gps = b2s.gps.gps_fn(scene, jnp.int32(jnp.array(marks))[:, ::-1])
    games[game_id] = games[game_id]._replace(gps=gps)
