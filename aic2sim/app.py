# %% api.py
#   This file contains the FastAPI server that serves the Parabellum environment.
# by: Noah Syrkis

# Imports
import uuid
from dataclasses import asdict, replace
from functools import partial

import aim
from ollama import chat
import cv2
import jax.numpy as jnp
import numpy as np
import parabellum as pb
from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from jax import random, tree, vmap
from omegaconf import DictConfig

import aic2sim as a2s

# Configure CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


with open("data/roe.txt", "r") as f:
    roe_str = f.read().strip()  # rules of engagement

with open("data/dot.txt", "r") as f:
    dot_str = f.read().strip().split("---")[0].strip()


# %% Globals
games = {}
fps = 10
n_steps = 100
sleep_time = 1 / fps

# Config
loc = dict(place="Palazzo della Civiltà Italiana, Rome, Italy", size=64)
red = dict(infantry=2, armor=0, airplane=0)
blue = dict(infantry=2, armor=0, airplane=0)
cfg = DictConfig(dict(steps=100, knn=4, blue=blue, red=red) | loc)

env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)
rng, key = random.split(random.PRNGKey(0))
bts = a2s.dsl.bts_fn(roe_str)
action_fn = vmap(a2s.act.action_fn, in_axes=(0, 0, 0, None, None, None, 0))
# targets = jnp.int32(jnp.arange(6).repeat(env.num_units // 6)).flatten()
targets = random.randint(rng, (env.num_units,), 0, 6)
plan = tree.map(lambda *x: jnp.stack(x), *tuple(map(partial(a2s.lxm.str_to_plan, dot_str, scene), (-1, 1))))  # type: ignore


def step_fn(rng, env, scene, obs: pb.types.Obs, state: pb.types.State, plan: a2s.types.Plan, gps, targets):
    rngs = random.split(rng, env.num_units)
    behavior = a2s.lxm.plan_fn(rng, bts, plan, state, scene)  # perhaps only update plan every m steps
    action = action_fn(rngs, obs, behavior, env, scene, gps, targets)
    obs, state = env.step(rng, scene, state, action)
    return (obs, state), (state, action)


# %% End points
@app.get("/init/{place}")
def init(place: str):  # should inlcude settings from frontend
    game_id = str(uuid.uuid4())
    rng = random.PRNGKey(0)
    step = partial(step_fn, rng, env, scene)
    gps = tree.map(jnp.zeros_like, a2s.gps.gps_fn(scene, jnp.int32(jnp.zeros((6, 2)))))
    games[game_id] = a2s.types.Game([rng], env, scene, step, gps, [])  # <- state_seq list
    terrain = cv2.resize(np.array(scene.terrain.building), dsize=(100, 100)).tolist()
    teams = scene.unit_teams.tolist()
    marks = {k: v for k, v in zip(a2s.utils.chess_to_int, gps.marks.tolist())}
    # print(marks)
    return {"game_id": game_id, "terrain": terrain, "size": cfg.size, "teams": teams, "marks": marks}


@app.get("/reset/{game_id}")
def reset(game_id: str):
    rng, key = random.split(games[game_id].rng[-1])
    obs, state = games[game_id].env.reset(rng=key, scene=games[game_id].scene)
    games[game_id].step_seq.append(a2s.types.Step(rng, obs, state, None))
    games[game_id].rng.append(rng)
    return {"state": asdict(tree.map(lambda x: x.tolist(), state)) | {"step": len(games[game_id].step_seq)}}


@app.get("/step/{game_id}")
def step(game_id: str):
    rng, key = random.split(games[game_id].step_seq[-1].rng)
    obs, state = games[game_id].step_seq[-1].obs, games[game_id].step_seq[-1].state
    (obs, state), (state, action) = games[game_id].step_fn(obs, state, plan, games[game_id].gps, targets)
    games[game_id].step_seq.append(a2s.types.Step(rng, obs, state, action))
    return {"state": asdict(tree.map(lambda x: x.tolist(), state)) | {"step": len(games[game_id].step_seq)}}


@app.post("/close/{game_id}")
async def close(game_id: str):
    del games[game_id]


@app.post("/marks/{game_id}")
async def marks(game_id: str, marks: list = Body(...)):
    gps = a2s.gps.gps_fn(scene, jnp.int32(jnp.array(marks))[:, ::-1])
    games[game_id] = replace(games[game_id], gps=gps)


async def chat_stream_generator(content: str):
    """Generator function for streaming chat responses"""
    stream = chat(
        model="hive",
        messages=[{"role": "user", "content": content}],
        stream=True,
    )

    for chunk in stream:
        if chunk["message"]["content"]:
            yield f"data: {chunk['message']['content']}\n\n"


# @app.post("/chat/stream")
# async def chat_stream(request: dict = Body(...)):
#     """HTTP streaming endpoint for chat"""
#     content = request.get("content", "")
#     if not content:
#         return {"error": "No content provided"}

#     return StreamingResponse(
#         chat_stream_generator(content),
#         media_type="text/plain",
#         headers={
#             "Cache-Control": "no-cache",
#             "Connection": "keep-alive",
#         },
#     )


@app.websocket("/chat/ws")
async def chat_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time chat streaming"""
    await websocket.accept()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            print(f"Received: {data}")

            # Stream response back to client
            stream = chat(
                model="hive",
                messages=[{"role": "user", "content": data}],
                stream=True,
            )

            for chunk in stream:
                content = chunk["message"]["content"]
                if content:
                    print(f"Chunk: '{content}'")

                    # Send chunk immediately
                    await websocket.send_text(content)

                    # Force processing of the send
                    await asyncio.gather(asyncio.sleep(0))

            # Send completion signal
            await websocket.send_text("")
            print("Stream complete")

    except WebSocketDisconnect:
        print("Disconnected")
