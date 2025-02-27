# main.py
import parabellum as pb
from jax import random, lax


# Functions
def step(state, rng):
    moving = random.normal(rng, (env.cfg.num_agents, 2))
    action = pb.env.Action(health=None, moving=moving)
    obs, state = env.step(rng, state, action)
    return state, state


# Environment
env = pb.env.Env(cfg=(cfg := pb.env.Conf()))
rng, key = random.split(random.PRNGKey(0))
obs, state = env.reset(key)
rngs = random.split(rng, 100)
state, seq = lax.scan(step, state, rngs)
