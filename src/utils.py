# utils.py
#    c2sim utility functions
# by: Noah Syrkis

# imports
import os
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import yaml


with open(f"data/{f_name}.yaml", "r") as f:
    tree = yaml.load(f, Loader=yaml.FullLoader)


# this function was used to split simulation trajectories up into seperate worlds (deparellelising)
# It was needed when i was trying to get the JAXMarl viz tool to work, but now i am buildings my own.
""" def split_worlds(seqs):
    worlds = {i :[] for i in range(config['n_envs'])}
    for j in tqdm(range(len(seqs))):
        key, state, action = seqs[j]
        for i in range(config['n_envs']):
            world_key    = key[j, :]
            world_action = {k: v[j] for k, v in action.items()}
            world_state = SMAXState(
                unit_positions=state.unit_positions[i, :],
                unit_types=state.unit_types[i, :],
                unit_teams=state.unit_teams[i, :],
                unit_health=state.unit_health[i, :],
                unit_weapon_cooldowns=state.unit_weapon_cooldowns[i, :],
                prev_actions=state.prev_actions[i, :],
                time=state.time[i],
                terminal=state.terminal[i],
                unit_alive=state.unit_alive[i, :]
            )
            worlds[i].append((world_key, world_state, world_action))
    return worlds """

# worlds = split_worlds(seqs)
