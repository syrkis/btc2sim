# smax.py
#   smax code
# by: Noah Syrkis

# imports
import jax
from functional import partial
from tqdm import tqdm
from jax import numpy as jnp, jit, vmap, random
from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario


# functions
def dist_fn(env, pos):  # computing the distances between all ally and enemy agents
    delta = pos[None, :, :] - pos[:, None, :]
    dist = jnp.sqrt((delta**2).sum(axis=2))
    dist = dist[: env.num_allies, env.num_allies :]
    return {"ally": dist, "enemy": dist.T}


def range_fn(env, dists, ranges):  # computing what targets are in range
    ally_range = dists["ally"] < ranges[: env.num_allies][:, None]
    enemy_range = dists["enemy"] < ranges[env.num_allies :][:, None]
    return {"ally": ally_range, "enemy": enemy_range}


def target_fn(acts, in_range, team):  # computing the one hot valid targets
    t_acts = jnp.stack([v for k, v in acts.items() if k.startswith(team)]).T
    t_targets = jnp.where(t_acts - 5 < 0, -1, t_acts - 5)  # first 5 are move actions
    t_attacks = jnp.eye(in_range[team].shape[2] + 1)[t_targets][:, :, :-1]
    return t_attacks * in_range[team]  # one hot valid targets


def attack_fn(env, state_seq, attacks=[]):  # one hot attack list
    for _, state, acts in state_seq:
        dists = vmap(partial(dist_fn, env))(state.unit_positions)
        ranges = env.unit_type_attack_ranges[state.unit_types]
        in_range = vmap(partial(range_fn, env))(dists, ranges)
        target = partial(target_fn, acts, in_range)
        attack = {"ally": target("ally"), "enemy": target("enemy")}
        attacks.append(attack)
    return attacks


def bullet_fn(env, states, bullet_seq=[]):
    attack_seq = attack_fn(env, states)

    def aux_fn(team):
        bullets = jnp.stack(jnp.where(one_hot[team] == 1)).T
        bullets = bullets.at[:, 2 if team == "ally" else 1].add(env.num_allies)
        return bullets

    for i, ((_, state, _), (_, n_state, _)) in enumerate(zip(states[:-1], states[1:])):
        one_hot = attack_seq[i]
        bullets = jnp.concatenate(list(map(aux_fn, ["ally", "enemy"])), axis=0)
        bullets_source = state.unit_positions[bullets[:, 0], bullets[:, 1], ...]
        bullets_target = n_state.unit_positions[bullets[:, 0], bullets[:, 2], ...]
        bullets = jnp.concatenate([bullets, bullets_source, bullets_target], axis=1)
        bullet_seq.append(bullets)
    return bullet_seq
