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
import yaml

# constants
with open("config.yaml", "r") as f:
    conf = yaml.safe_load(f)
    n_envs = conf["n_envs"]
    n_steps = conf["n_steps"]
    n_agents = conf["n_agents"]
    n_actions = conf["n_actions"]


# functions
def step_fn(bt, rng, old_state_v, obs_v, env):  # take a step in the env
    rng, act_rng, step_rng = random.split(rng, 3)
    act_keys = random.split(act_rng, env.num_agents * n_envs).reshape(-1, n_envs, 2)
    step_keys = random.split(step_rng, n_envs)
    acts = {a: bt(act_keys[i], obs_v[a], a) for i, a in enumerate(env.agents)}  # para
    obs_v, state_v, reward_v, _, _ = vmap(env.step)(step_keys, old_state_v, acts)
    return obs_v, (bt, rng, state_v), (step_keys, old_state_v, acts), reward_v


def traj_fn(bt, rng, env, state_seq=[], reward_seq=[]):  # take n_steps in m env
    rng, reset_rng = random.split(random.PRNGKey(0))  # split rng for reset and step
    reset_keys = random.split(reset_rng, n_envs)  # split reset rng for n_envs

    step = partial(step_fn, env=env)  # partial step function
    obs_v, state_v = vmap(env.reset)(reset_keys)  # initiate envs
    traj_state = (bt, rng, state_v)  # initial state for step_fn

    for _ in tqdm(range(n_steps)):  # take n steps in env and append to lists
        obs_v, traj_state, state_v, reward_v = step(*traj_state, obs_v)  # take step
        state_seq, reward_seq = state_seq + [state_v], reward_seq + [reward_v]
        return

    return state_seq, reward_seq


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
    for _, state, acts in tqdm(state_seq):
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

    state_zip = zip(states[:-1], states[1:])
    for i, ((_, state, _), (_, n_state, _)) in enumerate(state_zip):
        one_hot = attack_seq[i]
        bullets = jnp.concatenate([aux_fn("ally"), aux_fn("enemy")], axis=0)
        bullets_source = state.unit_positions[bullets[:, 0], bullets[:, 1], ...]
        bullets_target = n_state.unit_positions[bullets[:, 0], bullets[:, 2], ...]
        bullets = jnp.concatenate([bullets, bullets_source, bullets_target], axis=1)
        bullet_seq.append(bullets)
    return bullet_seq


def main():
    env = make("SMAX", num_allies=2, num_enemies=5)
    rng = random.PRNGKey(0)
    bt = make_bt(env, "data/bt.yaml")
    state_seq, reward_seq = traj_fn(bt, rng, env)
    bullet_seq = bullet_fn(env, state_seq)
    return state_seq, reward_seq, bullet_seq
