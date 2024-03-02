# smax.py
#   smax code
# by: Noah Syrkis

# imports
import jax
from jax import numpy as jnp, jit, vmap, random
from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario


# functions
def make_vtraj(config):  # returns a function that runs n_envs environments in parallel. Current actions are random.
    env = make(config['env'], **config['env_config'])
    config['n_agents'] = env.num_agents * config['n_envs']

    def init_runner_state(key):
        key, key_reset = random.split(key)
        key_reset      = random.split(key_reset, config['n_envs'])
        obsv, state    = vmap(env.reset)(key_reset)
        return (state, obsv, key)

    def env_step(runner_state, seqs):
        env_state, last_obs, key = runner_state   # random key for sampling actions
        key, key_act             = random.split(key)

        key_act = random.split(key_act, config['n_agents']).reshape((env.num_agents, config['n_envs'], -1))
        # this is the line we wanna inject the action into from.
        actions = {agent: vmap(env.action_space(agent).sample)(key_act[i]) for i, agent in enumerate(env.agents)}

        key, key_step = random.split(key)
        key_step      = random.split(key_step, config['n_envs'])

        obsv, env_state, _, _, infos = vmap(env.step)(key_step, env_state, actions)

        return (env_state, obsv, key), seqs + [env_state] # + [env_state]  # (state, obsv, reward, done, infos)

    def vtraj(key):
        key, key_init      = random.split(key)
        runner_state       = init_runner_state(key_init)
        runner_state, seqs = jax.lax.scan(env_step, runner_state, list(), length=config['max_steps'])
        return runner_state, seqs[0]

    return vtraj