# app.py
#    c2sim app
# by: Noah Syrkis

# imports
import streamlit as st
from matplotlib import pyplot as plt

from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jax import jit, vmap, random
import jax
import jax.numpy as jnp


from src.utils import scenarios
from src.bank import load_trees, parse_fn, dict_fn
from src.bt import make_bt

# constants
grammar = "grammar.lark"
page_title = "c2sims | meta gaming platform"
n_envs_sqrt = 3
n_envs = n_envs_sqrt**2

# set page config
conf = dict(initial_sidebar_state="expanded", layout="wide", page_title=page_title)
st.set_page_config(**conf)
plt.style.use("dark_background")


##################
# Main
##################


def main():
    st.title(page_title)
    scenario = sidebar_fn()
    tree_bank = load_trees()

    # columns
    top_cols = st.columns(2)
    low_cols = st.columns(2)

    # inputs
    chat = natural_language_fn(top_cols[0])
    tree = domain_langauge_fn(top_cols[1], tree_bank)

    # simulation
    rng = random.PRNGKey(4)
    seqs = simulate_fn(rng, tree, scenario)

    # outputs
    metrics = metrics_fn(low_cols[0], seqs)
    playbacks = playbacks_fn(low_cols[1], seqs)


##################
# Output Functions
##################


def simulate_fn(rng, bt, scenario):
    # simulate the behavior tree
    rng, key = random.split(rng)
    reset_keys = random.split(key, n_envs)
    env = make("SMAX", scenario=map_name_to_scenario(scenario))
    obs, state = vmap(env.reset)(reset_keys)
    seqs = []
    for i in range(15):  # could replace with a lax.scan for speed
        rng, key = random.split(rng)
        step_keys = random.split(key, n_envs)
        action = {a: bt(state, obs[a], a, env)[1] for a in env.agents}
        obs, state, reward, done, info = vmap(env.step)(step_keys, state, action)
        seqs.append((obs, state, reward, done, info))
    return seqs


def metrics_fn(col, seqs):
    col.write("Returns")
    returns = [seq[2] for seq in seqs]
    fig, axes = plt.subplots(n_envs_sqrt, n_envs_sqrt, figsize=(9, 9))
    for idx, ax in enumerate(axes.flatten()):
        ally_rets = []
        enemy_rets = []
        for ret in returns:
            ally_ret = sum([v[idx] for k, v in ret.items() if k.startswith("ally")])
            enemy_ret = sum([v[idx] for k, v in ret.items() if k.startswith("enemy")])
            ally_rets.append(ally_ret)
            enemy_rets.append(enemy_ret)

        ax.plot(
            range(len(returns)),
            jnp.cumsum(jnp.array(ally_rets)),
            color="blue",
            label="Ally",
        )
        ax.plot(
            range(len(returns)),
            jnp.cumsum(jnp.array(enemy_rets)),
            color="red",
            label="Enemy",
        )
        # hide axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        # set title
        ax.set_title(f"Env {idx}")
        if idx == 0:
            ax.legend()
    plt.tight_layout()
    col.pyplot(fig)
    return returns


def playbacks_fn(col, seqs):
    col.write("Playbacks")
    pass


##################
# Input Functions
##################


def natural_language_fn(col):
    col.text_area(
        "Natural Language",
        height=400,
        value="I am your AI commander assistant. How can I help you?",
    )


def domain_langauge_fn(col, tree_bank):
    default_tree = tree_bank[0]["tree_str"]
    tree_str = col.text_area("Domain Language", height=400, value=default_tree)
    try:
        tree_dict = parse_fn(tree_str)
    except:
        tree_dict = parse_fn(default_tree)
        col.warning("Invalid tree string. Using default tree.")
    tree = vmap(make_bt(dict_fn(tree_dict)), in_axes=(0, 0, None, None))
    # tree = vmap(tree_bank[0]["tree"], in_axes=(0, 0, None, None))
    # pretty print string of tree
    return jit(tree, static_argnums=(2, 3))


def sidebar_fn():
    st.sidebar.title("Settings")
    st.sidebar.write("Choose a scenario")
    scenario = st.sidebar.selectbox("", scenarios[::-1])
    return scenario


##################
# Run
##################

if __name__ == "__main__":
    main()
