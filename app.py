# app.py
#    c2sim app
# by: Noah Syrkis

# imports
import streamlit as st

from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jax import jit, vmap, random
import jax


from src.utils import scenarios
from src.bank import load_trees

# constants
grammar = "grammar.lark"
page_title = "c2sims | meta gaming platform"

# set page config
conf = dict(initial_sidebar_state="expanded", layout="wide", page_title=page_title)
st.set_page_config(**conf)


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
    rng = random.PRNGKey(0)
    seqs = simulate_fn(rng, tree, scenario)

    # outputs
    playbacks = playbacks_fn(low_cols[0], seqs)
    metrics = metrics_fn(low_cols[1], seqs)


##################
# Output Functions
##################


def simulate_fn(rng, bt, scenario):
    # simulate the behavior tree
    rng, key = random.split(rng)
    reset_keys = random.split(key, 6)
    env = make("SMAX", scenario=map_name_to_scenario(scenario))
    obs, state = vmap(env.reset)(reset_keys)
    seqs = []
    for i in range(10):  # could replace with a lax.scan for speed
        rng, key = random.split(rng)
        step_keys = random.split(key, 6)
        action = {a: bt(state, obs[a], a, env)[1] for a in env.agents}
        obs, state, reward, done, info = vmap(env.step)(step_keys, state, action)
        seqs.append((obs, state, reward, done, info))
    return seqs


def metrics_fn(col, seqs):
    col.write("Metrics")
    returns = [seq[2] for seq in seqs]
    col.write(returns)
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
    tree = col.text_area("Domain Language", height=400, value=tree_bank[0]["tree_str"])
    tree = vmap(tree_bank[0]["tree"], in_axes=(0, 0, None, None))
    # pretty print string of tree
    return tree


def sidebar_fn():
    st.sidebar.title("Settings")
    st.sidebar.write("Choose a scenario")
    scenario = st.sidebar.selectbox("", scenarios)
    return scenario


##################
# Run
##################

if __name__ == "__main__":
    main()
