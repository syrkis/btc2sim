# app.py
#    c2sim app
# by: Noah Syrkis

# imports
import streamlit as st
from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jax import vmap, jit, random
import lark
import darkdetect
import ollama

from src.utils import DEFAULT_BT, scenarios
from src.bank import grammar_fn, parse_fn, dict_fn
from src.bt import make_bt

# constants
grammar = "grammar.lark"
page_title = "Command and control simulations"


# functions
def play_fn(bt, scenario):
    btv = jit(vmap(bt, in_axes=(0, 0, None)), static_argnums=(2,))
    env = make("SMAX", scenario=map_name_to_scenario(scenario))
    rng = random.PRNGKey(0)
    obs, state = env.reset(rng)
    for _ in range(100):
        action = bt(obs)
        obs, state, _ = env.step(action, state)


# content
def main():
    st.set_page_config(page_title=page_title, page_icon="C2", layout="wide")
    st.title(page_title)
    st.sidebar.title("Settings")
    st.sidebar.write("Configure the simulation.")
    bt_str = st.sidebar.text_area("Behavior Tree", DEFAULT_BT.strip()).strip()
    bt_fn = dict_fn(parse_fn(bt_str))
    st.sidebar.write(bt_fn)
    scenario = st.sidebar.selectbox("Scenario", scenarios)
    if st.sidebar.button("Run"):
        play_fn(bt_fn, scenario)


if __name__ == "__main__":
    main()
