# app.py
#    c2sim app
# by: Noah Syrkis

# imports
from jax import random
import streamlit as st
from jaxmarl import make
import lark
import darkdetect

from src.utils import DEFAULT_BT

# constants
grammar = "grammar.lark"


def conf_fn():
    st.set_page_config(
        page_title="Command and Control Simulator",
        page_icon="C2",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def sidebar_fn():
    st.sidebar.title("Command and Control Simulator")
    st.sidebar.write("To the right you see the formal grammar for the trord language.")
    st.sidebar.write("To the left you can enter the behavior trees for each team.")
    st.sidebar.write("Below you can play the simulation.")
    st.sidebar.write(f"you are playing against this BT")
    st.sidebar.write(DEFAULT_BT)


def draw_bt(bt):
    pass


def header_fn():
    st.title("Command and Control Simulator")
    cols = st.columns(2)
    cols[0].write("To the right you see the formal grammar for the trord language.")
    cols[0].write("To the left you can enter the behavior trees for each team.")
    cols[0].write("Below you can play the simulation.")
    cols[0].write(f"you are playing against this BT")
    cols[0].write(DEFAULT_BT)

    # describe language
    cols[1].code(
        """
        node: action
            | condition
            | sequence
            | fallback

        nodes: node ( "::" node )*
        arg: /enemy_[0-9]*|friend_[0-9]*|north|east|south|west|center/
        args: ( STRING )*
        atomic: STRING args

        action: "A" "(" atomic ")"
        condition: "C" "(" atomic ")"
        sequence: "S" "(" nodes ")"
        fallback: "F" "(" nodes ")"
        """
    )


def cols_fn():
    st.write("Enter the behavior trees for each team.")
    cols = st.columns(2)
    allies_color = lambda: "White" if darkdetect.isDark() else "Black"
    enemies_color = lambda: "Black" if darkdetect.isDark() else "White"
    tree = cols[0].text_input(f"{allies_color()} team", DEFAULT_BT)
    st.text(tree)
    # parse tree
    tree = lark.Lark(grammar).parse(tree)


def play_fn():
    pass


def main():
    conf_fn()
    sidebar_fn()
    header_fn()
    cols_fn()
    env = make("SMAX")
    rng = random.PRNGKey(0)
    obs, state = env.reset(rng)


if __name__ == "__main__":
    main()
