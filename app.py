# app.py
#    c2sim app
# by: Noah Syrkis

# imports
import streamlit as st
from jaxmarl import make

from src.utils import DEFAULT_BT, scenarios
from src.bank import grammar_fn, parse_fn, dict_fn
from src.bt import make_bt

# constants
grammar = "grammar.lark"
page_title = "c2sims | meta gaming platform"

st.set_page_config(
    page_title=page_title,
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    sims_fn(*cols_fn(sidebar_fn()))


def cols_fn(scenario):
    st.title(page_title)
    cols = st.columns(2)
    with cols[0]:
        # llm chat
        st.text_area(
            "Chat",
            height=400,
            value="I am your AI commander assistant. How can I help you?",
        )
    with cols[1]:
        bt = st.text_area("Tree", DEFAULT_BT, height=400)
    st.image(
        "https://syrkis.ams3.digitaloceanspaces.com/noah/rhos/0.jpg",
        caption="SMAX playback of BT",
        use_column_width=True,
    )
    return bt, scenario


def sims_fn(bt, scenario):
    # parse
    grammar = grammar_fn()
    parse = parse_fn(grammar)
    dict = dict_fn(parse, bt)
    # make
    make(dict, scenario)


def sidebar_fn():
    st.sidebar.title("Settings")
    st.sidebar.write("Choose a scenario")
    scenario = st.sidebar.selectbox("", scenarios)
    return scenario


if __name__ == "__main__":
    main()
