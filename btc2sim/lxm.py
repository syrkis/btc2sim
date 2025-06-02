# %% Imports
import networkx as nx
import jax.numpy as jnp
from jax import tree, debug
import parabellum as pb
import pydot
from btc2sim.utils import chess_to_int, alpha_to_int, bt_to_int, nato_to_int
from btc2sim.types import Plan


def str_to_plan(dot_str, scene, team):  # plan for one team
    G = nx.DiGraph(nx.nx_pydot.from_pydot(pydot.graph_from_dot_data(dot_str)[0]))  # type: ignore
    nodes = {node: tuple(data.keys()) for node, data in G.nodes(data=True)}
    return tree.map(lambda *x: jnp.stack(x), *tuple(map(lambda x: node_to_step(scene, team, G, *x), nodes.items())))


def node_to_step(scene: pb.types.Scene, team, G, node, desc):
    move = jnp.array(desc[1] == "move")
    units = (jnp.tile(jnp.eye(3) == 1, len(scene.unit_teams))[:, :len(scene.unit_teams)][nato_to_int[desc[0]]] * scene.unit_teams == team) * 1
    # exit()
    coord = jnp.array(chess_to_int[desc[2]])
    btidx = jnp.array(bt_to_int[desc[3]])
    idxs = jnp.int8([alpha_to_int[e[0]] for e in G.edges() if e[1] == node])
    parent = jnp.zeros(len(G)).at[idxs].set(1)
    return Plan(units=units, move=move, coord=coord, btidx=btidx, parent=jnp.int32(parent))
