# %% Imports
import networkx as nx
import jax.numpy as jnp
from jax import tree, lax, debug
from jaxtyping import Array
import equinox as eqx
import parabellum as pb
import pydot
from aic2sim.utils import chess_to_int, alpha_to_int, bt_to_int, nato_to_int
from aic2sim.types import Plan
import ollama


# evaluate plan
def obs_fn(scene: pb.types.Scene, state: pb.types.State, marks):  # obs function for lxm (NOT units)
    return f"raster_map: {scene.terrain.building}\nunit_coord: {state.coord}\nunit_teams: {scene.unit_teams}"


@eqx.filter_jit
def plan_fn(rng: Array, bts, plan: Plan, state: pb.types.State, scene: pb.types.Scene):  # TODO: Focus
    def move(step):  # all units in focus within 10 meters of target position (fix quadratic)
        return ((jnp.linalg.norm(state.coord - step.coord) * step.units) < 10).all()

    def kill(step):  # all enemies dead within 10 meters of target  (this is quadratric and should be made smart)
        return ((jnp.linalg.norm(state.coord - step.coord) * ~step.units * (state.hp == 0)) < 10).any()

    def aux(plan: Plan):
        cond = lax.map(lambda step: lax.cond(step.move, move, kill, step), plan)
        # debug.breakpoint()
        # process cond better than argmin by scanning, through children.
        # idx = scan and mask through children (use instead of cond.argmin())
        return plan.btidx[cond.argmin()] * plan.units[cond.argmin()]

    idxs = lax.map(aux, plan).sum(0)  # mapping across teams (2 for now, but supports any number)
    return tree.map(lambda x: jnp.take(x, idxs, axis=0), bts)  # behavior


# Parse plan
def str_to_plan(dot_str, scene, team):  # plan for one team
    G = nx.DiGraph(nx.nx_pydot.from_pydot(pydot.graph_from_dot_data(dot_str)[0]))  # type: ignore
    nodes = {node: tuple(data.keys()) for node, data in G.nodes(data=True)}
    return tree.map(lambda *x: jnp.stack(x), *tuple(map(lambda x: node_to_step(scene, team, G, *x), nodes.items())))  # type: ignore


def node_to_step(scene: pb.types.Scene, team, G, node, desc):
    move = jnp.array(desc[1] == "move")
    units = (
        jnp.tile(jnp.eye(3) == 1, len(scene.unit_teams))[:, : len(scene.unit_teams)][nato_to_int[desc[0]]]
        * scene.unit_teams
        == team
    ) * 1
    coord = jnp.array(chess_to_int[desc[2]])
    btidx = jnp.array(bt_to_int[desc[3]])
    idxs = jnp.int8([alpha_to_int[e[0]] for e in G.edges() if e[1] == node])
    parent = jnp.zeros(len(G)).at[idxs].set(1)
    return Plan(units=units, move=move, coord=coord, btidx=btidx, parent=jnp.int32(parent))
