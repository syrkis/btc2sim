# %%
import numpy as np
import jax.numpy as jnp
from lark import Transformer
from btc2sim.dsl import all_vars, grammar
from btc2sim.types import Parent, Behavior


# %%
def compute_right_siblings(n_nodes):
    a = np.array(n_nodes)
    a = -a
    a[0] = np.sum(-a) + a[0]
    return np.cumsum(a)


# %%
class BehaviorTree(Transformer):
    def __init__(self, all_vars):
        self.all_vars = all_vars

    def node(self, args):
        return args[0]

    def nodes(self, args):
        return args

    def condition(self, args):  # returns four things
        return [(Parent.NONE, Parent.NONE, None, self.all_vars.index(" ".join(args[0])))]

    def action(self, args):  # returns four things
        return [(Parent.NONE, Parent.NONE, None, self.all_vars.index(" ".join(args[0])))]

    def sequence(self, args):  # returns four things
        array = []
        n_nodes = [len(child_array) for child_array in args[0]]
        n_right_siblings = compute_right_siblings(n_nodes)
        for child_id, child_array in enumerate(args[0]):
            array += [
                (
                    Parent.SEQUENCE if i == 0 else pred,
                    Parent.SEQUENCE if parent == Parent.NONE else parent,
                    n_right_siblings[child_id] if passing is None else passing,
                    a,
                )
                for i, (pred, parent, passing, a) in enumerate(child_array)
            ]
        return array

    def fallback(self, args):  # returns four things
        array = []
        n_nodes = [len(child_array) for child_array in args[0]]
        n_right_siblings = compute_right_siblings(n_nodes)
        for child_id, child_array in enumerate(args[0]):
            array += [
                (
                    Parent.FALLBACK if i == 0 else pred,
                    Parent.FALLBACK if parent == Parent.NONE else parent,
                    n_right_siblings[child_id] if passing is None else passing,
                    a,
                )
                for i, (pred, parent, passing, a) in enumerate(child_array)
            ]
        return array

    def atomic(self, args):
        return args[0]

    def move(self, args):
        return ["move"] + args

    def attack(self, args):
        return ["attack"] + args

    def stand(self, args):
        return ["stand"]

    def follow_map(self, args):
        return ["follow_map"] + args

    def heal(self, args):
        return ["heal"] + args

    def debug(self, args):
        return ["debug"] + args

    def in_sight(self, args):
        return ["in_sight"] + args

    def in_reach(self, args):
        return ["in_reach"] + args

    def is_type(self, args):
        return ["is_type"] + args

    def is_dying(self, args):
        return ["is_dying"] + args

    def is_in_forest(self, args):
        return ["is_hidden"]

    def sense(self, args):
        return str(args[0])

    def direction(self, args):
        return str(args[0])

    def foe(self, args):
        return str(args[0])

    def friend(self, args):
        return str(args[0])

    def piece(self, args):
        return str(args[0])

    def qualifier(self, args):
        return str(args[0])

    def unit(self, args):
        return str(args[0])

    def any(self, args):
        return str(args[0])

    def margin(self, args):
        return str(args[0])

    def source(self, args):
        return str(args[0])

    def steps(self, args):
        return str(args[0])

    def self(self, args):
        return str(args[0])

    def threshold(self, args):
        return str(args[0])


def txt2bts(txt, size=10):
    expr = grammar.parse(txt)
    trans = BehaviorTree(all_vars)
    A = trans.transform(expr)  # [(parent, atomic id)]
    parents = jnp.ones(size, dtype=jnp.int32) * Parent.NONE
    predecessors = jnp.ones(size, dtype=jnp.int32) * Parent.NONE
    atomics_id = jnp.ones(size, dtype=jnp.int32) * -1
    passings = jnp.zeros(size, dtype=jnp.int32)
    assert len(A) <= size, f"The expr has {len(A)} leaves which is larger than the defined size {size}."
    for i, (predecessor, parent, passing, atomic_id) in enumerate(A):
        predecessors = predecessors.at[i].set(predecessor)
        parents = parents.at[i].set(parent)
        passings = passings.at[i].set(0 if passing is None else passing)
        atomics_id = atomics_id.at[i].set(atomic_id)
    return Behavior(
        predecessors=predecessors,
        parents=parents,
        passings=passings,
        atomics_id=atomics_id,
    )
