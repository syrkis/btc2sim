# %% dsl.py
#   btc2sim dsl stuff
# by: Noah Syrkis

# Imports
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
from btc2sim.types import Behavior
import jax.numpy as jnp
from itertools import product
from functools import reduce


# %% Globals
with open("grammar.peg", "r") as f:
    grammar = Grammar(f.read())
    pieces = [m.literal for m in grammar["piece"].members]  # type: ignore
    directions = [m.literal for m in grammar["direction"].members]  # type: ignore
    move_fns = [("stand",)] + [("move", *comb) for comb in list(product(directions, pieces))]
    cond_fns = [("is_alive",)]
    i2v = sorted(move_fns + cond_fns)
    v2i = {var: i for i, var in enumerate(i2v)}


def idxs_fn(node):
    if node.get("type") in ["condition", "action"]:
        node = node[node.get("type")]
        return [v2i[(node,)] if type(node) is str else v2i[tuple(node.values())]]
    else:
        return reduce(lambda acc, child: acc + idxs_fn(child), node["children"], [])


def parent_fn(node):
    parents = []
    for child in node["children"]:
        if child["type"] not in ["sequence", "fallback"]:
            parents += [int(node["type"] == "sequence")]
        else:
            parents += parent_fn(child)
    return parents


def skips_fn(node):
    skips = []
    for i, child in enumerate(node["children"]):
        if child["type"] not in ["sequence", "fallback"]:
            skips += [len(node["children"]) - (i + 1)]
        else:
            skips += skips_fn(child)
    return skips


def prevs_fn(node):
    prevs = []
    for child in node["children"]:
        if child["type"] not in ["sequence", "fallback"]:
            prevs += [int(node["type"] == "sequence")]
        else:
            prevs += prevs_fn(child)
    return prevs


# %% Where the magic happens
def txt2bts(txt) -> Behavior:
    node = BehaviorTreeVisitor().visit(grammar.parse(txt))
    fns = [idxs_fn, parent_fn, skips_fn, prevs_fn]
    idxs, parent, skips, prevs = map(jnp.array, [fn(node) for fn in fns])
    print(idxs, parent, skips, prevs, sep="\n")
    exit()
    return Behavior(idxs=idxs, parent=parent, skips=skips, prevs=prevs)


# %% Visitor
class BehaviorTreeVisitor(NodeVisitor):
    def visit_tree(self, node, visited_children):
        """Process the full tree with all its nodes."""
        first_node, rest = visited_children

        if rest:
            # Start with the first node
            nodes = [first_node]

            # For each item in rest
            for item in rest:
                # The separator is at index 0 (a string)
                # The node is at index 1 (a dictionary)
                if isinstance(item, list) and len(item) > 1:
                    nodes.append(item[1])
                elif item is None:
                    continue
                else:
                    nodes.append(item)

            return nodes
        return first_node

    def visit_node(self, node, visited_children):
        """Process a node, which can be fallback, sequence, action or condition."""
        # Just return the first (and only) child
        return visited_children[0]

    def visit_fallback(self, node, visited_children):
        """Process a fallback node (F)."""
        # The structure is ["F", ws, "(", ws, tree, ws, ")", ws]
        return {"type": "fallback", "children": visited_children[4]}

    def visit_sequence(self, node, visited_children):
        """Process a sequence node (S)."""
        # The structure is ["S", ws, "(", ws, tree, ws, ")", ws]
        return {"type": "sequence", "children": visited_children[4]}

    def visit_action(self, node, visited_children):
        """Process an action node (A)."""
        # The structure is ["A", ws, move_or_stand, ws]
        action_data = visited_children[2]
        return {"type": "action", "action": action_data}

    def visit_condition(self, node, visited_children):
        """Process a condition node (C)."""
        # The structure is ["C", ws, condition, ws]
        return {"type": "condition", "condition": visited_children[2]}

    def visit_move(self, node, visited_children):
        """Process a move action."""
        # The structure is ["move", ws, direction, ws, piece]
        direction = visited_children[2]
        piece = visited_children[4]
        return {"name": "move", "direction": direction, "piece": piece}

    def visit_stand(self, node, visited_children):
        """Process a stand action."""
        return {"name": "stand"}

    def visit_to_from(self, node, visited_children):
        """Process the direction (to/from)."""
        return node.text

    def visit_piece(self, node, visited_children):
        """Process the chess piece."""
        return node.text

    def visit_cond(self, node, visited_children):
        """Process a condition."""
        return node.text

    # Handle whitespace and separators (usually just returning them or ignoring them)
    def visit_sep(self, node, visited_children):
        """Process a separator."""
        return None

    def visit_ws(self, node, visited_children):
        """Process whitespace."""
        return node.text

    # Generic visit method for any other nodes
    def generic_visit(self, node, visited_children):
        """The generic visit method."""
        if visited_children and len(visited_children) == 1:
            return visited_children[0]
        return visited_children or node.text
