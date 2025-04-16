# %% dsl.py
#   btc2sim dsl stuff
# by: Noah Syrkis

# Imports
from parsimonious.nodes import NodeVisitor
from btc2sim.types import Behavior
import jax.numpy as jnp
from jax import tree
from functools import reduce
from parsimonious.grammar import Grammar
from itertools import product


# %% Behavior dataclass
def txt2bts(txt) -> Behavior:
    node = BehaviorTreeVisitor().visit(grammar.parse(txt))
    fns = [idxs_fn, parent_fn, skips_fn, prevs_fn]
    idxs, parent, skips, prevs = map(jnp.array, [fn(node) for fn in fns])
    behavior = Behavior(idxs=idxs, parent=parent, skips=skips)
    return behavior


# %% Tree traversales
def skips_fn(node):
    def aux_fn(n):
        if n.get("type") in ["condition", "action"]:
            return 1
        return sum(aux_fn(child) for child in n["children"])

    if "children" not in node:
        return [0]  # No leaves after a leaf

    result = []
    total_leaves = aux_fn(node)
    for i, child in enumerate(node["children"]):
        child_leaves = aux_fn(child)
        leaves_after = total_leaves - sum(aux_fn(node["children"][j]) for j in range(i + 1))
        result.extend(
            [leaves_after] * child_leaves if child.get("type") in ["condition", "action"] else skips_fn(child)
        )
    return result


def prevs_fn(node):
    """
    Returns a list of length n_leafs. Each entry is 0 (for fallback) or 1 (for sequence).
    The entry indicates the parent type of the immediately preceding leaf.
    For the first leaf in a sequence of siblings, it gets the parent's type.
    For subsequent leaves, they get the type of their parent.
    """

    def traverse(n, parent_type):
        if n.get("type") in ["condition", "action"]:
            return [parent_type]  # Leaf node

        node_type = int(n["type"] == "sequence")  # 0 for fallback, 1 for sequence
        result = []
        for i, child in enumerate(n["children"]):
            # For first child, use parent's type, for others use current node's type
            prev_type = parent_type if i == 0 else node_type
            if child.get("type") in ["condition", "action"]:
                result.append(prev_type)
            else:
                result.extend(traverse(child, prev_type))
        return result

    # Start with 0 as default parent type for the root (could be either 0 or root's type)
    return traverse(node, 0)


def idxs_fn(node):  # CORRECT
    if node.get("type") in ["condition", "action"]:
        node = node[node.get("type")]
        return [t2i[(node,)] if type(node) is str else t2i[tuple(node.values())]]
    else:
        return reduce(lambda acc, child: acc + idxs_fn(child), node["children"], [])


def parent_fn(node):  # CORRECT
    parents = []
    for child in node["children"]:
        if "children" not in child:
            parents += [int(node["type"] == "sequence")]
        else:
            parents += parent_fn(child)
    return parents


# %% Visitor (parsimonious stuff) BELOW HERE THERE BE DRAGONS
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


# %% Grammar stuff
with open("grammar.peg", "r") as f:
    grammar = Grammar(f.read())
    pieces = [m.literal for m in grammar["piece"].members]  # type: ignore
    directions = [m.literal for m in grammar["direction"].members]  # type: ignore
    move_fns = [("stand",)] + [("move", *comb) for comb in list(product(directions, pieces))]
    cond_fns = [("is_alive",)]
    i2v = sorted(move_fns + cond_fns)
    t2i = {var: i for i, var in enumerate(i2v)}
