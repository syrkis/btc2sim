# %% dsl.py
#   btc2sim dsl stuff
# by: Noah Syrkis

# Imports
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
from btc2sim.types import BehaviorArray, Parent
import jax.numpy as jnp

grammar = Grammar(r"""
    tree        = node (sep node)*
    node        = fallback / sequence / action / condition

    fallback    = "F" ws "(" ws tree ws ")" ws
    sequence    = "S" ws "(" ws tree ws ")" ws
    action      = "A" ws (move / stand) ws
    condition   = "C" ws cond ws

    move        = "move" ws (to_from) ws (piece)
    to_from     = "to" / "from"
    piece      = "king" / "queen" / "rook" / "bishop" / "knight" / "pawn"
    stand       = "stand"

    cond        = "is_alive"

    sep         = ws "|>" ws
    ws          = ~r"\s*"
""")


class BehaviorTreeVisitor(NodeVisitor):
    def visit_tree(self, node, visited_children):
        """Process the full tree with all its nodes."""
        first_node, rest = visited_children
        if rest:
            # There are multiple nodes joined by separators
            nodes = [first_node]
            for sep_and_node in rest:
                # Each element is [separator, node]
                nodes.append(sep_and_node[1])
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
        # The structure is ["move", ws, to_from, ws, piece]
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
        return node.text

    def visit_ws(self, node, visited_children):
        """Process whitespace."""
        return node.text

    # Generic visit method for any other nodes
    def generic_visit(self, node, visited_children):
        """The generic visit method."""
        if visited_children and len(visited_children) == 1:
            return visited_children[0]
        return visited_children or node.text


def txt2bts(txt, size=7):
    visitor = BehaviorTreeVisitor()
    tree = grammar.parse(txt)
    result = visitor.visit(tree)
    for r in result:
        print(r)
    exit()
    parents = jnp.ones(size, dtype=jnp.int32) * Parent.NONE
    predecessors = jnp.ones(size, dtype=jnp.int32) * Parent.NONE
    atomics_id = jnp.ones(size, dtype=jnp.int32) * -1
    passings = jnp.zeros(size, dtype=jnp.int32)
    for i, (predecessor, parent, passing, atomic_id) in enumerate(A):
        predecessors = predecessors.at[i].set(predecessor)
        parents = parents.at[i].set(parent)
        passings = passings.at[i].set(0 if passing is None else passing)
        atomics_id = atomics_id.at[i].set(atomic_id)
    return BehaviorArray(pred=predecessors, parent=parents, passing=passings, atomics_id=atomics_id)


# Example usage:
# source = "S ( C is_alive |> A move from king |> A  stand )"
# tree = grammar.parse(source)
# visitor = BehaviorTreeVisitor()
# result = visitor.visit(tree)
# print(result)

# exit()
