---
title: C2SIM
author: Noah Syrkis
affiliation: IT University of Copenhagen
type: slides
---

# Overview

- SMAX
- Extensive work on visual playback of trajectory [@fig:smax].
    - [x] Costum SMAX vizualization.
    - [x] Show unit type, team, health, attacks, and reward.
    - [x] Runnning 10K+ parallel environments.

# Formal grammar

- We've defined a formal grammar (language) for behavior trees.
- The grammar is used to generate JAX-based trees.
- The trees are used to control the AI.

\tiny

    S (C (see enemy_0) :: C (see enemy_1) :: C (see enemy_2))
    F (C (see ally_0 ) :: C (see ally_1) :: C ( see ally_2 ))
    F (S (1 :: 2 :: A (attack any)) :: F (A (move center) :: A (stand)))
    

# Atomics

- Behavior Trees (BTs) are a way to model AI behavior.
- Instead of linear control flow, BTs use a tree structure.
- The leaves of the tree are atomic actions or conditions.
- Atomics are hand written JAX functions.


## DSL grammar

\small
```
tree      : sequence | fallback | decorator | atomic
atomic    : action | condition
nodes     : tree ( :: tree )*
sequence  : S ( nodes )
fallback  : F ( nodes )
decorator : D ( nodes )
action    : A ( STRING+ )
condition : C ( STRING+ )
```

## DSL example

\small
```
F (
  S (
    C ( see enemy_0 ) :: A ( attack enemy_0 )
  ) ::
  F (
    C ( see_enemy ) :: A ( find_enemy )
  ) ::
  A ( attack_enemy )
)
```
    
## Atomics

- Atomics are the leaves (actions/conditions) of the tree.
- They are JAX functions.
- Keep them simple and fast (complex behavior should come from the tree).
    - E.g. `move`, `attack`, `is_enemy`, `is_dead`, `n_in_range`, etc.
    - Maybe map out desired atomic functions.

## BTBank

- BTBank is a library for creating and running BTs.
- It is written in Python.
- sqlite3 is used to store the trees.

# Language model

- The language model is a transformer model.
- I/O architecture.
- The output is a sequence of tokens.