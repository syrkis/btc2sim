---
title: C2SIM
author: Noah Syrkis
affiliation: IT University of Copenhagen
type: slides
---

# Overview

The project^[https://github.com/syrkis/c2sim/] uses JAX^[https://github.com/google/jax/] throughout, with JaxMARL's^[https://blog.foersterlab.com/jaxmarl/] [@rutherford2023] StarCraft II-like SMAX as the environment. The agents are modelled using behaviour trees (BT). BTs are defined using a domain specific language (DSL) developped for the purpose. The ollama^[https://ollama.com/] library is used for the language modelling to map game states to human language and BTs, and vice versa.

## Overview (cont.)

- [x] SMAX visual playback (`src/{plot,smax}.py`).
- [x] BT function constructor (`src/{bt,atomics}.py`).
- [x] BT based trajectory (`src/smax.py`). (yet to JIT compile)
    - Must traverse all leafs always (for array programming)^[Has no effect on performance, as we are always as slow as slowest action].
- [x] Domain specific language (`grammar.lark`).
- [ ] Implement the BTBank (`src/bank.py`).
- [ ] Language out (`src/llm.py`).
- [ ] Language in (`src/llm.py`).


# SMAX

- Extensive work on visual playback of trajectory [@fig:smax].
    - [x] Costum SMAX vizualization.
    - [x] Show unit type, team, health, attacks, and reward.
    - [x] Successfully runnning 10K+ parallel environments.

---

![SMAX in parallel](figs/worlds_white.jpg){#fig:smax}

# Behaviour trees

- BT is for now is located in a .yaml file.
- Beginning move to sqlite3 database.
- JAX based tick functions for node and leafs.
- Full traversal happens every tick, using logical operations.
- No JIT compilation yet.

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