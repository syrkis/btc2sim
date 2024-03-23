---
title: C2SIM
author: Noah Syrkis
affiliation: IT University of Copenhagen
type: slides
---

# Overview

The project^[https://github.com/syrkis/c2sim/] uses JAX^[https://github.com/google/jax/] throughout, with JaxMARL's^[https://blog.foersterlab.com/jaxmarl/] SMAX as the main environment. The agents are modelled using behaviour trees (BT) stored in a sqlite3 database (we call it BTBank). The ollama^[https://ollama.com/] library is used for the language modelling to map game states to human language and BTs, and vice versa.

## Overview (cont.)

- [x] SMAX visual playback (`src/{plot,smax}.py`).
- [x] BT function constructor (`src/{bt,atomics}.py`).
- [ ] BT based trajectory (`src/smax.py`). (almost done)
    - Issues with returns happening at different tree depths.
    - solution: go through entire tree every time. (keep trees small)
        - this is a MUST for JAX array vmap (could be viewed as involuntary regularization).
- [ ] Implement the BTBank (`src/bank.py`).
- [ ] Language out (`src/llm.py`).
- [ ] Language in (`src/llm.py`).
- [ ] Smart way to generate atomics (gentic programming)?

# SMAX

- Extensive work on visual playback of trajectory [@fig:smax].
    - [x] Costum SMAX [@rutherford2023] vizualization.
    - [x] Show unit type, team, health, attacks, and reward.
    - [x] Successfully runnning 10K+ parallel environments.

---

![SMAX in parallel](figs/worlds_white.jpg){#fig:smax}

## SMAX (cont.)

    key = random.PRNGKey(0).split(num_envs)
    env = make('SMAX', num_allies=n, num_enemies=m)
    obs, state = vmap(env.reset)(key)
    for _ in range(num_steps):
        act = vmap(act_fn)(rng, env, obs, state)
        obs, state, (_) = vmap(env.step)(act, state)

# Behaviour trees

- Behaviour trees (BT) are a way to model the behaviour of agents.
- They are used in games and robotics.

## Atomics

- Atomics are the leaves of the tree.
- They are the actions that the agent can take.

## BTBank

- BTBank is a library for creating and running BTs.
- It is written in Python.
- sqlite3 is used to store the trees.

# Language model

- The language model is a transformer model.
- I/O architecture.
- The output is a sequence of tokens.

# Todo list (MVP)
