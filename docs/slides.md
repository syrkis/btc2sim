---
title: Tactical Autonomous Language-Operated Network
author: Noah Syrkis
affiliation: IT University of Copenhagen
type: slides
---

# Overview

::: columns
:::: column
- As much in JAX as possible.
::::
:::: column
![](tmp.jpg)
::::
:::

# SMAX

- Trying to get SMAX [@rutherford2023] to work.
- SMAX is something something something something
- Focus on unitcontrol (no buildings, resources, etc).

# Behavior trees

::: columns

:::: column

- Currently trying to get BT to work.
- LLM should make structured output.
- This output must be BT, following a grammar.

::::

:::: column

- Tools:
    - Overview by @lin2023
    - Grammar maker ^[https://grammar.intrinsiclabs.ai/].
    - Pydantic ^[https://github.com/pydantic/pydantic].

::::
:::

\framebreak

- BT output should follow a grammar.
- Military people like formal systems.
- BT should be formalized and validated.
- BT should be used for unit control and command issuing.

- Current aproach is to represent behaviors trees as

# Atomic functions

- Manually written.
- Should written with genetic programming?

# Mistral

    - LLM should output (or select) BT.
    - BT should be used for unit control.

# C

