---
title: C2SIM — Architecture
author: Noah Syrkis
type: slides
---

# Purpose

- A starcraft playing LLM commander.
- Behavior tree based.
- Human in the loop.

# Current state

- Trying to get SMAX [@rutherford2023] to work.
- Trying to get the behavior tree (BT) to work.
    - LLM should output (or select) BT.
    - BT should be used for unit control.

# SMAX

- Simplified Starcraft II environment.
- Focus on unitcontrol (no buildings, resources, etc).
- Focus should be to get the BT to work.

# Behavior Tree