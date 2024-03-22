# c2sim

https://github.com/syrkis/c2sim/assets/25223901/11e8cf5e-230c-4245-9e68-d8825dd53965

This repository contains the source code for the C2SIM project. The C2SIM project is a simulation of a command and control system for a military organization. The project is written in `python3.11.7` with a heavy emphasis on functional programming and the JAX ecosystem. `notebook.ipynb` contains a Jupyter notebook that demonstrates the basic functionality of the project, while `main.py` contains the main entry point for the simulation. `src/` contains the source code for the project, including `smax.py` for functionts related the SMAX environment, `bt.py` for behavior tree (BT) related functions, `atomics.py` for atomic finctions for use in BTs, and `utils.py` for utility functions. The project also includes exstentive plotting functions located in `src/plot.py` for visualising SMAX trajectories and BTs.

## Installation

To install the project, clone the repository and install the dependencies using `pip` (probably a good idea to use a virtual environment):

```bash
git clone
cd c2sim
pip install -r requirements.txt
```

## Usage

You almost certainly want to start by running the Jupyter notebook:

```bash
jupyter lab
```

This will open a Jupyter notebook in your browser. Open `notebook.ipynb` and run the cells to see the basic functionality of the project. If you want to run simulations from the command line, you can do so with:

```bash
python main.py
```
