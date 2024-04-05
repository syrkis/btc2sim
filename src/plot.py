# plot.py
#   plot code
# by: Noah Syrkis

# imports
import imageio
import matplotlib.pyplot as plt
import darkdetect
from matplotlib import rcParams
import numpy as np
from jax import numpy as jnp, vmap
from tqdm import tqdm
from src.smax import bullet_fn

# globals
rcParams["font.family"] = "monospace"
rcParams["font.monospace"] = "Fira Code"
bg = "black" if darkdetect.isDark() else "white"
ink = "white" if bg == "black" else "black"
markers = {0: "o", 1: "s", 2: "D", 3: "^", 4: "<", 5: ">", 6: "+"}

# params
tick_params = {
    "colors": ink,
    "direction": "in",
    "length": 6,
    "width": 1,
    "which": "both",
    "top": True,
    "bottom": True,
    "left": True,
    "right": True,
    "labelleft": False,
    "labelbottom": False,
}


# functions
def plot_fn(env, state_seq, reward_seq, expand=False):
    n_steps = len(state_seq)
    bullet_seq = bullet_fn(env, state_seq) if expand else None
    state_seq = state_seq if not expand else vmap(env.expand_state_seq)(state_seq)
    frames, returns = [], return_fn(reward_seq)
    unit_types = np.unique(np.array(state_seq[0][1].unit_types))
    fills = np.where(np.array(state_seq[0][1].unit_teams) == 1, ink, "None")
    for i, (_, state, _) in tqdm(enumerate(state_seq), total=len(state_seq)):
        fig, axes = plt.subplots(2, 3, figsize=(18.08, 12), facecolor=bg, dpi=50)
        bullets = bullet_seq[i // 8] if expand and i < (len(bullet_seq) * 8) else None
        args = (returns, state, bullets, i, unit_types, fills)
        seq = [(ax, j, *args) for j, ax in enumerate(axes.flatten())]
        for ax, j, *args in seq:
            axis_fn(ax, j, *args)
        frames.append(frame_fn(n_steps, fig, i // 8 if expand else i))
    fname = f"docs/figs/worlds_{bg}{'_laggy' if not expand else ''}.mp4"
    imageio.mimsave(fname, frames, fps=24 if expand else 3)


def return_fn(reward_seq):
    reward = [reward_fn(reward) for reward in reward_seq]
    ally = jnp.stack([v[0] for v in reward]).cumsum(axis=0)
    enemy = jnp.stack([v[1] for v in reward]).cumsum(axis=0)
    return {"ally": ally, "enemy": enemy}


def reward_fn(reward):
    ally_rewards = jnp.stack([v for k, v in reward.items() if k.startswith("ally")])
    enemy_rewards = jnp.stack([v for k, v in reward.items() if k.startswith("enemy")])
    return ally_rewards.sum(axis=0), enemy_rewards.sum(axis=0)


def frame_fn(n_steps, fig, idx):
    title = f"step : {str(idx).zfill(len(str(n_steps - 1)))} | model : random"
    fig.text(0.01, 0.5, title, va="center", rotation="vertical", fontsize=20, color=ink)
    sublot_params = {"hspace": 0.3, "wspace": 0.3, "left": 0.05, "right": 0.95}
    plt.subplots_adjust(**sublot_params)
    fig.canvas.draw()
    fig.tight_layout()
    width, height = fig.get_size_inches() * fig.get_dpi()
    shape = (int(height), int(width), 4)
    frame = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(shape)[..., :3]
    if idx == n_steps - 1:
        plt.savefig(f"docs/figs/worlds_{bg}.jpg", dpi=200)
    plt.close()  # close fig
    return frame


def axis_fn(ax, j, returns, state, bullets, i, unit_types, fills):
    aux_ax_fn(ax, bullets, returns, i, j)
    for unit_type in unit_types:
        idx = state.unit_types[j, :] == unit_type
        x = state.unit_positions[j, idx, 0]
        y = state.unit_positions[j, idx, 1]
        c = fills[j, idx]
        s = state.unit_health[j, idx] ** 1.5 * 0.1
        ax.scatter(x, y, s=s, c=c, edgecolor=ink, marker=markers[unit_type])


def aux_ax_fn(ax, bullets, returns, i, j):
    if bullets is not None:
        idx = bullets[:, 0] == j
        alpha = i % 8 / 8
        pos = (1 - alpha) * bullets[idx, 3:5] + alpha * bullets[idx, 5:]
        ax.scatter(pos[:, 0], pos[:, 1], s=10, c=ink, marker=",")
    ally_return = returns["ally"][i, j]
    enemy_return = returns["enemy"][i, j]
    ax.set_xlabel("{:.3f} | {:.3f}".format(ally_return, enemy_return), color=ink)
    ax.set_title(f"simulation {j+1}", color=ink)
    ax.set_facecolor(bg)
    ticks = np.arange(2, 31, 4)  # Assuming your grid goes from 0 to 32
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(**tick_params)
    ax.spines["top"].set_color(ink)
    ax.spines["bottom"].set_color(ink)
    ax.spines["left"].set_color(ink)
    ax.spines["right"].set_color(ink)
    ax.set_aspect("equal")
    ax.set_xlim(-2, 34)
    ax.set_ylim(-2, 34)
