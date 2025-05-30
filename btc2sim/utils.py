# imports
import esch
import jax.numpy as jnp
import numpy as np
from einops import rearrange, repeat
from jax import tree
from PIL import Image
from tqdm import tqdm

# %% Dicts
nato_to_int = dict(alpha=0, bravo=1, charlie=2, delta=3, echo=4, foxtrot=5)
int_to_nato = {v: k for k, v in nato_to_int.items()}

chess_to_int = dict(pawn=0, rook=1, knight=2, bishop=3, queen=4, king=5)
int_to_chess = {v: k for k, v in chess_to_int.items()}

bt_to_int = dict(scout=0, patrol=1, assault=2, defend=3, flank=4, retreat=5, ambush=6, support=7, recon=8, siege=9)
int_to_bt = {v: k for k, v in bt_to_int.items()}

alpha_to_int = dict(A=0, B=1, C=2, D=3, E=4, F=5)
int_to_alpha = {v: k for k, v in alpha_to_int.items()}


# %% Functions
def scene_fn(arr):
    arr = jnp.zeros_like(arr)
    start_idx = arr.shape[0] // 6
    end_idx = arr.shape[0] // 6 * 2
    arr = arr.at[start_idx + 30 : end_idx + 30, start_idx:end_idx].set(1)
    arr = arr.at[start_idx + 30 + 5 : end_idx + 30 - 5, start_idx + 5 : end_idx].set(0)
    return arr


def gif_fn(scene, seq, scale=4):  # animate positions TODO: remove dead units
    pos = seq.coords.astype(int)
    cord = jnp.concat((jnp.arange(pos.shape[0]).repeat(pos.shape[1])[..., None], pos.reshape(-1, 2)), axis=1).T
    idxs = cord[:, seq.health.flatten().astype(bool) > 0]
    mask = scene.terrain.building  # .at[*jnp.int32(gps.marks.T)].set(1)
    imgs = 1 - np.array(repeat(mask, "... -> a ...", a=len(pos)).at[*idxs].set(1))
    imgs = [Image.fromarray(img).resize(np.array(img.shape[:2]) * scale, Image.NEAREST) for img in imgs * 255]  # type: ignore
    imgs[0].save("/Users/nobr/desk/s3/btc2sim/sim.gif", save_all=True, append_images=imgs[1:], duration=10, loop=0)


def svg_fn(scene, seq):
    size = scene.terrain.building.shape[0]
    dwg = esch.init(size, size)
    esch.grid_fn(np.array(scene.terrain.building).T, dwg, shape="square")
    arr = np.array(rearrange(seq.coords[:, :, ::-1], "time unit coord -> unit coord time"), dtype=np.float32)
    esch.anim_sims_fn(arr, dwg, fps=100)
    esch.save(dwg, "/Users/nobr/desk/s3/btc2sim/sim.svg")


def svgs_fn(scene, seq):
    size = scene.terrain.building.shape[0]
    side = jnp.sqrt(seq.coords.shape[0]).astype(int).item()
    dwg = esch.init(size, size, side, side, line=True)
    for i in range(side):
        for j in range(side):
            sub_seq = tree.map(lambda x: x[i * side + j], seq)
            group = dwg.g()
            group.translate((size + 1) * i, (size + 1) * j)
            arr = np.array(rearrange(sub_seq.coords[:, :, ::-1], "t unit coord -> unit coord t"), dtype=np.float32)
            esch.grid_fn(np.array(scene.terrain.building).T, dwg, group, shape="square")
            esch.anim_sims_fn(arr, dwg, group, fps=48)
            dwg.add(group)
    esch.save(dwg, "/Users/nobr/desk/s3/btc2sim/sims.svg")


def typst_fn(bt_str):
    pass
