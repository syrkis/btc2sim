# imports
import jax.numpy as jnp
import numpy as np
from einops import repeat, rearrange
from PIL import Image
import esch

# def plot_grads(grads):
# fig, axes = plt.subplots(2, 3, figsize=(12, 8))
# for i, ax in enumerate(axes.flat):
# grad = grads[i]
# sns.heatmap(grad, ax=ax, cbar=False, cmap="twilight")
# sns.heatmap(scene.terrain.building, ax=ax, cbar=False, cmap="grey", alpha=0.01)
# ax.scatter(*gps.marks[targets[i]], c="red", marker="x", s=50)
# ax.set_title(f"Gradient {i + 1}")
# ax.axis("off")
# plt.tight_layout()
# plt.show()
#


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
    imgs = np.array(repeat(mask, "... -> a ...", a=len(pos)).at[*idxs].set(1))
    imgs = [Image.fromarray(img).resize(np.array(img.shape[:2]) * scale, Image.NEAREST) for img in imgs * 255]  # type: ignore
    imgs[0].save("/Users/nobr/desk/s3/btc2sim/output.gif", save_all=True, append_images=imgs[1:], duration=10, loop=0)


def svg_fn(scene, seq):
    dwg = esch.init(100, 100)
    esch.grid_fn(np.array(scene.terrain.building).T, dwg)
    arr = np.array(rearrange(seq.coords[:, :, ::-1], "time unit coord -> unit coord time"), dtype=np.float32)
    esch.anim_sims_fn(arr, dwg, fps=48)
    esch.save(dwg, "/Users/nobr/desk/s3/btc2sim/test.svg")


def typst_fn(bt_str):
    pass
