# imports
import jax.numpy as jnp
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
    return arr
