from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — required for headless test environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import Tensor
from PIL import Image

logger = logging.getLogger(__name__)


def plot_top_activating_patches(
    feature_id: int,
    feature_acts: Tensor,
    image_paths_by_id: dict[int, Path] | list[Path],
    patch_meta: Tensor,
    grid: int = 3,
    output_path: Path = Path("results/top_patches_feature_{feature_id}.png"),
) -> None:
    """Find top grid*grid patches by activation and display as a grid.

    ``image_paths_by_id`` is keyed/indexed by IMAGE ID (the value in
    ``patch_meta[:, 0]``), not by patch index. Use a dict if image_ids are
    sparse or non-contiguous; a list works if image_ids are 0..N-1. Passing a
    per-patch list of length ``n_patches`` will raise IndexError at runtime
    against the 3.3M-patch dataset (this is the A5-A7 regression).
    """
    n_top = min(grid * grid, len(feature_acts))
    if n_top == 0:
        logger.warning("plot_top_activating_patches: empty feature_acts for feature %d", feature_id)
        return

    topk = feature_acts.topk(n_top)
    top_indices = topk.indices
    top_values = topk.values

    fig, axes = plt.subplots(grid, grid, figsize=(grid * 2, grid * 2))
    if grid == 1:
        axes = np.array([[axes]])
    elif grid > 1:
        axes = np.array(axes).reshape(grid, grid)

    for flat_idx in range(grid * grid):
        row_idx = flat_idx // grid
        col_idx = flat_idx % grid
        ax = axes[row_idx, col_idx]
        ax.axis('off')

        if flat_idx >= n_top:
            ax.imshow(np.full((64, 64, 3), 128, dtype=np.uint8))
            continue

        patch_idx = top_indices[flat_idx].item()
        act_val = top_values[flat_idx].item()
        img_id = int(patch_meta[patch_idx, 0].item())
        if isinstance(image_paths_by_id, dict):
            img_path = image_paths_by_id.get(img_id)
        else:
            img_path = image_paths_by_id[img_id] if img_id < len(image_paths_by_id) else None
        if img_path is None:
            logger.warning("No image path for image_id %d; using gray fill", img_id)
            ax.imshow(np.full((64, 64, 3), 128, dtype=np.uint8))
            ax.set_title(f"act={act_val:.2f}", fontsize=7)
            continue

        row = patch_meta[patch_idx, 1].item()
        col = patch_meta[patch_idx, 2].item()

        # DINOv2 patch stride = 14 pixels in a 224x224 image
        top_px = row * 14
        left_px = col * 14
        bottom_px = top_px + 14
        right_px = left_px + 14

        patch_img = None
        if Path(img_path).exists():
            try:
                full_img = Image.open(img_path).convert('RGB')
                patch_img = full_img.crop((left_px, top_px, right_px, bottom_px))
                patch_img = patch_img.resize((64, 64), Image.BILINEAR)
            except Exception as e:
                logger.warning("Failed to load image %s: %s", img_path, e)
                patch_img = None
        else:
            logger.warning("Image file not found: %s", img_path)

        if patch_img is None:
            patch_arr = np.full((64, 64, 3), 128, dtype=np.uint8)
        else:
            patch_arr = np.array(patch_img)

        ax.imshow(patch_arr)
        ax.set_title(f"act={act_val:.2f}", fontsize=7)

    plt.suptitle(f"Feature {feature_id} -- Top Activating Patches", fontsize=10)
    plt.tight_layout()

    output_path = Path(str(output_path).replace("{feature_id}", str(feature_id)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved top activating patches to %s", output_path)


def plot_spatial_heatmap(
    image_path: Path,
    feature_acts_per_patch: Tensor,  # (256,) or (n_kept, 3+) — see below
    patch_meta: Tensor,               # (n_patches, 3) int [image_id, row, col]
    image_id: int,
    output_path: Path = Path("results/heatmap.png"),
) -> None:
    """Overlay a per-patch activation heatmap on the original image."""
    # Filter metadata to patches for this image
    mask = patch_meta[:, 0] == image_id
    img_patch_meta = patch_meta[mask]
    img_acts = feature_acts_per_patch[mask]

    # Build a 16×16 grid of activation values (zeros for missing patches)
    heatmap = np.zeros((16, 16), dtype=np.float32)
    for i in range(len(img_patch_meta)):
        r = img_patch_meta[i, 1].item()
        c = img_patch_meta[i, 2].item()
        if 0 <= r < 16 and 0 <= c < 16:
            act_val = img_acts[i].item() if img_acts.ndim == 1 else float(img_acts[i])
            heatmap[r, c] = act_val

    # Load original image and resize to 224×224
    try:
        orig_img = Image.open(image_path).convert('RGB').resize((224, 224), Image.BILINEAR)
    except Exception as e:
        logger.warning("Failed to load image %s: %s", image_path, e)
        orig_img = Image.new('RGB', (224, 224), (128, 128, 128))

    # Upsample 16×16 heatmap to 224×224 using PIL BILINEAR
    heatmap_img = Image.fromarray(heatmap).resize((224, 224), Image.BILINEAR)
    heatmap_arr = np.array(heatmap_img)

    # Normalize heatmap to [0, 1] for colormap
    hmap_min = heatmap_arr.min()
    hmap_max = heatmap_arr.max()
    if hmap_max > hmap_min:
        heatmap_norm = (heatmap_arr - hmap_min) / (hmap_max - hmap_min)
    else:
        heatmap_norm = np.zeros_like(heatmap_arr)

    # Apply colormap
    colormap = plt.cm.jet
    heatmap_rgba = colormap(heatmap_norm)  # (224, 224, 4)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    # Overlay heatmap on image with alpha=0.5
    orig_arr = np.array(orig_img).astype(np.float32)
    overlay = (orig_arr * 0.5 + heatmap_rgb.astype(np.float32) * 0.5).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(overlay)
    ax.axis('off')
    ax.set_title(f"Spatial Heatmap — image_id={image_id}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved spatial heatmap to %s", output_path)


def plot_mi_scatter(
    species_mi: np.ndarray,     # (n_features,) — MI with species
    disease_mi: np.ndarray,     # (n_features,) — MI with disease
    nested_level: np.ndarray,   # (n_features,) int — which Matryoshka level each feature belongs to
    output_path: Path = Path("results/mi_scatter.png"),
    nested_ks: tuple[int, ...] = (256, 768, 3072, 12288),
) -> None:
    """Scatterplot of species MI vs disease MI, colored by Matryoshka nesting level."""
    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = matplotlib.colormaps.get_cmap('tab10').resampled(len(nested_ks))

    level_labels = {
        nested_ks[0]: f"k={nested_ks[0]} (coarsest)",
        nested_ks[-1]: f"k={nested_ks[-1]} (finest)",
    }
    for k in nested_ks[1:-1]:
        level_labels[k] = f"k={k}"

    for i, k in enumerate(nested_ks):
        mask = nested_level == k
        if not np.any(mask):
            continue
        ax.scatter(
            species_mi[mask],
            disease_mi[mask],
            color=cmap(i),
            label=level_labels[k],
            alpha=0.7,
            s=20,
        )

    ax.set_xlabel("Species MI")
    ax.set_ylabel("Disease MI")
    ax.set_title("Species-Disease MI by Matryoshka Level")
    ax.legend(loc='best', fontsize=9)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved MI scatter to %s", output_path)


def plot_selectivity_per_level(
    msae_selectivity_df: pd.DataFrame,  # columns: feature_id, selectivity, level (int, Matryoshka level)
    output_path: Path = Path("results/selectivity_per_level.png"),
) -> None:
    """Line plot of mean selectivity ± std per Matryoshka nesting level."""
    grouped = msae_selectivity_df.groupby('level')['selectivity']
    means = grouped.mean().sort_index()
    stds = grouped.std().sort_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    levels = means.index.to_numpy()
    ax.errorbar(
        levels,
        means.to_numpy(),
        yerr=stds.fillna(0).to_numpy(),
        marker='o',
        capsize=4,
        label='Mean Selectivity ± Std',
    )
    ax.set_xlabel("Matryoshka Level (k)")
    ax.set_ylabel("Selectivity")
    ax.set_title("Mean Selectivity per Matryoshka Level")
    ax.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved selectivity per level to %s", output_path)


def plot_selectivity_comparison(
    comparison_df: pd.DataFrame,  # columns: method, species_selectivity, disease_selectivity
    output_path: Path = Path("results/selectivity_comparison.png"),
) -> None:
    """Grouped bar chart comparing mean species and disease selectivity across methods.

    ``comparison_df`` has one row per method (e.g. 'Neuron basis', 'PCA k=256',
    'Standard SAE', 'MSAE') and two numeric columns: ``species_selectivity`` and
    ``disease_selectivity``, each the mean over that method's features. Rows with
    NaN for a given column are treated as missing (the bar is omitted).
    """
    if comparison_df.empty:
        logger.warning("plot_selectivity_comparison: empty comparison_df — skipping plot.")
        return

    methods = comparison_df["method"].tolist()
    x = np.arange(len(methods))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(methods) + 2), 5))

    species_vals = comparison_df["species_selectivity"].to_numpy()
    disease_vals = comparison_df["disease_selectivity"].to_numpy()

    # np.isfinite treats NaN as missing; matplotlib skips NaN bars cleanly but
    # leaves a gap. That's fine — it makes absent baselines obvious.
    ax.bar(x - width / 2, species_vals, width, label="Species selectivity", color="#4C78A8")
    ax.bar(x + width / 2, disease_vals, width, label="Disease selectivity", color="#F58518")

    for i, v in enumerate(species_vals):
        if np.isfinite(v):
            ax.text(i - width / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(disease_vals):
        if np.isfinite(v):
            ax.text(i + width / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Mean selectivity (over features)")
    ax.set_title("Selectivity: MSAE vs. baselines")
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(bottom=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved selectivity comparison to %s", output_path)


def plot_training_curves(
    log_path: Path,
    output_path: Path = Path("results/training_curves.png"),
) -> None:
    """Load JSON-lines training log and plot val MSE, alive features, and mean L0."""
    log_path = Path(log_path)
    if not log_path.exists():
        logger.warning("Training log not found: %s — skipping plot.", log_path)
        return

    records = []
    try:
        with open(log_path, 'r') as f:
            records = json.load(f)
        if not isinstance(records, list):
            records = [records]
    except Exception as e:
        logger.warning("Failed to read training log %s: %s — skipping plot.", log_path, e)
        return

    if not records:
        logger.warning("Training log %s is empty — skipping plot.", log_path)
        return

    epochs = [r.get('epoch', i) for i, r in enumerate(records)]

    # Gather all level keys
    level_keys: list[str] = []
    for r in records:
        val_mse = r.get('val_mse_per_level', {})
        for k in val_mse:
            if str(k) not in level_keys:
                level_keys.append(str(k))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Subplot 1: Val MSE per level
    ax = axes[0]
    for lk in level_keys:
        vals = []
        for r in records:
            vmap = r.get('val_mse_per_level', {})
            vals.append(vmap.get(lk, vmap.get(int(lk) if lk.isdigit() else lk, float('nan'))))
        ax.plot(epochs, vals, marker='o', label=f"k={lk}")
    ax.set_title("Val MSE per Level")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=8)

    # Subplot 2: Alive feature count per level
    ax = axes[1]
    alive_keys: list[str] = []
    for r in records:
        amap = r.get('alive_features_per_level', {})
        for k in amap:
            if str(k) not in alive_keys:
                alive_keys.append(str(k))
    for lk in alive_keys:
        vals = []
        for r in records:
            amap = r.get('alive_features_per_level', {})
            vals.append(amap.get(lk, amap.get(int(lk) if lk.isdigit() else lk, float('nan'))))
        ax.plot(epochs, vals, marker='o', label=f"k={lk}")
    ax.set_title("Alive Features per Level")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # Subplot 3: Mean L0
    ax = axes[2]
    l0_vals = [r.get('mean_l0', float('nan')) for r in records]
    ax.plot(epochs, l0_vals, marker='o', color='tab:green')
    ax.set_title("Mean L0 Across Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean L0")

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved training curves to %s", output_path)
