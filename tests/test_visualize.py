"""Tests for src/msae/visualize.py"""
import numpy as np
import torch


def test_mi_scatter_produces_nonempty_png(tmp_path):
    """plot_mi_scatter saves a non-empty PNG to output_path."""
    from msae.visualize import plot_mi_scatter

    np.random.seed(42)
    n = 100
    species_mi = np.random.rand(n)
    disease_mi = np.random.rand(n)
    nested_level = np.random.choice([256, 768, 3072, 12288], size=n)

    output = tmp_path / "mi_scatter.png"
    plot_mi_scatter(species_mi, disease_mi, nested_level, output_path=output)

    assert output.exists(), "plot_mi_scatter must save to output_path"
    assert output.stat().st_size > 0, "Saved PNG must not be empty"


def test_training_curves_no_crash_on_missing_file(tmp_path):
    """plot_training_curves must not crash if log file doesn't exist."""
    from msae.visualize import plot_training_curves

    missing = tmp_path / "nonexistent.json"
    output = tmp_path / "curves.png"
    # Should not raise
    plot_training_curves(missing, output)


def test_spatial_heatmap_produces_png(tmp_path):
    """plot_spatial_heatmap saves a PNG for a synthetic single-image case."""
    from msae.visualize import plot_spatial_heatmap
    from PIL import Image as PILImage

    # Create a fake image
    img = PILImage.new('RGB', (224, 224), color=(128, 200, 100))
    img_path = tmp_path / "test_leaf.jpg"
    img.save(img_path)

    # Fake feature activations for 10 kept patches from image 0
    n_kept = 10
    feature_acts = torch.rand(n_kept)
    patch_meta = torch.zeros(n_kept, 3, dtype=torch.int32)
    patch_meta[:, 0] = 0  # all from image_id=0
    patch_meta[:, 1] = torch.arange(n_kept) % 16  # row
    patch_meta[:, 2] = torch.arange(n_kept) // 16  # col

    output = tmp_path / "heatmap.png"
    plot_spatial_heatmap(img_path, feature_acts, patch_meta, image_id=0, output_path=output)

    assert output.exists()
    assert output.stat().st_size > 0
