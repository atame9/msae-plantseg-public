"""Tests for src/msae/evaluate.py.

All tests are CPU-only with synthetic data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from msae.evaluate import (
    class_selectivity,
    compute_iou_vectorized,
    should_trigger_layer6_fallback,
)


# ---------------------------------------------------------------------------
# Test 1 — perfect feature selectivity ≈ 1
# ---------------------------------------------------------------------------

def test_class_selectivity_perfect_feature_is_one():
    """A feature that fires ONLY on class 0 should have selectivity ≈ 1."""
    n_patches, n_features, n_classes = 100, 5, 4
    # Feature 0 fires only on class 0; others fire uniformly
    acts = torch.zeros(n_patches, n_features)
    labels = np.zeros(n_patches, dtype=int)
    labels[25:] = np.tile([1, 2, 3], 25)[:75]  # distribute other classes
    acts[:25, 0] = 1.0   # feature 0 fires only on class 0 patches
    acts[25:, 1:] = 1.0  # other features fire on non-class-0 patches

    df = class_selectivity(acts, labels, n_classes)
    assert "selectivity" in df.columns, "Column must be named 'selectivity'"
    assert "feature_id" in df.columns
    feat0_sel = df.loc[df["feature_id"] == 0, "selectivity"].values[0]
    assert feat0_sel > 0.9, f"Perfect feature selectivity should be near 1, got {feat0_sel}"


# ---------------------------------------------------------------------------
# Test 2 — uniform feature selectivity ≈ 0
# ---------------------------------------------------------------------------

def test_class_selectivity_uniform_feature_is_zero():
    """A feature firing uniformly across all classes should have selectivity ≈ 0."""
    n_patches, n_features, n_classes = 80, 3, 4
    acts = torch.ones(n_patches, n_features)
    labels = np.array([i % n_classes for i in range(n_patches)])

    df = class_selectivity(acts, labels, n_classes)
    for _, row in df.iterrows():
        assert row["selectivity"] < 0.05, (
            f"Uniform feature {row['feature_id']} selectivity should be ~0, "
            f"got {row['selectivity']}"
        )


# ---------------------------------------------------------------------------
# Test 3 — healthy collapse changes H_max
# ---------------------------------------------------------------------------

def test_healthy_collapse_changes_h_max():
    """With healthy collapse, H_max changes (fewer effective classes)."""
    n_patches, n_features = 60, 2
    # Classes: 0=Tomato_healthy, 1=Apple_healthy, 2=Blight, 3=Rust (4 classes)
    acts = torch.ones(n_patches, n_features) * 0.25
    labels = np.array([i % 4 for i in range(n_patches)])

    df_no_collapse = class_selectivity(acts, labels, n_classes=4)
    df_with_collapse = class_selectivity(
        acts, labels, n_classes=4, healthy_class_ids=[0, 1]
    )

    # Collapse should change selectivity values (different H_max)
    sel_no = df_no_collapse["selectivity"].mean()
    sel_with = df_with_collapse["selectivity"].mean()
    # They won't be identical since H_max differs (log(4) vs log(3))
    assert sel_no != sel_with, (
        "Healthy collapse should change selectivity (different H_max)"
    )


# ---------------------------------------------------------------------------
# Test 4 — perfect IoU alignment ≈ 1
# ---------------------------------------------------------------------------

def test_iou_perfect_alignment_is_one():
    """Feature activation mask == segmentation mask → IoU == 1."""
    n_features, n_images = 2, 4
    # Create masks: first quadrant is True
    masks = torch.zeros(n_images, 16, 16, dtype=torch.bool)
    masks[:, :8, :8] = True

    # Create feature activations: high values exactly where mask is True, 0 elsewhere
    acts = torch.zeros(n_features, n_images, 16, 16)
    acts[:, :, :8, :8] = 2.0   # high activation on masked region
    acts[:, :, 8:, :] = 0.0    # zero elsewhere
    acts[:, :, :, 8:] = 0.0

    iou = compute_iou_vectorized(acts, masks, quantile=0.90, feature_chunk=2)
    # Both features should have IoU == 1
    assert iou.shape == (n_features,)
    for i in range(n_features):
        assert iou[i].item() > 0.95, (
            f"Perfect IoU expected ~1.0, got {iou[i].item()}"
        )


# ---------------------------------------------------------------------------
# Test 5 — disjoint IoU ≈ 0
# ---------------------------------------------------------------------------

def test_iou_disjoint_is_zero():
    """Feature fires only where mask is False → IoU == 0."""
    n_features, n_images = 2, 4
    masks = torch.zeros(n_images, 16, 16, dtype=torch.bool)
    masks[:, :8, :8] = True  # mask covers top-left

    acts = torch.zeros(n_features, n_images, 16, 16)
    acts[:, :, 8:, 8:] = 5.0  # feature fires only on bottom-right (disjoint from mask)

    iou = compute_iou_vectorized(acts, masks, quantile=0.90, feature_chunk=2)
    for i in range(n_features):
        assert iou[i].item() < 0.05, (
            f"Disjoint IoU expected ~0, got {iou[i].item()}"
        )


# ---------------------------------------------------------------------------
# Test 6 — layer-6 fallback threshold
# ---------------------------------------------------------------------------

def test_layer6_fallback_threshold():
    """gap < 0.05 triggers fallback; gap >= 0.05 does not."""
    # gap = 0.04 (below threshold) → trigger = True
    msae_df = pd.DataFrame({"feature_id": [0, 1], "selectivity": [0.54, 0.54]})  # mean=0.54
    neuron_df = pd.DataFrame({"feature_id": [0, 1], "selectivity": [0.50, 0.50]})  # mean=0.50
    # gap = 0.04 < 0.05
    trigger, diag = should_trigger_layer6_fallback(msae_df, neuron_df, threshold=0.05)
    assert trigger is True, f"gap=0.04 should trigger fallback, got {trigger}"
    assert abs(diag["gap"] - 0.04) < 1e-6

    # gap = 0.06 (above threshold) → trigger = False
    msae_df2 = pd.DataFrame({"feature_id": [0, 1], "selectivity": [0.56, 0.56]})  # mean=0.56
    trigger2, diag2 = should_trigger_layer6_fallback(msae_df2, neuron_df, threshold=0.05)
    assert trigger2 is False, f"gap=0.06 should NOT trigger fallback, got {trigger2}"


# ---------------------------------------------------------------------------
# D1 — transfer_correlation uses alignment CSV mapping
# ---------------------------------------------------------------------------

def _tiny_msae_encoder(input_dim: int = 8, max_features: int = 16):
    """Small fixed-weight MSAE suitable for CPU transfer_correlation tests."""
    from msae.models import MatryoshkaSAE

    torch.manual_seed(0)
    model = MatryoshkaSAE(
        input_dim=input_dim,
        max_features=max_features,
        nested_ks=(4, 8, 16),
    )
    model.eval()
    return model


def _make_per_patch_labels(class_names_per_patch, species_per_patch, disease_per_patch):
    """Build a minimal per-patch DataFrame matching the columns
    transfer_correlation touches (``class_name`` for grouping)."""
    return pd.DataFrame({
        "image_id": list(range(len(class_names_per_patch))),
        "class_name": class_names_per_patch,
        "species": species_per_patch,
        "disease": disease_per_patch,
    })


def test_transfer_correlation_uses_alignment_csv_mapping(tmp_path):
    """PlantSeg 'Tomato_Blight' and PlantVillage 'Tomato___Blight' never
    intersect as strings but DO map via the alignment CSV. The D1 fix must
    produce a non-zero cosine sim across the mapped pair.

    This specifically guards against the pre-fix set-intersection code path
    which silently returned mean_cosine_sim=0.0 on this exact scenario.
    """
    from msae.evaluate import transfer_correlation
    from msae.data import _save_class_alignment_csv

    input_dim = 8
    model = _tiny_msae_encoder(input_dim=input_dim)

    # Build 12 patches per dataset so the ≥8 n_overlap gate is satisfied
    # and neither groupby sees an empty group.
    ps_classes = ["Tomato_Blight", "Apple_Scab"] * 6         # 12 patches
    pv_classes = ["Tomato___Blight", "Apple___Scab"] * 6     # 12 patches

    ps_acts = torch.randn(len(ps_classes), input_dim)
    pv_acts = torch.randn(len(pv_classes), input_dim)

    ps_labels = _make_per_patch_labels(
        class_names_per_patch=ps_classes,
        species_per_patch=["Tomato", "Apple"] * 6,
        disease_per_patch=["Blight", "Scab"] * 6,
    )
    pv_labels = _make_per_patch_labels(
        class_names_per_patch=pv_classes,
        species_per_patch=["Tomato", "Apple"] * 6,
        disease_per_patch=["Blight", "Scab"] * 6,
    )

    # Alignment CSV maps PS→PV class names; header must claim ≥8 overlap to
    # bypass the qualitative early-return branch.
    alignment_df = pd.DataFrame({
        "plantseg_class": ["Tomato_Blight", "Apple_Scab"],
        "plantvillage_class": ["Tomato___Blight", "Apple___Scab"],
        "species": ["Tomato", "Apple"],
        "disease": ["Blight", "Scab"],
        "n_plantseg": [6, 6],
        "n_plantvillage": [6, 6],
    })
    csv_path = tmp_path / "class_alignment.csv"
    _save_class_alignment_csv(alignment_df, n_overlap=8, path=csv_path)

    result = transfer_correlation(
        msae_encoder=model,
        plantseg_acts=ps_acts,
        plantvillage_acts=pv_acts,
        plantseg_labels=ps_labels,
        plantvillage_labels=pv_labels,
        class_alignment_csv=csv_path,
        encode_batch_size=16,
    )

    assert result["mode"] == "quantitative"
    assert len(result["cosine_similarities"]) >= 1
    assert result["mean_cosine_sim"] != 0.0, (
        "D1 regression: mean_cosine_sim is 0 — alignment CSV mapping not "
        "being consumed (set-intersection path returned)"
    )


def test_transfer_correlation_raises_on_empty_match(tmp_path):
    """Alignment CSV whose class names don't exist in either label DataFrame
    must fail loud via the new `len(sims_list) > 0` assertion, not return a
    silent mean_cosine_sim=0.0 / 0.0 NaN."""
    from msae.evaluate import transfer_correlation
    from msae.data import _save_class_alignment_csv

    input_dim = 8
    model = _tiny_msae_encoder(input_dim=input_dim)

    # Labels use real class names, but the alignment CSV points to classes
    # that aren't in either labels DataFrame.
    ps_classes = ["Tomato_Blight"] * 12
    pv_classes = ["Tomato___Blight"] * 12

    ps_acts = torch.randn(len(ps_classes), input_dim)
    pv_acts = torch.randn(len(pv_classes), input_dim)

    ps_labels = _make_per_patch_labels(
        ps_classes, ["Tomato"] * 12, ["Blight"] * 12
    )
    pv_labels = _make_per_patch_labels(
        pv_classes, ["Tomato"] * 12, ["Blight"] * 12
    )

    alignment_df = pd.DataFrame({
        "plantseg_class": ["Wheat_Rust", "Corn_Smut"],
        "plantvillage_class": ["Wheat___Rust", "Corn___Smut"],
        "species": ["Wheat", "Corn"],
        "disease": ["Rust", "Smut"],
        "n_plantseg": [1, 1],
        "n_plantvillage": [1, 1],
    })
    csv_path = tmp_path / "class_alignment.csv"
    _save_class_alignment_csv(alignment_df, n_overlap=8, path=csv_path)

    import pytest
    with pytest.raises(AssertionError, match="no matched classes found"):
        transfer_correlation(
            msae_encoder=model,
            plantseg_acts=ps_acts,
            plantvillage_acts=pv_acts,
            plantseg_labels=ps_labels,
            plantvillage_labels=pv_labels,
            class_alignment_csv=csv_path,
            encode_batch_size=16,
        )


def test_transfer_correlation_qualitative_below_threshold(tmp_path):
    """n_overlap < 8 returns `mode='qualitative'` without running the model."""
    from msae.evaluate import transfer_correlation
    from msae.data import _save_class_alignment_csv

    input_dim = 8
    model = _tiny_msae_encoder(input_dim=input_dim)

    alignment_df = pd.DataFrame({
        "plantseg_class": ["Tomato_Blight"],
        "plantvillage_class": ["Tomato___Blight"],
        "species": ["Tomato"],
        "disease": ["Blight"],
        "n_plantseg": [1],
        "n_plantvillage": [1],
    })
    csv_path = tmp_path / "class_alignment.csv"
    _save_class_alignment_csv(alignment_df, n_overlap=5, path=csv_path)

    # Acts can be empty-ish; the function must short-circuit before encoding.
    ps_acts = torch.randn(4, input_dim)
    pv_acts = torch.randn(4, input_dim)
    ps_labels = _make_per_patch_labels(
        ["Tomato_Blight"] * 4, ["Tomato"] * 4, ["Blight"] * 4,
    )
    pv_labels = _make_per_patch_labels(
        ["Tomato___Blight"] * 4, ["Tomato"] * 4, ["Blight"] * 4,
    )

    result = transfer_correlation(
        msae_encoder=model,
        plantseg_acts=ps_acts,
        plantvillage_acts=pv_acts,
        plantseg_labels=ps_labels,
        plantvillage_labels=pv_labels,
        class_alignment_csv=csv_path,
        encode_batch_size=16,
    )

    assert result["mode"] == "qualitative"
    assert result["n_overlap"] == 5
    assert result["matched_classes"] == []
