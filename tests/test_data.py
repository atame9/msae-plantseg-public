"""
tests/test_data.py — Unit tests for src/msae/data.py.

All tests are CPU-only with synthetic fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from msae.data import (
    build_class_alignment,
    make_image_dataset,
    parse_plantseg_metadata,
    parse_plantvillage_metadata,
    resize_mask_to_patch_grid,
    stratified_sample,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_jpeg(path: Path) -> None:
    """Create a 32×32 RGB JPEG at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(100, 150, 200)).save(str(path), format="JPEG")


# ---------------------------------------------------------------------------
# Test 1: parse_plantseg_metadata smoke
# ---------------------------------------------------------------------------

def test_parse_plantseg_metadata_smoke(tmp_path):
    """Synthetic dir tree: 4 fake jpg files in 2 species × 2 disease subdirs."""
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Create 4 images in 2 class subdirs
    _make_tiny_jpeg(images_dir / "Tomato_Early_Blight" / "img1.jpg")
    _make_tiny_jpeg(images_dir / "Tomato_Early_Blight" / "img2.jpg")
    _make_tiny_jpeg(images_dir / "Apple_healthy" / "img3.jpg")
    _make_tiny_jpeg(images_dir / "Apple_healthy" / "img4.jpg")

    df = parse_plantseg_metadata(images_dir, masks_dir)

    # Shape
    assert len(df) == 4, f"Expected 4 rows, got {len(df)}"

    # Required columns present
    required_cols = {
        "image_id",
        "filepath",
        "mask_filepath",
        "species",
        "disease",
        "class_name",
        "has_segmentation_mask",
    }
    assert required_cols.issubset(set(df.columns)), (
        f"Missing columns: {required_cols - set(df.columns)}"
    )

    # Species parsing (normalised to lowercase by _norm_species)
    species_set = set(df["species"].tolist())
    assert "tomato" in species_set, f"Expected 'tomato' in species, got {species_set}"
    assert "apple" in species_set, f"Expected 'apple' in species, got {species_set}"

    # Disease parsing (normalised to lowercase underscores by _norm_disease)
    disease_set = set(df["disease"].tolist())
    assert "early_blight" in disease_set, (
        f"Expected 'early_blight' in diseases, got {disease_set}"
    )
    assert "healthy" in disease_set, f"Expected 'healthy' in diseases, got {disease_set}"

    # dtype checks — accept both object and StringDtype (pandas 2.x)
    assert pd.api.types.is_string_dtype(df["species"]), (
        f"species should be a string dtype, got {df['species'].dtype}"
    )
    assert pd.api.types.is_string_dtype(df["disease"]), (
        f"disease should be a string dtype, got {df['disease'].dtype}"
    )
    assert df["has_segmentation_mask"].dtype == bool, (
        f"has_segmentation_mask dtype should be bool, got {df['has_segmentation_mask'].dtype}"
    )

    # No masks → all False
    assert not df["has_segmentation_mask"].any(), "No masks present, all should be False"


def test_parse_plantseg_metadata_with_masks(tmp_path):
    """Verify has_segmentation_mask is True when a matching mask file exists."""
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"

    _make_tiny_jpeg(images_dir / "Tomato_Blight" / "img1.jpg")
    # Create a matching mask
    _make_tiny_jpeg(masks_dir / "Tomato_Blight" / "img1.jpg")

    df = parse_plantseg_metadata(images_dir, masks_dir)

    assert len(df) == 1
    assert bool(df.iloc[0]["has_segmentation_mask"]) is True
    assert df.iloc[0]["mask_filepath"] is not None


# ---------------------------------------------------------------------------
# Test 2: build_class_alignment overlap counting
# ---------------------------------------------------------------------------

def test_class_alignment_overlap_counting():
    """Hand-built DataFrames with 3 overlapping classes; assert n_overlap == 3."""
    # PlantSeg data
    ps_data = {
        "image_id": ["ps1", "ps2", "ps3", "ps4", "ps5", "ps6"],
        "filepath": ["f1", "f2", "f3", "f4", "f5", "f6"],
        "mask_filepath": [None] * 6,
        "species": ["Tomato", "Tomato", "Apple", "Apple", "Grape", "Pepper"],
        "disease": ["Blight", "Healthy", "Rust", "Scab", "Mildew", "Blight"],
        "class_name": [
            "Tomato_Blight",
            "Tomato_Healthy",
            "Apple_Rust",
            "Apple_Scab",
            "Grape_Mildew",
            "Pepper_Blight",
        ],
        "has_segmentation_mask": [False] * 6,
    }
    plantseg_df = pd.DataFrame(ps_data)

    # PlantVillage data — 3 overlapping classes (case may differ slightly)
    pv_data = {
        "image_id": ["pv1", "pv2", "pv3", "pv4"],
        "filepath": ["g1", "g2", "g3", "g4"],
        "mask_filepath": [None] * 4,
        "species": ["tomato", "tomato", "apple", "Strawberry"],  # case differs
        "disease": ["blight", "healthy", "rust", "disease"],  # case differs
        "class_name": [
            "tomato___blight",
            "tomato___healthy",
            "apple___rust",
            "Strawberry___disease",
        ],
        "has_segmentation_mask": [False] * 4,
    }
    plantvillage_df = pd.DataFrame(pv_data)

    alignment_df, n_overlap = build_class_alignment(plantseg_df, plantvillage_df)

    assert n_overlap == 3, f"Expected n_overlap == 3, got {n_overlap}"
    assert len(alignment_df) == 3, f"Expected 3 rows in alignment_df, got {len(alignment_df)}"

    # Check required columns
    required_cols = {
        "plantseg_class",
        "plantvillage_class",
        "species",
        "disease",
        "n_plantseg",
        "n_plantvillage",
    }
    assert required_cols.issubset(set(alignment_df.columns)), (
        f"Missing columns: {required_cols - set(alignment_df.columns)}"
    )

    # ---- Rule 6: extended expectations against the real-data shape ----
    # Fixture below uses species/disease strings that look like what the parsers
    # emit on real data post-_norm_species / _norm_disease: lowercase,
    # underscored, with PlantSeg's redundant plant-prefix still attached in the
    # disease column (build_class_alignment strips it symmetrically).
    ps_real = pd.DataFrame({
        "image_id": ["r1", "r2", "r3", "r4", "r5"],
        "filepath": ["p1", "p2", "p3", "p4", "p5"],
        "mask_filepath": [None] * 5,
        # species is _norm_species(Plant): "Apple" → "apple", "Bell pepper" → "bellpepper"
        "species": ["apple", "apple", "tomato", "potato", "bellpepper"],
        # disease is _norm_disease(Disease) which keeps PS's redundant plant
        # prefix (e.g. "apple scab" → "apple_scab").
        "disease": [
            "apple_scab",
            "apple_black_rot",
            "tomato_early_blight",
            "potato_late_blight",
            "bell_pepper_bacterial_spot",
        ],
        "class_name": [
            "apple_apple_scab",
            "apple_apple_black_rot",
            "tomato_tomato_early_blight",
            "potato_potato_late_blight",
            "bellpepper_bell_pepper_bacterial_spot",
        ],
        "has_segmentation_mask": [False] * 5,
    })
    # PV strings mirror what _norm_species / _norm_disease produce on directory
    # names: "Apple___Apple_scab" → species="apple", disease="apple_scab";
    # "Pepper,_bell___Bacterial_spot" → species="bellpepper" (alias),
    # disease="bacterial_spot".
    pv_real = pd.DataFrame({
        "image_id": ["v1", "v2", "v3", "v4", "v5"],
        "filepath": ["q1", "q2", "q3", "q4", "q5"],
        "mask_filepath": [None] * 5,
        "species": ["apple", "apple", "tomato", "potato", "bellpepper"],
        "disease": [
            "apple_scab",
            "black_rot",
            "early_blight",
            "late_blight",
            "bacterial_spot",
        ],
        "class_name": [
            "Apple___Apple_scab",
            "Apple___Black_rot",
            "Tomato___Early_blight",
            "Potato___Early_blight",
            "Pepper,_bell___Bacterial_spot",
        ],
        "has_segmentation_mask": [False] * 5,
    })
    alignment_real, n_real = build_class_alignment(ps_real, pv_real)

    # Build the canonical (species, disease) post-strip set.
    expected_pairs = {
        ("apple", "scab"),            # PS apple_scab ↔ PV apple_scab (prefix stripped both sides)
        ("apple", "black_rot"),       # PS apple_black_rot ↔ PV black_rot
        ("tomato", "early_blight"),   # PS tomato_early_blight ↔ PV early_blight
        ("potato", "late_blight"),    # PS potato_late_blight ↔ PV late_blight
        ("bellpepper", "bacterial_spot"),  # exercises _SPECIES_ALIAS
    }
    got_pairs = set(zip(alignment_real["species"].str.lower(),
                        alignment_real["disease"].str.lower()))
    missing = expected_pairs - got_pairs
    assert not missing, (
        f"build_class_alignment missed expected pairs: {missing}; got {got_pairs}"
    )
    assert n_real >= len(expected_pairs), (
        f"n_overlap={n_real} below expected subset size {len(expected_pairs)}"
    )


# ---------------------------------------------------------------------------
# Test 3: resize_mask_block_mean
# ---------------------------------------------------------------------------

def test_resize_mask_block_mean():
    """Three threshold cases for resize_mask_to_patch_grid."""
    # Case 1: 224×224 all-True → all 16×16 cells True
    mask_all_true = np.ones((224, 224), dtype=bool)
    result = resize_mask_to_patch_grid(mask_all_true)
    assert result.shape == (16, 16), f"Expected (16, 16), got {result.shape}"
    assert result.all(), "All-True mask should produce all-True result"

    # Case 2: 224×224 all-False → all 16×16 cells False
    mask_all_false = np.zeros((224, 224), dtype=bool)
    result = resize_mask_to_patch_grid(mask_all_false)
    assert result.shape == (16, 16)
    assert not result.any(), "All-False mask should produce all-False result"

    # Case 3: Exactly 50% True per block → strict >0.5 means False
    # Each 14×14 block has exactly 7/14 = 50% True rows → mean = 0.5, should be False
    block = np.zeros((14, 14), dtype=bool)
    block[:7, :] = True  # exactly 7/14 = 50%
    mask_half = np.tile(block, (16, 16))  # 224×224
    assert mask_half.shape == (224, 224)
    result = resize_mask_to_patch_grid(mask_half)
    assert result.shape == (16, 16)
    assert not result.any(), (
        "Exactly 50% True should not exceed the strict >0.5 threshold — all cells must be False"
    )


def test_resize_mask_non_divisible_size():
    """Masks with dimensions not divisible by 16 should be handled gracefully."""
    mask = np.ones((100, 100), dtype=bool)
    result = resize_mask_to_patch_grid(mask)
    assert result.shape == (16, 16)
    assert result.all(), "All-ones non-divisible mask should still produce all-True result"


# ---------------------------------------------------------------------------
# Test 4: make_image_dataset length and shape
# ---------------------------------------------------------------------------

def test_make_image_dataset_len_and_shape(tmp_path):
    """Create 2 fake images, build df, assert len==2 and tensor shape."""
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    _make_tiny_jpeg(img1)
    _make_tiny_jpeg(img2)

    df = pd.DataFrame(
        {
            "image_id": ["img1.jpg", "img2.jpg"],
            "filepath": [str(img1), str(img2)],
            "mask_filepath": [None, None],
            "species": ["Tomato", "Apple"],
            "disease": ["Blight", "Healthy"],
            "class_name": ["Tomato_Blight", "Apple_Healthy"],
            "has_segmentation_mask": [False, False],
        }
    )

    dataset = make_image_dataset(df, tmp_path)

    assert len(dataset) == 2, f"Expected dataset length 2, got {len(dataset)}"

    tensor, image_id = dataset[0]
    assert isinstance(tensor, torch.Tensor), "First element should be a tensor"
    assert tensor.shape == (3, 224, 224), f"Expected shape (3, 224, 224), got {tensor.shape}"
    assert isinstance(image_id, str), "Second element should be a str"

    tensor2, image_id2 = dataset[1]
    assert tensor2.shape == (3, 224, 224)
    assert isinstance(image_id2, str)


def test_make_image_dataset_missing_file(tmp_path):
    """Dataset should return zero tensor for a missing file rather than crashing."""
    df = pd.DataFrame(
        {
            "image_id": ["missing.jpg"],
            "filepath": [str(tmp_path / "nonexistent.jpg")],
            "mask_filepath": [None],
            "species": ["Tomato"],
            "disease": ["Blight"],
            "class_name": ["Tomato_Blight"],
            "has_segmentation_mask": [False],
        }
    )
    dataset = make_image_dataset(df, tmp_path)
    tensor, image_id = dataset[0]
    assert tensor.shape == (3, 224, 224)
    assert torch.all(tensor == 0), "Missing file should return a zero tensor"


# ---------------------------------------------------------------------------
# Test 5: stratified_sample
# ---------------------------------------------------------------------------

def test_stratified_sample_total():
    """Total sampled rows should not exceed n."""
    rng = np.random.default_rng(0)
    classes = ["A", "B", "C", "D"]
    class_list = rng.choice(classes, size=200)
    df = pd.DataFrame({"class_name": class_list, "value": range(200)})

    result = stratified_sample(df, n=50, seed=42)
    assert len(result) <= 50, f"Expected <= 50 rows, got {len(result)}"
    # All classes should be represented
    assert set(result["class_name"].unique()) == set(classes)


def test_stratified_sample_small_class():
    """A class with fewer rows than its share should contribute all its rows."""
    df = pd.DataFrame(
        {
            "class_name": ["A"] * 100 + ["B"] * 2,
            "value": range(102),
        }
    )
    result = stratified_sample(df, n=50, seed=42)
    # B should have all 2 rows since it has fewer than its proportional share
    b_count = (result["class_name"] == "B").sum()
    assert b_count == 2, f"Expected class B to have 2 rows (all), got {b_count}"


# ---------------------------------------------------------------------------
# Test 6: parse_plantvillage_metadata
# ---------------------------------------------------------------------------

def test_parse_plantvillage_metadata_smoke(tmp_path):
    """Verify triple-underscore splitting for PlantVillage."""
    images_dir = tmp_path / "pv_images"
    _make_tiny_jpeg(images_dir / "Tomato___Early_blight" / "img_pv1.jpg")
    _make_tiny_jpeg(images_dir / "Apple___healthy" / "img_pv2.jpg")

    df = parse_plantvillage_metadata(images_dir)

    assert len(df) == 2
    # species/disease are normalised via _norm_species / _norm_disease.
    assert set(df["species"].tolist()) == {"tomato", "apple"}
    assert set(df["disease"].tolist()) == {"early_blight", "healthy"}
    assert df["has_segmentation_mask"].dtype == bool
    assert not df["has_segmentation_mask"].any()
    assert df["mask_filepath"].isna().all() or df["mask_filepath"].isnull().all()
