"""
data.py — Data plumbing for MSAE project.

CPU-only module. No imports from extraction.py, models.py, or train.py.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Class-name normalisation
# ---------------------------------------------------------------------------
# PlantSeg's Metadatav2.csv uses human-readable Plant / Disease strings
# (e.g. Plant="Bell pepper", Disease="bell pepper bacterial spot"). PlantVillage's
# on-disk class directories use machine-readable names with decorations
# (e.g. "Pepper,_bell___Bacterial_spot", "Corn_(maize)___Common_rust_"). To
# merge the two on (species, disease) we normalise both sides into the same
# punctuation-free lowercase token space, then post-strip PS's redundant plant
# prefix from its disease column in build_class_alignment.
_SPECIES_ALIAS = {
    "cornmaize": "corn",
    "pepperbell": "bellpepper",
    "cherryincludingsour": "cherry",
}


def _norm_species(s: str) -> str:
    """Collapse a Plant/Species string to a punctuation-free lowercase token.

    Applies the _SPECIES_ALIAS table to reconcile PlantVillage's decorated
    species names ("Pepper,_bell" → "bellpepper", "Corn_(maize)" → "corn",
    "Cherry_(including_sour)" → "cherry") with PlantSeg's plain ones.
    """
    key = re.sub(r"[\s,()_]", "", s.lower())
    return _SPECIES_ALIAS.get(key, key)


def _norm_disease(d: str) -> str:
    """Lowercase, collapse spaces to underscores, deduplicate, strip edges."""
    return re.sub(r"_+", "_", d.lower().replace(" ", "_")).strip("_")


# ---------------------------------------------------------------------------
# 1. parse_plantseg_metadata
# ---------------------------------------------------------------------------

def parse_plantseg_metadata(
    images_dir: Path,
    masks_dir: Path,
    metadata_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Read PlantSeg labels from ``Metadatav2.csv``; emit the canonical schema.

    PlantSeg v5 ships with a ``plantsegv3/Metadatav2.csv`` mapping each image
    filename to (Plant, Disease, Split, Label file). We consume that CSV directly
    rather than re-parsing class names out of the reshape script's per-class
    symlink tree — one source of truth for labels.

    The returned ``species`` and ``disease`` columns are normalised via
    ``_norm_species`` / ``_norm_disease`` so they join cleanly against
    ``parse_plantvillage_metadata`` in ``build_class_alignment``.

    Rows are filtered down to those whose annotation PNG exists on disk inside
    ``masks_dir`` (split-aware lookup under ``masks_dir/plantsegv3/annotations``;
    falls back to the per-class symlinked layout ``masks_dir/<class_name>/<stem>.png``).

    Parameters
    ----------
    images_dir : Path
        The per-class symlink root (``<out>/images``). Used to derive the image
        ``filepath`` column as ``images_dir/<class_name>/<Name>``.
    masks_dir : Path
        The per-class symlink root (``<out>/masks``). Used to resolve each row's
        mask path; rows whose mask is missing are dropped.
    metadata_csv : Path, optional
        Explicit CSV path. Defaults to
        ``images_dir.parent / "plantsegv3" / "Metadatav2.csv"`` (i.e. whatever the
        reshape script extracted alongside ``images/`` and ``masks/``).

    Returns
    -------
    pd.DataFrame
        Columns: image_id, filepath, mask_filepath, species, disease, class_name,
        has_segmentation_mask.
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    if metadata_csv is None:
        metadata_csv = images_dir.parent / "plantsegv3" / "Metadatav2.csv"
    metadata_csv = Path(metadata_csv)

    if not metadata_csv.exists():
        # Fall back to the legacy per-class walk for synthetic test fixtures
        # and any caller that hasn't run the reshape script.
        logger.info(
            "parse_plantseg_metadata: no Metadatav2.csv at %s; falling back to "
            "per-class directory walk of %s", metadata_csv, images_dir,
        )
        return _parse_plantseg_from_tree(images_dir, masks_dir)

    meta = pd.read_csv(metadata_csv)
    required = {"Name", "Plant", "Disease", "Label file"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(
            f"parse_plantseg_metadata: Metadatav2.csv missing required columns {sorted(missing)}; "
            f"found {sorted(meta.columns)}"
        )

    records = []
    for _, row in meta.iterrows():
        plant = str(row["Plant"])
        disease_raw = str(row["Disease"])
        species = _norm_species(plant)
        disease = _norm_disease(disease_raw)
        class_name = f"{species}_{disease}"

        name = str(row["Name"])
        stem = Path(name).stem

        img_path = images_dir / class_name / name
        mask_path = masks_dir / class_name / f"{stem}.png"
        # Accept alternative mask extensions too, in case the reshape script ever
        # links masks through with a non-.png suffix.
        if not mask_path.exists():
            alt = None
            for ext in _IMAGE_EXTENSIONS:
                cand = masks_dir / class_name / f"{stem}{ext}"
                if cand.exists():
                    alt = cand
                    break
            mask_path = alt if alt is not None else mask_path

        # Rule 2: filter rows where the annotation PNG doesn't exist on disk.
        if not mask_path.exists():
            continue

        records.append(
            {
                "image_id": f"{class_name}/{name}",
                "filepath": str(img_path),
                "mask_filepath": str(mask_path),
                "species": species,
                "disease": disease,
                "class_name": class_name,
                "has_segmentation_mask": True,
            }
        )

    df = pd.DataFrame(
        records,
        columns=[
            "image_id",
            "filepath",
            "mask_filepath",
            "species",
            "disease",
            "class_name",
            "has_segmentation_mask",
        ],
    )
    df["has_segmentation_mask"] = df["has_segmentation_mask"].astype(bool)

    save_dir = images_dir.parent
    if save_dir.exists() and _is_writable(save_dir):
        try:
            df.to_csv(save_dir / "plantseg_labels.csv", index=False)
        except OSError as exc:
            logger.warning("Could not save plantseg_labels.csv: %s", exc)

    return df


def _parse_plantseg_from_tree(images_dir: Path, masks_dir: Path) -> pd.DataFrame:
    """Legacy tree-walk fallback used by tests that build per-class dirs by hand.

    Only used when ``Metadatav2.csv`` is absent. Applies the same
    ``_norm_species`` / ``_norm_disease`` normalisation as the CSV path so
    fixture data and real data share the downstream schema.
    """
    records = []
    for img_path in sorted(images_dir.rglob("*")):
        if img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue

        rel = img_path.relative_to(images_dir)
        image_id = str(rel)

        class_name_raw = rel.parts[0] if len(rel.parts) > 1 else img_path.stem
        # Split on the FIRST underscore so multi-word diseases stay intact.
        if "_" in class_name_raw:
            species_raw, disease_raw = class_name_raw.split("_", 1)
        else:
            species_raw = class_name_raw
            disease_raw = "unknown"
        species = _norm_species(species_raw)
        disease = _norm_disease(disease_raw)
        class_name = class_name_raw  # preserve on-disk dir name for path joins

        mask_filepath: Optional[str] = None
        mask_base = masks_dir / rel.parent / rel.stem
        for ext in _IMAGE_EXTENSIONS:
            candidate = mask_base.with_suffix(ext)
            if candidate.exists():
                mask_filepath = str(candidate)
                break

        records.append(
            {
                "image_id": image_id,
                "filepath": str(img_path),
                "mask_filepath": mask_filepath,
                "species": species,
                "disease": disease,
                "class_name": class_name,
                "has_segmentation_mask": mask_filepath is not None,
            }
        )

    df = pd.DataFrame(
        records,
        columns=[
            "image_id",
            "filepath",
            "mask_filepath",
            "species",
            "disease",
            "class_name",
            "has_segmentation_mask",
        ],
    )
    df["has_segmentation_mask"] = df["has_segmentation_mask"].astype(bool)
    return df


# ---------------------------------------------------------------------------
# 2. parse_plantvillage_metadata
# ---------------------------------------------------------------------------

def parse_plantvillage_metadata(images_dir: Path) -> pd.DataFrame:
    """Walk PlantVillage color/ directory, parse class dirs named Tomato___Early_blight.

    Parameters
    ----------
    images_dir : Path
        Root directory containing per-class subdirectories of images.

    Returns
    -------
    pd.DataFrame
        Same schema as parse_plantseg_metadata (mask_filepath=None always,
        has_segmentation_mask=False always).
    """
    images_dir = Path(images_dir)

    records = []
    for img_path in sorted(images_dir.rglob("*")):
        if img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue

        rel = img_path.relative_to(images_dir)
        image_id = str(rel)

        class_name = rel.parts[0] if len(rel.parts) > 1 else img_path.stem

        # PlantVillage uses triple-underscore separator
        if "___" in class_name:
            species_raw, disease_raw = class_name.split("___", 1)
        elif "_" in class_name:
            species_raw, disease_raw = class_name.split("_", 1)
        else:
            species_raw = class_name
            disease_raw = "unknown"
        species = _norm_species(species_raw)
        disease = _norm_disease(disease_raw)

        records.append(
            {
                "image_id": image_id,
                "filepath": str(img_path),
                "mask_filepath": None,
                "species": species,
                "disease": disease,
                "class_name": class_name,
                "has_segmentation_mask": False,
            }
        )

    df = pd.DataFrame(
        records,
        columns=[
            "image_id",
            "filepath",
            "mask_filepath",
            "species",
            "disease",
            "class_name",
            "has_segmentation_mask",
        ],
    )
    df["has_segmentation_mask"] = df["has_segmentation_mask"].astype(bool)
    return df


# ---------------------------------------------------------------------------
# 3. stratified_sample
# ---------------------------------------------------------------------------

def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Per-class stratified random sample of *n* total rows.

    Allocation strategy (multi-pass):
    1. Compute the average budget per undecided class at each pass:
       ``avg_budget = remaining_budget / n_undecided``.
    2. Any class whose available row count is at or below ``avg_budget`` is
       a "minority class" and is taken whole; its rows are removed from both
       the budget and the undecided pool before the next pass.
    3. Repeat until no new minority classes are found; then allocate the
       remaining budget proportionally among the uncapped classes.

    This ensures that small classes always contribute all of their rows,
    even when their exact proportional share would round to fewer than their
    count.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``class_name`` column.
    n : int
        Total target sample size.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Sampled rows with a reset index.
    """
    if "class_name" not in df.columns:
        raise ValueError("DataFrame must contain a 'class_name' column.")

    rng = np.random.default_rng(seed)
    class_counts = df["class_name"].value_counts().to_dict()
    classes = list(class_counts.keys())

    # Multi-pass proportional allocation with minority protection.
    # A class is "minority" at any pass if its available row count is less than
    # the average budget per undecided class (remaining_budget / n_undecided).
    # Minority classes are taken whole; the remainder re-distributes among the rest.
    remaining_budget = n
    alloc: dict[str, int] = {}
    undecided = list(classes)

    while undecided:
        next_undecided = []
        minority_found = False
        n_undecided = len(undecided)
        avg_budget = remaining_budget / n_undecided if n_undecided else 0

        for cls in undecided:
            cnt = class_counts[cls]
            if cnt <= avg_budget:
                # Minority class: take all rows
                alloc[cls] = cnt
                remaining_budget -= cnt
                minority_found = True
            else:
                next_undecided.append(cls)

        if not minority_found:
            # No more minorities — distribute remaining budget proportionally
            remaining_population = sum(class_counts[cls] for cls in next_undecided)
            for i, cls in enumerate(next_undecided):
                cnt = class_counts[cls]
                if i < len(next_undecided) - 1:
                    if remaining_population > 0:
                        share = remaining_budget * cnt / remaining_population
                        allocated = min(max(1, int(round(share))), cnt)
                    else:
                        allocated = 0
                else:
                    # Last class gets whatever is left
                    allocated = min(remaining_budget, cnt)
                alloc[cls] = allocated
                remaining_budget -= allocated
                remaining_population -= cnt
            break

        undecided = next_undecided

    # Sample each class according to its allocation
    sampled_parts = []
    for cls in classes:
        group = df[df["class_name"] == cls]
        k = alloc.get(cls, 0)
        if k <= 0:
            continue
        if k >= len(group):
            sampled_parts.append(group)
        else:
            idx = rng.choice(len(group), size=k, replace=False)
            sampled_parts.append(group.iloc[sorted(idx)])

    if not sampled_parts:
        return df.head(0).reset_index(drop=True)

    result = pd.concat(sampled_parts, ignore_index=True)

    # Trim if over budget (rounding accumulation can produce slightly > n)
    if len(result) > n:
        keep_idx = rng.choice(len(result), size=n, replace=False)
        result = result.iloc[sorted(keep_idx)].reset_index(drop=True)

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. build_class_alignment
# ---------------------------------------------------------------------------

def build_class_alignment(
    plantseg_df: pd.DataFrame,
    plantvillage_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, int]:
    """Find overlapping (species, disease) pairs between the two datasets.

    Matching is case-insensitive on both species and disease strings.

    Parameters
    ----------
    plantseg_df : pd.DataFrame
    plantvillage_df : pd.DataFrame
    save_path : Path, optional
        If provided, write the alignment CSV with a ``# n_overlap=N`` header
        comment as the very first line (pandas does not support this natively).

    Returns
    -------
    tuple[pd.DataFrame, int]
        alignment_df has columns: plantseg_class, plantvillage_class, species,
        disease, n_plantseg, n_plantvillage.
        The int is n_overlap (count of distinct matched (species, disease) pairs).
    """
    # Build lower-cased keys. The parsers already normalise species/disease
    # via _norm_species / _norm_disease, but belt-and-suspenders here so callers
    # that pass in legacy DataFrames (e.g. test fixtures) still converge.
    ps = plantseg_df.copy()
    pv = plantvillage_df.copy()

    ps["_species_lc"] = ps["species"].astype(str).str.lower()
    ps["_disease_lc"] = ps["disease"].astype(str).str.lower()
    pv["_species_lc"] = pv["species"].astype(str).str.lower()
    pv["_disease_lc"] = pv["disease"].astype(str).str.lower()

    # Symmetric prefix-strip: PlantSeg disease strings typically repeat the
    # species prefix ("apple black rot" for Plant=Apple → species="apple",
    # disease="apple_black_rot"). PlantVillage usually does not. Strip the
    # redundant prefix on both sides so the inner-join on (species, disease)
    # finds the real overlaps. Handles multi-word species where the normalised
    # species collapses spaces ("bellpepper") but the normalised disease does
    # not ("bell_pepper_bacterial_spot") by walking disease tokens until their
    # concatenation equals the species string.
    def _strip_species_prefix(species: str, disease: str) -> str:
        if not species or not disease:
            return disease
        tokens = disease.split("_")
        acc = ""
        for i, tok in enumerate(tokens):
            acc += tok
            if acc == species:
                return "_".join(tokens[i + 1:])
            if not species.startswith(acc):
                break
        return disease

    ps["_disease_stripped"] = [
        _strip_species_prefix(s, d)
        for s, d in zip(ps["_species_lc"], ps["_disease_lc"])
    ]
    pv["_disease_stripped"] = [
        _strip_species_prefix(s, d)
        for s, d in zip(pv["_species_lc"], pv["_disease_lc"])
    ]

    # Group by (species_lc, disease_stripped) to get canonical class names + counts
    ps_grp = (
        ps.groupby(["_species_lc", "_disease_stripped"])
        .agg(
            plantseg_class=("class_name", "first"),
            species_display=("species", "first"),
            n_plantseg=("image_id", "count"),
        )
        .reset_index()
    )
    pv_grp = (
        pv.groupby(["_species_lc", "_disease_stripped"])
        .agg(
            plantvillage_class=("class_name", "first"),
            n_plantvillage=("image_id", "count"),
        )
        .reset_index()
    )

    merged = ps_grp.merge(
        pv_grp, on=["_species_lc", "_disease_stripped"], how="inner"
    )
    n_overlap = len(merged)

    # Emit normalised species / disease (post-strip) as the canonical output
    # columns — downstream consumers (transfer_correlation, etc.) pair them
    # against the parsed labels which also carry normalised values.
    merged = merged.rename(
        columns={"_species_lc": "species", "_disease_stripped": "disease"}
    )
    alignment_df = merged[
        [
            "plantseg_class",
            "plantvillage_class",
            "species",
            "disease",
            "n_plantseg",
            "n_plantvillage",
        ]
    ].copy()

    if save_path is not None:
        _save_class_alignment_csv(alignment_df, n_overlap, Path(save_path))

    return alignment_df, n_overlap


def _save_class_alignment_csv(alignment_df: pd.DataFrame, n_overlap: int, path: Path) -> None:
    """Write alignment CSV with ``# n_overlap=N`` as the very first line."""
    with open(path, "w") as fh:
        fh.write(f"# n_overlap={n_overlap}\n")
        alignment_df.to_csv(fh, index=False)


# ---------------------------------------------------------------------------
# 5. make_image_dataset
# ---------------------------------------------------------------------------

_DINOV2_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

_DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        _DINOV2_NORMALIZE,
    ]
)


class _ImageDataset(Dataset):
    """Internal Dataset implementation returned by make_image_dataset."""

    def __init__(self, df: pd.DataFrame, transform: transforms.Compose) -> None:
        self._df = df.reset_index(drop=True)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int):  # noqa: ANN001
        row = self._df.iloc[idx]
        image_id = str(row["image_id"])
        filepath = str(row["filepath"])

        try:
            img = Image.open(filepath).convert("RGB")
            tensor = self._transform(img)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load image %s: %s — returning zero tensor.", filepath, exc)
            tensor = torch.zeros(3, 224, 224)

        return tensor, image_id


def make_image_dataset(df: pd.DataFrame, image_root: Path) -> Dataset:
    """Return a Dataset that yields (3×224×224 tensor, image_id_str) tuples.

    ``image_root`` is accepted for reference but filepaths in *df* are already
    absolute so it is not used to resolve paths.

    Missing files are handled gracefully: a zero tensor is returned and a
    warning is logged.
    """
    return _ImageDataset(df, _DEFAULT_TRANSFORM)


# ---------------------------------------------------------------------------
# 6. resize_mask_to_patch_grid
# ---------------------------------------------------------------------------

def resize_mask_to_patch_grid(mask: np.ndarray) -> np.ndarray:
    """Convert a per-pixel boolean mask to a 16×16 patch-level boolean mask.

    A cell in the output is True if >50% of the corresponding pixel block is
    True (strict threshold — exactly 50% yields False).

    If H or W are not exactly divisible by 16, the mask is first resized to the
    nearest larger multiple of 16 using PIL nearest-neighbour interpolation.

    Parameters
    ----------
    mask : np.ndarray
        Shape (H, W), dtype bool (or any numeric type castable to float).

    Returns
    -------
    np.ndarray
        Shape (16, 16), dtype bool.
    """
    mask = np.asarray(mask, dtype=np.float32)
    H, W = mask.shape

    # Pad to multiples of 16 if needed
    target_H = H if H % 16 == 0 else (H // 16 + 1) * 16
    target_W = W if W % 16 == 0 else (W // 16 + 1) * 16

    if target_H != H or target_W != W:
        pil_mask = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        pil_mask = pil_mask.resize((target_W, target_H), resample=Image.NEAREST)
        mask = np.array(pil_mask, dtype=np.float32) / 255.0
        H, W = target_H, target_W

    patch_h = H // 16
    patch_w = W // 16

    # Block-mean reduction: reshape to (16, patch_h, 16, patch_w)
    block = mask.reshape(16, patch_h, 16, patch_w)
    # Mean over (patch_h, patch_w) axes → (16, 16)
    cell_mean = block.mean(axis=(1, 3))

    return (cell_mean > 0.5).astype(bool)


# ---------------------------------------------------------------------------
# 7. save_masks_tensor
# ---------------------------------------------------------------------------

def save_masks_tensor(
    mask_paths: list[Path],
    output_path: Path,
    image_ids: list[str] | None = None,
) -> None:
    """Load each mask, reduce to 16x16, stack, and save as a torch dict.

    Parameters
    ----------
    mask_paths : list[Path]
        Paths to grayscale (or binary) mask images.
    output_path : Path
        Destination .pt file.
    image_ids : list[str] | None
        Canonical image IDs (matching the ``image_id`` column of
        ``plantseg_df``, which is ``str(rel)`` from ``parse_plantseg_metadata``,
        e.g. ``"Tomato_Blight/img1.jpg"``). MUST be the same length as
        ``mask_paths``. If None (legacy), falls back to ``p.stem`` and emits a
        warning -- downstream joins against ``plantseg_df['image_id']`` will
        not match, so the IoU evaluation will silently run on zero images.
    """
    if image_ids is not None:
        assert len(image_ids) == len(mask_paths), (
            f"image_ids length {len(image_ids)} != mask_paths length {len(mask_paths)}"
        )
    else:
        logger.warning(
            "save_masks_tensor called without image_ids; using p.stem fallback. "
            "Downstream joins against plantseg_df['image_id'] will fail because "
            "plantseg_df uses str(rel) (e.g. 'Tomato_Blight/img1.jpg') while "
            "p.stem only returns 'img1'. Pass image_ids explicitly."
        )

    grids = []
    out_image_ids: list[str] = []

    for i, p in enumerate(mask_paths):
        p = Path(p)
        try:
            img = Image.open(p).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
            grid = resize_mask_to_patch_grid(arr > 0.5)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load mask %s: %s -- using all-False grid.", p, exc)
            grid = np.zeros((16, 16), dtype=bool)

        grids.append(grid)
        out_image_ids.append(image_ids[i] if image_ids is not None else p.stem)

    masks_tensor = torch.tensor(np.stack(grids, axis=0), dtype=torch.bool)
    torch.save({"masks": masks_tensor, "image_ids": out_image_ids}, output_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_writable(path: Path) -> bool:
    """Return True if *path* is a writable directory."""
    import os

    return os.access(path, os.W_OK)
