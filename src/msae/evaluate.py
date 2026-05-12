from __future__ import annotations
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 0. Batched encoding helpers
# ---------------------------------------------------------------------------

def _chunked_encode(
    encode_fn: Callable[[Tensor], Tensor],
    acts: Tensor,
    chunk_size: int,
    device: torch.device,
    store_dtype: torch.dtype,
    show_progress: bool = True,
) -> Tensor:
    """Stream `acts` to `device` in row chunks, call ``encode_fn`` under
    inference_mode (+ bf16 autocast on CUDA), cast each chunk to ``store_dtype``
    on host, and return the concatenated CPU tensor.

    Private helper so :func:`encode_batched` and :func:`transfer_correlation`
    (Phase B) share the same numerics and memory discipline.
    """
    n = acts.shape[0]
    source_on_device = acts.device == device

    try:  # optional progress bar
        from tqdm.auto import tqdm  # type: ignore
        iterator = tqdm(range(0, n, chunk_size), disable=not show_progress, desc="encode")
    except Exception:
        iterator = range(0, n, chunk_size)

    chunks: list[Tensor] = []
    use_cuda_autocast = device.type == "cuda"

    with torch.inference_mode():
        for start in iterator:
            end = min(start + chunk_size, n)
            x = acts[start:end]
            if not source_on_device:
                x = x.to(device, non_blocking=True)
            if use_cuda_autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = encode_fn(x)
            else:
                z = encode_fn(x)
            # Cast + move to host *before* the next iteration so the GPU buffer
            # is released promptly.
            chunks.append(z.to(store_dtype).cpu())
            del z, x

    out = torch.cat(chunks, dim=0)
    del chunks
    return out


def encode_batched(
    model: nn.Module,
    acts: Tensor,
    batch_size: int = 16384,
    device: str | torch.device | None = None,
    store_dtype: torch.dtype = torch.bfloat16,
    show_progress: bool = True,
) -> Tensor:
    """Run ``model.encode()`` in batches; return concatenated output on CPU.

    Streams ``acts`` to GPU ``batch_size`` rows at a time, runs ``encode`` under
    ``inference_mode`` + bf16 autocast (when device is CUDA), casts each output
    chunk to ``store_dtype``, and concatenates on the host.

    Returns a tensor of shape ``(N, model.max_features_or_n_features)`` on CPU
    in the requested dtype.

    Raises
    ------
    RuntimeError
        If ``model`` does not expose an ``encode`` method.
    """
    if not hasattr(model, "encode"):
        raise RuntimeError(
            f"encode_batched: model of type {type(model).__name__} has no .encode(x) method. "
            "Expected MatryoshkaSAE or StandardSAE."
        )

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    return _chunked_encode(
        encode_fn=model.encode,
        acts=acts,
        chunk_size=batch_size,
        device=device,
        store_dtype=store_dtype,
        show_progress=show_progress,
    )


def encode_sparse(
    model: nn.Module,
    acts: Tensor,
    batch_size: int = 16384,
    device: str | torch.device | None = None,
    show_progress: bool = True,
):
    """Chunk-encode a dataset and return a ``scipy.sparse.csr_matrix`` on CPU.

    MSAE / StandardSAE encoder outputs are ReLU-sparse — at production dims
    (``max_features=12288``) on 3.3M PlantSeg patches, a dense fp32 view is
    ~162 GB and bf16 is ~81 GB, both OOM on a 32 GB host. At the trained L0
    target of 30–80 (``lambda_sparse_probe``), the matrix is ~99.5% zeros,
    so scipy CSR storage is ~1 GB.

    Downstream consumers (``class_selectivity``, ``compute_mi``,
    ``build_grid_acts_chunked``, ``transfer_correlation``, and the visualize
    top-patches slice) dispatch on ``scipy.sparse.issparse(z)`` and use the
    sparse path when it's set.

    Notes
    -----
    - We return fp32 (not bf16). scipy CSR values use numpy dtypes; numpy
      has no native bf16, and the downstream paths up-cast to fp32 anyway
      for the mass aggregation / IoU scatter / MI sklearn calls. Staying
      in fp32 avoids a second dtype conversion and keeps the R6 tolerances
      (``|Δ| <= 1e-6`` for selectivity) reachable.
    - This can't use ``torch.sparse_csr_tensor`` / ``torch.cat`` — torch
      2.11 CSR has no ``cat``, no ``.T``, no row/column slicing, no spmm.
      scipy supports all of these and round-trips bit-exact through the
      ``(crow_indices, col_indices, values)`` triplet.
    - Peak transient host memory is one dense chunk
      (``batch_size × n_features × 4`` bytes; ~800 MB at
      ``batch_size=16384``, ``n_features=12288``), released before the next
      iteration.
    """
    # scipy is already a core dep (sklearn pulls it in; pyproject.toml lists
    # scipy directly). Importing at call time keeps the module-level import
    # set minimal for non-sparse callers.
    import scipy.sparse as sp

    if not hasattr(model, "encode"):
        raise RuntimeError(
            f"encode_sparse: model of type {type(model).__name__} has no "
            ".encode(x) method. Expected MatryoshkaSAE or StandardSAE."
        )

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    n = acts.shape[0]
    source_on_device = acts.device == device
    use_cuda_autocast = device.type == "cuda"

    try:  # optional progress bar
        from tqdm.auto import tqdm  # type: ignore
        iterator = tqdm(
            range(0, n, batch_size), disable=not show_progress, desc="encode_sparse"
        )
    except Exception:
        iterator = range(0, n, batch_size)

    chunks: list[sp.csr_matrix] = []
    n_features: int | None = None

    with torch.inference_mode():
        for start in iterator:
            end = min(start + batch_size, n)
            x = acts[start:end]
            if not source_on_device:
                x = x.to(device, non_blocking=True)
            if use_cuda_autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = model.encode(x)
            else:
                z = model.encode(x)
            # Convert to fp32 CSR on CPU. We go via a dense chunk on CPU —
            # 805 MB transient per step at 16384×12288 — rather than
            # extracting nonzeros on the GPU, because scipy's csr_matrix(dense)
            # constructor is fast and correct, and the transient host
            # footprint is released immediately when the dense array goes
            # out of scope at the end of this iteration.
            z_np = z.float().cpu().numpy()
            chunk_csr = sp.csr_matrix(z_np)
            chunks.append(chunk_csr)
            if n_features is None:
                n_features = chunk_csr.shape[1]
            del z, z_np, x
            if use_cuda_autocast:
                torch.cuda.empty_cache()

    if not chunks:
        # Empty input. Produce an (0, n_features) CSR; n_features unknown
        # without running the encoder at least once — surface that.
        raise ValueError("encode_sparse: input acts has zero rows.")

    return sp.vstack(chunks, format="csr")


# ---------------------------------------------------------------------------
# 1. class_selectivity
# ---------------------------------------------------------------------------

def class_selectivity(
    feature_activations: Tensor,           # (n_patches, n_features) — activation values per patch
    labels: np.ndarray,                    # (n_patches,) int — class label per patch
    n_classes: int,
    healthy_class_ids: list[int] | None = None,
    patch_chunk: int = 262_144,
) -> pd.DataFrame:
    """Compute selectivity for each feature based on class entropy.

    Selectivity = 1 - H(f) / H_max, where H is the entropy of the activation-mass
    distribution across classes. A perfectly selective feature (fires only on one
    class) gets selectivity ~ 1. A uniform feature gets selectivity ~ 0.

    If ``healthy_class_ids`` is provided, all healthy classes are collapsed into
    a single merged class before computing entropy (spec line 597). This avoids
    artificially inflating disease selectivity via species x healthy cross
    products.

    Memory-bounded implementation (addresses A5-A6): the old code materialised
    ``feature_activations.T.float()`` at full grain -- ~150 GB host RAM at
    3.3M x 12288. This version keeps ``mass`` at ``(n_features, n_classes_eff)``
    (tiny; ~1.6 MB for 12288 x 35) and streams PATCHES through it in chunks of
    ``patch_chunk`` rows. A numpy lookup table replaces the slow
    ``np.vectorize(id_map.get)`` healthy remap.

    All work runs on ``feature_activations.device``; caller-side ``.cpu()`` is
    not required for GPU inputs.

    Returns
    -------
    pd.DataFrame with columns ``feature_id`` (int) and ``selectivity`` (float).
    """
    import scipy.sparse as sp
    is_sparse_input = sp.issparse(feature_activations)

    if is_sparse_input:
        n_patches, n_features = feature_activations.shape
    else:
        n_patches, n_features = feature_activations.shape

    # --- Healthy-class collapse via numpy LUT -------------------------------
    labels_remapped = labels
    if healthy_class_ids:
        non_healthy_ids = [i for i in range(n_classes) if i not in healthy_class_ids]
        id_map: dict[int, int] = {
            old: new for new, old in enumerate(sorted(non_healthy_ids))
        }
        merged_healthy_id = len(non_healthy_ids)
        for old_id in healthy_class_ids:
            id_map[old_id] = merged_healthy_id
        lut = np.arange(n_classes, dtype=np.int64)
        for old_id, new_id in id_map.items():
            lut[old_id] = new_id
        labels_remapped = lut[labels]
        n_classes_eff = len(non_healthy_ids) + 1
    else:
        n_classes_eff = n_classes

    if n_classes_eff <= 1:
        logger.warning(
            "class_selectivity: n_classes_eff=%d <= 1; selectivity is trivially 1.0",
            n_classes_eff,
        )

    # --- Mass aggregation: mass[f, c] = sum over patches p of z[p,f] * 1{label[p]==c}
    if is_sparse_input:
        # Sparse fast path: mass = Z.T @ L_oh, computed in scipy. Avoids
        # materializing a (n_features, chunk) dense transpose slice that
        # would otherwise force O(n_patches × n_features) fp32 memory.
        # L_oh is dense (n_patches × n_classes_eff), typically ~500 MB at
        # 3.3M × 35, and scipy's sparse-dense matmul handles it cleanly.
        L_oh = np.zeros(
            (n_patches, n_classes_eff), dtype=np.float32
        )
        L_oh[np.arange(n_patches, dtype=np.int64), labels_remapped] = 1.0
        # scipy returns np.ndarray (not np.matrix) for csr_matrix @ dense.
        mass_np = feature_activations.T @ L_oh  # (n_features, n_classes_eff)
        mass = torch.from_numpy(np.ascontiguousarray(mass_np, dtype=np.float32))
        # device is CPU for the sparse path (scipy is always host-side).
    else:
        device = feature_activations.device
        labels_t = torch.from_numpy(
            np.asarray(labels_remapped, dtype=np.int64)
        ).long().to(device)

        mass = torch.zeros(n_features, n_classes_eff, dtype=torch.float32, device=device)

        # Stream patches through the scatter_add. Within-chunk transpose is
        # ``(n_features, chunk)`` fp32 -- at chunk=262144 this is ~13 GB for 12288
        # features; drop ``patch_chunk`` if that's too much. For bf16 inputs the
        # ``.float()`` cast is a chunk-local temporary, not a global 150 GB copy.
        for p_start in range(0, n_patches, patch_chunk):
            p_end = min(p_start + patch_chunk, n_patches)
            chunk_labels = labels_t[p_start:p_end]                     # (chunk,)
            chunk_acts = feature_activations[p_start:p_end].T.float()  # (n_features, chunk)
            mass.scatter_add_(
                1,
                chunk_labels.unsqueeze(0).expand(n_features, -1),
                chunk_acts,
            )

    # --- Entropy per feature -----------------------------------------------
    p = mass / (mass.sum(dim=1, keepdim=True) + 1e-10)
    entropy = -(p * (p + 1e-10).log()).sum(dim=1)

    h_max = float(np.log(n_classes_eff)) if n_classes_eff > 1 else 1.0
    selectivity = (1.0 - entropy / h_max).clamp(0.0, 1.0)

    return pd.DataFrame({
        "feature_id": np.arange(n_features, dtype=int),
        "selectivity": selectivity.detach().cpu().numpy().astype(float),
    })


# ---------------------------------------------------------------------------
# 2. compute_iou_vectorized
# ---------------------------------------------------------------------------

def compute_iou_vectorized(
    feature_acts_per_image_grid: Tensor,   # (n_features, n_images, 16, 16) float
    masks: Tensor,                          # (n_images, 16, 16) bool
    quantile: float = 0.90,
    feature_chunk: int = 1024,
) -> Tensor:
    """Compute per-feature mean IoU against segmentation masks.

    For each feature chunk, threshold activations at the feature's 90th-percentile
    value (over all images × spatial positions), binarise, then compute IoU against
    the provided boolean masks.  Processes ``feature_chunk`` features at a time to
    bound peak memory.

    Returns
    -------
    Tensor of shape ``(n_features,)`` — mean IoU per feature.
    """
    n_features, n_images, H, W = feature_acts_per_image_grid.shape
    masks_expanded = masks.unsqueeze(0).bool()  # (1, n_images, H, W)

    results: list[Tensor] = []
    for start in range(0, n_features, feature_chunk):
        end = min(start + feature_chunk, n_features)
        acts_chunk = feature_acts_per_image_grid[start:end].float()  # (chunk, n_images, H, W)
        chunk_size = acts_chunk.shape[0]

        # Per-feature quantile threshold — flatten (n_images, H, W) per feature
        flat = acts_chunk.reshape(chunk_size, -1)           # (chunk, n_images*H*W)
        threshold = torch.quantile(flat, quantile, dim=1)  # (chunk,)
        threshold = threshold.view(chunk_size, 1, 1, 1)     # broadcast-ready

        pred = (acts_chunk >= threshold)  # (chunk, n_images, H, W) bool

        intersection = (pred & masks_expanded).float().sum(dim=(-2, -1))   # (chunk, n_images)
        union = (pred | masks_expanded).float().sum(dim=(-2, -1))          # (chunk, n_images)
        iou = intersection / (union + 1e-6)                                 # (chunk, n_images)
        mean_iou_chunk = iou.mean(dim=1)                                    # (chunk,)

        # Dead features: the per-feature 90th percentile threshold is 0 because
        # all activations are 0; (acts_chunk >= 0) then fires everywhere,
        # producing a spurious IoU = mask_coverage / 256. Force IoU=0.
        # ``build_grid_acts_chunked`` also zeros dead features before returning,
        # so this guard is idempotent when chained from that helper.
        dead_in_chunk = acts_chunk.amax(dim=(-3, -2, -1)) <= 0
        mean_iou_chunk = torch.where(
            dead_in_chunk, torch.zeros_like(mean_iou_chunk), mean_iou_chunk,
        )
        results.append(mean_iou_chunk)

    return torch.cat(results, dim=0)  # (n_features,)


# ---------------------------------------------------------------------------
# 2b. build_grid_acts_chunked
# ---------------------------------------------------------------------------

def build_grid_acts_chunked(
    z_all: Tensor,
    meta: Tensor,
    masked_img_indices: list[int] | Tensor,
    masks: Tensor,
    feature_chunk: int = 512,
    quantile: float = 0.90,
    device: str | torch.device | None = None,
) -> Tensor:
    """Compute per-feature mean IoU against masks without materialising the full grid.

    The naive approach allocates ``(n_features, n_masked, 16, 16)`` at once —
    ~144 GB at ``n_features=12288``. This function streams through features in
    chunks of ``feature_chunk`` instead: for each chunk, it scatters the chunk's
    encoded activations into a small ``(chunk, n_masked, 16, 16)`` grid, calls
    :func:`compute_iou_vectorized`, zeros out dead features, then frees the
    chunk buffer before moving on.

    Parameters
    ----------
    z_all
        ``(n_patches, n_features)`` encoded activations. Any dtype, CPU or GPU.
    meta
        ``(n_patches, 3)`` int tensor with columns ``[image_id, row, col]``.
        ``image_id`` is the global image index.
    masked_img_indices
        Global image indices to keep. Length must match ``masks.shape[0]``.
    masks
        ``(len(masked_img_indices), 16, 16)`` boolean tensor.
    feature_chunk
        Number of features per chunk.
    quantile
        Per-feature threshold quantile for IoU (forwarded to
        :func:`compute_iou_vectorized`).
    device
        Target device for the scatter/IoU math. Defaults to ``z_all``'s device,
        falling back to CUDA when available, else CPU.

    Returns
    -------
    Tensor of shape ``(n_features,)`` on CPU — mean IoU per feature. Features
    that are dead (``amax == 0`` over all kept patches) are returned as ``0.0``.
    """
    if device is None:
        if isinstance(z_all, Tensor) and z_all.is_cuda:
            device = z_all.device
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    import scipy.sparse as sp
    is_sparse_input = sp.issparse(z_all)

    # --- Normalise inputs --------------------------------------------------
    if isinstance(masked_img_indices, list):
        masked_idx_t = torch.tensor(masked_img_indices, dtype=torch.long)
    else:
        masked_idx_t = masked_img_indices.long().cpu()
    n_masked = int(masked_idx_t.shape[0])
    if masks.shape[0] != n_masked:
        raise ValueError(
            f"masks first dim ({masks.shape[0]}) must match len(masked_img_indices) ({n_masked})."
        )

    meta_cpu = meta.cpu()
    image_ids_all = meta_cpu[:, 0].long()

    # Patches we actually care about (image in the kept subset).
    keep = torch.isin(image_ids_all, masked_idx_t)
    if not bool(keep.any()):
        n_features = int(z_all.shape[1])
        return torch.zeros(n_features, dtype=torch.float32)

    meta_kept = meta_cpu[keep]
    if is_sparse_input:
        # Convert CSR → CSC once for the column-slice hot loop. Benchmarked on
        # a synthetic 3.3M × 12288 at density 0.005 (production-scale proxy):
        # CSR column slice averages 4.7 s per 512-feature chunk (24 × 4.7 s
        # = ~113 s total) because column slicing on CSR is O(nnz) per row.
        # CSC column slice is ~4.3 s per chunk; conversion is a one-off 2.5 s,
        # and the net saving is ~8 s end-to-end — modest but deterministic,
        # and the decision rule (chunk > 3 s triggers conversion) is met.
        # We do the conversion only here, not globally in encode_sparse,
        # because other consumers (class_selectivity via Z.T @ L_oh,
        # compute_mi row slice, transfer_correlation advanced row indexing)
        # prefer CSR. Transient peak during conversion is ~3.2 GB (CSR +
        # new CSC both live briefly); fits on the 32 GB host.
        # scipy row-subset via np index. `z_all_kept` stays a CSC matrix on
        # the host — we only materialize dense one feature-chunk at a time.
        keep_np = keep.cpu().numpy()
        z_all_kept = z_all[keep_np].tocsc()
    else:
        z_all_kept = z_all[keep]  # (n_kept, n_features), stays on its original device

    # Build a global-image-id -> subset-index lookup via a dense table.
    max_img_id = int(masked_idx_t.max().item())
    lookup = torch.full((max_img_id + 1,), -1, dtype=torch.long)
    lookup[masked_idx_t] = torch.arange(n_masked, dtype=torch.long)

    subset_idx = lookup[meta_kept[:, 0].long()]   # (n_kept,)
    row = meta_kept[:, 1].long()
    col = meta_kept[:, 2].long()
    # Flat index into a (n_masked, 16, 16) grid.
    flat_idx = (subset_idx * 256 + row * 16 + col).to(device)

    masks_dev = masks.to(device).bool()
    n_features = int(z_all_kept.shape[1])
    n_kept = int(z_all_kept.shape[0])

    results: list[Tensor] = []
    for f_start in range(0, n_features, feature_chunk):
        f_end = min(f_start + feature_chunk, n_features)
        chunk_size = f_end - f_start

        # z values for this chunk, shape (chunk_size, n_kept) on device, fp32.
        if is_sparse_input:
            # scipy column slice → dense → torch → device. The transient
            # fp32 array is (n_kept, chunk_size): at 3.3M × 512 that's
            # 6.6 GB on the host; drop ``feature_chunk`` if the host runs
            # tight. For sparse input the ``.T`` happens after the dense
            # materialization.
            col_slice = z_all_kept[:, f_start:f_end].toarray()  # (n_kept, chunk_size)
            z_chunk = (
                torch.from_numpy(col_slice)
                .to(device=device, dtype=torch.float32)
                .T.contiguous()
            )
            del col_slice
        else:
            z_chunk = z_all_kept[:, f_start:f_end].to(device=device, dtype=torch.float32).T.contiguous()

        grid = torch.zeros(chunk_size, n_masked, 16, 16, device=device, dtype=torch.float32)
        grid_flat = grid.view(chunk_size, -1)  # (chunk, n_masked*256)

        # index and src both shape (chunk_size, n_kept) for scatter_add_ on dim=1.
        index = flat_idx.unsqueeze(0).expand(chunk_size, n_kept)
        grid_flat.scatter_add_(1, index, z_chunk)

        # Dead mask: no activation anywhere in the kept subset.
        dead = grid.amax(dim=(-3, -2, -1)) == 0

        iou_chunk = compute_iou_vectorized(
            grid, masks_dev, quantile=quantile, feature_chunk=chunk_size
        )
        iou_chunk[dead] = 0.0

        results.append(iou_chunk.cpu())
        del grid, grid_flat, z_chunk, index
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(results, dim=0)


# ---------------------------------------------------------------------------
# 3. compute_mi
# ---------------------------------------------------------------------------

def compute_mi(
    feature_acts: Tensor,       # (n_patches, n_features) — use a float32 view
    labels: np.ndarray,         # (n_patches,) int
    estimator: str = "knn",     # ALWAYS use 'knn' as default -- do not change
    max_samples: int = 50_000,
    seed: int = 42,
) -> np.ndarray:
    """Compute mutual information between each feature and class labels.

    At notebook scale (3.3M patches x 12288 features) sklearn's kNN MI is
    multi-day CPU -- infeasible within a Colab session. This implementation
    sub-samples to ``max_samples`` rows STRATIFIED by label so class proportions
    are preserved. Sub-sampling preserves the species>disease MI ordering that
    Eval 6 depends on; the absolute MI values shrink with N.

    Also adds the ``.cpu()`` that was missing before ``.numpy()``, so GPU
    inputs work without a confusing TypeError.

    Parameters
    ----------
    feature_acts : Tensor of shape (n_patches, n_features)
    labels       : np.ndarray of shape (n_patches,)
    estimator    : 'knn' (default) or 'discrete_decile'
    max_samples  : cap on rows passed to sklearn
    seed         : RNG seed for the sub-sample

    Returns
    -------
    np.ndarray of shape (n_features,) -- MI value per feature.
    """
    logger.info(
        "compute_mi: estimator=%s  max_samples=%d  input_rows=%d",
        estimator, max_samples, feature_acts.shape[0],
    )

    import scipy.sparse as sp
    is_sparse_input = sp.issparse(feature_acts)
    n_rows = feature_acts.shape[0]

    if n_rows > max_samples:
        rng = np.random.default_rng(seed)
        unique_labels, counts = np.unique(labels, return_counts=True)
        # Proportional allocation with at least 1 per class.
        per_class = np.maximum(
            1, np.round(counts / counts.sum() * max_samples).astype(int)
        )
        while per_class.sum() > max_samples:
            per_class[per_class.argmax()] -= 1
        sample_idx_parts: list[np.ndarray] = []
        for lbl, n_take in zip(unique_labels, per_class):
            class_pool = np.where(labels == lbl)[0]
            n_take = min(int(n_take), len(class_pool))
            sample_idx_parts.append(rng.choice(class_pool, size=n_take, replace=False))
        sample_idx = np.concatenate(sample_idx_parts)
        feature_acts_sub = feature_acts[sample_idx]
        labels_sub = labels[sample_idx]
        logger.info(
            "compute_mi: sub-sampled to %d rows (stratified by label)",
            len(sample_idx),
        )
    else:
        feature_acts_sub = feature_acts
        labels_sub = labels

    # Materialize the (possibly subsampled) slice as a dense fp32 numpy array.
    # For the sparse path the slice is at most ~50k × n_features × 4 bytes
    # (≈2.5 GB at 12288 features) — well within 32 GB host RAM.
    if is_sparse_input:
        acts_np = feature_acts_sub.toarray().astype(np.float32, copy=False)
    else:
        acts_np = feature_acts_sub.detach().float().cpu().numpy()

    if estimator == "knn":
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(
            acts_np, labels_sub, discrete_features=False, random_state=seed,
        )
        return mi

    elif estimator == "discrete_decile":
        from sklearn.metrics import mutual_info_score

        n_patches, n_features = acts_np.shape
        mi = np.zeros(n_features, dtype=float)
        # Vectorise the quantile step across features; MI call stays per-feature
        # (sklearn's mutual_info_score is not vectorisable over columns).
        qs = np.quantile(acts_np, np.linspace(0, 1, 11), axis=0)  # (11, n_features)
        for f in range(n_features):
            col_q = np.unique(qs[1:-1, f])
            bin_labels = np.searchsorted(col_q, acts_np[:, f])
            mi[f] = mutual_info_score(labels_sub, bin_labels)
        return mi

    else:
        raise ValueError(
            f"Unknown estimator '{estimator}'. Choose 'knn' or 'discrete_decile'."
        )


# ---------------------------------------------------------------------------
# 4. transfer_correlation
# ---------------------------------------------------------------------------

def transfer_correlation(
    msae_encoder: nn.Module,
    plantseg_acts: Tensor,
    plantvillage_acts: Tensor,
    plantseg_labels: pd.DataFrame,
    plantvillage_labels: pd.DataFrame,
    class_alignment_csv: Path,
    encode_batch_size: int = 16384,
) -> dict:
    """Compute transfer correlation between PlantSeg and PlantVillage features.

    Reads ``n_overlap`` from the ``# n_overlap=N`` header of
    ``class_alignment_csv``; <8 returns a qualitative summary, >=8 computes
    per-class cosine similarity between mean encoded vectors.

    D1 fix: matching is now driven by the alignment CSV's row-level mapping
    between ``plantseg_class`` and ``plantvillage_class`` columns, not by a
    set-intersection over class-name strings. The old intersection silently
    returned mean_cosine_sim=0.0 whenever the two datasets' naming conventions
    diverged (PlantSeg ``Tomato_Early_Blight`` vs. PlantVillage
    ``Tomato___Early_blight`` never intersect as strings but do share
    ``(species, disease) = ("Tomato", "Early_Blight")``). An assertion fires
    if no matched class survives so the next regression here fails loud.

    Historical notes:

    1. Tuple-unpack (pre-existing fix): ``MatryoshkaSAE.forward()`` returns
       ``(z, recons_dict)`` so ``msae_encoder(plantseg_acts)`` was never a
       tensor. This function calls ``encode_batched`` which uses
       ``.encode()`` directly.
    2. Granularity (pre-existing fix): ``plantseg_acts`` is per-patch (3.3M
       rows) but callers historically passed per-image label DataFrames
       (19K rows). A length assertion catches that mismatch loudly. Pass
       per-patch labels via
       ``plantseg_df.iloc[meta[:, 0].numpy()].reset_index(drop=True)``.

    ``plantseg_acts`` / ``plantvillage_acts`` may be on CPU — they're streamed
    to GPU internally via ``encode_batched``.
    """
    class_alignment_csv = Path(class_alignment_csv)
    first_line = class_alignment_csv.read_text().splitlines()[0].strip()
    if not first_line.startswith("# n_overlap="):
        raise ValueError(
            f"{class_alignment_csv}: missing '# n_overlap=N' header "
            f"(got {first_line!r}). Re-run Stage 1 build_class_alignment."
        )
    n_overlap = int(first_line.split("=", 1)[1])

    if n_overlap < 8:
        return {"mode": "qualitative", "n_overlap": n_overlap, "matched_classes": []}

    # --- Encode both datasets sparsely (fp32 CSR on host). Full dense would
    # be ~162 GB at production dims — see encode_sparse docstring. All
    # downstream ops below use sparse-friendly row-indexing + mean(axis=0).
    ps_z = encode_sparse(
        msae_encoder, plantseg_acts,
        batch_size=encode_batch_size,
    )
    pv_z = encode_sparse(
        msae_encoder, plantvillage_acts,
        batch_size=encode_batch_size,
    )

    assert len(plantseg_labels) == ps_z.shape[0], (
        f"plantseg_labels has {len(plantseg_labels)} rows but ps_z has "
        f"{ps_z.shape[0]} -- labels must be PER-PATCH (build via "
        "plantseg_df.iloc[meta[:, 0].numpy()].reset_index(drop=True))"
    )
    assert len(plantvillage_labels) == pv_z.shape[0], (
        f"plantvillage_labels has {len(plantvillage_labels)} rows but pv_z "
        f"has {pv_z.shape[0]}"
    )

    ps_labels = plantseg_labels.reset_index(drop=True)
    pv_labels = plantvillage_labels.reset_index(drop=True)

    # scipy ``csr[idx].mean(axis=0)`` returns a dense ``np.matrix`` of shape
    # ``(1, n_features)`` on scipy 1.17 (the new sparse-array API is still
    # opt-in on that version, and csr_matrix predates it). ``np.asarray(...)``
    # coerces matrix → ndarray — critical, because downstream
    # ``F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))`` would otherwise
    # receive a 2-D (1, n_features) matrix interpreted as a batch of 1 and
    # silently compute the wrong shape. ``.ravel()`` gives the 1-D vector we
    # want.
    def _class_mean(z_sparse, idx_np: np.ndarray) -> Tensor:
        m = np.asarray(z_sparse[idx_np].mean(axis=0)).ravel()
        return torch.from_numpy(m.astype(np.float32, copy=False))

    ps_class_means: dict[str, Tensor] = {}
    for class_name, group in ps_labels.groupby("class_name"):
        idx = group.index.to_numpy()
        ps_class_means[class_name] = _class_mean(ps_z, idx)

    pv_class_means: dict[str, Tensor] = {}
    for class_name, group in pv_labels.groupby("class_name"):
        idx = group.index.to_numpy()
        pv_class_means[class_name] = _class_mean(pv_z, idx)

    # D1: consume alignment CSV rows past the `# n_overlap=N` header so the
    # (plantseg_class, plantvillage_class) mapping is honoured explicitly.
    alignment_df = pd.read_csv(class_alignment_csv, comment="#")
    matched: list[tuple[str, str]] = list(
        zip(alignment_df["plantseg_class"], alignment_df["plantvillage_class"])
    )

    sims_list: list[tuple[str, str, float]] = []
    for ps_class, pv_class in matched:
        if ps_class in ps_class_means and pv_class in pv_class_means:
            sim = F.cosine_similarity(
                ps_class_means[ps_class].unsqueeze(0),
                pv_class_means[pv_class].unsqueeze(0),
            ).item()
            sims_list.append((ps_class, pv_class, sim))

    assert len(sims_list) > 0, (
        f"transfer_correlation: no matched classes found. "
        f"Alignment CSV has {len(matched)} rows but none intersect with "
        f"the encoded class means. Check that plantseg_labels and "
        f"plantvillage_labels class_name columns match the alignment CSV's "
        f"plantseg_class / plantvillage_class."
    )

    cosine_similarities = {f"{ps}↔{pv}": s for ps, pv, s in sims_list}
    mean_cosine_sim = float(sum(s for _, _, s in sims_list) / len(sims_list))

    return {
        "mode": "quantitative",
        "n_overlap": n_overlap,
        "cosine_similarities": cosine_similarities,
        "mean_cosine_sim": mean_cosine_sim,
    }


# ---------------------------------------------------------------------------
# 5. concept_cooccurrence_pmi
# ---------------------------------------------------------------------------

def concept_cooccurrence_pmi(
    feature_acts_per_image: Tensor,   # (n_features, n_images) binary — 1 if feature fires on any patch
    threshold: float = 0.0,           # threshold for "fires"
    top_k_per_disease: int = 20,
    disease_labels: np.ndarray | None = None,  # (n_images,) int disease label
) -> pd.DataFrame:
    """Compute pairwise PMI for all feature pairs.

    For each image, a feature "fires" if its activation exceeds ``threshold``.
    PMI(i, j) = log( P(i,j) / (P(i) * P(j)) ), clipped to [-10, 10].

    If ``disease_labels`` is provided, PMI is computed per disease class.

    Returns
    -------
    pd.DataFrame with columns ``feature_i``, ``feature_j``, ``pmi``, ``disease_class``.
    """
    n_features, n_images = feature_acts_per_image.shape

    def _compute_pmi_for_subset(fires_sub: Tensor, image_indices: list[int], disease_class: int | str) -> list[dict]:
        """Compute top-k PMI pairs for a subset of images."""
        fires = fires_sub.float()  # (n_features, n_sub_images)
        n_sub = fires.shape[1]
        if n_sub == 0:
            return []

        p_i = fires.mean(dim=1)           # (n_features,)
        co = (fires @ fires.T) / n_sub    # (n_features, n_features)

        # PMI: log(P(i,j) / (P(i) * P(j)))
        outer = p_i.unsqueeze(1) * p_i.unsqueeze(0)   # (n_features, n_features)
        # Avoid division by zero
        pmi_mat = torch.log(co / (outer + 1e-10) + 1e-10).clamp(-10, 10)

        # Extract upper triangle (i < j) to avoid duplicates
        rows: list[dict] = []
        triu_indices = torch.triu_indices(n_features, n_features, offset=1)
        feat_i = triu_indices[0].numpy()
        feat_j = triu_indices[1].numpy()
        pmi_vals = pmi_mat[triu_indices[0], triu_indices[1]].numpy()

        # Take top-k by PMI
        n_pairs = len(pmi_vals)
        k = min(top_k_per_disease, n_pairs)
        if k == 0:
            return []
        top_indices = np.argpartition(pmi_vals, -k)[-k:]
        top_indices = top_indices[np.argsort(pmi_vals[top_indices])[::-1]]

        for idx in top_indices:
            rows.append({
                "feature_i": int(feat_i[idx]),
                "feature_j": int(feat_j[idx]),
                "pmi": float(pmi_vals[idx]),
                "disease_class": disease_class,
            })
        return rows

    fires = (feature_acts_per_image > threshold)  # (n_features, n_images) bool

    all_rows: list[dict] = []
    if disease_labels is None:
        all_rows = _compute_pmi_for_subset(fires, list(range(n_images)), "all")
    else:
        unique_diseases = np.unique(disease_labels)
        for d in unique_diseases:
            mask = disease_labels == d
            subset = fires[:, mask]
            all_rows.extend(_compute_pmi_for_subset(subset, np.where(mask)[0].tolist(), int(d)))

    if not all_rows:
        return pd.DataFrame(columns=["feature_i", "feature_j", "pmi", "disease_class"])

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# 6. should_trigger_layer6_fallback
# ---------------------------------------------------------------------------

def should_trigger_layer6_fallback(
    msae_disease_selectivity: pd.DataFrame,   # output of class_selectivity — has 'selectivity' column
    neuron_disease_selectivity: pd.DataFrame, # same schema
    threshold: float = 0.05,
) -> tuple[bool, dict]:
    """Determine if the MSAE disease selectivity is too close to the neuron baseline.

    Triggers a layer-6 fallback if the gap between MSAE and neuron-basis mean
    selectivity is below ``threshold`` (default 0.05).

    Returns
    -------
    (trigger_bool, diagnostic_dict)
        ``trigger_bool`` is True if fallback should be triggered.
        ``diagnostic_dict`` contains msae_mean, neuron_mean, gap, threshold.
    """
    msae_mean = float(msae_disease_selectivity["selectivity"].mean())
    neuron_mean = float(neuron_disease_selectivity["selectivity"].mean())
    gap = msae_mean - neuron_mean
    trigger = gap < threshold
    return (
        trigger,
        {
            "msae_mean": msae_mean,
            "neuron_mean": neuron_mean,
            "gap": gap,
            "threshold": threshold,
        },
    )
