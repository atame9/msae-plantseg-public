"""Parity tests for the sparse-scipy encode path vs the legacy dense path.

The evaluate pipeline switched to ``scipy.sparse.csr_matrix`` for the
per-patch × per-feature activation matrix to avoid OOM at production dims
(3.3M patches × 12288 features would be ~162 GB dense fp32 on a 32 GB host).
Every consumer that used to take a dense ``torch.Tensor`` now dispatches on
``scipy.sparse.issparse`` and takes a sparse-aware code path. The fixture's
R6 tolerances (``|Δ| <= 1e-6`` for selectivity, ``|Δ| <= 1e-4`` for MI)
require byte-for-byte parity modulo floating-point rounding; these tests
assert that.

All tests run on CPU with synthetic data so they run in seconds.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from msae.evaluate import (
    build_grid_acts_chunked,
    class_selectivity,
    compute_mi,
    encode_sparse,
)
from msae.models import MatryoshkaSAE


def _make_activations(n_patches: int, n_features: int, density: float, seed: int = 42):
    """Return (dense_torch_fp32, sparse_scipy_csr_fp32) with identical content."""
    rng = np.random.default_rng(seed)
    # Random sparsity pattern
    mask = rng.random((n_patches, n_features)) < density
    vals = rng.standard_normal((n_patches, n_features)).astype(np.float32)
    # Enforce ReLU-like non-negativity to mirror MSAE encoder output
    vals = np.abs(vals)
    vals[~mask] = 0.0
    dense = torch.from_numpy(vals)
    sparse = sp.csr_matrix(vals)
    return dense, sparse


# ---------------------------------------------------------------------------
# class_selectivity parity
# ---------------------------------------------------------------------------

def test_class_selectivity_sparse_matches_dense():
    n_patches, n_features, n_classes = 1000, 64, 5
    dense, sparse = _make_activations(n_patches, n_features, density=0.1)
    labels = np.random.default_rng(0).integers(0, n_classes, size=n_patches)

    d = class_selectivity(dense, labels, n_classes=n_classes)
    s = class_selectivity(sparse, labels, n_classes=n_classes)

    np.testing.assert_allclose(
        s["selectivity"].to_numpy(),
        d["selectivity"].to_numpy(),
        atol=1e-6,
        rtol=1e-6,
    )
    assert (s["feature_id"] == d["feature_id"]).all()


def test_class_selectivity_sparse_with_healthy_collapse():
    n_patches, n_features, n_classes = 400, 32, 6
    dense, sparse = _make_activations(n_patches, n_features, density=0.15)
    labels = np.random.default_rng(1).integers(0, n_classes, size=n_patches)
    healthy = [1, 3]  # collapse classes 1 & 3 into one merged "healthy" class

    d = class_selectivity(dense, labels, n_classes=n_classes, healthy_class_ids=healthy)
    s = class_selectivity(sparse, labels, n_classes=n_classes, healthy_class_ids=healthy)

    np.testing.assert_allclose(
        s["selectivity"].to_numpy(),
        d["selectivity"].to_numpy(),
        atol=1e-6,
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# compute_mi parity — subsample path and full path
# ---------------------------------------------------------------------------

def test_compute_mi_sparse_matches_dense_no_subsample():
    # Below max_samples so the subsample branch is skipped; sparse and dense
    # paths should produce bit-identical acts_np after .toarray() / .numpy().
    n_patches, n_features = 500, 16
    dense, sparse = _make_activations(n_patches, n_features, density=0.2)
    labels = np.random.default_rng(2).integers(0, 4, size=n_patches)

    mi_d = compute_mi(dense, labels, max_samples=10_000, seed=42)
    mi_s = compute_mi(sparse, labels, max_samples=10_000, seed=42)

    np.testing.assert_allclose(mi_s, mi_d, atol=1e-6, rtol=1e-6)


def test_compute_mi_sparse_matches_dense_with_subsample():
    # Above max_samples so stratified subsample kicks in; both paths use the
    # same seed-derived sample_idx so the resulting MI arrays must match.
    n_patches, n_features = 2_000, 16
    dense, sparse = _make_activations(n_patches, n_features, density=0.2)
    labels = np.random.default_rng(3).integers(0, 4, size=n_patches)

    mi_d = compute_mi(dense, labels, max_samples=500, seed=42)
    mi_s = compute_mi(sparse, labels, max_samples=500, seed=42)

    # sklearn kNN MI has an internal RNG; both paths pass random_state=seed,
    # so results are deterministic and identical.
    np.testing.assert_allclose(mi_s, mi_d, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# build_grid_acts_chunked parity
# ---------------------------------------------------------------------------

def test_build_grid_acts_chunked_sparse_matches_dense():
    # Small synthetic setup: 4 images × 4 patches (2×2 grid simplified to 16×16
    # for API) with 32 features. Masks threshold out half the grid.
    n_images, patches_per_img = 4, 256
    n_features = 32
    n_patches = n_images * patches_per_img
    dense, sparse = _make_activations(n_patches, n_features, density=0.25)

    # meta: (image_id, row, col) with image_id=i for patches [i*256:(i+1)*256)
    meta_rows = []
    for i in range(n_images):
        for r in range(16):
            for c in range(16):
                meta_rows.append((i, r, c))
    meta = torch.tensor(meta_rows, dtype=torch.long)

    masks = torch.zeros(n_images, 16, 16, dtype=torch.bool)
    masks[:, :8, :] = True  # top half is "foreground" for every image

    iou_d = build_grid_acts_chunked(
        z_all=dense,
        meta=meta,
        masked_img_indices=list(range(n_images)),
        masks=masks,
        feature_chunk=8,
        device="cpu",
    )
    iou_s = build_grid_acts_chunked(
        z_all=sparse,
        meta=meta,
        masked_img_indices=list(range(n_images)),
        masks=masks,
        feature_chunk=8,
        device="cpu",
    )

    np.testing.assert_allclose(iou_s.numpy(), iou_d.numpy(), atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# encode_sparse vs encode_batched roundtrip on a real MSAE on CPU
# ---------------------------------------------------------------------------

def test_encode_sparse_matches_encode_batched_on_msae():
    """A freshly initialized MSAE with ReLU-sparse output should encode to
    matching dense and sparse representations (sparse path is just a CSR
    materialization of the same values)."""
    torch.manual_seed(0)
    model = MatryoshkaSAE(input_dim=16, max_features=32, nested_ks=(8, 16, 32))
    # Give the encoder bias a push so we see both zeros and nonzeros.
    with torch.no_grad():
        model.encoder.bias.uniform_(-0.5, 0.5)
    model.eval()

    acts = torch.randn(200, 16)
    # Dense path (fp32 for fair comparison; bf16 default would hide tiny eps)
    from msae.evaluate import encode_batched
    dense = encode_batched(
        model, acts, batch_size=32, device="cpu",
        store_dtype=torch.float32, show_progress=False,
    )
    sparse = encode_sparse(
        model, acts, batch_size=32, device="cpu", show_progress=False,
    )

    assert dense.shape == sparse.shape
    # Ensure the sparse path captures ReLU zeros exactly and preserves nonzeros.
    dense_np = dense.detach().cpu().numpy()
    sparse_dense = sparse.toarray()
    np.testing.assert_allclose(sparse_dense, dense_np, atol=1e-6, rtol=1e-6)
    # Also confirm we actually produced some sparsity — otherwise the test
    # isn't exercising the CSR path at all.
    assert sparse.nnz < dense.numel(), (
        f"sparse encode produced full matrix (nnz={sparse.nnz}, numel={dense.numel()}); "
        "test setup does not exercise the sparse path"
    )
