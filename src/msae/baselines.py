from __future__ import annotations
import logging
import math

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def fit_pca_gpu(
    acts: Tensor,                    # (N, 768) bf16 or fp32, GPU or CPU
    n_components: int = 768,
    niter: int = 4,
    stability_tolerance: float = 0.01,
) -> dict:
    """torch.pca_lowrank on centered, fp32-cast data.

    Returns dict with keys: V, mean, explained_variance, explained_variance_ratio, stable.

    Stability check: compares recovered variance (sum of S**2 / (N-1)) against
    the TRUE total centered variance. The previous ``cumsum(EVR)[-1] == 1``
    check was tautological (ratio of a sum to itself). At q == D the ratio
    should be ~1.0; we fail unstable if it falls outside
    [1 - stability_tolerance, 1 + stability_tolerance] or is non-finite.
    """
    acts_f = acts.float()
    mean = acts_f.mean(dim=0, keepdim=True)
    acts_centered = acts_f - mean

    U, S, V = torch.pca_lowrank(acts_centered, q=n_components, niter=niter)

    explained_variance = S ** 2 / (acts.shape[0] - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()

    # Stability check: recovered vs. true total variance
    total_var = (acts_centered.pow(2).sum() / (acts.shape[0] - 1)).item()
    recovered_var = explained_variance.sum().item()
    ratio_recovered = (recovered_var / total_var) if total_var > 0 else 1.0
    lo = 1.0 - stability_tolerance
    hi = 1.0 + stability_tolerance
    if not (lo <= ratio_recovered <= hi) or not math.isfinite(ratio_recovered):
        logger.warning(
            "fit_pca_gpu: recovered_var/total_var = %.4f outside [%.4f, %.4f]. "
            "Marking result unstable.",
            ratio_recovered, lo, hi,
        )
        return {'stable': False, 'ratio_recovered': ratio_recovered}

    logger.info(
        "fit_pca_gpu: stable (recovered_var/total_var=%.4f, device=%s, N=%d, D=%d, q=%d)",
        ratio_recovered, acts.device, acts.shape[0], acts.shape[1], n_components,
    )

    return {
        'V': V.cpu(),
        'mean': mean.squeeze(0).cpu(),
        'explained_variance': explained_variance.cpu(),
        'explained_variance_ratio': explained_variance_ratio.cpu(),
        'stable': True,
        'ratio_recovered': ratio_recovered,
    }


def fit_pca_sklearn(
    acts_cpu: np.ndarray,    # (N, D) float32
    n_components: int = 768,
    batch_size: int = 65536,
) -> dict:
    """sklearn IncrementalPCA fallback. Same return schema as ``fit_pca_gpu``
    (plus ``mean``, exposed via ``ipca.mean_``)."""
    from sklearn.decomposition import IncrementalPCA

    if acts_cpu.dtype != np.float32:
        acts_cpu = acts_cpu.astype(np.float32, copy=False)

    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    ipca.fit(acts_cpu)

    return {
        'V': torch.from_numpy(ipca.components_.T.copy()),   # (D, n_components)
        'mean': torch.from_numpy(ipca.mean_.copy()),
        'explained_variance': torch.from_numpy(ipca.explained_variance_.copy()),
        'explained_variance_ratio': torch.from_numpy(ipca.explained_variance_ratio_.copy()),
        'stable': True,
    }


def fit_pca(
    acts: Tensor,
    n_components: int = 768,
) -> dict:
    """Public API. Tries GPU PCA first; falls back to sklearn on stability failure.

    Always returns the same dict schema:
    {'V': Tensor (D, n_components), 'mean': Tensor (D,),
     'explained_variance': Tensor, 'explained_variance_ratio': Tensor,
     'stable': True}
    """
    result = fit_pca_gpu(acts, n_components)
    if result['stable']:
        return result

    logger.warning("GPU PCA unstable; falling back to sklearn IncrementalPCA")
    return fit_pca_sklearn(acts.float().cpu().numpy(), n_components)


def project_pca(
    acts: Tensor,
    V: Tensor,
    k: int,
    mean: Tensor | None = None,
) -> Tensor:
    """Center acts (using the FITTED mean) and project onto the top-k components.

    Args:
        acts: (N, D) activations. Any device, any floating dtype.
        V:    (D, n_components) matrix from ``fit_pca`` output.
        k:    number of components to keep.
        mean: (D,) the fitted mean from ``fit_pca``. If None, falls back to
              ``acts.mean(dim=0)`` with a warning -- only correct when ``acts``
              is the same set the fit was performed on.

    Returns:
        (N, k) projected activations on ``acts.device``.
    """
    target_dtype = acts.dtype if acts.is_floating_point() else torch.float32
    V = V.to(acts.device, dtype=target_dtype)
    if mean is None:
        logger.warning(
            "project_pca: no fitted mean provided; centering on input mean. "
            "Pass mean=fit_pca_result['mean'] for correct projection of held-out data."
        )
        mean_use = acts.float().mean(dim=0, keepdim=True).to(target_dtype)
    else:
        mean_use = mean.to(acts.device, dtype=target_dtype).unsqueeze(0)
    return (acts.to(target_dtype) - mean_use) @ V[:, :k]


def neuron_basis_features(acts: Tensor) -> Tensor:
    """Identity -- returns acts unchanged.

    This is Baseline A: raw DINOv2 dimensions treated as features. Wrapped as a
    function so the eval pipeline treats all four methods uniformly.
    """
    return acts
