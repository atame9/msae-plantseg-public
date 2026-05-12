"""Tests for src/msae/baselines.py — all CPU, no GPU required."""
import torch

from msae.baselines import fit_pca_gpu, neuron_basis_features


def test_pca_recovers_diagonal_covariance():
    """Synthesize data with known covariance structure.
    The top PCA component should align with the planted highest-variance direction."""
    torch.manual_seed(42)
    N, D = 500, 16
    # Create data: first dim has variance 100, rest have variance 1
    data = torch.randn(N, D)
    data[:, 0] *= 10  # first dim has std=10, var=100

    result = fit_pca_gpu(data, n_components=D, niter=10, stability_tolerance=0.02)
    assert result['stable'], "PCA should be stable on well-conditioned data"

    V = result['V']  # (D, D)
    top_component = V[:, 0].abs()  # first principal component

    # Top component should load most heavily on dimension 0
    assert top_component.argmax().item() == 0, \
        f"Top PCA component should align with dim 0 (highest variance), got {top_component.argmax().item()}"

    # Explained variance ratio should sum to ~1.0
    evr_sum = result['explained_variance_ratio'].sum().item()
    assert abs(evr_sum - 1.0) < 0.02, f"EVR should sum to ~1.0, got {evr_sum}"


def test_pca_stability_check_triggers_fallback(monkeypatch):
    """If GPU PCA returns an unstable result (cumsum ratio far from 1), fit_pca
    should fall back to sklearn."""
    import msae.baselines as bl

    # Monkeypatch fit_pca_gpu to return unstable
    def fake_gpu_pca(acts, n_components=768, niter=4, stability_tolerance=0.01):
        return {'stable': False}

    sklearn_called = []
    original_sklearn = bl.fit_pca_sklearn
    def fake_sklearn(acts_cpu, n_components=768, batch_size=65536):
        sklearn_called.append(True)
        return original_sklearn(acts_cpu, n_components=min(n_components, acts_cpu.shape[1]))

    monkeypatch.setattr(bl, 'fit_pca_gpu', fake_gpu_pca)
    monkeypatch.setattr(bl, 'fit_pca_sklearn', fake_sklearn)

    acts = torch.randn(100, 16)
    result = bl.fit_pca(acts, n_components=8)

    assert len(sklearn_called) == 1, "sklearn fallback must be called when GPU PCA is unstable"
    assert result['stable'], "sklearn result must be marked stable"


def test_neuron_basis_is_identity():
    x = torch.randn(10, 768)
    result = neuron_basis_features(x)
    assert result is x, "neuron_basis_features must return the exact same tensor (identity)"
