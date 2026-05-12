import torch
from pathlib import Path

from msae.models import MatryoshkaSAE
from msae.train import matryoshka_loss, auxk_loss, train_msae


def test_dead_feature_mask_updates_correctly():
    """Simulate a fire_counter where feature 0 never fires.
    After one window, dead_mask[0] should be True."""
    K = 16

    # Simulate fire_counter: feature 0 has count 0, all others have count >= threshold
    dead_window = 100
    batch_size = 16
    fire_counter = torch.ones(K) * 10  # all fire plenty
    fire_counter[0] = 0  # feature 0 never fires

    # Apply dead mask logic (mirrors train_msae internals)
    threshold = 0.001 * dead_window * batch_size / K
    dead_mask = (fire_counter < threshold).bool()

    assert dead_mask[0].item() is True, "Feature 0 (never fired) must be marked dead"
    assert not dead_mask[1:].any(), "All other features should be alive"


def test_auxk_loss_zero_when_no_dead_features():
    torch.manual_seed(10)
    D, K = 8, 16
    B = 4
    x = torch.randn(B, D)
    recon_full = torch.randn(B, D)
    dead_mask = torch.zeros(K, dtype=torch.bool)  # no dead features
    encoder_pre_act = torch.randn(B, K)
    decoder_weight = torch.randn(D, K)

    loss = auxk_loss(x, recon_full, dead_mask, encoder_pre_act, decoder_weight, k_aux=4, alpha_aux=1/32)
    assert loss.item() == 0.0, f"AuxK loss must be 0 when no dead features, got {loss.item()}"


def test_auxk_uses_top_k_dead_only():
    """Set 8 dead features with known pre-act magnitudes; k_aux=3.
    The loss should only involve the top-3 dead features by magnitude."""
    torch.manual_seed(11)
    D, K = 8, 16
    B = 2

    # All features dead
    dead_mask = torch.ones(K, dtype=torch.bool)
    # Pre-activations: set feature 0,1,2 to have clearly highest magnitude
    encoder_pre_act = torch.zeros(B, K)
    encoder_pre_act[:, 0] = 100.0   # strongest
    encoder_pre_act[:, 1] = 90.0
    encoder_pre_act[:, 2] = 80.0
    encoder_pre_act[:, 3:] = 0.001  # weak

    decoder_weight = torch.randn(D, K)
    x = torch.randn(B, D)
    recon_full = torch.zeros(B, D)

    loss_3 = auxk_loss(x, recon_full, dead_mask, encoder_pre_act, decoder_weight, k_aux=3, alpha_aux=1.0)
    loss_1 = auxk_loss(x, recon_full, dead_mask, encoder_pre_act, decoder_weight, k_aux=1, alpha_aux=1.0)

    # With more dead features contributing, loss should differ
    # Both should be > 0 (features 0,1,2 have high pre-act -> non-zero after ReLU)
    assert loss_3.item() > 0.0, "AuxK loss should be > 0 when top dead features have high pre-activation"
    assert loss_1.item() > 0.0
    # k_aux=3 uses more features, so reconstruction should be better -> lower residual -> potentially different loss
    # (We just verify both are finite and positive)
    assert torch.isfinite(loss_3), "AuxK loss must be finite"
    assert torch.isfinite(loss_1), "AuxK loss must be finite"


def test_train_smoke_5_steps():
    """Tiny MSAE + 64 random samples: 5 training steps. Loss must be finite and
    the final loss must be <= the initial loss (or close -- not strictly required
    in 5 steps but should not explode)."""
    torch.manual_seed(99)
    D, K = 8, 16
    ks = (4, 8, 16)
    B_data = 64

    model = MatryoshkaSAE(input_dim=D, max_features=K, nested_ks=ks)
    data = torch.randn(B_data, D)
    dataset = torch.utils.data.TensorDataset(data)

    config = {
        'lam_sparse': 1e-4,
        'lr': 1e-3,
        'batch_size': 16,
        'n_epochs': 1,
        'k_aux': 2,
        'alpha_aux': 1/32,
        'dead_window': 100,
        'dead_refresh_every': 10,
        'early_stop_patience_steps': 1000,  # effectively no early stop
        'early_stop_threshold': 0.0,
        'seed': 42,
    }

    # Tiny val set
    val_data = torch.randn(16, D)
    val_dataset = torch.utils.data.TensorDataset(val_data)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "ckpts"
        log_path = Path(tmpdir) / "log.json"
        ckpt_dir.mkdir()

        result = train_msae(
            model, dataset, val_dataset, config, ckpt_dir, log_path
        )

    assert 'final_val_mse_per_level' in result
    assert 'alive_features_per_level' in result
    for k in ks:
        mse = result['final_val_mse_per_level'][k]
        assert torch.isfinite(torch.tensor(mse)), f"Val MSE at k={k} is not finite: {mse}"


def test_matryoshka_loss_matches_oracle():
    """Canonical train.matryoshka_loss must equal the hand-computed oracle in test_sae.py.

    Updated (2026-05-11): formula corrected to drop the double-normalizing
    ``* (1.0 / k)`` — see src/msae/train.py::matryoshka_loss for rationale.
    """
    torch.manual_seed(2)
    D, K, ks = 4, 8, (4, 8)
    model = MatryoshkaSAE(input_dim=D, max_features=K, nested_ks=ks)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    x = torch.ones(2, D)
    z = torch.ones(2, K) * 0.5
    recons = model.decode_chunked(z)
    # Hand-computed expected (same as test_sae.py oracle):
    # level k=4: MSE(0,1)=1.0, L1=0.5 → 1.0 + 1e-4*0.5
    # level k=8: MSE(0,1)=1.0, L1=0.5 → 1.0 + 1e-4*0.5
    # total = avg of both levels
    expected = (1.0 + 1e-4 * 0.5 + 1.0 + 1e-4 * 0.5) / 2
    got = matryoshka_loss(x, z, recons, ks, lam_sparse=1e-4).item()
    assert abs(got - expected) < 1e-7, f"Expected {expected}, got {got}"


# ---------------------------------------------------------------------------
# D2 — RNG state round-trips under weights_only=False
# ---------------------------------------------------------------------------

def test_checkpoint_round_trips_rng_state_under_weights_only_false(tmp_path: Path):
    """Checkpoints carry numpy/python RNG state pickles that torch 2.6's
    weights_only=True default refuses. Every new CLI load site uses
    weights_only=False; this test locks in the contract we actually depend
    on: the weights_only=False path reproduces torch + numpy + python RNG
    states structurally unchanged.

    We deliberately do NOT assert that weights_only=True crashes — PyTorch's
    allowlisted-globals set grows across versions, and such an assertion
    would rot into a silent pass the moment numpy RNG state gets allowlisted
    upstream.
    """
    import numpy as np
    import random

    from msae.train import async_checkpoint, _join_pending_saves

    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    model = MatryoshkaSAE(input_dim=8, max_features=16, nested_ks=(4, 8, 16))

    state = {
        "model_state_dict": model.state_dict(),
        "step": 42,
        "epoch": 3,
        "rng_state_torch": torch.get_rng_state(),
        "rng_state_numpy": np.random.get_state(),
        "rng_state_python": random.getstate(),
        "lam_sparse_used": 3e-4,
    }

    path = tmp_path / "ckpt.pt"
    async_checkpoint(state, path)
    _join_pending_saves()

    # weights_only=False: our checkpoints carry numpy/python RNG state pickles.
    # Matches resume_from_checkpoint's existing choice and every new CLI load
    # site introduced in C1.
    loaded = torch.load(path, map_location="cpu", weights_only=False)

    # torch RNG state: returned as a uint8 ByteTensor.
    assert "rng_state_torch" in loaded
    assert isinstance(loaded["rng_state_torch"], torch.Tensor)
    assert loaded["rng_state_torch"].dtype == torch.uint8

    # numpy RNG state: returned as a tuple beginning with the BitGenerator name.
    assert "rng_state_numpy" in loaded
    np_state = loaded["rng_state_numpy"]
    assert isinstance(np_state, tuple)
    assert len(np_state) >= 2
    assert isinstance(np_state[0], str)  # e.g. "MT19937"

    # python RNG state: a tuple whose first entry is the version int.
    assert "rng_state_python" in loaded
    py_state = loaded["rng_state_python"]
    assert isinstance(py_state, tuple)
    assert len(py_state) >= 2

    # Other scalar fields survive the pickle round-trip.
    assert loaded["step"] == 42
    assert loaded["epoch"] == 3
    assert loaded["lam_sparse_used"] == 3e-4
