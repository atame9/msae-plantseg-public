import pytest
import torch
import numpy as np


@pytest.fixture
def synth_acts():
    torch.manual_seed(42)
    return torch.randn(128, 768)  # fp32, CPU


@pytest.fixture
def synth_labels():
    rng = np.random.default_rng(42)
    species = rng.integers(0, 4, size=128)
    disease = rng.integers(0, 6, size=128)
    return species, disease


@pytest.fixture
def synth_masks():
    torch.manual_seed(42)
    return torch.randint(0, 2, (8, 16, 16)).bool()


@pytest.fixture
def tiny_msae():
    """D=8, K=16, ks=(4, 8, 16) — small enough for hand-computable tests."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    # Note: models.py does not exist yet; this fixture will be used once Stage 2 creates it.
    # Return a config dict instead so conftest does not fail when models.py is absent.
    return {"input_dim": 8, "max_features": 16, "nested_ks": (4, 8, 16)}


@pytest.fixture
def mock_dinov2_block():
    """Tiny nn.Sequential of 3 Linear blocks simulating model.blocks for hook tests."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(64, 64),
        nn.Linear(64, 64),
        nn.Linear(64, 64),
    )
