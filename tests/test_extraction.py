from __future__ import annotations

import torch
import torch.nn as nn

from msae.extraction import filter_patches_l2, register_layer_hook


def test_filter_patches_l2_removes_low_norm():
    torch.manual_seed(42)
    B, P, D = 2, 256, 768

    patches = torch.randn(B, P, D)
    patches[:, :5, :] = 1e-8

    image_ids = torch.arange(B, dtype=torch.int64)
    kept, meta = filter_patches_l2(patches, image_ids, percentile=0.20)

    low_norm_indices = set(range(5))

    for idx in low_norm_indices:
        r, c = idx // 16, idx % 16
        assert not any(
            (
                meta[i, 1].item() == r
                and meta[i, 2].item() == c
                and meta[i, 0].item() == 0
            )
            for i in range(meta.shape[0])
        ), f"Low-norm patch (row={r}, col={c}) should have been filtered"


def test_filter_patches_l2_metadata_format():
    torch.manual_seed(0)
    B, P, D = 3, 256, 768
    patches = torch.randn(B, P, D)
    image_ids = torch.tensor([10, 20, 30], dtype=torch.int64)

    kept, meta = filter_patches_l2(patches, image_ids, percentile=0.20)

    assert meta.dtype == torch.int32, f"Metadata must be int32, got {meta.dtype}"
    assert meta.ndim == 2 and meta.shape[1] == 3, (
        f"Metadata must be (N, 3), got {meta.shape}"
    )
    assert kept.dtype == torch.bfloat16, (
        f"Kept patches must be bf16, got {kept.dtype}"
    )
    assert kept.ndim == 2 and kept.shape[1] == D, (
        f"Kept patches must be (N, {D}), got {kept.shape}"
    )

    valid_ids = {10, 20, 30}
    assert set(meta[:, 0].tolist()).issubset(valid_ids)

    assert meta[:, 1].min().item() >= 0 and meta[:, 1].max().item() <= 15
    assert meta[:, 2].min().item() >= 0 and meta[:, 2].max().item() <= 15


def test_register_hook_collects_correct_layer():
    torch.manual_seed(5)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([
                nn.Linear(16, 16),
                nn.Linear(16, 16),
                nn.Linear(16, 16),
            ])

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    model = TinyModel()
    # set to inference mode
    model.train(False)

    buffer = []
    handle = register_layer_hook(model, layer_idx=1, buffer=buffer)

    x = torch.randn(4, 16)
    with torch.no_grad():
        model(x)

    handle.remove()

    assert len(buffer) == 1, f"Expected 1 captured tensor, got {len(buffer)}"

    with torch.no_grad():
        expected = model.blocks[1](model.blocks[0](x))

    assert torch.allclose(buffer[0], expected), (
        "Hook did not capture layer-1 output correctly"
    )
