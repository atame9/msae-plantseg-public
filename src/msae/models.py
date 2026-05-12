from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MatryoshkaSAE(nn.Module):
    def __init__(self, input_dim: int = 768, max_features: int = 12288,
                 nested_ks: tuple[int, ...] = (256, 768, 3072, 12288)):
        super().__init__()
        self.input_dim = input_dim
        self.max_features = max_features
        self.nested_ks = tuple(nested_ks)
        self.encoder = nn.Linear(input_dim, max_features, bias=True)
        self.decoder = nn.Linear(max_features, input_dim, bias=True)
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))

    def encode(self, x: Tensor) -> Tensor:
        # x shape: (B, input_dim)
        # Returns (B, max_features) — sparse via ReLU
        return F.relu(self.encoder(x - self.pre_bias))

    def decode_chunked(self, z: Tensor) -> dict[int, Tensor]:
        # Build chunk boundaries from nested_ks
        # chunks[0] = (0, nested_ks[0]), chunks[i] = (nested_ks[i-1], nested_ks[i])
        ks = self.nested_ks
        chunk_bounds = [(0, ks[0])] + [(ks[i - 1], ks[i]) for i in range(1, len(ks))]

        partial = torch.zeros(z.shape[0], self.input_dim, device=z.device, dtype=z.dtype)
        recons = {}
        for (a, b), k in zip(chunk_bounds, ks):
            # decoder.weight shape: (input_dim, max_features)
            # decoder.weight[:, a:b] shape: (input_dim, b-a)
            partial = partial + z[:, a:b] @ self.decoder.weight[:, a:b].T
            recons[k] = partial + self.decoder.bias + self.pre_bias
        return recons

    def forward(self, x: Tensor) -> tuple[Tensor, dict[int, Tensor]]:
        z = self.encode(x)
        return z, self.decode_chunked(z)


class StandardSAE(nn.Module):
    def __init__(self, input_dim: int = 768, n_features: int = 12288):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.encoder = nn.Linear(input_dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, input_dim, bias=True)
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))

    def encode(self, x: Tensor) -> Tensor:
        return F.relu(self.encoder(x - self.pre_bias))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encode(x)
        recon = z @ self.decoder.weight.T + self.decoder.bias + self.pre_bias
        return z, recon


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int = 768, num_classes: int = 115):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)
