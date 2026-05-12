from __future__ import annotations
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from msae.models import MatryoshkaSAE, StandardSAE, LinearProbe

logger = logging.getLogger(__name__)


class _NullContext:
    """Drop-in for torch.autocast on CPU paths so the same `with` block compiles."""
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        return None


def setup_perf_flags() -> None:
    """Idempotent runtime perf flags. Call once at notebook top after `import torch`.

    TF32 + matmul precision are no-ops on non-CUDA devices. cudnn.benchmark is safe
    when input shapes are fixed (true here: batch_size and feature_dim never vary).
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


@torch.compiler.disable
def normalize_decoder(model: nn.Module) -> None:
    """Unit-norm-per-column decoder normalization (spec lines 532-539).

    Runs in eager mode -- in-place parameter mutation under no_grad() breaks
    Dynamo tracing, so we hard-disable compile here. Single call per training step.

    Uses in-place `.div_()` rather than `model.decoder.weight.data = F.normalize(...)`.
    The `.data = ...` rebind would allocate a new storage and replace the parameter's
    tensor; fused Adam captures parameter pointers at optimizer construction time and
    Dynamo captures them at compile time, so a rebind silently breaks both. The
    `.clamp(min=1e-8)` guards against all-zero columns producing NaN.
    """
    with torch.no_grad():
        norms = model.decoder.weight.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        model.decoder.weight.data.div_(norms)


def async_checkpoint(state: dict, path: Path | str) -> None:
    """Synchronous checkpoint save.

    Writes the checkpoint to disk and returns when bytes are flushed. Function
    name preserved for call-site compatibility (originally async for high-latency
    storage backends). On local NVMe a 14 MB checkpoint writes in <100 ms.

    Call ``_join_pending_saves()`` after training loops — it's a no-op but
    retained for interface stability.
    """
    cpu_state: dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            cpu_state[k] = v.detach().cpu().clone()
        elif isinstance(v, dict):
            cpu_state[k] = {
                kk: (vv.detach().cpu().clone() if isinstance(vv, torch.Tensor) else vv)
                for kk, vv in v.items()
            }
        else:
            cpu_state[k] = v

    torch.save(cpu_state, str(path))


def _join_pending_saves(timeout_per_save: float = 120.0) -> None:
    """No-op; retained because training loops still call it.

    Previously blocked on background save threads. The async path is gone
    (see ``async_checkpoint``); saves now return when the write completes,
    so there is nothing to join.
    """
    _ = timeout_per_save  # keep signature stable


def _atomic_json_dump(obj: Any, path: Path | str) -> None:
    """Write JSON atomically: dump to a .tmp sibling, then os.replace().

    Avoids leaving a half-written training log on disk if the process dies mid-write
    (e.g. a Colab disconnect). ``os.replace`` is atomic on POSIX and Windows.
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp") if path.suffix else path.with_suffix(".tmp")
    with open(tmp_path, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


def resume_from_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Restore a training state written by ``async_checkpoint``.

    Loads model weights (required), optimizer state (if provided), and restores
    torch + numpy + python RNG states when present. Returns the remaining fields
    (``step``, ``epoch``, ``fire_counter``, ``dead_mask``, ``lam_sparse_used``)
    so the caller can resume the outer training loop.

    Backwards-compatible with older checkpoints that only have ``rng_state``
    (torch-only schema).
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])

    if 'rng_state_torch' in state:
        torch.set_rng_state(state['rng_state_torch'])
    elif 'rng_state' in state:  # legacy schema
        torch.set_rng_state(state['rng_state'])
    if 'rng_state_numpy' in state:
        np.random.set_state(state['rng_state_numpy'])
    if 'rng_state_python' in state:
        random.setstate(state['rng_state_python'])

    return {
        k: v for k, v in state.items()
        if k not in {'model_state_dict', 'optimizer',
                     'rng_state', 'rng_state_torch', 'rng_state_numpy', 'rng_state_python'}
    }


def matryoshka_loss(
    x: Tensor,
    z: Tensor,
    recons: dict[int, Tensor],
    nested_ks: tuple[int, ...],
    lam_sparse: float = 1e-4,
) -> Tensor:
    """Matryoshka MSE + per-level L1 loss (spec lines 496-504).

    Fix (2026-05-11): removed the extra ``* (1.0 / k)`` on the L1 term that
    was double-normalizing per feature count. ``z[:,:k].abs().mean()`` already
    divides by ``B * k``; the extra ``/ k`` made the effective sparsity
    gradient on outer features ~12288× weaker than on inner features, and
    ~12288× weaker than the same lam applied to a Standard SAE. Empirically
    (fixture sweep, bs=32768, 1-epoch), even lam_sparse=10000 could only
    move MSAE L0 from ~6828 → ~6673 (2% drop). Post-fix, MSAE's outer-feature
    sparsity matches Standard's, so the paper probe range (1e-2, 1e-1, 1.0)
    should find a valid candidate in the [10, 200] L0 gate.

    The hand-computed oracle test (tests/test_sae.py::test_matryoshka_loss_hand_example)
    is updated to match the corrected formula.
    """
    loss = 0.0
    for k in nested_ks:
        mse = F.mse_loss(recons[k], x)
        l1  = z[:, :k].abs().mean()
        loss = loss + mse + lam_sparse * l1
    return loss / len(nested_ks)


@torch.compiler.disable
def auxk_loss(
    x: Tensor,
    recon_full: Tensor,
    dead_features_mask: Tensor,   # (K,) bool -- True = dead
    encoder_pre_act: Tensor,      # (B, K) -- pre-ReLU encoder outputs (x @ W + b)
    decoder_weight: Tensor,       # (input_dim, K) -- decoder.weight
    k_aux: int = 512,
    alpha_aux: float = 1 / 32,
) -> Tensor:
    """AuxK auxiliary loss to encourage dead features to revive (spec lines 515-518).

    Stays in fp32 even under bf16 autocast to avoid underflow for small magnitudes.
    Compile-disabled: data-dependent control flow (early-return on no-dead-features)
    breaks Dynamo tracing.
    """
    # Cast all inputs to fp32 for numerical stability
    x = x.float()
    recon_full = recon_full.float()
    encoder_pre_act = encoder_pre_act.float()
    decoder_weight = decoder_weight.float()

    if not dead_features_mask.any():
        return torch.tensor(0.0, device=x.device)

    # Get pre-activation magnitudes for dead features only
    pre_act_dead = encoder_pre_act.clone()
    pre_act_dead[:, ~dead_features_mask] = float('-inf')

    # Pick top-k_aux by magnitude across dead features (per batch); order doesn't
    # matter -- we scatter the values back into a fixed-shape z_aux.
    n_dead = int(dead_features_mask.sum().item())
    k_actual = min(k_aux, n_dead)
    topk_vals, topk_idx = pre_act_dead.topk(k_actual, dim=1, sorted=False)

    # Create sparse z_aux: scatter top-k dead feature pre-activations back
    B, K = encoder_pre_act.shape
    z_aux = torch.zeros(B, K, device=x.device, dtype=torch.float32)
    z_aux.scatter_(1, topk_idx, F.relu(topk_vals))

    # Reconstruct from z_aux using full decoder weight
    recon_aux = z_aux @ decoder_weight.T

    # Residual target: x minus the full (main path) reconstruction
    residual = x - recon_full.detach()

    return alpha_aux * F.mse_loss(recon_aux, residual)


def maybe_compile(
    fn: Callable[..., Any],
    *,
    warmup: Callable[[Callable[..., Any]], None] | None = None,
    name: str = "fn",
) -> Callable[..., Any]:
    """Cascading torch.compile: max-autotune -> max-autotune-no-cudagraphs ->
    reduce-overhead -> eager. Always returns a callable; never blocks training.

    A warmup callable forces real compilation (and surfaces failures) before
    the training loop, paying the 2-8 min compile cost up front instead of
    inside step 1. Without warmup, compile is lazy and the cascade can't detect
    failures until the first call -- pass `warmup` to enable cascade detection.
    """
    for mode in ("max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"):
        try:
            compiled = torch.compile(fn, mode=mode)
            if warmup is not None:
                warmup(compiled)
            logger.info("torch.compile mode=%s succeeded for %s", mode, name)
            return compiled
        except Exception as e:
            logger.warning(
                "torch.compile mode=%s failed for %s: %s: %s",
                mode, name, type(e).__name__, e,
            )
    logger.warning("All torch.compile modes failed for %s; falling back to eager", name)
    return fn


def lambda_sparse_probe(
    model_factory: Callable[[], nn.Module],
    dataset: Dataset,
    candidates: tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0),
    n_steps: int = 500,
    target_l0_range: tuple[int, int] = (30, 80),
    seed: int = 42,
    device: str | torch.device = "cuda",
) -> float:
    """3-point sweep over lam_sparse candidates; pick the one whose L0 is
    closest to the midpoint of target_l0_range (spec lines 542-544).

    Works for both MatryoshkaSAE (dict-of-recons forward, matryoshka_loss) and
    StandardSAE (single-tensor recon, mse + lam * L1). The model factory returns
    a fresh instance per candidate so the sweep is independent.

    CUDA-required: the probe runs 600 forward+backward passes through a
    (768, 12288) SAE -- on CPU this is 30-50 minutes per session and the
    L0 estimate doesn't reflect the bf16 autocast path used in actual
    training. Pass `device='cuda'` (default) or set `lam_sparse` explicitly
    in the caller's config to skip the probe.

    Candidate range: paper's (1e-2, 1e-1, 1.0) was too narrow at bs=32768 even
    after the 2026-05-11 matryoshka_loss fix (removed double-normalizing
    ``* (1.0 / k)``). Post-fix, MSAE at lam=1.0 lands L0≈3170 (down from
    L0≈5557 pre-fix, so the fix IS working) but still above the [10, 200]
    gate. Widened candidates to include 10.0 and 100.0 and bumped ``n_steps``
    300 → 500 to push the probe across the bend where the sparsity gradient
    starts dominating MSE; the extrapolation (roughly halves per decade)
    predicts L0 ≈ 1500 at lam=10 and L0 ≈ 300 at lam=100, bracketing target.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError(
            f"lambda_sparse_probe requires CUDA (got device='{device}'). "
            "Probe is only meaningful at training scale and must match the "
            "bf16 autocast numerics of the real training loop. Either pass "
            "device='cuda' or set lam_sparse explicitly in train_msae's "
            "config to bypass the probe."
        )
    setup_perf_flags()

    batch_size = min(256, len(dataset))
    midpoint = (target_l0_range[0] + target_l0_range[1]) / 2.0
    results = []

    def _make_loader(shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            pin_memory=True,
            num_workers=0,
            persistent_workers=False,
        )

    for lam in candidates:
        model = model_factory().to(device)
        model.train()

        # Init pre_bias from one real batch so the L0 sweep sees the same
        # input distribution the real training loop does (spec line 530).
        # In-place `.copy_()` preserves the parameter tensor's storage pointer,
        # which matters for any downstream fused optimiser bindings; `.data = ...`
        # rebinds and would silently break that contract.
        for _probe_batch in _make_loader(shuffle=False):
            _init_x = _probe_batch[0] if isinstance(_probe_batch, (list, tuple)) else _probe_batch
            _init_x = _init_x.float().to(device)
            model.pre_bias.data.copy_(_init_x.mean(dim=0))
            break

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, fused=True)

        loader = _make_loader(shuffle=True)
        loader_iter = iter(loader)

        for step in range(n_steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                if isinstance(model, MatryoshkaSAE):
                    z, recons = out
                    loss = matryoshka_loss(x, z, recons, model.nested_ks, lam_sparse=lam)
                else:
                    # StandardSAE-style: (z, recon) single-level
                    z, recon = out
                    loss = F.mse_loss(recon, x) + lam * z.abs().mean()
            loss.backward()
            optimizer.step()
            normalize_decoder(model)

        # Measure mean L0 on a held-out batch under the same autocast envelope.
        model.train(False)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            try:
                held_out = next(iter(_make_loader(shuffle=True)))
            except StopIteration:
                held_out = batch
            x_val = held_out[0] if isinstance(held_out, (list, tuple)) else held_out
            x_val = x_val.to(device, non_blocking=True).float()
            out_val = model(x_val)
            z_val = out_val[0] if isinstance(out_val, tuple) else out_val
            l0 = (z_val > 0).float().sum(dim=1).mean().item()

        results.append((lam, l0))
        logger.info("lambda_sparse_probe: lam=%.1e  L0=%.1f", lam, l0)

    # Log results to probe_results.json
    probe_log = [{"lam_sparse": lam, "l0": l0} for lam, l0 in results]
    with open("probe_results.json", "w") as f:
        json.dump(probe_log, f, indent=2)

    # Validate: all extremes check
    valid = [(lam, l0) for lam, l0 in results if 10 <= l0 <= 200]
    if not valid:
        msg = "No lambda achieved L0 in [10, 200]. Results: " + str(results)
        raise RuntimeError(msg)

    # Pick winner: closest L0 to midpoint
    best_lam, best_l0 = min(results, key=lambda pair: abs(pair[1] - midpoint))
    logger.info("lambda_sparse_probe winner: lam=%.1e  L0=%.1f", best_lam, best_l0)
    return best_lam


def train_msae(
    msae: MatryoshkaSAE,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: dict,
    checkpoint_dir: Path,
    log_path: Path,
) -> dict:
    """Full MSAE training loop with AuxK, dead-feature tracking, early stopping.

    Config keys: lam_sparse, lr, batch_size, n_epochs, k_aux, alpha_aux,
                 dead_window, dead_refresh_every, early_stop_patience_steps,
                 early_stop_threshold, seed.
    """
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = next(msae.parameters()).device
    use_cuda = device.type == 'cuda'

    # Auto-move to CUDA when available. Without this, a user who forgot
    # `.to("cuda")` in the notebook gets a confusing CUDA-required RuntimeError
    # out of the lam_sparse probe below; with it, the happy path just works.
    if device.type != 'cuda' and torch.cuda.is_available():
        msae = msae.to('cuda')
        device = torch.device('cuda')
        use_cuda = True
        logger.info("train_msae: moved model to CUDA (was on CPU)")

    # Resolve lam_sparse via probe if not given. Probe is CUDA-required; if msae
    # is on CPU and lam_sparse isn't provided, we fail fast with a clear message
    # rather than burning ~50 minutes on a CPU probe that wouldn't match training
    # numerics anyway.
    lam_sparse = config.get('lam_sparse')
    if lam_sparse is None:
        input_dim = msae.input_dim
        max_features = msae.max_features
        nested_ks = msae.nested_ks

        def _factory() -> MatryoshkaSAE:
            return MatryoshkaSAE(input_dim=input_dim, max_features=max_features, nested_ks=nested_ks)

        lam_sparse = lambda_sparse_probe(
            model_factory=_factory,
            dataset=train_dataset,
            seed=seed,
            device=device,
        )
        logger.info("lam_sparse resolved by probe: %.1e", lam_sparse)

    lr = config.get('lr', 3e-4)
    batch_size = config.get('batch_size', 16384)
    n_epochs = config.get('n_epochs', 6)
    k_aux = config.get('k_aux', 512)
    alpha_aux = config.get('alpha_aux', 1 / 32)
    dead_window = config.get('dead_window', 50_000)
    dead_refresh_every = config.get('dead_refresh_every', 1500)
    early_stop_patience = config.get('early_stop_patience_steps', 96)
    early_stop_threshold = config.get('early_stop_threshold', 0.01)

    if use_cuda:
        setup_perf_flags()

    def _make_loader(ds: Dataset, bs: int, shuffle: bool, drop_last: bool) -> DataLoader:
        # pin_memory enables zero-copy H2D transfer when paired with non_blocking=True;
        # num_workers=0 is correct because the dataset is RAM-resident TensorDataset
        # (spec line 531) -- workers add overhead for no I/O parallelism gain.
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=use_cuda,
            num_workers=0,
            persistent_workers=False,
        )

    val_loader = _make_loader(val_dataset, batch_size, shuffle=False, drop_last=False)

    # drop_last=True on train: gives fixed batch shape, prevents torch.compile
    # recompile on the ragged last batch.
    train_steps_per_epoch = max(1, len(train_dataset) // batch_size)
    total_steps = n_epochs * train_steps_per_epoch

    optimizer = torch.optim.Adam(msae.parameters(), lr=lr, fused=use_cuda)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-5
    )

    # Init pre_bias on first batch (spec line 530) -- BEFORE compile so the
    # compiled graph isn't built around a zero pre_bias and then invalidated.
    # In-place `.copy_()` (not `.data = ...`) to preserve parameter tensor identity,
    # matching the fused-Adam / Dynamo pointer discipline documented in
    # ``normalize_decoder``.
    for b in _make_loader(train_dataset, batch_size, shuffle=False, drop_last=False):
        first_batch = b[0].float() if isinstance(b, (list, tuple)) else b.float()
        msae.pre_bias.data.copy_(first_batch.mean(dim=0).detach().to(device))
        break

    nested_ks_tuple = msae.nested_ks
    input_dim = msae.input_dim

    # Compiled hot path: encoder forward + decode + matryoshka loss in one graph.
    # Returns pre_act so AuxK can use it without re-running the encoder.
    # AuxK + decoder normalization are NOT compiled (data-dependent control flow
    # / in-place param mutation break Dynamo).
    def _forward_and_loss(
        x: Tensor, nested_ks: tuple[int, ...], lam: float
    ) -> tuple[Tensor, Tensor, dict[int, Tensor], Tensor]:
        pre_act = msae.encoder(x - msae.pre_bias)
        z = F.relu(pre_act)
        recons = msae.decode_chunked(z)
        loss = matryoshka_loss(x, z, recons, nested_ks, lam)
        return z, pre_act, recons, loss

    if use_cuda:
        def _warmup(compiled_fn: Callable[..., Any]) -> None:
            optimizer.zero_grad(set_to_none=True)
            dummy_x = torch.randn(batch_size, input_dim, device=device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, _, _, _w_loss = compiled_fn(dummy_x, nested_ks_tuple, lam_sparse)
            _w_loss.backward()
            optimizer.zero_grad(set_to_none=True)
            del dummy_x, _w_loss

        compiled_step = maybe_compile(_forward_and_loss, warmup=_warmup, name="msae_step")
    else:
        compiled_step = _forward_and_loss

    # Dead-feature tracking. Counter lives on ``device`` so the per-step
    # ``add_`` stays on GPU; a single ``.cpu()`` at refresh time amortises the
    # sync over ``dead_refresh_every`` steps instead of paying it per step.
    max_features = msae.max_features
    fire_counter = torch.zeros(max_features, dtype=torch.long, device=device)
    dead_mask = torch.zeros(max_features, dtype=torch.bool)
    dead_mask_dev = dead_mask.to(device)

    # Early stopping state
    best_val_mse: Optional[float] = None
    best_alive_count: Optional[int] = None
    steps_without_improvement = 0

    global_step = 0
    n_epochs_trained = 0
    log_entries = []

    for epoch in range(n_epochs):
        msae.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        train_loader = _make_loader(train_dataset, batch_size, shuffle=True, drop_last=True)

        for batch in train_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True).float()

            try:
                if use_cuda:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        z, encoder_pre_act, recons, loss = compiled_step(
                            x, nested_ks_tuple, lam_sparse
                        )
                    recon_full = recons[nested_ks_tuple[-1]]
                else:
                    z, encoder_pre_act, recons, loss = compiled_step(
                        x, nested_ks_tuple, lam_sparse
                    )
                    recon_full = recons[nested_ks_tuple[-1]]

                # AuxK in fp32, OUTSIDE autocast (spec line 518). Skip the call entirely
                # when no features are dead -- saves the function-call + tensor-alloc cost.
                if dead_mask_dev.any():
                    aux = auxk_loss(
                        x,
                        recon_full,
                        dead_mask_dev,
                        encoder_pre_act,
                        msae.decoder.weight,
                        k_aux=k_aux,
                        alpha_aux=alpha_aux,
                    )
                    total_loss = loss + aux
                else:
                    total_loss = loss

                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                # Decoder unit-norm (single call per step; spec lines 534-539)
                normalize_decoder(msae)

                # Update fire counter on ``device`` -- avoids the per-step
                # CUDA->CPU sync that a ``.cpu()`` per step would force.
                fire_counter.add_((z.detach() > 0).sum(dim=0).long())

                # Refresh dead mask. Single .cpu() at the boundary; prior
                # per-step .cpu() has been hoisted out.
                if (global_step + 1) % dead_refresh_every == 0:
                    threshold = 0.001 * dead_window * batch_size / max_features
                    fire_counter_cpu = fire_counter.cpu()
                    dead_mask = (fire_counter_cpu < threshold).bool()
                    dead_mask_dev = dead_mask.to(device)
                    fire_counter.zero_()
                    logger.debug("Step %d: dead features = %d", global_step, int(dead_mask.sum().item()))

                epoch_loss_sum += total_loss.item()
                epoch_steps += 1
                global_step += 1

            except RuntimeError as e:
                if use_cuda and "out of memory" in str(e).lower():
                    # Spec floor: batch_size=2048. Below that, the model itself
                    # is too large for the GPU -- continuing to halve would
                    # silently turn a sizing bug into a 1000x slowdown.
                    if batch_size <= 2048:
                        raise RuntimeError(
                            f"OOM at batch_size={batch_size} (floor=2048); "
                            "model too large for this GPU"
                        ) from e
                    logger.warning("OOM at batch_size=%d; halving to %d", batch_size, batch_size // 2)
                    torch.cuda.empty_cache()
                    batch_size = batch_size // 2
                    break
                else:
                    raise

        n_epochs_trained += 1

        # Validation -- same autocast envelope as training so per-level MSE numbers
        # are comparable across train and val (spec line 549).
        msae.train(False)
        val_mse_per_level: dict[int, float] = {k: 0.0 for k in nested_ks_tuple}
        val_mse_counts: dict[int, int] = {k: 0 for k in nested_ks_tuple}
        val_l0_sum = 0.0
        val_l0_count = 0

        autocast_ctx = (
            torch.autocast('cuda', dtype=torch.bfloat16) if use_cuda
            else _NullContext()
        )
        with torch.inference_mode(), autocast_ctx:
            for vbatch in val_loader:
                vx = vbatch[0] if isinstance(vbatch, (list, tuple)) else vbatch
                vx = vx.to(device, non_blocking=True).float()
                vz, vrecons = msae(vx)
                for k in nested_ks_tuple:
                    mse_k = F.mse_loss(vrecons[k].float(), vx).item()
                    val_mse_per_level[k] += mse_k * vx.shape[0]
                    val_mse_counts[k] += vx.shape[0]
                val_l0_sum += (vz > 0).float().sum(dim=1).mean().item() * vx.shape[0]
                val_l0_count += vx.shape[0]

        for k in nested_ks_tuple:
            if val_mse_counts[k] > 0:
                val_mse_per_level[k] /= val_mse_counts[k]
        mean_l0 = val_l0_sum / val_l0_count if val_l0_count > 0 else 0.0

        # Alive features per level (sample from training set)
        z_sample: Optional[Tensor] = None
        with torch.inference_mode():
            for sb in _make_loader(train_dataset, min(batch_size, 64), shuffle=False, drop_last=False):
                sx = sb[0] if isinstance(sb, (list, tuple)) else sb
                sx = sx.to(device, non_blocking=True).float()
                z_sample, _ = msae(sx)
                break
        msae.train(True)

        alive_per_level: dict[int, int] = {}
        if z_sample is not None:
            for k in nested_ks_tuple:
                alive_per_level[k] = int((z_sample[:, :k] > 0).any(dim=0).sum().item())
        else:
            for k in nested_ks_tuple:
                alive_per_level[k] = 0

        total_alive = sum(alive_per_level.values())
        avg_val_mse = sum(val_mse_per_level.values()) / len(val_mse_per_level)
        avg_train_loss = epoch_loss_sum / max(1, epoch_steps)

        log_entry = {
            'epoch': epoch,
            'global_step': global_step,
            'train_loss': avg_train_loss,
            'val_mse_per_level': val_mse_per_level,
            'mean_l0': mean_l0,
            'alive_features_per_level': alive_per_level,
            'lam_sparse': lam_sparse,
        }
        log_entries.append(log_entry)

        _atomic_json_dump(log_entries, log_path)

        logger.info(
            "Epoch %d: avg_val_mse=%.4f  mean_l0=%.1f  alive=%d  train_loss=%.4f",
            epoch, avg_val_mse, mean_l0, total_alive, avg_train_loss,
        )

        # Async checkpoint -- the .cpu() copies happen synchronously; the Drive
        # write (5-30s) runs on a background thread and is joined before return.
        # fire_counter + numpy + python RNG are saved for true reproducible
        # resume; ``resume_from_checkpoint`` in this module consumes them.
        ckpt_path = checkpoint_dir / f"msae_epoch_{epoch}.pt"
        async_checkpoint(
            {
                'model_state_dict': msae.state_dict(),
                'optimizer': optimizer.state_dict(),
                'dead_mask': dead_mask,
                'fire_counter': fire_counter,
                'step': global_step,
                'rng_state_torch': torch.get_rng_state(),
                'rng_state_numpy': np.random.get_state(),
                'rng_state_python': random.getstate(),
                'epoch': epoch,
                'lam_sparse_used': lam_sparse,
                # Structural hyperparams. Previously missing — evaluate/visualize
                # fell back to hardcoded production defaults (256,768,3072,12288)
                # which silently mislabeled levels in any run using different
                # nesting (e.g. smaller configs at (128,384,768)).
                'input_dim': msae.input_dim,
                'max_features': msae.max_features,
                'nested_ks': tuple(msae.nested_ks),
            },
            ckpt_path,
        )

        # Early stopping
        improved = False
        if best_val_mse is None or (best_val_mse - avg_val_mse) / max(abs(best_val_mse), 1e-9) > early_stop_threshold:
            best_val_mse = avg_val_mse
            improved = True
        if best_alive_count is None or (total_alive - best_alive_count) / max(abs(best_alive_count), 1) > early_stop_threshold:
            best_alive_count = total_alive
            improved = True

        if improved:
            steps_without_improvement = 0
        else:
            steps_without_improvement += epoch_steps

        if steps_without_improvement >= early_stop_patience:
            logger.info("Early stopping at epoch %d (no improvement for %d steps)", epoch, steps_without_improvement)
            break

    # Final val metrics
    msae.train(False)
    final_val_mse: dict[int, float] = {k: 0.0 for k in nested_ks_tuple}
    final_val_counts: dict[int, int] = {k: 0 for k in nested_ks_tuple}
    final_l0_sum = 0.0
    final_l0_count = 0
    z_alive_accum: Optional[Tensor] = None

    autocast_ctx = (
        torch.autocast('cuda', dtype=torch.bfloat16) if use_cuda
        else _NullContext()
    )
    with torch.inference_mode(), autocast_ctx:
        for vbatch in val_loader:
            vx = vbatch[0] if isinstance(vbatch, (list, tuple)) else vbatch
            vx = vx.to(device, non_blocking=True).float()
            vz, vrecons = msae(vx)
            for k in nested_ks_tuple:
                mse_k = F.mse_loss(vrecons[k].float(), vx).item()
                final_val_mse[k] += mse_k * vx.shape[0]
                final_val_counts[k] += vx.shape[0]
            final_l0_sum += (vz > 0).float().sum(dim=1).mean().item() * vx.shape[0]
            final_l0_count += vx.shape[0]
            if z_alive_accum is None:
                z_alive_accum = (vz > 0).any(dim=0)
            else:
                z_alive_accum = z_alive_accum | (vz > 0).any(dim=0)

    final_alive: dict[int, int] = {}
    for k in nested_ks_tuple:
        if final_val_counts[k] > 0:
            final_val_mse[k] /= final_val_counts[k]
        if z_alive_accum is not None:
            final_alive[k] = int(z_alive_accum[:k].sum().item())
        else:
            final_alive[k] = 0

    final_mean_l0 = final_l0_sum / final_l0_count if final_l0_count > 0 else 0.0

    _join_pending_saves()

    # Write a canonical ``msae_final.pt`` so notebook 03 has a stable filename
    # regardless of early-stop epoch. copy2 preserves the already-async-written
    # per-epoch checkpoint; both files live on Drive.
    checkpoint_dir = Path(checkpoint_dir)
    last_epoch = max(0, n_epochs_trained - 1)
    last_ckpt = checkpoint_dir / f"msae_epoch_{last_epoch}.pt"
    final_path = checkpoint_dir / "msae_final.pt"
    if last_ckpt.exists():
        shutil.copy2(last_ckpt, final_path)
        logger.info("train_msae: wrote final checkpoint to %s", final_path)
    else:
        logger.warning(
            "train_msae: last epoch checkpoint missing (%s); no msae_final.pt written",
            last_ckpt,
        )

    return {
        'final_val_mse_per_level': final_val_mse,
        'alive_features_per_level': final_alive,
        'mean_l0': final_mean_l0,
        'n_epochs_trained': n_epochs_trained,
        'lam_sparse_used': lam_sparse,
    }


def train_standard_sae(
    sae: StandardSAE,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: dict,
    checkpoint_dir: Path,
    log_path: Path,
) -> dict:
    """Training loop for single-level StandardSAE -- same structure as train_msae
    but single level K=sae.n_features (no per-level Matryoshka logic)."""
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Device resolution must happen BEFORE lam_sparse so the probe (which is
    # CUDA-required) can be dispatched on the right device. Auto-move to CUDA
    # when available so a notebook that forgot `.to("cuda")` still works.
    device = next(sae.parameters()).device
    use_cuda = device.type == 'cuda'
    if device.type != 'cuda' and torch.cuda.is_available():
        sae = sae.to('cuda')
        device = torch.device('cuda')
        use_cuda = True
        logger.info("train_standard_sae: moved model to CUDA (was on CPU)")

    lam_sparse = config.get('lam_sparse')
    if lam_sparse is None:
        n_features_probe = sae.n_features
        input_dim_probe = sae.input_dim

        # Local factory captures sae's hparams; the probe constructs fresh
        # instances per candidate so the sweep is independent.
        def _std_factory() -> StandardSAE:
            return StandardSAE(input_dim=input_dim_probe, n_features=n_features_probe)

        lam_sparse = lambda_sparse_probe(
            model_factory=_std_factory,
            dataset=train_dataset,
            seed=seed,
            device=device,
        )
        logger.info("train_standard_sae: lam_sparse resolved by probe: %.1e", lam_sparse)

    lr = config.get('lr', 3e-4)
    batch_size = config.get('batch_size', 16384)
    n_epochs = config.get('n_epochs', 6)
    k_aux = config.get('k_aux', 512)
    alpha_aux = config.get('alpha_aux', 1 / 32)
    dead_window = config.get('dead_window', 50_000)
    dead_refresh_every = config.get('dead_refresh_every', 1500)
    early_stop_patience = config.get('early_stop_patience_steps', 96)
    early_stop_threshold = config.get('early_stop_threshold', 0.01)

    if use_cuda:
        setup_perf_flags()

    def _make_loader(ds: Dataset, bs: int, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=use_cuda,
            num_workers=0,
            persistent_workers=False,
        )

    val_loader = _make_loader(val_dataset, batch_size, shuffle=False, drop_last=False)

    # Init pre_bias from first batch (BEFORE compile). In-place `.copy_()` per
    # the fused-Adam / Dynamo pointer discipline in ``normalize_decoder``.
    for b in _make_loader(train_dataset, batch_size, shuffle=False, drop_last=False):
        first_batch = b[0].float() if isinstance(b, (list, tuple)) else b.float()
        sae.pre_bias.data.copy_(first_batch.mean(dim=0).detach().to(device))
        break

    train_steps_per_epoch = max(1, len(train_dataset) // batch_size)
    total_steps = n_epochs * train_steps_per_epoch

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr, fused=use_cuda)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-5
    )

    n_features = sae.n_features
    input_dim = sae.input_dim

    # Compiled forward+loss closure -- mirrors train_msae but single-level loss.
    def _forward_and_loss(x: Tensor, lam: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pre_act = sae.encoder(x - sae.pre_bias)
        z = F.relu(pre_act)
        recon = z @ sae.decoder.weight.T + sae.decoder.bias + sae.pre_bias
        loss = F.mse_loss(recon, x) + lam * z.abs().mean()
        return z, pre_act, recon, loss

    if use_cuda:
        def _warmup(compiled_fn: Callable[..., Any]) -> None:
            optimizer.zero_grad(set_to_none=True)
            dummy_x = torch.randn(batch_size, input_dim, device=device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, _, _, _w_loss = compiled_fn(dummy_x, lam_sparse)
            _w_loss.backward()
            optimizer.zero_grad(set_to_none=True)
            del dummy_x, _w_loss

        compiled_step = maybe_compile(_forward_and_loss, warmup=_warmup, name="std_sae_step")
    else:
        compiled_step = _forward_and_loss

    fire_counter = torch.zeros(n_features, dtype=torch.long, device=device)
    dead_mask = torch.zeros(n_features, dtype=torch.bool)
    dead_mask_dev = dead_mask.to(device)

    best_val_mse: Optional[float] = None
    steps_without_improvement = 0
    global_step = 0
    n_epochs_trained = 0
    log_entries = []

    avg_val_mse = 0.0
    mean_l0 = 0.0
    alive_count = 0

    for epoch in range(n_epochs):
        sae.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        train_loader = _make_loader(train_dataset, batch_size, shuffle=True, drop_last=True)

        for batch in train_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True).float()

            try:
                if use_cuda:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        z, encoder_pre_act, recon, loss = compiled_step(x, lam_sparse)
                else:
                    z, encoder_pre_act, recon, loss = compiled_step(x, lam_sparse)

                if dead_mask_dev.any():
                    aux = auxk_loss(
                        x,
                        recon,
                        dead_mask_dev,
                        encoder_pre_act,
                        sae.decoder.weight,
                        k_aux=k_aux,
                        alpha_aux=alpha_aux,
                    )
                    total_loss = loss + aux
                else:
                    total_loss = loss

                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                normalize_decoder(sae)

                fire_counter.add_((z.detach() > 0).sum(dim=0).long())

                if (global_step + 1) % dead_refresh_every == 0:
                    threshold = 0.001 * dead_window * batch_size / n_features
                    fire_counter_cpu = fire_counter.cpu()
                    dead_mask = (fire_counter_cpu < threshold).bool()
                    dead_mask_dev = dead_mask.to(device)
                    fire_counter.zero_()

                epoch_loss_sum += total_loss.item()
                epoch_steps += 1
                global_step += 1

            except RuntimeError as e:
                if use_cuda and "out of memory" in str(e).lower():
                    if batch_size <= 2048:
                        raise RuntimeError(
                            f"OOM at batch_size={batch_size} (floor=2048); "
                            "model too large for this GPU"
                        ) from e
                    torch.cuda.empty_cache()
                    batch_size = batch_size // 2
                    logger.warning("OOM -- halved batch_size to %d", batch_size)
                    break
                else:
                    raise

        n_epochs_trained += 1

        # Validation
        sae.train(False)
        val_mse_sum = 0.0
        val_mse_count = 0
        val_l0_sum = 0.0
        val_l0_count = 0
        val_alive_accum: Optional[Tensor] = None

        autocast_ctx = (
            torch.autocast('cuda', dtype=torch.bfloat16) if use_cuda
            else _NullContext()
        )
        with torch.inference_mode(), autocast_ctx:
            for vbatch in val_loader:
                vx = vbatch[0] if isinstance(vbatch, (list, tuple)) else vbatch
                vx = vx.to(device, non_blocking=True).float()
                vz, vrecon = sae(vx)
                val_mse_sum += F.mse_loss(vrecon.float(), vx).item() * vx.shape[0]
                val_mse_count += vx.shape[0]
                val_l0_sum += (vz > 0).float().sum(dim=1).mean().item() * vx.shape[0]
                val_l0_count += vx.shape[0]
                if val_alive_accum is None:
                    val_alive_accum = (vz > 0).any(dim=0)
                else:
                    val_alive_accum = val_alive_accum | (vz > 0).any(dim=0)

        avg_val_mse = val_mse_sum / val_mse_count if val_mse_count > 0 else 0.0
        mean_l0 = val_l0_sum / val_l0_count if val_l0_count > 0 else 0.0
        alive_count = int(val_alive_accum.sum().item()) if val_alive_accum is not None else 0
        avg_train_loss = epoch_loss_sum / max(1, epoch_steps)

        log_entry = {
            'epoch': epoch,
            'global_step': global_step,
            'train_loss': avg_train_loss,
            'val_mse': avg_val_mse,
            'mean_l0': mean_l0,
            'alive_features': alive_count,
            'lam_sparse': lam_sparse,
        }
        log_entries.append(log_entry)

        _atomic_json_dump(log_entries, log_path)

        ckpt_path = checkpoint_dir / f"sae_epoch_{epoch}.pt"
        async_checkpoint(
            {
                'model_state_dict': sae.state_dict(),
                'optimizer': optimizer.state_dict(),
                'dead_mask': dead_mask,
                'fire_counter': fire_counter,
                'step': global_step,
                'rng_state_torch': torch.get_rng_state(),
                'rng_state_numpy': np.random.get_state(),
                'rng_state_python': random.getstate(),
                'epoch': epoch,
                'lam_sparse_used': lam_sparse,
                # Structural hyperparams (see train_msae for rationale).
                'input_dim': sae.input_dim,
                'n_features': sae.n_features,
            },
            ckpt_path,
        )

        # Early stopping
        improved = False
        if best_val_mse is None or (best_val_mse - avg_val_mse) / max(abs(best_val_mse), 1e-9) > early_stop_threshold:
            best_val_mse = avg_val_mse
            improved = True

        if improved:
            steps_without_improvement = 0
        else:
            steps_without_improvement += epoch_steps

        if steps_without_improvement >= early_stop_patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    sae.train(True)
    _join_pending_saves()

    # Canonical final-checkpoint filename so notebook 03 can load without
    # knowing which epoch early-stop fired on.
    checkpoint_dir = Path(checkpoint_dir)
    last_epoch = max(0, n_epochs_trained - 1)
    last_ckpt = checkpoint_dir / f"sae_epoch_{last_epoch}.pt"
    final_path = checkpoint_dir / "sae_final.pt"
    if last_ckpt.exists():
        shutil.copy2(last_ckpt, final_path)
        logger.info("train_standard_sae: wrote final checkpoint to %s", final_path)
    else:
        logger.warning(
            "train_standard_sae: last epoch checkpoint missing (%s); no sae_final.pt written",
            last_ckpt,
        )

    return {
        'final_val_mse': avg_val_mse,
        'alive_features': alive_count,
        'mean_l0': mean_l0,
        'n_epochs_trained': n_epochs_trained,
        'lam_sparse_used': lam_sparse,
    }


def train_linear_probe(
    cls_acts_path: Path,
    labels_df: pd.DataFrame,
    output_path: Path,
    n_epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    """Train a LinearProbe on CLS activations; gate val_acc >= 0.50.

    labels_df must have columns: image_id, class_name.
    cls_acts_path: path to .pt file with key 'cls' -- shape (N, 768).
    """
    from sklearn.model_selection import train_test_split

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load CLS activations
    data = torch.load(cls_acts_path, weights_only=True)
    cls_acts = data['cls']  # (N, 768)

    # Contract: CLS file holds one row per image, and labels_df is that same
    # per-image table. Any filtering of labels_df without re-extracting CLS will
    # silently mis-align labels here; fail loud.
    assert len(cls_acts) == len(labels_df), (
        f"cls_acts has {len(cls_acts)} rows, labels_df has {len(labels_df)} -- "
        "must match 1:1 (per-image)"
    )

    # Map image_ids to class labels
    class_names = labels_df['class_name'].values
    unique_classes = sorted(set(class_names))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    class_labels = np.array([class_to_idx[c] for c in class_names])

    # Stratified 80/20 split
    indices = np.arange(len(cls_acts))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=class_labels, random_state=seed
    )

    X_train = cls_acts[train_idx].float()
    X_val = cls_acts[val_idx].float()
    y_train = torch.tensor(class_labels[train_idx], dtype=torch.long)
    y_val = torch.tensor(class_labels[val_idx], dtype=torch.long)

    input_dim = cls_acts.shape[1]
    num_classes = len(unique_classes)
    probe = LinearProbe(input_dim=input_dim, num_classes=num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probe = probe.to(device)
    X_train = X_train.to(device)
    X_val = X_val.to(device)
    y_train = y_train.to(device)
    y_val = y_val.to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, fused=False)
    use_cuda = device.type == 'cuda'

    epoch_log = []
    for epoch in range(n_epochs):
        probe.train()
        if use_cuda:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = probe(X_train)
                loss = F.cross_entropy(logits, y_train)
        else:
            logits = probe(X_train)
            loss = F.cross_entropy(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probe.training = False
        with torch.no_grad():
            train_preds = probe(X_train).argmax(dim=1)
            train_acc = (train_preds == y_train).float().mean().item()
            val_preds = probe(X_val).argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()

        epoch_log.append({'epoch': epoch, 'train_acc': train_acc, 'val_acc': val_acc})
        logger.info("LinearProbe epoch %d: train_acc=%.3f  val_acc=%.3f", epoch, train_acc, val_acc)

    # Per-class val accuracy
    val_acc_per_class: dict[str, float] = {}
    probe.training = False
    with torch.no_grad():
        val_preds_final = probe(X_val).argmax(dim=1)
        for idx, cls_name in enumerate(unique_classes):
            mask = y_val == idx
            if mask.sum().item() > 0:
                val_acc_per_class[cls_name] = (val_preds_final[mask] == y_val[mask]).float().mean().item()
            else:
                val_acc_per_class[cls_name] = float('nan')

    final_val_acc = epoch_log[-1]['val_acc'] if epoch_log else 0.0

    # Gate
    if final_val_acc < 0.50:
        raise RuntimeError(
            f"Linear probe val_acc={final_val_acc:.3f} < 0.50 gate. "
            "Check feature extraction."
        )

    torch.save(probe.state_dict(), output_path)

    return {
        'train_acc': epoch_log[-1]['train_acc'] if epoch_log else 0.0,
        'val_acc': final_val_acc,
        'val_acc_per_class': val_acc_per_class,
        'n_classes': num_classes,
    }
