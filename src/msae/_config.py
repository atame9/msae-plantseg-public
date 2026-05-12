from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Source of truth for defaults lives in code. The JSON files under configs/ are
# kept in sync by the test_defaults_json_matches_code_constants check.

_MSAE_DEFAULTS: dict[str, Any] = {
    "lr": 3e-4,
    "batch_size": 16384,
    # n_epochs bumped 6 → 20 (C2a.2). At batch 16384, PlantSeg yields
    # ~191 train steps/epoch. 6 epochs = 1146 steps < dead_refresh_every
    # (1500), so the dead-feature refresh never fired and AuxK was dead
    # code. 20 epochs × 191 = ~3820 steps gives AuxK ~8 refresh cycles.
    "n_epochs": 20,
    "lam_sparse": None,
    "k_aux": 512,
    "alpha_aux": 0.03125,  # exactly 1/32 in binary floating point
    "dead_window": 50000,
    # dead_refresh_every 1500 → 500 (C2a.2). First refresh now at step 500
    # (early epoch 3) instead of never. 500 = ~8 refreshes across 3820-step
    # training run — the regime AuxK was designed for.
    "dead_refresh_every": 500,
    # early_stop_patience_steps 96 → 600 (C2a.2). Loop at train.py:717
    # increments steps_without_improvement by epoch_steps (~191) in one
    # shot per non-improving epoch, so patience=96 fired after any single
    # stalled epoch. 600 = ~3 epochs of plateau runway; matches the
    # bumped epoch count.
    "early_stop_patience_steps": 600,
    "early_stop_threshold": 0.01,
    "nested_ks": [256, 768, 3072, 12288],
    "max_features": 12288,
    "input_dim": 768,
    "seed": 42,
}

_STANDARD_DEFAULTS: dict[str, Any] = {
    "lr": 3e-4,
    "batch_size": 16384,
    # See _MSAE_DEFAULTS for rationale. Both configs move together so the
    # MSAE-vs-StandardSAE comparison stays valid under identical training
    # regimes.
    "n_epochs": 20,
    "lam_sparse": None,
    "k_aux": 512,
    "alpha_aux": 0.03125,
    "dead_window": 50000,
    "dead_refresh_every": 500,
    "early_stop_patience_steps": 600,
    "early_stop_threshold": 0.01,
    "n_features": 12288,
    "input_dim": 768,
    "seed": 42,
}


def _defaults_for(model: str) -> dict[str, Any]:
    if model == "msae":
        return _MSAE_DEFAULTS
    if model == "standard":
        return _STANDARD_DEFAULTS
    raise ValueError(f"Unknown model {model!r}; expected 'msae' or 'standard'")


def resolve_config(
    model: str,
    config_path: Path | str | None,
    cli_values: dict[str, Any],
) -> dict[str, Any]:
    """Merge defaults, optional JSON config file, and explicit CLI flags.

    Precedence (low → high): ``_DEFAULTS`` < ``config_path`` < ``cli_values``
    where ``cli_values`` only contributes entries whose value is not ``None``
    (argparse sentinel for "user didn't pass the flag").

    Unknown keys in the config file raise ``ValueError``. Typos like
    ``"learnig_rate"`` fail loudly, not silently.

    Parameters
    ----------
    model : {'msae', 'standard'}
        Selects which default set and which legal-key set to use.
    config_path : Path | str | None
        Optional JSON file with overrides. Passed through ``json.loads``.
    cli_values : dict
        All train-relevant flags from the parsed argparse namespace. Values
        that are ``None`` are treated as "not provided" and skipped.

    Returns
    -------
    dict
        The fully merged config, ready to pass into ``train.train_msae`` or
        ``train.train_standard_sae``.
    """
    defaults = _defaults_for(model)

    file_config: dict[str, Any] = {}
    if config_path is not None:
        config_path = Path(config_path)
        try:
            file_config = json.loads(config_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{config_path}: not valid JSON ({exc.msg} at line {exc.lineno})"
            ) from exc
        if not isinstance(file_config, dict):
            raise ValueError(
                f"{config_path}: top-level JSON must be an object, got "
                f"{type(file_config).__name__}"
            )

        unknown = sorted(set(file_config) - set(defaults))
        if unknown:
            raise ValueError(
                f"Unknown config keys: {unknown}; "
                f"known keys: {sorted(defaults)}"
            )

    resolved: dict[str, Any] = {**defaults, **file_config}
    for key, value in cli_values.items():
        if value is None:
            continue
        if key not in defaults:
            # CLI flags are restricted by argparse, so anything reaching here
            # that isn't in defaults is a programming error -- fail fast.
            raise ValueError(
                f"CLI flag {key!r} has no matching config key for model {model!r}"
            )
        resolved[key] = value

    # Normalise types that differ between CLI (string/list-of-int) and config (list):
    #   --nested-ks "256,768,3072,12288" → [256, 768, 3072, 12288]
    if "nested_ks" in resolved and isinstance(resolved["nested_ks"], str):
        resolved["nested_ks"] = [int(x) for x in resolved["nested_ks"].split(",")]

    return resolved
