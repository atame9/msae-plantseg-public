"""Minimal CI contract for the msae CLI.

Tests cover the pure-Python CLI surface:
- argparse wiring (help text lists all subcommands)
- config merge precedence (defaults < file < CLI)
- unknown-key rejection in JSON configs
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from msae._config import _MSAE_DEFAULTS, resolve_config


def test_help_text_mentions_all_subcommands() -> None:
    """`python -m msae.cli --help` must list every subcommand name."""
    result = subprocess.run(
        [sys.executable, "-m", "msae.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    out = result.stdout
    for sub in ("validate-data", "extract", "train", "evaluate", "visualize"):
        assert sub in out, f"--help output missing subcommand {sub!r}: {out}"


def test_config_precedence_file_over_defaults_then_cli_over_file(tmp_path: Path) -> None:
    """defaults < file < CLI — the only precedence that matters."""
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"lr": 1e-5}))

    # CLI dict mimics an argparse namespace: None = "flag not passed".
    cli = {"lr": 1e-6, "batch_size": None, "n_epochs": None}

    resolved = resolve_config("msae", cfg_path, cli)

    # CLI wins over file
    assert resolved["lr"] == 1e-6
    # File value survives for keys not set on CLI
    cli_only_file = {"batch_size": None}
    resolved_file = resolve_config("msae", cfg_path, cli_only_file)
    assert resolved_file["lr"] == 1e-5
    # Default survives when neither file nor CLI provide a value
    assert resolved_file["batch_size"] == _MSAE_DEFAULTS["batch_size"]


def test_config_unknown_key_raises(tmp_path: Path) -> None:
    """Typos like `learnig_rate` must fail loudly, not silently."""
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"learnig_rate": 1e-4}))
    with pytest.raises(ValueError, match="Unknown config keys"):
        resolve_config("msae", cfg_path, {})
