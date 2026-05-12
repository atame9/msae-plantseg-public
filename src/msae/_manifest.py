from __future__ import annotations

import json
import logging
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _git_sha() -> str | None:
    """Return the current HEAD SHA, or None if git isn't available / not a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _git_dirty() -> bool | None:
    """Return True if the working tree has uncommitted changes, None on error."""
    try:
        result = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode not in (0, 1):
        return None
    return result.returncode == 1


def _device_name() -> str | None:
    """Return the current CUDA device name, or None on CPU / unavailable."""
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


def write_manifest(
    out_dir: Path,
    argv: list[str],
    start_ts: float,
    end_ts: float,
    exit_status: int,
    resolved_config: dict[str, Any] | None = None,
) -> None:
    """Write ``<out_dir>/run_manifest.json`` with provenance for this CLI run.

    Captures argv, git state, torch/CUDA versions, hostname, timestamps, and the
    resolved training config (``train`` subcommand only). Called from a
    ``finally`` block so it fires on both success and failure. Directory is
    created if missing so failures that bail before the handler runs still get
    a manifest.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "argv": list(argv),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_name": _device_name(),
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "exit_status": exit_status,
        "resolved_config": resolved_config,
    }

    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))


def write_pip_freeze(out_dir: Path) -> None:
    """Write ``<out_dir>/pip_freeze.txt`` with the current environment packages.

    Uses ``sys.executable -m pip freeze`` so the correct interpreter's
    environment is captured (matters when running under a uv / venv).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pip_freeze.txt"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=False,
        )
        path.write_text(result.stdout if result.returncode == 0 else "")
    except (FileNotFoundError, OSError) as exc:
        logger.warning("pip freeze failed: %s", exc)
        path.write_text("")
