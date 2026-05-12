"""Command-line entrypoint for the msae package.

Run via ``python -m msae.cli <sub> <flags>`` or the ``msae`` console script.

Six subcommands:
- ``validate-data``  pre-flight check on PlantSeg / PlantVillage directories
- ``extract``        DINOv2 layer-8 activation extraction to chunk files
- ``probe``          LinearProbe on CLS; pre-training val_acc >= 0.50 gate
- ``train``          MSAE or StandardSAE training loop
- ``evaluate``       metric computation (selectivity, IoU, MI, transfer)
- ``visualize``      figure rendering from evaluate outputs

Every compute subcommand writes ``<out>/run_manifest.json`` and
``<out>/pip_freeze.txt`` on both success and failure.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch

from msae._config import resolve_config
from msae._manifest import write_manifest, write_pip_freeze

logger = logging.getLogger("msae.cli")

# Keys the CLI accepts as train flags. Kept in sync with the argparse setup
# below and with the _MSAE_DEFAULTS / _STANDARD_DEFAULTS sets in _config.py.
_TRAIN_CLI_KEYS_COMMON: tuple[str, ...] = (
    "lr",
    "batch_size",
    "n_epochs",
    "lam_sparse",
    "k_aux",
    "alpha_aux",
    "dead_window",
    "dead_refresh_every",
    "early_stop_patience_steps",
    "early_stop_threshold",
    "input_dim",
    "seed",
)
_TRAIN_CLI_KEYS_MSAE: tuple[str, ...] = _TRAIN_CLI_KEYS_COMMON + (
    "nested_ks",
    "max_features",
)
_TRAIN_CLI_KEYS_STANDARD: tuple[str, ...] = _TRAIN_CLI_KEYS_COMMON + ("n_features",)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msae",
        description="Matryoshka Sparse Autoencoder pipeline CLI.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # --- validate-data -----------------------------------------------------
    p_val = subparsers.add_parser(
        "validate-data",
        help="Pre-flight check on PlantSeg and PlantVillage directories.",
    )
    p_val.add_argument("--plantseg-images", type=Path, required=True)
    p_val.add_argument("--plantseg-masks", type=Path, required=True)
    p_val.add_argument("--plantvillage-images", type=Path, required=True)
    p_val.add_argument(
        "--strict",
        action="store_true",
        help="Require every PlantSeg image to have a matching mask.",
    )

    # --- extract -----------------------------------------------------------
    p_ext = subparsers.add_parser(
        "extract", help="Extract DINOv2 layer activations to chunk files."
    )
    p_ext.add_argument(
        "--dataset", choices=["plantseg", "plantvillage"], required=True
    )
    p_ext.add_argument("--images", type=Path, required=True)
    p_ext.add_argument(
        "--masks", type=Path, default=None,
        help="Masks directory (PlantSeg only).",
    )
    p_ext.add_argument(
        "--pv-labels", type=Path, default=None,
        help="Existing plantvillage_labels.csv for cross-dataset class alignment.",
    )
    p_ext.add_argument("--out", type=Path, required=True)
    p_ext.add_argument("--layer", type=int, default=8)
    p_ext.add_argument("--batch-size", type=int, default=256)
    p_ext.add_argument("--chunk-size-images", type=int, default=5000)
    p_ext.add_argument("--percentile", type=float, default=0.20)
    p_ext.add_argument("--no-cls", action="store_true")
    p_ext.add_argument("--no-l2-filter", action="store_true")
    p_ext.add_argument("--no-resume", action="store_true")
    p_ext.add_argument("--device", default=None)
    p_ext.add_argument("--seed", type=int, default=42)

    # --- probe -------------------------------------------------------------
    # Pre-training sanity gate: trains a LinearProbe on CLS activations from
    # `extract` and fails loud if val accuracy < 0.50 (see
    # train.train_linear_probe). Run between `extract` and `train` in the
    # fixture / paper pipeline so cheap feature-extraction bugs show up
    # before spending GPU time on SAE training.
    p_pr = subparsers.add_parser(
        "probe",
        help="Train a linear probe on CLS activations; gates on val_acc >= 0.50.",
    )
    p_pr.add_argument("--cls", type=Path, required=True, help="Path to cls.pt.")
    p_pr.add_argument(
        "--labels", type=Path, required=True,
        help="plantseg_labels.csv (must align 1:1 with cls.pt rows).",
    )
    p_pr.add_argument("--out", type=Path, required=True)
    p_pr.add_argument(
        "--n-epochs", type=int, default=200,
        help=(
            "Full-batch Adam steps on the (probe, label) pair. Default 200; "
            "the 2026-05-11 fixture resume found 20 epochs under-converged "
            "val_acc (12.5%% on 115 classes, gate fail) because the linear "
            "weights simply hadn't moved far enough yet. 200-500 epochs of "
            "full-batch Adam at 1e-3 is reliably ≥85%% for DINOv2 CLS on "
            "PlantSeg."
        ),
    )
    p_pr.add_argument("--lr", type=float, default=1e-3)
    p_pr.add_argument("--seed", type=int, default=42)

    # --- train -------------------------------------------------------------
    p_tr = subparsers.add_parser(
        "train", help="Train an MSAE or StandardSAE on extracted activations."
    )
    p_tr.add_argument("--model", choices=["msae", "standard"], required=True)
    p_tr.add_argument(
        "--acts", type=Path, required=True,
        help="Directory containing patches.pt (or chunk files to consolidate).",
    )
    p_tr.add_argument("--out", type=Path, required=True)
    p_tr.add_argument("--config", type=Path, default=None)
    # All train flags below have default=None so resolve_config can tell
    # "user passed the flag" from "user didn't pass it". See _config.py.
    p_tr.add_argument("--lr", type=float, default=None)
    p_tr.add_argument("--batch-size", type=int, default=None)
    p_tr.add_argument("--n-epochs", type=int, default=None)
    p_tr.add_argument("--lam-sparse", type=float, default=None)
    p_tr.add_argument(
        "--lam-sparse-probe", action="store_true",
        help="Force lam_sparse resolution via the 3-point probe (CUDA only).",
    )
    p_tr.add_argument("--k-aux", type=int, default=None)
    p_tr.add_argument("--alpha-aux", type=float, default=None)
    p_tr.add_argument("--dead-window", type=int, default=None)
    p_tr.add_argument("--dead-refresh-every", type=int, default=None)
    p_tr.add_argument("--early-stop-patience-steps", type=int, default=None)
    p_tr.add_argument("--early-stop-threshold", type=float, default=None)
    p_tr.add_argument(
        "--nested-ks", type=str, default=None,
        help="MSAE only. Comma-separated ints, e.g. '256,768,3072,12288'.",
    )
    p_tr.add_argument("--max-features", type=int, default=None, help="MSAE only.")
    p_tr.add_argument("--n-features", type=int, default=None, help="StandardSAE only.")
    p_tr.add_argument("--input-dim", type=int, default=None)
    p_tr.add_argument("--seed", type=int, default=None)
    p_tr.add_argument("--device", default=None)
    p_tr.add_argument("--resume", type=Path, default=None)

    # --- evaluate ----------------------------------------------------------
    p_ev = subparsers.add_parser(
        "evaluate", help="Compute selectivity / IoU / MI / transfer metrics."
    )
    p_ev.add_argument("--acts", type=Path, required=True, help="Path to patches.pt.")
    p_ev.add_argument("--meta", type=Path, required=True, help="Path to meta.pt.")
    p_ev.add_argument("--msae-ckpt", type=Path, required=True)
    p_ev.add_argument("--standard-ckpt", type=Path, default=None)
    p_ev.add_argument("--pca-result", type=Path, default=None)
    p_ev.add_argument("--plantseg-labels", type=Path, required=True)
    p_ev.add_argument("--plantvillage-acts", type=Path, default=None)
    p_ev.add_argument("--plantvillage-meta", type=Path, default=None)
    p_ev.add_argument("--plantvillage-labels", type=Path, default=None)
    p_ev.add_argument("--class-alignment-csv", type=Path, default=None)
    p_ev.add_argument("--masks", type=Path, default=None)
    p_ev.add_argument("--out", type=Path, required=True)
    p_ev.add_argument("--encode-batch-size", type=int, default=16384)
    p_ev.add_argument("--iou-feature-chunk", type=int, default=512)
    p_ev.add_argument("--mi-max-samples", type=int, default=50000)
    p_ev.add_argument("--device", default=None)
    p_ev.add_argument("--seed", type=int, default=42)
    p_ev.add_argument(
        "--no-fail-on-layer6-fallback",
        action="store_true",
        help=(
            "Do not raise SystemExit if the layer-6 fallback diagnostic triggers. "
            "The diagnostic JSON is still written unchanged. Useful for smoke "
            "tests with minimal data where the MSAE cannot meet the selectivity "
            "gap threshold. Do not set this for production runs — the guard "
            "exists to catch real layer-choice regressions in trained models."
        ),
    )

    # --- visualize ---------------------------------------------------------
    p_viz = subparsers.add_parser(
        "visualize", help="Render figures from evaluate outputs."
    )
    p_viz.add_argument("--eval-results", type=Path, required=True)
    p_viz.add_argument("--msae-ckpt", type=Path, required=True)
    p_viz.add_argument("--acts", type=Path, required=True)
    p_viz.add_argument("--meta", type=Path, required=True)
    p_viz.add_argument("--plantseg-labels", type=Path, required=True)
    p_viz.add_argument("--out", type=Path, required=True)
    p_viz.add_argument("--top-n-features", type=int, default=5)
    p_viz.add_argument("--heatmap-ids", type=str, default=None)
    p_viz.add_argument("--image-root", type=Path, default=None)
    p_viz.add_argument("--device", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    argv = argv if argv is not None else sys.argv[1:]
    parser = _build_parser()
    ns = parser.parse_args(argv)

    start_ts = time.time()
    exit_status = 0
    handler = _SUBCOMMANDS[ns.subcommand]
    try:
        handler(ns)
    except SystemExit as e:
        # Propagate explicit sys.exit calls (handlers use them for user errors)
        exit_status = int(e.code) if isinstance(e.code, int) else 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("CLI failed: %s", exc)
        exit_status = 1
    finally:
        end_ts = time.time()
        out_dir = getattr(ns, "out", None)
        if out_dir is not None:
            try:
                write_manifest(
                    Path(out_dir),
                    list(sys.argv),
                    start_ts,
                    end_ts,
                    exit_status,
                    resolved_config=getattr(ns, "_resolved_config", None),
                )
                write_pip_freeze(Path(out_dir))
            except Exception:  # noqa: BLE001
                logger.exception("manifest write failed")

    return exit_status


# ---------------------------------------------------------------------------
# validate-data handler
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _count_images(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(
        1 for p in d.rglob("*") if p.suffix.lower() in _IMAGE_EXTS and p.is_file()
    )


def _handle_validate_data(ns: argparse.Namespace) -> None:
    ps_images = Path(ns.plantseg_images)
    ps_masks = Path(ns.plantseg_masks)
    pv_images = Path(ns.plantvillage_images)

    for label, path in (
        ("plantseg-images", ps_images),
        ("plantseg-masks", ps_masks),
        ("plantvillage-images", pv_images),
    ):
        if not path.exists() or not path.is_dir():
            raise SystemExit(f"validate-data: {label} missing or not a directory: {path}")

    for label, path in (
        ("plantseg-images", ps_images),
        ("plantseg-masks", ps_masks),
        ("plantvillage-images", pv_images),
    ):
        n = _count_images(path)
        if n == 0:
            raise SystemExit(f"validate-data: {label} contains no image files: {path}")
        logger.info("%s: %d images", label, n)

    # PlantSeg: at least one class dir uses single-underscore (no triple)
    ps_classes = [d.name for d in ps_images.iterdir() if d.is_dir()]
    if not any("_" in c and "___" not in c for c in ps_classes):
        raise SystemExit(
            "validate-data: no PlantSeg class directory uses single-underscore "
            "Species_Disease naming (found: {})".format(ps_classes[:5])
        )

    # PlantVillage: at least one class dir uses triple-underscore
    pv_classes = [d.name for d in pv_images.iterdir() if d.is_dir()]
    if not any("___" in c for c in pv_classes):
        raise SystemExit(
            "validate-data: no PlantVillage class directory uses triple-underscore "
            "Species___Disease naming (found: {})".format(pv_classes[:5])
        )

    # Strict mode: every plantseg image has a matching mask.
    if ns.strict:
        missing: list[Path] = []
        for img in ps_images.rglob("*"):
            if img.suffix.lower() not in _IMAGE_EXTS or not img.is_file():
                continue
            rel = img.relative_to(ps_images)
            # Mask may share the image extension or be .png.
            candidates = [
                ps_masks / rel,
                (ps_masks / rel).with_suffix(".png"),
            ]
            if not any(c.exists() for c in candidates):
                missing.append(rel)
            if len(missing) > 20:
                break
        if missing:
            raise SystemExit(
                "validate-data --strict: {} images missing masks (first: {})".format(
                    len(missing), missing[0]
                )
            )
        logger.info("strict mode: every PlantSeg image has a matching mask")
    else:
        # Default: at least one mask must exist.
        n_masks = _count_images(ps_masks)
        if n_masks == 0:
            raise SystemExit(
                "validate-data: plantseg-masks contains zero images; use --strict to check per-image"
            )

    # Class alignment stats.
    from msae import data as msae_data

    ps_df = msae_data.parse_plantseg_metadata(ps_images, ps_masks)
    pv_df = msae_data.parse_plantvillage_metadata(pv_images)
    _, n_overlap = msae_data.build_class_alignment(ps_df, pv_df)
    logger.info("class_alignment n_overlap=%d", n_overlap)
    if n_overlap < 8:
        logger.warning(
            "n_overlap=%d < 8 — transfer_correlation will degrade to qualitative mode",
            n_overlap,
        )


# ---------------------------------------------------------------------------
# extract handler
# ---------------------------------------------------------------------------

def _handle_extract(ns: argparse.Namespace) -> None:
    import torch

    from msae import data as msae_data
    from msae import extraction as msae_ext

    out_dir = Path(ns.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = ns.device or _default_device()

    torch.manual_seed(ns.seed)

    if ns.dataset == "plantseg":
        if ns.masks is None:
            raise SystemExit("extract --dataset plantseg requires --masks")
        df = msae_data.parse_plantseg_metadata(Path(ns.images), Path(ns.masks))
        labels_csv = out_dir / "plantseg_labels.csv"
    else:
        df = msae_data.parse_plantvillage_metadata(Path(ns.images))
        labels_csv = out_dir / "plantvillage_labels.csv"

    df.to_csv(labels_csv, index=False)
    logger.info("wrote labels: %s (rows=%d)", labels_csv, len(df))

    if ns.dataset == "plantseg":
        # save_masks_tensor matches mask_paths 1:1 with image_ids; drop rows
        # without masks before the call so the zip is correct.
        aligned = [
            (row["image_id"], row["mask_filepath"])
            for _, row in df.iterrows()
            if isinstance(row.get("mask_filepath"), str) and row.get("mask_filepath")
        ]
        if aligned:
            masks_out = out_dir / "masks_16x16.pt"
            msae_data.save_masks_tensor(
                [Path(p) for _, p in aligned],
                masks_out,
                image_ids=[iid for iid, _ in aligned],
            )
            logger.info("wrote masks: %s (n=%d)", masks_out, len(aligned))

    # Build class_alignment.csv if PlantVillage labels are available.
    pv_labels_path: Path | None = None
    if ns.dataset == "plantseg":
        if ns.pv_labels is not None:
            pv_labels_path = Path(ns.pv_labels)
        else:
            candidate = out_dir.parent / "plantvillage_activations" / "plantvillage_labels.csv"
            if candidate.exists():
                pv_labels_path = candidate
        if pv_labels_path is not None and pv_labels_path.exists():
            import pandas as pd
            pv_df = pd.read_csv(pv_labels_path)
            align_df, n_overlap = msae_data.build_class_alignment(
                df, pv_df, save_path=out_dir / "class_alignment.csv"
            )
            logger.info("wrote class_alignment.csv (n_overlap=%d)", n_overlap)

    # Build the image dataset and run extraction.
    dataset = msae_data.make_image_dataset(df, Path(ns.images))

    model = msae_ext.setup_dinov2(device=device)

    msae_ext.extract_activations(
        model=model,
        dataset=dataset,
        layer_idx=ns.layer,
        batch_size=ns.batch_size,
        output_dir=out_dir,
        chunk_size_images=ns.chunk_size_images,
        apply_l2_filter=not ns.no_l2_filter,
        save_cls=not ns.no_cls,
        resume=not ns.no_resume,
        device=device,
    )

    # Consolidate chunk files into canonical patches.pt / meta.pt / cls.pt.
    result = msae_ext.consolidate_chunks(out_dir)
    logger.info("consolidated: %s", result)


# ---------------------------------------------------------------------------
# train handler
# ---------------------------------------------------------------------------

def _resolve_acts_tensor(acts_dir: Path) -> torch.Tensor:
    """Load patches tensor from ``acts_dir/patches.pt`` or consolidate chunks."""
    from msae import extraction as msae_ext

    patches_path = acts_dir / "patches.pt"
    if not patches_path.exists():
        logger.info("patches.pt missing; consolidating chunks in %s", acts_dir)
        msae_ext.consolidate_chunks(acts_dir)
    # weights_only=True: patches.pt only holds a tensor, no RNG pickles.
    data = torch.load(patches_path, map_location="cpu", weights_only=True)
    return data["patches"]


# ---------------------------------------------------------------------------
# probe handler
# ---------------------------------------------------------------------------


def _handle_probe(ns: argparse.Namespace) -> None:
    import pandas as pd

    from msae.train import train_linear_probe

    out_dir = Path(ns.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(ns.labels)
    # train_linear_probe raises RuntimeError if val_acc < 0.50. We let that
    # propagate so main()'s top-level handler records a non-zero exit in the
    # run manifest — the probe is a pre-training gate.
    result = train_linear_probe(
        cls_acts_path=Path(ns.cls),
        labels_df=labels_df,
        output_path=out_dir / "linear_probe.pt",
        n_epochs=ns.n_epochs,
        lr=ns.lr,
        seed=ns.seed,
    )
    (out_dir / "probe_result.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    logger.info(
        "probe: val_acc=%.3f (n_classes=%d)",
        result["val_acc"],
        result["n_classes"],
    )


def _handle_train(ns: argparse.Namespace) -> None:
    import numpy as np
    from torch.utils.data import TensorDataset

    from msae import train as msae_train
    from msae.models import MatryoshkaSAE, StandardSAE

    out_dir = Path(ns.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = _TRAIN_CLI_KEYS_MSAE if ns.model == "msae" else _TRAIN_CLI_KEYS_STANDARD
    cli_values: dict[str, Any] = {k: getattr(ns, k) for k in keys}

    # --lam-sparse-probe is a boolean "force the probe" override; we express
    # that as lam_sparse=None so downstream train_msae runs the probe.
    if ns.lam_sparse_probe:
        cli_values["lam_sparse"] = None

    resolved = resolve_config(ns.model, ns.config, cli_values)
    ns._resolved_config = resolved

    device = ns.device or _default_device()

    acts_dir = Path(ns.acts)
    patches = _resolve_acts_tensor(acts_dir)
    # Training loops use float activations; keep them fp32 on the host and let
    # the loaders move them device-side.
    patches = patches.float()

    # Deterministic 95/5 split by seed.
    rng = np.random.default_rng(resolved["seed"])
    n = patches.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, n // 20)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_ds = TensorDataset(patches[train_idx])
    val_ds = TensorDataset(patches[val_idx])

    input_dim = resolved["input_dim"]
    if ns.model == "msae":
        nested_ks = resolved["nested_ks"]
        if isinstance(nested_ks, str):
            nested_ks = [int(x) for x in nested_ks.split(",")]
        model: torch.nn.Module = MatryoshkaSAE(
            input_dim=input_dim,
            max_features=resolved["max_features"],
            nested_ks=tuple(nested_ks),
        )
    else:
        model = StandardSAE(
            input_dim=input_dim, n_features=resolved["n_features"]
        )

    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")

    # Optional resume.
    if ns.resume is not None:
        logger.info("resuming from %s", ns.resume)
        _extra = msae_train.resume_from_checkpoint(ns.resume, model)
        logger.info("resume returned extras: keys=%s", sorted(_extra))

    log_name = "msae_log.json" if ns.model == "msae" else "standard_log.json"
    log_path = out_dir / log_name

    # Strip keys train_* doesn't accept (they're CLI-only book-keeping).
    train_cfg = {k: v for k, v in resolved.items() if k not in {
        "input_dim", "nested_ks", "max_features", "n_features"
    }}

    if ns.model == "msae":
        result = msae_train.train_msae(
            msae=model,  # type: ignore[arg-type]
            train_dataset=train_ds,
            val_dataset=val_ds,
            config=train_cfg,
            checkpoint_dir=out_dir,
            log_path=log_path,
        )
    else:
        result = msae_train.train_standard_sae(
            sae=model,  # type: ignore[arg-type]
            train_dataset=train_ds,
            val_dataset=val_ds,
            config=train_cfg,
            checkpoint_dir=out_dir,
            log_path=log_path,
        )

    # Dump the training result summary alongside the checkpoint so the run dir
    # is self-describing.
    summary_path = out_dir / f"{ns.model}_result.json"
    summary_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("wrote summary: %s", summary_path)


# ---------------------------------------------------------------------------
# evaluate handler
# ---------------------------------------------------------------------------

def _torch_load_trusted(path: Path) -> Any:
    """Load a checkpoint we produced ourselves.

    weights_only=False: our checkpoints carry numpy/python RNG state pickles
    (see async_checkpoint / save_checkpoint). torch 2.6's weights_only=True
    default refuses those globals. Trusted source — we wrote these ourselves.
    Matches resume_from_checkpoint's existing choice in train.py.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def _handle_evaluate(ns: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd

    from msae import baselines as msae_baselines
    from msae import evaluate as msae_eval
    from msae.models import MatryoshkaSAE, StandardSAE

    out_dir = Path(ns.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = ns.device or _default_device()
    torch.manual_seed(ns.seed)
    np.random.seed(ns.seed)

    # weights_only=True: patches/meta hold only tensors, no pickles.
    patches = torch.load(ns.acts, map_location="cpu", weights_only=True)["patches"]
    meta = torch.load(ns.meta, map_location="cpu", weights_only=True)["meta"]

    plantseg_df = pd.read_csv(ns.plantseg_labels)
    # Per-patch labels: broadcast per-image DataFrame via meta[:,0] image_id.
    plantseg_patch_labels = plantseg_df.iloc[meta[:, 0].numpy()].reset_index(drop=True)

    # --- Load MSAE
    msae_state = _torch_load_trusted(ns.msae_ckpt)
    msae_sd = msae_state.get("model_state_dict", msae_state)
    # Infer shape from the decoder weight: (input_dim, max_features).
    max_features = msae_sd["decoder.weight"].shape[1]
    input_dim = msae_sd["decoder.weight"].shape[0]
    # Default nested_ks if not stored — mirror the production default.
    nested_ks = tuple(msae_state.get("nested_ks", (256, 768, 3072, 12288)))
    msae_model = MatryoshkaSAE(
        input_dim=input_dim, max_features=max_features, nested_ks=nested_ks
    )
    msae_model.load_state_dict(msae_sd)
    if device == "cuda" and torch.cuda.is_available():
        msae_model = msae_model.to("cuda")
    msae_model.eval()

    # --- Encode plantseg (sparse — MSAE encoder is ReLU, so full-dim dense
    # would be ~162 GB on 3.3M patches × 12288 features; scipy CSR is ~1 GB
    # at the trained L0 target of 30–80). All downstream evaluate consumers
    # (class_selectivity, compute_mi, build_grid_acts_chunked) dispatch on
    # scipy.sparse.issparse(z) and take the sparse path automatically.
    msae_z = msae_eval.encode_sparse(
        msae_model, patches,
        batch_size=ns.encode_batch_size,
        show_progress=False,
    )

    # --- Selectivity (species / disease)
    unique_species = sorted(plantseg_patch_labels["species"].dropna().unique().tolist())
    unique_disease = sorted(plantseg_patch_labels["disease"].dropna().unique().tolist())
    species_to_idx = {s: i for i, s in enumerate(unique_species)}
    disease_to_idx = {d: i for i, d in enumerate(unique_disease)}
    species_labels = np.array(
        [species_to_idx[s] for s in plantseg_patch_labels["species"]]
    )
    disease_labels = np.array(
        [disease_to_idx[d] for d in plantseg_patch_labels["disease"]]
    )

    # Healthy-class collapse (plan §1.8 Eval 1). All *_healthy classes flatten
    # to a single merged "healthy" super-class before the entropy is computed
    # to avoid inflating disease selectivity via species × healthy
    # cross-products. Both parsers normalize disease strings through
    # data._norm_disease, which emits the literal "healthy" for any *_healthy
    # input (see data.py::_norm_disease / parse_plantvillage_metadata).
    # PlantSeg has zero healthy rows (115 diseased classes), so the list is
    # empty and class_selectivity is a no-op on that branch. PlantVillage
    # has 14 healthy classes; the collapse fires there.
    healthy_disease_ids: list[int] | None = [
        disease_to_idx[d] for d in unique_disease if d == "healthy"
    ] or None

    msae_sel_species = msae_eval.class_selectivity(
        msae_z, species_labels, n_classes=len(unique_species)
    )
    msae_sel_species.to_csv(out_dir / "msae_selectivity_species.csv", index=False)
    msae_sel_disease = msae_eval.class_selectivity(
        msae_z, disease_labels, n_classes=len(unique_disease),
        healthy_class_ids=healthy_disease_ids,
    )
    msae_sel_disease.to_csv(out_dir / "msae_selectivity_disease.csv", index=False)

    # --- IoU against masks
    iou_tensor: torch.Tensor | None = None
    if ns.masks is not None:
        masks_data = torch.load(ns.masks, map_location="cpu", weights_only=True)
        masks = masks_data["masks"] if isinstance(masks_data, dict) else masks_data
        mask_image_ids = masks_data.get("image_ids") if isinstance(masks_data, dict) else None

        # Build global-index list matching plantseg_df order.
        if mask_image_ids is not None:
            ps_ids = plantseg_df["image_id"].tolist()
            id_to_idx = {iid: i for i, iid in enumerate(ps_ids)}
            masked_img_indices = [id_to_idx[iid] for iid in mask_image_ids if iid in id_to_idx]
            # Keep only the rows of masks that survive the mapping.
            keep = [i for i, iid in enumerate(mask_image_ids) if iid in id_to_idx]
            masks = masks[keep]
        else:
            masked_img_indices = list(range(masks.shape[0]))

        iou_tensor = msae_eval.build_grid_acts_chunked(
            z_all=msae_z,
            meta=meta,
            masked_img_indices=masked_img_indices,
            masks=masks.bool(),
            feature_chunk=ns.iou_feature_chunk,
        )
        torch.save({"iou_scores": iou_tensor}, out_dir / "iou_scores.pt")
        np.save(out_dir / "msae_selectivity_disease_iou.npy", iou_tensor.numpy())

    # --- Mutual information
    mi_species = msae_eval.compute_mi(
        msae_z, species_labels, max_samples=ns.mi_max_samples, seed=ns.seed
    )
    mi_disease = msae_eval.compute_mi(
        msae_z, disease_labels, max_samples=ns.mi_max_samples, seed=ns.seed
    )
    np.save(out_dir / "mi_species.npy", mi_species)
    np.save(out_dir / "mi_disease.npy", mi_disease)

    # --- Neuron baseline (raw DINOv2 activations as features)
    neuron_sel_species = msae_eval.class_selectivity(
        patches.float(), species_labels, n_classes=len(unique_species)
    )
    neuron_sel_species.to_csv(out_dir / "neuron_selectivity_species.csv", index=False)
    neuron_sel_disease = msae_eval.class_selectivity(
        patches.float(), disease_labels, n_classes=len(unique_disease),
        healthy_class_ids=healthy_disease_ids,
    )
    neuron_sel_disease.to_csv(out_dir / "neuron_selectivity_disease.csv", index=False)

    # --- StandardSAE (optional)
    if ns.standard_ckpt is not None:
        std_state = _torch_load_trusted(ns.standard_ckpt)
        std_sd = std_state.get("model_state_dict", std_state)
        n_features = std_sd["decoder.weight"].shape[1]
        std_model = StandardSAE(input_dim=input_dim, n_features=n_features)
        std_model.load_state_dict(std_sd)
        if device == "cuda" and torch.cuda.is_available():
            std_model = std_model.to("cuda")
        std_model.eval()
        std_z = msae_eval.encode_sparse(
            std_model, patches,
            batch_size=ns.encode_batch_size,
            show_progress=False,
        )
        std_sel_species = msae_eval.class_selectivity(
            std_z, species_labels, n_classes=len(unique_species)
        )
        std_sel_species.to_csv(out_dir / "standard_selectivity_species.csv", index=False)
        std_sel_disease = msae_eval.class_selectivity(
            std_z, disease_labels, n_classes=len(unique_disease),
            healthy_class_ids=healthy_disease_ids,
        )
        std_sel_disease.to_csv(out_dir / "standard_selectivity_disease.csv", index=False)

    # --- PCA (optional)
    if ns.pca_result is not None:
        pca_dict = _torch_load_trusted(ns.pca_result)
        V = pca_dict["V"]
        mean = pca_dict.get("mean")
        # Project at full rank to give the selectivity function a comparable signal.
        pca_z = msae_baselines.project_pca(
            patches.float(), V, k=V.shape[1], mean=mean
        )
        pca_sel_species = msae_eval.class_selectivity(
            pca_z, species_labels, n_classes=len(unique_species)
        )
        pca_sel_species.to_csv(out_dir / "pca_selectivity_species.csv", index=False)
        pca_sel_disease = msae_eval.class_selectivity(
            pca_z, disease_labels, n_classes=len(unique_disease),
            healthy_class_ids=healthy_disease_ids,
        )
        pca_sel_disease.to_csv(out_dir / "pca_selectivity_disease.csv", index=False)

    # --- Transfer correlation
    if (
        ns.plantvillage_acts is not None
        and ns.plantvillage_meta is not None
        and ns.plantvillage_labels is not None
        and ns.class_alignment_csv is not None
    ):
        pv_patches = torch.load(
            ns.plantvillage_acts, map_location="cpu", weights_only=True
        )["patches"]
        pv_meta = torch.load(
            ns.plantvillage_meta, map_location="cpu", weights_only=True
        )["meta"]
        pv_df = pd.read_csv(ns.plantvillage_labels)
        pv_patch_labels = pv_df.iloc[pv_meta[:, 0].numpy()].reset_index(drop=True)

        transfer = msae_eval.transfer_correlation(
            msae_encoder=msae_model,
            plantseg_acts=patches,
            plantvillage_acts=pv_patches,
            plantseg_labels=plantseg_patch_labels,
            plantvillage_labels=pv_patch_labels,
            class_alignment_csv=ns.class_alignment_csv,
            encode_batch_size=ns.encode_batch_size,
        )
        (out_dir / "transfer_correlation.json").write_text(
            json.dumps(transfer, indent=2, default=str)
        )

    # --- Layer-6 fallback diagnostic
    trigger, diag = msae_eval.should_trigger_layer6_fallback(
        msae_disease_selectivity=msae_sel_disease,
        neuron_disease_selectivity=neuron_sel_disease,
    )
    (out_dir / "layer6_fallback_diagnostic.json").write_text(
        json.dumps({"trigger": bool(trigger), **diag}, indent=2, default=str)
    )
    if trigger:
        msg = (
            "evaluate: layer-6 fallback triggered — see "
            f"{out_dir / 'layer6_fallback_diagnostic.json'}"
        )
        if getattr(ns, "no_fail_on_layer6_fallback", False):
            # Canary mode: diagnostic is still written so the code path is
            # exercised, but we do not abort. See --no-fail-on-layer6-fallback
            # help text for the full rationale.
            logger.warning("%s (continuing because --no-fail-on-layer6-fallback)", msg)
        else:
            raise SystemExit(msg)


# ---------------------------------------------------------------------------
# visualize handler
# ---------------------------------------------------------------------------

def _handle_visualize(ns: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd

    from msae import evaluate as msae_eval
    from msae import visualize as msae_viz
    from msae.models import MatryoshkaSAE

    out_dir = Path(ns.out)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = ns.device or _default_device()

    eval_dir = Path(ns.eval_results)

    # Load the MSAE checkpoint once upfront so we can use the actual nested_ks
    # from this run for every level-aware plot. Previously the MI scatter
    # hardcoded production defaults (256,768,3072,12288) and the selectivity
    # plot missed a 'level' column entirely — both were wiring bugs that
    # either mislabeled levels (MI) or crashed outright (selectivity).
    msae_state = _torch_load_trusted(ns.msae_ckpt)
    msae_sd = msae_state.get("model_state_dict", msae_state)
    max_features = msae_sd["decoder.weight"].shape[1]
    input_dim = msae_sd["decoder.weight"].shape[0]
    nested_ks = tuple(msae_state.get("nested_ks", (256, 768, 3072, 12288)))
    # Vectorized feature_id -> level k: feature j belongs to the smallest nest
    # size k such that j < k. We return the k-value (not the index), because:
    #   - plot_mi_scatter matches with `mask = nested_level == k` (k-value)
    #   - the notebook uses `next(k for k in nested_ks if fid < k)` (k-value)
    #   - plot_selectivity_per_level groups by this column; k-values group the
    #     same as indices for aggregation purposes.
    nested_ks_arr = np.asarray(nested_ks, dtype=np.int64)

    def levels_for(feature_ids: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(nested_ks_arr, feature_ids, side="right").clip(
            max=len(nested_ks_arr) - 1
        )
        return nested_ks_arr[idx]

    # Reload selectivity + MI outputs.
    mi_species_path = eval_dir / "mi_species.npy"
    mi_disease_path = eval_dir / "mi_disease.npy"
    if mi_species_path.exists() and mi_disease_path.exists():
        mi_species = np.load(mi_species_path)
        mi_disease = np.load(mi_disease_path)
        mi_levels = levels_for(np.arange(len(mi_species)))
        msae_viz.plot_mi_scatter(
            species_mi=mi_species,
            disease_mi=mi_disease,
            nested_level=mi_levels,
            output_path=fig_dir / "mi_scatter.png",
            nested_ks=nested_ks,
        )

    msae_sel_disease_path = eval_dir / "msae_selectivity_disease.csv"
    msae_sel_species_path = eval_dir / "msae_selectivity_species.csv"
    # Per-level selectivity plots — one per label family. The MSAE disease
    # plot was the only one the CLI rendered previously; species and baseline
    # CSVs (neuron / pca / standard) were written by evaluate but never
    # consumed, so baseline comparisons only existed in the notebook.
    for csv_path, fig_name in (
        (msae_sel_disease_path, "selectivity_per_level_disease.png"),
        (msae_sel_species_path, "selectivity_per_level_species.png"),
    ):
        if csv_path.exists():
            df = pd.read_csv(csv_path).copy()
            df["level"] = levels_for(df["feature_id"].to_numpy())
            msae_viz.plot_selectivity_per_level(df, output_path=fig_dir / fig_name)

    # Baseline comparison — mean selectivity per method, for whichever CSVs
    # evaluate actually produced this run. Neuron is always present; PCA and
    # Standard SAE are conditional on the corresponding --pca-result /
    # --standard-ckpt args, so entries missing their CSV are skipped.
    comparison_sources = [
        ("Neuron basis",  "neuron_selectivity_species.csv",  "neuron_selectivity_disease.csv"),
        ("PCA",           "pca_selectivity_species.csv",     "pca_selectivity_disease.csv"),
        ("Standard SAE",  "standard_selectivity_species.csv","standard_selectivity_disease.csv"),
        ("MSAE",          "msae_selectivity_species.csv",    "msae_selectivity_disease.csv"),
    ]
    def _mean_or_nan(path: Path) -> float:
        if not path.exists():
            return float("nan")
        col = pd.read_csv(path)["selectivity"]
        return float(col.mean()) if len(col) else float("nan")
    comparison_rows = []
    for label, sp_name, di_name in comparison_sources:
        sp_mean = _mean_or_nan(eval_dir / sp_name)
        di_mean = _mean_or_nan(eval_dir / di_name)
        # Include the row only if at least one side is available; otherwise
        # the method simply didn't run this evaluation.
        if np.isfinite(sp_mean) or np.isfinite(di_mean):
            comparison_rows.append({
                "method": label,
                "species_selectivity": sp_mean,
                "disease_selectivity": di_mean,
            })
    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        msae_viz.plot_selectivity_comparison(
            comparison_df, output_path=fig_dir / "selectivity_comparison.png"
        )
        # Also persist the raw means so downstream analysis / the paper table
        # doesn't have to re-aggregate from the per-feature CSVs.
        comparison_df.to_csv(fig_dir / "selectivity_comparison.csv", index=False)

    # Training curves (optional). Look next to the msae checkpoint first
    # (canonical location — train.py writes the log into checkpoint_dir),
    # then a couple of legacy layouts. Previously assumed eval_dir.parent had
    # a sibling 'checkpoints/' dir, which is not a universal contract: runs
    # with a standalone --eval-results would silently skip the training-curve
    # plot.
    msae_ckpt_dir = Path(ns.msae_ckpt).parent
    for log_name, out_name in (
        ("msae_log.json", "training_curves_msae.png"),
        ("standard_log.json", "training_curves_standard.png"),
    ):
        for candidate in (
            msae_ckpt_dir / log_name,
            eval_dir / log_name,
            eval_dir.parent / "checkpoints" / log_name,
        ):
            if candidate.exists():
                msae_viz.plot_training_curves(
                    candidate, output_path=fig_dir / out_name
                )
                break

    # Top-N disease-selective features → top_patches_<fid>.png
    plantseg_df = pd.read_csv(ns.plantseg_labels)
    # weights_only=True: patches/meta are tensor-only.
    patches = torch.load(ns.acts, map_location="cpu", weights_only=True)["patches"]
    meta = torch.load(ns.meta, map_location="cpu", weights_only=True)["meta"]

    msae_model = MatryoshkaSAE(
        input_dim=input_dim, max_features=max_features, nested_ks=nested_ks
    )
    msae_model.load_state_dict(msae_sd)
    if device == "cuda" and torch.cuda.is_available():
        msae_model = msae_model.to("cuda")
    msae_model.eval()

    # Sparse encode — visualize's msae_z has the same OOM risk as evaluate's
    # at production dims (see encode_sparse docstring). Top-patches and
    # heatmaps pull one column at a time, which scipy handles cheaply.
    msae_z = msae_eval.encode_sparse(
        msae_model, patches, batch_size=16384, show_progress=False,
    )

    def _feature_col(fid: int) -> torch.Tensor:
        """Return msae_z[:, fid] as a 1-D fp32 torch tensor (CPU)."""
        col = msae_z[:, fid].toarray().ravel().astype(np.float32, copy=False)
        return torch.from_numpy(col)

    if msae_sel_disease_path.exists():
        df_sel = pd.read_csv(msae_sel_disease_path).sort_values(
            "selectivity", ascending=False
        )
        # parse_plantseg_metadata always stores image_id as a string like
        # "<class>/<name>", so we key image paths by their positional row index
        # (which matches meta[:, 0] — the patch's image_id in that order).
        image_paths_by_id = {
            i: Path(p) for i, p in enumerate(plantseg_df["filepath"].tolist())
        }
        for feature_id in df_sel.head(ns.top_n_features)["feature_id"].tolist():
            msae_viz.plot_top_activating_patches(
                feature_id=int(feature_id),
                feature_acts=_feature_col(int(feature_id)),
                image_paths_by_id=image_paths_by_id,
                patch_meta=meta,
                output_path=fig_dir / f"top_patches_{int(feature_id)}.png",
            )

    # Optional hand-selected heatmaps.
    if ns.heatmap_ids:
        # Choose which feature to visualize: top disease-selective if available,
        # otherwise feature 0. Previously always hardcoded to feature 0, so
        # every heatmap_*.png rendered the same arbitrary channel regardless
        # of --heatmap-ids.
        if msae_sel_disease_path.exists():
            top_feature_id = int(
                pd.read_csv(msae_sel_disease_path)
                .sort_values("selectivity", ascending=False)
                .iloc[0]["feature_id"]
            )
        else:
            top_feature_id = 0
        ids = [s.strip() for s in ns.heatmap_ids.split(",") if s.strip()]
        heatmap_feature_col = _feature_col(top_feature_id)
        for raw_id in ids:
            # Resolve to a filesystem path via plantseg_df.
            row = plantseg_df[plantseg_df["image_id"].astype(str) == str(raw_id)]
            if row.empty:
                logger.warning("heatmap-id %s not found in plantseg labels", raw_id)
                continue
            image_path = Path(row.iloc[0]["filepath"])
            global_id = int(row.index[0])
            msae_viz.plot_spatial_heatmap(
                image_path=image_path,
                feature_acts_per_patch=heatmap_feature_col,
                patch_meta=meta,
                image_id=global_id,
                output_path=fig_dir / f"heatmap_{raw_id}_feat{top_feature_id}.png",
            )


# ---------------------------------------------------------------------------
# subcommand dispatch
# ---------------------------------------------------------------------------

_SUBCOMMANDS = {
    "validate-data": _handle_validate_data,
    "extract": _handle_extract,
    "probe": _handle_probe,
    "train": _handle_train,
    "evaluate": _handle_evaluate,
    "visualize": _handle_visualize,
}


if __name__ == "__main__":
    sys.exit(main())
