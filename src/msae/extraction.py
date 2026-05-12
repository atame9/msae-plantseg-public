from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


def setup_dinov2(device: str = "cuda") -> nn.Module:
    """Load DINOv2 ViT-B/14 from torch.hub, set to inference mode and target device.

    Does NOT call .half() — bf16 happens at the autocast site.
    Enables TF32 for matmul and cuDNN (spec line 322).
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitb14",
        verbose=False,
    )
    model.eval()
    model.to(device)
    return model


def register_layer_hook(
    model: nn.Module,
    layer_idx: int,
    buffer: list,
) -> torch.utils.hooks.RemovableHandle:
    """Register a forward hook on model.blocks[layer_idx].

    The hook appends the output tensor to *buffer* without moving it to CPU
    (CPU transfer inside the hook would cause a per-block CUDA sync).

    Returns the RemovableHandle so the caller can .remove() it after use.
    """

    def hook(module, input, output):  # noqa: ANN001
        buffer.append(output[0] if isinstance(output, tuple) else output)

    handle = model.blocks[layer_idx].register_forward_hook(hook)
    return handle


def filter_patches_l2(
    patches: Tensor,        # (B, 256, 768) — raw patch tokens, GPU or CPU
    image_ids: Tensor,      # (B,) int64 — image indices in the dataset
    percentile: float = 0.20,
) -> tuple[Tensor, Tensor]:
    """Per-image 20th-percentile L2-norm filter.

    Returns:
        kept_patches_bf16 : (n_kept, 768) bfloat16
        metadata_int32    : (n_kept, 3)   int32  — columns [image_id, row, col]
    """
    # Cast to float for norm computation regardless of input dtype.
    norms = patches.float().norm(dim=-1)          # (B, 256)

    thresh = torch.quantile(norms, percentile, dim=1, keepdim=True)  # (B, 1)
    keep = norms >= thresh                         # (B, 256) bool

    # Single nonzero call — returns (n_kept, 2) with [img_local_idx, patch_idx]
    nz = keep.nonzero()                            # (n_kept, 2)
    img_local_idx = nz[:, 0]                       # index into batch
    patch_idx = nz[:, 1]                           # 0..255

    # Gather kept patch embeddings and cast to bf16
    kept = patches[img_local_idx, patch_idx, :].to(torch.bfloat16)  # (n_kept, 768)

    # Build metadata: [image_id, row, col]
    image_id = image_ids[img_local_idx]           # (n_kept,) int64 -> cast below
    row = patch_idx // 16
    col = patch_idx % 16

    metadata = torch.stack(
        [image_id.to(torch.int32), row.to(torch.int32), col.to(torch.int32)],
        dim=1,
    )  # (n_kept, 3) int32

    return kept, metadata


def extract_activations(
    model: nn.Module,
    dataset: Dataset,
    layer_idx: int = 8,
    batch_size: int = 256,
    output_dir: Path = Path("/content/activations"),
    chunk_size_images: int = 5000,
    apply_l2_filter: bool = True,
    save_cls: bool = True,
    resume: bool = True,
    device: str = "cuda",
) -> dict:
    """Extract DINOv2 layer activations for every image in *dataset*.

    Saves chunk files ``patches_chunk_{N:04d}.pt`` (and optionally
    ``cls_chunk_{N:04d}.pt``) under *output_dir*, one chunk per
    ``chunk_size_images`` images.

    Returns a summary dict with keys: output_dir, n_images, n_kept_patches,
    chunks_written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ resume
    start_image_idx = 0
    start_chunk_idx = 0
    if resume:
        existing = sorted(output_dir.glob("patches_chunk_*.pt"))
        if existing:
            last_chunk = int(existing[-1].stem.split("_")[-1])
            start_chunk_idx = last_chunk + 1
            # Use sidecar JSON to compute the EXACT number of images per chunk
            # (chunks are >= chunk_size_images because we trigger AFTER the
            # increment; a heuristic ``start_chunk_idx * chunk_size_images``
            # causes duplicate image_ids on resume).
            start_image_idx = 0
            for i in range(start_chunk_idx):
                sidecar = output_dir / f"chunk_{i:04d}.json"
                if sidecar.exists():
                    start_image_idx += json.loads(sidecar.read_text())["n_images_in_chunk"]
                else:
                    logger.warning(
                        "Missing sidecar %s; falling back to heuristic start_image_idx "
                        "(duplicate patches possible on resume)", sidecar,
                    )
                    start_image_idx = start_chunk_idx * chunk_size_images
                    break
            logger.info(
                "Resuming from chunk %d (skipping first %d images)",
                start_chunk_idx,
                start_image_idx,
            )

    # ------------------------------------------------------------------ hook
    hook_buffer: list = []
    handle = register_layer_hook(model, layer_idx, hook_buffer)

    # ------------------------------------------------------------------ CPU fallback adjustments
    on_cpu = device == "cpu"
    if on_cpu:
        batch_size = min(batch_size, 4)

    import os as _os
    # num_workers: leave 1 vCPU for the main process (model forward + H2D copy).
    # On a 4-vCPU A100 host, 3 libjpeg-turbo workers produce ~600 MB/s of
    # decoded tensors — ~40× what DINOv2 ViT-B/14 consumes at batch 256 on A100.
    # CPU fallback stays serial (batch_size=4 is already tiny). Previous
    # num_workers=0 (Colab-era default) serialized decode with forward, held
    # the GPU at 0% util, and stretched a 15-min extract into hours.
    _nw = 0 if on_cpu else max(1, (_os.cpu_count() or 4) - 1)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_nw,
        pin_memory=not on_cpu,
        prefetch_factor=(4 if _nw > 0 else None),
        persistent_workers=False,
    )

    # ------------------------------------------------------------------ accumulators
    patch_acc: list = []
    meta_acc: list = []
    cls_acc: list = []

    total_images = 0
    total_kept = 0
    chunks_written = start_chunk_idx
    images_in_chunk = 0
    global_img_counter = 0

    # ------------------------------------------------------------------ helpers
    def _save_chunk(p_list, m_list, c_list, chunk_idx: int, n_imgs_in_chunk: int) -> None:
        if not p_list:
            return
        patches_t = torch.cat(p_list, dim=0)
        meta_t = torch.cat(m_list, dim=0)
        torch.save({"patches": patches_t}, output_dir / f"patches_chunk_{chunk_idx:04d}.pt")
        torch.save({"meta": meta_t}, output_dir / f"meta_chunk_{chunk_idx:04d}.pt")
        # Sidecar JSON lets resume compute the EXACT number of skipped images,
        # not a heuristic based on chunk_size_images.
        sidecar_path = output_dir / f"chunk_{chunk_idx:04d}.json"
        sidecar_path.write_text(json.dumps({
            "n_images_in_chunk": int(n_imgs_in_chunk),
            "n_kept_patches_in_chunk": int(patches_t.shape[0]),
        }))
        logger.info("Saved chunk %d: %d patches (%d images)",
                    chunk_idx, patches_t.shape[0], n_imgs_in_chunk)
        if save_cls and c_list:
            cls_t = torch.cat(c_list, dim=0)
            torch.save({"cls": cls_t}, output_dir / f"cls_chunk_{chunk_idx:04d}.pt")

    def _process_batch(batch_images: Tensor, batch_ids: Tensor) -> None:
        nonlocal total_images, total_kept, images_in_chunk, chunks_written

        # Heartbeat: log every ~10 batches so we see batch progression rather
        # than going silent between chunk-saves (every chunk_size_images, default
        # 5000). At batch_size=256 this is every ~2560 images — dense enough to
        # spot a stall within a minute even on the smaller PlantSeg extract.
        if total_images and total_images % (batch_images.shape[0] * 10) == 0:
            logger.info(
                "extract: %d images processed (%d kept patches)",
                total_images, total_kept,
            )
        hook_buffer.clear()
        batch_images = batch_images.to(device, non_blocking=not on_cpu)
        B = batch_images.shape[0]

        model(batch_images)

        if not hook_buffer:
            logger.warning("Hook buffer empty after forward pass!")
            return

        out = hook_buffer[0]   # (B, 257, 768)
        hook_buffer.clear()

        cls = out[:, 0, :].float()     # (B, 768) fp32
        patches = out[:, 1:, :]        # (B, 256, 768)

        if apply_l2_filter and not on_cpu:
            kept, meta = filter_patches_l2(patches, batch_ids.to(device))
            kept_cpu = kept.cpu()
            meta_cpu = meta.cpu()
        elif apply_l2_filter and on_cpu:
            kept, meta = filter_patches_l2(patches, batch_ids)
            kept_cpu = kept
            meta_cpu = meta
        else:
            gi = torch.arange(B, dtype=torch.int64).unsqueeze(1).expand(B, 256).reshape(-1)
            gp = torch.arange(256, dtype=torch.int64).unsqueeze(0).expand(B, 256).reshape(-1)
            kept = patches.reshape(B * 256, 768).to(torch.bfloat16)
            image_id = batch_ids[gi].to(torch.int32)
            row = (gp // 16).to(torch.int32)
            col = (gp % 16).to(torch.int32)
            meta = torch.stack([image_id, row, col], dim=1)
            kept_cpu = kept.cpu() if not on_cpu else kept
            meta_cpu = meta.cpu() if not on_cpu else meta

        cls_cpu = cls.cpu() if not on_cpu else cls

        patch_acc.append(kept_cpu)
        meta_acc.append(meta_cpu)
        if save_cls:
            cls_acc.append(cls_cpu)

        total_images += B
        total_kept += kept_cpu.shape[0]
        images_in_chunk += B

        if images_in_chunk >= chunk_size_images:
            _save_chunk(patch_acc, meta_acc, cls_acc, chunks_written, images_in_chunk)
            patch_acc.clear()
            meta_acc.clear()
            cls_acc.clear()
            chunks_written += 1
            images_in_chunk = 0

    # ------------------------------------------------------------------ main loop
    with torch.inference_mode():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                batch_images, batch_ids = batch
                # Dataset returns (Tensor, str) -> default_collate produces a
                # tuple of strings rather than a Tensor. Synthesise int indices
                # in that case so downstream .to(device) and scatter_add paths
                # work; string image IDs are not used by filter_patches_l2.
                if not isinstance(batch_ids, torch.Tensor):
                    batch_ids = torch.arange(
                        global_img_counter,
                        global_img_counter + batch_images.shape[0],
                        dtype=torch.int64,
                    )
            else:
                batch_images = batch
                batch_ids = torch.arange(
                    global_img_counter,
                    global_img_counter + batch_images.shape[0],
                    dtype=torch.int64,
                )

            global_img_counter += batch_images.shape[0]

            if global_img_counter <= start_image_idx:
                continue

            if on_cpu:
                _process_batch(batch_images, batch_ids)
            else:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _process_batch(batch_images, batch_ids)

    if patch_acc:
        _save_chunk(patch_acc, meta_acc, cls_acc, chunks_written, images_in_chunk)
        chunks_written += 1

    handle.remove()

    return {
        "output_dir": output_dir,
        "n_images": total_images,
        "n_kept_patches": total_kept,
        "chunks_written": chunks_written,
    }


def verify_raster_order(
    model: nn.Module,
    image_path: Path,
    layer_idx: int = 8,
    output_overlay_path: Path = Path("results/raster_check.png"),
    device: str = "cuda",
) -> None:
    """Visual sanity check: overlay per-patch L2 norms on the source image.

    Loads one image, extracts patch tokens at *layer_idx*, computes L2 norms,
    reshapes to 16x16, and saves an alpha-blended overlay PNG to
    *output_overlay_path*.

    This does NOT assert anything programmatically — the user must visually
    confirm that high-norm patches align with leaf/lesion regions.
    """
    import numpy as np
    from PIL import Image
    import torchvision.transforms as T
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    hook_buffer_v: list = []
    handle_v = register_layer_hook(model, layer_idx, hook_buffer_v)

    model.eval()
    with torch.inference_mode():
        if device != "cpu":
            with torch.amp.autocast(device, dtype=torch.bfloat16):
                model(img_tensor)
        else:
            model(img_tensor)

    handle_v.remove()

    out = hook_buffer_v[0]
    patches = out[0, 1:, :].float()
    norms = patches.norm(dim=-1)
    norm_grid = norms.cpu().reshape(16, 16).numpy()

    norm_resized = np.kron(norm_grid, np.ones((14, 14)))
    norm_min, norm_max = norm_resized.min(), norm_resized.max()
    if norm_max > norm_min:
        norm_norm = (norm_resized - norm_min) / (norm_max - norm_min)
    else:
        norm_norm = norm_resized

    img_display = np.array(img_pil.resize((224, 224)))

    output_overlay_path = Path(output_overlay_path)
    output_overlay_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_display)
    axes[0].set_title("Original (224x224)")
    axes[0].axis("off")

    axes[1].imshow(img_display)
    hm = axes[1].imshow(norm_norm, cmap="jet", alpha=0.5)
    axes[1].set_title(f"L2 norm overlay (layer blocks[{layer_idx}])")
    axes[1].axis("off")
    fig.colorbar(hm, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_overlay_path, dpi=100)
    plt.close(fig)

    logger.info("Raster-order verification saved to %s", output_overlay_path)


def consolidate_chunks(output_dir: Path) -> dict[str, Path]:
    """Merge all chunk files written by extract_activations into single tensors.

    Memory-safe implementation (see OOM on 2026-05-11 PlantVillage run, 17 GB
    of chunks + torch.cat of another 17 GB exceeded the 32 GB host): we
    preallocate the output tensor using sidecar n_kept_patches_in_chunk /
    direct shape inspection, then stream each chunk in, copy into the
    preallocated buffer, and ``del`` the chunk before reading the next.
    Peak host memory is ~1 × total tensor size rather than ~2 × from cat.

    Idempotent: if ``patches.pt`` already exists and its row count matches
    the sum of the chunk sidecars, returns immediately without re-loading
    anything — letting ``extract --resume`` be cheap on a fully-extracted
    dataset.

    Returns dict with keys: 'patches_path', 'cls_path', 'meta_path'.
    """
    output_dir = Path(output_dir)
    chunk_files = sorted(output_dir.glob("patches_chunk_*.pt"))
    patches_path = output_dir / "patches.pt"
    cls_path = output_dir / "cls.pt"
    meta_path = output_dir / "meta.pt"

    if not chunk_files:
        # Fully-consolidated output with no chunks left — return existing.
        if patches_path.exists():
            result: dict[str, Path] = {"patches_path": patches_path, "meta_path": meta_path}
            if cls_path.exists():
                result["cls_path"] = cls_path
            return result
        raise FileNotFoundError(f"No chunk files or consolidated files found in {output_dir}")

    # Collect sidecar row counts (per chunk) — cheap, just JSON reads.
    # Falls back to shape inspection when a sidecar is missing.
    def _chunk_row_counts(pattern: str, key: str) -> list[int]:
        counts: list[int] = []
        for cf in chunk_files:
            idx = cf.stem.split("_")[-1]
            other = output_dir / f"{pattern}_{idx}.pt"
            if not other.exists():
                counts.append(0)
                continue
            sidecar = output_dir / f"chunk_{idx}.json"
            if sidecar.exists() and pattern == "patches_chunk":
                try:
                    counts.append(int(json.loads(sidecar.read_text())["n_kept_patches_in_chunk"]))
                    continue
                except Exception:  # noqa: BLE001
                    pass
            # Fall back to shape inspection (opens the .pt briefly)
            t = torch.load(other, weights_only=True)[key]
            counts.append(int(t.shape[0]))
            del t
        return counts

    patches_counts = _chunk_row_counts("patches_chunk", "patches")
    total_patches = sum(patches_counts)

    # Idempotent short-circuit: if consolidated file already matches the
    # current chunk set, we're done. Avoids a 18 GB copy on a resumed run
    # where consolidation already succeeded.
    if patches_path.exists():
        try:
            existing = torch.load(patches_path, weights_only=True)["patches"]
            if int(existing.shape[0]) == total_patches:
                logger.info(
                    "consolidate_chunks: %s already has %d rows matching %d chunks — skipping reconsolidation",
                    patches_path, total_patches, len(chunk_files),
                )
                del existing
                result = {"patches_path": patches_path, "meta_path": meta_path}
                if cls_path.exists():
                    result["cls_path"] = cls_path
                return result
            del existing
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "consolidate_chunks: could not validate existing %s (%s); reconsolidating",
                patches_path, exc,
            )

    # Peek the first chunk to learn feature dim + dtype.
    first = torch.load(chunk_files[0], weights_only=True)["patches"]
    n_features = int(first.shape[1])
    patches_dtype = first.dtype
    del first

    # Preallocate — ~17 GB for PlantVillage at bf16, ~3.5 GB for PlantSeg.
    patches_out = torch.empty((total_patches, n_features), dtype=patches_dtype)

    offset = 0
    for cf, n in zip(chunk_files, patches_counts):
        data = torch.load(cf, weights_only=True)
        chunk = data["patches"]
        assert int(chunk.shape[0]) == n, f"{cf}: chunk has {chunk.shape[0]} rows; sidecar said {n}"
        patches_out[offset:offset + n] = chunk
        offset += n
        # Drop the chunk immediately. Keeping it alongside the preallocated
        # buffer is what caused the OOM on 32 GB hosts (2× peak memory).
        del data, chunk

    torch.save({"patches": patches_out}, patches_path)
    del patches_out  # release before the next concat runs

    # Same pattern for cls and meta — orders of magnitude smaller (~100 MB
    # and ~70 MB respectively for PlantVillage), so the simple cat is fine,
    # but we still del chunks as we go.
    cls_list: list[Tensor] = []
    meta_list: list[Tensor] = []
    for cf in chunk_files:
        idx = cf.stem.split("_")[-1]
        cls_chunk_path = output_dir / f"cls_chunk_{idx}.pt"
        meta_chunk_path = output_dir / f"meta_chunk_{idx}.pt"
        if cls_chunk_path.exists():
            cls_list.append(torch.load(cls_chunk_path, weights_only=True)["cls"])
        if meta_chunk_path.exists():
            meta_list.append(torch.load(meta_chunk_path, weights_only=True)["meta"])

    if cls_list:
        torch.save({"cls": torch.cat(cls_list, dim=0)}, cls_path)
    if meta_list:
        torch.save({"meta": torch.cat(meta_list, dim=0)}, meta_path)
    del cls_list, meta_list

    logger.info(
        "Consolidated %d chunks → patches=%s cls=%s meta=%s",
        len(chunk_files), patches_path,
        cls_path if cls_path.exists() else "(none)",
        meta_path if meta_path.exists() else "(none)",
    )
    result = {"patches_path": patches_path, "meta_path": meta_path}
    if cls_path.exists():
        result["cls_path"] = cls_path
    return result
