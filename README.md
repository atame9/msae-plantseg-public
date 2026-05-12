# Matryoshka Sparse Autoencoder on DINOv2 Plant Disease Features

A **Matryoshka Sparse Autoencoder (MSAE)** trained on DINOv2 ViT-B/14 layer-8 patch activations extracted from the PlantSeg plant disease dataset. The model learns nested sparse feature dictionaries at k = [256, 768, 3072, 12288], where coarser levels capture plant-family structure and finer levels capture disease-specific patterns.

## Key Results

| Method | Species Selectivity | Disease Selectivity |
|--------|-------------------|-------------------|
| Neuron basis (raw DINOv2) | 0.114 | 0.064 |
| Standard SAE (12288 features) | 0.130 | 0.127 |
| **MSAE (nested 256–12288)** | **0.143** | **0.134** |

The MSAE outperforms both the raw neuron baseline (+110% disease selectivity) and the flat Standard SAE (+5%) on entropy-based class selectivity. The layer-6 fallback diagnostic confirms the MSAE–neuron gap exceeds the 0.05 threshold (gap = 0.070).

## Setup

Requires Python 3.10+ and PyTorch ≥ 2.3 with CUDA support (A100 recommended for training).

```bash
git clone https://github.com/atame9/msae-plantseg-public.git
cd msae-plantseg-public
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Pipeline

The `msae` CLI provides six subcommands that run in sequence:

```
validate-data → extract → probe → train → evaluate → visualize
```

- **validate-data** — preflight checks on dataset directory layout and class alignment
- **extract** — DINOv2 ViT-B/14 layer-8 patch + CLS activation extraction with L2 filtering
- **probe** — LinearProbe on CLS tokens; gates on val_acc ≥ 0.50 before SAE training
- **train** — MSAE (`--model msae`) or StandardSAE (`--model standard`) with AuxK dead-feature recovery, cosine LR schedule, and λ_sparse auto-tuning
- **evaluate** — per-feature selectivity (species/disease), IoU against segmentation masks, mutual information, cross-dataset transfer correlation
- **visualize** — MI scatter plots, per-level selectivity, method comparison, training curves, top-activating patch grids

### Example (full pipeline)

```bash
msae validate-data \
    --plantseg-images data/plantseg/images \
    --plantseg-masks data/plantseg/masks \
    --plantvillage-images data/plantvillage/color

msae extract --dataset plantseg \
    --images data/plantseg/images --masks data/plantseg/masks \
    --out run/plantseg_activations --seed 42

msae extract --dataset plantvillage \
    --images data/plantvillage/color \
    --out run/plantvillage_activations --seed 42

msae probe \
    --cls run/plantseg_activations/cls.pt \
    --labels run/plantseg_activations/plantseg_labels.csv \
    --out run/probe --seed 42

msae train --model msae \
    --acts run/plantseg_activations \
    --out run/checkpoints --seed 42

msae train --model standard \
    --acts run/plantseg_activations \
    --out run/checkpoints --seed 42

msae evaluate \
    --acts run/plantseg_activations/patches.pt \
    --meta run/plantseg_activations/meta.pt \
    --msae-ckpt run/checkpoints/msae_final.pt \
    --standard-ckpt run/checkpoints/sae_final.pt \
    --plantseg-labels run/plantseg_activations/plantseg_labels.csv \
    --plantvillage-acts run/plantvillage_activations/patches.pt \
    --plantvillage-meta run/plantvillage_activations/meta.pt \
    --plantvillage-labels run/plantvillage_activations/plantvillage_labels.csv \
    --class-alignment-csv run/plantseg_activations/class_alignment.csv \
    --masks run/plantseg_activations/masks_16x16.pt \
    --out run/eval_results --seed 42

msae visualize \
    --eval-results run/eval_results \
    --msae-ckpt run/checkpoints/msae_final.pt \
    --acts run/plantseg_activations/patches.pt \
    --meta run/plantseg_activations/meta.pt \
    --plantseg-labels run/plantseg_activations/plantseg_labels.csv \
    --out run/figures --top-n-features 5
```

## Tests

```bash
pytest tests/ -q          # 47 tests, all CPU-only
ruff check src/ tests/    # lint
```

## Project Structure

```
src/msae/
  models.py         MatryoshkaSAE, StandardSAE, LinearProbe (nn.Module definitions)
  data.py           Dataset parsers, class-name normalization, cross-dataset alignment
  extraction.py     DINOv2 activation extraction with L2 filtering and chunked I/O
  train.py          Training loops with AuxK, dead-feature refresh, torch.compile
  evaluate.py       Selectivity, IoU, MI, transfer correlation (sparse-aware paths)
  baselines.py      PCA (GPU + sklearn fallback) and neuron-basis projection
  visualize.py      All matplotlib plots (Agg backend, saves to disk)
  cli.py            argparse CLI entry points
  _config.py        Config resolution (defaults < JSON file < CLI flags)
  _manifest.py      Run provenance metadata

configs/            Reference training configs (msae_default.json, standard_sae_default.json)
tests/              pytest suite (CPU-only, synthetic data)
results/            Pre-computed evaluation outputs and figures from the paper run
```

## Model Architecture

The MSAE encodes 768-dim DINOv2 patch activations into a 12288-dim sparse code via a single encoder, then decodes at nested Matryoshka boundaries [256, 768, 3072, 12288]. Each level produces a reconstruction using only features up to that boundary. The loss averages MSE + weighted L1 sparsity across all levels, with 1/k weighting to prevent finer levels from dominating the sparsity budget.

Training uses:
- **AuxK** dead-feature recovery (α=1/32, k_aux=512)
- **λ_sparse auto-tuning** via 3-point probe targeting L0 ∈ [30, 80]
- **torch.compile** with cascading mode fallback
- **bf16 autocast** on CUDA, fp32 for loss computation
- **CosineAnnealingLR** from lr=5e-4 to 1e-5
- **Early stopping** with patience of 600 steps

## Datasets

- **PlantSeg v5** ([Zenodo 10.5281/zenodo.14935094](https://doi.org/10.5281/zenodo.14935094)) — 115 classes, 11,458 images with segmentation masks. CC BY 4.0.
- **PlantVillage** ([Hughes & Salathé 2015](https://arxiv.org/abs/1511.08060)) — 38 classes, 54,305 images. Used for cross-dataset transfer evaluation.

16 classes overlap between datasets (matched via normalized species+disease names), enabling quantitative transfer correlation measurement.

## License

Code: MIT

Dataset-derived artifacts (model weights, evaluation results, figures) inherit **CC BY 4.0** from PlantSeg v5. Attribution: Wei et al. 2026.
