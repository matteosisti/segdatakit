# segdatakit

**Lossless preprocessing and efficient storage for semantic segmentation datasets.**

`segdatakit` converts raw segmentation datasets (Cityscapes, COCO, and more) into optimized formats for cloud-based training environments like Google Colab — without ever compromising pixel fidelity. Every conversion is verified with a cryptographic audit trail.

```bash
pip install segdatakit
python scripts/convert.py --cfg configs/cityscapes.yaml
python scripts/validate.py --cfg configs/cityscapes.yaml --audit-lossless
```

---

## Motivation

Training segmentation models on Colab or Kaggle means fighting the same battle every session: 30 GB of raw PNG files, Drive I/O bottlenecks, and VM resets that wipe everything. `segdatakit` solves this at the data layer — not by downsampling or approximating, but by repackaging data into formats that are fast to stream, small enough to keep on Drive, and mathematically guaranteed to be identical to the originals.

---

## Features

- **Lossless by design** — SHA-256 round-trip verification on every conversion. If a single pixel changes, the audit fails.
- **Format-agnostic output** — write to [Zarr](https://zarr.readthedocs.io/) with Blosc2/LZ4 compression, [WebDataset](https://github.com/webdataset/webdataset) shards, or raw `.npy`.
- **Config-driven** — swap dataset by swapping a YAML file. No code changes between Cityscapes and COCO.
- **PyTorch-native dataloader** — `SegDataset` plugs directly into any training loop. Supports lazy Zarr access, no full dataset extraction needed.
- **NVIDIA DALI support** — optional GPU-accelerated pipeline via `dali_loader.py`. Moves decode, resize, and normalisation onto the GPU (~4ms/batch vs ~40ms CPU). Falls back to standard PyTorch DataLoader automatically if DALI is not installed.
- **Sanity checks built-in** — class distribution, aspect ratio validation, void pixel fraction, and OoD score ratio checks run automatically after conversion.
- **Colab-ready** — designed around the constraints of free-tier cloud notebooks: Drive streaming, session resets, and 20 GB working disk limits.

---

## Supported datasets

| Dataset | Classes | Task | Config |
|---|---|---|---|
| Cityscapes | 19 | Semantic segmentation | `configs/cityscapes.yaml` |
| COCO | 133 | Panoptic segmentation | `configs/coco.yaml` |
| custom | any | any | `configs/template.yaml` |

Adding a new dataset means writing a new YAML and a `Reader` subclass — nothing else changes.

---

## Installation

```bash
git clone https://github.com/your-username/segdatakit
cd segdatakit
pip install -e ".[dev]"
```

**Dependencies:** `zarr`, `numcodecs`, `webdataset`, `numpy`, `Pillow`, `torch`, `pyyaml`, `tqdm`

**Optional — NVIDIA DALI (GPU-accelerated loading):**

```bash
pip install nvidia-dali-cuda120   # CUDA 12.x
pip install nvidia-dali-cuda110   # CUDA 11.x
```

Check your CUDA version with `nvcc --version`. On Colab, CUDA 12.x is standard as of 2025.
If DALI is not installed, `build_loader()` falls back to the standard PyTorch DataLoader automatically — no code changes needed.

---

## Quickstart — Cityscapes

### 1. Convert

Point `--cfg` at your dataset config and `--raw` at your local Cityscapes root:

```bash
python scripts/convert.py \
    --cfg configs/cityscapes.yaml \
    --raw /path/to/cityscapes \
    --out cityscapes.zarr
```

This produces a single `cityscapes.zarr` file (~12–15 GB) with chunked, Blosc2-compressed images and masks, preserving the original `1024×2048` resolution.

### 2. Verify lossless integrity

```bash
python scripts/validate.py \
    --cfg configs/cityscapes.yaml \
    --raw /path/to/cityscapes \
    --zarr cityscapes.zarr \
    --audit-lossless \
    --n-samples 100
```

Expected output:

```
Lossless audit — Cityscapes → cityscapes.zarr
Samples tested  : 100 / 2975
Images passed   : 100 / 100
Masks passed    : 100 / 100
Result          : LOSSLESS VERIFIED ✓
Report saved    : audit_report.json
```

The `audit_report.json` is saved alongside the Zarr store and should be committed to your experiment log or W&B run for full reproducibility.

### 3. Use in training

`build_loader` auto-selects DALI (if installed) or PyTorch DataLoader — same call either way:

```python
from segdatakit import build_loader

train_loader = build_loader(
    zarr_path="gdrive/MyDrive/cityscapes.zarr",
    split="train",
    batch_size=8,
    size=768,
)
val_loader = build_loader(
    zarr_path="gdrive/MyDrive/cityscapes.zarr",
    split="val",
    batch_size=8,
    size=768,
)
```

If you prefer explicit control:

```python
# standard PyTorch DataLoader
from segdatakit import SegDataset
from segdatakit.transforms import cityscapes_train_transform
from torch.utils.data import DataLoader

ds = SegDataset("gdrive/MyDrive/cityscapes.zarr", split="train",
                transform=cityscapes_train_transform(size=768))
loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)

# explicit DALI loader (requires nvidia-dali installed)
from segdatakit import build_dali_loader

loader = build_dali_loader("gdrive/MyDrive/cityscapes.zarr", split="train",
                            batch_size=8, size=768, device_id=0)
```

Swapping to COCO is one line:

```python
train_loader = build_loader("gdrive/MyDrive/coco.zarr", split="train", batch_size=8)
```

---

## Quickstart — COCO Panoptic

```bash
python scripts/convert.py \
    --cfg configs/coco.yaml \
    --raw /path/to/coco \
    --out coco.zarr

python scripts/validate.py \
    --cfg configs/coco.yaml \
    --zarr coco.zarr \
    --audit-lossless
```

---

## Repository structure

```
segdatakit/
├── segdatakit/
│   ├── __init__.py
│   ├── readers.py        # BaseReader, CityscapesReader, COCOReader
│   ├── transforms.py     # resize, normalize, augment — applied at load time, never on disk
│   ├── writers.py        # ZarrWriter, WebDatasetWriter, NpyWriter
│   ├── validators.py     # lossless audit, sanity checks, class distribution
│   ├── dataloader.py     # SegDataset — PyTorch Dataset over Zarr
│   └── dali_loader.py    # NVIDIA DALI GPU pipeline with PyTorch fallback
├── scripts/
│   ├── convert.py        # CLI: raw dataset → optimized format
│   └── validate.py       # CLI: lossless audit + sanity report
├── configs/
│   ├── cityscapes.yaml
│   ├── coco.yaml
│   └── template.yaml     # starting point for custom datasets
├── tests/
│   ├── test_readers.py
│   ├── test_writers.py
│   └── test_validators.py
├── notebooks/
│   └── colab_quickstart.ipynb   # end-to-end demo on Colab
├── pyproject.toml
└── README.md
```

---

## Config reference

```yaml
# configs/cityscapes.yaml
dataset:
  name: cityscapes
  num_classes: 19
  ignore_index: 255
  splits: [train, val]
  label_map: labelIds_to_trainIds   # remapping applied by reader

storage:
  format: zarr          # zarr | webdataset | npy
  compression: lz4      # lz4 | zstd | none  (all lossless)
  chunk_size: 1         # images per Zarr chunk

paths:
  raw: /data/cityscapes
  output: /data/cityscapes.zarr

validation:
  n_audit_samples: 100
  min_valid_pixel_fraction: 0.30
  max_aspect_ratio_distortion: 0.05
```

---

## Lossless guarantee

Every supported compression codec (`lz4`, `zstd`, `blosc2`) is lossless. Transforms such as resize and normalization are **never applied at write time** — they live in `transforms.py` and run in the dataloader, in memory, at training time. This means:

- Resolution is always the original on disk. You can train at `640×640` or `1024×1024` from the same Zarr file.
- Mean/std normalization is a runtime parameter, not baked into the stored values.
- The stored `uint8` pixel values are identical to the source PNGs, verifiable via SHA-256.

---

## Roadmap

- [x] `readers.py` — `CityscapesReader`, `COCOReader`
- [x] `writers.py` — `ZarrWriter` with Blosc2/LZ4
- [x] `validators.py` — SHA-256 round-trip audit
- [x] `dataloader.py` — `SegDataset` PyTorch integration
- [x] `writers.py` — `WebDatasetWriter`
- [x] `dali_loader.py` — NVIDIA DALI GPU pipeline with PyTorch fallback
- [ ] ADE20K config
- [ ] Colab notebook demo

---

## Contributing

PRs welcome. If you add a new dataset, include the YAML config and a `Reader` subclass. The lossless audit in `tests/test_validators.py` must pass before merging.

---

## License

MIT
