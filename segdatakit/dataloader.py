"""
segdatakit/dataloader.py

PyTorch Dataset implementations over the converted dataset formats.

SegDataset  — reads from a Zarr store (primary, recommended)
NpyDataset  — reads from plain .npy files (fallback)

Usage:
    from segdatakit import SegDataset
    from segdatakit.transforms import cityscapes_train_transform

    ds = SegDataset(
        path="gdrive/MyDrive/cityscapes.zarr",
        split="train",
        transform=cityscapes_train_transform(size=768),
    )
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Zarr-backed Dataset
# ---------------------------------------------------------------------------

class SegDataset(Dataset):
    """
    PyTorch Dataset over a segdatakit Zarr store.

    Lazy access: only the requested chunk (one image) is decompressed
    per __getitem__ call. The full store is never loaded into RAM.

    Parameters
    ----------
    path      : path to the .zarr store (local or mounted Drive path)
    split     : "train" | "val" | "test"
    transform : optional callable (image, mask) → (image, mask)
                use segdatakit.transforms presets or compose your own
    """

    def __init__(
        self,
        path: str | Path,
        split: str,
        transform=None,
    ):
        import zarr

        self.path      = Path(path)
        self.split     = split
        self.transform = transform

        if not self.path.exists():
            raise FileNotFoundError(
                f"Zarr store not found: {self.path}\n"
                f"Run scripts/convert.py first to generate it."
            )

        self._store = zarr.open(str(self.path), mode="r")
        self._validate_store()
        self._indices = self._load_split_indices(split)

        self.num_classes  = int(self._store.attrs.get("num_classes",  19))
        self.ignore_index = int(self._store.attrs.get("ignore_index", 255))

    def _validate_store(self) -> None:
        for key in ("images", "masks"):
            if key not in self._store:
                raise RuntimeError(
                    f"Zarr store at {self.path} is missing '{key}' array. "
                    f"It may be corrupted or created with an older version."
                )

    def _load_split_indices(self, split: str) -> list[int]:
        split_index = self._store.attrs.get("split_index", {})
        if not split_index:
            # no split info stored → use all indices
            return list(range(len(self._store["images"])))
        if split not in split_index:
            available = list(split_index.keys())
            raise ValueError(
                f"Split '{split}' not found in store. Available: {available}"
            )
        return split_index[split]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        store_idx = self._indices[idx]
        image = np.array(self._store["images"][store_idx], dtype=np.uint8)
        mask  = np.array(self._store["masks"][store_idx],  dtype=np.uint8)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # if transform did not convert to tensor, do it here
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = torch.from_numpy(
                    image.transpose(2, 0, 1).astype(np.float32) / 255.0
                )
            else:
                image = torch.from_numpy(
                    np.ascontiguousarray(image.transpose(2, 0, 1))
                )
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask.astype(np.int64))

        return image, mask

    def __repr__(self) -> str:
        return (
            f"SegDataset(path={self.path.name}, split={self.split}, "
            f"n={len(self)}, num_classes={self.num_classes})"
        )


# ---------------------------------------------------------------------------
# Npy-backed Dataset  (fallback)
# ---------------------------------------------------------------------------

class NpyDataset(Dataset):
    """
    PyTorch Dataset over plain .npy files written by NpyWriter.

    Slower than SegDataset for random access but requires no zarr dependency.
    """

    def __init__(self, path: str | Path, transform=None):
        self.img_dir   = Path(path) / "images"
        self.mask_dir  = Path(path) / "masks"
        self.transform = transform

        if not self.img_dir.exists():
            raise FileNotFoundError(f"images/ not found in {path}")

        self._img_paths  = sorted(self.img_dir.glob("*.npy"))
        self._mask_paths = sorted(self.mask_dir.glob("*.npy"))

        if len(self._img_paths) != len(self._mask_paths):
            raise RuntimeError(
                f"Mismatch: {len(self._img_paths)} images vs "
                f"{len(self._mask_paths)} masks in {path}"
            )

    def __len__(self) -> int:
        return len(self._img_paths)

    def __getitem__(self, idx: int):
        image = np.load(self._img_paths[idx])
        mask  = np.load(self._mask_paths[idx])

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(
                image.transpose(2, 0, 1).astype(np.float32) / 255.0
            )
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask.astype(np.int64))

        return image, mask
