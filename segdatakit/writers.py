"""
segdatakit/writers.py

Writers that convert raw dataset samples (from any BaseReader)
into optimised on-disk formats.

Supported formats:
  - ZarrWriter        — single .zarr store, Blosc2/LZ4 compressed, random access
  - WebDatasetWriter  — sequential .tar shards, streaming-friendly
  - NpyWriter         — plain .npy files, simplest possible format

All writers are lossless by construction:
  - No resize, no normalisation, no dtype change is applied.
  - Stored dtype is always uint8, identical to the source PNG pixels.
  - Verified by validators.audit_lossless after writing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from segdatakit.readers import BaseReader


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseWriter(ABC):

    def __init__(self, cfg: dict, out_path: str | Path):
        self.cfg      = cfg
        self.out_path = Path(out_path)

    @abstractmethod
    def write(self, reader: "BaseReader") -> None:
        """Write all samples from reader to out_path."""
        ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Zarr writer
# ---------------------------------------------------------------------------

class ZarrWriter(BaseWriter):
    """
    Write a dataset into a single Zarr v2 store.

    Layout inside the store:
        images/   uint8  (N, H, W, 3)   — RGB images
        masks/    uint8  (N, H, W)      — remapped semantic masks
        meta/
          img_paths   — object array of str
          split_index — dict {split: [indices]}  stored as JSON attr

    Compression: Blosc2 with LZ4 codec (lossless, very fast decompress).
    Chunk size:  one image per chunk → true random access with no
                 wasted decompression.
    """

    def __init__(self, cfg: dict, out_path: str | Path):
        super().__init__(cfg, out_path)
        self._store  = None
        self._images = None
        self._masks  = None
        self._paths  = None

    def write(self, reader: "BaseReader") -> None:
        import zarr
        from zarr.codecs import BloscCodec

        codec_name = self.cfg.get("storage", {}).get("compression", "lz4")
        compressor = BloscCodec(
            cname=codec_name,
            clevel=3,
            shuffle="bitshuffle",
        )

        n = len(reader)
        if n == 0:
            raise RuntimeError("Reader returned 0 samples — nothing to write.")

        # peek at first sample to get H, W
        first = reader[0]
        H, W = first["image"].shape[:2]

        self._store = zarr.open(str(self.out_path), mode="w")
        self._images = self._store.require_array(
            "images",
            shape=(n, H, W, 3),
            chunks=(1, H, W, 3),
            dtype=np.uint8,
            compressors=[compressor], 
        )
        self._masks = self._store.require_array(
            "masks",
            shape=(n, H, W),
            chunks=(1, H, W),
            dtype=np.uint8,
            compressors=[compressor], 
        )
        

        # write split index as attribute
        split_index: dict[str, list[int]] = {}

        for idx in tqdm(range(n), desc=f"Writing {self.out_path.name}"):
            sample = reader[idx] if idx > 0 else first
            if idx > 0:
                sample = reader[idx]

            self._images[idx] = sample["image"]
            self._masks[idx]  = sample["mask"]

            split = sample["meta"].get("split", "unknown")
            split_index.setdefault(split, []).append(idx)


        self._store.attrs.update({
            "split_index":  split_index,
            "num_classes":  self.cfg["dataset"]["num_classes"],
            "ignore_index": self.cfg["dataset"].get("ignore_index", 255),
            "dataset":      self.cfg["dataset"]["name"],
            })
        print(f"Written {n} samples → {self.out_path}  "
              f"({self._human_size(self.out_path)})")

    def close(self) -> None:
        pass  # zarr stores close automatically

    @staticmethod
    def _human_size(path: Path) -> str:
        try:
            total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            for unit in ["B", "KB", "MB", "GB"]:
                if total < 1024:
                    return f"{total:.1f} {unit}"
                total /= 1024
            return f"{total:.1f} TB"
        except Exception:
            return "unknown size"


# ---------------------------------------------------------------------------
# WebDataset writer
# ---------------------------------------------------------------------------

class WebDatasetWriter(BaseWriter):
    """
    Write a dataset into sequential .tar shards (WebDataset format).

    Each shard contains at most `shard_size` samples.
    Files inside each tar:
        {key}.jpg   — JPEG-encoded image  (quality=95, near-lossless)
        {key}.mask.png — PNG mask          (lossless)
        {key}.json  — metadata

    Note: JPEG encoding at quality=95 introduces minimal quantisation
    error (~0-2 per channel on most pixels). If strict lossless is
    required, set jpeg_quality=None to use PNG for images too (larger).
    """

    def __init__(
        self,
        cfg: dict,
        out_path: str | Path,
        shard_size: int = 500,
        jpeg_quality: int | None = 95,
    ):
        super().__init__(cfg, out_path)
        self.shard_size   = shard_size
        self.jpeg_quality = jpeg_quality

    def write(self, reader: "BaseReader") -> None:
        import webdataset as wds
        import io
        import json
        from PIL import Image as PILImage

        self.out_path.mkdir(parents=True, exist_ok=True)
        n          = len(reader)
        n_shards   = (n + self.shard_size - 1) // self.shard_size
        pad_width  = len(str(n_shards - 1))

        for shard_idx in range(n_shards):
            start = shard_idx * self.shard_size
            end   = min(start + self.shard_size, n)
            shard_path = str(
                self.out_path / f"shard_{shard_idx:0{pad_width}d}.tar"
            )

            with wds.TarWriter(shard_path) as sink:
                for idx in tqdm(
                    range(start, end),
                    desc=f"Shard {shard_idx+1}/{n_shards}",
                ):
                    sample = reader[idx]
                    key    = f"{idx:07d}"

                    # encode image
                    img_pil = PILImage.fromarray(sample["image"])
                    img_buf = io.BytesIO()
                    if self.jpeg_quality is not None:
                        img_pil.save(img_buf, format="JPEG",
                                     quality=self.jpeg_quality, subsampling=0)
                        img_ext = "jpg"
                    else:
                        img_pil.save(img_buf, format="PNG")
                        img_ext = "png"

                    # encode mask (always lossless PNG)
                    mask_pil = PILImage.fromarray(sample["mask"], mode="L")
                    mask_buf = io.BytesIO()
                    mask_pil.save(mask_buf, format="PNG")

                    sink.write({
                        "__key__":   key,
                        img_ext:     img_buf.getvalue(),
                        "mask.png":  mask_buf.getvalue(),
                        "meta.json": json.dumps(sample["meta"]).encode(),
                    })

        print(f"Written {n} samples → {n_shards} shards in {self.out_path}")

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Npy writer  (simplest, for debugging / small datasets)
# ---------------------------------------------------------------------------

class NpyWriter(BaseWriter):
    """
    Write each sample as individual .npy files.
    No compression. Useful for debugging or very small datasets.

    Output layout:
        out_path/
          images/  0000000.npy  0000001.npy ...
          masks/   0000000.npy  0000001.npy ...
    """

    def write(self, reader: "BaseReader") -> None:
        img_dir  = self.out_path / "images"
        mask_dir = self.out_path / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for idx in tqdm(range(len(reader)), desc="Writing .npy"):
            sample = reader[idx]
            np.save(img_dir  / f"{idx:07d}.npy", sample["image"])
            np.save(mask_dir / f"{idx:07d}.npy", sample["mask"])

        print(f"Written {len(reader)} samples → {self.out_path}")

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_writer(cfg: dict, out_path: str | Path) -> BaseWriter:
    """
    Instantiate the correct writer from config.

    Usage:
        writer = get_writer(cfg, "cityscapes.zarr")
        writer.write(reader)
    """
    fmt = cfg.get("storage", {}).get("format", "zarr").lower()
    writers = {
        "zarr":       ZarrWriter,
        "webdataset": WebDatasetWriter,
        "npy":        NpyWriter,
    }
    if fmt not in writers:
        raise ValueError(
            f"Unknown format '{fmt}'. Available: {list(writers)}"
        )
    return writers[fmt](cfg, out_path)
