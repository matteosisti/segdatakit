"""
segdatakit/readers.py

Dataset readers for raw segmentation datasets.
Each reader implements a common interface so the rest of the
pipeline never needs to know which dataset it is working with.

Supported:
  - CityscapesReader  (semantic segmentation, 19 classes)
  - COCOReader        (panoptic segmentation, 133 classes)

Adding a new dataset:
  1. Subclass BaseReader
  2. Implement __len__ and __getitem__
  3. Add a YAML config in configs/
  That's it — no other file needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseReader(ABC):
    """
    Common interface for all dataset readers.

    Every __getitem__ must return a dict with exactly these keys:
        image : np.ndarray  shape (H, W, 3)  dtype uint8
        mask  : np.ndarray  shape (H, W)     dtype uint8
        meta  : dict        arbitrary metadata (path, city, frame, ...)
    """

    def __init__(self, cfg: dict, split: str):
        self.cfg   = cfg
        self.split = split
        self._validate_split(split)

    def _validate_split(self, split: str) -> None:
        allowed = self.cfg["dataset"]["splits"]
        if split not in allowed:
            raise ValueError(f"Split '{split}' not in config splits: {allowed}")

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict: ...

    def __iter__(self) -> Iterator[dict]:
        for i in range(len(self)):
            yield self[i]

    def iter_split(self, split: str) -> Iterator[dict]:
        """Iterate over a different split without creating a new reader."""
        original = self.split
        self.split = split
        self._validate_split(split)
        self._reload_paths()
        try:
            yield from self
        finally:
            self.split = original
            self._reload_paths()

    def _reload_paths(self) -> None:
        """Override in subclass if paths depend on self.split."""
        pass


# ---------------------------------------------------------------------------
# Cityscapes
# ---------------------------------------------------------------------------

class CityscapesReader(BaseReader):
    """
    Reader for the official Cityscapes dataset.

    Expected directory structure (standard Cityscapes layout):
        root/
          leftImg8bit/
            train/  val/  test/
              {city}/
                {city}_{frame}_leftImg8bit.png
          gtFine/
            train/  val/  test/
              {city}/
                {city}_{frame}_gtFine_labelIds.png

    The reader remaps raw labelIds (0-33) to trainIds (0-18, 255=ignore)
    using the label_map defined in the YAML config.
    Remapping is applied in __getitem__ — the stored values on disk are
    never modified.
    """

    def __init__(self, cfg: dict, split: str):
        super().__init__(cfg, split)
        raw_root = Path(cfg["paths"]["raw"])
        self._img_root  = raw_root / "leftImg8bit"
        self._mask_root = raw_root / "gtFine"
        self._label_map = self._build_label_map(cfg)
        self._img_paths, self._mask_paths = self._collect_paths(split)

    # ------------------------------------------------------------------
    # Path collection
    # ------------------------------------------------------------------

    def _collect_paths(
        self, split: str
    ) -> tuple[list[Path], list[Path]]:
        img_dir  = self._img_root  / split
        mask_dir = self._mask_root / split

        if not img_dir.exists():
            raise FileNotFoundError(
                f"Cityscapes images not found at {img_dir}.\n"
                f"Check 'paths.raw' in your config points to the Cityscapes root."
            )

        img_paths, mask_paths = [], []

        for city_dir in sorted(img_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
                # derive the corresponding gtFine path
                stem = img_path.stem.replace("_leftImg8bit", "")
                mask_path = (
                    mask_dir / city_dir.name / f"{stem}_gtFine_labelIds.png"
                )
                if not mask_path.exists():
                    raise FileNotFoundError(
                        f"GT mask not found for {img_path.name}.\n"
                        f"Expected: {mask_path}"
                    )
                img_paths.append(img_path)
                mask_paths.append(mask_path)

        if not img_paths:
            raise RuntimeError(
                f"No images found in {img_dir}. "
                f"Check the dataset is extracted correctly."
            )

        return img_paths, mask_paths

    def _reload_paths(self) -> None:
        self._img_paths, self._mask_paths = self._collect_paths(self.split)

    # ------------------------------------------------------------------
    # Label remapping
    # ------------------------------------------------------------------

    @staticmethod
    def _build_label_map(cfg: dict) -> np.ndarray:
        """
        Build a 256-element lookup table for fast label remapping.
        label_map[raw_id] = train_id
        Any id not in the map defaults to ignore_index (255).
        """
        ignore = cfg["dataset"].get("ignore_index", 255)
        lut = np.full(256, fill_value=ignore, dtype=np.uint8)
        for raw_id, train_id in cfg["dataset"]["label_map"].items():
            lut[int(raw_id)] = int(train_id)
        return lut

    def _remap(self, mask: np.ndarray) -> np.ndarray:
        """Apply label_map lookup table to a raw labelIds mask."""
        return self._label_map[mask]

    # ------------------------------------------------------------------
    # BaseReader interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._img_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path  = self._img_paths[idx]
        mask_path = self._mask_paths[idx]

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask  = np.array(Image.open(mask_path), dtype=np.uint8)
        mask  = self._remap(mask)

        return {
            "image": image,
            "mask":  mask,
            "meta": {
                "idx":       idx,
                "img_path":  str(img_path),
                "mask_path": str(mask_path),
                "city":      img_path.parent.name,
                "split":     self.split,
            },
        }


# ---------------------------------------------------------------------------
# COCO Panoptic
# ---------------------------------------------------------------------------

class COCOReader(BaseReader):
    """
    Reader for COCO Panoptic segmentation.

    Expected directory structure:
        root/
          images/
            train2017/   val2017/
          annotations/
            panoptic_train2017.json
            panoptic_val2017.json
            panoptic_train2017/   (PNG panoptic maps)
            panoptic_val2017/

    Each panoptic PNG encodes instance and category via:
        panoptic_id = R + G*256 + B*256^2
    This reader decodes to a semantic mask (category_id per pixel,
    255 for void/crowd regions) compatible with the same interface
    as CityscapesReader.

    Note: COCO uses 1-indexed category ids. This reader remaps them
    to 0-indexed (0..132) using the label_map in the config.
    """

    def __init__(self, cfg: dict, split: str):
        super().__init__(cfg, split)
        import json

        raw_root  = Path(cfg["paths"]["raw"])
        self._img_root = raw_root / "images"
        self._ann_root = raw_root / "annotations"

        split_map = {"train": "train2017", "val": "val2017"}
        if split not in split_map:
            raise ValueError(f"COCOReader supports splits: {list(split_map)}")

        self._coco_split = split_map[split]
        ann_file = self._ann_root / f"panoptic_{self._coco_split}.json"

        if not ann_file.exists():
            raise FileNotFoundError(
                f"COCO annotation file not found: {ann_file}\n"
                f"Check 'paths.raw' in your config."
            )

        with open(ann_file) as f:
            self._ann = json.load(f)

        self._label_map   = self._build_category_map(self._ann, cfg)
        self._annotations = self._ann["annotations"]
        self._img_lookup  = {img["id"]: img for img in self._ann["images"]}

    @staticmethod
    def _build_category_map(ann: dict, cfg: dict) -> dict[int, int]:
        """
        Map COCO category_id (1-indexed, non-contiguous) to
        train_id (0-indexed, contiguous) using label_map from config.
        """
        ignore = cfg["dataset"].get("ignore_index", 255)
        label_map = cfg["dataset"].get("label_map", {})
        if label_map:
            return {int(k): int(v) for k, v in label_map.items()}
        # fallback: sort categories by id and assign 0-indexed train ids
        cats = sorted(ann["categories"], key=lambda c: c["id"])
        return {cat["id"]: i for i, cat in enumerate(cats)}

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, idx: int) -> dict:
        ann   = self._annotations[idx]
        img_info = self._img_lookup[ann["image_id"]]

        img_path = (
            self._img_root / self._coco_split / img_info["file_name"]
        )
        pan_path = (
            self._ann_root
            / f"panoptic_{self._coco_split}"
            / ann["file_name"]
        )

        image     = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        pan_rgb   = np.array(Image.open(pan_path).convert("RGB"),  dtype=np.uint32)

        # decode panoptic id: R + G*256 + B*256^2
        pan_id = pan_rgb[:, :, 0] + pan_rgb[:, :, 1] * 256 + pan_rgb[:, :, 2] * 65536

        # build id → category map from this annotation's segments_info
        id_to_cat = {}
        ignore_index = self.cfg["dataset"].get("ignore_index", 255)
        for seg in ann["segments_info"]:
            if seg.get("iscrowd", 0):
                id_to_cat[seg["id"]] = ignore_index
            else:
                id_to_cat[seg["id"]] = self._label_map.get(
                    seg["category_id"], ignore_index
                )

        # decode semantic mask
        mask = np.full(pan_id.shape, fill_value=ignore_index, dtype=np.uint8)
        for seg_id, cat in id_to_cat.items():
            mask[pan_id == seg_id] = cat

        return {
            "image": image,
            "mask":  mask,
            "meta": {
                "idx":        idx,
                "image_id":   ann["image_id"],
                "img_path":   str(img_path),
                "split":      self.split,
            },
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_reader(cfg: dict, split: str) -> BaseReader:
    """
    Instantiate the correct reader from config.

    Usage:
        import yaml
        cfg = yaml.safe_load(open("configs/cityscapes.yaml"))
        reader = get_reader(cfg, split="val")
    """
    name = cfg["dataset"]["name"].lower()
    readers = {
        "cityscapes": CityscapesReader,
        "coco":       COCOReader,
    }
    if name not in readers:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(readers)}. "
            f"To add a new dataset, subclass BaseReader and register it here."
        )
    return readers[name](cfg, split)
