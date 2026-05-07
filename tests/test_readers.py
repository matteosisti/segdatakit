"""
tests/test_readers.py

Unit tests for segdatakit/readers.py.
Uses a synthetic on-disk fixture (tmp_path) — no real dataset needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from segdatakit.readers import CityscapesReader, get_reader, BaseReader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CITYSCAPES_CFG = {
    "dataset": {
        "name": "cityscapes",
        "num_classes": 19,
        "ignore_index": 255,
        "splits": ["train", "val"],
        "label_map": {
            7: 0,   # road
            8: 1,   # sidewalk
            11: 2,  # building
            26: 13, # car
        },
    },
    "paths": {"raw": ""},  # filled per test
}


def make_cityscapes_fixture(root: Path, split: str = "val", n_cities: int = 2, n_per_city: int = 3):
    """
    Create a minimal synthetic Cityscapes directory structure.
    Images: 64x128 RGB PNG (small for speed)
    Masks:  64x128 grayscale PNG with raw labelIds
    """
    rng = np.random.default_rng(42)
    cities = [f"city{i:02d}" for i in range(n_cities)]
    valid_raw_ids = [7, 8, 11, 26]  # ids that map to trainIds in cfg

    for city in cities:
        img_dir  = root / "leftImg8bit" / split / city
        mask_dir = root / "gtFine"      / split / city
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        for frame in range(n_per_city):
            stem = f"{city}_00000{frame}"

            # image
            img_arr = rng.integers(0, 256, (64, 128, 3), dtype=np.uint8)
            Image.fromarray(img_arr).save(img_dir / f"{stem}_leftImg8bit.png")

            # mask — use only valid raw ids so remap produces non-ignore pixels
            raw_ids = np.array(valid_raw_ids, dtype=np.uint8)
            mask_arr = rng.choice(raw_ids, size=(64, 128)).astype(np.uint8)
            Image.fromarray(mask_arr, mode="L").save(
                mask_dir / f"{stem}_gtFine_labelIds.png"
            )

    return root


# ---------------------------------------------------------------------------
# CityscapesReader tests
# ---------------------------------------------------------------------------

def test_reader_length(tmp_path):
    make_cityscapes_fixture(tmp_path, split="val", n_cities=2, n_per_city=3)
    cfg = {**CITYSCAPES_CFG, "paths": {"raw": str(tmp_path)}}
    reader = CityscapesReader(cfg, split="val")
    assert len(reader) == 6  # 2 cities × 3 images


def test_reader_item_shapes(tmp_path):
    make_cityscapes_fixture(tmp_path, split="val", n_cities=1, n_per_city=2)
    cfg = {**CITYSCAPES_CFG, "paths": {"raw": str(tmp_path)}}
    reader = CityscapesReader(cfg, split="val")
    sample = reader[0]

    assert "image" in sample
    assert "mask"  in sample
    assert "meta"  in sample
    assert sample["image"].shape == (64, 128, 3)
    assert sample["mask"].shape  == (64, 128)
    assert sample["image"].dtype == np.uint8
    assert sample["mask"].dtype  == np.uint8


def test_reader_label_remapping(tmp_path):
    """
    Raw labelId 7 (road) must become trainId 0.
    Raw labelId 26 (car) must become trainId 13.
    Raw labelId not in map must become ignore_index (255).
    """
    make_cityscapes_fixture(tmp_path, split="val", n_cities=1, n_per_city=1)
    cfg = {**CITYSCAPES_CFG, "paths": {"raw": str(tmp_path)}}
    reader = CityscapesReader(cfg, split="val")

    # check LUT directly
    lut = reader._label_map
    assert lut[7]  == 0    # road → 0
    assert lut[8]  == 1    # sidewalk → 1
    assert lut[11] == 2    # building → 2
    assert lut[26] == 13   # car → 13
    assert lut[0]  == 255  # unlabeled → ignore
    assert lut[33] == 255  # not in map → ignore


def test_reader_mask_contains_only_valid_classes(tmp_path):
    make_cityscapes_fixture(tmp_path, split="val", n_cities=1, n_per_city=3)
    cfg = {**CITYSCAPES_CFG, "paths": {"raw": str(tmp_path)}}
    reader = CityscapesReader(cfg, split="val")

    for sample in reader:
        mask = sample["mask"]
        valid = mask[mask != 255]
        if valid.size > 0:
            assert int(valid.max()) < cfg["dataset"]["num_classes"]


def test_reader_meta_contains_expected_keys(tmp_path):
    make_cityscapes_fixture(tmp_path, split="val", n_cities=1, n_per_city=1)
    cfg = {**CITYSCAPES_CFG, "paths": {"raw": str(tmp_path)}}
    reader = CityscapesReader(cfg, split="val")
    meta = reader[0]["meta"]

    assert "idx"       in meta
    assert "img_path"  in meta
    assert "mask_path" in meta
    assert "city"      in meta
    assert "split"     in meta
    assert meta["split"] == "val"


def test_reader_iteration(tmp_path):
    make_cityscapes_fixture(tmp_path, split="val", n_cities=2, n_per_city=2)
    cfg = {**CITYSCAPES_CFG, "paths": {"raw": str(tmp_path)}}
    reader = CityscapesReader(cfg, split="val")

    samples = list(reader)
    assert len(samples) == 4
    for s in samples:
        assert s["image"].ndim == 3
        assert s["mask"].ndim  == 2


def test_reader_invalid_split_raises(tmp_path):
    make_cityscapes_fixture(tmp_path, split="val")
    cfg = {**CITYSCAPES_CFG, "paths": {"raw": str(tmp_path)}}
    with pytest.raises(ValueError, match="Split"):
        CityscapesReader(cfg, split="test_nonexistent")


def test_reader_missing_root_raises(tmp_path):
    cfg = {**CITYSCAPES_CFG, "paths": {"raw": str(tmp_path / "nonexistent")}}
    with pytest.raises(FileNotFoundError):
        CityscapesReader(cfg, split="val")


# ---------------------------------------------------------------------------
# get_reader factory
# ---------------------------------------------------------------------------

def test_get_reader_returns_cityscapes(tmp_path):
    make_cityscapes_fixture(tmp_path, split="val")
    cfg = {**CITYSCAPES_CFG, "paths": {"raw": str(tmp_path)}}
    reader = get_reader(cfg, split="val")
    assert isinstance(reader, CityscapesReader)


def test_get_reader_unknown_dataset_raises():
    cfg = {
        "dataset": {"name": "imagenet", "splits": ["val"], "num_classes": 1000,
                    "ignore_index": 255, "label_map": {}},
        "paths": {"raw": "/tmp"},
    }
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_reader(cfg, split="val")


# ---------------------------------------------------------------------------
# BaseReader interface contract
# ---------------------------------------------------------------------------

def test_base_reader_is_abstract():
    """Cannot instantiate BaseReader directly."""
    with pytest.raises(TypeError):
        BaseReader({}, "val")  # type: ignore


def test_reader_getitem_returns_correct_keys(tmp_path):
    make_cityscapes_fixture(tmp_path, split="val", n_cities=1, n_per_city=1)
    cfg = {**CITYSCAPES_CFG, "paths": {"raw": str(tmp_path)}}
    reader = CityscapesReader(cfg, split="val")
    sample = reader[0]
    assert set(sample.keys()) == {"image", "mask", "meta"}
