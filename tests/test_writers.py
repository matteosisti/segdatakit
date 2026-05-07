"""
tests/test_writers.py

Unit tests for segdatakit/writers.py.
Uses synthetic FakeReader — no real dataset required.
Verifies that ZarrWriter produces a store that passes the lossless audit.
"""

from __future__ import annotations

import numpy as np
import pytest

from segdatakit.writers import ZarrWriter, NpyWriter, get_writer
from segdatakit.validators import audit_lossless


# ---------------------------------------------------------------------------
# Helpers (reuse FakeReader from test_validators pattern)
# ---------------------------------------------------------------------------

class FakeReader:
    def __init__(self, n=8, h=64, w=128):
        self.n = n
        self.h = h
        self.w = w
        self.cfg = {
            "dataset": {"name": "fake", "num_classes": 19,
                        "ignore_index": 255, "splits": ["train"]},
            "storage": {"format": "zarr", "compression": "lz4"},
            "paths":   {"raw": "/tmp", "output": "/tmp/fake.zarr"},
        }
        rng = np.random.default_rng(0)
        self._images = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]
        self._masks  = [rng.integers(0, 19,  (h, w),    dtype=np.uint8) for _ in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "image": self._images[idx],
            "mask":  self._masks[idx],
            "meta":  {"idx": idx, "img_path": f"/fake/{idx}.png", "split": "train"},
        }

    def __iter__(self):
        for i in range(self.n):
            yield self[i]


# ---------------------------------------------------------------------------
# ZarrWriter tests
# ---------------------------------------------------------------------------

def test_zarr_writer_creates_store(tmp_path):
    reader = FakeReader(n=4)
    out    = tmp_path / "test.zarr"
    writer = ZarrWriter(reader.cfg, out)
    writer.write(reader)

    import zarr
    store = zarr.open(str(out), mode="r")
    assert "images" in store
    assert "masks"  in store
    assert store["images"].shape == (4, 64, 128, 3)
    assert store["masks"].shape  == (4, 64, 128)


def test_zarr_writer_is_lossless(tmp_path):
    reader = FakeReader(n=8)
    out    = tmp_path / "test.zarr"
    ZarrWriter(reader.cfg, out).write(reader)

    import zarr
    store  = zarr.open(str(out), mode="r")
    report = audit_lossless(reader, store, n_samples=8, seed=0)
    assert report.lossless is True
    assert report.failed   == 0


def test_zarr_writer_stores_split_index(tmp_path):
    reader = FakeReader(n=4)
    out    = tmp_path / "test.zarr"
    ZarrWriter(reader.cfg, out).write(reader)

    import zarr
    store = zarr.open(str(out), mode="r")
    assert "split_index" in store.attrs
    assert "train" in store.attrs["split_index"]


def test_zarr_writer_stores_metadata(tmp_path):
    reader = FakeReader(n=2)
    out    = tmp_path / "test.zarr"
    ZarrWriter(reader.cfg, out).write(reader)

    import zarr
    store = zarr.open(str(out), mode="r")
    assert store.attrs["num_classes"]  == 19
    assert store.attrs["ignore_index"] == 255
    assert store.attrs["dataset"]      == "fake"


def test_zarr_writer_dtype_preserved(tmp_path):
    reader = FakeReader(n=2)
    out    = tmp_path / "test.zarr"
    ZarrWriter(reader.cfg, out).write(reader)

    import zarr
    store = zarr.open(str(out), mode="r")
    assert store["images"].dtype == np.uint8
    assert store["masks"].dtype  == np.uint8


def test_zarr_writer_context_manager(tmp_path):
    reader = FakeReader(n=2)
    out    = tmp_path / "ctx.zarr"
    with ZarrWriter(reader.cfg, out) as writer:
        writer.write(reader)
    import zarr
    store = zarr.open(str(out), mode="r")
    assert store["images"].shape[0] == 2


# ---------------------------------------------------------------------------
# NpyWriter tests
# ---------------------------------------------------------------------------

def test_npy_writer_creates_files(tmp_path):
    reader = FakeReader(n=3)
    out    = tmp_path / "npy_out"
    NpyWriter(reader.cfg, out).write(reader)

    img_files  = sorted((out / "images").glob("*.npy"))
    mask_files = sorted((out / "masks").glob("*.npy"))
    assert len(img_files)  == 3
    assert len(mask_files) == 3


def test_npy_writer_values_correct(tmp_path):
    reader = FakeReader(n=2)
    out    = tmp_path / "npy_out"
    NpyWriter(reader.cfg, out).write(reader)

    img  = np.load(out / "images" / "0000000.npy")
    mask = np.load(out / "masks"  / "0000000.npy")
    np.testing.assert_array_equal(img,  reader[0]["image"])
    np.testing.assert_array_equal(mask, reader[0]["mask"])


# ---------------------------------------------------------------------------
# get_writer factory
# ---------------------------------------------------------------------------

def test_get_writer_returns_zarr(tmp_path):
    cfg = {"storage": {"format": "zarr", "compression": "lz4"},
           "dataset": {"name": "fake"}, "paths": {"output": str(tmp_path / "x.zarr")}}
    writer = get_writer(cfg, tmp_path / "x.zarr")
    assert isinstance(writer, ZarrWriter)


def test_get_writer_returns_npy(tmp_path):
    cfg = {"storage": {"format": "npy"},
           "dataset": {"name": "fake"}, "paths": {"output": str(tmp_path / "npy")}}
    writer = get_writer(cfg, tmp_path / "npy")
    assert isinstance(writer, NpyWriter)


def test_get_writer_unknown_format_raises(tmp_path):
    cfg = {"storage": {"format": "parquet"},
           "dataset": {"name": "fake"}, "paths": {"output": "/tmp"}}
    with pytest.raises(ValueError, match="Unknown format"):
        get_writer(cfg, "/tmp/out")
