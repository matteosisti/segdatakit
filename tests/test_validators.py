"""
tests/test_validators.py

Unit tests for segdatakit/validators.py.
These tests use synthetic data — no real dataset required.
"""

import numpy as np
import pytest

from segdatakit.validators import (
    array_hash,
    pixel_diff,
    check_aspect_ratio,
    check_ood_score_ratio,
    check_dataset_sanity,
    audit_lossless,
    AuditReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_image(h=1024, w=2048, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def make_mask(h=1024, w=2048, num_classes=19, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, num_classes, size=(h, w), dtype=np.uint8)


class FakeReader:
    """Minimal BaseReader stub for testing."""
    def __init__(self, n=10, num_classes=19):
        self.n = n
        self.cfg = {
            "dataset": {"name": "fake", "num_classes": num_classes, "ignore_index": 255},
            "validation": {"min_valid_pixel_fraction": 0.30},
        }
        self._images = [make_image(seed=i) for i in range(n)]
        self._masks  = [make_mask(seed=i)  for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"image": self._images[idx], "mask": self._masks[idx]}


class FakeZarrStore:
    """Mimics zarr.Group with images and masks arrays."""
    def __init__(self, reader: FakeReader, corrupt_idx: int | None = None):
        self._images = [r["image"].copy() for r in (reader[i] for i in range(len(reader)))]
        self._masks  = [r["mask"].copy()  for r in (reader[i] for i in range(len(reader)))]
        if corrupt_idx is not None:
            # flip one pixel to simulate a lossy conversion
            self._images[corrupt_idx][0, 0, 0] ^= 1

    class _Arr:
        def __init__(self, data):
            self._data = data
        def __getitem__(self, idx):
            return self._data[idx]

    @property
    def images(self):
        return self._Arr(self._images)

    @property
    def masks(self):
        return self._Arr(self._masks)

    def __getitem__(self, key):
        return getattr(self, key)


# ---------------------------------------------------------------------------
# array_hash
# ---------------------------------------------------------------------------

def test_hash_identical_arrays():
    arr = make_image(seed=7)
    assert array_hash(arr) == array_hash(arr.copy())


def test_hash_different_arrays():
    a = make_image(seed=1)
    b = make_image(seed=2)
    assert array_hash(a) != array_hash(b)


def test_hash_single_pixel_change():
    a = make_image(seed=3)
    b = a.copy()
    b[0, 0, 0] ^= 1
    assert array_hash(a) != array_hash(b)


# ---------------------------------------------------------------------------
# pixel_diff
# ---------------------------------------------------------------------------

def test_pixel_diff_identical():
    arr = make_image(seed=5)
    result = pixel_diff(arr, arr.copy())
    assert result.is_lossless is True
    assert result.max_abs_error == 0
    assert result.num_different_pixels == 0


def test_pixel_diff_single_error():
    a = make_image(seed=5)
    b = a.copy()
    b[10, 20, 1] = int(b[10, 20, 1]) ^ 100
    result = pixel_diff(a, b)
    assert result.is_lossless is False
    assert result.num_different_pixels == 1
    assert result.max_abs_error == 100


def test_pixel_diff_no_uint8_wraparound():
    # if we had 0 - 255 in uint8 it would wrap to 1, not -255
    a = np.array([[[0]]], dtype=np.uint8)
    b = np.array([[[255]]], dtype=np.uint8)
    result = pixel_diff(a, b)
    assert result.max_abs_error == 255   # correct, no wraparound


# ---------------------------------------------------------------------------
# audit_lossless
# ---------------------------------------------------------------------------

def test_audit_lossless_passes_on_perfect_copy():
    reader = FakeReader(n=10)
    store  = FakeZarrStore(reader)
    report = audit_lossless(reader, store, n_samples=10, seed=42)
    assert report.lossless is True
    assert report.failed == 0
    assert report.passed == 10


def test_audit_lossless_detects_corruption():
    reader = FakeReader(n=10)
    store  = FakeZarrStore(reader, corrupt_idx=3)
    report = audit_lossless(reader, store, n_samples=10, seed=42)
    assert report.lossless is False
    assert report.failed >= 1
    assert any(v["idx"] == 3 for v in report.violations)


def test_audit_report_is_serialisable(tmp_path):
    reader = FakeReader(n=5)
    store  = FakeZarrStore(reader)
    report = audit_lossless(reader, store, n_samples=5, seed=0)
    out = tmp_path / "audit.json"
    report.save(out)
    import json
    data = json.loads(out.read_text())
    assert data["lossless"] is True
    assert "timestamp" in data


# ---------------------------------------------------------------------------
# check_aspect_ratio
# ---------------------------------------------------------------------------

def test_aspect_ratio_no_distortion():
    result = check_aspect_ratio((1024, 2048), (512, 1024))
    assert result["exceeds_threshold"] is False
    assert result["distortion"] == pytest.approx(0.0, abs=1e-6)


def test_aspect_ratio_square_resize_distorts():
    # Cityscapes 1024x2048 → 640x640 is heavily distorted
    result = check_aspect_ratio((1024, 2048), (640, 640))
    assert result["exceeds_threshold"] is True
    assert result["distortion_pct"] > 50


def test_aspect_ratio_small_distortion_ok():
    # slight padding resize within threshold
    result = check_aspect_ratio((1024, 2048), (512, 1020))
    assert result["exceeds_threshold"] is False


# ---------------------------------------------------------------------------
# check_ood_score_ratio
# ---------------------------------------------------------------------------

def test_ood_ratio_correct_direction():
    anomaly = np.array([0.1, 0.2, 0.9, 0.8], dtype=np.float32)
    gt      = np.array([0,   0,   1,   1],   dtype=np.uint8)
    result  = check_ood_score_ratio(anomaly, gt)
    assert result["direction_correct"] is True
    assert result["ratio_ood_vs_inlier"] > 1.0
    assert result["warning"] is None


def test_ood_ratio_wrong_direction():
    anomaly = np.array([0.9, 0.8, 0.1, 0.2], dtype=np.float32)
    gt      = np.array([0,   0,   1,   1],   dtype=np.uint8)
    result  = check_ood_score_ratio(anomaly, gt)
    assert result["direction_correct"] is False
    assert result["ratio_ood_vs_inlier"] < 1.0
    assert result["warning"] is not None


# ---------------------------------------------------------------------------
# check_dataset_sanity
# ---------------------------------------------------------------------------

def test_sanity_passes_clean_reader():
    reader = FakeReader(n=20, num_classes=19)
    result = check_dataset_sanity(reader, n_samples=10)
    assert result["all_passed"] is True
    assert result["class_range_ok"] is True
    assert result["valid_pixel_fraction_ok"] is True
