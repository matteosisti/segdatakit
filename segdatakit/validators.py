"""
segdatakit/validators.py

Lossless round-trip verification and dataset sanity checks.
All checks are deterministic and produce a JSON audit report
that can be committed alongside the converted dataset.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from segdatakit.readers import BaseReader


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PixelDiffResult:
    max_abs_error: int
    mean_abs_error: float
    num_different_pixels: int
    pct_different: float
    is_lossless: bool


@dataclass
class SampleAuditResult:
    idx: int
    image_hash_original: str
    image_hash_reconstructed: str
    mask_hash_original: str
    mask_hash_reconstructed: str
    image_diff: PixelDiffResult
    mask_diff: PixelDiffResult
    passed: bool


@dataclass
class AuditReport:
    dataset: str
    zarr_path: str
    timestamp: str
    n_total: int
    n_sampled: int
    seed: int
    passed: int
    failed: int
    lossless: bool
    violations: list[dict] = field(default_factory=list)
    sanity: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    def print_summary(self) -> None:
        status = "LOSSLESS VERIFIED" if self.lossless else "LOSSLESS VIOLATED"
        print(f"\nLossless audit — {self.dataset} → {self.zarr_path}")
        print(f"  Samples tested  : {self.n_sampled} / {self.n_total}")
        print(f"  Images passed   : {self.passed} / {self.n_sampled}")
        print(f"  Masks passed    : {self.passed} / {self.n_sampled}")
        print(f"  Result          : {status}")
        if self.violations:
            print(f"  Violations      : {len(self.violations)}")
            for v in self.violations[:3]:
                print(f"    idx={v['idx']} max_err={v['image_diff']['max_abs_error']}")
        print()


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def array_hash(arr: np.ndarray) -> str:
    """SHA-256 of a numpy array. Deterministic regardless of dtype differences."""
    return hashlib.sha256(arr.astype(np.uint8).tobytes()).hexdigest()


def pixel_diff(original: np.ndarray, reconstructed: np.ndarray) -> PixelDiffResult:
    """
    Compute pixel-level difference between two uint8 arrays.
    Cast to int16 before subtraction to avoid uint8 wraparound.
    """
    a = original.astype(np.int16)
    b = reconstructed.astype(np.int16)
    diff = a - b
    abs_diff = np.abs(diff)
    return PixelDiffResult(
        max_abs_error=int(abs_diff.max()),
        mean_abs_error=float(abs_diff.mean()),
        num_different_pixels=int((diff != 0).sum()),
        pct_different=float((diff != 0).mean() * 100),
        is_lossless=bool((diff == 0).all()),
    )


# ---------------------------------------------------------------------------
# Round-trip audit
# ---------------------------------------------------------------------------

def audit_lossless(
    reader: "BaseReader",
    store,                   # zarr.Group opened in read mode
    n_samples: int = 100,
    seed: int = 42,
    zarr_path: str = "",
) -> AuditReport:
    """
    Verify lossless conversion by comparing n_samples random items
    between the original reader and the Zarr store.

    Parameters
    ----------
    reader      : BaseReader instance for the raw dataset
    store       : zarr.Group opened with zarr.open(path, mode="r")
    n_samples   : number of samples to check (default 100)
    seed        : random seed for reproducibility
    zarr_path   : path string for the report (cosmetic)

    Returns
    -------
    AuditReport with full details. Call .save() to persist as JSON.
    """
    rng = random.Random(seed)
    n_total = len(reader)
    indices = rng.sample(range(n_total), min(n_samples, n_total))

    passed = 0
    failed = 0
    violations = []

    for idx in indices:
        original = reader[idx]          # {"image": np.ndarray, "mask": np.ndarray}
        img_orig  = original["image"]
        mask_orig = original["mask"]

        img_recon  = np.array(store["images"][idx])
        mask_recon = np.array(store["masks"][idx])

        img_diff  = pixel_diff(img_orig, img_recon)
        mask_diff = pixel_diff(mask_orig, mask_recon)

        sample_passed = img_diff.is_lossless and mask_diff.is_lossless

        result = SampleAuditResult(
            idx=idx,
            image_hash_original=array_hash(img_orig),
            image_hash_reconstructed=array_hash(img_recon),
            mask_hash_original=array_hash(mask_orig),
            mask_hash_reconstructed=array_hash(mask_recon),
            image_diff=img_diff,
            mask_diff=mask_diff,
            passed=sample_passed,
        )

        if sample_passed:
            passed += 1
        else:
            failed += 1
            violations.append(asdict(result))

    cfg = reader.cfg
    report = AuditReport(
        dataset=cfg.get("dataset", {}).get("name", "unknown"),
        zarr_path=zarr_path,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        n_total=n_total,
        n_sampled=len(indices),
        seed=seed,
        passed=passed,
        failed=failed,
        lossless=(failed == 0),
        violations=violations,
    )

    return report


# ---------------------------------------------------------------------------
# Sanity checks (independent of lossless audit)
# ---------------------------------------------------------------------------

def check_dataset_sanity(reader: "BaseReader", n_samples: int = 20) -> dict:
    """
    Quick structural checks on the raw dataset before conversion.
    Returns a dict with check results — all should be True.
    """
    cfg = reader.cfg
    num_classes = cfg["dataset"]["num_classes"]
    ignore_index = cfg["dataset"].get("ignore_index", 255)
    min_valid_frac = cfg.get("validation", {}).get("min_valid_pixel_fraction", 0.30)

    results = {
        "len_nonzero": len(reader) > 0,
        "samples_checked": 0,
        "shape_consistent": True,
        "class_range_ok": True,
        "valid_pixel_fraction_ok": True,
        "issues": [],
    }

    rng = random.Random(42)
    indices = rng.sample(range(len(reader)), min(n_samples, len(reader)))

    shapes_seen = set()
    for idx in indices:
        sample = reader[idx]
        img   = sample["image"]
        mask  = sample["mask"]

        shapes_seen.add(img.shape[:2])

        # check class range
        valid_mask = mask[mask != ignore_index]
        if valid_mask.size > 0 and int(valid_mask.max()) >= num_classes:
            results["class_range_ok"] = False
            results["issues"].append(
                f"idx={idx}: mask contains class {valid_mask.max()} >= num_classes={num_classes}"
            )

        # check valid pixel fraction
        valid_frac = float((mask != ignore_index).mean())
        if valid_frac < min_valid_frac:
            results["valid_pixel_fraction_ok"] = False
            results["issues"].append(
                f"idx={idx}: valid pixel fraction {valid_frac:.2%} < {min_valid_frac:.2%}"
            )

    if len(shapes_seen) > 1:
        results["shape_consistent"] = False
        results["issues"].append(f"Inconsistent image shapes: {shapes_seen}")

    results["samples_checked"] = len(indices)
    results["shapes_seen"] = [list(s) for s in shapes_seen]
    results["all_passed"] = all([
        results["len_nonzero"],
        results["shape_consistent"],
        results["class_range_ok"],
        results["valid_pixel_fraction_ok"],
    ])

    return results


def check_aspect_ratio(original_hw: tuple[int, int], resized_hw: tuple[int, int]) -> dict:
    """
    Check if a resize operation introduces significant aspect ratio distortion.
    Returns a dict with distortion value and whether it exceeds the threshold.
    """
    orig_ratio   = original_hw[1] / original_hw[0]  # W/H
    resized_ratio = resized_hw[1] / resized_hw[0]
    distortion   = abs(orig_ratio - resized_ratio) / orig_ratio

    return {
        "original_hw": original_hw,
        "resized_hw": resized_hw,
        "original_ratio": round(orig_ratio, 4),
        "resized_ratio": round(resized_ratio, 4),
        "distortion": round(distortion, 4),
        "distortion_pct": round(distortion * 100, 2),
        "exceeds_threshold": distortion > 0.05,
    }


def check_ood_score_ratio(anomaly_map: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    Verify that anomaly scores are higher on OoD pixels than on inliers.
    A ratio < 1.0 indicates the scoring direction is inverted — likely a bug.

    Parameters
    ----------
    anomaly_map : float array HxW of anomaly scores
    gt_mask     : binary array HxW, 1 = OoD pixel, 0 = inlier
    """
    ood_pixels    = anomaly_map[gt_mask == 1]
    inlier_pixels = anomaly_map[gt_mask == 0]

    mean_ood    = float(ood_pixels.mean())    if ood_pixels.size    > 0 else 0.0
    mean_inlier = float(inlier_pixels.mean()) if inlier_pixels.size > 0 else 1e-8

    ratio = mean_ood / (mean_inlier + 1e-8)

    return {
        "mean_score_ood": round(mean_ood, 6),
        "mean_score_inlier": round(mean_inlier, 6),
        "ratio_ood_vs_inlier": round(ratio, 4),
        "direction_correct": ratio >= 1.0,
        "warning": None if ratio >= 1.0 else (
            f"Anomaly score ratio {ratio:.3f} < 1.0 — "
            "scores are lower on OoD than inlier pixels. "
            "Check for inverted score, wrong mask, or resize bug."
        ),
    }