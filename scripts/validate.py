"""
scripts/validate.py

CLI: validate a converted dataset store.

Usage:
    # full lossless audit
    python scripts/validate.py --cfg configs/cityscapes.yaml --zarr cityscapes.zarr --audit-lossless

    # sanity check on raw dataset only
    python scripts/validate.py --cfg configs/cityscapes.yaml --sanity-only

    # both
    python scripts/validate.py --cfg configs/cityscapes.yaml --zarr cityscapes.zarr --audit-lossless --report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate a segdatakit dataset store for lossless integrity."
    )
    p.add_argument("--cfg",             required=True)
    p.add_argument("--raw",             default=None,  help="Override paths.raw")
    p.add_argument("--zarr",            default=None,  help="Path to .zarr store to audit")
    p.add_argument("--split",           default="val", help="Split to audit (default: val)")
    p.add_argument("--audit-lossless",  action="store_true")
    p.add_argument("--sanity-only",     action="store_true")
    p.add_argument("--n-samples",       type=int, default=100)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--report",          action="store_true", help="Save JSON report")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    if args.raw:
        cfg["paths"]["raw"] = args.raw

    from segdatakit.readers import get_reader
    from segdatakit.validators import check_dataset_sanity, audit_lossless

    reader = get_reader(cfg, split=args.split)
    print(f"Dataset : {cfg['dataset']['name']}  split={args.split}  n={len(reader)}")

    # sanity check on raw data
    print("\nRunning dataset sanity checks...")
    sanity = check_dataset_sanity(reader, n_samples=min(args.n_samples, 20))
    for key, val in sanity.items():
        if key not in ("issues", "shapes_seen", "samples_checked"):
            icon = "✓" if val is True else ("✗" if val is False else " ")
            print(f"  [{icon}] {key}: {val}")
    if sanity["issues"]:
        print("\n  Issues found:")
        for issue in sanity["issues"]:
            print(f"    - {issue}")

    if args.sanity_only:
        sys.exit(0 if sanity["all_passed"] else 1)

    # lossless audit
    if args.audit_lossless:
        if not args.zarr:
            print("[error] --zarr is required for --audit-lossless")
            sys.exit(1)

        import zarr
        zarr_path = Path(args.zarr)
        if not zarr_path.exists():
            print(f"[error] Zarr store not found: {zarr_path}")
            sys.exit(1)

        print(f"\nRunning lossless audit ({args.n_samples} samples, seed={args.seed})...")
        store  = zarr.open(str(zarr_path), mode="r")
        report = audit_lossless(
            reader, store,
            n_samples=args.n_samples,
            seed=args.seed,
            zarr_path=str(zarr_path),
        )
        report.sanity = sanity
        report.print_summary()

        if args.report:
            out = zarr_path.parent / f"audit_{args.split}.json"
            report.save(out)
            print(f"Report saved: {out}")

        sys.exit(0 if report.lossless else 1)


if __name__ == "__main__":
    main()
