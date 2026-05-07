"""
scripts/convert.py

CLI: convert a raw segmentation dataset into an optimised format.

Usage:
    python scripts/convert.py --cfg configs/cityscapes.yaml --raw /data/cityscapes --out cityscapes.zarr
    python scripts/convert.py --cfg configs/coco.yaml --raw /data/coco --out coco.zarr --format webdataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert a raw segmentation dataset to Zarr / WebDataset / npy."
    )
    p.add_argument("--cfg",    required=True,  help="Path to dataset YAML config")
    p.add_argument("--raw",    default=None,   help="Override paths.raw in config")
    p.add_argument("--out",    default=None,   help="Override paths.output in config")
    p.add_argument("--split",  default=None,   help="Single split to convert (default: all splits in config)")
    p.add_argument("--format", default=None,   help="Override storage.format in config (zarr|webdataset|npy)")
    p.add_argument("--audit",  action="store_true", help="Run lossless audit after conversion")
    p.add_argument("--n-audit-samples", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.exists():
        print(f"[error] Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # apply CLI overrides
    if args.raw:
        cfg["paths"]["raw"] = args.raw
    if args.out:
        cfg["paths"]["output"] = args.out
    if args.format:
        cfg.setdefault("storage", {})["format"] = args.format

    out_path = Path(cfg["paths"]["output"])
    splits   = [args.split] if args.split else cfg["dataset"]["splits"]

    from segdatakit.readers import get_reader
    from segdatakit.writers import get_writer

    print(f"Dataset  : {cfg['dataset']['name']}")
    print(f"Splits   : {splits}")
    print(f"Format   : {cfg.get('storage', {}).get('format', 'zarr')}")
    print(f"Output   : {out_path}")
    print()

    for split in splits:
        print(f"--- Converting split: {split} ---")
        reader = get_reader(cfg, split=split)
        print(f"  Found {len(reader)} samples")

        split_out = out_path if len(splits) == 1 else Path(str(out_path).replace(".zarr", f"_{split}.zarr"))

        with get_writer(cfg, split_out) as writer:
            writer.write(reader)

        if args.audit:
            import zarr
            from segdatakit.validators import audit_lossless

            print(f"\nRunning lossless audit on {split}...")
            store  = zarr.open(str(split_out), mode="r")
            report = audit_lossless(
                reader, store,
                n_samples=args.n_audit_samples,
                zarr_path=str(split_out),
            )
            report.print_summary()
            audit_out = split_out.parent / f"audit_{split}.json"
            report.save(audit_out)

            if not report.lossless:
                print(f"[FAIL] Lossless audit failed — see {audit_out}")
                sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
