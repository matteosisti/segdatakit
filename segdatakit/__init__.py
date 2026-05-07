"""
segdatakit — lossless preprocessing and efficient storage
for semantic segmentation datasets.

Public API
----------
from segdatakit import SegDataset, get_reader, get_writer
from segdatakit.validators import audit_lossless, check_dataset_sanity
from segdatakit.transforms import cityscapes_train_transform, cityscapes_val_transform
"""

from segdatakit.readers import get_reader, BaseReader, CityscapesReader, COCOReader
from segdatakit.writers import get_writer, ZarrWriter, WebDatasetWriter, NpyWriter
from segdatakit.dataloader import SegDataset, NpyDataset
from segdatakit.dali_loader import build_loader, build_dali_loader, DALI_AVAILABLE
from segdatakit.validators import (
    audit_lossless,
    check_dataset_sanity,
    check_aspect_ratio,
    check_ood_score_ratio,
)

__version__ = "0.1.0"
__all__ = [
    "SegDataset",
    "NpyDataset",
    "build_loader",
    "build_dali_loader",
    "DALI_AVAILABLE",
    "get_reader",
    "get_writer",
    "BaseReader",
    "CityscapesReader",
    "COCOReader",
    "ZarrWriter",
    "WebDatasetWriter",
    "NpyWriter",
    "audit_lossless",
    "check_dataset_sanity",
    "check_aspect_ratio",
    "check_ood_score_ratio",
]
