"""
segdatakit/transforms.py

Runtime transforms applied in the dataloader — never on disk.
All transforms operate on (image, mask) pairs and return the same types.

Design principle:
  Transforms are composable via Compose. Each transform is a callable
  that takes (image: np.ndarray, mask: np.ndarray) and returns the same.
  This keeps them compatible with both numpy and PyTorch workflows.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseTransform:
    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class Compose(BaseTransform):
    """Chain multiple transforms sequentially."""

    def __init__(self, transforms: list[BaseTransform]):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


# ---------------------------------------------------------------------------
# Spatial transforms
# ---------------------------------------------------------------------------

class Resize(BaseTransform):
    """
    Resize image and mask to (height, width).

    Uses BILINEAR for images and NEAREST for masks to avoid
    introducing interpolated class values in the segmentation mask.

    Note: this intentionally changes spatial dimensions. The lossless
    guarantee applies to the stored Zarr data, not to runtime transforms.
    If you need to verify aspect ratio distortion, use
    validators.check_aspect_ratio() before applying this transform.
    """

    def __init__(self, height: int, width: int):
        self.size = (width, height)  # PIL uses (W, H)

    def __call__(self, image, mask):
        image = np.array(
            Image.fromarray(image).resize(self.size, Image.BILINEAR)
        )
        mask = np.array(
            Image.fromarray(mask, mode="L").resize(self.size, Image.NEAREST)
        )
        return image, mask


class ResizeAspectPreserving(BaseTransform):
    """
    Resize so the longer side equals `max_size`, then pad to a square.
    Preserves aspect ratio — no distortion.
    Padding value for masks is ignore_index (default 255).
    """

    def __init__(self, max_size: int, ignore_index: int = 255):
        self.max_size     = max_size
        self.ignore_index = ignore_index

    def __call__(self, image, mask):
        h, w    = image.shape[:2]
        scale   = self.max_size / max(h, w)
        new_h   = int(round(h * scale))
        new_w   = int(round(w * scale))

        image = np.array(
            Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)
        )
        mask = np.array(
            Image.fromarray(mask, mode="L").resize((new_w, new_h), Image.NEAREST)
        )

        # pad to square
        pad_h = self.max_size - new_h
        pad_w = self.max_size - new_w
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
        mask  = np.pad(mask,  ((0, pad_h), (0, pad_w)),          constant_values=self.ignore_index)

        return image, mask


class RandomHorizontalFlip(BaseTransform):
    """Flip image and mask horizontally with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, mask):
        if np.random.random() < self.p:
            image = np.fliplr(image).copy()
            mask  = np.fliplr(mask).copy()
        return image, mask


class RandomCrop(BaseTransform):
    """Randomly crop image and mask to (height, width)."""

    def __init__(self, height: int, width: int):
        self.h = height
        self.w = width

    def __call__(self, image, mask):
        H, W = image.shape[:2]
        if H < self.h or W < self.w:
            raise ValueError(
                f"Crop size ({self.h}, {self.w}) larger than image ({H}, {W})"
            )
        top  = np.random.randint(0, H - self.h + 1)
        left = np.random.randint(0, W - self.w + 1)
        image = image[top:top + self.h, left:left + self.w]
        mask  = mask[top:top + self.h,  left:left + self.w]
        return image, mask


# ---------------------------------------------------------------------------
# Photometric transforms (image only — mask unchanged)
# ---------------------------------------------------------------------------

class Normalize(BaseTransform):
    """
    Normalise image to float32 using ImageNet mean/std by default.
    Returns image as float32, mask unchanged as uint8.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        mean: list[float] | None = None,
        std:  list[float] | None = None,
    ):
        self.mean = np.array(mean or self.IMAGENET_MEAN, dtype=np.float32)
        self.std  = np.array(std  or self.IMAGENET_STD,  dtype=np.float32)

    def __call__(self, image, mask):
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image, mask


class ToTensor(BaseTransform):
    """
    Convert (H, W, C) uint8/float32 image to (C, H, W) float32 torch.Tensor.
    Mask becomes a (H, W) int64 torch.Tensor.
    """

    def __call__(self, image, mask):
        import torch
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        # HWC → CHW
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        mask  = torch.from_numpy(mask.astype(np.int64))
        return image, mask


# ---------------------------------------------------------------------------
# Preset pipelines
# ---------------------------------------------------------------------------

def cityscapes_train_transform(size: int = 768) -> Compose:
    """Standard training pipeline for Cityscapes."""
    return Compose([
        ResizeAspectPreserving(max_size=size),
        RandomHorizontalFlip(p=0.5),
        RandomCrop(height=size, width=size),
        Normalize(),
        ToTensor(),
    ])


def cityscapes_val_transform(size: int = 768) -> Compose:
    """Standard validation pipeline — no augmentation."""
    return Compose([
        ResizeAspectPreserving(max_size=size),
        Normalize(),
        ToTensor(),
    ])
