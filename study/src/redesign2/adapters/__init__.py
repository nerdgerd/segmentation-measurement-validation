"""Read-only inputs and write-free, audited geometric primitives."""

from .descriptors import describe
from .masks import MaskData, dice, load_annotation_mask

__all__ = ["MaskData", "dice", "describe", "load_annotation_mask"]
