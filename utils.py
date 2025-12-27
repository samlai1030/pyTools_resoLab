# -*- coding: utf-8 -*-
"""
Utility functions for image file I/O and preprocessing.
"""

import os
import numpy as np
from constants import MAX_FILE_SIZE_MB, MAX_IMAGE_DIMENSION, COMMON_RAW_SIZES


def read_raw_image(file_path, width=None, height=None, dtype=np.uint16):
    """
    Read a raw image file with validation and error checking.

    Parameters:
    - file_path: path to the raw file
    - width: image width (if known)
    - height: image height (if known)
    - dtype: data type (usually uint8, uint16, or float32)

    Returns:
    - numpy array: loaded image data or None on error
    """
    try:
        file_size = os.path.getsize(file_path)
        max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

        if file_size > max_size_bytes:
            raise MemoryError(
                f"File size ({file_size / (1024**2):.1f} MB) exceeds maximum ({MAX_FILE_SIZE_MB} MB)"
            )

        if width and height:
            if width < 1 or width > MAX_IMAGE_DIMENSION:
                raise ValueError(f"Width {width} out of range (1-{MAX_IMAGE_DIMENSION})")
            if height < 1 or height > MAX_IMAGE_DIMENSION:
                raise ValueError(f"Height {height} out of range (1-{MAX_IMAGE_DIMENSION})")

            expected_size = width * height * np.dtype(dtype).itemsize
            if file_size != expected_size:
                raise ValueError(
                    f"File size ({file_size}) doesn't match expected ({expected_size}) "
                    f"for {width}x{height} {dtype}"
                )

        with open(file_path, "rb") as f:
            raw_data = f.read()

        dtype_itemsize = np.dtype(dtype).itemsize
        if len(raw_data) % dtype_itemsize != 0:
            raise ValueError(f"File size not divisible by dtype size ({dtype_itemsize})")

        img_array = np.frombuffer(raw_data, dtype=dtype)

        if width and height:
            if img_array.size != width * height:
                raise ValueError(f"Array size doesn't match dimensions")
            img_array = img_array.reshape(height, width)
        else:
            total_pixels = len(img_array)
            side = int(np.sqrt(total_pixels))
            if side * side == total_pixels:
                img_array = img_array.reshape(side, side)
            else:
                print(f"Cannot determine dimensions. Total values: {total_pixels}")
                return img_array

        return img_array

    except Exception as e:
        print(f"Error reading raw file: {e}")
        return None


def remove_inactive_borders(image, threshold=0):
    """
    Remove black/inactive borders from image edges.

    Parameters:
    - image: numpy array (2D grayscale)
    - threshold: pixel value threshold (pixels <= threshold are inactive)

    Returns:
    - cropped image, crop_info dict
    """
    if image is None or image.size == 0:
        return image, None

    original_shape = image.shape
    row_mask = np.any(image > threshold, axis=1)
    col_mask = np.any(image > threshold, axis=0)

    active_rows = np.where(row_mask)[0]
    active_cols = np.where(col_mask)[0]

    if len(active_rows) == 0 or len(active_cols) == 0:
        return image, {
            "original_size": original_shape,
            "new_size": original_shape,
            "crop_top": 0, "crop_bottom": 0,
            "crop_left": 0, "crop_right": 0,
            "rows_removed": 0, "cols_removed": 0,
        }

    row_start, row_end = active_rows[0], active_rows[-1] + 1
    col_start, col_end = active_cols[0], active_cols[-1] + 1
    cropped = image[row_start:row_end, col_start:col_end]

    crop_info = {
        "original_size": original_shape,
        "new_size": cropped.shape,
        "crop_top": row_start,
        "crop_bottom": original_shape[0] - row_end,
        "crop_left": col_start,
        "crop_right": original_shape[1] - col_end,
        "rows_removed": original_shape[0] - cropped.shape[0],
        "cols_removed": original_shape[1] - cropped.shape[1],
    }

    return cropped, crop_info


def auto_detect_raw_dimensions(file_size, filename=None):
    """
    Auto-detect raw image dimensions based on file size and filename.

    Returns (width, height, dtype_name) tuple.
    """
    import re

    # Try to parse from filename
    if filename:
        basename = os.path.basename(filename)

        # Pattern: WxH
        match = re.search(r"(\d{3,5})[xX](\d{3,5})", basename)
        if match:
            w, h = int(match.group(1)), int(match.group(2))
            for bpp, dtype_name in [(1, "uint8"), (2, "uint16"), (4, "float32")]:
                if w * h * bpp == file_size:
                    return (w, h, dtype_name)

        # Pattern: W_H
        match = re.search(r"(\d{3,5})_(\d{3,5})", basename)
        if match:
            w, h = int(match.group(1)), int(match.group(2))
            for bpp, dtype_name in [(1, "uint8"), (2, "uint16"), (4, "float32")]:
                if w * h * bpp == file_size:
                    return (w, h, dtype_name)

    # Use common sizes from constants
    for w, h, bpp, dtype_name in COMMON_RAW_SIZES:
        if w * h * bpp == file_size:
            return (w, h, dtype_name)

    # Try square dimensions
    for bpp, dtype_name in [(1, "uint8"), (2, "uint16"), (4, "float32")]:
        pixels = file_size // bpp
        side = int(np.sqrt(pixels))
        if side * side * bpp == file_size:
            return (side, side, dtype_name)

    # Default fallback
    pixels_16bit = file_size // 2
    if pixels_16bit > 0:
        side = int(np.sqrt(pixels_16bit))
        return (side, side, "uint16")

    return (640, 640, "uint16")
