# -*- coding: utf-8 -*-
"""
Constants used across the pyTools_ResoLab application.
"""

import numpy as np

# SFR/MTF Calculation Constants
SUPERSAMPLING_FACTOR = 4
EDGE_GRADIENT_THRESHOLD = 50  # Minimum gradient magnitude for edge detection

# File handling limits
MAX_FILE_SIZE_MB = 500  # Maximum raw file size in MB
MAX_IMAGE_DIMENSION = 16384  # Maximum image dimension for memory safety

# Analysis thresholds
WHITE_REGION_MIN_PERCENT = 0.01  # Minimum percentage of pixels for white region analysis
EPSILON = np.finfo(float).eps  # Machine epsilon for division by zero protection

# Common raw image format options: (width, height, bytes_per_pixel, dtype_name, display_name)
RAW_FORMAT_OPTIONS = [
    # Auto detect option
    (0, 0, 0, "auto", "Auto Detect"),
    # Sensor common sizes - 16-bit
    (4000, 3000, 2, "uint16", "4000×3000 16bit (12MP)"),
    (4032, 3024, 2, "uint16", "4032×3024 16bit (12MP iPhone)"),
    (4608, 3456, 2, "uint16", "4608×3456 16bit (16MP)"),
    (4624, 3472, 2, "uint16", "4624×3472 16bit (Sony IMX)"),
    (4656, 3496, 2, "uint16", "4656×3496 16bit (Sony IMX)"),
    (5184, 3888, 2, "uint16", "5184×3888 16bit (20MP)"),
    (5472, 3648, 2, "uint16", "5472×3648 16bit (20MP)"),
    (6000, 4000, 2, "uint16", "6000×4000 16bit (24MP)"),
    (6016, 4016, 2, "uint16", "6016×4016 16bit (Sony A7)"),
    (6048, 4024, 2, "uint16", "6048×4024 16bit (Sony)"),
    (7952, 5304, 2, "uint16", "7952×5304 16bit (Canon 5D)"),
    (8192, 5464, 2, "uint16", "8192×5464 16bit (Canon R5)"),
    (8256, 5504, 2, "uint16", "8256×5504 16bit (45MP)"),
    (9504, 6336, 2, "uint16", "9504×6336 16bit (60MP)"),
    # Video/display resolutions - 16-bit
    (640, 480, 2, "uint16", "640×480 16bit (VGA)"),
    (640, 640, 2, "uint16", "640×640 16bit (Square)"),
    (800, 600, 2, "uint16", "800×600 16bit (SVGA)"),
    (1024, 768, 2, "uint16", "1024×768 16bit (XGA)"),
    (1280, 720, 2, "uint16", "1280×720 16bit (HD 720p)"),
    (1920, 1080, 2, "uint16", "1920×1080 16bit (Full HD)"),
    (2048, 1536, 2, "uint16", "2048×1536 16bit (3MP)"),
    (2560, 1440, 2, "uint16", "2560×1440 16bit (QHD)"),
    (2592, 1944, 2, "uint16", "2592×1944 16bit (5MP)"),
    (3264, 2448, 2, "uint16", "3264×2448 16bit (8MP)"),
    (3840, 2160, 2, "uint16", "3840×2160 16bit (4K UHD)"),
    (4096, 2160, 2, "uint16", "4096×2160 16bit (4K DCI)"),
    # Square sizes - 16-bit
    (256, 256, 2, "uint16", "256×256 16bit"),
    (512, 512, 2, "uint16", "512×512 16bit"),
    (1024, 1024, 2, "uint16", "1024×1024 16bit"),
    (2048, 2048, 2, "uint16", "2048×2048 16bit"),
    (4096, 4096, 2, "uint16", "4096×4096 16bit"),
    # 8-bit versions
    (640, 480, 1, "uint8", "640×480 8bit (VGA)"),
    (640, 640, 1, "uint8", "640×640 8bit (Square)"),
    (640, 1920, 1, "uint8", "640×1920 8bit"),
    (640, 641, 1, "uint8", "640×641 8bit"),
    (800, 600, 1, "uint8", "800×600 8bit (SVGA)"),
    (1024, 768, 1, "uint8", "1024×768 8bit (XGA)"),
    (1280, 720, 1, "uint8", "1280×720 8bit (HD 720p)"),
    (1920, 1080, 1, "uint8", "1920×1080 8bit (Full HD)"),
    (2048, 1536, 1, "uint8", "2048×1536 8bit (3MP)"),
    (2592, 1944, 1, "uint8", "2592×1944 8bit (5MP)"),
    (3264, 2448, 1, "uint8", "3264×2448 8bit (8MP)"),
    (4032, 3024, 1, "uint8", "4032×3024 8bit (12MP)"),
    (4096, 2160, 1, "uint8", "4096×2160 8bit (4K)"),
    (256, 256, 1, "uint8", "256×256 8bit"),
    (512, 512, 1, "uint8", "512×512 8bit"),
    (1024, 1024, 1, "uint8", "1024×1024 8bit"),
    (2048, 2048, 1, "uint8", "2048×2048 8bit"),
    (4096, 4096, 1, "uint8", "4096×4096 8bit"),
]

# Common raw image sizes for auto-detection: (width, height, bytes_per_pixel, dtype_name)
COMMON_RAW_SIZES = [
    # 8-bit versions first (more common for processed images)
    (640, 480, 1, "uint8"),  # VGA
    (640, 640, 1, "uint8"),  # Square
    (640, 641, 1, "uint8"),
    (800, 600, 1, "uint8"),  # SVGA
    (1024, 768, 1, "uint8"),  # XGA
    (1280, 720, 1, "uint8"),  # HD 720p
    (1920, 1080, 1, "uint8"),  # Full HD
    (2048, 1536, 1, "uint8"),  # 3MP
    (2592, 1944, 1, "uint8"),  # 5MP
    (3264, 2448, 1, "uint8"),  # 8MP
    (4032, 3024, 1, "uint8"),  # 12MP
    (4096, 2160, 1, "uint8"),  # 4K
    (256, 256, 1, "uint8"),  # Square test
    (640, 1920, 1, "uint8"),  # Innorev Tester
    (512, 512, 1, "uint8"),
    (1024, 1024, 1, "uint8"),
    (2048, 2048, 1, "uint8"),
    (4096, 4096, 1, "uint8"),
    # Sensor common sizes - 16-bit (raw sensor data)
    (4000, 3000, 2, "uint16"),  # 12MP sensor
    (4032, 3024, 2, "uint16"),  # 12MP iPhone
    (4608, 3456, 2, "uint16"),  # 16MP
    (4624, 3472, 2, "uint16"),  # Sony IMX
    (4656, 3496, 2, "uint16"),  # Sony IMX
    (5184, 3888, 2, "uint16"),  # 20MP
    (5472, 3648, 2, "uint16"),  # 20MP
    (6000, 4000, 2, "uint16"),  # 24MP
    (6016, 4016, 2, "uint16"),  # Sony A7
    (6048, 4024, 2, "uint16"),  # Sony
    (8256, 5504, 2, "uint16"),  # 45MP
    (8192, 5464, 2, "uint16"),  # Canon R5
    (7952, 5304, 2, "uint16"),  # Canon 5D
    (9504, 6336, 2, "uint16"),  # 60MP
    # Video/display resolutions - 16-bit
    (640, 480, 2, "uint16"),
    (640, 640, 2, "uint16"),
    (800, 600, 2, "uint16"),
    (1024, 768, 2, "uint16"),
    (1280, 720, 2, "uint16"),  # HD 720p
    (1920, 1080, 2, "uint16"),  # Full HD
    (2048, 1536, 2, "uint16"),  # 3MP
    (2560, 1440, 2, "uint16"),  # QHD
    (2592, 1944, 2, "uint16"),  # 5MP
    (3264, 2448, 2, "uint16"),  # 8MP
    (3840, 2160, 2, "uint16"),  # 4K UHD
    (4096, 2160, 2, "uint16"),  # 4K DCI
    # Square sizes - 16-bit (common for test patterns)
    (256, 256, 2, "uint16"),
    (512, 512, 2, "uint16"),
    (640, 640, 2, "uint16"),
    (1024, 1024, 2, "uint16"),
    (2048, 2048, 2, "uint16"),
    (4096, 4096, 2, "uint16"),
]
