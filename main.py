import os
import sys
import json

import cv2
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy import fftpack
from scipy.ndimage import gaussian_filter, gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, filtfilt, medfilt, savgol_filter, wiener

from mainUI import Ui_MainWindow


# Constants
SUPERSAMPLING_FACTOR = 4
EDGE_GRADIENT_THRESHOLD = 50  # Minimum gradient magnitude for edge detection
MAX_FILE_SIZE_MB = 500  # Maximum raw file size in MB
MAX_IMAGE_DIMENSION = 16384  # Maximum image dimension for memory safety
WHITE_REGION_MIN_PERCENT = (
    0.01  # Minimum percentage of pixels for white region analysis
)
EPSILON = np.finfo(float).eps  # Machine epsilon for division by zero protection


def read_raw_image(file_path, width=None, height=None, dtype=np.uint16):
    """
    Read a raw image file with validation and error checking

    Parameters:
    - file_path: path to the raw file
    - width: image width (if known)
    - height: image height (if known)
    - dtype: data type (usually uint8, uint16, or float32)

    Returns:
    - numpy array: loaded image data or None on error

    Raises:
    - ValueError: if file size doesn't match expected dimensions or dtype
    - IOError: if file cannot be read
    - MemoryError: if file is too large
    """
    try:
        # Validate file size before reading to prevent OOM
        file_size = os.path.getsize(file_path)
        max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

        if file_size > max_size_bytes:
            raise MemoryError(
                f"File size ({file_size / (1024**2):.1f} MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB} MB)"
            )

        # Validate dimensions if provided
        if width and height:
            if width < 1 or width > MAX_IMAGE_DIMENSION:
                raise ValueError(
                    f"Width {width} is out of valid range (1-{MAX_IMAGE_DIMENSION})"
                )
            if height < 1 or height > MAX_IMAGE_DIMENSION:
                raise ValueError(
                    f"Height {height} is out of valid range (1-{MAX_IMAGE_DIMENSION})"
                )

            # Calculate expected file size
            expected_size = width * height * np.dtype(dtype).itemsize
            if file_size != expected_size:
                raise ValueError(
                    f"File size ({file_size} bytes) doesn't match expected size "
                    f"({expected_size} bytes) for {width}x{height} image with dtype {dtype}"
                )

        # Read file data
        with open(file_path, "rb") as f:
            raw_data = f.read()

        # Validate data size is compatible with dtype
        dtype_itemsize = np.dtype(dtype).itemsize
        if len(raw_data) % dtype_itemsize != 0:
            raise ValueError(
                f"File size ({len(raw_data)} bytes) is not divisible by dtype size "
                f"({dtype_itemsize} bytes). Data may be corrupted."
            )

        # Convert to numpy array based on dtype
        if dtype == np.uint16:
            img_array = np.frombuffer(raw_data, dtype=np.uint16)
        elif dtype == np.uint8:
            img_array = np.frombuffer(raw_data, dtype=np.uint8)
        else:
            img_array = np.frombuffer(raw_data, dtype=dtype)

        # Reshape if dimensions are known
        if width and height:
            # Additional safety check before reshape
            if img_array.size != width * height:
                raise ValueError(
                    f"Array size ({img_array.size}) doesn't match dimensions "
                    f"({width}x{height}={width*height})"
                )
            img_array = img_array.reshape(height, width)
        else:
            # Try to guess square dimensions
            total_pixels = len(img_array)
            side = int(np.sqrt(total_pixels))
            if side * side == total_pixels:
                img_array = img_array.reshape(side, side)
            else:
                print(
                    f"Cannot determine image dimensions. Total values: {total_pixels}"
                )
                return img_array

        return img_array

    except (IOError, OSError) as e:
        print(f"Error reading file {file_path}: {e}")
        return None
    except MemoryError as e:
        print(f"Memory error: {e}")
        return None
    except ValueError as e:
        print(f"Validation error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error reading raw file: {e}")
        return None


class SFRCalculator:
    """物理運算核心：處理邊緣檢測與 SFR 計算"""

    @staticmethod
    def _apply_lsf_smoothing(lsf, method="wiener"):
        """
        應用選定的 LSF 平滑方法

        Parameters:
        - lsf: Line Spread Function 陣列
        - method: 平滑方法
          * "savgol": Savitzky-Golay filter (推薦)
          * "gaussian": 高斯濾波
          * "median": 中值濾波
          * "uniform": 均勻濾波
          * "butterworth": Butterworth IIR 濾波
          * "wiener": Wiener 自適應濾波
          * "none": 不平滑

        Returns:
        - 平滑後的 LSF 陣列
        """
        if method == "none" or len(lsf) <= 5:
            return lsf

        try:
            if method == "savgol":

                # Savitzky-Golay: 保留峰值特性的多項式平滑
                from scipy.signal import savgol_filter

                window_length = min(11, len(lsf) if len(lsf) % 2 == 1 else len(lsf) - 1)
                if window_length < 5:
                    window_length = 5
                return savgol_filter(lsf, window_length=window_length, polyorder=3)

            elif method == "gaussian":
                # 高斯濾波: 簡單平滑
                from scipy.ndimage import gaussian_filter1d

                return gaussian_filter1d(lsf, sigma=1.5)

            elif method == "median":
                # 中值濾波: 對異常值魯棒
                from scipy.signal import medfilt

                kernel_size = min(11, len(lsf) if len(lsf) % 2 == 1 else len(lsf) - 1)
                if kernel_size < 5:
                    kernel_size = 5
                return medfilt(lsf, kernel_size=kernel_size)

            elif method == "uniform":
                # 均勻濾波: 最快的平滑
                from scipy.ndimage import uniform_filter1d

                return uniform_filter1d(lsf, size=5)

            elif method == "butterworth":
                # Butterworth IIR 濾波: 頻率域控制
                try:
                    b, a = butter(2, 0.1)
                    return filtfilt(b, a, lsf)
                except (ValueError, RuntimeError) as e:

                    # Savitzky-Golay: 保留峰值特性的多項式平滑
                    from scipy.signal import savgol_filter

                    # 如果失敗，回退到 Savitzky-Golay
                    print(
                        f"Warning: Butterworth filter failed: {e}, falling back to Savitzky-Golay"
                    )
                    window_length = min(
                        11, len(lsf) if len(lsf) % 2 == 1 else len(lsf) - 1
                    )
                    if window_length < 5:
                        window_length = 5
                    return savgol_filter(lsf, window_length=window_length, polyorder=3)

            elif method == "wiener":
                # Wiener 自適應濾波: 噪聲自適應
                from scipy.signal import wiener

                mysize = min(11, len(lsf) if len(lsf) % 2 == 1 else len(lsf) - 1)
                if mysize < 5:
                    mysize = 5
                return wiener(lsf, mysize=mysize)

            else:
                # 未知方法，使用預設 Savitzky-Golay
                from scipy.signal import savgol_filter

                window_length = min(11, len(lsf) if len(lsf) % 2 == 1 else len(lsf) - 1)
                if window_length < 5:
                    window_length = 5
                return savgol_filter(lsf, window_length=window_length, polyorder=3)

        except Exception as e:
            # 如果任何濾波失敗，返回原始 LSF
            print(f"Warning: LSF smoothing method '{method}' failed: {e}")
            return lsf

    @staticmethod
    def detect_edge_orientation(roi_image):
        """
        檢測邊緣方向：垂直邊(V-edge) 或 水平邊(H-edge)。

        Returns:
        - edge_type: "V-Edge" (垂直), "H-Edge" (水平), or "Mixed"
        - confidence: 0-100, 邊緣方向的置信度
        - details: 詳細信息字典
        """
        if roi_image is None or roi_image.size == 0:
            return "No Edge", 0, {}

        # 轉為灰阶
        gray = (
            roi_image
            if len(roi_image.shape) == 2
            else cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        )
        gray = gray.astype(np.float64)

        # 使用 Sobel 算子計算梯度
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # 垂直邊 (x方向梯度)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # 水平邊 (y方向梯度)

        # 計算梯度的強度
        mag_x = np.sum(np.abs(sobelx))
        mag_y = np.sum(np.abs(sobely))

        # 計算梯度方向直方圖
        angle = np.arctan2(sobely, sobelx) * 180 / np.pi
        angle = np.mod(angle + 180, 180)  # 將角度標準化到 0-180

        # 統計邊緣方向
        v_edges = (
            np.sum((angle > 80) & (angle < 100)) / angle.size * 100
        )  # 垂直邊 ~90度
        h_edges = (
            np.sum(((angle > 170) | (angle < 10))) / angle.size * 100
        )  # 水平邊 ~0或180度

        # 判定邊緣類型
        edge_strength_ratio = mag_x / (mag_y + 1e-10)

        details = {
            "mag_x": mag_x,
            "mag_y": mag_y,
            "ratio_x_y": edge_strength_ratio,
            "v_edges_percent": v_edges,
            "h_edges_percent": h_edges,
            "mean_x": np.mean(np.abs(sobelx)),
            "mean_y": np.mean(np.abs(sobely)),
        }

        # 根據比率判定邊緣類型
        if edge_strength_ratio > 1.5:
            # 垂直邊：x方向梯度強
            confidence = min(100, (edge_strength_ratio - 1.0) * 50)
            return "V-Edge", confidence, details
        elif edge_strength_ratio < 0.67:
            # 水平邊：y方向梯度強
            confidence = min(100, (1.0 / edge_strength_ratio - 1.0) * 50)
            return "H-Edge", confidence, details
        else:
            # 混合邊
            confidence = 50
            return "Mixed", confidence, details

    @staticmethod
    def validate_edge(roi_image, threshold=50):
        """
        檢測是否為有效的 Slit Edge。
        判斷依據：
        1. 圖像梯度是否足夠強 (有邊緣)。
        2. 邊緣是否接近直線。

        Parameters:
        - roi_image: ROI 圖像
        - threshold: 邊緣檢測閾值 (默認 50)
        """
        if roi_image is None or roi_image.size == 0:
            return False, "Empty ROI", "No Edge", 0

        # 使用 Sobel 算子計算梯度
        gray = (
            roi_image
            if len(roi_image.shape) == 2
            else cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        )
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)

        # 簡單判定：最大梯度強度需大於閾值
        if np.max(magnitude) < threshold:
            return False, "Low Contrast / No Edge detected", "No Edge", 0

        # 檢測邊緣方向
        edge_type, confidence, _ = SFRCalculator.detect_edge_orientation(roi_image)

        return True, "Edge Detected", edge_type, confidence

    @staticmethod
    def standardize_roi_orientation(roi_image):
        """
        Standardize ROI orientation for SFR calculation.
        Ensures the edge is vertical with:
        1. Dark side on left, bright side on right
        2. Dark side (black) width increases toward bottom (slant direction)

        This standardization improves SFR measurement consistency by:
        1. Rotating horizontal edges (H-Edge) to vertical (V-Edge)
        2. Flipping horizontally if left side is brighter than right side
        3. Flipping vertically to ensure dark side width is larger at bottom

        Parameters:
        - roi_image: Input ROI image (numpy array)

        Returns:
        - standardized_roi: ROI with standardized orientation
        - edge_type: Detected edge type after standardization ("V-Edge" or original)
        - confidence: Edge detection confidence
        """
        if roi_image is None or roi_image.size == 0:
            return roi_image, None, 0.0

        img = roi_image.copy()

        # Helper function to get grayscale for analysis
        def get_gray(image):
            if image.ndim == 3 and image.shape[2] >= 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.ndim == 3:
                return np.mean(image, axis=2).astype(image.dtype)
            else:
                return image

        gray_for_analysis = get_gray(img)

        # Step 1: Detect current edge orientation
        edge_type, confidence, details = SFRCalculator.detect_edge_orientation(gray_for_analysis)

        # Step 2: If horizontal edge, rotate 90 degrees counter-clockwise to make it vertical
        if edge_type == "H-Edge":
            # Rotate image 90 degrees counter-clockwise
            img = np.rot90(img, k=1)
            gray_for_analysis = get_gray(img)

            # Re-detect orientation after rotation (should now be V-Edge)
            edge_type, confidence, details = SFRCalculator.detect_edge_orientation(gray_for_analysis)

        # Step 3: Ensure dark side is on left, bright side is on right
        h, w = gray_for_analysis.shape[:2]
        left_half_width = max(1, w // 2)

        left_mean = np.mean(gray_for_analysis[:, :left_half_width])
        right_mean = np.mean(gray_for_analysis[:, left_half_width:])

        # If left side is brighter than right side, flip horizontally
        if left_mean > right_mean:
            img = np.fliplr(img)
            gray_for_analysis = get_gray(img)

        # Step 4: Ensure dark side (black) width is larger at bottom than top
        # This standardizes the slant direction of the edge
        # Compare the edge position (transition point) between top and bottom rows
        h, w = gray_for_analysis.shape[:2]

        # Get top and bottom portions of the image
        top_portion_height = max(1, h // 4)
        bottom_portion_height = max(1, h // 4)

        top_rows = gray_for_analysis[:top_portion_height, :]
        bottom_rows = gray_for_analysis[-bottom_portion_height:, :]

        # Find edge position (where intensity crosses 50%) for top and bottom
        # Average each portion vertically to get a 1D profile
        top_profile = np.mean(top_rows, axis=0)
        bottom_profile = np.mean(bottom_rows, axis=0)

        # Normalize profiles
        def find_edge_position(profile):
            p_min, p_max = np.min(profile), np.max(profile)
            if p_max - p_min < 1e-6:
                return len(profile) // 2
            p_norm = (profile - p_min) / (p_max - p_min)
            # Find the position closest to 0.5 (edge center)
            return np.argmin(np.abs(p_norm - 0.5))

        top_edge_pos = find_edge_position(top_profile)
        bottom_edge_pos = find_edge_position(bottom_profile)

        # Dark side is on left. If dark side width at bottom < dark side width at top,
        # the edge position at bottom is less than at top, meaning we need to flip vertically
        # We want: bottom_edge_pos > top_edge_pos (dark region wider at bottom)
        if bottom_edge_pos < top_edge_pos:
            img = np.flipud(img)
            gray_for_analysis = get_gray(img)

        # Final edge detection for confidence
        edge_type, confidence, details = SFRCalculator.detect_edge_orientation(gray_for_analysis)

        return img, edge_type, confidence

    @staticmethod
    def calculate_sfr(
        roi_image,
        edge_type="V-Edge",
        compensate_bias=True,
        compensate_noise=True,
        lsf_smoothing_method="savgol",
        supersampling_factor=4,
    ):
        """
        計算 SFR (Spatial Frequency Response) - ISO 12233:2023 Standard with Compensation & LSF Smoothing

        符合 ISO 12233:2023 標準的空間頻率響應測量方法
        支持垂直邊(V-Edge)和水平邊(H-Edge)。
        包括白區域偏差和噪聲補償、LSF 峰值平滑。

        Parameters:
        - roi_image: ROI 圖像
        - edge_type: "V-Edge" 或 "H-Edge"
        - compensate_bias: 補償白區域偏差/亮度偏移 (預設 True)
        - compensate_noise: 補償白區域噪聲 (預設 True)
        - lsf_smoothing_method: LSF 平滑方法 (預設 "savgol")
          * "savgol": Savitzky-Golay filter (推薦，保留峰值特性)
          * "gaussian": 高斯濾波
          * "median": 中值濾波 (對異常值魯棒)
          * "uniform": 均勻平滑濾波 (最快)
          * "butterworth": Butterworth IIR 濾波 (頻率域控制)
          * "wiener": Wiener 自適應濾波 (噪聲自適應)
          * "none": 不進行 LSF 平滑
        - supersampling_factor: 超採樣因子 (預設 4, 範圍 1-16)

        Returns:
        - frequencies: 頻率陣列 (cycles/pixel)
        - sfr: SFR 值 (歸一化到 DC = 1)
        - esf: Edge Spread Function
        - lsf: Line Spread Function
        """
        # 轉為灰阶並正規化到 0-1 範圍
        img = roi_image.astype(np.float64)
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)

        # 正規化到 0-1
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        # 步驟 0a: 白區域偏差補償 (White Area Bias Compensation)
        # 補償白區域的亮度偏移/偏差
        if compensate_bias:
            # 提取白區域（值 > 0.9）
            white_mask = img > 0.9
            if np.sum(white_mask) > 10:  # 確保有足夠的白區域樣本
                white_level = np.mean(img[white_mask])
                # 標準白級應為 1.0，計算偏差
                bias_correction = 1.0 - white_level
                # 應用偏差補償
                img = img + bias_correction
                # 確保仍在 [0, 1] 範圍內
                img = np.clip(img, 0, 1)

        # 步驟 0b: 白區域噪聲補償 (White Area Noise Compensation)
        # 通過分析白區域噪聲，應用自適應低通濾波減少噪聲
        if compensate_noise:
            # 提取白區域（值 > 0.85）用於噪聲估計
            white_mask_noise = img > 0.85
            if np.sum(white_mask_noise) > 20:
                # 計算白區域的標準差（噪聲幅度）
                white_noise_std = np.std(img[white_mask_noise])

                # 如果噪聲顯著，應用自適應濾波
                if white_noise_std > 0.01:  # 噪聲閾值
                    from scipy.ndimage import gaussian_filter

                    # 應用高斯濾波，標準差基於測定的噪聲
                    sigma = white_noise_std * 0.5  # 調整濾波強度
                    if edge_type == "V-Edge":
                        # 只在行方向濾波（垂直方向），保留邊緣清晰度
                        img = gaussian_filter(img, sigma=(sigma, 0))
                    else:  # H-Edge
                        # 只在列方向濾波（水平方向），保留邊緣清晰度
                        img = gaussian_filter(img, sigma=(0, sigma))

        # Step 1: 邊緣提取與超採樣 (ISO 12233:2023 Section 7.1)
        if edge_type == "V-Edge":
            # 垂直邊：對寬度方向進行分析
            esf_raw = np.mean(img, axis=0)
            # 使用立方插值進行超採樣
            from scipy.interpolate import interp1d

            x_orig = np.arange(len(esf_raw))
            f_cubic = interp1d(x_orig, esf_raw, kind="cubic", fill_value="extrapolate")
            x_new = np.linspace(0, len(esf_raw) - 1, (len(esf_raw) - 1) * supersampling_factor + 1)
            esf = f_cubic(x_new)
        else:  # H-Edge
            # 水平邊：對高度方向進行分析
            esf_raw = np.mean(img, axis=1)
            from scipy.interpolate import interp1d

            x_orig = np.arange(len(esf_raw))
            f_cubic = interp1d(x_orig, esf_raw, kind="cubic", fill_value="extrapolate")
            x_new = np.linspace(0, len(esf_raw) - 1, (len(esf_raw) - 1) * supersampling_factor + 1)
            esf = f_cubic(x_new)

        # Step 2: 亞像素邊緣位置檢測與對齐 (ISO 12233:2023 Section 7.2)
        # 找到 50% 點的位置（邊緣中心）
        esf_min = np.min(esf)
        esf_max = np.max(esf)
        esf_normalized = (esf - esf_min) / (esf_max - esf_min + 1e-10)

        # 找到最接近 50% 的位置
        idx_50 = np.argmin(np.abs(esf_normalized - 0.5))

        # 使用線性插值精細定位 50% 點
        if idx_50 > 0 and idx_50 < len(esf_normalized) - 1:
            # 線性插值找到精確的 50% 位置
            if esf_normalized[idx_50] < 0.5:
                slope = esf_normalized[idx_50 + 1] - esf_normalized[idx_50]
                if slope != 0:
                    frac = (0.5 - esf_normalized[idx_50]) / slope
                else:
                    frac = 0
            else:
                slope = esf_normalized[idx_50] - esf_normalized[idx_50 - 1]
                if slope != 0:
                    frac = (esf_normalized[idx_50] - 0.5) / slope
                else:
                    frac = 0
        else:
            frac = 0

        # 邊緣對齐：移動 ESF 使得 50% 點對齐到整數位置
        edge_pos = idx_50 + frac
        shift_amount = edge_pos - int(edge_pos)

        # 使用循環移位和插值進行邊緣對齐
        if abs(shift_amount) > 1e-6:
            from scipy.ndimage import shift as ndimage_shift

            esf = ndimage_shift(esf, -shift_amount, order=1, mode="nearest")

        # Step 3: 計算 LSF (Line Spread Function)
        # ISO 12233:2023 使用一階差分
        lsf = np.diff(esf)

        # Step 3a: LSF Peak Smoothing - 可選的濾波方法
        # 應用選定的平滑方法以改善 LSF 峰值平滑度
        lsf = SFRCalculator._apply_lsf_smoothing(lsf, method=lsf_smoothing_method)

        # Step 4: 應用視窗函數減少頻譜洩漏 (ISO 12233:2023 推薦 Hann 窗)
        window = np.hanning(len(lsf))
        lsf_windowed = lsf * window

        # Step 5: FFT 轉換進行頻譜分析
        # Validate LSF has sufficient data for FFT
        if len(lsf_windowed) < 4:
            print(
                f"Warning: LSF too short ({len(lsf_windowed)} samples) for FFT analysis"
            )
            return None, None, esf, lsf

        if np.sum(np.abs(lsf_windowed)) < EPSILON:
            print("Warning: LSF sum is too small for meaningful FFT")
            return None, None, esf, lsf

        # 使用超採樣因子補零以改善頻率解析度
        n_fft = len(lsf_windowed)
        n_fft_padded = n_fft * supersampling_factor
        fft_res = np.abs(fftpack.fft(lsf_windowed, n=n_fft_padded))

        # Step 6: 頻率軸計算與歸一化 (ISO 12233:2023)
        # 考慮超採樣因子，頻率軸需要相應調整
        # d = 1/supersampling_factor 是超採樣後的像素間距
        freqs = fftpack.fftfreq(n_fft_padded, d=1.0/supersampling_factor)

        # 只取正頻率部分
        n_half = len(freqs) // 2
        sfr = fft_res[:n_half]
        frequencies = freqs[:n_half]

        # 歸一化：將 DC 分量設為 1 (ISO 12233:2023 Section 7.4)
        dc_component = sfr[0]
        if dc_component > EPSILON:
            sfr = sfr / dc_component
        else:
            print(
                f"Warning: DC component too small ({dc_component}), using fallback normalization"
            )
            sfr = sfr / EPSILON

        # 限制 SFR 到合理範圍 [0, 1]
        sfr = np.clip(sfr, 0, 1)

        # Step 7: 返回結果
        # 轉換頻率回到原始像素空間（考慮超採樣）
        # Note: Frequency scaling is already handled in fftfreq with d=1/supersampling_factor
        # No additional division is needed here to avoid double compensation
        frequencies = frequencies / supersampling_factor

        # 限制頻率範圍到 Nyquist 頻率 (0.5 cycles/pixel)
        valid_idx = frequencies <= 0.5
        frequencies = frequencies[valid_idx]
        sfr = sfr[valid_idx]

        return frequencies, sfr, esf, lsf


class ImageLabel(QLabel):
    """自定義 QLabel 用於處理滑鼠選取 ROI"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selection_start = None
        self.selection_end = None
        self.is_selecting = False
        self.pixmap_original = None
        self.pixmap_scaled = None
        self.roi_callback = None  # Callback function for ROI selection
        self.zoom_level = 1.0  # Zoom factor
        self.scroll_area = None  # Reference to scroll area
        self.selection_mode = "drag"  # "drag" or "click"
        self.parent_window = parent  # Reference to main window for mode access
        self.selection_info_text = ""  # Display selection info
        self.image_w = 640  # Image width
        self.image_h = 640  # Image height
        self.selection_rect = None  # Store selected 40x40 rectangle
        self.roi_sfr_value = None  # Store SFR value to display at ROI corner
        self.roi_position = None  # Store ROI (x, y, w, h) for SFR display
        self.roi_markers = []  # List of ROI markers [(x, y, w, h, name), ...]
        self._updating_zoom = False  # Guard flag to prevent recursion
        # Panning support for VIEW mode
        self.is_panning = False
        self.pan_start_pos = None
        self.pan_scroll_start_h = 0
        self.pan_scroll_start_v = 0
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)

    def mousePressEvent(self, event):
        if not self.pixmap_original:
            return

        # Right-click: Start panning (works in any mode)
        if event.button() == Qt.RightButton:
            self.is_panning = True
            self.pan_start_pos = event.pos()
            if self.scroll_area:
                self.pan_scroll_start_h = self.scroll_area.horizontalScrollBar().value()
                self.pan_scroll_start_v = self.scroll_area.verticalScrollBar().value()
            self.setCursor(Qt.ClosedHandCursor)
            return

        # Check if VIEW mode (panning mode) is active for left-click
        if self.parent_window and hasattr(self.parent_window, "view_mode"):
            view_mode = self.parent_window.view_mode
        else:
            view_mode = "sfr"  # Default to SFR mode

        if view_mode == "view":
            # VIEW mode: Start panning with left-click too
            if event.button() == Qt.LeftButton:
                self.is_panning = True
                self.pan_start_pos = event.pos()
                if self.scroll_area:
                    self.pan_scroll_start_h = self.scroll_area.horizontalScrollBar().value()
                    self.pan_scroll_start_v = self.scroll_area.verticalScrollBar().value()
                self.setCursor(Qt.ClosedHandCursor)
            return

        # Check if ROI Manual Mode is active - places multiple ROI markers (position picking only, no SFR calc)
        if self.parent_window and hasattr(self.parent_window, "roi_manual_mode") and self.parent_window.roi_manual_mode:
            if event.button() == Qt.LeftButton:
                click_pos = event.pos()

                # Get size from parent window (same as click select size)
                size = 40
                if self.parent_window and hasattr(self.parent_window, "click_select_size"):
                    size = self.parent_window.click_select_size

                half_size = size // 2

                # Calculate region centered at click point
                center_x = int(click_pos.x() / self.zoom_level)
                center_y = int(click_pos.y() / self.zoom_level)

                x = max(0, center_x - half_size)
                y = max(0, center_y - half_size)
                w = size
                h = size

                # Ensure within image bounds
                if x + w > self.image_w:
                    x = max(0, self.image_w - size)
                if y + h > self.image_h:
                    y = max(0, self.image_h - size)

                # ROI Plan Mode: Just store position, no SFR calculation
                # SFR will be calculated when user applies the saved config
                roi_num = len(self.roi_markers) + 1
                roi_name = f"ROI_{roi_num}"
                # Store with None for sfr_value - will be calculated on apply
                self.roi_markers.append((x, y, w, h, roi_name, None))

                # Also update parent's roi_markers list
                if self.parent_window and hasattr(self.parent_window, "roi_markers"):
                    self.parent_window.roi_markers.append((x, y, w, h, roi_name, None))
                    # Enable ROI Save button since we now have markers
                    if hasattr(self.parent_window, "ui"):
                        self.parent_window.ui.btn_roi_map_save.setEnabled(True)

                # Update display
                self.update()

                # Show info - position only, no SFR in plan mode
                if self.parent_window:
                    total = len(self.roi_markers)
                    self.parent_window.statusBar().showMessage(
                        f"📍 {roi_name} placed at ({x},{y}) {w}×{h} | Total: {total} ROI(s) - Click to add more"
                    )
            return

        # SFR mode: Get current selection mode from parent window
        if self.parent_window and hasattr(self.parent_window, "selection_mode"):
            current_mode = self.parent_window.selection_mode
        else:
            current_mode = "drag"

        if current_mode == "click":
            # Mode 2: Click with user-defined size - Single click to select area centered at click point
            if event.button() == Qt.LeftButton:
                # Clear any old drag selection when new click happens
                self.is_selecting = False
                self.selection_start = None
                self.selection_end = None

                click_pos = event.pos()

                # Get size from parent window (default 30)
                size = 30
                if self.parent_window and hasattr(
                    self.parent_window, "click_select_size"
                ):
                    size = self.parent_window.click_select_size

                half_size = size // 2  # Half size for centering

                # Calculate region centered at click point
                center_x = int(click_pos.x() / self.zoom_level)
                center_y = int(click_pos.y() / self.zoom_level)

                # Area centered at click: half_size pixels on each side
                x = max(0, center_x - half_size)
                y = max(0, center_y - half_size)
                w = size
                h = size

                # Ensure within image bounds
                if x + w > self.image_w:
                    x = max(0, self.image_w - size)
                if y + h > self.image_h:
                    y = max(0, self.image_h - size)

                # Store selection rectangle for drawing
                self.selection_rect = QRect(x, y, w, h)

                # Update selection info
                self.selection_info_text = f"Selected Area: {w}×{h} at ({x}, {y})"
                self.update()

                # Create rectangle and call callback
                rect = QRect(x, y, w, h)
                if self.roi_callback:
                    self.roi_callback(rect)
        else:
            # Mode 1: Drag Select (original behavior)
            if event.button() == Qt.LeftButton and self.pixmap_original:
                # Clear old click selection when new drag starts
                self.selection_rect = None

                self.selection_start = event.pos()
                self.selection_end = event.pos()
                self.is_selecting = True
                self.update()

    def mouseMoveEvent(self, event):
        # Check if panning (right-click drag or VIEW mode)
        if self.is_panning and self.scroll_area and self.pan_start_pos:
            delta = event.pos() - self.pan_start_pos
            self.scroll_area.horizontalScrollBar().setValue(
                self.pan_scroll_start_h - delta.x()
            )
            self.scroll_area.verticalScrollBar().setValue(
                self.pan_scroll_start_v - delta.y()
            )
            return

        if self.is_selecting and self.pixmap_original:
            self.selection_end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        # Check if releasing from panning (right-click or left-click in VIEW mode)
        if (event.button() == Qt.RightButton or event.button() == Qt.LeftButton) and self.is_panning:
            self.is_panning = False
            self.pan_start_pos = None
            # Restore cursor based on view mode
            if self.parent_window and hasattr(self.parent_window, "view_mode"):
                if self.parent_window.view_mode == "view":
                    self.setCursor(Qt.OpenHandCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            return

        if event.button() == Qt.LeftButton and self.is_selecting:
            self.is_selecting = False
            if self.roi_callback:
                self.roi_callback(self.get_roi_rect())
            self.update()

    def paintEvent(self, event):
        """Override paintEvent to draw selection square and crosshair"""
        super().paintEvent(event)

        # Create single painter for all drawing operations
        painter = QPainter(self)

        # Draw drag selection rectangle
        if self.is_selecting and self.selection_start and self.selection_end:
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))

            # Draw rectangle directly at current mouse positions (already in zoomed space)
            rect = QRect(self.selection_start, self.selection_end).normalized()
            painter.drawRect(rect)

        # Draw click selection square
        if self.selection_rect and self.pixmap_scaled:
            # Scale rectangle to current zoom level
            scaled_rect = QRect(
                int(self.selection_rect.x() * self.zoom_level),
                int(self.selection_rect.y() * self.zoom_level),
                int(self.selection_rect.width() * self.zoom_level),
                int(self.selection_rect.height() * self.zoom_level),
            )

            # Draw red rectangle outline
            pen = QPen(QColor(255, 0, 0))  # Red
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(scaled_rect)

            # Draw corner markers
            corner_size = 5
            for corner in [
                (scaled_rect.left(), scaled_rect.top()),
                (scaled_rect.right(), scaled_rect.top()),
                (scaled_rect.left(), scaled_rect.bottom()),
                (scaled_rect.right(), scaled_rect.bottom()),
            ]:
                painter.drawEllipse(
                    corner[0] - corner_size,
                    corner[1] - corner_size,
                    corner_size * 2,
                    corner_size * 2,
                )

        # Draw green crosshair at image center
        if self.pixmap_scaled:
            # Calculate center point in display coordinates
            center_x = self.pixmap_scaled.width() // 2
            center_y = self.pixmap_scaled.height() // 2

            # Draw crosshair
            pen = QPen(QColor(0, 255, 0))  # Green
            pen.setWidth(1)
            painter.setPen(pen)

            # Vertical line
            painter.drawLine(center_x, 0, center_x, self.pixmap_scaled.height())

            # Horizontal line
            painter.drawLine(0, center_y, self.pixmap_scaled.width(), center_y)

        # Draw SFR value at top-left corner of ROI
        if self.roi_sfr_value is not None and self.roi_position is not None:
            x, y, w, h = self.roi_position

            # Convert to display coordinates with zoom
            roi_top_left_x = int(x * self.zoom_level)
            roi_top_left_y = int((y + 5)  * self.zoom_level)

            # Prepare SFR text (show as percentage with 2 decimal places)
            sfr_text = f"{self.roi_sfr_value*100:.2f}%"

            # Setup font for text
            from PyQt5.QtGui import QFont
            font = QFont("Arial", 12)
            font.setBold(True)
            painter.setFont(font)

            # Calculate text size
            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(sfr_text)
            text_height = metrics.height()

            # Position text at top-left corner (slightly offset inside)
            text_x = roi_top_left_x + 5
            text_y = roi_top_left_y + text_height + 5

            # Draw semi-transparent background box
            bg_rect = QRect(text_x - 3, text_y - text_height - 2,
                           text_width + 6, text_height + 4)

            # Draw background (semi-transparent black)
            painter.fillRect(bg_rect, QColor(0, 0, 0, 200))

            # Draw border (white)
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawRect(bg_rect)

            # Draw text (white)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(text_x, text_y - 3, sfr_text)

        # Draw ROI markers (for ROI Manual mode and loaded ROIs)
        if self.roi_markers:
            from PyQt5.QtGui import QFont
            font = QFont("Arial", 10)
            font.setBold(True)
            painter.setFont(font)

            for marker in self.roi_markers:
                # Handle both 5-tuple (x, y, w, h, name) and 6-tuple (x, y, w, h, name, sfr_value)
                if len(marker) >= 6:
                    rx, ry, rw, rh, roi_name, sfr_value = marker[:6]
                else:
                    rx, ry, rw, rh, roi_name = marker[:5]
                    sfr_value = None

                # Convert to display coordinates with zoom
                disp_x = int(rx * self.zoom_level)
                disp_y = int(ry * self.zoom_level)
                disp_w = int(rw * self.zoom_level)
                disp_h = int(rh * self.zoom_level)

                # Draw ROI rectangle (red color)
                pen = QPen(QColor(255, 0, 0))  # Red
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawRect(disp_x, disp_y, disp_w, disp_h)

                # Prepare label text (name + SFR value if available)
                if sfr_value is not None:
                    label_text = f"{roi_name}: {sfr_value*100:.1f}%"
                else:
                    label_text = roi_name

                # Draw label on top of the rectangle
                metrics = painter.fontMetrics()
                text_width = metrics.horizontalAdvance(label_text)
                text_height = metrics.height()

                # Position text at top of ROI rectangle (inside, at top-left)
                text_x = disp_x + 3
                text_y = disp_y + text_height + 2

                # Draw semi-transparent background for label
                bg_rect = QRect(text_x - 2, disp_y + 2,
                               text_width + 4, text_height + 2)
                painter.fillRect(bg_rect, QColor(255, 0, 0, 200))  # Red background

                # Draw text (white)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(text_x, text_y, label_text)

        # End painter properly
        painter.end()

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if self.pixmap_original is None:
            return

        # Get scroll direction
        delta = event.angleDelta().y()

        # Zoom in/out based on scroll direction
        if delta > 0:
            # Scroll up - zoom in
            self.zoom_level *= 1.1
        else:
            # Scroll down - zoom out
            self.zoom_level /= 1.1

        # Limit zoom level
        self.zoom_level = max(0.5, min(self.zoom_level, 5.0))

        # Update display
        self.update_zoomed_image()

    def update_zoomed_image(self):
        """Update the displayed image with current zoom level"""
        if self.pixmap_original is None or self._updating_zoom:
            return

        # Set guard flag to prevent recursion
        self._updating_zoom = True

        try:
            # Calculate new size
            new_width = int(self.pixmap_original.width() * self.zoom_level)
            new_height = int(self.pixmap_original.height() * self.zoom_level)

            # Scale the pixmap maintaining aspect ratio
            self.pixmap_scaled = self.pixmap_original.scaled(
                new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(self.pixmap_scaled)

            # Update size to enable/show scrollbars
            self.setFixedSize(new_width, new_height)
        finally:
            # Always clear guard flag
            self._updating_zoom = False

    def set_roi_sfr_display(self, sfr_value, roi_x, roi_y, roi_w, roi_h):
        """Set SFR value and ROI position to display at top-right corner"""
        self.roi_sfr_value = sfr_value
        self.roi_position = (roi_x, roi_y, roi_w, roi_h)
        self.update()  # Trigger repaint

    def get_roi_rect(self):
        """Get ROI rectangle in image coordinates"""
        if not self.selection_start or not self.selection_end:
            return QRect()

        # Get selected rectangle in display coordinates
        rect = QRect(self.selection_start, self.selection_end).normalized()

        # Account for scroll position if zoomed
        if self.scroll_area and self.zoom_level != 1.0:
            scroll_x = self.scroll_area.horizontalScrollBar().value()
            scroll_y = self.scroll_area.verticalScrollBar().value()
            rect.translate(scroll_x, scroll_y)

        # Convert back to original image coordinates
        if self.zoom_level != 1.0:
            rect = QRect(
                int(rect.x() / self.zoom_level),
                int(rect.y() / self.zoom_level),
                int(rect.width() / self.zoom_level),
                int(rect.height() / self.zoom_level),
            )
        return rect


class MainWindow(QMainWindow):
    RECENT_FILES_PATH = "recent_files.json"

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


        # Data
        self.raw_data = None
        self.display_data = None  # Store display data for edge overlay
        self.current_image_path = None  # Track current loaded image file path
        self.current_roi_data = None  # Store current ROI selection data for saving
        self.image_w = 640  # 預設，實際應由使用者輸入
        self.image_h = 640

        # LSF Smoothing method selection
        self.lsf_smoothing_method = "none"  # Default method

        # Selection mode: "drag" or "click"
        self.selection_mode = "drag"  # Default: drag to select

        # Click select size (default 40x40)
        self.click_select_size = 40

        # SFR stabilize filter enable/disable
        self.sfr_stabilize_enabled = False  # Default: disabled

        # Supersampling factor for SFR calculation (1-16x, default 4x)
        self.supersampling_factor = 4  # Default: 4x

        # View mode: "sfr" for SFR analysis or "view" for panning
        self.view_mode = "sfr"  # Default: SFR mode

        # ROI plan mode: when enabled, SFR calculation is stopped (only ROI selection)
        self.roi_plan_mode = False  # Default: disabled
        self.roi_manual_mode = False  # ROI Manual mode for placing multiple ROIs
        self.roi_markers = []  # List of ROI markers [(x, y, w, h, name), ...]

        # Edge detection threshold (adjustable via slider)
        self.edge_threshold = 50  # Default: 50

        # Edge detection display mode
        self.edge_detect_enabled = False  # Show edge overlay on image
        self.edge_overlay_applied = False  # Track if edge overlay is applied
        self.locked_edge_mask = None  # Store locked edge pattern (frozen when Apply Edge clicked)

        # Nyquist frequency (0.1 to 1.0)
        self.ny_frequency = 0.5

        # Store last SFR data for re-plotting when Ny changes
        self.last_sfr_data = None

        # Recent files list (max 10 files)
        self.recent_files = []
        self.max_recent_files = 10

        # Recent ROI files list (max 10 ROI files)
        self.recent_roi_files = []
        self.max_recent_roi_files = 10
        self.RECENT_ROI_FILES_PATH = "recent_roi_files.json"

        self.load_recent_files()
        self.load_recent_roi_files()
        self.init_ui_connections()
        self.init_plots()
        self.update_recent_files_list()

    def init_ui_connections(self):
        self.ui.btn_load.clicked.connect(self.load_raw_file)
        self.ui.recent_files_combo.activated.connect(self.on_recent_file_selected)
        self.ui.radio_drag.toggled.connect(self.on_selection_mode_changed)
        self.ui.radio_click.toggled.connect(self.on_selection_mode_changed)
        self.ui.radio_script_roi.toggled.connect(self.on_selection_mode_changed)
        self.ui.click_size_input.valueChanged.connect(self.on_click_size_changed)
        self.ui.recent_roi_combo.activated.connect(self.on_recent_roi_selected)
        self.ui.btn_sfr_mode.clicked.connect(self.on_sfr_mode_clicked)
        self.ui.btn_view_mode.clicked.connect(self.on_view_mode_clicked)
        self.ui.method_combo.currentTextChanged.connect(self.on_smoothing_method_changed)
        self.ui.stabilize_checkbox.stateChanged.connect(self.on_stabilize_filter_changed)
        self.ui.supersampling_spinbox.valueChanged.connect(self.on_supersampling_changed)
        self.ui.edge_detect_checkbox.stateChanged.connect(self.on_edge_detect_changed)
        self.ui.edge_threshold_slider.valueChanged.connect(self.on_edge_threshold_changed)
        self.ui.btn_apply_edge.clicked.connect(self.on_apply_edge)
        self.ui.btn_erase_edge.clicked.connect(self.on_erase_edge)
        self.ui.ny_freq_slider.valueChanged.connect(self.on_ny_freq_slider_changed)

        # ROI Setting connections
        self.ui.checkBox.stateChanged.connect(self.on_roi_plan_mode_changed)
        self.ui.btn_roi_detact.clicked.connect(self.on_roi_detect)
        self.ui.btn_roi_manual.clicked.connect(self.on_roi_manual)
        self.ui.btn_roi_map_load.clicked.connect(self.on_roi_map_load)
        self.ui.btn_roi_map_apply.clicked.connect(self.on_roi_map_apply)
        self.ui.btn_roi_map_save.clicked.connect(self.on_roi_save)

        # Store current Ny frequency value (0.1 to 1.0, slider is 10-100)
        self.ny_frequency = 0.5

        self.image_label = ImageLabel(self)
        self.ui.scroll_area.setWidget(self.image_label)
        self.image_label.roi_callback = self.process_roi
        self.image_label.scroll_area = self.ui.scroll_area

        # Initialize recent ROI combo
        self.update_recent_roi_list()

        # Initially disable ROI Detect and ROI Manual buttons
        self.ui.btn_roi_detact.setEnabled(False)
        self.ui.btn_roi_manual.setEnabled(False)
        # Initially disable ROI Save button (enable when ROI markers exist)
        self.ui.btn_roi_map_save.setEnabled(False)
        # Initially disable recent ROI combo (only active when ROI map radio is selected)
        self.ui.recent_roi_combo.setEnabled(False)
        # Initially disable ROI Apply button (enable when .roi file is loaded)
        self.ui.btn_roi_map_apply.setEnabled(False)

        # Track loaded ROI file path
        self.loaded_roi_file_path = None


    def init_plots(self):
        self.figure = Figure(dpi=100)
        self.figure.patch.set_facecolor("white")
        self.canvas = FigureCanvas(self.figure)

        # Set size policy to allow the canvas to shrink/expand properly
        from PyQt5.QtWidgets import QSizePolicy
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        # Create a layout for the placeholder and add the canvas
        canvas_layout = QVBoxLayout(self.ui.canvas_placeholder)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self.canvas)

        # Set the placeholder to expand properly with max width
        self.ui.canvas_placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


        # 2x2 subplot layout:
        # (0,0) = SFR/MTF  |  (0,1) = ROI Image
        # (1,0) = ESF      |  (1,1) = LSF
        self.ax_sfr = self.figure.add_subplot(2, 2, 1)  # Top-left
        self.ax_roi = self.figure.add_subplot(2, 2, 2)  # Top-right (ROI image)
        self.ax_esf = self.figure.add_subplot(2, 2, 3)  # Bottom-left
        self.ax_lsf = self.figure.add_subplot(2, 2, 4)  # Bottom-right

        self.ax_sfr.set_title("SFR / MTF Result", fontsize=11, fontweight="bold")
        self.ax_sfr.set_xlabel("Frequency (cycles/pixel)", fontsize=10)
        self.ax_sfr.set_ylabel("MTF", fontsize=10)
        self.ax_sfr.grid(True, alpha=0.3)

        self.ax_roi.set_title("ROI Image", fontsize=10, fontweight="bold")
        self.ax_roi.axis('off')  # Hide axes for image display

        self.ax_esf.set_title(
            "ESF (Edge Spread Function)", fontsize=10, fontweight="bold"
        )
        self.ax_esf.set_xlabel("Position (pixels)", fontsize=9)
        self.ax_esf.set_ylabel("Intensity", fontsize=9)
        self.ax_esf.grid(True, alpha=0.3)

        self.ax_lsf.set_title(
            "LSF (Line Spread Function)", fontsize=10, fontweight="bold"
        )
        self.ax_lsf.set_xlabel("Position (pixels)", fontsize=9)
        self.ax_lsf.set_ylabel("Derivative", fontsize=9)
        self.ax_lsf.grid(True, alpha=0.3)

        self.figure.tight_layout()

        # Connect resize event to update figure layout
        self.canvas.mpl_connect('resize_event', self.on_canvas_resize)

    def on_canvas_resize(self, event):
        """Handle canvas resize to maintain proper figure layout"""
        try:
            self.figure.tight_layout()
            self.canvas.draw_idle()
        except Exception:
            pass  # Ignore errors during resize


    def closeEvent(self, event):
        self.save_recent_files()
        super().closeEvent(event)

    def save_recent_files(self):
        try:
            with open(self.RECENT_FILES_PATH, "w") as f:
                json.dump(self.recent_files, f)
        except Exception as e:
            print(f"Failed to save recent files: {e}")

    # ==================== ROI File Management ====================

    def get_roi_file_path(self, image_path=None):
        """
        Get the ROI file path for the given or current image.
        ROI file uses same basename as image with .roi extension.

        Parameters:
        - image_path: Path to image file (uses current_image_path if None)

        Returns:
        - str: Path to .roi file, or None if no image loaded
        """
        if image_path is None:
            image_path = self.current_image_path
        if image_path is None:
            return None

        # Replace extension with .roi
        base_path = os.path.splitext(image_path)[0]
        return f"{base_path}.roi"

    def save_roi_file(self, roi_x, roi_y, roi_w, roi_h, edge_type="Unknown", confidence=0.0):
        """
        Save current ROI configuration to a .roi JSON file.

        Parameters:
        - roi_x, roi_y: Top-left corner of ROI
        - roi_w, roi_h: Width and height of ROI
        - edge_type: Detected edge type ("V-Edge", "H-Edge", "Mixed")
        - confidence: Edge detection confidence (0-100)

        Returns:
        - str: Path to saved .roi file, or None on failure
        """
        if self.current_image_path is None:
            self.statusBar().showMessage("⚠️ No image loaded - cannot save ROI")
            return None

        roi_file_path = self.get_roi_file_path()
        if roi_file_path is None:
            return None

        from datetime import datetime

        roi_data = {
            "version": "1.0",
            "source_image": os.path.basename(self.current_image_path),
            "source_image_full_path": self.current_image_path,
            "timestamp": datetime.now().isoformat(),
            "image_dimensions": {
                "width": self.image_w,
                "height": self.image_h
            },
            "roi": {
                "x": roi_x,
                "y": roi_y,
                "w": roi_w,
                "h": roi_h
            },
            "edge_type": edge_type,
            "edge_threshold": self.edge_threshold,
            "confidence": confidence
        }

        try:
            with open(roi_file_path, "w") as f:
                json.dump(roi_data, f, indent=2)

            # Store current ROI data for reference
            self.current_roi_data = roi_data

            # Add to recent ROI files list
            self.add_to_recent_roi_files(roi_file_path)

            self.statusBar().showMessage(f"💾 ROI saved: {os.path.basename(roi_file_path)}")
            return roi_file_path

        except Exception as e:
            self.statusBar().showMessage(f"❌ Failed to save ROI: {e}")
            print(f"Failed to save ROI file: {e}")
            return None

    def load_roi_file(self, roi_file_path=None):
        """
        Load ROI configuration from a .roi JSON file.

        Parameters:
        - roi_file_path: Path to .roi file (auto-detect from current image if None)

        Returns:
        - dict: ROI data dictionary, or None on failure
        """
        # Auto-detect ROI file path if not provided
        if roi_file_path is None:
            roi_file_path = self.get_roi_file_path()

        if roi_file_path is None or not os.path.exists(roi_file_path):
            self.statusBar().showMessage("⚠️ No ROI file found for current image")
            return None

        try:
            with open(roi_file_path, "r") as f:
                roi_data = json.load(f)

            # Validate version
            version = roi_data.get("version", "unknown")
            if version != "1.0":
                print(f"Warning: ROI file version {version} may not be fully compatible")

            # Store loaded ROI data
            self.current_roi_data = roi_data

            self.statusBar().showMessage(f"📂 ROI loaded: {os.path.basename(roi_file_path)}")
            return roi_data

        except json.JSONDecodeError as e:
            self.statusBar().showMessage(f"❌ Invalid ROI file format: {e}")
            return None
        except Exception as e:
            self.statusBar().showMessage(f"❌ Failed to load ROI: {e}")
            print(f"Failed to load ROI file: {e}")
            return None

    def apply_roi_from_file(self, roi_file_path=None):
        """
        Load ROI file and apply it - shows ROI markers with SFR values on image.
        Does NOT update SFR plot when loading multiple ROIs.

        Parameters:
        - roi_file_path: Path to .roi file (auto-detect from current image if None)

        Returns:
        - bool: True if ROI was applied successfully
        """
        roi_data = self.load_roi_file(roi_file_path)
        if roi_data is None:
            return False

        # Check if this is a multi-ROI config file (has "rois" array) or single ROI file
        if "rois" in roi_data:
            # Multi-ROI config format - apply positions and calculate SFR for current image
            rois = roi_data.get("rois", [])
            if not rois:
                self.statusBar().showMessage("⚠️ No ROIs found in config file")
                return False

            # Clear existing markers first
            self.roi_markers = []
            self.image_label.roi_markers = []

            valid_count = 0
            # Apply ROI positions and calculate SFR for each on current image
            for roi_item in rois:
                x = roi_item.get("x", 0)
                y = roi_item.get("y", 0)
                w = roi_item.get("w", 40)
                h = roi_item.get("h", 40)
                name = roi_item.get("name", f"ROI_{len(self.roi_markers) + 1}")

                # Validate ROI is within image bounds
                if x + w > self.image_w or y + h > self.image_h:
                    continue

                # Calculate SFR value for this ROI on current image
                roi_image = self.raw_data[y:y+h, x:x+w]
                sfr_value = self.calculate_sfr_value_only(roi_image)

                # Add to markers list with calculated SFR value
                self.roi_markers.append((x, y, w, h, name, sfr_value))
                self.image_label.roi_markers.append((x, y, w, h, name, sfr_value))
                valid_count += 1

            # Enable ROI Save button (can save with new SFR values)
            self.ui.btn_roi_map_save.setEnabled(True)

            # Update image display
            self.image_label.update()

            self.statusBar().showMessage(
                f"📂 Applied {valid_count} ROI(s) from config - SFR values calculated for current image"
            )
            return True
        else:
            # Single ROI file format (legacy)
            roi = roi_data.get("roi", {})
            x = roi.get("x", 0)
            y = roi.get("y", 0)
            w = roi.get("w", 40)
            h = roi.get("h", 40)

            # Validate ROI is within current image bounds
            if x + w > self.image_w or y + h > self.image_h:
                self.statusBar().showMessage("⚠️ ROI coordinates exceed current image dimensions")
                return False

            # Create QRect and trigger ROI processing
            from PyQt5.QtCore import QRect
            rect = QRect(x, y, w, h)

            # Update selection display on image
            self.image_label.selection_rect = rect
            self.image_label.update()

            # Process the ROI
            self.process_roi(rect)

            edge_type = roi_data.get("edge_type", "Unknown")
            confidence = roi_data.get("confidence", 0)
            self.statusBar().showMessage(
                f"📂 Applied ROI from file: ({x},{y}) {w}×{h} | {edge_type} (Conf: {confidence:.1f}%)"
            )
            return True

    def calculate_sfr_value_only(self, roi_image):
        """
        Calculate SFR value at Ny/4 for a ROI without updating plot.
        Used for displaying SFR values on multiple ROI markers.

        Returns:
        - float: SFR value at Ny/4, or None if calculation fails
        """
        try:
            # Standardize ROI orientation (vertical edge, dark side left, bright side right)
            std_roi, std_edge_type, std_conf = SFRCalculator.standardize_roi_orientation(roi_image)

            # Validate edge on standardized ROI
            is_edge, msg, edge_type, confidence = SFRCalculator.validate_edge(
                std_roi, threshold=self.edge_threshold
            )
            if not is_edge:
                return None

            # Use standardized edge type if available
            used_edge_type = std_edge_type if std_edge_type is not None else edge_type

            # Calculate SFR on standardized ROI
            result = SFRCalculator.calculate_sfr(
                std_roi,
                edge_type=used_edge_type,
                compensate_bias=True,
                compensate_noise=True,
                lsf_smoothing_method=self.lsf_smoothing_method,
                supersampling_factor=self.supersampling_factor,
            )
            freqs, sfr_values, esf, lsf = result

            if freqs is None:
                return None

            # Get SFR value at ny/4
            ny_frequency = getattr(self, 'ny_frequency', 0.5)
            ny_4 = ny_frequency / 4
            frequencies_compensated = freqs * 4

            if len(frequencies_compensated) > 1:
                idx_ny4 = np.argmin(np.abs(frequencies_compensated - ny_4))
                if idx_ny4 < len(sfr_values):
                    sfr_at_ny4 = sfr_values[idx_ny4]
                    # Linear interpolation for more accuracy
                    if 0 < idx_ny4 < len(frequencies_compensated) - 1:
                        f1, f2 = frequencies_compensated[idx_ny4], frequencies_compensated[idx_ny4 + 1]
                        v1, v2 = sfr_values[idx_ny4], sfr_values[idx_ny4 + 1]
                        if abs(f2 - f1) > 1e-10:
                            sfr_at_ny4 = v1 + (ny_4 - f1) * (v2 - v1) / (f2 - f1)
                    return sfr_at_ny4

            return None
        except Exception as e:
            print(f"Error calculating SFR: {e}")
            return None

    def check_roi_file_exists(self):
        """
        Check if a .roi file exists for the current image.

        Returns:
        - bool: True if ROI file exists
        """
        roi_file_path = self.get_roi_file_path()
        return roi_file_path is not None and os.path.exists(roi_file_path)

    # ==================== Recent ROI Files Management ====================

    def load_recent_roi_files(self):
        """Load recent ROI files list from JSON file"""
        try:
            with open(self.RECENT_ROI_FILES_PATH, "r") as f:
                self.recent_roi_files = json.load(f)
        except Exception:
            self.recent_roi_files = []

    def save_recent_roi_files(self):
        """Save recent ROI files list to JSON file"""
        try:
            with open(self.RECENT_ROI_FILES_PATH, "w") as f:
                json.dump(self.recent_roi_files, f)
        except Exception as e:
            print(f"Failed to save recent ROI files: {e}")

    def add_to_recent_roi_files(self, roi_file_path):
        """Add ROI file to recent list"""
        if roi_file_path in self.recent_roi_files:
            self.recent_roi_files.remove(roi_file_path)
        self.recent_roi_files.insert(0, roi_file_path)
        if len(self.recent_roi_files) > self.max_recent_roi_files:
            self.recent_roi_files = self.recent_roi_files[:self.max_recent_roi_files]
        self.update_recent_roi_list()
        self.save_recent_roi_files()

    def update_recent_roi_list(self):
        """Update the recent ROI combo box"""
        self.ui.recent_roi_combo.clear()
        self.ui.recent_roi_combo.addItem("-- Select ROI File --")
        for f in self.recent_roi_files:
            # Show only filename in combo, store full path as data
            filename = os.path.basename(f)
            self.ui.recent_roi_combo.addItem(filename, f)

    def on_recent_roi_selected(self, index):
        """Handle selection of ROI file from combo - enable Apply button"""
        if index <= 0:  # Skip placeholder
            self.loaded_roi_file_path = None
            self.ui.btn_roi_map_apply.setEnabled(False)
            return
        roi_file_path = self.ui.recent_roi_combo.itemData(index)
        if roi_file_path and os.path.exists(roi_file_path):
            # Store the selected file path and enable Apply button
            self.loaded_roi_file_path = roi_file_path
            self.ui.btn_roi_map_apply.setEnabled(True)
            self.statusBar().showMessage(f"📂 ROI file selected: {os.path.basename(roi_file_path)} - Click 'ROI apply' to apply")
        elif roi_file_path:
            QMessageBox.warning(self, "ROI File Not Found", f"ROI file not found: {roi_file_path}")
            self.recent_roi_files.remove(roi_file_path)
            self.update_recent_roi_list()
            self.save_recent_roi_files()
            self.loaded_roi_file_path = None
            self.ui.btn_roi_map_apply.setEnabled(False)

    # ==================== End Recent ROI Files Management ====================

    # ==================== ROI Setting Handlers ====================

    def on_roi_plan_mode_changed(self):
        """Handle ROI Plan checkbox change - when enabled, enter planning mode for ROI positions"""
        self.roi_plan_mode = self.ui.checkBox.isChecked()

        if self.roi_plan_mode:
            # Enable ROI Manual and ROI Detect buttons
            self.ui.btn_roi_manual.setEnabled(True)
            # Enable ROI Detect only if edge is applied
            self.ui.btn_roi_detact.setEnabled(self.edge_overlay_applied and self.locked_edge_mask is not None)
            # Clear all marks when entering plan mode
            self.clear_all_marks()
            self.statusBar().showMessage("📋 ROI Plan Mode: Pick ROI positions → Save config → Apply to any image")
        else:
            # Disable ROI buttons when ROI Plan Mode is off
            self.ui.btn_roi_detact.setEnabled(False)
            self.ui.btn_roi_manual.setEnabled(False)
            # Exit ROI manual mode
            self.roi_manual_mode = False
            self.ui.btn_roi_manual.setChecked(False)
            # Clear ROI markers from image
            self.clear_roi_markers()
            self.statusBar().showMessage("📊 ROI Plan Mode: OFF - Ready for SFR measurement")

    def on_roi_detect(self):
        """Handle ROI Detect button - auto-detect edges for ROI placement"""
        if self.raw_data is None:
            self.statusBar().showMessage("⚠️ Load an image first")
            return
        # Clear all marks before detecting
        self.clear_all_marks()
        # TODO: Implement auto-detection of ROI positions based on edge detection
        self.statusBar().showMessage("🔍 ROI Detect: Auto-detecting edge positions...")

    def on_roi_manual(self):
        """Handle ROI Manual button - toggle manual ROI placement mode"""
        if self.raw_data is None:
            self.statusBar().showMessage("⚠️ Load an image first")
            return

        # Toggle ROI manual mode
        self.roi_manual_mode = not self.roi_manual_mode
        self.ui.btn_roi_manual.setChecked(self.roi_manual_mode)

        if self.roi_manual_mode:
            # Clear all marks when starting manual ROI mode
            self.clear_all_marks()
            # Enter ROI plan mode
            self.roi_plan_mode = True
            self.statusBar().showMessage(f"✋ ROI Manual: Click to place ROI markers (Size: {self.click_select_size}×{self.click_select_size})")
        else:
            num_markers = len(self.roi_markers)
            self.statusBar().showMessage(f"📋 ROI Manual mode OFF - {num_markers} ROI(s) placed")

    def clear_roi_markers(self):
        """Clear all ROI markers and visual marks from the image (except the loaded .raw image)"""
        self.clear_all_marks()
        # Disable ROI Save button since no markers exist
        self.ui.btn_roi_map_save.setEnabled(False)

    def on_roi_map_load(self):
        """Handle Load .roi button - open file dialog and save path to recent ROI combo"""
        from PyQt5.QtWidgets import QFileDialog
        roi_file, _ = QFileDialog.getOpenFileName(
            self, "Load ROI File", "", "ROI Files (*.roi);;All Files (*)"
        )
        if roi_file:
            # Save to recent ROI files
            self.add_to_recent_roi_files(roi_file)
            # Store the loaded file path
            self.loaded_roi_file_path = roi_file
            # Enable ROI Apply button
            self.ui.btn_roi_map_apply.setEnabled(True)
            # Select the file in combo
            self.ui.recent_roi_combo.setCurrentIndex(1)  # First item after placeholder
            self.statusBar().showMessage(f"📂 ROI file loaded: {os.path.basename(roi_file)} - Click 'ROI apply' to apply")

    def on_roi_map_apply(self):
        """Handle ROI Apply button - apply the loaded ROI config to current image"""
        if self.loaded_roi_file_path is None:
            # Try to get from combo selection
            index = self.ui.recent_roi_combo.currentIndex()
            if index > 0:
                self.loaded_roi_file_path = self.ui.recent_roi_combo.itemData(index)

        if self.loaded_roi_file_path and os.path.exists(self.loaded_roi_file_path):
            # Clear all marks before applying new ROI config
            self.clear_all_marks()
            self.apply_roi_from_file(self.loaded_roi_file_path)
        else:
            self.statusBar().showMessage("⚠️ No ROI file loaded - use 'Load .roi' first")

    def on_roi_load(self):
        """Handle ROI Load button - load ROI configuration from file"""
        if self.current_image_path is None:
            self.statusBar().showMessage("⚠️ Load an image first")
            return

        # Try to load ROI file for current image
        if self.check_roi_file_exists():
            self.apply_roi_from_file()
        else:
            # Show file dialog to select ROI file
            from PyQt5.QtWidgets import QFileDialog
            roi_file, _ = QFileDialog.getOpenFileName(
                self, "Load ROI File", "", "ROI Files (*.roi);;All Files (*)"
            )
            if roi_file:
                self.apply_roi_from_file(roi_file)

    def on_roi_save(self):
        """Handle ROI Save button - save all ROI markers to file"""
        if not self.roi_markers or len(self.roi_markers) == 0:
            self.statusBar().showMessage("⚠️ No ROI markers to save")
            return

        if self.current_image_path is None:
            self.statusBar().showMessage("⚠️ Load an image first")
            return

        # Save all ROI markers to file
        saved_path = self.save_roi_markers_file()
        if saved_path:
            self.statusBar().showMessage(f"💾 {len(self.roi_markers)} ROI(s) saved to: {os.path.basename(saved_path)}")

    def save_roi_markers_file(self):
        """Save all ROI markers to a .roi JSON file"""
        if self.current_image_path is None:
            return None

        roi_file_path = self.get_roi_file_path()
        if roi_file_path is None:
            return None

        from datetime import datetime

        # Build ROI config with positions only (SFR calculated when applied)
        roi_list = []
        for marker in self.roi_markers:
            if len(marker) >= 5:
                x, y, w, h, name = marker[:5]
            else:
                continue  # Skip invalid markers

            roi_item = {
                "name": name,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h)
            }
            roi_list.append(roi_item)

        roi_data = {
            "version": "1.0",
            "type": "roi_config",  # Indicates this is a position config file
            "source_image": os.path.basename(self.current_image_path),
            "timestamp": datetime.now().isoformat(),
            "image_dimensions": {
                "width": self.image_w,
                "height": self.image_h
            },
            "roi_size": self.click_select_size,
            "roi_count": len(roi_list),
            "rois": roi_list
        }

        try:
            with open(roi_file_path, "w") as f:
                json.dump(roi_data, f, indent=2)

            # Store current ROI data for reference
            self.current_roi_data = roi_data

            # Add to recent ROI files list
            self.add_to_recent_roi_files(roi_file_path)

            return roi_file_path

        except Exception as e:
            self.statusBar().showMessage(f"❌ Failed to save ROI: {e}")
            print(f"Failed to save ROI file: {e}")
            return None

    # ==================== End ROI Setting Handlers ====================

    # ==================== End ROI File Management ====================

    def load_recent_files(self):
        try:
            with open(self.RECENT_FILES_PATH, "r") as f:
                self.recent_files = json.load(f)
        except Exception:
            self.recent_files = []

    def add_to_recent_files(self, file_path):
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.recent_files.insert(0, file_path)
        if len(self.recent_files) > self.max_recent_files:
            self.recent_files = self.recent_files[:self.max_recent_files]
        self.update_recent_files_list()
        self.save_recent_files()  # Save immediately for persistence

    def update_recent_files_list(self):
        self.ui.recent_files_combo.clear()
        self.ui.recent_files_combo.addItem("-- Select Recent File --")
        for f in self.recent_files:
            # Show only filename in combo, store full path as data
            filename = os.path.basename(f)
            self.ui.recent_files_combo.addItem(filename, f)

    def on_recent_file_selected(self, index):
        if index <= 0:  # Skip the placeholder item
            return
        file_path = self.ui.recent_files_combo.itemData(index)
        if file_path and os.path.exists(file_path):
            self.load_raw_file_from_path(file_path)
            # Reset combo to placeholder after loading
            self.ui.recent_files_combo.setCurrentIndex(0)
        elif file_path:
            QMessageBox.warning(self, "File Not Found", f"File not found: {file_path}")
            self.recent_files.remove(file_path)
            self.update_recent_files_list()
            self.save_recent_files()  # Save after removing invalid file

    def load_raw_file_from_path(self, fname):
        # This is a refactor of load_raw_file to allow loading from a given path (no dialog)
        if not fname:
            return
        file_size = os.path.getsize(fname)
        detected_w, detected_h, detected_dtype = self.auto_detect_raw_dimensions(file_size, fname)
        if detected_w > 0 and detected_h > 0:
            self.image_w = detected_w
            self.image_h = detected_h
        w = self.image_w
        h = self.image_h
        dtype_options = {"uint8": np.uint8, "uint16": np.uint16, "float32": np.float32}
        dtype_choice = detected_dtype if detected_dtype in dtype_options else "uint16"
        selected_dtype = dtype_options[dtype_choice]
        try:
            self.raw_data = read_raw_image(
                fname, width=w, height=h, dtype=selected_dtype
            )
            if self.raw_data is not None:
                self.current_image_path = fname  # Track current loaded image
                if self.raw_data.dtype != np.uint8:
                    display_data = (
                        (self.raw_data - self.raw_data.min())
                        / (self.raw_data.max() - self.raw_data.min() + 1e-10)
                        * 255
                    ).astype(np.uint8)
                else:
                    display_data = self.raw_data
                self.display_image(display_data)
                self.ui.label_raw_load.setText(
                    f"Loaded: {fname} ({w}x{h}, {dtype_choice})"
                )
                self.add_to_recent_files(fname)
            else:
                QMessageBox.critical(self, "Error", "Failed to read raw file")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error reading raw file: {str(e)}")

    def load_raw_file(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Raw File", "", "Raw Files (*.raw);;All Files (*)"
        )
        if not fname:
            return

        # Auto-detect image dimensions based on file size and filename
        file_size = os.path.getsize(fname)
        detected_w, detected_h, detected_dtype = self.auto_detect_raw_dimensions(file_size, fname)

        # Auto-apply detected values
        if detected_w > 0 and detected_h > 0:
            self.image_w = detected_w
            self.image_h = detected_h

        dtype_options = {"uint8": np.uint8, "uint16": np.uint16, "float32": np.float32}
        dtype_choice = detected_dtype if detected_dtype in dtype_options else "uint16"
        selected_dtype = dtype_options[dtype_choice]

        try:
            # Use the improved read_raw_image function
            self.raw_data = read_raw_image(
                fname, width=self.image_w, height=self.image_h, dtype=selected_dtype
            )

            if self.raw_data is not None:
                self.current_image_path = fname  # Track current loaded image
                # Normalize to 8-bit for display if necessary
                if self.raw_data.dtype != np.uint8:
                    display_data = (
                        (self.raw_data - self.raw_data.min())
                        / (self.raw_data.max() - self.raw_data.min() + 1e-10)
                        * 255
                    ).astype(np.uint8)
                else:
                    display_data = self.raw_data

                self.display_image(display_data)
                self.ui.label_raw_load.setText(
                    f"Loaded: {fname} ({self.image_w}x{self.image_h}, {dtype_choice})"
                )
                self.add_to_recent_files(fname)
            else:
                QMessageBox.critical(self, "Error", "Failed to read raw file")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error reading raw file: {str(e)}")

    def auto_detect_raw_dimensions(self, file_size, filename=None):
        """
        Auto-detect raw image dimensions based on file size and optionally filename.
        Returns (width, height, dtype_name) tuple.

        Detection strategies:
        1. Parse dimensions from filename (e.g., image_1920x1080.raw, image_1920_1080_16bit.raw)
        2. Match against common raw image sizes
        3. Try common aspect ratios
        4. Try square dimensions
        """
        import re

        # Strategy 1: Try to parse dimensions from filename
        if filename:
            basename = os.path.basename(filename)

            # Pattern: WxH (e.g., 1920x1080, 4000x3000)
            match = re.search(r'(\d{3,5})[xX](\d{3,5})', basename)
            if match:
                w, h = int(match.group(1)), int(match.group(2))
                # Determine data type based on file size
                for bpp, dtype_name in [(1, "uint8"), (2, "uint16"), (4, "float32")]:
                    if w * h * bpp == file_size:
                        return (w, h, dtype_name)

            # Pattern: W_H (e.g., 1920_1080, 4000_3000)
            match = re.search(r'(\d{3,5})_(\d{3,5})', basename)
            if match:
                w, h = int(match.group(1)), int(match.group(2))
                for bpp, dtype_name in [(1, "uint8"), (2, "uint16"), (4, "float32")]:
                    if w * h * bpp == file_size:
                        return (w, h, dtype_name)

            # Pattern: W-H (e.g., 1920-1080)
            match = re.search(r'(\d{3,5})-(\d{3,5})', basename)
            if match:
                w, h = int(match.group(1)), int(match.group(2))
                for bpp, dtype_name in [(1, "uint8"), (2, "uint16"), (4, "float32")]:
                    if w * h * bpp == file_size:
                        return (w, h, dtype_name)

            # Check for bit depth hints in filename
            bit_hint = None
            if '8bit' in basename.lower() or '_8b' in basename.lower():
                bit_hint = 1
            elif '16bit' in basename.lower() or '_16b' in basename.lower():
                bit_hint = 2
            elif '32bit' in basename.lower() or 'float' in basename.lower():
                bit_hint = 4

        # Strategy 2: Common raw image dimensions to check
        common_sizes = [
            # (width, height, bytes_per_pixel, dtype_name)
            # Sensor common sizes - 16-bit first (more common for raw)
            (4000, 3000, 2, "uint16"),    # 12MP sensor
            (4032, 3024, 2, "uint16"),    # 12MP iPhone
            (4608, 3456, 2, "uint16"),    # 16MP
            (4624, 3472, 2, "uint16"),    # Sony IMX
            (4656, 3496, 2, "uint16"),    # Sony IMX
            (5184, 3888, 2, "uint16"),    # 20MP
            (5472, 3648, 2, "uint16"),    # 20MP
            (6000, 4000, 2, "uint16"),    # 24MP
            (6016, 4016, 2, "uint16"),    # Sony A7
            (6048, 4024, 2, "uint16"),    # Sony
            (8256, 5504, 2, "uint16"),    # 45MP
            (8192, 5464, 2, "uint16"),    # Canon R5
            (7952, 5304, 2, "uint16"),    # Canon 5D
            (9504, 6336, 2, "uint16"),    # 60MP
            # Video/display resolutions - 16-bit
            (640, 480, 2, "uint16"),
            (640, 640, 2, "uint16"),
            (800, 600, 2, "uint16"),
            (1024, 768, 2, "uint16"),
            (1280, 720, 2, "uint16"),     # HD 720p
            (1280, 960, 2, "uint16"),
            (1920, 1080, 2, "uint16"),    # Full HD
            (2048, 1536, 2, "uint16"),    # 3MP
            (2560, 1440, 2, "uint16"),    # QHD
            (2592, 1944, 2, "uint16"),    # 5MP
            (3264, 2448, 2, "uint16"),    # 8MP
            (3840, 2160, 2, "uint16"),    # 4K UHD
            (4096, 2160, 2, "uint16"),    # 4K DCI
            # Square sizes - 16-bit (common for test patterns)
            (256, 256, 2, "uint16"),
            (512, 512, 2, "uint16"),
            (640, 640, 2, "uint16"),
            (1024, 1024, 2, "uint16"),
            (2048, 2048, 2, "uint16"),
            (4096, 4096, 2, "uint16"),
            # 8-bit versions
            (640, 480, 1, "uint8"),
            (800, 600, 1, "uint8"),
            (1024, 768, 1, "uint8"),
            (1280, 720, 1, "uint8"),
            (1280, 960, 1, "uint8"),
            (1920, 1080, 1, "uint8"),
            (2048, 1536, 1, "uint8"),
            (2592, 1944, 1, "uint8"),
            (3264, 2448, 1, "uint8"),
            (4032, 3024, 1, "uint8"),
            (4096, 2160, 1, "uint8"),
            (256, 256, 1, "uint8"),
            (512, 512, 1, "uint8"),
            (640, 640, 1, "uint8"),
            (640, 641, 1, "uint8"),
            (1024, 1024, 1, "uint8"),
            (2048, 2048, 1, "uint8"),
            (4096, 4096, 1, "uint8"),
        ]

        # Check each common size
        for w, h, bpp, dtype_name in common_sizes:
            expected_size = w * h * bpp
            if file_size == expected_size:
                return (w, h, dtype_name)

        # Strategy 3: Try to find square image dimensions
        for bpp, dtype_name in [(2, "uint16"), (1, "uint8"), (4, "float32")]:
            pixels = file_size // bpp
            side = int(np.sqrt(pixels))
            if side * side * bpp == file_size:
                return (side, side, dtype_name)

        # Strategy 4: Try common aspect ratios (4:3, 16:9, 3:2, 1.5:1)
        for bpp, dtype_name in [(2, "uint16"), (1, "uint8"), (4, "float32")]:
            pixels = file_size // bpp
            if pixels == 0:
                continue

            # 4:3 aspect ratio
            h = int(np.sqrt(pixels * 3 / 4))
            w = int(h * 4 / 3)
            if w * h * bpp == file_size:
                return (w, h, dtype_name)

            # 16:9 aspect ratio
            h = int(np.sqrt(pixels * 9 / 16))
            w = int(h * 16 / 9)
            if w * h * bpp == file_size:
                return (w, h, dtype_name)

            # 3:2 aspect ratio
            h = int(np.sqrt(pixels * 2 / 3))
            w = int(h * 3 / 2)
            if w * h * bpp == file_size:
                return (w, h, dtype_name)

            # Try factorization for other dimensions
            # Find factors close to common aspect ratios
            for aspect_w, aspect_h in [(4, 3), (16, 9), (3, 2), (16, 10), (5, 4)]:
                h_try = int(np.sqrt(pixels * aspect_h / aspect_w))
                for h_offset in range(-2, 3):  # Try nearby values
                    h_test = h_try + h_offset
                    if h_test <= 0:
                        continue
                    if pixels % h_test == 0:
                        w_test = pixels // h_test
                        if w_test * h_test * bpp == file_size:
                            return (w_test, h_test, dtype_name)

        # Could not detect, return defaults based on file size estimation
        # Assume uint16 and try to guess reasonable dimensions
        pixels_16bit = file_size // 2
        if pixels_16bit > 0:
            side = int(np.sqrt(pixels_16bit))
            return (side, side, "uint16")

        return (640, 640, "uint16")

    def display_image(self, numpy_img):
        """將 NumPy array 轉換為 QPixmap 顯示"""
        # 簡單的顯示轉換，不處理 Demosaic，直接顯示 Raw 亮度
        disp_img = numpy_img.astype(np.uint8)
        h, w = disp_img.shape

        # Store display data for edge overlay
        self.display_data = disp_img.copy()

        # Reset zoom level for new image
        self.image_label.zoom_level = 1.0

        # Pass image dimensions to image_label
        self.image_label.image_w = w
        self.image_label.image_h = h

        # Check if locked edge pattern exists (Apply Edge was clicked)
        if self.edge_overlay_applied and self.locked_edge_mask is not None:
            # Use the locked edge pattern on new image
            self.display_with_locked_edge()
        else:
            # Show original image without edge
            bytes_per_line = w
            q_img = QImage(
                disp_img.tobytes(), w, h, bytes_per_line, QImage.Format_Grayscale8
            )
            pixmap = QPixmap.fromImage(q_img)

            # Store original pixmap
            self.image_label.pixmap_original = pixmap
            self.image_label.pixmap_scaled = pixmap

            # Set initial display size
            self.image_label.setPixmap(pixmap)

        self.image_label.setMinimumSize(640, 640)
        self.image_label.setMaximumSize(16777215, 16777215)

    def on_smoothing_method_changed(self):
        """Handle LSF smoothing method selection change"""
        self.lsf_smoothing_method = self.ui.method_combo.currentText()
        self.statusBar().showMessage(
            f"LSF Smoothing Method Changed: {self.lsf_smoothing_method}"
        )

    def on_stabilize_filter_changed(self):
        """Handle SFR stabilize filter checkbox change"""
        self.sfr_stabilize_enabled = self.ui.stabilize_checkbox.isChecked()

        if self.sfr_stabilize_enabled:
            self.statusBar().showMessage(
                "✓ SFR Stabilize Filter: ENABLED (will average 3 samples for stability)"
            )
        else:
            self.statusBar().showMessage(
                "SFR Stabilize Filter: DISABLED (single measurement)"
            )

    def on_supersampling_changed(self):
        """Handle supersampling factor spinbox change"""
        self.supersampling_factor = self.ui.supersampling_spinbox.value()
        self.statusBar().showMessage(
            f"Supersampling Factor: {self.supersampling_factor}x (px/cycle ratio maintained)"
        )

    def on_selection_mode_changed(self):
        """Handle selection mode change (Drag vs Click vs ROI map)"""
        if self.ui.radio_drag.isChecked():
            self.selection_mode = "drag"
            self.ui.recent_roi_combo.setEnabled(False)
            self.statusBar().showMessage("Selection Mode: Drag Select (draw rectangle)")
        elif self.ui.radio_click.isChecked():
            self.selection_mode = "click"
            self.ui.recent_roi_combo.setEnabled(False)
            self.statusBar().showMessage(
                f"Selection Mode: Click {self.click_select_size}×{self.click_select_size} (single click to select area)"
            )
        elif self.ui.radio_script_roi.isChecked():
            self.selection_mode = "script_roi"
            self.ui.recent_roi_combo.setEnabled(True)
            # Clear all SFR results and marks when entering ROI map mode
            self.clear_all_marks()
            self.statusBar().showMessage("Selection Mode: ROI map (select from saved ROI config files)")

    def clear_all_marks(self):
        """Clear all visual marks and SFR results from the image (keep only the loaded .raw image)"""
        # Clear ROI markers
        self.roi_markers = []
        if hasattr(self, 'image_label') and self.image_label:
            self.image_label.roi_markers = []

            # Clear selection rectangle
            self.image_label.selection_rect = None
            self.image_label.selection_start = None
            self.image_label.selection_end = None
            self.image_label.is_selecting = False

            # Clear SFR value display on image
            self.image_label.roi_sfr_value = None
            self.image_label.roi_position = None

            # Update image display
            self.image_label.update()

        # Clear SFR plots
        if hasattr(self, 'ax_esf') and self.ax_esf:
            self.ax_esf.clear()
            self.ax_esf.set_title("ESF (Edge Spread Function)", fontsize=10, fontweight="bold")
        if hasattr(self, 'ax_lsf') and self.ax_lsf:
            self.ax_lsf.clear()
            self.ax_lsf.set_title("LSF (Line Spread Function)", fontsize=10, fontweight="bold")
        if hasattr(self, 'ax_sfr') and self.ax_sfr:
            self.ax_sfr.clear()
            self.ax_sfr.set_title("SFR/MTF", fontsize=10, fontweight="bold")
        if hasattr(self, 'ax_roi') and self.ax_roi:
            self.ax_roi.clear()
            self.ax_roi.set_title("ROI Image", fontsize=10, fontweight="bold")
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.draw()

        # Clear last SFR data
        self.last_sfr_data = None

    def on_click_size_changed(self):
        """Handle click select size change"""
        self.click_select_size = self.ui.click_size_input.value()
        # Update radio button label
        self.ui.radio_click.setText(
            f"Click ({self.click_select_size}×{self.click_select_size})"
        )
        # Update status if click mode is active
        if self.ui.radio_click.isChecked():
            self.statusBar().showMessage(
                f"Selection Mode: Click {self.click_select_size}×{self.click_select_size} (single click to select area)"
            )

    def on_ny_freq_slider_changed(self):
        """Handle Nyquist frequency slider change and update SFR plot"""
        # Slider value is 10-100, convert to 0.1-1.0
        slider_val = self.ui.ny_freq_slider.value()
        self.ny_frequency = slider_val / 100.0

        # Update the label
        self.ui.ny_freq_value_label.setText(f"{self.ny_frequency:.2f}")
        self.statusBar().showMessage(f"Nyquist frequency set to {self.ny_frequency:.2f}")

        # If we have stored SFR data, re-plot with new Ny frequency
        if hasattr(self, 'last_sfr_data') and self.last_sfr_data is not None:
            freqs, sfr_values, esf, lsf, edge_type, roi_image = self.last_sfr_data
            self.plot_sfr(freqs, sfr_values, esf, lsf, edge_type, roi_image)

    def on_edge_threshold_changed(self):
        """Handle edge detection threshold slider change"""
        self.edge_threshold = self.ui.edge_threshold_slider.value()
        self.ui.edge_threshold_value_label.setText(str(self.edge_threshold))

        # If edge is locked (Apply Edge was clicked), don't update - keep locked pattern
        if self.locked_edge_mask is not None:
            self.statusBar().showMessage(f"🔒 Edge LOCKED - Threshold change ignored (click Erase Edge to unlock)")
            return

        # Update edge display if edge detect is enabled (preview mode only)
        if self.edge_detect_enabled and self.raw_data is not None:
            self.update_edge_display()
        else:
            self.statusBar().showMessage(f"🔍 Edge Detection Threshold: {self.edge_threshold}")

    def on_edge_detect_changed(self):
        """Handle edge detect checkbox change"""
        self.edge_detect_enabled = self.ui.edge_detect_checkbox.isChecked()

        # If edge is locked, use locked pattern
        if self.locked_edge_mask is not None:
            if self.edge_detect_enabled:
                self.display_with_locked_edge()
                self.statusBar().showMessage(f"🔒 Edge LOCKED - Showing fixed reference pattern")
            else:
                self.show_image_without_edge()
                self.statusBar().showMessage("❌ Edge Detect: OFF (locked pattern still saved)")
            return

        if self.edge_detect_enabled:
            if self.raw_data is not None:
                self.update_edge_display()
                self.statusBar().showMessage(f"✅ Edge Detect: ON (Threshold: {self.edge_threshold})")
            else:
                self.statusBar().showMessage("⚠️ Load an image first to see edge detection")
        else:
            # Restore original image
            if self.display_data is not None:
                self.show_image_without_edge()
            self.statusBar().showMessage("❌ Edge Detect: OFF")

    def update_edge_display(self):
        """Update the image display with thin edge overlay using Canny"""
        if self.raw_data is None or self.display_data is None:
            return

        # Use Canny edge detection for thin edges
        # Threshold slider controls the lower threshold, upper is 2x lower
        lower_threshold = self.edge_threshold
        upper_threshold = self.edge_threshold * 2

        # Apply Canny edge detection (produces thin, clean edges)
        edges = cv2.Canny(self.display_data, lower_threshold, upper_threshold)

        # Create edge mask (Canny outputs 255 for edges, 0 for non-edges)
        edge_mask = edges > 0

        # Create RGB image for display (edges in red)
        h, w = self.display_data.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = self.display_data  # Red channel
        rgb_image[:, :, 1] = self.display_data  # Green channel
        rgb_image[:, :, 2] = self.display_data  # Blue channel

        # Overlay edges in red color
        rgb_image[edge_mask, 0] = 255  # Red
        rgb_image[edge_mask, 1] = 0    # Green
        rgb_image[edge_mask, 2] = 0    # Blue

        # Convert to QImage and display
        bytes_per_line = 3 * w
        q_img = QImage(rgb_image.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Update image label
        self.image_label.pixmap_original = pixmap
        self.image_label.update_zoomed_image()

        # Count edge pixels
        edge_count = np.sum(edge_mask)
        edge_percent = edge_count / (h * w) * 100
        self.statusBar().showMessage(f"🔍 Edge Detect: {edge_count} pixels ({edge_percent:.1f}%) | Threshold: {self.edge_threshold}")

    def show_image_without_edge(self):
        """Restore original grayscale image without edge overlay"""
        if self.display_data is None:
            return

        h, w = self.display_data.shape
        bytes_per_line = w
        q_img = QImage(self.display_data.tobytes(), w, h, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)

        self.image_label.pixmap_original = pixmap
        self.image_label.update_zoomed_image()

    def on_apply_edge(self):
        """Apply and LOCK edge overlay - edge pattern becomes fixed reference"""
        if self.raw_data is None:
            self.statusBar().showMessage("⚠️ Load an image first")
            return

        # Calculate and lock the edge pattern
        lower_threshold = self.edge_threshold
        upper_threshold = self.edge_threshold * 2
        edges = cv2.Canny(self.display_data, lower_threshold, upper_threshold)
        self.locked_edge_mask = edges > 0  # Store the locked edge pattern

        self.ui.edge_detect_checkbox.setChecked(True)
        self.edge_detect_enabled = True
        self.edge_overlay_applied = True

        # Enable ROI Detect button if ROI Plan Mode is on
        if self.roi_plan_mode:
            self.ui.btn_roi_detact.setEnabled(True)

        # Display with locked edge
        self.display_with_locked_edge()
        self.statusBar().showMessage(f"🔒 Edge LOCKED (Threshold: {self.edge_threshold}) - Pattern fixed as reference")

    def on_erase_edge(self):
        """Remove edge overlay from the image display and clear locked pattern"""
        if self.display_data is None:
            self.statusBar().showMessage("⚠️ No image loaded")
            return

        self.ui.edge_detect_checkbox.setChecked(False)
        self.edge_detect_enabled = False
        self.edge_overlay_applied = False
        self.locked_edge_mask = None  # Clear the locked edge pattern

        # Disable ROI Detect button since edge is no longer applied
        self.ui.btn_roi_detact.setEnabled(False)

        self.show_image_without_edge()
        self.statusBar().showMessage("🧹 Edge Erased - Locked pattern cleared")

    def display_with_locked_edge(self):
        """Display current image with the locked edge pattern overlay"""
        if self.display_data is None or self.locked_edge_mask is None:
            return

        h, w = self.display_data.shape
        edge_h, edge_w = self.locked_edge_mask.shape

        # Create RGB image for display
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = self.display_data
        rgb_image[:, :, 1] = self.display_data
        rgb_image[:, :, 2] = self.display_data

        # Overlay locked edges in red (handle size mismatch)
        min_h = min(h, edge_h)
        min_w = min(w, edge_w)
        edge_region = self.locked_edge_mask[:min_h, :min_w]
        rgb_image[:min_h, :min_w, 0][edge_region] = 255  # Red
        rgb_image[:min_h, :min_w, 1][edge_region] = 0    # Green
        rgb_image[:min_h, :min_w, 2][edge_region] = 0    # Blue

        # Convert to QImage and display
        bytes_per_line = 3 * w
        q_img = QImage(rgb_image.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        self.image_label.pixmap_original = pixmap
        self.image_label.update_zoomed_image()

    def on_sfr_mode_clicked(self):
        """Switch to SFR analysis mode"""
        self.view_mode = "sfr"
        self.ui.btn_sfr_mode.setChecked(True)
        self.ui.btn_view_mode.setChecked(False)
        # Reset cursor to arrow
        self.image_label.setCursor(Qt.ArrowCursor)
        self.statusBar().showMessage("📊 SFR Mode: Click or drag to select ROI for analysis")

    def on_view_mode_clicked(self):
        """Switch to VIEW (panning) mode"""
        self.view_mode = "view"
        self.ui.btn_sfr_mode.setChecked(False)
        self.ui.btn_view_mode.setChecked(True)
        # Set cursor to open hand
        self.image_label.setCursor(Qt.OpenHandCursor)
        self.statusBar().showMessage("🖐 VIEW Mode: Click and drag to pan the image")

    def process_roi(self, rect):
        """處理使用者選取的區域 with optional stabilize filter"""
        if self.raw_data is None:
            return

        # 座標轉換
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

        # 邊界檢查
        if w < 5 or h < 5:
            return

        roi = self.raw_data[y : y + h, x : x + w]

        # ROI Plan Mode: Only show ROI info, skip SFR calculation
        if self.roi_plan_mode:
            # Detect edge type for display only
            is_edge, msg, edge_type, confidence = SFRCalculator.validate_edge(roi, threshold=self.edge_threshold)
            if is_edge:
                self.statusBar().showMessage(
                    f"📋 ROI Plan Mode: ({x},{y}) {w}×{h} | {edge_type} (Conf: {confidence:.1f}%) - SFR paused"
                )
                # Store ROI data for potential save
                self.current_roi_data = {
                    "roi": {"x": x, "y": y, "w": w, "h": h},
                    "edge_type": edge_type,
                    "confidence": confidence
                }
            else:
                self.statusBar().showMessage(
                    f"📋 ROI Plan Mode: ({x},{y}) {w}×{h} | No edge detected - SFR paused"
                )
            return  # Skip SFR calculation

        # Standardize ROI orientation: vertical edge with dark side left, bright side right
        std_roi, std_edge_type, std_confidence = SFRCalculator.standardize_roi_orientation(roi)

        # 4. Detect Slit Edge and Edge Orientation (using adjustable threshold) on standardized ROI
        is_edge, msg, edge_type, confidence = SFRCalculator.validate_edge(std_roi, threshold=self.edge_threshold)

        # Use standardized edge type if available
        if std_edge_type is not None:
            edge_type = std_edge_type

        if is_edge:
            if self.sfr_stabilize_enabled:
                # Stabilize filter: Average multiple samples
                self.statusBar().showMessage(
                    f"Edge Detected: {edge_type} - Collecting {3} samples for stability..."
                )
                self.process_roi_with_stabilize(std_roi, edge_type, x, y, w, h)
            else:
                # Normal single measurement
                self.statusBar().showMessage(
                    f"Edge Detected: {edge_type} (Confidence: {confidence:.1f}%) - Calculating SFR..."
                )
                # 5. Input to SFR Algo with edge type and compensation enabled (using standardized ROI)
                result = SFRCalculator.calculate_sfr(
                    std_roi,
                    edge_type=edge_type,
                    compensate_bias=True,
                    compensate_noise=True,
                    lsf_smoothing_method=self.lsf_smoothing_method,
                    supersampling_factor=self.supersampling_factor,
                )
                freqs, sfr_values, esf, lsf = result

                # 6. Show Result
                if freqs is not None:
                    sfr_at_ny4 = self.plot_sfr(freqs, sfr_values, esf, lsf, edge_type, roi_image=std_roi)

                    # Display SFR value at top-right corner of ROI
                    self.image_label.set_roi_sfr_display(sfr_at_ny4, x, y, w, h)

                    # Save ROI to file for future reuse (save original coordinates)
                    self.save_roi_file(x, y, w, h, edge_type, confidence)

                    mtf50_idx = np.argmin(np.abs(sfr_values - 0.5))
                    mtf50_val = freqs[mtf50_idx] if mtf50_idx < len(freqs) else 0
                    self.statusBar().showMessage(
                        f"{edge_type} Edge (Conf: {confidence:.1f}%) | MTF50: {mtf50_val:.3f} cy/px | SFR@ny/4: {sfr_at_ny4:.4f} | SFR Calculated (LSF-{self.lsf_smoothing_method})"
                    )
                else:
                    self.statusBar().showMessage("Error in SFR Calculation")
        else:
            self.statusBar().showMessage(f"Detection Failed: {msg}")
            self.ax_esf.clear()
            self.ax_lsf.clear()
            self.ax_sfr.clear()
            self.canvas.draw()

    def process_roi_with_stabilize(self, roi, edge_type, x, y, w, h):
        """Process ROI with multi-sample averaging for stability"""
        sfr_samples = []
        esf_samples = []
        lsf_samples = []
        freqs = None  # Initialize freqs to prevent crash
        num_samples = 3
        valid_samples = 0

        for i in range(num_samples):
            # Slight random offset (±1-2 pixels) to get different edge positions
            offset_x = np.random.randint(-2, 3)
            offset_y = np.random.randint(-2, 3)

            roi_x = max(0, min(x + offset_x, self.image_w - w))
            roi_y = max(0, min(y + offset_y, self.image_h - h))

            roi_sample = self.raw_data[roi_y : roi_y + h, roi_x : roi_x + w]

            # Standardize each sample orientation before validation and calculation
            roi_sample_std, sample_edge_type, sample_conf = SFRCalculator.standardize_roi_orientation(roi_sample)

            # Validate edge on standardized sample (using adjustable threshold)
            is_edge, msg, edge_type_check, confidence = SFRCalculator.validate_edge(
                roi_sample_std, threshold=self.edge_threshold
            )

            if is_edge and confidence > 50:  # Minimum confidence threshold
                # Use standardized edge type if available
                used_edge_type = sample_edge_type if sample_edge_type is not None else edge_type

                result = SFRCalculator.calculate_sfr(
                    roi_sample_std,
                    edge_type=used_edge_type,
                    compensate_bias=True,
                    compensate_noise=True,
                    lsf_smoothing_method=self.lsf_smoothing_method,
                    supersampling_factor=self.supersampling_factor,
                )
                freqs, sfr_values, esf, lsf = result

                if freqs is not None:
                    sfr_samples.append(sfr_values)
                    esf_samples.append(esf)
                    lsf_samples.append(lsf)
                    valid_samples += 1
                    self.statusBar().showMessage(
                        f"Collecting samples... {valid_samples}/{num_samples}"
                    )

        if valid_samples > 0 and freqs is not None:
            # Average all valid samples
            sfr_averaged = np.mean(sfr_samples, axis=0)
            esf_averaged = np.mean(esf_samples, axis=0)
            lsf_averaged = np.mean(lsf_samples, axis=0)

            # Calculate standard deviation as stability metric
            sfr_std = np.std(sfr_samples, axis=0)
            stability = np.mean(sfr_std)

            # Display results
            self.plot_sfr(freqs, sfr_averaged, esf_averaged, lsf_averaged, edge_type, roi_image=roi)

            # Get SFR value at ny/4 for display (same calculation as in plot_sfr)
            ny_frequency = getattr(self, 'ny_frequency', 0.5)

            ny_4 = ny_frequency / 4
            frequencies_compensated = freqs * 4

            if len(frequencies_compensated) > 1:
                idx_ny4 = np.argmin(np.abs(frequencies_compensated - ny_4))
                if idx_ny4 < len(sfr_averaged):
                    sfr_at_ny4 = sfr_averaged[idx_ny4]
                    if 0 < idx_ny4 < len(frequencies_compensated) - 1:
                        f1, f2 = frequencies_compensated[idx_ny4], frequencies_compensated[idx_ny4 + 1]
                        v1, v2 = sfr_averaged[idx_ny4], sfr_averaged[idx_ny4 + 1]
                        if abs(f2 - f1) > 1e-10:
                            sfr_at_ny4 = v1 + (ny_4 - f1) * (v2 - v1) / (f2 - f1)
                else:
                    sfr_at_ny4 = 0.0
            else:
                sfr_at_ny4 = 0.0

            # Display SFR value at top-left corner of ROI
            self.image_label.set_roi_sfr_display(sfr_at_ny4, x, y, w, h)

            # Save ROI to file for future reuse
            self.save_roi_file(x, y, w, h, edge_type, 85.0)  # Use default confidence for stabilized

            mtf50_idx = np.argmin(np.abs(sfr_averaged - 0.5))
            mtf50_val = freqs[mtf50_idx] if mtf50_idx < len(freqs) else 0

            self.statusBar().showMessage(
                f"{edge_type} Edge | MTF50: {mtf50_val:.3f} cy/px | Stability: ±{stability*100:.2f}% | "
                f"Samples: {valid_samples}/{num_samples} | SFR Calculated (✓ STABILIZED)"
            )
        else:
            self.statusBar().showMessage(f"Error: Could not collect valid edge samples")


    def plot_sfr(self, frequencies, sfr_values, esf, lsf, edge_type="V-Edge", roi_image=None):
        """
        Plot ESF, LSF, SFR/MTF and ROI image in four subplots (2x2 layout)

        Parameters:
        - frequencies: Frequency array for SFR plot (already compensated for supersampling)
        - sfr_values: SFR/MTF values corresponding to frequencies
        - esf: Edge Spread Function (oversampled)
        - lsf: Line Spread Function
        - edge_type: "V-Edge" or "H-Edge"
        - roi_image: ROI image array to display (optional)
        """
        # Use instance supersampling factor
        supersampling_factor = self.supersampling_factor

        # Clear all subplots
        self.ax_esf.clear()
        self.ax_lsf.clear()
        self.ax_sfr.clear()
        self.ax_roi.clear()

        # Plot ROI Image (top-right)
        if roi_image is not None:
            self.ax_roi.imshow(roi_image, cmap='gray', aspect='equal')
            self.ax_roi.set_title(f"ROI Image ({roi_image.shape[1]}×{roi_image.shape[0]})", fontsize=10, fontweight="bold")
        else:
            self.ax_roi.set_title("ROI Image", fontsize=10, fontweight="bold")
        self.ax_roi.axis('off')

        # Plot 1: ESF (Edge Spread Function)
        # Note: ESF is oversampled, so we need to account for that in x-axis
        esf_x = (
            np.arange(len(esf)) / supersampling_factor
        )  # Scale back to original pixel coordinates
        self.ax_esf.plot(esf_x, esf, "b-", linewidth=2)
        self.ax_esf.set_title(
            f"ESF (Edge Spread Function) - {supersampling_factor}x Oversampled",
            fontsize=10,
            fontweight="bold",
        )
        self.ax_esf.set_xlabel("Position (original pixels)", fontsize=9)
        self.ax_esf.set_ylabel("Intensity (0-1)", fontsize=9)
        self.ax_esf.grid(True, alpha=0.3)

        # Plot 2: LSF (Line Spread Function)
        lsf_x = (
            np.arange(len(lsf)) / supersampling_factor
        )  # Scale back to original pixel coordinates

        # If peak is on negative side, invert the LSF
        max_val = np.max(lsf)
        min_val = np.min(lsf)
        if abs(min_val) > abs(max_val):
            lsf = -lsf  # Invert LSF

        self.ax_lsf.plot(lsf_x, lsf, "r-", linewidth=2)
        self.ax_lsf.set_title(
            "LSF (Line Spread Function) - Derivative of ESF",
            fontsize=10,
            fontweight="bold",
        )
        self.ax_lsf.set_xlabel("Position (original pixels)", fontsize=9)
        self.ax_lsf.set_ylabel("Derivative Magnitude", fontsize=9)
        self.ax_lsf.grid(True, alpha=0.3)

        # Calculate and display FWHM for LSF
        try:
            # 1. Detect the highest peak position (use absolute value)
            lsf_abs = np.abs(lsf)
            peak_idx = np.argmax(lsf_abs)
            peak_val = lsf_abs[peak_idx]
            peak_x = lsf_x[peak_idx]

            if peak_val > 0:
                half_max = peak_val / 2

                # 2. Find FWHM - search left and right from peak for half-max crossing
                # Left side: search from peak towards left
                left_x = None
                for i in range(peak_idx, 0, -1):
                    if lsf_abs[i] >= half_max and lsf_abs[i-1] < half_max:
                        # Interpolate to find exact crossing point
                        x_a, x_b = lsf_x[i-1], lsf_x[i]
                        y_a, y_b = lsf_abs[i-1], lsf_abs[i]
                        if y_b != y_a:
                            left_x = x_a + (half_max - y_a) * (x_b - x_a) / (y_b - y_a)
                        else:
                            left_x = x_a
                        break

                # Right side: search from peak towards right
                right_x = None
                for i in range(peak_idx, len(lsf_abs) - 1):
                    if lsf_abs[i] >= half_max and lsf_abs[i+1] < half_max:
                        # Interpolate to find exact crossing point
                        x_a, x_b = lsf_x[i], lsf_x[i+1]
                        y_a, y_b = lsf_abs[i], lsf_abs[i+1]
                        if y_b != y_a:
                            right_x = x_a + (half_max - y_a) * (x_b - x_a) / (y_b - y_a)
                        else:
                            right_x = x_b
                        break

                if left_x is not None and right_x is not None and right_x > left_x:
                    fwhm = right_x - left_x

                    # Draw vertical lines for FWHM
                    self.ax_lsf.axvline(x=left_x, color="purple", linestyle="--", alpha=0.7, linewidth=1.5)
                    self.ax_lsf.axvline(x=right_x, color="purple", linestyle="--", alpha=0.7, linewidth=1.5)

                    # Draw horizontal line at half-max
                    self.ax_lsf.hlines(y=half_max, xmin=left_x, xmax=right_x, color="purple", linestyle="-", alpha=0.5, linewidth=1)

                    # Add text box for FWHM value
                    self.ax_lsf.text(
                        peak_x,
                        peak_val * 0.85,
                        f"FWHM: {fwhm:.2f} px",
                        fontsize=10,
                        fontweight="bold",
                        color="purple",
                        ha="center",
                        va="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="purple", alpha=0.9),
                    )
        except Exception as e:
            print(f"Could not calculate FWHM: {e}")

        # Store last SFR data for re-plotting when Ny changes
        self.last_sfr_data = (frequencies, sfr_values, esf, lsf, edge_type, roi_image)

        # Plot 3: SFR/MTF Result
        # Multiply frequencies by 4 to compensate for supersampling
        frequencies_compensated = frequencies * 4
        self.ax_sfr.plot(frequencies_compensated, sfr_values, "b-", linewidth=2.5, label="MTF")

        # Get Nyquist frequency from stored value (default 0.5)
        ny_frequency = getattr(self, 'ny_frequency', 0.5)

        # Calculate ny/4 reference position (ISO 12233:2023 compliant)
        # ny/4 = user_ny_frequency / 4
        ny_4 = ny_frequency / 4
        self.ax_sfr.axvline(
            x=ny_4,
            color="g",
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
            label=f"ny/4 ({ny_4:.4f} cy/px)",
        )

        # Calculate SFR value at ny/4 with interpolation for better accuracy
        if len(frequencies_compensated) > 1:
            # Find closest frequency to ny/4
            idx_ny4 = np.argmin(np.abs(frequencies_compensated - ny_4))

            # Linear interpolation if not exact match
            if idx_ny4 < len(sfr_values):
                sfr_at_ny4 = sfr_values[idx_ny4]

                # Try interpolation for better accuracy
                if idx_ny4 > 0 and idx_ny4 < len(frequencies_compensated) - 1:
                    # Linear interpolation between two nearest points
                    f1, f2 = (
                        frequencies_compensated[idx_ny4],
                        frequencies_compensated[idx_ny4 + 1],
                    )
                    v1, v2 = sfr_values[idx_ny4], sfr_values[idx_ny4 + 1]
                    if abs(f2 - f1) > 1e-10:
                        sfr_at_ny4 = v1 + (ny_4 - f1) * (v2 - v1) / (f2 - f1)
            else:
                sfr_at_ny4 = 0.0
        else:
            sfr_at_ny4 = 0.0

        # Add text annotation showing ny/4 SFR value (multiplied by 100) as bold text on plot
        ny4_text_value = sfr_at_ny4 * 100
        self.ax_sfr.text(
            ny_4,
            sfr_at_ny4 + 0.08,
            f"{ny4_text_value:.1f}",
            fontsize=11,
            fontweight="bold",
            color="green",
            ha="center",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="green",
                alpha=0.8,
            ),
        )

        self.ax_sfr.set_title(
            f"SFR / MTF Result - {edge_type} (ISO 12233:2023, 4x, Ny={ny_frequency})",
            fontsize=10,
            fontweight="bold",
        )
        self.ax_sfr.set_xlabel("Frequency (cycles/pixel) [4x compensated]", fontsize=9)
        self.ax_sfr.set_ylabel("MTF", fontsize=9)
        self.ax_sfr.set_ylim(0, 1.1)
        # Set x-axis limit to Nyquist frequency
        self.ax_sfr.set_xlim(0, ny_frequency * 1.05)  # Add 5% margin
        self.ax_sfr.legend(loc="upper right", fontsize=8)
        self.ax_sfr.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

        # Return SFR value at ny/4 for result description
        return sfr_at_ny4


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())