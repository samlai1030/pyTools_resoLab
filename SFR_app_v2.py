import os
import sys

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
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy import fftpack
from scipy.ndimage import gaussian_filter, gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, filtfilt, medfilt, savgol_filter, wiener


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

        # 轉為灰階
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
    def validate_edge(roi_image):
        """
        檢測是否為有效的 Slit Edge。
        判斷依據：
        1. 圖像梯度是否足夠強 (有邊緣)。
        2. 邊緣是否接近直線。
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

        # 簡單判定：最大梯度強度需大於某個閾值
        if np.max(magnitude) < 50:
            return False, "Low Contrast / No Edge detected", "No Edge", 0

        # 檢測邊緣方向
        edge_type, confidence, _ = SFRCalculator.detect_edge_orientation(roi_image)

        return True, "Edge Detected", edge_type, confidence

    @staticmethod
    def calculate_sfr(
        roi_image,
        edge_type="V-Edge",
        compensate_bias=True,
        compensate_noise=True,
        lsf_smoothing_method="savgol",
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

        Returns:
        - frequencies: 頻率陣列 (cycles/pixel)
        - sfr: SFR 值 (歸一化到 DC = 1)
        - esf: Edge Spread Function
        - lsf: Line Spread Function
        """
        # 轉為灰階並正規化到 0-1 範圍
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
            # 使用立方插值進行 4x 超採樣
            from scipy.interpolate import interp1d

            x_orig = np.arange(len(esf_raw))
            f_cubic = interp1d(x_orig, esf_raw, kind="cubic", fill_value="extrapolate")
            x_new = np.linspace(0, len(esf_raw) - 1, (len(esf_raw) - 1) * 4 + 1)
            esf = f_cubic(x_new)
        else:  # H-Edge
            # 水平邊：對高度方向進行分析
            esf_raw = np.mean(img, axis=1)
            from scipy.interpolate import interp1d

            x_orig = np.arange(len(esf_raw))
            f_cubic = interp1d(x_orig, esf_raw, kind="cubic", fill_value="extrapolate")
            x_new = np.linspace(0, len(esf_raw) - 1, (len(esf_raw) - 1) * 4 + 1)
            esf = f_cubic(x_new)

        # Step 2: 亞像素邊緣位置檢測與對齐 (ISO 12233:2023 Section 7.2)
        # ...existing code...
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

        # 使用 4x 補零以改善頻率解析度
        n_fft = len(lsf_windowed)
        n_fft_padded = n_fft * SUPERSAMPLING_FACTOR
        fft_res = np.abs(fftpack.fft(lsf_windowed, n=n_fft_padded))

        # Step 6: 頻率軸計算與歸一化 (ISO 12233:2023)
        # 考慮超採樣因子（4x），頻率軸需要相應調整
        freqs = fftpack.fftfreq(n_fft_padded, d=0.25)  # 0.25 是超採樣後的像素間距

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
        # Note: Frequency scaling is already handled in fftfreq with d=0.25
        # No additional division by 4 is needed here to avoid double compensation
        frequencies = frequencies / SUPERSAMPLING_FACTOR

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
        self._updating_zoom = False  # Guard flag to prevent recursion
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)

    def mousePressEvent(self, event):
        if not self.pixmap_original:
            return

        # Get current selection mode from parent window
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
        if self.is_selecting and self.pixmap_original:
            self.selection_end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyTools_SFR_Analyzer")
        self.resize(1600, 900)

        # Data
        self.raw_data = None
        self.image_w = 640  # 預設，實際應由使用者輸入
        self.image_h = 640

        # LSF Smoothing method selection
        self.lsf_smoothing_method = "savgol"  # Default method

        # Selection mode: "drag" or "click"
        self.selection_mode = "drag"  # Default: drag to select

        # Click select size (default 30x30)
        self.click_select_size = 30

        # SFR stabilize filter enable/disable
        self.sfr_stabilize_enabled = False  # Default: disabled

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_widget.setStyleSheet("background-color: #f0f0f0;")

        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Left Panel: Image View with Scrollbars
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_load = QPushButton("Load .raw File")
        self.btn_load.setMinimumHeight(35)
        self.btn_load.setStyleSheet("padding: 5px; font-size: 12px;")
        self.btn_load.clicked.connect(self.load_raw_file)

        # Selection Mode Selection (NEW)
        mode_label = QLabel("Selection Mode:")
        mode_label.setStyleSheet(
            "font-weight: bold; font-size: 10px; margin-top: 10px;"
        )

        mode_layout = QHBoxLayout()
        mode_layout.setContentsMargins(0, 5, 0, 5)

        self.radio_drag = QRadioButton("Drag Select")
        self.radio_drag.setChecked(True)
        self.radio_drag.setStyleSheet("font-size: 10px;")
        self.radio_drag.toggled.connect(self.on_selection_mode_changed)

        self.radio_click = QRadioButton("Click (40×40)")
        self.radio_click.setStyleSheet("font-size: 10px;")
        self.radio_click.toggled.connect(self.on_selection_mode_changed)

        # Input box for click select size
        size_label = QLabel("Size:")
        size_label.setStyleSheet("font-size: 10px; font-weight: bold;")

        self.click_size_input = QSpinBox()
        self.click_size_input.setMinimum(10)
        self.click_size_input.setMaximum(200)
        self.click_size_input.setValue(40)
        self.click_size_input.setStyleSheet("font-size: 10px; padding: 2px;")
        self.click_size_input.setMinimumWidth(50)
        self.click_size_input.valueChanged.connect(self.on_click_size_changed)

        mode_layout.addWidget(self.radio_drag)
        mode_layout.addWidget(self.radio_click)
        mode_layout.addWidget(size_label)
        mode_layout.addWidget(self.click_size_input)
        mode_layout.addStretch()

        left_layout.addWidget(self.btn_load)
        left_layout.addWidget(mode_label)
        left_layout.addLayout(mode_layout)

        # Create image label
        self.image_label = ImageLabel(self)
        self.image_label.setStyleSheet("border: 2px solid #333; background: black;")
        self.image_label.setMinimumSize(640, 640)
        self.image_label.setScaledContents(False)
        self.image_label.roi_callback = self.process_roi

        # Create scroll area for image
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setStyleSheet("border: 2px solid #333;")
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Set scroll_area reference in image_label
        self.image_label.scroll_area = self.scroll_area

        left_layout.addWidget(self.btn_load)
        left_layout.addWidget(self.scroll_area, 1)

        # Add ROI preview area with image and size info separated
        roi_container_layout = QVBoxLayout()
        roi_container_layout.setContentsMargins(0, 5, 0, 5)
        roi_container_layout.setSpacing(5)

        # ROI Preview Image Area (main display)
        self.roi_preview_label = QLabel("ROI Preview")
        self.roi_preview_label.setStyleSheet(
            "border: 2px solid #999; background: #ADD8E6; min-height: 150px;"
        )
        self.roi_preview_label.setAlignment(Qt.AlignCenter)
        self.roi_preview_label.setScaledContents(False)
        roi_container_layout.addWidget(self.roi_preview_label, 3)  # 3x space for image

        # ROI Size Information Area (separated)
        self.roi_size_label = QLabel("Size: ---")
        self.roi_size_label.setStyleSheet(
            "border: 1px solid #999; background: #E6F3FF; padding: 8px; font-weight: bold; font-size: 11px; text-align: center;"
        )
        self.roi_size_label.setAlignment(Qt.AlignCenter)
        self.roi_size_label.setMinimumHeight(40)
        roi_container_layout.addWidget(self.roi_size_label, 1)  # 1x space for size info

        left_layout.addLayout(roi_container_layout)

        # Right Panel: SFR Plot & Info
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)

        # LSF Smoothing Method Selection
        method_layout = QHBoxLayout()
        method_label = QLabel("LSF Smoothing Method:")
        method_label.setStyleSheet("font-weight: bold; font-size: 10px;")
        self.method_combo = QComboBox()
        self.method_combo.addItems(
            ["savgol", "gaussian", "median", "uniform", "butterworth", "wiener", "none"]
        )
        self.method_combo.setCurrentText("savgol")  # Default
        self.method_combo.setMinimumWidth(120)
        self.method_combo.setStyleSheet("font-size: 10px; padding: 3px;")
        self.method_combo.currentTextChanged.connect(self.on_smoothing_method_changed)

        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch()

        right_layout.addLayout(method_layout)

        # SFR Stabilize Filter Checkbox
        stabilize_layout = QHBoxLayout()
        self.stabilize_checkbox = QCheckBox("SFR Stabilize Filter (Average 3 samples)")
        self.stabilize_checkbox.setChecked(False)
        self.stabilize_checkbox.setStyleSheet("font-size: 10px;")
        self.stabilize_checkbox.stateChanged.connect(self.on_stabilize_filter_changed)

        stabilize_layout.addWidget(self.stabilize_checkbox)
        stabilize_layout.addStretch()

        right_layout.addLayout(stabilize_layout)

        # Info and Nyquist frequency controls
        info_ny_layout = QHBoxLayout()

        self.info_label = QLabel("Status: Ready")
        self.info_label.setMinimumHeight(40)
        self.info_label.setStyleSheet(
            "background: white; padding: 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 11px;"
        )
        self.info_label.setWordWrap(True)
        info_ny_layout.addWidget(self.info_label, 4)

        # Nyquist frequency input
        ny_label = QLabel("Ny:")
        ny_label.setStyleSheet("font-weight: bold; font-size: 10px;")

        self.ny_freq_input = QLineEdit()
        self.ny_freq_input.setText("0.5")
        self.ny_freq_input.setMaximumWidth(60)
        self.ny_freq_input.setStyleSheet("font-size: 10px; padding: 3px;")
        self.ny_freq_input.setToolTip("Nyquist frequency (0.0-1.0)")
        self.ny_freq_input.editingFinished.connect(self.on_ny_freq_changed)

        info_ny_layout.addWidget(ny_label)
        info_ny_layout.addWidget(self.ny_freq_input)

        right_layout.addLayout(info_ny_layout)

        # Matplotlib Figure - 2x2 subplot layout: 1.1=SFR (top), 2.1=ESF (bottom-left), 2.2=LSF (bottom-right)
        self.figure = Figure(figsize=(12, 9), dpi=100)
        self.figure.patch.set_facecolor("white")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(900, 600)
        self.canvas.setStyleSheet("background: white; border: 1px solid #ccc;")

        # Create subplots: 2 rows, 2 columns layout
        # Top row: SFR spanning full width
        self.ax_sfr = self.figure.add_subplot(211)

        # Bottom row: ESF (left) and LSF (right)
        self.ax_esf = self.figure.add_subplot(223)
        self.ax_lsf = self.figure.add_subplot(224)

        # Set titles and labels for SFR (top)
        self.ax_sfr.set_title("SFR / MTF Result", fontsize=11, fontweight="bold")
        self.ax_sfr.set_xlabel("Frequency (cycles/pixel)", fontsize=10)
        self.ax_sfr.set_ylabel("MTF", fontsize=10)
        self.ax_sfr.grid(True, alpha=0.3)

        # Set titles and labels for ESF (bottom-left)
        self.ax_esf.set_title(
            "ESF (Edge Spread Function)", fontsize=10, fontweight="bold"
        )
        self.ax_esf.set_xlabel("Position (pixels)", fontsize=9)
        self.ax_esf.set_ylabel("Intensity", fontsize=9)
        self.ax_esf.grid(True, alpha=0.3)

        # Set titles and labels for LSF (bottom-right)
        self.ax_lsf.set_title(
            "LSF (Line Spread Function)", fontsize=10, fontweight="bold"
        )
        self.ax_lsf.set_xlabel("Position (pixels)", fontsize=9)
        self.ax_lsf.set_ylabel("Derivative", fontsize=9)
        self.ax_lsf.grid(True, alpha=0.3)

        self.figure.tight_layout()

        right_layout.addWidget(self.info_label)
        right_layout.addWidget(self.canvas, 1)

        # Ratios - adjusted for better balance
        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 1)

    def load_raw_file(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Raw File", "", "Raw Files (*.raw);;All Files (*)"
        )
        if not fname:
            return

        # Get image dimensions
        w, ok = QInputDialog.getInt(self, "Raw Config", "Width:", self.image_w)
        if not ok:
            return
        self.image_w = w

        h, ok2 = QInputDialog.getInt(self, "Raw Config", "Height:", self.image_h)
        if not ok2:
            return
        self.image_h = h

        # Get data type
        dtype_options = {"uint8": np.uint8, "uint16": np.uint16, "float32": np.float32}
        dtype_names = list(dtype_options.keys())
        dtype_choice, ok3 = QInputDialog.getItem(
            self, "Raw Config", "Data Type:", dtype_names, 1
        )
        if not ok3:
            return

        selected_dtype = dtype_options[dtype_choice]

        try:
            # Use the improved read_raw_image function
            self.raw_data = read_raw_image(
                fname, width=self.image_w, height=self.image_h, dtype=selected_dtype
            )

            if self.raw_data is not None:
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
                self.info_label.setText(
                    f"Loaded: {fname} ({self.image_w}x{self.image_h}, {dtype_choice})"
                )
            else:
                QMessageBox.critical(self, "Error", "Failed to read raw file")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error reading raw file: {str(e)}")

    def display_image(self, numpy_img):
        """將 NumPy array 轉換為 QPixmap 顯示"""
        # 簡單的顯示轉換，不處理 Demosaic，直接顯示 Raw 亮度
        disp_img = numpy_img.astype(np.uint8)
        h, w = disp_img.shape
        bytes_per_line = w
        q_img = QImage(
            disp_img.tobytes(), w, h, bytes_per_line, QImage.Format_Grayscale8
        )
        pixmap = QPixmap.fromImage(q_img)

        # Reset zoom level for new image
        self.image_label.zoom_level = 1.0

        # Pass image dimensions to image_label
        self.image_label.image_w = w
        self.image_label.image_h = h

        # Store original pixmap
        self.image_label.pixmap_original = pixmap
        self.image_label.pixmap_scaled = pixmap

        # Set initial display size
        self.image_label.setPixmap(pixmap)
        self.image_label.setMinimumSize(640, 640)
        self.image_label.setMaximumSize(16777215, 16777215)

    def on_smoothing_method_changed(self):
        """Handle LSF smoothing method selection change"""
        self.lsf_smoothing_method = self.method_combo.currentText()
        self.info_label.setText(
            f"LSF Smoothing Method Changed: {self.lsf_smoothing_method}"
        )

    def on_stabilize_filter_changed(self):
        """Handle SFR stabilize filter checkbox change"""
        self.sfr_stabilize_enabled = self.stabilize_checkbox.isChecked()

        if self.sfr_stabilize_enabled:
            self.info_label.setText(
                "✓ SFR Stabilize Filter: ENABLED (will average 3 samples for stability)"
            )
        else:
            self.info_label.setText(
                "SFR Stabilize Filter: DISABLED (single measurement)"
            )

    def on_selection_mode_changed(self):
        """Handle selection mode change (Drag vs Click 40x40)"""
        if self.radio_drag.isChecked():
            self.selection_mode = "drag"
            self.info_label.setText("Selection Mode: Drag Select (draw rectangle)")
        else:
            self.selection_mode = "click"
            self.info_label.setText(
                f"Selection Mode: Click {self.click_select_size}×{self.click_select_size} (single click to select area)"
            )

    def on_click_size_changed(self):
        """Handle click select size change"""
        self.click_select_size = self.click_size_input.value()
        # Update radio button label
        self.radio_click.setText(
            f"Click ({self.click_select_size}×{self.click_select_size})"
        )
        # Update status if click mode is active
        if self.radio_click.isChecked():
            self.info_label.setText(
                f"Selection Mode: Click {self.click_select_size}×{self.click_select_size} (single click to select area)"
            )

    def on_ny_freq_changed(self):
        """Handle Nyquist frequency change"""
        try:
            ny_val = float(self.ny_freq_input.text())
            if ny_val <= 0 or ny_val > 1.0:
                self.ny_freq_input.setText("0.5")
                self.info_label.setText("Nyquist frequency must be between 0 and 1.0")
            else:
                self.info_label.setText(f"Nyquist frequency set to {ny_val}")
        except ValueError:
            self.ny_freq_input.setText("0.5")
            self.info_label.setText("Invalid Nyquist frequency value")

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

        # Display ROI preview
        self.display_roi_preview(roi)

        # 4. Detect Slit Edge and Edge Orientation
        is_edge, msg, edge_type, confidence = SFRCalculator.validate_edge(roi)

        if is_edge:
            if self.sfr_stabilize_enabled:
                # Stabilize filter: Average multiple samples
                self.info_label.setText(
                    f"Edge Detected: {edge_type} - Collecting {3} samples for stability..."
                )
                self.process_roi_with_stabilize(roi, edge_type, x, y, w, h)
            else:
                # Normal single measurement
                self.info_label.setText(
                    f"Edge Detected: {edge_type} (Confidence: {confidence:.1f}%) - Calculating SFR..."
                )
                # 5. Input to SFR Algo with edge type and compensation enabled
                result = SFRCalculator.calculate_sfr(
                    roi,
                    edge_type=edge_type,
                    compensate_bias=True,
                    compensate_noise=True,
                    lsf_smoothing_method=self.lsf_smoothing_method,
                )
                freqs, sfr_values, esf, lsf = result

                # 6. Show Result
                if freqs is not None:
                    sfr_at_ny4 = self.plot_sfr(freqs, sfr_values, esf, lsf, edge_type)

                    # Display SFR value at top-right corner of ROI
                    self.image_label.set_roi_sfr_display(sfr_at_ny4, x, y, w, h)

                    mtf50_idx = np.argmin(np.abs(sfr_values - 0.5))
                    mtf50_val = freqs[mtf50_idx] if mtf50_idx < len(freqs) else 0
                    self.info_label.setText(
                        f"{edge_type} Edge (Conf: {confidence:.1f}%) | MTF50: {mtf50_val:.3f} cy/px | SFR@ny/4: {sfr_at_ny4:.4f} | SFR Calculated (LSF-{self.lsf_smoothing_method})"
                    )
                else:
                    self.info_label.setText("Error in SFR Calculation")
        else:
            self.info_label.setText(f"Detection Failed: {msg}")
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

            # Validate edge
            is_edge, msg, edge_type_check, confidence = SFRCalculator.validate_edge(
                roi_sample
            )

            if is_edge and confidence > 50:  # Minimum confidence threshold
                result = SFRCalculator.calculate_sfr(
                    roi_sample,
                    edge_type=edge_type,
                    compensate_bias=True,
                    compensate_noise=True,
                    lsf_smoothing_method=self.lsf_smoothing_method,
                )
                freqs, sfr_values, esf, lsf = result

                if freqs is not None:
                    sfr_samples.append(sfr_values)
                    esf_samples.append(esf)
                    lsf_samples.append(lsf)
                    valid_samples += 1
                    self.info_label.setText(
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
            self.plot_sfr(freqs, sfr_averaged, esf_averaged, lsf_averaged, edge_type)

            # Get SFR value at ny/4 for display (same calculation as in plot_sfr)
            ny_frequency = 0.5  # Default Nyquist
            if hasattr(self, 'ny_freq_input') and self.ny_freq_input:
                try:
                    ny_frequency = float(self.ny_freq_input.text())
                    if ny_frequency <= 0 or ny_frequency > 1.0:
                        ny_frequency = 0.5
                except:
                    ny_frequency = 0.5

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

            mtf50_idx = np.argmin(np.abs(sfr_averaged - 0.5))
            mtf50_val = freqs[mtf50_idx] if mtf50_idx < len(freqs) else 0

            self.info_label.setText(
                f"{edge_type} Edge | MTF50: {mtf50_val:.3f} cy/px | Stability: ±{stability*100:.2f}% | "
                f"Samples: {valid_samples}/{num_samples} | SFR Calculated (✓ STABILIZED)"
            )
        else:
            self.info_label.setText(f"Error: Could not collect valid edge samples")

    def display_roi_preview(self, roi_image):
        """Display preview of selected ROI with dimensions in separate areas"""
        if roi_image is None or roi_image.size == 0:
            return

        # Get original ROI dimensions
        h_orig, w_orig = roi_image.shape

        # Normalize to 8-bit for display
        if roi_image.dtype != np.uint8:
            preview_data = (
                (roi_image - roi_image.min())
                / (roi_image.max() - roi_image.min() + 1e-10)
                * 255
            ).astype(np.uint8)
        else:
            preview_data = roi_image

        # Create QPixmap from preview
        h, w = preview_data.shape
        bytes_per_line = w
        q_img = QImage(
            preview_data.tobytes(), w, h, bytes_per_line, QImage.Format_Grayscale8
        )
        pixmap = QPixmap.fromImage(q_img)

        # Scale to fit in ROI preview area (max 350x200)
        max_w, max_h = 350, 200
        if pixmap.width() > max_w or pixmap.height() > max_h:
            pixmap = pixmap.scaledToWidth(max_w, Qt.SmoothTransformation)

        # Display image in ROI preview label (only image, no text)
        self.roi_preview_label.setPixmap(pixmap)

        # Display size information in separate size label
        self.roi_size_label.setText(f"Size: {w_orig}×{h_orig} pixels")
        self.roi_size_label.setStyleSheet(
            "border: 1px solid #999; background: #E6F3FF; padding: 8px; font-weight: bold; font-size: 12px; text-align: center; color: #003366;"
        )

    def plot_sfr(self, frequencies, sfr_values, esf, lsf, edge_type="V-Edge"):
        """
        Plot ESF, LSF, and SFR/MTF in three subplots

        Parameters:
        - frequencies: Frequency array for SFR plot (already compensated for supersampling)
        - sfr_values: SFR/MTF values corresponding to frequencies
        - esf: Edge Spread Function (4x oversampled)
        - lsf: Line Spread Function
        - edge_type: "V-Edge" or "H-Edge"
        """
        # Supersampling factor used in ISO 12233:2023
        SUPERSAMPLING_FACTOR = 4

        # Clear all subplots
        self.ax_esf.clear()
        self.ax_lsf.clear()
        self.ax_sfr.clear()

        # Plot 1: ESF (Edge Spread Function)
        # Note: ESF is 4x oversampled, so we need to account for that in x-axis
        esf_x = (
            np.arange(len(esf)) / SUPERSAMPLING_FACTOR
        )  # Scale back to original pixel coordinates
        self.ax_esf.plot(esf_x, esf, "b-", linewidth=2)
        self.ax_esf.set_title(
            "ESF (Edge Spread Function) - 4x Oversampled",
            fontsize=10,
            fontweight="bold",
        )
        self.ax_esf.set_xlabel("Position (original pixels)", fontsize=9)
        self.ax_esf.set_ylabel("Intensity (0-1)", fontsize=9)
        self.ax_esf.grid(True, alpha=0.3)

        # Plot 2: LSF (Line Spread Function)
        lsf_x = (
            np.arange(len(lsf)) / SUPERSAMPLING_FACTOR
        )  # Scale back to original pixel coordinates
        self.ax_lsf.plot(lsf_x, lsf, "r-", linewidth=2)
        self.ax_lsf.set_title(
            "LSF (Line Spread Function) - Derivative of ESF",
            fontsize=10,
            fontweight="bold",
        )
        self.ax_lsf.set_xlabel("Position (original pixels)", fontsize=9)
        self.ax_lsf.set_ylabel("Derivative Magnitude", fontsize=9)
        self.ax_lsf.grid(True, alpha=0.3)

        # Plot 3: SFR/MTF Result
        # Multiply frequencies by 4 to compensate for supersampling
        frequencies_compensated = frequencies * 4
        self.ax_sfr.plot(frequencies_compensated, sfr_values, "b-", linewidth=2.5, label="MTF")

        # Get Nyquist frequency from user input (default 0.5)
        ny_frequency = 0.5  # Default Nyquist
        if hasattr(self, "ny_freq_input") and self.ny_freq_input:
            try:
                ny_frequency = float(self.ny_freq_input.text())
                if ny_frequency <= 0 or ny_frequency > 1.0:
                    ny_frequency = 0.5
            except:
                ny_frequency = 0.5

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
            f"SFR / MTF Result - {edge_type} (ISO 12233:2023, 4x Supersampling, Ny={ny_frequency})",
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
