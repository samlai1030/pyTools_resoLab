# -*- coding: utf-8 -*-
"""
SFR/MTF Calculator - ISO 12233:2023 Standard Implementation.

This module provides the SFRCalculator class for calculating Spatial Frequency Response
(SFR) and Modulation Transfer Function (MTF) according to the ISO 12233:2023 standard.
"""

import cv2
import numpy as np
from scipy import fftpack
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, gaussian_filter1d, uniform_filter1d
from scipy.ndimage import shift as ndimage_shift
from scipy.signal import butter, filtfilt, medfilt, savgol_filter, wiener

from constants import EPSILON


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
                window_length = min(11, len(lsf) if len(lsf) % 2 == 1 else len(lsf) - 1)
                if window_length < 5:
                    window_length = 5
                return savgol_filter(lsf, window_length=window_length, polyorder=3)

            elif method == "gaussian":
                return gaussian_filter1d(lsf, sigma=1.5)

            elif method == "median":
                kernel_size = min(11, len(lsf) if len(lsf) % 2 == 1 else len(lsf) - 1)
                if kernel_size < 5:
                    kernel_size = 5
                return medfilt(lsf, kernel_size=kernel_size)

            elif method == "uniform":
                return uniform_filter1d(lsf, size=5)

            elif method == "butterworth":
                try:
                    b, a = butter(2, 0.1)
                    return filtfilt(b, a, lsf)
                except (ValueError, RuntimeError) as e:
                    print(f"Warning: Butterworth filter failed: {e}, falling back to Savitzky-Golay")
                    window_length = min(11, len(lsf) if len(lsf) % 2 == 1 else len(lsf) - 1)
                    if window_length < 5:
                        window_length = 5
                    return savgol_filter(lsf, window_length=window_length, polyorder=3)

            elif method == "wiener":
                mysize = min(11, len(lsf) if len(lsf) % 2 == 1 else len(lsf) - 1)
                if mysize < 5:
                    mysize = 5
                return wiener(lsf, mysize=mysize)

            else:
                window_length = min(11, len(lsf) if len(lsf) % 2 == 1 else len(lsf) - 1)
                if window_length < 5:
                    window_length = 5
                return savgol_filter(lsf, window_length=window_length, polyorder=3)

        except Exception as e:
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

        gray = (
            roi_image
            if len(roi_image.shape) == 2
            else cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        )
        gray = gray.astype(np.float64)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        mag_x = np.sum(np.abs(sobelx))
        mag_y = np.sum(np.abs(sobely))

        angle = np.arctan2(sobely, sobelx) * 180 / np.pi
        angle = np.mod(angle + 180, 180)

        v_edges = np.sum((angle > 80) & (angle < 100)) / angle.size * 100
        h_edges = np.sum(((angle > 170) | (angle < 10))) / angle.size * 100

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

        if edge_strength_ratio > 1.5:
            confidence = min(100, (edge_strength_ratio - 1.0) * 50)
            return "V-Edge", confidence, details
        elif edge_strength_ratio < 0.67:
            confidence = min(100, (1.0 / edge_strength_ratio - 1.0) * 50)
            return "H-Edge", confidence, details
        else:
            confidence = 50
            return "Mixed", confidence, details

    @staticmethod
    def validate_edge(roi_image, threshold=50):
        """
        檢測是否為有效的 Slit Edge。

        Parameters:
        - roi_image: ROI 圖像
        - threshold: 邊緣檢測閾值 (默認 50)
        """
        if roi_image is None or roi_image.size == 0:
            return False, "Empty ROI", "No Edge", 0

        gray = (
            roi_image
            if len(roi_image.shape) == 2
            else cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        )
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)

        if np.max(magnitude) < threshold:
            return False, "Low Contrast / No Edge detected", "No Edge", 0

        edge_type, confidence, _ = SFRCalculator.detect_edge_orientation(roi_image)

        return True, "Edge Detected", edge_type, confidence

    @staticmethod
    def standardize_roi_orientation(roi_image):
        """
        Standardize ROI orientation for SFR calculation.
        Ensures the edge is vertical with:
        1. Dark side on left, bright side on right
        2. Dark side (black) width increases toward bottom (slant direction)

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

        def get_gray(image):
            if image.ndim == 3 and image.shape[2] >= 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.ndim == 3:
                return np.mean(image, axis=2).astype(image.dtype)
            else:
                return image

        gray_for_analysis = get_gray(img)

        edge_type, confidence, details = SFRCalculator.detect_edge_orientation(gray_for_analysis)

        if edge_type == "H-Edge":
            img = np.rot90(img, k=1)
            gray_for_analysis = get_gray(img)
            edge_type, confidence, details = SFRCalculator.detect_edge_orientation(gray_for_analysis)

        h, w = gray_for_analysis.shape[:2]
        left_half_width = max(1, w // 2)

        left_mean = np.mean(gray_for_analysis[:, :left_half_width])
        right_mean = np.mean(gray_for_analysis[:, left_half_width:])

        if left_mean > right_mean:
            img = np.fliplr(img)
            gray_for_analysis = get_gray(img)

        h, w = gray_for_analysis.shape[:2]

        top_portion_height = max(1, h // 4)
        bottom_portion_height = max(1, h // 4)

        top_rows = gray_for_analysis[:top_portion_height, :]
        bottom_rows = gray_for_analysis[-bottom_portion_height:, :]

        top_profile = np.mean(top_rows, axis=0)
        bottom_profile = np.mean(bottom_rows, axis=0)

        def find_edge_position(profile):
            p_min, p_max = np.min(profile), np.max(profile)
            if p_max - p_min < 1e-6:
                return len(profile) // 2
            p_norm = (profile - p_min) / (p_max - p_min)
            return np.argmin(np.abs(p_norm - 0.5))

        top_edge_pos = find_edge_position(top_profile)
        bottom_edge_pos = find_edge_position(bottom_profile)

        if bottom_edge_pos < top_edge_pos:
            img = np.flipud(img)
            gray_for_analysis = get_gray(img)

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
        計算 SFR (Spatial Frequency Response) - ISO 12233:2023 Standard

        Parameters:
        - roi_image: ROI 圖像
        - edge_type: "V-Edge" 或 "H-Edge"
        - compensate_bias: 補償白區域偏差/亮度偏移 (預設 True)
        - compensate_noise: 補償白區域噪聲 (預設 True)
        - lsf_smoothing_method: LSF 平滑方法 (預設 "savgol")
        - supersampling_factor: 超採樣因子 (預設 4, 範圍 1-16)

        Returns:
        - frequencies: 頻率陣列 (cycles/pixel)
        - sfr: SFR 值 (歸一化到 DC = 1)
        - esf: Edge Spread Function
        - lsf: Line Spread Function
        """
        img = roi_image.astype(np.float64)
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)

        img_min = np.min(img)
        img_max = np.max(img)
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        if compensate_bias:
            white_mask = img > 0.9
            if np.sum(white_mask) > 10:
                white_level = np.mean(img[white_mask])
                bias_correction = 1.0 - white_level
                img = img + bias_correction
                img = np.clip(img, 0, 1)

        if compensate_noise:
            white_mask_noise = img > 0.85
            if np.sum(white_mask_noise) > 20:
                white_noise_std = np.std(img[white_mask_noise])
                if white_noise_std > 0.01:
                    sigma = white_noise_std * 0.5
                    if edge_type == "V-Edge":
                        img = gaussian_filter(img, sigma=(sigma, 0))
                    else:
                        img = gaussian_filter(img, sigma=(0, sigma))

        if edge_type == "V-Edge":
            esf_raw = np.mean(img, axis=0)
        else:
            esf_raw = np.mean(img, axis=1)

        x_orig = np.arange(len(esf_raw))
        f_cubic = interp1d(x_orig, esf_raw, kind="cubic", fill_value="extrapolate")
        x_new = np.linspace(0, len(esf_raw) - 1, (len(esf_raw) - 1) * supersampling_factor + 1)
        esf = f_cubic(x_new)

        esf_min = np.min(esf)
        esf_max = np.max(esf)
        esf_normalized = (esf - esf_min) / (esf_max - esf_min + 1e-10)

        idx_50 = np.argmin(np.abs(esf_normalized - 0.5))

        if idx_50 > 0 and idx_50 < len(esf_normalized) - 1:
            if esf_normalized[idx_50] < 0.5:
                slope = esf_normalized[idx_50 + 1] - esf_normalized[idx_50]
                frac = (0.5 - esf_normalized[idx_50]) / slope if slope != 0 else 0
            else:
                slope = esf_normalized[idx_50] - esf_normalized[idx_50 - 1]
                frac = (esf_normalized[idx_50] - 0.5) / slope if slope != 0 else 0
        else:
            frac = 0

        edge_pos = idx_50 + frac
        shift_amount = edge_pos - int(edge_pos)

        if abs(shift_amount) > 1e-6:
            esf = ndimage_shift(esf, -shift_amount, order=1, mode="nearest")

        lsf = np.diff(esf)
        lsf = SFRCalculator._apply_lsf_smoothing(lsf, method=lsf_smoothing_method)

        window = np.hanning(len(lsf))
        lsf_windowed = lsf * window

        if len(lsf_windowed) < 4:
            print(f"Warning: LSF too short ({len(lsf_windowed)} samples) for FFT analysis")
            return None, None, esf, lsf

        if np.sum(np.abs(lsf_windowed)) < EPSILON:
            print("Warning: LSF sum is too small for meaningful FFT")
            return None, None, esf, lsf

        n_fft = len(lsf_windowed)
        n_fft_padded = n_fft * supersampling_factor
        fft_res = np.abs(fftpack.fft(lsf_windowed, n=n_fft_padded))

        freqs = fftpack.fftfreq(n_fft_padded, d=1.0 / supersampling_factor)

        n_half = len(freqs) // 2
        sfr = fft_res[:n_half]
        frequencies = freqs[:n_half]

        dc_component = sfr[0]
        if dc_component > EPSILON:
            sfr = sfr / dc_component
        else:
            print(f"Warning: DC component too small ({dc_component}), using fallback normalization")
            sfr = sfr / EPSILON

        sfr = np.clip(sfr, 0, 1)

        frequencies = frequencies / supersampling_factor

        valid_idx = frequencies <= 0.5
        frequencies = frequencies[valid_idx]
        sfr = sfr[valid_idx]

        return frequencies, sfr, esf, lsf
