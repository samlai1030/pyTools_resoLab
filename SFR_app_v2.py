import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QInputDialog, QComboBox,
                             QScrollArea)
from PyQt5.QtCore import Qt, QRect, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import fftpack


def read_raw_image(file_path, width=None, height=None, dtype=np.uint16):
    """
    Read a raw image file

    Parameters:
    - file_path: path to the raw file
    - width: image width (if known)
    - height: image height (if known)
    - dtype: data type (usually uint8, uint16, or float32)
    """
    try:
        # Method 1: Try reading as binary data
        with open(file_path, 'rb') as f:
            raw_data = f.read()

        # Convert to numpy array
        if dtype == np.uint16:
            img_array = np.frombuffer(raw_data, dtype=np.uint16)
        elif dtype == np.uint8:
            img_array = np.frombuffer(raw_data, dtype=np.uint8)
        else:
            img_array = np.frombuffer(raw_data, dtype=dtype)

        # If dimensions are known, reshape
        if width and height:
            img_array = img_array.reshape(height, width)
        else:
            # Try to guess square dimensions
            total_pixels = len(img_array)
            side = int(np.sqrt(total_pixels))
            if side * side == total_pixels:
                img_array = img_array.reshape(side, side)
            else:
                print(f"Cannot determine image dimensions. Total values: {total_pixels}")
                return img_array

        return img_array

    except Exception as e:
        print(f"Error reading raw file: {e}")
        return None


class SFRCalculator:
    """物理運算核心：處理邊緣檢測與 SFR 計算"""

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
        gray = roi_image if len(roi_image.shape) == 2 else cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
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
        v_edges = np.sum((angle > 80) & (angle < 100)) / angle.size * 100  # 垂直邊 ~90度
        h_edges = np.sum(((angle > 170) | (angle < 10))) / angle.size * 100  # 水平邊 ~0或180度

        # 判定邊緣類型
        edge_strength_ratio = mag_x / (mag_y + 1e-10)

        details = {
            'mag_x': mag_x,
            'mag_y': mag_y,
            'ratio_x_y': edge_strength_ratio,
            'v_edges_percent': v_edges,
            'h_edges_percent': h_edges,
            'mean_x': np.mean(np.abs(sobelx)),
            'mean_y': np.mean(np.abs(sobely))
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
        gray = roi_image if len(roi_image.shape) == 2 else cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        # 簡單判定：最大梯度強度需大於某個閾值
        if np.max(magnitude) < 50:
            return False, "Low Contrast / No Edge detected", "No Edge", 0

        # 檢測邊緣方向
        edge_type, confidence, _ = SFRCalculator.detect_edge_orientation(roi_image)

        return True, "Edge Detected", edge_type, confidence

    @staticmethod
    def calculate_sfr(roi_image, edge_type="V-Edge"):
        """
        計算 SFR (Spatial Frequency Response)。
        支持垂直邊(V-Edge)和水平邊(H-Edge)。

        Parameters:
        - roi_image: ROI 圖像
        - edge_type: "V-Edge" 或 "H-Edge"
        """
        # 轉為灰階並正規化
        img = roi_image.astype(np.float64)
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)

        # 1. 取得 Edge Profile (ESF - Edge Spread Function)
        if edge_type == "V-Edge":
            # 垂直邊：對每一列(column)取平均
            esf = np.mean(img, axis=0)  # 行平均
        else:  # H-Edge
            # 水平邊：對每一行(row)取平均
            esf = np.mean(img, axis=1)  # 列平均

        # 2. 計算 LSF (Line Spread Function): ESF 的微分
        lsf = np.diff(esf)

        # 套用 Hamming Window 減少頻譜洩漏
        window = np.hamming(len(lsf))
        lsf_windowed = lsf * window

        # 3. FFT 轉換
        if np.sum(lsf_windowed) == 0:
            return None, None

        fft_res = np.abs(fftpack.fft(lsf_windowed))

        # 4. 取前半部分頻譜並歸一化 (DC component = 1)
        freqs = fftpack.fftfreq(len(lsf), d=1.0)
        n_half = len(freqs) // 2

        sfr = fft_res[:n_half]
        sfr = sfr / (sfr[0] + 1e-10)  # Normalize
        frequencies = freqs[:n_half]

        return frequencies, sfr


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
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.pixmap_original:
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
        if self.pixmap_original is None:
            return

        # Calculate new size
        new_width = int(self.pixmap_original.width() * self.zoom_level)
        new_height = int(self.pixmap_original.height() * self.zoom_level)

        # Scale the pixmap maintaining aspect ratio
        self.pixmap_scaled = self.pixmap_original.scaled(
            new_width, new_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(self.pixmap_scaled)

        # Update size to enable/show scrollbars
        self.setFixedSize(new_width, new_height)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.is_selecting and self.selection_start and self.selection_end:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))

            # Draw rectangle directly at current mouse positions (already in zoomed space)
            rect = QRect(self.selection_start, self.selection_end).normalized()
            painter.drawRect(rect)

    def get_roi_rect(self):
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
                int(rect.height() / self.zoom_level)
            )
        return rect


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Physics Engineer: Raw SFR Analyzer")
        self.resize(1400, 900)

        # Data
        self.raw_data = None
        self.image_w = 1920  # 預設，實際應由使用者輸入
        self.image_h = 1080

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

        # Create image label
        self.image_label = ImageLabel(self)
        self.image_label.setStyleSheet("border: 2px solid #333; background: black;")
        self.image_label.setMinimumSize(500, 500)
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

        # Add ROI preview label below image
        self.roi_preview_label = QLabel("ROI Preview")
        self.roi_preview_label.setStyleSheet("border: 1px solid #ccc; background: black; min-height: 100px;")
        self.roi_preview_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.roi_preview_label)

        # Right Panel: SFR Plot & Info
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.info_label = QLabel("Status: Ready")
        self.info_label.setMinimumHeight(40)
        self.info_label.setStyleSheet("background: white; padding: 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 11px;")
        self.info_label.setWordWrap(True)

        # Matplotlib Figure - optimized size
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.figure.patch.set_facecolor('white')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(600, 500)
        self.canvas.setStyleSheet("background: white; border: 1px solid #ccc;")

        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("SFR / MTF Curve", fontsize=11, fontweight='bold')
        self.ax.set_xlabel("Spatial Frequency (cycles/pixel)", fontsize=10)
        self.ax.set_ylabel("Modulation Transfer Function", fontsize=10)
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()

        right_layout.addWidget(self.info_label)
        right_layout.addWidget(self.canvas, 1)

        # Ratios - adjusted for better balance
        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 1)


    def load_raw_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Raw File", "", "Raw Files (*.raw);;All Files (*)")
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
        dtype_choice, ok3 = QInputDialog.getItem(self, "Raw Config", "Data Type:", dtype_names, 1)
        if not ok3:
            return

        selected_dtype = dtype_options[dtype_choice]

        try:
            # Use the improved read_raw_image function
            self.raw_data = read_raw_image(fname, width=self.image_w, height=self.image_h, dtype=selected_dtype)

            if self.raw_data is not None:
                # Normalize to 8-bit for display if necessary
                if self.raw_data.dtype != np.uint8:
                    display_data = ((self.raw_data - self.raw_data.min()) / (self.raw_data.max() - self.raw_data.min() + 1e-10) * 255).astype(np.uint8)
                else:
                    display_data = self.raw_data

                self.display_image(display_data)
                self.info_label.setText(f"Loaded: {fname} ({self.image_w}x{self.image_h}, {dtype_choice})")
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
        q_img = QImage(disp_img.tobytes(), w, h, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)

        # Reset zoom level for new image
        self.image_label.zoom_level = 1.0

        # Store original pixmap
        self.image_label.pixmap_original = pixmap
        self.image_label.pixmap_scaled = pixmap

        # Set initial display size
        self.image_label.setPixmap(pixmap)
        self.image_label.setMinimumSize(500, 500)
        self.image_label.setMaximumSize(16777215, 16777215)

    def process_roi(self, rect):
        """處理使用者選取的區域"""
        if self.raw_data is None: return

        # 座標轉換
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

        # 邊界檢查
        if w < 5 or h < 5: return

        roi = self.raw_data[y:y + h, x:x + w]

        # Display ROI preview
        self.display_roi_preview(roi)

        # 4. Detect Slit Edge and Edge Orientation
        is_edge, msg, edge_type, confidence = SFRCalculator.validate_edge(roi)

        if is_edge:
            self.info_label.setText(f"Edge Detected: {edge_type} (Confidence: {confidence:.1f}%) - Calculating SFR...")
            # 5. Input to SFR Algo with edge type
            freqs, sfr_values = SFRCalculator.calculate_sfr(roi, edge_type=edge_type)

            # 6. Show Result
            if freqs is not None:
                self.plot_sfr(freqs, sfr_values, edge_type)
                mtf50_idx = np.argmin(np.abs(sfr_values - 0.5))
                mtf50_val = freqs[mtf50_idx] if mtf50_idx < len(freqs) else 0
                self.info_label.setText(
                    f"{edge_type} Edge (Conf: {confidence:.1f}%) | MTF50: {mtf50_val:.3f} cy/px | SFR Calculated")
            else:
                self.info_label.setText("Error in SFR Calculation")
        else:
            self.info_label.setText(f"Detection Failed: {msg}")
            self.ax.clear()
            self.ax.grid(True)
            self.canvas.draw()

    def display_roi_preview(self, roi_image):
        """Display preview of selected ROI"""
        if roi_image is None or roi_image.size == 0:
            return

        # Normalize to 8-bit for display
        if roi_image.dtype != np.uint8:
            preview_data = ((roi_image - roi_image.min()) / (roi_image.max() - roi_image.min() + 1e-10) * 255).astype(np.uint8)
        else:
            preview_data = roi_image

        # Create QPixmap from preview
        h, w = preview_data.shape
        bytes_per_line = w
        q_img = QImage(preview_data.tobytes(), w, h, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)

        # Scale to fit in left panel (max 400x300)
        max_w, max_h = 400, 300
        if pixmap.width() > max_w or pixmap.height() > max_h:
            pixmap = pixmap.scaledToWidth(max_w, Qt.SmoothTransformation)

        # Display in ROI preview label
        self.roi_preview_label.setPixmap(pixmap)

    def plot_sfr(self, x, y, edge_type="V-Edge"):
        self.ax.clear()
        self.ax.plot(x, y, 'b-', linewidth=2.5, label='MTF')
        self.ax.set_title(f"SFR Result - {edge_type}", fontsize=11, fontweight='bold')
        self.ax.set_xlabel("Frequency (cycles/pixel)", fontsize=10)
        self.ax.set_ylabel("MTF", fontsize=10)
        self.ax.set_ylim(0, 1.1)
        self.ax.set_xlim(0, 0.5)
        self.ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='MTF50')
        self.ax.legend(loc='upper right', fontsize=9)
        self.ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
