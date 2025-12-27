# -*- coding: utf-8 -*-
"""
Custom QLabel widget for image display with ROI selection support.

This module provides the ImageLabel class for displaying images with:
- Drag selection for ROI
- Click selection with fixed size
- Zoom with mouse wheel
- Panning with right-click or VIEW mode
- ROI markers display with SFR values
"""

import numpy as np
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QLabel


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

        # Check if ROI Manual Mode is active
        if (
            self.parent_window
            and hasattr(self.parent_window, "roi_manual_mode")
            and self.parent_window.roi_manual_mode
        ):
            if event.button() == Qt.LeftButton:
                click_pos = event.pos()

                # Get size from parent window
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
                roi_num = len(self.roi_markers) + 1
                roi_name = f"ROI_{roi_num}"
                self.roi_markers.append((x, y, w, h, roi_name, None))

                # Also update parent's roi_markers list
                if self.parent_window and hasattr(self.parent_window, "roi_markers"):
                    self.parent_window.roi_markers.append((x, y, w, h, roi_name, None))
                    # Enable ROI Save button since we now have markers
                    if hasattr(self.parent_window, "ui"):
                        self.parent_window.ui.btn_roi_map_save.setEnabled(True)

                # Update display
                self.update()

                # Show info
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
            # Mode 2: Click with user-defined size
            if event.button() == Qt.LeftButton:
                self.is_selecting = False
                self.selection_start = None
                self.selection_end = None

                click_pos = event.pos()

                size = 30
                if self.parent_window and hasattr(self.parent_window, "click_select_size"):
                    size = self.parent_window.click_select_size

                half_size = size // 2

                center_x = int(click_pos.x() / self.zoom_level)
                center_y = int(click_pos.y() / self.zoom_level)

                x = max(0, center_x - half_size)
                y = max(0, center_y - half_size)
                w = size
                h = size

                if x + w > self.image_w:
                    x = max(0, self.image_w - size)
                if y + h > self.image_h:
                    y = max(0, self.image_h - size)

                self.selection_rect = QRect(x, y, w, h)
                self.selection_info_text = f"Selected Area: {w}×{h} at ({x}, {y})"
                self.update()

                rect = QRect(x, y, w, h)
                if self.roi_callback:
                    self.roi_callback(rect)
        else:
            # Mode 1: Drag Select
            if event.button() == Qt.LeftButton and self.pixmap_original:
                self.selection_rect = None
                self.selection_start = event.pos()
                self.selection_end = event.pos()
                self.is_selecting = True
                self.update()

    def mouseMoveEvent(self, event):
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
        if (event.button() == Qt.RightButton or event.button() == Qt.LeftButton) and self.is_panning:
            self.is_panning = False
            self.pan_start_pos = None
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

        painter = QPainter(self)

        # Draw drag selection rectangle
        if self.is_selecting and self.selection_start and self.selection_end:
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
            rect = QRect(self.selection_start, self.selection_end).normalized()
            painter.drawRect(rect)

        # Draw click selection square
        if self.selection_rect and self.pixmap_scaled:
            scaled_rect = QRect(
                int(self.selection_rect.x() * self.zoom_level),
                int(self.selection_rect.y() * self.zoom_level),
                int(self.selection_rect.width() * self.zoom_level),
                int(self.selection_rect.height() * self.zoom_level),
            )

            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(scaled_rect)

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
            center_x = self.pixmap_scaled.width() // 2
            center_y = self.pixmap_scaled.height() // 2

            pen = QPen(QColor(0, 255, 0))
            pen.setWidth(1)
            painter.setPen(pen)

            painter.drawLine(center_x, 0, center_x, self.pixmap_scaled.height())
            painter.drawLine(0, center_y, self.pixmap_scaled.width(), center_y)

        # Draw SFR value at top-left corner of ROI
        if self.roi_sfr_value is not None and self.roi_position is not None:
            x, y, w, h = self.roi_position

            roi_top_left_x = int(x * self.zoom_level)
            roi_top_left_y = int((y + 5) * self.zoom_level)

            sfr_text = f"{self.roi_sfr_value*100:.2f}%"

            font = QFont("Arial", 12)
            font.setBold(True)
            painter.setFont(font)

            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(sfr_text)
            text_height = metrics.height()

            text_x = roi_top_left_x + 5
            text_y = roi_top_left_y + text_height + 5

            bg_rect = QRect(text_x - 3, text_y - text_height - 2, text_width + 6, text_height + 4)
            painter.fillRect(bg_rect, QColor(0, 0, 0, 200))
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawRect(bg_rect)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(text_x, text_y - 3, sfr_text)

        # Draw ROI markers
        if self.roi_markers:
            font = QFont("Arial", 10)
            font.setBold(True)
            painter.setFont(font)

            for marker in self.roi_markers:
                if len(marker) >= 6:
                    rx, ry, rw, rh, roi_name, sfr_value = marker[:6]
                else:
                    rx, ry, rw, rh, roi_name = marker[:5]
                    sfr_value = None

                disp_x = int(rx * self.zoom_level)
                disp_y = int(ry * self.zoom_level)
                disp_w = int(rw * self.zoom_level)
                disp_h = int(rh * self.zoom_level)

                pen = QPen(QColor(255, 0, 0))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawRect(disp_x, disp_y, disp_w, disp_h)

                metrics = painter.fontMetrics()
                line_height = metrics.height()

                if sfr_value is not None:
                    line1 = f"{roi_name}:"
                    line2 = f"{sfr_value*100:.2f}%"
                else:
                    line1 = roi_name
                    line2 = None

                line1_width = metrics.horizontalAdvance(line1)
                line2_width = metrics.horizontalAdvance(line2) if line2 else 0
                max_text_width = max(line1_width, line2_width)
                total_height = line_height * 2 if line2 else line_height

                text_x = disp_x + 3
                text_y = disp_y + line_height + 2

                bg_rect = QRect(text_x - 2, disp_y + 2, max_text_width + 6, total_height + 4)
                painter.fillRect(bg_rect, QColor(255, 0, 0, 200))

                painter.setPen(QColor(255, 255, 255))
                painter.drawText(text_x, text_y, line1)

                if line2:
                    painter.drawText(text_x, text_y + line_height, line2)

        painter.end()

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if self.pixmap_original is None:
            return

        delta = event.angleDelta().y()

        if delta > 0:
            self.zoom_level *= 1.1
        else:
            self.zoom_level /= 1.1

        self.zoom_level = max(0.5, min(self.zoom_level, 5.0))
        self.update_zoomed_image()

    def update_zoomed_image(self):
        """Update the displayed image with current zoom level"""
        if self.pixmap_original is None or self._updating_zoom:
            return

        self._updating_zoom = True

        try:
            new_width = int(self.pixmap_original.width() * self.zoom_level)
            new_height = int(self.pixmap_original.height() * self.zoom_level)

            self.pixmap_scaled = self.pixmap_original.scaled(
                new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(self.pixmap_scaled)
            self.setFixedSize(new_width, new_height)
        finally:
            self._updating_zoom = False

    def set_roi_sfr_display(self, sfr_value, roi_x, roi_y, roi_w, roi_h):
        """Set SFR value and ROI position to display at top-right corner"""
        self.roi_sfr_value = sfr_value
        self.roi_position = (roi_x, roi_y, roi_w, roi_h)
        self.update()

    def get_roi_rect(self):
        """Get ROI rectangle in image coordinates"""
        if not self.selection_start or not self.selection_end:
            return QRect()

        rect = QRect(self.selection_start, self.selection_end).normalized()

        if self.scroll_area and self.zoom_level != 1.0:
            scroll_x = self.scroll_area.horizontalScrollBar().value()
            scroll_y = self.scroll_area.verticalScrollBar().value()
            rect.translate(scroll_x, scroll_y)

        if self.zoom_level != 1.0:
            rect = QRect(
                int(rect.x() / self.zoom_level),
                int(rect.y() / self.zoom_level),
                int(rect.width() / self.zoom_level),
                int(rect.height() / self.zoom_level),
            )
        return rect
