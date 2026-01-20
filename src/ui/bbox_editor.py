"""
Bbox Editor - Interactive bbox marking and editing widget
עורך Bbox - ווידג'ט אינטראקטיבי לסימון ועריכת bbox
"""

from PyQt6.QtWidgets import QLabel, QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QMouseEvent
import cv2
import numpy as np
from typing import Optional, Tuple, List


class BboxEditor(QLabel):
    """
    Interactive widget for marking and editing bounding boxes

    Features:
    - Click and drag to create new bbox
    - Drag bbox to move it
    - Resize bbox by dragging corners/edges
    - Visual feedback during editing

    Signals:
    - bbox_changed: Emitted when bbox is modified (x, y, w, h)
    """

    bbox_changed = pyqtSignal(tuple)  # (x, y, w, h) in original frame coordinates

    # Resize handle size (pixels)
    HANDLE_SIZE = 8

    # Resize modes
    RESIZE_NONE = 0
    RESIZE_TL = 1  # Top-Left
    RESIZE_TR = 2  # Top-Right
    RESIZE_BL = 3  # Bottom-Left
    RESIZE_BR = 4  # Bottom-Right
    RESIZE_T = 5   # Top edge
    RESIZE_B = 6   # Bottom edge
    RESIZE_L = 7   # Left edge
    RESIZE_R = 8   # Right edge
    MOVE = 9       # Move entire bbox

    def __init__(self, parent=None):
        super().__init__(parent)

        # Current frame
        self.current_frame = None
        self.frame_rgb = None
        self.scale_factor = 1.0
        self.display_offset = QPoint(0, 0)
        self.scaled_size = (0, 0)

        # Bbox in frame coordinates (x, y, w, h)
        self.bbox = None
        self.candidate_bboxes: List[Tuple[int, int, int, int, float]] = []
        self.hover_candidate_index: Optional[int] = None

        # Drawing state
        self.is_drawing = False
        self.draw_start = None
        self.draw_current = None

        # Editing state
        self.is_editing = False
        self.resize_mode = self.RESIZE_NONE
        self.edit_start_pos = None
        self.edit_start_bbox = None

        # Visual settings
        self.bbox_color = QColor(0, 255, 0)  # Green
        self.bbox_color_active = QColor(0, 255, 255)  # Cyan when editing
        self.handle_color = QColor(255, 255, 255)  # White handles
        self.line_width = 2

        # Widget settings
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)  # Track mouse for hover effects
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_frame(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None):
        """
        Set frame to display and optional initial bbox

        Args:
            frame: Frame to display (BGR format)
            bbox: Optional bbox (x, y, w, h) in frame coordinates
        """
        self.current_frame = frame.copy()
        self.frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.bbox = bbox

        self._update_display()

    def get_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Get current bbox in frame coordinates"""
        return self.bbox

    def clear_bbox(self):
        """Clear current bbox"""
        self.bbox = None
        self._update_display()

    def _update_display(self):
        """Update the displayed image with bbox overlay"""
        if self.frame_rgb is None:
            return

        # Convert to QImage
        h, w, ch = self.frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(self.frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Create pixmap and scale to fit widget
        pixmap = QPixmap.fromImage(qt_image)

        # Calculate scale factor
        widget_size = self.size()
        self.scale_factor = min(
            widget_size.width() / w,
            widget_size.height() / h
        )

        scaled_size = (int(w * self.scale_factor), int(h * self.scale_factor))
        self.scaled_size = scaled_size
        self.display_offset = QPoint(
            max(0, (widget_size.width() - scaled_size[0]) // 2),
            max(0, (widget_size.height() - scaled_size[1]) // 2)
        )
        scaled_pixmap = pixmap.scaled(
            scaled_size[0], scaled_size[1],
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # Draw bbox and handles on top
        if self.bbox or self.is_drawing or self.candidate_bboxes:
            painter = QPainter(scaled_pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Draw current bbox
            if self.bbox:
                self._draw_bbox(painter, self.bbox, self.is_editing)

            # Draw temporary bbox while drawing
            if self.is_drawing and self.draw_start and self.draw_current:
                temp_bbox = self._calculate_bbox_from_points(
                    self.draw_start, self.draw_current
                )
                self._draw_bbox(painter, temp_bbox, False, dashed=True)

            # Draw auto-detected candidate bboxes
            if self.candidate_bboxes:
                self._draw_candidate_bboxes(painter)

            painter.end()

        self.setPixmap(scaled_pixmap)

    def _draw_bbox(self, painter: QPainter, bbox: Tuple[int, int, int, int],
                   active: bool = False, dashed: bool = False):
        """Draw bbox with resize handles"""
        if bbox is None:
            return

        x, y, w, h = bbox

        # Scale to widget coordinates
        sx = int(x * self.scale_factor)
        sy = int(y * self.scale_factor)
        sw = int(w * self.scale_factor)
        sh = int(h * self.scale_factor)

        # Choose color
        color = self.bbox_color_active if active else self.bbox_color

        # Draw rectangle
        pen = QPen(color, self.line_width)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(sx, sy, sw, sh)

        # Draw resize handles (only if not drawing temporary bbox)
        if not dashed:
            self._draw_handles(painter, sx, sy, sw, sh)

    def _draw_handles(self, painter: QPainter, x: int, y: int, w: int, h: int):
        """Draw resize handles at corners and edges"""
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.setBrush(QBrush(self.handle_color))

        hs = self.HANDLE_SIZE

        # Corner handles
        handles = [
            (x - hs//2, y - hs//2),                    # Top-Left
            (x + w - hs//2, y - hs//2),                # Top-Right
            (x - hs//2, y + h - hs//2),                # Bottom-Left
            (x + w - hs//2, y + h - hs//2),            # Bottom-Right
            (x + w//2 - hs//2, y - hs//2),             # Top edge
            (x + w//2 - hs//2, y + h - hs//2),         # Bottom edge
            (x - hs//2, y + h//2 - hs//2),             # Left edge
            (x + w - hs//2, y + h//2 - hs//2),         # Right edge
        ]

        for hx, hy in handles:
            painter.drawRect(hx, hy, hs, hs)

    def _draw_candidate_bboxes(self, painter: QPainter):
        """Draw auto-detected candidate bboxes with confidence labels"""
        for idx, candidate in enumerate(self.candidate_bboxes):
            # Support tuples with or without confidence
            if len(candidate) >= 5:
                x, y, w, h, conf = candidate
            else:
                x, y, w, h = candidate
                conf = None

            sx = int(x * self.scale_factor)
            sy = int(y * self.scale_factor)
            sw = int(w * self.scale_factor)
            sh = int(h * self.scale_factor)

            color = QColor(255, 140, 0) if idx != self.hover_candidate_index else QColor(255, 200, 0)
            pen = QPen(color, 2 if idx != self.hover_candidate_index else 3)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(sx, sy, sw, sh)

            # Draw label above the bbox
            label = f"Auto #{idx + 1}"
            if conf is not None:
                label += f" ({conf:.0%})"

            fm = painter.fontMetrics()
            text_rect = fm.boundingRect(label)
            text_x = sx + max(0, (sw - text_rect.width()) // 2)
            text_y = sy - 8
            if text_y - text_rect.height() < 0:
                text_y = sy + sh + text_rect.height() + 6

            bg_rect = QRect(
                text_x - 4,
                text_y - text_rect.height(),
                text_rect.width() + 8,
                text_rect.height() + 4
            )
            painter.fillRect(bg_rect, QColor(0, 0, 0, 160))
            painter.drawText(text_x, text_y - 2, label)

    def _widget_to_frame_coords(self, point: QPoint) -> Tuple[int, int]:
        """Convert widget coordinates to frame coordinates"""
        if self.scale_factor == 0 or self.frame_rgb is None:
            return (0, 0)

        # Translate mouse position into pixmap space (account for centering offset)
        px = point.x() - self.display_offset.x()
        py = point.y() - self.display_offset.y()

        if self.scaled_size[0] > 0 and self.scaled_size[1] > 0:
            px = max(0, min(px, self.scaled_size[0] - 1))
            py = max(0, min(py, self.scaled_size[1] - 1))

        x = int(px / self.scale_factor)
        y = int(py / self.scale_factor)

        # Clamp to frame bounds
        if self.current_frame is not None:
            h, w = self.current_frame.shape[:2]
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))

        return (x, y)

    def _calculate_bbox_from_points(self, p1: QPoint, p2: QPoint) -> Tuple[int, int, int, int]:
        """Calculate bbox from two corner points"""
        x1, y1 = self._widget_to_frame_coords(p1)
        x2, y2 = self._widget_to_frame_coords(p2)

        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        return (x, y, w, h)

    def _get_resize_mode(self, pos: QPoint) -> int:
        """Determine resize mode based on mouse position"""
        if self.bbox is None:
            return self.RESIZE_NONE

        x, y, w, h = self.bbox

        # Scale to widget coordinates
        sx = int(x * self.scale_factor) + self.display_offset.x()
        sy = int(y * self.scale_factor) + self.display_offset.y()
        sw = int(w * self.scale_factor)
        sh = int(h * self.scale_factor)

        px = pos.x()
        py = pos.y()

        hs = self.HANDLE_SIZE
        margin = hs * 2  # Larger hit area

        # Check corners first (higher priority)
        if abs(px - sx) < margin and abs(py - sy) < margin:
            return self.RESIZE_TL
        if abs(px - (sx + sw)) < margin and abs(py - sy) < margin:
            return self.RESIZE_TR
        if abs(px - sx) < margin and abs(py - (sy + sh)) < margin:
            return self.RESIZE_BL
        if abs(px - (sx + sw)) < margin and abs(py - (sy + sh)) < margin:
            return self.RESIZE_BR

        # Check edges
        if abs(py - sy) < margin and sx < px < sx + sw:
            return self.RESIZE_T
        if abs(py - (sy + sh)) < margin and sx < px < sx + sw:
            return self.RESIZE_B
        if abs(px - sx) < margin and sy < py < sy + sh:
            return self.RESIZE_L
        if abs(px - (sx + sw)) < margin and sy < py < sy + sh:
            return self.RESIZE_R

        # Check if inside bbox (move mode)
        if sx < px < sx + sw and sy < py < sy + sh:
            return self.MOVE

        return self.RESIZE_NONE

    def _update_cursor(self, resize_mode: int):
        """Update cursor based on resize mode"""
        cursors = {
            self.RESIZE_NONE: Qt.CursorShape.CrossCursor,
            self.RESIZE_TL: Qt.CursorShape.SizeFDiagCursor,
            self.RESIZE_TR: Qt.CursorShape.SizeBDiagCursor,
            self.RESIZE_BL: Qt.CursorShape.SizeBDiagCursor,
            self.RESIZE_BR: Qt.CursorShape.SizeFDiagCursor,
            self.RESIZE_T: Qt.CursorShape.SizeVerCursor,
            self.RESIZE_B: Qt.CursorShape.SizeVerCursor,
            self.RESIZE_L: Qt.CursorShape.SizeHorCursor,
            self.RESIZE_R: Qt.CursorShape.SizeHorCursor,
            self.MOVE: Qt.CursorShape.SizeAllCursor,
        }
        self.setCursor(cursors.get(resize_mode, Qt.CursorShape.CrossCursor))

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press - start drawing or editing"""
        if event.button() != Qt.MouseButton.LeftButton:
            return

        pos = event.pos()

        # Check if clicking on existing bbox
        candidate_idx = self._get_candidate_index(pos)
        if candidate_idx is not None:
            chosen = self.candidate_bboxes[candidate_idx]
            self.hover_candidate_index = candidate_idx
            self.bbox = (chosen[0], chosen[1], chosen[2], chosen[3])
            self.is_drawing = False
            self.is_editing = False
            self.resize_mode = self.RESIZE_NONE
            self.bbox_changed.emit(self.bbox)
            self._update_display()
            return

        resize_mode = self._get_resize_mode(pos)

        if resize_mode != self.RESIZE_NONE:
            # Start editing existing bbox
            self.is_editing = True
            self.resize_mode = resize_mode
            self.edit_start_pos = pos
            self.edit_start_bbox = self.bbox
        else:
            # Start drawing new bbox
            self.is_drawing = True
            self.draw_start = pos
            self.draw_current = pos
            self.bbox = None  # Clear existing bbox

        self._update_display()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move - update drawing or editing"""
        pos = event.pos()

        if self.is_drawing:
            # Update temporary bbox while drawing
            self.draw_current = pos
            self._update_display()

        elif self.is_editing:
            # Update bbox based on resize mode
            self._update_bbox_from_mouse(pos)
            self._update_display()

        else:
            # Update cursor based on hover position
            candidate_idx = self._get_candidate_index(pos) if self.candidate_bboxes else None
            if candidate_idx is not None:
                if candidate_idx != self.hover_candidate_index:
                    self.hover_candidate_index = candidate_idx
                    self._update_display()
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                if self.hover_candidate_index is not None:
                    self.hover_candidate_index = None
                    self._update_display()
                resize_mode = self._get_resize_mode(pos)
                self._update_cursor(resize_mode)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release - finalize drawing or editing"""
        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self.is_drawing:
            # Finalize new bbox
            if self.draw_start and self.draw_current:
                self.bbox = self._calculate_bbox_from_points(
                    self.draw_start, self.draw_current
                )

                # Ensure minimum size
                if self.bbox[2] < 10 or self.bbox[3] < 10:
                    self.bbox = None  # Too small, discard
                else:
                    self.bbox_changed.emit(self.bbox)

            self.is_drawing = False
            self.draw_start = None
            self.draw_current = None

        elif self.is_editing:
            # Finalize editing
            if self.bbox:
                self.bbox_changed.emit(self.bbox)

            self.is_editing = False
            self.resize_mode = self.RESIZE_NONE
            self.edit_start_pos = None
            self.edit_start_bbox = None

        self._update_display()

    def _update_bbox_from_mouse(self, current_pos: QPoint):
        """Update bbox based on current mouse position and resize mode"""
        if not self.edit_start_bbox or not self.edit_start_pos:
            return

        # Calculate delta in frame coordinates
        start_x, start_y = self._widget_to_frame_coords(self.edit_start_pos)
        curr_x, curr_y = self._widget_to_frame_coords(current_pos)
        dx = curr_x - start_x
        dy = curr_y - start_y

        x, y, w, h = self.edit_start_bbox

        # Apply transformation based on resize mode
        if self.resize_mode == self.MOVE:
            x += dx
            y += dy

        elif self.resize_mode == self.RESIZE_TL:
            x += dx
            y += dy
            w -= dx
            h -= dy

        elif self.resize_mode == self.RESIZE_TR:
            y += dy
            w += dx
            h -= dy

        elif self.resize_mode == self.RESIZE_BL:
            x += dx
            w -= dx
            h += dy

        elif self.resize_mode == self.RESIZE_BR:
            w += dx
            h += dy

        elif self.resize_mode == self.RESIZE_T:
            y += dy
            h -= dy

        elif self.resize_mode == self.RESIZE_B:
            h += dy

        elif self.resize_mode == self.RESIZE_L:
            x += dx
            w -= dx

        elif self.resize_mode == self.RESIZE_R:
            w += dx

        # Ensure minimum size and valid coordinates
        if w < 10:
            w = 10
        if h < 10:
            h = 10

        # Clamp to frame bounds
        if self.current_frame is not None:
            frame_h, frame_w = self.current_frame.shape[:2]
            x = max(0, min(x, frame_w - w))
            y = max(0, min(y, frame_h - h))

        self.bbox = (x, y, w, h)

    def _get_candidate_index(self, pos: QPoint) -> Optional[int]:
        """Return index of candidate bbox under cursor (if any)"""
        if not self.candidate_bboxes or self.scale_factor == 0:
            return None

        px = pos.x()
        py = pos.y()

        for idx, candidate in enumerate(self.candidate_bboxes):
            x, y, w, h = candidate[:4]
            sx = int(x * self.scale_factor) + self.display_offset.x()
            sy = int(y * self.scale_factor) + self.display_offset.y()
            sw = int(w * self.scale_factor)
            sh = int(h * self.scale_factor)

            if sx <= px <= sx + sw and sy <= py <= sy + sh:
                return idx
        return None

    def set_candidate_bboxes(self, candidates: List[Tuple[int, int, int, int, float]]):
        """Show auto-detected bboxes for quick selection"""
        self.candidate_bboxes = candidates or []
        self.hover_candidate_index = None
        self._update_display()

    def clear_candidate_bboxes(self):
        """Hide auto-detected candidates"""
        self.candidate_bboxes = []
        self.hover_candidate_index = None
        self._update_display()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key.Key_Escape:
            # Cancel current operation
            if self.is_drawing:
                self.is_drawing = False
                self.draw_start = None
                self.draw_current = None
                self._update_display()
            elif self.is_editing:
                # Restore original bbox
                self.bbox = self.edit_start_bbox
                self.is_editing = False
                self.resize_mode = self.RESIZE_NONE
                self._update_display()

        elif event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            # Delete current bbox
            self.clear_bbox()

        super().keyPressEvent(event)
