"""
Tracking Review Dialog - Simplified responsive version
ממשק פשוט ורספונסיבי לסקירת מעקב
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QSlider, QListWidget, QListWidgetItem,
                             QWidget, QMessageBox, QTabWidget, QScrollArea,
                             QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QGuiApplication
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..tracking.tracker_manager import TrackerManager
from .bbox_editor import BboxEditor


# Simple color scheme
COLORS = {
    'bg': '#2b2b2b',
    'fg': '#ffffff',
    'accent': '#0078d4',
    'success': '#16c60c',
    'warning': '#f7630c',
    'error': '#e81123',
}

SIMPLE_STYLE = f"""
QDialog {{
    background-color: {COLORS['bg']};
    color: {COLORS['fg']};
}}
QLabel {{
    color: {COLORS['fg']};
}}
QPushButton {{
    background-color: {COLORS['accent']};
    color: {COLORS['fg']};
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-size: 12px;
}}
QPushButton:hover {{
    background-color: #1084d8;
}}
QPushButton#success {{
    background-color: {COLORS['success']};
}}
QPushButton#warning {{
    background-color: {COLORS['warning']};
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: #444;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {COLORS['accent']};
    width: 12px;
    margin: -4px 0;
    border-radius: 6px;
}}
"""


class SimpleConfidenceGraph(QWidget):
    """Simplified confidence graph"""
    frame_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracking_data = {}
        self.current_frame = 0
        self.setFixedHeight(80)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_data(self, tracking_data: Dict, player_id: int):
        self.tracking_data = tracking_data
        self.update()

    def set_current_frame(self, frame_idx: int):
        self.current_frame = frame_idx
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # Background
        painter.fillRect(0, 0, width, height, QColor(40, 40, 40))

        if not self.tracking_data:
            painter.end()
            return

        frames = sorted(self.tracking_data.keys())
        if not frames:
            painter.end()
            return

        min_frame = min(frames)
        max_frame = max(frames)
        frame_range = max_frame - min_frame

        if frame_range == 0:
            painter.end()
            return

        # Draw line
        points = []
        for frame_idx in frames:
            data = self.tracking_data[frame_idx]
            confidence = data.get('confidence', 0.0)
            x = int((frame_idx - min_frame) / frame_range * width)
            y = int((1 - confidence) * height)
            points.append((x, y, confidence))

        # Draw confidence line
        for i in range(len(points) - 1):
            x1, y1, conf1 = points[i]
            x2, y2, conf2 = points[i + 1]

            if conf1 < 0.5:
                color = QColor(255, 100, 100)
            elif conf1 < 0.7:
                color = QColor(255, 200, 0)
            else:
                color = QColor(0, 200, 255)

            painter.setPen(QPen(color, 2))
            painter.drawLine(x1, y1, x2, y2)

        # Current frame indicator
        if min_frame <= self.current_frame <= max_frame:
            x = int((self.current_frame - min_frame) / frame_range * width)
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawLine(x, 0, x, height)

        painter.end()

    def mousePressEvent(self, event):
        if not self.tracking_data:
            return

        frames = sorted(self.tracking_data.keys())
        if not frames:
            return

        min_frame = min(frames)
        max_frame = max(frames)
        frame_range = max_frame - min_frame

        if frame_range == 0:
            return

        x = event.pos().x()
        frame_idx = int(min_frame + (x / self.width()) * frame_range)
        frame_idx = max(min_frame, min(max_frame, frame_idx))

        self.frame_clicked.emit(frame_idx)


class TrackingReviewDialog(QDialog):
    """Simplified responsive dialog for tracking review"""

    def __init__(self, tracker_manager: TrackerManager,
                 tracking_data: Dict[int, Dict[int, Dict[str, any]]],
                 parent=None):
        super().__init__(parent)
        self.tracker_manager = tracker_manager
        self.tracking_data = tracking_data
        self.current_frame_idx = 0
        self.current_player_id = None

        self.setWindowTitle("Tracking Review")

        # Set window to fit screen
        screen = QGuiApplication.primaryScreen().geometry()
        window_width = min(1400, int(screen.width() * 0.95))
        window_height = min(900, int(screen.height() * 0.9))
        self.resize(window_width, window_height)

        self.setStyleSheet(SIMPLE_STYLE)

        self._init_ui()
        self._load_first_player()

    def _init_ui(self):
        """Initialize simplified UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Title
        title = QLabel("🎯 Tracking Review & Correction")
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 4px;")
        layout.addWidget(title)

        # Player selection (horizontal)
        player_layout = QHBoxLayout()
        player_layout.addWidget(QLabel("Player:"))

        self.player_list = QListWidget()
        self.player_list.setFixedHeight(60)
        self.player_list.currentItemChanged.connect(self._on_player_changed)

        for player_id, player in self.tracker_manager.players.items():
            item = QListWidgetItem(player.name)
            item.setData(Qt.ItemDataRole.UserRole, player_id)
            self.player_list.addItem(item)

        player_layout.addWidget(self.player_list, 1)

        # Stats
        self.stats_label = QLabel("Select player")
        self.stats_label.setStyleSheet("padding: 4px;")
        player_layout.addWidget(self.stats_label, 2)

        layout.addLayout(player_layout)

        # Confidence graph
        self.confidence_graph = SimpleConfidenceGraph()
        self.confidence_graph.frame_clicked.connect(self._jump_to_frame)
        layout.addWidget(self.confidence_graph)

        # Video preview (scrollable)
        video_scroll = QScrollArea()
        video_scroll.setWidgetResizable(True)
        video_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        video_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.bbox_editor = BboxEditor()
        # Set reasonable size that scales with video
        self.bbox_editor.setMinimumSize(640, 360)
        self.bbox_editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.bbox_editor.bbox_changed.connect(self._on_bbox_edited)

        video_scroll.setWidget(self.bbox_editor)
        layout.addWidget(video_scroll, 1)  # Takes most space

        # Frame controls
        controls_layout = QHBoxLayout()

        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.clicked.connect(self._prev_frame)
        controls_layout.addWidget(self.prev_btn)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(1, self.tracker_manager.total_frames - 1))
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        controls_layout.addWidget(self.frame_slider, 1)

        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self._next_frame)
        controls_layout.addWidget(self.next_btn)

        layout.addLayout(controls_layout)

        # Info and buttons
        bottom_layout = QHBoxLayout()

        self.frame_label = QLabel("Frame: 0/0")
        bottom_layout.addWidget(self.frame_label)

        self.conf_label = QLabel("Confidence: N/A")
        bottom_layout.addWidget(self.conf_label)

        bottom_layout.addStretch()

        fix_btn = QPushButton("🔧 Fix Frame")
        fix_btn.setObjectName("warning")
        fix_btn.clicked.connect(self._fix_frame)
        bottom_layout.addWidget(fix_btn)

        retrack_btn = QPushButton("🔄 Re-track")
        retrack_btn.setObjectName("success")
        retrack_btn.clicked.connect(self._retrack)
        bottom_layout.addWidget(retrack_btn)

        export_btn = QPushButton("✅ Continue to Export")
        export_btn.setObjectName("success")
        export_btn.clicked.connect(self.accept)
        bottom_layout.addWidget(export_btn)

        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(cancel_btn)

        layout.addLayout(bottom_layout)

        self.setLayout(layout)

    def _load_first_player(self):
        if self.player_list.count() > 0:
            self.player_list.setCurrentRow(0)

    def _on_player_changed(self, current, previous):
        if current is None:
            return

        player_id = current.data(Qt.ItemDataRole.UserRole)
        self.current_player_id = player_id

        if player_id in self.tracking_data:
            self.confidence_graph.set_data(self.tracking_data[player_id], player_id)

            frames = sorted(self.tracking_data[player_id].keys())
            if frames:
                self._jump_to_frame(frames[0])

        self._update_stats()

    def _update_stats(self):
        if self.current_player_id is None:
            return

        player = self.tracker_manager.get_player(self.current_player_id)
        player_data = self.tracking_data.get(self.current_player_id, {})

        if not player_data:
            self.stats_label.setText("No data")
            return

        total = len(player_data)
        lost = sum(1 for d in player_data.values() if d.get('bbox') is None)
        learning = sum(1 for d in player_data.values() if d.get('is_learning_frame', False))

        confidences = [d.get('confidence', 0.0) for d in player_data.values() if d.get('bbox') is not None]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        self.stats_label.setText(
            f"{player.name} | Frames: {total} | Lost: {lost} | "
            f"Learning: {learning} | Avg Conf: {avg_conf:.2f}"
        )

    def _jump_to_frame(self, frame_idx: int):
        self.current_frame_idx = frame_idx
        self.frame_slider.setValue(frame_idx)
        self._display_frame()

    def _on_frame_changed(self, value):
        self.current_frame_idx = value
        self._display_frame()

    def _prev_frame(self):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.frame_slider.setValue(self.current_frame_idx)

    def _next_frame(self):
        if self.current_frame_idx < self.tracker_manager.total_frames - 1:
            self.current_frame_idx += 1
            self.frame_slider.setValue(self.current_frame_idx)

    def _display_frame(self):
        if self.current_player_id is None:
            return

        frame = self.tracker_manager.get_frame(self.current_frame_idx)
        if frame is None:
            return

        player_data = self.tracking_data.get(self.current_player_id, {})
        current_data = player_data.get(self.current_frame_idx, {})
        bbox = current_data.get('bbox')
        confidence = current_data.get('confidence', 0.0)
        is_learning = current_data.get('is_learning_frame', False)

        self.bbox_editor.set_frame(frame, bbox)
        self.confidence_graph.set_current_frame(self.current_frame_idx)

        self.frame_label.setText(f"Frame: {self.current_frame_idx}/{self.tracker_manager.total_frames - 1}")

        if bbox is not None:
            learning_text = " 🟡" if is_learning else ""
            self.conf_label.setText(f"Confidence: {confidence:.2f}{learning_text}")
        else:
            self.conf_label.setText("Tracking Lost")

    def _fix_frame(self):
        QMessageBox.information(
            self,
            "Manual Correction",
            "How to edit:\n"
            "• Click and drag to create bbox\n"
            "• Drag corners to resize\n"
            "• Drag center to move\n"
            "• Press Delete to clear\n"
            "• Press ESC to cancel"
        )

    def _on_bbox_edited(self, bbox: Tuple[int, int, int, int]):
        if self.current_player_id is None:
            return

        self.tracker_manager.add_learning_frame_to_player(
            self.current_player_id,
            self.current_frame_idx,
            bbox
        )

        if self.current_player_id not in self.tracking_data:
            self.tracking_data[self.current_player_id] = {}

        self.tracking_data[self.current_player_id][self.current_frame_idx] = {
            'bbox': bbox,
            'confidence': 1.0,
            'is_learning_frame': True
        }

        self._display_frame()
        self._update_stats()
        self.confidence_graph.set_data(
            self.tracking_data[self.current_player_id],
            self.current_player_id
        )

        QMessageBox.information(
            self,
            "Learning Frame Added",
            f"Frame {self.current_frame_idx} marked as learning frame.\n"
            f"Click 'Re-track' to update tracking."
        )

    def _retrack(self):
        if self.current_player_id is None:
            return

        reply = QMessageBox.question(
            self,
            "Re-track",
            "Re-generate tracking with corrections?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        player_data = self.tracking_data.get(self.current_player_id, {})
        if not player_data:
            return

        frames = sorted(player_data.keys())
        start_frame = min(frames)
        end_frame = max(frames)

        try:
            new_data = self.tracker_manager.generate_tracking_data(
                start_frame=start_frame,
                end_frame=end_frame
            )

            if self.current_player_id in new_data:
                self.tracking_data[self.current_player_id] = new_data[self.current_player_id]

            self.confidence_graph.set_data(
                self.tracking_data[self.current_player_id],
                self.current_player_id
            )
            self._update_stats()
            self._display_frame()

            QMessageBox.information(self, "Success", "Tracking updated!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Re-track failed:\n{str(e)}")
