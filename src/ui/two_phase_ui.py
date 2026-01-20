"""
Two-Phase Tracking UI - Complete single-screen interface
ממשק דו-שלבי מלא - כל המערכת במסך אחד

Features:
- YOLO person detection with click-to-select
- Player list with checkboxes, names, and marker icons
- Tracking range control (start/end frame)
- Start/Stop tracking buttons
- Compact confidence graph
- Frame navigation
- Manual bbox correction
- Everything fits in one screen (no scrolling)
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider,
    QListWidget, QListWidgetItem, QWidget, QProgressBar, QCheckBox,
    QSpinBox, QDoubleSpinBox, QGroupBox, QFrame, QLineEdit, QMessageBox, QComboBox,
    QSizePolicy, QFileDialog, QGridLayout, QSpacerItem, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QCursor
import cv2
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from ..tracking.tracker_manager import TrackerManager
from ..tracking.person_detector import PersonDetector
from ..render.video_exporter import VideoExporter


# Simple dark theme - VS Code style
COLORS = {
    'bg_dark': '#1e1e1e',      # Dark background (like VS Code)
    'bg_medium': '#252526',    # Slightly lighter
    'bg_light': '#2d2d30',     # Input fields
    'accent': '#0e639c',       # Blue
    'success': '#16825d',      # Green
    'warning': '#c87d11',      # Orange
    'error': '#d13438',        # Red
    'text': '#ffffff',         # White text everywhere
    'text_muted': '#cccccc',   # Light gray
}


class CompactConfidenceGraph(QWidget):
    """Compact horizontal confidence graph"""

    frame_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracking_data = {}
        self.total_frames = 0
        self.current_frame = 0
        self.setMinimumHeight(52)
        self.setMaximumHeight(80)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

    def set_data(self, tracking_data: Dict, total_frames: int):
        """Set tracking data"""
        self.tracking_data = tracking_data
        self.total_frames = total_frames
        self.update()

    def set_current_frame(self, frame_idx: int):
        """Update current frame marker"""
        self.current_frame = frame_idx
        self.update()

    def paintEvent(self, event):
        """Draw confidence bars"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # Background
        painter.fillRect(0, 0, width, height, QColor(COLORS['bg_medium']))

        if not self.tracking_data or self.total_frames == 0:
            painter.end()
            return

        # Draw bars
        bar_width = max(1, width / self.total_frames)

        for frame_idx in range(self.total_frames):
            x = int(frame_idx * bar_width)

            frame_data = self.tracking_data.get(frame_idx, {})
            confidence = frame_data.get('confidence', 0.0)
            is_learning = frame_data.get('is_learning_frame', False)

            # Color based on confidence
            if is_learning:
                color = QColor(255, 215, 0)  # Gold - learning frame
            elif confidence >= 0.7:
                color = QColor(COLORS['success'])  # Green - good tracking
            elif confidence >= 0.6:
                color = QColor(COLORS['warning'])  # Orange - medium confidence
            elif confidence > 0:
                # RED for suspicious frames (confidence < 0.6)
                # These are likely wrong person detections!
                color = QColor(COLORS['error'])  # Red - SUSPICIOUS! Check this frame
            else:
                color = QColor(60, 60, 60)  # Dark gray - lost tracking

            # Draw bar
            bar_height = int(height * confidence) if confidence > 0 else 2
            painter.fillRect(x, height - bar_height, int(bar_width) + 1, bar_height, color)

        # Current frame indicator
        current_x = int(self.current_frame * bar_width)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawLine(current_x, 0, current_x, height)

        painter.end()

    def mousePressEvent(self, event):
        """Click to jump to frame"""
        if self.total_frames == 0:
            return

        x = event.pos().x()
        frame_idx = int((x / self.width()) * self.total_frames)
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))

        self.frame_clicked.emit(frame_idx)


class VideoPreviewWidget(QLabel):
    """Video preview with clickable detected people and manual bbox drawing"""

    person_clicked = pyqtSignal(int)  # Index of clicked person
    bbox_drawn = pyqtSignal(tuple)  # Manual bbox drawn (x, y, w, h)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 450)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"background-color: {COLORS['bg_medium']}; border: 2px solid {COLORS['accent']};")

        self.current_frame = None
        self.detected_people = []  # List of (x, y, w, h, confidence)
        self.player_bboxes = []  # List of (x, y, w, h, player_name, color)
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # Manual drawing mode
        self.manual_drawing_mode = False
        self.drawing = False
        self.start_point = None
        self.current_bbox = None

    def set_frame(self, frame: np.ndarray, detected_people: List = None, player_bboxes: List = None):
        """Set frame to display"""
        self.current_frame = frame.copy()
        self.detected_people = detected_people or []
        self.player_bboxes = player_bboxes or []
        self._update_display()

    def set_manual_drawing_mode(self, enabled: bool):
        """Enable/disable manual bbox drawing mode"""
        self.manual_drawing_mode = enabled
        if enabled:
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def _update_display(self):
        """Update displayed frame"""
        if self.current_frame is None:
            return

        # Draw on frame
        display_frame = self.current_frame.copy()

        # Draw detected people (green)
        for i, (x, y, w, h, conf) in enumerate(self.detected_people):
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            label = f"Person {i + 1} ({conf:.0%})"
            cv2.putText(display_frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw existing player bboxes
        for x, y, w, h, name, color in self.player_bboxes:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display_frame, name, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw current manual bbox (if drawing)
        if self.current_bbox:
            x, y, w, h = self.current_bbox
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)  # Yellow

        # Convert to QPixmap
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Scale to fit
        widget_size = self.size()
        scale_w = widget_size.width() / pixmap.width()
        scale_h = widget_size.height() / pixmap.height()
        scale = min(scale_w, scale_h)

        scaled_pixmap = pixmap.scaled(
            int(pixmap.width() * scale),
            int(pixmap.height() * scale),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.scale_factor = scale
        self.offset_x = (widget_size.width() - scaled_pixmap.width()) // 2
        self.offset_y = (widget_size.height() - scaled_pixmap.height()) // 2

        self.setPixmap(scaled_pixmap)

    def mousePressEvent(self, event):
        """Handle mouse press - either click on person or start drawing"""
        try:
            if self.current_frame is None:
                return

            # Prevent division by zero
            if self.scale_factor == 0:
                return

            # Convert click position to frame coordinates
            click_x = event.pos().x() - self.offset_x
            click_y = event.pos().y() - self.offset_y

            if click_x < 0 or click_y < 0:
                return

            frame_x = int(click_x / self.scale_factor)
            frame_y = int(click_y / self.scale_factor)

            # Clamp to frame bounds
            if self.current_frame is not None:
                h, w = self.current_frame.shape[:2]
                frame_x = max(0, min(frame_x, w - 1))
                frame_y = max(0, min(frame_y, h - 1))

            # Manual drawing mode
            if self.manual_drawing_mode:
                self.drawing = True
                self.start_point = (frame_x, frame_y)
                self.current_bbox = None
                return

            # Click on detected person mode
            if self.detected_people:
                # Check if clicked inside any detected person bbox
                for i, (x, y, w, h, conf) in enumerate(self.detected_people):
                    if x <= frame_x <= x + w and y <= frame_y <= y + h:
                        self.person_clicked.emit(i)
                        return
        except Exception as e:
            print(f"Error in mousePressEvent: {e}")
            import traceback
            traceback.print_exc()

    def mouseMoveEvent(self, event):
        """Handle mouse move during drawing"""
        if not self.drawing or not self.manual_drawing_mode:
            return

        # Convert to frame coordinates
        move_x = event.pos().x() - self.offset_x
        move_y = event.pos().y() - self.offset_y

        if move_x < 0 or move_y < 0:
            return

        frame_x = int(move_x / self.scale_factor)
        frame_y = int(move_y / self.scale_factor)

        # Calculate bbox
        x1, y1 = self.start_point
        x2, y2 = frame_x, frame_y

        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        self.current_bbox = (x, y, w, h)
        self._update_display()

    def mouseReleaseEvent(self, event):
        """Handle mouse release - finish drawing"""
        if not self.drawing or not self.manual_drawing_mode:
            return

        self.drawing = False

        if self.current_bbox:
            x, y, w, h = self.current_bbox
            # Only emit if bbox is large enough
            if w > 10 and h > 10:
                self.bbox_drawn.emit(self.current_bbox)
            self.current_bbox = None
            self._update_display()


class TrackingThread(QThread):
    """Background tracking thread"""

    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(dict)  # tracking_data
    error = pyqtSignal(str)

    def __init__(self, tracker_manager: TrackerManager, player_ids: List[str],
                 start_frame: int, end_frame: int):
        super().__init__()
        self.tracker_manager = tracker_manager
        self.player_ids = player_ids
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.should_stop = False

    def run(self):
        """Run tracking"""
        try:
            resume_start = self.tracker_manager.get_resume_start(self.start_frame)
            print(f"🔁 Resume tracking from frame {resume_start} (kept 0..{resume_start-1})")
            tracking_data = self.tracker_manager.generate_tracking_data(
                start_frame=resume_start,
                end_frame=self.end_frame,
                progress_callback=self._progress_callback
            )

            if not self.should_stop:
                self.finished.emit(tracking_data)
        except Exception as e:
            if not self.should_stop:
                self.error.emit(str(e))

    def _progress_callback(self, current: int, total: int):
        """Progress update"""
        if self.should_stop:
            raise InterruptedError("Stopped by user")
        self.progress.emit(current, total)

    def stop(self):
        """Stop tracking"""
        self.should_stop = True


class ExportThread(QThread):
    """Background export thread to keep UI responsive"""
    progress = pyqtSignal(int, str, int, int)  # percent, stage, current, total
    finished = pyqtSignal(bool)  # success

    def __init__(self, video_exporter: VideoExporter, video_path: str, tracking_data: dict,
                 output_path: str, start_frame: int, end_frame: int):
        super().__init__()
        self.video_exporter = video_exporter
        self.video_path = video_path
        self.tracking_data = tracking_data
        self.output_path = output_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.cancelled = False

    def run(self):
        def progress_cb(current, total, stage="render"):
            # Map to percent (expect total=100 when weighted)
            percent = int(current)
            self.progress.emit(percent, stage, current, total)

        success = self.video_exporter.export_tracked_video(
            self.video_path,
            self.tracking_data,
            self.output_path,
            progress_callback=progress_cb,
            tracking_start_frame=self.start_frame,
            tracking_end_frame=self.end_frame,
            should_cancel=lambda: self.cancelled
        )
        self.finished.emit(success)

    def cancel(self):
        self.cancelled = True


class PlayerListItemWidget(QWidget):
    """Player list item with checkbox"""

    def __init__(self, player_id: str, player_name: str, marker_style: str, parent=None):
        super().__init__(parent)
        self.player_id = player_id

        layout = QHBoxLayout()
        layout.setContentsMargins(4, 2, 4, 2)

        # Checkbox
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(True)
        layout.addWidget(self.checkbox)

        # Marker icon
        marker_icons = {
            'nba_iso_ring': '🏀',
            'floating_chevron': '〽️',
            'spotlight_modern': '💡',
            'crosshair': '🎯',
            'tactical_brackets': '🔲',
            'sonar_ripple': '🌊'
        }
        icon = marker_icons.get(marker_style, '⭕')

        # Name label
        label = QLabel(f"{icon} {player_name}")
        label.setStyleSheet("color: #ffffff; font-weight: 600;")
        layout.addWidget(label, stretch=1)

        self.setLayout(layout)


class TwoPhaseTrackingUI(QDialog):
    """Complete two-phase tracking UI - everything in one screen"""

    def __init__(self, tracker_manager: TrackerManager, parent=None):
        super().__init__(parent)
        self.tracker_manager = tracker_manager
        self.person_detector = PersonDetector()

        # State
        self.current_frame_idx = 0
        self.total_frames = tracker_manager.total_frames
        self.tracking_data = {}  # {player_id: {frame_idx: {bbox, confidence, ...}}}
        self.selected_player_id = None
        self.tracking_thread = None
        self.detected_people = []
        self.correction_mode = False  # when True, next manual bbox updates an existing player
        self.current_action_mode = "NONE"  # NONE | MANUAL_DRAW | CORRECTION | DETECT_PEOPLE
        self.video_exporter = VideoExporter(self.tracker_manager)
        self.resolved_frames = {}  # player_id -> set of frames corrected

        # Tracking range
        self.start_frame = 0
        self.end_frame = self.total_frames - 1

        self.setWindowTitle("Two-Phase Tracking - Phase 1: Review & Track")

        # Make window responsive - fit to screen
        from PyQt6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen().geometry()
        window_width = min(1400, int(screen.width() * 0.9))
        window_height = min(850, int(screen.height() * 0.85))
        self.resize(window_width, window_height)
        self.setMinimumSize(1200, 720)

        self._setup_ui()
        self._load_frame(0)

    def _setup_ui(self):
        """Setup complete UI"""
        main_layout = QHBoxLayout()
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # === LEFT SIDE: Video + Graph + Navigation ===
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Video preview
        self.video_preview = VideoPreviewWidget()
        self.video_preview.person_clicked.connect(self._on_person_clicked)
        left_layout.addWidget(self.video_preview)

        # Confidence graph
        graph_label = QLabel("📊 Confidence Timeline (click to jump):")
        graph_label.setStyleSheet("color: #ffffff; font-weight: 600; font-size: 13px;")
        left_layout.addWidget(graph_label)

        self.confidence_graph = CompactConfidenceGraph()
        self.confidence_graph.frame_clicked.connect(self._jump_to_frame)
        left_layout.addWidget(self.confidence_graph)

        # Frame navigation
        nav_layout = QHBoxLayout()

        self.frame_label = QLabel("Frame: 0/0")
        self.frame_label.setStyleSheet("color: #ffffff; font-weight: 600; letter-spacing: 0.2px;")
        nav_layout.addWidget(self.frame_label)

        nav_layout.addStretch()

        self.btn_first = QPushButton("⏮")
        self.btn_first.setFixedWidth(40)
        self.btn_first.clicked.connect(lambda: self._jump_to_frame(0))
        nav_layout.addWidget(self.btn_first)

        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedWidth(40)
        self.btn_prev.clicked.connect(self._prev_frame)
        nav_layout.addWidget(self.btn_prev)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.total_frames - 1)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        nav_layout.addWidget(self.frame_slider, stretch=1)

        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedWidth(40)
        self.btn_next.clicked.connect(self._next_frame)
        nav_layout.addWidget(self.btn_next)

        self.btn_last = QPushButton("⏭")
        self.btn_last.setFixedWidth(40)
        self.btn_last.clicked.connect(lambda: self._jump_to_frame(self.total_frames - 1))
        nav_layout.addWidget(self.btn_last)

        left_layout.addLayout(nav_layout)

        # Info label
        self.info_label = QLabel("Use 'Detect People' to find and add players →")
        self.info_label.setStyleSheet("color: #ffffff; font-weight: 600; font-size: 13px; padding: 5px;")
        left_layout.addWidget(self.info_label)

        left_widget.setLayout(left_layout)
        main_layout.addWidget(left_widget, stretch=2)

        # === RIGHT SIDE: Controls (scrollable) ===
        right_inner = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(12)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Player list
        player_group = self._create_player_group()
        player_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        right_layout.addWidget(player_group)

        # Tracking range
        range_group = self._create_range_group()
        range_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        right_layout.addWidget(range_group)

        # Actions
        actions_group = self._create_actions_group()
        right_layout.addWidget(actions_group)

        # Spacer to keep controls pinned to top
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        right_layout.addItem(spacer)

        # Load button - load existing tracking data
        self.btn_load = QPushButton("📂 Load Tracking Data")
        self.btn_load.setToolTip("Load previously saved tracking data from JSON file")
        self.btn_load.setProperty("variant", "accent")
        self.btn_load.setMinimumHeight(45)
        self.btn_load.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_load.setMinimumSize(QSize(0, 45))
        self.btn_load.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_load.clicked.connect(self._on_load_clicked)
        right_layout.addWidget(self.btn_load)

        # Export button - keep text but make it fit
        self.btn_export = QPushButton("✅ Export / Save")
        self.btn_export.setToolTip("Save tracking data and continue")
        self.btn_export.setProperty("variant", "accent-strong")
        self.btn_export.setMinimumHeight(56)
        self.btn_export.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_export.setMinimumSize(QSize(0, 45))
        self.btn_export.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_export.clicked.connect(self._on_export_clicked)
        self.btn_export.setEnabled(False)
        right_layout.addWidget(self.btn_export)

        right_inner.setLayout(right_layout)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setWidget(right_inner)
        right_scroll.setMinimumWidth(360)
        right_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        main_layout.addWidget(right_scroll, stretch=1)

        self.setLayout(main_layout)

        # Apply theme
        self._apply_theme()

    def _create_player_group(self) -> QGroupBox:
        """Create player list group"""
        group = QGroupBox("👥 Players")
        layout = QVBoxLayout()

        self.player_list = QListWidget()
        self.player_list.setMaximumHeight(200)
        self.player_list.itemClicked.connect(self._on_player_selected)
        layout.addWidget(self.player_list)

        # Buttons - compact with icons
        btn_layout = QHBoxLayout()
        self.btn_edit_player = QPushButton("✏️ Edit")
        self.btn_edit_player.setToolTip("Edit player name")
        self.btn_edit_player.setEnabled(False)
        self.btn_edit_player.clicked.connect(self._edit_player)
        self.btn_edit_player.setMinimumWidth(80)
        self.btn_edit_player.setMaximumWidth(120)
        self.btn_edit_player.setMinimumSize(QSize(140, 40))
        self.btn_edit_player.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        btn_layout.addWidget(self.btn_edit_player)

        self.btn_remove_player = QPushButton("🗑️ Remove")
        self.btn_remove_player.setToolTip("Remove player")
        self.btn_remove_player.setEnabled(False)
        self.btn_remove_player.clicked.connect(self._remove_player)
        self.btn_remove_player.setMinimumWidth(80)
        self.btn_remove_player.setMaximumWidth(120)
        self.btn_remove_player.setMinimumSize(QSize(140, 40))
        self.btn_remove_player.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        btn_layout.addWidget(self.btn_remove_player)

        layout.addLayout(btn_layout)
        group.setLayout(layout)
        return group

    def _create_range_group(self) -> QGroupBox:
        """Create tracking range group"""
        group = QGroupBox("📹 Tracking Range")
        layout = QVBoxLayout()

        # Start frame
        start_layout = QHBoxLayout()
        start_label = QLabel("Start:")
        start_label.setStyleSheet("color: #ffffff; font-size: 12px; font-weight: 600;")
        start_layout.addWidget(start_label)
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setMinimum(0)
        self.start_frame_spin.setMaximum(self.total_frames - 1)
        self.start_frame_spin.setValue(0)
        self.start_frame_spin.valueChanged.connect(self._on_range_changed)
        start_layout.addWidget(self.start_frame_spin, stretch=1)
        layout.addLayout(start_layout)

        self.btn_set_start = QPushButton("⬅️ Set Start")
        self.btn_set_start.setToolTip("Set current frame as tracking start")
        self.btn_set_start.setProperty("variant", "accent")
        self.btn_set_start.setMinimumHeight(42)
        self.btn_set_start.setMinimumSize(QSize(140, 42))
        self.btn_set_start.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.btn_set_start.clicked.connect(self._set_current_as_start)
        layout.addWidget(self.btn_set_start)

        # End frame
        end_layout = QHBoxLayout()
        end_label = QLabel("End:")
        end_label.setStyleSheet("color: #ffffff; font-size: 12px; font-weight: 600;")
        end_layout.addWidget(end_label)
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setMinimum(0)
        self.end_frame_spin.setMaximum(self.total_frames - 1)
        self.end_frame_spin.setValue(self.total_frames - 1)
        self.end_frame_spin.valueChanged.connect(self._on_range_changed)
        end_layout.addWidget(self.end_frame_spin, stretch=1)
        layout.addLayout(end_layout)

        self.btn_set_end = QPushButton("➡️ Set End")
        self.btn_set_end.setToolTip("Set current frame as tracking end")
        self.btn_set_end.setProperty("variant", "accent")
        self.btn_set_end.setMinimumHeight(42)
        self.btn_set_end.setMinimumSize(QSize(140, 42))
        self.btn_set_end.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.btn_set_end.clicked.connect(self._set_current_as_end)
        layout.addWidget(self.btn_set_end)

        group.setLayout(layout)
        return group

    DEBUG_ACTIONS_LAYOUT = False

    def _create_actions_group(self) -> QGroupBox:
        """Create actions group"""
        group = QGroupBox("⚙️ Actions")
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        def _std_button(btn: QPushButton, height: int = 44):
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.setMinimumHeight(height)
            btn.setMinimumSize(QSize(0, height))
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return btn

        # Detect + Manual row
        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        self.btn_detect = _std_button(QPushButton("🔍 Detect People"))
        self.btn_detect.setToolTip("Detect People (YOLO)")
        self.btn_detect.setProperty("variant", "ghost")
        self.btn_detect.clicked.connect(self._detect_people)
        top_row.addWidget(self.btn_detect)

        self.btn_manual_draw = _std_button(QPushButton("✏️ Manual Draw"))
        self.btn_manual_draw.setToolTip("Draw Bbox Manually")
        self.btn_manual_draw.setCheckable(True)
        self.btn_manual_draw.setProperty("variant", "ghost")
        self.btn_manual_draw.clicked.connect(self._toggle_manual_draw)
        top_row.addWidget(self.btn_manual_draw)
        layout.addLayout(top_row)

        # Correction - two buttons in a row
        correction_row = QHBoxLayout()
        correction_row.setSpacing(8)

        self.btn_add_correction = _std_button(QPushButton("✏️ Draw Correction"))
        self.btn_add_correction.setToolTip("Draw a correction bbox manually for the selected player at this frame")
        self.btn_add_correction.setProperty("variant", "ghost")
        self.btn_add_correction.clicked.connect(self._start_correction_mode)
        correction_row.addWidget(self.btn_add_correction)

        self.btn_detect_correction = _std_button(QPushButton("🔍 Detect for Correction"))
        self.btn_detect_correction.setToolTip("Use YOLO to detect people, then click to correct the selected player")
        self.btn_detect_correction.setProperty("variant", "ghost")
        self.btn_detect_correction.clicked.connect(self._start_detect_correction_mode)
        correction_row.addWidget(self.btn_detect_correction)

        layout.addLayout(correction_row)

        # Start/Stop row
        start_stop_row = QHBoxLayout()
        start_stop_row.setSpacing(8)
        self.btn_start_tracking = _std_button(QPushButton("▶️ Start Tracking"), height=48)
        self.btn_start_tracking.setProperty("variant", "positive")
        self.btn_start_tracking.clicked.connect(self._start_tracking)
        start_stop_row.addWidget(self.btn_start_tracking)

        self.btn_stop_tracking = _std_button(QPushButton("⏸️ Stop"), height=48)
        self.btn_stop_tracking.setProperty("variant", "danger")
        self.btn_stop_tracking.clicked.connect(self._stop_tracking)
        self.btn_stop_tracking.setEnabled(False)
        start_stop_row.addWidget(self.btn_stop_tracking)
        layout.addLayout(start_stop_row)

        # Mode & thresholds
        self.tracking_mode_combo = QComboBox()
        self.tracking_mode_combo.addItems(["Legacy (CSRT only)", "Hybrid (CSRT + YOLO) ⭐"])
        self.tracking_mode_combo.setCurrentIndex(1 if self.tracker_manager.tracking_config.get("mode") == "hybrid" else 0)
        self.tracking_mode_combo.currentIndexChanged.connect(self._on_tracking_config_changed)
        self.tracking_mode_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.tracking_mode_combo)

        thresholds_row1 = QHBoxLayout()
        thresholds_row1.setSpacing(8)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setDecimals(2)
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(self.tracker_manager.tracking_config.get("iou_min", 0.15))
        self.iou_spin.setPrefix("IoU≥ ")
        self.iou_spin.valueChanged.connect(self._on_tracking_config_changed)
        self.iou_spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        thresholds_row1.addWidget(self.iou_spin)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setDecimals(2)
        self.scale_spin.setRange(0.0, 1.5)
        self.scale_spin.setSingleStep(0.05)
        self.scale_spin.setValue(self.tracker_manager.tracking_config.get("scale_change_max", 0.35))
        self.scale_spin.setPrefix("ScaleΔ ")
        self.scale_spin.valueChanged.connect(self._on_tracking_config_changed)
        self.scale_spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        thresholds_row1.addWidget(self.scale_spin)
        layout.addLayout(thresholds_row1)

        thresholds_row2 = QHBoxLayout()
        thresholds_row2.setSpacing(8)
        self.center_spin = QSpinBox()
        self.center_spin.setRange(10, 400)
        self.center_spin.setSingleStep(5)
        self.center_spin.setValue(int(self.tracker_manager.tracking_config.get("center_jump_px", 80)))
        self.center_spin.setPrefix("Jump ")
        self.center_spin.valueChanged.connect(self._on_tracking_config_changed)
        self.center_spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        thresholds_row2.addWidget(self.center_spin)

        self.reacquire_spin = QSpinBox()
        self.reacquire_spin.setRange(1, 30)
        self.reacquire_spin.setValue(int(self.tracker_manager.tracking_config.get("reacquire_interval", 5)))
        self.reacquire_spin.setPrefix("YOLO/ ")
        self.reacquire_spin.valueChanged.connect(self._on_tracking_config_changed)
        self.reacquire_spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        thresholds_row2.addWidget(self.reacquire_spin)
        layout.addLayout(thresholds_row2)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.progress_bar)

        # Re-track
        self.btn_retrack = _std_button(QPushButton("🔄 Re-track"), height=45)
        self.btn_retrack.clicked.connect(self._retrack)
        self.btn_retrack.setEnabled(False)
        layout.addWidget(self.btn_retrack)

        # Export progress + cancel
        progress_row = QHBoxLayout()
        progress_row.setSpacing(8)
        self.export_progress = QProgressBar()
        self.export_progress.setVisible(False)
        self.export_progress.setTextVisible(True)
        self.export_progress.setFormat("Exporting… %p%")
        progress_row.addWidget(self.export_progress, 1)

        self.btn_cancel_export = _std_button(QPushButton("❌ Cancel Export"), height=40)
        self.btn_cancel_export.setProperty("variant", "danger")
        self.btn_cancel_export.setEnabled(False)
        self.btn_cancel_export.clicked.connect(self._cancel_export)
        progress_row.addWidget(self.btn_cancel_export)
        layout.addLayout(progress_row)

        layout.addStretch()
        group.setLayout(layout)

        if self.DEBUG_ACTIONS_LAYOUT:
            print(f"[ActionsLayout] layout type={type(layout).__name__}")
            buttons = [
                self.btn_detect, self.btn_manual_draw, self.btn_add_correction,
                self.btn_detect_correction, self.btn_start_tracking, self.btn_stop_tracking, self.btn_retrack
            ]
            for btn in buttons:
                g = btn.geometry()
                print(f"[ActionsLayout] {btn.text()}: geom=({g.x()},{g.y()},{g.width()},{g.height()}) "
                      f"sizeHint={btn.sizeHint()} minHint={btn.minimumSizeHint()}")

        return group

    def _apply_theme(self):
        """Simple clean dark theme - everything dark with white text"""
        stylesheet = f"""
            * {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text']};
            }}
            QDialog {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text']};
            }}
            QLabel {{
                color: {COLORS['text']};
                background-color: transparent;
                font-size: 13px;
            }}
            QPushButton {{
                background-color: #273141;
                color: #ffffff;
                border: 1px solid #344155;
                border-radius: 10px;
                padding: 10px 14px;
                font-weight: 600;
                text-align: left;
                letter-spacing: 0.1px;
            }}
            QPushButton:hover {{
                background-color: #303b4f;
                color: #ffffff;
            }}
            QPushButton:disabled {{
                background-color: #1b2230;
                color: #a5b3c5;
                border-color: #1f2735;
            }}
            QPushButton[variant="positive"] {{
                background-color: {COLORS['success']};
                color: #ffffff;
                border: none;
                text-align: center;
                font-weight: 700;
            }}
            QPushButton[variant="positive"]:hover {{
                background-color: #1dd876;
                color: #ffffff;
            }}
            QPushButton[variant="positive"]:disabled {{
                background-color: #1f3a2d;
                color: #8fc7a8;
            }}
            QPushButton[variant="danger"] {{
                background-color: {COLORS['error']};
                color: #ffffff;
                border: none;
                text-align: center;
                font-weight: 700;
            }}
            QPushButton[variant="danger"]:hover {{
                background-color: #ff7099;
                color: #ffffff;
            }}
            QPushButton[variant="danger"]:disabled {{
                background-color: #3a2230;
                color: #d5a3b8;
            }}
            QPushButton[variant="accent-strong"] {{
                background: linear-gradient(90deg, #2d8cff, #4c9dff);
                color: #ffffff;
                border: none;
                font-size: 15px;
                font-weight: 700;
                text-align: center;
            }}
            QPushButton[variant="accent-strong"]:disabled {{
                background-color: #2a3545;
                color: #b5c5d8;
                border: none;
                font-size: 15px;
                font-weight: 700;
                text-align: center;
            }}
            QPushButton[variant="accent"] {{
                background: linear-gradient(90deg, #2d8cff, #4c9dff);
                color: #ffffff;
                border: none;
                font-weight: 700;
                text-align: center;
            }}
            QPushButton[variant="accent"]:hover {{
                background: linear-gradient(90deg, #3c97ff, #5ca6ff);
                color: #ffffff;
            }}
            QPushButton[variant="accent"]:disabled {{
                background-color: #2a3545;
                color: #b5c5d8;
                border: none;
            }}
            QPushButton[variant="ghost"] {{
                background-color: #1c2432;
                color: #ffffff;
                border: 1px solid #2f3a4d;
                text-align: left;
            }}
            QPushButton[variant="ghost"]:hover {{
                background-color: #263248;
                color: #ffffff;
            }}
            QPushButton[variant="ghost"]:disabled {{
                background-color: #161c27;
                color: #7a8594;
                border-color: #1f2635;
            }}
            QGroupBox {{
                background-color: {COLORS['bg_dark']};
                border: 1px solid #3e3e42;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                padding-left: 8px;
                color: {COLORS['text']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {COLORS['text']};
                background-color: {COLORS['bg_dark']};
            }}
            QListWidget {{
                background-color: {COLORS['bg_light']};
                border: 1px solid #2b3344;
                border-radius: 8px;
                color: #ffffff;
                padding: 4px;
            }}
            QLineEdit {{
                background-color: {COLORS['bg_light']};
                border: 1px solid #3e3e42;
                border-radius: 6px;
                color: {COLORS['text']};
                padding: 8px;
                selection-background-color: {COLORS['accent']};
                selection-color: #ffffff;
            }}
            QSpinBox, QDoubleSpinBox {{
                background-color: {COLORS['bg_light']};
                border: 1px solid #3e3e42;
                border-radius: 6px;
                color: {COLORS['text']};
                padding: 6px;
            }}
            QComboBox {{
                background-color: {COLORS['bg_light']};
                border: 1px solid #2b3344;
                border-radius: 8px;
                color: #ffffff;
                padding: 6px 10px;
                font-weight: 600;
                min-height: 30px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #ffffff;
                width: 0;
                height: 0;
                margin-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {COLORS['bg_light']};
                color: #ffffff;
                selection-background-color: #2d8cff;
                selection-color: #ffffff;
                border: 1px solid #2b3344;
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                color: #ffffff;
                padding: 5px;
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: #2d8cff;
                color: #ffffff;
            }}
            QSlider::groove:horizontal {{
                background-color: #2a303f;
                height: 8px;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background-color: {COLORS['accent']};
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
                border: 1px solid #8abaff;
            }}
            QProgressBar {{
                background-color: {COLORS['bg_light']};
                border: 1px solid #2b3344;
                border-radius: 8px;
                text-align: center;
                color: #ffffff;
                height: 26px;
                font-weight: 700;
            }}
            QProgressBar::chunk {{
                background: linear-gradient(90deg, #1ac26b, #36d981);
                border-radius: 7px;
            }}
            QListWidget::item:selected {{
                background-color: #23304a;
            }}
        """
        self.setStyleSheet(stylesheet)

    def _update_player_list(self):
        """Update player list UI"""
        # Preserve checkbox states and current selection
        checked_players = set()
        for i in range(self.player_list.count()):
            item = self.player_list.item(i)
            widget = self.player_list.itemWidget(item)
            if widget and widget.checkbox.isChecked():
                checked_players.add(widget.player_id)

        previous_selection = self.selected_player_id
        self.player_list.clear()

        first_item = None

        for player_id, player in self.tracker_manager.players.items():
            item = QListWidgetItem(self.player_list)
            widget = PlayerListItemWidget(player_id, player.name, player.marker_style)
            item.setSizeHint(widget.sizeHint())
            self.player_list.addItem(item)
            self.player_list.setItemWidget(item, widget)

            should_check = player_id in checked_players or not checked_players or player_id == self.selected_player_id
            if should_check:
                widget.checkbox.setChecked(True)

            if first_item is None:
                first_item = item

            # Restore previous selection if possible
            if previous_selection and player_id == previous_selection:
                self.player_list.setCurrentItem(item)
                self._on_player_selected(item)

        # If nothing selected yet but we have players, select the first one so the graph shows data
        if self.player_list.count() > 0 and self.selected_player_id is None and first_item is not None:
            self.player_list.setCurrentItem(first_item)
            self._on_player_selected(first_item)
    def _load_frame(self, frame_idx: int):
        """Load and display frame"""
        self.current_frame_idx = frame_idx

        frame = self.tracker_manager.get_frame(frame_idx)
        if frame is None:
            return

        # Collect player bboxes for this frame
        player_bboxes = []
        for player_id, player_data in self.tracking_data.items():
            if frame_idx in player_data:
                bbox_data = player_data[frame_idx]
                bbox = bbox_data.get('bbox')
                if bbox:
                    player = self.tracker_manager.players.get(player_id)
                    if player:
                        color = (0, 255, 0) if player_id == self.selected_player_id else (150, 150, 150)
                        player_bboxes.append((*bbox, player.name, color))

        # Update video preview
        self.video_preview.set_frame(frame, self.detected_people, player_bboxes)

        # Update UI
        self.frame_label.setText(f"Frame: {frame_idx}/{self.total_frames - 1}")
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_idx)
        self.frame_slider.blockSignals(False)

        # Update confidence graph
        if self.selected_player_id and self.selected_player_id in self.tracking_data:
            self.confidence_graph.set_current_frame(frame_idx)

        # Update info
        if self.selected_player_id and self.selected_player_id in self.tracking_data:
            player_data = self.tracking_data[self.selected_player_id].get(frame_idx, {})
            conf = player_data.get('confidence', 0.0)
            quality = "Good" if conf >= 0.7 else "Medium" if conf >= 0.4 else "Poor" if conf > 0 else "Lost"
            player = self.tracker_manager.players.get(self.selected_player_id)
            player_name = player.name if player else "Unknown"
            self.info_label.setText(f"Selected: {player_name} | Confidence: {conf:.2f} | Quality: {quality}")
        else:
            self.info_label.setText("Use 'Detect People' to find and add players →")

    def _on_slider_changed(self, value):
        """Slider changed"""
        self._load_frame(value)

    def _on_tracking_config_changed(self, *_):
        """Update tracker config from UI controls"""
        mode = "hybrid" if self.tracking_mode_combo.currentIndex() == 1 else "legacy"
        self.tracker_manager.update_tracking_config(
            mode=mode,
            iou_min=float(self.iou_spin.value()),
            scale_change_max=float(self.scale_spin.value()),
            center_jump_px=float(self.center_spin.value()),
            reacquire_interval=int(self.reacquire_spin.value())
        )

    def _prev_frame(self):
        """Previous frame"""
        if self.current_frame_idx > 0:
            self._load_frame(self.current_frame_idx - 1)

    def _next_frame(self):
        """Next frame"""
        if self.current_frame_idx < self.total_frames - 1:
            self._load_frame(self.current_frame_idx + 1)

    def _jump_to_frame(self, frame_idx: int):
        """Jump to frame"""
        self._load_frame(frame_idx)

    def _on_player_selected(self, item):
        """Player selected"""
        widget = self.player_list.itemWidget(item)
        if widget:
            self.selected_player_id = widget.player_id

            # Update confidence graph
            if self.selected_player_id in self.tracking_data:
                self.confidence_graph.set_data(
                    self.tracking_data[self.selected_player_id],
                    self.total_frames
                )

            # Enable buttons
            self.btn_edit_player.setEnabled(True)
            self.btn_remove_player.setEnabled(True)

            # Refresh display
            self._load_frame(self.current_frame_idx)

    def _edit_player(self):
        """Edit player name"""
        if not self.selected_player_id:
            return

        player = self.tracker_manager.players.get(self.selected_player_id)
        if not player:
            return

        from PyQt6.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(
            self, "Edit Player Name",
            "Enter new name:",
            text=player.name
        )

        if ok and new_name.strip():
            player.name = new_name.strip()
            self._update_player_list()
            self._load_frame(self.current_frame_idx)

    def _remove_player(self):
        """Remove player"""
        if not self.selected_player_id:
            return

        player = self.tracker_manager.players.get(self.selected_player_id)
        player_name = player.name if player else "this player"

        reply = QMessageBox.question(
            self, "Remove Player",
            f"Remove {player_name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.tracker_manager.remove_player(self.selected_player_id)
            if self.selected_player_id in self.tracking_data:
                del self.tracking_data[self.selected_player_id]

            self.selected_player_id = None
            self._update_player_list()
            self._load_frame(self.current_frame_idx)

    def _on_range_changed(self):
        """Range changed"""
        self.start_frame = self.start_frame_spin.value()
        self.end_frame = self.end_frame_spin.value()

        if self.start_frame > self.end_frame:
            self.end_frame_spin.setValue(self.start_frame)

    def _set_current_as_start(self):
        """Set current frame as start"""
        self.start_frame_spin.setValue(self.current_frame_idx)

    def _set_current_as_end(self):
        """Set current frame as end"""
        self.end_frame_spin.setValue(self.current_frame_idx)

    def _detect_people(self):
        """Detect people using YOLO"""
        if not self.person_detector.is_available():
            QMessageBox.warning(
                self, "YOLO Not Available",
                "YOLO model not loaded.\n\nRun: pip install ultralytics"
            )
            return

        frame = self.tracker_manager.get_frame(self.current_frame_idx)
        if frame is None:
            return

        self.current_action_mode = "DETECT_PEOPLE"
        self.detected_people = self.person_detector.detect_people(frame, confidence_threshold=0.25)

        if not self.detected_people:
            QMessageBox.information(
                self, "No People Detected",
                "No people found in this frame.\n\nTry a different frame."
            )
            return

        # Refresh display
        self._load_frame(self.current_frame_idx)

        # User can see the detected people on screen - no need for message box

    def _toggle_manual_draw(self, checked: bool, show_message: bool = True):
        """Toggle manual bbox drawing mode"""
        # If correction mode is requested, keep that flag; otherwise reset
        if not checked:
            self.correction_mode = False
            if self.current_action_mode == "MANUAL_DRAW":
                self.current_action_mode = "NONE"

        self.video_preview.set_manual_drawing_mode(checked)

        if checked:
            self.current_action_mode = "MANUAL_DRAW"
            # Disable detect button
            self.btn_detect.setEnabled(False)
            # Clear detected people
            self.detected_people = []
            self._load_frame(self.current_frame_idx)
            # Disconnect first to avoid duplicate connections
            try:
                self.video_preview.bbox_drawn.disconnect(self._on_bbox_drawn)
            except:
                pass
            # Connect bbox_drawn signal
            self.video_preview.bbox_drawn.connect(self._on_bbox_drawn)

            if show_message:
                QMessageBox.information(
                    self, "Manual Drawing Mode",
                    "Click and drag on the video to draw a bounding box.\n\n"
                    "The bbox will be added as a new player."
                )
        else:
            # Re-enable detect button
            self.btn_detect.setEnabled(True)
            # Disconnect signal
            try:
                self.video_preview.bbox_drawn.disconnect(self._on_bbox_drawn)
            except:
                pass

    def _on_bbox_drawn(self, bbox: Tuple[int, int, int, int]):
        """Handle manual bbox drawn"""
        # Turn off drawing mode
        self.btn_manual_draw.setChecked(False)
        self.video_preview.set_manual_drawing_mode(False)
        self.btn_detect.setEnabled(True)

        # If we are in correction mode, add learning frame to selected player
        if self.correction_mode:
            self.correction_mode = False
            self.current_action_mode = "NONE"
            if not self.selected_player_id:
                return

            added = self.tracker_manager.add_learning_frame_to_player(
                self.selected_player_id, self.current_frame_idx, bbox, bbox
            )
            if added:
                # Track resolved frames to avoid re-asking
                self.resolved_frames.setdefault(self.selected_player_id, set()).add(int(self.current_frame_idx))
                print(f"[Correction] Saved for player {self.selected_player_id} at frame {self.current_frame_idx}. "
                      f"Resolved frames now: {sorted(self.resolved_frames[self.selected_player_id])}")
            # No popup; continue flow regardless of add success
            return

        # Otherwise, add as new player
            self.current_action_mode = "NONE"
            self._show_add_player_dialog(bbox, manual=True)

    def _on_person_clicked(self, person_idx: int):
        """Handle click on detected person"""
        if person_idx >= len(self.detected_people):
            return

        x, y, w, h, conf = self.detected_people[person_idx]
        bbox = (x, y, w, h)

        # Check if we're in correction mode
        if self.correction_mode and self.current_action_mode == "CORRECTION":
            # Add correction for selected player
            if not self.selected_player_id:
                QMessageBox.warning(self, "No Player Selected", "Please select a player first.")
                return

            # Add learning frame with detected bbox
            added = self.tracker_manager.add_learning_frame_to_player(
                self.selected_player_id, self.current_frame_idx, bbox, bbox
            )

            if added:
                # Track resolved frames
                self.resolved_frames.setdefault(self.selected_player_id, set()).add(int(self.current_frame_idx))
                player = self.tracker_manager.players.get(self.selected_player_id)
                player_name = player.name if player else f"Player {self.selected_player_id}"

                self.info_label.setText(f"✅ Correction added for {player_name} at frame {self.current_frame_idx}")
                print(f"[Correction] YOLO correction saved for player {self.selected_player_id} at frame {self.current_frame_idx}. "
                      f"Bbox={bbox}, Resolved frames: {sorted(self.resolved_frames[self.selected_player_id])}")

            # Exit correction mode
            self.correction_mode = False
            self.current_action_mode = "NONE"
            self.detected_people = []
            self._load_frame(self.current_frame_idx)
            return

        # Normal mode - show add player dialog
        self._show_add_player_dialog(bbox, manual=False, confidence=conf)

    def _show_add_player_dialog(self, bbox: Tuple[int, int, int, int], manual: bool = False, confidence: float = None):
        """Show dialog to add player"""
        x, y, w, h = bbox

        # Show dialog to get player info
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Player")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout()

        if manual:
            bbox_label = QLabel(f"Manual bbox: ({x}, {y}, {w}, {h})")
            bbox_label.setStyleSheet("color: #ffffff; font-size: 12px; font-weight: 600;")
            layout.addWidget(bbox_label)
        else:
            conf_text = f" ({confidence:.0%})" if confidence else ""
            detection_label = QLabel(f"Detected person at ({x}, {y}){conf_text}")
            detection_label.setStyleSheet("color: #ffffff; font-size: 12px; font-weight: 600;")
            layout.addWidget(detection_label)

        # Player name
        name_layout = QHBoxLayout()
        name_label = QLabel("Name:")
        name_label.setStyleSheet("color: #ffffff; font-size: 12px; font-weight: 600;")
        name_layout.addWidget(name_label)
        name_input = QLineEdit()
        name_input.setPlaceholderText(f"Player {len(self.tracker_manager.players) + 1}")
        name_layout.addWidget(name_input)
        layout.addLayout(name_layout)

        # Marker style
        style_layout = QHBoxLayout()
        marker_label = QLabel("Marker:")
        marker_label.setStyleSheet("color: #ffffff; font-size: 12px; font-weight: 600;")
        style_layout.addWidget(marker_label)
        style_combo = QComboBox()
        style_combo.addItems([
            "🏀 NBA Iso Ring",
            "〽️ Floating Chevron",
            "💡 Spotlight",
            "🎯 Crosshair",
            "🔲 Tactical Brackets",
            "🌊 Sonar Ripple"
        ])
        style_layout.addWidget(style_combo)
        layout.addLayout(style_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(dialog.reject)
        btn_layout.addWidget(btn_cancel)

        btn_add = QPushButton("Add Player")
        btn_add.clicked.connect(dialog.accept)
        btn_layout.addWidget(btn_add)
        layout.addLayout(btn_layout)

        dialog.setLayout(layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            player_name = name_input.text().strip() or f"Player {len(self.tracker_manager.players) + 1}"

            style_map = {
                0: "nba_iso_ring",
                1: "floating_chevron",
                2: "spotlight_modern",
                3: "crosshair",
                4: "tactical_brackets",
                5: "sonar_ripple"
            }
            marker_style = style_map.get(style_combo.currentIndex(), "nba_iso_ring")

            # Add player
            player_id = self.tracker_manager.add_player(
                name=player_name,
                marker_style=marker_style,
                initial_frame=self.current_frame_idx,
                bbox=(x, y, w, h)
            )

            # Update UI
            self.selected_player_id = player_id
            self._update_player_list()
            self.detected_people = []  # Clear detections
            self._load_frame(self.current_frame_idx)

            QMessageBox.information(
                self, "Player Added",
                f"{player_name} added!\n\nClick 'Start Tracking' to track this player."
            )

    def _start_tracking(self):
        """Start tracking"""
        if not self.tracker_manager.players:
            QMessageBox.warning(
                self, "No Players",
                "Add at least one player first.\n\nUse 'Detect People'."
            )
            return

        # Get selected players
        selected_players = []
        for i in range(self.player_list.count()):
            item = self.player_list.item(i)
            widget = self.player_list.itemWidget(item)
            if widget and widget.checkbox.isChecked():
                selected_players.append(widget.player_id)

        if not selected_players:
            QMessageBox.warning(
                self, "No Players Selected",
                "Select at least one player (checkbox)."
            )
            return

        # Start tracking
        self.tracking_thread = TrackingThread(
            self.tracker_manager,
            selected_players,
            self.start_frame,
            self.end_frame
        )
        self.tracking_thread.progress.connect(self._on_tracking_progress)
        self.tracking_thread.finished.connect(self._on_tracking_finished)
        self.tracking_thread.error.connect(self._on_tracking_error)

        # Update UI
        self.btn_start_tracking.setEnabled(False)
        self.btn_stop_tracking.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.tracking_thread.start()

    def _stop_tracking(self):
        """Stop tracking"""
        if self.tracking_thread:
            self.tracking_thread.stop()
            self.btn_stop_tracking.setEnabled(False)

    def _on_tracking_progress(self, current: int, total: int):
        """Tracking progress"""
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)

    def _on_tracking_finished(self, tracking_data: Dict):
        """Tracking finished"""
        self.tracking_data = tracking_data

        # Disconnect thread signals to avoid duplicate connections on next run
        if self.tracking_thread:
            try:
                self.tracking_thread.progress.disconnect()
                self.tracking_thread.finished.disconnect()
                self.tracking_thread.error.disconnect()
            except:
                pass
            self.tracking_thread = None

        # Update UI first (before any video operations)
        self.btn_start_tracking.setEnabled(True)
        self.btn_stop_tracking.setEnabled(False)
        self.btn_retrack.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Select first player automatically so the graph is populated
        if not self.selected_player_id and self.tracking_data:
            self.selected_player_id = next(iter(self.tracking_data.keys()))
            # Update list selection to match
            for i in range(self.player_list.count()):
                item = self.player_list.item(i)
                widget = self.player_list.itemWidget(item)
                if widget and widget.player_id == self.selected_player_id:
                    self.player_list.setCurrentItem(item)
                    break

        # Check for suspicious frames and offer to jump to first one
        if self.selected_player_id and self.selected_player_id in self.tracking_data:
            suspicious_frames = []
            for frame_idx, frame_data in self.tracking_data[self.selected_player_id].items():
                conf = frame_data.get('confidence', 0.0)
                if 0 < conf < 0.6 and not frame_data.get('is_learning_frame'):
                    suspicious_frames.append(frame_idx)

            if suspicious_frames:
                player = self.tracker_manager.players.get(self.selected_player_id)
                player_name = player.name if player else f"Player {self.selected_player_id}"

                reply = QMessageBox.question(
                    self,
                    "Suspicious Frames Detected",
                    f"⚠️  Found {len(suspicious_frames)} suspicious frames for {player_name}.\n\n"
                    f"These frames have low confidence (< 60%) and might be tracking the WRONG person.\n\n"
                    f"Jump to first suspicious frame ({min(suspicious_frames)}) to verify?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )

                if reply == QMessageBox.StandardButton.Yes:
                    first_suspicious = min(suspicious_frames)
                    self._jump_to_frame(first_suspicious)

        # Update graph
        if self.selected_player_id and self.selected_player_id in self.tracking_data:
            self.confidence_graph.set_data(
                self.tracking_data[self.selected_player_id],
                self.total_frames
            )

        # CRITICAL: Reset video_cap to ensure clean state for UI operations
        # After tracking, the video_cap is in an unknown state - reset it!
        if self.tracker_manager.video_cap:
            self.tracker_manager.video_cap.release()
            self.tracker_manager.video_cap = None
        # Create fresh capture for UI operations
        self.tracker_manager.video_cap = cv2.VideoCapture(self.tracker_manager.video_path)

        # Now it's safe to load a frame
        self._load_frame(self.current_frame_idx)

        # Find first problematic frame (low confidence or lost tracking)
        first_problem_frame = None
        if self.selected_player_id and self.selected_player_id in self.tracking_data:
            for frame_idx in sorted(self.tracking_data[self.selected_player_id].keys()):
                frame_data = self.tracking_data[self.selected_player_id][frame_idx]
                confidence = frame_data.get('confidence', 0.0)
                bbox = frame_data.get('bbox')
                learning_frames = set(self.tracker_manager.players.get(self.selected_player_id, {}).learning_frames.keys()) if hasattr(self.tracker_manager.players.get(self.selected_player_id, None), 'learning_frames') else set()
                resolved = self.resolved_frames.get(self.selected_player_id, set())

                # Problem: lost tracking or low confidence
                if bbox is None or confidence < 0.6:
                    # Skip if this frame already has learning frame/resolution
                    if frame_idx in learning_frames or frame_idx in resolved or frame_idx - 1 in resolved or frame_idx + 1 in resolved:
                        continue
                    first_problem_frame = frame_idx
                    break

        # Show completion message with option to jump to problem frame
        if first_problem_frame is not None:
            reply = QMessageBox.question(
                self, "Tracking Complete",
                f"Tracking done!\n\n"
                f"⚠️ Found problem at frame {first_problem_frame}\n"
                f"(Low confidence or lost tracking)\n\n"
                f"Jump to that frame now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.current_frame_idx = first_problem_frame
                self.frame_slider.setValue(first_problem_frame)
                self._load_frame(first_problem_frame)
                self.confidence_graph.set_current_frame(first_problem_frame)
        else:
            QMessageBox.information(
                self, "Tracking Complete",
                "✅ Tracking done!\n\n"
                "No major problems detected.\n"
                "Review the confidence graph to verify."
            )

    def _on_tracking_error(self, error_msg: str):
        """Tracking error"""
        # Disconnect thread signals to avoid duplicate connections on next run
        if self.tracking_thread:
            try:
                self.tracking_thread.progress.disconnect()
                self.tracking_thread.finished.disconnect()
                self.tracking_thread.error.disconnect()
            except:
                pass
            self.tracking_thread = None

        self.btn_start_tracking.setEnabled(True)
        self.btn_stop_tracking.setEnabled(False)
        self.progress_bar.setVisible(False)

        if "stopped by user" not in error_msg.lower():
            QMessageBox.critical(
                self, "Tracking Error",
                f"Error:\n\n{error_msg}"
            )

    def _retrack(self):
        """Re-track"""
        reply = QMessageBox.question(
            self, "Re-track",
            "Re-track with corrections?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._start_tracking()

    def _start_correction_mode(self):
        """Start correction mode: draw bbox to add learning frame for selected player"""
        if not self.selected_player_id:
            return

        self.correction_mode = True
        self.current_action_mode = "CORRECTION"
        # Enable drawing
        self.btn_manual_draw.setChecked(True)
        self._toggle_manual_draw(True, show_message=False)
        player = self.tracker_manager.players.get(self.selected_player_id)
        player_name = player.name if player else "player"

    def _start_detect_correction_mode(self):
        """Start detect correction mode: use YOLO to detect people for correction"""
        if not self.selected_player_id:
            QMessageBox.warning(
                self,
                "No Player Selected",
                "Please select a player from the list to add a correction."
            )
            return

        # Set correction mode
        self.correction_mode = True
        self.current_action_mode = "CORRECTION"

        # Run detection on current frame
        frame = self.tracker_manager.get_frame(self.current_frame_idx)
        if frame is None:
            QMessageBox.warning(self, "Error", "Failed to get current frame")
            return

        # Detect people with YOLO
        self.detected_people = self.person_detector.detect_people(frame, confidence_threshold=0.3)

        if not self.detected_people:
            QMessageBox.information(
                self,
                "No People Detected",
                "YOLO didn't detect any people in this frame.\nTry:\n- Using 'Draw Correction' instead\n- Moving to a clearer frame"
            )
            self.correction_mode = False
            self.current_action_mode = "NONE"
            return

        # Show detected people on video preview
        self._load_frame(self.current_frame_idx)

        player = self.tracker_manager.players.get(self.selected_player_id)
        player_name = player.name if player else "player"
        self.info_label.setText(f"✅ Detected {len(self.detected_people)} people. Click on {player_name} to correct.")

    def _on_load_clicked(self):
        """Load tracking data from JSON file"""
        # Choose JSON file to load
        video_dir = os.path.dirname(self.tracker_manager.video_path) if self.tracker_manager.video_path else os.path.expanduser("~")

        load_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Tracking Data",
            video_dir,
            "JSON Files (*.json);;All Files (*)"
        )

        if not load_path:
            return

        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

            # Check if this is new format (with 'players' and 'tracking_data') or old format
            if 'players' in loaded_data and 'tracking_data' in loaded_data:
                # New format - load player info and tracking data
                players_data = loaded_data['players']
                tracking_data = loaded_data['tracking_data']

                # Restore players to tracker_manager
                from ..tracking.tracker_manager import PlayerData
                for player_id_str, player_info in players_data.items():
                    player_id = int(player_id_str)
                    # Add player to tracker_manager if not exists
                    if player_id not in self.tracker_manager.players:
                        # Create PlayerData directly with the correct ID
                        # We need to get the first frame from tracking_data to use as initial_frame
                        first_frame_idx = 0
                        first_bbox = (100, 100, 50, 100)  # Default bbox
                        if player_id_str in tracking_data and tracking_data[player_id_str]:
                            # Get first frame from this player's tracking data
                            frame_indices = sorted([int(f) for f in tracking_data[player_id_str].keys()])
                            if frame_indices:
                                first_frame_idx = frame_indices[0]
                                first_frame_data = tracking_data[player_id_str][str(first_frame_idx)]
                                if first_frame_data.get('bbox'):
                                    first_bbox = tuple(first_frame_data['bbox'])

                        player = PlayerData(
                            player_id=player_id,
                            name=player_info.get('name', f'Player {player_id}'),
                            marker_style=player_info.get('marker_style', 'nba_iso_ring'),
                            initial_frame=first_frame_idx,
                            bbox=first_bbox
                        )
                        if player_info.get('color'):
                            player.color = tuple(player_info['color'])
                        self.tracker_manager.players[player_id] = player
                        # Update next_player_id to avoid conflicts
                        if player_id >= self.tracker_manager.next_player_id:
                            self.tracker_manager.next_player_id = player_id + 1
                    else:
                        # Update existing player
                        player = self.tracker_manager.players[player_id]
                        player.name = player_info.get('name', player.name)
                        player.marker_style = player_info.get('marker_style', player.marker_style)
                        if player_info.get('color'):
                            player.color = tuple(player_info['color'])
            else:
                # Old format - tracking_data is at root level
                tracking_data = loaded_data

            # Convert loaded tracking data back to tracking_data format
            self.tracking_data = {}
            for player_id_str, frames in tracking_data.items():
                player_id = int(player_id_str)
                self.tracking_data[player_id] = {}
                for frame_idx_str, frame_data in frames.items():
                    frame_idx = int(frame_idx_str)
                    bbox = frame_data.get('bbox')
                    if bbox:
                        bbox = tuple(bbox)
                    self.tracking_data[player_id][frame_idx] = {
                        'bbox': bbox,
                        'confidence': frame_data.get('confidence', 1.0),
                        'is_learning_frame': frame_data.get('is_learning_frame', False)
                    }

            # Update player list UI
            self._update_player_list()

            # Update UI state
            self.btn_export.setEnabled(True)

            # Update info label
            total_frames = sum(len(frames) for frames in self.tracking_data.values())
            num_players = len(self.tracking_data)
            self.info_label.setText(
                f"✅ Loaded tracking data: {num_players} player(s), {total_frames} total frames"
            )

            # Refresh the current frame to show loaded tracking
            self._load_frame(self.current_frame_idx)

            QMessageBox.information(
                self,
                "Load Successful",
                f"Tracking data loaded from:\n{load_path}\n\n"
                f"Players: {num_players}\n"
                f"Total frames tracked: {total_frames}\n\n"
                f"You can now export the video or continue editing."
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Failed",
                f"Could not load tracking data:\n{str(e)}"
            )

    def _on_export_clicked(self):
        """Save tracking data to JSON and optionally render video"""
        if not self.tracking_data:
            QMessageBox.warning(
                self,
                "No Tracking Data",
                "Run tracking first, then export."
            )
            return

        # Use video directory as default location for tracking data
        video_dir = os.path.dirname(self.tracker_manager.video_path) if self.tracker_manager.video_path else os.path.expanduser("~")
        default_path = os.path.join(video_dir, "tracking_data.json")

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Tracking Data",
            default_path,
            "JSON Files (*.json);;All Files (*)"
        )

        if not save_path:
            return

        # Save both player info AND tracking data
        serializable = {
            'players': {},
            'tracking_data': {}
        }

        # Save player information (names, marker styles, etc.)
        for player_id, player in self.tracker_manager.players.items():
            serializable['players'][str(player_id)] = {
                'name': player.name,
                'marker_style': player.marker_style,
                'color': player.color if hasattr(player, 'color') else None
            }

        # Save tracking data (bboxes, confidence, etc.)
        for player_id, frames in self.tracking_data.items():
            serializable['tracking_data'][str(player_id)] = {}
            for frame_idx, frame_data in frames.items():
                bbox = frame_data.get('bbox')
                if bbox:
                    bbox = [int(x) for x in bbox]
                serializable['tracking_data'][str(player_id)][str(frame_idx)] = {
                    'bbox': bbox,
                    'confidence': float(frame_data.get('confidence', 0.0)),
                    'is_learning_frame': bool(frame_data.get('is_learning_frame', False))
                }

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2)

            # Ask if user wants to render video with overlays
            render_reply = QMessageBox.question(
                self,
                "Render Video",
                "Tracking data saved.\n\nDo you want to export a video with overlays now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if render_reply == QMessageBox.StandardButton.Yes:
                # Choose output video path - use video directory as default
                video_dir = os.path.dirname(self.tracker_manager.video_path) if self.tracker_manager.video_path else os.path.expanduser("~")
                default_video_path = os.path.join(video_dir, "tracked_output.mp4")

                video_save_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Video with Tracking",
                    default_video_path,
                    "Video Files (*.mp4);;All Files (*)"
                )

                if video_save_path:
                    self._start_export(video_save_path)
                else:
                    QMessageBox.information(
                        self,
                        "Export Skipped",
                        "Video export skipped. JSON saved successfully."
                    )
                    self.accept()
            else:
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Tracking data saved to:\n{save_path}"
                )
                self.accept()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Could not save tracking data:\n{str(e)}"
            )

    def get_tracking_data(self) -> Dict:
        """Get tracking data"""
        return self.tracking_data

    def _start_export(self, video_save_path: str):
        """Kick off async export with progress UI"""
        # DEBUG: Verify tracking_data exists
        print(f"\n[Export Start Debug]")
        print(f"  tracking_data keys (players): {list(self.tracking_data.keys())}")
        for player_id in self.tracking_data:
            frames_with_bbox = sum(1 for f_data in self.tracking_data[player_id].values() if f_data.get('bbox') is not None)
            total_frames = len(self.tracking_data[player_id])
            print(f"  Player {player_id}: {frames_with_bbox}/{total_frames} frames have bbox")

        # Disable actions to avoid duplicates
        for btn in [self.btn_export, self.btn_start_tracking, self.btn_stop_tracking,
                    self.btn_retrack, self.btn_add_correction, self.btn_detect_correction,
                    self.btn_manual_draw, self.btn_detect]:
            btn.setEnabled(False)

        # Progress UI
        self.export_progress.setVisible(True)
        self.export_progress.setValue(0)
        self.export_progress.setFormat("Exporting… 0%")
        self.btn_cancel_export.setEnabled(True)

        self.export_thread = ExportThread(
            self.video_exporter,
            self.tracker_manager.video_path,
            self.tracking_data,
            video_save_path,
            self.start_frame,
            self.end_frame
        )
        self.export_thread.progress.connect(self._on_export_progress)
        self.export_thread.finished.connect(self._on_export_finished)
        self.export_thread.start()

    def _on_export_progress(self, percent: int, stage: str, current: int, total: int):
        """Update export progress bar"""
        self.export_progress.setValue(max(0, min(100, percent)))
        if stage == "render" and total:
            self.export_progress.setFormat(f"Rendering… {percent}%")
        elif stage == "audio":
            self.export_progress.setFormat(f"Merging audio… {percent}%")
        else:
            self.export_progress.setFormat(f"Exporting… {percent}%")

    def _on_export_finished(self, success: bool):
        """Handle export completion"""
        self.export_progress.setValue(100 if success else self.export_progress.value())
        self.export_progress.setFormat("Done" if success else "Failed")
        self.export_progress.setVisible(False)
        self.btn_cancel_export.setEnabled(False)

        for btn in [self.btn_export, self.btn_start_tracking, self.btn_stop_tracking,
                    self.btn_retrack, self.btn_add_correction, self.btn_detect_correction,
                    self.btn_manual_draw, self.btn_detect]:
            btn.setEnabled(True)

        if success:
            QMessageBox.information(self, "Export Complete", "Video exported successfully.")
            self.accept()
        else:
            QMessageBox.warning(self, "Export Failed", "Could not export video. Check FFmpeg installation and try again.")

    def _cancel_export(self):
        """Cancel export thread"""
        if hasattr(self, "export_thread") and self.export_thread is not None:
            self.export_thread.cancel()
        self.btn_cancel_export.setEnabled(False)
        self.export_progress.setFormat("Cancelling…")
