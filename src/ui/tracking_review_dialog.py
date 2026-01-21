"""
Tracking Review Dialog - UI for reviewing and correcting tracking data
חלון סקירת מעקב - ממשק לסקירה ותיקון נתוני מעקב

Improved version with better UX and modern design
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QSlider, QListWidget, QListWidgetItem,
                             QSplitter, QWidget, QProgressBar, QCheckBox,
                             QSpinBox, QGroupBox, QScrollArea, QMessageBox, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..tracking.tracker_manager import TrackerManager
from ..tracking.person_detector import PersonDetector
from .bbox_editor import BboxEditor


# Modern color scheme
COLORS = {
    'bg_dark': '#1e1e1e',
    'bg_medium': '#2d2d2d',
    'bg_light': '#3c3c3c',
    'accent': '#0078d4',
    'accent_hover': '#1084d8',
    'success': '#16c60c',
    'warning': '#f7630c',
    'error': '#e81123',
    'text': '#ffffff',
    'text_muted': '#999999',
    'border': '#555555',
}

# Modern stylesheet
MODERN_STYLE = f"""
QDialog {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text']};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    font-size: 13px;
}}

QGroupBox {{
    background-color: {COLORS['bg_medium']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: bold;
    color: {COLORS['text']};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
}}

QPushButton {{
    background-color: {COLORS['accent']};
    color: {COLORS['text']};
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: 500;
    min-width: 80px;
}}

QPushButton:hover {{
    background-color: {COLORS['accent_hover']};
}}

QPushButton:pressed {{
    background-color: #0d6ebd;
}}

QPushButton:disabled {{
    background-color: {COLORS['bg_light']};
    color: {COLORS['text_muted']};
}}

QPushButton#successButton {{
    background-color: {COLORS['success']};
}}

QPushButton#successButton:hover {{
    background-color: #13a10e;
}}

QPushButton#warningButton {{
    background-color: {COLORS['warning']};
}}

QPushButton#warningButton:hover {{
    background-color: #da570b;
}}

QLabel {{
    color: {COLORS['text']};
}}

QLabel#mutedLabel {{
    color: {COLORS['text_muted']};
}}

QLabel#headerLabel {{
    font-size: 16px;
    font-weight: bold;
    padding: 8px 0;
}}

QListWidget {{
    background-color: {COLORS['bg_light']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 4px;
    color: {COLORS['text']};
}}

QListWidget::item {{
    padding: 8px;
    border-radius: 3px;
}}

QListWidget::item:selected {{
    background-color: {COLORS['accent']};
}}

QListWidget::item:hover {{
    background-color: {COLORS['bg_light']};
}}

QSlider::groove:horizontal {{
    border: 1px solid {COLORS['border']};
    height: 6px;
    background: {COLORS['bg_light']};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {COLORS['accent']};
    border: 1px solid {COLORS['border']};
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background: {COLORS['accent_hover']};
}}

QCheckBox {{
    color: {COLORS['text']};
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 2px solid {COLORS['border']};
    background-color: {COLORS['bg_light']};
}}

QCheckBox::indicator:checked {{
    background-color: {COLORS['accent']};
    border-color: {COLORS['accent']};
}}

QSpinBox {{
    background-color: {COLORS['bg_light']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 4px 8px;
    color: {COLORS['text']};
}}

QScrollArea {{
    border: none;
    background-color: transparent;
}}

QFrame#separator {{
    background-color: {COLORS['border']};
    max-height: 1px;
    min-height: 1px;
}}
"""


class ConfidenceGraph(QWidget):
    """Widget for displaying confidence graph over time"""

    frame_clicked = pyqtSignal(int)  # Emits frame index when clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracking_data = {}
        self.player_id = None
        self.current_frame = 0
        self.setMinimumHeight(120)
        self.setMaximumHeight(200)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_data(self, tracking_data: Dict[int, Dict[str, any]], player_id: int):
        """Set tracking data to display"""
        self.tracking_data = tracking_data
        self.player_id = player_id
        self.update()

    def set_current_frame(self, frame_idx: int):
        """Update current frame indicator"""
        self.current_frame = frame_idx
        self.update()

    def paintEvent(self, event):
        """Draw the confidence graph"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        margin = 40

        # Background with gradient
        painter.fillRect(0, 0, width, height, QColor(30, 30, 30))

        if not self.tracking_data:
            painter.setPen(QPen(QColor(150, 150, 150), 1))
            painter.drawText(0, 0, width, height, Qt.AlignmentFlag.AlignCenter,
                           "No tracking data / אין נתוני מעקב")
            painter.end()
            return

        # Get frame range
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

        # Draw axes
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawLine(margin, margin, margin, height - margin)  # Y axis
        painter.drawLine(margin, height - margin, width - margin, height - margin)  # X axis

        # Draw grid
        painter.setPen(QPen(QColor(60, 60, 60), 1, Qt.PenStyle.DotLine))
        for i in range(0, 11, 2):  # Horizontal grid lines
            y = margin + (i * (height - 2 * margin) / 10)
            painter.drawLine(margin, int(y), width - margin, int(y))

        # Draw confidence zones with labels
        graph_height = height - 2 * margin
        graph_width = width - 2 * margin

        # High confidence zone (green)
        painter.fillRect(margin, margin, graph_width, int(graph_height * 0.2), QColor(0, 150, 0, 20))

        # Medium confidence zone (yellow)
        painter.fillRect(margin, margin + int(graph_height * 0.2), graph_width,
                        int(graph_height * 0.3), QColor(150, 150, 0, 20))

        # Low confidence zone (red)
        painter.fillRect(margin, margin + int(graph_height * 0.5), graph_width,
                        int(graph_height * 0.5), QColor(150, 0, 0, 20))

        # Draw Y-axis labels
        font = QFont("Arial", 9)
        painter.setFont(font)
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        for i in [0, 25, 50, 75, 100]:
            y = margin + int((100 - i) * graph_height / 100)
            painter.drawText(5, y + 5, f"{i/100:.1f}")

        # Calculate points
        points = []
        for frame_idx in frames:
            data = self.tracking_data[frame_idx]
            confidence = data.get('confidence', 0.0)

            x = margin + int((frame_idx - min_frame) / frame_range * graph_width)
            y = margin + int((1 - confidence) * graph_height)
            points.append((x, y, confidence, frame_idx, data.get('is_learning_frame', False)))

        # Draw confidence line with gradient effect
        painter.setPen(QPen(QColor(0, 200, 255), 3))
        for i in range(len(points) - 1):
            x1, y1, conf1, _, _ = points[i]
            x2, y2, conf2, _, _ = points[i + 1]

            # Color based on confidence
            if conf1 < 0.5 or conf2 < 0.5:
                painter.setPen(QPen(QColor(255, 100, 100), 3))
            elif conf1 < 0.7 or conf2 < 0.7:
                painter.setPen(QPen(QColor(255, 200, 0), 3))
            else:
                painter.setPen(QPen(QColor(0, 200, 255), 3))

            painter.drawLine(x1, y1, x2, y2)

        # Draw points
        for x, y, confidence, frame_idx, is_learning in points:
            if is_learning:
                # Learning frames - larger, gold color with glow
                painter.setPen(QPen(QColor(255, 215, 0, 100), 3))
                painter.setBrush(QColor(255, 215, 0))
                painter.drawEllipse(x - 6, y - 6, 12, 12)
            elif confidence < 0.5:
                # Low confidence - red
                painter.setPen(QPen(QColor(255, 0, 0), 1))
                painter.setBrush(QColor(255, 100, 100))
                painter.drawEllipse(x - 4, y - 4, 8, 8)
            elif confidence < 0.7:
                # Medium confidence - yellow
                painter.setPen(QPen(QColor(255, 200, 0), 1))
                painter.setBrush(QColor(255, 200, 0))
                painter.drawEllipse(x - 3, y - 3, 6, 6)
            else:
                # High confidence - cyan
                painter.setPen(QPen(QColor(0, 200, 255), 1))
                painter.setBrush(QColor(0, 200, 255))
                painter.drawEllipse(x - 2, y - 2, 4, 4)

        # Draw current frame indicator
        if min_frame <= self.current_frame <= max_frame:
            x = margin + int((self.current_frame - min_frame) / frame_range * graph_width)
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawLine(x, margin, x, height - margin)

            # Draw frame number at current position
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawText(x + 5, margin + 15, f"{self.current_frame}")

        # Draw X-axis labels
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        painter.drawText(margin, height - 10, f"{min_frame}")
        painter.drawText(width - margin - 30, height - 10, f"{max_frame}")
        painter.drawText(width // 2 - 20, height - 10, f"{(min_frame + max_frame) // 2}")

        painter.end()

    def mousePressEvent(self, event):
        """Handle mouse click to jump to frame"""
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

        margin = 40
        graph_width = self.width() - 2 * margin

        x = event.pos().x() - margin
        if x < 0 or x > graph_width:
            return

        # Convert x position to frame index
        frame_idx = int(min_frame + (x / graph_width) * frame_range)
        frame_idx = max(min_frame, min(max_frame, frame_idx))

        self.frame_clicked.emit(frame_idx)


class TrackingReviewDialog(QDialog):
    """Dialog for reviewing and correcting tracking data with modern UI"""

    def __init__(self, tracker_manager: TrackerManager,
                 tracking_data: Dict[int, Dict[int, Dict[str, any]]],
                 parent=None):
        super().__init__(parent)
        self.tracker_manager = tracker_manager
        self.tracking_data = tracking_data
        self.current_frame_idx = 0
        self.current_player_id = None
        self.problematic_frames = []
        self.person_detector: Optional[PersonDetector] = None
        self._last_displayed_frame: Optional[int] = None

        self.setWindowTitle("סקירת מעקב - Tracking Review")
        self.setMinimumSize(1000, 700)

        # Get screen size and set window to 90% of screen
        from PyQt6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen().geometry()
        window_width = int(screen.width() * 0.9)
        window_height = int(screen.height() * 0.85)
        self.resize(window_width, window_height)

        # Apply modern stylesheet
        self.setStyleSheet(MODERN_STYLE)

        self._init_ui()
        self._analyze_tracking_data()
        self._load_first_player()

    def _init_ui(self):
        """Initialize the user interface with modern design"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Header with title and instructions
        header_label = QLabel("🎯 Tracking Review & Correction")
        header_label.setObjectName("headerLabel")
        main_layout.addWidget(header_label)

        instructions = QLabel(
            "Click on the confidence graph to jump to frames • "
            "Use 'Fix Frame' to manually correct tracking • "
            "Click 'Re-track' after making corrections"
        )
        instructions.setObjectName("mutedLabel")
        instructions.setWordWrap(True)
        main_layout.addWidget(instructions)

        # Separator
        separator = QFrame()
        separator.setObjectName("separator")
        main_layout.addWidget(separator)

        # Top section: Player selection and statistics
        top_widget = QWidget()
        top_layout = QHBoxLayout()
        top_layout.setSpacing(12)

        # Player selection
        player_group = QGroupBox("📋 Players / שחקנים")
        player_layout = QVBoxLayout()
        player_layout.setSpacing(8)

        self.player_list = QListWidget()
        self.player_list.setMinimumHeight(80)
        self.player_list.setMaximumHeight(120)
        self.player_list.currentItemChanged.connect(self._on_player_changed)
        player_layout.addWidget(self.player_list)

        # Populate player list
        for player_id, player in self.tracker_manager.players.items():
            item = QListWidgetItem(f"👤 {player.name} (ID: {player_id})")
            item.setData(Qt.ItemDataRole.UserRole, player_id)
            self.player_list.addItem(item)

        player_group.setLayout(player_layout)
        top_layout.addWidget(player_group, 1)

        # Statistics panel
        stats_group = QGroupBox("📊 Tracking Statistics / סטטיסטיקות")
        stats_layout = QVBoxLayout()
        stats_layout.setSpacing(4)

        # Wrap stats in scroll area
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        stats_scroll.setMaximumHeight(120)

        self.stats_label = QLabel("Select a player to view statistics\nבחר שחקן לצפייה בסטטיסטיקות")
        self.stats_label.setWordWrap(True)
        stats_scroll.setWidget(self.stats_label)

        stats_layout.addWidget(stats_scroll)

        stats_group.setLayout(stats_layout)
        top_layout.addWidget(stats_group, 2)

        top_widget.setLayout(top_layout)
        main_layout.addWidget(top_widget)

        # Confidence graph section
        graph_group = QGroupBox("📈 Confidence Over Time / ביטחון לאורך זמן")
        graph_layout = QVBoxLayout()
        graph_layout.setSpacing(8)

        self.confidence_graph = ConfidenceGraph()
        self.confidence_graph.setMinimumHeight(100)
        self.confidence_graph.setMaximumHeight(150)
        self.confidence_graph.frame_clicked.connect(self._jump_to_frame)
        graph_layout.addWidget(self.confidence_graph)

        # Legend with better styling
        legend_layout = QHBoxLayout()
        legend_layout.addWidget(QLabel("🟡 Learning Frame (1.0)"))
        legend_layout.addWidget(QLabel("🟢 High Confidence (>0.7)"))
        legend_layout.addWidget(QLabel("🟡 Medium (0.5-0.7)"))
        legend_layout.addWidget(QLabel("🔴 Low (<0.5)"))
        legend_layout.addStretch()
        graph_layout.addLayout(legend_layout)

        graph_group.setLayout(graph_layout)
        main_layout.addWidget(graph_group)

        # Main content: Video preview and controls
        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side: Video preview with controls
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # Video preview group
        preview_group = QGroupBox("🎬 Video Preview / תצוגת וידאו")
        preview_layout = QVBoxLayout()

        # BboxEditor with responsive sizing
        from PyQt6.QtWidgets import QSizePolicy
        self.bbox_editor = BboxEditor()
        self.bbox_editor.setMinimumSize(400, 300)  # Much smaller minimum
        # Let it grow to fill available space
        self.bbox_editor.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.bbox_editor.setStyleSheet(f"background-color: {COLORS['bg_dark']}; border-radius: 4px;")
        self.bbox_editor.bbox_changed.connect(self._on_bbox_edited)
        preview_layout.addWidget(self.bbox_editor, 1)  # stretch factor 1

        # Frame controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(8)

        # Slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(1, self.tracker_manager.total_frames - 1))
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        controls_layout.addWidget(self.frame_slider)

        # Buttons row
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(8)

        self.prev_frame_btn = QPushButton("◀ Previous")
        self.prev_frame_btn.clicked.connect(self._prev_frame)
        buttons_layout.addWidget(self.prev_frame_btn)

        self.next_frame_btn = QPushButton("Next ▶")
        self.next_frame_btn.clicked.connect(self._next_frame)
        buttons_layout.addWidget(self.next_frame_btn)

        buttons_layout.addStretch()

        self.auto_detect_btn = QPushButton("🤖 Auto Detect")
        self.auto_detect_btn.clicked.connect(self._auto_detect_players)
        buttons_layout.addWidget(self.auto_detect_btn)

        self.fix_frame_btn = QPushButton("🔧 Fix Frame")
        self.fix_frame_btn.setObjectName("warningButton")
        self.fix_frame_btn.clicked.connect(self._fix_current_frame)
        buttons_layout.addWidget(self.fix_frame_btn)

        self.retrack_btn = QPushButton("🔄 Re-track")
        self.retrack_btn.setObjectName("successButton")
        self.retrack_btn.clicked.connect(self._retrack)
        buttons_layout.addWidget(self.retrack_btn)

        controls_layout.addLayout(buttons_layout)

        # Frame info row
        info_layout = QHBoxLayout()
        self.frame_info_label = QLabel("Frame: 0 / 0")
        info_layout.addWidget(self.frame_info_label)
        info_layout.addStretch()

        self.confidence_label = QLabel("Confidence: N/A")
        info_layout.addWidget(self.confidence_label)

        controls_layout.addLayout(info_layout)

        controls_widget.setLayout(controls_layout)
        preview_layout.addWidget(controls_widget)

        preview_group.setLayout(preview_layout)
        left_layout.addWidget(preview_group)

        left_widget.setLayout(left_layout)
        content_splitter.addWidget(left_widget)

        # Right side: Problematic frames list
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        problems_group = QGroupBox("⚠️  Problematic Frames / פריימים בעייתיים")
        problems_layout = QVBoxLayout()
        problems_layout.setSpacing(8)

        # Filters
        filter_layout = QHBoxLayout()

        self.show_low_conf_cb = QCheckBox("Low Confidence")
        self.show_low_conf_cb.setChecked(True)
        self.show_low_conf_cb.stateChanged.connect(self._update_problems_list)
        filter_layout.addWidget(self.show_low_conf_cb)

        self.show_lost_cb = QCheckBox("Lost Tracking")
        self.show_lost_cb.setChecked(True)
        self.show_lost_cb.stateChanged.connect(self._update_problems_list)
        filter_layout.addWidget(self.show_lost_cb)

        filter_layout.addStretch()

        filter_layout.addWidget(QLabel("Threshold:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setMinimum(0)
        self.threshold_spin.setMaximum(100)
        self.threshold_spin.setValue(50)
        self.threshold_spin.setSuffix("%")
        self.threshold_spin.valueChanged.connect(self._update_problems_list)
        filter_layout.addWidget(self.threshold_spin)

        problems_layout.addLayout(filter_layout)

        # Problems list with scroll
        self.problems_list = QListWidget()
        self.problems_list.setMinimumHeight(200)  # Minimum height
        self.problems_list.itemClicked.connect(self._on_problem_clicked)
        problems_layout.addWidget(self.problems_list, 1)  # stretch factor

        problems_group.setLayout(problems_layout)
        right_layout.addWidget(problems_group)

        right_widget.setLayout(right_layout)
        content_splitter.addWidget(right_widget)

        # Set splitter to distribute space properly
        # Don't use fixed sizes - let it adapt to screen
        content_splitter.setStretchFactor(0, 7)  # Video gets 70%
        content_splitter.setStretchFactor(1, 3)  # Problems list gets 30%

        main_layout.addWidget(content_splitter, 1)

        # Bottom buttons
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()

        self.cancel_btn = QPushButton("❌ Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(self.cancel_btn)

        self.export_btn = QPushButton("✅ Continue to Export")
        self.export_btn.setObjectName("successButton")
        self.export_btn.clicked.connect(self.accept)
        bottom_layout.addWidget(self.export_btn)

        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

    def _analyze_tracking_data(self):
        """Analyze tracking data to find problematic frames"""
        from ..tracking.tracking_analyzer import TrackingAnalyzer

        analyzer = TrackingAnalyzer()

        for player_id, player_data in self.tracking_data.items():
            issues = analyzer.analyze(
                player_data,
                self.tracker_manager.frame_width,
                self.tracker_manager.frame_height
            )

            for issue in issues:
                self.problematic_frames.append({
                    'player_id': player_id,
                    'frame': issue.frame_idx,
                    'type': issue.issue_type,
                    'severity': issue.severity,
                    'description': issue.description
                })

    def _load_first_player(self):
        """Load the first player in the list"""
        if self.player_list.count() > 0:
            self.player_list.setCurrentRow(0)

    def _on_player_changed(self, current, previous):
        """Handle player selection change"""
        if current is None:
            return

        player_id = current.data(Qt.ItemDataRole.UserRole)
        self.current_player_id = player_id
        self.bbox_editor.clear_candidate_bboxes()

        # Update confidence graph
        if player_id in self.tracking_data:
            self.confidence_graph.set_data(self.tracking_data[player_id], player_id)

        # Update statistics
        self._update_statistics()

        # Update problems list
        self._update_problems_list()

        # Display first frame
        if player_id in self.tracking_data:
            frames = sorted(self.tracking_data[player_id].keys())
            if frames:
                self._jump_to_frame(frames[0])

    def _update_statistics(self):
        """Update statistics display for current player"""
        if self.current_player_id is None:
            return

        player = self.tracker_manager.get_player(self.current_player_id)
        player_data = self.tracking_data.get(self.current_player_id, {})

        if not player_data:
            self.stats_label.setText("No tracking data / אין נתוני מעקב")
            return

        # Calculate statistics
        total_frames = len(player_data)
        lost_frames = sum(1 for data in player_data.values() if data.get('bbox') is None)
        learning_frames = sum(1 for data in player_data.values() if data.get('is_learning_frame', False))

        confidences = [data.get('confidence', 0.0) for data in player_data.values() if data.get('bbox') is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        min_confidence = min(confidences) if confidences else 0.0
        max_confidence = max(confidences) if confidences else 0.0

        # Quality assessment
        if avg_confidence >= 0.8:
            quality = "✅ Excellent"
            quality_color = COLORS['success']
        elif avg_confidence >= 0.6:
            quality = "⚠️  Good"
            quality_color = COLORS['warning']
        else:
            quality = "❌ Poor"
            quality_color = COLORS['error']

        stats_html = f"""
        <div style='line-height: 1.6;'>
            <p><b>{player.name}</b></p>
            <p>Total Frames: <b>{total_frames}</b></p>
            <p>Lost Frames: <b style='color: {COLORS['error']};'>{lost_frames}</b> ({100*lost_frames/total_frames:.1f}%)</p>
            <p>Learning Frames: <b style='color: {COLORS['success']};'>{learning_frames}</b></p>
            <hr style='border: 1px solid {COLORS['border']};'>
            <p>Avg Confidence: <b>{avg_confidence:.2f}</b></p>
            <p>Min: <b>{min_confidence:.2f}</b> | Max: <b>{max_confidence:.2f}</b></p>
            <p>Quality: <b style='color: {quality_color};'>{quality}</b></p>
        </div>
        """

        self.stats_label.setText(stats_html)

    def _update_problems_list(self):
        """Update the problematic frames list based on filters"""
        self.problems_list.clear()

        if self.current_player_id is None:
            return

        threshold = self.threshold_spin.value() / 100.0
        show_low_conf = self.show_low_conf_cb.isChecked()
        show_lost = self.show_lost_cb.isChecked()

        # Filter problems for current player
        player_problems = [p for p in self.problematic_frames if p['player_id'] == self.current_player_id]

        # Sort by frame number
        player_problems.sort(key=lambda x: x['frame'])

        # Add to list
        for problem in player_problems:
            # Apply filters
            if problem['type'] == 'lost' and not show_lost:
                continue
            if problem['type'] == 'low_confidence' and not show_low_conf:
                continue

            # Check threshold for confidence issues
            player_data = self.tracking_data[self.current_player_id]
            if problem['frame'] in player_data:
                conf = player_data[problem['frame']].get('confidence', 0.0)
                if problem['type'] == 'low_confidence' and conf > threshold:
                    continue

            # Create list item
            icon = "🔴" if problem['severity'] == 'critical' else "🟠" if problem['severity'] == 'high' else "🟡"
            item_text = f"{icon} Frame {problem['frame']}: {problem['description']}"

            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, problem['frame'])
            self.problems_list.addItem(item)

    def _on_problem_clicked(self, item):
        """Handle click on problematic frame"""
        frame_idx = item.data(Qt.ItemDataRole.UserRole)
        self._jump_to_frame(frame_idx)

    def _jump_to_frame(self, frame_idx: int):
        """Jump to specific frame"""
        self.current_frame_idx = frame_idx
        self.frame_slider.setValue(frame_idx)
        self._display_frame()

    def _on_frame_changed(self, value):
        """Handle frame slider change"""
        self.current_frame_idx = value
        self._display_frame()

    def _prev_frame(self):
        """Go to previous frame"""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.frame_slider.setValue(self.current_frame_idx)

    def _next_frame(self):
        """Go to next frame"""
        if self.current_frame_idx < self.tracker_manager.total_frames - 1:
            self.current_frame_idx += 1
            self.frame_slider.setValue(self.current_frame_idx)

    def _display_frame(self):
        """Display the current frame with bbox overlay"""
        if self.current_player_id is None:
            return

        if self._last_displayed_frame != self.current_frame_idx:
            self.bbox_editor.clear_candidate_bboxes()
            self._last_displayed_frame = self.current_frame_idx

        # Get frame
        frame = self.tracker_manager.get_frame(self.current_frame_idx)
        if frame is None:
            return

        # Get tracking data for this frame
        player_data = self.tracking_data.get(self.current_player_id, {})
        current_data = player_data.get(self.current_frame_idx, {})
        bbox = current_data.get('bbox')
        confidence = current_data.get('confidence', 0.0)
        is_learning = current_data.get('is_learning_frame', False)

        # Update bbox editor
        self.bbox_editor.set_frame(frame, bbox)

        # Update confidence graph
        self.confidence_graph.set_current_frame(self.current_frame_idx)

        # Update labels
        self.frame_info_label.setText(
            f"Frame: {self.current_frame_idx} / {self.tracker_manager.total_frames - 1}"
        )

        if bbox is not None:
            conf_color = COLORS['success'] if confidence >= 0.7 else COLORS['warning'] if confidence >= 0.5 else COLORS['error']
            learning_text = " 🟡 (Learning)" if is_learning else ""
            self.confidence_label.setText(
                f"<span style='color: {conf_color};'>Confidence: {confidence:.2f}{learning_text}</span>"
            )
        else:
            self.confidence_label.setText(
                f"<span style='color: {COLORS['error']};'>Tracking Lost</span>"
            )

    def _auto_detect_players(self):
        """Run automatic person detection and allow selecting a bbox"""
        if self.current_player_id is None:
            QMessageBox.information(self, "Select Player", "Select a player before running auto-detect.")
            return

        if self.person_detector is None:
            self.person_detector = PersonDetector()

        if not self.person_detector.is_available():
            QMessageBox.warning(
                self,
                "Detection Not Available",
                "Automatic player detection is not available.\n"
                "Install or fix YOLO dependencies and try again."
            )
            return

        frame = self.tracker_manager.get_frame(self.current_frame_idx)
        if frame is None:
            QMessageBox.warning(self, "Error", "Failed to load current frame for detection.")
            return

        detections = self.person_detector.detect_people(frame, confidence_threshold=0.25)
        if not detections:
            self.bbox_editor.clear_candidate_bboxes()
            QMessageBox.information(
                self,
                "No Players Detected",
                "No players were found in this frame.\n"
                "Try another frame or draw a bbox manually."
            )
            return

        self.bbox_editor.set_candidate_bboxes(detections)
        QMessageBox.information(
            self,
            "Detections Ready",
            f"Found {len(detections)} player candidates.\n"
            f"Click a highlighted box to use it, or keep editing manually."
        )

    def _fix_current_frame(self):
        """Allow user to manually fix current frame bbox"""
        QMessageBox.information(
            self,
            "Manual Correction / תיקון ידני",
            "<b>How to edit bounding box:</b><br><br>"
            "• <b>Auto-detect:</b> Click 'Auto Detect' then pick a highlighted box<br>"
            "• <b>Create:</b> Click and drag to draw new bbox<br>"
            "• <b>Resize:</b> Drag corners or edges<br>"
            "• <b>Move:</b> Drag the center<br>"
            "• <b>Delete:</b> Press Delete or Backspace<br>"
            "• <b>Cancel:</b> Press ESC<br><br>"
            "<b>איך לערוך bbox:</b><br><br>"
            "• <b>זיהוי אוטומטי:</b> לחץ 'זיהוי אוטומטי' ואז בחר ב-box מסומן<br>"
            "• <b>יצירה:</b> לחץ וגרור<br>"
            "• <b>שינוי גודל:</b> גרור פינות או קצוות<br>"
            "• <b>הזזה:</b> גרור את המרכז<br>"
            "• <b>מחיקה:</b> Delete או Backspace<br>"
            "• <b>ביטול:</b> ESC"
        )

    def _on_bbox_edited(self, bbox: Tuple[int, int, int, int]):
        """Handle bbox edit - automatically add as learning frame"""
        if self.current_player_id is None:
            return

        # Clear auto-detection overlays once a bbox is chosen
        self.bbox_editor.clear_candidate_bboxes()

        # Add to tracker manager as learning frame
        self.tracker_manager.add_learning_frame_to_player(
            self.current_player_id,
            self.current_frame_idx,
            bbox
        )

        # Update tracking data
        if self.current_player_id not in self.tracking_data:
            self.tracking_data[self.current_player_id] = {}

        self.tracking_data[self.current_player_id][self.current_frame_idx] = {
            'bbox': bbox,
            'confidence': 1.0,  # Perfect confidence for manual corrections
            'is_learning_frame': True
        }

        # Refresh display
        self._display_frame()
        self._update_statistics()

        # Update confidence graph
        self.confidence_graph.set_data(
            self.tracking_data[self.current_player_id],
            self.current_player_id
        )

        # Show confirmation
        QMessageBox.information(
            self,
            "Learning Frame Added / נוסף learning frame",
            f"Frame {self.current_frame_idx} has been marked as a learning frame.\n"
            f"Click 'Re-track' to update tracking with this correction.\n\n"
            f"פריים {self.current_frame_idx} סומן כ-learning frame.\n"
            f"לחץ 'מעקב מחדש' לעדכון המעקב עם התיקון."
        )

    def _retrack(self):
        """Re-generate tracking data with learning frames"""
        if self.current_player_id is None:
            return

        reply = QMessageBox.question(
            self,
            "Re-track / מעקב מחדש",
            "Re-generate tracking data with current corrections?\n"
            "This may take a few moments.\n\n"
            "לייצר מחדש נתוני מעקב עם התיקונים?\n"
            "זה עשוי לקחת מספר רגעים.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Get frame range for current player
        player_data = self.tracking_data.get(self.current_player_id, {})
        if not player_data:
            return

        frames = sorted(player_data.keys())
        start_frame = min(frames)
        end_frame = max(frames)

        # Re-generate tracking data
        try:
            new_data = self.tracker_manager.generate_tracking_data(
                start_frame=start_frame,
                end_frame=end_frame
            )

            # Update tracking data
            if self.current_player_id in new_data:
                self.tracking_data[self.current_player_id] = new_data[self.current_player_id]

            # Refresh display
            self.confidence_graph.set_data(
                self.tracking_data[self.current_player_id],
                self.current_player_id
            )
            self._update_statistics()
            self._update_problems_list()
            self._display_frame()

            QMessageBox.information(
                self,
                "Success / הצלחה",
                "Tracking data updated successfully!\n"
                "נתוני המעקב עודכנו בהצלחה!"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error / שגיאה",
                f"Failed to re-track:\n{str(e)}\n\n"
                f"נכשל במעקב מחדש:\n{str(e)}"
            )
