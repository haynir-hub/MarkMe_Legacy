"""
Player Selector - Dialog for selecting player marker style and name
With live preview of marker on the actual frame
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QComboBox, QGroupBox,
                             QSplitter, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap, QImage
import cv2
import numpy as np
from typing import Optional, Tuple


class PlayerSelector(QDialog):
    """Dialog for configuring player marker with live preview"""
    
    # Signal emitted when player is confirmed
    player_confirmed = pyqtSignal(str, str)  # name, style
    
    # Available styles with display names and internal keys
    STYLES = [
        ("🟣 Dynamic Ring 3D (Broadcast Purple)", "dynamic_ring_3d"),
        ("🛸 Alien Spotlight (Ceiling Beam)", "spotlight_alien"),
        ("🟢 Solid Floor Anchor (Green Ellipse)", "solid_anchor"),
        ("🔺 Defensive Radar (Coverage Cone)", "radar_defensive"),
        ("🎯 Sniper Scope (Crosshair Reticle)", "sniper_scope"),
    ]
    
    DESCRIPTIONS = {
        "dynamic_ring_3d": "טבעת סגולה תלת-ממדית על הרצפה עם אפקט פעימה",
        "spotlight_alien": "קרן אור צרה מהתקרה - מחשיך סביב השחקן",
        "solid_anchor": "אליפסה ירוקה מלאה על הרצפה מתחת לשחקן",
        "radar_defensive": "חרוט המראה את אזור הכיסוי ההגנתי של השחקן",
        "sniper_scope": "כוונת צלף - קרוסהייר גדול סביב השחקן",
    }
    
    def __init__(self, parent=None, frame: np.ndarray = None, 
                 bbox: Tuple[int, int, int, int] = None,
                 existing_name: str = "", existing_style: str = None):
        super().__init__(parent)
        self.setWindowTitle("בחר סגנון מרקר")
        self.setMinimumWidth(700)
        self.setMinimumHeight(500)
        
        # Store frame and bbox for preview
        self.frame = frame
        self.bbox = bbox
        self.renderer = None
        
        # Lazy load renderer
        if frame is not None and bbox is not None:
            try:
                from ..render.overlay_renderer import OverlayRenderer
                self.renderer = OverlayRenderer(use_segmentation=False)
            except ImportError:
                pass
        
        self._setup_ui(existing_name, existing_style)
        
        # Initial preview
        self._update_preview()
    
    def _setup_ui(self, existing_name: str, existing_style: str):
        """Setup UI with preview panel"""
        main_layout = QHBoxLayout()
        
        # ===== LEFT: Preview Panel =====
        preview_panel = QVBoxLayout()
        
        preview_label = QLabel("תצוגה מקדימה:")
        preview_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        preview_panel.addWidget(preview_label)
        
        self.preview_canvas = QLabel()
        self.preview_canvas.setMinimumSize(400, 300)
        self.preview_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_canvas.setStyleSheet("background-color: #1e1e1e; border: 1px solid #555; border-radius: 4px;")
        self.preview_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        preview_panel.addWidget(self.preview_canvas, stretch=1)
        
        # Hint
        hint_label = QLabel("💡 בחר סגנון מרקר מהרשימה - התצוגה תתעדכן בזמן אמת")
        hint_label.setStyleSheet("color: #888; font-size: 11px;")
        hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_panel.addWidget(hint_label)
        
        main_layout.addLayout(preview_panel, stretch=2)
        
        # ===== RIGHT: Controls Panel =====
        controls_panel = QVBoxLayout()
        
        # Player name
        name_group = QGroupBox("שם השחקן (אופציונלי)")
        name_layout = QVBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("הכנס שם שחקן...")
        self.name_input.setText(existing_name)
        name_layout.addWidget(self.name_input)
        name_group.setLayout(name_layout)
        controls_panel.addWidget(name_group)
        
        # Marker style
        style_group = QGroupBox("סגנון מרקר")
        style_layout = QVBoxLayout()
        
        self.style_combo = QComboBox()
        for display_name, _ in self.STYLES:
            self.style_combo.addItem(display_name)
        
        # Set existing style if provided
        if existing_style:
            for i, (_, style_key) in enumerate(self.STYLES):
                if style_key == existing_style:
                    self.style_combo.setCurrentIndex(i)
                    break
        
        # Description label
        self.style_description = QLabel()
        self.style_description.setWordWrap(True)
        self.style_description.setStyleSheet("color: #aaa; font-size: 11px; padding: 5px;")
        
        # Connect style change to preview update
        self.style_combo.currentIndexChanged.connect(self._on_style_changed)
        self._update_description()
        
        style_layout.addWidget(self.style_combo)
        style_layout.addWidget(self.style_description)
        style_group.setLayout(style_layout)
        controls_panel.addWidget(style_group)
        
        # Spacer
        controls_panel.addStretch()
        
        # Buttons
        button_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("ביטול")
        self.cancel_btn.clicked.connect(self.reject)
        self.confirm_btn = QPushButton("✓ אישור")
        self.confirm_btn.clicked.connect(self._on_confirm)
        self.confirm_btn.setDefault(True)
        self.confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #3f8cff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a9fff;
            }
        """)
        
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.confirm_btn)
        controls_panel.addLayout(button_layout)
        
        main_layout.addLayout(controls_panel, stretch=1)
        
        self.setLayout(main_layout)
    
    def _on_style_changed(self, index: int):
        """Handle style selection change"""
        self._update_description()
        self._update_preview()
    
    def _update_description(self):
        """Update style description based on selection"""
        style = self.get_selected_style()
        desc = self.DESCRIPTIONS.get(style, "")
        self.style_description.setText(desc)
    
    def _update_preview(self):
        """Update the preview canvas with current marker style"""
        if self.frame is None or self.bbox is None:
            self.preview_canvas.setText("אין תצוגה מקדימה זמינה")
            return
        
        if self.renderer is None:
            self.preview_canvas.setText("הרינדור לא זמין")
            return
        
        try:
            # Get current style
            style = self.get_selected_style()
            
            # Create preview frame (crop around bbox with padding)
            x, y, w, h = self.bbox
            frame_h, frame_w = self.frame.shape[:2]
            
            # Calculate crop region with padding
            padding = max(w, h) // 2
            crop_x1 = max(0, x - padding)
            crop_y1 = max(0, y - padding)
            crop_x2 = min(frame_w, x + w + padding)
            crop_y2 = min(frame_h, y + h + padding)
            
            # Crop frame
            cropped = self.frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            
            # Adjust bbox for cropped region
            adjusted_bbox = (x - crop_x1, y - crop_y1, w, h)
            
            # Create mock player object
            class MockPlayer:
                def __init__(self, bbox):
                    self.current_original_bbox = bbox
                    self.current_bbox = bbox
            
            mock_player = MockPlayer(adjusted_bbox)
            
            # Draw marker
            color = (255, 255, 0)  # Default cyan/yellow
            preview_frame = self.renderer.draw_marker(
                cropped, adjusted_bbox, style, color, mock_player,
                use_segmentation=False
            )
            
            # Convert to QPixmap and display
            self._display_frame(preview_frame)
            
        except Exception as e:
            print(f"Preview error: {e}")
            self.preview_canvas.setText(f"שגיאה בתצוגה: {e}")
    
    def _display_frame(self, frame: np.ndarray):
        """Display a frame on the preview canvas"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(rgb_frame.data.tobytes(), w, h, bytes_per_line, 
                         QImage.Format.Format_RGB888).copy()
        
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale to fit canvas while maintaining aspect ratio
        canvas_size = self.preview_canvas.size()
        scaled_pixmap = pixmap.scaled(
            canvas_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.preview_canvas.setPixmap(scaled_pixmap)
    
    def _on_confirm(self):
        """Handle confirm button click"""
        name = self.name_input.text().strip()
        if not name:
            name = f"Player {id(self) % 1000}"
        
        style = self.get_selected_style()
        self.player_confirmed.emit(name, style)
        self.accept()
    
    def get_selected_style(self) -> str:
        """Get selected marker style key"""
        index = self.style_combo.currentIndex()
        if 0 <= index < len(self.STYLES):
            return self.STYLES[index][1]
        return "dynamic_ring_3d"
    
    def resizeEvent(self, event):
        """Update preview on resize"""
        super().resizeEvent(event)
        self._update_preview()
