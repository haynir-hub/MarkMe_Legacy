"""
Time Range Dialog - Select start/end time for tracking
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSlider, QGroupBox)
from PyQt6.QtCore import Qt


class TimeRangeDialog(QDialog):
    """Dialog for selecting time range for tracking"""
    
    def __init__(self, total_frames: int, fps: float, parent=None):
        super().__init__(parent)
        self.total_frames = total_frames
        self.fps = fps
        self.start_frame = 0
        self.end_frame = total_frames - 1
        
        self.setWindowTitle("Select Time Range")
        self.setMinimumWidth(500)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel(
            "Select the time range for tracking.\n"
            "The tracker will only process frames within this range."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Start time
        start_group = QGroupBox("Start Time")
        start_layout = QVBoxLayout()
        
        self.start_label = QLabel()
        self._update_start_label()
        start_layout.addWidget(self.start_label)
        
        self.start_slider = QSlider(Qt.Orientation.Horizontal)
        self.start_slider.setMinimum(0)
        self.start_slider.setMaximum(self.total_frames - 1)
        self.start_slider.setValue(0)
        self.start_slider.valueChanged.connect(self._on_start_changed)
        start_layout.addWidget(self.start_slider)
        
        start_group.setLayout(start_layout)
        layout.addWidget(start_group)
        
        # End time
        end_group = QGroupBox("End Time")
        end_layout = QVBoxLayout()
        
        self.end_label = QLabel()
        self._update_end_label()
        end_layout.addWidget(self.end_label)
        
        self.end_slider = QSlider(Qt.Orientation.Horizontal)
        self.end_slider.setMinimum(0)
        self.end_slider.setMaximum(self.total_frames - 1)
        self.end_slider.setValue(self.total_frames - 1)
        self.end_slider.valueChanged.connect(self._on_end_changed)
        end_layout.addWidget(self.end_slider)
        
        end_group.setLayout(end_layout)
        layout.addWidget(end_group)
        
        # Summary
        self.summary_label = QLabel()
        self._update_summary()
        self.summary_label.setStyleSheet("font-weight: bold; padding: 10px;")
        layout.addWidget(self.summary_label)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(self.cancel_btn)
        
        self.reset_btn = QPushButton("Reset (Full Video)")
        self.reset_btn.clicked.connect(self._reset)
        buttons_layout.addWidget(self.reset_btn)
        
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setDefault(True)
        buttons_layout.addWidget(self.ok_btn)
        
        layout.addLayout(buttons_layout)
        
        self.setLayout(layout)
    
    def _frame_to_time(self, frame: int) -> str:
        """Convert frame number to time string"""
        seconds = frame / self.fps
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}:{secs:05.2f}"
    
    def _on_start_changed(self, value: int):
        """Handle start slider change"""
        # Ensure start <= end
        if value > self.end_slider.value():
            self.end_slider.setValue(value)
        
        self.start_frame = value
        self._update_start_label()
        self._update_summary()
    
    def _on_end_changed(self, value: int):
        """Handle end slider change"""
        # Ensure end >= start
        if value < self.start_slider.value():
            self.start_slider.setValue(value)
        
        self.end_frame = value
        self._update_end_label()
        self._update_summary()
    
    def _update_start_label(self):
        """Update start label"""
        time_str = self._frame_to_time(self.start_frame)
        self.start_label.setText(
            f"Frame {self.start_frame} ({time_str})"
        )
    
    def _update_end_label(self):
        """Update end label"""
        time_str = self._frame_to_time(self.end_frame)
        self.end_label.setText(
            f"Frame {self.end_frame} ({time_str})"
        )
    
    def _update_summary(self):
        """Update summary"""
        num_frames = self.end_frame - self.start_frame + 1
        duration = num_frames / self.fps
        
        self.summary_label.setText(
            f"📊 Selected Range: {num_frames} frames ({duration:.2f} seconds)"
        )
    
    def _reset(self):
        """Reset to full video"""
        self.start_slider.setValue(0)
        self.end_slider.setValue(self.total_frames - 1)
    
    def get_range(self) -> tuple:
        """Get selected range (start_frame, end_frame)"""
        return (self.start_frame, self.end_frame)






