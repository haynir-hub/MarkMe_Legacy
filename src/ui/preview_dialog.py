"""
Preview Dialog - Shows tracking preview before final export
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QCheckBox, QSlider, QSizePolicy, QSpinBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
import cv2
import numpy as np
from typing import Optional

from .video_canvas import VideoCanvas
from ..tracking.tracker_manager import TrackerManager
from ..render.overlay_renderer import OverlayRenderer


class PreviewDialog(QDialog):
    """Dialog for previewing tracking results before export"""
    
    # Signal when user approves export
    export_approved = pyqtSignal()
    # Signal when user wants to fix tracking (frame_idx, bbox, player_id)
    fix_tracking_requested = pyqtSignal(int, int, int, int, int, int)  # frame_idx, x, y, w, h, player_id
    # Signal when user wants to resume tracking from a specific frame
    resume_tracking_requested = pyqtSignal(int, int, int, int, int, int)  # frame_idx, x, y, w, h, player_id
    # Signal when user wants to re-track the entire video
    retrack_requested = pyqtSignal()
    
    def __init__(self, tracker_manager: TrackerManager, video_path: str, parent=None,
                 tracking_start_frame: Optional[int] = None, tracking_end_frame: Optional[int] = None):
        super().__init__(parent)
        self.tracker_manager = tracker_manager
        self.video_path = video_path
        self.overlay_renderer = OverlayRenderer()
        self.tracking_start_frame = tracking_start_frame
        self.tracking_end_frame = tracking_end_frame
        
        self.current_frame_idx = 0
        self.is_playing = False
        self.is_fullscreen = False
        self.approved = False
        self._waiting_for_fix_bbox = False
        self._fix_player_id = None
        self._fix_frame_idx = None  # Store frame where fix was applied
        self._fix_bbox = None  # Store bbox for fix
        
        self.setWindowTitle("Preview Tracking Results")
        self.setMinimumSize(1000, 700)
        
        self._setup_ui()
        self._setup_timer()
        self._setup_shortcuts()
        
        # Load first frame
        self._load_frame(0)
    
    def _setup_ui(self):
        """Setup UI components"""
        layout = QVBoxLayout()
        
        # Video canvas
        self.video_canvas = VideoCanvas()
        self.video_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_canvas.bbox_selected.connect(self._on_fix_bbox_selected)
        layout.addWidget(self.video_canvas)
        
        # Frame info
        self.frame_info = QLabel("Frame: 0 / 0")
        self.frame_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.frame_info)
        
        # Playback controls
        controls_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_play)
        controls_layout.addWidget(self.play_btn)
        
        self.prev_btn = QPushButton("⏮ Previous")
        self.prev_btn.clicked.connect(self._prev_frame)
        controls_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next ⏭")
        self.next_btn.clicked.connect(self._next_frame)
        controls_layout.addWidget(self.next_btn)
        
        self.fullscreen_btn = QPushButton("🖵 Fullscreen")
        self.fullscreen_btn.clicked.connect(self._toggle_fullscreen)
        controls_layout.addWidget(self.fullscreen_btn)
        
        layout.addLayout(controls_layout)
        
        # Timeline slider
        timeline_layout = QHBoxLayout()
        timeline_layout.addWidget(QLabel("Timeline:"))
        
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(max(0, self.tracker_manager.total_frames - 1))
        self.timeline_slider.setValue(0)
        self.timeline_slider.valueChanged.connect(self._on_slider_changed)
        timeline_layout.addWidget(self.timeline_slider)
        
        layout.addLayout(timeline_layout)
        
        # Fast frame navigation
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(QLabel("Quick Navigation:"))
        
        # Jump buttons
        self.jump_minus_100_btn = QPushButton("-100")
        self.jump_minus_100_btn.clicked.connect(lambda: self._jump_frames(-100))
        nav_layout.addWidget(self.jump_minus_100_btn)
        
        self.jump_minus_10_btn = QPushButton("-10")
        self.jump_minus_10_btn.clicked.connect(lambda: self._jump_frames(-10))
        nav_layout.addWidget(self.jump_minus_10_btn)
        
        # Frame number input
        nav_layout.addWidget(QLabel("Frame:"))
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setMinimum(1)
        self.frame_spinbox.setMaximum(max(1, self.tracker_manager.total_frames))
        self.frame_spinbox.setValue(1)
        self.frame_spinbox.valueChanged.connect(self._on_frame_spinbox_changed)
        nav_layout.addWidget(self.frame_spinbox)
        
        nav_layout.addWidget(QLabel(f"/ {self.tracker_manager.total_frames}"))
        
        self.jump_plus_10_btn = QPushButton("+10")
        self.jump_plus_10_btn.clicked.connect(lambda: self._jump_frames(10))
        nav_layout.addWidget(self.jump_plus_10_btn)
        
        self.jump_plus_100_btn = QPushButton("+100")
        self.jump_plus_100_btn.clicked.connect(lambda: self._jump_frames(100))
        nav_layout.addWidget(self.jump_plus_100_btn)
        
        nav_layout.addStretch()
        layout.addLayout(nav_layout)
        
        # Fix Tracking section (NEW!)
        fix_tracking_layout = QVBoxLayout()
        fix_tracking_layout.addWidget(QLabel("═══ Fix Tracking ═══"))
        
        fix_tracking_hint = QLabel("If tracking jumped to wrong player, draw a new bbox on current frame to restart tracking from here.")
        fix_tracking_hint.setStyleSheet("color: gray; font-size: 11px; font-style: italic;")
        fix_tracking_hint.setWordWrap(True)
        fix_tracking_layout.addWidget(fix_tracking_hint)
        
        fix_tracking_buttons = QHBoxLayout()
        self.fix_tracking_btn = QPushButton("🔧 Mark Fix Point")
        self.fix_tracking_btn.clicked.connect(self._start_fix_tracking)
        self.fix_tracking_btn.setToolTip("Draw a new bbox on current frame to mark where tracking should be corrected.")
        fix_tracking_buttons.addWidget(self.fix_tracking_btn)

        self.resume_tracking_btn = QPushButton("▶ Resume Tracking from Fix Point")
        self.resume_tracking_btn.clicked.connect(self._resume_tracking_from_fix)
        self.resume_tracking_btn.setToolTip("Resume tracking from the marked fix point. Previous tracking data will be preserved.")
        self.resume_tracking_btn.setEnabled(False)  # Disabled until fix point is marked
        fix_tracking_buttons.addWidget(self.resume_tracking_btn)

        fix_tracking_layout.addLayout(fix_tracking_buttons)

        # Re-track button (NEW!)
        retrack_button_layout = QHBoxLayout()
        self.retrack_btn = QPushButton("🔄 Re-track Entire Video")
        self.retrack_btn.clicked.connect(self._on_retrack_requested)
        self.retrack_btn.setToolTip("Re-run tracking on this video from scratch with all learning frames. This will replace the current tracking results.")
        self.retrack_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 8px;")
        retrack_button_layout.addWidget(self.retrack_btn)
        fix_tracking_layout.addLayout(retrack_button_layout)
        
        layout.addLayout(fix_tracking_layout)
        
        # Approval section
        approval_layout = QVBoxLayout()
        approval_layout.addWidget(QLabel("═══ Export Approval ═══"))
        
        self.approval_checkbox = QCheckBox("✅ I approve this tracking - ready to export")
        self.approval_checkbox.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.approval_checkbox.stateChanged.connect(self._on_approval_changed)
        approval_layout.addWidget(self.approval_checkbox)
        
        approval_hint = QLabel("Review the tracking carefully. Only approved videos will be exported.")
        approval_hint.setStyleSheet("color: gray; font-size: 11px; font-style: italic;")
        approval_hint.setWordWrap(True)
        approval_layout.addWidget(approval_hint)
        
        layout.addLayout(approval_layout)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        self.cancel_btn = QPushButton("❌ Cancel (Don't Export)")
        self.cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(self.cancel_btn)
        
        self.export_btn = QPushButton("📤 Proceed to Export")
        self.export_btn.clicked.connect(self._on_export_clicked)
        self.export_btn.setEnabled(False)  # Disabled until approved
        self.export_btn.setStyleSheet("font-weight: bold;")
        buttons_layout.addWidget(self.export_btn)
        
        layout.addLayout(buttons_layout)
        
        self.setLayout(layout)
    
    def _setup_timer(self):
        """Setup playback timer"""
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._playback_tick)
        
        # Calculate interval based on FPS
        fps = self.tracker_manager.fps if self.tracker_manager.fps > 0 else 30
        interval_ms = int(1000 / fps)
        self.playback_timer.setInterval(interval_ms)
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Space: Play/Pause
        QShortcut(Qt.Key.Key_Space, self, self._toggle_play)
        
        # Left/Right arrows: Previous/Next frame
        QShortcut(Qt.Key.Key_Left, self, self._prev_frame)
        QShortcut(Qt.Key.Key_Right, self, self._next_frame)
        
        # Alt+Left/Right: Jump 10 frames
        QShortcut(QKeySequence("Alt+Left"), self, lambda: self._jump_frames(-10))
        QShortcut(QKeySequence("Alt+Right"), self, lambda: self._jump_frames(10))
        
        # Ctrl+Left/Right: Jump 100 frames
        QShortcut(QKeySequence("Ctrl+Left"), self, lambda: self._jump_frames(-100))
        QShortcut(QKeySequence("Ctrl+Right"), self, lambda: self._jump_frames(100))
        
        # Page Up/Down: Jump 100 frames
        QShortcut(QKeySequence(Qt.Key.Key_PageUp), self, lambda: self._jump_frames(-100))
        QShortcut(QKeySequence(Qt.Key.Key_PageDown), self, lambda: self._jump_frames(100))
        
        # Home/End: Jump to first/last frame
        QShortcut(QKeySequence(Qt.Key.Key_Home), self, lambda: self._jump_to_frame(0))
        QShortcut(QKeySequence(Qt.Key.Key_End), self, self._jump_to_end)
        
        # F11 or F: Fullscreen
        QShortcut(Qt.Key.Key_F11, self, self._toggle_fullscreen)
        QShortcut(Qt.Key.Key_F, self, self._toggle_fullscreen)
        
        # Escape: Exit fullscreen
        QShortcut(Qt.Key.Key_Escape, self, self._exit_fullscreen)
    
    def _load_frame(self, frame_idx: int):
        """Load and display frame with tracking overlay"""
        if frame_idx < 0 or frame_idx >= self.tracker_manager.total_frames:
            return
        
        self.current_frame_idx = frame_idx
        
        # Get frame
        frame = self.tracker_manager.get_frame(frame_idx)
        if frame is None:
            return
        
        # Get tracking results
        players = self.tracker_manager.get_all_players()
        for player in players:
            stored_bbox = self.tracker_manager.get_bbox_at_frame(
                player.player_id, frame_idx
            )
            # CRITICAL: Always update current_bbox - set to None if no tracking data for this frame
            # This prevents showing bbox from a different frame
            player.current_bbox = stored_bbox

            # Calculate current_original_bbox from stored_bbox using padding offset
            if stored_bbox is not None and hasattr(player, 'padding_offset') and player.padding_offset != (0, 0, 0, 0):
                x, y, w, h = stored_bbox
                offset_x, offset_y, offset_w, offset_h = player.padding_offset
                # Reverse the padding: original = padded + offset
                orig_x = x + offset_x
                orig_y = y + offset_y
                orig_w = w - offset_w
                orig_h = h - offset_h
                player.current_original_bbox = (orig_x, orig_y, orig_w, orig_h)
            else:
                player.current_original_bbox = stored_bbox
        
        # Draw overlays only if frame is in tracking range
        frame_with_overlay = self.overlay_renderer.draw_all_markers(
            frame, 
            players,
            frame_idx=frame_idx,
            tracking_start_frame=self.tracking_start_frame,
            tracking_end_frame=self.tracking_end_frame
        )
        
        # Display
        self.video_canvas.set_frame(frame_with_overlay)
        
        # Update info
        self.frame_info.setText(
            f"Frame: {frame_idx + 1} / {self.tracker_manager.total_frames}"
        )
        
        # Update slider
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(frame_idx)
        self.timeline_slider.blockSignals(False)
        
        # Update spinbox (block signals to avoid recursion)
        if hasattr(self, 'frame_spinbox'):
            self.frame_spinbox.blockSignals(True)
            self.frame_spinbox.setValue(frame_idx + 1)  # Spinbox is 1-indexed
            self.frame_spinbox.blockSignals(False)
    
    def _toggle_play(self):
        """Toggle play/pause"""
        if self.is_playing:
            self._pause()
        else:
            self._play()
    
    def _play(self):
        """Start playback"""
        self.is_playing = True
        self.play_btn.setText("⏸ Pause")
        self.playback_timer.start()
    
    def _pause(self):
        """Pause playback"""
        self.is_playing = False
        self.play_btn.setText("▶ Play")
        self.playback_timer.stop()
    
    def _playback_tick(self):
        """Playback timer tick"""
        next_frame = self.current_frame_idx + 1
        if next_frame >= self.tracker_manager.total_frames:
            # Loop back to start
            next_frame = 0
        
        self._load_frame(next_frame)
    
    def _prev_frame(self):
        """Go to previous frame"""
        self._pause()
        prev_frame = max(0, self.current_frame_idx - 1)
        self._load_frame(prev_frame)
    
    def _next_frame(self):
        """Go to next frame"""
        self._pause()
        next_frame = min(self.tracker_manager.total_frames - 1, self.current_frame_idx + 1)
        self._load_frame(next_frame)
    
    def _on_slider_changed(self, value: int):
        """Handle slider value change"""
        self._pause()
        self._load_frame(value)
    
    def _on_frame_spinbox_changed(self, value: int):
        """Handle frame number spinbox change"""
        frame_idx = value - 1  # Convert from 1-indexed to 0-indexed
        if frame_idx != self.current_frame_idx:
            self._pause()
            self._load_frame(frame_idx)
    
    def _jump_frames(self, offset: int):
        """Jump forward or backward by specified number of frames"""
        new_frame = self.current_frame_idx + offset
        new_frame = max(0, min(new_frame, self.tracker_manager.total_frames - 1))
        if new_frame != self.current_frame_idx:
            self._pause()
            self._load_frame(new_frame)
    
    def _jump_to_frame(self, frame_idx: int):
        """Jump to specific frame index"""
        frame_idx = max(0, min(frame_idx, self.tracker_manager.total_frames - 1))
        if frame_idx != self.current_frame_idx:
            self._pause()
            self._load_frame(frame_idx)
    
    def _jump_to_end(self):
        """Jump to last frame"""
        last_frame = self.tracker_manager.total_frames - 1
        if last_frame >= 0 and last_frame != self.current_frame_idx:
            self._pause()
            self._load_frame(last_frame)
    
    def _toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.is_fullscreen:
            self._exit_fullscreen()
        else:
            self._enter_fullscreen()
    
    def _enter_fullscreen(self):
        """Enter fullscreen mode"""
        self.is_fullscreen = True
        self.showFullScreen()
        self.fullscreen_btn.setText("🗗 Exit Fullscreen")
    
    def _exit_fullscreen(self):
        """Exit fullscreen mode"""
        if self.is_fullscreen:
            self.is_fullscreen = False
            self.showNormal()
            self.fullscreen_btn.setText("🖵 Fullscreen")
    
    def _on_approval_changed(self, state):
        """Handle approval checkbox change"""
        self.approved = (state == Qt.CheckState.Checked.value)
        self.export_btn.setEnabled(self.approved)
        
        if self.approved:
            self.export_btn.setStyleSheet(
                "background-color: #4CAF50; color: white; font-weight: bold;"
            )
        else:
            self.export_btn.setStyleSheet("font-weight: bold;")
    
    def _on_export_clicked(self):
        """Handle export button click"""
        if self.approved:
            self.export_approved.emit()
            self.accept()
    
    def _start_fix_tracking(self):
        """Start fix tracking mode - user will draw bbox on current frame"""
        players = self.tracker_manager.get_all_players()
        if not players:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Players", "No players to fix tracking for.")
            return

        if len(players) == 1:
            # Only one player - use it
            self._fix_player_id = players[0].player_id
            player_name = players[0].name
            self._waiting_for_fix_bbox = True
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "תיקון מעקב - Draw Fix Point",
                f"צייר מלבן חדש סביב '{player_name}' בפריים הנוכחי (פריים {self.current_frame_idx + 1}).\n\n"
                f"Draw a new bounding box around the player to mark the correct position.\n"
                f"Tracking will be corrected from this point forward."
            )
        else:
            # Multiple players - let user choose
            from PyQt6.QtWidgets import QInputDialog
            player_names = [p.name for p in players]
            player_name, ok = QInputDialog.getItem(
                self,
                "Select Player",
                "באיזה שחקן רוצה לתקן את המעקב? Which player to fix?",
                player_names,
                0,
                False
            )
            if ok and player_name:
                # Find player by name
                for player in players:
                    if player.name == player_name:
                        self._fix_player_id = player.player_id
                        self._waiting_for_fix_bbox = True
                        from PyQt6.QtWidgets import QMessageBox
                        QMessageBox.information(
                            self,
                            "תיקון מעקב - Draw Fix Point",
                            f"צייר מלבן חדש סביב '{player_name}' בפריים הנוכחי (פריים {self.current_frame_idx + 1}).\n\n"
                            f"Draw a new bounding box around the player to mark the correct position.\n"
                            f"Tracking will be corrected from this point forward."
                        )
                        break
    
    def _on_fix_bbox_selected(self, x: int, y: int, w: int, h: int):
        """Handle bbox selection for fixing tracking"""
        if not self._waiting_for_fix_bbox:
            return
        
        if self._fix_player_id is None:
            return
        
        # Store fix point (don't emit signal yet - user will click "Resume Tracking")
        self._fix_frame_idx = self.current_frame_idx
        self._fix_bbox = (x, y, w, h)
        
        # Reset waiting state
        self._waiting_for_fix_bbox = False
        
        # Enable resume button
        self.resume_tracking_btn.setEnabled(True)
        
        # Find player name for message
        player_name = "Unknown"
        for player in self.tracker_manager.get_all_players():
            if player.player_id == self._fix_player_id:
                player_name = player.name
                break
        
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "Fix Point Marked",
            f"Fix point marked for '{player_name}' at frame {self.current_frame_idx + 1}.\n\n"
            f"Previous tracking data up to this frame will be preserved.\n"
            f"Click 'Resume Tracking from Fix Point' to continue tracking from here."
        )
    
    def _resume_tracking_from_fix(self):
        """Resume tracking from the marked fix point"""
        if self._fix_frame_idx is None or self._fix_bbox is None or self._fix_player_id is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Fix Point", "Please mark a fix point first.")
            return

        # Emit signal to resume tracking
        self.resume_tracking_requested.emit(
            self._fix_frame_idx,
            self._fix_bbox[0], self._fix_bbox[1], self._fix_bbox[2], self._fix_bbox[3],
            self._fix_player_id
        )

        # Reset fix point
        self._fix_frame_idx = None
        self._fix_bbox = None
        self._fix_player_id = None
        self.resume_tracking_btn.setEnabled(False)

    def _on_retrack_requested(self):
        """Handle re-track entire video request"""
        from PyQt6.QtWidgets import QMessageBox

        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Re-track Video",
            "This will re-run tracking on the entire video with all marked learning frames.\n\n"
            "Current tracking results will be replaced.\n\n"
            "Are you sure you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Emit signal to trigger re-tracking
            self.retrack_requested.emit()
            # Close preview dialog - main window will handle re-tracking and show new preview
            self.reject()

    def is_approved(self) -> bool:
        """Check if user approved export"""
        return self.approved
    
    def closeEvent(self, event):
        """Handle dialog close"""
        self._pause()
        super().closeEvent(event)

