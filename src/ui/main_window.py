"""
Main Window - Main application window
"""
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QListWidget,
                             QListWidgetItem, QProgressBar, QMessageBox,
                             QGroupBox, QSizePolicy, QDialog, QSlider,
                             QSpinBox, QLineEdit, QComboBox, QApplication, QScrollArea)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from typing import Optional, Tuple
import cv2
import numpy as np
from pathlib import Path
import os

from ..tracking.tracker_manager import TrackerManager
from ..tracking.project_manager import ProjectManager
from ..tracking.video_project import VideoProject, ProjectStatus
from ..tracking.person_detector import PersonDetector
from ..render.video_exporter import VideoExporter
from ..render.batch_exporter import BatchExportThread
from .video_canvas import VideoCanvas
from .player_selector import PlayerSelector
from .preview_dialog import PreviewDialog
from .batch_preview_dialog import BatchPreviewDialog


class CollapsibleSection(QWidget):
    """Simple collapsible section with chevron toggle"""
    def __init__(self, title: str, content: QWidget, default_open: bool = True):
        super().__init__()
        self.content = content
        self.toggle_btn = QPushButton()
        self.toggle_btn.setObjectName("sectionToggle")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(default_open)
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._title = title
        self._update_title()
        self.toggle_btn.clicked.connect(self._on_toggled)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.toggle_btn)
        layout.addWidget(self.content)
        self.setLayout(layout)
        self.content.setVisible(default_open)

    def _update_title(self):
        chevron = "▼" if self.toggle_btn.isChecked() else "▶"
        self.toggle_btn.setText(f"{chevron}  {self._title}")

    def _on_toggled(self, checked: bool):
        self.content.setVisible(checked)
        self._update_title()


class ExportThread(QThread):
    """Thread for running export process"""
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, tracker_manager: TrackerManager, video_path: str, output_path: str,
                 tracking_start_frame: Optional[int] = None, tracking_end_frame: Optional[int] = None):
        super().__init__()
        self.tracker_manager = tracker_manager
        self.video_path = video_path
        self.output_path = output_path
        self.tracking_start_frame = tracking_start_frame
        self.tracking_end_frame = tracking_end_frame
        self.cancelled = False
    
    def run(self):
        """Run export process"""
        try:
            from ..render.video_exporter import VideoExporter
            exporter = VideoExporter(self.tracker_manager)
            
            def progress_callback(current: int, total: int):
                self.progress.emit(current, total)
            
            success = exporter.export_video(
                self.video_path,
                self.output_path,
                progress_callback,
                self.tracking_start_frame,
                self.tracking_end_frame
            )
            
            if success:
                self.finished.emit(True, "Export completed successfully")
            else:
                self.finished.emit(False, "Export failed")
        except Exception as e:
            import traceback
            error_msg = f"Error during export: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.finished.emit(False, f"Error during export: {str(e)}")
    
    def cancel(self):
        """Cancel export process"""
        self.cancelled = True


class TrackingThread(QThread):
    """Thread for running tracking process"""
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(bool, str)  # success, message
    tracking_lost = pyqtSignal(int, str, int)  # player_id, player_name, frame_idx
    need_user_input = pyqtSignal(int, int, str, int)  # player_id, frame_idx, player_name, reason_code
    
    def __init__(self, tracker_manager: TrackerManager, video_path: str, trim_start: Optional[int] = None, trim_end: Optional[int] = None):
        super().__init__()
        self.tracker_manager = tracker_manager
        self.video_path = video_path
        self.trim_start = trim_start
        self.trim_end = trim_end
        self.cancelled = False
    
    def run(self):
        """Run tracking process"""
        cap = None
        try:
            players = self.tracker_manager.get_all_players()
            if not players:
                self.finished.emit(False, "No players to track")
                return
            
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.finished.emit(False, "Failed to open video")
                return
            
            total_frames = self.tracker_manager.total_frames
            if total_frames <= 0:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total_frames <= 0:
                self.finished.emit(False, "Invalid frame count")
                cap.release()
                return
            
            # Apply tracking range (start and end frames)
            start_frame = self.trim_start if self.trim_start is not None else 0
            end_frame = self.trim_end if self.trim_end is not None else (total_frames - 1)
            
            # Validate range
            if start_frame < 0:
                start_frame = 0
            if start_frame >= total_frames:
                start_frame = total_frames - 1
            if end_frame >= total_frames:
                end_frame = total_frames - 1
            if start_frame > end_frame:
                self.finished.emit(False, f"Invalid tracking range: start ({start_frame}) > end ({end_frame})")
                cap.release()
                return
            
            # Start from tracking start frame
            # CRITICAL: Use reliable frame seeking method
            frame_idx = start_frame
            tracking_frames = end_frame - start_frame + 1
            
            if start_frame > 0 or end_frame < total_frames - 1:
                print(f"🎬 Tracking range: frames {start_frame}-{end_frame} ({tracking_frames} frames)")
            else:
                print(f"🎬 Tracking from beginning to end ({total_frames} frames)")
            
            # Seek to start_frame using reliable method
            if start_frame > 0:
                # Try to seek directly
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                # If seek failed, read sequentially from beginning
                if actual_pos != start_frame:
                    print(f"⚠️ Seek to frame {start_frame} failed (actual: {actual_pos}), reading sequentially...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    for i in range(start_frame):
                        ret, _ = cap.read()
                        if not ret:
                            print(f"❌ ERROR: Failed to read to frame {start_frame}")
                            self.finished.emit(False, f"Failed to seek to start frame {start_frame}")
                            cap.release()
                            return
                else:
                    print(f"✅ Successfully seeked to frame {start_frame}")
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Track from start_frame to end_frame (only this range will have tracking)
            while frame_idx <= end_frame:
                if self.cancelled:
                    break
                
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"⚠️ WARNING: Failed to read frame {frame_idx}")
                    break
                
                for player in players:
                    # Check if we should initialize tracker for this player
                    # Initialize if:
                    # 1. Tracker not yet initialized
                    # 2. We've reached the player's initial_frame (where they were marked)
                    # 3. We're at or past the tracking start_frame
                    should_init = (not player.tracker.is_initialized and 
                                 frame_idx >= player.initial_frame and
                                 frame_idx >= start_frame)
                    
                    if should_init:
                        # Use the best learning frame for initialization
                        # Find the learning frame CLOSEST to the tracking start frame (by distance)
                        best_learning_frame = None
                        best_learning_bbox = None
                        min_distance = float('inf')
                        
                        # Sort learning frames by frame index (ascending)
                        learning_frames_sorted = sorted(player.learning_frames.items())
                        
                        # Find the learning frame closest to start_frame by distance
                        for learn_frame_idx, learn_bbox in learning_frames_sorted:
                            distance = abs(learn_frame_idx - start_frame)
                            if distance < min_distance:
                                min_distance = distance
                                best_learning_frame = learn_frame_idx
                                best_learning_bbox = learn_bbox
                        
                        # If no learning frames found, use player.bbox
                        if best_learning_bbox is None:
                            best_learning_bbox = player.bbox
                            best_learning_frame = player.initial_frame
                            print(f"⚠️ WARNING: No learning frames found, using initial bbox from frame {player.initial_frame}")
                        
                        # Warn if learning frame is too far from start frame (more than 50 frames)
                        if min_distance > 50:
                            print(f"⚠️ WARNING: Learning frame {best_learning_frame} is {min_distance} frames away from tracking start ({start_frame})")
                            print(f"   This may cause tracking to fail or track the wrong player!")
                            print(f"   Consider marking the player again near frame {start_frame} for better accuracy.")
                        
                        print(f"🔵 Initializing tracker for player {player.player_id} at frame {frame_idx}")
                        print(f"   Using learning frame {best_learning_frame} with bbox: {best_learning_bbox}")
                        print(f"   Distance from start: {min_distance} frames")
                        print(f"   Total learning frames: {len(player.learning_frames)}")
                        print(f"   Learning frames: {sorted(player.learning_frames.keys())}")
                        print(f"   Tracking start: {start_frame}, Current frame: {frame_idx}")
                        
                        # Validate bbox before initializing
                        if best_learning_bbox is None or len(best_learning_bbox) != 4:
                            print(f"❌ ERROR: Invalid bbox for player {player.player_id}: {best_learning_bbox}")
                            player.tracking_lost = True
                            continue
                        
                        x, y, w, h = best_learning_bbox
                        if w <= 0 or h <= 0:
                            print(f"❌ ERROR: Invalid bbox size for player {player.player_id}: {w}x{h}")
                            player.tracking_lost = True
                            continue
                        
                        # Check if bbox is within frame bounds
                        frame_h, frame_w = frame.shape[:2]
                        if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
                            print(f"⚠️ WARNING: Bbox {best_learning_bbox} is outside frame bounds ({frame_w}x{frame_h})")
                            print(f"   Clamping bbox to frame bounds...")
                            x = max(0, min(x, frame_w - 1))
                            y = max(0, min(y, frame_h - 1))
                            w = min(w, frame_w - x)
                            h = min(h, frame_h - y)
                            best_learning_bbox = (x, y, w, h)
                            print(f"   Adjusted bbox: {best_learning_bbox}")
                        
                        init_success = player.tracker.init_tracker(frame, best_learning_bbox)
                        player.tracking_lost = not init_success
                        if init_success:
                            player.current_bbox = best_learning_bbox
                            if player.player_id not in self.tracker_manager.tracking_results:
                                self.tracker_manager.tracking_results[player.player_id] = {}
                            self.tracker_manager.tracking_results[player.player_id][frame_idx] = best_learning_bbox
                            print(f"✅ Player {player.player_id} initialized at frame {frame_idx}, bbox={best_learning_bbox}")
                        else:
                            print(f"❌ ERROR: Failed to initialize tracker for player {player.player_id}")
                            print(f"   This may be due to:")
                            print(f"   - Incorrect bbox coordinates")
                            print(f"   - Frame mismatch (initializing at wrong frame)")
                            print(f"   - Tracker initialization error")
                
                # Update initialized trackers
                for player in players:
                    if not player.tracker.is_initialized:
                        continue

                    # Only update tracker if it's initialized and we're past the initial frame
                    if player.tracker.is_initialized and frame_idx > player.initial_frame:
                        # Check if this frame is a learning frame - if so, reinitialize tracker
                        if frame_idx in player.learning_frames:
                            # This is a learning frame! Use the exact bbox from learning frame
                            learning_bbox = player.learning_frames[frame_idx]

                            print(f"🔄 LEARNING FRAME DETECTED! Frame {frame_idx}")
                            print(f"   Player {player.player_id}: Reinitializing tracker with bbox={learning_bbox}")

                            # Reinitialize tracker with the correct bbox from learning frame
                            player.tracker.init_tracker(frame, learning_bbox)
                            bbox = learning_bbox

                            # Also update current_original_bbox from original_learning_frames
                            if frame_idx in player.original_learning_frames:
                                player.current_original_bbox = player.original_learning_frames[frame_idx]
                                print(f"   Updated current_original_bbox to {player.current_original_bbox}")
                            else:
                                # Fallback: calculate from padded bbox
                                if player.padding_offset != (0, 0, 0, 0):
                                    x, y, w, h = bbox
                                    offset_x, offset_y, offset_w, offset_h = player.padding_offset
                                    orig_x = x + offset_x
                                    orig_y = y + offset_y
                                    orig_w = w - offset_w
                                    orig_h = h - offset_h
                                    player.current_original_bbox = (orig_x, orig_y, orig_w, orig_h)
                                else:
                                    player.current_original_bbox = bbox
                        else:
                            # Normal tracking update
                            bbox = player.tracker.update(frame)

                            # Calculate current_original_bbox from current_bbox using padding offset
                            if bbox is not None and player.padding_offset != (0, 0, 0, 0):
                                x, y, w, h = bbox
                                offset_x, offset_y, offset_w, offset_h = player.padding_offset
                                # Reverse the padding: original = padded + offset
                                orig_x = x + offset_x
                                orig_y = y + offset_y
                                orig_w = w - offset_w
                                orig_h = h - offset_h
                                player.current_original_bbox = (orig_x, orig_y, orig_w, orig_h)
                            else:
                                player.current_original_bbox = bbox

                        player.current_bbox = bbox
                        was_tracking_lost = player.tracking_lost
                        player.tracking_lost = (bbox is None)
                        
                        # Check if tracking was just lost
                        if player.tracking_lost and not was_tracking_lost:
                            # Tracking just lost - emit signal and wait for user input
                            print(f"⚠️ Tracking lost for player {player.player_id} at frame {frame_idx}")
                            
                            # Store current frame for potential reinitialization
                            current_frame_for_reinit = frame.copy()
                            
                            # Emit signal to ask user for input
                            self.need_user_input.emit(player.player_id, frame_idx, player.name, 1)  # reason_code=1: tracking_lost
                            
                            # Wait for user input (non-blocking - main thread will handle dialog)
                            # We'll check user_input_received flag in next iteration
                            # For now, just continue and let main thread handle the dialog
                            
                            # Also emit tracking_lost signal for notification
                            self.tracking_lost.emit(player.player_id, player.name, frame_idx)
                        
                        # Log first few frames to catch jumps
                        if frame_idx <= player.initial_frame + 5:
                            prev_bbox = self.tracker_manager.tracking_results.get(player.player_id, {}).get(frame_idx - 1)
                            if prev_bbox and bbox:
                                # Check for large jumps
                                dx = abs(bbox[0] - prev_bbox[0])
                                dy = abs(bbox[1] - prev_bbox[1])
                                if dx > 50 or dy > 50:  # Large jump detected
                                    print(f"⚠️ WARNING: Large bbox jump at frame {frame_idx}!")
                                    print(f"   Player {player.player_id}: {prev_bbox} → {bbox}")
                                    print(f"   Jump: dx={dx}, dy={dy}")
                        if frame_idx % 10 == 0:  # Log every 10 frames
                            print(f"Frame {frame_idx}: Player {player.player_id} bbox={bbox}")
                    elif player.tracker.is_initialized and frame_idx == player.initial_frame:
                        # At initial frame - use the stored bbox
                        bbox = player.current_bbox
                    else:
                        # Tracker not initialized yet or before initial frame - no bbox
                        bbox = None
                    
                    # Store result (None if no tracking data for this frame)
                    if player.player_id not in self.tracker_manager.tracking_results:
                        self.tracker_manager.tracking_results[player.player_id] = {}
                    self.tracker_manager.tracking_results[player.player_id][frame_idx] = bbox
                
                # Progress relative to trim range
                progress_frame = frame_idx - start_frame + 1
                self.progress.emit(progress_frame, tracking_frames)
                frame_idx += 1
            
            if cap:
                cap.release()
            
            if frame_idx == 0:
                self.finished.emit(False, "Failed to process video frames")
            else:
                # Debug: Check tracking results
                print(f"\n=== Tracking Complete ===")
                for player in players:
                    results_count = len(self.tracker_manager.tracking_results.get(player.player_id, {}))
                    print(f"Player {player.player_id}: {results_count} frames tracked")
                    # Show first few and last few
                    if player.player_id in self.tracker_manager.tracking_results:
                        frames = sorted(self.tracker_manager.tracking_results[player.player_id].keys())
                        if len(frames) > 0:
                            print(f"  First 3 frames: {frames[:3]}")
                            print(f"  Last 3 frames: {frames[-3:]}")
                            # Show some bboxes
                            print(f"  Frame 0 bbox: {self.tracker_manager.tracking_results[player.player_id].get(0)}")
                            if len(frames) > 10:
                                print(f"  Frame 10 bbox: {self.tracker_manager.tracking_results[player.player_id].get(10)}")
                            if len(frames) > 20:
                                print(f"  Frame 20 bbox: {self.tracker_manager.tracking_results[player.player_id].get(20)}")
                print(f"=========================\n")
                
                self.finished.emit(True, "Tracking completed successfully")
        
        except Exception as e:
            import traceback
            error_msg = f"Error during tracking: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            if cap:
                cap.release()
            self.finished.emit(False, f"Error during tracking: {str(e)}")
    
    def cancel(self):
        """Cancel tracking process"""
        self.cancelled = True


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Markme - Batch Player Tracking")
        self.setMinimumSize(1400, 900)
        
        # State - NEW: Using ProjectManager for multiple videos
        self.project_manager = ProjectManager()
        self.current_frame_idx = 0
        self._waiting_for_bbox = False
        
        # Threads
        self.tracking_thread = None
        self.export_thread = None
        self.batch_export_thread = None
        self.batch_tracking_thread = None  # For sequential batch tracking
        self._pending_export_projects = []  # Projects waiting for export after tracking
        self.batch_tracking_projects = []  # Projects to track
        self.batch_tracking_index = 0      # Current index
        
        # UI Setup
        self._setup_ui()
        self._apply_modern_theme()
        
        # Setup keyboard shortcuts for frame navigation
        self._setup_keyboard_shortcuts()
        
        # Timer for preview updates
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self._update_preview)
    
    def _setup_ui(self):
        """Setup user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel - Controls
        left_panel = self._create_left_panel()
        self._apply_sidebar_constraints(left_panel)

        # Wrap sidebar in scroll area to avoid squashing
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setWidget(left_panel)
        sidebar_scroll.setMinimumWidth(320)
        sidebar_scroll.setMaximumWidth(380)
        main_layout.addWidget(sidebar_scroll, 0)
        
        # Right panel - Video preview
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, 3)

    def _apply_modern_theme(self):
        """Apply modern dark professional stylesheet"""
        qss = """
        * {
            font-family: "Segoe UI", "SF Pro Display", "Inter", sans-serif;
            color: #E0E0E0;
            font-size: 13px;
        }
        QWidget {
            background-color: #1a1c1f;
            color: #E0E0E0;
        }
        QLabel {
            color: #FFFFFF;
            font-size: 13px;
        }
        QGroupBox {
            border: 1px solid #2a2e34;
            border-radius: 6px;
            margin-top: 8px;
            padding: 10px 10px 12px 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
            color: #FFFFFF;
            font-weight: 700;
        }

        QPushButton {
            background-color: #2b2f34;
            border: 1px solid #3a3f46;
            border-radius: 6px;
            padding: 7px 10px;
            color: #f5f5f5;
            font-weight: 600;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #343941;
            border-color: #4a515b;
        }
        QPushButton:pressed {
            background-color: #25282d;
            border-color: #3a3f46;
        }
        QPushButton:disabled {
            background-color: #1f2226;
            border: 1px solid #2a2f35;
            color: #6c747d;
        }

        QPushButton#startTrackingBtn,
        QPushButton#exportBtn {
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3f8cff, stop:1 #2d6fde);
            border: 1px solid #2b63c2;
            border-radius: 6px;
            padding: 9px 12px;
            color: #ffffff;
            font-weight: 700;
        }
        QPushButton#startTrackingBtn:hover,
        QPushButton#exportBtn:hover {
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4c98ff, stop:1 #2f78e8);
        }
        QPushButton#startTrackingBtn:pressed,
        QPushButton#exportBtn:pressed {
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #357be0, stop:1 #275fb6);
        }

        QPushButton#sectionToggle {
            background-color: transparent;
            border: none;
            text-align: left;
            padding: 6px 8px;
            font-weight: 700;
            color: #cfd3d8;
        }
        QPushButton#sectionToggle:hover {
            background-color: rgba(255, 255, 255, 0.05);
        }

        QPushButton#addBtn,
        QPushButton#removeBtn,
        QPushButton#sidebarAction {
            background-color: transparent;
            border: 1px solid #3a3f46;
            color: #d0d4db;
        }
        QPushButton#addBtn:hover,
        QPushButton#removeBtn:hover,
        QPushButton#sidebarAction:hover {
            background-color: rgba(255, 255, 255, 0.04);
            border-color: #4a515b;
        }

        QWidget#playbackBar {
            background-color: #1f2125;
            border: 1px solid #2c3036;
            border-radius: 6px;
            padding: 4px;
        }
        QWidget#playbackBar QPushButton {
            background-color: #262a2f;
            border: 1px solid #333941;
            border-radius: 4px;
            padding: 8px 10px;
            min-width: 56px;
            color: #f0f2f5;
            font-weight: 600;
        }
        QWidget#playbackBar QPushButton:hover {
            background-color: #2f343c;
            border-color: #3f4550;
        }
        QWidget#playbackBar QPushButton:pressed {
            background-color: #24282e;
        }

        QLineEdit, QComboBox, QSpinBox, QTextEdit, QPlainTextEdit {
            background-color: #2A2A2A;
            border: 1px solid #444444;
            border-radius: 6px;
            padding: 7px 8px;
            color: #FFFFFF;
        }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QTextEdit:focus, QPlainTextEdit:focus {
            border: 1px solid #3f8cff;
        }
        QLineEdit::placeholder, QComboBox::placeholder, QSpinBox::placeholder, QTextEdit[placeholderText]:empty, QPlainTextEdit[placeholderText]:empty {
            color: #888888;
        }

        QListWidget, QTreeWidget, QTableWidget {
            background-color: #1c1f23;
            border: 1px solid #2a2e34;
            border-radius: 6px;
            selection-background-color: #2f3640;
            selection-color: #f5f7fb;
        }

        QProgressBar {
            border: 1px solid #2a2e34;
            border-radius: 6px;
            background-color: #1c1f23;
            text-align: center;
            padding: 2px;
            color: #e8e8e8;
        }
        QProgressBar::chunk {
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3f8cff, stop:1 #2d6fde);
            border-radius: 4px;
        }
        """
        self.setStyleSheet(qss)

    def _apply_sidebar_constraints(self, sidebar: QWidget):
        """Ensure sidebar contents keep size and can scroll on short windows"""
        for widget in sidebar.findChildren((QPushButton, QLineEdit, QComboBox, QSpinBox)):
            widget.setMinimumHeight(40)
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    
    def _create_left_panel(self) -> QWidget:
        """Create left control panel with video list for batch processing"""
        panel = QWidget()
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(380)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Videos List (collapsed by default)
        videos_content = QWidget()
        videos_layout = QVBoxLayout()
        videos_layout.setSpacing(8)
        videos_layout.setContentsMargins(6, 6, 6, 6)
        
        self.videos_list = QListWidget()
        self.videos_list.itemClicked.connect(self._on_video_selected)
        videos_layout.addWidget(self.videos_list)
        
        video_buttons_layout = QHBoxLayout()
        video_buttons_layout.setSpacing(6)
        self.add_videos_btn = QPushButton("➕ Add Videos")
        self.add_videos_btn.setObjectName("addBtn")
        self.add_videos_btn.clicked.connect(self._add_videos)
        video_buttons_layout.addWidget(self.add_videos_btn)
        
        self.remove_video_btn = QPushButton("➖ Remove")
        self.remove_video_btn.setObjectName("removeBtn")
        self.remove_video_btn.clicked.connect(self._remove_video)
        self.remove_video_btn.setEnabled(False)
        video_buttons_layout.addWidget(self.remove_video_btn)
        
        videos_layout.addLayout(video_buttons_layout)
        
        self.video_info_label = QLabel("No videos loaded")
        self.video_info_label.setWordWrap(True)
        self.video_info_label.setStyleSheet("font-size: 12px; color: #FFFFFF;")
        videos_layout.addWidget(self.video_info_label)
        videos_layout.addStretch()
        videos_content.setLayout(videos_layout)
        layout.addWidget(CollapsibleSection("📹 Video List", videos_content, default_open=False))
        
        # Players for current video (open)
        players_content = QWidget()
        players_layout = QVBoxLayout()
        players_layout.setSpacing(8)
        players_layout.setContentsMargins(6, 6, 6, 6)
        
        self.players_list = QListWidget()
        self.players_list.itemClicked.connect(self._on_player_selected)
        players_layout.addWidget(self.players_list)
        
        player_buttons_layout = QHBoxLayout()
        player_buttons_layout.setSpacing(6)
        self.add_player_btn = QPushButton("➕ Add Marker")
        self.add_player_btn.setObjectName("sidebarAction")
        self.add_player_btn.clicked.connect(self._add_player_marker)
        self.add_player_btn.setEnabled(False)
        player_buttons_layout.addWidget(self.add_player_btn)
        
        self.remove_player_btn = QPushButton("➖ Remove")
        self.remove_player_btn.setObjectName("sidebarAction")
        self.remove_player_btn.clicked.connect(self._remove_player)
        self.remove_player_btn.setEnabled(False)
        player_buttons_layout.addWidget(self.remove_player_btn)
        
        players_layout.addLayout(player_buttons_layout)
        players_layout.addStretch()
        players_content.setLayout(players_layout)
        layout.addWidget(CollapsibleSection("👥 Players (Current Video)", players_content, default_open=True))
        
        # Tracking Section (open)
        tracking_content = QWidget()
        tracking_layout = QVBoxLayout()
        tracking_layout.setSpacing(8)
        tracking_layout.setContentsMargins(6, 6, 6, 6)
        
        self.track_all_btn = QPushButton("▶ Start Tracking All Videos")
        self.track_all_btn.setObjectName("startTrackingBtn")
        self.track_all_btn.clicked.connect(self._track_all_videos)
        self.track_all_btn.setEnabled(False)
        self.track_all_btn.setToolTip("Track all videos with markers before export")
        tracking_layout.addWidget(self.track_all_btn)
        tracking_content.setLayout(tracking_layout)
        layout.addWidget(CollapsibleSection("🎯 Tracking", tracking_content, default_open=True))
        
        # Tracking Range Section (open)
        tracking_range_content = QWidget()
        tracking_range_layout = QVBoxLayout()
        tracking_range_layout.setSpacing(8)
        tracking_range_layout.setContentsMargins(6, 6, 6, 6)
        
        tracking_range_info_layout = QHBoxLayout()
        self.tracking_range_info_label = QLabel("Tracking: Full video")
        self.tracking_range_info_label.setWordWrap(True)
        self.tracking_range_info_label.setStyleSheet("font-size: 12px; color: #FFFFFF;")
        tracking_range_info_layout.addWidget(self.tracking_range_info_label)
        tracking_range_layout.addLayout(tracking_range_info_layout)
        
        # Start/End on same row
        start_end_row = QHBoxLayout()
        start_end_row.setSpacing(8)
        self.set_tracking_start_btn = QPushButton("📍 Start")
        self.set_tracking_start_btn.clicked.connect(self._set_tracking_start)
        self.set_tracking_start_btn.setEnabled(False)
        self.set_tracking_start_btn.setToolTip("Set frame where tracking should start (current frame). Video will play from beginning, but tracking markers will appear only from this frame.")
        start_end_row.addWidget(self.set_tracking_start_btn)
        
        self.set_tracking_end_btn = QPushButton("📍 End")
        self.set_tracking_end_btn.clicked.connect(self._set_tracking_end)
        self.set_tracking_end_btn.setEnabled(False)
        self.set_tracking_end_btn.setToolTip("Set frame where tracking should end (current frame). From this frame to the end, there will be no tracking markers.")
        start_end_row.addWidget(self.set_tracking_end_btn)
        tracking_range_layout.addLayout(start_end_row)
        
        # Clear button
        clear_buttons_layout = QHBoxLayout()
        clear_buttons_layout.setSpacing(8)
        self.clear_tracking_range_btn = QPushButton("🗑️ Clear All")
        self.clear_tracking_range_btn.clicked.connect(self._clear_tracking_range)
        self.clear_tracking_range_btn.setEnabled(False)
        self.clear_tracking_range_btn.setToolTip("Clear start and end frames (tracking will be on full video)")
        clear_buttons_layout.addWidget(self.clear_tracking_range_btn)
        tracking_range_layout.addLayout(clear_buttons_layout)
        
        tracking_range_content.setLayout(tracking_range_layout)
        layout.addWidget(CollapsibleSection("🎯 Tracking Range", tracking_range_content, default_open=True))
        
        # Batch Export (collapsed by default)
        export_content = QWidget()
        export_layout = QVBoxLayout()
        export_layout.setSpacing(8)
        export_layout.setContentsMargins(6, 6, 6, 6)
        
        self.export_all_btn = QPushButton("📤 Export All Videos")
        self.export_all_btn.setObjectName("exportBtn")
        self.export_all_btn.clicked.connect(self._batch_export)
        self.export_all_btn.setEnabled(False)
        export_layout.addWidget(self.export_all_btn)
        
        self.export_single_btn = QPushButton("📤 Export Current Video")
        self.export_single_btn.setObjectName("exportBtn")
        self.export_single_btn.clicked.connect(self._export_single)
        self.export_single_btn.setEnabled(False)
        export_layout.addWidget(self.export_single_btn)
        
        self.cancel_export_btn = QPushButton("❌ Cancel Export")
        self.cancel_export_btn.clicked.connect(self._cancel_export)
        self.cancel_export_btn.setEnabled(False)
        self.cancel_export_btn.setVisible(False)
        export_layout.addWidget(self.cancel_export_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        export_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        export_layout.addWidget(self.status_label)
        
        export_content.setLayout(export_layout)
        layout.addWidget(CollapsibleSection("🎬 Export", export_content, default_open=False))
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create right video preview panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Video canvas
        self.video_canvas = VideoCanvas()
        self.video_canvas.bbox_selected.connect(self._on_bbox_selected)
        self.video_canvas.person_clicked.connect(self._on_person_clicked)
        layout.addWidget(self.video_canvas)
        
        # Frame controls - IMPROVED with slider and jump buttons
        frame_controls = QVBoxLayout()
        
        # Top row: Slider for fast navigation
        slider_row = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setToolTip("Drag to jump to any frame quickly")
        slider_row.addWidget(self.frame_slider)
        frame_controls.addLayout(slider_row)
        
        # Middle row: Frame number input and navigation buttons (toolbar style)
        playback_bar = QWidget()
        playback_bar.setObjectName("playbackBar")
        nav_row = QHBoxLayout()
        nav_row.setContentsMargins(6, 6, 6, 6)
        nav_row.setSpacing(6)
        playback_bar.setLayout(nav_row)
        
        # Jump buttons
        self.jump_back_100_btn = QPushButton("⏪ -100")
        self.jump_back_100_btn.clicked.connect(lambda: self._jump_frames(-100))
        self.jump_back_100_btn.setEnabled(False)
        self.jump_back_100_btn.setToolTip("Jump back 100 frames (Ctrl+Left)")
        nav_row.addWidget(self.jump_back_100_btn)
        
        self.jump_back_10_btn = QPushButton("⏪ -10")
        self.jump_back_10_btn.clicked.connect(lambda: self._jump_frames(-10))
        self.jump_back_10_btn.setEnabled(False)
        self.jump_back_10_btn.setToolTip("Jump back 10 frames (Alt+Left)")
        nav_row.addWidget(self.jump_back_10_btn)
        
        self.prev_frame_btn = QPushButton("◀ -1")
        self.prev_frame_btn.clicked.connect(self._prev_frame)
        self.prev_frame_btn.setEnabled(False)
        self.prev_frame_btn.setToolTip("Previous frame (Left Arrow)")
        nav_row.addWidget(self.prev_frame_btn)
        
        # Frame number input
        frame_input_row = QHBoxLayout()
        frame_input_row.addWidget(QLabel("Frame:"))
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setMinimum(1)
        self.frame_spinbox.setMaximum(1)
        self.frame_spinbox.setValue(1)
        self.frame_spinbox.valueChanged.connect(self._on_frame_number_changed)
        self.frame_spinbox.setEnabled(False)
        self.frame_spinbox.setToolTip("Enter frame number directly (1-based)")
        self.frame_spinbox.setMaximumWidth(80)
        frame_input_row.addWidget(self.frame_spinbox)
        frame_input_row.addWidget(QLabel("/ 0"))
        self.total_frames_label = QLabel("0")
        frame_input_row.addWidget(self.total_frames_label)
        nav_row.addLayout(frame_input_row)
        
        self.next_frame_btn = QPushButton("+1 ▶")
        self.next_frame_btn.clicked.connect(self._next_frame)
        self.next_frame_btn.setEnabled(False)
        self.next_frame_btn.setToolTip("Next frame (Right Arrow)")
        nav_row.addWidget(self.next_frame_btn)
        
        self.jump_forward_10_btn = QPushButton("+10 ⏩")
        self.jump_forward_10_btn.clicked.connect(lambda: self._jump_frames(10))
        self.jump_forward_10_btn.setEnabled(False)
        self.jump_forward_10_btn.setToolTip("Jump forward 10 frames (Alt+Right)")
        nav_row.addWidget(self.jump_forward_10_btn)
        
        self.jump_forward_100_btn = QPushButton("+100 ⏩")
        self.jump_forward_100_btn.clicked.connect(lambda: self._jump_frames(100))
        self.jump_forward_100_btn.setEnabled(False)
        self.jump_forward_100_btn.setToolTip("Jump forward 100 frames (Ctrl+Right)")
        nav_row.addWidget(self.jump_forward_100_btn)
        
        frame_controls.addWidget(playback_bar)
        
        # Bottom row: Fullscreen button
        bottom_row = QHBoxLayout()
        self.fullscreen_btn = QPushButton("🖵 Fullscreen")
        self.fullscreen_btn.clicked.connect(self._toggle_fullscreen)
        self.fullscreen_btn.setEnabled(False)
        self.fullscreen_btn.setToolTip("View video in fullscreen (F11 or F)")
        bottom_row.addWidget(self.fullscreen_btn)
        bottom_row.addStretch()
        frame_controls.addLayout(bottom_row)
        
        layout.addLayout(frame_controls)
        panel.setLayout(layout)
        return panel
    
    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for frame navigation"""
        # Left/Right arrows: Previous/Next frame
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self._prev_frame)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self._next_frame)
        
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
    
    def _jump_to_frame(self, frame_idx: int):
        """Jump to specific frame index"""
        project = self.project_manager.get_current_project()
        if not project:
            return
        
        total = project.tracker_manager.total_frames
        if total <= 0:
            return
        
        frame_idx = max(0, min(frame_idx, total - 1))
        if frame_idx != self.current_frame_idx:
            self.current_frame_idx = frame_idx
            self._show_frame(frame_idx)
    
    def _jump_to_end(self):
        """Jump to last frame"""
        project = self.project_manager.get_current_project()
        if not project:
            return
        
        total = project.tracker_manager.total_frames
        if total <= 0:
            return
        
        last_frame = total - 1
        if last_frame != self.current_frame_idx:
            self.current_frame_idx = last_frame
            self._show_frame(last_frame)
    
    # ===== NEW: Batch Video Management =====
    
    def _add_videos(self):
        """Add multiple videos to the project"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "",
            "Video Files (*.mp4 *.mov *.mkv *.webm);;All Files (*)"
        )
        
        if not file_paths:
            return
        
        total_count = len(file_paths)
        added_count = 0
        failed_count = 0
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total_count)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"📥 Loading videos... (0/{total_count})")
        self.status_label.setStyleSheet("color: blue;")
        
        # Process each video
        for idx, file_path in enumerate(file_paths):
            # Update progress
            self.progress_bar.setValue(idx)
            self.status_label.setText(
                f"📥 Loading videos... ({idx}/{total_count}): {os.path.basename(file_path)}"
            )
            # Process events to update UI
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
            
            project = self.project_manager.add_project(file_path)
            if project:
                # Add to UI list
                item = QListWidgetItem(project.get_display_name())
                item.setData(Qt.ItemDataRole.UserRole, len(self.project_manager.projects) - 1)
                self.videos_list.addItem(item)
                added_count += 1
            else:
                failed_count += 1
        
        # Final update
        self.progress_bar.setValue(total_count)
        self.progress_bar.setVisible(False)
        
        # Update UI
        self._update_buttons()
        self.status_label.setText(f"✅ Added {added_count} videos" + 
                                 (f", ❌ {failed_count} failed" if failed_count > 0 else ""))
        self.status_label.setStyleSheet("color: green;")
    
    def _remove_video(self):
        """Remove selected video from project"""
        current_item = self.videos_list.currentItem()
        if not current_item:
            return
        
        index = current_item.data(Qt.ItemDataRole.UserRole)
        
        # Confirm removal
        project = self.project_manager.get_project(index)
        if project:
            reply = QMessageBox.question(
                self,
                "Remove Video",
                f"Remove {os.path.basename(project.video_path)}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.project_manager.remove_project(index)
                self.videos_list.takeItem(self.videos_list.row(current_item))
                
                # Clear canvas and players list
                self.video_canvas.clear_bboxes()
                self.players_list.clear()
                self.video_info_label.setText("No video selected")
                
                # Update all indices in list
                for i in range(self.videos_list.count()):
                    self.videos_list.item(i).setData(Qt.ItemDataRole.UserRole, i)
                
                self._update_buttons()
    
    def _on_video_selected(self, item: QListWidgetItem):
        """Handle video selection from list"""
        index = item.data(Qt.ItemDataRole.UserRole)
        self.project_manager.set_current_project(index)
        
        project = self.project_manager.get_current_project()
        if not project:
            return
        
        # Display video info
        self.video_info_label.setText(project.get_info_text())
        
        # Clear previous video's bboxes from canvas
        self.video_canvas.clear_bboxes()
        
        # Load first frame
        first_frame = project.tracker_manager.get_first_frame()
        if first_frame is not None:
            self.video_canvas.set_frame(first_frame)
            self.current_frame_idx = 0
        
        # Update players list
        self._update_players_list()
        
        # Show current project's markers if any
        if project.has_players():
            for player in project.get_players():
                if player.current_bbox:
                    self.video_canvas.add_bbox(
                        *player.current_bbox,
                        player.name,
                        player.marker_style,
                        player.color
                    )
        
        # Update buttons and frame info
        self._update_tracking_range_info()
        self._update_buttons()
        self._update_frame_info()
        self._update_frame_navigation_buttons()
        
        self.status_label.setText(f"📹 Loaded: {os.path.basename(project.video_path)}")
        self.status_label.setStyleSheet("color: blue;")
    
    def _update_players_list(self):
        """Update players list for current project"""
        self.players_list.clear()
        
        project = self.project_manager.get_current_project()
        if not project:
            return
        
        for player in project.get_players():
            learning_count = len(player.learning_frames)
            if learning_count > 1:
                item = QListWidgetItem(f"{player.name} ({player.marker_style}) - {learning_count} learning frames")
            else:
                item = QListWidgetItem(f"{player.name} ({player.marker_style})")
            item.setData(Qt.ItemDataRole.UserRole, player.player_id)
            self.players_list.addItem(item)
    
    def _update_buttons(self):
        """Update button states based on current project"""
        has_videos = self.project_manager.get_project_count() > 0
        current_project = self.project_manager.get_current_project()
        has_current = current_project is not None
        has_players = bool(current_project and current_project.has_players())
        has_ready_projects = len(self.project_manager.get_projects_for_export()) > 0
        
        self.remove_video_btn.setEnabled(has_current)
        self.add_player_btn.setEnabled(has_current)
        self.remove_player_btn.setEnabled(has_players)
        
        # Export single button: enabled only if has players and status is MARKED
        can_export_single = has_players and current_project and current_project.status == ProjectStatus.MARKED
        self.export_single_btn.setEnabled(can_export_single)
        
        self.export_all_btn.setEnabled(has_ready_projects)
        
        # Track buttons: enabled if has players
        # track_single_btn removed - tracking happens automatically during export
        self.track_all_btn.setEnabled(has_ready_projects)
        
        # Tracking range buttons: enabled if has current project
        self.set_tracking_start_btn.setEnabled(has_current)
        self.set_tracking_end_btn.setEnabled(has_current)
        self.clear_tracking_range_btn.setEnabled(has_current and (current_project.trim_start_frame is not None or current_project.trim_end_frame is not None))
        
        # Update tracking range info
        self._update_tracking_range_info()
    
    def _track_single_video_internal(self, project):
        """Internal method to track a specific project (used for re-tracking)"""
        if not project or not project.has_players():
            QMessageBox.warning(self, "No Players", "Please mark at least one player before tracking.")
            return

        # Start tracking
        self.status_label.setText("🔄 Re-tracking video...")
        self.progress_bar.setVisible(True)
        
        # Create tracking thread with tracking range
        self.tracking_thread = TrackingThread(
            project.tracker_manager,
            project.video_path,
            project.trim_start_frame,  # Start frame (None = from beginning)
            project.trim_end_frame     # End frame (None = to end)
        )
        
        def on_tracking_complete(success, message):
            if success:
                project.status = ProjectStatus.TRACKED
                self.status_label.setText("✅ Tracking complete!")
                self.status_label.setStyleSheet("color: green;")
                
                # Show preview automatically
                reply = QMessageBox.question(
                    self, 
                    "Tracking Complete",
                    "Tracking completed successfully!\n\n"
                    "Would you like to preview the tracking results?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    _show_preview_for_project(project)
            else:
                QMessageBox.warning(self, "Tracking Failed", f"Tracking failed: {message}")
                self.status_label.setText("❌ Tracking failed")
                self.status_label.setStyleSheet("color: red;")
            
            self.progress_bar.setVisible(False)
            self._update_buttons()
            
            # Clean up thread properly
            if self.tracking_thread is not None:
                try:
                    self.tracking_thread.finished.disconnect()
                    self.tracking_thread.progress.disconnect()
                    self.tracking_thread.deleteLater()
                except:
                    pass
                self.tracking_thread = None
        
        def _show_preview_for_project(proj):
            """Show preview dialog for project"""
            from .preview_dialog import PreviewDialog
            preview = PreviewDialog(proj.tracker_manager, proj.video_path, self)

            # Connect re-track signal
            def on_retrack_requested():
                """Handle re-track request from preview dialog"""
                # Close current preview (already closed by reject())
                # Start tracking on this single project
                self._track_single_video_internal(proj)

            preview.retrack_requested.connect(on_retrack_requested)
            preview.exec()
        
        self.tracking_thread.finished.connect(on_tracking_complete)
        self.tracking_thread.progress.connect(
            lambda current, total: self.progress_bar.setValue(int(current / total * 100))
        )
        
        # Handle tracking lost notifications
        def on_tracking_lost(player_id: int, player_name: str, frame_idx: int):
            QMessageBox.warning(
                self,
                "Tracking Lost",
                f"⚠️ Tracking lost for '{player_name}' at frame {frame_idx + 1}.\n\n"
                f"The system could not identify the player at this frame.\n"
                f"You may need to fix the tracking at this point."
            )
        
        self.tracking_thread.tracking_lost.connect(on_tracking_lost)

        self.tracking_thread.start()

    def _track_single_video(self):
        """Start tracking for current video only"""
        project = self.project_manager.get_current_project()
        if not project or not project.has_players():
            QMessageBox.warning(self, "No Players", "Please mark at least one player before tracking.")
            return

        # Use internal method to do the actual tracking
        self._track_single_video_internal(project)

    def _track_all_videos(self):
        """Start tracking for all videos with players"""
        projects_to_track = [p for p in self.project_manager.projects if p.has_players()]
        
        if not projects_to_track:
            QMessageBox.warning(self, "No Videos", "No videos with markers to track.")
            return
        
        # Confirm
        reply = QMessageBox.question(
            self,
            "Track All Videos",
            f"Track {len(projects_to_track)} video(s)?\n\n"
            f"This will process all videos with markers.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Track each project
        self.status_label.setText(f"🔄 Tracking {len(projects_to_track)} videos...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(100)
        
        # Track sequentially
        self._track_projects_sequentially(projects_to_track, 0)
    
    def _track_projects_sequentially(self, projects, index):
        """Track projects one by one - PROPERLY MANAGED THREADS"""
        # Store projects and index for iteration
        self.batch_tracking_projects = projects
        self.batch_tracking_index = index
        
        # Start tracking the first/next project
        self._start_next_batch_tracking()
    
    def _start_next_batch_tracking(self):
        """Start tracking the next project in batch - with proper thread cleanup"""
        # Check if we're done
        if self.batch_tracking_index >= len(self.batch_tracking_projects):
            # All done!
            self.progress_bar.setValue(100)
            self.progress_bar.setVisible(False)
            self._update_buttons()
            self.status_label.setText("✅ All tracking complete!")
            self.status_label.setStyleSheet("color: green;")
            
            # Check if this is for export workflow
            if hasattr(self, '_pending_export_projects') and self._pending_export_projects:
                # This is for export - don't show message, continue to preview
                self.batch_tracking_projects = []
                self.batch_tracking_index = 0
                return
            
            # Clean up
            self.batch_tracking_projects = []
            self.batch_tracking_index = 0
            
            QMessageBox.information(self, "Tracking Complete", 
                                   f"Successfully tracked all videos!\n\n"
                                   f"You can now export them.")
            return
        
        # Get current project
        project = self.batch_tracking_projects[self.batch_tracking_index]
        self.status_label.setText(
            f"🔄 Tracking {self.batch_tracking_index + 1}/{len(self.batch_tracking_projects)}: "
            f"{project.get_display_name()}..."
        )
        
        # Clean up previous thread if exists
        if self.batch_tracking_thread is not None:
            try:
                self.batch_tracking_thread.finished.disconnect()
                self.batch_tracking_thread.progress.disconnect()
                self.batch_tracking_thread.deleteLater()
            except:
                pass
            self.batch_tracking_thread = None
        
        # Create new tracking thread with tracking range
        self.batch_tracking_thread = TrackingThread(
            project.tracker_manager,
            project.video_path,
            project.trim_start_frame,  # Start frame (None = from beginning)
            project.trim_end_frame     # End frame (None = to end)
        )
        
        # Connect signals
        self.batch_tracking_thread.finished.connect(self._on_batch_tracking_finished)
        self.batch_tracking_thread.progress.connect(self._on_batch_tracking_progress)
        
        # Start tracking
        self.batch_tracking_thread.start()
    
    def _on_batch_tracking_finished(self, success: bool, message: str):
        """Handle completion of one video in batch tracking"""
        # Mark project as tracked if successful
        if success and self.batch_tracking_index < len(self.batch_tracking_projects):
            project = self.batch_tracking_projects[self.batch_tracking_index]
            project.status = ProjectStatus.TRACKED
        
        # Clean up current thread PROPERLY
        if self.batch_tracking_thread is not None:
            try:
                # Wait for thread to finish
                if self.batch_tracking_thread.isRunning():
                    self.batch_tracking_thread.quit()
                    self.batch_tracking_thread.wait(1000)  # Wait up to 1 second
                
                # Disconnect signals
                self.batch_tracking_thread.finished.disconnect()
                self.batch_tracking_thread.progress.disconnect()
                
                # Schedule for deletion
                self.batch_tracking_thread.deleteLater()
            except:
                pass
            
            self.batch_tracking_thread = None
        
        # Move to next project
        self.batch_tracking_index += 1
        
        # Use QTimer to ensure we're not in the middle of signal processing
        QTimer.singleShot(100, self._start_next_batch_tracking)
    
    def _on_batch_tracking_progress(self, current: int, total: int):
        """Update progress bar during batch tracking"""
        if len(self.batch_tracking_projects) == 0:
            return
        
        # Calculate overall progress
        overall_progress = (
            (self.batch_tracking_index / len(self.batch_tracking_projects)) + 
            ((current / total) / len(self.batch_tracking_projects))
        ) * 100
        
        self.progress_bar.setValue(int(overall_progress))
    
    def _batch_export(self):
        """Start batch export: Track all → Preview → Export"""
        projects_to_export = self.project_manager.get_projects_for_export()
        
        if not projects_to_export:
            QMessageBox.warning(self, "No Videos", "No videos are ready for export.\nPlease mark players in at least one video.")
            return
        
        # Check which projects need tracking
        projects_need_tracking = [p for p in projects_to_export if p.status != ProjectStatus.TRACKED]
        projects_already_tracked = [p for p in projects_to_export if p.status == ProjectStatus.TRACKED]
        
        if projects_need_tracking:
            # Need to track first
            reply = QMessageBox.question(
                self,
                "Tracking Required",
                f"{len(projects_need_tracking)} video(s) need tracking.\n"
                f"{len(projects_already_tracked)} video(s) already tracked.\n\n"
                f"Would you like to track all videos first, then preview, then export?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Track all, then preview, then export
                self._track_all_then_preview_then_export(projects_to_export)
            return
        else:
            # All already tracked - go straight to preview
            self._show_preview_then_export(projects_to_export)
    
    def _track_all_then_preview_then_export(self, projects_to_export):
        """Track all videos, then show preview, then export"""
        # Store projects for later use
        self._pending_export_projects = projects_to_export
        
        # Start tracking all projects
        self.status_label.setText(f"🔄 Tracking {len(projects_to_export)} videos before export...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(100)
        
        # Disable buttons
        self.export_all_btn.setEnabled(False)
        self.export_single_btn.setEnabled(False)
        self.add_videos_btn.setEnabled(False)
        self.add_player_btn.setEnabled(False)
        
        # Track sequentially
        self._track_projects_sequentially_for_export(projects_to_export, 0)
    
    def _track_projects_sequentially_for_export(self, projects, index):
        """Track projects one by one for export workflow"""
        # Store projects and index for iteration
        self.batch_tracking_projects = projects
        self.batch_tracking_index = index
        
        # Start tracking the first/next project
        self._start_next_batch_tracking_for_export()
    
    def _start_next_batch_tracking_for_export(self):
        """Start tracking the next project in batch for export workflow"""
        # Check if we're done
        if self.batch_tracking_index >= len(self.batch_tracking_projects):
            # All tracking done! Now show preview
            self.progress_bar.setValue(100)
            self.progress_bar.setVisible(False)
            self.status_label.setText("✅ All tracking complete! Opening preview...")
            self.status_label.setStyleSheet("color: green;")
            
            # Clean up
            self.batch_tracking_projects = []
            self.batch_tracking_index = 0
            
            # Show preview then export
            QTimer.singleShot(500, lambda: self._show_preview_then_export(self._pending_export_projects))
            return
        
        # Get current project
        project = self.batch_tracking_projects[self.batch_tracking_index]
        self.status_label.setText(
            f"🔄 Tracking {self.batch_tracking_index + 1}/{len(self.batch_tracking_projects)}: "
            f"{project.get_display_name()}..."
        )
        
        # Clean up previous thread if exists
        if self.batch_tracking_thread is not None:
            try:
                self.batch_tracking_thread.finished.disconnect()
                self.batch_tracking_thread.progress.disconnect()
                self.batch_tracking_thread.deleteLater()
            except:
                pass
            self.batch_tracking_thread = None
        
        # Create new tracking thread with tracking range
        self.batch_tracking_thread = TrackingThread(
            project.tracker_manager,
            project.video_path,
            project.trim_start_frame,  # Start frame (None = from beginning)
            project.trim_end_frame     # End frame (None = to end)
        )
        
        # Connect signals
        self.batch_tracking_thread.finished.connect(self._on_batch_tracking_finished_for_export)
        self.batch_tracking_thread.progress.connect(self._on_batch_tracking_progress)
        
        # Start tracking
        self.batch_tracking_thread.start()
    
    def _on_batch_tracking_finished_for_export(self, success: bool, message: str):
        """Handle completion of one video in batch tracking for export workflow"""
        # Mark project as tracked if successful
        if success and self.batch_tracking_index < len(self.batch_tracking_projects):
            project = self.batch_tracking_projects[self.batch_tracking_index]
            project.status = ProjectStatus.TRACKED
        
        # Clean up current thread PROPERLY
        if self.batch_tracking_thread is not None:
            try:
                # Wait for thread to finish
                if self.batch_tracking_thread.isRunning():
                    self.batch_tracking_thread.quit()
                    self.batch_tracking_thread.wait(1000)  # Wait up to 1 second
                
                # Disconnect signals
                self.batch_tracking_thread.finished.disconnect()
                self.batch_tracking_thread.progress.disconnect()
                
                # Schedule for deletion
                self.batch_tracking_thread.deleteLater()
            except:
                pass
            
            self.batch_tracking_thread = None
        
        # Move to next project
        self.batch_tracking_index += 1
        
        # Use QTimer to ensure we're not in the middle of signal processing
        QTimer.singleShot(100, self._start_next_batch_tracking_for_export)
    
    def _show_preview_then_export(self, projects_to_export):
        """Show batch preview dialog, then export if approved"""
        # Clear pending export projects
        if hasattr(self, '_pending_export_projects'):
            self._pending_export_projects = []
        
        # Show batch preview dialog
        batch_preview = BatchPreviewDialog(projects_to_export, self)
        
        def on_export_approved(approved_projects):
            # User approved some projects, proceed to export
            self._do_batch_export(approved_projects)
        
        batch_preview.export_approved.connect(on_export_approved)
        
        result = batch_preview.exec()
        
        if result == QDialog.DialogCode.Rejected:
            # User canceled
            self.status_label.setText("❌ Batch export canceled")
            self._update_buttons()
    
    def _do_batch_export(self, projects_to_export):
        """Actually perform batch export after approval"""
        if not projects_to_export:
            return
        
        # Ask for output directory
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not output_dir:
            self.status_label.setText("❌ Export canceled")
            return
        
        # Start batch export thread
        self.batch_export_thread = BatchExportThread(projects_to_export, output_dir)
        self.batch_export_thread.project_started.connect(self._on_batch_project_started)
        self.batch_export_thread.project_progress.connect(self._on_batch_project_progress)
        self.batch_export_thread.project_completed.connect(self._on_batch_project_completed)
        self.batch_export_thread.all_completed.connect(self._on_batch_all_completed)
        
        # Disable buttons
        self.export_all_btn.setEnabled(False)
        self.export_single_btn.setEnabled(False)
        self.add_videos_btn.setEnabled(False)
        self.add_player_btn.setEnabled(False)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(100)
        self.cancel_export_btn.setVisible(True)
        self.cancel_export_btn.setEnabled(True)
        self.status_label.setText(f"🔄 Exporting {len(projects_to_export)} approved videos...")
        self.status_label.setStyleSheet("color: orange;")
        
        self.batch_export_thread.start()
    
    def _cancel_export(self):
        """Cancel ongoing export"""
        if hasattr(self, 'batch_export_thread') and self.batch_export_thread and self.batch_export_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Cancel Export",
                "Are you sure you want to cancel the export?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Stop the thread
                self.batch_export_thread.terminate()
                self.batch_export_thread.wait()
                
                # Reset UI
                self.cancel_export_btn.setVisible(False)
                self.cancel_export_btn.setEnabled(False)
                self.progress_bar.setVisible(False)
                self.status_label.setText("❌ Export canceled by user")
                self.status_label.setStyleSheet("color: red;")
                
                # Re-enable buttons
                self._update_buttons()
    
    def _on_batch_project_started(self, index: int, name: str):
        """Handle batch project start"""
        self.status_label.setText(f"🔄 Processing: {name}")
        self.progress_bar.setValue(0)
    
    def _on_batch_project_progress(self, index: int, current: int, total: int):
        """Handle batch project progress"""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
    
    def _on_batch_project_completed(self, index: int, success: bool, message: str):
        """Handle batch project completion"""
        icon = "✅" if success else "❌"
        print(f"{icon} Project {index}: {message}")
        
        # Update video list display
        if index < self.videos_list.count():
            project = self.project_manager.get_project(index)
            if project:
                item = self.videos_list.item(index)
                item.setText(project.get_display_name())
    
    def _on_batch_all_completed(self, total: int, successful: int, failed: int):
        """Handle batch export completion"""
        self.progress_bar.setVisible(False)
        self.cancel_export_btn.setVisible(False)
        self.cancel_export_btn.setEnabled(False)
        
        message = f"🎬 Batch Export Complete!\n\n"
        message += f"✅ Successful: {successful}/{total}\n"
        if failed > 0:
            message += f"❌ Failed: {failed}/{total}"
        
        self.status_label.setText(f"✅ Done: {successful}/{total} videos")
        self.status_label.setStyleSheet("color: green;")
        
        QMessageBox.information(self, "Batch Export Complete", message)
        
        # Re-enable buttons
        self._update_buttons()
        self.add_videos_btn.setEnabled(True)
        self.add_player_btn.setEnabled(True)
    
    def _export_single(self):
        """Export current video only - with preview first"""
        project = self.project_manager.get_current_project()
        if not project or not project.has_players():
            QMessageBox.warning(self, "No Players", "Please mark at least one player before exporting.")
            return
        
        # Check if tracking is done
        if project.status != ProjectStatus.TRACKED:
            # Need to track first
            reply = QMessageBox.question(
                self,
                "Tracking Required",
                "Tracking has not been completed yet.\n\n"
                "Would you like to track now and then preview before export?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Start tracking, then preview, then export
                self._track_then_preview_single(project)
            return
        
        # Tracking is done, show preview
        self._show_preview_and_export_single(project)
    
    def _track_then_preview_single(self, project):
        """Track video, then show preview, then export"""
        # Start tracking
        self.status_label.setText("🔄 Tracking...")
        self.progress_bar.setVisible(True)
        
        # Create tracking thread with trim range
        # Create tracking thread with tracking range
        self.tracking_thread = TrackingThread(
            project.tracker_manager,
            project.video_path,
            project.trim_start_frame,  # Start frame (None = from beginning)
            project.trim_end_frame     # End frame (None = to end)
        )
        
        # Connect to preview after tracking completes
        def on_tracking_complete(success, message):
            if success:
                project.status = ProjectStatus.TRACKED
                self.status_label.setText("✅ Tracking complete! Opening preview...")
                self._show_preview_and_export_single(project)
            else:
                QMessageBox.warning(self, "Tracking Failed", f"Tracking failed: {message}")
                self.status_label.setText("❌ Tracking failed")
                self.progress_bar.setVisible(False)
        
        self.tracking_thread.finished.connect(on_tracking_complete)
        self.tracking_thread.progress.connect(
            lambda current, total: self.progress_bar.setValue(int(current / total * 100))
        )
        
        # Handle tracking lost notifications
        def on_tracking_lost(player_id: int, player_name: str, frame_idx: int):
            QMessageBox.warning(
                self,
                "Tracking Lost",
                f"⚠️ Tracking lost for '{player_name}' at frame {frame_idx + 1}.\n\n"
                f"The system could not identify the player at this frame.\n"
                f"You may need to fix the tracking at this point."
            )
        
        self.tracking_thread.tracking_lost.connect(on_tracking_lost)
        
        self.tracking_thread.start()
    
    def _show_preview_and_export_single(self, project):
        """Show preview dialog and export if approved"""
        self.progress_bar.setVisible(False)
        self.status_label.setText("📺 Preview - Review and approve...")
        
        # Show preview dialog with tracking range
        preview = PreviewDialog(
            project.tracker_manager, 
            project.video_path, 
            self,
            project.trim_start_frame,
            project.trim_end_frame
        )
        
        # Handle export approval
        def on_export_approved():
            # User approved, proceed to export
            self._do_export_single(project)
        
        preview.export_approved.connect(on_export_approved)
        
        # Handle fix tracking request (mark fix point - no longer used, but kept for compatibility)
        def on_fix_tracking_requested(frame_idx: int, x: int, y: int, w: int, h: int, player_id: int):
            # This is now handled by resume_tracking_requested
            pass
        
        preview.fix_tracking_requested.connect(on_fix_tracking_requested)
        
        # Handle resume tracking request (actually perform the fix and resume tracking)
        def on_resume_tracking_requested(frame_idx: int, x: int, y: int, w: int, h: int, player_id: int):
            self._resume_tracking_from_frame(project, frame_idx, x, y, w, h, player_id, preview)
        
        preview.resume_tracking_requested.connect(on_resume_tracking_requested)
        
        result = preview.exec()
        
        if result == QDialog.DialogCode.Rejected:
            # User canceled
            self.status_label.setText("❌ Export canceled")
    
    def _do_export_single(self, project):
        """Actually perform the export after approval"""
        # Ask for output file
        output_file, _ = QFileDialog.getSaveFileName(
            self,
            "Save Tracked Video",
            f"{os.path.splitext(os.path.basename(project.video_path))[0]}_tracked.mp4",
            "MP4 Video (*.mp4);;All Files (*)"
        )
        
        if not output_file:
            self.status_label.setText("❌ Export canceled")
            return
        
        # Start batch export with single project
        output_dir = os.path.dirname(output_file)
        self.batch_export_thread = BatchExportThread([project], output_dir)
        self.batch_export_thread.project_started.connect(self._on_batch_project_started)
        self.batch_export_thread.project_progress.connect(self._on_batch_project_progress)
        self.batch_export_thread.project_completed.connect(self._on_batch_project_completed)
        self.batch_export_thread.all_completed.connect(self._on_batch_all_completed)
        
        # Disable buttons
        self.export_single_btn.setEnabled(False)
        self.export_all_btn.setEnabled(False)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(100)
        self.status_label.setText("🔄 Exporting final video...")
        self.status_label.setStyleSheet("color: orange;")
        
        self.batch_export_thread.start()
    
    # ===== END: Batch Video Management =====
    
    def _load_video(self):
        """Load video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.mov *.mkv *.webm);;All Files (*)"
        )
        
        if not file_path:
            return
        
        # Probe video metadata
        metadata = self.tracker_manager.probe_video(file_path)
        if metadata is None:
            QMessageBox.warning(self, "Error", "Failed to read video metadata")
            return
        
        width = int(metadata.get("width", 0))
        height = int(metadata.get("height", 0))
        fps = metadata.get("fps", 30.0)
        duration = metadata.get("duration", 0.0)
        frame_count = int(metadata.get("frame_count", 0))
        
        # Validate duration
        if duration > 60:
            QMessageBox.warning(
                self,
                "Error",
                f"Video duration ({duration:.1f}s) exceeds maximum (60s)"
            )
            return
        
        if frame_count == 0:
            QMessageBox.warning(self, "Error", "Video contains no frames")
            return
        
        # Check resolution (should support up to 4K)
        if width * height > 3840 * 2160:
            QMessageBox.warning(
                self,
                "Warning",
                "Video resolution exceeds 4K. Performance may be affected."
            )
        
        # Load video
        self.video_path = file_path
        self.tracker_manager.reset()
        if not self.tracker_manager.load_video(file_path, metadata):
            QMessageBox.warning(self, "Error", "Failed to load video")
            return
        
        # Display first frame
        first_frame = self.tracker_manager.get_first_frame()
        if first_frame is not None:
            self.video_canvas.set_frame(first_frame)
            self.current_frame_idx = 0
            self._update_frame_info()
        
        # Update UI
        file_name = Path(file_path).name
        self.video_info_label.setText(
            f"File: {file_name}\n"
            f"Resolution: {width}x{height}\n"
            f"FPS: {fps:.2f}\n"
            f"Duration: {duration:.1f}s"
        )
        
        self.add_player_btn.setEnabled(True)
        self.video_canvas.clear_bboxes()
        self.players_list.clear()
        
        # Enable navigation buttons
        self.prev_frame_btn.setEnabled(self.current_frame_idx > 0)
        self.next_frame_btn.setEnabled(self.tracker_manager.total_frames > 1)
        self._update_frame_navigation_buttons()
    
    def _add_player_marker(self):
        """Add a new player marker using automatic person detection"""
        project = self.project_manager.get_current_project()
        if not project:
            QMessageBox.warning(self, "Warning", "Please select a video first.")
            return
        
        # Check if person detector is available
        if not hasattr(self, 'person_detector'):
            self.person_detector = PersonDetector()
        
        if not self.person_detector.is_available():
            reply = QMessageBox.question(
                self,
                "Detection Not Available",
                "Automatic person detection is not available.\n\n"
                "This might be due to:\n"
                "- PyTorch DLL loading issue (common on Windows)\n"
                "- Missing dependencies\n\n"
                "Would you like to:\n"
                "• Use manual selection (draw rectangle) - Recommended\n"
                "• Try to fix detection (requires restart)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Fallback to manual drawing
                self.status_label.setText("✏️ Draw bounding box on player")
                self.status_label.setStyleSheet("color: orange;")
                self._waiting_for_bbox = True
                return
            else:
                # Show instructions to fix
                QMessageBox.information(
                    self,
                    "Fix Detection",
                    "To fix detection, run in terminal:\n\n"
                    "pip uninstall torch torchvision -y\n"
                    "pip install torch torchvision\n\n"
                    "Then restart the application."
                )
                return
        
        # Get current frame
        current_frame = project.tracker_manager.get_frame(self.current_frame_idx)
        if current_frame is None:
            QMessageBox.warning(self, "Error", "Could not load current frame.")
            return
        
        # Show progress
        self.status_label.setText("🔍 Detecting people in frame...")
        self.status_label.setStyleSheet("color: blue;")
        QApplication.processEvents()
        
        # Detect people in current frame
        detections = self.person_detector.detect_people(current_frame, confidence_threshold=0.25)
        
        if not detections:
            self.status_label.setText("No people detected. Try another frame or draw manually.")
            self.status_label.setStyleSheet("color: orange;")
            try:
                self.statusBar().showMessage("No people detected in this frame. Try a different frame or draw manually.", 4000)
            except Exception:
                pass
            return
        
        # Show detected people on canvas
        self.video_canvas.set_detected_people(detections)
        self.video_canvas.enable_detection_mode(True)
        
        # Update status
        num_detected = len(detections)
        self.status_label.setText(f"✅ Found {num_detected} person(s). Click on a person to select them.")
        self.status_label.setStyleSheet("color: green;")
        try:
            self.statusBar().showMessage(f"🔍 System detected {num_detected} players. Click a player or draw a box.", 4000)
        except Exception:
            pass
        self._waiting_for_bbox = True
    
    def _on_person_clicked(self, x: int, y: int, w: int, h: int):
        """Handle clicking on a detected person"""
        print(f"_on_person_clicked called: bbox=({x}, {y}, {w}, {h})")
        
        # Disable detection mode
        self.video_canvas.enable_detection_mode(False)
        
        # Add padding to bbox for better tracking (symmetric)
        padding_x = max(int(w * 0.2), 10)
        padding_y = max(int(h * 0.2), 10)

        # Adjust bbox with padding
        x_padded = max(0, x - padding_x)
        y_padded = max(0, y - padding_y)
        w_padded = w + (padding_x * 2)
        h_padded = h + (padding_y * 2)
        
        # Make sure bbox doesn't exceed frame bounds
        project = self.project_manager.get_current_project()
        if project and project.tracker_manager.frame_width > 0 and project.tracker_manager.frame_height > 0:
            frame_w = project.tracker_manager.frame_width
            frame_h = project.tracker_manager.frame_height
            w_padded = min(w_padded, frame_w - x_padded)
            h_padded = min(h_padded, frame_h - y_padded)
        
        print(f"Added padding: original=({x}, {y}, {w}, {h}), padded=({x_padded}, {y_padded}, {w_padded}, {h_padded})")

        # Use the padded bbox for tracking, but pass original bbox for accurate marker placement
        self._on_bbox_selected(x_padded, y_padded, w_padded, h_padded, original_bbox=(x, y, w, h))
    
    def _on_bbox_selected(self, x: int, y: int, w: int, h: int,
                         original_bbox: Optional[Tuple[int, int, int, int]] = None):
        """Handle bounding box selection (manual or from detection)"""
        print(f"_on_bbox_selected called: bbox=({x}, {y}, {w}, {h}), original_bbox={original_bbox}")
        try:
            # Disable detection mode if active
            self.video_canvas.enable_detection_mode(False)
            
            # Validate bbox
            if w <= 0 or h <= 0:
                print(f"Invalid bbox size: {w}x{h}")
                QMessageBox.warning(self, "Error", "Invalid bounding box size.")
                self._waiting_for_bbox = False
                self.status_label.setText("Ready")
                self.status_label.setStyleSheet("")
                return
            
            # Get current project
            project = self.project_manager.get_current_project()
            if not project:
                QMessageBox.warning(self, "Error", "No video selected")
                self._waiting_for_bbox = False
                return
            
            # Check if there are existing players - always ask user to allow marking multiple players
            existing_players = project.tracker_manager.get_all_players()
            selected_player = None
            
            if existing_players:
                # Always ask user which player this is (or if it's a new player)
                # This allows marking multiple different players in the same frame
                from PyQt6.QtWidgets import QInputDialog
                player_names = [f"{p.name} (Frame {min(p.learning_frames.keys()) + 1})" for p in existing_players]
                player_names.append("➕ New Player")  # Option to create new player (with emoji for clarity)
                
                # Default to "New Player" to make it easy to mark multiple players
                default_index = len(player_names) - 1  # Last item = "New Player"
                
                player_name, ok = QInputDialog.getItem(
                    self,
                    "Select Player",
                    f"You have {len(existing_players)} player(s) marked.\n\n"
                    f"Which player is this?\n\n"
                    f"• Select an existing player to add a learning frame\n"
                    f"• Select '➕ New Player' to mark a different player\n\n"
                    f"Current frame: {self.current_frame_idx + 1}",
                    player_names,
                    default_index,  # Default to "New Player"
                    False
                )
                
                if not ok:
                    # User cancelled
                    self._waiting_for_bbox = False
                    self.status_label.setText("Ready")
                    self.status_label.setStyleSheet("")
                    return
                
                if player_name == "➕ New Player":
                    # User wants to create new player - fall through
                    selected_player = None
                else:
                    # Find selected player
                    selected_index = player_names.index(player_name)
                    selected_player = existing_players[selected_index]
                
                # If we have a selected player, add as learning frame
                if selected_player is not None:
                        
                    # Add learning frame directly (no need for warning - user already selected the player)
                    project.tracker_manager.add_learning_frame_to_player(
                        selected_player.player_id,
                        self.current_frame_idx,
                        (x, y, w, h)
                    )
                    
                    # Update canvas to show learning frame
                    color_map = {
                        'arrow': (0, 255, 255),  # Yellow
                        'circle': (0, 255, 255),  # Yellow
                        'rectangle': (255, 100, 0),  # Blue
                        'spotlight': (100, 255, 255),  # Cyan
                        'neon_ring': (255, 255, 255),  # White
                        'pulse': (0, 165, 255),  # Orange (BGR)
                        'gradient': (255, 0, 200),  # Purple
                        'dynamic_arrow': (0, 255, 200),  # Cyan
                        'hexagon': (255, 150, 0),  # Orange
                        'crosshair': (0, 255, 0),  # Green
                        'flame': (0, 100, 255)  # Orange/Red
                    }
                    color = color_map.get(selected_player.marker_style, (255, 255, 255))
                    self.video_canvas.add_bbox(x, y, w, h, f"{selected_player.name} (Learning)", selected_player.marker_style, color)
                    
                    # Update players list
                    self._update_players_list()
                    
                    # Refresh frame to show markers
                    self._show_frame(self.current_frame_idx)
                    
                    # Show success message (only if multiple learning frames)
                    learning_frames_count = len(selected_player.learning_frames)
                    if learning_frames_count > 1:
                        self.status_label.setText(f"✅ Added learning frame #{learning_frames_count} for {selected_player.name} at frame {self.current_frame_idx + 1}")
                        self.status_label.setStyleSheet("color: green;")
                    else:
                        self.status_label.setText(f"✅ Marked {selected_player.name} at frame {self.current_frame_idx + 1}")
                        self.status_label.setStyleSheet("color: green;")
                    
                    self._waiting_for_bbox = False
                    return
            
            # New player or user said "No" - create new player
            # Show selector dialog
            selector = PlayerSelector(self)
            
            def on_confirmed(name: str, style: str):
                try:
                    # Add player to project
                    player_id = project.add_player(
                        name, style, self.current_frame_idx, (x, y, w, h), original_bbox
                    )
                    
                    # Get color for style
                    color_map = {
                        'arrow': (0, 255, 255),  # Yellow
                        'circle': (0, 255, 255),  # Yellow
                        'rectangle': (255, 100, 0),  # Blue
                        'spotlight': (100, 255, 255),  # Cyan
                        'neon_ring': (255, 255, 255),  # White
                        'pulse': (0, 165, 255),  # Orange (BGR)
                        'gradient': (255, 0, 200),  # Purple
                        'dynamic_arrow': (0, 255, 200),  # Cyan
                        'hexagon': (255, 150, 0),  # Orange
                        'crosshair': (0, 255, 0),  # Green
                        'flame': (0, 100, 255)  # Orange/Red
                    }
                    color = color_map.get(style, (255, 255, 255))
                    
                    # Add to canvas
                    self.video_canvas.add_bbox(x, y, w, h, name, style, color)
                    
                    # Update players list
                    self._update_players_list()
                    
                    # Update video list display
                    current_index = self.project_manager.current_project_index
                    if current_index is not None:
                        self.videos_list.item(current_index).setText(project.get_display_name())
                    
                    # Refresh frame to show markers with overlay renderer
                    self._show_frame(self.current_frame_idx)
                    
                    # Update UI
                    self._update_buttons()
                    self.status_label.setText(f"✅ Added player: {name}")
                    self.status_label.setStyleSheet("color: green;")
                    self._waiting_for_bbox = False
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to add player: {str(e)}")
                    self.status_label.setText("❌ Error adding player")
                    self.status_label.setStyleSheet("color: red;")
                    self._waiting_for_bbox = False
            
            selector.player_confirmed.connect(on_confirmed)
            result = selector.exec()
            
            if result != QDialog.DialogCode.Accepted:
                self._waiting_for_bbox = False
                self.status_label.setText("")
                self.status_label.setStyleSheet("")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error selecting bounding box: {str(e)}")
            self._waiting_for_bbox = False
            self.status_label.setText("")
            self.status_label.setStyleSheet("")
    
    def _on_player_selected(self, item: QListWidgetItem):
        """Handle player selection in list"""
        self.remove_player_btn.setEnabled(True)
    
    def _remove_player(self):
        """Remove selected player"""
        current_item = self.players_list.currentItem()
        if not current_item:
            return
        
        # Get current project
        project = self.project_manager.get_current_project()
        if not project:
            return
        
        player_id = current_item.data(Qt.ItemDataRole.UserRole)
        
        # Remove from tracker
        project.tracker_manager.remove_player(player_id)
        
        # Remove from list
        row = self.players_list.row(current_item)
        self.players_list.takeItem(row)
        
        # Remove from canvas
        self.video_canvas.remove_bbox(row)
        
        # Update UI
        if self.players_list.count() == 0:
            self.remove_player_btn.setEnabled(False)
            # track_single_btn removed
        
        # Refresh display
        self._show_frame(self.current_frame_idx)
        
        # Update all buttons
        self._update_buttons()
    
    def _prev_frame(self):
        """Go to previous frame"""
        try:
            project = self.project_manager.get_current_project()
            if project and self.current_frame_idx > 0:
                self.current_frame_idx -= 1
                self._show_frame(self.current_frame_idx)
                # Stop auto-preview if manually navigating
                if self.preview_timer.isActive():
                    self.preview_timer.stop()
                # Update button states
                self._update_frame_navigation_buttons()
        except Exception as e:
            print(f"❌ Error going to previous frame: {e}")
            import traceback
            traceback.print_exc()
    
    def _next_frame(self):
        """Go to next frame"""
        try:
            project = self.project_manager.get_current_project()
            if project and project.tracker_manager.total_frames > 0:
                if self.current_frame_idx < project.tracker_manager.total_frames - 1:
                    self.current_frame_idx += 1
                    self._show_frame(self.current_frame_idx)
                # Stop auto-preview if manually navigating
                if self.preview_timer.isActive():
                    self.preview_timer.stop()
                # Update button states
                self._update_frame_navigation_buttons()
        except Exception as e:
            print(f"❌ Error going to next frame: {e}")
            import traceback
            traceback.print_exc()
    
    def _toggle_fullscreen(self):
        """Toggle fullscreen mode for video canvas"""
        project = self.project_manager.get_current_project()
        if not project:
            return
        
        # Create fullscreen window (not dialog - to avoid parent window showing)
        from PyQt6.QtWidgets import QWidget, QVBoxLayout
        from PyQt6.QtGui import QShortcut
        from PyQt6.QtCore import Qt
        
        fullscreen_window = QWidget()
        fullscreen_window.setWindowTitle("Fullscreen Video")
        fullscreen_window.setWindowFlags(
            Qt.WindowType.Window | 
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        fullscreen_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        
        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Add button bar at top (will be hidden when not needed)
        button_bar = QWidget()
        button_bar.setStyleSheet("background-color: rgba(0, 0, 0, 180); padding: 10px;")
        button_bar_layout = QHBoxLayout()
        button_bar_layout.setContentsMargins(10, 5, 10, 5)
        
        add_marker_btn = QPushButton("➕ Add Marker")
        add_marker_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        button_bar_layout.addWidget(add_marker_btn)
        button_bar_layout.addStretch()
        
        close_btn = QPushButton("❌ Close (ESC)")
        close_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px;")
        close_btn.clicked.connect(fullscreen_window.close)
        button_bar_layout.addWidget(close_btn)
        
        button_bar.setLayout(button_bar_layout)
        button_bar.setVisible(False)  # Hidden by default, show on hover or when needed
        layout.addWidget(button_bar)
        
        # Create new canvas for fullscreen
        fullscreen_canvas = VideoCanvas()
        fullscreen_canvas.setMinimumSize(800, 600)
        
        # Track if we're waiting for bbox in fullscreen
        fullscreen_waiting_for_bbox = [False]  # Use list to allow modification in nested functions
        
        # Get current frame with overlays
        frame = project.tracker_manager.get_frame(self.current_frame_idx)
        if frame is not None:
            from ..render.overlay_renderer import OverlayRenderer
            overlay_renderer = OverlayRenderer()
            
            # Get tracking results
            players = project.tracker_manager.get_all_players()
            for player in players:
                stored_bbox = project.tracker_manager.get_bbox_at_frame(
                    player.player_id, self.current_frame_idx
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
            frame_with_overlay = overlay_renderer.draw_all_markers(
                frame, 
                players,
                frame_idx=self.current_frame_idx,
                tracking_start_frame=project.trim_start_frame,
                tracking_end_frame=project.trim_end_frame
            )
            fullscreen_canvas.set_frame(frame_with_overlay)
        
        layout.addWidget(fullscreen_canvas)
        fullscreen_window.setLayout(layout)
        
        # Handle bbox selection in fullscreen
        def on_fullscreen_bbox_selected(x: int, y: int, w: int, h: int):
            if not fullscreen_waiting_for_bbox[0]:
                return
            
            # Validate bbox
            if w <= 0 or h <= 0:
                QMessageBox.warning(fullscreen_window, "Error", "Invalid bounding box size.")
                fullscreen_waiting_for_bbox[0] = False
                button_bar.setVisible(False)
                return
            
            # Show selector dialog (in fullscreen window context)
            from .player_selector import PlayerSelector
            selector = PlayerSelector(fullscreen_window)
            
            def on_confirmed(name: str, style: str):
                # Add player to project (manual bbox selection, no padding)
                player_id = project.add_player(
                    name, style, self.current_frame_idx, (x, y, w, h), None
                )
                
                # Get color for style
                color_map = {
                    'arrow': (0, 255, 255),  # Yellow
                    'circle': (0, 255, 255),  # Yellow
                    'rectangle': (255, 100, 0),  # Blue
                    'spotlight': (100, 255, 255),  # Cyan
                    'neon_ring': (255, 255, 255),  # White
                    'pulse': (0, 200, 255),  # Orange
                    'gradient': (255, 0, 200),  # Purple
                    'dynamic_arrow': (0, 255, 200),  # Cyan
                    'hexagon': (255, 150, 0),  # Orange
                    'crosshair': (0, 255, 0),  # Green
                    'flame': (0, 100, 255)  # Orange/Red
                }
                color = color_map.get(style, (255, 255, 255))
                
                # Update UI
                self._update_players_list()
                self._update_buttons()
                
                # Refresh fullscreen view
                frame = project.tracker_manager.get_frame(self.current_frame_idx)
                if frame is not None:
                    from ..render.overlay_renderer import OverlayRenderer
                    overlay_renderer = OverlayRenderer()
                    players = project.tracker_manager.get_all_players()
                    for player in players:
                        stored_bbox = project.tracker_manager.get_bbox_at_frame(
                            player.player_id, self.current_frame_idx
                        )
                        if stored_bbox is not None:
                            player.current_bbox = stored_bbox
                    frame_with_overlay = overlay_renderer.draw_all_markers(
                        frame, 
                        players,
                        frame_idx=self.current_frame_idx,
                        tracking_start_frame=project.trim_start_frame,
                        tracking_end_frame=project.trim_end_frame
                    )
                    fullscreen_canvas.set_frame(frame_with_overlay)
                
                fullscreen_waiting_for_bbox[0] = False
                button_bar.setVisible(False)
                
                QMessageBox.information(
                    fullscreen_window,
                    "Marker Added",
                    f"Marker '{name}' added successfully at frame {self.current_frame_idx + 1}."
                )
            
            selector.player_confirmed.connect(on_confirmed)
            selector.exec()
        
        fullscreen_canvas.bbox_selected.connect(on_fullscreen_bbox_selected)
        
        # Handle add marker button click
        def on_add_marker_clicked():
            fullscreen_waiting_for_bbox[0] = True
            button_bar.setVisible(True)
            QMessageBox.information(
                fullscreen_window,
                "Add Marker",
                "Draw a bounding box on the player you want to track.\n\n"
                "Click and drag on the video to select the player."
            )
        
        add_marker_btn.clicked.connect(on_add_marker_clicked)
        
        # Show/hide button bar on mouse move
        def show_button_bar():
            button_bar.setVisible(True)
        
        def hide_button_bar():
            if not fullscreen_waiting_for_bbox[0]:
                button_bar.setVisible(False)
        
        fullscreen_window.enterEvent = lambda e: show_button_bar()
        fullscreen_window.leaveEvent = lambda e: hide_button_bar()
        
        # Add keyboard shortcuts
        # Escape: Exit fullscreen
        QShortcut(Qt.Key.Key_Escape, fullscreen_window, fullscreen_window.close)
        
        # F11 or F: Exit fullscreen
        QShortcut(Qt.Key.Key_F11, fullscreen_window, fullscreen_window.close)
        QShortcut(Qt.Key.Key_F, fullscreen_window, fullscreen_window.close)
        
        # A or M: Add marker
        QShortcut(Qt.Key.Key_A, fullscreen_window, on_add_marker_clicked)
        QShortcut(Qt.Key.Key_M, fullscreen_window, on_add_marker_clicked)
        
        # Left/Right: Navigate frames
        def next_fullscreen_frame():
            if self.current_frame_idx < project.tracker_manager.total_frames - 1:
                self.current_frame_idx += 1
                frame = project.tracker_manager.get_frame(self.current_frame_idx)
                if frame is not None:
                    from ..render.overlay_renderer import OverlayRenderer
                    overlay_renderer = OverlayRenderer()
                    players = project.tracker_manager.get_all_players()
                    for player in players:
                        stored_bbox = project.tracker_manager.get_bbox_at_frame(
                            player.player_id, self.current_frame_idx
                        )
                        if stored_bbox is not None:
                            player.current_bbox = stored_bbox
                    # Draw overlays only if frame is in tracking range
                    frame_with_overlay = overlay_renderer.draw_all_markers(
                        frame, 
                        players,
                        frame_idx=self.current_frame_idx,
                        tracking_start_frame=project.trim_start_frame,
                        tracking_end_frame=project.trim_end_frame
                    )
                    fullscreen_canvas.set_frame(frame_with_overlay)
                    self._update_frame_info()
        
        def prev_fullscreen_frame():
            if self.current_frame_idx > 0:
                self.current_frame_idx -= 1
                frame = project.tracker_manager.get_frame(self.current_frame_idx)
                if frame is not None:
                    from ..render.overlay_renderer import OverlayRenderer
                    overlay_renderer = OverlayRenderer()
                    players = project.tracker_manager.get_all_players()
                    for player in players:
                        stored_bbox = project.tracker_manager.get_bbox_at_frame(
                            player.player_id, self.current_frame_idx
                        )
                        if stored_bbox is not None:
                            player.current_bbox = stored_bbox
                    # Draw overlays only if frame is in tracking range
                    frame_with_overlay = overlay_renderer.draw_all_markers(
                        frame, 
                        players,
                        frame_idx=self.current_frame_idx,
                        tracking_start_frame=project.trim_start_frame,
                        tracking_end_frame=project.trim_end_frame
                    )
                    fullscreen_canvas.set_frame(frame_with_overlay)
                    self._update_frame_info()
        
        QShortcut(Qt.Key.Key_Right, fullscreen_window, next_fullscreen_frame)
        QShortcut(Qt.Key.Key_Left, fullscreen_window, prev_fullscreen_frame)
        
        # Show window in fullscreen
        fullscreen_window.showFullScreen()
        # Keep reference to prevent garbage collection
        self._fullscreen_window = fullscreen_window
    
    def _update_frame_navigation_buttons(self):
        """Update frame navigation button states"""
        try:
            project = self.project_manager.get_current_project()
            if not project or project.tracker_manager.total_frames == 0:
                self.prev_frame_btn.setEnabled(False)
                self.next_frame_btn.setEnabled(False)
                self.jump_back_10_btn.setEnabled(False)
                self.jump_back_100_btn.setEnabled(False)
                self.jump_forward_10_btn.setEnabled(False)
                self.jump_forward_100_btn.setEnabled(False)
                return
            
            total = project.tracker_manager.total_frames
            if total > 0:
                can_go_back = self.current_frame_idx > 0
                can_go_forward = self.current_frame_idx < total - 1
                
                self.prev_frame_btn.setEnabled(can_go_back)
                self.next_frame_btn.setEnabled(can_go_forward)
                self.jump_back_10_btn.setEnabled(can_go_back)
                self.jump_back_100_btn.setEnabled(can_go_back)
                self.jump_forward_10_btn.setEnabled(can_go_forward)
                self.jump_forward_100_btn.setEnabled(can_go_forward)
                self.fullscreen_btn.setEnabled(True)
            else:
                self.prev_frame_btn.setEnabled(False)
                self.next_frame_btn.setEnabled(False)
                self.jump_back_10_btn.setEnabled(False)
                self.jump_back_100_btn.setEnabled(False)
                self.jump_forward_10_btn.setEnabled(False)
                self.jump_forward_100_btn.setEnabled(False)
                self.fullscreen_btn.setEnabled(False)
        except Exception as e:
            print(f"Error updating navigation buttons: {e}")
            self.prev_frame_btn.setEnabled(False)
            self.next_frame_btn.setEnabled(False)
            self.jump_back_10_btn.setEnabled(False)
            self.jump_back_100_btn.setEnabled(False)
            self.jump_forward_10_btn.setEnabled(False)
            self.jump_forward_100_btn.setEnabled(False)
            self.fullscreen_btn.setEnabled(False)
    
    def _show_frame(self, frame_idx: int):
        """Show specific frame"""
        try:
            project = self.project_manager.get_current_project()
            if not project:
                print("No current project")
                return
            
            tracker_manager = project.tracker_manager
            
            if frame_idx < 0 or (tracker_manager.total_frames > 0 and 
                               frame_idx >= tracker_manager.total_frames):
                print(f"Frame index out of bounds: {frame_idx}")
                return
            
            frame = tracker_manager.get_frame(frame_idx)
            if frame is None:
                print(f"❌ ERROR: Could not load frame {frame_idx}")
                return
            
            # Show markers if there are players (either tracked or just marked)
            if len(tracker_manager.players) > 0:
                from ..render.overlay_renderer import OverlayRenderer
                renderer = OverlayRenderer()
                players = tracker_manager.get_all_players()

                # Debug: print status
                if frame_idx % 30 == 0:
                    print(f"🔍 Frame {frame_idx}: project.status={project.status}")

                # If tracking results exist, use them (status might be TRACKED or MARKED if tracking just finished)
                has_tracking_results = len(tracker_manager.tracking_results) > 0
                if has_tracking_results:
                    # Update current_bbox from stored tracking results
                    # CRITICAL: Always update current_bbox - set to None if no tracking data for this frame
                    # This prevents showing bbox from a different frame
                    for player in players:
                        stored_bbox = tracker_manager.get_bbox_at_frame(
                            player.player_id, frame_idx
                        )
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
                            if frame_idx % 10 == 0:
                                print(f"📍 Frame {frame_idx}: stored_bbox={stored_bbox}, offset={player.padding_offset}, current_original_bbox={player.current_original_bbox}")
                        else:
                            player.current_original_bbox = stored_bbox
                            if frame_idx % 10 == 0:
                                print(f"⚠️ Frame {frame_idx}: No padding_offset! hasattr={hasattr(player, 'padding_offset')}, value={getattr(player, 'padding_offset', None)}")
                else:
                    # Tracking not started yet - show markers only on frames where players were marked
                    for player in players:
                        # Check if this frame is a learning frame for this player
                        if frame_idx in player.learning_frames:
                            # Use the bbox from learning frame
                            player.current_bbox = player.learning_frames[frame_idx]
                        elif frame_idx == player.initial_frame:
                            # Use the initial bbox
                            player.current_bbox = player.bbox
                        else:
                            # No marker for this frame yet (before tracking starts)
                            player.current_bbox = None
                
                # Draw markers (will skip if current_bbox is None)
                # For pre-tracking: show markers on all frames where players were marked
                # For post-tracking: respect tracking range
                tracking_start = project.trim_start_frame if project.trim_start_frame is not None else 0
                tracking_end = project.trim_end_frame if project.trim_end_frame is not None else (tracker_manager.total_frames - 1)

                frame_with_overlay = renderer.draw_all_markers(
                    frame,
                    players,
                    frame_idx=frame_idx,
                    tracking_start_frame=tracking_start if has_tracking_results else None,  # Only enforce range if tracking done
                    tracking_end_frame=tracking_end if has_tracking_results else None
                )
                self.video_canvas.set_frame(frame_with_overlay)
            else:
                # Just show frame without overlays
                self.video_canvas.set_frame(frame)
            
            self._update_frame_info()
        except Exception as e:
            print(f"❌ Error showing frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_frame_info(self):
        """Update frame information label and controls"""
        try:
            project = self.project_manager.get_current_project()
            if not project:
                self.frame_slider.setMaximum(0)
                self.frame_slider.setValue(0)
                self.frame_spinbox.setMaximum(1)
                self.frame_spinbox.setValue(1)
                self.total_frames_label.setText("0")
                return
            
            total = project.tracker_manager.total_frames
            
            # Update slider (block signals to prevent recursion)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setMaximum(max(0, total - 1))
            self.frame_slider.setValue(self.current_frame_idx)
            self.frame_slider.setEnabled(total > 0)
            self.frame_slider.blockSignals(False)
            
            # Update spinbox (block signals to prevent recursion)
            self.frame_spinbox.blockSignals(True)
            self.frame_spinbox.setMaximum(max(1, total))
            self.frame_spinbox.setValue(self.current_frame_idx + 1)  # 1-based
            self.frame_spinbox.setEnabled(total > 0)
            self.frame_spinbox.blockSignals(False)
            
            # Update total frames label
            self.total_frames_label.setText(str(total))
            
            # Update navigation buttons
            self._update_frame_navigation_buttons()
        except Exception as e:
            print(f"Error updating frame info: {e}")
    
    def _on_slider_changed(self, value: int):
        """Handle slider value change - jump to frame"""
        if value != self.current_frame_idx:
            self.current_frame_idx = value
            self._show_frame(value)
    
    def _on_frame_number_changed(self, value: int):
        """Handle frame number spinbox change - jump to frame (1-based)"""
        frame_idx = value - 1  # Convert to 0-based
        if frame_idx != self.current_frame_idx:
            self.current_frame_idx = frame_idx
            self._show_frame(frame_idx)
    
    def _jump_frames(self, offset: int):
        """Jump forward or backward by specified number of frames"""
        project = self.project_manager.get_current_project()
        if not project:
            return
        
        total = project.tracker_manager.total_frames
        if total <= 0:
            return
        
        new_idx = self.current_frame_idx + offset
        new_idx = max(0, min(new_idx, total - 1))
        
        if new_idx != self.current_frame_idx:
            self.current_frame_idx = new_idx
            self._show_frame(new_idx)
    
    def _start_tracking(self):
        """Start tracking process"""
        if not self.video_path or len(self.tracker_manager.players) == 0:
            return
        
        # Clear previous tracking data
        self.tracker_manager.tracking_results.clear()
        
        # Disable controls
        self.start_tracking_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.add_player_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(self.tracker_manager.total_frames)
        self.progress_bar.setValue(0)
        self.status_label.setText("Tracking in progress...")
        self.status_label.setStyleSheet("color: green;")
        
        # Create and start tracking thread (no trim for old code path)
        self.tracking_thread = TrackingThread(self.tracker_manager, self.video_path, None, None)
        self.tracking_thread.progress.connect(self._on_tracking_progress)
        self.tracking_thread.finished.connect(self._on_tracking_finished)
        self.tracking_thread.start()
    
    def _on_tracking_progress(self, current: int, total: int):
        """Handle tracking progress update"""
        self.progress_bar.setValue(current)
        self.status_label.setText(f"Tracking: {current}/{total} frames")
    
    def _on_tracking_finished(self, success: bool, message: str):
        """Handle tracking completion"""
        self.progress_bar.setVisible(False)
        
        if success:
            self.status_label.setText("Tracking completed!")
            self.status_label.setStyleSheet("color: green;")
            self.export_btn.setEnabled(True)
            
            # Start preview updates
            self.preview_timer.start(33)  # ~30 FPS preview
        else:
            self.status_label.setText(f"Error: {message}")
            self.status_label.setStyleSheet("color: red;")
            QMessageBox.warning(self, "Tracking Error", message)
        
        # Re-enable controls
        self.start_tracking_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.add_player_btn.setEnabled(True)
    
    def _update_preview(self):
        """Update video preview with tracking"""
        if not self.video_path:
            self.preview_timer.stop()
            return
        
        # Get current frame
        frame = self.tracker_manager.get_frame(self.current_frame_idx)
        if frame is None:
            self.preview_timer.stop()
            return
        
        # Draw overlays using stored tracking results
        from ..render.overlay_renderer import OverlayRenderer
        renderer = OverlayRenderer()
        players = self.tracker_manager.get_all_players()
        
        # Update current_bbox from stored tracking results
        # CRITICAL: Always update current_bbox - set to None if no tracking data for this frame
        # This prevents showing bbox from a different frame
        for player in players:
            stored_bbox = self.tracker_manager.get_bbox_at_frame(
                player.player_id, self.current_frame_idx
            )
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

        frame_with_overlay = renderer.draw_all_markers(frame, players)
        self.video_canvas.set_frame(frame_with_overlay)
        
        # Auto-advance frame for preview
        if self.current_frame_idx < self.tracker_manager.total_frames - 1:
            self.current_frame_idx += 1
            self._update_frame_info()
        else:
            self.current_frame_idx = 0  # Loop back to start
    
    def _export_video(self):
        """Export video with tracking"""
        if not self.video_path:
            return
        
        # Get output path
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Video",
            "output_with_tracking.mp4",
            "Video Files (*.mp4);;All Files (*)"
        )
        
        if not output_path:
            return
        
        # Disable controls
        self.export_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.start_tracking_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(self.tracker_manager.total_frames)
        self.progress_bar.setValue(0)
        self.status_label.setText("Exporting video...")
        self.status_label.setStyleSheet("color: blue;")
        
        # Create and start export thread (old code path - no project, no tracking range)
        self.export_thread = ExportThread(self.tracker_manager, self.video_path, output_path, None, None)
        self.export_thread.progress.connect(self._on_export_progress)
        self.export_thread.finished.connect(self._on_export_finished)
        self.export_thread.start()
    
    def _on_export_progress(self, current: int, total: int):
        """Handle export progress update"""
        self.progress_bar.setValue(current)
        self.status_label.setText(f"Exporting: {current}/{total} frames")
    
    def _on_export_finished(self, success: bool, message: str):
        """Handle export completion"""
        self.progress_bar.setVisible(False)
        
        if success:
            self.status_label.setText("Export completed!")
            self.status_label.setStyleSheet("color: green;")
            QMessageBox.information(
                self,
                "Success",
                message
            )
        else:
            self.status_label.setText(f"Export error")
            self.status_label.setStyleSheet("color: red;")
            QMessageBox.warning(self, "Export Error", message)
        
        # Re-enable controls
        self.export_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.start_tracking_btn.setEnabled(True)
    
    def _set_tracking_start(self):
        """Set tracking start frame to current frame"""
        project = self.project_manager.get_current_project()
        if not project:
            return
        
        # Validate: start must be before end (if end is set)
        if project.trim_end_frame is not None and self.current_frame_idx >= project.trim_end_frame:
            QMessageBox.warning(
                self,
                "Invalid Range",
                f"Start frame ({self.current_frame_idx + 1}) must be before end frame ({project.trim_end_frame + 1})"
            )
            return
        
        project.trim_start_frame = self.current_frame_idx
        self._update_tracking_range_info()
        self._update_buttons()
        
        QMessageBox.information(
            self,
            "Tracking Start Set",
            f"Tracking will start from frame {self.current_frame_idx + 1}.\n\n"
            f"The video will play from the beginning, but tracking markers will appear only from this frame."
        )
    
    def _set_tracking_end(self):
        """Set tracking end frame to current frame"""
        project = self.project_manager.get_current_project()
        if not project:
            return
        
        # Validate: end must be after start (if start is set)
        if project.trim_start_frame is not None and self.current_frame_idx <= project.trim_start_frame:
            QMessageBox.warning(
                self,
                "Invalid Range",
                f"End frame ({self.current_frame_idx + 1}) must be after start frame ({project.trim_start_frame + 1})"
            )
            return
        
        project.trim_end_frame = self.current_frame_idx
        self._update_tracking_range_info()
        self._update_buttons()
        
        QMessageBox.information(
            self,
            "Tracking End Set",
            f"Tracking will end at frame {self.current_frame_idx + 1}.\n\n"
            f"From this frame to the end, there will be no tracking markers."
        )
    
    def _clear_tracking_range(self):
        """Clear tracking start and end frames - tracking will be on full video"""
        project = self.project_manager.get_current_project()
        if not project:
            return
        
        project.trim_start_frame = None
        project.trim_end_frame = None
        self._update_tracking_range_info()
        self._update_buttons()
        
        QMessageBox.information(
            self,
            "Tracking Range Cleared",
            "Tracking will be on the full video (from beginning to end)."
        )
    
    def _update_tracking_range_info(self):
        """Update tracking range info label"""
        project = self.project_manager.get_current_project()
        if not project:
            self.tracking_range_info_label.setText("Tracking: Full video")
            return
        
        total_frames = project.tracker_manager.total_frames if project.tracker_manager else 0
        
        if project.trim_start_frame is None and project.trim_end_frame is None:
            self.tracking_range_info_label.setText(
                f"Tracking: Full video (Frame 1 to {total_frames})"
            )
        elif project.trim_start_frame is not None and project.trim_end_frame is not None:
            frames_with_tracking = project.trim_end_frame - project.trim_start_frame + 1
            self.tracking_range_info_label.setText(
                f"Tracking: Frame {project.trim_start_frame + 1} to {project.trim_end_frame + 1}\n"
                f"({frames_with_tracking} frames with tracking markers)"
            )
        elif project.trim_start_frame is not None:
            frames_with_tracking = total_frames - project.trim_start_frame
            self.tracking_range_info_label.setText(
                f"Tracking: From frame {project.trim_start_frame + 1} to end\n"
                f"({frames_with_tracking} frames with tracking markers)"
            )
        elif project.trim_end_frame is not None:
            frames_with_tracking = project.trim_end_frame + 1
            self.tracking_range_info_label.setText(
                f"Tracking: From start to frame {project.trim_end_frame + 1}\n"
                f"({frames_with_tracking} frames with tracking markers)"
            )
    
    def _fix_tracking(self, project: VideoProject, frame_idx: int, x: int, y: int, w: int, h: int, player_id: int):
        """Fix tracking by restarting from a specific frame with new bbox (OLD - kept for compatibility)"""
        # This function is kept for compatibility but is no longer used
        # The new flow uses _resume_tracking_from_frame instead
        pass
    
    def _resume_tracking_from_frame(self, project: VideoProject, frame_idx: int, x: int, y: int, w: int, h: int, player_id: int, preview_dialog):
        """Resume tracking from a specific frame, preserving all previous tracking data"""
        # Find player
        player = None
        for p in project.tracker_manager.get_all_players():
            if p.player_id == player_id:
                player = p
                break
        
        if not player:
            QMessageBox.warning(self, "Error", "Player not found.")
            return
        
        # IMPORTANT: Only delete tracking results from AFTER this frame (not including this frame)
        # This preserves all tracking data up to and including the fix point
        if player_id in project.tracker_manager.tracking_results:
            frames_to_delete = [
                f for f in project.tracker_manager.tracking_results[player_id].keys()
                if f > frame_idx  # Note: > not >=, so we keep the fix frame
            ]
            print(f"🔧 Fix Tracking: Deleting {len(frames_to_delete)} frames after frame {frame_idx}")
            for f in frames_to_delete:
                del project.tracker_manager.tracking_results[player_id][f]
        
        # CRITICAL: Update player's initial_frame to the fix point
        # This ensures the tracker will initialize at the correct frame
        old_initial_frame = player.initial_frame
        player.initial_frame = frame_idx
        print(f"🔧 Fix Tracking: Updated player {player_id} initial_frame from {old_initial_frame} to {frame_idx}")
        
        # Update player's bbox to the new fix bbox
        player.bbox = (x, y, w, h)
        player.current_bbox = (x, y, w, h)
        print(f"🔧 Fix Tracking: Updated player {player_id} bbox to {player.bbox}")
        
        # Save the new bbox at this frame (overwrite if exists)
        if player_id not in project.tracker_manager.tracking_results:
            project.tracker_manager.tracking_results[player_id] = {}
        project.tracker_manager.tracking_results[player_id][frame_idx] = (x, y, w, h)
        
        # Reset tracker - create new one (will be initialized at fix frame)
        from ..tracking.player_tracker import PlayerTracker, TrackerType
        player.tracker = PlayerTracker(TrackerType.CSRT)
        player.tracking_lost = False
        player.tracker.is_initialized = False  # Ensure it will be initialized at fix frame
        
        # Determine tracking end frame (use project's trim_end_frame if set)
        total_frames = project.tracker_manager.total_frames
        end_frame = project.trim_end_frame if project.trim_end_frame is not None else (total_frames - 1)
        
        # Start tracking from fix frame onwards
        self.status_label.setText(f"🔄 Resuming tracking from frame {frame_idx + 1}...")
        self.status_label.setStyleSheet("color: orange;")
        
        # Create tracking thread starting from fix frame
        self.tracking_thread = TrackingThread(
            project.tracker_manager,
            project.video_path,
            frame_idx,  # Start from fix frame
            end_frame   # End at project's end frame
        )
        
        def on_tracking_complete(success, message):
            if success:
                project.status = ProjectStatus.TRACKED
                self.status_label.setText("✅ Tracking resumed successfully!")
                self.status_label.setStyleSheet("color: green;")
                
                # Reload current frame in preview to show updated tracking
                preview_dialog._load_frame(preview_dialog.current_frame_idx)
                
                QMessageBox.information(
                    self,
                    "Tracking Resumed",
                    f"Tracking resumed successfully from frame {frame_idx + 1}.\n\n"
                    f"All previous tracking data has been preserved.\n"
                    f"The preview has been updated with the new tracking."
                )
            else:
                QMessageBox.warning(self, "Tracking Failed", f"Failed to resume tracking: {message}")
                self.status_label.setText("❌ Tracking resume failed")
                self.status_label.setStyleSheet("color: red;")
            
            # Clean up thread
            if self.tracking_thread is not None:
                try:
                    self.tracking_thread.finished.disconnect()
                    self.tracking_thread.progress.disconnect()
                    self.tracking_thread.deleteLater()
                except:
                    pass
                self.tracking_thread = None
        
        self.tracking_thread.finished.connect(on_tracking_complete)
        self.tracking_thread.progress.connect(
            lambda current, total: self.status_label.setText(
                f"🔄 Resuming tracking... {current}/{total} frames"
            )
        )
        
        # Handle tracking lost notifications
        def on_tracking_lost_resume(player_id: int, player_name: str, frame_idx: int):
            QMessageBox.warning(
                self,
                "Tracking Lost",
                f"⚠️ Tracking lost for '{player_name}' at frame {frame_idx + 1}.\n\n"
                f"The system could not identify the player at this frame.\n"
                f"You may need to fix the tracking at this point."
            )
        
        self.tracking_thread.tracking_lost.connect(on_tracking_lost_resume)
        
        self.tracking_thread.start()
        self.status_label.setText(f"✅ Tracking fix applied. Please run tracking again.")
        self.status_label.setStyleSheet("color: green;")
    
    def closeEvent(self, event):
        """Handle window close - PROPERLY CLEANUP ALL THREADS"""
        # Cancel any running threads
        if self.tracking_thread and self.tracking_thread.isRunning():
            self.tracking_thread.cancel()
            self.tracking_thread.wait()
            self.tracking_thread.deleteLater()
        
        if self.batch_tracking_thread and self.batch_tracking_thread.isRunning():
            self.batch_tracking_thread.cancelled = True
            self.batch_tracking_thread.quit()
            self.batch_tracking_thread.wait(2000)
            self.batch_tracking_thread.deleteLater()
        
        if self.export_thread and self.export_thread.isRunning():
            self.export_thread.cancel()
            self.export_thread.wait()
            self.export_thread.deleteLater()
        
        if self.batch_export_thread and self.batch_export_thread.isRunning():
            self.batch_export_thread.cancel()
            self.batch_export_thread.wait()
            self.batch_export_thread.deleteLater()
        
        # Release all projects
        self.project_manager.clear_all()
        event.accept()
