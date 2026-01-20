"""
Batch Exporter - Handles batch processing of multiple videos
"""
import os
import cv2
from typing import List, Optional
from PyQt6.QtCore import QThread, pyqtSignal

from ..tracking.video_project import VideoProject, ProjectStatus
from ..tracking.player_tracker import TrackerType
from .video_exporter import VideoExporter


class BatchExportThread(QThread):
    """Thread for batch processing multiple videos"""
    
    # Signals
    project_started = pyqtSignal(int, str)  # project_index, project_name
    project_progress = pyqtSignal(int, int, int)  # project_index, current_frame, total_frames
    project_completed = pyqtSignal(int, bool, str)  # project_index, success, message
    all_completed = pyqtSignal(int, int, int)  # total, successful, failed
    
    def __init__(self, projects: List[VideoProject], output_directory: str):
        """
        Initialize batch export thread
        
        Args:
            projects: List of video projects to process
            output_directory: Directory to save exported videos
        """
        super().__init__()
        self.projects = projects
        self.output_directory = output_directory
        self.cancelled = False
    
    def run(self):
        """Run batch export process"""
        total_projects = len(self.projects)
        successful = 0
        failed = 0
        
        for idx, project in enumerate(self.projects):
            if self.cancelled:
                break
            
            # Skip projects without players
            if not project.has_players():
                project.status = ProjectStatus.SKIPPED
                project.error_message = "No players marked"
                self.project_completed.emit(idx, False, "Skipped: No players marked")
                continue
            
            # Emit start signal
            project_name = os.path.basename(project.video_path)
            self.project_started.emit(idx, project_name)
            
            # Process project
            success = self._process_project(project, idx)
            
            if success:
                successful += 1
                project.status = ProjectStatus.EXPORTED
                self.project_completed.emit(idx, True, f"Successfully exported: {project.output_path}")
            else:
                failed += 1
                project.status = ProjectStatus.FAILED
                error_msg = project.error_message or "Unknown error"
                self.project_completed.emit(idx, False, f"Failed: {error_msg}")
        
        # Emit completion signal
        self.all_completed.emit(total_projects, successful, failed)
    
    def _process_project(self, project: VideoProject, project_idx: int) -> bool:
        """
        Process a single project: track + export
        
        Args:
            project: Video project to process
            project_idx: Index of project for progress signals
            
        Returns:
            True if successful
        """
        try:
            # Step 1: Tracking - ONLY if not already tracked
            # Check if tracking data already exists
            has_tracking_data = False
            if project.status == ProjectStatus.TRACKED:
                # Check if we have tracking results
                players = project.tracker_manager.get_all_players()
                if players:
                    # Check if at least one player has tracking results
                    for player in players:
                        if player.player_id in project.tracker_manager.tracking_results:
                            if len(project.tracker_manager.tracking_results[player.player_id]) > 0:
                                has_tracking_data = True
                                break
            
            if not has_tracking_data:
                # Need to track first
                project.status = ProjectStatus.TRACKING
                print(f"[Batch] Project {project_idx} needs tracking - starting now...")
                if not self._track_project(project, project_idx):
                    return False
                project.status = ProjectStatus.TRACKED
            else:
                # Already tracked - skip tracking step
                print(f"[Batch] Project {project_idx} already tracked - skipping tracking step")
                project.status = ProjectStatus.TRACKED
            
            # Step 2: Export
            project.status = ProjectStatus.EXPORTING
            if not self._export_project(project, project_idx):
                return False
            
            return True
            
        except Exception as e:
            project.error_message = f"Exception: {str(e)}"
            print(f"Error processing project {project_idx}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _track_project(self, project: VideoProject, project_idx: int) -> bool:
        """
        Perform tracking on project
        
        Args:
            project: Video project
            project_idx: Project index
            
        Returns:
            True if successful
        """
        try:
            tracker_manager = project.tracker_manager
            total_frames = tracker_manager.total_frames
            
            if total_frames == 0:
                project.error_message = "Video has no frames"
                return False
            
            # Get tracking range
            start_frame = project.trim_start_frame if project.trim_start_frame is not None else 0
            end_frame = project.trim_end_frame if project.trim_end_frame is not None else total_frames - 1
            
            # Open video capture for tracking
            cap = cv2.VideoCapture(project.video_path)
            if not cap.isOpened():
                project.error_message = "Failed to open video for tracking"
                return False
            
            # Skip to start frame if needed - use reliable seeking method
            frame_idx = start_frame
            players = tracker_manager.get_all_players()
            
            print(f"[Batch] Tracking project {project_idx}: {len(players)} players, frames {start_frame}-{end_frame} (total: {total_frames})")
            
            if start_frame > 0:
                # Try to seek directly
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                # If seek failed, read sequentially from beginning
                if actual_pos != start_frame:
                    print(f"[Batch] ⚠️ Seek to frame {start_frame} failed (actual: {actual_pos}), reading sequentially...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    for i in range(start_frame):
                        ret, _ = cap.read()
                        if not ret:
                            print(f"[Batch] ❌ ERROR: Failed to read to frame {start_frame}")
                            project.error_message = f"Failed to seek to start frame {start_frame}"
                            cap.release()
                            return False
                else:
                    print(f"[Batch] ✅ Successfully seeked to frame {start_frame}")
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            while frame_idx <= end_frame and frame_idx < total_frames and not self.cancelled:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"[Batch] ⚠️ WARNING: Failed to read frame {frame_idx}")
                    break
                
                # Initialize trackers at their initial frames
                # CRITICAL: Only initialize if we're at or past both the player's initial_frame AND the tracking start_frame
                for player in players:
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
                            print(f"[Batch] ⚠️ WARNING: No learning frames found, using initial bbox from frame {player.initial_frame}")
                        
                        # Warn if learning frame is too far from start frame (more than 50 frames)
                        if min_distance > 50:
                            print(f"[Batch] ⚠️ WARNING: Learning frame {best_learning_frame} is {min_distance} frames away from tracking start ({start_frame})")
                            print(f"   This may cause tracking to fail or track the wrong player!")
                            print(f"   Consider marking the player again near frame {start_frame} for better accuracy.")
                        
                        print(f"[Batch] Initializing tracker for player {player.player_id} at frame {frame_idx}")
                        print(f"   Using learning frame {best_learning_frame} with bbox: {best_learning_bbox}")
                        print(f"   Distance from start: {min_distance} frames")
                        print(f"   Total learning frames: {len(player.learning_frames)}")
                        print(f"   Tracking start: {start_frame}, Current frame: {frame_idx}")
                        
                        # Validate bbox before initializing
                        if best_learning_bbox is None or len(best_learning_bbox) != 4:
                            print(f"[Batch] ❌ ERROR: Invalid bbox for player {player.player_id}: {best_learning_bbox}")
                            continue
                        
                        x, y, w, h = best_learning_bbox
                        if w <= 0 or h <= 0:
                            print(f"[Batch] ❌ ERROR: Invalid bbox size for player {player.player_id}: {w}x{h}")
                            continue
                        
                        # Check if bbox is within frame bounds
                        frame_h, frame_w = frame.shape[:2]
                        if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
                            print(f"[Batch] ⚠️ WARNING: Bbox {best_learning_bbox} is outside frame bounds ({frame_w}x{frame_h})")
                            print(f"   Clamping bbox to frame bounds...")
                            x = max(0, min(x, frame_w - 1))
                            y = max(0, min(y, frame_h - 1))
                            w = min(w, frame_w - x)
                            h = min(h, frame_h - y)
                            best_learning_bbox = (x, y, w, h)
                            print(f"   Adjusted bbox: {best_learning_bbox}")
                        
                        init_success = player.tracker.init_tracker(frame, best_learning_bbox)
                        if init_success:
                            player.current_bbox = best_learning_bbox
                            if player.player_id not in tracker_manager.tracking_results:
                                tracker_manager.tracking_results[player.player_id] = {}
                            tracker_manager.tracking_results[player.player_id][frame_idx] = best_learning_bbox
                            print(f"[Batch] ✅ Player {player.player_id} initialized successfully")
                        else:
                            print(f"[Batch] ❌ ERROR: Failed to initialize tracker for player {player.player_id}")
                            print(f"   This may be due to:")
                            print(f"   - Incorrect bbox coordinates")
                            print(f"   - Frame mismatch (initializing at wrong frame)")
                            print(f"   - Tracker initialization error")
                
                # Update trackers
                # CRITICAL: Only update if tracker is initialized and we're past the initial frame
                for player in players:
                    if not player.tracker.is_initialized:
                        # Tracker not initialized yet - no bbox for this frame
                        bbox = None
                    elif frame_idx > player.initial_frame:
                        # Update tracker
                        bbox = player.tracker.update(frame)
                        player.current_bbox = bbox
                        player.tracking_lost = (bbox is None)
                    elif frame_idx == player.initial_frame:
                        # At initial frame - use the stored bbox
                        bbox = player.current_bbox
                    else:
                        # Before initial frame - no bbox
                        bbox = None
                    
                    # Store result (None if no tracking data for this frame)
                    if player.player_id not in tracker_manager.tracking_results:
                        tracker_manager.tracking_results[player.player_id] = {}
                    tracker_manager.tracking_results[player.player_id][frame_idx] = bbox
                
                # Emit progress
                self.project_progress.emit(project_idx, frame_idx + 1, total_frames)
                frame_idx += 1
            
            cap.release()
            
            if frame_idx == 0:
                project.error_message = "Failed to process any frames"
                return False
            
            print(f"[Batch] Tracking complete: {frame_idx} frames processed")
            return True
            
        except Exception as e:
            project.error_message = f"Tracking error: {str(e)}"
            print(f"[Batch] Tracking error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _export_project(self, project: VideoProject, project_idx: int) -> bool:
        """
        Export project with tracking overlays
        
        Args:
            project: Video project
            project_idx: Project index
            
        Returns:
            True if successful
        """
        try:
            # Generate output filename
            input_filename = os.path.basename(project.video_path)
            name, ext = os.path.splitext(input_filename)
            output_filename = f"{name}_tracked.mp4"
            output_path = os.path.join(self.output_directory, output_filename)
            
            # Create exporter
            exporter = VideoExporter(project.tracker_manager)
            
            # Export with progress callback
            def progress_callback(frame_idx, total_frames):
                if not self.cancelled:
                    self.project_progress.emit(project_idx, frame_idx, total_frames)
            
            success = exporter.export_video(
                project.video_path,
                output_path,
                progress_callback,
                project.trim_start_frame,
                project.trim_end_frame
            )
            
            if success:
                project.output_path = output_path
                print(f"[Batch] Export complete: {output_path}")
                return True
            else:
                project.error_message = "Export failed"
                print(f"[Batch] Export failed for project {project_idx}")
                return False
                
        except Exception as e:
            project.error_message = f"Export error: {str(e)}"
            print(f"[Batch] Export error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def cancel(self):
        """Cancel batch processing"""
        self.cancelled = True


