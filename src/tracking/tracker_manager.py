"""
Tracker Manager - Manages multiple player trackers
"""
import cv2
import numpy as np
import traceback
from typing import List, Dict, Optional, Tuple
from .player_tracker import PlayerTracker, TrackerType
from .person_detector import PersonDetector


class PlayerData:
    """Data structure for a tracked player"""
    def __init__(self, player_id: int, name: str, marker_style: str,
                 initial_frame: int, bbox: Tuple[int, int, int, int],
                 original_bbox: Optional[Tuple[int, int, int, int]] = None):
        self.player_id = player_id
        self.name = name
        self.marker_style = marker_style  # 'arrow', 'circle', 'rectangle'
        self.initial_frame = initial_frame  # First frame where player was marked (for tracking start)
        self.bbox = bbox  # Bbox at initial_frame (used for tracking initialization) - this is the PADDED bbox
        self.original_bbox = original_bbox or bbox  # Original bbox BEFORE padding (for accurate marker placement)
        # Learning frames: frames where user marked this player for learning (before tracking starts)
        # Format: {frame_idx: bbox}
        self.learning_frames: Dict[int, Tuple[int, int, int, int]] = {initial_frame: bbox}
        self.original_learning_frames: Dict[int, Tuple[int, int, int, int]] = {initial_frame: original_bbox or bbox}
        self.tracker = PlayerTracker(TrackerType.CSRT)
        self.current_bbox = bbox
        self.current_original_bbox = original_bbox or bbox
        self.tracking_lost = False
        self.color = self._get_default_color()

        # Calculate padding offset (difference between padded and original bbox)
        if original_bbox and original_bbox != bbox:
            orig_x, orig_y, orig_w, orig_h = original_bbox
            pad_x, pad_y, pad_w, pad_h = bbox
            self.padding_offset = (orig_x - pad_x, orig_y - pad_y, pad_w - orig_w, pad_h - orig_h)
        else:
            self.padding_offset = (0, 0, 0, 0)  # No padding
    
    def add_learning_frame(self, frame_idx: int, bbox: Tuple[int, int, int, int],
                          original_bbox: Optional[Tuple[int, int, int, int]] = None):
        """Add a learning frame for this player"""
        self.learning_frames[frame_idx] = bbox
        self.original_learning_frames[frame_idx] = original_bbox or bbox
        # Update initial_frame to the earliest learning frame
        if frame_idx < self.initial_frame:
            self.initial_frame = frame_idx
            self.bbox = bbox
            self.original_bbox = original_bbox or bbox
    
    def _get_default_color(self) -> Tuple[int, int, int]:
        """Get default color based on marker style"""
        color_map = {
            'arrow': (0, 255, 255),        # Yellow
            'circle': (0, 255, 255),       # Yellow (for 3D floor hoop)
            'rectangle': (255, 100, 0),    # Blue (forced in renderer)
            'spotlight': (0, 200, 255),    # Orange
            'spotlight_modern': (200, 255, 255),  # Cyan/white beam
            'outline': (255, 0, 255),      # Magenta
            'nba_iso_ring': (0, 215, 255), # Gold/Cyan glow
            'floating_chevron': (0, 255, 0),  # Bright green for aerial chevron
            'crosshair': (255, 255, 0),       # Neon cyan tactical scope
            'tactical_brackets': (0, 215, 255), # Brackets in same broadcast yellow
            'sonar_ripple': (0, 215, 255),     # Floor ripple in broadcast yellow
            'dramatic_floor_uplight': (200, 240, 255)  # Warm white for dramatic uplight
        }
        return color_map.get(self.marker_style, (255, 255, 255))


class TrackerManager:
    """Manages multiple player trackers"""

    # Failure detection thresholds
    MAX_SIZE_CHANGE_FACTOR = 0.20   # 20% size change between frames
    MAX_CENTER_SHIFT_FACTOR = 0.10  # 10% center shift relative to box size
    
    def __init__(self):
        self.players: Dict[int, PlayerData] = {}
        self.next_player_id = 1
        self.video_cap = None
        self.video_path: Optional[str] = None
        self.total_frames = 0
        self.fps = 30.0
        self.duration = 0.0
        self.frame_width = 0
        self.frame_height = 0
        self.current_frame_idx = 0
        # Store tracking results: {player_id: {frame_idx: bbox}}
        self.tracking_results: Dict[int, Dict[int, Tuple[int, int, int, int]]] = {}
        # Persistent tracking data with confidence (Phase 1)
        self.tracking_data: Dict[int, Dict[int, Dict[str, any]]] = {}
        # Track earliest frame needing recompute per player after corrections
        self.needs_recompute_from: Dict[int, int] = {}
        # Hybrid tracking configuration
        self.tracking_config = {
            "mode": "hybrid",  # hybrid (CSRT + YOLO auto-recovery) | legacy (CSRT only)
            "iou_min": 0.15,
            "scale_change_max": 0.35,
            "center_jump_px": 80.0,
            "reacquire_interval": 5,  # frames between YOLO attempts when lost
            "smoothing_alpha": 0.65,  # EMA smoothing factor (higher = more responsive, less lag)
            "lost_patience": 8,       # frames lost before aggressive search
            "learning_frame_grace": 20  # frames after learning frame before YOLO can intervene
        }
        self.person_detector = PersonDetector()
    
    def _is_valid_fps(self, fps: float) -> bool:
        return 1 <= fps <= 240
    
    def _is_valid_frame_count(self, frame_count: float) -> bool:
        return 1 <= frame_count <= 100000
    
    def _count_frames(self, cap: cv2.VideoCapture) -> int:
        """Count frames manually when CAP_PROP_FRAME_COUNT is unreliable"""
        frame_count = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count > 100000:
                break
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return frame_count
    
    def probe_video(self, video_path: str) -> Optional[Dict[str, float]]:
        """Read metadata from video without keeping the capture open"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not self._is_valid_fps(fps):
            fps = 30.0
        
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not self._is_valid_frame_count(frame_count):
            frame_count = self._count_frames(cap)
        else:
            frame_count = int(frame_count)
        
        duration = frame_count / fps if fps > 0 else 0.0
        
        cap.release()
        
        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "width": width,
            "height": height
        }
        
    def load_video(self, video_path: str, metadata: Optional[Dict[str, float]] = None) -> bool:
        """
        Load video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if video loaded successfully
        """
        try:
            if metadata is None:
                metadata = self.probe_video(video_path)
            if metadata is None:
                return False
            
            # Release existing capture if any
            if self.video_cap is not None:
                self.video_cap.release()
            
            video_cap = cv2.VideoCapture(video_path)
            if not video_cap.isOpened():
                return False
            
            self.video_cap = video_cap
            self.video_path = video_path
            self.fps = metadata.get("fps", 30.0)
            self.total_frames = int(metadata.get("frame_count", 0))
            self.duration = metadata.get("duration", 0.0)
            self.frame_width = int(metadata.get("width", 0))
            self.frame_height = int(metadata.get("height", 0))
            self.current_frame_idx = 0
            return True
        except Exception as e:
            print(f"Error loading video: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def add_player(self, name: str, marker_style: str,
                   initial_frame: int, bbox: Tuple[int, int, int, int],
                   original_bbox: Optional[Tuple[int, int, int, int]] = None) -> int:
        """
        Add a new player to track

        Args:
            name: Player name
            marker_style: Style of marker ('arrow', 'circle', 'rectangle')
            initial_frame: Frame index where player is marked
            bbox: Bounding box (x, y, width, height) - PADDED bbox for tracking
            original_bbox: Original bbox BEFORE padding (for accurate marker placement)

        Returns:
            Player ID
        """
        player_id = self.next_player_id
        self.next_player_id += 1

        player = PlayerData(player_id, name, marker_style, initial_frame, bbox, original_bbox)
        self.players[player_id] = player
        # Ensure tracking data structures exist for this player
        if player_id not in self.tracking_data:
            self.tracking_data[player_id] = {}
        if player_id not in self.tracking_results:
            self.tracking_results[player_id] = {}

        return player_id
    
    def add_learning_frame_to_player(self, player_id: int, frame_idx: int, bbox: Tuple[int, int, int, int],
                                    original_bbox: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """
        Add a learning frame to an existing player

        Args:
            player_id: Player ID
            frame_idx: Frame index where player is marked
            bbox: Bounding box (x, y, width, height) - PADDED bbox for tracking
            original_bbox: Original bbox BEFORE padding (for accurate marker placement)

        Returns:
            True if successful, False if player not found
        """
        if player_id not in self.players:
            return False

        # Ensure primitive ints for consistency
        frame_idx = int(frame_idx)
        bbox = tuple(int(v) for v in bbox)
        if original_bbox:
            original_bbox = tuple(int(v) for v in original_bbox)

        self.players[player_id].add_learning_frame(frame_idx, bbox, original_bbox)
        print(f"[Learning] Saved learning frame for player {player_id} at frame {frame_idx} bbox={bbox} "
              f"total={len(self.players[player_id].learning_frames)}")
        # Invalidate tracking from this frame onward for incremental resume
        self.invalidate_tracking_from(player_id, frame_idx)
        return True
    
    def update_tracking_config(self, **kwargs):
        """Update tracking configuration for hybrid mode"""
        for key, value in kwargs.items():
            if key in self.tracking_config:
                self.tracking_config[key] = value
    
    def invalidate_tracking_from(self, player_id: int, frame_idx: int):
        """
        Invalidate cached tracking data from frame_idx onward for a player.
        Keeps frames before frame_idx intact.
        """
        # Mark earliest recompute point
        if player_id not in self.needs_recompute_from:
            self.needs_recompute_from[player_id] = frame_idx
        else:
            self.needs_recompute_from[player_id] = min(self.needs_recompute_from[player_id], frame_idx)

        # Remove stale data in tracking_data
        if player_id in self.tracking_data:
            for f in list(self.tracking_data[player_id].keys()):
                if f >= frame_idx:
                    del self.tracking_data[player_id][f]

        # Remove stale data in tracking_results (used by export)
        if player_id in self.tracking_results:
            for f in list(self.tracking_results[player_id].keys()):
                if f >= frame_idx:
                    del self.tracking_results[player_id][f]
    
    def get_resume_start(self, requested_start: int = 0) -> int:
        """
        Compute earliest frame to resume tracking from, based on pending recompute markers.
        """
        if not self.needs_recompute_from:
            resume_start = max(0, requested_start)
        else:
            earliest = min(self.needs_recompute_from.values())
            base = 0 if requested_start is None else requested_start
            resume_start = max(base, earliest)
        print(f"🔁 Resume tracking from frame {resume_start} (kept 0..{resume_start-1})")
        return resume_start

    def _compute_iou(self, boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        """Compute IoU between two bboxes"""
        if boxA is None or boxB is None:
            return 0.0
        (xA, yA, wA, hA) = boxA
        (xB, yB, wB, hB) = boxB
        x1 = max(xA, xB)
        y1 = max(yA, yB)
        x2 = min(xA + wA, xB + wB)
        y2 = min(yA + hA, yB + hB)
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        boxA_area = wA * hA
        boxB_area = wB * hB
        return inter_area / float(boxA_area + boxB_area - inter_area)
    
    def update_trackers(self, frame: np.ndarray, frame_idx: int = None) -> Dict[int, Optional[Tuple[int, int, int, int]]]:
        """
        Update all trackers with current frame
        
        Args:
            frame: Current frame (BGR format)
            frame_idx: Optional frame index for storing results
            
        Returns:
            Dictionary mapping player_id to current bbox (or None if lost)
        """
        results = {}

        for player_id, player in self.players.items():
            # Debug: Print learning frames info every 100 frames
            if frame_idx is not None and frame_idx % 100 == 0:
                print(f"📊 Frame {frame_idx}: Player {player_id} learning_frames={list(player.learning_frames.keys())}")

            # Check if this frame is a learning frame - if so, reinitialize tracker
            if frame_idx is not None and frame_idx in player.learning_frames:
                # This is a learning frame! Use the exact bbox from learning frame
                learning_bbox = player.learning_frames[frame_idx]

                print(f"🔄 LEARNING FRAME DETECTED! Frame {frame_idx}")
                print(f"   Player {player_id}: Reinitializing tracker with bbox={learning_bbox}")

                # Reinitialize tracker with the correct bbox from learning frame
                # CRITICAL: Use init_tracker (not init) for proper initialization
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

                player.current_bbox = bbox
                player.tracking_lost = False
            else:
                # Normal tracking update
                bbox = player.tracker.update(frame)
                player.current_bbox = bbox
                player.tracking_lost = (bbox is None)

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

            results[player_id] = bbox

            # Store result if frame_idx provided
            if frame_idx is not None:
                if player_id not in self.tracking_results:
                    self.tracking_results[player_id] = {}
                self.tracking_results[player_id][frame_idx] = bbox

        return results
    
    def get_bbox_at_frame(self, player_id: int, frame_idx: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box for a player at a specific frame
        
        Args:
            player_id: Player ID
            frame_idx: Frame index
            
        Returns:
            Bounding box or None
        """
        if player_id in self.tracking_results:
            bbox = self.tracking_results[player_id].get(frame_idx)
            if frame_idx % 30 == 0:  # Log every 30 frames
                print(f"get_bbox_at_frame: player={player_id}, frame={frame_idx}, bbox={bbox}")
            return bbox
        else:
            print(f"ERROR: Player {player_id} not found in tracking_results!")
            return None
    
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get specific frame from video - uses multiple strategies for problematic codecs
        
        Args:
            frame_idx: Frame index (0-based)
            
        Returns:
            Frame as numpy array or None if error
        """
        if self.video_path is None:
            return None
        
        if frame_idx < 0 or (self.total_frames > 0 and frame_idx >= self.total_frames):
            return None
        
        try:
            # Strategy 1: Try with existing video_cap with reset
            if self.video_cap is not None and self.video_cap.isOpened():
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset first
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.video_cap.read()
                if ret and frame is not None:
                    return frame

            # Strategy 2: Open new capture and seek - VERIFY seek worked!
            temp_cap = cv2.VideoCapture(self.video_path)
            if temp_cap.isOpened():
                temp_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                # Verify the seek actually worked
                actual_pos = int(temp_cap.get(cv2.CAP_PROP_POS_FRAMES))

                if actual_pos == frame_idx:
                    ret, frame = temp_cap.read()
                    if ret and frame is not None:
                        temp_cap.release()
                        return frame

                # If we got here, Strategy 2 failed - try Strategy 3
                # Strategy 3: Sequential read (for codecs with broken seeking)
                temp_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_frame = None
                for i in range(frame_idx + 1):
                    ret, current_frame = temp_cap.read()
                    if not ret or current_frame is None:
                        temp_cap.release()
                        return None

                temp_cap.release()
                if current_frame is not None:
                    return current_frame

            return None

        except Exception as e:
            print(f"❌ Exception in get_frame for frame {frame_idx}: {e}")
            traceback.print_exc()
            return None
    
    def get_first_frame(self) -> Optional[np.ndarray]:
        """Get first frame of video"""
        return self.get_frame(0)
    
    def get_player(self, player_id: int) -> Optional[PlayerData]:
        """Get player data by ID"""
        return self.players.get(player_id)
    
    def get_all_players(self) -> List[PlayerData]:
        """Get all players - sorted by player_id for consistency"""
        return sorted(list(self.players.values()), key=lambda p: p.player_id)
    
    def remove_player(self, player_id: int) -> bool:
        """
        Remove a player from tracking
        
        Args:
            player_id: ID of player to remove
            
        Returns:
            True if player was removed
        """
        if player_id in self.players:
            self.players[player_id].tracker.reset()
            del self.players[player_id]
            return True
        return False
    
    def reset(self):
        """Reset all trackers"""
        for player in self.players.values():
            player.tracker.reset()
        self.players.clear()
        self.next_player_id = 1
        self.current_frame_idx = 0
        self.tracking_results.clear()
    
    def release(self):
        """Release video capture"""
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None

    def generate_tracking_data(self, start_frame: int = 0,
                              end_frame: Optional[int] = None,
                              progress_callback=None) -> Dict[int, Dict[int, Dict[str, any]]]:
        """
        Phase 1 of two-phase tracking: Generate raw tracking data without rendering.

        This function runs tracking on all players and stores coordinates with confidence scores.
        The data can then be reviewed and corrected before final export.

        Args:
            start_frame: Starting frame index (default: 0)
            end_frame: Ending frame index (default: last frame)
            progress_callback: Optional callback(current_frame, total_frames)

        Returns:
            Dictionary with structure:
            {
                player_id: {
                    frame_index: {
                        'bbox': (x, y, w, h),
                        'confidence': float,  # Tracker confidence (0.0-1.0)
                        'is_learning_frame': bool  # True if this was a user-marked frame
                    }
                }
            }

        Example:
            >>> tracker_manager.load_video("video.mp4")
            >>> tracker_manager.add_player("Player 1", "circle", 0, (100, 100, 50, 50))
            >>> data = tracker_manager.generate_tracking_data(0, 100)
            >>> # Review data, identify problematic frames
            >>> # Add corrections via add_learning_frame_to_player()
            >>> # Re-run generate_tracking_data() or just export
        """
        if self.video_path is None:
            raise ValueError("No video loaded. Call load_video() first.")

        if end_frame is None:
            end_frame = self.total_frames - 1

        # Validate frame range
        end_frame = min(end_frame, self.total_frames - 1)
        if start_frame < 0 or start_frame > end_frame:
            raise ValueError(f"Invalid frame range: {start_frame}-{end_frame}")

        # Determine resume start based on pending invalidations
        resume_start = self.get_resume_start(start_frame)

        # Log learning frames loaded
        for pid, player in self.players.items():
            print(f"[Learning] Player {pid} frames: {sorted(player.learning_frames.keys())}")

        print(f"🎯 Phase 1: Generating tracking data for frames {resume_start}-{end_frame}")
        print(f"   Players to track: {len(self.players)}")

        # Open video capture for sequential reading (faster than seeking)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        # Initialize tracking data structure (persistent)
        tracking_data = self.tracking_data if self.tracking_data else {}
        for player_id in self.players:
            if player_id not in tracking_data:
                tracking_data[player_id] = {}

        # Track last successful bbox per player for failure detection
        previous_bboxes: Dict[int, Optional[Tuple[int, int, int, int]]] = {}

        # Seek to resume_start
        cap.set(cv2.CAP_PROP_POS_FRAMES, resume_start)
        actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # If seeking failed, read sequentially from beginning
        if actual_pos != resume_start:
            print(f"⚠️  Seeking failed (wanted {resume_start}, got {actual_pos}). Reading sequentially...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for i in range(resume_start):
                ret, _ = cap.read()
                if not ret:
                    cap.release()
                    raise RuntimeError(f"Failed to read to start frame {resume_start}")

        # Precompute sorted learning frames for each player (for safe initialization)
        player_learning_frames_sorted: Dict[int, List[int]] = {}
        for player_id, player in self.players.items():
            if player.learning_frames:
                player_learning_frames_sorted[player_id] = sorted(player.learning_frames.keys())
            else:
                # This should normally never happen, but guard against race conditions
                print(f"⚠️  Player {player_id} has no learning_frames; "
                      f"tracking will use initial bbox only and may be less accurate.")
                player_learning_frames_sorted[player_id] = []

        # Remove stale cached entries from resume_start onward
        for player_id in tracking_data.keys():
            for f in list(tracking_data[player_id].keys()):
                if f >= resume_start:
                    del tracking_data[player_id][f]
        for player_id in self.tracking_results.keys():
            for f in list(self.tracking_results[player_id].keys()):
                if f >= resume_start:
                    del self.tracking_results[player_id][f]

        # Seed previous_bboxes with last known bbox before resume_start (if any)
        if resume_start > 0:
            for player_id in self.players:
                prev_frame = resume_start - 1
                prev_entry = tracking_data.get(player_id, {}).get(prev_frame)
                if prev_entry and prev_entry.get('bbox') is not None:
                    previous_bboxes[player_id] = prev_entry['bbox']
                    previous_bboxes[f"{player_id}_smooth"] = prev_entry['bbox']

        # Metrics
        reacquire_attempts = 0
        reacquire_success = 0
        frames_tracked_count = 0
        frames_lost_count = 0
        quality_sum = 0.0
        frames_since_success: Dict[int, int] = {}
        # Track last learning frame per player (to prevent YOLO from overriding corrections)
        last_learning_frame: Dict[int, int] = {}

        # Main tracking loop
        for current_frame_idx in range(resume_start, end_frame + 1):
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"⚠️  Failed to read frame {current_frame_idx}, stopping")
                break

            # Update progress
            if progress_callback:
                progress_callback(current_frame_idx - resume_start + 1, end_frame - resume_start + 1)

            # Track each player
            for player_id, player in self.players.items():
                learning_frames_sorted = player_learning_frames_sorted.get(player_id, [])
                has_learning_frames = bool(learning_frames_sorted)
                is_learning_frame = has_learning_frames and current_frame_idx in player.learning_frames

                failure_reason = None
                quality_score = 1.0
                suspicious_flags = 0
                raw_bbox = None
                smoothed_bbox = None

                # Initialize or reinitialize tracker at learning frames
                if current_frame_idx == resume_start or is_learning_frame:
                    init_bbox = None

                    # Use learning frame bbox if available, otherwise use player's initial bbox
                    if is_learning_frame:
                        init_bbox = player.learning_frames[current_frame_idx]
                        print(f"🔄 Frame {current_frame_idx}: Reinitializing player {player_id} with learning frame bbox={init_bbox}")
                    elif current_frame_idx == resume_start:
                        if has_learning_frames:
                            past_or_equal = [f for f in learning_frames_sorted if f <= resume_start]
                            if past_or_equal:
                                closest_frame = max(past_or_equal)
                                init_bbox = player.learning_frames[closest_frame]
                                print(
                                    f"🔄 Frame {current_frame_idx}: Initializing player {player_id} "
                                    f"with latest learning frame <= start (frame {closest_frame}) "
                                    f"bbox={init_bbox}"
                                )
                            else:
                                first_future = learning_frames_sorted[0]
                                print(
                                    f"⏸ Frame {current_frame_idx}: No learning frame before start for "
                                    f"player {player_id} (first at frame {first_future}); "
                                    f"skipping initialization until then."
                                )
                                init_bbox = None
                        else:
                            init_bbox = getattr(player, "bbox", None)
                            print(
                                f"⚠️  Frame {current_frame_idx}: Player {player_id} has NO learning_frames; "
                                f"falling back to player.bbox={init_bbox}"
                            )

                    if init_bbox is not None:
                        player.tracker.init_tracker(frame, init_bbox)
                        bbox = init_bbox
                        raw_bbox = bbox
                        success = True
                        confidence = 1.0
                        # Mark this as a learning frame for grace period
                        if is_learning_frame:
                            last_learning_frame[player_id] = current_frame_idx
                    else:
                        bbox = None
                        raw_bbox = None
                        success = False
                        confidence = 0.0
                else:
                    # Normal tracking update
                    bbox = player.tracker.update(frame)
                    raw_bbox = bbox
                    success = (bbox is not None)

                    confidence = 0.8 if success else 0.0

                    prev_bbox = previous_bboxes.get(player_id)
                    if success and bbox is not None and prev_bbox is not None:
                        iou_prev = self._compute_iou(bbox, prev_bbox)
                        if iou_prev < self.tracking_config["iou_min"]:
                            suspicious_flags += 1
                            quality_score -= 0.25

                        px, py, pw, ph = prev_bbox
                        if pw > 0 and ph > 0:
                            size_change = abs((bbox[2] * bbox[3]) - (pw * ph)) / (pw * ph)
                        else:
                            size_change = 0.0
                        if size_change > self.tracking_config["scale_change_max"]:
                            suspicious_flags += 1
                            quality_score -= 0.25

                        center_shift = ((bbox[0] + bbox[2] / 2) - (px + pw / 2)) ** 2 + ((bbox[1] + bbox[3] / 2) - (py + ph / 2)) ** 2
                        center_distance = center_shift ** 0.5
                        if center_distance > self.tracking_config["center_jump_px"]:
                            suspicious_flags += 1
                            quality_score -= 0.25

                        if suspicious_flags >= 2:
                            failure_reason = 'QUALITY_DROP'
                            success = False
                            confidence = max(0.0, quality_score)
                    else:
                        if not success:
                            quality_score = 0.0

                # Hybrid YOLO reacquire if needed
                allow_reacquire = False

                # Check if we're in grace period after learning frame
                in_grace_period = False
                if player_id in last_learning_frame:
                    frames_since_learning = current_frame_idx - last_learning_frame[player_id]
                    grace_frames = self.tracking_config.get("learning_frame_grace", 20)
                    if frames_since_learning < grace_frames:
                        in_grace_period = True
                        # Log grace period protection (only on failed tracks)
                        if not success and current_frame_idx % 10 == 0:
                            print(f"🛡️  Grace period active for player {player_id}: {frames_since_learning}/{grace_frames} frames since correction")

                # Only allow reacquisition if NOT in grace period
                if not in_grace_period:
                    if not success:
                        allow_reacquire = frames_since_success.get(player_id, 0) % max(1, self.tracking_config["reacquire_interval"]) == 0
                    if suspicious_flags >= 2:
                        allow_reacquire = True

                if self.tracking_config["mode"] == "hybrid" and allow_reacquire and not in_grace_period:
                    if self.person_detector.is_available():
                        reacquire_attempts += 1
                        dets = self.person_detector.detect_people(frame, confidence_threshold=0.25)
                        best_score = 0.0
                        best_bbox = None
                        prev_for_match = previous_bboxes.get(player_id)
                        for (dx, dy, dw, dh, conf_det) in dets:
                            candidate = (dx, dy, dw, dh)
                            iou = self._compute_iou(candidate, prev_for_match) if prev_for_match else 0.0
                            if prev_for_match:
                                px, py, pw, ph = prev_for_match
                                c1x, c1y = dx + dw / 2, dy + dh / 2
                                c2x, c2y = px + pw / 2, py + ph / 2
                                center_dist = ((c1x - c2x) ** 2 + (c1y - c2y) ** 2) ** 0.5
                                center_score = 1.0 / (1.0 + center_dist / max(self.tracking_config["center_jump_px"], 1))
                                prev_area = pw * ph
                                cand_area = dw * dh
                                size_ratio = min(cand_area, prev_area) / max(cand_area, prev_area) if prev_area > 0 else 0.5
                            else:
                                center_score = 0.5
                                size_ratio = 0.5
                            score = 0.5 * iou + 0.3 * center_score + 0.2 * size_ratio
                            if score > best_score:
                                best_score = score
                                best_bbox = candidate
                        if best_bbox is not None and best_score >= 0.25:
                            # IMPORTANT: Mark as low confidence if score is suspicious
                            # This helps identify frames where YOLO might have picked wrong person
                            if best_score < 0.5:
                                print(f"⚠️  Reacquire with LOW confidence for player {player_id} at frame {current_frame_idx} (score={best_score:.2f}) - VERIFY THIS!")
                                confidence = 0.5  # Mark as suspicious
                            else:
                                print(f"🔎 Reacquire success for player {player_id} at frame {current_frame_idx} (score={best_score:.2f})")
                                confidence = max(confidence, min(1.0, best_score + 0.3))

                            reacquire_success += 1
                            player.tracker.init_tracker(frame, best_bbox)
                            bbox = best_bbox
                            raw_bbox = best_bbox
                            success = True
                            failure_reason = None
                        else:
                            failure_reason = failure_reason or 'TRACKER_LOST'
                    else:
                        print("⚠️ YOLO unavailable - hybrid mode falling back to tracker only")

                # Smoothing (EMA)
                if success and raw_bbox is not None:
                    prev_smooth = previous_bboxes.get(f"{player_id}_smooth")
                    if prev_smooth is None:
                        smoothed_bbox = raw_bbox
                    else:
                        # Dynamic smoothing: less smoothing when moving fast to reduce lag
                        base_alpha = self.tracking_config.get("smoothing_alpha", 0.35)
                        c_raw = (raw_bbox[0] + raw_bbox[2] / 2, raw_bbox[1] + raw_bbox[3] / 2)
                        c_prev = (prev_smooth[0] + prev_smooth[2] / 2, prev_smooth[1] + prev_smooth[3] / 2)
                        center_distance = ((c_raw[0] - c_prev[0]) ** 2 + (c_raw[1] - c_prev[1]) ** 2) ** 0.5
                        jump_thresh = self.tracking_config.get("center_jump_px", 80.0) * 0.5
                        if center_distance > jump_thresh:
                            alpha = max(0.15, base_alpha * 0.5)
                        else:
                            alpha = base_alpha
                        sx = int(alpha * raw_bbox[0] + (1 - alpha) * prev_smooth[0])
                        sy = int(alpha * raw_bbox[1] + (1 - alpha) * prev_smooth[1])
                        sw = int(alpha * raw_bbox[2] + (1 - alpha) * prev_smooth[2])
                        sh = int(alpha * raw_bbox[3] + (1 - alpha) * prev_smooth[3])
                        smoothed_bbox = (sx, sy, sw, sh)
                    previous_bboxes[f"{player_id}_smooth"] = smoothed_bbox

                # Calculate original bbox (without padding) for accurate marker placement
                original_bbox = None
                if success and smoothed_bbox is not None:
                    if hasattr(player, 'padding_offset') and player.padding_offset != (0, 0, 0, 0):
                        x, y, w, h = smoothed_bbox
                        offset_x, offset_y, offset_w, offset_h = player.padding_offset
                        # Reverse the padding: original = padded + offset
                        orig_x = x + offset_x
                        orig_y = y + offset_y
                        orig_w = w - offset_w
                        orig_h = h - offset_h
                        original_bbox = (orig_x, orig_y, orig_w, orig_h)
                    else:
                        original_bbox = smoothed_bbox

                # Store tracking data
                if success and smoothed_bbox is not None:
                    tracking_data[player_id][current_frame_idx] = {
                        'bbox': smoothed_bbox,
                        'original_bbox': original_bbox,  # CRITICAL: Store original bbox for marker placement
                        'bbox_raw': raw_bbox,
                        'confidence': max(confidence, quality_score),
                        'is_learning_frame': is_learning_frame
                    }

                    if player_id not in self.tracking_results:
                        self.tracking_results[player_id] = {}
                    self.tracking_results[player_id][current_frame_idx] = smoothed_bbox

                    previous_bboxes[player_id] = smoothed_bbox
                    frames_tracked_count += 1
                    quality_sum += max(confidence, quality_score)
                    frames_since_success[player_id] = 0
                else:
                    tracking_data[player_id][current_frame_idx] = {
                        'bbox': None,
                        'original_bbox': None,
                        'bbox_raw': raw_bbox,
                        'confidence': 0.0,
                        'is_learning_frame': is_learning_frame,
                        'failure_reason': failure_reason if failure_reason else 'TRACKER_LOST'
                    }

                    if player_id not in self.tracking_results:
                        self.tracking_results[player_id] = {}
                    self.tracking_results[player_id][current_frame_idx] = None
                    frames_lost_count += 1
                    frames_since_success[player_id] = frames_since_success.get(player_id, 0) + 1

            # Log progress every 50 frames
            if current_frame_idx % 50 == 0:
                print(f"  ⚡ Processed {current_frame_idx - resume_start + 1}/{end_frame - resume_start + 1} frames")

        cap.release()

        # Persist tracking data and clear recompute markers
        self.tracking_data = tracking_data
        self.needs_recompute_from.clear()

        # Summary statistics
        print(f"\n✅ Phase 1 Complete: Generated tracking data")
        for player_id, player in self.players.items():
            frames_tracked = len([f for f in tracking_data[player_id] if tracking_data[player_id][f]['bbox'] is not None])
            frames_lost = len([f for f in tracking_data[player_id] if tracking_data[player_id][f]['bbox'] is None])
            learning_frames_count = len([f for f in tracking_data[player_id] if tracking_data[player_id][f]['is_learning_frame']])
            avg_confidence = sum([tracking_data[player_id][f]['confidence'] for f in tracking_data[player_id]]) / len(tracking_data[player_id]) if tracking_data[player_id] else 0.0

            print(f"   Player {player_id} ({player.name}):")
            print(f"      Frames tracked: {frames_tracked}")
            print(f"      Frames lost: {frames_lost}")
            print(f"      Learning frames used: {learning_frames_count}")
            print(f"      Average confidence: {avg_confidence:.2f}")

        # Global metrics
        avg_quality = quality_sum / frames_tracked_count if frames_tracked_count else 0.0
        print(f"\n📈 Hybrid Metrics:")
        print(f"   Tracked frames: {frames_tracked_count}, Lost: {frames_lost_count}")
        print(f"   Reacquire attempts: {reacquire_attempts}, success: {reacquire_success}")
        print(f"   Avg quality score: {avg_quality:.2f}")

        # Identify suspicious frames (low confidence)
        print(f"\n🔍 Quality Analysis:")
        for player_id, player in self.players.items():
            suspicious_frames = []
            for frame_idx, frame_data in tracking_data[player_id].items():
                conf = frame_data.get('confidence', 0.0)
                # Flag frames with confidence < 0.6 (suspicious)
                if 0 < conf < 0.6 and not frame_data.get('is_learning_frame'):
                    suspicious_frames.append((frame_idx, conf))

            if suspicious_frames:
                print(f"   ⚠️  Player {player_id} ({player.name}): {len(suspicious_frames)} SUSPICIOUS frames detected!")
                # Show first 5 suspicious frames
                for frame_idx, conf in sorted(suspicious_frames)[:5]:
                    print(f"      Frame {frame_idx}: confidence={conf:.2f} - VERIFY THIS FRAME!")
                if len(suspicious_frames) > 5:
                    print(f"      ... and {len(suspicious_frames) - 5} more")
            else:
                print(f"   ✅ Player {player_id} ({player.name}): No suspicious frames detected")

        print(f"\n💡 Next steps:")
        print(f"   1. Review tracking data to identify problematic frames")
        print(f"   2. Add corrections with add_learning_frame_to_player()")
        print(f"   3. Re-run generate_tracking_data() or export directly")

        return tracking_data
