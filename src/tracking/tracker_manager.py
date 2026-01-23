"""
Tracker Manager - Manages multiple player trackers
"""
import cv2
import numpy as np
import traceback
from typing import List, Dict, Optional, Tuple
from .player_tracker import PlayerTracker, TrackerType
from .person_detector import PersonDetector


class RadarKeyframe:
    """Data structure for a radar keyframe"""
    def __init__(self, frame_idx: int, angle: float, size: float = 1.0):
        """
        Args:
            frame_idx: Frame index for this keyframe
            angle: Direction angle in radians (0 = right, pi/2 = down, -pi/2 = up)
            size: Size multiplier for cone length (1.0 = default, 0.5 = half, 2.0 = double)
        """
        self.frame_idx = frame_idx
        self.angle = angle  # radians
        self.size = size  # multiplier


class PlayerData:
    """Data structure for a tracked player"""
    def __init__(self, player_id: int, name: str, marker_style: str,
                 initial_frame: int, bbox: Tuple[int, int, int, int],
                 original_bbox: Optional[Tuple[int, int, int, int]] = None):
        self.player_id = player_id
        self.name = name
        self.marker_style = marker_style
        self.initial_frame = initial_frame
        self.bbox = bbox  # This is the PADDED bbox
        self.original_bbox = original_bbox or bbox  # Original bbox BEFORE padding

        self.learning_frames: Dict[int, Tuple[int, int, int, int]] = {initial_frame: bbox}
        self.original_learning_frames: Dict[int, Tuple[int, int, int, int]] = {initial_frame: original_bbox or bbox}
        self.tracker = PlayerTracker(TrackerType.MIL)
        self.current_bbox = bbox
        self.current_original_bbox = original_bbox or bbox
        self.tracking_lost = False
        self.color = self._get_default_color()

        # Radar keyframes for manual radar direction/size control
        # Dict[frame_idx, RadarKeyframe]
        self.radar_keyframes: Dict[int, RadarKeyframe] = {}
        
        # Radar COLOR keyframes - allows changing color at specific frames
        # Dict[frame_idx, color_name] where color_name is 'green' or 'red'
        # Default color is green, changes apply from that frame onwards
        self.radar_color_keyframes: Dict[int, str] = {}
        
        # Per-player tracking time range
        # If None, uses project's global tracking range
        self.player_start_frame: Optional[int] = None
        self.player_end_frame: Optional[int] = None

        # Calculate padding offset
        if original_bbox and original_bbox != bbox:
            orig_x, orig_y, orig_w, orig_h = original_bbox
            pad_x, pad_y, pad_w, pad_h = bbox
            self.padding_offset = (orig_x - pad_x, orig_y - pad_y, pad_w - orig_w, pad_h - orig_h)
        else:
            self.padding_offset = (0, 0, 0, 0)
    
    def add_learning_frame(self, frame_idx: int, bbox: Tuple[int, int, int, int],
                          original_bbox: Optional[Tuple[int, int, int, int]] = None):
        """Add a learning frame for this player"""
        self.learning_frames[frame_idx] = bbox
        self.original_learning_frames[frame_idx] = original_bbox or bbox
        if frame_idx < self.initial_frame:
            self.initial_frame = frame_idx
            self.bbox = bbox
            self.original_bbox = original_bbox or bbox
    
    def _get_default_color(self) -> Tuple[int, int, int]:
        """Get default color based on marker style"""
        color_map = {
            'dynamic_ring_3d': (255, 0, 180),  # Broadcast Purple
            'spotlight_alien': (200, 255, 255),  # Cyan
            'solid_anchor': (0, 255, 100),  # Green
            'radar_defensive': (0, 50, 255),  # Red-Orange
            'sniper_scope': (0, 0, 255),  # Red
            'ball_marker': (0, 165, 255),  # Orange
            'fireball_trail': (0, 100, 255),  # Orange-Red
            'energy_rings': (255, 200, 0),  # Cyan
        }
        return color_map.get(self.marker_style, (255, 255, 255))

    def add_radar_keyframe(self, frame_idx: int, angle: float, size: float = 1.0):
        """Add a radar keyframe for manual radar control"""
        self.radar_keyframes[frame_idx] = RadarKeyframe(frame_idx, angle, size)

    def remove_radar_keyframe(self, frame_idx: int) -> bool:
        """Remove a radar keyframe"""
        if frame_idx in self.radar_keyframes:
            del self.radar_keyframes[frame_idx]
            return True
        return False

    def get_radar_params_at_frame(self, frame_idx: int) -> Optional[Tuple[float, float]]:
        """
        Get interpolated radar parameters (angle, size) for a given frame.
        Returns None if no keyframes are set (use automatic targeting).
        Returns (angle, size) if keyframes exist.
        """
        if not self.radar_keyframes:
            return None

        sorted_frames = sorted(self.radar_keyframes.keys())

        # Exact match
        if frame_idx in self.radar_keyframes:
            kf = self.radar_keyframes[frame_idx]
            return (kf.angle, kf.size)

        # Before first keyframe - use first keyframe values
        if frame_idx < sorted_frames[0]:
            kf = self.radar_keyframes[sorted_frames[0]]
            return (kf.angle, kf.size)

        # After last keyframe - use last keyframe values
        if frame_idx > sorted_frames[-1]:
            kf = self.radar_keyframes[sorted_frames[-1]]
            return (kf.angle, kf.size)

        # Interpolate between two keyframes
        prev_frame = None
        next_frame = None
        for i, f in enumerate(sorted_frames):
            if f > frame_idx:
                next_frame = f
                prev_frame = sorted_frames[i - 1]
                break

        if prev_frame is None or next_frame is None:
            return None

        # Linear interpolation
        t = (frame_idx - prev_frame) / (next_frame - prev_frame)
        kf1 = self.radar_keyframes[prev_frame]
        kf2 = self.radar_keyframes[next_frame]

        # Interpolate angle (handle wrap-around)
        angle_diff = kf2.angle - kf1.angle
        # Normalize to [-pi, pi]
        import math
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        angle = kf1.angle + t * angle_diff

        # Interpolate size
        size = kf1.size + t * (kf2.size - kf1.size)

        return (angle, size)

    def has_radar_keyframes(self) -> bool:
        """Check if this player has any radar keyframes"""
        return len(self.radar_keyframes) > 0

    # === RADAR COLOR KEYFRAMES ===
    
    def set_radar_color_at_frame(self, frame_idx: int, color_name: str):
        """
        Set radar color from this frame onwards.
        color_name: 'green' or 'red'
        """
        if color_name not in ('green', 'red'):
            raise ValueError("Color must be 'green' or 'red'")
        self.radar_color_keyframes[frame_idx] = color_name
    
    def remove_radar_color_keyframe(self, frame_idx: int) -> bool:
        """Remove a radar color keyframe"""
        if frame_idx in self.radar_color_keyframes:
            del self.radar_color_keyframes[frame_idx]
            return True
        return False
    
    def get_radar_color_at_frame(self, frame_idx: int) -> Tuple[int, int, int]:
        """
        Get radar color (BGR) for a given frame.
        Returns green by default, or the color set by the most recent keyframe before this frame.
        """
        # Default colors (BGR)
        RADAR_GREEN = (0, 255, 100)  # Bright green
        RADAR_RED = (0, 50, 255)     # Red-orange
        
        if not self.radar_color_keyframes:
            return RADAR_GREEN  # Default is green
        
        # Find the most recent keyframe at or before this frame
        sorted_frames = sorted(self.radar_color_keyframes.keys())
        active_color = 'green'  # Default
        
        for kf_frame in sorted_frames:
            if kf_frame <= frame_idx:
                active_color = self.radar_color_keyframes[kf_frame]
            else:
                break
        
        return RADAR_GREEN if active_color == 'green' else RADAR_RED
    
    def get_radar_color_keyframes_summary(self) -> str:
        """Get a summary of color keyframes for display"""
        if not self.radar_color_keyframes:
            return "Default (green)"
        parts = []
        for frame, color in sorted(self.radar_color_keyframes.items()):
            parts.append(f"Frame {frame}: {color}")
        return ", ".join(parts)

    # === PER-PLAYER TRACKING RANGE ===
    
    def set_tracking_range(self, start_frame: Optional[int], end_frame: Optional[int]):
        """Set custom tracking range for this player"""
        self.player_start_frame = start_frame
        self.player_end_frame = end_frame
    
    def get_tracking_range(self) -> Tuple[Optional[int], Optional[int]]:
        """Get custom tracking range (start, end)"""
        return (self.player_start_frame, self.player_end_frame)
    
    def is_visible_at_frame(self, frame_idx: int, global_start: int = 0, global_end: int = None) -> bool:
        """
        Check if this player should be visible at the given frame.
        
        IMPORTANT: A player is NEVER visible before their initial_frame (the frame
        where they were first marked). This ensures that if you mark a player at
        frame 200, they won't appear at frame 0.
        
        Uses player-specific range if set, otherwise uses global range,
        but always respects initial_frame as the absolute minimum.
        """
        # CRITICAL: Never show marker before the frame where player was first defined
        if frame_idx < self.initial_frame:
            return False
        
        # Check custom player range (if set)
        start = self.player_start_frame if self.player_start_frame is not None else global_start
        end = self.player_end_frame if self.player_end_frame is not None else global_end
        
        # Apply range constraints (but initial_frame already ensures minimum)
        if start is not None and frame_idx < start:
            return False
        if end is not None and frame_idx > end:
            return False
        return True


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
        self.tracking_results: Dict[int, Dict[int, Tuple[int, int, int, int]]] = {}
        self.tracking_data: Dict[int, Dict[int, Dict[str, any]]] = {}
        self.needs_recompute_from: Dict[int, int] = {}
        self.tracking_config = {
            "mode": "hybrid",
            "iou_min": 0.15,
            "scale_change_max": 0.35,
            "center_jump_px": 80.0,
            "reacquire_interval": 5,
            "smoothing_alpha": 0.65,
            "lost_patience": 8,
            "learning_frame_grace": 20
        }
        self.person_detector = PersonDetector()
    
    def _is_valid_fps(self, fps: float) -> bool:
        return 1 <= fps <= 240
    
    def _is_valid_frame_count(self, frame_count: float) -> bool:
        return 1 <= frame_count <= 100000
    
    def _count_frames(self, cap: cv2.VideoCapture) -> int:
        frame_count = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, _ = cap.read()
            if not ret: break
            frame_count += 1
            if frame_count > 100000: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return frame_count
    
    def probe_video(self, video_path: str) -> Optional[Dict[str, float]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not self._is_valid_fps(fps): fps = 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not self._is_valid_frame_count(frame_count): frame_count = self._count_frames(cap)
        else: frame_count = int(frame_count)
        duration = frame_count / fps if fps > 0 else 0.0
        cap.release()
        return {"fps": fps, "frame_count": frame_count, "duration": duration, "width": width, "height": height}
        
    def load_video(self, video_path: str, metadata: Optional[Dict[str, float]] = None) -> bool:
        try:
            if metadata is None: metadata = self.probe_video(video_path)
            if metadata is None: return False
            if self.video_cap is not None: self.video_cap.release()
            video_cap = cv2.VideoCapture(video_path)
            if not video_cap.isOpened(): return False
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
            return False
    
    def add_player(self, name: str, marker_style: str,
                   initial_frame: int, bbox: Tuple[int, int, int, int],
                   original_bbox: Optional[Tuple[int, int, int, int]] = None) -> int:
        player_id = self.next_player_id
        self.next_player_id += 1
        player = PlayerData(player_id, name, marker_style, initial_frame, bbox, original_bbox)
        self.players[player_id] = player
        if player_id not in self.tracking_data: self.tracking_data[player_id] = {}
        if player_id not in self.tracking_results: self.tracking_results[player_id] = {}
        return player_id
    
    def add_learning_frame_to_player(self, player_id: int, frame_idx: int, bbox: Tuple[int, int, int, int],
                                    original_bbox: Optional[Tuple[int, int, int, int]] = None,
                                    preserve_frame: bool = False) -> bool:
        """Add a learning frame. If preserve_frame=True, keep the frame itself (for corrections)."""
        if player_id not in self.players: return False
        self.players[player_id].add_learning_frame(int(frame_idx), tuple(int(v) for v in bbox),
                                                 tuple(int(v) for v in original_bbox) if original_bbox else None)
        self.invalidate_tracking_from(player_id, frame_idx, include_current=not preserve_frame)
        return True
    
    def invalidate_tracking_from(self, player_id: int, frame_idx: int, include_current: bool = True):
        """Invalidate tracking from a frame. include_current=False keeps that frame."""
        if player_id not in self.needs_recompute_from:
            self.needs_recompute_from[player_id] = frame_idx
        else:
            self.needs_recompute_from[player_id] = min(self.needs_recompute_from[player_id], frame_idx)

        compare = (lambda f: f >= frame_idx) if include_current else (lambda f: f > frame_idx)

        if player_id in self.tracking_data:
            for f in list(self.tracking_data[player_id].keys()):
                if compare(f): del self.tracking_data[player_id][f]
        if player_id in self.tracking_results:
            for f in list(self.tracking_results[player_id].keys()):
                if compare(f): del self.tracking_results[player_id][f]
    
    def get_resume_start(self, requested_start: int = 0) -> int:
        if not self.needs_recompute_from: return max(0, requested_start)
        return max(0 if requested_start is None else requested_start, min(self.needs_recompute_from.values()))

    def _compute_iou(self, boxA, boxB):
        if boxA is None or boxB is None: return 0.0
        x1 = max(boxA[0], boxB[0])
        y1 = max(boxA[1], boxB[1])
        x2 = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        y2 = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0: return 0.0
        return inter / float(boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter)
    
    def update_trackers(self, frame, frame_idx=None):
        results = {}
        for player_id, player in self.players.items():
            if frame_idx is not None and frame_idx in player.learning_frames:
                learning_bbox = player.learning_frames[frame_idx]
                player.tracker.init_tracker(frame, learning_bbox)
                bbox = learning_bbox
                player.current_original_bbox = player.original_learning_frames.get(frame_idx, bbox)
                player.current_bbox = bbox
                player.tracking_lost = False
            else:
                bbox = player.tracker.update(frame)
                player.current_bbox = bbox
                player.tracking_lost = (bbox is None)
                if bbox is not None and player.padding_offset != (0, 0, 0, 0):
                    px, py, pw, ph = player.padding_offset
                    player.current_original_bbox = (bbox[0] + px, bbox[1] + py, bbox[2] - pw, bbox[3] - ph)
                else: player.current_original_bbox = bbox
            results[player_id] = bbox
            if frame_idx is not None:
                if player_id not in self.tracking_results: self.tracking_results[player_id] = {}
                self.tracking_results[player_id][frame_idx] = bbox
        return results
    
    def get_bbox_at_frame(self, player_id, frame_idx):
        return self.tracking_results.get(player_id, {}).get(frame_idx)
    
    def get_frame(self, frame_idx):
        if self.video_path is None or frame_idx < 0: return None
        if self.video_cap and self.video_cap.isOpened():
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video_cap.read()
            if ret: return frame
        return None
    
    def get_first_frame(self): return self.get_frame(0)
    def get_player(self, player_id): return self.players.get(player_id)
    def get_all_players(self): return sorted(list(self.players.values()), key=lambda p: p.player_id)
    
    def remove_player(self, player_id: int) -> bool:
        """Remove a player from tracking"""
        if player_id not in self.players:
            return False
        del self.players[player_id]
        if player_id in self.tracking_data:
            del self.tracking_data[player_id]
        if player_id in self.tracking_results:
            del self.tracking_results[player_id]
        if player_id in self.needs_recompute_from:
            del self.needs_recompute_from[player_id]
        return True
    
    def release(self):
        if self.video_cap: self.video_cap.release(); self.video_cap = None

    def _is_ball_marker(self, marker_style: str) -> bool:
        """Check if this is a ball marker style"""
        return marker_style in ('ball_marker', 'fireball_trail', 'energy_rings')
    
    def _try_reacquire_ball(self, frame: np.ndarray, last_bbox: Tuple[int, int, int, int], 
                           search_radius: int = 150) -> Optional[Tuple[int, int, int, int]]:
        """
        Try to reacquire a lost ball using YOLO detection.
        Searches in a region around the last known position.
        
        Args:
            frame: Current video frame
            last_bbox: Last known bounding box of the ball
            search_radius: Radius to search around last position
            
        Returns:
            New bounding box if found, None otherwise
        """
        if last_bbox is None:
            return None
        
        try:
            # Calculate search region around last known position
            lx, ly, lw, lh = last_bbox
            center_x = lx + lw // 2
            center_y = ly + lh // 2
            
            h, w = frame.shape[:2]
            
            # Define search region (expand from last position)
            x1 = max(0, center_x - search_radius)
            y1 = max(0, center_y - search_radius)
            x2 = min(w, center_x + search_radius)
            y2 = min(h, center_y + search_radius)
            
            # Extract search region
            search_region = frame[y1:y2, x1:x2]
            
            if search_region.size == 0:
                        return None

            # Run ball detection on search region
            detections = self.person_detector.detect_balls(search_region, confidence_threshold=0.05)
            
            if detections:
                # Find the detection closest to the center (most likely to be our ball)
                best_det = None
                best_dist = float('inf')
                region_center_x = (x2 - x1) // 2
                region_center_y = (y2 - y1) // 2
                
                for det in detections:
                    dx, dy, dw, dh, conf = det
                    det_center_x = dx + dw // 2
                    det_center_y = dy + dh // 2
                    dist = ((det_center_x - region_center_x) ** 2 + (det_center_y - region_center_y) ** 2) ** 0.5
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_det = det
                
                if best_det:
                    dx, dy, dw, dh, conf = best_det
                    # Convert back to full frame coordinates
                    return (x1 + dx, y1 + dy, dw, dh)
            
            # If not found in search region, try full frame with aggressive detection
            if hasattr(self.person_detector, 'detect_balls_aggressive'):
                full_detections = self.person_detector.detect_balls_aggressive(frame)
                if full_detections:
                    # Find closest to last known position
                    best_det = None
                    best_dist = float('inf')
                    
                    for det in full_detections:
                        dx, dy, dw, dh, conf = det
                        det_center_x = dx + dw // 2
                        det_center_y = dy + dh // 2
                        dist = ((det_center_x - center_x) ** 2 + (det_center_y - center_y) ** 2) ** 0.5
                        
                        # Only consider if reasonably close (within 2x search radius)
                        if dist < search_radius * 2 and dist < best_dist:
                            best_dist = dist
                            best_det = det
                    
                    if best_det:
                        dx, dy, dw, dh, conf = best_det
                        return (dx, dy, dw, dh)

            return None

        except Exception as e:
            print(f"⚠️ Ball reacquisition error: {e}")
            return None
    
    def generate_tracking_data(self, start_frame=0, end_frame=None, progress_callback=None):
        if self.video_path is None: raise ValueError("No video loaded")
        if end_frame is None: end_frame = self.total_frames - 1
        resume_start = self.get_resume_start(start_frame)
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, resume_start)
        
        tracking_data = self.tracking_data
        
        # IMPORTANT: For each player, find their last known bbox BEFORE resume_start
        # This preserves tracking continuity when adding corrections
        last_bbox_before_resume: Dict[int, Optional[Tuple[int, int, int, int]]] = {}
        
        for pid in self.players:
            if pid not in tracking_data: tracking_data[pid] = {}
            
            # Find the last known bbox before resume_start (to preserve tracking continuity)
            last_bbox = None
            if tracking_data[pid]:
                frames_before = [f for f in tracking_data[pid].keys() if f < resume_start]
                if frames_before:
                    last_frame = max(frames_before)
                    data = tracking_data[pid][last_frame]
                    if data.get('bbox'):
                        last_bbox = data['bbox']
            last_bbox_before_resume[pid] = last_bbox
            
            # Only delete tracking data FROM resume_start onwards (preserve earlier data!)
            for f in list(tracking_data[pid].keys()):
                if f >= resume_start: del tracking_data[pid][f]

        # Track frames since each player's tracking was lost (for ball reacquisition)
        frames_lost: Dict[int, int] = {}
        last_good_bbox: Dict[int, Tuple[int, int, int, int]] = {}

        for f_idx in range(resume_start, end_frame + 1):
            ret, frame = cap.read()
            if not ret: break
            if progress_callback: progress_callback(f_idx - resume_start + 1, end_frame - resume_start + 1)
            
            for pid, player in self.players.items():
                # Skip frames before player's initial_frame
                # (player shouldn't be tracked before they were first marked)
                if f_idx < player.initial_frame:
                    continue
                
                is_learning = f_idx in player.learning_frames
                is_ball = self._is_ball_marker(player.marker_style)
                
                # Determine what bbox to use for initialization
                if is_learning:
                    # User explicitly marked this frame - use their bbox
                    bbox = player.learning_frames[f_idx]
                    player.tracker.init_tracker(frame, bbox)
                    success, conf = True, 1.0
                    frames_lost[pid] = 0
                    last_good_bbox[pid] = bbox
                elif f_idx == resume_start:
                    # First frame of this tracking run
                    # Priority: learning frame > last known bbox > initial bbox
                    if f_idx in player.learning_frames:
                        bbox = player.learning_frames[f_idx]
                    elif last_bbox_before_resume.get(pid):
                        # Use last known bbox to maintain continuity
                        bbox = last_bbox_before_resume[pid]
                    else:
                        # Fall back to initial bbox (only if this IS the initial frame)
                        bbox = player.bbox
                    player.tracker.init_tracker(frame, bbox)
                    success, conf = True, 1.0
                    frames_lost[pid] = 0
                    last_good_bbox[pid] = bbox
                elif f_idx == player.initial_frame:
                    # This is the very first frame for this player
                    bbox = player.learning_frames.get(f_idx, player.bbox)
                    player.tracker.init_tracker(frame, bbox)
                    success, conf = True, 1.0
                    frames_lost[pid] = 0
                    last_good_bbox[pid] = bbox
                else:
                    bbox = player.tracker.update(frame)
                    success = bbox is not None
                    conf = 0.8 if success else 0.0
                    
                    # Ball tracking enhancement: try to reacquire when lost
                    if not success and is_ball:
                        frames_lost[pid] = frames_lost.get(pid, 0) + 1
                        
                        # Try to reacquire every few frames (not every frame for performance)
                        if frames_lost[pid] <= 30 and frames_lost[pid] % 3 == 0:
                            last_bbox = last_good_bbox.get(pid)
                            if last_bbox:
                                print(f"🔍 Ball lost for player {pid}, attempting reacquisition (frame {f_idx})...")
                                reacquired_bbox = self._try_reacquire_ball(frame, last_bbox)
                                if reacquired_bbox:
                                    print(f"✅ Ball reacquired at {reacquired_bbox}")
                                    bbox = reacquired_bbox
                                    player.tracker.init_tracker(frame, bbox)
                            success = True
                                    conf = 0.6  # Lower confidence for reacquired
                                    frames_lost[pid] = 0
                    
                    if success:
                        frames_lost[pid] = 0
                        last_good_bbox[pid] = bbox
                
                orig_bbox = None
                if success:
                    px, py, pw, ph = player.padding_offset
                    orig_bbox = (bbox[0] + px, bbox[1] + py, bbox[2] - pw, bbox[3] - ph) if player.padding_offset != (0, 0, 0, 0) else bbox
                
                tracking_data[pid][f_idx] = {'bbox': bbox, 'original_bbox': orig_bbox, 'confidence': conf, 'is_learning_frame': is_learning}
                if pid not in self.tracking_results: self.tracking_results[pid] = {}
                self.tracking_results[pid][f_idx] = bbox

        cap.release()
        self.tracking_data = tracking_data
        self.needs_recompute_from.clear()
        return tracking_data
