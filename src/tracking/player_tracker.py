"""
Player Tracker - Handles tracking of individual players using OpenCV trackers
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
from enum import Enum


class TrackerType(Enum):
    """Available tracker types"""
    CSRT = "csrt"
    KCF = "kcf"


class PlayerTracker:
    """Tracks a single player through video frames"""
    
    def __init__(self, tracker_type: TrackerType = TrackerType.CSRT):
        """
        Initialize player tracker
        
        Args:
            tracker_type: Type of tracker to use (CSRT or KCF)
        """
        self.tracker_type = tracker_type
        self.tracker = None
        self.bbox = None
        self.is_initialized = False
        self.tracking_lost = False
        self.smoothing_buffer = []
        self.buffer_size = 10  # Increased from 5 to 10 for smoother tracking
        
    def init_tracker(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Initialize tracker with bounding box in first frame
        
        Args:
            frame: First frame (BGR format)
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            True if initialization successful
        """
        print(f"init_tracker called: frame.shape={frame.shape}, bbox={bbox}")
        try:
            # Validate inputs
            if frame is None or len(frame.shape) != 3:
                print(f"ERROR: Invalid frame shape: {frame.shape if frame is not None else None}")
                return False
            
            frame_h, frame_w = frame.shape[:2]
            x, y, w, h = bbox
            
            # Validate bbox dimensions
            if w <= 0 or h <= 0:
                print(f"ERROR: Invalid bbox dimensions: w={w}, h={h}")
                return False
            
            # Validate bbox is within frame bounds
            if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
                print(f"WARNING: bbox partially outside frame bounds!")
                print(f"  Frame: {frame_w}x{frame_h}")
                print(f"  BBox: x={x}, y={y}, w={w}, h={h}")
                print(f"  BBox right edge: {x+w}, bottom edge: {y+h}")
                # Clamp bbox to frame bounds
                x = max(0, min(x, frame_w - 1))
                y = max(0, min(y, frame_h - 1))
                w = min(w, frame_w - x)
                h = min(h, frame_h - y)
                bbox = (x, y, w, h)
                print(f"  Clamped to: {bbox}")
            
            # Create tracker based on type - try legacy API first (more reliable)
            if self.tracker_type == TrackerType.CSRT:
                try:
                    # Try legacy API first (more reliable in opencv-contrib-python)
                    print("Trying cv2.legacy.TrackerCSRT_create()")
                    self.tracker = cv2.legacy.TrackerCSRT_create()
                    print("Successfully created TrackerCSRT (legacy API)")
                except (AttributeError, Exception) as e:
                    print(f"Legacy API failed: {e}")
                    try:
                        # Try new API
                        print("Trying cv2.TrackerCSRT_create()")
                        self.tracker = cv2.TrackerCSRT_create()
                        print("Successfully created TrackerCSRT (new API)")
                    except (AttributeError, Exception) as e2:
                        print(f"New API also failed: {e2}")
                        # Fallback to KCF if CSRT not available
                        print("CSRT tracker not available, using KCF")
                        try:
                            self.tracker = cv2.legacy.TrackerKCF_create()
                        except:
                            self.tracker = cv2.TrackerKCF_create()
            else:
                try:
                    self.tracker = cv2.TrackerKCF_create()
                except AttributeError:
                    self.tracker = cv2.legacy.TrackerKCF_create()
            
            # Validate tracker was created
            if self.tracker is None:
                print("ERROR: Failed to create tracker!")
                return False
            
            print(f"Tracker created successfully, attempting init with bbox={bbox}")
            
            # Initialize tracker
            try:
                success = self.tracker.init(frame, bbox)
                print(f"tracker.init() returned: {success}")
                
                # If init() returned None, it might be an API issue - try creating tracker with legacy API
                if success is None:
                    print("⚠️ tracker.init() returned None! Trying to recreate with legacy API...")
                    try:
                        if self.tracker_type == TrackerType.CSRT:
                            self.tracker = cv2.legacy.TrackerCSRT_create()
                        else:
                            self.tracker = cv2.legacy.TrackerKCF_create()
                        success = self.tracker.init(frame, bbox)
                        print(f"Legacy tracker.init() returned: {success}")
                    except Exception as legacy_error:
                        print(f"Legacy API also failed: {legacy_error}")
                        return False
                
            except Exception as init_error:
                print(f"ERROR during tracker.init(): {init_error}")
                import traceback
                traceback.print_exc()
                return False
            
            # Check if initialization was successful
            if success:
                self.bbox = bbox
                self.is_initialized = True
                self.tracking_lost = False
                self.smoothing_buffer = [bbox]
                print(f"✅ Tracker initialized successfully!")
                return True
            else:
                print(f"❌ tracker.init() returned False or None!")
                return False
        except Exception as e:
            print(f"❌ Exception in init_tracker: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Update tracker with new frame

        Args:
            frame: Current frame (BGR format)

        Returns:
            Updated bounding box (x, y, width, height) or None if tracking lost
        """
        if not self.is_initialized or self.tracker is None:
            return None

        try:
            # Update tracker
            success, bbox = self.tracker.update(frame)

            if success:
                # Convert to tuple of integers
                x, y, w, h = [int(v) for v in bbox]

                # Validate bbox
                if w > 0 and h > 0 and x >= 0 and y >= 0:
                    self.bbox = (x, y, w, h)
                    self.tracking_lost = False

                    # DISABLED: Smoothing is now done in TrackerManager (EMA)
                    # to avoid double smoothing that causes lag
                    # Return raw bbox directly
                    return self.bbox
                else:
                    self.tracking_lost = True
                    return None
            else:
                self.tracking_lost = True
                return None

        except Exception as e:
            print(f"Error updating tracker: {e}")
            self.tracking_lost = True
            return None
    
    def _apply_smoothing(self) -> Tuple[int, int, int, int]:
        """
        Apply smoothing to bounding box using moving average
        
        Returns:
            Smoothed bounding box
        """
        if len(self.smoothing_buffer) < 2:
            return self.bbox
        
        # Calculate average of recent bboxes
        avg_x = int(np.mean([b[0] for b in self.smoothing_buffer]))
        avg_y = int(np.mean([b[1] for b in self.smoothing_buffer]))
        avg_w = int(np.mean([b[2] for b in self.smoothing_buffer]))
        avg_h = int(np.mean([b[3] for b in self.smoothing_buffer]))
        
        return (avg_x, avg_y, avg_w, avg_h)
    
    def get_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Get current bounding box"""
        return self.bbox
    
    def reset(self):
        """Reset tracker state"""
        self.tracker = None
        self.bbox = None
        self.is_initialized = False
        self.tracking_lost = False
        self.smoothing_buffer = []


