"""
Player Tracker - Handles tracking of individual players using OpenCV trackers
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
from enum import Enum


class TrackerType(Enum):
    """Available tracker types"""
    CSRT = "csrt"      # Legacy - may not be available
    KCF = "kcf"        # Legacy - may not be available
    MIL = "mil"        # Modern - always available
    NANO = "nano"      # Modern - lightweight DL-based


def _create_tracker():
    """
    Create the best available tracker for the current OpenCV version.
    
    OpenCV 4.5.1+ removed CSRT/KCF from main package.
    We try trackers in order of preference:
    1. TrackerMIL - simple, fast, always available in modern OpenCV
    2. TrackerCSRT (legacy) - if opencv-contrib-python is installed
    3. TrackerKCF (legacy) - fallback
    
    Returns:
        Tracker instance or None if all fail
    """
    tracker = None
    tracker_name = None
    
    # Try TrackerMIL first (available in OpenCV 4.5.1+)
    try:
        tracker = cv2.TrackerMIL_create()
        tracker_name = "TrackerMIL"
        print(f"✅ Created {tracker_name} (modern API)")
        return tracker, tracker_name
    except AttributeError:
        pass
    
    # Try legacy CSRT (requires opencv-contrib-python)
    try:
        tracker = cv2.legacy.TrackerCSRT_create()
        tracker_name = "TrackerCSRT (legacy)"
        print(f"✅ Created {tracker_name}")
        return tracker, tracker_name
    except AttributeError:
        pass
    
    # Try new CSRT API
    try:
        tracker = cv2.TrackerCSRT_create()
        tracker_name = "TrackerCSRT"
        print(f"✅ Created {tracker_name}")
        return tracker, tracker_name
    except AttributeError:
        pass
    
    # Try legacy KCF
    try:
        tracker = cv2.legacy.TrackerKCF_create()
        tracker_name = "TrackerKCF (legacy)"
        print(f"✅ Created {tracker_name}")
        return tracker, tracker_name
    except AttributeError:
        pass
    
    # Try new KCF API
    try:
        tracker = cv2.TrackerKCF_create()
        tracker_name = "TrackerKCF"
        print(f"✅ Created {tracker_name}")
        return tracker, tracker_name
    except AttributeError:
        pass
    
    print("❌ No compatible tracker found!")
    return None, None


class PlayerTracker:
    """Tracks a single player through video frames"""
    
    def __init__(self, tracker_type: TrackerType = TrackerType.MIL):
        """
        Initialize player tracker
        
        Args:
            tracker_type: Type of tracker to use (MIL recommended for modern OpenCV)
        """
        self.tracker_type = tracker_type
        self.tracker = None
        self.tracker_name = None
        self.bbox = None
        self.is_initialized = False
        self.tracking_lost = False
        self.smoothing_buffer = []
        self.buffer_size = 10
        
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
                # Clamp bbox to frame bounds
                x = max(0, min(x, frame_w - 1))
                y = max(0, min(y, frame_h - 1))
                w = min(w, frame_w - x)
                h = min(h, frame_h - y)
                bbox = (x, y, w, h)
                print(f"  Clamped to: {bbox}")
            
            # Create tracker using helper function
            self.tracker, self.tracker_name = _create_tracker()
            
            if self.tracker is None:
                print("❌ ERROR: Failed to create any tracker!")
                print("   Please ensure OpenCV is properly installed:")
                print("   pip install opencv-python")
                return False
            
            print(f"Attempting to initialize {self.tracker_name} with bbox={bbox}")
            
            # Initialize tracker
            try:
                result = self.tracker.init(frame, bbox)
                print(f"tracker.init() returned: {result}")
                
                # In OpenCV 4.5.1+, init() returns None but still works!
                # We consider initialization successful if:
                # 1. init() returns True (older API)
                # 2. init() returns None (newer API - tracker is still initialized)
                # Only fail if init() explicitly returns False
                if result is False:
                    print(f"❌ tracker.init() explicitly returned False!")
                    return False
                
            except Exception as init_error:
                print(f"ERROR during tracker.init(): {init_error}")
                import traceback
                traceback.print_exc()
                return False
            
            # Success - tracker is initialized
            self.bbox = bbox
            self.is_initialized = True
            self.tracking_lost = False
            self.smoothing_buffer = [bbox]
            print(f"✅ Tracker initialized successfully with {self.tracker_name}!")
            return True
                
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


