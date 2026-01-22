"""
Person and Ball Detector - Automatic detection using YOLO
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')  # Suppress YOLO warnings


class PersonDetector:
    """Detects people and balls in video frames using YOLO"""
    
    # COCO class IDs
    CLASS_PERSON = 0
    CLASS_SPORTS_BALL = 32
    
    def __init__(self, use_small_model: bool = False):
        """
        Initialize detector.
        
        Args:
            use_small_model: If True, use YOLOv8s (small) instead of YOLOv8n (nano).
                           YOLOv8s is slower but better at detecting small objects like balls.
        """
        self.model = None
        self.use_small_model = use_small_model
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model for detection"""
        try:
            # Try to import torch first to catch DLL errors early
            try:
                import torch
                print(f"✅ PyTorch version: {torch.__version__}")
            except Exception as torch_error:
                print(f"❌ ERROR: PyTorch not available: {torch_error}")
                print("   This is usually a DLL loading issue on Windows.")
                print("   Try: pip uninstall torch torchvision -y && pip install torch torchvision")
                self.model = None
                return
            
            from ultralytics import YOLO
            
            # Choose model based on requirements
            # yolov8n.pt - fastest, less accurate for small objects
            # yolov8s.pt - slower but better for small objects like balls
            model_name = 'yolov8s.pt' if self.use_small_model else 'yolov8n.pt'
            
            self.model = YOLO(model_name)
            print(f"✅ YOLO model loaded successfully ({model_name})")
            
        except ImportError:
            print("❌ ERROR: ultralytics not installed. Please run: pip install ultralytics")
            self.model = None
        except Exception as e:
            print(f"❌ ERROR loading YOLO model: {e}")
            print("   This might be a PyTorch DLL issue on Windows.")
            print("   Try reinstalling PyTorch: pip uninstall torch torchvision -y && pip install torch torchvision")
            self.model = None
    
    def detect_people(self, frame: np.ndarray, confidence_threshold: float = 0.25) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect people in a frame
        
        Args:
            frame: Frame as numpy array (BGR format)
            confidence_threshold: Minimum confidence for detection (0.0-1.0)
            
        Returns:
            List of (x, y, w, h, confidence) tuples for each detected person
            Returns empty list if model not loaded or no detections
        """
        return self._detect(frame, [self.CLASS_PERSON], confidence_threshold)
    
    def detect_balls(self, frame: np.ndarray, confidence_threshold: float = 0.15) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect sports balls in a frame
        
        Args:
            frame: Frame as numpy array (BGR format)
            confidence_threshold: Minimum confidence for detection (0.0-1.0)
                                Lower default for balls since they're harder to detect
            
        Returns:
            List of (x, y, w, h, confidence) tuples for each detected ball
            Returns empty list if model not loaded or no detections
        """
        return self._detect(frame, [self.CLASS_SPORTS_BALL], confidence_threshold)
    
    def detect_all(self, frame: np.ndarray, 
                   confidence_threshold: float = 0.25,
                   ball_confidence_threshold: float = 0.15) -> dict:
        """
        Detect both people and balls in a frame
        
        Args:
            frame: Frame as numpy array (BGR format)
            confidence_threshold: Minimum confidence for people detection
            ball_confidence_threshold: Minimum confidence for ball detection (lower for small objects)
            
        Returns:
            Dictionary with 'people' and 'balls' lists of (x, y, w, h, confidence) tuples
        """
        if self.model is None:
            return {'people': [], 'balls': []}
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect both classes in single pass for efficiency
            results = self.model(frame_rgb, 
                               classes=[self.CLASS_PERSON, self.CLASS_SPORTS_BALL], 
                               conf=min(confidence_threshold, ball_confidence_threshold),
                               verbose=False)
            
            people = []
            balls = []
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        x1, y1, x2, y2 = box.astype(int)
                        w = x2 - x1
                        h = y2 - y1
                        detection = (x1, y1, w, h, float(conf))
                        
                        if int(cls_id) == self.CLASS_PERSON and conf >= confidence_threshold:
                            people.append(detection)
                        elif int(cls_id) == self.CLASS_SPORTS_BALL and conf >= ball_confidence_threshold:
                            balls.append(detection)
            
            return {'people': people, 'balls': balls}
            
        except Exception as e:
            print(f"❌ ERROR in detection: {e}")
            import traceback
            traceback.print_exc()
            return {'people': [], 'balls': []}
    
    def _detect(self, frame: np.ndarray, classes: List[int], 
                confidence_threshold: float) -> List[Tuple[int, int, int, int, float]]:
        """
        Internal detection method
        
        Args:
            frame: Frame as numpy array (BGR format)
            classes: List of COCO class IDs to detect
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of (x, y, w, h, confidence) tuples
        """
        if self.model is None:
            return []
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.model(frame_rgb, classes=classes, conf=confidence_threshold, verbose=False)
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = box.astype(int)
                        w = x2 - x1
                        h = y2 - y1
                        detections.append((x1, y1, w, h, float(conf)))
            
            return detections
            
        except Exception as e:
            print(f"❌ ERROR in detection: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def is_available(self) -> bool:
        """Check if detector is available"""
        return self.model is not None
    
    def upgrade_to_small_model(self) -> bool:
        """
        Upgrade to YOLOv8s model for better small object detection.
        Returns True if upgrade successful.
        """
        if self.use_small_model:
            return True  # Already using small model
        
        self.use_small_model = True
        self._load_model()
        return self.model is not None
