"""
Person Detector - Automatic person detection using YOLO
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')  # Suppress YOLO warnings


class PersonDetector:
    """Detects people in video frames using YOLO"""
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model for person detection"""
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
            # Use YOLOv8n (nano) for speed, or YOLOv8s for better accuracy
            # Model will be downloaded automatically on first use
            self.model = YOLO('yolov8n.pt')  # nano version - fastest
            print("✅ YOLO model loaded successfully")
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
        if self.model is None:
            return []
        
        try:
            # YOLO expects RGB, but OpenCV uses BGR
            # Actually, YOLO can handle BGR directly, but let's be safe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run detection - only detect 'person' class (class 0)
            results = self.model(frame_rgb, classes=[0], conf=confidence_threshold, verbose=False)
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = box.astype(int)
                        w = x2 - x1
                        h = y2 - y1
                        
                        # Convert to (x, y, w, h, confidence) format
                        detections.append((x1, y1, w, h, float(conf)))
            
            return detections
            
        except Exception as e:
            print(f"❌ ERROR in person detection: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def is_available(self) -> bool:
        """Check if detector is available"""
        return self.model is not None

