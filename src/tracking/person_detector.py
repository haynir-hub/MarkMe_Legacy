"""
Person and Ball Detector - Advanced detection using YOLO
Supports multiple model sizes and multi-scale detection for small objects
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class ModelSize(Enum):
    """YOLO model sizes - larger = more accurate but slower"""
    NANO = "n"      # Fastest, least accurate
    SMALL = "s"     # Fast, decent accuracy
    MEDIUM = "m"    # Balanced
    LARGE = "l"     # Accurate, slower
    XLARGE = "x"    # Most accurate, slowest


class PersonDetector:
    """
    Advanced detector for people and balls using YOLO.
    Supports multiple model sizes and multi-scale detection for small objects.
    """
    
    # COCO class IDs
    CLASS_PERSON = 0
    CLASS_SPORTS_BALL = 32
    
    def __init__(self, model_size: ModelSize = ModelSize.MEDIUM, version: int = 8):
        """
        Initialize detector with specified model size.
        
        Args:
            model_size: Model size (NANO, SMALL, MEDIUM, LARGE, XLARGE)
                       Larger models are better at detecting small objects like balls.
            version: YOLO version (8 or 11). YOLOv11 is newer and may have better accuracy.
        """
        self.model = None
        self.model_size = model_size
        self.version = version
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            import torch
            print(f"✅ PyTorch version: {torch.__version__}")
            
            if torch.cuda.is_available():
                print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                print("ℹ️ Running on CPU (CUDA not available)")
            
        except Exception as torch_error:
            print(f"❌ ERROR: PyTorch not available: {torch_error}")
            self.model = None
            return
        
        try:
            from ultralytics import YOLO
            
            # Build model name based on version and size
            if self.version == 11:
                model_name = f"yolo11{self.model_size.value}.pt"
            else:
                model_name = f"yolov8{self.model_size.value}.pt"
            
            print(f"🔄 Loading {model_name}...")
            self.model = YOLO(model_name)
            print(f"✅ YOLO model loaded: {model_name}")
            print(f"   Model size: {self.model_size.name}")
            print(f"   Better for small objects: {self.model_size in [ModelSize.MEDIUM, ModelSize.LARGE, ModelSize.XLARGE]}")
            
        except Exception as e:
            print(f"❌ ERROR loading YOLO model: {e}")
            try:
                from ultralytics import YOLO
                print("⚠️ Falling back to yolov8n.pt...")
                self.model = YOLO('yolov8n.pt')
                print("✅ Fallback model loaded: yolov8n.pt")
            except:
                self.model = None
    
    def detect_people(self, frame: np.ndarray, confidence_threshold: float = 0.25) -> List[Tuple[int, int, int, int, float]]:
        """Detect people in a frame"""
        return self._detect(frame, [self.CLASS_PERSON], confidence_threshold)
    
    def detect_balls(self, frame: np.ndarray, confidence_threshold: float = 0.08) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect sports balls in a frame using multi-scale detection.
        Uses lower confidence threshold and multi-scale approach for small objects.
        """
        # First try normal detection
        detections = self._detect(frame, [self.CLASS_SPORTS_BALL], confidence_threshold)
        
        # If no balls found, try multi-scale detection
        if not detections:
            print("🔍 No balls found with normal detection, trying multi-scale...")
            detections = self._detect_multiscale(frame, [self.CLASS_SPORTS_BALL], confidence_threshold)
        
        return detections
    
    def detect_balls_aggressive(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Aggressively detect balls using all available techniques.
        Use this when normal detection fails.
        """
        all_detections = []
        
        # 1. Normal detection with very low threshold
        dets = self._detect(frame, [self.CLASS_SPORTS_BALL], confidence_threshold=0.05)
        all_detections.extend(dets)
        
        # 2. Multi-scale detection
        dets = self._detect_multiscale(frame, [self.CLASS_SPORTS_BALL], confidence_threshold=0.05)
        all_detections.extend(dets)
        
        # 3. Tiled detection (SAHI approach)
        dets = self._detect_tiled(frame, [self.CLASS_SPORTS_BALL], confidence_threshold=0.05)
        all_detections.extend(dets)
        
        # 4. Upscaled detection (2x)
        dets = self._detect_upscaled(frame, [self.CLASS_SPORTS_BALL], confidence_threshold=0.05, scale=2.0)
        all_detections.extend(dets)
        
        # Remove duplicates using NMS
        if all_detections:
            all_detections = self._apply_nms(all_detections, iou_threshold=0.4)
        
        return all_detections
    
    def detect_all(self, frame: np.ndarray, 
                   confidence_threshold: float = 0.25,
                   ball_confidence_threshold: float = 0.08,
                   aggressive_ball_detection: bool = False) -> Dict[str, List]:
        """Detect both people and balls in a frame."""
        people = self._detect(frame, [self.CLASS_PERSON], confidence_threshold)
        
        if aggressive_ball_detection:
            balls = self.detect_balls_aggressive(frame)
        else:
            balls = self.detect_balls(frame, ball_confidence_threshold)
        
        return {'people': people, 'balls': balls}
    
    def _detect(self, frame: np.ndarray, classes: List[int], 
                confidence_threshold: float) -> List[Tuple[int, int, int, int, float]]:
        """Internal detection method"""
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
            return []
    
    def _detect_multiscale(self, frame: np.ndarray, classes: List[int],
                          confidence_threshold: float) -> List[Tuple[int, int, int, int, float]]:
        """Multi-scale detection for small objects."""
        if self.model is None:
            return []
        
        all_detections = []
        h, w = frame.shape[:2]
        scales = [1.5, 2.0]
        
        for scale in scales:
            try:
                new_w = int(w * scale)
                new_h = int(h * scale)
                scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                dets = self._detect(scaled_frame, classes, confidence_threshold)
                
                for x, y, bw, bh, conf in dets:
                    orig_x = int(x / scale)
                    orig_y = int(y / scale)
                    orig_w = int(bw / scale)
                    orig_h = int(bh / scale)
                    all_detections.append((orig_x, orig_y, orig_w, orig_h, conf))
                    
            except Exception as e:
                print(f"⚠️ Multi-scale detection error at scale {scale}: {e}")
        
        if all_detections:
            all_detections = self._apply_nms(all_detections, iou_threshold=0.4)
        
        return all_detections
    
    def _detect_tiled(self, frame: np.ndarray, classes: List[int],
                      confidence_threshold: float, 
                      tile_size: int = 640,
                      overlap: float = 0.25) -> List[Tuple[int, int, int, int, float]]:
        """Tiled detection (SAHI-like approach) for small objects."""
        if self.model is None:
            return []
        
        all_detections = []
        h, w = frame.shape[:2]
        stride = int(tile_size * (1 - overlap))
        
        for y_start in range(0, h, stride):
            for x_start in range(0, w, stride):
                x_end = min(x_start + tile_size, w)
                y_end = min(y_start + tile_size, h)
                
                if x_end - x_start < 100 or y_end - y_start < 100:
                    continue
                
                tile = frame[y_start:y_end, x_start:x_end]
                tile_resized = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_CUBIC)
                
                try:
                    dets = self._detect(tile_resized, classes, confidence_threshold)
                    
                    scale_x = (x_end - x_start) / tile_size
                    scale_y = (y_end - y_start) / tile_size
                    
                    for x, y, bw, bh, conf in dets:
                        orig_x = int(x * scale_x) + x_start
                        orig_y = int(y * scale_y) + y_start
                        orig_w = int(bw * scale_x)
                        orig_h = int(bh * scale_y)
                        all_detections.append((orig_x, orig_y, orig_w, orig_h, conf))
                        
                except Exception as e:
                    pass
        
        if all_detections:
            all_detections = self._apply_nms(all_detections, iou_threshold=0.4)
        
        return all_detections
    
    def _detect_upscaled(self, frame: np.ndarray, classes: List[int],
                         confidence_threshold: float,
                         scale: float = 2.0) -> List[Tuple[int, int, int, int, float]]:
        """Simple upscaled detection"""
        if self.model is None:
            return []
        
        try:
            h, w = frame.shape[:2]
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            upscaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            dets = self._detect(upscaled, classes, confidence_threshold)
            
            scaled_dets = []
            for x, y, bw, bh, conf in dets:
                scaled_dets.append((
                    int(x / scale),
                    int(y / scale),
                    int(bw / scale),
                    int(bh / scale),
                    conf
                ))
            
            return scaled_dets
            
        except Exception as e:
            print(f"⚠️ Upscaled detection error: {e}")
            return []
    
    def _apply_nms(self, detections: List[Tuple[int, int, int, int, float]], 
                   iou_threshold: float = 0.4) -> List[Tuple[int, int, int, int, float]]:
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if not detections:
            return []
        
        boxes = []
        scores = []
        for x, y, w, h, conf in detections:
            boxes.append([x, y, x + w, y + h])
            scores.append(conf)
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=0.0,
            nms_threshold=iou_threshold
        )
        
        if len(indices) > 0:
            if isinstance(indices[0], (list, np.ndarray)):
                indices = [i[0] for i in indices]
            
            result = []
            for i in indices:
                x1, y1, x2, y2 = boxes[i]
                result.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(scores[i])))
            return result
        
        return []
    
    def is_available(self) -> bool:
        """Check if detector is available"""
        return self.model is not None
    
    def get_model_info(self) -> str:
        """Get information about current model"""
        if self.model is None:
            return "No model loaded"
        return f"YOLOv{self.version}{self.model_size.value} ({self.model_size.name})"
    
    def upgrade_model(self, new_size: ModelSize) -> bool:
        """Upgrade to a larger model for better accuracy."""
        old_size = self.model_size
        self.model_size = new_size
        self._load_model()
        
        if self.model is None:
            self.model_size = old_size
            self._load_model()
            return False
        
        return True


def create_ball_detector() -> PersonDetector:
    """Create a detector optimized for ball detection (MEDIUM model)."""
    return PersonDetector(model_size=ModelSize.MEDIUM)


def create_accurate_detector() -> PersonDetector:
    """Create the most accurate detector (XLARGE model - slower)."""
    return PersonDetector(model_size=ModelSize.XLARGE)
