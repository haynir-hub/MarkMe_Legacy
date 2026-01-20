"""
Video Canvas - Widget for displaying video and drawing bounding boxes
"""
from PyQt6.QtWidgets import QWidget, QLabel, QApplication
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
import cv2
import numpy as np
from typing import Optional, Tuple, List


class VideoCanvas(QLabel):
    """Canvas for displaying video frames and drawing bounding boxes"""
    
    # Signals
    bbox_selected = pyqtSignal(int, int, int, int)  # x, y, width, height
    frame_clicked = pyqtSignal(int, int)  # x, y
    person_clicked = pyqtSignal(int, int, int, int)  # x, y, w, h - when clicking on detected person
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #555;")
        
        # Drawing state
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.current_bbox = None
        self.displayed_bboxes = []  # List of (x, y, w, h, name, style, color)
        
        # Detection state
        self.detection_mode = False  # True when showing detected people
        self.detected_people = []  # List of (x, y, w, h, confidence) for detected people
        self.detected_bboxes_display = []  # List of (x, y, w, h) for display
        
        # Video frame
        self.current_frame = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
    
    def set_frame(self, frame: np.ndarray):
        """
        Set frame to display
        
        Args:
            frame: Frame as numpy array (BGR format)
        """
        if frame is None:
            print("VideoCanvas.set_frame: frame is None!")
            return
        
        # Check first pixel to verify frame is different
        first_pixel = tuple(frame[0, 0, :])
        old_first_pixel = tuple(self.current_frame[0, 0, :]) if self.current_frame is not None else None
        print(f"VideoCanvas: First pixel={first_pixel}, old={old_first_pixel}, different={first_pixel != old_first_pixel}")
        
        self.current_frame = frame.copy()
        self._update_display()
    
    def _update_display(self):
        """Update the displayed image"""
        if self.current_frame is None:
            return
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # Create QImage with copied data
        qt_image = QImage(rgb_frame.data.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        
        if qt_image.isNull():
            print("ERROR: QImage is null!")
            return
        
        # Scale to fit widget while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        widget_size = self.size()
        
        # Calculate scaling to fit widget
        scale_w = widget_size.width() / pixmap.width() if pixmap.width() > 0 else 1.0
        scale_h = widget_size.height() / pixmap.height() if pixmap.height() > 0 else 1.0
        scale = min(scale_w, scale_h)  # Use smaller scale to maintain aspect ratio
        
        scaled_pixmap = pixmap.scaled(
            int(pixmap.width() * scale),
            int(pixmap.height() * scale),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Make sure pixmap is not shared (detach it)
        scaled_pixmap = scaled_pixmap.copy()
        
        # Calculate scale factor and offset - CRITICAL for mouse coordinate conversion
        if pixmap.width() > 0 and pixmap.height() > 0:
            self.scale_factor = scaled_pixmap.width() / pixmap.width()
        else:
            self.scale_factor = 1.0
        
        # Calculate offset to center the scaled pixmap in the widget
        self.offset_x = (widget_size.width() - scaled_pixmap.width()) // 2
        self.offset_y = (widget_size.height() - scaled_pixmap.height()) // 2
        
        # Create painter to draw overlays
        painter = QPainter(scaled_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw detected people (in detection mode)
        if self.detection_mode and self.detected_bboxes_display:
            for i, (x, y, w, h) in enumerate(self.detected_bboxes_display):
                # Find confidence for this bbox
                conf = 0.0
                if i < len(self.detected_people):
                    _, _, _, _, conf = self.detected_people[i]
                
                # Draw with green color to indicate clickable detection
                pen = QPen(QColor(0, 255, 0))  # Green
                pen.setWidth(3)
                painter.setPen(pen)
                
                # Scale coordinates
                sx = int(x * self.scale_factor)
                sy = int(y * self.scale_factor)
                sw = int(w * self.scale_factor)
                sh = int(h * self.scale_factor)
                
                painter.drawRect(sx, sy, sw, sh)
                
                # Draw confidence label
                conf_text = f"Person {i+1} ({conf:.0%})"
                painter.setPen(QPen(QColor(0, 255, 0)))
                painter.setFont(painter.font())
                text_rect = painter.fontMetrics().boundingRect(conf_text)
                text_x = sx + (sw - text_rect.width()) // 2
                text_y = sy - 10
                if text_y < 0:
                    text_y = sy + sh + 20
                
                # Draw background for text
                bg_rect = QRect(text_x - 2, text_y - text_rect.height() - 2, 
                               text_rect.width() + 4, text_rect.height() + 4)
                painter.fillRect(bg_rect, QColor(0, 0, 0, 180))
                painter.drawText(text_x, text_y - 2, conf_text)
        
        # Draw existing bboxes
        for bbox_data in self.displayed_bboxes:
            x, y, w, h, name, style, color = bbox_data
            self._draw_bbox_on_pixmap(painter, x, y, w, h, name, style, color)
        
        # Draw current bbox being drawn
        if self.current_bbox:
            x, y, w, h = self.current_bbox
            self._draw_bbox_on_pixmap(painter, x, y, w, h, "", "rectangle", (255, 255, 255))
            print(f"Drawing current bbox: ({x}, {y}, {w}, {h})")
        
        painter.end()
        
        # Set the new pixmap
        self.setPixmap(scaled_pixmap)
        
        # Force updates
        self.update()
        self.repaint()
        QApplication.processEvents()
    
    def _draw_bbox_on_pixmap(self, painter: QPainter, x: int, y: int, 
                             w: int, h: int, name: str, style: str, color: Tuple[int, int, int]):
        """Draw bounding box on pixmap"""
        if self.scale_factor <= 0:
            return
        # Scale coordinates (don't add offset - we're drawing on the scaled pixmap, not the widget)
        sx = int(x * self.scale_factor)
        sy = int(y * self.scale_factor)
        sw = int(w * self.scale_factor)
        sh = int(h * self.scale_factor)
        
        # Set pen color
        pen = QPen(QColor(*color))
        pen.setWidth(2)
        painter.setPen(pen)
        
        if style == 'rectangle':
            painter.drawRect(sx, sy, sw, sh)
        elif style == 'circle':
            # Draw 3D floor hoop - ellipse with player in center
            center_x = sx + sw // 2
            
            # Calculate ellipse radii (matching reference image)
            radius_x = max(int(sw * 1.2), 40)  # Larger horizontal radius for 3D effect
            radius_y = max(int(sw * 0.35), 18)  # Vertical radius (compressed for perspective)
            
            # Center Y: Place at player's CENTER (torso area) for 3D effect
            # This makes the player appear IN THE MIDDLE of the floor hoop
            # Back part of hoop will be hidden behind player (3D layering)
            center_y = sy + int(sh * 0.5)  # Center of player body
            
            # Draw ellipse: QRect(left, top, width, height)
            # left = center_x - radius_x, top = center_y - radius_y
            painter.drawEllipse(center_x - radius_x, center_y - radius_y, radius_x * 2, radius_y * 2)
        elif style == 'arrow':
            # Draw arrow above
            arrow_y = max(0, sy - 30)
            arrow_x = sx + sw // 2
            points = [
                QPoint(arrow_x, arrow_y),
                QPoint(arrow_x - 15, arrow_y + 30),
                QPoint(arrow_x + 15, arrow_y + 30)
            ]
            painter.drawPolygon(points)
        
        # Draw name if provided - position it above the bbox, centered
        if name:
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.setFont(painter.font())
            # Get text metrics
            text_rect = painter.fontMetrics().boundingRect(name)
            # Center the text horizontally over the bbox
            text_x = sx + (sw - text_rect.width()) // 2
            text_y = sy - 10  # Position above the bbox
            # Draw background rectangle for text readability
            from PyQt6.QtCore import QRect
            bg_rect = QRect(text_x - 4, text_y - text_rect.height() - 2, 
                           text_rect.width() + 8, text_rect.height() + 4)
            painter.fillRect(bg_rect, QColor(0, 0, 0, 180))
            painter.drawText(text_x, text_y - 2, name)
    
    def add_bbox(self, x: int, y: int, w: int, h: int, 
                name: str = "", style: str = "rectangle", 
                color: Tuple[int, int, int] = (255, 255, 255)):
        """
        Add a bounding box to display
        
        Args:
            x, y, w, h: Bounding box coordinates
            name: Player name
            style: Marker style
            color: RGB color tuple
        """
        self.displayed_bboxes.append((x, y, w, h, name, style, color))
        self._update_display()
    
    def clear_bboxes(self):
        """Clear all displayed bounding boxes"""
        self.displayed_bboxes.clear()
        self.current_bbox = None
        self._update_display()
    
    def remove_bbox(self, index: int):
        """Remove a bounding box by index"""
        if 0 <= index < len(self.displayed_bboxes):
            self.displayed_bboxes.pop(index)
            self._update_display()
    
    def _get_image_coords_from_mouse(self, mouse_x: float, mouse_y: float) -> tuple:
        """Convert mouse coordinates to image coordinates"""
        if self.current_frame is None or self.scale_factor <= 0:
            return None, None
        
        # Get the pixmap currently displayed
        pixmap = self.pixmap()
        if pixmap is None:
            return None, None
        
        # Calculate position relative to scaled pixmap
        rel_x = mouse_x - self.offset_x
        rel_y = mouse_y - self.offset_y
        
        # Check if mouse is within the pixmap bounds
        if rel_x < 0 or rel_y < 0 or rel_x > pixmap.width() or rel_y > pixmap.height():
            return None, None
        
        # Convert to original image coordinates
        x = int(rel_x / self.scale_factor)
        y = int(rel_y / self.scale_factor)
        
        # Clamp to frame bounds
        h, w = self.current_frame.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        return x, y
    
    def mousePressEvent(self, event):
        """Handle mouse press for drawing bounding box or clicking on detected person"""
        print(f"MousePress: button={event.button()}, current_frame is None={self.current_frame is None}, detection_mode={self.detection_mode}")
        if event.button() == Qt.MouseButton.LeftButton and self.current_frame is not None:
            # Get mouse position
            mouse_x = event.position().x()
            mouse_y = event.position().y()
            print(f"MousePress: pos=({mouse_x}, {mouse_y})")
            
            # Convert to image coordinates
            x, y = self._get_image_coords_from_mouse(mouse_x, mouse_y)
            print(f"MousePress: image coords=({x}, {y})")
            if x is not None and y is not None:
                # If in detection mode, check if clicking on a detected person
                if self.detection_mode and self.detected_bboxes_display:
                    clicked_bbox = self._find_clicked_bbox(x, y)
                    if clicked_bbox:
                        print(f"MousePress: Clicked on detected person: {clicked_bbox}")
                        self.person_clicked.emit(*clicked_bbox)
                        return
                
                # Otherwise, start drawing bbox
                self.drawing = True
                self.start_point = QPoint(x, y)
                self.end_point = QPoint(x, y)
                self.current_bbox = None
                print(f"MousePress: Drawing started at ({x}, {y})")
    
    def _find_clicked_bbox(self, x: int, y: int) -> Optional[Tuple[int, int, int, int]]:
        """Find which detected bbox was clicked"""
        for bbox in self.detected_bboxes_display:
            bx, by, bw, bh = bbox
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return bbox
        return None
    
    def set_detected_people(self, detections: List[Tuple[int, int, int, int, float]]):
        """Set detected people to display"""
        self.detected_people = detections
        self.detected_bboxes_display = [(x, y, w, h) for x, y, w, h, conf in detections]
        self._update_display()
    
    def enable_detection_mode(self, enable: bool = True):
        """Enable or disable detection mode"""
        self.detection_mode = enable
        if not enable:
            self.detected_people = []
            self.detected_bboxes_display = []
        self._update_display()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for drawing bounding box"""
        if self.drawing and self.current_frame is not None:
            # Get mouse position
            mouse_x = event.position().x()
            mouse_y = event.position().y()
            
            # Convert to image coordinates
            x, y = self._get_image_coords_from_mouse(mouse_x, mouse_y)
            if x is not None and y is not None:
                self.end_point = QPoint(x, y)
                
                # Calculate bbox
                x1 = min(self.start_point.x(), self.end_point.x())
                y1 = min(self.start_point.y(), self.end_point.y())
                x2 = max(self.start_point.x(), self.end_point.x())
                y2 = max(self.start_point.y(), self.end_point.y())
                
                w = x2 - x1
                h = y2 - y1
                
                print(f"MouseMove: bbox=({x1}, {y1}, {w}, {h})")
                if w > 10 and h > 10:  # Minimum size
                    self.current_bbox = (x1, y1, w, h)
                    self._update_display()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release to finalize bounding box"""
        print(f"MouseRelease: button={event.button()}, drawing={self.drawing}")
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            
            if self.current_bbox:
                x, y, w, h = self.current_bbox
                print(f"MouseRelease: bbox=({x}, {y}, {w}, {h})")
                if w > 10 and h > 10:  # Minimum size
                    print(f"MouseRelease: Emitting bbox_selected signal")
                    self.bbox_selected.emit(x, y, w, h)
                else:
                    print(f"MouseRelease: Bbox too small ({w}x{h})")
            else:
                print("MouseRelease: No current_bbox")
            
            self.current_bbox = None
            self._update_display()
    
    def resizeEvent(self, event):
        """Handle widget resize"""
        super().resizeEvent(event)
        self._update_display()

