"""
Video Canvas - Widget for displaying video and drawing bounding boxes
Supports zoom with Cmd/Ctrl+Scroll for enhanced detection
"""
from PyQt6.QtWidgets import QWidget, QLabel, QApplication
from PyQt6.QtCore import Qt, QRect, QPoint, QPointF, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QWheelEvent
import cv2
import numpy as np
from typing import Optional, Tuple, List


class VideoCanvas(QLabel):
    """Canvas for displaying video frames and drawing bounding boxes
    
    Zoom Controls:
    - Mac: Cmd + Scroll (or pinch gesture)
    - PC: Ctrl + Scroll
    - Double-click: Reset zoom to 100%
    - Pan: Click and drag while zoomed (or Shift + drag)
    """
    
    # Signals
    bbox_selected = pyqtSignal(int, int, int, int)  # x, y, width, height
    frame_clicked = pyqtSignal(int, int)  # x, y
    person_clicked = pyqtSignal(int, int, int, int)  # x, y, w, h - when clicking on detected person
    zoom_changed = pyqtSignal(float, tuple)  # zoom_level, visible_region (x, y, w, h)
    zoom_detection_requested = pyqtSignal(np.ndarray, tuple)  # cropped_frame, original_region
    radar_direction_set = pyqtSignal(float)  # angle in radians
    
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
        
        # ===== ZOOM STATE =====
        self.zoom_level = 1.0  # 1.0 = 100%, 2.0 = 200%, etc.
        self.min_zoom = 1.0
        self.max_zoom = 8.0
        self.zoom_step = 0.15  # How much to zoom per scroll tick
        
        # Pan offset (in original frame coordinates)
        self.pan_x = 0.0  # Center X of visible region
        self.pan_y = 0.0  # Center Y of visible region
        
        # Panning state
        self.panning = False
        self.pan_start_pos = QPoint()
        self.pan_start_offset = (0.0, 0.0)
        
        # Enable mouse tracking for smooth panning
        self.setMouseTracking(True)

        # Show zoom indicator
        self.show_zoom_indicator = True

        # Radar editing mode
        self.radar_edit_mode = False
        self.radar_player_bbox = None  # (x, y, w, h) of player being edited
        self.radar_preview_angle = None  # Current preview angle
        self.radar_mouse_pos = None  # Current mouse position for preview
        self.radar_preview_size = 1.0  # Size multiplier (scroll to change)
    
    def set_frame(self, frame: np.ndarray, reset_zoom: bool = False):
        """
        Set frame to display
        
        Args:
            frame: Frame as numpy array (BGR format)
            reset_zoom: If True, reset zoom to 100%
        """
        if frame is None:
            print("VideoCanvas.set_frame: frame is None!")
            return
        
        self.current_frame = frame.copy()
        
        # Initialize pan center if not set
        if self.pan_x == 0.0 and self.pan_y == 0.0:
            h, w = frame.shape[:2]
            self.pan_x = w / 2
            self.pan_y = h / 2
        
        if reset_zoom:
            self.reset_zoom()
        else:
            self._update_display()
    
    def _update_display(self):
        """Update the displayed image with zoom and pan support"""
        if self.current_frame is None:
            return
        
        frame_h, frame_w = self.current_frame.shape[:2]
        widget_size = self.size()
        
        # Calculate visible region in original frame coordinates
        visible_region = self._get_visible_region()
        vx, vy, vw, vh = visible_region
        
        # Crop frame to visible region
        x1 = max(0, int(vx))
        y1 = max(0, int(vy))
        x2 = min(frame_w, int(vx + vw))
        y2 = min(frame_h, int(vy + vh))
        
        cropped_frame = self.current_frame[y1:y2, x1:x2]
        
        if cropped_frame.size == 0:
            return
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        ch_h, ch_w, ch = rgb_frame.shape
        bytes_per_line = ch * ch_w
        
        # Create QImage
        qt_image = QImage(rgb_frame.data.tobytes(), ch_w, ch_h, bytes_per_line, 
                         QImage.Format.Format_RGB888).copy()
        
        if qt_image.isNull():
            return
        
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale to fit widget
        scale_w = widget_size.width() / pixmap.width() if pixmap.width() > 0 else 1.0
        scale_h = widget_size.height() / pixmap.height() if pixmap.height() > 0 else 1.0
        display_scale = min(scale_w, scale_h)
        
        scaled_pixmap = pixmap.scaled(
            int(pixmap.width() * display_scale),
            int(pixmap.height() * display_scale),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ).copy()
        
        # Store scale factor for coordinate conversion
        # This maps from VISIBLE region to display pixels
        self.scale_factor = display_scale
        self._visible_region = visible_region  # Store for coordinate mapping
        
        # Calculate offset to center
        self.offset_x = (widget_size.width() - scaled_pixmap.width()) // 2
        self.offset_y = (widget_size.height() - scaled_pixmap.height()) // 2
        
        # Create painter for overlays
        painter = QPainter(scaled_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw detected people (adjusted for zoom)
        if self.detection_mode and self.detected_bboxes_display:
            for i, (bx, by, bw, bh) in enumerate(self.detected_bboxes_display):
                # Skip if outside visible region
                if bx + bw < vx or bx > vx + vw or by + bh < vy or by > vy + vh:
                    continue
                
                conf = self.detected_people[i][4] if i < len(self.detected_people) else 0.0
                
                pen = QPen(QColor(0, 255, 0))
                pen.setWidth(3)
                painter.setPen(pen)
                
                # Convert to display coordinates
                sx, sy, sw, sh = self._frame_to_display_coords(bx, by, bw, bh)
                painter.drawRect(int(sx), int(sy), int(sw), int(sh))
                
                # Draw label
                conf_text = f"Person {i+1} ({conf:.0%})"
                painter.setPen(QPen(QColor(0, 255, 0)))
                text_rect = painter.fontMetrics().boundingRect(conf_text)
                text_x = int(sx + (sw - text_rect.width()) // 2)
                text_y = int(sy - 10) if sy > 30 else int(sy + sh + 20)
                
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
        
        # Draw zoom indicator
        if self.show_zoom_indicator and self.zoom_level > 1.0:
            self._draw_zoom_indicator(painter, scaled_pixmap.width(), scaled_pixmap.height())

        # Draw radar preview if in radar edit mode
        if self.radar_edit_mode and self.radar_player_bbox and self.radar_mouse_pos:
            self._draw_radar_preview(painter)

        painter.end()
        
        self.setPixmap(scaled_pixmap)
        self.update()
    
    def _get_visible_region(self) -> Tuple[float, float, float, float]:
        """Get the visible region in original frame coordinates"""
        if self.current_frame is None:
            return (0, 0, 100, 100)
        
        frame_h, frame_w = self.current_frame.shape[:2]
        
        # Calculate visible size based on zoom
        visible_w = frame_w / self.zoom_level
        visible_h = frame_h / self.zoom_level
        
        # Calculate top-left corner based on pan center
        vx = self.pan_x - visible_w / 2
        vy = self.pan_y - visible_h / 2
        
        # Clamp to frame bounds
        vx = max(0, min(vx, frame_w - visible_w))
        vy = max(0, min(vy, frame_h - visible_h))
        
        return (vx, vy, visible_w, visible_h)
    
    def _frame_to_display_coords(self, x: float, y: float, w: float = 0, h: float = 0) -> Tuple[float, float, float, float]:
        """Convert original frame coordinates to display coordinates"""
        vx, vy, vw, vh = self._visible_region if hasattr(self, '_visible_region') else (0, 0, 1, 1)
        
        # Offset by visible region
        dx = (x - vx) * self.scale_factor
        dy = (y - vy) * self.scale_factor
        dw = w * self.scale_factor
        dh = h * self.scale_factor
        
        return (dx, dy, dw, dh)
    
    def _draw_zoom_indicator(self, painter: QPainter, width: int, height: int):
        """Draw zoom level indicator and minimap"""
        # Zoom percentage text
        zoom_text = f"🔍 {self.zoom_level:.0%}"
        font = QFont("Arial", 12, QFont.Weight.Bold)
        painter.setFont(font)
        
        text_rect = painter.fontMetrics().boundingRect(zoom_text)
        margin = 10
        
        # Background
        bg_rect = QRect(margin - 4, margin - 2, text_rect.width() + 12, text_rect.height() + 8)
        painter.fillRect(bg_rect, QColor(0, 0, 0, 180))
        painter.setPen(QPen(QColor(100, 200, 255)))
        painter.drawRect(bg_rect)
        
        # Text
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(margin, margin + text_rect.height(), zoom_text)
        
        # Draw minimap in bottom-right corner
        if self.current_frame is not None:
            mm_size = 80
            mm_margin = 10
            mm_x = width - mm_size - mm_margin
            mm_y = height - mm_size - mm_margin
            
            # Minimap background
            painter.fillRect(mm_x - 2, mm_y - 2, mm_size + 4, mm_size + 4, QColor(0, 0, 0, 150))
            
            # Create tiny thumbnail
            frame_h, frame_w = self.current_frame.shape[:2]
            aspect = frame_w / frame_h
            if aspect > 1:
                mm_w, mm_h = mm_size, int(mm_size / aspect)
            else:
                mm_w, mm_h = int(mm_size * aspect), mm_size
            
            # Draw visible region rectangle
            vx, vy, vw, vh = self._get_visible_region()
            rx = int(mm_x + (vx / frame_w) * mm_w)
            ry = int(mm_y + (vy / frame_h) * mm_h)
            rw = max(4, int((vw / frame_w) * mm_w))
            rh = max(4, int((vh / frame_h) * mm_h))
            
            # Outer frame
            painter.setPen(QPen(QColor(100, 100, 100)))
            painter.drawRect(mm_x, mm_y, mm_w, mm_h)
            
            # Visible region
            painter.setPen(QPen(QColor(255, 200, 0), 2))
            painter.drawRect(rx, ry, rw, rh)
    
    def _draw_bbox_on_pixmap(self, painter: QPainter, x: int, y: int, 
                             w: int, h: int, name: str, style: str, color: Tuple[int, int, int]):
        """Draw bounding box on pixmap (zoom-aware)"""
        if self.scale_factor <= 0:
            return
        
        # Convert frame coordinates to display coordinates (zoom-aware)
        sx, sy, sw, sh = self._frame_to_display_coords(x, y, w, h)
        sx, sy, sw, sh = int(sx), int(sy), int(sw), int(sh)
        
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
        """Convert mouse coordinates to original image coordinates (zoom-aware)"""
        if self.current_frame is None or self.scale_factor <= 0:
            return None, None
        
        pixmap = self.pixmap()
        if pixmap is None:
            return None, None
        
        # Position relative to scaled pixmap
        rel_x = mouse_x - self.offset_x
        rel_y = mouse_y - self.offset_y
        
        # Check bounds
        if rel_x < 0 or rel_y < 0 or rel_x > pixmap.width() or rel_y > pixmap.height():
            return None, None
        
        # Convert to visible region coordinates
        vx, vy, vw, vh = self._get_visible_region()
        
        # Map from display to original frame coordinates
        x = int(vx + (rel_x / self.scale_factor))
        y = int(vy + (rel_y / self.scale_factor))
        
        # Clamp to frame bounds
        h, w = self.current_frame.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        return x, y
    
    # ===== ZOOM METHODS =====
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming (Cmd/Ctrl + Scroll)"""
        modifiers = event.modifiers()

        # Check for Cmd (Mac) or Ctrl (PC/Linux) modifier
        zoom_modifier = (modifiers & Qt.KeyboardModifier.ControlModifier or
                        modifiers & Qt.KeyboardModifier.MetaModifier)

        if zoom_modifier and self.current_frame is not None:
            # Get mouse position for zoom centering
            mouse_pos = event.position()
            
            # Calculate zoom direction
            delta = event.angleDelta().y()
            
            if delta > 0:
                # Zoom in
                new_zoom = min(self.max_zoom, self.zoom_level * (1 + self.zoom_step))
            else:
                # Zoom out
                new_zoom = max(self.min_zoom, self.zoom_level / (1 + self.zoom_step))
            
            if new_zoom != self.zoom_level:
                # Get image coords under mouse before zoom
                img_x, img_y = self._get_image_coords_from_mouse(mouse_pos.x(), mouse_pos.y())
                
                if img_x is not None:
                    # Update zoom
                    self.zoom_level = new_zoom
                    
                    # Adjust pan to keep mouse position stable
                    self.pan_x = img_x
                    self.pan_y = img_y
                    
                    self._clamp_pan()
                    self._update_display()
                    
                    # Emit signal for detection trigger
                    self.zoom_changed.emit(self.zoom_level, self._get_visible_region())
                    
                    # Request detection on zoomed area if zoomed in significantly
                    if self.zoom_level >= 1.5:
                        self._request_zoom_detection()
            
            event.accept()
        else:
            # Pass to parent for normal scrolling
            super().wheelEvent(event)
    
    def _clamp_pan(self):
        """Clamp pan position to keep view within frame bounds"""
        if self.current_frame is None:
            return
        
        frame_h, frame_w = self.current_frame.shape[:2]
        visible_w = frame_w / self.zoom_level
        visible_h = frame_h / self.zoom_level
        
        # Ensure pan center keeps visible region within bounds
        min_x = visible_w / 2
        max_x = frame_w - visible_w / 2
        min_y = visible_h / 2
        max_y = frame_h - visible_h / 2
        
        self.pan_x = max(min_x, min(self.pan_x, max_x))
        self.pan_y = max(min_y, min(self.pan_y, max_y))
    
    def _request_zoom_detection(self):
        """Request detection on the visible zoomed region"""
        if self.current_frame is None:
            return
        
        vx, vy, vw, vh = self._get_visible_region()
        frame_h, frame_w = self.current_frame.shape[:2]
        
        # Crop the visible region
        x1, y1 = int(max(0, vx)), int(max(0, vy))
        x2, y2 = int(min(frame_w, vx + vw)), int(min(frame_h, vy + vh))
        
        cropped = self.current_frame[y1:y2, x1:x2].copy()
        
        # Emit signal with cropped frame and original region
        self.zoom_detection_requested.emit(cropped, (x1, y1, x2 - x1, y2 - y1))
    
    def reset_zoom(self):
        """Reset zoom to 100% and center view"""
        self.zoom_level = 1.0
        if self.current_frame is not None:
            h, w = self.current_frame.shape[:2]
            self.pan_x = w / 2
            self.pan_y = h / 2
        self._update_display()
        self.zoom_changed.emit(self.zoom_level, self._get_visible_region())
    
    def set_zoom(self, level: float, center_x: float = None, center_y: float = None):
        """Set zoom level programmatically"""
        self.zoom_level = max(self.min_zoom, min(self.max_zoom, level))
        
        if center_x is not None and center_y is not None:
            self.pan_x = center_x
            self.pan_y = center_y
        
        self._clamp_pan()
        self._update_display()
        self.zoom_changed.emit(self.zoom_level, self._get_visible_region())
    
    def zoom_to_bbox(self, x: int, y: int, w: int, h: int, padding: float = 1.5):
        """Zoom to fit a bounding box in view"""
        if self.current_frame is None:
            return
        
        frame_h, frame_w = self.current_frame.shape[:2]
        
        # Calculate zoom level to fit bbox with padding
        zoom_w = frame_w / (w * padding)
        zoom_h = frame_h / (h * padding)
        new_zoom = min(zoom_w, zoom_h, self.max_zoom)
        
        # Center on bbox
        self.pan_x = x + w / 2
        self.pan_y = y + h / 2
        self.zoom_level = max(self.min_zoom, new_zoom)
        
        self._clamp_pan()
        self._update_display()
        self.zoom_changed.emit(self.zoom_level, self._get_visible_region())
    
    def get_visible_frame_crop(self) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Get the currently visible cropped frame and its original coordinates"""
        if self.current_frame is None:
            return None, (0, 0, 0, 0)
        
        vx, vy, vw, vh = self._get_visible_region()
        frame_h, frame_w = self.current_frame.shape[:2]
        
        x1, y1 = int(max(0, vx)), int(max(0, vy))
        x2, y2 = int(min(frame_w, vx + vw)), int(min(frame_h, vy + vh))
        
        cropped = self.current_frame[y1:y2, x1:x2].copy()
        return cropped, (x1, y1, x2 - x1, y2 - y1)
    
    def mousePressEvent(self, event):
        """Handle mouse press for drawing bbox, clicking detected person, or panning"""
        if event.button() == Qt.MouseButton.LeftButton and self.current_frame is not None:
            mouse_x = event.position().x()
            mouse_y = event.position().y()

            # Handle radar edit mode - click to confirm direction
            if self.radar_edit_mode and self.radar_preview_angle is not None:
                angle = self.radar_preview_angle
                self.radar_direction_set.emit(angle)
                self.exit_radar_edit_mode()
                return

            # Check for pan mode: Middle button OR Shift+Left OR zoomed in + Alt+Left
            modifiers = event.modifiers()
            is_pan_mode = (modifiers & Qt.KeyboardModifier.ShiftModifier or
                          (self.zoom_level > 1.0 and modifiers & Qt.KeyboardModifier.AltModifier))

            if is_pan_mode and self.zoom_level > 1.0:
                # Start panning
                self.panning = True
                self.pan_start_pos = QPoint(int(mouse_x), int(mouse_y))
                self.pan_start_offset = (self.pan_x, self.pan_y)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                return

            x, y = self._get_image_coords_from_mouse(mouse_x, mouse_y)
            if x is not None and y is not None:
                # If in detection mode, check if clicking on a detected person
                if self.detection_mode and self.detected_bboxes_display:
                    clicked_bbox = self._find_clicked_bbox(x, y)
                    if clicked_bbox:
                        self.person_clicked.emit(*clicked_bbox)
                        return

                # Otherwise, start drawing bbox
                self.drawing = True
                self.start_point = QPoint(x, y)
                self.end_point = QPoint(x, y)
                self.current_bbox = None
        
        elif event.button() == Qt.MouseButton.MiddleButton and self.zoom_level > 1.0:
            # Middle mouse button for panning
            self.panning = True
            self.pan_start_pos = QPoint(int(event.position().x()), int(event.position().y()))
            self.pan_start_offset = (self.pan_x, self.pan_y)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
    
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
        """Handle mouse move for drawing bounding box or panning"""
        mouse_x = event.position().x()
        mouse_y = event.position().y()

        # Handle radar edit mode - update preview on mouse move
        if self.radar_edit_mode and self.current_frame is not None:
            x, y = self._get_image_coords_from_mouse(mouse_x, mouse_y)
            if x is not None and y is not None:
                self.radar_mouse_pos = (x, y)
                self._update_display()
            return

        # Handle panning
        if self.panning and self.current_frame is not None:
            # Calculate drag distance in screen pixels
            dx = mouse_x - self.pan_start_pos.x()
            dy = mouse_y - self.pan_start_pos.y()
            
            # Convert to frame coordinates (inverse of scale)
            frame_dx = dx / self.scale_factor
            frame_dy = dy / self.scale_factor
            
            # Update pan (subtract because we're dragging the view, not the content)
            self.pan_x = self.pan_start_offset[0] - frame_dx
            self.pan_y = self.pan_start_offset[1] - frame_dy
            
            self._clamp_pan()
            self._update_display()
            return
        
        # Handle bbox drawing
        if self.drawing and self.current_frame is not None:
            x, y = self._get_image_coords_from_mouse(mouse_x, mouse_y)
            if x is not None and y is not None:
                self.end_point = QPoint(x, y)
                
                x1 = min(self.start_point.x(), self.end_point.x())
                y1 = min(self.start_point.y(), self.end_point.y())
                x2 = max(self.start_point.x(), self.end_point.x())
                y2 = max(self.start_point.y(), self.end_point.y())
                
                w = x2 - x1
                h = y2 - y1
                
                if w > 10 and h > 10:
                    self.current_bbox = (x1, y1, w, h)
                    self._update_display()
        
        # Update cursor when zoomed
        if self.zoom_level > 1.0 and not self.panning and not self.drawing:
            modifiers = QApplication.keyboardModifiers()
            if modifiers & Qt.KeyboardModifier.ShiftModifier or modifiers & Qt.KeyboardModifier.AltModifier:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release to finalize bounding box or stop panning"""
        # Stop panning
        if self.panning:
            self.panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False

            if self.current_bbox:
                x, y, w, h = self.current_bbox
                if w > 10 and h > 10:
                    self.bbox_selected.emit(x, y, w, h)
                else:
                    # Small or no movement - treat as a click
                    # Emit frame_clicked with the center point
                    click_x = x + w // 2
                    click_y = y + h // 2
                    self.frame_clicked.emit(click_x, click_y)
            else:
                # No bbox drawn - emit click at start point
                self.frame_clicked.emit(self.start_point.x(), self.start_point.y())

            self.current_bbox = None
            self._update_display()
        
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def mouseDoubleClickEvent(self, event):
        """Double-click to reset zoom or zoom to clicked point"""
        if event.button() == Qt.MouseButton.LeftButton and self.current_frame is not None:
            if self.zoom_level > 1.0:
                # Reset zoom on double-click when zoomed
                self.reset_zoom()
            else:
                # Zoom in to clicked point
                x, y = self._get_image_coords_from_mouse(event.position().x(), event.position().y())
                if x is not None and y is not None:
                    self.set_zoom(2.0, x, y)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for zoom"""
        key = event.key()
        modifiers = event.modifiers()
        
        # Cmd/Ctrl + Plus: Zoom In
        if (modifiers & Qt.KeyboardModifier.ControlModifier or modifiers & Qt.KeyboardModifier.MetaModifier):
            if key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
                self.set_zoom(self.zoom_level * 1.25, self.pan_x, self.pan_y)
                event.accept()
                return
            elif key == Qt.Key.Key_Minus:
                self.set_zoom(self.zoom_level / 1.25, self.pan_x, self.pan_y)
                event.accept()
                return
            elif key == Qt.Key.Key_0:
                self.reset_zoom()
                event.accept()
                return
        
        # Escape: Cancel radar edit mode or reset zoom
        if key == Qt.Key.Key_Escape:
            if self.radar_edit_mode:
                self.exit_radar_edit_mode()
                event.accept()
                return
            elif self.zoom_level > 1.0:
                self.reset_zoom()
                event.accept()
                return
        
        super().keyPressEvent(event)
    
    def resizeEvent(self, event):
        """Handle widget resize"""
        super().resizeEvent(event)
        self._clamp_pan()  # Ensure pan stays valid after resize
        self._update_display()
    
    def focusInEvent(self, event):
        """Ensure widget accepts focus for keyboard events"""
        super().focusInEvent(event)
        self.setFocus()  # Ensure keyboard events work

    # ===== RADAR EDITING MODE =====

    def enter_radar_edit_mode(self, player_bbox: Tuple[int, int, int, int]):
        """
        Enter radar direction editing mode.

        Args:
            player_bbox: (x, y, w, h) of the player's bounding box
        """
        print(f"VideoCanvas.enter_radar_edit_mode called with bbox: {player_bbox}")
        self.radar_edit_mode = True
        self.radar_player_bbox = player_bbox
        self.radar_preview_angle = None
        self.radar_mouse_pos = None
        self.radar_preview_size = 1.0  # Reset size
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._update_display()
        print(f"  radar_edit_mode is now: {self.radar_edit_mode}")

    def exit_radar_edit_mode(self):
        """Exit radar direction editing mode"""
        self.radar_edit_mode = False
        self.radar_player_bbox = None
        self.radar_preview_angle = None
        self.radar_mouse_pos = None
        self.radar_preview_size = 1.0
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._update_display()

    def update_radar_bbox(self, player_bbox: Tuple[int, int, int, int]):
        """Update the radar bbox (called when player moves)"""
        if self.radar_edit_mode:
            self.radar_player_bbox = player_bbox
            self._update_display()

    def _draw_radar_preview(self, painter: QPainter):
        """Draw radar cone preview - distance to mouse = radar size, direction = radar angle"""
        import math

        if not self.radar_player_bbox or not self.radar_mouse_pos:
            return

        bx, by, bw, bh = self.radar_player_bbox
        mouse_x, mouse_y = self.radar_mouse_pos

        # Player center (at feet level)
        player_x = bx + bw // 2
        player_y = by + bh  # feet level

        # Calculate angle AND distance to mouse
        dx = mouse_x - player_x
        dy = mouse_y - player_y
        angle = math.atan2(dy, dx)
        distance = math.sqrt(dx * dx + dy * dy)

        self.radar_preview_angle = angle

        # Cone length = distance to mouse position!
        cone_length = max(20, int(distance))  # Minimum 20 pixels

        # Calculate size multiplier relative to default size (for saving to keyframe)
        base_cone_length = max(1, int(bh * 1.2))
        self.radar_preview_size = cone_length / base_cone_length

        cone_half_angle = 30  # degrees

        # Calculate cone edge angles
        left_angle = angle - math.radians(cone_half_angle)
        right_angle = angle + math.radians(cone_half_angle)

        # End points of cone (in frame coordinates) - ends at mouse distance
        end_left_x = player_x + cone_length * math.cos(left_angle)
        end_left_y = player_y + cone_length * math.sin(left_angle)
        end_right_x = player_x + cone_length * math.cos(right_angle)
        end_right_y = player_y + cone_length * math.sin(right_angle)

        # Convert to display coordinates
        origin_dx, origin_dy, _, _ = self._frame_to_display_coords(player_x, player_y, 0, 0)
        left_dx, left_dy, _, _ = self._frame_to_display_coords(end_left_x, end_left_y, 0, 0)
        right_dx, right_dy, _, _ = self._frame_to_display_coords(end_right_x, end_right_y, 0, 0)
        mouse_dx, mouse_dy, _, _ = self._frame_to_display_coords(mouse_x, mouse_y, 0, 0)

        # Radar colors - green style matching actual radar
        from PyQt6.QtGui import QPolygon, QBrush
        radar_green = QColor(0, 255, 100)
        radar_green_dark = QColor(0, 180, 60)
        radar_green_trans = QColor(0, 255, 100, 50)

        # Draw semi-transparent cone fill
        painter.setBrush(QBrush(radar_green_trans))
        painter.setPen(Qt.PenStyle.NoPen)

        polygon = QPolygon([
            QPoint(int(origin_dx), int(origin_dy)),
            QPoint(int(left_dx), int(left_dy)),
            QPoint(int(right_dx), int(right_dy))
        ])
        painter.drawPolygon(polygon)

        # Draw arc lines
        num_arcs = 4
        pen = QPen(radar_green_dark)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        for i in range(1, num_arcs + 1):
            arc_radius_frame = int(cone_length * i / num_arcs)
            arc_radius = int(arc_radius_frame * self.scale_factor)
            start_angle_deg = int(math.degrees(-right_angle)) * 16
            span_angle_deg = int(math.degrees(right_angle - left_angle)) * 16
            painter.drawArc(
                int(origin_dx - arc_radius), int(origin_dy - arc_radius),
                arc_radius * 2, arc_radius * 2,
                start_angle_deg, span_angle_deg
            )

        # Draw radial lines
        num_radials = 5
        for i in range(num_radials + 1):
            t = i / num_radials
            line_angle = left_angle + t * (right_angle - left_angle)
            end_x = player_x + cone_length * math.cos(line_angle)
            end_y = player_y + cone_length * math.sin(line_angle)
            end_dx, end_dy, _, _ = self._frame_to_display_coords(end_x, end_y, 0, 0)
            painter.drawLine(int(origin_dx), int(origin_dy), int(end_dx), int(end_dy))

        # Draw cone outline
        pen = QPen(radar_green)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawPolygon(polygon)

        # Draw origin point with glow effect
        painter.setBrush(QBrush(radar_green_dark))
        painter.drawEllipse(QPoint(int(origin_dx), int(origin_dy)), 8, 8)
        painter.setBrush(QBrush(radar_green))
        painter.drawEllipse(QPoint(int(origin_dx), int(origin_dy)), 5, 5)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawEllipse(QPoint(int(origin_dx), int(origin_dy)), 3, 3)

        # Draw crosshair at mouse position (shows exactly where radar edge will be)
        pen.setStyle(Qt.PenStyle.SolidLine)
        pen.setColor(QColor(255, 255, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        crosshair_size = 10
        painter.drawLine(int(mouse_dx - crosshair_size), int(mouse_dy),
                        int(mouse_dx + crosshair_size), int(mouse_dy))
        painter.drawLine(int(mouse_dx), int(mouse_dy - crosshair_size),
                        int(mouse_dx), int(mouse_dy + crosshair_size))

        # Draw instruction text
        painter.setPen(QPen(QColor(255, 255, 255)))
        font = painter.font()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(10, 25, "📡 RADAR: Move to aim | Distance = Size | Click to confirm | ESC to cancel")

