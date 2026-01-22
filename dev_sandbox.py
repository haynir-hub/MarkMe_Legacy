import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt
from src.ui.bbox_editor import BboxEditor
from src.tracking.person_detector import PersonDetector
from src.render.overlay_renderer import OverlayRenderer

class MockPlayer:
    """Helper to simulate player object for renderers"""
    def __init__(self, bbox):
        self.current_original_bbox = bbox
        self.current_bbox = bbox
        self.tracking_lost = False
        self.color = (255, 255, 0) # Default Cyan/Yellow

class SandboxEditor(BboxEditor):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_frame = None
        self.detected_bbox = None
        self.renderer = OverlayRenderer()
        # LIST OF STYLES TO TOGGLE
        self.styles = ['dynamic_ring_3d', 'spotlight_alien', 'solid_anchor', 'radar_defensive', 'sniper_scope'] 
        self.current_style_index = 0
        
        # Ensure focus for keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_test_data(self, frame, bbox):
        self.original_frame = frame.copy()
        self.detected_bbox = bbox
        self.refresh_view()

    def refresh_view(self):
        if self.original_frame is None:
            return
            
        current_style = self.styles[self.current_style_index]
        print(f"Refreshing view with style: {current_style}")
        
        # Create a display copy
        display_frame = self.original_frame.copy()
        
        # Draw the marker
        player = MockPlayer(self.detected_bbox)
        # Use Cyan/Gold default for spotlight, Purple is hardcoded for ring in renderer
        color = (255, 255, 0)

        display_frame = self.renderer.draw_marker(
            display_frame,
            self.detected_bbox,
            current_style,
            color,
            player
        )

        # Update parent BboxEditor
        self.set_frame(display_frame, self.detected_bbox)

    def keyPressEvent(self, event):
        # Press SPACE or M to cycle styles
        if event.key() == Qt.Key.Key_Space or event.key() == Qt.Key.Key_M:
            self.current_style_index = (self.current_style_index + 1) % len(self.styles)
            style_name = self.styles[self.current_style_index]
            print(f"\n>>> SWITCHED TO: {style_name} <<<")
            self.refresh_view()
        else:
            super().keyPressEvent(event)

def main():
    app = QApplication(sys.argv)
    
    # Load Video
    video_path = 'test.mp4'
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found. Please place it in the root directory.")
        return
        
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not read frame")
        return

    # Detect Player (Audience Filtering)
    # Note: PersonDetector loads yolov8n.pt internally
    detector = PersonDetector()
    detections = detector.detect_people(frame)
    
    h, w = frame.shape[:2]
    play_zone_bottom = int(h * 0.75) # Ignore bottom 25% (audience foreground)
    
    best_bbox = None
    min_dist = float('inf')
    center_target = (w // 2, h * 0.4)
    
    for x, y, bw, bh, conf in detections:
        feet_y = y + bh
        # Filter audience at bottom
        if feet_y > play_zone_bottom:
            continue
            
        cx, cy = x + bw/2, y + bh/2
        dist = ((cx - center_target[0])**2 + (cy - center_target[1])**2)**0.5
        
        if dist < min_dist:
            min_dist = dist
            best_bbox = (int(x), int(y), int(bw), int(bh))

    if best_bbox is None:
        print("No player found on court, using dummy.")
        best_bbox = (w//2 - 50, h//2 - 100, 100, 200)
    else:
        print(f"Selected Player Bbox: {best_bbox}")

    # Launch Editor
    editor = SandboxEditor()
    editor.set_test_data(frame, best_bbox)
    editor.setWindowTitle("Interactive Marker Sandbox - Press SPACE to switch")
    
    # Wrap in QMainWindow for proper maximize behavior
    window = QMainWindow()
    window.setCentralWidget(editor)
    window.setWindowTitle("MarkMe Legacy - Interactive Marker Sandbox")
    window.showMaximized()
    
    print("\n---------------------------------------------------")
    print(" SANDBOX RUNNING")
    print(" Press SPACEBAR to toggle between styles:")
    print("   - dynamic_ring_3d (Purple 3D Ring)")
    print("   - spotlight_alien (Alien Beam)")
    print("   - solid_anchor (Green Floor Ellipse)")
    print("   - radar_defensive (Red Radar Cone)")
    print("   - sniper_scope (Crosshair Reticle)")
    print("---------------------------------------------------\n")
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
