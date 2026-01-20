"""
Overlay Renderer - Renders visual markers on video frames
"""
import cv2
import numpy as np
from typing import Tuple, Optional
import math
from .modern_styles import ModernStyles


class OverlayRenderer:
    """Renders visual markers (arrows, circles, rectangles) on frames"""
    
    def __init__(self):
        self.arrow_size = 30
        self.circle_thickness = 3
        self.circle_glow_size = 5
        self.rectangle_thickness = 3
        
        # Position smoothing for markers (especially circles)
        self.position_buffers = {}  # player_id -> [(center_x, center_y), ...]
        
        # Modern styles instance
        self.modern_styles = ModernStyles()
        
        # Frame counter for animations
        self.frame_count = 0
        self.position_buffer_size = 5
    
    def draw_marker(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                   marker_style: str, color: Tuple[int, int, int],
                   player=None) -> np.ndarray:
        """
        Draw marker on frame based on style

        Args:
            frame: Frame to draw on (BGR format)
            bbox: Bounding box (x, y, width, height)
            marker_style: Style ('arrow', 'circle', 'rectangle', 'spotlight', 'outline',
                          'neon_ring', 'pulse', 'gradient', 'nba_iso_ring',
                          'tactical_brackets', 'sonar_ripple', 'floating_chevron',
                          'dynamic_arrow', 'hexagon', 'crosshair')
            color: BGR color tuple
            player: Player object (optional, for accessing original_bbox)

        Returns:
            Frame with marker drawn
        """
        if bbox is None:
            return frame

        # DEBUG: Log which marker is being used
        print(f"[MARKER DEBUG] Using marker_style: {marker_style}")

        # Increment frame counter for animations
        self.frame_count += 1

        x, y, w, h = bbox
        
        # Classic styles
        if marker_style == 'arrow':
            return self._draw_arrow(frame, bbox, color, player)
        elif marker_style == 'circle':
            return self._draw_circle(frame, bbox, color, player)
        elif marker_style == 'rectangle':
            return self._draw_rectangle(frame, bbox, color)
        elif marker_style == 'spotlight':
            return self._draw_spotlight(frame, bbox, color)
        elif marker_style == 'outline':
            return self._draw_outline(frame, bbox, color)
        
        # Modern styles
        elif marker_style == 'neon_ring':
            return self.modern_styles.draw_neon_ring(frame, bbox, color, player)
        elif marker_style == 'pulse':
            # Force orange color for pulse (BGR: 0, 165, 255)
            orange_color = (0, 165, 255)
            return self.modern_styles.draw_pulse_circle(frame, bbox, orange_color, self.frame_count, player)
        elif marker_style == 'gradient':
            # Gradient colors (purple variants)
            color1 = (255, 0, 200)  # Purple
            color2 = (200, 0, 255)  # Purple variant
            return self.modern_styles.draw_gradient_ring(frame, bbox, color1, color2, self.frame_count, player)
        elif marker_style == 'nba_iso_ring':
            return self.modern_styles.draw_nba_iso_ring(frame, bbox, color, self.frame_count, player)
        elif marker_style == 'tactical_brackets':
            return self.modern_styles.draw_tactical_brackets(frame, bbox, color, self.frame_count, player)
        elif marker_style == 'sonar_ripple':
            return self.modern_styles.draw_sonar_ripple(frame, bbox, color, self.frame_count, player)
        elif marker_style == 'floating_chevron':
            return self.modern_styles.draw_floating_chevron(frame, bbox, color, self.frame_count, player)
        elif marker_style == 'dynamic_arrow':
            # Bright cyan for high visibility
            cyan_color = (255, 255, 0)  # Bright yellow-cyan
            return self.modern_styles.draw_dynamic_arrow(frame, bbox, cyan_color, self.frame_count, player)
        elif marker_style == 'hexagon':
            return self.modern_styles.draw_hexagon_outline(frame, bbox, color, player)
        elif marker_style == 'crosshair':
            return self.modern_styles.draw_crosshair(frame, bbox, color)
        elif marker_style == 'spotlight_modern':
            # Cyan/white color for alien beam
            beam_color = (200, 255, 255)
            return self.modern_styles.draw_spotlight(frame, bbox, beam_color, player)
        elif marker_style == 'flame':
            # Gold color for premium star
            gold_color = (0, 215, 255)
            return self.modern_styles.draw_flame(frame, bbox, gold_color, self.frame_count, player)
        elif marker_style == 'dramatic_floor_uplight':
            # Warm white for dramatic floor uplight
            warm_white = (200, 240, 255)
            return self.modern_styles.draw_dramatic_floor_uplight(frame, bbox, warm_white, intensity=1.0, radius=None, player=player)
        else:
            return frame
    
    def _draw_arrow(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                   color: Tuple[int, int, int], player=None) -> np.ndarray:
        """
        Draw impressive 3D arrow above player's head - championship broadcast style

        Args:
            frame: Frame to draw on
            bbox: Bounding box (padded)
            color: Arrow color (bright yellow)
            player: Player object (for accessing original_bbox)

        Returns:
            Frame with arrow
        """
        x, y, w, h = bbox

        # Position arrow above head using original_bbox if available
        if hasattr(player, 'current_original_bbox') and player and player.current_original_bbox:
            orig_x, orig_y, orig_w, orig_h = player.current_original_bbox
            center_x = orig_x + orig_w // 2
            # Position much higher above original head
            arrow_y = max(0, orig_y - 60)
        else:
            # Fallback
            center_x = x + w // 2
            arrow_y = max(0, y - 60)

        arrow_x = center_x

        # Professional arrow design - sleek and modern, pointing DOWN at player
        arrow_size = max(int(w * 0.35), 35)
        shaft_width = max(int(arrow_size * 0.25), 8)
        head_width = max(int(arrow_size * 0.6), 20)

        # Arrow pointing DOWN: shaft at top, head at bottom (toward player)
        shaft_start_y = arrow_y
        shaft_end_y = arrow_y + int(arrow_size * 0.6)
        tip_y = arrow_y + arrow_size  # Tip points DOWN toward player

        # Professional yellow color (brighter, more saturated)
        yellow_bright = (0, 220, 255)  # Bright yellow in BGR
        yellow_glow = (100, 235, 255)  # Lighter glow

        # Draw outer glow for depth
        for i in range(5, 0, -1):
            overlay = frame.copy()
            glow_factor = 1.0 + (i * 0.15)

            # Glow for shaft
            glow_shaft_points = np.array([
                [arrow_x - int(shaft_width * glow_factor) // 2, shaft_start_y],
                [arrow_x + int(shaft_width * glow_factor) // 2, shaft_start_y],
                [arrow_x + int(shaft_width * glow_factor) // 2, shaft_end_y],
                [arrow_x - int(shaft_width * glow_factor) // 2, shaft_end_y]
            ], np.int32)
            cv2.fillPoly(overlay, [glow_shaft_points], yellow_glow)

            # Glow for arrowhead (pointing DOWN)
            glow_head_points = np.array([
                [arrow_x - int(head_width * glow_factor) // 2, shaft_end_y],
                [arrow_x + int(head_width * glow_factor) // 2, shaft_end_y],
                [arrow_x, tip_y]
            ], np.int32)
            cv2.fillPoly(overlay, [glow_head_points], yellow_glow)

            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

        # Draw main arrow shaft (rectangle at top)
        shaft_points = np.array([
            [arrow_x - shaft_width // 2, shaft_start_y],
            [arrow_x + shaft_width // 2, shaft_start_y],
            [arrow_x + shaft_width // 2, shaft_end_y],
            [arrow_x - shaft_width // 2, shaft_end_y]
        ], np.int32)
        cv2.fillPoly(frame, [shaft_points], yellow_bright)

        # Draw main arrow head (triangle pointing DOWN)
        head_points = np.array([
            [arrow_x - head_width // 2, shaft_end_y],
            [arrow_x + head_width // 2, shaft_end_y],
            [arrow_x, tip_y]
        ], np.int32)
        cv2.fillPoly(frame, [head_points], yellow_bright)

        # Add white highlight on arrow head for 3D effect
        highlight_points = np.array([
            [arrow_x - head_width // 4, shaft_end_y + 3],
            [arrow_x + head_width // 4, shaft_end_y + 3],
            [arrow_x, int(tip_y * 0.6 + shaft_end_y * 0.4)]
        ], np.int32)
        cv2.fillPoly(frame, [highlight_points], (200, 255, 255))

        # Add dark outline for definition
        dark_yellow = (0, 180, 220)
        cv2.polylines(frame, [shaft_points], True, dark_yellow, 2, cv2.LINE_AA)
        cv2.polylines(frame, [head_points], True, dark_yellow, 2, cv2.LINE_AA)
        
        return frame
    
    def _draw_circle(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                    color: Tuple[int, int, int], player=None) -> np.ndarray:
        """
        Draw 3D floor hoop around player's feet (like professional sports broadcasts)

        Args:
            frame: Frame to draw on
            bbox: Bounding box (padded)
            color: Circle color
            player: Player object (for accessing original_bbox)

        Returns:
            Frame with 3D floor hoop
        """
        x, y, w, h = bbox
        
        # Calculate ellipse center (feet position - on the floor)
        # X axis: Use precise center (padded bbox is fine for X)
        center_x = x + w // 2

        # Calculate ellipse size - proportional to player width
        radius_x = max(int(w * 0.6), 35)  # Horizontal radius
        radius_y = max(int(w * 0.15), 10)  # Vertical radius (flat ellipse)

        # Y axis: Position circle at feet level
        # Use original_bbox if available (before padding was added)
        # If not available, fallback to using the padded bbox
        if hasattr(player, 'current_original_bbox') and player.current_original_bbox:
            orig_x, orig_y, orig_w, orig_h = player.current_original_bbox
            # Feet are at the BOTTOM of the original bbox (where the actual feet are)
            feet_y = orig_y + orig_h
        else:
            # Fallback: assume no padding, use bottom of bbox
            feet_y = y + h

        center_y = feet_y - radius_y
        
        # Ensure within frame bounds
        frame_h, frame_w = frame.shape[:2]
        if center_x < 0 or center_x >= frame_w or center_y < 0 or center_y >= frame_h:
            return frame
        
        # Draw full circle (360 degrees) - MORE TRANSPARENT so it doesn't hide video
        # Draw glow effect (multiple ellipses with decreasing opacity) - REDUCED opacity
        for i in range(self.circle_glow_size):
            alpha = 0.12 - (i * 0.02)  # Much more transparent (was 0.25)
            glow_rx = radius_x + i * 3  # Smaller glow (was i * 4)
            glow_ry = radius_y + i * 1  # Smaller glow (was i * 2)
            overlay = frame.copy()
            cv2.ellipse(overlay, (center_x, center_y), (glow_rx, glow_ry), 0, 0, 360, color, 2)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw main ellipse (full circle - 360 degrees) - THINNER and MORE TRANSPARENT
        overlay = frame.copy()
        cv2.ellipse(overlay, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, color, max(1, self.circle_thickness - 1))
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)  # 60% opacity - more transparent
        
        # Add inner highlight for depth - MORE TRANSPARENT
        inner_rx = int(radius_x * 0.85)
        inner_ry = int(radius_y * 0.85)
        lighter_color = tuple(min(c + 60, 255) for c in color)
        overlay = frame.copy()
        cv2.ellipse(overlay, (center_x, center_y), (inner_rx, inner_ry), 0, 0, 360, lighter_color, 1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)  # 40% opacity - very transparent
        
        return frame
    
    def _draw_rectangle(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                       color: Tuple[int, int, int]) -> np.ndarray:
        """
        Draw clean blue border rectangle around player (no fill)
        Large size to allow freedom of movement

        Args:
            frame: Frame to draw on
            bbox: Bounding box (padded)
            color: Rectangle color (will be overridden to blue)

        Returns:
            Frame with rectangle
        """
        x, y, w, h = bbox

        # Force blue color for rectangle (BGR format)
        blue_color = (255, 100, 0)  # Bright blue

        # Expand rectangle to give more room (20% extra space on all sides)
        margin_x = int(w * 0.20)
        margin_y = int(h * 0.20)
        rect_x = max(0, x - margin_x)
        rect_y = max(0, y - margin_y)
        rect_w = w + (margin_x * 2)
        rect_h = h + (margin_y * 2)

        # Draw outer glow for depth
        padding = 2
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (rect_x - padding, rect_y - padding),
                     (rect_x + rect_w + padding, rect_y + rect_h + padding),
                     blue_color,
                     self.rectangle_thickness + 1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Draw main border (clean, no fill)
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), blue_color, self.rectangle_thickness)
        
        # Add corner highlights for professional look
        corner_size = 15
        corner_color = (255, 200, 100)  # Light cyan
        # Top-left corner
        cv2.line(frame, (rect_x, rect_y), (rect_x + corner_size, rect_y), corner_color, 2)
        cv2.line(frame, (rect_x, rect_y), (rect_x, rect_y + corner_size), corner_color, 2)
        # Top-right corner
        cv2.line(frame, (rect_x + rect_w, rect_y), (rect_x + rect_w - corner_size, rect_y), corner_color, 2)
        cv2.line(frame, (rect_x + rect_w, rect_y), (rect_x + rect_w, rect_y + corner_size), corner_color, 2)
        # Bottom-left corner
        cv2.line(frame, (rect_x, rect_y + rect_h), (rect_x + corner_size, rect_y + rect_h), corner_color, 2)
        cv2.line(frame, (rect_x, rect_y + rect_h), (rect_x, rect_y + rect_h - corner_size), corner_color, 2)
        # Bottom-right corner
        cv2.line(frame, (rect_x + rect_w, rect_y + rect_h), (rect_x + rect_w - corner_size, rect_y + rect_h), corner_color, 2)
        cv2.line(frame, (rect_x + rect_w, rect_y + rect_h), (rect_x + rect_w, rect_y + rect_h - corner_size), corner_color, 2)
        
        return frame
    
    def _draw_spotlight(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                       color: Tuple[int, int, int]) -> np.ndarray:
        """
        Draw spotlight effect on player (like stadium lighting)
        
        Args:
            frame: Frame to draw on
            bbox: Bounding box
            color: Spotlight color
            
        Returns:
            Frame with spotlight
        """
        x, y, w, h = bbox
        
        # Calculate spotlight center
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Calculate spotlight radius
        radius = max(w, h) // 2 + 20
        
        # Create spotlight mask with gradient
        overlay = frame.copy()
        
        # Draw multiple circles with decreasing opacity for smooth gradient
        for i in range(10, 0, -1):
            alpha = 0.15 * (i / 10)
            current_radius = int(radius * (1 + (10 - i) * 0.1))
            cv2.circle(overlay, (center_x, center_y), current_radius, color, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add bright center highlight
        cv2.circle(frame, (center_x, center_y), radius // 3, color, -1)
        overlay = frame.copy()
        cv2.circle(overlay, (center_x, center_y), radius // 3, (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        return frame
    
    def _draw_outline(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                     color: Tuple[int, int, int]) -> np.ndarray:
        """
        Draw professional glow outline around player (like TV graphics)
        
        Args:
            frame: Frame to draw on
            bbox: Bounding box
            color: Outline color
            
        Returns:
            Frame with outline
        """
        x, y, w, h = bbox
        
        # Create mask for the player area
        overlay = frame.copy()
        
        # Draw thick outer glow
        for i in range(10, 0, -1):
            alpha = 0.08 * (i / 10)
            thickness = i * 2
            padding = i * 3
            cv2.rectangle(overlay, 
                         (x - padding, y - padding), 
                         (x + w + padding, y + h + padding), 
                         color, 
                         thickness)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw main outline
        cv2.rectangle(frame, (x - 2, y - 2), (x + w + 2, y + h + 2), color, 3)
        
        # Add bright inner edge
        lighter_color = tuple(min(c + 80, 255) for c in color)
        cv2.rectangle(frame, (x + 2, y + 2), (x + w - 2, y + h - 2), lighter_color, 2)
        
        return frame
    
    def draw_all_markers(self, frame: np.ndarray, 
                        players_data: list,
                        frame_idx: Optional[int] = None,
                        tracking_start_frame: Optional[int] = None,
                        tracking_end_frame: Optional[int] = None) -> np.ndarray:
        """
        Draw all player markers on frame
        
        Args:
            frame: Frame to draw on
            players_data: List of player data objects with bbox, style, color
            frame_idx: Current frame index (for tracking range check)
            tracking_start_frame: Start frame for tracking (None = from beginning)
            tracking_end_frame: End frame for tracking (None = to end)
            
        Returns:
            Frame with all markers
        """
        result_frame = frame.copy()
        
        # Check if we should draw markers for this frame (respect tracking range)
        should_draw = True
        if frame_idx is not None:
            if tracking_start_frame is not None and frame_idx < tracking_start_frame:
                should_draw = False  # Before tracking start - don't draw!
            if tracking_end_frame is not None and frame_idx > tracking_end_frame:
                should_draw = False  # After tracking end - don't draw!

        # DEBUG: Log first few frames
        if frame_idx is not None and frame_idx < 3:
            print(f"[Overlay Debug] Frame {frame_idx}: should_draw={should_draw}, tracking_range=[{tracking_start_frame}, {tracking_end_frame}]")
            print(f"   Players with bbox: {sum(1 for p in players_data if p.current_bbox is not None)}/{len(players_data)}")

        if not should_draw:
            # Don't draw any markers - return frame as-is
            return result_frame

        # Special handling for spotlight_modern and dramatic_floor_uplight markers
        # When multiple players have these effects, we need to darken the frame ONCE
        # and then draw all the light effects on the same darkened frame

        spotlight_players = [p for p in players_data
                            if p.current_bbox is not None and p.marker_style == 'spotlight_modern']
        uplight_players = [p for p in players_data
                          if p.current_bbox is not None and p.marker_style == 'dramatic_floor_uplight']
        other_players = [p for p in players_data
                        if p.current_bbox is not None 
                        and p.marker_style != 'spotlight_modern' 
                        and p.marker_style != 'dramatic_floor_uplight']

        # If there are spotlight or uplight players, handle them specially
        if spotlight_players or uplight_players:
            # Darken the entire frame ONCE (not per player!)
            darkened_frame = (result_frame.astype(np.float32) * 0.50).astype(np.uint8)

            # Handle spotlight players
            if spotlight_players:
                # Collect all spotlight masks and combine them
                combined_mask = np.zeros((result_frame.shape[0], result_frame.shape[1]), dtype=np.float32)

                for player in spotlight_players:
                    # Get the mask for this spotlight (without actually drawing)
                    mask = self.modern_styles.get_spotlight_mask(
                        result_frame.shape,
                        player.current_bbox,
                        player
                    )
                    # Combine masks using maximum (brightest wins)
                    combined_mask = np.maximum(combined_mask, mask)

                # Apply the combined mask once
                print(f"[SPOTLIGHT RENDER] Applying spotlight mask, combined_mask shape: {combined_mask.shape}, max: {combined_mask.max():.3f}")
                result_frame = self.modern_styles.apply_spotlight_mask(
                    result_frame,
                    darkened_frame,
                    combined_mask
                )
                print(f"[SPOTLIGHT RENDER] After apply_spotlight_mask, result_frame dtype: {result_frame.dtype}, shape: {result_frame.shape}")

                # Draw floor circles for each spotlight player
                for player in spotlight_players:
                    result_frame = self.modern_styles.draw_spotlight_floor_circle(
                        result_frame,
                        player.current_bbox,
                        (200, 255, 255),  # Cyan beam color
                        player
                    )
            
            # Handle dramatic floor uplight players
            # Apply dimming once for all uplight players (if not already dimmed by spotlights)
            if uplight_players:
                # If we haven't dimmed yet (no spotlights), dim now for uplights
                if not spotlight_players:
                    dimming_factor = 0.7  # 30% darkening
                    result_frame = (result_frame.astype(np.float32) * dimming_factor).astype(np.uint8)
                
                # Draw each uplight (skip dimming since we already dimmed once)
                for player in uplight_players:
                    warm_white = (200, 240, 255)
                    result_frame = self.modern_styles.draw_dramatic_floor_uplight(
                        result_frame, 
                        player.current_bbox, 
                        warm_white, 
                        intensity=1.0, 
                        radius=None, 
                        player=player,
                        skip_dimming=True  # Skip dimming since we already dimmed once
                    )

        # Draw all other (non-spotlight) markers normally
        for player in other_players:
            result_frame = self.draw_marker(
                result_frame,
                player.current_bbox,
                player.marker_style,
                player.color,
                player
            )

        return result_frame
