import cv2
import numpy as np
import math
from typing import Tuple, Optional

class ModernStyles:
    """
    Modern Marker Styles - Clean implementation
    """
    @staticmethod
    def draw_dynamic_ring_3d(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                            color: Tuple[int, int, int] = (255, 0, 180), # Purple
                            frame_count: int = 0, player=None, full_ring: bool = False) -> np.ndarray:
        """
        Dynamic 3D Ring with trapezoid cutout under player.
        The area under the player's body is transparent, outer ring is solid.
        """
        x, y, w, h = bbox
        center_x = x + w // 2

        # Calculate feet level
        feet_y = y + h
        if hasattr(player, 'current_original_bbox') and player and player.current_original_bbox:
            _, orig_y, _, orig_h = player.current_original_bbox
            feet_y = orig_y + orig_h

        # Perspective setup
        radius_x = max(int(w * 0.7), 40)
        radius_y = max(int(w * 0.2), 12)

        # Breathing animation
        pulse = 0.5 + 0.5 * math.sin(frame_count * 0.15)

        # Define trapezoid for "under player" area (will be faded)
        feet_width = int(w * 0.7)  # Wider at feet
        body_width = int(w * 1.2)  # Much wider towards body
        cut_height = radius_y + 8  # Extends further back

        trapezoid_pts = np.array([
            [center_x - feet_width // 2, feet_y],
            [center_x + feet_width // 2, feet_y],
            [center_x + body_width // 2, feet_y - cut_height],
            [center_x - body_width // 2, feet_y - cut_height]
        ], np.int32)

        # Create trapezoid mask
        trap_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(trap_mask, trapezoid_pts, 255)

        # Helper to draw ring with trapezoid transparency
        def draw_ring_with_cutout(img, center, axes, col, thickness, alpha=1.0):
            overlay = img.copy()
            # Draw full ellipse
            cv2.ellipse(overlay, center, axes, 0, 0, 360, col, thickness, cv2.LINE_AA)

            # Create ring mask (just the ring outline area)
            ring_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.ellipse(ring_mask, center, axes, 0, 0, 360, 255, thickness + 4, cv2.LINE_AA)

            # Split into: outside trapezoid (solid) and inside trapezoid (faded)
            outside_trap = cv2.bitwise_and(ring_mask, cv2.bitwise_not(trap_mask))
            inside_trap = cv2.bitwise_and(ring_mask, trap_mask)

            # Blend outside trapezoid with full alpha
            outside_3ch = cv2.merge([outside_trap, outside_trap, outside_trap]).astype(np.float32) / 255.0
            inside_3ch = cv2.merge([inside_trap, inside_trap, inside_trap]).astype(np.float32) / 255.0

            img_f = img.astype(np.float32)
            overlay_f = overlay.astype(np.float32)

            # Outside: full alpha, Inside: 25% alpha
            result = img_f.copy()
            result = result * (1.0 - outside_3ch * alpha) + overlay_f * (outside_3ch * alpha)
            result = result * (1.0 - inside_3ch * alpha * 0.25) + overlay_f * (inside_3ch * alpha * 0.25)

            return np.clip(result, 0, 255).astype(np.uint8)

        # Draw Glow Layers
        for i in range(3):
            alpha = 0.3 - (i * 0.08)
            rx = radius_x + (i * 4) + int(pulse * 5)
            ry = radius_y + (i * 2) + int(pulse * 2)
            frame = draw_ring_with_cutout(frame, (center_x, feet_y), (rx, ry), color, 2, alpha)

        # Draw Main Ring Body
        frame = draw_ring_with_cutout(frame, (center_x, feet_y), (radius_x, radius_y), color, 3, 0.7)

        return frame

    @staticmethod
    def get_spotlight_mask(frame_shape, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Generates a strong 'Alien Beam' mask.
        Uses uint8 for drawing safety, then converts to float.
        """
        h, w = frame_shape[:2]
        # 1. Use uint8 for robust drawing
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
        
        x, y, bw, bh = bbox
        center_x = int(x + bw // 2)
        feet_y = int(y + bh)
        
        # Beam Geometry
        beam_top_width = max(int(bw * 3.5), 250)
        beam_bottom_width = max(int(bw * 1.2), 80)
        
        # Points for the trapezoid beam
        pts = np.array([
            [center_x - beam_top_width // 2, -100],      # Top Left (Offscreen)
            [center_x + beam_top_width // 2, -100],      # Top Right (Offscreen)
            [center_x + beam_bottom_width // 2, feet_y], # Bottom Right
            [center_x - beam_bottom_width // 2, feet_y]  # Bottom Left
        ], np.int32)
        
        # Draw solid white beam (255)
        cv2.fillConvexPoly(mask_u8, pts, 255)
        
        # Draw floor pool (ellipse)
        cv2.ellipse(mask_u8, (center_x, feet_y), 
                   (int(beam_bottom_width//1.5), int(beam_bottom_width//4)), 
                   0, 0, 360, 255, -1)
        
        # Blur
        blur_size = (75, 75) 
        mask_blurred = cv2.GaussianBlur(mask_u8, blur_size, 0)
        
        # Normalize to float 0.0 - 1.0
        return mask_blurred.astype(np.float32) / 255.0

    @staticmethod
    def apply_spotlight_mask(original_frame: np.ndarray, darkened_frame: np.ndarray, combined_mask: np.ndarray) -> np.ndarray:
        """Blends original and dark frames"""
        # Ensure mask is 3-channel
        if len(combined_mask.shape) == 2:
            combined_mask = cv2.merge([combined_mask, combined_mask, combined_mask])
            
        orig_f = original_frame.astype(np.float32)
        dark_f = darkened_frame.astype(np.float32)
        
        # Result = Light Areas (Original) + Dark Areas (Darkened)
        result = orig_f * combined_mask + dark_f * (1.0 - combined_mask)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def draw_spotlight_floor_circle(frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                                   color: Tuple[int, int, int]) -> np.ndarray:
        """Draws the bright ring at the base"""
        x, y, w, h = bbox
        center_x = x + w // 2
        feet_y = y + h
        
        rx = max(int(w * 0.6), 30)
        ry = max(int(w * 0.2), 10)
        
        # Stronger Outline
        cv2.ellipse(frame, (center_x, feet_y), (rx, ry), 0, 0, 360, color, 2, cv2.LINE_AA)

        return frame

    # =========================================================================
    # NEW MARKERS
    # =========================================================================

    @staticmethod
    def get_alien_spotlight_mask(frame_shape, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Generates mask for Alien Spotlight - narrow at top (ceiling light), wide at floor.
        Returns float mask 0.0-1.0 where 1.0 = lit area.
        """
        h, w = frame_shape[:2]
        mask_u8 = np.zeros((h, w), dtype=np.uint8)

        x, y, bw, bh = bbox
        center_x = int(x + bw // 2)
        feet_y = int(y + bh)

        # Beam geometry: NARROW at top (ceiling source), WIDE at floor
        beam_top_width = max(int(bw * 0.3), 20)  # Narrow at ceiling
        beam_bottom_width = max(int(bw * 1.8), 120)  # Wide at floor (matches floor ellipse)

        # Floor ellipse dimensions
        floor_rx = max(int(bw * 0.9), 60)
        floor_ry = max(int(bw * 0.28), 18)

        # Trapezoid points - narrow top, wide bottom
        pts = np.array([
            [center_x - beam_top_width // 2, 0],            # Top Left (narrow)
            [center_x + beam_top_width // 2, 0],            # Top Right (narrow)
            [center_x + beam_bottom_width // 2, feet_y],    # Bottom Right (wide)
            [center_x - beam_bottom_width // 2, feet_y]     # Bottom Left (wide)
        ], np.int32)

        # Draw beam
        cv2.fillConvexPoly(mask_u8, pts, 255)

        # Draw floor ellipse (extends the lit area on floor)
        cv2.ellipse(mask_u8, (center_x, feet_y),
                   (floor_rx, floor_ry),
                   0, 0, 360, 255, -1)

        # Blur for soft edges
        mask_blurred = cv2.GaussianBlur(mask_u8, (51, 51), 0)

        return mask_blurred.astype(np.float32) / 255.0

    @staticmethod
    def apply_alien_spotlight(original_frame: np.ndarray, darkened_frame: np.ndarray,
                              combined_mask: np.ndarray) -> np.ndarray:
        """Blends original (lit) and darkened frames using mask."""
        if len(combined_mask.shape) == 2:
            combined_mask = cv2.merge([combined_mask, combined_mask, combined_mask])

        orig_f = original_frame.astype(np.float32)
        dark_f = darkened_frame.astype(np.float32)

        # Lit areas = original, dark areas = darkened
        result = orig_f * combined_mask + dark_f * (1.0 - combined_mask)

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def draw_alien_spotlight_floor(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                                   color: Tuple[int, int, int], frame_count: int = 0) -> np.ndarray:
        """
        Draws a subtle light pool on the floor - like light falling from above.
        Very transparent so the floor is clearly visible underneath.
        """
        x, y, w, h = bbox
        center_x = x + w // 2
        feet_y = y + h

        # Pulsing animation
        pulse = 0.9 + 0.1 * math.sin(frame_count * 0.1)

        # Floor ellipse dimensions
        floor_rx = max(int(w * 1.0), 70)
        floor_ry = max(int(w * 0.32), 22)

        # Single very subtle light layer - additive style (brightens floor)
        overlay = frame.copy()
        # Use white/bright color for natural light appearance
        light_color = (255, 255, 240)  # Warm white
        cv2.ellipse(overlay, (center_x, feet_y), (floor_rx, floor_ry),
                   0, 0, 360, light_color, -1, cv2.LINE_AA)
        # Very low alpha - just a hint of brightness on the floor
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

        return frame

    @staticmethod
    def draw_solid_anchor(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                          color: Tuple[int, int, int] = (0, 255, 100),  # Neon Green (BGR)
                          player=None) -> np.ndarray:
        """
        Solid Floor Anchor - A glowing filled ellipse on the floor beneath the player.
        The area under the player (trapezoid shape) has semi-transparent fill so feet are visible.
        Outline is NOT drawn in the trapezoid area to avoid cutting the feet.
        """
        x, y, w, h = bbox
        center_x = x + w // 2

        # Calculate feet level
        feet_y = y + h
        if player and hasattr(player, 'current_original_bbox') and player.current_original_bbox:
            _, orig_y, _, orig_h = player.current_original_bbox
            feet_y = orig_y + orig_h

        # Ellipse dimensions (perspective-correct flattening)
        radius_x = max(int(w * 0.75), 45)
        radius_y = max(int(w * 0.25), 16)

        # Colors
        fill_color = (0, 255, 100)  # Neon green BGR
        outline_color = (180, 255, 180)  # Light green outline

        # Create mask for the full ellipse area
        ellipse_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.ellipse(ellipse_mask, (center_x, feet_y), (radius_x, radius_y),
                   0, 0, 360, 255, -1, cv2.LINE_AA)

        # Define the "under player" trapezoid area
        # Narrow at bottom (feet), wider at top (towards body)
        feet_width = int(w * 0.7)
        body_width = int(w * 1.2)
        cut_height = radius_y + 8

        trapezoid_pts = np.array([
            [center_x - feet_width // 2, feet_y],
            [center_x + feet_width // 2, feet_y],
            [center_x + body_width // 2, feet_y - cut_height],
            [center_x - body_width // 2, feet_y - cut_height]
        ], np.int32)

        # Create trapezoid mask (clipped to ellipse)
        trapezoid_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(trapezoid_mask, trapezoid_pts, 255)
        trapezoid_mask = cv2.bitwise_and(trapezoid_mask, ellipse_mask)

        # Create mask for area outside trapezoid (within ellipse)
        outside_trapezoid_mask = cv2.bitwise_and(ellipse_mask, cv2.bitwise_not(trapezoid_mask))

        # === STEP 1: Draw solid fill OUTSIDE trapezoid (60% opacity) ===
        overlay = frame.copy()
        cv2.ellipse(overlay, (center_x, feet_y), (radius_x, radius_y),
                   0, 0, 360, fill_color, -1, cv2.LINE_AA)

        outside_mask_3ch = cv2.merge([outside_trapezoid_mask] * 3).astype(np.float32) / 255.0
        frame_f = frame.astype(np.float32)
        overlay_f = overlay.astype(np.float32)
        frame = np.clip(frame_f * (1.0 - outside_mask_3ch * 0.6) + overlay_f * (outside_mask_3ch * 0.6), 0, 255).astype(np.uint8)

        # === STEP 2: Draw semi-transparent fill INSIDE trapezoid (35% opacity - visible but see-through) ===
        overlay_inner = frame.copy()
        cv2.fillConvexPoly(overlay_inner, trapezoid_pts, fill_color)
        
        trapezoid_mask_3ch = cv2.merge([trapezoid_mask] * 3).astype(np.float32) / 255.0
        frame_f = frame.astype(np.float32)
        overlay_inner_f = overlay_inner.astype(np.float32)
        frame = np.clip(frame_f * (1.0 - trapezoid_mask_3ch * 0.35) + overlay_inner_f * (trapezoid_mask_3ch * 0.35), 0, 255).astype(np.uint8)

        # === STEP 3: Draw outline ONLY on FRONT arc (bottom half) - not where legs are ===
        # The back/top of the ellipse (180-360 degrees) is hidden because the player stands there
        overlay_outline = frame.copy()
        
        # Draw only the FRONT arc (0 to 180 degrees = bottom half of ellipse)
        cv2.ellipse(overlay_outline, (center_x, feet_y), (radius_x, radius_y),
                   0, 0, 180, outline_color, 2, cv2.LINE_AA)

        # Create outline mask for front arc only
        outline_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.ellipse(outline_mask, (center_x, feet_y), (radius_x, radius_y),
                   0, 0, 180, 255, 4, cv2.LINE_AA)

        # Also exclude the trapezoid area from the outline (for the corners where it might overlap)
        outline_front_only = cv2.bitwise_and(outline_mask, cv2.bitwise_not(trapezoid_mask))

        # Blend outline (80% opacity, only front arc outside trapezoid)
        outline_mask_3ch = cv2.merge([outline_front_only] * 3).astype(np.float32) / 255.0
        frame_f = frame.astype(np.float32)
        outline_f = overlay_outline.astype(np.float32)
        frame = np.clip(frame_f * (1.0 - outline_mask_3ch * 0.8) + outline_f * (outline_mask_3ch * 0.8), 0, 255).astype(np.uint8)

        return frame

    @staticmethod
    def draw_defensive_radar(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                             color: Tuple[int, int, int] = (0, 0, 255),  # Red (BGR)
                             player=None, target_position: Tuple[int, int] = None,
                             manual_angle: float = None, manual_size: float = None,
                             frame_count: int = 0) -> np.ndarray:
        """
        Defensive Radar - A radar sweep cone with scan lines.

        Args:
            frame: The video frame
            bbox: Player bounding box (x, y, w, h)
            color: Radar color (BGR)
            player: Player object with tracking data
            target_position: (x, y) position of the opponent to face toward.
                           If None, defaults to UP direction. Ignored if manual_angle is set.
            manual_angle: Manual direction angle in radians (overrides target_position)
            manual_size: Manual size multiplier for cone length (1.0 = default)
            frame_count: Current frame number for animation
        """
        x, y, w, h = bbox

        # Get bbox values
        feet_y = y + h
        if player and hasattr(player, 'current_original_bbox') and player.current_original_bbox:
            ox, oy, ow, oh = player.current_original_bbox
            feet_y = oy + oh
            x, w = ox, ow
            h = oh  # Use original height for size calculation

        # Radar colors - use the passed color parameter
        # Default green: (0, 255, 100), Red: (0, 50, 255)
        radar_color_main = color
        # Create darker and glow variants based on main color
        b, g, r = color
        radar_color_dark = (max(0, b - 40), max(0, g - 75), max(0, r - 40))
        radar_color_glow = (min(255, b + 50), min(255, g + 50), min(255, r + 50))

        # Player center (at feet level)
        center_x = x + w // 2
        center_y = feet_y

        # Cone parameters - larger default size
        base_cone_length = int(h * 1.2)  # Larger radar
        size_multiplier = manual_size if manual_size is not None else 1.0
        cone_length = int(base_cone_length * size_multiplier)
        cone_half_angle = 30  # degrees - slightly narrower for radar look

        # Calculate direction angle
        if manual_angle is not None:
            direction_angle = manual_angle
        elif target_position is not None:
            target_x, target_y = target_position
            dx = target_x - center_x
            dy = target_y - center_y
            direction_angle = math.atan2(dy, dx)
        else:
            direction_angle = -math.pi / 2  # Default: up

        # Calculate cone edge angles
        left_angle = direction_angle - math.radians(cone_half_angle)
        right_angle = direction_angle + math.radians(cone_half_angle)

        # Origin point (center of feet)
        origin = (center_x, center_y)

        # End points of cone
        end_left = (
            int(center_x + cone_length * math.cos(left_angle)),
            int(center_y + cone_length * math.sin(left_angle))
        )
        end_right = (
            int(center_x + cone_length * math.cos(right_angle)),
            int(center_y + cone_length * math.sin(right_angle))
        )

        # Create triangle points
        pts = np.array([origin, end_left, end_right], np.int32)

        # Draw gradient-like effect with multiple layers
        overlay = frame.copy()

        # Outer glow (very transparent)
        glow_pts = np.array([
            origin,
            (int(center_x + (cone_length + 10) * math.cos(left_angle - 0.05)),
             int(center_y + (cone_length + 10) * math.sin(left_angle - 0.05))),
            (int(center_x + (cone_length + 10) * math.cos(right_angle + 0.05)),
             int(center_y + (cone_length + 10) * math.sin(right_angle + 0.05)))
        ], np.int32)
        cv2.fillPoly(overlay, [glow_pts], radar_color_dark)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

        # Main cone fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], radar_color_main)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        # Draw radar arc lines (concentric arcs)
        num_arcs = 4
        for i in range(1, num_arcs + 1):
            arc_radius = int(cone_length * i / num_arcs)
            # Draw arc between left and right angles
            start_angle_deg = int(math.degrees(left_angle))
            end_angle_deg = int(math.degrees(right_angle))
            cv2.ellipse(frame, origin, (arc_radius, arc_radius),
                       0, start_angle_deg, end_angle_deg,
                       radar_color_dark, 1, cv2.LINE_AA)

        # Draw radial lines
        num_radials = 5
        for i in range(num_radials + 1):
            t = i / num_radials
            line_angle = left_angle + t * (right_angle - left_angle)
            end_x = int(center_x + cone_length * math.cos(line_angle))
            end_y = int(center_y + cone_length * math.sin(line_angle))
            cv2.line(frame, origin, (end_x, end_y), radar_color_dark, 1, cv2.LINE_AA)

        # Draw sweep line (animated)
        sweep_speed = 0.1  # radians per frame
        sweep_angle = direction_angle + math.sin(frame_count * sweep_speed) * math.radians(cone_half_angle * 0.8)
        sweep_end = (
            int(center_x + cone_length * math.cos(sweep_angle)),
            int(center_y + cone_length * math.sin(sweep_angle))
        )
        cv2.line(frame, origin, sweep_end, radar_color_glow, 2, cv2.LINE_AA)

        # Draw cone outline
        cv2.polylines(frame, [pts], isClosed=True, color=radar_color_main, thickness=2, lineType=cv2.LINE_AA)

        # Draw origin point with glow
        cv2.circle(frame, origin, 8, radar_color_dark, -1, cv2.LINE_AA)
        cv2.circle(frame, origin, 5, radar_color_glow, -1, cv2.LINE_AA)
        cv2.circle(frame, origin, 3, (255, 255, 255), -1, cv2.LINE_AA)

        return frame

    @staticmethod
    def draw_sniper_scope(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                          color: Tuple[int, int, int] = (0, 0, 255),  # Red (BGR)
                          player=None, frame_count: int = 0) -> np.ndarray:
        """
        Sniper Scope - A large crosshair reticle around the player.
        The scope is big and doesn't obscure the player.
        """
        x, y, w, h = bbox

        # Use original bbox if available
        if player and hasattr(player, 'current_original_bbox') and player.current_original_bbox:
            x, y, w, h = player.current_original_bbox

        # Center of player
        center_x = x + w // 2
        center_y = y + h // 2

        # Scope radius - larger than the player
        outer_radius = max(int(max(w, h) * 0.85), 80)
        inner_radius = max(int(max(w, h) * 0.35), 30)

        # Colors
        scope_color = (0, 0, 255)  # Red
        glow_color = (0, 0, 180)   # Darker red for glow

        # Subtle pulse animation
        pulse = 1.0 + 0.05 * math.sin(frame_count * 0.1)
        outer_r = int(outer_radius * pulse)

        # Draw outer glow circle
        overlay = frame.copy()
        cv2.circle(overlay, (center_x, center_y), outer_r + 4, glow_color, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Draw main outer circle
        cv2.circle(frame, (center_x, center_y), outer_r, scope_color, 2, cv2.LINE_AA)

        # Gap size (where the crosshair lines don't draw - to not cover player)
        # Large gap to give player freedom of movement without being obscured
        gap = int(max(w, h) * 0.7)

        # Draw crosshair lines (4 directions, with gap in middle)
        line_thickness = 2

        # Top line
        cv2.line(frame, (center_x, center_y - outer_r - 15),
                (center_x, center_y - gap), scope_color, line_thickness, cv2.LINE_AA)
        # Bottom line
        cv2.line(frame, (center_x, center_y + gap),
                (center_x, center_y + outer_r + 15), scope_color, line_thickness, cv2.LINE_AA)
        # Left line
        cv2.line(frame, (center_x - outer_r - 15, center_y),
                (center_x - gap, center_y), scope_color, line_thickness, cv2.LINE_AA)
        # Right line
        cv2.line(frame, (center_x + gap, center_y),
                (center_x + outer_r + 15, center_y), scope_color, line_thickness, cv2.LINE_AA)

        # Draw small tick marks on outer circle (at 45 degree angles)
        tick_length = 12
        for angle_deg in [45, 135, 225, 315]:
            angle_rad = math.radians(angle_deg)
            # Outer point
            ox = int(center_x + (outer_r + tick_length) * math.cos(angle_rad))
            oy = int(center_y + (outer_r + tick_length) * math.sin(angle_rad))
            # Inner point (on circle)
            ix = int(center_x + outer_r * math.cos(angle_rad))
            iy = int(center_y + outer_r * math.sin(angle_rad))
            cv2.line(frame, (ix, iy), (ox, oy), scope_color, line_thickness, cv2.LINE_AA)

        return frame
