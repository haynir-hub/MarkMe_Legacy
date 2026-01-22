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
        h, w = frame_shape[:2]
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
        
        x, y, bw, bh = bbox
        center_x = int(x + bw // 2)
        feet_y = int(y + bh)
        
        beam_top_width = max(int(bw * 3.5), 250)
        beam_bottom_width = max(int(bw * 1.2), 80)
        
        pts = np.array([
            [center_x - beam_top_width // 2, -100],
            [center_x + beam_top_width // 2, -100],
            [center_x + beam_bottom_width // 2, feet_y],
            [center_x - beam_bottom_width // 2, feet_y]
        ], np.int32)
        
        cv2.fillConvexPoly(mask_u8, pts, 255)
        cv2.ellipse(mask_u8, (center_x, feet_y), 
                   (int(beam_bottom_width//1.5), int(beam_bottom_width//4)), 
                   0, 0, 360, 255, -1)
        
        blur_size = (75, 75) 
        mask_blurred = cv2.GaussianBlur(mask_u8, blur_size, 0)
        return mask_blurred.astype(np.float32) / 255.0

    @staticmethod
    def apply_spotlight_mask(original_frame: np.ndarray, darkened_frame: np.ndarray, combined_mask: np.ndarray) -> np.ndarray:
        if len(combined_mask.shape) == 2:
            combined_mask = cv2.merge([combined_mask, combined_mask, combined_mask])
        orig_f = original_frame.astype(np.float32)
        dark_f = darkened_frame.astype(np.float32)
        result = orig_f * combined_mask + dark_f * (1.0 - combined_mask)
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def draw_spotlight_floor_circle(frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                                   color: Tuple[int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        center_x = x + w // 2
        feet_y = y + h
        rx = max(int(w * 0.6), 30)
        ry = max(int(w * 0.2), 10)
        cv2.ellipse(frame, (center_x, feet_y), (rx, ry), 0, 0, 360, color, 2, cv2.LINE_AA)
        return frame

    @staticmethod
    def get_alien_spotlight_mask(frame_shape, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        h, w = frame_shape[:2]
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
        x, y, bw, bh = bbox
        center_x = int(x + bw // 2)
        feet_y = int(y + bh)
        beam_top_width = max(int(bw * 0.3), 20)
        beam_bottom_width = max(int(bw * 1.8), 120)
        floor_rx = max(int(bw * 0.9), 60)
        floor_ry = max(int(bw * 0.28), 18)
        pts = np.array([
            [center_x - beam_top_width // 2, 0],
            [center_x + beam_top_width // 2, 0],
            [center_x + beam_bottom_width // 2, feet_y],
            [center_x - beam_bottom_width // 2, feet_y]
        ], np.int32)
        cv2.fillConvexPoly(mask_u8, pts, 255)
        cv2.ellipse(mask_u8, (center_x, feet_y), (floor_rx, floor_ry), 0, 0, 360, 255, -1)
        mask_blurred = cv2.GaussianBlur(mask_u8, (51, 51), 0)
        return mask_blurred.astype(np.float32) / 255.0

    @staticmethod
    def apply_alien_spotlight(original_frame: np.ndarray, darkened_frame: np.ndarray,
                              combined_mask: np.ndarray) -> np.ndarray:
        if len(combined_mask.shape) == 2:
            combined_mask = cv2.merge([combined_mask, combined_mask, combined_mask])
        orig_f = original_frame.astype(np.float32)
        dark_f = darkened_frame.astype(np.float32)
        result = orig_f * combined_mask + dark_f * (1.0 - combined_mask)
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def draw_alien_spotlight_floor(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                                   color: Tuple[int, int, int], frame_count: int = 0) -> np.ndarray:
        x, y, w, h = bbox
        center_x = x + w // 2
        feet_y = y + h
        pulse = 0.9 + 0.1 * math.sin(frame_count * 0.1)
        floor_rx = max(int(w * 1.0), 70)
        floor_ry = max(int(w * 0.32), 22)
        overlay = frame.copy()
        light_color = (255, 255, 240)
        cv2.ellipse(overlay, (center_x, feet_y), (floor_rx, floor_ry), 0, 0, 360, light_color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
        return frame

    @staticmethod
    def draw_solid_anchor(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                          color: Tuple[int, int, int] = (0, 255, 100), player=None) -> np.ndarray:
        x, y, w, h = bbox
        center_x = x + w // 2
        feet_y = y + h
        if player and hasattr(player, 'current_original_bbox') and player.current_original_bbox:
            _, orig_y, _, orig_h = player.current_original_bbox
            feet_y = orig_y + orig_h
        radius_x = max(int(w * 0.75), 45)
        radius_y = max(int(w * 0.25), 16)
        fill_color = (0, 255, 100)
        outline_color = (180, 255, 180)
        ellipse_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.ellipse(ellipse_mask, (center_x, feet_y), (radius_x, radius_y), 0, 0, 360, 255, -1, cv2.LINE_AA)
        feet_width = int(w * 0.7)
        body_width = int(w * 1.2)
        cut_height = radius_y + 8
        trapezoid_pts = np.array([
            [center_x - feet_width // 2, feet_y],
            [center_x + feet_width // 2, feet_y],
            [center_x + body_width // 2, feet_y - cut_height],
            [center_x - body_width // 2, feet_y - cut_height]
        ], np.int32)
        trapezoid_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(trapezoid_mask, trapezoid_pts, 255)
        trapezoid_mask = cv2.bitwise_and(trapezoid_mask, ellipse_mask)
        outside_trapezoid_mask = cv2.bitwise_and(ellipse_mask, cv2.bitwise_not(trapezoid_mask))
        overlay = frame.copy()
        cv2.ellipse(overlay, (center_x, feet_y), (radius_x, radius_y), 0, 0, 360, fill_color, -1, cv2.LINE_AA)
        outside_mask_3ch = cv2.merge([outside_trapezoid_mask] * 3).astype(np.float32) / 255.0
        frame_f = frame.astype(np.float32)
        overlay_f = overlay.astype(np.float32)
        frame = np.clip(frame_f * (1.0 - outside_mask_3ch * 0.6) + overlay_f * (outside_mask_3ch * 0.6), 0, 255).astype(np.uint8)
        overlay_inner = frame.copy()
        cv2.fillConvexPoly(overlay_inner, trapezoid_pts, fill_color)
        trapezoid_mask_3ch = cv2.merge([trapezoid_mask] * 3).astype(np.float32) / 255.0
        frame_f = frame.astype(np.float32)
        overlay_inner_f = overlay_inner.astype(np.float32)
        frame = np.clip(frame_f * (1.0 - trapezoid_mask_3ch * 0.35) + overlay_inner_f * (trapezoid_mask_3ch * 0.35), 0, 255).astype(np.uint8)
        overlay_outline = frame.copy()
        cv2.ellipse(overlay_outline, (center_x, feet_y), (radius_x, radius_y), 0, 0, 180, outline_color, 2, cv2.LINE_AA)
        outline_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.ellipse(outline_mask, (center_x, feet_y), (radius_x, radius_y), 0, 0, 180, 255, 4, cv2.LINE_AA)
        outline_front_only = cv2.bitwise_and(outline_mask, cv2.bitwise_not(trapezoid_mask))
        outline_mask_3ch = cv2.merge([outline_front_only] * 3).astype(np.float32) / 255.0
        frame_f = frame.astype(np.float32)
        outline_f = overlay_outline.astype(np.float32)
        frame = np.clip(frame_f * (1.0 - outline_mask_3ch * 0.8) + outline_f * (outline_mask_3ch * 0.8), 0, 255).astype(np.uint8)
        return frame

    @staticmethod
    def draw_defensive_radar(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                             color: Tuple[int, int, int] = (0, 0, 255), player=None, 
                             target_position: Tuple[int, int] = None,
                             manual_angle: float = None, manual_size: float = None,
                             frame_count: int = 0) -> np.ndarray:
        x, y, w, h = bbox
        feet_y = y + h
        if player and hasattr(player, 'current_original_bbox') and player.current_original_bbox:
            ox, oy, ow, oh = player.current_original_bbox
            feet_y = oy + oh
            x, w = ox, ow
            h = oh
        radar_color_main = color
        b, g, r = color
        radar_color_dark = (max(0, b - 40), max(0, g - 75), max(0, r - 40))
        radar_color_glow = (min(255, b + 50), min(255, g + 50), min(255, r + 50))
        center_x = x + w // 2
        center_y = feet_y
        base_cone_length = int(h * 1.2)
        size_multiplier = manual_size if manual_size is not None else 1.0
        cone_length = int(base_cone_length * size_multiplier)
        cone_half_angle = 30
        if manual_angle is not None:
            direction_angle = manual_angle
        elif target_position is not None:
            target_x, target_y = target_position
            dx = target_x - center_x
            dy = target_y - center_y
            direction_angle = math.atan2(dy, dx)
        else:
            direction_angle = -math.pi / 2
        left_angle = direction_angle - math.radians(cone_half_angle)
        right_angle = direction_angle + math.radians(cone_half_angle)
        origin = (center_x, center_y)
        end_left = (int(center_x + cone_length * math.cos(left_angle)), int(center_y + cone_length * math.sin(left_angle)))
        end_right = (int(center_x + cone_length * math.cos(right_angle)), int(center_y + cone_length * math.sin(right_angle)))
        pts = np.array([origin, end_left, end_right], np.int32)
        overlay = frame.copy()
        glow_pts = np.array([
            origin,
            (int(center_x + (cone_length + 10) * math.cos(left_angle - 0.05)), int(center_y + (cone_length + 10) * math.sin(left_angle - 0.05))),
            (int(center_x + (cone_length + 10) * math.cos(right_angle + 0.05)), int(center_y + (cone_length + 10) * math.sin(right_angle + 0.05)))
        ], np.int32)
        cv2.fillPoly(overlay, [glow_pts], radar_color_dark)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], radar_color_main)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        num_arcs = 4
        for i in range(1, num_arcs + 1):
            arc_radius = int(cone_length * i / num_arcs)
            start_angle_deg = int(math.degrees(left_angle))
            end_angle_deg = int(math.degrees(right_angle))
            cv2.ellipse(frame, origin, (arc_radius, arc_radius), 0, start_angle_deg, end_angle_deg, radar_color_dark, 1, cv2.LINE_AA)
        num_radials = 5
        for i in range(num_radials + 1):
            t = i / num_radials
            line_angle = left_angle + t * (right_angle - left_angle)
            end_x = int(center_x + cone_length * math.cos(line_angle))
            end_y = int(center_y + cone_length * math.sin(line_angle))
            cv2.line(frame, origin, (end_x, end_y), radar_color_dark, 1, cv2.LINE_AA)
        sweep_speed = 0.1
        sweep_angle = direction_angle + math.sin(frame_count * sweep_speed) * math.radians(cone_half_angle * 0.8)
        sweep_end = (int(center_x + cone_length * math.cos(sweep_angle)), int(center_y + cone_length * math.sin(sweep_angle)))
        cv2.line(frame, origin, sweep_end, radar_color_glow, 2, cv2.LINE_AA)
        cv2.polylines(frame, [pts], isClosed=True, color=radar_color_main, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(frame, origin, 8, radar_color_dark, -1, cv2.LINE_AA)
        cv2.circle(frame, origin, 5, radar_color_glow, -1, cv2.LINE_AA)
        cv2.circle(frame, origin, 3, (255, 255, 255), -1, cv2.LINE_AA)
        return frame

    @staticmethod
    def draw_sniper_scope(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                          color: Tuple[int, int, int] = (0, 0, 255), player=None, frame_count: int = 0) -> np.ndarray:
        x, y, w, h = bbox
        if player and hasattr(player, 'current_original_bbox') and player.current_original_bbox:
            x, y, w, h = player.current_original_bbox
        center_x = x + w // 2
        center_y = y + h // 2
        outer_radius = max(int(max(w, h) * 0.85), 80)
        scope_color = (0, 0, 255)
        glow_color = (0, 0, 180)
        pulse = 1.0 + 0.05 * math.sin(frame_count * 0.1)
        outer_r = int(outer_radius * pulse)
        overlay = frame.copy()
        cv2.circle(overlay, (center_x, center_y), outer_r + 4, glow_color, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.circle(frame, (center_x, center_y), outer_r, scope_color, 2, cv2.LINE_AA)
        gap = int(max(w, h) * 0.7)
        line_thickness = 2
        cv2.line(frame, (center_x, center_y - outer_r - 15), (center_x, center_y - gap), scope_color, line_thickness, cv2.LINE_AA)
        cv2.line(frame, (center_x, center_y + gap), (center_x, center_y + outer_r + 15), scope_color, line_thickness, cv2.LINE_AA)
        cv2.line(frame, (center_x - outer_r - 15, center_y), (center_x - gap, center_y), scope_color, line_thickness, cv2.LINE_AA)
        cv2.line(frame, (center_x + gap, center_y), (center_x + outer_r + 15, center_y), scope_color, line_thickness, cv2.LINE_AA)
        tick_length = 12
        for angle_deg in [45, 135, 225, 315]:
            angle_rad = math.radians(angle_deg)
            ox = int(center_x + (outer_r + tick_length) * math.cos(angle_rad))
            oy = int(center_y + (outer_r + tick_length) * math.sin(angle_rad))
            ix = int(center_x + outer_r * math.cos(angle_rad))
            iy = int(center_y + outer_r * math.sin(angle_rad))
            cv2.line(frame, (ix, iy), (ox, oy), scope_color, line_thickness, cv2.LINE_AA)
        return frame

    # =========================================================================
    # BALL MARKERS - Transparent to show the ball with effects AROUND it
    # =========================================================================

    @staticmethod
    def draw_ball_marker(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                         color: Tuple[int, int, int] = (0, 165, 255), player=None, frame_count: int = 0) -> np.ndarray:
        """Ball Marker (Glowing) - A glowing ring AROUND the ball, keeping the ball visible."""
        x, y, w, h = bbox
        if player and hasattr(player, 'current_original_bbox') and player.current_original_bbox:
            x, y, w, h = player.current_original_bbox
        center_x = x + w // 2
        center_y = y + h // 2
        ball_radius = max(int(min(w, h) * 0.5), 10)
        ring_radius = ball_radius + 10
        pulse = 0.9 + 0.1 * math.sin(frame_count * 0.2)
        animated_ring_radius = int(ring_radius * pulse)
        b, g, r = color
        glow_color_outer = (max(0, b - 50), max(0, g - 50), max(0, r - 50))
        glow_color_bright = (min(255, b + 80), min(255, g + 80), min(255, r + 80))
        overlay = frame.copy()
        for i in range(4, 0, -1):
            glow_ring_radius = animated_ring_radius + i * 5
            thickness = 3 + i
            cv2.circle(overlay, (center_x, center_y), glow_ring_radius, glow_color_outer, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        overlay = frame.copy()
        cv2.circle(overlay, (center_x, center_y), animated_ring_radius, color, 4, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.circle(frame, (center_x, center_y), animated_ring_radius - 2, glow_color_bright, 2, cv2.LINE_AA)
        cv2.circle(frame, (center_x, center_y), animated_ring_radius + 3, glow_color_outer, 1, cv2.LINE_AA)
        sparkle_radius = animated_ring_radius + 8
        for angle_deg in [0, 90, 180, 270]:
            angle_rad = math.radians(angle_deg + frame_count * 2)
            sparkle_x = int(center_x + sparkle_radius * math.cos(angle_rad))
            sparkle_y = int(center_y + sparkle_radius * math.sin(angle_rad))
            cv2.circle(frame, (sparkle_x, sparkle_y), 3, glow_color_bright, -1, cv2.LINE_AA)
            cv2.circle(frame, (sparkle_x, sparkle_y), 2, (255, 255, 255), -1, cv2.LINE_AA)
        return frame

    @staticmethod
    def draw_fireball_trail(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                            color: Tuple[int, int, int] = (0, 100, 255), player=None, frame_count: int = 0) -> np.ndarray:
        """Fireball Trail - TRANSPARENT ball with flame trail AROUND/behind it."""
        x, y, w, h = bbox
        if player and hasattr(player, 'current_original_bbox') and player.current_original_bbox:
            x, y, w, h = player.current_original_bbox
        center_x = x + w // 2
        center_y = y + h // 2
        ball_radius = max(int(min(w, h) * 0.5), 12)
        fire_core = (0, 255, 255)
        fire_inner = (0, 165, 255)
        fire_outer = (0, 50, 255)
        fire_edge = (0, 0, 200)
        flicker = 0.8 + 0.2 * math.sin(frame_count * 0.4)
        trail_start_offset = ball_radius + 3
        trail_length = int(ball_radius * 4 * flicker)
        trail_width = int(ball_radius * 1.8)
        flame_pts = []
        num_segments = 10
        for i in range(num_segments + 1):
            t = i / num_segments
            segment_width = trail_width * (1 - t * 0.8)
            segment_x = center_x - trail_start_offset - int(trail_length * t)
            wave = math.sin(t * math.pi * 2 + frame_count * 0.3) * segment_width * 0.3
            flame_pts.append((segment_x, int(center_y - segment_width / 2 + wave)))
        for i in range(num_segments, -1, -1):
            t = i / num_segments
            segment_width = trail_width * (1 - t * 0.8)
            segment_x = center_x - trail_start_offset - int(trail_length * t)
            wave = math.sin(t * math.pi * 2 + frame_count * 0.3 + math.pi) * segment_width * 0.3
            flame_pts.append((segment_x, int(center_y + segment_width / 2 + wave)))
        flame_pts = np.array(flame_pts, np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [flame_pts], fire_edge)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        inner_scale = 0.7
        inner_pts = []
        for i in range(num_segments + 1):
            t = i / num_segments
            segment_width = trail_width * inner_scale * (1 - t * 0.85)
            segment_x = center_x - trail_start_offset - int(trail_length * inner_scale * t)
            wave = math.sin(t * math.pi * 2 + frame_count * 0.35) * segment_width * 0.25
            inner_pts.append((segment_x, int(center_y - segment_width / 2 + wave)))
        for i in range(num_segments, -1, -1):
            t = i / num_segments
            segment_width = trail_width * inner_scale * (1 - t * 0.85)
            segment_x = center_x - trail_start_offset - int(trail_length * inner_scale * t)
            wave = math.sin(t * math.pi * 2 + frame_count * 0.35 + math.pi) * segment_width * 0.25
            inner_pts.append((segment_x, int(center_y + segment_width / 2 + wave)))
        inner_pts = np.array(inner_pts, np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [inner_pts], fire_outer)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        ring_radius = ball_radius + 6
        overlay = frame.copy()
        cv2.circle(overlay, (center_x, center_y), ring_radius + 5, fire_edge, 4, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.circle(frame, (center_x, center_y), ring_radius, fire_outer, 3, cv2.LINE_AA)
        cv2.circle(frame, (center_x, center_y), ring_radius - 2, fire_inner, 2, cv2.LINE_AA)
        cv2.circle(frame, (center_x, center_y), ring_radius - 4, fire_core, 1, cv2.LINE_AA)
        num_particles = 6
        for i in range(num_particles):
            angle = (frame_count * 0.15 + i * math.pi * 2 / num_particles) % (math.pi * 2)
            particle_dist = ring_radius + 8 + math.sin(frame_count * 0.3 + i) * 5
            px = int(center_x + particle_dist * math.cos(angle))
            py = int(center_y + particle_dist * math.sin(angle))
            cv2.circle(frame, (px, py), 3, fire_inner, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 2, fire_core, -1, cv2.LINE_AA)
        return frame

    @staticmethod
    def draw_energy_rings(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                          color: Tuple[int, int, int] = (255, 200, 0), player=None, frame_count: int = 0) -> np.ndarray:
        """Energy Rings - Multiple rotating rings AROUND the ball. Ball stays visible."""
        x, y, w, h = bbox
        if player and hasattr(player, 'current_original_bbox') and player.current_original_bbox:
            x, y, w, h = player.current_original_bbox
        center_x = x + w // 2
        center_y = y + h // 2
        ball_radius = max(int(min(w, h) * 0.5), 10)
        ring_color_1 = (255, 255, 0)
        ring_color_2 = (255, 150, 0)
        ring_color_3 = (200, 100, 255)
        angle1 = frame_count * 0.08
        angle2 = frame_count * 0.06 + math.pi / 3
        angle3 = frame_count * 0.1 + math.pi * 2 / 3
        ring_radius = ball_radius + 15
        ring_thickness = 2
        def draw_tilted_ring(img, cx, cy, radius, tilt_angle, rotation_angle, color, thickness):
            num_points = 60
            points = []
            for i in range(num_points):
                theta = 2 * math.pi * i / num_points + rotation_angle
                px = radius * math.cos(theta)
                py = radius * math.sin(theta) * math.cos(tilt_angle)
                rot_x = px * math.cos(rotation_angle * 0.5) - py * math.sin(rotation_angle * 0.5)
                rot_y = px * math.sin(rotation_angle * 0.5) + py * math.cos(rotation_angle * 0.5)
                points.append((int(cx + rot_x), int(cy + rot_y)))
            points = np.array(points, np.int32)
            cv2.polylines(img, [points], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        overlay = frame.copy()
        draw_tilted_ring(overlay, center_x, center_y, ring_radius, math.pi/6, angle1, ring_color_1, ring_thickness + 2)
        draw_tilted_ring(overlay, center_x, center_y, ring_radius + 3, math.pi/3, angle2, ring_color_2, ring_thickness + 2)
        draw_tilted_ring(overlay, center_x, center_y, ring_radius + 6, math.pi/2.5, angle3, ring_color_3, ring_thickness + 2)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        draw_tilted_ring(frame, center_x, center_y, ring_radius, math.pi/6, angle1, ring_color_1, ring_thickness)
        draw_tilted_ring(frame, center_x, center_y, ring_radius + 3, math.pi/3, angle2, ring_color_2, ring_thickness)
        draw_tilted_ring(frame, center_x, center_y, ring_radius + 6, math.pi/2.5, angle3, ring_color_3, ring_thickness)
        particle_radius = 3
        for ring_angle, r_offset, p_color in [(angle1, 0, ring_color_1), (angle2, 3, ring_color_2), (angle3, 6, ring_color_3)]:
            for particle_pos in [0, math.pi]:
                theta = ring_angle + particle_pos
                px = int(center_x + (ring_radius + r_offset) * math.cos(theta))
                py = int(center_y + (ring_radius + r_offset) * math.sin(theta) * 0.5)
                cv2.circle(frame, (px, py), particle_radius + 1, p_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), particle_radius - 1, (255, 255, 255), -1, cv2.LINE_AA)
        highlight_ring_radius = ball_radius + 4
        cv2.circle(frame, (center_x, center_y), highlight_ring_radius, ring_color_1, 1, cv2.LINE_AA)
        return frame
