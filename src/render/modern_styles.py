"""
Modern Marker Styles - Beautiful, eye-catching marker styles for sports videos
"""
import cv2
import numpy as np
from typing import Tuple, Optional
import math


class ModernStyles:
    """Collection of modern, professional marker styles"""
    
    @staticmethod
    def draw_neon_ring(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                       color: Tuple[int, int, int] = (0, 255, 255), player=None) -> np.ndarray:
        """
        Neon glowing ring - modern style around feet with 3D layering effect
        Player body hides the back part of the ring (180-360 degrees)

        Args:
            frame: Frame to draw on
            bbox: Bounding box (padded)
            color: Ring color (BGR)
            player: Player object (for accessing original_bbox)

        Returns:
            Frame with neon ring
        """
        x, y, w, h = bbox
        center_x = x + w // 2

        # Calculate radii
        radius_x = max(int(w * 0.7), 40)  # Horizontal radius
        radius_y = max(int(w * 0.15), 10)  # Vertical radius (flat ellipse)

        # Position ring on floor at feet contact point
        # Use original_bbox if available (before padding was added)
        if hasattr(player, 'current_original_bbox') and player and player.current_original_bbox:
            orig_x, orig_y, orig_w, orig_h = player.current_original_bbox
            # Feet are at the BOTTOM of the original bbox (where the actual feet are)
            feet_level = orig_y + orig_h
        else:
            # Fallback: use mathematical calculation based on padding
            padding_factor = 0.20
            total_padding_multiplier = 1 + (2 * padding_factor)  # 1.4
            feet_ratio = (1 + padding_factor) / total_padding_multiplier  # 0.857
            feet_level = y + int(h * feet_ratio)

        center_y = feet_level - radius_y
        
        # Draw full neon ring (360 degrees) - MORE TRANSPARENT so it doesn't hide video
        # Reduced glow for less intrusion
        for i in range(6, 0, -1):
            overlay = frame.copy()
            glow_intensity = 0.25 - (i * 0.03)  # More transparent (was 0.5)
            cv2.ellipse(
                overlay,
                (center_x, center_y),
                (radius_x + i * 3, radius_y + i * 2),  # Smaller glow
                0, 0, 360,  # Full circle (360 degrees)
                color,
                max(1, int(3 * i / 6)),  # Thinner
                cv2.LINE_AA
            )
            cv2.addWeighted(overlay, glow_intensity, frame, 1.0 - glow_intensity, 0, frame)
        
        # Main ring (brightest) - MORE TRANSPARENT
        overlay = frame.copy()
        cv2.ellipse(
            overlay,
            (center_x, center_y),
            (radius_x, radius_y),
            0, 0, 360,  # Full circle
            color,
            3,  # Thinner (was 4)
            cv2.LINE_AA
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)  # 60% opacity
        
        # Add extra bright inner ring for visibility - MORE TRANSPARENT
        lighter_color = tuple(min(c + 100, 255) for c in color)
        overlay = frame.copy()
        cv2.ellipse(
            overlay,
            (center_x, center_y),
            (int(radius_x * 0.9), int(radius_y * 0.9)),
            0, 0, 360,  # Full circle
            lighter_color,
            2,
            cv2.LINE_AA
        )
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)  # 40% opacity
        
        return frame
    
    @staticmethod
    def draw_pulse_circle(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                          color: Tuple[int, int, int] = (0, 165, 255),  # Orange in BGR (B=0, G=165, R=255)
                          frame_count: int = 0, player=None) -> np.ndarray:
        """
        Pulsing circle animation - expands and contracts

        Args:
            frame: Frame to draw on
            bbox: Bounding box (padded)
            color: Circle color
            frame_count: Current frame number (for animation)
            player: Player object (for accessing original_bbox)

        Returns:
            Frame with pulsing circle
        """
        x, y, w, h = bbox
        center_x = x + w // 2

        # Base radius - LARGE ENOUGH to keep player inside during movement
        # Use larger radius to accommodate running/movement
        base_radius_x = max(int(w * 0.8), 45)  # Large - around feet with margin
        base_radius_y = max(int(w * 0.15), 10)  # Smaller vertical radius - flat on floor

        # Pulse animation (sine wave)
        pulse_factor = 1.0 + 0.15 * math.sin(frame_count * 0.2)
        radius_x = int(base_radius_x * pulse_factor)
        radius_y = int(base_radius_y * pulse_factor)

        # Position ring on floor at feet contact point
        # Use original_bbox if available (before padding was added)
        if hasattr(player, 'current_original_bbox') and player and player.current_original_bbox:
            orig_x, orig_y, orig_w, orig_h = player.current_original_bbox
            # Feet are at the BOTTOM of the original bbox (where the actual feet are)
            feet_level = orig_y + orig_h
        else:
            # Fallback: use padded bbox with offset
            floor_offset = int(h * 0.10)  # Add 10% of height to reach actual floor
            feet_level = (y + h) + floor_offset

        center_y = feet_level - radius_y  # Position so bottom touches floor
        
        # Draw full pulsing circle (360 degrees) - MORE TRANSPARENT so it doesn't hide video
        # Draw outer fading ring - MORE TRANSPARENT
        for i in range(3):
            alpha_factor = 1.0 - (i * 0.3)
            overlay = frame.copy()
            cv2.ellipse(
                overlay,
                (center_x, center_y),
                (radius_x + i * 4, radius_y + i * 2),  # Smaller glow
                0, 0, 360,  # Full circle
                color,
                2,
                cv2.LINE_AA
            )
            cv2.addWeighted(overlay, alpha_factor * 0.25, frame, 1.0 - alpha_factor * 0.25, 0, frame)  # More transparent
        
        # Draw main circle - MORE TRANSPARENT
        overlay = frame.copy()
        cv2.ellipse(
            overlay,
            (center_x, center_y),
            (radius_x, radius_y),
            0, 0, 360,  # Full circle
            color,
            2,  # Thinner (was 3)
            cv2.LINE_AA
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)  # 60% opacity
        
        return frame
    
    @staticmethod
    def draw_gradient_ring(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                          color1: Tuple[int, int, int] = (255, 200, 0),
                          color2: Tuple[int, int, int] = (255, 0, 200),
                          frame_count: int = 0, player=None) -> np.ndarray:
        """
        Gradient ring with rotating glow effect - purple with more presence

        Args:
            frame: Frame to draw on
            bbox: Bounding box (padded)
            color1: Start color (purple)
            color2: End color (purple variant)
            frame_count: Current frame number (for rotation animation)
            player: Player object (for accessing original_bbox)

        Returns:
            Frame with gradient ring
        """
        x, y, w, h = bbox
        center_x = x + w // 2

        # Calculate radii - LARGE ENOUGH to keep player inside during movement
        # Use larger radius to accommodate running/movement
        radius_x = max(int(w * 0.8), 45)  # Large size - around feet with margin
        radius_y = max(int(w * 0.15), 10)  # Smaller vertical radius - flat on floor

        # Position ring on floor at feet contact point
        # Use original_bbox if available (before padding was added)
        if hasattr(player, 'current_original_bbox') and player and player.current_original_bbox:
            orig_x, orig_y, orig_w, orig_h = player.current_original_bbox
            # Feet are at the BOTTOM of the original bbox (where the actual feet are)
            feet_level = orig_y + orig_h
        else:
            # Fallback: use padded bbox with offset
            floor_offset = int(h * 0.10)  # Add 10% of height to reach actual floor
            feet_level = (y + h) + floor_offset

        center_y = feet_level - radius_y  # Position so bottom touches floor
        
        # Draw rotating glow effect (outer glow that rotates)
        rotation_offset = int(frame_count * 2) % 360  # Rotate glow
        
        # Draw outer rotating glow
        for i in range(8, 0, -1):
            overlay = frame.copy()
            glow_radius_x = radius_x + i * 4
            glow_radius_y = radius_y + i * 2
            
            # Draw rotating segments
            for angle in range(0, 360, 30):
                glow_angle = (angle + rotation_offset) % 360
                # Calculate gradient color based on angle
                t = (glow_angle / 360.0)
                glow_color = (
                    int(color1[0] * (1 - t) + color2[0] * t),
                    int(color1[1] * (1 - t) + color2[1] * t),
                    int(color1[2] * (1 - t) + color2[2] * t)
                )
                
                cv2.ellipse(
                    overlay,
                    (center_x, center_y),
                    (glow_radius_x, glow_radius_y),
                    0,
                    glow_angle - 15,
                    glow_angle + 15,
                    glow_color,
                    3,
                    cv2.LINE_AA
                )
            cv2.addWeighted(overlay, 0.2 - (i * 0.02), frame, 1.0 - (0.2 - (i * 0.02)), 0, frame)
        
        # Draw main thick gradient ring
        overlay = frame.copy()
        num_segments = 360
        for angle in range(0, num_segments, 2):
            # Calculate gradient color
            t = angle / num_segments
            color = (
                int(color1[0] * (1 - t) + color2[0] * t),
                int(color1[1] * (1 - t) + color2[1] * t),
                int(color1[2] * (1 - t) + color2[2] * t)
            )
            
            # Draw arc segment with thicker line
            cv2.ellipse(
                overlay,
                (center_x, center_y),
                (radius_x, radius_y),
                0,
                angle,
                angle + 5,
                color,
                5,  # Thicker line for more presence
                cv2.LINE_AA
            )
        # Apply transparency
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)  # More opaque for presence
        
        # Add inner bright ring
        inner_overlay = frame.copy()
        for angle in range(0, 360, 3):
            t = angle / 360.0
            bright_color = (
                min(255, int(color1[0] * (1 - t) + color2[0] * t) + 50),
                min(255, int(color1[1] * (1 - t) + color2[1] * t) + 50),
                min(255, int(color1[2] * (1 - t) + color2[2] * t) + 50)
            )
            cv2.ellipse(
                inner_overlay,
                (center_x, center_y),
                (int(radius_x * 0.85), int(radius_y * 0.85)),
                0,
                angle,
                angle + 5,
                bright_color,
                2,
                cv2.LINE_AA
            )
        cv2.addWeighted(inner_overlay, 0.5, frame, 0.5, 0, frame)
        
        return frame

    @staticmethod
    def draw_nba_iso_ring(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                          color: Tuple[int, int, int] = (0, 215, 255),
                          frame_count: int = 0, player=None) -> np.ndarray:
        """
        NBA Iso Ring - flat, glowing ellipse on the floor with subtle pulse animation

        Args:
            frame: Frame to draw on (BGR)
            bbox: Bounding box (padded)
            color: Base ring color
            frame_count: Current frame number for animation
            player: Player object (optional, for accessing original_bbox)

        Returns:
            Frame with iso ring drawn
        """
        x, y, w, h = bbox
        center_x = x + w // 2

        # Base radii sized to player width; pulse adds presence without overpowering
        base_radius_x = max(int(w * 0.8), 55)
        base_radius_y = max(int(w * 0.18), 12)
        pulse = 0.5 * (1 + math.sin(frame_count * 0.2))
        radius_x = int(base_radius_x * (0.9 + 0.15 * pulse))
        radius_y = int(base_radius_y * (0.9 + 0.15 * pulse))
        thickness = max(2, int(3 + pulse * 3))

        # Place the ellipse at the feet using original bbox when available
        if hasattr(player, 'current_original_bbox') and player and player.current_original_bbox:
            orig_x, orig_y, orig_w, orig_h = player.current_original_bbox
            feet_level = orig_y + orig_h
        else:
            feet_level = y + h
        center_y = feet_level - max(2, int(radius_y * 0.35))

        # Glow layers for soft presence on the floor
        for i in range(3):
            overlay = frame.copy()
            glow_radius_x = radius_x + i * 6
            glow_radius_y = radius_y + i * 3
            glow_alpha = max(0.05, (0.18 - i * 0.04) + pulse * 0.08)
            cv2.ellipse(
                overlay,
                (center_x, center_y),
                (glow_radius_x, glow_radius_y),
                0, 0, 360,
                color,
                2,
                cv2.LINE_AA
            )
            cv2.addWeighted(overlay, glow_alpha, frame, 1.0 - glow_alpha, 0, frame)

        # Main ring with pulse-adjusted thickness
        overlay = frame.copy()
        cv2.ellipse(
            overlay,
            (center_x, center_y),
            (radius_x, radius_y),
            0, 0, 360,
            color,
            thickness,
            cv2.LINE_AA
        )
        main_alpha = 0.65 + 0.1 * pulse
        cv2.addWeighted(overlay, main_alpha, frame, 1.0 - main_alpha, 0, frame)

        # Inner highlight for extra clarity
        highlight_color = tuple(min(c + 40, 255) for c in color)
        overlay = frame.copy()
        cv2.ellipse(
            overlay,
            (center_x, center_y),
            (int(radius_x * 0.85), int(radius_y * 0.7)),
            0, 0, 360,
            highlight_color,
            1,
            cv2.LINE_AA
        )
        highlight_alpha = 0.35 + 0.1 * pulse
        cv2.addWeighted(overlay, highlight_alpha, frame, 1.0 - highlight_alpha, 0, frame)

        return frame

    @staticmethod
    def draw_floating_ar_tag(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                             color: Tuple[int, int, int] = (0, 215, 255),
                             frame_count: int = 0, player=None) -> np.ndarray:
        """
        Floating AR tag above head with connector line and glow base (success/fail styling)

        Args:
            frame: Frame to draw on
            bbox: Bounding box (padded)
            color: Accent color
            frame_count: Frame number for subtle bobbing
            player: Player object (optional)

        Returns:
            Frame with AR tag
        """
        x, y, w, h = bbox
        center_x = x + w // 2

        # Position above head using original bbox if available
        if hasattr(player, 'current_original_bbox') and player and player.current_original_bbox:
            orig_x, orig_y, orig_w, _ = player.current_original_bbox
            head_y = orig_y
            feet_y = orig_y + player.current_original_bbox[3]
        else:
            head_y = y
            feet_y = y + h

        bob = int(5 * np.sin(frame_count * 0.12))
        card_width = max(180, int(w * 0.85))
        card_height = 80
        card_bottom = max(0, head_y - int(h * 0.15) + bob)
        card_top = max(0, card_bottom - card_height)
        card_left = center_x - card_width // 2
        card_right = center_x + card_width // 2

        # Status/color logic
        success_color = (113, 204, 46)  # BGR for #2ecc71
        fail_color = (60, 76, 231)      # BGR for #e74c3c
        is_fail = bool(getattr(player, "tracking_lost", False))
        status_color = fail_color if is_fail else success_color
        status_icon = "✗" if is_fail else "✓"
        action_text = "Tracking Lost" if is_fail else "Tracking Active"
        comment_text = "Reacquire target" if is_fail else "Locked on player"
        name_text = player.name if player and getattr(player, "name", None) else "Player"

        # Drop shadow
        shadow_overlay = frame.copy()
        cv2.rectangle(shadow_overlay, (card_left + 3, card_top + 6), (card_right + 3, card_bottom + 6), (0, 0, 0), -1, cv2.LINE_AA)
        cv2.addWeighted(shadow_overlay, 0.35, frame, 0.65, 0, frame)

        # Card body
        card_overlay = frame.copy()
        body_color = (30, 30, 40)
        border_color = (60, 60, 70)
        cv2.rectangle(card_overlay, (card_left, card_top), (card_right, card_bottom), body_color, -1, cv2.LINE_AA)
        cv2.rectangle(card_overlay, (card_left, card_top), (card_right, card_bottom), border_color, 1, cv2.LINE_AA)
        cv2.rectangle(card_overlay, (card_left, card_top), (card_left + 5, card_bottom), color, -1, cv2.LINE_AA)

        # Text rows
        baseline = card_top + 20
        cv2.putText(card_overlay, name_text, (card_left + 12, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)

        # Action row and status icon
        cv2.putText(card_overlay, action_text, (card_left + 12, baseline + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        cv2.putText(card_overlay, status_icon, (card_right - 18, baseline + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

        # Comment row
        cv2.putText(card_overlay, comment_text, (card_left + 12, baseline + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 190), 1, cv2.LINE_AA)

        cv2.addWeighted(card_overlay, 0.9, frame, 0.1, 0, frame)

        # Connector to floor
        connector_overlay = frame.copy()
        cv2.line(connector_overlay, (center_x, card_bottom), (center_x, feet_y), color, 2, cv2.LINE_AA)
        cv2.addWeighted(connector_overlay, 0.7, frame, 0.3, 0, frame)

        # Base ellipse on floor with glow
        base_overlay = frame.copy()
        base_rx = max(12, int(w * 0.12))
        base_ry = max(5, int(w * 0.04))
        cv2.ellipse(base_overlay, (center_x, feet_y), (base_rx, base_ry), 0, 0, 360, color, -1, cv2.LINE_AA)
        cv2.ellipse(base_overlay, (center_x, feet_y), (base_rx + 4, base_ry + 2), 0, 0, 360, color, 1, cv2.LINE_AA)
        cv2.addWeighted(base_overlay, 0.6, frame, 0.4, 0, frame)

        return frame

    @staticmethod
    def draw_tactical_brackets(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                               color: Tuple[int, int, int] = (0, 215, 255),
                               frame_count: int = 0, player=None) -> np.ndarray:
        """
        Analytic-style tactical brackets that gently breathe

        Args:
            frame: Frame to draw on
            bbox: Bounding box (padded)
            color: Bracket color
            frame_count: Frame number for breathing animation
            player: Player object (unused)

        Returns:
            Frame with tactical brackets
        """
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2

        scale = 1.0 + 0.05 * np.sin(frame_count * 0.08)
        box_w = int(max(w * 0.9, 80) * scale)
        box_h = int(max(h * 0.85, 100) * scale)
        left = center_x - box_w // 2
        right = center_x + box_w // 2
        top = center_y - box_h // 2
        bottom = center_y + box_h // 2

        corner_len = int(min(box_w, box_h) * 0.18)
        thickness = 3

        def draw_corners(img, col, thick):
            corners = [
                ((left, top), (left + corner_len, top), (left, top + corner_len)),            # TL
                ((right, top), (right - corner_len, top), (right, top + corner_len)),        # TR
                ((left, bottom), (left + corner_len, bottom), (left, bottom - corner_len)),  # BL
                ((right, bottom), (right - corner_len, bottom), (right, bottom - corner_len))# BR
            ]
            for (p, h_end, v_end) in corners:
                cv2.line(img, p, h_end, col, thick, cv2.LINE_AA)
                cv2.line(img, p, v_end, col, thick, cv2.LINE_AA)

        # Glow layer
        glow_overlay = frame.copy()
        draw_corners(glow_overlay, color, thickness + 2)
        cv2.addWeighted(glow_overlay, 0.18, frame, 0.82, 0, frame)

        # Main brackets
        main_overlay = frame.copy()
        draw_corners(main_overlay, color, thickness)
        cv2.addWeighted(main_overlay, 0.85, frame, 0.15, 0, frame)

        return frame

    @staticmethod
    def draw_sonar_ripple(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                          color: Tuple[int, int, int] = (0, 215, 255),
                          frame_count: int = 0, player=None) -> np.ndarray:
        """
        Sonar-like ripples on the floor with perspective flattening

        Args:
            frame: Frame to draw on
            bbox: Bounding box (padded)
            color: Wave color
            frame_count: Frame number for ripple timing
            player: Player object (optional)

        Returns:
            Frame with sonar ripples
        """
        x, y, w, h = bbox
        center_x = x + w // 2

        if hasattr(player, 'current_original_bbox') and player and player.current_original_bbox:
            orig_x, orig_y, orig_w, orig_h = player.current_original_bbox
            feet_y = orig_y + orig_h
        else:
            feet_y = y + h

        base_radius_x = max(int(w * 0.75), 60)
        base_radius_y = max(int(w * 0.18), 12)

        # Emitter base
        emitter_overlay = frame.copy()
        cv2.ellipse(emitter_overlay, (center_x, feet_y), (int(base_radius_x * 0.35), int(base_radius_y * 0.3)), 0, 0, 360, color, -1, cv2.LINE_AA)
        cv2.addWeighted(emitter_overlay, 0.5, frame, 0.5, 0, frame)

        # Multiple ripples with phase offset
        for i in range(3):
            phase = frame_count * 0.08 + i * 0.65
            progress = (phase % (2 * math.pi)) / (2 * math.pi)  # 0-1
            scale = 0.6 + progress * 1.3
            alpha = max(0.0, 0.55 * (1 - progress))
            thickness = max(1, int(3 - progress * 2))

            overlay = frame.copy()
            cv2.ellipse(
                overlay,
                (center_x, feet_y),
                (int(base_radius_x * scale), int(base_radius_y * scale * 0.9)),
                0, 0, 360,
                color,
                thickness,
                cv2.LINE_AA
            )
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        return frame

    @staticmethod
    def draw_floating_chevron(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                              color: Tuple[int, int, int] = (0, 215, 255),
                              frame_count: int = 0, player=None) -> np.ndarray:
        """
        Floating chevron above the player's head with smooth bobbing and drop shadow

        Args:
            frame: Frame to draw on (BGR)
            bbox: Bounding box (padded)
            color: Chevron color
            frame_count: Current frame number for bobbing animation
            player: Player object (unused)

        Returns:
            Frame with floating chevron
        """
        x, y, w, _ = bbox
        center_x = x + w // 2
        head_y = y

        # Bobbing animation (soft sine wave)
        bob_offset = int(8 * np.sin(frame_count * 0.15))
        chevron_y = max(0, head_y - 60 + bob_offset)

        # Chevron size relative to player width
        half_width = max(20, int(w * 0.25))
        height = max(18, int(w * 0.18))

        # Define triangle points (pointing down)
        triangle = np.array([
            [center_x - half_width, chevron_y],
            [center_x + half_width, chevron_y],
            [center_x, chevron_y + height]
        ], np.int32)

        # Drop shadow (offset)
        shadow = triangle + np.array([2, 2])
        cv2.fillPoly(frame, [shadow], (0, 0, 0), lineType=cv2.LINE_AA)

        # Main chevron
        cv2.fillPoly(frame, [triangle], color, lineType=cv2.LINE_AA)

        return frame
    
    @staticmethod
    def draw_spotlight(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                       color: Tuple[int, int, int] = (100, 255, 255), player=None) -> np.ndarray:
        """
        CLEAN REWRITE - Simple spotlight beam with NO memory corruption
        Pure numpy operations, minimal OpenCV calls

        Args:
            frame: Frame to draw on
            bbox: Bounding box (padded)
            color: Light color (cyan/white)
            player: Player object (for accessing original_bbox)

        Returns:
            Frame with clean spotlight beam
        """
        x, y, w, h = bbox
        height, width = frame.shape[:2]

        # Get player center and feet position (with bounds checking)
        center_x = int(np.clip(x + w // 2, 0, width - 1))

        if hasattr(player, 'current_original_bbox') and player and player.current_original_bbox:
            orig_x, orig_y, orig_w, orig_h = player.current_original_bbox
            feet_y = int(np.clip(orig_y + orig_h, 0, height - 1))
        else:
            feet_y = int(np.clip(y + h, 0, height - 1))

        # Beam dimensions - cone shape (narrow at top, wide at bottom)
        top_width = max(int(w * 0.5), 30)
        bottom_width = max(int(w * 1.5), 80)

        # Create coordinate grids
        yy, xx = np.ogrid[:height, :width]

        # Calculate beam cone mask using vectorized operations
        # Linear interpolation of width from top to feet
        y_ratio = np.clip(yy / max(feet_y, 1), 0, 1)
        beam_width = top_width + (bottom_width - top_width) * y_ratio

        # Distance from center line
        dist_from_center = np.abs(xx - center_x)

        # Smooth cone mask (1.0 at center, 0.0 outside)
        cone_mask = np.clip(1.0 - (dist_from_center / (beam_width / 2 + 1)), 0, 1)
        cone_mask = cone_mask ** 2  # Smoother falloff

        # Only show beam above feet
        cone_mask = np.where(yy <= feet_y, cone_mask, 0.0)

        # Darken everything to 50%
        result = (frame * 0.5).astype(np.uint8)

        # Brighten beam area
        beam_brightness = 1.0 + cone_mask * 0.4
        brightened = np.clip(frame * beam_brightness[:, :, np.newaxis], 0, 255).astype(np.uint8)

        # Blend using mask
        mask_3d = np.stack([cone_mask, cone_mask, cone_mask], axis=2)
        result = (brightened * mask_3d + result * (1 - mask_3d)).astype(np.uint8)

        # Add floor circle (simple, direct drawing - no overlays)
        floor_radius_x = int(bottom_width * 0.4)
        floor_radius_y = int(bottom_width * 0.1)

        # Draw glowing circles directly on result (no copy+blend)
        cv2.ellipse(result, (center_x, feet_y), (floor_radius_x + 20, floor_radius_y + 8),
                   0, 0, 360, tuple(int(c * 0.3) for c in color), -1, cv2.LINE_AA)
        cv2.ellipse(result, (center_x, feet_y), (floor_radius_x + 10, floor_radius_y + 4),
                   0, 0, 360, tuple(int(c * 0.6) for c in color), -1, cv2.LINE_AA)
        cv2.ellipse(result, (center_x, feet_y), (floor_radius_x, floor_radius_y),
                   0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)

        return result

    @staticmethod
    def get_spotlight_mask(frame_shape: tuple, bbox: Tuple[int, int, int, int], player=None) -> np.ndarray:
        """
        Calculate spotlight mask without drawing (for combining multiple spotlights)

        Args:
            frame_shape: Shape of frame (height, width, channels)
            bbox: Bounding box (padded)
            player: Player object (for accessing original_bbox)

        Returns:
            2D mask array (values 0-1)
        """
        height, width = frame_shape[:2]
        x, y, w, h = bbox

        # Calculate light column center using original_bbox for accurate feet position
        center_x = x + w // 2

        # Get feet position
        if hasattr(player, 'current_original_bbox') and player and player.current_original_bbox:
            orig_x, orig_y, orig_w, orig_h = player.current_original_bbox
            feet_y = orig_y + orig_h
        else:
            feet_y = y + h

        center_y = feet_y

        # Light column width - narrower at top, wider at bottom (like a cone)
        # Make wider to accommodate player movement
        top_width = max(int(w * 0.4), 25)  # Narrow at top (wider than before)
        bottom_width = max(int(w * 1.5), 75)  # Wider at bottom (more room for movement)

        # Create smooth gradient mask for light column
        y_coords, x_coords = np.ogrid[:height, :width]

        # Calculate distance from center line (horizontal distance)
        # CRITICAL FIX: Ensure center_x is within bounds to prevent array corruption
        center_x = int(np.clip(center_x, 0, width - 1))
        dx = np.abs(x_coords - center_x)

        # Calculate width at each y position (linear interpolation from top to feet)
        # CRITICAL FIX: Ensure center_y is within bounds
        center_y = int(np.clip(center_y, 1, height - 1))
        y_normalized = np.clip(y_coords / max(center_y, 1), 0, 1)
        width_at_y = top_width + (bottom_width - top_width) * y_normalized

        # Distance from center line normalized by width at this y
        normalized_distance = dx / np.maximum(width_at_y, 1)

        # Create smooth falloff - fully lit in center, gradually darken towards edges
        falloff_start = 0.0
        falloff_end = 1.0

        # Smoothstep interpolation
        t = np.clip((normalized_distance - falloff_start) / (falloff_end - falloff_start), 0, 1)
        smoothstep = t * t * (3 - 2 * t)

        # Create beam cone mask (only inside the cone, above feet)
        beam_cone_mask = 1.0 - smoothstep  # 1.0 at center, 0.0 at cone edges

        # Fade from top to bottom
        top_fade = 1.0
        bottom_fade = 0.9
        y_fade = top_fade - (top_fade - bottom_fade) * np.clip(y_coords / max(center_y, 1), 0, 1)
        beam_cone_mask = beam_cone_mask * y_fade

        # Limit beam to ONLY above feet level
        beam_vertical_mask = np.where(y_coords <= feet_y, 1.0, 0.0)
        beam_cone_mask = beam_cone_mask * beam_vertical_mask

        return beam_cone_mask

    @staticmethod
    def apply_spotlight_mask(original_frame: np.ndarray, darkened_frame: np.ndarray,
                            combined_mask: np.ndarray) -> np.ndarray:
        """
        Apply combined spotlight mask to frame

        Args:
            original_frame: Original bright frame
            darkened_frame: Pre-darkened frame
            combined_mask: Combined mask from all spotlights (2D array, 0-1)

        Returns:
            Frame with spotlight effect
        """
        # FIXED: Use original video brightness in spotlight area, darken outside
        # Expand mask to 3 channels (uint8: 0-255 range)
        mask_uint8 = (np.clip(combined_mask, 0, 1) * 255).astype(np.uint8)
        mask_3d = np.stack([mask_uint8, mask_uint8, mask_uint8], axis=2)

        # Blend: original video in spotlight, darkened outside
        # result = (original * mask + darkened * (255 - mask)) / 255
        result = ((original_frame.astype(np.uint16) * mask_3d +
                  darkened_frame.astype(np.uint16) * (255 - mask_3d)) // 255).astype(np.uint8)

        return result

    @staticmethod
    def draw_spotlight_floor_circle(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                                    color: Tuple[int, int, int] = (100, 255, 255), player=None) -> np.ndarray:
        """
        Draw only the floor circle for spotlight (used after mask is applied)

        Args:
            frame: Frame to draw on
            bbox: Bounding box (padded)
            color: Light color (cyan/white)
            player: Player object (for accessing original_bbox)

        Returns:
            Frame with floor circle
        """
        x, y, w, h = bbox

        # Calculate center and feet position
        center_x = x + w // 2

        if hasattr(player, 'current_original_bbox') and player and player.current_original_bbox:
            orig_x, orig_y, orig_w, orig_h = player.current_original_bbox
            feet_y = orig_y + orig_h
        else:
            feet_y = y + h

        # Light column width - match the beam cone width
        bottom_width = max(int(w * 1.5), 75)

        # Floor circle dimensions
        floor_radius_x = int(bottom_width * 0.4)
        floor_radius_y = int(bottom_width * 0.1)

        # FIXED: Draw semi-transparent floor circles using alpha blending
        # Create overlay for transparent circles
        overlay = frame.copy()

        # Draw outer glow circle (very subtle)
        cv2.ellipse(overlay, (center_x, feet_y), (floor_radius_x + 15, floor_radius_y + 6),
                   0, 0, 360, tuple(int(c * 0.5) for c in color), -1, cv2.LINE_AA)

        # Draw middle glow circle
        cv2.ellipse(overlay, (center_x, feet_y), (floor_radius_x + 7, floor_radius_y + 3),
                   0, 0, 360, tuple(int(c * 0.7) for c in color), -1, cv2.LINE_AA)

        # Draw inner bright circle (white/bright)
        cv2.ellipse(overlay, (center_x, feet_y), (floor_radius_x, floor_radius_y),
                   0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)

        # Blend overlay with original frame (30% opacity for subtle effect)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        return frame

    @staticmethod
    def draw_spotlight_no_darken(original_frame: np.ndarray, darkened_frame: np.ndarray,
                                 bbox: Tuple[int, int, int, int],
                                 color: Tuple[int, int, int] = (100, 255, 255), player=None) -> np.ndarray:
        """
        Draw spotlight effect WITHOUT darkening the frame again (for multiple spotlights)
        This function is called from draw_all_markers when there are multiple spotlight players

        Args:
            original_frame: Original bright frame to blend
            darkened_frame: Pre-darkened frame (darkened once for all spotlights)
            bbox: Bounding box (padded)
            color: Light color (cyan/white)
            player: Player object (for accessing original_bbox)

        Returns:
            Frame with alien beam effect added
        """
        x, y, w, h = bbox
        height, width = original_frame.shape[:2]

        # Calculate light column center using original_bbox for accurate feet position
        center_x = x + w // 2

        # Get feet position for floor circle
        if hasattr(player, 'current_original_bbox') and player and player.current_original_bbox:
            orig_x, orig_y, orig_w, orig_h = player.current_original_bbox
            feet_y = orig_y + orig_h
        else:
            feet_y = y + h

        center_y = feet_y  # Beam goes to feet level

        # Light column width - narrower at top, wider at bottom (like a cone)
        # Make wider to accommodate player movement
        top_width = max(int(w * 0.4), 25)  # Narrow at top (wider than before)
        bottom_width = max(int(w * 1.5), 75)  # Wider at bottom (more room for movement)

        # Create a mask for the light column area
        y_coords, x_coords = np.ogrid[:height, :width]

        # CRITICAL FIX: Ensure center_x and center_y are within bounds to prevent array corruption
        center_x = int(np.clip(center_x, 0, width - 1))
        center_y = int(np.clip(center_y, 1, height - 1))

        # Calculate distance from center line for each pixel
        # Create cone-shaped mask
        width_at_y = np.where(y_coords <= center_y,
                             top_width + (bottom_width - top_width) * (y_coords / max(center_y, 1)),
                             bottom_width)

        distance_from_center = np.abs(x_coords - center_x)
        normalized_distance = np.clip(distance_from_center / (width_at_y / 2), 0, 1)

        # Use smoothstep function for natural transition
        falloff_start = 0.0
        falloff_end = 1.0

        # Smoothstep interpolation
        t = np.clip((normalized_distance - falloff_start) / (falloff_end - falloff_start), 0, 1)
        smoothstep = t * t * (3 - 2 * t)

        # Create beam cone mask (only inside the cone, above feet)
        # Inside cone: value between 0-1 (for smooth beam edges)
        # Outside cone OR below feet: 0 (completely dark)
        beam_cone_mask = 1.0 - smoothstep  # 1.0 at center, 0.0 at cone edges

        # Also fade from top to bottom (brighter at top, slightly dimmer at player level)
        top_fade = 1.0
        bottom_fade = 0.9
        y_fade = top_fade - (top_fade - bottom_fade) * np.clip(y_coords / max(center_y, 1), 0, 1)
        beam_cone_mask = beam_cone_mask * y_fade

        # CRITICAL: Limit beam to ONLY above feet level
        # Below feet OR outside cone: mask = 0 (use darkened frame)
        # Inside cone AND above feet: mask > 0 (brighten)
        beam_vertical_mask = np.where(y_coords <= feet_y, 1.0, 0.0)
        beam_cone_mask = beam_cone_mask * beam_vertical_mask

        # Calculate brightness boost ONLY inside the beam cone
        # Outside: no boost (will use darkened_frame)
        # Inside: boost from 1.0 to 1.3
        brightness_boost = 1.0 + beam_cone_mask * 0.3  # 1.0 outside, up to 1.3 inside

        # Brighten the beam area
        brightened_frame = np.clip(original_frame.astype(np.float32) * brightness_boost[:, :, np.newaxis], 0, 255).astype(np.uint8)

        # Create 3-channel mask for blending
        mask_3channel = np.stack([beam_cone_mask, beam_cone_mask, beam_cone_mask], axis=2)

        # Blend: use brightened frame inside cone, darkened frame everywhere else
        # This ensures uniform darkness outside the cone (no gradients in dark areas)
        result = (brightened_frame.astype(np.float32) * mask_3channel +
                 darkened_frame.astype(np.float32) * (1 - mask_3channel)).astype(np.uint8)

        # Add bright floor circle where beam hits the ground (like alien abduction)
        # Make floor circle match the bottom width of the cone exactly
        floor_radius_x = int(bottom_width * 0.5)  # Half of bottom_width = radius
        floor_radius_y = int(bottom_width * 0.12)  # Flat ellipse on floor

        # Draw glowing floor circle - use single overlay to prevent corruption
        floor_overlay = np.zeros_like(result, dtype=np.uint8)
        for i in range(4, 0, -1):
            glow_radius_x = floor_radius_x + i * 8
            glow_radius_y = floor_radius_y + i * 3
            alpha = 0.15 - (i * 0.03)
            # Scale color by alpha and draw on overlay
            glow_color = tuple(int(c * alpha * 4) for c in color)  # *4 to compensate for blend
            cv2.ellipse(floor_overlay, (center_x, feet_y), (glow_radius_x, glow_radius_y),
                       0, 0, 360, glow_color, -1, cv2.LINE_AA)

        # Main floor circle (brightest)
        cv2.ellipse(floor_overlay, (center_x, feet_y), (floor_radius_x, floor_radius_y),
                   0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)

        # Blend floor overlay in one operation
        result = cv2.addWeighted(floor_overlay, 0.35, result, 0.65, 0)

        # Add richer beam core and rim glows for a premium look
        core_color = tuple(min(c + 50, 255) for c in color)
        top_y = max(0, int(center_y - h * 1.6))
        top_y = int(np.clip(top_y, 0, height - 1))

        # CRITICAL FIX: Use a single transparent overlay to prevent memory corruption
        # Create a transparent overlay for all beam details
        beam_details = np.zeros_like(result, dtype=np.uint8)

        # Validate that we have enough space for the beam
        if top_y < center_y - 10:  # Only draw if beam has at least 10 pixels height
            try:
                # Calculate and validate core trapezoid points
                top_left_x = int(np.clip(center_x - int(top_width * 0.45), 0, width - 1))
                top_right_x = int(np.clip(center_x + int(top_width * 0.45), 0, width - 1))
                bottom_left_x = int(np.clip(center_x - int(bottom_width * 0.55), 0, width - 1))
                bottom_right_x = int(np.clip(center_x + int(bottom_width * 0.55), 0, width - 1))

                # Validate that we have a proper trapezoid (non-degenerate)
                if (top_right_x > top_left_x and bottom_right_x > bottom_left_x):
                    core_points = np.array([
                        [top_left_x, top_y],
                        [top_right_x, top_y],
                        [bottom_right_x, center_y],
                        [bottom_left_x, center_y]
                    ], np.int32)

                    # Draw on transparent overlay
                    cv2.fillPoly(beam_details, [core_points], core_color, lineType=cv2.LINE_AA)

                # Draw rim glow lines
                left_top_x = int(np.clip(center_x - top_width // 2, 0, width - 1))
                left_bottom_x = int(np.clip(center_x - bottom_width // 2, 0, width - 1))
                right_top_x = int(np.clip(center_x + top_width // 2, 0, width - 1))
                right_bottom_x = int(np.clip(center_x + bottom_width // 2, 0, width - 1))

                # Draw lines
                cv2.line(beam_details, (left_top_x, top_y), (left_bottom_x, center_y),
                        core_color, 2, cv2.LINE_AA)
                cv2.line(beam_details, (right_top_x, top_y), (right_bottom_x, center_y),
                        core_color, 2, cv2.LINE_AA)

                # Subtle vertical rays inside the beam
                for offset in (-1, 0, 1):
                    offset_top = int(np.clip(center_x + offset * max(2, top_width // 6), 0, width - 1))
                    offset_bottom = int(np.clip(center_x + offset * max(3, bottom_width // 6), 0, width - 1))
                    cv2.line(beam_details, (offset_top, top_y), (offset_bottom, center_y),
                            core_color, 1, cv2.LINE_AA)

            except Exception as e:
                # If any drawing operation fails, log it but don't crash
                print(f"[Spotlight Warning] Failed to draw beam details: {e}")

        # Blend beam details onto result in one operation
        # This prevents memory corruption from multiple blend operations
        result = cv2.addWeighted(beam_details, 0.25, result, 0.75, 0)

        # Hot center on the floor impact point
        inner_overlay = result.copy()
        inner_radius_x = max(6, int(floor_radius_x * 0.45))
        inner_radius_y = max(3, int(floor_radius_y * 0.6))
        cv2.ellipse(inner_overlay, (center_x, feet_y), (inner_radius_x, inner_radius_y),
                   0, 0, 360, core_color, -1, cv2.LINE_AA)
        cv2.addWeighted(inner_overlay, 0.55, result, 0.45, 0, result)

        return result

    @staticmethod
    def draw_dynamic_arrow(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                          color: Tuple[int, int, int] = (0, 255, 255),  # Bright cyan - very visible
                          frame_count: int = 0, player=None) -> np.ndarray:
        """
        Dynamic animated arrow - smooth bouncing with sharper design and bright colors

        Args:
            frame: Frame to draw on
            bbox: Bounding box (padded)
            color: Arrow color (bright cyan by default)
            frame_count: Frame number for animation
            player: Player object (for accessing original_bbox)

        Returns:
            Frame with dynamic arrow
        """
        x, y, w, h = bbox
        center_x = x + w // 2

        # Position arrow above head using original_bbox if available
        if hasattr(player, 'current_original_bbox') and player and player.current_original_bbox:
            orig_x, orig_y, orig_w, orig_h = player.current_original_bbox
            # Position much higher above original head
            base_arrow_y = max(0, orig_y - 65)
        else:
            # Fallback
            base_arrow_y = max(0, y - 65)

        # Smooth bounce animation
        bounce_offset = int(8 * math.sin(frame_count * 0.12))
        arrow_y = max(0, base_arrow_y + bounce_offset)

        # Arrow size - proportional to player size, smaller
        arrow_size = max(int(w * 0.25), 25)  # Smaller and proportional
        
        # Create sharper, more elegant arrow shape
        tip_y = arrow_y
        base_y = arrow_y + arrow_size
        base_width = arrow_size
        
        # Draw glow effect (multiple layers for smooth glow)
        for i in range(5, 0, -1):
            overlay = frame.copy()
            glow_size = arrow_size + i * 4
            glow_base_width = base_width + i * 3
            glow_tip_y = tip_y - i * 2
            glow_base_y = base_y + i * 2
            
            glow_points = np.array([
                [center_x, glow_tip_y],
                [center_x - glow_base_width // 2, glow_base_y],
                [center_x - glow_base_width // 3, glow_base_y - glow_size // 4],
                [center_x, glow_base_y - glow_size // 3],
                [center_x + glow_base_width // 3, glow_base_y - glow_size // 4],
                [center_x + glow_base_width // 2, glow_base_y]
            ], np.int32)
            
            cv2.fillPoly(overlay, [glow_points], color)
            cv2.addWeighted(overlay, 0.12 - (i * 0.02), frame, 1.0 - (0.12 - (i * 0.02)), 0, frame)
        
        # Draw main arrow with sharper, more elegant shape
        arrow_points = np.array([
            [center_x, tip_y],
            [center_x - base_width // 2, base_y],
            [center_x - base_width // 3, base_y - arrow_size // 4],  # Inner point for sharper look
            [center_x, base_y - arrow_size // 3],
            [center_x + base_width // 3, base_y - arrow_size // 4],  # Inner point
            [center_x + base_width // 2, base_y]
        ], np.int32)
        
        # Fill arrow
        cv2.fillPoly(frame, [arrow_points], color)
        
        # Draw outline for definition (thicker, more visible)
        cv2.polylines(frame, [arrow_points], True, (255, 255, 255), 3, cv2.LINE_AA)
        
        # Add highlight on top for depth
        highlight_color = tuple(min(c + 60, 255) for c in color)
        highlight_points = np.array([
            [center_x, tip_y],
            [center_x - base_width // 4, base_y - arrow_size // 2],
            [center_x, base_y - arrow_size // 2.5],
            [center_x + base_width // 4, base_y - arrow_size // 2]
        ], np.int32)
        cv2.fillPoly(frame, [highlight_points], highlight_color)
        
        # Add subtle inner outline for extra sharpness
        cv2.polylines(frame, [highlight_points], True, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame
    
    @staticmethod
    def draw_hexagon_outline(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                            color: Tuple[int, int, int] = (255, 150, 0), player=None) -> np.ndarray:
        """
        Hexagon outline - futuristic look with large size to contain entire player

        Args:
            frame: Frame to draw on
            bbox: Bounding box (padded)
            color: Hexagon color
            player: Player object (for accessing original_bbox)

        Returns:
            Frame with hexagon
        """
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2

        # Much larger hexagon to contain entire player with margin
        # Use height for size calculation to ensure it covers head to feet
        size_horizontal = max(int(w * 0.75), 50)  # Wider
        size_vertical = max(int(h * 0.65), 70)  # Taller to cover full body
        
        # Calculate hexagon points (elliptical to fit body shape better)
        points = []
        for i in range(6):
            angle = math.pi / 3 * i - math.pi / 2
            # Use different radii for horizontal and vertical
            px = int(center_x + size_horizontal * math.cos(angle))
            py = int(center_y + size_vertical * math.sin(angle))
            points.append([px, py])

        points = np.array(points, np.int32)
        
        # Draw glow
        for i in range(3, 0, -1):
            overlay = frame.copy()
            cv2.polylines(overlay, [points], True, color, i * 2, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw main hexagon
        cv2.polylines(frame, [points], True, color, 3, cv2.LINE_AA)
        
        return frame
    
    @staticmethod
    def draw_crosshair(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                       color: Tuple[int, int, int] = (255, 255, 0)) -> np.ndarray:
        """
        Tactical Scope crosshair with center gap and end ticks
        
        Args:
            frame: Frame to draw on
            bbox: Bounding box
            color: Crosshair color (default neon cyan)
            
        Returns:
            Frame with crosshair
        """
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        radius = max(int(max(w, h) * 0.5), 30)
        gap = max(10, int(radius * 0.2))  # Leave center clear
        tick = max(6, int(radius * 0.12))
        thickness = 2

        # Horizontal lines with center gap
        cv2.line(frame, (center_x - radius, center_y), (center_x - gap, center_y), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (center_x + gap, center_y), (center_x + radius, center_y), color, thickness, cv2.LINE_AA)

        # Vertical lines with center gap
        cv2.line(frame, (center_x, center_y - radius), (center_x, center_y - gap), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (center_x, center_y + gap), (center_x, center_y + radius), color, thickness, cv2.LINE_AA)

        # Ticks at outer ends for a professional scope look
        cv2.line(frame, (center_x - radius, center_y - tick), (center_x - radius, center_y + tick), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (center_x + radius, center_y - tick), (center_x + radius, center_y + tick), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (center_x - tick, center_y - radius), (center_x + tick, center_y - radius), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (center_x - tick, center_y + radius), (center_x + tick, center_y + radius), color, thickness, cv2.LINE_AA)

        return frame
    
    @staticmethod
    def draw_flame(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                   color: Tuple[int, int, int] = (0, 215, 255),
                   frame_count: int = 0, player=None) -> np.ndarray:
        """
        Premium golden star icon above player - "Player on Fire" indicator
        Professional championship-level broadcast style

        Args:
            frame: Frame to draw on
            bbox: Bounding box (padded)
            color: Star color (gold BGR)
            frame_count: Current frame number (for animation)
            player: Player object (for accessing original_bbox)

        Returns:
            Frame with golden star effect
        """
        x, y, w, h = bbox
        center_x = x + w // 2

        # Position star above head using original_bbox if available
        if hasattr(player, 'current_original_bbox') and player and player.current_original_bbox:
            orig_x, orig_y, orig_w, orig_h = player.current_original_bbox
            # Position above original head (slightly lower than arrows)
            star_y = max(0, orig_y - 50)
        else:
            # Fallback
            star_y = max(0, y - 50)

        # Star size proportional to player
        star_size = max(int(w * 0.35), 30)

        # Pulsing animation (subtle)
        pulse = 1.0 + 0.08 * math.sin(frame_count * 0.15)
        radius = int(star_size * pulse)
        
        # Create 5-pointed star (championship star)
        star_points = []
        num_points = 5

        for i in range(num_points * 2):
            angle = (i * math.pi / num_points) - (math.pi / 2)  # Start from top
            if i % 2 == 0:
                # Outer point
                current_radius = radius
            else:
                # Inner point
                current_radius = int(radius * 0.4)

            x_point = int(center_x + current_radius * math.cos(angle))
            y_point = int(star_y + current_radius * math.sin(angle))
            star_points.append([x_point, y_point])

        star_points = np.array(star_points, np.int32)

        # Gold colors for premium look
        gold_dark = (0, 165, 215)  # Darker gold
        gold_bright = (0, 215, 255)  # Bright gold
        gold_white = (200, 245, 255)  # Almost white gold

        # Draw outer glow (golden aura)
        for i in range(6, 0, -1):
            overlay = frame.copy()
            glow_size = int(radius * (1.0 + i * 0.12))

            # Create larger star for glow
            glow_points = []
            for j in range(num_points * 2):
                angle = (j * math.pi / num_points) - (math.pi / 2)
                if j % 2 == 0:
                    current_radius = glow_size
                else:
                    current_radius = int(glow_size * 0.4)
                x_point = int(center_x + current_radius * math.cos(angle))
                y_point = int(star_y + current_radius * math.sin(angle))
                glow_points.append([x_point, y_point])

            glow_points = np.array(glow_points, np.int32)
            cv2.fillPoly(overlay, [glow_points], gold_bright)
            cv2.addWeighted(overlay, 0.12 - (i * 0.015), frame, 1.0 - (0.12 - (i * 0.015)), 0, frame)

        # Draw main star (golden)
        cv2.fillPoly(frame, [star_points], gold_bright)

        # Add darker gold outline for definition
        cv2.polylines(frame, [star_points], True, gold_dark, 2, cv2.LINE_AA)

        # Add bright center highlight (white-gold)
        center_highlight_size = int(radius * 0.2)
        cv2.circle(frame, (center_x, star_y), center_highlight_size, gold_white, -1, cv2.LINE_AA)
        
        return frame
    
    @staticmethod
    def draw_dramatic_floor_uplight(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                                     color: Tuple[int, int, int] = (255, 255, 255),
                                     intensity: float = 1.0,
                                     radius: Optional[int] = None,
                                     player=None) -> np.ndarray:
        """
        Dramatic Floor Uplight - A powerful spotlight hitting the floor and shining upwards
        
        Creates a light source simulation with:
        - Full-screen dimming overlay for contrast
        - Perspective-flattened light patch on the floor (at player's feet)
        - Radial gradient from bright center to transparent edges
        - Strong blur and glow effects for bloom
        
        Args:
            frame: Frame to draw on (BGR format)
            bbox: Bounding box (padded)
            color: Light color (default white/warm)
            intensity: Light intensity multiplier (0.0-2.0, default 1.0)
            radius: Size of the light patch on floor (None = auto-calculate)
            player: Player object (for accessing original_bbox)
            
        Returns:
            Frame with dramatic floor uplight effect
        """
        x, y, w, h = bbox
        height, width = frame.shape[:2]
        
        # Get player's feet position (where the light source is)
        if hasattr(player, 'current_original_bbox') and player and player.current_original_bbox:
            orig_x, orig_y, orig_w, orig_h = player.current_original_bbox
            feet_x = orig_x + orig_w // 2
            feet_y = orig_y + orig_h
        else:
            feet_x = x + w // 2
            feet_y = y + h
        
        # Ensure coordinates are within bounds
        feet_x = int(np.clip(feet_x, 0, width - 1))
        feet_y = int(np.clip(feet_y, 0, height - 1))
        
        # Calculate light patch size (perspective-flattened oval)
        if radius is None:
            # Auto-calculate based on player width
            base_radius = max(int(w * 0.8), 50)
        else:
            base_radius = radius
        
        # Perspective transform: rotateX(65deg) makes it look flat on the floor
        # This means the vertical radius is much smaller than horizontal
        # The 65-degree angle means we see it from above, so it's flattened
        radius_x = base_radius  # Horizontal (full size)
        radius_y = max(int(base_radius * 0.15), 8)  # Vertical (flattened for perspective)
        
        # Create a working copy
        result = frame.copy()
        
        # Step 1: Apply global scene dimming (subtle darkening for contrast)
        # Use rgba(0,0,0,0.3) equivalent - darken by 30%
        dimming_factor = 0.7  # Keep 70% brightness = 30% darkening
        result = (result.astype(np.float32) * dimming_factor).astype(np.uint8)
        
        # Step 2: Create the light patch on the floor with radial gradient
        # Create a mask for the light patch area (larger than final size for blur)
        patch_size = int(base_radius * 2.5)  # Make it larger to accommodate blur
        patch_half = patch_size // 2
        
        # Create coordinate grids for the patch
        y_coords, x_coords = np.ogrid[:patch_size, :patch_size]
        center_y_patch = patch_half
        center_x_patch = patch_half
        
        # Calculate distance from center (elliptical for perspective)
        dx = (x_coords - center_x_patch) / radius_x
        dy = (y_coords - center_y_patch) / radius_y
        distance = np.sqrt(dx * dx + dy * dy)
        
        # Create radial gradient mask (bright center, transparent edges)
        # Use smooth falloff for natural look
        gradient_mask = np.clip(1.0 - distance, 0, 1)
        # Apply power curve for smoother falloff
        gradient_mask = np.power(gradient_mask, 1.5)
        
        # Apply intensity multiplier
        gradient_mask = gradient_mask * intensity
        
        # Create the light patch (warm white/white center)
        # Use warm white color: slightly warm (BGR: more blue/green, less red)
        warm_white = (200, 240, 255)  # Slightly warm white in BGR
        light_patch = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
        for c in range(3):
            light_patch[:, :, c] = warm_white[c] * gradient_mask
        
        # Step 3: Apply strong blur for bloom effect (20px blur)
        blur_size = 21  # Must be odd for Gaussian blur
        light_patch_blurred = cv2.GaussianBlur(light_patch.astype(np.uint8), (blur_size, blur_size), 20)
        
        # Step 4: Create upward glow effect (extends above the floor patch)
        # The glow should extend upward from the floor patch
        glow_height = int(base_radius * 2.5)  # How high the glow extends
        glow_width = int(base_radius * 1.8)   # Width of the glow cone
        
        # Create a vertical gradient mask for upward glow
        glow_mask = np.zeros((glow_height, glow_width, 3), dtype=np.float32)
        glow_y_coords, glow_x_coords = np.ogrid[:glow_height, :glow_width]
        
        # Horizontal falloff (narrower at top, wider at bottom)
        top_width = int(glow_width * 0.3)
        bottom_width = glow_width
        width_at_y = top_width + (bottom_width - top_width) * (glow_y_coords / max(glow_height - 1, 1))
        dist_from_center = np.abs(glow_x_coords - glow_width // 2)
        horizontal_falloff = np.clip(1.0 - (dist_from_center / (width_at_y / 2 + 1)), 0, 1)
        
        # Vertical falloff (brighter at bottom, fades upward)
        vertical_falloff = 1.0 - (glow_y_coords / max(glow_height - 1, 1))
        vertical_falloff = np.power(vertical_falloff, 1.2)  # Smooth fade
        
        # Combine falloffs
        glow_intensity = horizontal_falloff * vertical_falloff * intensity * 0.6  # 60% of patch brightness
        
        # Create glow patch
        for c in range(3):
            glow_mask[:, :, c] = warm_white[c] * glow_intensity
        
        # Blur the glow for smoothness
        glow_blurred = cv2.GaussianBlur(glow_mask.astype(np.uint8), (15, 15), 10)
        
        # Step 5: Composite everything onto the frame
        # First, place the upward glow (above the floor patch)
        glow_start_y = max(0, feet_y - glow_height)
        glow_start_x = max(0, feet_x - glow_width // 2)
        glow_end_y = feet_y
        glow_end_x = min(width, feet_x + glow_width // 2)
        
        # Calculate source region for glow
        glow_src_y_start = max(0, -(feet_y - glow_height))
        glow_src_x_start = max(0, -(feet_x - glow_width // 2))
        glow_src_y_end = glow_src_y_start + (glow_end_y - glow_start_y)
        glow_src_x_end = glow_src_x_start + (glow_end_x - glow_start_x)
        
        if glow_start_y < glow_end_y and glow_start_x < glow_end_x:
            glow_region = result[glow_start_y:glow_end_y, glow_start_x:glow_end_x]
            glow_src = glow_blurred[glow_src_y_start:glow_src_y_end, glow_src_x_start:glow_src_x_end]
            
            if glow_region.shape == glow_src.shape:
                # Blend glow onto frame (additive blending for light effect)
                glow_alpha = 0.4  # Glow opacity
                result[glow_start_y:glow_end_y, glow_start_x:glow_end_x] = cv2.addWeighted(
                    glow_region, 1.0 - glow_alpha,
                    glow_src, glow_alpha,
                    0
                )
        
        # Then, place the floor light patch (at feet position)
        patch_start_y = max(0, feet_y - patch_half)
        patch_start_x = max(0, feet_x - patch_half)
        patch_end_y = min(height, feet_y + patch_half)
        patch_end_x = min(width, feet_x + patch_half)
        
        # Calculate source region for patch
        patch_src_y_start = max(0, -(feet_y - patch_half))
        patch_src_x_start = max(0, -(feet_x - patch_half))
        patch_src_y_end = patch_src_y_start + (patch_end_y - patch_start_y)
        patch_src_x_end = patch_src_x_start + (patch_end_x - patch_start_x)
        
        if patch_start_y < patch_end_y and patch_start_x < patch_end_x:
            patch_region = result[patch_start_y:patch_end_y, patch_start_x:patch_end_x]
            patch_src = light_patch_blurred[patch_src_y_start:patch_src_y_end, patch_src_x_start:patch_src_x_end]
            
            if patch_region.shape == patch_src.shape:
                # Brighten the area where the light patch is (additive for light)
                patch_alpha = 0.7  # Patch opacity
                result[patch_start_y:patch_end_y, patch_start_x:patch_end_x] = cv2.addWeighted(
                    patch_region, 1.0 - patch_alpha,
                    patch_src, patch_alpha,
                    0
                )
        
        # Step 6: Brighten the player area (illuminate from below)
        # Create a mask that brightens the area around the player
        player_brighten_mask = np.zeros((height, width), dtype=np.float32)
        player_y_coords, player_x_coords = np.ogrid[:height, :width]
        
        # Brighten area above the light source
        player_center_x = feet_x
        player_top_y = max(0, feet_y - int(h * 1.5))  # Extend upward
        
        # Create elliptical brightening area
        brighten_radius_x = int(w * 1.2)
        brighten_radius_y = int(h * 1.5)
        
        dx_player = (player_x_coords - player_center_x) / brighten_radius_x
        dy_player = (player_y_coords - (feet_y - brighten_radius_y // 2)) / brighten_radius_y
        player_distance = np.sqrt(dx_player * dx_player + dy_player * dy_player)
        
        # Only brighten area above the light source
        player_mask = np.clip(1.0 - player_distance, 0, 1)
        player_mask = np.power(player_mask, 2.0)  # Smooth falloff
        player_mask = np.where(player_y_coords <= feet_y, player_mask, 0)  # Only above feet
        player_brighten_mask = player_mask * intensity * 0.3  # 30% brightness boost
        
        # Apply player brightening
        brighten_3d = np.stack([player_brighten_mask, player_brighten_mask, player_brighten_mask], axis=2)
        result = np.clip(result.astype(np.float32) * (1.0 + brighten_3d), 0, 255).astype(np.uint8)
        
        return result