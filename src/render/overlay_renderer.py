import cv2
import numpy as np
from typing import Tuple, Optional, List
from .modern_styles import ModernStyles
from .player_segmentation import get_segmenter, PlayerSegmentation
from .team_manager import get_team_manager, TeamManager


class OverlayRenderer:
    def __init__(self, use_segmentation: bool = False):
        self.modern_styles = ModernStyles()
        self.frame_count = 0
        self.use_segmentation = use_segmentation
        self._segmenter: Optional[PlayerSegmentation] = None
        self._team_manager: Optional[TeamManager] = None
        self._all_players: List = []
        self._current_frame_idx: int = 0  # Track current frame for radar keyframe interpolation

    @property
    def team_manager(self) -> TeamManager:
        if self._team_manager is None:
            self._team_manager = get_team_manager()
        return self._team_manager

    @property
    def segmenter(self) -> Optional[PlayerSegmentation]:
        if self._segmenter is None and self.use_segmentation:
            self._segmenter = get_segmenter()
        return self._segmenter

    def draw_marker(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                    marker_style: str, color: Tuple[int, int, int],
                    player=None, use_segmentation: bool = True) -> np.ndarray:
        if bbox is None:
            return frame
        self.frame_count += 1

        floor_marker_styles = {'solid_anchor', 'radar_defensive', 'dynamic_ring_3d'}

        if (use_segmentation and self.use_segmentation and
            marker_style in floor_marker_styles and self.segmenter and self.segmenter.enabled):

            def draw_func(f):
                if marker_style == 'dynamic_ring_3d':
                    return self.modern_styles.draw_dynamic_ring_3d(
                        f, bbox, (255, 0, 180), self.frame_count, player, full_ring=True)
                return self._draw_marker_internal(f, bbox, marker_style, color, player)

            return self.segmenter.render_with_segmentation(frame, bbox, draw_func)

        return self._draw_marker_internal(frame, bbox, marker_style, color, player)

    def _draw_marker_internal(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                              marker_style: str, color: Tuple[int, int, int],
                              player=None) -> np.ndarray:
        if marker_style == 'dynamic_ring_3d':
            return self.modern_styles.draw_dynamic_ring_3d(
                frame, bbox, (255, 0, 180), self.frame_count, player)

        elif marker_style == 'spotlight_alien':
            darkened = (frame.astype(np.float32) * 0.3).astype(np.uint8)
            mask = self.modern_styles.get_alien_spotlight_mask(frame.shape, bbox)
            frame = self.modern_styles.apply_alien_spotlight(frame, darkened, mask)
            frame = self.modern_styles.draw_alien_spotlight_floor(
                frame, bbox, (200, 255, 255), self.frame_count)
            return frame

        elif marker_style == 'solid_anchor':
            return self.modern_styles.draw_solid_anchor(frame, bbox, (0, 255, 100), player)

        elif marker_style == 'radar_defensive':
            # Check if player has manual radar keyframes
            manual_angle = None
            manual_size = None
            if player and hasattr(player, 'has_radar_keyframes') and player.has_radar_keyframes():
                # Get interpolated radar params for current frame
                radar_params = player.get_radar_params_at_frame(self._current_frame_idx)
                if radar_params:
                    manual_angle, manual_size = radar_params

            # Fall back to automatic targeting only if no manual keyframes
            target_pos = None
            if manual_angle is None:
                if player and hasattr(player, 'player_id') and self._all_players:
                    target_pos = self.team_manager.find_nearest_opponent_from_players(
                        player, self._all_players)

            # Get radar color from keyframes (green by default, can be switched to red)
            radar_color = (0, 255, 100)  # Default green
            if player and hasattr(player, 'get_radar_color_at_frame'):
                radar_color = player.get_radar_color_at_frame(self._current_frame_idx)

            return self.modern_styles.draw_defensive_radar(
                frame, bbox, radar_color, player, target_pos,
                manual_angle=manual_angle, manual_size=manual_size,
                frame_count=self.frame_count)

        elif marker_style == 'sniper_scope':
            return self.modern_styles.draw_sniper_scope(
                frame, bbox, (0, 0, 255), player, self.frame_count)

        elif marker_style == 'ball_marker':
            return self.modern_styles.draw_ball_marker(
                frame, bbox, (0, 165, 255), player, self.frame_count)

        elif marker_style == 'fireball_trail':
            return self.modern_styles.draw_fireball_trail(
                frame, bbox, (0, 100, 255), player, self.frame_count)

        elif marker_style == 'energy_rings':
            return self.modern_styles.draw_energy_rings(
                frame, bbox, (255, 200, 0), player, self.frame_count)

        else:
            return self._draw_rectangle(frame, bbox, color)

    def draw_all_markers(self, frame: np.ndarray, players_data: list,
                         frame_idx: Optional[int] = None,
                         tracking_start_frame: Optional[int] = None,
                         tracking_end_frame: Optional[int] = None) -> np.ndarray:
        result_frame = frame.copy()
        self._all_players = players_data
        # Update current frame index for radar keyframe interpolation
        if frame_idx is not None:
            self._current_frame_idx = frame_idx

        visible_players = []
        for p in players_data:
            if hasattr(p, 'current_bbox') and p.current_bbox is not None:
                should_draw = True
                if frame_idx is not None:
                    # Check player-specific tracking range first
                    if hasattr(p, 'is_visible_at_frame'):
                        should_draw = p.is_visible_at_frame(
                            frame_idx, 
                            global_start=tracking_start_frame or 0,
                            global_end=tracking_end_frame
                        )
                    else:
                        # Fallback to global range check
                        if tracking_start_frame is not None and frame_idx < tracking_start_frame:
                            should_draw = False
                        if tracking_end_frame is not None and frame_idx > tracking_end_frame:
                            should_draw = False
                if should_draw:
                    visible_players.append(p)

        alien_players = [p for p in visible_players if p.marker_style == 'spotlight_alien']

        if alien_players:
            darkened_frame = (result_frame.astype(np.float32) * 0.3).astype(np.uint8)
            combined_mask = np.zeros(result_frame.shape[:2], dtype=np.float32)

            for p in alien_players:
                mask = self.modern_styles.get_alien_spotlight_mask(
                    result_frame.shape, p.current_bbox)
                combined_mask = np.maximum(combined_mask, mask)

            result_frame = self.modern_styles.apply_alien_spotlight(
                result_frame, darkened_frame, combined_mask)

            for p in alien_players:
                result_frame = self.modern_styles.draw_alien_spotlight_floor(
                    result_frame, p.current_bbox, p.color, self.frame_count)

        for p in visible_players:
            if p.marker_style != 'spotlight_alien':
                result_frame = self.draw_marker(
                    result_frame, p.current_bbox, p.marker_style, p.color, p)

        return result_frame

    def _draw_rectangle(self, frame, bbox, color):
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w + 2, y + h + 2), color, 2)
        return frame

    def assign_player_to_team(self, player_id: int, team: str) -> None:
        self.team_manager.assign_team(player_id, team)

    def assign_team_a(self, player_ids: List[int]) -> None:
        for pid in player_ids:
            self.team_manager.assign_team(pid, TeamManager.TEAM_A)

    def assign_team_b(self, player_ids: List[int]) -> None:
        for pid in player_ids:
            self.team_manager.assign_team(pid, TeamManager.TEAM_B)

    def clear_team_assignments(self) -> None:
        self.team_manager.clear_assignments()

    def get_player_team(self, player_id: int) -> Optional[str]:
        return self.team_manager.get_team(player_id)
