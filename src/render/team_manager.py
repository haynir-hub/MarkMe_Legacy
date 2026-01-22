"""
Team Manager - Manages team assignments and finds nearest opponents.
Used for radar direction targeting.
"""
import math
from typing import Dict, List, Tuple, Optional


class TeamManager:
    """
    Manages player team assignments and provides utilities for
    finding opponents for radar targeting.
    """

    # Team identifiers
    TEAM_A = 'A'  # e.g., Defense
    TEAM_B = 'B'  # e.g., Offense
    TEAM_UNKNOWN = None

    def __init__(self):
        # Maps player_id -> team ('A', 'B', or None)
        self._player_teams: Dict[int, str] = {}

        # Optional: dominant colors for each team (BGR)
        self._team_colors: Dict[str, Tuple[int, int, int]] = {}

    def assign_team(self, player_id: int, team: str) -> None:
        """Assign a player to a team."""
        if team not in (self.TEAM_A, self.TEAM_B, self.TEAM_UNKNOWN):
            raise ValueError(f"Invalid team: {team}. Use TEAM_A, TEAM_B, or TEAM_UNKNOWN")
        self._player_teams[player_id] = team

    def get_team(self, player_id: int) -> Optional[str]:
        """Get the team assignment for a player."""
        return self._player_teams.get(player_id)

    def clear_assignments(self) -> None:
        """Clear all team assignments."""
        self._player_teams.clear()

    def get_players_by_team(self, team: str) -> List[int]:
        """Get all player IDs assigned to a team."""
        return [pid for pid, t in self._player_teams.items() if t == team]

    def set_team_color(self, team: str, color_bgr: Tuple[int, int, int]) -> None:
        """Set the dominant jersey color for a team (for future auto-detection)."""
        self._team_colors[team] = color_bgr

    def get_team_color(self, team: str) -> Optional[Tuple[int, int, int]]:
        """Get the jersey color for a team."""
        return self._team_colors.get(team)

    @staticmethod
    def get_player_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get the center point of a player's bounding box."""
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    @staticmethod
    def get_player_feet(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get the feet position (bottom center) of a player's bounding box."""
        x, y, w, h = bbox
        return (x + w // 2, y + h)

    @staticmethod
    def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def find_nearest_opponent(self, player_id: int, player_bbox: Tuple[int, int, int, int],
                              all_players: List[dict]) -> Optional[Tuple[int, int]]:
        """
        Find the nearest opponent's position for a given player.

        Args:
            player_id: The ID of the player looking for opponents
            player_bbox: The bounding box of the player
            all_players: List of all players with 'id' and 'bbox' keys

        Returns:
            (x, y) position of nearest opponent's center, or None if no opponent found
        """
        player_team = self.get_team(player_id)
        if player_team is None:
            return None

        # Determine opponent team
        opponent_team = self.TEAM_B if player_team == self.TEAM_A else self.TEAM_A

        player_pos = self.get_player_feet(player_bbox)

        nearest_pos = None
        nearest_dist = float('inf')

        for other in all_players:
            other_id = other.get('id')
            other_bbox = other.get('bbox')

            if other_id is None or other_bbox is None:
                continue

            # Check if this is an opponent
            if self.get_team(other_id) != opponent_team:
                continue

            other_pos = self.get_player_center(other_bbox)
            dist = self.distance(player_pos, other_pos)

            if dist < nearest_dist:
                nearest_dist = dist
                nearest_pos = other_pos

        return nearest_pos

    def find_nearest_opponent_from_players(self, player, all_players_objects) -> Optional[Tuple[int, int]]:
        """
        Find nearest opponent using player objects directly.

        Args:
            player: Player object with .id and .current_bbox attributes
            all_players_objects: List of all player objects

        Returns:
            (x, y) position of nearest opponent, or None
        """
        if not hasattr(player, 'id') or not hasattr(player, 'current_bbox'):
            return None

        if player.current_bbox is None:
            return None

        player_team = self.get_team(player.id)
        if player_team is None:
            return None

        opponent_team = self.TEAM_B if player_team == self.TEAM_A else self.TEAM_A
        player_pos = self.get_player_feet(player.current_bbox)

        nearest_pos = None
        nearest_dist = float('inf')

        for other in all_players_objects:
            if not hasattr(other, 'id') or not hasattr(other, 'current_bbox'):
                continue
            if other.current_bbox is None:
                continue
            if other.id == player.id:
                continue

            if self.get_team(other.id) != opponent_team:
                continue

            other_pos = self.get_player_center(other.current_bbox)
            dist = self.distance(player_pos, other_pos)

            if dist < nearest_dist:
                nearest_dist = dist
                nearest_pos = other_pos

        return nearest_pos


# Global instance for easy access
_team_manager: Optional[TeamManager] = None


def get_team_manager() -> TeamManager:
    """Get or create the global TeamManager instance."""
    global _team_manager
    if _team_manager is None:
        _team_manager = TeamManager()
    return _team_manager
