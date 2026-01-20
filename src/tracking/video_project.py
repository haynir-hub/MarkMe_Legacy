"""
Video Project - Represents a single video with its tracking data
"""
from typing import List, Dict, Optional, Tuple
from enum import Enum
from .tracker_manager import TrackerManager, PlayerData


class ProjectStatus(Enum):
    """Status of a video project"""
    PENDING = "pending"           # Video loaded, no players marked
    MARKED = "marked"             # Players marked, ready for tracking
    TRACKING = "tracking"         # Currently tracking
    TRACKED = "tracked"           # Tracking complete, ready for export
    EXPORTING = "exporting"       # Currently exporting
    EXPORTED = "exported"         # Export complete
    FAILED = "failed"             # Tracking or export failed
    SKIPPED = "skipped"           # Skipped (no players)


class VideoProject:
    """Represents a single video project with tracking data"""
    
    def __init__(self, video_path: str):
        """
        Initialize video project
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.tracker_manager = TrackerManager()
        self.status = ProjectStatus.PENDING
        self.error_message: Optional[str] = None
        self.output_path: Optional[str] = None
        
        # Video metadata
        self.metadata: Optional[Dict] = None
        self.is_loaded = False
        
        # Trim range - for tracking only part of video
        self.trim_start_frame: Optional[int] = None  # None = start from beginning
        self.trim_end_frame: Optional[int] = None    # None = track to end
    
    def load_video(self) -> bool:
        """
        Load video and probe metadata
        
        Returns:
            True if loaded successfully
        """
        try:
            # Probe metadata
            self.metadata = self.tracker_manager.probe_video(self.video_path)
            if self.metadata is None:
                self.error_message = "Failed to read video metadata"
                self.status = ProjectStatus.FAILED
                return False
            
            # Load video
            success = self.tracker_manager.load_video(self.video_path, self.metadata)
            if not success:
                self.error_message = "Failed to load video"
                self.status = ProjectStatus.FAILED
                return False
            
            self.is_loaded = True
            self.status = ProjectStatus.PENDING
            return True
        except Exception as e:
            self.error_message = f"Error loading video: {str(e)}"
            self.status = ProjectStatus.FAILED
            return False
    
    def add_player(self, name: str, marker_style: str,
                   initial_frame: int, bbox: Tuple[int, int, int, int],
                   original_bbox: Optional[Tuple[int, int, int, int]] = None) -> int:
        """Add player to track"""
        player_id = self.tracker_manager.add_player(name, marker_style, initial_frame, bbox, original_bbox)
        
        # Update status to marked if we have players
        if len(self.tracker_manager.players) > 0:
            self.status = ProjectStatus.MARKED
        
        return player_id
    
    def get_players(self) -> List[PlayerData]:
        """Get all players"""
        return self.tracker_manager.get_all_players()
    
    def has_players(self) -> bool:
        """Check if project has any players marked"""
        return len(self.tracker_manager.players) > 0
    
    def reset_tracking(self):
        """Reset tracking data"""
        self.tracker_manager.tracking_results.clear()
        for player in self.tracker_manager.players.values():
            player.tracker.reset()
        if self.has_players():
            self.status = ProjectStatus.MARKED
        else:
            self.status = ProjectStatus.PENDING
    
    def release(self):
        """Release video resources"""
        self.tracker_manager.release()
    
    def get_display_name(self) -> str:
        """Get display name for UI"""
        import os
        filename = os.path.basename(self.video_path)
        status_icon = {
            ProjectStatus.PENDING: "⏸️",
            ProjectStatus.MARKED: "✏️",
            ProjectStatus.TRACKING: "🔄",
            ProjectStatus.TRACKED: "✅",
            ProjectStatus.EXPORTING: "📤",
            ProjectStatus.EXPORTED: "🎬",
            ProjectStatus.FAILED: "❌",
            ProjectStatus.SKIPPED: "⏭️"
        }.get(self.status, "❓")
        
        player_count = len(self.tracker_manager.players)
        return f"{status_icon} {filename} ({player_count} players)"
    
    def get_info_text(self) -> str:
        """Get detailed info text for UI"""
        if not self.metadata:
            return "Not loaded"
        
        width = int(self.metadata.get('width', 0))
        height = int(self.metadata.get('height', 0))
        fps = self.metadata.get('fps', 0)
        duration = self.metadata.get('duration', 0)
        
        info = f"Resolution: {width}x{height}\n"
        info += f"FPS: {fps:.2f}\n"
        info += f"Duration: {duration:.1f}s\n"
        info += f"Players: {len(self.tracker_manager.players)}\n"
        info += f"Status: {self.status.value}"
        
        if self.error_message:
            info += f"\nError: {self.error_message}"
        
        return info


