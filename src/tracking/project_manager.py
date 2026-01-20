"""
Project Manager - Manages multiple video projects for batch processing
"""
from typing import List, Optional
from .video_project import VideoProject, ProjectStatus


class ProjectManager:
    """Manages multiple video projects"""
    
    def __init__(self):
        self.projects: List[VideoProject] = []
        self.current_project_index: Optional[int] = None
    
    def add_project(self, video_path: str) -> Optional[VideoProject]:
        """
        Add a new video project
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoProject if successful, None otherwise
        """
        # Check if already exists
        for project in self.projects:
            if project.video_path == video_path:
                return None  # Already added
        
        project = VideoProject(video_path)
        if project.load_video():
            self.projects.append(project)
            return project
        else:
            return None
    
    def remove_project(self, index: int) -> bool:
        """
        Remove project at index
        
        Args:
            index: Project index
            
        Returns:
            True if removed
        """
        if 0 <= index < len(self.projects):
            project = self.projects[index]
            project.release()
            self.projects.pop(index)
            
            # Update current index
            if self.current_project_index == index:
                self.current_project_index = None
            elif self.current_project_index is not None and self.current_project_index > index:
                self.current_project_index -= 1
            
            return True
        return False
    
    def get_project(self, index: int) -> Optional[VideoProject]:
        """Get project at index"""
        if 0 <= index < len(self.projects):
            return self.projects[index]
        return None
    
    def get_current_project(self) -> Optional[VideoProject]:
        """Get currently selected project"""
        if self.current_project_index is not None:
            return self.get_project(self.current_project_index)
        return None
    
    def set_current_project(self, index: int) -> bool:
        """Set current project by index"""
        if 0 <= index < len(self.projects):
            self.current_project_index = index
            return True
        return False
    
    def get_projects_for_export(self) -> List[VideoProject]:
        """
        Get all projects that are ready for batch export
        
        Returns:
            List of projects with status MARKED or TRACKED
        """
        return [
            p for p in self.projects 
            if p.status in [ProjectStatus.MARKED, ProjectStatus.TRACKED] and p.has_players()
        ]
    
    def get_project_count(self) -> int:
        """Get total number of projects"""
        return len(self.projects)
    
    def clear_all(self):
        """Clear all projects"""
        for project in self.projects:
            project.release()
        self.projects.clear()
        self.current_project_index = None
    
    def get_summary(self) -> dict:
        """Get summary statistics"""
        summary = {
            'total': len(self.projects),
            'pending': 0,
            'marked': 0,
            'tracked': 0,
            'exported': 0,
            'failed': 0,
            'skipped': 0,
            'ready_for_export': 0
        }
        
        for project in self.projects:
            if project.status == ProjectStatus.PENDING:
                summary['pending'] += 1
            elif project.status == ProjectStatus.MARKED:
                summary['marked'] += 1
                summary['ready_for_export'] += 1
            elif project.status == ProjectStatus.TRACKED:
                summary['tracked'] += 1
                summary['ready_for_export'] += 1
            elif project.status == ProjectStatus.EXPORTED:
                summary['exported'] += 1
            elif project.status == ProjectStatus.FAILED:
                summary['failed'] += 1
            elif project.status == ProjectStatus.SKIPPED:
                summary['skipped'] += 1
        
        return summary







