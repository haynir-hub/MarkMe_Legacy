"""
Batch Preview Dialog - Shows previews for multiple videos with approval checkboxes
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QCheckBox, QListWidget, QListWidgetItem,
                             QSizePolicy, QGroupBox, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import List

from ..tracking.video_project import VideoProject
from .preview_dialog import PreviewDialog


class ProjectPreviewItem(QListWidgetItem):
    """List item for project with approval checkbox"""
    
    def __init__(self, project: VideoProject, parent=None):
        super().__init__(parent)
        self.project = project
        self.approved = False
        self._update_text()
    
    def _update_text(self):
        """Update display text"""
        checkbox = "☑" if self.approved else "☐"
        status = "✅ APPROVED" if self.approved else "⏸ Pending Review"
        
        self.setText(
            f"{checkbox} {self.project.get_display_name()}\n"
            f"   Status: {status}\n"
            f"   Players: {len(self.project.get_players())}"
        )
        
        # Color based on approval
        if self.approved:
            self.setForeground(Qt.GlobalColor.darkGreen)
        else:
            self.setForeground(Qt.GlobalColor.black)
    
    def set_approved(self, approved: bool):
        """Set approval status"""
        self.approved = approved
        self._update_text()
    
    def is_approved(self) -> bool:
        """Check if approved"""
        return self.approved


class BatchPreviewDialog(QDialog):
    """Dialog for previewing multiple videos before batch export"""
    
    # Signal with list of approved projects
    export_approved = pyqtSignal(list)
    
    def __init__(self, projects: List[VideoProject], parent=None):
        super().__init__(parent)
        self.projects = projects
        self.project_items = {}  # project -> ProjectPreviewItem
        
        self.setWindowTitle(f"Batch Preview - {len(projects)} Videos")
        self.setMinimumSize(800, 600)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup UI components"""
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel(
            "📺 Review each video's tracking preview\n"
            "✅ Check the box to approve videos for export\n"
            "Only approved videos will be exported!"
        )
        instructions.setStyleSheet("font-size: 13px; padding: 10px; background-color: #E3F2FD;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Project list
        list_group = QGroupBox("Videos")
        list_layout = QVBoxLayout()
        
        self.project_list = QListWidget()
        self.project_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        list_layout.addWidget(self.project_list)
        
        # Populate list
        for project in self.projects:
            item = ProjectPreviewItem(project, self.project_list)
            self.project_items[project] = item
        
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        # Buttons for selected project
        project_buttons_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton("🎬 Preview Selected Video")
        self.preview_btn.clicked.connect(self._preview_selected)
        project_buttons_layout.addWidget(self.preview_btn)
        
        self.toggle_approval_btn = QPushButton("☑ Toggle Approval")
        self.toggle_approval_btn.clicked.connect(self._toggle_selected_approval)
        project_buttons_layout.addWidget(self.toggle_approval_btn)
        
        layout.addLayout(project_buttons_layout)
        
        # Quick actions
        quick_actions_layout = QHBoxLayout()
        
        self.approve_all_btn = QPushButton("✅ Approve All")
        self.approve_all_btn.clicked.connect(self._approve_all)
        quick_actions_layout.addWidget(self.approve_all_btn)
        
        self.reject_all_btn = QPushButton("❌ Reject All")
        self.reject_all_btn.clicked.connect(self._reject_all)
        quick_actions_layout.addWidget(self.reject_all_btn)
        
        layout.addLayout(quick_actions_layout)
        
        # Summary
        self.summary_label = QLabel()
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.summary_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.summary_label)
        
        # Export buttons
        buttons_layout = QHBoxLayout()
        
        self.cancel_btn = QPushButton("❌ Cancel Batch Export")
        self.cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(self.cancel_btn)
        
        self.export_btn = QPushButton("📤 Export Approved Videos")
        self.export_btn.clicked.connect(self._on_export_clicked)
        self.export_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        buttons_layout.addWidget(self.export_btn)
        
        layout.addLayout(buttons_layout)
        
        # Update summary AFTER export_btn is created
        self._update_summary()
        
        self.setLayout(layout)
    
    def _preview_selected(self):
        """Preview selected video"""
        current_item = self.project_list.currentItem()
        if not isinstance(current_item, ProjectPreviewItem):
            QMessageBox.information(self, "No Selection", "Please select a video to preview.")
            return
        
        self._show_preview(current_item)
    
    def _on_item_double_clicked(self, item):
        """Handle double-click on item"""
        if isinstance(item, ProjectPreviewItem):
            self._show_preview(item)
    
    def _show_preview(self, item: ProjectPreviewItem):
        """Show preview dialog for project"""
        project = item.project
        
        preview = PreviewDialog(project.tracker_manager, project.video_path, self)
        
        def on_export_approved():
            # User approved in preview
            item.set_approved(True)
            self._update_summary()
        
        preview.export_approved.connect(on_export_approved)
        preview.exec()
        
        # Update approval based on preview result
        if preview.is_approved():
            item.set_approved(True)
            self._update_summary()
    
    def _toggle_selected_approval(self):
        """Toggle approval for selected item"""
        current_item = self.project_list.currentItem()
        if not isinstance(current_item, ProjectPreviewItem):
            return
        
        current_item.set_approved(not current_item.is_approved())
        self._update_summary()
    
    def _approve_all(self):
        """Approve all videos"""
        for item in self.project_items.values():
            item.set_approved(True)
        self._update_summary()
    
    def _reject_all(self):
        """Reject all videos"""
        for item in self.project_items.values():
            item.set_approved(False)
        self._update_summary()
    
    def _update_summary(self):
        """Update summary label"""
        approved_count = sum(1 for item in self.project_items.values() if item.is_approved())
        total_count = len(self.project_items)
        
        self.summary_label.setText(
            f"✅ Approved: {approved_count} / {total_count} videos"
        )
        
        # Enable/disable export button (only if it exists)
        if hasattr(self, 'export_btn'):
            self.export_btn.setEnabled(approved_count > 0)
            
            if approved_count > 0:
                self.export_btn.setStyleSheet(
                    "background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px;"
                )
            else:
                self.export_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
    
    def _on_export_clicked(self):
        """Handle export button click"""
        approved_projects = [
            item.project
            for item in self.project_items.values()
            if item.is_approved()
        ]
        
        if not approved_projects:
            QMessageBox.warning(
                self,
                "No Approved Videos",
                "Please approve at least one video before exporting."
            )
            return
        
        # Confirm
        reply = QMessageBox.question(
            self,
            "Confirm Export",
            f"Export {len(approved_projects)} approved video(s)?\n\n"
            f"Unapproved videos will be skipped.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.export_approved.emit(approved_projects)
            self.accept()
    
    def get_approved_projects(self) -> List[VideoProject]:
        """Get list of approved projects"""
        return [
            item.project
            for item in self.project_items.values()
            if item.is_approved()
        ]

