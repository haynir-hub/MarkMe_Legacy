"""
Player Selector - Dialog for selecting player marker style and name
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QComboBox, QGroupBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor


class PlayerSelector(QDialog):
    """Dialog for configuring player marker"""
    
    # Signal emitted when player is confirmed
    player_confirmed = pyqtSignal(str, str)  # name, style
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Player Marker")
        self.setMinimumWidth(300)
        
        layout = QVBoxLayout()
        
        # Player name
        name_group = QGroupBox("Player Name (Optional)")
        name_layout = QVBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter player name...")
        name_layout.addWidget(self.name_input)
        name_group.setLayout(name_layout)
        layout.addWidget(name_group)
        
        # Marker style
        style_group = QGroupBox("Marker Style")
        style_layout = QVBoxLayout()
        self.style_combo = QComboBox()
        self.style_combo.addItems([
            "🏀 NBA Iso Ring (broadcast floor glow)",
            "〽️ Floating Chevron (FIFA-style)",
            "💡 Spotlight (alien beam)",
            "🎯 Tactical Crosshair",
            "🔲 Tactical Brackets",
            "🌊 Sonar Ripple",
            "✨ Dramatic Floor Uplight (cinematic lighting)"
        ])
        
        # Add description label that updates based on selection
        self.style_description = QLabel()
        self.style_description.setWordWrap(True)
        self.style_description.setStyleSheet("color: gray; font-size: 10px;")
        self.style_combo.currentIndexChanged.connect(self._update_description)
        self._update_description()  # Set initial description
        
        style_layout.addWidget(self.style_combo)
        style_layout.addWidget(self.style_description)
        style_group.setLayout(style_layout)
        layout.addWidget(style_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.confirm_btn = QPushButton("Confirm")
        self.confirm_btn.clicked.connect(self._on_confirm)
        self.confirm_btn.setDefault(True)
        
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.confirm_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _update_description(self):
        """Update style description based on selection"""
        descriptions = {
            0: "טבעת זוהרת בסגנון 2K",
            1: "חץ תלת-ממדי מרחף",
            2: "Light column / alien beam - custom spotlight",
            3: "Tactical scope crosshair with clear center",
            4: "סוגריים טקטיים נושמים בסגנון אנליטי",
            5: "גלי סונאר שטוחים על הרצפה",
            6: "Powerful floor spotlight with upward glow - cinematic dramatic lighting effect"
        }
        desc = descriptions.get(self.style_combo.currentIndex(), "")
        self.style_description.setText(desc)
    
    def _on_confirm(self):
        """Handle confirm button click"""
        name = self.name_input.text().strip()
        if not name:
            name = f"Player {id(self)}"  # Default name
        
        style_map = {
            0: "nba_iso_ring",
            1: "floating_chevron",
            2: "spotlight_modern",
            3: "crosshair",
            4: "tactical_brackets",
            5: "sonar_ripple",
            6: "dramatic_floor_uplight"
        }
        style = style_map.get(self.style_combo.currentIndex(), "nba_iso_ring")
        
        self.player_confirmed.emit(name, style)
        self.accept()
    
    def get_selected_style(self) -> str:
        """Get selected marker style"""
        style_map = {
            0: "nba_iso_ring",
            1: "floating_chevron",
            2: "spotlight_modern",
            3: "crosshair",
            4: "tactical_brackets",
            5: "sonar_ripple",
            6: "dramatic_floor_uplight"
        }
        return style_map.get(self.style_combo.currentIndex(), "nba_iso_ring")
