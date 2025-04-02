"""
GUI module for RPG Notes Automator.

This module provides the graphical user interface for the application using PyQt6.
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QTreeView, QFormLayout, QLineEdit, QComboBox
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QFileSystemModel

from config import (
    OUTPUT_DIR, TEMP_DIR, CHAT_LOG_SOURCE_DIR, 
    AUDIO_SOURCE_DIR, CONTEXT_DIR
)

class MainWindow(QMainWindow):
    """Main window of the RPG Notes Automator application."""
    
    def __init__(self):
        super().__init__()
        self.settings = QSettings("RPGNotes", "Automator")
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("RPG Notes Automator")
        self.setMinimumSize(800, 600)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create file management section
        file_section = self.create_file_section()
        main_layout.addWidget(file_section)
        
        # Create configuration section
        config_section = self.create_config_section()
        main_layout.addWidget(config_section)
        
        # Set layout proportions
        main_layout.setStretch(0, 2)  # File section takes 2/3
        main_layout.setStretch(1, 1)  # Config section takes 1/3
        
    def create_file_section(self):
        """Create the file management section."""
        section = QWidget()
        layout = QVBoxLayout(section)
        
        # Add file tree view
        self.file_tree = QTreeView()
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(str(OUTPUT_DIR))
        self.file_tree.setModel(self.file_model)
        self.file_tree.setRootIndex(self.file_model.index(str(OUTPUT_DIR)))
        layout.addWidget(self.file_tree)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        process_btn = QPushButton("Process Files")
        process_btn.clicked.connect(self.process_files)
        button_layout.addWidget(process_btn)
        
        layout.addLayout(button_layout)
        
        return section
        
    def create_config_section(self):
        """Create the configuration section."""
        section = QWidget()
        layout = QFormLayout(section)
        
        # Add directory selection fields
        self.output_dir = QLineEdit(str(OUTPUT_DIR))
        self.output_dir.setReadOnly(True)
        layout.addRow("Output Directory:", self.output_dir)
        
        self.temp_dir = QLineEdit(str(TEMP_DIR))
        self.temp_dir.setReadOnly(True)
        layout.addRow("Temp Directory:", self.temp_dir)
        
        # Add model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["gemini-1.5-pro", "gemini-1.0-pro"])
        layout.addRow("Model:", self.model_combo)
        
        return section
        
    def process_files(self):
        """Process the selected files."""
        # TODO: Implement file processing logic
        pass

def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 