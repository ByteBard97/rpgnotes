"""
GUI module for RPG Notes Automator.

This module provides the graphical user interface for the application using PyQt6.
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QTreeView, QFormLayout, QLineEdit, QComboBox,
    QStatusBar, QToolBar, QStyle, QFrame, QCheckBox,
    QProgressBar
)
from PyQt6.QtCore import Qt, QSettings, QSize
from PyQt6.QtGui import QFileSystemModel, QAction

from config import (
    OUTPUT_DIR, TEMP_DIR, CHAT_LOG_SOURCE_DIR, 
    AUDIO_SOURCE_DIR, CONTEXT_DIR
)

class FileDropFrame(QFrame):
    """A custom frame that accepts file drops."""
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        self.setMinimumHeight(100)
        
        # Add label
        layout = QVBoxLayout(self)
        self.label = QLabel("Drop audio or chat files here")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        
    def dragEnterEvent(self, event):
        """Handle drag enter events."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event):
        """Handle file drop events."""
        for url in event.mimeData().urls():
            # TODO: Process dropped files
            print(f"File dropped: {url.toLocalFile()}")

class MainWindow(QMainWindow):
    """Main window of the RPG Notes Automator application."""
    
    def __init__(self):
        super().__init__()
        self.settings = QSettings("RPGNotes", "Automator")
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("RPG Notes Automator")
        self.setMinimumSize(1000, 700)
        
        # Create toolbar
        self.create_toolbar()
        
        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create file management section
        file_section = self.create_file_section()
        main_layout.addWidget(file_section)
        
        # Create right panel (config + preview)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Add configuration section
        config_section = self.create_config_section()
        right_layout.addWidget(config_section)
        
        # Add preview section
        preview_section = self.create_preview_section()
        right_layout.addWidget(preview_section)
        
        main_layout.addWidget(right_panel)
        
        # Set layout proportions
        main_layout.setStretch(0, 2)  # File section takes 2/3
        main_layout.setStretch(1, 1)  # Right panel takes 1/3
        
    def create_toolbar(self):
        """Create the main toolbar."""
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)
        
        # Add actions
        new_action = QAction("New Session", self)
        new_action.setStatusTip("Start a new session")
        toolbar.addAction(new_action)
        
        process_action = QAction("Process Files", self)
        process_action.setStatusTip("Process selected files")
        toolbar.addAction(process_action)
        
        toolbar.addSeparator()
        
        settings_action = QAction("Settings", self)
        settings_action.setStatusTip("Open settings dialog")
        toolbar.addAction(settings_action)
        
    def create_file_section(self):
        """Create the file management section."""
        section = QWidget()
        layout = QVBoxLayout(section)
        
        # Add drop zone
        self.drop_zone = FileDropFrame()
        layout.addWidget(self.drop_zone)
        
        # Add file tree view with filters
        self.file_tree = QTreeView()
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(str(OUTPUT_DIR))
        self.file_model.setNameFilters(["*.txt", "*.md", "*.mp3", "*.wav"])
        self.file_model.setNameFilterDisables(False)
        self.file_tree.setModel(self.file_model)
        self.file_tree.setRootIndex(self.file_model.index(str(OUTPUT_DIR)))
        self.file_tree.setColumnWidth(0, 250)  # Name column
        self.file_tree.setColumnWidth(1, 100)  # Size column
        layout.addWidget(self.file_tree)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        return section
        
    def create_config_section(self):
        """Create the configuration section."""
        section = QWidget()
        layout = QFormLayout(section)
        
        # Add directory selection fields with browse buttons
        output_layout = QHBoxLayout()
        self.output_dir = QLineEdit(str(OUTPUT_DIR))
        self.output_dir.setReadOnly(True)
        output_browse = QPushButton("Browse...")
        output_browse.clicked.connect(lambda: self.browse_directory(self.output_dir))
        output_layout.addWidget(self.output_dir)
        output_layout.addWidget(output_browse)
        layout.addRow("Output Directory:", output_layout)
        
        temp_layout = QHBoxLayout()
        self.temp_dir = QLineEdit(str(TEMP_DIR))
        self.temp_dir.setReadOnly(True)
        temp_browse = QPushButton("Browse...")
        temp_browse.clicked.connect(lambda: self.browse_directory(self.temp_dir))
        temp_layout.addWidget(self.temp_dir)
        temp_layout.addWidget(temp_browse)
        layout.addRow("Temp Directory:", temp_layout)
        
        # Add API key input
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addRow("API Key:", self.api_key)
        
        # Add model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["gemini-1.5-pro", "gemini-1.0-pro"])
        layout.addRow("Model:", self.model_combo)
        
        # Add audio quality settings
        self.audio_quality = QComboBox()
        self.audio_quality.addItems(["High", "Medium", "Low"])
        layout.addRow("Audio Quality:", self.audio_quality)
        
        # Add delete temp files option
        self.delete_temp = QCheckBox("Delete temporary files after processing")
        layout.addRow("", self.delete_temp)
        
        return section
        
    def create_preview_section(self):
        """Create the markdown preview section."""
        section = QWidget()
        layout = QVBoxLayout(section)
        
        # Add preview label
        preview_label = QLabel("Preview")
        layout.addWidget(preview_label)
        
        # Add preview text area (placeholder for now)
        preview_area = QLabel("Markdown preview will appear here...")
        preview_area.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        preview_area.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        preview_area.setWordWrap(True)
        layout.addWidget(preview_area)
        
        return section
        
    def browse_directory(self, line_edit):
        """Open directory browser dialog."""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            line_edit.setText(directory)
            
    def process_files(self):
        """Process the selected files."""
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.statusBar.showMessage("Processing files...")
        # TODO: Implement actual file processing
        self.progress_bar.setValue(100)
        self.statusBar.showMessage("Processing complete")

def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 