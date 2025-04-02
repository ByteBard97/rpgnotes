"""
GUI module for RPG Notes Automator.

This module provides the graphical user interface for the application using PyQt6.
"""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QTreeView, QFormLayout, QLineEdit, QComboBox,
    QStatusBar, QToolBar, QStyle, QFrame, QCheckBox,
    QProgressBar, QGroupBox, QPushButton, QMessageBox,
    QTextEdit
)
from PyQt6.QtCore import Qt, QSettings, QSize, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QFileSystemModel, QAction, QIcon
from dotenv import load_dotenv, set_key

from config import (
    CONFIG,  # Main configuration dictionary
    OUTPUT_DIR, TEMP_DIR, CHAT_LOG_SOURCE_DIR,
    AUDIO_SOURCE_DIR, CONTEXT_DIR, GEMINI_API_KEY,
    GEMINI_MODEL_NAME, DELETE_TEMP_FILES, AUDIO_QUALITY,
    CHAT_LOG_OUTPUT_DIR, TRANSCRIPTIONS_OUTPUT_DIR,
    AUDIO_OUTPUT_DIR, TEMP_TRANSCRIPTIONS, DISCORD_MAPPING_FILE,
    WHISPER_PROMPT_FILE, SUMMARY_PROMPT_FILE, DETAILS_PROMPT_FILE,
    TEMPLATE_FILE
)
from audio import AudioProcessor
from chat import ChatLogProcessor
from transcription import TranscriptionProcessor
from summary import SessionNotesGenerator
from utils import get_newest_file, load_context_files, get_previous_summary_file

# Define a worker thread for processing
class ProcessingWorker(QThread):
    """Worker thread for background processing."""
    progress_updated = pyqtSignal(int, str)
    processing_complete = pyqtSignal(bool, str)
    
    def __init__(self, source_type, parent=None):
        super().__init__(parent)
        self.source_type = source_type
        self.is_running = True
        self.session_number = None
        self.audio_processor = AudioProcessor()
        self.chat_processor = ChatLogProcessor()
        self.transcription_processor = TranscriptionProcessor()
        self.session_notes_generator = SessionNotesGenerator()
        
    def run(self):
        """Run the processing workflow."""
        try:
            # 1. Process chat log
            self.progress_updated.emit(10, "Processing chat log...")
            self.session_number = self.chat_processor.process_chat_log()
            if not self.session_number:
                self.processing_complete.emit(False, "Failed to extract session number from chat log.")
                return
                
            # 2. Extract audio files
            self.progress_updated.emit(20, "Extracting audio files...")
            self.audio_processor.unzip_audio()
            
            # 3. Transcribe audio
            self.progress_updated.emit(30, "Transcribing audio (this may take a while)...")
            self.audio_processor.transcribe_audio()
            
            # 4. Combine transcriptions
            self.progress_updated.emit(70, "Combining transcriptions...")
            transcript_file = self.transcription_processor.combine_transcriptions(self.session_number)
            if not transcript_file:
                self.processing_complete.emit(False, "Failed to combine transcriptions.")
                return
                
            # 5. Generate session notes
            self.progress_updated.emit(80, "Generating session summary...")
            session_summary, session_data = self.session_notes_generator.generate_session_notes(
                transcript_file, self.session_number
            )
            
            # 6. Save summary file
            self.progress_updated.emit(90, "Saving session notes...")
            output_file = self.session_notes_generator.save_summary_file(
                session_summary, session_data, self.session_number
            )
            
            # 7. Cleanup if configured
            if DELETE_TEMP_FILES:
                self.progress_updated.emit(95, "Cleaning up temporary files...")
                try:
                    shutil.rmtree(TEMP_DIR)
                except Exception as e:
                    print(f"Error removing temporary directory: {e}")
            
            self.progress_updated.emit(100, "Processing complete!")
            self.processing_complete.emit(True, f"Session notes saved to: {output_file}")
            
        except Exception as e:
            self.processing_complete.emit(False, f"Error during processing: {str(e)}")
            
    def stop(self):
        """Stop the processing."""
        self.is_running = False

class ProgressWidget(QGroupBox):
    """Widget for showing detailed progress information."""
    
    def __init__(self, parent=None):
        super().__init__("Processing Progress", parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the progress widget UI."""
        layout = QVBoxLayout(self)
        
        # Current task
        self.task_label = QLabel("Waiting to start...")
        layout.addWidget(self.task_label)
        
        # Progress bar with percentage
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% (%v/%m files)")
        progress_layout.addWidget(self.progress_bar, stretch=1)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        progress_layout.addWidget(self.cancel_btn)
        layout.addLayout(progress_layout)
        
        # Stats layout
        stats_layout = QHBoxLayout()
        
        # Time remaining
        time_layout = QVBoxLayout()
        self.time_remaining_label = QLabel("Time Remaining: --:--:--")
        self.elapsed_time_label = QLabel("Elapsed Time: 00:00:00")
        time_layout.addWidget(self.time_remaining_label)
        time_layout.addWidget(self.elapsed_time_label)
        stats_layout.addLayout(time_layout)
        
        # Processing speed
        speed_layout = QVBoxLayout()
        self.speed_label = QLabel("Speed: -- files/hour")
        self.files_processed_label = QLabel("Processed: 0 files")
        speed_layout.addWidget(self.speed_label)
        speed_layout.addWidget(self.files_processed_label)
        stats_layout.addLayout(speed_layout)
        
        layout.addLayout(stats_layout)
        
        # Hide by default
        self.setVisible(False)
        
        # Setup timer for elapsed time updates
        self.start_time = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_elapsed_time)
        
    def start_processing(self, total_files):
        """Start the progress tracking."""
        self.setVisible(True)
        self.progress_bar.setMaximum(total_files)
        self.progress_bar.setValue(0)
        self.start_time = datetime.now()
        self.timer.start(1000)  # Update every second
        self.cancel_btn.setVisible(True)
        
    def update_progress(self, files_processed, current_task):
        """Update the progress information."""
        self.progress_bar.setValue(files_processed)
        self.task_label.setText(f"Current Task: {current_task}")
        self.files_processed_label.setText(f"Processed: {files_processed} files")
        
        # Calculate speed and estimated time remaining
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            if elapsed.total_seconds() > 0:
                speed = files_processed / (elapsed.total_seconds() / 3600)  # files per hour
                self.speed_label.setText(f"Speed: {speed:.1f} files/hour")
                
                if files_processed > 0:
                    total = self.progress_bar.maximum()
                    remaining_files = total - files_processed
                    time_per_file = elapsed / files_processed
                    time_remaining = time_per_file * remaining_files
                    self.time_remaining_label.setText(
                        f"Time Remaining: {str(time_remaining).split('.')[0]}"
                    )
    
    def update_elapsed_time(self):
        """Update the elapsed time display."""
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            self.elapsed_time_label.setText(
                f"Elapsed Time: {str(elapsed).split('.')[0]}"
            )
    
    def finish_processing(self):
        """Clean up after processing is complete."""
        self.timer.stop()
        self.cancel_btn.setVisible(False)
        self.task_label.setText("Processing Complete!")

class FileDropFrame(QFrame):
    """A custom frame that accepts file drops."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        self.setMinimumHeight(80)  # Reduced height
        self.setStyleSheet("""
            FileDropFrame {
                border: 2px dashed #999;
                border-radius: 5px;
                background-color: #f8f9fa;
            }
            FileDropFrame:hover {
                border-color: #666;
                background-color: #e9ecef;
            }
        """)
        
        # Add label with icon
        layout = QVBoxLayout(self)
        self.label = QLabel("Drop audio or chat files here")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #666;")
        layout.addWidget(self.label)
        
        # Store reference to main window
        self.main_window = parent
        
    def dragEnterEvent(self, event):
        """Handle drag enter events."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                FileDropFrame {
                    border: 2px dashed #0d6efd;
                    border-radius: 5px;
                    background-color: #e7f1ff;
                }
            """)
            
    def dragLeaveEvent(self, event):
        """Handle drag leave events."""
        self.setStyleSheet("""
            FileDropFrame {
                border: 2px dashed #999;
                border-radius: 5px;
                background-color: #f8f9fa;
            }
            FileDropFrame:hover {
                border-color: #666;
                background-color: #e9ecef;
            }
        """)
            
    def dropEvent(self, event):
        """Handle file drop events."""
        self.setStyleSheet("""
            FileDropFrame {
                border: 2px dashed #999;
                border-radius: 5px;
                background-color: #f8f9fa;
            }
            FileDropFrame:hover {
                border-color: #666;
                background-color: #e9ecef;
            }
        """)
        
        if not self.main_window:
            return
            
        audio_files = []
        chat_files = []
        
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            self.main_window.log(f"File dropped: {file_path}")
            
            # Determine file type and add to appropriate list
            if file_path.lower().endswith((".zip", ".flac", ".wav", ".mp3")):
                audio_files.append(file_path)
            elif file_path.lower().endswith(".json"):
                chat_files.append(file_path)
            else:
                self.main_window.log(f"Unsupported file type: {file_path}", error=True)
        
        # Process files based on type
        self.process_dropped_files(audio_files, chat_files)
        
    def process_dropped_files(self, audio_files, chat_files):
        """Process the dropped files."""
        if not self.main_window:
            return
            
        # Copy chat files to chat log source directory
        if chat_files:
            try:
                for chat_file in chat_files:
                    dest_path = CHAT_LOG_SOURCE_DIR / Path(chat_file).name
                    shutil.copy2(chat_file, dest_path)
                    self.main_window.log(f"Copied chat log to: {dest_path}")
            except Exception as e:
                self.main_window.log(f"Error copying chat files: {str(e)}", error=True)
        
        # Copy audio files to audio source directory
        if audio_files:
            try:
                for audio_file in audio_files:
                    dest_path = AUDIO_SOURCE_DIR / Path(audio_file).name
                    shutil.copy2(audio_file, dest_path)
                    self.main_window.log(f"Copied audio file to: {dest_path}")
            except Exception as e:
                self.main_window.log(f"Error copying audio files: {str(e)}", error=True)
        
        # Ask if user wants to process files
        if (audio_files or chat_files) and self.main_window:
            response = QMessageBox.question(
                self.main_window,
                "Process Files",
                "Files have been copied to source directories. Process them now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if response == QMessageBox.StandardButton.Yes:
                self.main_window.process_files()

class MainWindow(QMainWindow):
    """Main window of the RPG Notes Automator application."""
    
    def __init__(self):
        super().__init__()
        self.settings = QSettings("RPGNotes", "Automator")
        self.env_file = Path(".env")
        self.config_file = Path("config.json")
        self.init_ui()
        self.validate_directories()
        self.worker = None
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("RPG Notes Automator")
        self.setMinimumSize(1200, 800)  # Increased size
        
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
        
        # Create left panel (file management)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Source Selection
        source_layout = QHBoxLayout()
        source_label = QLabel("Audio Source:")
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Craig Bot", "Discord"])
        self.source_combo.currentTextChanged.connect(self.handle_source_change)
        source_layout.addWidget(source_label)
        source_layout.addWidget(self.source_combo)
        source_layout.addStretch()
        left_layout.addLayout(source_layout)
        
        # Drop Zone
        self.drop_zone = FileDropFrame(self)
        self.drop_zone.setMinimumHeight(100)
        left_layout.addWidget(self.drop_zone)
        
        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMinimumHeight(200)
        left_layout.addWidget(QLabel("Processing Log:"))
        left_layout.addWidget(self.log_display)
        
        # File List
        self.file_list = QTreeView()
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(str(OUTPUT_DIR))
        self.file_model.setNameFilters(["*.txt", "*.md", "*.mp3", "*.wav", "*.json"])
        self.file_model.setNameFilterDisables(False)
        self.file_list.setModel(self.file_model)
        self.file_list.setRootIndex(self.file_model.index(str(OUTPUT_DIR)))
        self.file_list.setColumnWidth(0, 250)  # Name column
        self.file_list.setColumnWidth(1, 100)  # Size column
        self.file_list.clicked.connect(self.handle_file_selection)
        left_layout.addWidget(QLabel("Generated Files:"))
        left_layout.addWidget(self.file_list)
        
        # Progress widget
        self.progress_widget = ProgressWidget(self)
        left_layout.addWidget(self.progress_widget)
        
        # Batch Controls
        batch_layout = QHBoxLayout()
        self.process_all_btn = QPushButton("Process All")
        self.process_all_btn.clicked.connect(self.process_files)
        self.skip_btn = QPushButton("Skip Selected")
        batch_layout.addWidget(self.process_all_btn)
        batch_layout.addWidget(self.skip_btn)
        left_layout.addLayout(batch_layout)
        
        main_layout.addWidget(left_panel)
        
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
        main_layout.setStretch(0, 2)  # Left panel takes 2/3
        main_layout.setStretch(1, 1)  # Right panel takes 1/3
        
    def validate_directories(self):
        """Validate and create required directories."""
        directories = [
            OUTPUT_DIR, TEMP_DIR, CHAT_LOG_OUTPUT_DIR, AUDIO_OUTPUT_DIR, 
            TRANSCRIPTIONS_OUTPUT_DIR, TEMP_TRANSCRIPTIONS, CONTEXT_DIR
        ]
        
        for directory in directories:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.log(f"Created directory: {directory}")
                except Exception as e:
                    self.log(f"Error creating directory {directory}: {e}", error=True)
                    QMessageBox.warning(
                        self, 
                        "Directory Error", 
                        f"Failed to create directory: {directory}\nError: {str(e)}"
                    )
    
    def log(self, message, error=False):
        """Add a message to the log display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = "[ERROR]" if error else "[INFO]"
        formatted_message = f"{timestamp} {prefix} {message}"
        self.log_display.append(formatted_message)
        if error:
            print(f"ERROR: {message}")
        else:
            print(f"INFO: {message}")
    
    def create_toolbar(self):
        """Create the main toolbar."""
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)
        
        # Add actions with icons
        new_action = QAction("New Session", self)
        new_action.setStatusTip("Start a new session")
        new_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        toolbar.addAction(new_action)
        
        process_action = QAction("Process Files", self)
        process_action.setStatusTip("Process selected files")
        process_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        process_action.triggered.connect(self.process_files)
        toolbar.addAction(process_action)
        
        toolbar.addSeparator()
        
        settings_action = QAction("Settings", self)
        settings_action.setStatusTip("Open settings dialog")
        settings_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        toolbar.addAction(settings_action)
        
    def create_file_section(self):
        """Create the file management section."""
        group = QGroupBox("File Management")
        layout = QVBoxLayout(group)
        
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
        
        return group
        
    def create_config_section(self):
        """Create the configuration section."""
        group = QGroupBox("Configuration")
        layout = QFormLayout(group)
        
        # Add directory selection fields with browse buttons
        output_layout = QHBoxLayout()
        self.output_dir = QLineEdit(str(OUTPUT_DIR))
        self.output_dir.setReadOnly(True)
        output_browse = QPushButton("Browse...")
        output_browse.setFixedWidth(100)  # Fixed width for consistency
        output_browse.clicked.connect(lambda: self.browse_directory(self.output_dir, "output"))
        output_layout.addWidget(self.output_dir)
        output_layout.addWidget(output_browse)
        layout.addRow("Output Directory:", output_layout)
        
        temp_layout = QHBoxLayout()
        self.temp_dir = QLineEdit(str(TEMP_DIR))
        self.temp_dir.setReadOnly(True)
        temp_browse = QPushButton("Browse...")
        temp_browse.setFixedWidth(100)  # Fixed width for consistency
        temp_browse.clicked.connect(lambda: self.browse_directory(self.temp_dir, "temp"))
        temp_layout.addWidget(self.temp_dir)
        temp_layout.addWidget(temp_browse)
        layout.addRow("Temp Directory:", temp_layout)
        
        # Add API key input with show/hide toggle and save button
        api_layout = QHBoxLayout()
        self.api_key_input = QLineEdit()
        if GEMINI_API_KEY:  # Load existing key if available
            self.api_key_input.setText(GEMINI_API_KEY)
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)  # Visible by default
        self.hide_key_btn = QPushButton("Hide")
        self.hide_key_btn.setCheckable(True)
        self.hide_key_btn.clicked.connect(self.toggle_api_key_visibility)
        self.save_key_btn = QPushButton("Save")
        self.save_key_btn.clicked.connect(self.save_api_key)
        api_layout.addWidget(self.api_key_input)
        api_layout.addWidget(self.hide_key_btn)
        api_layout.addWidget(self.save_key_btn)
        layout.addRow("API Key:", api_layout)
        
        # Add model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["gemini-1.5-pro", "gemini-1.0-pro"])
        current_model_idx = self.model_combo.findText(GEMINI_MODEL_NAME)
        if current_model_idx >= 0:
            self.model_combo.setCurrentIndex(current_model_idx)
        self.model_combo.currentTextChanged.connect(lambda text: self.save_config_value("models", "gemini", text))
        layout.addRow("Model:", self.model_combo)
        
        # Add audio quality settings
        self.audio_quality = QComboBox()
        self.audio_quality.addItems(["High", "Medium", "Low"])
        current_quality_idx = self.audio_quality.findText(AUDIO_QUALITY)
        if current_quality_idx >= 0:
            self.audio_quality.setCurrentIndex(current_quality_idx)
        self.audio_quality.currentTextChanged.connect(lambda text: self.save_config_value("settings", "audio_quality", text))
        layout.addRow("Audio Quality:", self.audio_quality)
        
        # Add delete temp files option
        self.delete_temp = QCheckBox("Delete temporary files after processing")
        self.delete_temp.setChecked(DELETE_TEMP_FILES)
        self.delete_temp.toggled.connect(lambda checked: self.save_config_value("settings", "delete_temp_files", checked))
        layout.addRow("", self.delete_temp)
        
        return group
        
    def create_preview_section(self):
        """Create the markdown preview section."""
        group = QGroupBox("Preview")
        layout = QVBoxLayout(group)
        
        # Add preview text area
        self.preview_area = QLabel("Markdown preview will appear here...")
        self.preview_area.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        self.preview_area.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.preview_area.setWordWrap(True)
        self.preview_area.setStyleSheet("""
            QLabel {
                background-color: white;
                padding: 10px;
                font-family: monospace;
            }
        """)
        self.preview_area.setMinimumHeight(300)  # Ensure enough height for preview
        layout.addWidget(self.preview_area)
        
        return group
        
    def browse_directory(self, line_edit, config_key):
        """Open directory browser dialog and save to config."""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            line_edit.setText(directory)
            self.save_config_value("directories", config_key, directory)
            # Update directory path in config module
            if config_key == "output":
                global OUTPUT_DIR
                OUTPUT_DIR = Path(directory)
            elif config_key == "temp":
                global TEMP_DIR
                TEMP_DIR = Path(directory)
            elif config_key == "chat_log_source":
                global CHAT_LOG_SOURCE_DIR
                CHAT_LOG_SOURCE_DIR = Path(directory)
            elif config_key == "audio_source":
                global AUDIO_SOURCE_DIR
                AUDIO_SOURCE_DIR = Path(directory)
            elif config_key == "context":
                global CONTEXT_DIR
                CONTEXT_DIR = Path(directory)
            
            # Update dependent directories
            self.validate_directories()
            self.log(f"Updated {config_key} directory to: {directory}")
            
    def process_files(self):
        """Process the selected files."""
        # Validate prerequisites
        if not GEMINI_API_KEY:
            QMessageBox.warning(
                self, 
                "Missing API Key", 
                "Please enter and save your Gemini API key before processing."
            )
            return
            
        # Check for source files
        source_type = self.source_combo.currentText()
        if source_type == "Craig Bot":
            if not get_newest_file(AUDIO_SOURCE_DIR, "craig-*.flac.zip"):
                QMessageBox.warning(
                    self, 
                    "Missing Audio Files", 
                    "No Craig bot recordings found in the audio source directory."
                )
                return
        else:  # Discord
            # Add Discord-specific validation here
            pass
            
        if not get_newest_file(CHAT_LOG_SOURCE_DIR, "*.json"):
            QMessageBox.warning(
                self, 
                "Missing Chat Logs", 
                "No chat log files found in the chat log source directory."
            )
            return
            
        # Confirm with user
        response = QMessageBox.question(
            self,
            "Confirm Processing",
            "This will process the newest audio recordings and chat logs. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if response == QMessageBox.StandardButton.No:
            return
            
        # Start processing
        self.log("Starting processing workflow...")
        self.progress_widget.start_processing(100)  # 100% total
        
        # Create worker thread
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            
        self.worker = ProcessingWorker(source_type)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.processing_complete.connect(self.processing_completed)
        self.worker.start()
        
        # Disable process button while running
        self.process_all_btn.setEnabled(False)
        
    def update_progress(self, progress, message):
        """Update the progress widget with new information."""
        self.progress_widget.update_progress(progress, message)
        self.log(message)
        
    def processing_completed(self, success, message):
        """Handle the completion of processing."""
        self.process_all_btn.setEnabled(True)
        self.progress_widget.finish_processing()
        
        if success:
            self.log(message)
            QMessageBox.information(
                self,
                "Processing Complete",
                message
            )
            
            # Reload file tree
            self.file_list.setRootIndex(self.file_model.index(str(OUTPUT_DIR)))
        else:
            self.log(message, error=True)
            QMessageBox.critical(
                self,
                "Processing Error",
                message
            )

    def toggle_api_key_visibility(self):
        if self.hide_key_btn.isChecked():
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.hide_key_btn.setText("Show")
        else:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.hide_key_btn.setText("Hide")

    def handle_source_change(self):
        """Handle changes to the audio source selection."""
        source = self.source_combo.currentText()
        self.log(f"Changed audio source to: {source}")
        if source == "Craig Bot":
            self.drop_zone.label.setText("Drop Craig bot zip files here")
        else:
            self.drop_zone.label.setText("Drop Discord audio files here")

    def save_api_key(self):
        """Save the API key to the .env file."""
        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Warning", "Please enter an API key before saving.")
            return
            
        try:
            set_key(self.env_file, "GEMINI_API_KEY", api_key)
            self.statusBar.showMessage("API key saved successfully", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save API key: {str(e)}")
            return

    def save_config_value(self, section, key, value):
        """Save a value to config.json."""
        try:
            # Load current config
            if self.config_file.exists():
                with open(self.config_file, "r") as f:
                    config = json.load(f)
            else:
                config = CONFIG  # Use default config from config.py
            
            # Update value
            if section not in config:
                config[section] = {}
            config[section][key] = value
            
            # Save config
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=4)
            
            self.statusBar.showMessage(f"Configuration saved successfully", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration: {str(e)}")
            return

    def handle_file_selection(self, index):
        """Handle selection of a file in the file tree view."""
        file_path = self.file_model.filePath(index)
        self.log(f"Selected file: {file_path}")
        
        if file_path.lower().endswith(".md"):
            try:
                with open(file_path, "r") as f:
                    md_content = f.read()
                self.update_preview(md_content)
            except Exception as e:
                self.log(f"Error loading file: {str(e)}", error=True)
        elif file_path.lower().endswith(".txt"):
            try:
                with open(file_path, "r") as f:
                    text_content = f.read()
                self.update_preview(f"```\n{text_content}\n```")
            except Exception as e:
                self.log(f"Error loading file: {str(e)}", error=True)
        elif file_path.lower().endswith(".json"):
            try:
                with open(file_path, "r") as f:
                    import json
                    json_data = json.load(f)
                    json_content = json.dumps(json_data, indent=2)
                self.update_preview(f"```json\n{json_content}\n```")
            except Exception as e:
                self.log(f"Error loading file: {str(e)}", error=True)
    
    def update_preview(self, markdown_content):
        """Update the preview area with markdown content."""
        # For now, just show the raw markdown
        # In a future version, this could render the markdown
        self.preview_area.setText(markdown_content)
        self.preview_area.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 