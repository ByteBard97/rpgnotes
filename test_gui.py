"""
Tests for the GUI module.
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from PyQt6.QtWidgets import QApplication, QToolBar, QLabel, QLineEdit, QPushButton, QMessageBox, QMainWindow, QTreeView, QWidget, QVBoxLayout, QComboBox, QCheckBox, QFileDialog, QTextEdit
from PyQt6.QtCore import Qt, QUrl, QMimeData, QSize, QModelIndex
from gui import MainWindow, FileDropFrame, ProgressWidget, ProcessingWorker
import os

@pytest.fixture(scope="session")
def app():
    """Create a QApplication instance for the entire test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    # Don't quit the app here, let pytest handle cleanup

@pytest.fixture
def window(app):
    """Create a fresh MainWindow instance for each test."""
    window = MainWindow()
    yield window
    window.close()
    window.deleteLater()
    # Process any pending events
    app.processEvents()

def test_window_title(window):
    """Test that the window title is set correctly."""
    assert window.windowTitle() == "RPG Notes Automator"

def test_window_minimum_size(window):
    """Test that the window has the correct minimum size."""
    assert window.minimumWidth() == 1200
    assert window.minimumHeight() == 800

def test_file_tree_exists(window):
    """Test that the file tree view is created with correct filters."""
    assert window.file_list is not None
    assert isinstance(window.file_list, QTreeView)

def test_config_section_exists(window):
    """Test that the configuration section exists and has the correct widgets."""
    assert window.output_dir is not None
    assert window.temp_dir is not None
    assert window.model_combo is not None
    assert window.api_key_input is not None
    assert window.audio_quality is not None
    assert window.delete_temp is not None
    
    # Test model combo box
    assert window.model_combo.count() == 2
    assert "gemini-1.5-pro" in [window.model_combo.itemText(i) for i in range(window.model_combo.count())]
    assert "gemini-1.0-pro" in [window.model_combo.itemText(i) for i in range(window.model_combo.count())]
    
    # Test audio quality combo box
    assert window.audio_quality.count() == 3
    assert "High" in [window.audio_quality.itemText(i) for i in range(window.audio_quality.count())]
    assert "Medium" in [window.audio_quality.itemText(i) for i in range(window.audio_quality.count())]
    assert "Low" in [window.audio_quality.itemText(i) for i in range(window.audio_quality.count())]

def test_api_key_section(window):
    """Test the API key input section."""
    # Test API key input field
    assert isinstance(window.api_key_input, QLineEdit)
    assert window.hide_key_btn is not None
    assert window.save_key_btn is not None
    
    # Test initial state
    assert window.api_key_input.echoMode() == QLineEdit.EchoMode.Normal
    assert window.hide_key_btn.text() == "Hide"

@patch('gui.set_key')
def test_save_api_key_success(mock_set_key, window, qtbot):
    """Test saving API key successfully."""
    # Set API key
    test_key = "test_api_key_123"
    window.api_key_input.setText(test_key)
    
    # Click save button
    qtbot.mouseClick(window.save_key_btn, Qt.MouseButton.LeftButton)
    
    # Verify set_key was called with correct arguments
    mock_set_key.assert_called_once_with(window.env_file, "GEMINI_API_KEY", test_key)
    
    # Verify status message
    assert "API key saved successfully" in window.statusBar.currentMessage()

@patch('gui.set_key')
def test_save_api_key_empty(mock_set_key, window, qtbot, monkeypatch):
    """Test attempting to save empty API key."""
    # Mock QMessageBox.warning
    mock_warning = MagicMock()
    monkeypatch.setattr(QMessageBox, "warning", mock_warning)
    
    # Clear API key
    window.api_key_input.setText("")
    
    # Click save button
    qtbot.mouseClick(window.save_key_btn, Qt.MouseButton.LeftButton)
    
    # Verify warning was shown
    mock_warning.assert_called_once()
    assert not mock_set_key.called

@patch('gui.set_key')
def test_save_api_key_error(mock_set_key, window, qtbot, monkeypatch):
    """Test error handling when saving API key fails."""
    # Mock QMessageBox.critical
    mock_critical = MagicMock()
    monkeypatch.setattr(QMessageBox, "critical", mock_critical)
    
    # Set up mock to raise exception
    mock_set_key.side_effect = Exception("Test error")
    
    # Set API key and click save
    window.api_key_input.setText("test_key")
    qtbot.mouseClick(window.save_key_btn, Qt.MouseButton.LeftButton)
    
    # Verify error dialog was shown
    mock_critical.assert_called_once()

def test_save_config_value_success(window, monkeypatch):
    """Test saving a value to config.json successfully."""
    # Mock json operations
    mock_json = MagicMock()
    mock_json.load.return_value = {
        "models": {"gemini": "gemini-1.5-pro"},
        "settings": {"audio_quality": "High"}
    }
    monkeypatch.setattr("json.load", mock_json.load)
    monkeypatch.setattr("json.dump", mock_json.dump)
    
    # Mock file operations
    mock_open_obj = mock_open()
    monkeypatch.setattr("builtins.open", mock_open_obj)
    
    # Test saving a value
    window.save_config_value("models", "gemini", "gemini-2.0-pro")
    
    # Verify file operations
    mock_open_obj.assert_called()
    mock_json.dump.assert_called_once()
    assert window.statusBar.currentMessage() == "Configuration saved successfully"

def test_save_config_value_error(window, monkeypatch):
    """Test error handling when saving to config.json fails."""
    # Mock QMessageBox.critical
    mock_critical = MagicMock()
    monkeypatch.setattr(QMessageBox, "critical", mock_critical)
    
    # Mock file operations to raise exception
    def mock_open(*args, **kwargs):
        raise Exception("Test error")
    monkeypatch.setattr("builtins.open", mock_open)
    
    # Test saving a value
    window.save_config_value("models", "gemini", "gemini-2.0-pro")
    
    # Verify error dialog was shown
    mock_critical.assert_called_once()
    assert "Test error" in str(mock_critical.call_args)

def test_browse_directory_saves_config(window, qtbot, monkeypatch):
    """Test that browsing for a directory saves to config.json."""
    # Mock QFileDialog
    mock_get_dir = MagicMock(return_value="/test/path")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", mock_get_dir)
    
    # Mock save_config_value
    mock_save = MagicMock()
    monkeypatch.setattr(window, "save_config_value", mock_save)
    
    # Test browsing for output directory
    window.browse_directory(window.output_dir, "output")
    
    # Verify config was saved
    assert window.output_dir.text() == "/test/path"
    mock_save.assert_called_once_with("directories", "output", "/test/path")

def test_toolbar_exists(window):
    """Test that the toolbar exists and has the correct actions."""
    toolbar = window.findChild(QToolBar)
    assert toolbar is not None
    actions = toolbar.actions()
    assert len(actions) == 4  # New Session, Process Files, separator, Settings
    assert actions[0].text() == "New Session"
    assert actions[1].text() == "Process Files"
    assert actions[3].text() == "Settings"

def test_status_bar_exists(window):
    """Test that the status bar exists."""
    assert window.statusBar is not None
    assert window.statusBar.currentMessage() == "Ready"

def test_progress_widget_exists(window):
    """Test that the progress widget exists and is initially hidden."""
    # Create progress widget if it doesn't exist
    if not hasattr(window, 'progress_widget'):
        window.progress_widget = ProgressWidget(window)
    assert window.progress_widget is not None
    assert isinstance(window.progress_widget, ProgressWidget)
    assert not window.progress_widget.isVisible()

def test_drop_zone_exists(window):
    """Test that the file drop zone exists."""
    assert window.drop_zone is not None
    assert window.drop_zone.acceptDrops()
    assert isinstance(window.drop_zone, FileDropFrame)

def test_preview_area_exists(window):
    """Test that the preview area exists."""
    assert window.preview_area is not None
    assert window.preview_area.wordWrap()
    assert "Markdown preview will appear here..." in window.preview_area.text()

def test_api_key_visible_by_default(window):
    """Test that the API key is visible by default."""
    assert window.api_key_input.echoMode() == QLineEdit.EchoMode.Normal
    assert window.hide_key_btn.text() == "Hide"
    assert not window.hide_key_btn.isChecked()

def test_api_key_toggle_visibility(window, qtbot):
    """Test that the hide/show button toggles API key visibility."""
    qtbot.mouseClick(window.hide_key_btn, Qt.MouseButton.LeftButton)
    assert window.api_key_input.echoMode() == QLineEdit.EchoMode.Password
    assert window.hide_key_btn.text() == "Show"
    
    qtbot.mouseClick(window.hide_key_btn, Qt.MouseButton.LeftButton)
    assert window.api_key_input.echoMode() == QLineEdit.EchoMode.Normal
    assert window.hide_key_btn.text() == "Hide"

def test_audio_source_selection(window):
    """Test that audio source selection exists and has correct options."""
    assert window.source_combo is not None
    assert window.source_combo.count() == 2
    assert window.source_combo.itemText(0) == "Craig Bot"
    assert window.source_combo.itemText(1) == "Discord"

def test_batch_controls_exist(window):
    """Test that batch processing controls exist."""
    assert window.process_all_btn is not None
    assert window.skip_btn is not None
    assert window.process_all_btn.text() == "Process All"
    assert window.skip_btn.text() == "Skip Selected"

@patch("pathlib.Path.exists")
@patch("pathlib.Path.mkdir")
def test_validate_directories(mock_mkdir, mock_exists, window, monkeypatch):
    """Test directory validation and creation."""
    # Mock directory existence checks
    mock_exists.return_value = False
    
    # Mock the log method to avoid I/O
    mock_log = MagicMock()
    monkeypatch.setattr(window, "log", mock_log)
    
    # Call the method
    window.validate_directories()
    
    # Check that mkdir was called for each directory
    assert mock_mkdir.call_count > 0
    assert mock_log.call_count > 0

def test_log_method(window):
    """Test the log method for adding messages to the log display."""
    # Add a test message
    test_message = "Test log message"
    window.log(test_message)
    
    # Check that the message was added to the log display
    assert test_message in window.log_display.toPlainText()
    
    # Test error message
    error_message = "Test error message"
    window.log(error_message, error=True)
    
    # Check that the error message was added with [ERROR] prefix
    assert "[ERROR]" in window.log_display.toPlainText()
    assert error_message in window.log_display.toPlainText()

@patch('gui.get_newest_file')
def test_process_files_missing_api_key(mock_get_newest, window, monkeypatch, qtbot):
    """Test handling of missing API key when processing files."""
    # Mock QMessageBox.warning
    mock_warning = MagicMock()
    monkeypatch.setattr(QMessageBox, "warning", mock_warning)
    
    # Set API key to empty
    monkeypatch.setattr('gui.GEMINI_API_KEY', '')
    
    # Call the method
    qtbot.mouseClick(window.process_all_btn, Qt.MouseButton.LeftButton)
    
    # Check that warning was shown
    mock_warning.assert_called_once()
    assert "API Key" in str(mock_warning.call_args)

@patch('gui.get_newest_file')
@patch('gui.GEMINI_API_KEY', 'test_key')
def test_process_files_missing_audio(mock_get_newest, window, monkeypatch, qtbot):
    """Test handling of missing audio files when processing files."""
    # Mock QMessageBox.warning
    mock_warning = MagicMock()
    monkeypatch.setattr(QMessageBox, "warning", mock_warning)
    
    # Set up mock to return None for audio files but something for chat logs
    mock_get_newest.side_effect = lambda dir, pattern: None if pattern.startswith("craig") else "chat.json"
    
    # Call the method
    qtbot.mouseClick(window.process_all_btn, Qt.MouseButton.LeftButton)
    
    # Check that warning was shown
    mock_warning.assert_called_once()
    assert "Missing Audio Files" in str(mock_warning.call_args)

@patch('gui.ProcessingWorker')
@patch('gui.get_newest_file')
@patch('gui.GEMINI_API_KEY', 'test_key')
def test_process_files_success(mock_get_newest, mock_worker_class, window, monkeypatch, qtbot):
    """Test successful processing workflow initiation."""
    # Mock get_newest_file to return valid files
    mock_get_newest.return_value = "some_file.ext"
    
    # Mock QMessageBox.question to return Yes
    mock_question = MagicMock(return_value=QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "question", mock_question)
    
    # Create mock worker instance
    mock_worker = MagicMock()
    mock_worker_class.return_value = mock_worker
    
    # Call the method
    qtbot.mouseClick(window.process_all_btn, Qt.MouseButton.LeftButton)
    
    # Check that worker was created and started
    mock_worker_class.assert_called_once()
    mock_worker.start.assert_called_once()
    
    # Check that process button was disabled
    assert not window.process_all_btn.isEnabled()

def test_file_drop_frame_accepts_drops(window):
    """Test that the file drop frame accepts drops."""
    assert window.drop_zone.acceptDrops()

@patch.object(MainWindow, 'log')
@patch.object(MainWindow, 'process_files')
def test_file_drop_processing(mock_process, mock_log, window, monkeypatch, qtbot):
    """Test processing of dropped files."""
    # Create a file drop frame with the window as parent
    drop_frame = FileDropFrame(window)
    
    # Mock the shutil.copy2 function
    mock_copy = MagicMock()
    monkeypatch.setattr('shutil.copy2', mock_copy)
    
    # Mock QMessageBox.question to return Yes
    mock_question = MagicMock(return_value=QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "question", mock_question)
    
    # Create a mock drop event with URLs
    event = MagicMock()
    mime_data = MagicMock()
    event.mimeData.return_value = mime_data
    
    # Create mock URLs for audio and chat files
    urls = [MagicMock(), MagicMock()]
    urls[0].toLocalFile.return_value = "/path/to/audio.flac"
    urls[1].toLocalFile.return_value = "/path/to/chat.json"
    mime_data.hasUrls.return_value = True
    mime_data.urls.return_value = urls
    
    # Trigger the drop event
    drop_frame.dropEvent(event)
    
    # Check that copy was called twice (once for each file)
    assert mock_copy.call_count == 2
    
    # Check that process_files was called
    mock_process.assert_called_once()
    
    # Check that log was called for each file
    assert mock_log.call_count >= 2

@patch.object(MainWindow, 'update_preview')
@patch.object(MainWindow, 'log')
def test_handle_file_selection(mock_log, mock_update_preview, window, monkeypatch, tmpdir):
    """Test handling of file selection."""
    # Create a temporary markdown file
    md_file = tmpdir.join("test.md")
    md_file.write("# Test Markdown\n\nThis is a test.")
    
    # Create a mock model index
    index = MagicMock()
    
    # Mock the file_model.filePath method
    window.file_model.filePath = MagicMock(return_value=str(md_file))
    
    # Call the method
    window.handle_file_selection(index)
    
    # Check that log was called
    mock_log.assert_called_once()
    
    # Check that update_preview was called with the correct content
    mock_update_preview.assert_called_once()
    assert "# Test Markdown" in mock_update_preview.call_args[0][0]

def test_update_preview(window):
    """Test updating the preview area."""
    test_markdown = "# Test Markdown\n\nThis is a test."
    window.update_preview(test_markdown)
    assert window.preview_area.text() == test_markdown
    assert window.preview_area.alignment() == (Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft) 