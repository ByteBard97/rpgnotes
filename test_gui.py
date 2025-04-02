"""
Tests for the GUI module.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from PyQt6.QtWidgets import QApplication, QToolBar, QLabel, QLineEdit, QPushButton, QMessageBox, QMainWindow, QTreeView, QWidget, QVBoxLayout, QComboBox, QCheckBox, QFileDialog
from PyQt6.QtCore import Qt, QUrl, QMimeData, QSize
from gui import MainWindow, FileDropFrame, ProgressWidget
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