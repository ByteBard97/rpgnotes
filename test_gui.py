"""
Tests for the GUI module.
"""

import pytest
from PyQt6.QtWidgets import QApplication, QToolBar, QLabel
from PyQt6.QtCore import Qt, QUrl, QMimeData
from gui import MainWindow, FileDropFrame

@pytest.fixture
def app():
    """Create a QApplication instance for testing."""
    app = QApplication([])
    yield app
    app.quit()

@pytest.fixture
def window(app):
    """Create a MainWindow instance for testing."""
    window = MainWindow()
    yield window
    window.close()

def test_window_title(window):
    """Test that the window title is set correctly."""
    assert window.windowTitle() == "RPG Notes Automator"

def test_window_minimum_size(window):
    """Test that the window has the correct minimum size."""
    assert window.minimumWidth() == 1000
    assert window.minimumHeight() == 700

def test_file_tree_exists(window):
    """Test that the file tree view is created with correct filters."""
    assert window.file_tree is not None
    assert window.file_model is not None
    assert window.file_model.nameFilters() == ["*.txt", "*.md", "*.mp3", "*.wav"]

def test_config_section_exists(window):
    """Test that the configuration section exists and has the correct widgets."""
    assert window.output_dir is not None
    assert window.temp_dir is not None
    assert window.model_combo is not None
    assert window.api_key is not None
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

def test_progress_bar_exists(window):
    """Test that the progress bar exists and is initially hidden."""
    assert window.progress_bar is not None
    assert not window.progress_bar.isVisible()

def test_drop_zone_exists(window):
    """Test that the file drop zone exists."""
    assert window.drop_zone is not None
    assert window.drop_zone.acceptDrops()
    assert isinstance(window.drop_zone, FileDropFrame) 