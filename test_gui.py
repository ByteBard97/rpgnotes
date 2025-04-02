"""
Tests for the GUI module.
"""

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from gui import MainWindow

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
    assert window.minimumWidth() == 800
    assert window.minimumHeight() == 600

def test_file_tree_exists(window):
    """Test that the file tree view is created."""
    assert window.file_tree is not None
    assert window.file_model is not None

def test_config_section_exists(window):
    """Test that the configuration section exists and has the correct widgets."""
    assert window.output_dir is not None
    assert window.temp_dir is not None
    assert window.model_combo is not None
    assert window.model_combo.count() == 2
    assert "gemini-1.5-pro" in [window.model_combo.itemText(i) for i in range(window.model_combo.count())]
    assert "gemini-1.0-pro" in [window.model_combo.itemText(i) for i in range(window.model_combo.count())] 