"""
Test module for the summary generation functionality.
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from models import SessionData
from summary import SessionNotesGenerator

@pytest.fixture
def mock_paths(tmp_path):
    """Create temporary paths for testing."""
    return {
        'summary_prompt': tmp_path / 'summary_prompt.txt',
        'details_prompt': tmp_path / 'details_prompt.txt',
        'template': tmp_path / 'template.md',
        'output': tmp_path / 'output',
        'context': tmp_path / 'context',
        'transcript': tmp_path / 'transcript.txt'
    }

@pytest.fixture
def mock_files(mock_paths):
    """Create mock files for testing."""
    mock_paths['summary_prompt'].write_text('Summary prompt')
    mock_paths['details_prompt'].write_text('Details prompt')
    mock_paths['template'].write_text('Session {number} - {title}\nDate: {date}\n\n{summary}\n\nEvents:\n{events}\n\nNPCs:\n{npcs}\n\nLocations:\n{locations}\n\nItems:\n{items}\n\nImages:\n{images}')
    mock_paths['output'].mkdir()
    mock_paths['context'].mkdir()
    mock_paths['transcript'].write_text('Test transcript')
    return mock_paths

@pytest.fixture
def mock_session_data():
    """Create mock session data for testing."""
    return SessionData(
        session_number=1,
        date=datetime.now(),
        title="Test Session",
        events=["Event 1", "Event 2"],
        npcs=["NPC 1", "NPC 2"],
        locations=["Location 1", "Location 2"],
        items=["Item 1", "Item 2"],
        images=["Image 1", "Image 2"]
    )

@pytest.fixture
def generator(mock_files):
    """Create a SessionNotesGenerator instance with mock files."""
    return SessionNotesGenerator(
        api_key="test_key",
        model_name="test_model",
        summary_prompt_file=mock_files['summary_prompt'],
        details_prompt_file=mock_files['details_prompt'],
        template_file=mock_files['template'],
        output_dir=mock_files['output'],
        context_dir=mock_files['context']
    )

def test_init(generator, mock_files):
    """Test initialization of SessionNotesGenerator."""
    assert generator.api_key == "test_key"
    assert generator.model_name == "test_model"
    assert generator.summary_prompt_file == mock_files['summary_prompt']
    assert generator.details_prompt_file == mock_files['details_prompt']
    assert generator.template_file == mock_files['template']
    assert generator.output_dir == mock_files['output']
    assert generator.context_dir == mock_files['context']

@patch('google.generativeai.GenerativeModel')
@patch('instructor.from_gemini')
def test_generate_session_notes(mock_instructor, mock_model, generator, mock_files, mock_session_data):
    """Test generation of session notes."""
    # Mock the Gemini model response
    mock_response = Mock()
    mock_response.text = "Test summary"
    mock_model.return_value.generate_content.return_value = mock_response
    
    # Mock the instructor client response
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_session_data
    mock_instructor.return_value = mock_client
    
    # Test the generation
    summary, data = generator.generate_session_notes(mock_files['transcript'], 1)
    
    assert summary == "Test summary"
    assert isinstance(data, SessionData)
    assert data.session_number == 1
    assert data.title == "Test Session"

def test_save_summary_file(generator, mock_files, mock_session_data):
    """Test saving summary to file."""
    summary = "Test summary"
    session_number = 1
    
    output_file = generator.save_summary_file(summary, mock_session_data, session_number)
    
    assert output_file.exists()
    content = output_file.read_text()
    assert "Session 1 - Test Session" in content
    assert "Test summary" in content
    assert "Event 1" in content
    assert "NPC 1" in content
    assert "Location 1" in content
    assert "Item 1" in content
    assert "Image 1" in content 