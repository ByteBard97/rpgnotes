"""
Utility functions for RPG Notes Automator.

This module contains utility functions used throughout the application.
"""

import os
import glob
import json
from pathlib import Path
from typing import Optional, Any, Dict, List

def get_newest_file(directory: Path, pattern: str) -> Optional[str]:
    """
    Finds the newest file matching the given pattern in a directory.
    
    Args:
        directory: The directory to search in
        pattern: The glob pattern to match
        
    Returns:
        The path to the newest matching file, or None if no files match
    """
    files = glob.glob(os.path.join(directory, pattern))
    return max(files, key=os.path.getmtime) if files else None

def prettify_json(filepath: Path) -> Optional[str]:
    """
    Reads, prettifies, and returns JSON data from a file.
    
    Args:
        filepath: The path to the JSON file
        
    Returns:
        A prettified JSON string, or None if there was an error
    """
    try:
        with open(filepath, 'r') as f:
            json_data = json.load(f)
        return json.dumps(json_data, indent=2)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error processing JSON in {filepath}: {e}")
        return None

def extract_session_number(json_data: Dict[str, Any]) -> Optional[int]:
    """
    Extracts the session number from the 'title' field in JSON data.
    
    Args:
        json_data: The JSON data to extract the session number from
        
    Returns:
        The session number, or None if it couldn't be extracted
    """
    try:
        title = json_data.get("data", {}).get("title")
        return int(title.split()[-1]) if title else None
    except (AttributeError, ValueError, IndexError) as e:
        print(f"Error extracting session number: {e}")
        return None

def load_context_files(context_dir: Path) -> str:
    """
    Loads all text files from the context directory.
    
    Args:
        context_dir: The directory containing context files
        
    Returns:
        A string containing the contents of all context files
    """
    context_data = ""
    if context_dir:
        for file_path in context_dir.glob("*.txt"):
            try:
                with open(file_path, "r") as f:
                    context_data += f.read() + "\n\n"
            except Exception as e:
                print(f"Error reading context file {file_path}: {e}")
    return context_data

def get_previous_summary_file(session_number: int, output_dir: Path) -> Optional[Path]:
    """
    Retrieves the filepath of the previous session's summary, if it exists.
    
    Args:
        session_number: The current session number
        output_dir: The directory containing session summaries
        
    Returns:
        The path to the previous session's summary file, or None if it doesn't exist
    """
    previous_session_number = session_number - 1
    if previous_session_number > 0:
        potential_previous_summary = output_dir / f"Session {previous_session_number} - *.md"
        previous_summary_files = sorted(
            potential_previous_summary.parent.glob(potential_previous_summary.name),
            key=os.path.getmtime,
            reverse=True
        )
        if previous_summary_files:
            return previous_summary_files[0]
    return None 