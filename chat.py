"""
Chat log processing module for RPG Notes Automator.

This module handles processing and extracting information from chat logs.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any

from config import CHAT_LOG_SOURCE_DIR, CHAT_LOG_OUTPUT_DIR
from utils import get_newest_file, prettify_json, extract_session_number

class ChatLogProcessor:
    """
    Handles processing of chat logs from FoundryVTT.
    
    This class is responsible for finding, parsing, and extracting information
    from chat log files.
    """
    
    def __init__(self, source_dir: Path = CHAT_LOG_SOURCE_DIR, output_dir: Path = CHAT_LOG_OUTPUT_DIR):
        """
        Initialize the ChatLogProcessor.
        
        Args:
            source_dir: Directory containing source chat logs
            output_dir: Directory for processed chat logs
        """
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_chat_log(self) -> Optional[int]:
        """
        Process the newest chat log, prettify it, and save it with the session number.
        
        Returns:
            The session number if successful, None otherwise
        """
        newest_chat_log = get_newest_file(self.source_dir, "*.json")
        if not newest_chat_log:
            print("No chat log found.")
            return None

        with open(newest_chat_log, 'r') as f:
            original_json_data = json.load(f)

        session_number = extract_session_number(original_json_data)
        if not session_number:
            return None

        # Check if chat log for this session already exists
        if (self.output_dir / f"session{session_number}.json").exists():
            print(f"Chat log for session {session_number} already exists. Skipping.")
            return session_number

        prettified_json_string = prettify_json(newest_chat_log)
        if not prettified_json_string:
            return None

        if session_number:
            output_filepath = self.output_dir / f"session{session_number}.json"
            with open(output_filepath, 'w') as f:
                f.write(prettified_json_string)
            print(f"Prettified chat log saved to: {output_filepath}")

        return session_number 