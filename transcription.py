"""
Transcription processing module for RPG Notes Automator.

This module handles combining and processing transcriptions.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import (
    TEMP_TRANSCRIPTIONS,
    TRANSCRIPTIONS_OUTPUT_DIR,
    DISCORD_MAPPING_FILE
)

class TranscriptionProcessor:
    """
    Handles combining and processing transcriptions.
    
    This class is responsible for combining individual transcriptions,
    adding speaker labels, and creating a formatted transcript.
    """
    
    def __init__(
        self,
        transcriptions_dir: Path = TEMP_TRANSCRIPTIONS,
        output_dir: Path = TRANSCRIPTIONS_OUTPUT_DIR,
        mapping_file: Path = DISCORD_MAPPING_FILE
    ):
        """
        Initialize the TranscriptionProcessor.
        
        Args:
            transcriptions_dir: Directory containing individual transcriptions
            output_dir: Directory for combined transcriptions
            mapping_file: File containing Discord user to character mapping
        """
        self.transcriptions_dir = transcriptions_dir
        self.output_dir = output_dir
        self.mapping_file = mapping_file
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def combine_transcriptions(self, session_number: int) -> Optional[Path]:
        """
        Combines JSON transcriptions, adds speaker labels, and creates a TXT file.
        
        Args:
            session_number: The session number
            
        Returns:
            The path to the combined TXT file, or None if there was an error
        """
        combined_json_path = self.output_dir / f"session{session_number}.json"
        combined_txt_path = self.output_dir / f"session{session_number}.txt"

        # Check if combined files already exist
        if combined_json_path.exists() and combined_txt_path.exists():
            print(f"Combined transcriptions for session {session_number} already exist. Skipping.")
            return combined_txt_path

        try:
            with open(self.mapping_file, "r") as f:
                discord_character_mapping = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Mapping file '{self.mapping_file}' not found. Using raw Discord usernames.")
            discord_character_mapping = {}

        all_segments = []
        for json_file in self.transcriptions_dir.glob("*.json"):
            print(f"Processing transcription file: {json_file.name}")
            
            # Extract speaker information directly from the JSON content
            with open(json_file, "r") as f:
                segments = json.load(f)
                
                # Check if the segments already have speaker information
                has_speaker_info = all(("speaker" in segment) for segment in segments if segments)
                
                if not has_speaker_info:
                    # Try to extract speaker from filename if not in JSON
                    try:
                        # Try Craig-style format first
                        if json_file.stem.startswith("craig-"):
                            parts = json_file.stem.split('-')
                            if len(parts) >= 8:
                                discord_user = parts[-1].replace('.flac', '')
                            else:
                                discord_user = "unknown"
                        else:
                            # Try the format from main.py as fallback
                            discord_user = json_file.stem.split("-")[1].lstrip("_").split("_")[0]
                            
                        # Map discord user to character name, with fallbacks
                        speaker = self._get_character_name(discord_user, discord_character_mapping)
                        print(f"Extracted speaker '{speaker}' for file {json_file.name}")
                        
                        # Add speaker to each segment
                        for segment in segments:
                            segment["speaker"] = speaker
                    except (IndexError, KeyError) as e:
                        print(f"Warning: Could not extract speaker from {json_file.name}: {e}. Skipping.")
                        continue
                
                # Filter out unwanted segments
                last_segment_text = None
                for segment in segments:
                    # Skip segments with high no_speech probability or common noise phrases
                    if segment.get("no_speech_prob", 0) > 0.3 or segment.get("text", "").strip() in ["Thank you.", "Dziękuję.", " ...", ""]:
                        continue
                        
                    current_text = segment.get("text", "").strip()
                    
                    # Skip duplicate segments
                    if last_segment_text == current_text:
                        continue
                        
                    last_segment_text = current_text
                    all_segments.append(segment)

        # Sort all segments by timestamp
        all_segments.sort(key=lambda x: x.get("start", 0))
        
        # Save combined JSON
        with open(combined_json_path, "w") as f:
            json.dump(all_segments, f, indent=2)
            
        # Create human-readable combined TXT
        with open(combined_txt_path, "w") as f:
            current_speaker = None
            for segment in all_segments:
                # Add speaker header when speaker changes
                if segment.get("speaker") != current_speaker:
                    f.write(f"\n\n[{segment.get('speaker', 'Unknown')}]\n")
                    current_speaker = segment.get("speaker")
                    
                # Add the text content
                f.write(segment.get("text", "").strip() + " ")
                
        print(f"Combined transcription saved to {combined_json_path} and {combined_txt_path}")
        print(f"Created transcription with {len(all_segments)} segments from multiple speakers")
        
        return combined_txt_path
    
    def _get_character_name(self, discord_username: str, mapping: dict) -> str:
        """
        Get character name using various username transformations.
        
        Args:
            discord_username: Discord username to map
            mapping: Dictionary mapping usernames to character names
            
        Returns:
            Mapped character name or original username
        """
        # Handle common username variations
        username_variations = [
            discord_username,                        # Original
            discord_username.lower(),                # Lowercase
            discord_username.split('.')[0],          # Before dot
            discord_username.split('_')[0],          # Before underscore
            discord_username.split('-')[0],          # Before dash
            ''.join(c for c in discord_username if c.isalnum())  # Alphanumeric only
        ]
        
        # Try each variation
        for variation in username_variations:
            if variation in mapping:
                return mapping[variation]
                
        # If no match found, return original
        return discord_username 