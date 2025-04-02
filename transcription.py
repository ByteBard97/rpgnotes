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
            try:
                discord_user = json_file.stem.split("-")[1].lstrip("_").split("_")[0]
                speaker = discord_character_mapping.get(discord_user, discord_user)
            except IndexError:
                print(f"Warning: Could not extract speaker from {json_file.name}. Skipping.")
                continue

            with open(json_file, "r") as f:
                segments = json.load(f)
                last_segment_text = None
                for segment in segments:
                    # Filter out low confidence segments and common noise
                    if segment["no_speech_prob"] > 0.3 or segment["text"].strip() in ["Thank you.", " ..."]:
                        continue
                    current_text = segment["text"].strip()
                    if last_segment_text == current_text:
                        continue
                    segment["speaker"] = speaker
                    last_segment_text = current_text
                    all_segments.append(segment)

        all_segments.sort(key=lambda x: x["start"])

        with open(combined_json_path, "w") as f:
            json.dump(all_segments, f, indent=2)

        with open(combined_txt_path, "w") as f:
            current_speaker = None
            for segment in all_segments:
                if segment["speaker"] != current_speaker:
                    f.write(f"\n\n[{segment['speaker']}]\n")
                    current_speaker = segment["speaker"]
                f.write(segment["text"].strip() + " ")

        print(f"Combined transcription saved to {combined_json_path} and {combined_txt_path}")
        return combined_txt_path 