"""
Audio processing module for RPG Notes Automator.

This module handles audio file extraction and transcription.
"""

import os
import sys
import time
import json
import zipfile
from pathlib import Path
from typing import List, Optional

import whisper
from tqdm import tqdm

from config import (
    AUDIO_SOURCE_DIR, 
    AUDIO_OUTPUT_DIR, 
    TEMP_TRANSCRIPTIONS,
    WHISPER_PROMPT_FILE
)

class CustomProgressBar(tqdm):
    """
    Custom progress bar to display elapsed and estimated remaining time.
    
    This class extends tqdm to provide more detailed progress information
    during long-running operations.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the custom progress bar."""
        super().__init__(*args, **kwargs)
        self._current = self.n
        self._start_time = time.time()
        self._last_update_time = self._start_time
        self._iteration_times = []

    def print_in_place(self, text: str) -> None:
        """
        Print text in place, overwriting the current line.
        
        Args:
            text: The text to print
        """
        sys.stdout.write("\r" + text)
        sys.stdout.flush()

    def update(self, n: int) -> None:
        """
        Update the progress bar with additional progress.
        
        Args:
            n: The amount of progress to add
        """
        super().update(n)
        self._current += n

        current_time = time.time()
        elapsed_time = current_time - self._start_time
        iteration_time = current_time - self._last_update_time
        self._iteration_times.append(iteration_time / n)
        average_iteration_time = sum(self._iteration_times) / len(self._iteration_times)
        remaining_items = self.total - self._current
        estimated_remaining_time = remaining_items * average_iteration_time

        elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(estimated_remaining_time))

        percentage = (self._current / self.total) * 100
        self.print_in_place(f"Progress: {percentage:.2f}% - Elapsed: {elapsed_time_str} - ETA: {remaining_time_str}")

        self._last_update_time = current_time

class AudioProcessor:
    """
    Handles audio file extraction and transcription.
    
    This class is responsible for extracting audio files from zip archives
    and transcribing them using Whisper.
    """
    
    def __init__(
        self, 
        source_dir: Path = AUDIO_SOURCE_DIR,
        output_dir: Path = AUDIO_OUTPUT_DIR,
        transcriptions_dir: Path = TEMP_TRANSCRIPTIONS,
        prompt_file: Path = WHISPER_PROMPT_FILE
    ):
        """
        Initialize the AudioProcessor.
        
        Args:
            source_dir: Directory containing source audio zip files
            output_dir: Directory for extracted audio files
            transcriptions_dir: Directory for transcription output
            prompt_file: File containing the initial prompt for Whisper
        """
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.transcriptions_dir = transcriptions_dir
        self.prompt_file = prompt_file
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transcriptions_dir.mkdir(parents=True, exist_ok=True)
    
    def unzip_audio(self) -> None:
        """
        Unzip the newest FLAC zip file to the audio output directory.
        """
        # Check if audio files already exist
        if any(self.output_dir.glob("*.flac")):
            print("Audio files already exist. Skipping unzip.")
            return

        from utils import get_newest_file
        newest_zip = get_newest_file(self.source_dir, "craig-*.flac.zip")
        if not newest_zip:
            print("No matching audio zip file found.")
            return

        try:
            with zipfile.ZipFile(newest_zip, 'r') as zip_ref:
                zip_ref.extractall(self.output_dir)
            print(f"Extracted audio to: {self.output_dir}")

            # Delete non-FLAC files
            for filename in os.listdir(self.output_dir):
                file_path = self.output_dir / filename
                if file_path.is_file() and not filename.endswith(".flac"):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

            os.remove(newest_zip)
            print(f"Deleted zip file: {newest_zip}")

        except zipfile.BadZipFile:
            print(f"Error: {newest_zip} is not a valid zip file.")
    
    def transcribe_audio(self) -> None:
        """
        Transcribe FLAC audio files using Whisper.
        """
        model_name = "large"
        device = "cuda"

        should_transcribe = False
        audio_files = sorted(self.output_dir.glob("*.flac"), key=os.path.getsize)
        for audio_file in audio_files:
            json_output_path = self.transcriptions_dir / f"{audio_file.stem}.json"
            if json_output_path.exists():
                continue
            should_transcribe = True

        if not should_transcribe:
            print(f"All audio files already transcribed. Skipping.")
            return

        # Inject custom progress bar into Whisper
        transcribe_module = sys.modules['whisper.transcribe']
        transcribe_module.tqdm.tqdm = CustomProgressBar

        try:
            model = whisper.load_model(model_name, device=device, download_root="./models/")
        except RuntimeError as e:
            print(f"Error loading Whisper model: {e}")
            print("Ensure you have a compatible CUDA version or use device='cpu'.")
            return

        with open(self.prompt_file, "r") as f:
            initial_prompt = f.read().strip()

        for audio_file in audio_files:
            json_output_path = self.transcriptions_dir / f"{audio_file.stem}.json"
            if json_output_path.exists():
                print(f"Skipping '{audio_file.name}' (already transcribed).")
                continue

            print(f"Transcribing {audio_file.name}...")
            try:
                result = model.transcribe(
                    str(audio_file),
                    language="en",  # Changed from "pl" to "en"
                    initial_prompt=initial_prompt
                )
                with open(json_output_path, "w") as f:
                    json.dump(result["segments"], f, indent=2)
                print(f"\nTranscription of '{audio_file.name}' saved to '{json_output_path}'.")
            except Exception as e:
                print(f"Error transcribing '{audio_file.name}': {e}") 