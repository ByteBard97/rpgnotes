"""
Audio processing module for RPG Notes Automator.

This module handles audio file extraction and transcription.
"""

import os
import sys
import time
import json
import zipfile
import torch
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline
)

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
    and transcribing them using Whisper from Hugging Face Transformers.
    """
    
    def __init__(
        self, 
        source_dir: Path = AUDIO_SOURCE_DIR,
        output_dir: Path = AUDIO_OUTPUT_DIR,
        transcriptions_dir: Path = TEMP_TRANSCRIPTIONS,
        prompt_file: Path = WHISPER_PROMPT_FILE,
        model_id: str = "openai/whisper-large-v3"
    ):
        """
        Initialize the AudioProcessor.
        
        Args:
            source_dir: Directory containing source audio zip files
            output_dir: Directory for extracted audio files
            transcriptions_dir: Directory for transcription output
            prompt_file: File containing the initial prompt for Whisper
            model_id: Hugging Face model ID for Whisper
        """
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.transcriptions_dir = transcriptions_dir
        self.prompt_file = prompt_file
        self.model_id = model_id
        
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
        Transcribe FLAC audio files using Whisper from Hugging Face Transformers.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
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

        print(f"Loading Whisper model {self.model_id} on {device}...")
        
        try:
            # Load the model and processor
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id, 
                torch_dtype=torch_dtype, 
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            model.to(device)
            
            processor = AutoProcessor.from_pretrained(self.model_id)
            
            # Create the pipeline
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device,
            )
            
            # Load the initial prompt
            with open(self.prompt_file, "r") as f:
                initial_prompt = f.read().strip()
                
            # Configure generation parameters to include timestamps
            generate_kwargs = {
                "language": "english",  # Can be changed to match your language
                "task": "transcribe",
                "initial_prompt": initial_prompt,
                "return_timestamps": True
            }
            
            # Process each audio file
            for audio_file in audio_files:
                json_output_path = self.transcriptions_dir / f"{audio_file.stem}.json"
                if json_output_path.exists():
                    print(f"Skipping '{audio_file.name}' (already transcribed).")
                    continue
                
                print(f"Transcribing {audio_file.name}...")
                try:
                    # Transcribe with timestamps
                    result = pipe(
                        str(audio_file),
                        chunk_length_s=30,  # Process 30 seconds at a time
                        stride_length_s=[5, 5],  # Overlap between chunks
                        batch_size=8,  # Adjust based on GPU memory
                        generate_kwargs=generate_kwargs,
                        return_timestamps=True
                    )
                    
                    # Convert the HF format to match the expected OpenAI Whisper format
                    # This ensures compatibility with the rest of the pipeline
                    segments = []
                    
                    # Process the chunks which contain timestamp info
                    for i, chunk in enumerate(result["chunks"]):
                        segment = {
                            "id": i,
                            "text": chunk["text"],
                            "start": chunk["timestamp"][0],
                            "end": chunk["timestamp"][1],
                            "no_speech_prob": 0.1  # Default value as HF doesn't provide this
                        }
                        segments.append(segment)
                    
                    # Save the segments in the same format as original Whisper
                    with open(json_output_path, "w") as f:
                        json.dump(segments, f, indent=2)
                        
                    print(f"\nTranscription of '{audio_file.name}' saved to '{json_output_path}'.")
                except Exception as e:
                    print(f"Error transcribing '{audio_file.name}': {e}")
                    
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            print("If using CUDA, ensure you have a compatible version or use device='cpu'.") 