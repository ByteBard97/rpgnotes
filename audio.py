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
import traceback
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline
)

# Try to import audio processing libraries with fallbacks
try:
    import soundfile as sf
except ImportError:
    print("WARNING: soundfile not installed. Install with 'pip install soundfile'")
    sf = None

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    print("WARNING: pydub not installed. Install with 'pip install pydub'")
    AudioSegment = None
    PYDUB_AVAILABLE = False

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

            # Process each FLAC file to extract metadata
            speaker_count = 0
            for filename in os.listdir(self.output_dir):
                file_path = self.output_dir / filename
                if file_path.is_file() and filename.endswith(".flac"):
                    # Extract speaker info from filename
                    # Format: craig-yyyy-mm-dd-hh-mm-ss-XXXXXXXXXX-username.flac
                    try:
                        # Parse the Discord username from the filename
                        parts = filename.split('-')
                        if len(parts) >= 8:  # Make sure we have enough parts
                            username = parts[-1].replace('.flac', '')
                            print(f"Found audio for user: {username}")
                            speaker_count += 1
                    except Exception as e:
                        print(f"Error parsing filename {filename}: {e}")
                elif file_path.is_file() and not filename.endswith(".flac"):
                    # Delete non-FLAC files
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

            print(f"Found {speaker_count} speaker audio files.")
            os.remove(newest_zip)
            print(f"Deleted zip file: {newest_zip}")

        except zipfile.BadZipFile:
            print(f"Error: {newest_zip} is not a valid zip file.")
            
    def extract_speaker_from_filename(self, filename: str) -> str:
        """
        Extract the speaker name from a Craig bot filename.
        
        Args:
            filename: The Craig bot filename
            
        Returns:
            The extracted speaker name (Discord username)
        """
        try:
            # Format: craig-yyyy-mm-dd-hh-mm-ss-XXXXXXXXXX-username.flac
            parts = filename.split('-')
            if len(parts) >= 8:
                username = parts[-1].replace('.flac', '')
                return username
            return "unknown"
        except Exception as e:
            print(f"Error extracting speaker from {filename}: {e}")
            return "unknown"
            
    def get_character_name(self, discord_username: str) -> str:
        """
        Get the character name for a Discord username from the mapping file.
        
        Args:
            discord_username: The Discord username
            
        Returns:
            The mapped character name, or the original username if no mapping exists
        """
        try:
            from config import DISCORD_MAPPING_FILE
            if os.path.exists(DISCORD_MAPPING_FILE):
                with open(DISCORD_MAPPING_FILE, 'r') as f:
                    mappings = json.load(f)
                    return mappings.get(discord_username, discord_username)
            return discord_username
        except Exception as e:
            print(f"Error loading Discord mapping for {discord_username}: {e}")
            return discord_username
    
    def load_audio_file(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load an audio file using the best available method.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        file_path_str = str(audio_path)
        
        # Try pydub first if available (handles many formats with or without ffmpeg)
        if PYDUB_AVAILABLE:
            try:
                print(f"Loading with pydub: {file_path_str}")
                audio = AudioSegment.from_file(file_path_str)
                # Convert to numpy array for whisper
                samples = np.array(audio.get_array_of_samples())
                if audio.channels > 1:
                    # If stereo or multi-channel, average all channels
                    samples = samples.reshape((-1, audio.channels)).mean(axis=1)
                # Convert to float32 and normalize to [-1, 1]
                samples = samples.astype(np.float32) / (2**15 if audio.sample_width == 2 else 2**31)
                return samples, audio.frame_rate
            except Exception as e:
                print(f"Pydub error: {e}, falling back to soundfile")
        
        # Try soundfile as fallback
        if sf is not None:
            try:
                print(f"Loading with soundfile: {file_path_str}")
                audio_data, sample_rate = sf.read(file_path_str)
                return audio_data, sample_rate
            except Exception as e:
                print(f"Soundfile error: {e}")
        
        # Last resort - try using HF audio loading directly
        try:
            from datasets import load_dataset
            print(f"Loading with datasets: {file_path_str}")
            # Create temporary dataset-like dict
            audio_dict = {"audio": {"path": file_path_str}}
            # The feature extractor will handle loading directly
            return audio_dict, None
        except Exception as e:
            print(f"Dataset loading error: {e}")
            raise ValueError(f"Could not load audio file {audio_path} with any available method")
    
    def transcribe_audio(self) -> None:
        """
        Transcribe FLAC audio files using Whisper from Hugging Face Transformers.
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
            
            print(f"Model successfully loaded on {device}")
            
            # If GPU, verify it's actually using it
            if torch.cuda.is_available():
                gpu_tensors = any(param.is_cuda for param in model.parameters())
                print(f"Model loaded on GPU: {gpu_tensors}")
                
            # Load the initial prompt if available
            initial_prompt = None
            if self.prompt_file and os.path.exists(self.prompt_file):
                with open(self.prompt_file, "r") as f:
                    initial_prompt = f.read().strip()
                    print(f"Using initial prompt from {self.prompt_file}")
                
            # Process each audio file sequentially to avoid GPU memory issues
            print(f"Processing {len(audio_files)} audio files sequentially...")
            for i, audio_file in enumerate(audio_files):
                json_output_path = self.transcriptions_dir / f"{audio_file.stem}.json"
                if json_output_path.exists():
                    print(f"Skipping '{audio_file.name}' (already transcribed).")
                    continue
                
                # Extract speaker information from filename
                discord_username = self.extract_speaker_from_filename(audio_file.name)
                character_name = self.get_character_name(discord_username)
                
                print(f"Transcribing {audio_file.name} - Speaker: {character_name} ({i+1}/{len(audio_files)})")
                try:
                    # Load audio using the best available method
                    audio_data, sampling_rate = self.load_audio_file(audio_file)
                    
                    # Check if we got a dict (direct file reference) or array
                    if isinstance(audio_data, dict):
                        print("Using file path directly for transcription")
                        input_data = audio_data
                        is_dict_input = True
                    else:
                        print(f"Processing audio array (length: {len(audio_data)}, rate: {sampling_rate})")
                        input_data = {"array": audio_data, "sampling_rate": sampling_rate}
                        is_dict_input = False
                        
                        # For very long files, use chunking but we'll do it with array data
                        max_length = 10 * 60 * sampling_rate  # 10 minutes at sampling rate
                        use_chunking = len(audio_data) > max_length if hasattr(audio_data, '__len__') else False
                    
                    # Create appropriate parameters based on input type
                    params = {
                        "return_timestamps": True,
                        "generate_kwargs": {
                            "language": "en", 
                            "task": "transcribe"
                        }
                    }
                    
                    # Add chunking parameters if needed
                    if is_dict_input or use_chunking:
                        params.update({
                            "chunk_length_s": 30,
                            "stride_length_s": [5, 5],
                            "batch_size": 8
                        })
                        
                    # Run transcription
                    result = pipe(input_data, **params)
                    
                    # Convert the HF format to match the expected OpenAI Whisper format
                    segments = []
                    
                    # Process the chunks which contain timestamp info
                    if 'chunks' in result:
                        for i, chunk in enumerate(result["chunks"]):
                            segment = {
                                "id": i,
                                "text": chunk["text"],
                                "start": chunk["timestamp"][0],
                                "end": chunk["timestamp"][1],
                                "no_speech_prob": 0.1,  # Default value as HF doesn't provide this
                                "speaker": character_name  # Add speaker information
                            }
                            segments.append(segment)
                    else:
                        # If no chunks, create a single segment with the full text
                        audio_duration = len(audio_data) / sampling_rate if not is_dict_input else 0
                        segment = {
                            "id": 0,
                            "text": result["text"],
                            "start": 0.0,
                            "end": float(audio_duration),
                            "no_speech_prob": 0.1,
                            "speaker": character_name  # Add speaker information
                        }
                        segments.append(segment)
                    
                    # Save the segments in the same format as original Whisper
                    with open(json_output_path, "w") as f:
                        json.dump(segments, f, indent=2)
                        
                    print(f"\nTranscription of '{audio_file.name}' saved to '{json_output_path}'.")
                    print(f"Segments: {len(segments)}")
                    
                    # Give the GPU a small break between files
                    if torch.cuda.is_available() and i < len(audio_files) - 1:
                        print("Giving GPU a brief rest...")
                        torch.cuda.empty_cache()
                        time.sleep(1)  # Brief pause between files
                        
                except Exception as e:
                    print(f"Error transcribing '{audio_file.name}': {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    
            print(f"Completed transcription of all {len(audio_files)} audio files.")
                    
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            print("If using CUDA, ensure you have a compatible version or use device='cpu'.") 