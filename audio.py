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
import re
import threading

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
    Custom progress bar that shows elapsed and ETA.
    """
    @property
    def format_dict(self):
        d = super().format_dict
        total_time = d["elapsed"] * (d["total"] / max(d["n"], 1))
        eta = total_time - d["elapsed"]
        
        # Format as HH:MM:SS
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(d["elapsed"]))
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
        
        d.update(elapsed_str=elapsed_str, eta_str=eta_str)
        return d

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bar_format = "Progress: {percentage:3.2f}% - Elapsed: {elapsed_str} - ETA: {eta_str}{bar}"

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
        
        # Get all audio files sorted by size
        audio_files = sorted(self.source_dir.glob("*.flac"), key=os.path.getsize)
        if not audio_files:
            print("No FLAC files found to transcribe.")
            return

        print(f"Loading Whisper model {self.model_id} on {device}...")
        
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
            
        # Process each audio file
        total_files = len(audio_files)
        files_pbar = CustomProgressBar(total=total_files, desc="Files", unit="file")
        
        for file_index, audio_file in enumerate(audio_files):
            json_output_path = self.transcriptions_dir / f"{audio_file.stem}.json"
            print(f"\nTranscribing file {file_index+1}/{total_files}: {audio_file.name}...")
            
            # Load audio using the best available method
            audio_data, sampling_rate = self.load_audio_file(audio_file)
            
            # Check if we got a dict (direct file reference) or array
            if isinstance(audio_data, dict):
                print("Using file path directly for transcription")
                input_data = audio_data
                is_dict_input = True
            else:
                print(f"Processing audio array (length: {len(audio_data)}, rate: {sampling_rate})")
                audio_duration_minutes = len(audio_data) / (sampling_rate * 60) if hasattr(audio_data, '__len__') else 0
                print(f"Audio duration: ~{audio_duration_minutes:.2f} minutes")
                
                input_data = {"array": audio_data, "sampling_rate": sampling_rate}
                is_dict_input = False
                
                # For very long files, use chunking but we'll do it with array data
                max_length = 10 * 60 * sampling_rate  # 10 minutes at sampling rate
                use_chunking = len(audio_data) > max_length if hasattr(audio_data, '__len__') else False
            
            # Create parameters that match the Hugging Face examples exactly
            # Set up the generate_kwargs dictionary separately
            generate_kwargs = {
                "language": "en", 
                "task": "transcribe",
                "max_new_tokens": 256,
                "condition_on_prev_tokens": True,
                "compression_ratio_threshold": 1.35,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6
            }
            
            # Prepare for timing and progress updates
            print(f"Starting transcription at {time.strftime('%H:%M:%S')}...")
            
            # Check GPU memory before transcription
            if torch.cuda.is_available():
                before_mem = torch.cuda.memory_allocated() / 1024**2
                print(f"GPU memory before transcription: {before_mem:.1f}MB")
            
            start_time = time.time()
            
            # Setup progress updates for long transcriptions
            if is_dict_input or use_chunking:
                print("Using chunked processing with progress updates...")
                # Calculate estimated completion time based on audio duration
                if 'audio_duration_minutes' in locals() and audio_duration_minutes > 0:
                    # Estimate based on historical processing speeds
                    # Medium model processes ~8-10x realtime on RTX 4070
                    est_processing_minutes = audio_duration_minutes / 8
                    est_completion_time = time.time() + (est_processing_minutes * 60)
                    est_completion_str = time.strftime("%H:%M:%S", time.localtime(est_completion_time))
                    print(f"Estimated completion time: {est_completion_str} (based on ~8x realtime processing)")
                
                # Create a thread to show progress updates
                stop_progress_thread = False
                
                def progress_timer():
                    last_time = time.time()
                    update_interval = 10  # Update every 10 seconds instead of 30
                    while not stop_progress_thread:
                        current_time = time.time()
                        # Update at regular intervals
                        if current_time - last_time >= update_interval:
                            elapsed = current_time - start_time
                            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                            
                            # Check GPU memory during processing
                            gpu_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                            
                            # Calculate remaining time if we have audio duration
                            if 'audio_duration_minutes' in locals() and audio_duration_minutes > 0:
                                # Calculate current processing speed (as a fraction of realtime)
                                current_speed = elapsed / 60 / audio_duration_minutes
                                if current_speed > 0:
                                    # Calculate remaining time based on current speed
                                    processed_minutes = audio_duration_minutes * current_speed
                                    remaining_minutes = audio_duration_minutes - processed_minutes
                                    remaining_sec = (remaining_minutes / current_speed) * 60
                                    
                                    # Ensure we don't show negative time
                                    if remaining_sec > 0:
                                        remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_sec))
                                        print(f"\rElapsed: {elapsed_str} | Remaining: {remaining_str} | Speed: {current_speed:.2f}x realtime | GPU: {gpu_mem:.0f}MB", end="")
                                    else:
                                        print(f"\rElapsed: {elapsed_str} | Almost done... | Speed: {current_speed:.2f}x realtime | GPU: {gpu_mem:.0f}MB", end="")
                                else:
                                    print(f"\rElapsed: {elapsed_str} | Still calculating... | GPU: {gpu_mem:.0f}MB", end="")
                            else:
                                print(f"\rStill transcribing... Elapsed: {elapsed_str} | GPU: {gpu_mem:.0f}MB", end="")
                            
                            last_time = current_time
                        time.sleep(1)
                
                # Start the progress thread
                progress_thread = threading.Thread(target=progress_timer)
                progress_thread.daemon = True
                progress_thread.start()
                
                try:
                    # Simple approach for chunked processing
                    result = pipe(
                        input_data,
                        chunk_length_s=30,
                        stride_length_s=[7, 7],
                        batch_size=8,
                        return_timestamps=True
                    )
                finally:
                    # Stop the progress thread
                    stop_progress_thread = True
                    progress_thread.join(timeout=1.0)
            else:
                # Simple approach for short audio
                result = pipe(input_data, return_timestamps=True)
            
            # Calculate timing stats
            elapsed_time = time.time() - start_time
            
            # Check GPU memory after transcription
            if torch.cuda.is_available():
                after_mem = torch.cuda.memory_allocated() / 1024**2
                print(f"GPU memory after transcription: {after_mem:.1f}MB")
                if 'before_mem' in locals():
                    print(f"GPU memory change: {after_mem - before_mem:.1f}MB")
            
            proc_speed = audio_duration_minutes / (elapsed_time / 60) if 'audio_duration_minutes' in locals() else 0
            print(f"Transcription completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
            if 'audio_duration_minutes' in locals() and audio_duration_minutes > 0:
                print(f"Processing speed: {proc_speed:.2f}x realtime")
            
            print()  # New line after processing
            
            # Convert the HF format to match the expected OpenAI Whisper format
            segments = []
            
            # Process the chunks which contain timestamp info
            if 'chunks' in result:
                # First pass: clean each segment's text and build the list
                cleaned_chunks = []
                for i, chunk in enumerate(result["chunks"]):
                    # Clean up text by removing excessive repetitions
                    text = chunk["text"].strip()
                    
                    # More aggressive repetition cleaning
                    # 1. Remove repetitive single word sequences (like "okay okay okay okay")
                    text = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', text)
                    
                    # 2. Remove repetitive "you" sequences specifically (common in Discord recordings)
                    text = re.sub(r'\byou\s+you\s+you(\s+you)*\b', r'you', text)
                    
                    # 3. Fix common stutter patterns
                    text = re.sub(r'\b(\w+)(-\1){1,}\b', r'\1', text)
                    
                    # 4. Remove repetitive "What?" sequences 
                    text = re.sub(r'(What\?[\s,\.]+){2,}', r'What? ', text)
                    
                    # 5. Remove repetitive "Okay" sequences
                    text = re.sub(r'(Okay\.?[\s,\.]+){2,}', r'Okay. ', text)
                    
                    # 6. Filter other common repetitive patterns
                    text = re.sub(r"(I'm (?:going to|gonna) [^\.]{3,})\. \1\.", r"\1.", text)
                    
                    # 7. Remove excessive spacing
                    text = re.sub(r'\s{2,}', ' ', text)
                    
                    # Skip empty segments
                    if not text or text.isspace():
                        continue
                        
                    # Ensure end time is greater than start time
                    start_time = chunk["timestamp"][0]
                    end_time = max(chunk["timestamp"][1], start_time + 0.1)
                    
                    cleaned_chunks.append({
                        "text": text,
                        "start": start_time,
                        "end": end_time
                    })
                
                # Second pass: filter out redundant segments and assign IDs
                final_segments = []
                prev_segment_text = ""
                
                for i, chunk in enumerate(cleaned_chunks):
                    current_text = chunk["text"]
                    
                    # Skip if identical to previous segment or very similar
                    if current_text == prev_segment_text:
                        continue
                    
                    # Check if this segment is very similar to the previous one
                    # For single word/phrase segments like "Okay" or "Yeah"
                    if (len(current_text.split()) <= 2 and 
                        len(prev_segment_text.split()) <= 2 and
                        current_text.split()[0] == prev_segment_text.split()[0]):
                        continue
                    
                    # Add the segment to final list
                    segment = {
                        "id": len(final_segments),
                        "text": current_text,
                        "start": chunk["start"],
                        "end": chunk["end"],
                        "no_speech_prob": 0.1  # Default value as HF doesn't provide this
                    }
                    final_segments.append(segment)
                    prev_segment_text = current_text
                
                segments = final_segments
            else:
                # If no chunks, create a single segment with the full text
                audio_duration = len(audio_data) / sampling_rate if not is_dict_input else 0
                segment = {
                    "id": 0,
                    "text": result["text"],
                    "start": 0.0,
                    "end": float(audio_duration),
                    "no_speech_prob": 0.1
                }
                segments.append(segment)
            
            # Save the segments in the same format as original Whisper
            with open(json_output_path, "w") as f:
                json.dump(segments, f, indent=2)
                
            print(f"\nTranscription of '{audio_file.name}' saved to '{json_output_path}'.")
            print(f"Segments: {len(segments)}")
            
            # Update the progress bar for files
            files_pbar.update(1)
        
        files_pbar.close() 