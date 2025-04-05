"""
Process D&D session audio with improved segmentation and batch transcription.

This script processes a specific audio file using:
1. Audio segmentation to identify speech segments
2. Batch processing for more efficient transcription
"""

import os
import time
import json
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import Dataset
import re
import matplotlib.pyplot as plt

from audio_segmenter import AudioSegmenter

# Configure paths
TARGET_FILE = "recordings/friday_march_4/2-iusegentoobtw_0o.flac"
OUTPUT_DIR = Path("recordings/friday_march_4/output")
TRANSCRIPTION_DIR = Path("recordings/friday_march_4/transcriptions")
TEMP_DIR = Path("temp")

# Ensure output directories exist
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
TRANSCRIPTION_DIR.mkdir(exist_ok=True, parents=True)
TEMP_DIR.mkdir(exist_ok=True)

class CustomProgressBar:
    """Simple progress bar for tracking processing progress"""
    def __init__(self, total, desc="Processing", unit="it"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.unit = unit
        self.start_time = time.time()
        
    def update(self, amount=1):
        self.current += amount
        elapsed = time.time() - self.start_time
        percent = 100 * (self.current / self.total)
        
        # Calculate ETA
        if self.current > 0:
            items_per_sec = self.current / elapsed
            eta = (self.total - self.current) / items_per_sec if items_per_sec > 0 else 0
            eta_str = f"{eta:.1f}s" if eta < 60 else f"{eta/60:.1f}m"
        else:
            eta_str = "?"
            
        print(f"\r{self.desc}: {self.current}/{self.total} [{percent:.1f}%] | {elapsed:.1f}s | ETA: {eta_str}", end="")
        
    def close(self):
        elapsed = time.time() - self.start_time
        print(f"\r{self.desc}: {self.total}/{self.total} [100.0%] | {elapsed:.1f}s | Complete")


def load_audio(file_path):
    """Load audio file with proper error handling"""
    print(f"Loading audio file: {file_path}")
    try:
        audio_data, sample_rate = sf.read(file_path)
        
        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        return audio_data, sample_rate
        
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None


def segment_audio(file_path):
    """Segment audio file to detect speech portions"""
    print("Segmenting audio to identify speech...")
    
    # Initialize segmenter with optimized parameters
    segmenter = AudioSegmenter(
        amplitude_threshold=0.005,  # Optimal threshold from testing
        min_segment_length=0.1,     # Catch short speech bursts
        min_silence_length=0.3,     # Detect short silences
        merge_threshold=1.2,        # Merge segments that are close
        padding=0.2                 # Add context to segments
    )
    
    # Load the audio file
    audio_data, sample_rate = load_audio(file_path)
    if audio_data is None:
        return [], None, None
    
    # Get raw segments
    raw_segments = segmenter.detect_segments(audio_data, sample_rate)
    
    # Merge close segments
    merged_segments = segmenter.merge_close_segments(raw_segments)
    
    # Add segment IDs and prepare audio data for each segment
    audio_segments = []
    for i, segment in enumerate(merged_segments):
        segment_id = i
        start_time = segment['start']
        end_time = segment['end']
        duration = end_time - start_time
        
        # Calculate sample indices
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Ensure we don't exceed array bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        
        # Extract segment audio data
        segment_audio = audio_data[start_sample:end_sample]
        
        # Create segment with audio data
        audio_segments.append({
            'id': segment_id,
            'start': start_time,
            'end': end_time,
            'duration': duration,
            'audio_data': segment_audio
        })
    
    print(f"Found {len(audio_segments)} speech segments")
    return audio_segments, audio_data, sample_rate


def transcribe_segments_batch(segments, sample_rate, model_id="openai/whisper-large-v3", batch_size=4):
    """Transcribe audio segments in batches using a simpler approach"""
    print(f"\nTranscribing {len(segments)} segments using batched processing...")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"Loading Whisper model {model_id} on {device}...")
    
    # Load the model and processor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Create the pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    # Transcribe in batches
    transcribed_segments = []
    segments_pbar = CustomProgressBar(total=len(segments), desc="Transcribing segments", unit="segment")
    
    # Process in batches
    for start_idx in range(0, len(segments), batch_size):
        end_idx = min(start_idx + batch_size, len(segments))
        current_batch = segments[start_idx:end_idx]
        
        # Process each segment in the batch one by one
        for segment in current_batch:
            segment_id = segment['id']
            start_time = segment['start']
            end_time = segment['end']
            duration = segment['duration']
            segment_audio = segment['audio_data']
            
            # Prepare input for the pipeline
            inputs = {"array": segment_audio, "sampling_rate": sample_rate}
            
            # Process the segment
            try:
                result = pipe(
                    inputs,
                    return_timestamps=True,
                    generate_kwargs={"language": "en", "task": "transcribe"}
                )
                
                # Get the text
                text = result.get("text", "").strip()
                
                # Clean up text
                text = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', text)  # Remove repetitive words
                text = re.sub(r'\byou\s+you\s+you(\s+you)*\b', r'you', text)  # Fix "you you you"
                text = re.sub(r'\b(\w+)(-\1){1,}\b', r'\1', text)  # Fix stutter patterns
                text = re.sub(r'\s{2,}', ' ', text)  # Remove excessive spacing
                
                # Skip empty segments
                if not text or text.isspace():
                    segments_pbar.update(1)
                    continue
                
                # Create segment with transcription
                transcribed_segment = {
                    "id": segment_id,
                    "text": text,
                    "start": start_time,
                    "end": end_time,
                    "duration": duration,
                    "no_speech_prob": 0.1  # Default value
                }
                
                transcribed_segments.append(transcribed_segment)
                
            except Exception as e:
                print(f"\nError transcribing segment {segment_id}: {e}")
            
            # Update progress
            segments_pbar.update(1)
            
        # Force CUDA cache cleanup after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    segments_pbar.close()
    print(f"Completed transcription of {len(transcribed_segments)} segments")
    return transcribed_segments


def visualize_segments(audio_data, sample_rate, segments, output_path):
    """Create visualizations of the detected segments in smaller time chunks"""
    print(f"Creating visualizations of detected segments...")
    
    # Calculate audio duration
    total_duration = len(audio_data) / sample_rate
    
    # Number of chunks to create
    num_chunks = 20
    chunk_duration = total_duration / num_chunks
    
    # Create a directory for the visualizations
    output_dir = Path(str(output_path).rsplit('.', 1)[0] + "_chunks")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create overview visualization (main image with all segments)
    plt.figure(figsize=(15, 6))
    times = np.arange(len(audio_data)) / sample_rate
    plt.plot(times, audio_data, color='blue', alpha=0.5)
    
    # Highlight all segments in overview
    for segment in segments:
        plt.axvspan(segment['start'], segment['end'], color='green', alpha=0.3)
    
    # Add labels for overview
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Full Audio Overview - Detected Speech Segments')
    
    # Save overview
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Create chunk visualizations
    for chunk_idx in range(num_chunks):
        # Calculate time range for this chunk
        start_time = chunk_idx * chunk_duration
        end_time = min((chunk_idx + 1) * chunk_duration, total_duration)
        
        # Calculate sample range
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Filter segments that appear in this time range
        chunk_segments = [seg for seg in segments 
                          if seg['end'] >= start_time and seg['start'] <= end_time]
        
        # Skip if no segments in this chunk
        if not chunk_segments and np.max(np.abs(audio_data[start_sample:end_sample])) < 0.01:
            continue
        
        # Create figure for this chunk
        plt.figure(figsize=(15, 6))
        
        # Calculate time values for this chunk
        chunk_times = np.arange(start_sample, end_sample) / sample_rate
        
        # Plot waveform for this chunk
        plt.plot(chunk_times, audio_data[start_sample:end_sample], color='blue', alpha=0.5)
        
        # Highlight segments in this chunk
        for segment in chunk_segments:
            # Only highlight the part of the segment that's in this chunk
            highlight_start = max(segment['start'], start_time)
            highlight_end = min(segment['end'], end_time)
            
            plt.axvspan(highlight_start, highlight_end, color='green', alpha=0.3)
            
            # Only add label if segment midpoint is in this chunk
            mid_point = (segment['start'] + segment['end']) / 2
            if start_time <= mid_point <= end_time:
                plt.text(mid_point, 0, f"Seg {segment['id']}", 
                        horizontalalignment='center', verticalalignment='center',
                        bbox=dict(facecolor='white', alpha=0.7))
        
        # Format times as MM:SS for title
        start_mm = int(start_time / 60)
        start_ss = int(start_time % 60)
        end_mm = int(end_time / 60)
        end_ss = int(end_time % 60)
        
        # Add labels for this chunk
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title(f'Speech Segments [{start_mm:02d}:{start_ss:02d} - {end_mm:02d}:{end_ss:02d}]')
        
        # Set x-axis limits for this chunk
        plt.xlim(start_time, end_time)
        
        # Save this chunk
        chunk_path = output_dir / f"segment_chunk_{chunk_idx+1:02d}.png"
        plt.tight_layout()
        plt.savefig(chunk_path)
        plt.close()
    
    print(f"Overview visualization saved to {output_path}")
    print(f"Detailed visualizations saved to {output_dir}/")
    
    return output_dir


def save_text_transcript(segments, output_path):
    """Save a human-readable transcript with timestamps"""
    print(f"Saving human-readable transcript to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in segments:
            # Format timestamps as MM:SS
            start_mm = int(segment['start'] / 60)
            start_ss = int(segment['start'] % 60)
            end_mm = int(segment['end'] / 60)
            end_ss = int(segment['end'] % 60)
            
            timestamp = f"[{start_mm:02d}:{start_ss:02d} - {end_mm:02d}:{end_ss:02d}]"
            
            # Write the line
            f.write(f"{timestamp} {segment['text']}\n\n")


def main():
    """Main processing function"""
    start_time = time.time()
    print("=== D&D Session Audio Processing ===")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("No GPU available, using CPU (this will be slow)")
    
    target_path = Path(TARGET_FILE)
    if not target_path.exists():
        print(f"Error: Target file not found at {target_path}")
        return
    
    # Get audio file details
    file_size_mb = target_path.stat().st_size / (1024 * 1024)
    print(f"Processing audio file: {target_path.name} ({file_size_mb:.1f} MB)")
    
    # Segment the audio
    segments, audio_data, sample_rate = segment_audio(target_path)
    if not segments:
        print("No speech segments detected. Exiting.")
        return
    
    # Calculate audio duration
    audio_duration = len(audio_data) / sample_rate
    audio_minutes = audio_duration / 60
    print(f"Audio duration: {audio_minutes:.2f} minutes")
    
    # Visualize segments
    visualization_path = OUTPUT_DIR / f"{target_path.stem}_segments.png"
    chunks_dir = visualize_segments(audio_data, sample_rate, segments, visualization_path)
    
    # Transcribe segments in batches
    transcribed_segments = transcribe_segments_batch(segments, sample_rate, batch_size=4)
    
    # Save JSON results
    json_output_path = TRANSCRIPTION_DIR / f"{target_path.stem}.json"
    with open(json_output_path, "w") as f:
        json.dump(transcribed_segments, f, indent=2)
    
    # Save human-readable transcript
    text_output_path = TRANSCRIPTION_DIR / f"{target_path.stem}.txt"
    save_text_transcript(transcribed_segments, text_output_path)
    
    # Calculate stats
    elapsed_time = time.time() - start_time
    proc_speed = audio_minutes / (elapsed_time / 60) if audio_minutes > 0 else 0
    
    print("\n=== Processing Complete ===")
    print(f"Processed {len(segments)} segments, transcribed {len(transcribed_segments)}")
    print(f"Processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Processing speed: {proc_speed:.2f}x realtime")
    print(f"Results saved to:")
    print(f"  - {json_output_path}")
    print(f"  - {text_output_path}")
    print(f"  - {visualization_path}")
    print(f"  - {chunks_dir}/ (detailed segment visualizations)")


if __name__ == "__main__":
    main() 