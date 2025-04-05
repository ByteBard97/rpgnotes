"""
Test script for transcribing audio files from a specific directory.
"""

import os
import time
import shutil
from pathlib import Path
import torch
from audio import AudioProcessor

def format_time(seconds):
    """Format seconds into HH:MM:SS."""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def main():
    # Check GPU availability
    print("\n=== GPU Information ===")
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # Convert to MB
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
        print(f"GPU Device: {device}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Initial GPU Memory: {memory_allocated:.1f}MB allocated, {memory_reserved:.1f}MB reserved")
    else:
        print("No GPU available, running on CPU")
    
    # Setup paths
    base_dir = Path("recordings/friday_march_4")
    output_dir = base_dir / "output"
    transcriptions_dir = base_dir / "transcriptions"
    
    # Clear existing output directories
    print("\nClearing existing output directories...")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    if transcriptions_dir.exists():
        shutil.rmtree(transcriptions_dir)
    
    # Create fresh output directories
    output_dir.mkdir(exist_ok=True)
    transcriptions_dir.mkdir(exist_ok=True)
    
    # Initialize audio processor
    processor = AudioProcessor(
        source_dir=base_dir,
        output_dir=output_dir,
        transcriptions_dir=transcriptions_dir,
        model_id="openai/whisper-small"  # Using small model for faster processing
    )
    
    print("\n=== Starting Audio Processing Test ===\n")
    
    # List all FLAC files
    flac_files = list(base_dir.glob("*.flac"))
    print(f"Found {len(flac_files)} FLAC files:")
    total_size = 0
    for file in flac_files:
        size_mb = file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"- {file.name} ({size_mb:.1f}MB)")
    print(f"\nTotal audio size: {total_size:.1f}MB")
    
    print("\nProcessing files...")
    start_time = time.time()
    try:
        # Process the audio files
        processor.transcribe_audio()
        
        # Calculate processing statistics
        end_time = time.time()
        total_time = end_time - start_time
        mb_per_second = total_size / total_time if total_time > 0 else 0
        
        print(f"\nProcessing completed in {format_time(total_time)}")
        print(f"Processing speed: {mb_per_second:.2f}MB/s")
        
        if torch.cuda.is_available():
            final_memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
            final_memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
            print(f"Final GPU Memory: {final_memory_allocated:.1f}MB allocated, {final_memory_reserved:.1f}MB reserved")
        
        # Print out some results
        print("\nTranscription Results:")
        json_files = list(transcriptions_dir.glob("*.json"))
        if not json_files:
            print("No transcription files were created!")
        else:
            for json_file in json_files:
                print(f"\nReading {json_file.name}:")
                with open(json_file, "r") as f:
                    import json
                    segments = json.load(f)
                    print(f"Number of segments: {len(segments)}")
                    # Print first few segments as sample
                    for i, segment in enumerate(segments[:3]):
                        print(f"\nSegment {i+1}:")
                        print(f"Speaker: {segment.get('speaker', 'Unknown')}")
                        print(f"Time: [{segment['start']:.2f}s - {segment['end']:.2f}s]")
                        print(f"Text: {segment['text']}")
                    if len(segments) > 3:
                        print("\n... (more segments available)")
    
    except Exception as e:
        import traceback
        print(f"\nError during processing: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 