"""
Test script for AudioProcessor with GPU acceleration.
"""

import os
import tempfile
import time
from pathlib import Path

import torch
import soundfile as sf
from datasets import load_dataset
from audio import AudioProcessor

def main():
    """Run test for AudioProcessor."""
    print("\n=== Testing AudioProcessor with GPU ===")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")
    
    # Create a temporary test environment
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup directories
        temp_path = Path(temp_dir)
        source_dir = temp_path / "source"
        output_dir = temp_path / "audio"
        transcriptions_dir = temp_path / "transcriptions"
        
        # Create directories
        source_dir.mkdir()
        output_dir.mkdir()
        transcriptions_dir.mkdir()
        
        # Create a text file to use as a prompt
        prompt_file = temp_path / "prompt.txt"
        with open(prompt_file, "w") as f:
            f.write("This is a test prompt.")
        
        # Load test dataset and save a sample audio file
        print("Loading test audio dataset...")
        dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:1]")
        
        # Save audio to file
        audio_sample = dataset[0]["audio"]
        audio_data = audio_sample["array"]
        sampling_rate = audio_sample["sampling_rate"]
        flac_path = output_dir / "test_sample.flac"
        
        print(f"Saving test audio file to {flac_path}")
        sf.write(flac_path, audio_data, sampling_rate, format="flac")
        
        # Create AudioProcessor instance
        processor = AudioProcessor(
            source_dir=source_dir,
            output_dir=output_dir,
            transcriptions_dir=transcriptions_dir,
            prompt_file=prompt_file,
            model_id="openai/whisper-tiny"  # Use tiny model for faster testing
        )
        
        # Transcribe audio
        print("\nTranscribing audio...")
        start_time = time.time()
        processor.transcribe_audio()
        total_time = time.time() - start_time
        
        # Check results
        expected_output = transcriptions_dir / "test_sample.json"
        if expected_output.exists():
            print(f"\nSuccess! Transcription completed in {total_time:.2f} seconds")
            print(f"Output file: {expected_output}")
            
            # Display content of the output file
            import json
            with open(expected_output, "r") as f:
                result = json.load(f)
            
            print(f"\nTranscription Result:")
            print(f"Number of segments: {len(result)}")
            for i, segment in enumerate(result):
                print(f"Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s")
                print(f"Text: {segment['text']}")
        else:
            print(f"Error: Output file {expected_output} not found")

if __name__ == "__main__":
    main() 