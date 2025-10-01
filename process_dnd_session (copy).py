"""
Process D&D session audio with improved segmentation and batch transcription.

This script processes D&D session recordings to:
1. Segment speech from multiple audio files
2. Transcribe each segment with speaker identification
3. Assemble a complete transcript suitable for LLM processing
"""

import os
import time
import json
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import re
import matplotlib.pyplot as plt
import librosa
import config

from audio_segmenter import AudioSegmenter

# Player mapping (Discord username to character name)
PLAYER_MAPPING = {
    "nakor_the_blue_rider": "Dungeon Master",
    "iusegentoobtw": "Astreus",
    "mden2": "Delphi, Aella, and Rheana"
    # Add more player mappings as needed
}

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


def transcribe_segments_batch(segments, sample_rate, model_id="openai/whisper-large-v3", batch_size=8):
    """Transcribe audio segments in batches with manual processing for stability."""
    print(f"\nTranscribing {len(segments)} segments using manual batching (batch size: {batch_size})...")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"Loading Whisper model {model_id} on {device}...")
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)

    transcribed_texts = []
    pbar = CustomProgressBar(total=len(segments), desc="Transcribing segments", unit="segment")

    # Manually process in batches
    for start_idx in range(0, len(segments), batch_size):
        end_idx = min(start_idx + batch_size, len(segments))
        current_batch_segments = segments[start_idx:end_idx]
        
        # Prepare audio for the current batch
        audio_inputs = [seg['audio_data'] for seg in current_batch_segments]
        
        # Preprocess the batch
        inputs = processor(audio_inputs, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        input_features = inputs.input_features.to(device, dtype=torch_dtype)

        # Generate transcriptions for the batch
        with torch.no_grad():
            generated_ids = model.generate(input_features, language="en", task="transcribe")
        
        # Decode and store results for the batch
        batch_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        transcribed_texts.extend(batch_texts)

        pbar.update(len(current_batch_segments))

    pbar.close()

    # Assemble final results
    transcribed_segments = []
    for i, text in enumerate(transcribed_texts):
        original_segment = segments[i]
        
        text = text.strip()
        # Clean up text
        text = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', text)  # Remove repetitive words
        text = re.sub(r'\byou\s+you\s+you(\s+you)*\b', r'you', text)  # Fix "you you you"
        text = re.sub(r'\b(\w+)(-\1){1,}\b', r'\1', text)  # Fix stutter patterns
        text = re.sub(r'\s{2,}', ' ', text)  # Remove excessive spacing

        # Skip empty segments
        if not text or text.isspace():
            continue
        
        # Create segment with transcription
        transcribed_segment = {
            "id": original_segment['id'],
            "text": text,
            "start": original_segment['start'],
            "end": original_segment['end'],
            "duration": original_segment['duration']
        }
        
        transcribed_segments.append(transcribed_segment)
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Completed transcription of {len(transcribed_segments)} segments")
    return transcribed_segments


def process_audio_in_chunks(file_path, segmenter, chunk_duration_s=600):
    """
    Load and segment audio in chunks to avoid memory issues with large files.
    Returns a list of all detected speech segments with correct timestamps.
    """
    print(f"Processing audio in {chunk_duration_s // 60}-minute chunks to conserve memory...")
    
    try:
        from pydub import AudioSegment
        from pydub.utils import mediainfo
    except ImportError:
        print("Error: pydub is required for chunked processing. Please install it.")
        return [], None, None
        
    info = mediainfo(file_path)
    original_sr = int(info['sample_rate'])
    target_sr = 16000
    
    audio = AudioSegment.from_file(file_path)
    duration_s = len(audio) / 1000.0
    
    all_segments_with_audio = []
    
    pbar = CustomProgressBar(total=int(duration_s), desc="Segmenting audio", unit="s")

    for start_s in range(0, int(duration_s), chunk_duration_s):
        end_s = min(start_s + chunk_duration_s, duration_s)
        
        # Get chunk from pydub audio
        chunk = audio[start_s * 1000:end_s * 1000]
        
        # Convert to numpy array
        samples = np.array(chunk.get_array_of_samples())
        if chunk.channels > 1:
            samples = samples.reshape((-1, chunk.channels)).mean(axis=1)
            
        # Normalize to float32
        samples = samples.astype(np.float32) / (2 ** 15 if chunk.sample_width == 2 else 2 ** 31)
        
        # Resample
        if original_sr != target_sr:
            samples = librosa.resample(samples, orig_sr=original_sr, target_sr=target_sr)
        
        # Detect segments in the current chunk
        raw_chunk_segments = segmenter.detect_segments(samples, target_sr)

        # Adjust timestamps to be relative to the full audio
        for seg in raw_chunk_segments:
            seg['start'] += start_s
            seg['end'] += start_s
            
            # Extract audio data for this specific segment
            start_sample = int((seg['start'] - start_s) * target_sr)
            end_sample = int((seg['end'] - start_s) * target_sr)
            seg['audio_data'] = samples[start_sample:end_sample]

        all_segments_with_audio.extend(raw_chunk_segments)
        pbar.update(chunk_duration_s)
        
    pbar.close()
    
    # Merge close segments across the entire timeline
    merged_segments = segmenter.merge_close_segments(all_segments_with_audio)

    # Final pass to add IDs and ensure audio data is present
    final_segments = []
    for i, seg in enumerate(merged_segments):
        final_segments.append({
            'id': i,
            'start': seg['start'],
            'end': seg['end'],
            'duration': seg['end'] - seg['start'],
            'audio_data': seg['audio_data']
        })

    print(f"Found {len(final_segments)} speech segments in total.")
    return final_segments, target_sr


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


def save_text_transcript(segments, output_path, speaker_name=None):
    """Save a human-readable transcript with timestamps and optional speaker name"""
    speaker_prefix = f"{speaker_name}: " if speaker_name else ""
    print(f"Saving transcript to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in segments:
            # Format timestamps as MM:SS
            start_mm = int(segment['start'] / 60)
            start_ss = int(segment['start'] % 60)
            end_mm = int(segment['end'] / 60)
            end_ss = int(segment['end'] % 60)
            
            timestamp = f"[{start_mm:02d}:{start_ss:02d} - {end_mm:02d}:{end_ss:02d}]"
            
            # Write the line with optional speaker name
            f.write(f"{timestamp} {speaker_prefix}{segment['text']}\n\n")


def extract_player_name(file_name):
    """Extract player name from filename like '1-nakor_the_blue_rider_0o.flac'"""
    # Extract part between first dash and last underscore
    match = re.search(r'^\d+-(.+?)(?:_\do)?\.flac$', file_name)
    if match:
        return match.group(1)
    return None


def process_audio_file(file_path, player_mapping):
    """Process a single audio file and return segments with player info"""
    file_path = Path(file_path)
    player_name = extract_player_name(file_path.name)
    character_name = player_mapping.get(player_name, f"Unknown ({player_name})")
    
    print(f"\n=== Processing {file_path.name} ({character_name}) ===")
    
    # Initialize a single segmenter instance
    segmenter = AudioSegmenter(
        amplitude_threshold=0.005,
        min_segment_length=0.1,
        min_silence_length=0.3,
        merge_threshold=1.2,
        padding=0.2
    )

    # Process audio using the new chunk-based method
    segments, sample_rate = process_audio_in_chunks(file_path, segmenter)
    
    if not segments:
        print("No speech segments detected. Skipping.")
        return None
    
    # Get output paths from config
    output_dir = config.OUTPUT_DIR
    session_transcriptions_dir = config.SESSION_TRANSCRIPTIONS_DIR

    # Transcribe segments
    transcribed_segments = transcribe_segments_batch(segments, sample_rate)

    # Add player and character info to segments
    for segment in transcribed_segments:
        segment['player'] = player_name
        segment['character'] = character_name

    # Save JSON results to SESSION-SPECIFIC transcription location
    json_output_path = None
    if session_transcriptions_dir:
        json_output_path = session_transcriptions_dir / f"{file_path.stem}.json"
        if json_output_path:
             with open(json_output_path, "w") as f:
                 json.dump(transcribed_segments, f, indent=2)
        else:
            print("Warning: Could not determine session transcription JSON output path.")

    # Save human-readable transcript to SESSION-SPECIFIC transcription location
    text_output_path = None
    if session_transcriptions_dir:
        text_output_path = session_transcriptions_dir / f"{file_path.stem}.txt"
        if text_output_path:
            save_text_transcript(transcribed_segments, text_output_path, character_name)
        else:
             print("Warning: Could not determine session transcription text output path.")

    # Audio duration is now known before processing, but we can state it again
    try:
        from pydub.utils import mediainfo
        info = mediainfo(str(file_path))
        audio_duration = float(info['duration'])
        audio_minutes = audio_duration / 60
        print(f"Audio duration: {audio_minutes:.2f} minutes")
    except Exception:
        # Fallback if mediainfo fails
        print("Could not calculate exact audio duration.")

    print(f"Processed {len(segments)} segments, transcribed {len(transcribed_segments)}")
    print(f"Results saved to:")
    if json_output_path: print(f"  - {json_output_path}")
    if text_output_path: print(f"  - {text_output_path}")
    
    return transcribed_segments


def add_session_context(session_dir, session_number=None, campaign_date=None):
    """Create a context section with information about the session and campaign"""
    context = []
    
    # Basic session info
    if session_number:
        context.append(f"SESSION {session_number}")
    
    # Add date (either provided or today's date)
    if campaign_date:
        context.append(f"DATE: {campaign_date}")
    else:
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        context.append(f"DATE: {today}")
    
    # Add campaign context
    context.append("\nCAMPAIGN CONTEXT:")
    # This can be loaded from a separate file or entered manually
    # For now, we'll add a placeholder
    context.append("The heroes of the Adventure continue their journey.")

    # Add character information from file if it exists
    character_file = session_dir / "characters.txt"
    if character_file.exists():
        with open(character_file, "r", encoding="utf-8") as f:
            context.append("\nCHARACTERS:")
            context.append(f.read().strip())
    
    # Check if there's a context file
    context_file = session_dir / "session_context.txt"
    if context_file.exists():
        with open(context_file, "r", encoding="utf-8") as f:
            context.append("\nADDITIONAL CONTEXT:")
            context.append(f.read().strip())
    
    return "\n".join(context)


def assemble_combined_transcript(all_segments, session_dir=None, session_number=None, campaign_date=None):
    """Assemble a combined transcript from all players, merging consecutive lines from the same speaker"""
    if not all_segments or len(all_segments) == 0:
        return None
    
    # Get session context if parameters are provided
    context = ""
    if session_dir:
        context = add_session_context(session_dir, session_number, campaign_date)
    
    # Flatten the list if it's a list of lists
    flat_segments = []
    for segments in all_segments:
        if segments:
            flat_segments.extend(segments)
    
    # Sort by start time
    sorted_segments = sorted(flat_segments, key=lambda x: x['start'])
    
    # Create combined transcript with merged consecutive speaker lines
    merged_lines = []
    current_speaker = None
    current_text = ""
    
    for segment in sorted_segments:
        speaker = segment['character']
        text = segment['text']
        
        if speaker == current_speaker:
            # Same speaker continues, append to current text
            current_text += " " + text
        else:
            # New speaker, add previous speaker's entry if it exists
            if current_speaker:
                merged_lines.append(f"{current_speaker}: {current_text}")
            
            # Start new entry
            current_speaker = speaker
            current_text = text
    
    # Don't forget the last speaker's entry
    if current_speaker:
        merged_lines.append(f"{current_speaker}: {current_text}")
    
    # Combine context with merged transcript
    combined_text = []
    if context:
        combined_text.append(context)
        combined_text.append("\n\nSESSION TRANSCRIPT:\n")
    
    combined_text.extend(merged_lines)
    
    # Get final output paths from config
    # Save combined/llm files to the SESSION transcript directory
    session_transcriptions_dir = config.SESSION_TRANSCRIPTIONS_DIR
    if not session_transcriptions_dir:
        print("Error: Session transcription output directory not configured.")
        return "\n".join(combined_text) # Return text but don't save

    # Ensure final output dir exists (initialize_config should handle this)
    # session_transcriptions_dir.mkdir(parents=True, exist_ok=True)

    # Save combined transcript with simpler filenames
    # session_name_for_file = re.sub(r'\W+', '_', config._loaded_config.get("session_name", "session")) # No longer needed for filename
    combined_path = session_transcriptions_dir / "combined.txt"
    llm_path = session_transcriptions_dir / "llm.txt"

    final_transcript_text = "\n".join(combined_text)

    # Write the files
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(final_transcript_text)

    with open(llm_path, "w", encoding="utf-8") as f:
        f.write(final_transcript_text)

    print(f"\nCombined transcript saved to:")
    print(f"  - {combined_path}")
    print(f"  - {llm_path} (LLM-friendly format)")

    return final_transcript_text # Return the text as well


def process_session(session_dir):
    """Process all audio files in a session directory"""
    session_dir = Path(session_dir)
    print(f"=== Processing D&D Session in {session_dir} ===")
    
    # --- Load Player Mapping --- 
    player_mapping = {}
    mapping_file = config.DISCORD_MAPPING_FILE # Get path from config
    if mapping_file and mapping_file.exists():
        with open(mapping_file, 'r', encoding='utf-8') as f:
            player_mapping = json.load(f)
        print(f"Loaded player mapping from: {mapping_file}")
    else:
        print(f"Warning: Player mapping file not found at {mapping_file}. Character names will be unknown.")

    # Find all FLAC files
    flac_files = list(session_dir.glob("*.flac"))
    if not flac_files:
        print("No FLAC files found.")
        return
    
    print(f"Found {len(flac_files)} audio files:")
    for file in flac_files:
        size_mb = file.stat().st_size / (1024 * 1024)
        player_name = extract_player_name(file.name)
        character_name = player_mapping.get(player_name, f"Unknown ({player_name})")
        print(f"- {file.name}: {size_mb:.1f} MB ({character_name})")
    
    # Get the directory for session transcripts
    session_transcriptions_dir = config.SESSION_TRANSCRIPTIONS_DIR
    if not session_transcriptions_dir:
        print("Error: Session transcriptions directory not configured. Cannot check for existing transcripts.")
        # Decide how to proceed: maybe exit, or maybe force re-transcription?
        # For now, let's try to continue but things will likely fail later.
        pass # Or potentially: return

    # Process each file
    start_time = time.time()
    all_segments = []

    for file in flac_files:
        segments = None
        json_output_path = None

        # Construct expected output path BEFORE processing
        if session_transcriptions_dir:
            json_output_path = session_transcriptions_dir / f"{file.stem}.json"

        # Check if the transcript already exists
        if json_output_path and json_output_path.exists():
            print(f"\n--- Found existing transcript for {file.name}. Loading... ---")
            with open(json_output_path, 'r', encoding='utf-8') as f:
                segments = json.load(f)
            print(f"Successfully loaded {len(segments)} segments from {json_output_path}")
            # Optional: Basic validation - check if it's a list
            if not isinstance(segments, list):
                print(f"Warning: Loaded data from {json_output_path} is not a list. Re-transcribing.")
                segments = None # Force re-transcription

        # If segments couldn't be loaded or didn't exist, process the audio file
        if segments is None:
            print(f"\n--- Processing transcript for {file.name}... --- ")
            segments = process_audio_file(file, player_mapping)

        # Add the loaded/processed segments to the list for final assembly
        if segments: # Ensure we don't append None if processing failed
             all_segments.append(segments)
        else:
            print(f"Warning: No segments obtained for {file.name}. It will be excluded from the combined transcript.")

    # Assemble combined transcript (now saves to correct dir)
    combined_transcript_text = assemble_combined_transcript(all_segments, session_dir=session_dir)
    
    # Calculate session stats
    elapsed_time = time.time() - start_time
    print(f"\n=== Session Processing Complete ===")
    print(f"Processed {len(flac_files)} audio files")
    print(f"Processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Process D&D session audio and generate transcripts")
    # Make session_dir required
    parser.add_argument("session_dir", type=str, help="Directory containing session audio files (e.g., recordings/session_name)")

    args = parser.parse_args()
    session_directory = Path(args.session_dir).resolve()

    if not session_directory.is_dir():
        print(f"Error: Session directory not found: {session_directory}")
        exit(1)

    # --- Initialize Configuration EARLY ---
    # This loads global config and merges session_config.json if found
    # MUST be called before accessing config values implicitly via __getattr__
    config.initialize_config(session_directory)

    # --- Check GPU (can now potentially use config settings if needed) ---
    # Example: Check a setting from config if it exists
    # use_gpu = config.settings.get("use_gpu", torch.cuda.is_available()) 
    # For now, keep original check:
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("No GPU available, using CPU (this will be slow)")

    # --- Start Processing ---
    # Pass the determined session directory to the main processing function
    process_session(session_directory) 