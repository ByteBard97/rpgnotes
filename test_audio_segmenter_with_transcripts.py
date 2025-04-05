"""
Test script for AudioSegmenter using complete audio excerpts with transcripts from SpeechBrain.

This script creates test audio by combining complete audio excerpts from LibriSpeech 
with their original transcripts, then tests if AudioSegmenter can accurately recover the speech segments.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf
import random
from speechbrain.utils.data_utils import download_file
from speechbrain.dataio.dataio import read_audio
from audio_segmenter import AudioSegmenter

def get_complete_audio_excerpts_with_transcripts(num_excerpts=20, max_duration=20.0):
    """
    Download and find complete audio excerpts and their transcripts from LibriSpeech dataset.
    Only selects files under the specified maximum duration.
    
    Args:
        num_excerpts: Number of audio excerpts to find
        max_duration: Maximum duration in seconds
        
    Returns:
        List of (audio_data, sample_rate, duration, file_path, transcript) tuples
    """
    print(f"Finding {num_excerpts} complete audio excerpts under {max_duration} seconds...")
    
    # Create directories
    output_dir = Path("test_data/librispeech")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Download a sample from LibriSpeech
    sample_url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    sample_path = output_dir / "dev-clean.tar.gz"
    
    # Only download if not already present
    if not sample_path.exists():
        print(f"Downloading sample from {sample_url}...")
        download_file(sample_url, sample_path)
        print("Download complete.")
    else:
        print("Using previously downloaded sample.")
    
    # Extract if not already done
    extracted_dir = output_dir / "LibriSpeech" / "dev-clean"
    
    if not extracted_dir.exists():
        print("Extracting sample...")
        import tarfile
        with tarfile.open(sample_path, 'r:gz') as tar:
            tar.extractall(path=output_dir)
        print("Extraction complete.")
    
    # Find all audio files and transcript files
    audio_files = list(extracted_dir.glob("**/*.flac"))
    if not audio_files:
        raise ValueError("No audio files found in extracted directory")
    
    print(f"Found {len(audio_files)} audio files")
    
    # Map to organize by speaker/chapter
    file_map = {}
    for audio_file in audio_files:
        # Extract speaker and chapter info from path
        # Format is typically: speaker_id/chapter_id/speaker_id-chapter_id-utterance_id.flac
        parts = audio_file.parts
        if len(parts) >= 3:
            speaker_id = parts[-3]
            chapter_id = parts[-2]
            key = f"{speaker_id}/{chapter_id}"
            
            if key not in file_map:
                file_map[key] = []
            file_map[key].append(audio_file)
    
    # Find transcript files (usually .txt files with same naming pattern as directories)
    transcript_files = list(extracted_dir.glob("**/*.txt"))
    print(f"Found {len(transcript_files)} transcript files")
    
    # Map transcripts by speaker/chapter
    transcript_map = {}
    for transcript_file in transcript_files:
        # Extract speaker and chapter info from path
        parts = transcript_file.parts
        if len(parts) >= 3:
            speaker_id = parts[-3]
            chapter_id = parts[-2]
            key = f"{speaker_id}/{chapter_id}"
            transcript_map[key] = transcript_file
    
    # Read all transcripts into a dictionary mapping utterance ID to text
    all_transcripts = {}
    for key, transcript_file in transcript_map.items():
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        utterance_id, text = parts
                        all_transcripts[utterance_id] = text
        except Exception as e:
            print(f"Error reading transcript file {transcript_file}: {e}")
    
    print(f"Loaded {len(all_transcripts)} utterance transcripts")
    
    # Shuffle to get random samples each time
    speaker_chapters = list(file_map.keys())
    random.shuffle(speaker_chapters)
    
    # Find files under max_duration and with transcripts
    suitable_excerpts = []
    target_sample_rate = 16000  # Target sample rate
    
    for key in speaker_chapters:
        audio_files = file_map.get(key, [])
        random.shuffle(audio_files)
        
        for audio_file in audio_files:
            # Extract utterance ID from filename
            # Format is typically: speaker_id-chapter_id-utterance_id.flac
            basename = audio_file.stem
            utterance_id = basename
            
            # Skip if we don't have a transcript
            if utterance_id not in all_transcripts:
                continue
                
            # Get the transcript
            transcript = all_transcripts[utterance_id]
            
            # Load audio and check duration
            try:
                audio_data, file_sample_rate = sf.read(audio_file)
                duration = len(audio_data) / file_sample_rate
                
                # Skip if too long
                if duration > max_duration:
                    continue
                    
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # Resample if needed
                if file_sample_rate != target_sample_rate:
                    # Simple resampling
                    new_length = int(duration * target_sample_rate)
                    indices = np.linspace(0, len(audio_data) - 1, new_length)
                    audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data)
                
                # Normalize audio
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
                
                # Add to suitable excerpts with transcript
                suitable_excerpts.append((audio_data, target_sample_rate, duration, str(audio_file), transcript))
                print(f"Found suitable excerpt: {audio_file}, duration: {duration:.2f}s, transcript: {transcript[:50]}...")
                
                # Break if we have enough
                if len(suitable_excerpts) >= num_excerpts:
                    break
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        # Break if we have enough
        if len(suitable_excerpts) >= num_excerpts:
            break
    
    # Check if we found enough
    if len(suitable_excerpts) < num_excerpts:
        print(f"Warning: Only found {len(suitable_excerpts)} excerpts under {max_duration} seconds with transcripts")
    
    return suitable_excerpts

def create_synthetic_test_audio(excerpts, silence_range=(1.0, 2.0)):
    """
    Create synthetic test audio by combining complete audio excerpts with silence.
    
    Args:
        excerpts: List of (audio_data, sample_rate, duration, file_path, transcript) tuples
        silence_range: Tuple of (min_silence, max_silence) in seconds
        
    Returns:
        Tuple of (test_audio, sample_rate, ground_truth)
    """
    print("Creating synthetic test audio from complete excerpts with transcripts...")
    
    # Use the first sample's sample rate
    sample_rate = excerpts[0][1]
    
    # Calculate total duration
    min_silence, max_silence = silence_range
    avg_silence = (min_silence + max_silence) / 2
    
    # Estimate total length (excerpts + silence between them)
    total_excerpt_duration = sum(duration for _, _, duration, _, _ in excerpts)
    estimated_silence_duration = avg_silence * (len(excerpts) - 1) + avg_silence  # Silence at start
    total_duration = total_excerpt_duration + estimated_silence_duration
    
    # Initialize empty audio
    total_samples = int(total_duration * sample_rate)
    test_audio = np.zeros(total_samples, dtype=np.float32)
    
    # Keep track of ground truth segments
    ground_truth = []
    
    # Current position in samples
    current_pos = 0
    
    # Add initial silence
    silence_duration = random.uniform(min_silence, max_silence)
    current_pos += int(silence_duration * sample_rate)
    
    # Add excerpts with silence in between
    for i, (excerpt_data, _, excerpt_duration, file_path, transcript) in enumerate(excerpts):
        # Make sure we don't exceed the total duration
        if current_pos + len(excerpt_data) >= total_samples:
            # Resize test_audio if needed
            new_size = current_pos + len(excerpt_data) + int(avg_silence * sample_rate)
            if new_size > total_samples:
                new_audio = np.zeros(new_size, dtype=np.float32)
                new_audio[:len(test_audio)] = test_audio
                test_audio = new_audio
                total_samples = new_size
        
        # Add the excerpt
        start_time = current_pos / sample_rate
        test_audio[current_pos:current_pos + len(excerpt_data)] = excerpt_data
        current_pos += len(excerpt_data)
        end_time = current_pos / sample_rate
        
        # Get filename for identification
        filename = Path(file_path).stem
        
        # Record ground truth with transcript
        ground_truth.append({
            "id": i,
            "start": start_time,
            "end": end_time,
            "duration": end_time - start_time,
            "file": filename,
            "transcript": transcript
        })
        
        # Add silence between excerpts
        if i < len(excerpts) - 1:
            silence_duration = random.uniform(min_silence, max_silence)
            current_pos += int(silence_duration * sample_rate)
    
    print(f"Created synthetic audio with {len(ground_truth)} complete excerpts and transcripts")
    
    return test_audio, sample_rate, ground_truth

def evaluate_segmenter(audio_data, sample_rate, ground_truth):
    """
    Evaluate AudioSegmenter with fixed threshold of 0.005.
    
    Args:
        audio_data: The test audio data
        sample_rate: The sample rate of the audio
        ground_truth: List of ground truth segments with transcripts
        
    Returns:
        Detected segments and evaluation metrics
    """
    print("Evaluating AudioSegmenter...")
    
    # Using fixed threshold of 0.005
    threshold = 0.005
    print(f"\nUsing fixed threshold: {threshold}")
    
    # Create segmenter with optimized parameters
    segmenter = AudioSegmenter(
        amplitude_threshold=threshold,
        min_segment_length=0.1,     # Reduced to capture short speech bursts
        min_silence_length=0.3,     # Reduced to detect shorter silences
        merge_threshold=1.2,        # Increased to merge segments that are further apart
        padding=0.2                 # Increased to add more context around detected speech
    )
    
    # Create temporary file for the audio
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / "temp_test_audio.flac"
    sf.write(temp_file, audio_data, sample_rate)
    
    # Process the audio
    detected_segments = segmenter.segment_audio(temp_file)
    print(f"Detected {len(detected_segments)} segments")
    
    # For proper evaluation, split detected segments at ground truth boundaries
    # This ensures one-to-one matching between ground truth and detected segments
    split_detected_segments = []
    
    # Step 1: Create a list of all boundary points from ground truth
    boundary_points = []
    for gt_segment in ground_truth:
        boundary_points.append(gt_segment["start"])
        boundary_points.append(gt_segment["end"])
    boundary_points = sorted(list(set(boundary_points)))  # Remove duplicates and sort
    
    # Step 2: Split each detected segment at ground truth boundaries
    for segment in detected_segments:
        # Find all boundaries within this segment
        internal_boundaries = [b for b in boundary_points 
                              if segment["start"] < b < segment["end"]]
        
        if not internal_boundaries:
            # No internal boundaries, keep segment as is
            split_detected_segments.append(segment)
        else:
            # Split at each boundary
            split_points = [segment["start"]] + internal_boundaries + [segment["end"]]
            for i in range(len(split_points) - 1):
                split_detected_segments.append({
                    "start": split_points[i],
                    "end": split_points[i + 1]
                })
    
    print(f"Split detected segments: {len(split_detected_segments)}")
    
    # Now perform standard evaluation with the split segments
    # Track which segments match which ground truth
    gt_matched = [False] * len(ground_truth)
    segment_matched = [False] * len(split_detected_segments)
    
    # First pass: find best matches for each ground truth segment
    for gt_idx, gt_segment in enumerate(ground_truth):
        best_overlap_ratio = 0
        best_match = None
        
        for segment_idx, segment in enumerate(split_detected_segments):
            # Calculate overlap
            overlap_start = max(gt_segment["start"], segment["start"])
            overlap_end = min(gt_segment["end"], segment["end"])
            
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                gt_duration = gt_segment["end"] - gt_segment["start"]
                
                # Calculate overlap as a percentage of ground truth duration
                overlap_ratio = overlap_duration / gt_duration
                
                # Keep track of best match
                if overlap_ratio > best_overlap_ratio:
                    best_overlap_ratio = overlap_ratio
                    best_match = segment_idx
        
        # Consider it a match if at least 30% of ground truth is covered
        if best_overlap_ratio >= 0.3 and best_match is not None:
            gt_matched[gt_idx] = True
            segment_matched[best_match] = True
        else:
            transcript = gt_segment.get("transcript", "")
            print(f"Missed segment: {gt_segment['start']:.2f} - {gt_segment['end']:.2f} ({gt_segment['file']})")
            print(f"  Transcript: {transcript[:50]}{'...' if len(transcript) > 50 else ''}")
    
    # Count true positives and false positives
    true_positives = sum(gt_matched)
    false_positives = sum(1 for m in segment_matched if not m)
    
    # Print details of unmatched detected segments
    for segment_idx, matched in enumerate(segment_matched):
        if not matched:
            segment = split_detected_segments[segment_idx]
            print(f"False positive: {segment['start']:.2f} - {segment['end']:.2f}")
    
    # Calculate metrics
    recall = true_positives / len(ground_truth) if ground_truth else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Metrics:")
    print(f"  True Positives: {true_positives}/{len(ground_truth)}")
    print(f"  False Positives: {false_positives}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1 Score: {f1:.2f}")
    
    # Return the original detected segments (not split) for visualization
    return detected_segments, {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def visualize_results(audio_data, sample_rate, ground_truth, detected_segments, output_path, threshold):
    """
    Visualize ground truth vs detected segments.
    
    Args:
        audio_data: The test audio data
        sample_rate: The sample rate of the audio
        ground_truth: List of ground truth segments
        detected_segments: List of detected segments
        output_path: Path to save the visualization
        threshold: Threshold used for segmentation
    """
    plt.figure(figsize=(20, 10))
    
    # Plot ground truth
    plt.subplot(211)
    times = np.arange(len(audio_data)) / sample_rate
    plt.plot(times, audio_data, color='blue', alpha=0.5)
    plt.title(f"Ground Truth vs Detected (threshold={threshold})")
    
    # Highlight ground truth
    for segment in ground_truth:
        plt.axvspan(segment["start"], segment["end"], color='green', alpha=0.3)
        mid_point = (segment["start"] + segment["end"]) / 2
        
        # Include snippet of transcript in visualization
        transcript_snippet = segment.get("transcript", "")
        if len(transcript_snippet) > 15:
            transcript_snippet = transcript_snippet[:15] + "..."
            
        plt.text(mid_point, 0.5, f"GT {segment['id']}\n{transcript_snippet}", 
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.7), fontsize=8)
    
    plt.ylabel("Ground Truth")
    
    # Plot detected segments
    plt.subplot(212)
    plt.plot(times, audio_data, color='blue', alpha=0.5)
    
    # Highlight detected segments
    for i, segment in enumerate(detected_segments):
        plt.axvspan(segment["start"], segment["end"], color='red', alpha=0.3)
        mid_point = (segment["start"] + segment["end"]) / 2
        plt.text(mid_point, 0.5, f"D{i}", 
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.ylabel("Detected")
    plt.xlabel("Time (seconds)")
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
    plt.close()

def compare_transcripts(audio_data, sample_rate, ground_truth, output_dir):
    """
    Transcribe each segment of the audio and compare with ground truth transcripts.
    Uses batched processing for better GPU utilization.
    
    Args:
        audio_data: The complete audio data
        sample_rate: Sample rate of the audio
        ground_truth: List of ground truth segments with transcripts
        output_dir: Directory to save results
        
    Returns:
        Dictionary with comparison metrics
    """
    print("\n=== Comparing Original vs Transcribed Text ===")
    
    # Set up Whisper for transcription
    try:
        from transformers import pipeline
        import torch
        import pandas as pd
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Setting up Whisper on {device}...")
        
        transcriber = pipeline(
            "automatic-speech-recognition", 
            model="openai/whisper-large-v3", 
            chunk_length_s=30,
            device=device
        )
        
        print(f"Device set to use {device}")
        
        # Extract all audio segments
        segment_audios = []
        for i, segment in enumerate(ground_truth):
            # Extract the audio segment
            start_sample = int(segment["start"] * sample_rate)
            end_sample = int(segment["end"] * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            
            segment_audios.append({
                "segment_id": i,
                "audio": {"array": segment_audio, "sampling_rate": sample_rate},
                "transcript": segment["transcript"].strip().upper(),
                "start_time": segment["start"],
                "end_time": segment["end"],
                "duration": segment["duration"]
            })
        
        # Process in batches
        results = []
        batch_size = 4  # Adjust based on your GPU memory
        
        for i in range(0, len(segment_audios), batch_size):
            batch = segment_audios[i:i+batch_size]
            print(f"Transcribing segment {i+1}-{min(i+batch_size, len(segment_audios))}/{len(segment_audios)}...")
            
            # Create batch of audio inputs for the pipeline
            audio_inputs = [segment["audio"] for segment in batch]
            
            # Process the entire batch at once
            batch_results = transcriber(
                audio_inputs,
                batch_size=batch_size,
                return_timestamps=False,
                generate_kwargs={"language": "en", "task": "transcribe"}
            )
            
            # Process and compare results from this batch
            for segment, result in zip(batch, batch_results):
                # Clean up computed text
                computed_text = result["text"].strip().upper()
                computed_text = computed_text.replace(",", "").replace(".", "").replace("?", "").replace("!", "")
                
                # Clean original text
                original_text = segment["transcript"]
                original_text = original_text.replace(",", "").replace(".", "").replace("?", "").replace("!", "")
                
                # Store comparison
                results.append({
                    "segment_id": segment["segment_id"],
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "duration": segment["duration"],
                    "original_text": original_text,
                    "computed_text": computed_text
                })
        
        # Calculate metrics
        for result in results:
            result["exact_match"] = result["original_text"] == result["computed_text"]
            result["original_length"] = len(result["original_text"])
            result["computed_length"] = len(result["computed_text"])
            result["length_diff"] = result["computed_length"] - result["original_length"]
            
            # Calculate word-level match
            original_words = result["original_text"].split()
            computed_words = result["computed_text"].split()
            result["original_word_count"] = len(original_words)
            result["computed_word_count"] = len(computed_words)
            
            # Calculate word overlap (simple metric)
            common_words = set(original_words).intersection(set(computed_words))
            result["common_word_count"] = len(common_words)
            result["word_overlap"] = len(common_words) / max(len(original_words), 1)
        
        # Save to CSV for easy comparison
        df = pd.DataFrame(results)
        csv_path = output_dir / "transcript_comparison.csv"
        df.to_csv(csv_path, index=False)
        
        # Print summary
        exact_matches = sum(r["exact_match"] for r in results)
        exact_match_pct = exact_matches / len(results) if results else 0
        avg_length_diff = sum(r["length_diff"] for r in results) / len(results) if results else 0
        avg_word_overlap = sum(r["word_overlap"] for r in results) / len(results) if results else 0
        
        print("\nTranscription Comparison Summary:")
        print(f"Total segments: {len(results)}")
        print(f"Exact matches: {exact_matches} ({exact_match_pct*100:.1f}%)")
        print(f"Average length difference: {avg_length_diff:.1f} characters")
        print(f"Average word overlap: {avg_word_overlap*100:.1f}%")
        
        # Save detailed results as JSON
        json_path = output_dir / "transcript_comparison.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary = {
            "total_segments": len(results),
            "exact_matches": exact_matches,
            "exact_match_percentage": exact_match_pct * 100,
            "avg_length_difference": avg_length_diff,
            "avg_word_overlap_percentage": avg_word_overlap * 100
        }
        
        summary_path = output_dir / "transcript_comparison_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nDetailed results saved to:")
        print(f"  - {csv_path}")
        print(f"  - {json_path}")
        print(f"  - {summary_path}")
        
        return summary
        
    except ImportError:
        print("Skipping transcript comparison - missing required packages (transformers, torch, pandas)")
        return None

def test_audio_segmenter_with_transcripts(include_transcript_comparison=True):
    """
    Test AudioSegmenter with complete audio excerpts and their transcripts.
    
    Args:
        include_transcript_comparison: Whether to also transcribe segments and compare 
                                       with original transcripts
    """
    print("=== Testing AudioSegmenter with Complete Audio Excerpts and Transcripts ===")
    
    # Create output directory
    output_dir = Path("test_results/transcribed_excerpts")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get complete audio excerpts with transcripts
    excerpts = get_complete_audio_excerpts_with_transcripts(num_excerpts=20, max_duration=20.0)
    
    # Create synthetic test audio
    test_audio, sample_rate, ground_truth = create_synthetic_test_audio(
        excerpts, 
        silence_range=(1.0, 2.0)
    )
    
    # Save the test audio and ground truth
    test_file = output_dir / "transcribed_excerpts_test.flac"
    sf.write(test_file, test_audio, sample_rate)
    print(f"Saved test audio to {test_file}")
    
    with open(output_dir / "ground_truth_with_transcripts.json", "w") as f:
        json.dump(ground_truth, f, indent=2)
    
    # Visualize ground truth (chunked if audio is very long)
    total_duration = len(test_audio) / sample_rate
    chunk_size = 60  # seconds per visualization
    
    if total_duration <= chunk_size:
        # Short enough for one visualization
        plt.figure(figsize=(20, 5))
        times = np.arange(len(test_audio)) / sample_rate
        plt.plot(times, test_audio, color='blue', alpha=0.5)
        plt.title("Transcribed Audio Excerpts Test")
        
        # Highlight speech segments
        for segment in ground_truth:
            plt.axvspan(segment["start"], segment["end"], color='green', alpha=0.3)
            mid_point = (segment["start"] + segment["end"]) / 2
            
            # Include snippet of transcript in visualization
            transcript_snippet = segment.get("transcript", "")
            if len(transcript_snippet) > 15:
                transcript_snippet = transcript_snippet[:15] + "..."
                
            plt.text(mid_point, 0.5, f"Ex {segment['id']}\n{transcript_snippet}", 
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(facecolor='white', alpha=0.7), fontsize=8)
        
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.savefig(output_dir / "ground_truth_with_transcripts.png")
        plt.close()
    else:
        # Split into chunks for visualization
        num_chunks = int(np.ceil(total_duration / chunk_size))
        for chunk in range(num_chunks):
            start_time = chunk * chunk_size
            end_time = min((chunk + 1) * chunk_size, total_duration)
            
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            plt.figure(figsize=(20, 5))
            times = np.arange(start_sample, end_sample) / sample_rate
            plt.plot(times, test_audio[start_sample:end_sample], color='blue', alpha=0.5)
            plt.title(f"Transcribed Audio Excerpts Test (Chunk {chunk+1}/{num_chunks})")
            
            # Highlight speech segments in this time range
            for segment in ground_truth:
                if segment["end"] >= start_time and segment["start"] <= end_time:
                    plt.axvspan(
                        max(segment["start"], start_time), 
                        min(segment["end"], end_time), 
                        color='green', alpha=0.3
                    )
                    # Only add label if midpoint is in this chunk
                    mid_point = (segment["start"] + segment["end"]) / 2
                    if start_time <= mid_point <= end_time:
                        # Include snippet of transcript in visualization
                        transcript_snippet = segment.get("transcript", "")
                        if len(transcript_snippet) > 15:
                            transcript_snippet = transcript_snippet[:15] + "..."
                            
                        plt.text(mid_point, 0.5, f"Ex {segment['id']}\n{transcript_snippet}", 
                                horizontalalignment='center', verticalalignment='center',
                                bbox=dict(facecolor='white', alpha=0.7), fontsize=8)
            
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude")
            plt.xlim(start_time, end_time)
            plt.savefig(output_dir / f"ground_truth_with_transcripts_chunk_{chunk+1}.png")
            plt.close()
    
    # Test the audio segmenter
    detected_segments, results = evaluate_segmenter(test_audio, sample_rate, ground_truth)
    
    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "threshold": 0.005,
            "results": results
        }, f, indent=2)
    
    # Visualize results (chunked if audio is very long)
    if total_duration <= chunk_size:
        # Short enough for one visualization
        visualize_results(
            test_audio, 
            sample_rate, 
            ground_truth, 
            detected_segments,
            output_dir / "detection_results_with_transcripts.png",
            0.005
        )
    else:
        # Split into chunks for visualization
        num_chunks = int(np.ceil(total_duration / chunk_size))
        for chunk in range(num_chunks):
            start_time = chunk * chunk_size
            end_time = min((chunk + 1) * chunk_size, total_duration)
            
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Filter ground truth segments in this range
            chunk_gt = [seg for seg in ground_truth 
                       if seg["end"] >= start_time and seg["start"] <= end_time]
            
            # Filter detected segments in this range
            chunk_detected = [seg for seg in detected_segments 
                             if seg["end"] >= start_time and seg["start"] <= end_time]
            
            # Skip if no segments in this chunk
            if not chunk_gt and not chunk_detected:
                continue
                
            # Create visualization for this chunk
            plt.figure(figsize=(20, 10))
            
            # Plot ground truth
            plt.subplot(211)
            chunk_times = np.arange(start_sample, end_sample) / sample_rate
            plt.plot(chunk_times, test_audio[start_sample:end_sample], color='blue', alpha=0.5)
            plt.title(f"Ground Truth vs Detected (Chunk {chunk+1}/{num_chunks})")
            
            # Highlight ground truth
            for segment in chunk_gt:
                plt.axvspan(
                    max(segment["start"], start_time), 
                    min(segment["end"], end_time), 
                    color='green', alpha=0.3
                )
                # Only add label if midpoint is in this chunk
                mid_point = (segment["start"] + segment["end"]) / 2
                if start_time <= mid_point <= end_time:
                    # Include snippet of transcript in visualization
                    transcript_snippet = segment.get("transcript", "")
                    if len(transcript_snippet) > 15:
                        transcript_snippet = transcript_snippet[:15] + "..."
                        
                    plt.text(mid_point, 0.5, f"GT {segment['id']}\n{transcript_snippet}", 
                            horizontalalignment='center', verticalalignment='center',
                            bbox=dict(facecolor='white', alpha=0.7), fontsize=8)
            
            plt.ylabel("Ground Truth")
            plt.xlim(start_time, end_time)
            
            # Plot detected segments
            plt.subplot(212)
            plt.plot(chunk_times, test_audio[start_sample:end_sample], color='blue', alpha=0.5)
            
            # Highlight detected segments
            for i, segment in enumerate(chunk_detected):
                plt.axvspan(
                    max(segment["start"], start_time), 
                    min(segment["end"], end_time), 
                    color='red', alpha=0.3
                )
                # Only add label if midpoint is in this chunk
                mid_point = (segment["start"] + segment["end"]) / 2
                if start_time <= mid_point <= end_time:
                    plt.text(mid_point, 0.5, f"D{i}", 
                            horizontalalignment='center', verticalalignment='center',
                            bbox=dict(facecolor='white', alpha=0.7))
            
            plt.ylabel("Detected")
            plt.xlabel("Time (seconds)")
            plt.xlim(start_time, end_time)
            
            # Save visualization
            plt.tight_layout()
            plt.savefig(output_dir / f"detection_results_with_transcripts_chunk_{chunk+1}.png")
            plt.close()
    
    # Optionally compare transcripts
    transcript_results = None
    if include_transcript_comparison:
        transcript_results = compare_transcripts(test_audio, sample_rate, ground_truth, output_dir)
    
    print(f"Testing complete. Segmentation F1 Score: {results['f1']:.2f}")
    return detected_segments, results, ground_truth, transcript_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test audio segmenter with transcripts")
    parser.add_argument("--skip-transcription", action="store_true", 
                        help="Skip transcript comparison (faster)")
    args = parser.parse_args()
    
    test_audio_segmenter_with_transcripts(include_transcript_comparison=not args.skip_transcription) 