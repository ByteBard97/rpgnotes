import argparse
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_silence

def find_split_point(audio, search_radius_ms=30000, silence_len_ms=1000):
    """Find a good split point (silence) near the middle of the audio."""
    duration_ms = len(audio)
    midpoint_ms = duration_ms // 2
    
    # Search for silence in a window around the midpoint
    search_start = max(0, midpoint_ms - search_radius_ms)
    search_end = min(duration_ms, midpoint_ms + search_radius_ms)
    audio_search_area = audio[search_start:search_end]
    
    # Calculate a silence threshold relative to the audio's loudness
    # A 16 dB drop is a common starting point for silence detection.
    silence_thresh_db = audio_search_area.dBFS - 16

    print(f"Searching for at least {silence_len_ms/1000}s of silence near the midpoint...")
    print(f"Loudness in search area: {audio_search_area.dBFS:.2f} dBFS, Silence threshold: {silence_thresh_db:.2f} dBFS")

    silences = detect_silence(
        audio_search_area,
        min_silence_len=silence_len_ms,
        silence_thresh=silence_thresh_db
    )

    if silences:
        # Find the silence chunk closest to the absolute midpoint of the whole file
        best_split_point = -1
        min_dist_to_midpoint = float('inf')

        for start, end in silences:
            # Calculate the middle of the detected silence, relative to the whole audio
            silence_midpoint_abs = search_start + start + ((end - start) // 2)
            dist = abs(midpoint_ms - silence_midpoint_abs)
            
            if dist < min_dist_to_midpoint:
                min_dist_to_midpoint = dist
                best_split_point = silence_midpoint_abs
        
        print(f"Found suitable silence. Splitting at {best_split_point / 1000:.2f}s.")
        return best_split_point

    else:
        # Fallback to splitting directly at the midpoint if no silence is found
        print("No suitable silence found near the midpoint. Splitting directly in half.")
        return midpoint_ms

def split_audio_file(file_path_str: str):
    """
    Splits an audio file into two halves at a point of silence near the middle.
    """
    file_path = Path(file_path_str)
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    print(f"Loading audio file: {file_path.name}...")
    try:
        audio = AudioSegment.from_file(file_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return
        
    split_at_ms = find_split_point(audio)
    
    part1 = audio[:split_at_ms]
    part2 = audio[split_at_ms:]
    
    # Create output filenames
    output_dir = file_path.parent
    base_name = file_path.stem
    extension = file_path.suffix
    
    output_path1 = output_dir / f"{base_name}_part1{extension}"
    output_path2 = output_dir / f"{base_name}_part2{extension}"
    
    print(f"Exporting part 1: {output_path1.name} ({len(part1)/1000:.2f}s)")
    part1.export(output_path1, format=extension.lstrip('.'))
    
    print(f"Exporting part 2: {output_path2.name} ({len(part2)/1000:.2f}s)")
    part2.export(output_path2, format=extension.lstrip('.'))
    
    print("\nSplit complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a FLAC audio file in half at a point of silence near the middle.")
    parser.add_argument("file_path", type=str, help="Path to the audio file to split.")
    
    args = parser.parse_args()
    
    split_audio_file(args.file_path) 