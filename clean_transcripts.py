import re
import argparse
import json
from pathlib import Path
import difflib
from typing import Set, List, Tuple
from difflib import SequenceMatcher


def clean_text(text: str) -> str:
    """
    Cleans transcribed text by focusing ONLY on removing extreme repetition
    (e.g., "walls of text") to minimize false positives.
    """
    # --- Rule 1: Collapse severe word repetition ---
    # Looks for a single word repeated 10 or more times and collapses it to one instance.
    # This is targeted at the most obvious transcription errors.
    # The (?:\s*,?\s*) part allows for optional commas and spaces between repeats.
    text = re.sub(r"(\b\w+'?\w*\b)(?:\s*,?\s*\1){9,}", r"\1", text, flags=re.IGNORECASE)

    # --- Rule 2: Collapse severe short phrase repetition ---
    # Looks for a two-word phrase repeated 10 or more times.
    phrase_pattern = r"(\b\w+'?\w*\s+\w+'?\w*\b)(?:\s*,?\s*\1){9,}"
    text = re.sub(phrase_pattern, r"\1", text, flags=re.IGNORECASE)

    return text.strip()


def highlight_diff(original, cleaned):
    """
    Creates a compact, inline diff string highlighting changes between two texts.
    Removed words are wrapped in [- ... -] and added words in [+ ... +].
    """
    matcher = SequenceMatcher(None, original.split(), cleaned.split())
    output = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            output.append(" ".join(original.split()[i1:i2]))
        elif tag == 'delete':
            output.append(f"[- {' '.join(original.split()[i1:i2])} -]")
        elif tag == 'insert':
            output.append(f"[+ {' '.join(cleaned.split()[j1:j2])} +]")
        elif tag == 'replace':
            output.append(f"[- {' '.join(original.split()[i1:i2])} -]")
            output.append(f"[+ {' '.join(cleaned.split()[j1:j2])} +]")
    return " ".join(output)

def process_json_file(file_path, dry_run=False):
    """Processes a single JSON transcript file."""
    print(f"\n--- Processing {file_path.name} ---")
    try:
        original_content = file_path.read_text(encoding='utf-8')
        segments = json.loads(original_content)
        
        if not isinstance(segments, list):
            print("  Error: JSON file is not a list of segments. Skipping.")
            return

        cleaned_segments = []
        has_changes = False
        
        for segment in segments:
            if 'text' in segment and isinstance(segment['text'], str):
                original_text = segment['text']
                cleaned_text = clean_text(original_text)
                
                if original_text != cleaned_text:
                    has_changes = True
                    print(f"\n  Change in segment ID {segment.get('id', 'N/A')} "
                          f"({segment.get('start', 0):.2f}s - {segment.get('end', 0):.2f}s):")
                    
                    # Use the new inline diff function
                    inline_diff = highlight_diff(original_text, cleaned_text)
                    print(f"    {inline_diff}")
                
                segment['text'] = cleaned_text
            cleaned_segments.append(segment)

        if not has_changes:
            print("No changes needed.")
            return

        # If in dry-run mode, just print a message and stop processing this file.
        if dry_run:
            print("  (Dry run mode: No changes will be saved.)")
            return

        # Ask user if they want to save the cleaned version
        choice = input(f"\nSave cleaned version of {file_path.name}? (y/n): ").lower()
        if choice == 'y':
            new_file_path = file_path.with_suffix('.cleaned.json')
            with new_file_path.open('w', encoding='utf-8') as f:
                json.dump(cleaned_segments, f, indent=2)
            print(f"Cleaned file saved to: {new_file_path}")
        else:
            print(f"Skipping save.")

    except json.JSONDecodeError:
        print(f"  Error: Invalid JSON in {file_path.name}. Skipping.")
    except Exception as e:
        print(f"  Error processing file {file_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean transcription files by removing repetitions and other ASR artifacts."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a single file or a directory containing .json transcript files."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in trial mode to see potential changes without prompting to save them."
    )
    args = parser.parse_args()
    
    target_path = Path(args.path)

    if args.dry_run:
        print("--- RUNNING IN DRY-RUN MODE: NO FILES WILL BE MODIFIED ---\n")
    
    if not target_path.exists():
        print(f"Error: Path not found: {target_path}")
        exit(1)
        
    files_to_process = []
    if target_path.is_dir():
        print(f"Scanning directory: {target_path}")
        # Use rglob to find all 'transcriptions' subdirectories and then get json files
        transcription_dirs = list(target_path.rglob("transcriptions"))
        if not transcription_dirs:
            # Fallback for if they point directly at a session dir, or a dir with jsons
            transcription_dirs.append(target_path)

        for trans_dir in transcription_dirs:
            files = list(trans_dir.glob("*.json"))
            files_to_process.extend(files)

        if not files_to_process:
            print("No .json files found in any 'transcriptions' subdirectories.")
            exit(0)
            
        print(f"Found {len(files_to_process)} JSON files to process.")
        for file_path in files_to_process:
            if file_path.name.endswith('.cleaned.json'):
                continue
            process_json_file(file_path, dry_run=args.dry_run)

    elif target_path.is_file():
        if target_path.suffix == '.json':
            process_json_file(target_path, dry_run=args.dry_run)
        else:
            print("Error: Please provide a .json file.")
    
    print("\nDone.")
