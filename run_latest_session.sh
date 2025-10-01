#!/bin/bash

# Script to automatically process the latest D&D session in the recordings folder

echo "Finding the latest session in recordings folder..."

# Define the recordings directory
RECORDINGS_DIR="./recordings"

# Check if recordings directory exists
if [ ! -d "$RECORDINGS_DIR" ]; then
    echo "Error: Recordings directory not found at: $RECORDINGS_DIR" >&2
    echo "Please ensure you're running this script from the rpgnotes project root directory." >&2
    exit 1
fi

# Find the latest session directory (most recently modified)
# -maxdepth 1 prevents it from going into subdirectories of the sessions
LATEST_SESSION_PATH=$(find "$RECORDINGS_DIR" -maxdepth 1 -mindepth 1 -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

# Check if a directory was found
if [ -z "$LATEST_SESSION_PATH" ]; then
    echo "Error: No session directories found in: $RECORDINGS_DIR" >&2
    exit 1
fi

LATEST_SESSION_NAME=$(basename "$LATEST_SESSION_PATH")
LAST_MODIFIED=$(stat -c %y "$LATEST_SESSION_PATH")

echo "Latest session found: $LATEST_SESSION_NAME"
echo "Path: $LATEST_SESSION_PATH"
echo "Last modified: $LAST_MODIFIED"

# Check if the session processing script exists
SESSION_PROCESSING_SCRIPT="process_dnd_session.py"
if [ ! -f "$SESSION_PROCESSING_SCRIPT" ]; then
    echo "Error: Session processing script not found at: $SESSION_PROCESSING_SCRIPT" >&2
    exit 1
fi

# Activate conda environment if it's not already active
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "rpgnotes" ]; then
    echo ""
    echo "Activating 'rpgnotes' conda environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate rpgnotes
    if [ $? -ne 0 ]; then
        echo "Error: Failed to activate conda environment 'rpgnotes'." >&2
        echo "Please make sure it has been created successfully." >&2
        exit 1
    fi
fi

echo ""
echo "Starting processing of latest session..."
echo "========================================"

# Run the session processing script with the latest session directory
# Using `python` will now use the one from the activated conda environment
python "$SESSION_PROCESSING_SCRIPT" "$LATEST_SESSION_PATH"

# Deactivate the environment after finishing
# echo "Deactivating conda environment."
# conda deactivate # Optional: you might want to stay in the env

echo "========================================"
echo "Latest session processing completed." 