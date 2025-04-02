"""
Configuration module for RPG Notes Automator.

This module handles loading and managing configuration settings from environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration (from .env) ---
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))
TEMP_DIR = Path(os.getenv("TEMP_DIR", "./temp"))
CHAT_LOG_SOURCE_DIR = Path(os.getenv("CHAT_LOG_SOURCE_DIR", "./source/chatlogs"))
AUDIO_SOURCE_DIR = Path(os.getenv("AUDIO_SOURCE_DIR", "./source/audio"))
DISCORD_MAPPING_FILE = Path(os.getenv("DISCORD_MAPPING_FILE", "./discord_speaker_mapping.json"))
WHISPER_PROMPT_FILE = Path(os.getenv("WHISPER_PROMPT_FILE", "./prompts/whisper_prompt.txt"))
SUMMARY_PROMPT_FILE = Path(os.getenv("SUMMARY_PROMPT_FILE", "./prompts/summary_prompt.txt"))
DETAILS_PROMPT_FILE = Path(os.getenv("DETAILS_PROMPT_FILE", "./prompts/details_prompt.txt"))
TEMPLATE_FILE = Path(os.getenv("TEMPLATE_FILE", "./template.md"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DELETE_TEMP_FILES = os.getenv("DELETE_TEMP_FILES", "False").lower() == "true"
CONTEXT_DIR = Path(os.getenv("CONTEXT_DIR", "./context"))
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro")

# --- Setup Directories ---
CHAT_LOG_OUTPUT_DIR = OUTPUT_DIR / "_chat_log"
TRANSCRIPTIONS_OUTPUT_DIR = OUTPUT_DIR / "_transcripts"
AUDIO_OUTPUT_DIR = TEMP_DIR / "audio"
TEMP_TRANSCRIPTIONS = TEMP_DIR / "transcriptions"

# Create directories if they don't exist
for directory in [OUTPUT_DIR, TEMP_DIR, CHAT_LOG_OUTPUT_DIR, AUDIO_OUTPUT_DIR, 
                 TRANSCRIPTIONS_OUTPUT_DIR, TEMP_TRANSCRIPTIONS, CONTEXT_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 