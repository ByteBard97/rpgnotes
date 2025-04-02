"""
Configuration module for RPG Notes Automator.

This module handles loading and managing configuration settings from both
environment variables (.env) and configuration file (config.json).
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables (sensitive data)
load_dotenv()

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json file."""
    config_file = Path("config.json")
    if not config_file.exists():
        raise FileNotFoundError(
            "config.json not found. Please create it using the template in the documentation."
        )
    
    with open(config_file, "r") as f:
        return json.load(f)

# Load configuration
try:
    CONFIG = load_config()
except Exception as e:
    print(f"Error loading config.json: {e}")
    print("Using default configuration...")
    CONFIG = {
        "directories": {
            "output": "./output",
            "temp": "./temp",
            "chat_log_source": "./source/chatlogs",
            "audio_source": "./source/audio",
            "context": "./context"
        },
        "files": {
            "discord_mapping": "./discord_speaker_mapping.json",
            "whisper_prompt": "./prompts/whisper_prompt.txt",
            "summary_prompt": "./prompts/summary_prompt.txt",
            "details_prompt": "./prompts/details_prompt.txt",
            "template": "./template.md"
        },
        "models": {
            "gemini": "gemini-1.5-pro"
        },
        "settings": {
            "delete_temp_files": True,
            "audio_quality": "High"
        }
    }

# Directory paths
OUTPUT_DIR = Path(CONFIG["directories"]["output"])
TEMP_DIR = Path(CONFIG["directories"]["temp"])
CHAT_LOG_SOURCE_DIR = Path(CONFIG["directories"]["chat_log_source"])
AUDIO_SOURCE_DIR = Path(CONFIG["directories"]["audio_source"])
CONTEXT_DIR = Path(CONFIG["directories"]["context"])

# File paths
DISCORD_MAPPING_FILE = Path(CONFIG["files"]["discord_mapping"])
WHISPER_PROMPT_FILE = Path(CONFIG["files"]["whisper_prompt"])
SUMMARY_PROMPT_FILE = Path(CONFIG["files"]["summary_prompt"])
DETAILS_PROMPT_FILE = Path(CONFIG["files"]["details_prompt"])
TEMPLATE_FILE = Path(CONFIG["files"]["template"])

# API keys and models (from .env)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = CONFIG["models"]["gemini"]

# Settings
DELETE_TEMP_FILES = CONFIG["settings"]["delete_temp_files"]
AUDIO_QUALITY = CONFIG["settings"]["audio_quality"]

# Derived paths
CHAT_LOG_OUTPUT_DIR = OUTPUT_DIR / "_chat_log"
TRANSCRIPTIONS_OUTPUT_DIR = OUTPUT_DIR / "_transcripts"
AUDIO_OUTPUT_DIR = TEMP_DIR / "audio"
TEMP_TRANSCRIPTIONS = TEMP_DIR / "transcriptions"

# Create directories if they don't exist
for directory in [OUTPUT_DIR, TEMP_DIR, CHAT_LOG_OUTPUT_DIR, AUDIO_OUTPUT_DIR, 
                 TRANSCRIPTIONS_OUTPUT_DIR, TEMP_TRANSCRIPTIONS, CONTEXT_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 