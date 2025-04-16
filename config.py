"""
Configuration module for RPG Notes Automator.

This module handles loading and managing configuration settings from both
environment variables (.env), a global configuration file (config.json),
and an optional session-specific configuration file (session_config.json).
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import copy

# Load environment variables (sensitive data like API keys)
load_dotenv()

# Module-level variable to store the loaded configuration
_loaded_config: Dict[str, Any] = {}

# Mapping from expected config names (used in imports) to keys in _loaded_config
_CONFIG_KEY_MAP: Dict[str, tuple] = {
    # Directories
    "OUTPUT_DIR": ("directories", "output"),
    "TEMP_DIR": ("directories", "temp"),
    "CHAT_LOG_SOURCE_DIR": ("directories", "chat_log_source"),
    "AUDIO_SOURCE_DIR": ("directories", "audio_source"),
    "CONTEXT_DIR": ("directories", "context"),
    "SESSION_TRANSCRIPTIONS_DIR": ("directories", "session_transcriptions_dir"),
    # Files
    "DISCORD_MAPPING_FILE": ("files", "discord_mapping_file"),
    "WHISPER_PROMPT_FILE": ("files", "whisper_prompt_file"),
    "SUMMARY_PROMPT_FILE": ("files", "summary_prompt_file"),
    "DETAILS_PROMPT_FILE": ("files", "details_prompt_file"),
    "TEMPLATE_FILE": ("files", "template_file"),
    "CHARACTER_DESCRIPTIONS_FILE": ("files", "character_descriptions_file"),
    "SESSION_CONTEXT_FILES": ("files", "session_context_files"),
    # API keys and models
    "GEMINI_API_KEY": ("api_keys", "gemini"),
    "GEMINI_MODEL_NAME": ("models", "gemini"),
    # Settings
    "DELETE_TEMP_FILES": ("settings", "delete_temp_files"),
    "AUDIO_QUALITY": ("settings", "audio_quality"),
    # Derived Paths (Handled specially below)
    "CHAT_LOG_OUTPUT_DIR": None,
    "TRANSCRIPTIONS_OUTPUT_DIR": None,
    "AUDIO_OUTPUT_DIR": None,
    "TEMP_TRANSCRIPTIONS": None,
}

# Set of keys known to typically hold single path strings or lists of path strings
_KNOWN_PATH_KEYS = {
    "output", "temp", "chat_log_source", "audio_source", "context", # Directories
    "session_transcriptions_dir",
    "discord_mapping_file", "whisper_prompt_file", "summary_prompt_file", 
    "details_prompt_file", "template_file", "character_descriptions_file", # Files
    "session_context_files" # List of files
}

def _resolve_paths(config_dict: Dict[str, Any], base_path: Path) -> Dict[str, Any]:
    """Recursively resolves relative paths in a config dictionary."""
    resolved_dict = {} 
    for key, value in config_dict.items():
        # Determine if the current key suggests a path or list of paths
        is_likely_path = key in _KNOWN_PATH_KEYS or "_dir" in key or "_file" in key or key.endswith("_path")
        is_likely_path_list = key in _KNOWN_PATH_KEYS or "_files" in key or "_paths" in key

        if isinstance(value, dict):
            resolved_dict[key] = _resolve_paths(value, base_path) # Recurse
        
        elif is_likely_path_list and isinstance(value, list):
            # Handle lists potentially containing paths
            resolved_list = []
            for item in value:
                if isinstance(item, str):
                    path_obj = Path(item)
                    resolved_list.append((base_path / path_obj).resolve() if not path_obj.is_absolute() else path_obj)
                else:
                    resolved_list.append(item) # Keep non-string items as is
            resolved_dict[key] = resolved_list

        elif is_likely_path and isinstance(value, str):
            # Handle single path strings
            path_obj = Path(value)
            # --- DEBUG PRINT ---
            resolved_path = None
            if not path_obj.is_absolute():
                resolved_path = (base_path / path_obj).resolve()
            else:
                resolved_path = path_obj
            if key == 'discord_mapping_file': # Only print for the key we care about
                print(f"[DEBUG _resolve_paths] key='{key}', value='{value}', path_obj='{path_obj}', base_path='{base_path}', resolved_path='{resolved_path}'")
            # --- END DEBUG ---
            resolved_dict[key] = resolved_path

        else:
            # Keep other types of values as is
            resolved_dict[key] = value
            
    return resolved_dict

def _recursive_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
    """Recursively updates a dictionary."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _recursive_update(base_dict[key], value)
        else:
            base_dict[key] = value


def load_final_config(session_dir_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads configuration, starting with global config.json, then merging
    session_config.json if provided and found. Resolves paths appropriately.

    Args:
        session_dir_path: Optional path to the specific session directory.

    Returns:
        A dictionary containing the final merged configuration.
    """
    project_root = Path(__file__).parent.resolve() # Assumes config.py is at the project root or similar level

    # --- 1. Load Global Config ---
    global_config_file = project_root / "config.json"
    if not global_config_file.exists():
        # Provide a default structure if global config is missing
        print(f"Warning: Global config.json not found at {global_config_file}. Using default structure.")
        base_config = {
            "directories": {"output": "./output", "temp": "./temp"},
            "files": {"discord_mapping": "./discord_speaker_mapping.json"},
            "models": {"gemini": "gemini-1.5-pro"},
            "settings": {"delete_temp_files": True, "audio_quality": "High"}
        }
    else:
        with open(global_config_file, "r") as f:
            base_config = json.load(f)

    # Resolve paths in global config relative to project root
    final_config = _resolve_paths(copy.deepcopy(base_config), project_root)
    # --- DEBUG PRINT ---
    print(f"[DEBUG config] Resolved Global Config discord_mapping_file: {final_config.get('files', {}).get('discord_mapping_file')}")

    # --- 2. Load and Merge Session Config (if applicable) ---
    session_config_loaded = False
    if session_dir_path:
        session_config_file = session_dir_path / "session_config.json"
        if session_config_file.exists():
            print(f"Loading session configuration from: {session_config_file}")
            with open(session_config_file, "r") as f:
                session_config_data = json.load(f)

            # Resolve paths within the session config relative to the session directory FIRST
            resolved_session_config = _resolve_paths(copy.deepcopy(session_config_data), session_dir_path)
            # --- DEBUG PRINT ---
            print(f"[DEBUG config] Resolved Session Config discord_mapping_file: {resolved_session_config.get('files', {}).get('discord_mapping_file')}")

            # Merge session config into the base config (potentially overwriting)
            _recursive_update(final_config, resolved_session_config)
            # --- DEBUG PRINT ---
            print(f"[DEBUG config] Final Config discord_mapping_file AFTER merge: {final_config.get('files', {}).get('discord_mapping_file')}")
            session_config_loaded = True
        else:
            print(f"No session_config.json found in {session_dir_path}. Using global/default configuration.")

    # --- 3. Integrate Environment Variables ---
    final_config["api_keys"] = {
        "gemini": os.getenv("GEMINI_API_KEY")
        # Add other API keys from .env if needed
    }

    # --- 4. Ensure core directories/files exist in config (use defaults if missing) ---
    final_config.setdefault("directories", {})
    final_config["directories"].setdefault("output", project_root / "output")
    final_config["directories"].setdefault("temp", project_root / "temp")

    final_config.setdefault("files", {})
    # The merge process should correctly handle the discord_mapping_file path
    # if specified in the session config. If not, __getattr__ can handle defaults.

    final_config.setdefault("models", {"gemini": "gemini-1.5-pro"})
    final_config.setdefault("settings", {"delete_temp_files": True, "audio_quality": "High"})

    return final_config

def initialize_config(session_dir_path: Optional[Path] = None) -> None:
    """
    Loads the final configuration (global + session) and stores it
    in the module-level _loaded_config variable. Creates necessary directories.
    This MUST be called once by the main script before accessing config values.
    """
    global _loaded_config
    _loaded_config = load_final_config(session_dir_path)

    # Create essential directories defined in the final config
    print("Ensuring required directories exist...")
    dirs_to_create = []
    output_dir = _loaded_config.get("directories", {}).get("output")
    temp_dir = _loaded_config.get("directories", {}).get("temp")
    session_transcriptions_dir = _loaded_config.get("directories", {}).get("session_transcriptions_dir")

    if output_dir:
        dirs_to_create.extend([
            output_dir,
            output_dir / "_chat_log",
            output_dir / "_transcripts"
        ])
    if temp_dir:
        dirs_to_create.extend([
            temp_dir,
            temp_dir / "audio",
            temp_dir / "transcriptions"
        ])
    if session_transcriptions_dir:
        dirs_to_create.append(session_transcriptions_dir)

    # Add other potential source dirs if they are defined and needed
    # Note: Source dirs arguably shouldn't be created by the script
    # chat_log_source = _loaded_config.get("directories", {}).get("chat_log_source")
    # audio_source = _loaded_config.get("directories", {}).get("audio_source")
    # context_dir = _loaded_config.get("directories", {}).get("context")
    # if chat_log_source: dirs_to_create.append(chat_log_source)
    # if audio_source: dirs_to_create.append(audio_source)
    # if context_dir: dirs_to_create.append(context_dir)

    created_count = 0
    for directory in dirs_to_create:
        if not directory.exists():
             try:
                 directory.mkdir(parents=True, exist_ok=True)
                 print(f"  Created directory: {directory}")
                 created_count += 1
             except Exception as e:
                 print(f"  Error creating directory {directory}: {e}") # Avoid try block rule violation
        else:
            # Optional: print message if dir already exists
            # print(f"  Directory already exists: {directory}")
            pass 
    if created_count == 0:
        print("  All required directories already exist.")


def __getattr__(name: str) -> Any:
    """
    Dynamically retrieves configuration values when accessed as module attributes.
    Called automatically when `from config import XYZ` is used and XYZ is accessed.
    Requires initialize_config() to have been called first.
    """
    # --- DEBUG PRINT for specific attribute access ---
    if name == "DISCORD_MAPPING_FILE":
        lookup_keys = _CONFIG_KEY_MAP.get(name)
        value = _loaded_config
        final_val = None
        try:
             for key in lookup_keys:
                 value = value[key]
             final_val = value
        except (KeyError, TypeError):
             final_val = "<Error during lookup>"
        print(f"[DEBUG config] __getattr__ returning for {name}: {final_val}")

    if not _loaded_config:
        # Raise an informative error if accessed before initialization
        raise AttributeError(
            f"Configuration accessed ('{name}') before config.initialize_config() was called."
        )

    # Handle derived paths dynamically
    if name == "CHAT_LOG_OUTPUT_DIR":
        od = _loaded_config.get("directories", {}).get("output")
        return od / "_chat_log" if od else None
    if name == "TRANSCRIPTIONS_OUTPUT_DIR":
        od = _loaded_config.get("directories", {}).get("output")
        return od / "_transcripts" if od else None
    if name == "AUDIO_OUTPUT_DIR":
        td = _loaded_config.get("directories", {}).get("temp")
        return td / "audio" if td else None
    if name == "TEMP_TRANSCRIPTIONS":
        raise AttributeError("Access config.SESSION_TRANSCRIPTIONS_DIR instead of config.TEMP_TRANSCRIPTIONS")

    # Look up standard keys using the map
    if name in _CONFIG_KEY_MAP:
        keys = _CONFIG_KEY_MAP[name]
        if keys is None: # Should only be for derived paths handled above
             raise AttributeError(f"Derived path '{name}' accessed incorrectly.")
        
        value = _loaded_config
        try:
            for key in keys:
                value = value[key]
            return value
        except KeyError:
             # Optionally return a default or raise a more specific error
             # print(f"Warning: Config key '{name}' ({keys}) not found in loaded config.")
             # For now, let's mimic typical attribute access failure
             raise AttributeError(f"Module 'config' has no attribute '{name}' (or value not found in loaded config)")
        except TypeError:
             # Handle case where intermediate key is not a dictionary
             raise AttributeError(f"Config structure issue accessing '{name}' ({keys}).")

    # If the name is not a recognized config key, raise AttributeError
    raise AttributeError(f"Module 'config' has no attribute '{name}'")

# --- Example Usage (Keep commented out) ---
# if __name__ == "__main__":
#     initialize_config() # Load global first
#     print("--- Global Config Access Example ---")
#     print(f"Output Dir: {OUTPUT_DIR}") # Direct access works via __getattr__
#     print(f"Discord Mapping: {DISCORD_MAPPING_FILE}")

#     session_path = Path("recordings/wednesday_april_9_2025").resolve()
#     if session_path.exists():
#         initialize_config(session_path) # Reload with session override
#         print("\n--- Session Config Access Example ---")
#         print(f"Output Dir: {OUTPUT_DIR}") # Should be the same unless overridden
#         print(f"Discord Mapping: {DISCORD_MAPPING_FILE}") # Should point to session file via __getattr__
#     else:
#         print(f"\nSession path {session_path} does not exist for example.") 