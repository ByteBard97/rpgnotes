"""
Session notes generation module for RPG Notes Automator.

This module handles generating session notes using the Gemini API.
"""

import time
from pathlib import Path
from typing import Tuple, Optional

import instructor
import google.generativeai as genai

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    SUMMARY_PROMPT_FILE,
    DETAILS_PROMPT_FILE,
    TEMPLATE_FILE,
    OUTPUT_DIR,
    CONTEXT_DIR
)
from models import SessionData
from utils import load_context_files, get_previous_summary_file

class SessionNotesGenerator:
    """
    Handles generating session notes using the Gemini API.
    
    This class is responsible for generating detailed session summaries
    and extracting structured data from transcripts.
    """
    
    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        model_name: str = GEMINI_MODEL_NAME,
        summary_prompt_file: Path = SUMMARY_PROMPT_FILE,
        details_prompt_file: Path = DETAILS_PROMPT_FILE,
        template_file: Path = TEMPLATE_FILE,
        output_dir: Path = OUTPUT_DIR,
        context_dir: Path = CONTEXT_DIR
    ):
        """
        Initialize the SessionNotesGenerator.
        
        Args:
            api_key: The Gemini API key
            model_name: The Gemini model name
            summary_prompt_file: File containing the summary prompt
            details_prompt_file: File containing the details prompt
            template_file: File containing the output template
            output_dir: Directory for output files
            context_dir: Directory containing context files
        """
        self.api_key = api_key
        self.model_name = model_name
        self.summary_prompt_file = summary_prompt_file
        self.details_prompt_file = details_prompt_file
        self.template_file = template_file
        self.output_dir = output_dir
        self.context_dir = context_dir
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
    
    def generate_session_notes(self, transcript_file: Path, session_number: int) -> Tuple[str, SessionData]:
        """
        Generates session notes using the Gemini API.
        
        Args:
            transcript_file: The path to the transcript file
            session_number: The session number
            
        Returns:
            A tuple containing the session summary and session data
        """
        context_data = load_context_files(self.context_dir)

        # --- Generate Detailed Summary ---
        with open(self.summary_prompt_file, "r") as f:
            summary_prompt = f.read()

        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=summary_prompt,
        )

        summary_messages = [
            {"role": "user", "parts": ["Please write a detailed and comprehensive summary."]},
        ]

        if context_data:
            summary_messages.append({"role": "user", "parts": ["Additional context:\n", context_data]})

        # Add previous summary for context if available
        previous_summary_file = get_previous_summary_file(session_number, self.output_dir)
        if previous_summary_file:
            print("Using previous summary for additional context.")
            with open(previous_summary_file, "r") as f:
                summary_messages.append({"role": "user", "parts": ["Summary from the previous session for additional context:\n", f.read()]})

        with open(transcript_file, "r") as f:
            summary_messages.append({"role": "user", "parts": ["Transcript:\n", f.read()]})

        summary_response = model.generate_content(
            summary_messages,
            generation_config=genai.GenerationConfig(),
            stream=False
        )
        session_summary = summary_response.text
        print("Session summary generated.")

        print(f"Session Summary: {session_summary=}")

        # --- Generate Details (Title, Events, etc.) ---
        print("Waiting for 10 seconds for next request.")
        time.sleep(10)

        with open(self.details_prompt_file, "r") as f:
            details_prompt = f.read()

        client = instructor.from_gemini(
            client=genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=details_prompt,
            ),
            mode=instructor.Mode.GEMINI_JSON,
        )

        details_messages = [
                {"role": "user", "content": "Please extract details from the following summary."},
                {"role": "user", "content": session_summary},
            ]
        if context_data:
            details_messages.append({"role": "user", "parts": ["Additional context:\n", context_data]})
        session_data = client.chat.completions.create(
            messages=details_messages,
            response_model=SessionData,
            max_retries=3,
        )
        print("Session details generated.")
        return session_summary, session_data
    
    def save_summary_file(self, session_summary: str, session_data: SessionData, session_number: int) -> Path:
        """
        Saves the generated summary to a Markdown file.
        
        Args:
            session_summary: The session summary
            session_data: The session data
            session_number: The session number
            
        Returns:
            The path to the saved file
        """
        with open(self.template_file, "r") as f:
            template = f.read()

        output = template.format(
            number=session_number,
            title=session_data.title,
            date=session_data.date.strftime("%d.%m.%Y"),
            summary=session_summary,
            events="\n".join(f"* {event}" for event in session_data.events),
            npcs="\n".join(f"* {npc}" for npc in session_data.npcs),
            locations="\n".join(f"* {loc}" for loc in session_data.locations),
            items="\n".join(f"* {item}" for item in session_data.items),
            images="\n".join(f"* {image}" for image in session_data.images),
        )

        output_file = self.output_dir / f"Session {session_number} - {session_data.title}.md"
        with open(output_file, "w") as f:
            f.write(output)
        print(f"Session notes saved to {output_file}")
        return output_file 