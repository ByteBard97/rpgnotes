"""
Data models for RPG Notes Automator.

This module defines the data structures used throughout the application using Pydantic.
"""

import datetime
from pydantic import BaseModel, Field, field_validator

class SessionData(BaseModel):
    """
    Pydantic model for session data extracted by Gemini.
    
    This model represents the structured data extracted from the session transcript,
    including session number, date, title, events, NPCs, locations, items, and image prompts.
    """
    number: int | None = Field(description="Session number.")
    date: datetime.date | None = Field(description="Session date. Try to find in context or use the date of the last Monday.")
    events: list[str] = Field(description="Short list of the most important events or decisions.")
    title: str = Field(description="Session title. Should be short but descriptive.")
    npcs: list[str] = Field(description="Short list of the most important NPCs.")
    locations: list[str] = Field(description="Short list of the most important locations.")
    items: list[str] = Field(description="Short list of the most important items.")
    images: list[str] = Field(description="""List of prompts to use in AI image generators in **English**.
                              Try not to use proper names, replace character names with descriptions of their appearance.
                              Use different artistic styles. Start each with the word 'Draw'.""")

    @field_validator("date", mode="before")
    @classmethod
    def validate_date(cls, value):
        """
        Validates and converts date strings to datetime.date objects.
        
        Args:
            value: The date value to validate
            
        Returns:
            A datetime.date object
            
        Raises:
            ValueError: If the date format is incorrect
        """
        if isinstance(value, str):
            for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
                try:
                    return datetime.datetime.strptime(value, fmt).date()
                except ValueError:
                    pass
            raise ValueError("Incorrect date format. Expected YYYY-MM-DD or DD.MM.YYYY.")

        return value or (datetime.date.today() - datetime.timedelta(days=datetime.date.today().weekday()))

class TranscriptionSegment(BaseModel):
    """
    Model for a single segment of transcribed audio.
    
    This model represents a single segment of transcribed audio, including
    the text, start time, end time, confidence, and speaker.
    """
    text: str = Field(description="The transcribed text of this segment.")
    start: float = Field(description="Start time of the segment in seconds.")
    end: float = Field(description="End time of the segment in seconds.")
    no_speech_prob: float = Field(description="Probability that this segment contains no speech.")
    speaker: str | None = Field(description="The speaker of this segment.", default=None) 