"""
Audio segmentation module for detecting speech segments in audio files.
"""

import os
import numpy as np
from pathlib import Path
import soundfile as sf
import librosa
from typing import List, Dict, Union, Tuple


class AudioSegmenter:
    """
    Preprocesses audio files to detect speech segments based on amplitude thresholding.
    """
    
    def __init__(
        self,
        amplitude_threshold: float = 0.02,  # Amplitude threshold for speech detection
        min_segment_length: float = 0.5,    # Minimum segment length in seconds
        min_silence_length: float = 0.5,    # Minimum silence length to split segments
        merge_threshold: float = 1.0,       # Merge segments closer than this (in seconds)
        padding: float = 0.1               # Add padding to beginning/end of segments (in seconds)
    ):
        """
        Initialize the AudioSegmenter with configurable parameters.
        
        Args:
            amplitude_threshold: Volume level (0.0-1.0) above which audio is considered speech
            min_segment_length: Minimum length in seconds for a valid speech segment
            min_silence_length: Minimum silence duration to consider as a break
            merge_threshold: Segments with gaps less than this will be merged
            padding: Amount of padding to add before/after each segment
        """
        self.amplitude_threshold = amplitude_threshold
        self.min_segment_length = min_segment_length
        self.min_silence_length = min_silence_length 
        self.merge_threshold = merge_threshold
        self.padding = padding
    
    def load_audio(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return as numpy array.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Try soundfile first (better for FLAC)
            audio_data, sample_rate = sf.read(str(file_path))
            
            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)
                
            return audio_data, sample_rate
            
        except Exception as e:
            print(f"Soundfile error: {e}")
            
            # Fall back to librosa
            try:
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                return audio_data, sample_rate
            except Exception as e2:
                print(f"Librosa error: {e2}")
                raise ValueError(f"Could not load audio file {file_path}")
    
    def detect_segments(self, audio_data: np.ndarray, sample_rate: int) -> List[Dict[str, float]]:
        """
        Detect speech segments in audio based on amplitude thresholding.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            List of dicts with 'start' and 'end' times in seconds
        """
        # Normalize audio
        audio_data = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else audio_data
        
        # Calculate frame-level energy (using RMS)
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop
        
        # Get RMS energy for each frame
        rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Apply threshold to detect speech frames
        speech_frames = rms > self.amplitude_threshold
        
        # Find speech segment boundaries
        segments = []
        in_segment = False
        segment_start = 0
        
        for i, is_speech in enumerate(speech_frames):
            frame_time = i * hop_length / sample_rate
            
            if is_speech and not in_segment:
                # Speech start
                segment_start = frame_time
                in_segment = True
                
            elif not is_speech and in_segment:
                # Speech end
                segment_end = frame_time
                
                # Only keep segments that exceed minimum length
                segment_length = segment_end - segment_start
                if segment_length >= self.min_segment_length:
                    # Add padding (ensure we don't go below 0 or exceed audio length)
                    padded_start = max(0, segment_start - self.padding)
                    padded_end = min(len(audio_data) / sample_rate, segment_end + self.padding)
                    
                    segments.append({
                        'start': padded_start,
                        'end': padded_end
                    })
                
                in_segment = False
        
        # Handle case where audio ends during speech
        if in_segment:
            segment_end = len(speech_frames) * hop_length / sample_rate
            
            # Only keep segments that exceed minimum length
            segment_length = segment_end - segment_start
            if segment_length >= self.min_segment_length:
                # Add padding (ensure we don't go below 0 or exceed audio length)
                padded_start = max(0, segment_start - self.padding)
                padded_end = min(len(audio_data) / sample_rate, segment_end + self.padding)
                
                segments.append({
                    'start': padded_start,
                    'end': padded_end
                })
        
        return segments
    
    def merge_close_segments(self, segments: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Merge segments that are close together in time.
        
        Args:
            segments: List of segment dicts with 'start' and 'end' times
            
        Returns:
            List of merged segment dicts
        """
        if not segments:
            return []
            
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        
        # Initialize with first segment
        merged = [sorted_segments[0]]
        
        # Merge subsequent segments if they're close enough
        for segment in sorted_segments[1:]:
            last_segment = merged[-1]
            
            # If this segment starts soon after the last one ends, merge them
            if segment['start'] - last_segment['end'] <= self.merge_threshold:
                last_segment['end'] = segment['end']  # Extend the previous segment
            else:
                merged.append(segment)  # Add as a new segment
                
        return merged
    
    def segment_audio(self, file_path: Union[str, Path]) -> List[Dict[str, float]]:
        """
        Process an audio file and return a list of speech segments.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            List of dicts with 'start' and 'end' times in seconds
        """
        # Load the audio
        audio_data, sample_rate = self.load_audio(file_path)
        
        # Detect segments based on amplitude
        segments = self.detect_segments(audio_data, sample_rate)
        
        # Merge segments that are close together
        merged_segments = self.merge_close_segments(segments)
        
        return merged_segments 