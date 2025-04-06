# RPG Notes Automator: Requirements Document

## 1. Overview
RPG Notes Automator is a desktop application that processes Discord/Craig bot recordings from RPG sessions, transcribes them, and generates quest logs using AI summarization.

## 2. Input Requirements
- **Primary Input**: Discord Craig bot recordings (.flac files or .zip archives containing .flac files)
- **Secondary Inputs**:
  - Discord username to character name mapping
  - Session context for summarization
  - Previous session summaries (optional)

## 3. Core Processing Features

### 3.1. Audio Processing
- Automatically segment audio to identify speech portions
- Transcribe speech using Whisper ASR models
- Map speakers to character names using provided discord-to-character mapping

### 3.2. Progress Tracking
- Display overall progress with percentage completion bar
- Show current file being processed and file-specific progress
- Allow processing to be paused and resumed
- Estimate remaining time based on current processing speed

### 3.3. GPU Optimization
- Automatically detect available GPU VRAM
- Select appropriate Whisper model size based on available resources
- Provide fallback to smaller models or CPU if insufficient GPU resources

### 3.4. Transcript Generation
- Generate individual transcripts per speaker/character
- Create combined chronological transcript with speaker labels
- Save transcripts in both raw and formatted versions for review

### 3.5. Summary Generation
- Connect to Gemini API for LLM-based summarization
- Generate structured quest log entries from transcripts
- Incorporate session context in summarization process

## 4. User Interface Requirements

### 4.1. Configuration Management
- Simple interface to configure Discord username to character name mapping
- Easy import of session context (text or PDF)
- Settings persistence across sessions

### 4.2. Processing Controls
- Start/pause/resume buttons for transcript processing
- Clear progress indicators showing:
  - Current file being processed
  - Overall completion percentage
  - Estimated time remaining
  - Current processing stage (extracting, segmenting, transcribing, summarizing)

### 4.3. Output Management
- Preview panel for generated transcripts and summaries
- Export options for different formats (TXT, MD, PDF)
- Organization of outputs by session date/number

## 5. Technical Requirements

### 5.1. Performance
- Efficient batch processing of audio segments
- GPU acceleration for transcription when available
- Memory-efficient processing for large audio files

### 5.2. Error Handling
- Graceful handling of interruptions (pausing/resuming)
- Recovery mechanisms for failed transcriptions
- Clear error messages for troubleshooting

### 5.3. Storage
- Configurable storage locations for temporary and output files
- Option to clean up temporary files after processing
- Reasonable file organization scheme

## 6. Implementation Priorities

### Phase 1: Core Processing
1. Audio segmentation and transcription pipeline
2. Proper progress tracking and pause/resume functionality
3. GPU detection and model selection

### Phase 2: UI Enhancements
1. Discord username to character mapping interface
2. Session context import functionality
3. Improved progress visualization

### Phase 3: Summarization
1. Gemini API integration
2. Quest log generation
3. Output formatting and export options 