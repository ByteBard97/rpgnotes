# RPG Notes Automator - Design Document

## Overview
This document outlines the design for a user-friendly GUI application that automates the process of generating detailed session notes for tabletop role-playing games (TTRPGs) run over Discord.

## Technology Stack

### GUI Framework: PyQt6
- **Core Components**
  - PyQt6 for GUI framework
  - Qt Designer for UI design
  - Qt Linguist for translations
  - Qt Test for GUI testing
- **Key Features**
  - Signal/Slot system for event handling
  - QThread for background processing
  - QWebEngineView for markdown preview
  - QTreeView for file management
  - Custom widgets for specialized needs
  - Style sheets for theming
- **Development Tools**
  - Qt Designer for visual UI design
  - Qt Creator for development
  - PyInstaller for packaging
  - pytest-qt for testing

## GUI Requirements

### 1. File Management Section
- **Drag & Drop Interface**
  - QTreeView with custom model for file display
  - QMimeData for drag-drop handling
  - Custom drop zone widget
  - File type validation using QMimeDatabase
- **File Display**
  - QTreeView with custom model
  - QFileSystemWatcher for live updates
  - Custom item delegates for status indicators
  - Context menus for file operations

### 2. Configuration Section
- **Settings Panel**
  - QSettings for configuration persistence
  - QFormLayout for settings layout
  - Custom widgets for API key management
  - QComboBox for model selection
  - QFileDialog for directory selection
- **Configuration Persistence**
  - JSON/INI file storage
  - Settings migration system
  - Configuration validation
  - Auto-backup system

### 3. Progress Monitoring
- **Visual Progress Tracking**
  - QProgressBar with custom styling
  - QTextEdit for log display
  - QThread for background processing
  - Custom progress dialog
- **Error Handling**
  - QMessageBox for error display
  - Custom error dialog
  - Log viewer widget
  - Error recovery system

### 4. Session Management
- **Session Overview**
  - QTableView for session list
  - Custom model for session data
  - QSortFilterProxyModel for search/filter
  - Export dialog with format options
- **Session Details**
  - QTabWidget for different views
  - Custom metadata display
  - File relationship viewer
  - History timeline widget

### 5. Preview & Editing
- **Content Preview**
  - QWebEngineView for markdown
  - Custom markdown renderer
  - Template preview widget
  - Live preview system
- **Editing Capabilities**
  - QTextEdit with markdown support
  - Custom toolbar for formatting
  - Version control integration
  - Auto-save system

### 6. Advanced Features
- **Batch Processing**
  - QThreadPool for parallel processing
  - Queue management system
  - Progress tracking widget
  - Cancel/retry system
- **Scheduling**
  - QTimer for scheduling
  - Calendar widget integration
  - Reminder system
  - Background task manager
- **Customization**
  - QStyle system for themes
  - Language resource system
  - Shortcut manager
  - Layout persistence

## Packaging Strategy

### 1. Distribution Method
- **Executable Creation**
  - PyInstaller for standalone executable
  - Auto-py-to-exe for GUI packaging
  - Inno Setup for Windows installer
  - UPX compression for size optimization

### 2. Package Contents
- **Core Components**
  - Python runtime
  - Required libraries
  - Default configurations
  - Basic templates
- **Optional Components**
  - Whisper models (downloadable)
  - CUDA support (optional)
  - Additional language packs
  - Example templates

### 3. Directory Structure
```
RPGNotes/
├── RPGNotes.exe
├── models/
│   └── whisper-large-v3/
├── config/
│   ├── default.env
│   └── templates/
├── user_data/
│   ├── sessions/
│   ├── settings/
│   └── logs/
└── README.txt
```

### 4. Installation Requirements
- **System Requirements**
  - Windows 10/11
  - 4GB RAM minimum
  - 2GB free disk space
  - Internet connection for initial setup
- **Optional Requirements**
  - NVIDIA GPU for CUDA support
  - Additional disk space for Whisper models

### 5. First-Run Experience
- **Setup Wizard**
  - API key configuration
  - Directory selection
  - Model download options
  - Basic tutorial
- **Configuration**
  - Default settings
  - Template selection
  - Directory structure creation
  - Permission checks

## Technical Considerations

### 1. Performance
- **Memory Management**
  - QThread for background processing
  - Smart pointer usage
  - Resource cleanup in destructors
  - Memory-efficient data structures
- **Processing Optimization**
  - Thread pool for parallel tasks
  - Chunked file processing
  - Progress update throttling
  - Cancel handling with QThread

### 2. Security
- **API Key Storage**
  - QSettings with encryption
  - Secure credential storage
  - Access control system
  - Key rotation support
- **File Handling**
  - Safe file operations
  - Backup system
  - Error recovery
  - File integrity checks

### 3. Error Handling
- **User Feedback**
  - QMessageBox system
  - Custom error dialogs
  - Log viewer widget
  - Error recovery wizards
- **System Recovery**
  - Auto-save system
  - State preservation
  - Cleanup procedures
  - Error logging system

## Development Phases

### Phase 1: Core GUI
- Basic window setup
- File management widgets
- Progress monitoring
- Simple configuration
- Essential error handling

### Phase 2: Enhanced Features
- Session management
- Preview/editing
- Advanced configuration
- Batch processing
- Threading implementation

### Phase 3: Polish
- Theme system
- Language system
- Scheduling
- Performance optimization
- UI/UX refinement

### Phase 4: Distribution
- Installer creation
- Documentation
- Example content
- Update mechanism
- Release packaging

## Technical Requirements

### Python Environment
- Python 3.10+
- Virtual environment management
- Dependency management with pip
- Type hints throughout codebase
- PEP 8 compliance

### Dependencies
- PyQt6 >= 6.4.0
- pytest-qt >= 4.2.0
- black for code formatting
- mypy for type checking
- pylint for code analysis
- pytest for testing
- pytest-cov for coverage

### Development Environment
- VS Code with Python extensions
- Qt Designer for UI design
- Git for version control
- GitHub Actions for CI/CD
- Docker for development environment

## Architecture

### Design Pattern
- MVVM (Model-View-ViewModel) pattern
- ViewModels for business logic
- Models for data management
- Views for UI components
- Services for external interactions

### State Management
- QSettings for application state
- Observable pattern for UI updates
- Event bus for component communication
- State persistence strategy

### Component Communication
- Signal/Slot system for UI events
- Event bus for cross-component communication
- Service layer for external interactions
- Dependency injection for testing

## Development Workflow

### Version Control
- Git flow branching strategy
- Feature branches from develop
- Release branches from main
- Semantic versioning
- Conventional commits

### Code Review Process
- Pull request template
- Code review checklist
- Automated testing
- Style guide compliance
- Documentation updates

### CI/CD Pipeline
- GitHub Actions workflow
- Automated testing
- Code quality checks
- Build automation
- Release automation

### Documentation
- Code documentation
- API documentation
- User documentation
- Development guide
- Contributing guide 