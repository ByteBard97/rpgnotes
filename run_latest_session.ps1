# Script to automatically process the latest D&D session in the recordings folder

Write-Host "Finding the latest session in recordings folder..."

# Define the recordings directory
$recordingsDir = ".\recordings"

# Check if recordings directory exists
if (-not (Test-Path $recordingsDir)) {
    Write-Error "Recordings directory not found at: $recordingsDir"
    Write-Error "Please ensure you're running this script from the rpgnotes project root directory."
    exit 1
}

# Get all directories in the recordings folder, sorted by last write time (newest first)
$sessionDirs = Get-ChildItem -Path $recordingsDir -Directory | Sort-Object LastWriteTime -Descending

# Check if any directories were found
if ($sessionDirs.Count -eq 0) {
    Write-Error "No session directories found in: $recordingsDir"
    exit 1
}

# Get the latest (newest) session directory
$latestSession = $sessionDirs[0]
$latestSessionPath = $latestSession.FullName

Write-Host "Latest session found: $($latestSession.Name)"
Write-Host "Path: $latestSessionPath"
Write-Host "Last modified: $($latestSession.LastWriteTime)"

# Ask for confirmation
$confirmation = Read-Host "Do you want to process this session? (y/N)"
if ($confirmation -notmatch "^[Yy]$") {
    Write-Host "Processing cancelled."
    exit 0
}

# Check if the session processing script exists
$sessionProcessingScript = ".\run_session_processing.ps1"
if (-not (Test-Path $sessionProcessingScript)) {
    Write-Error "Session processing script not found at: $sessionProcessingScript"
    exit 1
}

Write-Host ""
Write-Host "Starting processing of latest session..."
Write-Host "========================================"

# Run the session processing script with the latest session directory
& $sessionProcessingScript $latestSessionPath

Write-Host "========================================"
Write-Host "Latest session processing completed." 