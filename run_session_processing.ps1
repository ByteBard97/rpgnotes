param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$SessionDirectory
)

# Get the absolute path of the session directory
$absoluteSessionDir = Resolve-Path -Path $SessionDirectory

Write-Host "Starting session processing for: $absoluteSessionDir"

# --- Step 1: Deactivate Conda Environment ---
# Check if conda is active by looking for the CONDA_PREFIX environment variable
if ($env:CONDA_PREFIX) {
    Write-Host "Deactivating active Conda environment..."
    try {
        conda deactivate
    } catch {
        Write-Warning "Could not execute 'conda deactivate'. Please ensure conda is in your PATH if you intended to use it. Continuing..."
    }
} else {
    Write-Host "No active Conda environment detected. Skipping deactivation."
}

# --- Step 2: Run the Python script with the venv interpreter ---
# Define the path to the python executable within the project's virtual environment
$pythonExecutable = ".\venv\Scripts\python.exe"

if (-not (Test-Path $pythonExecutable)) {
    Write-Error "Could not find the Python interpreter in the virtual environment at: $pythonExecutable"
    Write-Error "Please ensure you have created the virtual environment by running 'python -m venv venv' and installed dependencies."
    exit 1
}

$scriptToRun = ".\process_dnd_session.py"

if (-not (Test-Path $scriptToRun)) {
    Write-Error "Could not find the main Python script at: $scriptToRun"
    exit 1
}

Write-Host "Running the D&D session processor..."
Write-Host "Command: $pythonExecutable $scriptToRun `"$absoluteSessionDir`""
Write-Host "--------------------------------------------------"

# Execute the python script directly with the venv interpreter, passing the session directory argument
& $pythonExecutable $scriptToRun "$absoluteSessionDir"

Write-Host "--------------------------------------------------"
Write-Host "Script execution finished." 