# How to Run the Session Transcription Script

This guide explains how to run the D&D session transcription script.

## Prerequisites

1.  **Python Environment:** Ensure you have the correct Python version (3.10+) installed and have created the project's virtual environment (`venv`).
2.  **Session Directory Setup:** Before running, make sure the target session directory (e.g., `recordings/your_session_date`) is correctly configured with your audio files and session-specific `json` configuration files. See the `README.md` for more details on setup.

## How to Run

A PowerShell script is provided to simplify the execution process. It automatically handles activating the correct virtual environment.

1.  **Open Terminal:** Launch your terminal application (e.g., PowerShell, Command Prompt, Windows Terminal).

2.  **Navigate to Project Directory:** Change your current directory to the root of the `rpgnotes` project.
    ```powershell
    cd C:\path\to\your\projects\rpgnotes 
    # Example: cd C:\Users\geoff\Desktop\projects\rpgnotes
    ```

3.  **Run the Helper Script:** Execute the `run_session_processing.ps1` script, providing the relative path to the target session directory as the argument.
    ```powershell
    .\run_session_processing.ps1 path/to/your/session_directory
    ```

    **Example:**
    To process the session located in `recordings/friday_may_23_2025`, you would run:
    ```powershell
    .\run_session_processing.ps1 recordings/friday_may_23_2025
    ```

> **Note: Execution Policy Error**
> If you get an error mentioning "running scripts is disabled on this system" the first time you run the script, you need to adjust your PowerShell Execution Policy. Run the following command **in an administrator PowerShell** and then try again:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
> ```
> Press `Y` and Enter to confirm. You should only need to do this once.

---
## Process Latest Session (Convenience Script)

For quick processing of the most recent session, use the convenience script:

```powershell
.\run_latest_session.ps1
```

This script will:
- Automatically find the newest folder in the `recordings` directory
- Show you which session it found and ask for confirmation
- Run the session processing script on that folder

This is perfect for when you just recorded a session and want to quickly process it without having to remember the exact folder name.

---
## Manual Execution (Advanced)
If you prefer to run the script manually, follow these steps:
1. Deactivate any active Conda environment (`conda deactivate`).
2. Activate the project's virtual environment (`.\venv\Scripts\Activate.ps1`).
3. Run the python script directly (`python process_dnd_session.py path/to/your/session_directory`).

## Notes

*   The script will load configuration first from the global `config.json` and then override settings based on the `