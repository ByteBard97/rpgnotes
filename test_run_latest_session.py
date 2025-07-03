"""
Test script for run_latest_session.ps1 PowerShell script.
"""

import os
import tempfile
import shutil
import subprocess
import time
from pathlib import Path


def test_latest_session_finder():
    """Test that the PowerShell script finds the newest directory"""
    
    # Create a temporary directory structure
    temp_dir = Path(tempfile.mkdtemp())
    recordings_dir = temp_dir / "recordings"
    recordings_dir.mkdir()
    
    # Create test session directories with different timestamps
    test_sessions = [
        "session_2024_01_01",
        "session_2024_01_15", 
        "session_2024_02_01"
    ]
    
    session_dirs = []
    for i, session_name in enumerate(test_sessions):
        session_path = recordings_dir / session_name
        session_path.mkdir()
        
        # Add a small delay to ensure different timestamps
        time.sleep(0.1)
        
        # Create a dummy file to make it look like a real session
        dummy_file = session_path / "dummy.txt"
        dummy_file.write_text(f"Session {session_name}")
        
        session_dirs.append(session_path)
    
    print(f"Created test directories in: {recordings_dir}")
    for session_dir in session_dirs:
        stat = session_dir.stat()
        print(f"  {session_dir.name}: {stat.st_mtime}")
    
    # The newest should be the last one created
    newest_expected = session_dirs[-1]
    print(f"Expected newest: {newest_expected.name}")
    
    # Test PowerShell command to find newest directory
    ps_command = f"""
    $recordingsDir = "{recordings_dir}"
    $sessionDirs = Get-ChildItem -Path $recordingsDir -Directory | Sort-Object LastWriteTime -Descending
    if ($sessionDirs.Count -gt 0) {{
        $latestSession = $sessionDirs[0]
        Write-Output $latestSession.Name
    }}
    """
    
    try:
        result = subprocess.run(
            ["powershell", "-Command", ps_command],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            newest_found = result.stdout.strip()
            print(f"PowerShell found newest: {newest_found}")
            
            if newest_found == newest_expected.name:
                print("✓ Test PASSED: PowerShell correctly identified newest directory")
            else:
                print("✗ Test FAILED: PowerShell found wrong directory")
                print(f"  Expected: {newest_expected.name}")
                print(f"  Found: {newest_found}")
        else:
            print(f"✗ Test FAILED: PowerShell command failed")
            print(f"  Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("✗ Test FAILED: PowerShell command timed out")
    except FileNotFoundError:
        print("✗ Test SKIPPED: PowerShell not available (not on Windows or not in PATH)")
    
    # Clean up
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary directory: {temp_dir}")


def test_script_validation():
    """Test that the PowerShell script file exists and has proper structure"""
    
    script_path = Path("run_latest_session.ps1")
    
    if not script_path.exists():
        print("✗ Test FAILED: run_latest_session.ps1 does not exist")
        return
    
    # Read the script content
    script_content = script_path.read_text()
    
    # Check for key components
    required_elements = [
        "recordings",
        "Get-ChildItem",
        "Sort-Object LastWriteTime -Descending",
        "run_session_processing.ps1",
        "Read-Host"
    ]
    
    missing_elements = []
    for element in required_elements:
        if element not in script_content:
            missing_elements.append(element)
    
    if missing_elements:
        print(f"✗ Test FAILED: Script missing required elements: {missing_elements}")
    else:
        print("✓ Test PASSED: Script contains all required elements")
    
    # Check script length is reasonable
    line_count = len(script_content.split('\n'))
    if line_count < 30:
        print("✗ Test FAILED: Script seems too short")
    elif line_count > 100:
        print("✗ Test FAILED: Script is getting too long")
    else:
        print(f"✓ Test PASSED: Script length is reasonable ({line_count} lines)")


if __name__ == "__main__":
    print("Testing run_latest_session.ps1 PowerShell script...")
    print("=" * 50)
    
    test_script_validation()
    print()
    test_latest_session_finder()
    
    print("=" * 50)
    print("Testing complete.") 