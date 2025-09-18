import os
import subprocess
from unittest.mock import MagicMock, mock_open, patch

import pytest

from wave_surrogate.flac.api_flac import FLACClient


@pytest.fixture
def mock_executable_path(tmp_path):
    """Creates a dummy executable file for testing."""
    exe_path = tmp_path / "flac2d9_console.exe"
    exe_path.touch()
    return str(exe_path)


def test_flac_client_initialization(mock_executable_path):
    """Tests successful initialization of FLACClient."""
    client = FLACClient(executable_path=mock_executable_path)
    assert client.executable_path == mock_executable_path


def test_flac_client_initialization_file_not_found():
    """Tests that FLACClient raises FileNotFoundError for a non-existent executable."""
    with pytest.raises(FileNotFoundError):
        FLACClient(executable_path="/non/existent/path/flac.exe")


@patch("subprocess.check_output")
def test_get_windows_path(mock_check_output, mock_executable_path):
    """Tests the _get_windows_path method."""
    client = FLACClient(executable_path=mock_executable_path)
    # Simulate the output of `wslpath -w` which includes a newline
    mock_check_output.return_value = "C:\\path\\to\\file\n"

    wsl_path = "/mnt/c/path/to/file"
    # Get the absolute path as the method does
    abs_wsl_path = os.path.abspath(wsl_path)
    win_path = client._get_windows_path(wsl_path)

    mock_check_output.assert_called_once_with(
        ["wslpath", "-w", abs_wsl_path], text=True
    )
    assert win_path == "C:\\path\\to\\file"


@patch("subprocess.check_output", side_effect=FileNotFoundError)
def test_get_windows_path_wslpath_not_found(
    mock_check_output, mock_executable_path, caplog
):
    """Tests error handling when wslpath is not found."""
    client = FLACClient(executable_path=mock_executable_path)
    win_path = client._get_windows_path("/mnt/c/some/path")

    assert win_path is None
    assert "Error: Could not convert path using 'wslpath'" in caplog.text


@patch("subprocess.Popen")
@patch("wave_surrogate.flac.api_flac.FLACClient._get_windows_path")
@patch("builtins.open", new_callable=mock_open)
def test_run_python_script(
    mock_file, mock_get_windows_path, mock_popen, mock_executable_path, tmp_path
):
    """Tests the run_python_script method with mocks."""
    # 1. Setup Mocks
    # Mock the path conversion to return predictable Windows-style paths
    mock_get_windows_path.side_effect = [
        "C:\\scripts\\my_script.py",
        "C:\\projects\\run_script.dat",
    ]

    # Mock the Popen process to simulate FLAC execution
    mock_process = MagicMock()
    mock_process.stdout.readline.return_value = ""  # End of output
    mock_popen.return_value = mock_process

    # 2. Setup Test Environment
    # Create a dummy Python script to be "run"
    script_path = tmp_path / "my_script.py"
    script_path.touch()

    # 3. Initialize the client and run the method
    client = FLACClient(executable_path=mock_executable_path)
    client.run_python_script(python_script_path=str(script_path))

    # 4. Assertions
    # Verify that the .dat file was created and written to correctly
    dat_file_path = os.path.join("./outputs/flac/projects/", "run_script.dat")
    mock_file.assert_called_once_with(dat_file_path, "w")
    handle = mock_file()
    handle.write.assert_any_call("model new\n")
    handle.write.assert_any_call(
        'python exec(open(r"C:\\scripts\\my_script.py").read())\n'
    )
    handle.write.assert_any_call("exit\n")

    # Verify that the FLAC executable was called with the correct command
    expected_command = [
        mock_executable_path,
        "call",
        "C:\\projects\\run_script.dat",
    ]
    mock_popen.assert_called_once_with(
        expected_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    # Verify that the process output was read and the process was waited on
    mock_process.stdout.readline.assert_called_once()
    mock_process.wait.assert_called_once()


def test_run_python_script_file_not_found(mock_executable_path, caplog):
    """Tests that an error is logged if the Python script does not exist."""
    client = FLACClient(executable_path=mock_executable_path)
    client.run_python_script(python_script_path="/non/existent/script.py")

    assert (
        "❌ Error: The specified script '/non/existent/script.py' does not exist."
        in caplog.text
    )
