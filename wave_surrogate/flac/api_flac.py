import os
import subprocess
import textwrap

from wave_surrogate.logging_setup import setup_logging

logger = setup_logging()


class FLACClient:
    """
    A client to interact with the FLAC2D console application, specifically
    for running Python scripts within the FLAC environment from a WSL instance.
    """

    def __init__(self, executable_path):
        """
        Initializes the FLAC client.

        Args:
            executable_path (str): The WSL path to the FLAC executable
                                   (e.g., '/mnt/c/.../flac2d9_console.exe').
        """
        if not os.path.exists(executable_path):
            raise FileNotFoundError(
                f"The FLAC executable was not found at '{executable_path}'"
            )
        self.executable_path = executable_path
        logger.info("✅ FLACClient initialized.")

    def _get_windows_path(self, wsl_path):
        """Converts a WSL path to its Windows equivalent using wslpath."""
        try:
            abs_path = os.path.abspath(wsl_path)
            return subprocess.check_output(
                ["wslpath", "-w", abs_path], text=True
            ).strip()
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.error(f"Error: Could not convert path using 'wslpath'. {e}")
            return None

    def run_python_script(
        self, python_script_path, project_dir="./outputs/flac/projects/"
    ):
        """
        Executes a given Python script inside the FLAC environment.

        This method creates a temporary .dat file to call the Python script,
        handles path conversions, and runs the FLAC process.

        Args:
            python_script_path (str): The WSL path to the Python script to execute.
            project_dir (str, optional): The directory to store the temporary
                                         .dat file. Defaults to './outputs/flac/projects/'.
        """
        # 1. Validate the input Python script path
        if not os.path.exists(python_script_path):
            logger.error(
                f"❌ Error: The specified script '{python_script_path}' does not exist."
            )
            return

        os.makedirs(project_dir, exist_ok=True)

        # 2. Convert the Python script's path to a Windows path for FLAC
        py_win_path = self._get_windows_path(python_script_path)
        if not py_win_path:
            return

        # 3. Create a temporary .dat file to launch the Python script
        dat_file_path = os.path.join(project_dir, "run_script.dat")
        with open(dat_file_path, "w") as f:
            f.write("model new\n")
            f.write(f'python exec(open(r"{py_win_path}").read())\n')
            f.write("exit\n")
        logger.info(f"Generated temporary FLAC data file: '{dat_file_path}'")

        # 4. Convert the .dat file's path for the command line
        dat_win_path = self._get_windows_path(dat_file_path)
        if not dat_win_path:
            return

        # 5. Execute the command
        logger.info(
            f"\n🚀 Launching FLAC2D to execute '{os.path.basename(python_script_path)}'..."
        )
        command = [self.executable_path, "call", dat_win_path]
        return self._execute_command(command)

    def _execute_command(self, command):
        """
        A helper method to run a subprocess, stream its output, and return it.
        """
        output_lines = []
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    line = line.strip()
                    print(line)
                    output_lines.append(line)

            process.wait()
            logger.info("\n🎉 FLAC run finished!")
            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return None


# --- DEMONSTRATION OF HOW TO USE THE CLASS ---
if __name__ == "__main__":
    # --- CRUCIAL CONFIGURATION ---
    FLAC2D_EXECUTABLE = "/mnt/c/Program Files/Itasca/Itasca Software Subscription/exe64/flac2d9_console.exe"

    # 1. Create a sample Python script to simulate an "existing" file
    script_dir = "./flac_scripts"
    os.makedirs(script_dir, exist_ok=True)
    my_script_path = os.path.join(script_dir, "my_model_setup.py")

    script_content = textwrap.dedent("""
    import itasca as it
    print("--- my_model_setup.py executing inside FLAC2D ---")
    it.command("model new")
    it.command("model large-strain off")
    it.command("zone create quad size 10 10")
    it.command("zone cmodel assign elastic")
    it.command("zone property density 2950 young 12e9 poisson 0.25")
    it.command("cycle 1")
    print(it.zone.count()) # It needs to print 100
    print("Model created and solved successfully.")
    print("--- my_model_setup.py has finished ---")
    """)

    with open(my_script_path, "w") as f:
        f.write(script_content)

    logger.info(f"Created a sample script to run: '{my_script_path}'\n")

    # 2. Use the FLACClient to run the script
    try:
        # Initialize the client
        client = FLACClient(executable_path=FLAC2D_EXECUTABLE)

        # Run the existing Python script and capture the output
        output = client.run_python_script(python_script_path=my_script_path)

        # Assert that the output contains the expected zone count
        if output:
            assert "100" in output
            logger.info("✅ Assertion successful: '100' found in the output.")

    except FileNotFoundError as e:
        logger.error(f"❌ Initialization failed: {e}")
        logger.info("Please update the FLAC2D_EXECUTABLE variable in the script.")
