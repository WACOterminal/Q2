# agentQ/app/core/sast_tool.py
import subprocess
import json
import structlog
from pathlib import Path
from agentQ.app.core.toolbox import Tool

logger = structlog.get_logger(__name__)

def sast_scan_directory(directory_path: str, config: dict = None) -> str:
    """
    Performs a Static Application Security Test (SAST) scan on a directory using the 'bandit' tool.

    Args:
        directory_path (str): The absolute path to the directory to scan.

    Returns:
        str: A JSON string of the scan results, or an error message.
    """
    target_path = Path(directory_path)
    if not target_path.is_dir():
        return f"Error: The provided path '{directory_path}' is not a valid directory."

    logger.info("Running SAST scan on directory", path=str(target_path))

    # Command to run bandit and get JSON output
    # -r: recursive, -f: format (json)
    command = [
        "bandit",
        "-r",
        str(target_path),
        "-f",
        "json",
        "-c",
        "/app/bandit.yaml" # Assuming a custom config file is mounted in the container
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False # Do not throw exception on non-zero exit code
        )

        if result.returncode not in [0, 1]: # Bandit exits 1 if issues are found, 0 if not.
            logger.error("Bandit scan failed", stderr=result.stderr)
            return f"Error: Bandit scan failed with exit code {result.returncode}. Stderr: {result.stderr}"

        # Bandit outputs JSON with results and potential errors
        return result.stdout

    except FileNotFoundError:
        logger.error("Bandit command not found. Is it installed in the environment?")
        return "Error: 'bandit' command not found. Please ensure it is installed."
    except Exception as e:
        logger.error("An unexpected error occurred during SAST scan", exc_info=True)
        return f"Error: An unexpected error occurred: {e}"

sast_tool = Tool(
    name="sast_scan_directory",
    description="Performs a SAST scan on a given directory using bandit and returns a JSON report of vulnerabilities.",
    func=sast_scan_directory,
) 