
import logging
import subprocess
import json
from pathlib import Path

from agentQ.app.core.toolbox import Tool
from agentQ.app.core.file_system_tool import SANDBOX_DIR

logger = logging.getLogger(__name__)

def run_linter(file_path: str, config: dict = {}) -> str:
    """
    Runs the flake8 linter on a specific Python file within the workspace and returns the results as JSON.

    Args:
        file_path (str): The relative path to the Python file to lint.

    Returns:
        A JSON string containing the linting results, or an error message.
    """
    target_path = SANDBOX_DIR / file_path
    logger.info(f"Running flake8 linter on file: {target_path}")

    if not target_path.is_file():
        return f"Error: File not found at '{file_path}'."

    try:
        # We use --output-format=json to get structured data from flake8
        command = [
            "flake8", 
            str(target_path),
            "--output-format=json"
        ]
        
        # We can use subprocess.run here as flake8 is not interactive and will terminate.
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False 
        )

        if result.returncode != 0 and result.stdout:
            # Flake8 with json format outputs errors to stdout
            try:
                linting_errors = json.loads(result.stdout)
                if not linting_errors:
                     return json.dumps({"status": "success", "message": "No linting issues found."})
                return json.dumps({"status": "issues_found", "errors": linting_errors})
            except json.JSONDecodeError:
                return f"Error: Failed to parse flake8 JSON output. Raw output: {result.stdout}"
        elif result.returncode == 0:
            return json.dumps({"status": "success", "message": "No linting issues found."})
        else:
            return f"Error: Linter failed to run. Stderr: {result.stderr}"

    except FileNotFoundError:
        return "Error: 'flake8' command not found. Please ensure it is installed in the agent's environment."
    except Exception as e:
        logger.error(f"An unexpected error occurred while running the linter: {e}", exc_info=True)
        return f"Error: An unexpected error occurred: {e}"

# --- Tool Registration Object ---
linter_tool = Tool(
    name="run_linter_on_file",
    description="Runs a flake8 linter on a Python file to check for code quality issues, style violations, and syntax errors. Returns a list of issues found.",
    func=run_linter
) 