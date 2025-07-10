# agentQ/app/core/sast_tool.py
import json
import subprocess
from typing import List, Dict, Any

from agentQ.app.core.toolbox import Tool

def static_analysis_security_tool_func(path: str) -> Dict[str, Any]:
    """
    Runs a static analysis security scan on the specified file or directory path using Semgrep.
    The tool captures the findings and returns them in a structured JSON format.
    Only use this tool on a specific directory, not the whole codebase.
    For example, `app/` or `tests/`.

    Args:
        path (str): The path to the file or directory to scan.

    Returns:
        Dict[str, Any]: A dictionary containing the scan results.
    """
    try:
        # The `--json` flag tells semgrep to output results in JSON format.
        # The `--config "auto"` flag tells semgrep to automatically select the best rules to run.
        command = ["semgrep", "scan", "--json", "--config", "auto", path]
        
        # We use subprocess.run to execute the command.
        # `capture_output=True` captures stdout and stderr.
        # `text=True` decodes stdout/stderr as text.
        # `check=True` raises a CalledProcessError if the command returns a non-zero exit code.
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        
        # The JSON output from semgrep is a string, so we parse it into a Python dictionary.
        return json.loads(result.stdout)
    
    except FileNotFoundError:
        return {"error": "Semgrep not found. Make sure it is installed and in the system's PATH."}
    except subprocess.CalledProcessError as e:
        # If semgrep returns a non-zero exit code, it might indicate an error during the scan.
        return {
            "error": "Semgrep scan failed.",
            "stdout": e.stdout,
            "stderr": e.stderr
        }
    except json.JSONDecodeError:
        # This might happen if the output is not valid JSON for some reason.
        return {"error": "Failed to parse Semgrep JSON output."}
    except Exception as e:
        # Catch any other unexpected errors.
        return {"error": f"An unexpected error occurred: {str(e)}"}

# --- Tool Registration ---
static_analysis_security_tool = Tool(
    name="static_analysis_security_tool",
    description="Runs a static analysis security scan on the specified file or directory path using Semgrep. The tool captures the findings and returns them in a structured JSON format. Only use this tool on a specific directory, not the whole codebase. For example, `app/` or `tests/`.",
    func=static_analysis_security_tool_func
) 