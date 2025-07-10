
import logging
import os
from pathlib import Path
from typing import List, Dict
import subprocess
import asyncio

from agentQ.app.core.toolbox import Tool
from shared.q_agentsandbox_client.client import AgentSandboxClient

logger = logging.getLogger(__name__)

# --- Configuration ---
# SECURITY: Define a safe, sandboxed directory for all file operations.
# The agent will not be able to read or write files outside this directory.
SANDBOX_DIR = Path("./workspace/").resolve()

def _is_path_safe(path: Path) -> bool:
    """Checks if the resolved path is within the sandboxed directory."""
    resolved_path = path.resolve()
    return SANDBOX_DIR in resolved_path.parents or resolved_path == SANDBOX_DIR

def read_file(path: str, config: dict = {}) -> str:
    """
    Reads the content of a specific file within the agent's workspace.
    
    Args:
        path (str): The relative path to the file from the workspace root.
        
    Returns:
        The content of the file as a string, or an error message.
    """
    target_path = SANDBOX_DIR / path
    
    if not _is_path_safe(target_path):
        return f"Error: Path '{path}' is outside the allowed workspace."

    try:
        if not target_path.is_file():
            return f"Error: File not found at '{path}'."
        
        with open(target_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        logger.error(f"Error reading file '{path}': {e}", exc_info=True)
        return f"Error: An unexpected error occurred while reading the file: {e}"

def write_file(path: str, content: str, config: dict = {}) -> str:
    """
    Writes content to a specific file within the agent's workspace.
    This will create the file if it doesn't exist and overwrite it if it does.
    
    Args:
        path (str): The relative path to the file.
        content (str): The new content to write to the file.
        
    Returns:
        A success or error message.
    """
    target_path = SANDBOX_DIR / path

    if not _is_path_safe(target_path):
        return f"Error: Path '{path}' is outside the allowed workspace."

    try:
        # Create parent directories if they don't exist
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully wrote {len(content)} characters to '{path}'."
    except Exception as e:
        logger.error(f"Error writing to file '{path}': {e}", exc_info=True)
        return f"Error: An unexpected error occurred while writing the file: {e}"

def list_directory(path: str = ".", config: dict = {}) -> str:
    """
    Lists the contents of a directory within the agent's workspace.
    
    Args:
        path (str): The relative path to the directory. Defaults to the workspace root.
        
    Returns:
        A formatted string listing the directory contents, or an error message.
    """
    target_path = SANDBOX_DIR / path

    if not _is_path_safe(target_path):
        return f"Error: Path '{path}' is outside the allowed workspace."

    try:
        if not target_path.is_dir():
            return f"Error: Directory not found at '{path}'."
        
        contents = os.listdir(target_path)
        
        # Separate directories and files for cleaner output
        dirs = [d for d in contents if (target_path / d).is_dir()]
        files = [f for f in contents if (target_path / f).is_file()]
        
        output = f"Contents of '{path}':\n"
        output += "Directories:\n" + ("\n".join(f"- {d}/" for d in sorted(dirs)) if dirs else "  (none)")
        output += "\n\nFiles:\n" + ("\n".join(f"- {f}" for f in sorted(files)) if files else "  (none)")
        
        return output
    except Exception as e:
        logger.error(f"Error listing directory '{path}': {e}", exc_info=True)
        return f"Error: An unexpected error occurred while listing the directory: {e}"

def run_command(command: str, config: Dict = {}) -> str:
    """
    Executes a shell command within a secure, isolated sandbox environment.
    
    Args:
        command (str): The command to execute (e.g., "pytest tests/").
        
    Returns:
        A string containing the combined stdout and stderr of the command, plus the exit code.
    """
    logger.info(f"Executing sandboxed command: '{command}'")
    
    sandbox_service_url = config.get("agent_sandbox_url")
    if not sandbox_service_url:
        return "Error: agent_sandbox_url not found in tool configuration."

    client = AgentSandboxClient(base_url=sandbox_service_url)
    
    async def execute_in_sandbox():
        container_id = None
        try:
            container_id = await client.create_sandbox()
            if not container_id:
                return "Error: Failed to create sandbox environment."

            result = await client.execute_command(container_id, command)
            if not result:
                return "Error: Failed to execute command in sandbox."
            
            output = f"Exit Code: {result.get('exit_code')}\n"
            output += f"--- OUTPUT ---\n{result.get('output')}"
            return output
        finally:
            if container_id:
                await client.remove_sandbox(container_id)
            await client.close()

    try:
        return asyncio.run(execute_in_sandbox())
    except Exception as e:
        logger.error(f"Error executing command '{command}' via sandbox: {e}", exc_info=True)
        return f"Error: An unexpected error occurred while running the sandboxed command: {e}"


# --- Tool Registration Objects ---

read_file_tool = Tool(
    name="read_file",
    description="Reads the entire content of a file from the workspace. Use this to understand existing code or documents before making changes.",
    func=read_file
)

write_file_tool = Tool(
    name="write_file",
    description="Writes or overwrites a file in the workspace with new content. Use this to create new files or modify existing ones after reading them.",
    func=write_file
)

list_directory_tool = Tool(
    name="list_directory",
    description="Lists all files and subdirectories within a specified directory in the workspace. Use this to navigate the file system.",
    func=list_directory
)

run_command_tool = Tool(
    name="run_command",
    description="Executes a shell command (e.g., 'pytest', 'ls -l') inside the sandboxed workspace. Use this to run tests, build code, or perform other command-line operations.",
    func=run_command
) 