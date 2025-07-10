
import logging
import ast
from pathlib import Path

from agentQ.app.core.toolbox import Tool
from agentQ.app.core.file_system_tool import read_file, write_file, SANDBOX_DIR

logger = logging.getLogger(__name__)

class RenameVariableTransformer(ast.NodeTransformer):
    """
    An AST NodeTransformer that safely renames a variable within a specific function.
    """
    def __init__(self, target_function_name: str, old_name: str, new_name: str):
        self.target_function_name = target_function_name
        self.old_name = old_name
        self.new_name = new_name
        self._in_target_function = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == self.target_function_name:
            self._in_target_function = True
            self.generic_visit(node)
            self._in_target_function = False
        return node

    def visit_Name(self, node: ast.Name):
        if self._in_target_function and node.id == self.old_name:
            return ast.Name(id=self.new_name, ctx=node.ctx)
        return node
    
    def visit_arg(self, node: ast.arg):
        if self._in_target_function and node.arg == self.old_name:
            return ast.arg(arg=self.new_name, annotation=node.annotation)
        return node

def rename_variable_in_function(
    file_path: str, 
    function_name: str, 
    old_variable_name: str, 
    new_variable_name: str,
    config: dict = {}
) -> str:
    """
    Safely renames a variable within a specific function in a given Python file using an AST.
    This will read the file, perform the transformation, and write the file back.

    Args:
        file_path (str): The path to the Python file in the workspace.
        function_name (str): The name of the function in which to rename the variable.
        old_variable_name (str): The current name of the variable.
        new_variable_name (str): The new name for the variable.

    Returns:
        A confirmation or error message.
    """
    logger.info(f"Attempting to rename variable '{old_variable_name}' to '{new_variable_name}' in function '{function_name}' of file '{file_path}'.")
    
    try:
        # Read the file content using the existing file system tool
        source_code = read_file(file_path)
        if source_code.startswith("Error:"):
            return f"Error: Could not read file for AST transformation. {source_code}"

        # Parse the source code into an AST
        tree = ast.parse(source_code)

        # Apply the transformation
        transformer = RenameVariableTransformer(function_name, old_variable_name, new_variable_name)
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

        # Unparse the modified AST back into code
        new_code = ast.unparse(new_tree)

        # Write the changes back to the file
        write_result = write_file(file_path, new_code)
        if write_result.startswith("Error:"):
            return f"Error: Could not write file after AST transformation. {write_result}"

        return f"Successfully renamed variable '{old_variable_name}' to '{new_variable_name}' in file '{file_path}'."

    except SyntaxError as e:
        return f"Error: Could not parse Python file '{file_path}'. Invalid syntax: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during AST transformation: {e}", exc_info=True)
        return f"Error: An unexpected error occurred: {e}"


# --- Tool Registration Object ---
ast_rename_variable_tool = Tool(
    name="rename_variable_in_function_ast",
    description="Safely renames a variable (including parameters) within a specific function of a Python file using an Abstract Syntax Tree (AST). Use this for safe, targeted refactoring.",
    func=rename_variable_in_function
) 