
import logging
import ast
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re
import json

from agentQ.app.core.toolbox import Tool
from shared.q_pulse_client.client import QuantumPulseClient

logger = logging.getLogger(__name__)

class CodeAnalyzer:
    """Analyzes Python code to extract function information for test generation"""
    
    def __init__(self):
        pass  # Remove the unused pulse_client initialization
    
    def extract_function_info(self, file_path: str, function_name: str) -> Optional[Dict[str, Any]]:
        """Extract detailed information about a function from a Python file"""
        try:
            with open(file_path, 'r') as f:
                source_code = f.read()
            
            # Parse the AST
            tree = ast.parse(source_code)
            
            # Find the function
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    return self._analyze_function(node, source_code, file_path)
            
            logger.error(f"Function '{function_name}' not found in {file_path}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return None
    
    def _analyze_function(self, func_node: ast.FunctionDef, source_code: str, file_path: str) -> Dict[str, Any]:
        """Analyze a function AST node to extract detailed information"""
        # Extract function signature
        args = []
        defaults = []
        
        for arg in func_node.args.args:
            args.append(arg.arg)
        
        for default in func_node.args.defaults:
            if isinstance(default, ast.Constant):
                defaults.append(default.value)
            else:
                defaults.append(ast.unparse(default))
        
        # Extract return type annotation if present
        return_type = None
        if func_node.returns:
            return_type = ast.unparse(func_node.returns)
        
        # Extract docstring
        docstring = ast.get_docstring(func_node)
        
        # Extract function body
        body_lines = source_code.split('\n')[func_node.lineno - 1:func_node.end_lineno]
        function_body = '\n'.join(body_lines)
        
        # Analyze function complexity
        complexity_info = self._analyze_complexity(func_node)
        
        # Extract dependencies (imports used, other functions called)
        dependencies = self._extract_dependencies(func_node)
        
        return {
            'name': func_node.name,
            'args': args,
            'defaults': defaults,
            'return_type': return_type,
            'docstring': docstring,
            'body': function_body,
            'complexity': complexity_info,
            'dependencies': dependencies,
            'decorators': [ast.unparse(d) for d in func_node.decorator_list],
            'is_async': isinstance(func_node, ast.AsyncFunctionDef),
            'file_path': file_path  # Add file_path to the info
        }
    
    def _analyze_complexity(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze function complexity metrics"""
        complexity = {
            'lines': func_node.end_lineno - func_node.lineno + 1 if func_node.end_lineno else 1,
            'branches': 0,
            'loops': 0,
            'try_blocks': 0,
            'returns': 0
        }
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.IfExp)):
                complexity['branches'] += 1
            elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                complexity['loops'] += 1
            elif isinstance(node, ast.Try):
                complexity['try_blocks'] += 1
            elif isinstance(node, ast.Return):
                complexity['returns'] += 1
        
        return complexity
    
    def _extract_dependencies(self, func_node: ast.FunctionDef) -> Dict[str, List[str]]:
        """Extract function dependencies"""
        dependencies = {
            'function_calls': [],
            'attributes': [],
            'imports': []
        }
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    dependencies['function_calls'].append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    dependencies['function_calls'].append(ast.unparse(node.func))
            elif isinstance(node, ast.Attribute):
                dependencies['attributes'].append(ast.unparse(node))
        
        # Remove duplicates
        dependencies['function_calls'] = list(set(dependencies['function_calls']))
        dependencies['attributes'] = list(set(dependencies['attributes']))
        
        return dependencies


class TestGenerator:
    """Generates comprehensive tests using AI"""
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        # Initialize with a default base URL - this should be configured properly
        self.pulse_client = QuantumPulseClient(base_url="http://quantumpulse:8000")
    
    async def generate_test(self, file_path: str, function_name: str, config: Optional[Dict[str, Any]] = None) -> str:
        """Generate comprehensive test for a function using AI"""
        # Analyze the function
        func_info = self.analyzer.extract_function_info(file_path, function_name)
        if not func_info:
            return f"# Error: Could not analyze function '{function_name}' in {file_path}"
        
        # Add file_path to func_info if not present
        if 'file_path' not in func_info:
            func_info['file_path'] = file_path
        
        # Prepare the prompt for the AI
        prompt = self._create_test_generation_prompt(func_info, config or {})
        
        try:
            # Create the chat request
            from shared.q_pulse_client.models import QPChatRequest, QPChatMessage
            
            request = QPChatRequest(
                model=config.get('model', 'gpt-4') if config else 'gpt-4',
                messages=[
                    QPChatMessage(
                        role="system",
                        content="You are an expert Python test engineer. Generate comprehensive, well-structured tests using pytest conventions."
                    ),
                    QPChatMessage(
                        role="user",
                        content=prompt
                    )
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Call the AI model to generate tests
            response = await self.pulse_client.get_chat_completion(request)
            
            # Extract the generated content
            if response.choices and len(response.choices) > 0:
                test_code = self._extract_test_code(response.choices[0].message.content)
            else:
                raise ValueError("No response from AI model")
            
            # Validate the generated test
            if self._validate_test_code(test_code, func_info):
                return test_code
            else:
                logger.warning("Generated test failed validation, attempting to fix...")
                return self._fix_test_code(test_code, func_info)
                
        except Exception as e:
            logger.error(f"Failed to generate test using AI: {e}")
            # Fallback to template-based generation
            return self._generate_template_test(func_info)
    
    def _create_test_generation_prompt(self, func_info: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Create a detailed prompt for test generation"""
        test_style = config.get('test_style', 'comprehensive')
        include_edge_cases = config.get('include_edge_cases', True)
        include_mocks = config.get('include_mocks', True)
        
        prompt = f"""Generate comprehensive pytest tests for the following Python function:

Function Name: {func_info['name']}
Parameters: {', '.join(func_info['args'])}
Return Type: {func_info.get('return_type', 'Any')}
Is Async: {func_info['is_async']}

Function Implementation:
```python
{func_info['body']}
```

Function Complexity:
- Lines of code: {func_info['complexity']['lines']}
- Branches: {func_info['complexity']['branches']}
- Loops: {func_info['complexity']['loops']}
- Try blocks: {func_info['complexity']['try_blocks']}

Dependencies:
- Function calls: {', '.join(func_info['dependencies']['function_calls'][:5])}

Requirements:
1. Use pytest framework with proper fixtures
2. Include docstrings explaining what each test does
3. Test happy path scenarios
4. {"Include edge cases and error conditions" if include_edge_cases else "Focus on main functionality"}
5. {"Mock external dependencies" if include_mocks else "Assume dependencies work correctly"}
6. Use descriptive test names following test_<function>_<scenario> pattern
7. Include parametrized tests where appropriate
8. Add assertions with meaningful error messages

Generate the test code:"""
        
        if func_info.get('docstring'):
            prompt = f"Function Docstring:\n{func_info['docstring']}\n\n" + prompt
        
        return prompt
    
    def _extract_test_code(self, ai_response: str) -> str:
        """Extract Python code from AI response"""
        # Look for code blocks
        code_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(code_pattern, ai_response, re.DOTALL)
        
        if matches:
            return matches[0]
        
        # If no code blocks, try to extract code-like content
        lines = ai_response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith(('import ', 'from ', 'def ', '@', 'class ')):
                in_code = True
            
            if in_code:
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else ai_response
    
    def _validate_test_code(self, test_code: str, func_info: Dict[str, Any]) -> bool:
        """Validate that the generated test code is syntactically correct"""
        try:
            # Check syntax
            ast.parse(test_code)
            
            # Check that it contains test functions
            tree = ast.parse(test_code)
            test_functions = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')
            ]
            
            if not test_functions:
                logger.warning("No test functions found in generated code")
                return False
            
            # Check that tests reference the target function
            function_name = func_info['name']
            contains_function_call = any(
                function_name in ast.unparse(node)
                for node in ast.walk(tree)
            )
            
            if not contains_function_call:
                logger.warning(f"Generated tests don't seem to test {function_name}")
                return False
            
            return True
            
        except SyntaxError as e:
            logger.error(f"Generated test has syntax error: {e}")
            return False
    
    def _fix_test_code(self, test_code: str, func_info: Dict[str, Any]) -> str:
        """Attempt to fix common issues in generated test code"""
        # Add missing imports
        if 'import pytest' not in test_code:
            test_code = 'import pytest\n' + test_code
        
        # Add import for the function being tested
        if f"from {func_info['name']}" not in test_code and f"import {func_info['name']}" not in test_code:
            # Assume the function is in the module being tested
            module_name = Path(func_info.get('file_path', 'module')).stem
            test_code = f"from {module_name} import {func_info['name']}\n" + test_code
        
        return test_code
    
    def _generate_template_test(self, func_info: Dict[str, Any]) -> str:
        """Generate a template-based test as fallback"""
        function_name = func_info['name']
        args = func_info['args']
        is_async = func_info['is_async']
        
        # Generate test parameters
        test_params = []
        for i, arg in enumerate(args):
            if arg == 'self':
                continue
            # Generate appropriate test values based on parameter names
            if 'id' in arg.lower():
                test_params.append('"test_id"')
            elif 'name' in arg.lower():
                test_params.append('"test_name"')
            elif 'number' in arg.lower() or 'count' in arg.lower():
                test_params.append('42')
            elif 'flag' in arg.lower() or 'is_' in arg:
                test_params.append('True')
            elif 'data' in arg.lower():
                test_params.append('{"key": "value"}')
            elif 'list' in arg.lower() or 'items' in arg.lower():
                test_params.append('[1, 2, 3]')
            else:
                test_params.append(f'"test_{arg}"')
        
        # Generate the test
        test_code = f"""import pytest
{"import asyncio" if is_async else ""}
from unittest.mock import Mock, patch

# Import the function to test
# from your_module import {function_name}


{"@pytest.mark.asyncio" if is_async else ""}
def {"async " if is_async else ""}test_{function_name}_basic():
    \"\"\"Test basic functionality of {function_name}\"\"\"
    # Arrange
    {chr(10).join(f"    {arg} = {val}" for arg, val in zip(args[1:], test_params)) if args[1:] else "    # No parameters"}
    
    # Act
    result = {"await " if is_async else ""}{function_name}({', '.join(args[1:])})
    
    # Assert
    assert result is not None
    # TODO: Add specific assertions based on expected behavior


{"@pytest.mark.asyncio" if is_async else ""}
def {"async " if is_async else ""}test_{function_name}_with_invalid_input():
    \"\"\"Test {function_name} with invalid input\"\"\"
    # Test with None
    with pytest.raises((TypeError, ValueError, AttributeError)):
        {"await " if is_async else ""}{function_name}({', '.join(['None'] * len(args[1:]))})


@pytest.mark.parametrize("test_input,expected", [
    ({', '.join(test_params)}, "expected_result_1"),
    # Add more test cases here
])
{"@pytest.mark.asyncio" if is_async else ""}
def {"async " if is_async else ""}test_{function_name}_parametrized(test_input, expected):
    \"\"\"Parametrized test for {function_name}\"\"\"
    result = {"await " if is_async else ""}{function_name}(test_input)
    assert result == expected
"""
        
        # Add complexity-specific tests
        if func_info['complexity']['branches'] > 2:
            test_code += f"""

{"@pytest.mark.asyncio" if is_async else ""}
def {"async " if is_async else ""}test_{function_name}_all_branches():
    \"\"\"Test all conditional branches in {function_name}\"\"\"
    # TODO: Add tests for each conditional branch
    pass
"""
        
        if func_info['complexity']['try_blocks'] > 0:
            test_code += f"""

{"@pytest.mark.asyncio" if is_async else ""}
def {"async " if is_async else ""}test_{function_name}_exception_handling():
    \"\"\"Test exception handling in {function_name}\"\"\"
    # TODO: Test exception scenarios
    with pytest.raises(Exception):
        {"await " if is_async else ""}{function_name}({', '.join(test_params)})
"""
        
        return test_code


# Synchronous wrapper for the async test generator
def generate_test(file_path: str, function_name: str, config: dict = {}) -> str:
    """
    Generates a new test for a specific function in a file using an AI model.

    Args:
        file_path (str): The relative path to the Python file to generate a test for.
        function_name (str): The name of the function to generate a test for.
        config (dict): Configuration options for test generation

    Returns:
        A string containing the generated test code, or an error message.
    """
    logger.info(f"Generating AI-powered test for function '{function_name}' in file: {file_path}")
    
    generator = TestGenerator()
    
    # Run the async function synchronously
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        test_code = loop.run_until_complete(generator.generate_test(file_path, function_name, config))
        return test_code
    except Exception as e:
        logger.error(f"Failed to generate test: {e}")
        # Fallback to synchronous template generation
        analyzer = CodeAnalyzer()
        func_info = analyzer.extract_function_info(file_path, function_name)
        if func_info:
            return generator._generate_template_test(func_info)
        else:
            return f"# Error: Failed to generate test for {function_name}"

# --- Tool Registration Object ---
test_generator_tool = Tool(
    name="generate_new_test",
    description="Generates comprehensive tests for a specific function using AI analysis and code generation.",
    func=generate_test
) 