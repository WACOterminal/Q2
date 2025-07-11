# managerQ/app/core/conditional_evaluator.py
"""
Enhanced conditional logic evaluator for workflow engine
Provides sophisticated condition evaluation with operators, functions, and error handling
"""

import re
import json
import structlog
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from jinja2 import Environment, BaseLoader, select_autoescape, TemplateSyntaxError
from jinja2.sandbox import SandboxedEnvironment

logger = structlog.get_logger(__name__)

class ConditionalEvaluator:
    """Enhanced conditional logic evaluator for workflows"""
    
    def __init__(self):
        # Use sandboxed environment for security
        self.jinja_env = SandboxedEnvironment(
            loader=BaseLoader(),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Register custom functions
        self._register_custom_functions()
        
        # Condition operators
        self.operators = {
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b,
            '<=': lambda a, b: a <= b,
            '>': lambda a, b: a > b,
            '>=': lambda a, b: a >= b,
            'in': lambda a, b: a in b,
            'not_in': lambda a, b: a not in b,
            'contains': lambda a, b: b in a,
            'starts_with': lambda a, b: str(a).startswith(str(b)),
            'ends_with': lambda a, b: str(a).endswith(str(b)),
            'matches': lambda a, b: bool(re.match(str(b), str(a))),
            'is_empty': lambda a, b: not a,
            'is_not_empty': lambda a, b: bool(a),
            'is_null': lambda a, b: a is None,
            'is_not_null': lambda a, b: a is not None
        }
    
    def _register_custom_functions(self):
        """Register custom functions for conditional evaluation"""
        
        def now():
            """Get current timestamp"""
            return datetime.now()
        
        def today():
            """Get today's date"""
            return datetime.now().date()
        
        def days_ago(days: int):
            """Get date N days ago"""
            return datetime.now() - timedelta(days=days)
        
        def hours_ago(hours: int):
            """Get time N hours ago"""
            return datetime.now() - timedelta(hours=hours)
        
        def parse_date(date_str: str):
            """Parse date string"""
            try:
                return datetime.fromisoformat(date_str)
            except ValueError:
                return None
        
        def extract_json(text: str, key: str):
            """Extract value from JSON string"""
            try:
                data = json.loads(text) if isinstance(text, str) else text
                return data.get(key) if isinstance(data, dict) else None
            except (json.JSONDecodeError, TypeError):
                return None
        
        def count_items(items):
            """Count items in list or dict"""
            try:
                return len(items) if items else 0
            except TypeError:
                return 0
        
        def sum_values(items, key: Optional[str] = None):
            """Sum numeric values"""
            try:
                if key and isinstance(items, list):
                    return sum(item.get(key, 0) for item in items if isinstance(item, dict))
                elif isinstance(items, list):
                    return sum(x for x in items if isinstance(x, (int, float)))
                else:
                    return 0
            except (TypeError, AttributeError):
                return 0
        
        def avg_values(items, key: Optional[str] = None):
            """Average numeric values"""
            try:
                total = sum_values(items, key)
                count = count_items(items)
                return total / count if count > 0 else 0
            except (TypeError, ZeroDivisionError):
                return 0
        
        def filter_by(items, condition_key: str, condition_value: Any):
            """Filter items by condition"""
            try:
                if not isinstance(items, list):
                    return []
                return [item for item in items 
                       if isinstance(item, dict) and item.get(condition_key) == condition_value]
            except (TypeError, AttributeError):
                return []
        
        def has_all(items, required_items):
            """Check if all required items are present"""
            try:
                required_set = set(required_items) if isinstance(required_items, list) else {required_items}
                item_set = set(items) if isinstance(items, list) else {items}
                return required_set.issubset(item_set)
            except TypeError:
                return False
        
        def has_any(items, possible_items):
            """Check if any of the possible items are present"""
            try:
                possible_set = set(possible_items) if isinstance(possible_items, list) else {possible_items}
                item_set = set(items) if isinstance(items, list) else {items}
                return bool(possible_set.intersection(item_set))
            except TypeError:
                return False
        
        def regex_match(text: str, pattern: str):
            """Check if text matches regex pattern"""
            try:
                return bool(re.match(pattern, str(text)))
            except (re.error, TypeError):
                return False
        
        def regex_search(text: str, pattern: str):
            """Search for regex pattern in text"""
            try:
                return bool(re.search(pattern, str(text)))
            except (re.error, TypeError):
                return False
        
        def is_numeric(value):
            """Check if value is numeric"""
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                return False
        
        def to_number(value, default=0):
            """Convert value to number with fallback"""
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        def percentage_of(value, total):
            """Calculate percentage"""
            try:
                return (value / total) * 100 if total != 0 else 0
            except (TypeError, ZeroDivisionError):
                return 0
        
        # Register all functions
        functions = {
            'now': now,
            'today': today,
            'days_ago': days_ago,
            'hours_ago': hours_ago,
            'parse_date': parse_date,
            'extract_json': extract_json,
            'count_items': count_items,
            'sum_values': sum_values,
            'avg_values': avg_values,
            'filter_by': filter_by,
            'has_all': has_all,
            'has_any': has_any,
            'regex_match': regex_match,
            'regex_search': regex_search,
            'is_numeric': is_numeric,
            'to_number': to_number,
            'percentage_of': percentage_of
        }
        
        self.jinja_env.globals.update(functions)
    
    def evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate a condition expression with enhanced context
        
        Args:
            condition: Jinja2 template expression to evaluate
            context: Context dictionary with variables and task results
            
        Returns:
            Boolean result of condition evaluation
        """
        try:
            # Prepare enhanced context
            enhanced_context = self._prepare_context(context)
            
            # Create and render template
            template = self.jinja_env.from_string(condition)
            result = template.render(enhanced_context)
            
            # Convert result to boolean
            return self._to_boolean(result)
            
        except TemplateSyntaxError as e:
            logger.error(f"Condition syntax error: {e}", condition=condition)
            return False
        except Exception as e:
            logger.error(f"Condition evaluation error: {e}", condition=condition, exc_info=True)
            return False
    
    def evaluate_complex_condition(self, condition_spec: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Evaluate complex condition specification with operators and logic
        
        Args:
            condition_spec: Complex condition specification
            context: Context dictionary
            
        Returns:
            Boolean result of condition evaluation
        """
        try:
            if 'expression' in condition_spec:
                # Simple Jinja2 expression
                return self.evaluate_condition(condition_spec['expression'], context)
            
            elif 'operator' in condition_spec:
                # Operator-based condition
                return self._evaluate_operator_condition(condition_spec, context)
            
            elif 'all' in condition_spec:
                # All conditions must be true (AND logic)
                return all(self.evaluate_complex_condition(cond, context) 
                          for cond in condition_spec['all'])
            
            elif 'any' in condition_spec:
                # Any condition must be true (OR logic)
                return any(self.evaluate_complex_condition(cond, context) 
                          for cond in condition_spec['any'])
            
            elif 'not' in condition_spec:
                # Negation
                return not self.evaluate_complex_condition(condition_spec['not'], context)
            
            else:
                logger.warning(f"Unknown condition specification: {condition_spec}")
                return False
                
        except Exception as e:
            logger.error(f"Complex condition evaluation error: {e}", 
                        condition_spec=condition_spec, exc_info=True)
            return False
    
    def _evaluate_operator_condition(self, condition_spec: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate operator-based condition"""
        operator = condition_spec.get('operator')
        left_value = self._get_value(condition_spec.get('left'), context)
        right_value = self._get_value(condition_spec.get('right'), context)
        
        if operator not in self.operators:
            logger.error(f"Unknown operator: {operator}")
            return False
        
        try:
            return self.operators[operator](left_value, right_value)
        except Exception as e:
            logger.error(f"Operator evaluation error: {e}", 
                        operator=operator, left=left_value, right=right_value)
            return False
    
    def _get_value(self, value_spec: Any, context: Dict[str, Any]) -> Any:
        """Get value from specification (literal, context reference, or expression)"""
        if isinstance(value_spec, dict):
            if 'context' in value_spec:
                # Context reference like {"context": "tasks.task1.result"}
                return self._get_context_value(value_spec['context'], context)
            elif 'expression' in value_spec:
                # Jinja2 expression like {"expression": "{{ tasks.task1.count + 5 }}"}
                template = self.jinja_env.from_string(value_spec['expression'])
                return template.render(self._prepare_context(context))
            elif 'literal' in value_spec:
                # Literal value like {"literal": 42}
                return value_spec['literal']
        
        # Assume literal value
        return value_spec
    
    def _get_context_value(self, path: str, context: Dict[str, Any]) -> Any:
        """Get value from context using dot notation path"""
        try:
            keys = path.split('.')
            value = context
            
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                elif hasattr(value, key):
                    value = getattr(value, key)
                else:
                    return None
                    
                if value is None:
                    return None
            
            return value
        except Exception:
            return None
    
    def _prepare_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare enhanced context for evaluation"""
        enhanced_context = context.copy()
        
        # Add utility objects
        enhanced_context.update({
            'datetime': datetime,
            'timedelta': timedelta,
            'json': json,
            're': re
        })
        
        # Process task results to make them more accessible
        if 'tasks' in enhanced_context:
            enhanced_context['tasks'] = self._process_task_results(enhanced_context['tasks'])
        
        return enhanced_context
    
    def _process_task_results(self, tasks: Dict[str, Any]) -> Dict[str, Any]:
        """Process task results to make them more accessible in conditions"""
        processed_tasks = {}
        
        for task_id, task_data in tasks.items():
            if isinstance(task_data, dict):
                processed_tasks[task_id] = task_data.copy()
            elif isinstance(task_data, str):
                # Try to parse JSON strings
                try:
                    parsed = json.loads(task_data)
                    processed_tasks[task_id] = {
                        'result': parsed,
                        'raw': task_data,
                        'type': 'json'
                    }
                except json.JSONDecodeError:
                    processed_tasks[task_id] = {
                        'result': task_data,
                        'raw': task_data,
                        'type': 'string'
                    }
            else:
                processed_tasks[task_id] = {
                    'result': task_data,
                    'raw': str(task_data),
                    'type': type(task_data).__name__
                }
        
        return processed_tasks
    
    def _to_boolean(self, value: Any) -> bool:
        """Convert various types to boolean"""
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
        elif isinstance(value, (int, float)):
            return value != 0
        elif value is None:
            return False
        else:
            return bool(value)
    
    def validate_condition(self, condition: str) -> Dict[str, Any]:
        """
        Validate condition syntax and return validation result
        
        Args:
            condition: Condition expression to validate
            
        Returns:
            Validation result with success status and error details
        """
        try:
            # Try to parse the template
            template = self.jinja_env.from_string(condition)
            
            # Try to render with minimal context to check for syntax errors
            minimal_context = {'tasks': {}, 'shared_context': {}}
            template.render(minimal_context)
            
            return {
                'valid': True,
                'message': 'Condition syntax is valid'
            }
            
        except TemplateSyntaxError as e:
            return {
                'valid': False,
                'error': 'syntax_error',
                'message': f'Template syntax error: {e}',
                'line': getattr(e, 'lineno', None)
            }
        except Exception as e:
            return {
                'valid': False,
                'error': 'unknown_error',
                'message': f'Validation error: {e}'
            }
    
    def get_condition_dependencies(self, condition: str) -> List[str]:
        """
        Extract task dependencies from condition expression
        
        Args:
            condition: Condition expression
            
        Returns:
            List of task IDs that the condition depends on
        """
        dependencies = []
        
        try:
            # Parse the template to extract variables
            template = self.jinja_env.from_string(condition)
            
            # Look for task references in the form tasks.task_id
            task_pattern = r'tasks\.([a-zA-Z_][a-zA-Z0-9_]*)'
            matches = re.findall(task_pattern, condition)
            dependencies.extend(matches)
            
        except Exception as e:
            logger.warning(f"Could not extract dependencies from condition: {e}", 
                          condition=condition)
        
        return list(set(dependencies))  # Remove duplicates

# Global instance
conditional_evaluator = ConditionalEvaluator() 