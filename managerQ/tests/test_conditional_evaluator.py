# managerQ/tests/test_conditional_evaluator.py
import pytest
import json
from datetime import datetime, timedelta

from managerQ.app.core.conditional_evaluator import ConditionalEvaluator

@pytest.fixture
def evaluator():
    """Create a conditional evaluator instance"""
    return ConditionalEvaluator()

@pytest.fixture
def sample_context():
    """Create sample context for testing"""
    return {
        'tasks': {
            'task1': {
                'result': 'success',
                'count': 5,
                'items': ['a', 'b', 'c']
            },
            'task2': {
                'result': {'status': 'completed', 'score': 85}
            },
            'task3': {
                'result': 'failure'
            }
        },
        'shared_context': {
            'environment': 'production',
            'threshold': 10,
            'user_roles': ['admin', 'developer']
        }
    }

class TestBasicConditionEvaluation:
    """Test basic condition evaluation"""
    
    def test_simple_equality_condition(self, evaluator, sample_context):
        """Test simple equality condition"""
        condition = "{{ tasks.task1.result == 'success' }}"
        assert evaluator.evaluate_condition(condition, sample_context) is True
        
        condition = "{{ tasks.task1.result == 'failure' }}"
        assert evaluator.evaluate_condition(condition, sample_context) is False
    
    def test_numeric_comparison(self, evaluator, sample_context):
        """Test numeric comparison conditions"""
        condition = "{{ tasks.task1.count > 3 }}"
        assert evaluator.evaluate_condition(condition, sample_context) is True
        
        condition = "{{ tasks.task1.count <= 5 }}"
        assert evaluator.evaluate_condition(condition, sample_context) is True
        
        condition = "{{ tasks.task1.count > 10 }}"
        assert evaluator.evaluate_condition(condition, sample_context) is False
    
    def test_string_operations(self, evaluator, sample_context):
        """Test string operation conditions"""
        condition = "{{ 'success' in tasks.task1.result }}"
        assert evaluator.evaluate_condition(condition, sample_context) is True
        
        condition = "{{ tasks.task1.result.startswith('succ') }}"
        assert evaluator.evaluate_condition(condition, sample_context) is True
    
    def test_list_operations(self, evaluator, sample_context):
        """Test list operation conditions"""
        condition = "{{ 'a' in tasks.task1.items }}"
        assert evaluator.evaluate_condition(condition, sample_context) is True
        
        condition = "{{ tasks.task1.items|length > 2 }}"
        assert evaluator.evaluate_condition(condition, sample_context) is True
    
    def test_nested_object_access(self, evaluator, sample_context):
        """Test accessing nested objects in conditions"""
        condition = "{{ tasks.task2.result.status == 'completed' }}"
        assert evaluator.evaluate_condition(condition, sample_context) is True
        
        condition = "{{ tasks.task2.result.score >= 80 }}"
        assert evaluator.evaluate_condition(condition, sample_context) is True

class TestCustomFunctions:
    """Test custom functions in conditional evaluator"""
    
    def test_count_items_function(self, evaluator, sample_context):
        """Test count_items custom function"""
        condition = "{{ count_items(tasks.task1.items) == 3 }}"
        assert evaluator.evaluate_condition(condition, sample_context) is True
    
    def test_extract_json_function(self, evaluator):
        """Test extract_json custom function"""
        context = {
            'tasks': {
                'task1': {
                    'result': '{"status": "success", "code": 200}'
                }
            }
        }
        
        condition = "{{ extract_json(tasks.task1.result, 'status') == 'success' }}"
        assert evaluator.evaluate_condition(condition, context) is True
        
        condition = "{{ extract_json(tasks.task1.result, 'code') == 200 }}"
        assert evaluator.evaluate_condition(condition, context) is True
    
    def test_has_all_function(self, evaluator, sample_context):
        """Test has_all custom function"""
        condition = "{{ has_all(shared_context.user_roles, ['admin']) }}"
        assert evaluator.evaluate_condition(condition, sample_context) is True
        
        condition = "{{ has_all(shared_context.user_roles, ['admin', 'developer']) }}"
        assert evaluator.evaluate_condition(condition, sample_context) is True
        
        condition = "{{ has_all(shared_context.user_roles, ['admin', 'superuser']) }}"
        assert evaluator.evaluate_condition(condition, sample_context) is False
    
    def test_has_any_function(self, evaluator, sample_context):
        """Test has_any custom function"""
        condition = "{{ has_any(shared_context.user_roles, ['admin', 'superuser']) }}"
        assert evaluator.evaluate_condition(condition, sample_context) is True
        
        condition = "{{ has_any(shared_context.user_roles, ['guest', 'anonymous']) }}"
        assert evaluator.evaluate_condition(condition, sample_context) is False
    
    def test_regex_functions(self, evaluator):
        """Test regex matching functions"""
        context = {
            'tasks': {
                'task1': {
                    'result': 'Error: Connection failed on port 8080'
                }
            }
        }
        
        condition = "{{ regex_search(tasks.task1.result, 'Error:.*failed') }}"
        assert evaluator.evaluate_condition(condition, context) is True
        
        condition = "{{ regex_match(tasks.task1.result, 'Error:.*') }}"
        assert evaluator.evaluate_condition(condition, context) is True
    
    def test_numeric_functions(self, evaluator):
        """Test numeric utility functions"""
        context = {
            'tasks': {
                'task1': {
                    'numbers': [10, 20, 30, 40]
                }
            }
        }
        
        condition = "{{ sum_values(tasks.task1.numbers) == 100 }}"
        assert evaluator.evaluate_condition(condition, context) is True
        
        condition = "{{ avg_values(tasks.task1.numbers) == 25 }}"
        assert evaluator.evaluate_condition(condition, context) is True
    
    def test_datetime_functions(self, evaluator):
        """Test datetime utility functions"""
        condition = "{{ now().year == %d }}" % datetime.now().year
        assert evaluator.evaluate_condition(condition, {}) is True
        
        condition = "{{ hours_ago(2) < now() }}"
        assert evaluator.evaluate_condition(condition, {}) is True

class TestComplexConditions:
    """Test complex condition evaluation with operators and logic"""
    
    def test_operator_based_condition(self, evaluator, sample_context):
        """Test operator-based conditions"""
        condition_spec = {
            'operator': '==',
            'left': {'context': 'tasks.task1.result'},
            'right': {'literal': 'success'}
        }
        
        assert evaluator.evaluate_complex_condition(condition_spec, sample_context) is True
        
        condition_spec = {
            'operator': '>',
            'left': {'context': 'tasks.task1.count'},
            'right': {'literal': 3}
        }
        
        assert evaluator.evaluate_complex_condition(condition_spec, sample_context) is True
    
    def test_all_logic_condition(self, evaluator, sample_context):
        """Test ALL (AND) logic conditions"""
        condition_spec = {
            'all': [
                {
                    'operator': '==',
                    'left': {'context': 'tasks.task1.result'},
                    'right': {'literal': 'success'}
                },
                {
                    'operator': '>',
                    'left': {'context': 'tasks.task1.count'},
                    'right': {'literal': 3}
                }
            ]
        }
        
        assert evaluator.evaluate_complex_condition(condition_spec, sample_context) is True
        
        # Make one condition false
        condition_spec['all'][1]['right']['literal'] = 10
        assert evaluator.evaluate_complex_condition(condition_spec, sample_context) is False
    
    def test_any_logic_condition(self, evaluator, sample_context):
        """Test ANY (OR) logic conditions"""
        condition_spec = {
            'any': [
                {
                    'operator': '==',
                    'left': {'context': 'tasks.task1.result'},
                    'right': {'literal': 'failure'}  # This is false
                },
                {
                    'operator': '>',
                    'left': {'context': 'tasks.task1.count'},
                    'right': {'literal': 3}  # This is true
                }
            ]
        }
        
        assert evaluator.evaluate_complex_condition(condition_spec, sample_context) is True
        
        # Make both conditions false
        condition_spec['any'][1]['right']['literal'] = 10
        assert evaluator.evaluate_complex_condition(condition_spec, sample_context) is False
    
    def test_not_logic_condition(self, evaluator, sample_context):
        """Test NOT logic conditions"""
        condition_spec = {
            'not': {
                'operator': '==',
                'left': {'context': 'tasks.task1.result'},
                'right': {'literal': 'failure'}
            }
        }
        
        assert evaluator.evaluate_complex_condition(condition_spec, sample_context) is True
    
    def test_nested_complex_conditions(self, evaluator, sample_context):
        """Test nested complex conditions"""
        condition_spec = {
            'all': [
                {
                    'operator': '==',
                    'left': {'context': 'tasks.task1.result'},
                    'right': {'literal': 'success'}
                },
                {
                    'any': [
                        {
                            'operator': '>',
                            'left': {'context': 'tasks.task1.count'},
                            'right': {'literal': 10}
                        },
                        {
                            'operator': '==',
                            'left': {'context': 'shared_context.environment'},
                            'right': {'literal': 'production'}
                        }
                    ]
                }
            ]
        }
        
        assert evaluator.evaluate_complex_condition(condition_spec, sample_context) is True

class TestAdvancedOperators:
    """Test advanced operators"""
    
    def test_string_operators(self, evaluator):
        """Test string-specific operators"""
        context = {
            'tasks': {
                'task1': {
                    'message': 'Hello World',
                    'filename': 'test.txt'
                }
            }
        }
        
        condition_spec = {
            'operator': 'starts_with',
            'left': {'context': 'tasks.task1.message'},
            'right': {'literal': 'Hello'}
        }
        assert evaluator.evaluate_complex_condition(condition_spec, context) is True
        
        condition_spec = {
            'operator': 'ends_with',
            'left': {'context': 'tasks.task1.filename'},
            'right': {'literal': '.txt'}
        }
        assert evaluator.evaluate_complex_condition(condition_spec, context) is True
        
        condition_spec = {
            'operator': 'contains',
            'left': {'context': 'tasks.task1.message'},
            'right': {'literal': 'World'}
        }
        assert evaluator.evaluate_complex_condition(condition_spec, context) is True
    
    def test_null_operators(self, evaluator):
        """Test null checking operators"""
        context = {
            'tasks': {
                'task1': {
                    'value': None,
                    'data': 'some_data'
                }
            }
        }
        
        condition_spec = {
            'operator': 'is_null',
            'left': {'context': 'tasks.task1.value'},
            'right': {'literal': None}
        }
        assert evaluator.evaluate_complex_condition(condition_spec, context) is True
        
        condition_spec = {
            'operator': 'is_not_null',
            'left': {'context': 'tasks.task1.data'},
            'right': {'literal': None}
        }
        assert evaluator.evaluate_complex_condition(condition_spec, context) is True
    
    def test_empty_operators(self, evaluator):
        """Test empty checking operators"""
        context = {
            'tasks': {
                'task1': {
                    'empty_list': [],
                    'full_list': [1, 2, 3],
                    'empty_string': '',
                    'full_string': 'data'
                }
            }
        }
        
        condition_spec = {
            'operator': 'is_empty',
            'left': {'context': 'tasks.task1.empty_list'},
            'right': {'literal': None}
        }
        assert evaluator.evaluate_complex_condition(condition_spec, context) is True
        
        condition_spec = {
            'operator': 'is_not_empty',
            'left': {'context': 'tasks.task1.full_list'},
            'right': {'literal': None}
        }
        assert evaluator.evaluate_complex_condition(condition_spec, context) is True

class TestErrorHandling:
    """Test error handling in conditional evaluation"""
    
    def test_invalid_syntax(self, evaluator):
        """Test handling of invalid template syntax"""
        condition = "{{ invalid syntax {{ }}"
        result = evaluator.evaluate_condition(condition, {})
        assert result is False
    
    def test_missing_context_value(self, evaluator):
        """Test handling of missing context values"""
        condition = "{{ tasks.nonexistent.result == 'success' }}"
        result = evaluator.evaluate_condition(condition, {'tasks': {}})
        assert result is False
    
    def test_type_error_in_condition(self, evaluator):
        """Test handling of type errors in conditions"""
        context = {
            'tasks': {
                'task1': {
                    'result': 'string_value'
                }
            }
        }
        
        # This should handle the error gracefully
        condition = "{{ tasks.task1.result + 5 }}"
        result = evaluator.evaluate_condition(condition, context)
        assert result is False
    
    def test_unknown_operator(self, evaluator, sample_context):
        """Test handling of unknown operators"""
        condition_spec = {
            'operator': 'unknown_operator',
            'left': {'context': 'tasks.task1.result'},
            'right': {'literal': 'success'}
        }
        
        result = evaluator.evaluate_complex_condition(condition_spec, sample_context)
        assert result is False

class TestConditionValidation:
    """Test condition validation functionality"""
    
    def test_valid_condition_validation(self, evaluator):
        """Test validation of valid conditions"""
        condition = "{{ tasks.task1.result == 'success' }}"
        result = evaluator.validate_condition(condition)
        
        assert result['valid'] is True
        assert 'message' in result
    
    def test_invalid_condition_validation(self, evaluator):
        """Test validation of invalid conditions"""
        condition = "{{ invalid syntax {{ }}"
        result = evaluator.validate_condition(condition)
        
        assert result['valid'] is False
        assert result['error'] == 'syntax_error'
        assert 'message' in result
    
    def test_dependency_extraction(self, evaluator):
        """Test extraction of task dependencies from conditions"""
        condition = "{{ tasks.task1.result == 'success' and tasks.task2.count > 5 }}"
        dependencies = evaluator.get_condition_dependencies(condition)
        
        assert 'task1' in dependencies
        assert 'task2' in dependencies
        assert len(dependencies) == 2

class TestBooleanConversion:
    """Test boolean conversion functionality"""
    
    def test_boolean_conversion(self, evaluator):
        """Test various boolean conversions"""
        assert evaluator._to_boolean(True) is True
        assert evaluator._to_boolean(False) is False
        assert evaluator._to_boolean("true") is True
        assert evaluator._to_boolean("false") is False
        assert evaluator._to_boolean("1") is True
        assert evaluator._to_boolean("0") is False
        assert evaluator._to_boolean(1) is True
        assert evaluator._to_boolean(0) is False
        assert evaluator._to_boolean(None) is False
        assert evaluator._to_boolean([1, 2, 3]) is True
        assert evaluator._to_boolean([]) is False

class TestContextProcessing:
    """Test context processing functionality"""
    
    def test_task_result_processing(self, evaluator):
        """Test processing of task results"""
        raw_tasks = {
            'task1': '{"status": "success", "count": 5}',
            'task2': 'simple string result',
            'task3': {'already': 'processed'}
        }
        
        processed = evaluator._process_task_results(raw_tasks)
        
        # JSON string should be parsed
        assert processed['task1']['type'] == 'json'
        assert processed['task1']['result']['status'] == 'success'
        assert processed['task1']['result']['count'] == 5
        
        # String should be preserved
        assert processed['task2']['type'] == 'string'
        assert processed['task2']['result'] == 'simple string result'
        
        # Dict should be copied
        assert processed['task3']['already'] == 'processed'
    
    def test_context_value_extraction(self, evaluator):
        """Test extraction of values from context using dot notation"""
        context = {
            'level1': {
                'level2': {
                    'value': 'target_value'
                }
            }
        }
        
        result = evaluator._get_context_value('level1.level2.value', context)
        assert result == 'target_value'
        
        # Test missing path
        result = evaluator._get_context_value('level1.missing.value', context)
        assert result is None 