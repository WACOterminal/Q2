#!/usr/bin/env python3
"""
Validation script for the Agent Coordination Framework Integration Testing Suite
Ensures all test components work correctly before running the full test suite
"""

import sys
import os
import importlib
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FrameworkValidator:
    """Validates the integration testing framework"""
    
    def __init__(self):
        self.validation_results = {
            'imports': {},
            'fixtures': {},
            'test_components': {},
            'performance': {},
            'overall_status': 'unknown'
        }
    
    def validate_imports(self) -> bool:
        """Validate that all required modules can be imported"""
        logger.info("Validating imports...")
        
        required_modules = [
            'managerQ.app.core.agent_registry',
            'managerQ.app.core.task_dispatcher',
            'managerQ.app.core.failure_handler',
            'managerQ.app.core.agent_communication',
            'managerQ.app.core.coordination_protocols',
            'managerQ.app.core.performance_monitor',
            'managerQ.app.core.predictive_autoscaler',
            'shared.pulsar_client'
        ]
        
        test_modules = [
            'pytest',
            'asyncio',
            'unittest.mock',
            'numpy'
        ]
        
        all_success = True
        
        for module_name in required_modules:
            try:
                module = importlib.import_module(module_name)
                self.validation_results['imports'][module_name] = 'success'
                logger.info(f"✓ Successfully imported {module_name}")
            except ImportError as e:
                self.validation_results['imports'][module_name] = f'failed: {e}'
                logger.error(f"✗ Failed to import {module_name}: {e}")
                all_success = False
        
        for module_name in test_modules:
            try:
                importlib.import_module(module_name)
                self.validation_results['imports'][module_name] = 'success'
                logger.info(f"✓ Successfully imported test module {module_name}")
            except ImportError as e:
                self.validation_results['imports'][module_name] = f'failed: {e}'
                logger.error(f"✗ Failed to import test module {module_name}: {e}")
                all_success = False
        
        return all_success
    
    def validate_test_fixtures(self) -> bool:
        """Validate that test fixtures work correctly"""
        logger.info("Validating test fixtures...")
        
        try:
            # Import test configuration
            from managerQ.tests.integration.conftest import (
                TEST_CONFIG,
                TestDataGenerator,
                assert_agent_healthy,
                assert_system_stable
            )
            
            # Test data generator
            generator = TestDataGenerator()
            
            # Generate test data
            task_data = generator.generate_task_data(5)
            self.validation_results['fixtures']['task_data'] = len(task_data) == 5
            
            failure_scenarios = generator.generate_failure_scenarios()
            self.validation_results['fixtures']['failure_scenarios'] = len(failure_scenarios) > 0
            
            performance_scenarios = generator.generate_performance_scenarios()
            self.validation_results['fixtures']['performance_scenarios'] = len(performance_scenarios) > 0
            
            logger.info("✓ Test fixtures validated successfully")
            return True
            
        except Exception as e:
            self.validation_results['fixtures']['error'] = str(e)
            logger.error(f"✗ Test fixtures validation failed: {e}")
            return False
    
    def validate_coordination_components(self) -> bool:
        """Validate that coordination components can be instantiated"""
        logger.info("Validating coordination components...")
        
        try:
            from unittest.mock import Mock
            from managerQ.app.core.agent_registry import AgentRegistry, Agent, AgentCapabilities
            from managerQ.app.core.task_dispatcher import TaskDispatcher
            from managerQ.app.core.failure_handler import FailureHandler
            from managerQ.app.core.agent_communication import AgentCommunicationHub
            from managerQ.app.core.coordination_protocols import CoordinationProtocolManager
            from managerQ.app.core.performance_monitor import PerformanceMonitor
            from managerQ.app.core.predictive_autoscaler import PredictiveAutoScaler
            
            # Mock Pulsar client
            mock_pulsar = Mock()
            mock_pulsar._client = Mock()
            mock_pulsar._connect = Mock()
            mock_pulsar.publish_message = Mock()
            
            # Test component instantiation
            components = {}
            
            # Agent Registry
            components['agent_registry'] = AgentRegistry(mock_pulsar)
            logger.info("✓ AgentRegistry instantiated")
            
            # Task Dispatcher
            components['task_dispatcher'] = TaskDispatcher(mock_pulsar, components['agent_registry'])
            logger.info("✓ TaskDispatcher instantiated")
            
            # Failure Handler
            components['failure_handler'] = FailureHandler(
                components['agent_registry'], 
                components['task_dispatcher'], 
                mock_pulsar
            )
            logger.info("✓ FailureHandler instantiated")
            
            # Communication Hub
            components['communication_hub'] = AgentCommunicationHub(components['agent_registry'], mock_pulsar)
            logger.info("✓ AgentCommunicationHub instantiated")
            
            # Coordination Manager
            components['coordination_manager'] = CoordinationProtocolManager(
                "test_node", 
                components['agent_registry'], 
                components['communication_hub']
            )
            logger.info("✓ CoordinationProtocolManager instantiated")
            
            # Performance Monitor
            components['performance_monitor'] = PerformanceMonitor(
                components['agent_registry'],
                components['task_dispatcher'],
                components['failure_handler'],
                components['communication_hub'],
                components['coordination_manager']
            )
            logger.info("✓ PerformanceMonitor instantiated")
            
            # Predictive Autoscaler
            components['predictive_autoscaler'] = PredictiveAutoScaler(
                components['agent_registry'],
                components['task_dispatcher'],
                components['performance_monitor']
            )
            logger.info("✓ PredictiveAutoScaler instantiated")
            
            self.validation_results['test_components'] = {
                name: 'success' for name in components.keys()
            }
            
            return True
            
        except Exception as e:
            self.validation_results['test_components']['error'] = str(e)
            logger.error(f"✗ Component validation failed: {e}")
            return False
    
    async def validate_async_functionality(self) -> bool:
        """Validate async functionality works correctly"""
        logger.info("Validating async functionality...")
        
        try:
            # Test basic async operations
            await asyncio.sleep(0.1)
            
            # Test async fixture creation
            from unittest.mock import Mock
            from managerQ.app.core.agent_registry import AgentRegistry
            
            mock_pulsar = Mock()
            registry = AgentRegistry(mock_pulsar)
            
            # Test async methods exist
            assert hasattr(registry, 'process_heartbeat')
            
            logger.info("✓ Async functionality validated")
            return True
            
        except Exception as e:
            self.validation_results['async']['error'] = str(e)
            logger.error(f"✗ Async validation failed: {e}")
            return False
    
    def validate_performance_tools(self) -> bool:
        """Validate performance measurement tools"""
        logger.info("Validating performance tools...")
        
        try:
            # Test performance timer
            from managerQ.tests.integration.conftest import TEST_CONFIG
            
            # Mock performance timer
            start_time = time.time()
            time.sleep(0.01)  # Small sleep
            end_time = time.time()
            
            duration = end_time - start_time
            assert duration > 0.005  # Should be at least 5ms
            
            # Test thresholds
            thresholds = TEST_CONFIG['performance_thresholds']
            assert 'agent_selection_time_ms' in thresholds
            assert 'metrics_collection_time_ms' in thresholds
            
            self.validation_results['performance']['timer'] = 'success'
            self.validation_results['performance']['thresholds'] = 'success'
            
            logger.info("✓ Performance tools validated")
            return True
            
        except Exception as e:
            self.validation_results['performance']['error'] = str(e)
            logger.error(f"✗ Performance tools validation failed: {e}")
            return False
    
    def validate_test_environment(self) -> bool:
        """Validate test environment setup"""
        logger.info("Validating test environment...")
        
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 8):
                logger.error(f"✗ Python 3.8+ required, got {python_version}")
                return False
            
            # Check test directory structure
            test_dir = Path(__file__).parent
            required_files = [
                'test_coordination_framework.py',
                'test_runner.py',
                'conftest.py',
                'run_tests.sh'
            ]
            
            for file_name in required_files:
                file_path = test_dir / file_name
                if not file_path.exists():
                    logger.error(f"✗ Required test file missing: {file_name}")
                    return False
                else:
                    logger.info(f"✓ Found test file: {file_name}")
            
            # Check if run_tests.sh is executable
            run_tests_path = test_dir / 'run_tests.sh'
            if not os.access(run_tests_path, os.X_OK):
                logger.warning("⚠ run_tests.sh may not be executable")
            
            logger.info("✓ Test environment validated")
            return True
            
        except Exception as e:
            self.validation_results['environment'] = {'error': str(e)}
            logger.error(f"✗ Test environment validation failed: {e}")
            return False
    
    def run_mini_test_suite(self) -> bool:
        """Run a minimal test suite to validate framework functionality"""
        logger.info("Running mini test suite...")
        
        try:
            import subprocess
            import tempfile
            
            # Create a minimal test file
            mini_test_content = '''
import pytest
import asyncio
from unittest.mock import Mock

@pytest.mark.asyncio
async def test_basic_functionality():
    """Basic test to validate framework works"""
    assert True
    
def test_imports():
    """Test that imports work"""
    from managerQ.app.core.agent_registry import AgentRegistry
    assert AgentRegistry is not None
    
def test_mock_pulsar():
    """Test mock Pulsar client"""
    mock_pulsar = Mock()
    mock_pulsar.publish_message = Mock()
    mock_pulsar.publish_message("test")
    assert mock_pulsar.publish_message.called
'''
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(mini_test_content)
                temp_test_file = f.name
            
            try:
                # Run the mini test
                result = subprocess.run([
                    sys.executable, '-m', 'pytest', temp_test_file, '-v'
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    logger.info("✓ Mini test suite passed")
                    return True
                else:
                    logger.error(f"✗ Mini test suite failed: {result.stderr}")
                    return False
                    
            finally:
                os.unlink(temp_test_file)
                
        except Exception as e:
            logger.error(f"✗ Mini test suite failed: {e}")
            return False
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        # Count successes and failures
        total_checks = 0
        passed_checks = 0
        
        for category, results in self.validation_results.items():
            if category == 'overall_status':
                continue
                
            if isinstance(results, dict):
                for check, result in results.items():
                    total_checks += 1
                    if result == 'success' or result is True:
                        passed_checks += 1
        
        success_rate = (passed_checks / max(total_checks, 1)) * 100
        
        report = {
            'validation_timestamp': time.time(),
            'validation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'success_rate': success_rate,
            'overall_status': 'PASS' if success_rate >= 80 else 'FAIL',
            'detailed_results': self.validation_results,
            'recommendations': []
        }
        
        # Add recommendations based on failures
        if success_rate < 100:
            report['recommendations'].append("Some validation checks failed. Review detailed results.")
        
        if success_rate < 80:
            report['recommendations'].append("Critical validation failures detected. Framework may not be ready for testing.")
        
        if success_rate >= 95:
            report['recommendations'].append("Framework validation passed. Ready for full integration testing.")
        
        return report
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        logger.info("Starting comprehensive framework validation...")
        
        validation_steps = [
            ('imports', self.validate_imports),
            ('test_fixtures', self.validate_test_fixtures),
            ('coordination_components', self.validate_coordination_components),
            ('async_functionality', self.validate_async_functionality),
            ('performance_tools', self.validate_performance_tools),
            ('test_environment', self.validate_test_environment),
            ('mini_test_suite', self.run_mini_test_suite)
        ]
        
        results = {}
        overall_success = True
        
        for step_name, validation_func in validation_steps:
            logger.info(f"Running validation step: {step_name}")
            
            try:
                if asyncio.iscoroutinefunction(validation_func):
                    success = await validation_func()
                else:
                    success = validation_func()
                
                results[step_name] = success
                
                if not success:
                    overall_success = False
                    logger.error(f"✗ Validation step failed: {step_name}")
                else:
                    logger.info(f"✓ Validation step passed: {step_name}")
                    
            except Exception as e:
                results[step_name] = False
                overall_success = False
                logger.error(f"✗ Validation step error: {step_name} - {e}")
        
        # Generate final report
        report = self.generate_validation_report()
        report['step_results'] = results
        report['overall_success'] = overall_success
        
        logger.info(f"Framework validation complete. Overall success: {overall_success}")
        
        return report

async def main():
    """Main validation entry point"""
    validator = FrameworkValidator()
    
    try:
        report = await validator.run_full_validation()
        
        # Print summary
        print("\n" + "="*60)
        print("AGENT COORDINATION FRAMEWORK VALIDATION SUMMARY")
        print("="*60)
        print(f"Overall Status: {report['overall_status']}")
        print(f"Success Rate: {report['success_rate']:.1f}%")
        print(f"Passed Checks: {report['passed_checks']}/{report['total_checks']}")
        print(f"Validation Date: {report['validation_date']}")
        
        if report['recommendations']:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
        
        # Save detailed report
        report_file = Path(__file__).parent / f"validation_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nDetailed report saved to: {report_file}")
        
        # Return appropriate exit code
        return 0 if report['overall_success'] else 1
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        return 1

if __name__ == "__main__":
    # Run validation
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 