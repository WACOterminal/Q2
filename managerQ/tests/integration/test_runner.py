#!/usr/bin/env python3
"""
Comprehensive test runner for Agent Coordination Framework
Provides test categorization, performance metrics, and detailed reporting
"""

import pytest
import asyncio
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result with metrics"""
    test_name: str
    category: str
    status: str  # passed, failed, skipped
    duration: float
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None

@dataclass
class TestSuiteResults:
    """Complete test suite results"""
    total_tests: int
    passed: int
    failed: int
    skipped: int
    total_duration: float
    categories: Dict[str, Dict[str, int]]
    detailed_results: List[TestResult]
    system_info: Dict[str, Any]
    timestamp: str

class CoordinationTestRunner:
    """Enhanced test runner for coordination framework"""
    
    def __init__(self):
        self.test_categories = {
            'unit': 'Individual component tests',
            'integration': 'Component integration tests',
            'end_to_end': 'Complete workflow tests',
            'performance': 'Performance and load tests',
            'stress': 'Stress and chaos tests',
            'failure': 'Failure scenario tests'
        }
        
        self.test_markers = {
            'unit': '-m "not integration and not e2e and not performance"',
            'integration': '-m "integration"',
            'end_to_end': '-m "e2e"',
            'performance': '-m "performance"',
            'stress': '-m "stress"',
            'failure': '-m "failure"',
            'smoke': '-m "smoke"',
            'critical': '-m "critical"'
        }
    
    def run_tests(self, category: str = "all", verbose: bool = True, 
                 parallel: bool = False, report_format: str = "json") -> TestSuiteResults:
        """Run tests for specified category"""
        
        logger.info(f"Starting coordination framework tests - Category: {category}")
        start_time = time.time()
        
        # Prepare pytest arguments
        pytest_args = [
            "managerQ/tests/integration/",
            "-v" if verbose else "-q",
            "--tb=short",
            f"--junitxml=test_results_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"
        ]
        
        # Add category-specific markers
        if category != "all" and category in self.test_markers:
            pytest_args.extend(self.test_markers[category].split())
        
        # Add parallel execution if requested
        if parallel:
            pytest_args.extend(["-n", "auto"])
        
        # Run tests
        logger.info(f"Executing: pytest {' '.join(pytest_args)}")
        exit_code = pytest.main(pytest_args)
        
        # Calculate metrics
        duration = time.time() - start_time
        
        # Parse results (simplified - in real implementation would parse pytest output)
        results = TestSuiteResults(
            total_tests=0,
            passed=0,
            failed=0,
            skipped=0,
            total_duration=duration,
            categories={},
            detailed_results=[],
            system_info=self._get_system_info(),
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Tests completed in {duration:.2f}s - Exit code: {exit_code}")
        
        return results
    
    def run_smoke_tests(self) -> bool:
        """Run quick smoke tests for basic functionality"""
        logger.info("Running smoke tests...")
        
        smoke_tests = [
            "test_coordination_framework.py::TestAgentRegistryIntegration::test_agent_discovery_and_selection",
            "test_coordination_framework.py::TestTaskDispatcherIntegration::test_intelligent_task_routing",
            "test_coordination_framework.py::TestAgentCommunicationIntegration::test_direct_messaging"
        ]
        
        pytest_args = smoke_tests + ["-v", "--tb=short"]
        exit_code = pytest.main(pytest_args)
        
        success = exit_code == 0
        logger.info(f"Smoke tests {'PASSED' if success else 'FAILED'}")
        return success
    
    def run_performance_benchmarks(self) -> Dict[str, float]:
        """Run performance benchmarks"""
        logger.info("Running performance benchmarks...")
        
        benchmark_results = {}
        
        # Agent selection benchmark
        pytest_args = [
            "test_coordination_framework.py::TestPerformanceBenchmarks::test_agent_selection_performance",
            "-v", "-s"
        ]
        pytest.main(pytest_args)
        
        # Metrics collection benchmark  
        pytest_args = [
            "test_coordination_framework.py::TestPerformanceBenchmarks::test_metrics_collection_performance",
            "-v", "-s"
        ]
        pytest.main(pytest_args)
        
        return benchmark_results
    
    def validate_system_readiness(self) -> Dict[str, bool]:
        """Validate system is ready for testing"""
        logger.info("Validating system readiness...")
        
        checks = {
            'imports': self._check_imports(),
            'dependencies': self._check_dependencies(),
            'configuration': self._check_configuration(),
        }
        
        all_passed = all(checks.values())
        logger.info(f"System readiness: {'READY' if all_passed else 'NOT READY'}")
        
        return checks
    
    def _check_imports(self) -> bool:
        """Check if all required modules can be imported"""
        try:
            from managerQ.app.core.agent_registry import AgentRegistry
            from managerQ.app.core.task_dispatcher import TaskDispatcher
            from managerQ.app.core.failure_handler import FailureHandler
            from managerQ.app.core.agent_communication import AgentCommunicationHub
            from managerQ.app.core.coordination_protocols import CoordinationProtocolManager
            from managerQ.app.core.performance_monitor import PerformanceMonitor
            from managerQ.app.core.predictive_autoscaler import PredictiveAutoScaler
            return True
        except ImportError as e:
            logger.error(f"Import error: {e}")
            return False
    
    def _check_dependencies(self) -> bool:
        """Check if required dependencies are available"""
        try:
            import pytest
            import asyncio
            import numpy
            return True
        except ImportError as e:
            logger.error(f"Dependency error: {e}")
            return False
    
    def _check_configuration(self) -> bool:
        """Check if test configuration is valid"""
        # Add configuration validation logic here
        return True
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for test reporting"""
        import platform
        import sys
        
        return {
            'platform': platform.platform(),
            'python_version': sys.version,
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'memory_total': self._get_memory_info(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_memory_info(self) -> str:
        """Get memory information"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return f"{memory.total // (1024**3)}GB"
        except ImportError:
            return "Unknown"
    
    def generate_report(self, results: TestSuiteResults, format: str = "json") -> str:
        """Generate test report in specified format"""
        
        if format == "json":
            return json.dumps(asdict(results), indent=2, default=str)
        elif format == "html":
            return self._generate_html_report(results)
        elif format == "markdown":
            return self._generate_markdown_report(results)
        else:
            return str(results)
    
    def _generate_html_report(self, results: TestSuiteResults) -> str:
        """Generate HTML test report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Coordination Framework Test Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .metric {{ text-align: center; padding: 10px; background-color: #e9e9e9; border-radius: 5px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .skipped {{ color: orange; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Agent Coordination Framework Test Results</h1>
                <p>Generated: {results.timestamp}</p>
                <p>Duration: {results.total_duration:.2f}s</p>
            </div>
            
            <div class="summary">
                <div class="metric">
                    <h3>Total Tests</h3>
                    <p>{results.total_tests}</p>
                </div>
                <div class="metric passed">
                    <h3>Passed</h3>
                    <p>{results.passed}</p>
                </div>
                <div class="metric failed">
                    <h3>Failed</h3>
                    <p>{results.failed}</p>
                </div>
                <div class="metric skipped">
                    <h3>Skipped</h3>
                    <p>{results.skipped}</p>
                </div>
            </div>
            
            <h2>System Information</h2>
            <pre>{json.dumps(results.system_info, indent=2)}</pre>
        </body>
        </html>
        """
        return html
    
    def _generate_markdown_report(self, results: TestSuiteResults) -> str:
        """Generate Markdown test report"""
        
        pass_rate = (results.passed / max(results.total_tests, 1)) * 100
        
        markdown = f"""
# Agent Coordination Framework Test Results

**Generated:** {results.timestamp}  
**Duration:** {results.total_duration:.2f}s  
**Pass Rate:** {pass_rate:.1f}%

## Summary

| Metric | Count |
|--------|--------|
| Total Tests | {results.total_tests} |
| ✅ Passed | {results.passed} |
| ❌ Failed | {results.failed} |
| ⏭️ Skipped | {results.skipped} |

## System Information

```json
{json.dumps(results.system_info, indent=2)}
```

## Test Categories

"""
        
        for category, counts in results.categories.items():
            markdown += f"### {category.title()}\n"
            for status, count in counts.items():
                markdown += f"- {status}: {count}\n"
            markdown += "\n"
        
        return markdown

def main():
    """Main test runner entry point"""
    parser = argparse.ArgumentParser(description="Coordination Framework Test Runner")
    parser.add_argument("--category", "-c", default="all", 
                       choices=["all", "unit", "integration", "e2e", "performance", "stress"],
                       help="Test category to run")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests only")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--validate", action="store_true", help="Validate system readiness")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run tests in parallel")
    parser.add_argument("--report", "-r", choices=["json", "html", "markdown"], 
                       default="json", help="Report format")
    parser.add_argument("--output", "-o", help="Output file for report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    runner = CoordinationTestRunner()
    
    # Validate system readiness
    if args.validate or args.category != "smoke":
        readiness = runner.validate_system_readiness()
        if not all(readiness.values()):
            logger.error("System not ready for testing")
            return 1
    
    # Run smoke tests
    if args.smoke:
        success = runner.run_smoke_tests()
        return 0 if success else 1
    
    # Run performance benchmarks
    if args.benchmark:
        benchmarks = runner.run_performance_benchmarks()
        print(json.dumps(benchmarks, indent=2))
        return 0
    
    # Run main test suite
    results = runner.run_tests(
        category=args.category,
        verbose=args.verbose,
        parallel=args.parallel,
        report_format=args.report
    )
    
    # Generate report
    report = runner.generate_report(results, args.report)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        logger.info(f"Report written to {args.output}")
    else:
        print(report)
    
    # Return appropriate exit code
    return 0 if results.failed == 0 else 1

if __name__ == "__main__":
    exit(main()) 