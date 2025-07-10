#!/bin/bash
# Agent Coordination Framework Integration Test Suite
# Comprehensive test runner with multiple execution modes

set -e

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TEST_DIR="$PROJECT_ROOT/managerQ/tests/integration"
REPORTS_DIR="$PROJECT_ROOT/test_reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODE="full"
PARALLEL=false
VERBOSE=false
REPORT_FORMAT="html"
CLEANUP=true
GENERATE_COVERAGE=false

# Usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -m, --mode MODE          Test mode: smoke, unit, integration, e2e, performance, stress, full (default: full)"
    echo "  -p, --parallel           Run tests in parallel"
    echo "  -v, --verbose            Verbose output"
    echo "  -r, --report FORMAT      Report format: json, html, markdown (default: html)"
    echo "  -c, --coverage           Generate coverage report"
    echo "  --no-cleanup             Don't cleanup test artifacts"
    echo "  --ci                     CI mode (non-interactive, minimal output)"
    echo "  -h, --help               Show this help message"
    echo
    echo "Examples:"
    echo "  $0 --mode smoke          Run smoke tests only"
    echo "  $0 --mode performance -v Run performance tests with verbose output"
    echo "  $0 --parallel --coverage Run all tests in parallel with coverage"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -r|--report)
            REPORT_FORMAT="$2"
            shift 2
            ;;
        -c|--coverage)
            GENERATE_COVERAGE=true
            shift
            ;;
        --no-cleanup)
            CLEANUP=false
            shift
            ;;
        --ci)
            CI_MODE=true
            VERBOSE=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Setup test environment
setup_environment() {
    log_info "Setting up test environment..."
    
    # Create reports directory
    mkdir -p "$REPORTS_DIR"
    
    # Set Python path
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    # Install test dependencies if needed
    if [[ ! -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
        log_info "Creating virtual environment..."
        python3 -m venv "$PROJECT_ROOT/.venv"
        source "$PROJECT_ROOT/.venv/bin/activate"
        pip install -r "$PROJECT_ROOT/requirements.txt"
        pip install pytest pytest-asyncio pytest-cov pytest-html pytest-xdist
    else
        source "$PROJECT_ROOT/.venv/bin/activate"
    fi
    
    # Check dependencies
    python -c "import pytest, asyncio" || {
        log_error "Required dependencies not installed"
        exit 1
    }
    
    log_success "Environment setup complete"
}

# Validate system readiness
validate_system() {
    log_info "Validating system readiness..."
    
    cd "$PROJECT_ROOT"
    
    # Check if coordination framework components can be imported
    python -c "
from managerQ.app.core.agent_registry import AgentRegistry
from managerQ.app.core.task_dispatcher import TaskDispatcher
from managerQ.app.core.failure_handler import FailureHandler
from managerQ.app.core.agent_communication import AgentCommunicationHub
from managerQ.app.core.coordination_protocols import CoordinationProtocolManager
from managerQ.app.core.performance_monitor import PerformanceMonitor
from managerQ.app.core.predictive_autoscaler import PredictiveAutoScaler
print('All imports successful')
" || {
        log_error "System validation failed - cannot import coordination framework components"
        exit 1
    }
    
    log_success "System validation passed"
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    cd "$PROJECT_ROOT"
    
    python "$TEST_DIR/test_runner.py" --smoke --verbose || {
        log_error "Smoke tests failed"
        return 1
    }
    
    log_success "Smoke tests passed"
}

# Run specific test category
run_test_category() {
    local category=$1
    log_info "Running $category tests..."
    
    cd "$PROJECT_ROOT"
    
    # Build pytest command
    local pytest_cmd="python -m pytest $TEST_DIR/test_coordination_framework.py"
    
    # Add verbosity
    if [[ "$VERBOSE" == "true" ]]; then
        pytest_cmd="$pytest_cmd -v"
    else
        pytest_cmd="$pytest_cmd -q"
    fi
    
    # Add parallel execution
    if [[ "$PARALLEL" == "true" ]]; then
        pytest_cmd="$pytest_cmd -n auto"
    fi
    
    # Add coverage
    if [[ "$GENERATE_COVERAGE" == "true" ]]; then
        pytest_cmd="$pytest_cmd --cov=managerQ.app.core --cov-report=html:$REPORTS_DIR/coverage_$TIMESTAMP"
    fi
    
    # Add category-specific markers
    case $category in
        "smoke")
            pytest_cmd="$pytest_cmd -m smoke"
            ;;
        "unit")
            pytest_cmd="$pytest_cmd -m 'not integration and not e2e and not performance'"
            ;;
        "integration")
            pytest_cmd="$pytest_cmd -m integration"
            ;;
        "e2e")
            pytest_cmd="$pytest_cmd -m e2e"
            ;;
        "performance")
            pytest_cmd="$pytest_cmd -m performance"
            ;;
        "stress")
            pytest_cmd="$pytest_cmd -m stress"
            ;;
        "failure")
            pytest_cmd="$pytest_cmd -m failure"
            ;;
        "full")
            # Run all tests
            ;;
        *)
            log_error "Unknown test category: $category"
            return 1
            ;;
    esac
    
    # Add report generation
    pytest_cmd="$pytest_cmd --html=$REPORTS_DIR/report_${category}_$TIMESTAMP.html --self-contained-html"
    pytest_cmd="$pytest_cmd --junitxml=$REPORTS_DIR/junit_${category}_$TIMESTAMP.xml"
    
    log_info "Executing: $pytest_cmd"
    
    # Run tests
    if eval "$pytest_cmd"; then
        log_success "$category tests passed"
        return 0
    else
        log_error "$category tests failed"
        return 1
    fi
}

# Generate comprehensive report
generate_report() {
    log_info "Generating comprehensive test report..."
    
    cd "$PROJECT_ROOT"
    
    # Use the test runner to generate detailed report
    python "$TEST_DIR/test_runner.py" \
        --category "$MODE" \
        --report "$REPORT_FORMAT" \
        --output "$REPORTS_DIR/coordination_framework_report_$TIMESTAMP.$REPORT_FORMAT" \
        --verbose
    
    log_success "Report generated: $REPORTS_DIR/coordination_framework_report_$TIMESTAMP.$REPORT_FORMAT"
}

# Run performance benchmarks
run_benchmarks() {
    log_info "Running performance benchmarks..."
    
    cd "$PROJECT_ROOT"
    
    python "$TEST_DIR/test_runner.py" --benchmark > "$REPORTS_DIR/benchmarks_$TIMESTAMP.json"
    
    log_success "Benchmarks complete: $REPORTS_DIR/benchmarks_$TIMESTAMP.json"
}

# Cleanup test artifacts
cleanup_artifacts() {
    if [[ "$CLEANUP" == "true" ]]; then
        log_info "Cleaning up test artifacts..."
        
        # Remove temporary files
        find "$PROJECT_ROOT" -name "*.pyc" -delete
        find "$PROJECT_ROOT" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        find "$PROJECT_ROOT" -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
        
        # Clean up test databases/temp files
        rm -f "$PROJECT_ROOT"/test_*.db 2>/dev/null || true
        rm -rf "$PROJECT_ROOT"/test_data_* 2>/dev/null || true
        
        log_success "Cleanup complete"
    fi
}

# Main execution
main() {
    log_info "Starting Agent Coordination Framework Integration Tests"
    log_info "Mode: $MODE, Parallel: $PARALLEL, Verbose: $VERBOSE, Report: $REPORT_FORMAT"
    
    # Track overall success
    overall_success=true
    
    # Setup
    setup_environment || exit 1
    validate_system || exit 1
    
    # Run tests based on mode
    case $MODE in
        "smoke")
            run_smoke_tests || overall_success=false
            ;;
        "benchmarks")
            run_benchmarks || overall_success=false
            ;;
        "full")
            # Run all test categories
            for category in "smoke" "unit" "integration" "e2e" "performance"; do
                if ! run_test_category "$category"; then
                    overall_success=false
                    log_warning "Category $category failed, continuing with other tests..."
                fi
            done
            ;;
        *)
            run_test_category "$MODE" || overall_success=false
            ;;
    esac
    
    # Generate reports
    if [[ "$MODE" != "smoke" ]]; then
        generate_report || log_warning "Report generation failed"
    fi
    
    # Cleanup
    cleanup_artifacts
    
    # Summary
    if [[ "$overall_success" == "true" ]]; then
        log_success "All tests completed successfully!"
        log_info "Reports available in: $REPORTS_DIR"
        exit 0
    else
        log_error "Some tests failed. Check reports for details."
        exit 1
    fi
}

# Trap for cleanup on exit
trap cleanup_artifacts EXIT

# Execute main function
main "$@" 