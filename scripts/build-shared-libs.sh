#!/bin/bash

# Q Platform Shared Libraries Build Script
# This script builds and packages all shared libraries used by Q Platform services

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SHARED_DIR="$PROJECT_ROOT/shared"
BUILD_DIR="$PROJECT_ROOT/build"
DIST_DIR="$PROJECT_ROOT/dist"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PYTHON_VERSION="3.11"
VIRTUAL_ENV=""
CLEAN=false
INSTALL_DEPS=false
PACKAGE_ONLY=false
UPLOAD_PYPI=false

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build Q Platform shared libraries

OPTIONS:
    --python-version VERSION   Python version to use [default: 3.11]
    --venv PATH                Path to virtual environment
    --clean                    Clean build artifacts before building
    --install-deps             Install build dependencies
    --package-only             Only package, don't build
    --upload-pypi              Upload to PyPI after building
    -h, --help                 Show this help message

EXAMPLES:
    $0 --clean --install-deps
    $0 --venv /path/to/venv --package-only
    $0 --upload-pypi

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --venv)
            VIRTUAL_ENV="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --install-deps)
            INSTALL_DEPS=true
            shift
            ;;
        --package-only)
            PACKAGE_ONLY=true
            shift
            ;;
        --upload-pypi)
            UPLOAD_PYPI=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Function to validate prerequisites
validate_prerequisites() {
    print_info "Validating prerequisites..."
    
    # Check Python version
    if ! python3 --version | grep -q "$PYTHON_VERSION"; then
        print_warning "Python $PYTHON_VERSION not found, using default Python 3"
    fi
    
    # Check required tools
    local required_tools=("python3" "pip" "setuptools" "wheel" "twine")
    for tool in "${required_tools[@]}"; do
        if ! command -v $tool &> /dev/null; then
            print_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done
    
    print_success "Prerequisites validated"
}

# Function to setup virtual environment
setup_venv() {
    if [[ -n "$VIRTUAL_ENV" ]]; then
        print_info "Using virtual environment: $VIRTUAL_ENV"
        source "$VIRTUAL_ENV/bin/activate"
    else
        print_info "Setting up virtual environment..."
        python3 -m venv "$BUILD_DIR/venv"
        source "$BUILD_DIR/venv/bin/activate"
        VIRTUAL_ENV="$BUILD_DIR/venv"
        print_success "Virtual environment created at $VIRTUAL_ENV"
    fi
}

# Function to install dependencies
install_dependencies() {
    if [[ "$INSTALL_DEPS" == "true" ]]; then
        print_info "Installing build dependencies..."
        
        pip install --upgrade pip
        pip install --upgrade setuptools wheel twine build
        pip install --upgrade -r "$PROJECT_ROOT/constraints.txt"
        
        print_success "Build dependencies installed"
    fi
}

# Function to clean build artifacts
clean_build() {
    if [[ "$CLEAN" == "true" ]]; then
        print_info "Cleaning build artifacts..."
        
        # Remove build directories
        rm -rf "$BUILD_DIR" "$DIST_DIR"
        
        # Clean Python cache
        find "$SHARED_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find "$SHARED_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
        find "$SHARED_DIR" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
        
        print_success "Build artifacts cleaned"
    fi
}

# Function to create directories
create_directories() {
    print_info "Creating build directories..."
    
    mkdir -p "$BUILD_DIR" "$DIST_DIR"
    
    print_success "Build directories created"
}

# Function to build a shared library
build_shared_library() {
    local lib_name="$1"
    local lib_dir="$SHARED_DIR/$lib_name"
    
    if [[ ! -d "$lib_dir" ]]; then
        print_warning "Library directory $lib_dir does not exist, skipping..."
        return 0
    fi
    
    print_info "Building $lib_name..."
    
    # Create setup.py if it doesn't exist
    if [[ ! -f "$lib_dir/setup.py" ]]; then
        print_info "Creating setup.py for $lib_name..."
        
        cat > "$lib_dir/setup.py" << EOF
#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="q_platform_${lib_name}",
    version="1.0.0",
    description="Q Platform ${lib_name} shared library",
    author="Q Platform Team",
    author_email="team@q-platform.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2.0.0",
        "httpx>=0.24.0",
        "structlog>=22.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
EOF
    fi
    
    # Create __init__.py if it doesn't exist
    if [[ ! -f "$lib_dir/__init__.py" ]]; then
        echo "# Q Platform $lib_name shared library" > "$lib_dir/__init__.py"
    fi
    
    # Build the library
    cd "$lib_dir"
    
    if [[ "$PACKAGE_ONLY" != "true" ]]; then
        python setup.py build
    fi
    
    # Create source distribution
    python setup.py sdist bdist_wheel
    
    # Copy to dist directory
    cp dist/* "$DIST_DIR/"
    
    print_success "$lib_name built successfully"
}

# Function to build all shared libraries
build_all_libraries() {
    print_info "Building all shared libraries..."
    
    # List of shared libraries to build
    local libraries=(
        "pulsar_client"
        "opentelemetry"
        "q_auth_parser"
        "q_collaboration_schemas"
        "q_feedback_schemas"
        "q_h2m_client"
        "q_knowledgegraph_client"
        "q_memory_schemas"
        "q_messaging_schemas"
        "q_pulse_client"
        "q_ui_schemas"
        "q_vectorstore_client"
        "observability"
        "q_agentsandbox_client"
    )
    
    local success_count=0
    for lib in "${libraries[@]}"; do
        if build_shared_library "$lib"; then
            ((success_count++))
        fi
    done
    
    print_success "Built $success_count/${#libraries[@]} shared libraries"
}

# Function to create requirements files
create_requirements() {
    print_info "Creating requirements files..."
    
    # Create requirements.txt for all shared libraries
    cat > "$DIST_DIR/requirements.txt" << EOF
# Q Platform Shared Libraries Requirements
# Generated automatically by build script

# Core dependencies
pydantic>=2.0.0
httpx>=0.24.0
structlog>=22.0.0
pulsar-client>=3.0.0
fastavro>=1.7.0

# OpenTelemetry
opentelemetry-api>=1.15.0
opentelemetry-sdk>=1.15.0
opentelemetry-exporter-otlp-proto-grpc>=1.15.0
opentelemetry-instrumentation>=0.36b0

# Database clients
pymilvus>=2.3.0
gremlinpython>=3.6.0
elasticsearch>=8.0.0
pyignite>=0.5.0

# Security
python-jose>=3.3.0
cryptography>=3.4.8

# Utilities
pyyaml>=6.0
python-dotenv>=1.0.0
hvac>=1.0.0
EOF
    
    # Create individual library requirements
    for lib_dir in "$SHARED_DIR"/*; do
        if [[ -d "$lib_dir" ]]; then
            lib_name=$(basename "$lib_dir")
            if [[ -f "$lib_dir/requirements.txt" ]]; then
                cp "$lib_dir/requirements.txt" "$DIST_DIR/requirements-$lib_name.txt"
            fi
        fi
    done
    
    print_success "Requirements files created"
}

# Function to upload to PyPI
upload_to_pypi() {
    if [[ "$UPLOAD_PYPI" == "true" ]]; then
        print_info "Uploading to PyPI..."
        
        if [[ -z "${TWINE_USERNAME:-}" ]] || [[ -z "${TWINE_PASSWORD:-}" ]]; then
            print_error "TWINE_USERNAME and TWINE_PASSWORD environment variables must be set"
            exit 1
        fi
        
        twine upload "$DIST_DIR"/*.tar.gz "$DIST_DIR"/*.whl
        
        print_success "Uploaded to PyPI"
    fi
}

# Function to create installation script
create_install_script() {
    print_info "Creating installation script..."
    
    cat > "$DIST_DIR/install-shared-libs.sh" << 'EOF'
#!/bin/bash

# Q Platform Shared Libraries Installation Script

set -euo pipefail

DIST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing Q Platform shared libraries..."

# Install all wheel files
for wheel in "$DIST_DIR"/*.whl; do
    if [[ -f "$wheel" ]]; then
        echo "Installing $(basename "$wheel")..."
        pip install "$wheel"
    fi
done

echo "Q Platform shared libraries installed successfully!"
EOF
    
    chmod +x "$DIST_DIR/install-shared-libs.sh"
    
    print_success "Installation script created"
}

# Function to show build summary
show_build_summary() {
    print_info "Build Summary:"
    echo "Build directory: $BUILD_DIR"
    echo "Distribution directory: $DIST_DIR"
    echo "Virtual environment: $VIRTUAL_ENV"
    echo
    
    print_info "Generated files:"
    ls -la "$DIST_DIR"
    echo
    
    print_info "Installation:"
    echo "Run: $DIST_DIR/install-shared-libs.sh"
    echo "Or: pip install $DIST_DIR/*.whl"
    echo
    
    print_success "Build completed successfully!"
}

# Main execution
main() {
    print_info "Starting Q Platform shared libraries build..."
    
    # Validate prerequisites
    validate_prerequisites
    
    # Clean build artifacts
    clean_build
    
    # Create directories
    create_directories
    
    # Setup virtual environment
    setup_venv
    
    # Install dependencies
    install_dependencies
    
    # Build all libraries
    build_all_libraries
    
    # Create requirements files
    create_requirements
    
    # Create installation script
    create_install_script
    
    # Upload to PyPI
    upload_to_pypi
    
    # Show summary
    show_build_summary
}

# Execute main function
main "$@" 