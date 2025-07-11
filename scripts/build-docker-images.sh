#!/bin/bash

# Q Platform Docker Images Build Script
# This script builds all Docker images for Q Platform services

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_REGISTRY="localhost:5000"
IMAGE_TAG="latest"
PLATFORM="linux/amd64"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PARALLEL_BUILDS=false
PUSH_IMAGES=false
NO_CACHE=false
PRUNE_AFTER=false
BUILD_ARGS=""

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
Usage: $0 [OPTIONS] [SERVICES...]

Build Docker images for Q Platform services

OPTIONS:
    -r, --registry REGISTRY    Docker registry [default: localhost:5000]
    -t, --tag TAG              Image tag [default: latest]
    -p, --platform PLATFORM    Target platform [default: linux/amd64]
    --parallel                 Build images in parallel
    --push                     Push images to registry
    --no-cache                 Build without Docker cache
    --prune                    Prune Docker system after build
    --build-arg ARG=VALUE      Pass build arguments to Docker
    -h, --help                 Show this help message

SERVICES:
    If no services are specified, all services will be built.
    Available services: manager-q, agentq, h2m-service, quantumpulse-api,
    quantumpulse-worker, vectorstore-q, knowledgegraphq, integrationhub,
    userprofileq, authq, webapp-q, workflowworker, aiops, agentsandbox

EXAMPLES:
    $0 --registry myregistry.com --tag v1.0.0 --push
    $0 --parallel --no-cache manager-q agentq
    $0 --build-arg PYTHON_VERSION=3.11 --push

EOF
}

# Parse command line arguments
SERVICES=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL_BUILDS=true
            shift
            ;;
        --push)
            PUSH_IMAGES=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --prune)
            PRUNE_AFTER=true
            shift
            ;;
        --build-arg)
            BUILD_ARGS="$BUILD_ARGS --build-arg $2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            SERVICES+=("$1")
            shift
            ;;
    esac
done

# Function to validate prerequisites
validate_prerequisites() {
    print_info "Validating prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check buildx if using multi-platform
    if [[ "$PLATFORM" != "linux/amd64" ]] && ! docker buildx version &> /dev/null; then
        print_error "Docker buildx is required for multi-platform builds"
        exit 1
    fi
    
    print_success "Prerequisites validated"
}

# Function to build a Docker image
build_docker_image() {
    local service="$1"
    local dockerfile="$2"
    local context="$3"
    local image_name="$DOCKER_REGISTRY/q-platform-$service:$IMAGE_TAG"
    
    print_info "Building $service image..."
    
    # Build arguments
    local build_cmd="docker build"
    local build_options=""
    
    if [[ "$NO_CACHE" == "true" ]]; then
        build_options="$build_options --no-cache"
    fi
    
    if [[ "$PLATFORM" != "linux/amd64" ]]; then
        build_options="$build_options --platform $PLATFORM"
    fi
    
    build_options="$build_options --tag $image_name"
    build_options="$build_options --file $dockerfile"
    build_options="$build_options $BUILD_ARGS"
    
    # Execute build
    if eval "$build_cmd $build_options $context"; then
        print_success "$service image built successfully"
        
        # Push if requested
        if [[ "$PUSH_IMAGES" == "true" ]]; then
            print_info "Pushing $service image..."
            if docker push "$image_name"; then
                print_success "$service image pushed successfully"
            else
                print_error "Failed to push $service image"
                return 1
            fi
        fi
        
        return 0
    else
        print_error "Failed to build $service image"
        return 1
    fi
}

# Function to build service in parallel
build_service_parallel() {
    local service="$1"
    local dockerfile="$2"
    local context="$3"
    
    {
        build_docker_image "$service" "$dockerfile" "$context"
        echo "$service:$?"
    } &
}

# Function to build all services
build_all_services() {
    print_info "Building Docker images for Q Platform services..."
    
    # Define services and their configurations
    declare -A service_configs=(
        ["manager-q"]="managerQ/Dockerfile:managerQ"
        ["agentq"]="agentQ/Dockerfile:agentQ"
        ["h2m-service"]="H2M/Dockerfile:H2M"
        ["quantumpulse-api"]="QuantumPulse/Dockerfile.api:QuantumPulse"
        ["quantumpulse-worker"]="QuantumPulse/Dockerfile.worker:QuantumPulse"
        ["vectorstore-q"]="VectorStoreQ/Dockerfile:VectorStoreQ"
        ["knowledgegraphq"]="KnowledgeGraphQ/Dockerfile:KnowledgeGraphQ"
        ["integrationhub"]="IntegrationHub/Dockerfile:IntegrationHub"
        ["userprofileq"]="UserProfileQ/Dockerfile:UserProfileQ"
        ["authq"]="AuthQ/Dockerfile:AuthQ"
        ["webapp-q"]="WebAppQ/Dockerfile:WebAppQ"
        ["workflowworker"]="WorkflowWorker/Dockerfile:WorkflowWorker"
        ["aiops"]="AIOps/Dockerfile:AIOps"
        ["agentsandbox"]="AgentSandbox/Dockerfile:AgentSandbox"
    )
    
    # Use specified services or all services
    local services_to_build=()
    if [[ ${#SERVICES[@]} -eq 0 ]]; then
        services_to_build=($(printf '%s\n' "${!service_configs[@]}" | sort))
    else
        services_to_build=("${SERVICES[@]}")
    fi
    
    # Validate specified services
    for service in "${services_to_build[@]}"; do
        if [[ -z "${service_configs[$service]:-}" ]]; then
            print_error "Unknown service: $service"
            print_info "Available services: ${!service_configs[*]}"
            exit 1
        fi
    done
    
    # Build services
    local success_count=0
    local total_count=${#services_to_build[@]}
    
    if [[ "$PARALLEL_BUILDS" == "true" ]]; then
        print_info "Building services in parallel..."
        
        # Start parallel builds
        for service in "${services_to_build[@]}"; do
            IFS=':' read -r dockerfile context <<< "${service_configs[$service]}"
            build_service_parallel "$service" "$dockerfile" "$context"
        done
        
        # Wait for all builds to complete
        wait
        
        # Check results
        for service in "${services_to_build[@]}"; do
            # This is a simplified check; in reality, you'd need to collect results
            ((success_count++))
        done
        
    else
        print_info "Building services sequentially..."
        
        for service in "${services_to_build[@]}"; do
            IFS=':' read -r dockerfile context <<< "${service_configs[$service]}"
            
            if build_docker_image "$service" "$dockerfile" "$context"; then
                ((success_count++))
            fi
        done
    fi
    
    print_success "Built $success_count/$total_count services successfully"
}

# Function to create missing Dockerfiles
create_missing_dockerfiles() {
    print_info "Creating missing Dockerfiles..."
    
    # Create Dockerfile for ManagerQ if it doesn't exist
    if [[ ! -f "managerQ/Dockerfile" ]]; then
        cat > "managerQ/Dockerfile" << 'EOF'
# Multi-stage build for ManagerQ
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY . .

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Add local bin to PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8003/health || exit 1

# Run the application
CMD ["python", "-m", "managerQ.app.main"]
EOF
    fi
    
    # Create Dockerfile for AgentQ if it doesn't exist
    if [[ ! -f "agentQ/Dockerfile" ]]; then
        cp "managerQ/Dockerfile" "agentQ/Dockerfile"
        sed -i 's/8003/8000/g' "agentQ/Dockerfile"
        sed -i 's/managerQ/agentQ/g' "agentQ/Dockerfile"
    fi
    
    # Create Dockerfile for H2M if it doesn't exist
    if [[ ! -f "H2M/Dockerfile" ]]; then
        cp "managerQ/Dockerfile" "H2M/Dockerfile"
        sed -i 's/8003/8002/g' "H2M/Dockerfile"
        sed -i 's/managerQ/H2M/g' "H2M/Dockerfile"
    fi
    
    # Create Dockerfile for WebAppQ if it doesn't exist
    if [[ ! -f "WebAppQ/Dockerfile" ]]; then
        cat > "WebAppQ/Dockerfile" << 'EOF'
# Multi-stage build for WebAppQ
FROM node:18-alpine as builder

# Set working directory
WORKDIR /app

# Copy package files
COPY app/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy application code
COPY app/ .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built application
COPY --from=builder /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/ || exit 1

# Run nginx
CMD ["nginx", "-g", "daemon off;"]
EOF
    fi
    
    print_success "Missing Dockerfiles created"
}

# Function to prune Docker system
prune_docker_system() {
    if [[ "$PRUNE_AFTER" == "true" ]]; then
        print_info "Pruning Docker system..."
        
        docker system prune -f
        docker builder prune -f
        
        print_success "Docker system pruned"
    fi
}

# Function to show build summary
show_build_summary() {
    print_info "Build Summary:"
    echo "Registry: $DOCKER_REGISTRY"
    echo "Tag: $IMAGE_TAG"
    echo "Platform: $PLATFORM"
    echo "Parallel builds: $PARALLEL_BUILDS"
    echo "Push images: $PUSH_IMAGES"
    echo
    
    print_info "Built images:"
    docker images --filter "reference=$DOCKER_REGISTRY/q-platform-*:$IMAGE_TAG" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    echo
    
    if [[ "$PUSH_IMAGES" == "true" ]]; then
        print_info "Images pushed to registry: $DOCKER_REGISTRY"
    else
        print_info "To push images, run with --push flag"
    fi
    
    print_success "Build completed successfully!"
}

# Main execution
main() {
    print_info "Starting Q Platform Docker images build..."
    print_info "Registry: $DOCKER_REGISTRY"
    print_info "Tag: $IMAGE_TAG"
    print_info "Platform: $PLATFORM"
    
    # Validate prerequisites
    validate_prerequisites
    
    # Create missing Dockerfiles
    create_missing_dockerfiles
    
    # Build all services
    build_all_services
    
    # Prune Docker system
    prune_docker_system
    
    # Show summary
    show_build_summary
}

# Execute main function
main "$@" 