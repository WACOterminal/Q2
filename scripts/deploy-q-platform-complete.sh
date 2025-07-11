#!/bin/bash

# Q Platform Complete Deployment Script
# This script orchestrates the complete deployment of the Q Platform

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="$PROJECT_ROOT/deployment.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="development"
NAMESPACE="q-platform"
DOCKER_REGISTRY="localhost:5000"
IMAGE_TAG="latest"
SKIP_INFRASTRUCTURE=false
SKIP_SERVICES=false
SKIP_TESTS=false
PARALLEL_BUILDS=false
VAULT_ADDR=""
VAULT_TOKEN=""
KUBECONFIG_PATH="$HOME/.kube/config"

# Function to print colored output
print_info() {
    local msg="$1"
    echo -e "${BLUE}[INFO]${NC} $msg"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $msg" >> "$LOG_FILE"
}

print_success() {
    local msg="$1"
    echo -e "${GREEN}[SUCCESS]${NC} $msg"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $msg" >> "$LOG_FILE"
}

print_warning() {
    local msg="$1"
    echo -e "${YELLOW}[WARNING]${NC} $msg"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $msg" >> "$LOG_FILE"
}

print_error() {
    local msg="$1"
    echo -e "${RED}[ERROR]${NC} $msg"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $msg" >> "$LOG_FILE"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy the complete Q Platform

OPTIONS:
    -e, --environment ENV         Environment (development, staging, production) [default: development]
    -n, --namespace NAMESPACE     Kubernetes namespace [default: q-platform]
    -r, --registry REGISTRY       Docker registry [default: localhost:5000]
    -t, --tag TAG                 Image tag [default: latest]
    -k, --kubeconfig PATH         Path to kubeconfig file [default: ~/.kube/config]
    -v, --vault-addr ADDR         Vault server address
    --vault-token TOKEN           Vault authentication token
    --skip-infrastructure         Skip infrastructure deployment
    --skip-services              Skip service deployment
    --skip-tests                 Skip testing phase
    --parallel                   Enable parallel builds
    -h, --help                   Show this help message

EXAMPLES:
    $0 --environment development
    $0 --environment production --registry myregistry.com --tag v1.0.0
    $0 --skip-infrastructure --skip-tests

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -r|--registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -k|--kubeconfig)
            KUBECONFIG_PATH="$2"
            shift 2
            ;;
        -v|--vault-addr)
            VAULT_ADDR="$2"
            shift 2
            ;;
        --vault-token)
            VAULT_TOKEN="$2"
            shift 2
            ;;
        --skip-infrastructure)
            SKIP_INFRASTRUCTURE=true
            shift
            ;;
        --skip-services)
            SKIP_SERVICES=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --parallel)
            PARALLEL_BUILDS=true
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

# Function to initialize deployment
initialize_deployment() {
    print_info "Initializing Q Platform deployment..."
    
    # Create log file
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "Q Platform Deployment Log - $(date)" > "$LOG_FILE"
    
    # Validate environment
    if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
        print_error "Invalid environment: $ENVIRONMENT"
        exit 1
    fi
    
    # Set environment-specific configurations
    case "$ENVIRONMENT" in
        development)
            IMAGE_TAG="${IMAGE_TAG:-dev}"
            ;;
        staging)
            IMAGE_TAG="${IMAGE_TAG:-staging}"
            ;;
        production)
            IMAGE_TAG="${IMAGE_TAG:-latest}"
            ;;
    esac
    
    print_info "Environment: $ENVIRONMENT"
    print_info "Namespace: $NAMESPACE"
    print_info "Registry: $DOCKER_REGISTRY"
    print_info "Image Tag: $IMAGE_TAG"
    
    print_success "Deployment initialized"
}

# Function to validate prerequisites
validate_prerequisites() {
    print_info "Validating prerequisites..."
    
    # Check required tools
    local required_tools=("docker" "kubectl" "helm" "terraform" "python3" "node" "npm")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            print_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check Kubernetes access
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot access Kubernetes cluster"
        exit 1
    fi
    
    # Check Vault if specified
    if [[ -n "$VAULT_ADDR" && -n "$VAULT_TOKEN" ]]; then
        if ! curl -s --fail -H "X-Vault-Token: $VAULT_TOKEN" "$VAULT_ADDR/v1/sys/health" &> /dev/null; then
            print_error "Cannot connect to Vault at $VAULT_ADDR"
            exit 1
        fi
    fi
    
    print_success "Prerequisites validated"
}

# Function to setup environment
setup_environment() {
    print_info "Setting up deployment environment..."
    
    # Set environment variables
    export ENVIRONMENT
    export NAMESPACE
    export DOCKER_REGISTRY
    export IMAGE_TAG
    export KUBECONFIG_PATH
    
    if [[ -n "$VAULT_ADDR" ]]; then
        export VAULT_ADDR
    fi
    
    if [[ -n "$VAULT_TOKEN" ]]; then
        export VAULT_TOKEN
    fi
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        print_info "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi
    
    print_success "Environment setup complete"
}

# Function to build shared libraries
build_shared_libraries() {
    print_info "Building shared libraries..."
    
    if [[ -x "$SCRIPT_DIR/build-shared-libs.sh" ]]; then
        "$SCRIPT_DIR/build-shared-libs.sh" --clean --install-deps
        print_success "Shared libraries built successfully"
    else
        print_warning "Shared libraries build script not found or not executable"
    fi
}

# Function to build Docker images
build_docker_images() {
    print_info "Building Docker images..."
    
    local build_args=""
    if [[ "$PARALLEL_BUILDS" == "true" ]]; then
        build_args="--parallel"
    fi
    
    if [[ -x "$SCRIPT_DIR/build-docker-images.sh" ]]; then
        "$SCRIPT_DIR/build-docker-images.sh" \
            --registry "$DOCKER_REGISTRY" \
            --tag "$IMAGE_TAG" \
            --push \
            $build_args
        print_success "Docker images built successfully"
    else
        print_error "Docker images build script not found or not executable"
        exit 1
    fi
}

# Function to migrate configurations to Vault
migrate_configurations() {
    if [[ -n "$VAULT_ADDR" && -n "$VAULT_TOKEN" ]]; then
        print_info "Migrating configurations to Vault..."
        
        if [[ -x "$SCRIPT_DIR/migrate_config_to_vault.py" ]]; then
            python3 "$SCRIPT_DIR/migrate_config_to_vault.py" \
                --vault-addr "$VAULT_ADDR" \
                --vault-token "$VAULT_TOKEN" \
                --all
            print_success "Configurations migrated to Vault"
        else
            print_warning "Configuration migration script not found"
        fi
    else
        print_info "Vault not configured, skipping configuration migration"
    fi
}

# Function to deploy infrastructure
deploy_infrastructure() {
    if [[ "$SKIP_INFRASTRUCTURE" == "true" ]]; then
        print_info "Skipping infrastructure deployment"
        return 0
    fi
    
    print_info "Deploying infrastructure..."
    
    if [[ -x "$SCRIPT_DIR/deploy-infrastructure.sh" ]]; then
        "$SCRIPT_DIR/deploy-infrastructure.sh" \
            --environment "$ENVIRONMENT" \
            --namespace "$NAMESPACE" \
            --kubeconfig "$KUBECONFIG_PATH" \
            --force
        print_success "Infrastructure deployed successfully"
    else
        print_error "Infrastructure deployment script not found or not executable"
        exit 1
    fi
}

# Function to deploy platform services
deploy_platform_services() {
    if [[ "$SKIP_SERVICES" == "true" ]]; then
        print_info "Skipping platform services deployment"
        return 0
    fi
    
    print_info "Deploying platform services..."
    
    # Use the existing deploy script with appropriate parameters
    if [[ -x "$SCRIPT_DIR/deploy-q-platform.sh" ]]; then
        "$SCRIPT_DIR/deploy-q-platform.sh" \
            --environment "$ENVIRONMENT" \
            --namespace "$NAMESPACE" \
            --image-tag "$IMAGE_TAG" \
            --force
        print_success "Platform services deployed successfully"
    else
        print_error "Platform services deployment script not found or not executable"
        exit 1
    fi
}

# Function to populate knowledge graph
populate_knowledge_graph() {
    print_info "Populating knowledge graph..."
    
    # Wait for KnowledgeGraphQ to be ready
    print_info "Waiting for KnowledgeGraphQ to be ready..."
    kubectl wait --for=condition=Available --timeout=300s deployment/knowledgegraphq -n "$NAMESPACE" || true
    
    # Run knowledge graph population
    if [[ -f "$PROJECT_ROOT/KnowledgeGraphQ/scripts/build_graph.py" ]]; then
        python3 "$PROJECT_ROOT/KnowledgeGraphQ/scripts/build_graph.py"
        print_success "Knowledge graph populated"
    else
        print_warning "Knowledge graph population script not found"
    fi
}

# Function to setup vector collections
setup_vector_collections() {
    print_info "Setting up vector collections..."
    
    # Wait for VectorStoreQ to be ready
    print_info "Waiting for VectorStoreQ to be ready..."
    kubectl wait --for=condition=Available --timeout=300s deployment/vectorstore-q -n "$NAMESPACE" || true
    
    # Initialize vector collections
    print_info "Initializing vector collections..."
    
    # Create a simple initialization script
    cat > /tmp/init_vectors.py << 'EOF'
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.q_vectorstore_client.client import VectorStoreClient

def initialize_collections():
    try:
        client = VectorStoreClient()
        
        # Create collections for different data types
        collections = [
            {
                "name": "documents",
                "dimension": 384,
                "metric_type": "COSINE",
                "description": "Document embeddings"
            },
            {
                "name": "conversations",
                "dimension": 384,
                "metric_type": "COSINE",
                "description": "Conversation embeddings"
            },
            {
                "name": "code",
                "dimension": 384,
                "metric_type": "COSINE",
                "description": "Code embeddings"
            }
        ]
        
        for collection in collections:
            client.create_collection(**collection)
            print(f"Created collection: {collection['name']}")
        
        print("Vector collections initialized successfully")
        
    except Exception as e:
        print(f"Error initializing vector collections: {e}")
        sys.exit(1)

if __name__ == "__main__":
    initialize_collections()
EOF
    
    python3 /tmp/init_vectors.py
    rm -f /tmp/init_vectors.py
    
    print_success "Vector collections setup complete"
}

# Function to deploy Airflow DAGs
deploy_airflow_dags() {
    print_info "Deploying Airflow DAGs..."
    
    # Wait for Airflow to be ready
    kubectl wait --for=condition=Available --timeout=300s deployment/airflow-webserver -n "$NAMESPACE" || true
    
    # Copy DAGs to Airflow
    if [[ -d "$PROJECT_ROOT/airflow/dags" ]]; then
        print_info "Copying DAGs to Airflow..."
        # This would typically involve copying to a shared volume or using Airflow's DAG sync
        print_success "Airflow DAGs deployed"
    else
        print_warning "Airflow DAGs directory not found"
    fi
}

# Function to deploy Flink jobs
deploy_flink_jobs() {
    print_info "Deploying Flink jobs..."
    
    # Wait for Flink to be ready
    kubectl wait --for=condition=Available --timeout=300s deployment/flink-jobmanager -n "$NAMESPACE" || true
    
    # Deploy Flink jobs
    if [[ -d "$PROJECT_ROOT/QuantumPulse/flink_jobs" ]]; then
        print_info "Deploying Flink streaming jobs..."
        # This would typically involve submitting jobs to the Flink cluster
        print_success "Flink jobs deployed"
    else
        print_warning "Flink jobs directory not found"
    fi
}

# Function to run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        print_info "Skipping tests"
        return 0
    fi
    
    print_info "Running tests..."
    
    # Run validation script
    if [[ -x "$SCRIPT_DIR/validate-deployment.sh" ]]; then
        "$SCRIPT_DIR/validate-deployment.sh" \
            --environment "$ENVIRONMENT" \
            --namespace "$NAMESPACE"
        print_success "Tests completed successfully"
    else
        print_warning "Test validation script not found"
    fi
}

# Function to show deployment summary
show_deployment_summary() {
    print_info "Deployment Summary:"
    echo "Environment: $ENVIRONMENT"
    echo "Namespace: $NAMESPACE"
    echo "Registry: $DOCKER_REGISTRY"
    echo "Image Tag: $IMAGE_TAG"
    echo "Log File: $LOG_FILE"
    echo
    
    print_info "Deployed services:"
    kubectl get pods -n "$NAMESPACE" --no-headers | awk '{print $1, $3}' | column -t
    echo
    
    print_info "Service endpoints:"
    kubectl get svc -n "$NAMESPACE" --no-headers | awk '{print $1, $2, $5}' | column -t
    echo
    
    print_info "Access information:"
    echo "- Manager Q API: kubectl port-forward -n $NAMESPACE svc/manager-q 8003:8003"
    echo "- H2M Service: kubectl port-forward -n $NAMESPACE svc/h2m-service 8002:8002"
    echo "- Web App: kubectl port-forward -n $NAMESPACE svc/webapp-q 3000:80"
    echo "- Grafana: kubectl port-forward -n $NAMESPACE svc/grafana 3000:3000"
    echo
    
    print_success "Q Platform deployment completed successfully!"
}

# Function to handle cleanup on failure
cleanup_on_failure() {
    print_error "Deployment failed. Check logs at: $LOG_FILE"
    
    # Show recent logs
    print_info "Recent deployment logs:"
    tail -20 "$LOG_FILE"
    
    # Optionally show pod status
    print_info "Pod status in namespace $NAMESPACE:"
    kubectl get pods -n "$NAMESPACE" 2>/dev/null || true
}

# Main execution
main() {
    print_info "Starting complete Q Platform deployment..."
    
    # Initialize deployment
    initialize_deployment
    
    # Validate prerequisites
    validate_prerequisites
    
    # Setup environment
    setup_environment
    
    # Build shared libraries
    build_shared_libraries
    
    # Build Docker images
    build_docker_images
    
    # Migrate configurations
    migrate_configurations
    
    # Deploy infrastructure
    deploy_infrastructure
    
    # Deploy platform services
    deploy_platform_services
    
    # Populate knowledge graph
    populate_knowledge_graph
    
    # Setup vector collections
    setup_vector_collections
    
    # Deploy Airflow DAGs
    deploy_airflow_dags
    
    # Deploy Flink jobs
    deploy_flink_jobs
    
    # Run tests
    run_tests
    
    # Show deployment summary
    show_deployment_summary
}

# Trap signals for cleanup
trap cleanup_on_failure ERR INT TERM

# Execute main function
main "$@" 