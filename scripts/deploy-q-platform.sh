#!/bin/bash

# Q Platform Deployment Script
# This script handles the complete deployment of the Q Platform to Kubernetes

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HELM_CHART_PATH="$PROJECT_ROOT/helm/q-platform"
TERRAFORM_PATH="$PROJECT_ROOT/infra/terraform"
KUBECTL_TIMEOUT="600s"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="development"
NAMESPACE="q-platform"
RELEASE_NAME="q-platform"
DRY_RUN=false
SKIP_INFRASTRUCTURE=false
SKIP_PLATFORM=false
FORCE_UPDATE=false
BACKUP_BEFORE_DEPLOY=false
ROLLBACK_ON_FAILURE=false
WAIT_FOR_READY=true
TIMEOUT="20m"
VALUES_FILE=""
IMAGE_TAG="latest"
VAULT_ADDR=""
VAULT_TOKEN=""

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

Deploy the Q Platform to Kubernetes

OPTIONS:
    -e, --environment ENVIRONMENT   Target environment (development, staging, production)
    -n, --namespace NAMESPACE       Kubernetes namespace (default: q-platform)
    -r, --release-name NAME         Helm release name (default: q-platform)
    -v, --values-file FILE          Custom values file path
    -t, --image-tag TAG             Docker image tag (default: latest)
    --vault-addr ADDR               Vault server address
    --vault-token TOKEN             Vault authentication token
    --dry-run                       Perform a dry run without making changes
    --skip-infrastructure           Skip infrastructure deployment
    --skip-platform                 Skip platform deployment
    --force-update                  Force update even if no changes detected
    --backup-before-deploy          Create backup before deployment
    --rollback-on-failure          Rollback on deployment failure
    --no-wait                       Don't wait for deployment to be ready
    --timeout DURATION              Deployment timeout (default: 20m)
    -h, --help                      Show this help message

EXAMPLES:
    # Deploy to development environment
    $0 -e development

    # Deploy to production with custom values
    $0 -e production -v values-prod.yaml --backup-before-deploy

    # Dry run for staging
    $0 -e staging --dry-run

    # Deploy only infrastructure
    $0 -e production --skip-platform

    # Deploy with custom image tag
    $0 -e staging -t v1.2.3

EOF
}

# Function to validate prerequisites
validate_prerequisites() {
    print_info "Validating prerequisites..."
    
    # Check for required tools
    local required_tools=("kubectl" "helm" "terraform" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            print_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check Kubernetes connectivity
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check Helm repositories
    print_info "Adding Helm repositories..."
    helm repo add bitnami https://charts.bitnami.com/bitnami || true
    helm repo add milvus https://milvus-io.github.io/milvus-helm/ || true
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts || true
    helm repo add grafana https://grafana.github.io/helm-charts || true
    helm repo add jaegertracing https://jaegertracing.github.io/helm-charts || true
    helm repo add argo https://argoproj.github.io/argo-helm || true
    helm repo add hashicorp https://helm.releases.hashicorp.com || true
    helm repo add external-secrets https://charts.external-secrets.io || true
    helm repo update
    
    print_success "Prerequisites validated"
}

# Function to validate environment
validate_environment() {
    print_info "Validating environment configuration..."
    
    # Check if environment values file exists
    if [ -z "$VALUES_FILE" ]; then
        VALUES_FILE="$HELM_CHART_PATH/values-$ENVIRONMENT.yaml"
    fi
    
    if [ ! -f "$VALUES_FILE" ]; then
        print_error "Values file not found: $VALUES_FILE"
        exit 1
    fi
    
    # Validate Helm chart
    helm lint "$HELM_CHART_PATH" --values "$VALUES_FILE"
    
    # Validate Terraform configuration
    if [ "$SKIP_INFRASTRUCTURE" = false ]; then
        print_info "Validating Terraform configuration..."
        cd "$TERRAFORM_PATH"
        terraform init -backend=false
        terraform validate
        cd - > /dev/null
    fi
    
    print_success "Environment configuration validated"
}

# Function to create namespace if it doesn't exist
ensure_namespace() {
    print_info "Ensuring namespace '$NAMESPACE' exists..."
    
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        if [ "$DRY_RUN" = true ]; then
            print_info "Would create namespace: $NAMESPACE"
        else
            kubectl create namespace "$NAMESPACE"
            kubectl label namespace "$NAMESPACE" app.kubernetes.io/managed-by=q-platform
            print_success "Created namespace: $NAMESPACE"
        fi
    else
        print_info "Namespace '$NAMESPACE' already exists"
    fi
}

# Function to backup current deployment
backup_deployment() {
    if [ "$BACKUP_BEFORE_DEPLOY" = true ]; then
        print_info "Creating backup of current deployment..."
        
        local backup_dir="$PROJECT_ROOT/backups/$(date +%Y%m%d-%H%M%S)"
        mkdir -p "$backup_dir"
        
        # Backup Helm values
        if helm list -n "$NAMESPACE" | grep -q "$RELEASE_NAME"; then
            helm get values "$RELEASE_NAME" -n "$NAMESPACE" > "$backup_dir/helm-values.yaml"
            helm get manifest "$RELEASE_NAME" -n "$NAMESPACE" > "$backup_dir/helm-manifest.yaml"
        fi
        
        # Backup Kubernetes resources
        kubectl get all -n "$NAMESPACE" -o yaml > "$backup_dir/kubernetes-resources.yaml"
        
        # Backup persistent volumes
        kubectl get pv -o yaml > "$backup_dir/persistent-volumes.yaml"
        
        # Backup secrets (metadata only, not data)
        kubectl get secrets -n "$NAMESPACE" -o yaml | \
            sed 's/data:/# data:/g' | \
            sed 's/  [^#].*: .*/#  [REDACTED]/g' > "$backup_dir/secrets-metadata.yaml"
        
        print_success "Backup created at: $backup_dir"
        echo "$backup_dir" > "$PROJECT_ROOT/.last-backup"
    fi
}

# Function to deploy infrastructure
deploy_infrastructure() {
    if [ "$SKIP_INFRASTRUCTURE" = true ]; then
        print_info "Skipping infrastructure deployment"
        return 0
    fi
    
    print_info "Deploying infrastructure with Terraform..."
    
    cd "$TERRAFORM_PATH"
    
    # Initialize Terraform
    terraform init
    
    # Select or create workspace
    if terraform workspace list | grep -q "$ENVIRONMENT"; then
        terraform workspace select "$ENVIRONMENT"
    else
        terraform workspace new "$ENVIRONMENT"
    fi
    
    # Plan deployment
    local plan_file="$ENVIRONMENT.tfplan"
    terraform plan -var-file="environments/$ENVIRONMENT.tfvars" -out="$plan_file"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "Terraform plan created (dry run mode)"
        cd - > /dev/null
        return 0
    fi
    
    # Apply deployment
    terraform apply -auto-approve "$plan_file"
    
    # Clean up plan file
    rm -f "$plan_file"
    
    cd - > /dev/null
    print_success "Infrastructure deployed successfully"
}

# Function to deploy platform
deploy_platform() {
    if [ "$SKIP_PLATFORM" = true ]; then
        print_info "Skipping platform deployment"
        return 0
    fi
    
    print_info "Deploying Q Platform with Helm..."
    
    local helm_args=(
        "upgrade" "--install" "$RELEASE_NAME" "$HELM_CHART_PATH"
        "--namespace" "$NAMESPACE"
        "--values" "$VALUES_FILE"
        "--set" "global.imageTag=$IMAGE_TAG"
        "--set" "global.environment=$ENVIRONMENT"
        "--timeout" "$TIMEOUT"
    )
    
    if [ "$WAIT_FOR_READY" = true ]; then
        helm_args+=("--wait")
    fi
    
    if [ "$DRY_RUN" = true ]; then
        helm_args+=("--dry-run")
    fi
    
    if [ "$FORCE_UPDATE" = true ]; then
        helm_args+=("--force")
    fi
    
    # Add Vault configuration if provided
    if [ -n "$VAULT_ADDR" ] && [ -n "$VAULT_TOKEN" ]; then
        helm_args+=("--set" "security.vault.server.address=$VAULT_ADDR")
        helm_args+=("--set-string" "security.vault.token=$VAULT_TOKEN")
    fi
    
    # Execute Helm deployment
    if helm "${helm_args[@]}"; then
        print_success "Platform deployed successfully"
    else
        print_error "Platform deployment failed"
        if [ "$ROLLBACK_ON_FAILURE" = true ]; then
            rollback_deployment
        fi
        exit 1
    fi
}

# Function to verify deployment
verify_deployment() {
    print_info "Verifying deployment..."
    
    # Check if all deployments are ready
    local deployments=(
        "manager-q"
        "agent-q-default"
        "h2m-service"
        "quantum-pulse-api"
        "vector-store-q"
        "knowledge-graph-q"
        "integration-hub"
        "user-profile-q"
        "webapp-q"
    )
    
    for deployment in "${deployments[@]}"; do
        if kubectl get deployment "$deployment" -n "$NAMESPACE" &> /dev/null; then
            print_info "Checking deployment: $deployment"
            if ! kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"; then
                print_error "Deployment $deployment failed to become ready"
                return 1
            fi
        else
            print_warning "Deployment $deployment not found (may be disabled)"
        fi
    done
    
    # Check pod status
    print_info "Checking pod status..."
    kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=q-platform
    
    # Check service status
    print_info "Checking service status..."
    kubectl get services -n "$NAMESPACE"
    
    # Run basic health checks
    print_info "Running health checks..."
    local health_check_script="$PROJECT_ROOT/scripts/health-check.sh"
    if [ -f "$health_check_script" ]; then
        bash "$health_check_script" "$ENVIRONMENT" "$NAMESPACE"
    fi
    
    print_success "Deployment verification completed"
}

# Function to rollback deployment
rollback_deployment() {
    print_warning "Rolling back deployment..."
    
    # Rollback Helm release
    if helm list -n "$NAMESPACE" | grep -q "$RELEASE_NAME"; then
        helm rollback "$RELEASE_NAME" -n "$NAMESPACE"
        print_info "Helm release rolled back"
    fi
    
    # Restore from backup if available
    if [ -f "$PROJECT_ROOT/.last-backup" ]; then
        local backup_dir
        backup_dir=$(cat "$PROJECT_ROOT/.last-backup")
        if [ -d "$backup_dir" ]; then
            print_info "Restoring from backup: $backup_dir"
            # Add restore logic here if needed
        fi
    fi
    
    print_warning "Rollback completed"
}

# Function to run smoke tests
run_smoke_tests() {
    print_info "Running smoke tests..."
    
    local smoke_test_script="$PROJECT_ROOT/scripts/smoke-tests.sh"
    if [ -f "$smoke_test_script" ]; then
        bash "$smoke_test_script" "$ENVIRONMENT" "$NAMESPACE"
    else
        print_warning "Smoke test script not found, skipping"
    fi
}

# Function to show deployment summary
show_deployment_summary() {
    print_success "Deployment Summary"
    echo "===================="
    echo "Environment: $ENVIRONMENT"
    echo "Namespace: $NAMESPACE"
    echo "Release: $RELEASE_NAME"
    echo "Image Tag: $IMAGE_TAG"
    echo "Values File: $VALUES_FILE"
    echo "Dry Run: $DRY_RUN"
    echo "===================="
    
    if [ "$DRY_RUN" = false ]; then
        echo ""
        print_info "Deployment completed successfully!"
        echo ""
        echo "Access URLs:"
        echo "  API: https://api.$ENVIRONMENT.q-platform.your-domain.com"
        echo "  App: https://app.$ENVIRONMENT.q-platform.your-domain.com"
        echo "  Grafana: https://grafana.$ENVIRONMENT.q-platform.your-domain.com"
        echo "  ArgoCD: https://argocd.$ENVIRONMENT.q-platform.your-domain.com"
        echo ""
        echo "Next steps:"
        echo "  1. Run smoke tests: ./scripts/smoke-tests.sh $ENVIRONMENT"
        echo "  2. Monitor deployment: kubectl get pods -n $NAMESPACE -w"
        echo "  3. Check logs: kubectl logs -f deployment/manager-q -n $NAMESPACE"
    fi
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
        -r|--release-name)
            RELEASE_NAME="$2"
            shift 2
            ;;
        -v|--values-file)
            VALUES_FILE="$2"
            shift 2
            ;;
        -t|--image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --vault-addr)
            VAULT_ADDR="$2"
            shift 2
            ;;
        --vault-token)
            VAULT_TOKEN="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-infrastructure)
            SKIP_INFRASTRUCTURE=true
            shift
            ;;
        --skip-platform)
            SKIP_PLATFORM=true
            shift
            ;;
        --force-update)
            FORCE_UPDATE=true
            shift
            ;;
        --backup-before-deploy)
            BACKUP_BEFORE_DEPLOY=true
            shift
            ;;
        --rollback-on-failure)
            ROLLBACK_ON_FAILURE=true
            shift
            ;;
        --no-wait)
            WAIT_FOR_READY=false
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
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

# Validate environment parameter
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    print_error "Invalid environment. Must be one of: development, staging, production"
    exit 1
fi

# Main deployment flow
main() {
    print_info "Starting Q Platform deployment..."
    print_info "Environment: $ENVIRONMENT"
    print_info "Namespace: $NAMESPACE"
    print_info "Release: $RELEASE_NAME"
    print_info "Image Tag: $IMAGE_TAG"
    print_info "Dry Run: $DRY_RUN"
    
    # Pre-deployment steps
    validate_prerequisites
    validate_environment
    ensure_namespace
    backup_deployment
    
    # Deployment steps
    deploy_infrastructure
    deploy_platform
    
    # Post-deployment steps
    if [ "$DRY_RUN" = false ]; then
        verify_deployment
        run_smoke_tests
    fi
    
    show_deployment_summary
}

# Trap signals for cleanup
trap 'print_error "Deployment interrupted"; exit 1' INT TERM

# Execute main function
main "$@" 