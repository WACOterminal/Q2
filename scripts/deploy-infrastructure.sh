#!/bin/bash

# Q Platform Infrastructure Deployment Script
# This script deploys the complete infrastructure stack using Terraform

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TERRAFORM_DIR="$PROJECT_ROOT/infra/terraform"
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
DRY_RUN=false
DESTROY=false
FORCE=false
VAULT_ADDR=""
VAULT_TOKEN=""
KUBECONFIG_PATH="$HOME/.kube/config"

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

Deploy Q Platform infrastructure using Terraform

OPTIONS:
    -e, --environment ENV      Environment (development, staging, production) [default: development]
    -n, --namespace NAMESPACE  Kubernetes namespace [default: q-platform]
    -k, --kubeconfig PATH      Path to kubeconfig file [default: ~/.kube/config]
    -v, --vault-addr ADDR      Vault server address
    -t, --vault-token TOKEN    Vault authentication token
    --dry-run                  Show what would be deployed without making changes
    --destroy                  Destroy infrastructure instead of deploying
    --force                    Force deployment without confirmation
    -h, --help                 Show this help message

EXAMPLES:
    $0 --environment development
    $0 --environment production --force
    $0 --destroy --environment staging
    $0 --dry-run --environment production

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
        -k|--kubeconfig)
            KUBECONFIG_PATH="$2"
            shift 2
            ;;
        -v|--vault-addr)
            VAULT_ADDR="$2"
            shift 2
            ;;
        -t|--vault-token)
            VAULT_TOKEN="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --destroy)
            DESTROY=true
            shift
            ;;
        --force)
            FORCE=true
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
    
    # Check required tools
    local required_tools=("terraform" "kubectl" "helm")
    for tool in "${required_tools[@]}"; do
        if ! command -v $tool &> /dev/null; then
            print_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done
    
    # Check Terraform version
    local terraform_version=$(terraform version -json | jq -r '.terraform_version' 2>/dev/null || echo "unknown")
    print_info "Terraform version: $terraform_version"
    
    # Check kubectl access
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot access Kubernetes cluster. Check your kubeconfig."
        exit 1
    fi
    
    # Check kubeconfig file
    if [[ ! -f "$KUBECONFIG_PATH" ]]; then
        print_error "Kubeconfig file not found: $KUBECONFIG_PATH"
        exit 1
    fi
    
    # Check Vault connectivity if specified
    if [[ -n "$VAULT_ADDR" ]]; then
        if [[ -z "$VAULT_TOKEN" ]]; then
            print_error "Vault token is required when Vault address is specified"
            exit 1
        fi
        
        if ! curl -s --fail -H "X-Vault-Token: $VAULT_TOKEN" "$VAULT_ADDR/v1/sys/health" &> /dev/null; then
            print_error "Cannot connect to Vault at $VAULT_ADDR"
            exit 1
        fi
        print_info "Vault connectivity verified"
    fi
    
    print_success "Prerequisites validated"
}

# Function to initialize Terraform
init_terraform() {
    print_info "Initializing Terraform..."
    
    cd "$TERRAFORM_DIR"
    
    # Initialize Terraform
    terraform init -upgrade
    
    # Validate configuration
    terraform validate
    
    print_success "Terraform initialized"
}

# Function to plan deployment
plan_deployment() {
    print_info "Planning infrastructure deployment..."
    
    cd "$TERRAFORM_DIR"
    
    # Create variables file
    cat > terraform.tfvars << EOF
kubeconfig_path = "$KUBECONFIG_PATH"
namespace = "$NAMESPACE"
environment = "$ENVIRONMENT"
EOF
    
    # Add Vault configuration if provided
    if [[ -n "$VAULT_ADDR" ]]; then
        echo "vault_address = \"$VAULT_ADDR\"" >> terraform.tfvars
    fi
    
    # Plan the deployment
    if [[ "$DESTROY" == "true" ]]; then
        terraform plan -destroy -var-file="terraform.tfvars" -out=tfplan
    else
        terraform plan -var-file="terraform.tfvars" -out=tfplan
    fi
    
    print_success "Deployment plan created"
}

# Function to apply deployment
apply_deployment() {
    print_info "Applying infrastructure deployment..."
    
    cd "$TERRAFORM_DIR"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "DRY RUN: Would apply the following plan:"
        terraform show tfplan
        return 0
    fi
    
    # Confirmation prompt
    if [[ "$FORCE" != "true" ]]; then
        echo
        print_warning "This will deploy/modify infrastructure in the '$ENVIRONMENT' environment"
        print_warning "Namespace: $NAMESPACE"
        print_warning "Kubeconfig: $KUBECONFIG_PATH"
        echo
        read -p "Are you sure you want to proceed? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    # Apply the plan
    terraform apply tfplan
    
    print_success "Infrastructure deployment completed"
}

# Function to wait for services to be ready
wait_for_services() {
    print_info "Waiting for services to be ready..."
    
    local services=(
        "vault/vault"
        "$NAMESPACE/keycloak"
        "$NAMESPACE/pulsar-broker"
        "$NAMESPACE/cassandra"
        "$NAMESPACE/elasticsearch"
        "$NAMESPACE/milvus"
        "$NAMESPACE/ignite"
        "$NAMESPACE/prometheus"
        "$NAMESPACE/grafana"
    )
    
    for service in "${services[@]}"; do
        local namespace=$(echo "$service" | cut -d'/' -f1)
        local service_name=$(echo "$service" | cut -d'/' -f2)
        
        print_info "Waiting for $service_name in namespace $namespace..."
        
        # Wait for deployment to be ready
        if kubectl get deployment "$service_name" -n "$namespace" &> /dev/null; then
            kubectl wait --for=condition=Available --timeout=600s deployment/"$service_name" -n "$namespace"
        elif kubectl get statefulset "$service_name" -n "$namespace" &> /dev/null; then
            kubectl wait --for=condition=Ready --timeout=600s statefulset/"$service_name" -n "$namespace"
        else
            print_warning "Service $service_name not found or not a deployment/statefulset"
        fi
    done
    
    print_success "All services are ready"
}

# Function to verify deployment
verify_deployment() {
    print_info "Verifying deployment..."
    
    # Check namespace
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        print_error "Namespace $NAMESPACE does not exist"
        exit 1
    fi
    
    # Check critical services
    local critical_services=("vault" "keycloak" "pulsar-broker" "cassandra" "elasticsearch" "milvus")
    for service in "${critical_services[@]}"; do
        local namespace_to_check="$NAMESPACE"
        if [[ "$service" == "vault" ]]; then
            namespace_to_check="vault"
        fi
        
        if ! kubectl get pods -n "$namespace_to_check" -l "app=$service" --field-selector=status.phase=Running | grep -q Running; then
            print_error "Service $service is not running properly"
            exit 1
        fi
    done
    
    # Check Helm releases
    print_info "Checking Helm releases..."
    helm list -A
    
    # Get service endpoints
    print_info "Service endpoints:"
    cd "$TERRAFORM_DIR"
    terraform output -json 2>/dev/null | jq -r 'to_entries[] | select(.key | endswith("_endpoint")) | "\(.key): \(.value.value)"' || true
    
    print_success "Deployment verification completed"
}

# Function to cleanup on failure
cleanup_on_failure() {
    print_error "Deployment failed. Cleaning up..."
    
    cd "$TERRAFORM_DIR"
    
    # Remove terraform plan file
    if [[ -f "tfplan" ]]; then
        rm -f tfplan
    fi
    
    # Remove variables file
    if [[ -f "terraform.tfvars" ]]; then
        rm -f terraform.tfvars
    fi
}

# Function to show deployment summary
show_deployment_summary() {
    print_info "Deployment Summary:"
    echo "Environment: $ENVIRONMENT"
    echo "Namespace: $NAMESPACE"
    echo "Kubeconfig: $KUBECONFIG_PATH"
    
    if [[ "$DESTROY" == "true" ]]; then
        echo "Action: Infrastructure destroyed"
    else
        echo "Action: Infrastructure deployed"
    fi
    
    echo
    print_info "Access information:"
    echo "- Vault: kubectl port-forward -n vault svc/vault 8200:8200"
    echo "- Keycloak: kubectl port-forward -n $NAMESPACE svc/keycloak 8080:8080"
    echo "- Pulsar Admin: kubectl port-forward -n $NAMESPACE svc/pulsar-admin 8080:8080"
    echo "- Grafana: kubectl port-forward -n $NAMESPACE svc/grafana 3000:3000"
    echo "- Prometheus: kubectl port-forward -n $NAMESPACE svc/prometheus 9090:9090"
    echo
    print_success "Deployment completed successfully!"
}

# Main execution
main() {
    print_info "Starting Q Platform infrastructure deployment..."
    print_info "Environment: $ENVIRONMENT"
    print_info "Namespace: $NAMESPACE"
    print_info "Destroy mode: $DESTROY"
    print_info "Dry run: $DRY_RUN"
    
    # Validate prerequisites
    validate_prerequisites
    
    # Initialize Terraform
    init_terraform
    
    # Plan deployment
    plan_deployment
    
    # Apply deployment
    apply_deployment
    
    # Wait for services if not destroying
    if [[ "$DESTROY" != "true" && "$DRY_RUN" != "true" ]]; then
        wait_for_services
        verify_deployment
    fi
    
    # Show summary
    show_deployment_summary
}

# Trap signals for cleanup
trap cleanup_on_failure ERR INT TERM

# Execute main function
main "$@" 