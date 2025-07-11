#!/bin/bash

# Q Platform Deployment Validation Script
# This script validates the complete Q Platform deployment

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="development"
NAMESPACE="q-platform"
TIMEOUT="300s"
VERBOSE=false
OUTPUT_FORMAT="text"
SAVE_REPORT=false
REPORT_FILE=""

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

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

Validate Q Platform deployment

OPTIONS:
    -e, --environment ENVIRONMENT   Target environment (development, staging, production)
    -n, --namespace NAMESPACE       Kubernetes namespace (default: q-platform)
    -t, --timeout TIMEOUT           Timeout for checks (default: 300s)
    -v, --verbose                   Enable verbose output
    -f, --format FORMAT             Output format (text, json, html)
    -s, --save-report              Save report to file
    -r, --report-file FILE         Report file path
    -h, --help                     Show this help message

EXAMPLES:
    # Validate development environment
    $0 -e development

    # Validate with verbose output and save report
    $0 -e production -v -s -r validation-report.html

    # Validate with custom namespace
    $0 -e staging -n custom-namespace

EOF
}

# Function to increment counters
increment_total() {
    ((TOTAL_CHECKS++))
}

increment_passed() {
    ((PASSED_CHECKS++))
}

increment_failed() {
    ((FAILED_CHECKS++))
}

increment_warning() {
    ((WARNING_CHECKS++))
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to validate prerequisites
validate_prerequisites() {
    print_info "Validating prerequisites..."
    
    local required_tools=("kubectl" "helm" "curl" "jq")
    for tool in "${required_tools[@]}"; do
        increment_total
        if command_exists "$tool"; then
            print_success "$tool is available"
            increment_passed
        else
            print_error "$tool is not available"
            increment_failed
        fi
    done
    
    # Check Kubernetes connectivity
    increment_total
    if kubectl cluster-info &> /dev/null; then
        print_success "Kubernetes cluster is accessible"
        increment_passed
    else
        print_error "Cannot connect to Kubernetes cluster"
        increment_failed
        return 1
    fi
    
    # Check namespace exists
    increment_total
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        print_success "Namespace '$NAMESPACE' exists"
        increment_passed
    else
        print_error "Namespace '$NAMESPACE' does not exist"
        increment_failed
    fi
}

# Function to check infrastructure components
check_infrastructure() {
    print_info "Checking infrastructure components..."
    
    # Check Apache Pulsar
    increment_total
    if kubectl get statefulset -n "$NAMESPACE" | grep -q pulsar; then
        local pulsar_ready
        pulsar_ready=$(kubectl get statefulset -n "$NAMESPACE" -o jsonpath='{.items[?(@.metadata.labels.app=="pulsar")].status.readyReplicas}' 2>/dev/null || echo "0")
        if [ "$pulsar_ready" -gt 0 ]; then
            print_success "Apache Pulsar is running ($pulsar_ready replicas)"
            increment_passed
        else
            print_error "Apache Pulsar is not ready"
            increment_failed
        fi
    else
        print_warning "Apache Pulsar not found (may be disabled)"
        increment_warning
    fi
    
    # Check Apache Cassandra
    increment_total
    if kubectl get statefulset -n "$NAMESPACE" | grep -q cassandra; then
        local cassandra_ready
        cassandra_ready=$(kubectl get statefulset -n "$NAMESPACE" -o jsonpath='{.items[?(@.metadata.labels.app=="cassandra")].status.readyReplicas}' 2>/dev/null || echo "0")
        if [ "$cassandra_ready" -gt 0 ]; then
            print_success "Apache Cassandra is running ($cassandra_ready replicas)"
            increment_passed
        else
            print_error "Apache Cassandra is not ready"
            increment_failed
        fi
    else
        print_warning "Apache Cassandra not found (may be disabled)"
        increment_warning
    fi
    
    # Check Milvus
    increment_total
    if kubectl get deployment -n "$NAMESPACE" | grep -q milvus; then
        local milvus_ready
        milvus_ready=$(kubectl get deployment -n "$NAMESPACE" -o jsonpath='{.items[?(@.metadata.labels.app=="milvus")].status.readyReplicas}' 2>/dev/null || echo "0")
        if [ "$milvus_ready" -gt 0 ]; then
            print_success "Milvus is running ($milvus_ready replicas)"
            increment_passed
        else
            print_error "Milvus is not ready"
            increment_failed
        fi
    else
        print_warning "Milvus not found (may be disabled)"
        increment_warning
    fi
    
    # Check Apache Ignite
    increment_total
    if kubectl get statefulset -n "$NAMESPACE" | grep -q ignite; then
        local ignite_ready
        ignite_ready=$(kubectl get statefulset -n "$NAMESPACE" -o jsonpath='{.items[?(@.metadata.labels.app=="ignite")].status.readyReplicas}' 2>/dev/null || echo "0")
        if [ "$ignite_ready" -gt 0 ]; then
            print_success "Apache Ignite is running ($ignite_ready replicas)"
            increment_passed
        else
            print_error "Apache Ignite is not ready"
            increment_failed
        fi
    else
        print_warning "Apache Ignite not found (may be disabled)"
        increment_warning
    fi
    
    # Check Elasticsearch
    increment_total
    if kubectl get statefulset -n "$NAMESPACE" | grep -q elasticsearch; then
        local elasticsearch_ready
        elasticsearch_ready=$(kubectl get statefulset -n "$NAMESPACE" -o jsonpath='{.items[?(@.metadata.labels.app=="elasticsearch")].status.readyReplicas}' 2>/dev/null || echo "0")
        if [ "$elasticsearch_ready" -gt 0 ]; then
            print_success "Elasticsearch is running ($elasticsearch_ready replicas)"
            increment_passed
        else
            print_error "Elasticsearch is not ready"
            increment_failed
        fi
    else
        print_warning "Elasticsearch not found (may be disabled)"
        increment_warning
    fi
    
    # Check MinIO
    increment_total
    if kubectl get deployment -n "$NAMESPACE" | grep -q minio; then
        local minio_ready
        minio_ready=$(kubectl get deployment -n "$NAMESPACE" -o jsonpath='{.items[?(@.metadata.labels.app=="minio")].status.readyReplicas}' 2>/dev/null || echo "0")
        if [ "$minio_ready" -gt 0 ]; then
            print_success "MinIO is running ($minio_ready replicas)"
            increment_passed
        else
            print_error "MinIO is not ready"
            increment_failed
        fi
    else
        print_warning "MinIO not found (may be disabled)"
        increment_warning
    fi
}

# Function to check Q Platform services
check_q_platform_services() {
    print_info "Checking Q Platform services..."
    
    local services=(
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
    
    for service in "${services[@]}"; do
        increment_total
        if kubectl get deployment "$service" -n "$NAMESPACE" &> /dev/null; then
            local ready_replicas
            ready_replicas=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
            local desired_replicas
            desired_replicas=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
            
            if [ "$ready_replicas" -eq "$desired_replicas" ] && [ "$ready_replicas" -gt 0 ]; then
                print_success "$service is running ($ready_replicas/$desired_replicas replicas)"
                increment_passed
            else
                print_error "$service is not ready ($ready_replicas/$desired_replicas replicas)"
                increment_failed
            fi
        else
            print_warning "$service deployment not found (may be disabled)"
            increment_warning
        fi
    done
}

# Function to check services and networking
check_services_networking() {
    print_info "Checking services and networking..."
    
    local services=(
        "manager-q"
        "h2m-service"
        "quantum-pulse-api"
        "vector-store-q"
        "knowledge-graph-q"
        "integration-hub"
        "user-profile-q"
        "webapp-q"
    )
    
    for service in "${services[@]}"; do
        increment_total
        if kubectl get service "$service" -n "$NAMESPACE" &> /dev/null; then
            local endpoints
            endpoints=$(kubectl get endpoints "$service" -n "$NAMESPACE" -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null || echo "")
            
            if [ -n "$endpoints" ]; then
                print_success "$service service has endpoints"
                increment_passed
            else
                print_error "$service service has no endpoints"
                increment_failed
            fi
        else
            print_warning "$service service not found (may be disabled)"
            increment_warning
        fi
    done
}

# Function to check persistent volumes
check_persistent_volumes() {
    print_info "Checking persistent volumes..."
    
    # Check PVCs
    increment_total
    local pvcs
    pvcs=$(kubectl get pvc -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    if [ "$pvcs" -gt 0 ]; then
        print_success "Found $pvcs persistent volume claims"
        increment_passed
        
        # Check PVC status
        local bound_pvcs
        bound_pvcs=$(kubectl get pvc -n "$NAMESPACE" --no-headers 2>/dev/null | grep -c "Bound" || echo "0")
        increment_total
        if [ "$bound_pvcs" -eq "$pvcs" ]; then
            print_success "All PVCs are bound ($bound_pvcs/$pvcs)"
            increment_passed
        else
            print_error "Some PVCs are not bound ($bound_pvcs/$pvcs)"
            increment_failed
        fi
    else
        print_warning "No persistent volume claims found"
        increment_warning
    fi
}

# Function to check secrets and configmaps
check_secrets_configmaps() {
    print_info "Checking secrets and configmaps..."
    
    # Check secrets
    increment_total
    local secrets
    secrets=$(kubectl get secrets -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    if [ "$secrets" -gt 0 ]; then
        print_success "Found $secrets secrets"
        increment_passed
    else
        print_error "No secrets found"
        increment_failed
    fi
    
    # Check configmaps
    increment_total
    local configmaps
    configmaps=$(kubectl get configmaps -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    if [ "$configmaps" -gt 0 ]; then
        print_success "Found $configmaps configmaps"
        increment_passed
    else
        print_error "No configmaps found"
        increment_failed
    fi
}

# Function to check ingress and external access
check_ingress_access() {
    print_info "Checking ingress and external access..."
    
    # Check ingress controller
    increment_total
    if kubectl get deployment -n ingress-nginx ingress-nginx-controller &> /dev/null; then
        print_success "Ingress controller is available"
        increment_passed
    else
        print_warning "Ingress controller not found in ingress-nginx namespace"
        increment_warning
    fi
    
    # Check ingress resources
    increment_total
    local ingresses
    ingresses=$(kubectl get ingress -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    if [ "$ingresses" -gt 0 ]; then
        print_success "Found $ingresses ingress resources"
        increment_passed
    else
        print_warning "No ingress resources found"
        increment_warning
    fi
}

# Function to check security components
check_security_components() {
    print_info "Checking security components..."
    
    # Check Vault
    increment_total
    if kubectl get deployment -n vault vault &> /dev/null; then
        local vault_ready
        vault_ready=$(kubectl get deployment -n vault vault -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        if [ "$vault_ready" -gt 0 ]; then
            print_success "Vault is running ($vault_ready replicas)"
            increment_passed
        else
            print_error "Vault is not ready"
            increment_failed
        fi
    else
        print_warning "Vault not found (may be disabled)"
        increment_warning
    fi
    
    # Check External Secrets Operator
    increment_total
    if kubectl get deployment -n external-secrets-system external-secrets &> /dev/null; then
        local eso_ready
        eso_ready=$(kubectl get deployment -n external-secrets-system external-secrets -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        if [ "$eso_ready" -gt 0 ]; then
            print_success "External Secrets Operator is running ($eso_ready replicas)"
            increment_passed
        else
            print_error "External Secrets Operator is not ready"
            increment_failed
        fi
    else
        print_warning "External Secrets Operator not found (may be disabled)"
        increment_warning
    fi
    
    # Check Keycloak
    increment_total
    if kubectl get deployment -n "$NAMESPACE" keycloak &> /dev/null; then
        local keycloak_ready
        keycloak_ready=$(kubectl get deployment -n "$NAMESPACE" keycloak -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        if [ "$keycloak_ready" -gt 0 ]; then
            print_success "Keycloak is running ($keycloak_ready replicas)"
            increment_passed
        else
            print_error "Keycloak is not ready"
            increment_failed
        fi
    else
        print_warning "Keycloak not found (may be disabled)"
        increment_warning
    fi
}

# Function to check monitoring components
check_monitoring_components() {
    print_info "Checking monitoring components..."
    
    # Check Prometheus
    increment_total
    if kubectl get deployment -n "$NAMESPACE" prometheus-server &> /dev/null; then
        local prometheus_ready
        prometheus_ready=$(kubectl get deployment -n "$NAMESPACE" prometheus-server -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        if [ "$prometheus_ready" -gt 0 ]; then
            print_success "Prometheus is running ($prometheus_ready replicas)"
            increment_passed
        else
            print_error "Prometheus is not ready"
            increment_failed
        fi
    else
        print_warning "Prometheus not found (may be disabled)"
        increment_warning
    fi
    
    # Check Grafana
    increment_total
    if kubectl get deployment -n "$NAMESPACE" grafana &> /dev/null; then
        local grafana_ready
        grafana_ready=$(kubectl get deployment -n "$NAMESPACE" grafana -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        if [ "$grafana_ready" -gt 0 ]; then
            print_success "Grafana is running ($grafana_ready replicas)"
            increment_passed
        else
            print_error "Grafana is not ready"
            increment_failed
        fi
    else
        print_warning "Grafana not found (may be disabled)"
        increment_warning
    fi
    
    # Check Jaeger
    increment_total
    if kubectl get deployment -n "$NAMESPACE" jaeger &> /dev/null; then
        local jaeger_ready
        jaeger_ready=$(kubectl get deployment -n "$NAMESPACE" jaeger -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        if [ "$jaeger_ready" -gt 0 ]; then
            print_success "Jaeger is running ($jaeger_ready replicas)"
            increment_passed
        else
            print_error "Jaeger is not ready"
            increment_failed
        fi
    else
        print_warning "Jaeger not found (may be disabled)"
        increment_warning
    fi
}

# Function to run health checks
run_health_checks() {
    print_info "Running health checks..."
    
    # Health check for Manager Q
    increment_total
    if kubectl get service manager-q -n "$NAMESPACE" &> /dev/null; then
        local manager_q_ip
        manager_q_ip=$(kubectl get service manager-q -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
        if curl -f -s "http://$manager_q_ip:8003/health" &> /dev/null; then
            print_success "Manager Q health check passed"
            increment_passed
        else
            print_error "Manager Q health check failed"
            increment_failed
        fi
    else
        print_warning "Manager Q service not found"
        increment_warning
    fi
    
    # Health check for H2M Service
    increment_total
    if kubectl get service h2m-service -n "$NAMESPACE" &> /dev/null; then
        local h2m_ip
        h2m_ip=$(kubectl get service h2m-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
        if curl -f -s "http://$h2m_ip:8002/health" &> /dev/null; then
            print_success "H2M Service health check passed"
            increment_passed
        else
            print_error "H2M Service health check failed"
            increment_failed
        fi
    else
        print_warning "H2M Service not found"
        increment_warning
    fi
    
    # Health check for Vector Store Q
    increment_total
    if kubectl get service vector-store-q -n "$NAMESPACE" &> /dev/null; then
        local vectorstore_ip
        vectorstore_ip=$(kubectl get service vector-store-q -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
        if curl -f -s "http://$vectorstore_ip:8001/health" &> /dev/null; then
            print_success "Vector Store Q health check passed"
            increment_passed
        else
            print_error "Vector Store Q health check failed"
            increment_failed
        fi
    else
        print_warning "Vector Store Q service not found"
        increment_warning
    fi
}

# Function to check resource usage
check_resource_usage() {
    print_info "Checking resource usage..."
    
    # Check CPU usage
    increment_total
    local cpu_usage
    cpu_usage=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '{sum+=$2} END {print sum}' || echo "0")
    if [ "$cpu_usage" -gt 0 ]; then
        print_success "CPU usage: ${cpu_usage}m"
        increment_passed
    else
        print_warning "Unable to get CPU usage metrics"
        increment_warning
    fi
    
    # Check memory usage
    increment_total
    local memory_usage
    memory_usage=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '{sum+=$3} END {print sum}' || echo "0")
    if [ "$memory_usage" -gt 0 ]; then
        print_success "Memory usage: ${memory_usage}Mi"
        increment_passed
    else
        print_warning "Unable to get memory usage metrics"
        increment_warning
    fi
}

# Function to generate report
generate_report() {
    local report_content=""
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$OUTPUT_FORMAT" in
        "json")
            report_content=$(cat << EOF
{
  "timestamp": "$timestamp",
  "environment": "$ENVIRONMENT",
  "namespace": "$NAMESPACE",
  "summary": {
    "total_checks": $TOTAL_CHECKS,
    "passed": $PASSED_CHECKS,
    "failed": $FAILED_CHECKS,
    "warnings": $WARNING_CHECKS,
    "success_rate": "$(( PASSED_CHECKS * 100 / TOTAL_CHECKS ))%"
  },
  "status": "$([ $FAILED_CHECKS -eq 0 ] && echo "healthy" || echo "unhealthy")"
}
EOF
            )
            ;;
        "html")
            report_content=$(cat << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Q Platform Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { margin: 20px 0; }
        .success { color: green; }
        .error { color: red; }
        .warning { color: orange; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Q Platform Validation Report</h1>
        <p>Generated: $timestamp</p>
        <p>Environment: $ENVIRONMENT</p>
        <p>Namespace: $NAMESPACE</p>
    </div>
    <div class="summary">
        <h2>Summary</h2>
        <p>Total Checks: $TOTAL_CHECKS</p>
        <p class="success">Passed: $PASSED_CHECKS</p>
        <p class="error">Failed: $FAILED_CHECKS</p>
        <p class="warning">Warnings: $WARNING_CHECKS</p>
        <p>Success Rate: $(( PASSED_CHECKS * 100 / TOTAL_CHECKS ))%</p>
    </div>
</body>
</html>
EOF
            )
            ;;
        *)
            report_content=$(cat << EOF
Q Platform Validation Report
============================
Generated: $timestamp
Environment: $ENVIRONMENT
Namespace: $NAMESPACE

Summary:
--------
Total Checks: $TOTAL_CHECKS
Passed: $PASSED_CHECKS
Failed: $FAILED_CHECKS
Warnings: $WARNING_CHECKS
Success Rate: $(( PASSED_CHECKS * 100 / TOTAL_CHECKS ))%

Status: $([ $FAILED_CHECKS -eq 0 ] && echo "HEALTHY" || echo "UNHEALTHY")
EOF
            )
            ;;
    esac
    
    if [ "$SAVE_REPORT" = true ]; then
        if [ -z "$REPORT_FILE" ]; then
            REPORT_FILE="q-platform-validation-$(date +%Y%m%d-%H%M%S).$OUTPUT_FORMAT"
        fi
        echo "$report_content" > "$REPORT_FILE"
        print_success "Report saved to: $REPORT_FILE"
    fi
    
    echo "$report_content"
}

# Function to show summary
show_summary() {
    echo ""
    print_info "Validation Summary"
    echo "===================="
    echo "Total Checks: $TOTAL_CHECKS"
    echo "Passed: $PASSED_CHECKS"
    echo "Failed: $FAILED_CHECKS"
    echo "Warnings: $WARNING_CHECKS"
    echo "Success Rate: $(( PASSED_CHECKS * 100 / TOTAL_CHECKS ))%"
    echo ""
    
    if [ $FAILED_CHECKS -eq 0 ]; then
        print_success "Q Platform deployment is HEALTHY"
        return 0
    else
        print_error "Q Platform deployment has ISSUES"
        return 1
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
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -f|--format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        -s|--save-report)
            SAVE_REPORT=true
            shift
            ;;
        -r|--report-file)
            REPORT_FILE="$2"
            SAVE_REPORT=true
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

# Main validation flow
main() {
    print_info "Starting Q Platform validation..."
    print_info "Environment: $ENVIRONMENT"
    print_info "Namespace: $NAMESPACE"
    print_info "Timeout: $TIMEOUT"
    
    # Run all validation checks
    validate_prerequisites
    check_infrastructure
    check_q_platform_services
    check_services_networking
    check_persistent_volumes
    check_secrets_configmaps
    check_ingress_access
    check_security_components
    check_monitoring_components
    run_health_checks
    check_resource_usage
    
    # Generate report
    generate_report
    
    # Show summary and exit with appropriate code
    show_summary
}

# Execute main function
main "$@" 