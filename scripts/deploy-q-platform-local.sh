#!/bin/bash

# Q Platform Local Deployment Script
# This script builds, pushes, and deploys the Q Platform to local minikube

set -e

MINIKUBE_IP=$(minikube ip)
REGISTRY_URL="$MINIKUBE_IP:5000"

echo "🚀 Starting Q Platform Local Deployment"
echo "📍 Cluster IP: $MINIKUBE_IP"
echo "🗂️ Registry: $REGISTRY_URL"

# Check if minikube is running
if ! minikube status > /dev/null 2>&1; then
    echo "❌ Minikube is not running. Please start it first:"
    echo "   minikube start --driver=docker --memory=8192 --cpus=4"
    exit 1
fi

# Configure Docker to use minikube's Docker daemon
echo "🔧 Configuring Docker environment..."
eval $(minikube docker-env)

# Build Docker images
echo "🏗️ Building Docker images..."

# List of services to build with their directory names
declare -A SERVICES=(
    ["agentQ"]="agentq"
    ["managerQ"]="managerq"
    ["H2M"]="h2m"
    ["KnowledgeGraphQ"]="knowledgegraphq"
    ["VectorStoreQ"]="vectorstoreq"
    ["IntegrationHub"]="integrationhub"
    ["UserProfileQ"]="userprofileq"
    ["WebAppQ"]="webappq"
    ["QuantumPulse"]="quantumpulse"
    ["AuthQ"]="authq"
    ["AIOps"]="aiops"
    ["AgentSandbox"]="agentsandbox"
    ["WorkflowWorker"]="workflowworker"
)

# Function to prepare service for building
prepare_service() {
    local service_dir="$1"
    local service_name="$2"
    
    echo "🔧 Preparing $service_name..."
    
    # Check if Dockerfile exists
    if [ ! -f "$service_dir/Dockerfile" ]; then
        echo "   ⚠️  No Dockerfile found for $service_name, skipping..."
        return 1
    fi
    
    # Copy shared directory if it doesn't exist in the service directory
    if [ ! -d "$service_dir/shared" ] && [ -d "shared" ]; then
        echo "   Copying shared directory..."
        cp -r shared "$service_dir/shared"
    fi
    
    return 0
}

# Function to cleanup service after building
cleanup_service() {
    local service_dir="$1"
    
    # Remove copied shared directory if it exists
    if [ -d "$service_dir/shared" ]; then
        echo "   Cleaning up shared directory..."
        rm -rf "$service_dir/shared"
    fi
}

for service in "${!SERVICES[@]}"; do
    if [ -d "$service" ]; then
        SERVICE_LOWER="${SERVICES[$service]}"
        
        # Prepare service for building
        if prepare_service "$service" "$SERVICE_LOWER"; then
            echo "📦 Building $service..."
            
            # Build the image from the service directory
            cd "$service"
            docker build -t "$REGISTRY_URL/$SERVICE_LOWER:latest" .
            
            # Tag for local use
            docker tag "$REGISTRY_URL/$SERVICE_LOWER:latest" "$SERVICE_LOWER:latest"
            
            echo "✅ Built $service as $SERVICE_LOWER"
            cd ..
            
            # Cleanup after building
            cleanup_service "$service"
        fi
    else
        echo "⚠️  Directory $service not found, skipping..."
    fi
done

# Apply Kubernetes manifests
echo "🎯 Deploying to Kubernetes..."

# Create namespace if it doesn't exist
kubectl create namespace q-platform --dry-run=client -o yaml | kubectl apply -f -

# Deploy using kustomize
echo "📝 Applying Kubernetes manifests..."
kubectl apply -k infra/kubernetes/base/

# Wait for deployments to be ready
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment --all -n q-platform

# Show deployment status
echo "📊 Deployment Status:"
kubectl get pods -n q-platform

# Show services
echo "🌐 Services:"
kubectl get services -n q-platform

# Show ingress if available
if kubectl get ingress -n q-platform > /dev/null 2>&1; then
    echo "🚪 Ingress:"
    kubectl get ingress -n q-platform
fi

echo "🎉 Q Platform deployment complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Check pod status: kubectl get pods -n q-platform"
echo "   2. View logs: kubectl logs -f deployment/manager-q -n q-platform"
echo "   3. Port forward services: kubectl port-forward service/manager-q 8003:8003 -n q-platform"
echo "   4. Access services via minikube: minikube service list"
echo ""
echo "🛠️ Troubleshooting:"
echo "   - Check events: kubectl get events -n q-platform"
echo "   - Describe failing pods: kubectl describe pod <pod-name> -n q-platform"
echo "   - View all resources: kubectl get all -n q-platform" 