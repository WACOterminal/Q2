#!/bin/bash

# Simple Q Platform Deployment Script
# Build and deploy services one by one

set -e

echo "🚀 Simple Q Platform Deployment"

# Configure Docker environment
eval $(minikube docker-env)

# Function to build a service
build_service() {
    local service_dir="$1"
    local service_name="$2"
    
    echo "📦 Building $service_name..."
    
    if [ ! -d "$service_dir" ]; then
        echo "❌ Directory $service_dir not found"
        return 1
    fi
    
    if [ ! -f "$service_dir/Dockerfile" ]; then
        echo "❌ No Dockerfile found in $service_dir"
        return 1
    fi
    
    # Clean up any existing shared directory
    rm -rf "$service_dir/shared" 2>/dev/null || true
    
    # Copy shared directory
    if [ -d "shared" ]; then
        cp -r shared "$service_dir/shared"
    fi
    
    # Build the image
    cd "$service_dir"
    docker build -t "$service_name:latest" .
    echo "✅ Successfully built $service_name"
    cd ..
    
    # Clean up
    rm -rf "$service_dir/shared" 2>/dev/null || true
    
    return 0
}

# Function to deploy a service
deploy_service() {
    local service_name="$1"
    local port="$2"
    
    echo "🎯 Deploying $service_name..."
    
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $service_name
  namespace: q-platform
spec:
  replicas: 1
  selector:
    matchLabels:
      app: $service_name
  template:
    metadata:
      labels:
        app: $service_name
    spec:
      containers:
      - name: $service_name
        image: $service_name:latest
        imagePullPolicy: Never
        ports:
        - containerPort: $port
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 512Mi
---
apiVersion: v1
kind: Service
metadata:
  name: $service_name
  namespace: q-platform
spec:
  selector:
    app: $service_name
  ports:
  - port: $port
    targetPort: $port
  type: ClusterIP
EOF
    
    echo "✅ Deployed $service_name"
}

# Ensure namespace exists
kubectl create namespace q-platform --dry-run=client -o yaml | kubectl apply -f -

# Build and deploy services
echo "🏗️ Building services..."

if build_service "managerQ" "managerq"; then
    deploy_service "managerq" "8003"
fi

if build_service "agentQ" "agentq"; then
    deploy_service "agentq" "8000"
fi

if build_service "WebAppQ" "webappq"; then
    deploy_service "webappq" "3000"
fi

# Wait for deployments
echo "⏳ Waiting for deployments..."
kubectl wait --for=condition=available --timeout=300s deployment --all -n q-platform || true

# Show status
echo "📊 Deployment Status:"
kubectl get pods -n q-platform
kubectl get services -n q-platform

# Set up port forwarding for easy access
echo "🌐 Setting up port forwarding..."
kubectl port-forward service/managerq 8003:8003 -n q-platform &
kubectl port-forward service/webappq 3000:3000 -n q-platform &

echo ""
echo "🎉 Deployment Complete!"
echo ""
echo "🔗 Access URLs:"
echo "   Manager Q: http://localhost:8003"
echo "   Web App: http://localhost:3000"
echo ""
echo "📋 Useful Commands:"
echo "   Check pods: kubectl get pods -n q-platform"
echo "   View logs: kubectl logs -f deployment/managerq -n q-platform"
echo "   Stop port forwarding: pkill -f 'kubectl port-forward'" 