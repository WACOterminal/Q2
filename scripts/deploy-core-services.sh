#!/bin/bash

# Deploy Core Q Platform Services
# This script builds and deploys the essential services

set -e

echo "🚀 Deploying Core Q Platform Services"

# Configure Docker to use minikube's Docker daemon
eval $(minikube docker-env)

# Core services to deploy
SERVICES=("managerQ" "agentQ" "H2M" "WebAppQ")

# Build each service
for service in "${SERVICES[@]}"; do
    if [ -d "$service" ]; then
        echo "📦 Building $service..."
        
        # Clean up any existing shared directory
        rm -rf "$service/shared"
        
        # Copy shared directory
        cp -r shared "$service/shared"
        
        # Build the image
        cd "$service"
        SERVICE_LOWER=$(echo "$service" | tr '[:upper:]' '[:lower:]')
        docker build -t "192.168.67.2:5000/$SERVICE_LOWER:latest" .
        cd ..
        
        # Clean up
        rm -rf "$service/shared"
        
        echo "✅ Built $service"
    else
        echo "⚠️  Directory $service not found, skipping..."
    fi
done

# Deploy to Kubernetes
echo "🎯 Deploying to Kubernetes..."

# Create basic deployments
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: manager-q
  namespace: q-platform
spec:
  replicas: 1
  selector:
    matchLabels:
      app: manager-q
  template:
    metadata:
      labels:
        app: manager-q
    spec:
      containers:
      - name: manager-q
        image: 192.168.67.2:5000/managerq:latest
        imagePullPolicy: Never
        ports:
        - containerPort: 8003
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
  name: manager-q
  namespace: q-platform
spec:
  selector:
    app: manager-q
  ports:
  - port: 8003
    targetPort: 8003
  type: ClusterIP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webappq
  namespace: q-platform
spec:
  replicas: 1
  selector:
    matchLabels:
      app: webappq
  template:
    metadata:
      labels:
        app: webappq
    spec:
      containers:
      - name: webappq
        image: 192.168.67.2:5000/webappq:latest
        imagePullPolicy: Never
        ports:
        - containerPort: 3000
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 200m
            memory: 256Mi
---
apiVersion: v1
kind: Service
metadata:
  name: webappq
  namespace: q-platform
spec:
  selector:
    app: webappq
  ports:
  - port: 3000
    targetPort: 3000
  type: ClusterIP
EOF

# Wait for deployments
echo "⏳ Waiting for deployments..."
kubectl wait --for=condition=available --timeout=300s deployment/manager-q -n q-platform || true
kubectl wait --for=condition=available --timeout=300s deployment/webappq -n q-platform || true

# Show status
echo "📊 Deployment Status:"
kubectl get pods -n q-platform
kubectl get services -n q-platform

# Port forward for easy access
echo "🌐 Setting up port forwarding..."
kubectl port-forward service/manager-q 8003:8003 -n q-platform &
kubectl port-forward service/webappq 3000:3000 -n q-platform &

echo "🎉 Core services deployed!"
echo ""
echo "🔗 Access URLs:"
echo "   Manager Q: http://localhost:8003"
echo "   Web App: http://localhost:3000"
echo ""
echo "📋 Commands:"
echo "   View logs: kubectl logs -f deployment/manager-q -n q-platform"
echo "   Check status: kubectl get pods -n q-platform"
echo "   Stop port forwarding: pkill -f 'kubectl port-forward'" 