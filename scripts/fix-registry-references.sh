#!/bin/bash

# Fix Q Platform Registry References
# This script updates all "your-registry" references to use the local minikube registry

MINIKUBE_IP=$(minikube ip)
REGISTRY_URL="$MINIKUBE_IP:5000"

echo "🔧 Fixing registry references to use: $REGISTRY_URL"

# Find and replace all your-registry references
find . -name "*.yaml" -type f -exec sed -i "s|your-registry\.com|$REGISTRY_URL|g" {} \;
find . -name "*.yaml" -type f -exec sed -i "s|your-registry/|$REGISTRY_URL/|g" {} \;
find . -name "*.yaml" -type f -exec sed -i "s|your-registry|$REGISTRY_URL|g" {} \;

echo "✅ Updated registry references to: $REGISTRY_URL"

# Setup port forwarding for registry access
echo "🔗 Setting up registry port forwarding..."
kubectl port-forward --namespace kube-system service/registry 5000:80 &
REGISTRY_PID=$!

echo "✅ Registry available at localhost:5000"
echo "📝 Registry PID: $REGISTRY_PID (kill with: kill $REGISTRY_PID)"

# Create a helper script to stop the port forwarding
cat > scripts/stop-registry-forwarding.sh << 'EOF'
#!/bin/bash
pkill -f "kubectl port-forward.*registry"
echo "✅ Stopped registry port forwarding"
EOF

chmod +x scripts/stop-registry-forwarding.sh

echo "🎉 Registry configuration complete!"
echo "   - Registry URL: $REGISTRY_URL"
echo "   - Port forwarding running in background"
echo "   - Stop with: ./scripts/stop-registry-forwarding.sh" 