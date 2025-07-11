#!/bin/bash
pkill -f "kubectl port-forward.*registry"
echo "✅ Stopped registry port forwarding"
