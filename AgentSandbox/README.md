# AgentSandbox Service

## Overview

The `AgentSandbox` service provides secure, isolated environments for the execution of agent-generated code and commands. It is a critical component of the Q Platform's security model, ensuring that agent actions are performed in a controlled and resource-limited manner.

## Architecture

The service is composed of two main parts:

1.  **The Sandbox Base Image (`AgentSandbox/image`):** A minimal, secure Docker image that contains a non-root user and a basic set of tools (Python, curl, git) that an agent might need. This image serves as the foundation for all sandbox instances.

2.  **The Sandbox Manager Service (`AgentSandbox/app`):** A FastAPI application that exposes an API for managing the lifecycle of sandbox containers. It uses the Docker SDK to start, stop, and execute commands in containers based on the sandbox base image.

### Key Security Features

-   **Containerization:** All agent code is executed inside Docker containers, providing strong isolation from the host system.
-   **File System Isolation:** Each sandbox is provided with its own private workspace volume, preventing access to the host file system.
-   **Resource Limiting:** Sandboxes are created with strict CPU and memory limits to prevent resource exhaustion.
-   **Network Policies:** Networking is disabled by default for all sandboxes, and must be explicitly enabled for tasks that require it.
-   **Seccomp Profiles:** A default seccomp profile is applied to restrict the system calls a container can make, significantly reducing the potential attack surface.

## API Endpoints

All endpoints are available under the `/api/v1` prefix.

| Method | Endpoint                             | Description                                  |
|--------|--------------------------------------|----------------------------------------------|
| `POST` | `/sandboxes`                         | Creates a new sandbox environment.           |
| `POST` | `/sandboxes/{container_id}/execute`  | Executes a command in a specific sandbox.    |
| `POST` | `/sandboxes/{container_id}/files/upload` | Uploads a file to a sandbox's workspace.   |
| `GET`  | `/sandboxes/{container_id}/files/download`| Downloads a file from a sandbox's workspace.|
| `DELETE`| `/sandboxes/{container_id}`          | Stops and removes a specific sandbox.        |

## Getting Started

### 1. Build the Sandbox Base Image

Before running the service, you must build the sandbox base image.

```bash
docker build -t qagi/agent-sandbox:latest AgentSandbox/image
```

### 2. Run the Service

The service can be run like any other FastAPI application:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn AgentSandbox.app.main:app --host 0.0.0.0 --port 8004
```

The `agentQ` service is pre-configured to use this service for its `run_command` tool.
