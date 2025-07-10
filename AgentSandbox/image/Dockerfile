# Use a minimal and secure base image
FROM debian:11-slim

# Install essential tools that an agent might need
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for the agent to run as
RUN groupadd -r agent && useradd -r -g agent -m -s /bin/bash agent

# Create a dedicated workspace for the agent
RUN mkdir -p /workspace && chown -R agent:agent /workspace

# Switch to the non-root user
USER agent
WORKDIR /workspace

# Set a simple default command
CMD ["/bin/bash"] 