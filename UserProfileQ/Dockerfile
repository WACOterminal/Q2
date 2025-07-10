# Stage 1: Builder
FROM python:3.9-slim AS builder

WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Build wheels for shared libraries
COPY ./shared /usr/src/app/shared
RUN pip wheel --no-cache-dir --wheel-dir=/usr/src/app/wheels /usr/src/app/shared/observability /usr/src/app/shared/q_auth_parser

# Build wheels for application dependencies
COPY ./UserProfileQ/requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/usr/src/app/wheels -r requirements.txt

# Stage 2: Final image
FROM python:3.9-slim

# Create a non-root user
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser

WORKDIR /app

# Copy built wheels from builder stage
COPY --from=builder /usr/src/app/wheels /wheels

# Copy application code
COPY ./UserProfileQ/app /app/app

# Install all dependencies from wheels
RUN pip install --no-cache /wheels/*

# Change ownership and switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 