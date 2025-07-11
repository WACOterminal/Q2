import sys
import os
# Add the shared directory to the path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import FastAPI
import uvicorn
import logging
import structlog
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect
import json

from app.api import chat, feedback, registry
from app.core.config import config
from app.services.ignite_client import ignite_client
from app.services.h2m_pulsar import h2m_pulsar_client
from app.core.thought_listener import thought_listener
from app.core.copilot_handler import copilot_handler
from shared.observability.logging_config import setup_logging
from shared.observability.metrics import setup_metrics
from shared.opentelemetry.tracing import setup_tracing

# --- Logging and Metrics Setup ---
setup_logging()
logger = structlog.get_logger(__name__)

# --- FastAPI App ---
app = FastAPI(
    title=config.service_name,
    version=config.version,
    description="Human-to-Machine (H2M) service for conversational AI orchestration."
)

# Setup Prometheus metrics and tracing
setup_metrics(app, app_name=config.service_name)
setup_tracing(app, service_name=config.service_name)

@app.on_event("startup")
async def startup_event():
    """Initializes and starts all background services."""
    logger.info("H2M starting up...")
    try:
        await ignite_client.connect()
        h2m_pulsar_client.start_producers()
        
        # --- NEW: Initialize and start the CoPilotHandler ---
        copilot_handler.client = h2m_pulsar_client.client
        copilot_handler.start()
        
        thought_listener.start()
    except Exception as e:
        logger.critical(f"Could not initialize H2M services on startup: {e}", exc_info=True)

@app.on_event("shutdown")
async def shutdown_event():
    """Stops all background services gracefully."""
    logger.info("H2M shutting down...")
    await ignite_client.disconnect()
    h2m_pulsar_client.close()
    thought_listener.stop()

# Include the API routers
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["Feedback"])
app.include_router(registry.router, prefix="/api/v1/registry", tags=["Model Registry"])

# --- NEW: Co-Pilot WebSocket Endpoint ---
@app.websocket("/api/v1/copilot/ws/{conversation_id}")
async def copilot_websocket_endpoint(websocket: WebSocket, conversation_id: str):
    """Handles the co-pilot WebSocket connection for a specific conversation."""
    await copilot_handler.register_session(conversation_id, websocket)
    try:
        while True:
            # This is where we receive feedback from the human co-pilot
            data = await websocket.receive_text()
            response_data = json.loads(data)
            await copilot_handler.handle_human_response(response_data)
    except WebSocketDisconnect:
        copilot_handler.unregister_session(conversation_id)
        logger.info(f"Client disconnected from co-pilot session: {conversation_id}")


@app.get("/health", tags=["Health"])
def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok", "ignite_connected": ignite_client.client.is_connected()}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.api.host,
        port=config.api.port,
        reload=True
    ) 