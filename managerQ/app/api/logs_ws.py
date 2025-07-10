
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from managerQ.app.core.log_streamer import log_streamer
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for clients to subscribe to real-time logs for a specific task.
    """
    await websocket.accept()
    await log_streamer.add_subscriber(task_id, websocket)
    
    try:
        while True:
            # Keep the connection alive, listening for messages (e.g., a "close" message)
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(f"Log stream client disconnected for task_id: {task_id}")
    finally:
        await log_streamer.remove_subscriber(task_id, websocket) 