from fastapi import FastAPI
import uvicorn
import logging
import structlog
import asyncio
from contextlib import asynccontextmanager
import threading
import time
import random
import json
from pulsar.schema import BytesSchema

from app.api.endpoints import inference, fine_tuning, chat
from app.core.pulsar_client import PulsarManager
from app.core import pulsar_client as pulsar_manager_module
from app.core.config import config
from shared.opentelemetry.tracing import setup_tracing
from shared.observability.logging_config import setup_logging
from shared.observability.metrics import setup_metrics
from app.stream_processors.qgan_handler import start_qgan_handler, stop_qgan_handler

# --- Logging and Metrics Setup ---
setup_logging()
logger = structlog.get_logger(__name__)

# --- FastAPI App ---
app = FastAPI(
    title="QuantumPulse",
    version="0.2.0",
    description="A unified service for LLM inference, routing, and fine-tuning."
)

# Setup Prometheus metrics
setup_metrics(app, app_name=config.service_name)

# Setup OpenTelemetry
setup_tracing(app, service_name=config.service_name)

# --- NEW: Market Data Producer ---
class MarketDataProducer(threading.Thread):
    def __init__(self, client, topic):
        super().__init__()
        self.client = client
        self.topic = topic
        self._running = False

    def run(self):
        self._running = True
        producer = self.client.get_producer(self.topic, schema=BytesSchema())
        stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        prices = {s: random.uniform(150, 2000) for s in stocks}

        while self._running:
            # Simulate market data
            for stock in stocks:
                price = prices[stock]
                # Normal market fluctuation
                change = random.uniform(-0.5, 0.5)
                new_price = price + change
                
                # Chance for an anomalous spike
                if random.random() < 0.01:
                    spike = random.uniform(5, 10)
                    new_price += spike if random.random() < 0.5 else -spike
                    
                prices[stock] = max(0, new_price)
                
                payload = {
                    "symbol": stock,
                    "price": round(prices[stock], 2),
                    "timestamp": time.time()
                }
                producer.send(json.dumps(payload).encode('utf-8'))
            
            time.sleep(0.1) # High frequency stream

    def stop(self):
        self._running = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("QuantumPulse API starting up...")
    
    # Initialize Pulsar client
    pulsar_manager_module.pulsar_manager = PulsarManager(
        service_url=config.pulsar.service_url,
        token=config.pulsar.token,
        tls_trust_certs_file_path=config.pulsar.tls_trust_certs_file_path
    )
    try:
        pulsar_manager_module.pulsar_manager.connect()
    except Exception as e:
        logger.error(f"Failed to connect to Pulsar on startup: {e}", exc_info=True)
        # Depending on the desired behavior, you might want to exit the application
        # exit(1)

    # --- NEW: Start the market data producer ---
    market_data_topic = "persistent://public/default/market-data"
    market_producer = MarketDataProducer(pulsar_manager_module.pulsar_manager, market_data_topic)
    market_producer.start()

    # --- NEW: Start the QGAN command handler ---
    start_qgan_handler()
    
    yield
    
    logger.info("QuantumPulse API shutting down...")
    stop_qgan_handler() # Stop the QGAN handler
    market_producer.stop()
    market_producer.join()
    pulsar_manager_module.pulsar_manager.close()

# --- API Routers ---
app.include_router(inference.router, prefix="/v1/inference", tags=["Inference"])
app.include_router(fine_tuning.router, prefix="/v1/fine-tune", tags=["Fine-Tuning"])
app.include_router(chat.router, prefix="/v1/chat", tags=["Chat"])

@app.get("/health", tags=["Health"])
def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.api.host,
        port=config.api.port,
        reload=True # Use reload for development
    ) 