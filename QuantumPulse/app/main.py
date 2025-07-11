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
        self.api_providers = self._initialize_providers()
        self.last_prices = {}  # Cache for anomaly detection

    def _initialize_providers(self):
        """Initialize connections to real market data providers"""
        providers = []
        
        # Try to initialize multiple providers for redundancy
        try:
            # Alpha Vantage
            from shared.vault_client import VaultClient
            vault = VaultClient()
            av_creds = vault.read_secret_data("market_data/alphavantage")
            if av_creds:
                providers.append({
                    "name": "alphavantage",
                    "api_key": av_creds.get("api_key"),
                    "base_url": "https://www.alphavantage.co/query"
                })
        except Exception as e:
            logger.warning(f"Failed to initialize Alpha Vantage: {e}")
        
        try:
            # Yahoo Finance (no API key needed)
            providers.append({
                "name": "yahoo",
                "base_url": "https://query1.finance.yahoo.com/v8/finance/chart/"
            })
        except Exception as e:
            logger.warning(f"Failed to initialize Yahoo Finance: {e}")
        
        try:
            # IEX Cloud
            iex_creds = vault.read_secret_data("market_data/iexcloud")
            if iex_creds:
                providers.append({
                    "name": "iexcloud",
                    "api_key": iex_creds.get("api_key"),
                    "base_url": "https://cloud.iexapis.com/stable"
                })
        except Exception as e:
            logger.warning(f"Failed to initialize IEX Cloud: {e}")
        
        # Fallback to mock if no providers available
        if not providers:
            logger.warning("No market data providers available, using mock data")
            providers.append({"name": "mock"})
        
        return providers

    def _fetch_real_market_data(self, symbols):
        """Fetch real market data from available providers"""
        import httpx
        market_data = {}
        
        for provider in self.api_providers:
            try:
                if provider["name"] == "alphavantage" and "api_key" in provider:
                    # Alpha Vantage real-time quotes
                    with httpx.Client() as client:
                        for symbol in symbols:
                            response = client.get(
                                provider["base_url"],
                                params={
                                    "function": "GLOBAL_QUOTE",
                                    "symbol": symbol,
                                    "apikey": provider["api_key"]
                                }
                            )
                            if response.status_code == 200:
                                data = response.json()
                                if "Global Quote" in data:
                                    quote = data["Global Quote"]
                                    market_data[symbol] = {
                                        "price": float(quote.get("05. price", 0)),
                                        "change": float(quote.get("09. change", 0)),
                                        "change_percent": quote.get("10. change percent", "0%"),
                                        "volume": int(quote.get("06. volume", 0)),
                                        "timestamp": time.time()
                                    }
                    if market_data:
                        return market_data
                        
                elif provider["name"] == "yahoo":
                    # Yahoo Finance real-time quotes
                    with httpx.Client() as client:
                        for symbol in symbols:
                            response = client.get(
                                f"{provider['base_url']}{symbol}",
                                params={"interval": "1m", "range": "1d"}
                            )
                            if response.status_code == 200:
                                data = response.json()
                                if "chart" in data and "result" in data["chart"]:
                                    result = data["chart"]["result"][0]
                                    if "meta" in result:
                                        meta = result["meta"]
                                        market_data[symbol] = {
                                            "price": meta.get("regularMarketPrice", 0),
                                            "previousClose": meta.get("previousClose", 0),
                                            "change": meta.get("regularMarketPrice", 0) - meta.get("previousClose", 0),
                                            "volume": meta.get("regularMarketVolume", 0),
                                            "timestamp": time.time()
                                        }
                    if market_data:
                        return market_data
                        
                elif provider["name"] == "iexcloud" and "api_key" in provider:
                    # IEX Cloud real-time quotes
                    with httpx.Client() as client:
                        symbols_str = ",".join(symbols)
                        response = client.get(
                            f"{provider['base_url']}/stock/market/batch",
                            params={
                                "symbols": symbols_str,
                                "types": "quote",
                                "token": provider["api_key"]
                            }
                        )
                        if response.status_code == 200:
                            data = response.json()
                            for symbol, info in data.items():
                                if "quote" in info:
                                    quote = info["quote"]
                                    market_data[symbol] = {
                                        "price": quote.get("latestPrice", 0),
                                        "change": quote.get("change", 0),
                                        "change_percent": quote.get("changePercent", 0),
                                        "volume": quote.get("volume", 0),
                                        "timestamp": time.time()
                                    }
                    if market_data:
                        return market_data
                        
            except Exception as e:
                logger.error(f"Failed to fetch from {provider['name']}: {e}")
                continue
        
        # If all providers fail, return mock data
        return self._generate_mock_data(symbols)
    
    def _generate_mock_data(self, symbols):
        """Generate mock market data as fallback"""
        market_data = {}
        for symbol in symbols:
            if symbol not in self.last_prices:
                self.last_prices[symbol] = random.uniform(150, 2000)
            
            price = self.last_prices[symbol]
            change = random.uniform(-0.5, 0.5)
            
            # Simulate occasional anomalies
            if random.random() < 0.01:
                spike = random.uniform(5, 10)
                change = spike if random.random() < 0.5 else -spike
            
            new_price = max(0, price + change)
            self.last_prices[symbol] = new_price
            
            market_data[symbol] = {
                "price": round(new_price, 2),
                "change": round(change, 2),
                "change_percent": round((change / price) * 100, 2),
                "volume": random.randint(1000000, 10000000),
                "timestamp": time.time()
            }
        
        return market_data
    
    def _detect_anomalies(self, symbol, current_data):
        """Detect anomalies in market data"""
        anomalies = []
        
        if symbol in self.last_prices:
            last_price = self.last_prices[symbol]
            current_price = current_data["price"]
            
            # Price spike detection
            price_change_pct = abs((current_price - last_price) / last_price) * 100
            if price_change_pct > 2.0:  # 2% change threshold
                anomalies.append({
                    "type": "price_spike",
                    "severity": "high" if price_change_pct > 5.0 else "medium",
                    "change_percent": price_change_pct,
                    "direction": "up" if current_price > last_price else "down"
                })
            
            # Volume anomaly detection
            if "volume" in current_data and current_data["volume"] > 5000000:
                anomalies.append({
                    "type": "volume_spike",
                    "severity": "medium",
                    "volume": current_data["volume"]
                })
        
        return anomalies

    def run(self):
        self._running = True
        producer = self.client.get_producer(self.topic, schema=BytesSchema())
        
        # Main stock symbols to monitor
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ"]
        
        # For high-frequency simulation, we'll cache real data and interpolate
        real_data_cache = {}
        last_fetch_time = 0
        fetch_interval = 60  # Fetch real data every 60 seconds
        
        while self._running:
            current_time = time.time()
            
            # Fetch new real data periodically
            if current_time - last_fetch_time > fetch_interval:
                real_data_cache = self._fetch_real_market_data(symbols)
                last_fetch_time = current_time
            
            # Stream data with interpolation for high frequency
            for symbol in symbols:
                if symbol in real_data_cache:
                    base_data = real_data_cache[symbol]
                    
                    # Add small random variations for high-frequency simulation
                    current_data = {
                        "symbol": symbol,
                        "price": base_data["price"] + random.uniform(-0.01, 0.01),
                        "change": base_data.get("change", 0),
                        "change_percent": base_data.get("change_percent", 0),
                        "volume": base_data.get("volume", 0),
                        "timestamp": time.time(),
                        "source": next((p["name"] for p in self.api_providers if p["name"] != "mock"), "unknown")
                    }
                    
                    # Detect anomalies
                    anomalies = self._detect_anomalies(symbol, current_data)
                    if anomalies:
                        current_data["anomalies"] = anomalies
                    
                    # Send to Pulsar
                    producer.send(json.dumps(current_data).encode('utf-8'))
                    
                    # Update last price for anomaly detection
                    self.last_prices[symbol] = current_data["price"]
            
            # High frequency streaming (10 updates per second)
            time.sleep(0.1)

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