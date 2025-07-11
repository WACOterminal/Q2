# agentQ/app/core/finops_tools.py
import structlog
import json
from datetime import datetime, timedelta
import random
from agentQ.app.core.toolbox import Tool

logger = structlog.get_logger(__name__)

def get_cloud_spend(config: dict = None) -> str:
    """
    Retrieves the current month-to-date cloud spend from the billing provider.
    (This is a mock tool and returns generated data).
    """
    logger.info("Fetching mock cloud spend data.")
    
    # Simulate a breakdown of costs by service
    services = ["QuantumPulse", "WebAppQ", "ManagerQ", "VectorStoreQ", "KnowledgeGraphQ"]
    spend_data = {
        "report_date": datetime.utcnow().isoformat(),
        "total_spend_usd": round(random.uniform(5000, 7500), 2),
        "spend_by_service": [
            {"service": s, "cost_usd": round(random.uniform(800, 1500), 2)} for s in services
        ]
    }
    # Add a random cost spike for the agent to find
    spend_data["spend_by_service"][random.randint(0, len(services)-1)]["cost_usd"] *= 2.5 

    return json.dumps(spend_data, indent=2)

def get_llm_usage_costs(config: dict = None) -> str:
    """
    Retrieves the current month-to-date LLM API usage costs.
    (This is a mock tool and returns generated data).
    """
    logger.info("Fetching mock LLM usage costs.")
    
    # Simulate a breakdown of costs by model
    models = ["openai-gpt4", "anthropic-claude3-opus", "google-gemini-pro"]
    cost_data = {
        "report_date": datetime.utcnow().isoformat(),
        "total_cost_usd": round(random.uniform(2000, 3000), 2),
        "cost_by_model": [
            {"model": m, "cost_usd": round(random.uniform(500, 1000), 2), "total_requests": random.randint(10000, 50000)} for m in models
        ]
    }
    # Add a random cost spike for the agent to find
    cost_data["cost_by_model"][random.randint(0, len(models)-1)]["cost_usd"] *= 3

    return json.dumps(cost_data, indent=2)

def get_venture_pnl_summary(config: dict = None) -> str:
    """
    Retrieves a summary of the Profit & Loss for autonomous ventures.
    (This is a mock tool and returns generated data).
    """
    logger.info("Fetching Venture P&L summary.")
    
    revenue = random.uniform(500, 2000)
    costs = revenue * random.uniform(0.2, 0.6) # Costs are a % of revenue
    profit = revenue - costs
    
    summary = {
        "total_ventures": random.randint(5, 15),
        "total_revenue_usd": round(revenue, 2),
        "total_cost_usd": round(costs, 2),
        "net_profit_usd": round(profit, 2)
    }
    return json.dumps(summary)

cloud_spend_tool = Tool(
    name="get_cloud_spend",
    description="Retrieves the current month-to-date cloud spend, broken down by service.",
    func=get_cloud_spend
)

llm_usage_tool = Tool(
    name="get_llm_usage_costs",
    description="Retrieves the current month-to-date LLM API usage costs, broken down by model.",
    func=get_llm_usage_costs
)

venture_pnl_tool = Tool(
    name="get_venture_pnl_summary",
    description="Retrieves a Profit & Loss (P&L) summary for completed autonomous ventures.",
    func=get_venture_pnl_summary
)

finops_tools = [cloud_spend_tool, llm_usage_tool, venture_pnl_tool] 