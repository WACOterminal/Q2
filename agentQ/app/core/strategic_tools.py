import structlog
import json
import random
from agentQ.app.core.toolbox import Tool

logger = structlog.get_logger(__name__)

def get_finops_summary(config: dict = None) -> str:
    """
    Retrieves a high-level summary of the last FinOps report.
    (This is a mock tool and returns generated data).
    """
    logger.info("Fetching FinOps summary.")
    summary = {
        "total_spend_usd": random.uniform(8000, 12000),
        "highest_cost_service": random.choice(["QuantumPulse", "VectorStoreQ"]),
        "cost_trend_percent": random.uniform(-5, 10)
    }
    return json.dumps(summary)

def get_security_summary(config: dict = None) -> str:
    """
    Retrieves a high-level summary of the last security scan.
    (This is a mock tool and returns generated data).
    """
    logger.info("Fetching Security summary.")
    summary = {
        "active_vulnerabilities": random.randint(2, 10),
        "most_common_vulnerability": random.choice(["Hardcoded Secret", "SQL Injection"]),
        "new_vulnerabilities_since_last_scan": random.randint(0, 3)
    }
    return json.dumps(summary)

def get_rca_summary(config: dict = None) -> str:
    """
    Retrieves a summary of the most recent Root Cause Analysis reports.
    (This is a mock tool and returns generated data).
    """
    logger.info("Fetching RCA summary.")
    summary = {
        "total_incidents_last_24h": random.randint(1, 5),
        "most_frequent_root_cause": random.choice(["Database Connection Exhaustion", "Memory Leak"]),
        "auto_remediated_incidents": random.randint(0, 1)
    }
    return json.dumps(summary)

def get_platform_kpis(config: dict = None) -> str:
    """
    Retrieves key performance indicators (KPIs) from the knowledge graph.
    (This is a mock tool and returns generated data).
    """
    logger.info("Fetching platform KPIs.")
    summary = {
        "workflow_success_rate_percent": random.uniform(95, 99.8),
        "most_used_agent_personality": random.choice(["devops", "data_analyst"]),
        "average_workflow_duration_seconds": random.uniform(120, 300)
    }
    return json.dumps(summary)


# --- Tool Registration ---
finops_summary_tool = Tool(
    name="get_finops_summary",
    description="Provides a high-level summary of the platform's financial performance.",
    func=get_finops_summary
)

security_summary_tool = Tool(
    name="get_security_summary",
    description="Provides a high-level summary of the platform's security posture.",
    func=get_security_summary
)

rca_summary_tool = Tool(
    name="get_rca_summary",
    description="Provides a high-level summary of recent production incidents and their root causes.",
    func=get_rca_summary
)

platform_kpi_tool = Tool(
    name="get_platform_kpis",
    description="Provides key performance indicators about the platform's overall operational efficiency.",
    func=get_platform_kpis
)

strategic_tools = [finops_summary_tool, security_summary_tool, rca_summary_tool, platform_kpi_tool] 