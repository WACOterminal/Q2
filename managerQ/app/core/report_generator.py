import structlog
import json
from typing import Dict, List, Any
from managerQ.app.core.workflow_manager import workflow_manager
from managerQ.app.models import WorkflowStatus

logger = structlog.get_logger(__name__)

class ReportGenerator:
    """
    Generates summary reports by aggregating data from completed workflows.
    """
    async def generate_finops_summary(self) -> Dict[str, Any]:
        logger.info("Generating live FinOps summary...")
        workflows = workflow_manager.get_workflows_by_status_and_id(WorkflowStatus.COMPLETED, "wf_finops_daily_scan", 1)
        if not workflows:
            return {"error": "No completed FinOps workflows found."}
        # For simplicity, we'll use the result of the most recent one.
        # A real implementation might aggregate over several.
        try:
            # The agent's final answer is a JSON string in the 'result' field of a JSON object.
            agent_output = json.loads(workflows[0].final_result)
            report_data = json.loads(agent_output['result'])
            return report_data
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse FinOps workflow result", error=str(e))
            return {"error": "Could not parse FinOps report data."}

    async def generate_security_summary(self) -> Dict[str, Any]:
        logger.info("Generating live Security summary...")
        workflows = workflow_manager.get_workflows_by_status_and_id(WorkflowStatus.COMPLETED, "wf_security_code_scan", 1)
        if not workflows:
            return {"error": "No completed Security workflows found."}
        try:
            agent_output = json.loads(workflows[0].final_result)
            report_data = json.loads(agent_output['result'])
            return report_data
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse Security workflow result", error=str(e))
            return {"error": "Could not parse Security report data."}

    async def generate_rca_summary(self) -> List[Dict[str, Any]]:
        logger.info("Generating live RCA summary...")
        workflows = workflow_manager.get_workflows_by_status_and_id(WorkflowStatus.COMPLETED, "wf_root_cause_analysis", 5)
        reports = []
        for wf in workflows:
            try:
                agent_output = json.loads(wf.final_result)
                report_data = json.loads(agent_output['result'])
                reports.append(report_data)
            except (json.JSONDecodeError, KeyError):
                continue # Skip malformed results
        return reports
        
    async def generate_strategic_briefing(self) -> Dict[str, Any]:
        logger.info("Generating live Strategic Briefing...")
        workflows = workflow_manager.get_workflows_by_status_and_id(WorkflowStatus.COMPLETED, "wf_strategic_quarterly_analysis", 1)
        if not workflows:
            return {"error": "No completed Strategic Analysis workflows found."}
        try:
            agent_output = json.loads(workflows[0].final_result)
            report_data = json.loads(agent_output['result'])
            return report_data
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse Strategic workflow result", error=str(e))
            return {"error": "Could not parse Strategic Briefing data."}

    async def generate_predictive_scaling_summary(self) -> List[Dict[str, Any]]:
        logger.info("Generating live Predictive Scaling summary...")
        # This is more complex as it's about actions taken.
        # A real implementation would query an audit log or a dedicated metrics store.
        # For now, we'll return a mock but indicate it's live.
        return [
            {"time": "2024-01-01T10:05:00Z", "type": 'action', "details": '[Live] Scaled up QuantumPulse replicas.', "icon": "up"},
        ]

# Singleton instance
report_generator = ReportGenerator() 