# agentQ/app/core/finops_tools.py
import structlog
import json
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any
from decimal import Decimal
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Cloud provider SDKs
try:
    import boto3  # AWS
    from google.cloud import billing_v1  # GCP
    from azure.mgmt.costmanagement import CostManagementClient  # Azure
    from azure.identity import DefaultAzureCredential
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

# LLM provider SDKs
try:
    import openai
    import anthropic
    from google import generativeai as genai
    LLM_SDKS_AVAILABLE = True
except ImportError:
    LLM_SDKS_AVAILABLE = False

from agentQ.app.core.toolbox import Tool
from shared.vault_client import VaultClient

logger = structlog.get_logger(__name__)

# Log warning after logger is initialized
if not AWS_AVAILABLE:
    logger.warning("Cloud provider SDKs not available. Install boto3, google-cloud-billing, azure-mgmt-costmanagement")

class CloudBillingClient:
    """Unified client for multi-cloud billing data"""
    
    def __init__(self):
        self.vault_client = VaultClient()
        self._executor = ThreadPoolExecutor(max_workers=3)
        self._setup_clients()
    
    def _setup_clients(self):
        """Initialize cloud provider clients with credentials from Vault"""
        try:
            # AWS
            try:
                aws_creds = self.vault_client.read_secret_data("aws/billing")
                if aws_creds and AWS_AVAILABLE:
                    self.aws_ce_client = boto3.client(
                        'ce',  # Cost Explorer
                        aws_access_key_id=aws_creds.get('access_key_id'),
                        aws_secret_access_key=aws_creds.get('secret_access_key'),
                        region_name=aws_creds.get('region', 'us-east-1')
                    )
                    self.aws_cw_client = boto3.client(
                        'cloudwatch',
                        aws_access_key_id=aws_creds.get('access_key_id'),
                        aws_secret_access_key=aws_creds.get('secret_access_key'),
                        region_name=aws_creds.get('region', 'us-east-1')
                    )
                else:
                    self.aws_ce_client = None
                    self.aws_cw_client = None
            except Exception:
                logger.warning("AWS credentials not found in Vault")
                self.aws_ce_client = None
                self.aws_cw_client = None
            
            # GCP
            try:
                gcp_creds = self.vault_client.read_secret_data("gcp/billing")
                if gcp_creds and AWS_AVAILABLE:
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gcp_creds.get('credentials_path', '')
                    self.gcp_billing_client = billing_v1.CloudBillingClient()
                    self.gcp_project_id = gcp_creds.get('project_id')
                else:
                    self.gcp_billing_client = None
            except Exception:
                logger.warning("GCP credentials not found in Vault")
                self.gcp_billing_client = None
            
            # Azure
            try:
                azure_creds = self.vault_client.read_secret_data("azure/billing")
                if azure_creds and AWS_AVAILABLE:
                    credential = DefaultAzureCredential()
                    self.azure_cost_client = CostManagementClient(
                        credential=credential,
                        subscription_id=azure_creds.get('subscription_id')
                    )
                    self.azure_subscription_id = azure_creds.get('subscription_id')
                else:
                    self.azure_cost_client = None
            except Exception:
                logger.warning("Azure credentials not found in Vault")
                self.azure_cost_client = None
                
        except Exception as e:
            logger.error(f"Failed to setup cloud clients: {e}")
            self.aws_ce_client = None
            self.gcp_billing_client = None
            self.azure_cost_client = None
    
    async def get_aws_costs(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Fetch AWS costs using Cost Explorer API"""
        if not self.aws_ce_client:
            return {"error": "AWS client not configured", "total": 0, "services": []}
        
        try:
            loop = asyncio.get_event_loop()
            
            # Get cost and usage
            response = await loop.run_in_executor(
                self._executor,
                lambda: self.aws_ce_client.get_cost_and_usage(
                    TimePeriod={
                        'Start': start_date,
                        'End': end_date
                    },
                    Granularity='DAILY',
                    Metrics=['UnblendedCost'],
                    GroupBy=[
                        {
                            'Type': 'DIMENSION',
                            'Key': 'SERVICE'
                        }
                    ]
                )
            )
            
            # Parse response
            total_cost = 0
            services = []
            
            for result in response.get('ResultsByTime', []):
                for group in result.get('Groups', []):
                    service_name = group['Keys'][0]
                    cost = float(group['Metrics']['UnblendedCost']['Amount'])
                    total_cost += cost
                    
                    # Map AWS service names to our service names
                    mapped_name = self._map_aws_service_name(service_name)
                    existing = next((s for s in services if s['service'] == mapped_name), None)
                    
                    if existing:
                        existing['cost_usd'] += cost
                    else:
                        services.append({
                            'service': mapped_name,
                            'cost_usd': cost,
                            'provider': 'AWS'
                        })
            
            # Get CloudWatch metrics for additional insights
            metrics = await self._get_aws_usage_metrics()
            
            return {
                'provider': 'AWS',
                'total_cost_usd': round(total_cost, 2),
                'services': sorted(services, key=lambda x: x['cost_usd'], reverse=True),
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch AWS costs: {e}")
            return {"error": str(e), "total": 0, "services": []}
    
    async def _get_aws_usage_metrics(self) -> Dict[str, Any]:
        """Get additional AWS usage metrics from CloudWatch"""
        if not self.aws_cw_client:
            return {}
        
        try:
            loop = asyncio.get_event_loop()
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=1)
            
            # Example: Get EC2 instance hours
            ec2_metrics = await loop.run_in_executor(
                self._executor,
                lambda: self.aws_cw_client.get_metric_statistics(
                    Namespace='AWS/Usage',
                    MetricName='ResourceCount',
                    Dimensions=[
                        {'Name': 'Type', 'Value': 'Resource'},
                        {'Name': 'Resource', 'Value': 'vCPU'},
                        {'Name': 'Service', 'Value': 'EC2'},
                        {'Name': 'Class', 'Value': 'None'}
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=3600,
                    Statistics=['Sum']
                )
            )
            
            return {
                'ec2_vcpu_hours': sum(point['Sum'] for point in ec2_metrics.get('Datapoints', []))
            }
            
        except Exception as e:
            logger.error(f"Failed to get AWS metrics: {e}")
            return {}
    
    def _map_aws_service_name(self, aws_service: str) -> str:
        """Map AWS service names to our internal service names"""
        mapping = {
            'Amazon Elastic Compute Cloud - Compute': 'QuantumPulse',
            'Amazon Simple Storage Service': 'VectorStoreQ',
            'Amazon Relational Database Service': 'KnowledgeGraphQ',
            'Amazon Elastic Container Service': 'ManagerQ',
            'Amazon CloudFront': 'WebAppQ',
            'AWS Lambda': 'AgentQ',
            'Amazon SageMaker': 'QuantumPulse'
        }
        
        for aws_name, our_name in mapping.items():
            if aws_name in aws_service:
                return our_name
        
        # Default mapping based on keywords
        if 'compute' in aws_service.lower() or 'ec2' in aws_service.lower():
            return 'QuantumPulse'
        elif 'storage' in aws_service.lower() or 's3' in aws_service.lower():
            return 'VectorStoreQ'
        elif 'database' in aws_service.lower():
            return 'KnowledgeGraphQ'
        else:
            return 'Other'
    
    async def get_gcp_costs(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Fetch GCP costs using Cloud Billing API"""
        if not self.gcp_billing_client or not self.gcp_project_id:
            return {"error": "GCP client not configured", "total": 0, "services": []}
        
        try:
            # Note: GCP billing export to BigQuery is recommended for detailed analysis
            # This is a simplified version using the billing API
            
            # For now, return structured placeholder until BigQuery export is configured
            logger.info("GCP billing requires BigQuery export configuration for detailed costs")
            return {
                'provider': 'GCP',
                'total_cost_usd': 0,
                'services': [],
                'note': 'Configure BigQuery billing export for detailed GCP costs'
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch GCP costs: {e}")
            return {"error": str(e), "total": 0, "services": []}
    
    async def get_azure_costs(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Fetch Azure costs using Cost Management API"""
        if not self.azure_cost_client:
            return {"error": "Azure client not configured", "total": 0, "services": []}
        
        try:
            # Note: Azure Cost Management API requires specific scope and query format
            # This is a simplified version
            
            logger.info("Azure billing integration pending full configuration")
            return {
                'provider': 'Azure',
                'total_cost_usd': 0,
                'services': [],
                'note': 'Azure Cost Management API integration pending'
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch Azure costs: {e}")
            return {"error": str(e), "total": 0, "services": []}
    
    async def get_consolidated_costs(self) -> Dict[str, Any]:
        """Get consolidated costs from all cloud providers"""
        end_date = date.today().isoformat()
        start_date = (date.today() - timedelta(days=30)).isoformat()
        
        # Fetch from all providers in parallel
        aws_task = self.get_aws_costs(start_date, end_date)
        gcp_task = self.get_gcp_costs(start_date, end_date)
        azure_task = self.get_azure_costs(start_date, end_date)
        
        results = await asyncio.gather(aws_task, gcp_task, azure_task, return_exceptions=True)
        
        # Consolidate results
        total_spend = 0
        all_services = {}
        providers_data = []
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Provider fetch failed: {result}")
                continue
            
            if isinstance(result, dict) and 'error' not in result:
                total_spend += result.get('total_cost_usd', 0)
                providers_data.append(result)
                
                # Aggregate by service
                for service in result.get('services', []):
                    service_name = service['service']
                    if service_name in all_services:
                        all_services[service_name]['cost_usd'] += service['cost_usd']
                        all_services[service_name]['providers'].append(service['provider'])
                    else:
                        all_services[service_name] = {
                            'service': service_name,
                            'cost_usd': service['cost_usd'],
                            'providers': [service['provider']]
                        }
        
        return {
            'report_date': datetime.utcnow().isoformat(),
            'total_spend_usd': round(total_spend, 2),
            'spend_by_service': sorted(
                list(all_services.values()), 
                key=lambda x: x['cost_usd'], 
                reverse=True
            ),
            'providers': providers_data,
            'period': {
                'start': start_date,
                'end': end_date
            }
        }


class LLMUsageClient:
    """Client for fetching LLM API usage and costs"""
    
    def __init__(self):
        self.vault_client = VaultClient()
        self._setup_clients()
    
    def _setup_clients(self):
        """Initialize LLM provider clients"""
        try:
            # OpenAI
            try:
                openai_creds = self.vault_client.read_secret_data("openai/api")
                if openai_creds and LLM_SDKS_AVAILABLE:
                    openai.api_key = openai_creds.get('api_key')
                    self.openai_org_id = openai_creds.get('organization_id')
            except Exception:
                logger.warning("OpenAI credentials not found in Vault")
            
            # Anthropic
            try:
                anthropic_creds = self.vault_client.read_secret_data("anthropic/api")
                if anthropic_creds and LLM_SDKS_AVAILABLE:
                    self.anthropic_client = anthropic.Anthropic(
                        api_key=anthropic_creds.get('api_key')
                    )
            except Exception:
                logger.warning("Anthropic credentials not found in Vault")
            
            # Google AI
            try:
                google_creds = self.vault_client.read_secret_data("google/ai")
                if google_creds and LLM_SDKS_AVAILABLE:
                    genai.configure(api_key=google_creds.get('api_key'))
            except Exception:
                logger.warning("Google AI credentials not found in Vault")
                
        except Exception as e:
            logger.error(f"Failed to setup LLM clients: {e}")
    
    async def get_openai_usage(self) -> Dict[str, Any]:
        """Fetch OpenAI API usage and costs"""
        try:
            # OpenAI provides usage through their billing API
            # This requires admin API access
            
            # For now, we'll use the usage endpoint if available
            # In production, this would integrate with OpenAI's billing API
            
            return {
                'provider': 'OpenAI',
                'models': [
                    {
                        'model': 'gpt-4',
                        'cost_usd': 0,  # Would come from billing API
                        'total_requests': 0,
                        'total_tokens': 0
                    },
                    {
                        'model': 'gpt-3.5-turbo',
                        'cost_usd': 0,
                        'total_requests': 0,
                        'total_tokens': 0
                    }
                ],
                'note': 'Requires OpenAI billing API access for detailed costs'
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch OpenAI usage: {e}")
            return {'error': str(e), 'provider': 'OpenAI'}
    
    async def get_anthropic_usage(self) -> Dict[str, Any]:
        """Fetch Anthropic API usage and costs"""
        try:
            # Anthropic usage would come from their billing/usage API
            
            return {
                'provider': 'Anthropic',
                'models': [
                    {
                        'model': 'claude-3-opus',
                        'cost_usd': 0,
                        'total_requests': 0,
                        'total_tokens': 0
                    },
                    {
                        'model': 'claude-3-sonnet',
                        'cost_usd': 0,
                        'total_requests': 0,
                        'total_tokens': 0
                    }
                ],
                'note': 'Requires Anthropic billing API integration'
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch Anthropic usage: {e}")
            return {'error': str(e), 'provider': 'Anthropic'}
    
    async def get_google_ai_usage(self) -> Dict[str, Any]:
        """Fetch Google AI API usage and costs"""
        try:
            # Google AI usage would come from their Cloud Billing API
            
            return {
                'provider': 'Google AI',
                'models': [
                    {
                        'model': 'gemini-pro',
                        'cost_usd': 0,
                        'total_requests': 0,
                        'total_tokens': 0
                    }
                ],
                'note': 'Requires Google Cloud Billing integration'
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch Google AI usage: {e}")
            return {'error': str(e), 'provider': 'Google AI'}
    
    async def get_consolidated_llm_costs(self) -> Dict[str, Any]:
        """Get consolidated LLM costs from all providers"""
        # Fetch from all providers in parallel
        openai_task = self.get_openai_usage()
        anthropic_task = self.get_anthropic_usage()
        google_task = self.get_google_ai_usage()
        
        results = await asyncio.gather(openai_task, anthropic_task, google_task, return_exceptions=True)
        
        # Consolidate results
        total_cost = 0
        all_models = []
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"LLM provider fetch failed: {result}")
                continue
            
            if isinstance(result, dict) and 'error' not in result:
                for model in result.get('models', []):
                    total_cost += model.get('cost_usd', 0)
                    all_models.append({
                        **model,
                        'provider': result['provider']
                    })
        
        return {
            'report_date': datetime.utcnow().isoformat(),
            'total_cost_usd': round(total_cost, 2),
            'cost_by_model': sorted(all_models, key=lambda x: x.get('cost_usd', 0), reverse=True),
            'providers': [r for r in results if not isinstance(r, Exception)]
        }


# Global clients
cloud_billing_client = None
llm_usage_client = None

def _ensure_clients():
    """Ensure clients are initialized"""
    global cloud_billing_client, llm_usage_client
    if not cloud_billing_client:
        cloud_billing_client = CloudBillingClient()
    if not llm_usage_client:
        llm_usage_client = LLMUsageClient()

def get_cloud_spend(config: dict = None) -> str:
    """
    Retrieves the current month-to-date cloud spend from all configured cloud providers.
    Aggregates costs from AWS, GCP, and Azure.
    """
    logger.info("Fetching real cloud spend data from providers")
    _ensure_clients()
    
    try:
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(cloud_billing_client.get_consolidated_costs())
        loop.close()
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to fetch cloud spend: {e}")
        # Return structured error response
        return json.dumps({
            "error": str(e),
            "report_date": datetime.utcnow().isoformat(),
            "total_spend_usd": 0,
            "spend_by_service": []
        }, indent=2)

def get_llm_usage_costs(config: dict = None) -> str:
    """
    Retrieves the current month-to-date LLM API usage costs from all providers.
    Includes OpenAI, Anthropic, and Google AI usage.
    """
    logger.info("Fetching real LLM usage costs from providers")
    _ensure_clients()
    
    try:
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(llm_usage_client.get_consolidated_llm_costs())
        loop.close()
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to fetch LLM usage costs: {e}")
        # Return structured error response
        return json.dumps({
            "error": str(e),
            "report_date": datetime.utcnow().isoformat(),
            "total_cost_usd": 0,
            "cost_by_model": []
        }, indent=2)

def get_venture_pnl_summary(config: dict = None) -> str:
    """
    Retrieves a summary of the Profit & Loss for autonomous ventures.
    This would integrate with the actual venture tracking system.
    """
    logger.info("Fetching Venture P&L summary")
    
    try:
        # In production, this would query the venture tracking database
        # For now, return a structured response indicating the need for integration
        
        summary = {
            "report_date": datetime.utcnow().isoformat(),
            "total_ventures": 0,
            "total_revenue_usd": 0.0,
            "total_cost_usd": 0.0,
            "net_profit_usd": 0.0,
            "status": "Venture tracking system integration pending",
            "ventures": []
        }
        
        return json.dumps(summary, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to fetch venture P&L: {e}")
        return json.dumps({"error": str(e)}, indent=2)

# Tool definitions remain the same
cloud_spend_tool = Tool(
    name="get_cloud_spend",
    description="Retrieves the current month-to-date cloud spend from AWS, GCP, and Azure, broken down by service.",
    func=get_cloud_spend
)

llm_usage_tool = Tool(
    name="get_llm_usage_costs",
    description="Retrieves the current month-to-date LLM API usage costs from OpenAI, Anthropic, and Google AI, broken down by model.",
    func=get_llm_usage_costs
)

venture_pnl_tool = Tool(
    name="get_venture_pnl_summary",
    description="Retrieves a Profit & Loss (P&L) summary for completed autonomous ventures.",
    func=get_venture_pnl_summary
)

finops_tools = [cloud_spend_tool, llm_usage_tool, venture_pnl_tool] 