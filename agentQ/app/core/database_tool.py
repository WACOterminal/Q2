
import logging
import duckdb
from typing import List, Dict, Any

from agentQ.app.core.toolbox import Tool

logger = logging.getLogger(__name__)

# In a real production system, this would use a proper SQL client (psycopg2, mysql-connector, etc.)
# and connect to the managerQ database replica.
# For self-contained demonstration, we'll use DuckDB to query a local file representation.
# This assumes the workflow data is exported to a file like 'workflows.json'.
DB_PATH = "workspace/managerq_export.duckdb"

def query_workflows(sql_query: str, config: dict = {}) -> str:
    """
    Executes a read-only SQL query against the historical workflow database.
    
    Args:
        sql_query (str): The SQL query to execute.
        
    Returns:
        A JSON string representing the query results, or an error message.
    """
    logger.info(f"Executing workflow DB query: {sql_query}")

    # For security, ensure the query is read-only
    if not sql_query.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed."

    try:
        con = duckdb.connect(database=DB_PATH, read_only=True)
        results = con.execute(sql_query).fetchdf()
        con.close()
        
        # Convert dataframe to JSON
        return results.to_json(orient="records")

    except Exception as e:
        logger.error(f"Failed to execute workflow DB query: {e}", exc_info=True)
        return f"Error: An unexpected error occurred while querying the database: {e}"


# --- Tool Registration Object ---
query_database_tool = Tool(
    name="query_workflow_database",
    description="Executes a read-only SQL query against the historical workflow database to analyze past performance, find failed tasks, or identify trends.",
    func=query_workflows
) 