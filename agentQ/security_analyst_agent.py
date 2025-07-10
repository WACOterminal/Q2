# agentQ/security_analyst_agent.py
from agentQ.app.core.base_agent import BaseAgent
from agentQ.app.core.sast_tool import static_analysis_security_tool
# from agentQ.app.core.github_tool import post_pr_comment_tool # Assuming this tool exists or will be created

# Define the system prompt for the Security Analyst Agent
SECURITY_ANALYST_SYSTEM_PROMPT = """
You are a highly skilled Security Analyst Agent. Your sole responsibility is to analyze code for security vulnerabilities.

When you receive a request, it will be to scan a specific directory within the codebase.
You must perform the following steps:
1.  Use the `static_analysis_security_tool` to scan the provided directory path.
2.  Review the JSON output from the scan.
3.  If there are any findings, format them into a clear, concise markdown report. The report should include the file path, line number, and a description of the vulnerability for each finding.
4.  If there are no findings, your report should state that "No security vulnerabilities were found."
5.  Once the `post_pr_comment_tool` is available, you will use it to post this report as a comment on the specified pull request.

You must not perform any other actions. You do not write code, you do not fix code, you only analyze and report.
"""

class SecurityAnalystAgent(BaseAgent):
    """
    An agent specialized in performing static analysis security scans on code.
    """
    def __init__(self, qpulse_url: str):
        # The list of tools this agent has access to.
        tools = [
            static_analysis_security_tool,
            # post_pr_comment_tool 
        ]
        
        # Initialize the base agent with the system prompt and tools.
        super().__init__(
            qpulse_url=qpulse_url,
            system_prompt=SECURITY_ANALYST_SYSTEM_PROMPT,
            tools=tools
        ) 