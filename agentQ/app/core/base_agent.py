
# agentQ/app/core/base_agent.py
import logging
import json
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from agentQ.app.core.toolbox import Toolbox
from shared.q_pulse_client.client import QuantumPulseClient
from shared.q_pulse_client.models import QPChatRequest, QPChatMessage

logger = logging.getLogger(__name__)

@dataclass
class ReActStep:
    """Represents a single step in the ReAct loop"""
    step_number: int
    thought: str
    action: str
    action_input: str
    observation: str
    timestamp: datetime

@dataclass
class AgentMemory:
    """Stores agent's conversation memory and reasoning history"""
    conversation_id: str
    messages: List[QPChatMessage]
    react_steps: List[ReActStep]
    context: Dict[str, Any]
    start_time: datetime
    last_update: datetime

class BaseAgent:
    """
    A base class for specialized, single-purpose agents with complete ReAct loop implementation.
    """
    def __init__(self, qpulse_url: str, system_prompt: str, tools: list = [], max_iterations: int = 10):
        self.qpulse_client = QuantumPulseClient(base_url=qpulse_url)
        self.system_prompt = system_prompt
        self.toolbox = Toolbox()
        self.max_iterations = max_iterations
        self.agent_memory: Optional[AgentMemory] = None
        
        # Register tools
        for tool in tools:
            self.toolbox.register_tool(tool)
        
        # ReAct prompting templates
        self.react_prompt_template = """
You are an intelligent agent that can reason step by step and use tools to solve problems.

Available tools:
{tools}

Instructions:
1. First, think about the problem step by step (Thought)
2. Then, decide on an action to take (Action)
3. Provide the action input (Action Input)
4. Observe the results (Observation)
5. Repeat until you can provide a final answer

Format your response EXACTLY as follows:
Thought: [Your reasoning about what to do next]
Action: [The action you want to take - either a tool name or 'Finish']
Action Input: [The input for the action, or your final answer if Action is 'Finish']

Remember:
- Use 'Finish' as the Action when you have a complete answer
- Always provide clear reasoning in your Thought
- Be specific in your Action Input
- Only use available tools
"""

        logger.info(f"BaseAgent initialized with {len(tools)} tools")

    async def run(self, prompt: str, conversation_id: Optional[str] = None) -> str:
        """
        Execute the complete ReAct loop to solve the given problem.
        
        Args:
            prompt: The user's question or task
            conversation_id: Optional conversation ID for memory continuity
            
        Returns:
            The final answer from the agent
        """
        # Initialize agent memory
        if conversation_id is None:
            conversation_id = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        await self._initialize_memory(conversation_id, prompt)
        
        logger.info(f"Starting ReAct loop for conversation: {conversation_id}")
        
        try:
            # Execute the ReAct loop
            final_answer = await self._execute_react_loop(prompt)
            
            # Update memory with completion
            if self.agent_memory:
                self.agent_memory.last_update = datetime.now()
            
            logger.info(f"ReAct loop completed successfully for conversation: {conversation_id}")
            return final_answer
            
        except Exception as e:
            logger.error(f"Error in ReAct loop for conversation {conversation_id}: {e}", exc_info=True)
            return f"I encountered an error while processing your request: {str(e)}"
        finally:
            # Clean up resources
            await self.qpulse_client.close()

    async def _initialize_memory(self, conversation_id: str, initial_prompt: str):
        """Initialize agent memory for the conversation"""
        system_message = QPChatMessage(
            role="system",
            content=self.system_prompt.format(tools=self.toolbox.get_tool_descriptions())
        )
        
        user_message = QPChatMessage(
            role="user",
            content=initial_prompt
        )
        
        self.agent_memory = AgentMemory(
            conversation_id=conversation_id,
            messages=[system_message, user_message],
            react_steps=[],
            context={},
            start_time=datetime.now(),
            last_update=datetime.now()
        )

    async def _execute_react_loop(self, initial_prompt: str) -> str:
        """Execute the main ReAct reasoning loop"""
        step_number = 1
        
        # Build the initial prompt with ReAct formatting
        current_prompt = self.react_prompt_template.format(
            tools=self.toolbox.get_tool_descriptions()
        ) + f"\n\nUser Question: {initial_prompt}\n\nNow, let's think step by step:"
        
        while step_number <= self.max_iterations:
            logger.info(f"ReAct step {step_number}/{self.max_iterations}")
            
            # Get LLM response
            response = await self._get_llm_response(current_prompt)
            
            # Parse the response
            thought, action, action_input = self._parse_react_response(response)
            
            if not thought or not action:
                logger.warning(f"Failed to parse ReAct response at step {step_number}")
                return "I'm having trouble understanding how to proceed. Please try rephrasing your question."
            
            # Log the reasoning step
            logger.info(f"Step {step_number} - Thought: {thought[:100]}...")
            logger.info(f"Step {step_number} - Action: {action}")
            logger.info(f"Step {step_number} - Action Input: {action_input[:100]}...")
            
            # Check if the agent wants to finish
            if action.lower() == "finish":
                logger.info(f"Agent finished at step {step_number}")
                return action_input
            
            # Execute the action
            observation = await self._execute_action(action, action_input)
            
            # Store the step in memory
            react_step = ReActStep(
                step_number=step_number,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                timestamp=datetime.now()
            )
            
            if self.agent_memory:
                self.agent_memory.react_steps.append(react_step)
            
            # Update prompt for next iteration
            current_prompt = self._build_next_prompt(initial_prompt, 
                                                     self.agent_memory.react_steps if self.agent_memory else [])
            
            step_number += 1
        
        # If we've reached max iterations without finishing
        logger.warning(f"ReAct loop reached max iterations ({self.max_iterations})")
        return "I've reached my maximum number of reasoning steps. Let me provide the best answer I can based on what I've learned so far."

    async def _get_llm_response(self, prompt: str) -> str:
        """Get a response from the LLM"""
        # Create messages for the chat completion
        messages = [
            QPChatMessage(role="user", content=prompt)
        ]
        
        request = QPChatRequest(
            model="default",  # Use default model
            messages=messages,
            temperature=0.1,  # Low temperature for more consistent reasoning
            max_tokens=1500
        )
        
        try:
            response = await self.qpulse_client.get_chat_completion(request)
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                logger.error("No response choices returned from LLM")
                return "I'm having trouble generating a response. Please try again."
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            return f"Error communicating with the reasoning engine: {str(e)}"

    def _parse_react_response(self, response: str) -> Tuple[str, str, str]:
        """Parse the LLM response to extract thought, action, and action input"""
        thought = ""
        action = ""
        action_input = ""
        
        try:
            # Extract thought
            thought_match = re.search(r"Thought:\s*(.*?)(?=\n(?:Action:|$))", response, re.DOTALL | re.IGNORECASE)
            if thought_match:
                thought = thought_match.group(1).strip()
            
            # Extract action
            action_match = re.search(r"Action:\s*(.*?)(?=\n(?:Action Input:|$))", response, re.DOTALL | re.IGNORECASE)
            if action_match:
                action = action_match.group(1).strip()
            
            # Extract action input
            action_input_match = re.search(r"Action Input:\s*(.*?)(?=\n(?:Observation:|$))", response, re.DOTALL | re.IGNORECASE)
            if action_input_match:
                action_input = action_input_match.group(1).strip()
            
        except Exception as e:
            logger.error(f"Error parsing ReAct response: {e}")
            logger.debug(f"Response content: {response}")
        
        return thought, action, action_input

    async def _execute_action(self, action: str, action_input: str) -> str:
        """Execute the specified action with the given input"""
        try:
            # Check if it's a tool action
            if action in self.toolbox._tools:
                logger.info(f"Executing tool: {action}")
                
                # Parse action input if it's JSON
                try:
                    if action_input.startswith('{') and action_input.endswith('}'):
                        action_kwargs = json.loads(action_input)
                    else:
                        # If not JSON, treat as a single string argument
                        action_kwargs = {"query": action_input}
                except json.JSONDecodeError:
                    # If JSON parsing fails, treat as string
                    action_kwargs = {"query": action_input}
                
                # Execute the tool
                result = self.toolbox.execute_tool(action, context_manager=None, **action_kwargs)
                return f"Tool execution result: {result}"
            
            else:
                # Unknown action
                available_tools = list(self.toolbox._tools.keys())
                return f"Unknown action '{action}'. Available tools: {available_tools}"
                
        except Exception as e:
            logger.error(f"Error executing action '{action}': {e}")
            return f"Error executing action '{action}': {str(e)}"

    def _build_next_prompt(self, initial_prompt: str, steps: List[ReActStep]) -> str:
        """Build the prompt for the next iteration including all previous steps"""
        prompt = self.react_prompt_template.format(
            tools=self.toolbox.get_tool_descriptions()
        )
        
        prompt += f"\n\nUser Question: {initial_prompt}\n\n"
        
        # Add all previous steps
        for step in steps:
            prompt += f"Thought: {step.thought}\n"
            prompt += f"Action: {step.action}\n"
            prompt += f"Action Input: {step.action_input}\n"
            prompt += f"Observation: {step.observation}\n\n"
        
        prompt += "Now, let's continue thinking step by step:"
        
        return prompt

    def get_reasoning_history(self) -> List[Dict[str, Any]]:
        """Get the agent's reasoning history for debugging or analysis"""
        if not self.agent_memory:
            return []
        
        return [
            {
                "step": step.step_number,
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation,
                "timestamp": step.timestamp.isoformat()
            }
            for step in self.agent_memory.react_steps
        ]

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's memory and performance"""
        if not self.agent_memory:
            return {}
        
        duration = (self.agent_memory.last_update - self.agent_memory.start_time).total_seconds()
        
        return {
            "conversation_id": self.agent_memory.conversation_id,
            "total_steps": len(self.agent_memory.react_steps),
            "duration_seconds": duration,
            "start_time": self.agent_memory.start_time.isoformat(),
            "last_update": self.agent_memory.last_update.isoformat(),
            "tools_used": list(set(step.action for step in self.agent_memory.react_steps if step.action != "Finish")),
            "context_size": len(self.agent_memory.context)
        } 