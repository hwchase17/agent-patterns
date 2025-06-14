"""
Main orchestrator agent that manages background agents and user interaction.
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

from .state import OrchestratorState, BackgroundAgent, AgentStatus
from .remote_client import RemoteAgentClient


logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Main orchestrator agent that manages background agents while maintaining user interaction.
    
    The agent operates in a loop where each iteration:
    1. Checks status of all background agents
    2. Processes user input if available
    3. Can spawn new background agents based on user requests
    4. Provides updates on background agent progress
    """
    
    def __init__(self, agent_endpoints: Dict[str, str] = None):
        """
        Initialize the orchestrator agent.
        
        Args:
            agent_endpoints: Dictionary mapping agent names to their endpoints
        """
        self.agent_endpoints = agent_endpoints or {}
        self.remote_client = None
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph for the orchestrator."""
        
        # Create the graph
        workflow = StateGraph(OrchestratorState)
        
        # Add nodes
        workflow.add_node("monitor_agents", self._monitor_agents_node)
        workflow.add_node("process_user_input", self._process_user_input_node)
        workflow.add_node("spawn_agent", self._spawn_agent_node)
        workflow.add_node("respond_to_user", self._respond_to_user_node)
        
        # Set entry point
        workflow.set_entry_point("monitor_agents")
        
        # Add edges
        workflow.add_edge("monitor_agents", "process_user_input")
        workflow.add_conditional_edges(
            "process_user_input",
            self._should_spawn_agent,
            {
                "spawn": "spawn_agent",
                "respond": "respond_to_user"
            }
        )
        workflow.add_edge("spawn_agent", "respond_to_user")
        workflow.add_conditional_edges(
            "respond_to_user",
            self._should_continue,
            {
                "continue": "monitor_agents",
                "end": END
            }
        )
        
        return workflow.compile()
    
    async def _monitor_agents_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Node that monitors the status of all background agents.
        This runs on every iteration to keep agent statuses up to date.
        """
        logger.info(f"Monitoring {len(state.background_agents)} background agents")
        
        # Update iteration count
        state.iteration_count += 1
        state.last_status_check = datetime.now()
        
        # Get active agents that need status checking
        active_agents = state.get_active_agents()
        
        if not active_agents:
            logger.debug("No active agents to monitor")
            return {"iteration_count": state.iteration_count, "last_status_check": state.last_status_check}
        
        # Check status of all active agents
        if not self.remote_client:
            self.remote_client = RemoteAgentClient()
        
        try:
            async with self.remote_client as client:
                status_updates = await client.check_multiple_agents(active_agents)
                
                # Log status changes
                for thread_id, new_status in status_updates.items():
                    agent = state.background_agents[thread_id]
                    if agent.status != new_status:
                        logger.info(f"Agent {thread_id} status changed: {agent.status} -> {new_status}")
                
        except Exception as e:
            logger.error(f"Error monitoring agents: {e}")
        
        # Prepare status summary for the response
        status_summary = state.get_status_summary()
        logger.info(f"Agent status summary: {status_summary}")
        
        return {
            "iteration_count": state.iteration_count,
            "last_status_check": state.last_status_check
        }
    
    async def _process_user_input_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Node that processes user input and determines what action to take.
        """
        # Get the latest user message
        user_message = None
        if state.messages:
            last_message = state.messages[-1]
            if isinstance(last_message, HumanMessage):
                user_message = last_message.content
                state.user_input = user_message
        
        if not user_message:
            logger.debug("No new user input to process")
            return {"user_input": state.user_input}
        
        logger.info(f"Processing user input: {user_message}")
        
        # Analyze user input to determine intent
        user_input_lower = user_message.lower()
        
        # Check if user wants to spawn an agent
        spawn_keywords = ["start", "run", "execute", "spawn", "create", "launch"]
        agent_keywords = ["agent", "task", "background", "async"]
        
        wants_to_spawn = any(keyword in user_input_lower for keyword in spawn_keywords) and \
                        any(keyword in user_input_lower for keyword in agent_keywords)
        
        return {
            "user_input": user_message,
            "wants_to_spawn_agent": wants_to_spawn
        }
    
    async def _spawn_agent_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Node that spawns a new background agent based on user request.
        """
        if not state.user_input:
            return {}
        
        # Check if we can spawn more agents
        if not state.can_spawn_agent():
            active_count = len(state.get_active_agents())
            logger.warning(f"Cannot spawn agent: {active_count}/{state.max_concurrent_agents} agents active")
            return {
                "spawn_error": f"Maximum concurrent agents ({state.max_concurrent_agents}) reached"
            }
        
        # For this example, we'll use a default agent endpoint
        # In a real implementation, this would be determined from user input
        agent_name = "default_agent"
        agent_endpoint = self.agent_endpoints.get(agent_name, "http://localhost:8001")
        
        # Extract task description from user input
        task_description = state.user_input
        
        # Prepare input data
        input_data = {
            "task": task_description,
            "user_request": state.user_input,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            if not self.remote_client:
                self.remote_client = RemoteAgentClient()
            
            async with self.remote_client as client:
                background_agent = await client.spawn_agent(
                    agent_endpoint=agent_endpoint,
                    agent_name=agent_name,
                    task_description=task_description,
                    input_data=input_data,
                    metadata={"spawned_by": "orchestrator", "user_request": state.user_input}
                )
                
                # Add to state
                state.add_background_agent(background_agent)
                
                logger.info(f"Successfully spawned agent {background_agent.thread_id}")
                
                return {
                    "spawned_agent_id": background_agent.thread_id,
                    "spawned_agent_name": agent_name
                }
                
        except Exception as e:
            logger.error(f"Failed to spawn agent: {e}")
            return {"spawn_error": str(e)}
    
    async def _respond_to_user_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """
        Node that generates a response to the user based on current state.
        """
        # Build response based on what happened
        response_parts = []
        
        # Status update on background agents
        status_summary = state.get_status_summary()
        active_count = status_summary.get("running", 0) + status_summary.get("pending", 0)
        completed_count = status_summary.get("completed", 0)
        failed_count = status_summary.get("failed", 0)
        
        if active_count > 0 or completed_count > 0 or failed_count > 0:
            status_msg = f"Background agents status: {active_count} active, {completed_count} completed, {failed_count} failed"
            response_parts.append(status_msg)
        
        # Handle spawn results
        if hasattr(state, 'spawned_agent_id') and state.spawned_agent_id:
            response_parts.append(f"✅ Started background agent {state.spawned_agent_id}")
        elif hasattr(state, 'spawn_error') and state.spawn_error:
            response_parts.append(f"❌ Failed to start agent: {state.spawn_error}")
        
        # Handle completed agents
        completed_agents = state.get_completed_agents()
        for agent in completed_agents:
            if agent.result:
                response_parts.append(f"✅ Agent {agent.thread_id} completed: {agent.result}")
        
        # Handle failed agents
        failed_agents = state.get_failed_agents()
        for agent in failed_agents:
            response_parts.append(f"❌ Agent {agent.thread_id} failed: {agent.error_message}")
        
        # Default response if user input but no specific action
        if state.user_input and not response_parts:
            response_parts.append("I'm monitoring your background agents. How can I help you?")
        
        # Combine response parts
        if response_parts:
            response_text = "
        else:
            response_text = f"Monitoring {active_count} background agents..."
        
        # Add AI message to conversation
        ai_message = AIMessage(content=response_text)
        
        return {
            "messages": state.messages + [ai_message]
        }
    
    def _should_spawn_agent(self, state: OrchestratorState) -> str:
        """Conditional edge function to determine if we should spawn an agent."""
        return "spawn" if getattr(state, 'wants_to_spawn_agent', False) else "respond"
    
    def _should_continue(self, state: OrchestratorState) -> str:
        """Conditional edge function to determine if we should continue the loop."""
        # Continue if there are active agents or if we're in interactive mode
        active_agents = state.get_active_agents()
        return "continue" if len(active_agents) > 0 else "end"
    
    async def run(self, initial_message: str = None, max_iterations: int = 100) -> OrchestratorState:
        """
        Run the orchestrator agent.
        
        Args:
            initial_message: Optional initial message to start the conversation
            max_iterations: Maximum number of iterations to run
            
        Returns:
            Final state of the orchestrator
        """
        # Initialize state
        initial_state = OrchestratorState()
        
        if initial_message:
            initial_state.messages = [HumanMessage(content=initial_message)]
        
        # Run the graph
        try:
            final_state = await self.graph.ainvoke(initial_state)
            return final_state
        except Exception as e:
            logger.error(f"Error running orchestrator: {e}")
            raise
    
    async def stream_run(self, initial_message: str = None):
        """
        Run the orchestrator agent with streaming output.
        
        Args:
            initial_message: Optional initial message to start the conversation
            
        Yields:
            State updates as they occur
        """
        # Initialize state
        initial_state = OrchestratorState()
        
        if initial_message:
            initial_state.messages = [HumanMessage(content=initial_message)]
        
        # Stream the graph execution
        async for state_update in self.graph.astream(initial_state):
            yield state_update

