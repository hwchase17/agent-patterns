"""
Multi-Agent Management Chat System

This module implements a management agent that can coordinate and monitor
multiple remote agents deployed on LangGraph Platform. It provides capabilities
for task delegation, progress monitoring, and agent control.
"""

import os
import uuid
import asyncio
from typing import Annotated, Dict, List, Optional, Any
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command, Send
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_client, get_sync_client
from langgraph_sdk.schema import RunStatus, ThreadStatus
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentManagementState(MessagesState):
    """Extended state for agent management with tracking capabilities."""
    
    # Track active remote agents and their runs
    active_runs: Dict[str, Dict[str, Any]] = {}
    
    # Track agent configurations
    remote_agents: Dict[str, Dict[str, Any]] = {}
    
    # Current operation context
    current_operation: Optional[str] = None
    
    # Progress tracking
    progress_updates: List[Dict[str, Any]] = []



class MultiAgentManager:
    """
    Multi-Agent Management System
    
    Coordinates multiple remote agents deployed on LangGraph Platform,
    providing task delegation, monitoring, and control capabilities.
    """
    
    def __init__(
        self,
        platform_url: Optional[str] = None,
        api_key: Optional[str] = None,
        remote_agents_config: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize the Multi-Agent Manager.
        
        Args:
            platform_url: URL of the LangGraph Platform deployment
            api_key: LangSmith API key for authentication
            remote_agents_config: Configuration for remote agents
        """
        self.platform_url = platform_url or os.getenv("LANGGRAPH_PLATFORM_URL")
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY")
        
        # Initialize SDK clients
        if self.platform_url:
            self.client = get_client(url=self.platform_url, api_key=self.api_key)
            self.sync_client = get_sync_client(url=self.platform_url, api_key=self.api_key)
        else:
            self.client = None
            self.sync_client = None
        
        # Default remote agents configuration
        self.remote_agents_config = remote_agents_config or {
            "research_agent": {
                "graph_name": "research_agent",
                "description": "Handles research and information gathering tasks",
                "capabilities": ["web_search", "data_analysis", "fact_checking"]
            },
            "data_agent": {
                "graph_name": "data_agent", 
                "description": "Processes and analyzes data",
                "capabilities": ["data_processing", "calculations", "statistics"]
            },
            "writing_agent": {
                "graph_name": "writing_agent",
                "description": "Handles content creation and writing tasks",
                "capabilities": ["content_writing", "summarization", "editing"]
            }
        }
        
        # Initialize remote graph connections
        self.remote_graphs = {}
        self._initialize_remote_graphs()
        
        # Build the management graph
        self.graph = self._build_management_graph()
    
    def _initialize_remote_graphs(self):
        """Initialize RemoteGraph connections for each configured agent."""
        if not self.platform_url:
            print("Warning: No platform URL provided. Remote graphs will not be initialized.")
            return
            
        for agent_name, config in self.remote_agents_config.items():
            try:
                # Initialize RemoteGraph with both sync and async clients for full functionality
                if self.client and self.sync_client:
                    remote_graph = RemoteGraph(
                        name=config["graph_name"],
                        client=self.client,
                        sync_client=self.sync_client,
                        api_key=self.api_key
                    )
                else:
                    # Fallback to URL-based initialization
                    remote_graph = RemoteGraph(
                        name=config["graph_name"],
                        url=self.platform_url,
                        api_key=self.api_key
                    )
                
                # Wrap the remote graph with enhanced functionality
                enhanced_graph = self._create_enhanced_remote_graph(agent_name, remote_graph, config)
                self.remote_graphs[agent_name] = enhanced_graph
                print(f"Initialized remote graph for {agent_name}")
                
            except Exception as e:
                print(f"Failed to initialize remote graph for {agent_name}: {e}")
                # Create a fallback placeholder
                self.remote_graphs[agent_name] = self._create_fallback_remote_graph(agent_name, config)
    
    def _create_enhanced_remote_graph(self, agent_name: str, remote_graph: RemoteGraph, config: Dict[str, Any]):
        """Create an enhanced remote graph wrapper with additional functionality."""
        
        def enhanced_remote_node(state: AgentManagementState):
            """Enhanced remote graph node with proper state management and error handling."""
            try:
                # Extract the task information from state
                messages = state.get("messages", [])
                run_id = state.get("run_id")
                priority = state.get("priority", "medium")
                
                # Create thread for persistence if we have SDK clients
                thread_config = None
                if self.sync_client and run_id:
                    try:
                        # Create or get existing thread
                        thread = self.sync_client.threads.create(
                            metadata={"agent_name": agent_name, "run_id": run_id, "priority": priority}
                        )
                        thread_config = {"configurable": {"thread_id": thread["thread_id"]}}
                        print(f"Created thread {thread['thread_id']} for {agent_name} run {run_id}")
                    except Exception as e:
                        print(f"Failed to create thread for {agent_name}: {e}")
                
                # Invoke the remote graph with proper configuration
                if thread_config:
                    result = remote_graph.invoke(
                        {"messages": messages},
                        config=thread_config
                    )
                else:
                    result = remote_graph.invoke({"messages": messages})
                
                # Update state with results
                updated_state = {
                    "messages": result.get("messages", []),
                    "current_operation": f"completed_{agent_name}",
                }
                
                # Add progress update
                if "progress_updates" not in updated_state:
                    updated_state["progress_updates"] = state.get("progress_updates", [])
                
                updated_state["progress_updates"].append({
                    "timestamp": datetime.now().isoformat(),
                    "event": "task_completed",
                    "agent": agent_name,
                    "run_id": run_id,
                    "message": f"Task completed by {agent_name}",
                    "thread_id": thread_config["configurable"]["thread_id"] if thread_config else None
                })
                
                return updated_state
                
            except Exception as e:
                print(f"Error in enhanced remote node for {agent_name}: {e}")
                # Return error state
                return {
                    "messages": [
                        AIMessage(content=f"Error executing task on {agent_name}: {str(e)}")
                    ],
                    "current_operation": f"error_{agent_name}",
                    "progress_updates": state.get("progress_updates", []) + [{
                        "timestamp": datetime.now().isoformat(),
                        "event": "task_error",
                        "agent": agent_name,
                        "run_id": state.get("run_id"),
                        "message": f"Error in {agent_name}: {str(e)}"
                    }]
                }
        
        return enhanced_remote_node
    
    def _create_fallback_remote_graph(self, agent_name: str, config: Dict[str, Any]):
        """Create a fallback node when RemoteGraph initialization fails."""
        def fallback_node(state: AgentManagementState):
            print(f"Using fallback node for {agent_name}")
            return {
                "messages": [
                    AIMessage(content=f"[Fallback Mode] {agent_name} is not available. "
                                    f"This would normally handle: {config['description']}")
                ],
                "current_operation": f"fallback_{agent_name}",
                "progress_updates": state.get("progress_updates", []) + [{
                    "timestamp": datetime.now().isoformat(),
                    "event": "fallback_used",
                    "agent": agent_name,
                    "run_id": state.get("run_id"),
                    "message": f"Fallback mode activated for {agent_name}"
                }]
            }
        return fallback_node
    
    async def _invoke_remote_graph_async(self, agent_name: str, input_data: Dict[str, Any], 
                                       thread_config: Optional[Dict[str, Any]] = None):
        """Asynchronously invoke a remote graph."""
        if agent_name not in self.remote_graphs:
            raise ValueError(f"Remote graph {agent_name} not found")
        
        remote_graph = self.remote_graphs[agent_name]
        
        try:
            if hasattr(remote_graph, 'ainvoke'):
                if thread_config:
                    result = await remote_graph.ainvoke(input_data, config=thread_config)
                else:
                    result = await remote_graph.ainvoke(input_data)
            else:
                # Fallback to sync invoke in thread pool
                loop = asyncio.get_event_loop()
                if thread_config:
                    result = await loop.run_in_executor(
                        None, lambda: remote_graph.invoke(input_data, config=thread_config)
                    )
                else:
                    result = await loop.run_in_executor(
                        None, lambda: remote_graph.invoke(input_data)
                    )
            return result
        except Exception as e:
            print(f"Error invoking remote graph {agent_name}: {e}")
            raise
    
    def _invoke_remote_graph_sync(self, agent_name: str, input_data: Dict[str, Any],
                                thread_config: Optional[Dict[str, Any]] = None):
        """Synchronously invoke a remote graph."""
        if agent_name not in self.remote_graphs:
            raise ValueError(f"Remote graph {agent_name} not found")
        
        remote_graph = self.remote_graphs[agent_name]
        
        try:
            if thread_config:
                result = remote_graph.invoke(input_data, config=thread_config)
            else:
                result = remote_graph.invoke(input_data)
            return result
        except Exception as e:
            print(f"Error invoking remote graph {agent_name}: {e}")
            raise
    
    def get_remote_graph_status(self, agent_name: str) -> Dict[str, Any]:
        """Get the status of a remote graph connection."""
        if agent_name not in self.remote_graphs:
            return {"status": "not_found", "message": f"Agent {agent_name} not configured"}
        
        try:
            # Check if the remote graph is available and properly initialized
            remote_graph = self.remote_graphs[agent_name]
            if hasattr(remote_graph, '__call__'):
                # This is our enhanced wrapper function
                return {
                    "status": "connected", 
                    "message": f"Remote graph {agent_name} is available",
                    "graph_name": self.remote_agents_config[agent_name]["graph_name"]
                }
            else:
                return {
                    "status": "fallback",
                    "message": f"Remote graph {agent_name} is in fallback mode"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error checking {agent_name}: {str(e)}"
            }
    
    def list_available_agents(self) -> Dict[str, Dict[str, Any]]:
        """List all available remote agents and their status."""
        agents_status = {}
        for agent_name in self.remote_agents_config.keys():
            agents_status[agent_name] = {
                **self.remote_agents_config[agent_name],
                "connection_status": self.get_remote_graph_status(agent_name)
            }
        return agents_status
    
    # Run Management Methods using RunsClient
    
    def create_run(self, agent_name: str, input_data: Dict[str, Any], 
                   thread_id: Optional[str] = None, priority: str = "medium",
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new run for a remote agent using RunsClient."""
        if not self.sync_client:
            raise ValueError("SDK client not initialized. Cannot create runs.")
        
        if agent_name not in self.remote_agents_config:
            raise ValueError(f"Agent {agent_name} not configured")
        
        try:
            # Get the assistant ID for the agent (assuming it matches the graph name)
            assistant_id = self.remote_agents_config[agent_name]["graph_name"]
            
            # Prepare run metadata
            run_metadata = {
                "agent_name": agent_name,
                "priority": priority,
                "created_by": "management_agent",
                **(metadata or {})
            }
            
            # Create the run
            run = self.sync_client.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
                input=input_data,
                metadata=run_metadata
            )
            
            print(f"Created run {run.run_id} for {agent_name} on thread {run.thread_id}")
            return {
                "run_id": run.run_id,
                "thread_id": run.thread_id,
                "assistant_id": run.assistant_id,
                "status": run.status,
                "created_at": run.created_at,
                "agent_name": agent_name
            }
            
        except Exception as e:
            print(f"Error creating run for {agent_name}: {e}")
            raise
    
    def get_run(self, run_id: str) -> Dict[str, Any]:
        """Get details of a specific run using RunsClient."""
        if not self.sync_client:
            raise ValueError("SDK client not initialized. Cannot get run details.")
        
        try:
            run = self.sync_client.runs.get(run_id)
            return {
                "run_id": run.run_id,
                "thread_id": run.thread_id,
                "assistant_id": run.assistant_id,
                "status": run.status,
                "created_at": run.created_at,
                "updated_at": run.updated_at,
                "metadata": run.metadata
            }
        except Exception as e:
            print(f"Error getting run {run_id}: {e}")
            raise
    
    def list_runs(self, thread_id: Optional[str] = None, 
                  limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """List runs using RunsClient with optional filtering."""
        if not self.sync_client:
            raise ValueError("SDK client not initialized. Cannot list runs.")
        
        try:
            runs = self.sync_client.runs.list(
                thread_id=thread_id,
                limit=limit,
                offset=offset
            )
            
            return [
                {
                    "run_id": run.run_id,
                    "thread_id": run.thread_id,
                    "assistant_id": run.assistant_id,
                    "status": run.status,
                    "created_at": run.created_at,
                    "updated_at": run.updated_at,
                    "metadata": run.metadata
                }
                for run in runs
            ]
        except Exception as e:
            print(f"Error listing runs: {e}")
            raise
    
    def cancel_run(self, run_id: str) -> Dict[str, Any]:
        """Cancel a running agent using RunsClient.cancel()."""
        if not self.sync_client:
            raise ValueError("SDK client not initialized. Cannot cancel run.")
        
        try:
            # Cancel the run
            self.sync_client.runs.cancel(run_id)
            
            # Get updated run status
            run = self.sync_client.runs.get(run_id)
            
            print(f"Cancelled run {run_id}. Status: {run.status}")
            return {
                "run_id": run.run_id,
                "status": run.status,
                "updated_at": run.updated_at,
                "cancelled": True
            }
        except Exception as e:
            print(f"Error cancelling run {run_id}: {e}")
            raise
    
    def wait_for_run(self, run_id: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Wait for a run to complete using RunsClient.wait()."""
        if not self.sync_client:
            raise ValueError("SDK client not initialized. Cannot wait for run.")
        
        try:
            # Wait for the run to complete
            run = self.sync_client.runs.wait(run_id, timeout=timeout)
            
            print(f"Run {run_id} completed with status: {run.status}")
            return {
                "run_id": run.run_id,
                "thread_id": run.thread_id,
                "status": run.status,
                "updated_at": run.updated_at,
                "completed": run.status in ["success", "error", "timeout", "interrupted"]
            }
        except Exception as e:
            print(f"Error waiting for run {run_id}: {e}")
            raise
    
    def stream_run(self, run_id: str):
        """Stream run updates using RunsClient.join_stream()."""
        if not self.sync_client:
            raise ValueError("SDK client not initialized. Cannot stream run.")
        
        try:
            # Stream run updates
            for chunk in self.sync_client.runs.join_stream(run_id):
                yield {
                    "run_id": run_id,
                    "event": chunk.event,
                    "data": chunk.data,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            print(f"Error streaming run {run_id}: {e}")
            raise
    
    def get_run_status(self, run_id: str) -> str:
        """Get the current status of a run using RunStatus enum states."""
        try:
            run = self.get_run(run_id)
            return run["status"]
        except Exception as e:
            print(f"Error getting run status for {run_id}: {e}")
            return "unknown"
    
    def monitor_active_runs(self) -> Dict[str, Dict[str, Any]]:
        """Monitor all active runs and return their current status."""
        if not self.sync_client:
            return {}
        
        try:
            # Get recent runs (active ones should be at the top)
            runs = self.list_runs(limit=50)
            
            active_runs = {}
            for run in runs:
                if run["status"] in ["pending", "running"]:
                    active_runs[run["run_id"]] = {
                        "run_id": run["run_id"],
                        "thread_id": run["thread_id"],
                        "assistant_id": run["assistant_id"],
                        "status": run["status"],
                        "created_at": run["created_at"],
                        "updated_at": run["updated_at"],
                        "agent_name": run["metadata"].get("agent_name", "unknown") if run["metadata"] else "unknown"
                    }
            
            return active_runs
        except Exception as e:
            print(f"Error monitoring active runs: {e}")
            return {}
    
    # Thread Management Methods using ThreadsClient
    
    def get_thread_state(self, thread_id: str) -> Dict[str, Any]:
        """Get the current state of a thread using ThreadsClient.get_state()."""
        if not self.sync_client:
            raise ValueError("SDK client not initialized. Cannot get thread state.")
        
        try:
            thread_state = self.sync_client.threads.get_state(thread_id)
            return {
                "thread_id": thread_id,
                "values": thread_state.values,
                "next": thread_state.next,
                "config": thread_state.config,
                "metadata": thread_state.metadata,
                "created_at": thread_state.created_at,
                "parent_config": thread_state.parent_config
            }
        except Exception as e:
            print(f"Error getting thread state for {thread_id}: {e}")
            raise
    
    def update_thread_state(self, thread_id: str, values: Dict[str, Any], 
                           as_node: Optional[str] = None) -> Dict[str, Any]:
        """Update the state of a thread using ThreadsClient.update_state()."""
        if not self.sync_client:
            raise ValueError("SDK client not initialized. Cannot update thread state.")
        
        try:
            updated_state = self.sync_client.threads.update_state(
                thread_id=thread_id,
                values=values,
                as_node=as_node
            )
            
            print(f"Updated thread {thread_id} state")
            return {
                "thread_id": thread_id,
                "values": updated_state.values,
                "next": updated_state.next,
                "config": updated_state.config,
                "metadata": updated_state.metadata,
                "updated": True
            }
        except Exception as e:
            print(f"Error updating thread state for {thread_id}: {e}")
            raise
    
    def get_thread_history(self, thread_id: str, limit: int = 10, 
                          before: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get the history of a thread using ThreadsClient.get_history()."""
        if not self.sync_client:
            raise ValueError("SDK client not initialized. Cannot get thread history.")
        
        try:
            history = self.sync_client.threads.get_history(
                thread_id=thread_id,
                limit=limit,
                before=before
            )
            
            history_list = []
            for state in history:
                history_list.append({
                    "values": state.values,
                    "next": state.next,
                    "config": state.config,
                    "metadata": state.metadata,
                    "created_at": state.created_at,
                    "parent_config": state.parent_config
                })
            
            return history_list
        except Exception as e:
            print(f"Error getting thread history for {thread_id}: {e}")
            raise
    
    def monitor_thread_progress(self, thread_id: str) -> Dict[str, Any]:
        """Monitor the progress of a specific thread by analyzing its state and history."""
        try:
            # Get current thread state
            current_state = self.get_thread_state(thread_id)
            
            # Get recent history to understand progress
            history = self.get_thread_history(thread_id, limit=5)
            
            # Analyze progress
            progress_info = {
                "thread_id": thread_id,
                "current_state": current_state,
                "history_count": len(history),
                "last_updated": current_state.get("created_at"),
                "next_steps": current_state.get("next", []),
                "is_active": len(current_state.get("next", [])) > 0,
                "conversation_length": len(current_state.get("values", {}).get("messages", []))
            }
            
            # Extract key progress indicators
            if history:
                progress_info["recent_activity"] = [
                    {
                        "timestamp": state.get("created_at"),
                        "next_steps": state.get("next", []),
                        "message_count": len(state.get("values", {}).get("messages", []))
                    }
                    for state in history[:3]
                ]
            
            return progress_info
        except Exception as e:
            print(f"Error monitoring thread progress for {thread_id}: {e}")
            return {"thread_id": thread_id, "error": str(e)}
    
    def get_active_threads(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active threads by analyzing recent runs."""
        if not self.sync_client:
            return {}
        
        try:
            # Get recent runs to find active threads
            recent_runs = self.list_runs(limit=50)
            
            active_threads = {}
            thread_ids = set()
            
            # Collect unique thread IDs from recent runs
            for run in recent_runs:
                if run["status"] in ["pending", "running"]:
                    thread_ids.add(run["thread_id"])
            
            # Get progress information for each active thread
            for thread_id in thread_ids:
                try:
                    progress = self.monitor_thread_progress(thread_id)
                    active_threads[thread_id] = progress
                except Exception as e:
                    active_threads[thread_id] = {
                        "thread_id": thread_id,
                        "error": f"Failed to get progress: {str(e)}"
                    }
            
            return active_threads
        except Exception as e:
            print(f"Error getting active threads: {e}")
            return {}
    
    def _create_thread_management_tools(self):
        """Create tools for thread management operations."""
        
        @tool(name="get_thread_state", description="Get the current state of a specific thread")
        def get_thread_state_tool(
            thread_id: Annotated[str, "The thread ID to get state for"]
        ) -> str:
            """Get the current state of a specific thread."""
            try:
                state = self.get_thread_state(thread_id)
                message_count = len(state.get("values", {}).get("messages", []))
                next_steps = state.get("next", [])
                return f"Thread {thread_id}: {message_count} messages, Next steps: {next_steps}"
            except Exception as e:
                return f"Error getting thread state: {str(e)}"
        
        @tool(name="monitor_thread_progress", description="Monitor the progress of a specific thread")
        def monitor_thread_progress_tool(
            thread_id: Annotated[str, "The thread ID to monitor progress for"]
        ) -> str:
            """Monitor the progress of a specific thread."""
            try:
                progress = self.monitor_thread_progress(thread_id)
                if "error" in progress:
                    return f"Error monitoring thread {thread_id}: {progress['error']}"
                
                result = f"Thread {thread_id} Progress:
                result += f"- Active: {progress['is_active']}
                result += f"- Messages: {progress['conversation_length']}
                result += f"- Next steps: {progress['next_steps']}
                result += f"- Last updated: {progress['last_updated']}
                
                if progress.get('recent_activity'):
                    result += "Recent activity:
                    for activity in progress['recent_activity']:
                        result += f"  - {activity['timestamp']}: {activity['message_count']} messages
                
                return result
            except Exception as e:
                return f"Error monitoring thread progress: {str(e)}"
        
        @tool(name="get_active_threads", description="Get information about all active threads")
        def get_active_threads_tool() -> str:
            """Get information about all currently active threads."""
            try:
                active_threads = self.get_active_threads()
                if not active_threads:
                    return "No active threads found."
                
                result = f"Active Threads ({len(active_threads)}):
                for thread_id, info in active_threads.items():
                    if "error" in info:
                        result += f"- {thread_id}: Error - {info['error']}
                    else:
                        result += f"- {thread_id}: {info['conversation_length']} messages, "
                        result += f"Active: {info['is_active']}
                
                return result
            except Exception as e:
                return f"Error getting active threads: {str(e)}"
        
        @tool(name="get_thread_history", description="Get the history of a specific thread")
        def get_thread_history_tool(
            thread_id: Annotated[str, "The thread ID to get history for"],
            limit: Annotated[int, "Number of history entries to retrieve"] = 5
        ) -> str:
            """Get the history of a specific thread."""
            try:
                history = self.get_thread_history(thread_id, limit=limit)
                result = f"Thread {thread_id} History (last {len(history)} entries):
                for i, state in enumerate(history):
                    message_count = len(state.get("values", {}).get("messages", []))
                    result += f"{i+1}. {state.get('created_at', 'N/A')}: {message_count} messages
                return result
            except Exception as e:
                return f"Error getting thread history: {str(e)}"
        
        return [get_thread_state_tool, monitor_thread_progress_tool, get_active_threads_tool, get_thread_history_tool]
    
    def _create_handoff_tool(self, agent_name: str, agent_config: Dict[str, Any]):
        """Create a handoff tool for delegating tasks to a specific remote agent."""
        
        @tool(
            name=f"delegate_to_{agent_name}",
            description=f"Delegate task to {agent_name}. {agent_config['description']}. "
                       f"Capabilities: {', '.join(agent_config['capabilities'])}"
        )
        def handoff_tool(
            task_description: Annotated[str, "Clear description of the task to delegate"],
            priority: Annotated[str, "Task priority: low, medium, high"] = "medium",
            state: Annotated[AgentManagementState, InjectedState] = None,
            tool_call_id: Annotated[str, InjectedToolCallId] = None,
        ) -> Command:
            """Delegate a task to a remote agent."""
            
            # Create task message for the remote agent
            task_message = HumanMessage(content=task_description)
            
            # Generate unique run ID for tracking
            run_id = str(uuid.uuid4())
            
            # Update state with new run tracking
            updated_state = state.copy() if state else {}
            if "active_runs" not in updated_state:
                updated_state["active_runs"] = {}
            
            updated_state["active_runs"][run_id] = {
                "agent_name": agent_name,
                "task_description": task_description,
                "priority": priority,
                "status": "delegated",
                "created_at": datetime.now().isoformat(),
                "tool_call_id": tool_call_id
            }
            
            # Add progress update
            if "progress_updates" not in updated_state:
                updated_state["progress_updates"] = []
            
            updated_state["progress_updates"].append({
                "timestamp": datetime.now().isoformat(),
                "event": "task_delegated",
                "agent": agent_name,
                "run_id": run_id,
                "message": f"Delegated task to {agent_name}: {task_description[:100]}..."
            })
            
            # Create tool response message
            tool_message = {
                "role": "tool",
                "content": f"Task delegated to {agent_name} (Run ID: {run_id}). "
                          f"The agent will process: {task_description}",
                "name": f"delegate_to_{agent_name}",
                "tool_call_id": tool_call_id,
            }
            
            # Update messages
            updated_messages = (state.get("messages", []) if state else []) + [tool_message]
            updated_state["messages"] = updated_messages
            updated_state["current_operation"] = f"delegated_to_{agent_name}"
            
            # Use Send to pass specific task to remote agent
            agent_input = {
                "messages": [task_message],
                "run_id": run_id,
                "priority": priority
            }
            
            return Command(
                goto=[Send(agent_name, agent_input)],
                update=updated_state,
                graph=Command.PARENT,
            )
        
        return handoff_tool
    
    def _build_management_graph(self) -> StateGraph:
        """Build the main management graph with supervisor pattern."""
        
        # Create handoff tools for each remote agent
        handoff_tools = []
        for agent_name, agent_config in self.remote_agents_config.items():
            tool = self._create_handoff_tool(agent_name, agent_config)
            handoff_tools.append(tool)
        
        # Add thread and run management tools
        thread_management_tools = self._create_thread_management_tools()
        all_tools = handoff_tools + thread_management_tools
        
        # Create the supervisor agent with all tools
        supervisor_agent = create_react_agent(
            model=self.llm,
            tools=all_tools,
            prompt=self._get_supervisor_prompt(),
            name="supervisor"
        )
        
        # Build the state graph
        graph = StateGraph(AgentManagementState)
        
        # Add supervisor node
        graph.add_node("supervisor", supervisor_agent)
        
        # Add remote agent nodes for receiving delegated tasks
        for agent_name in self.remote_agents_config.keys():
            # Create remote agent node that handles delegated tasks
            remote_agent_node = self._create_remote_agent_node(agent_name)
            graph.add_node(agent_name, remote_agent_node)
        
        # Set up edges
        graph.add_edge(START, "supervisor")
        
        # Add conditional edges from supervisor to remote agents
        # The handoff tools will use Command.goto with Send() to route to specific agents
        for agent_name in self.remote_agents_config.keys():
            graph.add_edge(agent_name, "supervisor")  # Remote agents return to supervisor
        
        return graph
    
    def _create_remote_agent_node(self, agent_name: str):
        """Create a node for a remote agent that can receive delegated tasks."""
        
        def remote_agent_node(state: AgentManagementState) -> AgentManagementState:
            """Handle delegated tasks for a remote agent."""
            
            # Get the remote graph for this agent
            if agent_name in self.remote_graphs:
                remote_graph = self.remote_graphs[agent_name]
                
                try:
                    # Extract the task from the state
                    messages = state.get("messages", [])
                    if messages:
                        # Use the remote graph to process the task
                        result = remote_graph({"messages": messages})
                        
                        # Update state with the response
                        updated_state = state.copy()
                        if "messages" in result:
                            updated_state["messages"] = result["messages"]
                        
                        # Add progress update
                        progress_update = {
                            "timestamp": datetime.now().isoformat(),
                            "event": "task_completed",
                            "agent": agent_name,
                            "message": f"Task completed by {agent_name}",
                            "details": {"response_received": True}
                        }
                        
                        progress_updates = updated_state.get("progress_updates", [])
                        progress_updates.append(progress_update)
                        updated_state["progress_updates"] = progress_updates
                        
                        # Update current operation
                        updated_state["current_operation"] = f"completed_by_{agent_name}"
                        
                        return updated_state
                
                except Exception as e:
                    print(f"Error in remote agent {agent_name}: {e}")
                    # Return error state
                    error_state = state.copy()
                    error_message = AIMessage(
                        content=f"Error processing task with {agent_name}: {str(e)}"
                    )
                    error_state["messages"] = state.get("messages", []) + [error_message]
                    
                    # Add error progress update
                    progress_update = {
                        "timestamp": datetime.now().isoformat(),
                        "event": "task_error",
                        "agent": agent_name,
                        "message": f"Error in {agent_name}: {str(e)}",
                        "details": {"error": True}
                    }
                    
                    progress_updates = error_state.get("progress_updates", [])
                    progress_updates.append(progress_update)
                    error_state["progress_updates"] = progress_updates
                    
                    return error_state
            
            # Fallback if remote graph not available
            fallback_state = state.copy()
            fallback_message = AIMessage(
                content=f"Remote agent {agent_name} is not available. Task could not be processed."
            )
            fallback_state["messages"] = state.get("messages", []) + [fallback_message]
            
            return fallback_state
        
        return remote_agent_node







