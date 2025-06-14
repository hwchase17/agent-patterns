"""
State schema for the orchestrator agent to track background agent threads.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


class AgentStatus(str, Enum):
    """Status of a background agent thread."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BackgroundAgent(BaseModel):
    """Information about a background agent thread."""
    thread_id: str = Field(..., description="Unique thread ID for the agent")
    agent_name: str = Field(..., description="Name/type of the background agent")
    agent_endpoint: str = Field(..., description="URL endpoint of the remote agent")
    status: AgentStatus = Field(default=AgentStatus.PENDING, description="Current status")
    task_description: str = Field(..., description="Description of the task assigned to this agent")
    created_at: datetime = Field(default_factory=datetime.now, description="When the agent was created")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last status update time")
    result: Optional[Any] = Field(default=None, description="Result from the agent when completed")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def update_status(self, status: AgentStatus, result: Optional[Any] = None, error: Optional[str] = None):
        """Update the agent status and timestamp."""
        self.status = status
        self.updated_at = datetime.now()
        if result is not None:
            self.result = result
        if error is not None:
            self.error_message = error

    @property
    def is_active(self) -> bool:
        """Check if the agent is still active (not in a terminal state)."""
        return self.status in [AgentStatus.PENDING, AgentStatus.RUNNING]

    @property
    def is_completed(self) -> bool:
        """Check if the agent has completed successfully."""
        return self.status == AgentStatus.COMPLETED

    @property
    def has_failed(self) -> bool:
        """Check if the agent has failed."""
        return self.status == AgentStatus.FAILED


class OrchestratorState(MessagesState):
    """
    State for the orchestrator agent that manages background agents.
    Extends MessagesState to maintain conversation history with the user.
    """
    # Background agents being managed
    background_agents: Dict[str, BackgroundAgent] = Field(
        default_factory=dict,
        description="Dictionary of thread_id -> BackgroundAgent"
    )
    
    # User interaction state
    user_input: Optional[str] = Field(
        default=None,
        description="Latest user input"
    )
    
    # System state
    iteration_count: int = Field(
        default=0,
        description="Number of iterations the orchestrator has run"
    )
    
    last_status_check: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last background agent status check"
    )
    
    # Configuration
    max_concurrent_agents: int = Field(
        default=5,
        description="Maximum number of concurrent background agents"
    )

    def add_background_agent(self, agent: BackgroundAgent) -> None:
        """Add a new background agent to track."""
        self.background_agents[agent.thread_id] = agent

    def get_active_agents(self) -> List[BackgroundAgent]:
        """Get all active background agents."""
        return [agent for agent in self.background_agents.values() if agent.is_active]

    def get_completed_agents(self) -> List[BackgroundAgent]:
        """Get all completed background agents."""
        return [agent for agent in self.background_agents.values() if agent.is_completed]

    def get_failed_agents(self) -> List[BackgroundAgent]:
        """Get all failed background agents."""
        return [agent for agent in self.background_agents.values() if agent.has_failed]

    def update_agent_status(self, thread_id: str, status: AgentStatus, 
                          result: Optional[Any] = None, error: Optional[str] = None) -> bool:
        """Update the status of a background agent."""
        if thread_id in self.background_agents:
            self.background_agents[thread_id].update_status(status, result, error)
            return True
        return False

    def remove_agent(self, thread_id: str) -> bool:
        """Remove a background agent from tracking."""
        if thread_id in self.background_agents:
            del self.background_agents[thread_id]
            return True
        return False

    def can_spawn_agent(self) -> bool:
        """Check if we can spawn a new background agent."""
        active_count = len(self.get_active_agents())
        return active_count < self.max_concurrent_agents

    def get_status_summary(self) -> Dict[str, int]:
        """Get a summary of agent statuses."""
        summary = {status.value: 0 for status in AgentStatus}
        for agent in self.background_agents.values():
            summary[agent.status.value] += 1
        return summary


