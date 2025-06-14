"""State management for the LangGraph agent system."""

from typing import List, Optional, Literal
from pydantic import BaseModel
from langchain_core.messages import BaseMessage


class AgentState(BaseModel):
    """State shared between agents in the workflow."""
    
    # The original user query/task
    user_query: str
    
    # Messages exchanged between agents
    messages: List[BaseMessage] = []
    
    # Current step in the workflow
    current_step: Literal["react", "review", "complete"] = "react"
    
    # Output from the ReAct agent
    react_output: Optional[str] = None
    
    # Feedback from the reviewer agent
    reviewer_feedback: Optional[str] = None
    
    # Decision from the reviewer
    reviewer_decision: Optional[Literal["approve", "feedback"]] = None
    
    # Final result when workflow is complete
    final_result: Optional[str] = None
    
    # Number of iterations (to prevent infinite loops)
    iteration_count: int = 0
    
    # Maximum iterations allowed
    max_iterations: int = 5

