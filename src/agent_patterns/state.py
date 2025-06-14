"""State definition for the agent."""

from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State for the agent graph."""
    
    # Messages in the conversation
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Token count tracking
    current_token_count: int
    
    # Maximum token limit before summarization
    max_token_limit: int
    
    # Whether summarization was performed in the last step
    summarized: bool
    
    # Original message count before summarization (for tracking)
    original_message_count: Optional[int]
