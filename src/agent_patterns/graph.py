"""Main graph definition for the agent."""

import os
from typing import Dict, Any, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .tools import TOOLS
from .utils import count_tokens, create_summary_message, should_summarize


def get_llm(model_name: str = None) -> Any:
    """Get the appropriate LLM based on configuration.
    
    Args:
        model_name: Optional model name override
        
    Returns:
        Configured LLM instance
    """
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "gpt-4")
    
    if model_name.startswith("gpt"):
        return ChatOpenAI(
            model=model_name,
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif model_name.startswith("claude"):
        return ChatAnthropic(
            model=model_name,
            temperature=0.1,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    else:
        # Default to OpenAI
        return ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )


def check_token_limit(state: AgentState, config: RunnableConfig) -> AgentState:
    """Check if token limit is exceeded and summarize if needed.
    
    Args:
        state: Current agent state
        config: Runnable configuration
        
    Returns:
        Updated state with potential summarization
    """
    messages = state["messages"]
    max_tokens = state.get("max_token_limit", 4000)
    model_name = os.getenv("MODEL_NAME", "gpt-4")
    
    # Count current tokens
    current_tokens = count_tokens(messages, model_name)
    
    # Update token count in state
    state["current_token_count"] = current_tokens
    
    # Check if summarization is needed
    if should_summarize(current_tokens, max_tokens):
        # Keep the last message (usually the most recent user input)
        if len(messages) > 1:
            last_message = messages[-1]
            messages_to_summarize = messages[:-1]
            
            # Create summary
            summary_text = create_summary_message(messages_to_summarize, model_name)
            summary_message = SystemMessage(content=summary_text)
            
            # Update messages with summary + last message
            state["messages"] = [summary_message, last_message]
            state["summarized"] = True
            state["original_message_count"] = len(messages)
            
            # Recalculate token count after summarization
            state["current_token_count"] = count_tokens(state["messages"], model_name)
        else:
            state["summarized"] = False
    else:
        state["summarized"] = False
    
    return state


def agent_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Main agent processing node.
    
    Args:
        state: Current agent state
        config: Runnable configuration
        
    Returns:
        Updated state with agent response
    """
    messages = state["messages"]
    
    # Get LLM
    llm = get_llm()
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(TOOLS)
    
    # Get response from LLM
    response = llm_with_tools.invoke(messages)
    
    # Add response to messages
    state["messages"] = messages + [response]
    
    return state


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine whether to continue with tool calls or end.
    
    Args:
        state: Current agent state
        
    Returns:
        Next node to execute
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message has tool calls, continue to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end the conversation
    return "end"


def create_graph() -> StateGraph:
    """Create the main agent graph.
    
    Returns:
        Configured StateGraph
    """
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("check_tokens", check_token_limit)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(TOOLS))
    
    # Set entry point
    workflow.set_entry_point("check_tokens")
    
    # Add edges
    workflow.add_edge("check_tokens", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "check_tokens")
    
    return workflow


# Create the compiled graph
memory = MemorySaver()
workflow = create_graph()
graph = workflow.compile(checkpointer=memory)


# Export for langgraph.json
__all__ = ["graph"]

