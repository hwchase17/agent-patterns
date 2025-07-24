"""
React Agent with Forced First Tool Call

This module implements a React agent that forces the first tool call to be a specific tool,
then switches to normal React behavior for subsequent interactions.
"""

from typing import Annotated, Literal, Optional, Sequence, Union, List
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, START, END, add_messages
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import InMemorySaver


# Define the forced initialization tool
@tool(return_direct=True)
def initialize_session() -> str:
    """
    Initialize the agent session. This tool is always called first to set up the conversation context.
        
    Returns:
        A greeting message acknowledging the session initialization
    """
    return "Session initialized! I'm ready to help you with weather information, calculations, or general questions. What would you like to know?"


# Define additional tools for normal React behavior
@tool
def get_weather(location: str) -> str:
    """
    Get the current weather for a specified location.
    
    Args:
        location: The city or location to get weather for
        
    Returns:
        Weather information for the location
    """
    # Simulate weather data
    weather_data = {
        "new york": "Sunny, 72°F",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
        "san francisco": "Foggy, 16°C"
    }
    location_lower = location.lower()
    return weather_data.get(location_lower, f"Weather data not available for {location}")


@tool
def calculate(expression: str) -> str:
    """
    Perform mathematical calculations.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")
        
    Returns:
        The result of the calculation
    """
    try:
        # Simple evaluation for basic math expressions
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@tool
def search_info(query: str) -> str:
    """
    Search for information on a given topic.
    
    Args:
        query: The search query or topic
        
    Returns:
        Information about the topic
    """
    # Simulate search results
    search_results = {
        "python": "Python is a high-level programming language known for its simplicity and readability.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "ai": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines."
    }
    query_lower = query.lower()
    for key in search_results:
        if key in query_lower:
            return search_results[key]
    return f"Here's some general information about '{query}': This is a topic that would benefit from further research."


# Enhanced state to track initialization status
class ForcedFirstToolState(MessagesState):
    """State that tracks whether the forced first tool has been called."""
    initialized: bool = False


def should_initialize(state: ForcedFirstToolState) -> Literal["initialize", "normal_agent"]:
    """
    Determine whether to force initialization or proceed with normal agent behavior.
    
    Args:
        state: Current state of the agent
        
    Returns:
        "initialize" if not yet initialized, "normal_agent" otherwise
    """
    return "initialize" if not state.get("initialized", False) else "normal_agent"


def initialize_node(state: ForcedFirstToolState):
    """
    Node that forces the initialize_session tool call.
    
    Args:
        state: Current state of the agent
        
    Returns:
        Updated state with initialization tool call
    """
    # Create a model that's forced to call the initialize_session tool
    model = init_chat_model("anthropic:claude-3-haiku-20240307")
    forced_model = model.bind_tools(
        [initialize_session],
        tool_choice={"type": "tool", "name": "initialize_session"}
    )
    
    # Get the last user message
    messages = state["messages"]
    
    # Call the model with forced tool choice
    response = forced_model.invoke(messages)
    
    # Execute the tool call
    tool_node = ToolNode([initialize_session])
    tool_result = tool_node.invoke({"messages": [response]})
    
    # Return updated state with initialization complete
    return {
        "messages": [response] + tool_result["messages"],
        "initialized": True
    }


def normal_agent_node(state: ForcedFirstToolState):
    """
    Node that handles normal React agent behavior after initialization.
    
    Args:
        state: Current state of the agent
        
    Returns:
        Updated state from normal agent execution
    """
    # Create normal React agent with all tools
    model = init_chat_model("anthropic:claude-3-haiku-20240307")
    all_tools = [get_weather, calculate, search_info]
    
    # Create a temporary agent for this interaction
    temp_agent = create_react_agent(model=model, tools=all_tools)
    
    # Invoke the agent with current messages
    result = temp_agent.invoke({"messages": state["messages"]})
    
    # Return the updated messages while preserving initialization status
    return {
        "messages": result["messages"],
        "initialized": state.get("initialized", True)
    }


def create_forced_first_tool_agent() -> StateGraph:
    """
    Create a React agent that forces the first tool call to be the initialize_session tool.
        
    Returns:
        A compiled StateGraph that forces first tool call then switches to normal React behavior
    """
    # Create the state graph
    builder = StateGraph(ForcedFirstToolState)
    
    # Add nodes
    builder.add_node("initialize", initialize_node)
    builder.add_node("normal_agent", normal_agent_node)
    
    # Add conditional edge from START
    builder.add_conditional_edges(START, should_initialize)
    
    # Add edges
    builder.add_edge("initialize", END)
    builder.add_edge("normal_agent", END)
    
    # Compile with checkpointer for memory
    return builder.compile(checkpointer=InMemorySaver())


# Create the main agent instance
app = create_forced_first_tool_agent()


if __name__ == "__main__":
    # Example usage
    print("React Agent with Forced First Tool Call")
    print("=" * 50)
    
    # Test the agent
    config = {"configurable": {"thread_id": "test-thread"}}
    
    # First interaction - should force initialize_session tool
    result1 = app.invoke(
        {"messages": [HumanMessage("Hello, I want to know about the weather")]},
        config=config
    )
    
    print("First interaction result:")
    for message in result1["messages"]:
        if hasattr(message, 'content') and message.content:
            print(f"{message.__class__.__name__}: {message.content}")
    
    print("
    
    # Second interaction - should use normal React behavior
    result2 = app.invoke(
        {"messages": [HumanMessage("What's the weather in New York?")]},
        config=config
    )
    
    print("Second interaction result:")
    for message in result2["messages"][-3:]:  # Show last few messages
        if hasattr(message, 'content') and message.content:
            print(f"{message.__class__.__name__}: {message.content}")


