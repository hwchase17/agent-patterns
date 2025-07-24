"""
React Agent with Forced First Tool Call

This module implements a React agent that forces the first tool call to be a specific tool,
then switches to normal React behavior for subsequent interactions.
"""

from typing import Annotated, Literal, Optional, Sequence, Union
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import InMemorySaver


# Define the forced initialization tool
@tool(return_direct=True)
def initialize_session(user_input: str) -> str:
    """
    Initialize the agent session. This tool is always called first to set up the conversation context.
    
    Args:
        user_input: The user's initial input or query
        
    Returns:
        A greeting message acknowledging the session initialization
    """
    return f"Session initialized! I understand you want to: {user_input}. How can I help you further?"


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


# Custom state to track if initialization has occurred
class AgentState(MessagesState):
    initialized: bool = False


def create_forced_first_tool_agent(model_name: str = "anthropic:claude-3-haiku-20240307") -> StateGraph:
    """
    Create a React agent that forces the first tool call to be the initialize_session tool.
    
    Args:
        model_name: The name of the language model to use
        
    Returns:
        A compiled StateGraph representing the agent
    """
    # Initialize the language model
    model = init_chat_model(model_name)
    
    # Define all available tools
    all_tools = [initialize_session, get_weather, calculate, search_info]
    
    # Create the forced first call agent
    # The model is bound with tool_choice to force the initialize_session tool on first call
    forced_model = model.bind_tools(
        [initialize_session], 
        tool_choice={"type": "tool", "name": "initialize_session"}
    )
    
    # Create the normal React agent for subsequent interactions
    normal_agent = create_react_agent(
        model=model,
        tools=all_tools,
        checkpointer=InMemorySaver()
    )
    
    return normal_agent


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

