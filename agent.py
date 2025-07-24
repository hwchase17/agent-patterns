"""
React Agent with Forced First Tool Call

This module implements a React agent that forces the first tool call to be a specific tool,
then switches to normal React behavior for subsequent interactions.

Key Features:
- Forces the first tool call to be 'initialize_session' 
- Uses return_direct=True to prevent infinite loops
- Switches to normal React behavior after initialization
- Maintains conversation state using MessagesState
- Demonstrates proper tool forcing patterns with LangGraph
"""

from typing import Literal, List, Dict, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import InMemorySaver


# Define the forced initialization tool with return_direct=True
@tool(return_direct=True)
def initialize_session() -> str:
    """
    Initialize the agent session. This tool is always called first to set up the conversation context.
    Using return_direct=True prevents infinite loops as documented in LangGraph tool forcing patterns.
    
    Returns:
        A greeting message acknowledging the session initialization
    """
    return "🚀 Session initialized! I'm ready to help you with weather information, calculations, or general questions. What would you like to know?"


# Define additional tools for normal React behavior after initialization
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
        "new york": "Sunny, 72°F with light breeze",
        "london": "Cloudy, 15°C with occasional drizzle",
        "tokyo": "Rainy, 18°C with heavy showers",
        "san francisco": "Foggy, 16°C with typical marine layer",
        "paris": "Partly cloudy, 20°C with mild winds",
        "sydney": "Clear, 25°C with sunshine"
    }
    location_lower = location.lower()
    return weather_data.get(location_lower, f"Weather data not available for {location}. Try: New York, London, Tokyo, San Francisco, Paris, or Sydney.")


@tool
def calculate(expression: str) -> str:
    """
    Perform mathematical calculations safely.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "10 * 5", "100 / 4")
        
    Returns:
        The result of the calculation
    """
    try:
        # Safe evaluation for basic math expressions
        # In production, you'd want to use a more secure math parser
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return f"Error: Expression contains invalid characters. Use only numbers and +, -, *, /, (, )"
        
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
    # Simulate search results with more comprehensive data
    search_results = {
        "python": "Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum in 1991.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with graph-based workflows.",
        "ai": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn.",
        "react": "ReAct (Reasoning and Acting) is a paradigm that combines reasoning and acting in language models for better problem-solving.",
        "llm": "Large Language Models (LLMs) are AI models trained on vast amounts of text data to understand and generate human-like text."
    }
    
    query_lower = query.lower()
    for key in search_results:
        if key in query_lower:
            return search_results[key]
    
    return f"Here's some general information about '{query}': This topic would benefit from further research. Try asking about Python, LangGraph, AI, React, or LLM for more detailed information."


# Enhanced state to track initialization status
class ForcedFirstToolState(MessagesState):
    """
    State that extends MessagesState to track whether the forced first tool has been called.
    This enables the transition from forced mode to normal React behavior.
    """
    initialized: bool = False


def should_initialize(state: ForcedFirstToolState) -> Literal["force_init", "normal_react"]:
    """
    Conditional routing function that determines whether to force initialization or use normal React behavior.
    
    Args:
        state: Current state of the agent
        
    Returns:
        "force_init" if not yet initialized, "normal_react" otherwise
    """
    return "force_init" if not state.get("initialized", False) else "normal_react"


def force_init_node(state: ForcedFirstToolState) -> Dict[str, Any]:
    """
    Node that forces the initialize_session tool call using model.bind_tools() with tool_choice parameter.
    This implements the forced first tool call pattern as documented in LangGraph.
    
    Args:
        state: Current state of the agent
        
    Returns:
        Updated state with forced initialization tool call and results
    """
    # Initialize model and bind with forced tool choice
    model = init_chat_model("anthropic:claude-3-haiku-20240307")
    forced_model = model.bind_tools(
        [initialize_session],
        tool_choice={"type": "tool", "name": "initialize_session"}
    )
    
    # Get current messages
    messages = state["messages"]
    
    # Force the model to call initialize_session tool
    ai_response = forced_model.invoke(messages)
    
    # Execute the forced tool call
    tool_node = ToolNode([initialize_session])
    tool_result = tool_node.invoke({"messages": [ai_response]})
    
    # Return updated state with initialization complete
    return {
        "messages": [ai_response] + tool_result["messages"],
        "initialized": True
    }


def normal_react_node(state: ForcedFirstToolState) -> Dict[str, Any]:
    """
    Node that handles normal React agent behavior after initialization.
    Uses create_react_agent() with all available tools for standard React behavior.
    
    Args:
        state: Current state of the agent
        
    Returns:
        Updated state from normal React agent execution
    """
    # Create normal React agent with all tools (excluding the forced init tool)
    model = init_chat_model("anthropic:claude-3-haiku-20240307")
    normal_tools = [get_weather, calculate, search_info]
    
    # Create React agent for normal behavior
    react_agent = create_react_agent(
        model=model,
        tools=normal_tools,
        prompt="You are a helpful assistant. Use the available tools to answer user questions accurately."
    )
    
    # Invoke the React agent with current messages
    result = react_agent.invoke({"messages": state["messages"]})
    
    # Return updated messages while preserving initialization status
    return {
        "messages": result["messages"],
        "initialized": state.get("initialized", True)
    }


def create_forced_first_tool_agent() -> StateGraph:
    """
    Create a React agent that forces the first tool call to be the initialize_session tool,
    then switches to normal React behavior for subsequent interactions.
    
    This implementation follows the LangGraph patterns for:
    1. Tool forcing using model.bind_tools() with tool_choice parameter
    2. Using return_direct=True to prevent infinite loops
    3. State management with MessagesState
    4. Conditional routing between forced and normal modes
    
    Returns:
        A compiled StateGraph that implements the forced first tool call pattern
    """
    # Create the state graph with custom state
    builder = StateGraph(ForcedFirstToolState)
    
    # Add nodes for forced initialization and normal React behavior
    builder.add_node("force_init", force_init_node)
    builder.add_node("normal_react", normal_react_node)
    
    # Add conditional routing from START based on initialization status
    builder.add_conditional_edges(
        START,
        should_initialize,
        {
            "force_init": "force_init",
            "normal_react": "normal_react"
        }
    )
    
    # Both nodes end the execution
    builder.add_edge("force_init", END)
    builder.add_edge("normal_react", END)
    
    # Compile with memory checkpointer for conversation persistence
    return builder.compile(checkpointer=InMemorySaver())


# Create the main agent instance
app = create_forced_first_tool_agent()


if __name__ == "__main__":
    # Example usage demonstrating forced first tool call behavior
    print("🤖 React Agent with Forced First Tool Call")
    print("=" * 60)
    print("This agent demonstrates:")
    print("1. Forced first tool call to 'initialize_session'")
    print("2. Transition to normal React behavior after initialization")
    print("3. Proper state management with MessagesState")
    print("=" * 60)
    
    # Test the agent with persistent thread
    config = {"configurable": {"thread_id": "demo-thread"}}
    
    # First interaction - should force initialize_session tool
    print("\n🔥 FIRST INTERACTION (Should force initialize_session tool):")
    result1 = app.invoke(
        {"messages": [HumanMessage("Hello, I want to know about the weather in Tokyo")]},
        config=config
    )
    
    print("Messages from first interaction:")
    for i, message in enumerate(result1["messages"], 1):
        if hasattr(message, 'content') and message.content:
            print(f"  {i}. {message.__class__.__name__}: {message.content}")
    
    print(f"\nInitialization status: {result1.get('initialized', False)}")
    
    print("\n" + "=" * 60)
    
    # Second interaction - should use normal React behavior
    print("\n⚡ SECOND INTERACTION (Should use normal React behavior):")
    result2 = app.invoke(
        {"messages": [HumanMessage("What's the weather in New York?")]},
        config=config
    )
    
    print("Messages from second interaction (last 3):")
    for i, message in enumerate(result2["messages"][-3:], 1):
        if hasattr(message, 'content') and message.content:
            print(f"  {i}. {message.__class__.__name__}: {message.content}")
    
    print(f"\nInitialization status: {result2.get('initialized', False)}")
    
    print("\n" + "=" * 60)
    print("✅ Demo completed! The agent successfully:")
    print("   - Forced the first tool call to 'initialize_session'")
    print("   - Switched to normal React behavior for subsequent interactions")
    print("   - Maintained conversation state across interactions")


