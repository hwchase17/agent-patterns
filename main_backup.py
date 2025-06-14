"""
LangGraph React Agent with Forced First Tool Call

This demonstrates a React agent that forces a specific tool call on the first invocation,
then allows free tool selection afterward.
"""

import os
from typing import Annotated, Dict, Any, List
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    is_first_call: bool
    forced_tool_used: bool


# Define tools that the agent can use
@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression. Use this for any math calculations."""
    try:
        # Simple evaluation - in production, use a safer math parser
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@tool
def weather_tool(location: str) -> str:
    """Get weather information for a location. This is the FORCED first tool."""
    # Simulated weather data - in production, use a real weather API
    weather_data = {
        "new york": "Sunny, 72°F",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 20°C",
        "default": "Partly cloudy, 68°F"
    }
    location_lower = location.lower()
    weather = weather_data.get(location_lower, weather_data["default"])
    return f"Weather in {location}: {weather}"


@tool
def search_tool(query: str) -> str:
    """Search for information on a given topic."""
    # Simulated search results - in production, use a real search API
    return f"Search results for '{query}': Here are some relevant articles and information about {query}."


@tool
def translator_tool(text: str, target_language: str = "spanish") -> str:
    """Translate text to a target language."""
    # Simulated translation - in production, use a real translation API
    translations = {
        "hello": {"spanish": "hola", "french": "bonjour", "german": "hallo"},
        "goodbye": {"spanish": "adiós", "french": "au revoir", "german": "auf wiedersehen"},
        "thank you": {"spanish": "gracias", "french": "merci", "german": "danke"}
    }
    
    text_lower = text.lower()
    if text_lower in translations and target_language in translations[text_lower]:
        return f"'{text}' in {target_language} is '{translations[text_lower][target_language]}'"
    else:
        return f"Translation of '{text}' to {target_language}: [simulated translation]"


# Create the list of available tools
tools = [calculator, weather_tool, search_tool, translator_tool]
tool_node = ToolNode(tools)


def create_react_agent():
    """Create a React agent with forced first tool call functionality."""
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here")
    )
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    def agent_node(state: AgentState) -> Dict[str, Any]:
        """The main agent reasoning node."""
        messages = state["messages"]
        is_first_call = state.get("is_first_call", True)
        forced_tool_used = state.get("forced_tool_used", False)
        
        # If this is the first call and we haven't used the forced tool yet
        if is_first_call and not forced_tool_used:
            # Create a system message that forces the weather tool
            system_prompt = """You are a helpful assistant. 
            IMPORTANT: For the very first user query, you MUST use the weather_tool to check the weather 
            for any location (you can choose a default location if none is specified). 
            After using the weather tool once, you can then use any available tools as needed.
            
            Available tools: calculator, weather_tool, search_tool, translator_tool"""
            
            messages_with_system = [{"role": "system", "content": system_prompt}] + messages
        else:
            # Normal operation - agent can choose any tool
            system_prompt = """You are a helpful assistant with access to various tools. 
            Use the appropriate tool based on the user's request.
            
            Available tools: calculator, weather_tool, search_tool, translator_tool"""
            
            messages_with_system = [{"role": "system", "content": system_prompt}] + messages
        
        # Get response from LLM
        response = llm_with_tools.invoke(messages_with_system)
        
        # Update state
        updates = {"messages": [response]}
        
        # Check if weather tool was called (forced tool)
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "weather_tool":
                    updates["forced_tool_used"] = True
                    updates["is_first_call"] = False
        
        return updates
    
    def should_continue(state: AgentState) -> str:
        """Determine whether to continue with tool calls or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the last message has tool calls, go to tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        # Otherwise, end the conversation
        return END
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    
    # Add memory
    memory = MemorySaver()
    
    # Compile the graph
    app = workflow.compile(checkpointer=memory)
    
    return app


def main():
    """Demonstrate the React agent with forced first tool call."""
    print("🤖 LangGraph React Agent with Forced First Tool Call")
    print("=" * 60)
    print("This agent will FORCE a weather check on the first call,")
    print("then allow free tool selection afterward.
    
    # Create the agent
    agent = create_react_agent()
    
    # Configuration for the agent
    config = {"configurable": {"thread_id": "demo-thread"}}
    
    # Initial state
    initial_state = {
        "messages": [],
        "is_first_call": True,
        "forced_tool_used": False
    }
    
    print("Agent created successfully!")
    print("Ready to demonstrate forced first tool call behavior.")
    print("
    print("1. Ask any question - it will force a weather check first")
    print("2. Then ask follow-up questions - it will use appropriate tools")
    print("
    print("- 'What is 2 + 2?' (will check weather first, then calculate)")
    print("- 'Search for Python tutorials' (will use search tool)")
    print("- 'Translate hello to French' (will use translator tool)")


if __name__ == "__main__":
    main()

