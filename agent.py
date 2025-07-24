"""
LangGraph Chat Agent with Tool Calling Capabilities

This module creates a simple chat agent using LangGraph's create_react_agent
with configurable system prompt and tool calling capabilities.
"""

from typing import List
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.checkpoint.memory import InMemorySaver


def get_weather(city: str) -> str:
    """
    Get weather information for a given city.
    
    This is a simple demonstration tool that returns mock weather data.
    In a real application, this would connect to a weather API.
    
    Args:
        city: The name of the city to get weather for
        
    Returns:
        A string describing the weather in the specified city
    """
    # Mock weather data for demonstration
    weather_data = {
        "san francisco": "Foggy and cool, 62°F",
        "new york": "Partly cloudy, 68°F", 
        "london": "Rainy, 55°F",
        "tokyo": "Sunny, 75°F",
        "paris": "Overcast, 60°F"
    }
    
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"The weather in {city} is: {weather_data[city_lower]}"
    else:
        return f"Weather data for {city} is not available. It's probably nice though!"


def create_dynamic_prompt(state: AgentState, config: RunnableConfig) -> List[AnyMessage]:
    """
    Create a dynamic system prompt based on configuration.
    
    This allows the system prompt to be updated at runtime through the config.
    """
    # Get the system prompt from config, with a default fallback
    system_prompt = config.get("configurable", {}).get(
        "system_prompt", 
        "You are a helpful AI assistant with access to tools. Use the available tools when needed to provide accurate and helpful responses."
    )
    
    # Return the system message plus the conversation history
    return [{"role": "system", "content": system_prompt}] + state["messages"]

