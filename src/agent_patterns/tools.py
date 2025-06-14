"""Tools for the agent."""

import json
import random
from typing import Dict, Any
from langchain_core.tools import tool


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.
    
    Args:
        location: The city and state/country to get weather for
        
    Returns:
        Weather information as a string
    """
    # Simulate weather API call
    weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "partly cloudy"]
    temperature = random.randint(-10, 35)
    condition = random.choice(weather_conditions)
    
    return f"The weather in {location} is {condition} with a temperature of {temperature}°C."


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression safely.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
        
    Returns:
        Result of the calculation as a string
    """
    try:
        # Only allow basic mathematical operations for safety
        allowed_chars = set('0123456789+-*/()., ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression. Only numbers and basic operators (+, -, *, /, parentheses) are allowed."
        
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@tool
def search_knowledge(query: str) -> str:
    """Search for information on a given topic.
    
    Args:
        query: The search query
        
    Returns:
        Simulated search results
    """
    # Simulate knowledge search
    knowledge_base = {
        "python": "Python is a high-level programming language known for its simplicity and readability.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "ai": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.",
        "machine learning": "Machine Learning is a subset of AI that enables computers to learn without being explicitly programmed.",
    }
    
    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return f"Found information about '{query}': {value}"
    
    return f"No specific information found for '{query}'. This is a simulated knowledge base with limited entries."


@tool
def get_random_fact() -> str:
    """Get a random interesting fact.
    
    Returns:
        A random fact as a string
    """
    facts = [
        "Octopuses have three hearts and blue blood.",
        "A group of flamingos is called a 'flamboyance'.",
        "Honey never spoils - archaeologists have found edible honey in ancient Egyptian tombs.",
        "A single cloud can weigh more than a million pounds.",
        "Bananas are berries, but strawberries aren't.",
        "The shortest war in history lasted only 38-45 minutes.",
        "A shrimp's heart is in its head.",
        "There are more possible games of chess than atoms in the observable universe."
    ]
    
    return random.choice(facts)


# List of all available tools
TOOLS = [get_weather, calculate, search_knowledge, get_random_fact]
