#!/usr/bin/env python3
"""
Example script demonstrating the TwoStageAgent with a basic ReAct agent.

This script shows the complete workflow from initial execution through review
and potential iteration, including:
1. Creating a basic ReAct agent with simple tools
2. Setting up the TwoStageAgent with review mechanism
3. Running the workflow and observing the review process
4. Handling multiple iterations when improvements are needed
"""

import os
from typing import Any
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Import our TwoStageAgent implementation
from agent import TwoStageAgent, TwoStageState


# Define some simple tools for the ReAct agent to demonstrate functionality
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        The result of the calculation as a string
    """
    try:
        # Simple safety check - only allow basic math operations
        allowed_chars = set('0123456789+-*/()., ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating expression: {str(e)}"


@tool
def word_counter(text: str) -> str:
    """
    Count the number of words in a given text.
    
    Args:
        text: The text to count words in
    
    Returns:
        The word count as a string
    """
    word_count = len(text.split())
    return f"The text contains {word_count} words."


@tool
def text_reverser(text: str) -> str:
    """
    Reverse the given text.
    
    Args:
        text: The text to reverse
    
    Returns:
        The reversed text
    """
    return f"Reversed text: {text[::-1]}"


def create_mock_model():
    """
    Create a mock model for demonstration purposes.
    In a real implementation, you would use an actual LLM like OpenAI GPT or Anthropic Claude.
    """
    class MockModel:
        def invoke(self, messages):
            # Simple mock responses for demonstration
            last_message = messages[-1].content if messages else ""
            
            if "calculate" in last_message.lower() or "math" in last_message.lower():
                return "I'll help you with that calculation. Let me use the calculator tool."
            elif "count" in last_message.lower() and "word" in last_message.lower():
                return "I'll count the words in that text for you."
            elif "reverse" in last_message.lower():
                return "I'll reverse that text for you."
            else:
                return "I understand your request and will help you with it."
    
    return MockModel()


def main():
    """
    Main function demonstrating the TwoStageAgent workflow.
    """
    print("🚀 TwoStageAgent Example Demonstration")
    print("=" * 50)
    
    # Note: In a real implementation, you would use an actual LLM
    # For this example, we're using a mock model for demonstration
    print("⚠️  Note: This example uses a mock model for demonstration.")
    print("   In production, replace with an actual LLM (OpenAI, Anthropic, etc.)")
    print()
    
    # Create the mock model
    model = create_mock_model()
    
    # Define the tools available to the ReAct agent
    tools = [calculator, word_counter, text_reverser]
    
    # Create a basic ReAct agent
    print("📝 Step 1: Creating ReAct agent with tools...")
    react_agent = create_react_agent(
        model=model,
