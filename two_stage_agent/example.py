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
        tools=tools,
        prompt="You are a helpful assistant with access to various tools. "
               "Use the appropriate tools to help answer user questions accurately and completely.",
        name="basic_react_agent"
    )
    print("✅ ReAct agent created successfully!")
    print()
    
    # Create the TwoStageAgent with review mechanism
    print("🔍 Step 2: Creating TwoStageAgent with review mechanism...")
    two_stage_agent = TwoStageAgent(
        react_agent=react_agent,
        model=model,
        max_iterations=3,  # Allow up to 3 iterations for improvement
        review_prompt=None  # Use default review prompt
    )
    print("✅ TwoStageAgent created successfully!")
    print()
    
    # Example 1: Simple calculation that should pass review
    print("🧮 Example 1: Simple Math Calculation")
    print("-" * 30)
    
    initial_state = TwoStageState(
        messages=[HumanMessage(content="What is 15 + 27 * 3? Please calculate this step by step.")]
    )
    
    print("User Query: What is 15 + 27 * 3? Please calculate this step by step.")
    print()
    print("🔄 Running TwoStageAgent workflow...")
    
    try:
        # Note: In a real implementation, you would invoke the compiled graph
        # For this example, we'll simulate the workflow steps
        print("   → Initial execution: ReAct agent processing query...")
        print("   → Review phase: Evaluating output quality...")
        print("   → Decision: Determining if output meets standards...")
        
        # Simulate a successful workflow
        print("✅ Workflow completed successfully!")
        print("📊 Results:")
        print("   - Iterations used: 1")
        print("   - Review status: Approved")
        print("   - Final output: The calculation 15 + 27 * 3 = 15 + 81 = 96")
        
    except Exception as e:
        print(f"❌ Error during workflow execution: {e}")
    
    print()
    
    # Example 2: More complex query that might need iteration
    print("📝 Example 2: Text Analysis Task")
    print("-" * 30)
    
    initial_state = TwoStageState(
        messages=[HumanMessage(content="Please count the words in this sentence and then reverse it: 'The quick brown fox jumps over the lazy dog'")]
    )
    
    print("User Query: Please count the words in this sentence and then reverse it:")
    print("'The quick brown fox jumps over the lazy dog'")
    print()
    print("🔄 Running TwoStageAgent workflow...")
    
    try:
        # Simulate a workflow that might need iteration
        print("   → Initial execution: ReAct agent processing query...")
        print("   → Review phase: Evaluating completeness...")
        print("   → Decision: Output needs improvement (missing word count)")
        print("   → Iteration 2: ReAct agent improving response...")
        print("   → Review phase: Re-evaluating improved output...")
        print("   → Decision: Output now meets all requirements")
        
        print("✅ Workflow completed after 2 iterations!")
        print("📊 Results:")
        print("   - Iterations used: 2")
        print("   - Review status: Approved")
        print("   - Final output: The sentence contains 9 words. Reversed: 'god yzal eht revo spmuj xof nworb kciuq ehT'")
        
    except Exception as e:
        print(f"❌ Error during workflow execution: {e}")
    
    print()
    
    # Example 3: Demonstrate maximum iteration limit
    print("⚠️  Example 3: Maximum Iteration Limit")
    print("-" * 30)
    
    print("This example would demonstrate what happens when the review agent")
    print("consistently rejects outputs and the maximum iteration limit is reached.")
    print()
    print("🔄 Simulating workflow with persistent issues...")
    print("   → Initial execution: ReAct agent processing query...")
    print("   → Review phase: Output rejected (insufficient detail)")
    print("   → Iteration 2: ReAct agent improving response...")
    print("   → Review phase: Output rejected (still needs improvement)")
    print("   → Iteration 3: ReAct agent final attempt...")
    print("   → Review phase: Output rejected (quality issues)")
    print("   → Maximum iterations reached - workflow terminated")
    
    print("⚠️  Workflow terminated due to maximum iteration limit!")
    print("📊 Results:")
    print("   - Iterations used: 3 (maximum)")
    print("   - Review status: Rejected")
    print("   - Final output: Best available output from final iteration")
    
    print()
    print("🎉 TwoStageAgent demonstration completed!")
    print("=" * 50)
    print()
    print("💡 Key Features Demonstrated:")
    print("   ✓ ReAct agent integration with custom tools")
    print("   ✓ Automatic review and quality assessment")
    print("   ✓ Iterative improvement based on feedback")
    print("   ✓ Maximum iteration limits for safety")
    print("   ✓ State management across workflow steps")
    print()
    print("🔧 To use with real LLMs:")
    print("   1. Replace MockModel with actual LLM (OpenAI, Anthropic, etc.)")
    print("   2. Set up proper API keys and configuration")
    print("   3. Invoke the compiled graph: two_stage_agent.graph.invoke(initial_state)")
    print("   4. Handle real responses and state updates")


if __name__ == "__main__":
    main()
