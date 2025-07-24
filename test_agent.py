#!/usr/bin/env python3
"""
Test script for the React agent with forced first tool call functionality.

This script demonstrates that the initialize_session tool is always called first,
regardless of the user's input, before normal React behavior takes over.
"""

import os
from typing import Dict, Any
from agent import create_forced_first_tool_agent


def test_forced_first_tool_call():
    """
    Test that the agent always calls the initialize_session tool first,
    regardless of user input, then switches to normal React behavior.
    """
    print("=" * 60)
    print("TESTING FORCED FIRST TOOL CALL BEHAVIOR")
    print("=" * 60)
    
    # Create the agent
    agent = create_forced_first_tool_agent()
    
    # Test cases with different types of inputs
    test_cases = [
        {
            "name": "Weather Query",
            "input": "What's the weather like in New York?",
            "description": "User asks about weather - should force init first, then handle weather"
        },
        {
            "name": "Math Calculation",
            "input": "Calculate 25 * 4 + 10",
            "description": "User asks for calculation - should force init first, then calculate"
        },
        {
            "name": "General Information",
            "input": "Tell me about artificial intelligence",
            "description": "User asks for information - should force init first, then search"
        },
        {
            "name": "Greeting",
            "input": "Hello, how are you?",
            "description": "Simple greeting - should force init first, then respond"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*20} TEST CASE {i}: {test_case['name']} {'='*20}")
        print(f"Description: {test_case['description']}")
        print(f"User Input: '{test_case['input']}'")
        print("-" * 60)
        
        try:
            # Create a new thread for each test to ensure clean state
            thread_id = f"test_thread_{i}"
            config = {"configurable": {"thread_id": thread_id}}
            
            # Invoke the agent
            result = agent.invoke(
                {"messages": [("user", test_case["input"])]},
                config=config
            )
            
            # Analyze the messages to show the forced first tool call
            messages = result.get("messages", [])
            
            print("EXECUTION FLOW:")
            tool_calls_found = []
            
            for j, message in enumerate(messages):
                if hasattr(message, 'type'):
                    if message.type == "human":
                        print(f"  {j+1}. USER: {message.content}")
                    elif message.type == "ai":
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            for tool_call in message.tool_calls:
                                tool_name = tool_call.get('name', 'unknown')
                                tool_calls_found.append(tool_name)
                                print(f"  {j+1}. AI TOOL CALL: {tool_name}")
                        else:
                            content = getattr(message, 'content', str(message))
                            if content:
                                print(f"  {j+1}. AI RESPONSE: {content[:100]}...")
                    elif message.type == "tool":
                        tool_name = getattr(message, 'name', 'unknown')
                        content = getattr(message, 'content', str(message))
                        print(f"  {j+1}. TOOL RESULT ({tool_name}): {content[:100]}...")
            
            # Verify forced first tool call
            if tool_calls_found:
                first_tool = tool_calls_found[0]
                if first_tool == "initialize_session":
                    print(f"\n✅ SUCCESS: First tool called was '{first_tool}' (forced initialization)")
                    if len(tool_calls_found) > 1:
                        print(f"✅ SUCCESS: Subsequent tools called: {tool_calls_found[1:]}")
                        print("✅ SUCCESS: Normal React behavior activated after initialization")
                    else:
                        print("ℹ️  INFO: Only initialization tool called (return_direct=True)")
                else:
                    print(f"❌ FAILURE: First tool was '{first_tool}', expected 'initialize_session'")
            else:
                print("❌ FAILURE: No tool calls detected")
                
        except Exception as e:
            print(f"❌ ERROR: Test failed with exception: {e}")
        
        print("-" * 60)
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY:")
    print("- All test cases should show 'initialize_session' as the first tool called")
    print("- This demonstrates the forced first tool call behavior")
    print("- After initialization, normal React behavior should take over")
    print("=" * 60)


if __name__ == "__main__":
    test_forced_first_tool_call()


