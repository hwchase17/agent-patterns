#!/usr/bin/env python3
"""
Example usage script for the form-filling agent.

This script demonstrates how to use the form-filling agent to collect
user information through a multi-section form with validation and
progress tracking.
"""

import asyncio
from agent import create_form_agent


async def run_form_example():
    """
    Example demonstrating the form-filling agent in action.
    
    This example shows:
    - Multi-section form processing (Personal Info, Contact Details, Preferences)
    - Human-in-the-loop input collection with validation
    - Progress tracking between sections
    - Error handling and re-prompting for invalid input
    - Form completion summary
    """
    print("=== Form-Filling Agent Example ===")
    print("This example will guide you through filling out a multi-section form.")
    print("The form has three sections:")
    print("1. Personal Information (first name, last name, age, date of birth)")
    print("2. Contact Details (email, phone, address, city)")
    print("3. Preferences (newsletter, communication method, interests)")
    print("
    input()
    
    # Create the form agent
    agent = create_form_agent()
    
    # Configuration for this session
    config = {
        "configurable": {
            "thread_id": "example_session_001"
        }
    }
    
    try:
        # Start the form-filling process
        print("
        
        # Initialize the form
        result = await agent.ainvoke({}, config=config)
        
        # The agent will handle the entire form-filling process
        # through human-in-the-loop interactions
        print("
        print(f"Session ID: {result.get('session_id', 'N/A')}")
        
    except KeyboardInterrupt:
        print("
        print("Your progress has been saved and you can resume later.")
        
    except Exception as e:
        print(f"
        print("Please check your configuration and try again.")


async def resume_form_example():
    """
    Example demonstrating how to resume a form-filling session.
    
    This shows how the agent's persistence features allow users
    to resume form filling from where they left off.
    """
    print("=== Resume Form Example ===")
    thread_id = input("Enter the thread ID to resume (or press Enter for default): ").strip()
    if not thread_id:
        thread_id = "example_session_001"
    
    agent = create_form_agent()
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        print(f"
        result = await agent.ainvoke({}, config=config)
        print("
        
    except Exception as e:
        print(f"


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Start new form")
    print("2. Resume existing form")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "2":
        asyncio.run(resume_form_example())
    else:
        asyncio.run(run_form_example())


