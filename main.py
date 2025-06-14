#!/usr/bin/env python3
"""
Main execution script to demonstrate the LangGraph dual-agent system.

This script showcases a ReAct agent working with a reviewer agent in a feedback loop.
The ReAct agent processes tasks, and the reviewer either approves the output or 
provides feedback for improvement.
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agents.coordinator import CoordinatorAgent


def setup_environment():
    """Set up environment variables and API keys."""
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in a .env file or environment variable.")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        
        # For demonstration purposes, allow user to input key
        api_key = input("Enter your OpenAI API key (or press Enter to use mock mode): ").strip()
        if not api_key:
            print("Running in mock mode - the agents will use placeholder responses.")
            return "mock-key"
    
    return api_key


def run_example_queries(coordinator: CoordinatorAgent):
    """Run several example queries to demonstrate the agent system."""
    
    examples = [
        {
            "title": "Simple Math Question",
            "query": "What is 15 * 23 + 7? Please show your work."
        },
        {
            "title": "Research Question", 
            "query": "What are the main benefits of renewable energy sources?"
        },
        {
            "title": "Complex Analysis",
            "query": "Analyze the pros and cons of remote work and provide a balanced conclusion."
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"
        print(f"EXAMPLE {i}: {example['title']}")
        print(f"{'='*60}")
        print(f"Query: {example['query']}")
        print(f"
        print("🤖 Starting dual-agent workflow...")
        
        try:
            result = coordinator.run(example['query'])
            
            print(f"
            print(f"Iterations: {result.get('iteration_count', 0)}")
            print(f"Final Status: {result.get('current_step', 'unknown')}")
            print(f"
            print(f"{result.get('final_result', 'No result available')}")
            
        except Exception as e:
            print(f"❌ Error running example: {str(e)}")
        
        print(f"


def main():
    """Main execution function."""
    print("🚀 LangGraph Dual-Agent System Demo")
    print("ReAct Agent + Reviewer Agent with Feedback Loop")
    print("="*60)
    
    # Setup environment
    api_key = setup_environment()
    
    # Initialize the coordinator agent
    print("
    coordinator = CoordinatorAgent(openai_api_key=api_key)
    print("✅ Coordinator agent ready!")
    
    # Run example queries
    run_example_queries(coordinator)
    
    print(f"
    print("🎉 Demo completed! The dual-agent system demonstrated:")
    print("   • ReAct agent processing queries with tools")
    print("   • Reviewer agent evaluating outputs")
    print("   • Feedback loop for iterative improvement")
    print("   • Automatic completion after max iterations")


if __name__ == "__main__":
    main()

