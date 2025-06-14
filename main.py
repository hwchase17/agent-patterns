#!/usr/bin/env python3
"""
Main entry point for the LangGraph Orchestrator Agent.

This script provides a command-line interface to run the orchestrator agent
that manages background agents asynchronously while maintaining user interaction.
"""

import argparse
import asyncio
import logging
import sys
from typing import Dict, Optional
import json

from langchain_core.messages import HumanMessage

from src.agent_patterns.orchestrator import OrchestratorAgent
from src.agent_patterns.state import OrchestratorState


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('orchestrator.log')
        ]
    )


def load_agent_endpoints(config_file: Optional[str] = None) -> Dict[str, str]:
    """
    Load agent endpoints from configuration file or use defaults.
    
    Args:
        config_file: Path to JSON configuration file
        
    Returns:
        Dictionary mapping agent names to their endpoints
    """
    default_endpoints = {
        "simple_agent": "http://localhost:8001",
        "math_agent": "http://localhost:8002",
        "text_agent": "http://localhost:8003"
    }
    
    if config_file:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('agent_endpoints', default_endpoints)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning(f"Could not load config file {config_file}: {e}")
            logging.info("Using default agent endpoints")
    
    return default_endpoints


async def interactive_mode(orchestrator: OrchestratorAgent) -> None:
    """
    Run the orchestrator in interactive mode where user can chat with it.
    """
    print("🤖 LangGraph Orchestrator Agent")
    print("=" * 50)
    print("You can now chat with the orchestrator agent.")
    print("The agent can spawn background agents and monitor their progress.")
    print("Type 'quit', 'exit', or 'bye' to stop.")
    print("Type 'status' to see current background agent status.")
    print("Type 'help' for more commands.")
    print("=" * 50)
    
    state = OrchestratorState()
    
    while True:
        try:
            # Get user input
            user_input = input("
            
            if not user_input:
                continue
                
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'status':
                active_agents = state.get_active_agents()
                if active_agents:
                    print(f"📊 Active background agents: {len(active_agents)}")
                    for agent in active_agents:
                        print(f"  - {agent.agent_name} ({agent.thread_id}): {agent.status.value}")
                else:
                    print("📊 No active background agents")
                continue
            elif user_input.lower() == 'help':
                print("Available commands:")
                print("  - 'status': Show current background agent status")
                print("  - 'quit'/'exit'/'bye': Exit the program")
                print("  - Any other text: Chat with the orchestrator")
                continue
            
            # Add user message to state
            state.messages.append(HumanMessage(content=user_input))
            
            # Run one iteration of the orchestrator
            print("🤖 Orchestrator: ", end="", flush=True)
            
            # Use streaming mode for better user experience
            async for update in orchestrator.stream_run():
                if 'respond_to_user' in update:
                    # Get the latest AI message
                    messages = update['respond_to_user'].get('messages', [])
                    if messages and hasattr(messages[-1], 'content'):
                        print(messages[-1].content)
                        break
            
            # Update our local state with the latest from the orchestrator
            final_state = await orchestrator.run()
            state = final_state
            
        except KeyboardInterrupt:
            print("
            break
        except Exception as e:
            logging.error(f"Error in interactive mode: {e}")
            print(f"❌ Error: {e}")


async def batch_mode(orchestrator: OrchestratorAgent, message: str, max_iterations: int) -> None:
    """
    Run the orchestrator in batch mode with a single message.
    """
    print(f"🚀 Running orchestrator with message: '{message}'")
    print(f"📊 Max iterations: {max_iterations}")
    print("=" * 50)
    
    try:
        final_state = await orchestrator.run(
            initial_message=message,
            max_iterations=max_iterations
        )
        
        print("
        print(f"📝 Total messages: {len(final_state.messages)}")
        print(f"🤖 Background agents: {len(final_state.background_agents)}")
        
        # Show final status
        if final_state.background_agents:
            print("
            for agent_id, agent in final_state.background_agents.items():
                print(f"  - {agent.agent_name} ({agent.thread_id}): {agent.status.value}")
        
    except Exception as e:
        logging.error(f"Error in batch mode: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LangGraph Orchestrator Agent - Manage background agents asynchronously"
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file (JSON format)"
    )
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    parser.add_argument(
        "--message", "-m",
        help="Run in batch mode with this message (non-interactive)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum iterations for batch mode (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Load agent endpoints
    agent_endpoints = load_agent_endpoints(args.config)
    logging.info(f"Loaded agent endpoints: {list(agent_endpoints.keys())}")
    
    # Create orchestrator
    orchestrator = OrchestratorAgent(agent_endpoints=agent_endpoints)
    
    # Run in appropriate mode
    if args.message:
        await batch_mode(orchestrator, args.message, args.max_iterations)
    else:
        await interactive_mode(orchestrator)


if __name__ == "__main__":
    asyncio.run(main())

