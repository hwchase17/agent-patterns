#!/usr/bin/env python3
"""
Main demonstration script for the LangGraph Agent.

This script shows how to use the agent with various example interactions,
including tool usage and automatic conversation summarization when token limits are reached.
"""

import os
import asyncio
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Import the agent graph
from src.agent_patterns.graph import graph
from src.agent_patterns.state import AgentState


def print_separator(title: str = "") -> None:
    """Print a visual separator with optional title."""
    if title:
        print(f"
    else:
        print("="*60)


def print_message(message: Any, prefix: str = "") -> None:
    """Print a message with proper formatting."""
    if hasattr(message, 'content'):
        content = message.content
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print(f"{prefix}🤖 Assistant: {content}")
            for tool_call in message.tool_calls:
                print(f"   🔧 Tool Call: {tool_call['name']}({tool_call['args']})")
        else:
            print(f"{prefix}🤖 Assistant: {content}")
    elif hasattr(message, 'name'):
        # Tool message
        print(f"{prefix}🛠️  Tool Result: {message.content}")
    else:
        print(f"{prefix}📝 {message}")


def check_environment() -> bool:
    """Check if required environment variables are set."""
    model_name = os.getenv("MODEL_NAME", "gpt-4")
    
    if model_name.startswith("gpt"):
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Error: OPENAI_API_KEY not found in environment variables.")
            print("Please set your OpenAI API key in the .env file.")
            return False
    elif model_name.startswith("claude"):
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("❌ Error: ANTHROPIC_API_KEY not found in environment variables.")
            print("Please set your Anthropic API key in the .env file.")
            return False
    
    return True


async def run_conversation(messages: list, thread_id: str = "demo") -> Dict[str, Any]:
    """Run a conversation with the agent."""
    config = {"configurable": {"thread_id": thread_id}}
    
    # Create initial state
    initial_state = {
        "messages": messages,
        "current_token_count": 0,
        "max_token_limit": int(os.getenv("MAX_TOKEN_LIMIT", "4000")),
        "summarized": False,
        "original_message_count": None
    }
    
    # Run the graph
    result = await graph.ainvoke(initial_state, config)
    return result


async def demo_basic_conversation():
    """Demonstrate basic conversation without tools."""
    print_separator("Basic Conversation Demo")
    print("👋 Starting a basic conversation with the agent...")
    
    messages = [HumanMessage(content="Hello! Can you introduce yourself and tell me what you can do?")]
    
    result = await run_conversation(messages, "basic_demo")
    
    print("
    for msg in result["messages"]:
        if hasattr(msg, 'content'):
            if msg.content.startswith("Hello"):  # User message
                print(f"👤 User: {msg.content}")
            else:
                print_message(msg)
    
    print(f"
    print(f"📝 Summarized: {result.get('summarized', False)}")


async def demo_tool_usage():
    """Demonstrate various tool usage scenarios."""
    print_separator("Tool Usage Demo")
    print("🔧 Demonstrating various tools available to the agent...")
    
    # Weather tool
    print("
    messages = [HumanMessage(content="What's the weather like in New York?")]
    result = await run_conversation(messages, "weather_demo")
    
    for msg in result["messages"][-2:]:  # Show last 2 messages (assistant + tool result)
        print_message(msg, "   ")
    
    # Calculator tool
    print("
    messages = [HumanMessage(content="Can you calculate 15 * 23 + 47 for me?")]
    result = await run_conversation(messages, "calc_demo")
    
    for msg in result["messages"][-2:]:
        print_message(msg, "   ")
    
    # Knowledge search tool
    print("
    messages = [HumanMessage(content="Tell me about LangGraph")]
    result = await run_conversation(messages, "knowledge_demo")
    
    for msg in result["messages"][-2:]:
        print_message(msg, "   ")
    
    # Random fact tool
    print("
    messages = [HumanMessage(content="Give me a random interesting fact")]
    result = await run_conversation(messages, "fact_demo")
    
    for msg in result["messages"][-2:]:
        print_message(msg, "   ")


async def demo_multiple_tools():
    """Demonstrate using multiple tools in one conversation."""
    print_separator("Multiple Tools Demo")
    print("🔄 Demonstrating multiple tool usage in a single conversation...")
    
    messages = [
        HumanMessage(content="Hi! Can you help me with a few things?"),
    ]
    
    # Start conversation
    result = await run_conversation(messages, "multi_demo")
    
    # Add more requests
    messages = result["messages"] + [
        HumanMessage(content="First, what's the weather in London?")
    ]
    result = await run_conversation(messages, "multi_demo")
    
    messages = result["messages"] + [
        HumanMessage(content="Now calculate 100 / 4 + 25")
    ]
    result = await run_conversation(messages, "multi_demo")
    
    messages = result["messages"] + [
        HumanMessage(content="And give me a random fact!")
    ]
    result = await run_conversation(messages, "multi_demo")
    
    print("
    for i, msg in enumerate(result["messages"]):
        if hasattr(msg, 'content'):
            if "Hi! Can you help" in msg.content or "First, what's" in msg.content or "Now calculate" in msg.content or "And give me" in msg.content:
                print(f"👤 User: {msg.content}")
            else:
                print_message(msg)
    
    print(f"
    print(f"📝 Conversation summarized: {result.get('summarized', False)}")


async def demo_token_limit_and_summarization():
    """Demonstrate token limit handling and conversation summarization."""
    print_separator("Token Limit & Summarization Demo")
    print("📏 Demonstrating automatic conversation summarization when approaching token limits...")
    
    # Create a conversation that will exceed token limits
    long_messages = [
        HumanMessage(content="Let's have a long conversation. Can you tell me about artificial intelligence in detail?"),
    ]
    
    # Start with a detailed AI explanation
    result = await run_conversation(long_messages, "token_demo")
    
    # Add more long messages to trigger summarization
    for i in range(3):
        long_messages = result["messages"] + [
            HumanMessage(content=f"That's interesting! Can you also explain machine learning, deep learning, neural networks, and their applications in detail? This is question {i+1} of several I have about AI and related technologies. Please be comprehensive in your response.")
        ]
        result = await run_conversation(long_messages, "token_demo")
        
        print(f"
        print(f"   Token count: {result.get('current_token_count', 'N/A')}")
        print(f"   Summarized: {result.get('summarized', False)}")
        print(f"   Message count: {len(result['messages'])}")
        
        if result.get('summarized'):
            print("   ✅ Conversation was summarized to stay within token limits!")
            break
    
    print("
    for msg in result["messages"]:
        if hasattr(msg, 'content'):
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            if msg.content.startswith("Let's have a long") or "That's interesting" in msg.content:
                print(f"👤 User: {content_preview}")
            else:
                print(f"🤖 Assistant: {content_preview}")


async def main():
    """Main demonstration function."""
    print("🚀 LangGraph Agent Demonstration")
    print("This script demonstrates the capabilities of the LangGraph agent.")
    
    # Check environment
    if not check_environment():
        print("
        return
    
    print("✅ Environment check passed!")
    
    try:
        # Run demonstrations
        await demo_basic_conversation()
        await demo_tool_usage()
        await demo_multiple_tools()
        await demo_token_limit_and_summarization()
        
        print_separator("Demo Complete")
        print("🎉 All demonstrations completed successfully!")
        print("💡 You can now use the agent in your own applications by importing the graph from src.agent_patterns.graph")
        
    except Exception as e:
        print(f"
        print("Please check your API keys and environment configuration.")


if __name__ == "__main__":
    asyncio.run(main())

