"""Utility functions for token counting and message management."""

import tiktoken
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


def count_tokens(messages: List[BaseMessage], model: str = "gpt-4") -> int:
    """Count tokens in a list of messages.
    
    Args:
        messages: List of messages to count tokens for
        model: Model name to use for token counting (default: gpt-4)
        
    Returns:
        Total number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
    
    total_tokens = 0
    
    for message in messages:
        # Count tokens for message content
        if hasattr(message, 'content') and message.content:
            total_tokens += len(encoding.encode(str(message.content)))
        
        # Add tokens for message metadata (role, etc.)
        # This is an approximation - actual token count may vary slightly
        total_tokens += 4  # Approximate overhead per message
    
    return total_tokens


def create_summary_message(messages: List[BaseMessage], model: str = "gpt-4") -> str:
    """Create a summary of the conversation messages.
    
    Args:
        messages: List of messages to summarize
        model: Model name (for context)
        
    Returns:
        Summary text
    """
    if not messages:
        return "No messages to summarize."
    
    # Separate different types of messages
    human_messages = []
    ai_messages = []
    system_messages = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            human_messages.append(str(msg.content))
        elif isinstance(msg, AIMessage):
            ai_messages.append(str(msg.content))
        elif isinstance(msg, SystemMessage):
            system_messages.append(str(msg.content))
    
    # Create summary
    summary_parts = []
    
    if system_messages:
        summary_parts.append(f"System instructions: {'; '.join(system_messages[:2])}")
    
    if human_messages:
        summary_parts.append(f"User discussed: {'; '.join(human_messages[-3:])}")
    
    if ai_messages:
        summary_parts.append(f"Assistant provided: {'; '.join(ai_messages[-3:])}")
    
    summary = f"[CONVERSATION SUMMARY - {len(messages)} messages summarized]: " + "; ".join(summary_parts)
    
    return summary


def should_summarize(current_tokens: int, max_tokens: int, buffer_ratio: float = 0.8) -> bool:
    """Determine if messages should be summarized based on token count.
    
    Args:
        current_tokens: Current token count
        max_tokens: Maximum allowed tokens
        buffer_ratio: Ratio of max_tokens to trigger summarization (default: 0.8)
        
    Returns:
        True if summarization should be performed
    """
    return current_tokens >= (max_tokens * buffer_ratio)
