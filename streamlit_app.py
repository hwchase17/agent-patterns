"""
Streamlit App for LangGraph Agent Interaction and Feedback

This application provides a web interface for interacting with a LangGraph agent
deployed using 'langgraph dev'. It supports streaming responses, feedback collection,
configuration updates, and rerun functionality.
"""

import streamlit as st
import asyncio
import time
from typing import Dict, List, Any, Optional
from langgraph_sdk import get_sync_client
from langgraph.pregel.remote import RemoteGraph
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
LANGGRAPH_URL = "http://localhost:2024"
AGENT_NAME = "agent"
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant with access to tools. Use the available tools when needed to provide accurate and helpful responses."


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "last_user_input" not in st.session_state:
        st.session_state.last_user_input = ""
    if "agent_response" not in st.session_state:
        st.session_state.agent_response = ""
    if "show_feedback" not in st.session_state:
        st.session_state.show_feedback = False
    if "client" not in st.session_state:
        st.session_state.client = None
    if "remote_graph" not in st.session_state:
        st.session_state.remote_graph = None


def get_langgraph_client():
    """Initialize and return LangGraph client and RemoteGraph."""
    try:
        # Initialize the sync client
        client = get_sync_client(url=LANGGRAPH_URL)
        
        # Initialize RemoteGraph
        remote_graph = RemoteGraph(AGENT_NAME, url=LANGGRAPH_URL)
        
        return client, remote_graph
    except Exception as e:
        st.error(f"Failed to connect to LangGraph server: {e}")
        st.error("Make sure the LangGraph server is running with 'langgraph dev'")
        return None, None


def create_thread_if_needed(client):
    """Create a new thread if one doesn't exist."""
    if st.session_state.thread_id is None:
        try:
            thread = client.threads.create()
            st.session_state.thread_id = thread["thread_id"]
            st.success(f"Created new conversation thread: {st.session_state.thread_id}")
        except Exception as e:
            st.error(f"Failed to create thread: {e}")
            return False
    return True


def stream_agent_response(remote_graph, user_input: str, system_prompt: str):
    """Stream the agent response and display it in real-time."""
    if not st.session_state.thread_id:
        st.error("No active thread. Please refresh the page.")
        return ""
    
    # Configuration with thread_id and system prompt
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "system_prompt": system_prompt
        }
    }
    
    # Input message
    input_message = {
        "messages": [{"role": "user", "content": user_input}]
    }
    
    # Create placeholder for streaming response
    response_placeholder = st.empty()
    full_response = ""
    
    try:
        # Stream the response
        with st.spinner("Agent is thinking..."):
            for chunk in remote_graph.stream(input_message, config=config, stream_mode="updates"):
                # Process different types of chunks
                for node_name, node_output in chunk.items():
                    if "messages" in node_output:
                        messages = node_output["messages"]
                        for message in messages:
                            if hasattr(message, 'content') and message.content:
                                if message.type == "ai":
                                    full_response += message.content
                                    response_placeholder.markdown(f"**Agent:** {full_response}")
                            elif isinstance(message, dict) and message.get("content"):
                                if message.get("role") == "assistant":
                                    full_response += message["content"]
                                    response_placeholder.markdown(f"**Agent:** {full_response}")
        
        # If no response was captured through streaming, try to get the final state
        if not full_response:
            try:
                final_state = remote_graph.get_state(config)
                if final_state and "messages" in final_state.values:
                    messages = final_state.values["messages"]
                    for message in reversed(messages):  # Get the last AI message
                        if (hasattr(message, 'type') and message.type == "ai") or \
                           (isinstance(message, dict) and message.get("role") == "assistant"):
                            content = message.content if hasattr(message, 'content') else message.get("content", "")
                            if content:
                                full_response = content
                                response_placeholder.markdown(f"**Agent:** {full_response}")
                                break
            except Exception as e:
                st.warning(f"Could not retrieve final state: {e}")
        
        return full_response
        
    except Exception as e:
        st.error(f"Error streaming response: {e}")
        return ""


def display_chat_history():
    """Display the chat history."""
    st.subheader("💬 Chat History")
    
    if not st.session_state.messages:
        st.info("No messages yet. Start a conversation below!")
        return
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Agent:** {message['content']}")
        st.markdown("---")


def handle_feedback_and_rerun():
    """Handle feedback collection and configuration update."""
    if not st.session_state.show_feedback:
        return
    
    st.subheader("📝 Provide Feedback")
    st.write("How was the agent's response? Your feedback will be used to improve the system prompt.")
    
    feedback = st.text_area(
        "Enter your feedback:",
        placeholder="e.g., The response was too verbose, please be more concise...",
        height=100
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Update Configuration", type="primary"):
            if feedback.strip():
                # Update system prompt based on feedback
                updated_prompt = f"{st.session_state.system_prompt}
                st.session_state.system_prompt = updated_prompt
                st.success("Configuration updated with your feedback!")
                
                # Show the updated system prompt
                with st.expander("View Updated System Prompt"):
                    st.text_area("Current System Prompt:", value=st.session_state.system_prompt, height=150, disabled=True)
            else:
                st.warning("Please provide feedback before updating the configuration.")
    
    with col2:
        if st.button("Rerun Previous Input", type="secondary"):
            if st.session_state.last_user_input:
                st.info(f"Rerunning with input: '{st.session_state.last_user_input}'")
                
                # Clear the previous response from display
                st.session_state.show_feedback = False
                
                # Rerun the agent with updated configuration
                if st.session_state.remote_graph:
                    response = stream_agent_response(
                        st.session_state.remote_graph, 
                        st.session_state.last_user_input, 
                        st.session_state.system_prompt
                    )
                    
                    if response:
                        # Update the last message in history
                        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                            st.session_state.messages[-1]["content"] = response
                        else:
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        st.session_state.agent_response = response
                        st.session_state.show_feedback = True
                        st.rerun()
            else:
                st.warning("No previous input to rerun.")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="LangGraph Agent Chat",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 LangGraph Agent Chat Interface")
    st.markdown("Connect to your LangGraph agent, chat with streaming responses, and provide feedback!")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Connection status
        if st.button("Connect to LangGraph Server"):
            client, remote_graph = get_langgraph_client()
            if client and remote_graph:
                st.session_state.client = client
                st.session_state.remote_graph = remote_graph
                st.success("Connected to LangGraph server!")
                
                # Create thread
                create_thread_if_needed(client)
        
        # Display connection status
        if st.session_state.client:
            st.success("✅ Connected to LangGraph")
            if st.session_state.thread_id:
                st.info(f"Thread ID: {st.session_state.thread_id[:8]}...")
        else:
            st.error("❌ Not connected")
            st.info("Click 'Connect to LangGraph Server' to start")
        
        # System prompt configuration
        st.subheader("System Prompt")
        current_prompt = st.text_area(
            "Current system prompt:",
            value=st.session_state.system_prompt,
            height=150,
            help="This prompt guides the agent's behavior"
        )
        
        if st.button("Update System Prompt"):
            st.session_state.system_prompt = current_prompt
            st.success("System prompt updated!")
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat input
        if st.session_state.client and st.session_state.remote_graph:
            user_input = st.text_input(
                "Enter your message:",
                placeholder="Ask me anything! I have access to weather information...",
                key="user_input"
            )
            
            if st.button("Send", type="primary") and user_input:
                # Add user message to history
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.last_user_input = user_input
                
                # Stream agent response
                response = stream_agent_response(st.session_state.remote_graph, user_input, st.session_state.system_prompt)
                
                if response:
                    # Add agent response to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.agent_response = response
                    st.session_state.show_feedback = True
                    st.rerun()
        else:
            st.info("Please connect to the LangGraph server first using the sidebar.")
        
        # Display chat history
        display_chat_history()
    
    with col2:
        # Feedback and rerun section
        handle_feedback_and_rerun()


if __name__ == "__main__":
    main()


