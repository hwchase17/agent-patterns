"""
Streamlit app for LangGraph agent with feedback loop and configuration updates.
"""

import streamlit as st
import time
from typing import Dict, Any, List
from langgraph_sdk import get_sync_client
from langchain_core.messages import HumanMessage, AIMessage


# Configure Streamlit page
st.set_page_config(
    page_title="LangGraph Agent with Feedback Loop",
    page_icon="🤖",
    layout="wide"
)


def initialize_client():
    """Initialize the LangGraph SDK client."""
    try:
        client = get_sync_client(url="http://localhost:2024")
        return client
    except Exception as e:
        st.error(f"Failed to connect to LangGraph server: {e}")
        st.error("Make sure to run 'langgraph dev' in your terminal first!")
        return None


def stream_agent_response(client, user_input: str, system_prompt: str):
    """Stream the agent response and return the complete conversation."""
    messages = []
    response_container = st.empty()
    current_response = ""
    
    try:
        # Create the input for the agent
        agent_input = {
            "messages": [{"role": "human", "content": user_input}]
        }
        
        # Configuration with system prompt
        config = {
            "configurable": {
                "system_prompt": system_prompt
            }
        }
        
        # Stream the agent execution
        for chunk in client.runs.stream(
            None,  # Threadless run
            "agent",  # Assistant ID from langgraph.json
            input=agent_input,
            config=config,
            stream_mode="messages"
        ):
            if chunk.event == "messages/partial":
                # Handle partial message updates
                if chunk.data and len(chunk.data) > 0:
                    message = chunk.data[-1]  # Get the latest message
                    if hasattr(message, 'content') and message.content:
                        current_response = message.content
                        response_container.markdown(f"**Assistant:** {current_response}")
            
            elif chunk.event == "messages/complete":
                # Handle complete messages
                if chunk.data and len(chunk.data) > 0:
                    for message in chunk.data:
                        if hasattr(message, 'content') and message.content:
                            if hasattr(message, 'type') and message.type == "ai":
                                current_response = message.content
                                messages.append({"role": "assistant", "content": message.content})
                            elif hasattr(message, 'type') and message.type == "human":
                                messages.append({"role": "user", "content": message.content})
        
        # Final update with complete response
        if current_response:
            response_container.markdown(f"**Assistant:** {current_response}")
            if not any(msg.get("role") == "assistant" for msg in messages):
                messages.append({"role": "assistant", "content": current_response})
        
        return messages, current_response
        
    except Exception as e:
        st.error(f"Error during agent execution: {e}")
        return [], ""


def update_system_prompt_based_on_feedback(current_prompt: str, feedback: str) -> str:
    """
    Simple logic to update system prompt based on user feedback.
    In a real application, this could use an LLM to intelligently modify the prompt.
    """
    feedback_lower = feedback.lower()
    
    # Simple keyword-based prompt modifications
    if "more detailed" in feedback_lower or "more verbose" in feedback_lower:
        if "detailed" not in current_prompt.lower():
            return current_prompt + " Provide detailed and comprehensive responses."
    
    elif "shorter" in feedback_lower or "concise" in feedback_lower or "brief" in feedback_lower:
        if "concise" not in current_prompt.lower():
            return current_prompt + " Keep responses concise and to the point."
    
    elif "friendly" in feedback_lower or "casual" in feedback_lower:
        if "friendly" not in current_prompt.lower():
            return current_prompt + " Use a friendly and casual tone."
    
    elif "formal" in feedback_lower or "professional" in feedback_lower:
        if "professional" not in current_prompt.lower():
            return current_prompt + " Maintain a professional and formal tone."
    
    elif "creative" in feedback_lower:
        if "creative" not in current_prompt.lower():
            return current_prompt + " Be creative and think outside the box."
    
    else:
        # Generic improvement based on feedback
        return current_prompt + f" Note: User feedback - {feedback}"
    
    return current_prompt


def main():
    """Main Streamlit application."""
    st.title("🤖 LangGraph Agent with Feedback Loop")
    st.markdown("---")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_user_input" not in st.session_state:
        st.session_state.last_user_input = ""
    if "last_response" not in st.session_state:
        st.session_state.last_response = ""
    if "show_feedback" not in st.session_state:
        st.session_state.show_feedback = False
    
    # Initialize client
    client = initialize_client()
    if not client:
        st.stop()
    
    # Configuration section
    st.header("⚙️ Configuration")
    
    # System prompt configuration
    default_prompt = "You are a helpful AI assistant. Be concise and helpful in your responses."
    system_prompt = st.text_area(
        "System Prompt:",
        value=st.session_state.get("system_prompt", default_prompt),
        height=100,
        help="Configure how the agent should behave"
    )
    st.session_state.system_prompt = system_prompt
    
    st.markdown("---")
    
    # Chat interface
    st.header("💬 Chat with Agent")
    
    # User input
    user_input = st.text_input(
        "Your message:",
        placeholder="Ask me anything...",
        key="user_input"
    )
    
    # Run agent button
    if st.button("🚀 Run Agent", type="primary"):
        if user_input.strip():
            st.session_state.last_user_input = user_input
            st.session_state.show_feedback = False
            
            # Display user message
            st.markdown(f"**You:** {user_input}")
            
            # Stream agent response
            with st.spinner("Agent is thinking..."):
                messages, response = stream_agent_response(client, user_input, system_prompt)
                st.session_state.messages = messages
                st.session_state.last_response = response
                st.session_state.show_feedback = True
        else:
            st.warning("Please enter a message first!")
    
    # Feedback section (shown after agent response)
    if st.session_state.show_feedback and st.session_state.last_response:
        st.markdown("---")
        st.header("📝 Provide Feedback")
        
        feedback = st.text_area(
            "How can the agent improve its response?",
            placeholder="e.g., 'Be more detailed', 'Use simpler language', 'Be more creative'...",
            height=80
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Submit Feedback & Update Config"):
                if feedback.strip():
                    # Update system prompt based on feedback
                    updated_prompt = update_system_prompt_based_on_feedback(
                        st.session_state.system_prompt, feedback
                    )
                    st.session_state.system_prompt = updated_prompt
                    
                    st.success("✅ Configuration updated based on your feedback!")
                    st.info(f"**Updated System Prompt:** {updated_prompt}")
                    
                    # Show rerun option
                    st.session_state.show_rerun = True
                else:
                    st.warning("Please provide feedback first!")
        
        with col2:
            if st.session_state.get("show_rerun", False):
                if st.button("🔄 Rerun with Updated Config"):
                    if st.session_state.last_user_input:
                        st.session_state.show_feedback = False
                        st.session_state.show_rerun = False
                        
                        # Display user message again
                        st.markdown(f"**You:** {st.session_state.last_user_input}")
                        
                        # Stream agent response with updated config
                        with st.spinner("Agent is thinking with updated configuration..."):
                            messages, response = stream_agent_response(
                                client, 
                                st.session_state.last_user_input, 
                                st.session_state.system_prompt
                            )
                            st.session_state.messages = messages
                            st.session_state.last_response = response
                            st.session_state.show_feedback = True
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("""
        **How to use:**
        1. Configure the system prompt
        2. Enter your message
        3. Click "Run Agent" to get a response
        4. Provide feedback on the response
        5. Submit feedback to update configuration
        6. Rerun with the updated configuration
        
        **Features:**
        - Real-time streaming responses
        - Configurable system prompts
        - Feedback-based configuration updates
        - Tool calling capabilities (web search)
        """)
        
        if st.button("🔄 Reset Configuration"):
            st.session_state.system_prompt = default_prompt
            st.session_state.messages = []
            st.session_state.show_feedback = False
            st.session_state.show_rerun = False
            st.rerun()


if __name__ == "__main__":
    main()

