"""
Simple LangGraph chat agent with configurable system prompt and tool calling.
"""

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages


class ConfigSchema(TypedDict):
    """Configuration schema for the agent."""
    system_prompt: str


class State(TypedDict):
    """The agent state containing messages."""
    messages: Annotated[list[BaseMessage], add_messages]


def create_agent():
    """Create and return the LangGraph agent with tools."""
    
    # Initialize the language model
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Create a simple search tool for demonstration
    search_tool = TavilySearchResults(
        max_results=3,
        search_depth="basic",
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )
    
    tools = [search_tool]
    
    # Create the react agent with configurable system prompt
    agent = create_react_agent(
        model=model,
        tools=tools,
        state_schema=State,
        config_schema=ConfigSchema,
    )
    
    return agent


# Create the graph instance for LangGraph server
graph = create_agent()

