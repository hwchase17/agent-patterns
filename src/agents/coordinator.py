"""Main LangGraph coordinator agent that manages the workflow between sub-agents."""

from typing import Dict, Any
from langgraph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub

from ..state.agent_state import AgentState


class CoordinatorAgent:
    """Main coordinator agent that orchestrates the workflow between ReAct and Reviewer agents."""
    
    def __init__(self, openai_api_key: str):
        """Initialize the coordinator with necessary components."""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_api_key
        )
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow that coordinates between agents."""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each step in the workflow
        workflow.add_node("react_agent", self._react_agent_node)
        workflow.add_node("reviewer_agent", self._reviewer_agent_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Set the entry point
        workflow.set_entry_point("react_agent")
        
        # Add edges with conditional routing
        workflow.add_edge("react_agent", "reviewer_agent")
        workflow.add_conditional_edges(
            "reviewer_agent",
            self._should_continue,
            {
                "continue": "react_agent",
                "finish": "finalize"
            }
        )
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _react_agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Node that handles the ReAct agent processing."""
        # Create a simple ReAct-style agent
        tools = [
            Tool(
                name="search",
                description="Search for information when you need to find facts or data",
                func=lambda query: f"Search results for '{query}': [This is a mock search result]"
            ),
            Tool(
                name="calculate",
                description="Perform mathematical calculations",
                func=lambda expression: f"Calculation result: {expression} = [mock result]"
            )
        ]
        
        # Get ReAct prompt from hub or create a simple one
        try:
            react_prompt = hub.pull("hwchase17/react")
        except:
            # Fallback prompt if hub is not available
            react_prompt = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
        
        # Create and execute the ReAct agent
        agent = create_react_agent(self.llm, tools, react_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # Prepare input for the agent
        input_text = state.user_query
        if state.reviewer_feedback:
            input_text += f"
        
        # Execute the agent
        result = agent_executor.invoke({"input": input_text})
        
        # Update state
        return {
            "react_output": result["output"],
            "current_step": "review",
            "iteration_count": state.iteration_count + 1,
            "messages": state.messages + [AIMessage(content=result["output"])]
        }
    
    def _reviewer_agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Node that handles the reviewer agent evaluation."""
        review_prompt = f"""
You are a reviewer agent. Your job is to evaluate the output from a ReAct agent and decide whether:
1. The result is good enough and should be approved (respond with "APPROVE")
2. The result needs improvement and you should provide feedback (respond with "FEEDBACK: [your feedback]")

Original user query: {state.user_query}
ReAct agent output: {state.react_output}
Current iteration: {state.iteration_count}
Max iterations: {state.max_iterations}

Please evaluate the output and respond with either "APPROVE" or "FEEDBACK: [specific feedback]".
If we've reached the maximum iterations ({state.max_iterations}), you should approve to prevent infinite loops.
"""
        
        response = self.llm.invoke([HumanMessage(content=review_prompt)])
        review_content = response.content.strip()
        
        if review_content.startswith("APPROVE") or state.iteration_count >= state.max_iterations:
            return {
                "reviewer_decision": "approve",
                "current_step": "complete",
                "messages": state.messages + [HumanMessage(content="Approved by reviewer")]
            }
        else:
            feedback = review_content.replace("FEEDBACK:", "").strip()
            return {
                "reviewer_decision": "feedback",
                "reviewer_feedback": feedback,
                "current_step": "react",
                "messages": state.messages + [HumanMessage(content=feedback)]
            }
    
    def _finalize_node(self, state: AgentState) -> Dict[str, Any]:
        """Node that finalizes the workflow and sets the final result."""
        return {
            "final_result": state.react_output,
            "current_step": "complete"
        }
    
    def _should_continue(self, state: AgentState) -> str:
        """Conditional edge function to determine next step after reviewer."""
        if state.reviewer_decision == "approve" or state.iteration_count >= state.max_iterations:
            return "finish"
        else:
            return "continue"
    
    def run(self, user_query: str) -> Dict[str, Any]:
        """Run the complete workflow for a given user query."""
        initial_state = AgentState(user_query=user_query)
        final_state = self.workflow.invoke(initial_state)
        return final_state

