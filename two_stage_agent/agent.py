"""
Two-Stage Agent Workflow with Review Mechanism

This module implements a generic two-stage agent that accepts any ReAct agent
and adds a review mechanism. The workflow consists of:
1. Initial execution node that runs the provided ReAct agent
2. Review node that evaluates the output using another agent
3. Conditional logic to either finish or loop back with feedback
"""

from typing import Annotated, Literal, Optional, Any, Dict
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langchain_core.runnables import Runnable


class ReviewResult(BaseModel):
    """Structured output for review agent assessment."""
    is_acceptable: bool = Field(description="Whether the output is acceptable")
    feedback: str = Field(description="Feedback for improvement if not acceptable")
    confidence: float = Field(description="Confidence in the assessment (0-1)", ge=0, le=1)


class TwoStageState(MessagesState):
    """Extended state for two-stage agent workflow."""
    # Review tracking
    review_status: str = "pending"  # "pending", "approved", "rejected"
    iteration_count: int = 0
    max_iterations: int = 3
    
    # Feedback and results
    current_result: Optional[str] = None
    review_feedback: Optional[str] = None
    final_output: Optional[str] = None


class TwoStageAgent:
    """
    A generic two-stage agent that wraps any ReAct agent with a review mechanism.
    
    The workflow:
    1. Execute the provided ReAct agent
    2. Review the output with a review agent
    3. Either finish (if approved) or loop back with feedback (if rejected)
    """
    
    def __init__(
        self,
        react_agent: Runnable,
        model: Any,
        max_iterations: int = 3,
        review_prompt: Optional[str] = None
    ):
        """
        Initialize the TwoStageAgent.
        
        Args:
            react_agent: The ReAct agent to wrap with review mechanism
            model: The language model to use for the review agent
            max_iterations: Maximum number of iterations before stopping
            review_prompt: Custom prompt for the review agent
        """
        self.react_agent = react_agent
        self.model = model
        self.max_iterations = max_iterations
        
        # Default review prompt
        self.review_prompt = review_prompt or """
You are a quality reviewer. Your job is to evaluate the output from another agent and determine if it adequately addresses the user's request.

Evaluation criteria:
1. Does the output directly answer the user's question or complete the requested task?
2. Is the information accurate and well-reasoned?
3. Is the response complete and not missing important details?
4. Is the output clear and well-structured?

If the output is acceptable, set is_acceptable to true.
If the output needs improvement, set is_acceptable to false and provide specific feedback on what needs to be improved.

Be constructive in your feedback and specific about what changes are needed.
"""
        
        # Create the review agent
        self.review_agent = create_react_agent(
            model=self.model,
            tools=[],  # Review agent doesn't need tools, just evaluation
            prompt=self.review_prompt,
            response_format=ReviewResult,
            name="review_agent"
        )
        
        # Build the workflow graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the two-stage workflow graph."""
        builder = StateGraph(TwoStageState)
        
        # Add nodes
        builder.add_node("initial_execution", self._initial_execution_node)
        builder.add_node("review", self._review_node)
        builder.add_node("finalize", self._finalize_node)
        
        # Add edges
        builder.add_edge(START, "initial_execution")
        builder.add_edge("initial_execution", "review")
        builder.add_conditional_edges(
            "review",
            self._should_continue,
            {
                "continue": "initial_execution",
                "finish": "finalize"
            }
        )
        builder.add_edge("finalize", END)
        
        return builder.compile()
    
    def _initial_execution_node(self, state: TwoStageState) -> Dict[str, Any]:
        """Execute the initial ReAct agent."""
        # Increment iteration count
        iteration_count = state.get("iteration_count", 0) + 1
        
        # If we have feedback from previous iteration, add it to the messages
        messages = state["messages"].copy()
        if state.get("review_feedback") and iteration_count > 1:
            feedback_msg = HumanMessage(
                content=f"Previous attempt needs improvement. Feedback: {state['review_feedback']}. Please revise your response."
            )
            messages.append(feedback_msg)
        
        # Execute the ReAct agent
        result = self.react_agent.invoke({"messages": messages})
        
        # Extract the final response
        final_message = result["messages"][-1]
        current_result = final_message.content if hasattr(final_message, 'content') else str(final_message)
        
        return {
            "messages": result["messages"],
            "current_result": current_result,
            "iteration_count": iteration_count,
            "review_status": "pending"
        }
    
