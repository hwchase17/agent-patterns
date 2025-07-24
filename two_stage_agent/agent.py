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


def create_review_agent(model: Any, custom_prompt: Optional[str] = None) -> Runnable:
    """
    Create a review agent using LangGraph's create_react_agent.
    
    The review agent is designed to evaluate outputs from other agents and provide
    structured feedback indicating whether the result is acceptable or needs improvement.
    
    Args:
        model: The language model to use for the review agent
        custom_prompt: Optional custom prompt for the review agent
        
    Returns:
        A configured review agent that returns structured ReviewResult feedback
    """
    
    # Default comprehensive review prompt
    default_prompt = """
You are an expert quality reviewer and evaluator. Your role is to carefully assess outputs from other AI agents and determine whether they adequately address the user's request.

## Evaluation Criteria:

1. **Completeness**: Does the output fully answer the user's question or complete the requested task?
2. **Accuracy**: Is the information provided accurate, factual, and well-reasoned?
3. **Relevance**: Does the response directly address what was asked without unnecessary tangents?
4. **Clarity**: Is the output clear, well-structured, and easy to understand?
5. **Depth**: Does the response provide sufficient detail and context where appropriate?

## Your Assessment Process:

1. Read the output carefully and compare it against the original user request
2. Evaluate each criterion above
3. Make a binary decision: is this output acceptable as-is, or does it need improvement?
4. If improvement is needed, provide specific, actionable feedback

## Response Guidelines:

- Set `is_acceptable` to `true` only if the output meets all criteria satisfactorily
- Set `is_acceptable` to `false` if any significant issues are present
- Provide constructive, specific feedback in the `feedback` field when rejecting
- Include your confidence level (0.0 to 1.0) in your assessment
- Be thorough but concise in your evaluation

## Examples of Unacceptable Outputs:
- Incomplete answers that don't fully address the question
- Factually incorrect information
- Vague or unclear responses
- Responses that miss the main point of the request
- Outputs with significant logical errors or inconsistencies

Remember: Your goal is to ensure high-quality outputs while providing helpful guidance for improvement when needed.
"""
    
    prompt = custom_prompt or default_prompt
    
    return create_react_agent(
        model=model,
        tools=[],  # Review agent focuses on evaluation, doesn't need external tools
        prompt=prompt,
        response_format=ReviewResult,
        name="review_agent"
    )


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
        
        # Create the review agent using the dedicated factory function
        self.review_agent = create_review_agent(
            model=model,
            custom_prompt=review_prompt
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
    
    def _review_node(self, state: TwoStageState) -> Dict[str, Any]:
        """Review the output from the initial execution."""
        current_result = state.get("current_result", "")
        
        if not current_result:
            return {
                "review_status": "rejected",
                "review_feedback": "No output to review"
            }
        
        # Create review message
        review_message = HumanMessage(
            content=f"Please review this output and determine if it's acceptable:\n\n{current_result}"
        )
        
        # Get review from the review agent
        review_result = self.review_agent.invoke({"messages": [review_message]})
        
        # Extract review response
        review_response = review_result["messages"][-1].content
        
        # Parse the review response to determine if acceptable
        # Look for key indicators in the response
        is_acceptable = any(indicator in review_response.lower() for indicator in [
            "acceptable", "approved", "good", "satisfactory", "adequate", "meets requirements"
        ]) and not any(indicator in review_response.lower() for indicator in [
            "not acceptable", "rejected", "needs improvement", "inadequate", "insufficient"
        ])
        
        if is_acceptable:
            return {
                "review_status": "approved",
                "review_feedback": review_response,
                "final_output": current_result
            }
        else:
            return {
                "review_status": "rejected",
                "review_feedback": review_response
            }
    
    def _should_continue(self, state: TwoStageState) -> Literal["continue", "finish"]:
        """Determine whether to continue the workflow or finish based on review status."""
        review_status = state.get("review_status", "pending")
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)
        
        # If approved, finish the workflow
        if review_status == "approved":
            return "finish"
        
        # If we've reached max iterations, finish anyway to prevent infinite loops
        if iteration_count >= max_iterations:
            return "finish"
        
        # If rejected and under max iterations, continue
        if review_status == "rejected":
            return "continue"
        
        # Default case (shouldn't happen, but safe fallback)
        return "finish"
    
    def _finalize_node(self, state: TwoStageState) -> Dict[str, Any]:
        """Finalize the workflow and prepare the final output."""
        final_output = state.get("final_output")
        current_result = state.get("current_result")
        review_status = state.get("review_status", "pending")
        iteration_count = state.get("iteration_count", 0)
        
        # Use final_output if available (approved), otherwise use current_result
        output = final_output or current_result or "No output generated"
        
        return {
            "final_output": output,
            "review_status": review_status,
            "iteration_count": iteration_count
        }
    def _review_node(self, state: TwoStageState) -> Dict[str, Any]:
        """Review the output from the initial execution."""
        current_result = state.get("current_result", "")
        
        if not current_result:
            return {
                "review_status": "rejected",
                "review_feedback": "No output to review"
            }
        
        # Create review message
        review_message = HumanMessage(
            content=f"Please review this output and determine if it's acceptable:
        )
        
        # Get review from the review agent
        review_result = self.review_agent.invoke({"messages": [review_message]})
        
        # Extract review response
        review_response = review_result["messages"][-1].content
        
        # Parse the review response to determine if acceptable
        # Look for key indicators in the response
        is_acceptable = any(indicator in review_response.lower() for indicator in [
            "acceptable", "approved", "good", "satisfactory", "adequate", "meets requirements"
        ]) and not any(indicator in review_response.lower() for indicator in [
            "not acceptable", "rejected", "needs improvement", "inadequate", "insufficient"
        ])
        
        if is_acceptable:
            return {
                "review_status": "approved",
                "review_feedback": review_response,
                "final_output": current_result
            }
        else:
            return {
                "review_status": "rejected", 
                "review_feedback": review_response
            }
    
    def _should_continue(self, state: TwoStageState) -> Literal["continue", "finish"]:
        """Determine whether to continue the workflow or finish based on review status."""
        review_status = state.get("review_status", "pending")
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)
        
        # If approved, finish the workflow
        if review_status == "approved":
            return "finish"
        
        # If we've reached max iterations, finish anyway to prevent infinite loops
        if iteration_count >= max_iterations:
            return "finish"
        
        # If rejected and under max iterations, continue
        if review_status == "rejected":
            return "continue"
        
        # Default case (shouldn't happen, but safe fallback)
        return "finish"
    
    def _finalize_node(self, state: TwoStageState) -> Dict[str, Any]:
        """Finalize the workflow and prepare the final output."""
        final_output = state.get("final_output")
        current_result = state.get("current_result")
        review_status = state.get("review_status", "pending")
        iteration_count = state.get("iteration_count", 0)
        
        # Use final_output if available (approved), otherwise use current_result
        output = final_output or current_result or "No output generated"
        
        return {
            "final_output": output,
            "review_status": review_status,
            "iteration_count": iteration_count
        }



