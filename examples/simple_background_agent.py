"""
Simple example background agent for testing the orchestrator system.

This agent implements a basic LangGraph agent that can be spawned remotely
and provides HTTP endpoints for status monitoring and result retrieval.
"""
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import logging
from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState


logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status of a background task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SpawnRequest(BaseModel):
    """Request to spawn a new agent task."""
    task_description: str
    input_data: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class SpawnResponse(BaseModel):
    """Response from spawning an agent."""
    thread_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    """Response for status check."""
    thread_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TaskState(MessagesState):
    """State for individual background tasks."""
    task_description: str
    input_data: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None


class BackgroundTask:
    """Represents a running background task."""
    
    def __init__(self, thread_id: str, task_description: str, input_data: Dict[str, Any]):
        self.thread_id = thread_id
        self.task_description = task_description
        self.input_data = input_data
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.started_at = datetime.now()
        self.completed_at = None
        self.task = None
        self.cancelled = False


class SimpleBackgroundAgent:
    """
    Simple background agent that can process tasks asynchronously.
    
    This agent demonstrates:
    - Receiving task requests via HTTP
    - Processing tasks in the background using LangGraph
    - Providing status updates and results
    - Handling cancellation
    """
    
    def __init__(self):
        self.tasks: Dict[str, BackgroundTask] = {}
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build a simple LangGraph for processing tasks."""
        
        def process_task(state: TaskState) -> TaskState:
            """Process the task - this is where the actual work happens."""
            task_description = state["task_description"]
            input_data = state["input_data"]
            
            # Simulate some work based on the task description
            if "math" in task_description.lower():
                # Simple math task
                if "add" in task_description.lower() and "numbers" in input_data:
                    numbers = input_data["numbers"]
                    result = sum(numbers)
                    state["result"] = f"The sum of {numbers} is {result}"
                elif "multiply" in task_description.lower() and "numbers" in input_data:
                    numbers = input_data["numbers"]
                    result = 1
                    for num in numbers:
                        result *= num
                    state["result"] = f"The product of {numbers} is {result}"
                else:
                    state["result"] = "Math task completed, but no specific operation found"
            
            elif "text" in task_description.lower():
                # Simple text processing task
                if "text" in input_data:
                    text = input_data["text"]
                    word_count = len(text.split())
                    char_count = len(text)
                    state["result"] = f"Text analysis: {word_count} words, {char_count} characters"
                else:
                    state["result"] = "Text task completed, but no text provided"
            
            elif "wait" in task_description.lower():
                # Simulate a long-running task
                import time
                wait_time = input_data.get("seconds", 5)
                time.sleep(wait_time)
                state["result"] = f"Waited for {wait_time} seconds"
            
            else:
                # Default task
                state["result"] = f"Completed task: {task_description}"
            
            # Add a message to the conversation
            state["messages"].append(AIMessage(content=state["result"]))
            
            return state
        
        # Create the graph
        workflow = StateGraph(TaskState)
        workflow.add_node("process", process_task)
        workflow.set_entry_point("process")
        workflow.add_edge("process", END)
        
        return workflow.compile()
    
    async def spawn_task(self, request: SpawnRequest) -> str:
        """Spawn a new background task."""
        thread_id = str(uuid.uuid4())
        
        # Create task
        task = BackgroundTask(
            thread_id=thread_id,
            task_description=request.task_description,
            input_data=request.input_data
        )
        
        self.tasks[thread_id] = task
        
        # Start the task asynchronously
        task.task = asyncio.create_task(self._run_task(task))
        
        logger.info(f"Spawned task {thread_id}: {request.task_description}")
        return thread_id
    
    async def _run_task(self, task: BackgroundTask):
        """Run a task in the background."""
        try:
            task.status = TaskStatus.RUNNING
            
            # Check if cancelled before starting
            if task.cancelled:
                task.status = TaskStatus.CANCELLED
                return
            
            # Create initial state
            initial_state = TaskState(
                messages=[HumanMessage(content=task.task_description)],
                task_description=task.task_description,
                input_data=task.input_data
            )
            
            # Run the graph
            final_state = await asyncio.get_event_loop().run_in_executor(
                None, self.graph.invoke, initial_state
            )
            
            # Check if cancelled during execution
            if task.cancelled:
                task.status = TaskStatus.CANCELLED
                return
            
            # Task completed successfully
            task.result = final_state.get("result")
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            logger.info(f"Task {task.thread_id} completed successfully")
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            logger.error(f"Task {task.thread_id} failed: {e}")
    
    def get_task_status(self, thread_id: str) -> Optional[BackgroundTask]:
        """Get the status of a task."""
        return self.tasks.get(thread_id)
    
    def cancel_task(self, thread_id: str) -> bool:
        """Cancel a running task."""
        task = self.tasks.get(thread_id)
        if not task:
            return False
        
        task.cancelled = True
        if task.task and not task.task.done():
            task.task.cancel()
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        
        logger.info(f"Cancelled task {thread_id}")
        return True


# Create the agent instance
agent = SimpleBackgroundAgent()

# Create FastAPI app
app = FastAPI(title="Simple Background Agent", version="1.0.0")


@app.post("/spawn", response_model=SpawnResponse)
async def spawn_agent(request: SpawnRequest):
    """Spawn a new background agent task."""
    try:
        thread_id = await agent.spawn_task(request)
        return SpawnResponse(
            thread_id=thread_id,
            status="running",
            message=f"Task spawned with ID: {thread_id}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{thread_id}", response_model=StatusResponse)
async def get_status(thread_id: str):
    """Get the status of a background task."""
    task = agent.get_task_status(thread_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return StatusResponse(
        thread_id=thread_id,
        status=task.status.value,
        result=task.result,
        error=task.error,
        started_at=task.started_at,
        completed_at=task.completed_at
    )


@app.post("/cancel/{thread_id}")
async def cancel_agent(thread_id: str):
    """Cancel a running background task."""
    success = agent.cancel_task(thread_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {"message": f"Task {thread_id} cancelled"}


@app.get("/result/{thread_id}")
async def get_result(thread_id: str):
    """Get the result of a completed task."""
    task = agent.get_task_status(thread_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=400, 
            detail=f"Task is not completed. Current status: {task.status.value}"
        )
    
    return {"result": task.result}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "active_tasks": len(agent.tasks)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

