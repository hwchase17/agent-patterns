"""
Remote agent client for communicating with other LangGraph agents.
"""
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import httpx
import logging
from pydantic import BaseModel

from .state import BackgroundAgent, AgentStatus


logger = logging.getLogger(__name__)


class AgentRequest(BaseModel):
    """Request payload for spawning a remote agent."""
    task_description: str
    input_data: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentResponse(BaseModel):
    """Response from a remote agent."""
    thread_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RemoteAgentClient:
    """
    Client for communicating with remote LangGraph agents.
    
    This client handles:
    - Spawning remote agents and getting thread IDs
    - Checking status of running threads
    - Retrieving results from completed threads
    - Managing HTTP connections and error handling
    """
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        """
        Initialize the remote agent client.
        
        Args:
            timeout: HTTP request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
    
    async def spawn_agent(
        self,
        agent_endpoint: str,
        agent_name: str,
        task_description: str,
        input_data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BackgroundAgent:
        """
        Spawn a new remote agent and return a BackgroundAgent instance.
        
        Args:
            agent_endpoint: URL endpoint of the remote agent
            agent_name: Name/type of the agent
            task_description: Description of the task
            input_data: Input data for the agent
            config: Optional configuration for the agent
            metadata: Optional metadata
            
        Returns:
            BackgroundAgent instance with thread_id and initial status
            
        Raises:
            httpx.HTTPError: If the HTTP request fails
            ValueError: If the response is invalid
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        request_payload = AgentRequest(
            task_description=task_description,
            input_data=input_data,
            config=config or {},
            metadata=metadata or {}
        )
        
        # Try to spawn the agent with retries
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = await self._client.post(
                    f"{agent_endpoint}/spawn",
                    json=request_payload.model_dump(),
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                response_data = response.json()
                thread_id = response_data.get("thread_id")
                
                if not thread_id:
                    raise ValueError("No thread_id in response")
                
                # Create BackgroundAgent instance
                background_agent = BackgroundAgent(
                    thread_id=thread_id,
                    agent_name=agent_name,
                    agent_endpoint=agent_endpoint,
                    task_description=task_description,
                    status=AgentStatus.RUNNING,
                    metadata=metadata or {}
                )
                
                logger.info(f"Successfully spawned agent {agent_name} with thread_id: {thread_id}")
                return background_agent
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed to spawn agent: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # If all retries failed, raise the last exception
        raise last_exception
    
    async def check_agent_status(self, agent: BackgroundAgent) -> AgentStatus:
        """
        Check the status of a remote agent.
        
        Args:
            agent: BackgroundAgent instance to check
            
        Returns:
            Updated AgentStatus
            
        Raises:
            httpx.HTTPError: If the HTTP request fails
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            response = await self._client.get(
                f"{agent.agent_endpoint}/status/{agent.thread_id}"
            )
            response.raise_for_status()
            
            response_data = response.json()
            status_str = response_data.get("status", "unknown")
            
            # Map response status to our AgentStatus enum
            status_mapping = {
                "pending": AgentStatus.PENDING,
                "running": AgentStatus.RUNNING,
                "completed": AgentStatus.COMPLETED,
                "failed": AgentStatus.FAILED,
                "cancelled": AgentStatus.CANCELLED,
                "success": AgentStatus.COMPLETED,  # Alternative success status
                "error": AgentStatus.FAILED,      # Alternative error status
            }
            
            new_status = status_mapping.get(status_str.lower(), AgentStatus.FAILED)
            
            # Update agent with new status and any result/error
            result = response_data.get("result")
            error = response_data.get("error")
            
            agent.update_status(new_status, result=result, error=error)
            
            logger.debug(f"Agent {agent.thread_id} status: {new_status}")
            return new_status
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to check status for agent {agent.thread_id}: {e}")
            # Mark as failed if we can't reach it
            agent.update_status(AgentStatus.FAILED, error=str(e))
            return AgentStatus.FAILED
    
    async def check_multiple_agents(self, agents: List[BackgroundAgent]) -> Dict[str, AgentStatus]:
        """
        Check the status of multiple agents concurrently.
        
        Args:
            agents: List of BackgroundAgent instances to check
            
        Returns:
            Dictionary mapping thread_id to updated status
        """
        if not agents:
            return {}
        
        # Create tasks for concurrent status checking
        tasks = [
            asyncio.create_task(self.check_agent_status(agent))
            for agent in agents
        ]
        
        # Wait for all status checks to complete
        statuses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build result dictionary
        result = {}
        for agent, status in zip(agents, statuses):
            if isinstance(status, Exception):
                logger.error(f"Error checking agent {agent.thread_id}: {status}")
                agent.update_status(AgentStatus.FAILED, error=str(status))
                result[agent.thread_id] = AgentStatus.FAILED
            else:
                result[agent.thread_id] = status
        
        return result
    
    async def cancel_agent(self, agent: BackgroundAgent) -> bool:
        """
        Cancel a running remote agent.
        
        Args:
            agent: BackgroundAgent instance to cancel
            
        Returns:
            True if successfully cancelled, False otherwise
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            response = await self._client.post(
                f"{agent.agent_endpoint}/cancel/{agent.thread_id}"
            )
            response.raise_for_status()
            
            agent.update_status(AgentStatus.CANCELLED)
            logger.info(f"Successfully cancelled agent {agent.thread_id}")
            return True
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to cancel agent {agent.thread_id}: {e}")
            return False
    
    async def get_agent_result(self, agent: BackgroundAgent) -> Optional[Any]:
        """
        Get the result from a completed agent.
        
        Args:
            agent: BackgroundAgent instance
            
        Returns:
            Agent result if available, None otherwise
        """
        if agent.status != AgentStatus.COMPLETED:
            return None
        
        if agent.result is not None:
            return agent.result
        
        # Try to fetch result from remote agent
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            response = await self._client.get(
                f"{agent.agent_endpoint}/result/{agent.thread_id}"
            )
            response.raise_for_status()
            
            response_data = response.json()
            result = response_data.get("result")
            
            # Update agent with result
            agent.result = result
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to get result for agent {agent.thread_id}: {e}")
            return None

