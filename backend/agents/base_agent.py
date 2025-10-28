"""
Base Agent Class for Multi-Agent Reasoning System
Provides common functionality for all specialized agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import time
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ReasoningStep(BaseModel):
    """Represents a single step in the reasoning chain"""
    step_name: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    duration: float
    confidence: float
    agent_used: str
    timestamp: datetime

class AgentResult(BaseModel):
    """Standard result format for all agents"""
    success: bool
    data: Dict[str, Any]
    confidence: float
    reasoning_steps: List[ReasoningStep]
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

class BaseAgent(ABC):
    """Base class for all specialized agents in the reasoning system"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.success_rate = 0.0
        
    async def execute(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Execute the agent with performance tracking"""
        start_time = time.time()
        self.execution_count += 1
        
        try:
            logger.info(f"Executing {self.name} agent with input: {input_data}")
            
            # Call the specific agent implementation
            result = await self._process(input_data, context or {})
            
            # Calculate performance metrics
            duration = time.time() - start_time
            self.total_execution_time += duration
            
            # Create reasoning step
            reasoning_step = ReasoningStep(
                step_name=self.name,
                input_data=input_data,
                output_data=result.data,
                duration=duration,
                confidence=result.confidence,
                agent_used=self.name,
                timestamp=datetime.now()
            )
            
            # Add reasoning step to result
            result.reasoning_steps.append(reasoning_step)
            
            # Update success rate
            if result.success:
                self.success_rate = (self.success_rate * (self.execution_count - 1) + 1.0) / self.execution_count
            else:
                self.success_rate = (self.success_rate * (self.execution_count - 1)) / self.execution_count
            
            logger.info(f"{self.name} agent completed in {duration:.2f}s with confidence {result.confidence:.2f}")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.total_execution_time += duration
            
            logger.error(f"Error in {self.name} agent: {str(e)}")
            
            # Update success rate for failure
            self.success_rate = (self.success_rate * (self.execution_count - 1)) / self.execution_count
            
            return AgentResult(
                success=False,
                data={},
                confidence=0.0,
                reasoning_steps=[],
                error=str(e),
                metadata={"duration": duration, "agent": self.name}
            )
    
    @abstractmethod
    async def _process(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        """Implement the specific agent logic"""
        pass
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this agent"""
        avg_execution_time = self.total_execution_time / max(self.execution_count, 1)
        
        return {
            "agent_name": self.name,
            "execution_count": self.execution_count,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": avg_execution_time,
            "success_rate": self.success_rate,
            "description": self.description
        }