# Agricultural Workflow Engine
# Manages pre-built agricultural scenarios and step-by-step guidance

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from enum import Enum
import json

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"

class StepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"

class WorkflowStep:
    """Individual step in an agricultural workflow"""
    
    def __init__(self, step_id: str, title: str, description: str, 
                 tools_required: List[str] = None, estimated_time: int = 5,
                 prerequisites: List[str] = None, optional: bool = False):
        self.step_id = step_id
        self.title = title
        self.description = description
        self.tools_required = tools_required or []
        self.estimated_time = estimated_time  # minutes
        self.prerequisites = prerequisites or []
        self.optional = optional
        self.status = StepStatus.PENDING
        self.result_data = {}
        self.started_at = None
        self.completed_at = None
        self.notes = ""

class AgriculturalWorkflow:
    """Complete agricultural workflow with multiple steps"""
    
    def __init__(self, workflow_id: str, title: str, description: str, 
                 category: str, difficulty: str = "beginner"):
        self.workflow_id = workflow_id
        self.title = title
        self.description = description
        self.category = category  # crop_selection, pest_management, irrigation, harvest_timing
        self.difficulty = difficulty  # beginner, intermediate, advanced
        self.steps: List[WorkflowStep] = []
        self.status = WorkflowStatus.NOT_STARTED
        self.current_step_index = 0
        self.created_at = datetime.now(timezone.utc)
        self.started_at = None
        self.completed_at = None
        self.user_id = None
        self.progress_percentage = 0
        self.estimated_total_time = 0
        self.actual_time_spent = 0

class WorkflowEngine:
    """
    Agricultural Workflow Engine for pre-built farming scenarios
    Provides step-by-step guidance for complex agricultural processes
    """
    
    def __init__(self, db, mcp_client, voice_interface=None):
        self.db = db
        self.mcp_client = mcp_client
        self.voice_interface = voice_interface
        self.workflows = {}
        self.user_workflow_instances = {}
        
        # Initialize pre-built workflows
        self._initialize_workflows()
    
    def _initialize_workflows(self):
        """Initialize pre-built agricultural workflows"""
        
        # 1. Crop Selection Workflow
        crop_selection = AgriculturalWorkflow(
            "crop_selection",
            "Smart Crop Selection Guide",
            "Complete guide to select the best crops based on soil, weather, and market conditions",
            "crop_selection",
            "beginner"
        )
        
        crop_selection.steps = [
            WorkflowStep(
                "soil_analysis",
                "Analyze Soil Conditions",
                "Test and analyze your soil's NPK levels, pH, and organic content",
                tools_required=["soil-health"],
                estimated_time=10
            ),
            WorkflowStep(
                "weather_check",
                "Check Weather Forecast",
                "Review 7-day weather forecast and seasonal patterns",
                tools_required=["weather"],
                estimated_time=5
            ),
            WorkflowStep(
                "market_research",
                "Research Market Prices",
                "Check current and predicted prices for potential crops",
                tools_required=["crop-price", "mandi-price"],
                estimated_time=15
            ),
            WorkflowStep(
                "crop_recommendation",
                "Get Crop Recommendations",
                "Analyze all data to recommend the best crops for your conditions",
                tools_required=["search"],
                estimated_time=10,
                prerequisites=["soil_analysis", "weather_check", "market_research"]
            ),
            WorkflowStep(
                "financial_planning",
                "Plan Investment and Returns",
                "Calculate expected costs, yields, and profits",
                estimated_time=15,
                prerequisites=["crop_recommendation"]
            )
        ]
        
        # 2. Pest Management Workflow
        pest_management = AgriculturalWorkflow(
            "pest_management",
            "Integrated Pest Management",
            "Comprehensive pest identification and treatment workflow",
            "pest_management",
            "intermediate"
        )
        
        pest_management.steps = [
            WorkflowStep(
                "pest_identification",
                "Identify Pest or Disease",
                "Upload images and describe symptoms to identify the problem",
                tools_required=["pest-identifier"],
                estimated_time=10
            ),
            WorkflowStep(
                "weather_correlation",
                "Check Weather Impact",
                "Analyze how weather conditions affect pest development",
                tools_required=["weather"],
                estimated_time=5,
                prerequisites=["pest_identification"]
            ),
            WorkflowStep(
                "treatment_options",
                "Explore Treatment Options",
                "Research organic and chemical treatment methods",
                tools_required=["search"],
                estimated_time=15,
                prerequisites=["pest_identification"]
            ),
            WorkflowStep(
                "cost_analysis",
                "Analyze Treatment Costs",
                "Compare costs of different treatment approaches",
                tools_required=["mandi-price"],
                estimated_time=10,
                prerequisites=["treatment_options"]
            ),
            WorkflowStep(
                "implementation_plan",
                "Create Implementation Plan",
                "Develop timeline and application schedule for treatment",
                estimated_time=20,
                prerequisites=["weather_correlation", "cost_analysis"]
            )
        ]
        
        # 3. Irrigation Planning Workflow
        irrigation_planning = AgriculturalWorkflow(
            "irrigation_planning",
            "Smart Irrigation Planning",
            "Optimize water usage based on soil, weather, and crop requirements",
            "irrigation",
            "intermediate"
        )
        
        irrigation_planning.steps = [
            WorkflowStep(
                "soil_moisture_check",
                "Check Soil Moisture Levels",
                "Assess current soil moisture and water retention capacity",
                tools_required=["soil-health"],
                estimated_time=10
            ),
            WorkflowStep(
                "weather_forecast",
                "Review Weather Forecast",
                "Check rainfall predictions and temperature patterns",
                tools_required=["weather"],
                estimated_time=5
            ),
            WorkflowStep(
                "crop_water_needs",
                "Determine Crop Water Requirements",
                "Calculate water needs based on crop type and growth stage",
                tools_required=["search"],
                estimated_time=15,
                prerequisites=["soil_moisture_check"]
            ),
            WorkflowStep(
                "irrigation_schedule",
                "Create Irrigation Schedule",
                "Develop optimal watering schedule based on all factors",
                estimated_time=15,
                prerequisites=["weather_forecast", "crop_water_needs"]
            ),
            WorkflowStep(
                "water_conservation",
                "Plan Water Conservation",
                "Implement water-saving techniques and monitoring",
                estimated_time=10,
                prerequisites=["irrigation_schedule"],
                optional=True
            )
        ]
        
        # 4. Harvest Timing Workflow
        harvest_timing = AgriculturalWorkflow(
            "harvest_timing",
            "Optimal Harvest Timing",
            "Determine the best time to harvest for maximum yield and profit",
            "harvest_timing",
            "advanced"
        )
        
        harvest_timing.steps = [
            WorkflowStep(
                "crop_maturity_check",
                "Assess Crop Maturity",
                "Evaluate crop maturity indicators and readiness",
                estimated_time=15
            ),
            WorkflowStep(
                "weather_window",
                "Find Weather Window",
                "Identify optimal weather conditions for harvesting",
                tools_required=["weather"],
                estimated_time=10
            ),
            WorkflowStep(
                "market_timing",
                "Analyze Market Timing",
                "Check current prices and predict optimal selling time",
                tools_required=["crop-price", "mandi-price"],
                estimated_time=20
            ),
            WorkflowStep(
                "logistics_planning",
                "Plan Harvest Logistics",
                "Organize labor, equipment, and transportation",
                estimated_time=25,
                prerequisites=["crop_maturity_check", "weather_window"]
            ),
            WorkflowStep(
                "quality_optimization",
                "Optimize Harvest Quality",
                "Implement best practices for maximum quality and storage life",
                estimated_time=15,
                prerequisites=["logistics_planning"],
                optional=True
            )
        ]
        
        # Store workflows
        self.workflows = {
            "crop_selection": crop_selection,
            "pest_management": pest_management,
            "irrigation_planning": irrigation_planning,
            "harvest_timing": harvest_timing
        }
        
        # Calculate estimated times
        for workflow in self.workflows.values():
            workflow.estimated_total_time = sum(step.estimated_time for step in workflow.steps)
    
    async def start_workflow(self, workflow_id: str, user_id: str, 
                           initial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Start a new workflow instance for a user"""
        
        if workflow_id not in self.workflows:
            return {"success": False, "error": f"Workflow '{workflow_id}' not found"}
        
        # Create workflow instance
        workflow_instance = AgriculturalWorkflow(
            workflow_id,
            self.workflows[workflow_id].title,
            self.workflows[workflow_id].description,
            self.workflows[workflow_id].category,
            self.workflows[workflow_id].difficulty
        )
        
        # Copy steps from template
        workflow_instance.steps = [
            WorkflowStep(
                step.step_id,
                step.title,
                step.description,
                step.tools_required.copy(),
                step.estimated_time,
                step.prerequisites.copy(),
                step.optional
            ) for step in self.workflows[workflow_id].steps
        ]
        
        workflow_instance.user_id = user_id
        workflow_instance.status = WorkflowStatus.IN_PROGRESS
        workflow_instance.started_at = datetime.now(timezone.utc)
        
        # Store instance
        instance_key = f"{user_id}_{workflow_id}_{int(datetime.now().timestamp())}"
        self.user_workflow_instances[instance_key] = workflow_instance
        
        # Save to database
        await self._save_workflow_instance(instance_key, workflow_instance)
        
        return {
            "success": True,
            "instance_id": instance_key,
            "workflow": self._serialize_workflow(workflow_instance),
            "next_step": self._get_next_step(workflow_instance)
        }
    
    async def execute_workflow_step(self, instance_id: str, step_id: str, 
                                  step_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a specific step in a workflow"""
        
        if instance_id not in self.user_workflow_instances:
            return {"success": False, "error": "Workflow instance not found"}
        
        workflow = self.user_workflow_instances[instance_id]
        step = self._find_step(workflow, step_id)
        
        if not step:
            return {"success": False, "error": f"Step '{step_id}' not found"}
        
        # Check prerequisites
        if not self._check_prerequisites(workflow, step):
            missing = [p for p in step.prerequisites if not self._is_step_completed(workflow, p)]
            return {
                "success": False, 
                "error": f"Prerequisites not met: {', '.join(missing)}"
            }
        
        # Mark step as in progress
        step.status = StepStatus.IN_PROGRESS
        step.started_at = datetime.now(timezone.utc)
        
        try:
            # Execute step based on required tools
            step_result = await self._execute_step_tools(step, step_data or {})
            
            # Mark step as completed
            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.now(timezone.utc)
            step.result_data = step_result
            
            # Update workflow progress
            self._update_workflow_progress(workflow)
            
            # Save to database
            await self._save_workflow_instance(instance_id, workflow)
            
            return {
                "success": True,
                "step_result": step_result,
                "workflow_progress": workflow.progress_percentage,
                "next_step": self._get_next_step(workflow),
                "workflow_status": workflow.status.value
            }
            
        except Exception as e:
            step.status = StepStatus.FAILED
            step.notes = str(e)
            logger.error(f"Workflow step execution failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "step_status": step.status.value
            }
    
    async def _execute_step_tools(self, step: WorkflowStep, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tools required for a workflow step"""
        
        results = {}
        
        for tool in step.tools_required:
            try:
                if tool == "soil-health":
                    result = await self.mcp_client.call_tool("soil-health", step_data)
                    results["soil_analysis"] = result
                    
                elif tool == "weather":
                    weather_params = {
                        "location": step_data.get("location", "India"),
                        "days": step_data.get("forecast_days", 7),
                        "include_farming_alerts": True
                    }
                    result = await self.mcp_client.call_tool("weather", weather_params)
                    results["weather_forecast"] = result
                    
                elif tool == "crop-price":
                    price_params = {
                        "state": step_data.get("state"),
                        "commodity": step_data.get("commodity"),
                        "district": step_data.get("district")
                    }
                    result = await self.mcp_client.get_crop_price(**price_params)
                    results["crop_prices"] = result
                    
                elif tool == "mandi-price":
                    mandi_params = {
                        "commodity": step_data.get("commodity"),
                        "state": step_data.get("state"),
                        "district": step_data.get("district"),
                        "include_predictions": True
                    }
                    result = await self.mcp_client.call_tool("mandi-price", mandi_params)
                    results["mandi_analysis"] = result
                    
                elif tool == "pest-identifier":
                    pest_params = {
                        "crop": step_data.get("crop"),
                        "symptoms": step_data.get("symptoms", ""),
                        "location": step_data.get("location")
                    }
                    result = await self.mcp_client.call_tool("pest-identifier", pest_params)
                    results["pest_analysis"] = result
                    
                elif tool == "search":
                    search_query = step_data.get("search_query", f"agricultural advice {step.title}")
                    result = await self.mcp_client.search_web(search_query)
                    results["research_data"] = result
                    
            except Exception as e:
                logger.error(f"Tool execution failed for {tool}: {e}")
                results[f"{tool}_error"] = str(e)
        
        return results
    
    def _find_step(self, workflow: AgriculturalWorkflow, step_id: str) -> Optional[WorkflowStep]:
        """Find a step by ID in the workflow"""
        return next((step for step in workflow.steps if step.step_id == step_id), None)
    
    def _check_prerequisites(self, workflow: AgriculturalWorkflow, step: WorkflowStep) -> bool:
        """Check if all prerequisites for a step are completed"""
        return all(self._is_step_completed(workflow, prereq) for prereq in step.prerequisites)
    
    def _is_step_completed(self, workflow: AgriculturalWorkflow, step_id: str) -> bool:
        """Check if a specific step is completed"""
        step = self._find_step(workflow, step_id)
        return step and step.status == StepStatus.COMPLETED
    
    def _get_next_step(self, workflow: AgriculturalWorkflow) -> Optional[Dict[str, Any]]:
        """Get the next available step in the workflow"""
        for step in workflow.steps:
            if step.status == StepStatus.PENDING and self._check_prerequisites(workflow, step):
                return {
                    "step_id": step.step_id,
                    "title": step.title,
                    "description": step.description,
                    "tools_required": step.tools_required,
                    "estimated_time": step.estimated_time,
                    "optional": step.optional
                }
        return None
    
    def _update_workflow_progress(self, workflow: AgriculturalWorkflow):
        """Update workflow progress percentage"""
        completed_steps = sum(1 for step in workflow.steps if step.status == StepStatus.COMPLETED)
        total_steps = len(workflow.steps)
        workflow.progress_percentage = int((completed_steps / total_steps) * 100)
        
        # Check if workflow is complete
        if completed_steps == total_steps:
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.now(timezone.utc)
    
    def _serialize_workflow(self, workflow: AgriculturalWorkflow) -> Dict[str, Any]:
        """Serialize workflow for API response"""
        return {
            "workflow_id": workflow.workflow_id,
            "title": workflow.title,
            "description": workflow.description,
            "category": workflow.category,
            "difficulty": workflow.difficulty,
            "status": workflow.status.value,
            "progress_percentage": workflow.progress_percentage,
            "estimated_total_time": workflow.estimated_total_time,
            "steps": [
                {
                    "step_id": step.step_id,
                    "title": step.title,
                    "description": step.description,
                    "status": step.status.value,
                    "tools_required": step.tools_required,
                    "estimated_time": step.estimated_time,
                    "optional": step.optional,
                    "prerequisites": step.prerequisites
                } for step in workflow.steps
            ]
        }
    
    async def _save_workflow_instance(self, instance_id: str, workflow: AgriculturalWorkflow):
        """Save workflow instance to database"""
        try:
            workflow_data = {
                "instance_id": instance_id,
                "user_id": workflow.user_id,
                "workflow_data": self._serialize_workflow(workflow),
                "created_at": workflow.created_at,
                "updated_at": datetime.now(timezone.utc)
            }
            
            await self.db.workflow_instances.update_one(
                {"instance_id": instance_id},
                {"$set": workflow_data},
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Failed to save workflow instance: {e}")
    
    async def get_user_workflows(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all workflow instances for a user"""
        try:
            instances = await self.db.workflow_instances.find(
                {"user_id": user_id}
            ).sort("created_at", -1).to_list(50)
            
            return [instance["workflow_data"] for instance in instances]
            
        except Exception as e:
            logger.error(f"Failed to get user workflows: {e}")
            return []
    
    def get_available_workflows(self) -> List[Dict[str, Any]]:
        """Get list of all available workflow templates"""
        return [
            {
                "workflow_id": workflow.workflow_id,
                "title": workflow.title,
                "description": workflow.description,
                "category": workflow.category,
                "difficulty": workflow.difficulty,
                "estimated_time": workflow.estimated_total_time,
                "step_count": len(workflow.steps)
            } for workflow in self.workflows.values()
        ]