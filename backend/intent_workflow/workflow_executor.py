"""
Workflow Executor - Executes multi-step workflows based on classified intents
Handles task sequencing and error recovery
"""

import logging
from typing import Dict, Any, Callable, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WorkflowTask:
    """Represents a single task in a workflow"""
    name: str
    function: Callable
    required_params: List[str]
    on_error: str = "stop"  # "stop", "continue", or "fallback"


class WorkflowExecutor:
    """Executes workflows sequentially with error handling"""
    
    def __init__(self):
        """Initialize workflow executor"""
        self.workflows = {}
        self.task_results = {}
    
    def register_workflow(self, intent: str, tasks: List[WorkflowTask]):
        """Register a workflow for a specific intent"""
        self.workflows[intent] = tasks
        logger.info(f"Registered workflow for intent: {intent}")
    
    def execute(self, intent: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute workflow for the given intent
        
        Args:
            intent: The classified intent
            parameters: Extracted parameters from user query
            
        Returns:
            Dict with execution results, errors, and metadata
        """
        
        if intent not in self.workflows:
            return {
                "success": False,
                "error": f"No workflow registered for intent: {intent}",
                "results": {}
            }
        
        tasks = self.workflows[intent]
        self.task_results = {}
        
        logger.info(f"Starting workflow execution for intent: {intent}")
        logger.info(f"Tasks to execute: {[t.name for t in tasks]}")
        
        execution_result = {
            "intent": intent,
            "success": True,
            "tasks_executed": [],
            "tasks_failed": [],
            "results": {},
            "errors": {}
        }
        
        for task in tasks:
            try:
                # Prepare parameters for this task
                task_params = self._prepare_task_params(task, parameters)
                
                logger.info(f"Executing task: {task.name}")
                logger.debug(f"Task parameters: {task_params}")
                
                # Execute the task
                result = task.function(**task_params)
                
                self.task_results[task.name] = result
                execution_result["results"][task.name] = result
                execution_result["tasks_executed"].append(task.name)
                
                logger.info(f"Task '{task.name}' completed successfully")
                
                # Update parameters with results for next tasks
                if isinstance(result, dict):
                    parameters.update(result)
                
            except Exception as e:
                logger.error(f"Error executing task '{task.name}': {str(e)}")
                execution_result["errors"][task.name] = str(e)
                execution_result["tasks_failed"].append(task.name)
                execution_result["success"] = False
                
                if task.on_error == "stop":
                    logger.info(f"Stopping workflow due to error in '{task.name}'")
                    break
                elif task.on_error == "continue":
                    logger.info(f"Continuing workflow despite error in '{task.name}'")
                    continue
        
        if execution_result["tasks_failed"]:
            execution_result["success"] = False
        
        logger.info(f"Workflow execution completed. Success: {execution_result['success']}")
        return execution_result
    
    def _prepare_task_params(self, task: WorkflowTask, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare parameters for a task, extracting from available parameters
        and previous task results.
        
        Passes ALL available parameters (not just required ones) so that
        optional params like use_llm, user_prompt, subject, body, meeting_link
        are also forwarded to task functions.
        """
        # Start with ALL current parameters
        task_params = dict(parameters)
        
        # Also merge in any results from previous tasks
        for task_name, results in self.task_results.items():
            if isinstance(results, dict):
                for key, value in results.items():
                    if key not in task_params:
                        task_params[key] = value
        
        # Warn if any required params are still missing
        for param in task.required_params:
            if param not in task_params:
                logger.warning(f"Required parameter '{param}' not found for task '{task.name}'")
        
        return task_params


class WorkflowBuilder:
    """Helper to build workflows declaratively"""
    
    @staticmethod
    def build_workflows() -> Dict[str, List[WorkflowTask]]:
        """
        Define all workflows - this is where you specify the order of tasks
        for each intent
        """
        workflows = {}
        
        # These will be registered by the main backend
        # This method returns the structure, actual tasks are injected later
        
        return workflows
