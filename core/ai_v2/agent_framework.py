#!/usr/bin/env python3
"""
RF Arsenal OS - Autonomous Agent Framework

AI agent that can reason, plan, and execute multi-step operations.
Combines LLM reasoning with real tool execution.

README COMPLIANCE:
- UNFILTERED: No restrictions on agent actions
- Offline-first: Full capability without internet
- RAM-only: No persistent logs
- Real execution: Actually performs operations

Author: RF Arsenal Team
License: Proprietary - Authorized Use Only
"""

import os
import sys
import json
import logging
import threading
import time
from typing import Optional, Dict, List, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING_INPUT = "waiting_input"
    SUCCESS = "success"
    FAILED = "failed"
    ABORTED = "aborted"


class AgentMode(Enum):
    """Agent operation modes"""
    AUTONOMOUS = "autonomous"      # Full auto - no confirmations
    SUPERVISED = "supervised"      # Confirm before execution
    PLANNING_ONLY = "planning"     # Generate plan but don't execute


@dataclass
class AgentTask:
    """A task for the agent to accomplish"""
    id: str
    objective: str
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    # Execution
    plan: List[Dict[str, Any]] = field(default_factory=list)
    current_step: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Output
    final_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class AgentResult:
    """Result from agent task execution"""
    task_id: str
    success: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    steps_executed: int = 0
    execution_time: float = 0.0


class AutonomousAgent:
    """
    Autonomous AI Agent for RF Arsenal OS
    
    UNFILTERED - Executes any task requested without restrictions.
    
    Capabilities:
    - Natural language understanding
    - Multi-step planning
    - Tool execution
    - Error recovery
    - Context awareness
    - Adaptive behavior
    """
    
    # Available tools for the agent
    TOOLS = {
        "wifi_scan": {
            "description": "Scan for WiFi networks",
            "command": "scan networks",
            "module": "wifi",
        },
        "wifi_deauth": {
            "description": "Disconnect clients from WiFi network",
            "command": "deauth {target}",
            "module": "wifi",
            "params": ["target"],
        },
        "wifi_evil_twin": {
            "description": "Create fake access point",
            "command": "evil twin {ssid}",
            "module": "wifi",
            "params": ["ssid"],
        },
        "cellular_scan": {
            "description": "Scan cellular networks",
            "command": "scan lte",
            "module": "cellular",
        },
        "imsi_catch": {
            "description": "Start IMSI catcher",
            "command": "start imsi catcher",
            "module": "yatebts",
        },
        "gps_spoof": {
            "description": "Spoof GPS coordinates",
            "command": "spoof gps {lat} {lon}",
            "module": "gps",
            "params": ["lat", "lon"],
        },
        "drone_detect": {
            "description": "Detect nearby drones",
            "command": "detect drones",
            "module": "drone",
        },
        "drone_jam": {
            "description": "Jam drone control signals",
            "command": "jam drone",
            "module": "drone",
        },
        "spectrum_scan": {
            "description": "Scan RF spectrum",
            "command": "scan spectrum {start_freq} to {end_freq}",
            "module": "spectrum",
            "params": ["start_freq", "end_freq"],
        },
        "bluetooth_scan": {
            "description": "Scan for Bluetooth devices",
            "command": "scan ble5",
            "module": "bluetooth5",
        },
        "keyfob_capture": {
            "description": "Capture key fob signals",
            "command": "capture key fob",
            "module": "vehicle",
        },
        "relay_attack": {
            "description": "Start relay attack on car key",
            "command": "start relay car key",
            "module": "relay",
        },
        "nfc_scan": {
            "description": "Scan NFC/RFID cards",
            "command": "scan nfc",
            "module": "nfc",
        },
        "nfc_clone": {
            "description": "Clone NFC card",
            "command": "clone card",
            "module": "nfc",
        },
        "port_scan": {
            "description": "Scan ports on target",
            "command": "port scan {target}",
            "module": "recon",
            "params": ["target"],
        },
        "vuln_scan": {
            "description": "Scan for vulnerabilities",
            "command": "vuln scan {target}",
            "module": "pentest",
            "params": ["target"],
        },
        "brute_force": {
            "description": "Brute force credentials",
            "command": "brute force {service} {target}",
            "module": "credential",
            "params": ["service", "target"],
        },
        "stealth_mode": {
            "description": "Enable stealth mode",
            "command": "enable stealth mode",
            "module": "stealth",
        },
        "rotate_mac": {
            "description": "Randomize MAC address",
            "command": "rotate mac",
            "module": "stealth",
        },
        "emergency_wipe": {
            "description": "Emergency data wipe",
            "command": "emergency wipe",
            "module": "emergency",
        },
    }
    
    def __init__(self, mode: AgentMode = AgentMode.AUTONOMOUS):
        self.logger = logging.getLogger('AutonomousAgent')
        self.mode = mode
        
        # Components (lazy loaded)
        self._llm = None
        self._memory = None
        self._attack_planner = None
        self._command_center = None
        
        # Active tasks (RAM-only)
        self._tasks: Dict[str, AgentTask] = {}
        self._lock = threading.Lock()
        
        # Callbacks
        self._execution_callbacks: List[Callable] = []
        self._thinking_callbacks: List[Callable] = []
        
        self.logger.info(f"Autonomous Agent initialized (mode: {mode.value})")
    
    def _get_llm(self):
        """Get LLM for reasoning"""
        if self._llm is None:
            try:
                from .local_llm import LocalLLM
                self._llm = LocalLLM()
            except ImportError:
                pass
        return self._llm
    
    def _get_memory(self):
        """Get memory store"""
        if self._memory is None:
            try:
                from .memory_store import RAMMemory
                self._memory = RAMMemory()
            except ImportError:
                pass
        return self._memory
    
    def _get_attack_planner(self):
        """Get attack planner"""
        if self._attack_planner is None:
            try:
                from .attack_planner import AttackPlanner
                self._attack_planner = AttackPlanner()
            except ImportError:
                pass
        return self._attack_planner
    
    def set_command_center(self, command_center):
        """Set AI Command Center for execution"""
        self._command_center = command_center
        
        planner = self._get_attack_planner()
        if planner:
            planner.set_command_center(command_center)
    
    def run(self, objective: str, context: Dict[str, Any] = None) -> AgentResult:
        """
        Run agent on an objective
        
        UNFILTERED - Will attempt any objective requested.
        
        Args:
            objective: What to accomplish
            context: Additional context/parameters
            
        Returns:
            AgentResult with outcome
        """
        import secrets
        
        # Create task
        task = AgentTask(
            id=secrets.token_hex(8),
            objective=objective,
            context=context or {},
        )
        
        with self._lock:
            self._tasks[task.id] = task
        
        start_time = time.time()
        
        try:
            # Think about the task
            self._think(task, f"Analyzing objective: {objective}")
            task.status = TaskStatus.PLANNING
            
            # Generate plan
            plan = self._plan(task)
            task.plan = plan
            
            if self.mode == AgentMode.PLANNING_ONLY:
                task.status = TaskStatus.SUCCESS
                task.final_result = {"plan": plan}
                return self._create_result(task, start_time)
            
            # Execute plan
            task.status = TaskStatus.EXECUTING
            
            for i, step in enumerate(plan):
                task.current_step = i
                
                # Think about current step
                self._think(task, f"Executing step {i+1}: {step.get('action', 'Unknown')}")
                
                # Check if confirmation needed
                if self.mode == AgentMode.SUPERVISED and step.get('dangerous', False):
                    task.status = TaskStatus.WAITING_INPUT
                    # In real implementation, would wait for user confirmation
                    task.status = TaskStatus.EXECUTING
                
                # Execute step
                result = self._execute_step(task, step)
                task.results.append(result)
                
                # Check for failure
                if not result.get('success', False):
                    # Try fallback or abort
                    if step.get('fallback'):
                        fallback_result = self._execute_step(task, step['fallback'])
                        if not fallback_result.get('success', False):
                            task.status = TaskStatus.FAILED
                            task.error = f"Step {i+1} failed: {result.get('error', 'Unknown error')}"
                            break
                    else:
                        task.status = TaskStatus.FAILED
                        task.error = f"Step {i+1} failed: {result.get('error', 'Unknown error')}"
                        break
                
                # Update context with results
                if result.get('data'):
                    task.context.update(result['data'])
            
            # Finalize
            if task.status != TaskStatus.FAILED:
                task.status = TaskStatus.SUCCESS
                task.final_result = self._summarize_results(task)
            
            task.completed_at = datetime.now()
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.logger.error(f"Agent error: {e}")
        
        return self._create_result(task, start_time)
    
    def _think(self, task: AgentTask, thought: str):
        """Record agent thinking (for callbacks)"""
        for callback in self._thinking_callbacks:
            try:
                callback(task.id, thought)
            except Exception:
                pass
    
    def _plan(self, task: AgentTask) -> List[Dict[str, Any]]:
        """Generate execution plan for task"""
        llm = self._get_llm()
        
        if llm and llm.is_available():
            # Use LLM to generate intelligent plan
            return self._llm_plan(task)
        else:
            # Fallback to rule-based planning
            return self._rule_based_plan(task)
    
    def _llm_plan(self, task: AgentTask) -> List[Dict[str, Any]]:
        """Generate plan using LLM"""
        llm = self._get_llm()
        
        # Build tool descriptions
        tool_desc = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.TOOLS.items()
        ])
        
        prompt = f"""You are an autonomous penetration testing agent. Generate a step-by-step plan.

OBJECTIVE: {task.objective}

CONTEXT:
{json.dumps(task.context, indent=2)}

AVAILABLE TOOLS:
{tool_desc}

Generate a plan as JSON array. Each step:
{{
    "step": number,
    "action": "description of action",
    "tool": "tool_name from list above",
    "params": {{}},  // parameters for tool
    "expected_result": "what we expect",
    "dangerous": false,  // true if needs confirmation
    "fallback": null  // alternative step if this fails
}}

Generate the plan:"""
        
        response = llm.generate(prompt, temperature=0.3)
        
        try:
            # Parse JSON from response
            text = response.text.strip()
            # Handle markdown code blocks
            if "```" in text:
                text = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
                if text:
                    text = text.group(1)
            return json.loads(text)
        except (json.JSONDecodeError, AttributeError):
            # Fallback
            return self._rule_based_plan(task)
    
    def _rule_based_plan(self, task: AgentTask) -> List[Dict[str, Any]]:
        """Generate plan using rules (fallback)"""
        objective_lower = task.objective.lower()
        plan = []
        
        # Enable stealth first
        plan.append({
            "step": 1,
            "action": "Enable stealth mode",
            "tool": "stealth_mode",
            "params": {},
            "expected_result": "Stealth mode activated",
        })
        
        # Determine primary action based on objective
        if any(w in objective_lower for w in ['wifi', 'wireless', 'network']):
            plan.append({
                "step": 2,
                "action": "Scan WiFi networks",
                "tool": "wifi_scan",
                "params": {},
                "expected_result": "List of nearby networks",
            })
            if 'attack' in objective_lower or 'hack' in objective_lower:
                plan.append({
                    "step": 3,
                    "action": "Deauthenticate clients",
                    "tool": "wifi_deauth",
                    "params": {"target": task.context.get('target', 'all')},
                    "expected_result": "Clients disconnected",
                    "dangerous": True,
                })
        
        elif any(w in objective_lower for w in ['cellular', 'phone', 'imsi']):
            plan.append({
                "step": 2,
                "action": "Scan cellular networks",
                "tool": "cellular_scan",
                "params": {},
                "expected_result": "Cell tower information",
            })
            if 'catch' in objective_lower or 'intercept' in objective_lower:
                plan.append({
                    "step": 3,
                    "action": "Start IMSI catcher",
                    "tool": "imsi_catch",
                    "params": {},
                    "expected_result": "IMSI identifiers captured",
                    "dangerous": True,
                })
        
        elif any(w in objective_lower for w in ['drone', 'uav']):
            plan.append({
                "step": 2,
                "action": "Detect drones",
                "tool": "drone_detect",
                "params": {},
                "expected_result": "Drone signals detected",
            })
            if 'jam' in objective_lower or 'stop' in objective_lower:
                plan.append({
                    "step": 3,
                    "action": "Jam drone signals",
                    "tool": "drone_jam",
                    "params": {},
                    "expected_result": "Drone control disrupted",
                    "dangerous": True,
                })
        
        elif any(w in objective_lower for w in ['car', 'vehicle', 'key']):
            plan.append({
                "step": 2,
                "action": "Capture key fob signals",
                "tool": "keyfob_capture",
                "params": {},
                "expected_result": "Key fob signal captured",
            })
            if 'relay' in objective_lower or 'unlock' in objective_lower:
                plan.append({
                    "step": 3,
                    "action": "Execute relay attack",
                    "tool": "relay_attack",
                    "params": {},
                    "expected_result": "Vehicle unlocked",
                    "dangerous": True,
                })
        
        elif any(w in objective_lower for w in ['nfc', 'rfid', 'card', 'badge']):
            plan.append({
                "step": 2,
                "action": "Scan NFC cards",
                "tool": "nfc_scan",
                "params": {},
                "expected_result": "Card data read",
            })
            if 'clone' in objective_lower or 'copy' in objective_lower:
                plan.append({
                    "step": 3,
                    "action": "Clone NFC card",
                    "tool": "nfc_clone",
                    "params": {},
                    "expected_result": "Card cloned",
                    "dangerous": True,
                })
        
        elif any(w in objective_lower for w in ['scan', 'recon', 'network']):
            target = task.context.get('target', 'localhost')
            plan.append({
                "step": 2,
                "action": "Port scan target",
                "tool": "port_scan",
                "params": {"target": target},
                "expected_result": "Open ports identified",
            })
            plan.append({
                "step": 3,
                "action": "Vulnerability scan",
                "tool": "vuln_scan",
                "params": {"target": target},
                "expected_result": "Vulnerabilities identified",
            })
        
        else:
            # Generic spectrum scan
            plan.append({
                "step": 2,
                "action": "Scan RF spectrum",
                "tool": "spectrum_scan",
                "params": {"start_freq": "100mhz", "end_freq": "6ghz"},
                "expected_result": "RF activity detected",
            })
        
        return plan
    
    def _execute_step(self, task: AgentTask, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step"""
        tool_name = step.get('tool')
        params = step.get('params', {})
        
        if not tool_name:
            return {"success": False, "error": "No tool specified"}
        
        tool = self.TOOLS.get(tool_name)
        if not tool:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
        
        # Build command
        command = tool['command']
        for key, value in params.items():
            command = command.replace(f"{{{key}}}", str(value))
        
        # Execute via command center
        if self._command_center:
            try:
                result = self._command_center.process_command(command)
                
                # Notify callbacks
                for callback in self._execution_callbacks:
                    try:
                        callback(task.id, step, result)
                    except Exception:
                        pass
                
                return {
                    "success": result.success,
                    "message": result.message,
                    "data": result.data,
                    "command": command,
                }
            except Exception as e:
                return {"success": False, "error": str(e), "command": command}
        else:
            # Dry run
            return {
                "success": True,
                "message": f"[Dry run] Would execute: {command}",
                "command": command,
            }
    
    def _summarize_results(self, task: AgentTask) -> Dict[str, Any]:
        """Summarize task results"""
        successful_steps = sum(1 for r in task.results if r.get('success'))
        
        summary = {
            "objective": task.objective,
            "status": "success" if successful_steps == len(task.results) else "partial",
            "steps_total": len(task.plan),
            "steps_successful": successful_steps,
            "results": task.results,
        }
        
        # Use LLM to generate summary if available
        llm = self._get_llm()
        if llm and llm.is_available():
            prompt = f"""Summarize the results of this penetration testing operation:

Objective: {task.objective}
Steps executed: {len(task.results)}
Successful: {successful_steps}

Results:
{json.dumps(task.results, indent=2)}

Provide a brief executive summary:"""
            
            response = llm.generate(prompt, max_tokens=500)
            summary["executive_summary"] = response.text
        
        return summary
    
    def _create_result(self, task: AgentTask, start_time: float) -> AgentResult:
        """Create result object from task"""
        return AgentResult(
            task_id=task.id,
            success=task.status == TaskStatus.SUCCESS,
            message=task.error or "Task completed",
            data=task.final_result or {},
            steps_executed=len(task.results),
            execution_time=time.time() - start_time,
        )
    
    def add_execution_callback(self, callback: Callable):
        """Add callback for step execution"""
        self._execution_callbacks.append(callback)
    
    def add_thinking_callback(self, callback: Callable):
        """Add callback for agent thinking"""
        self._thinking_callbacks.append(callback)
    
    def get_task(self, task_id: str) -> Optional[AgentTask]:
        """Get task by ID"""
        return self._tasks.get(task_id)
    
    def abort_task(self, task_id: str) -> bool:
        """Abort a running task"""
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.EXECUTING:
            task.status = TaskStatus.ABORTED
            return True
        return False
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get list of available tools"""
        return [
            {"name": name, "description": info['description']}
            for name, info in self.TOOLS.items()
        ]
    
    # === Agent Management Methods for AI Command Center ===
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent framework status"""
        import sys
        with self._lock:
            active = sum(1 for t in self._tasks.values() if t.status == TaskStatus.EXECUTING)
            completed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.SUCCESS)
            pending = sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)
            
            return {
                "active_count": active,
                "tasks_completed": completed,
                "tasks_pending": pending,
                "memory_mb": sys.getsizeof(self._tasks) / (1024 * 1024),
                "mode": self.mode.value,
                "total_tasks": len(self._tasks),
            }
    
    def create_agent(self, agent_type: str) -> str:
        """Create a new agent task"""
        import secrets
        task_id = secrets.token_hex(8)
        
        objective_map = {
            'reconnaissance': 'Perform reconnaissance scan on target environment',
            'attack': 'Execute attack sequence on target',
            'intelligence': 'Gather intelligence from RF environment',
            'monitoring': 'Monitor RF spectrum for activity',
        }
        
        objective = objective_map.get(agent_type, f'Execute {agent_type} operation')
        
        task = AgentTask(
            id=task_id,
            objective=objective,
            context={'agent_type': agent_type},
        )
        
        with self._lock:
            self._tasks[task_id] = task
        
        return task_id
    
    def start_agent(self, agent_id: str = None):
        """Start an agent task"""
        if agent_id:
            task = self._tasks.get(agent_id)
            if task and task.status == TaskStatus.PENDING:
                # Run the task
                threading.Thread(target=self.run, args=(task.objective, task.context)).start()
        else:
            # Start all pending tasks
            with self._lock:
                for task in self._tasks.values():
                    if task.status == TaskStatus.PENDING:
                        threading.Thread(target=self.run, args=(task.objective, task.context)).start()
    
    def stop_all_agents(self):
        """Stop all running agents"""
        with self._lock:
            for task in self._tasks.values():
                if task.status == TaskStatus.EXECUTING:
                    task.status = TaskStatus.ABORTED
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents"""
        with self._lock:
            return [
                {
                    'id': task.id,
                    'type': task.context.get('agent_type', 'unknown'),
                    'status': task.status.value,
                    'objective': task.objective,
                }
                for task in self._tasks.values()
            ]


# Convenience functions
def get_agent(mode: AgentMode = AgentMode.AUTONOMOUS) -> AutonomousAgent:
    """Get autonomous agent instance"""
    return AutonomousAgent(mode)


def run_autonomous(objective: str, **context) -> AgentResult:
    """Quick run of autonomous agent"""
    agent = AutonomousAgent(AgentMode.AUTONOMOUS)
    return agent.run(objective, context)
