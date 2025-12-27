"""
RF Arsenal OS - Mission Scripting Engine
YAML-based mission scripting for automated RF operations.
"""

import yaml
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import re


class StepType(Enum):
    """Mission step types"""
    COMMAND = "command"
    SCAN = "scan"
    CAPTURE = "capture"
    TRANSMIT = "transmit"
    WAIT = "wait"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    ALERT = "alert"
    LOG = "log"
    HARDWARE = "hardware"
    ANALYZE = "analyze"
    EXPORT = "export"


class StepStatus(Enum):
    """Step execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class MissionStep:
    """Single mission step"""
    step_id: str
    step_type: StepType
    name: str
    parameters: Dict[str, Any]
    timeout_s: float = 60.0
    retry_count: int = 0
    continue_on_error: bool = False
    condition: Optional[str] = None  # Python expression
    status: StepStatus = StepStatus.PENDING
    result: Optional[Dict] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "name": self.name,
            "parameters": self.parameters,
            "timeout_s": self.timeout_s,
            "retry_count": self.retry_count,
            "continue_on_error": self.continue_on_error,
            "condition": self.condition,
            "status": self.status.value,
            "result": self.result,
            "error": self.error
        }


@dataclass
class MissionScript:
    """Complete mission script"""
    mission_id: str
    name: str
    description: str = ""
    author: str = ""
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    steps: List[MissionStep] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    
    @classmethod
    def from_yaml(cls, yaml_content: str) -> 'MissionScript':
        """Parse mission script from YAML"""
        data = yaml.safe_load(yaml_content)
        
        mission = cls(
            mission_id=data.get("mission_id", f"mission_{int(time.time())}"),
            name=data.get("name", "Unnamed Mission"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            version=data.get("version", "1.0"),
            tags=data.get("tags", []),
            variables=data.get("variables", {}),
            requirements=data.get("requirements", [])
        )
        
        # Parse steps
        for i, step_data in enumerate(data.get("steps", [])):
            step = MissionStep(
                step_id=step_data.get("id", f"step_{i}"),
                step_type=StepType(step_data.get("type", "command")),
                name=step_data.get("name", f"Step {i}"),
                parameters=step_data.get("parameters", {}),
                timeout_s=step_data.get("timeout", 60.0),
                retry_count=step_data.get("retry", 0),
                continue_on_error=step_data.get("continue_on_error", False),
                condition=step_data.get("condition")
            )
            mission.steps.append(step)
            
        return mission
        
    @classmethod
    def from_file(cls, filepath: str) -> 'MissionScript':
        """Load mission script from file"""
        with open(filepath, 'r') as f:
            return cls.from_yaml(f.read())
            
    def to_yaml(self) -> str:
        """Export mission script to YAML"""
        data = {
            "mission_id": self.mission_id,
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "version": self.version,
            "tags": self.tags,
            "variables": self.variables,
            "requirements": self.requirements,
            "steps": [
                {
                    "id": s.step_id,
                    "type": s.step_type.value,
                    "name": s.name,
                    "parameters": s.parameters,
                    "timeout": s.timeout_s,
                    "retry": s.retry_count,
                    "continue_on_error": s.continue_on_error,
                    "condition": s.condition
                }
                for s in self.steps
            ]
        }
        return yaml.dump(data, default_flow_style=False)


class MissionEngine:
    """
    Mission execution engine for running scripted RF operations.
    
    Features:
    - YAML mission script execution
    - Variable substitution
    - Conditional execution
    - Loops and parallel execution
    - Error handling with retry
    - Real-time progress tracking
    - Pause/resume support
    - Safe abort with cleanup
    """
    
    def __init__(self, command_handler: Optional[Callable] = None):
        """
        Initialize mission engine.
        
        Args:
            command_handler: Function to execute commands (receives command string, returns result)
        """
        self.command_handler = command_handler or (lambda cmd: {"success": True})
        
        # Current mission state
        self._current_mission: Optional[MissionScript] = None
        self._running = False
        self._paused = False
        self._abort_requested = False
        
        # Variables (mission + runtime)
        self._variables: Dict[str, Any] = {}
        
        # Results
        self._step_results: Dict[str, Dict] = {}
        
        # Progress tracking
        self._current_step_index = 0
        self._progress_callbacks: List[Callable] = []
        
        # Thread
        self._execution_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
    def load_mission(self, mission: MissionScript) -> bool:
        """
        Load a mission script for execution.
        
        Args:
            mission: Mission script to load
            
        Returns:
            Success status
        """
        if self._running:
            return False
            
        with self._lock:
            self._current_mission = mission
            self._variables = mission.variables.copy()
            self._step_results = {}
            self._current_step_index = 0
            
            # Reset step statuses
            for step in mission.steps:
                step.status = StepStatus.PENDING
                step.result = None
                step.error = None
                
        return True
        
    def start(self) -> bool:
        """Start mission execution"""
        if self._running or not self._current_mission:
            return False
            
        self._running = True
        self._paused = False
        self._abort_requested = False
        
        self._execution_thread = threading.Thread(
            target=self._execute_mission,
            daemon=True
        )
        self._execution_thread.start()
        
        return True
        
    def pause(self) -> None:
        """Pause mission execution"""
        self._paused = True
        
    def resume(self) -> None:
        """Resume paused mission"""
        self._paused = False
        
    def abort(self) -> None:
        """Abort mission with cleanup"""
        self._abort_requested = True
        self._running = False
        
    def _execute_mission(self) -> None:
        """Main mission execution loop"""
        if not self._current_mission:
            return
            
        for i, step in enumerate(self._current_mission.steps):
            if self._abort_requested:
                step.status = StepStatus.SKIPPED
                continue
                
            # Wait if paused
            while self._paused and not self._abort_requested:
                time.sleep(0.1)
                
            if self._abort_requested:
                step.status = StepStatus.SKIPPED
                continue
                
            self._current_step_index = i
            
            # Check condition
            if step.condition:
                if not self._evaluate_condition(step.condition):
                    step.status = StepStatus.SKIPPED
                    self._notify_progress()
                    continue
                    
            # Execute step with retry
            success = False
            attempt = 0
            
            while attempt <= step.retry_count and not success:
                attempt += 1
                step.status = StepStatus.RUNNING
                step.start_time = time.time()
                self._notify_progress()
                
                try:
                    result = self._execute_step(step)
                    step.result = result
                    step.status = StepStatus.COMPLETED
                    success = True
                except Exception as e:
                    step.error = str(e)
                    if attempt > step.retry_count:
                        step.status = StepStatus.FAILED
                        
                step.end_time = time.time()
                
            # Store result
            self._step_results[step.step_id] = {
                "status": step.status.value,
                "result": step.result,
                "error": step.error,
                "duration": step.end_time - step.start_time if step.end_time and step.start_time else 0
            }
            
            self._notify_progress()
            
            # Check if we should continue
            if step.status == StepStatus.FAILED and not step.continue_on_error:
                break
                
        self._running = False
        
    def _execute_step(self, step: MissionStep) -> Dict:
        """Execute a single step"""
        # Substitute variables in parameters
        params = self._substitute_variables(step.parameters)
        
        if step.step_type == StepType.COMMAND:
            command = params.get("command", "")
            return self._execute_command(command)
            
        elif step.step_type == StepType.SCAN:
            return self._execute_scan(params)
            
        elif step.step_type == StepType.CAPTURE:
            return self._execute_capture(params)
            
        elif step.step_type == StepType.WAIT:
            duration = params.get("duration", 1.0)
            time.sleep(duration)
            return {"waited": duration}
            
        elif step.step_type == StepType.CONDITION:
            return self._execute_conditional(params)
            
        elif step.step_type == StepType.LOOP:
            return self._execute_loop(params)
            
        elif step.step_type == StepType.PARALLEL:
            return self._execute_parallel(params)
            
        elif step.step_type == StepType.ALERT:
            return self._execute_alert(params)
            
        elif step.step_type == StepType.LOG:
            message = params.get("message", "")
            print(f"[MISSION LOG] {message}")
            return {"logged": message}
            
        elif step.step_type == StepType.HARDWARE:
            return self._execute_hardware(params)
            
        elif step.step_type == StepType.ANALYZE:
            return self._execute_analyze(params)
            
        elif step.step_type == StepType.EXPORT:
            return self._execute_export(params)
            
        else:
            return {"error": f"Unknown step type: {step.step_type}"}
            
    def _execute_command(self, command: str) -> Dict:
        """Execute a command through the handler"""
        result = self.command_handler(command)
        
        # Update variables if command returns values
        if isinstance(result, dict):
            for key, value in result.items():
                self._variables[f"last_{key}"] = value
                
        return result
        
    def _execute_scan(self, params: Dict) -> Dict:
        """Execute RF scan"""
        freq_start = params.get("freq_start", 100e6)
        freq_end = params.get("freq_end", 1000e6)
        step = params.get("step", 1e6)
        dwell_time = params.get("dwell_time", 0.1)
        
        command = f"scan spectrum {freq_start/1e6:.3f}MHz to {freq_end/1e6:.3f}MHz step {step/1e6:.3f}MHz dwell {dwell_time}s"
        return self._execute_command(command)
        
    def _execute_capture(self, params: Dict) -> Dict:
        """Execute signal capture"""
        frequency = params.get("frequency", 100e6)
        duration = params.get("duration", 1.0)
        sample_rate = params.get("sample_rate", 1e6)
        
        command = f"capture iq at {frequency/1e6:.3f}MHz for {duration}s at {sample_rate/1e6:.1f}MSPS"
        return self._execute_command(command)
        
    def _execute_conditional(self, params: Dict) -> Dict:
        """Execute conditional logic"""
        condition = params.get("condition", "True")
        then_command = params.get("then", "")
        else_command = params.get("else", "")
        
        if self._evaluate_condition(condition):
            if then_command:
                return self._execute_command(then_command)
        else:
            if else_command:
                return self._execute_command(else_command)
                
        return {"condition": condition, "result": self._evaluate_condition(condition)}
        
    def _execute_loop(self, params: Dict) -> Dict:
        """Execute loop"""
        count = params.get("count", 1)
        commands = params.get("commands", [])
        delay = params.get("delay", 0)
        
        results = []
        for i in range(count):
            self._variables["loop_index"] = i
            
            for cmd in commands:
                result = self._execute_command(cmd)
                results.append(result)
                
            if delay > 0:
                time.sleep(delay)
                
        return {"iterations": count, "results": results}
        
    def _execute_parallel(self, params: Dict) -> Dict:
        """Execute commands in parallel"""
        commands = params.get("commands", [])
        
        threads = []
        results = [None] * len(commands)
        
        def run_command(index, cmd):
            results[index] = self._execute_command(cmd)
            
        for i, cmd in enumerate(commands):
            t = threading.Thread(target=run_command, args=(i, cmd))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        return {"parallel_results": results}
        
    def _execute_alert(self, params: Dict) -> Dict:
        """Generate alert"""
        message = params.get("message", "Alert")
        level = params.get("level", "info")
        
        # In production, this would trigger actual alerts
        print(f"[ALERT - {level.upper()}] {message}")
        
        return {"alert": message, "level": level}
        
    def _execute_hardware(self, params: Dict) -> Dict:
        """Execute hardware action"""
        action = params.get("action", "")
        device = params.get("device", "")
        
        command = f"hardware {device} {action}"
        return self._execute_command(command)
        
    def _execute_analyze(self, params: Dict) -> Dict:
        """Execute analysis"""
        analysis_type = params.get("type", "spectrum")
        target = params.get("target", "")
        
        command = f"analyze {analysis_type} {target}"
        return self._execute_command(command)
        
    def _execute_export(self, params: Dict) -> Dict:
        """Export data"""
        format_type = params.get("format", "json")
        destination = params.get("destination", "")
        
        command = f"export {format_type} to {destination}"
        return self._execute_command(command)
        
    def _substitute_variables(self, params: Dict) -> Dict:
        """Substitute variables in parameters"""
        result = {}
        
        for key, value in params.items():
            if isinstance(value, str):
                # Replace ${var} patterns
                for var_name, var_value in self._variables.items():
                    value = value.replace(f"${{{var_name}}}", str(var_value))
                result[key] = value
            elif isinstance(value, dict):
                result[key] = self._substitute_variables(value)
            elif isinstance(value, list):
                result[key] = [
                    self._substitute_variables({"v": v})["v"] if isinstance(v, (str, dict)) else v
                    for v in value
                ]
            else:
                result[key] = value
                
        return result
        
    def _evaluate_condition(self, condition: str) -> bool:
        """Safely evaluate condition expression"""
        # Replace variable references
        for var_name, var_value in self._variables.items():
            condition = condition.replace(f"${{{var_name}}}", repr(var_value))
            
        # Safe evaluation (limited functions)
        safe_dict = {
            "True": True,
            "False": False,
            "None": None,
            "len": len,
            "int": int,
            "float": float,
            "str": str,
            "abs": abs,
            "min": min,
            "max": max
        }
        
        try:
            return bool(eval(condition, {"__builtins__": {}}, safe_dict))
        except Exception:
            return False
            
    def _notify_progress(self) -> None:
        """Notify progress callbacks"""
        status = self.get_status()
        for callback in self._progress_callbacks:
            try:
                callback(status)
            except Exception:
                pass
                
    def set_variable(self, name: str, value: Any) -> None:
        """Set runtime variable"""
        self._variables[name] = value
        
    def get_variable(self, name: str) -> Any:
        """Get variable value"""
        return self._variables.get(name)
        
    def register_progress_callback(self, callback: Callable) -> None:
        """Register progress callback"""
        self._progress_callbacks.append(callback)
        
    def get_status(self) -> Dict[str, Any]:
        """Get current mission status"""
        if not self._current_mission:
            return {"status": "no_mission_loaded"}
            
        completed = sum(1 for s in self._current_mission.steps if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in self._current_mission.steps if s.status == StepStatus.FAILED)
        
        return {
            "mission_id": self._current_mission.mission_id,
            "mission_name": self._current_mission.name,
            "running": self._running,
            "paused": self._paused,
            "current_step": self._current_step_index,
            "total_steps": len(self._current_mission.steps),
            "completed_steps": completed,
            "failed_steps": failed,
            "progress_percent": (completed / len(self._current_mission.steps) * 100) if self._current_mission.steps else 0,
            "step_results": self._step_results,
            "variables": self._variables
        }


# Example mission YAML template
EXAMPLE_MISSION_YAML = '''
# RF Arsenal OS Mission Script
mission_id: wifi_recon_001
name: WiFi Reconnaissance Mission
description: Automated WiFi network discovery and analysis
author: Operator
version: "1.0"
tags:
  - wifi
  - reconnaissance
  - automated

variables:
  target_freq: 2.437e9
  scan_duration: 60
  capture_duration: 10

requirements:
  - wifi_module
  - spectrum_analyzer

steps:
  - id: init_hardware
    type: hardware
    name: Initialize WiFi Hardware
    parameters:
      device: wlan0
      action: enable monitor mode
      
  - id: scan_networks
    type: scan
    name: Scan for WiFi Networks
    parameters:
      freq_start: 2.4e9
      freq_end: 2.5e9
      step: 5e6
      dwell_time: 0.5
    timeout: 120
    
  - id: capture_target
    type: capture
    name: Capture Target Network
    parameters:
      frequency: ${target_freq}
      duration: ${capture_duration}
      sample_rate: 20e6
    condition: "${last_networks_found} > 0"
    
  - id: analyze
    type: analyze
    name: Analyze Captured Data
    parameters:
      type: wifi_demod
      target: last_capture
      
  - id: export_results
    type: export
    name: Export Results
    parameters:
      format: json
      destination: /tmp/wifi_recon_results.json
      
  - id: complete
    type: alert
    name: Mission Complete
    parameters:
      message: "WiFi reconnaissance complete. Found ${networks_count} networks."
      level: info
'''
