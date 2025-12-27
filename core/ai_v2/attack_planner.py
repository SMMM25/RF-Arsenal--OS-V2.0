#!/usr/bin/env python3
"""
RF Arsenal OS - Autonomous Attack Planner

AI-powered attack chain planning and execution.
Generates multi-step attack sequences from high-level objectives.

README COMPLIANCE:
- UNFILTERED: Plans any attack requested
- Offline-first: Works without internet
- RAM-only: No persistent attack logs
- Real operations: Executes actual attacks

Author: RF Arsenal Team
License: Proprietary - Authorized Use Only
"""

import os
import sys
import json
import logging
import threading
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AttackPhase(Enum):
    """MITRE ATT&CK-aligned phases"""
    RECONNAISSANCE = "reconnaissance"
    RESOURCE_DEVELOPMENT = "resource_development"
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    COMMAND_AND_CONTROL = "command_and_control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"


class AttackStatus(Enum):
    """Attack step status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    ABORTED = "aborted"


@dataclass
class AttackStep:
    """A single step in an attack chain"""
    id: str
    phase: AttackPhase
    name: str
    description: str
    command: str
    module: str
    
    # Execution
    status: AttackStatus = AttackStatus.PENDING
    result: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Flow control
    depends_on: List[str] = field(default_factory=list)
    fallback_steps: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    
    # OPSEC
    opsec_notes: str = ""
    risk_level: str = "medium"  # low, medium, high, critical
    stealth_required: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "phase": self.phase.value,
            "name": self.name,
            "description": self.description,
            "command": self.command,
            "module": self.module,
            "status": self.status.value,
            "result": self.result,
            "risk_level": self.risk_level,
            "opsec_notes": self.opsec_notes,
        }


@dataclass 
class AttackChain:
    """A complete attack chain"""
    id: str
    name: str
    objective: str
    
    # Steps
    steps: List[AttackStep] = field(default_factory=list)
    current_step: int = 0
    
    # Status
    status: AttackStatus = AttackStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Context
    target: Dict[str, Any] = field(default_factory=dict)
    discovered_data: Dict[str, Any] = field(default_factory=dict)
    credentials: List[Dict[str, Any]] = field(default_factory=list)
    
    # OPSEC
    stealth_mode: bool = True
    auto_abort_on_detection: bool = True
    
    def add_step(self, step: AttackStep):
        """Add a step to the chain"""
        step.id = f"step_{len(self.steps) + 1}"
        self.steps.append(step)
    
    def get_current_step(self) -> Optional[AttackStep]:
        """Get current step"""
        if 0 <= self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None
    
    def advance(self) -> bool:
        """Advance to next step"""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            return True
        return False
    
    def get_progress(self) -> Dict[str, Any]:
        """Get attack progress"""
        completed = sum(1 for s in self.steps if s.status == AttackStatus.SUCCESS)
        failed = sum(1 for s in self.steps if s.status == AttackStatus.FAILED)
        
        return {
            "total_steps": len(self.steps),
            "current_step": self.current_step + 1,
            "completed": completed,
            "failed": failed,
            "progress_percent": (completed / len(self.steps) * 100) if self.steps else 0,
            "status": self.status.value,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "objective": self.objective,
            "status": self.status.value,
            "progress": self.get_progress(),
            "steps": [s.to_dict() for s in self.steps],
            "target": self.target,
            "stealth_mode": self.stealth_mode,
        }


class AttackPlanner:
    """
    Autonomous attack planning and execution
    
    UNFILTERED - Plans and executes any attack requested.
    This is a professional penetration testing tool.
    """
    
    # Attack templates for common objectives
    ATTACK_TEMPLATES = {
        "wifi_compromise": {
            "phases": [
                {
                    "phase": AttackPhase.RECONNAISSANCE,
                    "name": "WiFi Network Discovery",
                    "command": "scan networks",
                    "module": "wifi",
                    "description": "Scan for target WiFi networks and identify security",
                },
                {
                    "phase": AttackPhase.RECONNAISSANCE,
                    "name": "Client Enumeration",
                    "command": "scan wifi clients",
                    "module": "wifi",
                    "description": "Identify connected clients for targeting",
                },
                {
                    "phase": AttackPhase.CREDENTIAL_ACCESS,
                    "name": "Capture Handshake",
                    "command": "capture handshake {target_ssid}",
                    "module": "wifi",
                    "description": "Capture WPA handshake for offline cracking",
                },
                {
                    "phase": AttackPhase.CREDENTIAL_ACCESS,
                    "name": "Deauth Attack",
                    "command": "deauth {target_bssid}",
                    "module": "wifi",
                    "description": "Force client reconnection to capture handshake",
                },
                {
                    "phase": AttackPhase.INITIAL_ACCESS,
                    "name": "Evil Twin",
                    "command": "evil twin {target_ssid}",
                    "module": "wifi",
                    "description": "Create rogue AP to capture credentials",
                },
            ],
        },
        
        "cellular_intercept": {
            "phases": [
                {
                    "phase": AttackPhase.RECONNAISSANCE,
                    "name": "Cell Tower Scan",
                    "command": "scan lte",
                    "module": "cellular",
                    "description": "Scan for nearby cell towers and frequencies",
                },
                {
                    "phase": AttackPhase.RESOURCE_DEVELOPMENT,
                    "name": "Start Rogue BTS",
                    "command": "start bts",
                    "module": "yatebts",
                    "description": "Start rogue base station to attract targets",
                },
                {
                    "phase": AttackPhase.COLLECTION,
                    "name": "IMSI Catching",
                    "command": "start imsi catcher",
                    "module": "yatebts",
                    "description": "Capture IMSI identifiers from connected devices",
                },
                {
                    "phase": AttackPhase.COLLECTION,
                    "name": "SMS Interception",
                    "command": "intercept sms",
                    "module": "yatebts",
                    "description": "Intercept SMS messages from targets",
                },
            ],
        },
        
        "vehicle_attack": {
            "phases": [
                {
                    "phase": AttackPhase.RECONNAISSANCE,
                    "name": "Key Fob Signal Capture",
                    "command": "capture key fob 433mhz",
                    "module": "vehicle",
                    "description": "Capture key fob signals for analysis",
                },
                {
                    "phase": AttackPhase.RECONNAISSANCE,
                    "name": "TPMS Scan",
                    "command": "scan tpms",
                    "module": "vehicle",
                    "description": "Scan for TPMS sensors to identify vehicles",
                },
                {
                    "phase": AttackPhase.INITIAL_ACCESS,
                    "name": "RollJam Attack",
                    "command": "start rolljam",
                    "module": "rolljam",
                    "description": "Execute RollJam to capture valid code",
                },
                {
                    "phase": AttackPhase.INITIAL_ACCESS,
                    "name": "Relay Attack",
                    "command": "start relay car key",
                    "module": "relay",
                    "description": "Relay key fob signal to unlock vehicle",
                },
            ],
        },
        
        "iot_compromise": {
            "phases": [
                {
                    "phase": AttackPhase.RECONNAISSANCE,
                    "name": "Smart Home Scan",
                    "command": "scan smart home",
                    "module": "iot",
                    "description": "Discover IoT devices and protocols",
                },
                {
                    "phase": AttackPhase.RECONNAISSANCE,
                    "name": "ZigBee Sniff",
                    "command": "sniff zigbee",
                    "module": "iot",
                    "description": "Capture ZigBee traffic for analysis",
                },
                {
                    "phase": AttackPhase.CREDENTIAL_ACCESS,
                    "name": "Extract Network Key",
                    "command": "extract zigbee key",
                    "module": "iot",
                    "description": "Extract ZigBee network encryption key",
                },
                {
                    "phase": AttackPhase.INITIAL_ACCESS,
                    "name": "Unlock Smart Lock",
                    "command": "unlock smart lock",
                    "module": "iot",
                    "description": "Send unlock command to smart lock",
                },
            ],
        },
        
        "drone_takeover": {
            "phases": [
                {
                    "phase": AttackPhase.RECONNAISSANCE,
                    "name": "Drone Detection",
                    "command": "detect drones",
                    "module": "drone",
                    "description": "Scan for drone control frequencies",
                },
                {
                    "phase": AttackPhase.DEFENSE_EVASION,
                    "name": "GPS Jam Drone",
                    "command": "jam gps",
                    "module": "jamming",
                    "description": "Jam GPS to force drone hover/RTH",
                },
                {
                    "phase": AttackPhase.INITIAL_ACCESS,
                    "name": "Control Link Takeover",
                    "command": "hijack drone",
                    "module": "drone",
                    "description": "Take over drone control link",
                },
            ],
        },
        
        "network_pentest": {
            "phases": [
                {
                    "phase": AttackPhase.RECONNAISSANCE,
                    "name": "Port Scan",
                    "command": "port scan {target}",
                    "module": "recon",
                    "description": "Scan for open ports and services",
                },
                {
                    "phase": AttackPhase.RECONNAISSANCE,
                    "name": "Service Fingerprint",
                    "command": "service fingerprint {target}",
                    "module": "recon",
                    "description": "Identify service versions",
                },
                {
                    "phase": AttackPhase.RECONNAISSANCE,
                    "name": "Vulnerability Scan",
                    "command": "vuln scan {target}",
                    "module": "pentest",
                    "description": "Scan for known vulnerabilities",
                },
                {
                    "phase": AttackPhase.INITIAL_ACCESS,
                    "name": "Exploit Vulnerability",
                    "command": "exploit {cve}",
                    "module": "exploit",
                    "description": "Exploit discovered vulnerability",
                },
                {
                    "phase": AttackPhase.CREDENTIAL_ACCESS,
                    "name": "Credential Attack",
                    "command": "brute force ssh {target}",
                    "module": "credential",
                    "description": "Attempt credential attack on services",
                },
                {
                    "phase": AttackPhase.COMMAND_AND_CONTROL,
                    "name": "Deploy C2 Beacon",
                    "command": "generate beacon",
                    "module": "c2",
                    "description": "Deploy C2 beacon for persistent access",
                },
            ],
        },
    }
    
    def __init__(self):
        self.logger = logging.getLogger('AttackPlanner')
        self._llm = None  # Lazy loaded
        self._command_center = None  # Reference to AI Command Center
        
        # Active attack chains (RAM-only)
        self._active_chains: Dict[str, AttackChain] = {}
        self._lock = threading.Lock()
        
        # Execution callbacks
        self._step_callbacks: List[Callable[[AttackStep], None]] = []
        self._progress_callbacks: List[Callable[[AttackChain], None]] = []
        
        self.logger.info("Attack Planner initialized (unfiltered mode)")
    
    def _get_llm(self):
        """Get LLM for intelligent planning"""
        if self._llm is None:
            try:
                from .local_llm import LocalLLM
                self._llm = LocalLLM()
            except ImportError:
                pass
        return self._llm
    
    def set_command_center(self, command_center):
        """Set reference to AI Command Center for execution"""
        self._command_center = command_center
    
    def plan_attack(
        self,
        objective: str,
        target: Dict[str, Any] = None,
        constraints: Dict[str, Any] = None,
    ) -> AttackChain:
        """
        Generate an attack plan for an objective
        
        UNFILTERED - Will plan any attack requested.
        
        Args:
            objective: High-level attack objective
            target: Target information
            constraints: Any operational constraints
            
        Returns:
            AttackChain with planned steps
        """
        import secrets
        
        chain = AttackChain(
            id=secrets.token_hex(8),
            name=f"Attack: {objective[:50]}",
            objective=objective,
            target=target or {},
        )
        
        # Apply constraints
        if constraints:
            chain.stealth_mode = constraints.get('stealth', True)
            chain.auto_abort_on_detection = constraints.get('auto_abort', True)
        
        # Try to find matching template
        template = self._find_template(objective)
        
        if template:
            # Use template
            chain = self._apply_template(chain, template, target)
        else:
            # Use LLM to generate custom plan
            chain = self._generate_custom_plan(chain, objective, target, constraints)
        
        # Store chain
        with self._lock:
            self._active_chains[chain.id] = chain
        
        return chain
    
    def _find_template(self, objective: str) -> Optional[Dict]:
        """Find a matching attack template"""
        objective_lower = objective.lower()
        
        # Keyword matching
        if any(w in objective_lower for w in ['wifi', 'wireless', 'wlan', 'access point']):
            return self.ATTACK_TEMPLATES['wifi_compromise']
        elif any(w in objective_lower for w in ['cellular', 'cell', 'imsi', 'sms', 'phone']):
            return self.ATTACK_TEMPLATES['cellular_intercept']
        elif any(w in objective_lower for w in ['car', 'vehicle', 'key fob', 'tpms']):
            return self.ATTACK_TEMPLATES['vehicle_attack']
        elif any(w in objective_lower for w in ['iot', 'smart home', 'smart lock', 'zigbee']):
            return self.ATTACK_TEMPLATES['iot_compromise']
        elif any(w in objective_lower for w in ['drone', 'uav', 'quadcopter']):
            return self.ATTACK_TEMPLATES['drone_takeover']
        elif any(w in objective_lower for w in ['network', 'server', 'web', 'pentest']):
            return self.ATTACK_TEMPLATES['network_pentest']
        
        return None
    
    def _apply_template(
        self,
        chain: AttackChain,
        template: Dict,
        target: Dict[str, Any],
    ) -> AttackChain:
        """Apply a template to create attack steps"""
        for i, phase_data in enumerate(template['phases']):
            # Substitute target variables in command
            command = phase_data['command']
            if target:
                for key, value in target.items():
                    command = command.replace(f"{{{key}}}", str(value))
            
            step = AttackStep(
                id=f"step_{i+1}",
                phase=phase_data['phase'],
                name=phase_data['name'],
                description=phase_data['description'],
                command=command,
                module=phase_data['module'],
            )
            
            chain.steps.append(step)
        
        return chain
    
    def _generate_custom_plan(
        self,
        chain: AttackChain,
        objective: str,
        target: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> AttackChain:
        """Generate custom attack plan using LLM"""
        llm = self._get_llm()
        
        if not llm or not llm.is_available():
            # Fallback to basic plan
            step = AttackStep(
                id="step_1",
                phase=AttackPhase.RECONNAISSANCE,
                name="Initial Reconnaissance",
                description=f"Gather information about: {objective}",
                command=f"scan {objective}",
                module="system",
            )
            chain.steps.append(step)
            return chain
        
        # Use LLM to generate plan
        steps = llm.plan_attack(objective, {"target": target, "constraints": constraints})
        
        for i, step_data in enumerate(steps):
            # Map phase string to enum
            phase_str = step_data.get('phase', 'reconnaissance').lower().replace(' ', '_')
            try:
                phase = AttackPhase[phase_str.upper()]
            except KeyError:
                phase = AttackPhase.EXECUTION
            
            step = AttackStep(
                id=f"step_{i+1}",
                phase=phase,
                name=step_data.get('action', f'Step {i+1}'),
                description=step_data.get('expected_outcome', ''),
                command=step_data.get('command', ''),
                module=step_data.get('module', 'system'),
                opsec_notes=step_data.get('opsec_notes', ''),
            )
            
            chain.steps.append(step)
        
        return chain
    
    def execute_chain(
        self,
        chain_id: str,
        auto_advance: bool = True,
    ) -> bool:
        """
        Execute an attack chain
        
        Args:
            chain_id: ID of chain to execute
            auto_advance: Automatically advance through steps
            
        Returns:
            True if execution started
        """
        chain = self._active_chains.get(chain_id)
        if not chain:
            self.logger.error(f"Chain not found: {chain_id}")
            return False
        
        chain.status = AttackStatus.IN_PROGRESS
        chain.started_at = datetime.now()
        
        if auto_advance:
            # Execute all steps
            while chain.current_step < len(chain.steps):
                step = chain.get_current_step()
                if step:
                    success = self._execute_step(chain, step)
                    
                    if not success and chain.auto_abort_on_detection:
                        chain.status = AttackStatus.ABORTED
                        break
                    
                    chain.advance()
                else:
                    break
            
            # Mark complete
            if chain.status != AttackStatus.ABORTED:
                chain.status = AttackStatus.SUCCESS
                chain.completed_at = datetime.now()
        
        return True
    
    def execute_step(self, chain_id: str) -> Optional[Dict[str, Any]]:
        """Execute current step of a chain"""
        chain = self._active_chains.get(chain_id)
        if not chain:
            return None
        
        step = chain.get_current_step()
        if not step:
            return None
        
        success = self._execute_step(chain, step)
        
        return {
            "step": step.to_dict(),
            "success": success,
            "can_advance": chain.current_step < len(chain.steps) - 1,
        }
    
    def _execute_step(self, chain: AttackChain, step: AttackStep) -> bool:
        """Execute a single attack step"""
        step.status = AttackStatus.IN_PROGRESS
        step.started_at = datetime.now()
        
        # Notify callbacks
        for callback in self._step_callbacks:
            try:
                callback(step)
            except Exception:
                pass
        
        try:
            # Execute command via AI Command Center
            if self._command_center:
                result = self._command_center.process_command(step.command)
                step.result = {
                    "success": result.success,
                    "message": result.message,
                    "data": result.data,
                }
                step.status = AttackStatus.SUCCESS if result.success else AttackStatus.FAILED
            else:
                # Dry run - just mark as success
                step.result = {"message": "Dry run - command not executed"}
                step.status = AttackStatus.SUCCESS
            
            step.completed_at = datetime.now()
            
            # Store any discovered data
            if step.result.get('data'):
                chain.discovered_data[step.id] = step.result['data']
            
            return step.status == AttackStatus.SUCCESS
            
        except Exception as e:
            step.status = AttackStatus.FAILED
            step.result = {"error": str(e)}
            step.completed_at = datetime.now()
            
            # Retry logic
            if step.retry_count < step.max_retries:
                step.retry_count += 1
                return self._execute_step(chain, step)
            
            return False
    
    def advance_chain(self, chain_id: str) -> bool:
        """Manually advance to next step"""
        chain = self._active_chains.get(chain_id)
        if chain:
            return chain.advance()
        return False
    
    def abort_chain(self, chain_id: str) -> bool:
        """Abort an active chain"""
        chain = self._active_chains.get(chain_id)
        if chain:
            chain.status = AttackStatus.ABORTED
            chain.completed_at = datetime.now()
            return True
        return False
    
    def get_chain(self, chain_id: str) -> Optional[AttackChain]:
        """Get a chain by ID"""
        return self._active_chains.get(chain_id)
    
    def get_active_chains(self) -> List[AttackChain]:
        """Get all active chains"""
        return list(self._active_chains.values())
    
    def add_step_callback(self, callback: Callable[[AttackStep], None]):
        """Add callback for step execution"""
        self._step_callbacks.append(callback)
    
    def add_progress_callback(self, callback: Callable[[AttackChain], None]):
        """Add callback for chain progress"""
        self._progress_callbacks.append(callback)
    
    def cleanup_completed(self):
        """Remove completed chains from memory"""
        with self._lock:
            to_remove = [
                cid for cid, chain in self._active_chains.items()
                if chain.status in [AttackStatus.SUCCESS, AttackStatus.ABORTED, AttackStatus.FAILED]
            ]
            for cid in to_remove:
                del self._active_chains[cid]
    
    # === Methods for AI Command Center Integration ===
    
    def get_status(self) -> Dict[str, Any]:
        """Get attack planner status"""
        active_chains = [c for c in self._active_chains.values() if c.status == AttackStatus.IN_PROGRESS]
        
        total_steps = sum(len(c.steps) for c in active_chains)
        completed_steps = sum(c.current_step for c in active_chains)
        success_count = sum(1 for c in self._active_chains.values() if c.status == AttackStatus.SUCCESS)
        
        current_chain = active_chains[0] if active_chains else None
        
        return {
            "active": len(active_chains) > 0,
            "phase": current_chain.name if current_chain else "None",
            "progress": int((completed_steps / total_steps) * 100) if total_steps > 0 else 0,
            "targets_found": len(current_chain.discovered_data) if current_chain else 0,
            "attacks_executed": completed_steps,
            "success_rate": (success_count / len(self._active_chains)) * 100 if self._active_chains else 0,
            "total_chains": len(self._active_chains),
        }
    
    def create_plan(self, target: str = None) -> Dict[str, Any]:
        """Create a new attack plan (convenience method)"""
        chain = self.plan_attack(target or "all targets in range")
        
        return {
            "plan_id": chain.id if chain else None,
            "steps": [s.to_dict() for s in chain.steps] if chain else [],
            "target": target or "all in range",
        }
    
    def execute_current_plan(self) -> Dict[str, Any]:
        """Execute the most recently created plan"""
        if not self._active_chains:
            return {"status": "error", "message": "No plan to execute"}
        
        # Get most recent chain
        chain_id = list(self._active_chains.keys())[-1]
        
        # Start execution in background
        import threading
        threading.Thread(target=self.execute, args=(chain_id,)).start()
        
        chain = self._active_chains[chain_id]
        return {
            "status": "running",
            "completed": chain.current_step,
            "total": len(chain.steps),
        }
    
    def stop(self):
        """Stop all active attacks"""
        for chain_id in list(self._active_chains.keys()):
            self.abort_chain(chain_id)
    
    def autonomous_attack(self, target: str = None) -> Dict[str, Any]:
        """Run a fully autonomous attack"""
        chain = self.plan_attack(target or "all accessible targets")
        
        if chain:
            # Start execution
            import threading
            threading.Thread(target=self.execute, args=(chain.id,)).start()
            
            return {
                "chain_id": chain.id,
                "status": "autonomous",
                "target": target or "auto-detect",
                "steps": len(chain.steps),
            }
        
        return {"status": "error", "message": "Failed to create attack plan"}


# Convenience function
def get_attack_planner() -> AttackPlanner:
    """Get attack planner instance"""
    return AttackPlanner()
