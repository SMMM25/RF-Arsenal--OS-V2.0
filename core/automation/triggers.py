"""
RF Arsenal OS - Trigger Actions System
Event-driven trigger system for automated responses to RF events.
"""

import time
import threading
import hashlib
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import json


class TriggerType(Enum):
    """Types of trigger conditions"""
    SIGNAL_DETECTED = "signal_detected"
    SIGNAL_LOST = "signal_lost"
    POWER_THRESHOLD = "power_threshold"
    FREQUENCY_MATCH = "frequency_match"
    PATTERN_MATCH = "pattern_match"
    DEVICE_DETECTED = "device_detected"
    IMSI_DETECTED = "imsi_detected"
    TIME_BASED = "time_based"
    COUNTER = "counter"
    COMPOUND = "compound"  # Multiple conditions
    CUSTOM = "custom"  # Custom Python function


class ActionType(Enum):
    """Types of actions to execute"""
    COMMAND = "command"
    CAPTURE = "capture"
    ALERT = "alert"
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SCRIPT = "script"
    CHAIN = "chain"  # Trigger another trigger
    EMERGENCY = "emergency"
    CUSTOM = "custom"


class TriggerState(Enum):
    """Trigger states"""
    ARMED = "armed"
    DISARMED = "disarmed"
    TRIGGERED = "triggered"
    COOLDOWN = "cooldown"


@dataclass
class TriggerCondition:
    """A trigger condition"""
    condition_id: str
    trigger_type: TriggerType
    parameters: Dict[str, Any]
    
    # For compound triggers
    logic: str = "AND"  # AND, OR
    sub_conditions: List['TriggerCondition'] = field(default_factory=list)
    
    def evaluate(self, event_data: Dict) -> bool:
        """Evaluate if condition is met"""
        if self.trigger_type == TriggerType.SIGNAL_DETECTED:
            freq = event_data.get("frequency_hz", 0)
            target_freq = self.parameters.get("frequency_hz", 0)
            tolerance = self.parameters.get("tolerance_hz", 100e3)
            return abs(freq - target_freq) <= tolerance
            
        elif self.trigger_type == TriggerType.SIGNAL_LOST:
            signal_present = event_data.get("signal_present", True)
            return not signal_present
            
        elif self.trigger_type == TriggerType.POWER_THRESHOLD:
            power = event_data.get("power_dbm", -200)
            threshold = self.parameters.get("threshold_dbm", -60)
            direction = self.parameters.get("direction", "above")  # above or below
            
            if direction == "above":
                return power >= threshold
            else:
                return power <= threshold
                
        elif self.trigger_type == TriggerType.FREQUENCY_MATCH:
            freq = event_data.get("frequency_hz", 0)
            patterns = self.parameters.get("patterns", [])
            
            for pattern in patterns:
                if isinstance(pattern, dict):
                    if pattern.get("start", 0) <= freq <= pattern.get("end", float('inf')):
                        return True
                else:
                    if abs(freq - pattern) <= self.parameters.get("tolerance_hz", 100e3):
                        return True
            return False
            
        elif self.trigger_type == TriggerType.PATTERN_MATCH:
            data = event_data.get("data", "")
            pattern = self.parameters.get("pattern", "")
            regex = self.parameters.get("regex", False)
            
            if regex:
                return bool(re.search(pattern, str(data)))
            else:
                return pattern in str(data)
                
        elif self.trigger_type == TriggerType.DEVICE_DETECTED:
            device_type = event_data.get("device_type", "")
            target_types = self.parameters.get("device_types", [])
            return device_type in target_types
            
        elif self.trigger_type == TriggerType.IMSI_DETECTED:
            imsi = event_data.get("imsi", "")
            target_imsis = self.parameters.get("imsis", [])
            mcc_mnc = self.parameters.get("mcc_mnc", "")
            
            if target_imsis and imsi in target_imsis:
                return True
            if mcc_mnc and imsi.startswith(mcc_mnc):
                return True
            return False
            
        elif self.trigger_type == TriggerType.TIME_BASED:
            current_hour = time.localtime().tm_hour
            current_minute = time.localtime().tm_min
            
            start_hour = self.parameters.get("start_hour", 0)
            end_hour = self.parameters.get("end_hour", 24)
            
            return start_hour <= current_hour < end_hour
            
        elif self.trigger_type == TriggerType.COUNTER:
            count = event_data.get("count", 0)
            threshold = self.parameters.get("threshold", 1)
            return count >= threshold
            
        elif self.trigger_type == TriggerType.COMPOUND:
            if self.logic == "AND":
                return all(c.evaluate(event_data) for c in self.sub_conditions)
            else:  # OR
                return any(c.evaluate(event_data) for c in self.sub_conditions)
                
        elif self.trigger_type == TriggerType.CUSTOM:
            # Custom evaluation function (passed in parameters)
            eval_func = self.parameters.get("eval_func")
            if callable(eval_func):
                return eval_func(event_data)
            return False
            
        return False
        
    def to_dict(self) -> Dict:
        return {
            "condition_id": self.condition_id,
            "trigger_type": self.trigger_type.value,
            "parameters": self.parameters,
            "logic": self.logic,
            "sub_conditions": [c.to_dict() for c in self.sub_conditions]
        }


@dataclass
class TriggerAction:
    """Action to execute when trigger fires"""
    action_id: str
    action_type: ActionType
    parameters: Dict[str, Any]
    
    # Execution settings
    delay_s: float = 0.0
    async_execution: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "parameters": self.parameters,
            "delay_s": self.delay_s,
            "async_execution": self.async_execution
        }


@dataclass 
class Trigger:
    """Complete trigger definition"""
    trigger_id: str
    name: str
    description: str = ""
    condition: Optional[TriggerCondition] = None
    actions: List[TriggerAction] = field(default_factory=list)
    
    # Trigger settings
    enabled: bool = True
    one_shot: bool = False  # Disarm after first trigger
    cooldown_s: float = 5.0  # Minimum time between triggers
    max_triggers: int = 0  # 0 = unlimited
    
    # Runtime state
    state: TriggerState = TriggerState.ARMED
    trigger_count: int = 0
    last_triggered: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "trigger_id": self.trigger_id,
            "name": self.name,
            "description": self.description,
            "condition": self.condition.to_dict() if self.condition else None,
            "actions": [a.to_dict() for a in self.actions],
            "enabled": self.enabled,
            "one_shot": self.one_shot,
            "cooldown_s": self.cooldown_s,
            "max_triggers": self.max_triggers,
            "state": self.state.value,
            "trigger_count": self.trigger_count,
            "last_triggered": self.last_triggered
        }


class TriggerEngine:
    """
    Production-grade trigger engine for RF Arsenal OS.
    
    Features:
    - Multiple trigger types (signal, power, pattern, device, IMSI)
    - Compound conditions (AND/OR logic)
    - Multiple actions per trigger
    - Cooldown periods
    - One-shot triggers
    - Action chaining
    - Custom Python triggers
    - Real-time event processing
    - Thread-safe operation
    """
    
    def __init__(self, command_handler: Optional[Callable] = None):
        """
        Initialize trigger engine.
        
        Args:
            command_handler: Function to execute commands
        """
        self.command_handler = command_handler or (lambda cmd: {"success": True})
        
        # Trigger storage
        self._triggers: Dict[str, Trigger] = {}
        
        # Event queue
        self._event_queue: List[Dict] = []
        self._event_lock = threading.Lock()
        
        # Processing state
        self._running = False
        self._processor_thread: Optional[threading.Thread] = None
        
        # Counters for counter-type triggers
        self._counters: Dict[str, int] = {}
        
        # Action handlers
        self._action_handlers: Dict[ActionType, Callable] = {
            ActionType.COMMAND: self._handle_command_action,
            ActionType.CAPTURE: self._handle_capture_action,
            ActionType.ALERT: self._handle_alert_action,
            ActionType.LOG: self._handle_log_action,
            ActionType.WEBHOOK: self._handle_webhook_action,
            ActionType.CHAIN: self._handle_chain_action,
            ActionType.EMERGENCY: self._handle_emergency_action,
        }
        
        # Callbacks
        self._trigger_callbacks: List[Callable] = []
        
    def create_trigger(self,
                      name: str,
                      condition: TriggerCondition,
                      actions: List[TriggerAction],
                      **kwargs) -> str:
        """
        Create and add a new trigger.
        
        Args:
            name: Trigger name
            condition: Trigger condition
            actions: Actions to execute
            **kwargs: Additional trigger parameters
            
        Returns:
            Trigger ID
        """
        trigger_id = hashlib.sha256(f"{name}_{time.time()}".encode()).hexdigest()[:16]
        
        trigger = Trigger(
            trigger_id=trigger_id,
            name=name,
            condition=condition,
            actions=actions,
            **kwargs
        )
        
        self._triggers[trigger_id] = trigger
        return trigger_id
        
    def add_trigger(self, trigger: Trigger) -> str:
        """Add existing trigger"""
        self._triggers[trigger.trigger_id] = trigger
        return trigger.trigger_id
        
    def remove_trigger(self, trigger_id: str) -> bool:
        """Remove a trigger"""
        if trigger_id in self._triggers:
            del self._triggers[trigger_id]
            return True
        return False
        
    def arm_trigger(self, trigger_id: str) -> bool:
        """Arm a trigger"""
        if trigger_id in self._triggers:
            self._triggers[trigger_id].state = TriggerState.ARMED
            self._triggers[trigger_id].enabled = True
            return True
        return False
        
    def disarm_trigger(self, trigger_id: str) -> bool:
        """Disarm a trigger"""
        if trigger_id in self._triggers:
            self._triggers[trigger_id].state = TriggerState.DISARMED
            self._triggers[trigger_id].enabled = False
            return True
        return False
        
    def arm_all(self) -> None:
        """Arm all triggers"""
        for trigger in self._triggers.values():
            trigger.state = TriggerState.ARMED
            trigger.enabled = True
            
    def disarm_all(self) -> None:
        """Disarm all triggers"""
        for trigger in self._triggers.values():
            trigger.state = TriggerState.DISARMED
            trigger.enabled = False
            
    def start(self) -> None:
        """Start the trigger engine"""
        if self._running:
            return
            
        self._running = True
        self._processor_thread = threading.Thread(
            target=self._process_events,
            daemon=True
        )
        self._processor_thread.start()
        
    def stop(self) -> None:
        """Stop the trigger engine"""
        self._running = False
        
    def process_event(self, event_type: str, event_data: Dict) -> List[str]:
        """
        Process an event and check triggers.
        
        Args:
            event_type: Type of event
            event_data: Event data
            
        Returns:
            List of triggered trigger IDs
        """
        event = {
            "type": event_type,
            "data": event_data,
            "timestamp": time.time()
        }
        
        with self._event_lock:
            self._event_queue.append(event)
            
        return self._check_triggers(event_data)
        
    def _process_events(self) -> None:
        """Background event processor"""
        while self._running:
            events_to_process = []
            
            with self._event_lock:
                events_to_process = self._event_queue.copy()
                self._event_queue.clear()
                
            for event in events_to_process:
                self._check_triggers(event["data"])
                
            time.sleep(0.01)  # 10ms polling
            
    def _check_triggers(self, event_data: Dict) -> List[str]:
        """Check all triggers against event data"""
        triggered_ids = []
        current_time = time.time()
        
        for trigger in self._triggers.values():
            # Skip disabled triggers
            if not trigger.enabled or trigger.state == TriggerState.DISARMED:
                continue
                
            # Check cooldown
            if trigger.state == TriggerState.COOLDOWN:
                if trigger.last_triggered and (current_time - trigger.last_triggered) < trigger.cooldown_s:
                    continue
                trigger.state = TriggerState.ARMED
                
            # Check max triggers
            if trigger.max_triggers > 0 and trigger.trigger_count >= trigger.max_triggers:
                trigger.state = TriggerState.DISARMED
                continue
                
            # Evaluate condition
            if trigger.condition and trigger.condition.evaluate(event_data):
                self._fire_trigger(trigger, event_data)
                triggered_ids.append(trigger.trigger_id)
                
        return triggered_ids
        
    def _fire_trigger(self, trigger: Trigger, event_data: Dict) -> None:
        """Fire a trigger and execute its actions"""
        trigger.state = TriggerState.TRIGGERED
        trigger.trigger_count += 1
        trigger.last_triggered = time.time()
        
        # Notify callbacks
        for callback in self._trigger_callbacks:
            try:
                callback(trigger, event_data)
            except Exception:
                pass
                
        # Execute actions
        for action in trigger.actions:
            if action.async_execution:
                threading.Thread(
                    target=self._execute_action,
                    args=(action, event_data),
                    daemon=True
                ).start()
            else:
                self._execute_action(action, event_data)
                
        # Update state
        if trigger.one_shot:
            trigger.state = TriggerState.DISARMED
            trigger.enabled = False
        else:
            trigger.state = TriggerState.COOLDOWN
            
    def _execute_action(self, action: TriggerAction, event_data: Dict) -> None:
        """Execute a single action"""
        # Apply delay
        if action.delay_s > 0:
            time.sleep(action.delay_s)
            
        # Get handler
        handler = self._action_handlers.get(action.action_type)
        if handler:
            try:
                handler(action, event_data)
            except Exception as e:
                print(f"[TRIGGER ERROR] Action {action.action_id} failed: {e}")
        elif action.action_type == ActionType.CUSTOM:
            self._handle_custom_action(action, event_data)
            
    def _handle_command_action(self, action: TriggerAction, event_data: Dict) -> None:
        """Execute command action"""
        command = action.parameters.get("command", "")
        
        # Substitute event data placeholders
        for key, value in event_data.items():
            command = command.replace(f"${{{key}}}", str(value))
            
        self.command_handler(command)
        
    def _handle_capture_action(self, action: TriggerAction, event_data: Dict) -> None:
        """Execute capture action"""
        frequency = action.parameters.get("frequency_hz") or event_data.get("frequency_hz", 0)
        duration = action.parameters.get("duration_s", 5.0)
        
        command = f"capture iq at {frequency/1e6:.3f}MHz for {duration}s"
        self.command_handler(command)
        
    def _handle_alert_action(self, action: TriggerAction, event_data: Dict) -> None:
        """Execute alert action"""
        message = action.parameters.get("message", "Alert triggered")
        level = action.parameters.get("level", "warning")
        
        # Substitute placeholders
        for key, value in event_data.items():
            message = message.replace(f"${{{key}}}", str(value))
            
        print(f"[ALERT - {level.upper()}] {message}")
        
        # In production, this would send actual alerts (SMS, push notification, etc.)
        
    def _handle_log_action(self, action: TriggerAction, event_data: Dict) -> None:
        """Execute log action"""
        message = action.parameters.get("message", "")
        filepath = action.parameters.get("filepath")
        
        # Substitute placeholders
        for key, value in event_data.items():
            message = message.replace(f"${{{key}}}", str(value))
            
        log_entry = {
            "timestamp": time.time(),
            "message": message,
            "event_data": event_data
        }
        
        if filepath:
            try:
                with open(filepath, 'a') as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception:
                pass
        else:
            print(f"[TRIGGER LOG] {message}")
            
    def _handle_webhook_action(self, action: TriggerAction, event_data: Dict) -> None:
        """Execute webhook action (offline-safe)"""
        url = action.parameters.get("url", "")
        method = action.parameters.get("method", "POST")
        
        # In production offline mode, queue for later or skip
        print(f"[WEBHOOK] Would {method} to {url} with data: {event_data}")
        
    def _handle_chain_action(self, action: TriggerAction, event_data: Dict) -> None:
        """Chain to another trigger"""
        target_trigger_id = action.parameters.get("trigger_id")
        
        if target_trigger_id in self._triggers:
            trigger = self._triggers[target_trigger_id]
            self._fire_trigger(trigger, event_data)
            
    def _handle_emergency_action(self, action: TriggerAction, event_data: Dict) -> None:
        """Execute emergency action"""
        emergency_type = action.parameters.get("type", "alert")
        
        if emergency_type == "stop_all":
            self.command_handler("emergency stop")
        elif emergency_type == "wipe":
            self.command_handler("emergency wipe")
        elif emergency_type == "stealth":
            self.command_handler("go offline")
            self.command_handler("enable stealth mode")
            
    def _handle_custom_action(self, action: TriggerAction, event_data: Dict) -> None:
        """Execute custom Python action"""
        func = action.parameters.get("function")
        if callable(func):
            try:
                func(event_data, action.parameters)
            except Exception as e:
                print(f"[TRIGGER ERROR] Custom action failed: {e}")
                
    def increment_counter(self, counter_name: str, amount: int = 1) -> int:
        """Increment a counter and return new value"""
        self._counters[counter_name] = self._counters.get(counter_name, 0) + amount
        
        # Process as event for counter triggers
        self.process_event("counter", {
            "counter_name": counter_name,
            "count": self._counters[counter_name]
        })
        
        return self._counters[counter_name]
        
    def reset_counter(self, counter_name: str) -> None:
        """Reset a counter"""
        self._counters[counter_name] = 0
        
    def get_counter(self, counter_name: str) -> int:
        """Get counter value"""
        return self._counters.get(counter_name, 0)
        
    def register_callback(self, callback: Callable) -> None:
        """Register callback for trigger events"""
        self._trigger_callbacks.append(callback)
        
    def get_trigger(self, trigger_id: str) -> Optional[Trigger]:
        """Get trigger by ID"""
        return self._triggers.get(trigger_id)
        
    def get_all_triggers(self) -> List[Dict]:
        """Get all triggers"""
        return [t.to_dict() for t in self._triggers.values()]
        
    def get_armed_triggers(self) -> List[Dict]:
        """Get armed triggers"""
        return [
            t.to_dict() for t in self._triggers.values()
            if t.state == TriggerState.ARMED
        ]
        
    def get_status(self) -> Dict[str, Any]:
        """Get trigger engine status"""
        return {
            "running": self._running,
            "total_triggers": len(self._triggers),
            "armed_triggers": sum(1 for t in self._triggers.values() if t.state == TriggerState.ARMED),
            "triggered_count": sum(t.trigger_count for t in self._triggers.values()),
            "counters": self._counters.copy()
        }


# Convenience functions for creating common triggers
def create_signal_trigger(name: str,
                         frequency_hz: float,
                         tolerance_hz: float = 100e3,
                         command: str = "",
                         **kwargs) -> Trigger:
    """Create a signal detection trigger"""
    condition = TriggerCondition(
        condition_id=f"cond_{name}",
        trigger_type=TriggerType.SIGNAL_DETECTED,
        parameters={
            "frequency_hz": frequency_hz,
            "tolerance_hz": tolerance_hz
        }
    )
    
    actions = []
    if command:
        actions.append(TriggerAction(
            action_id=f"action_{name}",
            action_type=ActionType.COMMAND,
            parameters={"command": command}
        ))
        
    return Trigger(
        trigger_id=f"trigger_{name}_{int(time.time())}",
        name=name,
        condition=condition,
        actions=actions,
        **kwargs
    )


def create_imsi_trigger(name: str,
                       imsis: List[str] = None,
                       mcc_mnc: str = "",
                       alert_message: str = "",
                       **kwargs) -> Trigger:
    """Create an IMSI detection trigger"""
    condition = TriggerCondition(
        condition_id=f"cond_{name}",
        trigger_type=TriggerType.IMSI_DETECTED,
        parameters={
            "imsis": imsis or [],
            "mcc_mnc": mcc_mnc
        }
    )
    
    actions = [
        TriggerAction(
            action_id=f"alert_{name}",
            action_type=ActionType.ALERT,
            parameters={
                "message": alert_message or "IMSI detected: ${imsi}",
                "level": "warning"
            }
        )
    ]
    
    return Trigger(
        trigger_id=f"trigger_{name}_{int(time.time())}",
        name=name,
        condition=condition,
        actions=actions,
        **kwargs
    )


def create_power_threshold_trigger(name: str,
                                  threshold_dbm: float,
                                  direction: str = "above",
                                  command: str = "",
                                  **kwargs) -> Trigger:
    """Create a power threshold trigger"""
    condition = TriggerCondition(
        condition_id=f"cond_{name}",
        trigger_type=TriggerType.POWER_THRESHOLD,
        parameters={
            "threshold_dbm": threshold_dbm,
            "direction": direction
        }
    )
    
    actions = []
    if command:
        actions.append(TriggerAction(
            action_id=f"action_{name}",
            action_type=ActionType.COMMAND,
            parameters={"command": command}
        ))
        
    return Trigger(
        trigger_id=f"trigger_{name}_{int(time.time())}",
        name=name,
        condition=condition,
        actions=actions,
        **kwargs
    )
