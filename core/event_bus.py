"""
RF Arsenal OS - Event Bus & Message System
==========================================

Centralized event bus for cross-module communication and coordination.
Enables loose coupling between modules while maintaining system coherence.

Author: RF Arsenal Core Team
Version: 2.0.0
License: Authorized Use Only
"""

import logging
from typing import Callable, List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
from threading import Lock
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class EventType(Enum):
    """System-wide event types."""
    # Device events
    DEVICE_DETECTED = "device_detected"
    DEVICE_IDENTIFIED = "device_identified"
    DEVICE_LOST = "device_lost"
    
    # Signal events
    SIGNAL_DETECTED = "signal_detected"
    SIGNAL_CLASSIFIED = "signal_classified"
    SIGNAL_LOST = "signal_lost"
    
    # Geolocation events
    POSITION_UPDATE = "position_update"
    MOVEMENT_DETECTED = "movement_detected"
    TARGET_ARRIVED = "target_arrived"
    
    # Security events
    ANOMALY_DETECTED = "anomaly_detected"
    THREAT_DETECTED = "threat_detected"
    STEALTH_VIOLATION = "stealth_violation"
    TRANSMISSION_DETECTED = "transmission_detected"
    
    # System events
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    MODULE_LOADED = "module_loaded"
    MODULE_ERROR = "module_error"


@dataclass
class Event:
    """Event message with metadata."""
    event_type: EventType
    timestamp: datetime
    source_module: str
    data: Dict[str, Any] = field(default_factory=dict)
    priority: str = "normal"  # "low", "normal", "high", "critical"
    
    def to_dict(self) -> Dict:
        """Convert event to dictionary."""
        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'source_module': self.source_module,
            'data': self.data,
            'priority': self.priority
        }


class EventBus:
    """
    Centralized event bus for system-wide coordination.
    
    Features:
    - Event publishing and subscription
    - Priority-based event handling
    - Event filtering
    - Event history and logging
    - Thread-safe operations
    - Wildcard subscriptions
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize event bus.
        
        Args:
            max_history: Maximum number of events to keep in history
        """
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._wildcard_subscribers: List[Callable] = []
        self._event_history: List[Event] = []
        self._max_history = max_history
        self._lock = Lock()
        
        logger.info("✅ Event Bus initialized")
    
    def subscribe(self, event_type: EventType, callback: Callable):
        """
        Subscribe to specific event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs (signature: callback(event))
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            
            if callback not in self._subscribers[event_type]:
                self._subscribers[event_type].append(callback)
                logger.info(f"✅ Subscribed to {event_type.value}")
    
    def subscribe_all(self, callback: Callable):
        """
        Subscribe to all events (wildcard subscription).
        
        Args:
            callback: Function to call for any event
        """
        with self._lock:
            if callback not in self._wildcard_subscribers:
                self._wildcard_subscribers.append(callback)
                logger.info("✅ Subscribed to all events (wildcard)")
    
    def unsubscribe(self, event_type: EventType, callback: Callable):
        """
        Unsubscribe from event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback function to remove
        """
        with self._lock:
            if event_type in self._subscribers:
                if callback in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(callback)
                    logger.info(f"🔕 Unsubscribed from {event_type.value}")
    
    def publish(self, event: Event):
        """
        Publish event to all subscribers.
        
        Args:
            event: Event to publish
        """
        with self._lock:
            # Add to history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]
            
            # Log event
            logger.debug(f"📢 Event: {event.event_type.value} from {event.source_module}")
            
            # Notify specific subscribers
            subscribers = self._subscribers.get(event.event_type, [])
            for callback in subscribers:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Subscriber callback failed: {e}")
            
            # Notify wildcard subscribers
            for callback in self._wildcard_subscribers:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Wildcard callback failed: {e}")
    
    def emit(self, 
             event_type: EventType,
             source_module: str,
             data: Optional[Dict] = None,
             priority: str = "normal"):
        """
        Convenience method to create and publish event.
        
        Args:
            event_type: Type of event
            source_module: Module emitting the event
            data: Event data (optional)
            priority: Event priority (default: "normal")
        """
        event = Event(
            event_type=event_type,
            timestamp=datetime.now(),
            source_module=source_module,
            data=data or {},
            priority=priority
        )
        self.publish(event)
    
    def get_history(self, 
                    event_type: Optional[EventType] = None,
                    limit: int = 100) -> List[Event]:
        """
        Get event history.
        
        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return
        
        Returns:
            List of events
        """
        with self._lock:
            events = self._event_history
            
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            return events[-limit:]
    
    def get_statistics(self) -> Dict:
        """Get event bus statistics."""
        with self._lock:
            # Count by event type
            by_type = {}
            by_priority = {'low': 0, 'normal': 0, 'high': 0, 'critical': 0}
            
            for event in self._event_history:
                event_type = event.event_type.value
                by_type[event_type] = by_type.get(event_type, 0) + 1
                by_priority[event.priority] += 1
            
            # Count subscribers
            total_subscribers = sum(len(subs) for subs in self._subscribers.values())
            
            return {
                'total_events': len(self._event_history),
                'by_type': by_type,
                'by_priority': by_priority,
                'total_subscribers': total_subscribers,
                'wildcard_subscribers': len(self._wildcard_subscribers),
                'event_types_subscribed': len(self._subscribers)
            }


# Singleton instance
_event_bus_instance: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get singleton event bus instance."""
    global _event_bus_instance
    if _event_bus_instance is None:
        _event_bus_instance = EventBus()
    return _event_bus_instance


# Convenience functions
def emit_event(event_type: EventType, source_module: str, data: Optional[Dict] = None, priority: str = "normal"):
    """Convenience function to emit event using singleton."""
    get_event_bus().emit(event_type, source_module, data, priority)


def subscribe_event(event_type: EventType, callback: Callable):
    """Convenience function to subscribe using singleton."""
    get_event_bus().subscribe(event_type, callback)


if __name__ == "__main__":
    # Test event bus
    print("📢 RF Arsenal OS - Event Bus Test\n")
    
    bus = EventBus()
    
    # Define callback
    def on_device_detected(event: Event):
        print(f"✅ Device detected callback: {event.data}")
    
    def on_any_event(event: Event):
        print(f"📢 Any event: {event.event_type.value}")
    
    # Subscribe
    bus.subscribe(EventType.DEVICE_DETECTED, on_device_detected)
    bus.subscribe_all(on_any_event)
    
    # Emit events
    print("Emitting device detected event...")
    bus.emit(
        EventType.DEVICE_DETECTED,
        source_module="device_fingerprinting",
        data={'imsi': 'abc123', 'manufacturer': 'Apple'},
        priority="high"
    )
    
    print("\nEmitting anomaly detected event...")
    bus.emit(
        EventType.ANOMALY_DETECTED,
        source_module="anomaly_detector",
        data={'type': 'rogue_base_station'},
        priority="critical"
    )
    
    print(f"\n📊 Statistics:")
    import json
    print(json.dumps(bus.get_statistics(), indent=2))
