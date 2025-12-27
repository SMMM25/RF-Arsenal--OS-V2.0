"""
RF Arsenal OS - Session Recording System
Complete session recording and playback for RF operations.
Supports RAM-only mode for stealth operations.
"""

import time
import json
import gzip
import hashlib
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import io


class EventType(Enum):
    """Types of recordable events"""
    COMMAND = "command"
    SIGNAL_CAPTURE = "signal_capture"
    HARDWARE_ACTION = "hardware_action"
    FREQUENCY_CHANGE = "frequency_change"
    GAIN_CHANGE = "gain_change"
    DETECTION = "detection"
    TRANSMISSION = "transmission"
    ALERT = "alert"
    NOTE = "note"
    MARKER = "marker"
    SCREENSHOT = "screenshot"
    IQ_CAPTURE = "iq_capture"
    DECODE = "decode"
    ATTACK = "attack"
    MEASUREMENT = "measurement"


@dataclass
class SessionEvent:
    """Single recorded event"""
    event_id: str
    event_type: EventType
    timestamp: float
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, d: Dict) -> 'SessionEvent':
        return cls(
            event_id=d["event_id"],
            event_type=EventType(d["event_type"]),
            timestamp=d["timestamp"],
            data=d["data"],
            metadata=d.get("metadata", {})
        )


@dataclass
class SessionMetadata:
    """Session metadata"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    operator: str = "anonymous"
    mission_name: str = ""
    hardware: List[str] = field(default_factory=list)
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    checksum: str = ""


class SessionRecorder:
    """
    Production-grade session recorder for RF operations.
    
    Features:
    - Complete operation logging
    - IQ capture references
    - Command history
    - Hardware state tracking
    - Signal detections
    - Timestamped markers and notes
    - Compressed export
    - RAM-only mode for stealth
    - Secure deletion
    - Playback support
    """
    
    def __init__(self,
                 session_id: Optional[str] = None,
                 ram_only: bool = False,
                 max_events: int = 100000):
        """
        Initialize session recorder.
        
        Args:
            session_id: Unique session identifier (auto-generated if None)
            ram_only: Keep all data in RAM only (no disk writes)
            max_events: Maximum events to store
        """
        self.session_id = session_id or self._generate_session_id()
        self.ram_only = ram_only
        self.max_events = max_events
        
        # Event storage
        self._events: deque = deque(maxlen=max_events)
        self._event_index: Dict[str, SessionEvent] = {}
        
        # Session state
        self._metadata = SessionMetadata(
            session_id=self.session_id,
            start_time=time.time()
        )
        self._recording = False
        self._event_counter = 0
        
        # Hardware state tracking
        self._hardware_state: Dict[str, Dict] = {}
        
        # IQ capture references (in RAM)
        self._iq_captures: Dict[str, bytes] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Callbacks for real-time event streaming
        self._event_callbacks: List[Callable] = []
        
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = str(time.time())
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]
        
    def start_recording(self, operator: str = "", mission_name: str = "") -> None:
        """Start recording session"""
        with self._lock:
            self._recording = True
            self._metadata.operator = operator
            self._metadata.mission_name = mission_name
            self._metadata.start_time = time.time()
            
            # Record start event
            self.record_event(
                EventType.MARKER,
                {"action": "session_start"},
                {"operator": operator, "mission": mission_name}
            )
            
    def stop_recording(self) -> SessionMetadata:
        """Stop recording and return metadata"""
        with self._lock:
            self._recording = False
            self._metadata.end_time = time.time()
            
            # Calculate checksum
            self._metadata.checksum = self._calculate_checksum()
            
            # Record stop event
            self.record_event(
                EventType.MARKER,
                {"action": "session_stop"},
                {"duration_s": self._metadata.end_time - self._metadata.start_time}
            )
            
            return self._metadata
            
    def record_event(self, event_type: EventType, 
                    data: Dict[str, Any],
                    metadata: Optional[Dict] = None) -> str:
        """
        Record an event.
        
        Args:
            event_type: Type of event
            data: Event data
            metadata: Optional metadata
            
        Returns:
            Event ID
        """
        with self._lock:
            self._event_counter += 1
            event_id = f"{self.session_id}_{self._event_counter:08d}"
            
            event = SessionEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=time.time(),
                data=data,
                metadata=metadata or {}
            )
            
            self._events.append(event)
            self._event_index[event_id] = event
            
            # Notify callbacks
            for callback in self._event_callbacks:
                try:
                    callback(event)
                except Exception:
                    pass
                    
            return event_id
            
    def record_command(self, command: str, result: Optional[str] = None) -> str:
        """Record a command execution"""
        return self.record_event(
            EventType.COMMAND,
            {"command": command, "result": result}
        )
        
    def record_signal_capture(self, frequency_hz: float, 
                             bandwidth_hz: float,
                             power_dbm: float,
                             signal_type: str = "unknown",
                             iq_reference: Optional[str] = None) -> str:
        """Record a signal capture"""
        return self.record_event(
            EventType.SIGNAL_CAPTURE,
            {
                "frequency_hz": frequency_hz,
                "bandwidth_hz": bandwidth_hz,
                "power_dbm": power_dbm,
                "signal_type": signal_type,
                "iq_reference": iq_reference
            }
        )
        
    def record_detection(self, detection_type: str,
                        details: Dict[str, Any]) -> str:
        """Record a detection (IMSI, device, etc.)"""
        return self.record_event(
            EventType.DETECTION,
            {"detection_type": detection_type, "details": details}
        )
        
    def record_hardware_action(self, hardware_id: str,
                              action: str,
                              parameters: Dict[str, Any]) -> str:
        """Record hardware action"""
        return self.record_event(
            EventType.HARDWARE_ACTION,
            {
                "hardware_id": hardware_id,
                "action": action,
                "parameters": parameters
            }
        )
        
    def record_iq_capture(self, capture_id: str,
                         iq_data: bytes,
                         metadata: Dict[str, Any]) -> str:
        """
        Record IQ capture data.
        
        Args:
            capture_id: Unique capture ID
            iq_data: Raw IQ bytes (stored in RAM if ram_only mode)
            metadata: Capture metadata (sample rate, frequency, etc.)
            
        Returns:
            Event ID
        """
        # Store IQ data in RAM
        self._iq_captures[capture_id] = iq_data
        
        return self.record_event(
            EventType.IQ_CAPTURE,
            {
                "capture_id": capture_id,
                "size_bytes": len(iq_data),
                **metadata
            }
        )
        
    def add_note(self, note: str) -> str:
        """Add timestamped note"""
        return self.record_event(EventType.NOTE, {"note": note})
        
    def add_marker(self, marker_name: str, marker_data: Optional[Dict] = None) -> str:
        """Add named marker"""
        return self.record_event(
            EventType.MARKER,
            {"marker_name": marker_name, "data": marker_data or {}}
        )
        
    def get_event(self, event_id: str) -> Optional[SessionEvent]:
        """Get event by ID"""
        return self._event_index.get(event_id)
        
    def get_events(self, 
                  event_type: Optional[EventType] = None,
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  limit: int = 1000) -> List[SessionEvent]:
        """
        Query events with filters.
        
        Args:
            event_type: Filter by event type
            start_time: Filter events after this time
            end_time: Filter events before this time
            limit: Maximum events to return
            
        Returns:
            List of matching events
        """
        results = []
        
        for event in self._events:
            if event_type and event.event_type != event_type:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
                
            results.append(event)
            
            if len(results) >= limit:
                break
                
        return results
        
    def get_iq_capture(self, capture_id: str) -> Optional[bytes]:
        """Get stored IQ capture data"""
        return self._iq_captures.get(capture_id)
        
    def get_command_history(self, limit: int = 100) -> List[Dict]:
        """Get command history"""
        commands = self.get_events(event_type=EventType.COMMAND, limit=limit)
        return [
            {
                "timestamp": e.timestamp,
                "command": e.data.get("command"),
                "result": e.data.get("result")
            }
            for e in commands
        ]
        
    def get_detections(self) -> List[Dict]:
        """Get all detections"""
        detections = self.get_events(event_type=EventType.DETECTION)
        return [e.to_dict() for e in detections]
        
    def _calculate_checksum(self) -> str:
        """Calculate checksum of all events"""
        event_data = json.dumps([e.to_dict() for e in self._events], sort_keys=True)
        return hashlib.sha256(event_data.encode()).hexdigest()
        
    def export_session(self, include_iq: bool = False) -> bytes:
        """
        Export session to compressed bytes.
        
        Args:
            include_iq: Include IQ capture data (can be large)
            
        Returns:
            Compressed session data
        """
        with self._lock:
            session_data = {
                "metadata": {
                    "session_id": self._metadata.session_id,
                    "start_time": self._metadata.start_time,
                    "end_time": self._metadata.end_time,
                    "operator": self._metadata.operator,
                    "mission_name": self._metadata.mission_name,
                    "hardware": self._metadata.hardware,
                    "notes": self._metadata.notes,
                    "tags": self._metadata.tags,
                    "checksum": self._calculate_checksum()
                },
                "events": [e.to_dict() for e in self._events],
                "version": "1.0"
            }
            
            if include_iq:
                # Include IQ captures as base64
                import base64
                session_data["iq_captures"] = {
                    k: base64.b64encode(v).decode()
                    for k, v in self._iq_captures.items()
                }
                
            # Compress
            json_data = json.dumps(session_data, indent=2)
            compressed = gzip.compress(json_data.encode())
            
            return compressed
            
    def save_to_file(self, filepath: str, include_iq: bool = False) -> bool:
        """
        Save session to file (not available in RAM-only mode).
        
        Args:
            filepath: Output file path
            include_iq: Include IQ data
            
        Returns:
            Success status
        """
        if self.ram_only:
            return False
            
        try:
            data = self.export_session(include_iq)
            with open(filepath, 'wb') as f:
                f.write(data)
            return True
        except Exception:
            return False
            
    @classmethod
    def load_from_file(cls, filepath: str) -> 'SessionRecorder':
        """Load session from file"""
        with open(filepath, 'rb') as f:
            compressed = f.read()
            
        return cls.load_from_bytes(compressed)
        
    @classmethod
    def load_from_bytes(cls, data: bytes) -> 'SessionRecorder':
        """Load session from compressed bytes"""
        json_data = gzip.decompress(data)
        session_data = json.loads(json_data)
        
        # Create recorder
        recorder = cls(session_id=session_data["metadata"]["session_id"])
        
        # Load metadata
        recorder._metadata = SessionMetadata(
            session_id=session_data["metadata"]["session_id"],
            start_time=session_data["metadata"]["start_time"],
            end_time=session_data["metadata"].get("end_time"),
            operator=session_data["metadata"].get("operator", ""),
            mission_name=session_data["metadata"].get("mission_name", ""),
            hardware=session_data["metadata"].get("hardware", []),
            notes=session_data["metadata"].get("notes", ""),
            tags=session_data["metadata"].get("tags", []),
            checksum=session_data["metadata"].get("checksum", "")
        )
        
        # Load events
        for event_dict in session_data["events"]:
            event = SessionEvent.from_dict(event_dict)
            recorder._events.append(event)
            recorder._event_index[event.event_id] = event
            
        # Load IQ captures if present
        if "iq_captures" in session_data:
            import base64
            for k, v in session_data["iq_captures"].items():
                recorder._iq_captures[k] = base64.b64decode(v)
                
        return recorder
        
    def secure_delete(self) -> None:
        """Securely delete all session data from RAM"""
        with self._lock:
            # Overwrite event data
            for event in self._events:
                event.data = {}
                event.metadata = {}
                
            # Clear collections
            self._events.clear()
            self._event_index.clear()
            
            # Overwrite IQ captures
            for key in self._iq_captures:
                self._iq_captures[key] = b'\x00' * len(self._iq_captures[key])
            self._iq_captures.clear()
            
            # Clear metadata
            self._metadata = SessionMetadata(
                session_id="deleted",
                start_time=0
            )
            
    def register_callback(self, callback: Callable) -> None:
        """Register callback for real-time event streaming"""
        self._event_callbacks.append(callback)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get session statistics"""
        event_counts = {}
        for event in self._events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
        duration = 0
        if self._metadata.end_time:
            duration = self._metadata.end_time - self._metadata.start_time
        elif self._events:
            duration = time.time() - self._metadata.start_time
            
        return {
            "session_id": self.session_id,
            "total_events": len(self._events),
            "event_counts": event_counts,
            "duration_s": duration,
            "iq_captures": len(self._iq_captures),
            "iq_size_bytes": sum(len(v) for v in self._iq_captures.values()),
            "recording": self._recording,
            "ram_only": self.ram_only
        }


class SessionPlayback:
    """
    Session playback engine for replaying recorded sessions.
    """
    
    def __init__(self, session: SessionRecorder):
        """
        Initialize playback engine.
        
        Args:
            session: Recorded session to play back
        """
        self.session = session
        self._events = list(session._events)
        self._current_index = 0
        self._playback_speed = 1.0
        self._playing = False
        self._paused = False
        
        # Callbacks for event playback
        self._event_handlers: Dict[EventType, List[Callable]] = {}
        
    def set_speed(self, speed: float) -> None:
        """Set playback speed multiplier"""
        self._playback_speed = max(0.1, min(10.0, speed))
        
    def register_handler(self, event_type: EventType, handler: Callable) -> None:
        """Register handler for event type during playback"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        
    def play(self) -> None:
        """Start or resume playback"""
        self._playing = True
        self._paused = False
        
        threading.Thread(target=self._playback_loop, daemon=True).start()
        
    def pause(self) -> None:
        """Pause playback"""
        self._paused = True
        
    def stop(self) -> None:
        """Stop playback"""
        self._playing = False
        self._current_index = 0
        
    def seek(self, timestamp: float) -> None:
        """Seek to specific timestamp"""
        for i, event in enumerate(self._events):
            if event.timestamp >= timestamp:
                self._current_index = i
                return
                
    def _playback_loop(self) -> None:
        """Main playback loop"""
        while self._playing and self._current_index < len(self._events):
            if self._paused:
                time.sleep(0.1)
                continue
                
            event = self._events[self._current_index]
            
            # Calculate delay to next event
            if self._current_index + 1 < len(self._events):
                next_event = self._events[self._current_index + 1]
                delay = (next_event.timestamp - event.timestamp) / self._playback_speed
                delay = max(0, min(delay, 10))  # Cap delay at 10 seconds
            else:
                delay = 0
                
            # Execute handlers
            if event.event_type in self._event_handlers:
                for handler in self._event_handlers[event.event_type]:
                    try:
                        handler(event)
                    except Exception:
                        pass
                        
            self._current_index += 1
            
            if delay > 0:
                time.sleep(delay)
                
    def get_status(self) -> Dict[str, Any]:
        """Get playback status"""
        return {
            "playing": self._playing,
            "paused": self._paused,
            "current_index": self._current_index,
            "total_events": len(self._events),
            "progress_percent": (self._current_index / len(self._events) * 100) if self._events else 0,
            "playback_speed": self._playback_speed
        }
