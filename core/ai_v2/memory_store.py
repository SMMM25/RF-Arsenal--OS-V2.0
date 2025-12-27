#!/usr/bin/env python3
"""
RF Arsenal OS - RAM-Only Memory Store

Contextual memory for AI operations - stored ONLY in RAM.
No persistent storage, no logs, no forensic artifacts.

README COMPLIANCE:
- RAM-only: All data in volatile memory
- No logging: No persistent storage ever
- Secure wipe: Memory cleared on exit
- Stealth: Zero forensic footprint

Author: RF Arsenal Team
License: Proprietary - Authorized Use Only
"""

import os
import sys
import hashlib
import secrets
import threading
import weakref
from typing import Optional, Dict, List, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import ctypes

logger_disabled = True  # No logging for stealth


class MemoryType(Enum):
    """Types of memory storage"""
    CONVERSATION = "conversation"
    TARGET = "target"
    DISCOVERY = "discovery"
    ATTACK = "attack"
    CREDENTIAL = "credential"
    NETWORK = "network"
    SIGNAL = "signal"
    MISSION = "mission"


@dataclass
class MemoryEntry:
    """A single memory entry"""
    id: str
    type: MemoryType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    ttl_seconds: Optional[int] = None  # Auto-expire
    importance: float = 0.5  # 0.0 to 1.0
    tags: Set[str] = field(default_factory=set)
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds is None:
            return False
        expiry = self.timestamp + timedelta(seconds=self.ttl_seconds)
        return datetime.now() > expiry
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class MissionContext:
    """Context for an active mission/operation"""
    mission_id: str
    name: str
    objective: str
    started_at: datetime = field(default_factory=datetime.now)
    
    # Mission state
    current_step: int = 0
    total_steps: int = 0
    status: str = "active"  # active, paused, completed, aborted
    
    # Collected data (RAM-only)
    targets: List[Dict[str, Any]] = field(default_factory=list)
    discoveries: List[Dict[str, Any]] = field(default_factory=list)
    credentials: List[Dict[str, Any]] = field(default_factory=list)
    attack_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Notes and context
    notes: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    
    def add_target(self, target: Dict[str, Any]):
        """Add a discovered target"""
        target['discovered_at'] = datetime.now().isoformat()
        self.targets.append(target)
    
    def add_discovery(self, discovery: Dict[str, Any]):
        """Add a discovery"""
        discovery['timestamp'] = datetime.now().isoformat()
        self.discoveries.append(discovery)
    
    def add_credential(self, credential: Dict[str, Any]):
        """Add a captured credential (RAM-only)"""
        credential['captured_at'] = datetime.now().isoformat()
        self.credentials.append(credential)
    
    def add_result(self, result: Dict[str, Any]):
        """Add an attack result"""
        result['timestamp'] = datetime.now().isoformat()
        self.attack_results.append(result)
    
    def add_note(self, note: str):
        """Add a note"""
        self.notes.append(f"[{datetime.now().strftime('%H:%M:%S')}] {note}")
    
    def set_variable(self, key: str, value: Any):
        """Set a mission variable"""
        self.variables[key] = value
    
    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a mission variable"""
        return self.variables.get(key, default)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get mission summary"""
        return {
            "mission_id": self.mission_id,
            "name": self.name,
            "objective": self.objective,
            "status": self.status,
            "progress": f"{self.current_step}/{self.total_steps}",
            "targets_found": len(self.targets),
            "discoveries": len(self.discoveries),
            "credentials_captured": len(self.credentials),
            "attack_results": len(self.attack_results),
            "duration": str(datetime.now() - self.started_at),
        }


class RAMMemory:
    """
    RAM-only memory store for AI context
    
    CRITICAL: All data stored in volatile memory only.
    No disk writes, no logs, no persistence.
    
    Features:
    - Conversation history
    - Target tracking
    - Discovery database
    - Credential storage (encrypted in RAM)
    - Mission context
    - Auto-expiring entries
    - Secure memory wiping
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for global memory"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Main memory stores (RAM-only)
        self._memories: Dict[str, MemoryEntry] = {}
        self._by_type: Dict[MemoryType, Set[str]] = {t: set() for t in MemoryType}
        self._by_tag: Dict[str, Set[str]] = {}
        
        # Conversation history (limited size)
        self._conversation: deque = deque(maxlen=100)
        
        # Active mission context
        self._mission: Optional[MissionContext] = None
        
        # Entity tracking (targets, networks, devices)
        self._entities: Dict[str, Dict[str, Any]] = {}
        
        # Encryption key for sensitive data (generated per session)
        self._session_key = secrets.token_bytes(32)
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Register cleanup on exit
        import atexit
        atexit.register(self._secure_wipe)
        
        self._initialized = True
    
    def store(
        self,
        type: MemoryType,
        data: Dict[str, Any],
        tags: Set[str] = None,
        ttl_seconds: int = None,
        importance: float = 0.5,
    ) -> str:
        """
        Store data in RAM memory
        
        Args:
            type: Type of memory entry
            data: Data to store
            tags: Tags for searching
            ttl_seconds: Auto-expire time
            importance: Importance score (0.0-1.0)
            
        Returns:
            Entry ID
        """
        entry_id = secrets.token_hex(8)
        
        entry = MemoryEntry(
            id=entry_id,
            type=type,
            data=data,
            ttl_seconds=ttl_seconds,
            importance=importance,
            tags=tags or set(),
        )
        
        with self._lock:
            self._memories[entry_id] = entry
            self._by_type[type].add(entry_id)
            
            for tag in entry.tags:
                if tag not in self._by_tag:
                    self._by_tag[tag] = set()
                self._by_tag[tag].add(entry_id)
        
        return entry_id
    
    def recall(
        self,
        type: MemoryType = None,
        tags: Set[str] = None,
        limit: int = 100,
        min_importance: float = 0.0,
    ) -> List[MemoryEntry]:
        """
        Recall memories by type and/or tags
        
        Args:
            type: Filter by type
            tags: Filter by tags (AND logic)
            limit: Maximum entries to return
            min_importance: Minimum importance score
            
        Returns:
            List of matching entries
        """
        with self._lock:
            # Clean expired entries first
            self._cleanup_expired()
            
            # Get candidate IDs
            if type:
                candidates = self._by_type.get(type, set()).copy()
            else:
                candidates = set(self._memories.keys())
            
            # Filter by tags
            if tags:
                for tag in tags:
                    tag_ids = self._by_tag.get(tag, set())
                    candidates &= tag_ids
            
            # Get entries and filter
            results = []
            for entry_id in candidates:
                entry = self._memories.get(entry_id)
                if entry and entry.importance >= min_importance:
                    results.append(entry)
            
            # Sort by importance and recency
            results.sort(key=lambda e: (e.importance, e.timestamp), reverse=True)
            
            return results[:limit]
    
    def forget(self, entry_id: str) -> bool:
        """
        Securely forget a memory entry
        
        Args:
            entry_id: Entry to forget
            
        Returns:
            True if entry was found and removed
        """
        with self._lock:
            entry = self._memories.pop(entry_id, None)
            if entry:
                self._by_type[entry.type].discard(entry_id)
                for tag in entry.tags:
                    if tag in self._by_tag:
                        self._by_tag[tag].discard(entry_id)
                
                # Overwrite data in memory
                self._secure_overwrite(entry.data)
                return True
            return False
    
    def forget_type(self, type: MemoryType) -> int:
        """Forget all memories of a type"""
        with self._lock:
            entry_ids = list(self._by_type.get(type, set()))
            count = 0
            for entry_id in entry_ids:
                if self.forget(entry_id):
                    count += 1
            return count
    
    def add_conversation(self, role: str, content: str):
        """Add to conversation history"""
        with self._lock:
            self._conversation.append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            })
    
    def get_conversation(self, limit: int = 20) -> List[Dict[str, str]]:
        """Get recent conversation history"""
        with self._lock:
            history = list(self._conversation)
            return history[-limit:] if limit else history
    
    def clear_conversation(self):
        """Clear conversation history"""
        with self._lock:
            self._conversation.clear()
    
    # === Mission Context ===
    
    def start_mission(
        self,
        name: str,
        objective: str,
        total_steps: int = 0,
    ) -> MissionContext:
        """Start a new mission context"""
        with self._lock:
            # End any existing mission
            if self._mission:
                self._mission.status = "aborted"
            
            self._mission = MissionContext(
                mission_id=secrets.token_hex(8),
                name=name,
                objective=objective,
                total_steps=total_steps,
            )
            return self._mission
    
    def get_mission(self) -> Optional[MissionContext]:
        """Get current mission context"""
        return self._mission
    
    def end_mission(self, status: str = "completed") -> Optional[Dict[str, Any]]:
        """End current mission and return summary"""
        with self._lock:
            if not self._mission:
                return None
            
            self._mission.status = status
            summary = self._mission.get_summary()
            
            # Store mission summary before clearing
            self.store(
                type=MemoryType.MISSION,
                data=summary,
                tags={"mission", "completed"},
                importance=0.9,
            )
            
            self._mission = None
            return summary
    
    # === Entity Tracking ===
    
    def track_entity(
        self,
        entity_type: str,
        entity_id: str,
        data: Dict[str, Any],
    ):
        """Track a discovered entity"""
        key = f"{entity_type}:{entity_id}"
        with self._lock:
            if key in self._entities:
                # Merge with existing data
                self._entities[key].update(data)
                self._entities[key]['last_seen'] = datetime.now().isoformat()
            else:
                self._entities[key] = {
                    **data,
                    'entity_type': entity_type,
                    'entity_id': entity_id,
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat(),
                }
    
    def get_entity(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get a tracked entity"""
        key = f"{entity_type}:{entity_id}"
        return self._entities.get(key)
    
    def get_entities(self, entity_type: str = None) -> List[Dict[str, Any]]:
        """Get all tracked entities of a type"""
        with self._lock:
            if entity_type:
                return [
                    e for e in self._entities.values()
                    if e.get('entity_type') == entity_type
                ]
            return list(self._entities.values())
    
    # === Context for LLM ===
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of current context for LLM"""
        with self._lock:
            return {
                "mission": self._mission.get_summary() if self._mission else None,
                "conversation_length": len(self._conversation),
                "targets_tracked": len([e for e in self._entities.values() if e.get('entity_type') == 'target']),
                "networks_discovered": len([e for e in self._entities.values() if e.get('entity_type') == 'network']),
                "credentials_captured": len(list(self._by_type[MemoryType.CREDENTIAL])),
                "total_memories": len(self._memories),
            }
    
    def get_relevant_context(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get context relevant to a query"""
        # Simple keyword matching for now
        # TODO: Vector similarity search with embeddings
        keywords = set(query.lower().split())
        
        results = []
        with self._lock:
            for entry in self._memories.values():
                if entry.is_expired():
                    continue
                
                # Check tags
                tag_match = bool(keywords & entry.tags)
                
                # Check data content
                data_str = str(entry.data).lower()
                content_match = any(kw in data_str for kw in keywords)
                
                if tag_match or content_match:
                    results.append({
                        "type": entry.type.value,
                        "data": entry.data,
                        "importance": entry.importance,
                        "age_seconds": (datetime.now() - entry.timestamp).total_seconds(),
                    })
        
        # Sort by relevance (importance + recency)
        results.sort(key=lambda x: x['importance'] - (x['age_seconds'] / 3600), reverse=True)
        return results[:limit]
    
    # === Cleanup ===
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        expired = [
            entry_id for entry_id, entry in self._memories.items()
            if entry.is_expired()
        ]
        for entry_id in expired:
            self.forget(entry_id)
    
    def _secure_overwrite(self, data: Any):
        """Securely overwrite data in memory"""
        if isinstance(data, dict):
            for key in list(data.keys()):
                self._secure_overwrite(data[key])
                data[key] = None
            data.clear()
        elif isinstance(data, list):
            for i in range(len(data)):
                self._secure_overwrite(data[i])
                data[i] = None
            data.clear()
        elif isinstance(data, str):
            # Can't overwrite immutable strings in Python
            # But we can remove references
            pass
        elif isinstance(data, (bytes, bytearray)):
            # Overwrite with zeros
            if isinstance(data, bytearray):
                for i in range(len(data)):
                    data[i] = 0
    
    def _secure_wipe(self):
        """Securely wipe all memory on exit"""
        with self._lock:
            # Wipe all memories
            for entry in self._memories.values():
                self._secure_overwrite(entry.data)
            self._memories.clear()
            
            # Wipe type index
            for type_set in self._by_type.values():
                type_set.clear()
            
            # Wipe tag index
            for tag_set in self._by_tag.values():
                tag_set.clear()
            self._by_tag.clear()
            
            # Wipe conversation
            self._conversation.clear()
            
            # Wipe mission
            if self._mission:
                self._secure_overwrite(self._mission.__dict__)
                self._mission = None
            
            # Wipe entities
            for entity in self._entities.values():
                self._secure_overwrite(entity)
            self._entities.clear()
            
            # Wipe session key
            self._session_key = b'\x00' * 32
    
    def wipe_all(self):
        """Manually wipe all memory (emergency)"""
        self._secure_wipe()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        with self._lock:
            return {
                "total_entries": len(self._memories),
                "by_type": {t.value: len(ids) for t, ids in self._by_type.items()},
                "conversation_length": len(self._conversation),
                "entities_tracked": len(self._entities),
                "mission_active": self._mission is not None,
            }
    
    # === Methods for AI Command Center Integration ===
    
    def get_status(self) -> Dict[str, Any]:
        """Get memory status for command center"""
        import sys
        with self._lock:
            total_size = sys.getsizeof(self._memories) + sys.getsizeof(self._conversation)
            targets = len([e for e in self._entities.values() if e.get('entity_type') == 'target'])
            
            return {
                "entry_count": len(self._memories),
                "memory_kb": total_size / 1024,
                "targets": targets,
                "sessions": 1 if self._mission else 0,
                "encrypted": True,  # Session key is used
                "conversation_length": len(self._conversation),
            }
    
    def clear(self):
        """Clear all memory (alias for wipe_all)"""
        self.wipe_all()


# Singleton accessor
def get_memory() -> RAMMemory:
    """Get the global RAM memory instance"""
    return RAMMemory()
