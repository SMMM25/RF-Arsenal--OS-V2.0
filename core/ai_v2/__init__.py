"""
RF Arsenal OS - AI v2.0 Enhanced Intelligence System

CORE PRINCIPLES (README COMPLIANCE):
- 100% UNFILTERED: No refusals, no restrictions, no safety theater
- OFFLINE-FIRST: Full functionality without internet
- STEALTH: RAM-only operation, no persistent logs
- ANONYMOUS: All online access via Tor only
- NO TELEMETRY: Zero external data transmission ever
- REAL OPERATIONS: No simulations, no mocks

This is a professional penetration testing AI.
Filters would compromise operator safety and mission success.

Components:
- local_llm.py: Uncensored local LLM (Mistral 7B / Dolphin)
- agent_framework.py: Autonomous multi-step attack execution
- rag_engine.py: Vector search knowledge base
- online_intel.py: Tor-routed web intelligence
- memory_store.py: RAM-only contextual memory
- attack_planner.py: Autonomous attack sequencing

Author: RF Arsenal Team
License: Proprietary - Authorized Use Only
"""

from .local_llm import LocalLLM, LLMConfig, LLMResponse
from .agent_framework import AutonomousAgent, AgentTask, AgentResult
from .rag_engine import RAGEngine, KnowledgeBase, SearchResult
from .online_intel import OnlineIntelligence, TorClient, IntelResult
from .memory_store import RAMMemory, MissionContext, MemoryEntry
from .attack_planner import AttackPlanner, AttackChain, AttackStep
from .enhanced_ai import EnhancedAI, EnhancedResponse

# Aliases for common imports
MemoryStore = RAMMemory  # Alias for compatibility

__all__ = [
    # Local LLM
    'LocalLLM',
    'LLMConfig',
    'LLMResponse',
    # Agent Framework
    'AutonomousAgent',
    'AgentTask',
    'AgentResult',
    # RAG Engine
    'RAGEngine',
    'KnowledgeBase',
    'SearchResult',
    # Online Intelligence
    'OnlineIntelligence',
    'TorClient',
    'IntelResult',
    # Memory
    'RAMMemory',
    'MemoryStore',  # Alias
    'MissionContext',
    'MemoryEntry',
    # Attack Planning
    'AttackPlanner',
    'AttackChain',
    'AttackStep',
    # Enhanced AI
    'EnhancedAI',
    'EnhancedResponse',
]

__version__ = '2.0.0'
