"""
RF Arsenal OS - Threat Intelligence Modules
============================================

Real-time threat intelligence and vulnerability tracking.

README COMPLIANCE:
✅ Stealth-First: No identifiable queries
✅ RAM-Only: All intel stored in memory
✅ No Telemetry: Zero outbound tracking
✅ Offline-First: Works with cached data
✅ Real-World Functional: Production threat intel
"""

from .threat_intelligence import (
    # Enums
    Severity,
    ExploitAvailability,
    ThreatActorType,
    IoCType,
    IntelSource,
    
    # Data structures
    CVE,
    Exploit,
    ThreatActor,
    IoC,
    ThreatFeed,
    
    # Components
    CVEDatabase,
    ExploitDatabase,
    ThreatActorTracker,
    IoCManager,
    DarkWebMonitor,
    
    # Main platform
    ThreatIntelligencePlatform,
)

# Alias for consistency
ThreatIntelPlatform = ThreatIntelligencePlatform

__all__ = [
    # Enums
    'Severity',
    'ExploitAvailability',
    'ThreatActorType',
    'IoCType',
    'IntelSource',
    
    # Data structures
    'CVE',
    'Exploit',
    'ThreatActor',
    'IoC',
    'ThreatFeed',
    
    # Components
    'CVEDatabase',
    'ExploitDatabase',
    'ThreatActorTracker',
    'IoCManager',
    'DarkWebMonitor',
    
    # Main platform
    'ThreatIntelligencePlatform',
    'ThreatIntelPlatform',  # Alias
]
