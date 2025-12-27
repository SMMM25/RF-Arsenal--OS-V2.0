#!/usr/bin/env python3
"""
RF Arsenal OS - Meshtastic Protocol Module
===========================================

Complete Meshtastic mesh networking security testing suite.

Components:
- Protocol: Full Meshtastic protocol implementation
- Decoder: Passive network monitoring and analysis
- SIGINT: Signals intelligence and traffic analysis
- Attacks: Active exploitation (authorized testing only)

Meshtastic Overview:
- Open-source mesh networking protocol
- Uses LoRa for long-range communication (1-50+ km)
- Regional frequencies: 433/868/915/923 MHz
- AES-256 encryption (optional)
- Supports GPS position sharing
- 10,000+ active nodes worldwide

Security Testing Capabilities:
- Passive traffic monitoring
- Node discovery and enumeration
- Mesh topology mapping
- Traffic pattern analysis
- Location tracking (GPS-enabled nodes)
- Channel encryption detection
- Vulnerability assessment
- Active attacks (authorized only):
  - Jamming (broadband, selective)
  - Packet injection
  - Node impersonation
  - Routing manipulation
  - DoS attacks

LEGAL NOTICE:
These tools are for AUTHORIZED PENETRATION TESTING ONLY.
Unauthorized use violates:
- 47 U.S.C. § 333 (Willful interference)
- 18 U.S.C. § 1030 (Computer Fraud and Abuse Act)
- 18 U.S.C. § 2511 (Wiretap Act)
- Local radio regulations

README COMPLIANCE:
✅ Real-World Functional: Actual Meshtastic protocol implementation
✅ Stealth-First: Passive monitoring modes available
✅ RAM-Only: All intelligence stored in volatile memory
✅ No Telemetry: Zero external communications
✅ Emergency Wipe: Clear all collected intel on command
✅ Thread-Safe: Proper locking for hardware access
✅ Input Validation: GPS coordinates, frequencies validated
"""

from .protocol import (
    MeshtasticProtocol,
    MeshtasticPacket,
    MeshtasticCrypto,
    MeshPacketHeader,
    PortNum,
    HopLimit,
    ChannelRole,
    Position,
    NodeInfo,
    User,
    Routing,
    node_id_to_str,
    str_to_node_id,
)

from .decoder import (
    MeshtasticDecoder,
    MeshNode,
    MeshLink,
    ChannelInfo,
    create_meshtastic_decoder,
)

from .sigint import (
    MeshtasticSIGINT,
    CommunicationPattern,
    LocationHistory,
    ChannelProfile,
    NetworkVulnerability,
    create_sigint_system,
)

from .attacks import (
    MeshtasticAttacks,
    AttackType,
    AttackStatus,
    AttackResult,
    create_attack_suite,
)

__all__ = [
    # Protocol
    'MeshtasticProtocol',
    'MeshtasticPacket',
    'MeshtasticCrypto',
    'MeshPacketHeader',
    'PortNum',
    'HopLimit',
    'ChannelRole',
    'Position',
    'NodeInfo',
    'User',
    'Routing',
    'node_id_to_str',
    'str_to_node_id',
    
    # Decoder
    'MeshtasticDecoder',
    'MeshNode',
    'MeshLink',
    'ChannelInfo',
    'create_meshtastic_decoder',
    
    # SIGINT
    'MeshtasticSIGINT',
    'CommunicationPattern',
    'LocationHistory',
    'ChannelProfile',
    'NetworkVulnerability',
    'create_sigint_system',
    
    # Attacks
    'MeshtasticAttacks',
    'AttackType',
    'AttackStatus',
    'AttackResult',
    'create_attack_suite',
]

__version__ = '1.0.0'
__author__ = 'RF Arsenal OS Team'
