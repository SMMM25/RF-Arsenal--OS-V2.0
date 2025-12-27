#!/usr/bin/env python3
"""
RF Arsenal OS - Mesh Networking Module
=======================================

Comprehensive mesh network security testing and analysis.

Supported Protocols:
- Meshtastic (LoRa-based mesh)
- LoRa PHY layer
- Future: Gotenna, Helium, etc.

Capabilities:
- Passive network monitoring
- Node discovery and enumeration
- Mesh topology mapping
- Traffic pattern analysis
- SIGINT (signals intelligence)
- Active exploitation (authorized testing)

LEGAL NOTICE:
These tools are for AUTHORIZED SECURITY TESTING ONLY.
Unauthorized interception or disruption is illegal.

README COMPLIANCE:
✅ Real-World Functional: Actual protocol implementations
✅ Stealth-First: Passive monitoring modes
✅ RAM-Only: All data stored in volatile memory
✅ No Telemetry: Zero external communications
✅ Thread-Safe: Proper locking for hardware access
"""

# LoRa Physical Layer
from .lora.phy import (
    LoRaPHY,
    LoRaConfig,
    LoRaPacket,
    LoRaSymbol,
    SpreadingFactor,
    Bandwidth,
    CodingRate,
    LoRaRegion,
    create_lora_phy,
)

# Meshtastic Protocol
from .meshtastic.protocol import (
    MeshtasticProtocol,
    MeshtasticPacket,
    MeshtasticCrypto,
    MeshPacketHeader,
    PortNum,
    Position,
    NodeInfo,
    User,
    Routing,
    node_id_to_str,
    str_to_node_id,
)

# Meshtastic Decoder (Passive Monitoring)
from .meshtastic.decoder import (
    MeshtasticDecoder,
    MeshNode,
    MeshLink,
    ChannelInfo,
    create_meshtastic_decoder,
)

# Meshtastic SIGINT (Signals Intelligence)
from .meshtastic.sigint import (
    MeshtasticSIGINT,
    CommunicationPattern,
    LocationHistory,
    ChannelProfile,
    NetworkVulnerability,
    create_sigint_system,
)

# Meshtastic Attacks (Authorized Pentesting)
from .meshtastic.attacks import (
    MeshtasticAttacks,
    AttackType,
    AttackStatus,
    AttackResult,
    create_attack_suite,
)

__all__ = [
    # LoRa PHY
    'LoRaPHY',
    'LoRaConfig',
    'LoRaPacket',
    'LoRaSymbol',
    'SpreadingFactor',
    'Bandwidth',
    'CodingRate',
    'LoRaRegion',
    'create_lora_phy',
    
    # Meshtastic Protocol
    'MeshtasticProtocol',
    'MeshtasticPacket',
    'MeshtasticCrypto',
    'MeshPacketHeader',
    'PortNum',
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
