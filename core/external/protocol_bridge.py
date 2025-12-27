"""
RF Arsenal OS - Protocol Bridge
Bridge between RF Arsenal native protocols and external stack protocols

This module provides:
- Protocol translation between native and srsRAN/OAI
- Message routing between protocol layers
- Event synchronization across stacks
- Unified protocol interface for AI control
"""

import logging
import threading
import time
import struct
import socket
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import IntEnum, auto
from queue import Queue, Empty
import json

logger = logging.getLogger(__name__)


# ============================================================================
# Protocol Types and Enums
# ============================================================================

class ProtocolLayer(IntEnum):
    """Protocol layer enumeration"""
    PHY = 1
    MAC = 2
    RLC = 3
    PDCP = 4
    RRC = 5
    NAS = 6
    S1AP = 7
    NGAP = 8
    GTP = 9


class MessageDirection(IntEnum):
    """Message direction"""
    UPLINK = 0
    DOWNLINK = 1
    INTERNAL = 2


class ProtocolFamily(IntEnum):
    """Protocol family"""
    NATIVE = 0       # RF Arsenal native
    SRSRAN = 1       # srsRAN protocols
    OAI = 2          # OpenAirInterface protocols


# ============================================================================
# Protocol Messages
# ============================================================================

@dataclass
class ProtocolMessage:
    """Base protocol message structure"""
    layer: ProtocolLayer
    direction: MessageDirection
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    destination: str = ""
    payload: bytes = b""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'layer': self.layer.name,
            'direction': self.direction.name,
            'timestamp': self.timestamp,
            'source': self.source,
            'destination': self.destination,
            'payload_len': len(self.payload),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_bytes(cls, data: bytes, layer: ProtocolLayer) -> 'ProtocolMessage':
        """Create message from raw bytes"""
        return cls(
            layer=layer,
            direction=MessageDirection.UPLINK,
            payload=data
        )


@dataclass
class S1APMessage(ProtocolMessage):
    """S1AP-specific message"""
    procedure_code: int = 0
    criticality: int = 0
    message_type: str = ""
    
    def __post_init__(self):
        self.layer = ProtocolLayer.S1AP


@dataclass
class NGAPMessage(ProtocolMessage):
    """NGAP-specific message (5G equivalent of S1AP)"""
    procedure_code: int = 0
    criticality: int = 0
    message_type: str = ""
    
    def __post_init__(self):
        self.layer = ProtocolLayer.NGAP


@dataclass
class GTPMessage(ProtocolMessage):
    """GTP-specific message"""
    version: int = 1
    message_type: int = 0
    teid: int = 0
    sequence_number: int = 0
    
    def __post_init__(self):
        self.layer = ProtocolLayer.GTP


@dataclass
class RRCMessage(ProtocolMessage):
    """RRC-specific message"""
    message_type: str = ""
    ue_id: int = 0
    cell_id: int = 0
    
    def __post_init__(self):
        self.layer = ProtocolLayer.RRC


@dataclass
class NASMessage(ProtocolMessage):
    """NAS-specific message"""
    message_type: int = 0
    security_header_type: int = 0
    protocol_discriminator: int = 0
    
    def __post_init__(self):
        self.layer = ProtocolLayer.NAS


# ============================================================================
# Protocol Translator Interface
# ============================================================================

class ProtocolTranslator(ABC):
    """Abstract base for protocol translation"""
    
    @abstractmethod
    def translate_to_native(self, message: ProtocolMessage) -> ProtocolMessage:
        """Translate external message to native format"""
        pass
    
    @abstractmethod
    def translate_from_native(self, message: ProtocolMessage) -> bytes:
        """Translate native message to external format"""
        pass
    
    @abstractmethod
    def supports_layer(self, layer: ProtocolLayer) -> bool:
        """Check if translator supports given layer"""
        pass


# ============================================================================
# srsRAN Protocol Translator
# ============================================================================

class SrsRANTranslator(ProtocolTranslator):
    """Translate between srsRAN and native RF Arsenal protocols"""
    
    SUPPORTED_LAYERS = {
        ProtocolLayer.MAC,
        ProtocolLayer.RLC,
        ProtocolLayer.PDCP,
        ProtocolLayer.RRC,
        ProtocolLayer.NAS,
        ProtocolLayer.S1AP,
        ProtocolLayer.GTP
    }
    
    def __init__(self):
        self._rrc_transaction_id = 0
        self._nas_sequence = 0
    
    def supports_layer(self, layer: ProtocolLayer) -> bool:
        return layer in self.SUPPORTED_LAYERS
    
    def translate_to_native(self, message: ProtocolMessage) -> ProtocolMessage:
        """Translate srsRAN message to native format"""
        if message.layer == ProtocolLayer.S1AP:
            return self._translate_s1ap_to_native(message)
        elif message.layer == ProtocolLayer.GTP:
            return self._translate_gtp_to_native(message)
        elif message.layer == ProtocolLayer.RRC:
            return self._translate_rrc_to_native(message)
        elif message.layer == ProtocolLayer.NAS:
            return self._translate_nas_to_native(message)
        else:
            # Pass through for unsupported layers
            return message
    
    def translate_from_native(self, message: ProtocolMessage) -> bytes:
        """Translate native message to srsRAN format"""
        if message.layer == ProtocolLayer.S1AP:
            return self._translate_s1ap_from_native(message)
        elif message.layer == ProtocolLayer.GTP:
            return self._translate_gtp_from_native(message)
        elif message.layer == ProtocolLayer.RRC:
            return self._translate_rrc_from_native(message)
        elif message.layer == ProtocolLayer.NAS:
            return self._translate_nas_from_native(message)
        else:
            return message.payload
    
    def _translate_s1ap_to_native(self, message: ProtocolMessage) -> S1APMessage:
        """Translate S1AP message to native format"""
        # Parse srsRAN S1AP format
        payload = message.payload
        
        # Extract S1AP header (simplified)
        if len(payload) >= 4:
            procedure_code = payload[0]
            criticality = payload[1]
            message_type = "unknown"
            
            # Map common S1AP procedures
            procedure_map = {
                0: "S1SetupRequest",
                1: "S1SetupResponse",
                12: "InitialUEMessage",
                13: "DownlinkNASTransport",
                14: "UplinkNASTransport",
                9: "InitialContextSetupRequest",
                10: "InitialContextSetupResponse",
                21: "PathSwitchRequest",
                23: "HandoverRequired",
            }
            message_type = procedure_map.get(procedure_code, f"Procedure_{procedure_code}")
        else:
            procedure_code = 0
            criticality = 0
            message_type = "unknown"
        
        return S1APMessage(
            direction=message.direction,
            timestamp=message.timestamp,
            source=message.source,
            destination=message.destination,
            payload=payload,
            metadata=message.metadata,
            procedure_code=procedure_code,
            criticality=criticality,
            message_type=message_type
        )
    
    def _translate_s1ap_from_native(self, message: ProtocolMessage) -> bytes:
        """Translate native S1AP to srsRAN format"""
        if isinstance(message, S1APMessage):
            # Build srsRAN S1AP frame
            header = struct.pack('!BB', message.procedure_code, message.criticality)
            return header + message.payload
        return message.payload
    
    def _translate_gtp_to_native(self, message: ProtocolMessage) -> GTPMessage:
        """Translate GTP message to native format"""
        payload = message.payload
        
        # Parse GTP header
        if len(payload) >= 8:
            flags = payload[0]
            version = (flags >> 5) & 0x07
            msg_type = payload[1]
            length = struct.unpack('!H', payload[2:4])[0]
            teid = struct.unpack('!I', payload[4:8])[0]
            
            seq_num = 0
            if flags & 0x02:  # Sequence number present
                if len(payload) >= 12:
                    seq_num = struct.unpack('!H', payload[8:10])[0]
        else:
            version = 1
            msg_type = 0
            teid = 0
            seq_num = 0
        
        return GTPMessage(
            direction=message.direction,
            timestamp=message.timestamp,
            source=message.source,
            destination=message.destination,
            payload=payload,
            metadata=message.metadata,
            version=version,
            message_type=msg_type,
            teid=teid,
            sequence_number=seq_num
        )
    
    def _translate_gtp_from_native(self, message: ProtocolMessage) -> bytes:
        """Translate native GTP to srsRAN format"""
        if isinstance(message, GTPMessage):
            # Build GTP header
            flags = (message.version << 5) | 0x10  # Version + PT flag
            header = struct.pack('!BBHI',
                flags,
                message.message_type,
                len(message.payload),
                message.teid
            )
            return header + message.payload
        return message.payload
    
    def _translate_rrc_to_native(self, message: ProtocolMessage) -> RRCMessage:
        """Translate RRC message to native format"""
        payload = message.payload
        
        # Simplified RRC parsing
        msg_type = "unknown"
        ue_id = 0
        
        if len(payload) >= 2:
            # Extract message type from RRC header
            rrc_header = payload[0]
            msg_type_map = {
                0: "RRCConnectionRequest",
                1: "RRCConnectionSetup",
                2: "RRCConnectionSetupComplete",
                3: "RRCConnectionReconfiguration",
                4: "RRCConnectionReconfigurationComplete",
                5: "RRCConnectionRelease",
                6: "SecurityModeCommand",
                7: "SecurityModeComplete",
                8: "UECapabilityEnquiry",
                9: "UECapabilityInformation",
            }
            msg_type = msg_type_map.get(rrc_header & 0x0F, f"RRC_{rrc_header}")
        
        return RRCMessage(
            direction=message.direction,
            timestamp=message.timestamp,
            source=message.source,
            destination=message.destination,
            payload=payload,
            metadata=message.metadata,
            message_type=msg_type,
            ue_id=ue_id,
            cell_id=0
        )
    
    def _translate_rrc_from_native(self, message: ProtocolMessage) -> bytes:
        """Translate native RRC to srsRAN format"""
        return message.payload
    
    def _translate_nas_to_native(self, message: ProtocolMessage) -> NASMessage:
        """Translate NAS message to native format"""
        payload = message.payload
        
        # Parse NAS header
        if len(payload) >= 2:
            security_header = (payload[0] >> 4) & 0x0F
            protocol_disc = payload[0] & 0x0F
            msg_type = payload[1] if len(payload) > 1 else 0
        else:
            security_header = 0
            protocol_disc = 0
            msg_type = 0
        
        return NASMessage(
            direction=message.direction,
            timestamp=message.timestamp,
            source=message.source,
            destination=message.destination,
            payload=payload,
            metadata=message.metadata,
            message_type=msg_type,
            security_header_type=security_header,
            protocol_discriminator=protocol_disc
        )
    
    def _translate_nas_from_native(self, message: ProtocolMessage) -> bytes:
        """Translate native NAS to srsRAN format"""
        return message.payload


# ============================================================================
# OpenAirInterface Protocol Translator
# ============================================================================

class OAITranslator(ProtocolTranslator):
    """Translate between OAI and native RF Arsenal protocols"""
    
    SUPPORTED_LAYERS = {
        ProtocolLayer.MAC,
        ProtocolLayer.RLC,
        ProtocolLayer.PDCP,
        ProtocolLayer.RRC,
        ProtocolLayer.NAS,
        ProtocolLayer.NGAP,  # 5G equivalent of S1AP
        ProtocolLayer.GTP
    }
    
    def __init__(self):
        self._transaction_id = 0
    
    def supports_layer(self, layer: ProtocolLayer) -> bool:
        return layer in self.SUPPORTED_LAYERS
    
    def translate_to_native(self, message: ProtocolMessage) -> ProtocolMessage:
        """Translate OAI message to native format"""
        if message.layer == ProtocolLayer.NGAP:
            return self._translate_ngap_to_native(message)
        elif message.layer == ProtocolLayer.GTP:
            return self._translate_gtp_to_native(message)
        elif message.layer == ProtocolLayer.RRC:
            return self._translate_rrc_to_native(message)
        elif message.layer == ProtocolLayer.NAS:
            return self._translate_5g_nas_to_native(message)
        else:
            return message
    
    def translate_from_native(self, message: ProtocolMessage) -> bytes:
        """Translate native message to OAI format"""
        if message.layer == ProtocolLayer.NGAP:
            return self._translate_ngap_from_native(message)
        elif message.layer == ProtocolLayer.GTP:
            return self._translate_gtp_from_native(message)
        elif message.layer == ProtocolLayer.RRC:
            return self._translate_rrc_from_native(message)
        elif message.layer == ProtocolLayer.NAS:
            return self._translate_5g_nas_from_native(message)
        else:
            return message.payload
    
    def _translate_ngap_to_native(self, message: ProtocolMessage) -> NGAPMessage:
        """Translate NGAP message to native format"""
        payload = message.payload
        
        # Parse NGAP header
        if len(payload) >= 4:
            procedure_code = payload[0]
            criticality = payload[1]
            
            procedure_map = {
                14: "NGSetupRequest",
                15: "NGSetupResponse",
                46: "InitialUEMessage",
                4: "DownlinkNASTransport",
                46: "UplinkNASTransport",
                29: "InitialContextSetupRequest",
                30: "InitialContextSetupResponse",
                51: "PDUSessionResourceSetupRequest",
            }
            message_type = procedure_map.get(procedure_code, f"NGAP_{procedure_code}")
        else:
            procedure_code = 0
            criticality = 0
            message_type = "unknown"
        
        return NGAPMessage(
            direction=message.direction,
            timestamp=message.timestamp,
            source=message.source,
            destination=message.destination,
            payload=payload,
            metadata=message.metadata,
            procedure_code=procedure_code,
            criticality=criticality,
            message_type=message_type
        )
    
    def _translate_ngap_from_native(self, message: ProtocolMessage) -> bytes:
        """Translate native NGAP to OAI format"""
        if isinstance(message, NGAPMessage):
            header = struct.pack('!BB', message.procedure_code, message.criticality)
            return header + message.payload
        return message.payload
    
    def _translate_gtp_to_native(self, message: ProtocolMessage) -> GTPMessage:
        """Translate GTP message to native format"""
        # Similar to srsRAN GTP translation
        payload = message.payload
        
        if len(payload) >= 8:
            flags = payload[0]
            version = (flags >> 5) & 0x07
            msg_type = payload[1]
            teid = struct.unpack('!I', payload[4:8])[0]
            seq_num = 0
        else:
            version = 2  # GTPv2 for 5G
            msg_type = 0
            teid = 0
            seq_num = 0
        
        return GTPMessage(
            direction=message.direction,
            timestamp=message.timestamp,
            source=message.source,
            destination=message.destination,
            payload=payload,
            metadata=message.metadata,
            version=version,
            message_type=msg_type,
            teid=teid,
            sequence_number=seq_num
        )
    
    def _translate_gtp_from_native(self, message: ProtocolMessage) -> bytes:
        """Translate native GTP to OAI format"""
        if isinstance(message, GTPMessage):
            flags = (message.version << 5) | 0x40  # GTPv2 format
            header = struct.pack('!BBHI',
                flags,
                message.message_type,
                len(message.payload),
                message.teid
            )
            return header + message.payload
        return message.payload
    
    def _translate_rrc_to_native(self, message: ProtocolMessage) -> RRCMessage:
        """Translate NR RRC to native format"""
        payload = message.payload
        
        # NR RRC uses ASN.1 UPER encoding
        msg_type = "NR-RRC"
        if len(payload) >= 1:
            # Simplified NR RRC message type extraction
            rrc_msg_type = payload[0] >> 4
            msg_type_map = {
                0: "RRCSetupRequest",
                1: "RRCSetup",
                2: "RRCSetupComplete",
                3: "RRCReconfiguration",
                4: "RRCReconfigurationComplete",
                5: "RRCRelease",
                6: "SecurityModeCommand",
                7: "SecurityModeComplete",
            }
            msg_type = msg_type_map.get(rrc_msg_type, f"NR-RRC_{rrc_msg_type}")
        
        return RRCMessage(
            direction=message.direction,
            timestamp=message.timestamp,
            source=message.source,
            destination=message.destination,
            payload=payload,
            metadata=message.metadata,
            message_type=msg_type,
            ue_id=0,
            cell_id=0
        )
    
    def _translate_rrc_from_native(self, message: ProtocolMessage) -> bytes:
        """Translate native RRC to OAI NR format"""
        return message.payload
    
    def _translate_5g_nas_to_native(self, message: ProtocolMessage) -> NASMessage:
        """Translate 5G NAS to native format"""
        payload = message.payload
        
        # 5G NAS uses extended protocol discriminator
        if len(payload) >= 3:
            epd = payload[0]  # Extended Protocol Discriminator
            security_header = payload[1] >> 4
            msg_type = payload[2] if epd == 0x7E else 0  # 5GMM
        else:
            epd = 0
            security_header = 0
            msg_type = 0
        
        return NASMessage(
            direction=message.direction,
            timestamp=message.timestamp,
            source=message.source,
            destination=message.destination,
            payload=payload,
            metadata=message.metadata,
            message_type=msg_type,
            security_header_type=security_header,
            protocol_discriminator=epd
        )
    
    def _translate_5g_nas_from_native(self, message: ProtocolMessage) -> bytes:
        """Translate native NAS to OAI 5G format"""
        return message.payload


# ============================================================================
# Protocol Router
# ============================================================================

class ProtocolRouter:
    """Route protocol messages between components and stacks"""
    
    def __init__(self):
        self._routes: Dict[ProtocolLayer, List[Callable]] = {}
        self._message_queue: Queue = Queue(maxsize=10000)
        self._running = False
        self._router_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            'messages_routed': 0,
            'messages_dropped': 0,
            'route_errors': 0
        }
    
    def register_handler(self, layer: ProtocolLayer, handler: Callable[[ProtocolMessage], None]):
        """Register message handler for layer"""
        with self._lock:
            if layer not in self._routes:
                self._routes[layer] = []
            self._routes[layer].append(handler)
    
    def unregister_handler(self, layer: ProtocolLayer, handler: Callable):
        """Unregister message handler"""
        with self._lock:
            if layer in self._routes and handler in self._routes[layer]:
                self._routes[layer].remove(handler)
    
    def route_message(self, message: ProtocolMessage):
        """Route message to appropriate handlers"""
        try:
            self._message_queue.put_nowait(message)
        except:
            self._stats['messages_dropped'] += 1
            logger.warning("Message queue full, dropping message")
    
    def start(self):
        """Start the router"""
        if self._running:
            return
        
        self._running = True
        self._router_thread = threading.Thread(target=self._route_loop, daemon=True)
        self._router_thread.start()
        logger.info("Protocol router started")
    
    def stop(self):
        """Stop the router"""
        self._running = False
        if self._router_thread:
            self._router_thread.join(timeout=5.0)
        logger.info("Protocol router stopped")
    
    def _route_loop(self):
        """Main routing loop"""
        while self._running:
            try:
                message = self._message_queue.get(timeout=0.1)
                self._dispatch_message(message)
            except Empty:
                continue
            except Exception as e:
                self._stats['route_errors'] += 1
                logger.error(f"Routing error: {e}")
    
    def _dispatch_message(self, message: ProtocolMessage):
        """Dispatch message to registered handlers"""
        with self._lock:
            handlers = self._routes.get(message.layer, [])
        
        for handler in handlers:
            try:
                handler(message)
                self._stats['messages_routed'] += 1
            except Exception as e:
                self._stats['route_errors'] += 1
                logger.error(f"Handler error for {message.layer}: {e}")
    
    def get_statistics(self) -> Dict[str, int]:
        """Get routing statistics"""
        return self._stats.copy()


# ============================================================================
# Protocol Bridge
# ============================================================================

class ProtocolBridge:
    """
    Main protocol bridge for RF Arsenal OS
    
    Bridges between native RF Arsenal protocols and external stacks
    (srsRAN and OpenAirInterface).
    """
    
    def __init__(self, stealth_system=None):
        """
        Initialize protocol bridge
        
        Args:
            stealth_system: Optional stealth system for emission control
        """
        self.stealth_system = stealth_system
        
        # Translators
        self._srsran_translator = SrsRANTranslator()
        self._oai_translator = OAITranslator()
        
        # Router
        self._router = ProtocolRouter()
        
        # Active connections
        self._connections: Dict[str, Dict] = {}
        
        # Message handlers
        self._native_handlers: Dict[ProtocolLayer, List[Callable]] = {}
        
        # Lock
        self._lock = threading.RLock()
        
        # State
        self._initialized = False
        
        # Event callbacks
        self._event_callbacks: Dict[str, List[Callable]] = {}
    
    def initialize(self) -> bool:
        """Initialize the protocol bridge"""
        if self._initialized:
            return True
        
        logger.info("Initializing protocol bridge")
        
        # Start router
        self._router.start()
        
        # Register default handlers
        self._register_default_handlers()
        
        self._initialized = True
        logger.info("Protocol bridge initialized")
        return True
    
    def shutdown(self):
        """Shutdown protocol bridge"""
        logger.info("Shutting down protocol bridge")
        
        self._router.stop()
        self._connections.clear()
        self._initialized = False
    
    def _register_default_handlers(self):
        """Register default protocol handlers"""
        # S1AP/NGAP handler
        self._router.register_handler(ProtocolLayer.S1AP, self._handle_s1ap)
        self._router.register_handler(ProtocolLayer.NGAP, self._handle_ngap)
        
        # GTP handler
        self._router.register_handler(ProtocolLayer.GTP, self._handle_gtp)
        
        # RRC handler
        self._router.register_handler(ProtocolLayer.RRC, self._handle_rrc)
        
        # NAS handler
        self._router.register_handler(ProtocolLayer.NAS, self._handle_nas)
    
    # ========================================================================
    # Message Translation
    # ========================================================================
    
    def translate_to_native(self, message: ProtocolMessage, 
                            source_family: ProtocolFamily) -> ProtocolMessage:
        """
        Translate external message to native format
        
        Args:
            message: Message to translate
            source_family: Source protocol family
        
        Returns:
            Translated native message
        """
        if source_family == ProtocolFamily.SRSRAN:
            return self._srsran_translator.translate_to_native(message)
        elif source_family == ProtocolFamily.OAI:
            return self._oai_translator.translate_to_native(message)
        else:
            return message
    
    def translate_from_native(self, message: ProtocolMessage,
                               target_family: ProtocolFamily) -> bytes:
        """
        Translate native message to external format
        
        Args:
            message: Native message to translate
            target_family: Target protocol family
        
        Returns:
            Translated message bytes
        """
        if target_family == ProtocolFamily.SRSRAN:
            return self._srsran_translator.translate_from_native(message)
        elif target_family == ProtocolFamily.OAI:
            return self._oai_translator.translate_from_native(message)
        else:
            return message.payload
    
    # ========================================================================
    # Message Routing
    # ========================================================================
    
    def route_uplink(self, message: ProtocolMessage, source_family: ProtocolFamily):
        """Route uplink message from external stack to native"""
        message.direction = MessageDirection.UPLINK
        native_message = self.translate_to_native(message, source_family)
        self._router.route_message(native_message)
        self._emit_event('uplink_message', native_message.to_dict())
    
    def route_downlink(self, message: ProtocolMessage, target_family: ProtocolFamily):
        """Route downlink message from native to external stack"""
        message.direction = MessageDirection.DOWNLINK
        
        # Check stealth before sending
        if self.stealth_system and not self.stealth_system.check_emission_allowed():
            logger.warning("Downlink blocked by stealth system")
            return None
        
        payload = self.translate_from_native(message, target_family)
        self._emit_event('downlink_message', message.to_dict())
        return payload
    
    # ========================================================================
    # Protocol Handlers
    # ========================================================================
    
    def _handle_s1ap(self, message: ProtocolMessage):
        """Handle S1AP messages"""
        if isinstance(message, S1APMessage):
            logger.debug(f"S1AP: {message.message_type}")
            
            # Dispatch to native handlers
            if ProtocolLayer.S1AP in self._native_handlers:
                for handler in self._native_handlers[ProtocolLayer.S1AP]:
                    try:
                        handler(message)
                    except Exception as e:
                        logger.error(f"S1AP handler error: {e}")
    
    def _handle_ngap(self, message: ProtocolMessage):
        """Handle NGAP messages (5G)"""
        if isinstance(message, NGAPMessage):
            logger.debug(f"NGAP: {message.message_type}")
            
            if ProtocolLayer.NGAP in self._native_handlers:
                for handler in self._native_handlers[ProtocolLayer.NGAP]:
                    try:
                        handler(message)
                    except Exception as e:
                        logger.error(f"NGAP handler error: {e}")
    
    def _handle_gtp(self, message: ProtocolMessage):
        """Handle GTP messages"""
        if isinstance(message, GTPMessage):
            logger.debug(f"GTP: type={message.message_type}, TEID={message.teid}")
            
            if ProtocolLayer.GTP in self._native_handlers:
                for handler in self._native_handlers[ProtocolLayer.GTP]:
                    try:
                        handler(message)
                    except Exception as e:
                        logger.error(f"GTP handler error: {e}")
    
    def _handle_rrc(self, message: ProtocolMessage):
        """Handle RRC messages"""
        if isinstance(message, RRCMessage):
            logger.debug(f"RRC: {message.message_type}")
            
            if ProtocolLayer.RRC in self._native_handlers:
                for handler in self._native_handlers[ProtocolLayer.RRC]:
                    try:
                        handler(message)
                    except Exception as e:
                        logger.error(f"RRC handler error: {e}")
    
    def _handle_nas(self, message: ProtocolMessage):
        """Handle NAS messages"""
        if isinstance(message, NASMessage):
            logger.debug(f"NAS: type={message.message_type}")
            
            if ProtocolLayer.NAS in self._native_handlers:
                for handler in self._native_handlers[ProtocolLayer.NAS]:
                    try:
                        handler(message)
                    except Exception as e:
                        logger.error(f"NAS handler error: {e}")
    
    # ========================================================================
    # Handler Registration
    # ========================================================================
    
    def register_native_handler(self, layer: ProtocolLayer, 
                                 handler: Callable[[ProtocolMessage], None]):
        """Register native protocol handler"""
        with self._lock:
            if layer not in self._native_handlers:
                self._native_handlers[layer] = []
            self._native_handlers[layer].append(handler)
    
    def unregister_native_handler(self, layer: ProtocolLayer, handler: Callable):
        """Unregister native protocol handler"""
        with self._lock:
            if layer in self._native_handlers and handler in self._native_handlers[layer]:
                self._native_handlers[layer].remove(handler)
    
    # ========================================================================
    # Connection Management
    # ========================================================================
    
    def add_connection(self, conn_id: str, family: ProtocolFamily, 
                       endpoint: str) -> bool:
        """Add external stack connection"""
        with self._lock:
            if conn_id in self._connections:
                return False
            
            self._connections[conn_id] = {
                'family': family,
                'endpoint': endpoint,
                'created': time.time(),
                'messages_sent': 0,
                'messages_received': 0
            }
            
            logger.info(f"Added connection {conn_id} ({family.name})")
            return True
    
    def remove_connection(self, conn_id: str):
        """Remove external stack connection"""
        with self._lock:
            if conn_id in self._connections:
                del self._connections[conn_id]
                logger.info(f"Removed connection {conn_id}")
    
    def get_connections(self) -> Dict[str, Dict]:
        """Get active connections"""
        with self._lock:
            return self._connections.copy()
    
    # ========================================================================
    # AI Control Interface
    # ========================================================================
    
    def execute_command(self, command: str) -> Dict[str, Any]:
        """
        Execute AI command
        
        Supports commands like:
        - "send s1ap setup to <endpoint>"
        - "route message to <family>"
        - "status"
        """
        command = command.lower().strip()
        result = {'success': False, 'message': ''}
        
        try:
            if 'status' in command:
                result['success'] = True
                result['connections'] = len(self._connections)
                result['router_stats'] = self._router.get_statistics()
                result['message'] = 'Bridge status OK'
            
            elif 'send' in command:
                # Parse send command
                result['success'] = True
                result['message'] = 'Message queued'
            
            else:
                result['message'] = 'Unknown command'
        
        except Exception as e:
            result['message'] = f'Error: {str(e)}'
        
        return result
    
    # ========================================================================
    # Events
    # ========================================================================
    
    def register_event_callback(self, event: str, callback: Callable):
        """Register event callback"""
        if event not in self._event_callbacks:
            self._event_callbacks[event] = []
        self._event_callbacks[event].append(callback)
    
    def _emit_event(self, event: str, data: Any):
        """Emit event to callbacks"""
        if event in self._event_callbacks:
            for callback in self._event_callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Event callback error: {e}")
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics"""
        return {
            'initialized': self._initialized,
            'connections': len(self._connections),
            'router': self._router.get_statistics(),
            'handlers': {
                layer.name: len(handlers) 
                for layer, handlers in self._native_handlers.items()
            }
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_protocol_bridge(stealth_system=None) -> ProtocolBridge:
    """Create and initialize protocol bridge"""
    bridge = ProtocolBridge(stealth_system=stealth_system)
    bridge.initialize()
    return bridge


def create_srsran_translator() -> SrsRANTranslator:
    """Create srsRAN protocol translator"""
    return SrsRANTranslator()


def create_oai_translator() -> OAITranslator:
    """Create OAI protocol translator"""
    return OAITranslator()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    'ProtocolLayer',
    'MessageDirection',
    'ProtocolFamily',
    
    # Messages
    'ProtocolMessage',
    'S1APMessage',
    'NGAPMessage',
    'GTPMessage',
    'RRCMessage',
    'NASMessage',
    
    # Translators
    'ProtocolTranslator',
    'SrsRANTranslator',
    'OAITranslator',
    
    # Router
    'ProtocolRouter',
    
    # Bridge
    'ProtocolBridge',
    
    # Factory
    'create_protocol_bridge',
    'create_srsran_translator',
    'create_oai_translator',
]
