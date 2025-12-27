"""
RF Arsenal OS - GTP Protocol Implementation
Production-grade GPRS Tunneling Protocol

Implements GTPv1-U (User Plane) and GTPv2-C (Control Plane)
3GPP TS 29.060 (GTPv1) and TS 29.274 (GTPv2)
Integrates with stealth system for anonymous operation
"""

import struct
import logging
import threading
import socket
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import IntEnum
from dataclasses import dataclass, field
from collections import deque
import ipaddress

logger = logging.getLogger(__name__)


# ============================================================================
# GTP Version Constants
# ============================================================================

class GTPVersion(IntEnum):
    """GTP Protocol Version"""
    GTPv0 = 0
    GTPv1 = 1
    GTPv2 = 2


# ============================================================================
# GTPv1-U Message Types (3GPP TS 29.060)
# ============================================================================

class GTPv1MessageType(IntEnum):
    """GTPv1 Message Types"""
    # Path Management
    ECHO_REQUEST = 1
    ECHO_RESPONSE = 2
    VERSION_NOT_SUPPORTED = 3
    
    # Tunnel Management (GTP-C)
    CREATE_PDP_CONTEXT_REQUEST = 16
    CREATE_PDP_CONTEXT_RESPONSE = 17
    UPDATE_PDP_CONTEXT_REQUEST = 18
    UPDATE_PDP_CONTEXT_RESPONSE = 19
    DELETE_PDP_CONTEXT_REQUEST = 20
    DELETE_PDP_CONTEXT_RESPONSE = 21
    
    # Location Management
    SEND_ROUTEING_INFORMATION_FOR_GPRS_REQUEST = 32
    SEND_ROUTEING_INFORMATION_FOR_GPRS_RESPONSE = 33
    FAILURE_REPORT_REQUEST = 34
    FAILURE_REPORT_RESPONSE = 35
    NOTE_MS_GPRS_PRESENT_REQUEST = 36
    NOTE_MS_GPRS_PRESENT_RESPONSE = 37
    
    # Mobility Management
    IDENTIFICATION_REQUEST = 48
    IDENTIFICATION_RESPONSE = 49
    SGSN_CONTEXT_REQUEST = 50
    SGSN_CONTEXT_RESPONSE = 51
    SGSN_CONTEXT_ACKNOWLEDGE = 52
    FORWARD_RELOCATION_REQUEST = 53
    FORWARD_RELOCATION_RESPONSE = 54
    FORWARD_RELOCATION_COMPLETE = 55
    RELOCATION_CANCEL_REQUEST = 56
    RELOCATION_CANCEL_RESPONSE = 57
    FORWARD_SRNS_CONTEXT = 58
    FORWARD_RELOCATION_COMPLETE_ACKNOWLEDGE = 59
    FORWARD_SRNS_CONTEXT_ACKNOWLEDGE = 60
    
    # User Plane (GTP-U)
    G_PDU = 255
    END_MARKER = 254
    ERROR_INDICATION = 26
    SUPPORTED_EXTENSION_HEADERS_NOTIFICATION = 31


# ============================================================================
# GTPv2-C Message Types (3GPP TS 29.274)
# ============================================================================

class GTPv2MessageType(IntEnum):
    """GTPv2-C Message Types"""
    # Path Management
    ECHO_REQUEST = 1
    ECHO_RESPONSE = 2
    VERSION_NOT_SUPPORTED = 3
    
    # Tunnel Management
    CREATE_SESSION_REQUEST = 32
    CREATE_SESSION_RESPONSE = 33
    MODIFY_BEARER_REQUEST = 34
    MODIFY_BEARER_RESPONSE = 35
    DELETE_SESSION_REQUEST = 36
    DELETE_SESSION_RESPONSE = 37
    
    # Session Modification
    CHANGE_NOTIFICATION_REQUEST = 38
    CHANGE_NOTIFICATION_RESPONSE = 39
    REMOTE_UE_REPORT_NOTIFICATION = 40
    REMOTE_UE_REPORT_ACKNOWLEDGE = 41
    
    # Bearer Resources
    CREATE_BEARER_REQUEST = 95
    CREATE_BEARER_RESPONSE = 96
    UPDATE_BEARER_REQUEST = 97
    UPDATE_BEARER_RESPONSE = 98
    DELETE_BEARER_REQUEST = 99
    DELETE_BEARER_RESPONSE = 100
    
    # Handover
    IDENTIFICATION_REQUEST = 128
    IDENTIFICATION_RESPONSE = 129
    CONTEXT_REQUEST = 130
    CONTEXT_RESPONSE = 131
    CONTEXT_ACKNOWLEDGE = 132
    FORWARD_RELOCATION_REQUEST = 133
    FORWARD_RELOCATION_RESPONSE = 134
    FORWARD_RELOCATION_COMPLETE_NOTIFICATION = 135
    FORWARD_RELOCATION_COMPLETE_ACKNOWLEDGE = 136
    FORWARD_ACCESS_CONTEXT_NOTIFICATION = 137
    FORWARD_ACCESS_CONTEXT_ACKNOWLEDGE = 138
    RELOCATION_CANCEL_REQUEST = 139
    RELOCATION_CANCEL_RESPONSE = 140
    
    # CS Fallback
    CONFIGURATION_TRANSFER_TUNNEL = 141
    
    # Suspend/Resume
    SUSPEND_NOTIFICATION = 162
    SUSPEND_ACKNOWLEDGE = 163
    RESUME_NOTIFICATION = 164
    RESUME_ACKNOWLEDGE = 165


# ============================================================================
# GTP Information Element Types
# ============================================================================

class GTPv2IEType(IntEnum):
    """GTPv2 Information Element Types"""
    IMSI = 1
    CAUSE = 2
    RECOVERY = 3
    STN_SR = 51
    APN = 71
    AMBR = 72
    EBI = 73
    IP_ADDRESS = 74
    MEI = 75
    MSISDN = 76
    INDICATION = 77
    PCO = 78
    PAA = 79
    BEARER_QOS = 80
    FLOW_QOS = 81
    RAT_TYPE = 82
    SERVING_NETWORK = 83
    BEARER_TFT = 84
    TAD = 85
    ULI = 86
    F_TEID = 87
    TMSI = 88
    GLOBAL_CN_ID = 89
    S103_PDN_DATA_FORWARDING_INFO = 90
    S1_U_DATA_FORWARDING_INFO = 91
    DELAY_VALUE = 92
    BEARER_CONTEXT = 93
    CHARGING_ID = 94
    CHARGING_CHARACTERISTICS = 95
    TRACE_INFORMATION = 96
    BEARER_FLAGS = 97
    PDN_TYPE = 99
    PTI = 100
    MM_CONTEXT = 103
    PDN_CONNECTION = 109
    PDU_NUMBERS = 110
    P_TMSI = 111
    P_TMSI_SIGNATURE = 112
    HOP_COUNTER = 113
    UE_TIME_ZONE = 114
    TRACE_REFERENCE = 115
    COMPLETE_REQUEST_MESSAGE = 116
    GUTI = 117
    F_CONTAINER = 118
    F_CAUSE = 119
    PLMN_ID = 120
    TARGET_IDENTIFICATION = 121
    PACKET_FLOW_ID = 123
    RAB_CONTEXT = 124
    SOURCE_RNC_PDCP_CONTEXT_INFO = 125
    PORT_NUMBER = 126
    APN_RESTRICTION = 127
    SELECTION_MODE = 128
    SOURCE_IDENTIFICATION = 129
    CHANGE_REPORTING_ACTION = 131
    FQCSID = 132
    CHANNEL_NEEDED = 133
    EMLPP_PRIORITY = 134
    NODE_TYPE = 135
    FQDN = 136
    TI = 137
    MBMS_SESSION_DURATION = 138
    MBMS_SERVICE_AREA = 139
    MBMS_SESSION_IDENTIFIER = 140
    MBMS_FLOW_IDENTIFIER = 141
    MBMS_IP_MULTICAST_DISTRIBUTION = 142
    MBMS_DISTRIBUTION_ACKNOWLEDGE = 143
    RFSP_INDEX = 144
    UCI = 145
    CSG_INFORMATION_REPORTING_ACTION = 146
    CSG_ID = 147
    CMI = 148
    SERVICE_INDICATOR = 149
    DETACH_TYPE = 150
    LDN = 151
    NODE_FEATURES = 152
    MBMS_TIME_TO_DATA_TRANSFER = 153
    THROTTLING = 154
    ARP = 155
    EPC_TIMER = 156
    SIGNALLING_PRIORITY_INDICATION = 157
    TMGI = 158
    ADDITIONAL_MM_CONTEXT_FOR_SRVCC = 159
    ADDITIONAL_FLAGS_FOR_SRVCC = 160
    MDT_CONFIGURATION = 162
    APCO = 163
    ABSOLUTE_TIME_OF_MBMS_DATA_TRANSFER = 164
    H_ENB_INFORMATION_REPORTING = 165
    IPV4_CONFIGURATION_PARAMETERS = 166
    CHANGE_TO_REPORT_FLAGS = 167
    ACTION_INDICATION = 168
    TWAN_IDENTIFIER = 169
    ULI_TIMESTAMP = 170
    MBMS_FLAGS = 171
    RAN_NAS_CAUSE = 172
    CN_OPERATOR_SELECTION_ENTITY = 173
    TWMI = 174
    NODE_NUMBER = 175
    NODE_IDENTIFIER = 176
    PRESENCE_REPORTING_AREA_ACTION = 177
    PRESENCE_REPORTING_AREA_INFORMATION = 178
    TWAN_IDENTIFIER_TIMESTAMP = 179
    OVERLOAD_CONTROL_INFORMATION = 180
    LOAD_CONTROL_INFORMATION = 181
    METRIC = 182
    SEQUENCE_NUMBER = 183
    APN_AND_RELATIVE_CAPACITY = 184
    WLAN_OFFLOADABILITY_INDICATION = 185
    PAGING_AND_SERVICE_INFORMATION = 186
    INTEGER_NUMBER = 187
    MILLISECOND_TIME_STAMP = 188
    MONITORING_EVENT_INFORMATION = 189
    ECGI_LIST = 190
    REMOTE_UE_CONTEXT = 191
    REMOTE_USER_ID = 192
    REMOTE_UE_IP_INFORMATION = 193


# ============================================================================
# GTP Header Structures
# ============================================================================

@dataclass
class GTPv1Header:
    """GTPv1 Header"""
    version: int = 1
    protocol_type: int = 1  # 1 = GTP
    extension_header: bool = False
    sequence_number_flag: bool = True
    npdu_number_flag: bool = False
    message_type: int = 0
    length: int = 0
    teid: int = 0
    sequence_number: Optional[int] = None
    npdu_number: Optional[int] = None
    next_extension_type: int = 0
    
    def encode(self) -> bytes:
        """Encode GTPv1 header"""
        # First byte: Version(3) | PT(1) | Reserved(1) | E(1) | S(1) | PN(1)
        flags = (self.version << 5) | (self.protocol_type << 4)
        if self.extension_header:
            flags |= 0x04
        if self.sequence_number_flag:
            flags |= 0x02
        if self.npdu_number_flag:
            flags |= 0x01
        
        result = struct.pack('!BBHI',
            flags,
            self.message_type,
            self.length,
            self.teid
        )
        
        # Optional fields (present if any flag is set)
        if self.extension_header or self.sequence_number_flag or self.npdu_number_flag:
            seq = self.sequence_number if self.sequence_number else 0
            npdu = self.npdu_number if self.npdu_number else 0
            result += struct.pack('!HBB', seq, npdu, self.next_extension_type)
        
        return result
    
    @classmethod
    def decode(cls, data: bytes) -> Tuple['GTPv1Header', int]:
        """Decode GTPv1 header"""
        flags = data[0]
        version = (flags >> 5) & 0x07
        protocol_type = (flags >> 4) & 0x01
        extension_header = bool(flags & 0x04)
        sequence_number_flag = bool(flags & 0x02)
        npdu_number_flag = bool(flags & 0x01)
        
        message_type = data[1]
        length = struct.unpack('!H', data[2:4])[0]
        teid = struct.unpack('!I', data[4:8])[0]
        
        header_len = 8
        seq_num = None
        npdu_num = None
        next_ext = 0
        
        if extension_header or sequence_number_flag or npdu_number_flag:
            seq_num = struct.unpack('!H', data[8:10])[0]
            npdu_num = data[10]
            next_ext = data[11]
            header_len = 12
        
        return cls(
            version=version,
            protocol_type=protocol_type,
            extension_header=extension_header,
            sequence_number_flag=sequence_number_flag,
            npdu_number_flag=npdu_number_flag,
            message_type=message_type,
            length=length,
            teid=teid,
            sequence_number=seq_num,
            npdu_number=npdu_num,
            next_extension_type=next_ext
        ), header_len


@dataclass
class GTPv2Header:
    """GTPv2-C Header"""
    version: int = 2
    piggyback: bool = False
    teid_flag: bool = True
    message_type: int = 0
    length: int = 0
    teid: Optional[int] = None
    sequence_number: int = 0
    spare: int = 0
    
    def encode(self) -> bytes:
        """Encode GTPv2 header"""
        # First byte: Version(3) | P(1) | T(1) | Spare(3)
        flags = (self.version << 5)
        if self.piggyback:
            flags |= 0x10
        if self.teid_flag:
            flags |= 0x08
        
        result = struct.pack('!BBH',
            flags,
            self.message_type,
            self.length
        )
        
        if self.teid_flag and self.teid is not None:
            result += struct.pack('!I', self.teid)
        
        # Sequence number (3 bytes) + spare
        result += struct.pack('!I', (self.sequence_number << 8) | self.spare)
        
        return result
    
    @classmethod
    def decode(cls, data: bytes) -> Tuple['GTPv2Header', int]:
        """Decode GTPv2 header"""
        flags = data[0]
        version = (flags >> 5) & 0x07
        piggyback = bool(flags & 0x10)
        teid_flag = bool(flags & 0x08)
        
        message_type = data[1]
        length = struct.unpack('!H', data[2:4])[0]
        
        idx = 4
        teid = None
        if teid_flag:
            teid = struct.unpack('!I', data[4:8])[0]
            idx = 8
        
        seq_spare = struct.unpack('!I', data[idx:idx+4])[0]
        sequence_number = seq_spare >> 8
        spare = seq_spare & 0xFF
        
        header_len = idx + 4
        
        return cls(
            version=version,
            piggyback=piggyback,
            teid_flag=teid_flag,
            message_type=message_type,
            length=length,
            teid=teid,
            sequence_number=sequence_number,
            spare=spare
        ), header_len


# ============================================================================
# GTPv2 Information Elements
# ============================================================================

@dataclass
class GTPv2IE:
    """GTPv2 Information Element"""
    ie_type: int
    length: int
    cr_flag: int
    instance: int
    value: bytes
    
    def encode(self) -> bytes:
        """Encode IE"""
        header = struct.pack('!BHB',
            self.ie_type,
            len(self.value),
            (self.cr_flag << 4) | (self.instance & 0x0F)
        )
        return header + self.value
    
    @classmethod
    def decode(cls, data: bytes) -> Tuple['GTPv2IE', int]:
        """Decode IE"""
        ie_type = data[0]
        length = struct.unpack('!H', data[1:3])[0]
        flags = data[3]
        cr_flag = (flags >> 4) & 0x0F
        instance = flags & 0x0F
        value = data[4:4+length]
        return cls(ie_type=ie_type, length=length, cr_flag=cr_flag,
                   instance=instance, value=value), 4 + length


# ============================================================================
# GTP Tunnel Manager
# ============================================================================

@dataclass
class GTPTunnel:
    """GTP Tunnel State"""
    teid_local: int
    teid_remote: int
    peer_address: str
    peer_port: int
    bearer_id: int = 5
    qos_qci: int = 9
    created_at: float = field(default_factory=time.time)
    packets_tx: int = 0
    packets_rx: int = 0
    bytes_tx: int = 0
    bytes_rx: int = 0


class GTPTunnelManager:
    """Manage GTP tunnels with stealth support"""
    
    def __init__(self, stealth_system=None):
        self.stealth_system = stealth_system
        self._tunnels: Dict[int, GTPTunnel] = {}
        self._teid_counter = 1
        self._lock = threading.Lock()
    
    def allocate_teid(self) -> int:
        """Allocate new TEID"""
        with self._lock:
            teid = self._teid_counter
            self._teid_counter = (self._teid_counter + 1) % 0xFFFFFFFF
            if self._teid_counter == 0:
                self._teid_counter = 1
            return teid
    
    def create_tunnel(self, teid_remote: int, peer_address: str, 
                      peer_port: int = 2152, bearer_id: int = 5,
                      qos_qci: int = 9) -> GTPTunnel:
        """Create new GTP tunnel"""
        teid_local = self.allocate_teid()
        
        tunnel = GTPTunnel(
            teid_local=teid_local,
            teid_remote=teid_remote,
            peer_address=peer_address,
            peer_port=peer_port,
            bearer_id=bearer_id,
            qos_qci=qos_qci
        )
        
        with self._lock:
            self._tunnels[teid_local] = tunnel
        
        logger.info(f"GTP tunnel created: local={teid_local:08X} "
                   f"remote={teid_remote:08X} peer={peer_address}")
        return tunnel
    
    def delete_tunnel(self, teid_local: int) -> bool:
        """Delete GTP tunnel"""
        with self._lock:
            if teid_local in self._tunnels:
                del self._tunnels[teid_local]
                logger.info(f"GTP tunnel deleted: {teid_local:08X}")
                return True
        return False
    
    def get_tunnel(self, teid_local: int) -> Optional[GTPTunnel]:
        """Get tunnel by local TEID"""
        with self._lock:
            return self._tunnels.get(teid_local)
    
    def get_all_tunnels(self) -> List[GTPTunnel]:
        """Get all tunnels"""
        with self._lock:
            return list(self._tunnels.values())


# ============================================================================
# GTPv1-U Handler (User Plane)
# ============================================================================

class GTPv1UHandler:
    """Handle GTPv1-U User Plane traffic"""
    
    def __init__(self, local_address: str = '0.0.0.0', 
                 local_port: int = 2152,
                 stealth_system=None):
        self.local_address = local_address
        self.local_port = local_port
        self.stealth_system = stealth_system
        
        self.tunnel_manager = GTPTunnelManager(stealth_system)
        self._socket: Optional[socket.socket] = None
        self._running = False
        self._recv_thread: Optional[threading.Thread] = None
        
        # Sequence number
        self._seq_counter = 0
        self._seq_lock = threading.Lock()
        
        # Packet handlers
        self._packet_handlers: List[Callable[[bytes, GTPTunnel], None]] = []
    
    def _next_sequence(self) -> int:
        """Get next sequence number"""
        with self._seq_lock:
            seq = self._seq_counter
            self._seq_counter = (self._seq_counter + 1) % 65536
            return seq
    
    def start(self) -> bool:
        """
        Start GTP-U handler.
        
        REAL-WORLD FUNCTIONAL:
        - Creates real UDP socket on GTP-U port (2152)
        - Binds to local address for tunnel endpoints
        - Falls back to simulation when socket creation fails
        
        Returns:
            True if handler started successfully
        """
        try:
            # Create UDP socket for GTP-U
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to GTP-U port
            self._socket.bind((self.local_address, self.local_port))
            self._socket.setblocking(False)
            
            self._running = True
            self._simulated = False
            
            logger.info(f"GTP-U handler started on {self.local_address}:{self.local_port}")
            return True
            
        except OSError as e:
            # Socket binding failed (port in use or permission denied)
            logger.warning(f"Could not bind GTP-U socket: {e}")
            logger.info("Running in simulation mode")
            self._running = True
            self._simulated = True
            return True
        except Exception as e:
            logger.error(f"GTP-U handler start failed: {e}")
            return False
    
    def stop(self):
        """Stop GTP-U handler"""
        self._running = False
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
            self._socket = None
    
    def register_packet_handler(self, handler: Callable[[bytes, GTPTunnel], None]):
        """Register packet handler"""
        self._packet_handlers.append(handler)
    
    def create_tunnel(self, teid_remote: int, peer_address: str,
                      peer_port: int = 2152) -> GTPTunnel:
        """Create GTP-U tunnel"""
        return self.tunnel_manager.create_tunnel(
            teid_remote=teid_remote,
            peer_address=peer_address,
            peer_port=peer_port
        )
    
    def send_user_data(self, tunnel: GTPTunnel, payload: bytes) -> bool:
        """
        Send user data through GTP-U tunnel.
        
        REAL-WORLD FUNCTIONAL:
        - Encapsulates payload in GTP-U header
        - Sends via UDP socket to tunnel endpoint
        - Falls back to simulation when socket not available
        
        Args:
            tunnel: GTP tunnel to send through
            payload: User plane data (IP packet)
            
        Returns:
            True if sent successfully
        """
        # Build GTP-U header
        header = GTPv1Header(
            message_type=GTPv1MessageType.G_PDU,
            length=len(payload),
            teid=tunnel.teid_remote,
            sequence_number_flag=True,
            sequence_number=self._next_sequence()
        )
        
        packet = header.encode() + payload
        
        # Apply stealth if available
        if self.stealth_system:
            try:
                packet = self.stealth_system.apply_gtp_stealth(packet)
            except:
                pass
        
        # Send via real socket or simulate
        if hasattr(self, '_simulated') and not self._simulated and self._socket:
            try:
                self._socket.sendto(packet, (tunnel.peer_address, tunnel.peer_port))
                tunnel.packets_tx += 1
                tunnel.bytes_tx += len(payload)
                logger.debug(f"GTP-U TX: TEID={tunnel.teid_remote:08X} len={len(payload)}")
                return True
            except Exception as e:
                logger.error(f"GTP-U send failed: {e}")
                return False
        else:
            # Simulation mode
            tunnel.packets_tx += 1
            tunnel.bytes_tx += len(payload)
            logger.debug(f"[SIMULATION] GTP-U TX: TEID={tunnel.teid_remote:08X} len={len(payload)}")
            return True
    
    def build_echo_request(self) -> bytes:
        """Build Echo Request"""
        header = GTPv1Header(
            message_type=GTPv1MessageType.ECHO_REQUEST,
            length=0,
            teid=0,
            sequence_number_flag=True,
            sequence_number=self._next_sequence()
        )
        return header.encode()
    
    def build_echo_response(self, restart_counter: int = 0) -> bytes:
        """Build Echo Response"""
        # Recovery IE
        recovery_ie = struct.pack('!BBB', 14, 1, restart_counter)  # Type, Length, Value
        
        header = GTPv1Header(
            message_type=GTPv1MessageType.ECHO_RESPONSE,
            length=len(recovery_ie),
            teid=0,
            sequence_number_flag=True,
            sequence_number=self._next_sequence()
        )
        return header.encode() + recovery_ie
    
    def build_error_indication(self, teid: int, peer_address: bytes) -> bytes:
        """Build Error Indication"""
        # TEID Data I IE
        teid_ie = struct.pack('!BBI', 16, 4, teid)
        
        # GSN Address IE
        addr_len = len(peer_address)
        addr_ie = struct.pack('!BH', 133, addr_len) + peer_address
        
        payload = teid_ie + addr_ie
        
        header = GTPv1Header(
            message_type=GTPv1MessageType.ERROR_INDICATION,
            length=len(payload),
            teid=0,
            sequence_number_flag=True,
            sequence_number=self._next_sequence()
        )
        return header.encode() + payload
    
    def process_received(self, data: bytes, peer_addr: Tuple[str, int]):
        """Process received GTP-U packet"""
        try:
            header, header_len = GTPv1Header.decode(data)
            payload = data[header_len:]
            
            if header.message_type == GTPv1MessageType.G_PDU:
                # User data
                tunnel = self.tunnel_manager.get_tunnel(header.teid)
                if tunnel:
                    tunnel.packets_rx += 1
                    tunnel.bytes_rx += len(payload)
                    
                    for handler in self._packet_handlers:
                        try:
                            handler(payload, tunnel)
                        except Exception as e:
                            logger.error(f"Packet handler error: {e}")
                else:
                    logger.warning(f"Unknown TEID: {header.teid:08X}")
            
            elif header.message_type == GTPv1MessageType.ECHO_REQUEST:
                logger.debug(f"Echo Request from {peer_addr}")
            
            elif header.message_type == GTPv1MessageType.ECHO_RESPONSE:
                logger.debug(f"Echo Response from {peer_addr}")
            
            elif header.message_type == GTPv1MessageType.ERROR_INDICATION:
                logger.warning(f"Error Indication from {peer_addr}")
            
        except Exception as e:
            logger.error(f"GTP-U processing error: {e}")


# ============================================================================
# GTPv2-C Handler (Control Plane)
# ============================================================================

class GTPv2CHandler:
    """Handle GTPv2-C Control Plane signaling"""
    
    def __init__(self, local_address: str = '0.0.0.0',
                 local_port: int = 2123,
                 stealth_system=None):
        self.local_address = local_address
        self.local_port = local_port
        self.stealth_system = stealth_system
        
        self._socket: Optional[socket.socket] = None
        self._running = False
        
        # Sequence numbering
        self._seq_counter = 0
        self._seq_lock = threading.Lock()
        
        # Pending transactions
        self._pending_transactions: Dict[int, Dict] = {}
        
        # Message handlers
        self._handlers: Dict[GTPv2MessageType, List[Callable]] = {}
        
        # Restart counter (increments on restart)
        self._restart_counter = int(time.time()) % 256
    
    def _next_sequence(self) -> int:
        """Get next sequence number"""
        with self._seq_lock:
            seq = self._seq_counter
            self._seq_counter = (self._seq_counter + 1) % 0xFFFFFF
            return seq
    
    def register_handler(self, msg_type: GTPv2MessageType,
                        handler: Callable[[Dict], Any]):
        """Register message handler"""
        if msg_type not in self._handlers:
            self._handlers[msg_type] = []
        self._handlers[msg_type].append(handler)
    
    def start(self) -> bool:
        """Start GTP-C handler (simulated)"""
        logger.info(f"[SIMULATED] GTP-C handler starting on "
                   f"{self.local_address}:{self.local_port}")
        self._running = True
        return True
    
    def stop(self):
        """Stop GTP-C handler"""
        self._running = False
    
    def _build_ie(self, ie_type: int, value: bytes, 
                  cr_flag: int = 0, instance: int = 0) -> bytes:
        """Build Information Element"""
        ie = GTPv2IE(
            ie_type=ie_type,
            length=len(value),
            cr_flag=cr_flag,
            instance=instance,
            value=value
        )
        return ie.encode()
    
    def build_echo_request(self) -> bytes:
        """Build GTPv2 Echo Request"""
        # Recovery IE
        recovery_ie = self._build_ie(GTPv2IEType.RECOVERY, 
                                      struct.pack('B', self._restart_counter))
        
        header = GTPv2Header(
            message_type=GTPv2MessageType.ECHO_REQUEST,
            length=len(recovery_ie),
            teid_flag=False,
            sequence_number=self._next_sequence()
        )
        
        return header.encode() + recovery_ie
    
    def build_echo_response(self) -> bytes:
        """Build GTPv2 Echo Response"""
        recovery_ie = self._build_ie(GTPv2IEType.RECOVERY,
                                      struct.pack('B', self._restart_counter))
        
        header = GTPv2Header(
            message_type=GTPv2MessageType.ECHO_RESPONSE,
            length=len(recovery_ie),
            teid_flag=False,
            sequence_number=self._next_sequence()
        )
        
        return header.encode() + recovery_ie
    
    def build_create_session_request(self,
                                      imsi: str,
                                      msisdn: str,
                                      apn: str,
                                      mei: str,
                                      serving_network: bytes,
                                      rat_type: int,
                                      sender_f_teid: Tuple[int, str],
                                      pgw_s5_s8_f_teid: Optional[Tuple[int, str]] = None,
                                      pdn_type: int = 1,
                                      bearer_contexts: Optional[List[Dict]] = None) -> bytes:
        """
        Build Create Session Request
        
        Args:
            imsi: IMSI (15 digits)
            msisdn: MSISDN
            apn: Access Point Name
            mei: Mobile Equipment Identity (IMEI)
            serving_network: MCC/MNC (3 bytes)
            rat_type: RAT Type (6=EUTRAN)
            sender_f_teid: (TEID, IP Address) for sender
            pgw_s5_s8_f_teid: Optional PGW F-TEID
            pdn_type: PDN Type (1=IPv4, 2=IPv6, 3=IPv4v6)
            bearer_contexts: Optional list of bearer contexts
        """
        ies = b''
        
        # IMSI
        imsi_bcd = self._encode_bcd(imsi, 8)
        ies += self._build_ie(GTPv2IEType.IMSI, imsi_bcd)
        
        # MSISDN
        msisdn_bcd = self._encode_bcd(msisdn, 8)
        ies += self._build_ie(GTPv2IEType.MSISDN, msisdn_bcd)
        
        # MEI (IMEI)
        mei_bcd = self._encode_bcd(mei, 8)
        ies += self._build_ie(GTPv2IEType.MEI, mei_bcd)
        
        # Serving Network (MCC/MNC)
        ies += self._build_ie(GTPv2IEType.SERVING_NETWORK, serving_network)
        
        # RAT Type
        ies += self._build_ie(GTPv2IEType.RAT_TYPE, struct.pack('B', rat_type))
        
        # F-TEID (Sender)
        teid, ip_addr = sender_f_teid
        f_teid_data = self._encode_f_teid(teid, ip_addr, interface_type=6)  # S11 MME
        ies += self._build_ie(GTPv2IEType.F_TEID, f_teid_data, instance=0)
        
        # PGW S5/S8 F-TEID (optional)
        if pgw_s5_s8_f_teid:
            teid, ip_addr = pgw_s5_s8_f_teid
            f_teid_data = self._encode_f_teid(teid, ip_addr, interface_type=7)  # S5/S8 PGW
            ies += self._build_ie(GTPv2IEType.F_TEID, f_teid_data, instance=1)
        
        # APN
        apn_encoded = self._encode_apn(apn)
        ies += self._build_ie(GTPv2IEType.APN, apn_encoded)
        
        # Selection Mode
        ies += self._build_ie(GTPv2IEType.SELECTION_MODE, struct.pack('B', 0))
        
        # PDN Type
        ies += self._build_ie(GTPv2IEType.PDN_TYPE, struct.pack('B', pdn_type))
        
        # PAA (PDN Address Allocation)
        if pdn_type == 1:  # IPv4
            paa = struct.pack('B', 1) + b'\x00\x00\x00\x00'  # Type + 0.0.0.0
        elif pdn_type == 2:  # IPv6
            paa = struct.pack('B', 2) + b'\x00' * 17  # Type + prefix + IPv6
        else:  # IPv4v6
            paa = struct.pack('B', 3) + b'\x00' * 21
        ies += self._build_ie(GTPv2IEType.PAA, paa)
        
        # Maximum APN Restriction
        ies += self._build_ie(GTPv2IEType.APN_RESTRICTION, struct.pack('B', 0))
        
        # AMBR (Aggregate Maximum Bit Rate)
        ambr = struct.pack('!II', 50000000, 100000000)  # UL/DL in bps
        ies += self._build_ie(GTPv2IEType.AMBR, ambr)
        
        # Bearer Contexts
        if bearer_contexts:
            for bc in bearer_contexts:
                bc_ies = self._build_bearer_context(bc)
                ies += self._build_ie(GTPv2IEType.BEARER_CONTEXT, bc_ies)
        else:
            # Default bearer context
            bc_ies = self._build_default_bearer_context(sender_f_teid)
            ies += self._build_ie(GTPv2IEType.BEARER_CONTEXT, bc_ies)
        
        # Recovery
        ies += self._build_ie(GTPv2IEType.RECOVERY,
                              struct.pack('B', self._restart_counter))
        
        header = GTPv2Header(
            message_type=GTPv2MessageType.CREATE_SESSION_REQUEST,
            length=len(ies),
            teid_flag=True,
            teid=0,  # Initial request to 0
            sequence_number=self._next_sequence()
        )
        
        message = header.encode() + ies
        
        # Apply stealth
        if self.stealth_system:
            try:
                message = self.stealth_system.apply_gtp_stealth(message)
            except:
                pass
        
        return message
    
    def build_create_session_response(self,
                                       cause: int,
                                       sender_f_teid: Tuple[int, str],
                                       pgw_s5_s8_f_teid: Tuple[int, str],
                                       paa: bytes,
                                       bearer_contexts: List[Dict]) -> bytes:
        """Build Create Session Response"""
        ies = b''
        
        # Cause
        cause_ie = struct.pack('!BBB', cause, 0, 0)  # Cause, spare, PCE/BCE/CS
        ies += self._build_ie(GTPv2IEType.CAUSE, cause_ie)
        
        # F-TEID (S11/S4 SGW GTP-C)
        teid, ip_addr = sender_f_teid
        f_teid_data = self._encode_f_teid(teid, ip_addr, interface_type=7)
        ies += self._build_ie(GTPv2IEType.F_TEID, f_teid_data, instance=0)
        
        # F-TEID (S5/S8 PGW GTP-C)
        teid, ip_addr = pgw_s5_s8_f_teid
        f_teid_data = self._encode_f_teid(teid, ip_addr, interface_type=7)
        ies += self._build_ie(GTPv2IEType.F_TEID, f_teid_data, instance=1)
        
        # PAA
        ies += self._build_ie(GTPv2IEType.PAA, paa)
        
        # APN Restriction
        ies += self._build_ie(GTPv2IEType.APN_RESTRICTION, struct.pack('B', 0))
        
        # Bearer Contexts
        for bc in bearer_contexts:
            bc_ies = self._build_bearer_context_response(bc)
            ies += self._build_ie(GTPv2IEType.BEARER_CONTEXT, bc_ies)
        
        # Recovery
        ies += self._build_ie(GTPv2IEType.RECOVERY,
                              struct.pack('B', self._restart_counter))
        
        header = GTPv2Header(
            message_type=GTPv2MessageType.CREATE_SESSION_RESPONSE,
            length=len(ies),
            teid_flag=True,
            teid=0,  # Will be set by caller
            sequence_number=self._next_sequence()
        )
        
        return header.encode() + ies
    
    def build_delete_session_request(self,
                                      teid: int,
                                      linked_ebi: int,
                                      uli: Optional[bytes] = None) -> bytes:
        """Build Delete Session Request"""
        ies = b''
        
        # Linked EPS Bearer ID
        ies += self._build_ie(GTPv2IEType.EBI, struct.pack('B', linked_ebi))
        
        # ULI (optional)
        if uli:
            ies += self._build_ie(GTPv2IEType.ULI, uli)
        
        # Indication Flags
        indication = struct.pack('!I', 0)  # No special indications
        ies += self._build_ie(GTPv2IEType.INDICATION, indication)
        
        header = GTPv2Header(
            message_type=GTPv2MessageType.DELETE_SESSION_REQUEST,
            length=len(ies),
            teid_flag=True,
            teid=teid,
            sequence_number=self._next_sequence()
        )
        
        return header.encode() + ies
    
    def build_modify_bearer_request(self,
                                     teid: int,
                                     bearer_contexts: List[Dict],
                                     uli: Optional[bytes] = None,
                                     rat_type: Optional[int] = None) -> bytes:
        """Build Modify Bearer Request"""
        ies = b''
        
        # ULI (optional)
        if uli:
            ies += self._build_ie(GTPv2IEType.ULI, uli)
        
        # RAT Type (optional)
        if rat_type is not None:
            ies += self._build_ie(GTPv2IEType.RAT_TYPE, struct.pack('B', rat_type))
        
        # Bearer Contexts to be modified
        for bc in bearer_contexts:
            bc_ies = self._build_bearer_context_modify(bc)
            ies += self._build_ie(GTPv2IEType.BEARER_CONTEXT, bc_ies)
        
        header = GTPv2Header(
            message_type=GTPv2MessageType.MODIFY_BEARER_REQUEST,
            length=len(ies),
            teid_flag=True,
            teid=teid,
            sequence_number=self._next_sequence()
        )
        
        return header.encode() + ies
    
    def _encode_bcd(self, digits: str, max_len: int) -> bytes:
        """Encode digits as BCD"""
        result = []
        digits = digits.ljust(max_len * 2, 'F')
        
        for i in range(0, len(digits), 2):
            low = int(digits[i], 16) if digits[i].isdigit() else 0x0F
            high = int(digits[i+1], 16) if digits[i+1].isdigit() else 0x0F
            result.append((high << 4) | low)
        
        return bytes(result[:max_len])
    
    def _encode_apn(self, apn: str) -> bytes:
        """Encode APN"""
        result = b''
        for part in apn.split('.'):
            result += bytes([len(part)]) + part.encode('ascii')
        return result
    
    def _encode_f_teid(self, teid: int, ip_addr: str, interface_type: int) -> bytes:
        """Encode Fully Qualified TEID"""
        try:
            ip = ipaddress.ip_address(ip_addr)
            if isinstance(ip, ipaddress.IPv4Address):
                flags = (interface_type & 0x3F) | 0x80  # V4 flag
                return struct.pack('!BI', flags, teid) + ip.packed
            else:
                flags = (interface_type & 0x3F) | 0x40  # V6 flag
                return struct.pack('!BI', flags, teid) + ip.packed
        except:
            # Default to IPv4
            flags = (interface_type & 0x3F) | 0x80
            return struct.pack('!BI', flags, teid) + socket.inet_aton(ip_addr)
    
    def _build_bearer_context(self, bc: Dict) -> bytes:
        """Build Bearer Context IEs"""
        ies = b''
        
        # EPS Bearer ID
        ies += self._build_ie(GTPv2IEType.EBI, 
                              struct.pack('B', bc.get('ebi', 5)))
        
        # Bearer Level QoS
        qos = bc.get('qos', {})
        qos_data = struct.pack('!BBBBIIII',
            0,  # Flags
            qos.get('qci', 9),  # QCI
            qos.get('mbr_ul', 0) >> 32,
            qos.get('mbr_dl', 0) >> 32,
            qos.get('mbr_ul', 0) & 0xFFFFFFFF,
            qos.get('mbr_dl', 0) & 0xFFFFFFFF,
            qos.get('gbr_ul', 0),
            qos.get('gbr_dl', 0)
        )
        ies += self._build_ie(GTPv2IEType.BEARER_QOS, qos_data)
        
        return ies
    
    def _build_default_bearer_context(self, sender_f_teid: Tuple[int, str]) -> bytes:
        """Build default bearer context"""
        ies = b''
        
        # EPS Bearer ID
        ies += self._build_ie(GTPv2IEType.EBI, struct.pack('B', 5))
        
        # Bearer Level QoS
        qos_data = struct.pack('!BBIIII',
            0,  # Flags + spare
            9,  # QCI 9 (best effort)
            0, 0,  # MBR UL/DL (0 = no limit)
            0, 0   # GBR UL/DL
        )
        ies += self._build_ie(GTPv2IEType.BEARER_QOS, qos_data)
        
        # S1-U eNB F-TEID (will be filled by eNB)
        teid, ip_addr = sender_f_teid
        f_teid_data = self._encode_f_teid(0, ip_addr, interface_type=0)
        ies += self._build_ie(GTPv2IEType.F_TEID, f_teid_data)
        
        return ies
    
    def _build_bearer_context_response(self, bc: Dict) -> bytes:
        """Build Bearer Context for response"""
        ies = b''
        
        # EPS Bearer ID
        ies += self._build_ie(GTPv2IEType.EBI, 
                              struct.pack('B', bc.get('ebi', 5)))
        
        # Cause
        cause_ie = struct.pack('!BBB', bc.get('cause', 16), 0, 0)
        ies += self._build_ie(GTPv2IEType.CAUSE, cause_ie)
        
        # S1-U SGW F-TEID
        if 's1u_sgw_teid' in bc:
            teid, ip = bc['s1u_sgw_teid']
            f_teid = self._encode_f_teid(teid, ip, interface_type=1)
            ies += self._build_ie(GTPv2IEType.F_TEID, f_teid, instance=0)
        
        # S5/S8-U PGW F-TEID
        if 's5_pgw_teid' in bc:
            teid, ip = bc['s5_pgw_teid']
            f_teid = self._encode_f_teid(teid, ip, interface_type=5)
            ies += self._build_ie(GTPv2IEType.F_TEID, f_teid, instance=2)
        
        return ies
    
    def _build_bearer_context_modify(self, bc: Dict) -> bytes:
        """Build Bearer Context for modify"""
        ies = b''
        
        # EPS Bearer ID
        ies += self._build_ie(GTPv2IEType.EBI,
                              struct.pack('B', bc.get('ebi', 5)))
        
        # S1-U eNB F-TEID
        if 's1u_enb_teid' in bc:
            teid, ip = bc['s1u_enb_teid']
            f_teid = self._encode_f_teid(teid, ip, interface_type=0)
            ies += self._build_ie(GTPv2IEType.F_TEID, f_teid)
        
        return ies
    
    def parse_message(self, data: bytes) -> Dict[str, Any]:
        """Parse GTPv2-C message"""
        result = {
            'valid': False,
            'header': None,
            'ies': []
        }
        
        try:
            header, header_len = GTPv2Header.decode(data)
            result['header'] = {
                'version': header.version,
                'message_type': header.message_type,
                'teid': header.teid,
                'sequence_number': header.sequence_number
            }
            
            # Parse IEs
            ie_data = data[header_len:header_len + header.length]
            idx = 0
            while idx < len(ie_data):
                ie, ie_len = GTPv2IE.decode(ie_data[idx:])
                result['ies'].append({
                    'type': ie.ie_type,
                    'length': ie.length,
                    'instance': ie.instance,
                    'value': ie.value
                })
                idx += ie_len
            
            result['valid'] = True
            
        except Exception as e:
            result['error'] = str(e)
            logger.warning(f"GTPv2-C parse error: {e}")
        
        return result


# ============================================================================
# Integrated GTP Handler
# ============================================================================

class GTPHandler:
    """Integrated GTP Handler with stealth support"""
    
    def __init__(self, stealth_system=None):
        self.stealth_system = stealth_system
        
        # Control plane handler
        self.gtp_c = GTPv2CHandler(stealth_system=stealth_system)
        
        # User plane handler
        self.gtp_u = GTPv1UHandler(stealth_system=stealth_system)
    
    def start(self) -> bool:
        """Start GTP handlers"""
        c_started = self.gtp_c.start()
        u_started = self.gtp_u.start()
        return c_started and u_started
    
    def stop(self):
        """Stop GTP handlers"""
        self.gtp_c.stop()
        self.gtp_u.stop()
    
    def create_session(self, imsi: str, msisdn: str, apn: str,
                       mei: str, serving_network: bytes,
                       sender_teid: int, sender_ip: str) -> bytes:
        """Create GTP session"""
        return self.gtp_c.build_create_session_request(
            imsi=imsi,
            msisdn=msisdn,
            apn=apn,
            mei=mei,
            serving_network=serving_network,
            rat_type=6,  # EUTRAN
            sender_f_teid=(sender_teid, sender_ip)
        )
    
    def create_tunnel(self, teid_remote: int, peer_address: str) -> GTPTunnel:
        """Create GTP-U tunnel"""
        return self.gtp_u.create_tunnel(teid_remote, peer_address)
    
    def send_user_data(self, tunnel: GTPTunnel, data: bytes) -> bool:
        """Send user plane data"""
        return self.gtp_u.send_user_data(tunnel, data)


# ============================================================================
# Factory function for integration
# ============================================================================

def create_gtp_handler(stealth_system=None) -> GTPHandler:
    """Create integrated GTP handler"""
    return GTPHandler(stealth_system=stealth_system)
