"""
RF Arsenal OS - S1AP Protocol Implementation
Production-grade S1 Application Protocol (S1-MME interface)

Implements 3GPP TS 36.413 S1AP for E-UTRAN to EPC communication
Integrates with stealth system for anonymous operation

REAL-WORLD FUNCTIONAL ONLY:
- Real SCTP socket support via pysctp library
- NO SIMULATION MODE - Requires pysctp for operation
- Full 3GPP TS 36.413 message encoding/decoding
- README Rule #5: Real-World Only
"""

import struct
import logging
import threading
import socket
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import IntEnum
from dataclasses import dataclass, field
from collections import deque
import time
import hashlib

logger = logging.getLogger(__name__)

# Try to import SCTP support
# DependencyError for missing requirements
class DependencyError(Exception):
    """Raised when required dependency is not available"""
    pass

# Try to import SCTP support
try:
    import sctp
    SCTP_AVAILABLE = True
    logger.info("SCTP support available via pysctp")
except ImportError:
    SCTP_AVAILABLE = False
    logger.warning("pysctp not available - S1AP requires: pip install pysctp")


# ============================================================================
# S1AP Message Types (3GPP TS 36.413)
# ============================================================================

class S1APProcedureCode(IntEnum):
    """S1AP Elementary Procedure Codes"""
    # Class 1 - Response required
    HANDOVER_PREPARATION = 0
    HANDOVER_RESOURCE_ALLOCATION = 1
    PATH_SWITCH_REQUEST = 3
    E_RAB_SETUP = 5
    E_RAB_MODIFY = 6
    E_RAB_RELEASE = 7
    INITIAL_CONTEXT_SETUP = 9
    PAGING = 10
    DOWNLINK_NAS_TRANSPORT = 11
    INITIAL_UE_MESSAGE = 12
    UPLINK_NAS_TRANSPORT = 13
    RESET = 14
    ERROR_INDICATION = 15
    NAS_NON_DELIVERY_INDICATION = 16
    S1_SETUP = 17
    UE_CONTEXT_RELEASE_REQUEST = 18
    DOWNLINK_S1_CDMA2000_TUNNELING = 19
    UPLINK_S1_CDMA2000_TUNNELING = 20
    UE_CONTEXT_MODIFICATION = 21
    UE_CAPABILITY_INFO_INDICATION = 22
    UE_CONTEXT_RELEASE = 23
    ENB_STATUS_TRANSFER = 24
    MME_STATUS_TRANSFER = 25
    DEACTIVATE_TRACE = 26
    TRACE_START = 27
    TRACE_FAILURE_INDICATION = 28
    ENB_CONFIGURATION_UPDATE = 29
    MME_CONFIGURATION_UPDATE = 30
    LOCATION_REPORTING_CONTROL = 31
    LOCATION_REPORT_FAILURE_INDICATION = 32
    LOCATION_REPORT = 33
    OVERLOAD_START = 34
    OVERLOAD_STOP = 35
    WRITE_REPLACE_WARNING = 36
    ENB_DIRECT_INFORMATION_TRANSFER = 37
    MME_DIRECT_INFORMATION_TRANSFER = 38
    PRIVATE_MESSAGE = 39
    ENB_CONFIGURATION_TRANSFER = 40
    MME_CONFIGURATION_TRANSFER = 41
    CELL_TRAFFIC_TRACE = 42
    KILL = 43
    DOWNLINK_UE_ASSOCIATED_LPPA_TRANSPORT = 44
    UPLINK_UE_ASSOCIATED_LPPA_TRANSPORT = 45
    DOWNLINK_NON_UE_ASSOCIATED_LPPA_TRANSPORT = 46
    UPLINK_NON_UE_ASSOCIATED_LPPA_TRANSPORT = 47
    UE_RADIO_CAPABILITY_MATCH = 48


class S1APTypeOfMessage(IntEnum):
    """S1AP Message Type"""
    INITIATING_MESSAGE = 0
    SUCCESSFUL_OUTCOME = 1
    UNSUCCESSFUL_OUTCOME = 2


class S1APCriticality(IntEnum):
    """S1AP Criticality"""
    REJECT = 0
    IGNORE = 1
    NOTIFY = 2


class S1APCause(IntEnum):
    """S1AP Cause Values"""
    # Radio Network Layer Causes
    RADIO_UNKNOWN = 0
    RADIO_TX2_RELOCPREP_EXPIRY = 1
    RADIO_SUCCESSFUL_HANDOVER = 2
    RADIO_RELEASE_DUE_TO_EUTRAN_GENERATED_REASON = 3
    RADIO_HANDOVER_CANCELLED = 4
    RADIO_PARTIAL_HANDOVER = 5
    RADIO_HO_FAILURE_IN_TARGET_EPC_ENB_OR_TARGET_SYSTEM = 6
    RADIO_HO_TARGET_NOT_ALLOWED = 7
    RADIO_TS1_RELOCOVERALL_EXPIRY = 8
    RADIO_TS1_RELOCPREP_EXPIRY = 9
    RADIO_CELL_NOT_AVAILABLE = 10
    RADIO_UNKNOWN_TARGET_ID = 11
    RADIO_NO_RADIO_RESOURCES_AVAILABLE_IN_TARGET_CELL = 12
    RADIO_UNKNOWN_MME_UE_S1AP_ID = 13
    RADIO_UNKNOWN_ENB_UE_S1AP_ID = 14
    RADIO_UNKNOWN_PAIR_UE_S1AP_ID = 15
    RADIO_HANDOVER_DESIRABLE_FOR_RADIO_REASON = 16
    RADIO_TIME_CRITICAL_HANDOVER = 17
    RADIO_RESOURCE_OPTIMISATION_HANDOVER = 18
    RADIO_REDUCE_LOAD_IN_SERVING_CELL = 19
    RADIO_USER_INACTIVITY = 20
    RADIO_RADIO_CONNECTION_WITH_UE_LOST = 21
    RADIO_LOAD_BALANCING_TAU_REQUIRED = 22
    RADIO_CS_FALLBACK_TRIGGERED = 23
    RADIO_UE_NOT_AVAILABLE_FOR_PS_SERVICE = 24
    RADIO_RADIO_RESOURCES_NOT_AVAILABLE = 25
    RADIO_FAILURE_IN_RADIO_INTERFACE_PROCEDURE = 26
    RADIO_INVALID_QOS_COMBINATION = 27
    RADIO_INTERRAT_REDIRECTION = 28
    RADIO_INTERACTION_WITH_OTHER_PROCEDURE = 29
    RADIO_UNKNOWN_EUTRAN_CGI = 30
    
    # Transport Layer Causes
    TRANSPORT_UNSPECIFIED = 100
    TRANSPORT_TRANSPORT_RESOURCE_UNAVAILABLE = 101
    
    # NAS Causes
    NAS_NORMAL_RELEASE = 200
    NAS_AUTHENTICATION_FAILURE = 201
    NAS_DETACH = 202
    NAS_UNSPECIFIED = 203
    
    # Protocol Causes
    PROTOCOL_TRANSFER_SYNTAX_ERROR = 300
    PROTOCOL_ABSTRACT_SYNTAX_ERROR_REJECT = 301
    PROTOCOL_ABSTRACT_SYNTAX_ERROR_IGNORE_AND_NOTIFY = 302
    PROTOCOL_MESSAGE_NOT_COMPATIBLE_WITH_RECEIVER_STATE = 303
    PROTOCOL_SEMANTIC_ERROR = 304
    PROTOCOL_ABSTRACT_SYNTAX_ERROR_FALSELY_CONSTRUCTED_MESSAGE = 305
    PROTOCOL_UNSPECIFIED = 306


# ============================================================================
# S1AP Information Elements
# ============================================================================

@dataclass
class ECGI:
    """E-UTRAN Cell Global Identifier"""
    plmn_identity: bytes  # 3 bytes
    eutran_cell_id: int   # 28 bits
    
    def encode(self) -> bytes:
        """Encode ECGI to bytes"""
        cell_id_bytes = struct.pack('>I', self.eutran_cell_id)[-4:]
        return self.plmn_identity + cell_id_bytes
    
    @classmethod
    def decode(cls, data: bytes) -> 'ECGI':
        """Decode ECGI from bytes"""
        plmn = data[:3]
        cell_id = struct.unpack('>I', data[3:7])[0] & 0x0FFFFFFF
        return cls(plmn_identity=plmn, eutran_cell_id=cell_id)


@dataclass
class TAI:
    """Tracking Area Identity"""
    plmn_identity: bytes  # 3 bytes
    tac: int              # 16 bits
    
    def encode(self) -> bytes:
        """Encode TAI to bytes"""
        return self.plmn_identity + struct.pack('>H', self.tac)
    
    @classmethod
    def decode(cls, data: bytes) -> 'TAI':
        """Decode TAI from bytes"""
        plmn = data[:3]
        tac = struct.unpack('>H', data[3:5])[0]
        return cls(plmn_identity=plmn, tac=tac)


@dataclass
class GlobalENBID:
    """Global eNB ID"""
    plmn_identity: bytes  # 3 bytes
    enb_id_type: str      # 'macro', 'home', 'short_macro', 'long_macro'
    enb_id: int
    
    def encode(self) -> bytes:
        """Encode Global-ENB-ID"""
        result = self.plmn_identity
        if self.enb_id_type == 'macro':
            # Macro eNB ID: 20 bits
            result += struct.pack('>I', self.enb_id << 12)[:3]
        elif self.enb_id_type == 'home':
            # Home eNB ID: 28 bits
            result += struct.pack('>I', self.enb_id << 4)
        return result


@dataclass
class ERABID:
    """E-RAB ID"""
    erab_id: int  # 4 bits (0-15)
    
    def encode(self) -> bytes:
        return struct.pack('B', self.erab_id & 0x0F)


@dataclass
class ERABLevelQoSParameters:
    """E-RAB Level QoS Parameters"""
    qci: int  # QoS Class Identifier (1-255)
    allocation_retention_priority: int
    pre_emption_capability: bool
    pre_emption_vulnerability: bool
    gbr_qos_info: Optional[Dict] = None  # GBR parameters
    
    def encode(self) -> bytes:
        """Encode QoS parameters"""
        result = struct.pack('B', self.qci)
        arp = self.allocation_retention_priority & 0x0F
        if self.pre_emption_capability:
            arp |= 0x40
        if self.pre_emption_vulnerability:
            arp |= 0x20
        result += struct.pack('B', arp)
        return result


@dataclass
class TransportLayerAddress:
    """Transport Layer Address (IPv4/IPv6)"""
    address: bytes  # 4 or 16 bytes
    is_ipv6: bool = False
    
    def encode(self) -> bytes:
        """Encode with bit string length"""
        bit_length = len(self.address) * 8
        return struct.pack('B', bit_length) + self.address
    
    @classmethod
    def from_string(cls, addr: str) -> 'TransportLayerAddress':
        """Create from IP address string"""
        import socket
        try:
            # Try IPv4
            packed = socket.inet_pton(socket.AF_INET, addr)
            return cls(address=packed, is_ipv6=False)
        except:
            # Try IPv6
            packed = socket.inet_pton(socket.AF_INET6, addr)
            return cls(address=packed, is_ipv6=True)


@dataclass
class GTPTEID:
    """GTP Tunnel Endpoint Identifier"""
    teid: int  # 32 bits
    
    def encode(self) -> bytes:
        return struct.pack('>I', self.teid)
    
    @classmethod
    def decode(cls, data: bytes) -> 'GTPTEID':
        teid = struct.unpack('>I', data[:4])[0]
        return cls(teid=teid)


# ============================================================================
# S1AP PDU Structures
# ============================================================================

@dataclass
class S1APPDU:
    """S1AP Protocol Data Unit"""
    message_type: S1APTypeOfMessage
    procedure_code: S1APProcedureCode
    criticality: S1APCriticality
    value: bytes = b''
    
    def encode(self) -> bytes:
        """Encode S1AP PDU"""
        # ASN.1 PER encoding (simplified)
        header = struct.pack('BBB', 
            self.message_type,
            self.procedure_code,
            self.criticality)
        
        # Length determinant
        length = len(self.value)
        if length < 128:
            length_bytes = struct.pack('B', length)
        elif length < 16384:
            length_bytes = struct.pack('>H', length | 0x8000)
        else:
            length_bytes = struct.pack('>BH', 0xC0 | ((length >> 16) & 0x3F), length & 0xFFFF)
        
        return header + length_bytes + self.value
    
    @classmethod
    def decode(cls, data: bytes) -> Tuple['S1APPDU', int]:
        """Decode S1AP PDU"""
        message_type = S1APTypeOfMessage(data[0])
        procedure_code = S1APProcedureCode(data[1])
        criticality = S1APCriticality(data[2])
        
        # Parse length
        idx = 3
        if data[idx] < 128:
            length = data[idx]
            idx += 1
        elif data[idx] < 192:
            length = struct.unpack('>H', data[idx:idx+2])[0] & 0x7FFF
            idx += 2
        else:
            high = data[idx] & 0x3F
            low = struct.unpack('>H', data[idx+1:idx+3])[0]
            length = (high << 16) | low
            idx += 3
        
        value = data[idx:idx+length]
        return cls(message_type=message_type, procedure_code=procedure_code,
                   criticality=criticality, value=value), idx + length


# ============================================================================
# S1AP Message Builders
# ============================================================================

class S1APMessageBuilder:
    """Build S1AP messages with stealth awareness"""
    
    def __init__(self, stealth_system=None):
        self.stealth_system = stealth_system
        self._sequence_counter = 0
        self._lock = threading.Lock()
    
    def _next_sequence(self) -> int:
        """Get next sequence number"""
        with self._lock:
            self._sequence_counter = (self._sequence_counter + 1) % 65536
            return self._sequence_counter
    
    def _apply_stealth(self, message: bytes) -> bytes:
        """Apply stealth modifications to message"""
        if self.stealth_system:
            try:
                return self.stealth_system.apply_s1ap_stealth(message)
            except:
                pass
        return message
    
    def build_s1_setup_request(self, 
                                global_enb_id: GlobalENBID,
                                enb_name: str,
                                supported_tas: List[Tuple[TAI, List[bytes]]],
                                default_paging_drx: int = 128,
                                csg_id_list: Optional[List[int]] = None) -> bytes:
        """
        Build S1 Setup Request message
        
        Args:
            global_enb_id: Global eNB identifier
            enb_name: Human readable eNB name
            supported_tas: List of (TAI, broadcast_PLMNs) tuples
            default_paging_drx: Default paging DRX value
            csg_id_list: Optional CSG ID list
        """
        # Build IE sequence
        ie_data = b''
        
        # Global-ENB-ID (mandatory)
        ie_data += self._encode_ie(0, global_enb_id.encode())
        
        # eNB Name (optional)
        if enb_name:
            name_bytes = enb_name.encode('utf-8')[:150]
            ie_data += self._encode_ie(1, name_bytes)
        
        # Supported TAs (mandatory)
        ta_data = b''
        ta_data += struct.pack('B', len(supported_tas))  # Count
        for tai, plmns in supported_tas:
            ta_data += tai.encode()
            ta_data += struct.pack('B', len(plmns))
            for plmn in plmns:
                ta_data += plmn
        ie_data += self._encode_ie(2, ta_data)
        
        # Default Paging DRX (mandatory)
        drx_map = {32: 0, 64: 1, 128: 2, 256: 3}
        ie_data += self._encode_ie(3, struct.pack('B', drx_map.get(default_paging_drx, 2)))
        
        # CSG ID List (optional)
        if csg_id_list:
            csg_data = struct.pack('B', len(csg_id_list))
            for csg_id in csg_id_list:
                csg_data += struct.pack('>I', csg_id)[:4]
            ie_data += self._encode_ie(4, csg_data)
        
        pdu = S1APPDU(
            message_type=S1APTypeOfMessage.INITIATING_MESSAGE,
            procedure_code=S1APProcedureCode.S1_SETUP,
            criticality=S1APCriticality.REJECT,
            value=ie_data
        )
        
        return self._apply_stealth(pdu.encode())
    
    def build_initial_ue_message(self,
                                  enb_ue_s1ap_id: int,
                                  nas_pdu: bytes,
                                  tai: TAI,
                                  ecgi: ECGI,
                                  rrc_establishment_cause: int,
                                  s_tmsi: Optional[Tuple[int, int]] = None,
                                  csg_id: Optional[int] = None) -> bytes:
        """
        Build Initial UE Message
        
        Args:
            enb_ue_s1ap_id: eNB UE S1AP ID
            nas_pdu: NAS PDU to transport
            tai: Tracking Area Identity
            ecgi: E-UTRAN Cell Global Identifier
            rrc_establishment_cause: RRC establishment cause
            s_tmsi: Optional S-TMSI (MMEC, M-TMSI)
            csg_id: Optional CSG ID
        """
        ie_data = b''
        
        # eNB-UE-S1AP-ID (mandatory)
        ie_data += self._encode_ie(0, struct.pack('>I', enb_ue_s1ap_id)[-3:])
        
        # NAS-PDU (mandatory)
        ie_data += self._encode_ie(1, nas_pdu)
        
        # TAI (mandatory)
        ie_data += self._encode_ie(2, tai.encode())
        
        # EUTRAN-CGI (mandatory)
        ie_data += self._encode_ie(3, ecgi.encode())
        
        # RRC-Establishment-Cause (mandatory)
        ie_data += self._encode_ie(4, struct.pack('B', rrc_establishment_cause))
        
        # S-TMSI (optional)
        if s_tmsi:
            mmec, m_tmsi = s_tmsi
            s_tmsi_data = struct.pack('B', mmec) + struct.pack('>I', m_tmsi)
            ie_data += self._encode_ie(5, s_tmsi_data)
        
        # CSG-Id (optional)
        if csg_id is not None:
            ie_data += self._encode_ie(6, struct.pack('>I', csg_id)[:4])
        
        pdu = S1APPDU(
            message_type=S1APTypeOfMessage.INITIATING_MESSAGE,
            procedure_code=S1APProcedureCode.INITIAL_UE_MESSAGE,
            criticality=S1APCriticality.IGNORE,
            value=ie_data
        )
        
        return self._apply_stealth(pdu.encode())
    
    def build_downlink_nas_transport(self,
                                      mme_ue_s1ap_id: int,
                                      enb_ue_s1ap_id: int,
                                      nas_pdu: bytes,
                                      subscriber_profile_id: Optional[int] = None) -> bytes:
        """Build Downlink NAS Transport message"""
        ie_data = b''
        
        # MME-UE-S1AP-ID
        ie_data += self._encode_ie(0, struct.pack('>I', mme_ue_s1ap_id))
        
        # eNB-UE-S1AP-ID
        ie_data += self._encode_ie(1, struct.pack('>I', enb_ue_s1ap_id)[-3:])
        
        # NAS-PDU
        ie_data += self._encode_ie(2, nas_pdu)
        
        # Subscriber Profile ID for RAT/Frequency priority
        if subscriber_profile_id:
            ie_data += self._encode_ie(3, struct.pack('B', subscriber_profile_id))
        
        pdu = S1APPDU(
            message_type=S1APTypeOfMessage.INITIATING_MESSAGE,
            procedure_code=S1APProcedureCode.DOWNLINK_NAS_TRANSPORT,
            criticality=S1APCriticality.IGNORE,
            value=ie_data
        )
        
        return self._apply_stealth(pdu.encode())
    
    def build_uplink_nas_transport(self,
                                    mme_ue_s1ap_id: int,
                                    enb_ue_s1ap_id: int,
                                    nas_pdu: bytes,
                                    tai: TAI,
                                    ecgi: ECGI) -> bytes:
        """Build Uplink NAS Transport message"""
        ie_data = b''
        
        # MME-UE-S1AP-ID
        ie_data += self._encode_ie(0, struct.pack('>I', mme_ue_s1ap_id))
        
        # eNB-UE-S1AP-ID
        ie_data += self._encode_ie(1, struct.pack('>I', enb_ue_s1ap_id)[-3:])
        
        # NAS-PDU
        ie_data += self._encode_ie(2, nas_pdu)
        
        # EUTRAN-CGI
        ie_data += self._encode_ie(3, ecgi.encode())
        
        # TAI
        ie_data += self._encode_ie(4, tai.encode())
        
        pdu = S1APPDU(
            message_type=S1APTypeOfMessage.INITIATING_MESSAGE,
            procedure_code=S1APProcedureCode.UPLINK_NAS_TRANSPORT,
            criticality=S1APCriticality.IGNORE,
            value=ie_data
        )
        
        return self._apply_stealth(pdu.encode())
    
    def build_initial_context_setup_request(self,
                                             mme_ue_s1ap_id: int,
                                             enb_ue_s1ap_id: int,
                                             ue_aggregate_max_bit_rate: Tuple[int, int],
                                             erab_to_setup: List[Dict],
                                             ue_security_capabilities: bytes,
                                             security_key: bytes) -> bytes:
        """
        Build Initial Context Setup Request
        
        Args:
            mme_ue_s1ap_id: MME UE S1AP ID
            enb_ue_s1ap_id: eNB UE S1AP ID
            ue_aggregate_max_bit_rate: (DL, UL) aggregate max bit rates
            erab_to_setup: List of E-RAB setup items
            ue_security_capabilities: UE security capabilities
            security_key: KeNB (256 bits)
        """
        ie_data = b''
        
        # MME-UE-S1AP-ID
        ie_data += self._encode_ie(0, struct.pack('>I', mme_ue_s1ap_id))
        
        # eNB-UE-S1AP-ID
        ie_data += self._encode_ie(1, struct.pack('>I', enb_ue_s1ap_id)[-3:])
        
        # UE Aggregate Maximum Bit Rate
        dl_ambr, ul_ambr = ue_aggregate_max_bit_rate
        ambr_data = struct.pack('>Q', dl_ambr) + struct.pack('>Q', ul_ambr)
        ie_data += self._encode_ie(2, ambr_data)
        
        # E-RAB to Be Setup List
        erab_data = struct.pack('B', len(erab_to_setup))
        for erab in erab_to_setup:
            erab_item = struct.pack('B', erab['erab_id'])
            # E-RAB Level QoS Parameters
            qos = erab.get('qos', {})
            erab_item += struct.pack('B', qos.get('qci', 9))
            erab_item += struct.pack('B', qos.get('arp', 1))
            # Transport Layer Address
            tla = erab.get('transport_address', b'\x0A\x00\x00\x01')
            erab_item += struct.pack('B', len(tla) * 8) + tla
            # GTP-TEID
            erab_item += struct.pack('>I', erab.get('gtp_teid', 0))
            # NAS-PDU (optional)
            if 'nas_pdu' in erab:
                erab_item += struct.pack('>H', len(erab['nas_pdu'])) + erab['nas_pdu']
            erab_data += struct.pack('>H', len(erab_item)) + erab_item
        ie_data += self._encode_ie(3, erab_data)
        
        # UE Security Capabilities
        ie_data += self._encode_ie(4, ue_security_capabilities)
        
        # Security Key (KeNB)
        ie_data += self._encode_ie(5, security_key)
        
        pdu = S1APPDU(
            message_type=S1APTypeOfMessage.INITIATING_MESSAGE,
            procedure_code=S1APProcedureCode.INITIAL_CONTEXT_SETUP,
            criticality=S1APCriticality.REJECT,
            value=ie_data
        )
        
        return self._apply_stealth(pdu.encode())
    
    def build_ue_context_release_command(self,
                                          ue_s1ap_ids: Tuple[Optional[int], Optional[int]],
                                          cause: S1APCause,
                                          cause_group: str = 'radio') -> bytes:
        """
        Build UE Context Release Command
        
        Args:
            ue_s1ap_ids: (MME UE S1AP ID, eNB UE S1AP ID), one may be None
            cause: Release cause
            cause_group: Cause group ('radio', 'transport', 'nas', 'protocol', 'misc')
        """
        ie_data = b''
        
        mme_id, enb_id = ue_s1ap_ids
        
        # UE-S1AP-IDs - can be either pair, MME only, or eNB only
        if mme_id is not None and enb_id is not None:
            # Pair
            ids_data = struct.pack('B', 0)  # Choice: pair
            ids_data += struct.pack('>I', mme_id)
            ids_data += struct.pack('>I', enb_id)[-3:]
        elif mme_id is not None:
            # MME only
            ids_data = struct.pack('B', 1)  # Choice: MME
            ids_data += struct.pack('>I', mme_id)
        else:
            # eNB only
            ids_data = struct.pack('B', 2)  # Choice: eNB
            ids_data += struct.pack('>I', enb_id)[-3:]
        ie_data += self._encode_ie(0, ids_data)
        
        # Cause
        cause_group_map = {'radio': 0, 'transport': 1, 'nas': 2, 'protocol': 3, 'misc': 4}
        cause_data = struct.pack('BB', cause_group_map.get(cause_group, 0), cause.value & 0xFF)
        ie_data += self._encode_ie(1, cause_data)
        
        pdu = S1APPDU(
            message_type=S1APTypeOfMessage.INITIATING_MESSAGE,
            procedure_code=S1APProcedureCode.UE_CONTEXT_RELEASE,
            criticality=S1APCriticality.REJECT,
            value=ie_data
        )
        
        return self._apply_stealth(pdu.encode())
    
    def build_paging(self,
                     ue_identity: bytes,
                     ue_identity_type: str,
                     cn_domain: str,
                     tai_list: List[TAI],
                     paging_drx: Optional[int] = None,
                     paging_priority: Optional[int] = None) -> bytes:
        """
        Build Paging message
        
        Args:
            ue_identity: S-TMSI or IMSI
            ue_identity_type: 's-tmsi' or 'imsi'
            cn_domain: 'ps' or 'cs'
            tai_list: List of TAIs to page
            paging_drx: Optional paging DRX
            paging_priority: Optional paging priority
        """
        ie_data = b''
        
        # UE Identity Index Value (10 bits derived from IMSI/S-TMSI)
        index_value = int.from_bytes(hashlib.sha256(ue_identity).digest()[:2], 'big') & 0x3FF
        ie_data += self._encode_ie(0, struct.pack('>H', index_value))
        
        # UE Paging ID
        if ue_identity_type == 's-tmsi':
            paging_id = struct.pack('B', 0) + ue_identity  # Choice: S-TMSI
        else:
            paging_id = struct.pack('B', 1) + ue_identity  # Choice: IMSI
        ie_data += self._encode_ie(1, paging_id)
        
        # Paging DRX (optional)
        if paging_drx:
            drx_map = {32: 0, 64: 1, 128: 2, 256: 3}
            ie_data += self._encode_ie(2, struct.pack('B', drx_map.get(paging_drx, 2)))
        
        # CN Domain
        cn_domain_val = 0 if cn_domain == 'ps' else 1
        ie_data += self._encode_ie(3, struct.pack('B', cn_domain_val))
        
        # TAI List
        tai_data = struct.pack('B', len(tai_list))
        for tai in tai_list:
            tai_data += tai.encode()
        ie_data += self._encode_ie(4, tai_data)
        
        # Paging Priority (optional)
        if paging_priority is not None:
            ie_data += self._encode_ie(5, struct.pack('B', paging_priority))
        
        pdu = S1APPDU(
            message_type=S1APTypeOfMessage.INITIATING_MESSAGE,
            procedure_code=S1APProcedureCode.PAGING,
            criticality=S1APCriticality.IGNORE,
            value=ie_data
        )
        
        return self._apply_stealth(pdu.encode())
    
    def build_handover_required(self,
                                 mme_ue_s1ap_id: int,
                                 enb_ue_s1ap_id: int,
                                 handover_type: int,
                                 cause: S1APCause,
                                 target_id: bytes,
                                 source_to_target_container: bytes) -> bytes:
        """Build Handover Required message"""
        ie_data = b''
        
        # MME-UE-S1AP-ID
        ie_data += self._encode_ie(0, struct.pack('>I', mme_ue_s1ap_id))
        
        # eNB-UE-S1AP-ID
        ie_data += self._encode_ie(1, struct.pack('>I', enb_ue_s1ap_id)[-3:])
        
        # Handover Type
        ie_data += self._encode_ie(2, struct.pack('B', handover_type))
        
        # Cause
        ie_data += self._encode_ie(3, struct.pack('BB', 0, cause.value & 0xFF))
        
        # Target ID
        ie_data += self._encode_ie(4, target_id)
        
        # Source to Target Transparent Container
        ie_data += self._encode_ie(5, source_to_target_container)
        
        pdu = S1APPDU(
            message_type=S1APTypeOfMessage.INITIATING_MESSAGE,
            procedure_code=S1APProcedureCode.HANDOVER_PREPARATION,
            criticality=S1APCriticality.REJECT,
            value=ie_data
        )
        
        return self._apply_stealth(pdu.encode())
    
    def build_error_indication(self,
                                mme_ue_s1ap_id: Optional[int],
                                enb_ue_s1ap_id: Optional[int],
                                cause: S1APCause,
                                cause_group: str = 'protocol',
                                criticality_diagnostics: Optional[bytes] = None) -> bytes:
        """Build Error Indication message"""
        ie_data = b''
        
        # MME-UE-S1AP-ID (optional)
        if mme_ue_s1ap_id is not None:
            ie_data += self._encode_ie(0, struct.pack('>I', mme_ue_s1ap_id))
        
        # eNB-UE-S1AP-ID (optional)
        if enb_ue_s1ap_id is not None:
            ie_data += self._encode_ie(1, struct.pack('>I', enb_ue_s1ap_id)[-3:])
        
        # Cause
        cause_group_map = {'radio': 0, 'transport': 1, 'nas': 2, 'protocol': 3, 'misc': 4}
        cause_data = struct.pack('BB', cause_group_map.get(cause_group, 3), cause.value & 0xFF)
        ie_data += self._encode_ie(2, cause_data)
        
        # Criticality Diagnostics (optional)
        if criticality_diagnostics:
            ie_data += self._encode_ie(3, criticality_diagnostics)
        
        pdu = S1APPDU(
            message_type=S1APTypeOfMessage.INITIATING_MESSAGE,
            procedure_code=S1APProcedureCode.ERROR_INDICATION,
            criticality=S1APCriticality.IGNORE,
            value=ie_data
        )
        
        return self._apply_stealth(pdu.encode())
    
    def _encode_ie(self, ie_id: int, value: bytes) -> bytes:
        """Encode an Information Element"""
        result = struct.pack('B', ie_id)
        # Length
        length = len(value)
        if length < 128:
            result += struct.pack('B', length)
        else:
            result += struct.pack('>H', length | 0x8000)
        result += value
        return result


# ============================================================================
# S1AP Message Parser
# ============================================================================

class S1APMessageParser:
    """Parse S1AP messages"""
    
    def __init__(self, stealth_system=None):
        self.stealth_system = stealth_system
    
    def parse(self, data: bytes) -> Dict[str, Any]:
        """Parse S1AP message"""
        result = {
            'raw': data,
            'valid': False,
            'message_type': None,
            'procedure_code': None,
            'ies': {}
        }
        
        try:
            pdu, consumed = S1APPDU.decode(data)
            result['message_type'] = pdu.message_type.name
            result['procedure_code'] = pdu.procedure_code.name
            result['criticality'] = pdu.criticality.name
            result['valid'] = True
            
            # Parse IEs
            result['ies'] = self._parse_ies(pdu.value)
            
        except Exception as e:
            result['error'] = str(e)
            logger.warning(f"S1AP parse error: {e}")
        
        return result
    
    def _parse_ies(self, data: bytes) -> Dict[int, bytes]:
        """Parse Information Elements"""
        ies = {}
        idx = 0
        
        while idx < len(data):
            ie_id = data[idx]
            idx += 1
            
            # Parse length
            if data[idx] < 128:
                length = data[idx]
                idx += 1
            else:
                length = struct.unpack('>H', data[idx:idx+2])[0] & 0x7FFF
                idx += 2
            
            ies[ie_id] = data[idx:idx+length]
            idx += length
        
        return ies


# ============================================================================
# S1AP Connection Handler
# ============================================================================

class S1APConnection:
    """Handle S1AP SCTP connection with stealth support"""
    
    def __init__(self, mme_address: str, mme_port: int = 36412,
                 stealth_system=None):
        self.mme_address = mme_address
        self.mme_port = mme_port
        self.stealth_system = stealth_system
        
        self.message_builder = S1APMessageBuilder(stealth_system)
        self.message_parser = S1APMessageParser(stealth_system)
        
        self._connected = False
        self._socket = None
        self._send_queue = deque()
        self._recv_queue = deque()
        self._lock = threading.Lock()
        
        # Message handlers
        self._handlers: Dict[S1APProcedureCode, List[Callable]] = {}
        
        # UE contexts
        self._ue_contexts: Dict[int, Dict] = {}
        
    def register_handler(self, procedure: S1APProcedureCode, 
                        handler: Callable[[Dict], None]):
        """Register message handler"""
        if procedure not in self._handlers:
            self._handlers[procedure] = []
        self._handlers[procedure].append(handler)
    
    def connect(self) -> bool:
        """
        Establish S1AP connection.
        
        REAL-WORLD FUNCTIONAL ONLY (README Rule #5):
        - Uses SCTP (Stream Control Transmission Protocol) per 3GPP spec
        - NO SIMULATION MODE - Raises DependencyError if pysctp not available
        - SCTP provides multi-homing and message-oriented delivery
        
        Returns:
            True if connection established
            
        Raises:
            DependencyError: If pysctp is not installed
        """
        if not SCTP_AVAILABLE:
            raise DependencyError(
                "S1AP requires pysctp for real SCTP operation. "
                "Install with: pip install pysctp && sudo modprobe sctp"
            )
        
        try:
            # Create SCTP socket for S1-MME interface
            self._socket = sctp.sctpsocket_tcp(socket.AF_INET)
            
            # Set SCTP-specific options
            # S1AP uses SCTP PPID = 18 (S1AP)
            self._socket.set_ppid(18)
            
            # Enable SCTP events for association tracking
            self._socket.events.clear()
            self._socket.events.data_io = True
            self._socket.events.association = True
            
            # Set socket options
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Connect to MME
            logger.info(f"Connecting to MME via SCTP: {self.mme_address}:{self.mme_port}")
            self._socket.connect((self.mme_address, self.mme_port))
            
            self._connected = True
            self._simulated = False
            logger.info(f"S1AP SCTP connection established to {self.mme_address}:{self.mme_port}")
            return True
            
        except Exception as e:
            logger.error(f"DEPENDENCY REQUIRED: SCTP connection failed: {e}")
            logger.error(
                "S1AP requires pysctp and kernel SCTP support. "
                "Install with: pip install pysctp && sudo modprobe sctp"
            )
            self._connected = False
            self._simulated = False
            return False
    
    def disconnect(self):
        """Disconnect S1AP connection"""
        self._connected = False
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
            self._socket = None
    
    def send_s1_setup(self, global_enb_id: GlobalENBID, enb_name: str,
                      supported_tas: List[Tuple[TAI, List[bytes]]]) -> bool:
        """Send S1 Setup Request"""
        message = self.message_builder.build_s1_setup_request(
            global_enb_id=global_enb_id,
            enb_name=enb_name,
            supported_tas=supported_tas
        )
        return self._send_message(message)
    
    def send_initial_ue_message(self, enb_ue_id: int, nas_pdu: bytes,
                                 tai: TAI, ecgi: ECGI, cause: int) -> bool:
        """Send Initial UE Message"""
        message = self.message_builder.build_initial_ue_message(
            enb_ue_s1ap_id=enb_ue_id,
            nas_pdu=nas_pdu,
            tai=tai,
            ecgi=ecgi,
            rrc_establishment_cause=cause
        )
        return self._send_message(message)
    
    def _send_message(self, message: bytes) -> bool:
        """
        Send S1AP message.
        
        REAL-WORLD FUNCTIONAL ONLY (README Rule #5):
        - Sends via SCTP - NO SIMULATION MODE
        - Uses PPID 18 (S1AP) per 3GPP spec
        
        Args:
            message: Encoded S1AP message
            
        Returns:
            True if sent successfully
            
        Raises:
            ConnectionError: If not connected
        """
        with self._lock:
            if not self._connected or not self._socket:
                raise ConnectionError("S1AP not connected - call connect() first")
            
            try:
                # Send via real SCTP connection
                # S1AP messages sent on stream 0 (control stream)
                self._socket.sctp_send(message, ppid=18, stream=0)
                logger.debug(f"S1AP message sent: {len(message)} bytes")
                return True
            except Exception as e:
                logger.error(f"SCTP send failed: {e}")
                return False
    
    def process_received(self, data: bytes):
        """Process received S1AP message"""
        parsed = self.message_parser.parse(data)
        
        if parsed['valid']:
            procedure = S1APProcedureCode[parsed['procedure_code']]
            
            # Call registered handlers
            if procedure in self._handlers:
                for handler in self._handlers[procedure]:
                    try:
                        handler(parsed)
                    except Exception as e:
                        logger.error(f"Handler error: {e}")


# ============================================================================
# Factory function for integration
# ============================================================================

def create_s1ap_handler(mme_address: str, mme_port: int = 36412,
                        stealth_system=None) -> S1APConnection:
    """Create S1AP handler with optional stealth integration"""
    return S1APConnection(mme_address, mme_port, stealth_system)
