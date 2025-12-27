#!/usr/bin/env python3
"""
RF Arsenal OS - RRC (Radio Resource Control) Protocol

Production-grade RRC implementation for LTE/5G.
Handles connection management, security, and mobility.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import threading

from .asn1 import PEREncoder, PERDecoder, BitBuffer, ASN1Constraint

logger = logging.getLogger(__name__)


class RRCState(Enum):
    """RRC connection state"""
    IDLE = "idle"
    CONNECTED = "connected"
    INACTIVE = "inactive"  # 5G NR


class RRCMessageType(Enum):
    """RRC message types (simplified)"""
    # DL-CCCH
    RRC_CONNECTION_SETUP = 0
    RRC_CONNECTION_REJECT = 1
    RRC_CONNECTION_REESTABLISHMENT = 2
    RRC_CONNECTION_REESTABLISHMENT_REJECT = 3
    
    # DL-DCCH
    RRC_CONNECTION_RECONFIGURATION = 4
    RRC_CONNECTION_RELEASE = 5
    SECURITY_MODE_COMMAND = 6
    UE_CAPABILITY_ENQUIRY = 7
    DL_INFORMATION_TRANSFER = 8
    
    # UL-CCCH
    RRC_CONNECTION_REQUEST = 10
    RRC_CONNECTION_REESTABLISHMENT_REQUEST = 11
    
    # UL-DCCH
    RRC_CONNECTION_SETUP_COMPLETE = 12
    RRC_CONNECTION_RECONFIGURATION_COMPLETE = 13
    SECURITY_MODE_COMPLETE = 14
    SECURITY_MODE_FAILURE = 15
    UE_CAPABILITY_INFORMATION = 16
    UL_INFORMATION_TRANSFER = 17
    
    # Broadcast
    MASTER_INFORMATION_BLOCK = 20
    SYSTEM_INFORMATION_BLOCK_TYPE_1 = 21
    SYSTEM_INFORMATION = 22


@dataclass
class PLMNIdentity:
    """Public Land Mobile Network Identity"""
    mcc: str = "001"  # Mobile Country Code (3 digits)
    mnc: str = "01"   # Mobile Network Code (2-3 digits)
    
    def encode(self) -> bytes:
        """Encode PLMN-Identity"""
        encoder = PEREncoder()
        buf = BitBuffer()
        
        # MCC (3 digits, 4 bits each)
        for digit in self.mcc:
            buf.write_bits(int(digit), 4)
        
        # MNC (2-3 digits)
        if len(self.mnc) == 2:
            buf.write_bits(0xF, 4)  # Filler
        for digit in self.mnc:
            buf.write_bits(int(digit), 4)
        
        return buf.get_bytes()


@dataclass
class CellIdentity:
    """Cell Identity (28 bits)"""
    enb_id: int = 0      # eNodeB ID (20 bits)
    cell_id: int = 0     # Cell ID (8 bits)
    
    def encode(self) -> int:
        """Encode as 28-bit value"""
        return (self.enb_id << 8) | self.cell_id


@dataclass
class MIB:
    """
    Master Information Block
    
    Broadcast every 40ms in PBCH.
    """
    dl_bandwidth: int = 50        # N_RB: 6, 15, 25, 50, 75, 100
    phich_config_duration: int = 0  # 0: normal, 1: extended
    phich_config_resource: int = 0  # 0: 1/6, 1: 1/2, 2: 1, 3: 2
    system_frame_number: int = 0   # SFN (8 MSBs, 2 LSBs in subframe)
    spare: int = 0                 # 10 bits spare
    
    def encode(self) -> bytes:
        """Encode MIB (24 bits payload)"""
        encoder = PEREncoder()
        buf = BitBuffer()
        
        # dl-Bandwidth (3 bits)
        bw_map = {6: 0, 15: 1, 25: 2, 50: 3, 75: 4, 100: 5}
        buf.write_bits(bw_map.get(self.dl_bandwidth, 3), 3)
        
        # phich-Config (3 bits)
        buf.write_bit(self.phich_config_duration)
        buf.write_bits(self.phich_config_resource, 2)
        
        # systemFrameNumber (8 bits - MSBs of SFN)
        buf.write_bits(self.system_frame_number >> 2, 8)
        
        # spare (10 bits)
        buf.write_bits(self.spare, 10)
        
        return buf.get_bytes()
    
    @classmethod
    def decode(cls, data: bytes) -> 'MIB':
        """Decode MIB from bytes"""
        buf = BitBuffer(data)
        
        bw_map = {0: 6, 1: 15, 2: 25, 3: 50, 4: 75, 5: 100}
        dl_bandwidth = bw_map[buf.read_bits(3)]
        
        phich_duration = buf.read_bit()
        phich_resource = buf.read_bits(2)
        
        sfn = buf.read_bits(8) << 2
        spare = buf.read_bits(10)
        
        return cls(
            dl_bandwidth=dl_bandwidth,
            phich_config_duration=phich_duration,
            phich_config_resource=phich_resource,
            system_frame_number=sfn,
            spare=spare
        )


@dataclass
class SIB1:
    """
    System Information Block Type 1
    
    Contains cell access information and scheduling.
    """
    plmn_identity_list: List[PLMNIdentity] = field(default_factory=lambda: [PLMNIdentity()])
    tracking_area_code: int = 0x0001
    cell_identity: CellIdentity = field(default_factory=CellIdentity)
    cell_barred: bool = False
    intra_freq_reselection: bool = True
    csg_indication: bool = False
    q_rxlev_min: int = -70  # dBm * 2
    frequency_band_indicator: int = 7
    scheduling_info_list: List[Dict] = field(default_factory=list)
    si_window_length: int = 20  # ms
    system_info_value_tag: int = 0
    
    def encode(self) -> bytes:
        """Encode SIB1"""
        encoder = PEREncoder()
        buf = BitBuffer()
        
        # Extension marker
        buf.write_bit(0)
        
        # Optional fields bitmap
        # cellAccessRelatedInfo.csg-Identity present
        buf.write_bit(0)  # Not present
        
        # cellAccessRelatedInfo
        # plmn-IdentityList (1-6 entries)
        buf.write_bits(len(self.plmn_identity_list) - 1, 3)  # 0-5 encoded
        
        for plmn in self.plmn_identity_list:
            # PLMN-IdentityInfo
            buf.write_bit(0)  # Extension marker
            
            # plmn-Identity
            plmn_bytes = plmn.encode()
            for byte in plmn_bytes:
                buf.write_bits(byte, 8)
            
            # cellReservedForOperatorUse
            buf.write_bit(0)  # notReserved
        
        # trackingAreaCode (16 bits)
        buf.write_bits(self.tracking_area_code, 16)
        
        # cellIdentity (28 bits)
        buf.write_bits(self.cell_identity.encode(), 28)
        
        # cellBarred
        buf.write_bit(1 if self.cell_barred else 0)
        
        # intraFreqReselection
        buf.write_bit(0 if self.intra_freq_reselection else 1)
        
        # csg-Indication
        buf.write_bit(1 if self.csg_indication else 0)
        
        # cellSelectionInfo
        buf.write_bit(0)  # Extension marker
        buf.write_bit(0)  # q-RxLevMinOffset not present
        
        # q-RxLevMin (-70..-22 by 2)
        buf.write_bits((-self.q_rxlev_min - 22) // 2, 6)
        
        # freqBandIndicator (1-64)
        buf.write_bits(self.frequency_band_indicator - 1, 6)
        
        # schedulingInfoList (1-32 entries)
        si_count = max(1, len(self.scheduling_info_list))
        buf.write_bits(si_count - 1, 5)
        
        for _ in range(si_count):
            # SchedulingInfo
            buf.write_bits(0, 3)  # si-Periodicity: rf8
            buf.write_bits(0, 5)  # sib-MappingInfo: empty
        
        # si-WindowLength
        window_map = {1: 0, 2: 1, 5: 2, 10: 3, 15: 4, 20: 5, 40: 6}
        buf.write_bits(window_map.get(self.si_window_length, 5), 3)
        
        # systemInfoValueTag (0-31)
        buf.write_bits(self.system_info_value_tag, 5)
        
        return buf.get_bytes()


@dataclass
class SIB2:
    """
    System Information Block Type 2
    
    Contains radio resource configuration common to all UEs.
    """
    ac_barring_for_emergency: bool = False
    time_alignment_timer: int = 7  # sf500 - infinity
    
    # RACH configuration
    number_of_ra_preambles: int = 64
    size_of_ra_preambles_group_a: int = 52
    power_ramping_step: int = 0  # dB0
    preamble_initial_target_power: int = -90  # dBm
    max_harq_msg3_tx: int = 5
    
    # PRACH configuration
    prach_config_index: int = 0
    prach_frequency_offset: int = 0
    
    # PUSCH configuration
    n_sb: int = 1
    hopping_mode: int = 0  # interSubFrame
    pusch_hopping_offset: int = 0
    enable_64qam: bool = False
    
    # PUCCH configuration
    delta_pucch_shift: int = 1  # ds1
    n_rb_cqi: int = 0
    n_cs_an: int = 0
    n1_pucch_an: int = 0
    
    # SRS configuration
    srs_bandwidth_config: int = 0
    srs_subframe_config: int = 0
    
    # UL power control
    p0_nominal_pusch: int = -80
    alpha: int = 7  # al1
    p0_nominal_pucch: int = -100
    
    def encode(self) -> bytes:
        """Encode SIB2"""
        encoder = PEREncoder()
        buf = BitBuffer()
        
        # Extension marker
        buf.write_bit(0)
        
        # Optional fields bitmap
        buf.write_bit(0)  # ac-BarringInfo not present
        buf.write_bit(0)  # mbsfn-SubframeConfigList not present
        
        # radioResourceConfigCommon
        buf.write_bit(0)  # Extension marker
        
        # rach-ConfigCommon
        buf.write_bit(0)  # Extension marker
        buf.write_bit(0)  # preambleInfo.preamblesGroupAConfig not present
        
        # numberOfRA-Preambles
        preamble_map = {4: 0, 8: 1, 12: 2, 16: 3, 20: 4, 24: 5, 28: 6, 32: 7,
                        36: 8, 40: 9, 44: 10, 48: 11, 52: 12, 56: 13, 60: 14, 64: 15}
        buf.write_bits(preamble_map.get(self.number_of_ra_preambles, 15), 4)
        
        # powerRampingParameters
        buf.write_bits(self.power_ramping_step, 2)
        buf.write_bits((-self.preamble_initial_target_power - 120) // 2, 4)
        
        # ra-SupervisionInfo
        buf.write_bits(0, 3)  # preambleTransMax: n3
        buf.write_bits(0, 3)  # ra-ResponseWindowSize: sf2
        buf.write_bits(0, 3)  # mac-ContentionResolutionTimer: sf8
        
        buf.write_bits(self.max_harq_msg3_tx - 1, 3)
        
        # prach-Config
        buf.write_bits(0, 4)  # rootSequenceIndex (0-837)
        buf.write_bits(self.prach_config_index, 6)
        buf.write_bit(0)  # highSpeedFlag
        buf.write_bits(0, 6)  # zeroCorrelationZoneConfig
        buf.write_bits(self.prach_frequency_offset, 7)
        
        # pdsch-ConfigCommon
        buf.write_bits(0, 2)  # referenceSignalPower (-60..50)
        buf.write_bits(0, 2)  # p-b (0..3)
        
        # pusch-ConfigCommon
        buf.write_bits(self.n_sb - 1, 2)
        buf.write_bits(self.hopping_mode, 1)
        buf.write_bits(self.pusch_hopping_offset, 7)
        buf.write_bit(1 if self.enable_64qam else 0)
        
        # UL reference signals
        buf.write_bit(0)  # groupHoppingEnabled
        buf.write_bits(0, 5)  # groupAssignmentPUSCH
        buf.write_bit(0)  # sequenceHoppingEnabled
        buf.write_bits(0, 9)  # cyclicShift
        
        # pucch-ConfigCommon
        buf.write_bits(self.delta_pucch_shift, 2)
        buf.write_bits(self.n_rb_cqi, 4)
        buf.write_bits(self.n_cs_an, 3)
        buf.write_bits(self.n1_pucch_an, 11)
        
        # soundingRS-UL-ConfigCommon
        buf.write_bit(0)  # Not setup
        
        # uplinkPowerControlCommon
        buf.write_bits((-self.p0_nominal_pusch + 126), 8)
        buf.write_bits(self.alpha, 3)
        buf.write_bits((-self.p0_nominal_pucch + 127), 5)
        buf.write_bits(0, 2)  # deltaFList-PUCCH
        buf.write_bits(0, 3)  # deltaPreambleMsg3
        
        # ul-CyclicPrefixLength
        buf.write_bit(0)  # len1
        
        # ue-TimersAndConstants
        buf.write_bits(0, 3)  # t300: ms100
        buf.write_bits(0, 3)  # t301: ms100
        buf.write_bits(0, 3)  # t310: ms0
        buf.write_bits(0, 3)  # n310: n1
        buf.write_bits(0, 3)  # t311: ms1000
        buf.write_bits(0, 3)  # n311: n1
        
        # freqInfo
        buf.write_bit(0)  # ul-CarrierFreq not present
        buf.write_bit(0)  # ul-Bandwidth not present
        buf.write_bits(0, 5)  # additionalSpectrumEmission
        
        # timeAlignmentTimerCommon
        buf.write_bits(self.time_alignment_timer, 3)
        
        return buf.get_bytes()


@dataclass 
class RRCMessage:
    """Generic RRC message container"""
    message_type: RRCMessageType
    payload: bytes = b''
    transaction_id: int = 0
    
    def encode(self) -> bytes:
        """Encode complete RRC message"""
        buf = BitBuffer()
        
        # Message type indicator depends on channel
        if self.message_type.value < 10:
            # DL-CCCH or DL-DCCH
            buf.write_bits(self.message_type.value, 4)
        elif self.message_type.value < 20:
            # UL-CCCH or UL-DCCH
            buf.write_bits(self.message_type.value - 10, 4)
        else:
            # Broadcast
            buf.write_bits(self.message_type.value - 20, 4)
        
        # Transaction ID (for DL-DCCH/UL-DCCH)
        if 4 <= self.message_type.value < 10 or 12 <= self.message_type.value < 18:
            buf.write_bits(self.transaction_id, 2)
        
        # Append payload
        header = buf.get_bytes()
        return header + self.payload


class RRCConnectionRequest:
    """RRC Connection Request message"""
    
    def __init__(self, ue_identity: bytes, establishment_cause: int = 3):
        """
        Args:
            ue_identity: S-TMSI or random value (40 bits)
            establishment_cause: 0-7 (3 = mo-Data)
        """
        self.ue_identity = ue_identity
        self.establishment_cause = establishment_cause
    
    def encode(self) -> bytes:
        """Encode RRCConnectionRequest"""
        buf = BitBuffer()
        
        # UL-CCCH-Message
        buf.write_bit(0)  # c1 choice
        buf.write_bits(0, 2)  # rrcConnectionRequest
        
        # RRCConnectionRequest
        # criticalExtensions
        buf.write_bit(0)  # rrcConnectionRequest-r8
        
        # RRCConnectionRequest-r8-IEs
        # ue-Identity - CHOICE
        buf.write_bit(0)  # randomValue (40 bits)
        
        # randomValue or s-TMSI
        for byte in self.ue_identity[:5]:
            buf.write_bits(byte, 8)
        
        # establishmentCause (8 values)
        buf.write_bits(self.establishment_cause, 3)
        
        # spare (1 bit)
        buf.write_bit(0)
        
        return buf.get_bytes()


class RRCConnectionSetup:
    """RRC Connection Setup message"""
    
    def __init__(self, transaction_id: int = 0, 
                 srb1_config: Optional[Dict] = None,
                 mac_main_config: Optional[Dict] = None):
        self.transaction_id = transaction_id
        self.srb1_config = srb1_config or {}
        self.mac_main_config = mac_main_config or {}
    
    def encode(self) -> bytes:
        """Encode RRCConnectionSetup"""
        buf = BitBuffer()
        
        # DL-CCCH-Message
        buf.write_bit(0)  # c1 choice
        buf.write_bits(0, 2)  # rrcConnectionSetup
        
        # RRCConnectionSetup
        buf.write_bits(self.transaction_id, 2)  # rrc-TransactionIdentifier
        
        # criticalExtensions
        buf.write_bit(0)  # c1
        buf.write_bits(0, 3)  # rrcConnectionSetup-r8
        
        # RRCConnectionSetup-r8-IEs
        buf.write_bit(0)  # Extension marker
        buf.write_bit(0)  # nonCriticalExtension not present
        
        # radioResourceConfigDedicated
        buf.write_bit(0)  # Extension marker
        
        # Optional fields
        buf.write_bit(1)  # srb-ToAddModList present
        buf.write_bit(0)  # drb-ToAddModList not present
        buf.write_bit(0)  # drb-ToReleaseList not present
        buf.write_bit(1)  # mac-MainConfig present
        buf.write_bit(0)  # sps-Config not present
        buf.write_bit(1)  # physicalConfigDedicated present
        
        # srb-ToAddModList (1 entry for SRB1)
        buf.write_bits(0, 2)  # 1 entry
        
        # SRB-ToAddMod for SRB1
        buf.write_bit(0)  # Extension marker
        buf.write_bit(0)  # rlc-Config not present (default)
        buf.write_bit(0)  # logicalChannelConfig not present (default)
        buf.write_bits(0, 2)  # srb-Identity: SRB1
        
        # mac-MainConfig (explicit)
        buf.write_bit(0)  # explicitValue
        buf.write_bit(0)  # Extension marker
        
        # Optional fields for MAC-MainConfig
        buf.write_bit(0)  # ul-SCH-Config not present
        buf.write_bit(0)  # drx-Config not present
        buf.write_bits(0, 3)  # timeAlignmentTimerDedicated: sf500
        buf.write_bit(0)  # phr-Config not present
        
        # physicalConfigDedicated
        buf.write_bit(0)  # Extension marker
        
        # Optional fields (all not present for simplicity)
        for _ in range(10):
            buf.write_bit(0)
        
        return buf.get_bytes()


class RRCHandler:
    """
    RRC Protocol Handler
    
    Manages RRC state machine and message processing.
    Integrates with stealth features for covert operations.
    """
    
    def __init__(self, cell_config: Optional[Dict] = None,
                 stealth_mode: bool = True):
        self.cell_config = cell_config or {}
        self.stealth_mode = stealth_mode
        self.logger = logging.getLogger('RRCHandler')
        
        # State
        self.state = RRCState.IDLE
        self._lock = threading.Lock()
        
        # Connected UEs
        self.ue_contexts: Dict[int, Dict] = {}
        self.next_rnti = 0x0001
        
        # System information
        self.mib = MIB()
        self.sib1 = SIB1()
        self.sib2 = SIB2()
        
        # Message handlers
        self._handlers = {
            RRCMessageType.RRC_CONNECTION_REQUEST: self._handle_connection_request,
            RRCMessageType.RRC_CONNECTION_SETUP_COMPLETE: self._handle_setup_complete,
            RRCMessageType.SECURITY_MODE_COMPLETE: self._handle_security_complete,
            RRCMessageType.UL_INFORMATION_TRANSFER: self._handle_ul_info_transfer,
        }
    
    def configure_cell(self, mcc: str, mnc: str, 
                       tac: int, cell_id: int,
                       dl_bandwidth: int = 50):
        """Configure cell parameters"""
        with self._lock:
            self.sib1.plmn_identity_list = [PLMNIdentity(mcc=mcc, mnc=mnc)]
            self.sib1.tracking_area_code = tac
            self.sib1.cell_identity = CellIdentity(enb_id=cell_id >> 8, 
                                                   cell_id=cell_id & 0xFF)
            self.mib.dl_bandwidth = dl_bandwidth
            
            self.logger.info(f"Cell configured: MCC={mcc}, MNC={mnc}, "
                           f"TAC={tac}, CellID={cell_id}")
    
    def generate_mib(self, sfn: int) -> bytes:
        """Generate MIB for given system frame number"""
        with self._lock:
            self.mib.system_frame_number = sfn
            return self.mib.encode()
    
    def generate_sib1(self) -> bytes:
        """Generate SIB1"""
        with self._lock:
            return self.sib1.encode()
    
    def generate_sib2(self) -> bytes:
        """Generate SIB2"""
        with self._lock:
            return self.sib2.encode()
    
    def process_message(self, data: bytes, rnti: int = 0) -> Optional[bytes]:
        """
        Process incoming RRC message.
        
        Returns response message if any.
        """
        try:
            buf = BitBuffer(data)
            
            # Determine message type
            c1_choice = buf.read_bit()
            if c1_choice == 0:
                msg_type_idx = buf.read_bits(2)
                
                # Map to message type based on channel
                if rnti == 0:  # UL-CCCH
                    if msg_type_idx == 0:
                        msg_type = RRCMessageType.RRC_CONNECTION_REQUEST
                    else:
                        msg_type = RRCMessageType.RRC_CONNECTION_REESTABLISHMENT_REQUEST
                else:  # UL-DCCH
                    msg_type = RRCMessageType(msg_type_idx + 12)
                
                # Call handler
                if msg_type in self._handlers:
                    return self._handlers[msg_type](data, rnti)
            
        except Exception as e:
            self.logger.error(f"Error processing RRC message: {e}")
        
        return None
    
    def _handle_connection_request(self, data: bytes, rnti: int) -> Optional[bytes]:
        """Handle RRC Connection Request"""
        self.logger.info("Received RRC Connection Request")
        
        # Allocate C-RNTI
        with self._lock:
            c_rnti = self.next_rnti
            self.next_rnti += 1
            
            # Create UE context
            self.ue_contexts[c_rnti] = {
                'state': RRCState.IDLE,
                'created': time.time(),
                'security_configured': False,
            }
        
        # Generate Connection Setup
        setup = RRCConnectionSetup(transaction_id=0)
        response = setup.encode()
        
        self.logger.info(f"Sending RRC Connection Setup, C-RNTI={c_rnti}")
        
        # In stealth mode, add random delay
        if self.stealth_mode:
            time.sleep(np.random.uniform(0.001, 0.005))
        
        return response
    
    def _handle_setup_complete(self, data: bytes, rnti: int) -> Optional[bytes]:
        """Handle RRC Connection Setup Complete"""
        self.logger.info(f"Received RRC Connection Setup Complete, RNTI={rnti}")
        
        with self._lock:
            if rnti in self.ue_contexts:
                self.ue_contexts[rnti]['state'] = RRCState.CONNECTED
        
        # Would trigger NAS processing here
        return None
    
    def _handle_security_complete(self, data: bytes, rnti: int) -> Optional[bytes]:
        """Handle Security Mode Complete"""
        self.logger.info(f"Received Security Mode Complete, RNTI={rnti}")
        
        with self._lock:
            if rnti in self.ue_contexts:
                self.ue_contexts[rnti]['security_configured'] = True
        
        return None
    
    def _handle_ul_info_transfer(self, data: bytes, rnti: int) -> Optional[bytes]:
        """Handle UL Information Transfer (contains NAS message)"""
        self.logger.info(f"Received UL Information Transfer, RNTI={rnti}")
        
        # Extract NAS PDU and forward to NAS handler
        # (Would be implemented with NAS integration)
        
        return None
    
    def release_connection(self, rnti: int, cause: int = 0) -> bytes:
        """Generate RRC Connection Release"""
        buf = BitBuffer()
        
        # DL-DCCH-Message
        buf.write_bit(0)  # c1
        buf.write_bits(5, 4)  # rrcConnectionRelease
        
        # rrc-TransactionIdentifier
        buf.write_bits(0, 2)
        
        # criticalExtensions
        buf.write_bit(0)  # c1
        buf.write_bits(0, 2)  # rrcConnectionRelease-r8
        
        # RRCConnectionRelease-r8-IEs
        buf.write_bit(0)  # Extension marker
        buf.write_bits(cause, 2)  # releaseCause
        buf.write_bit(0)  # redirectedCarrierInfo not present
        buf.write_bit(0)  # idleModeMobilityControlInfo not present
        buf.write_bit(0)  # nonCriticalExtension not present
        
        # Update context
        with self._lock:
            if rnti in self.ue_contexts:
                del self.ue_contexts[rnti]
        
        return buf.get_bytes()
    
    def get_connected_ues(self) -> List[Dict]:
        """Get list of connected UEs"""
        with self._lock:
            return [
                {'rnti': rnti, **ctx}
                for rnti, ctx in self.ue_contexts.items()
            ]
    
    def get_cell_info(self) -> Dict:
        """Get current cell configuration"""
        with self._lock:
            return {
                'plmn': f"{self.sib1.plmn_identity_list[0].mcc}-{self.sib1.plmn_identity_list[0].mnc}",
                'tac': self.sib1.tracking_area_code,
                'cell_id': self.sib1.cell_identity.encode(),
                'bandwidth': self.mib.dl_bandwidth,
                'connected_ues': len(self.ue_contexts),
            }
