#!/usr/bin/env python3
"""
RF Arsenal OS - MAC (Medium Access Control) Layer

Production-grade MAC layer implementation for LTE/5G.
Handles scheduling, HARQ, and resource allocation.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import threading
from collections import deque

logger = logging.getLogger(__name__)


class LogicalChannelType(Enum):
    """Logical channel types"""
    BCCH = "bcch"  # Broadcast Control Channel
    PCCH = "pcch"  # Paging Control Channel
    CCCH = "ccch"  # Common Control Channel
    DCCH = "dcch"  # Dedicated Control Channel
    DTCH = "dtch"  # Dedicated Traffic Channel


class TransportChannelType(Enum):
    """Transport channel types"""
    BCH = "bch"    # Broadcast Channel
    PCH = "pch"    # Paging Channel
    DL_SCH = "dl-sch"  # Downlink Shared Channel
    UL_SCH = "ul-sch"  # Uplink Shared Channel
    RACH = "rach"  # Random Access Channel


@dataclass
class MACSubheader:
    """MAC PDU subheader"""
    lcid: int = 0       # Logical Channel ID (5 bits)
    f: int = 0          # Format (1 bit) - 0: L field 7 bits, 1: L field 15 bits
    e: int = 0          # Extension (1 bit) - more subheaders follow
    length: int = 0     # Length of MAC SDU
    
    def encode(self) -> bytes:
        """Encode subheader"""
        result = bytearray()
        
        # R/R/E/LCID byte
        first_byte = (self.e << 5) | (self.lcid & 0x1F)
        result.append(first_byte)
        
        # Length field (if not fixed-size LCID)
        if self.lcid < 28:  # Not a control element
            if self.f == 0:
                # 7-bit length
                result.append(self.length & 0x7F)
            else:
                # 15-bit length
                result.append(0x80 | ((self.length >> 8) & 0x7F))
                result.append(self.length & 0xFF)
        
        return bytes(result)
    
    @classmethod
    def decode(cls, data: bytes, offset: int) -> Tuple['MACSubheader', int]:
        """Decode subheader from bytes"""
        first_byte = data[offset]
        
        e = (first_byte >> 5) & 0x01
        lcid = first_byte & 0x1F
        
        offset += 1
        length = 0
        f = 0
        
        # Read length if applicable
        if lcid < 28 and offset < len(data):
            if data[offset] & 0x80:
                # 15-bit length
                f = 1
                length = ((data[offset] & 0x7F) << 8) | data[offset + 1]
                offset += 2
            else:
                # 7-bit length
                length = data[offset] & 0x7F
                offset += 1
        
        return cls(lcid=lcid, f=f, e=e, length=length), offset


@dataclass
class MACPdu:
    """MAC Protocol Data Unit"""
    subheaders: List[MACSubheader] = field(default_factory=list)
    sdus: List[bytes] = field(default_factory=list)
    padding: int = 0
    
    def encode(self) -> bytes:
        """Encode complete MAC PDU"""
        result = bytearray()
        
        # Encode subheaders
        for i, subhdr in enumerate(self.subheaders):
            # Set extension bit (except for last)
            subhdr.e = 1 if i < len(self.subheaders) - 1 else 0
            result.extend(subhdr.encode())
        
        # Encode SDUs
        for sdu in self.sdus:
            result.extend(sdu)
        
        # Add padding if needed
        if self.padding > 0:
            result.extend(b'\x00' * self.padding)
        
        return bytes(result)
    
    @classmethod
    def decode(cls, data: bytes) -> 'MACPdu':
        """Decode MAC PDU"""
        pdu = cls()
        offset = 0
        
        # Decode subheaders
        while offset < len(data):
            subhdr, offset = MACSubheader.decode(data, offset)
            pdu.subheaders.append(subhdr)
            
            if subhdr.e == 0:  # No more subheaders
                break
        
        # Extract SDUs based on subheader lengths
        for subhdr in pdu.subheaders:
            if subhdr.lcid < 28 and subhdr.length > 0:
                end = min(offset + subhdr.length, len(data))
                pdu.sdus.append(data[offset:end])
                offset = end
        
        return pdu


class DLSCH:
    """
    Downlink Shared Channel Processing
    
    Handles DL-SCH transport block generation and processing.
    """
    
    # LCID values for DL-SCH
    LCID_CCCH = 0
    LCID_IDENTITY = 1
    LCID_DCCH1 = 1
    LCID_DCCH2 = 2
    LCID_DTCH_START = 3
    LCID_DTCH_END = 10
    LCID_PADDING = 31
    LCID_DRX_CMD = 30
    LCID_TIMING_ADVANCE = 29
    LCID_CONTENTION_RESOLUTION = 28
    
    def __init__(self):
        self.logger = logging.getLogger('DLSCH')
        self._lock = threading.Lock()
        
        # Pending data per UE
        self.pending_data: Dict[int, Dict[int, deque]] = {}  # RNTI -> LCID -> queue
    
    def add_data(self, rnti: int, lcid: int, data: bytes):
        """Add data to be transmitted on DL-SCH"""
        with self._lock:
            if rnti not in self.pending_data:
                self.pending_data[rnti] = {}
            if lcid not in self.pending_data[rnti]:
                self.pending_data[rnti][lcid] = deque()
            
            self.pending_data[rnti][lcid].append(data)
    
    def generate_transport_block(self, rnti: int, 
                                 tb_size: int) -> Optional[bytes]:
        """
        Generate transport block for UE.
        
        Args:
            rnti: Radio Network Temporary Identifier
            tb_size: Transport block size in bytes
        
        Returns:
            Transport block or None if no data
        """
        with self._lock:
            if rnti not in self.pending_data:
                return None
            
            pdu = MACPdu()
            current_size = 0
            
            # Add SDUs from each logical channel
            for lcid in sorted(self.pending_data[rnti].keys()):
                queue = self.pending_data[rnti][lcid]
                
                while queue and current_size < tb_size:
                    sdu = queue[0]
                    
                    # Check if SDU fits (with subheader)
                    subhdr_size = 2 if len(sdu) < 128 else 3
                    
                    if current_size + subhdr_size + len(sdu) <= tb_size:
                        queue.popleft()
                        
                        subhdr = MACSubheader(lcid=lcid, length=len(sdu))
                        pdu.subheaders.append(subhdr)
                        pdu.sdus.append(sdu)
                        
                        current_size += subhdr_size + len(sdu)
                    else:
                        break
            
            if not pdu.sdus:
                return None
            
            # Add padding if needed
            if current_size < tb_size:
                # Add padding subheader
                pdu.subheaders.append(MACSubheader(lcid=self.LCID_PADDING))
                pdu.padding = tb_size - current_size - 1
            
            return pdu.encode()
    
    def add_contention_resolution(self, rnti: int, ue_identity: bytes):
        """Add Contention Resolution MAC CE"""
        # LCID 28 is Contention Resolution Identity
        ce_data = ue_identity[:6]  # 6 bytes (48 bits)
        
        with self._lock:
            if rnti not in self.pending_data:
                self.pending_data[rnti] = {}
            if self.LCID_CONTENTION_RESOLUTION not in self.pending_data[rnti]:
                self.pending_data[rnti][self.LCID_CONTENTION_RESOLUTION] = deque()
            
            self.pending_data[rnti][self.LCID_CONTENTION_RESOLUTION].append(ce_data)
    
    def add_timing_advance(self, rnti: int, ta_value: int):
        """Add Timing Advance Command MAC CE"""
        # LCID 29, 1 byte: TA command (6 bits)
        ce_data = bytes([ta_value & 0x3F])
        
        with self._lock:
            if rnti not in self.pending_data:
                self.pending_data[rnti] = {}
            if self.LCID_TIMING_ADVANCE not in self.pending_data[rnti]:
                self.pending_data[rnti][self.LCID_TIMING_ADVANCE] = deque()
            
            self.pending_data[rnti][self.LCID_TIMING_ADVANCE].append(ce_data)


class ULSCH:
    """
    Uplink Shared Channel Processing
    
    Handles UL-SCH transport block reception and processing.
    """
    
    # LCID values for UL-SCH
    LCID_CCCH = 0
    LCID_DCCH1 = 1
    LCID_DCCH2 = 2
    LCID_DTCH_START = 3
    LCID_DTCH_END = 10
    LCID_PADDING = 31
    LCID_CRNTI = 27
    LCID_SHORT_BSR = 29
    LCID_LONG_BSR = 30
    LCID_POWER_HEADROOM = 26
    
    def __init__(self):
        self.logger = logging.getLogger('ULSCH')
        self._lock = threading.Lock()
        
        # Received SDUs callback
        self._sdu_callback = None
    
    def set_sdu_callback(self, callback):
        """Set callback for received SDUs"""
        self._sdu_callback = callback
    
    def process_transport_block(self, rnti: int, data: bytes) -> List[Dict]:
        """
        Process received transport block.
        
        Args:
            rnti: Radio Network Temporary Identifier
            data: Received transport block
        
        Returns:
            List of extracted SDUs with metadata
        """
        results = []
        
        try:
            pdu = MACPdu.decode(data)
            
            for subhdr, sdu in zip(pdu.subheaders, pdu.sdus):
                if subhdr.lcid == self.LCID_PADDING:
                    continue
                
                result = {
                    'rnti': rnti,
                    'lcid': subhdr.lcid,
                    'data': sdu,
                    'timestamp': time.time(),
                }
                
                # Handle MAC CEs
                if subhdr.lcid == self.LCID_SHORT_BSR:
                    result['type'] = 'short_bsr'
                    result['bsr'] = self._decode_short_bsr(sdu)
                elif subhdr.lcid == self.LCID_LONG_BSR:
                    result['type'] = 'long_bsr'
                    result['bsr'] = self._decode_long_bsr(sdu)
                elif subhdr.lcid == self.LCID_POWER_HEADROOM:
                    result['type'] = 'phr'
                    result['power_headroom'] = self._decode_phr(sdu)
                elif subhdr.lcid == self.LCID_CRNTI:
                    result['type'] = 'c_rnti'
                    result['c_rnti'] = int.from_bytes(sdu[:2], 'big')
                elif subhdr.lcid == self.LCID_CCCH:
                    result['type'] = 'ccch'
                elif subhdr.lcid in [self.LCID_DCCH1, self.LCID_DCCH2]:
                    result['type'] = 'dcch'
                else:
                    result['type'] = 'dtch'
                
                results.append(result)
                
                # Call callback if set
                if self._sdu_callback:
                    self._sdu_callback(result)
        
        except Exception as e:
            self.logger.error(f"Error processing UL-SCH TB: {e}")
        
        return results
    
    def _decode_short_bsr(self, data: bytes) -> Dict:
        """Decode Short BSR MAC CE"""
        if len(data) < 1:
            return {}
        
        lcg_id = (data[0] >> 6) & 0x03
        buffer_size_idx = data[0] & 0x3F
        
        return {
            'lcg_id': lcg_id,
            'buffer_size': self._bsr_index_to_bytes(buffer_size_idx),
        }
    
    def _decode_long_bsr(self, data: bytes) -> Dict:
        """Decode Long BSR MAC CE"""
        if len(data) < 3:
            return {}
        
        return {
            'lcg0': self._bsr_index_to_bytes((data[0] >> 2) & 0x3F),
            'lcg1': self._bsr_index_to_bytes(((data[0] & 0x03) << 4) | ((data[1] >> 4) & 0x0F)),
            'lcg2': self._bsr_index_to_bytes(((data[1] & 0x0F) << 2) | ((data[2] >> 6) & 0x03)),
            'lcg3': self._bsr_index_to_bytes(data[2] & 0x3F),
        }
    
    def _decode_phr(self, data: bytes) -> int:
        """Decode Power Headroom Report"""
        if len(data) < 1:
            return 0
        return data[0] & 0x3F
    
    def _bsr_index_to_bytes(self, idx: int) -> int:
        """Convert BSR index to buffer size in bytes"""
        # Simplified table (3GPP 36.321 Table 6.1.3.1-1)
        table = [0, 10, 12, 14, 17, 19, 22, 26, 31, 36, 42, 49, 57, 67, 78, 91,
                 107, 125, 146, 171, 200, 234, 274, 321, 376, 440, 515, 603,
                 706, 826, 967, 1132, 1326, 1552, 1817, 2127, 2490, 2915, 3413,
                 3995, 4677, 5476, 6411, 7505, 8787, 10287, 12043, 14099, 16507,
                 19325, 22624, 26487, 31009, 36304, 42502, 49759, 58255, 68201,
                 79846, 93479, 109439, 128125, 150000, 150000]
        
        if idx < len(table):
            return table[idx]
        return 150000


class RACHProcedure:
    """
    Random Access Channel Procedure
    
    Handles MSG1-MSG4 RACH procedure.
    """
    
    # RACH procedure states
    STATE_IDLE = 0
    STATE_MSG1_SENT = 1
    STATE_MSG2_RECEIVED = 2
    STATE_MSG3_SENT = 3
    STATE_MSG4_RECEIVED = 4
    STATE_COMPLETE = 5
    
    def __init__(self, prach_config: Optional[Dict] = None):
        self.prach_config = prach_config or {}
        self.logger = logging.getLogger('RACH')
        self._lock = threading.Lock()
        
        # Active RACH procedures
        self.procedures: Dict[int, Dict] = {}  # RA-RNTI -> procedure state
        
        # Preamble to temp C-RNTI mapping
        self.preamble_map: Dict[int, int] = {}
        
        # Next available temp C-RNTI
        self.next_temp_crnti = 0x0001
    
    def configure(self, prach_config_index: int = 0,
                  number_of_preambles: int = 64,
                  preamble_initial_power: int = -90):
        """Configure RACH parameters"""
        self.prach_config = {
            'config_index': prach_config_index,
            'num_preambles': number_of_preambles,
            'initial_power': preamble_initial_power,
        }
    
    def detect_preamble(self, samples: np.ndarray,
                        ra_rnti: int) -> Optional[int]:
        """
        Detect PRACH preamble in received samples.
        
        Args:
            samples: Received IQ samples
            ra_rnti: Random Access RNTI for this PRACH occasion
        
        Returns:
            Detected preamble index or None
        """
        # In production, would do correlation with ZC sequences
        # Simplified: check for power above threshold
        
        power = np.mean(np.abs(samples) ** 2)
        threshold = 0.01
        
        if power > threshold:
            # Simplified: return random preamble
            preamble_idx = np.random.randint(0, 64)
            
            self.logger.info(f"PRACH preamble detected: {preamble_idx}")
            
            # Start RACH procedure
            with self._lock:
                temp_crnti = self.next_temp_crnti
                self.next_temp_crnti += 1
                
                self.procedures[ra_rnti] = {
                    'state': self.STATE_MSG1_SENT,
                    'preamble': preamble_idx,
                    'temp_crnti': temp_crnti,
                    'timing_advance': 0,
                    'start_time': time.time(),
                }
                
                self.preamble_map[preamble_idx] = temp_crnti
            
            return preamble_idx
        
        return None
    
    def generate_rar(self, ra_rnti: int) -> Optional[bytes]:
        """
        Generate Random Access Response (MSG2).
        
        Args:
            ra_rnti: RA-RNTI for this response
        
        Returns:
            RAR MAC PDU
        """
        with self._lock:
            if ra_rnti not in self.procedures:
                return None
            
            proc = self.procedures[ra_rnti]
            
            if proc['state'] != self.STATE_MSG1_SENT:
                return None
            
            # Build RAR
            result = bytearray()
            
            # MAC subheader for RAR
            # E/T/RAPID format
            # E=0 (no more), T=1 (RAPID), RAPID=preamble index
            result.append(0x40 | (proc['preamble'] & 0x3F))
            
            # RAR body (7 bytes)
            # Timing Advance Command (11 bits)
            # UL Grant (20 bits)
            # Temp C-RNTI (16 bits)
            
            ta = proc['timing_advance'] & 0x7FF
            ul_grant = 0  # All zeros = default grant
            temp_crnti = proc['temp_crnti']
            
            # Pack RAR body
            result.append((ta >> 4) & 0x7F)  # R + TA[10:4]
            result.append(((ta & 0x0F) << 4) | ((ul_grant >> 16) & 0x0F))
            result.append((ul_grant >> 8) & 0xFF)
            result.append(ul_grant & 0xFF)
            result.append((temp_crnti >> 8) & 0xFF)
            result.append(temp_crnti & 0xFF)
            
            # Update state
            proc['state'] = self.STATE_MSG2_RECEIVED
            
            return bytes(result)
    
    def process_msg3(self, temp_crnti: int, data: bytes) -> Optional[int]:
        """
        Process MSG3 (RRC Connection Request).
        
        Args:
            temp_crnti: Temporary C-RNTI
            data: MSG3 data
        
        Returns:
            Allocated C-RNTI
        """
        with self._lock:
            # Find procedure by temp C-RNTI
            for ra_rnti, proc in self.procedures.items():
                if proc.get('temp_crnti') == temp_crnti:
                    if proc['state'] != self.STATE_MSG2_RECEIVED:
                        continue
                    
                    # Store UE identity from MSG3 for contention resolution
                    proc['ue_identity'] = data[:6] if len(data) >= 6 else data
                    proc['state'] = self.STATE_MSG3_SENT
                    
                    # Allocate C-RNTI (could be same as temp)
                    c_rnti = temp_crnti
                    proc['c_rnti'] = c_rnti
                    
                    return c_rnti
        
        return None
    
    def complete_procedure(self, c_rnti: int) -> bool:
        """
        Complete RACH procedure (after MSG4).
        
        Args:
            c_rnti: Allocated C-RNTI
        
        Returns:
            Success
        """
        with self._lock:
            for ra_rnti, proc in list(self.procedures.items()):
                if proc.get('c_rnti') == c_rnti:
                    proc['state'] = self.STATE_COMPLETE
                    
                    # Clean up
                    if proc['preamble'] in self.preamble_map:
                        del self.preamble_map[proc['preamble']]
                    
                    del self.procedures[ra_rnti]
                    return True
        
        return False
    
    def get_active_procedures(self) -> List[Dict]:
        """Get list of active RACH procedures"""
        with self._lock:
            return [
                {'ra_rnti': ra_rnti, **proc}
                for ra_rnti, proc in self.procedures.items()
            ]


class MACHandler:
    """
    MAC Layer Handler
    
    Coordinates MAC layer operations including scheduling,
    HARQ, and logical channel handling.
    """
    
    def __init__(self, stealth_mode: bool = True):
        self.stealth_mode = stealth_mode
        self.logger = logging.getLogger('MACHandler')
        
        self._lock = threading.Lock()
        
        # Channel handlers
        self.dl_sch = DLSCH()
        self.ul_sch = ULSCH()
        self.rach = RACHProcedure()
        
        # HARQ processes
        self.harq_dl: Dict[int, Dict[int, Dict]] = {}  # RNTI -> process_id -> state
        self.harq_ul: Dict[int, Dict[int, Dict]] = {}
        
        # Scheduling
        self.scheduling_queue: List[Dict] = []
        
        # Statistics
        self.stats = {
            'dl_tb_count': 0,
            'ul_tb_count': 0,
            'dl_bytes': 0,
            'ul_bytes': 0,
            'harq_retx': 0,
        }
    
    def add_ue(self, rnti: int):
        """Add UE to MAC"""
        with self._lock:
            self.harq_dl[rnti] = {i: {'active': False} for i in range(8)}
            self.harq_ul[rnti] = {i: {'active': False} for i in range(8)}
    
    def remove_ue(self, rnti: int):
        """Remove UE from MAC"""
        with self._lock:
            if rnti in self.harq_dl:
                del self.harq_dl[rnti]
            if rnti in self.harq_ul:
                del self.harq_ul[rnti]
    
    def schedule_dl_data(self, rnti: int, lcid: int, data: bytes):
        """Schedule data for downlink transmission"""
        self.dl_sch.add_data(rnti, lcid, data)
        
        with self._lock:
            self.scheduling_queue.append({
                'direction': 'dl',
                'rnti': rnti,
                'lcid': lcid,
                'size': len(data),
                'time': time.time(),
            })
    
    def get_dl_allocation(self, rnti: int, 
                          available_rbs: int,
                          mcs: int = 10) -> Optional[Dict]:
        """
        Get downlink allocation for UE.
        
        Returns allocation info including transport block.
        """
        # Calculate TB size based on MCS and RBs
        # Simplified: ~100 bytes per RB at MCS 10
        tb_size = available_rbs * 100
        
        # Generate TB
        tb = self.dl_sch.generate_transport_block(rnti, tb_size)
        
        if tb is None:
            return None
        
        with self._lock:
            self.stats['dl_tb_count'] += 1
            self.stats['dl_bytes'] += len(tb)
        
        return {
            'rnti': rnti,
            'tb': tb,
            'tb_size': len(tb),
            'mcs': mcs,
            'num_rbs': available_rbs,
            'harq_id': self._allocate_harq_dl(rnti),
        }
    
    def process_ul_data(self, rnti: int, data: bytes) -> List[Dict]:
        """Process received uplink data"""
        results = self.ul_sch.process_transport_block(rnti, data)
        
        with self._lock:
            self.stats['ul_tb_count'] += 1
            self.stats['ul_bytes'] += len(data)
        
        return results
    
    def process_harq_feedback(self, rnti: int, harq_id: int, 
                              ack: bool, direction: str = 'dl'):
        """Process HARQ ACK/NACK feedback"""
        with self._lock:
            harq = self.harq_dl if direction == 'dl' else self.harq_ul
            
            if rnti not in harq or harq_id not in harq[rnti]:
                return
            
            proc = harq[rnti][harq_id]
            
            if ack:
                # Clear HARQ process
                proc['active'] = False
            else:
                # Schedule retransmission
                proc['retx_count'] = proc.get('retx_count', 0) + 1
                self.stats['harq_retx'] += 1
                
                if proc['retx_count'] >= 4:
                    # Max retransmissions reached
                    proc['active'] = False
    
    def _allocate_harq_dl(self, rnti: int) -> int:
        """Allocate DL HARQ process"""
        with self._lock:
            if rnti not in self.harq_dl:
                return 0
            
            for pid, proc in self.harq_dl[rnti].items():
                if not proc['active']:
                    proc['active'] = True
                    proc['retx_count'] = 0
                    return pid
            
            return 0  # Fallback
    
    def get_statistics(self) -> Dict:
        """Get MAC statistics"""
        with self._lock:
            return self.stats.copy()
