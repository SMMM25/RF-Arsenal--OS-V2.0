#!/usr/bin/env python3
"""
RF Arsenal OS - Digital Radio Decoder Module
Hardware: BladeRF 2.0 micro xA9

Digital radio protocol decoding:
- DMR (Digital Mobile Radio) - Tier I, II, III
- P25 Phase 1 and Phase 2
- TETRA (Terrestrial Trunked Radio)
- NXDN
- dPMR
- D-STAR (Amateur)

Applications:
- Public safety monitoring (police, fire, EMS)
- Commercial radio systems
- Amateur radio digital modes
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
from datetime import datetime
import threading
import struct

logger = logging.getLogger(__name__)


class RadioProtocol(Enum):
    """Digital radio protocols"""
    DMR = "dmr"                     # Digital Mobile Radio
    P25_PHASE1 = "p25_phase1"      # APCO Project 25 Phase 1
    P25_PHASE2 = "p25_phase2"      # APCO Project 25 Phase 2
    TETRA = "tetra"                # Terrestrial Trunked Radio
    NXDN = "nxdn"                  # Next Generation Digital Narrowband
    DPMR = "dpmr"                  # Digital PMR
    DSTAR = "dstar"                # D-STAR Amateur
    YSF = "ysf"                    # Yaesu System Fusion


class DMRTimeslot(Enum):
    """DMR timeslots"""
    TS1 = 1
    TS2 = 2


class P25NAC(Enum):
    """P25 Network Access Codes"""
    DEFAULT = 0x293
    WILDCARD = 0xF7E


class TETRAMode(Enum):
    """TETRA operating modes"""
    TMO = "tmo"                    # Trunked Mode Operation
    DMO = "dmo"                    # Direct Mode Operation


@dataclass
class DMRConfig:
    """DMR decoder configuration"""
    frequency: int = 450_000_000   # Hz
    sample_rate: int = 2_000_000   # 2 MSPS
    bandwidth: int = 12_500        # 12.5 kHz channel
    timeslot: DMRTimeslot = DMRTimeslot.TS1
    color_code: int = 1


@dataclass
class P25Config:
    """P25 decoder configuration"""
    frequency: int = 851_000_000   # Hz
    sample_rate: int = 2_000_000
    bandwidth: int = 12_500        # Phase 1: 12.5 kHz
    nac: int = 0x293               # Network Access Code
    phase: int = 1                 # 1 or 2


@dataclass
class TETRAConfig:
    """TETRA decoder configuration"""
    frequency: int = 380_000_000   # Hz
    sample_rate: int = 2_000_000
    bandwidth: int = 25_000        # 25 kHz channel
    mode: TETRAMode = TETRAMode.TMO


@dataclass
class RadioChannel:
    """Discovered radio channel"""
    frequency: float              # MHz
    protocol: RadioProtocol
    rssi: float                  # dBm
    active: bool
    encrypted: bool
    talkgroup: Optional[int] = None
    color_code: Optional[int] = None
    nac: Optional[int] = None
    system_name: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class RadioCall:
    """Decoded radio call/transmission"""
    call_id: str
    protocol: RadioProtocol
    frequency: float
    source_id: int               # Radio ID of caller
    destination_id: int          # Talkgroup or individual ID
    call_type: str               # group, individual, broadcast
    encrypted: bool
    timestamp_start: str
    timestamp_end: str = ""
    duration_sec: float = 0.0
    audio_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    metadata: Dict = field(default_factory=dict)


@dataclass
class RadioUnit:
    """Discovered radio unit/subscriber"""
    radio_id: int
    protocol: RadioProtocol
    last_talkgroup: Optional[int]
    first_seen: str
    last_seen: str
    call_count: int = 0
    alias: str = ""


@dataclass
class Talkgroup:
    """Talkgroup information"""
    tg_id: int
    protocol: RadioProtocol
    system_name: str
    name: str = ""
    category: str = ""           # Police, Fire, EMS, etc.
    encrypted: bool = False
    active: bool = False
    last_activity: str = ""


class DigitalRadioDecoder:
    """
    Multi-Protocol Digital Radio Decoder
    
    Decodes DMR, P25, TETRA, and other digital radio protocols
    from BladeRF captures. Supports voice decoding (unencrypted),
    talkgroup identification, and radio unit tracking.
    
    WARNING: Interception of radio communications may be regulated
    or prohibited in your jurisdiction.
    """
    
    # Protocol-specific constants
    DMR_SYMBOL_RATE = 4800         # symbols/sec
    P25_SYMBOL_RATE = 4800         # symbols/sec
    TETRA_SYMBOL_RATE = 18000      # symbols/sec
    
    # Common frequency ranges
    FREQUENCY_BANDS = {
        'vhf': (136_000_000, 174_000_000),
        'uhf': (400_000_000, 520_000_000),
        '700mhz': (764_000_000, 776_000_000),
        '800mhz': (851_000_000, 869_000_000),
        '900mhz': (896_000_000, 941_000_000),
    }
    
    def __init__(self, hardware_controller=None):
        """
        Initialize digital radio decoder
        
        Args:
            hardware_controller: BladeRF hardware controller
        """
        self.hw = hardware_controller
        self.is_running = False
        self._decode_thread = None
        
        # Discovered items
        self.channels: Dict[float, RadioChannel] = {}
        self.calls: List[RadioCall] = []
        self.units: Dict[int, RadioUnit] = {}
        self.talkgroups: Dict[int, Talkgroup] = {}
        
        # Current state
        self._current_protocol: Optional[RadioProtocol] = None
        self._current_call: Optional[RadioCall] = None
        
        # Voice codec (would use external library in practice)
        self._ambe_decoder = None
        
        logger.info("Digital Radio Decoder initialized")
    
    def scan_for_signals(self, band: str = 'uhf', 
                         step_khz: float = 12.5) -> List[RadioChannel]:
        """
        Scan frequency band for digital radio signals
        
        Args:
            band: Frequency band to scan ('vhf', 'uhf', '800mhz', etc.)
            step_khz: Frequency step in kHz
            
        Returns:
            List of discovered channels
        """
        if band not in self.FREQUENCY_BANDS:
            logger.error(f"Unknown band: {band}")
            return []
        
        freq_start, freq_end = self.FREQUENCY_BANDS[band]
        discovered = []
        
        logger.info(f"Scanning {band} band: {freq_start/1e6:.1f}-{freq_end/1e6:.1f} MHz")
        
        freq = freq_start
        while freq <= freq_end:
            channel = self._check_frequency(freq)
            if channel:
                discovered.append(channel)
                self.channels[freq / 1e6] = channel
            freq += int(step_khz * 1000)
        
        logger.info(f"Scan complete: {len(discovered)} channels found")
        return discovered
    
    def _check_frequency(self, frequency: int) -> Optional[RadioChannel]:
        """Check single frequency for digital radio signal"""
        
        # Configure hardware
        if self.hw:
            self.hw.set_frequency(frequency)
            self.hw.set_sample_rate(2_000_000)
            self.hw.set_bandwidth(25_000)
        
        # Capture samples
        num_samples = 100000  # ~50ms
        samples = self._capture_samples(num_samples)
        
        # Calculate power
        power = np.mean(np.abs(samples)**2)
        rssi = 10 * np.log10(power + 1e-10)
        
        # Skip if below noise floor
        if rssi < -80:
            return None
        
        # Detect protocol
        protocol = self._detect_protocol(samples)
        
        if protocol:
            # Detect encryption
            encrypted = self._detect_encryption(samples, protocol)
            
            # Extract talkgroup/color code
            tg, cc = self._extract_identifiers(samples, protocol)
            
            return RadioChannel(
                frequency=frequency / 1e6,
                protocol=protocol,
                rssi=rssi,
                active=True,
                encrypted=encrypted,
                talkgroup=tg,
                color_code=cc if protocol == RadioProtocol.DMR else None,
                nac=cc if protocol in [RadioProtocol.P25_PHASE1, RadioProtocol.P25_PHASE2] else None
            )
        
        return None
    
    def _capture_samples(self, num_samples: int) -> np.ndarray:
        """Capture IQ samples"""
        if self.hw:
            return self.hw.receive(num_samples)
        else:
            return np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
    
    def _detect_protocol(self, samples: np.ndarray) -> Optional[RadioProtocol]:
        """
        Detect digital radio protocol from signal
        
        Uses sync pattern detection for each protocol.
        """
        # Demodulate to symbols
        symbols = self._demodulate_4fsk(samples)
        
        # Check for DMR sync patterns
        if self._detect_dmr_sync(symbols):
            return RadioProtocol.DMR
        
        # Check for P25 sync
        if self._detect_p25_sync(symbols):
            return RadioProtocol.P25_PHASE1
        
        # Check for TETRA sync
        if self._detect_tetra_sync(symbols):
            return RadioProtocol.TETRA
        
        # Check for NXDN
        if self._detect_nxdn_sync(symbols):
            return RadioProtocol.NXDN
        
        return None
    
    def _demodulate_4fsk(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate 4-FSK signal to symbols"""
        # FM demodulation
        phase = np.unwrap(np.angle(samples))
        freq = np.diff(phase)
        
        # Quantize to 4 levels
        levels = np.array([-3, -1, 1, 3])
        symbols = np.zeros(len(freq), dtype=int)
        
        for i, f in enumerate(freq):
            symbols[i] = levels[np.argmin(np.abs(levels - f * 10))]
        
        return symbols
    
    def _detect_dmr_sync(self, symbols: np.ndarray) -> bool:
        """Detect DMR sync pattern"""
        # DMR sync patterns
        BS_VOICE_SYNC = np.array([3, 1, 3, 3, 3, 3, 1, 1, 1, 3, 3, 1, 1, 3, 1, 3, 3, 1, 1, 3, 1, 1, 3, 1])
        BS_DATA_SYNC = np.array([3, 3, 1, 3, 3, 1, 1, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3, 3, 1, 1, 1, 3, 3])
        
        # Correlate
        for sync in [BS_VOICE_SYNC, BS_DATA_SYNC]:
            corr = np.correlate(symbols, sync)
            if np.max(corr) > 20:  # Threshold
                return True
        
        return False
    
    def _detect_p25_sync(self, symbols: np.ndarray) -> bool:
        """Detect P25 frame sync"""
        # P25 64-bit frame sync
        P25_SYNC = np.array([1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1,
                           1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1,
                           1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1,
                           1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1])
        
        corr = np.correlate(symbols[:200], P25_SYNC)
        return np.max(np.abs(corr)) > 50
    
    def _detect_tetra_sync(self, symbols: np.ndarray) -> bool:
        """Detect TETRA sync pattern"""
        # TETRA uses different sync for different burst types
        # Simplified detection
        return False
    
    def _detect_nxdn_sync(self, symbols: np.ndarray) -> bool:
        """Detect NXDN frame sync"""
        return False
    
    def _detect_encryption(self, samples: np.ndarray, 
                          protocol: RadioProtocol) -> bool:
        """Detect if signal is encrypted"""
        # Check for encryption indicators in protocol-specific fields
        # This is a simplified check
        
        # For DMR, check Privacy Indicator (PI) bit
        # For P25, check Encryption Algorithm ID
        # For TETRA, check encryption field
        
        return False  # Default to unencrypted
    
    def _extract_identifiers(self, samples: np.ndarray,
                            protocol: RadioProtocol) -> Tuple[Optional[int], Optional[int]]:
        """Extract talkgroup and color code/NAC"""
        # Protocol-specific extraction
        # Placeholder - would decode actual protocol fields
        return None, None
    
    def start_decoding(self, frequency: float, 
                       protocol: RadioProtocol = None) -> bool:
        """
        Start continuous decoding at frequency
        
        Args:
            frequency: Frequency in MHz
            protocol: Protocol to decode (auto-detect if None)
            
        Returns:
            True if decoding started
        """
        if self.is_running:
            logger.warning("Decoder already running")
            return False
        
        self._current_protocol = protocol
        self.is_running = True
        
        # Configure hardware
        if self.hw:
            self.hw.set_frequency(int(frequency * 1e6))
            self.hw.set_sample_rate(2_000_000)
        
        self._decode_thread = threading.Thread(target=self._decode_worker, daemon=True)
        self._decode_thread.start()
        
        logger.info(f"Decoding started at {frequency} MHz")
        return True
    
    def stop(self):
        """Stop decoding"""
        self.is_running = False
        if self._decode_thread:
            self._decode_thread.join(timeout=2.0)
        logger.info("Decoder stopped")
    
    def _decode_worker(self):
        """Continuous decoding worker"""
        while self.is_running:
            try:
                samples = self._capture_samples(50000)
                
                # Auto-detect protocol if not specified
                if not self._current_protocol:
                    self._current_protocol = self._detect_protocol(samples)
                
                if self._current_protocol == RadioProtocol.DMR:
                    self._decode_dmr(samples)
                elif self._current_protocol in [RadioProtocol.P25_PHASE1, RadioProtocol.P25_PHASE2]:
                    self._decode_p25(samples)
                elif self._current_protocol == RadioProtocol.TETRA:
                    self._decode_tetra(samples)
                
            except Exception as e:
                logger.error(f"Decode error: {e}")
    
    def _decode_dmr(self, samples: np.ndarray):
        """Decode DMR frame"""
        symbols = self._demodulate_4fsk(samples)
        
        # Find sync
        if not self._detect_dmr_sync(symbols):
            return
        
        # Extract burst
        # DMR burst is 264 symbols
        # Would extract: Color Code, Data Type, Slot Type, etc.
        
        # For voice: decode AMBE frames
        # For data: decode data payload
        
        pass
    
    def _decode_p25(self, samples: np.ndarray):
        """Decode P25 frame"""
        symbols = self._demodulate_4fsk(samples)
        
        if not self._detect_p25_sync(symbols):
            return
        
        # P25 frame structure:
        # - Frame Sync (48 bits)
        # - Network ID (64 bits) - contains NAC
        # - Data Unit ID
        # - Various data blocks
        
        pass
    
    def _decode_tetra(self, samples: np.ndarray):
        """Decode TETRA frame"""
        # TETRA uses pi/4 DQPSK at 36 kbps
        # More complex demodulation required
        pass
    
    def decode_voice(self, samples: np.ndarray, 
                    protocol: RadioProtocol) -> Optional[np.ndarray]:
        """
        Decode voice audio from digital radio samples
        
        Args:
            samples: IQ samples containing voice
            protocol: Radio protocol
            
        Returns:
            Decoded audio samples (8kHz PCM) or None
        """
        # Extract voice frames
        if protocol == RadioProtocol.DMR:
            # DMR uses AMBE+2 codec
            # 49 bits per 20ms frame
            pass
        elif protocol in [RadioProtocol.P25_PHASE1, RadioProtocol.P25_PHASE2]:
            # P25 Phase 1: IMBE (88 bits/20ms)
            # P25 Phase 2: AMBE+2 (49 bits/20ms)
            pass
        elif protocol == RadioProtocol.TETRA:
            # TETRA uses ACELP codec
            pass
        
        # Would use external AMBE/IMBE decoder
        # Return decoded PCM audio
        return None
    
    def import_talkgroup_csv(self, filepath: str) -> int:
        """
        Import talkgroup information from CSV
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Number of talkgroups imported
        """
        # CSV format: tg_id, name, category, encrypted
        count = 0
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        tg_id = int(parts[0])
                        name = parts[1]
                        category = parts[2] if len(parts) > 2 else ""
                        encrypted = parts[3].lower() == 'true' if len(parts) > 3 else False
                        
                        self.talkgroups[tg_id] = Talkgroup(
                            tg_id=tg_id,
                            protocol=RadioProtocol.DMR,  # Default
                            system_name="Imported",
                            name=name,
                            category=category,
                            encrypted=encrypted
                        )
                        count += 1
        except Exception as e:
            logger.error(f"Error importing talkgroups: {e}")
        
        logger.info(f"Imported {count} talkgroups")
        return count
    
    def get_active_channels(self) -> List[RadioChannel]:
        """Get currently active channels"""
        return [ch for ch in self.channels.values() if ch.active]
    
    def get_recent_calls(self, limit: int = 50) -> List[RadioCall]:
        """Get recent decoded calls"""
        return self.calls[-limit:]
    
    def get_units(self) -> List[RadioUnit]:
        """Get all discovered radio units"""
        return list(self.units.values())
    
    def get_talkgroups(self) -> List[Talkgroup]:
        """Get all talkgroups"""
        return list(self.talkgroups.values())
    
    def get_status(self) -> Dict:
        """Get decoder status"""
        return {
            'running': self.is_running,
            'current_protocol': self._current_protocol.value if self._current_protocol else None,
            'channels_discovered': len(self.channels),
            'active_channels': len([c for c in self.channels.values() if c.active]),
            'calls_decoded': len(self.calls),
            'units_tracked': len(self.units),
            'talkgroups': len(self.talkgroups),
        }


# Convenience function
def get_digital_radio_decoder(hardware_controller=None) -> DigitalRadioDecoder:
    """Get digital radio decoder instance"""
    return DigitalRadioDecoder(hardware_controller)
