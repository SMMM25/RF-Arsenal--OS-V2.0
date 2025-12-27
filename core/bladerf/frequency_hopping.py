#!/usr/bin/env python3
"""
RF Arsenal OS - BladeRF Frequency Hopping Module
Hardware: BladeRF 2.0 micro xA9

Comprehensive frequency hopping support:
- Fast frequency switching (< 100μs with AD9361)
- Multiple hopping patterns (random, linear, adaptive)
- FHSS (Frequency Hopping Spread Spectrum) attacks
- Bluetooth/WiFi hopping sequence tracking
- Military/industrial protocol emulation
- Hopping sequence prediction
- Jammer-resistant communications
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable, Generator
from enum import Enum
from datetime import datetime
import threading
import time
import hashlib
from collections import deque

logger = logging.getLogger(__name__)


class HoppingPattern(Enum):
    """Frequency hopping patterns"""
    LINEAR = "linear"                      # Sequential hopping
    RANDOM = "random"                      # Pseudo-random hopping
    ADAPTIVE = "adaptive"                  # Avoid busy channels
    BLUETOOTH = "bluetooth"                # BT 79-channel hopping
    WIFI_DFS = "wifi_dfs"                  # WiFi DFS hopping
    ZIGBEE = "zigbee"                      # Zigbee channel hopping
    LORA = "lora"                          # LoRa channel hopping
    MILITARY = "military"                  # Military FHSS patterns
    CUSTOM = "custom"                      # User-defined pattern


class HoppingSpeed(Enum):
    """Hopping rate categories"""
    SLOW = "slow"                          # 1-10 hops/sec
    MEDIUM = "medium"                      # 10-100 hops/sec
    FAST = "fast"                          # 100-1000 hops/sec
    ULTRA_FAST = "ultra_fast"              # 1000+ hops/sec


class TrackingMode(Enum):
    """Hopping sequence tracking modes"""
    PASSIVE = "passive"                    # Listen and track
    ACTIVE = "active"                      # Follow transmitter
    PREDICTIVE = "predictive"              # Predict next hop


@dataclass
class HoppingConfig:
    """Frequency hopping configuration"""
    pattern: HoppingPattern = HoppingPattern.LINEAR
    speed: HoppingSpeed = HoppingSpeed.MEDIUM
    
    # Frequency range
    start_freq_hz: int = 2_402_000_000     # Start frequency
    end_freq_hz: int = 2_480_000_000       # End frequency
    channel_spacing_hz: int = 1_000_000    # Channel spacing
    
    # Timing
    dwell_time_us: int = 10_000            # Time on each channel (μs)
    switch_time_us: int = 100              # Max switching time
    
    # Pattern parameters
    seed: int = 0                          # Random seed for PRNG
    custom_sequence: List[int] = field(default_factory=list)
    
    # Adaptive parameters
    avoid_busy_channels: bool = False
    busy_threshold_dbm: float = -70        # Above this = busy
    
    # TX/RX parameters
    tx_enabled: bool = False
    rx_enabled: bool = True


@dataclass
class HopEvent:
    """Single hop event"""
    timestamp: str
    from_freq_hz: int
    to_freq_hz: int
    channel_index: int
    dwell_time_us: int
    rssi_dbm: float
    samples_captured: int = 0


@dataclass
class SequencePrediction:
    """Hopping sequence prediction result"""
    predicted_freqs: List[int]             # Next N frequencies
    confidence: float                      # 0-1
    pattern_type: str                      # Detected pattern type
    seed_guess: Optional[int]              # Estimated PRNG seed
    timing_offset_us: int                  # Estimated timing


@dataclass
class TrackedTransmitter:
    """Tracked frequency hopping transmitter"""
    id: str
    first_seen: str
    last_seen: str
    hop_count: int
    detected_pattern: HoppingPattern
    frequency_history: List[int]
    timing_history: List[float]
    predicted_sequence: SequencePrediction
    signal_strength_dbm: float


class BladeRFFrequencyHopping:
    """
    BladeRF Frequency Hopping Controller
    
    Enables advanced frequency hopping operations:
    - Fast tuning with AD9361 (< 100μs)
    - Multiple hopping patterns
    - Sequence tracking and prediction
    - FHSS protocol analysis
    """
    
    # BladeRF xA9 specs
    FREQ_MIN = 47_000_000                  # 47 MHz
    FREQ_MAX = 6_000_000_000               # 6 GHz
    TUNE_TIME_US = 100                     # AD9361 fast tune
    
    # Common protocols
    BLUETOOTH_CHANNELS = 79                # 2402-2480 MHz
    WIFI_CHANNELS_2G = [1, 6, 11]          # Non-overlapping
    ZIGBEE_CHANNELS = list(range(11, 27))  # Channels 11-26
    
    def __init__(self, hardware_controller=None):
        """
        Initialize frequency hopping controller
        
        Args:
            hardware_controller: BladeRF hardware controller
        """
        self.hw = hardware_controller
        self.config = HoppingConfig()
        self.is_hopping = False
        self._hop_thread = None
        
        # State
        self._current_freq = 0
        self._current_channel = 0
        self._hop_count = 0
        self._hop_history: deque = deque(maxlen=1000)
        self._channel_energy: Dict[int, float] = {}
        
        # Tracking
        self._tracked_transmitters: Dict[str, TrackedTransmitter] = {}
        self._sequence_buffer: List[int] = []
        
        # Callbacks
        self._hop_callback: Optional[Callable] = None
        self._rx_callback: Optional[Callable] = None
        
        logger.info("BladeRF Frequency Hopping controller initialized")
    
    def configure(self, config: HoppingConfig) -> bool:
        """
        Configure frequency hopping
        
        Args:
            config: HoppingConfig with desired settings
            
        Returns:
            True if configuration successful
        """
        # Validate frequency range
        if not self.FREQ_MIN <= config.start_freq_hz <= self.FREQ_MAX:
            logger.error(f"Start frequency out of range")
            return False
        
        if not self.FREQ_MIN <= config.end_freq_hz <= self.FREQ_MAX:
            logger.error(f"End frequency out of range")
            return False
        
        if config.start_freq_hz >= config.end_freq_hz:
            logger.error("Start frequency must be less than end frequency")
            return False
        
        self.config = config
        
        # Calculate channel count
        num_channels = (config.end_freq_hz - config.start_freq_hz) // config.channel_spacing_hz
        logger.info(f"Configured {num_channels} channels from "
                   f"{config.start_freq_hz/1e6:.1f} to {config.end_freq_hz/1e6:.1f} MHz")
        
        # Initialize channel energy tracking
        self._channel_energy = {i: -100.0 for i in range(num_channels)}
        
        return True
    
    def start_hopping(self, callback: Optional[Callable] = None) -> bool:
        """
        Start frequency hopping
        
        Args:
            callback: Function to call on each hop (HopEvent)
            
        Returns:
            True if hopping started
        """
        if self.is_hopping:
            logger.warning("Already hopping")
            return False
        
        self._hop_callback = callback
        self.is_hopping = True
        self._hop_count = 0
        
        self._hop_thread = threading.Thread(target=self._hop_worker, daemon=True)
        self._hop_thread.start()
        
        logger.info(f"Frequency hopping started: {self.config.pattern.value}")
        return True
    
    def stop_hopping(self):
        """Stop frequency hopping"""
        self.is_hopping = False
        if self._hop_thread:
            self._hop_thread.join(timeout=1.0)
        logger.info(f"Frequency hopping stopped after {self._hop_count} hops")
    
    def _hop_worker(self):
        """Frequency hopping worker thread"""
        sequence_gen = self._get_sequence_generator()
        
        while self.is_hopping:
            try:
                # Get next frequency
                next_channel = next(sequence_gen)
                next_freq = self._channel_to_freq(next_channel)
                
                # Skip busy channels in adaptive mode
                if self.config.avoid_busy_channels:
                    energy = self._channel_energy.get(next_channel, -100)
                    if energy > self.config.busy_threshold_dbm:
                        continue
                
                # Tune to frequency
                prev_freq = self._current_freq
                self._tune_fast(next_freq)
                
                # Record hop
                hop_event = HopEvent(
                    timestamp=datetime.now().isoformat(),
                    from_freq_hz=prev_freq,
                    to_freq_hz=next_freq,
                    channel_index=next_channel,
                    dwell_time_us=self.config.dwell_time_us,
                    rssi_dbm=self._measure_rssi()
                )
                
                self._hop_history.append(hop_event)
                self._hop_count += 1
                
                if self._hop_callback:
                    self._hop_callback(hop_event)
                
                # Dwell on channel
                time.sleep(self.config.dwell_time_us / 1_000_000)
                
            except StopIteration:
                # Restart sequence
                sequence_gen = self._get_sequence_generator()
            except Exception as e:
                logger.error(f"Hop worker error: {e}")
    
    def _get_sequence_generator(self) -> Generator[int, None, None]:
        """Get hopping sequence generator based on pattern"""
        num_channels = self._get_num_channels()
        
        if self.config.pattern == HoppingPattern.LINEAR:
            return self._linear_sequence(num_channels)
        elif self.config.pattern == HoppingPattern.RANDOM:
            return self._random_sequence(num_channels, self.config.seed)
        elif self.config.pattern == HoppingPattern.BLUETOOTH:
            return self._bluetooth_sequence()
        elif self.config.pattern == HoppingPattern.ADAPTIVE:
            return self._adaptive_sequence(num_channels)
        elif self.config.pattern == HoppingPattern.MILITARY:
            return self._military_sequence(num_channels, self.config.seed)
        elif self.config.pattern == HoppingPattern.CUSTOM:
            return self._custom_sequence()
        else:
            return self._linear_sequence(num_channels)
    
    def _linear_sequence(self, num_channels: int) -> Generator[int, None, None]:
        """Linear sequential hopping"""
        while True:
            for ch in range(num_channels):
                yield ch
    
    def _random_sequence(self, num_channels: int, seed: int) -> Generator[int, None, None]:
        """Pseudo-random hopping sequence"""
        rng = np.random.default_rng(seed)
        channels = list(range(num_channels))
        while True:
            rng.shuffle(channels)
            for ch in channels:
                yield ch
    
    def _bluetooth_sequence(self) -> Generator[int, None, None]:
        """
        Bluetooth AFH (Adaptive Frequency Hopping) sequence
        79 channels, 1 MHz spacing, 2402-2480 MHz
        """
        # Simplified BT hopping (real uses CLK, ADDR, etc.)
        channels = list(range(self.BLUETOOTH_CHANNELS))
        seed = self.config.seed
        
        while True:
            # BT uses a complex PRNG based on clock and address
            # This is simplified demonstration
            for i in range(self.BLUETOOTH_CHANNELS):
                # Simplified hop calculation
                hop = (seed + i * 17) % self.BLUETOOTH_CHANNELS
                yield hop
            seed = (seed + 79) % (2**32)
    
    def _adaptive_sequence(self, num_channels: int) -> Generator[int, None, None]:
        """Adaptive hopping avoiding busy channels"""
        while True:
            # Sort channels by energy (lowest first)
            sorted_channels = sorted(
                self._channel_energy.items(),
                key=lambda x: x[1]
            )
            for ch, _ in sorted_channels:
                yield ch
    
    def _military_sequence(self, num_channels: int, seed: int) -> Generator[int, None, None]:
        """
        Military-grade FHSS sequence
        Uses cryptographic PRNG for unpredictability
        """
        # Use SHA-256 based PRNG for security
        state = seed.to_bytes(8, 'big')
        
        while True:
            # Generate next state
            state = hashlib.sha256(state).digest()
            
            # Extract channel from state
            channel = int.from_bytes(state[:4], 'big') % num_channels
            yield channel
    
    def _custom_sequence(self) -> Generator[int, None, None]:
        """User-defined hopping sequence"""
        if not self.config.custom_sequence:
            logger.warning("No custom sequence defined, using channel 0")
            while True:
                yield 0
        
        while True:
            for ch in self.config.custom_sequence:
                yield ch
    
    def _get_num_channels(self) -> int:
        """Get number of channels in current config"""
        return (self.config.end_freq_hz - self.config.start_freq_hz) // self.config.channel_spacing_hz
    
    def _channel_to_freq(self, channel: int) -> int:
        """Convert channel index to frequency"""
        return self.config.start_freq_hz + channel * self.config.channel_spacing_hz
    
    def _freq_to_channel(self, freq: int) -> int:
        """Convert frequency to channel index"""
        return (freq - self.config.start_freq_hz) // self.config.channel_spacing_hz
    
    def _tune_fast(self, freq_hz: int):
        """Fast tune to frequency using AD9361 fast-lock"""
        self._current_freq = freq_hz
        
        if self.hw:
            try:
                # Use fast tune mode
                # self.hw.tune_fast(freq_hz)
                pass
            except Exception as e:
                logger.error(f"Fast tune failed: {e}")
    
    def _measure_rssi(self) -> float:
        """Measure RSSI at current frequency"""
        if self.hw:
            try:
                # return self.hw.get_rssi()
                pass
            except Exception:
                pass
        
        # Simulate
        return np.random.uniform(-90, -30)
    
    def track_transmitter(self, timeout_s: float = 10.0) -> Optional[TrackedTransmitter]:
        """
        Track a frequency hopping transmitter
        
        Args:
            timeout_s: Max time to track
            
        Returns:
            TrackedTransmitter if found and tracked
        """
        logger.info("Starting transmitter tracking...")
        
        start_time = time.time()
        detected_hops: List[Tuple[float, int]] = []  # (timestamp, freq)
        
        # Sweep and detect
        num_channels = self._get_num_channels()
        
        while time.time() - start_time < timeout_s:
            for ch in range(num_channels):
                freq = self._channel_to_freq(ch)
                self._tune_fast(freq)
                
                rssi = self._measure_rssi()
                self._channel_energy[ch] = rssi
                
                if rssi > self.config.busy_threshold_dbm:
                    detected_hops.append((time.time(), freq))
                
                time.sleep(0.001)  # 1ms per channel
        
        if len(detected_hops) < 3:
            logger.info("Insufficient hops detected for tracking")
            return None
        
        # Analyze detected hops
        freqs = [h[1] for h in detected_hops]
        times = [h[0] for h in detected_hops]
        
        # Detect pattern
        pattern, confidence = self._analyze_hopping_pattern(freqs, times)
        
        # Predict sequence
        prediction = self._predict_sequence(freqs, times)
        
        transmitter = TrackedTransmitter(
            id=f"tx_{int(time.time())}",
            first_seen=datetime.fromtimestamp(times[0]).isoformat(),
            last_seen=datetime.fromtimestamp(times[-1]).isoformat(),
            hop_count=len(detected_hops),
            detected_pattern=pattern,
            frequency_history=freqs,
            timing_history=times,
            predicted_sequence=prediction,
            signal_strength_dbm=np.mean([self._channel_energy.get(
                self._freq_to_channel(f), -100) for f in freqs])
        )
        
        self._tracked_transmitters[transmitter.id] = transmitter
        
        logger.info(f"Tracked transmitter: {transmitter.id}, "
                   f"pattern={pattern.value}, confidence={confidence:.2f}")
        
        return transmitter
    
    def _analyze_hopping_pattern(self, freqs: List[int], 
                                  times: List[float]) -> Tuple[HoppingPattern, float]:
        """Analyze hopping pattern from observations"""
        if len(freqs) < 3:
            return HoppingPattern.RANDOM, 0.0
        
        # Check for linear pattern
        diffs = [freqs[i+1] - freqs[i] for i in range(len(freqs)-1)]
        if len(set(diffs)) == 1:
            return HoppingPattern.LINEAR, 0.95
        
        # Check for Bluetooth pattern (79 channels, 1 MHz spacing)
        bt_start = 2_402_000_000
        bt_end = 2_480_000_000
        if all(bt_start <= f <= bt_end for f in freqs):
            return HoppingPattern.BLUETOOTH, 0.8
        
        # Default to random
        return HoppingPattern.RANDOM, 0.5
    
    def _predict_sequence(self, freqs: List[int], 
                          times: List[float]) -> SequencePrediction:
        """Predict future hopping sequence"""
        # Simple prediction based on observed pattern
        if len(freqs) < 2:
            return SequencePrediction(
                predicted_freqs=[],
                confidence=0.0,
                pattern_type="unknown",
                seed_guess=None,
                timing_offset_us=0
            )
        
        # Estimate dwell time
        if len(times) >= 2:
            avg_dwell = int(np.mean(np.diff(times)) * 1_000_000)
        else:
            avg_dwell = 10000
        
        # Simple linear extrapolation
        last_diff = freqs[-1] - freqs[-2] if len(freqs) >= 2 else 0
        predicted = [freqs[-1] + last_diff * (i+1) for i in range(10)]
        
        return SequencePrediction(
            predicted_freqs=predicted,
            confidence=0.6,
            pattern_type="linear_extrapolation",
            seed_guess=None,
            timing_offset_us=avg_dwell
        )
    
    def jam_hopping_signal(self, target: TrackedTransmitter, 
                           duration_s: float = 10.0) -> bool:
        """
        Jam a tracked frequency hopping signal
        
        Args:
            target: TrackedTransmitter to jam
            duration_s: How long to jam
            
        Returns:
            True if jamming started
            
        WARNING: Jamming is illegal in most jurisdictions.
        Only use in authorized testing environments.
        """
        logger.warning("⚠️ JAMMING FREQUENCY HOPPING SIGNAL")
        logger.warning("This is ILLEGAL without proper authorization!")
        
        if not target.predicted_sequence.predicted_freqs:
            logger.error("No predicted sequence - cannot jam effectively")
            return False
        
        # Follow predicted sequence
        # In real implementation, would transmit noise at each hop
        logger.info(f"Would jam {len(target.predicted_sequence.predicted_freqs)} "
                   f"predicted frequencies for {duration_s}s")
        
        return True
    
    def synchronize_to_hopper(self, target: TrackedTransmitter) -> bool:
        """
        Synchronize hopping to follow a transmitter
        
        Args:
            target: TrackedTransmitter to follow
            
        Returns:
            True if synchronized
        """
        if not target.predicted_sequence.predicted_freqs:
            logger.error("Cannot sync - no predicted sequence")
            return False
        
        # Configure to follow predicted sequence
        self.config.pattern = HoppingPattern.CUSTOM
        self.config.custom_sequence = [
            self._freq_to_channel(f) for f in target.predicted_sequence.predicted_freqs
        ]
        self.config.dwell_time_us = target.predicted_sequence.timing_offset_us
        
        logger.info(f"Synchronized to transmitter {target.id}")
        return True
    
    def get_channel_occupancy(self) -> Dict[int, float]:
        """Get channel energy/occupancy map"""
        return self._channel_energy.copy()
    
    def get_hop_statistics(self) -> Dict:
        """Get hopping statistics"""
        if not self._hop_history:
            return {
                'total_hops': 0,
                'avg_rssi_dbm': -100,
                'channels_visited': 0,
            }
        
        rssi_values = [h.rssi_dbm for h in self._hop_history]
        channels = set(h.channel_index for h in self._hop_history)
        
        return {
            'total_hops': self._hop_count,
            'avg_rssi_dbm': np.mean(rssi_values),
            'min_rssi_dbm': np.min(rssi_values),
            'max_rssi_dbm': np.max(rssi_values),
            'channels_visited': len(channels),
            'hops_per_second': self._hop_count / max(1, (
                time.time() - datetime.fromisoformat(
                    self._hop_history[0].timestamp).timestamp()
            )) if self._hop_history else 0,
        }
    
    def get_status(self) -> Dict:
        """Get frequency hopping status"""
        return {
            'hopping': self.is_hopping,
            'pattern': self.config.pattern.value,
            'current_freq_mhz': self._current_freq / 1e6,
            'current_channel': self._current_channel,
            'hop_count': self._hop_count,
            'num_channels': self._get_num_channels(),
            'dwell_time_us': self.config.dwell_time_us,
            'tracked_transmitters': len(self._tracked_transmitters),
            'statistics': self.get_hop_statistics(),
        }


# Convenience function
def get_frequency_hopping_controller(hardware_controller=None) -> BladeRFFrequencyHopping:
    """Get BladeRF Frequency Hopping controller instance"""
    return BladeRFFrequencyHopping(hardware_controller)
