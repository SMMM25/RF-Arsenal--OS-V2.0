#!/usr/bin/env python3
"""
RF Arsenal OS - Stealth FPGA Controller
Hardware-accelerated stealth mode features

Implements:
- Frequency hopping with hardware timing
- Soft power ramping
- Burst transmission control
- Emission masking
- Real-time pattern generation
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class HoppingPattern(Enum):
    """Frequency hopping pattern types"""
    SEQUENTIAL = "sequential"  # Sequential through list
    RANDOM = "random"  # Pseudo-random
    PN_SEQUENCE = "pn_sequence"  # PN-code based
    ADAPTIVE = "adaptive"  # AI-driven adaptive
    LPI = "lpi"  # Low probability of intercept pattern
    CUSTOM = "custom"  # User-defined pattern


class StealthLevel(Enum):
    """Stealth operation levels"""
    NONE = "none"  # No stealth
    LOW = "low"  # Basic measures
    MEDIUM = "medium"  # Enhanced measures
    HIGH = "high"  # Maximum stealth
    ADAPTIVE = "adaptive"  # AI-controlled


class TransmissionMode(Enum):
    """Transmission mode types"""
    CONTINUOUS = "continuous"
    BURST = "burst"
    SPREAD_SPECTRUM = "spread_spectrum"
    ULTRA_SHORT_BURST = "ultra_short_burst"


@dataclass
class FrequencyHopConfig:
    """Frequency hopping configuration"""
    # Frequency list
    frequencies: List[int] = field(default_factory=list)  # Hz
    
    # Hopping parameters
    pattern: HoppingPattern = HoppingPattern.RANDOM
    hop_rate_hz: float = 100.0  # Hops per second
    dwell_time_ms: float = 10.0  # Time on each frequency
    
    # Band limits
    min_frequency: int = 70_000_000  # 70 MHz
    max_frequency: int = 6_000_000_000  # 6 GHz
    channel_bandwidth: int = 200_000  # 200 kHz
    
    # Synchronization
    sync_sequence_length: int = 64
    sync_guard_time_us: float = 100.0
    
    # Anti-jam features
    avoid_frequencies: List[int] = field(default_factory=list)
    adaptive_avoidance: bool = False
    
    # PN sequence parameters (if pattern == PN_SEQUENCE)
    pn_polynomial: int = 0x1021  # CRC-16-CCITT
    pn_seed: int = 0xFFFF
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.frequencies and self.pattern != HoppingPattern.ADAPTIVE:
            return False
        if self.hop_rate_hz <= 0:
            return False
        if self.dwell_time_ms <= 0:
            return False
        return True
    
    def calculate_hop_count(self) -> int:
        """Calculate number of frequency channels"""
        if self.frequencies:
            return len(self.frequencies)
        return int((self.max_frequency - self.min_frequency) / self.channel_bandwidth)


@dataclass
class PowerRampConfig:
    """Power ramping configuration"""
    # Ramp parameters
    ramp_up_time_ms: float = 5.0
    ramp_down_time_ms: float = 5.0
    ramp_shape: str = "cosine"  # "linear", "cosine", "exponential"
    
    # Power limits
    min_power_dbm: float = -30.0
    max_power_dbm: float = 10.0
    target_power_dbm: float = 0.0
    
    # Safety limits
    absolute_max_dbm: float = 20.0
    emergency_cutoff_enabled: bool = True
    
    # Adaptation
    adapt_to_environment: bool = False
    target_snr_db: float = 20.0
    
    def validate(self) -> bool:
        """Validate configuration"""
        if self.min_power_dbm >= self.max_power_dbm:
            return False
        if self.target_power_dbm > self.absolute_max_dbm:
            return False
        if self.ramp_up_time_ms <= 0 or self.ramp_down_time_ms <= 0:
            return False
        return True


@dataclass
class StealthProfile:
    """Complete stealth operation profile"""
    name: str = "default"
    level: StealthLevel = StealthLevel.MEDIUM
    
    # Frequency hopping
    hop_config: FrequencyHopConfig = field(default_factory=FrequencyHopConfig)
    
    # Power control
    power_config: PowerRampConfig = field(default_factory=PowerRampConfig)
    
    # Transmission timing
    transmission_mode: TransmissionMode = TransmissionMode.BURST
    burst_duration_ms: float = 10.0
    burst_interval_ms: float = 100.0
    duty_cycle: float = 0.1  # 10%
    
    # Emission control
    spectral_mask_enabled: bool = True
    spurious_suppression_db: float = 50.0
    
    # Anti-detection measures
    randomize_timing: bool = True
    timing_jitter_ms: float = 2.0
    frequency_offset_randomization: bool = True
    max_frequency_offset_hz: int = 1000
    
    # Synchronization
    sync_enabled: bool = True
    sync_interval_hops: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'level': self.level.value,
            'transmission_mode': self.transmission_mode.value,
            'burst_duration_ms': self.burst_duration_ms,
            'burst_interval_ms': self.burst_interval_ms,
            'duty_cycle': self.duty_cycle,
            'hop_rate_hz': self.hop_config.hop_rate_hz,
            'power_max_dbm': self.power_config.max_power_dbm,
        }


class StealthFPGAController:
    """
    Hardware-accelerated stealth mode controller
    
    Interfaces with FPGA stealth processor for:
    - Sub-millisecond frequency hopping
    - Precise power ramping
    - Synchronized burst transmission
    - Real-time pattern adaptation
    """
    
    # FPGA register addresses (matching stealth_processor.vhd)
    class Register:
        CONTROL = 0x0050
        STATUS = 0x0054
        HOP_RATE = 0x0058
        HOP_PATTERN = 0x005C
        POWER_TARGET = 0x0060
        POWER_RAMP_RATE = 0x0064
        POWER_LIMIT = 0x0068
        BURST_DURATION = 0x006C
        BURST_INTERVAL = 0x0070
        SYNC_COUNTER = 0x0074
        ERROR_FLAGS = 0x0078
        PATTERN_MEM_BASE = 0x2000
    
    # Predefined stealth profiles
    PROFILES = {
        'low_observable': StealthProfile(
            name='low_observable',
            level=StealthLevel.HIGH,
            hop_config=FrequencyHopConfig(
                pattern=HoppingPattern.PN_SEQUENCE,
                hop_rate_hz=1000,
                dwell_time_ms=1.0,
            ),
            power_config=PowerRampConfig(
                max_power_dbm=0.0,
                ramp_shape='cosine',
            ),
            transmission_mode=TransmissionMode.ULTRA_SHORT_BURST,
            burst_duration_ms=1.0,
            burst_interval_ms=500.0,
            duty_cycle=0.002,
        ),
        'moderate_stealth': StealthProfile(
            name='moderate_stealth',
            level=StealthLevel.MEDIUM,
            hop_config=FrequencyHopConfig(
                pattern=HoppingPattern.RANDOM,
                hop_rate_hz=100,
                dwell_time_ms=10.0,
            ),
            power_config=PowerRampConfig(
                max_power_dbm=10.0,
            ),
            transmission_mode=TransmissionMode.BURST,
            burst_duration_ms=10.0,
            burst_interval_ms=100.0,
            duty_cycle=0.1,
        ),
        'spread_spectrum': StealthProfile(
            name='spread_spectrum',
            level=StealthLevel.MEDIUM,
            hop_config=FrequencyHopConfig(
                pattern=HoppingPattern.PN_SEQUENCE,
                hop_rate_hz=10000,
                dwell_time_ms=0.1,
            ),
            transmission_mode=TransmissionMode.SPREAD_SPECTRUM,
        ),
        'adaptive': StealthProfile(
            name='adaptive',
            level=StealthLevel.ADAPTIVE,
            hop_config=FrequencyHopConfig(
                pattern=HoppingPattern.ADAPTIVE,
                adaptive_avoidance=True,
            ),
            power_config=PowerRampConfig(
                adapt_to_environment=True,
            ),
        ),
    }
    
    def __init__(
        self,
        fpga_controller: Any,
        event_callback: Optional[Callable] = None
    ):
        """
        Initialize Stealth FPGA Controller
        
        Args:
            fpga_controller: FPGAController instance
            event_callback: Callback for stealth events
        """
        self._fpga = fpga_controller
        self._event_callback = event_callback
        
        # Current state
        self._active = False
        self._profile: Optional[StealthProfile] = None
        self._current_frequency: int = 0
        self._current_power_dbm: float = -30.0
        
        # Hop sequence state
        self._hop_sequence: List[int] = []
        self._hop_index: int = 0
        self._pn_state: int = 0
        
        # Statistics
        self._stats = {
            'hops_executed': 0,
            'bursts_transmitted': 0,
            'power_adjustments': 0,
            'avoided_frequencies': 0,
            'sync_events': 0,
        }
        
        # Monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        
        logger.info("StealthFPGAController initialized")
    
    @property
    def is_active(self) -> bool:
        return self._active
    
    @property
    def current_frequency(self) -> int:
        return self._current_frequency
    
    @property
    def current_power_dbm(self) -> float:
        return self._current_power_dbm
    
    @property
    def stats(self) -> Dict[str, int]:
        return self._stats.copy()
    
    # =========================================================================
    # Profile Management
    # =========================================================================
    
    def get_profile(self, name: str) -> Optional[StealthProfile]:
        """Get predefined profile by name"""
        return self.PROFILES.get(name)
    
    def list_profiles(self) -> List[str]:
        """List available profile names"""
        return list(self.PROFILES.keys())
    
    def create_custom_profile(
        self,
        name: str,
        base_profile: Optional[str] = None,
        **kwargs
    ) -> StealthProfile:
        """
        Create custom stealth profile
        
        Args:
            name: Profile name
            base_profile: Name of profile to base on
            **kwargs: Profile parameter overrides
            
        Returns:
            New StealthProfile
        """
        if base_profile and base_profile in self.PROFILES:
            import copy
            profile = copy.deepcopy(self.PROFILES[base_profile])
            profile.name = name
        else:
            profile = StealthProfile(name=name)
        
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
            elif hasattr(profile.hop_config, key):
                setattr(profile.hop_config, key, value)
            elif hasattr(profile.power_config, key):
                setattr(profile.power_config, key, value)
        
        return profile
    
    # =========================================================================
    # Activation / Deactivation
    # =========================================================================
    
    async def activate(
        self,
        profile: Optional[StealthProfile] = None,
        profile_name: Optional[str] = None
    ) -> bool:
        """
        Activate stealth mode
        
        Args:
            profile: StealthProfile to use
            profile_name: Or name of predefined profile
            
        Returns:
            True if activated successfully
        """
        if self._active:
            logger.warning("Stealth already active, deactivating first")
            await self.deactivate()
        
        # Get profile
        if profile is None:
            if profile_name:
                profile = self.PROFILES.get(profile_name)
            else:
                profile = self.PROFILES['moderate_stealth']
        
        if profile is None:
            logger.error("No valid profile specified")
            return False
        
        self._profile = profile
        
        try:
            logger.info(f"Activating stealth mode: {profile.name}")
            
            # Configure FPGA
            await self._configure_fpga(profile)
            
            # Generate hop sequence
            self._generate_hop_sequence(profile.hop_config)
            
            # Start hardware
            self._start_hardware()
            
            # Start monitoring
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self._active = True
            self._emit_event('stealth_activated', profile.to_dict())
            
            logger.info("Stealth mode activated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate stealth: {e}")
            return False
    
    async def deactivate(self) -> bool:
        """Deactivate stealth mode"""
        if not self._active:
            return True
        
        try:
            logger.info("Deactivating stealth mode")
            
            # Stop monitoring
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
                self._monitoring_task = None
            
            # Stop hardware
            self._stop_hardware()
            
            # Ramp down power
            await self._ramp_power(self._profile.power_config.min_power_dbm)
            
            self._active = False
            self._profile = None
            
            self._emit_event('stealth_deactivated', None)
            
            logger.info("Stealth mode deactivated")
            return True
            
        except Exception as e:
            logger.error(f"Deactivation failed: {e}")
            return False
    
    # =========================================================================
    # Frequency Hopping
    # =========================================================================
    
    def set_hop_frequencies(self, frequencies: List[int]) -> bool:
        """
        Set frequency hop list
        
        Args:
            frequencies: List of frequencies in Hz
            
        Returns:
            True if set successfully
        """
        if self._profile is None:
            logger.error("No active profile")
            return False
        
        # Validate frequencies
        config = self._profile.hop_config
        valid_frequencies = []
        
        for freq in frequencies:
            if config.min_frequency <= freq <= config.max_frequency:
                if freq not in config.avoid_frequencies:
                    valid_frequencies.append(freq)
        
        if not valid_frequencies:
            logger.error("No valid frequencies")
            return False
        
        config.frequencies = valid_frequencies
        self._generate_hop_sequence(config)
        
        # Update FPGA
        self._write_hop_pattern()
        
        logger.info(f"Set {len(valid_frequencies)} hop frequencies")
        return True
    
    def add_avoid_frequency(self, frequency: int) -> None:
        """Add frequency to avoid list"""
        if self._profile:
            if frequency not in self._profile.hop_config.avoid_frequencies:
                self._profile.hop_config.avoid_frequencies.append(frequency)
                self._stats['avoided_frequencies'] += 1
    
    def generate_frequency_band(
        self,
        start_freq: int,
        end_freq: int,
        channel_spacing: int
    ) -> List[int]:
        """
        Generate frequency list for a band
        
        Args:
            start_freq: Start frequency in Hz
            end_freq: End frequency in Hz
            channel_spacing: Channel spacing in Hz
            
        Returns:
            List of channel frequencies
        """
        frequencies = []
        freq = start_freq
        
        while freq <= end_freq:
            frequencies.append(freq)
            freq += channel_spacing
        
        return frequencies
    
    def _generate_hop_sequence(self, config: FrequencyHopConfig) -> None:
        """Generate frequency hop sequence"""
        if not config.frequencies:
            # Generate default frequencies
            config.frequencies = self.generate_frequency_band(
                config.min_frequency,
                config.max_frequency,
                config.channel_bandwidth
            )
        
        num_freqs = len(config.frequencies)
        
        if config.pattern == HoppingPattern.SEQUENTIAL:
            self._hop_sequence = list(range(num_freqs))
            
        elif config.pattern == HoppingPattern.RANDOM:
            self._hop_sequence = list(range(num_freqs))
            random.shuffle(self._hop_sequence)
            
        elif config.pattern == HoppingPattern.PN_SEQUENCE:
            self._hop_sequence = self._generate_pn_sequence(
                num_freqs,
                config.pn_polynomial,
                config.pn_seed
            )
            
        elif config.pattern == HoppingPattern.LPI:
            # Low probability of intercept - maximize distance between hops
            self._hop_sequence = self._generate_lpi_sequence(num_freqs)
            
        else:
            self._hop_sequence = list(range(num_freqs))
        
        self._hop_index = 0
        logger.info(f"Generated hop sequence: {len(self._hop_sequence)} entries")
    
    def _generate_pn_sequence(
        self,
        length: int,
        polynomial: int,
        seed: int
    ) -> List[int]:
        """Generate PN sequence for hopping"""
        sequence = []
        state = seed
        used = set()
        
        for _ in range(length * 10):  # Allow retries
            if len(sequence) >= length:
                break
            
            # LFSR step
            bit = 0
            for i in range(16):
                if polynomial & (1 << i):
                    bit ^= (state >> i) & 1
            state = ((state << 1) | bit) & 0xFFFF
            
            # Map to frequency index
            idx = state % length
            if idx not in used:
                sequence.append(idx)
                used.add(idx)
        
        # Fill remaining with sequential
        for i in range(length):
            if i not in used:
                sequence.append(i)
        
        self._pn_state = state
        return sequence
    
    def _generate_lpi_sequence(self, length: int) -> List[int]:
        """Generate LPI hopping sequence"""
        # Use prime-step pattern for maximum separation
        sequence = []
        step = self._find_coprime(length)
        idx = 0
        
        for _ in range(length):
            sequence.append(idx)
            idx = (idx + step) % length
        
        return sequence
    
    def _find_coprime(self, n: int) -> int:
        """Find a number coprime to n"""
        import math
        for i in range(n // 3, n):
            if math.gcd(i, n) == 1:
                return i
        return 1
    
    def get_next_frequency(self) -> int:
        """Get next frequency in hop sequence"""
        if not self._hop_sequence or not self._profile:
            return 0
        
        idx = self._hop_sequence[self._hop_index]
        freq = self._profile.hop_config.frequencies[idx]
        
        self._hop_index = (self._hop_index + 1) % len(self._hop_sequence)
        self._current_frequency = freq
        self._stats['hops_executed'] += 1
        
        return freq
    
    # =========================================================================
    # Power Control
    # =========================================================================
    
    async def set_power(self, power_dbm: float) -> bool:
        """
        Set transmit power with ramping
        
        Args:
            power_dbm: Target power in dBm
            
        Returns:
            True if successful
        """
        if self._profile is None:
            return False
        
        config = self._profile.power_config
        
        # Clamp to limits
        power_dbm = max(config.min_power_dbm, min(config.absolute_max_dbm, power_dbm))
        
        return await self._ramp_power(power_dbm)
    
    async def _ramp_power(self, target_dbm: float) -> bool:
        """Ramp power to target with smooth transition"""
        if self._profile is None:
            return False
        
        config = self._profile.power_config
        current = self._current_power_dbm
        
        if abs(target_dbm - current) < 0.1:
            return True
        
        # Calculate ramp parameters
        if target_dbm > current:
            ramp_time = config.ramp_up_time_ms / 1000
        else:
            ramp_time = config.ramp_down_time_ms / 1000
        
        steps = int(ramp_time * 1000)  # 1ms steps
        step_size = (target_dbm - current) / steps if steps > 0 else target_dbm - current
        
        try:
            for i in range(steps):
                power = current + step_size * (i + 1)
                
                # Apply ramp shape
                if config.ramp_shape == 'cosine':
                    t = (i + 1) / steps
                    power = current + (target_dbm - current) * (1 - np.cos(np.pi * t)) / 2
                elif config.ramp_shape == 'exponential':
                    t = (i + 1) / steps
                    power = current + (target_dbm - current) * (1 - np.exp(-3 * t))
                
                self._set_fpga_power(power)
                self._current_power_dbm = power
                self._stats['power_adjustments'] += 1
                
                await asyncio.sleep(0.001)  # 1ms steps
            
            return True
            
        except Exception as e:
            logger.error(f"Power ramp failed: {e}")
            return False
    
    def _set_fpga_power(self, power_dbm: float) -> None:
        """Set FPGA power register"""
        if self._fpga is None:
            return
        
        # Convert to FPGA format (0.1 dB units, offset by 90)
        power_raw = int((power_dbm + 90) * 10)
        power_raw = max(0, min(1500, power_raw))  # Clamp to valid range
        
        self._fpga._write_register(self.Register.POWER_TARGET, power_raw)
    
    # =========================================================================
    # Burst Transmission
    # =========================================================================
    
    def configure_burst(
        self,
        duration_ms: float,
        interval_ms: float
    ) -> bool:
        """
        Configure burst transmission parameters
        
        Args:
            duration_ms: Burst duration in milliseconds
            interval_ms: Interval between bursts
            
        Returns:
            True if configured
        """
        if self._profile is None:
            return False
        
        self._profile.burst_duration_ms = duration_ms
        self._profile.burst_interval_ms = interval_ms
        self._profile.duty_cycle = duration_ms / interval_ms
        
        # Update FPGA registers
        if self._fpga:
            # Convert to clock cycles (assuming 30.72 MHz)
            clock_rate = 30_720_000
            duration_cycles = int(duration_ms * clock_rate / 1000)
            interval_cycles = int(interval_ms * clock_rate / 1000)
            
            self._fpga._write_register(self.Register.BURST_DURATION, duration_cycles)
            self._fpga._write_register(self.Register.BURST_INTERVAL, interval_cycles)
        
        return True
    
    def trigger_burst(self) -> bool:
        """Manually trigger a transmission burst"""
        if not self._active:
            return False
        
        if self._fpga:
            # Set burst trigger bit
            ctrl = self._fpga._read_register(self.Register.CONTROL)
            self._fpga._write_register(self.Register.CONTROL, ctrl | 0x0100)
        
        self._stats['bursts_transmitted'] += 1
        return True
    
    # =========================================================================
    # FPGA Interface
    # =========================================================================
    
    async def _configure_fpga(self, profile: StealthProfile) -> None:
        """Configure FPGA for stealth operation"""
        if self._fpga is None:
            logger.warning("No FPGA controller, running in simulation mode")
            return
        
        # Configure hopping
        hop_rate_cycles = int(30_720_000 / profile.hop_config.hop_rate_hz)
        self._fpga._write_register(self.Register.HOP_RATE, hop_rate_cycles)
        
        # Configure power ramping
        ramp_rate = int(profile.power_config.ramp_up_time_ms * 1000)  # microseconds
        self._fpga._write_register(self.Register.POWER_RAMP_RATE, ramp_rate)
        
        # Set power limit
        max_power_raw = int((profile.power_config.absolute_max_dbm + 90) * 10)
        self._fpga._write_register(self.Register.POWER_LIMIT, max_power_raw)
        
        # Configure burst timing
        clock_rate = 30_720_000
        duration_cycles = int(profile.burst_duration_ms * clock_rate / 1000)
        interval_cycles = int(profile.burst_interval_ms * clock_rate / 1000)
        self._fpga._write_register(self.Register.BURST_DURATION, duration_cycles)
        self._fpga._write_register(self.Register.BURST_INTERVAL, interval_cycles)
        
        # Write hop pattern to FPGA memory
        self._write_hop_pattern()
    
    def _write_hop_pattern(self) -> None:
        """Write frequency hop pattern to FPGA memory"""
        if self._fpga is None or self._profile is None:
            return
        
        frequencies = self._profile.hop_config.frequencies
        
        for i, freq in enumerate(frequencies[:256]):  # Max 256 frequencies
            self._fpga._write_register(self.Register.PATTERN_MEM_BASE + i * 4, freq)
        
        # Set pattern length
        self._fpga._write_register(self.Register.HOP_PATTERN, len(frequencies))
    
    def _start_hardware(self) -> None:
        """Start FPGA stealth processor"""
        if self._fpga is None:
            return
        
        # Enable stealth processor with appropriate flags
        control = 0x0001  # Enable
        
        if self._profile:
            if self._profile.hop_config.pattern != HoppingPattern.SEQUENTIAL:
                control |= 0x0002  # Enable hopping
            
            if self._profile.power_config.ramp_up_time_ms > 0:
                control |= 0x0004  # Enable power ramping
            
            if self._profile.transmission_mode != TransmissionMode.CONTINUOUS:
                control |= 0x0008  # Enable burst mode
        
        self._fpga._write_register(self.Register.CONTROL, control)
    
    def _stop_hardware(self) -> None:
        """Stop FPGA stealth processor"""
        if self._fpga:
            self._fpga._write_register(self.Register.CONTROL, 0x0000)
    
    # =========================================================================
    # Monitoring
    # =========================================================================
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while self._active:
            try:
                # Check FPGA status
                if self._fpga:
                    status = self._fpga._read_register(self.Register.STATUS)
                    error_flags = self._fpga._read_register(self.Register.ERROR_FLAGS)
                    
                    if error_flags:
                        logger.warning(f"Stealth error flags: 0x{error_flags:04X}")
                        self._emit_event('stealth_error', {'flags': error_flags})
                    
                    # Update sync counter
                    sync = self._fpga._read_register(self.Register.SYNC_COUNTER)
                    if sync > 0:
                        self._stats['sync_events'] = sync
                
                await asyncio.sleep(0.1)  # 100ms monitoring interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(1.0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current stealth status"""
        return {
            'active': self._active,
            'profile': self._profile.name if self._profile else None,
            'level': self._profile.level.value if self._profile else None,
            'current_frequency': self._current_frequency,
            'current_power_dbm': self._current_power_dbm,
            'hop_index': self._hop_index,
            'stats': self._stats.copy(),
        }
    
    # =========================================================================
    # AI Integration
    # =========================================================================
    
    async def ai_command(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process AI command
        
        Commands:
        - activate: Activate stealth with profile
        - deactivate: Deactivate stealth
        - set_profile: Change profile
        - set_power: Set transmit power
        - add_avoid: Add frequency to avoid list
        - trigger_burst: Trigger transmission burst
        - get_status: Get current status
        """
        try:
            if command == "activate":
                profile_name = parameters.get('profile', 'moderate_stealth')
                success = await self.activate(profile_name=profile_name)
                return {'success': success, 'active': self._active}
            
            elif command == "deactivate":
                success = await self.deactivate()
                return {'success': success, 'active': self._active}
            
            elif command == "set_profile":
                profile_name = parameters.get('name')
                if profile_name in self.PROFILES:
                    if self._active:
                        await self.deactivate()
                    success = await self.activate(profile_name=profile_name)
                    return {'success': success, 'profile': profile_name}
                return {'success': False, 'error': 'Unknown profile'}
            
            elif command == "set_power":
                power = parameters.get('power_dbm', 0.0)
                success = await self.set_power(power)
                return {'success': success, 'power_dbm': self._current_power_dbm}
            
            elif command == "add_avoid":
                freq = parameters.get('frequency')
                if freq:
                    self.add_avoid_frequency(freq)
                    return {'success': True}
                return {'success': False, 'error': 'No frequency specified'}
            
            elif command == "trigger_burst":
                success = self.trigger_burst()
                return {'success': success}
            
            elif command == "get_status":
                return {'success': True, 'status': self.get_status()}
            
            else:
                return {'success': False, 'error': f'Unknown command: {command}'}
                
        except Exception as e:
            logger.error(f"AI command failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _emit_event(self, event_type: str, data: Any) -> None:
        """Emit stealth event"""
        if self._event_callback:
            try:
                self._event_callback(event_type, data)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
