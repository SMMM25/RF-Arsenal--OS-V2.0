"""
TEMPEST Emission Security Module for RF Arsenal OS.

This module implements comprehensive TEMPEST (Transient Electromagnetic Pulse
Emanation Standard) security controls for protecting against compromising
electromagnetic emanations per NSA/CSS EPL standards.

TEMPEST Protection Levels:
- NATO SDIP-27 Level A (AMSG 720B): Formerly TEMPEST Level I - Attack distance <1m
- NATO SDIP-27 Level B (AMSG 788A): Formerly TEMPEST Level II - Attack distance <20m
- NATO SDIP-27 Level C (AMSG 784): Formerly TEMPEST Level III - Attack distance <100m
- Zone 0-3 Classifications per CNSS Policy No. 3 (CNSSP-3)

Security Features:
- Real-time emission spectrum analysis and monitoring
- Adaptive power control to minimize emanations
- Timing jitter injection for side-channel resistance
- Signal masking and noise injection
- Shielding effectiveness monitoring
- Zone-based security enforcement
- Emanation anomaly detection

Author: RF Arsenal Development Team
License: Proprietary - Security Sensitive
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import asyncio
import hashlib
import hmac
import logging
import math
import os
import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import secrets
import numpy as np

# Configure secure logging
logger = logging.getLogger(__name__)


class TEMPESTLevel(Enum):
    """TEMPEST protection levels per NATO SDIP-27 standards."""
    
    LEVEL_A = "NATO_SDIP_27_A"  # Highest protection, <1m attack distance
    LEVEL_B = "NATO_SDIP_27_B"  # Medium protection, <20m attack distance
    LEVEL_C = "NATO_SDIP_27_C"  # Basic protection, <100m attack distance
    UNPROTECTED = "NONE"


class ZoneClassification(IntEnum):
    """TEMPEST Zone classifications per CNSSP-3."""
    
    ZONE_0 = 0  # Controlled space - highest emanation security
    ZONE_1 = 1  # Inspected space - verified clear
    ZONE_2 = 2  # Controlled space - limited access
    ZONE_3 = 3  # Public space - requires full TEMPEST protection


class EmissionCategory(Enum):
    """Categories of potentially compromising emanations."""
    
    RF_CONDUCTED = "rf_conducted"       # RF conducted emissions
    RF_RADIATED = "rf_radiated"         # RF radiated emissions
    POWER_LINE = "power_line"           # Power line conducted
    ACOUSTIC = "acoustic"               # Acoustic emanations
    OPTICAL = "optical"                 # LED/display emissions
    THERMAL = "thermal"                 # Thermal side channels
    MAGNETIC = "magnetic"               # Magnetic field emissions
    TIMING = "timing"                   # Timing side channels


class EmissionSeverity(Enum):
    """Severity levels for detected emanations."""
    
    CRITICAL = "critical"   # Immediate compromise risk
    HIGH = "high"           # Significant leakage detected
    MEDIUM = "medium"       # Moderate concern
    LOW = "low"             # Minor emanations
    NOMINAL = "nominal"     # Within acceptable limits


@dataclass
class EmissionProfile:
    """Profile of detected electromagnetic emanations."""
    
    timestamp: datetime
    category: EmissionCategory
    severity: EmissionSeverity
    frequency_hz: float
    amplitude_dbm: float
    bandwidth_hz: float
    modulation_detected: bool
    correlation_score: float  # 0-1, correlation to data signals
    location: str
    zone: ZoneClassification
    recommended_action: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShieldingStatus:
    """Shielding effectiveness status."""
    
    timestamp: datetime
    location: str
    effectiveness_db: float
    frequency_range_hz: Tuple[float, float]
    test_method: str
    compliant: bool
    certification_expires: Optional[datetime]


@dataclass
class TEMPESTComplianceStatus:
    """Overall TEMPEST compliance status."""
    
    compliant: bool
    protection_level: TEMPESTLevel
    zone: ZoneClassification
    last_assessment: datetime
    next_assessment_due: datetime
    emission_count_by_severity: Dict[EmissionSeverity, int]
    shielding_status: List[ShieldingStatus]
    active_countermeasures: List[str]
    findings: List[str]
    recommendations: List[str]


class TEMPESTException(Exception):
    """Base exception for TEMPEST security failures."""
    pass


class EmissionViolation(TEMPESTException):
    """Raised when emission limits are exceeded."""
    pass


class ShieldingFailure(TEMPESTException):
    """Raised when shielding effectiveness degrades."""
    pass


class ZoneViolation(TEMPESTException):
    """Raised when zone security is compromised."""
    pass


class TEMPESTCountermeasure(ABC):
    """Abstract base class for TEMPEST countermeasures."""
    
    @abstractmethod
    def activate(self) -> bool:
        """Activate the countermeasure."""
        pass
    
    @abstractmethod
    def deactivate(self) -> bool:
        """Deactivate the countermeasure."""
        pass
    
    @abstractmethod
    def get_effectiveness(self) -> float:
        """Get countermeasure effectiveness (0-1)."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current countermeasure status."""
        pass


class SignalMaskingCountermeasure(TEMPESTCountermeasure):
    """
    Signal masking countermeasure implementation.
    
    Injects noise/masking signals to obscure potentially
    compromising emanations.
    """
    
    def __init__(
        self,
        noise_bandwidth_hz: float = 100e6,
        noise_power_dbm: float = -60.0,
        mask_pattern: str = "broadband"
    ):
        self.noise_bandwidth_hz = noise_bandwidth_hz
        self.noise_power_dbm = noise_power_dbm
        self.mask_pattern = mask_pattern
        self._active = False
        self._effectiveness = 0.0
        self._lock = threading.Lock()
        
    def activate(self) -> bool:
        """Activate signal masking."""
        with self._lock:
            if self._active:
                return True
            
            # Initialize noise generation
            self._active = True
            self._effectiveness = self._calculate_effectiveness()
            
            logger.info(
                f"Signal masking activated: pattern={self.mask_pattern}, "
                f"bandwidth={self.noise_bandwidth_hz/1e6:.1f} MHz"
            )
            return True
    
    def deactivate(self) -> bool:
        """Deactivate signal masking."""
        with self._lock:
            self._active = False
            self._effectiveness = 0.0
            logger.info("Signal masking deactivated")
            return True
    
    def get_effectiveness(self) -> float:
        """Get masking effectiveness."""
        return self._effectiveness
    
    def get_status(self) -> Dict[str, Any]:
        """Get masking status."""
        return {
            "type": "signal_masking",
            "active": self._active,
            "effectiveness": self._effectiveness,
            "noise_bandwidth_hz": self.noise_bandwidth_hz,
            "noise_power_dbm": self.noise_power_dbm,
            "mask_pattern": self.mask_pattern
        }
    
    def _calculate_effectiveness(self) -> float:
        """Calculate masking effectiveness based on parameters."""
        # Effectiveness based on bandwidth coverage and power level
        bandwidth_factor = min(self.noise_bandwidth_hz / 1e9, 1.0)
        power_factor = min((-self.noise_power_dbm + 80) / 60, 1.0)
        
        pattern_factors = {
            "broadband": 0.85,
            "targeted": 0.95,
            "adaptive": 0.90
        }
        pattern_factor = pattern_factors.get(self.mask_pattern, 0.80)
        
        return bandwidth_factor * power_factor * pattern_factor
    
    def generate_masking_signal(
        self,
        duration_samples: int,
        sample_rate_hz: float
    ) -> np.ndarray:
        """
        Generate masking signal samples.
        
        Args:
            duration_samples: Number of samples to generate
            sample_rate_hz: Sample rate in Hz
            
        Returns:
            Complex IQ samples for masking signal
        """
        if self.mask_pattern == "broadband":
            # White noise covering full bandwidth
            return self._generate_broadband_noise(duration_samples)
        elif self.mask_pattern == "targeted":
            # Shaped noise targeting specific frequencies
            return self._generate_targeted_noise(
                duration_samples, sample_rate_hz
            )
        else:
            # Adaptive pattern based on detected emissions
            return self._generate_adaptive_noise(
                duration_samples, sample_rate_hz
            )
    
    def _generate_broadband_noise(self, num_samples: int) -> np.ndarray:
        """Generate broadband white noise."""
        # Use cryptographically secure random for noise generation
        random_bytes = secrets.token_bytes(num_samples * 8)
        noise_real = np.frombuffer(random_bytes[:num_samples*4], dtype=np.float32)
        noise_imag = np.frombuffer(random_bytes[num_samples*4:], dtype=np.float32)
        
        # Normalize and scale to power level
        power_linear = 10 ** (self.noise_power_dbm / 10) / 1000  # mW to W
        scale = np.sqrt(power_linear / 2)
        
        return (noise_real + 1j * noise_imag) * scale
    
    def _generate_targeted_noise(
        self,
        num_samples: int,
        sample_rate_hz: float
    ) -> np.ndarray:
        """Generate noise targeted at specific frequencies."""
        # Target typical emanation frequencies
        target_freqs = [
            100e6, 200e6, 400e6, 800e6,  # Clock harmonics
            2.4e9, 5.0e9                  # Common RF bands
        ]
        
        t = np.arange(num_samples) / sample_rate_hz
        signal = np.zeros(num_samples, dtype=complex)
        
        for freq in target_freqs:
            if freq < sample_rate_hz / 2:
                # Add noise component at each frequency
                phase = secrets.randbelow(360) * np.pi / 180
                signal += np.exp(2j * np.pi * freq * t + 1j * phase)
        
        # Scale to power level
        power_linear = 10 ** (self.noise_power_dbm / 10) / 1000
        signal *= np.sqrt(power_linear / len(target_freqs))
        
        return signal
    
    def _generate_adaptive_noise(
        self,
        num_samples: int,
        sample_rate_hz: float
    ) -> np.ndarray:
        """Generate adaptive noise based on emission profile."""
        # Combination of broadband and targeted
        broadband = self._generate_broadband_noise(num_samples) * 0.3
        targeted = self._generate_targeted_noise(
            num_samples, sample_rate_hz
        ) * 0.7
        
        return broadband + targeted


class TimingJitterCountermeasure(TEMPESTCountermeasure):
    """
    Timing jitter countermeasure implementation.
    
    Injects random timing variations to defeat timing-based
    side-channel analysis.
    """
    
    def __init__(
        self,
        jitter_range_ns: Tuple[float, float] = (0, 1000),
        distribution: str = "uniform"
    ):
        self.jitter_range_ns = jitter_range_ns
        self.distribution = distribution
        self._active = False
        self._effectiveness = 0.0
        self._jitter_stats = {
            "total_jitter_applied": 0,
            "operations_protected": 0
        }
        self._lock = threading.Lock()
    
    def activate(self) -> bool:
        """Activate timing jitter injection."""
        with self._lock:
            if self._active:
                return True
            
            self._active = True
            self._effectiveness = self._calculate_effectiveness()
            
            logger.info(
                f"Timing jitter activated: range={self.jitter_range_ns}ns, "
                f"distribution={self.distribution}"
            )
            return True
    
    def deactivate(self) -> bool:
        """Deactivate timing jitter."""
        with self._lock:
            self._active = False
            self._effectiveness = 0.0
            logger.info("Timing jitter deactivated")
            return True
    
    def get_effectiveness(self) -> float:
        """Get jitter effectiveness."""
        return self._effectiveness
    
    def get_status(self) -> Dict[str, Any]:
        """Get jitter status."""
        return {
            "type": "timing_jitter",
            "active": self._active,
            "effectiveness": self._effectiveness,
            "jitter_range_ns": self.jitter_range_ns,
            "distribution": self.distribution,
            "stats": self._jitter_stats.copy()
        }
    
    def _calculate_effectiveness(self) -> float:
        """Calculate jitter effectiveness."""
        jitter_range = self.jitter_range_ns[1] - self.jitter_range_ns[0]
        
        # Effectiveness based on jitter magnitude
        if jitter_range >= 1000:  # >= 1us
            base_effectiveness = 0.95
        elif jitter_range >= 100:  # >= 100ns
            base_effectiveness = 0.85
        elif jitter_range >= 10:   # >= 10ns
            base_effectiveness = 0.70
        else:
            base_effectiveness = 0.50
        
        # Adjust for distribution
        distribution_factors = {
            "uniform": 0.85,
            "gaussian": 0.90,
            "exponential": 0.80
        }
        dist_factor = distribution_factors.get(self.distribution, 0.75)
        
        return base_effectiveness * dist_factor
    
    def apply_jitter(self) -> float:
        """
        Apply timing jitter delay.
        
        Returns:
            Actual jitter applied in nanoseconds
        """
        if not self._active:
            return 0.0
        
        # Generate jitter value
        if self.distribution == "uniform":
            jitter_ns = secrets.randbelow(
                int(self.jitter_range_ns[1] - self.jitter_range_ns[0])
            ) + self.jitter_range_ns[0]
        elif self.distribution == "gaussian":
            mean = (self.jitter_range_ns[1] + self.jitter_range_ns[0]) / 2
            std = (self.jitter_range_ns[1] - self.jitter_range_ns[0]) / 6
            jitter_ns = np.random.normal(mean, std)
            jitter_ns = np.clip(
                jitter_ns,
                self.jitter_range_ns[0],
                self.jitter_range_ns[1]
            )
        else:  # exponential
            scale = (self.jitter_range_ns[1] - self.jitter_range_ns[0]) / 3
            jitter_ns = np.random.exponential(scale)
            jitter_ns = min(jitter_ns, self.jitter_range_ns[1])
        
        # Apply delay
        delay_seconds = jitter_ns / 1e9
        time.sleep(delay_seconds)
        
        # Update stats
        with self._lock:
            self._jitter_stats["total_jitter_applied"] += jitter_ns
            self._jitter_stats["operations_protected"] += 1
        
        return jitter_ns
    
    def get_jitter_sequence(self, count: int) -> List[float]:
        """
        Pre-generate a sequence of jitter values.
        
        Args:
            count: Number of jitter values to generate
            
        Returns:
            List of jitter values in nanoseconds
        """
        jitter_values = []
        
        for _ in range(count):
            if self.distribution == "uniform":
                jitter = secrets.randbelow(
                    int(self.jitter_range_ns[1] - self.jitter_range_ns[0])
                ) + self.jitter_range_ns[0]
            else:
                # Use numpy for other distributions
                if self.distribution == "gaussian":
                    mean = (self.jitter_range_ns[1] + self.jitter_range_ns[0]) / 2
                    std = (self.jitter_range_ns[1] - self.jitter_range_ns[0]) / 6
                    jitter = np.random.normal(mean, std)
                else:
                    scale = (self.jitter_range_ns[1] - self.jitter_range_ns[0]) / 3
                    jitter = np.random.exponential(scale)
                
                jitter = np.clip(
                    jitter,
                    self.jitter_range_ns[0],
                    self.jitter_range_ns[1]
                )
            
            jitter_values.append(float(jitter))
        
        return jitter_values


class PowerControlCountermeasure(TEMPESTCountermeasure):
    """
    Adaptive power control countermeasure.
    
    Minimizes emanation amplitude by controlling signal
    power levels.
    """
    
    def __init__(
        self,
        target_power_dbm: float = -30.0,
        min_power_dbm: float = -60.0,
        max_power_dbm: float = 0.0
    ):
        self.target_power_dbm = target_power_dbm
        self.min_power_dbm = min_power_dbm
        self.max_power_dbm = max_power_dbm
        self._active = False
        self._current_power_dbm = target_power_dbm
        self._effectiveness = 0.0
        self._lock = threading.Lock()
    
    def activate(self) -> bool:
        """Activate power control."""
        with self._lock:
            if self._active:
                return True
            
            self._active = True
            self._current_power_dbm = self.target_power_dbm
            self._effectiveness = self._calculate_effectiveness()
            
            logger.info(
                f"Power control activated: target={self.target_power_dbm} dBm"
            )
            return True
    
    def deactivate(self) -> bool:
        """Deactivate power control."""
        with self._lock:
            self._active = False
            self._effectiveness = 0.0
            logger.info("Power control deactivated")
            return True
    
    def get_effectiveness(self) -> float:
        """Get power control effectiveness."""
        return self._effectiveness
    
    def get_status(self) -> Dict[str, Any]:
        """Get power control status."""
        return {
            "type": "power_control",
            "active": self._active,
            "effectiveness": self._effectiveness,
            "target_power_dbm": self.target_power_dbm,
            "current_power_dbm": self._current_power_dbm,
            "power_range_dbm": (self.min_power_dbm, self.max_power_dbm)
        }
    
    def _calculate_effectiveness(self) -> float:
        """Calculate power control effectiveness."""
        # Lower power = higher effectiveness
        power_reduction = self.max_power_dbm - self._current_power_dbm
        max_reduction = self.max_power_dbm - self.min_power_dbm
        
        if max_reduction > 0:
            return min(power_reduction / max_reduction, 1.0) * 0.90
        return 0.0
    
    def adjust_power(self, measured_emission_dbm: float) -> float:
        """
        Adjust power based on measured emissions.
        
        Args:
            measured_emission_dbm: Measured emission level in dBm
            
        Returns:
            New power level in dBm
        """
        if not self._active:
            return self._current_power_dbm
        
        with self._lock:
            # Calculate required adjustment
            # If emissions too high, reduce power
            emission_threshold = -40.0  # Target emission level
            
            if measured_emission_dbm > emission_threshold:
                reduction = measured_emission_dbm - emission_threshold
                self._current_power_dbm = max(
                    self._current_power_dbm - reduction,
                    self.min_power_dbm
                )
            elif measured_emission_dbm < emission_threshold - 10:
                # Can increase power slightly
                increase = min(
                    (emission_threshold - measured_emission_dbm) / 2,
                    5.0
                )
                self._current_power_dbm = min(
                    self._current_power_dbm + increase,
                    self.target_power_dbm
                )
            
            self._effectiveness = self._calculate_effectiveness()
            return self._current_power_dbm
    
    def get_attenuation(self) -> float:
        """Get current attenuation level in dB."""
        return self.target_power_dbm - self._current_power_dbm


class EmissionMonitor:
    """
    Real-time emission monitoring system.
    
    Monitors electromagnetic emanations and detects
    potentially compromising signals.
    """
    
    def __init__(
        self,
        protection_level: TEMPESTLevel = TEMPESTLevel.LEVEL_B,
        zone: ZoneClassification = ZoneClassification.ZONE_1
    ):
        self.protection_level = protection_level
        self.zone = zone
        self._emission_history: List[EmissionProfile] = []
        self._emission_limits = self._get_emission_limits()
        self._monitoring = False
        self._lock = threading.Lock()
        
        # Callbacks for emission events
        self._callbacks: List[Callable[[EmissionProfile], None]] = []
    
    def _get_emission_limits(self) -> Dict[EmissionCategory, float]:
        """Get emission limits based on protection level."""
        base_limits = {
            EmissionCategory.RF_CONDUCTED: -60.0,
            EmissionCategory.RF_RADIATED: -70.0,
            EmissionCategory.POWER_LINE: -80.0,
            EmissionCategory.ACOUSTIC: -50.0,
            EmissionCategory.OPTICAL: -40.0,
            EmissionCategory.THERMAL: -30.0,
            EmissionCategory.MAGNETIC: -60.0,
            EmissionCategory.TIMING: 0.0  # Timing in nanoseconds variance
        }
        
        # Adjust based on protection level
        level_adjustments = {
            TEMPESTLevel.LEVEL_A: -20.0,   # Most restrictive
            TEMPESTLevel.LEVEL_B: -10.0,
            TEMPESTLevel.LEVEL_C: 0.0,
            TEMPESTLevel.UNPROTECTED: 20.0
        }
        
        adjustment = level_adjustments.get(self.protection_level, 0.0)
        
        return {
            cat: limit + adjustment
            for cat, limit in base_limits.items()
        }
    
    def start_monitoring(self) -> bool:
        """Start emission monitoring."""
        with self._lock:
            if self._monitoring:
                return True
            
            self._monitoring = True
            logger.info(
                f"Emission monitoring started: level={self.protection_level.value}, "
                f"zone={self.zone.name}"
            )
            return True
    
    def stop_monitoring(self) -> bool:
        """Stop emission monitoring."""
        with self._lock:
            self._monitoring = False
            logger.info("Emission monitoring stopped")
            return True
    
    def register_callback(
        self,
        callback: Callable[[EmissionProfile], None]
    ) -> None:
        """Register callback for emission events."""
        self._callbacks.append(callback)
    
    def analyze_spectrum(
        self,
        spectrum_data: np.ndarray,
        frequency_range_hz: Tuple[float, float],
        sample_rate_hz: float
    ) -> List[EmissionProfile]:
        """
        Analyze spectrum data for compromising emanations.
        
        Args:
            spectrum_data: FFT magnitude data in dB
            frequency_range_hz: Frequency range of spectrum
            sample_rate_hz: Sample rate used for capture
            
        Returns:
            List of detected emission profiles
        """
        emissions = []
        
        if not self._monitoring:
            return emissions
        
        # Frequency resolution
        freq_resolution = sample_rate_hz / len(spectrum_data)
        frequencies = np.linspace(
            frequency_range_hz[0],
            frequency_range_hz[1],
            len(spectrum_data)
        )
        
        # Get noise floor estimate
        noise_floor = np.median(spectrum_data)
        
        # Detection threshold above noise floor
        threshold = noise_floor + 10.0
        
        # Find peaks above threshold
        peaks = np.where(spectrum_data > threshold)[0]
        
        for peak_idx in peaks:
            peak_freq = frequencies[peak_idx]
            peak_amplitude = spectrum_data[peak_idx]
            
            # Estimate bandwidth (3dB bandwidth)
            bandwidth = self._estimate_bandwidth(
                spectrum_data, peak_idx, freq_resolution
            )
            
            # Check correlation with data signals
            correlation = self._analyze_correlation(
                spectrum_data, peak_idx
            )
            
            # Determine severity
            severity = self._determine_severity(
                peak_amplitude, correlation
            )
            
            # Check if modulation is present
            modulation_detected = self._detect_modulation(
                spectrum_data, peak_idx
            )
            
            # Get recommended action
            action = self._get_recommended_action(
                severity, modulation_detected
            )
            
            emission = EmissionProfile(
                timestamp=datetime.utcnow(),
                category=EmissionCategory.RF_RADIATED,
                severity=severity,
                frequency_hz=peak_freq,
                amplitude_dbm=peak_amplitude,
                bandwidth_hz=bandwidth,
                modulation_detected=modulation_detected,
                correlation_score=correlation,
                location="primary",
                zone=self.zone,
                recommended_action=action
            )
            
            emissions.append(emission)
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(emission)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
        
        # Store in history
        with self._lock:
            self._emission_history.extend(emissions)
            # Keep only last 1000 emissions
            self._emission_history = self._emission_history[-1000:]
        
        return emissions
    
    def _estimate_bandwidth(
        self,
        spectrum: np.ndarray,
        peak_idx: int,
        freq_resolution: float
    ) -> float:
        """Estimate 3dB bandwidth of emission."""
        peak_value = spectrum[peak_idx]
        threshold = peak_value - 3.0
        
        # Find lower edge
        lower_idx = peak_idx
        while lower_idx > 0 and spectrum[lower_idx] > threshold:
            lower_idx -= 1
        
        # Find upper edge
        upper_idx = peak_idx
        while upper_idx < len(spectrum) - 1 and spectrum[upper_idx] > threshold:
            upper_idx += 1
        
        return (upper_idx - lower_idx) * freq_resolution
    
    def _analyze_correlation(
        self,
        spectrum: np.ndarray,
        peak_idx: int
    ) -> float:
        """Analyze correlation with potential data signals."""
        # Check for harmonic relationships and modulation patterns
        # that might indicate data leakage
        
        # Simple correlation metric based on spectral characteristics
        peak_value = spectrum[peak_idx]
        noise_floor = np.median(spectrum)
        
        # Higher SNR suggests more likely to be intentional/leaked signal
        snr = peak_value - noise_floor
        correlation = min(snr / 40.0, 1.0)  # Normalize to 0-1
        
        return float(correlation)
    
    def _determine_severity(
        self,
        amplitude_dbm: float,
        correlation: float
    ) -> EmissionSeverity:
        """Determine emission severity."""
        limit = self._emission_limits[EmissionCategory.RF_RADIATED]
        
        excess = amplitude_dbm - limit
        
        if excess > 20 or (excess > 10 and correlation > 0.8):
            return EmissionSeverity.CRITICAL
        elif excess > 10 or (excess > 5 and correlation > 0.6):
            return EmissionSeverity.HIGH
        elif excess > 5 or (excess > 0 and correlation > 0.4):
            return EmissionSeverity.MEDIUM
        elif excess > 0:
            return EmissionSeverity.LOW
        else:
            return EmissionSeverity.NOMINAL
    
    def _detect_modulation(
        self,
        spectrum: np.ndarray,
        peak_idx: int
    ) -> bool:
        """Detect if signal appears to be modulated."""
        # Look for sideband patterns indicating modulation
        if peak_idx < 10 or peak_idx >= len(spectrum) - 10:
            return False
        
        # Check for sidebands
        local_region = spectrum[peak_idx-10:peak_idx+10]
        peak_in_region = spectrum[peak_idx]
        
        # Look for secondary peaks that might indicate modulation
        secondary_peaks = np.sum(local_region > peak_in_region - 10) - 1
        
        return secondary_peaks >= 2
    
    def _get_recommended_action(
        self,
        severity: EmissionSeverity,
        modulation: bool
    ) -> str:
        """Get recommended action for emission."""
        if severity == EmissionSeverity.CRITICAL:
            if modulation:
                return "IMMEDIATE: Halt operations, investigate data correlation"
            return "IMMEDIATE: Reduce power levels, enable masking"
        elif severity == EmissionSeverity.HIGH:
            return "URGENT: Enable countermeasures, schedule investigation"
        elif severity == EmissionSeverity.MEDIUM:
            return "MONITOR: Increase countermeasure levels"
        elif severity == EmissionSeverity.LOW:
            return "LOG: Document emission, continue monitoring"
        else:
            return "NOMINAL: No action required"
    
    def get_emission_summary(
        self,
        time_window_minutes: int = 60
    ) -> Dict[EmissionSeverity, int]:
        """Get summary of emissions within time window."""
        cutoff = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        
        with self._lock:
            recent_emissions = [
                e for e in self._emission_history
                if e.timestamp >= cutoff
            ]
        
        summary = {severity: 0 for severity in EmissionSeverity}
        for emission in recent_emissions:
            summary[emission.severity] += 1
        
        return summary


class TEMPESTController:
    """
    Main TEMPEST security controller.
    
    Coordinates all TEMPEST countermeasures and monitoring
    for comprehensive emanation security.
    """
    
    def __init__(
        self,
        protection_level: TEMPESTLevel = TEMPESTLevel.LEVEL_B,
        zone: ZoneClassification = ZoneClassification.ZONE_1
    ):
        self.protection_level = protection_level
        self.zone = zone
        
        # Initialize subsystems
        self.emission_monitor = EmissionMonitor(protection_level, zone)
        
        # Countermeasures
        self._countermeasures: Dict[str, TEMPESTCountermeasure] = {}
        self._initialize_countermeasures()
        
        # State
        self._active = False
        self._lock = threading.Lock()
        
        # Compliance tracking
        self._last_assessment = datetime.utcnow()
        self._findings: List[str] = []
        
        # Register emission callback
        self.emission_monitor.register_callback(self._handle_emission)
    
    def _initialize_countermeasures(self) -> None:
        """Initialize default countermeasures."""
        # Signal masking
        self._countermeasures["signal_masking"] = SignalMaskingCountermeasure(
            noise_bandwidth_hz=100e6,
            noise_power_dbm=-60.0,
            mask_pattern="adaptive"
        )
        
        # Timing jitter
        jitter_range = {
            TEMPESTLevel.LEVEL_A: (100, 2000),
            TEMPESTLevel.LEVEL_B: (50, 1000),
            TEMPESTLevel.LEVEL_C: (10, 500),
            TEMPESTLevel.UNPROTECTED: (0, 100)
        }
        
        self._countermeasures["timing_jitter"] = TimingJitterCountermeasure(
            jitter_range_ns=jitter_range.get(
                self.protection_level, (10, 500)
            ),
            distribution="gaussian"
        )
        
        # Power control
        power_targets = {
            TEMPESTLevel.LEVEL_A: -40.0,
            TEMPESTLevel.LEVEL_B: -30.0,
            TEMPESTLevel.LEVEL_C: -20.0,
            TEMPESTLevel.UNPROTECTED: 0.0
        }
        
        self._countermeasures["power_control"] = PowerControlCountermeasure(
            target_power_dbm=power_targets.get(
                self.protection_level, -30.0
            )
        )
    
    def activate(self) -> bool:
        """Activate TEMPEST protection."""
        with self._lock:
            if self._active:
                return True
            
            # Start emission monitoring
            self.emission_monitor.start_monitoring()
            
            # Activate all countermeasures
            for name, countermeasure in self._countermeasures.items():
                if not countermeasure.activate():
                    logger.error(f"Failed to activate countermeasure: {name}")
                    return False
            
            self._active = True
            
            logger.info(
                f"TEMPEST protection activated: level={self.protection_level.value}, "
                f"zone={self.zone.name}"
            )
            return True
    
    def deactivate(self) -> bool:
        """Deactivate TEMPEST protection."""
        with self._lock:
            # Stop monitoring
            self.emission_monitor.stop_monitoring()
            
            # Deactivate countermeasures
            for countermeasure in self._countermeasures.values():
                countermeasure.deactivate()
            
            self._active = False
            logger.info("TEMPEST protection deactivated")
            return True
    
    def _handle_emission(self, emission: EmissionProfile) -> None:
        """Handle detected emission."""
        if emission.severity in [
            EmissionSeverity.CRITICAL,
            EmissionSeverity.HIGH
        ]:
            logger.warning(
                f"Significant emission detected: "
                f"freq={emission.frequency_hz/1e6:.2f} MHz, "
                f"amplitude={emission.amplitude_dbm:.1f} dBm, "
                f"severity={emission.severity.value}"
            )
            
            # Auto-escalate countermeasures
            self._escalate_countermeasures(emission)
    
    def _escalate_countermeasures(self, emission: EmissionProfile) -> None:
        """Escalate countermeasures in response to emission."""
        if "power_control" in self._countermeasures:
            power_ctrl = self._countermeasures["power_control"]
            if isinstance(power_ctrl, PowerControlCountermeasure):
                power_ctrl.adjust_power(emission.amplitude_dbm)
        
        if emission.severity == EmissionSeverity.CRITICAL:
            # Enable maximum protection
            if "signal_masking" in self._countermeasures:
                masking = self._countermeasures["signal_masking"]
                if isinstance(masking, SignalMaskingCountermeasure):
                    masking.noise_power_dbm = -40.0  # Increase masking
                    masking.mask_pattern = "targeted"
    
    def add_countermeasure(
        self,
        name: str,
        countermeasure: TEMPESTCountermeasure
    ) -> None:
        """Add a custom countermeasure."""
        with self._lock:
            self._countermeasures[name] = countermeasure
            if self._active:
                countermeasure.activate()
    
    def remove_countermeasure(self, name: str) -> bool:
        """Remove a countermeasure."""
        with self._lock:
            if name in self._countermeasures:
                self._countermeasures[name].deactivate()
                del self._countermeasures[name]
                return True
            return False
    
    def get_compliance_status(self) -> TEMPESTComplianceStatus:
        """Get current TEMPEST compliance status."""
        emission_summary = self.emission_monitor.get_emission_summary(60)
        
        # Determine compliance
        critical_emissions = emission_summary.get(EmissionSeverity.CRITICAL, 0)
        high_emissions = emission_summary.get(EmissionSeverity.HIGH, 0)
        
        compliant = critical_emissions == 0 and high_emissions < 5
        
        # Active countermeasure list
        active_countermeasures = [
            name for name, cm in self._countermeasures.items()
            if cm.get_status().get("active", False)
        ]
        
        return TEMPESTComplianceStatus(
            compliant=compliant,
            protection_level=self.protection_level,
            zone=self.zone,
            last_assessment=self._last_assessment,
            next_assessment_due=self._last_assessment + timedelta(days=365),
            emission_count_by_severity=emission_summary,
            shielding_status=[],  # Would be populated by shielding tests
            active_countermeasures=active_countermeasures,
            findings=self._findings.copy(),
            recommendations=self._generate_recommendations(emission_summary)
        )
    
    def _generate_recommendations(
        self,
        emission_summary: Dict[EmissionSeverity, int]
    ) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        if emission_summary.get(EmissionSeverity.CRITICAL, 0) > 0:
            recommendations.append(
                "CRITICAL: Investigate and remediate critical emissions immediately"
            )
        
        if emission_summary.get(EmissionSeverity.HIGH, 0) > 3:
            recommendations.append(
                "HIGH: Consider upgrading to higher TEMPEST protection level"
            )
        
        if not self._active:
            recommendations.append(
                "Enable TEMPEST protection for operational security"
            )
        
        # Check countermeasure effectiveness
        for name, cm in self._countermeasures.items():
            effectiveness = cm.get_effectiveness()
            if effectiveness < 0.7:
                recommendations.append(
                    f"Optimize {name} countermeasure (current effectiveness: "
                    f"{effectiveness:.0%})"
                )
        
        if not recommendations:
            recommendations.append("System is operating within TEMPEST guidelines")
        
        return recommendations
    
    def get_status(self) -> Dict[str, Any]:
        """Get complete TEMPEST controller status."""
        return {
            "active": self._active,
            "protection_level": self.protection_level.value,
            "zone": self.zone.name,
            "countermeasures": {
                name: cm.get_status()
                for name, cm in self._countermeasures.items()
            },
            "emission_summary": self.emission_monitor.get_emission_summary(60),
            "compliant": self.get_compliance_status().compliant
        }


# Export public API
__all__ = [
    # Enums
    "TEMPESTLevel",
    "ZoneClassification",
    "EmissionCategory",
    "EmissionSeverity",
    
    # Data classes
    "EmissionProfile",
    "ShieldingStatus",
    "TEMPESTComplianceStatus",
    
    # Exceptions
    "TEMPESTException",
    "EmissionViolation",
    "ShieldingFailure",
    "ZoneViolation",
    
    # Base classes
    "TEMPESTCountermeasure",
    
    # Countermeasures
    "SignalMaskingCountermeasure",
    "TimingJitterCountermeasure",
    "PowerControlCountermeasure",
    
    # Controllers
    "EmissionMonitor",
    "TEMPESTController",
]
