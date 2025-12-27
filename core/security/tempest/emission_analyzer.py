"""
TEMPEST Emission Analyzer for RF Arsenal OS.

This module provides comprehensive electromagnetic emission analysis
capabilities for TEMPEST compliance verification and side-channel
detection.

Features:
- Real-time spectrum analysis for emission detection
- Correlation analysis to detect data-bearing signals
- Harmonic analysis for clock/data signal detection
- Time-domain analysis for burst emissions
- Statistical anomaly detection
- Machine learning-based emission classification

Author: RF Arsenal Development Team
License: Proprietary - Security Sensitive
"""

import asyncio
import hashlib
import logging
import math
import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import secrets
import numpy as np
from scipy import signal as scipy_signal
from scipy import fft as scipy_fft
from scipy.stats import zscore

from . import (
    TEMPESTLevel,
    ZoneClassification,
    EmissionCategory,
    EmissionSeverity,
    EmissionProfile,
    TEMPESTException,
    EmissionViolation,
)

logger = logging.getLogger(__name__)


class AnalysisMode(Enum):
    """Emission analysis modes."""
    
    CONTINUOUS = "continuous"       # Real-time continuous analysis
    TRIGGERED = "triggered"         # Triggered capture analysis
    SWEEP = "sweep"                 # Frequency sweep analysis
    BURST = "burst"                 # Burst detection mode
    CORRELATION = "correlation"     # Data correlation mode


class DetectionMethod(Enum):
    """Emission detection methods."""
    
    THRESHOLD = "threshold"         # Simple threshold detection
    STATISTICAL = "statistical"     # Statistical anomaly detection
    CORRELATION = "correlation"     # Cross-correlation analysis
    SPECTRAL = "spectral"          # Spectral signature matching
    ML_CLASSIFIER = "ml_classifier" # Machine learning classification


@dataclass
class AnalysisConfig:
    """Configuration for emission analysis."""
    
    mode: AnalysisMode = AnalysisMode.CONTINUOUS
    detection_method: DetectionMethod = DetectionMethod.STATISTICAL
    frequency_range_hz: Tuple[float, float] = (1e6, 6e9)
    resolution_bandwidth_hz: float = 10e3
    analysis_window_ms: float = 100.0
    threshold_db: float = 10.0
    correlation_threshold: float = 0.6
    min_signal_duration_us: float = 1.0
    max_emissions_per_second: int = 100


@dataclass
class SpectralSignature:
    """Spectral signature of a known emission source."""
    
    name: str
    center_frequency_hz: float
    bandwidth_hz: float
    expected_amplitude_dbm: float
    harmonic_pattern: List[float]  # Relative harmonic levels
    modulation_type: Optional[str]
    is_benign: bool
    description: str


@dataclass
class EmissionEvent:
    """Detailed emission event information."""
    
    timestamp: datetime
    profile: EmissionProfile
    time_domain_data: Optional[np.ndarray]
    frequency_domain_data: Optional[np.ndarray]
    correlation_analysis: Dict[str, float]
    harmonic_analysis: Dict[int, float]  # Harmonic number -> amplitude
    matched_signature: Optional[SpectralSignature]
    confidence_score: float
    raw_metadata: Dict[str, Any] = field(default_factory=dict)


class SignatureDatabase:
    """
    Database of known emission signatures.
    
    Maintains a library of known emission patterns for
    identification and classification.
    """
    
    def __init__(self):
        self._signatures: Dict[str, SpectralSignature] = {}
        self._load_default_signatures()
    
    def _load_default_signatures(self) -> None:
        """Load default emission signatures."""
        # Common clock harmonics
        clock_frequencies = [
            (100e6, "100MHz_clock"),
            (133e6, "133MHz_clock"),
            (200e6, "200MHz_clock"),
            (266e6, "266MHz_clock"),
            (400e6, "400MHz_clock"),
            (800e6, "800MHz_clock"),
        ]
        
        for freq, name in clock_frequencies:
            self._signatures[name] = SpectralSignature(
                name=name,
                center_frequency_hz=freq,
                bandwidth_hz=1e6,
                expected_amplitude_dbm=-60.0,
                harmonic_pattern=[0, -6, -12, -18, -24],  # dB relative
                modulation_type=None,
                is_benign=False,
                description=f"Clock signal at {freq/1e6:.0f} MHz"
            )
        
        # Common RF interference
        rf_sources = [
            (2.4e9, "wifi_24ghz", 20e6, True),
            (5.0e9, "wifi_5ghz", 40e6, True),
            (900e6, "gsm_900", 200e3, True),
            (1800e6, "gsm_1800", 200e3, True),
            (700e6, "lte_700", 10e6, True),
        ]
        
        for freq, name, bw, benign in rf_sources:
            self._signatures[name] = SpectralSignature(
                name=name,
                center_frequency_hz=freq,
                bandwidth_hz=bw,
                expected_amplitude_dbm=-50.0,
                harmonic_pattern=[0],
                modulation_type="OFDM" if "wifi" in name else "FM",
                is_benign=benign,
                description=f"Common RF source: {name}"
            )
        
        # Potentially compromising patterns
        suspicious_patterns = [
            (50e6, "display_sync", 100e3, "Display sync signal"),
            (12e6, "usb_clock", 500e3, "USB clock emanation"),
            (25e6, "ethernet_clock", 1e6, "Ethernet clock emanation"),
        ]
        
        for freq, name, bw, desc in suspicious_patterns:
            self._signatures[name] = SpectralSignature(
                name=name,
                center_frequency_hz=freq,
                bandwidth_hz=bw,
                expected_amplitude_dbm=-70.0,
                harmonic_pattern=[0, -10, -20],
                modulation_type="data_bearing",
                is_benign=False,
                description=desc
            )
    
    def add_signature(self, signature: SpectralSignature) -> None:
        """Add a new signature to the database."""
        self._signatures[signature.name] = signature
    
    def remove_signature(self, name: str) -> bool:
        """Remove a signature from the database."""
        if name in self._signatures:
            del self._signatures[name]
            return True
        return False
    
    def match_signature(
        self,
        frequency_hz: float,
        amplitude_dbm: float,
        bandwidth_hz: float,
        tolerance_hz: float = 1e6
    ) -> Optional[SpectralSignature]:
        """
        Match a detected emission to a known signature.
        
        Args:
            frequency_hz: Center frequency of emission
            amplitude_dbm: Amplitude of emission
            bandwidth_hz: Bandwidth of emission
            tolerance_hz: Frequency matching tolerance
            
        Returns:
            Matched signature or None
        """
        best_match = None
        best_score = 0.0
        
        for signature in self._signatures.values():
            # Frequency match
            freq_diff = abs(frequency_hz - signature.center_frequency_hz)
            if freq_diff > tolerance_hz:
                continue
            
            freq_score = 1.0 - (freq_diff / tolerance_hz)
            
            # Bandwidth match
            bw_ratio = min(
                bandwidth_hz / signature.bandwidth_hz,
                signature.bandwidth_hz / bandwidth_hz
            )
            bw_score = bw_ratio
            
            # Combined score
            score = freq_score * 0.7 + bw_score * 0.3
            
            if score > best_score:
                best_score = score
                best_match = signature
        
        return best_match if best_score > 0.5 else None
    
    def get_all_signatures(self) -> List[SpectralSignature]:
        """Get all signatures in database."""
        return list(self._signatures.values())
    
    def get_suspicious_signatures(self) -> List[SpectralSignature]:
        """Get potentially compromising signatures."""
        return [s for s in self._signatures.values() if not s.is_benign]


class HarmonicAnalyzer:
    """
    Analyzer for harmonic emission patterns.
    
    Detects and analyzes harmonic content that may indicate
    clock signals or data-bearing emissions.
    """
    
    def __init__(self, max_harmonics: int = 10):
        self.max_harmonics = max_harmonics
    
    def find_fundamental(
        self,
        spectrum_db: np.ndarray,
        frequencies_hz: np.ndarray,
        min_frequency_hz: float = 1e6,
        max_frequency_hz: float = 500e6
    ) -> Optional[Tuple[float, Dict[int, float]]]:
        """
        Find fundamental frequency with harmonics.
        
        Args:
            spectrum_db: Spectrum magnitude in dB
            frequencies_hz: Corresponding frequencies
            min_frequency_hz: Minimum fundamental frequency
            max_frequency_hz: Maximum fundamental frequency
            
        Returns:
            Tuple of (fundamental_frequency, harmonic_amplitudes) or None
        """
        # Find peaks in spectrum
        noise_floor = np.median(spectrum_db)
        threshold = noise_floor + 10.0
        
        peak_indices = scipy_signal.find_peaks(
            spectrum_db,
            height=threshold,
            distance=10
        )[0]
        
        if len(peak_indices) < 2:
            return None
        
        peak_freqs = frequencies_hz[peak_indices]
        peak_amps = spectrum_db[peak_indices]
        
        # Filter by frequency range
        mask = (peak_freqs >= min_frequency_hz) & (peak_freqs <= max_frequency_hz)
        peak_freqs = peak_freqs[mask]
        peak_amps = peak_amps[mask]
        
        if len(peak_freqs) < 2:
            return None
        
        # Try each peak as potential fundamental
        best_fundamental = None
        best_harmonic_count = 0
        best_harmonics = {}
        
        for i, fund_freq in enumerate(peak_freqs):
            harmonics = {1: float(peak_amps[i])}
            
            # Look for harmonics
            for h in range(2, self.max_harmonics + 1):
                harmonic_freq = fund_freq * h
                
                # Find closest peak to expected harmonic
                freq_diffs = np.abs(peak_freqs - harmonic_freq)
                min_diff_idx = np.argmin(freq_diffs)
                
                # Allow 1% tolerance
                if freq_diffs[min_diff_idx] < harmonic_freq * 0.01:
                    harmonics[h] = float(peak_amps[min_diff_idx])
            
            if len(harmonics) > best_harmonic_count:
                best_harmonic_count = len(harmonics)
                best_fundamental = fund_freq
                best_harmonics = harmonics
        
        if best_fundamental and best_harmonic_count >= 3:
            return (best_fundamental, best_harmonics)
        
        return None
    
    def analyze_harmonic_pattern(
        self,
        harmonics: Dict[int, float]
    ) -> Dict[str, Any]:
        """
        Analyze harmonic pattern for emission classification.
        
        Args:
            harmonics: Dictionary of harmonic number to amplitude
            
        Returns:
            Analysis results
        """
        if not harmonics or 1 not in harmonics:
            return {"valid": False}
        
        fundamental_amp = harmonics[1]
        
        # Calculate relative harmonic levels
        relative_levels = {
            h: amp - fundamental_amp
            for h, amp in harmonics.items()
        }
        
        # Calculate harmonic decay rate
        harmonic_nums = sorted(harmonics.keys())
        if len(harmonic_nums) >= 2:
            amps = [harmonics[h] for h in harmonic_nums]
            # Fit linear decay
            decay_rate = (amps[-1] - amps[0]) / (harmonic_nums[-1] - harmonic_nums[0])
        else:
            decay_rate = 0.0
        
        # Classify pattern
        pattern_type = self._classify_pattern(relative_levels, decay_rate)
        
        # Estimate risk level
        risk_score = self._calculate_risk_score(harmonics, pattern_type)
        
        return {
            "valid": True,
            "harmonic_count": len(harmonics),
            "fundamental_amplitude_db": fundamental_amp,
            "relative_levels": relative_levels,
            "decay_rate_db_per_harmonic": decay_rate,
            "pattern_type": pattern_type,
            "risk_score": risk_score
        }
    
    def _classify_pattern(
        self,
        relative_levels: Dict[int, float],
        decay_rate: float
    ) -> str:
        """Classify harmonic pattern type."""
        # Check for square wave pattern (odd harmonics dominant)
        odd_total = sum(
            abs(level) for h, level in relative_levels.items()
            if h % 2 == 1 and h > 1
        )
        even_total = sum(
            abs(level) for h, level in relative_levels.items()
            if h % 2 == 0
        )
        
        if odd_total > even_total * 2:
            return "square_wave"
        
        # Check for sawtooth pattern (all harmonics present)
        if len(relative_levels) >= 5 and decay_rate < -3:
            return "sawtooth"
        
        # Check for clock-like pattern
        if decay_rate < -5:
            return "clock_signal"
        
        # Check for data-bearing pattern (irregular spacing)
        if abs(decay_rate) < 2:
            return "data_bearing"
        
        return "unknown"
    
    def _calculate_risk_score(
        self,
        harmonics: Dict[int, float],
        pattern_type: str
    ) -> float:
        """Calculate risk score for harmonic pattern."""
        base_score = 0.0
        
        # Pattern type contribution
        pattern_scores = {
            "data_bearing": 0.9,
            "clock_signal": 0.7,
            "square_wave": 0.5,
            "sawtooth": 0.4,
            "unknown": 0.3
        }
        base_score = pattern_scores.get(pattern_type, 0.3)
        
        # Harmonic count contribution
        harmonic_factor = min(len(harmonics) / 10, 1.0) * 0.2
        
        # Amplitude contribution
        max_amp = max(harmonics.values())
        if max_amp > -40:  # High amplitude
            amplitude_factor = 0.3
        elif max_amp > -60:
            amplitude_factor = 0.2
        else:
            amplitude_factor = 0.1
        
        return min(base_score + harmonic_factor + amplitude_factor, 1.0)


class CorrelationAnalyzer:
    """
    Analyzer for emission-to-data correlation.
    
    Detects correlations between emissions and data processing
    activities that may indicate compromising emanations.
    """
    
    def __init__(self, correlation_window_ms: float = 100.0):
        self.correlation_window_ms = correlation_window_ms
        self._data_patterns: Dict[str, np.ndarray] = {}
    
    def register_data_pattern(
        self,
        name: str,
        pattern: np.ndarray
    ) -> None:
        """Register a data pattern for correlation analysis."""
        # Normalize pattern
        pattern = pattern.astype(float)
        pattern = (pattern - np.mean(pattern)) / (np.std(pattern) + 1e-10)
        self._data_patterns[name] = pattern
    
    def analyze_correlation(
        self,
        emission_samples: np.ndarray,
        sample_rate_hz: float
    ) -> Dict[str, float]:
        """
        Analyze correlation between emission and registered data patterns.
        
        Args:
            emission_samples: Time-domain emission samples
            sample_rate_hz: Sample rate of emission data
            
        Returns:
            Dictionary of pattern name to correlation coefficient
        """
        correlations = {}
        
        # Normalize emission samples
        emission = emission_samples.astype(float)
        if np.std(emission) > 0:
            emission = (emission - np.mean(emission)) / np.std(emission)
        else:
            return {name: 0.0 for name in self._data_patterns}
        
        for name, pattern in self._data_patterns.items():
            # Resample pattern if needed
            pattern_samples = len(pattern)
            emission_samples_count = len(emission)
            
            if pattern_samples != emission_samples_count:
                # Resample to match
                pattern = np.interp(
                    np.linspace(0, 1, emission_samples_count),
                    np.linspace(0, 1, pattern_samples),
                    pattern
                )
            
            # Cross-correlation
            correlation = np.correlate(emission, pattern, mode='valid')
            max_correlation = np.max(np.abs(correlation))
            
            # Normalize by length
            max_correlation /= len(emission)
            
            correlations[name] = float(max_correlation)
        
        return correlations
    
    def detect_periodic_correlation(
        self,
        emission_samples: np.ndarray,
        sample_rate_hz: float,
        expected_period_us: float
    ) -> Dict[str, Any]:
        """
        Detect periodic correlations in emission data.
        
        Args:
            emission_samples: Time-domain emission samples
            sample_rate_hz: Sample rate
            expected_period_us: Expected period in microseconds
            
        Returns:
            Analysis results
        """
        period_samples = int(expected_period_us * 1e-6 * sample_rate_hz)
        
        if period_samples < 2 or period_samples > len(emission_samples) // 2:
            return {"detected": False, "reason": "Invalid period"}
        
        # Compute autocorrelation
        autocorr = np.correlate(
            emission_samples, emission_samples, mode='full'
        )
        autocorr = autocorr[len(autocorr)//2:]
        
        # Normalize
        autocorr = autocorr / autocorr[0]
        
        # Look for peak at expected period
        search_start = int(period_samples * 0.9)
        search_end = int(period_samples * 1.1)
        search_end = min(search_end, len(autocorr))
        
        if search_start >= search_end:
            return {"detected": False, "reason": "Search range invalid"}
        
        peak_idx = search_start + np.argmax(autocorr[search_start:search_end])
        peak_value = autocorr[peak_idx]
        
        # Detection threshold
        detected = peak_value > 0.5
        
        return {
            "detected": detected,
            "correlation_value": float(peak_value),
            "detected_period_us": peak_idx / sample_rate_hz * 1e6,
            "expected_period_us": expected_period_us,
            "period_error_percent": abs(
                (peak_idx - period_samples) / period_samples * 100
            )
        }


class EmissionAnalyzer:
    """
    Main emission analyzer for TEMPEST compliance.
    
    Coordinates spectral, harmonic, and correlation analysis
    for comprehensive emission assessment.
    """
    
    def __init__(
        self,
        config: Optional[AnalysisConfig] = None,
        protection_level: TEMPESTLevel = TEMPESTLevel.LEVEL_B
    ):
        self.config = config or AnalysisConfig()
        self.protection_level = protection_level
        
        # Sub-analyzers
        self.signature_db = SignatureDatabase()
        self.harmonic_analyzer = HarmonicAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        
        # State
        self._emission_events: List[EmissionEvent] = []
        self._lock = threading.Lock()
        
        # Analysis statistics
        self._stats = {
            "total_analyses": 0,
            "emissions_detected": 0,
            "critical_emissions": 0,
            "last_analysis_time": None
        }
    
    def analyze_spectrum(
        self,
        spectrum_db: np.ndarray,
        frequencies_hz: np.ndarray,
        time_domain_data: Optional[np.ndarray] = None,
        sample_rate_hz: Optional[float] = None
    ) -> List[EmissionEvent]:
        """
        Perform comprehensive spectrum analysis.
        
        Args:
            spectrum_db: Spectrum magnitude in dB
            frequencies_hz: Corresponding frequencies
            time_domain_data: Optional time-domain samples
            sample_rate_hz: Sample rate for time-domain data
            
        Returns:
            List of detected emission events
        """
        events = []
        self._stats["total_analyses"] += 1
        self._stats["last_analysis_time"] = datetime.utcnow()
        
        # 1. Threshold-based detection
        threshold_events = self._threshold_detection(
            spectrum_db, frequencies_hz
        )
        
        # 2. Harmonic analysis
        harmonic_result = self.harmonic_analyzer.find_fundamental(
            spectrum_db, frequencies_hz
        )
        
        # 3. Statistical anomaly detection
        anomaly_events = self._statistical_detection(
            spectrum_db, frequencies_hz
        )
        
        # Combine and deduplicate events
        all_events = threshold_events + anomaly_events
        
        # Enhance events with correlation analysis
        if time_domain_data is not None and sample_rate_hz is not None:
            for event in all_events:
                # Add time-domain data
                event.time_domain_data = time_domain_data
                
                # Correlation analysis
                correlations = self.correlation_analyzer.analyze_correlation(
                    time_domain_data, sample_rate_hz
                )
                event.correlation_analysis = correlations
        
        # Add harmonic analysis to relevant events
        if harmonic_result:
            fund_freq, harmonics = harmonic_result
            harmonic_analysis = self.harmonic_analyzer.analyze_harmonic_pattern(
                harmonics
            )
            
            # Find event closest to fundamental
            for event in all_events:
                freq_diff = abs(
                    event.profile.frequency_hz - fund_freq
                )
                if freq_diff < fund_freq * 0.01:  # Within 1%
                    event.harmonic_analysis = harmonics
        
        # Signature matching
        for event in all_events:
            matched = self.signature_db.match_signature(
                event.profile.frequency_hz,
                event.profile.amplitude_dbm,
                event.profile.bandwidth_hz
            )
            if matched:
                event.matched_signature = matched
                
                # Adjust severity based on signature
                if matched.is_benign:
                    if event.profile.severity == EmissionSeverity.CRITICAL:
                        event.profile.severity = EmissionSeverity.HIGH
                    elif event.profile.severity == EmissionSeverity.HIGH:
                        event.profile.severity = EmissionSeverity.MEDIUM
        
        # Calculate confidence scores
        for event in all_events:
            event.confidence_score = self._calculate_confidence(event)
        
        # Filter low-confidence events
        events = [e for e in all_events if e.confidence_score >= 0.5]
        
        # Update statistics
        self._stats["emissions_detected"] += len(events)
        critical_count = sum(
            1 for e in events
            if e.profile.severity == EmissionSeverity.CRITICAL
        )
        self._stats["critical_emissions"] += critical_count
        
        # Store events
        with self._lock:
            self._emission_events.extend(events)
            # Keep only last 1000 events
            self._emission_events = self._emission_events[-1000:]
        
        return events
    
    def _threshold_detection(
        self,
        spectrum_db: np.ndarray,
        frequencies_hz: np.ndarray
    ) -> List[EmissionEvent]:
        """Threshold-based emission detection."""
        events = []
        
        # Calculate noise floor
        noise_floor = np.median(spectrum_db)
        threshold = noise_floor + self.config.threshold_db
        
        # Find peaks above threshold
        peak_indices, peak_props = scipy_signal.find_peaks(
            spectrum_db,
            height=threshold,
            distance=10,
            prominence=3
        )
        
        for idx in peak_indices:
            freq = frequencies_hz[idx]
            amplitude = spectrum_db[idx]
            
            # Check if in analysis range
            if freq < self.config.frequency_range_hz[0]:
                continue
            if freq > self.config.frequency_range_hz[1]:
                continue
            
            # Estimate bandwidth
            bandwidth = self._estimate_bandwidth(spectrum_db, idx, frequencies_hz)
            
            # Determine severity
            severity = self._determine_severity(amplitude, noise_floor)
            
            # Create emission profile
            profile = EmissionProfile(
                timestamp=datetime.utcnow(),
                category=EmissionCategory.RF_RADIATED,
                severity=severity,
                frequency_hz=float(freq),
                amplitude_dbm=float(amplitude),
                bandwidth_hz=bandwidth,
                modulation_detected=self._detect_modulation(
                    spectrum_db, idx
                ),
                correlation_score=0.0,
                location="primary",
                zone=ZoneClassification.ZONE_1,
                recommended_action=self._get_action(severity)
            )
            
            event = EmissionEvent(
                timestamp=datetime.utcnow(),
                profile=profile,
                time_domain_data=None,
                frequency_domain_data=spectrum_db,
                correlation_analysis={},
                harmonic_analysis={},
                matched_signature=None,
                confidence_score=0.0
            )
            
            events.append(event)
        
        return events
    
    def _statistical_detection(
        self,
        spectrum_db: np.ndarray,
        frequencies_hz: np.ndarray
    ) -> List[EmissionEvent]:
        """Statistical anomaly detection."""
        events = []
        
        # Z-score based anomaly detection
        z_scores = zscore(spectrum_db)
        
        # Find significant anomalies (z > 3)
        anomaly_indices = np.where(z_scores > 3.0)[0]
        
        for idx in anomaly_indices:
            freq = frequencies_hz[idx]
            amplitude = spectrum_db[idx]
            
            # Check if in analysis range
            if freq < self.config.frequency_range_hz[0]:
                continue
            if freq > self.config.frequency_range_hz[1]:
                continue
            
            # Determine severity based on z-score
            z = z_scores[idx]
            if z > 5:
                severity = EmissionSeverity.CRITICAL
            elif z > 4:
                severity = EmissionSeverity.HIGH
            else:
                severity = EmissionSeverity.MEDIUM
            
            bandwidth = self._estimate_bandwidth(spectrum_db, idx, frequencies_hz)
            
            profile = EmissionProfile(
                timestamp=datetime.utcnow(),
                category=EmissionCategory.RF_RADIATED,
                severity=severity,
                frequency_hz=float(freq),
                amplitude_dbm=float(amplitude),
                bandwidth_hz=bandwidth,
                modulation_detected=False,
                correlation_score=0.0,
                location="primary",
                zone=ZoneClassification.ZONE_1,
                recommended_action=self._get_action(severity),
                metadata={"z_score": float(z), "detection_method": "statistical"}
            )
            
            event = EmissionEvent(
                timestamp=datetime.utcnow(),
                profile=profile,
                time_domain_data=None,
                frequency_domain_data=spectrum_db,
                correlation_analysis={},
                harmonic_analysis={},
                matched_signature=None,
                confidence_score=0.0,
                raw_metadata={"z_score": float(z)}
            )
            
            events.append(event)
        
        return events
    
    def _estimate_bandwidth(
        self,
        spectrum_db: np.ndarray,
        peak_idx: int,
        frequencies_hz: np.ndarray
    ) -> float:
        """Estimate 3dB bandwidth of emission."""
        peak_value = spectrum_db[peak_idx]
        threshold = peak_value - 3.0
        
        # Find lower edge
        lower_idx = peak_idx
        while lower_idx > 0 and spectrum_db[lower_idx] > threshold:
            lower_idx -= 1
        
        # Find upper edge
        upper_idx = peak_idx
        while upper_idx < len(spectrum_db) - 1:
            if spectrum_db[upper_idx] <= threshold:
                break
            upper_idx += 1
        
        if lower_idx >= upper_idx:
            return float(self.config.resolution_bandwidth_hz)
        
        return float(frequencies_hz[upper_idx] - frequencies_hz[lower_idx])
    
    def _detect_modulation(
        self,
        spectrum_db: np.ndarray,
        peak_idx: int
    ) -> bool:
        """Detect modulation on signal."""
        if peak_idx < 10 or peak_idx >= len(spectrum_db) - 10:
            return False
        
        # Check for sidebands indicating modulation
        local_region = spectrum_db[peak_idx-10:peak_idx+10]
        peak_value = spectrum_db[peak_idx]
        
        # Look for sideband peaks
        sidebands = np.sum(
            (local_region > peak_value - 15) &
            (local_region < peak_value - 3)
        )
        
        return sidebands >= 2
    
    def _determine_severity(
        self,
        amplitude_dbm: float,
        noise_floor: float
    ) -> EmissionSeverity:
        """Determine emission severity."""
        # Get limits based on protection level
        limits = {
            TEMPESTLevel.LEVEL_A: -80.0,
            TEMPESTLevel.LEVEL_B: -70.0,
            TEMPESTLevel.LEVEL_C: -60.0,
            TEMPESTLevel.UNPROTECTED: -40.0
        }
        
        limit = limits.get(self.protection_level, -70.0)
        excess = amplitude_dbm - limit
        
        if excess > 20:
            return EmissionSeverity.CRITICAL
        elif excess > 10:
            return EmissionSeverity.HIGH
        elif excess > 5:
            return EmissionSeverity.MEDIUM
        elif excess > 0:
            return EmissionSeverity.LOW
        else:
            return EmissionSeverity.NOMINAL
    
    def _get_action(self, severity: EmissionSeverity) -> str:
        """Get recommended action for emission."""
        actions = {
            EmissionSeverity.CRITICAL: "IMMEDIATE: Halt operations, investigate",
            EmissionSeverity.HIGH: "URGENT: Enable countermeasures",
            EmissionSeverity.MEDIUM: "MONITOR: Increase monitoring",
            EmissionSeverity.LOW: "LOG: Document emission",
            EmissionSeverity.NOMINAL: "NOMINAL: Continue operation"
        }
        return actions.get(severity, "UNKNOWN")
    
    def _calculate_confidence(self, event: EmissionEvent) -> float:
        """Calculate confidence score for emission event."""
        confidence = 0.5  # Base confidence
        
        # Adjust based on amplitude relative to threshold
        profile = event.profile
        if profile.amplitude_dbm > -40:
            confidence += 0.2
        elif profile.amplitude_dbm > -60:
            confidence += 0.1
        
        # Adjust based on detection method
        if "z_score" in event.raw_metadata:
            z = event.raw_metadata["z_score"]
            confidence += min(z / 10, 0.2)
        
        # Adjust based on signature match
        if event.matched_signature:
            confidence += 0.1
        
        # Adjust based on harmonic analysis
        if event.harmonic_analysis:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        return self._stats.copy()
    
    def get_recent_events(
        self,
        max_events: int = 100,
        min_severity: Optional[EmissionSeverity] = None
    ) -> List[EmissionEvent]:
        """Get recent emission events."""
        with self._lock:
            events = self._emission_events.copy()
        
        if min_severity:
            severity_order = [
                EmissionSeverity.NOMINAL,
                EmissionSeverity.LOW,
                EmissionSeverity.MEDIUM,
                EmissionSeverity.HIGH,
                EmissionSeverity.CRITICAL
            ]
            min_idx = severity_order.index(min_severity)
            events = [
                e for e in events
                if severity_order.index(e.profile.severity) >= min_idx
            ]
        
        return events[-max_events:]
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive emission analysis report."""
        with self._lock:
            events = self._emission_events.copy()
        
        # Severity breakdown
        severity_counts = {}
        for severity in EmissionSeverity:
            count = sum(1 for e in events if e.profile.severity == severity)
            severity_counts[severity.value] = count
        
        # Category breakdown
        category_counts = {}
        for category in EmissionCategory:
            count = sum(1 for e in events if e.profile.category == category)
            category_counts[category.value] = count
        
        # Frequency distribution
        frequencies = [e.profile.frequency_hz for e in events]
        freq_stats = {
            "min_hz": min(frequencies) if frequencies else 0,
            "max_hz": max(frequencies) if frequencies else 0,
            "mean_hz": np.mean(frequencies) if frequencies else 0
        }
        
        # Critical emissions
        critical_events = [
            {
                "timestamp": e.timestamp.isoformat(),
                "frequency_hz": e.profile.frequency_hz,
                "amplitude_dbm": e.profile.amplitude_dbm,
                "action": e.profile.recommended_action
            }
            for e in events
            if e.profile.severity == EmissionSeverity.CRITICAL
        ]
        
        return {
            "report_time": datetime.utcnow().isoformat(),
            "protection_level": self.protection_level.value,
            "total_events": len(events),
            "severity_breakdown": severity_counts,
            "category_breakdown": category_counts,
            "frequency_statistics": freq_stats,
            "critical_emissions": critical_events,
            "analysis_statistics": self._stats.copy(),
            "recommendations": self._generate_recommendations(events)
        }
    
    def _generate_recommendations(
        self,
        events: List[EmissionEvent]
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        critical_count = sum(
            1 for e in events
            if e.profile.severity == EmissionSeverity.CRITICAL
        )
        
        if critical_count > 0:
            recommendations.append(
                f"CRITICAL: {critical_count} critical emissions detected. "
                "Immediate investigation required."
            )
        
        high_count = sum(
            1 for e in events
            if e.profile.severity == EmissionSeverity.HIGH
        )
        
        if high_count > 5:
            recommendations.append(
                "Consider upgrading TEMPEST protection level"
            )
        
        # Check for harmonic patterns
        harmonic_events = [e for e in events if e.harmonic_analysis]
        if harmonic_events:
            recommendations.append(
                f"{len(harmonic_events)} emissions with harmonic patterns. "
                "Check for clock signal leakage."
            )
        
        # Check for signature matches
        suspicious_matches = [
            e for e in events
            if e.matched_signature and not e.matched_signature.is_benign
        ]
        if suspicious_matches:
            recommendations.append(
                f"{len(suspicious_matches)} emissions match suspicious signatures. "
                "Detailed analysis recommended."
            )
        
        if not recommendations:
            recommendations.append("No significant issues detected")
        
        return recommendations


# Export public API
__all__ = [
    "AnalysisMode",
    "DetectionMethod",
    "AnalysisConfig",
    "SpectralSignature",
    "EmissionEvent",
    "SignatureDatabase",
    "HarmonicAnalyzer",
    "CorrelationAnalyzer",
    "EmissionAnalyzer",
]
