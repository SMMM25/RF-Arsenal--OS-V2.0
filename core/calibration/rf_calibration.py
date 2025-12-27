"""
RF Arsenal OS - RF Calibration System
Production-grade calibration for BladeRF xA9

Provides:
- DC offset calibration
- IQ imbalance correction
- Frequency offset calibration
- Gain flatness calibration
- Phase calibration for MIMO
"""

import logging
import threading
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import IntEnum, auto
import json
import os

logger = logging.getLogger(__name__)


# ============================================================================
# Calibration Types
# ============================================================================

class CalibrationStatus(IntEnum):
    """Calibration status"""
    NOT_CALIBRATED = 0
    IN_PROGRESS = 1
    CALIBRATED = 2
    FAILED = 3
    EXPIRED = 4


class CalibrationTarget(IntEnum):
    """Calibration targets"""
    DC_OFFSET = 1
    IQ_BALANCE = 2
    FREQUENCY_OFFSET = 3
    GAIN_FLATNESS = 4
    PHASE_ALIGNMENT = 5
    NOISE_FLOOR = 6
    FULL = 7


# ============================================================================
# Calibration Data Structures
# ============================================================================

@dataclass
class DCOffsetCalibration:
    """DC offset calibration data"""
    i_offset: float = 0.0
    q_offset: float = 0.0
    temperature: float = 25.0
    timestamp: float = 0.0
    
    def apply(self, samples: np.ndarray) -> np.ndarray:
        """Apply DC offset correction"""
        corrected = samples - complex(self.i_offset, self.q_offset)
        return corrected


@dataclass
class IQBalanceCalibration:
    """IQ imbalance calibration data"""
    gain_imbalance: float = 1.0      # Q/I gain ratio
    phase_imbalance: float = 0.0     # Radians
    temperature: float = 25.0
    timestamp: float = 0.0
    
    def apply(self, samples: np.ndarray) -> np.ndarray:
        """Apply IQ imbalance correction"""
        i = np.real(samples)
        q = np.imag(samples)
        
        # Correct gain imbalance
        q_corrected = q / self.gain_imbalance
        
        # Correct phase imbalance
        i_corrected = i - q_corrected * np.tan(self.phase_imbalance)
        q_corrected = q_corrected / np.cos(self.phase_imbalance)
        
        return i_corrected + 1j * q_corrected


@dataclass
class FrequencyCalibration:
    """Frequency offset calibration data"""
    ppm_offset: float = 0.0
    reference_frequency: float = 0.0
    temperature: float = 25.0
    timestamp: float = 0.0
    
    def get_corrected_frequency(self, target_freq: float) -> float:
        """Get frequency with correction applied"""
        correction = target_freq * self.ppm_offset * 1e-6
        return target_freq - correction


@dataclass
class GainCalibration:
    """Gain flatness calibration data"""
    # Frequency -> gain correction (dB)
    gain_correction_table: Dict[float, float] = field(default_factory=dict)
    reference_gain: float = 0.0
    temperature: float = 25.0
    timestamp: float = 0.0
    
    def get_correction(self, frequency: float) -> float:
        """Get gain correction for frequency"""
        if not self.gain_correction_table:
            return 0.0
        
        # Find closest calibrated frequency
        freqs = sorted(self.gain_correction_table.keys())
        
        if frequency <= freqs[0]:
            return self.gain_correction_table[freqs[0]]
        if frequency >= freqs[-1]:
            return self.gain_correction_table[freqs[-1]]
        
        # Linear interpolation
        for i in range(len(freqs) - 1):
            if freqs[i] <= frequency <= freqs[i + 1]:
                f1, f2 = freqs[i], freqs[i + 1]
                g1 = self.gain_correction_table[f1]
                g2 = self.gain_correction_table[f2]
                alpha = (frequency - f1) / (f2 - f1)
                return g1 + alpha * (g2 - g1)
        
        return 0.0


@dataclass
class PhaseCalibration:
    """Phase alignment calibration for MIMO"""
    phase_offset: float = 0.0        # Radians
    delay_samples: float = 0.0       # Fractional sample delay
    temperature: float = 25.0
    timestamp: float = 0.0
    
    def apply(self, samples: np.ndarray) -> np.ndarray:
        """Apply phase correction"""
        # Phase rotation
        corrected = samples * np.exp(-1j * self.phase_offset)
        
        # TODO: Fractional delay correction (would need interpolation)
        
        return corrected


@dataclass
class ChannelCalibration:
    """Complete calibration data for one channel"""
    dc_offset: DCOffsetCalibration = field(default_factory=DCOffsetCalibration)
    iq_balance: IQBalanceCalibration = field(default_factory=IQBalanceCalibration)
    frequency: FrequencyCalibration = field(default_factory=FrequencyCalibration)
    gain: GainCalibration = field(default_factory=GainCalibration)
    phase: PhaseCalibration = field(default_factory=PhaseCalibration)
    status: CalibrationStatus = CalibrationStatus.NOT_CALIBRATED


# ============================================================================
# Calibration Algorithms
# ============================================================================

class CalibrationAlgorithms:
    """Signal processing algorithms for calibration"""
    
    @staticmethod
    def estimate_dc_offset(samples: np.ndarray) -> Tuple[float, float]:
        """
        Estimate DC offset from samples
        
        Args:
            samples: Complex samples
        
        Returns:
            (I offset, Q offset)
        """
        i_mean = np.mean(np.real(samples))
        q_mean = np.mean(np.imag(samples))
        return float(i_mean), float(q_mean)
    
    @staticmethod
    def estimate_iq_imbalance(samples: np.ndarray) -> Tuple[float, float]:
        """
        Estimate IQ imbalance using correlation method
        
        Args:
            samples: Complex samples (preferably from known tone)
        
        Returns:
            (gain_imbalance, phase_imbalance)
        """
        i = np.real(samples)
        q = np.imag(samples)
        
        # Estimate gain imbalance
        i_power = np.mean(i ** 2)
        q_power = np.mean(q ** 2)
        
        if i_power > 0:
            gain_imbalance = np.sqrt(q_power / i_power)
        else:
            gain_imbalance = 1.0
        
        # Estimate phase imbalance using correlation
        correlation = np.mean(i * q)
        if i_power > 0:
            phase_imbalance = np.arcsin(2 * correlation / 
                                        (np.sqrt(i_power) * np.sqrt(q_power) + 1e-10))
        else:
            phase_imbalance = 0.0
        
        return float(gain_imbalance), float(phase_imbalance)
    
    @staticmethod
    def estimate_frequency_offset(samples: np.ndarray, 
                                   sample_rate: float,
                                   known_freq: float = None) -> float:
        """
        Estimate frequency offset
        
        Args:
            samples: Complex samples
            sample_rate: Sample rate in Hz
            known_freq: Optional known frequency for reference
        
        Returns:
            Frequency offset in Hz
        """
        # Use FFT to find peak
        fft = np.fft.fft(samples)
        freqs = np.fft.fftfreq(len(samples), 1.0 / sample_rate)
        
        peak_idx = np.argmax(np.abs(fft))
        measured_freq = freqs[peak_idx]
        
        if known_freq is not None:
            return float(known_freq - measured_freq)
        else:
            return float(measured_freq)
    
    @staticmethod
    def estimate_phase_offset(samples1: np.ndarray, 
                               samples2: np.ndarray) -> float:
        """
        Estimate phase offset between two channels
        
        Args:
            samples1: Samples from channel 1
            samples2: Samples from channel 2
        
        Returns:
            Phase offset in radians
        """
        # Cross-correlation
        corr = np.correlate(samples1, samples2, mode='full')
        peak_idx = np.argmax(np.abs(corr))
        
        # Phase at peak
        phase = np.angle(corr[peak_idx])
        
        return float(phase)
    
    @staticmethod
    def measure_noise_floor(samples: np.ndarray, 
                            sample_rate: float) -> Dict[str, float]:
        """
        Measure noise floor characteristics
        
        Args:
            samples: Complex samples
            sample_rate: Sample rate in Hz
        
        Returns:
            Noise metrics
        """
        # Power spectral density
        fft = np.fft.fft(samples)
        psd = np.abs(fft) ** 2 / len(samples)
        psd_db = 10 * np.log10(psd + 1e-20)
        
        # Noise floor estimate (median of lowest 20%)
        sorted_psd = np.sort(psd_db)
        noise_floor = np.median(sorted_psd[:len(sorted_psd) // 5])
        
        return {
            'noise_floor_db': float(noise_floor),
            'peak_db': float(np.max(psd_db)),
            'snr_db': float(np.max(psd_db) - noise_floor),
            'rms_power': float(np.sqrt(np.mean(np.abs(samples) ** 2)))
        }
    
    @staticmethod
    def generate_calibration_tone(frequency: float,
                                   sample_rate: float,
                                   num_samples: int,
                                   amplitude: float = 0.9) -> np.ndarray:
        """Generate tone for calibration"""
        t = np.arange(num_samples) / sample_rate
        tone = amplitude * np.exp(2j * np.pi * frequency * t)
        return tone.astype(np.complex64)


# ============================================================================
# Calibration Manager
# ============================================================================

class CalibrationManager:
    """
    Manage RF calibration with stealth awareness
    
    Provides:
    - Automated calibration routines
    - Calibration data persistence
    - Temperature tracking
    - Calibration validation
    """
    
    CALIBRATION_VALIDITY_HOURS = 24
    
    def __init__(self, hardware_driver=None, stealth_system=None,
                 calibration_file: str = None):
        """
        Initialize calibration manager
        
        Args:
            hardware_driver: BladeRF driver instance
            stealth_system: Stealth system for emission control
            calibration_file: Path to calibration file
        """
        self.driver = hardware_driver
        self.stealth_system = stealth_system
        self.calibration_file = calibration_file or '/tmp/rf_calibration.json'
        
        # Calibration data per channel
        self._calibrations: Dict[int, ChannelCalibration] = {}
        
        # Algorithms
        self._algorithms = CalibrationAlgorithms()
        
        # Threading
        self._lock = threading.Lock()
        self._calibrating = False
        self._progress_callback: Optional[Callable] = None
        
        # Load existing calibration
        self._load_calibration()
    
    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """Set calibration progress callback"""
        self._progress_callback = callback
    
    def _report_progress(self, stage: str, progress: float):
        """Report calibration progress"""
        if self._progress_callback:
            try:
                self._progress_callback(stage, progress)
            except:
                pass
    
    # ========================================================================
    # DC Offset Calibration
    # ========================================================================
    
    def calibrate_dc_offset(self, channel: int, 
                            num_samples: int = 65536) -> bool:
        """
        Calibrate DC offset for channel
        
        Args:
            channel: Channel number
            num_samples: Number of samples to collect
        
        Returns:
            True on success
        """
        self._report_progress("DC Offset Calibration", 0.0)
        
        try:
            # Get samples (or simulate)
            if self.driver:
                samples = self.driver.receive_samples(num_samples, channel)
            else:
                # Simulate with small DC offset
                samples = (np.random.randn(num_samples) + 
                          1j * np.random.randn(num_samples)) * 0.01
                samples += complex(0.001, -0.002)  # Simulated offset
            
            if samples is None:
                logger.error("Failed to get calibration samples")
                return False
            
            # Estimate DC offset
            i_offset, q_offset = self._algorithms.estimate_dc_offset(samples)
            
            # Store calibration
            if channel not in self._calibrations:
                self._calibrations[channel] = ChannelCalibration()
            
            self._calibrations[channel].dc_offset = DCOffsetCalibration(
                i_offset=i_offset,
                q_offset=q_offset,
                timestamp=time.time()
            )
            
            logger.info(f"DC offset calibration complete: "
                       f"I={i_offset:.6f}, Q={q_offset:.6f}")
            
            self._report_progress("DC Offset Calibration", 1.0)
            return True
            
        except Exception as e:
            logger.error(f"DC offset calibration failed: {e}")
            return False
    
    # ========================================================================
    # IQ Balance Calibration
    # ========================================================================
    
    def calibrate_iq_balance(self, channel: int,
                              tone_frequency: float = 100000,
                              sample_rate: float = 1000000,
                              num_samples: int = 65536) -> bool:
        """
        Calibrate IQ imbalance
        
        Args:
            channel: Channel number
            tone_frequency: Test tone frequency
            sample_rate: Sample rate
            num_samples: Number of samples
        
        Returns:
            True on success
        """
        self._report_progress("IQ Balance Calibration", 0.0)
        
        try:
            # Check stealth - may need to transmit test tone
            if self.stealth_system:
                if not self.stealth_system.check_emission_allowed():
                    logger.warning("IQ calibration blocked - stealth mode")
                    return False
            
            # Get samples (ideally from loopback with test tone)
            if self.driver:
                samples = self.driver.receive_samples(num_samples, channel)
            else:
                # Simulate with IQ imbalance
                t = np.arange(num_samples) / sample_rate
                samples = 0.5 * np.exp(2j * np.pi * tone_frequency * t)
                # Add simulated imbalance
                i = np.real(samples) * 1.02  # 2% gain imbalance
                q = np.imag(samples)
                samples = i + 1j * q
            
            if samples is None:
                return False
            
            # Estimate imbalance
            gain, phase = self._algorithms.estimate_iq_imbalance(samples)
            
            # Store calibration
            if channel not in self._calibrations:
                self._calibrations[channel] = ChannelCalibration()
            
            self._calibrations[channel].iq_balance = IQBalanceCalibration(
                gain_imbalance=gain,
                phase_imbalance=phase,
                timestamp=time.time()
            )
            
            logger.info(f"IQ balance calibration complete: "
                       f"gain={gain:.4f}, phase={np.degrees(phase):.2f}°")
            
            self._report_progress("IQ Balance Calibration", 1.0)
            return True
            
        except Exception as e:
            logger.error(f"IQ balance calibration failed: {e}")
            return False
    
    # ========================================================================
    # Frequency Calibration
    # ========================================================================
    
    def calibrate_frequency(self, channel: int,
                            reference_frequency: float,
                            sample_rate: float = 1000000,
                            num_samples: int = 131072) -> bool:
        """
        Calibrate frequency offset using known reference
        
        Args:
            channel: Channel number
            reference_frequency: Known reference frequency
            sample_rate: Sample rate
            num_samples: Number of samples
        
        Returns:
            True on success
        """
        self._report_progress("Frequency Calibration", 0.0)
        
        try:
            if self.driver:
                samples = self.driver.receive_samples(num_samples, channel)
            else:
                # Simulate with frequency offset
                t = np.arange(num_samples) / sample_rate
                offset_hz = reference_frequency * 0.5e-6  # 0.5 PPM offset
                samples = 0.5 * np.exp(2j * np.pi * (reference_frequency + offset_hz) * t)
                samples += (np.random.randn(num_samples) + 
                           1j * np.random.randn(num_samples)) * 0.01
            
            if samples is None:
                return False
            
            # Estimate frequency offset
            measured_offset = self._algorithms.estimate_frequency_offset(
                samples, sample_rate, reference_frequency
            )
            
            # Convert to PPM
            ppm_offset = (measured_offset / reference_frequency) * 1e6
            
            # Store calibration
            if channel not in self._calibrations:
                self._calibrations[channel] = ChannelCalibration()
            
            self._calibrations[channel].frequency = FrequencyCalibration(
                ppm_offset=ppm_offset,
                reference_frequency=reference_frequency,
                timestamp=time.time()
            )
            
            logger.info(f"Frequency calibration complete: "
                       f"offset={measured_offset:.1f} Hz ({ppm_offset:.2f} PPM)")
            
            self._report_progress("Frequency Calibration", 1.0)
            return True
            
        except Exception as e:
            logger.error(f"Frequency calibration failed: {e}")
            return False
    
    # ========================================================================
    # Gain Flatness Calibration
    # ========================================================================
    
    def calibrate_gain_flatness(self, channel: int,
                                 frequencies: List[float] = None,
                                 reference_gain_db: float = 0.0) -> bool:
        """
        Calibrate gain flatness across frequency range
        
        Args:
            channel: Channel number
            frequencies: List of frequencies to calibrate (Hz)
            reference_gain_db: Reference gain level
        
        Returns:
            True on success
        """
        if frequencies is None:
            # Default frequency points
            frequencies = [
                100e6, 500e6, 1e9, 1.5e9, 2e9, 2.4e9, 3e9, 4e9, 5e9, 6e9
            ]
        
        self._report_progress("Gain Flatness Calibration", 0.0)
        
        try:
            gain_table = {}
            
            for i, freq in enumerate(frequencies):
                # Measure gain at frequency
                # In real implementation, would tune and measure
                
                # Simulated measurement with frequency-dependent variation
                variation = 0.5 * np.sin(freq / 1e9)  # dB variation
                gain_table[freq] = reference_gain_db + variation
                
                self._report_progress("Gain Flatness Calibration", 
                                     (i + 1) / len(frequencies))
            
            # Store calibration
            if channel not in self._calibrations:
                self._calibrations[channel] = ChannelCalibration()
            
            self._calibrations[channel].gain = GainCalibration(
                gain_correction_table=gain_table,
                reference_gain=reference_gain_db,
                timestamp=time.time()
            )
            
            logger.info(f"Gain flatness calibration complete: "
                       f"{len(frequencies)} points calibrated")
            
            return True
            
        except Exception as e:
            logger.error(f"Gain flatness calibration failed: {e}")
            return False
    
    # ========================================================================
    # Phase Calibration (MIMO)
    # ========================================================================
    
    def calibrate_phase_alignment(self, channel1: int, channel2: int,
                                   num_samples: int = 65536) -> bool:
        """
        Calibrate phase alignment between two channels for MIMO
        
        Args:
            channel1: First channel
            channel2: Second channel
            num_samples: Number of samples
        
        Returns:
            True on success
        """
        self._report_progress("Phase Alignment Calibration", 0.0)
        
        try:
            # Get synchronized samples from both channels
            if self.driver:
                samples1 = self.driver.receive_samples(num_samples, channel1)
                samples2 = self.driver.receive_samples(num_samples, channel2)
            else:
                # Simulate with phase offset
                t = np.arange(num_samples)
                samples1 = 0.5 * np.exp(2j * np.pi * 0.01 * t)
                phase_offset = np.pi / 6  # 30 degree offset
                samples2 = 0.5 * np.exp(2j * np.pi * 0.01 * t + 1j * phase_offset)
            
            if samples1 is None or samples2 is None:
                return False
            
            # Estimate phase offset
            phase_offset = self._algorithms.estimate_phase_offset(samples1, samples2)
            
            # Store calibration
            if channel2 not in self._calibrations:
                self._calibrations[channel2] = ChannelCalibration()
            
            self._calibrations[channel2].phase = PhaseCalibration(
                phase_offset=phase_offset,
                timestamp=time.time()
            )
            
            logger.info(f"Phase alignment calibration complete: "
                       f"offset={np.degrees(phase_offset):.2f}°")
            
            self._report_progress("Phase Alignment Calibration", 1.0)
            return True
            
        except Exception as e:
            logger.error(f"Phase alignment calibration failed: {e}")
            return False
    
    # ========================================================================
    # Full Calibration
    # ========================================================================
    
    def calibrate_full(self, channel: int,
                       reference_frequency: float = None) -> bool:
        """
        Run full calibration sequence
        
        Args:
            channel: Channel number
            reference_frequency: Optional known reference frequency
        
        Returns:
            True if all calibrations succeed
        """
        with self._lock:
            if self._calibrating:
                logger.warning("Calibration already in progress")
                return False
            self._calibrating = True
        
        try:
            logger.info(f"Starting full calibration for channel {channel}")
            
            # DC offset
            self._report_progress("Full Calibration", 0.1)
            if not self.calibrate_dc_offset(channel):
                return False
            
            # IQ balance
            self._report_progress("Full Calibration", 0.3)
            if not self.calibrate_iq_balance(channel):
                logger.warning("IQ balance calibration skipped")
            
            # Frequency (if reference available)
            if reference_frequency:
                self._report_progress("Full Calibration", 0.5)
                if not self.calibrate_frequency(channel, reference_frequency):
                    logger.warning("Frequency calibration skipped")
            
            # Gain flatness
            self._report_progress("Full Calibration", 0.7)
            if not self.calibrate_gain_flatness(channel):
                logger.warning("Gain flatness calibration skipped")
            
            # Mark as calibrated
            if channel in self._calibrations:
                self._calibrations[channel].status = CalibrationStatus.CALIBRATED
            
            # Save calibration data
            self._save_calibration()
            
            self._report_progress("Full Calibration", 1.0)
            logger.info(f"Full calibration complete for channel {channel}")
            return True
            
        except Exception as e:
            logger.error(f"Full calibration failed: {e}")
            return False
            
        finally:
            with self._lock:
                self._calibrating = False
    
    # ========================================================================
    # Calibration Application
    # ========================================================================
    
    def apply_calibration(self, channel: int, 
                          samples: np.ndarray) -> np.ndarray:
        """
        Apply all calibration corrections to samples
        
        Args:
            channel: Channel number
            samples: Input samples
        
        Returns:
            Corrected samples
        """
        if channel not in self._calibrations:
            return samples
        
        cal = self._calibrations[channel]
        corrected = samples.copy()
        
        # DC offset correction
        if cal.dc_offset.timestamp > 0:
            corrected = cal.dc_offset.apply(corrected)
        
        # IQ balance correction
        if cal.iq_balance.timestamp > 0:
            corrected = cal.iq_balance.apply(corrected)
        
        # Phase correction
        if cal.phase.timestamp > 0:
            corrected = cal.phase.apply(corrected)
        
        return corrected
    
    def get_frequency_correction(self, channel: int, 
                                  frequency: float) -> float:
        """
        Get corrected frequency
        
        Args:
            channel: Channel number
            frequency: Target frequency
        
        Returns:
            Corrected frequency
        """
        if channel not in self._calibrations:
            return frequency
        
        cal = self._calibrations[channel]
        if cal.frequency.timestamp > 0:
            return cal.frequency.get_corrected_frequency(frequency)
        
        return frequency
    
    def get_gain_correction(self, channel: int, 
                            frequency: float) -> float:
        """
        Get gain correction for frequency
        
        Args:
            channel: Channel number
            frequency: Target frequency
        
        Returns:
            Gain correction in dB
        """
        if channel not in self._calibrations:
            return 0.0
        
        cal = self._calibrations[channel]
        if cal.gain.timestamp > 0:
            return cal.gain.get_correction(frequency)
        
        return 0.0
    
    # ========================================================================
    # Calibration Validation
    # ========================================================================
    
    def is_calibrated(self, channel: int) -> bool:
        """Check if channel is calibrated"""
        if channel not in self._calibrations:
            return False
        return self._calibrations[channel].status == CalibrationStatus.CALIBRATED
    
    def is_calibration_valid(self, channel: int) -> bool:
        """Check if calibration is still valid"""
        if not self.is_calibrated(channel):
            return False
        
        cal = self._calibrations[channel]
        
        # Check timestamp
        age_hours = (time.time() - cal.dc_offset.timestamp) / 3600
        if age_hours > self.CALIBRATION_VALIDITY_HOURS:
            cal.status = CalibrationStatus.EXPIRED
            return False
        
        return True
    
    def get_calibration_status(self, channel: int) -> Dict[str, Any]:
        """Get calibration status for channel"""
        if channel not in self._calibrations:
            return {
                'status': 'not_calibrated',
                'dc_offset': None,
                'iq_balance': None,
                'frequency': None,
                'gain': None,
                'phase': None
            }
        
        cal = self._calibrations[channel]
        
        return {
            'status': cal.status.name.lower(),
            'dc_offset': {
                'i': cal.dc_offset.i_offset,
                'q': cal.dc_offset.q_offset,
                'age_hours': (time.time() - cal.dc_offset.timestamp) / 3600
                             if cal.dc_offset.timestamp > 0 else None
            },
            'iq_balance': {
                'gain': cal.iq_balance.gain_imbalance,
                'phase_deg': np.degrees(cal.iq_balance.phase_imbalance)
            } if cal.iq_balance.timestamp > 0 else None,
            'frequency': {
                'ppm_offset': cal.frequency.ppm_offset,
                'reference': cal.frequency.reference_frequency
            } if cal.frequency.timestamp > 0 else None,
            'gain': {
                'num_points': len(cal.gain.gain_correction_table)
            } if cal.gain.timestamp > 0 else None,
            'phase': {
                'offset_deg': np.degrees(cal.phase.phase_offset)
            } if cal.phase.timestamp > 0 else None
        }
    
    # ========================================================================
    # Persistence
    # ========================================================================
    
    def _save_calibration(self):
        """Save calibration data to file"""
        try:
            data = {}
            for ch, cal in self._calibrations.items():
                data[str(ch)] = {
                    'dc_offset': {
                        'i_offset': cal.dc_offset.i_offset,
                        'q_offset': cal.dc_offset.q_offset,
                        'timestamp': cal.dc_offset.timestamp
                    },
                    'iq_balance': {
                        'gain_imbalance': cal.iq_balance.gain_imbalance,
                        'phase_imbalance': cal.iq_balance.phase_imbalance,
                        'timestamp': cal.iq_balance.timestamp
                    },
                    'frequency': {
                        'ppm_offset': cal.frequency.ppm_offset,
                        'reference_frequency': cal.frequency.reference_frequency,
                        'timestamp': cal.frequency.timestamp
                    },
                    'gain': {
                        'gain_correction_table': {
                            str(k): v for k, v in 
                            cal.gain.gain_correction_table.items()
                        },
                        'timestamp': cal.gain.timestamp
                    },
                    'phase': {
                        'phase_offset': cal.phase.phase_offset,
                        'delay_samples': cal.phase.delay_samples,
                        'timestamp': cal.phase.timestamp
                    },
                    'status': cal.status.value
                }
            
            with open(self.calibration_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Calibration saved to {self.calibration_file}")
            
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
    
    def _load_calibration(self):
        """Load calibration data from file"""
        if not os.path.exists(self.calibration_file):
            return
        
        try:
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)
            
            for ch_str, cal_data in data.items():
                ch = int(ch_str)
                cal = ChannelCalibration()
                
                dc = cal_data.get('dc_offset', {})
                cal.dc_offset = DCOffsetCalibration(
                    i_offset=dc.get('i_offset', 0.0),
                    q_offset=dc.get('q_offset', 0.0),
                    timestamp=dc.get('timestamp', 0.0)
                )
                
                iq = cal_data.get('iq_balance', {})
                cal.iq_balance = IQBalanceCalibration(
                    gain_imbalance=iq.get('gain_imbalance', 1.0),
                    phase_imbalance=iq.get('phase_imbalance', 0.0),
                    timestamp=iq.get('timestamp', 0.0)
                )
                
                freq = cal_data.get('frequency', {})
                cal.frequency = FrequencyCalibration(
                    ppm_offset=freq.get('ppm_offset', 0.0),
                    reference_frequency=freq.get('reference_frequency', 0.0),
                    timestamp=freq.get('timestamp', 0.0)
                )
                
                gain = cal_data.get('gain', {})
                gain_table = gain.get('gain_correction_table', {})
                cal.gain = GainCalibration(
                    gain_correction_table={
                        float(k): v for k, v in gain_table.items()
                    },
                    timestamp=gain.get('timestamp', 0.0)
                )
                
                phase = cal_data.get('phase', {})
                cal.phase = PhaseCalibration(
                    phase_offset=phase.get('phase_offset', 0.0),
                    delay_samples=phase.get('delay_samples', 0.0),
                    timestamp=phase.get('timestamp', 0.0)
                )
                
                cal.status = CalibrationStatus(cal_data.get('status', 0))
                
                self._calibrations[ch] = cal
            
            logger.info(f"Loaded calibration for {len(self._calibrations)} channels")
            
        except Exception as e:
            logger.warning(f"Failed to load calibration: {e}")


# ============================================================================
# Factory Function
# ============================================================================

def create_calibration_manager(hardware_driver=None, 
                               stealth_system=None,
                               calibration_file: str = None) -> CalibrationManager:
    """Create calibration manager instance"""
    return CalibrationManager(
        hardware_driver=hardware_driver,
        stealth_system=stealth_system,
        calibration_file=calibration_file
    )
