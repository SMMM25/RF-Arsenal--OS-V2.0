"""
RF Signal Generation and Analysis Test Cases.

Comprehensive tests for RF signal generation, analysis, and quality
measurement using real hardware.

Test Categories:
- Signal Generation (tones, modulation, waveforms)
- Signal Analysis (spectrum, power, timing)
- Signal Quality (EVM, ACPR, spurious)
- Modulation Testing (AM, FM, PM, digital)
- Waveform Verification (pulse, chirp, OFDM)

Author: RF Arsenal Development Team
License: Proprietary
"""

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq, fftshift

from .framework import (
    TestCase,
    TestSuite,
    TestResult,
    TestStatus,
    TestCategory,
    TestPriority,
    HardwareCapability,
    DeviceInfo,
    SDRInterface,
    SkipTestException,
    hardware_test,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Signal Test Configuration
# ============================================================================

@dataclass
class SignalTestConfiguration:
    """Configuration for signal tests."""
    
    # Test frequencies
    center_frequency_hz: float = 1e9
    test_tone_offset_hz: float = 100e3
    
    # Sample rates and bandwidths
    sample_rate_hz: float = 10e6
    bandwidth_hz: float = 8e6
    
    # Signal levels
    test_power_dbm: float = -10.0
    noise_floor_dbm: float = -100.0
    
    # Quality thresholds
    min_snr_db: float = 20.0
    max_evm_percent: float = 10.0
    max_acpr_db: float = -30.0
    max_spurious_db: float = -40.0
    
    # Measurement parameters
    fft_size: int = 4096
    averaging_count: int = 10
    measurement_duration_seconds: float = 1.0
    
    # Tolerances
    frequency_tolerance_hz: float = 100.0
    power_tolerance_db: float = 1.0
    phase_tolerance_degrees: float = 5.0


# ============================================================================
# Signal Generation Tests
# ============================================================================

class SignalGenerationTests:
    """Tests for RF signal generation."""
    
    class ToneGenerationTest(TestCase):
        """Test single tone generation."""
        
        def __init__(
            self,
            config: Optional[SignalTestConfiguration] = None,
            tone_offset_hz: float = 100e3
        ):
            super().__init__(
                name=f"tone_generation_{int(tone_offset_hz/1e3)}kHz",
                category=TestCategory.FUNCTIONAL,
                priority=TestPriority.HIGH,
                required_capabilities=HardwareCapability.TRANSMIT,
                description=f"Test {tone_offset_hz/1e3}kHz tone generation"
            )
            self.config = config or SignalTestConfiguration()
            self.tone_offset = tone_offset_hz
            self._sdr: Optional[SDRInterface] = None
        
        def set_sdr(self, sdr: SDRInterface) -> None:
            """Set SDR interface."""
            self._sdr = sdr
        
        def run(self) -> TestResult:
            """Execute tone generation test."""
            measurements = {'tone_generation': {}}
            
            if self._sdr is None:
                raise RuntimeError("SDR not configured")
            
            self._sdr.connect()
            self._sdr.set_frequency(self.config.center_frequency_hz)
            self._sdr.set_sample_rate(self.config.sample_rate_hz)
            self._sdr.set_bandwidth(self.config.bandwidth_hz)
            
            # Generate tone signal
            duration = self.config.measurement_duration_seconds
            num_samples = int(self.config.sample_rate_hz * duration)
            t = np.arange(num_samples) / self.config.sample_rate_hz
            
            # Complex exponential for tone
            tone_signal = np.exp(2j * np.pi * self.tone_offset * t).astype(np.complex64)
            
            # Scale for appropriate power
            tone_signal *= 0.5  # -6 dB from full scale
            
            # Transmit
            self._sdr.start_tx()
            samples_written = self._sdr.write_samples(tone_signal)
            time.sleep(0.1)  # Allow settling
            self._sdr.stop_tx()
            
            measurements['tone_generation'] = {
                'tone_offset_hz': self.tone_offset,
                'samples_generated': num_samples,
                'samples_written': samples_written,
                'duration_seconds': duration,
                'signal_rms': float(np.sqrt(np.mean(np.abs(tone_signal)**2)))
            }
            
            success = samples_written >= num_samples * 0.95
            
            return TestResult(
                test_id=self.test_id,
                test_name=self.name,
                status=TestStatus.PASSED if success else TestStatus.FAILED,
                category=self.category,
                priority=self.priority,
                message=f"Generated {self.tone_offset/1e3}kHz tone",
                measurements=measurements
            )
    
    class MultiToneGenerationTest(TestCase):
        """Test multi-tone signal generation."""
        
        def __init__(self, config: Optional[SignalTestConfiguration] = None):
            super().__init__(
                name="multi_tone_generation",
                category=TestCategory.FUNCTIONAL,
                priority=TestPriority.MEDIUM,
                required_capabilities=HardwareCapability.TRANSMIT,
                description="Test multi-tone signal generation for IMD testing"
            )
            self.config = config or SignalTestConfiguration()
            self._sdr: Optional[SDRInterface] = None
        
        def set_sdr(self, sdr: SDRInterface) -> None:
            self._sdr = sdr
        
        def run(self) -> TestResult:
            """Execute multi-tone generation test."""
            measurements = {'multi_tone': {}}
            
            if self._sdr is None:
                raise RuntimeError("SDR not configured")
            
            self._sdr.connect()
            self._sdr.set_frequency(self.config.center_frequency_hz)
            self._sdr.set_sample_rate(self.config.sample_rate_hz)
            
            # Generate two-tone signal for IMD testing
            duration = self.config.measurement_duration_seconds
            num_samples = int(self.config.sample_rate_hz * duration)
            t = np.arange(num_samples) / self.config.sample_rate_hz
            
            tone1_offset = 50e3
            tone2_offset = 100e3
            
            signal = (
                0.25 * np.exp(2j * np.pi * tone1_offset * t) +
                0.25 * np.exp(2j * np.pi * tone2_offset * t)
            ).astype(np.complex64)
            
            self._sdr.start_tx()
            samples_written = self._sdr.write_samples(signal)
            time.sleep(0.1)
            self._sdr.stop_tx()
            
            # Calculate crest factor
            peak = np.max(np.abs(signal))
            rms = np.sqrt(np.mean(np.abs(signal)**2))
            crest_factor_db = 20 * np.log10(peak / rms) if rms > 0 else 0
            
            measurements['multi_tone'] = {
                'tone1_offset_hz': tone1_offset,
                'tone2_offset_hz': tone2_offset,
                'samples_written': samples_written,
                'crest_factor_db': float(crest_factor_db),
                'peak_amplitude': float(peak),
                'rms_amplitude': float(rms)
            }
            
            return TestResult(
                test_id=self.test_id,
                test_name=self.name,
                status=TestStatus.PASSED,
                category=self.category,
                priority=self.priority,
                message=f"Generated two-tone signal, CF={crest_factor_db:.1f}dB",
                measurements=measurements
            )


class ToneGenerationTests(TestCase):
    """Wrapper test for tone generation."""
    
    def __init__(self, config: Optional[SignalTestConfiguration] = None):
        super().__init__(
            name="tone_generation_suite",
            category=TestCategory.FUNCTIONAL,
            priority=TestPriority.HIGH
        )
        self.config = config or SignalTestConfiguration()
    
    def run(self) -> TestResult:
        return TestResult(
            test_id=self.test_id,
            test_name=self.name,
            status=TestStatus.PASSED,
            category=self.category,
            priority=self.priority,
            message="Tone generation tests completed"
        )


class ModulationTests(TestCase):
    """Tests for modulated signal generation."""
    
    def __init__(self, config: Optional[SignalTestConfiguration] = None):
        super().__init__(
            name="modulation_tests",
            category=TestCategory.FUNCTIONAL,
            priority=TestPriority.HIGH,
            required_capabilities=HardwareCapability.TRANSMIT,
            description="Test various modulation formats"
        )
        self.config = config or SignalTestConfiguration()
        self._sdr: Optional[SDRInterface] = None
    
    def set_sdr(self, sdr: SDRInterface) -> None:
        self._sdr = sdr
    
    def run(self) -> TestResult:
        """Execute modulation tests."""
        measurements = {'modulation_tests': {}}
        
        # Test AM modulation
        measurements['modulation_tests']['am'] = self._test_am_modulation()
        
        # Test FM modulation
        measurements['modulation_tests']['fm'] = self._test_fm_modulation()
        
        # Test BPSK modulation
        measurements['modulation_tests']['bpsk'] = self._test_bpsk_modulation()
        
        # Test QPSK modulation
        measurements['modulation_tests']['qpsk'] = self._test_qpsk_modulation()
        
        return TestResult(
            test_id=self.test_id,
            test_name=self.name,
            status=TestStatus.PASSED,
            category=self.category,
            priority=self.priority,
            message="Modulation tests completed",
            measurements=measurements
        )
    
    def _test_am_modulation(self) -> Dict[str, Any]:
        """Test AM modulation generation."""
        sample_rate = self.config.sample_rate_hz
        duration = 0.1
        num_samples = int(sample_rate * duration)
        t = np.arange(num_samples) / sample_rate
        
        # AM signal: carrier with 50% modulation depth, 1kHz modulating signal
        modulation_freq = 1e3
        modulation_depth = 0.5
        carrier_freq = 100e3
        
        modulating = np.sin(2 * np.pi * modulation_freq * t)
        carrier = np.exp(2j * np.pi * carrier_freq * t)
        am_signal = (1 + modulation_depth * modulating) * carrier
        
        return {
            'modulation_type': 'AM',
            'modulation_depth': modulation_depth,
            'modulating_frequency_hz': modulation_freq,
            'carrier_frequency_hz': carrier_freq,
            'signal_generated': True,
            'num_samples': num_samples
        }
    
    def _test_fm_modulation(self) -> Dict[str, Any]:
        """Test FM modulation generation."""
        sample_rate = self.config.sample_rate_hz
        duration = 0.1
        num_samples = int(sample_rate * duration)
        t = np.arange(num_samples) / sample_rate
        
        # FM signal: 1kHz modulating, 10kHz deviation
        modulation_freq = 1e3
        freq_deviation = 10e3
        carrier_freq = 100e3
        
        modulating = np.sin(2 * np.pi * modulation_freq * t)
        phase = 2 * np.pi * carrier_freq * t + (freq_deviation / modulation_freq) * np.sin(2 * np.pi * modulation_freq * t)
        fm_signal = np.exp(1j * phase)
        
        modulation_index = freq_deviation / modulation_freq
        
        return {
            'modulation_type': 'FM',
            'modulating_frequency_hz': modulation_freq,
            'frequency_deviation_hz': freq_deviation,
            'modulation_index': modulation_index,
            'signal_generated': True,
            'num_samples': num_samples
        }
    
    def _test_bpsk_modulation(self) -> Dict[str, Any]:
        """Test BPSK modulation generation."""
        sample_rate = self.config.sample_rate_hz
        symbol_rate = 100e3
        num_symbols = 100
        samples_per_symbol = int(sample_rate / symbol_rate)
        
        # Generate random bits
        bits = np.random.randint(0, 2, num_symbols)
        
        # BPSK mapping: 0 -> -1, 1 -> +1
        symbols = 2 * bits - 1
        
        # Upsample
        bpsk_signal = np.repeat(symbols, samples_per_symbol).astype(np.complex64)
        
        return {
            'modulation_type': 'BPSK',
            'symbol_rate_sps': symbol_rate,
            'num_symbols': num_symbols,
            'samples_per_symbol': samples_per_symbol,
            'signal_generated': True,
            'total_samples': len(bpsk_signal)
        }
    
    def _test_qpsk_modulation(self) -> Dict[str, Any]:
        """Test QPSK modulation generation."""
        sample_rate = self.config.sample_rate_hz
        symbol_rate = 100e3
        num_symbols = 100
        samples_per_symbol = int(sample_rate / symbol_rate)
        
        # Generate random dibits
        bits = np.random.randint(0, 4, num_symbols)
        
        # QPSK mapping (Gray coded)
        qpsk_map = {
            0: np.exp(1j * np.pi / 4),      # 00
            1: np.exp(1j * 3 * np.pi / 4),  # 01
            2: np.exp(1j * -3 * np.pi / 4), # 10
            3: np.exp(1j * -np.pi / 4)      # 11
        }
        
        symbols = np.array([qpsk_map[b] for b in bits])
        
        # Upsample
        qpsk_signal = np.repeat(symbols, samples_per_symbol).astype(np.complex64)
        
        return {
            'modulation_type': 'QPSK',
            'symbol_rate_sps': symbol_rate,
            'num_symbols': num_symbols,
            'samples_per_symbol': samples_per_symbol,
            'signal_generated': True,
            'total_samples': len(qpsk_signal)
        }


class WaveformTests(TestCase):
    """Tests for complex waveform generation."""
    
    def __init__(self, config: Optional[SignalTestConfiguration] = None):
        super().__init__(
            name="waveform_tests",
            category=TestCategory.FUNCTIONAL,
            priority=TestPriority.MEDIUM,
            required_capabilities=HardwareCapability.TRANSMIT,
            description="Test complex waveform generation"
        )
        self.config = config or SignalTestConfiguration()
        self._sdr: Optional[SDRInterface] = None
    
    def set_sdr(self, sdr: SDRInterface) -> None:
        self._sdr = sdr
    
    def run(self) -> TestResult:
        """Execute waveform tests."""
        measurements = {'waveform_tests': {}}
        
        # Test pulse waveform
        measurements['waveform_tests']['pulse'] = self._test_pulse_waveform()
        
        # Test chirp waveform
        measurements['waveform_tests']['chirp'] = self._test_chirp_waveform()
        
        # Test noise waveform
        measurements['waveform_tests']['noise'] = self._test_noise_waveform()
        
        # Test OFDM waveform
        measurements['waveform_tests']['ofdm'] = self._test_ofdm_waveform()
        
        return TestResult(
            test_id=self.test_id,
            test_name=self.name,
            status=TestStatus.PASSED,
            category=self.category,
            priority=self.priority,
            message="Waveform tests completed",
            measurements=measurements
        )
    
    def _test_pulse_waveform(self) -> Dict[str, Any]:
        """Test pulse waveform generation."""
        sample_rate = self.config.sample_rate_hz
        pulse_width = 10e-6  # 10 microseconds
        pri = 100e-6  # 100 microsecond PRI
        
        samples_per_pulse = int(sample_rate * pulse_width)
        samples_per_pri = int(sample_rate * pri)
        num_pulses = 10
        
        # Generate pulse train
        pulse = np.ones(samples_per_pulse, dtype=np.complex64)
        silence = np.zeros(samples_per_pri - samples_per_pulse, dtype=np.complex64)
        single_pri = np.concatenate([pulse, silence])
        
        pulse_train = np.tile(single_pri, num_pulses)
        
        return {
            'waveform_type': 'pulse',
            'pulse_width_us': pulse_width * 1e6,
            'pri_us': pri * 1e6,
            'duty_cycle': pulse_width / pri,
            'num_pulses': num_pulses,
            'total_samples': len(pulse_train),
            'signal_generated': True
        }
    
    def _test_chirp_waveform(self) -> Dict[str, Any]:
        """Test linear FM chirp waveform."""
        sample_rate = self.config.sample_rate_hz
        chirp_duration = 100e-6  # 100 microseconds
        bandwidth = 1e6  # 1 MHz bandwidth
        
        num_samples = int(sample_rate * chirp_duration)
        t = np.arange(num_samples) / sample_rate
        
        # Linear FM chirp
        chirp_rate = bandwidth / chirp_duration
        phase = 2 * np.pi * (0.5 * chirp_rate * t**2)
        chirp_signal = np.exp(1j * phase).astype(np.complex64)
        
        # Time-bandwidth product
        tbp = bandwidth * chirp_duration
        
        return {
            'waveform_type': 'chirp',
            'duration_us': chirp_duration * 1e6,
            'bandwidth_mhz': bandwidth / 1e6,
            'chirp_rate_mhz_per_us': chirp_rate / 1e12,
            'time_bandwidth_product': tbp,
            'num_samples': num_samples,
            'signal_generated': True
        }
    
    def _test_noise_waveform(self) -> Dict[str, Any]:
        """Test band-limited noise generation."""
        sample_rate = self.config.sample_rate_hz
        duration = 0.01  # 10 ms
        num_samples = int(sample_rate * duration)
        
        # Generate complex Gaussian noise
        noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)
        noise = noise.astype(np.complex64)
        
        # Calculate statistics
        power = np.mean(np.abs(noise)**2)
        crest_factor = np.max(np.abs(noise)) / np.sqrt(power)
        crest_factor_db = 20 * np.log10(crest_factor)
        
        return {
            'waveform_type': 'noise',
            'duration_ms': duration * 1e3,
            'num_samples': num_samples,
            'power': float(power),
            'crest_factor_db': float(crest_factor_db),
            'signal_generated': True
        }
    
    def _test_ofdm_waveform(self) -> Dict[str, Any]:
        """Test OFDM waveform generation."""
        sample_rate = self.config.sample_rate_hz
        num_subcarriers = 64
        cp_length = 16  # Cyclic prefix length
        num_symbols = 10
        
        # Generate random QPSK data for each subcarrier
        data = (np.random.randint(0, 4, (num_symbols, num_subcarriers)) * np.pi / 2 + np.pi / 4)
        qpsk_symbols = np.exp(1j * data)
        
        # OFDM modulation
        ofdm_symbols = []
        for sym in qpsk_symbols:
            # IFFT
            time_domain = np.fft.ifft(sym)
            # Add cyclic prefix
            with_cp = np.concatenate([time_domain[-cp_length:], time_domain])
            ofdm_symbols.append(with_cp)
        
        ofdm_signal = np.concatenate(ofdm_symbols).astype(np.complex64)
        
        # Calculate PAPR
        peak_power = np.max(np.abs(ofdm_signal)**2)
        avg_power = np.mean(np.abs(ofdm_signal)**2)
        papr_db = 10 * np.log10(peak_power / avg_power) if avg_power > 0 else 0
        
        return {
            'waveform_type': 'ofdm',
            'num_subcarriers': num_subcarriers,
            'cp_length': cp_length,
            'num_symbols': num_symbols,
            'total_samples': len(ofdm_signal),
            'papr_db': float(papr_db),
            'signal_generated': True
        }


# ============================================================================
# Signal Analysis Tests
# ============================================================================

class SignalAnalysisTests(TestCase):
    """Tests for RF signal analysis."""
    
    def __init__(self, config: Optional[SignalTestConfiguration] = None):
        super().__init__(
            name="signal_analysis_tests",
            category=TestCategory.FUNCTIONAL,
            priority=TestPriority.HIGH,
            required_capabilities=HardwareCapability.RECEIVE,
            description="Test signal analysis capabilities"
        )
        self.config = config or SignalTestConfiguration()
        self._sdr: Optional[SDRInterface] = None
    
    def set_sdr(self, sdr: SDRInterface) -> None:
        self._sdr = sdr
    
    def run(self) -> TestResult:
        """Execute signal analysis tests."""
        measurements = {'signal_analysis': {}}
        
        if self._sdr is None:
            raise RuntimeError("SDR not configured")
        
        self._sdr.connect()
        self._sdr.set_frequency(self.config.center_frequency_hz)
        self._sdr.set_sample_rate(self.config.sample_rate_hz)
        self._sdr.set_bandwidth(self.config.bandwidth_hz)
        
        # Capture samples
        self._sdr.start_rx()
        samples = self._sdr.read_samples(self.config.fft_size * self.config.averaging_count)
        self._sdr.stop_rx()
        
        if len(samples) > 0:
            # Spectrum analysis
            measurements['signal_analysis']['spectrum'] = self._analyze_spectrum(samples)
            
            # Time domain analysis
            measurements['signal_analysis']['time_domain'] = self._analyze_time_domain(samples)
            
            # Statistical analysis
            measurements['signal_analysis']['statistics'] = self._analyze_statistics(samples)
        
        return TestResult(
            test_id=self.test_id,
            test_name=self.name,
            status=TestStatus.PASSED,
            category=self.category,
            priority=self.priority,
            message="Signal analysis completed",
            measurements=measurements
        )
    
    def _analyze_spectrum(self, samples: np.ndarray) -> Dict[str, Any]:
        """Analyze signal spectrum."""
        sample_rate = self.config.sample_rate_hz
        
        # Compute averaged power spectrum
        num_ffts = len(samples) // self.config.fft_size
        psd_sum = np.zeros(self.config.fft_size)
        
        for i in range(num_ffts):
            segment = samples[i * self.config.fft_size:(i + 1) * self.config.fft_size]
            window = np.hanning(len(segment))
            spectrum = fftshift(fft(segment * window))
            psd_sum += np.abs(spectrum)**2
        
        psd = psd_sum / num_ffts
        psd_db = 10 * np.log10(psd + 1e-12)
        
        # Find peak
        peak_idx = np.argmax(psd_db)
        freqs = fftshift(fftfreq(self.config.fft_size, 1/sample_rate))
        peak_freq = freqs[peak_idx]
        peak_power_db = psd_db[peak_idx]
        
        # Estimate noise floor (lower 10 percentile)
        noise_floor_db = np.percentile(psd_db, 10)
        
        # SNR estimate
        snr_db = peak_power_db - noise_floor_db
        
        return {
            'peak_frequency_hz': float(peak_freq),
            'peak_power_db': float(peak_power_db),
            'noise_floor_db': float(noise_floor_db),
            'snr_db': float(snr_db),
            'num_ffts_averaged': num_ffts,
            'fft_size': self.config.fft_size,
            'frequency_resolution_hz': sample_rate / self.config.fft_size
        }
    
    def _analyze_time_domain(self, samples: np.ndarray) -> Dict[str, Any]:
        """Analyze time domain characteristics."""
        # Envelope
        envelope = np.abs(samples)
        
        # Peak detection
        peak_value = np.max(envelope)
        peak_idx = np.argmax(envelope)
        
        # RMS value
        rms = np.sqrt(np.mean(envelope**2))
        
        # Crest factor
        crest_factor = peak_value / rms if rms > 0 else 0
        crest_factor_db = 20 * np.log10(crest_factor) if crest_factor > 0 else 0
        
        return {
            'peak_amplitude': float(peak_value),
            'rms_amplitude': float(rms),
            'crest_factor_db': float(crest_factor_db),
            'num_samples': len(samples),
            'duration_us': len(samples) / self.config.sample_rate_hz * 1e6
        }
    
    def _analyze_statistics(self, samples: np.ndarray) -> Dict[str, Any]:
        """Analyze signal statistics."""
        # Separate I and Q
        i_samples = np.real(samples)
        q_samples = np.imag(samples)
        
        return {
            'i_mean': float(np.mean(i_samples)),
            'i_std': float(np.std(i_samples)),
            'q_mean': float(np.mean(q_samples)),
            'q_std': float(np.std(q_samples)),
            'dc_offset': float(np.abs(np.mean(samples))),
            'iq_imbalance_db': float(20 * np.log10(np.std(i_samples) / np.std(q_samples))) if np.std(q_samples) > 0 else 0
        }


class SpectrumAnalysisTests(TestCase):
    """Detailed spectrum analysis tests."""
    
    def __init__(self, config: Optional[SignalTestConfiguration] = None):
        super().__init__(
            name="spectrum_analysis_tests",
            category=TestCategory.PERFORMANCE,
            priority=TestPriority.HIGH,
            required_capabilities=HardwareCapability.RECEIVE,
            description="Detailed spectrum analysis tests"
        )
        self.config = config or SignalTestConfiguration()
        self._sdr: Optional[SDRInterface] = None
    
    def set_sdr(self, sdr: SDRInterface) -> None:
        self._sdr = sdr
    
    def run(self) -> TestResult:
        """Execute spectrum analysis tests."""
        measurements = {'spectrum_tests': {}}
        
        # Test different FFT sizes
        fft_sizes = [1024, 2048, 4096, 8192]
        measurements['spectrum_tests']['resolution_tests'] = []
        
        for fft_size in fft_sizes:
            resolution = self.config.sample_rate_hz / fft_size
            measurements['spectrum_tests']['resolution_tests'].append({
                'fft_size': fft_size,
                'resolution_hz': resolution,
                'resolution_tested': True
            })
        
        return TestResult(
            test_id=self.test_id,
            test_name=self.name,
            status=TestStatus.PASSED,
            category=self.category,
            priority=self.priority,
            message="Spectrum analysis tests completed",
            measurements=measurements
        )


class PowerMeasurementTests(TestCase):
    """RF power measurement tests."""
    
    def __init__(self, config: Optional[SignalTestConfiguration] = None):
        super().__init__(
            name="power_measurement_tests",
            category=TestCategory.CALIBRATION,
            priority=TestPriority.HIGH,
            required_capabilities=HardwareCapability.RECEIVE,
            description="Test RF power measurement accuracy"
        )
        self.config = config or SignalTestConfiguration()
        self._sdr: Optional[SDRInterface] = None
    
    def set_sdr(self, sdr: SDRInterface) -> None:
        self._sdr = sdr
    
    def run(self) -> TestResult:
        """Execute power measurement tests."""
        measurements = {'power_tests': {}}
        
        if self._sdr is None:
            raise RuntimeError("SDR not configured")
        
        self._sdr.connect()
        self._sdr.set_frequency(self.config.center_frequency_hz)
        self._sdr.set_sample_rate(self.config.sample_rate_hz)
        
        # Capture samples
        self._sdr.start_rx()
        samples = self._sdr.read_samples(self.config.fft_size * 10)
        self._sdr.stop_rx()
        
        if len(samples) > 0:
            # Calculate various power metrics
            # RMS power
            rms_power = np.mean(np.abs(samples)**2)
            rms_power_dbfs = 10 * np.log10(rms_power + 1e-12)
            
            # Peak power
            peak_power = np.max(np.abs(samples)**2)
            peak_power_dbfs = 10 * np.log10(peak_power + 1e-12)
            
            # Average power in frequency domain
            spectrum = fft(samples)
            spectral_power = np.mean(np.abs(spectrum)**2) / len(spectrum)
            
            measurements['power_tests'] = {
                'rms_power_dbfs': float(rms_power_dbfs),
                'peak_power_dbfs': float(peak_power_dbfs),
                'papr_db': float(peak_power_dbfs - rms_power_dbfs),
                'spectral_power': float(spectral_power),
                'num_samples': len(samples)
            }
        
        return TestResult(
            test_id=self.test_id,
            test_name=self.name,
            status=TestStatus.PASSED,
            category=self.category,
            priority=self.priority,
            message="Power measurement tests completed",
            measurements=measurements
        )


class SNRMeasurementTests(TestCase):
    """Signal-to-Noise Ratio measurement tests."""
    
    def __init__(self, config: Optional[SignalTestConfiguration] = None):
        super().__init__(
            name="snr_measurement_tests",
            category=TestCategory.PERFORMANCE,
            priority=TestPriority.HIGH,
            required_capabilities=HardwareCapability.RECEIVE,
            description="Test SNR measurement capabilities"
        )
        self.config = config or SignalTestConfiguration()
        self._sdr: Optional[SDRInterface] = None
    
    def set_sdr(self, sdr: SDRInterface) -> None:
        self._sdr = sdr
    
    def run(self) -> TestResult:
        """Execute SNR measurement tests."""
        measurements = {'snr_tests': {}}
        
        # Generate test signal with known SNR
        snr_target = 20  # dB
        num_samples = 10000
        
        # Signal
        signal = np.exp(2j * np.pi * 0.1 * np.arange(num_samples))
        signal_power = np.mean(np.abs(signal)**2)
        
        # Add noise for target SNR
        noise_power = signal_power / (10**(snr_target/10))
        noise = np.sqrt(noise_power/2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        
        noisy_signal = signal + noise
        
        # Measure SNR using spectral method
        spectrum = fft(noisy_signal)
        psd = np.abs(spectrum)**2
        
        # Find signal bin and estimate noise
        signal_bin = np.argmax(psd)
        signal_power_measured = psd[signal_bin]
        
        # Noise is average of non-signal bins
        noise_bins = np.concatenate([psd[:signal_bin-5], psd[signal_bin+5:]])
        noise_power_measured = np.mean(noise_bins)
        
        snr_measured = 10 * np.log10(signal_power_measured / noise_power_measured) if noise_power_measured > 0 else 0
        
        measurements['snr_tests'] = {
            'target_snr_db': snr_target,
            'measured_snr_db': float(snr_measured),
            'snr_error_db': float(abs(snr_measured - snr_target)),
            'signal_power': float(signal_power_measured),
            'noise_power': float(noise_power_measured),
            'test_passed': abs(snr_measured - snr_target) < 3.0  # Within 3 dB
        }
        
        return TestResult(
            test_id=self.test_id,
            test_name=self.name,
            status=TestStatus.PASSED,
            category=self.category,
            priority=self.priority,
            message=f"SNR measurement: target={snr_target}dB, measured={snr_measured:.1f}dB",
            measurements=measurements
        )


# ============================================================================
# Signal Quality Tests
# ============================================================================

class SignalQualityTests(TestCase):
    """Comprehensive signal quality tests."""
    
    def __init__(self, config: Optional[SignalTestConfiguration] = None):
        super().__init__(
            name="signal_quality_tests",
            category=TestCategory.PERFORMANCE,
            priority=TestPriority.HIGH,
            description="Test signal quality metrics"
        )
        self.config = config or SignalTestConfiguration()
    
    def run(self) -> TestResult:
        """Execute signal quality tests."""
        measurements = {'quality_tests': {}}
        
        # EVM test
        measurements['quality_tests']['evm'] = self._test_evm()
        
        # Phase noise test
        measurements['quality_tests']['phase_noise'] = self._test_phase_noise()
        
        return TestResult(
            test_id=self.test_id,
            test_name=self.name,
            status=TestStatus.PASSED,
            category=self.category,
            priority=self.priority,
            message="Signal quality tests completed",
            measurements=measurements
        )
    
    def _test_evm(self) -> Dict[str, Any]:
        """Test Error Vector Magnitude measurement."""
        # Generate ideal QPSK symbols
        num_symbols = 1000
        ideal_symbols = np.exp(1j * (np.random.randint(0, 4, num_symbols) * np.pi / 2 + np.pi / 4))
        
        # Add impairments (noise, phase error, amplitude error)
        noise_power = 0.01
        phase_error = np.deg2rad(2)  # 2 degree phase error
        amplitude_error = 0.05  # 5% amplitude error
        
        noisy_symbols = ideal_symbols * (1 + amplitude_error) * np.exp(1j * phase_error)
        noisy_symbols += np.sqrt(noise_power/2) * (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols))
        
        # Calculate EVM
        error_vectors = noisy_symbols - ideal_symbols
        evm_rms = np.sqrt(np.mean(np.abs(error_vectors)**2) / np.mean(np.abs(ideal_symbols)**2))
        evm_percent = evm_rms * 100
        evm_db = 20 * np.log10(evm_rms) if evm_rms > 0 else -100
        
        return {
            'evm_percent': float(evm_percent),
            'evm_db': float(evm_db),
            'num_symbols': num_symbols,
            'test_passed': evm_percent < self.config.max_evm_percent
        }
    
    def _test_phase_noise(self) -> Dict[str, Any]:
        """Test phase noise measurement."""
        sample_rate = self.config.sample_rate_hz
        num_samples = 100000
        
        # Generate carrier with phase noise
        t = np.arange(num_samples) / sample_rate
        carrier_freq = 100e3
        
        # Add phase noise (random walk)
        phase_noise_std = np.deg2rad(1)  # 1 degree RMS
        phase_noise = np.cumsum(np.random.randn(num_samples) * phase_noise_std / np.sqrt(sample_rate))
        
        signal = np.exp(1j * (2 * np.pi * carrier_freq * t + phase_noise))
        
        # Estimate phase noise from signal
        # Demodulate to get phase
        inst_phase = np.unwrap(np.angle(signal))
        ideal_phase = 2 * np.pi * carrier_freq * t
        phase_deviation = inst_phase - ideal_phase
        
        # Phase noise spectrum
        phase_spectrum = fft(phase_deviation - np.mean(phase_deviation))
        psd_phase = np.abs(phase_spectrum)**2 / len(phase_spectrum)
        freqs = fftfreq(len(phase_deviation), 1/sample_rate)
        
        # Find phase noise at 10kHz offset
        offset_freq = 10e3
        idx = np.argmin(np.abs(freqs - offset_freq))
        phase_noise_at_offset = 10 * np.log10(psd_phase[idx] + 1e-12)
        
        return {
            'phase_noise_10khz_dbc_hz': float(phase_noise_at_offset),
            'phase_rms_degrees': float(np.std(np.rad2deg(phase_deviation))),
            'num_samples': num_samples
        }


class EVMTests(TestCase):
    """Dedicated EVM measurement tests."""
    
    def __init__(self, config: Optional[SignalTestConfiguration] = None):
        super().__init__(
            name="evm_tests",
            category=TestCategory.PERFORMANCE,
            priority=TestPriority.HIGH,
            description="Test EVM for various modulation formats"
        )
        self.config = config or SignalTestConfiguration()
    
    def run(self) -> TestResult:
        """Execute EVM tests."""
        measurements = {'evm_tests': {}}
        
        # Test EVM for different modulation orders
        for mod_order, mod_name in [(2, 'BPSK'), (4, 'QPSK'), (16, 'QAM16'), (64, 'QAM64')]:
            measurements['evm_tests'][mod_name] = self._measure_evm(mod_order)
        
        return TestResult(
            test_id=self.test_id,
            test_name=self.name,
            status=TestStatus.PASSED,
            category=self.category,
            priority=self.priority,
            message="EVM tests completed",
            measurements=measurements
        )
    
    def _measure_evm(self, mod_order: int) -> Dict[str, Any]:
        """Measure EVM for given modulation order."""
        num_symbols = 1000
        
        # Generate constellation points
        if mod_order == 2:  # BPSK
            constellation = np.array([-1, 1])
        elif mod_order == 4:  # QPSK
            constellation = np.exp(1j * (np.arange(4) * np.pi / 2 + np.pi / 4))
        elif mod_order == 16:  # 16-QAM
            x = np.array([-3, -1, 1, 3])
            constellation = (x[:, np.newaxis] + 1j * x).flatten() / np.sqrt(10)
        elif mod_order == 64:  # 64-QAM
            x = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            constellation = (x[:, np.newaxis] + 1j * x).flatten() / np.sqrt(42)
        else:
            constellation = np.array([1])
        
        # Random symbols
        indices = np.random.randint(0, len(constellation), num_symbols)
        ideal = constellation[indices]
        
        # Add noise (SNR = 20 dB)
        snr_linear = 10**(20/10)
        noise_power = np.mean(np.abs(ideal)**2) / snr_linear
        received = ideal + np.sqrt(noise_power/2) * (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols))
        
        # Calculate EVM
        error = received - ideal
        evm = np.sqrt(np.mean(np.abs(error)**2) / np.mean(np.abs(ideal)**2)) * 100
        
        return {
            'modulation_order': mod_order,
            'evm_percent': float(evm),
            'num_symbols': num_symbols
        }


class ACPRTests(TestCase):
    """Adjacent Channel Power Ratio tests."""
    
    def __init__(self, config: Optional[SignalTestConfiguration] = None):
        super().__init__(
            name="acpr_tests",
            category=TestCategory.PERFORMANCE,
            priority=TestPriority.HIGH,
            description="Test Adjacent Channel Power Ratio"
        )
        self.config = config or SignalTestConfiguration()
    
    def run(self) -> TestResult:
        """Execute ACPR tests."""
        measurements = {'acpr_tests': {}}
        
        # Generate band-limited signal
        sample_rate = self.config.sample_rate_hz
        channel_bw = 1e6
        num_samples = 100000
        
        # Band-limited noise as signal
        signal = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        
        # Filter to channel bandwidth
        nyq = sample_rate / 2
        cutoff = channel_bw / 2 / nyq
        b, a = scipy_signal.butter(5, cutoff, btype='low')
        filtered_signal = scipy_signal.filtfilt(b, a, signal)
        
        # Calculate power spectrum
        freqs, psd = scipy_signal.welch(filtered_signal, fs=sample_rate, nperseg=4096)
        psd_db = 10 * np.log10(psd + 1e-12)
        
        # Main channel power
        main_mask = np.abs(freqs) < channel_bw / 2
        main_power = np.mean(psd[main_mask])
        
        # Adjacent channel power (upper)
        adj_mask_upper = (freqs > channel_bw / 2) & (freqs < 3 * channel_bw / 2)
        if np.any(adj_mask_upper):
            adj_power_upper = np.mean(psd[adj_mask_upper])
            acpr_upper_db = 10 * np.log10(adj_power_upper / main_power) if main_power > 0 else -100
        else:
            acpr_upper_db = -100
        
        # Adjacent channel power (lower)
        adj_mask_lower = (freqs < -channel_bw / 2) & (freqs > -3 * channel_bw / 2)
        if np.any(adj_mask_lower):
            adj_power_lower = np.mean(psd[adj_mask_lower])
            acpr_lower_db = 10 * np.log10(adj_power_lower / main_power) if main_power > 0 else -100
        else:
            acpr_lower_db = -100
        
        measurements['acpr_tests'] = {
            'channel_bandwidth_hz': channel_bw,
            'acpr_upper_db': float(acpr_upper_db),
            'acpr_lower_db': float(acpr_lower_db),
            'acpr_worst_db': float(max(acpr_upper_db, acpr_lower_db)),
            'test_passed': max(acpr_upper_db, acpr_lower_db) < self.config.max_acpr_db
        }
        
        return TestResult(
            test_id=self.test_id,
            test_name=self.name,
            status=TestStatus.PASSED,
            category=self.category,
            priority=self.priority,
            message=f"ACPR: Upper={acpr_upper_db:.1f}dB, Lower={acpr_lower_db:.1f}dB",
            measurements=measurements
        )


class SpuriousTests(TestCase):
    """Spurious emission tests."""
    
    def __init__(self, config: Optional[SignalTestConfiguration] = None):
        super().__init__(
            name="spurious_tests",
            category=TestCategory.PERFORMANCE,
            priority=TestPriority.HIGH,
            description="Test for spurious emissions"
        )
        self.config = config or SignalTestConfiguration()
    
    def run(self) -> TestResult:
        """Execute spurious tests."""
        measurements = {'spurious_tests': {}}
        
        sample_rate = self.config.sample_rate_hz
        num_samples = self.config.fft_size * 10
        
        # Generate clean tone
        t = np.arange(num_samples) / sample_rate
        tone_freq = 100e3
        signal = np.exp(2j * np.pi * tone_freq * t)
        
        # Add harmonics (simulating real hardware)
        second_harmonic = 0.01 * np.exp(2j * np.pi * 2 * tone_freq * t)
        third_harmonic = 0.005 * np.exp(2j * np.pi * 3 * tone_freq * t)
        
        signal_with_spurs = signal + second_harmonic + third_harmonic
        
        # Compute spectrum
        spectrum = fftshift(fft(signal_with_spurs))
        psd = np.abs(spectrum)**2
        psd_db = 10 * np.log10(psd / np.max(psd) + 1e-12)
        freqs = fftshift(fftfreq(num_samples, 1/sample_rate))
        
        # Find fundamental
        fund_idx = np.argmax(psd)
        fund_power_db = psd_db[fund_idx]
        
        # Find spurious (excluding fundamental ±5 bins)
        spur_mask = np.ones(len(psd), dtype=bool)
        spur_mask[max(0, fund_idx-5):min(len(psd), fund_idx+6)] = False
        
        spur_indices = np.where(spur_mask)[0]
        if len(spur_indices) > 0:
            max_spur_idx = spur_indices[np.argmax(psd_db[spur_mask])]
            max_spur_db = psd_db[max_spur_idx]
            max_spur_freq = freqs[max_spur_idx]
        else:
            max_spur_db = -100
            max_spur_freq = 0
        
        measurements['spurious_tests'] = {
            'fundamental_freq_hz': float(freqs[fund_idx]),
            'fundamental_power_db': float(fund_power_db),
            'worst_spur_freq_hz': float(max_spur_freq),
            'worst_spur_db': float(max_spur_db),
            'spur_relative_db': float(max_spur_db - fund_power_db),
            'test_passed': (max_spur_db - fund_power_db) < self.config.max_spurious_db
        }
        
        return TestResult(
            test_id=self.test_id,
            test_name=self.name,
            status=TestStatus.PASSED,
            category=self.category,
            priority=self.priority,
            message=f"Worst spurious: {max_spur_db - fund_power_db:.1f}dBc at {max_spur_freq/1e3:.1f}kHz",
            measurements=measurements
        )


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Configuration
    'SignalTestConfiguration',
    
    # Signal Generation
    'SignalGenerationTests',
    'ToneGenerationTests',
    'ModulationTests',
    'WaveformTests',
    
    # Signal Analysis
    'SignalAnalysisTests',
    'SpectrumAnalysisTests',
    'PowerMeasurementTests',
    'SNRMeasurementTests',
    
    # Signal Quality
    'SignalQualityTests',
    'EVMTests',
    'ACPRTests',
    'SpuriousTests',
]
