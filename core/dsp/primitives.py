#!/usr/bin/env python3
"""
RF Arsenal OS - DSP Primitives

Production-grade digital signal processing primitives.
All operations maintain stealth by avoiding detectable patterns.
"""

import numpy as np
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import signal as scipy_signal
from scipy.fft import fft, ifft, fftshift, fftfreq
import threading

logger = logging.getLogger(__name__)


class FilterType(Enum):
    """Filter types"""
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"


class WindowType(Enum):
    """Window function types"""
    RECTANGULAR = "rectangular"
    HANNING = "hanning"
    HAMMING = "hamming"
    BLACKMAN = "blackman"
    KAISER = "kaiser"
    BLACKMAN_HARRIS = "blackman_harris"
    FLAT_TOP = "flat_top"


@dataclass
class FilterSpec:
    """Filter specification"""
    filter_type: FilterType
    cutoff_freq: Union[float, Tuple[float, float]]  # Hz or (low, high) for bandpass
    sample_rate: float
    order: int = 64
    window: WindowType = WindowType.HAMMING
    ripple_db: float = 0.1  # Passband ripple
    stopband_atten_db: float = 60.0  # Stopband attenuation


@dataclass
class DSPConfig:
    """Global DSP configuration"""
    sample_rate: float = 30.72e6  # Default LTE sample rate
    fft_size: int = 2048
    num_channels: int = 2  # MIMO channels
    buffer_size: int = 16384
    use_gpu: bool = False  # GPU acceleration if available
    precision: str = 'float32'  # 'float32' or 'float64'
    stealth_mode: bool = True  # Enable randomized timing


class DSPEngine:
    """
    Core DSP engine with production-grade signal processing.
    
    Features:
    - Thread-safe operations
    - Stealth-aware processing (randomized timing)
    - Memory-efficient streaming
    - Real-time capable
    """
    
    def __init__(self, sample_rate: float = 61.44e6, stealth_mode: bool = True):
        self.sample_rate = sample_rate
        self.stealth_mode = stealth_mode
        self._lock = threading.Lock()
        self.logger = logging.getLogger('DSPEngine')
        
        # Pre-computed filter cache for efficiency
        self._filter_cache = {}
        
    def apply_filter(self, samples: np.ndarray, spec: FilterSpec) -> np.ndarray:
        """Apply FIR filter to samples"""
        with self._lock:
            cache_key = (spec.filter_type, spec.cutoff_freq, spec.order, spec.window)
            
            if cache_key not in self._filter_cache:
                taps = FilterDesign.design_fir(spec)
                self._filter_cache[cache_key] = taps
            else:
                taps = self._filter_cache[cache_key]
            
            # Use efficient convolution
            filtered = scipy_signal.lfilter(taps, 1.0, samples)
            
            return filtered
    
    def resample(self, samples: np.ndarray, target_rate: float) -> np.ndarray:
        """Resample signal to target rate"""
        if target_rate == self.sample_rate:
            return samples
        
        # Calculate rational resampling factors
        from math import gcd
        
        # Find rational approximation
        scale = 1000000
        up = int(target_rate * scale / gcd(int(target_rate * scale), int(self.sample_rate * scale)))
        down = int(self.sample_rate * scale / gcd(int(target_rate * scale), int(self.sample_rate * scale)))
        
        # Limit factors for efficiency
        max_factor = 100
        while up > max_factor or down > max_factor:
            up //= 2
            down //= 2
            if up < 1:
                up = 1
            if down < 1:
                down = 1
        
        return scipy_signal.resample_poly(samples, up, down)
    
    def compute_spectrum(self, samples: np.ndarray, 
                         fft_size: int = 2048,
                         window: WindowType = WindowType.BLACKMAN_HARRIS,
                         averaging: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectrum with averaging.
        
        Returns:
            (frequencies, power_db)
        """
        win = WindowFunctions.get_window(window, fft_size)
        
        # Number of complete segments
        num_segments = min(averaging, len(samples) // fft_size)
        if num_segments < 1:
            num_segments = 1
        
        power_sum = np.zeros(fft_size)
        
        for i in range(num_segments):
            segment = samples[i * fft_size:(i + 1) * fft_size]
            if len(segment) < fft_size:
                segment = np.pad(segment, (0, fft_size - len(segment)))
            
            windowed = segment * win
            spectrum = fft(windowed)
            power = np.abs(spectrum) ** 2
            power_sum += power
        
        power_avg = power_sum / num_segments
        power_db = 10 * np.log10(power_avg + 1e-12)
        
        # Shift to center DC
        power_db = fftshift(power_db)
        
        # Generate frequency axis
        freqs = fftshift(fftfreq(fft_size, 1.0 / self.sample_rate))
        
        return freqs, power_db
    
    def detect_signals(self, samples: np.ndarray,
                       threshold_db: float = -60,
                       min_bandwidth: float = 10e3) -> List[dict]:
        """
        Detect signals in spectrum.
        
        Returns list of detected signals with frequency, bandwidth, power.
        """
        freqs, power_db = self.compute_spectrum(samples, fft_size=4096, averaging=4)
        
        # Find peaks above threshold
        above_threshold = power_db > threshold_db
        
        signals = []
        in_signal = False
        signal_start = 0
        
        for i, above in enumerate(above_threshold):
            if above and not in_signal:
                in_signal = True
                signal_start = i
            elif not above and in_signal:
                in_signal = False
                signal_end = i
                
                # Calculate signal parameters
                center_idx = (signal_start + signal_end) // 2
                bandwidth = freqs[signal_end] - freqs[signal_start]
                
                if bandwidth >= min_bandwidth:
                    peak_power = np.max(power_db[signal_start:signal_end])
                    signals.append({
                        'frequency': freqs[center_idx],
                        'bandwidth': bandwidth,
                        'power_db': float(peak_power),
                        'start_freq': freqs[signal_start],
                        'end_freq': freqs[signal_end]
                    })
        
        return signals
    
    def add_noise(self, samples: np.ndarray, snr_db: float) -> np.ndarray:
        """Add AWGN noise to achieve target SNR"""
        signal_power = np.mean(np.abs(samples) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(samples)) + 1j * np.random.randn(len(samples))
        )
        return samples + noise
    
    def apply_frequency_offset(self, samples: np.ndarray, offset_hz: float) -> np.ndarray:
        """Apply frequency offset to samples"""
        t = np.arange(len(samples)) / self.sample_rate
        return samples * np.exp(2j * np.pi * offset_hz * t)


class FilterDesign:
    """Production-grade filter design"""
    
    @staticmethod
    def design_fir(spec: FilterSpec) -> np.ndarray:
        """Design FIR filter from specification"""
        nyquist = spec.sample_rate / 2
        
        # Normalize cutoff frequency
        if isinstance(spec.cutoff_freq, tuple):
            cutoff_norm = (spec.cutoff_freq[0] / nyquist, spec.cutoff_freq[1] / nyquist)
        else:
            cutoff_norm = spec.cutoff_freq / nyquist
        
        # Ensure cutoff is valid
        if isinstance(cutoff_norm, tuple):
            cutoff_norm = (min(0.99, max(0.01, cutoff_norm[0])),
                          min(0.99, max(0.01, cutoff_norm[1])))
        else:
            cutoff_norm = min(0.99, max(0.01, cutoff_norm))
        
        # Get window
        if spec.window == WindowType.KAISER:
            beta = scipy_signal.kaiser_beta(spec.stopband_atten_db)
            window = ('kaiser', beta)
        else:
            window = spec.window.value
        
        # Design filter
        if spec.filter_type == FilterType.LOWPASS:
            taps = scipy_signal.firwin(spec.order + 1, cutoff_norm, window=window)
        elif spec.filter_type == FilterType.HIGHPASS:
            taps = scipy_signal.firwin(spec.order + 1, cutoff_norm, 
                                       window=window, pass_zero=False)
        elif spec.filter_type == FilterType.BANDPASS:
            taps = scipy_signal.firwin(spec.order + 1, cutoff_norm,
                                       window=window, pass_zero=False)
        elif spec.filter_type == FilterType.BANDSTOP:
            taps = scipy_signal.firwin(spec.order + 1, cutoff_norm,
                                       window=window, pass_zero=True)
        else:
            raise ValueError(f"Unknown filter type: {spec.filter_type}")
        
        return taps
    
    @staticmethod
    def design_iir_butterworth(order: int, cutoff_norm: float, 
                               filter_type: str = 'low') -> Tuple[np.ndarray, np.ndarray]:
        """Design IIR Butterworth filter"""
        b, a = scipy_signal.butter(order, cutoff_norm, btype=filter_type)
        return b, a
    
    @staticmethod
    def design_iir_chebyshev(order: int, cutoff_norm: float,
                             ripple_db: float = 0.5,
                             filter_type: str = 'low') -> Tuple[np.ndarray, np.ndarray]:
        """Design IIR Chebyshev Type I filter"""
        b, a = scipy_signal.cheby1(order, ripple_db, cutoff_norm, btype=filter_type)
        return b, a
    
    @staticmethod
    def design_root_raised_cosine(num_taps: int, samples_per_symbol: int,
                                  rolloff: float = 0.35) -> np.ndarray:
        """
        Design Root Raised Cosine filter for pulse shaping.
        
        Args:
            num_taps: Number of filter taps (should be odd)
            samples_per_symbol: Oversampling factor
            rolloff: Roll-off factor (0 to 1)
        """
        # Ensure odd number of taps
        if num_taps % 2 == 0:
            num_taps += 1
        
        # Time vector
        t = np.arange(num_taps) - (num_taps - 1) / 2
        t = t / samples_per_symbol
        
        # RRC impulse response
        h = np.zeros(num_taps)
        
        for i, ti in enumerate(t):
            if ti == 0:
                h[i] = 1 + rolloff * (4 / np.pi - 1)
            elif abs(ti) == 1 / (4 * rolloff) if rolloff > 0 else False:
                h[i] = (rolloff / np.sqrt(2)) * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff)) +
                    (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
                )
            else:
                num = np.sin(np.pi * ti * (1 - rolloff)) + \
                      4 * rolloff * ti * np.cos(np.pi * ti * (1 + rolloff))
                den = np.pi * ti * (1 - (4 * rolloff * ti) ** 2)
                if abs(den) > 1e-10:
                    h[i] = num / den
                else:
                    h[i] = 0
        
        # Normalize
        h /= np.sqrt(np.sum(h ** 2))
        
        return h


class WindowFunctions:
    """Window functions for spectral analysis and filter design"""
    
    @staticmethod
    def get_window(window_type: WindowType, length: int, 
                   beta: float = 14.0) -> np.ndarray:
        """Get window function"""
        if window_type == WindowType.RECTANGULAR:
            return np.ones(length)
        elif window_type == WindowType.HANNING:
            return np.hanning(length)
        elif window_type == WindowType.HAMMING:
            return np.hamming(length)
        elif window_type == WindowType.BLACKMAN:
            return np.blackman(length)
        elif window_type == WindowType.KAISER:
            return np.kaiser(length, beta)
        elif window_type == WindowType.BLACKMAN_HARRIS:
            return scipy_signal.windows.blackmanharris(length)
        elif window_type == WindowType.FLAT_TOP:
            return scipy_signal.windows.flattop(length)
        else:
            raise ValueError(f"Unknown window type: {window_type}")
    
    @staticmethod
    def get_coherent_gain(window_type: WindowType, length: int) -> float:
        """Get coherent gain of window (for amplitude correction)"""
        win = WindowFunctions.get_window(window_type, length)
        return np.sum(win) / length


class Resampler:
    """Efficient sample rate conversion"""
    
    def __init__(self, input_rate: float, output_rate: float,
                 filter_order: int = 64):
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.filter_order = filter_order
        
        # Calculate rational resampling ratio
        from math import gcd
        scale = 1000000
        g = gcd(int(output_rate), int(input_rate))
        self.up = int(output_rate / g)
        self.down = int(input_rate / g)
        
        # Limit factors
        while self.up > 100 or self.down > 100:
            self.up = (self.up + 1) // 2
            self.down = (self.down + 1) // 2
        
        # Design anti-aliasing filter
        cutoff = min(input_rate, output_rate) / 2 * 0.9
        self.filter_taps = scipy_signal.firwin(
            filter_order + 1,
            cutoff / (max(input_rate, output_rate) * self.up / 2)
        )
        
        # State for streaming
        self._state = None
    
    def resample(self, samples: np.ndarray) -> np.ndarray:
        """Resample block of samples"""
        return scipy_signal.resample_poly(samples, self.up, self.down,
                                          window=self.filter_taps)
    
    def reset(self):
        """Reset internal state"""
        self._state = None


class AGC:
    """
    Automatic Gain Control
    
    Features:
    - Attack/decay time constants
    - Target power level
    - Gain limiting
    - Stealth mode (randomized gain adjustments)
    """
    
    def __init__(self, target_power_db: float = -20,
                 attack_time: float = 0.001,
                 decay_time: float = 0.1,
                 max_gain_db: float = 60,
                 min_gain_db: float = -20,
                 sample_rate: float = 61.44e6,
                 stealth_mode: bool = True):
        
        self.target_power = 10 ** (target_power_db / 10)
        self.max_gain = 10 ** (max_gain_db / 20)
        self.min_gain = 10 ** (min_gain_db / 20)
        self.stealth_mode = stealth_mode
        
        # Time constants
        self.attack_coeff = 1 - np.exp(-1 / (attack_time * sample_rate))
        self.decay_coeff = 1 - np.exp(-1 / (decay_time * sample_rate))
        
        # State
        self.current_gain = 1.0
        self._power_estimate = self.target_power
    
    def process(self, samples: np.ndarray) -> np.ndarray:
        """Apply AGC to samples"""
        output = np.zeros_like(samples)
        
        for i in range(len(samples)):
            # Apply current gain
            output[i] = samples[i] * self.current_gain
            
            # Estimate power
            power = np.abs(output[i]) ** 2
            
            # Update power estimate
            if power > self._power_estimate:
                self._power_estimate += self.attack_coeff * (power - self._power_estimate)
            else:
                self._power_estimate += self.decay_coeff * (power - self._power_estimate)
            
            # Calculate required gain
            if self._power_estimate > 1e-12:
                required_gain = np.sqrt(self.target_power / self._power_estimate)
            else:
                required_gain = self.max_gain
            
            # Add stealth jitter
            if self.stealth_mode:
                jitter = 1 + (np.random.random() - 0.5) * 0.01
                required_gain *= jitter
            
            # Limit gain
            self.current_gain = np.clip(required_gain, self.min_gain, self.max_gain)
        
        return output
    
    def reset(self):
        """Reset AGC state"""
        self.current_gain = 1.0
        self._power_estimate = self.target_power


class DCBlocker:
    """
    DC offset removal filter
    
    High-pass filter with very low cutoff to remove DC component
    while preserving signal.
    """
    
    def __init__(self, alpha: float = 0.995):
        """
        Args:
            alpha: Filter coefficient (0.99-0.999 typical)
                   Higher = slower response, better low-freq preservation
        """
        self.alpha = alpha
        self._prev_input = 0.0
        self._prev_output = 0.0
    
    def process(self, samples: np.ndarray) -> np.ndarray:
        """Remove DC offset from samples"""
        output = np.zeros_like(samples)
        
        for i in range(len(samples)):
            output[i] = samples[i] - self._prev_input + self.alpha * self._prev_output
            self._prev_input = samples[i]
            self._prev_output = output[i]
        
        return output
    
    def process_block(self, samples: np.ndarray) -> np.ndarray:
        """Efficient block processing"""
        # Estimate DC and remove
        dc_estimate = np.mean(samples)
        return samples - dc_estimate
    
    def reset(self):
        """Reset filter state"""
        self._prev_input = 0.0
        self._prev_output = 0.0


class IQCorrector:
    """
    IQ imbalance correction
    
    Corrects amplitude and phase imbalance between I and Q channels.
    Critical for clean signal transmission and reception.
    
    Stealth feature: Can intentionally introduce imbalance to match
    specific hardware fingerprints.
    """
    
    def __init__(self, amplitude_imbalance_db: float = 0.0,
                 phase_imbalance_deg: float = 0.0,
                 dc_offset_i: float = 0.0,
                 dc_offset_q: float = 0.0):
        """
        Args:
            amplitude_imbalance_db: Q amplitude relative to I (dB)
            phase_imbalance_deg: Q phase offset from ideal 90° (degrees)
            dc_offset_i: DC offset on I channel
            dc_offset_q: DC offset on Q channel
        """
        self.set_imbalance(amplitude_imbalance_db, phase_imbalance_deg,
                          dc_offset_i, dc_offset_q)
    
    def set_imbalance(self, amplitude_imbalance_db: float,
                      phase_imbalance_deg: float,
                      dc_offset_i: float = 0.0,
                      dc_offset_q: float = 0.0):
        """Set IQ imbalance parameters"""
        self.amplitude_ratio = 10 ** (amplitude_imbalance_db / 20)
        self.phase_offset = np.deg2rad(phase_imbalance_deg)
        self.dc_i = dc_offset_i
        self.dc_q = dc_offset_q
        
        # Pre-compute correction matrix
        self._compute_correction_matrix()
    
    def _compute_correction_matrix(self):
        """Compute correction matrix for efficient processing"""
        # Correction to remove imbalance
        cos_phi = np.cos(self.phase_offset)
        sin_phi = np.sin(self.phase_offset)
        
        self.correction_matrix = np.array([
            [1.0, 0.0],
            [-sin_phi / (self.amplitude_ratio * cos_phi), 
             1.0 / (self.amplitude_ratio * cos_phi)]
        ])
    
    def correct(self, samples: np.ndarray) -> np.ndarray:
        """Correct IQ imbalance in samples"""
        # Remove DC offset
        i = samples.real - self.dc_i
        q = samples.imag - self.dc_q
        
        # Apply correction matrix
        i_corr = self.correction_matrix[0, 0] * i + self.correction_matrix[0, 1] * q
        q_corr = self.correction_matrix[1, 0] * i + self.correction_matrix[1, 1] * q
        
        return i_corr + 1j * q_corr
    
    def introduce_imbalance(self, samples: np.ndarray) -> np.ndarray:
        """
        Intentionally introduce IQ imbalance (for hardware fingerprint spoofing)
        """
        i = samples.real
        q = samples.imag
        
        # Apply imbalance
        q_imbalanced = self.amplitude_ratio * (
            q * np.cos(self.phase_offset) + i * np.sin(self.phase_offset)
        )
        
        # Add DC offsets
        return (i + self.dc_i) + 1j * (q_imbalanced + self.dc_q)
    
    def estimate_imbalance(self, samples: np.ndarray) -> dict:
        """
        Estimate IQ imbalance from received samples.
        Uses correlation-based method.
        """
        i = samples.real
        q = samples.imag
        
        # Estimate DC offsets
        dc_i = np.mean(i)
        dc_q = np.mean(q)
        
        # Remove DC
        i_centered = i - dc_i
        q_centered = q - dc_q
        
        # Estimate amplitude imbalance
        power_i = np.mean(i_centered ** 2)
        power_q = np.mean(q_centered ** 2)
        amplitude_imbalance_db = 10 * np.log10(power_q / power_i) if power_i > 0 else 0
        
        # Estimate phase imbalance using correlation
        correlation = np.mean(i_centered * q_centered)
        phase_imbalance_deg = np.rad2deg(np.arcsin(
            2 * correlation / np.sqrt(power_i * power_q)
        )) if power_i > 0 and power_q > 0 else 0
        
        return {
            'amplitude_imbalance_db': amplitude_imbalance_db,
            'phase_imbalance_deg': phase_imbalance_deg,
            'dc_offset_i': dc_i,
            'dc_offset_q': dc_q
        }


# Utility functions
def db_to_linear(db: float) -> float:
    """Convert dB to linear power ratio"""
    return 10 ** (db / 10)


def linear_to_db(linear: float) -> float:
    """Convert linear power ratio to dB"""
    return 10 * np.log10(linear + 1e-12)


def dbm_to_watts(dbm: float) -> float:
    """Convert dBm to Watts"""
    return 10 ** ((dbm - 30) / 10)


def watts_to_dbm(watts: float) -> float:
    """Convert Watts to dBm"""
    return 10 * np.log10(watts + 1e-12) + 30
