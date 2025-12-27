"""
RF Arsenal OS - Spectrum Analyzer and Waterfall Display
Real-time spectrum visualization with FFT analysis, peak detection,
waterfall display, and persistence modes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from collections import deque


class WindowFunction(Enum):
    """FFT window functions"""
    RECTANGULAR = "rectangular"
    HAMMING = "hamming"
    HANNING = "hanning"
    BLACKMAN = "blackman"
    BLACKMAN_HARRIS = "blackman_harris"
    KAISER = "kaiser"
    FLAT_TOP = "flat_top"


class AveragingMode(Enum):
    """Spectrum averaging modes"""
    OFF = "off"
    SAMPLE = "sample"
    LOG_POWER = "log_power"
    VOLTAGE = "voltage"
    MAX_HOLD = "max_hold"
    MIN_HOLD = "min_hold"


@dataclass
class SpectrumPeak:
    """Detected spectrum peak"""
    frequency_hz: float
    power_dbm: float
    bandwidth_hz: float
    snr_db: float
    prominence: float
    index: int


@dataclass
class SpectrumSettings:
    """Spectrum analyzer settings"""
    center_frequency: float = 0.0  # Hz
    span: float = 1e6  # Hz
    rbw: float = 10e3  # Resolution bandwidth Hz
    vbw: float = 10e3  # Video bandwidth Hz
    fft_size: int = 4096
    window: WindowFunction = WindowFunction.BLACKMAN_HARRIS
    averaging_mode: AveragingMode = AveragingMode.LOG_POWER
    averaging_count: int = 10
    reference_level: float = 0.0  # dBm
    scale_db_per_div: float = 10.0
    detector_mode: str = "sample"  # sample, peak, average
    

class SpectrumAnalyzer:
    """
    Production-grade spectrum analyzer for RF signal analysis.
    
    Features:
    - Real-time FFT spectrum computation
    - Multiple window functions
    - Configurable averaging modes
    - Peak detection with threshold
    - Spur detection
    - Channel power measurement
    - Occupied bandwidth measurement
    - Adjacent channel power ratio (ACPR)
    - Memory-efficient operation
    """
    
    def __init__(self,
                 sample_rate: float = 1e6,
                 center_frequency: float = 0.0,
                 fft_size: int = 4096,
                 window: WindowFunction = WindowFunction.BLACKMAN_HARRIS):
        """
        Initialize spectrum analyzer.
        
        Args:
            sample_rate: Sample rate in Hz
            center_frequency: Center frequency in Hz
            fft_size: FFT size (power of 2)
            window: Window function
        """
        self.sample_rate = sample_rate
        self.settings = SpectrumSettings(
            center_frequency=center_frequency,
            span=sample_rate,
            fft_size=fft_size,
            window=window
        )
        
        # Generate window
        self._window = self._generate_window()
        
        # Averaging buffer
        self._avg_buffer: List[np.ndarray] = []
        self._max_hold: Optional[np.ndarray] = None
        self._min_hold: Optional[np.ndarray] = None
        
        # Spectrum data
        self._current_spectrum: Optional[np.ndarray] = None
        self._frequencies: Optional[np.ndarray] = None
        
        # Peak detection
        self._detected_peaks: List[SpectrumPeak] = []
        self._peak_threshold_dbm = -60.0
        
        # Callbacks
        self._update_callbacks: List[Callable] = []
        
        # Threading
        self._lock = threading.Lock()
        
        # Update frequency array
        self._update_frequencies()
        
    def _generate_window(self) -> np.ndarray:
        """Generate FFT window"""
        n = self.settings.fft_size
        
        if self.settings.window == WindowFunction.RECTANGULAR:
            return np.ones(n)
        elif self.settings.window == WindowFunction.HAMMING:
            return np.hamming(n)
        elif self.settings.window == WindowFunction.HANNING:
            return np.hanning(n)
        elif self.settings.window == WindowFunction.BLACKMAN:
            return np.blackman(n)
        elif self.settings.window == WindowFunction.BLACKMAN_HARRIS:
            # 4-term Blackman-Harris
            a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
            k = np.arange(n)
            return a0 - a1*np.cos(2*np.pi*k/n) + a2*np.cos(4*np.pi*k/n) - a3*np.cos(6*np.pi*k/n)
        elif self.settings.window == WindowFunction.KAISER:
            return np.kaiser(n, beta=14)
        elif self.settings.window == WindowFunction.FLAT_TOP:
            a0, a1, a2, a3, a4 = 0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368
            k = np.arange(n)
            return a0 - a1*np.cos(2*np.pi*k/n) + a2*np.cos(4*np.pi*k/n) - a3*np.cos(6*np.pi*k/n) + a4*np.cos(8*np.pi*k/n)
        else:
            return np.ones(n)
            
    def _update_frequencies(self) -> None:
        """Update frequency array based on settings"""
        self._frequencies = np.fft.fftshift(
            np.fft.fftfreq(self.settings.fft_size, 1/self.sample_rate)
        ) + self.settings.center_frequency
        
    def process_samples(self, iq_samples: np.ndarray) -> np.ndarray:
        """
        Process IQ samples and compute spectrum.
        
        Args:
            iq_samples: Complex IQ samples
            
        Returns:
            Power spectrum in dBm
        """
        if len(iq_samples) < self.settings.fft_size:
            # Zero-pad if needed
            iq_samples = np.pad(iq_samples, 
                               (0, self.settings.fft_size - len(iq_samples)),
                               mode='constant')
        elif len(iq_samples) > self.settings.fft_size:
            # Use only what we need
            iq_samples = iq_samples[:self.settings.fft_size]
            
        # Apply window
        windowed = iq_samples * self._window
        
        # Compute FFT
        fft_result = np.fft.fftshift(np.fft.fft(windowed))
        
        # Compute power spectrum in dBm
        # Assuming 50 ohm impedance and normalized samples
        power_linear = np.abs(fft_result)**2 / self.settings.fft_size**2
        
        # Convert to dBm (reference to 1mW into 50 ohms)
        # Power = V^2/R, 1mW = 0 dBm
        with np.errstate(divide='ignore'):
            power_dbm = 10 * np.log10(power_linear + 1e-20) + 30  # Add 30 to convert from dBW to dBm
            
        # Apply averaging
        spectrum_dbm = self._apply_averaging(power_dbm)
        
        with self._lock:
            self._current_spectrum = spectrum_dbm
            
        # Detect peaks
        self._detect_peaks(spectrum_dbm)
        
        # Notify callbacks
        for callback in self._update_callbacks:
            callback(self.get_spectrum_data())
            
        return spectrum_dbm
        
    def _apply_averaging(self, power_dbm: np.ndarray) -> np.ndarray:
        """Apply selected averaging mode"""
        if self.settings.averaging_mode == AveragingMode.OFF:
            return power_dbm
            
        elif self.settings.averaging_mode == AveragingMode.MAX_HOLD:
            if self._max_hold is None:
                self._max_hold = power_dbm.copy()
            else:
                self._max_hold = np.maximum(self._max_hold, power_dbm)
            return self._max_hold
            
        elif self.settings.averaging_mode == AveragingMode.MIN_HOLD:
            if self._min_hold is None:
                self._min_hold = power_dbm.copy()
            else:
                self._min_hold = np.minimum(self._min_hold, power_dbm)
            return self._min_hold
            
        elif self.settings.averaging_mode == AveragingMode.LOG_POWER:
            self._avg_buffer.append(power_dbm)
            if len(self._avg_buffer) > self.settings.averaging_count:
                self._avg_buffer.pop(0)
            return np.mean(self._avg_buffer, axis=0)
            
        elif self.settings.averaging_mode == AveragingMode.VOLTAGE:
            # Average in linear voltage domain
            linear = 10**(power_dbm / 10)
            self._avg_buffer.append(np.sqrt(linear))
            if len(self._avg_buffer) > self.settings.averaging_count:
                self._avg_buffer.pop(0)
            avg_voltage = np.mean(self._avg_buffer, axis=0)
            return 10 * np.log10(avg_voltage**2 + 1e-20)
            
        else:
            return power_dbm
            
    def _detect_peaks(self, spectrum_dbm: np.ndarray) -> None:
        """Detect peaks in spectrum"""
        self._detected_peaks = []
        
        if self._frequencies is None:
            return
            
        # Simple peak detection
        for i in range(2, len(spectrum_dbm) - 2):
            if (spectrum_dbm[i] > spectrum_dbm[i-1] and 
                spectrum_dbm[i] > spectrum_dbm[i+1] and
                spectrum_dbm[i] > spectrum_dbm[i-2] and
                spectrum_dbm[i] > spectrum_dbm[i+2] and
                spectrum_dbm[i] > self._peak_threshold_dbm):
                
                # Calculate prominence
                left_min = np.min(spectrum_dbm[max(0, i-10):i])
                right_min = np.min(spectrum_dbm[i:min(len(spectrum_dbm), i+10)])
                prominence = spectrum_dbm[i] - max(left_min, right_min)
                
                # Estimate bandwidth at -3dB
                threshold_3db = spectrum_dbm[i] - 3
                left_idx = i
                right_idx = i
                while left_idx > 0 and spectrum_dbm[left_idx] > threshold_3db:
                    left_idx -= 1
                while right_idx < len(spectrum_dbm)-1 and spectrum_dbm[right_idx] > threshold_3db:
                    right_idx += 1
                bandwidth = abs(self._frequencies[right_idx] - self._frequencies[left_idx])
                
                # Estimate SNR (peak vs noise floor)
                noise_floor = np.median(spectrum_dbm)
                snr = spectrum_dbm[i] - noise_floor
                
                peak = SpectrumPeak(
                    frequency_hz=float(self._frequencies[i]),
                    power_dbm=float(spectrum_dbm[i]),
                    bandwidth_hz=float(bandwidth),
                    snr_db=float(snr),
                    prominence=float(prominence),
                    index=i
                )
                self._detected_peaks.append(peak)
                
        # Sort by power
        self._detected_peaks.sort(key=lambda p: p.power_dbm, reverse=True)
        
    def measure_channel_power(self, 
                             center_freq: float, 
                             bandwidth: float) -> Dict[str, float]:
        """
        Measure integrated channel power.
        
        Args:
            center_freq: Channel center frequency
            bandwidth: Channel bandwidth
            
        Returns:
            Channel power measurements
        """
        if self._current_spectrum is None or self._frequencies is None:
            return {"error": "No spectrum data"}
            
        # Find frequency indices for channel
        freq_start = center_freq - bandwidth/2
        freq_stop = center_freq + bandwidth/2
        
        mask = (self._frequencies >= freq_start) & (self._frequencies <= freq_stop)
        
        if not np.any(mask):
            return {"error": "Channel outside spectrum"}
            
        channel_spectrum = self._current_spectrum[mask]
        
        # Integrate power (sum in linear)
        linear_power = 10**(channel_spectrum / 10)
        total_power = np.sum(linear_power)
        channel_power_dbm = 10 * np.log10(total_power + 1e-20)
        
        # Calculate PSD
        rbw = self.sample_rate / self.settings.fft_size
        psd_dbm_hz = channel_power_dbm - 10 * np.log10(bandwidth)
        
        return {
            "channel_power_dbm": float(channel_power_dbm),
            "psd_dbm_hz": float(psd_dbm_hz),
            "peak_power_dbm": float(np.max(channel_spectrum)),
            "bandwidth_hz": bandwidth
        }
        
    def measure_obw(self, percent: float = 99.0) -> Dict[str, float]:
        """
        Measure Occupied Bandwidth.
        
        Args:
            percent: Percentage of power to contain (default 99%)
            
        Returns:
            OBW measurement results
        """
        if self._current_spectrum is None or self._frequencies is None:
            return {"error": "No spectrum data"}
            
        # Convert to linear power
        linear_power = 10**(self._current_spectrum / 10)
        total_power = np.sum(linear_power)
        target_power = total_power * (percent / 100)
        
        # Find bandwidth containing target power
        cumsum = np.cumsum(linear_power)
        
        # Find indices
        threshold_low = total_power * (1 - percent/100) / 2
        threshold_high = total_power - threshold_low
        
        idx_low = np.searchsorted(cumsum, threshold_low)
        idx_high = np.searchsorted(cumsum, threshold_high)
        
        obw = abs(self._frequencies[idx_high] - self._frequencies[idx_low])
        center = (self._frequencies[idx_high] + self._frequencies[idx_low]) / 2
        
        return {
            "obw_hz": float(obw),
            "center_frequency_hz": float(center),
            "percent": percent,
            "lower_frequency_hz": float(self._frequencies[idx_low]),
            "upper_frequency_hz": float(self._frequencies[idx_high])
        }
        
    def measure_acpr(self,
                    main_channel_bw: float,
                    adjacent_channel_bw: float,
                    channel_spacing: float) -> Dict[str, float]:
        """
        Measure Adjacent Channel Power Ratio.
        
        Args:
            main_channel_bw: Main channel bandwidth
            adjacent_channel_bw: Adjacent channel bandwidth
            channel_spacing: Spacing between channel centers
            
        Returns:
            ACPR measurements
        """
        center_freq = self.settings.center_frequency
        
        # Measure main channel
        main = self.measure_channel_power(center_freq, main_channel_bw)
        
        # Measure lower adjacent channel
        lower = self.measure_channel_power(
            center_freq - channel_spacing, 
            adjacent_channel_bw
        )
        
        # Measure upper adjacent channel
        upper = self.measure_channel_power(
            center_freq + channel_spacing,
            adjacent_channel_bw
        )
        
        if "error" in main or "error" in lower or "error" in upper:
            return {"error": "Channel measurement failed"}
            
        return {
            "main_channel_power_dbm": main["channel_power_dbm"],
            "lower_adjacent_power_dbm": lower["channel_power_dbm"],
            "upper_adjacent_power_dbm": upper["channel_power_dbm"],
            "acpr_lower_db": lower["channel_power_dbm"] - main["channel_power_dbm"],
            "acpr_upper_db": upper["channel_power_dbm"] - main["channel_power_dbm"]
        }
        
    def get_spectrum_data(self) -> Dict[str, Any]:
        """Get current spectrum data for visualization"""
        with self._lock:
            return {
                "frequencies": self._frequencies.tolist() if self._frequencies is not None else [],
                "power_dbm": self._current_spectrum.tolist() if self._current_spectrum is not None else [],
                "peaks": [
                    {
                        "frequency_hz": p.frequency_hz,
                        "power_dbm": p.power_dbm,
                        "bandwidth_hz": p.bandwidth_hz,
                        "snr_db": p.snr_db
                    } for p in self._detected_peaks[:20]  # Top 20 peaks
                ],
                "settings": {
                    "center_frequency": self.settings.center_frequency,
                    "span": self.settings.span,
                    "rbw": self.settings.rbw,
                    "fft_size": self.settings.fft_size,
                    "window": self.settings.window.value,
                    "averaging_mode": self.settings.averaging_mode.value
                },
                "noise_floor_dbm": float(np.median(self._current_spectrum)) if self._current_spectrum is not None else -100
            }
            
    def set_center_frequency(self, freq_hz: float) -> None:
        """Set center frequency"""
        self.settings.center_frequency = freq_hz
        self._update_frequencies()
        
    def set_span(self, span_hz: float) -> None:
        """Set frequency span (adjusts sample rate needed)"""
        self.settings.span = span_hz
        
    def set_window(self, window: WindowFunction) -> None:
        """Set window function"""
        self.settings.window = window
        self._window = self._generate_window()
        
    def set_averaging(self, mode: AveragingMode, count: int = 10) -> None:
        """Set averaging mode and count"""
        self.settings.averaging_mode = mode
        self.settings.averaging_count = count
        self._avg_buffer = []
        self._max_hold = None
        self._min_hold = None
        
    def set_peak_threshold(self, threshold_dbm: float) -> None:
        """Set peak detection threshold"""
        self._peak_threshold_dbm = threshold_dbm
        
    def reset_averaging(self) -> None:
        """Reset averaging buffers"""
        self._avg_buffer = []
        self._max_hold = None
        self._min_hold = None
        
    def register_callback(self, callback: Callable) -> None:
        """Register callback for spectrum updates"""
        self._update_callbacks.append(callback)


class WaterfallDisplay:
    """
    Waterfall (spectrogram) display for time-frequency analysis.
    
    Features:
    - Rolling waterfall with configurable history
    - Color mapping with adjustable dynamic range
    - Time markers
    - Frequency markers
    - Export capability
    """
    
    def __init__(self,
                 fft_size: int = 1024,
                 history_lines: int = 200,
                 sample_rate: float = 1e6,
                 center_frequency: float = 0.0):
        """
        Initialize waterfall display.
        
        Args:
            fft_size: FFT size for each line
            history_lines: Number of historical lines to display
            sample_rate: Sample rate in Hz
            center_frequency: Center frequency in Hz
        """
        self.fft_size = fft_size
        self.history_lines = history_lines
        self.sample_rate = sample_rate
        self.center_frequency = center_frequency
        
        # Waterfall data (rows = time, cols = frequency)
        self._waterfall_data = np.zeros((history_lines, fft_size))
        self._timestamps = deque(maxlen=history_lines)
        
        # Display settings
        self.min_db = -100
        self.max_db = 0
        self.color_map = "viridis"
        
        # Window
        self._window = np.blackman(fft_size)
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Current line index
        self._current_line = 0
        
    def add_line(self, iq_samples: np.ndarray) -> np.ndarray:
        """
        Add one line to the waterfall.
        
        Args:
            iq_samples: Complex IQ samples for one line
            
        Returns:
            The computed spectrum line in dB
        """
        if len(iq_samples) < self.fft_size:
            iq_samples = np.pad(iq_samples, 
                               (0, self.fft_size - len(iq_samples)),
                               mode='constant')
        elif len(iq_samples) > self.fft_size:
            iq_samples = iq_samples[:self.fft_size]
            
        # Apply window and compute FFT
        windowed = iq_samples * self._window
        fft_result = np.fft.fftshift(np.fft.fft(windowed))
        
        # Compute power in dB
        power_db = 20 * np.log10(np.abs(fft_result) + 1e-20)
        
        with self._lock:
            # Roll waterfall data up
            self._waterfall_data = np.roll(self._waterfall_data, -1, axis=0)
            self._waterfall_data[-1] = power_db
            
            # Add timestamp
            self._timestamps.append(time.time())
            
            self._current_line += 1
            
        return power_db
        
    def process_samples(self, iq_samples: np.ndarray) -> None:
        """
        Process a batch of IQ samples, creating multiple waterfall lines.
        
        Args:
            iq_samples: Complex IQ samples
        """
        # Split into FFT-sized chunks
        num_lines = len(iq_samples) // self.fft_size
        
        for i in range(num_lines):
            chunk = iq_samples[i * self.fft_size:(i + 1) * self.fft_size]
            self.add_line(chunk)
            
    def get_waterfall_data(self) -> Dict[str, Any]:
        """Get waterfall data for visualization"""
        with self._lock:
            # Calculate frequency axis
            frequencies = np.fft.fftshift(
                np.fft.fftfreq(self.fft_size, 1/self.sample_rate)
            ) + self.center_frequency
            
            # Clip to display range
            display_data = np.clip(self._waterfall_data, self.min_db, self.max_db)
            
            # Normalize to 0-255 for color mapping
            normalized = ((display_data - self.min_db) / 
                         (self.max_db - self.min_db) * 255).astype(np.uint8)
            
            return {
                "data": normalized.tolist(),
                "frequencies": frequencies.tolist(),
                "timestamps": list(self._timestamps),
                "min_db": self.min_db,
                "max_db": self.max_db,
                "fft_size": self.fft_size,
                "history_lines": self.history_lines,
                "total_lines": self._current_line
            }
            
    def set_dynamic_range(self, min_db: float, max_db: float) -> None:
        """Set display dynamic range"""
        self.min_db = min_db
        self.max_db = max_db
        
    def clear(self) -> None:
        """Clear waterfall data"""
        with self._lock:
            self._waterfall_data = np.zeros((self.history_lines, self.fft_size))
            self._timestamps.clear()
            self._current_line = 0
            
    def export_image(self, filepath: str) -> bool:
        """Export waterfall as image (requires PIL)"""
        try:
            from PIL import Image
            
            with self._lock:
                # Normalize to 0-255
                normalized = ((self._waterfall_data - self.min_db) / 
                             (self.max_db - self.min_db) * 255)
                normalized = np.clip(normalized, 0, 255).astype(np.uint8)
                
                # Create image
                img = Image.fromarray(normalized, mode='L')
                img.save(filepath)
                return True
        except Exception:
            return False
