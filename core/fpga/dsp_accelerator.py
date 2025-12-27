#!/usr/bin/env python3
"""
RF Arsenal OS - DSP Accelerator Interface
Host-side interface for FPGA DSP acceleration

Provides high-level Python interface to hardware accelerated:
- FFT/IFFT operations
- FIR/IIR filtering
- Modulation/Demodulation
- Signal generation
"""

import asyncio
import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class DSPOperation(Enum):
    """DSP operation types"""
    FFT = "fft"
    IFFT = "ifft"
    FIR_FILTER = "fir_filter"
    IIR_FILTER = "iir_filter"
    CONVOLUTION = "convolution"
    CORRELATION = "correlation"
    QAM_MOD = "qam_mod"
    QAM_DEMOD = "qam_demod"
    PSK_MOD = "psk_mod"
    PSK_DEMOD = "psk_demod"
    OFDM_MOD = "ofdm_mod"
    OFDM_DEMOD = "ofdm_demod"
    RESAMPLE = "resample"
    MIXER = "mixer"
    AGC = "agc"


class WindowType(Enum):
    """Window function types"""
    NONE = "none"
    HANNING = "hanning"
    HAMMING = "hamming"
    BLACKMAN = "blackman"
    KAISER = "kaiser"
    FLATTOP = "flattop"


class FilterType(Enum):
    """Filter types"""
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"
    ARBITRARY = "arbitrary"


@dataclass
class FFTConfig:
    """FFT engine configuration"""
    size: int = 2048
    inverse: bool = False
    window: WindowType = WindowType.HANNING
    normalize: bool = True
    overlap: int = 0  # Overlap samples for streaming
    shift: bool = True  # FFT shift (center DC)
    
    def validate(self) -> bool:
        """Validate configuration"""
        valid_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        return self.size in valid_sizes and 0 <= self.overlap < self.size


@dataclass
class FilterConfig:
    """Filter configuration"""
    filter_type: FilterType = FilterType.LOWPASS
    order: int = 64
    cutoff_freq: float = 0.25  # Normalized frequency (0-1)
    cutoff_high: Optional[float] = None  # For bandpass/bandstop
    coefficients: List[float] = field(default_factory=list)
    gain: float = 1.0
    
    # IIR specific
    is_iir: bool = False
    feedback_coefficients: List[float] = field(default_factory=list)
    
    def validate(self) -> bool:
        """Validate configuration"""
        if self.order < 1 or self.order > 256:
            return False
        if not 0 < self.cutoff_freq <= 0.5:
            return False
        if self.filter_type in (FilterType.BANDPASS, FilterType.BANDSTOP):
            if self.cutoff_high is None or self.cutoff_high <= self.cutoff_freq:
                return False
        return True


@dataclass
class ModulatorConfig:
    """Modulator configuration"""
    modulation_type: str = "qpsk"  # bpsk, qpsk, 8psk, 16qam, 64qam, 256qam
    samples_per_symbol: int = 4
    pulse_shaping: str = "root_raised_cosine"
    rolloff_factor: float = 0.35
    symbol_rate: float = 1e6
    
    # OFDM specific
    nfft: int = 2048
    num_subcarriers: int = 1200  # Active subcarriers
    cp_length: int = 144  # Cyclic prefix length
    
    def validate(self) -> bool:
        """Validate configuration"""
        valid_mods = ['bpsk', 'qpsk', '8psk', '16qam', '64qam', '256qam']
        return (
            self.modulation_type in valid_mods and
            self.samples_per_symbol >= 1 and
            0 < self.rolloff_factor <= 1
        )


class DSPAccelerator:
    """
    High-level interface to FPGA DSP acceleration
    
    Provides NumPy-compatible API for hardware-accelerated
    signal processing operations.
    """
    
    # Data format constants
    SAMPLE_FORMAT = np.int16  # Q15 fixed-point
    MAX_SAMPLE_VALUE = 32767
    MIN_SAMPLE_VALUE = -32768
    
    def __init__(
        self,
        fpga_controller: Any,
        use_hardware: bool = True,
        fallback_to_software: bool = True
    ):
        """
        Initialize DSP Accelerator
        
        Args:
            fpga_controller: FPGAController instance
            use_hardware: Whether to use hardware acceleration
            fallback_to_software: Fall back to NumPy if hardware unavailable
        """
        self._fpga = fpga_controller
        self._use_hardware = use_hardware
        self._fallback_to_software = fallback_to_software
        
        # Current configurations
        self._fft_config = FFTConfig()
        self._filter_config = FilterConfig()
        self._mod_config = ModulatorConfig()
        
        # Pre-computed coefficients
        self._window_cache: Dict[Tuple[int, WindowType], np.ndarray] = {}
        self._filter_coef_cache: Dict[str, np.ndarray] = {}
        
        # Performance statistics
        self._stats = {
            'fft_count': 0,
            'filter_count': 0,
            'mod_count': 0,
            'hw_operations': 0,
            'sw_fallbacks': 0,
            'total_samples': 0,
        }
        
        logger.info("DSPAccelerator initialized")
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get performance statistics"""
        return self._stats.copy()
    
    @property
    def hardware_available(self) -> bool:
        """Check if hardware acceleration is available"""
        return self._fpga is not None and self._fpga.is_configured
    
    # =========================================================================
    # FFT Operations
    # =========================================================================
    
    def configure_fft(self, config: FFTConfig) -> bool:
        """
        Configure FFT engine
        
        Args:
            config: FFT configuration
            
        Returns:
            True if configured successfully
        """
        if not config.validate():
            logger.error("Invalid FFT configuration")
            return False
        
        self._fft_config = config
        
        # Configure hardware if available
        if self.hardware_available and self._use_hardware:
            return self._fpga.configure_fft(
                size=config.size,
                inverse=config.inverse,
                window=config.window.value
            )
        
        return True
    
    def fft(
        self,
        data: np.ndarray,
        config: Optional[FFTConfig] = None
    ) -> np.ndarray:
        """
        Compute FFT using hardware acceleration
        
        Args:
            data: Input complex samples
            config: Optional configuration override
            
        Returns:
            FFT output (complex array)
        """
        config = config or self._fft_config
        self._stats['fft_count'] += 1
        self._stats['total_samples'] += len(data)
        
        # Pad to FFT size if needed
        if len(data) < config.size:
            data = np.pad(data, (0, config.size - len(data)))
        elif len(data) > config.size:
            # Process in chunks
            return self._streaming_fft(data, config)
        
        # Apply window if specified
        if config.window != WindowType.NONE:
            window = self._get_window(config.size, config.window)
            data = data * window
        
        # Try hardware acceleration
        if self.hardware_available and self._use_hardware:
            try:
                result = self._hardware_fft(data, config)
                if result is not None:
                    self._stats['hw_operations'] += 1
                    return result
            except Exception as e:
                logger.warning(f"Hardware FFT failed: {e}")
        
        # Software fallback
        if self._fallback_to_software:
            self._stats['sw_fallbacks'] += 1
            return self._software_fft(data, config)
        
        raise RuntimeError("FFT computation failed and fallback disabled")
    
    def ifft(
        self,
        data: np.ndarray,
        config: Optional[FFTConfig] = None
    ) -> np.ndarray:
        """
        Compute IFFT using hardware acceleration
        
        Args:
            data: Input frequency domain data
            config: Optional configuration override
            
        Returns:
            IFFT output (complex array)
        """
        config = config or FFTConfig(
            size=self._fft_config.size,
            inverse=True,
            window=WindowType.NONE,
            normalize=True
        )
        config.inverse = True
        
        return self.fft(data, config)
    
    def _hardware_fft(
        self,
        data: np.ndarray,
        config: FFTConfig
    ) -> Optional[np.ndarray]:
        """Execute FFT on FPGA hardware"""
        # Convert to Q15 format
        input_bytes = self._to_q15(data)
        
        # Execute on hardware
        result_bytes = self._fpga.execute_fft(input_bytes)
        
        if result_bytes is None:
            return None
        
        # Convert back to complex
        result = self._from_q15(result_bytes, len(data))
        
        # Apply normalization
        if config.normalize:
            result = result / config.size
        
        # Apply shift if requested
        if config.shift:
            result = np.fft.fftshift(result)
        
        return result
    
    def _software_fft(
        self,
        data: np.ndarray,
        config: FFTConfig
    ) -> np.ndarray:
        """Software FFT fallback using NumPy"""
        if config.inverse:
            result = np.fft.ifft(data)
        else:
            result = np.fft.fft(data)
        
        if config.shift and not config.inverse:
            result = np.fft.fftshift(result)
        
        return result
    
    def _streaming_fft(
        self,
        data: np.ndarray,
        config: FFTConfig
    ) -> np.ndarray:
        """Process long data in streaming FFT mode"""
        results = []
        step = config.size - config.overlap
        
        for i in range(0, len(data) - config.size + 1, step):
            chunk = data[i:i + config.size]
            result = self.fft(chunk, config)
            results.append(result)
        
        return np.concatenate(results)
    
    # =========================================================================
    # Filter Operations
    # =========================================================================
    
    def configure_filter(self, config: FilterConfig) -> bool:
        """
        Configure filter engine
        
        Args:
            config: Filter configuration
            
        Returns:
            True if configured successfully
        """
        if not config.validate():
            logger.error("Invalid filter configuration")
            return False
        
        self._filter_config = config
        
        # Design filter if coefficients not provided
        if not config.coefficients:
            config.coefficients = self._design_filter(config)
        
        # Configure hardware if available
        if self.hardware_available and self._use_hardware:
            return self._fpga.configure_filter(
                coefficients=config.coefficients,
                filter_type='iir' if config.is_iir else 'fir'
            )
        
        return True
    
    def filter(
        self,
        data: np.ndarray,
        config: Optional[FilterConfig] = None
    ) -> np.ndarray:
        """
        Apply filter using hardware acceleration
        
        Args:
            data: Input samples
            config: Optional configuration override
            
        Returns:
            Filtered output
        """
        config = config or self._filter_config
        self._stats['filter_count'] += 1
        self._stats['total_samples'] += len(data)
        
        # Try hardware acceleration
        if self.hardware_available and self._use_hardware:
            try:
                result = self._hardware_filter(data, config)
                if result is not None:
                    self._stats['hw_operations'] += 1
                    return result
            except Exception as e:
                logger.warning(f"Hardware filter failed: {e}")
        
        # Software fallback
        if self._fallback_to_software:
            self._stats['sw_fallbacks'] += 1
            return self._software_filter(data, config)
        
        raise RuntimeError("Filter computation failed and fallback disabled")
    
    def lowpass_filter(
        self,
        data: np.ndarray,
        cutoff: float,
        order: int = 64
    ) -> np.ndarray:
        """
        Apply lowpass filter
        
        Args:
            data: Input samples
            cutoff: Normalized cutoff frequency (0-0.5)
            order: Filter order
            
        Returns:
            Filtered output
        """
        config = FilterConfig(
            filter_type=FilterType.LOWPASS,
            order=order,
            cutoff_freq=cutoff
        )
        config.coefficients = self._design_filter(config)
        return self.filter(data, config)
    
    def highpass_filter(
        self,
        data: np.ndarray,
        cutoff: float,
        order: int = 64
    ) -> np.ndarray:
        """Apply highpass filter"""
        config = FilterConfig(
            filter_type=FilterType.HIGHPASS,
            order=order,
            cutoff_freq=cutoff
        )
        config.coefficients = self._design_filter(config)
        return self.filter(data, config)
    
    def bandpass_filter(
        self,
        data: np.ndarray,
        low_cutoff: float,
        high_cutoff: float,
        order: int = 64
    ) -> np.ndarray:
        """Apply bandpass filter"""
        config = FilterConfig(
            filter_type=FilterType.BANDPASS,
            order=order,
            cutoff_freq=low_cutoff,
            cutoff_high=high_cutoff
        )
        config.coefficients = self._design_filter(config)
        return self.filter(data, config)
    
    def custom_filter(
        self,
        data: np.ndarray,
        coefficients: List[float]
    ) -> np.ndarray:
        """Apply custom FIR filter with given coefficients"""
        config = FilterConfig(
            filter_type=FilterType.ARBITRARY,
            order=len(coefficients),
            coefficients=list(coefficients)
        )
        return self.filter(data, config)
    
    def _hardware_filter(
        self,
        data: np.ndarray,
        config: FilterConfig
    ) -> Optional[np.ndarray]:
        """Execute filter on FPGA hardware"""
        # Convert to Q15 format
        input_bytes = self._to_q15(data)
        
        # Execute on hardware
        result_bytes = self._fpga.execute_filter(input_bytes)
        
        if result_bytes is None:
            return None
        
        # Convert back to complex
        result = self._from_q15(result_bytes, len(data))
        
        # Apply gain
        return result * config.gain
    
    def _software_filter(
        self,
        data: np.ndarray,
        config: FilterConfig
    ) -> np.ndarray:
        """Software filter fallback"""
        coefficients = np.array(config.coefficients)
        
        if config.is_iir:
            # IIR filter (use scipy if available)
            try:
                from scipy import signal
                b = coefficients
                a = np.array(config.feedback_coefficients) if config.feedback_coefficients else [1.0]
                return signal.lfilter(b, a, data) * config.gain
            except ImportError:
                logger.warning("scipy not available, using basic convolution")
        
        # FIR filter using convolution
        return np.convolve(data, coefficients, mode='same') * config.gain
    
    def _design_filter(self, config: FilterConfig) -> List[float]:
        """Design filter coefficients"""
        cache_key = f"{config.filter_type.value}_{config.order}_{config.cutoff_freq}_{config.cutoff_high}"
        
        if cache_key in self._filter_coef_cache:
            return list(self._filter_coef_cache[cache_key])
        
        try:
            from scipy import signal
            
            if config.filter_type == FilterType.LOWPASS:
                coeffs = signal.firwin(config.order, config.cutoff_freq)
            elif config.filter_type == FilterType.HIGHPASS:
                coeffs = signal.firwin(config.order, config.cutoff_freq, pass_zero=False)
            elif config.filter_type == FilterType.BANDPASS:
                coeffs = signal.firwin(config.order, [config.cutoff_freq, config.cutoff_high], pass_zero=False)
            elif config.filter_type == FilterType.BANDSTOP:
                coeffs = signal.firwin(config.order, [config.cutoff_freq, config.cutoff_high])
            else:
                coeffs = np.ones(config.order) / config.order
            
            self._filter_coef_cache[cache_key] = coeffs
            return list(coeffs)
            
        except ImportError:
            # Simple sinc filter design
            return self._design_sinc_filter(config)
    
    def _design_sinc_filter(self, config: FilterConfig) -> List[float]:
        """Simple sinc filter design (fallback)"""
        n = np.arange(config.order)
        m = config.order // 2
        
        # Sinc function
        fc = config.cutoff_freq
        h = np.sinc(2 * fc * (n - m))
        
        # Apply window
        window = np.hamming(config.order)
        h = h * window
        
        # Normalize
        h = h / np.sum(h)
        
        return list(h)
    
    # =========================================================================
    # Modulation Operations
    # =========================================================================
    
    def configure_modulator(self, config: ModulatorConfig) -> bool:
        """Configure modulator"""
        if not config.validate():
            logger.error("Invalid modulator configuration")
            return False
        
        self._mod_config = config
        return True
    
    def modulate_qam(
        self,
        bits: np.ndarray,
        config: Optional[ModulatorConfig] = None
    ) -> np.ndarray:
        """
        QAM modulation
        
        Args:
            bits: Input bit array
            config: Modulator configuration
            
        Returns:
            Complex modulated symbols
        """
        config = config or self._mod_config
        self._stats['mod_count'] += 1
        
        mod_type = config.modulation_type.lower()
        
        if mod_type == 'bpsk':
            return self._modulate_bpsk(bits)
        elif mod_type == 'qpsk':
            return self._modulate_qpsk(bits)
        elif mod_type == '16qam':
            return self._modulate_16qam(bits)
        elif mod_type == '64qam':
            return self._modulate_64qam(bits)
        elif mod_type == '256qam':
            return self._modulate_256qam(bits)
        else:
            raise ValueError(f"Unsupported modulation: {mod_type}")
    
    def demodulate_qam(
        self,
        symbols: np.ndarray,
        config: Optional[ModulatorConfig] = None
    ) -> np.ndarray:
        """
        QAM demodulation
        
        Args:
            symbols: Complex input symbols
            config: Modulator configuration
            
        Returns:
            Demodulated bits
        """
        config = config or self._mod_config
        
        mod_type = config.modulation_type.lower()
        
        if mod_type == 'bpsk':
            return self._demodulate_bpsk(symbols)
        elif mod_type == 'qpsk':
            return self._demodulate_qpsk(symbols)
        elif mod_type == '16qam':
            return self._demodulate_16qam(symbols)
        elif mod_type == '64qam':
            return self._demodulate_64qam(symbols)
        else:
            raise ValueError(f"Unsupported demodulation: {mod_type}")
    
    def generate_ofdm_symbol(
        self,
        data_symbols: np.ndarray,
        config: Optional[ModulatorConfig] = None
    ) -> np.ndarray:
        """
        Generate OFDM symbol
        
        Args:
            data_symbols: Frequency domain data symbols
            config: OFDM configuration
            
        Returns:
            Time domain OFDM symbol with cyclic prefix
        """
        config = config or self._mod_config
        
        # Create frequency domain frame
        freq_data = np.zeros(config.nfft, dtype=complex)
        
        # Map data to subcarriers (centered around DC)
        num_data = min(len(data_symbols), config.num_subcarriers)
        start_idx = (config.nfft - config.num_subcarriers) // 2
        freq_data[start_idx:start_idx + num_data] = data_symbols[:num_data]
        
        # Hardware or software IFFT
        if self.hardware_available and self._use_hardware:
            try:
                input_bytes = self._to_q15(freq_data)
                result_bytes = self._fpga.generate_ofdm_symbol(input_bytes)
                if result_bytes:
                    time_data = self._from_q15(result_bytes, config.nfft)
                else:
                    time_data = np.fft.ifft(freq_data)
            except Exception as e:
                logger.warning(f"Hardware OFDM failed: {e}")
                time_data = np.fft.ifft(freq_data)
        else:
            time_data = np.fft.ifft(freq_data)
        
        # Add cyclic prefix
        cp = time_data[-config.cp_length:]
        ofdm_symbol = np.concatenate([cp, time_data])
        
        return ofdm_symbol
    
    def _modulate_bpsk(self, bits: np.ndarray) -> np.ndarray:
        """BPSK modulation"""
        return 2.0 * bits.astype(float) - 1.0 + 0j
    
    def _modulate_qpsk(self, bits: np.ndarray) -> np.ndarray:
        """QPSK modulation"""
        # Reshape to pairs
        bits = bits[:len(bits) // 2 * 2].reshape(-1, 2)
        
        symbols = np.zeros(len(bits), dtype=complex)
        for i, (b0, b1) in enumerate(bits):
            real = 1 if b0 else -1
            imag = 1 if b1 else -1
            symbols[i] = (real + 1j * imag) / np.sqrt(2)
        
        return symbols
    
    def _modulate_16qam(self, bits: np.ndarray) -> np.ndarray:
        """16-QAM modulation"""
        # Reshape to groups of 4
        bits = bits[:len(bits) // 4 * 4].reshape(-1, 4)
        
        # Gray-coded constellation
        constellation = {
            (0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3
        }
        
        symbols = np.zeros(len(bits), dtype=complex)
        for i, b in enumerate(bits):
            real = constellation[tuple(b[:2])]
            imag = constellation[tuple(b[2:])]
            symbols[i] = (real + 1j * imag) / np.sqrt(10)
        
        return symbols
    
    def _modulate_64qam(self, bits: np.ndarray) -> np.ndarray:
        """64-QAM modulation"""
        bits = bits[:len(bits) // 6 * 6].reshape(-1, 6)
        
        def bits_to_level(b):
            levels = [-7, -5, -3, -1, 1, 3, 5, 7]
            idx = b[0] * 4 + b[1] * 2 + b[2]
            return levels[idx]
        
        symbols = np.zeros(len(bits), dtype=complex)
        for i, b in enumerate(bits):
            real = bits_to_level(b[:3])
            imag = bits_to_level(b[3:])
            symbols[i] = (real + 1j * imag) / np.sqrt(42)
        
        return symbols
    
    def _modulate_256qam(self, bits: np.ndarray) -> np.ndarray:
        """256-QAM modulation"""
        bits = bits[:len(bits) // 8 * 8].reshape(-1, 8)
        
        def bits_to_level(b):
            levels = [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]
            idx = b[0] * 8 + b[1] * 4 + b[2] * 2 + b[3]
            return levels[idx]
        
        symbols = np.zeros(len(bits), dtype=complex)
        for i, b in enumerate(bits):
            real = bits_to_level(b[:4])
            imag = bits_to_level(b[4:])
            symbols[i] = (real + 1j * imag) / np.sqrt(170)
        
        return symbols
    
    def _demodulate_bpsk(self, symbols: np.ndarray) -> np.ndarray:
        """BPSK demodulation"""
        return (np.real(symbols) > 0).astype(np.uint8)
    
    def _demodulate_qpsk(self, symbols: np.ndarray) -> np.ndarray:
        """QPSK demodulation"""
        bits = np.zeros(len(symbols) * 2, dtype=np.uint8)
        bits[0::2] = (np.real(symbols) > 0).astype(np.uint8)
        bits[1::2] = (np.imag(symbols) > 0).astype(np.uint8)
        return bits
    
    def _demodulate_16qam(self, symbols: np.ndarray) -> np.ndarray:
        """16-QAM demodulation (hard decision)"""
        bits = np.zeros(len(symbols) * 4, dtype=np.uint8)
        
        # Decision regions
        real = np.real(symbols) * np.sqrt(10)
        imag = np.imag(symbols) * np.sqrt(10)
        
        for i, (r, im) in enumerate(zip(real, imag)):
            # Real component bits
            bits[i * 4 + 0] = 1 if r > 0 else 0
            bits[i * 4 + 1] = 1 if abs(r) < 2 else 0
            # Imag component bits
            bits[i * 4 + 2] = 1 if im > 0 else 0
            bits[i * 4 + 3] = 1 if abs(im) < 2 else 0
        
        return bits
    
    def _demodulate_64qam(self, symbols: np.ndarray) -> np.ndarray:
        """64-QAM demodulation"""
        bits = np.zeros(len(symbols) * 6, dtype=np.uint8)
        
        real = np.real(symbols) * np.sqrt(42)
        imag = np.imag(symbols) * np.sqrt(42)
        
        def level_to_bits(level):
            if level < -6:
                return [0, 0, 0]
            elif level < -4:
                return [0, 0, 1]
            elif level < -2:
                return [0, 1, 1]
            elif level < 0:
                return [0, 1, 0]
            elif level < 2:
                return [1, 1, 0]
            elif level < 4:
                return [1, 1, 1]
            elif level < 6:
                return [1, 0, 1]
            else:
                return [1, 0, 0]
        
        for i, (r, im) in enumerate(zip(real, imag)):
            bits[i * 6:i * 6 + 3] = level_to_bits(r)
            bits[i * 6 + 3:i * 6 + 6] = level_to_bits(im)
        
        return bits
    
    # =========================================================================
    # Signal Generation
    # =========================================================================
    
    def generate_tone(
        self,
        frequency: float,
        duration: float,
        sample_rate: float,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Generate complex tone (CW signal)
        
        Args:
            frequency: Frequency in Hz
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            amplitude: Signal amplitude (0-1)
            
        Returns:
            Complex tone signal
        """
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate
        return amplitude * np.exp(2j * np.pi * frequency * t)
    
    def generate_chirp(
        self,
        start_freq: float,
        end_freq: float,
        duration: float,
        sample_rate: float,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Generate linear chirp signal
        
        Args:
            start_freq: Start frequency in Hz
            end_freq: End frequency in Hz
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            amplitude: Signal amplitude
            
        Returns:
            Complex chirp signal
        """
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate
        
        # Linear frequency sweep
        k = (end_freq - start_freq) / duration
        phase = 2 * np.pi * (start_freq * t + 0.5 * k * t**2)
        
        return amplitude * np.exp(1j * phase)
    
    def generate_noise(
        self,
        num_samples: int,
        power_db: float = 0.0,
        noise_type: str = "gaussian"
    ) -> np.ndarray:
        """
        Generate noise signal
        
        Args:
            num_samples: Number of samples
            power_db: Noise power in dB
            noise_type: "gaussian" or "uniform"
            
        Returns:
            Complex noise signal
        """
        power = 10 ** (power_db / 10)
        sigma = np.sqrt(power / 2)
        
        if noise_type == "gaussian":
            real = np.random.randn(num_samples) * sigma
            imag = np.random.randn(num_samples) * sigma
        else:  # uniform
            real = (np.random.rand(num_samples) - 0.5) * 2 * sigma * np.sqrt(3)
            imag = (np.random.rand(num_samples) - 0.5) * 2 * sigma * np.sqrt(3)
        
        return real + 1j * imag
    
    def generate_preamble(
        self,
        preamble_type: str = "zadoff_chu",
        length: int = 64,
        root: int = 25
    ) -> np.ndarray:
        """
        Generate synchronization preamble
        
        Args:
            preamble_type: "zadoff_chu", "gold", "barker"
            length: Sequence length
            root: Root index for Zadoff-Chu
            
        Returns:
            Complex preamble sequence
        """
        if preamble_type == "zadoff_chu":
            n = np.arange(length)
            seq = np.exp(-1j * np.pi * root * n * (n + 1) / length)
            return seq
        
        elif preamble_type == "barker":
            # Barker-13 sequence
            barker = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
            # Pad or truncate
            if length < len(barker):
                return barker[:length].astype(complex)
            else:
                return np.tile(barker, length // len(barker) + 1)[:length].astype(complex)
        
        else:
            # Default: random QPSK
            bits = np.random.randint(0, 2, length * 2)
            return self._modulate_qpsk(bits)
    
    # =========================================================================
    # Utility Functions
    # =========================================================================
    
    def _get_window(self, size: int, window_type: WindowType) -> np.ndarray:
        """Get or compute window function"""
        cache_key = (size, window_type)
        
        if cache_key in self._window_cache:
            return self._window_cache[cache_key]
        
        if window_type == WindowType.HANNING:
            window = np.hanning(size)
        elif window_type == WindowType.HAMMING:
            window = np.hamming(size)
        elif window_type == WindowType.BLACKMAN:
            window = np.blackman(size)
        elif window_type == WindowType.KAISER:
            window = np.kaiser(size, 14)
        elif window_type == WindowType.FLATTOP:
            # Flat-top window
            n = np.arange(size)
            a0, a1, a2, a3, a4 = 0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368
            window = a0 - a1 * np.cos(2*np.pi*n/size) + a2 * np.cos(4*np.pi*n/size) - \
                     a3 * np.cos(6*np.pi*n/size) + a4 * np.cos(8*np.pi*n/size)
        else:
            window = np.ones(size)
        
        self._window_cache[cache_key] = window
        return window
    
    def _to_q15(self, data: np.ndarray) -> bytes:
        """Convert complex array to Q15 interleaved bytes"""
        # Scale to Q15 range
        real = np.clip(np.real(data) * self.MAX_SAMPLE_VALUE, 
                      self.MIN_SAMPLE_VALUE, self.MAX_SAMPLE_VALUE).astype(np.int16)
        imag = np.clip(np.imag(data) * self.MAX_SAMPLE_VALUE,
                      self.MIN_SAMPLE_VALUE, self.MAX_SAMPLE_VALUE).astype(np.int16)
        
        # Interleave I/Q
        interleaved = np.empty(len(data) * 2, dtype=np.int16)
        interleaved[0::2] = real
        interleaved[1::2] = imag
        
        return interleaved.tobytes()
    
    def _from_q15(self, data: bytes, num_samples: int) -> np.ndarray:
        """Convert Q15 interleaved bytes to complex array"""
        interleaved = np.frombuffer(data, dtype=np.int16)
        
        real = interleaved[0::2].astype(float) / self.MAX_SAMPLE_VALUE
        imag = interleaved[1::2].astype(float) / self.MAX_SAMPLE_VALUE
        
        return real + 1j * imag
    
    def reset_stats(self) -> None:
        """Reset performance statistics"""
        for key in self._stats:
            self._stats[key] = 0
