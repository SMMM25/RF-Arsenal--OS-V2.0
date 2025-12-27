#!/usr/bin/env python3
"""
RF Arsenal OS - SoapySDR Hardware Abstraction Layer

Universal SDR backend supporting:
- BladeRF (1.0, 2.0 micro, 2.0 xA4/xA9)
- HackRF One
- RTL-SDR (RTL2832U)
- USRP (B200, B210, N200, X300)
- LimeSDR (Mini, USB)
- Airspy (R2, Mini, HF+)
- PlutoSDR (ADALM-PLUTO)

This module provides a unified interface for all SDR hardware,
enabling seamless switching between devices without code changes.

Author: RF Arsenal Team
License: For authorized security research only
"""

import logging
import threading
import queue
import time
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple, Union
from abc import ABC, abstractmethod

# Try to import SoapySDR
try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_TX, SOAPY_SDR_CF32, SOAPY_SDR_CS16
    SOAPY_AVAILABLE = True
except ImportError:
    SOAPY_AVAILABLE = False
    logging.warning("SoapySDR not installed. Install with: sudo apt install python3-soapysdr")


class SDRType(Enum):
    """Supported SDR hardware types"""
    BLADERF = "bladerf"
    BLADERF2 = "bladerf2"
    HACKRF = "hackrf"
    RTLSDR = "rtlsdr"
    USRP = "uhd"
    LIMESDR = "lime"
    AIRSPY = "airspy"
    AIRSPYHF = "airspyhf"
    PLUTOSDR = "plutosdr"
    SDRPLAY = "sdrplay"
    REMOTE = "remote"  # SoapyRemote
    UNKNOWN = "unknown"


class StreamFormat(Enum):
    """IQ sample formats"""
    CF32 = "CF32"   # Complex float32 (recommended)
    CS16 = "CS16"   # Complex int16 (native for most SDRs)
    CS8 = "CS8"     # Complex int8 (RTL-SDR native)
    CU8 = "CU8"     # Complex uint8


class StreamDirection(Enum):
    """Stream direction"""
    RX = auto()
    TX = auto()


class GainMode(Enum):
    """Gain control mode"""
    MANUAL = auto()
    AGC_SLOW = auto()
    AGC_FAST = auto()
    AGC_HYBRID = auto()


@dataclass
class SDRDeviceInfo:
    """Information about a detected SDR device"""
    driver: str
    label: str
    serial: str
    hardware: str
    sdr_type: SDRType
    index: int
    capabilities: Dict[str, Any] = field(default_factory=dict)
    
    # Hardware specifications
    freq_range: Tuple[float, float] = (0, 6e9)
    sample_rate_range: Tuple[float, float] = (0, 61.44e6)
    bandwidth_range: Tuple[float, float] = (0, 56e6)
    tx_capable: bool = False
    full_duplex: bool = False
    mimo_capable: bool = False
    num_channels: int = 1
    
    def __str__(self) -> str:
        return f"{self.label} [{self.driver}] (S/N: {self.serial})"


@dataclass
class StreamConfig:
    """Configuration for RX/TX streaming"""
    frequency: float = 100e6           # Center frequency in Hz
    sample_rate: float = 2e6           # Sample rate in Hz
    bandwidth: float = 0               # 0 = auto (match sample rate)
    gain: float = 30                   # Gain in dB
    gain_mode: GainMode = GainMode.MANUAL
    antenna: str = ""                  # Antenna port (empty = default)
    channel: int = 0                   # Channel index
    dc_offset_mode: bool = True        # Enable DC offset correction
    iq_balance_mode: bool = True       # Enable IQ balance correction
    buffer_size: int = 16384           # Samples per buffer
    num_buffers: int = 8               # Number of buffers
    format: StreamFormat = StreamFormat.CF32


@dataclass
class IQCapture:
    """Captured IQ data"""
    samples: np.ndarray
    frequency: float
    sample_rate: float
    bandwidth: float
    timestamp: float
    duration: float
    device: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def save(self, filepath: str, format: str = "npy"):
        """Save capture to file"""
        if format == "npy":
            np.save(filepath, self.samples)
        elif format == "cs16":
            # Convert to complex int16
            samples_i16 = np.zeros(len(self.samples) * 2, dtype=np.int16)
            samples_i16[0::2] = (self.samples.real * 32767).astype(np.int16)
            samples_i16[1::2] = (self.samples.imag * 32767).astype(np.int16)
            samples_i16.tofile(filepath)
        elif format == "cf32":
            self.samples.astype(np.complex64).tofile(filepath)
    
    @classmethod
    def load(cls, filepath: str, frequency: float = 0, sample_rate: float = 0):
        """Load capture from file"""
        if filepath.endswith('.npy'):
            samples = np.load(filepath)
        elif filepath.endswith('.cs16'):
            raw = np.fromfile(filepath, dtype=np.int16)
            samples = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64) / 32767
        elif filepath.endswith('.cf32'):
            samples = np.fromfile(filepath, dtype=np.complex64)
        else:
            samples = np.fromfile(filepath, dtype=np.complex64)
        
        return cls(
            samples=samples,
            frequency=frequency,
            sample_rate=sample_rate,
            bandwidth=sample_rate,
            timestamp=time.time(),
            duration=len(samples) / sample_rate if sample_rate > 0 else 0,
            device="file"
        )


class DSPProcessor:
    """Real-time DSP processing pipeline"""
    
    def __init__(self):
        self.logger = logging.getLogger('DSP-Processor')
        self._fft_size = 2048
        self._window = np.hanning(self._fft_size)
        self._averaging = 4
    
    def compute_fft(self, samples: np.ndarray, fft_size: int = None) -> np.ndarray:
        """Compute FFT magnitude spectrum in dB"""
        if fft_size is None:
            fft_size = self._fft_size
        
        if len(samples) < fft_size:
            samples = np.pad(samples, (0, fft_size - len(samples)))
        
        # Apply window and compute FFT
        windowed = samples[:fft_size] * np.hanning(fft_size)
        fft_result = np.fft.fftshift(np.fft.fft(windowed))
        
        # Convert to dB
        magnitude = np.abs(fft_result)
        magnitude[magnitude == 0] = 1e-10  # Avoid log(0)
        power_db = 20 * np.log10(magnitude)
        
        return power_db
    
    def compute_spectrogram(self, samples: np.ndarray, fft_size: int = 1024, 
                           overlap: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spectrogram (time-frequency representation)"""
        hop = fft_size - overlap
        num_frames = (len(samples) - fft_size) // hop + 1
        
        spectrogram = np.zeros((num_frames, fft_size))
        window = np.hanning(fft_size)
        
        for i in range(num_frames):
            start = i * hop
            frame = samples[start:start + fft_size] * window
            fft_result = np.fft.fftshift(np.fft.fft(frame))
            spectrogram[i] = 20 * np.log10(np.abs(fft_result) + 1e-10)
        
        return spectrogram
    
    def estimate_frequency(self, samples: np.ndarray, sample_rate: float) -> float:
        """Estimate dominant frequency using FFT peak detection"""
        fft_result = np.fft.fft(samples)
        freqs = np.fft.fftfreq(len(samples), 1/sample_rate)
        
        # Find peak in positive frequencies
        positive_mask = freqs >= 0
        peak_idx = np.argmax(np.abs(fft_result[positive_mask]))
        
        return freqs[positive_mask][peak_idx]
    
    def measure_power(self, samples: np.ndarray) -> float:
        """Measure average power in dB"""
        power = np.mean(np.abs(samples) ** 2)
        return 10 * np.log10(power + 1e-10)
    
    def detect_signal(self, samples: np.ndarray, threshold_db: float = -50) -> bool:
        """Simple signal detection based on power threshold"""
        return self.measure_power(samples) > threshold_db
    
    def demodulate_fm(self, samples: np.ndarray, sample_rate: float, 
                      audio_rate: float = 48000) -> np.ndarray:
        """FM demodulation"""
        # Compute instantaneous phase
        phase = np.angle(samples)
        
        # Differentiate phase (frequency is derivative of phase)
        freq = np.diff(np.unwrap(phase))
        
        # Normalize
        freq = freq / (2 * np.pi) * sample_rate
        
        # Decimate to audio rate
        decimation = int(sample_rate / audio_rate)
        if decimation > 1:
            freq = freq[::decimation]
        
        return freq.astype(np.float32)
    
    def demodulate_am(self, samples: np.ndarray) -> np.ndarray:
        """AM demodulation (envelope detection)"""
        return np.abs(samples).astype(np.float32)
    
    def lowpass_filter(self, samples: np.ndarray, cutoff: float, 
                       sample_rate: float, order: int = 64) -> np.ndarray:
        """Simple FIR lowpass filter"""
        nyq = sample_rate / 2
        normalized_cutoff = cutoff / nyq
        
        # Design FIR filter using sinc function
        n = np.arange(order)
        h = np.sinc(2 * normalized_cutoff * (n - order/2)) * np.hanning(order)
        h = h / np.sum(h)  # Normalize
        
        # Apply filter
        return np.convolve(samples, h, mode='same')
    
    def bandpass_filter(self, samples: np.ndarray, low_cutoff: float,
                        high_cutoff: float, sample_rate: float) -> np.ndarray:
        """Bandpass filter using frequency domain"""
        fft_result = np.fft.fft(samples)
        freqs = np.fft.fftfreq(len(samples), 1/sample_rate)
        
        # Create bandpass mask
        mask = (np.abs(freqs) >= low_cutoff) & (np.abs(freqs) <= high_cutoff)
        fft_result[~mask] = 0
        
        return np.fft.ifft(fft_result)


class SoapySDRDevice:
    """
    Universal SDR device wrapper using SoapySDR
    
    Provides unified interface for all supported SDR hardware.
    """
    
    def __init__(self, device_info: SDRDeviceInfo = None, device_args: str = ""):
        self.logger = logging.getLogger('SoapySDR-Device')
        self.device_info = device_info
        self._device = None
        self._rx_stream = None
        self._tx_stream = None
        self._rx_config = StreamConfig()
        self._tx_config = StreamConfig()
        self._streaming = False
        self._rx_thread = None
        self._rx_queue = queue.Queue(maxsize=100)
        self._rx_callback = None
        self._stop_event = threading.Event()
        self.dsp = DSPProcessor()
        
        # Device arguments for opening
        self._device_args = device_args
        
        if not SOAPY_AVAILABLE:
            self.logger.warning("SoapySDR not available - running in simulation mode")
    
    @staticmethod
    def enumerate_devices() -> List[SDRDeviceInfo]:
        """Discover all available SDR devices"""
        devices = []
        
        if not SOAPY_AVAILABLE:
            logging.warning("SoapySDR not available for device enumeration")
            return devices
        
        try:
            results = SoapySDR.Device.enumerate()
            
            for idx, result in enumerate(results):
                driver = result.get('driver', 'unknown')
                
                # Determine SDR type from driver
                sdr_type = SDRType.UNKNOWN
                for st in SDRType:
                    if st.value in driver.lower():
                        sdr_type = st
                        break
                
                # Build device info
                info = SDRDeviceInfo(
                    driver=driver,
                    label=result.get('label', f'{driver}:{idx}'),
                    serial=result.get('serial', 'N/A'),
                    hardware=result.get('hardware', driver),
                    sdr_type=sdr_type,
                    index=idx,
                    capabilities=dict(result)
                )
                
                # Get detailed capabilities by opening device temporarily
                try:
                    dev = SoapySDR.Device(result)
                    
                    # Frequency range
                    freq_ranges = dev.getFrequencyRange(SOAPY_SDR_RX, 0)
                    if freq_ranges:
                        info.freq_range = (freq_ranges[0].minimum(), freq_ranges[-1].maximum())
                    
                    # Sample rate range
                    rate_ranges = dev.getSampleRateRange(SOAPY_SDR_RX, 0)
                    if rate_ranges:
                        info.sample_rate_range = (rate_ranges[0].minimum(), rate_ranges[-1].maximum())
                    
                    # TX capability
                    info.tx_capable = dev.getNumChannels(SOAPY_SDR_TX) > 0
                    
                    # Number of channels
                    info.num_channels = dev.getNumChannels(SOAPY_SDR_RX)
                    
                    # MIMO capable if multiple channels
                    info.mimo_capable = info.num_channels > 1
                    
                    # Full duplex if both RX and TX available
                    info.full_duplex = info.tx_capable and info.num_channels > 0
                    
                    dev = None  # Release device
                    
                except Exception as e:
                    logging.debug(f"Could not get detailed info for {driver}: {e}")
                
                devices.append(info)
                
        except Exception as e:
            logging.error(f"Error enumerating devices: {e}")
        
        return devices
    
    def open(self, device_args: str = None) -> bool:
        """Open the SDR device"""
        if not SOAPY_AVAILABLE:
            self.logger.info("Simulation mode: device open (no hardware)")
            return True
        
        try:
            args = device_args or self._device_args
            if self.device_info:
                args = f"driver={self.device_info.driver}"
                if self.device_info.serial != 'N/A':
                    args += f",serial={self.device_info.serial}"
            
            self._device = SoapySDR.Device(args)
            self.logger.info(f"Opened device: {args}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to open device: {e}")
            return False
    
    def close(self):
        """Close the SDR device"""
        self.stop_streaming()
        
        if self._rx_stream:
            try:
                self._device.closeStream(self._rx_stream)
            except:
                pass
            self._rx_stream = None
        
        if self._tx_stream:
            try:
                self._device.closeStream(self._tx_stream)
            except:
                pass
            self._tx_stream = None
        
        self._device = None
        self.logger.info("Device closed")
    
    def configure_rx(self, config: StreamConfig) -> bool:
        """Configure RX parameters"""
        self._rx_config = config
        
        if not SOAPY_AVAILABLE or not self._device:
            self.logger.info(f"Simulation: RX configured - {config.frequency/1e6:.3f} MHz, {config.sample_rate/1e6:.2f} MSPS")
            return True
        
        try:
            ch = config.channel
            
            # Set frequency
            self._device.setFrequency(SOAPY_SDR_RX, ch, config.frequency)
            
            # Set sample rate
            self._device.setSampleRate(SOAPY_SDR_RX, ch, config.sample_rate)
            
            # Set bandwidth (0 = auto)
            if config.bandwidth > 0:
                self._device.setBandwidth(SOAPY_SDR_RX, ch, config.bandwidth)
            
            # Set gain
            if config.gain_mode == GainMode.MANUAL:
                self._device.setGainMode(SOAPY_SDR_RX, ch, False)
                self._device.setGain(SOAPY_SDR_RX, ch, config.gain)
            else:
                self._device.setGainMode(SOAPY_SDR_RX, ch, True)
            
            # Set antenna if specified
            if config.antenna:
                self._device.setAntenna(SOAPY_SDR_RX, ch, config.antenna)
            
            # DC offset correction
            if self._device.hasDCOffsetMode(SOAPY_SDR_RX, ch):
                self._device.setDCOffsetMode(SOAPY_SDR_RX, ch, config.dc_offset_mode)
            
            # IQ balance correction
            if self._device.hasIQBalanceMode(SOAPY_SDR_RX, ch):
                self._device.setIQBalanceMode(SOAPY_SDR_RX, ch, config.iq_balance_mode)
            
            self.logger.info(f"RX configured: {config.frequency/1e6:.3f} MHz, "
                           f"{config.sample_rate/1e6:.2f} MSPS, {config.gain:.1f} dB")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to configure RX: {e}")
            return False
    
    def configure_tx(self, config: StreamConfig) -> bool:
        """Configure TX parameters"""
        self._tx_config = config
        
        if not SOAPY_AVAILABLE or not self._device:
            self.logger.info(f"Simulation: TX configured - {config.frequency/1e6:.3f} MHz, {config.sample_rate/1e6:.2f} MSPS")
            return True
        
        try:
            ch = config.channel
            
            self._device.setFrequency(SOAPY_SDR_TX, ch, config.frequency)
            self._device.setSampleRate(SOAPY_SDR_TX, ch, config.sample_rate)
            
            if config.bandwidth > 0:
                self._device.setBandwidth(SOAPY_SDR_TX, ch, config.bandwidth)
            
            self._device.setGain(SOAPY_SDR_TX, ch, config.gain)
            
            if config.antenna:
                self._device.setAntenna(SOAPY_SDR_TX, ch, config.antenna)
            
            self.logger.info(f"TX configured: {config.frequency/1e6:.3f} MHz, "
                           f"{config.sample_rate/1e6:.2f} MSPS, {config.gain:.1f} dB")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to configure TX: {e}")
            return False
    
    def setup_rx_stream(self) -> bool:
        """Setup RX stream"""
        if not SOAPY_AVAILABLE or not self._device:
            return True
        
        try:
            # Determine format
            fmt = SOAPY_SDR_CF32 if self._rx_config.format == StreamFormat.CF32 else SOAPY_SDR_CS16
            
            self._rx_stream = self._device.setupStream(
                SOAPY_SDR_RX, fmt, [self._rx_config.channel]
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup RX stream: {e}")
            return False
    
    def setup_tx_stream(self) -> bool:
        """Setup TX stream"""
        if not SOAPY_AVAILABLE or not self._device:
            return True
        
        try:
            fmt = SOAPY_SDR_CF32 if self._tx_config.format == StreamFormat.CF32 else SOAPY_SDR_CS16
            
            self._tx_stream = self._device.setupStream(
                SOAPY_SDR_TX, fmt, [self._tx_config.channel]
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup TX stream: {e}")
            return False
    
    def start_streaming(self, callback: Callable[[np.ndarray], None] = None):
        """Start RX streaming in background thread"""
        if self._streaming:
            return
        
        self._rx_callback = callback
        self._stop_event.clear()
        self._streaming = True
        
        if SOAPY_AVAILABLE and self._device and self._rx_stream:
            self._device.activateStream(self._rx_stream)
        
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()
        self.logger.info("Streaming started")
    
    def stop_streaming(self):
        """Stop RX streaming"""
        if not self._streaming:
            return
        
        self._stop_event.set()
        self._streaming = False
        
        if self._rx_thread:
            self._rx_thread.join(timeout=2.0)
            self._rx_thread = None
        
        if SOAPY_AVAILABLE and self._device and self._rx_stream:
            try:
                self._device.deactivateStream(self._rx_stream)
            except:
                pass
        
        self.logger.info("Streaming stopped")
    
    def _rx_loop(self):
        """RX streaming loop"""
        buffer_size = self._rx_config.buffer_size
        
        if self._rx_config.format == StreamFormat.CF32:
            buffer = np.zeros(buffer_size, dtype=np.complex64)
        else:
            buffer = np.zeros(buffer_size * 2, dtype=np.int16)
        
        while not self._stop_event.is_set():
            try:
                if SOAPY_AVAILABLE and self._device and self._rx_stream:
                    # Real hardware read
                    sr = self._device.readStream(self._rx_stream, [buffer], buffer_size)
                    
                    if sr.ret > 0:
                        samples = buffer[:sr.ret].copy()
                        
                        # Convert CS16 to CF32 if needed
                        if self._rx_config.format == StreamFormat.CS16:
                            samples = (samples[0::2] + 1j * samples[1::2]).astype(np.complex64) / 32767
                        
                        if self._rx_callback:
                            self._rx_callback(samples)
                        
                        if not self._rx_queue.full():
                            self._rx_queue.put(samples)
                else:
                    # Simulation mode - generate noise
                    time.sleep(buffer_size / self._rx_config.sample_rate)
                    noise = (np.random.randn(buffer_size) + 1j * np.random.randn(buffer_size)) * 0.01
                    samples = noise.astype(np.complex64)
                    
                    if self._rx_callback:
                        self._rx_callback(samples)
                    
                    if not self._rx_queue.full():
                        self._rx_queue.put(samples)
                        
            except Exception as e:
                if not self._stop_event.is_set():
                    self.logger.error(f"RX error: {e}")
                    time.sleep(0.1)
    
    def read_samples(self, num_samples: int, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Read samples (blocking)"""
        samples = []
        collected = 0
        start_time = time.time()
        
        while collected < num_samples:
            if time.time() - start_time > timeout:
                break
            
            try:
                chunk = self._rx_queue.get(timeout=0.1)
                samples.append(chunk)
                collected += len(chunk)
            except queue.Empty:
                continue
        
        if samples:
            return np.concatenate(samples)[:num_samples]
        return None
    
    def capture(self, duration: float, filename: str = None) -> IQCapture:
        """Capture IQ samples for specified duration"""
        num_samples = int(duration * self._rx_config.sample_rate)
        
        self.logger.info(f"Capturing {duration:.2f}s ({num_samples} samples) at "
                        f"{self._rx_config.frequency/1e6:.3f} MHz")
        
        # Setup and start streaming if not already
        was_streaming = self._streaming
        if not was_streaming:
            self.setup_rx_stream()
            self.start_streaming()
        
        # Collect samples
        samples = self.read_samples(num_samples, timeout=duration + 1.0)
        
        # Stop if we started it
        if not was_streaming:
            self.stop_streaming()
        
        if samples is None:
            samples = np.zeros(num_samples, dtype=np.complex64)
        
        capture = IQCapture(
            samples=samples,
            frequency=self._rx_config.frequency,
            sample_rate=self._rx_config.sample_rate,
            bandwidth=self._rx_config.bandwidth or self._rx_config.sample_rate,
            timestamp=time.time(),
            duration=duration,
            device=self.device_info.label if self.device_info else "simulation",
            metadata={
                'gain': self._rx_config.gain,
                'antenna': self._rx_config.antenna,
            }
        )
        
        if filename:
            capture.save(filename)
            self.logger.info(f"Capture saved to {filename}")
        
        return capture
    
    def transmit(self, samples: np.ndarray, repeat: int = 1) -> bool:
        """Transmit IQ samples"""
        if not SOAPY_AVAILABLE or not self._device:
            self.logger.info(f"Simulation: TX {len(samples)} samples, repeat={repeat}")
            return True
        
        if not self._tx_stream:
            if not self.setup_tx_stream():
                return False
        
        try:
            self._device.activateStream(self._tx_stream)
            
            # Convert to appropriate format
            if self._tx_config.format == StreamFormat.CS16:
                tx_buffer = np.zeros(len(samples) * 2, dtype=np.int16)
                tx_buffer[0::2] = (samples.real * 32767).astype(np.int16)
                tx_buffer[1::2] = (samples.imag * 32767).astype(np.int16)
            else:
                tx_buffer = samples.astype(np.complex64)
            
            for _ in range(repeat):
                offset = 0
                while offset < len(samples):
                    chunk_size = min(self._tx_config.buffer_size, len(samples) - offset)
                    chunk = tx_buffer[offset:offset + chunk_size]
                    
                    sr = self._device.writeStream(self._tx_stream, [chunk], chunk_size)
                    if sr.ret < 0:
                        raise Exception(f"TX error: {sr.ret}")
                    
                    offset += sr.ret
            
            self._device.deactivateStream(self._tx_stream)
            self.logger.info(f"Transmitted {len(samples)} samples, repeat={repeat}")
            return True
            
        except Exception as e:
            self.logger.error(f"TX failed: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get device information"""
        info = {
            'driver': self.device_info.driver if self.device_info else 'simulation',
            'hardware': self.device_info.hardware if self.device_info else 'simulation',
            'serial': self.device_info.serial if self.device_info else 'N/A',
            'soapy_available': SOAPY_AVAILABLE,
            'device_open': self._device is not None,
            'streaming': self._streaming,
        }
        
        if SOAPY_AVAILABLE and self._device:
            try:
                info['hardware_info'] = self._device.getHardwareInfo()
                info['antennas_rx'] = list(self._device.listAntennas(SOAPY_SDR_RX, 0))
                info['antennas_tx'] = list(self._device.listAntennas(SOAPY_SDR_TX, 0)) if self.device_info and self.device_info.tx_capable else []
                info['gains_rx'] = list(self._device.listGains(SOAPY_SDR_RX, 0))
                info['sample_rates'] = [r for r in self._device.listSampleRates(SOAPY_SDR_RX, 0)]
            except:
                pass
        
        return info
    
    # Convenience methods
    def set_frequency(self, freq: float, direction: StreamDirection = StreamDirection.RX):
        """Set center frequency"""
        if direction == StreamDirection.RX:
            self._rx_config.frequency = freq
            if self._device:
                self._device.setFrequency(SOAPY_SDR_RX, self._rx_config.channel, freq)
        else:
            self._tx_config.frequency = freq
            if self._device:
                self._device.setFrequency(SOAPY_SDR_TX, self._tx_config.channel, freq)
    
    def set_sample_rate(self, rate: float, direction: StreamDirection = StreamDirection.RX):
        """Set sample rate"""
        if direction == StreamDirection.RX:
            self._rx_config.sample_rate = rate
            if self._device:
                self._device.setSampleRate(SOAPY_SDR_RX, self._rx_config.channel, rate)
        else:
            self._tx_config.sample_rate = rate
            if self._device:
                self._device.setSampleRate(SOAPY_SDR_TX, self._tx_config.channel, rate)
    
    def set_gain(self, gain: float, direction: StreamDirection = StreamDirection.RX):
        """Set gain in dB"""
        if direction == StreamDirection.RX:
            self._rx_config.gain = gain
            if self._device:
                self._device.setGain(SOAPY_SDR_RX, self._rx_config.channel, gain)
        else:
            self._tx_config.gain = gain
            if self._device:
                self._device.setGain(SOAPY_SDR_TX, self._tx_config.channel, gain)
    
    def set_bandwidth(self, bw: float, direction: StreamDirection = StreamDirection.RX):
        """Set bandwidth"""
        if direction == StreamDirection.RX:
            self._rx_config.bandwidth = bw
            if self._device:
                self._device.setBandwidth(SOAPY_SDR_RX, self._rx_config.channel, bw)
        else:
            self._tx_config.bandwidth = bw
            if self._device:
                self._device.setBandwidth(SOAPY_SDR_TX, self._tx_config.channel, bw)


class SDRManager:
    """
    High-level SDR management class
    
    Manages device discovery, selection, and provides unified API
    for all SDR operations across different hardware.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('SDR-Manager')
        self._devices: List[SDRDeviceInfo] = []
        self._active_device: Optional[SoapySDRDevice] = None
        self._device_cache: Dict[str, SoapySDRDevice] = {}
    
    def scan_devices(self) -> List[SDRDeviceInfo]:
        """Scan for available SDR devices"""
        self._devices = SoapySDRDevice.enumerate_devices()
        
        if self._devices:
            self.logger.info(f"Found {len(self._devices)} SDR device(s):")
            for dev in self._devices:
                self.logger.info(f"  - {dev}")
        else:
            self.logger.warning("No SDR devices found")
        
        return self._devices
    
    def list_devices(self) -> List[SDRDeviceInfo]:
        """Get list of discovered devices"""
        return self._devices
    
    def get_device(self, index: int = 0) -> Optional[SoapySDRDevice]:
        """Get device by index"""
        if not self._devices:
            self.scan_devices()
        
        if index < len(self._devices):
            device_info = self._devices[index]
            cache_key = f"{device_info.driver}:{device_info.serial}"
            
            if cache_key not in self._device_cache:
                device = SoapySDRDevice(device_info)
                self._device_cache[cache_key] = device
            
            return self._device_cache[cache_key]
        
        return None
    
    def get_device_by_type(self, sdr_type: SDRType) -> Optional[SoapySDRDevice]:
        """Get first device of specified type"""
        if not self._devices:
            self.scan_devices()
        
        for dev in self._devices:
            if dev.sdr_type == sdr_type:
                return self.get_device(dev.index)
        
        return None
    
    def get_device_by_serial(self, serial: str) -> Optional[SoapySDRDevice]:
        """Get device by serial number"""
        if not self._devices:
            self.scan_devices()
        
        for dev in self._devices:
            if dev.serial == serial:
                return self.get_device(dev.index)
        
        return None
    
    def select_device(self, index: int = 0) -> bool:
        """Select and open a device as the active device"""
        device = self.get_device(index)
        if device:
            if device.open():
                self._active_device = device
                return True
        return False
    
    def get_active_device(self) -> Optional[SoapySDRDevice]:
        """Get the currently active device"""
        return self._active_device
    
    def close_all(self):
        """Close all devices"""
        for device in self._device_cache.values():
            device.close()
        self._device_cache.clear()
        self._active_device = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get manager status"""
        return {
            'soapy_available': SOAPY_AVAILABLE,
            'devices_found': len(self._devices),
            'devices': [str(d) for d in self._devices],
            'active_device': str(self._active_device.device_info) if self._active_device else None,
        }


# Singleton instance
_sdr_manager: Optional[SDRManager] = None


def get_sdr_manager() -> SDRManager:
    """Get the global SDR manager instance"""
    global _sdr_manager
    if _sdr_manager is None:
        _sdr_manager = SDRManager()
    return _sdr_manager


# Hardware specifications database
SDR_SPECIFICATIONS = {
    SDRType.BLADERF: {
        'name': 'BladeRF 1.0',
        'freq_range': (300e6, 3.8e9),
        'sample_rate_max': 40e6,
        'bandwidth_max': 28e6,
        'bits': 12,
        'full_duplex': True,
        'tx_capable': True,
    },
    SDRType.BLADERF2: {
        'name': 'BladeRF 2.0 micro',
        'freq_range': (47e6, 6e9),
        'sample_rate_max': 61.44e6,
        'bandwidth_max': 56e6,
        'bits': 12,
        'full_duplex': True,
        'tx_capable': True,
        'mimo': True,
    },
    SDRType.HACKRF: {
        'name': 'HackRF One',
        'freq_range': (1e6, 6e9),
        'sample_rate_max': 20e6,
        'bandwidth_max': 20e6,
        'bits': 8,
        'full_duplex': False,
        'tx_capable': True,
    },
    SDRType.RTLSDR: {
        'name': 'RTL-SDR',
        'freq_range': (24e6, 1.766e9),
        'sample_rate_max': 3.2e6,
        'bandwidth_max': 3.2e6,
        'bits': 8,
        'full_duplex': False,
        'tx_capable': False,
    },
    SDRType.USRP: {
        'name': 'USRP (Ettus)',
        'freq_range': (70e6, 6e9),
        'sample_rate_max': 200e6,
        'bandwidth_max': 160e6,
        'bits': 16,
        'full_duplex': True,
        'tx_capable': True,
        'mimo': True,
    },
    SDRType.LIMESDR: {
        'name': 'LimeSDR',
        'freq_range': (100e3, 3.8e9),
        'sample_rate_max': 61.44e6,
        'bandwidth_max': 61.44e6,
        'bits': 12,
        'full_duplex': True,
        'tx_capable': True,
        'mimo': True,
    },
    SDRType.AIRSPY: {
        'name': 'Airspy R2',
        'freq_range': (24e6, 1.8e9),
        'sample_rate_max': 10e6,
        'bandwidth_max': 10e6,
        'bits': 12,
        'full_duplex': False,
        'tx_capable': False,
    },
    SDRType.PLUTOSDR: {
        'name': 'ADALM-PLUTO',
        'freq_range': (325e6, 3.8e9),
        'sample_rate_max': 61.44e6,
        'bandwidth_max': 20e6,
        'bits': 12,
        'full_duplex': True,
        'tx_capable': True,
    },
}
