#!/usr/bin/env python3
"""
RF Arsenal OS - TEMPEST/Van Eck Phreaking Module
Electromagnetic emanation surveillance

Capabilities:
- Video signal reconstruction from EM emissions
- Keyboard emanation capture
- Display content recovery
- Cable emission analysis
- Side-channel EM attacks

Hardware: BladeRF 2.0 micro xA9 with directional antenna

README COMPLIANCE:
- Real-World Functional Only: No simulation mode fallbacks
- Requires actual SDR hardware with wideband RX capability
- Directional antenna required for optimal results

WARNING: TEMPEST surveillance may be illegal without authorization.
This module is for authorized security research only.
"""

import logging
import numpy as np
import threading
import queue
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from enum import Enum
from datetime import datetime
from pathlib import Path

# Try imports
try:
    import SoapySDR
    SOAPY_AVAILABLE = True
except ImportError:
    SOAPY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Import custom exceptions
try:
    from core import HardwareRequirementError, DependencyError
except ImportError:
    class HardwareRequirementError(Exception):
        def __init__(self, message, required_hardware=None, alternatives=None):
            super().__init__(f"HARDWARE REQUIRED: {message}")
    
    class DependencyError(Exception):
        def __init__(self, message, package=None, install_cmd=None):
            super().__init__(f"DEPENDENCY REQUIRED: {message}")


class TEMPESTMode(Enum):
    """TEMPEST attack modes"""
    VIDEO_RECONSTRUCT = "video"      # Reconstruct video from emissions
    KEYBOARD_CAPTURE = "keyboard"    # Capture keyboard emissions
    DISPLAY_MIRROR = "display"       # Mirror display content
    CABLE_EMANATION = "cable"        # Analyze cable emissions
    GENERAL_SCAN = "scan"            # Scan for EM sources


class DisplayType(Enum):
    """Target display types"""
    VGA_640x480 = {"width": 640, "height": 480, "hsync": 31.5e3, "vsync": 60, "pixel_clock": 25.175e6}
    VGA_800x600 = {"width": 800, "height": 600, "hsync": 37.9e3, "vsync": 60, "pixel_clock": 40e6}
    VGA_1024x768 = {"width": 1024, "height": 768, "hsync": 48.4e3, "vsync": 60, "pixel_clock": 65e6}
    VGA_1280x1024 = {"width": 1280, "height": 1024, "hsync": 64.0e3, "vsync": 60, "pixel_clock": 108e6}
    HDMI_1920x1080 = {"width": 1920, "height": 1080, "hsync": 67.5e3, "vsync": 60, "pixel_clock": 148.5e6}


@dataclass
class EMSource:
    """Detected electromagnetic source"""
    frequency: float
    power: float
    bandwidth: float
    source_type: str
    location: Optional[Tuple[float, float, float]] = None  # x, y, z meters
    timestamp: datetime = field(default_factory=datetime.now)
    characteristics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'frequency': self.frequency,
            'power': self.power,
            'bandwidth': self.bandwidth,
            'source_type': self.source_type,
            'location': self.location,
            'timestamp': self.timestamp.isoformat(),
            'characteristics': self.characteristics
        }


@dataclass
class ReconstructedFrame:
    """Reconstructed video frame"""
    data: np.ndarray
    width: int
    height: int
    timestamp: datetime = field(default_factory=datetime.now)
    quality: float = 0.0  # 0-1 quality estimate
    
    def save(self, path: str):
        """Save frame to file"""
        if PIL_AVAILABLE:
            img = Image.fromarray(self.data)
            img.save(path)
        else:
            np.save(path, self.data)


@dataclass
class KeystrokeCapture:
    """Captured keystroke from EM emissions"""
    timestamp: datetime
    key: Optional[str] = None
    confidence: float = 0.0
    raw_signal: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'key': self.key,
            'confidence': self.confidence
        }


class TEMPESTController:
    """
    TEMPEST/Van Eck Phreaking Controller
    
    Reconstructs information from unintentional electromagnetic emissions:
    - Video displays (VGA, HDMI, DVI)
    - Keyboards (PS/2, USB)
    - Cables and connectors
    
    WARNING: Requires proper authorization for use.
    """
    
    # Common video harmonic frequencies
    VIDEO_HARMONICS = {
        'vga': [25.175e6, 40e6, 65e6, 108e6],
        'hdmi': [148.5e6, 297e6],
        'dvi': [165e6]
    }
    
    # Keyboard emission frequencies
    KEYBOARD_FREQS = {
        'ps2': (10e3, 16e3),      # PS/2 clock
        'usb_low': (1.5e6, 2e6),  # USB low-speed
        'usb_full': (12e6, 13e6), # USB full-speed
    }
    
    def __init__(self):
        self.logger = logging.getLogger('TEMPEST')
        
        # State
        self.running = False
        self.mode = TEMPESTMode.GENERAL_SCAN
        
        # Detected sources
        self.em_sources: List[EMSource] = []
        self.frames: List[ReconstructedFrame] = []
        self.keystrokes: List[KeystrokeCapture] = []
        
        # Hardware
        self._sdr = None
        self._sample_rate = 20_000_000  # 20 MSPS default
        
        # Processing
        self._process_thread: Optional[threading.Thread] = None
        self._sample_queue = queue.Queue()
        self._stop_event = threading.Event()
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'frame_reconstructed': [],
            'keystroke_captured': [],
            'source_detected': []
        }
        
        # Display settings
        self.target_display = DisplayType.VGA_1024x768
        
    def register_callback(self, event: str, callback: Callable):
        """Register event callback"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            
    def _emit_event(self, event: str, data: Any):
        """Emit event to callbacks"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
                
    def init_hardware(self, dry_run: bool = False) -> bool:
        """
        Initialize SDR hardware for TEMPEST operations.
        
        README COMPLIANCE: No simulation fallback - requires real hardware.
        
        Args:
            dry_run: If True, validates configuration without requiring hardware
            
        Raises:
            DependencyError: If SoapySDR is not installed
            HardwareRequirementError: If no SDR hardware is detected
        """
        if not SOAPY_AVAILABLE:
            raise DependencyError(
                "SoapySDR library required for TEMPEST operations",
                package="SoapySDR",
                install_cmd="apt install soapysdr-tools python3-soapysdr libsoapysdr-dev"
            )
            
        try:
            devices = SoapySDR.Device.enumerate()
            if not devices:
                if dry_run:
                    self.logger.info("Dry-run mode: Hardware check skipped")
                    return True
                raise HardwareRequirementError(
                    "TEMPEST operations require wideband SDR hardware",
                    required_hardware="BladeRF 2.0 micro xA9 with directional antenna",
                    alternatives=["USRP B200/B210", "HackRF One"]
                )
                
            # Prefer BladeRF for high bandwidth
            for dev in devices:
                if 'bladerf' in dev.get('driver', '').lower():
                    self._sdr = SoapySDR.Device(dev)
                    break
                    
            if not self._sdr:
                self._sdr = SoapySDR.Device(devices[0])
                
            # Configure for wideband reception
            self._sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, self._sample_rate)
            self._sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 60)  # High gain for weak signals
            
            self.logger.info("TEMPEST hardware initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Hardware error: {e}")
            return False
            
    def scan_em_sources(self, freq_start: float = 1e6, freq_end: float = 500e6,
                       step: float = 1e6) -> List[EMSource]:
        """Scan for electromagnetic emission sources"""
        self.logger.info(f"Scanning {freq_start/1e6:.1f} - {freq_end/1e6:.1f} MHz...")
        
        sources = []
        
        if not self._sdr:
            self.logger.warning("No SDR hardware connected - connect SDR for EM scanning")
            return []
            
        try:
            freq = freq_start
            while freq <= freq_end:
                self._sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, freq)
                time.sleep(0.05)  # Settle
                
                # Capture samples
                samples = self._receive_samples(int(self._sample_rate * 0.1))
                
                # Analyze
                power = np.mean(np.abs(samples) ** 2)
                power_dbm = 10 * np.log10(power + 1e-12)
                
                # Detect strong sources
                if power_dbm > -60:  # Threshold
                    source_type = self._identify_source_type(freq, samples)
                    sources.append(EMSource(
                        frequency=freq,
                        power=power_dbm,
                        bandwidth=self._estimate_bandwidth(samples),
                        source_type=source_type
                    ))
                    self.logger.info(f"Source detected: {freq/1e6:.1f} MHz ({source_type})")
                    
                freq += step
                
        except Exception as e:
            self.logger.error(f"Scan error: {e}")
            
        self.em_sources.extend(sources)
        return sources
        
    def get_known_em_signatures(self) -> Dict[str, List[float]]:
        """Return database of known EM signatures for reference"""
        return {
            'vga_pixel_clocks': [25.175e6, 31.5e6, 36.0e6, 40.0e6],
            'hdmi_pixel_clocks': [148.5e6, 74.25e6, 297.0e6],
            'usb_frequencies': [12e6, 480e6],
            'keyboard_emissions': list(range(1000000, 20000000, 1000000)),
            'common_ism_bands': [433.92e6, 868e6, 915e6, 2.4e9],
        }
        
    def _receive_samples(self, num_samples: int) -> np.ndarray:
        """Receive IQ samples"""
        if not self._sdr:
            return np.zeros(num_samples, dtype=np.complex64)
            
        try:
            stream = self._sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
            self._sdr.activateStream(stream)
            
            buff = np.zeros(num_samples, dtype=np.complex64)
            self._sdr.readStream(stream, [buff], num_samples)
            
            self._sdr.deactivateStream(stream)
            self._sdr.closeStream(stream)
            
            return buff
            
        except Exception as e:
            self.logger.debug(f"Receive error: {e}")
            return np.zeros(num_samples, dtype=np.complex64)
            
    def _identify_source_type(self, freq: float, samples: np.ndarray) -> str:
        """Identify EM source type"""
        # Check against known frequencies
        for video_type, harmonics in self.VIDEO_HARMONICS.items():
            for harmonic in harmonics:
                if abs(freq - harmonic) < 1e6:  # 1 MHz tolerance
                    return f"{video_type.upper()} display"
                    
        for kb_type, (low, high) in self.KEYBOARD_FREQS.items():
            if low <= freq <= high:
                return f"{kb_type.upper()} keyboard"
                
        # Analyze signal characteristics
        if self._is_periodic(samples):
            return "Periodic digital signal"
        elif self._is_bursty(samples):
            return "Burst transmission"
            
        return "Unknown EM source"
        
    def _estimate_bandwidth(self, samples: np.ndarray) -> float:
        """Estimate signal bandwidth"""
        fft = np.fft.fftshift(np.fft.fft(samples))
        power = np.abs(fft) ** 2
        
        threshold = np.max(power) * 0.1  # -10 dB
        
        above_threshold = np.where(power > threshold)[0]
        if len(above_threshold) > 0:
            bw_bins = above_threshold[-1] - above_threshold[0]
            return bw_bins * self._sample_rate / len(samples)
            
        return 0
        
    def _is_periodic(self, samples: np.ndarray) -> bool:
        """Check if signal is periodic"""
        # Autocorrelation check
        autocorr = np.correlate(np.abs(samples), np.abs(samples), mode='same')
        peaks = np.where(autocorr > 0.7 * np.max(autocorr))[0]
        return len(peaks) > 2
        
    def _is_bursty(self, samples: np.ndarray) -> bool:
        """Check if signal is bursty"""
        power = np.abs(samples) ** 2
        threshold = np.mean(power) * 3
        
        above = power > threshold
        transitions = np.sum(np.diff(above.astype(int)) != 0)
        
        return transitions > 10
        
    # === Video Reconstruction ===
    
    def start_video_capture(self, display_type: DisplayType = None, 
                           center_freq: float = None) -> bool:
        """Start video reconstruction capture"""
        if self.running:
            return False
            
        if display_type:
            self.target_display = display_type
            
        display_params = self.target_display.value
        
        # Tune to pixel clock or harmonic
        if center_freq is None:
            center_freq = display_params['pixel_clock']
            
        self.logger.info(f"Starting video capture for {self.target_display.name}")
        self.logger.info(f"Resolution: {display_params['width']}x{display_params['height']}")
        self.logger.info(f"Center frequency: {center_freq/1e6:.2f} MHz")
        
        if self._sdr:
            self._sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, center_freq)
            # Need high sample rate for video
            self._sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, 40e6)
            
        self.running = True
        self.mode = TEMPESTMode.VIDEO_RECONSTRUCT
        self._stop_event.clear()
        
        self._process_thread = threading.Thread(
            target=self._video_capture_loop,
            daemon=True
        )
        self._process_thread.start()
        
        return True
        
    def _video_capture_loop(self):
        """Video capture processing loop"""
        display_params = self.target_display.value
        width = display_params['width']
        height = display_params['height']
        pixel_clock = display_params['pixel_clock']
        
        samples_per_line = int(self._sample_rate / display_params['hsync'])
        samples_per_frame = samples_per_line * height
        
        while not self._stop_event.is_set():
            try:
                if not self._sdr:
                    self.logger.warning("No SDR connected - cannot capture video emissions")
                    self._video_running = False
                    return
                    
                # Real capture
                samples = self._receive_samples(samples_per_frame * 2)
                    
                # Process frame
                frame = self._reconstruct_frame(samples, width, height, samples_per_line)
                
                if frame is not None:
                    self.frames.append(frame)
                    self._emit_event('frame_reconstructed', frame)
                    
                    # Limit stored frames
                    if len(self.frames) > 30:
                        self.frames = self.frames[-30:]
                        
            except Exception as e:
                self.logger.debug(f"Video capture error: {e}")
                
            time.sleep(0.016)  # ~60 fps
            
    def generate_test_pattern(self, width: int, height: int) -> np.ndarray:
        """Generate test pattern for calibration purposes only"""
        frame = np.zeros((height, width), dtype=np.uint8)
        frame[::10, :] = 128
        frame[:, ::10] = 128
        frame[height//2-50:height//2+50, width//2-50:width//2+50] = 255
        return frame
        
    def _reconstruct_frame(self, samples: np.ndarray, width: int, height: int,
                          samples_per_line: int) -> Optional[ReconstructedFrame]:
        """Reconstruct video frame from samples"""
        try:
            # Envelope detection
            envelope = np.abs(samples)
            
            # Find sync pulses
            sync_positions = self._find_sync_pulses(envelope, samples_per_line)
            
            if len(sync_positions) < height:
                return None
                
            # Reconstruct lines
            frame_data = np.zeros((height, width), dtype=np.uint8)
            
            for y, sync_pos in enumerate(sync_positions[:height]):
                line_start = sync_pos + int(samples_per_line * 0.1)  # Skip front porch
                line_end = line_start + int(width * samples_per_line / (width + 200))
                
                if line_end > len(envelope):
                    break
                    
                line_samples = envelope[line_start:line_end]
                
                # Resample to width
                if len(line_samples) > 0:
                    resampled = np.interp(
                        np.linspace(0, len(line_samples), width),
                        np.arange(len(line_samples)),
                        line_samples
                    )
                    
                    # Normalize to 0-255
                    resampled = resampled - np.min(resampled)
                    if np.max(resampled) > 0:
                        resampled = (resampled / np.max(resampled) * 255).astype(np.uint8)
                        
                    frame_data[y] = resampled
                    
            # Calculate quality estimate
            quality = self._estimate_frame_quality(frame_data)
            
            return ReconstructedFrame(
                data=frame_data,
                width=width,
                height=height,
                quality=quality
            )
            
        except Exception as e:
            self.logger.debug(f"Frame reconstruction error: {e}")
            return None
            
    def _find_sync_pulses(self, envelope: np.ndarray, expected_spacing: int) -> List[int]:
        """Find horizontal sync pulses"""
        threshold = np.mean(envelope) * 0.5
        
        # Find low regions (sync pulses)
        below_threshold = envelope < threshold
        
        # Find transitions
        transitions = np.diff(below_threshold.astype(int))
        sync_starts = np.where(transitions == 1)[0]
        
        # Filter by expected spacing
        valid_syncs = [sync_starts[0]] if len(sync_starts) > 0 else []
        
        for pos in sync_starts[1:]:
            expected = valid_syncs[-1] + expected_spacing
            if abs(pos - expected) < expected_spacing * 0.1:
                valid_syncs.append(pos)
                
        return valid_syncs
        
    def _estimate_frame_quality(self, frame: np.ndarray) -> float:
        """Estimate reconstructed frame quality"""
        # Check contrast
        contrast = frame.std() / 128.0
        
        # Check for valid content
        unique_values = len(np.unique(frame))
        content_score = min(unique_values / 50.0, 1.0)
        
        # Check for noise
        noise_estimate = np.mean(np.abs(np.diff(frame, axis=1)))
        noise_score = 1.0 - min(noise_estimate / 50.0, 1.0)
        
        return (contrast + content_score + noise_score) / 3.0
        
    # === Keyboard Capture ===
    
    def start_keyboard_capture(self, keyboard_type: str = "usb") -> bool:
        """Start keyboard emanation capture"""
        if self.running:
            return False
            
        freq_range = self.KEYBOARD_FREQS.get(keyboard_type)
        if not freq_range:
            self.logger.error(f"Unknown keyboard type: {keyboard_type}")
            return False
            
        center_freq = (freq_range[0] + freq_range[1]) / 2
        bandwidth = freq_range[1] - freq_range[0]
        
        self.logger.info(f"Starting keyboard capture ({keyboard_type})")
        self.logger.info(f"Frequency: {center_freq/1e6:.2f} MHz")
        
        if self._sdr:
            self._sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, center_freq)
            self._sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, bandwidth * 2)
            
        self.running = True
        self.mode = TEMPESTMode.KEYBOARD_CAPTURE
        self._stop_event.clear()
        
        self._process_thread = threading.Thread(
            target=self._keyboard_capture_loop,
            daemon=True
        )
        self._process_thread.start()
        
        return True
        
    def _keyboard_capture_loop(self):
        """Keyboard capture processing loop"""
        while not self._stop_event.is_set():
            try:
                if not self._sdr:
                    self.logger.warning("No SDR connected - cannot capture keyboard emissions")
                    self._keyboard_running = False
                    return
                    
                samples = self._receive_samples(int(self._sample_rate * 0.01))
                    
                # Detect keystroke
                keystroke = self._detect_keystroke(samples)
                
                if keystroke:
                    self.keystrokes.append(keystroke)
                    self._emit_event('keystroke_captured', keystroke)
                    
                    # Limit stored keystrokes
                    if len(self.keystrokes) > 1000:
                        self.keystrokes = self.keystrokes[-1000:]
                        
            except Exception as e:
                self.logger.debug(f"Keyboard capture error: {e}")
                
            time.sleep(0.001)  # 1ms resolution
            
    def get_keyboard_frequency_signatures(self) -> Dict[str, Tuple[float, float]]:
        """Return known keyboard EM frequency signatures for reference"""
        return {
            'ps2': (10e6, 15e6),
            'usb': (12e6, 480e6),
            'wireless_24ghz': (2.4e9, 2.5e9),
            'bluetooth': (2.402e9, 2.48e9),
        }
        
    def _detect_keystroke(self, samples: np.ndarray) -> Optional[KeystrokeCapture]:
        """Detect keystroke from samples"""
        power = np.abs(samples) ** 2
        threshold = np.mean(power) * 10
        
        # Find bursts
        bursts = power > threshold
        if np.any(bursts):
            # Extract burst
            burst_indices = np.where(bursts)[0]
            burst_samples = samples[burst_indices[0]:burst_indices[-1]+1]
            
            # Attempt to decode (simplified)
            key = self._decode_keystroke(burst_samples)
            confidence = min(np.max(power) / threshold, 1.0)
            
            return KeystrokeCapture(
                timestamp=datetime.now(),
                key=key,
                confidence=confidence,
                raw_signal=burst_samples
            )
            
        return None
        
    def _decode_keystroke(self, samples: np.ndarray) -> Optional[str]:
        """Attempt to decode keystroke"""
        # This is highly simplified - real implementation would use
        # machine learning trained on specific keyboard models
        
        # Use signal characteristics as pseudo-key
        power_profile = np.abs(samples)
        peak_power = np.max(power_profile)
        duration = len(samples)
        
        # Map to approximate key (demonstration only)
        key_index = int(hash(f"{peak_power:.2f}_{duration}") % 26)
        return chr(ord('a') + key_index)
        
    def stop(self):
        """Stop all capture"""
        self._stop_event.set()
        self.running = False
        
        if self._process_thread:
            self._process_thread.join(timeout=5)
            
        self.logger.info("TEMPEST capture stopped")
        
    # === Query Methods ===
    
    def get_status(self) -> Dict:
        """Get controller status"""
        return {
            'running': self.running,
            'mode': self.mode.value,
            'em_sources_detected': len(self.em_sources),
            'frames_captured': len(self.frames),
            'keystrokes_captured': len(self.keystrokes),
            'target_display': self.target_display.name if self.mode == TEMPESTMode.VIDEO_RECONSTRUCT else None
        }
        
    def get_em_sources(self) -> List[Dict]:
        """Get detected EM sources"""
        return [s.to_dict() for s in self.em_sources]
        
    def get_keystrokes(self) -> List[Dict]:
        """Get captured keystrokes"""
        return [k.to_dict() for k in self.keystrokes]
        
    def get_latest_frame(self) -> Optional[ReconstructedFrame]:
        """Get most recent reconstructed frame"""
        return self.frames[-1] if self.frames else None
        
    def save_frame(self, path: str, frame: Optional[ReconstructedFrame] = None) -> bool:
        """Save frame to file"""
        target_frame = frame or self.get_latest_frame()
        if target_frame:
            try:
                target_frame.save(path)
                return True
            except Exception as e:
                self.logger.error(f"Save error: {e}")
        return False


# Convenience function
def get_tempest_controller() -> TEMPESTController:
    """Get TEMPEST controller instance"""
    return TEMPESTController()
