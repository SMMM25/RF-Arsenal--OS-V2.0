#!/usr/bin/env python3
"""
RF Arsenal OS - Signal Replay Library
Capture, store, analyze, and replay RF signals

CAPABILITIES:
- Capture RF signals from SDR hardware
- Store signals with metadata in organized library
- Analyze signal characteristics (frequency, modulation, encoding)
- Modify signals before replay
- Replay captured signals
- Build personal signal database

COMMON USE CASES:
- Keyfobs (cars, garage doors)
- Wireless sensors (door/window, motion)
- Remote controls
- Tire pressure monitors (TPMS)
- Wireless doorbells
- ISM band devices (433 MHz, 915 MHz)

WHITE HAT USE: Authorized security testing of wireless systems
"""

import os
import sys
import json
import hashlib
import logging
import time
import struct
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any, Callable
from enum import Enum, auto
from datetime import datetime
import base64
import numpy as np

logger = logging.getLogger(__name__)


class ModulationType(Enum):
    """Common modulation types for captured signals"""
    OOK = "OOK"              # On-Off Keying (most keyfobs)
    ASK = "ASK"              # Amplitude Shift Keying
    FSK = "FSK"              # Frequency Shift Keying
    GFSK = "GFSK"            # Gaussian FSK (Bluetooth, etc)
    PSK = "PSK"              # Phase Shift Keying
    QPSK = "QPSK"            # Quadrature PSK
    MSK = "MSK"              # Minimum Shift Keying
    LORA = "LoRa"            # LoRa spread spectrum
    UNKNOWN = "Unknown"


class SignalCategory(Enum):
    """Categories for organizing signals"""
    KEYFOB = "keyfob"
    GARAGE_DOOR = "garage_door"
    CAR_KEY = "car_key"
    WIRELESS_SENSOR = "wireless_sensor"
    DOORBELL = "doorbell"
    REMOTE_CONTROL = "remote_control"
    TPMS = "tpms"
    WEATHER_STATION = "weather_station"
    SMART_HOME = "smart_home"
    INDUSTRIAL = "industrial"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class EncodingType(Enum):
    """Common encoding schemes"""
    RAW = "raw"
    MANCHESTER = "manchester"
    BIPHASE = "biphase"
    PWM = "pwm"
    PPM = "ppm"
    NRZ = "nrz"
    PRINCETON = "princeton"      # PT2260/PT2262 chips
    HOLTEK = "holtek"            # HT6P20 chips
    KEELOQ = "keeloq"            # Rolling code
    NICE_FLO = "nice_flo"        # Nice Flor-S
    CAME = "came"                # Came remotes
    UNKNOWN = "unknown"


@dataclass
class SignalMetadata:
    """Metadata for a captured signal"""
    signal_id: str
    name: str
    description: str = ""
    category: SignalCategory = SignalCategory.UNKNOWN
    
    # RF characteristics
    frequency: int = 0                  # Hz
    sample_rate: int = 0                # samples/sec
    bandwidth: int = 0                  # Hz
    modulation: ModulationType = ModulationType.UNKNOWN
    encoding: EncodingType = EncodingType.UNKNOWN
    
    # Timing
    bit_rate: int = 0                   # bits/sec
    symbol_duration_us: int = 0         # microseconds
    preamble_length: int = 0
    
    # Capture info
    capture_time: str = ""
    capture_location: str = ""
    capture_device: str = ""
    signal_strength_dbm: int = -100
    
    # Analysis results
    decoded_data: str = ""              # Hex string of decoded bits
    rolling_code: bool = False          # Uses rolling code?
    replay_safe: bool = True            # Safe to replay?
    
    # File info
    raw_file: str = ""                  # Path to raw IQ data
    demod_file: str = ""                # Path to demodulated data
    file_size_bytes: int = 0
    duration_ms: int = 0
    
    # Tags for organization
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # Integrity
    checksum: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['category'] = self.category.value
        data['modulation'] = self.modulation.value
        data['encoding'] = self.encoding.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SignalMetadata':
        """Create from dictionary"""
        data['category'] = SignalCategory(data.get('category', 'unknown'))
        data['modulation'] = ModulationType(data.get('modulation', 'Unknown'))
        data['encoding'] = EncodingType(data.get('encoding', 'unknown'))
        return cls(**data)


@dataclass
class CaptureSettings:
    """Settings for signal capture"""
    frequency: int = 433_920_000        # 433.92 MHz default
    sample_rate: int = 2_000_000        # 2 MSPS
    bandwidth: int = 1_000_000          # 1 MHz
    gain: int = 40                      # dB
    duration_ms: int = 5000             # 5 seconds
    trigger_level: float = 0.1          # Auto-trigger threshold
    pre_trigger_ms: int = 100           # Pre-trigger buffer


class SignalAnalyzer:
    """Analyze captured RF signals"""
    
    # Common frequencies and their typical uses
    COMMON_FREQUENCIES = {
        315_000_000: ("315 MHz", "US garage doors, older keyfobs"),
        433_920_000: ("433.92 MHz", "EU keyfobs, sensors, remotes"),
        868_000_000: ("868 MHz", "EU smart home, LoRa"),
        915_000_000: ("915 MHz", "US ISM band, LoRa"),
        2_400_000_000: ("2.4 GHz", "WiFi, Bluetooth, ZigBee"),
    }
    
    def __init__(self):
        self.logger = logging.getLogger('SignalAnalyzer')
        
    def analyze_signal(self, iq_data: np.ndarray, sample_rate: int, 
                       frequency: int) -> Dict[str, Any]:
        """
        Analyze captured IQ signal
        
        Returns analysis results including modulation type, timing, etc.
        """
        results = {
            'frequency': frequency,
            'sample_rate': sample_rate,
            'duration_ms': len(iq_data) * 1000 / sample_rate,
            'modulation': ModulationType.UNKNOWN,
            'encoding': EncodingType.UNKNOWN,
            'bit_rate': 0,
            'decoded_data': '',
            'confidence': 0.0
        }
        
        try:
            # Calculate signal envelope (amplitude)
            envelope = np.abs(iq_data)
            
            # Estimate noise floor and signal level
            noise_floor = np.percentile(envelope, 10)
            signal_level = np.percentile(envelope, 90)
            snr = 20 * np.log10(signal_level / noise_floor) if noise_floor > 0 else 0
            results['snr_db'] = snr
            
            # Detect modulation type
            results['modulation'] = self._detect_modulation(iq_data, envelope)
            
            # For OOK/ASK signals, try to decode
            if results['modulation'] in (ModulationType.OOK, ModulationType.ASK):
                timing, bits = self._decode_ook(envelope, sample_rate)
                results['symbol_duration_us'] = timing
                results['bit_rate'] = int(1_000_000 / timing) if timing > 0 else 0
                results['decoded_data'] = bits
                results['encoding'] = self._detect_encoding(bits)
                results['confidence'] = 0.8 if bits else 0.3
            
            # Check for rolling code indicators
            results['rolling_code'] = self._check_rolling_code(results)
            
        except Exception as e:
            self.logger.error(f"Analysis error: {e}")
            results['error'] = str(e)
            
        return results
    
    def _detect_modulation(self, iq_data: np.ndarray, 
                           envelope: np.ndarray) -> ModulationType:
        """Detect modulation type from IQ data"""
        # Calculate amplitude variance
        amp_variance = np.var(envelope)
        amp_mean = np.mean(envelope)
        
        # Calculate phase
        phase = np.angle(iq_data)
        phase_variance = np.var(np.diff(phase))
        
        # Calculate frequency deviation (for FSK)
        inst_freq = np.diff(np.unwrap(phase))
        freq_variance = np.var(inst_freq)
        
        # Simple heuristic classification
        # OOK/ASK: High amplitude variance, low phase variance
        # FSK: Low amplitude variance, high frequency variance
        # PSK: Low amplitude variance, high phase variance
        
        amp_ratio = amp_variance / (amp_mean**2) if amp_mean > 0 else 0
        
        if amp_ratio > 0.3:
            # High amplitude modulation
            if amp_ratio > 0.8:
                return ModulationType.OOK
            else:
                return ModulationType.ASK
        elif freq_variance > 0.1:
            return ModulationType.FSK
        elif phase_variance > 0.5:
            return ModulationType.PSK
        
        return ModulationType.UNKNOWN
    
    def _decode_ook(self, envelope: np.ndarray, 
                    sample_rate: int) -> Tuple[int, str]:
        """
        Decode OOK signal to bits
        
        Returns: (symbol_duration_us, bit_string)
        """
        # Threshold the signal
        threshold = (np.max(envelope) + np.min(envelope)) / 2
        binary = (envelope > threshold).astype(int)
        
        # Find edges (transitions)
        edges = np.diff(binary)
        rising = np.where(edges == 1)[0]
        falling = np.where(edges == -1)[0]
        
        if len(rising) < 2 or len(falling) < 2:
            return 0, ""
        
        # Calculate pulse widths
        pulse_widths = []
        for i in range(min(len(rising), len(falling)) - 1):
            if rising[i] < falling[i]:
                # High pulse
                width = falling[i] - rising[i]
                pulse_widths.append(('H', width))
            if i < len(falling) and falling[i] < rising[i + 1] if i + 1 < len(rising) else True:
                # Low pulse
                if i + 1 < len(rising):
                    width = rising[i + 1] - falling[i]
                    pulse_widths.append(('L', width))
        
        if not pulse_widths:
            return 0, ""
        
        # Estimate symbol duration from most common pulse width
        widths = [w for _, w in pulse_widths]
        symbol_samples = int(np.median(widths))
        symbol_duration_us = int(symbol_samples * 1_000_000 / sample_rate)
        
        # Decode bits based on pulse widths
        bits = []
        for level, width in pulse_widths:
            num_symbols = round(width / symbol_samples)
            if level == 'H':
                bits.extend(['1'] * num_symbols)
            else:
                bits.extend(['0'] * num_symbols)
        
        bit_string = ''.join(bits)
        
        # Convert to hex for display
        hex_str = ""
        for i in range(0, len(bit_string) - 7, 8):
            byte = bit_string[i:i+8]
            hex_str += format(int(byte, 2), '02X')
        
        return symbol_duration_us, hex_str
    
    def _detect_encoding(self, bit_data: str) -> EncodingType:
        """Detect encoding scheme from decoded bits"""
        if not bit_data:
            return EncodingType.UNKNOWN
        
        # Look for common patterns
        # Princeton (PT2260): 12 bits address + 4 bits data, repeated
        if len(bit_data) >= 32:
            # Check for repetition pattern
            half = len(bit_data) // 2
            if bit_data[:half] == bit_data[half:2*half]:
                return EncodingType.PRINCETON
        
        # Manchester encoding: transitions in middle of each bit
        # (would need raw signal, not decoded bits)
        
        return EncodingType.RAW
    
    def _check_rolling_code(self, analysis: Dict) -> bool:
        """Check if signal likely uses rolling code"""
        # Rolling codes typically have:
        # - Fixed portion (serial number)
        # - Variable portion (counter/code)
        # - Usually longer than simple fixed codes
        
        decoded = analysis.get('decoded_data', '')
        
        # KeeLoq is typically 66 bits
        if len(decoded) * 4 >= 64:  # hex chars * 4 bits
            return True
        
        # Check for known rolling code patterns
        # (This would need multiple captures to confirm)
        
        return False


class SignalLibrary:
    """
    Signal Replay Library
    Store, organize, and manage captured RF signals
    """
    
    def __init__(self, library_path: str = None):
        self.logger = logging.getLogger('SignalLibrary')
        
        # Set library path
        if library_path:
            self.library_path = Path(library_path)
        else:
            self.library_path = Path.home() / '.rf_arsenal' / 'signal_library'
        
        # Create directories
        self.library_path.mkdir(parents=True, exist_ok=True)
        (self.library_path / 'signals').mkdir(exist_ok=True)
        (self.library_path / 'raw').mkdir(exist_ok=True)
        
        # Load catalog
        self.catalog_file = self.library_path / 'catalog.json'
        self.catalog: Dict[str, SignalMetadata] = {}
        self._load_catalog()
        
        # Analyzer
        self.analyzer = SignalAnalyzer()
        
        self.logger.info(f"Signal library initialized: {self.library_path}")
        self.logger.info(f"Signals in library: {len(self.catalog)}")
    
    def _load_catalog(self):
        """Load signal catalog from disk"""
        if self.catalog_file.exists():
            try:
                with open(self.catalog_file, 'r') as f:
                    data = json.load(f)
                    for signal_id, meta in data.items():
                        self.catalog[signal_id] = SignalMetadata.from_dict(meta)
            except Exception as e:
                self.logger.error(f"Failed to load catalog: {e}")
                self.catalog = {}
    
    def _save_catalog(self):
        """Save signal catalog to disk"""
        try:
            data = {sid: meta.to_dict() for sid, meta in self.catalog.items()}
            with open(self.catalog_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save catalog: {e}")
    
    def _generate_signal_id(self) -> str:
        """Generate unique signal ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_part = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
        return f"SIG_{timestamp}_{random_part}"
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum"""
        return hashlib.sha256(data).hexdigest()
    
    def capture_signal(self, hardware_controller, settings: CaptureSettings,
                       name: str, category: SignalCategory = SignalCategory.UNKNOWN,
                       description: str = "") -> Optional[SignalMetadata]:
        """
        Capture RF signal from hardware
        
        Args:
            hardware_controller: SDR hardware controller
            settings: Capture settings
            name: Name for the captured signal
            category: Signal category
            description: Optional description
            
        Returns:
            SignalMetadata if successful, None otherwise
        """
        signal_id = self._generate_signal_id()
        self.logger.info(f"Capturing signal: {name} ({signal_id})")
        
        try:
            # Configure hardware
            if hardware_controller:
                hardware_controller.configure({
                    'frequency': settings.frequency,
                    'sample_rate': settings.sample_rate,
                    'bandwidth': settings.bandwidth,
                    'rx_gain': settings.gain
                })
            
            # Calculate number of samples
            num_samples = int(settings.sample_rate * settings.duration_ms / 1000)
            
            # Capture IQ data
            if hardware_controller:
                iq_data = hardware_controller.receive_samples(num_samples)
            else:
                # Generate dummy data for testing
                self.logger.warning("No hardware - generating test signal")
                t = np.arange(num_samples) / settings.sample_rate
                # Simulate OOK signal
                bits = np.repeat([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], num_samples // 10)[:num_samples]
                carrier = np.exp(2j * np.pi * 10000 * t)
                iq_data = bits * carrier * 0.8 + np.random.randn(num_samples) * 0.1
            
            # Analyze the signal
            analysis = self.analyzer.analyze_signal(
                iq_data, settings.sample_rate, settings.frequency
            )
            
            # Save raw IQ data
            raw_file = self.library_path / 'raw' / f"{signal_id}.iq"
            iq_bytes = iq_data.astype(np.complex64).tobytes()
            with open(raw_file, 'wb') as f:
                f.write(iq_bytes)
            
            # Calculate checksum
            checksum = self._calculate_checksum(iq_bytes)
            
            # Create metadata
            metadata = SignalMetadata(
                signal_id=signal_id,
                name=name,
                description=description,
                category=category,
                frequency=settings.frequency,
                sample_rate=settings.sample_rate,
                bandwidth=settings.bandwidth,
                modulation=analysis.get('modulation', ModulationType.UNKNOWN),
                encoding=analysis.get('encoding', EncodingType.UNKNOWN),
                bit_rate=analysis.get('bit_rate', 0),
                symbol_duration_us=analysis.get('symbol_duration_us', 0),
                capture_time=datetime.now().isoformat(),
                capture_device=hardware_controller.__class__.__name__ if hardware_controller else "Test",
                decoded_data=analysis.get('decoded_data', ''),
                rolling_code=analysis.get('rolling_code', False),
                replay_safe=not analysis.get('rolling_code', False),
                raw_file=str(raw_file),
                file_size_bytes=len(iq_bytes),
                duration_ms=settings.duration_ms,
                checksum=checksum
            )
            
            # Add to catalog
            self.catalog[signal_id] = metadata
            self._save_catalog()
            
            self.logger.info(f"Signal captured successfully: {signal_id}")
            self.logger.info(f"  Modulation: {metadata.modulation.value}")
            self.logger.info(f"  Decoded: {metadata.decoded_data[:32]}..." if metadata.decoded_data else "  No decode")
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Capture failed: {e}")
            return None
    
    def import_signal(self, file_path: str, name: str, 
                      frequency: int, sample_rate: int,
                      category: SignalCategory = SignalCategory.UNKNOWN,
                      description: str = "") -> Optional[SignalMetadata]:
        """
        Import signal from external file (raw IQ, WAV, etc.)
        """
        signal_id = self._generate_signal_id()
        self.logger.info(f"Importing signal: {file_path}")
        
        try:
            # Read file
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.wav':
                # WAV file
                import wave
                with wave.open(str(file_path), 'rb') as wav:
                    frames = wav.readframes(wav.getnframes())
                    iq_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                    # Assume stereo = IQ
                    if wav.getnchannels() == 2:
                        iq_data = iq_data[::2] + 1j * iq_data[1::2]
            else:
                # Assume raw IQ (complex64)
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                iq_data = np.frombuffer(raw_data, dtype=np.complex64)
            
            # Analyze
            analysis = self.analyzer.analyze_signal(iq_data, sample_rate, frequency)
            
            # Copy to library
            raw_file = self.library_path / 'raw' / f"{signal_id}.iq"
            iq_bytes = iq_data.astype(np.complex64).tobytes()
            with open(raw_file, 'wb') as f:
                f.write(iq_bytes)
            
            checksum = self._calculate_checksum(iq_bytes)
            
            # Create metadata
            metadata = SignalMetadata(
                signal_id=signal_id,
                name=name,
                description=description,
                category=category,
                frequency=frequency,
                sample_rate=sample_rate,
                modulation=analysis.get('modulation', ModulationType.UNKNOWN),
                encoding=analysis.get('encoding', EncodingType.UNKNOWN),
                bit_rate=analysis.get('bit_rate', 0),
                decoded_data=analysis.get('decoded_data', ''),
                rolling_code=analysis.get('rolling_code', False),
                replay_safe=not analysis.get('rolling_code', False),
                raw_file=str(raw_file),
                file_size_bytes=len(iq_bytes),
                duration_ms=int(len(iq_data) * 1000 / sample_rate),
                capture_time=datetime.now().isoformat(),
                checksum=checksum
            )
            
            self.catalog[signal_id] = metadata
            self._save_catalog()
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Import failed: {e}")
            return None
    
    def replay_signal(self, signal_id: str, hardware_controller,
                      repeat: int = 1, delay_ms: int = 100,
                      tx_gain: int = 40) -> bool:
        """
        Replay a captured signal
        
        Args:
            signal_id: ID of signal to replay
            hardware_controller: SDR hardware controller
            repeat: Number of times to repeat
            delay_ms: Delay between repeats (ms)
            tx_gain: Transmit gain (dB)
            
        Returns:
            True if successful
        """
        if signal_id not in self.catalog:
            self.logger.error(f"Signal not found: {signal_id}")
            return False
        
        metadata = self.catalog[signal_id]
        
        # Safety check
        if not metadata.replay_safe:
            self.logger.warning(f"Signal uses rolling code - replay may not work!")
        
        self.logger.info(f"Replaying signal: {metadata.name} ({signal_id})")
        self.logger.info(f"  Frequency: {metadata.frequency / 1e6:.3f} MHz")
        self.logger.info(f"  Repeats: {repeat}")
        
        try:
            # Load signal data
            with open(metadata.raw_file, 'rb') as f:
                raw_data = f.read()
            
            # Verify integrity
            if self._calculate_checksum(raw_data) != metadata.checksum:
                self.logger.error("Signal file corrupted - checksum mismatch!")
                return False
            
            iq_data = np.frombuffer(raw_data, dtype=np.complex64)
            
            if hardware_controller:
                # Configure hardware
                hardware_controller.configure({
                    'frequency': metadata.frequency,
                    'sample_rate': metadata.sample_rate,
                    'bandwidth': metadata.bandwidth,
                    'tx_gain': tx_gain
                })
                
                # Replay
                for i in range(repeat):
                    self.logger.info(f"  Transmitting {i+1}/{repeat}...")
                    hardware_controller.transmit_samples(iq_data)
                    
                    if i < repeat - 1:
                        time.sleep(delay_ms / 1000)
                
                self.logger.info("Replay complete")
                return True
            else:
                self.logger.warning("No hardware - simulating replay")
                for i in range(repeat):
                    self.logger.info(f"  [SIMULATED] Transmitting {i+1}/{repeat}...")
                    time.sleep(delay_ms / 1000)
                return True
                
        except Exception as e:
            self.logger.error(f"Replay failed: {e}")
            return False
    
    def modify_signal(self, signal_id: str, 
                      modifications: Dict[str, Any]) -> Optional[str]:
        """
        Create modified copy of a signal
        
        Modifications can include:
        - frequency_shift: Hz to shift
        - amplitude_scale: Multiplier
        - time_stretch: Time scaling factor
        - bit_flip: List of bit positions to flip
        
        Returns new signal_id if successful
        """
        if signal_id not in self.catalog:
            return None
        
        metadata = self.catalog[signal_id]
        
        try:
            # Load original
            with open(metadata.raw_file, 'rb') as f:
                iq_data = np.frombuffer(f.read(), dtype=np.complex64).copy()
            
            # Apply modifications
            if 'frequency_shift' in modifications:
                shift = modifications['frequency_shift']
                t = np.arange(len(iq_data)) / metadata.sample_rate
                iq_data *= np.exp(2j * np.pi * shift * t)
                
            if 'amplitude_scale' in modifications:
                iq_data *= modifications['amplitude_scale']
                
            if 'time_stretch' in modifications:
                from scipy import signal as scipy_signal
                factor = modifications['time_stretch']
                new_length = int(len(iq_data) * factor)
                iq_data = scipy_signal.resample(iq_data, new_length)
            
            # Save as new signal
            new_id = self._generate_signal_id()
            raw_file = self.library_path / 'raw' / f"{new_id}.iq"
            iq_bytes = iq_data.astype(np.complex64).tobytes()
            
            with open(raw_file, 'wb') as f:
                f.write(iq_bytes)
            
            # Create new metadata
            new_metadata = SignalMetadata(
                signal_id=new_id,
                name=f"{metadata.name} (modified)",
                description=f"Modified from {signal_id}: {modifications}",
                category=metadata.category,
                frequency=metadata.frequency + modifications.get('frequency_shift', 0),
                sample_rate=metadata.sample_rate,
                bandwidth=metadata.bandwidth,
                modulation=metadata.modulation,
                encoding=metadata.encoding,
                capture_time=datetime.now().isoformat(),
                raw_file=str(raw_file),
                file_size_bytes=len(iq_bytes),
                duration_ms=int(len(iq_data) * 1000 / metadata.sample_rate),
                checksum=self._calculate_checksum(iq_bytes),
                tags=metadata.tags + ['modified'],
                notes=f"Source: {signal_id}"
            )
            
            self.catalog[new_id] = new_metadata
            self._save_catalog()
            
            return new_id
            
        except Exception as e:
            self.logger.error(f"Modification failed: {e}")
            return None
    
    def delete_signal(self, signal_id: str) -> bool:
        """Delete a signal from the library"""
        if signal_id not in self.catalog:
            return False
        
        metadata = self.catalog[signal_id]
        
        try:
            # Delete raw file
            if metadata.raw_file and Path(metadata.raw_file).exists():
                Path(metadata.raw_file).unlink()
            
            # Delete demod file if exists
            if metadata.demod_file and Path(metadata.demod_file).exists():
                Path(metadata.demod_file).unlink()
            
            # Remove from catalog
            del self.catalog[signal_id]
            self._save_catalog()
            
            self.logger.info(f"Deleted signal: {signal_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Delete failed: {e}")
            return False
    
    def search_signals(self, 
                       category: SignalCategory = None,
                       frequency_range: Tuple[int, int] = None,
                       modulation: ModulationType = None,
                       tags: List[str] = None,
                       name_contains: str = None) -> List[SignalMetadata]:
        """Search for signals matching criteria"""
        results = []
        
        for metadata in self.catalog.values():
            # Category filter
            if category and metadata.category != category:
                continue
            
            # Frequency filter
            if frequency_range:
                if not (frequency_range[0] <= metadata.frequency <= frequency_range[1]):
                    continue
            
            # Modulation filter
            if modulation and metadata.modulation != modulation:
                continue
            
            # Tags filter
            if tags:
                if not any(t in metadata.tags for t in tags):
                    continue
            
            # Name filter
            if name_contains:
                if name_contains.lower() not in metadata.name.lower():
                    continue
            
            results.append(metadata)
        
        return results
    
    def get_signal(self, signal_id: str) -> Optional[SignalMetadata]:
        """Get signal metadata by ID"""
        return self.catalog.get(signal_id)
    
    def list_signals(self, limit: int = 50) -> List[SignalMetadata]:
        """List all signals in library"""
        signals = list(self.catalog.values())
        # Sort by capture time, newest first
        signals.sort(key=lambda s: s.capture_time, reverse=True)
        return signals[:limit]
    
    def get_statistics(self) -> Dict:
        """Get library statistics"""
        stats = {
            'total_signals': len(self.catalog),
            'by_category': {},
            'by_modulation': {},
            'total_size_mb': 0,
            'frequency_range': {'min': float('inf'), 'max': 0},
            'rolling_code_signals': 0
        }
        
        for metadata in self.catalog.values():
            # By category
            cat = metadata.category.value
            stats['by_category'][cat] = stats['by_category'].get(cat, 0) + 1
            
            # By modulation
            mod = metadata.modulation.value
            stats['by_modulation'][mod] = stats['by_modulation'].get(mod, 0) + 1
            
            # Size
            stats['total_size_mb'] += metadata.file_size_bytes / (1024 * 1024)
            
            # Frequency range
            if metadata.frequency > 0:
                stats['frequency_range']['min'] = min(stats['frequency_range']['min'], metadata.frequency)
                stats['frequency_range']['max'] = max(stats['frequency_range']['max'], metadata.frequency)
            
            # Rolling code
            if metadata.rolling_code:
                stats['rolling_code_signals'] += 1
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        
        return stats
    
    def export_signal(self, signal_id: str, output_path: str, 
                      format: str = 'iq') -> bool:
        """Export signal to external file"""
        if signal_id not in self.catalog:
            return False
        
        metadata = self.catalog[signal_id]
        
        try:
            with open(metadata.raw_file, 'rb') as f:
                iq_data = np.frombuffer(f.read(), dtype=np.complex64)
            
            output_path = Path(output_path)
            
            if format == 'iq':
                # Raw IQ
                with open(output_path, 'wb') as f:
                    f.write(iq_data.tobytes())
                    
            elif format == 'wav':
                # WAV file (stereo I/Q)
                import wave
                with wave.open(str(output_path), 'wb') as wav:
                    wav.setnchannels(2)
                    wav.setsampwidth(2)  # 16-bit
                    wav.setframerate(metadata.sample_rate)
                    
                    # Interleave I and Q
                    i_data = np.real(iq_data)
                    q_data = np.imag(iq_data)
                    interleaved = np.empty(len(iq_data) * 2, dtype=np.int16)
                    interleaved[0::2] = (i_data * 32767).astype(np.int16)
                    interleaved[1::2] = (q_data * 32767).astype(np.int16)
                    wav.writeframes(interleaved.tobytes())
                    
            elif format == 'json':
                # Metadata only
                with open(output_path, 'w') as f:
                    json.dump(metadata.to_dict(), f, indent=2)
            
            else:
                self.logger.error(f"Unknown format: {format}")
                return False
            
            self.logger.info(f"Exported {signal_id} to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return False


# Global instance
_signal_library: Optional[SignalLibrary] = None


def get_signal_library(library_path: str = None) -> SignalLibrary:
    """Get global signal library instance"""
    global _signal_library
    if _signal_library is None:
        _signal_library = SignalLibrary(library_path)
    return _signal_library


if __name__ == "__main__":
    # Demo
    logging.basicConfig(level=logging.INFO)
    
    library = SignalLibrary()
    
    # Test capture (no hardware)
    settings = CaptureSettings(
        frequency=433_920_000,
        sample_rate=2_000_000,
        duration_ms=1000
    )
    
    metadata = library.capture_signal(
        None, settings, 
        name="Test Keyfob",
        category=SignalCategory.KEYFOB,
        description="Test capture"
    )
    
    if metadata:
        print(f"\nCaptured: {metadata.signal_id}")
        print(f"  Name: {metadata.name}")
        print(f"  Frequency: {metadata.frequency / 1e6} MHz")
        print(f"  Modulation: {metadata.modulation.value}")
        print(f"  Decoded: {metadata.decoded_data[:32] if metadata.decoded_data else 'None'}")
    
    # List signals
    print(f"\nLibrary contains {len(library.catalog)} signals")
    
    # Statistics
    stats = library.get_statistics()
    print(f"Statistics: {json.dumps(stats, indent=2)}")
