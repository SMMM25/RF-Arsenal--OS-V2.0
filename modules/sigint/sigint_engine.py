#!/usr/bin/env python3
"""
RF Arsenal OS - SIGINT (Signals Intelligence) Engine
Hardware: BladeRF 2.0 micro xA9
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class SIGINTConfig:
    """SIGINT Configuration"""
    frequency: int = 2_400_000_000  # Center frequency
    sample_rate: int = 40_000_000   # 40 MSPS
    bandwidth: int = 40_000_000     # 40 MHz
    collection_mode: str = "passive"  # passive, active, targeted
    demodulation: str = "auto"      # auto, am, fm, ssb, digital

@dataclass
class Intercept:
    """Intercepted Signal"""
    frequency: int
    timestamp: datetime
    duration: float
    signal_type: str
    modulation: str
    bandwidth: int
    power: float
    metadata: Dict
    raw_data: Optional[np.ndarray] = None
    decoded_data: Optional[bytes] = None

class SIGINTEngine:
    """Signals Intelligence Collection and Analysis Engine"""
    
    # Signal classification patterns
    SIGNAL_SIGNATURES = {
        'cellular_2g': {
            'bandwidth': (200_000, 300_000),
            'modulation': 'GMSK',
            'frequency_range': (850e6, 1900e6)
        },
        'cellular_3g': {
            'bandwidth': (5_000_000, 5_000_000),
            'modulation': 'WCDMA',
            'frequency_range': (1900e6, 2100e6)
        },
        'cellular_4g': {
            'bandwidth': (10_000_000, 20_000_000),
            'modulation': 'OFDM',
            'frequency_range': (700e6, 2600e6)
        },
        'wifi': {
            'bandwidth': (20_000_000, 40_000_000),
            'modulation': 'OFDM',
            'frequency_range': (2400e6, 5850e6)
        },
        'bluetooth': {
            'bandwidth': (1_000_000, 1_000_000),
            'modulation': 'GFSK',
            'frequency_range': (2400e6, 2483e6)
        },
        'military_vhf': {
            'bandwidth': (25_000, 50_000),
            'modulation': 'FM',
            'frequency_range': (30e6, 90e6)
        },
        'military_uhf': {
            'bandwidth': (25_000, 50_000),
            'modulation': 'FM',
            'frequency_range': (225e6, 400e6)
        },
        'satellite': {
            'bandwidth': (100_000, 500_000),
            'modulation': 'QPSK',
            'frequency_range': (1000e6, 2000e6)
        },
        'radar': {
            'bandwidth': (1_000_000, 100_000_000),
            'modulation': 'Pulse',
            'frequency_range': (1e9, 10e9)
        }
    }
    
    def __init__(self, hardware_controller):
        """
        Initialize SIGINT engine
        
        Args:
            hardware_controller: BladeRF hardware controller instance
        """
        self.hw = hardware_controller
        self.config = SIGINTConfig()
        self.is_running = False
        self.intercepts: List[Intercept] = []
        self.signal_database: Dict[int, List[Intercept]] = defaultdict(list)
        self.patterns: Dict[str, int] = defaultdict(int)
        
    def configure(self, config: SIGINTConfig) -> bool:
        """Configure SIGINT engine"""
        try:
            self.config = config
            
            # Configure BladeRF
            if not self.hw.configure_hardware({
                'frequency': config.frequency,
                'sample_rate': config.sample_rate,
                'bandwidth': config.bandwidth,
                'rx_gain': 40,
                'tx_gain': 0
            }):
                logger.error("Failed to configure hardware")
                return False
                
            logger.info(f"SIGINT configured: {config.frequency/1e6:.1f} MHz, "
                       f"Mode: {config.collection_mode}")
            return True
            
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            return False
    
    def collect_passive(self, duration: float = 60.0) -> List[Intercept]:
        """
        Passive SIGINT collection (monitor only)
        
        Args:
            duration: Collection duration in seconds
            
        Returns:
            List of intercepted signals
        """
        try:
            logger.info(f"Starting passive SIGINT collection ({duration}s)...")
            self.is_running = True
            
            start_time = datetime.now()
            intercepts = []
            
            while self.is_running:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= duration:
                    break
                
                # Receive samples
                samples = self.hw.receive_samples(
                    int(self.config.sample_rate * 0.1)  # 100ms
                )
                
                if samples is None:
                    continue
                
                # Analyze samples
                detected_signals = self._analyze_samples(samples)
                
                # Process each detected signal
                for signal_info in detected_signals:
                    intercept = self._create_intercept(signal_info, samples)
                    intercepts.append(intercept)
                    
                    # Store in database
                    freq = intercept.frequency
                    self.signal_database[freq].append(intercept)
            
            self.intercepts.extend(intercepts)
            logger.info(f"Collected {len(intercepts)} intercepts")
            return intercepts
            
        except Exception as e:
            logger.error(f"Passive collection error: {e}")
            return []
    
    def _analyze_samples(self, samples: np.ndarray) -> List[Dict]:
        """Analyze IQ samples for signals"""
        signals = []
        
        # Perform FFT
        fft = np.fft.fftshift(np.fft.fft(samples))
        power_spectrum = np.abs(fft) ** 2
        power_db = 10 * np.log10(power_spectrum + 1e-12)
        
        # Detect peaks
        threshold = np.mean(power_db) + 15  # 15 dB above noise
        
        for i in range(10, len(power_db) - 10):
            if (power_db[i] > threshold and 
                power_db[i] > power_db[i-1] and 
                power_db[i] > power_db[i+1]):
                
                # Calculate frequency offset
                freq_bins = np.fft.fftshift(
                    np.fft.fftfreq(len(samples), 1/self.config.sample_rate)
                )
                freq_offset = freq_bins[i]
                frequency = int(self.config.frequency + freq_offset)
                
                # Estimate bandwidth
                bandwidth = self._estimate_bandwidth(power_db, i)
                
                # Classify signal
                signal_type, modulation = self._classify_signal(
                    frequency, bandwidth, power_db[i]
                )
                
                signals.append({
                    'frequency': frequency,
                    'power': power_db[i],
                    'bandwidth': bandwidth,
                    'signal_type': signal_type,
                    'modulation': modulation,
                    'peak_idx': i
                })
        
        return signals
    
    def _estimate_bandwidth(self, power_db: np.ndarray, peak_idx: int) -> int:
        """Estimate signal bandwidth"""
        peak_power = power_db[peak_idx]
        threshold = peak_power - 20  # -20 dB
        
        # Find edges
        left = peak_idx
        while left > 0 and power_db[left] > threshold:
            left -= 1
        
        right = peak_idx
        while right < len(power_db) - 1 and power_db[right] > threshold:
            right += 1
        
        # Calculate bandwidth
        bin_width = self.config.sample_rate / len(power_db)
        bandwidth = int((right - left) * bin_width)
        
        return bandwidth
    
    def _classify_signal(self, frequency: int, bandwidth: int, 
                        power: float) -> Tuple[str, str]:
        """Classify signal type and modulation"""
        # Check against known signatures
        for signal_type, signature in self.SIGNAL_SIGNATURES.items():
            freq_range = signature['frequency_range']
            bw_range = signature['bandwidth']
            
            # Check frequency range
            if not (freq_range[0] <= frequency <= freq_range[1]):
                continue
            
            # Check bandwidth
            if isinstance(bw_range, tuple):
                if bw_range[0] <= bandwidth <= bw_range[1]:
                    return signal_type, signature['modulation']
            else:
                if abs(bandwidth - bw_range) < bw_range * 0.5:
                    return signal_type, signature['modulation']
        
        # Unknown signal
        return "unknown", "unknown"
    
    def _create_intercept(self, signal_info: Dict, 
                         samples: np.ndarray) -> Intercept:
        """Create intercept record"""
        # Extract signal samples
        peak_idx = signal_info.get('peak_idx', 0)
        signal_samples = self._extract_signal(samples, peak_idx)
        
        # Attempt demodulation
        decoded = self._demodulate_signal(
            signal_samples, 
            signal_info['modulation']
        )
        
        # Build metadata
        metadata = {
            'snr': signal_info['power'] - (-80),  # Assume -80 dBm noise floor
            'collection_mode': self.config.collection_mode,
            'hardware': 'BladeRF 2.0 micro xA9'
        }
        
        intercept = Intercept(
            frequency=signal_info['frequency'],
            timestamp=datetime.now(),
            duration=len(samples) / self.config.sample_rate,
            signal_type=signal_info['signal_type'],
            modulation=signal_info['modulation'],
            bandwidth=signal_info['bandwidth'],
            power=signal_info['power'],
            metadata=metadata,
            raw_data=signal_samples,
            decoded_data=decoded
        )
        
        return intercept
    
    def _extract_signal(self, samples: np.ndarray, peak_idx: int) -> np.ndarray:
        """Extract signal around peak"""
        # Simple extraction (in production, use proper filtering)
        window_size = min(1000, len(samples))
        start = max(0, peak_idx - window_size // 2)
        end = min(len(samples), start + window_size)
        return samples[start:end]
    
    def _demodulate_signal(self, samples: np.ndarray, 
                          modulation: str) -> Optional[bytes]:
        """Demodulate signal based on modulation type"""
        try:
            if modulation == "FM":
                return self._demodulate_fm(samples)
            elif modulation == "AM":
                return self._demodulate_am(samples)
            elif modulation in ["GMSK", "GFSK"]:
                return self._demodulate_fsk(samples)
            else:
                return None
        except Exception as e:
            logger.debug(f"Demodulation error: {e}")
            return None
    
    def _demodulate_fm(self, samples: np.ndarray) -> bytes:
        """FM demodulation"""
        # Simple FM demodulation using phase difference
        phase = np.angle(samples)
        phase_diff = np.diff(phase)
        
        # Unwrap phase
        phase_diff = np.unwrap(phase_diff)
        
        # Convert to bytes (simplified)
        audio = (phase_diff / np.pi * 127 + 127).astype(np.uint8)
        return audio.tobytes()
    
    def _demodulate_am(self, samples: np.ndarray) -> bytes:
        """AM demodulation"""
        # Envelope detection
        envelope = np.abs(samples)
        
        # Convert to bytes
        audio = (envelope / np.max(envelope) * 255).astype(np.uint8)
        return audio.tobytes()
    
    def _demodulate_fsk(self, samples: np.ndarray) -> bytes:
        """FSK demodulation"""
        # Simplified FSK demodulation
        # In production, use proper FSK demodulator
        
        # Detect frequency shifts
        phase = np.angle(samples)
        freq = np.diff(phase)
        
        # Threshold to bits
        threshold = np.median(freq)
        bits = (freq > threshold).astype(np.uint8)
        
        # Convert bits to bytes
        num_bytes = len(bits) // 8
        bytes_data = np.packbits(bits[:num_bytes*8])
        
        return bytes_data.tobytes()
    
    def collect_targeted(self, target_frequency: int, 
                        duration: float = 10.0) -> List[Intercept]:
        """
        Targeted SIGINT collection (specific frequency)
        
        Args:
            target_frequency: Target frequency in Hz
            duration: Collection duration in seconds
            
        Returns:
            List of intercepts
        """
        try:
            logger.info(f"Targeted collection: {target_frequency/1e6:.1f} MHz")
            
            # Configure for target
            self.config.frequency = target_frequency
            self.configure(self.config)
            
            # Collect
            return self.collect_passive(duration)
            
        except Exception as e:
            logger.error(f"Targeted collection error: {e}")
            return []
    
    def pattern_analysis(self) -> Dict[str, any]:
        """Analyze patterns in collected signals"""
        try:
            logger.info("Performing pattern analysis...")
            
            if not self.intercepts:
                return {}
            
            # Analyze by signal type
            signal_types = defaultdict(int)
            modulation_types = defaultdict(int)
            frequency_usage = defaultdict(int)
            time_patterns = defaultdict(list)
            
            for intercept in self.intercepts:
                signal_types[intercept.signal_type] += 1
                modulation_types[intercept.modulation] += 1
                
                # Frequency usage (rounded to MHz)
                freq_mhz = int(intercept.frequency / 1e6)
                frequency_usage[freq_mhz] += 1
                
                # Time patterns (by hour)
                hour = intercept.timestamp.hour
                time_patterns[hour].append(intercept.frequency)
            
            # Find most active frequencies
            top_frequencies = sorted(
                frequency_usage.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            analysis = {
                'total_intercepts': len(self.intercepts),
                'signal_types': dict(signal_types),
                'modulation_types': dict(modulation_types),
                'top_frequencies': top_frequencies,
                'time_patterns': {
                    h: len(freqs) for h, freqs in time_patterns.items()
                },
                'collection_period': {
                    'start': min(i.timestamp for i in self.intercepts),
                    'end': max(i.timestamp for i in self.intercepts)
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Pattern analysis error: {e}")
            return {}
    
    def correlate_signals(self) -> List[Dict]:
        """Correlate related signals"""
        try:
            logger.info("Correlating signals...")
            
            correlations = []
            
            # Group by frequency proximity (within 1 MHz)
            frequency_groups = defaultdict(list)
            for intercept in self.intercepts:
                freq_key = int(intercept.frequency / 1e6)
                frequency_groups[freq_key].append(intercept)
            
            # Analyze each group
            for freq_mhz, group in frequency_groups.items():
                if len(group) < 2:
                    continue
                
                # Check for patterns
                signal_types = [i.signal_type for i in group]
                if len(set(signal_types)) == 1:
                    # Same signal type, likely related
                    correlations.append({
                        'frequency': freq_mhz * 1e6,
                        'count': len(group),
                        'signal_type': signal_types[0],
                        'time_span': (
                            max(i.timestamp for i in group) - 
                            min(i.timestamp for i in group)
                        ).total_seconds(),
                        'correlation': 'same_type'
                    })
            
            logger.info(f"Found {len(correlations)} correlations")
            return correlations
            
        except Exception as e:
            logger.error(f"Correlation error: {e}")
            return []
    
    def export_intercepts(self, filename: str = "intercepts.json") -> bool:
        """Export intercepts to file"""
        try:
            import json
            
            export_data = []
            for intercept in self.intercepts:
                export_data.append({
                    'frequency': intercept.frequency,
                    'timestamp': intercept.timestamp.isoformat(),
                    'duration': intercept.duration,
                    'signal_type': intercept.signal_type,
                    'modulation': intercept.modulation,
                    'bandwidth': intercept.bandwidth,
                    'power': intercept.power,
                    'metadata': intercept.metadata
                })
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {len(export_data)} intercepts to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Export error: {e}")
            return False
    
    def get_intercepts(self) -> List[Intercept]:
        """Get all intercepts"""
        return self.intercepts
    
    def get_signal_database(self) -> Dict[int, List[Intercept]]:
        """Get signal database"""
        return dict(self.signal_database)
    
    def stop(self):
        """Stop SIGINT operations"""
        self.is_running = False
        logger.info("SIGINT engine stopped")

def main():
    """Test SIGINT engine"""
    from core.hardware import HardwareController
    
    # Initialize hardware
    hw = HardwareController()
    if not hw.connect():
        print("Failed to connect to BladeRF")
        return
    
    # Create SIGINT engine
    sigint = SIGINTEngine(hw)
    
    # Configure
    config = SIGINTConfig(
        frequency=2_450_000_000,  # 2.45 GHz
        collection_mode="passive"
    )
    
    if not sigint.configure(config):
        print("Configuration failed")
        return
    
    print("RF Arsenal OS - SIGINT Engine")
    print("=" * 50)
    
    # Passive collection
    print("\nStarting passive SIGINT collection (10s)...")
    intercepts = sigint.collect_passive(duration=10.0)
    
    print(f"\nCollected {len(intercepts)} intercepts:")
    for i, intercept in enumerate(intercepts[:5]):  # Show first 5
        print(f"{i+1}. {intercept.frequency/1e6:.3f} MHz - "
              f"{intercept.signal_type} ({intercept.modulation}) - "
              f"{intercept.power:.1f} dBm")
    
    # Pattern analysis
    if intercepts:
        print("\nPerforming pattern analysis...")
        analysis = sigint.pattern_analysis()
        
        print(f"\nTotal intercepts: {analysis.get('total_intercepts', 0)}")
        print(f"Signal types: {analysis.get('signal_types', {})}")
        print(f"Top frequencies: {analysis.get('top_frequencies', [])[:3]}")
        
        # Export
        print("\nExporting intercepts...")
        sigint.export_intercepts("sigint_intercepts.json")
    
    sigint.stop()
    hw.disconnect()

if __name__ == "__main__":
    main()
