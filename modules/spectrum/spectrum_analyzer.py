#!/usr/bin/env python3
"""
RF Arsenal OS - Real-time Spectrum Analyzer
Hardware: BladeRF 2.0 micro xA9
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class SpectrumConfig:
    """Spectrum Analyzer Configuration"""
    start_freq: int = 70_000_000      # 70 MHz
    stop_freq: int = 6_000_000_000    # 6 GHz
    sample_rate: int = 40_000_000     # 40 MSPS
    fft_size: int = 2048              # FFT size
    averaging: int = 10               # Number of averages
    rbw: int = 10_000                 # Resolution bandwidth (Hz)
    detector: str = "peak"            # peak, average, sample

@dataclass
class Signal:
    """Detected Signal"""
    frequency: int
    power: float
    bandwidth: int
    modulation: str
    timestamp: datetime

class SpectrumAnalyzer:
    """Real-time Spectrum Analyzer with Signal Detection"""
    
    def __init__(self, hardware_controller):
        """
        Initialize spectrum analyzer
        
        Args:
            hardware_controller: BladeRF hardware controller instance
        """
        self.hw = hardware_controller
        self.config = SpectrumConfig()
        self.is_running = False
        self.spectrum_history = deque(maxlen=100)  # Keep last 100 sweeps
        self.detected_signals: Dict[int, Signal] = {}
        
    def configure(self, config: SpectrumConfig) -> bool:
        """Configure spectrum analyzer"""
        try:
            self.config = config
            
            logger.info(f"Spectrum analyzer configured: "
                       f"{config.start_freq/1e6:.1f}-{config.stop_freq/1e6:.1f} MHz, "
                       f"RBW: {config.rbw/1e3:.1f} kHz")
            return True
            
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            return False
    
    def sweep(self) -> np.ndarray:
        """
        Perform full spectrum sweep
        
        Returns:
            Power spectrum in dBm
        """
        try:
            # Calculate sweep parameters
            span = self.config.stop_freq - self.config.start_freq
            step_size = self.config.sample_rate
            num_steps = int(np.ceil(span / step_size))
            
            # Initialize spectrum array
            num_bins = num_steps * self.config.fft_size
            frequencies = np.linspace(self.config.start_freq, 
                                     self.config.stop_freq, 
                                     num_bins)
            spectrum = np.zeros(num_bins)
            
            logger.info(f"Sweeping {span/1e6:.1f} MHz in {num_steps} steps...")
            
            # Sweep across frequency range
            for step in range(num_steps):
                center_freq = self.config.start_freq + step * step_size
                
                # Configure hardware for this step
                self.hw.configure_hardware({
                    'frequency': int(center_freq),
                    'sample_rate': self.config.sample_rate,
                    'bandwidth': self.config.sample_rate,
                    'rx_gain': 40,
                    'tx_gain': 0
                })
                
                # Receive samples
                samples = self.hw.receive_samples(self.config.fft_size * self.config.averaging)
                
                if samples is None:
                    continue
                
                # Process with FFT
                power_spectrum = self._compute_power_spectrum(samples)
                
                # Store in full spectrum
                start_idx = step * self.config.fft_size
                end_idx = start_idx + self.config.fft_size
                if end_idx <= len(spectrum):
                    spectrum[start_idx:end_idx] = power_spectrum
            
            # Store in history
            self.spectrum_history.append({
                'frequencies': frequencies,
                'spectrum': spectrum,
                'timestamp': datetime.now()
            })
            
            return spectrum
            
        except Exception as e:
            logger.error(f"Sweep error: {e}")
            return np.array([])
    
    def _compute_power_spectrum(self, samples: np.ndarray) -> np.ndarray:
        """Compute power spectrum from IQ samples"""
        # Reshape for averaging
        num_ffts = len(samples) // self.config.fft_size
        samples_reshaped = samples[:num_ffts * self.config.fft_size].reshape(
            num_ffts, self.config.fft_size
        )
        
        # Apply window function (Hann)
        window = np.hanning(self.config.fft_size)
        samples_windowed = samples_reshaped * window
        
        # Compute FFT
        fft_result = np.fft.fftshift(np.fft.fft(samples_windowed, axis=1), axes=1)
        
        # Compute power
        power = np.abs(fft_result) ** 2
        
        # Apply detector
        if self.config.detector == "peak":
            power_spectrum = np.max(power, axis=0)
        elif self.config.detector == "average":
            power_spectrum = np.mean(power, axis=0)
        else:  # sample
            power_spectrum = power[0]
        
        # Convert to dBm
        power_dbm = 10 * np.log10(power_spectrum + 1e-12)
        
        return power_dbm
    
    def continuous_sweep(self, duration: float = 10.0, callback=None) -> bool:
        """
        Continuous spectrum sweeping
        
        Args:
            duration: Duration in seconds
            callback: Optional callback function for each sweep
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Continuous sweep for {duration}s")
            self.is_running = True
            
            start_time = datetime.now()
            sweep_count = 0
            
            while self.is_running:
                # Check duration
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= duration:
                    break
                
                # Perform sweep
                spectrum = self.sweep()
                sweep_count += 1
                
                # Call callback if provided
                if callback and len(spectrum) > 0:
                    callback(spectrum, self.spectrum_history[-1]['frequencies'])
                
                logger.debug(f"Sweep {sweep_count} complete")
            
            logger.info(f"Completed {sweep_count} sweeps")
            return True
            
        except Exception as e:
            logger.error(f"Continuous sweep error: {e}")
            return False
    
    def detect_signals(self, threshold_db: float = -60.0) -> List[Signal]:
        """
        Detect signals in spectrum
        
        Args:
            threshold_db: Detection threshold in dBm
            
        Returns:
            List of detected signals
        """
        try:
            if not self.spectrum_history:
                logger.warning("No spectrum data available")
                return []
            
            # Get latest spectrum
            latest = self.spectrum_history[-1]
            frequencies = latest['frequencies']
            spectrum = latest['spectrum']
            
            # Find peaks above threshold
            peaks = self._find_peaks(spectrum, threshold_db)
            
            signals = []
            for peak_idx in peaks:
                freq = int(frequencies[peak_idx])
                power = spectrum[peak_idx]
                
                # Estimate bandwidth
                bandwidth = self._estimate_bandwidth(spectrum, peak_idx)
                
                # Classify modulation
                modulation = self._classify_modulation(spectrum, peak_idx, bandwidth)
                
                signal = Signal(
                    frequency=freq,
                    power=power,
                    bandwidth=bandwidth,
                    modulation=modulation,
                    timestamp=datetime.now()
                )
                
                signals.append(signal)
                self.detected_signals[freq] = signal
            
            logger.info(f"Detected {len(signals)} signals")
            return signals
            
        except Exception as e:
            logger.error(f"Signal detection error: {e}")
            return []
    
    def _find_peaks(self, spectrum: np.ndarray, threshold: float) -> List[int]:
        """Find peaks in spectrum"""
        peaks = []
        
        # Simple peak detection
        for i in range(1, len(spectrum) - 1):
            if (spectrum[i] > threshold and 
                spectrum[i] > spectrum[i-1] and 
                spectrum[i] > spectrum[i+1]):
                peaks.append(i)
        
        # Merge close peaks
        if len(peaks) > 1:
            merged_peaks = [peaks[0]]
            for peak in peaks[1:]:
                if peak - merged_peaks[-1] > 10:  # Minimum separation
                    merged_peaks.append(peak)
            peaks = merged_peaks
        
        return peaks
    
    def _estimate_bandwidth(self, spectrum: np.ndarray, peak_idx: int) -> int:
        """Estimate signal bandwidth"""
        peak_power = spectrum[peak_idx]
        threshold = peak_power - 20  # -20 dB from peak
        
        # Find bandwidth at -20 dB
        left_idx = peak_idx
        while left_idx > 0 and spectrum[left_idx] > threshold:
            left_idx -= 1
        
        right_idx = peak_idx
        while right_idx < len(spectrum) - 1 and spectrum[right_idx] > threshold:
            right_idx += 1
        
        # Calculate bandwidth
        bin_width = self.config.sample_rate / self.config.fft_size
        bandwidth = int((right_idx - left_idx) * bin_width)
        
        return bandwidth
    
    def _classify_modulation(self, spectrum: np.ndarray, peak_idx: int, 
                            bandwidth: int) -> str:
        """Classify modulation type (simplified)"""
        # Simplified modulation classification based on bandwidth and shape
        
        if bandwidth < 10_000:
            return "Narrowband (AM/FM/CW)"
        elif bandwidth < 200_000:
            return "Voice (AM/FM/SSB)"
        elif bandwidth < 1_000_000:
            return "Digital (FSK/PSK)"
        elif bandwidth < 10_000_000:
            return "Wideband (WiFi/Bluetooth)"
        else:
            return "Wideband (LTE/5G/Video)"
    
    def waterfall_data(self, num_sweeps: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get waterfall display data
        
        Args:
            num_sweeps: Number of recent sweeps to include
            
        Returns:
            Tuple of (frequencies, waterfall_matrix)
        """
        try:
            if not self.spectrum_history:
                return np.array([]), np.array([[]])
            
            # Get recent sweeps
            recent_sweeps = list(self.spectrum_history)[-num_sweeps:]
            
            if not recent_sweeps:
                return np.array([]), np.array([[]])
            
            # Build waterfall matrix
            frequencies = recent_sweeps[0]['frequencies']
            waterfall = np.array([sweep['spectrum'] for sweep in recent_sweeps])
            
            return frequencies, waterfall
            
        except Exception as e:
            logger.error(f"Waterfall data error: {e}")
            return np.array([]), np.array([[]])
    
    def track_signal(self, target_frequency: int, duration: float = 10.0) -> Dict:
        """
        Track specific signal over time
        
        Args:
            target_frequency: Frequency to track (Hz)
            duration: Tracking duration (seconds)
            
        Returns:
            Tracking data (time series of power)
        """
        try:
            logger.info(f"Tracking signal at {target_frequency/1e6:.1f} MHz")
            
            # Configure for target frequency
            self.hw.configure_hardware({
                'frequency': target_frequency,
                'sample_rate': self.config.sample_rate,
                'bandwidth': self.config.sample_rate,
                'rx_gain': 40,
                'tx_gain': 0
            })
            
            # Track over time
            timestamps = []
            power_values = []
            
            start_time = datetime.now()
            while (datetime.now() - start_time).total_seconds() < duration:
                # Receive samples
                samples = self.hw.receive_samples(self.config.fft_size)
                
                if samples is None:
                    continue
                
                # Compute power
                power = np.mean(np.abs(samples) ** 2)
                power_dbm = 10 * np.log10(power + 1e-12)
                
                timestamps.append(datetime.now())
                power_values.append(power_dbm)
            
            tracking_data = {
                'frequency': target_frequency,
                'timestamps': timestamps,
                'power': power_values,
                'duration': duration
            }
            
            logger.info(f"Tracked signal: {len(power_values)} samples")
            return tracking_data
            
        except Exception as e:
            logger.error(f"Signal tracking error: {e}")
            return {}
    
    def measure_occupancy(self, threshold_db: float = -80.0) -> Dict[str, float]:
        """
        Measure spectrum occupancy
        
        Args:
            threshold_db: Occupancy threshold in dBm
            
        Returns:
            Occupancy statistics
        """
        try:
            if not self.spectrum_history:
                return {}
            
            latest = self.spectrum_history[-1]
            spectrum = latest['spectrum']
            
            # Calculate occupancy
            total_bins = len(spectrum)
            occupied_bins = np.sum(spectrum > threshold_db)
            occupancy_percent = (occupied_bins / total_bins) * 100
            
            # Calculate band-specific occupancy
            band_occupancy = {}
            
            # Common bands
            bands = {
                'VHF': (30e6, 300e6),
                'UHF': (300e6, 3e9),
                'WiFi_2.4GHz': (2.4e9, 2.5e9),
                'WiFi_5GHz': (5.15e9, 5.85e9),
            }
            
            frequencies = latest['frequencies']
            for band_name, (start, stop) in bands.items():
                mask = (frequencies >= start) & (frequencies <= stop)
                if np.any(mask):
                    band_spectrum = spectrum[mask]
                    band_occupied = np.sum(band_spectrum > threshold_db)
                    band_total = len(band_spectrum)
                    band_occupancy[band_name] = (band_occupied / band_total) * 100
            
            occupancy_stats = {
                'overall_occupancy': occupancy_percent,
                'occupied_bins': int(occupied_bins),
                'total_bins': total_bins,
                'band_occupancy': band_occupancy,
                'timestamp': datetime.now().isoformat()
            }
            
            return occupancy_stats
            
        except Exception as e:
            logger.error(f"Occupancy measurement error: {e}")
            return {}
    
    def find_quiet_frequency(self, bandwidth: int = 1_000_000) -> Optional[int]:
        """
        Find quiet frequency with specified bandwidth
        
        Args:
            bandwidth: Required bandwidth in Hz
            
        Returns:
            Quiet frequency in Hz or None
        """
        try:
            if not self.spectrum_history:
                return None
            
            latest = self.spectrum_history[-1]
            frequencies = latest['frequencies']
            spectrum = latest['spectrum']
            
            # Find quietest region
            bin_width = (frequencies[1] - frequencies[0]) if len(frequencies) > 1 else 1
            bins_needed = int(bandwidth / bin_width)
            
            min_power = float('inf')
            quiet_idx = 0
            
            for i in range(len(spectrum) - bins_needed):
                avg_power = np.mean(spectrum[i:i+bins_needed])
                if avg_power < min_power:
                    min_power = avg_power
                    quiet_idx = i + bins_needed // 2
            
            quiet_freq = int(frequencies[quiet_idx])
            logger.info(f"Quiet frequency found: {quiet_freq/1e6:.1f} MHz "
                       f"(power: {min_power:.1f} dBm)")
            
            return quiet_freq
            
        except Exception as e:
            logger.error(f"Find quiet frequency error: {e}")
            return None
    
    def get_spectrum_history(self) -> List[Dict]:
        """Get spectrum history"""
        return list(self.spectrum_history)
    
    def get_detected_signals(self) -> Dict[int, Signal]:
        """Get all detected signals"""
        return self.detected_signals
    
    def stop(self):
        """Stop spectrum analyzer"""
        self.is_running = False
        logger.info("Spectrum analyzer stopped")

def main():
    """Test spectrum analyzer"""
    from core.hardware import HardwareController
    
    # Initialize hardware
    hw = HardwareController()
    if not hw.connect():
        print("Failed to connect to BladeRF")
        return
    
    # Create spectrum analyzer
    sa = SpectrumAnalyzer(hw)
    
    # Configure
    config = SpectrumConfig(
        start_freq=2_400_000_000,  # 2.4 GHz
        stop_freq=2_500_000_000,   # 2.5 GHz
        fft_size=2048,
        averaging=10
    )
    
    if not sa.configure(config):
        print("Configuration failed")
        return
    
    print("RF Arsenal OS - Spectrum Analyzer")
    print("=" * 50)
    
    # Perform sweep
    print("\nPerforming spectrum sweep...")
    spectrum = sa.sweep()
    
    if len(spectrum) > 0:
        print(f"Sweep complete: {len(spectrum)} bins")
        print(f"Max power: {np.max(spectrum):.1f} dBm")
        print(f"Min power: {np.min(spectrum):.1f} dBm")
        
        # Detect signals
        print("\nDetecting signals...")
        signals = sa.detect_signals(threshold_db=-60.0)
        
        print(f"\nDetected {len(signals)} signal(s):")
        for sig in signals:
            print(f"  {sig.frequency/1e6:.3f} MHz: {sig.power:.1f} dBm, "
                  f"BW: {sig.bandwidth/1e3:.1f} kHz, {sig.modulation}")
        
        # Measure occupancy
        print("\nMeasuring spectrum occupancy...")
        occupancy = sa.measure_occupancy()
        print(f"Overall occupancy: {occupancy.get('overall_occupancy', 0):.1f}%")
        
        # Find quiet frequency
        print("\nFinding quiet frequency...")
        quiet_freq = sa.find_quiet_frequency(bandwidth=1_000_000)
        if quiet_freq:
            print(f"Quiet frequency: {quiet_freq/1e6:.1f} MHz")
    
    sa.stop()
    hw.disconnect()

if __name__ == "__main__":
    main()
