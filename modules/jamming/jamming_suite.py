#!/usr/bin/env python3
"""
RF Arsenal OS - Electronic Warfare Jamming Suite
Hardware: BladeRF 2.0 micro xA9
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class JammingConfig:
    """Jamming Configuration"""
    frequency: int = 2_400_000_000  # Center frequency
    sample_rate: int = 40_000_000   # 40 MSPS
    bandwidth: int = 40_000_000     # 40 MHz
    tx_power: int = 30              # dBm (maximum safe power)
    mode: str = "noise"             # noise, tone, sweep, pulse, barrage

class JammingSuite:
    """Multi-band Electronic Warfare Jamming Suite"""
    
    # Common frequency bands for jamming
    BANDS = {
        'cellular_2g': (850_000_000, 1_900_000_000),
        'cellular_3g': (1_900_000_000, 2_100_000_000),
        'cellular_4g': (700_000_000, 2_600_000_000),
        'cellular_5g': (3_500_000_000, 3_800_000_000),
        'wifi_2.4': (2_400_000_000, 2_483_500_000),
        'wifi_5': (5_150_000_000, 5_850_000_000),
        'gps_l1': (1_575_420_000, 1_575_420_000),
        'gps_l2': (1_227_600_000, 1_227_600_000),
        'bluetooth': (2_400_000_000, 2_483_500_000),
        'zigbee': (2_405_000_000, 2_480_000_000),
        'drone': (2_400_000_000, 5_850_000_000),
        'radio_vhf': (30_000_000, 300_000_000),
        'radio_uhf': (300_000_000, 3_000_000_000),
    }
    
    def __init__(self, hardware_controller):
        """
        Initialize jamming suite
        
        Args:
            hardware_controller: BladeRF hardware controller instance
        """
        self.hw = hardware_controller
        self.config = JammingConfig()
        self.is_running = False
        self.active_jammers: Dict[str, Dict] = {}
        
    def configure(self, config: JammingConfig) -> bool:
        """Configure jamming parameters"""
        try:
            self.config = config
            
            # Configure BladeRF
            if not self.hw.configure_hardware({
                'frequency': config.frequency,
                'sample_rate': config.sample_rate,
                'bandwidth': config.bandwidth,
                'tx_gain': config.tx_power,
                'rx_gain': 0  # Not needed for jamming
            }):
                logger.error("Failed to configure hardware")
                return False
                
            logger.info(f"Jammer configured: {config.frequency/1e6:.1f} MHz, "
                       f"BW={config.bandwidth/1e6:.0f} MHz, "
                       f"Mode: {config.mode}, Power: {config.tx_power} dBm")
            return True
            
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            return False
    
    def jam_band(self, band_name: str, mode: str = "noise") -> bool:
        """
        Jam a specific frequency band
        
        Args:
            band_name: Band name from BANDS dictionary
            mode: Jamming mode (noise, tone, sweep, pulse, barrage)
            
        Returns:
            True if successful
        """
        try:
            if band_name not in self.BANDS:
                logger.error(f"Unknown band: {band_name}")
                return False
            
            freq_start, freq_end = self.BANDS[band_name]
            center_freq = (freq_start + freq_end) // 2
            bandwidth = freq_end - freq_start
            
            logger.info(f"Jamming {band_name}: {freq_start/1e6:.1f}-{freq_end/1e6:.1f} MHz")
            
            # Configure for band
            self.config.frequency = center_freq
            self.config.bandwidth = min(bandwidth, 56_000_000)  # BladeRF limit
            self.config.mode = mode
            
            if not self.configure(self.config):
                return False
            
            # Generate jamming signal
            jamming_signal = self._generate_jamming_signal()
            
            # Transmit
            if self.hw.transmit_continuous(jamming_signal):
                self.is_running = True
                self.active_jammers[band_name] = {
                    'frequency': center_freq,
                    'bandwidth': bandwidth,
                    'mode': mode,
                    'started': datetime.now().isoformat()
                }
                logger.info(f"Jammer active: {band_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Jamming error: {e}")
            return False
    
    def _generate_jamming_signal(self) -> np.ndarray:
        """Generate jamming signal based on mode"""
        duration = 0.01  # 10ms
        num_samples = int(self.config.sample_rate * duration)
        
        if self.config.mode == "noise":
            return self._generate_noise_jamming(num_samples)
        elif self.config.mode == "tone":
            return self._generate_tone_jamming(num_samples)
        elif self.config.mode == "sweep":
            return self._generate_sweep_jamming(num_samples)
        elif self.config.mode == "pulse":
            return self._generate_pulse_jamming(num_samples)
        elif self.config.mode == "barrage":
            return self._generate_barrage_jamming(num_samples)
        else:
            logger.warning(f"Unknown mode: {self.config.mode}, using noise")
            return self._generate_noise_jamming(num_samples)
    
    def _generate_noise_jamming(self, num_samples: int) -> np.ndarray:
        """Generate white noise jamming (most effective general purpose)"""
        # Complex white Gaussian noise
        noise = (np.random.randn(num_samples) + 
                1j * np.random.randn(num_samples)) / np.sqrt(2)
        
        # Apply band-limiting filter (simplified)
        noise *= 0.7  # Power
        
        return noise
    
    def _generate_tone_jamming(self, num_samples: int) -> np.ndarray:
        """Generate single tone jamming (narrow band)"""
        t = np.linspace(0, num_samples / self.config.sample_rate, 
                       num_samples, endpoint=False)
        
        # Single tone at center
        tone = np.exp(2j * np.pi * 0 * t)  # Baseband tone
        tone *= 0.8
        
        return tone
    
    def _generate_sweep_jamming(self, num_samples: int) -> np.ndarray:
        """Generate frequency sweep jamming (effective against frequency hopping)"""
        duration = num_samples / self.config.sample_rate
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Linear frequency sweep across bandwidth
        f_start = -self.config.bandwidth / 2
        f_end = self.config.bandwidth / 2
        
        # Chirp signal
        instantaneous_freq = f_start + (f_end - f_start) * t / duration
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / self.config.sample_rate
        
        sweep = np.exp(1j * phase)
        sweep *= 0.7
        
        return sweep
    
    def _generate_pulse_jamming(self, num_samples: int) -> np.ndarray:
        """Generate pulse jamming (effective against radar)"""
        signal = np.zeros(num_samples, dtype=np.complex64)
        
        # Generate pulses
        pulse_width = num_samples // 10  # 10% duty cycle
        pulse_interval = num_samples // 5  # 5 pulses
        
        for i in range(5):
            start = i * pulse_interval
            end = start + pulse_width
            if end <= num_samples:
                # High power pulse
                signal[start:end] = 1.0 + 0j
        
        return signal
    
    def _generate_barrage_jamming(self, num_samples: int) -> np.ndarray:
        """Generate barrage jamming (multiple tones across bandwidth)"""
        signal = np.zeros(num_samples, dtype=np.complex64)
        t = np.linspace(0, num_samples / self.config.sample_rate, 
                       num_samples, endpoint=False)
        
        # Generate multiple tones across bandwidth
        num_tones = 20
        frequencies = np.linspace(-self.config.bandwidth / 2, 
                                 self.config.bandwidth / 2, 
                                 num_tones)
        
        for freq in frequencies:
            tone = np.exp(2j * np.pi * freq * t)
            signal += tone
        
        # Normalize
        signal /= num_tones
        signal *= 0.7
        
        return signal
    
    def jam_frequency(self, frequency: int, bandwidth: int = 20_000_000,
                     mode: str = "noise") -> bool:
        """
        Jam a specific frequency
        
        Args:
            frequency: Center frequency in Hz
            bandwidth: Bandwidth in Hz
            mode: Jamming mode
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Jamming {frequency/1e6:.1f} MHz (BW: {bandwidth/1e6:.0f} MHz)")
            
            self.config.frequency = frequency
            self.config.bandwidth = bandwidth
            self.config.mode = mode
            
            if not self.configure(self.config):
                return False
            
            jamming_signal = self._generate_jamming_signal()
            
            if self.hw.transmit_continuous(jamming_signal):
                self.is_running = True
                logger.info(f"Jammer active at {frequency/1e6:.1f} MHz")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Frequency jamming error: {e}")
            return False
    
    def adaptive_jamming(self, target_signals: List[Dict]) -> bool:
        """
        Adaptive jamming based on detected signals
        
        Args:
            target_signals: List of detected signals with frequency info
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Adaptive jamming: {len(target_signals)} targets")
            
            if not target_signals:
                return False
            
            # Analyze target signals
            frequencies = [s['frequency'] for s in target_signals]
            bandwidths = [s.get('bandwidth', 1_000_000) for s in target_signals]
            
            # Find optimal jamming frequency (center of mass)
            center_freq = int(np.mean(frequencies))
            max_bw = int(max(bandwidths))
            
            # Choose mode based on signal characteristics
            if len(target_signals) > 5:
                mode = "barrage"  # Multiple targets
            elif max_bw > 10_000_000:
                mode = "sweep"  # Wideband targets
            else:
                mode = "noise"  # General purpose
            
            logger.info(f"Adaptive mode: {mode}, Center: {center_freq/1e6:.1f} MHz")
            
            return self.jam_frequency(center_freq, max_bw * 2, mode)
            
        except Exception as e:
            logger.error(f"Adaptive jamming error: {e}")
            return False
    
    def reactive_jamming(self, listen_duration: float = 0.1) -> bool:
        """
        Reactive jamming (listen, then jam when signal detected)
        
        Args:
            listen_duration: Listen duration in seconds
            
        Returns:
            True if successful
        """
        try:
            logger.info("Reactive jamming mode")
            
            # Listen for signals
            samples = self.hw.receive_samples(
                int(self.config.sample_rate * listen_duration)
            )
            
            if samples is None:
                return False
            
            # Detect signal
            power = np.abs(samples) ** 2
            threshold = np.mean(power) + 3 * np.std(power)
            
            if np.any(power > threshold):
                # Signal detected, jam it
                logger.info("Signal detected, jamming...")
                
                # Analyze frequency
                fft = np.fft.fftshift(np.fft.fft(samples))
                power_spectrum = np.abs(fft) ** 2
                peak_idx = np.argmax(power_spectrum)
                
                # Calculate frequency offset
                freq_bins = np.fft.fftshift(np.fft.fftfreq(len(samples), 
                                                           1/self.config.sample_rate))
                detected_offset = freq_bins[peak_idx]
                
                # Jam at detected frequency
                jam_freq = int(self.config.frequency + detected_offset)
                return self.jam_frequency(jam_freq, mode="tone")
            else:
                logger.debug("No signal detected")
                return False
            
        except Exception as e:
            logger.error(f"Reactive jamming error: {e}")
            return False
    
    def multi_band_jamming(self, bands: List[str]) -> bool:
        """
        Jam multiple bands simultaneously (frequency hopping)
        
        Args:
            bands: List of band names to jam
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Multi-band jamming: {bands}")
            
            # Validate bands
            valid_bands = [b for b in bands if b in self.BANDS]
            if not valid_bands:
                logger.error("No valid bands specified")
                return False
            
            # Hop between bands
            for band in valid_bands:
                freq_start, freq_end = self.BANDS[band]
                center_freq = (freq_start + freq_end) // 2
                
                self.config.frequency = center_freq
                self.configure(self.config)
                
                # Jam for short duration
                jamming_signal = self._generate_jamming_signal()
                self.hw.transmit_burst(jamming_signal)
                
                logger.debug(f"Jammed {band}")
            
            self.is_running = True
            return True
            
        except Exception as e:
            logger.error(f"Multi-band jamming error: {e}")
            return False
    
    def protocol_specific_jamming(self, protocol: str) -> bool:
        """
        Protocol-specific jamming (optimized for target protocol)
        
        Args:
            protocol: Protocol name (wifi, bluetooth, zigbee, lte, etc.)
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Protocol-specific jamming: {protocol}")
            
            if protocol.lower() == "wifi":
                # Jam WiFi beacon frames
                return self.jam_band('wifi_2.4', mode="pulse")
            
            elif protocol.lower() == "bluetooth":
                # Jam Bluetooth frequency hopping
                return self.jam_band('bluetooth', mode="sweep")
            
            elif protocol.lower() == "zigbee":
                # Jam ZigBee channels
                return self.jam_band('zigbee', mode="barrage")
            
            elif protocol.lower() in ["lte", "4g"]:
                # Jam LTE control channels
                return self.jam_band('cellular_4g', mode="tone")
            
            elif protocol.lower() == "gps":
                # Jam GPS L1
                return self.jam_band('gps_l1', mode="noise")
            
            else:
                logger.error(f"Unknown protocol: {protocol}")
                return False
            
        except Exception as e:
            logger.error(f"Protocol jamming error: {e}")
            return False
    
    def get_active_jammers(self) -> Dict:
        """Get all active jammers"""
        return self.active_jammers
    
    def stop(self):
        """Stop all jamming operations"""
        self.is_running = False
        self.active_jammers.clear()
        self.hw.stop_transmission()
        logger.info("All jamming operations stopped")

def main():
    """Test jamming suite"""
    from core.hardware import HardwareController
    
    # Initialize hardware
    hw = HardwareController()
    if not hw.connect():
        print("Failed to connect to BladeRF")
        return
    
    # Create jamming suite
    jammer = JammingSuite(hw)
    
    # Configure
    config = JammingConfig(
        frequency=2_400_000_000,
        mode="noise"
    )
    
    if not jammer.configure(config):
        print("Configuration failed")
        return
    
    print("RF Arsenal OS - Jamming Suite Demo")
    print("=" * 50)
    
    # Demo: Jam WiFi (commented for safety)
    # print("\nJamming WiFi 2.4 GHz band...")
    # jammer.jam_band('wifi_2.4', mode='sweep')
    # time.sleep(5)
    
    # Demo: Protocol-specific jamming
    # print("\nProtocol-specific jamming: Bluetooth...")
    # jammer.protocol_specific_jamming('bluetooth')
    
    print("\nAvailable bands:")
    for band_name in jammer.BANDS.keys():
        print(f"  - {band_name}")
    
    print("\nJamming modes:")
    print("  - noise: White noise (general purpose)")
    print("  - tone: Single tone (narrow band)")
    print("  - sweep: Frequency sweep (frequency hopping)")
    print("  - pulse: Pulsed jamming (radar)")
    print("  - barrage: Multiple tones (wideband)")
    
    jammer.stop()
    hw.disconnect()

if __name__ == "__main__":
    main()
