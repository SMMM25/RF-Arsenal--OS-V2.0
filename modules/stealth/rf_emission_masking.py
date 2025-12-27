#!/usr/bin/env python3
"""
RF Emission Masking - Physical Layer Stealth
Disguise RF transmissions as legitimate signals
"""

import numpy as np
import time
import secrets
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Legitimate signal types to mimic"""
    WIFI_BEACON = "wifi_beacon"
    CELLULAR_HEARTBEAT = "cellular_heartbeat"
    GPS_SATELLITE = "gps_satellite"
    BLUETOOTH_ADVERTISING = "bluetooth_advertising"
    ZIGBEE_MESH = "zigbee_mesh"


@dataclass
class EmissionProfile:
    """RF emission characteristics"""
    power_range: Tuple[float, float]  # dBm min/max
    duty_cycle: float  # 0.0 to 1.0
    burst_duration_ms: float
    hop_interval_ms: float
    jitter_ms: float


class RFEmissionMasker:
    """
    Disguise RF transmissions as legitimate signals
    Prevents direction finding and signal identification
    """
    
    def __init__(self, hardware_controller):
        self.hardware = hardware_controller
        self.active_profile = None
        self.power_cycling_enabled = False
        self.frequency_hopping_enabled = False
        
        # Legitimate signal profiles
        self.profiles = {
            SignalType.WIFI_BEACON: EmissionProfile(
                power_range=(10, 20),
                duty_cycle=0.01,  # Beacons are brief
                burst_duration_ms=2.5,
                hop_interval_ms=102.4,  # Standard beacon interval
                jitter_ms=5.0
            ),
            SignalType.CELLULAR_HEARTBEAT: EmissionProfile(
                power_range=(15, 23),
                duty_cycle=0.05,
                burst_duration_ms=10,
                hop_interval_ms=480,  # 4G heartbeat
                jitter_ms=50
            ),
            SignalType.BLUETOOTH_ADVERTISING: EmissionProfile(
                power_range=(0, 4),
                duty_cycle=0.02,
                burst_duration_ms=1.0,
                hop_interval_ms=100,  # BLE advertising interval
                jitter_ms=10
            )
        }
        
        # Frequency hopping state
        self.current_frequency = None
        self.hop_sequence = []
        self.hop_index = 0
        
    def enable_legitimate_signal_mimicry(self, signal_type: SignalType):
        """
        Mimic legitimate commercial signals
        Clone timing patterns and power characteristics
        """
        self.active_profile = self.profiles[signal_type]
        print(f"[STEALTH] Mimicking {signal_type.value} signal patterns")
        
        if signal_type == SignalType.WIFI_BEACON:
            return self._mimic_wifi_beacon()
        elif signal_type == SignalType.CELLULAR_HEARTBEAT:
            return self._mimic_cellular_heartbeat()
        elif signal_type == SignalType.BLUETOOTH_ADVERTISING:
            return self._mimic_bluetooth_advertising()
            
    def _mimic_wifi_beacon(self) -> Dict:
        """Clone WiFi beacon timing patterns from real APs
        
        SECURITY: Uses secrets.SystemRandom for cryptographically secure
        randomization to prevent RF fingerprinting attacks.
        """
        csprng = secrets.SystemRandom()
        return {
            'interval_ms': 102.4,  # Standard beacon interval
            'jitter_ms': csprng.uniform(-5, 5),
            'power_dbm': csprng.uniform(10, 20),
            'duration_ms': 2.5,
            'pattern': 'periodic_burst'
        }
        
    def _mimic_cellular_heartbeat(self) -> Dict:
        """Replicate cellular tower heartbeat signatures
        
        SECURITY: Uses secrets.SystemRandom for CSPRNG.
        """
        csprng = secrets.SystemRandom()
        return {
            'interval_ms': 480,  # 4G paging cycle
            'jitter_ms': csprng.uniform(-50, 50),
            'power_dbm': csprng.uniform(15, 23),
            'duration_ms': 10,
            'pattern': 'heartbeat_pulse'
        }
        
    def _mimic_bluetooth_advertising(self) -> Dict:
        """Emulate Bluetooth device advertising patterns
        
        SECURITY: Uses secrets module for CSPRNG.
        """
        csprng = secrets.SystemRandom()
        return {
            'interval_ms': secrets.choice([20, 100, 152.5, 211.25, 318.75]),
            'jitter_ms': csprng.uniform(-10, 10),
            'power_dbm': csprng.uniform(0, 4),
            'duration_ms': 1.0,
            'pattern': 'ble_advertising'
        }
        
    def enable_power_cycling(self, enabled: bool = True):
        """
        Randomized power level variation
        Prevents fixed signature detection
        """
        self.power_cycling_enabled = enabled
        if enabled:
            print("[STEALTH] Power cycling enabled - varying TX power")
            
    def get_next_power_level(self) -> float:
        """
        Calculate next cryptographically randomized power level
        Uses micro-burst transmission and intentional jitter
        
        SECURITY: Uses secrets.SystemRandom for CSPRNG to prevent
        power signature prediction attacks.
        """
        if not self.power_cycling_enabled or not self.active_profile:
            return 20.0  # Default power
            
        min_power, max_power = self.active_profile.power_range
        
        csprng = secrets.SystemRandom()
        # Cryptographically randomized power within profile range
        base_power = csprng.uniform(min_power, max_power)
        
        # Add intentional jitter (±2 dB)
        jitter = csprng.uniform(-2, 2)
        
        return max(min_power, min(max_power, base_power + jitter))
        
    def calculate_burst_timing(self) -> Tuple[float, float]:
        """
        Calculate micro-burst transmission timing
        Returns (duration_ms, interval_ms)
        """
        if not self.active_profile:
            return (10.0, 100.0)
            
        # Base timing from profile
        duration = self.active_profile.burst_duration_ms
        interval = self.active_profile.hop_interval_ms
        
        csprng = secrets.SystemRandom()
        # Add timing jitter to prevent pattern recognition
        duration += csprng.uniform(-self.active_profile.jitter_ms/10, 
                                   self.active_profile.jitter_ms/10)
        interval += csprng.uniform(-self.active_profile.jitter_ms, 
                                  self.active_profile.jitter_ms)
        
        return (max(0.1, duration), max(1.0, interval))
        
    def enable_frequency_agility(self, enabled: bool = True):
        """
        Rapid frequency hopping
        Changes frequency every 10-100ms
        """
        self.frequency_hopping_enabled = enabled
        if enabled:
            print("[STEALTH] Frequency agility enabled - rapid hopping")
            self._generate_hop_sequence()
            
    def _generate_hop_sequence(self):
        """
        Generate pseudo-random frequency hopping pattern
        Avoids sequential patterns, syncs with legitimate usage
        """
        # Define frequency bands (example: 2.4 GHz WiFi channels)
        wifi_channels = [2412, 2417, 2422, 2427, 2432, 2437, 2442, 2447, 
                        2452, 2457, 2462, 2467, 2472]
        
        # Create cryptographically randomized hopping sequence
        csprng = secrets.SystemRandom()
        self.hop_sequence = csprng.sample(wifi_channels * 10, 100)
        self.hop_index = 0
        
    def get_next_frequency(self) -> float:
        """
        Get next frequency in hopping sequence
        Returns frequency in MHz
        """
        if not self.frequency_hopping_enabled or not self.hop_sequence:
            return 2412.0  # Default channel 1
            
        freq = self.hop_sequence[self.hop_index]
        self.hop_index = (self.hop_index + 1) % len(self.hop_sequence)
        return float(freq)
        
    def calculate_hop_interval(self) -> float:
        """
        Calculate next hop interval (10-100ms)
        Cryptographically randomized to prevent pattern detection
        """
        csprng = secrets.SystemRandom()
        return csprng.uniform(10, 100)
        
    def apply_spread_spectrum(self, signal: np.ndarray, 
                             spreading_factor: int = 8) -> np.ndarray:
        """
        Apply spread-spectrum technique to signal
        Distributes energy across wider bandwidth
        """
        # Generate pseudo-random spreading code
        spreading_code = np.random.choice([-1, 1], size=len(signal) * spreading_factor)
        
        # Repeat signal to match spreading code length
        repeated_signal = np.repeat(signal, spreading_factor)
        
        # Apply spreading
        spread_signal = repeated_signal * spreading_code
        
        return spread_signal
        
    def sync_with_legitimate_traffic(self) -> Dict:
        """
        Sync with legitimate frequency usage peaks
        Transmit when channel is naturally busy
        """
        # Simulate channel usage detection
        channel_busy_times = [
            # Peak hours for different services
            {'start': '08:00', 'end': '10:00', 'type': 'cellular_morning'},
            {'start': '12:00', 'end': '13:00', 'type': 'wifi_lunch'},
            {'start': '17:00', 'end': '19:00', 'type': 'cellular_evening'},
            {'start': '20:00', 'end': '23:00', 'type': 'wifi_evening'}
        ]
        
        current_hour = time.localtime().tm_hour
        
        for period in channel_busy_times:
            start_hour = int(period['start'].split(':')[0])
            end_hour = int(period['end'].split(':')[0])
            
            if start_hour <= current_hour < end_hour:
                return {
                    'recommend_transmit': True,
                    'channel_busy': True,
                    'traffic_type': period['type'],
                    'cover_level': 'high'
                }
                
        return {
            'recommend_transmit': False,
            'channel_busy': False,
            'traffic_type': 'low_traffic',
            'cover_level': 'low'
        }
        
    def get_stealth_transmission_params(self) -> Dict:
        """
        Get complete set of stealth transmission parameters
        Combines all masking techniques
        """
        power = self.get_next_power_level()
        frequency = self.get_next_frequency()
        burst_duration, burst_interval = self.calculate_burst_timing()
        hop_interval = self.calculate_hop_interval()
        traffic_sync = self.sync_with_legitimate_traffic()
        
        return {
            'power_dbm': power,
            'frequency_mhz': frequency,
            'burst_duration_ms': burst_duration,
            'burst_interval_ms': burst_interval,
            'hop_interval_ms': hop_interval,
            'traffic_sync': traffic_sync,
            'mimicry_active': self.active_profile is not None,
            'power_cycling': self.power_cycling_enabled,
            'frequency_hopping': self.frequency_hopping_enabled
        }


class HardwareFingerprint:
    """
    Mask unique hardware identifiers
    Prevent RF fingerprinting and device identification
    """
    
    def __init__(self, hardware_controller):
        self.hardware = hardware_controller
        self.spoofing_enabled = False
        
    def enable_clock_skew_randomization(self, enabled: bool = True):
        """
        Randomize timing characteristics
        Prevents clock skew fingerprinting
        """
        self.spoofing_enabled = enabled
        if enabled:
            print("[STEALTH] Clock skew randomization enabled")
            
    def add_timing_offset(self, base_time_ns: int) -> int:
        """
        Add variable timing offset to transmissions
        Mimics different device manufacturers
        
        SECURITY: Uses CSPRNG to prevent timing fingerprinting.
        """
        if not self.spoofing_enabled:
            return base_time_ns
            
        # Different manufacturers have different clock accuracies
        # Typical range: ±20 ppm to ±100 ppm
        csprng = secrets.SystemRandom()
        offset_ppm = csprng.uniform(-100, 100)
        offset_ns = int(base_time_ns * offset_ppm / 1e6)
        
        return base_time_ns + offset_ns
        
    def spoof_iq_imbalance(self) -> Tuple[float, float]:
        """
        Adjust I/Q imbalance to mimic target devices
        Returns (amplitude_imbalance_db, phase_imbalance_deg)
        
        SECURITY: Uses CSPRNG to prevent IQ fingerprinting.
        """
        # Typical I/Q imbalance for different device types
        device_profiles = {
            'high_end_sdr': (0.1, 0.5),    # Very low imbalance
            'commercial_wifi': (0.5, 2.0),  # Moderate imbalance
            'cheap_iot': (1.5, 5.0)        # Higher imbalance
        }
        
        csprng = secrets.SystemRandom()
        profile = secrets.choice(list(device_profiles.values()))
        amplitude_db = csprng.uniform(0, profile[0])
        phase_deg = csprng.uniform(-profile[1], profile[1])
        
        return (amplitude_db, phase_deg)
        
    def generate_frequency_error(self) -> float:
        """
        Match frequency error patterns of target devices
        Returns frequency offset in Hz
        """
        # Different device types have different frequency errors
        # Cheap oscillators: ±20 ppm
        # Good oscillators: ±2 ppm
        # TCXO: ±1 ppm
        
        csprng = secrets.SystemRandom()
        ppm_error = csprng.uniform(-20, 20)
        center_freq_hz = 2.4e9  # Example: 2.4 GHz
        frequency_error_hz = center_freq_hz * ppm_error / 1e6
        
        return frequency_error_hz
        
    def spoof_dc_offset(self) -> Tuple[float, float]:
        """
        Clone DC offset signatures
        Returns (I_offset, Q_offset) as fraction of full scale
        
        SECURITY: Uses CSPRNG to prevent DC offset fingerprinting.
        """
        # Typical DC offset: 0.5% to 5% of full scale
        csprng = secrets.SystemRandom()
        i_offset = csprng.uniform(-0.05, 0.05)
        q_offset = csprng.uniform(-0.05, 0.05)
        
        return (i_offset, q_offset)
        
    def generate_phase_noise_profile(self, offset_hz: np.ndarray) -> np.ndarray:
        """
        Replicate phase noise characteristics of target device
        offset_hz: array of frequency offsets from carrier
        Returns: phase noise in dBc/Hz
        """
        # Simplified phase noise model
        # Real devices have specific profiles
        
        phase_noise = np.zeros_like(offset_hz)
        
        # 1/f^3 region (close to carrier)
        mask_1f3 = offset_hz < 1000
        phase_noise[mask_1f3] = -60 - 30 * np.log10(offset_hz[mask_1f3] / 100)
        
        # 1/f^2 region (mid offset)
        mask_1f2 = (offset_hz >= 1000) & (offset_hz < 100000)
        phase_noise[mask_1f2] = -90 - 20 * np.log10(offset_hz[mask_1f2] / 1000)
        
        # Noise floor (far from carrier)
        mask_floor = offset_hz >= 100000
        phase_noise[mask_floor] = -140
        
        # Add random variation to mimic different devices
        phase_noise += random.uniform(-5, 5)
        
        return phase_noise
        
    def get_hardware_spoof_params(self) -> Dict:
        """
        Get complete set of hardware fingerprint spoofing parameters
        """
        iq_imbalance = self.spoof_iq_imbalance()
        freq_error = self.generate_frequency_error()
        dc_offset = self.spoof_dc_offset()
        
        return {
            'timing_offset_ns': random.randint(-1000, 1000),
            'iq_amplitude_imbalance_db': iq_imbalance[0],
            'iq_phase_imbalance_deg': iq_imbalance[1],
            'frequency_error_hz': freq_error,
            'dc_offset_i': dc_offset[0],
            'dc_offset_q': dc_offset[1],
            'spoofing_enabled': self.spoofing_enabled
        }


# Example usage and testing
if __name__ == "__main__":
    # Hardware stub for demo - requires real hardware in production
    class HardwareStub:
        """Demo stub - replace with real hardware controller"""
        pass
    
    hw = HardwareStub()
    
    # Test RF emission masking
    print("=== RF Emission Masking Test ===")
    masker = RFEmissionMasker(hw)
    
    masker.enable_legitimate_signal_mimicry(SignalType.WIFI_BEACON)
    masker.enable_power_cycling(True)
    masker.enable_frequency_agility(True)
    
    # Get stealth transmission parameters
    for i in range(5):
        params = masker.get_stealth_transmission_params()
        print(f"\nTransmission {i+1}:")
        print(f"  Power: {params['power_dbm']:.2f} dBm")
        print(f"  Frequency: {params['frequency_mhz']:.1f} MHz")
        print(f"  Burst: {params['burst_duration_ms']:.2f} ms")
        print(f"  Interval: {params['burst_interval_ms']:.2f} ms")
        
    # Test hardware fingerprint spoofing
    print("\n=== Hardware Fingerprint Spoofing Test ===")
    fingerprint = HardwareFingerprint(hw)
    fingerprint.enable_clock_skew_randomization(True)
    
    for i in range(3):
        params = fingerprint.get_hardware_spoof_params()
        print(f"\nDevice Profile {i+1}:")
        print(f"  I/Q Amplitude Imbalance: {params['iq_amplitude_imbalance_db']:.3f} dB")
        print(f"  I/Q Phase Imbalance: {params['iq_phase_imbalance_deg']:.3f}°")
        print(f"  Frequency Error: {params['frequency_error_hz']:.1f} Hz")
        print(f"  DC Offset I: {params['dc_offset_i']:.4f}")
        print(f"  DC Offset Q: {params['dc_offset_q']:.4f}")
