#!/usr/bin/env python3
"""
RF Arsenal OS - Satellite Communications Module
Hardware: BladeRF 2.0 micro xA9
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)

@dataclass
class SatelliteConfig:
    """Satellite Configuration"""
    frequency: int = 137_500_000  # 137.5 MHz (NOAA weather satellites)
    sample_rate: int = 2_048_000  # 2.048 MSPS
    bandwidth: int = 40_000       # 40 kHz
    mode: str = "rx"              # rx, tx, tracking

@dataclass
class Satellite:
    """Satellite Information"""
    name: str
    norad_id: int
    frequency: int
    modulation: str
    type: str  # weather, communication, military, amateur
    elevation: float
    azimuth: float
    doppler: float
    timestamp: datetime

@dataclass
class SatellitePass:
    """Satellite Pass Information"""
    satellite: str
    aos_time: datetime  # Acquisition of Signal
    los_time: datetime  # Loss of Signal
    max_elevation: float
    duration: float

class SatelliteCommunications:
    """Satellite Communications and Tracking System"""
    
    # Common satellite frequencies (MHz)
    SATELLITE_FREQUENCIES = {
        # Weather satellites (NOAA APT)
        'noaa_15': 137_620_000,
        'noaa_18': 137_912_500,
        'noaa_19': 137_100_000,
        
        # Weather satellites (Meteor)
        'meteor_m2': 137_100_000,
        
        # Amateur satellites
        'iss': 145_800_000,  # ISS voice
        'iss_sstv': 145_800_000,
        'ao_73': 145_960_000,
        
        # Iridium
        'iridium': 1_621_000_000,
        
        # GPS L1
        'gps_l1': 1_575_420_000,
        
        # Inmarsat
        'inmarsat': 1_545_000_000,
    }
    
    # Satellite types and modulations
    SATELLITE_INFO = {
        'noaa': {'type': 'weather', 'modulation': 'APT', 'bandwidth': 40_000},
        'meteor': {'type': 'weather', 'modulation': 'LRPT', 'bandwidth': 120_000},
        'iss': {'type': 'amateur', 'modulation': 'FM', 'bandwidth': 12_500},
        'iridium': {'type': 'communication', 'modulation': 'QPSK', 'bandwidth': 30_000},
    }
    
    def __init__(self, hardware_controller):
        """
        Initialize satellite communications system
        
        Args:
            hardware_controller: BladeRF hardware controller instance
        """
        self.hw = hardware_controller
        self.config = SatelliteConfig()
        self.is_running = False
        self.tracked_satellites: Dict[str, Satellite] = {}
        self.received_data: List[bytes] = []
        
        # Observer location (default to equator)
        self.observer_lat = 0.0
        self.observer_lon = 0.0
        self.observer_alt = 0.0
        
    def configure(self, config: SatelliteConfig) -> bool:
        """Configure satellite communications"""
        try:
            self.config = config
            
            # Configure BladeRF
            if not self.hw.configure_hardware({
                'frequency': config.frequency,
                'sample_rate': config.sample_rate,
                'bandwidth': config.bandwidth,
                'rx_gain': 40,
                'tx_gain': 20
            }):
                logger.error("Failed to configure hardware")
                return False
                
            logger.info(f"Satellite configured: {config.frequency/1e6:.3f} MHz, "
                       f"Mode: {config.mode}")
            return True
            
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            return False
    
    def set_observer_location(self, latitude: float, longitude: float, 
                             altitude: float = 0.0):
        """
        Set observer location for satellite tracking
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            altitude: Altitude in meters
        """
        self.observer_lat = latitude
        self.observer_lon = longitude
        self.observer_alt = altitude
        logger.info(f"Observer location: {latitude:.4f}, {longitude:.4f}, {altitude:.0f}m")
    
    def track_satellite(self, satellite_name: str, duration: float = 600.0) -> bool:
        """
        Track satellite and compensate for Doppler shift
        
        Args:
            satellite_name: Satellite name
            duration: Tracking duration in seconds
            
        Returns:
            True if successful
        """
        try:
            if satellite_name not in self.SATELLITE_FREQUENCIES:
                logger.error(f"Unknown satellite: {satellite_name}")
                return False
            
            base_freq = self.SATELLITE_FREQUENCIES[satellite_name]
            logger.info(f"Tracking {satellite_name} at {base_freq/1e6:.3f} MHz")
            
            self.is_running = True
            start_time = datetime.now()
            
            while self.is_running:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= duration:
                    break
                
                # Calculate satellite position (simplified)
                elevation, azimuth = self._calculate_satellite_position(
                    satellite_name, datetime.now()
                )
                
                # Calculate Doppler shift
                doppler_shift = self._calculate_doppler_shift(
                    base_freq, elevation, azimuth
                )
                
                # Adjust receiver frequency
                rx_freq = int(base_freq + doppler_shift)
                
                if rx_freq != self.config.frequency:
                    self.config.frequency = rx_freq
                    self.configure(self.config)
                    logger.debug(f"Doppler compensation: {doppler_shift/1e3:.1f} kHz")
                
                # Receive data
                samples = self.hw.receive_samples(
                    int(self.config.sample_rate * 1.0)  # 1 second
                )
                
                if samples is not None:
                    # Decode satellite data
                    decoded = self._decode_satellite_data(samples, satellite_name)
                    if decoded:
                        self.received_data.append(decoded)
                
                # Update tracking info
                self.tracked_satellites[satellite_name] = Satellite(
                    name=satellite_name,
                    norad_id=0,  # Would lookup from TLE
                    frequency=rx_freq,
                    modulation=self.SATELLITE_INFO.get(
                        satellite_name.split('_')[0], {}
                    ).get('modulation', 'Unknown'),
                    type=self.SATELLITE_INFO.get(
                        satellite_name.split('_')[0], {}
                    ).get('type', 'Unknown'),
                    elevation=elevation,
                    azimuth=azimuth,
                    doppler=doppler_shift,
                    timestamp=datetime.now()
                )
            
            logger.info(f"Tracking complete: {len(self.received_data)} packets received")
            return True
            
        except Exception as e:
            logger.error(f"Tracking error: {e}")
            return False
    
    def _calculate_satellite_position(self, satellite_name: str, 
                                     time: datetime) -> Tuple[float, float]:
        """Calculate satellite elevation and azimuth (simplified)"""
        # Simplified satellite position calculation
        # In production, use SGP4/SDP4 with TLE data
        
        # Simulate satellite pass (circular orbit approximation)
        minutes_since_epoch = (time - datetime(2024, 1, 1)).total_seconds() / 60
        
        # LEO satellite orbital period ~90 minutes
        orbital_phase = (minutes_since_epoch % 90) / 90 * 360  # degrees
        
        # Calculate elevation (0-90 degrees)
        elevation = abs(math.sin(math.radians(orbital_phase)) * 90)
        
        # Calculate azimuth (0-360 degrees)
        azimuth = (orbital_phase + self.observer_lon) % 360
        
        return elevation, azimuth
    
    def _calculate_doppler_shift(self, base_freq: int, elevation: float,
                                 azimuth: float) -> float:
        """Calculate Doppler shift based on satellite position"""
        # Simplified Doppler calculation
        # In production, use precise orbital velocity calculations
        
        # LEO satellite velocity ~7.8 km/s
        satellite_velocity = 7800  # m/s
        
        # Component of velocity towards observer
        radial_velocity = satellite_velocity * math.cos(math.radians(elevation))
        
        # Doppler shift formula: Δf = (v/c) * f
        c = 299_792_458  # speed of light (m/s)
        doppler = (radial_velocity / c) * base_freq
        
        # Account for whether satellite is approaching or receding
        if elevation < 45:  # Rising
            doppler = -doppler
        
        return doppler
    
    def _decode_satellite_data(self, samples: np.ndarray, 
                              satellite_name: str) -> Optional[bytes]:
        """Decode satellite transmission"""
        sat_type = satellite_name.split('_')[0]
        
        if sat_type == 'noaa':
            return self._decode_apt(samples)
        elif sat_type == 'meteor':
            return self._decode_lrpt(samples)
        elif sat_type == 'iss':
            return self._decode_fm_voice(samples)
        else:
            return None
    
    def _decode_apt(self, samples: np.ndarray) -> Optional[bytes]:
        """Decode NOAA APT (Automatic Picture Transmission)"""
        # APT is AM modulated at 2400 Hz carrier
        
        # AM demodulation
        envelope = np.abs(samples)
        
        # Resample to APT rate (4160 samples per line)
        apt_rate = 4160 * 2  # 2 lines per second
        
        # Normalize
        apt_signal = envelope[:apt_rate]
        apt_signal = (apt_signal / np.max(apt_signal) * 255).astype(np.uint8)
        
        return apt_signal.tobytes()
    
    def _decode_lrpt(self, samples: np.ndarray) -> Optional[bytes]:
        """Decode Meteor LRPT (Low Rate Picture Transmission)"""
        # LRPT uses QPSK modulation
        
        # QPSK demodulation (simplified)
        phase = np.angle(samples)
        
        # Decode to bits (simplified)
        symbols = ((phase + np.pi) / (np.pi / 2)).astype(int) % 4
        
        # Convert symbols to bytes
        bits = []
        for symbol in symbols:
            bits.extend([int(b) for b in format(symbol, '02b')])
        
        # Pack to bytes
        num_bytes = len(bits) // 8
        data = np.packbits(bits[:num_bytes*8])
        
        return data.tobytes()
    
    def _decode_fm_voice(self, samples: np.ndarray) -> Optional[bytes]:
        """Decode FM voice transmission"""
        # FM demodulation
        phase = np.angle(samples)
        phase_diff = np.diff(phase)
        phase_diff = np.unwrap(phase_diff)
        
        # Convert to audio
        audio = (phase_diff / np.pi * 127 + 127).astype(np.uint8)
        
        return audio.tobytes()
    
    def receive_weather_satellite(self, satellite_name: str) -> Optional[bytes]:
        """
        Receive weather satellite image
        
        Args:
            satellite_name: Weather satellite name (e.g., 'noaa_19')
            
        Returns:
            Image data or None
        """
        try:
            if satellite_name not in self.SATELLITE_FREQUENCIES:
                logger.error(f"Unknown satellite: {satellite_name}")
                return None
            
            logger.info(f"Receiving from {satellite_name}...")
            
            # Track satellite for full pass (~15 minutes)
            self.track_satellite(satellite_name, duration=900.0)
            
            # Combine received data
            if self.received_data:
                full_image = b''.join(self.received_data)
                logger.info(f"Received {len(full_image)} bytes")
                return full_image
            
            return None
            
        except Exception as e:
            logger.error(f"Reception error: {e}")
            return None
    
    def predict_passes(self, satellite_name: str, hours: int = 24) -> List[SatellitePass]:
        """
        Predict satellite passes
        
        Args:
            satellite_name: Satellite name
            hours: Prediction window in hours
            
        Returns:
            List of satellite passes
        """
        try:
            logger.info(f"Predicting passes for {satellite_name} ({hours}h)")
            
            passes = []
            current_time = datetime.now()
            end_time = current_time + timedelta(hours=hours)
            
            # Simplified pass prediction
            # In production, use SGP4/SDP4 with TLE data
            
            # LEO satellites pass every ~90 minutes
            orbital_period = 90  # minutes
            
            check_time = current_time
            while check_time < end_time:
                # Calculate satellite position
                elevation, azimuth = self._calculate_satellite_position(
                    satellite_name, check_time
                )
                
                # Check if satellite is visible (elevation > 10°)
                if elevation > 10:
                    # Find AOS (Acquisition of Signal)
                    aos_time = check_time
                    
                    # Find maximum elevation
                    max_elevation = elevation
                    max_time = check_time
                    
                    # Find LOS (Loss of Signal)
                    los_time = check_time
                    duration = 0
                    
                    # Simulate pass
                    while elevation > 10 and duration < 20:  # Max 20 minutes
                        check_time += timedelta(minutes=1)
                        duration += 1
                        elevation, _ = self._calculate_satellite_position(
                            satellite_name, check_time
                        )
                        
                        if elevation > max_elevation:
                            max_elevation = elevation
                            max_time = check_time
                    
                    los_time = check_time
                    
                    # Create pass entry
                    satellite_pass = SatellitePass(
                        satellite=satellite_name,
                        aos_time=aos_time,
                        los_time=los_time,
                        max_elevation=max_elevation,
                        duration=duration
                    )
                    passes.append(satellite_pass)
                
                # Jump to next orbital period
                check_time += timedelta(minutes=orbital_period)
            
            logger.info(f"Found {len(passes)} passes")
            return passes
            
        except Exception as e:
            logger.error(f"Pass prediction error: {e}")
            return []
    
    def transmit_to_satellite(self, satellite_name: str, data: bytes) -> bool:
        """
        Transmit to satellite (amateur satellites only)
        
        Args:
            satellite_name: Satellite name
            data: Data to transmit
            
        Returns:
            True if successful
        """
        try:
            if satellite_name not in self.SATELLITE_FREQUENCIES:
                logger.error(f"Unknown satellite: {satellite_name}")
                return False
            
            # Check if amateur satellite
            sat_type = self.SATELLITE_INFO.get(satellite_name.split('_')[0], {})
            if sat_type.get('type') != 'amateur':
                logger.error(f"Transmission only allowed to amateur satellites")
                return False
            
            logger.info(f"Transmitting to {satellite_name}")
            
            # Configure for uplink frequency
            uplink_freq = self.SATELLITE_FREQUENCIES[satellite_name]
            self.config.frequency = uplink_freq
            self.config.mode = 'tx'
            self.configure(self.config)
            
            # Modulate data (FM)
            modulated = self._modulate_fm(data)
            
            # Transmit
            if self.hw.transmit_burst(modulated):
                logger.info("Transmission complete")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Transmission error: {e}")
            return False
    
    def _modulate_fm(self, data: bytes) -> np.ndarray:
        """Modulate data for FM transmission"""
        samples_per_byte = 128
        signal = np.zeros(len(data) * samples_per_byte, dtype=np.complex64)
        
        for i, byte in enumerate(data):
            start = i * samples_per_byte
            end = start + samples_per_byte
            
            # FM modulation (frequency deviation based on byte value)
            deviation = (byte - 128) * 1000  # Hz
            t = np.linspace(0, samples_per_byte / self.config.sample_rate,
                          samples_per_byte, endpoint=False)
            signal[start:end] = np.exp(2j * np.pi * deviation * t)
        
        signal *= 0.3
        return signal
    
    def get_tracked_satellites(self) -> Dict[str, Satellite]:
        """Get all tracked satellites"""
        return self.tracked_satellites
    
    def get_received_data(self) -> List[bytes]:
        """Get all received data"""
        return self.received_data
    
    def stop(self):
        """Stop satellite operations"""
        self.is_running = False
        self.hw.stop_transmission()
        logger.info("Satellite operations stopped")

def main():
    """Test satellite communications"""
    from core.hardware import HardwareController
    
    # Initialize hardware
    hw = HardwareController()
    if not hw.connect():
        print("Failed to connect to BladeRF")
        return
    
    # Create satellite system
    satcom = SatelliteCommunications(hw)
    
    # Set observer location (example: New York)
    satcom.set_observer_location(40.7128, -74.0060, 10.0)
    
    print("RF Arsenal OS - Satellite Communications")
    print("=" * 50)
    
    # Predict passes
    print("\nPredicting NOAA-19 passes (24h)...")
    passes = satcom.predict_passes('noaa_19', hours=24)
    
    print(f"\nFound {len(passes)} passes:")
    for i, pass_info in enumerate(passes[:5], 1):
        print(f"{i}. AOS: {pass_info.aos_time.strftime('%Y-%m-%d %H:%M')}, "
              f"Max El: {pass_info.max_elevation:.1f}°, "
              f"Duration: {pass_info.duration:.0f} min")
    
    # Configure for NOAA-19
    config = SatelliteConfig(
        frequency=137_100_000,  # NOAA-19
        mode="rx"
    )
    
    if satcom.configure(config):
        print("\nConfigured for NOAA-19 reception")
        print("Ready to track satellite...")
    
    satcom.stop()
    hw.disconnect()

if __name__ == "__main__":
    main()
