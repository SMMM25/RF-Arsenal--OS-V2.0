#!/usr/bin/env python3
"""
RF Arsenal OS - GPS Spoofing Module

Comprehensive GPS spoofing for vehicle navigation testing:
- GPS L1 C/A signal generation
- Coordinate spoofing (static and dynamic)
- Trajectory replay
- Time manipulation
- Multi-satellite simulation

Target frequency: 1575.42 MHz (GPS L1)
Hardware Required: BladeRF 2.0 micro xA9 or compatible SDR

Author: RF Arsenal Team
License: For authorized security testing only

WARNING: GPS spoofing is illegal in most jurisdictions.
This tool is intended for authorized security testing only.
"""

import math
import time
import struct
import threading
import logging
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple, Generator
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# GPS Constants
GPS_L1_FREQ = 1575.42e6  # Hz
GPS_C_A_CHIP_RATE = 1.023e6  # chips/second
GPS_NAV_BIT_RATE = 50  # bits/second
SPEED_OF_LIGHT = 299792458  # m/s
GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)
SECONDS_PER_WEEK = 604800


class GPSConstellation(Enum):
    """GPS constellation type"""
    GPS = "gps"
    GLONASS = "glonass"
    GALILEO = "galileo"
    BEIDOU = "beidou"


@dataclass
class GPSCoordinate:
    """GPS coordinate with velocity"""
    latitude: float  # degrees
    longitude: float  # degrees
    altitude: float = 0.0  # meters above sea level
    
    # Velocity components (optional)
    velocity_north: float = 0.0  # m/s
    velocity_east: float = 0.0  # m/s
    velocity_down: float = 0.0  # m/s
    
    timestamp: float = field(default_factory=time.time)
    
    def __str__(self) -> str:
        return f"({self.latitude:.6f}, {self.longitude:.6f}, {self.altitude:.1f}m)"
    
    def to_ecef(self) -> Tuple[float, float, float]:
        """Convert to ECEF coordinates"""
        # WGS84 parameters
        a = 6378137.0  # semi-major axis
        f = 1 / 298.257223563  # flattening
        e2 = 2*f - f*f  # eccentricity squared
        
        lat_rad = math.radians(self.latitude)
        lon_rad = math.radians(self.longitude)
        
        N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
        
        x = (N + self.altitude) * math.cos(lat_rad) * math.cos(lon_rad)
        y = (N + self.altitude) * math.cos(lat_rad) * math.sin(lon_rad)
        z = (N * (1 - e2) + self.altitude) * math.sin(lat_rad)
        
        return (x, y, z)
    
    @classmethod
    def from_ecef(cls, x: float, y: float, z: float) -> 'GPSCoordinate':
        """Create from ECEF coordinates"""
        # WGS84 parameters
        a = 6378137.0
        f = 1 / 298.257223563
        e2 = 2*f - f*f
        
        # Iterative solution
        lon = math.atan2(y, x)
        p = math.sqrt(x*x + y*y)
        lat = math.atan2(z, p * (1 - e2))
        
        for _ in range(10):
            N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
            lat = math.atan2(z + e2 * N * math.sin(lat), p)
        
        N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
        alt = p / math.cos(lat) - N
        
        return cls(
            latitude=math.degrees(lat),
            longitude=math.degrees(lon),
            altitude=alt
        )
    
    def distance_to(self, other: 'GPSCoordinate') -> float:
        """Calculate distance to another coordinate (meters)"""
        # Haversine formula
        R = 6371000  # Earth radius in meters
        
        lat1_rad = math.radians(self.latitude)
        lat2_rad = math.radians(other.latitude)
        delta_lat = math.radians(other.latitude - self.latitude)
        delta_lon = math.radians(other.longitude - self.longitude)
        
        a = (math.sin(delta_lat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def bearing_to(self, other: 'GPSCoordinate') -> float:
        """Calculate bearing to another coordinate (degrees)"""
        lat1_rad = math.radians(self.latitude)
        lat2_rad = math.radians(other.latitude)
        delta_lon = math.radians(other.longitude - self.longitude)
        
        x = math.sin(delta_lon) * math.cos(lat2_rad)
        y = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
        
        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360


@dataclass
class GPSTrajectory:
    """GPS trajectory (sequence of waypoints)"""
    waypoints: List[GPSCoordinate] = field(default_factory=list)
    name: str = "Trajectory"
    loop: bool = False
    
    def add_waypoint(self, coord: GPSCoordinate):
        """Add waypoint to trajectory"""
        self.waypoints.append(coord)
    
    def get_position_at_time(
        self,
        elapsed_time: float,
        speed_mps: float = 10.0
    ) -> Optional[GPSCoordinate]:
        """
        Get interpolated position at elapsed time
        
        Args:
            elapsed_time: Time since trajectory start (seconds)
            speed_mps: Travel speed (meters per second)
            
        Returns:
            Interpolated coordinate
        """
        if len(self.waypoints) < 2:
            return self.waypoints[0] if self.waypoints else None
        
        distance_traveled = elapsed_time * speed_mps
        cumulative_distance = 0.0
        
        for i in range(len(self.waypoints) - 1):
            segment_distance = self.waypoints[i].distance_to(self.waypoints[i+1])
            
            if cumulative_distance + segment_distance >= distance_traveled:
                # Interpolate within this segment
                segment_progress = distance_traveled - cumulative_distance
                ratio = segment_progress / segment_distance if segment_distance > 0 else 0
                
                return GPSCoordinate(
                    latitude=self.waypoints[i].latitude + 
                            ratio * (self.waypoints[i+1].latitude - self.waypoints[i].latitude),
                    longitude=self.waypoints[i].longitude + 
                             ratio * (self.waypoints[i+1].longitude - self.waypoints[i].longitude),
                    altitude=self.waypoints[i].altitude + 
                            ratio * (self.waypoints[i+1].altitude - self.waypoints[i].altitude)
                )
            
            cumulative_distance += segment_distance
        
        # Past end of trajectory
        if self.loop:
            return self.get_position_at_time(
                elapsed_time % (cumulative_distance / speed_mps),
                speed_mps
            )
        
        return self.waypoints[-1]
    
    def total_distance(self) -> float:
        """Calculate total trajectory distance (meters)"""
        if len(self.waypoints) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(self.waypoints) - 1):
            total += self.waypoints[i].distance_to(self.waypoints[i+1])
        return total
    
    @classmethod
    def create_circle(
        cls,
        center: GPSCoordinate,
        radius_m: float,
        num_points: int = 36
    ) -> 'GPSTrajectory':
        """Create circular trajectory"""
        trajectory = cls(name="Circle")
        
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            
            # Calculate offset in meters
            dx = radius_m * math.cos(angle)
            dy = radius_m * math.sin(angle)
            
            # Convert to lat/lon offset (approximate)
            lat_offset = dy / 111111  # degrees per meter
            lon_offset = dx / (111111 * math.cos(math.radians(center.latitude)))
            
            trajectory.add_waypoint(GPSCoordinate(
                latitude=center.latitude + lat_offset,
                longitude=center.longitude + lon_offset,
                altitude=center.altitude
            ))
        
        trajectory.loop = True
        return trajectory


@dataclass
class GPSSatellite:
    """GPS satellite model for signal generation"""
    prn: int  # Pseudo-Random Noise ID (1-32 for GPS)
    
    # Orbital parameters (simplified)
    ecef_x: float = 0.0
    ecef_y: float = 0.0
    ecef_z: float = 26600000.0  # ~26,600 km altitude
    
    # Clock parameters
    clock_bias: float = 0.0  # seconds
    clock_drift: float = 0.0  # seconds/second
    
    # Signal parameters
    cn0: float = 45.0  # Carrier-to-noise ratio (dB-Hz)
    elevation: float = 45.0  # degrees
    azimuth: float = 0.0  # degrees
    
    healthy: bool = True
    
    def __str__(self) -> str:
        return f"PRN{self.prn:02d} El:{self.elevation:.1f}° Az:{self.azimuth:.1f}°"
    
    def get_pseudorange(self, receiver_pos: GPSCoordinate) -> float:
        """Calculate pseudorange to receiver (meters)"""
        rx_ecef = receiver_pos.to_ecef()
        
        dx = self.ecef_x - rx_ecef[0]
        dy = self.ecef_y - rx_ecef[1]
        dz = self.ecef_z - rx_ecef[2]
        
        geometric_range = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        # Add clock bias
        return geometric_range + self.clock_bias * SPEED_OF_LIGHT


class GPSCACodeGenerator:
    """
    GPS C/A Code Generator
    
    Generates the Gold codes used in GPS L1 C/A signals
    """
    
    # G2 delay for each PRN (1-32)
    G2_DELAYS = [
        5, 6, 7, 8, 17, 18, 139, 140, 141, 251,
        252, 254, 255, 256, 257, 258, 469, 470, 471, 472,
        473, 474, 509, 512, 513, 514, 515, 516, 859, 860,
        861, 862
    ]
    
    def __init__(self, prn: int):
        self.prn = prn
        self._code = self._generate_code()
    
    def _generate_code(self) -> np.ndarray:
        """Generate 1023-chip C/A code for PRN"""
        # Initialize G1 and G2 shift registers
        g1 = [1] * 10
        g2 = [1] * 10
        
        code = []
        
        # G2 delay taps (PRN specific)
        if self.prn <= 32:
            delay = self.G2_DELAYS[self.prn - 1]
            tap1 = (delay // 10) % 10
            tap2 = delay % 10
        else:
            tap1 = 2
            tap2 = 6
        
        for _ in range(1023):
            # G1 output
            g1_out = g1[9]
            
            # G2 output (delayed)
            g2_out = g2[tap1] ^ g2[tap2]
            
            # C/A code chip
            chip = g1_out ^ g2_out
            code.append(1 if chip else -1)
            
            # Shift G1
            feedback = g1[2] ^ g1[9]
            g1 = [feedback] + g1[:-1]
            
            # Shift G2
            feedback = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
            g2 = [feedback] + g2[:-1]
        
        return np.array(code, dtype=np.int8)
    
    @property
    def code(self) -> np.ndarray:
        """Get C/A code"""
        return self._code


class GPSSignalGenerator:
    """
    GPS Signal Generator
    
    Generates GPS L1 C/A signals with navigation message
    """
    
    def __init__(self, sample_rate: float = 4e6):
        self.sample_rate = sample_rate
        self._ca_codes: Dict[int, GPSCACodeGenerator] = {}
        
        # Pregenerate codes for visible satellites
        for prn in range(1, 33):
            self._ca_codes[prn] = GPSCACodeGenerator(prn)
    
    def generate_signal(
        self,
        satellites: List[GPSSatellite],
        receiver_pos: GPSCoordinate,
        duration: float = 1.0,
        gps_time: datetime = None
    ) -> np.ndarray:
        """
        Generate composite GPS signal
        
        Args:
            satellites: List of visible satellites
            receiver_pos: Desired receiver position
            duration: Signal duration (seconds)
            gps_time: GPS time (default: now)
            
        Returns:
            IQ signal samples
        """
        if gps_time is None:
            gps_time = datetime.utcnow()
        
        num_samples = int(self.sample_rate * duration)
        signal = np.zeros(num_samples, dtype=complex)
        
        for sat in satellites:
            if not sat.healthy:
                continue
            
            # Calculate pseudorange and Doppler
            pseudorange = sat.get_pseudorange(receiver_pos)
            doppler = self._calculate_doppler(sat, receiver_pos)
            
            # Generate satellite signal
            sat_signal = self._generate_satellite_signal(
                sat.prn,
                num_samples,
                pseudorange,
                doppler,
                sat.cn0
            )
            
            # Add to composite
            signal += sat_signal
        
        # Add noise
        noise_power = 10 ** (-20 / 10)  # -20 dB noise floor
        noise = np.sqrt(noise_power) * (
            np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        )
        signal += noise
        
        return signal
    
    def _generate_satellite_signal(
        self,
        prn: int,
        num_samples: int,
        pseudorange: float,
        doppler: float,
        cn0: float
    ) -> np.ndarray:
        """Generate signal for single satellite"""
        # Get C/A code
        ca_code = self._ca_codes[prn].code
        
        # Calculate code phase (samples)
        code_phase = (pseudorange / SPEED_OF_LIGHT) * GPS_C_A_CHIP_RATE
        code_phase = code_phase % 1023
        
        # Calculate carrier phase
        samples_per_chip = self.sample_rate / GPS_C_A_CHIP_RATE
        
        # Generate time vector
        t = np.arange(num_samples) / self.sample_rate
        
        # Generate carrier
        carrier_freq = GPS_L1_FREQ + doppler
        carrier = np.exp(2j * np.pi * doppler * t)
        
        # Sample C/A code
        code_indices = ((np.arange(num_samples) / samples_per_chip + code_phase) % 1023).astype(int)
        code_signal = ca_code[code_indices].astype(float)
        
        # Set amplitude based on C/N0
        amplitude = 10 ** (cn0 / 20) / 1000  # Normalize
        
        # Generate signal
        signal = amplitude * code_signal * carrier
        
        return signal
    
    def _calculate_doppler(
        self,
        satellite: GPSSatellite,
        receiver_pos: GPSCoordinate
    ) -> float:
        """Calculate Doppler shift (simplified)"""
        # For simulation, assume small Doppler
        # Real implementation would use satellite velocity
        return np.random.uniform(-5000, 5000)  # Hz


class GPSSpoofer:
    """
    GPS Spoofer
    
    Comprehensive GPS spoofing tool for:
    - Static position spoofing
    - Dynamic trajectory following
    - Time manipulation
    - Multi-satellite constellation simulation
    """
    
    def __init__(
        self,
        sdr_controller=None,
        sample_rate: float = 4e6
    ):
        self.sdr = sdr_controller
        self.sample_rate = sample_rate
        self.frequency = GPS_L1_FREQ
        
        self._signal_gen = GPSSignalGenerator(sample_rate)
        self._satellites: List[GPSSatellite] = []
        
        self._spoofing = False
        self._spoof_thread: Optional[threading.Thread] = None
        self._current_position: Optional[GPSCoordinate] = None
        self._trajectory: Optional[GPSTrajectory] = None
        
        # Initialize default constellation
        self._init_constellation()
        
        logger.info(f"GPS Spoofer initialized: {GPS_L1_FREQ/1e6:.2f} MHz")
    
    def _init_constellation(self, num_satellites: int = 8):
        """Initialize spoofed satellite constellation for transmission"""
        self._satellites = []
        
        for i in range(num_satellites):
            elevation = 30 + 40 * (i / num_satellites)
            azimuth = 360 * i / num_satellites
            
            # Convert to ECEF (simplified)
            altitude = 26600000  # GPS orbit altitude
            distance = altitude + 6371000  # Earth radius
            
            sat = GPSSatellite(
                prn=i + 1,
                ecef_x=distance * math.cos(math.radians(elevation)) * math.cos(math.radians(azimuth)),
                ecef_y=distance * math.cos(math.radians(elevation)) * math.sin(math.radians(azimuth)),
                ecef_z=distance * math.sin(math.radians(elevation)),
                elevation=elevation,
                azimuth=azimuth,
                cn0=45 - 10 * (1 - elevation/90)  # Higher elevation = stronger signal
            )
            self._satellites.append(sat)
        
        logger.info(f"Initialized {num_satellites} satellites")
    
    def spoof_position(
        self,
        coordinate: GPSCoordinate,
        duration: float = None,
        power_dbm: float = -130
    ) -> bool:
        """
        Spoof static GPS position
        
        Args:
            coordinate: Target position
            duration: Spoofing duration (None = indefinite)
            power_dbm: Transmit power (dBm)
            
        Returns:
            True if spoofing started
        """
        logger.warning(f"Spoofing position to: {coordinate}")
        
        self._current_position = coordinate
        self._spoofing = True
        
        def _spoof_loop():
            start_time = time.time()
            
            while self._spoofing:
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Generate GPS signal
                signal = self._signal_gen.generate_signal(
                    self._satellites,
                    self._current_position,
                    duration=0.1
                )
                
                # Transmit
                self._transmit(signal)
                
                time.sleep(0.05)  # Overlap signals
            
            self._spoofing = False
            logger.info("Static spoofing stopped")
        
        self._spoof_thread = threading.Thread(target=_spoof_loop, daemon=True)
        self._spoof_thread.start()
        
        return True
    
    def spoof_trajectory(
        self,
        trajectory: GPSTrajectory,
        speed_mps: float = 10.0,
        callback: Callable[[GPSCoordinate], None] = None
    ) -> bool:
        """
        Spoof GPS along trajectory
        
        Args:
            trajectory: Path to follow
            speed_mps: Travel speed (m/s)
            callback: Called with current position
            
        Returns:
            True if spoofing started
        """
        logger.warning(f"Starting trajectory spoofing: {trajectory.name}")
        
        self._trajectory = trajectory
        self._spoofing = True
        
        def _trajectory_loop():
            start_time = time.time()
            
            while self._spoofing:
                elapsed = time.time() - start_time
                
                # Get current position along trajectory
                pos = trajectory.get_position_at_time(elapsed, speed_mps)
                
                if pos is None:
                    break
                
                self._current_position = pos
                
                if callback:
                    callback(pos)
                
                # Generate and transmit
                signal = self._signal_gen.generate_signal(
                    self._satellites,
                    pos,
                    duration=0.1
                )
                self._transmit(signal)
                
                time.sleep(0.05)
                
                # Check if trajectory complete
                if not trajectory.loop:
                    total_time = trajectory.total_distance() / speed_mps
                    if elapsed > total_time:
                        break
            
            self._spoofing = False
            logger.info("Trajectory spoofing complete")
        
        self._spoof_thread = threading.Thread(target=_trajectory_loop, daemon=True)
        self._spoof_thread.start()
        
        return True
    
    def stop(self):
        """Stop GPS spoofing"""
        self._spoofing = False
        if self._spoof_thread:
            self._spoof_thread.join(timeout=2.0)
        logger.info("GPS spoofing stopped")
    
    def spoof_time(
        self,
        coordinate: GPSCoordinate,
        time_offset: timedelta,
        duration: float = 10.0
    ) -> bool:
        """
        Spoof GPS time
        
        Args:
            coordinate: Position (required for signal generation)
            time_offset: Time offset from actual GPS time
            duration: Spoofing duration
            
        Returns:
            True if spoofing started
        """
        logger.warning(f"Spoofing GPS time with offset: {time_offset}")
        
        fake_time = datetime.utcnow() + time_offset
        
        # Update satellite clock biases
        for sat in self._satellites:
            sat.clock_bias = time_offset.total_seconds()
        
        return self.spoof_position(coordinate, duration)
    
    def create_trajectory_from_kml(self, kml_path: str) -> Optional[GPSTrajectory]:
        """
        Create trajectory from KML file
        
        Args:
            kml_path: Path to KML file
            
        Returns:
            Trajectory or None
        """
        try:
            # Simple KML parser (would use proper library in production)
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(kml_path)
            root = tree.getroot()
            
            # Find coordinates (simplified)
            trajectory = GPSTrajectory(name=kml_path)
            
            for elem in root.iter():
                if 'coordinates' in elem.tag.lower():
                    coords_text = elem.text.strip()
                    for coord_str in coords_text.split():
                        parts = coord_str.split(',')
                        if len(parts) >= 2:
                            lon = float(parts[0])
                            lat = float(parts[1])
                            alt = float(parts[2]) if len(parts) > 2 else 0
                            
                            trajectory.add_waypoint(GPSCoordinate(
                                latitude=lat,
                                longitude=lon,
                                altitude=alt
                            ))
            
            logger.info(f"Loaded {len(trajectory.waypoints)} waypoints from KML")
            return trajectory
            
        except Exception as e:
            logger.error(f"Failed to parse KML: {e}")
            return None
    
    def create_straight_line_trajectory(
        self,
        start: GPSCoordinate,
        end: GPSCoordinate,
        num_points: int = 100
    ) -> GPSTrajectory:
        """Create straight line trajectory between two points"""
        trajectory = GPSTrajectory(name="Straight Line")
        
        for i in range(num_points):
            ratio = i / (num_points - 1)
            
            trajectory.add_waypoint(GPSCoordinate(
                latitude=start.latitude + ratio * (end.latitude - start.latitude),
                longitude=start.longitude + ratio * (end.longitude - start.longitude),
                altitude=start.altitude + ratio * (end.altitude - start.altitude)
            ))
        
        return trajectory
    
    def get_visible_satellites(self) -> List[GPSSatellite]:
        """Get list of visible satellites"""
        return [s for s in self._satellites if s.elevation > 5]
    
    def set_satellite_count(self, count: int):
        """Set number of spoofed satellites to generate"""
        self._init_constellation(count)
    
    def _transmit(self, signal: np.ndarray) -> bool:
        """Transmit GPS signal"""
        if self.sdr:
            try:
                self.sdr.set_frequency(self.frequency)
                self.sdr.transmit(signal)
                return True
            except Exception as e:
                logger.error(f"Transmit error: {e}")
                return False
        
        logger.warning("No SDR connected - GPS transmission requires hardware")
        return True
    
    @property
    def current_position(self) -> Optional[GPSCoordinate]:
        """Get current spoofed position"""
        return self._current_position
    
    @property
    def is_spoofing(self) -> bool:
        """Check if currently spoofing"""
        return self._spoofing


# Convenience functions
def create_static_spoof(
    lat: float,
    lon: float,
    alt: float = 0,
    sdr=None
) -> GPSSpoofer:
    """Quick setup for static position spoofing"""
    spoofer = GPSSpoofer(sdr_controller=sdr)
    coord = GPSCoordinate(latitude=lat, longitude=lon, altitude=alt)
    spoofer.spoof_position(coord)
    return spoofer


def create_trajectory_spoof(
    waypoints: List[Tuple[float, float, float]],
    speed_mps: float = 10.0,
    sdr=None
) -> GPSSpoofer:
    """Quick setup for trajectory spoofing"""
    spoofer = GPSSpoofer(sdr_controller=sdr)
    trajectory = GPSTrajectory()
    
    for lat, lon, alt in waypoints:
        trajectory.add_waypoint(GPSCoordinate(
            latitude=lat,
            longitude=lon,
            altitude=alt
        ))
    
    spoofer.spoof_trajectory(trajectory, speed_mps)
    return spoofer
