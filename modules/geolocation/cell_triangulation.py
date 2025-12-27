#!/usr/bin/env python3
"""
RF Arsenal OS - Cellular Geolocation & Triangulation
Precise location tracking via cellular signals
"""

import logging
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os

from core.anonymization import get_anonymizer

logger = logging.getLogger(__name__)


@dataclass
class CellMeasurement:
    """Single cellular signal measurement"""
    timestamp: float
    cell_id: str
    lac: int  # Location Area Code
    mcc: int  # Mobile Country Code
    mnc: int  # Mobile Network Code
    rssi: float  # Signal strength in dBm
    ta: Optional[int] = None  # Timing Advance (GSM)
    latitude: Optional[float] = None  # Capture location
    longitude: Optional[float] = None
    altitude: Optional[float] = None


@dataclass
class Position:
    """Calculated position with accuracy"""
    latitude: float
    longitude: float
    altitude: float
    accuracy: float  # meters
    timestamp: float
    method: str  # 'timing_advance', 'rssi_triangulation', 'cell_id'
    confidence: float  # 0.0 - 1.0


@dataclass
class MovementTrack:
    """Movement tracking data"""
    imsi: str  # 🔐 Anonymized IMSI hash
    positions: List[Position]
    total_distance: float  # meters
    average_speed: float  # m/s
    max_speed: float  # m/s
    duration: float  # seconds


class CellularGeolocation:
    """
    Precise location tracking via cellular signals
    
    Methods:
    1. Timing Advance (TA) - Distance from tower (GSM)
    2. RSSI triangulation - Multiple capture points
    3. Cell ID → GPS mapping - Public databases
    """
    
    def __init__(self, anonymize_identifiers: bool = True):
        """Initialize geolocation engine
        
        Args:
            anonymize_identifiers: Automatically anonymize IMSI (default: True)
        """
        self.cell_database = {}  # Cell ID → GPS mapping
        self.measurements_cache = {}  # IMSI_hash → measurements
        self.tracking_sessions = {}  # IMSI_hash → MovementTrack
        
        # 🔐 Centralized anonymization
        self.anonymize_identifiers = anonymize_identifiers
        self.anonymizer = get_anonymizer()
        
        # Speed of light constant for TA calculations
        self.SPEED_OF_LIGHT = 299792458  # m/s
        
        # GSM timing advance parameters
        self.TA_BIT_PERIOD = 3.69e-6  # seconds (1 bit period)
        self.TA_DISTANCE_PER_UNIT = self.SPEED_OF_LIGHT * self.TA_BIT_PERIOD  # ~1113 meters
        
        logger.info(f"Cellular geolocation engine initialized (Anonymization: {anonymize_identifiers})")
    
    def load_cell_database(self, database_path: str) -> bool:
        """
        Load cell tower database
        
        Format: Cell ID, LAC, MCC, MNC, Lat, Lon, Range
        Source: OpenCellID, Mozilla Location Services
        
        Args:
            database_path: Path to cell database JSON/CSV
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(database_path, 'r') as f:
                data = json.load(f)
                self.cell_database = data
                logger.info(f"Loaded {len(self.cell_database)} cell towers from database")
                return True
        except Exception as e:
            logger.error(f"Failed to load cell database: {e}")
            return False
    
    def add_measurement(self, imsi: str, measurement: CellMeasurement):
        """
        Add a signal measurement for target
        
        Args:
            imsi: Target IMSI (will be anonymized automatically)
            measurement: Cellular measurement data
        """
        # 🔐 SECURITY FIX: Always anonymize IMSI before storage
        imsi_hash = self.anonymizer.anonymize_imsi(imsi) if self.anonymize_identifiers else imsi
        
        if imsi_hash not in self.measurements_cache:
            self.measurements_cache[imsi_hash] = []
        
        self.measurements_cache[imsi_hash].append(measurement)
        
        # Keep only last 100 measurements per IMSI
        if len(self.measurements_cache[imsi_hash]) > 100:
            self.measurements_cache[imsi_hash] = self.measurements_cache[imsi_hash][-100:]
        
        logger.debug(f"Added measurement for {imsi_hash}: Cell {measurement.cell_id}, RSSI {measurement.rssi} dBm")
    
    def calculate_position_timing_advance(self, 
                                         cell_id: str,
                                         timing_advance: int) -> Optional[Position]:
        """
        Calculate position using Timing Advance (GSM only)
        
        TA indicates distance from base station:
        Distance = TA × 550 meters (approximately)
        
        Args:
            cell_id: Cell tower ID
            timing_advance: TA value (0-63 for GSM)
            
        Returns:
            Position with circular area around tower
        """
        # Look up cell tower location
        cell_info = self.cell_database.get(cell_id)
        if not cell_info:
            logger.warning(f"Cell {cell_id} not in database")
            return None
        
        # Calculate distance from tower
        # TA ranges 0-63, each unit = 550m approximately
        distance = timing_advance * 550  # meters
        
        # Accuracy is the TA resolution (±550m)
        accuracy = 550.0
        
        position = Position(
            latitude=cell_info['lat'],
            longitude=cell_info['lon'],
            altitude=cell_info.get('altitude', 0),
            accuracy=accuracy,
            timestamp=time.time(),
            method='timing_advance',
            confidence=0.6  # Medium confidence
        )
        
        logger.info(f"TA position: ~{distance:.0f}m from tower {cell_id}")
        return position
    
    def calculate_position_rssi_triangulation(self,
                                             measurements: List[CellMeasurement]) -> Optional[Position]:
        """
        Calculate position using RSSI triangulation
        
        Requires measurements from 3+ different locations
        Uses signal strength to estimate distance, then triangulates
        
        Args:
            measurements: List of measurements with known capture locations
            
        Returns:
            Triangulated position
        """
        # Need at least 3 measurements with known positions
        valid_measurements = [m for m in measurements 
                            if m.latitude is not None and m.longitude is not None]
        
        if len(valid_measurements) < 3:
            logger.warning(f"Need 3+ measurements for triangulation, got {len(valid_measurements)}")
            return None
        
        # Calculate distances from RSSI using path loss model
        positions = []
        weights = []
        
        for m in valid_measurements:
            # Estimate distance from RSSI (rough approximation)
            # Assuming typical cellular TX power of 33 dBm
            tx_power = 33  # dBm
            path_loss = tx_power - m.rssi
            
            # Distance in meters (simplified model)
            estimated_distance = 10 ** (path_loss / 20.0)
            
            positions.append((m.latitude, m.longitude))
            
            # Weight by signal strength (stronger signal = more reliable)
            weight = 1.0 / (1.0 + abs(m.rssi))
            weights.append(weight)
        
        # Weighted average of positions
        weights = np.array(weights)
        weights /= weights.sum()
        
        lat = sum(p[0] * w for p, w in zip(positions, weights))
        lon = sum(p[1] * w for p, w in zip(positions, weights))
        
        # Calculate accuracy estimate (std deviation of positions)
        lat_std = np.std([p[0] for p in positions])
        lon_std = np.std([p[1] for p in positions])
        accuracy = np.sqrt(lat_std**2 + lon_std**2) * 111320  # Convert degrees to meters
        
        position = Position(
            latitude=lat,
            longitude=lon,
            altitude=0,
            accuracy=max(accuracy, 100),  # Minimum 100m accuracy
            timestamp=time.time(),
            method='rssi_triangulation',
            confidence=min(0.8, len(valid_measurements) / 5.0)
        )
        
        logger.info(f"RSSI triangulation: {lat:.6f}, {lon:.6f} (±{accuracy:.0f}m)")
        return position
    
    def calculate_position_cell_id(self, cell_id: str) -> Optional[Position]:
        """
        Calculate position using Cell ID lookup
        
        Least accurate method - returns cell tower location
        
        Args:
            cell_id: Cell tower ID
            
        Returns:
            Position at cell tower location
        """
        cell_info = self.cell_database.get(cell_id)
        if not cell_info:
            logger.warning(f"Cell {cell_id} not in database")
            return None
        
        # Accuracy is the cell tower range
        accuracy = cell_info.get('range', 5000)  # Default 5km
        
        position = Position(
            latitude=cell_info['lat'],
            longitude=cell_info['lon'],
            altitude=cell_info.get('altitude', 0),
            accuracy=accuracy,
            timestamp=time.time(),
            method='cell_id',
            confidence=0.3  # Low confidence
        )
        
        logger.info(f"Cell ID position: {cell_info['lat']:.6f}, {cell_info['lon']:.6f} (±{accuracy:.0f}m)")
        return position
    
    def calculate_position(self, 
                          imsi: str,
                          measurements: Optional[List[CellMeasurement]] = None) -> Optional[Position]:
        """
        Calculate target position using best available method
        
        Priority:
        1. Timing Advance (most accurate)
        2. RSSI triangulation
        3. Cell ID lookup
        
        Args:
            imsi: Target IMSI (will be anonymized automatically)
            measurements: Optional measurements (uses cache if None)
            
        Returns:
            Best position estimate
        """
        # 🔐 SECURITY FIX: Anonymize IMSI for cache lookup
        imsi_hash = self.anonymizer.anonymize_imsi(imsi) if self.anonymize_identifiers else imsi
        
        if measurements is None:
            measurements = self.measurements_cache.get(imsi_hash, [])
        
        if not measurements:
            logger.warning(f"No measurements for {imsi}")
            return None
        
        # Get most recent measurement
        latest = measurements[-1]
        
        # Try Timing Advance first (GSM only)
        if latest.ta is not None:
            position = self.calculate_position_timing_advance(latest.cell_id, latest.ta)
            if position:
                return position
        
        # Try RSSI triangulation if we have multiple measurements
        if len(measurements) >= 3:
            position = self.calculate_position_rssi_triangulation(measurements)
            if position:
                return position
        
        # Fall back to Cell ID lookup
        position = self.calculate_position_cell_id(latest.cell_id)
        return position
    
    def start_tracking(self, imsi: str):
        """
        Start movement tracking session
        
        Args:
            imsi: Target IMSI (will be anonymized automatically)
        """
        # 🔐 SECURITY FIX: Anonymize IMSI before storage
        imsi_hash = self.anonymizer.anonymize_imsi(imsi) if self.anonymize_identifiers else imsi
        
        self.tracking_sessions[imsi_hash] = MovementTrack(
            imsi=imsi_hash,  # 🔐 Store anonymized version
            positions=[],
            total_distance=0.0,
            average_speed=0.0,
            max_speed=0.0,
            duration=0.0
        )
        logger.info(f"Started tracking {imsi_hash}")
    
    def track_movement(self, imsi: str, duration: float) -> Optional[MovementTrack]:
        """
        Track target movement over time
        
        Args:
            imsi: Target IMSI (will be anonymized automatically)
            duration: Tracking duration in seconds
            
        Returns:
            Movement tracking data with GPS trail
        """
        # 🔐 SECURITY FIX: Anonymize IMSI
        imsi_hash = self.anonymizer.anonymize_imsi(imsi) if self.anonymize_identifiers else imsi
        
        if imsi_hash not in self.tracking_sessions:
            self.start_tracking(imsi)  # Will anonymize internally
        
        track = self.tracking_sessions[imsi_hash]
        start_time = time.time()
        
        logger.info(f"Tracking {imsi_hash} for {duration}s...")
        
        while time.time() - start_time < duration:
            # Calculate current position
            position = self.calculate_position(imsi)  # Pass original, will anonymize internally
            
            if position:
                track.positions.append(position)
                
                # Calculate distance from previous position
                if len(track.positions) > 1:
                    prev = track.positions[-2]
                    distance = self._haversine_distance(
                        prev.latitude, prev.longitude,
                        position.latitude, position.longitude
                    )
                    track.total_distance += distance
                    
                    # Calculate speed
                    time_diff = position.timestamp - prev.timestamp
                    if time_diff > 0:
                        speed = distance / time_diff
                        track.max_speed = max(track.max_speed, speed)
                
                logger.info(f"Position {len(track.positions)}: "
                          f"{position.latitude:.6f}, {position.longitude:.6f} "
                          f"(±{position.accuracy:.0f}m)")
            
            time.sleep(5)  # Update every 5 seconds
        
        # Calculate final statistics
        track.duration = time.time() - start_time
        if track.duration > 0:
            track.average_speed = track.total_distance / track.duration
        
        logger.info(f"Tracking complete: {len(track.positions)} positions, "
                   f"{track.total_distance:.0f}m total distance, "
                   f"avg speed {track.average_speed:.1f} m/s")
        
        return track
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """
        Calculate distance between two GPS coordinates
        
        Args:
            lat1, lon1: First position
            lat2, lon2: Second position
            
        Returns:
            Distance in meters
        """
        # Radius of Earth in meters
        R = 6371000
        
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        # Haversine formula
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def export_kml(self, imsi: str, output_path: str) -> bool:
        """
        Export tracking data to KML for Google Earth
        
        Args:
            imsi: Target IMSI (will be anonymized automatically)
            output_path: Output KML file path
            
        Returns:
            True if exported successfully
        """
        # 🔐 SECURITY FIX: Anonymize IMSI for session lookup
        imsi_hash = self.anonymizer.anonymize_imsi(imsi) if self.anonymize_identifiers else imsi
        
        track = self.tracking_sessions.get(imsi_hash)
        if not track or not track.positions:
            logger.error(f"No tracking data for {imsi_hash}")
            return False
        
        try:
            kml = self._generate_kml(track)
            
            with open(output_path, 'w') as f:
                f.write(kml)
            
            logger.info(f"Exported KML to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"KML export failed: {e}")
            return False
    
    def _generate_kml(self, track: MovementTrack) -> str:
        """Generate KML document from track"""
        kml = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>RF Arsenal - Target {track.imsi}</name>
    <description>Movement tracking: {len(track.positions)} positions over {track.duration:.0f}s</description>
    
    <Style id="trackLine">
      <LineStyle>
        <color>ff0000ff</color>
        <width>3</width>
      </LineStyle>
    </Style>
    
    <Style id="startPoint">
      <IconStyle>
        <color>ff00ff00</color>
        <scale>1.2</scale>
      </IconStyle>
    </Style>
    
    <Style id="endPoint">
      <IconStyle>
        <color>ff0000ff</color>
        <scale>1.2</scale>
      </IconStyle>
    </Style>
'''
        
        # Add start point
        if track.positions:
            start = track.positions[0]
            kml += f'''
    <Placemark>
      <name>Start</name>
      <description>Started at {datetime.fromtimestamp(start.timestamp).strftime('%Y-%m-%d %H:%M:%S')}</description>
      <styleUrl>#startPoint</styleUrl>
      <Point>
        <coordinates>{start.longitude},{start.latitude},{start.altitude}</coordinates>
      </Point>
    </Placemark>
'''
        
        # Add end point
        if len(track.positions) > 1:
            end = track.positions[-1]
            kml += f'''
    <Placemark>
      <name>End</name>
      <description>Ended at {datetime.fromtimestamp(end.timestamp).strftime('%Y-%m-%d %H:%M:%S')}</description>
      <styleUrl>#endPoint</styleUrl>
      <Point>
        <coordinates>{end.longitude},{end.latitude},{end.altitude}</coordinates>
      </Point>
    </Placemark>
'''
        
        # Add track line
        kml += '''
    <Placemark>
      <name>Movement Trail</name>
      <styleUrl>#trackLine</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>
'''
        
        for pos in track.positions:
            kml += f'          {pos.longitude},{pos.latitude},{pos.altitude}\n'
        
        kml += '''        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>'''
        
        return kml
    
    def get_statistics(self, imsi: str) -> Optional[Dict]:
        """
        Get tracking statistics for target
        
        Args:
            imsi: Target IMSI (will be anonymized automatically)
            
        Returns:
            Dictionary with tracking stats
        """
        # 🔐 SECURITY FIX: Anonymize IMSI for session lookup
        imsi_hash = self.anonymizer.anonymize_imsi(imsi) if self.anonymize_identifiers else imsi
        
        track = self.tracking_sessions.get(imsi_hash)
        if not track:
            return None
        
        return {
            'imsi': imsi_hash,  # 🔐 Return anonymized version
            'total_positions': len(track.positions),
            'total_distance_m': track.total_distance,
            'total_distance_km': track.total_distance / 1000,
            'duration_seconds': track.duration,
            'duration_minutes': track.duration / 60,
            'average_speed_ms': track.average_speed,
            'average_speed_kmh': track.average_speed * 3.6,
            'max_speed_ms': track.max_speed,
            'max_speed_kmh': track.max_speed * 3.6,
            'start_time': datetime.fromtimestamp(track.positions[0].timestamp).isoformat() if track.positions else None,
            'end_time': datetime.fromtimestamp(track.positions[-1].timestamp).isoformat() if track.positions else None
        }


if __name__ == "__main__":
    # Test geolocation engine
    logging.basicConfig(level=logging.INFO)
    
    print("RF Arsenal OS - Cellular Geolocation Test")
    print("=" * 50)
    
    geo = CellularGeolocation()
    
    # Simulate measurements
    test_imsi = "001010000000001"
    
    print(f"\n[+] Simulating measurements for {imsi}...")
    
    # Add simulated measurements
    geo.add_measurement(test_imsi, CellMeasurement(
        timestamp=time.time(),
        cell_id="310-410-12345",
        lac=1000,
        mcc=310,
        mnc=410,
        rssi=-75.0,
        ta=15,
        latitude=37.7749,
        longitude=-122.4194,
        altitude=10
    ))
    
    print("\n[+] Test complete")
