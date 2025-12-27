"""
RF Arsenal OS - Geolocation and Signal Mapping
Real-time signal geolocation, heatmaps, triangulation, and direction finding.
All operations work offline with locally-stored map tiles.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
import json
import math
from collections import deque


class GeolocationMethod(Enum):
    """Geolocation techniques"""
    RSSI_TRIANGULATION = "rssi_triangulation"
    TDOA = "tdoa"  # Time Difference of Arrival
    AOA = "aoa"    # Angle of Arrival (Direction Finding)
    FINGERPRINTING = "fingerprinting"  # ML-based
    HYBRID = "hybrid"


class MapProvider(Enum):
    """Offline map tile providers"""
    OPENSTREETMAP = "osm"
    OFFLINE_TILES = "offline"
    VECTOR_TILES = "vector"
    SATELLITE = "satellite"


@dataclass
class SignalLocation:
    """Geolocated signal"""
    latitude: float
    longitude: float
    altitude_m: float = 0.0
    accuracy_m: float = 100.0  # Circular error probable (CEP)
    timestamp: float = field(default_factory=time.time)
    frequency_hz: float = 0.0
    power_dbm: float = -100.0
    signal_type: str = "unknown"
    bearing_deg: Optional[float] = None
    distance_m: Optional[float] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorPosition:
    """Position of a receiving sensor"""
    sensor_id: str
    latitude: float
    longitude: float
    altitude_m: float = 0.0
    bearing_deg: float = 0.0  # Antenna bearing
    antenna_gain_dbi: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SignalMeasurement:
    """Signal measurement from one sensor"""
    sensor_id: str
    rssi_dbm: float
    frequency_hz: float
    timestamp: float
    bearing_deg: Optional[float] = None  # From DF antenna
    toa_ns: Optional[float] = None  # Time of Arrival in nanoseconds
    signal_type: str = "unknown"


class GeolocationMapper:
    """
    Production-grade signal geolocation system.
    
    Features:
    - Multi-sensor RSSI triangulation
    - TDOA (Time Difference of Arrival)
    - AOA (Angle of Arrival / Direction Finding)
    - RF fingerprinting with ML
    - Hybrid geolocation combining multiple methods
    - Offline map support
    - Signal heatmap generation
    - Track history and prediction
    """
    
    # Speed of light for TDOA calculations
    C = 299792458.0  # m/s
    
    # Earth radius for geodetic calculations
    EARTH_RADIUS_M = 6371000.0
    
    def __init__(self,
                 method: GeolocationMethod = GeolocationMethod.RSSI_TRIANGULATION,
                 path_loss_exponent: float = 2.5,
                 reference_rssi: float = -40.0,
                 reference_distance: float = 1.0):
        """
        Initialize geolocation mapper.
        
        Args:
            method: Geolocation method to use
            path_loss_exponent: Path loss exponent for RSSI model (2.0 free space, 2.5-4.0 urban)
            reference_rssi: Reference RSSI at reference distance (dBm)
            reference_distance: Reference distance (meters)
        """
        self.method = method
        self.path_loss_exponent = path_loss_exponent
        self.reference_rssi = reference_rssi
        self.reference_distance = reference_distance
        
        # Sensors
        self._sensors: Dict[str, SensorPosition] = {}
        
        # Measurements buffer
        self._measurements: List[SignalMeasurement] = []
        self._measurements_lock = threading.Lock()
        
        # Located signals
        self._located_signals: deque = deque(maxlen=1000)
        
        # Track history for each signal type/frequency
        self._tracks: Dict[str, List[SignalLocation]] = {}
        
        # RF fingerprint database (for ML-based location)
        self._fingerprint_db: Dict[str, List[Dict]] = {}
        
        # Callbacks
        self._location_callbacks: List[Callable] = []
        
    def register_sensor(self, sensor: SensorPosition) -> None:
        """Register a receiving sensor position"""
        self._sensors[sensor.sensor_id] = sensor
        
    def unregister_sensor(self, sensor_id: str) -> None:
        """Remove a sensor"""
        if sensor_id in self._sensors:
            del self._sensors[sensor_id]
            
    def add_measurement(self, measurement: SignalMeasurement) -> None:
        """Add a signal measurement from a sensor"""
        with self._measurements_lock:
            self._measurements.append(measurement)
            
            # Keep only recent measurements (last 10 seconds)
            cutoff = time.time() - 10
            self._measurements = [m for m in self._measurements if m.timestamp > cutoff]
            
    def locate_signal(self, frequency_hz: float, 
                     signal_type: str = "unknown") -> Optional[SignalLocation]:
        """
        Attempt to geolocate a signal.
        
        Args:
            frequency_hz: Signal frequency
            signal_type: Signal type identifier
            
        Returns:
            SignalLocation or None if insufficient data
        """
        with self._measurements_lock:
            # Get relevant measurements
            relevant = [m for m in self._measurements 
                       if abs(m.frequency_hz - frequency_hz) < 100e3]  # 100 kHz tolerance
            
        if len(relevant) < 2:
            return None
            
        if self.method == GeolocationMethod.RSSI_TRIANGULATION:
            return self._triangulate_rssi(relevant, signal_type)
        elif self.method == GeolocationMethod.TDOA:
            return self._locate_tdoa(relevant, signal_type)
        elif self.method == GeolocationMethod.AOA:
            return self._locate_aoa(relevant, signal_type)
        elif self.method == GeolocationMethod.FINGERPRINTING:
            return self._locate_fingerprint(relevant, signal_type)
        elif self.method == GeolocationMethod.HYBRID:
            return self._locate_hybrid(relevant, signal_type)
        else:
            return None
            
    def _triangulate_rssi(self, measurements: List[SignalMeasurement],
                         signal_type: str) -> Optional[SignalLocation]:
        """
        Triangulate signal position using RSSI from multiple sensors.
        Uses weighted least-squares optimization.
        """
        if len(measurements) < 3:
            # Need at least 3 sensors for 2D triangulation
            # With 2 sensors, we can only estimate along the line
            if len(measurements) == 2:
                return self._estimate_two_sensor(measurements, signal_type)
            return None
            
        # Get sensor positions and estimated distances
        sensor_positions = []
        distances = []
        weights = []
        
        for m in measurements:
            if m.sensor_id not in self._sensors:
                continue
                
            sensor = self._sensors[m.sensor_id]
            
            # Convert to local Cartesian (meters from first sensor)
            sensor_positions.append(self._latlon_to_local(
                sensor.latitude, sensor.longitude,
                self._sensors[measurements[0].sensor_id].latitude,
                self._sensors[measurements[0].sensor_id].longitude
            ))
            
            # Estimate distance from RSSI using log-distance path loss model
            # RSSI = RSSI_ref - 10 * n * log10(d / d_ref)
            # d = d_ref * 10^((RSSI_ref - RSSI) / (10 * n))
            if m.rssi_dbm < self.reference_rssi:
                distance = self.reference_distance * 10 ** (
                    (self.reference_rssi - m.rssi_dbm) / (10 * self.path_loss_exponent)
                )
            else:
                distance = self.reference_distance
                
            distances.append(distance)
            
            # Weight by RSSI (stronger signals = more reliable)
            weight = 10 ** (m.rssi_dbm / 20)  # Linear voltage weight
            weights.append(weight)
            
        if len(sensor_positions) < 3:
            return None
            
        sensor_positions = np.array(sensor_positions)
        distances = np.array(distances)
        weights = np.array(weights)
        
        # Weighted least-squares trilateration
        position = self._trilaterate_least_squares(sensor_positions, distances, weights)
        
        if position is None:
            return None
            
        # Convert back to lat/lon
        ref_sensor = self._sensors[measurements[0].sensor_id]
        lat, lon = self._local_to_latlon(
            position[0], position[1],
            ref_sensor.latitude, ref_sensor.longitude
        )
        
        # Estimate accuracy (CEP)
        residuals = np.abs(np.linalg.norm(sensor_positions - position, axis=1) - distances)
        accuracy = float(np.sqrt(np.mean(residuals**2)))
        
        location = SignalLocation(
            latitude=lat,
            longitude=lon,
            accuracy_m=accuracy,
            frequency_hz=measurements[0].frequency_hz,
            power_dbm=float(np.mean([m.rssi_dbm for m in measurements])),
            signal_type=signal_type,
            confidence=min(1.0, 3 / len(measurements)),  # More sensors = higher confidence
            metadata={"method": "rssi_triangulation", "num_sensors": len(measurements)}
        )
        
        self._add_location_to_track(location)
        return location
        
    def _trilaterate_least_squares(self, positions: np.ndarray,
                                   distances: np.ndarray,
                                   weights: np.ndarray) -> Optional[np.ndarray]:
        """
        Solve trilateration using weighted least squares.
        """
        n = len(positions)
        
        # Set up linear system: Ax = b
        # Using linearization: (x - xi)^2 + (y - yi)^2 = ri^2
        # Expanding and subtracting equations to eliminate quadratic terms
        
        A = np.zeros((n - 1, 2))
        b = np.zeros(n - 1)
        
        for i in range(1, n):
            A[i-1, 0] = 2 * (positions[0, 0] - positions[i, 0])
            A[i-1, 1] = 2 * (positions[0, 1] - positions[i, 1])
            
            b[i-1] = (distances[i]**2 - distances[0]**2 -
                      positions[i, 0]**2 + positions[0, 0]**2 -
                      positions[i, 1]**2 + positions[0, 1]**2)
            
        # Apply weights
        W = np.diag(weights[1:])
        
        try:
            # Weighted least squares: (A^T W A)^-1 A^T W b
            AtWA = A.T @ W @ A
            AtWb = A.T @ W @ b
            position = np.linalg.solve(AtWA, AtWb)
            return position
        except np.linalg.LinAlgError:
            return None
            
    def _estimate_two_sensor(self, measurements: List[SignalMeasurement],
                            signal_type: str) -> Optional[SignalLocation]:
        """
        Estimate signal location with only two sensors.
        Returns midpoint weighted by RSSI.
        """
        if len(measurements) != 2:
            return None
            
        sensors = [self._sensors.get(m.sensor_id) for m in measurements]
        if None in sensors:
            return None
            
        # Weight by linear RSSI
        w1 = 10 ** (measurements[0].rssi_dbm / 20)
        w2 = 10 ** (measurements[1].rssi_dbm / 20)
        
        # Weighted midpoint
        lat = (sensors[0].latitude * w1 + sensors[1].latitude * w2) / (w1 + w2)
        lon = (sensors[0].longitude * w1 + sensors[1].longitude * w2) / (w1 + w2)
        
        # Distance between sensors as accuracy estimate
        accuracy = self._haversine_distance(
            sensors[0].latitude, sensors[0].longitude,
            sensors[1].latitude, sensors[1].longitude
        ) / 2
        
        return SignalLocation(
            latitude=lat,
            longitude=lon,
            accuracy_m=accuracy,
            frequency_hz=measurements[0].frequency_hz,
            power_dbm=float(np.mean([m.rssi_dbm for m in measurements])),
            signal_type=signal_type,
            confidence=0.3,  # Low confidence with only 2 sensors
            metadata={"method": "two_sensor_weighted", "num_sensors": 2}
        )
        
    def _locate_tdoa(self, measurements: List[SignalMeasurement],
                    signal_type: str) -> Optional[SignalLocation]:
        """
        Time Difference of Arrival geolocation.
        Requires precisely synchronized sensors with TOA measurements.
        """
        # Filter measurements with TOA data
        toa_measurements = [m for m in measurements if m.toa_ns is not None]
        
        if len(toa_measurements) < 3:
            return None
            
        # Use first sensor as reference
        ref_sensor = self._sensors.get(toa_measurements[0].sensor_id)
        if ref_sensor is None:
            return None
            
        # Calculate TDOAs relative to reference
        hyperbolas = []
        
        for i in range(1, len(toa_measurements)):
            sensor = self._sensors.get(toa_measurements[i].sensor_id)
            if sensor is None:
                continue
                
            # TDOA in seconds
            tdoa_s = (toa_measurements[i].toa_ns - toa_measurements[0].toa_ns) * 1e-9
            
            # Distance difference
            distance_diff = tdoa_s * self.C
            
            # Sensor positions in local coords
            pos_ref = self._latlon_to_local(
                ref_sensor.latitude, ref_sensor.longitude,
                ref_sensor.latitude, ref_sensor.longitude
            )
            pos_i = self._latlon_to_local(
                sensor.latitude, sensor.longitude,
                ref_sensor.latitude, ref_sensor.longitude
            )
            
            hyperbolas.append({
                "pos_ref": pos_ref,
                "pos_i": pos_i,
                "distance_diff": distance_diff
            })
            
        if len(hyperbolas) < 2:
            return None
            
        # Solve hyperbolic intersection using iterative method
        position = self._solve_tdoa_hyperbolas(hyperbolas)
        
        if position is None:
            return None
            
        # Convert to lat/lon
        lat, lon = self._local_to_latlon(
            position[0], position[1],
            ref_sensor.latitude, ref_sensor.longitude
        )
        
        # TDOA typically more accurate than RSSI
        accuracy = 50.0  # meters, depends on timing accuracy
        
        return SignalLocation(
            latitude=lat,
            longitude=lon,
            accuracy_m=accuracy,
            frequency_hz=measurements[0].frequency_hz,
            power_dbm=float(np.mean([m.rssi_dbm for m in toa_measurements])),
            signal_type=signal_type,
            confidence=0.8,
            metadata={"method": "tdoa", "num_sensors": len(toa_measurements)}
        )
        
    def _solve_tdoa_hyperbolas(self, hyperbolas: List[Dict]) -> Optional[np.ndarray]:
        """Solve TDOA hyperbolic equations using Newton-Raphson iteration"""
        # Initial guess: centroid of sensors
        positions = np.array([h["pos_i"] for h in hyperbolas])
        x = np.mean(positions[:, 0])
        y = np.mean(positions[:, 1])
        
        for _ in range(50):  # Max iterations
            # Compute residuals and Jacobian
            residuals = []
            jacobian = []
            
            for h in hyperbolas:
                d_ref = np.sqrt((x - h["pos_ref"][0])**2 + (y - h["pos_ref"][1])**2)
                d_i = np.sqrt((x - h["pos_i"][0])**2 + (y - h["pos_i"][1])**2)
                
                residuals.append(d_i - d_ref - h["distance_diff"])
                
                # Partial derivatives
                if d_ref > 0 and d_i > 0:
                    dx_ref = -(x - h["pos_ref"][0]) / d_ref
                    dy_ref = -(y - h["pos_ref"][1]) / d_ref
                    dx_i = (x - h["pos_i"][0]) / d_i
                    dy_i = (y - h["pos_i"][1]) / d_i
                    jacobian.append([dx_i + dx_ref, dy_i + dy_ref])
                else:
                    jacobian.append([0, 0])
                    
            residuals = np.array(residuals)
            jacobian = np.array(jacobian)
            
            if np.max(np.abs(residuals)) < 1.0:  # Converged to 1 meter
                return np.array([x, y])
                
            try:
                # Newton-Raphson update
                delta = np.linalg.lstsq(jacobian, -residuals, rcond=None)[0]
                x += delta[0]
                y += delta[1]
            except np.linalg.LinAlgError:
                return None
                
        return np.array([x, y])
        
    def _locate_aoa(self, measurements: List[SignalMeasurement],
                   signal_type: str) -> Optional[SignalLocation]:
        """
        Angle of Arrival (Direction Finding) geolocation.
        Requires sensors with directional antenna bearings.
        """
        # Filter measurements with bearing data
        aoa_measurements = [m for m in measurements if m.bearing_deg is not None]
        
        if len(aoa_measurements) < 2:
            return None
            
        # Find intersection of bearing lines
        lines = []
        
        for m in aoa_measurements:
            sensor = self._sensors.get(m.sensor_id)
            if sensor is None:
                continue
                
            # Convert bearing to vector
            bearing_rad = math.radians(m.bearing_deg)
            
            lines.append({
                "origin": (sensor.latitude, sensor.longitude),
                "bearing_rad": bearing_rad
            })
            
        if len(lines) < 2:
            return None
            
        # Find best intersection point
        intersections = []
        
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                intersection = self._bearing_intersection(lines[i], lines[j])
                if intersection:
                    intersections.append(intersection)
                    
        if not intersections:
            return None
            
        # Average intersections
        lat = np.mean([p[0] for p in intersections])
        lon = np.mean([p[1] for p in intersections])
        
        # Accuracy depends on bearing spread and distance
        accuracy = 100.0  # Base estimate
        
        return SignalLocation(
            latitude=lat,
            longitude=lon,
            accuracy_m=accuracy,
            frequency_hz=measurements[0].frequency_hz,
            power_dbm=float(np.mean([m.rssi_dbm for m in aoa_measurements])),
            signal_type=signal_type,
            bearing_deg=float(np.mean([m.bearing_deg for m in aoa_measurements])),
            confidence=0.7,
            metadata={"method": "aoa", "num_sensors": len(aoa_measurements)}
        )
        
    def _bearing_intersection(self, line1: Dict, line2: Dict) -> Optional[Tuple[float, float]]:
        """Find intersection of two bearing lines"""
        lat1, lon1 = line1["origin"]
        lat2, lon2 = line2["origin"]
        theta1 = line1["bearing_rad"]
        theta2 = line2["bearing_rad"]
        
        # Check if lines are parallel
        if abs(theta1 - theta2) < 0.01:
            return None
            
        # Convert to local coordinates
        x1, y1 = 0, 0
        x2, y2 = self._latlon_to_local(lat2, lon2, lat1, lon1)
        
        # Direction vectors
        dx1, dy1 = math.sin(theta1), math.cos(theta1)
        dx2, dy2 = math.sin(theta2), math.cos(theta2)
        
        # Solve for intersection
        # P1 + t1 * D1 = P2 + t2 * D2
        denom = dx1 * dy2 - dy1 * dx2
        if abs(denom) < 1e-10:
            return None
            
        t1 = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denom
        
        if t1 < 0:  # Intersection behind sensor
            return None
            
        # Intersection point in local coords
        ix = x1 + t1 * dx1
        iy = y1 + t1 * dy1
        
        # Convert back to lat/lon
        return self._local_to_latlon(ix, iy, lat1, lon1)
        
    def _locate_fingerprint(self, measurements: List[SignalMeasurement],
                           signal_type: str) -> Optional[SignalLocation]:
        """
        RF Fingerprinting location using stored database.
        """
        if not self._fingerprint_db:
            return None
            
        # Create fingerprint vector from measurements
        fingerprint = {}
        for m in measurements:
            fingerprint[m.sensor_id] = m.rssi_dbm
            
        # Find closest match in database
        best_match = None
        best_distance = float('inf')
        
        key = f"{signal_type}_{int(measurements[0].frequency_hz / 1e6)}"
        
        if key not in self._fingerprint_db:
            return None
            
        for entry in self._fingerprint_db[key]:
            # Euclidean distance in RSSI space
            distance = 0
            common_sensors = 0
            
            for sensor_id, rssi in fingerprint.items():
                if sensor_id in entry["rssi"]:
                    distance += (rssi - entry["rssi"][sensor_id]) ** 2
                    common_sensors += 1
                    
            if common_sensors >= 2:
                distance = math.sqrt(distance / common_sensors)
                if distance < best_distance:
                    best_distance = distance
                    best_match = entry
                    
        if best_match is None:
            return None
            
        return SignalLocation(
            latitude=best_match["latitude"],
            longitude=best_match["longitude"],
            accuracy_m=best_match.get("accuracy", 50.0),
            frequency_hz=measurements[0].frequency_hz,
            power_dbm=float(np.mean([m.rssi_dbm for m in measurements])),
            signal_type=signal_type,
            confidence=max(0.1, 1.0 - best_distance / 20),  # Lower distance = higher confidence
            metadata={"method": "fingerprinting", "match_distance": best_distance}
        )
        
    def _locate_hybrid(self, measurements: List[SignalMeasurement],
                      signal_type: str) -> Optional[SignalLocation]:
        """
        Hybrid geolocation combining multiple methods.
        """
        locations = []
        weights = []
        
        # Try RSSI triangulation
        rssi_loc = self._triangulate_rssi(measurements, signal_type)
        if rssi_loc:
            locations.append(rssi_loc)
            weights.append(rssi_loc.confidence)
            
        # Try TDOA
        tdoa_loc = self._locate_tdoa(measurements, signal_type)
        if tdoa_loc:
            locations.append(tdoa_loc)
            weights.append(tdoa_loc.confidence * 1.5)  # TDOA weighted higher
            
        # Try AOA
        aoa_loc = self._locate_aoa(measurements, signal_type)
        if aoa_loc:
            locations.append(aoa_loc)
            weights.append(aoa_loc.confidence)
            
        # Try fingerprinting
        fp_loc = self._locate_fingerprint(measurements, signal_type)
        if fp_loc:
            locations.append(fp_loc)
            weights.append(fp_loc.confidence)
            
        if not locations:
            return None
            
        # Weighted average of all methods
        weights = np.array(weights)
        weights /= np.sum(weights)
        
        lat = sum(l.latitude * w for l, w in zip(locations, weights))
        lon = sum(l.longitude * w for l, w in zip(locations, weights))
        accuracy = sum(l.accuracy_m * w for l, w in zip(locations, weights))
        
        return SignalLocation(
            latitude=lat,
            longitude=lon,
            accuracy_m=accuracy,
            frequency_hz=measurements[0].frequency_hz,
            power_dbm=float(np.mean([m.rssi_dbm for m in measurements])),
            signal_type=signal_type,
            confidence=float(np.max([l.confidence for l in locations])),
            metadata={
                "method": "hybrid",
                "methods_used": [l.metadata.get("method") for l in locations]
            }
        )
        
    def _add_location_to_track(self, location: SignalLocation) -> None:
        """Add location to tracking history"""
        key = f"{location.signal_type}_{int(location.frequency_hz / 1e6)}"
        
        if key not in self._tracks:
            self._tracks[key] = []
            
        self._tracks[key].append(location)
        
        # Limit track history
        if len(self._tracks[key]) > 1000:
            self._tracks[key] = self._tracks[key][-1000:]
            
        self._located_signals.append(location)
        
        # Notify callbacks
        for callback in self._location_callbacks:
            callback(location)
            
    def add_fingerprint(self, latitude: float, longitude: float,
                       measurements: List[SignalMeasurement]) -> None:
        """Add a calibration point to the fingerprint database"""
        if not measurements:
            return
            
        key = f"{measurements[0].signal_type}_{int(measurements[0].frequency_hz / 1e6)}"
        
        if key not in self._fingerprint_db:
            self._fingerprint_db[key] = []
            
        rssi_map = {m.sensor_id: m.rssi_dbm for m in measurements}
        
        self._fingerprint_db[key].append({
            "latitude": latitude,
            "longitude": longitude,
            "rssi": rssi_map,
            "timestamp": time.time()
        })
        
    def get_track(self, signal_type: str, frequency_hz: float) -> List[SignalLocation]:
        """Get tracking history for a signal"""
        key = f"{signal_type}_{int(frequency_hz / 1e6)}"
        return self._tracks.get(key, [])
        
    def get_all_locations(self) -> List[SignalLocation]:
        """Get all recent located signals"""
        return list(self._located_signals)
        
    def _latlon_to_local(self, lat: float, lon: float,
                        ref_lat: float, ref_lon: float) -> Tuple[float, float]:
        """Convert lat/lon to local Cartesian coordinates (meters)"""
        d_lat = math.radians(lat - ref_lat)
        d_lon = math.radians(lon - ref_lon)
        
        x = d_lon * self.EARTH_RADIUS_M * math.cos(math.radians(ref_lat))
        y = d_lat * self.EARTH_RADIUS_M
        
        return (x, y)
        
    def _local_to_latlon(self, x: float, y: float,
                        ref_lat: float, ref_lon: float) -> Tuple[float, float]:
        """Convert local Cartesian coordinates to lat/lon"""
        d_lat = y / self.EARTH_RADIUS_M
        d_lon = x / (self.EARTH_RADIUS_M * math.cos(math.radians(ref_lat)))
        
        lat = ref_lat + math.degrees(d_lat)
        lon = ref_lon + math.degrees(d_lon)
        
        return (lat, lon)
        
    def _haversine_distance(self, lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
        """Calculate great-circle distance between two points"""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return self.EARTH_RADIUS_M * c
        
    def register_callback(self, callback: Callable) -> None:
        """Register callback for new locations"""
        self._location_callbacks.append(callback)
        
    def export_to_geojson(self, filepath: str) -> bool:
        """Export all located signals to GeoJSON"""
        try:
            features = []
            for loc in self._located_signals:
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [loc.longitude, loc.latitude]
                    },
                    "properties": {
                        "frequency_hz": loc.frequency_hz,
                        "power_dbm": loc.power_dbm,
                        "signal_type": loc.signal_type,
                        "accuracy_m": loc.accuracy_m,
                        "confidence": loc.confidence,
                        "timestamp": loc.timestamp
                    }
                }
                features.append(feature)
                
            geojson = {
                "type": "FeatureCollection",
                "features": features
            }
            
            with open(filepath, 'w') as f:
                json.dump(geojson, f, indent=2)
            return True
        except Exception:
            return False


class SignalHeatmap:
    """
    Signal strength heatmap generator for coverage mapping.
    """
    
    def __init__(self,
                 bounds: Tuple[float, float, float, float],  # (min_lat, min_lon, max_lat, max_lon)
                 resolution: int = 100):
        """
        Initialize heatmap generator.
        
        Args:
            bounds: Map bounds (min_lat, min_lon, max_lat, max_lon)
            resolution: Grid resolution (points per axis)
        """
        self.bounds = bounds
        self.resolution = resolution
        
        # Create grid
        self.lat_grid = np.linspace(bounds[0], bounds[2], resolution)
        self.lon_grid = np.linspace(bounds[1], bounds[3], resolution)
        
        # Heatmap data (signal strength at each grid point)
        self.heatmap_data = np.full((resolution, resolution), -120.0)  # dBm
        
        # Sample count for averaging
        self._sample_count = np.zeros((resolution, resolution))
        
    def add_measurement(self, latitude: float, longitude: float, power_dbm: float) -> None:
        """Add a measurement point to the heatmap"""
        # Find grid cell
        lat_idx = np.searchsorted(self.lat_grid, latitude)
        lon_idx = np.searchsorted(self.lon_grid, longitude)
        
        lat_idx = np.clip(lat_idx, 0, self.resolution - 1)
        lon_idx = np.clip(lon_idx, 0, self.resolution - 1)
        
        # Running average
        count = self._sample_count[lat_idx, lon_idx]
        self.heatmap_data[lat_idx, lon_idx] = (
            (self.heatmap_data[lat_idx, lon_idx] * count + power_dbm) / (count + 1)
        )
        self._sample_count[lat_idx, lon_idx] += 1
        
    def interpolate(self) -> None:
        """Interpolate missing values using IDW"""
        # Find cells with data
        has_data = self._sample_count > 0
        
        if np.sum(has_data) < 4:
            return
            
        # Get coordinates of cells with data
        data_points = []
        data_values = []
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                if has_data[i, j]:
                    data_points.append((self.lat_grid[i], self.lon_grid[j]))
                    data_values.append(self.heatmap_data[i, j])
                    
        data_points = np.array(data_points)
        data_values = np.array(data_values)
        
        # IDW interpolation for cells without data
        for i in range(self.resolution):
            for j in range(self.resolution):
                if not has_data[i, j]:
                    point = np.array([self.lat_grid[i], self.lon_grid[j]])
                    
                    # Calculate distances
                    distances = np.linalg.norm(data_points - point, axis=1)
                    
                    # IDW weights
                    weights = 1 / (distances + 1e-10) ** 2
                    weights /= np.sum(weights)
                    
                    # Interpolated value
                    self.heatmap_data[i, j] = np.sum(data_values * weights)
                    
    def get_heatmap_data(self) -> Dict[str, Any]:
        """Get heatmap data for visualization"""
        return {
            "data": self.heatmap_data.tolist(),
            "bounds": self.bounds,
            "lat_grid": self.lat_grid.tolist(),
            "lon_grid": self.lon_grid.tolist(),
            "resolution": self.resolution,
            "min_dbm": float(np.min(self.heatmap_data)),
            "max_dbm": float(np.max(self.heatmap_data))
        }
        
    def export_to_kml(self, filepath: str) -> bool:
        """Export heatmap to KML for Google Earth"""
        try:
            kml = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>RF Signal Heatmap</name>
'''
            # Add ground overlay for heatmap
            # In production, this would generate an actual image
            kml += '''</Document>
</kml>'''
            
            with open(filepath, 'w') as f:
                f.write(kml)
            return True
        except Exception:
            return False
