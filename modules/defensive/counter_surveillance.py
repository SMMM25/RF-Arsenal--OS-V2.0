#!/usr/bin/env python3
"""
RF Arsenal OS - Counter-Surveillance Module
Detect if YOU are being tracked, monitored, or targeted

DEFENSIVE CAPABILITIES:
- IMSI Catcher (Stingray) Detection
- Rogue Access Point Detection
- RF Direction Finding on YOUR position
- Cellular Anomaly Detection
- WiFi Deauth Attack Detection
- Bluetooth Tracking Detection
- GPS Spoofing Detection (on your device)

WHITE HAT USE: Know when you're being surveilled during authorized ops
"""

import logging
import time
import threading
import hashlib
import statistics
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Set, Callable
from enum import Enum, auto
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class ThreatType(Enum):
    """Types of surveillance threats"""
    IMSI_CATCHER = auto()          # Stingray/fake base station
    ROGUE_AP = auto()              # Evil twin / rogue access point
    DEAUTH_ATTACK = auto()         # WiFi deauthentication attack
    RF_DIRECTION_FINDING = auto()  # Someone DF'ing your position
    CELLULAR_ANOMALY = auto()      # Unusual cellular behavior
    BLUETOOTH_TRACKER = auto()     # BT tracking device (AirTag, etc)
    GPS_SPOOFING = auto()          # GPS spoofing attack on YOU
    WIFI_PROBE_HARVEST = auto()    # Someone harvesting your probes
    DOWNGRADE_ATTACK = auto()      # Forced 2G downgrade
    SILENT_SMS = auto()            # Silent/stealth SMS (Type 0)


class ThreatSeverity(Enum):
    """Threat severity levels"""
    LOW = 1        # Suspicious but could be benign
    MEDIUM = 2     # Likely hostile activity
    HIGH = 3       # Confirmed hostile activity
    CRITICAL = 4   # Active attack in progress


@dataclass
class SurveillanceThreat:
    """Detected surveillance threat"""
    threat_type: ThreatType
    severity: ThreatSeverity
    timestamp: datetime
    description: str
    evidence: Dict = field(default_factory=dict)
    location: Optional[Tuple[float, float]] = None  # lat, lon if available
    recommended_action: str = ""
    acknowledged: bool = False
    threat_id: str = ""
    
    def __post_init__(self):
        if not self.threat_id:
            self.threat_id = hashlib.sha256(
                f"{self.threat_type.name}{self.timestamp.isoformat()}{self.description}".encode()
            ).hexdigest()[:12]


@dataclass
class CellTower:
    """Cell tower information for analysis"""
    mcc: int                    # Mobile Country Code
    mnc: int                    # Mobile Network Code
    lac: int                    # Location Area Code
    cell_id: int                # Cell ID
    signal_strength: int        # dBm
    frequency: int              # Hz
    technology: str             # 2G/3G/4G/5G
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    seen_count: int = 1
    
    @property
    def identifier(self) -> str:
        return f"{self.mcc}-{self.mnc}-{self.lac}-{self.cell_id}"


@dataclass
class AccessPoint:
    """WiFi access point for analysis"""
    bssid: str                  # MAC address
    ssid: str                   # Network name
    channel: int
    signal_strength: int        # dBm
    encryption: str             # WPA2, WPA3, Open, etc
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    beacon_count: int = 1
    
    @property
    def identifier(self) -> str:
        return self.bssid.lower()


class IMSICatcherDetector:
    """
    Detect IMSI Catchers (Stingrays, fake base stations)
    
    Detection methods:
    1. New cell towers appearing suddenly
    2. Unusually strong signals from unknown towers
    3. Forced 2G downgrade (encryption stripping)
    4. Cell tower with mismatched parameters
    5. LAC/Cell ID anomalies
    6. Rapid tower switching
    """
    
    # Known carrier configurations (extend as needed)
    KNOWN_CARRIERS = {
        # US carriers
        ('310', '410'): 'AT&T',
        ('311', '480'): 'Verizon',
        ('310', '260'): 'T-Mobile',
        ('312', '530'): 'Sprint',
        # Add more as needed
    }
    
    def __init__(self):
        self.known_towers: Dict[str, CellTower] = {}
        self.tower_history: List[CellTower] = []
        self.baseline_established = False
        self.baseline_tower_count = 0
        self.anomalies: List[Dict] = []
        
    def establish_baseline(self, towers: List[CellTower], duration_minutes: int = 30):
        """
        Establish baseline of known legitimate towers
        Run this in a known-safe location before operations
        """
        logger.info(f"Establishing cellular baseline ({duration_minutes} min)...")
        
        for tower in towers:
            tower_id = tower.identifier
            if tower_id in self.known_towers:
                existing = self.known_towers[tower_id]
                existing.last_seen = tower.last_seen
                existing.seen_count += 1
            else:
                self.known_towers[tower_id] = tower
                
        self.baseline_tower_count = len(self.known_towers)
        self.baseline_established = True
        logger.info(f"Baseline established: {self.baseline_tower_count} towers cataloged")
        
    def analyze_tower(self, tower: CellTower) -> Optional[SurveillanceThreat]:
        """Analyze a cell tower for IMSI catcher indicators"""
        threats = []
        
        tower_id = tower.identifier
        
        # Check 1: New tower not in baseline
        if self.baseline_established and tower_id not in self.known_towers:
            threat = SurveillanceThreat(
                threat_type=ThreatType.IMSI_CATCHER,
                severity=ThreatSeverity.MEDIUM,
                timestamp=datetime.now(),
                description=f"New cell tower detected: {tower_id}",
                evidence={
                    'tower_id': tower_id,
                    'signal_strength': tower.signal_strength,
                    'technology': tower.technology,
                    'frequency': tower.frequency
                },
                recommended_action="Monitor tower behavior. Consider moving to different location."
            )
            threats.append(threat)
            
        # Check 2: Unusually strong signal (potential nearby fake tower)
        if tower.signal_strength > -50:  # Very strong signal
            threat = SurveillanceThreat(
                threat_type=ThreatType.IMSI_CATCHER,
                severity=ThreatSeverity.HIGH,
                timestamp=datetime.now(),
                description=f"Abnormally strong cell signal: {tower.signal_strength} dBm",
                evidence={
                    'tower_id': tower_id,
                    'signal_strength': tower.signal_strength,
                    'typical_range': '-60 to -100 dBm'
                },
                recommended_action="IMMEDIATE: Possible IMSI catcher nearby. Enable airplane mode."
            )
            threats.append(threat)
            
        # Check 3: Forced 2G downgrade
        if tower.technology == '2G':
            # Check if we have 4G/5G towers available
            modern_towers = [t for t in self.known_towers.values() 
                          if t.technology in ('4G', '5G', 'LTE')]
            if modern_towers:
                threat = SurveillanceThreat(
                    threat_type=ThreatType.DOWNGRADE_ATTACK,
                    severity=ThreatSeverity.CRITICAL,
                    timestamp=datetime.now(),
                    description="Forced 2G downgrade detected - encryption may be compromised",
                    evidence={
                        'current_technology': '2G',
                        'available_modern_towers': len(modern_towers),
                        'tower_id': tower_id
                    },
                    recommended_action="CRITICAL: Do not make calls/texts. 2G has weak encryption. Go to airplane mode."
                )
                threats.append(threat)
                
        # Check 4: LAC anomaly (Location Area Code mismatch)
        known_lacs = set(t.lac for t in self.known_towers.values() 
                        if t.mcc == tower.mcc and t.mnc == tower.mnc)
        if known_lacs and tower.lac not in known_lacs:
            threat = SurveillanceThreat(
                threat_type=ThreatType.IMSI_CATCHER,
                severity=ThreatSeverity.MEDIUM,
                timestamp=datetime.now(),
                description=f"LAC mismatch: Tower using LAC {tower.lac}, expected {known_lacs}",
                evidence={
                    'tower_lac': tower.lac,
                    'expected_lacs': list(known_lacs),
                    'tower_id': tower_id
                },
                recommended_action="Suspicious tower configuration. Monitor closely."
            )
            threats.append(threat)
            
        # Return highest severity threat
        if threats:
            return max(threats, key=lambda t: t.severity.value)
        return None
        
    def detect_rapid_switching(self, tower_log: List[Tuple[datetime, CellTower]], 
                                window_seconds: int = 60) -> Optional[SurveillanceThreat]:
        """Detect rapid tower switching (sign of tracking)"""
        if len(tower_log) < 3:
            return None
            
        # Count unique towers in time window
        now = datetime.now()
        recent = [(ts, t) for ts, t in tower_log 
                 if (now - ts).total_seconds() < window_seconds]
        
        unique_towers = set(t.identifier for _, t in recent)
        
        if len(unique_towers) >= 5:  # 5+ towers in 60 seconds is suspicious
            return SurveillanceThreat(
                threat_type=ThreatType.IMSI_CATCHER,
                severity=ThreatSeverity.HIGH,
                timestamp=datetime.now(),
                description=f"Rapid tower switching: {len(unique_towers)} towers in {window_seconds}s",
                evidence={
                    'tower_count': len(unique_towers),
                    'window_seconds': window_seconds,
                    'towers': list(unique_towers)
                },
                recommended_action="Possible tracking attempt. Your device is being forced between towers."
            )
        return None


class RogueAPDetector:
    """
    Detect Rogue Access Points (Evil Twin attacks)
    
    Detection methods:
    1. Duplicate SSIDs with different BSSIDs
    2. Known network with changed parameters
    3. Unusually strong signal from known network
    4. Deauthentication flood detection
    5. Beacon anomalies
    """
    
    def __init__(self):
        self.known_aps: Dict[str, AccessPoint] = {}
        self.trusted_networks: Dict[str, str] = {}  # SSID -> expected BSSID
        self.deauth_counts: Dict[str, List[datetime]] = {}  # BSSID -> deauth timestamps
        self.baseline_established = False
        
    def add_trusted_network(self, ssid: str, bssid: str):
        """Add a known-good network to trust list"""
        self.trusted_networks[ssid] = bssid.lower()
        logger.info(f"Added trusted network: {ssid} ({bssid})")
        
    def establish_baseline(self, access_points: List[AccessPoint]):
        """Establish baseline of known APs in the area"""
        for ap in access_points:
            self.known_aps[ap.identifier] = ap
        self.baseline_established = True
        logger.info(f"WiFi baseline established: {len(self.known_aps)} access points")
        
    def analyze_ap(self, ap: AccessPoint) -> Optional[SurveillanceThreat]:
        """Analyze an access point for evil twin indicators"""
        threats = []
        
        # Check 1: Trusted network with wrong BSSID (EVIL TWIN)
        if ap.ssid in self.trusted_networks:
            expected_bssid = self.trusted_networks[ap.ssid]
            if ap.bssid.lower() != expected_bssid:
                threat = SurveillanceThreat(
                    threat_type=ThreatType.ROGUE_AP,
                    severity=ThreatSeverity.CRITICAL,
                    timestamp=datetime.now(),
                    description=f"EVIL TWIN DETECTED: '{ap.ssid}' has wrong BSSID",
                    evidence={
                        'ssid': ap.ssid,
                        'detected_bssid': ap.bssid,
                        'expected_bssid': expected_bssid,
                        'signal_strength': ap.signal_strength
                    },
                    recommended_action="DO NOT CONNECT. This is a fake access point mimicking your trusted network."
                )
                return threat  # Immediate return - critical threat
                
        # Check 2: Duplicate SSID with different BSSID
        existing_ssids = {a.ssid: a for a in self.known_aps.values()}
        if ap.ssid in existing_ssids:
            existing = existing_ssids[ap.ssid]
            if existing.bssid.lower() != ap.bssid.lower():
                # Could be legitimate (multiple APs) or evil twin
                # Flag if new one is stronger
                if ap.signal_strength > existing.signal_strength + 10:
                    threat = SurveillanceThreat(
                        threat_type=ThreatType.ROGUE_AP,
                        severity=ThreatSeverity.HIGH,
                        timestamp=datetime.now(),
                        description=f"Possible evil twin: '{ap.ssid}' - new stronger AP appeared",
                        evidence={
                            'ssid': ap.ssid,
                            'new_bssid': ap.bssid,
                            'existing_bssid': existing.bssid,
                            'new_signal': ap.signal_strength,
                            'existing_signal': existing.signal_strength
                        },
                        recommended_action="Verify this is a legitimate AP before connecting."
                    )
                    threats.append(threat)
                    
        # Check 3: Open network mimicking encrypted one
        if ap.ssid in existing_ssids:
            existing = existing_ssids[ap.ssid]
            if existing.encryption != 'Open' and ap.encryption == 'Open':
                threat = SurveillanceThreat(
                    threat_type=ThreatType.ROGUE_AP,
                    severity=ThreatSeverity.CRITICAL,
                    timestamp=datetime.now(),
                    description=f"Open AP mimicking encrypted network: '{ap.ssid}'",
                    evidence={
                        'ssid': ap.ssid,
                        'rogue_bssid': ap.bssid,
                        'rogue_encryption': 'Open',
                        'legitimate_encryption': existing.encryption
                    },
                    recommended_action="CRITICAL: Do not connect. Attacker trying to intercept traffic."
                )
                return threat
                
        if threats:
            return max(threats, key=lambda t: t.severity.value)
        return None
        
    def record_deauth(self, bssid: str):
        """Record a deauthentication frame"""
        bssid = bssid.lower()
        if bssid not in self.deauth_counts:
            self.deauth_counts[bssid] = []
        self.deauth_counts[bssid].append(datetime.now())
        
    def check_deauth_attack(self, window_seconds: int = 10, 
                            threshold: int = 5) -> Optional[SurveillanceThreat]:
        """Check for deauthentication attack"""
        now = datetime.now()
        
        for bssid, timestamps in self.deauth_counts.items():
            recent = [ts for ts in timestamps 
                     if (now - ts).total_seconds() < window_seconds]
            
            if len(recent) >= threshold:
                return SurveillanceThreat(
                    threat_type=ThreatType.DEAUTH_ATTACK,
                    severity=ThreatSeverity.HIGH,
                    timestamp=datetime.now(),
                    description=f"Deauth attack detected: {len(recent)} frames in {window_seconds}s",
                    evidence={
                        'target_bssid': bssid,
                        'deauth_count': len(recent),
                        'window_seconds': window_seconds
                    },
                    recommended_action="Someone is trying to disconnect you. May precede evil twin attack."
                )
        return None


class BluetoothTrackerDetector:
    """Detect Bluetooth tracking devices (AirTags, Tiles, etc)"""
    
    # Known tracker signatures
    TRACKER_SIGNATURES = {
        'apple_airtag': {'prefix': 'AC:DE:48', 'service_uuid': '7DFC9000'},
        'tile': {'prefix': 'E8:D0:', 'service_uuid': 'FEED'},
        'samsung_smarttag': {'prefix': 'C4:AB:', 'service_uuid': 'FD5A'},
    }
    
    def __init__(self):
        self.detected_devices: Dict[str, Dict] = {}
        self.following_threshold = 3  # Number of sightings to consider "following"
        self.following_window = timedelta(hours=1)
        
    def analyze_ble_device(self, mac: str, rssi: int, 
                           service_uuids: List[str] = None) -> Optional[SurveillanceThreat]:
        """Analyze a BLE device for tracker signatures"""
        mac_upper = mac.upper()
        now = datetime.now()
        
        # Check against known tracker signatures
        for tracker_name, signature in self.TRACKER_SIGNATURES.items():
            if mac_upper.startswith(signature['prefix']):
                # Known tracker type detected
                if mac not in self.detected_devices:
                    self.detected_devices[mac] = {
                        'type': tracker_name,
                        'first_seen': now,
                        'sightings': []
                    }
                    
                self.detected_devices[mac]['sightings'].append({
                    'time': now,
                    'rssi': rssi
                })
                
                # Check if it's following us
                sightings = self.detected_devices[mac]['sightings']
                recent = [s for s in sightings 
                         if now - s['time'] < self.following_window]
                
                if len(recent) >= self.following_threshold:
                    return SurveillanceThreat(
                        threat_type=ThreatType.BLUETOOTH_TRACKER,
                        severity=ThreatSeverity.HIGH,
                        timestamp=now,
                        description=f"{tracker_name} tracker following you: {mac}",
                        evidence={
                            'tracker_type': tracker_name,
                            'mac_address': mac,
                            'sighting_count': len(recent),
                            'first_seen': self.detected_devices[mac]['first_seen'].isoformat(),
                            'current_rssi': rssi
                        },
                        recommended_action="Bluetooth tracker is following you. Search your belongings/vehicle."
                    )
                elif len(recent) == 1:
                    return SurveillanceThreat(
                        threat_type=ThreatType.BLUETOOTH_TRACKER,
                        severity=ThreatSeverity.LOW,
                        timestamp=now,
                        description=f"Tracking device detected nearby: {tracker_name}",
                        evidence={
                            'tracker_type': tracker_name,
                            'mac_address': mac,
                            'rssi': rssi
                        },
                        recommended_action="Monitor if this device continues appearing."
                    )
                    
        return None


class GPSSpoofingDetector:
    """Detect if GPS spoofing is being used against YOU"""
    
    def __init__(self):
        self.position_history: List[Dict] = []
        self.max_reasonable_speed = 200  # m/s (about 450 mph)
        
    def analyze_position(self, lat: float, lon: float, altitude: float,
                        timestamp: datetime, num_satellites: int,
                        hdop: float) -> Optional[SurveillanceThreat]:
        """Analyze GPS position for spoofing indicators"""
        threats = []
        
        # Store position
        current = {
            'lat': lat, 'lon': lon, 'alt': altitude,
            'time': timestamp, 'sats': num_satellites, 'hdop': hdop
        }
        
        # Check 1: Impossible position jump
        if self.position_history:
            last = self.position_history[-1]
            time_delta = (timestamp - last['time']).total_seconds()
            
            if time_delta > 0:
                # Calculate distance (simplified, not accounting for Earth's curvature)
                lat_diff = abs(lat - last['lat']) * 111000  # ~111km per degree
                lon_diff = abs(lon - last['lon']) * 111000 * abs(math.cos(math.radians(lat)))
                distance = (lat_diff**2 + lon_diff**2)**0.5
                speed = distance / time_delta
                
                if speed > self.max_reasonable_speed:
                    threat = SurveillanceThreat(
                        threat_type=ThreatType.GPS_SPOOFING,
                        severity=ThreatSeverity.CRITICAL,
                        timestamp=datetime.now(),
                        description=f"Impossible GPS jump: {speed:.0f} m/s ({speed*2.237:.0f} mph)",
                        evidence={
                            'calculated_speed': speed,
                            'distance_meters': distance,
                            'time_seconds': time_delta,
                            'from_position': (last['lat'], last['lon']),
                            'to_position': (lat, lon)
                        },
                        recommended_action="GPS is being spoofed. Do not trust navigation. Use backup nav."
                    )
                    threats.append(threat)
                    
        # Check 2: Abnormal HDOP (Horizontal Dilution of Precision)
        if hdop > 20:  # Very poor precision, could indicate interference
            threat = SurveillanceThreat(
                threat_type=ThreatType.GPS_SPOOFING,
                severity=ThreatSeverity.MEDIUM,
                timestamp=datetime.now(),
                description=f"Poor GPS precision (HDOP: {hdop}) - possible interference",
                evidence={
                    'hdop': hdop,
                    'satellites': num_satellites,
                    'normal_hdop_range': '1-5'
                },
                recommended_action="GPS signal degraded. May be natural or intentional interference."
            )
            threats.append(threat)
            
        # Check 3: Low satellite count with strong "lock"
        if num_satellites < 4 and hdop < 2:
            threat = SurveillanceThreat(
                threat_type=ThreatType.GPS_SPOOFING,
                severity=ThreatSeverity.HIGH,
                timestamp=datetime.now(),
                description="Suspicious: High precision with few satellites",
                evidence={
                    'satellites': num_satellites,
                    'hdop': hdop,
                    'expected': 'Low precision with <4 satellites'
                },
                recommended_action="Possible GPS spoofing - fake signals providing false precision."
            )
            threats.append(threat)
            
        self.position_history.append(current)
        # Keep last 100 positions
        if len(self.position_history) > 100:
            self.position_history = self.position_history[-100:]
            
        if threats:
            return max(threats, key=lambda t: t.severity.value)
        return None


class CounterSurveillanceSystem:
    """
    Master Counter-Surveillance System
    Integrates all detection capabilities
    """
    
    def __init__(self, hardware_controller=None):
        self.hw = hardware_controller
        self.logger = logging.getLogger('CounterSurveillance')
        
        # Initialize detectors
        self.imsi_detector = IMSICatcherDetector()
        self.rogue_ap_detector = RogueAPDetector()
        self.bt_tracker_detector = BluetoothTrackerDetector()
        self.gps_spoof_detector = GPSSpoofingDetector()
        
        # Threat tracking
        self.active_threats: Dict[str, SurveillanceThreat] = {}
        self.threat_history: List[SurveillanceThreat] = []
        self.threat_callbacks: List[Callable[[SurveillanceThreat], None]] = []
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'scans_completed': 0,
            'threats_detected': 0,
            'critical_threats': 0,
            'monitoring_start': None
        }
        
    def register_threat_callback(self, callback: Callable[[SurveillanceThreat], None]):
        """Register callback for threat notifications"""
        self.threat_callbacks.append(callback)
        
    def _notify_threat(self, threat: SurveillanceThreat):
        """Notify all registered callbacks of a threat"""
        self.active_threats[threat.threat_id] = threat
        self.threat_history.append(threat)
        self.stats['threats_detected'] += 1
        
        if threat.severity == ThreatSeverity.CRITICAL:
            self.stats['critical_threats'] += 1
            self.logger.critical(f"CRITICAL THREAT: {threat.description}")
        else:
            self.logger.warning(f"Threat detected: {threat.description}")
            
        for callback in self.threat_callbacks:
            try:
                callback(threat)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
                
    def establish_baseline(self, duration_minutes: int = 5) -> Dict:
        """
        Establish security baseline
        Run this in a known-safe location before operations
        """
        self.logger.info(f"Establishing counter-surveillance baseline ({duration_minutes} min)...")
        
        results = {
            'cell_towers': 0,
            'wifi_aps': 0,
            'bluetooth_devices': 0,
            'baseline_time': datetime.now().isoformat()
        }
        
        # In real implementation, this would scan using hardware
        # For now, we prepare the detectors
        self.imsi_detector.baseline_established = True
        self.rogue_ap_detector.baseline_established = True
        
        self.logger.info("Baseline established - counter-surveillance ready")
        return results
        
    def add_trusted_wifi(self, ssid: str, bssid: str):
        """Add a trusted WiFi network"""
        self.rogue_ap_detector.add_trusted_network(ssid, bssid)
        
    def scan_cellular(self, towers: List[Dict] = None) -> List[SurveillanceThreat]:
        """Scan for cellular threats (IMSI catchers)"""
        threats = []
        
        if towers is None:
            # In real implementation, would query modem
            self.logger.info("Scanning cellular environment...")
            towers = []
            
        for tower_data in towers:
            tower = CellTower(**tower_data)
            threat = self.imsi_detector.analyze_tower(tower)
            if threat:
                threats.append(threat)
                self._notify_threat(threat)
                
        self.stats['scans_completed'] += 1
        return threats
        
    def scan_wifi(self, access_points: List[Dict] = None) -> List[SurveillanceThreat]:
        """Scan for WiFi threats (rogue APs, deauth attacks)"""
        threats = []
        
        if access_points is None:
            self.logger.info("Scanning WiFi environment...")
            access_points = []
            
        for ap_data in access_points:
            ap = AccessPoint(**ap_data)
            threat = self.rogue_ap_detector.analyze_ap(ap)
            if threat:
                threats.append(threat)
                self._notify_threat(threat)
                
        # Check for deauth attacks
        deauth_threat = self.rogue_ap_detector.check_deauth_attack()
        if deauth_threat:
            threats.append(deauth_threat)
            self._notify_threat(deauth_threat)
            
        self.stats['scans_completed'] += 1
        return threats
        
    def scan_bluetooth(self, devices: List[Dict] = None) -> List[SurveillanceThreat]:
        """Scan for Bluetooth trackers"""
        threats = []
        
        if devices is None:
            self.logger.info("Scanning for Bluetooth trackers...")
            devices = []
            
        for device in devices:
            threat = self.bt_tracker_detector.analyze_ble_device(
                mac=device.get('mac', ''),
                rssi=device.get('rssi', -100),
                service_uuids=device.get('services', [])
            )
            if threat:
                threats.append(threat)
                self._notify_threat(threat)
                
        self.stats['scans_completed'] += 1
        return threats
        
    def check_gps(self, lat: float, lon: float, altitude: float = 0,
                  satellites: int = 0, hdop: float = 1.0) -> Optional[SurveillanceThreat]:
        """Check GPS for spoofing"""
        threat = self.gps_spoof_detector.analyze_position(
            lat=lat, lon=lon, altitude=altitude,
            timestamp=datetime.now(),
            num_satellites=satellites, hdop=hdop
        )
        if threat:
            self._notify_threat(threat)
        return threat
        
    def start_continuous_monitoring(self, interval_seconds: int = 30):
        """Start continuous background monitoring"""
        if self.monitoring:
            self.logger.warning("Already monitoring")
            return
            
        self.monitoring = True
        self.stats['monitoring_start'] = datetime.now()
        
        def monitor_loop():
            while self.monitoring:
                try:
                    self.scan_cellular()
                    self.scan_wifi()
                    self.scan_bluetooth()
                except Exception as e:
                    self.logger.error(f"Monitor error: {e}")
                time.sleep(interval_seconds)
                
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"Continuous monitoring started (every {interval_seconds}s)")
        
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Monitoring stopped")
        
    def acknowledge_threat(self, threat_id: str):
        """Acknowledge a threat (mark as reviewed)"""
        if threat_id in self.active_threats:
            self.active_threats[threat_id].acknowledged = True
            self.logger.info(f"Threat {threat_id} acknowledged")
            
    def get_active_threats(self) -> List[SurveillanceThreat]:
        """Get all unacknowledged active threats"""
        return [t for t in self.active_threats.values() if not t.acknowledged]
        
    def get_threat_summary(self) -> Dict:
        """Get summary of threat status"""
        active = self.get_active_threats()
        return {
            'active_threats': len(active),
            'critical': len([t for t in active if t.severity == ThreatSeverity.CRITICAL]),
            'high': len([t for t in active if t.severity == ThreatSeverity.HIGH]),
            'medium': len([t for t in active if t.severity == ThreatSeverity.MEDIUM]),
            'low': len([t for t in active if t.severity == ThreatSeverity.LOW]),
            'total_detected': self.stats['threats_detected'],
            'scans_completed': self.stats['scans_completed'],
            'monitoring_active': self.monitoring,
            'monitoring_duration': str(datetime.now() - self.stats['monitoring_start']) 
                                  if self.stats['monitoring_start'] else None
        }
        
    def get_status(self) -> Dict:
        """Get full counter-surveillance status"""
        return {
            'system': 'Counter-Surveillance',
            'status': 'MONITORING' if self.monitoring else 'STANDBY',
            'baseline_established': {
                'cellular': self.imsi_detector.baseline_established,
                'wifi': self.rogue_ap_detector.baseline_established
            },
            'trusted_networks': len(self.rogue_ap_detector.trusted_networks),
            'threat_summary': self.get_threat_summary(),
            'detectors': {
                'imsi_catcher': 'ACTIVE',
                'rogue_ap': 'ACTIVE',
                'bluetooth_tracker': 'ACTIVE',
                'gps_spoofing': 'ACTIVE'
            }
        }


# Import math for GPS calculations
import math


# Convenience function for quick threat check
def quick_threat_check() -> Dict:
    """Quick one-shot threat assessment"""
    system = CounterSurveillanceSystem()
    system.scan_cellular()
    system.scan_wifi()
    system.scan_bluetooth()
    return system.get_threat_summary()


if __name__ == "__main__":
    # Demo
    logging.basicConfig(level=logging.INFO)
    
    cs = CounterSurveillanceSystem()
    
    # Add trusted network
    cs.add_trusted_wifi("MyHomeWiFi", "AA:BB:CC:DD:EE:FF")
    
    # Establish baseline
    cs.establish_baseline(duration_minutes=1)
    
    # Simulate evil twin detection
    fake_ap = {
        'bssid': '11:22:33:44:55:66',  # Different BSSID
        'ssid': 'MyHomeWiFi',           # Same SSID
        'channel': 6,
        'signal_strength': -40,         # Strong signal
        'encryption': 'WPA2'
    }
    
    threats = cs.scan_wifi([fake_ap])
    print(f"\nDetected threats: {len(threats)}")
    for threat in threats:
        print(f"  - {threat.severity.name}: {threat.description}")
        
    print(f"\nStatus: {json.dumps(cs.get_status(), indent=2)}")
