#!/usr/bin/env python3
"""
RF Arsenal OS - Visual RF Threat Map & Dashboard
Real-time situational awareness display

FEATURES:
- Live RF signal visualization
- Threat classification (friendly/hostile/unknown)
- Signal strength heatmap
- Device tracking with direction estimates
- Stealth footprint visualization
- OPSEC score integration
- Counter-surveillance alerts
- Hardware health status

Supports both GUI (PyQt6) and Terminal (Rich) modes
"""

import logging
import time
import threading
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
from enum import Enum, auto
from datetime import datetime, timedelta
from collections import deque
import json

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """RF signal classification"""
    WIFI_24 = auto()        # 2.4 GHz WiFi
    WIFI_5 = auto()         # 5 GHz WiFi
    CELLULAR_2G = auto()    # GSM
    CELLULAR_3G = auto()    # UMTS
    CELLULAR_4G = auto()    # LTE
    CELLULAR_5G = auto()    # NR
    BLUETOOTH = auto()      # Bluetooth/BLE
    BLUETOOTH_LE = auto()   # BLE specifically
    GPS = auto()            # GPS signals
    DRONE = auto()          # Drone control/video
    ISM_433 = auto()        # 433 MHz ISM
    ISM_915 = auto()        # 915 MHz ISM
    UNKNOWN = auto()        # Unclassified


class ThreatLevel(Enum):
    """Signal threat classification"""
    FRIENDLY = 0        # Known-good device
    NEUTRAL = 1         # Unknown but not suspicious
    SUSPICIOUS = 2      # Potentially hostile
    HOSTILE = 3         # Confirmed threat
    

class DeviceType(Enum):
    """Detected device types"""
    PHONE = auto()
    LAPTOP = auto()
    TABLET = auto()
    IOT_DEVICE = auto()
    ACCESS_POINT = auto()
    CELL_TOWER = auto()
    DRONE = auto()
    TRACKER = auto()
    VEHICLE = auto()
    UNKNOWN = auto()


@dataclass
class RFSignal:
    """Detected RF signal"""
    signal_id: str
    signal_type: SignalType
    frequency: float            # Hz
    signal_strength: int        # dBm
    timestamp: datetime
    threat_level: ThreatLevel = ThreatLevel.NEUTRAL
    device_type: DeviceType = DeviceType.UNKNOWN
    identifier: str = ""        # MAC, IMSI, etc
    name: str = ""              # SSID, device name, etc
    direction: Optional[float] = None  # Degrees from north
    distance_estimate: Optional[float] = None  # Meters
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    
    def age_seconds(self) -> float:
        return (datetime.now() - self.timestamp).total_seconds()


@dataclass 
class ThreatAlert:
    """Active threat alert"""
    alert_id: str
    severity: str               # CRITICAL, HIGH, MEDIUM, LOW
    title: str
    description: str
    timestamp: datetime
    signal: Optional[RFSignal] = None
    acknowledged: bool = False
    recommended_action: str = ""


@dataclass
class SystemHealth:
    """Hardware and system health status"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    temperature: float = 0.0
    sdr_connected: bool = False
    sdr_name: str = ""
    gps_lock: bool = False
    network_status: str = "OFFLINE"
    stealth_level: str = "MAXIMUM"
    uptime_seconds: int = 0


class SignalTracker:
    """Track signals over time"""
    
    def __init__(self, max_history: int = 1000):
        self.signals: Dict[str, RFSignal] = {}
        self.history: deque = deque(maxlen=max_history)
        self.signal_counts: Dict[SignalType, int] = {t: 0 for t in SignalType}
        
    def update_signal(self, signal: RFSignal):
        """Update or add a signal"""
        self.signals[signal.signal_id] = signal
        self.history.append(signal)
        self.signal_counts[signal.signal_type] += 1
        
    def get_active_signals(self, max_age_seconds: int = 60) -> List[RFSignal]:
        """Get signals seen within the time window"""
        return [s for s in self.signals.values() 
                if s.age_seconds() < max_age_seconds]
                
    def get_threats(self) -> List[RFSignal]:
        """Get all suspicious/hostile signals"""
        return [s for s in self.signals.values()
                if s.threat_level in (ThreatLevel.SUSPICIOUS, ThreatLevel.HOSTILE)]
                
    def get_by_type(self, signal_type: SignalType) -> List[RFSignal]:
        """Get signals of a specific type"""
        return [s for s in self.signals.values() if s.signal_type == signal_type]
        
    def cleanup_stale(self, max_age_seconds: int = 300):
        """Remove stale signals"""
        stale_ids = [sid for sid, sig in self.signals.items() 
                    if sig.age_seconds() > max_age_seconds]
        for sid in stale_ids:
            del self.signals[sid]


class ThreatMap:
    """
    RF Threat Map - Spatial visualization of RF environment
    """
    
    def __init__(self, grid_size: int = 100):
        self.grid_size = grid_size
        self.center_lat: float = 0.0
        self.center_lon: float = 0.0
        self.zoom_level: int = 15
        self.signals: Dict[str, RFSignal] = {}
        self.threat_zones: List[Dict] = []
        
    def set_center(self, lat: float, lon: float):
        """Set map center position"""
        self.center_lat = lat
        self.center_lon = lon
        
    def add_signal(self, signal: RFSignal):
        """Add signal to map"""
        self.signals[signal.signal_id] = signal
        
        # If hostile, create threat zone
        if signal.threat_level == ThreatLevel.HOSTILE:
            self.threat_zones.append({
                'center': (signal.latitude, signal.longitude),
                'radius': signal.distance_estimate or 100,
                'signal_id': signal.signal_id,
                'type': signal.signal_type.name
            })
            
    def get_signals_in_view(self) -> List[RFSignal]:
        """Get all signals visible in current view"""
        # In real implementation, would filter by lat/lon bounds
        return list(self.signals.values())
        
    def estimate_direction(self, signal_strength_samples: List[Tuple[float, int]]) -> float:
        """
        Estimate signal direction from multiple samples
        Each sample is (antenna_angle, signal_strength)
        """
        if not signal_strength_samples:
            return 0.0
            
        # Simple weighted average
        total_weight = 0
        weighted_sum = 0
        
        for angle, strength in signal_strength_samples:
            # Stronger signals have more weight
            weight = 10 ** (strength / 20)  # Convert dBm to linear
            weighted_sum += angle * weight
            total_weight += weight
            
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class RFThreatDashboard:
    """
    Main RF Threat Dashboard
    Integrates all visualization and monitoring
    """
    
    def __init__(self, hardware_controller=None):
        self.hw = hardware_controller
        self.logger = logging.getLogger('ThreatDashboard')
        
        # Core components
        self.signal_tracker = SignalTracker()
        self.threat_map = ThreatMap()
        
        # Alerts
        self.active_alerts: Dict[str, ThreatAlert] = {}
        self.alert_history: List[ThreatAlert] = []
        
        # System status
        self.health = SystemHealth()
        self.opsec_score: int = 100
        self.stealth_footprint: Dict[str, float] = {
            'rf_emissions': 0.0,
            'network_exposure': 0.0,
            'device_visibility': 0.0
        }
        
        # Monitoring
        self.monitoring = False
        self.update_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable[[ThreatAlert], None]] = []
        
        # Position
        self.current_lat: float = 0.0
        self.current_lon: float = 0.0
        
        # Statistics
        self.stats = {
            'signals_detected': 0,
            'threats_identified': 0,
            'alerts_generated': 0,
            'uptime_start': datetime.now()
        }
        
    def register_update_callback(self, callback: Callable):
        """Register callback for dashboard updates"""
        self.update_callbacks.append(callback)
        
    def register_alert_callback(self, callback: Callable[[ThreatAlert], None]):
        """Register callback for new alerts"""
        self.alert_callbacks.append(callback)
        
    def _notify_update(self):
        """Notify all update callbacks"""
        for callback in self.update_callbacks:
            try:
                callback(self.get_dashboard_state())
            except Exception as e:
                self.logger.error(f"Update callback error: {e}")
                
    def _notify_alert(self, alert: ThreatAlert):
        """Notify all alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
                
    def set_position(self, lat: float, lon: float):
        """Set current position for map centering"""
        self.current_lat = lat
        self.current_lon = lon
        self.threat_map.set_center(lat, lon)
        
    def process_signal(self, signal: RFSignal):
        """Process and classify a detected RF signal"""
        self.stats['signals_detected'] += 1
        
        # Auto-classify threat level based on signal characteristics
        signal.threat_level = self._classify_threat(signal)
        
        # Update tracker and map
        self.signal_tracker.update_signal(signal)
        self.threat_map.add_signal(signal)
        
        # Generate alert if hostile
        if signal.threat_level in (ThreatLevel.HOSTILE, ThreatLevel.SUSPICIOUS):
            self._generate_alert(signal)
            
        self._notify_update()
        
    def _classify_threat(self, signal: RFSignal) -> ThreatLevel:
        """Classify signal threat level"""
        # Check if already classified
        if signal.threat_level != ThreatLevel.NEUTRAL:
            return signal.threat_level
            
        threat_indicators = 0
        
        # Strong unexpected signal
        if signal.signal_strength > -40:
            threat_indicators += 1
            
        # Tracker device types
        if signal.device_type == DeviceType.TRACKER:
            threat_indicators += 2
            
        # Drone signals
        if signal.signal_type == SignalType.DRONE:
            threat_indicators += 1
            
        # Unknown strong cellular
        if signal.signal_type in (SignalType.CELLULAR_2G, SignalType.CELLULAR_3G,
                                  SignalType.CELLULAR_4G) and signal.signal_strength > -50:
            threat_indicators += 1
            
        # Classify based on indicators
        if threat_indicators >= 3:
            return ThreatLevel.HOSTILE
        elif threat_indicators >= 2:
            return ThreatLevel.SUSPICIOUS
        else:
            return ThreatLevel.NEUTRAL
            
    def _generate_alert(self, signal: RFSignal):
        """Generate alert for threatening signal"""
        self.stats['alerts_generated'] += 1
        self.stats['threats_identified'] += 1
        
        severity = "HIGH" if signal.threat_level == ThreatLevel.HOSTILE else "MEDIUM"
        
        alert = ThreatAlert(
            alert_id=f"ALT-{self.stats['alerts_generated']:04d}",
            severity=severity,
            title=f"{signal.threat_level.name} {signal.signal_type.name} Detected",
            description=f"Signal: {signal.name or signal.identifier or 'Unknown'}, "
                       f"Strength: {signal.signal_strength} dBm",
            timestamp=datetime.now(),
            signal=signal,
            recommended_action=self._get_recommended_action(signal)
        )
        
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self._notify_alert(alert)
        
        self.logger.warning(f"ALERT: {alert.title} - {alert.description}")
        
    def _get_recommended_action(self, signal: RFSignal) -> str:
        """Get recommended action for a threat"""
        actions = {
            SignalType.CELLULAR_2G: "Possible IMSI catcher. Consider airplane mode.",
            SignalType.DRONE: "Drone detected. Check for surveillance. Consider jamming.",
            SignalType.BLUETOOTH_LE: "Possible tracker. Search belongings.",
            SignalType.WIFI_24: "Possible rogue AP. Do not connect.",
            SignalType.WIFI_5: "Possible rogue AP. Verify before connecting.",
        }
        return actions.get(signal.signal_type, "Monitor situation. Consider evasive action.")
        
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            
    def dismiss_alert(self, alert_id: str):
        """Dismiss and remove an alert"""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            
    def update_health(self, health: SystemHealth):
        """Update system health status"""
        self.health = health
        self._notify_update()
        
    def update_opsec_score(self, score: int):
        """Update OPSEC score"""
        self.opsec_score = max(0, min(100, score))
        self._notify_update()
        
    def update_stealth_footprint(self, rf: float = None, network: float = None, 
                                  device: float = None):
        """Update stealth footprint metrics (0.0 = invisible, 1.0 = fully exposed)"""
        if rf is not None:
            self.stealth_footprint['rf_emissions'] = max(0, min(1, rf))
        if network is not None:
            self.stealth_footprint['network_exposure'] = max(0, min(1, network))
        if device is not None:
            self.stealth_footprint['device_visibility'] = max(0, min(1, device))
        self._notify_update()
        
    def get_signal_summary(self) -> Dict:
        """Get summary of detected signals"""
        active = self.signal_tracker.get_active_signals()
        
        by_type = {}
        for sig_type in SignalType:
            signals = [s for s in active if s.signal_type == sig_type]
            if signals:
                by_type[sig_type.name] = {
                    'count': len(signals),
                    'strongest': max(s.signal_strength for s in signals),
                    'threats': len([s for s in signals 
                                   if s.threat_level in (ThreatLevel.HOSTILE, ThreatLevel.SUSPICIOUS)])
                }
                
        return {
            'total_active': len(active),
            'by_type': by_type,
            'total_threats': len(self.signal_tracker.get_threats()),
            'signal_types_detected': len(by_type)
        }
        
    def get_threat_summary(self) -> Dict:
        """Get threat summary"""
        threats = self.signal_tracker.get_threats()
        active_alerts = [a for a in self.active_alerts.values() if not a.acknowledged]
        
        return {
            'active_threats': len(threats),
            'hostile': len([t for t in threats if t.threat_level == ThreatLevel.HOSTILE]),
            'suspicious': len([t for t in threats if t.threat_level == ThreatLevel.SUSPICIOUS]),
            'unacknowledged_alerts': len(active_alerts),
            'critical_alerts': len([a for a in active_alerts if a.severity == "CRITICAL"]),
            'high_alerts': len([a for a in active_alerts if a.severity == "HIGH"])
        }
        
    def get_dashboard_state(self) -> Dict:
        """Get complete dashboard state for rendering"""
        uptime = (datetime.now() - self.stats['uptime_start']).total_seconds()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'position': {
                'lat': self.current_lat,
                'lon': self.current_lon
            },
            'signals': self.get_signal_summary(),
            'threats': self.get_threat_summary(),
            'alerts': [
                {
                    'id': a.alert_id,
                    'severity': a.severity,
                    'title': a.title,
                    'description': a.description,
                    'time': a.timestamp.isoformat(),
                    'acknowledged': a.acknowledged
                }
                for a in sorted(self.active_alerts.values(), 
                              key=lambda x: x.timestamp, reverse=True)[:10]
            ],
            'health': {
                'cpu': self.health.cpu_usage,
                'memory': self.health.memory_usage,
                'temperature': self.health.temperature,
                'sdr_connected': self.health.sdr_connected,
                'sdr_name': self.health.sdr_name,
                'gps_lock': self.health.gps_lock,
                'network': self.health.network_status,
                'stealth': self.health.stealth_level
            },
            'opsec_score': self.opsec_score,
            'stealth_footprint': self.stealth_footprint,
            'stats': {
                'signals_detected': self.stats['signals_detected'],
                'threats_identified': self.stats['threats_identified'],
                'alerts_generated': self.stats['alerts_generated'],
                'uptime_seconds': int(uptime)
            }
        }
        
    def get_terminal_display(self) -> str:
        """Generate terminal-based dashboard display"""
        state = self.get_dashboard_state()
        
        # Build display
        lines = []
        lines.append("=" * 70)
        lines.append("           RF ARSENAL OS - THREAT DASHBOARD")
        lines.append("=" * 70)
        lines.append("")
        
        # Status bar
        opsec_bar = self._make_bar(state['opsec_score'], 100, 20)
        stealth = state['health']['stealth']
        network = state['health']['network']
        lines.append(f"OPSEC: [{opsec_bar}] {state['opsec_score']}%  |  "
                    f"Stealth: {stealth}  |  Network: {network}")
        lines.append("")
        
        # Threat summary
        threats = state['threats']
        threat_color = "🔴" if threats['hostile'] > 0 else ("🟡" if threats['suspicious'] > 0 else "🟢")
        lines.append(f"THREATS: {threat_color} {threats['active_threats']} active  "
                    f"({threats['hostile']} hostile, {threats['suspicious']} suspicious)")
        lines.append(f"ALERTS:  {threats['unacknowledged_alerts']} unacknowledged  "
                    f"({threats['critical_alerts']} critical, {threats['high_alerts']} high)")
        lines.append("")
        
        # Signal summary
        signals = state['signals']
        lines.append(f"SIGNALS: {signals['total_active']} active  |  "
                    f"{signals['signal_types_detected']} types  |  "
                    f"{signals['total_threats']} threats")
        lines.append("")
        
        # Signal breakdown
        if signals['by_type']:
            lines.append("Signal Types:")
            for sig_type, info in signals['by_type'].items():
                threat_indicator = f" ⚠️{info['threats']}" if info['threats'] > 0 else ""
                lines.append(f"  {sig_type:15} : {info['count']:3} signals  "
                           f"(max: {info['strongest']} dBm){threat_indicator}")
            lines.append("")
            
        # Active alerts
        if state['alerts']:
            lines.append("ACTIVE ALERTS:")
            for alert in state['alerts'][:5]:
                severity_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(alert['severity'], "⚪")
                ack = "✓" if alert['acknowledged'] else " "
                lines.append(f"  [{ack}] {severity_icon} {alert['severity']:8} : {alert['title']}")
            lines.append("")
            
        # Stealth footprint
        lines.append("STEALTH FOOTPRINT:")
        footprint = state['stealth_footprint']
        rf_bar = self._make_bar(footprint['rf_emissions'] * 100, 100, 15)
        net_bar = self._make_bar(footprint['network_exposure'] * 100, 100, 15)
        dev_bar = self._make_bar(footprint['device_visibility'] * 100, 100, 15)
        lines.append(f"  RF Emissions:    [{rf_bar}] {footprint['rf_emissions']*100:.0f}%")
        lines.append(f"  Network Exposure:[{net_bar}] {footprint['network_exposure']*100:.0f}%")
        lines.append(f"  Device Visible:  [{dev_bar}] {footprint['device_visibility']*100:.0f}%")
        lines.append("")
        
        # Hardware status
        health = state['health']
        sdr_status = f"✅ {health['sdr_name']}" if health['sdr_connected'] else "❌ Not connected"
        gps_status = "✅ Locked" if health['gps_lock'] else "❌ No lock"
        lines.append("HARDWARE:")
        lines.append(f"  SDR: {sdr_status}  |  GPS: {gps_status}")
        lines.append(f"  CPU: {health['cpu']:.1f}%  |  MEM: {health['memory']:.1f}%  |  "
                    f"TEMP: {health['temperature']:.1f}°C")
        lines.append("")
        
        # Footer
        lines.append("-" * 70)
        uptime_str = self._format_uptime(state['stats']['uptime_seconds'])
        lines.append(f"Signals: {state['stats']['signals_detected']}  |  "
                    f"Threats: {state['stats']['threats_identified']}  |  "
                    f"Alerts: {state['stats']['alerts_generated']}  |  "
                    f"Uptime: {uptime_str}")
        lines.append("=" * 70)
        
        return "\n".join(lines)
        
    def _make_bar(self, value: float, max_val: float, width: int) -> str:
        """Create ASCII progress bar"""
        filled = int((value / max_val) * width) if max_val > 0 else 0
        return "█" * filled + "░" * (width - filled)
        
    def _format_uptime(self, seconds: int) -> str:
        """Format uptime as human readable"""
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
            
    def start_monitoring(self, update_interval: float = 1.0):
        """Start dashboard monitoring loop"""
        if self.monitoring:
            return
            
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                try:
                    # Cleanup stale signals
                    self.signal_tracker.cleanup_stale()
                    
                    # Update system health (in real impl, would query actual hardware)
                    self._update_system_health()
                    
                    # Notify update
                    self._notify_update()
                    
                except Exception as e:
                    self.logger.error(f"Monitor loop error: {e}")
                    
                time.sleep(update_interval)
                
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        self.logger.info("Dashboard monitoring started")
        
    def stop_monitoring(self):
        """Stop dashboard monitoring"""
        self.monitoring = False
        self.logger.info("Dashboard monitoring stopped")
        
    def _update_system_health(self):
        """Update system health metrics"""
        try:
            import psutil
            self.health.cpu_usage = psutil.cpu_percent()
            self.health.memory_usage = psutil.virtual_memory().percent
            
            # Temperature (Linux)
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries:
                        self.health.temperature = entries[0].current
                        break
        except ImportError:
            pass
        except Exception as e:
            self.logger.debug(f"Health update error: {e}")


class ThreatMapRenderer:
    """
    ASCII-based threat map renderer for terminal display
    """
    
    def __init__(self, width: int = 60, height: int = 30):
        self.width = width
        self.height = height
        
    def render(self, signals: List[RFSignal], center_lat: float, center_lon: float,
               scale_meters: float = 500) -> str:
        """Render ASCII threat map"""
        # Initialize grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Center marker (YOU)
        cx, cy = self.width // 2, self.height // 2
        grid[cy][cx] = '◉'
        
        # Plot signals
        for signal in signals:
            if signal.latitude and signal.longitude:
                # Calculate relative position
                dx = (signal.longitude - center_lon) * 111000 * math.cos(math.radians(center_lat))
                dy = (signal.latitude - center_lat) * 111000
                
                # Scale to grid
                gx = int(cx + (dx / scale_meters) * (self.width / 2))
                gy = int(cy - (dy / scale_meters) * (self.height / 2))  # Invert Y
                
                # Check bounds
                if 0 <= gx < self.width and 0 <= gy < self.height:
                    # Choose symbol based on threat
                    if signal.threat_level == ThreatLevel.HOSTILE:
                        symbol = '⚠'
                    elif signal.threat_level == ThreatLevel.SUSPICIOUS:
                        symbol = '?'
                    elif signal.signal_type == SignalType.WIFI_24 or signal.signal_type == SignalType.WIFI_5:
                        symbol = 'W'
                    elif signal.signal_type in (SignalType.CELLULAR_2G, SignalType.CELLULAR_3G,
                                                SignalType.CELLULAR_4G, SignalType.CELLULAR_5G):
                        symbol = 'C'
                    elif signal.signal_type == SignalType.DRONE:
                        symbol = 'D'
                    elif signal.signal_type == SignalType.BLUETOOTH or signal.signal_type == SignalType.BLUETOOTH_LE:
                        symbol = 'B'
                    else:
                        symbol = '·'
                        
                    grid[gy][gx] = symbol
                    
            elif signal.direction is not None and signal.distance_estimate:
                # Plot by direction/distance
                rad = math.radians(signal.direction)
                dist_scaled = (signal.distance_estimate / scale_meters) * (min(self.width, self.height) / 2)
                gx = int(cx + dist_scaled * math.sin(rad))
                gy = int(cy - dist_scaled * math.cos(rad))
                
                if 0 <= gx < self.width and 0 <= gy < self.height:
                    grid[gy][gx] = '·'
                    
        # Add border and compass
        lines = []
        lines.append("┌" + "─" * self.width + "┐ N")
        for i, row in enumerate(grid):
            prefix = "│"
            suffix = "│"
            if i == self.height // 2:
                suffix = "│ E"
            lines.append(prefix + "".join(row) + suffix)
        lines.append("└" + "─" * self.width + "┘ S")
        lines.append(" " * (self.width // 2) + "Scale: " + f"{scale_meters}m")
        
        # Legend
        lines.append("")
        lines.append("Legend: ◉=You  ⚠=Hostile  ?=Suspicious  W=WiFi  C=Cell  D=Drone  B=BT")
        
        return "\n".join(lines)


# Convenience function
def create_dashboard(hardware_controller=None) -> RFThreatDashboard:
    """Create and return a configured dashboard instance"""
    return RFThreatDashboard(hardware_controller)


if __name__ == "__main__":
    # Demo
    logging.basicConfig(level=logging.INFO)
    
    dashboard = RFThreatDashboard()
    
    # Add some test signals
    test_signals = [
        RFSignal(
            signal_id="wifi-001",
            signal_type=SignalType.WIFI_24,
            frequency=2.4e9,
            signal_strength=-65,
            timestamp=datetime.now(),
            identifier="AA:BB:CC:DD:EE:FF",
            name="HomeNetwork",
            latitude=37.7749,
            longitude=-122.4194
        ),
        RFSignal(
            signal_id="cell-001",
            signal_type=SignalType.CELLULAR_4G,
            frequency=700e6,
            signal_strength=-45,  # Suspiciously strong
            timestamp=datetime.now(),
            identifier="310-410-12345",
            name="Unknown Tower",
            latitude=37.7750,
            longitude=-122.4190
        ),
        RFSignal(
            signal_id="drone-001",
            signal_type=SignalType.DRONE,
            frequency=2.4e9,
            signal_strength=-55,
            timestamp=datetime.now(),
            device_type=DeviceType.DRONE,
            name="DJI Mavic",
            direction=45,
            distance_estimate=200
        )
    ]
    
    dashboard.set_position(37.7749, -122.4194)
    
    for sig in test_signals:
        dashboard.process_signal(sig)
        
    # Update stealth footprint
    dashboard.update_stealth_footprint(rf=0.1, network=0.0, device=0.2)
    dashboard.update_opsec_score(85)
    
    # Print terminal display
    print(dashboard.get_terminal_display())
    
    # Print threat map
    renderer = ThreatMapRenderer(width=50, height=20)
    print("\n" + renderer.render(
        list(dashboard.signal_tracker.signals.values()),
        37.7749, -122.4194, 
        scale_meters=300
    ))
