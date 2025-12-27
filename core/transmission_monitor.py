"""
RF Arsenal OS - Transmission Monitor & RF Emission Logging
==========================================================

CRITICAL SECURITY MODULE: Monitors and logs ALL RF transmissions to detect
stealth violations and ensure operational security.

Author: RF Arsenal Security Team
Version: 2.0.0
License: Authorized Use Only
"""

import logging
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from threading import Lock
from enum import Enum
import json

logger = logging.getLogger(__name__)


class TransmissionType(Enum):
    """Types of RF transmissions."""
    CELLULAR_2G = "cellular_2g"
    CELLULAR_3G = "cellular_3g"
    CELLULAR_4G = "cellular_4g"
    CELLULAR_5G = "cellular_5g"
    WIFI = "wifi"
    BLUETOOTH = "bluetooth"
    GPS = "gps"
    DRONE = "drone"
    CUSTOM = "custom"


class TransmissionEvent:
    """Represents a single RF transmission event."""
    
    def __init__(self,
                 tx_type: TransmissionType,
                 frequency: float,
                 power_dbm: float,
                 duration_ms: float,
                 source_module: str,
                 data_size_bytes: int = 0,
                 target: Optional[str] = None,
                 metadata: Optional[dict] = None):
        """
        Initialize transmission event.
        
        Args:
            tx_type: Type of transmission
            frequency: Frequency in Hz
            power_dbm: Transmission power in dBm
            duration_ms: Duration in milliseconds
            source_module: Module that initiated transmission
            data_size_bytes: Size of transmitted data
            target: Target identifier (anonymized)
            metadata: Additional metadata
        """
        self.timestamp = datetime.now()
        self.tx_type = tx_type
        self.frequency = frequency
        self.power_dbm = power_dbm
        self.duration_ms = duration_ms
        self.source_module = source_module
        self.data_size_bytes = data_size_bytes
        self.target = target
        self.metadata = metadata or {}
    
    def to_dict(self) -> dict:
        """Convert event to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'type': self.tx_type.value,
            'frequency_mhz': self.frequency / 1e6,
            'power_dbm': self.power_dbm,
            'duration_ms': self.duration_ms,
            'source_module': self.source_module,
            'data_size_bytes': self.data_size_bytes,
            'target': self.target,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"[{self.timestamp.strftime('%H:%M:%S')}] "
            f"{self.tx_type.value} @ {self.frequency/1e6:.2f}MHz "
            f"({self.power_dbm}dBm, {self.duration_ms}ms) "
            f"from {self.source_module}"
        )


class TransmissionMonitor:
    """
    Monitors and logs all RF transmissions across the system.
    
    Features:
    - Real-time transmission tracking
    - Stealth violation detection
    - Transmission statistics and analytics
    - Export to JSON/CSV for forensic analysis
    - Alert system for unexpected transmissions
    """
    
    def __init__(self, 
                 alert_on_transmission: bool = True,
                 max_log_size: int = 10000):
        """
        Initialize transmission monitor.
        
        Args:
            alert_on_transmission: Alert on ANY transmission (default: True)
            max_log_size: Maximum number of events to keep in memory
        """
        self.alert_on_transmission = alert_on_transmission
        self.max_log_size = max_log_size
        self._transmission_log: List[TransmissionEvent] = []
        self._lock = Lock()
        self._alert_callbacks = []
        self._total_transmissions = 0
        self._total_tx_time_ms = 0.0
        
        logger.info(f"📡 Transmission Monitor initialized (Alert mode: {alert_on_transmission})")
    
    def log_transmission(self, event: TransmissionEvent) -> bool:
        """
        Log a transmission event and check for violations.
        
        Args:
            event: TransmissionEvent to log
        
        Returns:
            True if transmission was logged, False if rejected
        """
        with self._lock:
            # Add to log
            self._transmission_log.append(event)
            self._total_transmissions += 1
            self._total_tx_time_ms += event.duration_ms
            
            # Trim log if needed
            if len(self._transmission_log) > self.max_log_size:
                self._transmission_log = self._transmission_log[-self.max_log_size:]
            
            # Alert if configured
            if self.alert_on_transmission:
                alert_msg = f"⚠️  RF TRANSMISSION DETECTED: {event}"
                logger.warning(alert_msg)
                
                # Trigger callbacks
                for callback in self._alert_callbacks:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")
            
            logger.info(f"📡 Transmission logged: {event}")
            return True
    
    def register_alert_callback(self, callback):
        """
        Register callback for transmission alerts.
        
        Args:
            callback: Function(TransmissionEvent) to call on transmission
        """
        self._alert_callbacks.append(callback)
        logger.info(f"✅ Alert callback registered ({len(self._alert_callbacks)} total)")
    
    def get_transmissions(self, 
                         since: Optional[datetime] = None,
                         tx_type: Optional[TransmissionType] = None,
                         source_module: Optional[str] = None) -> List[TransmissionEvent]:
        """
        Get filtered transmission log.
        
        Args:
            since: Only return transmissions after this time
            tx_type: Filter by transmission type
            source_module: Filter by source module
        
        Returns:
            List of matching TransmissionEvent objects
        """
        with self._lock:
            filtered = self._transmission_log.copy()
            
            if since:
                filtered = [t for t in filtered if t.timestamp >= since]
            
            if tx_type:
                filtered = [t for t in filtered if t.tx_type == tx_type]
            
            if source_module:
                filtered = [t for t in filtered if t.source_module == source_module]
            
            return filtered
    
    def get_statistics(self, window_minutes: Optional[int] = None) -> dict:
        """
        Get transmission statistics.
        
        Args:
            window_minutes: Only analyze last N minutes (None = all time)
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            transmissions = self._transmission_log
            
            if window_minutes:
                cutoff = datetime.now() - timedelta(minutes=window_minutes)
                transmissions = [t for t in transmissions if t.timestamp >= cutoff]
            
            if not transmissions:
                return {
                    'total_transmissions': 0,
                    'window_minutes': window_minutes,
                    'message': 'No transmissions recorded'
                }
            
            # Calculate statistics
            total_power = sum(t.power_dbm for t in transmissions)
            total_duration = sum(t.duration_ms for t in transmissions)
            
            # Group by type
            by_type = {}
            for t in transmissions:
                type_name = t.tx_type.value
                if type_name not in by_type:
                    by_type[type_name] = {'count': 0, 'duration_ms': 0}
                by_type[type_name]['count'] += 1
                by_type[type_name]['duration_ms'] += t.duration_ms
            
            # Group by module
            by_module = {}
            for t in transmissions:
                if t.source_module not in by_module:
                    by_module[t.source_module] = 0
                by_module[t.source_module] += 1
            
            return {
                'window_minutes': window_minutes or 'all_time',
                'total_transmissions': len(transmissions),
                'total_duration_seconds': total_duration / 1000,
                'average_power_dbm': total_power / len(transmissions),
                'by_type': by_type,
                'by_module': by_module,
                'first_transmission': transmissions[0].timestamp.isoformat(),
                'last_transmission': transmissions[-1].timestamp.isoformat()
            }
    
    def export_to_json(self, filepath: str, since: Optional[datetime] = None):
        """
        Export transmission log to JSON file.
        
        Args:
            filepath: Output file path
            since: Only export transmissions after this time
        """
        transmissions = self.get_transmissions(since=since)
        
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_events': len(transmissions),
            'statistics': self.get_statistics(),
            'transmissions': [t.to_dict() for t in transmissions]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"📁 Exported {len(transmissions)} transmissions to {filepath}")
    
    def generate_report(self, window_minutes: Optional[int] = None) -> str:
        """
        Generate human-readable transmission report.
        
        Args:
            window_minutes: Analyze last N minutes (None = all time)
        
        Returns:
            Formatted report string
        """
        stats = self.get_statistics(window_minutes)
        
        if stats.get('total_transmissions', 0) == 0:
            return (
                "=" * 60 + "\n"
                "📡 RF TRANSMISSION MONITOR - NO TRANSMISSIONS DETECTED\n"
                "=" * 60 + "\n\n"
                "✅ STEALTH STATUS: MAXIMUM (No RF emissions detected)\n"
                "=" * 60
            )
        
        report = [
            "=" * 60,
            "📡 RF TRANSMISSION MONITOR REPORT",
            "=" * 60,
            "",
            f"⚠️  WARNING: {stats['total_transmissions']} TRANSMISSIONS DETECTED",
            f"Time Window: {stats['window_minutes']}",
            f"Total Duration: {stats['total_duration_seconds']:.2f} seconds",
            f"Average Power: {stats['average_power_dbm']:.1f} dBm",
            "",
            "📊 Transmissions by Type:",
        ]
        
        for tx_type, data in stats['by_type'].items():
            report.append(
                f"  {tx_type}: {data['count']} transmissions "
                f"({data['duration_ms']/1000:.2f}s total)"
            )
        
        report.extend([
            "",
            "🔧 Transmissions by Module:",
        ])
        
        for module, count in stats['by_module'].items():
            report.append(f"  {module}: {count} transmissions")
        
        report.extend([
            "",
            "=" * 60,
            "⚠️  OPERATIONAL SECURITY NOTICE:",
            "RF transmissions may compromise stealth operations.",
            "Review source modules and disable active features if needed.",
            "=" * 60
        ])
        
        return "\n".join(report)
    
    def clear_log(self):
        """Clear transmission log (use with caution)."""
        with self._lock:
            count = len(self._transmission_log)
            self._transmission_log.clear()
            logger.warning(f"🗑️  Transmission log cleared ({count} events removed)")


# Singleton instance
_monitor_instance: Optional[TransmissionMonitor] = None


def get_transmission_monitor() -> TransmissionMonitor:
    """Get singleton transmission monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = TransmissionMonitor(alert_on_transmission=True)
    return _monitor_instance


def log_transmission(event: TransmissionEvent):
    """Quick transmission logging using singleton."""
    get_transmission_monitor().log_transmission(event)


if __name__ == "__main__":
    # Test transmission monitoring
    print("📡 RF Arsenal OS - Transmission Monitor Test\n")
    
    monitor = TransmissionMonitor(alert_on_transmission=True)
    
    # Log some test transmissions
    print("Logging test transmissions...\n")
    
    event1 = TransmissionEvent(
        tx_type=TransmissionType.CELLULAR_4G,
        frequency=1842.6e6,
        power_dbm=23.0,
        duration_ms=150,
        source_module="hackrf_controller"
    )
    monitor.log_transmission(event1)
    
    event2 = TransmissionEvent(
        tx_type=TransmissionType.WIFI,
        frequency=2437e6,
        power_dbm=20.0,
        duration_ms=50,
        source_module="wifi_controller"
    )
    monitor.log_transmission(event2)
    
    print("\n" + monitor.generate_report())
