"""
RF Arsenal OS - AI Anomaly Detector
===================================

Machine learning-based anomaly detection for RF signals and network behavior.
Identifies suspicious patterns, rogue devices, and potential security threats.

Author: RF Arsenal AI Team
Version: 2.0.0
License: Authorized Use Only
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    ROGUE_BASE_STATION = "rogue_base_station"
    IMSI_CATCHER = "imsi_catcher"
    GPS_SPOOFING = "gps_spoofing"
    SIGNAL_JAMMING = "signal_jamming"
    UNUSUAL_FREQUENCY = "unusual_frequency"
    POWER_ANOMALY = "power_anomaly"
    TIMING_ANOMALY = "timing_anomaly"
    PROTOCOL_VIOLATION = "protocol_violation"
    UNKNOWN_DEVICE = "unknown_device"


@dataclass
class Anomaly:
    """Detected anomaly with details."""
    anomaly_type: AnomalyType
    severity: str  # "low", "medium", "high", "critical"
    confidence: float  # 0.0 - 1.0
    description: str
    timestamp: datetime
    location: Optional[Tuple[float, float]] = None  # (lat, lon)
    frequency: Optional[float] = None
    metadata: Dict = None


class AIAnomalyDetector:
    """
    AI-powered anomaly detection for RF Arsenal.
    
    Features:
    - Rogue base station detection
    - IMSI catcher identification
    - GPS spoofing detection
    - Signal jamming alerts
    - Behavioral analysis
    - Pattern recognition
    """
    
    def __init__(self, history_window_seconds: int = 300):
        """
        Initialize anomaly detector.
        
        Args:
            history_window_seconds: Time window for historical analysis
        """
        self.history_window = history_window_seconds
        self.anomalies: List[Anomaly] = []
        
        # Historical data for baseline
        self.power_history = deque(maxlen=1000)
        self.frequency_history = deque(maxlen=1000)
        self.cell_id_history = deque(maxlen=100)
        self.timing_history = deque(maxlen=1000)
        
        # Known cell towers (for rogue detection)
        self.known_cell_ids: set = set()
        
        # Baseline statistics
        self.baseline = {
            'power_mean': -85.0,
            'power_std': 10.0,
            'freq_mean': 1800e6,
            'freq_std': 100e6
        }
        
        logger.info(f"✅ AI Anomaly Detector initialized (window: {history_window_seconds}s)")
    
    def detect_anomalies(self, 
                        signal_power: float,
                        frequency: float,
                        cell_id: Optional[str] = None,
                        timing_advance: Optional[int] = None) -> List[Anomaly]:
        """
        Detect anomalies in current signal parameters.
        
        Args:
            signal_power: Signal power in dBm
            frequency: Frequency in Hz
            cell_id: Cell tower ID (optional)
            timing_advance: Timing advance value (optional)
        
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Update history
        self.power_history.append(signal_power)
        self.frequency_history.append(frequency)
        if cell_id:
            self.cell_id_history.append(cell_id)
        if timing_advance is not None:
            self.timing_history.append(timing_advance)
        
        # Check for power anomalies
        power_anomaly = self._detect_power_anomaly(signal_power)
        if power_anomaly:
            anomalies.append(power_anomaly)
        
        # Check for frequency anomalies
        freq_anomaly = self._detect_frequency_anomaly(frequency)
        if freq_anomaly:
            anomalies.append(freq_anomaly)
        
        # Check for rogue base station
        if cell_id:
            rogue_anomaly = self._detect_rogue_base_station(cell_id, signal_power)
            if rogue_anomaly:
                anomalies.append(rogue_anomaly)
        
        # Check for IMSI catcher indicators
        imsi_catcher_anomaly = self._detect_imsi_catcher(cell_id, signal_power)
        if imsi_catcher_anomaly:
            anomalies.append(imsi_catcher_anomaly)
        
        # Check for jamming
        jamming_anomaly = self._detect_jamming()
        if jamming_anomaly:
            anomalies.append(jamming_anomaly)
        
        # Store detected anomalies
        for anomaly in anomalies:
            self.anomalies.append(anomaly)
            logger.warning(f"🚨 ANOMALY DETECTED: {anomaly.anomaly_type.value} (severity: {anomaly.severity})")
        
        # Keep only recent anomalies
        cutoff = datetime.now() - timedelta(seconds=self.history_window)
        self.anomalies = [a for a in self.anomalies if a.timestamp > cutoff]
        
        return anomalies
    
    def _detect_power_anomaly(self, power: float) -> Optional[Anomaly]:
        """Detect abnormal signal power levels."""
        if len(self.power_history) < 10:
            return None  # Need baseline
        
        mean_power = np.mean(list(self.power_history))
        std_power = np.std(list(self.power_history))
        
        # Z-score anomaly detection
        z_score = abs((power - mean_power) / (std_power + 1e-6))
        
        if z_score > 3.0:  # 3 sigma threshold
            severity = "high" if z_score > 5.0 else "medium"
            confidence = min(z_score / 5.0, 1.0)
            
            return Anomaly(
                anomaly_type=AnomalyType.POWER_ANOMALY,
                severity=severity,
                confidence=confidence,
                description=f"Signal power {power:.1f} dBm deviates {z_score:.1f}σ from baseline {mean_power:.1f} dBm",
                timestamp=datetime.now(),
                metadata={'z_score': z_score, 'mean': mean_power, 'std': std_power}
            )
        
        return None
    
    def _detect_frequency_anomaly(self, frequency: float) -> Optional[Anomaly]:
        """Detect unusual frequency usage."""
        if len(self.frequency_history) < 10:
            return None
        
        # Check if frequency is significantly different from recent history
        recent_freqs = list(self.frequency_history)[-50:]
        
        if abs(frequency - np.mean(recent_freqs)) > 100e6:  # 100 MHz difference
            return Anomaly(
                anomaly_type=AnomalyType.UNUSUAL_FREQUENCY,
                severity="medium",
                confidence=0.7,
                description=f"Unusual frequency {frequency/1e6:.2f} MHz detected",
                timestamp=datetime.now(),
                frequency=frequency
            )
        
        return None
    
    def _detect_rogue_base_station(self, cell_id: str, power: float) -> Optional[Anomaly]:
        """Detect rogue/unknown base stations."""
        # Check if cell ID is known
        if cell_id and cell_id not in self.known_cell_ids:
            # New cell tower - could be rogue or legitimate
            if power > -70:  # Very strong signal for unknown cell
                return Anomaly(
                    anomaly_type=AnomalyType.ROGUE_BASE_STATION,
                    severity="high",
                    confidence=0.8,
                    description=f"Unknown cell tower {cell_id} with unusually strong signal ({power:.1f} dBm)",
                    timestamp=datetime.now(),
                    metadata={'cell_id': cell_id, 'power': power}
                )
            else:
                # Add to known cells with low confidence
                self.known_cell_ids.add(cell_id)
        
        return None
    
    def _detect_imsi_catcher(self, cell_id: Optional[str], power: float) -> Optional[Anomaly]:
        """Detect IMSI catcher indicators."""
        if not cell_id:
            return None
        
        # IMSI catcher indicators:
        # 1. Strong signal + new cell ID
        # 2. Rapid cell ID changes
        # 3. Downgrade to 2G/3G
        
        if len(self.cell_id_history) < 5:
            return None
        
        recent_cells = list(self.cell_id_history)[-10:]
        unique_cells = len(set(recent_cells))
        
        # Rapid cell ID switching (indicator of IMSI catcher)
        if unique_cells > 5 and power > -80:
            return Anomaly(
                anomaly_type=AnomalyType.IMSI_CATCHER,
                severity="critical",
                confidence=0.85,
                description=f"Potential IMSI catcher: {unique_cells} unique cells in short time",
                timestamp=datetime.now(),
                metadata={'unique_cells': unique_cells, 'cell_id': cell_id}
            )
        
        return None
    
    def _detect_jamming(self) -> Optional[Anomaly]:
        """Detect signal jamming."""
        if len(self.power_history) < 20:
            return None
        
        recent_powers = list(self.power_history)[-20:]
        
        # Jamming indicators:
        # 1. Sustained high power
        # 2. Low variance (constant jamming)
        
        mean_power = np.mean(recent_powers)
        std_power = np.std(recent_powers)
        
        if mean_power > -60 and std_power < 2.0:  # High power + low variance
            return Anomaly(
                anomaly_type=AnomalyType.SIGNAL_JAMMING,
                severity="critical",
                confidence=0.9,
                description=f"Signal jamming detected: sustained power {mean_power:.1f} dBm",
                timestamp=datetime.now(),
                metadata={'mean_power': mean_power, 'std': std_power}
            )
        
        return None
    
    def add_known_cell(self, cell_id: str):
        """Add a cell ID to the known (legitimate) list."""
        self.known_cell_ids.add(cell_id)
        logger.info(f"Added known cell: {cell_id}")
    
    def load_known_cells(self, cell_ids: List[str]):
        """Load list of known legitimate cell towers."""
        self.known_cell_ids.update(cell_ids)
        logger.info(f"Loaded {len(cell_ids)} known cell towers")
    
    def get_statistics(self) -> Dict:
        """Get anomaly detection statistics."""
        if not self.anomalies:
            return {'total_anomalies': 0}
        
        # Count by type
        by_type = {}
        by_severity = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for anomaly in self.anomalies:
            anomaly_type = anomaly.anomaly_type.value
            by_type[anomaly_type] = by_type.get(anomaly_type, 0) + 1
            by_severity[anomaly.severity] += 1
        
        # Critical anomalies in last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_critical = [
            a for a in self.anomalies 
            if a.timestamp > one_hour_ago and a.severity == "critical"
        ]
        
        return {
            'total_anomalies': len(self.anomalies),
            'by_type': by_type,
            'by_severity': by_severity,
            'critical_last_hour': len(recent_critical),
            'average_confidence': np.mean([a.confidence for a in self.anomalies])
        }
    
    def generate_report(self) -> str:
        """Generate human-readable anomaly report."""
        if not self.anomalies:
            return "✅ No anomalies detected"
        
        stats = self.get_statistics()
        
        report = [
            "=" * 60,
            "🚨 RF ARSENAL OS - ANOMALY DETECTION REPORT",
            "=" * 60,
            "",
            f"Total Anomalies: {stats['total_anomalies']}",
            f"Critical (Last Hour): {stats['critical_last_hour']}",
            "",
            "📊 By Severity:",
        ]
        
        for severity, count in stats['by_severity'].items():
            if count > 0:
                report.append(f"  {severity.upper()}: {count}")
        
        report.extend([
            "",
            "🔍 By Type:",
        ])
        
        for anomaly_type, count in stats['by_type'].items():
            report.append(f"  {anomaly_type}: {count}")
        
        report.extend([
            "",
            "⚠️  Recent Critical Anomalies:",
        ])
        
        critical_anomalies = [a for a in self.anomalies[-10:] if a.severity == "critical"]
        for anomaly in critical_anomalies:
            report.append(f"  [{anomaly.timestamp.strftime('%H:%M:%S')}] {anomaly.description}")
        
        report.append("=" * 60)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test anomaly detector
    print("🤖 RF Arsenal OS - AI Anomaly Detector Test\n")
    
    detector = AIAnomalyDetector()
    
    # Simulate normal signals
    print("Simulating normal signals...")
    for i in range(20):
        detector.detect_anomalies(-85 + np.random.randn() * 5, 1842.6e6)
    
    # Simulate anomaly
    print("\nSimulating power anomaly...")
    anomalies = detector.detect_anomalies(-50, 1842.6e6)  # Very high power
    
    for anomaly in anomalies:
        print(f"  ⚠️  {anomaly.description}")
    
    print(f"\n{detector.generate_report()}")
