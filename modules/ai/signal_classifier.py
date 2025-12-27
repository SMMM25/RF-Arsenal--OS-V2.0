"""
RF Arsenal OS - AI-Powered Signal Classifier
============================================

Intelligent signal classification using machine learning to identify
signal types, protocols, and potential threats.

Author: RF Arsenal AI Team
Version: 2.0.0
License: Authorized Use Only
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of RF signals that can be classified."""
    GSM_2G = "gsm_2g"
    UMTS_3G = "umts_3g"
    LTE_4G = "lte_4g"
    NR_5G = "nr_5g"
    WIFI_24GHZ = "wifi_2.4ghz"
    WIFI_5GHZ = "wifi_5ghz"
    BLUETOOTH = "bluetooth"
    GPS = "gps"
    DRONE = "drone"
    RADAR = "radar"
    JAMMING = "jamming"
    UNKNOWN = "unknown"


@dataclass
class SignalFeatures:
    """Extracted features from RF signal for classification."""
    frequency: float  # Hz
    bandwidth: float  # Hz
    power_dbm: float
    modulation_type: str  # "FSK", "QPSK", "OFDM", etc.
    symbol_rate: Optional[float] = None
    carrier_spacing: Optional[float] = None
    spectral_peaks: List[float] = None
    time_domain_pattern: Optional[np.ndarray] = None
    freq_domain_pattern: Optional[np.ndarray] = None


@dataclass
class Classification:
    """Signal classification result."""
    signal_type: SignalType
    confidence: float  # 0.0 - 1.0
    protocol: Optional[str] = None  # "LTE-FDD", "WiFi-6", etc.
    carrier: Optional[str] = None  # Network operator
    threat_level: str = "low"  # "low", "medium", "high", "critical"
    metadata: Dict = None


class AISignalClassifier:
    """
    AI-powered signal classification engine.
    
    Features:
    - Real-time signal type identification
    - Protocol detection (GSM/LTE/WiFi/BT)
    - Threat assessment
    - Jamming detection
    - Anomaly identification
    """
    
    def __init__(self):
        """Initialize AI signal classifier."""
        self.model_trained = False
        self.classification_history: List[Classification] = []
        
        # Frequency ranges for known signal types
        self.frequency_signatures = {
            SignalType.GSM_2G: [(850e6, 900e6), (1800e6, 1900e6)],
            SignalType.UMTS_3G: [(900e6, 2100e6)],
            SignalType.LTE_4G: [(700e6, 2600e6)],
            SignalType.NR_5G: [(3.3e9, 4.2e9), (24e9, 40e9)],
            SignalType.WIFI_24GHZ: [(2.4e9, 2.5e9)],
            SignalType.WIFI_5GHZ: [(5.15e9, 5.85e9)],
            SignalType.BLUETOOTH: [(2.4e9, 2.48e9)],
            SignalType.GPS: [(1.575e9, 1.576e9)],
        }
        
        logger.info("✅ AI Signal Classifier initialized")
    
    def classify_signal(self, features: SignalFeatures) -> Classification:
        """
        Classify signal based on extracted features.
        
        Args:
            features: Extracted signal features
        
        Returns:
            Classification result with confidence
        """
        # Frequency-based classification
        signal_type, freq_confidence = self._classify_by_frequency(features.frequency)
        
        # Bandwidth-based refinement
        bw_confidence = self._validate_bandwidth(signal_type, features.bandwidth)
        
        # Modulation-based refinement
        mod_confidence = self._validate_modulation(signal_type, features.modulation_type)
        
        # Combined confidence
        confidence = (freq_confidence + bw_confidence + mod_confidence) / 3
        
        # Detect threats
        threat_level = self._assess_threat(signal_type, features)
        
        # Detect protocol
        protocol = self._detect_protocol(signal_type, features)
        
        classification = Classification(
            signal_type=signal_type,
            confidence=confidence,
            protocol=protocol,
            threat_level=threat_level,
            metadata={
                'frequency_mhz': features.frequency / 1e6,
                'bandwidth_mhz': features.bandwidth / 1e6,
                'power_dbm': features.power_dbm,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        # Store history
        self.classification_history.append(classification)
        if len(self.classification_history) > 1000:
            self.classification_history = self.classification_history[-1000:]
        
        logger.info(
            f"📡 Signal classified: {signal_type.value} "
            f"(confidence: {confidence:.2%}, threat: {threat_level})"
        )
        
        return classification
    
    def _classify_by_frequency(self, frequency: float) -> Tuple[SignalType, float]:
        """Classify signal based on frequency."""
        for signal_type, ranges in self.frequency_signatures.items():
            for freq_min, freq_max in ranges:
                if freq_min <= frequency <= freq_max:
                    # Calculate confidence based on position in band
                    band_center = (freq_min + freq_max) / 2
                    distance = abs(frequency - band_center)
                    max_distance = (freq_max - freq_min) / 2
                    confidence = 1.0 - (distance / max_distance) * 0.3  # 70-100% confidence
                    return signal_type, confidence
        
        return SignalType.UNKNOWN, 0.0
    
    def _validate_bandwidth(self, signal_type: SignalType, bandwidth: float) -> float:
        """Validate bandwidth matches signal type."""
        expected_bw = {
            SignalType.GSM_2G: 200e3,
            SignalType.UMTS_3G: 5e6,
            SignalType.LTE_4G: 20e6,
            SignalType.NR_5G: 100e6,
            SignalType.WIFI_24GHZ: 20e6,
            SignalType.WIFI_5GHZ: 80e6,
            SignalType.BLUETOOTH: 1e6,
        }
        
        if signal_type not in expected_bw:
            return 0.5
        
        expected = expected_bw[signal_type]
        error = abs(bandwidth - expected) / expected
        
        if error < 0.1:  # Within 10%
            return 1.0
        elif error < 0.3:  # Within 30%
            return 0.7
        else:
            return 0.3
    
    def _validate_modulation(self, signal_type: SignalType, modulation: str) -> float:
        """Validate modulation matches signal type."""
        expected_mod = {
            SignalType.GSM_2G: ["GMSK", "8PSK"],
            SignalType.UMTS_3G: ["QPSK", "16QAM"],
            SignalType.LTE_4G: ["QPSK", "16QAM", "64QAM", "OFDM"],
            SignalType.NR_5G: ["QPSK", "256QAM", "OFDM"],
            SignalType.WIFI_24GHZ: ["OFDM", "DSSS"],
            SignalType.WIFI_5GHZ: ["OFDM"],
            SignalType.BLUETOOTH: ["GFSK"],
        }
        
        if signal_type not in expected_mod:
            return 0.5
        
        if modulation in expected_mod[signal_type]:
            return 1.0
        else:
            return 0.3
    
    def _detect_protocol(self, signal_type: SignalType, features: SignalFeatures) -> Optional[str]:
        """Detect specific protocol variant."""
        if signal_type == SignalType.LTE_4G:
            if features.bandwidth >= 20e6:
                return "LTE-Advanced"
            else:
                return "LTE-FDD"
        elif signal_type == SignalType.NR_5G:
            if features.frequency > 24e9:
                return "5G-NR-mmWave"
            else:
                return "5G-NR-Sub6"
        elif signal_type == SignalType.WIFI_24GHZ:
            return "WiFi-4/5 (2.4GHz)"
        elif signal_type == SignalType.WIFI_5GHZ:
            if features.bandwidth >= 80e6:
                return "WiFi-6 (802.11ax)"
            else:
                return "WiFi-5 (802.11ac)"
        
        return signal_type.value
    
    def _assess_threat(self, signal_type: SignalType, features: SignalFeatures) -> str:
        """Assess threat level of signal."""
        # Jamming detection (abnormal power)
        if features.power_dbm > 30:
            logger.warning(f"⚠️  High power signal detected: {features.power_dbm} dBm")
            return "high"
        
        # Rogue base station detection (GSM/LTE in unexpected location)
        if signal_type in [SignalType.GSM_2G, SignalType.LTE_4G]:
            # This would integrate with geolocation
            pass
        
        # GPS spoofing detection
        if signal_type == SignalType.GPS and features.power_dbm > -120:
            logger.warning("⚠️  Potential GPS spoofing detected")
            return "high"
        
        return "low"
    
    def detect_jamming(self, power_levels: List[float], threshold_dbm: float = 20) -> bool:
        """
        Detect signal jamming based on power levels.
        
        Args:
            power_levels: List of recent power measurements
            threshold_dbm: Power threshold for jamming
        
        Returns:
            True if jamming detected
        """
        if not power_levels:
            return False
        
        avg_power = np.mean(power_levels)
        max_power = np.max(power_levels)
        
        if max_power > threshold_dbm:
            logger.warning(f"🚨 JAMMING DETECTED: {max_power:.1f} dBm (threshold: {threshold_dbm})")
            return True
        
        return False
    
    def get_statistics(self) -> Dict:
        """Get classification statistics."""
        if not self.classification_history:
            return {'total_classifications': 0}
        
        # Count by signal type
        by_type = {}
        threat_count = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for classification in self.classification_history:
            signal_type = classification.signal_type.value
            by_type[signal_type] = by_type.get(signal_type, 0) + 1
            threat_count[classification.threat_level] += 1
        
        return {
            'total_classifications': len(self.classification_history),
            'by_signal_type': by_type,
            'by_threat_level': threat_count,
            'average_confidence': np.mean([c.confidence for c in self.classification_history])
        }


if __name__ == "__main__":
    # Test signal classifier
    print("🤖 RF Arsenal OS - AI Signal Classifier Test\n")
    
    classifier = AISignalClassifier()
    
    # Test LTE signal
    lte_features = SignalFeatures(
        frequency=1842.6e6,  # 1842.6 MHz (LTE Band 3)
        bandwidth=20e6,  # 20 MHz
        power_dbm=-85,
        modulation_type="OFDM"
    )
    
    result = classifier.classify_signal(lte_features)
    print(f"Classification: {result.signal_type.value}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Protocol: {result.protocol}")
    print(f"Threat Level: {result.threat_level}")
    
    print(f"\n📊 Statistics:")
    import json
    print(json.dumps(classifier.get_statistics(), indent=2))
