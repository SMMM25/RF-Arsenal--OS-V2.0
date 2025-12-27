#!/usr/bin/env python3
"""
RF Arsenal OS - ML Device Fingerprinting
Real-time device identification using RF signatures
"""

import logging
import numpy as np
import pickle
import json
import hashlib
import hmac
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import time

# Security: Model file integrity verification key
# In production, this should be loaded from secure storage
_MODEL_SIGNING_KEY = b'rf_arsenal_model_integrity_v1'

@dataclass
class DeviceProfile:
    """Identified device information"""
    manufacturer: str
    model: str
    os: str
    baseband_version: str
    confidence: float
    timestamp: str
    
    def to_dict(self) -> dict:
        return asdict(self)

@dataclass 
class RFSignature:
    """Complete RF signature for device fingerprinting"""
    # Timing characteristics
    timing_advance: float           # GSM timing advance (microseconds)
    frame_timing_offset: float      # Frame synchronization offset
    preamble_duration: float        # Connection preamble length
    
    # Power characteristics  
    tx_power_profile: List[float]   # Transmit power over time
    power_ramp_rate: float          # Power ramp-up speed
    power_control_accuracy: float   # How accurately device follows power control
    
    # Modulation characteristics
    evm_rms: float                  # Error Vector Magnitude (%)
    phase_error: float              # Phase error (degrees)
    frequency_error: float          # Frequency offset (Hz)
    iq_imbalance: float            # I/Q imbalance (dB)
    
    # Protocol behavior
    random_access_pattern: str      # RACH preamble selection pattern
    retransmission_strategy: str    # ARQ/HARQ behavior
    buffer_status_reporting: str    # BSR frequency pattern
    
    # Chipset-specific features
    clock_drift: float              # Crystal oscillator drift (ppm)
    spurious_emissions: List[float] # Frequency/amplitude of spurious
    filter_rolloff: float           # Transmit filter characteristics
    dac_linearity: float           # DAC non-linearity metric
    
    # Network interaction
    attach_procedure_timing: float  # Time to complete attach (ms)
    handover_latency: float        # Handover completion time (ms)
    
    # Metadata
    frequency: float               # Operating frequency (MHz)
    technology: str               # 2G/3G/4G/5G
    timestamp: str
    imsi: Optional[str] = None
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert signature to feature vector for ML"""
        features = [
            self.timing_advance,
            self.frame_timing_offset,
            self.preamble_duration,
            np.mean(self.tx_power_profile) if self.tx_power_profile else 0,
            np.std(self.tx_power_profile) if self.tx_power_profile else 0,
            self.power_ramp_rate,
            self.power_control_accuracy,
            self.evm_rms,
            self.phase_error,
            self.frequency_error,
            self.iq_imbalance,
            hash(self.random_access_pattern) % 1000,  # Hash to numeric
            hash(self.retransmission_strategy) % 1000,
            hash(self.buffer_status_reporting) % 1000,
            self.clock_drift,
            np.mean(self.spurious_emissions) if self.spurious_emissions else 0,
            len(self.spurious_emissions),
            self.filter_rolloff,
            self.dac_linearity,
            self.attach_procedure_timing,
            self.handover_latency,
            self.frequency,
            hash(self.technology) % 100
        ]
        return np.array(features)

class MLDeviceFingerprinting:
    """
    Machine Learning-based device fingerprinting
    Uses RF signatures to identify device manufacturer, model, and baseband
    """
    
    def __init__(self, model_path: Optional[str] = None, passive_mode: bool = True, anonymize_logs: bool = True):
        """
        Initialize ML Device Fingerprinting engine
        
        Args:
            model_path: Path to pre-trained model file
            passive_mode: If True, enforce receive-only operation (STEALTH MODE)
            anonymize_logs: If True, hash all IMSI/IMEI identifiers for privacy
        """
        self.logger = logging.getLogger('DeviceFingerprinting')
        self.model_path = model_path or "data/ml_models/device_fingerprinting.pkl"
        
        # 🔒 STEALTH & PRIVACY SETTINGS
        self.passive_mode = passive_mode
        self.anonymize_logs = anonymize_logs
        self.transmit_enabled = not passive_mode
        self.active_probing_enabled = not passive_mode
        
        # ML model (will use RandomForest or Neural Network)
        self.model = None
        self.label_encoder = {}
        self.feature_scaler = None
        
        # Training data storage
        self.training_signatures: List[RFSignature] = []
        self.training_labels: List[DeviceProfile] = []
        
        # Real-time identification cache (uses anonymized keys if enabled)
        self.identified_devices: Dict[str, DeviceProfile] = {}  # IMSI_hash -> Profile
        
        # Log stealth mode status
        if self.passive_mode:
            self.logger.warning("🔒 STEALTH MODE ENABLED: Passive-only operation (no transmission)")
        else:
            self.logger.warning("⚠️  ACTIVE MODE: Transmission enabled (may be detectable!)")
        
        if self.anonymize_logs:
            self.logger.info("🔐 PRIVACY MODE: Identifier anonymization enabled")
        
        self._load_model()
    
    def _load_model(self):
        """
        Load pre-trained model from disk with integrity verification.
        
        SECURITY: pickle.load() can execute arbitrary code. We mitigate this by:
        1. Verifying HMAC signature before deserializing
        2. Only loading from trusted, local model files
        3. Model files should only be created by this system
        """
        try:
            model_path = Path(self.model_path)
            sig_path = Path(str(self.model_path) + '.sig')
            
            if not model_path.exists():
                self.logger.warning("No pre-trained model found - training required")
                return
                
            with open(model_path, 'rb') as f:
                model_bytes = f.read()
            
            # Verify integrity signature if it exists
            if sig_path.exists():
                with open(sig_path, 'r') as f:
                    stored_sig = f.read().strip()
                computed_sig = hmac.new(_MODEL_SIGNING_KEY, model_bytes, hashlib.sha256).hexdigest()
                if not hmac.compare_digest(stored_sig, computed_sig):
                    self.logger.error("❌ Model integrity check FAILED - file may be tampered")
                    self.logger.error("   Refusing to load potentially malicious model")
                    return
                self.logger.info("✅ Model integrity verified")
            else:
                self.logger.warning("⚠️  No model signature found - loading unsigned model")
                self.logger.warning("   Consider re-saving model to generate signature")
            
            # Load the verified model
            model_data = pickle.loads(model_bytes)
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.feature_scaler = model_data['scaler']
            self.logger.info(f"Loaded ML model from {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
    
    def _save_model(self):
        """
        Save trained model to disk with integrity signature.
        
        SECURITY: Generates HMAC signature for integrity verification on load.
        """
        try:
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            model_data = {
                'model': self.model,
                'label_encoder': self.label_encoder,
                'scaler': self.feature_scaler,
                'timestamp': datetime.now().isoformat()
            }
            
            # Serialize model
            model_bytes = pickle.dumps(model_data)
            
            # Generate integrity signature
            signature = hmac.new(_MODEL_SIGNING_KEY, model_bytes, hashlib.sha256).hexdigest()
            
            # Write model file
            with open(self.model_path, 'wb') as f:
                f.write(model_bytes)
            
            # Write signature file
            sig_path = str(self.model_path) + '.sig'
            with open(sig_path, 'w') as f:
                f.write(signature)
            
            self.logger.info(f"Model saved to {self.model_path}")
            self.logger.info(f"Integrity signature saved to {sig_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
    
    def add_training_data(self, signature: RFSignature, profile: DeviceProfile):
        """Add labeled training data"""
        self.training_signatures.append(signature)
        self.training_labels.append(profile)
        self.logger.debug(f"Added training sample: {profile.manufacturer} {profile.model}")
    
    def train_model(self, validation_split: float = 0.2):
        """
        Train the ML model on collected signatures
        Returns: (accuracy, training_time)
        """
        if len(self.training_signatures) < 10:
            self.logger.error("Insufficient training data (need at least 10 samples)")
            return 0.0, 0.0
        
        start_time = time.time()
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.metrics import accuracy_score, classification_report
            
            # Convert signatures to feature vectors
            X = np.array([sig.to_feature_vector() for sig in self.training_signatures])
            
            # Encode labels (manufacturer, model, OS)
            y_manufacturer = [label.manufacturer for label in self.training_labels]
            y_model = [label.model for label in self.training_labels]
            y_os = [label.os for label in self.training_labels]
            
            # Create composite label for multi-target classification
            y_composite = [f"{m}|{mo}|{os}" for m, mo, os in 
                          zip(y_manufacturer, y_model, y_os)]
            
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_composite)
            self.label_encoder = le
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.feature_scaler = scaler
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=validation_split, random_state=42
            )
            
            # Train Random Forest classifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            training_time = time.time() - start_time
            
            self.logger.info(f"Model trained in {training_time:.2f}s")
            self.logger.info(f"Validation accuracy: {accuracy*100:.1f}%")
            
            # Save model
            self._save_model()
            
            return accuracy, training_time
            
        except ImportError:
            self.logger.error("scikit-learn not installed - run: pip install scikit-learn")
            return 0.0, 0.0
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return 0.0, 0.0
    
    def identify_device(self, signature: RFSignature) -> Optional[DeviceProfile]:
        """
        Identify device from RF signature
        Returns: DeviceProfile with confidence score
        """
        if self.model is None:
            self.logger.error("No trained model available")
            return None
        
        try:
            # Extract features
            X = signature.to_feature_vector().reshape(1, -1)
            X_scaled = self.feature_scaler.transform(X)
            
            # Predict
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Decode prediction
            label = self.label_encoder.inverse_transform([prediction])[0]
            manufacturer, model, os = label.split('|')
            
            # Get confidence (max probability)
            confidence = float(np.max(probabilities))
            
            # Create device profile
            profile = DeviceProfile(
                manufacturer=manufacturer,
                model=model,
                os=os,
                baseband_version="Unknown",  # Would need separate classifier
                confidence=confidence,
                timestamp=datetime.now().isoformat()
            )
            
            # Cache result (with anonymization if enabled)
            if signature.imsi:
                imsi_key = self._anonymize_identifier(signature.imsi)
                self.identified_devices[imsi_key] = profile
            
            self.logger.info(
                f"Device identified: {manufacturer} {model} "
                f"(OS: {os}, confidence: {confidence*100:.1f}%)"
            )
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Identification failed: {e}")
            return None
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores from trained model"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        feature_names = [
            'timing_advance', 'frame_timing_offset', 'preamble_duration',
            'tx_power_mean', 'tx_power_std', 'power_ramp_rate',
            'power_control_accuracy', 'evm_rms', 'phase_error',
            'frequency_error', 'iq_imbalance', 'random_access_pattern',
            'retransmission_strategy', 'buffer_status_reporting',
            'clock_drift', 'spurious_mean', 'spurious_count',
            'filter_rolloff', 'dac_linearity', 'attach_timing',
            'handover_latency', 'frequency', 'technology'
        ]
        
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))
    
    def profile_network(self) -> Dict:
        """
        Generate statistics on identified devices in network
        Returns: Network profile with device demographics
        """
        if not self.identified_devices:
            return {"error": "No devices identified yet"}
        
        # Count by manufacturer
        manufacturers = {}
        os_versions = {}
        models = {}
        
        for profile in self.identified_devices.values():
            manufacturers[profile.manufacturer] = manufacturers.get(profile.manufacturer, 0) + 1
            os_versions[profile.os] = os_versions.get(profile.os, 0) + 1
            models[profile.model] = models.get(profile.model, 0) + 1
        
        return {
            'total_devices': len(self.identified_devices),
            'manufacturers': manufacturers,
            'os_versions': os_versions,
            'models': models,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_report(self) -> str:
        """Generate human-readable device fingerprinting report"""
        profile = self.profile_network()
        
        if 'error' in profile:
            return "No devices fingerprinted yet"
        
        report = [
            "=" * 60,
            "RF ARSENAL OS - DEVICE FINGERPRINTING REPORT",
            "=" * 60,
            f"Generated: {profile['timestamp']}",
            f"Total Devices: {profile['total_devices']}",
            "",
            "MANUFACTURERS:",
            "-" * 60
        ]
        
        for mfr, count in sorted(profile['manufacturers'].items(), 
                                 key=lambda x: x[1], reverse=True):
            pct = (count / profile['total_devices']) * 100
            report.append(f"  {mfr:20s} {count:3d} devices ({pct:5.1f}%)")
        
        report.extend([
            "",
            "OPERATING SYSTEMS:",
            "-" * 60
        ])
        
        for os, count in sorted(profile['os_versions'].items(),
                                key=lambda x: x[1], reverse=True):
            pct = (count / profile['total_devices']) * 100
            report.append(f"  {os:20s} {count:3d} devices ({pct:5.1f}%)")
        
        report.extend([
            "",
            "TOP MODELS:",
            "-" * 60
        ])
        
        top_models = sorted(profile['models'].items(), 
                          key=lambda x: x[1], reverse=True)[:10]
        for model, count in top_models:
            pct = (count / profile['total_devices']) * 100
            report.append(f"  {model:30s} {count:3d} ({pct:5.1f}%)")
        
        # Security insights
        report.extend([
            "",
            "SECURITY INSIGHTS:",
            "-" * 60
        ])
        
        old_os = sum(1 for p in self.identified_devices.values() 
                    if any(old in p.os.lower() for old in ['9', '10', '11', 'kitkat', 'lollipop']))
        
        if old_os > 0:
            report.append(f"  ⚠️  {old_os} devices running outdated OS versions")
        else:
            report.append("  ✅ All devices running current OS versions")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def _anonymize_identifier(self, identifier: str) -> str:
        """
        Hash identifier for privacy (IMSI/IMEI/MAC address)
        
        Args:
            identifier: Raw identifier to anonymize
            
        Returns:
            Hashed identifier (first 12 chars of SHA-256) or original if anonymization disabled
        """
        if not self.anonymize_logs:
            return identifier
        
        import hashlib
        hash_obj = hashlib.sha256(identifier.encode())
        return hash_obj.hexdigest()[:12]  # Short hash for readability
    
    def is_passive_mode(self) -> bool:
        """Check if operating in passive-only (stealth) mode"""
        return self.passive_mode
    
    def set_passive_mode(self, enabled: bool):
        """
        Enable/disable passive-only mode
        
        Args:
            enabled: True for stealth mode (no transmission), False for active mode
        """
        if enabled and not self.passive_mode:
            self.logger.warning("🔒 Enabling STEALTH MODE: Passive-only operation")
            self.passive_mode = True
            self.transmit_enabled = False
            self.active_probing_enabled = False
        elif not enabled and self.passive_mode:
            self.logger.warning("⚠️  Disabling stealth mode - Active transmission ENABLED (DETECTABLE!)")
            self.passive_mode = False
            self.transmit_enabled = True
            self.active_probing_enabled = True
    
    def validate_stealth_operation(self, operation: str) -> bool:
        """
        Validate if an operation is allowed in stealth mode
        
        Args:
            operation: Operation name to check
            
        Returns:
            True if operation is stealth-safe, False otherwise
        
        Raises:
            RuntimeError: If operation breaks stealth and passive_mode is enabled
        """
        # Operations that require transmission (break stealth)
        active_operations = {
            'imsi_catcher': 'IMSI catching requires fake cell tower (active transmission)',
            'active_geolocation': 'Active geolocation requires ranging queries',
            'device_probing': 'Device probing requires direct queries',
            'network_injection': 'Packet injection requires transmission',
            'jamming': 'Jamming requires active RF interference'
        }
        
        if operation in active_operations and self.passive_mode:
            error_msg = f"⚠️  STEALTH VIOLATION: {active_operations[operation]}"
            self.logger.error(error_msg)
            raise RuntimeError(f"Operation '{operation}' not allowed in passive mode")
        
        return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    fingerprinter = MLDeviceFingerprinting()
    print("ML Device Fingerprinting initialized")
    print(f"Model loaded: {fingerprinter.model is not None}")
