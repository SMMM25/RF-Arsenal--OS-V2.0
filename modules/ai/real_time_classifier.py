#!/usr/bin/env python3
"""
RF Arsenal OS - Real-Time Device Classifier
Live device identification from RF captures
"""

import logging
import threading
import queue
import time
from typing import Dict, Optional, Callable
from datetime import datetime
from collections import defaultdict

from modules.ai.device_fingerprinting import MLDeviceFingerprinting, RFSignature, DeviceProfile
from core.anonymization import get_anonymizer

logger = logging.getLogger(__name__)


class RealTimeClassifier:
    """
    Real-time device classification engine
    
    Features:
    - Continuous RF signature capture
    - Real-time classification
    - Device tracking (IMSI → device profile)
    - Live statistics
    - Callback notifications
    """
    
    def __init__(self, fingerprint_engine: MLDeviceFingerprinting, passive_mode: bool = True):
        """
        Initialize real-time classifier
        
        Args:
            fingerprint_engine: Trained ML fingerprinting engine
            passive_mode: If True, enforce stealth mode (passive-only operation)
        """
        self.fingerprint = fingerprint_engine
        
        # 🔒 Stealth enforcement: inherit from fingerprint engine
        self.passive_mode = passive_mode or fingerprint_engine.passive_mode
        self.anonymize_logs = fingerprint_engine.anonymize_logs
        
        # 🔐 Centralized anonymization
        self.anonymizer = get_anonymizer()
        
        # Processing queue
        self.signature_queue = queue.Queue(maxsize=1000)
        
        # Tracked devices (ALWAYS use anonymized keys for security)
        self.devices = {}  # IMSI_hash → DeviceProfile
        self.device_stats = defaultdict(int)  # Device type → count
        
        # Processing state
        self.running = False
        self.worker_thread = None
        
        # Callbacks
        self.callbacks = {
            'new_device': [],
            'device_update': [],
            'stats_update': []
        }
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'total_identified': 0,
            'processing_rate': 0.0,
            'start_time': None
        }
        
        logger.info("Real-time classifier initialized")
        
        if self.passive_mode:
            logger.warning("🔒 Classifier in STEALTH MODE: Passive-only operation")
        if self.anonymize_logs:
            logger.info("🔐 Identifier anonymization ENABLED for all tracked devices")
    
    def register_callback(self, event_type: str, callback: Callable):
        """
        Register callback for events
        
        Args:
            event_type: 'new_device', 'device_update', 'stats_update'
            callback: Callback function
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            logger.info(f"Registered callback for {event_type}")
    
    def _trigger_callbacks(self, event_type: str, data: Dict):
        """Trigger all callbacks for an event"""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def start(self):
        """Start real-time classification"""
        if self.running:
            logger.warning("Classifier already running")
            return
        
        self.running = True
        self.stats['start_time'] = time.time()
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info("Real-time classifier started")
    
    def stop(self):
        """Stop real-time classification"""
        if not self.running:
            return
        
        self.running = False
        
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        
        logger.info("Real-time classifier stopped")
    
    def add_signature(self, imsi: str, signature: RFSignature) -> bool:
        """
        Add RF signature for classification
        
        Args:
            imsi: Target IMSI
            signature: RF signature data
            
        Returns:
            True if added to queue
        """
        try:
            self.signature_queue.put_nowait((imsi, signature))
            return True
        except queue.Full:
            logger.warning("Signature queue full, dropping sample")
            return False
    
    def _processing_loop(self):
        """Main processing loop (runs in worker thread)"""
        logger.info("Processing loop started")
        
        last_stats_update = time.time()
        processing_times = []
        
        while self.running:
            try:
                # Get signature from queue (with timeout)
                try:
                    imsi, signature = self.signature_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                start_time = time.time()
                
                # Classify device
                profile = self.fingerprint.identify_device(signature)
                
                if profile:
                    self.stats['total_identified'] += 1
                    
                    # 🔐 SECURITY FIX: Always anonymize IMSI before storage
                    imsi_hash = self.anonymizer.anonymize_imsi(imsi)
                    
                    # Check if new device
                    is_new = imsi_hash not in self.devices
                    
                    # Update device tracking (using anonymized key)
                    self.devices[imsi_hash] = profile
                    device_key = f"{profile.manufacturer} {profile.model}"
                    self.device_stats[device_key] += 1
                    
                    # Store IMSI in fingerprint engine cache (also anonymized)
                    self.fingerprint.identified_devices[imsi_hash] = profile
                    
                    # Trigger callbacks (with anonymized IMSI)
                    event_data = {
                        'imsi': imsi_hash,  # 🔐 Always anonymized
                        'profile': profile,
                        'timestamp': datetime.now()
                    }
                    
                    if is_new:
                        self._trigger_callbacks('new_device', event_data)
                        logger.info(f"New device detected: {imsi_hash} → {profile.manufacturer} {profile.model}")
                    else:
                        self._trigger_callbacks('device_update', event_data)
                
                self.stats['total_processed'] += 1
                
                # Track processing time
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Keep last 100 times for rate calculation
                if len(processing_times) > 100:
                    processing_times.pop(0)
                
                # Update statistics periodically
                if time.time() - last_stats_update > 5.0:
                    if processing_times:
                        avg_time = sum(processing_times) / len(processing_times)
                        self.stats['processing_rate'] = 1.0 / avg_time if avg_time > 0 else 0
                    
                    self._trigger_callbacks('stats_update', self.get_statistics())
                    last_stats_update = time.time()
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
        
        logger.info("Processing loop stopped")
    
    def get_device_profile(self, imsi: str) -> Optional[DeviceProfile]:
        """
        Get device profile for IMSI
        
        Args:
            imsi: Target IMSI
            
        Returns:
            Device profile or None
        """
        return self.devices.get(imsi)
    
    def get_statistics(self) -> Dict:
        """Get real-time statistics"""
        runtime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        
        return {
            'runtime_seconds': runtime,
            'total_processed': self.stats['total_processed'],
            'total_identified': self.stats['total_identified'],
            'identification_rate': self.stats['total_identified'] / self.stats['total_processed'] if self.stats['total_processed'] > 0 else 0,
            'processing_rate': self.stats['processing_rate'],
            'queue_size': self.signature_queue.qsize(),
            'tracked_devices': len(self.devices),
            'device_breakdown': dict(self.device_stats)
        }
    
    def get_network_summary(self) -> Dict:
        """Get network device summary"""
        if not self.devices:
            return {'error': 'No devices detected'}
        
        # Count by manufacturer
        by_manufacturer = defaultdict(int)
        by_os = defaultdict(int)
        by_os_version = defaultdict(int)
        
        for profile in self.devices.values():
            by_manufacturer[profile.manufacturer] += 1
            by_os[profile.os] += 1
            by_os_version[profile.os] += 1
        
        total = len(self.devices)
        
        return {
            'total_devices': total,
            'by_manufacturer': dict(by_manufacturer),
            'by_os': dict(by_os),
            'by_os_version': dict(by_os_version),
            'percentages': {
                'by_manufacturer': {k: v/total*100 for k, v in by_manufacturer.items()},
                'by_os': {k: v/total*100 for k, v in by_os.items()}
            }
        }
    
    def generate_live_report(self) -> str:
        """Generate live network report"""
        stats = self.get_statistics()
        summary = self.get_network_summary()
        
        if 'error' in summary:
            return "No devices detected yet"
        
        report = []
        report.append("=" * 60)
        report.append("RF ARSENAL OS - LIVE NETWORK PROFILE")
        report.append("=" * 60)
        report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Runtime: {stats['runtime_seconds']:.0f}s")
        report.append(f"Processing Rate: {stats['processing_rate']:.1f} devices/sec")
        report.append("")
        
        report.append(f"Total Devices Detected: {summary['total_devices']}")
        report.append("")
        
        # OS breakdown
        report.append("Operating Systems:")
        for os_name, count in sorted(summary['by_os'].items(), key=lambda x: x[1], reverse=True):
            pct = summary['percentages']['by_os'][os_name]
            report.append(f"  • {os_name}: {count} devices ({pct:.1f}%)")
        report.append("")
        
        # Manufacturer breakdown
        report.append("Manufacturers:")
        for mfg, count in sorted(summary['by_manufacturer'].items(), key=lambda x: x[1], reverse=True):
            pct = summary['percentages']['by_manufacturer'][mfg]
            report.append(f"  • {mfg}: {count} devices ({pct:.1f}%)")
        report.append("")
        
        # Top devices
        report.append("Top Device Models:")
        top_devices = sorted(self.device_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        for device, count in top_devices:
            report.append(f"  • {device}: {count} devices")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def export_device_list(self, output_path: str) -> bool:
        """
        Export detected devices to JSON
        
        Args:
            output_path: Output file path
            
        Returns:
            True if exported successfully
        """
        try:
            import json
            
            devices_data = []
            
            for imsi, profile in self.devices.items():
                devices_data.append({
                    'imsi': imsi,
                    'manufacturer': profile.manufacturer,
                    'model': profile.model,
                    'os': profile.os,
                    'baseband_version': profile.baseband_version,
                    'confidence': profile.confidence,
                    'timestamp': profile.timestamp
                })
            
            with open(output_path, 'w') as f:
                json.dump(devices_data, f, indent=2)
            
            logger.info(f"Exported {len(devices_data)} devices to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False


class LiveDashboard:
    """
    Terminal-based live dashboard for real-time classification
    """
    
    def __init__(self, classifier: RealTimeClassifier):
        """
        Initialize live dashboard
        
        Args:
            classifier: Real-time classifier instance
        """
        self.classifier = classifier
        self.running = False
        self.update_thread = None
    
    def start(self):
        """Start live dashboard"""
        if self.running:
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info("Live dashboard started")
    
    def stop(self):
        """Stop live dashboard"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
    
    def _update_loop(self):
        """Dashboard update loop"""
        while self.running:
            try:
                # Clear screen (ANSI escape code)
                print("\033[2J\033[H", end='')
                
                # Print report
                report = self.classifier.generate_live_report()
                print(report)
                
                # Wait before next update
                time.sleep(2.0)
                
            except Exception as e:
                logger.error(f"Dashboard error: {e}")
                break


if __name__ == "__main__":
    # Test real-time classifier
    logging.basicConfig(level=logging.INFO)
    
    print("RF Arsenal OS - Real-Time Classifier Test")
    print("=" * 60)
    
    from modules.ai.device_fingerprinting import MLDeviceFingerprinting
    from modules.ai.training_data_generator import TrainingDataGenerator
    
    # Train model
    print("\n[+] Training model...")
    generator = TrainingDataGenerator()
    dataset = generator.generate_dataset(samples_per_device=50)
    
    fingerprint = MLDeviceFingerprinting()
    for sig, profile in dataset:
        fingerprint.add_training_data(sig, profile)
    
    accuracy, train_time = fingerprint.train_model()
    print(f"[+] Model trained: {accuracy*100:.1f}% accuracy in {train_time:.2f}s")
    
    # Create real-time classifier
    print("\n[+] Starting real-time classifier...")
    classifier = RealTimeClassifier(fingerprint)
    
    # Register callbacks
    def on_new_device(data):
        profile = data['profile']
        print(f"\n[NEW] {data['imsi']}: {profile.manufacturer} {profile.model} "
              f"({profile.confidence*100:.0f}%)")
    
    classifier.register_callback('new_device', on_new_device)
    
    # Start classifier
    classifier.start()
    
    # Simulate device detection
    print("[+] Simulating device detection...")
    
    for i in range(10):
        # Generate random signature
        mfr = list(generator.device_database.keys())[i % 5]
        model = list(generator.device_database[mfr].keys())[0]
        signature = generator.generate_signature(mfr, model)
        
        if signature:
            imsi = f"31041000000000{i:02d}"
            classifier.add_signature(imsi, signature)
            time.sleep(0.5)
    
    # Wait for processing
    time.sleep(3)
    
    # Print statistics
    print("\n[+] Statistics:")
    stats = classifier.get_statistics()
    print(f"  Processed: {stats['total_processed']}")
    print(f"  Identified: {stats['total_identified']}")
    print(f"  Rate: {stats['processing_rate']:.1f} devices/sec")
    
    # Print network summary
    print("\n[+] Network Summary:")
    summary = classifier.get_network_summary()
    if 'error' not in summary:
        print(f"  Total devices: {summary['total_devices']}")
        for os_name, pct in summary['percentages']['by_os'].items():
            print(f"  {os_name}: {pct:.1f}%")
    
    # Stop classifier
    classifier.stop()
    
    print("\n[+] Test complete")
