#!/usr/bin/env python3
"""
RF Arsenal OS - Training Data Generator
Generate synthetic RF signatures for ML model training
"""

import logging
import numpy as np
import json
import csv
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import random

from modules.ai.device_fingerprinting import RFSignature, DeviceProfile

class TrainingDataGenerator:
    """
    Generate synthetic training data for device fingerprinting
    Based on known RF characteristics of commercial devices
    """
    
    def __init__(self):
        self.logger = logging.getLogger('TrainingDataGenerator')
        
        # Known device RF characteristics database
        self.device_database = {
            # Apple devices (Qualcomm/Intel modems)
            'Apple': {
                'iPhone 14 Pro': {
                    'os': 'iOS 17.x',
                    'baseband': 'Qualcomm X65',
                    'characteristics': {
                        'timing_advance': (2.0, 0.5),      # Mean, StdDev
                        'evm_rms': (2.5, 0.3),
                        'phase_error': (1.5, 0.2),
                        'frequency_error': (50, 20),
                        'clock_drift': (0.1, 0.05),
                        'power_ramp_rate': (5000, 500),
                        'attach_timing': (450, 50)
                    }
                },
                'iPhone 13': {
                    'os': 'iOS 16.x',
                    'baseband': 'Qualcomm X60',
                    'characteristics': {
                        'timing_advance': (2.2, 0.6),
                        'evm_rms': (2.8, 0.4),
                        'phase_error': (1.8, 0.3),
                        'frequency_error': (60, 25),
                        'clock_drift': (0.15, 0.06),
                        'power_ramp_rate': (4800, 600),
                        'attach_timing': (480, 60)
                    }
                },
                'iPhone 12': {
                    'os': 'iOS 15.x',
                    'baseband': 'Qualcomm X55',
                    'characteristics': {
                        'timing_advance': (2.5, 0.7),
                        'evm_rms': (3.0, 0.5),
                        'phase_error': (2.0, 0.4),
                        'frequency_error': (70, 30),
                        'clock_drift': (0.2, 0.08),
                        'power_ramp_rate': (4500, 700),
                        'attach_timing': (520, 70)
                    }
                }
            },
            
            # Samsung devices (Exynos/Snapdragon variants)
            'Samsung': {
                'Galaxy S23 Ultra': {
                    'os': 'Android 14',
                    'baseband': 'Snapdragon X70',
                    'characteristics': {
                        'timing_advance': (2.3, 0.6),
                        'evm_rms': (2.7, 0.4),
                        'phase_error': (1.6, 0.3),
                        'frequency_error': (55, 22),
                        'clock_drift': (0.12, 0.05),
                        'power_ramp_rate': (5200, 550),
                        'attach_timing': (440, 55)
                    }
                },
                'Galaxy S22': {
                    'os': 'Android 13',
                    'baseband': 'Exynos 2200',
                    'characteristics': {
                        'timing_advance': (2.6, 0.8),
                        'evm_rms': (3.2, 0.6),
                        'phase_error': (2.2, 0.5),
                        'frequency_error': (75, 35),
                        'clock_drift': (0.25, 0.1),
                        'power_ramp_rate': (4600, 800),
                        'attach_timing': (510, 75)
                    }
                },
                'Galaxy A54': {
                    'os': 'Android 13',
                    'baseband': 'Exynos 1380',
                    'characteristics': {
                        'timing_advance': (3.0, 1.0),
                        'evm_rms': (3.8, 0.8),
                        'phase_error': (2.8, 0.6),
                        'frequency_error': (90, 40),
                        'clock_drift': (0.35, 0.15),
                        'power_ramp_rate': (4200, 900),
                        'attach_timing': (580, 90)
                    }
                }
            },
            
            # Google Pixel (Samsung Tensor chips)
            'Google': {
                'Pixel 8 Pro': {
                    'os': 'Android 14',
                    'baseband': 'Tensor G3',
                    'characteristics': {
                        'timing_advance': (2.1, 0.5),
                        'evm_rms': (2.6, 0.35),
                        'phase_error': (1.4, 0.25),
                        'frequency_error': (48, 18),
                        'clock_drift': (0.09, 0.04),
                        'power_ramp_rate': (5300, 500),
                        'attach_timing': (430, 50)
                    }
                },
                'Pixel 7': {
                    'os': 'Android 13',
                    'baseband': 'Tensor G2',
                    'characteristics': {
                        'timing_advance': (2.4, 0.6),
                        'evm_rms': (2.9, 0.45),
                        'phase_error': (1.7, 0.35),
                        'frequency_error': (58, 24),
                        'clock_drift': (0.14, 0.06),
                        'power_ramp_rate': (5000, 600),
                        'attach_timing': (465, 60)
                    }
                }
            },
            
            # Xiaomi devices (Snapdragon)
            'Xiaomi': {
                'Xiaomi 13 Pro': {
                    'os': 'Android 13',
                    'baseband': 'Snapdragon X70',
                    'characteristics': {
                        'timing_advance': (2.7, 0.8),
                        'evm_rms': (3.3, 0.7),
                        'phase_error': (2.3, 0.5),
                        'frequency_error': (80, 35),
                        'clock_drift': (0.28, 0.12),
                        'power_ramp_rate': (4700, 750),
                        'attach_timing': (530, 80)
                    }
                },
                'Redmi Note 12': {
                    'os': 'Android 12',
                    'baseband': 'Snapdragon 4 Gen 1',
                    'characteristics': {
                        'timing_advance': (3.5, 1.2),
                        'evm_rms': (4.2, 1.0),
                        'phase_error': (3.5, 0.8),
                        'frequency_error': (110, 50),
                        'clock_drift': (0.45, 0.2),
                        'power_ramp_rate': (3900, 1000),
                        'attach_timing': (650, 100)
                    }
                }
            },
            
            # OnePlus devices
            'OnePlus': {
                'OnePlus 11': {
                    'os': 'Android 13',
                    'baseband': 'Snapdragon X70',
                    'characteristics': {
                        'timing_advance': (2.5, 0.7),
                        'evm_rms': (3.0, 0.5),
                        'phase_error': (2.0, 0.4),
                        'frequency_error': (70, 30),
                        'clock_drift': (0.22, 0.09),
                        'power_ramp_rate': (4800, 700),
                        'attach_timing': (500, 70)
                    }
                }
            }
        }
    
    def generate_signature(self, manufacturer: str, model: str, 
                          noise_level: float = 0.1) -> Optional[RFSignature]:
        """
        Generate synthetic RF signature for a device
        
        Args:
            manufacturer: Device manufacturer
            model: Device model
            noise_level: Amount of random noise to add (0.0-1.0)
        
        Returns:
            RFSignature object with realistic RF characteristics
        """
        try:
            device_data = self.device_database[manufacturer][model]
            chars = device_data['characteristics']
            
            # Generate realistic values with Gaussian noise
            def sample(mean_std):
                mean, std = mean_std
                value = np.random.normal(mean, std * (1 + noise_level))
                return max(0, value)  # No negative values
            
            # Generate power profile (10 samples over time)
            base_power = np.random.uniform(20, 23)  # dBm
            power_profile = [
                base_power + np.random.normal(0, 0.5) 
                for _ in range(10)
            ]
            
            # Generate spurious emissions (typically 2-5 spurs)
            num_spurs = random.randint(2, 5)
            spurious = [
                np.random.uniform(-70, -50)  # dBc
                for _ in range(num_spurs)
            ]
            
            # Random protocol patterns
            rach_patterns = ['sequential', 'random', 'adaptive', 'fixed_offset']
            retx_strategies = ['exponential_backoff', 'linear', 'immediate']
            bsr_patterns = ['periodic_50ms', 'periodic_100ms', 'event_triggered']
            
            signature = RFSignature(
                timing_advance=sample(chars['timing_advance']),
                frame_timing_offset=np.random.uniform(-0.5, 0.5),
                preamble_duration=np.random.uniform(0.8, 1.2),
                
                tx_power_profile=power_profile,
                power_ramp_rate=sample(chars['power_ramp_rate']),
                power_control_accuracy=np.random.uniform(0.8, 0.98),
                
                evm_rms=sample(chars['evm_rms']),
                phase_error=sample(chars['phase_error']),
                frequency_error=sample(chars['frequency_error']),
                iq_imbalance=np.random.uniform(0.1, 0.5),
                
                random_access_pattern=random.choice(rach_patterns),
                retransmission_strategy=random.choice(retx_strategies),
                buffer_status_reporting=random.choice(bsr_patterns),
                
                clock_drift=sample(chars['clock_drift']),
                spurious_emissions=spurious,
                filter_rolloff=np.random.uniform(0.2, 0.35),
                dac_linearity=np.random.uniform(40, 60),
                
                attach_procedure_timing=sample(chars['attach_timing']),
                handover_latency=np.random.uniform(30, 80),
                
                frequency=random.choice([1800, 2100, 2600]),  # MHz
                technology=random.choice(['4G', '5G']),
                timestamp=datetime.now().isoformat()
            )
            
            return signature
            
        except KeyError:
            self.logger.error(f"Device not in database: {manufacturer} {model}")
            return None
    
    def generate_dataset(self, samples_per_device: int = 50, 
                        noise_level: float = 0.1) -> List[tuple]:
        """
        Generate complete training dataset
        
        Returns:
            List of (RFSignature, DeviceProfile) tuples
        """
        dataset = []
        
        for manufacturer, models in self.device_database.items():
            for model, data in models.items():
                self.logger.info(f"Generating {samples_per_device} samples for {manufacturer} {model}")
                
                for _ in range(samples_per_device):
                    signature = self.generate_signature(manufacturer, model, noise_level)
                    if signature:
                        profile = DeviceProfile(
                            manufacturer=manufacturer,
                            model=model,
                            os=data['os'],
                            baseband_version=data['baseband'],
                            confidence=1.0,  # Ground truth
                            timestamp=signature.timestamp
                        )
                        dataset.append((signature, profile))
        
        self.logger.info(f"Generated {len(dataset)} total training samples")
        return dataset
    
    def augment_signature(self, signature: RFSignature, 
                         augmentations: int = 5) -> List[RFSignature]:
        """
        Create augmented versions of a signature (data augmentation)
        Useful for increasing training data diversity
        """
        augmented = []
        
        for _ in range(augmentations):
            # Create copy with slight variations
            aug_sig = RFSignature(
                timing_advance=signature.timing_advance * np.random.uniform(0.95, 1.05),
                frame_timing_offset=signature.frame_timing_offset + np.random.normal(0, 0.1),
                preamble_duration=signature.preamble_duration * np.random.uniform(0.98, 1.02),
                
                tx_power_profile=[p + np.random.normal(0, 0.2) for p in signature.tx_power_profile],
                power_ramp_rate=signature.power_ramp_rate * np.random.uniform(0.9, 1.1),
                power_control_accuracy=min(1.0, signature.power_control_accuracy + np.random.normal(0, 0.02)),
                
                evm_rms=signature.evm_rms * np.random.uniform(0.95, 1.05),
                phase_error=signature.phase_error * np.random.uniform(0.95, 1.05),
                frequency_error=signature.frequency_error + np.random.normal(0, 5),
                iq_imbalance=signature.iq_imbalance * np.random.uniform(0.9, 1.1),
                
                random_access_pattern=signature.random_access_pattern,
                retransmission_strategy=signature.retransmission_strategy,
                buffer_status_reporting=signature.buffer_status_reporting,
                
                clock_drift=signature.clock_drift * np.random.uniform(0.95, 1.05),
                spurious_emissions=[s + np.random.normal(0, 1) for s in signature.spurious_emissions],
                filter_rolloff=signature.filter_rolloff * np.random.uniform(0.98, 1.02),
                dac_linearity=signature.dac_linearity + np.random.normal(0, 1),
                
                attach_procedure_timing=signature.attach_procedure_timing * np.random.uniform(0.9, 1.1),
                handover_latency=signature.handover_latency * np.random.uniform(0.9, 1.1),
                
                frequency=signature.frequency,
                technology=signature.technology,
                timestamp=datetime.now().isoformat(),
                imsi=signature.imsi
            )
            augmented.append(aug_sig)
        
        return augmented
    
    def export_dataset(self, dataset: List[tuple], output_path: str, 
                      format: str = 'json'):
        """
        Export training dataset to file
        
        Args:
            dataset: List of (RFSignature, DeviceProfile) tuples
            output_path: Output file path
            format: 'json' or 'csv'
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            # JSON export
            data = []
            for sig, profile in dataset:
                data.append({
                    'signature': {
                        'timing_advance': sig.timing_advance,
                        'frame_timing_offset': sig.frame_timing_offset,
                        'preamble_duration': sig.preamble_duration,
                        'tx_power_profile': sig.tx_power_profile,
                        'power_ramp_rate': sig.power_ramp_rate,
                        'power_control_accuracy': sig.power_control_accuracy,
                        'evm_rms': sig.evm_rms,
                        'phase_error': sig.phase_error,
                        'frequency_error': sig.frequency_error,
                        'iq_imbalance': sig.iq_imbalance,
                        'random_access_pattern': sig.random_access_pattern,
                        'retransmission_strategy': sig.retransmission_strategy,
                        'buffer_status_reporting': sig.buffer_status_reporting,
                        'clock_drift': sig.clock_drift,
                        'spurious_emissions': sig.spurious_emissions,
                        'filter_rolloff': sig.filter_rolloff,
                        'dac_linearity': sig.dac_linearity,
                        'attach_procedure_timing': sig.attach_procedure_timing,
                        'handover_latency': sig.handover_latency,
                        'frequency': sig.frequency,
                        'technology': sig.technology
                    },
                    'label': profile.to_dict()
                })
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Exported {len(data)} samples to {output_path}")
            
        elif format == 'csv':
            # CSV export
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    'manufacturer', 'model', 'os', 'baseband',
                    'timing_advance', 'evm_rms', 'phase_error', 'frequency_error',
                    'clock_drift', 'power_ramp_rate', 'attach_timing', 'frequency',
                    'technology'
                ])
                
                # Data rows
                for sig, profile in dataset:
                    writer.writerow([
                        profile.manufacturer, profile.model, profile.os, profile.baseband_version,
                        sig.timing_advance, sig.evm_rms, sig.phase_error, sig.frequency_error,
                        sig.clock_drift, sig.power_ramp_rate, sig.attach_procedure_timing,
                        sig.frequency, sig.technology
                    ])
            
            self.logger.info(f"Exported {len(dataset)} samples to {output_path}")
    
    def load_dataset(self, input_path: str) -> List[tuple]:
        """Load dataset from JSON file"""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        dataset = []
        for item in data:
            sig_data = item['signature']
            label_data = item['label']
            
            sig = RFSignature(**sig_data)
            profile = DeviceProfile(**label_data)
            dataset.append((sig, profile))
        
        self.logger.info(f"Loaded {len(dataset)} samples from {input_path}")
        return dataset
    
    def generate_stealth_safe_dataset(self, samples_per_device: int = 200, 
                                     output_path: Optional[str] = None) -> List[tuple]:
        """
        🔒 STEALTH-SAFE TRAINING WORKFLOW
        
        Generate training dataset using ONLY synthetic data (no field captures required).
        This allows model training without any RF transmissions or detectable activity.
        
        Args:
            samples_per_device: Number of synthetic samples per device type (default: 200)
            output_path: Optional path to save dataset (JSON format)
            
        Returns:
            List of (RFSignature, DeviceProfile) tuples
        """
        self.logger.warning("🔒 STEALTH-SAFE TRAINING: Using 100% synthetic data (zero field captures)")
        
        # Generate dataset using only synthetic signatures
        dataset = self.generate_dataset(
            samples_per_device=samples_per_device,
            noise_level=0.15  # Higher noise for better generalization
        )
        
        # Optionally save to disk
        if output_path:
            self.export_dataset(dataset, output_path)
            self.logger.info(f"✅ Stealth-safe training data saved: {output_path}")
        
        self.logger.info("""  
✅ STEALTH VERIFICATION:
   - Zero RF transmissions required
   - Zero field captures performed
   - Model can be trained entirely offline
   - Safe for operational deployment
        """)
        
        return dataset
    
    def validate_no_real_captures(self, dataset: List[tuple]) -> bool:
        """
        Validate that dataset contains no real field captures (stealth verification)
        
        Args:
            dataset: Training dataset to validate
            
        Returns:
            True if dataset is stealth-safe (no real captures), False otherwise
        """
        # Check for indicators of real captures
        for signature, profile in dataset:
            # Real captures often have IMSI attached
            if signature.imsi and not signature.imsi.startswith('synthetic_'):
                self.logger.warning(
                    f"⚠️  Dataset contains potential real capture with IMSI: {signature.imsi[:8]}..."
                )
                return False
        
        self.logger.info("✅ Dataset validated: No real captures detected (stealth-safe)")
        return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example: Generate training dataset
    generator = TrainingDataGenerator()
    dataset = generator.generate_dataset(samples_per_device=50)
    generator.export_dataset(dataset, "data/training/device_signatures.json")
    
    print(f"Generated {len(dataset)} training samples")
    print(f"Manufacturers: {len(generator.device_database)}")
    print(f"Total device models: {sum(len(models) for models in generator.device_database.values())}")
