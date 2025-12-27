#!/usr/bin/env python3
"""
RF Arsenal OS - HackRF One Controller
$300 budget-friendly SDR for pentesting

Specifications:
- Frequency: 1 MHz - 6 GHz
- Sample Rate: 1-20 MSPS  
- Bandwidth: 1.75-20 MHz
- Half-duplex (TX or RX, not simultaneous)
- 8-bit ADC/DAC
- Cost: ~$300

Best for: Budget pentesting, WiFi, GPS spoofing, spectrum analysis
"""

import logging
import subprocess
import tempfile
import os
import numpy as np
from typing import Optional, Dict, Any

from core.stealth_enforcement import get_stealth_enforcer, OperationType
from core.transmission_monitor import get_transmission_monitor, TransmissionEvent, TransmissionType

logger = logging.getLogger(__name__)


class HackRFController:
    """HackRF One SDR Controller"""
    
    def __init__(self, serial: Optional[str] = None, passive_mode: bool = True):
        """
        Initialize HackRF controller
        
        Args:
            serial: HackRF serial number (optional)
            passive_mode: If True, block all transmission (default: True)
        """
        self.serial = serial
        self.connected = False
        self.config = {
            'frequency': 2_450_000_000,
            'sample_rate': 10_000_000,
            'bandwidth': 10_000_000,
            'tx_gain': 30,
            'rx_lna_gain': 16,
            'rx_vga_gain': 20,
            'amp_enable': True
        }
        
        # 🔒 Stealth enforcement
        self.passive_mode = passive_mode
        self.stealth_enforcer = get_stealth_enforcer()
        self.tx_monitor = get_transmission_monitor()
        
        logger.info(f"HackRF controller initialized (Passive mode: {passive_mode})")
    
    def connect(self) -> bool:
        """Connect to HackRF device"""
        try:
            result = subprocess.run(['hackrf_info'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.connected = True
                if not self.serial:
                    for line in result.stdout.split('\n'):
                        if 'Serial number' in line:
                            self.serial = line.split(':')[1].strip()
                            break
                logger.info(f"Connected to HackRF: {self.serial}")
                return True
            return False
        except FileNotFoundError:
            logger.error("HackRF tools not installed")
            return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def configure(self, frequency: int, sample_rate: int = 10_000_000, 
                  bandwidth: int = None, tx_gain: int = 30,
                  rx_lna_gain: int = 16, rx_vga_gain: int = 20,
                  amp_enable: bool = True) -> bool:
        """Configure HackRF parameters"""
        if not (1_000_000 <= frequency <= 6_000_000_000):
            logger.error(f"Frequency out of range: {frequency/1e6:.2f} MHz")
            return False
        
        if bandwidth is None:
            bandwidth = int(sample_rate * 0.75)
        
        self.config.update({
            'frequency': frequency,
            'sample_rate': sample_rate,
            'bandwidth': bandwidth,
            'tx_gain': tx_gain,
            'rx_lna_gain': rx_lna_gain,
            'rx_vga_gain': rx_vga_gain,
            'amp_enable': amp_enable
        })
        
        logger.info(f"HackRF configured: {frequency/1e6:.2f} MHz, {sample_rate/1e6:.2f} MSPS")
        return True
    
    def transmit(self, samples: np.ndarray, repeat: bool = False) -> bool:
        """
        Transmit samples on HackRF
        
        SECURITY: This function validates stealth compliance before transmission.
        
        Args:
            samples: Complex IQ samples to transmit
            repeat: Repeat transmission continuously
        
        Returns:
            True if transmission succeeded
        
        Raises:
            StealthViolationError: If passive mode is enabled
        """
        # 🔒 CRITICAL SECURITY CHECK: Validate stealth before transmission
        self.stealth_enforcer.validate_transmit({
            'device': 'HackRF',
            'frequency_mhz': self.config['frequency'] / 1e6,
            'sample_rate': self.config['sample_rate'],
            'samples_count': len(samples)
        })
        
        if not self.connected:
            logger.error("HackRF not connected")
            return False
        
        try:
            # Convert to int8 IQ format
            samples_normalized = samples / np.max(np.abs(samples))
            samples_int8 = (samples_normalized * 127).astype(np.int8)
            samples_interleaved = np.zeros(len(samples_int8) * 2, dtype=np.int8)
            samples_interleaved[0::2] = samples_int8.real
            samples_interleaved[1::2] = samples_int8.imag
            
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.iq')
            temp_file.write(samples_interleaved.tobytes())
            temp_file.close()
            
            # Transmit
            cmd = [
                'hackrf_transfer', '-t', temp_file.name,
                '-f', str(self.config['frequency']),
                '-s', str(self.config['sample_rate']),
                '-x', str(self.config['tx_gain']),
                '-a', '1' if self.config['amp_enable'] else '0'
            ]
            
            if repeat:
                cmd.append('-R')
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            os.unlink(temp_file.name)
            
            if result.returncode == 0:
                # 📡 Log transmission for security audit
                duration_ms = (len(samples) / self.config['sample_rate']) * 1000
                tx_event = TransmissionEvent(
                    tx_type=TransmissionType.CUSTOM,
                    frequency=float(self.config['frequency']),
                    power_dbm=float(self.config['tx_gain']),
                    duration_ms=duration_ms,
                    source_module="hackrf_controller",
                    data_size_bytes=len(samples) * 2,  # IQ samples
                    metadata={'repeat': repeat, 'sample_rate': self.config['sample_rate']}
                )
                self.tx_monitor.log_transmission(tx_event)
                
                logger.info(f"Transmitted {len(samples)} samples")
                return True
            else:
                logger.error(f"Transmission failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Transmission error: {e}")
            return False
    
    def receive(self, num_samples: int, timeout: int = 10) -> Optional[np.ndarray]:
        """Receive samples from HackRF"""
        if not self.connected:
            logger.error("HackRF not connected")
            return None
        
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.iq')
            temp_file.close()
            
            cmd = [
                'hackrf_transfer', '-r', temp_file.name,
                '-f', str(self.config['frequency']),
                '-s', str(self.config['sample_rate']),
                '-l', str(self.config['rx_lna_gain']),
                '-g', str(self.config['rx_vga_gain']),
                '-a', '1' if self.config['amp_enable'] else '0',
                '-n', str(num_samples * 2)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0 and os.path.exists(temp_file.name):
                samples_int8 = np.fromfile(temp_file.name, dtype=np.int8)
                samples_complex = (samples_int8[0::2].astype(np.float32) + 
                                 1j * samples_int8[1::2].astype(np.float32)) / 128.0
                os.unlink(temp_file.name)
                logger.info(f"Received {len(samples_complex)} samples")
                return samples_complex
            else:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                return None
        except Exception as e:
            logger.error(f"Reception error: {e}")
            return None
    
    def get_info(self) -> Dict[str, Any]:
        """Get HackRF device information"""
        if not self.connected:
            return {'error': 'Not connected'}
        
        return {
            'model': 'HackRF One',
            'serial': self.serial,
            'frequency_range': '1 MHz - 6 GHz',
            'sample_rate': f"{self.config['sample_rate']/1e6:.2f} MSPS",
            'tx_capable': True,
            'rx_capable': True,
            'full_duplex': False,
            'half_duplex': True
        }
    
    def close(self):
        """Close HackRF connection"""
        if self.connected:
            self.connected = False
            logger.info("HackRF connection closed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("HackRF One Controller Test")
    hackrf = HackRFController()
    if hackrf.connect():
        print("✓ Connected")
        hackrf.configure(frequency=2_437_000_000, sample_rate=20_000_000)
        hackrf.close()
    else:
        print("✗ Connection failed")
