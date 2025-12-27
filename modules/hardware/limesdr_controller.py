#!/usr/bin/env python3
"""
RF Arsenal OS - LimeSDR Controller
$159 budget full-duplex SDR

Specifications:
- Frequency: 10 MHz - 3.5 GHz
- Sample Rate: 0.1-30.72 MSPS
- Bandwidth: 5-40 MHz
- Full-duplex (simultaneous TX/RX)
- 12-bit ADC/DAC
- Cost: ~$159

Best for: Budget full-duplex, Cellular (2G/3G/4G), WiFi, LTE base stations
"""

import logging
import numpy as np
from typing import Optional, Dict, Any

from core.stealth_enforcement import get_stealth_enforcer, OperationType
from core.transmission_monitor import get_transmission_monitor, TransmissionEvent, TransmissionType

logger = logging.getLogger(__name__)


class LimeSDRController:
    """LimeSDR Mini Controller"""
    
    def __init__(self, device_str: Optional[str] = None, passive_mode: bool = True):
        """
        Initialize LimeSDR controller
        
        Args:
            device_str: Device string/serial (optional)
            passive_mode: If True, block all transmission (default: True)
        """
        self.device = None
        self.device_str = device_str
        self.connected = False
        self.soapy_available = False
        
        try:
            import SoapySDR
            self.soapy_available = True
        except ImportError:
            logger.warning("SoapySDR not installed")
        
        self.config = {
            'frequency': 2_450_000_000,
            'sample_rate': 10_000_000,
            'bandwidth': 10_000_000,
            'tx_gain': 60,
            'rx_gain': 40,
            'antenna_tx': 'BAND2',
            'antenna_rx': 'LNAW'
        }
        
        # 🔒 Stealth enforcement
        self.passive_mode = passive_mode
        self.stealth_enforcer = get_stealth_enforcer()
        self.tx_monitor = get_transmission_monitor()
        
        logger.info(f"LimeSDR controller initialized (Passive mode: {passive_mode})")
    
    def connect(self) -> bool:
        """Connect to LimeSDR device"""
        if not self.soapy_available:
            logger.error("SoapySDR library not available")
            return False
        
        try:
            import SoapySDR
            results = SoapySDR.Device.enumerate()
            
            if not results:
                logger.error("No LimeSDR devices found")
                return False
            
            if self.device_str:
                device_args = {'driver': 'lime', 'serial': self.device_str}
            else:
                device_args = results[0]
            
            self.device = SoapySDR.Device(device_args)
            self.connected = True
            
            device_info = self.device.getHardwareInfo()
            logger.info(f"Connected to LimeSDR: {device_info.get('hardwareVersion', 'Unknown')}")
            return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def configure(self, frequency: int, sample_rate: int = 10_000_000,
                  bandwidth: int = None, tx_gain: int = 60,
                  rx_gain: int = 40, antenna_tx: str = 'BAND2',
                  antenna_rx: str = 'LNAW') -> bool:
        """Configure LimeSDR parameters"""
        if not self.connected:
            logger.error("LimeSDR not connected")
            return False
        
        if not (10_000_000 <= frequency <= 3_500_000_000):
            logger.error(f"Frequency out of range: {frequency/1e6:.2f} MHz")
            return False
        
        if bandwidth is None:
            bandwidth = int(sample_rate * 0.75)
        
        try:
            import SoapySDR
            from SoapySDR import SOAPY_SDR_TX, SOAPY_SDR_RX
            
            # Configure TX
            self.device.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
            self.device.setFrequency(SOAPY_SDR_TX, 0, frequency)
            self.device.setBandwidth(SOAPY_SDR_TX, 0, bandwidth)
            self.device.setGain(SOAPY_SDR_TX, 0, tx_gain)
            self.device.setAntenna(SOAPY_SDR_TX, 0, antenna_tx)
            
            # Configure RX
            self.device.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
            self.device.setFrequency(SOAPY_SDR_RX, 0, frequency)
            self.device.setBandwidth(SOAPY_SDR_RX, 0, bandwidth)
            self.device.setGain(SOAPY_SDR_RX, 0, rx_gain)
            self.device.setAntenna(SOAPY_SDR_RX, 0, antenna_rx)
            
            self.config.update({
                'frequency': frequency,
                'sample_rate': sample_rate,
                'bandwidth': bandwidth,
                'tx_gain': tx_gain,
                'rx_gain': rx_gain,
                'antenna_tx': antenna_tx,
                'antenna_rx': antenna_rx
            })
            
            logger.info(f"LimeSDR configured: {frequency/1e6:.2f} MHz, {sample_rate/1e6:.2f} MSPS")
            return True
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            return False
    
    def transmit(self, samples: np.ndarray, repeat: bool = False) -> bool:
        """
        Transmit samples on LimeSDR
        
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
            'device': 'LimeSDR',
            'frequency_mhz': self.config['frequency'] / 1e6,
            'sample_rate': self.config['sample_rate'],
            'samples_count': len(samples)
        })
        
        if not self.connected:
            logger.error("LimeSDR not connected")
            return False
        
        try:
            import SoapySDR
            from SoapySDR import SOAPY_SDR_TX, SOAPY_SDR_CF32
            
            tx_stream = self.device.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
            self.device.activateStream(tx_stream)
            
            samples_cf32 = samples.astype(np.complex64)
            
            while True:
                status = self.device.writeStream(tx_stream, [samples_cf32], len(samples_cf32))
                if not repeat:
                    break
            
            self.device.deactivateStream(tx_stream)
            self.device.closeStream(tx_stream)
            
            # 📡 Log transmission for security audit
            duration_ms = (len(samples) / self.config['sample_rate']) * 1000
            tx_event = TransmissionEvent(
                tx_type=TransmissionType.CUSTOM,
                frequency=float(self.config['frequency']),
                power_dbm=float(self.config['tx_gain']),
                duration_ms=duration_ms,
                source_module="limesdr_controller",
                data_size_bytes=len(samples) * 8,  # Complex64 = 8 bytes
                metadata={'repeat': repeat, 'sample_rate': self.config['sample_rate']}
            )
            self.tx_monitor.log_transmission(tx_event)
            
            logger.info(f"Transmitted {len(samples)} samples")
            return True
        except Exception as e:
            logger.error(f"Transmission error: {e}")
            return False
    
    def receive(self, num_samples: int, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Receive samples from LimeSDR"""
        if not self.connected:
            logger.error("LimeSDR not connected")
            return None
        
        try:
            import SoapySDR
            from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
            
            rx_stream = self.device.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
            self.device.activateStream(rx_stream)
            
            samples = np.zeros(num_samples, dtype=np.complex64)
            total_received = 0
            timeout_us = int(timeout * 1e6)
            
            while total_received < num_samples:
                status = self.device.readStream(
                    rx_stream, [samples[total_received:]],
                    num_samples - total_received, timeoutUs=timeout_us
                )
                if status.ret > 0:
                    total_received += status.ret
                else:
                    break
            
            self.device.deactivateStream(rx_stream)
            self.device.closeStream(rx_stream)
            
            if total_received > 0:
                logger.info(f"Received {total_received} samples")
                return samples[:total_received]
            return None
        except Exception as e:
            logger.error(f"Reception error: {e}")
            return None
    
    def get_info(self) -> Dict[str, Any]:
        """Get LimeSDR device information"""
        if not self.connected:
            return {'error': 'Not connected'}
        
        hw_info = self.device.getHardwareInfo()
        return {
            'model': 'LimeSDR Mini',
            'hardware_version': hw_info.get('hardwareVersion', 'Unknown'),
            'firmware_version': hw_info.get('firmwareVersion', 'Unknown'),
            'frequency_range': '10 MHz - 3.5 GHz',
            'sample_rate': f"{self.config['sample_rate']/1e6:.2f} MSPS",
            'tx_capable': True,
            'rx_capable': True,
            'full_duplex': True
        }
    
    def close(self):
        """Close LimeSDR connection"""
        if self.connected and self.device:
            self.device = None
            self.connected = False
            logger.info("LimeSDR connection closed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("LimeSDR Controller Test")
    lime = LimeSDRController()
    if lime.soapy_available and lime.connect():
        print("✓ Connected")
        lime.configure(frequency=2_437_000_000, sample_rate=20_000_000)
        lime.close()
    else:
        print("✗ SoapySDR not installed or connection failed")
