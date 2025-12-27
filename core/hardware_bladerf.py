#!/usr/bin/env python3
"""
RF Arsenal OS - BladeRF Hardware Controller
Implementation for Nuand BladeRF 2.0 micro xA9 SDR
"""

import logging
from typing import Optional
from .hardware_interface import HardwareInterface, HardwareConfig, HardwareStatus

logger = logging.getLogger(__name__)

try:
    import bladerf
    BLADERF_AVAILABLE = True
    logger.info("BladeRF library loaded successfully")
except ImportError:
    BLADERF_AVAILABLE = False
    logger.warning("BladeRF library not found. Install with: pip install bladerf")


class BladeRFHardware(HardwareInterface):
    """
    BladeRF 2.0 micro xA9 hardware controller
    Frequency range: 47 MHz - 6 GHz
    """
    
    # BladeRF specific constants
    MIN_FREQUENCY = 47_000_000  # 47 MHz
    MAX_FREQUENCY = 6_000_000_000  # 6 GHz
    MIN_SAMPLE_RATE = 520_833  # ~521 kHz
    MAX_SAMPLE_RATE = 61_440_000  # 61.44 MHz
    MIN_BANDWIDTH = 200_000  # 200 kHz
    MAX_BANDWIDTH = 56_000_000  # 56 MHz
    TX_GAIN_MIN = -89
    TX_GAIN_MAX = 60
    RX_GAIN_MIN = 0
    RX_GAIN_MAX = 60
    
    def __init__(self):
        super().__init__()
        self.device: Optional[bladerf.BladeRF] = None
        self._tx_enabled = False
        self._rx_enabled = False
    
    def connect(self) -> bool:
        """Connect to BladeRF device"""
        if not BLADERF_AVAILABLE:
            self._logger.error("BladeRF library not available")
            return False
        
        try:
            self._logger.info("Connecting to BladeRF device...")
            
            # Open first available device
            self.device = bladerf.BladeRF()
            
            # Get device info
            info = self.device.get_device_info()
            self._logger.info(f"Connected to BladeRF: {info}")
            
            # Get firmware version
            fw_version = self.device.get_firmware_version()
            self._logger.info(f"Firmware version: {fw_version}")
            
            # Get FPGA version
            fpga_version = self.device.get_fpga_version()
            self._logger.info(f"FPGA version: {fpga_version}")
            
            self.status = HardwareStatus.CONNECTED
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to connect to BladeRF: {e}")
            self.status = HardwareStatus.ERROR
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from BladeRF device"""
        try:
            if self.device is not None:
                self._logger.info("Disconnecting from BladeRF...")
                
                # Stop any active operations
                self.stop_tx()
                self.stop_rx()
                
                # Close device
                self.device.close()
                self.device = None
                
                self.status = HardwareStatus.DISCONNECTED
                self._logger.info("BladeRF disconnected successfully")
                return True
            
            return True
            
        except Exception as e:
            self._logger.error(f"Error disconnecting from BladeRF: {e}")
            return False
    
    def configure(self, config: HardwareConfig) -> bool:
        """Configure BladeRF with specified parameters"""
        if self.device is None:
            self._logger.error("Device not connected")
            return False
        
        try:
            self._logger.info(
                f"Configuring BladeRF - Freq: {config.frequency/1e6:.2f} MHz, "
                f"SR: {config.sample_rate/1e6:.2f} Msps, "
                f"BW: {config.bandwidth/1e6:.2f} MHz, "
                f"TX Gain: {config.tx_gain} dB, RX Gain: {config.rx_gain} dB"
            )
            
            # Validate parameters
            if not self._validate_config(config):
                return False
            
            # Configure TX channel
            self.device.set_frequency(bladerf.CHANNEL_TX(config.channel), config.frequency)
            self.device.set_sample_rate(bladerf.CHANNEL_TX(config.channel), config.sample_rate)
            self.device.set_bandwidth(bladerf.CHANNEL_TX(config.channel), config.bandwidth)
            self.device.set_gain(bladerf.CHANNEL_TX(config.channel), config.tx_gain)
            
            # Configure RX channel
            self.device.set_frequency(bladerf.CHANNEL_RX(config.channel), config.frequency)
            self.device.set_sample_rate(bladerf.CHANNEL_RX(config.channel), config.sample_rate)
            self.device.set_bandwidth(bladerf.CHANNEL_RX(config.channel), config.bandwidth)
            self.device.set_gain(bladerf.CHANNEL_RX(config.channel), config.rx_gain)
            
            self.config = config
            self._logger.info("BladeRF configuration successful")
            return True
            
        except Exception as e:
            self._logger.error(f"Configuration failed: {e}")
            return False
    
    def start_tx(self) -> bool:
        """Start transmitting"""
        if self.device is None:
            self._logger.error("Device not connected")
            return False
        
        try:
            if not self._tx_enabled:
                channel = bladerf.CHANNEL_TX(0)
                self.device.enable_module(channel, True)
                self._tx_enabled = True
                self.status = HardwareStatus.TRANSMITTING
                self._logger.info("TX started")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to start TX: {e}")
            return False
    
    def stop_tx(self) -> bool:
        """Stop transmitting"""
        if self.device is None:
            return True
        
        try:
            if self._tx_enabled:
                channel = bladerf.CHANNEL_TX(0)
                self.device.enable_module(channel, False)
                self._tx_enabled = False
                self.status = HardwareStatus.CONNECTED
                self._logger.info("TX stopped")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to stop TX: {e}")
            return False
    
    def start_rx(self) -> bool:
        """Start receiving"""
        if self.device is None:
            self._logger.error("Device not connected")
            return False
        
        try:
            if not self._rx_enabled:
                channel = bladerf.CHANNEL_RX(0)
                self.device.enable_module(channel, True)
                self._rx_enabled = True
                self.status = HardwareStatus.RECEIVING
                self._logger.info("RX started")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to start RX: {e}")
            return False
    
    def stop_rx(self) -> bool:
        """Stop receiving"""
        if self.device is None:
            return True
        
        try:
            if self._rx_enabled:
                channel = bladerf.CHANNEL_RX(0)
                self.device.enable_module(channel, False)
                self._rx_enabled = False
                self.status = HardwareStatus.CONNECTED
                self._logger.info("RX stopped")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to stop RX: {e}")
            return False
    
    def transmit(self, samples: bytes, num_samples: int) -> bool:
        """Transmit IQ samples"""
        if self.device is None or not self._tx_enabled:
            self._logger.error("TX not ready")
            return False
        
        try:
            # Sync transmit
            channel = bladerf.CHANNEL_TX(0)
            self.device.sync_tx(samples, num_samples)
            return True
            
        except Exception as e:
            self._logger.error(f"Transmission failed: {e}")
            return False
    
    def receive(self, num_samples: int) -> Optional[bytes]:
        """Receive IQ samples"""
        if self.device is None or not self._rx_enabled:
            self._logger.error("RX not ready")
            return None
        
        try:
            # Sync receive
            channel = bladerf.CHANNEL_RX(0)
            samples = self.device.sync_rx(num_samples)
            return samples
            
        except Exception as e:
            self._logger.error(f"Reception failed: {e}")
            return None
    
    def get_status(self) -> dict:
        """Get current hardware status"""
        status = {
            'status': self.status.value,
            'connected': self.device is not None,
            'config': self.config,
            'hardware_type': 'BladeRF 2.0 micro xA9',
            'tx_enabled': self._tx_enabled,
            'rx_enabled': self._rx_enabled
        }
        
        if self.device is not None:
            try:
                status['firmware_version'] = str(self.device.get_firmware_version())
                status['fpga_version'] = str(self.device.get_fpga_version())
                status['device_info'] = str(self.device.get_device_info())
            except Exception as e:
                self._logger.error(f"Failed to get device info: {e}")
        
        return status
    
    def emergency_shutdown(self) -> bool:
        """Emergency shutdown - immediately stop all RF activity"""
        self._logger.warning("EMERGENCY SHUTDOWN INITIATED")
        
        try:
            # Stop TX first (safety critical)
            if self.device is not None and self._tx_enabled:
                channel = bladerf.CHANNEL_TX(0)
                self.device.enable_module(channel, False)
                # Set TX gain to minimum
                self.device.set_gain(channel, self.TX_GAIN_MIN)
                self._tx_enabled = False
            
            # Stop RX
            if self.device is not None and self._rx_enabled:
                channel = bladerf.CHANNEL_RX(0)
                self.device.enable_module(channel, False)
                self._rx_enabled = False
            
            self.status = HardwareStatus.CONNECTED
            self._logger.warning("Emergency shutdown completed")
            return True
            
        except Exception as e:
            self._logger.error(f"Emergency shutdown failed: {e}")
            return False
    
    def _validate_config(self, config: HardwareConfig) -> bool:
        """Validate configuration against hardware limits"""
        # Validate frequency
        if not (self.MIN_FREQUENCY <= config.frequency <= self.MAX_FREQUENCY):
            self._logger.error(
                f"Frequency {config.frequency/1e6:.2f} MHz outside range "
                f"({self.MIN_FREQUENCY/1e6:.0f}-{self.MAX_FREQUENCY/1e6:.0f} MHz)"
            )
            return False
        
        # Validate sample rate
        if not (self.MIN_SAMPLE_RATE <= config.sample_rate <= self.MAX_SAMPLE_RATE):
            self._logger.error(
                f"Sample rate {config.sample_rate/1e6:.2f} Msps outside range "
                f"({self.MIN_SAMPLE_RATE/1e6:.2f}-{self.MAX_SAMPLE_RATE/1e6:.2f} Msps)"
            )
            return False
        
        # Validate bandwidth
        if not (self.MIN_BANDWIDTH <= config.bandwidth <= self.MAX_BANDWIDTH):
            self._logger.error(
                f"Bandwidth {config.bandwidth/1e6:.2f} MHz outside range "
                f"({self.MIN_BANDWIDTH/1e6:.2f}-{self.MAX_BANDWIDTH/1e6:.2f} MHz)"
            )
            return False
        
        # Validate TX gain
        if not (self.TX_GAIN_MIN <= config.tx_gain <= self.TX_GAIN_MAX):
            self._logger.error(
                f"TX gain {config.tx_gain} dB outside range "
                f"({self.TX_GAIN_MIN}-{self.TX_GAIN_MAX} dB)"
            )
            return False
        
        # Validate RX gain
        if not (self.RX_GAIN_MIN <= config.rx_gain <= self.RX_GAIN_MAX):
            self._logger.error(
                f"RX gain {config.rx_gain} dB outside range "
                f"({self.RX_GAIN_MIN}-{self.RX_GAIN_MAX} dB)"
            )
            return False
        
        return True
