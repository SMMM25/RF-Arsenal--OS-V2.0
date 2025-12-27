"""
RF Arsenal OS - Thread-Safe Hardware Control Module
BladeRF 2.0 micro xA9 Controller with Singleton Pattern

CRITICAL: AUTHORIZED USE ONLY
This module controls SDR hardware for authorized penetration testing.
"""

import time
import logging
import threading
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Import from hardware abstraction layer
from .hardware_interface import (
    HardwareInterface, 
    HardwareConfig, 
    HardwareStatus,
    FrequencyBand,
    NoHardwareFallback
)

# Try to import BladeRF hardware
try:
    from .hardware_bladerf import BladeRFHardware
    BLADERF_AVAILABLE = True
except ImportError:
    BLADERF_AVAILABLE = False
    logging.warning("BladeRF hardware not available - using no-hardware mode")


class BladeRFController:
    """
    Thread-Safe Singleton Hardware Controller
    
    Wraps hardware interface with thread-safe operations and singleton pattern.
    Automatically selects BladeRF hardware or no-hardware fallback based on availability.
    
    Capabilities:
    - Frequency: 47 MHz - 6 GHz
    - 2x2 MIMO (2 TX, 2 RX channels)
    - Sample rate: up to 61.44 MHz
    - Bandwidth: 200 kHz - 56 MHz
    - USB 3.0 interface
    """
    
    _instance: Optional['BladeRFController'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern - only one instance allowed"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize hardware controller (called only once due to singleton)"""
        # Prevent re-initialization
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._operation_lock = threading.RLock()  # Reentrant lock for nested operations
        self.logger = logging.getLogger(__name__)
        
        # Select hardware implementation
        if BLADERF_AVAILABLE:
            self.logger.info("Initializing BladeRF hardware controller")
            self.hardware: HardwareInterface = BladeRFHardware()
        else:
            self.logger.warning("BladeRF not available - using no-hardware fallback")
            self.hardware: HardwareInterface = NoHardwareFallback()
    
    @classmethod
    def get_instance(cls) -> 'BladeRFController':
        """Get the singleton instance"""
        return cls()
    
    def connect(self) -> bool:
        """Thread-safe connect to hardware"""
        with self._operation_lock:
            return self.hardware.connect()
    
    def disconnect(self) -> bool:
        """Thread-safe disconnect from hardware"""
        with self._operation_lock:
            return self.hardware.disconnect()
    
    def configure(self, config: HardwareConfig) -> bool:
        """Thread-safe hardware configuration"""
        with self._operation_lock:
            return self.hardware.configure(config)
    
    def start_tx(self) -> bool:
        """Thread-safe start transmission"""
        with self._operation_lock:
            return self.hardware.start_tx()
    
    def stop_tx(self) -> bool:
        """Thread-safe stop transmission"""
        with self._operation_lock:
            return self.hardware.stop_tx()
    
    def start_rx(self) -> bool:
        """Thread-safe start reception"""
        with self._operation_lock:
            return self.hardware.start_rx()
    
    def stop_rx(self) -> bool:
        """Thread-safe stop reception"""
        with self._operation_lock:
            return self.hardware.stop_rx()
    
    def transmit(self, samples: bytes, num_samples: int) -> bool:
        """Thread-safe transmit samples"""
        with self._operation_lock:
            return self.hardware.transmit(samples, num_samples)
    
    def receive(self, num_samples: int) -> Optional[bytes]:
        """Thread-safe receive samples"""
        with self._operation_lock:
            return self.hardware.receive(num_samples)
    
    def get_status(self) -> Dict:
        """Thread-safe get hardware status"""
        with self._operation_lock:
            return self.hardware.get_status()
    
    def emergency_shutdown(self) -> bool:
        """Thread-safe emergency shutdown"""
        with self._operation_lock:
            return self.hardware.emergency_shutdown()
    
    @property
    def is_connected(self) -> bool:
        """Check if hardware is connected"""
        status = self.get_status()
        return status.get('connected', False)
    
    @property
    def current_config(self) -> Optional[HardwareConfig]:
        """Get current configuration"""
        status = self.get_status()
        return status.get('config')
    
    @property
    def tx_active(self) -> bool:
        """Check if TX is active"""
        return self.hardware.status == HardwareStatus.TRANSMITTING
    
    @property
    def rx_active(self) -> bool:
        """Check if RX is active"""
        return self.hardware.status == HardwareStatus.RECEIVING


class HardwarePresets:
    """Predefined hardware configurations for common operations"""
    
    @staticmethod
    def cellular_2g(band: int = 900) -> HardwareConfig:
        """2G/GSM configuration"""
        return HardwareConfig(
            frequency=band * 1_000_000,  # 850/900/1800/1900 MHz
            sample_rate=2_000_000,       # 2 MHz
            bandwidth=200_000,           # 200 kHz
            tx_gain=30,
            rx_gain=40
        )
    
    @staticmethod
    def cellular_4g(band: int = 1800) -> HardwareConfig:
        """4G/LTE configuration"""
        return HardwareConfig(
            frequency=band * 1_000_000,
            sample_rate=10_000_000,      # 10 MHz
            bandwidth=10_000_000,        # 10 MHz
            tx_gain=35,
            rx_gain=40
        )
    
    @staticmethod
    def cellular_5g(frequency: int = 3_500_000_000) -> HardwareConfig:
        """5G NR configuration"""
        return HardwareConfig(
            frequency=frequency,
            sample_rate=30_000_000,      # 30 MHz
            bandwidth=30_000_000,        # 30 MHz
            tx_gain=35,
            rx_gain=40
        )
    
    @staticmethod
    def wifi_2_4ghz() -> HardwareConfig:
        """WiFi 2.4 GHz configuration"""
        return HardwareConfig(
            frequency=2_437_000_000,     # Channel 6
            sample_rate=20_000_000,      # 20 MHz
            bandwidth=20_000_000,        # 20 MHz
            tx_gain=30,
            rx_gain=40
        )
    
    @staticmethod
    def wifi_5ghz() -> HardwareConfig:
        """WiFi 5 GHz configuration"""
        return HardwareConfig(
            frequency=5_180_000_000,     # Channel 36
            sample_rate=40_000_000,      # 40 MHz
            bandwidth=40_000_000,        # 40 MHz
            tx_gain=30,
            rx_gain=40
        )
    
    @staticmethod
    def gps_l1() -> HardwareConfig:
        """GPS L1 configuration"""
        return HardwareConfig(
            frequency=1_575_420_000,     # L1 C/A
            sample_rate=2_500_000,       # 2.5 MHz
            bandwidth=2_500_000,         # 2.5 MHz
            tx_gain=25,
            rx_gain=45
        )
    
    @staticmethod
    def drone_2_4ghz() -> HardwareConfig:
        """Drone 2.4 GHz configuration"""
        return HardwareConfig(
            frequency=2_400_000_000,
            sample_rate=20_000_000,
            bandwidth=20_000_000,
            tx_gain=35,
            rx_gain=40
        )
    
    @staticmethod
    def spectrum_analyzer(start_freq: int = 400_000_000, 
                         bandwidth: int = 20_000_000) -> HardwareConfig:
        """Spectrum analyzer configuration"""
        return HardwareConfig(
            frequency=start_freq,
            sample_rate=bandwidth,
            bandwidth=bandwidth,
            tx_gain=0,
            rx_gain=50
        )


# Singleton instance
_hardware_controller: Optional[BladeRFController] = None

def get_hardware_controller() -> BladeRFController:
    """Get the global hardware controller instance"""
    global _hardware_controller
    if _hardware_controller is None:
        _hardware_controller = BladeRFController()
    return _hardware_controller
