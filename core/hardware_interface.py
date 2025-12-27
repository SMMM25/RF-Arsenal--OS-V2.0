#!/usr/bin/env python3
"""
RF Arsenal OS - Hardware Abstraction Layer
Provides a unified interface for SDR hardware with mock support for testing
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class HardwareStatus(Enum):
    """Hardware operational status"""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    TRANSMITTING = "transmitting"
    RECEIVING = "receiving"
    ERROR = "error"


@dataclass
class HardwareConfig:
    """Hardware configuration parameters"""
    frequency: int  # Hz
    sample_rate: int  # Samples per second
    bandwidth: int  # Hz
    tx_gain: int  # dB
    rx_gain: int  # dB
    channel: int = 0
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.frequency < 0:
            raise ValueError(f"Frequency must be positive, got {self.frequency}")
        if self.sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {self.sample_rate}")
        if self.bandwidth <= 0:
            raise ValueError(f"Bandwidth must be positive, got {self.bandwidth}")


@dataclass
class FrequencyBand:
    """Pre-defined frequency bands for common operations"""
    # HF/VHF/UHF
    HF = (3_000_000, 30_000_000)  # 3-30 MHz
    VHF = (30_000_000, 300_000_000)  # 30-300 MHz
    UHF = (300_000_000, 3_000_000_000)  # 0.3-3 GHz
    
    # Cellular bands
    GSM_900 = (890_000_000, 960_000_000)
    GSM_1800 = (1_710_000_000, 1_880_000_000)
    UMTS_2100 = (1_920_000_000, 2_170_000_000)
    LTE_BAND_7 = (2_500_000_000, 2_690_000_000)
    
    # WiFi
    WIFI_2_4GHZ = (2_400_000_000, 2_500_000_000)
    WIFI_5GHZ = (5_150_000_000, 5_850_000_000)
    
    # GPS
    GPS_L1 = (1_575_420_000, 1_575_420_000)
    GPS_L2 = (1_227_600_000, 1_227_600_000)
    
    # Drones
    DRONE_2_4GHZ = (2_400_000_000, 2_483_500_000)
    DRONE_5_8GHZ = (5_725_000_000, 5_850_000_000)


class HardwareInterface(ABC):
    """
    Abstract base class for all SDR hardware implementations.
    Provides a unified API for BladeRF, HackRF, LimeSDR, etc.
    """
    
    def __init__(self):
        self.status = HardwareStatus.DISCONNECTED
        self.config: Optional[HardwareConfig] = None
        self._logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the hardware device
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the hardware device
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def configure(self, config: HardwareConfig) -> bool:
        """
        Configure hardware with specified parameters
        
        Args:
            config: Hardware configuration parameters
        
        Returns:
            True if configuration successful, False otherwise
        """
        pass
    
    @abstractmethod
    def start_tx(self) -> bool:
        """
        Start transmitting
        
        Returns:
            True if TX started successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def stop_tx(self) -> bool:
        """
        Stop transmitting
        
        Returns:
            True if TX stopped successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def start_rx(self) -> bool:
        """
        Start receiving
        
        Returns:
            True if RX started successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def stop_rx(self) -> bool:
        """
        Stop receiving
        
        Returns:
            True if RX stopped successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def transmit(self, samples: bytes, num_samples: int) -> bool:
        """
        Transmit samples
        
        Args:
            samples: IQ samples to transmit
            num_samples: Number of samples
        
        Returns:
            True if transmission successful, False otherwise
        """
        pass
    
    @abstractmethod
    def receive(self, num_samples: int) -> Optional[bytes]:
        """
        Receive samples
        
        Args:
            num_samples: Number of samples to receive
        
        Returns:
            IQ samples if successful, None otherwise
        """
        pass
    
    @abstractmethod
    def get_status(self) -> dict:
        """
        Get current hardware status
        
        Returns:
            Dictionary containing status information
        """
        pass
    
    @abstractmethod
    def emergency_shutdown(self) -> bool:
        """
        Emergency shutdown - immediately stop all RF activity
        
        Returns:
            True if shutdown successful, False otherwise
        """
        pass
    
    def validate_frequency(self, frequency: int) -> bool:
        """
        Validate that frequency is within safe operating range
        
        Args:
            frequency: Frequency in Hz
        
        Returns:
            True if valid, False otherwise
        """
        # BladeRF 2.0 micro xA9 range: 47 MHz - 6 GHz
        MIN_FREQ = 47_000_000  # 47 MHz
        MAX_FREQ = 6_000_000_000  # 6 GHz
        
        if not (MIN_FREQ <= frequency <= MAX_FREQ):
            self._logger.error(
                f"Frequency {frequency/1e6:.2f} MHz outside safe range "
                f"({MIN_FREQ/1e6:.0f}-{MAX_FREQ/1e6:.0f} MHz)"
            )
            return False
        
        return True
    
    def validate_gain(self, gain: int, gain_type: str = "tx") -> bool:
        """
        Validate gain value
        
        Args:
            gain: Gain in dB
            gain_type: "tx" or "rx"
        
        Returns:
            True if valid, False otherwise
        """
        if gain_type == "tx":
            # TX gain range: -89 to 60 dB (typical for BladeRF)
            if not (-89 <= gain <= 60):
                self._logger.error(f"TX gain {gain} dB outside range (-89 to 60 dB)")
                return False
        elif gain_type == "rx":
            # RX gain range: 0 to 60 dB
            if not (0 <= gain <= 60):
                self._logger.error(f"RX gain {gain} dB outside range (0 to 60 dB)")
                return False
        
        return True


class NoHardwareFallback(HardwareInterface):
    """
    No-hardware fallback implementation for testing and development without physical SDR
    """
    
    def __init__(self):
        super().__init__()
        self._logger.info("Initializing No Hardware Fallback (No physical SDR)")
    
    def connect(self) -> bool:
        """No-hardware connect"""
        self._logger.info("[NO HW]: Connecting to hardware...")
        self.status = HardwareStatus.CONNECTED
        return True
    
    def disconnect(self) -> bool:
        """No-hardware disconnect"""
        self._logger.info("[NO HW]: Disconnecting from hardware...")
        self.status = HardwareStatus.DISCONNECTED
        return True
    
    def configure(self, config: HardwareConfig) -> bool:
        """No-hardware configure"""
        self._logger.info(
            f"[NO HW]: Configuring - Freq: {config.frequency/1e6:.2f} MHz, "
            f"SR: {config.sample_rate/1e6:.2f} Msps, "
            f"BW: {config.bandwidth/1e6:.2f} MHz"
        )
        
        # Validate configuration
        if not self.validate_frequency(config.frequency):
            return False
        if not self.validate_gain(config.tx_gain, "tx"):
            return False
        if not self.validate_gain(config.rx_gain, "rx"):
            return False
        
        self.config = config
        return True
    
    def start_tx(self) -> bool:
        """No-hardware start TX"""
        self._logger.info("[NO HW]: Starting TX")
        self.status = HardwareStatus.TRANSMITTING
        return True
    
    def stop_tx(self) -> bool:
        """No-hardware stop TX"""
        self._logger.info("[NO HW]: Stopping TX")
        self.status = HardwareStatus.CONNECTED
        return True
    
    def start_rx(self) -> bool:
        """No-hardware start RX"""
        self._logger.info("[NO HW]: Starting RX")
        self.status = HardwareStatus.RECEIVING
        return True
    
    def stop_rx(self) -> bool:
        """No-hardware stop RX"""
        self._logger.info("[NO HW]: Stopping RX")
        self.status = HardwareStatus.CONNECTED
        return True
    
    def transmit(self, samples: bytes, num_samples: int) -> bool:
        """No-hardware transmit"""
        self._logger.debug(f"[NO HW]: Transmitting {num_samples} samples")
        return True
    
    def receive(self, num_samples: int) -> Optional[bytes]:
        """No-hardware receive"""
        self._logger.debug(f"[NO HW]: Receiving {num_samples} samples")
        # Return mock data
        return bytes(num_samples * 4)  # 4 bytes per IQ sample (I16, Q16)
    
    def get_status(self) -> dict:
        """No-hardware get status"""
        return {
            'status': self.status.value,
            'connected': self.status != HardwareStatus.DISCONNECTED,
            'config': self.config,
            'hardware_type': 'Mock',
            'firmware_version': 'Mock v1.0'
        }
    
    def emergency_shutdown(self) -> bool:
        """No-hardware emergency shutdown"""
        self._logger.warning("[NO HW]: EMERGENCY SHUTDOWN")
        self.stop_tx()
        self.stop_rx()
        self.status = HardwareStatus.DISCONNECTED
        return True
