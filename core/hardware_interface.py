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


class HardwareRequiredError(Exception):
    """
    Exception raised when SDR hardware is required but not available.
    
    README COMPLIANCE: Rule #5 - Real-World Functional Only
    No simulation modes or mock data in production.
    """
    pass


class NoHardwareFallback(HardwareInterface):
    """
    Hardware requirement enforcer - raises errors when no SDR is connected.
    
    README COMPLIANCE: Rule #5 - Real-World Functional Only
    This class does NOT provide mock data. It raises HardwareRequiredError
    for any operation that requires actual hardware.
    """
    
    def __init__(self):
        super().__init__()
        self._logger.warning(
            "HARDWARE REQUIRED: No SDR hardware detected. "
            "Connect BladeRF, HackRF, or compatible SDR to use RF features."
        )
        self.status = HardwareStatus.DISCONNECTED
    
    def _raise_hardware_required(self, operation: str):
        """Raise HardwareRequiredError with helpful message"""
        raise HardwareRequiredError(
            f"HARDWARE REQUIRED: Cannot perform '{operation}' without SDR hardware.\n"
            f"  Supported devices: BladeRF 2.0 micro xA9, HackRF One, RTL-SDR, USRP\n"
            f"  Install drivers: sudo apt install libbladerf-dev hackrf librtlsdr-dev\n"
            f"  Then reconnect your SDR device."
        )
    
    def connect(self) -> bool:
        """Raise error - hardware required"""
        self._raise_hardware_required("connect")
        return False
    
    def disconnect(self) -> bool:
        """Disconnect is safe without hardware"""
        self.status = HardwareStatus.DISCONNECTED
        return True
    
    def configure(self, config: HardwareConfig) -> bool:
        """Raise error - hardware required"""
        self._raise_hardware_required("configure")
        return False
    
    def start_tx(self) -> bool:
        """Raise error - hardware required"""
        self._raise_hardware_required("start_tx")
        return False
    
    def stop_tx(self) -> bool:
        """Stop TX is safe without hardware"""
        return True
    
    def start_rx(self) -> bool:
        """Raise error - hardware required"""
        self._raise_hardware_required("start_rx")
        return False
    
    def stop_rx(self) -> bool:
        """Stop RX is safe without hardware"""
        return True
    
    def transmit(self, samples: bytes, num_samples: int) -> bool:
        """Raise error - hardware required"""
        self._raise_hardware_required("transmit")
        return False
    
    def receive(self, num_samples: int) -> Optional[bytes]:
        """Raise error - hardware required for RF reception"""
        self._raise_hardware_required("receive")
        return None
    
    def get_status(self) -> dict:
        """Return status indicating no hardware"""
        return {
            'status': 'no_hardware',
            'connected': False,
            'config': None,
            'hardware_type': None,
            'error': 'HARDWARE REQUIRED: No SDR device connected'
        }
    
    def emergency_shutdown(self) -> bool:
        """Emergency shutdown is safe without hardware"""
        self.status = HardwareStatus.DISCONNECTED
        return True
