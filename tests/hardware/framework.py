"""
Hardware Integration Test Framework.

Core framework for comprehensive hardware testing with real RF devices.
Provides infrastructure for test discovery, execution, fixture management,
hardware abstraction, and result reporting.

Features:
- Automatic hardware discovery and initialization
- Test isolation and cleanup
- Parallel test execution support
- Hardware capability detection
- Test dependency management
- Detailed logging and reporting
- CI/CD integration support

Author: RF Arsenal Development Team
License: Proprietary
"""

import asyncio
import functools
import hashlib
import inspect
import json
import logging
import os
import platform
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto, Flag
from io import StringIO
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generator, Generic, List, Optional, 
    Set, Tuple, Type, TypeVar, Union
)

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Enumerations
# ============================================================================

class TestCategory(Enum):
    """Categories of hardware tests."""
    
    SMOKE = "smoke"                      # Quick sanity checks
    FUNCTIONAL = "functional"            # Basic functionality
    PERFORMANCE = "performance"          # Performance benchmarks
    STRESS = "stress"                    # Stress and reliability
    CALIBRATION = "calibration"          # Calibration verification
    DIAGNOSTIC = "diagnostic"            # Hardware diagnostics
    INTEGRATION = "integration"          # Integration tests
    E2E = "e2e"                          # End-to-end tests
    REGRESSION = "regression"            # Regression tests
    ACCEPTANCE = "acceptance"            # Acceptance criteria


class TestPriority(Enum):
    """Test execution priority."""
    
    CRITICAL = 1      # Must pass for any release
    HIGH = 2          # Important functionality
    MEDIUM = 3        # Standard tests
    LOW = 4           # Nice-to-have validation
    OPTIONAL = 5      # Optional/experimental


class TestStatus(Enum):
    """Test execution status."""
    
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"


class HardwareCapability(Flag):
    """Hardware capability flags."""
    
    NONE = 0
    TRANSMIT = auto()
    RECEIVE = auto()
    FULL_DUPLEX = auto()
    HALF_DUPLEX = auto()
    FREQUENCY_HOP = auto()
    WIDEBAND = auto()
    NARROWBAND = auto()
    FPGA_ACCELERATION = auto()
    GPIO = auto()
    CLOCK_SYNC = auto()
    MIMO = auto()
    EXTERNAL_CLOCK = auto()
    BIAS_TEE = auto()
    PREAMP = auto()
    ATTENUATOR = auto()
    CALIBRATION = auto()
    SPECTRUM_ANALYZER = auto()
    SIGNAL_GENERATOR = auto()


class DeviceType(Enum):
    """Types of RF hardware devices."""
    
    SDR = "sdr"
    SPECTRUM_ANALYZER = "spectrum_analyzer"
    SIGNAL_GENERATOR = "signal_generator"
    POWER_METER = "power_meter"
    NETWORK_ANALYZER = "network_analyzer"
    OSCILLOSCOPE = "oscilloscope"
    FREQUENCY_COUNTER = "frequency_counter"
    ATTENUATOR = "attenuator"
    SWITCH_MATRIX = "switch_matrix"
    AMPLIFIER = "amplifier"
    FILTER = "filter"


class ConnectionType(Enum):
    """Hardware connection types."""
    
    USB = "usb"
    ETHERNET = "ethernet"
    PCIE = "pcie"
    GPIB = "gpib"
    SERIAL = "serial"
    WIFI = "wifi"
    BLUETOOTH = "bluetooth"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DeviceInfo:
    """Information about a hardware device."""
    
    device_id: str
    device_type: DeviceType
    manufacturer: str
    model: str
    serial_number: str
    firmware_version: str = ""
    hardware_revision: str = ""
    driver_version: str = ""
    connection_type: ConnectionType = ConnectionType.USB
    connection_info: Dict[str, Any] = field(default_factory=dict)
    capabilities: HardwareCapability = HardwareCapability.NONE
    frequency_range: Tuple[float, float] = (0.0, 0.0)
    bandwidth_range: Tuple[float, float] = (0.0, 0.0)
    sample_rate_range: Tuple[float, float] = (0.0, 0.0)
    gain_range: Tuple[float, float] = (0.0, 0.0)
    power_range: Tuple[float, float] = (0.0, 0.0)
    is_available: bool = True
    last_seen: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'device_id': self.device_id,
            'device_type': self.device_type.value,
            'manufacturer': self.manufacturer,
            'model': self.model,
            'serial_number': self.serial_number,
            'firmware_version': self.firmware_version,
            'hardware_revision': self.hardware_revision,
            'driver_version': self.driver_version,
            'connection_type': self.connection_type.value,
            'connection_info': self.connection_info,
            'capabilities': self.capabilities.value,
            'frequency_range': self.frequency_range,
            'bandwidth_range': self.bandwidth_range,
            'sample_rate_range': self.sample_rate_range,
            'gain_range': self.gain_range,
            'power_range': self.power_range,
            'is_available': self.is_available,
            'last_seen': self.last_seen.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class DeviceCapabilities:
    """Detailed device capabilities."""
    
    device_id: str
    
    # Frequency capabilities
    min_frequency_hz: float = 0.0
    max_frequency_hz: float = 0.0
    frequency_resolution_hz: float = 1.0
    frequency_accuracy_ppm: float = 0.0
    
    # Bandwidth capabilities
    min_bandwidth_hz: float = 0.0
    max_bandwidth_hz: float = 0.0
    bandwidth_steps: List[float] = field(default_factory=list)
    
    # Sample rate capabilities
    min_sample_rate_sps: float = 0.0
    max_sample_rate_sps: float = 0.0
    sample_rate_steps: List[float] = field(default_factory=list)
    
    # Gain capabilities
    min_gain_db: float = 0.0
    max_gain_db: float = 0.0
    gain_step_db: float = 1.0
    agc_supported: bool = False
    
    # Power capabilities
    min_power_dbm: float = -100.0
    max_power_dbm: float = 20.0
    power_accuracy_db: float = 1.0
    
    # Channel capabilities
    num_rx_channels: int = 1
    num_tx_channels: int = 1
    full_duplex: bool = False
    
    # Digital capabilities
    adc_bits: int = 8
    dac_bits: int = 8
    fpga_type: str = ""
    fpga_size: int = 0
    
    # Special features
    bias_tee: bool = False
    external_clock_input: bool = False
    pps_input: bool = False
    gpio_pins: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'device_id': self.device_id,
            'frequency': {
                'min_hz': self.min_frequency_hz,
                'max_hz': self.max_frequency_hz,
                'resolution_hz': self.frequency_resolution_hz,
                'accuracy_ppm': self.frequency_accuracy_ppm
            },
            'bandwidth': {
                'min_hz': self.min_bandwidth_hz,
                'max_hz': self.max_bandwidth_hz,
                'steps': self.bandwidth_steps
            },
            'sample_rate': {
                'min_sps': self.min_sample_rate_sps,
                'max_sps': self.max_sample_rate_sps,
                'steps': self.sample_rate_steps
            },
            'gain': {
                'min_db': self.min_gain_db,
                'max_db': self.max_gain_db,
                'step_db': self.gain_step_db,
                'agc': self.agc_supported
            },
            'power': {
                'min_dbm': self.min_power_dbm,
                'max_dbm': self.max_power_dbm,
                'accuracy_db': self.power_accuracy_db
            },
            'channels': {
                'rx': self.num_rx_channels,
                'tx': self.num_tx_channels,
                'full_duplex': self.full_duplex
            },
            'digital': {
                'adc_bits': self.adc_bits,
                'dac_bits': self.dac_bits,
                'fpga_type': self.fpga_type,
                'fpga_size': self.fpga_size
            },
            'features': {
                'bias_tee': self.bias_tee,
                'external_clock': self.external_clock_input,
                'pps_input': self.pps_input,
                'gpio_pins': self.gpio_pins
            }
        }


@dataclass
class TestResult:
    """Result of a single test execution."""
    
    test_id: str
    test_name: str
    status: TestStatus
    category: TestCategory
    priority: TestPriority
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Results
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    measurements: Dict[str, Any] = field(default_factory=dict)
    
    # Error information
    error_type: str = ""
    error_message: str = ""
    stack_trace: str = ""
    
    # Hardware context
    device_id: str = ""
    device_info: Optional[DeviceInfo] = None
    
    # Artifacts
    logs: List[str] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    data_files: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate duration if end time is set."""
        if self.end_time and self.duration_seconds == 0:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
    
    @property
    def passed(self) -> bool:
        """Check if test passed."""
        return self.status == TestStatus.PASSED
    
    @property
    def failed(self) -> bool:
        """Check if test failed."""
        return self.status in [TestStatus.FAILED, TestStatus.ERROR, TestStatus.TIMEOUT]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'test_id': self.test_id,
            'test_name': self.test_name,
            'status': self.status.value,
            'category': self.category.value,
            'priority': self.priority.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'message': self.message,
            'details': self.details,
            'measurements': self.measurements,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'stack_trace': self.stack_trace,
            'device_id': self.device_id,
            'device_info': self.device_info.to_dict() if self.device_info else None,
            'logs': self.logs,
            'passed': self.passed
        }


@dataclass
class HardwareTestConfig:
    """Configuration for hardware tests."""
    
    # Test selection
    categories: List[TestCategory] = field(default_factory=lambda: [TestCategory.FUNCTIONAL])
    priorities: List[TestPriority] = field(default_factory=lambda: [
        TestPriority.CRITICAL, TestPriority.HIGH, TestPriority.MEDIUM
    ])
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    
    # Hardware selection
    device_types: List[DeviceType] = field(default_factory=list)
    device_ids: List[str] = field(default_factory=list)
    required_capabilities: HardwareCapability = HardwareCapability.NONE
    
    # Execution settings
    timeout_seconds: float = 60.0
    retry_count: int = 0
    retry_delay_seconds: float = 1.0
    parallel_execution: bool = False
    max_parallel_tests: int = 4
    fail_fast: bool = False
    
    # Environment
    temp_directory: str = ""
    output_directory: str = ""
    log_level: str = "INFO"
    capture_logs: bool = True
    save_artifacts: bool = True
    
    # Reporting
    report_format: str = "html"
    generate_junit: bool = True
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'categories': [c.value for c in self.categories],
            'priorities': [p.value for p in self.priorities],
            'include_patterns': self.include_patterns,
            'exclude_patterns': self.exclude_patterns,
            'device_types': [d.value for d in self.device_types],
            'device_ids': self.device_ids,
            'required_capabilities': self.required_capabilities.value,
            'timeout_seconds': self.timeout_seconds,
            'retry_count': self.retry_count,
            'parallel_execution': self.parallel_execution,
            'fail_fast': self.fail_fast,
            'report_format': self.report_format,
            'verbose': self.verbose
        }


@dataclass
class TestEnvironment:
    """Test execution environment information."""
    
    # System info
    hostname: str = field(default_factory=lambda: platform.node())
    platform: str = field(default_factory=lambda: platform.system())
    platform_version: str = field(default_factory=lambda: platform.version())
    architecture: str = field(default_factory=lambda: platform.machine())
    python_version: str = field(default_factory=lambda: platform.python_version())
    
    # Hardware info
    cpu_count: int = field(default_factory=lambda: os.cpu_count() or 1)
    memory_gb: float = 0.0
    
    # RF Arsenal info
    rf_arsenal_version: str = ""
    rf_arsenal_commit: str = ""
    
    # Test run info
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Connected devices
    connected_devices: List[DeviceInfo] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'hostname': self.hostname,
            'platform': self.platform,
            'platform_version': self.platform_version,
            'architecture': self.architecture,
            'python_version': self.python_version,
            'cpu_count': self.cpu_count,
            'memory_gb': self.memory_gb,
            'rf_arsenal_version': self.rf_arsenal_version,
            'rf_arsenal_commit': self.rf_arsenal_commit,
            'run_id': self.run_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'connected_devices': [d.to_dict() for d in self.connected_devices]
        }


# ============================================================================
# Hardware Interfaces
# ============================================================================

class HardwareInterface(ABC):
    """Abstract base class for hardware interfaces."""
    
    def __init__(self, device_info: DeviceInfo):
        """Initialize hardware interface."""
        self.device_info = device_info
        self._connected = False
        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @property
    def device_id(self) -> str:
        """Get device ID."""
        return self.device_info.device_id
    
    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to hardware."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from hardware."""
        pass
    
    @abstractmethod
    def reset(self) -> bool:
        """Reset hardware to default state."""
        pass
    
    @abstractmethod
    def self_test(self) -> Tuple[bool, str]:
        """Perform hardware self-test."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current hardware status."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class SDRInterface(HardwareInterface):
    """Interface for Software Defined Radio devices."""
    
    @abstractmethod
    def set_frequency(self, frequency_hz: float) -> bool:
        """Set center frequency."""
        pass
    
    @abstractmethod
    def get_frequency(self) -> float:
        """Get current center frequency."""
        pass
    
    @abstractmethod
    def set_sample_rate(self, sample_rate_sps: float) -> bool:
        """Set sample rate."""
        pass
    
    @abstractmethod
    def get_sample_rate(self) -> float:
        """Get current sample rate."""
        pass
    
    @abstractmethod
    def set_bandwidth(self, bandwidth_hz: float) -> bool:
        """Set RF bandwidth."""
        pass
    
    @abstractmethod
    def get_bandwidth(self) -> float:
        """Get current RF bandwidth."""
        pass
    
    @abstractmethod
    def set_gain(self, gain_db: float, channel: int = 0) -> bool:
        """Set gain."""
        pass
    
    @abstractmethod
    def get_gain(self, channel: int = 0) -> float:
        """Get current gain."""
        pass
    
    @abstractmethod
    def start_rx(self, num_samples: int = 0) -> bool:
        """Start receiving samples."""
        pass
    
    @abstractmethod
    def stop_rx(self) -> None:
        """Stop receiving."""
        pass
    
    @abstractmethod
    def read_samples(self, num_samples: int) -> np.ndarray:
        """Read IQ samples."""
        pass
    
    @abstractmethod
    def start_tx(self) -> bool:
        """Start transmitting."""
        pass
    
    @abstractmethod
    def stop_tx(self) -> None:
        """Stop transmitting."""
        pass
    
    @abstractmethod
    def write_samples(self, samples: np.ndarray) -> int:
        """Write IQ samples for transmission."""
        pass


class SignalGeneratorInterface(HardwareInterface):
    """Interface for signal generator devices."""
    
    @abstractmethod
    def set_frequency(self, frequency_hz: float) -> bool:
        """Set output frequency."""
        pass
    
    @abstractmethod
    def set_power(self, power_dbm: float) -> bool:
        """Set output power."""
        pass
    
    @abstractmethod
    def set_modulation(self, mod_type: str, params: Dict[str, Any]) -> bool:
        """Set modulation parameters."""
        pass
    
    @abstractmethod
    def output_enable(self, enable: bool) -> bool:
        """Enable/disable RF output."""
        pass


class SpectrumAnalyzerInterface(HardwareInterface):
    """Interface for spectrum analyzer devices."""
    
    @abstractmethod
    def set_center_frequency(self, frequency_hz: float) -> bool:
        """Set center frequency."""
        pass
    
    @abstractmethod
    def set_span(self, span_hz: float) -> bool:
        """Set frequency span."""
        pass
    
    @abstractmethod
    def set_rbw(self, rbw_hz: float) -> bool:
        """Set resolution bandwidth."""
        pass
    
    @abstractmethod
    def set_reference_level(self, level_dbm: float) -> bool:
        """Set reference level."""
        pass
    
    @abstractmethod
    def get_trace(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get frequency and power trace data."""
        pass
    
    @abstractmethod
    def measure_peak(self) -> Tuple[float, float]:
        """Measure peak frequency and power."""
        pass


# ============================================================================
# Hardware Discovery
# ============================================================================

class HardwareDiscovery:
    """
    Automatic hardware discovery and enumeration.
    
    Scans for connected RF hardware devices and provides
    information about their capabilities.
    """
    
    def __init__(self):
        """Initialize hardware discovery."""
        self._devices: Dict[str, DeviceInfo] = {}
        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"{__name__}.HardwareDiscovery")
    
    def discover_all(self) -> List[DeviceInfo]:
        """
        Discover all connected hardware devices.
        
        Returns:
            List of discovered device information
        """
        with self._lock:
            self._devices.clear()
            
            # Discover different device types
            self._discover_hackrf()
            self._discover_bladerf()
            self._discover_usrp()
            self._discover_rtlsdr()
            self._discover_limesdr()
            self._discover_plutosdr()
            self._discover_airspy()
            
            return list(self._devices.values())
    
    def _discover_hackrf(self) -> None:
        """Discover HackRF devices."""
        try:
            # Check for hackrf_info
            result = subprocess.run(
                ['hackrf_info'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse hackrf_info output
                output = result.stdout
                serial_match = re.search(r'Serial number: (\w+)', output)
                firmware_match = re.search(r'Firmware Version: ([\w.-]+)', output)
                
                if serial_match:
                    serial = serial_match.group(1)
                    firmware = firmware_match.group(1) if firmware_match else ""
                    
                    device = DeviceInfo(
                        device_id=f"hackrf_{serial[:8]}",
                        device_type=DeviceType.SDR,
                        manufacturer="Great Scott Gadgets",
                        model="HackRF One",
                        serial_number=serial,
                        firmware_version=firmware,
                        connection_type=ConnectionType.USB,
                        capabilities=(
                            HardwareCapability.TRANSMIT |
                            HardwareCapability.RECEIVE |
                            HardwareCapability.HALF_DUPLEX |
                            HardwareCapability.WIDEBAND
                        ),
                        frequency_range=(1e6, 6e9),
                        bandwidth_range=(1.75e6, 20e6),
                        sample_rate_range=(2e6, 20e6),
                        gain_range=(0, 62)
                    )
                    self._devices[device.device_id] = device
                    self._logger.info(f"Discovered HackRF: {serial}")
        except FileNotFoundError:
            self._logger.debug("hackrf_info not found")
        except subprocess.TimeoutExpired:
            self._logger.warning("hackrf_info timed out")
        except Exception as e:
            self._logger.error(f"Error discovering HackRF: {e}")
    
    def _discover_bladerf(self) -> None:
        """Discover BladeRF devices."""
        try:
            result = subprocess.run(
                ['bladeRF-cli', '-p'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and 'Serial' in result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'Serial' in line:
                        # Parse device info
                        parts = line.split()
                        serial = parts[-1] if parts else "unknown"
                        
                        # Detect model
                        model = "BladeRF x40"
                        if 'x115' in result.stdout.lower():
                            model = "BladeRF x115"
                        elif 'xa4' in result.stdout.lower():
                            model = "BladeRF xA4"
                        elif 'xa9' in result.stdout.lower():
                            model = "BladeRF xA9"
                        
                        device = DeviceInfo(
                            device_id=f"bladerf_{serial[:8]}",
                            device_type=DeviceType.SDR,
                            manufacturer="Nuand",
                            model=model,
                            serial_number=serial,
                            connection_type=ConnectionType.USB,
                            capabilities=(
                                HardwareCapability.TRANSMIT |
                                HardwareCapability.RECEIVE |
                                HardwareCapability.FULL_DUPLEX |
                                HardwareCapability.WIDEBAND |
                                HardwareCapability.FPGA_ACCELERATION
                            ),
                            frequency_range=(47e6, 6e9),
                            bandwidth_range=(200e3, 56e6),
                            sample_rate_range=(520.834e3, 61.44e6),
                            gain_range=(-15, 60)
                        )
                        self._devices[device.device_id] = device
                        self._logger.info(f"Discovered BladeRF: {serial}")
        except FileNotFoundError:
            self._logger.debug("bladeRF-cli not found")
        except Exception as e:
            self._logger.error(f"Error discovering BladeRF: {e}")
    
    def _discover_usrp(self) -> None:
        """Discover USRP devices."""
        try:
            result = subprocess.run(
                ['uhd_find_devices'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and 'serial' in result.stdout.lower():
                # Parse UHD device info
                serial_match = re.search(r'serial=(\w+)', result.stdout)
                product_match = re.search(r'product=(\w+)', result.stdout)
                
                if serial_match:
                    serial = serial_match.group(1)
                    product = product_match.group(1) if product_match else "USRP"
                    
                    # Determine capabilities based on product
                    caps = (
                        HardwareCapability.TRANSMIT |
                        HardwareCapability.RECEIVE |
                        HardwareCapability.FULL_DUPLEX
                    )
                    
                    if product in ['X300', 'X310']:
                        caps |= HardwareCapability.WIDEBAND | HardwareCapability.FPGA_ACCELERATION
                        freq_range = (10e6, 6e9)
                    elif product in ['B200', 'B210']:
                        caps |= HardwareCapability.WIDEBAND
                        freq_range = (70e6, 6e9)
                    else:
                        freq_range = (50e6, 2.2e9)
                    
                    device = DeviceInfo(
                        device_id=f"usrp_{serial[:8]}",
                        device_type=DeviceType.SDR,
                        manufacturer="Ettus Research",
                        model=product,
                        serial_number=serial,
                        connection_type=ConnectionType.USB,
                        capabilities=caps,
                        frequency_range=freq_range,
                        bandwidth_range=(200e3, 56e6),
                        sample_rate_range=(200e3, 61.44e6),
                        gain_range=(0, 76)
                    )
                    self._devices[device.device_id] = device
                    self._logger.info(f"Discovered USRP: {product} ({serial})")
        except FileNotFoundError:
            self._logger.debug("uhd_find_devices not found")
        except Exception as e:
            self._logger.error(f"Error discovering USRP: {e}")
    
    def _discover_rtlsdr(self) -> None:
        """Discover RTL-SDR devices."""
        try:
            result = subprocess.run(
                ['rtl_test', '-t'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            output = result.stdout + result.stderr
            
            if 'Found' in output and 'device' in output:
                # Parse RTL-SDR info
                serial_match = re.search(r'SN: (\w+)', output)
                tuner_match = re.search(r'(\w+) tuner', output)
                
                serial = serial_match.group(1) if serial_match else f"rtl_{secrets.token_hex(4)}"
                tuner = tuner_match.group(1) if tuner_match else "R820T"
                
                device = DeviceInfo(
                    device_id=f"rtlsdr_{serial[:8]}",
                    device_type=DeviceType.SDR,
                    manufacturer="Various",
                    model=f"RTL-SDR ({tuner})",
                    serial_number=serial,
                    connection_type=ConnectionType.USB,
                    capabilities=HardwareCapability.RECEIVE | HardwareCapability.WIDEBAND,
                    frequency_range=(24e6, 1.766e9),
                    bandwidth_range=(0.5e6, 3.2e6),
                    sample_rate_range=(225.001e3, 3.2e6),
                    gain_range=(0, 49.6)
                )
                self._devices[device.device_id] = device
                self._logger.info(f"Discovered RTL-SDR: {tuner}")
        except FileNotFoundError:
            self._logger.debug("rtl_test not found")
        except Exception as e:
            self._logger.error(f"Error discovering RTL-SDR: {e}")
    
    def _discover_limesdr(self) -> None:
        """Discover LimeSDR devices."""
        try:
            result = subprocess.run(
                ['LimeUtil', '--find'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and 'LimeSDR' in result.stdout:
                serial_match = re.search(r'serial=(\w+)', result.stdout)
                serial = serial_match.group(1) if serial_match else f"lime_{secrets.token_hex(4)}"
                
                is_mini = 'Mini' in result.stdout
                model = "LimeSDR Mini" if is_mini else "LimeSDR USB"
                
                device = DeviceInfo(
                    device_id=f"limesdr_{serial[:8]}",
                    device_type=DeviceType.SDR,
                    manufacturer="Lime Microsystems",
                    model=model,
                    serial_number=serial,
                    connection_type=ConnectionType.USB,
                    capabilities=(
                        HardwareCapability.TRANSMIT |
                        HardwareCapability.RECEIVE |
                        HardwareCapability.FULL_DUPLEX |
                        HardwareCapability.WIDEBAND |
                        HardwareCapability.FPGA_ACCELERATION
                    ),
                    frequency_range=(100e3, 3.8e9),
                    bandwidth_range=(1.5e6, 60e6),
                    sample_rate_range=(100e3, 61.44e6),
                    gain_range=(0, 73)
                )
                self._devices[device.device_id] = device
                self._logger.info(f"Discovered LimeSDR: {model}")
        except FileNotFoundError:
            self._logger.debug("LimeUtil not found")
        except Exception as e:
            self._logger.error(f"Error discovering LimeSDR: {e}")
    
    def _discover_plutosdr(self) -> None:
        """Discover PlutoSDR devices."""
        try:
            result = subprocess.run(
                ['iio_info', '-s'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and 'pluto' in result.stdout.lower():
                serial_match = re.search(r'serial=(\w+)', result.stdout)
                serial = serial_match.group(1) if serial_match else f"pluto_{secrets.token_hex(4)}"
                
                device = DeviceInfo(
                    device_id=f"plutosdr_{serial[:8]}",
                    device_type=DeviceType.SDR,
                    manufacturer="Analog Devices",
                    model="ADALM-PLUTO",
                    serial_number=serial,
                    connection_type=ConnectionType.USB,
                    capabilities=(
                        HardwareCapability.TRANSMIT |
                        HardwareCapability.RECEIVE |
                        HardwareCapability.FULL_DUPLEX
                    ),
                    frequency_range=(325e6, 3.8e9),
                    bandwidth_range=(200e3, 20e6),
                    sample_rate_range=(520.834e3, 61.44e6),
                    gain_range=(-1, 73)
                )
                self._devices[device.device_id] = device
                self._logger.info(f"Discovered PlutoSDR: {serial}")
        except FileNotFoundError:
            self._logger.debug("iio_info not found")
        except Exception as e:
            self._logger.error(f"Error discovering PlutoSDR: {e}")
    
    def _discover_airspy(self) -> None:
        """Discover Airspy devices."""
        try:
            result = subprocess.run(
                ['airspy_info'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                serial_match = re.search(r'Serial Number: (\w+)', result.stdout)
                serial = serial_match.group(1) if serial_match else f"airspy_{secrets.token_hex(4)}"
                
                is_mini = 'Mini' in result.stdout
                model = "Airspy Mini" if is_mini else "Airspy R2"
                
                device = DeviceInfo(
                    device_id=f"airspy_{serial[:8]}",
                    device_type=DeviceType.SDR,
                    manufacturer="Airspy",
                    model=model,
                    serial_number=serial,
                    connection_type=ConnectionType.USB,
                    capabilities=HardwareCapability.RECEIVE | HardwareCapability.WIDEBAND,
                    frequency_range=(24e6, 1.8e9),
                    bandwidth_range=(0.5e6, 6e6),
                    sample_rate_range=(2.5e6, 10e6),
                    gain_range=(0, 45)
                )
                self._devices[device.device_id] = device
                self._logger.info(f"Discovered Airspy: {model}")
        except FileNotFoundError:
            self._logger.debug("airspy_info not found")
        except Exception as e:
            self._logger.error(f"Error discovering Airspy: {e}")
    
    def get_device(self, device_id: str) -> Optional[DeviceInfo]:
        """Get device by ID."""
        return self._devices.get(device_id)
    
    def get_devices_by_type(self, device_type: DeviceType) -> List[DeviceInfo]:
        """Get devices by type."""
        return [d for d in self._devices.values() if d.device_type == device_type]
    
    def get_devices_by_capability(self, capability: HardwareCapability) -> List[DeviceInfo]:
        """Get devices with specified capability."""
        return [d for d in self._devices.values() if capability in d.capabilities]


# ============================================================================
# Test Infrastructure
# ============================================================================

class TestCase:
    """
    Base class for hardware test cases.
    
    Provides structure for test definition, setup, execution, and teardown.
    """
    
    def __init__(
        self,
        name: str,
        category: TestCategory = TestCategory.FUNCTIONAL,
        priority: TestPriority = TestPriority.MEDIUM,
        timeout_seconds: float = 60.0,
        required_capabilities: HardwareCapability = HardwareCapability.NONE,
        tags: Optional[List[str]] = None,
        description: str = ""
    ):
        """Initialize test case."""
        self.name = name
        self.category = category
        self.priority = priority
        self.timeout_seconds = timeout_seconds
        self.required_capabilities = required_capabilities
        self.tags = tags or []
        self.description = description
        
        self.test_id = f"{self.__class__.__name__}_{name}_{secrets.token_hex(4)}"
        self._device: Optional[HardwareInterface] = None
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def set_device(self, device: HardwareInterface) -> None:
        """Set the hardware device for this test."""
        self._device = device
    
    def setup(self) -> None:
        """
        Set up test prerequisites.
        
        Override this method to perform test-specific setup.
        Called before each test execution.
        """
        pass
    
    def teardown(self) -> None:
        """
        Clean up after test execution.
        
        Override this method to perform test-specific cleanup.
        Called after each test execution regardless of result.
        """
        pass
    
    def run(self) -> TestResult:
        """
        Execute the test.
        
        Override this method to implement test logic.
        
        Returns:
            TestResult with test outcome
        """
        raise NotImplementedError("Subclasses must implement run()")
    
    def skip_if(self, condition: bool, reason: str = "") -> None:
        """
        Skip test if condition is true.
        
        Args:
            condition: Condition to check
            reason: Reason for skipping
        
        Raises:
            SkipTestException if condition is true
        """
        if condition:
            raise SkipTestException(reason)
    
    def require_capability(self, capability: HardwareCapability) -> None:
        """
        Require hardware capability for this test.
        
        Args:
            capability: Required capability
        
        Raises:
            SkipTestException if capability not available
        """
        if self._device and self._device.device_info:
            if capability not in self._device.device_info.capabilities:
                raise SkipTestException(
                    f"Device lacks required capability: {capability.name}"
                )


class SkipTestException(Exception):
    """Exception raised to skip a test."""
    pass


class TestFixture:
    """
    Test fixture for shared setup and teardown.
    
    Provides common setup and teardown logic that can be
    shared across multiple test cases.
    """
    
    def __init__(self, name: str = ""):
        """Initialize fixture."""
        self.name = name or self.__class__.__name__
        self._setup_done = False
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def setup_fixture(self) -> None:
        """
        One-time fixture setup.
        
        Called once before any tests using this fixture.
        """
        if not self._setup_done:
            self.setup()
            self._setup_done = True
    
    def teardown_fixture(self) -> None:
        """
        One-time fixture teardown.
        
        Called once after all tests using this fixture.
        """
        if self._setup_done:
            self.teardown()
            self._setup_done = False
    
    def setup(self) -> None:
        """Override to implement fixture setup."""
        pass
    
    def teardown(self) -> None:
        """Override to implement fixture teardown."""
        pass


class TestSuite:
    """
    Collection of related test cases.
    
    Organizes tests into logical groups with shared fixtures
    and configuration.
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        fixtures: Optional[List[TestFixture]] = None
    ):
        """Initialize test suite."""
        self.name = name
        self.description = description
        self.fixtures = fixtures or []
        self._tests: List[TestCase] = []
        self._logger = logging.getLogger(f"{__name__}.TestSuite.{name}")
    
    def add_test(self, test: TestCase) -> None:
        """Add test case to suite."""
        self._tests.append(test)
    
    def add_tests(self, tests: List[TestCase]) -> None:
        """Add multiple test cases."""
        self._tests.extend(tests)
    
    def get_tests(
        self,
        category: Optional[TestCategory] = None,
        priority: Optional[TestPriority] = None,
        tags: Optional[List[str]] = None
    ) -> List[TestCase]:
        """
        Get tests matching criteria.
        
        Args:
            category: Filter by category
            priority: Filter by priority
            tags: Filter by tags
        
        Returns:
            List of matching test cases
        """
        tests = self._tests.copy()
        
        if category:
            tests = [t for t in tests if t.category == category]
        
        if priority:
            tests = [t for t in tests if t.priority == priority]
        
        if tags:
            tests = [t for t in tests if any(tag in t.tags for tag in tags)]
        
        return tests
    
    @property
    def test_count(self) -> int:
        """Get number of tests in suite."""
        return len(self._tests)


class TestRunner:
    """
    Executes hardware test suites and manages test lifecycle.
    
    Features:
    - Test discovery and selection
    - Parallel execution support
    - Timeout handling
    - Retry logic
    - Result collection
    - Progress reporting
    """
    
    def __init__(
        self,
        config: Optional[HardwareTestConfig] = None,
        discovery: Optional[HardwareDiscovery] = None
    ):
        """Initialize test runner."""
        self.config = config or HardwareTestConfig()
        self.discovery = discovery or HardwareDiscovery()
        self.environment = TestEnvironment()
        
        self._results: List[TestResult] = []
        self._suites: List[TestSuite] = []
        self._lock = threading.RLock()
        self._stop_requested = False
        
        self._logger = logging.getLogger(f"{__name__}.TestRunner")
    
    def register_suite(self, suite: TestSuite) -> None:
        """Register a test suite."""
        self._suites.append(suite)
        self._logger.info(f"Registered suite: {suite.name} ({suite.test_count} tests)")
    
    def discover_hardware(self) -> List[DeviceInfo]:
        """
        Discover connected hardware.
        
        Returns:
            List of discovered devices
        """
        devices = self.discovery.discover_all()
        self.environment.connected_devices = devices
        
        self._logger.info(f"Discovered {len(devices)} devices")
        for device in devices:
            self._logger.info(f"  - {device.manufacturer} {device.model} ({device.serial_number})")
        
        return devices
    
    def run_all(self) -> List[TestResult]:
        """
        Run all registered tests.
        
        Returns:
            List of test results
        """
        self._results.clear()
        self._stop_requested = False
        self.environment.start_time = datetime.now()
        
        # Discover hardware
        self.discover_hardware()
        
        # Collect all tests
        all_tests = []
        for suite in self._suites:
            tests = suite.get_tests()
            all_tests.extend([(suite, test) for test in tests])
        
        # Filter tests based on config
        filtered_tests = self._filter_tests(all_tests)
        
        self._logger.info(f"Running {len(filtered_tests)} tests")
        
        # Execute tests
        if self.config.parallel_execution:
            self._run_parallel(filtered_tests)
        else:
            self._run_sequential(filtered_tests)
        
        self.environment.end_time = datetime.now()
        
        return self._results
    
    def _filter_tests(
        self,
        tests: List[Tuple[TestSuite, TestCase]]
    ) -> List[Tuple[TestSuite, TestCase]]:
        """Filter tests based on configuration."""
        filtered = []
        
        for suite, test in tests:
            # Filter by category
            if test.category not in self.config.categories:
                continue
            
            # Filter by priority
            if test.priority not in self.config.priorities:
                continue
            
            # Filter by include patterns
            if self.config.include_patterns:
                if not any(re.search(p, test.name) for p in self.config.include_patterns):
                    continue
            
            # Filter by exclude patterns
            if self.config.exclude_patterns:
                if any(re.search(p, test.name) for p in self.config.exclude_patterns):
                    continue
            
            filtered.append((suite, test))
        
        return filtered
    
    def _run_sequential(
        self,
        tests: List[Tuple[TestSuite, TestCase]]
    ) -> None:
        """Run tests sequentially."""
        for suite, test in tests:
            if self._stop_requested:
                break
            
            result = self._execute_test(suite, test)
            self._results.append(result)
            
            if self.config.fail_fast and result.failed:
                self._logger.warning("Fail fast triggered, stopping tests")
                break
    
    def _run_parallel(
        self,
        tests: List[Tuple[TestSuite, TestCase]]
    ) -> None:
        """Run tests in parallel."""
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_tests) as executor:
            futures = {
                executor.submit(self._execute_test, suite, test): (suite, test)
                for suite, test in tests
            }
            
            for future in futures:
                if self._stop_requested:
                    break
                
                try:
                    result = future.result(timeout=self.config.timeout_seconds * 2)
                    with self._lock:
                        self._results.append(result)
                    
                    if self.config.fail_fast and result.failed:
                        self._stop_requested = True
                except FuturesTimeoutError:
                    suite, test = futures[future]
                    result = TestResult(
                        test_id=test.test_id,
                        test_name=test.name,
                        status=TestStatus.TIMEOUT,
                        category=test.category,
                        priority=test.priority,
                        error_message="Test execution timed out"
                    )
                    with self._lock:
                        self._results.append(result)
    
    def _execute_test(
        self,
        suite: TestSuite,
        test: TestCase
    ) -> TestResult:
        """
        Execute a single test with full lifecycle management.
        
        Args:
            suite: Test suite containing the test
            test: Test case to execute
        
        Returns:
            Test result
        """
        start_time = datetime.now()
        
        result = TestResult(
            test_id=test.test_id,
            test_name=test.name,
            status=TestStatus.RUNNING,
            category=test.category,
            priority=test.priority,
            start_time=start_time
        )
        
        self._logger.info(f"Running test: {test.name}")
        
        try:
            # Setup fixtures
            for fixture in suite.fixtures:
                fixture.setup_fixture()
            
            # Setup test
            test.setup()
            
            # Run test with timeout
            test_result = self._run_with_timeout(test.run, test.timeout_seconds)
            
            # Update result from test execution
            if test_result:
                result.status = test_result.status
                result.message = test_result.message
                result.details = test_result.details
                result.measurements = test_result.measurements
            else:
                result.status = TestStatus.PASSED
                result.message = "Test completed successfully"
            
        except SkipTestException as e:
            result.status = TestStatus.SKIPPED
            result.message = str(e) or "Test skipped"
            self._logger.info(f"Test skipped: {test.name} - {result.message}")
            
        except AssertionError as e:
            result.status = TestStatus.FAILED
            result.error_type = "AssertionError"
            result.error_message = str(e)
            result.stack_trace = traceback.format_exc()
            self._logger.warning(f"Test failed: {test.name} - {e}")
            
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_type = type(e).__name__
            result.error_message = str(e)
            result.stack_trace = traceback.format_exc()
            self._logger.error(f"Test error: {test.name} - {e}")
            
        finally:
            # Teardown test
            try:
                test.teardown()
            except Exception as e:
                self._logger.error(f"Teardown error: {e}")
            
            # Teardown fixtures
            for fixture in suite.fixtures:
                try:
                    fixture.teardown_fixture()
                except Exception as e:
                    self._logger.error(f"Fixture teardown error: {e}")
        
        # Finalize result
        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        
        status_str = result.status.value.upper()
        self._logger.info(
            f"Test {status_str}: {test.name} ({result.duration_seconds:.2f}s)"
        )
        
        return result
    
    def _run_with_timeout(
        self,
        func: Callable[[], TestResult],
        timeout: float
    ) -> Optional[TestResult]:
        """
        Run function with timeout.
        
        Args:
            func: Function to run
            timeout: Timeout in seconds
        
        Returns:
            Function result or None if timeout
        """
        result = [None]
        exception = [None]
        
        def wrapper():
            try:
                result[0] = func()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=wrapper)
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            raise TimeoutError(f"Test timed out after {timeout} seconds")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]
    
    def stop(self) -> None:
        """Request test execution to stop."""
        self._stop_requested = True
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get test execution summary.
        
        Returns:
            Summary dictionary
        """
        total = len(self._results)
        passed = sum(1 for r in self._results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self._results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in self._results if r.status == TestStatus.ERROR)
        skipped = sum(1 for r in self._results if r.status == TestStatus.SKIPPED)
        timeouts = sum(1 for r in self._results if r.status == TestStatus.TIMEOUT)
        
        total_duration = sum(r.duration_seconds for r in self._results)
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'skipped': skipped,
            'timeouts': timeouts,
            'pass_rate': (passed / total * 100) if total > 0 else 0,
            'total_duration_seconds': total_duration,
            'environment': self.environment.to_dict(),
            'results': [r.to_dict() for r in self._results]
        }


# ============================================================================
# Hardware Test Framework
# ============================================================================

class HardwareTestFramework:
    """
    Main framework for comprehensive hardware integration testing.
    
    Provides a unified interface for:
    - Hardware discovery and management
    - Test suite registration and execution
    - Result collection and reporting
    - CI/CD integration
    
    Example usage:
        framework = HardwareTestFramework()
        framework.discover_hardware()
        framework.register_suite(MySDRTestSuite())
        results = framework.run_tests()
        framework.generate_report("test_report.html")
    """
    
    def __init__(self, config: Optional[HardwareTestConfig] = None):
        """Initialize the framework."""
        self.config = config or HardwareTestConfig()
        self.discovery = HardwareDiscovery()
        self.runner = TestRunner(self.config, self.discovery)
        
        self._devices: Dict[str, DeviceInfo] = {}
        self._logger = logging.getLogger(f"{__name__}.HardwareTestFramework")
        
        # Configure logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        root_logger = logging.getLogger(__name__.split('.')[0])
        root_logger.setLevel(level)
        root_logger.addHandler(handler)
    
    def discover_hardware(self) -> List[DeviceInfo]:
        """
        Discover all connected hardware devices.
        
        Returns:
            List of discovered device information
        """
        devices = self.discovery.discover_all()
        self._devices = {d.device_id: d for d in devices}
        
        self._logger.info(f"Discovered {len(devices)} hardware devices")
        return devices
    
    def get_device(self, device_id: str) -> Optional[DeviceInfo]:
        """Get device by ID."""
        return self._devices.get(device_id)
    
    def get_available_devices(self) -> List[DeviceInfo]:
        """Get all available devices."""
        return [d for d in self._devices.values() if d.is_available]
    
    def register_suite(self, suite: TestSuite) -> None:
        """
        Register a test suite.
        
        Args:
            suite: Test suite to register
        """
        self.runner.register_suite(suite)
        self._logger.info(f"Registered test suite: {suite.name}")
    
    def run_tests(
        self,
        categories: Optional[List[TestCategory]] = None,
        device_ids: Optional[List[str]] = None
    ) -> List[TestResult]:
        """
        Execute registered tests.
        
        Args:
            categories: Categories to run (all if None)
            device_ids: Specific devices to test (all if None)
        
        Returns:
            List of test results
        """
        # Update config if parameters provided
        if categories:
            self.config.categories = categories
        if device_ids:
            self.config.device_ids = device_ids
        
        self._logger.info("Starting test execution")
        results = self.runner.run_all()
        self._logger.info(f"Completed {len(results)} tests")
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test execution summary."""
        return self.runner.get_summary()
    
    def generate_report(
        self,
        output_path: str,
        format: str = "html"
    ) -> str:
        """
        Generate test report.
        
        Args:
            output_path: Path for output file
            format: Report format (html, json, junit)
        
        Returns:
            Path to generated report
        """
        summary = self.get_summary()
        
        if format.lower() == "json":
            content = json.dumps(summary, indent=2, default=str)
        elif format.lower() == "junit":
            content = self._generate_junit_xml(summary)
        else:
            content = self._generate_html_report(summary)
        
        with open(output_path, 'w') as f:
            f.write(content)
        
        self._logger.info(f"Generated {format} report: {output_path}")
        return output_path
    
    def _generate_html_report(self, summary: Dict[str, Any]) -> str:
        """Generate HTML test report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Hardware Integration Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .passed {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .skipped {{ color: #ffc107; }}
        .error {{ color: #dc3545; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #333; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .status-passed {{ background: #d4edda; }}
        .status-failed {{ background: #f8d7da; }}
        .status-skipped {{ background: #fff3cd; }}
        .status-error {{ background: #f8d7da; }}
    </style>
</head>
<body>
    <h1>Hardware Integration Test Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Tests:</strong> {summary['total']}</p>
        <p class="passed"><strong>Passed:</strong> {summary['passed']}</p>
        <p class="failed"><strong>Failed:</strong> {summary['failed']}</p>
        <p class="error"><strong>Errors:</strong> {summary['errors']}</p>
        <p class="skipped"><strong>Skipped:</strong> {summary['skipped']}</p>
        <p><strong>Pass Rate:</strong> {summary['pass_rate']:.1f}%</p>
        <p><strong>Duration:</strong> {summary['total_duration_seconds']:.2f}s</p>
    </div>
    
    <h2>Test Results</h2>
    <table>
        <tr>
            <th>Test Name</th>
            <th>Status</th>
            <th>Category</th>
            <th>Duration</th>
            <th>Message</th>
        </tr>
"""
        
        for result in summary['results']:
            status = result['status']
            status_class = f"status-{status}"
            
            html += f"""        <tr class="{status_class}">
            <td>{result['test_name']}</td>
            <td>{status.upper()}</td>
            <td>{result['category']}</td>
            <td>{result['duration_seconds']:.2f}s</td>
            <td>{result.get('message', '') or result.get('error_message', '')}</td>
        </tr>
"""
        
        html += """    </table>
    
    <h2>Environment</h2>
    <table>
        <tr><th>Property</th><th>Value</th></tr>
"""
        
        env = summary['environment']
        html += f"""        <tr><td>Hostname</td><td>{env['hostname']}</td></tr>
        <tr><td>Platform</td><td>{env['platform']} {env['platform_version']}</td></tr>
        <tr><td>Architecture</td><td>{env['architecture']}</td></tr>
        <tr><td>Python Version</td><td>{env['python_version']}</td></tr>
        <tr><td>CPU Count</td><td>{env['cpu_count']}</td></tr>
        <tr><td>Run ID</td><td>{env['run_id']}</td></tr>
        <tr><td>Start Time</td><td>{env['start_time']}</td></tr>
        <tr><td>End Time</td><td>{env['end_time']}</td></tr>
    </table>
    
    <h2>Connected Devices</h2>
    <table>
        <tr><th>Device</th><th>Manufacturer</th><th>Model</th><th>Serial</th></tr>
"""
        
        for device in env['connected_devices']:
            html += f"""        <tr>
            <td>{device['device_id']}</td>
            <td>{device['manufacturer']}</td>
            <td>{device['model']}</td>
            <td>{device['serial_number']}</td>
        </tr>
"""
        
        html += """    </table>
</body>
</html>"""
        
        return html
    
    def _generate_junit_xml(self, summary: Dict[str, Any]) -> str:
        """Generate JUnit XML report."""
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="Hardware Integration Tests" tests="{summary['total']}" failures="{summary['failed']}" errors="{summary['errors']}" skipped="{summary['skipped']}" time="{summary['total_duration_seconds']}">
  <testsuite name="HardwareTests" tests="{summary['total']}" failures="{summary['failed']}" errors="{summary['errors']}" skipped="{summary['skipped']}" time="{summary['total_duration_seconds']}">
"""
        
        for result in summary['results']:
            xml += f"""    <testcase name="{result['test_name']}" classname="{result['category']}" time="{result['duration_seconds']}">
"""
            
            if result['status'] == 'failed':
                xml += f"""      <failure message="{result.get('error_message', '')}" type="{result.get('error_type', 'AssertionError')}">
{result.get('stack_trace', '')}
      </failure>
"""
            elif result['status'] == 'error':
                xml += f"""      <error message="{result.get('error_message', '')}" type="{result.get('error_type', 'Exception')}">
{result.get('stack_trace', '')}
      </error>
"""
            elif result['status'] == 'skipped':
                xml += f"""      <skipped message="{result.get('message', '')}"/>
"""
            
            xml += """    </testcase>
"""
        
        xml += """  </testsuite>
</testsuites>"""
        
        return xml


# ============================================================================
# Decorators
# ============================================================================

def hardware_test(
    category: TestCategory = TestCategory.FUNCTIONAL,
    priority: TestPriority = TestPriority.MEDIUM,
    timeout: float = 60.0,
    requires: HardwareCapability = HardwareCapability.NONE,
    tags: Optional[List[str]] = None
):
    """
    Decorator for marking methods as hardware tests.
    
    Args:
        category: Test category
        priority: Test priority
        timeout: Test timeout in seconds
        requires: Required hardware capabilities
        tags: Test tags for filtering
    
    Example:
        @hardware_test(category=TestCategory.SMOKE, priority=TestPriority.CRITICAL)
        def test_device_connection(self):
            # Test implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        wrapper._is_hardware_test = True
        wrapper._test_category = category
        wrapper._test_priority = priority
        wrapper._test_timeout = timeout
        wrapper._test_requires = requires
        wrapper._test_tags = tags or []
        
        return wrapper
    return decorator


def skip_if_no_hardware(device_type: Optional[DeviceType] = None):
    """
    Decorator to skip test if hardware is not available.
    
    Args:
        device_type: Specific device type required
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if device_type:
                # Check for specific device type
                discovery = HardwareDiscovery()
                devices = discovery.get_devices_by_type(device_type)
                if not devices:
                    raise SkipTestException(
                        f"No {device_type.value} devices available"
                    )
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def retry_on_failure(retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry test on failure.
    
    Args:
        retries: Number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except (AssertionError, Exception) as e:
                    last_exception = e
                    if attempt < retries:
                        time.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Enumerations
    'TestCategory',
    'TestPriority',
    'TestStatus',
    'HardwareCapability',
    'DeviceType',
    'ConnectionType',
    
    # Data classes
    'DeviceInfo',
    'DeviceCapabilities',
    'TestResult',
    'HardwareTestConfig',
    'TestEnvironment',
    
    # Interfaces
    'HardwareInterface',
    'SDRInterface',
    'SignalGeneratorInterface',
    'SpectrumAnalyzerInterface',
    
    # Discovery
    'HardwareDiscovery',
    
    # Test infrastructure
    'TestCase',
    'TestFixture',
    'TestSuite',
    'TestRunner',
    'SkipTestException',
    
    # Framework
    'HardwareTestFramework',
    
    # Decorators
    'hardware_test',
    'skip_if_no_hardware',
    'retry_on_failure',
]
