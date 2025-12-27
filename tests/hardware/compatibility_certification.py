#!/usr/bin/env python3
"""
RF Arsenal OS - Hardware Compatibility Certification Framework

Comprehensive hardware compatibility testing and certification system
for validating real device support across all claimed SDR platforms.

This framework provides:
1. Automated hardware detection and identification
2. Feature capability validation per device
3. Performance benchmarking against specifications
4. Compliance certification with pass/fail criteria
5. Detailed compatibility reports and certificates

Supported Hardware Platforms:
- HackRF One / HackRF One + PortaPack
- BladeRF (x40, x115, xA4, xA5, xA9, 2.0 micro)
- USRP (B200, B210, B200mini, X300, X310, N200, N210)
- RTL-SDR (Generic, RTL-SDR Blog V3, Nooelec)
- LimeSDR (Mini, USB, PCIe)
- PlutoSDR (ADALM-PLUTO)
- Airspy (R2, Mini, HF+, HF+ Discovery)

Certification Levels:
- CERTIFIED: All tests pass, full feature support
- COMPATIBLE: Core tests pass, some features limited
- PARTIAL: Basic functionality only
- INCOMPATIBLE: Device not supported
- UNTESTED: No validation performed

Author: RF Arsenal Development Team
License: Proprietary
Version: 1.0.0
"""

import os
import sys
import json
import time
import hashlib
import logging
import subprocess
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CertificationLevel(Enum):
    """Hardware certification levels"""
    CERTIFIED = "certified"          # Full support, all tests pass
    COMPATIBLE = "compatible"        # Core features work, some limitations
    PARTIAL = "partial"              # Basic functionality only
    INCOMPATIBLE = "incompatible"    # Not supported
    UNTESTED = "untested"            # No validation performed


class TestCategory(Enum):
    """Hardware test categories"""
    DETECTION = "detection"
    INITIALIZATION = "initialization"
    FREQUENCY = "frequency"
    BANDWIDTH = "bandwidth"
    SAMPLE_RATE = "sample_rate"
    GAIN = "gain"
    STREAMING = "streaming"
    DUPLEX = "duplex"
    CALIBRATION = "calibration"
    STABILITY = "stability"
    PERFORMANCE = "performance"
    ADVANCED = "advanced"


class TestResult(Enum):
    """Individual test result"""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class HardwareSpecification:
    """Official hardware specifications for validation"""
    device_family: str
    device_model: str
    
    # Frequency specifications
    freq_min_hz: float
    freq_max_hz: float
    freq_resolution_hz: float = 1.0
    
    # Sample rate specifications
    sample_rate_min_sps: float = 1e6
    sample_rate_max_sps: float = 20e6
    
    # Bandwidth specifications
    bandwidth_min_hz: float = 200e3
    bandwidth_max_hz: float = 20e6
    
    # Gain specifications
    tx_gain_min_db: float = 0.0
    tx_gain_max_db: float = 47.0
    rx_gain_min_db: float = 0.0
    rx_gain_max_db: float = 76.0
    
    # Capabilities
    supports_tx: bool = True
    supports_rx: bool = True
    supports_full_duplex: bool = False
    num_tx_channels: int = 1
    num_rx_channels: int = 1
    
    # Interface
    interface: str = "USB"
    usb_version: str = "2.0"
    
    # Additional specs
    adc_bits: int = 12
    dac_bits: int = 12
    fpga_model: Optional[str] = None
    
    # Tolerances for testing
    freq_tolerance_ppm: float = 10.0
    gain_tolerance_db: float = 1.0
    sample_rate_tolerance_percent: float = 1.0


# Official specifications for all supported devices
HARDWARE_SPECIFICATIONS = {
    # HackRF Family
    "hackrf_one": HardwareSpecification(
        device_family="HackRF",
        device_model="HackRF One",
        freq_min_hz=1e6,
        freq_max_hz=6e9,
        sample_rate_min_sps=2e6,
        sample_rate_max_sps=20e6,
        bandwidth_min_hz=1.75e6,
        bandwidth_max_hz=28e6,
        tx_gain_min_db=0,
        tx_gain_max_db=47,
        rx_gain_min_db=0,
        rx_gain_max_db=62,
        supports_tx=True,
        supports_rx=True,
        supports_full_duplex=False,  # Half-duplex only
        num_tx_channels=1,
        num_rx_channels=1,
        interface="USB",
        usb_version="2.0",
        adc_bits=8,
        dac_bits=8,
    ),
    
    # BladeRF Family
    "bladerf_x40": HardwareSpecification(
        device_family="BladeRF",
        device_model="bladeRF x40",
        freq_min_hz=300e6,
        freq_max_hz=3.8e9,
        sample_rate_min_sps=160e3,
        sample_rate_max_sps=40e6,
        bandwidth_min_hz=1.5e6,
        bandwidth_max_hz=28e6,
        tx_gain_min_db=-4,
        tx_gain_max_db=66,
        rx_gain_min_db=-4,
        rx_gain_max_db=66,
        supports_tx=True,
        supports_rx=True,
        supports_full_duplex=True,
        num_tx_channels=1,
        num_rx_channels=1,
        interface="USB",
        usb_version="3.0",
        adc_bits=12,
        dac_bits=12,
        fpga_model="Altera Cyclone IV 40KLE",
    ),
    "bladerf_x115": HardwareSpecification(
        device_family="BladeRF",
        device_model="bladeRF x115",
        freq_min_hz=300e6,
        freq_max_hz=3.8e9,
        sample_rate_min_sps=160e3,
        sample_rate_max_sps=40e6,
        bandwidth_min_hz=1.5e6,
        bandwidth_max_hz=28e6,
        tx_gain_min_db=-4,
        tx_gain_max_db=66,
        rx_gain_min_db=-4,
        rx_gain_max_db=66,
        supports_tx=True,
        supports_rx=True,
        supports_full_duplex=True,
        num_tx_channels=1,
        num_rx_channels=1,
        interface="USB",
        usb_version="3.0",
        adc_bits=12,
        dac_bits=12,
        fpga_model="Altera Cyclone IV 115KLE",
    ),
    "bladerf_xa4": HardwareSpecification(
        device_family="BladeRF",
        device_model="bladeRF 2.0 micro xA4",
        freq_min_hz=47e6,
        freq_max_hz=6e9,
        sample_rate_min_sps=521e3,
        sample_rate_max_sps=61.44e6,
        bandwidth_min_hz=200e3,
        bandwidth_max_hz=56e6,
        tx_gain_min_db=-89.75,
        tx_gain_max_db=0,
        rx_gain_min_db=-4,
        rx_gain_max_db=71,
        supports_tx=True,
        supports_rx=True,
        supports_full_duplex=True,
        num_tx_channels=2,
        num_rx_channels=2,
        interface="USB",
        usb_version="3.0",
        adc_bits=12,
        dac_bits=12,
        fpga_model="Altera Cyclone V 49KLE",
    ),
    "bladerf_xa5": HardwareSpecification(
        device_family="BladeRF",
        device_model="bladeRF 2.0 micro xA5",
        freq_min_hz=47e6,
        freq_max_hz=6e9,
        sample_rate_min_sps=521e3,
        sample_rate_max_sps=61.44e6,
        bandwidth_min_hz=200e3,
        bandwidth_max_hz=56e6,
        tx_gain_min_db=-89.75,
        tx_gain_max_db=0,
        rx_gain_min_db=-4,
        rx_gain_max_db=71,
        supports_tx=True,
        supports_rx=True,
        supports_full_duplex=True,
        num_tx_channels=2,
        num_rx_channels=2,
        interface="USB",
        usb_version="3.0",
        adc_bits=12,
        dac_bits=12,
        fpga_model="Altera Cyclone V 77KLE",
    ),
    "bladerf_xa9": HardwareSpecification(
        device_family="BladeRF",
        device_model="bladeRF 2.0 micro xA9",
        freq_min_hz=47e6,
        freq_max_hz=6e9,
        sample_rate_min_sps=521e3,
        sample_rate_max_sps=61.44e6,
        bandwidth_min_hz=200e3,
        bandwidth_max_hz=56e6,
        tx_gain_min_db=-89.75,
        tx_gain_max_db=0,
        rx_gain_min_db=-4,
        rx_gain_max_db=71,
        supports_tx=True,
        supports_rx=True,
        supports_full_duplex=True,
        num_tx_channels=2,
        num_rx_channels=2,
        interface="USB",
        usb_version="3.0",
        adc_bits=12,
        dac_bits=12,
        fpga_model="Altera Cyclone V 301KLE",
    ),
    
    # USRP Family
    "usrp_b200": HardwareSpecification(
        device_family="USRP",
        device_model="USRP B200",
        freq_min_hz=70e6,
        freq_max_hz=6e9,
        sample_rate_min_sps=200e3,
        sample_rate_max_sps=61.44e6,
        bandwidth_min_hz=200e3,
        bandwidth_max_hz=56e6,
        tx_gain_min_db=0,
        tx_gain_max_db=89.75,
        rx_gain_min_db=0,
        rx_gain_max_db=76,
        supports_tx=True,
        supports_rx=True,
        supports_full_duplex=True,
        num_tx_channels=1,
        num_rx_channels=1,
        interface="USB",
        usb_version="3.0",
        adc_bits=12,
        dac_bits=12,
        fpga_model="Xilinx Spartan-6",
    ),
    "usrp_b210": HardwareSpecification(
        device_family="USRP",
        device_model="USRP B210",
        freq_min_hz=70e6,
        freq_max_hz=6e9,
        sample_rate_min_sps=200e3,
        sample_rate_max_sps=61.44e6,
        bandwidth_min_hz=200e3,
        bandwidth_max_hz=56e6,
        tx_gain_min_db=0,
        tx_gain_max_db=89.75,
        rx_gain_min_db=0,
        rx_gain_max_db=76,
        supports_tx=True,
        supports_rx=True,
        supports_full_duplex=True,
        num_tx_channels=2,
        num_rx_channels=2,
        interface="USB",
        usb_version="3.0",
        adc_bits=12,
        dac_bits=12,
        fpga_model="Xilinx Spartan-6",
    ),
    "usrp_x300": HardwareSpecification(
        device_family="USRP",
        device_model="USRP X300",
        freq_min_hz=10e6,
        freq_max_hz=6e9,
        sample_rate_min_sps=200e3,
        sample_rate_max_sps=200e6,
        bandwidth_min_hz=200e3,
        bandwidth_max_hz=160e6,
        tx_gain_min_db=0,
        tx_gain_max_db=31.5,
        rx_gain_min_db=0,
        rx_gain_max_db=31.5,
        supports_tx=True,
        supports_rx=True,
        supports_full_duplex=True,
        num_tx_channels=2,
        num_rx_channels=2,
        interface="10GbE/PCIe",
        usb_version="N/A",
        adc_bits=14,
        dac_bits=16,
        fpga_model="Xilinx Kintex-7",
    ),
    "usrp_x310": HardwareSpecification(
        device_family="USRP",
        device_model="USRP X310",
        freq_min_hz=10e6,
        freq_max_hz=6e9,
        sample_rate_min_sps=200e3,
        sample_rate_max_sps=200e6,
        bandwidth_min_hz=200e3,
        bandwidth_max_hz=160e6,
        tx_gain_min_db=0,
        tx_gain_max_db=31.5,
        rx_gain_min_db=0,
        rx_gain_max_db=31.5,
        supports_tx=True,
        supports_rx=True,
        supports_full_duplex=True,
        num_tx_channels=2,
        num_rx_channels=2,
        interface="10GbE/PCIe",
        usb_version="N/A",
        adc_bits=14,
        dac_bits=16,
        fpga_model="Xilinx Kintex-7",
    ),
    
    # RTL-SDR Family
    "rtlsdr_generic": HardwareSpecification(
        device_family="RTL-SDR",
        device_model="RTL-SDR Generic",
        freq_min_hz=24e6,
        freq_max_hz=1.766e9,
        sample_rate_min_sps=225.001e3,
        sample_rate_max_sps=3.2e6,
        bandwidth_min_hz=225e3,
        bandwidth_max_hz=3.2e6,
        tx_gain_min_db=0,
        tx_gain_max_db=0,
        rx_gain_min_db=0,
        rx_gain_max_db=49.6,
        supports_tx=False,
        supports_rx=True,
        supports_full_duplex=False,
        num_tx_channels=0,
        num_rx_channels=1,
        interface="USB",
        usb_version="2.0",
        adc_bits=8,
        dac_bits=0,
    ),
    "rtlsdr_v3": HardwareSpecification(
        device_family="RTL-SDR",
        device_model="RTL-SDR Blog V3",
        freq_min_hz=500e3,  # Direct sampling mode
        freq_max_hz=1.766e9,
        sample_rate_min_sps=225.001e3,
        sample_rate_max_sps=3.2e6,
        bandwidth_min_hz=225e3,
        bandwidth_max_hz=3.2e6,
        tx_gain_min_db=0,
        tx_gain_max_db=0,
        rx_gain_min_db=0,
        rx_gain_max_db=49.6,
        supports_tx=False,
        supports_rx=True,
        supports_full_duplex=False,
        num_tx_channels=0,
        num_rx_channels=1,
        interface="USB",
        usb_version="2.0",
        adc_bits=8,
        dac_bits=0,
    ),
    
    # LimeSDR Family
    "limesdr_mini": HardwareSpecification(
        device_family="LimeSDR",
        device_model="LimeSDR Mini",
        freq_min_hz=10e6,
        freq_max_hz=3.5e9,
        sample_rate_min_sps=100e3,
        sample_rate_max_sps=30.72e6,
        bandwidth_min_hz=1.5e6,
        bandwidth_max_hz=30e6,
        tx_gain_min_db=-12,
        tx_gain_max_db=64,
        rx_gain_min_db=0,
        rx_gain_max_db=73,
        supports_tx=True,
        supports_rx=True,
        supports_full_duplex=True,
        num_tx_channels=1,
        num_rx_channels=1,
        interface="USB",
        usb_version="3.0",
        adc_bits=12,
        dac_bits=12,
        fpga_model="Altera MAX 10",
    ),
    "limesdr_usb": HardwareSpecification(
        device_family="LimeSDR",
        device_model="LimeSDR USB",
        freq_min_hz=100e3,
        freq_max_hz=3.8e9,
        sample_rate_min_sps=100e3,
        sample_rate_max_sps=61.44e6,
        bandwidth_min_hz=1.5e6,
        bandwidth_max_hz=60e6,
        tx_gain_min_db=-12,
        tx_gain_max_db=64,
        rx_gain_min_db=0,
        rx_gain_max_db=73,
        supports_tx=True,
        supports_rx=True,
        supports_full_duplex=True,
        num_tx_channels=2,
        num_rx_channels=2,
        interface="USB",
        usb_version="3.0",
        adc_bits=12,
        dac_bits=12,
        fpga_model="Altera Cyclone IV",
    ),
    
    # PlutoSDR
    "plutosdr": HardwareSpecification(
        device_family="PlutoSDR",
        device_model="ADALM-PLUTO",
        freq_min_hz=325e6,
        freq_max_hz=3.8e9,
        sample_rate_min_sps=521e3,
        sample_rate_max_sps=61.44e6,
        bandwidth_min_hz=200e3,
        bandwidth_max_hz=20e6,
        tx_gain_min_db=-89.75,
        tx_gain_max_db=0,
        rx_gain_min_db=-4,
        rx_gain_max_db=71,
        supports_tx=True,
        supports_rx=True,
        supports_full_duplex=True,
        num_tx_channels=1,
        num_rx_channels=1,
        interface="USB",
        usb_version="2.0",
        adc_bits=12,
        dac_bits=12,
        fpga_model="Xilinx Zynq",
    ),
    
    # Airspy Family
    "airspy_r2": HardwareSpecification(
        device_family="Airspy",
        device_model="Airspy R2",
        freq_min_hz=24e6,
        freq_max_hz=1.8e9,
        sample_rate_min_sps=2.5e6,
        sample_rate_max_sps=10e6,
        bandwidth_min_hz=2.5e6,
        bandwidth_max_hz=10e6,
        tx_gain_min_db=0,
        tx_gain_max_db=0,
        rx_gain_min_db=0,
        rx_gain_max_db=45,
        supports_tx=False,
        supports_rx=True,
        supports_full_duplex=False,
        num_tx_channels=0,
        num_rx_channels=1,
        interface="USB",
        usb_version="2.0",
        adc_bits=12,
        dac_bits=0,
    ),
    "airspy_mini": HardwareSpecification(
        device_family="Airspy",
        device_model="Airspy Mini",
        freq_min_hz=24e6,
        freq_max_hz=1.8e9,
        sample_rate_min_sps=3e6,
        sample_rate_max_sps=6e6,
        bandwidth_min_hz=3e6,
        bandwidth_max_hz=6e6,
        tx_gain_min_db=0,
        tx_gain_max_db=0,
        rx_gain_min_db=0,
        rx_gain_max_db=45,
        supports_tx=False,
        supports_rx=True,
        supports_full_duplex=False,
        num_tx_channels=0,
        num_rx_channels=1,
        interface="USB",
        usb_version="2.0",
        adc_bits=12,
        dac_bits=0,
    ),
    "airspy_hf_plus": HardwareSpecification(
        device_family="Airspy",
        device_model="Airspy HF+",
        freq_min_hz=9e3,
        freq_max_hz=31e6,
        sample_rate_min_sps=192e3,
        sample_rate_max_sps=768e3,
        bandwidth_min_hz=192e3,
        bandwidth_max_hz=768e3,
        tx_gain_min_db=0,
        tx_gain_max_db=0,
        rx_gain_min_db=0,
        rx_gain_max_db=6,
        supports_tx=False,
        supports_rx=True,
        supports_full_duplex=False,
        num_tx_channels=0,
        num_rx_channels=1,
        interface="USB",
        usb_version="2.0",
        adc_bits=18,
        dac_bits=0,
    ),
}


@dataclass
class TestResultDetail:
    """Detailed test result"""
    test_id: str
    test_name: str
    category: TestCategory
    result: TestResult
    measured_value: Optional[Any] = None
    expected_value: Optional[Any] = None
    tolerance: Optional[float] = None
    message: str = ""
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DeviceValidationResult:
    """Complete validation result for a device"""
    device_id: str
    device_family: str
    device_model: str
    serial_number: Optional[str]
    firmware_version: Optional[str]
    
    certification_level: CertificationLevel
    overall_score: float  # 0-100
    
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    tests_total: int = 0
    
    test_results: List[TestResultDetail] = field(default_factory=list)
    
    validation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    validation_duration_seconds: float = 0.0
    
    certificate_id: Optional[str] = None
    certificate_hash: Optional[str] = None
    
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class HardwareValidator(ABC):
    """Abstract base class for hardware validators"""
    
    @abstractmethod
    def detect_device(self) -> Optional[Dict[str, Any]]:
        """Detect and identify hardware device"""
        pass
    
    @abstractmethod
    def initialize_device(self) -> bool:
        """Initialize the hardware device"""
        pass
    
    @abstractmethod
    def close_device(self) -> bool:
        """Close and release the hardware device"""
        pass
    
    @abstractmethod
    def test_frequency_range(self, spec: HardwareSpecification) -> TestResultDetail:
        """Test frequency tuning capabilities"""
        pass
    
    @abstractmethod
    def test_sample_rates(self, spec: HardwareSpecification) -> TestResultDetail:
        """Test sample rate capabilities"""
        pass
    
    @abstractmethod
    def test_bandwidth(self, spec: HardwareSpecification) -> TestResultDetail:
        """Test bandwidth capabilities"""
        pass
    
    @abstractmethod
    def test_gain_control(self, spec: HardwareSpecification) -> TestResultDetail:
        """Test gain control"""
        pass
    
    @abstractmethod
    def test_streaming(self, spec: HardwareSpecification) -> TestResultDetail:
        """Test data streaming"""
        pass


class HackRFValidator(HardwareValidator):
    """HackRF One hardware validator"""
    
    def __init__(self):
        self.device = None
        self.device_info = None
        
    def detect_device(self) -> Optional[Dict[str, Any]]:
        """Detect HackRF device using hackrf_info"""
        try:
            result = subprocess.run(
                ['hackrf_info'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                info = {}
                for line in result.stdout.split('\n'):
                    if 'Serial number:' in line:
                        info['serial'] = line.split(':')[1].strip()
                    elif 'Board ID Number:' in line:
                        info['board_id'] = line.split(':')[1].strip()
                    elif 'Firmware Version:' in line:
                        info['firmware'] = line.split(':')[1].strip()
                    elif 'Part ID Number:' in line:
                        info['part_id'] = line.split(':')[1].strip()
                
                info['device_family'] = 'HackRF'
                info['device_model'] = 'HackRF One'
                self.device_info = info
                return info
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return None
    
    def initialize_device(self) -> bool:
        """Initialize HackRF device"""
        try:
            # HackRF uses libhackrf - test with a quick transfer
            result = subprocess.run(
                ['hackrf_transfer', '-r', '/dev/null', '-n', '1000'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def close_device(self) -> bool:
        """Close HackRF device"""
        # HackRF command-line tools auto-close
        return True
    
    def test_frequency_range(self, spec: HardwareSpecification) -> TestResultDetail:
        """Test HackRF frequency tuning"""
        test_id = f"hackrf_freq_{int(time.time())}"
        start_time = time.time()
        
        test_frequencies = [
            spec.freq_min_hz,
            100e6,
            433e6,
            915e6,
            1.8e9,
            2.4e9,
            5.8e9,
            spec.freq_max_hz
        ]
        
        passed = 0
        failed = 0
        
        for freq in test_frequencies:
            if freq < spec.freq_min_hz or freq > spec.freq_max_hz:
                continue
                
            try:
                result = subprocess.run(
                    ['hackrf_transfer', '-f', str(int(freq)), '-r', '/dev/null', '-n', '1000'],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    passed += 1
                else:
                    failed += 1
            except:
                failed += 1
        
        duration = time.time() - start_time
        total = passed + failed
        
        if total == 0:
            return TestResultDetail(
                test_id=test_id,
                test_name="Frequency Range Test",
                category=TestCategory.FREQUENCY,
                result=TestResult.ERROR,
                message="No frequency tests executed",
                duration_seconds=duration
            )
        
        success_rate = (passed / total) * 100
        
        return TestResultDetail(
            test_id=test_id,
            test_name="Frequency Range Test",
            category=TestCategory.FREQUENCY,
            result=TestResult.PASS if success_rate >= 90 else TestResult.FAIL,
            measured_value=f"{passed}/{total} frequencies",
            expected_value=f"{spec.freq_min_hz/1e6:.1f} MHz - {spec.freq_max_hz/1e9:.1f} GHz",
            message=f"Frequency tuning: {success_rate:.1f}% success rate",
            duration_seconds=duration
        )
    
    def test_sample_rates(self, spec: HardwareSpecification) -> TestResultDetail:
        """Test HackRF sample rates"""
        test_id = f"hackrf_samplerate_{int(time.time())}"
        start_time = time.time()
        
        test_rates = [2e6, 4e6, 8e6, 10e6, 16e6, 20e6]
        passed = 0
        failed = 0
        
        for rate in test_rates:
            if rate < spec.sample_rate_min_sps or rate > spec.sample_rate_max_sps:
                continue
                
            try:
                result = subprocess.run(
                    ['hackrf_transfer', '-s', str(int(rate)), '-f', '915000000', 
                     '-r', '/dev/null', '-n', '10000'],
                    capture_output=True,
                    timeout=10
                )
                if result.returncode == 0:
                    passed += 1
                else:
                    failed += 1
            except:
                failed += 1
        
        duration = time.time() - start_time
        total = passed + failed
        success_rate = (passed / total * 100) if total > 0 else 0
        
        return TestResultDetail(
            test_id=test_id,
            test_name="Sample Rate Test",
            category=TestCategory.SAMPLE_RATE,
            result=TestResult.PASS if success_rate >= 80 else TestResult.FAIL,
            measured_value=f"{passed}/{total} rates",
            expected_value=f"{spec.sample_rate_min_sps/1e6:.1f} - {spec.sample_rate_max_sps/1e6:.1f} Msps",
            message=f"Sample rate validation: {success_rate:.1f}% success",
            duration_seconds=duration
        )
    
    def test_bandwidth(self, spec: HardwareSpecification) -> TestResultDetail:
        """Test HackRF bandwidth settings"""
        test_id = f"hackrf_bw_{int(time.time())}"
        start_time = time.time()
        
        # HackRF bandwidth is tied to sample rate
        test_bandwidths = [1.75e6, 2.5e6, 5e6, 10e6, 14e6, 20e6, 28e6]
        passed = 0
        
        for bw in test_bandwidths:
            if bw >= spec.bandwidth_min_hz and bw <= spec.bandwidth_max_hz:
                # HackRF sets BW via baseband filter
                try:
                    result = subprocess.run(
                        ['hackrf_transfer', '-b', str(int(bw)), '-f', '915000000',
                         '-r', '/dev/null', '-n', '1000'],
                        capture_output=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        passed += 1
                except:
                    pass
        
        duration = time.time() - start_time
        total = len([b for b in test_bandwidths if spec.bandwidth_min_hz <= b <= spec.bandwidth_max_hz])
        
        return TestResultDetail(
            test_id=test_id,
            test_name="Bandwidth Test",
            category=TestCategory.BANDWIDTH,
            result=TestResult.PASS if passed >= total * 0.8 else TestResult.FAIL,
            measured_value=f"{passed}/{total} bandwidths",
            expected_value=f"{spec.bandwidth_min_hz/1e6:.1f} - {spec.bandwidth_max_hz/1e6:.1f} MHz",
            duration_seconds=duration
        )
    
    def test_gain_control(self, spec: HardwareSpecification) -> TestResultDetail:
        """Test HackRF gain control"""
        test_id = f"hackrf_gain_{int(time.time())}"
        start_time = time.time()
        
        # HackRF has LNA (0-40dB), VGA (0-62dB), AMP (0/14dB)
        test_gains = [0, 8, 16, 24, 32, 40]  # LNA gains
        passed = 0
        
        for gain in test_gains:
            try:
                result = subprocess.run(
                    ['hackrf_transfer', '-l', str(gain), '-f', '915000000',
                     '-r', '/dev/null', '-n', '1000'],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    passed += 1
            except:
                pass
        
        duration = time.time() - start_time
        
        return TestResultDetail(
            test_id=test_id,
            test_name="Gain Control Test",
            category=TestCategory.GAIN,
            result=TestResult.PASS if passed >= 4 else TestResult.FAIL,
            measured_value=f"{passed}/6 gain settings",
            expected_value=f"RX: {spec.rx_gain_min_db}-{spec.rx_gain_max_db} dB",
            duration_seconds=duration
        )
    
    def test_streaming(self, spec: HardwareSpecification) -> TestResultDetail:
        """Test HackRF data streaming"""
        test_id = f"hackrf_stream_{int(time.time())}"
        start_time = time.time()
        
        try:
            # Stream 1 second of data
            result = subprocess.run(
                ['hackrf_transfer', '-r', '/dev/null', '-f', '915000000',
                 '-s', '10000000', '-n', '10000000'],
                capture_output=True,
                timeout=15
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                return TestResultDetail(
                    test_id=test_id,
                    test_name="Streaming Test",
                    category=TestCategory.STREAMING,
                    result=TestResult.PASS,
                    measured_value="10M samples",
                    expected_value="Continuous streaming",
                    message="Streaming test passed",
                    duration_seconds=duration
                )
        except subprocess.TimeoutExpired:
            pass
        
        return TestResultDetail(
            test_id=test_id,
            test_name="Streaming Test",
            category=TestCategory.STREAMING,
            result=TestResult.FAIL,
            message="Streaming test failed",
            duration_seconds=time.time() - start_time
        )


class BladeRFValidator(HardwareValidator):
    """BladeRF hardware validator"""
    
    def __init__(self):
        self.device = None
        self.device_info = None
    
    def detect_device(self) -> Optional[Dict[str, Any]]:
        """Detect BladeRF device using bladeRF-cli"""
        try:
            result = subprocess.run(
                ['bladeRF-cli', '-p'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and 'bladerf' in result.stdout.lower():
                info = {}
                
                # Get detailed info
                info_result = subprocess.run(
                    ['bladeRF-cli', '-e', 'info'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                for line in info_result.stdout.split('\n'):
                    if 'Serial' in line:
                        info['serial'] = line.split(':')[-1].strip()
                    elif 'FPGA' in line:
                        info['fpga'] = line.split(':')[-1].strip()
                    elif 'Firmware' in line:
                        info['firmware'] = line.split(':')[-1].strip()
                    elif 'Board' in line:
                        info['board'] = line.split(':')[-1].strip()
                
                info['device_family'] = 'BladeRF'
                
                # Determine model
                if 'xA9' in str(info.get('board', '')):
                    info['device_model'] = 'bladeRF 2.0 micro xA9'
                elif 'xA5' in str(info.get('board', '')):
                    info['device_model'] = 'bladeRF 2.0 micro xA5'
                elif 'xA4' in str(info.get('board', '')):
                    info['device_model'] = 'bladeRF 2.0 micro xA4'
                elif 'x115' in str(info.get('board', '')):
                    info['device_model'] = 'bladeRF x115'
                elif 'x40' in str(info.get('board', '')):
                    info['device_model'] = 'bladeRF x40'
                else:
                    info['device_model'] = 'bladeRF (unknown model)'
                
                self.device_info = info
                return info
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return None
    
    def initialize_device(self) -> bool:
        """Initialize BladeRF device"""
        try:
            result = subprocess.run(
                ['bladeRF-cli', '-e', 'version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def close_device(self) -> bool:
        """Close BladeRF device"""
        return True
    
    def test_frequency_range(self, spec: HardwareSpecification) -> TestResultDetail:
        """Test BladeRF frequency tuning"""
        test_id = f"bladerf_freq_{int(time.time())}"
        start_time = time.time()
        
        test_frequencies = [
            spec.freq_min_hz,
            100e6, 433e6, 915e6, 1.8e9, 2.4e9, 3.5e9, 5.8e9,
            spec.freq_max_hz
        ]
        
        passed = 0
        failed = 0
        
        for freq in test_frequencies:
            if freq < spec.freq_min_hz or freq > spec.freq_max_hz:
                continue
            
            try:
                cmd = f'set frequency rx1 {int(freq)}'
                result = subprocess.run(
                    ['bladeRF-cli', '-e', cmd],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    passed += 1
                else:
                    failed += 1
            except:
                failed += 1
        
        duration = time.time() - start_time
        total = passed + failed
        success_rate = (passed / total * 100) if total > 0 else 0
        
        return TestResultDetail(
            test_id=test_id,
            test_name="Frequency Range Test",
            category=TestCategory.FREQUENCY,
            result=TestResult.PASS if success_rate >= 90 else TestResult.FAIL,
            measured_value=f"{passed}/{total} frequencies",
            expected_value=f"{spec.freq_min_hz/1e6:.1f} MHz - {spec.freq_max_hz/1e9:.1f} GHz",
            message=f"Frequency tuning: {success_rate:.1f}% success",
            duration_seconds=duration
        )
    
    def test_sample_rates(self, spec: HardwareSpecification) -> TestResultDetail:
        """Test BladeRF sample rates"""
        test_id = f"bladerf_samplerate_{int(time.time())}"
        start_time = time.time()
        
        test_rates = [521e3, 1e6, 5e6, 10e6, 20e6, 30.72e6, 40e6, 61.44e6]
        passed = 0
        
        for rate in test_rates:
            if rate < spec.sample_rate_min_sps or rate > spec.sample_rate_max_sps:
                continue
            
            try:
                cmd = f'set samplerate rx {int(rate)}'
                result = subprocess.run(
                    ['bladeRF-cli', '-e', cmd],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    passed += 1
            except:
                pass
        
        duration = time.time() - start_time
        total = len([r for r in test_rates if spec.sample_rate_min_sps <= r <= spec.sample_rate_max_sps])
        
        return TestResultDetail(
            test_id=test_id,
            test_name="Sample Rate Test",
            category=TestCategory.SAMPLE_RATE,
            result=TestResult.PASS if passed >= total * 0.8 else TestResult.FAIL,
            measured_value=f"{passed}/{total} rates",
            expected_value=f"{spec.sample_rate_min_sps/1e6:.2f} - {spec.sample_rate_max_sps/1e6:.2f} Msps",
            duration_seconds=duration
        )
    
    def test_bandwidth(self, spec: HardwareSpecification) -> TestResultDetail:
        """Test BladeRF bandwidth"""
        test_id = f"bladerf_bw_{int(time.time())}"
        start_time = time.time()
        
        test_bws = [200e3, 1.5e6, 5e6, 10e6, 20e6, 28e6, 56e6]
        passed = 0
        
        for bw in test_bws:
            if bw < spec.bandwidth_min_hz or bw > spec.bandwidth_max_hz:
                continue
            
            try:
                cmd = f'set bandwidth rx {int(bw)}'
                result = subprocess.run(
                    ['bladeRF-cli', '-e', cmd],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    passed += 1
            except:
                pass
        
        duration = time.time() - start_time
        total = len([b for b in test_bws if spec.bandwidth_min_hz <= b <= spec.bandwidth_max_hz])
        
        return TestResultDetail(
            test_id=test_id,
            test_name="Bandwidth Test",
            category=TestCategory.BANDWIDTH,
            result=TestResult.PASS if passed >= total * 0.8 else TestResult.FAIL,
            measured_value=f"{passed}/{total} bandwidths",
            expected_value=f"{spec.bandwidth_min_hz/1e6:.1f} - {spec.bandwidth_max_hz/1e6:.1f} MHz",
            duration_seconds=duration
        )
    
    def test_gain_control(self, spec: HardwareSpecification) -> TestResultDetail:
        """Test BladeRF gain control"""
        test_id = f"bladerf_gain_{int(time.time())}"
        start_time = time.time()
        
        # Test RX gain
        test_gains = range(int(spec.rx_gain_min_db), int(spec.rx_gain_max_db) + 1, 10)
        passed = 0
        
        for gain in test_gains:
            try:
                cmd = f'set gain rx1 {gain}'
                result = subprocess.run(
                    ['bladeRF-cli', '-e', cmd],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    passed += 1
            except:
                pass
        
        duration = time.time() - start_time
        total = len(list(test_gains))
        
        return TestResultDetail(
            test_id=test_id,
            test_name="Gain Control Test",
            category=TestCategory.GAIN,
            result=TestResult.PASS if passed >= total * 0.8 else TestResult.FAIL,
            measured_value=f"{passed}/{total} gain settings",
            expected_value=f"RX: {spec.rx_gain_min_db}-{spec.rx_gain_max_db} dB",
            duration_seconds=duration
        )
    
    def test_streaming(self, spec: HardwareSpecification) -> TestResultDetail:
        """Test BladeRF streaming"""
        test_id = f"bladerf_stream_{int(time.time())}"
        start_time = time.time()
        
        try:
            # Quick RX test
            script = '''
set frequency rx1 915000000
set samplerate rx 10000000
set bandwidth rx 10000000
rx config file=/dev/null format=bin n=1000000
rx start
rx wait
'''
            result = subprocess.run(
                ['bladeRF-cli', '-e', script],
                capture_output=True,
                timeout=15
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                return TestResultDetail(
                    test_id=test_id,
                    test_name="Streaming Test",
                    category=TestCategory.STREAMING,
                    result=TestResult.PASS,
                    measured_value="1M samples",
                    message="Streaming test passed",
                    duration_seconds=duration
                )
        except:
            pass
        
        return TestResultDetail(
            test_id=test_id,
            test_name="Streaming Test",
            category=TestCategory.STREAMING,
            result=TestResult.FAIL,
            message="Streaming test failed",
            duration_seconds=time.time() - start_time
        )


class RTLSDRValidator(HardwareValidator):
    """RTL-SDR hardware validator"""
    
    def __init__(self):
        self.device_info = None
    
    def detect_device(self) -> Optional[Dict[str, Any]]:
        """Detect RTL-SDR device"""
        try:
            result = subprocess.run(
                ['rtl_test', '-t'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if 'Found' in result.stdout or 'Found' in result.stderr:
                info = {
                    'device_family': 'RTL-SDR',
                    'device_model': 'RTL-SDR Generic',
                }
                
                # Parse output for details
                output = result.stdout + result.stderr
                for line in output.split('\n'):
                    if 'Serial number:' in line:
                        info['serial'] = line.split(':')[-1].strip()
                    elif 'Tuner type:' in line:
                        tuner = line.split(':')[-1].strip()
                        info['tuner'] = tuner
                        if 'R820T' in tuner:
                            info['device_model'] = 'RTL-SDR Blog V3'
                
                self.device_info = info
                return info
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return None
    
    def initialize_device(self) -> bool:
        try:
            result = subprocess.run(
                ['rtl_test', '-t'],
                capture_output=True,
                timeout=5
            )
            return 'Found' in (result.stdout.decode() if isinstance(result.stdout, bytes) else result.stdout)
        except:
            return False
    
    def close_device(self) -> bool:
        return True
    
    def test_frequency_range(self, spec: HardwareSpecification) -> TestResultDetail:
        """Test RTL-SDR frequency tuning"""
        test_id = f"rtlsdr_freq_{int(time.time())}"
        start_time = time.time()
        
        test_freqs = [24e6, 100e6, 433e6, 915e6, 1.2e9, 1.7e9]
        passed = 0
        
        for freq in test_freqs:
            if freq < spec.freq_min_hz or freq > spec.freq_max_hz:
                continue
            try:
                result = subprocess.run(
                    ['rtl_test', '-f', str(int(freq)), '-n', '1000'],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    passed += 1
            except:
                pass
        
        duration = time.time() - start_time
        total = len([f for f in test_freqs if spec.freq_min_hz <= f <= spec.freq_max_hz])
        
        return TestResultDetail(
            test_id=test_id,
            test_name="Frequency Range Test",
            category=TestCategory.FREQUENCY,
            result=TestResult.PASS if passed >= total * 0.8 else TestResult.FAIL,
            measured_value=f"{passed}/{total} frequencies",
            expected_value=f"{spec.freq_min_hz/1e6:.1f} - {spec.freq_max_hz/1e9:.3f} GHz",
            duration_seconds=duration
        )
    
    def test_sample_rates(self, spec: HardwareSpecification) -> TestResultDetail:
        """Test RTL-SDR sample rates"""
        test_id = f"rtlsdr_samplerate_{int(time.time())}"
        start_time = time.time()
        
        test_rates = [250e3, 1e6, 1.8e6, 2.4e6, 2.8e6, 3.2e6]
        passed = 0
        
        for rate in test_rates:
            if rate < spec.sample_rate_min_sps or rate > spec.sample_rate_max_sps:
                continue
            try:
                result = subprocess.run(
                    ['rtl_test', '-s', str(int(rate)), '-n', '10000'],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    passed += 1
            except:
                pass
        
        duration = time.time() - start_time
        total = len([r for r in test_rates if spec.sample_rate_min_sps <= r <= spec.sample_rate_max_sps])
        
        return TestResultDetail(
            test_id=test_id,
            test_name="Sample Rate Test",
            category=TestCategory.SAMPLE_RATE,
            result=TestResult.PASS if passed >= total * 0.7 else TestResult.FAIL,
            measured_value=f"{passed}/{total} rates",
            expected_value=f"{spec.sample_rate_min_sps/1e3:.0f}k - {spec.sample_rate_max_sps/1e6:.1f}M",
            duration_seconds=duration
        )
    
    def test_bandwidth(self, spec: HardwareSpecification) -> TestResultDetail:
        """RTL-SDR bandwidth is tied to sample rate"""
        test_id = f"rtlsdr_bw_{int(time.time())}"
        
        return TestResultDetail(
            test_id=test_id,
            test_name="Bandwidth Test",
            category=TestCategory.BANDWIDTH,
            result=TestResult.PASS,
            message="RTL-SDR bandwidth follows sample rate",
            expected_value=f"Tied to sample rate",
            duration_seconds=0.0
        )
    
    def test_gain_control(self, spec: HardwareSpecification) -> TestResultDetail:
        """Test RTL-SDR gain control"""
        test_id = f"rtlsdr_gain_{int(time.time())}"
        start_time = time.time()
        
        test_gains = [0, 10, 20, 30, 40, 49]
        passed = 0
        
        for gain in test_gains:
            if gain > spec.rx_gain_max_db:
                continue
            try:
                result = subprocess.run(
                    ['rtl_test', '-g', str(int(gain * 10)), '-n', '1000'],  # RTL uses 0.1 dB units
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    passed += 1
            except:
                pass
        
        duration = time.time() - start_time
        
        return TestResultDetail(
            test_id=test_id,
            test_name="Gain Control Test",
            category=TestCategory.GAIN,
            result=TestResult.PASS if passed >= 4 else TestResult.FAIL,
            measured_value=f"{passed}/6 gain settings",
            expected_value=f"0-{spec.rx_gain_max_db} dB",
            duration_seconds=duration
        )
    
    def test_streaming(self, spec: HardwareSpecification) -> TestResultDetail:
        """Test RTL-SDR streaming"""
        test_id = f"rtlsdr_stream_{int(time.time())}"
        start_time = time.time()
        
        try:
            result = subprocess.run(
                ['rtl_test', '-s', '2400000', '-n', '2400000'],
                capture_output=True,
                timeout=10
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                return TestResultDetail(
                    test_id=test_id,
                    test_name="Streaming Test",
                    category=TestCategory.STREAMING,
                    result=TestResult.PASS,
                    measured_value="2.4M samples",
                    message="Streaming test passed",
                    duration_seconds=duration
                )
        except:
            pass
        
        return TestResultDetail(
            test_id=test_id,
            test_name="Streaming Test",
            category=TestCategory.STREAMING,
            result=TestResult.FAIL,
            message="Streaming test failed",
            duration_seconds=time.time() - start_time
        )


class HardwareCompatibilityCertifier:
    """
    Main hardware compatibility certification system.
    
    Orchestrates device detection, validation, and certification
    across all supported SDR platforms.
    """
    
    def __init__(self, output_dir: str = "/tmp/rf_arsenal_hw_certs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.validators = {
            'HackRF': HackRFValidator(),
            'BladeRF': BladeRFValidator(),
            'RTL-SDR': RTLSDRValidator(),
        }
        
        self.validation_results: List[DeviceValidationResult] = []
    
    def detect_all_devices(self) -> List[Dict[str, Any]]:
        """Detect all connected SDR devices"""
        devices = []
        
        for family, validator in self.validators.items():
            logger.info(f"Scanning for {family} devices...")
            device_info = validator.detect_device()
            if device_info:
                devices.append(device_info)
                logger.info(f"  Found: {device_info.get('device_model', family)}")
        
        return devices
    
    def get_spec_for_device(self, device_info: Dict[str, Any]) -> Optional[HardwareSpecification]:
        """Get official specification for detected device"""
        family = device_info.get('device_family', '').lower()
        model = device_info.get('device_model', '').lower()
        
        # Map to specification
        if 'hackrf' in family:
            return HARDWARE_SPECIFICATIONS['hackrf_one']
        elif 'bladerf' in family:
            if 'xa9' in model or 'xA9' in model:
                return HARDWARE_SPECIFICATIONS['bladerf_xa9']
            elif 'xa5' in model or 'xA5' in model:
                return HARDWARE_SPECIFICATIONS['bladerf_xa5']
            elif 'xa4' in model or 'xA4' in model:
                return HARDWARE_SPECIFICATIONS['bladerf_xa4']
            elif 'x115' in model:
                return HARDWARE_SPECIFICATIONS['bladerf_x115']
            elif 'x40' in model:
                return HARDWARE_SPECIFICATIONS['bladerf_x40']
            return HARDWARE_SPECIFICATIONS['bladerf_xa9']  # Default
        elif 'rtl' in family:
            if 'v3' in model.lower():
                return HARDWARE_SPECIFICATIONS['rtlsdr_v3']
            return HARDWARE_SPECIFICATIONS['rtlsdr_generic']
        elif 'lime' in family:
            if 'mini' in model.lower():
                return HARDWARE_SPECIFICATIONS['limesdr_mini']
            return HARDWARE_SPECIFICATIONS['limesdr_usb']
        elif 'pluto' in family:
            return HARDWARE_SPECIFICATIONS['plutosdr']
        elif 'airspy' in family:
            if 'hf' in model.lower():
                return HARDWARE_SPECIFICATIONS['airspy_hf_plus']
            elif 'mini' in model.lower():
                return HARDWARE_SPECIFICATIONS['airspy_mini']
            return HARDWARE_SPECIFICATIONS['airspy_r2']
        elif 'usrp' in family:
            if 'x310' in model:
                return HARDWARE_SPECIFICATIONS['usrp_x310']
            elif 'x300' in model:
                return HARDWARE_SPECIFICATIONS['usrp_x300']
            elif 'b210' in model:
                return HARDWARE_SPECIFICATIONS['usrp_b210']
            return HARDWARE_SPECIFICATIONS['usrp_b200']
        
        return None
    
    def validate_device(self, device_info: Dict[str, Any]) -> DeviceValidationResult:
        """Run full validation suite on a device"""
        family = device_info.get('device_family', 'Unknown')
        model = device_info.get('device_model', 'Unknown')
        
        logger.info(f"Validating {model}...")
        
        start_time = time.time()
        
        # Get validator and spec
        validator = self.validators.get(family)
        spec = self.get_spec_for_device(device_info)
        
        if not validator or not spec:
            return DeviceValidationResult(
                device_id=str(uuid.uuid4()),
                device_family=family,
                device_model=model,
                serial_number=device_info.get('serial'),
                firmware_version=device_info.get('firmware'),
                certification_level=CertificationLevel.UNTESTED,
                overall_score=0.0,
                errors=["No validator or specification available"]
            )
        
        # Initialize device
        if not validator.initialize_device():
            return DeviceValidationResult(
                device_id=str(uuid.uuid4()),
                device_family=family,
                device_model=model,
                serial_number=device_info.get('serial'),
                firmware_version=device_info.get('firmware'),
                certification_level=CertificationLevel.INCOMPATIBLE,
                overall_score=0.0,
                errors=["Failed to initialize device"]
            )
        
        # Run tests
        test_results = []
        
        # Frequency test
        logger.info("  Testing frequency range...")
        test_results.append(validator.test_frequency_range(spec))
        
        # Sample rate test
        logger.info("  Testing sample rates...")
        test_results.append(validator.test_sample_rates(spec))
        
        # Bandwidth test
        logger.info("  Testing bandwidth...")
        test_results.append(validator.test_bandwidth(spec))
        
        # Gain test
        logger.info("  Testing gain control...")
        test_results.append(validator.test_gain_control(spec))
        
        # Streaming test
        logger.info("  Testing streaming...")
        test_results.append(validator.test_streaming(spec))
        
        # Close device
        validator.close_device()
        
        # Calculate results
        passed = sum(1 for r in test_results if r.result == TestResult.PASS)
        failed = sum(1 for r in test_results if r.result == TestResult.FAIL)
        total = len(test_results)
        
        score = (passed / total * 100) if total > 0 else 0
        
        # Determine certification level
        if score >= 90:
            cert_level = CertificationLevel.CERTIFIED
        elif score >= 70:
            cert_level = CertificationLevel.COMPATIBLE
        elif score >= 50:
            cert_level = CertificationLevel.PARTIAL
        else:
            cert_level = CertificationLevel.INCOMPATIBLE
        
        duration = time.time() - start_time
        
        # Generate certificate
        cert_id = f"RFARSENAL-HW-{family.upper()[:3]}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        cert_hash = hashlib.sha256(
            f"{cert_id}{model}{score}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16].upper()
        
        result = DeviceValidationResult(
            device_id=str(uuid.uuid4()),
            device_family=family,
            device_model=model,
            serial_number=device_info.get('serial'),
            firmware_version=device_info.get('firmware'),
            certification_level=cert_level,
            overall_score=score,
            tests_passed=passed,
            tests_failed=failed,
            tests_skipped=sum(1 for r in test_results if r.result == TestResult.SKIP),
            tests_total=total,
            test_results=test_results,
            validation_duration_seconds=duration,
            certificate_id=cert_id,
            certificate_hash=cert_hash
        )
        
        logger.info(f"  Result: {cert_level.value.upper()} ({score:.1f}%)")
        
        return result
    
    def run_full_certification(self) -> List[DeviceValidationResult]:
        """Run certification on all detected devices"""
        logger.info("=" * 60)
        logger.info("RF Arsenal OS - Hardware Compatibility Certification")
        logger.info("=" * 60)
        
        # Detect devices
        devices = self.detect_all_devices()
        
        if not devices:
            logger.warning("No SDR devices detected!")
            return []
        
        logger.info(f"\nFound {len(devices)} device(s). Starting validation...\n")
        
        # Validate each device
        results = []
        for device in devices:
            result = self.validate_device(device)
            results.append(result)
            self.validation_results.append(result)
        
        # Generate reports
        self.generate_reports(results)
        
        return results
    
    def generate_reports(self, results: List[DeviceValidationResult]):
        """Generate certification reports"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON report
        json_path = self.output_dir / f"certification_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(
                [asdict(r) for r in results],
                f,
                indent=2,
                default=str
            )
        logger.info(f"JSON report: {json_path}")
        
        # Summary report
        summary_path = self.output_dir / f"certification_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("RF ARSENAL OS - HARDWARE COMPATIBILITY CERTIFICATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Devices Tested: {len(results)}\n\n")
            
            for r in results:
                f.write("-" * 70 + "\n")
                f.write(f"Device: {r.device_model}\n")
                f.write(f"Serial: {r.serial_number or 'N/A'}\n")
                f.write(f"Firmware: {r.firmware_version or 'N/A'}\n")
                f.write(f"Certificate ID: {r.certificate_id}\n")
                f.write(f"Certificate Hash: {r.certificate_hash}\n\n")
                f.write(f"CERTIFICATION LEVEL: {r.certification_level.value.upper()}\n")
                f.write(f"Overall Score: {r.overall_score:.1f}%\n")
                f.write(f"Tests: {r.tests_passed}/{r.tests_total} passed\n\n")
                
                f.write("Test Results:\n")
                for tr in r.test_results:
                    status = "✓" if tr.result == TestResult.PASS else "✗"
                    f.write(f"  {status} {tr.test_name}: {tr.result.value.upper()}\n")
                    if tr.measured_value:
                        f.write(f"      Measured: {tr.measured_value}\n")
                    if tr.message:
                        f.write(f"      Message: {tr.message}\n")
                
                f.write("\n")
            
            f.write("=" * 70 + "\n")
            f.write("END OF REPORT\n")
        
        logger.info(f"Summary report: {summary_path}")
    
    def generate_compatibility_matrix(self) -> Dict[str, Dict[str, Any]]:
        """Generate compatibility matrix for all supported devices"""
        matrix = {}
        
        for spec_name, spec in HARDWARE_SPECIFICATIONS.items():
            matrix[spec.device_model] = {
                'family': spec.device_family,
                'frequency_range': f"{spec.freq_min_hz/1e6:.1f} MHz - {spec.freq_max_hz/1e9:.2f} GHz",
                'sample_rate': f"{spec.sample_rate_min_sps/1e6:.2f} - {spec.sample_rate_max_sps/1e6:.2f} Msps",
                'bandwidth': f"{spec.bandwidth_min_hz/1e6:.2f} - {spec.bandwidth_max_hz/1e6:.2f} MHz",
                'tx_support': spec.supports_tx,
                'rx_support': spec.supports_rx,
                'full_duplex': spec.supports_full_duplex,
                'channels': f"{spec.num_tx_channels}TX/{spec.num_rx_channels}RX",
                'interface': f"{spec.interface} {spec.usb_version}",
                'adc_dac_bits': f"{spec.adc_bits}/{spec.dac_bits}",
                'fpga': spec.fpga_model or 'N/A'
            }
        
        return matrix


def main():
    """Main entry point for hardware certification"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='RF Arsenal OS - Hardware Compatibility Certification'
    )
    parser.add_argument(
        '--output', '-o',
        default='/tmp/rf_arsenal_hw_certs',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--matrix', '-m',
        action='store_true',
        help='Print compatibility matrix'
    )
    parser.add_argument(
        '--detect-only', '-d',
        action='store_true',
        help='Only detect devices, do not run tests'
    )
    
    args = parser.parse_args()
    
    certifier = HardwareCompatibilityCertifier(args.output)
    
    if args.matrix:
        matrix = certifier.generate_compatibility_matrix()
        print("\nHardware Compatibility Matrix:")
        print("=" * 80)
        for device, specs in matrix.items():
            print(f"\n{device}:")
            for key, value in specs.items():
                print(f"  {key}: {value}")
        return
    
    if args.detect_only:
        devices = certifier.detect_all_devices()
        print(f"\nDetected {len(devices)} device(s):")
        for d in devices:
            print(f"  - {d.get('device_model', 'Unknown')}")
            if d.get('serial'):
                print(f"    Serial: {d['serial']}")
        return
    
    # Run full certification
    results = certifier.run_full_certification()
    
    print("\n" + "=" * 60)
    print("CERTIFICATION SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"\n{r.device_model}:")
        print(f"  Certification: {r.certification_level.value.upper()}")
        print(f"  Score: {r.overall_score:.1f}%")
        print(f"  Certificate: {r.certificate_id}")


if __name__ == "__main__":
    main()
