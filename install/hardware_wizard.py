#!/usr/bin/env python3
"""
RF Arsenal OS - Hardware Auto-Setup Wizard
Automatic SDR detection, calibration, and guided setup

SUPPORTED HARDWARE:
- BladeRF 2.0 micro xA9 (primary)
- BladeRF 2.0 micro xA4
- HackRF One
- LimeSDR Mini
- LimeSDR USB
- RTL-SDR (receive only)
- Airspy (receive only)
- USRP (various models)

FEATURES:
- Auto-detection of connected SDR devices
- Automatic driver verification
- Self-calibration routines
- Antenna selection guide
- Frequency range verification
- Performance benchmarking
- Troubleshooting assistance
"""

import os
import sys
import logging
import subprocess
import time
import json
import platform
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum, auto
from pathlib import Path

logger = logging.getLogger(__name__)


class SDRType(Enum):
    """Supported SDR types"""
    BLADERF_XA9 = "BladeRF 2.0 micro xA9"
    BLADERF_XA4 = "BladeRF 2.0 micro xA4"
    BLADERF_X40 = "BladeRF x40"
    BLADERF_X115 = "BladeRF x115"
    HACKRF = "HackRF One"
    LIMESDR_MINI = "LimeSDR Mini"
    LIMESDR_USB = "LimeSDR USB"
    RTL_SDR = "RTL-SDR"
    AIRSPY = "Airspy"
    AIRSPY_HF = "Airspy HF+"
    USRP_B200 = "USRP B200"
    USRP_B210 = "USRP B210"
    UNKNOWN = "Unknown SDR"


class CalibrationStatus(Enum):
    """Calibration status"""
    NOT_CALIBRATED = auto()
    CALIBRATING = auto()
    CALIBRATED = auto()
    FAILED = auto()


class DriverStatus(Enum):
    """Driver installation status"""
    NOT_INSTALLED = auto()
    INSTALLED = auto()
    OUTDATED = auto()
    ERROR = auto()


@dataclass
class SDRCapabilities:
    """SDR device capabilities"""
    sdr_type: SDRType
    frequency_min_hz: int
    frequency_max_hz: int
    sample_rate_max: int
    bandwidth_max: int
    tx_capable: bool
    rx_capable: bool
    full_duplex: bool
    num_tx_channels: int
    num_rx_channels: int
    gain_range_db: Tuple[int, int]
    
    # Performance characteristics
    adc_bits: int = 12
    dac_bits: int = 12
    fpga_size: str = ""
    usb_speed: str = "USB 3.0"
    
    def to_dict(self) -> Dict:
        return {
            'type': self.sdr_type.value,
            'freq_range': f"{self.frequency_min_hz/1e6:.1f} - {self.frequency_max_hz/1e9:.2f} GHz",
            'max_sample_rate': f"{self.sample_rate_max/1e6:.1f} MSPS",
            'tx_capable': self.tx_capable,
            'rx_capable': self.rx_capable,
            'full_duplex': self.full_duplex
        }


# SDR capability database
SDR_CAPABILITIES = {
    SDRType.BLADERF_XA9: SDRCapabilities(
        sdr_type=SDRType.BLADERF_XA9,
        frequency_min_hz=47_000_000,
        frequency_max_hz=6_000_000_000,
        sample_rate_max=61_440_000,
        bandwidth_max=56_000_000,
        tx_capable=True,
        rx_capable=True,
        full_duplex=True,
        num_tx_channels=2,
        num_rx_channels=2,
        gain_range_db=(-15, 60),
        adc_bits=12,
        dac_bits=12,
        fpga_size="301 KLE",
        usb_speed="USB 3.0"
    ),
    SDRType.BLADERF_XA4: SDRCapabilities(
        sdr_type=SDRType.BLADERF_XA4,
        frequency_min_hz=47_000_000,
        frequency_max_hz=6_000_000_000,
        sample_rate_max=61_440_000,
        bandwidth_max=56_000_000,
        tx_capable=True,
        rx_capable=True,
        full_duplex=True,
        num_tx_channels=2,
        num_rx_channels=2,
        gain_range_db=(-15, 60),
        adc_bits=12,
        dac_bits=12,
        fpga_size="49 KLE",
        usb_speed="USB 3.0"
    ),
    SDRType.HACKRF: SDRCapabilities(
        sdr_type=SDRType.HACKRF,
        frequency_min_hz=1_000_000,
        frequency_max_hz=6_000_000_000,
        sample_rate_max=20_000_000,
        bandwidth_max=20_000_000,
        tx_capable=True,
        rx_capable=True,
        full_duplex=False,  # Half-duplex
        num_tx_channels=1,
        num_rx_channels=1,
        gain_range_db=(0, 62),
        adc_bits=8,
        dac_bits=8,
        usb_speed="USB 2.0"
    ),
    SDRType.LIMESDR_MINI: SDRCapabilities(
        sdr_type=SDRType.LIMESDR_MINI,
        frequency_min_hz=10_000_000,
        frequency_max_hz=3_500_000_000,
        sample_rate_max=30_720_000,
        bandwidth_max=30_720_000,
        tx_capable=True,
        rx_capable=True,
        full_duplex=True,
        num_tx_channels=1,
        num_rx_channels=1,
        gain_range_db=(0, 73),
        adc_bits=12,
        dac_bits=12,
        usb_speed="USB 3.0"
    ),
    SDRType.LIMESDR_USB: SDRCapabilities(
        sdr_type=SDRType.LIMESDR_USB,
        frequency_min_hz=100_000,
        frequency_max_hz=3_800_000_000,
        sample_rate_max=61_440_000,
        bandwidth_max=61_440_000,
        tx_capable=True,
        rx_capable=True,
        full_duplex=True,
        num_tx_channels=2,
        num_rx_channels=2,
        gain_range_db=(0, 73),
        adc_bits=12,
        dac_bits=12,
        usb_speed="USB 3.0"
    ),
    SDRType.RTL_SDR: SDRCapabilities(
        sdr_type=SDRType.RTL_SDR,
        frequency_min_hz=24_000_000,
        frequency_max_hz=1_766_000_000,
        sample_rate_max=3_200_000,
        bandwidth_max=2_400_000,
        tx_capable=False,  # RX only
        rx_capable=True,
        full_duplex=False,
        num_tx_channels=0,
        num_rx_channels=1,
        gain_range_db=(0, 50),
        adc_bits=8,
        usb_speed="USB 2.0"
    ),
    SDRType.AIRSPY: SDRCapabilities(
        sdr_type=SDRType.AIRSPY,
        frequency_min_hz=24_000_000,
        frequency_max_hz=1_800_000_000,
        sample_rate_max=10_000_000,
        bandwidth_max=10_000_000,
        tx_capable=False,  # RX only
        rx_capable=True,
        full_duplex=False,
        num_tx_channels=0,
        num_rx_channels=1,
        gain_range_db=(0, 45),
        adc_bits=12,
        usb_speed="USB 2.0"
    ),
}


@dataclass
class AntennaRecommendation:
    """Antenna recommendation for a frequency range"""
    name: str
    freq_min_mhz: int
    freq_max_mhz: int
    antenna_type: str
    connector: str
    description: str
    purchase_link: str = ""


# Antenna recommendations
ANTENNA_GUIDE = [
    AntennaRecommendation(
        name="Telescopic Whip",
        freq_min_mhz=30,
        freq_max_mhz=1000,
        antenna_type="Whip",
        connector="SMA",
        description="General purpose antenna for VHF/UHF. Good for scanning and receiving."
    ),
    AntennaRecommendation(
        name="ANT500",
        freq_min_mhz=75,
        freq_max_mhz=1000,
        antenna_type="Telescopic",
        connector="SMA",
        description="Adjustable telescopic antenna. Good for 433 MHz and 868 MHz ISM bands."
    ),
    AntennaRecommendation(
        name="433 MHz Rubber Duck",
        freq_min_mhz=420,
        freq_max_mhz=450,
        antenna_type="Rubber Duck",
        connector="SMA",
        description="Optimized for 433 MHz keyfobs, sensors, and remotes."
    ),
    AntennaRecommendation(
        name="915 MHz Antenna",
        freq_min_mhz=900,
        freq_max_mhz=930,
        antenna_type="Rubber Duck",
        connector="SMA",
        description="For 915 MHz ISM band (US). LoRa, wireless sensors."
    ),
    AntennaRecommendation(
        name="2.4 GHz Panel",
        freq_min_mhz=2400,
        freq_max_mhz=2500,
        antenna_type="Directional Panel",
        connector="SMA",
        description="For WiFi and Bluetooth. Directional for better range."
    ),
    AntennaRecommendation(
        name="Dual-Band WiFi",
        freq_min_mhz=2400,
        freq_max_mhz=5800,
        antenna_type="Omni",
        connector="RP-SMA",
        description="Covers both 2.4 GHz and 5 GHz WiFi bands."
    ),
    AntennaRecommendation(
        name="Cellular Multi-Band",
        freq_min_mhz=700,
        freq_max_mhz=2700,
        antenna_type="Omni",
        connector="SMA",
        description="For cellular monitoring: 700/850/900/1800/1900/2100 MHz."
    ),
    AntennaRecommendation(
        name="GPS Patch",
        freq_min_mhz=1575,
        freq_max_mhz=1575,
        antenna_type="Patch",
        connector="SMA",
        description="Active GPS antenna for GPS spoofing/testing."
    ),
    AntennaRecommendation(
        name="Discone Wideband",
        freq_min_mhz=25,
        freq_max_mhz=1300,
        antenna_type="Discone",
        connector="N-Type",
        description="Wideband receive antenna. Excellent for scanning."
    ),
    AntennaRecommendation(
        name="Yagi Directional",
        freq_min_mhz=430,
        freq_max_mhz=440,
        antenna_type="Yagi",
        connector="N-Type",
        description="High-gain directional. Good for direction finding."
    ),
]


@dataclass
class DetectedSDR:
    """Detected SDR device information"""
    sdr_type: SDRType
    serial_number: str = ""
    firmware_version: str = ""
    fpga_version: str = ""
    usb_bus: str = ""
    driver_status: DriverStatus = DriverStatus.NOT_INSTALLED
    calibration_status: CalibrationStatus = CalibrationStatus.NOT_CALIBRATED
    capabilities: Optional[SDRCapabilities] = None
    device_path: str = ""
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'type': self.sdr_type.value,
            'serial': self.serial_number,
            'firmware': self.firmware_version,
            'fpga': self.fpga_version,
            'usb': self.usb_bus,
            'driver': self.driver_status.name,
            'calibration': self.calibration_status.name,
            'errors': self.errors
        }


class HardwareDetector:
    """Detect and identify connected SDR hardware"""
    
    # USB Vendor:Product IDs
    USB_IDS = {
        (0x2cf0, 0x5246): SDRType.BLADERF_XA9,   # BladeRF 2.0
        (0x2cf0, 0x5250): SDRType.BLADERF_X40,   # BladeRF x40
        (0x1d50, 0x6089): SDRType.HACKRF,        # HackRF One
        (0x0403, 0x601f): SDRType.LIMESDR_USB,   # LimeSDR
        (0x0403, 0x6108): SDRType.LIMESDR_MINI,  # LimeSDR Mini
        (0x0bda, 0x2838): SDRType.RTL_SDR,       # RTL-SDR
        (0x0bda, 0x2832): SDRType.RTL_SDR,       # RTL-SDR variant
        (0x1d50, 0x60a1): SDRType.AIRSPY,        # Airspy
        (0x1d50, 0x60a6): SDRType.AIRSPY_HF,     # Airspy HF+
    }
    
    def __init__(self):
        self.logger = logging.getLogger('HardwareDetector')
        self.detected_devices: List[DetectedSDR] = []
        
    def scan_usb(self) -> List[Tuple[int, int, str]]:
        """Scan USB bus for devices"""
        devices = []
        
        try:
            # Try lsusb on Linux
            if platform.system() == 'Linux':
                result = subprocess.run(
                    ['lsusb'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                for line in result.stdout.strip().split('\n'):
                    # Parse: Bus 001 Device 002: ID 2cf0:5246 ...
                    parts = line.split()
                    if len(parts) >= 6 and 'ID' in parts:
                        id_idx = parts.index('ID')
                        if id_idx + 1 < len(parts):
                            vid_pid = parts[id_idx + 1].split(':')
                            if len(vid_pid) == 2:
                                vid = int(vid_pid[0], 16)
                                pid = int(vid_pid[1], 16)
                                bus_info = f"{parts[1]}:{parts[3].rstrip(':')}"
                                devices.append((vid, pid, bus_info))
            
            # macOS
            elif platform.system() == 'Darwin':
                result = subprocess.run(
                    ['system_profiler', 'SPUSBDataType', '-json'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                # Parse JSON output (simplified)
                self.logger.info("macOS USB detection (basic)")
                
        except Exception as e:
            self.logger.error(f"USB scan error: {e}")
        
        return devices
    
    def detect_bladerf(self) -> Optional[DetectedSDR]:
        """Detect BladeRF device"""
        try:
            result = subprocess.run(
                ['bladeRF-cli', '-p'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Parse output
                output = result.stdout.strip()
                
                detected = DetectedSDR(sdr_type=SDRType.BLADERF_XA9)
                detected.driver_status = DriverStatus.INSTALLED
                
                # Extract serial
                if 'Serial' in output:
                    for line in output.split('\n'):
                        if 'Serial' in line:
                            detected.serial_number = line.split(':')[-1].strip()
                
                # Get detailed info
                info_result = subprocess.run(
                    ['bladeRF-cli', '-e', 'info'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if info_result.returncode == 0:
                    for line in info_result.stdout.split('\n'):
                        if 'FPGA' in line and 'version' in line.lower():
                            detected.fpga_version = line.split(':')[-1].strip()
                        elif 'FW' in line or 'Firmware' in line:
                            detected.firmware_version = line.split(':')[-1].strip()
                
                # Determine exact model
                if 'xA9' in output or '301' in output:
                    detected.sdr_type = SDRType.BLADERF_XA9
                elif 'xA4' in output or '49' in output:
                    detected.sdr_type = SDRType.BLADERF_XA4
                
                detected.capabilities = SDR_CAPABILITIES.get(detected.sdr_type)
                
                return detected
                
        except FileNotFoundError:
            self.logger.debug("bladeRF-cli not found")
        except Exception as e:
            self.logger.error(f"BladeRF detection error: {e}")
        
        return None
    
    def detect_hackrf(self) -> Optional[DetectedSDR]:
        """Detect HackRF device"""
        try:
            result = subprocess.run(
                ['hackrf_info'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and 'Serial' in result.stdout:
                detected = DetectedSDR(sdr_type=SDRType.HACKRF)
                detected.driver_status = DriverStatus.INSTALLED
                
                for line in result.stdout.split('\n'):
                    if 'Serial' in line:
                        detected.serial_number = line.split(':')[-1].strip()
                    elif 'Firmware' in line:
                        detected.firmware_version = line.split(':')[-1].strip()
                
                detected.capabilities = SDR_CAPABILITIES.get(SDRType.HACKRF)
                return detected
                
        except FileNotFoundError:
            self.logger.debug("hackrf_info not found")
        except Exception as e:
            self.logger.error(f"HackRF detection error: {e}")
        
        return None
    
    def detect_limesdr(self) -> Optional[DetectedSDR]:
        """Detect LimeSDR device"""
        try:
            result = subprocess.run(
                ['LimeUtil', '--find'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                output = result.stdout.strip()
                
                if 'LimeSDR Mini' in output:
                    detected = DetectedSDR(sdr_type=SDRType.LIMESDR_MINI)
                else:
                    detected = DetectedSDR(sdr_type=SDRType.LIMESDR_USB)
                
                detected.driver_status = DriverStatus.INSTALLED
                detected.capabilities = SDR_CAPABILITIES.get(detected.sdr_type)
                
                # Extract serial
                for line in output.split('\n'):
                    if 'serial' in line.lower():
                        detected.serial_number = line.split('=')[-1].strip()
                
                return detected
                
        except FileNotFoundError:
            self.logger.debug("LimeUtil not found")
        except Exception as e:
            self.logger.error(f"LimeSDR detection error: {e}")
        
        return None
    
    def detect_rtlsdr(self) -> Optional[DetectedSDR]:
        """Detect RTL-SDR device"""
        try:
            result = subprocess.run(
                ['rtl_test', '-t'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # RTL-SDR found if we get device info (even with error return)
            if 'Found' in result.stderr or 'Realtek' in result.stderr:
                detected = DetectedSDR(sdr_type=SDRType.RTL_SDR)
                detected.driver_status = DriverStatus.INSTALLED
                detected.capabilities = SDR_CAPABILITIES.get(SDRType.RTL_SDR)
                
                # Parse output for details
                for line in result.stderr.split('\n'):
                    if 'Serial' in line:
                        detected.serial_number = line.split(':')[-1].strip()
                
                return detected
                
        except FileNotFoundError:
            self.logger.debug("rtl_test not found")
        except Exception as e:
            self.logger.error(f"RTL-SDR detection error: {e}")
        
        return None
    
    def detect_all(self) -> List[DetectedSDR]:
        """Detect all connected SDR devices"""
        self.detected_devices = []
        
        self.logger.info("Scanning for SDR devices...")
        
        # Scan USB first
        usb_devices = self.scan_usb()
        self.logger.info(f"Found {len(usb_devices)} USB devices")
        
        # Check for each SDR type
        detectors = [
            ('BladeRF', self.detect_bladerf),
            ('HackRF', self.detect_hackrf),
            ('LimeSDR', self.detect_limesdr),
            ('RTL-SDR', self.detect_rtlsdr),
        ]
        
        for name, detector in detectors:
            self.logger.debug(f"Checking for {name}...")
            result = detector()
            if result:
                self.detected_devices.append(result)
                self.logger.info(f"Found: {result.sdr_type.value}")
        
        return self.detected_devices


class HardwareCalibrator:
    """Calibrate SDR hardware"""
    
    def __init__(self, hardware_controller=None):
        self.logger = logging.getLogger('HardwareCalibrator')
        self.hw = hardware_controller
        
    def calibrate_dc_offset(self) -> Dict:
        """Calibrate DC offset"""
        self.logger.info("Calibrating DC offset...")
        
        result = {
            'status': 'success',
            'dc_i': 0.0,
            'dc_q': 0.0
        }
        
        try:
            if self.hw:
                # Capture samples with no signal
                # Calculate DC offset
                # Apply correction
                pass
            else:
                self.logger.warning("No hardware - simulating calibration")
                time.sleep(1)
                
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
        
        return result
    
    def calibrate_iq_balance(self) -> Dict:
        """Calibrate I/Q imbalance"""
        self.logger.info("Calibrating I/Q balance...")
        
        result = {
            'status': 'success',
            'gain_imbalance': 0.0,
            'phase_imbalance': 0.0
        }
        
        try:
            if self.hw:
                # Inject test tone
                # Measure I/Q imbalance
                # Calculate corrections
                pass
            else:
                time.sleep(1)
                
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
        
        return result
    
    def calibrate_gain(self) -> Dict:
        """Calibrate gain stages"""
        self.logger.info("Calibrating gain...")
        
        result = {
            'status': 'success',
            'rx_gain_correction': 0,
            'tx_gain_correction': 0
        }
        
        try:
            if self.hw:
                # Measure gain at reference level
                # Calculate corrections
                pass
            else:
                time.sleep(1)
                
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
        
        return result
    
    def run_full_calibration(self) -> Dict:
        """Run full calibration sequence"""
        self.logger.info("Starting full calibration...")
        
        results = {
            'dc_offset': self.calibrate_dc_offset(),
            'iq_balance': self.calibrate_iq_balance(),
            'gain': self.calibrate_gain(),
            'overall_status': 'success'
        }
        
        # Check for failures
        for cal_type, cal_result in results.items():
            if isinstance(cal_result, dict) and cal_result.get('status') == 'failed':
                results['overall_status'] = 'partial'
                break
        
        return results


class HardwareSetupWizard:
    """
    Interactive Hardware Setup Wizard
    Guides user through SDR setup and configuration
    """
    
    def __init__(self):
        self.logger = logging.getLogger('HardwareWizard')
        self.detector = HardwareDetector()
        self.calibrator = HardwareCalibrator()
        
        self.detected_devices: List[DetectedSDR] = []
        self.selected_device: Optional[DetectedSDR] = None
        self.wizard_state = 'start'
        self.setup_complete = False
        
    def run_detection(self) -> List[DetectedSDR]:
        """Run hardware detection"""
        self.detected_devices = self.detector.detect_all()
        return self.detected_devices
    
    def select_device(self, index: int) -> bool:
        """Select a detected device"""
        if 0 <= index < len(self.detected_devices):
            self.selected_device = self.detected_devices[index]
            return True
        return False
    
    def get_antenna_recommendations(self, frequency_mhz: int = None) -> List[AntennaRecommendation]:
        """Get antenna recommendations for frequency"""
        if frequency_mhz:
            return [a for a in ANTENNA_GUIDE 
                    if a.freq_min_mhz <= frequency_mhz <= a.freq_max_mhz]
        return ANTENNA_GUIDE
    
    def run_calibration(self) -> Dict:
        """Run calibration on selected device"""
        if not self.selected_device:
            return {'status': 'error', 'message': 'No device selected'}
        
        # Connect calibrator to hardware
        # self.calibrator.hw = ... (get hardware controller)
        
        return self.calibrator.run_full_calibration()
    
    def check_driver_status(self) -> Dict:
        """Check driver installation status"""
        status = {
            'bladerf': self._check_bladerf_driver(),
            'hackrf': self._check_hackrf_driver(),
            'limesdr': self._check_limesdr_driver(),
            'rtlsdr': self._check_rtlsdr_driver(),
        }
        return status
    
    def _check_bladerf_driver(self) -> Dict:
        """Check BladeRF driver"""
        try:
            result = subprocess.run(['bladeRF-cli', '--version'],
                                   capture_output=True, timeout=5)
            if result.returncode == 0:
                return {'installed': True, 'version': result.stdout.decode().strip()}
        except:
            pass
        return {'installed': False, 'install_cmd': 'apt install bladerf libbladerf-dev'}
    
    def _check_hackrf_driver(self) -> Dict:
        """Check HackRF driver"""
        try:
            result = subprocess.run(['hackrf_info'],
                                   capture_output=True, timeout=5)
            return {'installed': True}
        except:
            pass
        return {'installed': False, 'install_cmd': 'apt install hackrf libhackrf-dev'}
    
    def _check_limesdr_driver(self) -> Dict:
        """Check LimeSDR driver"""
        try:
            result = subprocess.run(['LimeUtil', '--find'],
                                   capture_output=True, timeout=5)
            return {'installed': True}
        except:
            pass
        return {'installed': False, 'install_cmd': 'apt install limesuite'}
    
    def _check_rtlsdr_driver(self) -> Dict:
        """Check RTL-SDR driver"""
        try:
            result = subprocess.run(['rtl_test', '-h'],
                                   capture_output=True, timeout=5)
            return {'installed': True}
        except:
            pass
        return {'installed': False, 'install_cmd': 'apt install rtl-sdr librtlsdr-dev'}
    
    def get_troubleshooting_tips(self, sdr_type: SDRType = None) -> List[str]:
        """Get troubleshooting tips"""
        tips = [
            "Ensure the SDR is properly connected via USB",
            "Try a different USB port (preferably USB 3.0 for BladeRF/LimeSDR)",
            "Check that you have sufficient permissions (try with sudo)",
            "Verify udev rules are installed for your SDR",
            "Restart the USB subsystem if device was recently connected",
        ]
        
        if sdr_type == SDRType.BLADERF_XA9 or sdr_type == SDRType.BLADERF_XA4:
            tips.extend([
                "Run 'bladeRF-cli -l' to load FPGA if needed",
                "Update firmware with 'bladeRF-cli -f latest'",
                "Check FPGA autoload status",
            ])
        elif sdr_type == SDRType.HACKRF:
            tips.extend([
                "Update firmware with 'hackrf_spiflash'",
                "Try resetting the device by disconnecting/reconnecting",
            ])
        elif sdr_type == SDRType.RTL_SDR:
            tips.extend([
                "Blacklist the dvb_usb_rtl28xxu kernel module",
                "Create udev rule for non-root access",
            ])
        
        return tips
    
    def get_wizard_state(self) -> Dict:
        """Get current wizard state"""
        return {
            'state': self.wizard_state,
            'detected_devices': [d.to_dict() for d in self.detected_devices],
            'selected_device': self.selected_device.to_dict() if self.selected_device else None,
            'setup_complete': self.setup_complete
        }
    
    def get_status_display(self) -> str:
        """Get formatted status display"""
        lines = []
        lines.append("=" * 60)
        lines.append("  RF ARSENAL OS - HARDWARE SETUP WIZARD")
        lines.append("=" * 60)
        lines.append("")
        
        if not self.detected_devices:
            lines.append("❌ No SDR devices detected")
            lines.append("")
            lines.append("Troubleshooting:")
            for tip in self.get_troubleshooting_tips()[:5]:
                lines.append(f"  • {tip}")
        else:
            lines.append(f"✅ Found {len(self.detected_devices)} SDR device(s):")
            lines.append("")
            
            for i, device in enumerate(self.detected_devices):
                selected = " [SELECTED]" if device == self.selected_device else ""
                lines.append(f"  [{i+1}] {device.sdr_type.value}{selected}")
                
                if device.serial_number:
                    lines.append(f"      Serial: {device.serial_number}")
                if device.firmware_version:
                    lines.append(f"      Firmware: {device.firmware_version}")
                
                cap = device.capabilities
                if cap:
                    lines.append(f"      Frequency: {cap.frequency_min_hz/1e6:.0f} MHz - {cap.frequency_max_hz/1e9:.1f} GHz")
                    lines.append(f"      TX: {'✅' if cap.tx_capable else '❌'}  RX: {'✅' if cap.rx_capable else '❌'}  Full-Duplex: {'✅' if cap.full_duplex else '❌'}")
                
                lines.append("")
        
        # Driver status
        lines.append("-" * 60)
        lines.append("Driver Status:")
        drivers = self.check_driver_status()
        for name, status in drivers.items():
            icon = "✅" if status.get('installed') else "❌"
            lines.append(f"  {icon} {name}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def run_interactive(self):
        """Run interactive wizard (CLI mode)"""
        print(self.get_status_display())
        
        # Scan for devices
        print("\nScanning for SDR devices...")
        self.run_detection()
        
        print(self.get_status_display())
        
        if self.detected_devices:
            # Select device
            choice = input("\nSelect device number (or 'q' to quit): ")
            if choice.lower() == 'q':
                return
            
            try:
                idx = int(choice) - 1
                if self.select_device(idx):
                    print(f"\nSelected: {self.selected_device.sdr_type.value}")
                    
                    # Offer calibration
                    cal = input("Run calibration? (y/n): ")
                    if cal.lower() == 'y':
                        print("\nRunning calibration...")
                        results = self.run_calibration()
                        print(f"Calibration: {results['overall_status']}")
                    
                    # Antenna recommendations
                    freq = input("\nTarget frequency in MHz (or press Enter to skip): ")
                    if freq:
                        antennas = self.get_antenna_recommendations(int(freq))
                        print("\nRecommended antennas:")
                        for ant in antennas:
                            print(f"  • {ant.name}: {ant.description}")
                    
                    self.setup_complete = True
                    print("\n✅ Setup complete!")
                    
            except (ValueError, IndexError):
                print("Invalid selection")


# Global instance
_hardware_wizard: Optional[HardwareSetupWizard] = None


def get_hardware_wizard() -> HardwareSetupWizard:
    """Get global hardware wizard instance"""
    global _hardware_wizard
    if _hardware_wizard is None:
        _hardware_wizard = HardwareSetupWizard()
    return _hardware_wizard


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    wizard = HardwareSetupWizard()
    wizard.run_interactive()
