#!/usr/bin/env python3
"""
RF Arsenal OS - SDR Hardware Abstraction Layer
Multi-SDR support for maximum hardware flexibility
"""

import logging
import subprocess
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class SDRType(Enum):
    """Supported SDR hardware types"""
    BLADERF = "bladerf"
    HACKRF = "hackrf"
    LIMESDR = "limesdr"
    USRP = "usrp"
    RTLSDR = "rtlsdr"
    PLUTO = "pluto"
    UNKNOWN = "unknown"


class SDRCapability(Enum):
    """SDR capability flags"""
    TX = "transmit"
    RX = "receive"
    FULL_DUPLEX = "full_duplex"
    MIMO = "mimo"
    GPS = "gps"
    EXTERNAL_CLOCK = "external_clock"


@dataclass
class SDRCapabilities:
    """SDR hardware capabilities"""
    sdr_type: SDRType
    model: str
    
    # Frequency capabilities
    freq_min: int  # Hz
    freq_max: int  # Hz
    
    # Sample rate capabilities
    sample_rate_min: int  # Hz
    sample_rate_max: int  # Hz
    
    # Bandwidth capabilities
    bandwidth_min: int  # Hz
    bandwidth_max: int  # Hz
    
    # TX capabilities
    tx_capable: bool
    tx_power_min: int  # dBm
    tx_power_max: int  # dBm
    
    # RX capabilities
    rx_capable: bool
    rx_gain_min: int  # dB
    rx_gain_max: int  # dB
    
    # Advanced features
    channels: int  # Number of RX/TX channels
    full_duplex: bool
    mimo_capable: bool
    gps_capable: bool
    
    # Cost and availability
    typical_cost_usd: int
    availability: str  # "common", "moderate", "rare"
    
    # Use case recommendations
    recommended_for: List[str]
    limitations: List[str]


class SDRHardwareAbstraction:
    """
    Hardware abstraction layer for multiple SDR platforms
    
    Provides unified interface for:
    - Hardware detection and enumeration
    - Capability querying
    - Configuration
    - TX/RX operations
    - Error handling and fallbacks
    """
    
    def __init__(self):
        self.detected_sdrs: List[Tuple[SDRType, Any]] = []
        self.active_sdr = None
        self.active_sdr_type = SDRType.UNKNOWN
        
        # Hardware capabilities database
        self.capabilities_db = self._initialize_capabilities_db()
        
        logger.info("SDR Hardware Abstraction Layer initialized")
        
    def _initialize_capabilities_db(self) -> Dict[SDRType, SDRCapabilities]:
        """Initialize SDR capabilities database"""
        return {
            SDRType.BLADERF: SDRCapabilities(
                sdr_type=SDRType.BLADERF,
                model="BladeRF 2.0 micro xA9",
                freq_min=47_000_000,      # 47 MHz
                freq_max=6_000_000_000,   # 6 GHz
                sample_rate_min=520_833,  # 520.833 kHz
                sample_rate_max=61_440_000,  # 61.44 MHz
                bandwidth_min=200_000,    # 200 kHz
                bandwidth_max=56_000_000, # 56 MHz
                tx_capable=True,
                tx_power_min=-35,         # dBm
                tx_power_max=10,          # dBm
                rx_capable=True,
                rx_gain_min=-15,          # dB
                rx_gain_max=60,           # dB
                channels=2,
                full_duplex=True,
                mimo_capable=True,
                gps_capable=False,
                typical_cost_usd=450,
                availability="common",
                recommended_for=[
                    "Cellular (2G/3G/4G/5G)",
                    "WiFi",
                    "GPS spoofing",
                    "Professional pentesting",
                    "Full-duplex operations",
                    "MIMO applications"
                ],
                limitations=[
                    "Higher cost",
                    "USB 3.0 required for max performance"
                ]
            ),
            
            SDRType.HACKRF: SDRCapabilities(
                sdr_type=SDRType.HACKRF,
                model="HackRF One",
                freq_min=1_000_000,       # 1 MHz
                freq_max=6_000_000_000,   # 6 GHz
                sample_rate_min=1_000_000,  # 1 MHz
                sample_rate_max=20_000_000, # 20 MHz
                bandwidth_min=1_750_000,  # 1.75 MHz
                bandwidth_max=20_000_000, # 20 MHz
                tx_capable=True,
                tx_power_min=-20,         # dBm
                tx_power_max=10,          # dBm
                rx_capable=True,
                rx_gain_min=0,            # dB
                rx_gain_max=62,           # dB (LNA + VGA)
                channels=1,
                full_duplex=False,        # Half-duplex only
                mimo_capable=False,
                gps_capable=False,
                typical_cost_usd=300,
                availability="common",
                recommended_for=[
                    "Budget-conscious pentesting",
                    "Spectrum analysis",
                    "WiFi (2.4/5 GHz)",
                    "GPS spoofing",
                    "Drone detection",
                    "Learning/education"
                ],
                limitations=[
                    "Half-duplex only (no simultaneous TX/RX)",
                    "8-bit ADC (lower dynamic range)",
                    "Lower sample rate vs BladeRF",
                    "No MIMO"
                ]
            ),
            
            SDRType.LIMESDR: SDRCapabilities(
                sdr_type=SDRType.LIMESDR,
                model="LimeSDR Mini",
                freq_min=10_000_000,      # 10 MHz
                freq_max=3_500_000_000,   # 3.5 GHz
                sample_rate_min=100_000,  # 100 kHz
                sample_rate_max=30_720_000, # 30.72 MHz
                bandwidth_min=5_000_000,  # 5 MHz
                bandwidth_max=40_000_000, # 40 MHz
                tx_capable=True,
                tx_power_min=-20,         # dBm
                tx_power_max=10,          # dBm
                rx_capable=True,
                rx_gain_min=0,            # dB
                rx_gain_max=70,           # dB
                channels=1,               # Mini has 1x1, full LimeSDR has 2x2
                full_duplex=True,
                mimo_capable=False,       # Mini is 1x1, full LimeSDR is 2x2
                gps_capable=False,
                typical_cost_usd=159,
                availability="common",
                recommended_for=[
                    "Budget full-duplex",
                    "Cellular (2G/3G/4G)",
                    "WiFi",
                    "Open-source projects",
                    "LTE base stations"
                ],
                limitations=[
                    "Lower frequency range (no 5G mmWave)",
                    "USB 3.0 required",
                    "Limited community support vs HackRF"
                ]
            ),
            
            SDRType.USRP: SDRCapabilities(
                sdr_type=SDRType.USRP,
                model="USRP B200/B210",
                freq_min=70_000_000,      # 70 MHz
                freq_max=6_000_000_000,   # 6 GHz
                sample_rate_min=200_000,  # 200 kHz
                sample_rate_max=61_440_000, # 61.44 MHz
                bandwidth_min=200_000,    # 200 kHz
                bandwidth_max=56_000_000, # 56 MHz
                tx_capable=True,
                tx_power_min=-20,         # dBm
                tx_power_max=20,          # dBm
                rx_capable=True,
                rx_gain_min=0,            # dB
                rx_gain_max=76,           # dB
                channels=2,               # B210 has 2x2 MIMO
                full_duplex=True,
                mimo_capable=True,
                gps_capable=True,         # Optional GPSDO
                typical_cost_usd=700,
                availability="moderate",
                recommended_for=[
                    "Professional research",
                    "High-performance applications",
                    "Cellular (2G/3G/4G/5G)",
                    "MIMO testing",
                    "Precise frequency control (GPSDO)"
                ],
                limitations=[
                    "High cost",
                    "Overkill for basic testing",
                    "Requires USB 3.0"
                ]
            ),
            
            SDRType.RTLSDR: SDRCapabilities(
                sdr_type=SDRType.RTLSDR,
                model="RTL-SDR",
                freq_min=24_000_000,      # 24 MHz
                freq_max=1_766_000_000,   # 1.766 GHz
                sample_rate_min=225_000,  # 225 kHz
                sample_rate_max=3_200_000, # 3.2 MHz
                bandwidth_min=225_000,    # 225 kHz
                bandwidth_max=3_200_000,  # 3.2 MHz
                tx_capable=False,         # RX ONLY
                tx_power_min=0,
                tx_power_max=0,
                rx_capable=True,
                rx_gain_min=0,            # dB
                rx_gain_max=50,           # dB
                channels=1,
                full_duplex=False,
                mimo_capable=False,
                gps_capable=False,
                typical_cost_usd=25,
                availability="common",
                recommended_for=[
                    "Spectrum monitoring only",
                    "Passive SIGINT",
                    "Learning SDR basics",
                    "Budget spectrum analyzer",
                    "ADS-B tracking",
                    "IMSI detection (passive)"
                ],
                limitations=[
                    "RX ONLY - No transmission",
                    "Limited bandwidth (3.2 MHz max)",
                    "8-bit ADC",
                    "Cannot do active attacks"
                ]
            ),
            
            SDRType.PLUTO: SDRCapabilities(
                sdr_type=SDRType.PLUTO,
                model="ADALM-PLUTO",
                freq_min=325_000_000,     # 325 MHz (hackable to 70 MHz)
                freq_max=3_800_000_000,   # 3.8 GHz (hackable to 6 GHz)
                sample_rate_min=65_105,   # 65.105 kHz
                sample_rate_max=61_440_000, # 61.44 MHz (hackable)
                bandwidth_min=200_000,    # 200 kHz
                bandwidth_max=20_000_000, # 20 MHz (56 MHz hackable)
                tx_capable=True,
                tx_power_min=-20,         # dBm
                tx_power_max=7,           # dBm
                rx_capable=True,
                rx_gain_min=-3,           # dB
                rx_gain_max=71,           # dB
                channels=1,
                full_duplex=True,
                mimo_capable=False,
                gps_capable=False,
                typical_cost_usd=150,
                availability="common",
                recommended_for=[
                    "Budget full-duplex",
                    "Portable operations",
                    "WiFi testing",
                    "Cellular (with hacks)",
                    "Education"
                ],
                limitations=[
                    "Requires firmware hacks for full capability",
                    "Limited TX power",
                    "Single channel only"
                ]
            )
        }
    
    def auto_detect_sdr(self) -> List[Tuple[SDRType, Any]]:
        """
        Auto-detect available SDR hardware
        
        Returns:
            List of (SDRType, device_object) tuples
        """
        self.detected_sdrs = []
        
        # Detect BladeRF
        try:
            bladerf_devices = self._detect_bladerf()
            self.detected_sdrs.extend([(SDRType.BLADERF, dev) for dev in bladerf_devices])
        except Exception as e:
            logger.debug(f"BladeRF detection failed: {e}")
        
        # Detect HackRF
        try:
            hackrf_devices = self._detect_hackrf()
            self.detected_sdrs.extend([(SDRType.HACKRF, dev) for dev in hackrf_devices])
        except Exception as e:
            logger.debug(f"HackRF detection failed: {e}")
        
        # Detect LimeSDR
        try:
            lime_devices = self._detect_limesdr()
            self.detected_sdrs.extend([(SDRType.LIMESDR, dev) for dev in lime_devices])
        except Exception as e:
            logger.debug(f"LimeSDR detection failed: {e}")
        
        # Detect USRP
        try:
            usrp_devices = self._detect_usrp()
            self.detected_sdrs.extend([(SDRType.USRP, dev) for dev in usrp_devices])
        except Exception as e:
            logger.debug(f"USRP detection failed: {e}")
        
        # Detect RTL-SDR
        try:
            rtlsdr_devices = self._detect_rtlsdr()
            self.detected_sdrs.extend([(SDRType.RTLSDR, dev) for dev in rtlsdr_devices])
        except Exception as e:
            logger.debug(f"RTL-SDR detection failed: {e}")
        
        # Detect PlutoSDR
        try:
            pluto_devices = self._detect_pluto()
            self.detected_sdrs.extend([(SDRType.PLUTO, dev) for dev in pluto_devices])
        except Exception as e:
            logger.debug(f"PlutoSDR detection failed: {e}")
        
        if self.detected_sdrs:
            logger.info(f"Detected {len(self.detected_sdrs)} SDR device(s):")
            for sdr_type, device in self.detected_sdrs:
                logger.info(f"  - {sdr_type.value}: {device}")
        else:
            logger.warning("No SDR hardware detected")
        
        return self.detected_sdrs
    
    def _detect_bladerf(self) -> List[str]:
        """Detect BladeRF devices"""
        devices = []
        try:
            import bladerf
            device_list = bladerf.get_device_list()
            for dev_info in device_list:
                devices.append(dev_info.serial)
            logger.info(f"Found {len(devices)} BladeRF device(s)")
        except ImportError:
            logger.debug("BladeRF library not installed")
        except Exception as e:
            logger.debug(f"BladeRF detection error: {e}")
        return devices
    
    def _detect_hackrf(self) -> List[str]:
        """Detect HackRF devices"""
        devices = []
        try:
            result = subprocess.run(
                ['hackrf_info'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and 'Serial number' in result.stdout:
                # Parse serial number from output
                for line in result.stdout.split('\n'):
                    if 'Serial number' in line:
                        serial = line.split(':')[1].strip()
                        devices.append(serial)
            logger.info(f"Found {len(devices)} HackRF device(s)")
        except FileNotFoundError:
            logger.debug("HackRF tools not installed")
        except Exception as e:
            logger.debug(f"HackRF detection error: {e}")
        return devices
    
    def _detect_limesdr(self) -> List[str]:
        """Detect LimeSDR devices"""
        devices = []
        try:
            result = subprocess.run(
                ['LimeUtil', '--find'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse device list from output
                for line in result.stdout.split('\n'):
                    if 'LimeSDR' in line or 'serial=' in line:
                        devices.append(line.strip())
            logger.info(f"Found {len(devices)} LimeSDR device(s)")
        except FileNotFoundError:
            logger.debug("LimeSDR tools not installed")
        except Exception as e:
            logger.debug(f"LimeSDR detection error: {e}")
        return devices
    
    def _detect_usrp(self) -> List[str]:
        """Detect USRP devices"""
        devices = []
        try:
            result = subprocess.run(
                ['uhd_find_devices'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse device list from output
                for line in result.stdout.split('\n'):
                    if 'serial:' in line:
                        serial = line.split('serial:')[1].strip()
                        devices.append(serial)
            logger.info(f"Found {len(devices)} USRP device(s)")
        except FileNotFoundError:
            logger.debug("UHD tools not installed")
        except Exception as e:
            logger.debug(f"USRP detection error: {e}")
        return devices
    
    def _detect_rtlsdr(self) -> List[int]:
        """Detect RTL-SDR devices"""
        devices = []
        try:
            result = subprocess.run(
                ['rtl_test', '-t'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse device count from output
                for line in result.stdout.split('\n'):
                    if 'Found' in line and 'device' in line:
                        import re
                        match = re.search(r'Found (\d+)', line)
                        if match:
                            count = int(match.group(1))
                            devices = list(range(count))
            logger.info(f"Found {len(devices)} RTL-SDR device(s)")
        except FileNotFoundError:
            logger.debug("RTL-SDR tools not installed")
        except Exception as e:
            logger.debug(f"RTL-SDR detection error: {e}")
        return devices
    
    def _detect_pluto(self) -> List[str]:
        """Detect ADALM-PLUTO devices"""
        devices = []
        try:
            # Pluto appears as USB device or network device
            result = subprocess.run(
                ['iio_info'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and 'PlutoSDR' in result.stdout:
                # Parse device URIs
                for line in result.stdout.split('\n'):
                    if 'uri:' in line or 'usb:' in line or 'ip:' in line:
                        devices.append(line.strip())
            logger.info(f"Found {len(devices)} PlutoSDR device(s)")
        except FileNotFoundError:
            logger.debug("libiio tools not installed")
        except Exception as e:
            logger.debug(f"PlutoSDR detection error: {e}")
        return devices
    
    def get_capabilities(self, sdr_type: SDRType) -> Optional[SDRCapabilities]:
        """
        Get capabilities for specific SDR type
        
        Args:
            sdr_type: SDR type to query
            
        Returns:
            SDRCapabilities object or None if unknown
        """
        return self.capabilities_db.get(sdr_type)
    
    def select_best_sdr(self, requirements: Dict[str, Any]) -> Optional[Tuple[SDRType, Any]]:
        """
        Select best available SDR for given requirements
        
        Args:
            requirements: Dict with keys:
                - 'frequency': Required frequency in Hz
                - 'bandwidth': Required bandwidth in Hz
                - 'tx_required': True if TX needed
                - 'full_duplex': True if simultaneous TX/RX needed
                - 'mimo': True if MIMO required
                
        Returns:
            (SDRType, device) tuple or None if no suitable SDR found
        """
        if not self.detected_sdrs:
            logger.warning("No SDRs detected, run auto_detect_sdr() first")
            return None
        
        freq = requirements.get('frequency', 0)
        bandwidth = requirements.get('bandwidth', 0)
        tx_required = requirements.get('tx_required', False)
        full_duplex = requirements.get('full_duplex', False)
        mimo = requirements.get('mimo', False)
        
        suitable_sdrs = []
        
        for sdr_type, device in self.detected_sdrs:
            caps = self.get_capabilities(sdr_type)
            if not caps:
                continue
            
            # Check frequency range
            if freq < caps.freq_min or freq > caps.freq_max:
                continue
            
            # Check bandwidth
            if bandwidth > caps.bandwidth_max:
                continue
            
            # Check TX capability
            if tx_required and not caps.tx_capable:
                continue
            
            # Check full-duplex
            if full_duplex and not caps.full_duplex:
                continue
            
            # Check MIMO
            if mimo and not caps.mimo_capable:
                continue
            
            suitable_sdrs.append((sdr_type, device, caps))
        
        if not suitable_sdrs:
            logger.warning("No SDR meets requirements")
            return None
        
        # Prioritize by capabilities and cost
        # Score: higher is better
        def score_sdr(sdr_tuple):
            sdr_type, device, caps = sdr_tuple
            score = 0
            
            # Prefer full-duplex
            if caps.full_duplex:
                score += 100
            
            # Prefer MIMO
            if caps.mimo_capable:
                score += 50
            
            # Prefer wider bandwidth
            score += caps.bandwidth_max / 1_000_000  # MHz
            
            # Prefer wider frequency range
            freq_range = caps.freq_max - caps.freq_min
            score += freq_range / 1_000_000_000  # GHz
            
            # Slight preference for lower cost
            score -= caps.typical_cost_usd / 100
            
            return score
        
        suitable_sdrs.sort(key=score_sdr, reverse=True)
        best_sdr_type, best_device, best_caps = suitable_sdrs[0]
        
        logger.info(f"Selected {best_sdr_type.value} ({best_caps.model}) as best match")
        return (best_sdr_type, best_device)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current SDR status
        
        Returns:
            Dictionary with status information
        """
        if not self.active_sdr:
            return {'active': False, 'error': 'No SDR initialized'}
        
        caps = self.get_capabilities(self.active_sdr_type)
        
        return {
            'active': True,
            'sdr_type': self.active_sdr_type.value,
            'model': caps.model if caps else 'Unknown',
            'tx_capable': caps.tx_capable if caps else False,
            'rx_capable': caps.rx_capable if caps else True,
            'full_duplex': caps.full_duplex if caps else False,
            'mimo_capable': caps.mimo_capable if caps else False,
            'detected_count': len(self.detected_sdrs)
        }
    
    def close(self):
        """Close active SDR connection"""
        if self.active_sdr:
            try:
                if self.active_sdr_type == SDRType.BLADERF:
                    self.active_sdr.close()
                elif self.active_sdr_type == SDRType.RTLSDR:
                    self.active_sdr.close()
                
                logger.info(f"{self.active_sdr_type.value} closed")
            except Exception as e:
                logger.error(f"Error closing SDR: {e}")
            finally:
                self.active_sdr = None
                self.active_sdr_type = SDRType.UNKNOWN


if __name__ == "__main__":
    # Test hardware abstraction
    logging.basicConfig(level=logging.INFO)
    
    print("RF Arsenal OS - SDR Hardware Abstraction Layer Test")
    print("=" * 60)
    
    sdr_hal = SDRHardwareAbstraction()
    
    # Auto-detect
    print("\n[+] Auto-detecting SDR hardware...")
    detected = sdr_hal.auto_detect_sdr()
    
    if not detected:
        print("[-] No SDR hardware detected")
        print("\nSupported SDRs:")
        for sdr_type in SDRType:
            if sdr_type == SDRType.UNKNOWN:
                continue
            caps = sdr_hal.get_capabilities(sdr_type)
            if caps:
                print(f"  - {caps.model} (${caps.typical_cost_usd})")
                print(f"    Frequency: {caps.freq_min/1e6:.0f} - {caps.freq_max/1e6:.0f} MHz")
                print(f"    TX: {'Yes' if caps.tx_capable else 'No'}, "
                      f"RX: {'Yes' if caps.rx_capable else 'No'}, "
                      f"Full-Duplex: {'Yes' if caps.full_duplex else 'No'}")
                print()
    else:
        print(f"\n[+] Detected {len(detected)} SDR device(s)")
        
        # Select best for 2.4 GHz WiFi
        requirements = {
            'frequency': 2_450_000_000,  # 2.45 GHz
            'bandwidth': 20_000_000,     # 20 MHz
            'tx_required': True,
            'full_duplex': False,
            'mimo': False
        }
        
        print("\n[+] Selecting best SDR for 2.4 GHz WiFi testing...")
        best = sdr_hal.select_best_sdr(requirements)
        
        if best:
            sdr_type, device = best
            print(f"[+] Selected: {sdr_type.value}")
            caps = sdr_hal.get_capabilities(sdr_type)
            print(f"    Model: {caps.model}")
            print(f"    Cost: ${caps.typical_cost_usd}")
            print(f"    Full-Duplex: {caps.full_duplex}")
            print(f"    Recommended for: {', '.join(caps.recommended_for[:3])}")
