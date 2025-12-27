"""
RF Arsenal OS - Common External Stack Utilities

Shared utilities, base classes, and helpers for external protocol stack integration.
Provides common functionality used by both srsRAN and OpenAirInterface controllers.
"""

import os
import sys
import time
import json
import logging
import threading
import subprocess
import shutil
import socket
import struct
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

__version__ = '1.0.0'


# ============================================================================
# Common Enums
# ============================================================================

class ComponentState(IntEnum):
    """Universal component operational state"""
    STOPPED = 0
    STARTING = 1
    RUNNING = 2
    STOPPING = 3
    ERROR = 4
    CONFIGURED = 5
    DEGRADED = 6


class NetworkGeneration(IntEnum):
    """Cellular network generation"""
    LTE_4G = 4
    NR_5G = 5


class DeploymentType(IntEnum):
    """Deployment type"""
    SIMULATION = 0
    HARDWARE = 1
    RF_CHAMBER = 2
    FIELD = 3


# ============================================================================
# Common Configuration Structures
# ============================================================================

@dataclass
class BaseRFConfig:
    """Base RF configuration used by all stacks"""
    # SDR Device
    device_type: str = "bladerf"
    device_serial: str = ""
    device_args: str = ""
    
    # TX/RX Gains
    tx_gain_db: float = 50.0
    rx_gain_db: float = 40.0
    
    # Antenna Configuration
    num_tx_antennas: int = 1
    num_rx_antennas: int = 1
    
    # Clock/Reference
    clock_source: str = "internal"
    time_source: str = "internal"
    sync_source: str = "internal"
    
    # Frequency (general)
    center_frequency_hz: float = 2680e6
    
    # Bandwidth
    sample_rate_hz: float = 30.72e6
    bandwidth_hz: float = 10e6
    
    def to_srsran_args(self) -> str:
        """Convert to srsRAN device arguments"""
        args = f"device_type={self.device_type}"
        if self.device_serial:
            args += f",serial={self.device_serial}"
        if self.device_args:
            args += f",{self.device_args}"
        return args
    
    def to_oai_args(self) -> str:
        """Convert to OAI device arguments"""
        if self.device_type == "bladerf":
            args = "type=bladerf"
        elif self.device_type == "uhd":
            args = "type=b200"
        else:
            args = f"type={self.device_type}"
        
        if self.device_serial:
            args += f",serial={self.device_serial}"
        if self.device_args:
            args += f",{self.device_args}"
        return args


@dataclass
class BaseCellConfig:
    """Base cell configuration"""
    cell_id: int = 1
    physical_cell_id: int = 0
    tac: int = 1
    
    # PLMN
    mcc: str = "001"
    mnc: str = "01"
    
    @property
    def plmn(self) -> str:
        """Get PLMN string"""
        return f"{self.mcc}{self.mnc}"
    
    def validate(self) -> Tuple[bool, str]:
        """Validate cell configuration"""
        if len(self.mcc) != 3 or not self.mcc.isdigit():
            return False, "MCC must be 3 digits"
        if len(self.mnc) not in (2, 3) or not self.mnc.isdigit():
            return False, "MNC must be 2 or 3 digits"
        if self.cell_id < 0:
            return False, "Cell ID must be non-negative"
        if self.tac < 0 or self.tac > 0xFFFFFF:
            return False, "TAC out of range"
        return True, "OK"


@dataclass
class UECredentials:
    """UE authentication credentials"""
    imsi: str = "001010123456789"
    key: str = "00112233445566778899aabbccddeeff"
    opc: str = "63bfa50ee6523365ff14c1f45f88737d"
    amf: str = "8000"
    sqn: str = "000000000000"
    
    def validate(self) -> Tuple[bool, str]:
        """Validate credentials"""
        if len(self.imsi) != 15 or not self.imsi.isdigit():
            return False, "IMSI must be 15 digits"
        if len(self.key) != 32:
            return False, "Key must be 32 hex characters"
        if len(self.opc) != 32:
            return False, "OPc must be 32 hex characters"
        return True, "OK"
    
    def to_milenage_params(self) -> Dict[str, str]:
        """Get Milenage parameters"""
        return {
            'imsi': self.imsi,
            'ki': self.key,
            'opc': self.opc,
            'amf': self.amf,
            'sqn': self.sqn
        }


# ============================================================================
# LTE Frequency Utilities
# ============================================================================

class LTEFrequencyUtils:
    """LTE frequency calculation utilities (3GPP TS 36.101)"""
    
    # Band definitions: band -> (dl_low_mhz, dl_high_mhz, ul_low_mhz, ul_high_mhz, earfcn_offset)
    BANDS = {
        1: (2110, 2170, 1920, 1980, 0),
        2: (1930, 1990, 1850, 1910, 600),
        3: (1805, 1880, 1710, 1785, 1200),
        4: (2110, 2155, 1710, 1755, 1950),
        5: (869, 894, 824, 849, 2400),
        7: (2620, 2690, 2500, 2570, 2750),
        8: (925, 960, 880, 915, 3450),
        12: (729, 746, 699, 716, 5010),
        13: (746, 756, 777, 787, 5180),
        14: (758, 768, 788, 798, 5280),
        17: (734, 746, 704, 716, 5730),
        18: (860, 875, 815, 830, 5850),
        19: (875, 890, 830, 845, 6000),
        20: (791, 821, 832, 862, 6150),
        25: (1930, 1995, 1850, 1915, 8040),
        26: (859, 894, 814, 849, 8690),
        28: (758, 803, 703, 748, 9210),
        38: (2570, 2620, 2570, 2620, 37750),
        39: (1880, 1920, 1880, 1920, 38250),
        40: (2300, 2400, 2300, 2400, 38650),
        41: (2496, 2690, 2496, 2690, 39650),
        42: (3400, 3600, 3400, 3600, 41590),
        43: (3600, 3800, 3600, 3800, 43590),
        66: (2110, 2200, 1710, 1780, 66436),
    }
    
    @classmethod
    def freq_to_earfcn(cls, freq_hz: float, band: int = None) -> int:
        """Convert frequency to EARFCN"""
        freq_mhz = freq_hz / 1e6
        
        if band:
            if band in cls.BANDS:
                dl_low, dl_high, _, _, offset = cls.BANDS[band]
                if dl_low <= freq_mhz <= dl_high:
                    return int(offset + (freq_mhz - dl_low) * 10)
        
        # Auto-detect band
        for b, (dl_low, dl_high, _, _, offset) in cls.BANDS.items():
            if dl_low <= freq_mhz <= dl_high:
                return int(offset + (freq_mhz - dl_low) * 10)
        
        # Default to band 7
        return 3350
    
    @classmethod
    def earfcn_to_freq(cls, earfcn: int) -> float:
        """Convert EARFCN to frequency in Hz"""
        for band, (dl_low, dl_high, _, _, offset) in cls.BANDS.items():
            earfcn_low = offset
            earfcn_high = offset + int((dl_high - dl_low) * 10)
            
            if earfcn_low <= earfcn <= earfcn_high:
                freq_mhz = dl_low + (earfcn - offset) / 10.0
                return freq_mhz * 1e6
        
        return 2680e6  # Default
    
    @classmethod
    def get_band_for_freq(cls, freq_hz: float) -> Optional[int]:
        """Get band number for frequency"""
        freq_mhz = freq_hz / 1e6
        
        for band, (dl_low, dl_high, _, _, _) in cls.BANDS.items():
            if dl_low <= freq_mhz <= dl_high:
                return band
        
        return None
    
    @classmethod
    def bandwidth_to_prbs(cls, bw_mhz: float) -> int:
        """Convert bandwidth to PRB count"""
        mapping = {
            1.4: 6, 3: 15, 5: 25, 10: 50, 15: 75, 20: 100
        }
        return mapping.get(bw_mhz, 50)


# ============================================================================
# NR Frequency Utilities
# ============================================================================

class NRFrequencyUtils:
    """5G NR frequency calculation utilities (3GPP TS 38.101)"""
    
    # FR1 Bands
    FR1_BANDS = {
        1: (2110, 2170, 1920, 1980),
        3: (1805, 1880, 1710, 1785),
        5: (869, 894, 824, 849),
        7: (2620, 2690, 2500, 2570),
        8: (925, 960, 880, 915),
        28: (758, 803, 703, 748),
        38: (2570, 2620, 2570, 2620),
        41: (2496, 2690, 2496, 2690),
        66: (2110, 2200, 1710, 1780),
        77: (3300, 4200, 3300, 4200),
        78: (3300, 3800, 3300, 3800),
        79: (4400, 5000, 4400, 5000),
    }
    
    # FR2 Bands (mmWave)
    FR2_BANDS = {
        257: (26500, 29500, 26500, 29500),
        258: (24250, 27500, 24250, 27500),
        260: (37000, 40000, 37000, 40000),
        261: (27500, 28350, 27500, 28350),
    }
    
    @classmethod
    def freq_to_nrarfcn(cls, freq_hz: float) -> int:
        """Convert frequency to NR-ARFCN"""
        freq_mhz = freq_hz / 1e6
        
        if freq_mhz < 3000:
            # Range 0-3000 MHz
            return int(freq_mhz / 0.005)
        elif freq_mhz < 24250:
            # Range 3000-24250 MHz
            return int(600000 + (freq_mhz - 3000) / 0.015)
        else:
            # FR2 (24250+ MHz)
            return int(2016667 + (freq_mhz - 24250.08) / 0.060)
    
    @classmethod
    def nrarfcn_to_freq(cls, nrarfcn: int) -> float:
        """Convert NR-ARFCN to frequency in Hz"""
        if nrarfcn < 600000:
            freq_mhz = nrarfcn * 0.005
        elif nrarfcn < 2016667:
            freq_mhz = 3000 + (nrarfcn - 600000) * 0.015
        else:
            freq_mhz = 24250.08 + (nrarfcn - 2016667) * 0.060
        
        return freq_mhz * 1e6
    
    @classmethod
    def get_scs_for_band(cls, band: int) -> int:
        """Get recommended subcarrier spacing for band"""
        if band in cls.FR2_BANDS:
            return 120  # FR2 typically uses 120 kHz SCS
        elif band in (77, 78, 79):
            return 30   # n77/n78/n79 use 30 kHz SCS
        else:
            return 15   # Default FR1
    
    @classmethod
    def bandwidth_to_prbs(cls, bw_mhz: float, scs_khz: int = 30) -> int:
        """Convert bandwidth to PRB count for given SCS"""
        # Based on 3GPP TS 38.101-1 Table 5.3.2-1
        if scs_khz == 15:
            mapping = {5: 25, 10: 52, 15: 79, 20: 106, 25: 133, 30: 160, 40: 216, 50: 270}
        elif scs_khz == 30:
            mapping = {5: 11, 10: 24, 15: 38, 20: 51, 25: 65, 30: 78, 40: 106, 50: 133, 60: 162, 80: 217, 100: 273}
        elif scs_khz == 60:
            mapping = {10: 11, 15: 18, 20: 24, 25: 31, 30: 38, 40: 51, 50: 65, 60: 79, 80: 107, 100: 135}
        else:
            return 106
        
        return mapping.get(int(bw_mhz), 106)


# ============================================================================
# Process Management Base
# ============================================================================

class BaseProcessManager(ABC):
    """Base class for process management"""
    
    def __init__(self, name: str):
        self.name = name
        self._process: Optional[subprocess.Popen] = None
        self._state = ComponentState.STOPPED
        self._lock = threading.Lock()
        
        self._output_buffer: List[str] = []
        self._max_output_lines = 2000
        
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        
        self._state_callbacks: List[Callable] = []
        self._output_callbacks: List[Callable] = []
        
        self._start_time: Optional[float] = None
        self._restart_count = 0
    
    @property
    def state(self) -> ComponentState:
        return self._state
    
    @property
    def is_running(self) -> bool:
        return self._state == ComponentState.RUNNING
    
    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process else None
    
    @property
    def uptime(self) -> float:
        if self._start_time and self.is_running:
            return time.time() - self._start_time
        return 0.0
    
    def add_state_callback(self, callback: Callable[[ComponentState], None]):
        """Add state change callback"""
        self._state_callbacks.append(callback)
    
    def add_output_callback(self, callback: Callable[[str], None]):
        """Add output callback"""
        self._output_callbacks.append(callback)
    
    def _set_state(self, state: ComponentState):
        """Update state and notify callbacks"""
        old_state = self._state
        self._state = state
        
        if old_state != state:
            for callback in self._state_callbacks:
                try:
                    callback(state)
                except Exception as e:
                    logger.error(f"State callback error: {e}")
    
    def _read_output(self, pipe, stream_name: str):
        """Read output from process pipe"""
        try:
            for line in iter(pipe.readline, b''):
                decoded = line.decode('utf-8', errors='replace').rstrip()
                
                self._output_buffer.append(f"[{stream_name}] {decoded}")
                if len(self._output_buffer) > self._max_output_lines:
                    self._output_buffer.pop(0)
                
                for callback in self._output_callbacks:
                    try:
                        callback(decoded)
                    except:
                        pass
                
                self._parse_output(decoded)
                
        except Exception as e:
            logger.debug(f"Output reader ended: {e}")
    
    @abstractmethod
    def _parse_output(self, line: str):
        """Parse output for key events - to be implemented by subclasses"""
        pass
    
    def get_output(self, last_n: int = 100) -> List[str]:
        """Get recent output lines"""
        return self._output_buffer[-last_n:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get process status"""
        return {
            'name': self.name,
            'state': self._state.name,
            'pid': self.pid,
            'uptime': self.uptime,
            'restart_count': self._restart_count,
            'output_lines': len(self._output_buffer)
        }


# ============================================================================
# Network Utilities
# ============================================================================

class NetworkUtils:
    """Network-related utilities"""
    
    @staticmethod
    def check_port_available(port: int, host: str = '127.0.0.1') -> bool:
        """Check if port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex((host, port))
                return result != 0
        except:
            return False
    
    @staticmethod
    def find_available_port(start: int = 5000, end: int = 6000) -> Optional[int]:
        """Find an available port in range"""
        for port in range(start, end):
            if NetworkUtils.check_port_available(port):
                return port
        return None
    
    @staticmethod
    def get_local_ip() -> str:
        """Get local IP address"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(('8.8.8.8', 80))
                return s.getsockname()[0]
        except:
            return '127.0.0.1'
    
    @staticmethod
    def create_virtual_interface(name: str, ip: str, netmask: str = '255.255.255.0') -> bool:
        """Create virtual network interface"""
        try:
            subprocess.run(
                ['ip', 'link', 'add', name, 'type', 'dummy'],
                check=True, capture_output=True
            )
            subprocess.run(
                ['ip', 'addr', 'add', f'{ip}/{netmask}', 'dev', name],
                check=True, capture_output=True
            )
            subprocess.run(
                ['ip', 'link', 'set', name, 'up'],
                check=True, capture_output=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create interface {name}: {e}")
            return False
    
    @staticmethod
    def delete_virtual_interface(name: str) -> bool:
        """Delete virtual network interface"""
        try:
            subprocess.run(
                ['ip', 'link', 'delete', name],
                check=True, capture_output=True
            )
            return True
        except:
            return False


# ============================================================================
# Installation Checker
# ============================================================================

class InstallationChecker:
    """Check installation status of external software"""
    
    @staticmethod
    def check_srsran() -> Dict[str, Any]:
        """Check srsRAN installation"""
        result = {
            'installed': False,
            'version': None,
            'components': {},
            'path': None
        }
        
        components = {
            'srsenb': 'srsENB (LTE eNodeB)',
            'srsue': 'srsUE (LTE UE)',
            'srsepc': 'srsEPC (Core Network)',
            'srsgnb': 'srsGNB (5G gNodeB)'
        }
        
        for exe, desc in components.items():
            path = shutil.which(exe)
            if path:
                result['components'][exe] = {
                    'available': True,
                    'path': path,
                    'description': desc
                }
                result['installed'] = True
                if not result['path']:
                    result['path'] = os.path.dirname(path)
            else:
                result['components'][exe] = {
                    'available': False,
                    'path': None,
                    'description': desc
                }
        
        # Try to get version
        if result['installed']:
            try:
                proc = subprocess.run(
                    ['srsenb', '--version'],
                    capture_output=True, text=True, timeout=5
                )
                if proc.returncode == 0:
                    result['version'] = proc.stdout.strip()
            except:
                pass
        
        return result
    
    @staticmethod
    def check_oai() -> Dict[str, Any]:
        """Check OpenAirInterface installation"""
        result = {
            'installed': False,
            'version': None,
            'components': {},
            'path': None
        }
        
        components = {
            'nr-softmodem': 'NR gNodeB (5G)',
            'lte-softmodem': 'LTE eNodeB',
            'nr-uesoftmodem': 'NR UE (5G)',
            'lte-uesoftmodem': 'LTE UE'
        }
        
        # Check standard locations
        oai_paths = [
            '/opt/openairinterface5g',
            os.path.expanduser('~/openairinterface5g'),
            os.environ.get('OAI_HOME', '')
        ]
        
        for exe, desc in components.items():
            # Check PATH
            path = shutil.which(exe)
            
            # Check OAI build directories
            if not path:
                for oai_path in oai_paths:
                    if oai_path:
                        build_path = os.path.join(
                            oai_path, 'cmake_targets/ran_build/build', exe
                        )
                        if os.path.isfile(build_path) and os.access(build_path, os.X_OK):
                            path = build_path
                            break
            
            if path:
                result['components'][exe] = {
                    'available': True,
                    'path': path,
                    'description': desc
                }
                result['installed'] = True
                if not result['path']:
                    result['path'] = os.path.dirname(os.path.dirname(path))
            else:
                result['components'][exe] = {
                    'available': False,
                    'path': None,
                    'description': desc
                }
        
        return result
    
    @staticmethod
    def check_docker() -> Dict[str, Any]:
        """Check Docker installation for containerized deployment"""
        result = {
            'installed': False,
            'version': None,
            'compose_available': False
        }
        
        if shutil.which('docker'):
            result['installed'] = True
            try:
                proc = subprocess.run(
                    ['docker', '--version'],
                    capture_output=True, text=True, timeout=5
                )
                result['version'] = proc.stdout.strip()
            except:
                pass
        
        if shutil.which('docker-compose') or shutil.which('docker'):
            result['compose_available'] = True
        
        return result


# ============================================================================
# Logging Configuration
# ============================================================================

class StackLogger:
    """Centralized logging for external stacks"""
    
    def __init__(self, name: str, log_dir: str = '/var/log/rfarsenal'):
        self.name = name
        self.log_dir = log_dir
        
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger(f'rfarsenal.external.{name}')
        
        # File handler
        log_file = os.path.join(log_dir, f'{name}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    'ComponentState',
    'NetworkGeneration',
    'DeploymentType',
    
    # Configurations
    'BaseRFConfig',
    'BaseCellConfig',
    'UECredentials',
    
    # Frequency Utilities
    'LTEFrequencyUtils',
    'NRFrequencyUtils',
    
    # Process Management
    'BaseProcessManager',
    
    # Network Utilities
    'NetworkUtils',
    
    # Installation
    'InstallationChecker',
    
    # Logging
    'StackLogger',
]
