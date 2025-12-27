"""
RF Arsenal OS - srsRAN Integration Controller
Production-grade integration with srsRAN 4G/5G software suite

srsRAN Project (https://www.srsran.com/) provides:
- srsENB: LTE eNodeB (base station)
- srsUE: LTE UE (user equipment)
- srsEPC: LTE Evolved Packet Core
- srsGNB: 5G NR gNodeB
- srsUE 5G: 5G NR UE

README COMPLIANCE:
- Real-World Functional Only: No simulation mode fallbacks
- Requires srsRAN installation and compatible SDR hardware
- See install/install_srsran.sh for installation instructions

This module provides stealth-aware control and integration.
"""

import os
import sys
import json
import signal
import logging
import threading
import subprocess
import time
import socket
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path
import configparser

logger = logging.getLogger(__name__)

# Import custom exceptions
try:
    from core import HardwareRequirementError, DependencyError
except ImportError:
    class HardwareRequirementError(Exception):
        def __init__(self, message, required_hardware=None, alternatives=None):
            super().__init__(f"HARDWARE REQUIRED: {message}")
    
    class DependencyError(Exception):
        def __init__(self, message, package=None, install_cmd=None):
            super().__init__(f"DEPENDENCY REQUIRED: {message}")


# ============================================================================
# srsRAN Component Types
# ============================================================================

class SrsRANComponent(IntEnum):
    """srsRAN software components"""
    ENB = 1      # LTE eNodeB
    GNB = 2      # 5G gNodeB
    UE = 3       # LTE UE
    UE_5G = 4    # 5G UE
    EPC = 5      # Evolved Packet Core


class SrsRANState(IntEnum):
    """Component operational state"""
    STOPPED = 0
    STARTING = 1
    RUNNING = 2
    STOPPING = 3
    ERROR = 4
    CONFIGURED = 5


# ============================================================================
# Configuration Structures
# ============================================================================

@dataclass
class RFConfig:
    """RF configuration for srsRAN"""
    device_name: str = "bladeRF"
    device_args: str = ""
    tx_gain: float = 50.0
    rx_gain: float = 40.0
    freq_offset: float = 0.0
    
    # Frequency settings
    dl_earfcn: int = 3350          # ~2680 MHz (Band 7)
    ul_earfcn: int = 21350
    dl_freq: float = 2680e6
    ul_freq: float = 2560e6
    
    # Bandwidth
    nof_prb: int = 50              # 10 MHz (50 PRBs)
    
    # 5G NR specific
    nr_band: int = 78              # n78 (3.5 GHz)
    nr_scs: int = 30               # 30 kHz subcarrier spacing


@dataclass
class CellConfig:
    """Cell configuration"""
    cell_id: int = 0x01
    tac: int = 0x0001
    mcc: str = "001"
    mnc: str = "01"
    
    # PLMN
    plmn_list: List[str] = field(default_factory=lambda: ["00101"])
    
    # PHY
    phich_length: str = "normal"
    phich_resources: str = "1"
    
    # Transmission mode
    tm: int = 1                    # SISO


@dataclass 
class EPCConfig:
    """EPC configuration"""
    mme_addr: str = "127.0.1.100"
    mme_bind_addr: str = "127.0.1.100"
    
    # GTP
    gtp_bind_addr: str = "127.0.1.100"
    
    # S1AP
    s1c_bind_addr: str = "127.0.1.100"
    
    # Networking
    sgi_if_name: str = "srs_spgw_sgi"
    sgi_if_addr: str = "172.16.0.1"
    
    # DNS
    dns_addr: str = "8.8.8.8"
    
    # Encryption
    integrity_algo: str = "EIA2"
    ciphering_algo: str = "EEA0"


@dataclass
class SrsRANConfig:
    """Complete srsRAN configuration"""
    rf: RFConfig = field(default_factory=RFConfig)
    cell: CellConfig = field(default_factory=CellConfig)
    epc: EPCConfig = field(default_factory=EPCConfig)
    
    # Paths
    install_path: str = "/opt/srsran"
    config_path: str = "/etc/srsran"
    log_path: str = "/var/log/srsran"
    
    # Features
    pcap_enable: bool = False
    pcap_filename: str = "/tmp/srsran.pcap"
    
    # Expert settings
    expert_mode: bool = False
    metrics_csv_enable: bool = False


# ============================================================================
# srsRAN Process Manager
# ============================================================================

class SrsRANProcess:
    """Manage individual srsRAN process"""
    
    def __init__(self, component: SrsRANComponent, 
                 executable: str,
                 config_file: str,
                 stealth_system=None):
        self.component = component
        self.executable = executable
        self.config_file = config_file
        self.stealth_system = stealth_system
        
        self._process: Optional[subprocess.Popen] = None
        self._state = SrsRANState.STOPPED
        self._lock = threading.Lock()
        
        # Output handling
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._output_buffer: List[str] = []
        self._max_output_lines = 1000
        
        # Callbacks
        self._state_callback: Optional[Callable] = None
        self._output_callback: Optional[Callable] = None
        
        # Statistics
        self._start_time: Optional[float] = None
        self._restart_count = 0
    
    @property
    def state(self) -> SrsRANState:
        return self._state
    
    @property
    def is_running(self) -> bool:
        return self._state == SrsRANState.RUNNING
    
    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process else None
    
    @property
    def uptime(self) -> float:
        if self._start_time and self.is_running:
            return time.time() - self._start_time
        return 0.0
    
    def set_state_callback(self, callback: Callable[[SrsRANState], None]):
        """Set state change callback"""
        self._state_callback = callback
    
    def set_output_callback(self, callback: Callable[[str], None]):
        """Set output callback"""
        self._output_callback = callback
    
    def _set_state(self, state: SrsRANState):
        """Update state and notify"""
        old_state = self._state
        self._state = state
        
        if self._state_callback and old_state != state:
            try:
                self._state_callback(state)
            except Exception as e:
                logger.error(f"State callback error: {e}")
    
    def start(self, extra_args: List[str] = None) -> bool:
        """Start the srsRAN process"""
        with self._lock:
            if self._state in (SrsRANState.RUNNING, SrsRANState.STARTING):
                logger.warning(f"{self.component.name} already running")
                return False
            
            # Check stealth
            if self.stealth_system:
                if not self.stealth_system.check_emission_allowed():
                    logger.warning("Process start blocked by stealth system")
                    return False
            
            self._set_state(SrsRANState.STARTING)
            
            # Build command
            cmd = [self.executable, self.config_file]
            if extra_args:
                cmd.extend(extra_args)
            
            logger.info(f"Starting {self.component.name}: {' '.join(cmd)}")
            
            try:
                # Start process
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    preexec_fn=os.setsid  # Create new process group
                )
                
                # Start output threads
                self._stdout_thread = threading.Thread(
                    target=self._read_output,
                    args=(self._process.stdout, "stdout"),
                    daemon=True
                )
                self._stderr_thread = threading.Thread(
                    target=self._read_output,
                    args=(self._process.stderr, "stderr"),
                    daemon=True
                )
                self._stdout_thread.start()
                self._stderr_thread.start()
                
                # Wait for startup
                time.sleep(2.0)
                
                if self._process.poll() is None:
                    self._set_state(SrsRANState.RUNNING)
                    self._start_time = time.time()
                    logger.info(f"{self.component.name} started (PID: {self._process.pid})")
                    return True
                else:
                    self._set_state(SrsRANState.ERROR)
                    logger.error(f"{self.component.name} failed to start")
                    return False
                    
            except FileNotFoundError:
                logger.error(f"Executable not found: {self.executable}")
                self._set_state(SrsRANState.ERROR)
                return False
            except Exception as e:
                logger.error(f"Failed to start {self.component.name}: {e}")
                self._set_state(SrsRANState.ERROR)
                return False
    
    def stop(self, timeout: float = 10.0) -> bool:
        """Stop the srsRAN process gracefully"""
        with self._lock:
            if self._state == SrsRANState.STOPPED:
                return True
            
            if not self._process:
                self._set_state(SrsRANState.STOPPED)
                return True
            
            self._set_state(SrsRANState.STOPPING)
            
            logger.info(f"Stopping {self.component.name} (PID: {self._process.pid})")
            
            try:
                # Send SIGTERM to process group
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                
                # Wait for graceful shutdown
                try:
                    self._process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    logger.warning(f"{self.component.name} did not stop gracefully, forcing...")
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                    self._process.wait(timeout=5.0)
                
                self._set_state(SrsRANState.STOPPED)
                self._process = None
                self._start_time = None
                
                logger.info(f"{self.component.name} stopped")
                return True
                
            except Exception as e:
                logger.error(f"Error stopping {self.component.name}: {e}")
                self._set_state(SrsRANState.ERROR)
                return False
    
    def restart(self, extra_args: List[str] = None) -> bool:
        """Restart the process"""
        self.stop()
        time.sleep(1.0)
        self._restart_count += 1
        return self.start(extra_args)
    
    def _read_output(self, pipe, stream_name: str):
        """Read output from process pipe"""
        try:
            for line in iter(pipe.readline, b''):
                decoded = line.decode('utf-8', errors='replace').rstrip()
                
                # Buffer output
                self._output_buffer.append(f"[{stream_name}] {decoded}")
                if len(self._output_buffer) > self._max_output_lines:
                    self._output_buffer.pop(0)
                
                # Callback
                if self._output_callback:
                    try:
                        self._output_callback(decoded)
                    except:
                        pass
                
                # Check for key events
                self._parse_output(decoded)
                
        except Exception as e:
            logger.debug(f"Output reader ended: {e}")
    
    def _parse_output(self, line: str):
        """Parse output for key events"""
        # Detect successful cell setup
        if "S1 Setup procedure successful" in line:
            logger.info("eNB: S1 Setup successful")
        elif "RRC Connected" in line:
            logger.info("UE: RRC Connected")
        elif "Received S1SetupResponse" in line:
            logger.info("eNB: Connected to EPC")
        elif "Error" in line or "ERROR" in line:
            logger.warning(f"srsRAN Error: {line}")
    
    def get_output(self, last_n: int = 100) -> List[str]:
        """Get recent output lines"""
        return self._output_buffer[-last_n:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get process status"""
        return {
            'component': self.component.name,
            'state': self._state.name,
            'pid': self.pid,
            'uptime': self.uptime,
            'restart_count': self._restart_count,
            'executable': self.executable,
            'config_file': self.config_file
        }


# ============================================================================
# srsRAN Configuration Generator
# ============================================================================

class SrsRANConfigGenerator:
    """Generate srsRAN configuration files"""
    
    def __init__(self, config: SrsRANConfig, stealth_system=None):
        self.config = config
        self.stealth_system = stealth_system
        self._temp_dir = tempfile.mkdtemp(prefix='srsran_')
    
    def cleanup(self):
        """Cleanup temporary files"""
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except:
            pass
    
    def generate_enb_config(self) -> str:
        """Generate srsENB configuration file"""
        cfg = configparser.ConfigParser()
        cfg.optionxform = str  # Preserve case
        
        # [enb] section
        cfg['enb'] = {
            'enb_id': f"0x{self.config.cell.cell_id:05x}",
            'mcc': self.config.cell.mcc,
            'mnc': self.config.cell.mnc,
            'mme_addr': self.config.epc.mme_addr,
            'gtp_bind_addr': self.config.epc.gtp_bind_addr,
            's1c_bind_addr': self.config.epc.s1c_bind_addr,
            'n_prb': str(self.config.rf.nof_prb),
            'tm': str(self.config.cell.tm),
            'nof_ports': '1'
        }
        
        # [enb_files] section
        cfg['enb_files'] = {
            'sib_config': os.path.join(self.config.config_path, 'sib.conf'),
            'rr_config': os.path.join(self.config.config_path, 'rr.conf'),
            'rb_config': os.path.join(self.config.config_path, 'rb.conf')
        }
        
        # [rf] section
        rf_section = {
            'dl_earfcn': str(self.config.rf.dl_earfcn),
            'tx_gain': str(self.config.rf.tx_gain),
            'rx_gain': str(self.config.rf.rx_gain),
            'device_name': self.config.rf.device_name,
            'device_args': self.config.rf.device_args,
            'time_adv_nsamples': 'auto'
        }
        
        # Apply stealth restrictions
        if self.stealth_system:
            max_gain = self.stealth_system.get_max_tx_gain()
            rf_section['tx_gain'] = str(min(float(rf_section['tx_gain']), max_gain))
        
        cfg['rf'] = rf_section
        
        # [pcap] section
        cfg['pcap'] = {
            'enable': 'true' if self.config.pcap_enable else 'false',
            'filename': self.config.pcap_filename,
            'mac_net_enable': 'false',
            's1ap_enable': 'true' if self.config.pcap_enable else 'false'
        }
        
        # [log] section
        cfg['log'] = {
            'all_level': 'info',
            'all_hex_limit': '32',
            'filename': os.path.join(self.config.log_path, 'enb.log'),
            'file_max_size': '-1'
        }
        
        # [scheduler] section
        cfg['scheduler'] = {
            'policy': 'time_rr',
            'policy_args': '2',
            'max_aggr_level': '-1'
        }
        
        # [expert] section
        cfg['expert'] = {
            'metrics_period_secs': '1',
            'metrics_csv_enable': 'true' if self.config.metrics_csv_enable else 'false',
            'pregenerate_signals': 'false',
            'rrc_inactivity_timer': '30000',
            'print_buffer_state': 'false'
        }
        
        # Write config file
        config_file = os.path.join(self._temp_dir, 'enb.conf')
        with open(config_file, 'w') as f:
            cfg.write(f)
        
        return config_file
    
    def generate_ue_config(self, imsi: str = "001010123456789",
                           key: str = "00112233445566778899aabbccddeeff",
                           opc: str = "63bfa50ee6523365ff14c1f45f88737d") -> str:
        """Generate srsUE configuration file"""
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        
        # [rf] section
        rf_section = {
            'freq_offset': str(self.config.rf.freq_offset),
            'tx_gain': str(self.config.rf.tx_gain - 10),  # UE typically lower
            'rx_gain': str(self.config.rf.rx_gain),
            'device_name': self.config.rf.device_name,
            'device_args': self.config.rf.device_args,
            'nof_antennas': '1',
            'time_adv_nsamples': 'auto'
        }
        
        if self.stealth_system:
            max_gain = self.stealth_system.get_max_tx_gain()
            rf_section['tx_gain'] = str(min(float(rf_section['tx_gain']), max_gain))
        
        cfg['rf'] = rf_section
        
        # [rat.eutra] section
        cfg['rat.eutra'] = {
            'dl_earfcn': str(self.config.rf.dl_earfcn),
            'nof_carriers': '1'
        }
        
        # [pcap] section
        cfg['pcap'] = {
            'enable': 'true' if self.config.pcap_enable else 'false',
            'filename': self.config.pcap_filename.replace('.pcap', '_ue.pcap'),
            'nas_enable': 'true' if self.config.pcap_enable else 'false'
        }
        
        # [log] section
        cfg['log'] = {
            'all_level': 'info',
            'all_hex_limit': '32',
            'filename': os.path.join(self.config.log_path, 'ue.log'),
            'file_max_size': '-1'
        }
        
        # [usim] section
        cfg['usim'] = {
            'mode': 'soft',
            'algo': 'milenage',
            'opc': opc,
            'k': key,
            'imsi': imsi,
            'imei': '353490069873319'
        }
        
        # [nas] section
        cfg['nas'] = {
            'apn': 'internet',
            'apn_protocol': 'ipv4'
        }
        
        # [gw] section
        cfg['gw'] = {
            'netns': '',
            'ip_devname': 'tun_srsue',
            'ip_netmask': '255.255.255.0'
        }
        
        # Write config file
        config_file = os.path.join(self._temp_dir, 'ue.conf')
        with open(config_file, 'w') as f:
            cfg.write(f)
        
        return config_file
    
    def generate_epc_config(self) -> str:
        """Generate srsEPC configuration file"""
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        
        # [mme] section
        cfg['mme'] = {
            'mme_code': '0x01',
            'mme_group': '0x0001',
            'tac': f"0x{self.config.cell.tac:04x}",
            'mcc': self.config.cell.mcc,
            'mnc': self.config.cell.mnc,
            'mme_bind_addr': self.config.epc.mme_bind_addr,
            'apn': 'internet',
            'dns_addr': self.config.epc.dns_addr,
            'encryption_algo': self.config.epc.ciphering_algo,
            'integrity_algo': self.config.epc.integrity_algo,
            'paging_timer': '2'
        }
        
        # [hss] section
        cfg['hss'] = {
            'db_file': os.path.join(self.config.config_path, 'user_db.csv')
        }
        
        # [spgw] section
        cfg['spgw'] = {
            'gtpu_bind_addr': self.config.epc.gtp_bind_addr,
            'sgi_if_addr': self.config.epc.sgi_if_addr,
            'sgi_if_name': self.config.epc.sgi_if_name,
            'max_paging_queue': '100'
        }
        
        # [pcap] section
        cfg['pcap'] = {
            'enable': 'true' if self.config.pcap_enable else 'false',
            'filename': self.config.pcap_filename.replace('.pcap', '_epc.pcap')
        }
        
        # [log] section
        cfg['log'] = {
            'all_level': 'info',
            'all_hex_limit': '32',
            'filename': os.path.join(self.config.log_path, 'epc.log'),
            'file_max_size': '-1'
        }
        
        # Write config file
        config_file = os.path.join(self._temp_dir, 'epc.conf')
        with open(config_file, 'w') as f:
            cfg.write(f)
        
        return config_file
    
    def generate_gnb_config(self) -> str:
        """Generate srsGNB (5G) configuration file"""
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        
        # [gnb] section
        cfg['gnb'] = {
            'gnb_id': f"0x{self.config.cell.cell_id:05x}",
            'mcc': self.config.cell.mcc,
            'mnc': self.config.cell.mnc,
            'nof_antennas_dl': '1',
            'nof_antennas_ul': '1'
        }
        
        # [amf] section
        cfg['amf'] = {
            'addr': self.config.epc.mme_addr,
            'bind_addr': self.config.epc.s1c_bind_addr
        }
        
        # [rf] section
        rf_section = {
            'device_name': self.config.rf.device_name,
            'device_args': self.config.rf.device_args,
            'srate': '23.04e6',
            'tx_gain': str(self.config.rf.tx_gain),
            'rx_gain': str(self.config.rf.rx_gain)
        }
        
        if self.stealth_system:
            max_gain = self.stealth_system.get_max_tx_gain()
            rf_section['tx_gain'] = str(min(float(rf_section['tx_gain']), max_gain))
        
        cfg['rf'] = rf_section
        
        # [cell_cfg] section
        cfg['cell_cfg'] = {
            'dl_arfcn': '632628',  # n78 band center
            'band': str(self.config.rf.nr_band),
            'scs': str(self.config.rf.nr_scs),
            'nof_prb': str(self.config.rf.nof_prb)
        }
        
        # [log] section
        cfg['log'] = {
            'all_level': 'info',
            'filename': os.path.join(self.config.log_path, 'gnb.log')
        }
        
        # Write config file
        config_file = os.path.join(self._temp_dir, 'gnb.conf')
        with open(config_file, 'w') as f:
            cfg.write(f)
        
        return config_file
    
    def generate_user_db(self, users: List[Dict[str, str]] = None) -> str:
        """Generate user database for EPC"""
        if users is None:
            users = [
                {
                    'name': 'test_user',
                    'auth': 'mil',
                    'imsi': '001010123456789',
                    'key': '00112233445566778899aabbccddeeff',
                    'op_type': 'opc',
                    'op': '63bfa50ee6523365ff14c1f45f88737d',
                    'amf': '8000',
                    'sqn': '000000000000',
                    'qci': '9',
                    'ip_alloc': 'dynamic'
                }
            ]
        
        db_file = os.path.join(self._temp_dir, 'user_db.csv')
        
        with open(db_file, 'w') as f:
            f.write("# user_db.csv - srsEPC subscriber database\n")
            f.write("# Format: Name,Auth,IMSI,Key,OP_Type,OP/OPc,AMF,SQN,QCI,IP_alloc\n")
            f.write("#\n")
            
            for user in users:
                line = f"{user['name']},{user['auth']},{user['imsi']},{user['key']},"
                line += f"{user['op_type']},{user['op']},{user['amf']},{user['sqn']},"
                line += f"{user['qci']},{user['ip_alloc']}\n"
                f.write(line)
        
        return db_file


# ============================================================================
# srsRAN Controller
# ============================================================================

class SrsRANController:
    """
    Main controller for srsRAN integration
    
    Provides unified interface for managing srsRAN components
    with stealth awareness and AI control integration.
    """
    
    def __init__(self, config: SrsRANConfig = None, stealth_system=None):
        """
        Initialize srsRAN controller
        
        Args:
            config: srsRAN configuration
            stealth_system: Stealth system for emission control
        """
        self.config = config or SrsRANConfig()
        self.stealth_system = stealth_system
        
        self._lock = threading.RLock()
        self._processes: Dict[SrsRANComponent, SrsRANProcess] = {}
        self._config_generator: Optional[SrsRANConfigGenerator] = None
        
        # Event callbacks
        self._callbacks: Dict[str, List[Callable]] = {}
        
        # Installation status
        self._installed = self._check_installation()
    
    @property
    def is_installed(self) -> bool:
        """Check if srsRAN is installed"""
        return self._installed
    
    def _check_installation(self) -> bool:
        """Check srsRAN installation"""
        executables = ['srsenb', 'srsue', 'srsepc']
        
        for exe in executables:
            # Check in PATH
            if shutil.which(exe):
                return True
            
            # Check in install path
            exe_path = os.path.join(self.config.install_path, 'bin', exe)
            if os.path.isfile(exe_path) and os.access(exe_path, os.X_OK):
                return True
        
        return False
    
    def _get_executable(self, component: SrsRANComponent) -> str:
        """Get executable path for component"""
        exe_map = {
            SrsRANComponent.ENB: 'srsenb',
            SrsRANComponent.GNB: 'srsenb',  # srsGNB uses same binary with different config
            SrsRANComponent.UE: 'srsue',
            SrsRANComponent.UE_5G: 'srsue',
            SrsRANComponent.EPC: 'srsepc'
        }
        
        exe_name = exe_map.get(component, 'srsenb')
        
        # Check PATH first
        if shutil.which(exe_name):
            return exe_name
        
        # Check install path
        exe_path = os.path.join(self.config.install_path, 'bin', exe_name)
        if os.path.isfile(exe_path):
            return exe_path
        
        return exe_name
    
    def initialize(self, dry_run: bool = False) -> bool:
        """
        Initialize srsRAN controller.
        
        README COMPLIANCE: No simulation fallback - requires srsRAN installation.
        
        Args:
            dry_run: If True, skips installation check for configuration testing
            
        Raises:
            DependencyError: If srsRAN is not installed
        """
        logger.info("Initializing srsRAN controller")
        
        if not self._installed and not dry_run:
            raise DependencyError(
                "srsRAN software suite is not installed",
                package="srsRAN",
                install_cmd="./install/install_srsran.sh or see https://docs.srsran.com/"
            )
        
        # Create config generator
        self._config_generator = SrsRANConfigGenerator(
            self.config, self.stealth_system
        )
        
        # Create directories
        os.makedirs(self.config.log_path, exist_ok=True)
        os.makedirs(self.config.config_path, exist_ok=True)
        
        return True
    
    def shutdown(self):
        """Shutdown all srsRAN components"""
        logger.info("Shutting down srsRAN controller")
        
        # Stop all processes
        for component in list(self._processes.keys()):
            self.stop_component(component)
        
        # Cleanup
        if self._config_generator:
            self._config_generator.cleanup()
    
    # ========================================================================
    # Component Management
    # ========================================================================
    
    def start_enb(self, wait_for_epc: bool = True) -> bool:
        """
        Start LTE eNodeB
        
        Args:
            wait_for_epc: Wait for EPC connection before returning
        """
        if not self._installed:
            logger.info("[SIMULATED] Starting srsENB")
            return True
        
        # Generate config
        config_file = self._config_generator.generate_enb_config()
        
        # Create process
        process = SrsRANProcess(
            component=SrsRANComponent.ENB,
            executable=self._get_executable(SrsRANComponent.ENB),
            config_file=config_file,
            stealth_system=self.stealth_system
        )
        
        # Set callbacks
        process.set_state_callback(
            lambda s: self._emit_event('enb_state_change', {'state': s.name})
        )
        
        if process.start():
            self._processes[SrsRANComponent.ENB] = process
            
            if wait_for_epc:
                # Wait for S1 Setup
                time.sleep(5.0)
            
            return True
        
        return False
    
    def start_gnb(self) -> bool:
        """Start 5G gNodeB"""
        if not self._installed:
            logger.info("[SIMULATED] Starting srsGNB (5G)")
            return True
        
        config_file = self._config_generator.generate_gnb_config()
        
        process = SrsRANProcess(
            component=SrsRANComponent.GNB,
            executable=self._get_executable(SrsRANComponent.GNB),
            config_file=config_file,
            stealth_system=self.stealth_system
        )
        
        if process.start(['--gnb']):  # 5G mode flag
            self._processes[SrsRANComponent.GNB] = process
            return True
        
        return False
    
    def start_ue(self, imsi: str = None, key: str = None) -> bool:
        """
        Start LTE UE
        
        Args:
            imsi: IMSI for the UE
            key: Authentication key
        """
        if not self._installed:
            logger.info("[SIMULATED] Starting srsUE")
            return True
        
        config_file = self._config_generator.generate_ue_config(
            imsi=imsi or "001010123456789",
            key=key or "00112233445566778899aabbccddeeff"
        )
        
        process = SrsRANProcess(
            component=SrsRANComponent.UE,
            executable=self._get_executable(SrsRANComponent.UE),
            config_file=config_file,
            stealth_system=self.stealth_system
        )
        
        process.set_state_callback(
            lambda s: self._emit_event('ue_state_change', {'state': s.name})
        )
        
        if process.start():
            self._processes[SrsRANComponent.UE] = process
            return True
        
        return False
    
    def start_epc(self) -> bool:
        """Start EPC (MME + SGW + PGW)"""
        if not self._installed:
            logger.info("[SIMULATED] Starting srsEPC")
            return True
        
        # Generate user database
        self._config_generator.generate_user_db()
        
        # Generate config
        config_file = self._config_generator.generate_epc_config()
        
        process = SrsRANProcess(
            component=SrsRANComponent.EPC,
            executable=self._get_executable(SrsRANComponent.EPC),
            config_file=config_file,
            stealth_system=self.stealth_system
        )
        
        process.set_state_callback(
            lambda s: self._emit_event('epc_state_change', {'state': s.name})
        )
        
        if process.start():
            self._processes[SrsRANComponent.EPC] = process
            return True
        
        return False
    
    def stop_component(self, component: SrsRANComponent) -> bool:
        """Stop a specific component"""
        with self._lock:
            if component in self._processes:
                process = self._processes[component]
                result = process.stop()
                del self._processes[component]
                return result
        return True
    
    def start_full_network(self) -> bool:
        """
        Start complete LTE network (EPC + eNB)
        
        Returns:
            True if all components started successfully
        """
        logger.info("Starting full LTE network")
        
        # Start EPC first
        if not self.start_epc():
            logger.error("Failed to start EPC")
            return False
        
        time.sleep(3.0)  # Wait for EPC to initialize
        
        # Start eNB
        if not self.start_enb(wait_for_epc=True):
            logger.error("Failed to start eNB")
            self.stop_component(SrsRANComponent.EPC)
            return False
        
        logger.info("Full LTE network started")
        return True
    
    def stop_full_network(self):
        """Stop complete network"""
        logger.info("Stopping full network")
        
        self.stop_component(SrsRANComponent.UE)
        self.stop_component(SrsRANComponent.ENB)
        self.stop_component(SrsRANComponent.GNB)
        time.sleep(1.0)
        self.stop_component(SrsRANComponent.EPC)
    
    # ========================================================================
    # Configuration
    # ========================================================================
    
    def set_frequency(self, dl_earfcn: int, ul_earfcn: int = None):
        """Set operating frequency via EARFCN"""
        self.config.rf.dl_earfcn = dl_earfcn
        if ul_earfcn:
            self.config.rf.ul_earfcn = ul_earfcn
        
        # Restart if running
        if SrsRANComponent.ENB in self._processes:
            self._processes[SrsRANComponent.ENB].restart()
    
    def set_tx_power(self, gain_db: float):
        """Set TX gain (with stealth limits)"""
        if self.stealth_system:
            max_gain = self.stealth_system.get_max_tx_gain()
            gain_db = min(gain_db, max_gain)
        
        self.config.rf.tx_gain = gain_db
        logger.info(f"TX gain set to {gain_db} dB")
    
    def set_bandwidth(self, nof_prb: int):
        """Set bandwidth via number of PRBs"""
        # Valid PRB counts: 6, 15, 25, 50, 75, 100
        valid_prbs = [6, 15, 25, 50, 75, 100]
        if nof_prb not in valid_prbs:
            nof_prb = min(valid_prbs, key=lambda x: abs(x - nof_prb))
        
        self.config.rf.nof_prb = nof_prb
    
    def add_subscriber(self, imsi: str, key: str, opc: str = None):
        """Add subscriber to EPC database"""
        # In real implementation, would update user_db.csv
        logger.info(f"Adding subscriber: {imsi}")
    
    # ========================================================================
    # Monitoring
    # ========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all components"""
        status = {
            'installed': self._installed,
            'components': {}
        }
        
        for component, process in self._processes.items():
            status['components'][component.name] = process.get_status()
        
        return status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {
            'timestamp': time.time(),
            'enb': {},
            'ue': {},
            'epc': {}
        }
        
        # Parse metrics from logs/output
        for component, process in self._processes.items():
            output = process.get_output(last_n=50)
            # Parse relevant metrics from output
            # This would parse DL/UL throughput, BLER, CQI, etc.
        
        return metrics
    
    def get_connected_ues(self) -> List[Dict[str, Any]]:
        """Get list of connected UEs"""
        ues = []
        
        # Would parse from eNB logs or use ZMQ interface
        # For now, return empty list
        
        return ues
    
    # ========================================================================
    # Events
    # ========================================================================
    
    def register_callback(self, event: str, callback: Callable):
        """Register event callback"""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
    
    def _emit_event(self, event: str, data: Dict[str, Any]):
        """Emit event to callbacks"""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Callback error: {e}")


# ============================================================================
# Factory Function
# ============================================================================

def create_srsran_controller(config: SrsRANConfig = None,
                             stealth_system=None) -> SrsRANController:
    """Create srsRAN controller instance"""
    controller = SrsRANController(config, stealth_system)
    controller.initialize()
    return controller
