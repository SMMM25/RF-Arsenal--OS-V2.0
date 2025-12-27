"""
RF Arsenal OS - OpenAirInterface Integration Controller
Production-grade integration with OpenAirInterface 5G/LTE software suite

OpenAirInterface (https://openairinterface.org/) provides:
- OAI RAN: eNB/gNB (base station)
- OAI UE: LTE/NR UE
- OAI CN: 5G Core Network (AMF, SMF, UPF, etc.)

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
import yaml
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# OAI Component Types
# ============================================================================

class OAIComponent(IntEnum):
    """OpenAirInterface components"""
    ENB = 1          # LTE eNodeB
    GNB = 2          # 5G gNodeB
    NR_UE = 3        # 5G NR UE
    LTE_UE = 4       # LTE UE
    AMF = 5          # 5G Access and Mobility Management
    SMF = 6          # 5G Session Management Function
    UPF = 7          # 5G User Plane Function
    NRF = 8          # 5G Network Repository Function
    AUSF = 9         # 5G Authentication Server Function
    UDM = 10         # 5G Unified Data Management
    UDR = 11         # 5G Unified Data Repository
    NSSF = 12        # 5G Network Slice Selection Function


class OAIState(IntEnum):
    """Component operational state"""
    STOPPED = 0
    STARTING = 1
    RUNNING = 2
    STOPPING = 3
    ERROR = 4
    CONFIGURED = 5


class OAIDeploymentMode(IntEnum):
    """Deployment mode"""
    STANDALONE = 0       # Standalone deployment
    DOCKER = 1          # Docker containerized
    KUBERNETES = 2      # Kubernetes orchestrated


# ============================================================================
# Configuration Structures
# ============================================================================

@dataclass
class OAIRFConfig:
    """RF configuration for OAI RAN"""
    # SDR Configuration
    sdr_type: str = "bladerf"      # uhd, bladerf, lmssdr
    device_args: str = ""
    
    # TX/RX Gains
    tx_gain: float = 50.0
    rx_gain: float = 60.0
    
    # Frequency Configuration
    dl_frequency_hz: float = 2680e6      # Band 7 DL
    ul_frequency_hz: float = 2560e6      # Band 7 UL
    band: int = 7
    
    # NR Configuration
    nr_band: int = 78                    # n78 (3.5 GHz)
    nr_dl_frequency_hz: float = 3619.2e6
    nr_ssb_frequency_hz: float = 3619.2e6
    nr_scs: int = 30                     # Subcarrier spacing kHz
    
    # Bandwidth
    prb_count: int = 106                 # 40 MHz for NR
    lte_prb_count: int = 50              # 10 MHz for LTE
    
    # MIMO
    num_tx_antennas: int = 1
    num_rx_antennas: int = 1
    
    # Clock/Timing
    clock_source: str = "internal"
    time_source: str = "internal"


@dataclass
class OAICellConfig:
    """Cell configuration"""
    # Cell Identity
    cell_id: int = 0
    physical_cell_id: int = 0
    
    # PLMN
    mcc: str = "001"
    mnc: str = "01"
    tac: int = 1
    
    # SSB Configuration (NR)
    ssb_offset_point_a: int = 0
    ssb_periodicity_ms: int = 20
    
    # PRACH Configuration
    prach_config_index: int = 98
    
    # Reference Signals
    pdsch_ref_signal_power: int = -27


@dataclass
class OAICoreConfig:
    """5G Core Network configuration"""
    # AMF Configuration
    amf_ip: str = "192.168.70.132"
    amf_port: int = 38412
    
    # SMF Configuration
    smf_ip: str = "192.168.70.133"
    
    # UPF Configuration
    upf_ip: str = "192.168.70.134"
    n3_ip: str = "192.168.70.134"
    n6_ip: str = "192.168.70.134"
    
    # NRF Configuration
    nrf_ip: str = "192.168.70.130"
    nrf_port: int = 8080
    
    # Network Configuration
    dnn: str = "oai"
    network_name: str = "OAI-RFArsenal"
    
    # Security
    integrity_algorithm: List[str] = field(default_factory=lambda: ["NIA1", "NIA2"])
    ciphering_algorithm: List[str] = field(default_factory=lambda: ["NEA0", "NEA1", "NEA2"])


@dataclass
class OAIConfig:
    """Complete OAI configuration"""
    rf: OAIRFConfig = field(default_factory=OAIRFConfig)
    cell: OAICellConfig = field(default_factory=OAICellConfig)
    core: OAICoreConfig = field(default_factory=OAICoreConfig)
    
    # Paths
    oai_path: str = "/opt/openairinterface5g"
    oai_cn_path: str = "/opt/oai-cn5g"
    config_path: str = "/etc/oai"
    log_path: str = "/var/log/oai"
    
    # Deployment
    deployment_mode: OAIDeploymentMode = OAIDeploymentMode.STANDALONE
    docker_compose_file: str = ""
    
    # Features
    enable_logging: bool = True
    enable_pcap: bool = False
    pcap_path: str = "/tmp/oai.pcap"
    
    # Performance tuning
    thread_pool_size: int = 8
    real_time_priority: bool = True


# ============================================================================
# OAI Process Manager
# ============================================================================

class OAIProcess:
    """Manage individual OAI process"""
    
    def __init__(self, component: OAIComponent,
                 executable: str,
                 config_file: str,
                 stealth_system=None):
        self.component = component
        self.executable = executable
        self.config_file = config_file
        self.stealth_system = stealth_system
        
        self._process: Optional[subprocess.Popen] = None
        self._state = OAIState.STOPPED
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
        
        # Connection tracking
        self._connected_ues: Dict[str, Dict] = {}
    
    @property
    def state(self) -> OAIState:
        return self._state
    
    @property
    def is_running(self) -> bool:
        return self._state == OAIState.RUNNING
    
    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process else None
    
    @property
    def uptime(self) -> float:
        if self._start_time and self.is_running:
            return time.time() - self._start_time
        return 0.0
    
    def set_state_callback(self, callback: Callable[[OAIState], None]):
        self._state_callback = callback
    
    def set_output_callback(self, callback: Callable[[str], None]):
        self._output_callback = callback
    
    def _set_state(self, state: OAIState):
        old_state = self._state
        self._state = state
        
        if self._state_callback and old_state != state:
            try:
                self._state_callback(state)
            except Exception as e:
                logger.error(f"State callback error: {e}")
    
    def start(self, extra_args: List[str] = None) -> bool:
        """Start OAI process"""
        with self._lock:
            if self._state in (OAIState.RUNNING, OAIState.STARTING):
                logger.warning(f"{self.component.name} already running")
                return False
            
            # Check stealth
            if self.stealth_system:
                if not self.stealth_system.check_emission_allowed():
                    logger.warning("Process start blocked by stealth system")
                    return False
            
            self._set_state(OAIState.STARTING)
            
            # Build command
            cmd = [self.executable]
            
            if self.config_file:
                cmd.extend(['-O', self.config_file])
            
            if extra_args:
                cmd.extend(extra_args)
            
            logger.info(f"Starting {self.component.name}: {' '.join(cmd)}")
            
            try:
                # Set environment for real-time
                env = os.environ.copy()
                env['OAI_GDBSTACKS'] = '1'
                
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    env=env,
                    preexec_fn=os.setsid
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
                time.sleep(3.0)
                
                if self._process.poll() is None:
                    self._set_state(OAIState.RUNNING)
                    self._start_time = time.time()
                    logger.info(f"{self.component.name} started (PID: {self._process.pid})")
                    return True
                else:
                    self._set_state(OAIState.ERROR)
                    logger.error(f"{self.component.name} failed to start")
                    return False
                    
            except FileNotFoundError:
                logger.error(f"Executable not found: {self.executable}")
                self._set_state(OAIState.ERROR)
                return False
            except Exception as e:
                logger.error(f"Failed to start {self.component.name}: {e}")
                self._set_state(OAIState.ERROR)
                return False
    
    def stop(self, timeout: float = 15.0) -> bool:
        """Stop OAI process gracefully"""
        with self._lock:
            if self._state == OAIState.STOPPED:
                return True
            
            if not self._process:
                self._set_state(OAIState.STOPPED)
                return True
            
            self._set_state(OAIState.STOPPING)
            
            logger.info(f"Stopping {self.component.name} (PID: {self._process.pid})")
            
            try:
                # Send SIGINT first (graceful)
                os.killpg(os.getpgid(self._process.pid), signal.SIGINT)
                
                try:
                    self._process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    logger.warning(f"{self.component.name} did not stop gracefully, forcing...")
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                    self._process.wait(timeout=5.0)
                
                self._set_state(OAIState.STOPPED)
                self._process = None
                self._start_time = None
                
                logger.info(f"{self.component.name} stopped")
                return True
                
            except Exception as e:
                logger.error(f"Error stopping {self.component.name}: {e}")
                self._set_state(OAIState.ERROR)
                return False
    
    def restart(self, extra_args: List[str] = None) -> bool:
        """Restart the process"""
        self.stop()
        time.sleep(2.0)
        self._restart_count += 1
        return self.start(extra_args)
    
    def _read_output(self, pipe, stream_name: str):
        """Read output from process pipe"""
        try:
            for line in iter(pipe.readline, b''):
                decoded = line.decode('utf-8', errors='replace').rstrip()
                
                self._output_buffer.append(f"[{stream_name}] {decoded}")
                if len(self._output_buffer) > self._max_output_lines:
                    self._output_buffer.pop(0)
                
                if self._output_callback:
                    try:
                        self._output_callback(decoded)
                    except:
                        pass
                
                self._parse_output(decoded)
                
        except Exception as e:
            logger.debug(f"Output reader ended: {e}")
    
    def _parse_output(self, line: str):
        """Parse output for key events"""
        # gNB events
        if "NG Setup procedure successful" in line:
            logger.info("gNB: NG Setup successful (connected to AMF)")
        elif "Initial UE message" in line:
            logger.info("gNB: Initial UE message received")
        elif "PDU Session Establishment" in line:
            logger.info("gNB: PDU Session established")
        
        # UE events
        if "RRC Connected" in line:
            logger.info("UE: RRC Connected")
        elif "Registration complete" in line:
            logger.info("UE: Registration complete")
        
        # Error events
        if "ERROR" in line or "error" in line.lower():
            logger.warning(f"OAI Error: {line}")
    
    def get_output(self, last_n: int = 100) -> List[str]:
        return self._output_buffer[-last_n:]
    
    def get_status(self) -> Dict[str, Any]:
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
# OAI Configuration Generator
# ============================================================================

class OAIConfigGenerator:
    """Generate OAI configuration files"""
    
    def __init__(self, config: OAIConfig, stealth_system=None):
        self.config = config
        self.stealth_system = stealth_system
        self._temp_dir = tempfile.mkdtemp(prefix='oai_')
    
    def cleanup(self):
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except:
            pass
    
    def generate_gnb_config(self) -> str:
        """Generate gNB configuration file"""
        gnb_config = {
            'Active_gNBs': ['gNB-RFArsenal'],
            'Asn1_verbosity': 'none',
            
            'gNBs': [{
                'gNB_ID': self.config.cell.cell_id,
                'gNB_name': 'gNB-RFArsenal',
                'tracking_area_code': self.config.cell.tac,
                
                'plmn_list': [{
                    'mcc': int(self.config.cell.mcc),
                    'mnc': int(self.config.cell.mnc),
                    'mnc_length': len(self.config.cell.mnc),
                    'snssaiList': [{
                        'sst': 1,
                        'sd': '0xffffff'
                    }]
                }],
                
                'nr_cellid': 12345678,
                
                # Physical parameters
                'min_rxtxtime': 6,
                
                # Security
                'do_CSIRS': 1,
                'do_SRS': 1,
                
                # SSB Configuration
                'servingCellConfigCommon': [{
                    'physCellId': self.config.cell.physical_cell_id,
                    'absoluteFrequencySSB': self._freq_to_arfcn(
                        self.config.rf.nr_ssb_frequency_hz
                    ),
                    'dl_frequencyBand': self.config.rf.nr_band,
                    'dl_absoluteFrequencyPointA': self._freq_to_arfcn(
                        self.config.rf.nr_dl_frequency_hz - 
                        (self.config.rf.prb_count * 12 * self.config.rf.nr_scs * 1000 / 2)
                    ),
                    'dl_offstToCarrier': 0,
                    'dl_subcarrierSpacing': f'kHz{self.config.rf.nr_scs}',
                    'dl_carrierBandwidth': self.config.rf.prb_count,
                    
                    'ul_frequencyBand': self.config.rf.nr_band,
                    'ul_absoluteFrequencyPointA': self._freq_to_arfcn(
                        self.config.rf.nr_dl_frequency_hz - 
                        (self.config.rf.prb_count * 12 * self.config.rf.nr_scs * 1000 / 2)
                    ),
                    'ul_offstToCarrier': 0,
                    'ul_subcarrierSpacing': f'kHz{self.config.rf.nr_scs}',
                    'ul_carrierBandwidth': self.config.rf.prb_count,
                    
                    'initialDLBWPlocationAndBandwidth': 28875,
                    'initialDLBWPsubcarrierSpacing': f'kHz{self.config.rf.nr_scs}',
                    'initialDLBWPcontrolResourceSetZero': 12,
                    'initialDLBWPsearchSpaceZero': 0,
                    
                    'initialULBWPlocationAndBandwidth': 28875,
                    'initialULBWPsubcarrierSpacing': f'kHz{self.config.rf.nr_scs}',
                    
                    'pMax': 20,
                    
                    'ssb_PositionsInBurst_PR': 2,
                    'ssb_PositionsInBurst_Bitmap': 1,
                    'ssb_periodicityServingCell': self.config.cell.ssb_periodicity_ms,
                    'dmrs_TypeA_Position': 0,
                    
                    'prach_ConfigurationIndex': self.config.cell.prach_config_index,
                    'prach_RootSequenceIndex': 1,
                    'prach_msg1_FDM': 0,
                    'prach_msg1_FrequencyStart': 0,
                    'zeroCorrelationZoneConfig': 13,
                    'preambleReceivedTargetPower': -96,
                    'preambleTransMax': 10,
                    'powerRampingStep': 4,
                    'ra_ResponseWindow': 5,
                    'msg3_DeltaPreamble': 1,
                    'p0_NominalWithGrant': -90,
                    'pucch_GroupHopping': 0,
                    'hoppingId': 40,
                    'p0_nominal': -90
                }],
                
                # SDAP configuration
                'sdap_config': {
                    'defaultDRB': True
                },
                
                # AMF configuration
                'amf_ip_address': [{
                    'ipv4': self.config.core.amf_ip,
                    'ipv6': '::1',
                    'active': 'yes',
                    'preference': 'ipv4'
                }],
                
                'NETWORK_INTERFACES': {
                    'GNB_INTERFACE_NAME_FOR_NG_AMF': 'eth0',
                    'GNB_IPV4_ADDRESS_FOR_NG_AMF': 
                        self.config.core.amf_ip.rsplit('.', 1)[0] + '.1/24',
                    'GNB_INTERFACE_NAME_FOR_NGU': 'eth0',
                    'GNB_IPV4_ADDRESS_FOR_NGU': 
                        self.config.core.upf_ip.rsplit('.', 1)[0] + '.1/24',
                    'GNB_PORT_FOR_S1U': 2152
                }
            }],
            
            'MACRLCs': [{
                'num_cc': 1,
                'tr_s_preference': 'local_L1',
                'tr_n_preference': 'local_RRC',
                'pusch_TargetSNRx10': 200,
                'pucch_TargetSNRx10': 150,
                'ulsch_max_frame_inactivity': 0
            }],
            
            'L1s': [{
                'num_cc': 1,
                'tr_n_preference': 'local_mac',
                'prach_dtx_threshold': 120,
                'pucch0_dtx_threshold': 150,
                'ofdm_offset_divisor': 8
            }],
            
            'RUs': [{
                'local_rf': 'yes',
                'nb_tx': self.config.rf.num_tx_antennas,
                'nb_rx': self.config.rf.num_rx_antennas,
                'att_tx': 0,
                'att_rx': 0,
                'bands': [self.config.rf.nr_band],
                'max_pdschReferenceSignalPower': self.config.cell.pdsch_ref_signal_power,
                'max_rxgain': int(self.config.rf.rx_gain),
                'eNB_instances': [0],
                'sdr_addrs': self._get_sdr_args()
            }],
            
            'THREAD_STRUCT': {
                'parallel_config': 'PARALLEL_SINGLE_THREAD',
                'worker_config': 'WORKER_ENABLE'
            },
            
            'log_config': {
                'global_log_level': 'info',
                'hw_log_level': 'info',
                'phy_log_level': 'info',
                'mac_log_level': 'info',
                'rlc_log_level': 'info',
                'pdcp_log_level': 'info',
                'rrc_log_level': 'info',
                'ngap_log_level': 'info'
            }
        }
        
        # Apply stealth restrictions
        if self.stealth_system:
            max_gain = self.stealth_system.get_max_tx_gain()
            if gnb_config['RUs'][0]['att_tx'] < (60 - max_gain):
                gnb_config['RUs'][0]['att_tx'] = int(60 - max_gain)
        
        config_file = os.path.join(self._temp_dir, 'gnb.conf')
        
        # Write as OAI config format (custom format, not pure YAML)
        with open(config_file, 'w') as f:
            self._write_oai_config(f, gnb_config)
        
        return config_file
    
    def generate_nr_ue_config(self, imsi: str = "001010000000001",
                               key: str = "fec86ba6eb707ed08905757b1bb44b8f",
                               opc: str = "C42449363BBAD02B66D16BC975D77CC1") -> str:
        """Generate NR UE configuration file"""
        ue_config = {
            'uicc0': {
                'imsi': imsi,
                'key': key,
                'opc': opc,
                'dnn': self.config.core.dnn,
                'nssai_sst': 1,
                'nssai_sd': 16777215
            },
            
            'sa': 1,
            'rfsimulator': {
                'serveraddr': 'server'
            },
            
            'log_config': {
                'global_log_level': 'info',
                'phy_log_level': 'info',
                'mac_log_level': 'info',
                'rlc_log_level': 'info',
                'pdcp_log_level': 'info',
                'rrc_log_level': 'info',
                'nas_log_level': 'info'
            },
            
            'RUs': [{
                'local_rf': 'yes',
                'nb_tx': self.config.rf.num_tx_antennas,
                'nb_rx': self.config.rf.num_rx_antennas,
                'att_tx': 10,
                'att_rx': 0,
                'bands': [self.config.rf.nr_band],
                'max_rxgain': int(self.config.rf.rx_gain),
                'sdr_addrs': self._get_sdr_args()
            }]
        }
        
        config_file = os.path.join(self._temp_dir, 'nr_ue.conf')
        
        with open(config_file, 'w') as f:
            self._write_oai_config(f, ue_config)
        
        return config_file
    
    def generate_core_docker_compose(self) -> str:
        """Generate Docker Compose file for 5G Core"""
        compose = {
            'version': '3.8',
            'services': {
                'mysql': {
                    'container_name': 'mysql',
                    'image': 'mysql:8.0',
                    'environment': {
                        'MYSQL_DATABASE': 'oai_db',
                        'MYSQL_USER': 'oai',
                        'MYSQL_PASSWORD': 'oai_password',
                        'MYSQL_ROOT_PASSWORD': 'root_password'
                    },
                    'volumes': [
                        './mysql/init.sql:/docker-entrypoint-initdb.d/init.sql'
                    ],
                    'healthcheck': {
                        'test': ['CMD', 'mysqladmin', 'ping', '-h', 'localhost'],
                        'interval': '10s',
                        'timeout': '5s',
                        'retries': 5
                    },
                    'networks': ['core_network']
                },
                
                'oai-nrf': {
                    'container_name': 'oai-nrf',
                    'image': 'oaisoftwarealliance/oai-nrf:v1.5.1',
                    'environment': {
                        'NRF_INTERFACE_NAME_FOR_SBI': 'eth0',
                        'NRF_INTERFACE_PORT_FOR_SBI': 80,
                        'NRF_INTERFACE_HTTP2_PORT_FOR_SBI': 8080,
                        'NRF_API_VERSION': 'v1'
                    },
                    'networks': {
                        'core_network': {
                            'ipv4_address': self.config.core.nrf_ip
                        }
                    }
                },
                
                'oai-amf': {
                    'container_name': 'oai-amf',
                    'image': 'oaisoftwarealliance/oai-amf:v1.5.1',
                    'depends_on': ['mysql', 'oai-nrf'],
                    'environment': {
                        'INSTANCE': 0,
                        'PID_DIRECTORY': '/var/run',
                        'MCC': self.config.cell.mcc,
                        'MNC': self.config.cell.mnc,
                        'REGION_ID': 128,
                        'AMF_SET_ID': 1,
                        'SERVED_GUAMI_MCC_0': self.config.cell.mcc,
                        'SERVED_GUAMI_MNC_0': self.config.cell.mnc,
                        'SERVED_GUAMI_REGION_ID_0': 128,
                        'SERVED_GUAMI_AMF_SET_ID_0': 1,
                        'SERVED_GUAMI_MCC_1': self.config.cell.mcc,
                        'SERVED_GUAMI_MNC_1': self.config.cell.mnc,
                        'SERVED_GUAMI_REGION_ID_1': 128,
                        'SERVED_GUAMI_AMF_SET_ID_1': 1,
                        'PLMN_SUPPORT_MCC': self.config.cell.mcc,
                        'PLMN_SUPPORT_MNC': self.config.cell.mnc,
                        'PLMN_SUPPORT_TAC': self.config.cell.tac,
                        'SST_0': 1,
                        'SD_0': '0xffffff',
                        'AMF_INTERFACE_NAME_FOR_NGAP': 'eth0',
                        'AMF_INTERFACE_NAME_FOR_N11': 'eth0',
                        'SMF_INSTANCE_ID_0': 1,
                        'SMF_FQDN_0': 'oai-smf',
                        'SMF_IPV4_ADDR_0': self.config.core.smf_ip,
                        'SMF_HTTP_VERSION_0': 'v1',
                        'NRF_IPV4_ADDRESS': self.config.core.nrf_ip,
                        'NRF_PORT': self.config.core.nrf_port,
                        'NRF_API_VERSION': 'v1',
                        'AUSF_IPV4_ADDRESS': '0.0.0.0',
                        'AUSF_PORT': 8080,
                        'AUSF_API_VERSION': 'v1',
                        'NF_REGISTRATION': 'yes',
                        'SMF_SELECTION': 'yes',
                        'USE_FQDN_DNS': 'yes',
                        'MYSQL_SERVER': '192.168.70.131',
                        'MYSQL_USER': 'oai',
                        'MYSQL_PASS': 'oai_password',
                        'MYSQL_DB': 'oai_db',
                        'OPERATOR_KEY': 'c42449363bbad02b66d16bc975d77cc1'
                    },
                    'networks': {
                        'core_network': {
                            'ipv4_address': self.config.core.amf_ip
                        }
                    }
                },
                
                'oai-smf': {
                    'container_name': 'oai-smf',
                    'image': 'oaisoftwarealliance/oai-smf:v1.5.1',
                    'depends_on': ['oai-nrf', 'oai-amf'],
                    'environment': {
                        'INSTANCE': 0,
                        'PID_DIRECTORY': '/var/run',
                        'SMF_INTERFACE_NAME_FOR_N4': 'eth0',
                        'SMF_INTERFACE_NAME_FOR_SBI': 'eth0',
                        'SMF_INTERFACE_PORT_FOR_SBI': 80,
                        'SMF_INTERFACE_HTTP2_PORT_FOR_SBI': 8080,
                        'SMF_API_VERSION': 'v1',
                        'DEFAULT_DNS_IPV4_ADDRESS': '8.8.8.8',
                        'DEFAULT_DNS_SEC_IPV4_ADDRESS': '8.8.4.4',
                        'AMF_IPV4_ADDRESS': self.config.core.amf_ip,
                        'AMF_PORT': 8080,
                        'AMF_API_VERSION': 'v1',
                        'UDM_IPV4_ADDRESS': '0.0.0.0',
                        'UDM_PORT': 8080,
                        'UDM_API_VERSION': 'v2',
                        'UPF_IPV4_ADDRESS': self.config.core.upf_ip,
                        'UPF_FQDN_0': 'oai-spgwu',
                        'NRF_IPV4_ADDRESS': self.config.core.nrf_ip,
                        'NRF_PORT': self.config.core.nrf_port,
                        'NRF_API_VERSION': 'v1',
                        'REGISTER_NRF': 'yes',
                        'DISCOVER_UPF': 'yes',
                        'USE_FQDN_DNS': 'yes',
                        'DNN_NI0': self.config.core.dnn,
                        'TYPE0': 'IPv4',
                        'DNN_RANGE0': '12.1.1.2 - 12.1.1.128',
                        'NSSAI_SST0': 1,
                        'SESSION_AMBR_UL0': '200Mbps',
                        'SESSION_AMBR_DL0': '400Mbps'
                    },
                    'networks': {
                        'core_network': {
                            'ipv4_address': self.config.core.smf_ip
                        }
                    }
                },
                
                'oai-spgwu': {
                    'container_name': 'oai-spgwu',
                    'image': 'oaisoftwarealliance/oai-spgwu-tiny:v1.5.1',
                    'depends_on': ['oai-nrf', 'oai-smf'],
                    'cap_add': ['NET_ADMIN', 'SYS_ADMIN'],
                    'cap_drop': ['ALL'],
                    'privileged': True,
                    'environment': {
                        'INSTANCE': 0,
                        'PID_DIRECTORY': '/var/run',
                        'SGW_INTERFACE_NAME_FOR_S1U_S12_S4_UP': 'eth0',
                        'SGW_INTERFACE_NAME_FOR_SX': 'eth0',
                        'PGW_INTERFACE_NAME_FOR_SGI': 'eth0',
                        'NETWORK_UE_NAT_OPTION': 'yes',
                        'NETWORK_UE_IP': '12.1.1.0/24',
                        'SPGWC0_IP_ADDRESS': self.config.core.smf_ip,
                        'BYPASS_UL_PFCP_RULES': 'no',
                        'MCC': self.config.cell.mcc,
                        'MNC': self.config.cell.mnc,
                        'MNC03': f'{int(self.config.cell.mnc):03d}',
                        'TAC': self.config.cell.tac,
                        'GW_ID': 1,
                        'REALM': '3gppnetwork.org',
                        'ENABLE_5G_FEATURES': 'yes',
                        'REGISTER_NRF': 'yes',
                        'USE_FQDN_NRF': 'yes',
                        'UPF_FQDN_5G': 'oai-spgwu',
                        'NRF_IPV4_ADDRESS': self.config.core.nrf_ip,
                        'NRF_PORT': self.config.core.nrf_port,
                        'NRF_API_VERSION': 'v1',
                        'NSSAI_SST_0': 1,
                        'NSSAI_SD_0': '0xffffff',
                        'DNN_0': self.config.core.dnn
                    },
                    'networks': {
                        'core_network': {
                            'ipv4_address': self.config.core.upf_ip
                        }
                    }
                }
            },
            
            'networks': {
                'core_network': {
                    'driver': 'bridge',
                    'ipam': {
                        'config': [{
                            'subnet': '192.168.70.0/24',
                            'gateway': '192.168.70.1'
                        }]
                    }
                }
            }
        }
        
        compose_file = os.path.join(self._temp_dir, 'docker-compose.yaml')
        
        with open(compose_file, 'w') as f:
            yaml.dump(compose, f, default_flow_style=False)
        
        return compose_file
    
    def _freq_to_arfcn(self, freq_hz: float) -> int:
        """Convert frequency to NR-ARFCN"""
        freq_mhz = freq_hz / 1e6
        
        if freq_mhz < 3000:
            # FR1 range 1
            return int((freq_mhz - 0) / 0.005)
        elif freq_mhz < 24250:
            # FR1 range 2
            return int(600000 + (freq_mhz - 3000) / 0.015)
        else:
            # FR2
            return int(2016667 + (freq_mhz - 24250.08) / 0.060)
    
    def _get_sdr_args(self) -> str:
        """Get SDR arguments string"""
        if self.config.rf.sdr_type == 'bladerf':
            args = f"type=bladerf"
            if self.config.rf.device_args:
                args += f",{self.config.rf.device_args}"
            return args
        elif self.config.rf.sdr_type == 'uhd':
            return f"type=b200,{self.config.rf.device_args}"
        else:
            return self.config.rf.device_args
    
    def _write_oai_config(self, f, config: Dict, indent: int = 0):
        """Write OAI-style configuration file"""
        prefix = "  " * indent
        
        for key, value in config.items():
            if isinstance(value, dict):
                f.write(f"{prefix}{key} = {{\n")
                self._write_oai_config(f, value, indent + 1)
                f.write(f"{prefix}}};\n")
            elif isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], dict):
                    f.write(f"{prefix}{key} = (\n")
                    for i, item in enumerate(value):
                        f.write(f"{prefix}  {{\n")
                        self._write_oai_config(f, item, indent + 2)
                        f.write(f"{prefix}  }}")
                        if i < len(value) - 1:
                            f.write(",")
                        f.write("\n")
                    f.write(f"{prefix});\n")
                else:
                    f.write(f"{prefix}{key} = [{', '.join(str(v) for v in value)}];\n")
            elif isinstance(value, str):
                f.write(f'{prefix}{key} = "{value}";\n')
            elif isinstance(value, bool):
                f.write(f"{prefix}{key} = {str(value).lower()};\n")
            else:
                f.write(f"{prefix}{key} = {value};\n")


# ============================================================================
# OAI Controller
# ============================================================================

class OAIController:
    """
    Main controller for OpenAirInterface integration
    
    Provides unified interface for managing OAI components
    with stealth awareness and AI control integration.
    """
    
    def __init__(self, config: OAIConfig = None, stealth_system=None):
        self.config = config or OAIConfig()
        self.stealth_system = stealth_system
        
        self._lock = threading.RLock()
        self._processes: Dict[OAIComponent, OAIProcess] = {}
        self._config_generator: Optional[OAIConfigGenerator] = None
        
        # Docker management
        self._core_running = False
        
        # Event callbacks
        self._callbacks: Dict[str, List[Callable]] = {}
        
        # Installation status
        self._installed = self._check_installation()
    
    @property
    def is_installed(self) -> bool:
        return self._installed
    
    def _check_installation(self) -> bool:
        """Check OAI installation"""
        # Check for gNB binary
        gnb_paths = [
            os.path.join(self.config.oai_path, 'cmake_targets/ran_build/build/nr-softmodem'),
            '/usr/local/bin/nr-softmodem',
            shutil.which('nr-softmodem')
        ]
        
        for path in gnb_paths:
            if path and os.path.isfile(path) and os.access(path, os.X_OK):
                return True
        
        # Check for Docker-based deployment
        if shutil.which('docker') and shutil.which('docker-compose'):
            return True
        
        return False
    
    def _get_executable(self, component: OAIComponent) -> str:
        """Get executable path for component"""
        exe_map = {
            OAIComponent.GNB: 'nr-softmodem',
            OAIComponent.ENB: 'lte-softmodem',
            OAIComponent.NR_UE: 'nr-uesoftmodem',
            OAIComponent.LTE_UE: 'lte-uesoftmodem'
        }
        
        exe_name = exe_map.get(component)
        if not exe_name:
            return None
        
        # Check PATH
        if shutil.which(exe_name):
            return exe_name
        
        # Check OAI build path
        build_path = os.path.join(
            self.config.oai_path,
            'cmake_targets/ran_build/build',
            exe_name
        )
        if os.path.isfile(build_path):
            return build_path
        
        return exe_name
    
    def initialize(self) -> bool:
        """Initialize OAI controller"""
        logger.info("Initializing OpenAirInterface controller")
        
        if not self._installed:
            logger.warning("OAI not installed - running in simulation mode")
        
        # Create config generator
        self._config_generator = OAIConfigGenerator(
            self.config, self.stealth_system
        )
        
        # Create directories
        os.makedirs(self.config.log_path, exist_ok=True)
        os.makedirs(self.config.config_path, exist_ok=True)
        
        return True
    
    def shutdown(self):
        """Shutdown all OAI components"""
        logger.info("Shutting down OAI controller")
        
        # Stop RAN components
        for component in list(self._processes.keys()):
            self.stop_component(component)
        
        # Stop core network
        if self._core_running:
            self.stop_core()
        
        # Cleanup
        if self._config_generator:
            self._config_generator.cleanup()
    
    # ========================================================================
    # RAN Component Management
    # ========================================================================
    
    def start_gnb(self, sa_mode: bool = True) -> bool:
        """
        Start 5G gNodeB
        
        Args:
            sa_mode: Standalone mode (True) or NSA (False)
        """
        if not self._installed:
            logger.info("[SIMULATED] Starting OAI gNB")
            return True
        
        config_file = self._config_generator.generate_gnb_config()
        
        process = OAIProcess(
            component=OAIComponent.GNB,
            executable=self._get_executable(OAIComponent.GNB),
            config_file=config_file,
            stealth_system=self.stealth_system
        )
        
        extra_args = ['--sa'] if sa_mode else []
        extra_args.extend([
            '--continuous-tx', '1',
            '-E',  # 3-quarter sampling
        ])
        
        process.set_state_callback(
            lambda s: self._emit_event('gnb_state_change', {'state': s.name})
        )
        
        if process.start(extra_args):
            self._processes[OAIComponent.GNB] = process
            return True
        
        return False
    
    def start_nr_ue(self, imsi: str = None, key: str = None) -> bool:
        """Start NR UE"""
        if not self._installed:
            logger.info("[SIMULATED] Starting OAI NR UE")
            return True
        
        config_file = self._config_generator.generate_nr_ue_config(
            imsi=imsi or "001010000000001",
            key=key or "fec86ba6eb707ed08905757b1bb44b8f"
        )
        
        process = OAIProcess(
            component=OAIComponent.NR_UE,
            executable=self._get_executable(OAIComponent.NR_UE),
            config_file=config_file,
            stealth_system=self.stealth_system
        )
        
        extra_args = [
            '-E',
            '--sa',
            '--numerology', str(self.config.rf.nr_scs // 15 - 1)
        ]
        
        process.set_state_callback(
            lambda s: self._emit_event('ue_state_change', {'state': s.name})
        )
        
        if process.start(extra_args):
            self._processes[OAIComponent.NR_UE] = process
            return True
        
        return False
    
    def stop_component(self, component: OAIComponent) -> bool:
        """Stop a specific component"""
        with self._lock:
            if component in self._processes:
                process = self._processes[component]
                result = process.stop()
                del self._processes[component]
                return result
        return True
    
    # ========================================================================
    # Core Network Management
    # ========================================================================
    
    def start_core(self) -> bool:
        """
        Start 5G Core Network using Docker
        """
        if self.config.deployment_mode != OAIDeploymentMode.DOCKER:
            logger.warning("Docker deployment mode not selected")
            return False
        
        if not shutil.which('docker-compose'):
            logger.error("docker-compose not found")
            return False
        
        logger.info("Starting 5G Core Network")
        
        # Generate docker-compose
        compose_file = self._config_generator.generate_core_docker_compose()
        
        try:
            result = subprocess.run(
                ['docker-compose', '-f', compose_file, 'up', '-d'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                self._core_running = True
                logger.info("5G Core Network started")
                return True
            else:
                logger.error(f"Failed to start core: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting core: {e}")
            return False
    
    def stop_core(self) -> bool:
        """Stop 5G Core Network"""
        if not self._core_running:
            return True
        
        logger.info("Stopping 5G Core Network")
        
        try:
            result = subprocess.run(
                ['docker-compose', 'down'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            self._core_running = False
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Error stopping core: {e}")
            return False
    
    def start_full_network(self, sa_mode: bool = True) -> bool:
        """Start complete 5G network (Core + gNB)"""
        logger.info("Starting full 5G network")
        
        # Start Core first
        if self.config.deployment_mode == OAIDeploymentMode.DOCKER:
            if not self.start_core():
                logger.error("Failed to start core network")
                return False
            
            time.sleep(30)  # Wait for core to initialize
        
        # Start gNB
        if not self.start_gnb(sa_mode=sa_mode):
            logger.error("Failed to start gNB")
            self.stop_core()
            return False
        
        logger.info("Full 5G network started")
        return True
    
    def stop_full_network(self):
        """Stop complete 5G network"""
        logger.info("Stopping full 5G network")
        
        self.stop_component(OAIComponent.NR_UE)
        self.stop_component(OAIComponent.GNB)
        time.sleep(2.0)
        self.stop_core()
    
    # ========================================================================
    # Configuration
    # ========================================================================
    
    def set_frequency(self, frequency_hz: float, band: int = None):
        """Set operating frequency"""
        self.config.rf.nr_dl_frequency_hz = frequency_hz
        self.config.rf.nr_ssb_frequency_hz = frequency_hz
        
        if band:
            self.config.rf.nr_band = band
    
    def set_tx_power(self, gain_db: float):
        """Set TX gain with stealth limits"""
        if self.stealth_system:
            max_gain = self.stealth_system.get_max_tx_gain()
            gain_db = min(gain_db, max_gain)
        
        self.config.rf.tx_gain = gain_db
        logger.info(f"TX gain set to {gain_db} dB")
    
    def set_bandwidth(self, prb_count: int):
        """Set bandwidth via PRB count"""
        self.config.rf.prb_count = prb_count
    
    # ========================================================================
    # Monitoring
    # ========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all components"""
        status = {
            'installed': self._installed,
            'core_running': self._core_running,
            'components': {}
        }
        
        for component, process in self._processes.items():
            status['components'][component.name] = process.get_status()
        
        return status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {
            'timestamp': time.time(),
            'gnb': {},
            'ue': {}
        }
        
        # Would parse metrics from OAI KPIs
        
        return metrics
    
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

def create_oai_controller(config: OAIConfig = None,
                          stealth_system=None) -> OAIController:
    """Create OAI controller instance"""
    controller = OAIController(config, stealth_system)
    controller.initialize()
    return controller
