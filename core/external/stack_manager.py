"""
RF Arsenal OS - External Stack Manager
Unified interface for srsRAN and OpenAirInterface integration

Provides:
- Automatic stack detection and selection
- Unified API for LTE/5G operations
- Stealth-aware operation
- AI control integration
"""

import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import IntEnum, auto

logger = logging.getLogger(__name__)


# ============================================================================
# Stack Types
# ============================================================================

class StackType(IntEnum):
    """Supported protocol stack types"""
    NATIVE = 0       # RF Arsenal native implementation
    SRSRAN = 1       # srsRAN suite
    OAI = 2          # OpenAirInterface
    AUTO = 3         # Automatic selection


class NetworkMode(IntEnum):
    """Network operation mode"""
    LTE = 0          # 4G LTE only
    NR_SA = 1        # 5G NR Standalone
    NR_NSA = 2       # 5G NR Non-Standalone
    AUTO = 3         # Best available


class ComponentRole(IntEnum):
    """Component role"""
    BASE_STATION = 0  # eNB/gNB
    UE = 1            # User Equipment
    CORE = 2          # EPC/5GC


# ============================================================================
# Stack Configuration
# ============================================================================

@dataclass
class StackConfig:
    """Stack configuration"""
    preferred_stack: StackType = StackType.AUTO
    network_mode: NetworkMode = NetworkMode.AUTO
    
    # Common RF settings
    frequency_hz: float = 2680e6
    bandwidth_mhz: float = 10.0
    tx_gain_db: float = 50.0
    rx_gain_db: float = 40.0
    
    # Cell settings
    cell_id: int = 1
    tac: int = 1
    mcc: str = "001"
    mnc: str = "01"
    
    # Features
    enable_pcap: bool = False
    enable_metrics: bool = True
    
    # Stealth
    stealth_mode: bool = True


# ============================================================================
# Stack Status
# ============================================================================

@dataclass
class StackStatus:
    """Stack operational status"""
    stack_type: StackType
    network_mode: NetworkMode
    is_running: bool
    base_station_active: bool
    core_active: bool
    connected_ues: int
    uptime: float
    error_message: Optional[str] = None


# ============================================================================
# External Stack Manager
# ============================================================================

class ExternalStackManager:
    """
    Unified manager for external protocol stacks
    
    Provides a single interface for controlling srsRAN and
    OpenAirInterface with automatic fallback and stealth support.
    """
    
    def __init__(self, config: StackConfig = None, stealth_system=None):
        """
        Initialize stack manager
        
        Args:
            config: Stack configuration
            stealth_system: Stealth system for emission control
        """
        self.config = config or StackConfig()
        self.stealth_system = stealth_system
        
        self._lock = threading.RLock()
        self._active_stack: Optional[StackType] = None
        self._network_mode: Optional[NetworkMode] = None
        
        # Controllers (lazy loaded)
        self._srsran_controller = None
        self._oai_controller = None
        
        # Stack availability
        self._stack_availability: Dict[StackType, bool] = {}
        
        # Event callbacks
        self._callbacks: Dict[str, List[Callable]] = {}
        
        # Detect available stacks
        self._detect_stacks()
    
    def _detect_stacks(self):
        """Detect available protocol stacks"""
        logger.info("Detecting available protocol stacks...")
        
        # Check srsRAN
        try:
            from .srsran.srsran_controller import SrsRANController, SrsRANConfig
            
            srsran_config = SrsRANConfig()
            self._srsran_controller = SrsRANController(
                config=srsran_config,
                stealth_system=self.stealth_system
            )
            self._stack_availability[StackType.SRSRAN] = self._srsran_controller.is_installed
            
            if self._srsran_controller.is_installed:
                logger.info("srsRAN detected and available")
            else:
                logger.warning(
                    "DEPENDENCY REQUIRED: srsRAN not installed. "
                    "Install with: ./install/install_srsran.sh"
                )
                self._stack_availability[StackType.SRSRAN] = False  # No simulation
                
        except ImportError as e:
            logger.warning(f"srsRAN module not available: {e}")
            self._stack_availability[StackType.SRSRAN] = False
        
        # Check OpenAirInterface
        try:
            from .openairinterface.oai_controller import OAIController, OAIConfig
            
            oai_config = OAIConfig()
            self._oai_controller = OAIController(
                config=oai_config,
                stealth_system=self.stealth_system
            )
            self._stack_availability[StackType.OAI] = self._oai_controller.is_installed
            
            if self._oai_controller.is_installed:
                logger.info("OpenAirInterface detected and available")
            else:
                logger.warning(
                    "DEPENDENCY REQUIRED: OpenAirInterface not installed. "
                    "Install with: ./install/install_oai.sh"
                )
                self._stack_availability[StackType.OAI] = False  # No simulation
                
        except ImportError as e:
            logger.warning(f"OAI module not available: {e}")
            self._stack_availability[StackType.OAI] = False
        
        # Native stack always available
        self._stack_availability[StackType.NATIVE] = True
        
        logger.info(f"Stack availability: {self._stack_availability}")
    
    def _select_stack(self, preferred: StackType = None) -> StackType:
        """Select best available stack"""
        preferred = preferred or self.config.preferred_stack
        
        if preferred == StackType.AUTO:
            # Priority: OAI (5G), srsRAN (4G/5G), Native
            if self._stack_availability.get(StackType.OAI):
                return StackType.OAI
            elif self._stack_availability.get(StackType.SRSRAN):
                return StackType.SRSRAN
            else:
                return StackType.NATIVE
        
        if self._stack_availability.get(preferred):
            return preferred
        
        # Fallback
        logger.warning(f"Preferred stack {preferred.name} not available, using fallback")
        return self._select_stack(StackType.AUTO)
    
    def _select_network_mode(self, preferred: NetworkMode = None) -> NetworkMode:
        """Select best network mode"""
        preferred = preferred or self.config.network_mode
        
        if preferred == NetworkMode.AUTO:
            # Prefer 5G SA if using OAI, otherwise LTE
            if self._active_stack == StackType.OAI:
                return NetworkMode.NR_SA
            else:
                return NetworkMode.LTE
        
        return preferred
    
    # ========================================================================
    # Initialization
    # ========================================================================
    
    def initialize(self, stack: StackType = None, 
                   mode: NetworkMode = None) -> bool:
        """
        Initialize selected stack
        
        Args:
            stack: Stack type to initialize
            mode: Network mode
        
        Returns:
            True on success
        """
        with self._lock:
            # Select stack
            self._active_stack = self._select_stack(stack)
            self._network_mode = self._select_network_mode(mode)
            
            logger.info(f"Initializing {self._active_stack.name} stack "
                       f"in {self._network_mode.name} mode")
            
            # Initialize selected controller
            if self._active_stack == StackType.SRSRAN:
                if self._srsran_controller:
                    self._apply_config_to_srsran()
                    return self._srsran_controller.initialize()
            
            elif self._active_stack == StackType.OAI:
                if self._oai_controller:
                    self._apply_config_to_oai()
                    return self._oai_controller.initialize()
            
            elif self._active_stack == StackType.NATIVE:
                logger.info("Using native RF Arsenal stack")
                return True
            
            return False
    
    def shutdown(self):
        """Shutdown active stack"""
        with self._lock:
            logger.info("Shutting down stack manager")
            
            if self._srsran_controller:
                self._srsran_controller.shutdown()
            
            if self._oai_controller:
                self._oai_controller.shutdown()
            
            self._active_stack = None
            self._network_mode = None
    
    def _apply_config_to_srsran(self):
        """Apply configuration to srsRAN controller"""
        if not self._srsran_controller:
            return
        
        ctrl = self._srsran_controller
        
        # Apply RF settings
        ctrl.config.rf.tx_gain = self.config.tx_gain_db
        ctrl.config.rf.rx_gain = self.config.rx_gain_db
        
        # Apply cell settings
        ctrl.config.cell.cell_id = self.config.cell_id
        ctrl.config.cell.tac = self.config.tac
        ctrl.config.cell.mcc = self.config.mcc
        ctrl.config.cell.mnc = self.config.mnc
        
        # Bandwidth to PRB mapping
        bw_prb_map = {5: 25, 10: 50, 15: 75, 20: 100}
        bw_mhz = int(self.config.bandwidth_mhz)
        ctrl.config.rf.nof_prb = bw_prb_map.get(bw_mhz, 50)
        
        # PCAP
        ctrl.config.pcap_enable = self.config.enable_pcap
    
    def _apply_config_to_oai(self):
        """Apply configuration to OAI controller"""
        if not self._oai_controller:
            return
        
        ctrl = self._oai_controller
        
        # Apply RF settings
        ctrl.config.rf.tx_gain = self.config.tx_gain_db
        ctrl.config.rf.rx_gain = self.config.rx_gain_db
        ctrl.config.rf.nr_dl_frequency_hz = self.config.frequency_hz
        ctrl.config.rf.nr_ssb_frequency_hz = self.config.frequency_hz
        
        # Apply cell settings
        ctrl.config.cell.cell_id = self.config.cell_id
        ctrl.config.cell.tac = self.config.tac
        ctrl.config.cell.mcc = self.config.mcc
        ctrl.config.cell.mnc = self.config.mnc
    
    # ========================================================================
    # Network Operations
    # ========================================================================
    
    def start_network(self, include_core: bool = True) -> bool:
        """
        Start cellular network
        
        Args:
            include_core: Include core network (EPC/5GC)
        
        Returns:
            True on success
        """
        if not self._active_stack:
            logger.error("Stack not initialized")
            return False
        
        # Check stealth
        if self.stealth_system:
            if not self.stealth_system.check_emission_allowed():
                logger.warning("Network start blocked by stealth system")
                return False
        
        logger.info(f"Starting network ({self._active_stack.name})")
        
        if self._active_stack == StackType.SRSRAN:
            if include_core:
                return self._srsran_controller.start_full_network()
            else:
                return self._srsran_controller.start_enb()
        
        elif self._active_stack == StackType.OAI:
            sa_mode = self._network_mode == NetworkMode.NR_SA
            if include_core:
                return self._oai_controller.start_full_network(sa_mode=sa_mode)
            else:
                return self._oai_controller.start_gnb(sa_mode=sa_mode)
        
        elif self._active_stack == StackType.NATIVE:
            logger.info("[NATIVE] Network simulation started")
            return True
        
        return False
    
    def stop_network(self):
        """Stop cellular network"""
        logger.info("Stopping network")
        
        if self._active_stack == StackType.SRSRAN:
            self._srsran_controller.stop_full_network()
        
        elif self._active_stack == StackType.OAI:
            self._oai_controller.stop_full_network()
        
        elif self._active_stack == StackType.NATIVE:
            logger.info("[NATIVE] Network simulation stopped")
    
    def start_ue(self, imsi: str = None, key: str = None) -> bool:
        """
        Start UE
        
        Args:
            imsi: IMSI for the UE
            key: Authentication key
        
        Returns:
            True on success
        """
        logger.info("Starting UE")
        
        if self._active_stack == StackType.SRSRAN:
            return self._srsran_controller.start_ue(imsi=imsi, key=key)
        
        elif self._active_stack == StackType.OAI:
            return self._oai_controller.start_nr_ue(imsi=imsi, key=key)
        
        elif self._active_stack == StackType.NATIVE:
            logger.info("[NATIVE] UE simulation started")
            return True
        
        return False
    
    def stop_ue(self):
        """Stop UE"""
        logger.info("Stopping UE")
        
        if self._active_stack == StackType.SRSRAN:
            from .srsran.srsran_controller import SrsRANComponent
            self._srsran_controller.stop_component(SrsRANComponent.UE)
        
        elif self._active_stack == StackType.OAI:
            from .openairinterface.oai_controller import OAIComponent
            self._oai_controller.stop_component(OAIComponent.NR_UE)
    
    # ========================================================================
    # Configuration
    # ========================================================================
    
    def set_frequency(self, frequency_hz: float):
        """Set operating frequency"""
        self.config.frequency_hz = frequency_hz
        
        if self._active_stack == StackType.SRSRAN:
            # Convert to EARFCN
            earfcn = self._freq_to_earfcn(frequency_hz)
            self._srsran_controller.set_frequency(earfcn)
        
        elif self._active_stack == StackType.OAI:
            self._oai_controller.set_frequency(frequency_hz)
        
        logger.info(f"Frequency set to {frequency_hz/1e6:.1f} MHz")
    
    def set_tx_power(self, gain_db: float):
        """Set TX power (gain)"""
        if self.stealth_system:
            max_gain = self.stealth_system.get_max_tx_gain()
            gain_db = min(gain_db, max_gain)
        
        self.config.tx_gain_db = gain_db
        
        if self._active_stack == StackType.SRSRAN:
            self._srsran_controller.set_tx_power(gain_db)
        
        elif self._active_stack == StackType.OAI:
            self._oai_controller.set_tx_power(gain_db)
        
        logger.info(f"TX power set to {gain_db} dB")
    
    def set_bandwidth(self, bandwidth_mhz: float):
        """Set bandwidth"""
        self.config.bandwidth_mhz = bandwidth_mhz
        
        # PRB mapping for different bandwidths
        bw_prb_map = {
            1.4: 6, 3: 15, 5: 25, 10: 50, 15: 75, 20: 100,
            40: 106, 50: 133, 100: 273  # NR bandwidths
        }
        
        prb_count = bw_prb_map.get(bandwidth_mhz, 50)
        
        if self._active_stack == StackType.SRSRAN:
            self._srsran_controller.set_bandwidth(prb_count)
        
        elif self._active_stack == StackType.OAI:
            self._oai_controller.set_bandwidth(prb_count)
        
        logger.info(f"Bandwidth set to {bandwidth_mhz} MHz ({prb_count} PRBs)")
    
    def _freq_to_earfcn(self, freq_hz: float) -> int:
        """Convert frequency to EARFCN (simplified)"""
        freq_mhz = freq_hz / 1e6
        
        # Band 7 (2620-2690 MHz DL)
        if 2620 <= freq_mhz <= 2690:
            return int((freq_mhz - 2620) * 10) + 2750
        
        # Band 3 (1805-1880 MHz DL)
        if 1805 <= freq_mhz <= 1880:
            return int((freq_mhz - 1805) * 10) + 1200
        
        # Band 1 (2110-2170 MHz DL)
        if 2110 <= freq_mhz <= 2170:
            return int((freq_mhz - 2110) * 10) + 0
        
        # Default to band 7
        return 3350
    
    # ========================================================================
    # Status and Monitoring
    # ========================================================================
    
    def get_status(self) -> StackStatus:
        """Get comprehensive status"""
        status = StackStatus(
            stack_type=self._active_stack or StackType.NATIVE,
            network_mode=self._network_mode or NetworkMode.LTE,
            is_running=False,
            base_station_active=False,
            core_active=False,
            connected_ues=0,
            uptime=0.0
        )
        
        if self._active_stack == StackType.SRSRAN:
            srs_status = self._srsran_controller.get_status()
            status.is_running = bool(srs_status.get('components'))
            status.base_station_active = 'ENB' in [
                c for c in srs_status.get('components', {}).keys()
            ]
            
        elif self._active_stack == StackType.OAI:
            oai_status = self._oai_controller.get_status()
            status.is_running = oai_status.get('core_running', False) or \
                               bool(oai_status.get('components'))
            status.base_station_active = 'GNB' in [
                c for c in oai_status.get('components', {}).keys()
            ]
            status.core_active = oai_status.get('core_running', False)
        
        return status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {
            'stack': self._active_stack.name if self._active_stack else 'none',
            'mode': self._network_mode.name if self._network_mode else 'none'
        }
        
        if self._active_stack == StackType.SRSRAN:
            metrics.update(self._srsran_controller.get_metrics())
        
        elif self._active_stack == StackType.OAI:
            metrics.update(self._oai_controller.get_metrics())
        
        return metrics
    
    def get_connected_ues(self) -> List[Dict[str, Any]]:
        """Get list of connected UEs"""
        if self._active_stack == StackType.SRSRAN:
            return self._srsran_controller.get_connected_ues()
        
        elif self._active_stack == StackType.OAI:
            # OAI would need KPI parsing
            return []
        
        return []
    
    def get_available_stacks(self) -> Dict[str, bool]:
        """Get available stacks"""
        return {
            stack.name: available 
            for stack, available in self._stack_availability.items()
        }
    
    # ========================================================================
    # AI Integration
    # ========================================================================
    
    def execute_ai_command(self, command: str) -> Dict[str, Any]:
        """
        Execute AI command
        
        Supports commands like:
        - "start lte network"
        - "start 5g network"
        - "set frequency 2680 mhz"
        - "set power 30 db"
        - "stop network"
        """
        command = command.lower().strip()
        result = {'success': False, 'message': ''}
        
        try:
            if 'start' in command:
                if '5g' in command or 'nr' in command:
                    self.initialize(mode=NetworkMode.NR_SA)
                    success = self.start_network()
                    result['success'] = success
                    result['message'] = '5G network started' if success else 'Failed to start'
                
                elif 'lte' in command or '4g' in command:
                    self.initialize(mode=NetworkMode.LTE)
                    success = self.start_network()
                    result['success'] = success
                    result['message'] = 'LTE network started' if success else 'Failed to start'
                
                elif 'ue' in command:
                    success = self.start_ue()
                    result['success'] = success
                    result['message'] = 'UE started' if success else 'Failed to start'
            
            elif 'stop' in command:
                if 'ue' in command:
                    self.stop_ue()
                else:
                    self.stop_network()
                result['success'] = True
                result['message'] = 'Stopped'
            
            elif 'frequency' in command or 'freq' in command:
                # Extract frequency value
                import re
                match = re.search(r'(\d+(?:\.\d+)?)\s*(mhz|ghz)', command)
                if match:
                    value = float(match.group(1))
                    unit = match.group(2)
                    if unit == 'ghz':
                        value *= 1e9
                    else:
                        value *= 1e6
                    self.set_frequency(value)
                    result['success'] = True
                    result['message'] = f'Frequency set to {value/1e6:.1f} MHz'
            
            elif 'power' in command or 'gain' in command:
                import re
                match = re.search(r'(\d+(?:\.\d+)?)\s*db', command)
                if match:
                    value = float(match.group(1))
                    self.set_tx_power(value)
                    result['success'] = True
                    result['message'] = f'TX power set to {value} dB'
            
            elif 'bandwidth' in command or 'bw' in command:
                import re
                match = re.search(r'(\d+(?:\.\d+)?)\s*mhz', command)
                if match:
                    value = float(match.group(1))
                    self.set_bandwidth(value)
                    result['success'] = True
                    result['message'] = f'Bandwidth set to {value} MHz'
            
            elif 'status' in command:
                status = self.get_status()
                result['success'] = True
                result['message'] = f'Stack: {status.stack_type.name}, ' \
                                   f'Mode: {status.network_mode.name}, ' \
                                   f'Running: {status.is_running}'
            
            else:
                result['message'] = 'Unknown command'
        
        except Exception as e:
            result['message'] = f'Error: {str(e)}'
            logger.error(f"AI command error: {e}")
        
        return result
    
    # ========================================================================
    # Events
    # ========================================================================
    
    def register_callback(self, event: str, callback: Callable):
        """Register event callback"""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)


# ============================================================================
# Factory Functions
# ============================================================================

def create_stack_manager(config: StackConfig = None,
                         stealth_system=None) -> ExternalStackManager:
    """Create external stack manager instance"""
    return ExternalStackManager(config=config, stealth_system=stealth_system)


def quick_start_lte(stealth_system=None) -> ExternalStackManager:
    """Quick start LTE network"""
    manager = create_stack_manager(stealth_system=stealth_system)
    manager.initialize(mode=NetworkMode.LTE)
    return manager


def quick_start_5g(stealth_system=None) -> ExternalStackManager:
    """Quick start 5G network"""
    config = StackConfig(
        preferred_stack=StackType.OAI,
        network_mode=NetworkMode.NR_SA
    )
    manager = create_stack_manager(config=config, stealth_system=stealth_system)
    manager.initialize()
    return manager
