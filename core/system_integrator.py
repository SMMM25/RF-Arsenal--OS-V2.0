"""
RF Arsenal OS - System Integrator
Production-grade integration layer for all RF modules

Provides unified interface for:
- Hardware control (BladeRF xA9)
- Protocol stacks (2G/3G/4G/5G)
- DSP pipeline
- Stealth operations
- AI control interface
"""

import logging
import threading
import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import IntEnum, auto

logger = logging.getLogger(__name__)


# ============================================================================
# System States
# ============================================================================

class SystemState(IntEnum):
    """System operational state"""
    OFFLINE = 0
    INITIALIZING = 1
    READY = 2
    OPERATING = 3
    STEALTH = 4
    EMERGENCY = 5
    SHUTDOWN = 6


class ModuleState(IntEnum):
    """Module operational state"""
    UNLOADED = 0
    LOADING = 1
    READY = 2
    ACTIVE = 3
    ERROR = 4
    DISABLED = 5


# ============================================================================
# Module Descriptors
# ============================================================================

@dataclass
class ModuleDescriptor:
    """Module descriptor"""
    name: str
    version: str
    category: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    stealth_compatible: bool = True
    ai_controllable: bool = True
    state: ModuleState = ModuleState.UNLOADED


# ============================================================================
# System Configuration
# ============================================================================

@dataclass
class SystemConfig:
    """System-wide configuration"""
    # Hardware
    hardware_enabled: bool = True
    hardware_simulated: bool = True
    bladerf_serial: Optional[str] = None
    
    # Stealth
    stealth_mode: bool = True
    max_tx_power_dbm: float = -30.0
    emission_timeout_s: float = 5.0
    
    # Network
    tor_enabled: bool = True
    vpn_enabled: bool = False
    mac_randomization: bool = True
    
    # AI
    ai_enabled: bool = True
    voice_enabled: bool = False
    
    # Security
    ram_only_mode: bool = True
    secure_delete: bool = True
    panic_enabled: bool = True
    
    # Logging
    log_level: str = 'INFO'
    log_to_file: bool = False


# ============================================================================
# System Integrator
# ============================================================================

class RFArsenalSystem:
    """
    Main system integrator for RF Arsenal OS
    
    Coordinates all modules with stealth and AI integration.
    """
    
    VERSION = '1.0.0'
    
    def __init__(self, config: SystemConfig = None):
        """
        Initialize RF Arsenal system
        
        Args:
            config: System configuration
        """
        self.config = config or SystemConfig()
        self._state = SystemState.OFFLINE
        self._lock = threading.RLock()
        
        # Module registry
        self._modules: Dict[str, ModuleDescriptor] = {}
        self._module_instances: Dict[str, Any] = {}
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        # Subsystems (lazy loaded)
        self._hardware = None
        self._stealth = None
        self._emergency = None
        self._calibration = None
        self._dsp = None
        self._protocols = None
        self._ai = None
        
        # Initialize logging
        self._setup_logging()
        
        logger.info(f"RF Arsenal OS v{self.VERSION} initializing")
    
    def _setup_logging(self):
        """Configure logging"""
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # ========================================================================
    # System Lifecycle
    # ========================================================================
    
    def initialize(self) -> bool:
        """
        Initialize system and all subsystems
        
        Returns:
            True on success
        """
        with self._lock:
            if self._state != SystemState.OFFLINE:
                logger.warning("System already initialized")
                return False
            
            self._state = SystemState.INITIALIZING
            logger.info("System initialization starting")
            
            try:
                # Initialize stealth first (controls other systems)
                if self.config.stealth_mode:
                    self._init_stealth()
                
                # Initialize emergency system
                self._init_emergency()
                
                # Initialize hardware
                if self.config.hardware_enabled:
                    self._init_hardware()
                
                # Initialize calibration
                self._init_calibration()
                
                # Initialize AI
                if self.config.ai_enabled:
                    self._init_ai()
                
                # Register built-in modules
                self._register_builtin_modules()
                
                self._state = SystemState.READY
                self._emit_event('system_ready', {})
                
                logger.info("System initialization complete")
                return True
                
            except Exception as e:
                logger.error(f"System initialization failed: {e}")
                self._state = SystemState.OFFLINE
                return False
    
    def shutdown(self, emergency: bool = False):
        """
        Shutdown system
        
        Args:
            emergency: If True, perform emergency shutdown
        """
        with self._lock:
            if emergency:
                self._state = SystemState.EMERGENCY
                logger.warning("EMERGENCY SHUTDOWN INITIATED")
                
                # Emergency procedures
                if self._hardware:
                    self._hardware.emergency_shutdown()
                
                if self._emergency:
                    self._emergency.panic()
            else:
                self._state = SystemState.SHUTDOWN
                logger.info("Normal shutdown initiated")
            
            # Cleanup
            for name, instance in self._module_instances.items():
                try:
                    if hasattr(instance, 'shutdown'):
                        instance.shutdown()
                except Exception as e:
                    logger.error(f"Module {name} shutdown error: {e}")
            
            self._module_instances.clear()
            
            if self._hardware:
                self._hardware.disconnect()
            
            self._state = SystemState.OFFLINE
            self._emit_event('system_shutdown', {'emergency': emergency})
            
            logger.info("System shutdown complete")
    
    # ========================================================================
    # Subsystem Initialization
    # ========================================================================
    
    def _init_stealth(self):
        """Initialize stealth system"""
        try:
            from core.stealth import StealthSystem, NetworkAnonymity
            
            self._stealth = {
                'system': StealthSystem(
                    ram_only=self.config.ram_only_mode,
                    secure_delete=self.config.secure_delete
                ),
                'network': NetworkAnonymity(
                    tor_enabled=self.config.tor_enabled,
                    mac_randomization=self.config.mac_randomization
                )
            }
            
            logger.info("Stealth system initialized")
        except ImportError as e:
            logger.warning(f"Stealth module not available: {e}")
            self._stealth = None
    
    def _init_emergency(self):
        """Initialize emergency system"""
        try:
            from core.emergency import EmergencySystem
            
            self._emergency = EmergencySystem(
                panic_enabled=self.config.panic_enabled
            )
            
            # Register panic callback
            self._emergency.register_callback(
                'panic',
                lambda: self.shutdown(emergency=True)
            )
            
            logger.info("Emergency system initialized")
        except ImportError as e:
            logger.warning(f"Emergency module not available: {e}")
            self._emergency = None
    
    def _init_hardware(self):
        """Initialize hardware driver"""
        try:
            from core.hardware.bladerf_driver import create_bladerf_driver
            
            stealth_system = self._stealth['system'] if self._stealth else None
            
            self._hardware = create_bladerf_driver(
                serial=self.config.bladerf_serial,
                stealth_system=stealth_system
            )
            
            if self._hardware.connect():
                logger.info(f"Hardware initialized: "
                           f"{self._hardware.device_info.board_name}")
            
        except ImportError as e:
            logger.warning(f"Hardware module not available: {e}")
            self._hardware = None
    
    def _init_calibration(self):
        """Initialize calibration system"""
        try:
            from core.calibration.rf_calibration import create_calibration_manager
            
            stealth_system = self._stealth['system'] if self._stealth else None
            
            self._calibration = create_calibration_manager(
                hardware_driver=self._hardware,
                stealth_system=stealth_system
            )
            
            logger.info("Calibration system initialized")
        except ImportError as e:
            logger.warning(f"Calibration module not available: {e}")
            self._calibration = None
    
    def _init_ai(self):
        """Initialize AI control system"""
        try:
            from modules.ai.ai_controller import AIController
            from modules.ai.text_ai import TextAIInterface
            
            self._ai = {
                'controller': AIController(),
                'text': TextAIInterface(system=self)
            }
            
            logger.info("AI control system initialized")
        except ImportError as e:
            logger.warning(f"AI module not available: {e}")
            self._ai = None
    
    # ========================================================================
    # Module Management
    # ========================================================================
    
    def _register_builtin_modules(self):
        """Register built-in modules"""
        builtin_modules = [
            ModuleDescriptor(
                name='cellular_2g',
                version='1.0.0',
                category='cellular',
                description='2G/GSM base station emulation',
                stealth_compatible=True,
                ai_controllable=True
            ),
            ModuleDescriptor(
                name='cellular_3g',
                version='1.0.0',
                category='cellular',
                description='3G/UMTS base station emulation',
                stealth_compatible=True,
                ai_controllable=True
            ),
            ModuleDescriptor(
                name='cellular_4g',
                version='1.0.0',
                category='cellular',
                description='4G/LTE base station emulation',
                dependencies=['dsp', 'protocols'],
                stealth_compatible=True,
                ai_controllable=True
            ),
            ModuleDescriptor(
                name='cellular_5g',
                version='1.0.0',
                category='cellular',
                description='5G/NR base station emulation',
                dependencies=['dsp', 'protocols'],
                stealth_compatible=True,
                ai_controllable=True
            ),
            ModuleDescriptor(
                name='wifi',
                version='1.0.0',
                category='wifi',
                description='WiFi attack suite',
                stealth_compatible=True,
                ai_controllable=True
            ),
            ModuleDescriptor(
                name='gps',
                version='1.0.0',
                category='positioning',
                description='GPS spoofing',
                stealth_compatible=True,
                ai_controllable=True
            ),
            ModuleDescriptor(
                name='drone',
                version='1.0.0',
                category='drone',
                description='Drone warfare suite',
                stealth_compatible=True,
                ai_controllable=True
            ),
            ModuleDescriptor(
                name='spectrum',
                version='1.0.0',
                category='analysis',
                description='Spectrum analyzer',
                stealth_compatible=True,
                ai_controllable=True
            ),
            ModuleDescriptor(
                name='jamming',
                version='1.0.0',
                category='jamming',
                description='Electronic warfare jamming',
                stealth_compatible=False,
                ai_controllable=True
            ),
            ModuleDescriptor(
                name='sigint',
                version='1.0.0',
                category='intelligence',
                description='SIGINT collection',
                stealth_compatible=True,
                ai_controllable=True
            ),
            ModuleDescriptor(
                name='radar',
                version='1.0.0',
                category='radar',
                description='Radar systems',
                stealth_compatible=True,
                ai_controllable=True
            ),
            ModuleDescriptor(
                name='iot_rfid',
                version='1.0.0',
                category='iot',
                description='IoT/RFID security',
                stealth_compatible=True,
                ai_controllable=True
            ),
            ModuleDescriptor(
                name='satellite',
                version='1.0.0',
                category='satellite',
                description='Satellite communications',
                stealth_compatible=True,
                ai_controllable=True
            ),
            ModuleDescriptor(
                name='amateur',
                version='1.0.0',
                category='amateur',
                description='Amateur radio',
                stealth_compatible=True,
                ai_controllable=True
            ),
            ModuleDescriptor(
                name='protocol_analyzer',
                version='1.0.0',
                category='analysis',
                description='Protocol analyzer',
                stealth_compatible=True,
                ai_controllable=True
            ),
        ]
        
        for module in builtin_modules:
            self._modules[module.name] = module
        
        logger.info(f"Registered {len(builtin_modules)} built-in modules")
    
    def load_module(self, name: str) -> bool:
        """
        Load and initialize a module
        
        Args:
            name: Module name
        
        Returns:
            True on success
        """
        if name not in self._modules:
            logger.error(f"Unknown module: {name}")
            return False
        
        descriptor = self._modules[name]
        
        # Check stealth compatibility
        if self._state == SystemState.STEALTH and not descriptor.stealth_compatible:
            logger.warning(f"Module {name} not compatible with stealth mode")
            return False
        
        # Check dependencies
        for dep in descriptor.dependencies:
            if dep not in self._module_instances:
                logger.warning(f"Module {name} requires {dep}")
                return False
        
        descriptor.state = ModuleState.LOADING
        
        try:
            # Dynamic import based on category
            module_instance = self._import_module(descriptor)
            
            if module_instance:
                self._module_instances[name] = module_instance
                descriptor.state = ModuleState.READY
                self._emit_event('module_loaded', {'name': name})
                logger.info(f"Module {name} loaded")
                return True
            
        except Exception as e:
            logger.error(f"Failed to load module {name}: {e}")
            descriptor.state = ModuleState.ERROR
        
        return False
    
    def _import_module(self, descriptor: ModuleDescriptor) -> Any:
        """Import module based on descriptor"""
        try:
            if descriptor.category == 'cellular':
                if descriptor.name == 'cellular_4g':
                    from modules.cellular.lte_4g import LTEBaseStation
                    return LTEBaseStation(
                        hardware=self._hardware,
                        stealth=self._stealth['system'] if self._stealth else None
                    )
                elif descriptor.name == 'cellular_5g':
                    from modules.cellular.nr_5g import NRBaseStation
                    return NRBaseStation(
                        hardware=self._hardware,
                        stealth=self._stealth['system'] if self._stealth else None
                    )
            
            elif descriptor.category == 'wifi':
                from modules.wifi.wifi_attacks import WiFiAttackSuite
                return WiFiAttackSuite()
            
            elif descriptor.category == 'positioning':
                from modules.gps.gps_spoofer import GPSSpoofer
                return GPSSpoofer()
            
            elif descriptor.category == 'drone':
                from modules.drone.drone_warfare import DroneWarfare
                return DroneWarfare()
            
            elif descriptor.category == 'analysis':
                if descriptor.name == 'spectrum':
                    from modules.spectrum.spectrum_analyzer import SpectrumAnalyzer
                    return SpectrumAnalyzer()
                elif descriptor.name == 'protocol_analyzer':
                    from modules.protocol.protocol_analyzer import ProtocolAnalyzer
                    return ProtocolAnalyzer()
            
            elif descriptor.category == 'jamming':
                from modules.jamming.jamming_suite import JammingSuite
                return JammingSuite()
            
            elif descriptor.category == 'intelligence':
                from modules.sigint.sigint_engine import SIGINTEngine
                return SIGINTEngine()
            
            # Add more categories as needed
            
        except ImportError as e:
            logger.warning(f"Module import failed: {e}")
        
        return None
    
    def unload_module(self, name: str) -> bool:
        """Unload a module"""
        if name not in self._module_instances:
            return False
        
        try:
            instance = self._module_instances[name]
            if hasattr(instance, 'shutdown'):
                instance.shutdown()
            
            del self._module_instances[name]
            self._modules[name].state = ModuleState.UNLOADED
            
            self._emit_event('module_unloaded', {'name': name})
            logger.info(f"Module {name} unloaded")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload module {name}: {e}")
            return False
    
    def get_module(self, name: str) -> Optional[Any]:
        """Get loaded module instance"""
        return self._module_instances.get(name)
    
    def list_modules(self) -> List[Dict[str, Any]]:
        """List all registered modules"""
        return [
            {
                'name': m.name,
                'version': m.version,
                'category': m.category,
                'description': m.description,
                'state': m.state.name,
                'stealth_compatible': m.stealth_compatible,
                'ai_controllable': m.ai_controllable
            }
            for m in self._modules.values()
        ]
    
    # ========================================================================
    # Stealth Operations
    # ========================================================================
    
    def enter_stealth_mode(self) -> bool:
        """Enter stealth mode"""
        with self._lock:
            if self._state != SystemState.READY:
                logger.warning("Cannot enter stealth: system not ready")
                return False
            
            logger.info("Entering stealth mode")
            
            # Activate stealth system
            if self._stealth:
                self._stealth['system'].activate()
                self._stealth['network'].enable()
            
            # Configure hardware for stealth
            if self._hardware:
                self._hardware.set_stealth_mode(True)
            
            # Disable non-stealth-compatible modules
            for name, descriptor in self._modules.items():
                if not descriptor.stealth_compatible and name in self._module_instances:
                    self.unload_module(name)
            
            self._state = SystemState.STEALTH
            self._emit_event('stealth_activated', {})
            
            return True
    
    def exit_stealth_mode(self) -> bool:
        """Exit stealth mode"""
        with self._lock:
            if self._state != SystemState.STEALTH:
                return False
            
            logger.info("Exiting stealth mode")
            
            if self._stealth:
                self._stealth['system'].deactivate()
            
            if self._hardware:
                self._hardware.set_stealth_mode(False)
            
            self._state = SystemState.READY
            self._emit_event('stealth_deactivated', {})
            
            return True
    
    # ========================================================================
    # AI Control Interface
    # ========================================================================
    
    def execute_command(self, command: str) -> Dict[str, Any]:
        """
        Execute natural language command via AI
        
        Args:
            command: Natural language command
        
        Returns:
            Command result
        """
        if not self._ai:
            return {'success': False, 'error': 'AI not available'}
        
        try:
            result = self._ai['text'].process_command(command)
            return {'success': True, 'result': result}
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_ai_capabilities(self) -> List[str]:
        """Get list of AI-controllable capabilities"""
        capabilities = []
        
        for name, descriptor in self._modules.items():
            if descriptor.ai_controllable:
                capabilities.append(name)
        
        return capabilities
    
    # ========================================================================
    # Hardware Operations
    # ========================================================================
    
    def configure_frequency(self, frequency: float, 
                            bandwidth: float = None,
                            sample_rate: float = None) -> bool:
        """Configure RF frequency"""
        if not self._hardware:
            logger.warning("Hardware not available")
            return False
        
        from core.hardware.bladerf_driver import BladeRFChannel
        
        # Apply to both RX and TX
        for channel in [BladeRFChannel.RX0, BladeRFChannel.TX0]:
            if not self._hardware.set_frequency(channel, int(frequency)):
                return False
            
            if bandwidth:
                self._hardware.set_bandwidth(channel, int(bandwidth))
            
            if sample_rate:
                self._hardware.set_sample_rate(channel, int(sample_rate))
        
        return True
    
    def calibrate_system(self, full: bool = False) -> Dict[str, Any]:
        """Run system calibration"""
        if not self._calibration:
            return {'success': False, 'error': 'Calibration not available'}
        
        results = {}
        
        from core.hardware.bladerf_driver import BladeRFChannel
        
        for channel in [BladeRFChannel.RX0, BladeRFChannel.TX0]:
            if full:
                success = self._calibration.calibrate_full(channel.value)
            else:
                success = self._calibration.calibrate_dc_offset(channel.value)
            
            results[channel.name] = self._calibration.get_calibration_status(
                channel.value
            )
        
        return {'success': True, 'results': results}
    
    # ========================================================================
    # Event System
    # ========================================================================
    
    def register_event_handler(self, event: str, handler: Callable):
        """Register event handler"""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
    
    def _emit_event(self, event: str, data: Dict[str, Any]):
        """Emit event to handlers"""
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")
    
    # ========================================================================
    # Status and Diagnostics
    # ========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'version': self.VERSION,
            'state': self._state.name,
            'timestamp': time.time()
        }
        
        # Hardware status
        if self._hardware:
            status['hardware'] = self._hardware.get_statistics()
        else:
            status['hardware'] = {'available': False}
        
        # Stealth status
        if self._stealth:
            status['stealth'] = {
                'active': self._state == SystemState.STEALTH,
                'network_anonymity': self._stealth['network'].is_active() 
                                     if hasattr(self._stealth['network'], 'is_active') 
                                     else False
            }
        else:
            status['stealth'] = {'available': False}
        
        # Module status
        status['modules'] = {
            'registered': len(self._modules),
            'loaded': len(self._module_instances),
            'active': [name for name, desc in self._modules.items() 
                      if desc.state == ModuleState.ACTIVE]
        }
        
        # Calibration status
        if self._calibration:
            status['calibration'] = {
                'available': True
            }
        else:
            status['calibration'] = {'available': False}
        
        return status
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run system diagnostics"""
        results = {
            'timestamp': time.time(),
            'tests': {}
        }
        
        # Hardware self-test
        if self._hardware:
            results['tests']['hardware'] = self._hardware.run_self_test()
        
        # Module health check
        module_health = {}
        for name, instance in self._module_instances.items():
            if hasattr(instance, 'health_check'):
                module_health[name] = instance.health_check()
            else:
                module_health[name] = {'status': 'unknown'}
        results['tests']['modules'] = module_health
        
        return results


# ============================================================================
# Factory Functions
# ============================================================================

def create_system(config: SystemConfig = None) -> RFArsenalSystem:
    """Create and initialize RF Arsenal system"""
    system = RFArsenalSystem(config)
    return system


def quick_start(stealth: bool = True) -> RFArsenalSystem:
    """Quick start with common configuration"""
    config = SystemConfig(
        stealth_mode=stealth,
        hardware_enabled=True,
        hardware_simulated=True,
        ai_enabled=True
    )
    
    system = create_system(config)
    system.initialize()
    
    if stealth:
        system.enter_stealth_mode()
    
    return system
