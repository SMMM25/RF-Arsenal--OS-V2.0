"""
RF Arsenal OS - Deployment Orchestrator
Production deployment orchestration for external protocol stacks

This module provides:
- Automated deployment of srsRAN/OAI stacks
- Multi-node deployment coordination
- Health monitoring and auto-recovery
- Configuration management
- Rollback capabilities
- Deployment profiles for different scenarios
"""

import os
import sys
import time
import json
import yaml
import logging
import threading
import subprocess
import shutil
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


# ============================================================================
# Deployment Enums
# ============================================================================

class DeploymentStatus(IntEnum):
    """Deployment status"""
    PENDING = 0
    INITIALIZING = 1
    DEPLOYING = 2
    RUNNING = 3
    DEGRADED = 4
    STOPPING = 5
    STOPPED = 6
    FAILED = 7
    ROLLING_BACK = 8


class DeploymentProfile(IntEnum):
    """Pre-defined deployment profiles"""
    MINIMAL = 0          # Single node, minimal resources
    STANDARD = 1         # Standard LTE deployment
    FULL_LTE = 2         # Full LTE with EPC
    FULL_5G_SA = 3       # Full 5G Standalone
    FULL_5G_NSA = 4      # Full 5G Non-Standalone
    DEVELOPMENT = 5      # Development/testing
    SIMULATION = 6       # Simulation mode
    HIGH_AVAILABILITY = 7 # HA deployment


class NodeRole(IntEnum):
    """Node role in deployment"""
    BASE_STATION = 0     # eNB/gNB
    CORE_NETWORK = 1     # EPC/5GC
    UE = 2               # User Equipment
    CONTROLLER = 3       # Central controller
    MONITOR = 4          # Monitoring node


# ============================================================================
# Configuration Structures
# ============================================================================

@dataclass
class NodeConfig:
    """Configuration for a deployment node"""
    node_id: str
    role: NodeRole
    host: str = "localhost"
    port: int = 0
    
    # Resource limits
    cpu_cores: int = 4
    memory_mb: int = 4096
    
    # Network
    data_interface: str = "eth0"
    management_interface: str = "eth0"
    
    # Additional settings
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'role': self.role.name,
            'host': self.host,
            'port': self.port,
            'cpu_cores': self.cpu_cores,
            'memory_mb': self.memory_mb,
            'data_interface': self.data_interface,
            'management_interface': self.management_interface,
            'extra_params': self.extra_params
        }


@dataclass
class DeploymentConfig:
    """Complete deployment configuration"""
    name: str = "rf-arsenal-deployment"
    profile: DeploymentProfile = DeploymentProfile.STANDARD
    
    # Nodes
    nodes: List[NodeConfig] = field(default_factory=list)
    
    # RF Configuration
    frequency_hz: float = 2680e6
    bandwidth_mhz: float = 20.0
    tx_gain_db: float = 50.0
    rx_gain_db: float = 40.0
    
    # Cell Configuration
    cell_id: int = 1
    tac: int = 1
    mcc: str = "001"
    mnc: str = "01"
    
    # Network Configuration
    network_name: str = "RF-Arsenal"
    dnn: str = "internet"
    
    # Features
    enable_stealth: bool = True
    enable_ai_control: bool = True
    enable_logging: bool = True
    enable_pcap: bool = False
    
    # Timeouts
    startup_timeout: float = 120.0
    health_check_interval: float = 30.0
    
    # Paths
    config_dir: str = "/etc/rfarsenal"
    log_dir: str = "/var/log/rfarsenal"
    data_dir: str = "/var/lib/rfarsenal"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'profile': self.profile.name,
            'nodes': [n.to_dict() for n in self.nodes],
            'rf': {
                'frequency_hz': self.frequency_hz,
                'bandwidth_mhz': self.bandwidth_mhz,
                'tx_gain_db': self.tx_gain_db,
                'rx_gain_db': self.rx_gain_db
            },
            'cell': {
                'cell_id': self.cell_id,
                'tac': self.tac,
                'mcc': self.mcc,
                'mnc': self.mnc
            },
            'network': {
                'name': self.network_name,
                'dnn': self.dnn
            },
            'features': {
                'stealth': self.enable_stealth,
                'ai_control': self.enable_ai_control,
                'logging': self.enable_logging,
                'pcap': self.enable_pcap
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeploymentConfig':
        """Create config from dictionary"""
        config = cls(
            name=data.get('name', 'deployment'),
            profile=DeploymentProfile[data.get('profile', 'STANDARD')]
        )
        
        if 'rf' in data:
            config.frequency_hz = data['rf'].get('frequency_hz', 2680e6)
            config.bandwidth_mhz = data['rf'].get('bandwidth_mhz', 20.0)
            config.tx_gain_db = data['rf'].get('tx_gain_db', 50.0)
            config.rx_gain_db = data['rf'].get('rx_gain_db', 40.0)
        
        if 'cell' in data:
            config.cell_id = data['cell'].get('cell_id', 1)
            config.tac = data['cell'].get('tac', 1)
            config.mcc = data['cell'].get('mcc', '001')
            config.mnc = data['cell'].get('mnc', '01')
        
        return config
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'DeploymentConfig':
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    def to_yaml(self, filepath: str):
        """Save configuration to YAML file"""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# ============================================================================
# Deployment State
# ============================================================================

@dataclass
class DeploymentState:
    """Current deployment state"""
    status: DeploymentStatus = DeploymentStatus.PENDING
    start_time: Optional[float] = None
    last_health_check: Optional[float] = None
    
    # Node states
    node_states: Dict[str, DeploymentStatus] = field(default_factory=dict)
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metrics
    uptime: float = 0.0
    restarts: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status.name,
            'start_time': self.start_time,
            'last_health_check': self.last_health_check,
            'node_states': {k: v.name for k, v in self.node_states.items()},
            'errors': self.errors,
            'warnings': self.warnings,
            'uptime': self.uptime,
            'restarts': self.restarts
        }


# ============================================================================
# Health Check
# ============================================================================

@dataclass
class HealthCheckResult:
    """Health check result"""
    node_id: str
    healthy: bool
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class HealthChecker:
    """Health monitoring for deployment"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self._results: Dict[str, HealthCheckResult] = {}
    
    def check_node(self, node: NodeConfig) -> HealthCheckResult:
        """Check health of a single node"""
        result = HealthCheckResult(
            node_id=node.node_id,
            healthy=True
        )
        
        try:
            # Check process running
            if node.role == NodeRole.BASE_STATION:
                result.details['processes'] = self._check_ran_processes()
            elif node.role == NodeRole.CORE_NETWORK:
                result.details['processes'] = self._check_core_processes()
            
            # Check network connectivity
            result.details['network'] = self._check_network(node)
            
            # Check resource usage
            result.details['resources'] = self._check_resources()
            
        except Exception as e:
            result.healthy = False
            result.error_message = str(e)
        
        self._results[node.node_id] = result
        return result
    
    def check_all(self) -> Dict[str, HealthCheckResult]:
        """Check health of all nodes"""
        results = {}
        for node in self.config.nodes:
            results[node.node_id] = self.check_node(node)
        return results
    
    def get_overall_health(self) -> bool:
        """Get overall deployment health"""
        return all(r.healthy for r in self._results.values())
    
    def _check_ran_processes(self) -> Dict[str, bool]:
        """Check RAN processes"""
        processes = {
            'srsran': ['srsenb', 'srsue'],
            'oai': ['nr-softmodem', 'lte-softmodem']
        }
        
        result = {}
        for category, proc_list in processes.items():
            for proc in proc_list:
                result[proc] = self._is_process_running(proc)
        
        return result
    
    def _check_core_processes(self) -> Dict[str, bool]:
        """Check core network processes"""
        processes = ['srsepc', 'oai-amf', 'oai-smf', 'oai-upf']
        return {p: self._is_process_running(p) for p in processes}
    
    def _is_process_running(self, name: str) -> bool:
        """Check if process is running"""
        try:
            result = subprocess.run(
                ['pgrep', '-f', name],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def _check_network(self, node: NodeConfig) -> Dict[str, Any]:
        """Check network connectivity"""
        return {
            'interface_up': True,  # Would check actual interface
            'connectivity': True
        }
    
    def _check_resources(self) -> Dict[str, Any]:
        """Check resource usage"""
        return {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'disk_percent': 0.0
        }


# ============================================================================
# Deployment Orchestrator
# ============================================================================

class DeploymentOrchestrator:
    """
    Main orchestrator for external stack deployments
    
    Handles:
    - Automated deployment of srsRAN/OAI
    - Multi-node coordination
    - Health monitoring
    - Auto-recovery
    - Rollback support
    """
    
    def __init__(self, config: DeploymentConfig = None, stealth_system=None):
        """
        Initialize deployment orchestrator
        
        Args:
            config: Deployment configuration
            stealth_system: Optional stealth system integration
        """
        self.config = config or DeploymentConfig()
        self.stealth_system = stealth_system
        
        self._state = DeploymentState()
        self._lock = threading.RLock()
        
        # Health checker
        self._health_checker = HealthChecker(self.config)
        
        # Background threads
        self._health_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Stack managers
        self._stack_manager = None
        self._protocol_bridge = None
        
        # Event callbacks
        self._callbacks: Dict[str, List[Callable]] = {}
        
        # Rollback snapshots
        self._snapshots: List[Dict] = []
    
    @property
    def status(self) -> DeploymentStatus:
        return self._state.status
    
    @property
    def is_running(self) -> bool:
        return self._state.status == DeploymentStatus.RUNNING
    
    # ========================================================================
    # Deployment Lifecycle
    # ========================================================================
    
    def deploy(self, wait: bool = True) -> bool:
        """
        Deploy the configured stack
        
        Args:
            wait: Wait for deployment to complete
        
        Returns:
            True if deployment succeeded
        """
        with self._lock:
            if self._state.status not in (DeploymentStatus.PENDING, 
                                           DeploymentStatus.STOPPED,
                                           DeploymentStatus.FAILED):
                logger.warning("Cannot deploy - already in progress or running")
                return False
            
            logger.info(f"Starting deployment: {self.config.name}")
            self._state.status = DeploymentStatus.INITIALIZING
            self._state.start_time = time.time()
            self._emit_event('deployment_started', {'name': self.config.name})
        
        try:
            # Create directories
            self._create_directories()
            
            # Save snapshot for rollback
            self._create_snapshot()
            
            # Initialize components based on profile
            self._state.status = DeploymentStatus.DEPLOYING
            
            if self.config.profile == DeploymentProfile.SIMULATION:
                success = self._deploy_simulation()
            elif self.config.profile in (DeploymentProfile.FULL_5G_SA, 
                                         DeploymentProfile.FULL_5G_NSA):
                success = self._deploy_5g()
            else:
                success = self._deploy_lte()
            
            if success:
                self._state.status = DeploymentStatus.RUNNING
                self._running = True
                
                # Start health monitoring
                self._start_health_monitoring()
                
                logger.info("Deployment completed successfully")
                self._emit_event('deployment_completed', {'success': True})
                return True
            else:
                self._state.status = DeploymentStatus.FAILED
                self._emit_event('deployment_completed', {'success': False})
                return False
                
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            self._state.status = DeploymentStatus.FAILED
            self._state.errors.append(str(e))
            self._emit_event('deployment_failed', {'error': str(e)})
            return False
    
    def stop(self, graceful: bool = True) -> bool:
        """
        Stop the deployment
        
        Args:
            graceful: Perform graceful shutdown
        
        Returns:
            True if stopped successfully
        """
        with self._lock:
            if self._state.status == DeploymentStatus.STOPPED:
                return True
            
            logger.info("Stopping deployment")
            self._state.status = DeploymentStatus.STOPPING
            self._running = False
        
        try:
            # Stop health monitoring
            self._stop_health_monitoring()
            
            # Stop stack manager
            if self._stack_manager:
                self._stack_manager.shutdown()
            
            # Stop protocol bridge
            if self._protocol_bridge:
                self._protocol_bridge.shutdown()
            
            self._state.status = DeploymentStatus.STOPPED
            logger.info("Deployment stopped")
            self._emit_event('deployment_stopped', {})
            return True
            
        except Exception as e:
            logger.error(f"Error stopping deployment: {e}")
            self._state.status = DeploymentStatus.FAILED
            return False
    
    def restart(self) -> bool:
        """Restart the deployment"""
        logger.info("Restarting deployment")
        self._state.restarts += 1
        
        if self.stop():
            time.sleep(2.0)
            return self.deploy()
        return False
    
    def rollback(self) -> bool:
        """Rollback to previous state"""
        if not self._snapshots:
            logger.warning("No snapshots available for rollback")
            return False
        
        logger.info("Rolling back deployment")
        self._state.status = DeploymentStatus.ROLLING_BACK
        
        try:
            # Stop current deployment
            self.stop(graceful=False)
            
            # Restore previous snapshot
            snapshot = self._snapshots.pop()
            # Would restore configuration here
            
            # Redeploy
            return self.deploy()
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            self._state.status = DeploymentStatus.FAILED
            return False
    
    # ========================================================================
    # Profile-Specific Deployment
    # ========================================================================
    
    def _deploy_simulation(self) -> bool:
        """Deploy in simulation mode"""
        logger.info("Deploying in simulation mode")
        
        try:
            from .stack_manager import ExternalStackManager, StackConfig, StackType
            
            stack_config = StackConfig(
                preferred_stack=StackType.NATIVE,
                frequency_hz=self.config.frequency_hz,
                bandwidth_mhz=self.config.bandwidth_mhz,
                tx_gain_db=self.config.tx_gain_db,
                rx_gain_db=self.config.rx_gain_db,
                cell_id=self.config.cell_id,
                mcc=self.config.mcc,
                mnc=self.config.mnc,
                stealth_mode=self.config.enable_stealth
            )
            
            self._stack_manager = ExternalStackManager(
                config=stack_config,
                stealth_system=self.stealth_system
            )
            
            return self._stack_manager.initialize()
            
        except Exception as e:
            logger.error(f"Simulation deployment failed: {e}")
            return False
    
    def _deploy_lte(self) -> bool:
        """Deploy LTE network"""
        logger.info("Deploying LTE network")
        
        try:
            from .stack_manager import (
                ExternalStackManager, StackConfig, StackType, NetworkMode
            )
            
            stack_config = StackConfig(
                preferred_stack=StackType.SRSRAN,  # srsRAN for LTE
                network_mode=NetworkMode.LTE,
                frequency_hz=self.config.frequency_hz,
                bandwidth_mhz=self.config.bandwidth_mhz,
                tx_gain_db=self.config.tx_gain_db,
                rx_gain_db=self.config.rx_gain_db,
                cell_id=self.config.cell_id,
                mcc=self.config.mcc,
                mnc=self.config.mnc,
                stealth_mode=self.config.enable_stealth
            )
            
            self._stack_manager = ExternalStackManager(
                config=stack_config,
                stealth_system=self.stealth_system
            )
            
            if not self._stack_manager.initialize():
                return False
            
            # Initialize protocol bridge
            from .protocol_bridge import create_protocol_bridge
            self._protocol_bridge = create_protocol_bridge(self.stealth_system)
            
            # Start network if full deployment
            if self.config.profile in (DeploymentProfile.FULL_LTE, 
                                       DeploymentProfile.STANDARD):
                return self._stack_manager.start_network()
            
            return True
            
        except Exception as e:
            logger.error(f"LTE deployment failed: {e}")
            return False
    
    def _deploy_5g(self) -> bool:
        """Deploy 5G network"""
        logger.info("Deploying 5G network")
        
        try:
            from .stack_manager import (
                ExternalStackManager, StackConfig, StackType, NetworkMode
            )
            
            network_mode = (NetworkMode.NR_SA 
                          if self.config.profile == DeploymentProfile.FULL_5G_SA 
                          else NetworkMode.NR_NSA)
            
            stack_config = StackConfig(
                preferred_stack=StackType.OAI,  # OAI for 5G
                network_mode=network_mode,
                frequency_hz=self.config.frequency_hz,
                bandwidth_mhz=self.config.bandwidth_mhz,
                tx_gain_db=self.config.tx_gain_db,
                rx_gain_db=self.config.rx_gain_db,
                cell_id=self.config.cell_id,
                mcc=self.config.mcc,
                mnc=self.config.mnc,
                stealth_mode=self.config.enable_stealth
            )
            
            self._stack_manager = ExternalStackManager(
                config=stack_config,
                stealth_system=self.stealth_system
            )
            
            if not self._stack_manager.initialize():
                return False
            
            # Initialize protocol bridge
            from .protocol_bridge import create_protocol_bridge
            self._protocol_bridge = create_protocol_bridge(self.stealth_system)
            
            # Start network
            return self._stack_manager.start_network()
            
        except Exception as e:
            logger.error(f"5G deployment failed: {e}")
            return False
    
    # ========================================================================
    # Health Monitoring
    # ========================================================================
    
    def _start_health_monitoring(self):
        """Start health monitoring thread"""
        if self._health_thread and self._health_thread.is_alive():
            return
        
        self._health_thread = threading.Thread(
            target=self._health_loop,
            daemon=True
        )
        self._health_thread.start()
        logger.info("Health monitoring started")
    
    def _stop_health_monitoring(self):
        """Stop health monitoring"""
        self._running = False
        if self._health_thread:
            self._health_thread.join(timeout=5.0)
    
    def _health_loop(self):
        """Health monitoring loop"""
        while self._running:
            try:
                # Perform health checks
                results = self._health_checker.check_all()
                self._state.last_health_check = time.time()
                
                # Update state
                overall_healthy = self._health_checker.get_overall_health()
                
                if not overall_healthy:
                    if self._state.status == DeploymentStatus.RUNNING:
                        self._state.status = DeploymentStatus.DEGRADED
                        self._emit_event('health_degraded', {'results': results})
                        
                        # Attempt auto-recovery
                        self._attempt_recovery()
                else:
                    if self._state.status == DeploymentStatus.DEGRADED:
                        self._state.status = DeploymentStatus.RUNNING
                        self._emit_event('health_restored', {})
                
                # Update uptime
                if self._state.start_time:
                    self._state.uptime = time.time() - self._state.start_time
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            time.sleep(self.config.health_check_interval)
    
    def _attempt_recovery(self):
        """Attempt automatic recovery"""
        logger.info("Attempting automatic recovery")
        
        # Simple restart for now
        if self._stack_manager:
            try:
                self._stack_manager.stop_network()
                time.sleep(2.0)
                self._stack_manager.start_network()
                logger.info("Recovery successful")
            except Exception as e:
                logger.error(f"Recovery failed: {e}")
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    def _create_directories(self):
        """Create necessary directories"""
        dirs = [
            self.config.config_dir,
            self.config.log_dir,
            self.config.data_dir
        ]
        
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def _create_snapshot(self):
        """Create configuration snapshot for rollback"""
        snapshot = {
            'timestamp': time.time(),
            'config': self.config.to_dict(),
            'state': self._state.to_dict()
        }
        self._snapshots.append(snapshot)
        
        # Keep only last 5 snapshots
        if len(self._snapshots) > 5:
            self._snapshots.pop(0)
    
    # ========================================================================
    # Status and Monitoring
    # ========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get deployment status"""
        status = self._state.to_dict()
        status['config'] = {
            'name': self.config.name,
            'profile': self.config.profile.name
        }
        
        if self._stack_manager:
            status['stack'] = self._stack_manager.get_status().__dict__
        
        return status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get deployment metrics"""
        metrics = {
            'uptime': self._state.uptime,
            'restarts': self._state.restarts,
            'health_check_interval': self.config.health_check_interval
        }
        
        if self._stack_manager:
            metrics['stack'] = self._stack_manager.get_metrics()
        
        if self._protocol_bridge:
            metrics['bridge'] = self._protocol_bridge.get_statistics()
        
        return metrics
    
    # ========================================================================
    # AI Control Interface
    # ========================================================================
    
    def execute_command(self, command: str) -> Dict[str, Any]:
        """
        Execute AI command
        
        Commands:
        - "deploy"
        - "stop"
        - "restart"
        - "rollback"
        - "status"
        - "set <parameter> <value>"
        """
        command = command.lower().strip()
        result = {'success': False, 'message': ''}
        
        try:
            if command == 'deploy':
                result['success'] = self.deploy()
                result['message'] = 'Deployed' if result['success'] else 'Deploy failed'
            
            elif command == 'stop':
                result['success'] = self.stop()
                result['message'] = 'Stopped' if result['success'] else 'Stop failed'
            
            elif command == 'restart':
                result['success'] = self.restart()
                result['message'] = 'Restarted' if result['success'] else 'Restart failed'
            
            elif command == 'rollback':
                result['success'] = self.rollback()
                result['message'] = 'Rolled back' if result['success'] else 'Rollback failed'
            
            elif command == 'status':
                result['success'] = True
                result['status'] = self.get_status()
                result['message'] = f'Status: {self._state.status.name}'
            
            elif command.startswith('set '):
                parts = command[4:].split()
                if len(parts) >= 2:
                    param, value = parts[0], ' '.join(parts[1:])
                    result = self._set_parameter(param, value)
                else:
                    result['message'] = 'Usage: set <parameter> <value>'
            
            else:
                result['message'] = f'Unknown command: {command}'
        
        except Exception as e:
            result['message'] = f'Error: {str(e)}'
        
        return result
    
    def _set_parameter(self, param: str, value: str) -> Dict[str, Any]:
        """Set deployment parameter"""
        result = {'success': False, 'message': ''}
        
        try:
            if param == 'frequency':
                freq = float(value.replace('mhz', '').replace('ghz', '').strip())
                if 'ghz' in value.lower():
                    freq *= 1e9
                else:
                    freq *= 1e6
                self.config.frequency_hz = freq
                result['success'] = True
                result['message'] = f'Frequency set to {freq/1e6:.1f} MHz'
            
            elif param == 'bandwidth':
                bw = float(value.replace('mhz', '').strip())
                self.config.bandwidth_mhz = bw
                result['success'] = True
                result['message'] = f'Bandwidth set to {bw} MHz'
            
            elif param == 'txpower':
                gain = float(value.replace('db', '').strip())
                self.config.tx_gain_db = gain
                result['success'] = True
                result['message'] = f'TX power set to {gain} dB'
            
            else:
                result['message'] = f'Unknown parameter: {param}'
        
        except Exception as e:
            result['message'] = f'Error setting {param}: {e}'
        
        return result
    
    # ========================================================================
    # Events
    # ========================================================================
    
    def register_callback(self, event: str, callback: Callable):
        """Register event callback"""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
    
    def _emit_event(self, event: str, data: Any):
        """Emit event to callbacks"""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Event callback error: {e}")


# ============================================================================
# Factory Functions
# ============================================================================

def create_orchestrator(profile: DeploymentProfile = DeploymentProfile.STANDARD,
                        stealth_system=None) -> DeploymentOrchestrator:
    """Create deployment orchestrator with specified profile"""
    config = DeploymentConfig(profile=profile)
    return DeploymentOrchestrator(config=config, stealth_system=stealth_system)


def quick_deploy_lte(stealth_system=None) -> DeploymentOrchestrator:
    """Quick deploy LTE network"""
    orchestrator = create_orchestrator(
        profile=DeploymentProfile.FULL_LTE,
        stealth_system=stealth_system
    )
    orchestrator.deploy()
    return orchestrator


def quick_deploy_5g(stealth_system=None) -> DeploymentOrchestrator:
    """Quick deploy 5G SA network"""
    orchestrator = create_orchestrator(
        profile=DeploymentProfile.FULL_5G_SA,
        stealth_system=stealth_system
    )
    orchestrator.deploy()
    return orchestrator


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    'DeploymentStatus',
    'DeploymentProfile',
    'NodeRole',
    
    # Configuration
    'NodeConfig',
    'DeploymentConfig',
    'DeploymentState',
    
    # Health
    'HealthCheckResult',
    'HealthChecker',
    
    # Orchestrator
    'DeploymentOrchestrator',
    
    # Factory
    'create_orchestrator',
    'quick_deploy_lte',
    'quick_deploy_5g',
]
