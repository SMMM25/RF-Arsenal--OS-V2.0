"""
RF Arsenal OS - External Protocol Stack Integration

Production-grade integration with srsRAN and OpenAirInterface.
Provides unified interface for real-world LTE/5G operations.

This module provides:
- Unified stack management (srsRAN, OAI, Native)
- Protocol translation and bridging
- Configuration management
- Stealth-aware operations
- AI control integration

Example Usage:
    # Quick start LTE network
    from core.external import quick_start_lte
    manager = quick_start_lte()
    manager.start_network()
    
    # Full control with custom config
    from core.external import ExternalStackManager, StackConfig, StackType
    config = StackConfig(
        preferred_stack=StackType.SRSRAN,
        frequency_hz=2680e6,
        bandwidth_mhz=20.0
    )
    manager = ExternalStackManager(config=config)
    manager.initialize()
    manager.start_network()
"""

# Stack Manager
from .stack_manager import (
    ExternalStackManager,
    StackType,
    NetworkMode,
    ComponentRole,
    StackConfig,
    StackStatus,
    create_stack_manager,
    quick_start_lte,
    quick_start_5g
)

# Protocol Bridge
from .protocol_bridge import (
    ProtocolBridge,
    ProtocolLayer,
    MessageDirection,
    ProtocolFamily,
    ProtocolMessage,
    S1APMessage,
    NGAPMessage,
    GTPMessage,
    RRCMessage,
    NASMessage,
    SrsRANTranslator,
    OAITranslator,
    ProtocolRouter,
    create_protocol_bridge,
)

# Common Utilities
from .common import (
    ComponentState,
    NetworkGeneration,
    DeploymentType,
    BaseRFConfig,
    BaseCellConfig,
    UECredentials,
    LTEFrequencyUtils,
    NRFrequencyUtils,
    NetworkUtils,
    InstallationChecker,
    StackLogger,
)

# srsRAN Controller (lazy import to avoid import errors if not available)
try:
    from .srsran.srsran_controller import (
        SrsRANController,
        SrsRANConfig,
        SrsRANComponent,
        SrsRANState,
        RFConfig as SrsRANRFConfig,
        CellConfig as SrsRANCellConfig,
        EPCConfig as SrsRANEPCConfig,
        create_srsran_controller,
    )
    SRSRAN_AVAILABLE = True
except ImportError:
    SRSRAN_AVAILABLE = False

# OAI Controller (lazy import)
try:
    from .openairinterface.oai_controller import (
        OAIController,
        OAIConfig,
        OAIComponent,
        OAIState,
        OAIRFConfig,
        OAICellConfig,
        OAICoreConfig,
        OAIDeploymentMode,
        create_oai_controller,
    )
    OAI_AVAILABLE = True
except ImportError:
    OAI_AVAILABLE = False


# Version
__version__ = '1.0.0'


# All exports
__all__ = [
    # Stack Manager
    'ExternalStackManager',
    'StackType',
    'NetworkMode',
    'ComponentRole',
    'StackConfig',
    'StackStatus',
    'create_stack_manager',
    'quick_start_lte',
    'quick_start_5g',
    
    # Protocol Bridge
    'ProtocolBridge',
    'ProtocolLayer',
    'MessageDirection',
    'ProtocolFamily',
    'ProtocolMessage',
    'S1APMessage',
    'NGAPMessage',
    'GTPMessage',
    'RRCMessage',
    'NASMessage',
    'SrsRANTranslator',
    'OAITranslator',
    'ProtocolRouter',
    'create_protocol_bridge',
    
    # Common Utilities
    'ComponentState',
    'NetworkGeneration',
    'DeploymentType',
    'BaseRFConfig',
    'BaseCellConfig',
    'UECredentials',
    'LTEFrequencyUtils',
    'NRFrequencyUtils',
    'NetworkUtils',
    'InstallationChecker',
    'StackLogger',
    
    # Availability flags
    'SRSRAN_AVAILABLE',
    'OAI_AVAILABLE',
]

# Conditionally add srsRAN exports
if SRSRAN_AVAILABLE:
    __all__.extend([
        'SrsRANController',
        'SrsRANConfig',
        'SrsRANComponent',
        'SrsRANState',
        'SrsRANRFConfig',
        'SrsRANCellConfig',
        'SrsRANEPCConfig',
        'create_srsran_controller',
    ])

# Conditionally add OAI exports
if OAI_AVAILABLE:
    __all__.extend([
        'OAIController',
        'OAIConfig',
        'OAIComponent',
        'OAIState',
        'OAIRFConfig',
        'OAICellConfig',
        'OAICoreConfig',
        'OAIDeploymentMode',
        'create_oai_controller',
    ])
