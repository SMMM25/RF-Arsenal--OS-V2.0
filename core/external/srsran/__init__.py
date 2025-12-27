"""
RF Arsenal OS - srsRAN Integration Package

Production-grade srsRAN integration for LTE and 5G NR.
Supports eNB, gNB, UE, and EPC deployment.

Features:
- Full LTE eNB/UE support
- 5G NR gNB support (srsRAN 4G/5G)
- EPC (MME + SGW/PGW) integration
- Stealth-aware operation
- Real-time configuration
- AI control integration

Example:
    from core.external.srsran import SrsRANController, SrsRANConfig
    
    config = SrsRANConfig()
    controller = SrsRANController(config=config)
    controller.initialize()
    controller.start_full_network()
"""

from .srsran_controller import (
    # Controller
    SrsRANController,
    
    # Configuration
    SrsRANConfig,
    RFConfig,
    CellConfig,
    EPCConfig,
    
    # Components and States
    SrsRANComponent,
    SrsRANState,
    
    # Process Management
    SrsRANProcess,
    
    # Configuration Generator
    SrsRANConfigGenerator,
    
    # Factory Function
    create_srsran_controller,
)

__version__ = '1.0.0'

__all__ = [
    # Main Controller
    'SrsRANController',
    
    # Configuration Classes
    'SrsRANConfig',
    'RFConfig',
    'CellConfig',
    'EPCConfig',
    
    # Enums
    'SrsRANComponent',
    'SrsRANState',
    
    # Internal Classes
    'SrsRANProcess',
    'SrsRANConfigGenerator',
    
    # Factory
    'create_srsran_controller',
]
