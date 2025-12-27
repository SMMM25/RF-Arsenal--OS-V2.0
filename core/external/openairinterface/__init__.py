"""
RF Arsenal OS - OpenAirInterface Integration Package

Production-grade OpenAirInterface (OAI) integration for 5G NR and LTE.
Supports gNB, UE, and 5G Core Network deployment.

Features:
- Full 5G NR Standalone (SA) support
- LTE eNB/UE support
- Docker-based 5G Core deployment
- Stealth-aware operation
- Real-time configuration
- AI control integration

Example:
    from core.external.openairinterface import OAIController, OAIConfig
    
    config = OAIConfig()
    controller = OAIController(config=config)
    controller.initialize()
    controller.start_gnb(sa_mode=True)
"""

from .oai_controller import (
    # Controller
    OAIController,
    
    # Configuration
    OAIConfig,
    OAIRFConfig,
    OAICellConfig,
    OAICoreConfig,
    
    # Components and States
    OAIComponent,
    OAIState,
    OAIDeploymentMode,
    
    # Process Management
    OAIProcess,
    
    # Configuration Generator
    OAIConfigGenerator,
    
    # Factory Function
    create_oai_controller,
)

__version__ = '1.0.0'

__all__ = [
    # Main Controller
    'OAIController',
    
    # Configuration Classes
    'OAIConfig',
    'OAIRFConfig',
    'OAICellConfig',
    'OAICoreConfig',
    
    # Enums
    'OAIComponent',
    'OAIState',
    'OAIDeploymentMode',
    
    # Internal Classes
    'OAIProcess',
    'OAIConfigGenerator',
    
    # Factory
    'create_oai_controller',
]
