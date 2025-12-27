"""
RF Arsenal OS - Core Module

Core hardware control, signal processing, and system functionality.
Production-grade implementation for RF security testing.

README COMPLIANCE:
- Real-World Functional Only: No simulation modes in production
- All hardware requirements must be explicitly stated
"""


# =============================================================================
# CUSTOM EXCEPTIONS - README Compliant (No Simulation Fallbacks)
# =============================================================================

class RFArsenalError(Exception):
    """Base exception for RF Arsenal OS."""
    pass


class HardwareRequirementError(RFArsenalError):
    """
    Raised when required hardware is not available.
    
    README COMPLIANCE: Instead of falling back to simulation mode,
    we explicitly inform the user what hardware is required.
    """
    def __init__(self, message: str, required_hardware: str = None, 
                 alternatives: list = None):
        self.required_hardware = required_hardware
        self.alternatives = alternatives or []
        
        full_message = f"HARDWARE REQUIRED: {message}"
        if required_hardware:
            full_message += f"\n  Required: {required_hardware}"
        if alternatives:
            full_message += f"\n  Alternatives: {', '.join(alternatives)}"
        full_message += "\n  Use --dry-run for signal processing testing without hardware."
        
        super().__init__(full_message)


class DependencyError(RFArsenalError):
    """Raised when a required software dependency is missing."""
    def __init__(self, message: str, package: str = None, install_cmd: str = None):
        self.package = package
        self.install_cmd = install_cmd
        
        full_message = f"DEPENDENCY REQUIRED: {message}"
        if package:
            full_message += f"\n  Package: {package}"
        if install_cmd:
            full_message += f"\n  Install: {install_cmd}"
        
        super().__init__(full_message)


class OperationNotPermittedError(RFArsenalError):
    """Raised when an operation violates stealth/OPSEC requirements."""
    pass


# Core hardware interfaces
from .hardware_interface import HardwareConfig, HardwareStatus

# Main hardware controller (renamed from hardware.py to avoid conflict with hardware/ dir)
from .hardware_controller import BladeRFController, HardwarePresets

# Stealth and emergency systems
from .stealth import StealthSystem, NetworkAnonymity
from .emergency import EmergencySystem

# Production-grade hardware drivers
from .hardware.bladerf_driver import (
    BladeRFDriver,
    BladeRFChannel,
    BladeRFGainMode,
    create_bladerf_driver
)

# Calibration framework
from .calibration.rf_calibration import (
    CalibrationManager,
    CalibrationStatus,
    create_calibration_manager
)

__all__ = [
    # Custom exceptions (README compliant - no simulation fallbacks)
    'RFArsenalError',
    'HardwareRequirementError',
    'DependencyError',
    'OperationNotPermittedError',
    # Legacy interfaces
    'BladeRFController',
    'HardwarePresets',
    'StealthSystem',
    'NetworkAnonymity',
    'EmergencySystem',
    # New production interfaces
    'BladeRFDriver',
    'BladeRFChannel',
    'BladeRFGainMode',
    'create_bladerf_driver',
    'CalibrationManager',
    'CalibrationStatus',
    'create_calibration_manager',
]

__version__ = '2.0.0'
