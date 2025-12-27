"""
RF Arsenal OS - Core Module

Core hardware control, signal processing, and system functionality.
Production-grade implementation for RF security testing.
"""

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

__version__ = '1.0.0'
