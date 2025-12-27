"""
RF Arsenal OS - Hardware Abstraction Layer

Production-grade hardware drivers for SDR platforms.
Provides unified interface for BladeRF 2.0 xA9 and other SDRs.
"""

from .bladerf_driver import (
    BladeRFDriver,
    BladeRFChannel,
    BladeRFGainMode,
    BladeRFFormat,
    BladeRFModule,
    BladeRFLoopback,
    ChannelConfig,
    StreamConfig,
    CalibrationData,
    BladeRFDeviceInfo,
    create_bladerf_driver
)

# Import from hardware_controller for backward compatibility
# This allows `from core.hardware import BladeRFController` to work
from ..hardware_controller import BladeRFController, HardwarePresets, FrequencyBand
from ..hardware_interface import HardwareConfig, HardwareStatus

# Alias for backward compatibility
HardwareController = BladeRFController

__all__ = [
    # New driver interface
    'BladeRFDriver',
    'BladeRFChannel',
    'BladeRFGainMode',
    'BladeRFFormat',
    'BladeRFModule',
    'BladeRFLoopback',
    'ChannelConfig',
    'StreamConfig',
    'CalibrationData',
    'BladeRFDeviceInfo',
    'create_bladerf_driver',
    # Legacy interface (backward compatible)
    'BladeRFController',
    'HardwareController',
    'HardwarePresets',
    'FrequencyBand',
    'HardwareConfig',
    'HardwareStatus',
]

__version__ = '1.0.0'
