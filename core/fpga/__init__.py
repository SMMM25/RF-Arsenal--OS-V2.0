#!/usr/bin/env python3
"""
RF Arsenal OS - FPGA Acceleration Module
Host-side controller for custom BladeRF FPGA images

This module provides:
- FPGA image management (load, flash, verify)
- DSP acceleration interface (FFT, filters, modulators)
- Stealth mode FPGA control
- Real-time signal processing acceleration
- LTE/5G hardware acceleration

Designed for BladeRF 2.0 micro xA9 with custom FPGA images
"""

from .fpga_controller import (
    FPGAController,
    FPGAConfig,
    FPGAStatus,
    FPGAMode,
    AcceleratorType,
)
from .fpga_image_manager import (
    FPGAImageManager,
    FPGAImage,
    ImageType,
    ImageMetadata,
)
from .dsp_accelerator import (
    DSPAccelerator,
    FFTConfig,
    FilterConfig,
    FilterType,
    ModulatorConfig,
    DSPOperation,
    WindowType,
)
from .stealth_fpga import (
    StealthFPGAController,
    StealthProfile,
    FrequencyHopConfig,
    PowerRampConfig,
)
from .lte_accelerator import (
    LTEAccelerator,
    OFDMConfig,
    ChannelConfig,
    LTEFrameConfig,
)

__all__ = [
    # Main controller
    'FPGAController',
    'FPGAConfig',
    'FPGAStatus',
    'FPGAMode',
    'AcceleratorType',
    # Image management
    'FPGAImageManager',
    'FPGAImage',
    'ImageType',
    'ImageMetadata',
    # DSP acceleration
    'DSPAccelerator',
    'FFTConfig',
    'FilterConfig',
    'FilterType',
    'ModulatorConfig',
    'DSPOperation',
    'WindowType',
    # Stealth mode
    'StealthFPGAController',
    'StealthProfile',
    'FrequencyHopConfig',
    'PowerRampConfig',
    # LTE/5G acceleration
    'LTEAccelerator',
    'OFDMConfig',
    'ChannelConfig',
    'LTEFrameConfig',
]

__version__ = '1.0.0'
__author__ = 'RF Arsenal Team'
