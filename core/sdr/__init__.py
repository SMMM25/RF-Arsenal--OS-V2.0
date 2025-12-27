#!/usr/bin/env python3
"""
RF Arsenal OS - SoapySDR Hardware Abstraction Layer

Universal SDR backend supporting multiple hardware platforms:
- BladeRF (1.0, 2.0 micro xA4/xA9)
- HackRF One
- RTL-SDR (RTL2832U)
- USRP (B200, B210, N200, X300)
- LimeSDR (Mini, USB)
- Airspy (R2, Mini, HF+)
- PlutoSDR (ADALM-PLUTO)
- SDRplay (RSP1, RSP2, RSPdx)

This module provides unified access to all SDR hardware through
the SoapySDR abstraction layer.
"""

from .soapy_backend import (
    # Core classes
    SoapySDRDevice,
    SDRManager,
    DSPProcessor,
    
    # Data classes
    SDRDeviceInfo,
    StreamConfig,
    IQCapture,
    
    # Enums
    SDRType,
    StreamFormat,
    StreamDirection,
    GainMode,
    
    # Singleton accessor
    get_sdr_manager,
    
    # Constants
    SDR_SPECIFICATIONS,
    SOAPY_AVAILABLE,
)

__all__ = [
    # Core classes
    'SoapySDRDevice',
    'SDRManager',
    'DSPProcessor',
    
    # Data classes
    'SDRDeviceInfo',
    'StreamConfig',
    'IQCapture',
    
    # Enums
    'SDRType',
    'StreamFormat',
    'StreamDirection',
    'GainMode',
    
    # Singleton accessor
    'get_sdr_manager',
    
    # Constants
    'SDR_SPECIFICATIONS',
    'SOAPY_AVAILABLE',
]

__version__ = "1.0.0"
