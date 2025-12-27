#!/usr/bin/env python3
"""
RF Arsenal OS - BladeRF Core Modules
Hardware: BladeRF 2.0 micro xA9

Modules:
- MIMO 2x2: Full spatial multiplexing, beamforming, DoA
- AGC: Hardware Automatic Gain Control exposure
- Calibration: RF calibration and DC offset correction
- Frequency Hopping: Adaptive frequency hopping support
- XB-200: Transverter board for HF/VHF coverage (9 kHz - 300 MHz)
"""

from .mimo import BladeRFMIMO, MIMOConfig, MIMOMode, DOAResult, BeamPattern
from .agc import BladeRFAGC, AGCConfig, AGCMode, CalibrationMode, CalibrationResult
from .frequency_hopping import (
    BladeRFFrequencyHopping, HoppingConfig, HoppingPattern, 
    HoppingSpeed, TrackedTransmitter
)
from .xb200 import BladeRFXB200, XB200Config, XB200Band, XB200FilterBank

__all__ = [
    # MIMO
    'BladeRFMIMO',
    'MIMOConfig', 
    'MIMOMode',
    'DOAResult',
    'BeamPattern',
    # AGC
    'BladeRFAGC',
    'AGCConfig',
    'AGCMode',
    'CalibrationMode',
    'CalibrationResult',
    # Frequency Hopping
    'BladeRFFrequencyHopping',
    'HoppingConfig',
    'HoppingPattern',
    'HoppingSpeed',
    'TrackedTransmitter',
    # XB-200
    'BladeRFXB200',
    'XB200Config',
    'XB200Band',
    'XB200FilterBank',
]
