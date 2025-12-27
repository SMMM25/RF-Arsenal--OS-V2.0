"""
RF Arsenal OS - ADS-B Module

Aircraft tracking and (authorized) signal injection:
- ADS-B reception at 1090 MHz
- Real-time aircraft tracking
- Position, altitude, velocity decoding
- Mode S decoding
- Signal injection (AUTHORIZED RESEARCH ONLY)

Hardware: BladeRF 2.0 micro xA9

WARNING: ADS-B spoofing is ILLEGAL in most jurisdictions.
This module is for authorized security research only.
"""

from .adsb_attacks import (
    ADSBController,
    Aircraft,
    ADSBMessage,
    ADSBMessageType,
    get_adsb_controller
)

__all__ = [
    'ADSBController',
    'Aircraft',
    'ADSBMessage',
    'ADSBMessageType',
    'get_adsb_controller'
]

__version__ = '1.0.0'
