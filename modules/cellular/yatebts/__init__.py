"""
RF Arsenal OS - YateBTS Integration Module

Real GSM/LTE Base Station operations:
- IMSI/IMEI Capture
- SMS Interception
- Voice Call Interception
- Location Tracking
- Device Targeting

Hardware: BladeRF 2.0 micro xA9
Software: YateBTS, Yate

WARNING: These capabilities require proper legal authorization.
Use only in authorized testing environments.
"""

from .yatebts_controller import (
    YateBTSController,
    BTSConfig,
    BTSMode,
    CellularBand,
    CapturedDevice,
    get_yatebts_controller
)

__all__ = [
    'YateBTSController',
    'BTSConfig',
    'BTSMode',
    'CellularBand',
    'CapturedDevice',
    'get_yatebts_controller'
]

__version__ = '1.0.0'
