"""
RF Arsenal OS - TEMPEST/Van Eck Phreaking Module

Electromagnetic emanation surveillance:
- Video signal reconstruction from EM emissions
- Keyboard emanation capture
- Display content recovery
- EM source scanning and identification

Hardware: BladeRF 2.0 micro xA9 with directional antenna

WARNING: TEMPEST surveillance may be illegal without authorization.
This module is for authorized security research only.
"""

from .tempest_attacks import (
    TEMPESTController,
    TEMPESTMode,
    DisplayType,
    EMSource,
    ReconstructedFrame,
    KeystrokeCapture,
    get_tempest_controller
)

__all__ = [
    'TEMPESTController',
    'TEMPESTMode',
    'DisplayType',
    'EMSource',
    'ReconstructedFrame',
    'KeystrokeCapture',
    'get_tempest_controller'
]

__version__ = '1.0.0'
