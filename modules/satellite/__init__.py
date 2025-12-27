"""
RF Arsenal OS - Satellite Module

Satellite communications tracking and reception.
"""

from .satcom import SatelliteCommunications, SatelliteConfig, Satellite as SatelliteInfo, SatellitePass

__all__ = [
    'SatelliteCommunications',
    'SatelliteConfig',
    'SatelliteInfo',
    'SatellitePass',
]

__version__ = '1.0.0'
