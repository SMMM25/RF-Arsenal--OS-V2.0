#!/usr/bin/env python3
"""
RF Arsenal OS - Vehicle Penetration Testing Module

Comprehensive vehicle security testing framework supporting:
- CAN Bus analysis and injection
- UDS (Unified Diagnostic Services) protocol
- Key Fob attacks (capture, replay, rolljam)
- TPMS (Tire Pressure Monitoring System) spoofing
- GPS spoofing for vehicle navigation
- Bluetooth/BLE attacks on infotainment systems
- V2X (Vehicle-to-Everything) communication attacks

Hardware Requirements:
- BladeRF 2.0 micro xA9 (RF attacks: key fob, TPMS, GPS, V2X)
- USB CAN adapter (CANable, ELM327) for CAN bus
- USB Bluetooth adapter for BLE attacks

Author: RF Arsenal Team
License: For authorized security testing only
"""

from .can_bus import (
    CANBusController,
    CANFrame,
    CANInterface,
    CANProtocol,
    CANSpeed,
    CANFilter,
    CANError,
)

from .uds import (
    UDSClient,
    UDSService,
    UDSSession,
    UDSSecurityLevel,
    DiagnosticTroubleCode,
    UDSResponse,
    UDSError,
)

from .key_fob import (
    KeyFobAttack,
    KeyFobProtocol,
    KeyFobCapture,
    RollingCodeAnalyzer,
    RollJamAttack,
    KeyFobReplay,
)

from .tpms import (
    TPMSSpoofer,
    TPMSProtocol,
    TPMSSensor,
    TPMSPacket,
    TPMSManufacturer,
)

from .gps_spoof import (
    GPSSpoofer,
    GPSCoordinate,
    GPSTrajectory,
    GPSSatellite,
    GPSSignalGenerator,
)

from .bluetooth_vehicle import (
    VehicleBLEScanner,
    VehicleBLEAttack,
    OBDBluetoothExploit,
    InfotainmentAttack,
    BLEVulnerability,
)

from .v2x import (
    V2XAttack,
    V2XProtocol,
    BSMSpoofer,
    V2XJammer,
    DSRCAttack,
    CV2XAttack,
)

__all__ = [
    # CAN Bus
    'CANBusController',
    'CANFrame',
    'CANInterface',
    'CANProtocol',
    'CANSpeed',
    'CANFilter',
    'CANError',
    # UDS
    'UDSClient',
    'UDSService',
    'UDSSession',
    'UDSSecurityLevel',
    'DiagnosticTroubleCode',
    'UDSResponse',
    'UDSError',
    # Key Fob
    'KeyFobAttack',
    'KeyFobProtocol',
    'KeyFobCapture',
    'RollingCodeAnalyzer',
    'RollJamAttack',
    'KeyFobReplay',
    # TPMS
    'TPMSSpoofer',
    'TPMSProtocol',
    'TPMSSensor',
    'TPMSPacket',
    'TPMSManufacturer',
    # GPS Spoofing
    'GPSSpoofer',
    'GPSCoordinate',
    'GPSTrajectory',
    'GPSSatellite',
    'GPSSignalGenerator',
    # Bluetooth/BLE
    'VehicleBLEScanner',
    'VehicleBLEAttack',
    'OBDBluetoothExploit',
    'InfotainmentAttack',
    'BLEVulnerability',
    # V2X
    'V2XAttack',
    'V2XProtocol',
    'BSMSpoofer',
    'V2XJammer',
    'DSRCAttack',
    'CV2XAttack',
]

__version__ = '1.0.0'
__author__ = 'RF Arsenal Team'
