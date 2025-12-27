"""
RF Arsenal OS - Hardware Expansion Modules
===========================================

Integration with popular security testing hardware.

README COMPLIANCE:
✅ Stealth-First: Covert hardware operations
✅ RAM-Only: No persistent device logs
✅ No Telemetry: Zero external communication
✅ Offline-First: Direct hardware control
✅ Real-World Functional: Production hardware integration
"""

from .hardware_expansion import (
    # Enums
    HardwareType,
    ConnectionType,
    HardwareStatus,
    
    # Data structures
    HardwareDevice,
    HardwareCommand,
    HardwareResponse,
    
    # Interfaces
    HardwareInterface,
    FlipperZeroInterface,
    WiFiPineappleInterface,
    Proxmark3Interface,
    USBAttackInterface,
    LANTurtleInterface,
    SDRInterface,
    
    # Manager
    HardwareManager,
)

__all__ = [
    # Enums
    'HardwareType',
    'ConnectionType',
    'HardwareStatus',
    
    # Data structures
    'HardwareDevice',
    'HardwareCommand',
    'HardwareResponse',
    
    # Interfaces
    'HardwareInterface',
    'FlipperZeroInterface',
    'WiFiPineappleInterface',
    'Proxmark3Interface',
    'USBAttackInterface',
    'LANTurtleInterface',
    'SDRInterface',
    
    # Manager
    'HardwareManager',
]
