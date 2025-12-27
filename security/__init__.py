"""
RF Arsenal OS - Security Module

Anti-forensics, identity management, and physical security.
"""

from .anti_forensics import EncryptedRAMOverlay, ProcessHiding, SecureBoot
from .covert_storage import CovertFileSystem as CovertStorage
from .identity_management import PersonaManager, Persona
from .physical_security import TamperDetection, FaradayMode, PhysicalSecurityMonitor

__all__ = [
    'EncryptedRAMOverlay',
    'ProcessHiding',
    'SecureBoot',
    'CovertStorage',
    'PersonaManager',
    'Persona',
    'TamperDetection',
    'FaradayMode',
    'PhysicalSecurityMonitor',
]

__version__ = '1.0.0'
