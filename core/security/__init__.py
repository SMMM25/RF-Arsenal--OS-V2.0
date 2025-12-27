#!/usr/bin/env python3
"""
RF Arsenal OS - Security Certification Module
FIPS 140-3 and TEMPEST compliance implementation

This module provides:
- FIPS 140-3 Level 2/3 compliant cryptographic operations
- TEMPEST emission security controls
- Secure key management and storage
- Compliance auditing and reporting
- Zeroization and secure memory handling

Standards Compliance:
- FIPS 140-3: Security Requirements for Cryptographic Modules
- FIPS 197: Advanced Encryption Standard (AES)
- FIPS 180-4: Secure Hash Standard (SHA-2/SHA-3)
- FIPS 186-5: Digital Signature Standard (DSS)
- NIST SP 800-56A/B: Key Establishment
- NIST SP 800-90A/B/C: Random Number Generation
- TEMPEST/EMSEC: Emission Security Standards
- NSA/CSS EPL: Evaluated Products List requirements
"""

# FIPS 140-3 Core Module
from .fips import (
    FIPSCryptoModule,
    FIPSConfig,
    FIPSSecurityLevel,
    FIPSOperationalState,
    CryptoAlgorithm,
    KeyType,
)

# Crypto Engine
from .fips.crypto_engine import (
    CryptoEngine,
    AESMode,
    HashAlgorithm,
    SignatureAlgorithm,
)

# Key Management
from .fips.key_manager import (
    KeyManager,
    CryptoKey,
    KeyState,
    KeyUsage,
)

# Random Number Generation
from .fips.rng import (
    DRBGEngine,
    DRBGType,
    EntropySource,
)

# Self-Tests
from .fips.self_test import (
    KnownAnswerTests,
    FIPSTestType,
    FIPSTestResult,
)

# TEMPEST Core
from .tempest import (
    TEMPESTLevel,
    ZoneClassification,
    EmissionCategory,
    EmissionProfile,
    TEMPESTException,
)

# TEMPEST Emission Analysis
from .tempest.emission_analyzer import (
    EmissionAnalyzer,
    AnalysisMode,
    SpectralSignature,
)

# TEMPEST Shielding
from .tempest.shielding import (
    ShieldingType,
    ShieldingMonitor,
    ShieldingProfile,
)

# Compliance and Auditing
from .compliance import (
    AuditLogger,
    ComplianceReport,
    AuditEvent,
    ComplianceStatus,
)

__all__ = [
    # FIPS 140-3
    'FIPSCryptoModule',
    'FIPSConfig',
    'FIPSSecurityLevel',
    'FIPSOperationalState',
    'CryptoAlgorithm',
    'KeyType',
    'CryptoEngine',
    'AESMode',
    'HashAlgorithm',
    'SignatureAlgorithm',
    'KeyManager',
    'CryptoKey',
    'KeyState',
    'KeyUsage',
    'DRBGEngine',
    'DRBGType',
    'EntropySource',
    'KnownAnswerTests',
    'FIPSTestType',
    'FIPSTestResult',
    # TEMPEST
    'TEMPESTLevel',
    'ZoneClassification',
    'EmissionCategory',
    'EmissionProfile',
    'TEMPESTException',
    'EmissionAnalyzer',
    'AnalysisMode',
    'SpectralSignature',
    'ShieldingType',
    'ShieldingMonitor',
    'ShieldingProfile',
    # Compliance
    'AuditLogger',
    'ComplianceReport',
    'AuditEvent',
    'ComplianceStatus',
]

__version__ = '1.0.0'
__author__ = 'RF Arsenal Security Team'
__compliance__ = ['FIPS 140-3', 'TEMPEST', 'NSA/CSS EPL']
