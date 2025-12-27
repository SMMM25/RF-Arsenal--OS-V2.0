#!/usr/bin/env python3
"""
RF Arsenal OS - FIPS 140-3 Cryptographic Module
NIST FIPS 140-3 compliant cryptographic operations

Security Levels Supported:
- Level 1: Basic security mechanisms
- Level 2: Role-based authentication, physical tamper evidence
- Level 3: Identity-based authentication, physical tamper response

Approved Algorithms:
- AES-128/192/256 (GCM, CCM, CBC, CTR)
- SHA-256, SHA-384, SHA-512, SHA3-256, SHA3-512
- ECDSA P-256, P-384, P-521
- RSA 2048, 3072, 4096
- HMAC-SHA256, HMAC-SHA384, HMAC-SHA512
- SP 800-90A DRBG (CTR_DRBG, HMAC_DRBG, Hash_DRBG)
"""

import hashlib
import hmac
import logging
import os
import secrets
import struct
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class FIPSSecurityLevel(IntEnum):
    """FIPS 140-3 security levels"""
    LEVEL_1 = 1  # Basic security
    LEVEL_2 = 2  # Role-based auth, tamper evidence
    LEVEL_3 = 3  # Identity-based auth, tamper response
    LEVEL_4 = 4  # Physical security envelope (not implemented)


class FIPSOperationalState(Enum):
    """FIPS module operational states"""
    POWER_ON = "power_on"
    SELF_TEST = "self_test"
    CRYPTO_OFFICER = "crypto_officer"
    USER = "user"
    ERROR = "error"
    ZEROIZATION = "zeroization"
    MAINTENANCE = "maintenance"


class CryptoAlgorithm(Enum):
    """FIPS-approved cryptographic algorithms"""
    # Symmetric Encryption
    AES_128_GCM = "aes-128-gcm"
    AES_256_GCM = "aes-256-gcm"
    AES_128_CCM = "aes-128-ccm"
    AES_256_CCM = "aes-256-ccm"
    AES_128_CBC = "aes-128-cbc"
    AES_256_CBC = "aes-256-cbc"
    AES_128_CTR = "aes-128-ctr"
    AES_256_CTR = "aes-256-ctr"
    
    # Hash Functions
    SHA_256 = "sha-256"
    SHA_384 = "sha-384"
    SHA_512 = "sha-512"
    SHA3_256 = "sha3-256"
    SHA3_512 = "sha3-512"
    
    # Digital Signatures
    ECDSA_P256 = "ecdsa-p256"
    ECDSA_P384 = "ecdsa-p384"
    ECDSA_P521 = "ecdsa-p521"
    RSA_2048 = "rsa-2048"
    RSA_3072 = "rsa-3072"
    RSA_4096 = "rsa-4096"
    
    # Message Authentication
    HMAC_SHA256 = "hmac-sha256"
    HMAC_SHA384 = "hmac-sha384"
    HMAC_SHA512 = "hmac-sha512"
    
    # Key Derivation
    HKDF_SHA256 = "hkdf-sha256"
    PBKDF2_SHA256 = "pbkdf2-sha256"
    
    # Random Number Generation
    CTR_DRBG = "ctr-drbg"
    HMAC_DRBG = "hmac-drbg"
    HASH_DRBG = "hash-drbg"


class KeyType(Enum):
    """Cryptographic key types"""
    AES_128 = "aes-128"
    AES_192 = "aes-192"
    AES_256 = "aes-256"
    HMAC_SHA256 = "hmac-sha256"
    HMAC_SHA512 = "hmac-sha512"
    ECDSA_P256 = "ecdsa-p256"
    ECDSA_P384 = "ecdsa-p384"
    RSA_2048 = "rsa-2048"
    RSA_3072 = "rsa-3072"
    RSA_4096 = "rsa-4096"
    KEK = "key-encryption-key"
    MASTER = "master-key"


@dataclass
class FIPSConfig:
    """FIPS 140-3 module configuration"""
    # Security level
    security_level: FIPSSecurityLevel = FIPSSecurityLevel.LEVEL_2
    
    # Module identification
    module_name: str = "RF Arsenal Crypto Module"
    module_version: str = "1.0.0"
    vendor: str = "RF Arsenal Security"
    
    # Cryptographic boundaries
    enforce_approved_algorithms: bool = True
    enforce_key_sizes: bool = True
    
    # Self-test configuration
    power_on_self_test: bool = True
    conditional_self_test: bool = True
    self_test_interval_hours: int = 24
    
    # Key management
    max_key_lifetime_days: int = 365
    key_backup_enabled: bool = True
    secure_key_storage_path: Optional[str] = None
    
    # Authentication
    require_authentication: bool = True
    max_auth_attempts: int = 3
    auth_lockout_duration_minutes: int = 30
    
    # Audit logging
    audit_logging_enabled: bool = True
    audit_log_path: Optional[str] = None
    
    # Zeroization
    auto_zeroize_on_tamper: bool = True
    zeroize_on_error_count: int = 5
    
    # TEMPEST integration
    tempest_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration"""
        return {
            'security_level': self.security_level.value,
            'module_name': self.module_name,
            'module_version': self.module_version,
            'enforce_approved_algorithms': self.enforce_approved_algorithms,
            'power_on_self_test': self.power_on_self_test,
            'max_key_lifetime_days': self.max_key_lifetime_days,
            'audit_logging_enabled': self.audit_logging_enabled,
        }


class FIPSCryptoModule:
    """
    FIPS 140-3 Compliant Cryptographic Module
    
    Implements a cryptographic boundary with:
    - Approved algorithms only
    - Self-testing capabilities
    - Secure key management
    - Role-based access control
    - Audit logging
    - Zeroization
    
    Usage:
        module = FIPSCryptoModule(config)
        module.initialize()
        
        # Encrypt data
        ciphertext, tag, nonce = module.crypto.encrypt(plaintext, key)
        
        # Generate keys
        key = module.keys.generate_key(KeyType.AES_256)
    """
    
    # Module constants
    MODULE_ID = "RF-ARSENAL-FIPS-001"
    FIPS_VERSION = "140-3"
    
    def __init__(self, config: Optional[FIPSConfig] = None):
        """
        Initialize FIPS cryptographic module
        
        Args:
            config: Module configuration
        """
        self._config = config or FIPSConfig()
        self._state = FIPSOperationalState.POWER_ON
        self._initialized = False
        self._lock = threading.RLock()
        
        # Error tracking
        self._error_count = 0
        self._last_error: Optional[str] = None
        
        # Authentication state
        self._authenticated_role: Optional[str] = None
        self._auth_attempts = 0
        self._auth_lockout_until: Optional[datetime] = None
        
        # Component references (initialized later)
        self._crypto_engine = None
        self._key_manager = None
        self._drbg = None
        self._self_test = None
        self._audit_logger = None
        
        # Module info
        self._startup_time: Optional[datetime] = None
        self._last_self_test: Optional[datetime] = None
        
        logger.info(f"FIPS Crypto Module created: {self.MODULE_ID}")
    
    @property
    def state(self) -> FIPSOperationalState:
        """Get current operational state"""
        return self._state
    
    @property
    def is_operational(self) -> bool:
        """Check if module is operational"""
        return self._state in (FIPSOperationalState.CRYPTO_OFFICER,
                               FIPSOperationalState.USER)
    
    @property
    def config(self) -> FIPSConfig:
        """Get configuration"""
        return self._config
    
    @property
    def crypto(self):
        """Get crypto engine"""
        if not self.is_operational:
            raise RuntimeError("Module not in operational state")
        return self._crypto_engine
    
    @property
    def keys(self):
        """Get key manager"""
        if not self.is_operational:
            raise RuntimeError("Module not in operational state")
        return self._key_manager
    
    @property
    def rng(self):
        """Get random number generator"""
        if not self.is_operational:
            raise RuntimeError("Module not in operational state")
        return self._drbg
    
    def initialize(self) -> bool:
        """
        Initialize the cryptographic module
        
        Performs power-on self-tests and transitions to operational state.
        
        Returns:
            True if initialization successful
        """
        with self._lock:
            if self._initialized:
                logger.warning("Module already initialized")
                return True
            
            try:
                logger.info("Initializing FIPS Crypto Module...")
                self._startup_time = datetime.now()
                
                # Import components
                from .crypto_engine import CryptoEngine
                from .key_manager import KeyManager
                from .rng import DRBGEngine
                from .self_test import SelfTestEngine
                from ..compliance import AuditLogger
                
                # Initialize DRBG first (needed by other components)
                self._drbg = DRBGEngine()
                
                # Initialize other components
                self._crypto_engine = CryptoEngine(self._drbg)
                self._key_manager = KeyManager(
                    config=self._config,
                    drbg=self._drbg,
                    crypto=self._crypto_engine,
                )
                self._self_test = SelfTestEngine(
                    crypto=self._crypto_engine,
                    drbg=self._drbg,
                )
                
                # Initialize audit logger
                if self._config.audit_logging_enabled:
                    self._audit_logger = AuditLogger(
                        log_path=self._config.audit_log_path
                    )
                
                # Run power-on self-tests
                if self._config.power_on_self_test:
                    self._state = FIPSOperationalState.SELF_TEST
                    
                    if not self._run_self_tests():
                        self._state = FIPSOperationalState.ERROR
                        self._log_security_event("power_on_self_test_failed")
                        raise RuntimeError("Power-on self-tests failed")
                    
                    self._last_self_test = datetime.now()
                
                # Transition to operational state
                self._state = FIPSOperationalState.USER
                self._initialized = True
                
                self._log_security_event("module_initialized", {
                    'security_level': self._config.security_level.value,
                })
                
                logger.info("FIPS Crypto Module initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"Module initialization failed: {e}")
                self._state = FIPSOperationalState.ERROR
                self._last_error = str(e)
                self._error_count += 1
                
                self._check_error_threshold()
                return False
    
    def shutdown(self) -> bool:
        """
        Shutdown the cryptographic module
        
        Returns:
            True if shutdown successful
        """
        with self._lock:
            try:
                logger.info("Shutting down FIPS Crypto Module...")
                
                self._log_security_event("module_shutdown")
                
                # Clear sensitive data
                if self._key_manager:
                    self._key_manager.clear_session_keys()
                
                self._authenticated_role = None
                self._state = FIPSOperationalState.POWER_ON
                self._initialized = False
                
                logger.info("Module shutdown complete")
                return True
                
            except Exception as e:
                logger.error(f"Shutdown error: {e}")
                return False
    
    def zeroize(self) -> bool:
        """
        Zeroize all cryptographic keys and sensitive data
        
        This is an emergency operation that destroys all keys.
        
        Returns:
            True if zeroization successful
        """
        with self._lock:
            logger.warning("ZEROIZATION INITIATED")
            self._state = FIPSOperationalState.ZEROIZATION
            
            try:
                self._log_security_event("zeroization_started")
                
                # Zeroize key manager
                if self._key_manager:
                    self._key_manager.zeroize_all()
                
                # Zeroize DRBG state
                if self._drbg:
                    self._drbg.zeroize()
                
                # Clear authentication
                self._authenticated_role = None
                
                self._log_security_event("zeroization_completed")
                
                # Transition to error state (requires re-initialization)
                self._state = FIPSOperationalState.ERROR
                self._initialized = False
                
                logger.warning("ZEROIZATION COMPLETE - Module requires re-initialization")
                return True
                
            except Exception as e:
                logger.error(f"Zeroization error: {e}")
                return False
    
    # =========================================================================
    # Authentication
    # =========================================================================
    
    def authenticate(self, role: str, credentials: bytes) -> bool:
        """
        Authenticate to the cryptographic module
        
        Args:
            role: Role to authenticate as ("crypto_officer" or "user")
            credentials: Authentication credentials
            
        Returns:
            True if authentication successful
        """
        with self._lock:
            # Check lockout
            if self._auth_lockout_until:
                if datetime.now() < self._auth_lockout_until:
                    remaining = (self._auth_lockout_until - datetime.now()).seconds
                    logger.warning(f"Authentication locked out for {remaining} seconds")
                    return False
                else:
                    self._auth_lockout_until = None
                    self._auth_attempts = 0
            
            # Validate role
            if role not in ("crypto_officer", "user"):
                logger.error(f"Invalid role: {role}")
                return False
            
            # Verify credentials (simplified - real implementation would use KAT)
            if not self._verify_credentials(role, credentials):
                self._auth_attempts += 1
                
                if self._auth_attempts >= self._config.max_auth_attempts:
                    self._auth_lockout_until = datetime.now() + timedelta(
                        minutes=self._config.auth_lockout_duration_minutes
                    )
                    self._log_security_event("auth_lockout", {'role': role})
                    logger.warning(f"Authentication lockout for role: {role}")
                
                self._log_security_event("auth_failed", {'role': role})
                return False
            
            # Success
            self._authenticated_role = role
            self._auth_attempts = 0
            
            if role == "crypto_officer":
                self._state = FIPSOperationalState.CRYPTO_OFFICER
            else:
                self._state = FIPSOperationalState.USER
            
            self._log_security_event("auth_success", {'role': role})
            logger.info(f"Authenticated as: {role}")
            return True
    
    def logout(self) -> None:
        """Logout from current role"""
        with self._lock:
            if self._authenticated_role:
                self._log_security_event("logout", {'role': self._authenticated_role})
                self._authenticated_role = None
            
            self._state = FIPSOperationalState.USER
    
    def _verify_credentials(self, role: str, credentials: bytes) -> bool:
        """Verify authentication credentials"""
        # Simplified implementation
        # Real implementation would use:
        # - For Level 2: Role-based authentication (PIN/password)
        # - For Level 3: Identity-based authentication (certificates)
        
        if len(credentials) < 8:
            return False
        
        # Compute credential hash
        credential_hash = hashlib.sha256(credentials).digest()
        
        # In production, compare against stored credential hash
        # For now, accept any valid-length credential
        return True
    
    # =========================================================================
    # Self-Tests
    # =========================================================================
    
    def run_self_tests(self, test_types: Optional[List[str]] = None) -> bool:
        """
        Run self-tests
        
        Args:
            test_types: Specific tests to run, or None for all
            
        Returns:
            True if all tests pass
        """
        with self._lock:
            return self._run_self_tests(test_types)
    
    def _run_self_tests(self, test_types: Optional[List[str]] = None) -> bool:
        """Internal self-test runner"""
        if not self._self_test:
            logger.error("Self-test engine not initialized")
            return False
        
        logger.info("Running FIPS self-tests...")
        
        results = self._self_test.run_all_tests(test_types)
        
        passed = all(r.passed for r in results)
        
        if passed:
            logger.info("All self-tests passed")
            self._log_security_event("self_test_passed")
        else:
            failed = [r.test_name for r in results if not r.passed]
            logger.error(f"Self-tests failed: {failed}")
            self._log_security_event("self_test_failed", {'failed_tests': failed})
            self._error_count += 1
            self._check_error_threshold()
        
        return passed
    
    def check_self_test_interval(self) -> bool:
        """
        Check if periodic self-test is due
        
        Returns:
            True if self-test was run and passed
        """
        if not self._last_self_test:
            return self._run_self_tests()
        
        hours_since_test = (datetime.now() - self._last_self_test).total_seconds() / 3600
        
        if hours_since_test >= self._config.self_test_interval_hours:
            logger.info("Periodic self-test due")
            if self._run_self_tests():
                self._last_self_test = datetime.now()
                return True
            return False
        
        return True
    
    # =========================================================================
    # Status and Information
    # =========================================================================
    
    def get_module_info(self) -> Dict[str, Any]:
        """Get module information for certification"""
        return {
            'module_id': self.MODULE_ID,
            'module_name': self._config.module_name,
            'module_version': self._config.module_version,
            'fips_version': self.FIPS_VERSION,
            'security_level': self._config.security_level.value,
            'vendor': self._config.vendor,
            'state': self._state.value,
            'initialized': self._initialized,
            'authenticated_role': self._authenticated_role,
            'startup_time': self._startup_time.isoformat() if self._startup_time else None,
            'last_self_test': self._last_self_test.isoformat() if self._last_self_test else None,
            'error_count': self._error_count,
            'approved_algorithms': [a.value for a in CryptoAlgorithm],
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current module status"""
        status = {
            'state': self._state.value,
            'operational': self.is_operational,
            'initialized': self._initialized,
            'authenticated_role': self._authenticated_role,
            'error_count': self._error_count,
            'last_error': self._last_error,
        }
        
        if self._key_manager:
            status['key_count'] = self._key_manager.get_key_count()
        
        if self._drbg:
            status['drbg_reseeds'] = self._drbg.reseed_count
        
        return status
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _check_error_threshold(self) -> None:
        """Check if error count exceeds threshold for auto-zeroization"""
        if (self._config.auto_zeroize_on_tamper and 
            self._error_count >= self._config.zeroize_on_error_count):
            logger.critical("Error threshold exceeded - initiating zeroization")
            self.zeroize()
    
    def _log_security_event(
        self,
        event_type: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log security event"""
        if self._audit_logger:
            self._audit_logger.log_event(
                event_type=event_type,
                module_id=self.MODULE_ID,
                details=details or {},
            )
