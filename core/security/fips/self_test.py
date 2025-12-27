"""
FIPS 140-3 Self-Test and Health Monitoring Module for RF Arsenal OS.

This module implements comprehensive self-testing capabilities required
for FIPS 140-3 compliance, including power-up self-tests, conditional
self-tests, and continuous health monitoring.

FIPS 140-3 Self-Test Requirements:
- Pre-operational self-tests (power-up)
- Conditional self-tests
- Cryptographic algorithm tests (KAT)
- Software/firmware integrity tests
- Critical functions tests

Author: RF Arsenal Development Team
License: Proprietary - Security Sensitive
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import asyncio
import hashlib
import hmac
import logging
import os
import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import secrets
import binascii

# Import cryptographic primitives
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac as crypto_hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger(__name__)


class FIPSTestType(Enum):
    """Types of FIPS 140-3 self-tests."""
    
    POWER_UP = "power_up"               # Pre-operational self-test
    CONDITIONAL = "conditional"         # Conditional self-test
    CONTINUOUS = "continuous"           # Continuous health check
    PERIODIC = "periodic"               # Scheduled periodic test
    ON_DEMAND = "on_demand"            # Manual/triggered test


class TestCategory(Enum):
    """Categories of self-tests."""
    
    KAT_ENCRYPTION = "kat_encryption"   # Known Answer Test - Encryption
    KAT_DECRYPTION = "kat_decryption"   # Known Answer Test - Decryption
    KAT_HASH = "kat_hash"               # Known Answer Test - Hash
    KAT_MAC = "kat_mac"                 # Known Answer Test - MAC
    KAT_SIGNATURE = "kat_signature"     # Known Answer Test - Digital Signature
    KAT_DRBG = "kat_drbg"              # Known Answer Test - DRBG
    KAT_KDF = "kat_kdf"                # Known Answer Test - KDF
    PAIRWISE_CONSISTENCY = "pairwise"   # Pairwise consistency test
    SOFTWARE_INTEGRITY = "integrity"     # Software/firmware integrity
    CRITICAL_FUNCTIONS = "critical"     # Critical function tests


class FIPSTestResult(Enum):
    """Self-test result codes."""
    
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"
    PENDING = "pending"


class ModuleState(Enum):
    """FIPS 140-3 module states."""
    
    POWER_OFF = "power_off"
    SELF_TEST = "self_test"
    CRYPTO_OFFICER = "crypto_officer"
    USER = "user"
    ERROR = "error"
    ZEROIZATION = "zeroization"


@dataclass
class TestVector:
    """Known Answer Test vector."""
    
    test_name: str
    algorithm: str
    key: bytes
    input_data: bytes
    expected_output: bytes
    iv: Optional[bytes] = None
    aad: Optional[bytes] = None
    tag: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FIPSTestResultRecord:
    """Record of a self-test execution."""
    
    test_id: str
    test_type: FIPSTestType
    test_category: TestCategory
    algorithm: str
    result: FIPSTestResult
    timestamp: datetime
    duration_ms: float
    details: str
    error_code: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class HealthStatus:
    """Module health status."""
    
    healthy: bool
    module_state: ModuleState
    last_test_time: datetime
    tests_passed: int
    tests_failed: int
    uptime_seconds: float
    error_count: int
    warnings: List[str]
    critical_alerts: List[str]


class SelfTestException(Exception):
    """Exception raised when a self-test fails."""
    
    def __init__(self, test_name: str, message: str, error_code: str = "E001"):
        self.test_name = test_name
        self.error_code = error_code
        super().__init__(f"[{error_code}] Self-test '{test_name}' failed: {message}")


class KnownAnswerTests:
    """
    Known Answer Test (KAT) implementation.
    
    Provides pre-computed test vectors for cryptographic
    algorithm validation per FIPS 140-3.
    """
    
    def __init__(self):
        self._test_vectors: Dict[str, List[TestVector]] = {}
        self._load_test_vectors()
    
    def _load_test_vectors(self) -> None:
        """Load FIPS-approved test vectors."""
        # AES-256-CBC test vectors (from NIST)
        self._test_vectors["AES-256-CBC"] = [
            TestVector(
                test_name="AES-256-CBC-ENCRYPT-1",
                algorithm="AES-256-CBC",
                key=bytes.fromhex(
                    "603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4"
                ),
                input_data=bytes.fromhex(
                    "6bc1bee22e409f96e93d7e117393172a"
                ),
                expected_output=bytes.fromhex(
                    "f58c4c04d6e5f1ba779eabfb5f7bfbd6"
                ),
                iv=bytes.fromhex(
                    "000102030405060708090a0b0c0d0e0f"
                )
            ),
            TestVector(
                test_name="AES-256-CBC-ENCRYPT-2",
                algorithm="AES-256-CBC",
                key=bytes.fromhex(
                    "603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4"
                ),
                input_data=bytes.fromhex(
                    "ae2d8a571e03ac9c9eb76fac45af8e51"
                ),
                expected_output=bytes.fromhex(
                    "9cfc4e967edb808d679f777bc6702c7d"
                ),
                iv=bytes.fromhex(
                    "f58c4c04d6e5f1ba779eabfb5f7bfbd6"
                )
            )
        ]
        
        # AES-256-GCM test vectors
        self._test_vectors["AES-256-GCM"] = [
            TestVector(
                test_name="AES-256-GCM-ENCRYPT-1",
                algorithm="AES-256-GCM",
                key=bytes.fromhex(
                    "feffe9928665731c6d6a8f9467308308"
                    "feffe9928665731c6d6a8f9467308308"
                ),
                input_data=bytes.fromhex(
                    "d9313225f88406e5a55909c5aff5269a"
                    "86a7a9531534f7da2e4c303d8a318a72"
                    "1c3c0c95956809532fcf0e2449a6b525"
                    "b16aedf5aa0de657ba637b39"
                ),
                expected_output=bytes.fromhex(
                    "522dc1f099567d07f47f37a32a84427d"
                    "643a8cdcbfe5c0c97598a2bd2555d1aa"
                    "8cb08e48590dbb3da7b08b1056828838"
                    "c5f61e6393ba7a0abcc9f662"
                ),
                iv=bytes.fromhex(
                    "cafebabefacedbaddecaf888"
                ),
                aad=bytes.fromhex(
                    "feedfacedeadbeeffeedfacedeadbeefabaddad2"
                ),
                tag=bytes.fromhex(
                    "76fc6ece0f4e1768cddf8853bb2d551b"
                )
            )
        ]
        
        # SHA-256 test vectors
        self._test_vectors["SHA-256"] = [
            TestVector(
                test_name="SHA-256-HASH-1",
                algorithm="SHA-256",
                key=b"",  # No key for hash
                input_data=b"abc",
                expected_output=bytes.fromhex(
                    "ba7816bf8f01cfea414140de5dae2223"
                    "b00361a396177a9cb410ff61f20015ad"
                )
            ),
            TestVector(
                test_name="SHA-256-HASH-2",
                algorithm="SHA-256",
                key=b"",
                input_data=b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                expected_output=bytes.fromhex(
                    "248d6a61d20638b8e5c026930c3e6039"
                    "a33ce45964ff2167f6ecedd419db06c1"
                )
            )
        ]
        
        # SHA-384 test vectors
        self._test_vectors["SHA-384"] = [
            TestVector(
                test_name="SHA-384-HASH-1",
                algorithm="SHA-384",
                key=b"",
                input_data=b"abc",
                expected_output=bytes.fromhex(
                    "cb00753f45a35e8bb5a03d699ac65007"
                    "272c32ab0eded1631a8b605a43ff5bed"
                    "8086072ba1e7cc2358baeca134c825a7"
                )
            )
        ]
        
        # SHA-512 test vectors
        self._test_vectors["SHA-512"] = [
            TestVector(
                test_name="SHA-512-HASH-1",
                algorithm="SHA-512",
                key=b"",
                input_data=b"abc",
                expected_output=bytes.fromhex(
                    "ddaf35a193617abacc417349ae204131"
                    "12e6fa4e89a97ea20a9eeee64b55d39a"
                    "2192992a274fc1a836ba3c23a3feebbd"
                    "454d4423643ce80e2a9ac94fa54ca49f"
                )
            )
        ]
        
        # HMAC-SHA-256 test vectors
        self._test_vectors["HMAC-SHA-256"] = [
            TestVector(
                test_name="HMAC-SHA-256-1",
                algorithm="HMAC-SHA-256",
                key=bytes.fromhex(
                    "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b"
                    "0b0b0b0b"
                ),
                input_data=b"Hi There",
                expected_output=bytes.fromhex(
                    "b0344c61d8db38535ca8afceaf0bf12b"
                    "881dc200c9833da726e9376c2e32cff7"
                )
            )
        ]
    
    def get_test_vectors(self, algorithm: str) -> List[TestVector]:
        """Get test vectors for an algorithm."""
        return self._test_vectors.get(algorithm, [])
    
    def run_aes_cbc_kat(self, vector: TestVector) -> Tuple[bool, str]:
        """Run AES-CBC Known Answer Test."""
        try:
            cipher = Cipher(
                algorithms.AES(vector.key),
                modes.CBC(vector.iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            result = encryptor.update(vector.input_data) + encryptor.finalize()
            
            if result == vector.expected_output:
                return True, "AES-CBC encryption matches expected output"
            else:
                return False, f"Output mismatch: got {result.hex()}, expected {vector.expected_output.hex()}"
                
        except Exception as e:
            return False, f"Exception during test: {str(e)}"
    
    def run_aes_gcm_kat(self, vector: TestVector) -> Tuple[bool, str]:
        """Run AES-GCM Known Answer Test."""
        try:
            cipher = Cipher(
                algorithms.AES(vector.key),
                modes.GCM(vector.iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            if vector.aad:
                encryptor.authenticate_additional_data(vector.aad)
            
            result = encryptor.update(vector.input_data) + encryptor.finalize()
            tag = encryptor.tag
            
            if result == vector.expected_output and tag == vector.tag:
                return True, "AES-GCM encryption and tag match expected output"
            else:
                details = []
                if result != vector.expected_output:
                    details.append(f"ciphertext mismatch")
                if tag != vector.tag:
                    details.append(f"tag mismatch")
                return False, f"Output mismatch: {', '.join(details)}"
                
        except Exception as e:
            return False, f"Exception during test: {str(e)}"
    
    def run_hash_kat(self, vector: TestVector) -> Tuple[bool, str]:
        """Run hash Known Answer Test."""
        try:
            hash_alg = {
                "SHA-256": hashes.SHA256(),
                "SHA-384": hashes.SHA384(),
                "SHA-512": hashes.SHA512()
            }.get(vector.algorithm)
            
            if not hash_alg:
                return False, f"Unknown hash algorithm: {vector.algorithm}"
            
            digest = hashes.Hash(hash_alg, backend=default_backend())
            digest.update(vector.input_data)
            result = digest.finalize()
            
            if result == vector.expected_output:
                return True, f"{vector.algorithm} hash matches expected output"
            else:
                return False, f"Hash mismatch: got {result.hex()}"
                
        except Exception as e:
            return False, f"Exception during test: {str(e)}"
    
    def run_hmac_kat(self, vector: TestVector) -> Tuple[bool, str]:
        """Run HMAC Known Answer Test."""
        try:
            hash_alg = {
                "HMAC-SHA-256": hashes.SHA256(),
                "HMAC-SHA-384": hashes.SHA384(),
                "HMAC-SHA-512": hashes.SHA512()
            }.get(vector.algorithm)
            
            if not hash_alg:
                return False, f"Unknown HMAC algorithm: {vector.algorithm}"
            
            h = crypto_hmac.HMAC(vector.key, hash_alg, backend=default_backend())
            h.update(vector.input_data)
            result = h.finalize()
            
            if result == vector.expected_output:
                return True, f"{vector.algorithm} matches expected output"
            else:
                return False, f"HMAC mismatch: got {result.hex()}"
                
        except Exception as e:
            return False, f"Exception during test: {str(e)}"


class PairwiseConsistencyTests:
    """
    Pairwise Consistency Tests for asymmetric algorithms.
    
    Verifies that key pairs are mathematically consistent
    per FIPS 140-3 requirements.
    """
    
    def run_rsa_consistency_test(
        self,
        key_size: int = 2048
    ) -> Tuple[bool, str]:
        """
        Run RSA pairwise consistency test.
        
        Generates a key pair, signs data, and verifies signature.
        """
        try:
            # Generate key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend()
            )
            public_key = private_key.public_key()
            
            # Test data
            test_data = secrets.token_bytes(32)
            
            # Sign
            signature = private_key.sign(
                test_data,
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            
            # Verify
            try:
                public_key.verify(
                    signature,
                    test_data,
                    padding.PKCS1v15(),
                    hashes.SHA256()
                )
                return True, f"RSA-{key_size} pairwise consistency verified"
            except InvalidSignature:
                return False, "RSA signature verification failed"
                
        except Exception as e:
            return False, f"Exception during RSA test: {str(e)}"
    
    def run_ecdsa_consistency_test(
        self,
        curve: str = "P-256"
    ) -> Tuple[bool, str]:
        """
        Run ECDSA pairwise consistency test.
        
        Generates a key pair, signs data, and verifies signature.
        """
        try:
            # Select curve
            curves = {
                "P-256": ec.SECP256R1(),
                "P-384": ec.SECP384R1(),
                "P-521": ec.SECP521R1()
            }
            
            curve_obj = curves.get(curve)
            if not curve_obj:
                return False, f"Unknown curve: {curve}"
            
            # Generate key pair
            private_key = ec.generate_private_key(
                curve_obj,
                backend=default_backend()
            )
            public_key = private_key.public_key()
            
            # Test data
            test_data = secrets.token_bytes(32)
            
            # Sign
            signature = private_key.sign(
                test_data,
                ec.ECDSA(hashes.SHA256())
            )
            
            # Verify
            try:
                public_key.verify(
                    signature,
                    test_data,
                    ec.ECDSA(hashes.SHA256())
                )
                return True, f"ECDSA-{curve} pairwise consistency verified"
            except InvalidSignature:
                return False, "ECDSA signature verification failed"
                
        except Exception as e:
            return False, f"Exception during ECDSA test: {str(e)}"


class SoftwareIntegrityTests:
    """
    Software/Firmware Integrity Tests.
    
    Verifies integrity of cryptographic module software
    per FIPS 140-3 requirements.
    """
    
    def __init__(self):
        self._integrity_hashes: Dict[str, bytes] = {}
        self._hmac_key: bytes = secrets.token_bytes(32)
    
    def register_module(
        self,
        module_path: str,
        expected_hash: bytes
    ) -> None:
        """Register a module for integrity checking."""
        self._integrity_hashes[module_path] = expected_hash
    
    def compute_module_hash(
        self,
        module_data: bytes
    ) -> bytes:
        """Compute integrity hash for module data."""
        h = crypto_hmac.HMAC(
            self._hmac_key,
            hashes.SHA256(),
            backend=default_backend()
        )
        h.update(module_data)
        return h.finalize()
    
    def verify_module_integrity(
        self,
        module_path: str,
        module_data: bytes
    ) -> Tuple[bool, str]:
        """
        Verify integrity of a module.
        
        Args:
            module_path: Path/identifier of module
            module_data: Module binary data
            
        Returns:
            Tuple of (passed, details)
        """
        expected_hash = self._integrity_hashes.get(module_path)
        
        if expected_hash is None:
            # No pre-registered hash, compute and store
            computed_hash = self.compute_module_hash(module_data)
            self._integrity_hashes[module_path] = computed_hash
            return True, f"Computed and stored integrity hash for {module_path}"
        
        # Verify against stored hash
        computed_hash = self.compute_module_hash(module_data)
        
        if computed_hash == expected_hash:
            return True, f"Integrity verified for {module_path}"
        else:
            return False, f"Integrity check failed for {module_path}: hash mismatch"
    
    def run_integrity_self_test(self) -> Tuple[bool, str]:
        """
        Run self-test on integrity checking mechanism.
        
        Verifies that the integrity checking algorithm works correctly.
        """
        # Known test data and expected hash
        test_data = b"FIPS 140-3 Integrity Test Data"
        
        # Compute hash
        hash1 = self.compute_module_hash(test_data)
        
        # Verify consistency
        hash2 = self.compute_module_hash(test_data)
        
        if hash1 != hash2:
            return False, "Integrity hash not consistent"
        
        # Verify modification detection
        modified_data = b"FIPS 140-3 Integrity Test Data!"
        hash3 = self.compute_module_hash(modified_data)
        
        if hash1 == hash3:
            return False, "Integrity check failed to detect modification"
        
        return True, "Integrity checking mechanism verified"


class SelfTestManager:
    """
    Main Self-Test Manager for FIPS 140-3 compliance.
    
    Coordinates all self-tests and maintains module state.
    """
    
    def __init__(self):
        # Test components
        self._kat_tests = KnownAnswerTests()
        self._pairwise_tests = PairwiseConsistencyTests()
        self._integrity_tests = SoftwareIntegrityTests()
        
        # State
        self._module_state = ModuleState.POWER_OFF
        self._test_results: List[FIPSTestResultRecord] = []
        self._start_time = datetime.utcnow()
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0
        }
        
        # Callbacks
        self._failure_callbacks: List[Callable[[FIPSTestResultRecord], None]] = []
    
    def register_failure_callback(
        self,
        callback: Callable[[FIPSTestResultRecord], None]
    ) -> None:
        """Register callback for test failures."""
        self._failure_callbacks.append(callback)
    
    def _record_result(
        self,
        test_type: FIPSTestType,
        test_category: TestCategory,
        algorithm: str,
        passed: bool,
        details: str,
        duration_ms: float,
        error_code: Optional[str] = None
    ) -> FIPSTestResultRecord:
        """Record a test result."""
        result = FIPSTestResult.PASS if passed else FIPSTestResult.FAIL
        
        record = FIPSTestResultRecord(
            test_id=secrets.token_hex(8),
            test_type=test_type,
            test_category=test_category,
            algorithm=algorithm,
            result=result,
            timestamp=datetime.utcnow(),
            duration_ms=duration_ms,
            details=details,
            error_code=error_code if not passed else None,
            error_message=details if not passed else None
        )
        
        with self._lock:
            self._test_results.append(record)
            self._stats["total_tests"] += 1
            
            if passed:
                self._stats["passed"] += 1
            else:
                self._stats["failed"] += 1
                # Trigger failure callbacks
                for callback in self._failure_callbacks:
                    try:
                        callback(record)
                    except Exception as e:
                        logger.error(f"Failure callback error: {e}")
        
        return record
    
    def run_power_up_self_tests(self) -> Tuple[bool, List[FIPSTestResultRecord]]:
        """
        Run all power-up self-tests.
        
        Required by FIPS 140-3 before module can process
        cryptographic operations.
        
        Returns:
            Tuple of (all_passed, test_records)
        """
        with self._lock:
            self._module_state = ModuleState.SELF_TEST
        
        results = []
        all_passed = True
        
        logger.info("Starting FIPS 140-3 power-up self-tests")
        
        # 1. Software Integrity Test
        start = time.time()
        passed, details = self._integrity_tests.run_integrity_self_test()
        duration = (time.time() - start) * 1000
        
        record = self._record_result(
            FIPSTestType.POWER_UP,
            TestCategory.SOFTWARE_INTEGRITY,
            "HMAC-SHA-256",
            passed,
            details,
            duration,
            "E100" if not passed else None
        )
        results.append(record)
        all_passed = all_passed and passed
        
        # 2. AES-256-CBC KAT
        for vector in self._kat_tests.get_test_vectors("AES-256-CBC"):
            start = time.time()
            passed, details = self._kat_tests.run_aes_cbc_kat(vector)
            duration = (time.time() - start) * 1000
            
            record = self._record_result(
                FIPSTestType.POWER_UP,
                TestCategory.KAT_ENCRYPTION,
                "AES-256-CBC",
                passed,
                details,
                duration,
                "E101" if not passed else None
            )
            results.append(record)
            all_passed = all_passed and passed
        
        # 3. AES-256-GCM KAT
        for vector in self._kat_tests.get_test_vectors("AES-256-GCM"):
            start = time.time()
            passed, details = self._kat_tests.run_aes_gcm_kat(vector)
            duration = (time.time() - start) * 1000
            
            record = self._record_result(
                FIPSTestType.POWER_UP,
                TestCategory.KAT_ENCRYPTION,
                "AES-256-GCM",
                passed,
                details,
                duration,
                "E102" if not passed else None
            )
            results.append(record)
            all_passed = all_passed and passed
        
        # 4. SHA-256 KAT
        for vector in self._kat_tests.get_test_vectors("SHA-256"):
            start = time.time()
            passed, details = self._kat_tests.run_hash_kat(vector)
            duration = (time.time() - start) * 1000
            
            record = self._record_result(
                FIPSTestType.POWER_UP,
                TestCategory.KAT_HASH,
                "SHA-256",
                passed,
                details,
                duration,
                "E103" if not passed else None
            )
            results.append(record)
            all_passed = all_passed and passed
        
        # 5. SHA-384 KAT
        for vector in self._kat_tests.get_test_vectors("SHA-384"):
            start = time.time()
            passed, details = self._kat_tests.run_hash_kat(vector)
            duration = (time.time() - start) * 1000
            
            record = self._record_result(
                FIPSTestType.POWER_UP,
                TestCategory.KAT_HASH,
                "SHA-384",
                passed,
                details,
                duration,
                "E104" if not passed else None
            )
            results.append(record)
            all_passed = all_passed and passed
        
        # 6. SHA-512 KAT
        for vector in self._kat_tests.get_test_vectors("SHA-512"):
            start = time.time()
            passed, details = self._kat_tests.run_hash_kat(vector)
            duration = (time.time() - start) * 1000
            
            record = self._record_result(
                FIPSTestType.POWER_UP,
                TestCategory.KAT_HASH,
                "SHA-512",
                passed,
                details,
                duration,
                "E105" if not passed else None
            )
            results.append(record)
            all_passed = all_passed and passed
        
        # 7. HMAC-SHA-256 KAT
        for vector in self._kat_tests.get_test_vectors("HMAC-SHA-256"):
            start = time.time()
            passed, details = self._kat_tests.run_hmac_kat(vector)
            duration = (time.time() - start) * 1000
            
            record = self._record_result(
                FIPSTestType.POWER_UP,
                TestCategory.KAT_MAC,
                "HMAC-SHA-256",
                passed,
                details,
                duration,
                "E106" if not passed else None
            )
            results.append(record)
            all_passed = all_passed and passed
        
        # 8. RSA Pairwise Consistency Test
        for key_size in [2048, 3072, 4096]:
            start = time.time()
            passed, details = self._pairwise_tests.run_rsa_consistency_test(key_size)
            duration = (time.time() - start) * 1000
            
            record = self._record_result(
                FIPSTestType.POWER_UP,
                TestCategory.PAIRWISE_CONSISTENCY,
                f"RSA-{key_size}",
                passed,
                details,
                duration,
                "E107" if not passed else None
            )
            results.append(record)
            all_passed = all_passed and passed
        
        # 9. ECDSA Pairwise Consistency Test
        for curve in ["P-256", "P-384", "P-521"]:
            start = time.time()
            passed, details = self._pairwise_tests.run_ecdsa_consistency_test(curve)
            duration = (time.time() - start) * 1000
            
            record = self._record_result(
                FIPSTestType.POWER_UP,
                TestCategory.PAIRWISE_CONSISTENCY,
                f"ECDSA-{curve}",
                passed,
                details,
                duration,
                "E108" if not passed else None
            )
            results.append(record)
            all_passed = all_passed and passed
        
        # Update module state
        with self._lock:
            if all_passed:
                self._module_state = ModuleState.CRYPTO_OFFICER
                logger.info(f"Power-up self-tests PASSED ({len(results)} tests)")
            else:
                self._module_state = ModuleState.ERROR
                failed_count = sum(1 for r in results if r.result == FIPSTestResult.FAIL)
                logger.error(
                    f"Power-up self-tests FAILED ({failed_count}/{len(results)} failed)"
                )
        
        return all_passed, results
    
    def run_conditional_self_test(
        self,
        algorithm: str
    ) -> Tuple[bool, FIPSTestResultRecord]:
        """
        Run conditional self-test for specific algorithm.
        
        Required before first use of algorithm after key generation
        or import.
        
        Args:
            algorithm: Algorithm to test
            
        Returns:
            Tuple of (passed, test_record)
        """
        start = time.time()
        passed = False
        details = ""
        category = TestCategory.KAT_ENCRYPTION
        
        if algorithm.startswith("AES"):
            vectors = self._kat_tests.get_test_vectors(algorithm)
            if vectors:
                if "GCM" in algorithm:
                    passed, details = self._kat_tests.run_aes_gcm_kat(vectors[0])
                else:
                    passed, details = self._kat_tests.run_aes_cbc_kat(vectors[0])
            else:
                passed = False
                details = f"No test vectors for {algorithm}"
        
        elif algorithm.startswith("SHA"):
            category = TestCategory.KAT_HASH
            vectors = self._kat_tests.get_test_vectors(algorithm)
            if vectors:
                passed, details = self._kat_tests.run_hash_kat(vectors[0])
            else:
                passed = False
                details = f"No test vectors for {algorithm}"
        
        elif algorithm.startswith("HMAC"):
            category = TestCategory.KAT_MAC
            vectors = self._kat_tests.get_test_vectors(algorithm)
            if vectors:
                passed, details = self._kat_tests.run_hmac_kat(vectors[0])
            else:
                passed = False
                details = f"No test vectors for {algorithm}"
        
        elif algorithm.startswith("RSA"):
            category = TestCategory.PAIRWISE_CONSISTENCY
            key_size = int(algorithm.split("-")[1]) if "-" in algorithm else 2048
            passed, details = self._pairwise_tests.run_rsa_consistency_test(key_size)
        
        elif algorithm.startswith("ECDSA"):
            category = TestCategory.PAIRWISE_CONSISTENCY
            curve = algorithm.split("-")[1] if "-" in algorithm else "P-256"
            passed, details = self._pairwise_tests.run_ecdsa_consistency_test(curve)
        
        else:
            details = f"Unknown algorithm: {algorithm}"
        
        duration = (time.time() - start) * 1000
        
        record = self._record_result(
            FIPSTestType.CONDITIONAL,
            category,
            algorithm,
            passed,
            details,
            duration,
            "E200" if not passed else None
        )
        
        return passed, record
    
    def run_periodic_self_tests(self) -> Tuple[bool, List[FIPSTestResultRecord]]:
        """
        Run periodic self-tests.
        
        Should be called at regular intervals per security policy.
        
        Returns:
            Tuple of (all_passed, test_records)
        """
        # Run subset of power-up tests
        results = []
        all_passed = True
        
        # Run one test from each category
        test_configs = [
            ("AES-256-CBC", TestCategory.KAT_ENCRYPTION, self._kat_tests.run_aes_cbc_kat),
            ("SHA-256", TestCategory.KAT_HASH, self._kat_tests.run_hash_kat),
            ("HMAC-SHA-256", TestCategory.KAT_MAC, self._kat_tests.run_hmac_kat),
        ]
        
        for algorithm, category, test_func in test_configs:
            vectors = self._kat_tests.get_test_vectors(algorithm)
            if vectors:
                start = time.time()
                passed, details = test_func(vectors[0])
                duration = (time.time() - start) * 1000
                
                record = self._record_result(
                    FIPSTestType.PERIODIC,
                    category,
                    algorithm,
                    passed,
                    details,
                    duration,
                    "E300" if not passed else None
                )
                results.append(record)
                all_passed = all_passed and passed
        
        return all_passed, results
    
    def get_health_status(self) -> HealthStatus:
        """Get current module health status."""
        with self._lock:
            uptime = (datetime.utcnow() - self._start_time).total_seconds()
            
            last_test_time = (
                self._test_results[-1].timestamp if self._test_results
                else self._start_time
            )
            
            warnings = []
            critical_alerts = []
            
            # Check for recent failures
            recent_failures = [
                r for r in self._test_results[-10:]
                if r.result == FIPSTestResult.FAIL
            ]
            
            if recent_failures:
                warnings.append(
                    f"{len(recent_failures)} test failures in last 10 tests"
                )
            
            # Check module state
            if self._module_state == ModuleState.ERROR:
                critical_alerts.append("Module in ERROR state")
            
            return HealthStatus(
                healthy=(
                    self._module_state not in [ModuleState.ERROR, ModuleState.POWER_OFF]
                    and self._stats["failed"] == 0
                ),
                module_state=self._module_state,
                last_test_time=last_test_time,
                tests_passed=self._stats["passed"],
                tests_failed=self._stats["failed"],
                uptime_seconds=uptime,
                error_count=self._stats["errors"],
                warnings=warnings,
                critical_alerts=critical_alerts
            )
    
    def get_module_state(self) -> ModuleState:
        """Get current module state."""
        with self._lock:
            return self._module_state
    
    def get_test_history(
        self,
        limit: int = 100,
        test_type: Optional[FIPSTestType] = None
    ) -> List[FIPSTestResultRecord]:
        """Get test history."""
        with self._lock:
            records = self._test_results.copy()
        
        if test_type:
            records = [r for r in records if r.test_type == test_type]
        
        return records[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get test statistics."""
        with self._lock:
            return {
                **self._stats,
                "module_state": self._module_state.value,
                "uptime_seconds": (
                    datetime.utcnow() - self._start_time
                ).total_seconds()
            }
    
    def enter_error_state(self, reason: str) -> None:
        """Force module into error state."""
        with self._lock:
            self._module_state = ModuleState.ERROR
            logger.critical(f"Module entering ERROR state: {reason}")


# Export public API
__all__ = [
    "FIPSTestType",
    "TestCategory",
    "FIPSTestResult",
    "ModuleState",
    "TestVector",
    "FIPSTestResultRecord",
    "HealthStatus",
    "SelfTestException",
    "KnownAnswerTests",
    "PairwiseConsistencyTests",
    "SoftwareIntegrityTests",
    "SelfTestManager",
]
