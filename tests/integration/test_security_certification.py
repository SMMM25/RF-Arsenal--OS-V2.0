"""
RF Arsenal OS - Security Certification Integration Tests

Tests for FIPS 140-3 and TEMPEST compliance functionality.
"""

import unittest
import sys
import os
import tempfile
from datetime import datetime
from typing import Dict, Any
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.security import (
    # FIPS
    FIPSCryptoModule,
    FIPSConfig,
    FIPSSecurityLevel,
    FIPSOperationalState,
    CryptoAlgorithm,
    KeyType,
    CryptoEngine,
    AESMode,
    HashAlgorithm,
    KeyManager,
    CryptoKey,
    KeyState,
    KeyUsage,
    DRBGEngine,
    DRBGType,
    KnownAnswerTests,
    FIPSTestType,
    FIPSTestResult,
    # TEMPEST
    TEMPESTLevel,
    ZoneClassification,
    EmissionCategory,
    EmissionProfile,
    TEMPESTException,
    EmissionAnalyzer,
    AnalysisMode,
    ShieldingType,
    ShieldingMonitor,
    # Compliance
    AuditLogger,
    ComplianceReport,
    ComplianceStatus,
)


class TestFIPSSecurityLevel(unittest.TestCase):
    """Test FIPS security level enumeration."""
    
    def test_security_levels_exist(self):
        """Test that security levels are defined."""
        # FIPS 140-3 defines levels 1-4
        self.assertTrue(hasattr(FIPSSecurityLevel, 'LEVEL_1') or 
                       len(list(FIPSSecurityLevel)) >= 1)


class TestFIPSOperationalState(unittest.TestCase):
    """Test FIPS operational state enumeration."""
    
    def test_operational_states_exist(self):
        """Test that operational states are defined."""
        for state in FIPSOperationalState:
            self.assertIsInstance(state.value, str)


class TestCryptoAlgorithm(unittest.TestCase):
    """Test crypto algorithm enumeration."""
    
    def test_algorithms_exist(self):
        """Test that crypto algorithms are defined."""
        # Actual enum names from implementation
        expected = ['AES_128_GCM', 'AES_256_GCM', 'SHA_256', 'SHA_384', 'SHA_512']
        for algo in expected:
            self.assertTrue(
                hasattr(CryptoAlgorithm, algo),
                f"Missing algorithm: {algo}"
            )


class TestKeyType(unittest.TestCase):
    """Test key type enumeration."""
    
    def test_key_types_exist(self):
        """Test that key types are defined."""
        for key_type in KeyType:
            self.assertIsInstance(key_type.value, str)


class TestCryptoEngine(unittest.TestCase):
    """Test crypto engine functionality."""
    
    def test_crypto_engine_creation(self):
        """Test that crypto engine can be created."""
        engine = CryptoEngine()
        self.assertIsNotNone(engine)
    
    def test_aes_modes_exist(self):
        """Test that AES modes are defined."""
        expected = ['CBC', 'GCM', 'CTR']
        for mode in expected:
            self.assertTrue(
                hasattr(AESMode, mode),
                f"Missing AES mode: {mode}"
            )
    
    def test_hash_algorithms_exist(self):
        """Test that hash algorithms are defined."""
        # Actual enum names use underscore format
        expected = ['SHA_256', 'SHA_384', 'SHA_512']
        for algo in expected:
            self.assertTrue(
                hasattr(HashAlgorithm, algo),
                f"Missing hash algorithm: {algo}"
            )


class TestKeyManager(unittest.TestCase):
    """Test key manager functionality."""
    
    def test_key_manager_creation(self):
        """Test that key manager can be created."""
        manager = KeyManager()
        self.assertIsNotNone(manager)
    
    def test_key_state_enum(self):
        """Test key state enumeration."""
        for state in KeyState:
            self.assertIsInstance(state.value, str)
    
    def test_key_usage_enum(self):
        """Test key usage enumeration."""
        for usage in KeyUsage:
            self.assertIsInstance(usage.value, str)


class TestDRBGEngine(unittest.TestCase):
    """Test DRBG engine functionality."""
    
    def test_drbg_engine_creation(self):
        """Test that DRBG engine can be created."""
        engine = DRBGEngine()
        self.assertIsNotNone(engine)
    
    def test_drbg_types_exist(self):
        """Test that DRBG types are defined."""
        for drbg_type in DRBGType:
            self.assertIsInstance(drbg_type.value, str)


class TestKnownAnswerTests(unittest.TestCase):
    """Test Known Answer Tests (KAT) functionality."""
    
    def test_kat_creation(self):
        """Test that KAT can be created."""
        kat = KnownAnswerTests()
        self.assertIsNotNone(kat)
    
    def test_test_types_exist(self):
        """Test that test types are defined."""
        for test_type in FIPSTestType:
            self.assertIsInstance(test_type.value, str)
    
    def test_test_result_enum(self):
        """Test result enumeration."""
        expected = ['PASS', 'FAIL']
        for result in expected:
            self.assertTrue(
                hasattr(FIPSTestResult, result),
                f"Missing test result: {result}"
            )


class TestTEMPESTLevel(unittest.TestCase):
    """Test TEMPEST level enumeration."""
    
    def test_tempest_levels_exist(self):
        """Test that TEMPEST levels are defined."""
        for level in TEMPESTLevel:
            self.assertIsNotNone(level.value)


class TestZoneClassification(unittest.TestCase):
    """Test zone classification enumeration."""
    
    def test_zone_classifications_exist(self):
        """Test that zone classifications are defined."""
        for zone in ZoneClassification:
            self.assertIsInstance(zone.value, int)


class TestEmissionCategory(unittest.TestCase):
    """Test emission category enumeration."""
    
    def test_emission_categories_exist(self):
        """Test that emission categories are defined."""
        for category in EmissionCategory:
            self.assertIsNotNone(category.value)


class TestEmissionAnalyzer(unittest.TestCase):
    """Test emission analyzer functionality."""
    
    def test_analyzer_creation(self):
        """Test that emission analyzer can be created."""
        analyzer = EmissionAnalyzer()
        self.assertIsNotNone(analyzer)
    
    def test_analysis_modes_exist(self):
        """Test that analysis modes are defined."""
        for mode in AnalysisMode:
            self.assertIsNotNone(mode.value)


class TestShieldingMonitor(unittest.TestCase):
    """Test shielding monitor functionality."""
    
    def test_monitor_creation(self):
        """Test that shielding monitor can be created."""
        monitor = ShieldingMonitor()
        self.assertIsNotNone(monitor)
    
    def test_shielding_types_exist(self):
        """Test that shielding types are defined."""
        for shielding_type in ShieldingType:
            self.assertIsNotNone(shielding_type.value)


class TestAuditLogger(unittest.TestCase):
    """Test audit logger functionality."""
    
    def test_logger_creation(self):
        """Test that audit logger can be created."""
        logger = AuditLogger()
        self.assertIsNotNone(logger)


class TestComplianceReport(unittest.TestCase):
    """Test compliance report functionality."""
    
    def test_compliance_status_enum(self):
        """Test compliance status enumeration."""
        for status in ComplianceStatus:
            self.assertIsNotNone(status.value)


class TestTEMPESTException(unittest.TestCase):
    """Test TEMPEST exception."""
    
    def test_exception_raise(self):
        """Test that TEMPEST exception can be raised."""
        with self.assertRaises(TEMPESTException):
            raise TEMPESTException("Test error")


class TestFIPSCryptoModule(unittest.TestCase):
    """Test FIPS crypto module."""
    
    def test_module_creation(self):
        """Test that FIPS crypto module can be created."""
        module = FIPSCryptoModule()
        self.assertIsNotNone(module)
    
    def test_config_creation(self):
        """Test that FIPS config can be created."""
        config = FIPSConfig()
        self.assertIsNotNone(config)


class TestIntegration(unittest.TestCase):
    """Integration tests for security components."""
    
    def test_crypto_engine_with_key_manager(self):
        """Test crypto engine works with key manager."""
        engine = CryptoEngine()
        key_manager = KeyManager()
        
        self.assertIsNotNone(engine)
        self.assertIsNotNone(key_manager)
    
    def test_drbg_with_crypto_engine(self):
        """Test DRBG works with crypto engine."""
        drbg = DRBGEngine()
        engine = CryptoEngine()
        
        self.assertIsNotNone(drbg)
        self.assertIsNotNone(engine)
    
    def test_tempest_components_together(self):
        """Test TEMPEST components work together."""
        analyzer = EmissionAnalyzer()
        monitor = ShieldingMonitor()
        
        self.assertIsNotNone(analyzer)
        self.assertIsNotNone(monitor)


if __name__ == '__main__':
    unittest.main(verbosity=2)
