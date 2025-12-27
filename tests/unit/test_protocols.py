"""
RF Arsenal OS - Protocol Unit Tests

Unit tests for cellular protocol implementations.
Tests ASN.1, RRC, NAS, S1AP, and GTP modules.
"""

import unittest
import struct
import numpy as np
import sys
import os
import importlib.util

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# Check if Crypto is available for NAS tests
try:
    from Crypto.Cipher import AES
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


def load_asn1_module():
    """Load ASN1 module directly to avoid __init__.py import chain"""
    spec = importlib.util.spec_from_file_location(
        "asn1", 
        os.path.join(PROJECT_ROOT, "core", "protocols", "asn1.py")
    )
    asn1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(asn1)
    return asn1





class TestASN1(unittest.TestCase):
    """Test ASN.1 encoding/decoding"""
    
    @classmethod
    def setUpClass(cls):
        """Load module once for all tests"""
        cls.asn1 = load_asn1_module()
    
    def test_per_encoder_creation(self):
        """Test PER encoder can be created"""
        encoder = self.asn1.PEREncoder()
        self.assertIsNotNone(encoder)
    
    def test_per_integer_encoding(self):
        """Test PER integer encoding with constraint"""
        encoder = self.asn1.PEREncoder()
        
        # Constrained integer with range
        constraint = self.asn1.ASN1Constraint(min_value=0, max_value=15)
        encoded = encoder.encode_constrained_integer(5, constraint)
        
        self.assertIsNotNone(encoded)
        # BitBuffer uses get_bytes() and buffer attribute
        self.assertGreater(len(encoded.get_bytes()), 0)
    
    def test_per_bitstring_encoding(self):
        """Test PER bit string encoding"""
        encoder = self.asn1.PEREncoder()
        
        bits = [1, 0, 1, 1, 0, 0, 1, 0]
        constraint = self.asn1.ASN1Constraint(min_value=1, max_value=16)
        encoded = encoder.encode_bit_string(bits, constraint)
        
        self.assertIsNotNone(encoded)
    
    def test_per_octet_string_encoding(self):
        """Test PER octet string encoding"""
        encoder = self.asn1.PEREncoder()
        
        data = b'\x01\x02\x03\x04'
        constraint = self.asn1.ASN1Constraint(min_value=1, max_value=10)
        encoded = encoder.encode_octet_string(data, constraint)
        
        self.assertIsNotNone(encoded)
    
    def test_asn1_encoder_creation(self):
        """Test ASN1 encoder can be created"""
        encoder = self.asn1.ASN1Encoder()
        self.assertIsNotNone(encoder)
    
    def test_asn1_decoder_creation(self):
        """Test ASN1 decoder can be created"""
        decoder = self.asn1.ASN1Decoder()
        self.assertIsNotNone(decoder)


@unittest.skipUnless(HAS_CRYPTO, "PyCrypto/PyCryptodome not installed")
class TestRRC(unittest.TestCase):
    """Test RRC protocol handling - requires Crypto for imports"""
    
    def test_rrc_handler_creation(self):
        """Test RRC handler can be created"""
        from core.protocols.rrc import RRCHandler
        handler = RRCHandler()
        self.assertIsNotNone(handler)
    
    def test_mib_creation(self):
        """Test MIB can be created"""
        from core.protocols.rrc import MIB
        mib = MIB()
        self.assertIsNotNone(mib)
    
    def test_sib1_creation(self):
        """Test SIB1 can be created"""
        from core.protocols.rrc import SIB1
        sib1 = SIB1()
        self.assertIsNotNone(sib1)
    
    def test_rrc_state_enum(self):
        """Test RRC state enumeration"""
        from core.protocols.rrc import RRCState
        self.assertTrue(hasattr(RRCState, 'IDLE'))
        self.assertTrue(hasattr(RRCState, 'CONNECTED'))


@unittest.skipUnless(HAS_CRYPTO, "PyCrypto/PyCryptodome not installed")
class TestNAS(unittest.TestCase):
    """Test NAS protocol handling - requires Crypto module"""
    
    def test_nas_handler_creation(self):
        """Test NAS handler can be created"""
        from core.protocols.nas import NASHandler
        
        handler = NASHandler()
        self.assertIsNotNone(handler)
    
    def test_attach_request_creation(self):
        """Test AttachRequest can be created"""
        from core.protocols.nas import AttachRequest
        
        req = AttachRequest()
        self.assertIsNotNone(req)


@unittest.skipUnless(HAS_CRYPTO, "PyCrypto/PyCryptodome not installed")
class TestS1AP(unittest.TestCase):
    """Test S1AP protocol handling - requires Crypto for some imports"""
    
    def test_s1ap_message_builder_creation(self):
        """Test S1AP message builder can be created"""
        from core.protocols.s1ap import S1APMessageBuilder
        
        builder = S1APMessageBuilder()
        self.assertIsNotNone(builder)
    
    def test_s1ap_procedure_codes(self):
        """Test S1AP procedure codes exist"""
        from core.protocols.s1ap import S1APProcedureCode
        
        self.assertTrue(hasattr(S1APProcedureCode, 'S1_SETUP'))


@unittest.skipUnless(HAS_CRYPTO, "PyCrypto/PyCryptodome not installed")
class TestGTP(unittest.TestCase):
    """Test GTP protocol handling - requires Crypto for some imports"""
    
    def test_gtpv1_header_creation(self):
        """Test GTPv1 header can be created"""
        from core.protocols.gtp import GTPv1Header
        
        header = GTPv1Header()
        self.assertIsNotNone(header)
    
    def test_gtpv2_header_creation(self):
        """Test GTPv2 header can be created"""
        from core.protocols.gtp import GTPv2Header
        
        header = GTPv2Header()
        self.assertIsNotNone(header)
    
    def test_gtp_message_types(self):
        """Test GTP message types exist"""
        from core.protocols.gtp import GTPv1MessageType, GTPv2MessageType
        
        self.assertTrue(hasattr(GTPv1MessageType, 'ECHO_REQUEST'))
        self.assertTrue(hasattr(GTPv2MessageType, 'ECHO_REQUEST'))


@unittest.skipUnless(HAS_CRYPTO, "PyCrypto/PyCryptodome not installed")
class TestMAC(unittest.TestCase):
    """Test MAC layer handling"""
    
    def test_mac_handler_creation(self):
        """Test MAC handler can be created"""
        from core.protocols.mac import MACHandler
        
        handler = MACHandler()
        self.assertIsNotNone(handler)
    
    def test_rach_procedure_creation(self):
        """Test RACH procedure can be created"""
        from core.protocols.mac import RACHProcedure
        
        rach = RACHProcedure()
        self.assertIsNotNone(rach)


class TestBitBuffer(unittest.TestCase):
    """Test BitBuffer utility class"""
    
    @classmethod
    def setUpClass(cls):
        """Load module once for all tests"""
        cls.asn1 = load_asn1_module()
    
    def test_bitbuffer_creation(self):
        """Test BitBuffer can be created"""
        buf = self.asn1.BitBuffer()
        self.assertIsNotNone(buf)
    
    def test_bitbuffer_write_bits(self):
        """Test writing bits to buffer"""
        buf = self.asn1.BitBuffer()
        buf.write_bits(0b1010, 4)
        
        self.assertGreater(len(buf.buffer), 0)
    
    def test_bitbuffer_read_bits(self):
        """Test reading bits from buffer"""
        buf = self.asn1.BitBuffer()
        buf.write_bits(0b10101010, 8)
        
        buf.byte_offset = 0
        buf.bit_offset = 0
        value = buf.read_bits(4)
        
        self.assertEqual(value, 0b1010)
    
    def test_bitbuffer_to_bytes(self):
        """Test converting buffer to bytes"""
        buf = self.asn1.BitBuffer()
        buf.write_bits(0xFF, 8)
        
        result = buf.get_bytes()
        self.assertIsInstance(result, bytes)


class TestASN1Types(unittest.TestCase):
    """Test ASN.1 type enumeration"""
    
    @classmethod
    def setUpClass(cls):
        """Load module once for all tests"""
        cls.asn1 = load_asn1_module()
    
    def test_asn1_types_exist(self):
        """Test that ASN1 types are defined"""
        expected_types = ['INTEGER', 'BOOLEAN', 'BIT_STRING', 'OCTET_STRING']
        for type_name in expected_types:
            self.assertTrue(hasattr(self.asn1.ASN1Types, type_name), f"Missing: {type_name}")


class TestASN1Constraint(unittest.TestCase):
    """Test ASN.1 constraint handling"""
    
    @classmethod
    def setUpClass(cls):
        """Load module once for all tests"""
        cls.asn1 = load_asn1_module()
    
    def test_constraint_creation(self):
        """Test constraint can be created"""
        constraint = self.asn1.ASN1Constraint(min_value=0, max_value=100)
        
        self.assertEqual(constraint.min_value, 0)
        self.assertEqual(constraint.max_value, 100)


if __name__ == '__main__':
    unittest.main(verbosity=2)
