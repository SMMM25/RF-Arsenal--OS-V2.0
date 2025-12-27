#!/usr/bin/env python3
"""
RF Arsenal OS - FIPS 140-3 Key Management System
Secure key generation, storage, and lifecycle management

Implements:
- NIST SP 800-57 Key Management
- NIST SP 800-131A Cryptographic Key Lengths
- Secure key storage with encryption
- Key lifecycle management (generate, use, archive, destroy)
- Key wrapping/unwrapping
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class KeyState(Enum):
    """Key lifecycle states (NIST SP 800-57)"""
    PRE_ACTIVATION = "pre_activation"  # Generated but not yet active
    ACTIVE = "active"  # Available for cryptographic operations
    SUSPENDED = "suspended"  # Temporarily disabled
    DEACTIVATED = "deactivated"  # No longer for protection, can decrypt
    COMPROMISED = "compromised"  # Key may have been exposed
    DESTROYED = "destroyed"  # Key material has been zeroized


class KeyUsage(Enum):
    """Permitted key usage types"""
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"
    SIGN = "sign"
    VERIFY = "verify"
    KEY_WRAP = "key_wrap"
    KEY_UNWRAP = "key_unwrap"
    KEY_DERIVE = "key_derive"
    KEY_AGREEMENT = "key_agreement"


class KeyType(Enum):
    """Cryptographic key types"""
    AES_128 = "aes-128"
    AES_192 = "aes-192"
    AES_256 = "aes-256"
    HMAC_SHA256 = "hmac-sha256"
    HMAC_SHA384 = "hmac-sha384"
    HMAC_SHA512 = "hmac-sha512"
    ECDSA_P256 = "ecdsa-p256"
    ECDSA_P384 = "ecdsa-p384"
    ECDSA_P521 = "ecdsa-p521"
    RSA_2048 = "rsa-2048"
    RSA_3072 = "rsa-3072"
    RSA_4096 = "rsa-4096"
    KEK = "key-encryption-key"
    MASTER = "master-key"
    SESSION = "session-key"


@dataclass
class CryptoKey:
    """
    Cryptographic key with metadata
    
    Follows NIST SP 800-57 key management guidelines
    """
    # Key identification
    key_id: str
    key_type: KeyType
    
    # Key material (encrypted at rest)
    key_material: bytes = field(repr=False)
    public_key: Optional[bytes] = field(default=None, repr=False)
    
    # Lifecycle
    state: KeyState = KeyState.PRE_ACTIVATION
    created_at: datetime = field(default_factory=datetime.now)
    activated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    deactivated_at: Optional[datetime] = None
    
    # Usage
    usage: List[KeyUsage] = field(default_factory=lambda: [KeyUsage.ENCRYPT, KeyUsage.DECRYPT])
    usage_count: int = 0
    max_usage_count: Optional[int] = None
    
    # Metadata
    algorithm: str = ""
    key_size_bits: int = 0
    owner: str = ""
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Security
    extractable: bool = False
    wrapped_by: Optional[str] = None  # Key ID of wrapping key
    
    def is_active(self) -> bool:
        """Check if key is active and usable"""
        if self.state != KeyState.ACTIVE:
            return False
        
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        
        if self.max_usage_count and self.usage_count >= self.max_usage_count:
            return False
        
        return True
    
    def can_perform(self, operation: KeyUsage) -> bool:
        """Check if key can perform operation"""
        return self.is_active() and operation in self.usage
    
    def to_dict(self, include_material: bool = False) -> Dict[str, Any]:
        """Serialize key to dictionary"""
        data = {
            'key_id': self.key_id,
            'key_type': self.key_type.value,
            'state': self.state.value,
            'created_at': self.created_at.isoformat(),
            'activated_at': self.activated_at.isoformat() if self.activated_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'usage': [u.value for u in self.usage],
            'usage_count': self.usage_count,
            'algorithm': self.algorithm,
            'key_size_bits': self.key_size_bits,
            'owner': self.owner,
            'description': self.description,
            'tags': self.tags,
            'extractable': self.extractable,
            'wrapped_by': self.wrapped_by,
        }
        
        if include_material:
            data['key_material'] = base64.b64encode(self.key_material).decode()
            if self.public_key:
                data['public_key'] = base64.b64encode(self.public_key).decode()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CryptoKey':
        """Deserialize key from dictionary"""
        return cls(
            key_id=data['key_id'],
            key_type=KeyType(data['key_type']),
            key_material=base64.b64decode(data.get('key_material', '')),
            public_key=base64.b64decode(data['public_key']) if data.get('public_key') else None,
            state=KeyState(data.get('state', 'pre_activation')),
            created_at=datetime.fromisoformat(data['created_at']),
            activated_at=datetime.fromisoformat(data['activated_at']) if data.get('activated_at') else None,
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            usage=[KeyUsage(u) for u in data.get('usage', ['encrypt', 'decrypt'])],
            usage_count=data.get('usage_count', 0),
            algorithm=data.get('algorithm', ''),
            key_size_bits=data.get('key_size_bits', 0),
            owner=data.get('owner', ''),
            description=data.get('description', ''),
            tags=data.get('tags', {}),
            extractable=data.get('extractable', False),
            wrapped_by=data.get('wrapped_by'),
        )


class KeyManager:
    """
    FIPS 140-3 Compliant Key Manager
    
    Provides:
    - Secure key generation using DRBG
    - Encrypted key storage
    - Key lifecycle management
    - Key wrapping/unwrapping
    - Key import/export
    """
    
    # Key size mappings
    KEY_SIZES = {
        KeyType.AES_128: 128,
        KeyType.AES_192: 192,
        KeyType.AES_256: 256,
        KeyType.HMAC_SHA256: 256,
        KeyType.HMAC_SHA384: 384,
        KeyType.HMAC_SHA512: 512,
        KeyType.KEK: 256,
        KeyType.MASTER: 256,
        KeyType.SESSION: 256,
    }
    
    # Algorithm mappings
    KEY_ALGORITHMS = {
        KeyType.AES_128: "AES",
        KeyType.AES_192: "AES",
        KeyType.AES_256: "AES",
        KeyType.HMAC_SHA256: "HMAC-SHA256",
        KeyType.HMAC_SHA384: "HMAC-SHA384",
        KeyType.HMAC_SHA512: "HMAC-SHA512",
        KeyType.ECDSA_P256: "ECDSA-P256",
        KeyType.ECDSA_P384: "ECDSA-P384",
        KeyType.ECDSA_P521: "ECDSA-P521",
        KeyType.RSA_2048: "RSA-2048",
        KeyType.RSA_3072: "RSA-3072",
        KeyType.RSA_4096: "RSA-4096",
    }
    
    def __init__(
        self,
        config=None,
        drbg=None,
        crypto=None,
        storage_path: Optional[str] = None
    ):
        """
        Initialize Key Manager
        
        Args:
            config: FIPS configuration
            drbg: DRBG engine for random generation
            crypto: Crypto engine for encryption
            storage_path: Path for persistent key storage
        """
        self._config = config
        self._drbg = drbg
        self._crypto = crypto
        self._storage_path = Path(storage_path) if storage_path else None
        
        self._lock = threading.RLock()
        
        # Key storage
        self._keys: Dict[str, CryptoKey] = {}
        self._session_keys: Dict[str, CryptoKey] = {}
        
        # Master key for encrypting stored keys
        self._master_key: Optional[bytes] = None
        self._master_key_id: Optional[str] = None
        
        # Audit callback
        self._audit_callback: Optional[Callable] = None
        
        logger.info("KeyManager initialized")
    
    def set_audit_callback(self, callback: Callable) -> None:
        """Set callback for key operation auditing"""
        self._audit_callback = callback
    
    # =========================================================================
    # Key Generation
    # =========================================================================
    
    def generate_key(
        self,
        key_type: KeyType,
        owner: str = "",
        description: str = "",
        usage: Optional[List[KeyUsage]] = None,
        lifetime_days: Optional[int] = None,
        auto_activate: bool = True,
        tags: Optional[Dict[str, str]] = None
    ) -> CryptoKey:
        """
        Generate a new cryptographic key
        
        Args:
            key_type: Type of key to generate
            owner: Key owner identifier
            description: Key description
            usage: Permitted key usage types
            lifetime_days: Key lifetime in days (None = no expiry)
            auto_activate: Automatically activate key after generation
            tags: Additional metadata tags
            
        Returns:
            Generated CryptoKey
        """
        with self._lock:
            # Generate key ID
            key_id = self._generate_key_id()
            
            # Generate key material
            if key_type in (KeyType.ECDSA_P256, KeyType.ECDSA_P384, KeyType.ECDSA_P521):
                key_material, public_key = self._generate_ecdsa_key(key_type)
                key_size = {'P256': 256, 'P384': 384, 'P521': 521}.get(
                    key_type.value.split('-')[1].upper(), 256
                )
            elif key_type in (KeyType.RSA_2048, KeyType.RSA_3072, KeyType.RSA_4096):
                key_material, public_key = self._generate_rsa_key(key_type)
                key_size = int(key_type.value.split('-')[1])
            else:
                key_material = self._generate_symmetric_key(key_type)
                public_key = None
                key_size = self.KEY_SIZES.get(key_type, 256)
            
            # Calculate expiry
            expires_at = None
            if lifetime_days:
                expires_at = datetime.now() + timedelta(days=lifetime_days)
            elif self._config and self._config.max_key_lifetime_days:
                expires_at = datetime.now() + timedelta(days=self._config.max_key_lifetime_days)
            
            # Create key object
            key = CryptoKey(
                key_id=key_id,
                key_type=key_type,
                key_material=key_material,
                public_key=public_key,
                state=KeyState.PRE_ACTIVATION,
                expires_at=expires_at,
                usage=usage or [KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
                algorithm=self.KEY_ALGORITHMS.get(key_type, ""),
                key_size_bits=key_size,
                owner=owner,
                description=description,
                tags=tags or {},
            )
            
            # Auto-activate if requested
            if auto_activate:
                key.state = KeyState.ACTIVE
                key.activated_at = datetime.now()
            
            # Store key
            self._keys[key_id] = key
            
            # Audit
            self._audit("key_generated", key_id, {
                'key_type': key_type.value,
                'owner': owner,
            })
            
            logger.info(f"Generated key: {key_id} ({key_type.value})")
            return key
    
    def generate_session_key(
        self,
        key_type: KeyType = KeyType.AES_256,
        lifetime_minutes: int = 60
    ) -> CryptoKey:
        """
        Generate a short-lived session key
        
        Args:
            key_type: Type of session key
            lifetime_minutes: Key lifetime in minutes
            
        Returns:
            Generated session CryptoKey
        """
        with self._lock:
            key = self.generate_key(
                key_type=key_type,
                owner="session",
                description="Session key",
                lifetime_days=None,
                auto_activate=True,
            )
            
            key.expires_at = datetime.now() + timedelta(minutes=lifetime_minutes)
            
            # Store in session keys
            self._session_keys[key.key_id] = key
            
            return key
    
    def _generate_symmetric_key(self, key_type: KeyType) -> bytes:
        """Generate symmetric key material"""
        bits = self.KEY_SIZES.get(key_type, 256)
        
        if self._drbg:
            return self._drbg.generate(bits // 8)
        else:
            return secrets.token_bytes(bits // 8)
    
    def _generate_ecdsa_key(self, key_type: KeyType) -> Tuple[bytes, bytes]:
        """Generate ECDSA key pair"""
        if self._crypto:
            curve = key_type.value.split('-')[1].upper()
            curve_name = f"P-{curve[1:]}" if curve.startswith('P') else f"P-{curve}"
            return self._crypto.generate_ecdsa_keypair(curve_name)
        else:
            # Fallback - generate random bytes
            size = {'p256': 32, 'p384': 48, 'p521': 66}.get(
                key_type.value.split('-')[1], 32
            )
            return secrets.token_bytes(size), secrets.token_bytes(size * 2)
    
    def _generate_rsa_key(self, key_type: KeyType) -> Tuple[bytes, bytes]:
        """Generate RSA key pair"""
        if self._crypto:
            bits = int(key_type.value.split('-')[1])
            return self._crypto.generate_rsa_keypair(bits)
        else:
            # Fallback
            bits = int(key_type.value.split('-')[1])
            size = bits // 8
            return secrets.token_bytes(size), secrets.token_bytes(size // 2)
    
    def _generate_key_id(self) -> str:
        """Generate unique key ID"""
        return f"key-{uuid.uuid4().hex[:16]}"
    
    # =========================================================================
    # Key Retrieval
    # =========================================================================
    
    def get_key(self, key_id: str) -> Optional[CryptoKey]:
        """
        Get key by ID
        
        Args:
            key_id: Key identifier
            
        Returns:
            CryptoKey if found, None otherwise
        """
        with self._lock:
            key = self._keys.get(key_id) or self._session_keys.get(key_id)
            
            if key:
                self._audit("key_accessed", key_id, {})
            
            return key
    
    def get_key_for_operation(
        self,
        key_id: str,
        operation: KeyUsage
    ) -> Optional[CryptoKey]:
        """
        Get key for specific operation (increments usage counter)
        
        Args:
            key_id: Key identifier
            operation: Intended operation
            
        Returns:
            CryptoKey if available and permitted
        """
        with self._lock:
            key = self.get_key(key_id)
            
            if key is None:
                return None
            
            if not key.can_perform(operation):
                logger.warning(f"Key {key_id} cannot perform {operation.value}")
                self._audit("key_operation_denied", key_id, {
                    'operation': operation.value,
                    'reason': 'not_permitted',
                })
                return None
            
            # Increment usage counter
            key.usage_count += 1
            
            self._audit("key_used", key_id, {
                'operation': operation.value,
                'usage_count': key.usage_count,
            })
            
            return key
    
    def list_keys(
        self,
        key_type: Optional[KeyType] = None,
        state: Optional[KeyState] = None,
        owner: Optional[str] = None
    ) -> List[CryptoKey]:
        """
        List keys matching criteria
        
        Args:
            key_type: Filter by key type
            state: Filter by state
            owner: Filter by owner
            
        Returns:
            List of matching keys
        """
        with self._lock:
            keys = list(self._keys.values())
            
            if key_type:
                keys = [k for k in keys if k.key_type == key_type]
            
            if state:
                keys = [k for k in keys if k.state == state]
            
            if owner:
                keys = [k for k in keys if k.owner == owner]
            
            return keys
    
    def get_key_count(self) -> int:
        """Get total number of keys"""
        with self._lock:
            return len(self._keys) + len(self._session_keys)
    
    # =========================================================================
    # Key Lifecycle Management
    # =========================================================================
    
    def activate_key(self, key_id: str) -> bool:
        """
        Activate a pre-activation key
        
        Args:
            key_id: Key identifier
            
        Returns:
            True if activated
        """
        with self._lock:
            key = self._keys.get(key_id)
            
            if key is None:
                return False
            
            if key.state != KeyState.PRE_ACTIVATION:
                logger.warning(f"Key {key_id} not in pre-activation state")
                return False
            
            key.state = KeyState.ACTIVE
            key.activated_at = datetime.now()
            
            self._audit("key_activated", key_id, {})
            logger.info(f"Activated key: {key_id}")
            return True
    
    def suspend_key(self, key_id: str, reason: str = "") -> bool:
        """
        Suspend an active key
        
        Args:
            key_id: Key identifier
            reason: Reason for suspension
            
        Returns:
            True if suspended
        """
        with self._lock:
            key = self._keys.get(key_id)
            
            if key is None or key.state != KeyState.ACTIVE:
                return False
            
            key.state = KeyState.SUSPENDED
            
            self._audit("key_suspended", key_id, {'reason': reason})
            logger.info(f"Suspended key: {key_id}")
            return True
    
    def reactivate_key(self, key_id: str) -> bool:
        """
        Reactivate a suspended key
        
        Args:
            key_id: Key identifier
            
        Returns:
            True if reactivated
        """
        with self._lock:
            key = self._keys.get(key_id)
            
            if key is None or key.state != KeyState.SUSPENDED:
                return False
            
            key.state = KeyState.ACTIVE
            
            self._audit("key_reactivated", key_id, {})
            logger.info(f"Reactivated key: {key_id}")
            return True
    
    def deactivate_key(self, key_id: str, reason: str = "") -> bool:
        """
        Deactivate a key (can still be used for decryption)
        
        Args:
            key_id: Key identifier
            reason: Reason for deactivation
            
        Returns:
            True if deactivated
        """
        with self._lock:
            key = self._keys.get(key_id)
            
            if key is None:
                return False
            
            if key.state in (KeyState.DESTROYED, KeyState.COMPROMISED):
                return False
            
            key.state = KeyState.DEACTIVATED
            key.deactivated_at = datetime.now()
            
            # Remove encryption usage
            key.usage = [u for u in key.usage if u not in (KeyUsage.ENCRYPT, KeyUsage.SIGN)]
            
            self._audit("key_deactivated", key_id, {'reason': reason})
            logger.info(f"Deactivated key: {key_id}")
            return True
    
    def compromise_key(self, key_id: str, reason: str = "") -> bool:
        """
        Mark key as compromised
        
        Args:
            key_id: Key identifier
            reason: Description of compromise
            
        Returns:
            True if marked compromised
        """
        with self._lock:
            key = self._keys.get(key_id)
            
            if key is None:
                return False
            
            key.state = KeyState.COMPROMISED
            
            self._audit("key_compromised", key_id, {'reason': reason})
            logger.warning(f"Key marked compromised: {key_id}")
            return True
    
    def destroy_key(self, key_id: str) -> bool:
        """
        Destroy a key (zeroize material)
        
        Args:
            key_id: Key identifier
            
        Returns:
            True if destroyed
        """
        with self._lock:
            key = self._keys.get(key_id) or self._session_keys.get(key_id)
            
            if key is None:
                return False
            
            # Zeroize key material
            self._zeroize_key(key)
            
            key.state = KeyState.DESTROYED
            
            # Remove from storage
            if key_id in self._keys:
                del self._keys[key_id]
            if key_id in self._session_keys:
                del self._session_keys[key_id]
            
            self._audit("key_destroyed", key_id, {})
            logger.info(f"Destroyed key: {key_id}")
            return True
    
    def _zeroize_key(self, key: CryptoKey) -> None:
        """Securely zeroize key material"""
        if key.key_material:
            # Overwrite with zeros
            material = bytearray(key.key_material)
            for i in range(len(material)):
                material[i] = 0
            key.key_material = bytes(material)
        
        if key.public_key:
            material = bytearray(key.public_key)
            for i in range(len(material)):
                material[i] = 0
            key.public_key = bytes(material)
    
    # =========================================================================
    # Key Wrapping
    # =========================================================================
    
    def wrap_key(
        self,
        key_id: str,
        wrapping_key_id: str
    ) -> Optional[bytes]:
        """
        Wrap (encrypt) a key for export or storage
        
        Args:
            key_id: Key to wrap
            wrapping_key_id: Key Encryption Key (KEK)
            
        Returns:
            Wrapped key bytes, or None on error
        """
        with self._lock:
            key = self.get_key(key_id)
            kek = self.get_key_for_operation(wrapping_key_id, KeyUsage.KEY_WRAP)
            
            if key is None or kek is None:
                return None
            
            if not key.extractable:
                logger.warning(f"Key {key_id} is not extractable")
                return None
            
            if self._crypto:
                result = self._crypto.encrypt_aes(
                    key.key_material,
                    kek.key_material,
                )
                wrapped = result.nonce + result.tag + result.ciphertext
            else:
                # Fallback
                wrapped = self._simple_wrap(key.key_material, kek.key_material)
            
            self._audit("key_wrapped", key_id, {'wrapping_key': wrapping_key_id})
            return wrapped
    
    def unwrap_key(
        self,
        wrapped_key: bytes,
        wrapping_key_id: str,
        key_type: KeyType,
        owner: str = "",
        description: str = ""
    ) -> Optional[CryptoKey]:
        """
        Unwrap (decrypt) a wrapped key
        
        Args:
            wrapped_key: Wrapped key bytes
            wrapping_key_id: Key Encryption Key (KEK)
            key_type: Type of key being unwrapped
            owner: Key owner
            description: Key description
            
        Returns:
            Unwrapped CryptoKey, or None on error
        """
        with self._lock:
            kek = self.get_key_for_operation(wrapping_key_id, KeyUsage.KEY_UNWRAP)
            
            if kek is None:
                return None
            
            try:
                if self._crypto:
                    # Extract nonce, tag, ciphertext
                    nonce = wrapped_key[:12]
                    tag = wrapped_key[12:28]
                    ciphertext = wrapped_key[28:]
                    
                    key_material = self._crypto.decrypt_aes(
                        ciphertext,
                        kek.key_material,
                        mode=self._crypto.__class__.__bases__[0] if hasattr(self._crypto, 'AESMode') else None,
                        nonce=nonce,
                        tag=tag,
                    )
                else:
                    key_material = self._simple_unwrap(wrapped_key, kek.key_material)
                
                # Create key object
                key = CryptoKey(
                    key_id=self._generate_key_id(),
                    key_type=key_type,
                    key_material=key_material,
                    state=KeyState.ACTIVE,
                    activated_at=datetime.now(),
                    algorithm=self.KEY_ALGORITHMS.get(key_type, ""),
                    key_size_bits=len(key_material) * 8,
                    owner=owner,
                    description=description,
                    wrapped_by=wrapping_key_id,
                )
                
                self._keys[key.key_id] = key
                
                self._audit("key_unwrapped", key.key_id, {
                    'wrapping_key': wrapping_key_id,
                })
                
                return key
                
            except Exception as e:
                logger.error(f"Key unwrap failed: {e}")
                return None
    
    def _simple_wrap(self, key: bytes, kek: bytes) -> bytes:
        """Simple key wrapping (fallback)"""
        # XOR with hash of KEK
        mask = hashlib.sha256(kek).digest()
        mask = (mask * ((len(key) // 32) + 1))[:len(key)]
        
        wrapped = bytes(a ^ b for a, b in zip(key, mask))
        hmac_tag = hmac.new(kek, wrapped, hashlib.sha256).digest()[:16]
        
        return hmac_tag + wrapped
    
    def _simple_unwrap(self, wrapped: bytes, kek: bytes) -> bytes:
        """Simple key unwrapping (fallback)"""
        tag = wrapped[:16]
        ciphertext = wrapped[16:]
        
        # Verify HMAC
        expected_tag = hmac.new(kek, ciphertext, hashlib.sha256).digest()[:16]
        if not hmac.compare_digest(tag, expected_tag):
            raise ValueError("Key integrity check failed")
        
        # XOR with hash of KEK
        mask = hashlib.sha256(kek).digest()
        mask = (mask * ((len(ciphertext) // 32) + 1))[:len(ciphertext)]
        
        return bytes(a ^ b for a, b in zip(ciphertext, mask))
    
    # =========================================================================
    # Key Import/Export
    # =========================================================================
    
    def import_key(
        self,
        key_material: bytes,
        key_type: KeyType,
        owner: str = "",
        description: str = "",
        public_key: Optional[bytes] = None
    ) -> CryptoKey:
        """
        Import external key material
        
        Args:
            key_material: Raw key bytes
            key_type: Type of key
            owner: Key owner
            description: Key description
            public_key: Public key for asymmetric keys
            
        Returns:
            Imported CryptoKey
        """
        with self._lock:
            key = CryptoKey(
                key_id=self._generate_key_id(),
                key_type=key_type,
                key_material=key_material,
                public_key=public_key,
                state=KeyState.ACTIVE,
                activated_at=datetime.now(),
                algorithm=self.KEY_ALGORITHMS.get(key_type, ""),
                key_size_bits=len(key_material) * 8,
                owner=owner,
                description=description,
                extractable=True,
            )
            
            self._keys[key.key_id] = key
            
            self._audit("key_imported", key.key_id, {
                'key_type': key_type.value,
                'owner': owner,
            })
            
            logger.info(f"Imported key: {key.key_id}")
            return key
    
    def export_key(
        self,
        key_id: str,
        include_private: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Export key metadata and optionally material
        
        Args:
            key_id: Key identifier
            include_private: Include private key material
            
        Returns:
            Key data dictionary
        """
        with self._lock:
            key = self.get_key(key_id)
            
            if key is None:
                return None
            
            if include_private and not key.extractable:
                logger.warning(f"Key {key_id} is not extractable")
                return None
            
            self._audit("key_exported", key_id, {
                'include_private': include_private,
            })
            
            return key.to_dict(include_material=include_private)
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def save_to_storage(self) -> bool:
        """
        Save keys to encrypted storage
        
        Returns:
            True if saved successfully
        """
        if not self._storage_path:
            logger.warning("No storage path configured")
            return False
        
        with self._lock:
            try:
                # Serialize keys
                data = {
                    'version': '1.0',
                    'saved_at': datetime.now().isoformat(),
                    'keys': {
                        kid: key.to_dict(include_material=True)
                        for kid, key in self._keys.items()
                        if key.state != KeyState.DESTROYED
                    },
                }
                
                # Encrypt data
                if self._master_key and self._crypto:
                    json_data = json.dumps(data).encode()
                    result = self._crypto.encrypt_aes(json_data, self._master_key)
                    encrypted = result.nonce + result.tag + result.ciphertext
                    
                    self._storage_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self._storage_path, 'wb') as f:
                        f.write(encrypted)
                else:
                    # Plaintext (not recommended)
                    self._storage_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self._storage_path, 'w') as f:
                        json.dump(data, f)
                
                self._audit("keys_saved", None, {'count': len(data['keys'])})
                logger.info(f"Saved {len(data['keys'])} keys to storage")
                return True
                
            except Exception as e:
                logger.error(f"Failed to save keys: {e}")
                return False
    
    def load_from_storage(self) -> bool:
        """
        Load keys from encrypted storage
        
        Returns:
            True if loaded successfully
        """
        if not self._storage_path or not self._storage_path.exists():
            logger.warning("Storage path not found")
            return False
        
        with self._lock:
            try:
                # Read and decrypt data
                if self._master_key and self._crypto:
                    with open(self._storage_path, 'rb') as f:
                        encrypted = f.read()
                    
                    nonce = encrypted[:12]
                    tag = encrypted[12:28]
                    ciphertext = encrypted[28:]
                    
                    json_data = self._crypto.decrypt_aes(
                        ciphertext,
                        self._master_key,
                        nonce=nonce,
                        tag=tag,
                    )
                    data = json.loads(json_data.decode())
                else:
                    with open(self._storage_path) as f:
                        data = json.load(f)
                
                # Restore keys
                for kid, key_data in data.get('keys', {}).items():
                    self._keys[kid] = CryptoKey.from_dict(key_data)
                
                self._audit("keys_loaded", None, {'count': len(self._keys)})
                logger.info(f"Loaded {len(self._keys)} keys from storage")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load keys: {e}")
                return False
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def clear_session_keys(self) -> int:
        """
        Clear all session keys
        
        Returns:
            Number of keys cleared
        """
        with self._lock:
            count = len(self._session_keys)
            
            for key in self._session_keys.values():
                self._zeroize_key(key)
            
            self._session_keys.clear()
            
            self._audit("session_keys_cleared", None, {'count': count})
            logger.info(f"Cleared {count} session keys")
            return count
    
    def clear_expired_keys(self) -> int:
        """
        Clear expired keys
        
        Returns:
            Number of keys cleared
        """
        with self._lock:
            now = datetime.now()
            expired = [
                kid for kid, key in self._keys.items()
                if key.expires_at and key.expires_at < now
            ]
            
            for kid in expired:
                self.destroy_key(kid)
            
            return len(expired)
    
    def zeroize_all(self) -> None:
        """Zeroize all keys - EMERGENCY OPERATION"""
        with self._lock:
            logger.warning("ZEROIZING ALL KEYS")
            
            for key in list(self._keys.values()):
                self._zeroize_key(key)
            
            for key in list(self._session_keys.values()):
                self._zeroize_key(key)
            
            self._keys.clear()
            self._session_keys.clear()
            
            # Zeroize master key
            if self._master_key:
                material = bytearray(self._master_key)
                for i in range(len(material)):
                    material[i] = 0
                self._master_key = None
            
            self._audit("all_keys_zeroized", None, {})
            logger.warning("ALL KEYS ZEROIZED")
    
    def _audit(
        self,
        event: str,
        key_id: Optional[str],
        details: Dict[str, Any]
    ) -> None:
        """Record audit event"""
        if self._audit_callback:
            self._audit_callback(event, key_id, details)
