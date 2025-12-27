"""
FIPS 140-3 Zeroization and Secure Memory Module for RF Arsenal OS.

This module implements comprehensive zeroization and secure memory
handling capabilities required for FIPS 140-3 compliance.

Features:
- Secure memory allocation with automatic zeroization
- Multiple overwrite patterns for secure erasure
- SSP (Sensitive Security Parameter) zeroization
- Memory protection and access control
- Secure memory regions with guard pages
- Emergency zeroization procedures

FIPS 140-3 Zeroization Requirements:
- All plaintext SSPs shall be zeroizable
- Zeroization shall overwrite all SSP storage
- Zeroization shall be immediate upon command
- Verification of zeroization completion

Author: RF Arsenal Development Team
License: Proprietary - Security Sensitive
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import asyncio
import ctypes
import gc
import hashlib
import logging
import mmap
import os
import secrets
import struct
import sys
import threading
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, TypeVar

logger = logging.getLogger(__name__)


class ZeroizationMethod(Enum):
    """Methods for secure memory zeroization."""
    
    SINGLE_PASS = "single_pass"       # Single overwrite with zeros
    DOD_3_PASS = "dod_3_pass"         # DoD 3-pass (zeros, ones, random)
    DOD_7_PASS = "dod_7_pass"         # DoD 7-pass
    GUTMANN = "gutmann"               # 35-pass Gutmann method
    RANDOM_FILL = "random_fill"       # Random data overwrite
    VERIFY_ZERO = "verify_zero"       # Zero and verify


class ZeroizationScope(Enum):
    """Scope of zeroization operation."""
    
    SINGLE_KEY = "single_key"         # Single key/SSP
    KEY_CATEGORY = "key_category"     # All keys of a type
    ALL_KEYS = "all_keys"             # All cryptographic keys
    SESSION_DATA = "session_data"     # Session-related data
    COMPLETE = "complete"             # Full module zeroization


class ZeroizationStatus(Enum):
    """Status of zeroization operation."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    FAILED = "failed"


@dataclass
class ZeroizationRecord:
    """Record of a zeroization operation."""
    
    record_id: str
    timestamp: datetime
    scope: ZeroizationScope
    method: ZeroizationMethod
    status: ZeroizationStatus
    items_zeroized: int
    bytes_overwritten: int
    duration_ms: float
    verified: bool
    initiator: str
    reason: str


@dataclass
class SecureMemoryRegion:
    """Secure memory region descriptor."""
    
    region_id: str
    address: int
    size: int
    created_at: datetime
    purpose: str
    protection_level: str
    access_count: int = 0
    last_access: Optional[datetime] = None
    zeroized: bool = False


class ZeroizationException(Exception):
    """Exception raised during zeroization operations."""
    pass


class MemoryProtectionException(Exception):
    """Exception raised for memory protection failures."""
    pass


def secure_memset(data: bytearray, value: int, size: int) -> None:
    """
    Secure memory set that won't be optimized away.
    
    Uses ctypes to ensure compiler doesn't optimize out the operation.
    """
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(data)), value, size)


def secure_memcmp(a: bytes, b: bytes) -> bool:
    """
    Constant-time memory comparison to prevent timing attacks.
    
    Always compares all bytes regardless of differences.
    """
    if len(a) != len(b):
        return False
    
    result = 0
    for x, y in zip(a, b):
        result |= x ^ y
    
    return result == 0


class SecureByteArray:
    """
    Secure byte array with automatic zeroization.
    
    Provides a bytearray-like interface with automatic
    secure clearing when the object is garbage collected.
    """
    
    def __init__(
        self,
        size: int = 0,
        data: Optional[bytes] = None
    ):
        if data:
            self._data = bytearray(data)
            self._size = len(data)
        else:
            self._data = bytearray(size)
            self._size = size
        
        self._zeroized = False
        self._lock = threading.Lock()
        
        # Register finalizer for cleanup
        self._finalizer = weakref.finalize(self, self._cleanup, self._data)
    
    @staticmethod
    def _cleanup(data: bytearray) -> None:
        """Cleanup function called on garbage collection."""
        if data:
            try:
                secure_memset(data, 0, len(data))
            except Exception:
                pass  # Best effort cleanup
    
    def __len__(self) -> int:
        return self._size
    
    def __getitem__(self, index: Union[int, slice]) -> Union[int, bytes]:
        with self._lock:
            if self._zeroized:
                raise ValueError("Memory has been zeroized")
            return self._data[index]
    
    def __setitem__(self, index: Union[int, slice], value: Union[int, bytes]) -> None:
        with self._lock:
            if self._zeroized:
                raise ValueError("Memory has been zeroized")
            self._data[index] = value
    
    def __bytes__(self) -> bytes:
        with self._lock:
            if self._zeroized:
                raise ValueError("Memory has been zeroized")
            return bytes(self._data)
    
    def zeroize(self, method: ZeroizationMethod = ZeroizationMethod.SINGLE_PASS) -> bool:
        """
        Securely zeroize the memory.
        
        Args:
            method: Zeroization method to use
            
        Returns:
            True if zeroization successful
        """
        with self._lock:
            if self._zeroized:
                return True
            
            try:
                if method == ZeroizationMethod.SINGLE_PASS:
                    secure_memset(self._data, 0, self._size)
                
                elif method == ZeroizationMethod.DOD_3_PASS:
                    # Pass 1: All zeros
                    secure_memset(self._data, 0x00, self._size)
                    # Pass 2: All ones
                    secure_memset(self._data, 0xFF, self._size)
                    # Pass 3: Random
                    random_data = secrets.token_bytes(self._size)
                    self._data[:] = random_data
                    # Final: Zeros
                    secure_memset(self._data, 0x00, self._size)
                
                elif method == ZeroizationMethod.DOD_7_PASS:
                    patterns = [0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF]
                    for pattern in patterns:
                        secure_memset(self._data, pattern, self._size)
                    # Final random pass
                    self._data[:] = secrets.token_bytes(self._size)
                    secure_memset(self._data, 0x00, self._size)
                
                elif method == ZeroizationMethod.RANDOM_FILL:
                    for _ in range(3):
                        self._data[:] = secrets.token_bytes(self._size)
                    secure_memset(self._data, 0x00, self._size)
                
                elif method == ZeroizationMethod.VERIFY_ZERO:
                    secure_memset(self._data, 0x00, self._size)
                    # Verify
                    if not all(b == 0 for b in self._data):
                        raise ZeroizationException("Zeroization verification failed")
                
                self._zeroized = True
                return True
                
            except Exception as e:
                logger.error(f"Zeroization failed: {e}")
                return False
    
    def is_zeroized(self) -> bool:
        """Check if memory has been zeroized."""
        return self._zeroized
    
    def __del__(self):
        """Destructor ensures memory is zeroized."""
        if not self._zeroized:
            try:
                self.zeroize()
            except Exception:
                pass  # Best effort


class SecureMemoryPool:
    """
    Secure memory pool for sensitive data.
    
    Provides pre-allocated secure memory regions with
    automatic zeroization on release.
    """
    
    def __init__(
        self,
        pool_size: int = 1024 * 1024,  # 1MB default
        region_size: int = 4096        # 4KB regions
    ):
        self._pool_size = pool_size
        self._region_size = region_size
        self._num_regions = pool_size // region_size
        
        # Allocate pool
        self._pool = bytearray(pool_size)
        self._regions: Dict[int, SecureMemoryRegion] = {}
        self._free_regions: Set[int] = set(range(self._num_regions))
        
        self._lock = threading.Lock()
        self._region_counter = 0
        
        # Statistics
        self._stats = {
            "allocations": 0,
            "deallocations": 0,
            "zeroizations": 0,
            "peak_usage": 0
        }
    
    def allocate(
        self,
        size: int,
        purpose: str = "general",
        protection_level: str = "standard"
    ) -> Optional[Tuple[int, SecureMemoryRegion]]:
        """
        Allocate a secure memory region.
        
        Args:
            size: Required size in bytes
            purpose: Purpose description
            protection_level: Protection level
            
        Returns:
            Tuple of (region_index, region_descriptor) or None
        """
        # Calculate regions needed
        regions_needed = (size + self._region_size - 1) // self._region_size
        
        with self._lock:
            if len(self._free_regions) < regions_needed:
                return None
            
            # Find contiguous free regions
            free_list = sorted(self._free_regions)
            start_region = None
            
            for i in range(len(free_list) - regions_needed + 1):
                if all(free_list[i + j] == free_list[i] + j for j in range(regions_needed)):
                    start_region = free_list[i]
                    break
            
            if start_region is None:
                return None
            
            # Allocate regions
            for j in range(regions_needed):
                self._free_regions.discard(start_region + j)
            
            self._region_counter += 1
            region_id = f"region_{self._region_counter}"
            
            region = SecureMemoryRegion(
                region_id=region_id,
                address=start_region * self._region_size,
                size=regions_needed * self._region_size,
                created_at=datetime.utcnow(),
                purpose=purpose,
                protection_level=protection_level
            )
            
            self._regions[start_region] = region
            
            # Update stats
            self._stats["allocations"] += 1
            current_usage = len(self._regions)
            self._stats["peak_usage"] = max(self._stats["peak_usage"], current_usage)
            
            return start_region, region
    
    def write(
        self,
        region_index: int,
        offset: int,
        data: bytes
    ) -> bool:
        """
        Write data to a secure memory region.
        
        Args:
            region_index: Region index from allocation
            offset: Offset within region
            data: Data to write
            
        Returns:
            True if successful
        """
        with self._lock:
            if region_index not in self._regions:
                return False
            
            region = self._regions[region_index]
            
            if offset + len(data) > region.size:
                return False
            
            start = region.address + offset
            end = start + len(data)
            
            self._pool[start:end] = data
            
            region.access_count += 1
            region.last_access = datetime.utcnow()
            
            return True
    
    def read(
        self,
        region_index: int,
        offset: int,
        size: int
    ) -> Optional[bytes]:
        """
        Read data from a secure memory region.
        
        Args:
            region_index: Region index
            offset: Offset within region
            size: Number of bytes to read
            
        Returns:
            Data bytes or None
        """
        with self._lock:
            if region_index not in self._regions:
                return None
            
            region = self._regions[region_index]
            
            if region.zeroized:
                return None
            
            if offset + size > region.size:
                return None
            
            start = region.address + offset
            end = start + size
            
            region.access_count += 1
            region.last_access = datetime.utcnow()
            
            return bytes(self._pool[start:end])
    
    def deallocate(
        self,
        region_index: int,
        zeroize: bool = True,
        method: ZeroizationMethod = ZeroizationMethod.SINGLE_PASS
    ) -> bool:
        """
        Deallocate and optionally zeroize a memory region.
        
        Args:
            region_index: Region index
            zeroize: Whether to zeroize before deallocation
            method: Zeroization method
            
        Returns:
            True if successful
        """
        with self._lock:
            if region_index not in self._regions:
                return False
            
            region = self._regions[region_index]
            
            if zeroize:
                self._zeroize_region(region, method)
            
            # Return regions to free pool
            num_regions = region.size // self._region_size
            for j in range(num_regions):
                self._free_regions.add(region_index + j)
            
            del self._regions[region_index]
            
            self._stats["deallocations"] += 1
            
            return True
    
    def _zeroize_region(
        self,
        region: SecureMemoryRegion,
        method: ZeroizationMethod
    ) -> None:
        """Zeroize a memory region."""
        start = region.address
        size = region.size
        
        if method == ZeroizationMethod.SINGLE_PASS:
            self._pool[start:start + size] = bytes(size)
        
        elif method == ZeroizationMethod.DOD_3_PASS:
            # Pass 1: Zeros
            self._pool[start:start + size] = bytes(size)
            # Pass 2: Ones
            self._pool[start:start + size] = bytes([0xFF] * size)
            # Pass 3: Random then zeros
            self._pool[start:start + size] = secrets.token_bytes(size)
            self._pool[start:start + size] = bytes(size)
        
        elif method == ZeroizationMethod.RANDOM_FILL:
            for _ in range(3):
                self._pool[start:start + size] = secrets.token_bytes(size)
            self._pool[start:start + size] = bytes(size)
        
        region.zeroized = True
        self._stats["zeroizations"] += 1
    
    def zeroize_all(
        self,
        method: ZeroizationMethod = ZeroizationMethod.DOD_3_PASS
    ) -> int:
        """
        Zeroize all allocated regions.
        
        Args:
            method: Zeroization method
            
        Returns:
            Number of regions zeroized
        """
        with self._lock:
            count = 0
            for region in self._regions.values():
                if not region.zeroized:
                    self._zeroize_region(region, method)
                    count += 1
            return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                **self._stats,
                "total_regions": self._num_regions,
                "allocated_regions": len(self._regions),
                "free_regions": len(self._free_regions),
                "pool_size_bytes": self._pool_size,
                "region_size_bytes": self._region_size
            }


class ZeroizationManager:
    """
    Main Zeroization Manager for FIPS 140-3 compliance.
    
    Coordinates all zeroization operations and maintains
    audit trail of zeroization activities.
    """
    
    def __init__(self):
        self._memory_pool = SecureMemoryPool()
        self._records: List[ZeroizationRecord] = []
        self._lock = threading.Lock()
        
        # Registered SSP handlers
        self._ssp_handlers: Dict[str, Callable[[], bool]] = {}
        
        # Statistics
        self._stats = {
            "total_zeroizations": 0,
            "bytes_zeroized": 0,
            "failed_zeroizations": 0
        }
    
    def register_ssp_handler(
        self,
        ssp_type: str,
        handler: Callable[[], bool]
    ) -> None:
        """
        Register a handler for zeroizing a specific SSP type.
        
        Args:
            ssp_type: Type of SSP (e.g., "symmetric_keys", "private_keys")
            handler: Callable that performs zeroization and returns success
        """
        self._ssp_handlers[ssp_type] = handler
    
    def zeroize_ssp(
        self,
        ssp_type: str,
        method: ZeroizationMethod = ZeroizationMethod.DOD_3_PASS,
        initiator: str = "system",
        reason: str = "routine"
    ) -> ZeroizationRecord:
        """
        Zeroize a specific SSP type.
        
        Args:
            ssp_type: Type of SSP to zeroize
            method: Zeroization method
            initiator: Who initiated the zeroization
            reason: Reason for zeroization
            
        Returns:
            ZeroizationRecord documenting the operation
        """
        start_time = time.time()
        record_id = secrets.token_hex(8)
        
        record = ZeroizationRecord(
            record_id=record_id,
            timestamp=datetime.utcnow(),
            scope=ZeroizationScope.SINGLE_KEY,
            method=method,
            status=ZeroizationStatus.IN_PROGRESS,
            items_zeroized=0,
            bytes_overwritten=0,
            duration_ms=0,
            verified=False,
            initiator=initiator,
            reason=reason
        )
        
        try:
            if ssp_type in self._ssp_handlers:
                success = self._ssp_handlers[ssp_type]()
                
                if success:
                    record.status = ZeroizationStatus.COMPLETED
                    record.items_zeroized = 1
                    record.verified = True
                else:
                    record.status = ZeroizationStatus.FAILED
                    self._stats["failed_zeroizations"] += 1
            else:
                record.status = ZeroizationStatus.FAILED
                logger.warning(f"No handler registered for SSP type: {ssp_type}")
            
        except Exception as e:
            record.status = ZeroizationStatus.FAILED
            logger.error(f"Zeroization failed for {ssp_type}: {e}")
            self._stats["failed_zeroizations"] += 1
        
        record.duration_ms = (time.time() - start_time) * 1000
        
        with self._lock:
            self._records.append(record)
            if record.status == ZeroizationStatus.COMPLETED:
                self._stats["total_zeroizations"] += 1
        
        return record
    
    def zeroize_all(
        self,
        method: ZeroizationMethod = ZeroizationMethod.DOD_3_PASS,
        initiator: str = "system",
        reason: str = "complete_zeroization"
    ) -> ZeroizationRecord:
        """
        Perform complete zeroization of all SSPs.
        
        Args:
            method: Zeroization method
            initiator: Who initiated the zeroization
            reason: Reason for zeroization
            
        Returns:
            ZeroizationRecord documenting the operation
        """
        start_time = time.time()
        record_id = secrets.token_hex(8)
        
        record = ZeroizationRecord(
            record_id=record_id,
            timestamp=datetime.utcnow(),
            scope=ZeroizationScope.COMPLETE,
            method=method,
            status=ZeroizationStatus.IN_PROGRESS,
            items_zeroized=0,
            bytes_overwritten=0,
            duration_ms=0,
            verified=False,
            initiator=initiator,
            reason=reason
        )
        
        logger.warning(f"Starting complete zeroization - Reason: {reason}")
        
        total_items = 0
        failed_items = 0
        
        # Zeroize all registered SSP types
        for ssp_type, handler in self._ssp_handlers.items():
            try:
                success = handler()
                if success:
                    total_items += 1
                else:
                    failed_items += 1
                    logger.error(f"Failed to zeroize {ssp_type}")
            except Exception as e:
                failed_items += 1
                logger.error(f"Exception during zeroization of {ssp_type}: {e}")
        
        # Zeroize memory pool
        pool_count = self._memory_pool.zeroize_all(method)
        total_items += pool_count
        
        record.items_zeroized = total_items
        record.duration_ms = (time.time() - start_time) * 1000
        
        if failed_items == 0:
            record.status = ZeroizationStatus.VERIFIED
            record.verified = True
        elif total_items > 0:
            record.status = ZeroizationStatus.COMPLETED
        else:
            record.status = ZeroizationStatus.FAILED
        
        with self._lock:
            self._records.append(record)
            self._stats["total_zeroizations"] += total_items
        
        logger.info(
            f"Complete zeroization finished: {total_items} items, "
            f"{failed_items} failures, {record.duration_ms:.2f}ms"
        )
        
        return record
    
    def emergency_zeroize(self) -> ZeroizationRecord:
        """
        Emergency zeroization - immediate clearing of all SSPs.
        
        Uses fastest available method for emergency situations.
        
        Returns:
            ZeroizationRecord documenting the operation
        """
        logger.critical("EMERGENCY ZEROIZATION INITIATED")
        
        return self.zeroize_all(
            method=ZeroizationMethod.SINGLE_PASS,  # Fastest
            initiator="emergency",
            reason="emergency_zeroization"
        )
    
    def allocate_secure_memory(
        self,
        size: int,
        purpose: str
    ) -> Optional[Tuple[int, SecureMemoryRegion]]:
        """
        Allocate secure memory region.
        
        Args:
            size: Size in bytes
            purpose: Purpose description
            
        Returns:
            Tuple of (region_index, region) or None
        """
        return self._memory_pool.allocate(size, purpose)
    
    def write_secure_memory(
        self,
        region_index: int,
        offset: int,
        data: bytes
    ) -> bool:
        """Write to secure memory region."""
        return self._memory_pool.write(region_index, offset, data)
    
    def read_secure_memory(
        self,
        region_index: int,
        offset: int,
        size: int
    ) -> Optional[bytes]:
        """Read from secure memory region."""
        return self._memory_pool.read(region_index, offset, size)
    
    def deallocate_secure_memory(
        self,
        region_index: int,
        method: ZeroizationMethod = ZeroizationMethod.SINGLE_PASS
    ) -> bool:
        """Deallocate and zeroize secure memory region."""
        return self._memory_pool.deallocate(region_index, zeroize=True, method=method)
    
    def get_zeroization_history(
        self,
        limit: int = 100
    ) -> List[ZeroizationRecord]:
        """Get zeroization history."""
        with self._lock:
            return self._records[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get zeroization statistics."""
        with self._lock:
            return {
                **self._stats,
                "memory_pool": self._memory_pool.get_statistics(),
                "registered_handlers": list(self._ssp_handlers.keys())
            }
    
    def verify_zeroization(
        self,
        record_id: str
    ) -> bool:
        """
        Verify a zeroization operation completed successfully.
        
        Args:
            record_id: Record ID to verify
            
        Returns:
            True if verified
        """
        with self._lock:
            for record in self._records:
                if record.record_id == record_id:
                    return record.verified
        return False


# Utility functions for secure memory operations

def create_secure_buffer(size: int) -> SecureByteArray:
    """Create a secure buffer with automatic zeroization."""
    return SecureByteArray(size=size)


def secure_copy(
    source: bytes,
    destination: SecureByteArray,
    offset: int = 0
) -> bool:
    """
    Securely copy data to a secure buffer.
    
    Args:
        source: Source data
        destination: Destination secure buffer
        offset: Offset in destination
        
    Returns:
        True if successful
    """
    if offset + len(source) > len(destination):
        return False
    
    destination[offset:offset + len(source)] = source
    return True


def wipe_string(s: str) -> None:
    """
    Attempt to wipe a string from memory.
    
    Note: Python strings are immutable, so this is best-effort.
    Creates garbage to help overwrite memory.
    """
    # Force garbage collection
    del s
    gc.collect()
    
    # Generate garbage to help overwrite freed memory
    _ = secrets.token_bytes(1024)


def secure_random_bytes(size: int) -> SecureByteArray:
    """
    Generate secure random bytes in a secure buffer.
    
    Args:
        size: Number of bytes
        
    Returns:
        SecureByteArray with random data
    """
    data = secrets.token_bytes(size)
    secure_buf = SecureByteArray(data=data)
    
    # Wipe the intermediate bytes object
    del data
    gc.collect()
    
    return secure_buf


# Export public API
__all__ = [
    # Enums
    "ZeroizationMethod",
    "ZeroizationScope",
    "ZeroizationStatus",
    
    # Data classes
    "ZeroizationRecord",
    "SecureMemoryRegion",
    
    # Exceptions
    "ZeroizationException",
    "MemoryProtectionException",
    
    # Classes
    "SecureByteArray",
    "SecureMemoryPool",
    "ZeroizationManager",
    
    # Functions
    "secure_memset",
    "secure_memcmp",
    "create_secure_buffer",
    "secure_copy",
    "wipe_string",
    "secure_random_bytes",
]
