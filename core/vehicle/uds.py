#!/usr/bin/env python3
"""
RF Arsenal OS - UDS (Unified Diagnostic Services) Protocol

ISO 14229-1 compliant diagnostic protocol implementation for:
- ECU diagnostics and programming
- Security access bypass
- Firmware extraction
- Memory read/write operations
- DTC (Diagnostic Trouble Code) manipulation

Author: RF Arsenal Team
License: For authorized security testing only
"""

import struct
import time
import logging
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from datetime import datetime

from .can_bus import CANBusController, CANFrame

logger = logging.getLogger(__name__)


class UDSService(IntEnum):
    """UDS Service IDs (ISO 14229-1)"""
    # Diagnostic and Communication Management
    DIAGNOSTIC_SESSION_CONTROL = 0x10
    ECU_RESET = 0x11
    SECURITY_ACCESS = 0x27
    COMMUNICATION_CONTROL = 0x28
    TESTER_PRESENT = 0x3E
    ACCESS_TIMING_PARAMETERS = 0x83
    SECURED_DATA_TRANSMISSION = 0x84
    CONTROL_DTC_SETTINGS = 0x85
    RESPONSE_ON_EVENT = 0x86
    LINK_CONTROL = 0x87
    
    # Data Transmission
    READ_DATA_BY_ID = 0x22
    READ_MEMORY_BY_ADDRESS = 0x23
    READ_SCALING_DATA_BY_ID = 0x24
    READ_DATA_BY_PERIODIC_ID = 0x2A
    DYNAMICALLY_DEFINE_DATA_ID = 0x2C
    WRITE_DATA_BY_ID = 0x2E
    WRITE_MEMORY_BY_ADDRESS = 0x3D
    
    # Stored Data Transmission
    CLEAR_DIAGNOSTIC_INFO = 0x14
    READ_DTC_INFO = 0x19
    
    # Input/Output Control
    IO_CONTROL_BY_ID = 0x2F
    
    # Remote Activation
    ROUTINE_CONTROL = 0x31
    
    # Upload/Download
    REQUEST_DOWNLOAD = 0x34
    REQUEST_UPLOAD = 0x35
    TRANSFER_DATA = 0x36
    REQUEST_TRANSFER_EXIT = 0x37
    REQUEST_FILE_TRANSFER = 0x38


class UDSSession(IntEnum):
    """UDS Diagnostic Session Types"""
    DEFAULT = 0x01
    PROGRAMMING = 0x02
    EXTENDED_DIAGNOSTIC = 0x03
    SAFETY_SYSTEM_DIAGNOSTIC = 0x04
    # Manufacturer specific: 0x40-0x5F
    # Vehicle manufacturer specific: 0x60-0x7E


class UDSSecurityLevel(IntEnum):
    """UDS Security Access Levels"""
    LEVEL_1 = 0x01  # Request seed
    LEVEL_2 = 0x02  # Send key
    LEVEL_3 = 0x03
    LEVEL_4 = 0x04
    # Extended levels
    LEVEL_17 = 0x11
    LEVEL_18 = 0x12
    LEVEL_19 = 0x13


class UDSNegativeResponse(IntEnum):
    """UDS Negative Response Codes"""
    GENERAL_REJECT = 0x10
    SERVICE_NOT_SUPPORTED = 0x11
    SUBFUNCTION_NOT_SUPPORTED = 0x12
    INCORRECT_MESSAGE_LENGTH = 0x13
    RESPONSE_TOO_LONG = 0x14
    BUSY_REPEAT_REQUEST = 0x21
    CONDITIONS_NOT_CORRECT = 0x22
    REQUEST_SEQUENCE_ERROR = 0x24
    NO_RESPONSE_FROM_SUBNET = 0x25
    FAILURE_PREVENTS_EXECUTION = 0x26
    REQUEST_OUT_OF_RANGE = 0x31
    SECURITY_ACCESS_DENIED = 0x33
    INVALID_KEY = 0x35
    EXCEEDED_ATTEMPTS = 0x36
    REQUIRED_TIME_DELAY = 0x37
    UPLOAD_DOWNLOAD_NOT_ACCEPTED = 0x70
    TRANSFER_DATA_SUSPENDED = 0x71
    PROGRAMMING_FAILURE = 0x72
    WRONG_BLOCK_SEQUENCE = 0x73
    RESPONSE_PENDING = 0x78
    SUBFUNCTION_NOT_SUPPORTED_IN_SESSION = 0x7E
    SERVICE_NOT_SUPPORTED_IN_SESSION = 0x7F


class UDSError(Exception):
    """UDS protocol error"""
    def __init__(self, message: str, nrc: UDSNegativeResponse = None):
        super().__init__(message)
        self.nrc = nrc


@dataclass
class UDSResponse:
    """UDS response container"""
    service_id: int
    data: bytes
    is_positive: bool
    negative_response_code: Optional[UDSNegativeResponse] = None
    raw_frame: Optional[CANFrame] = None
    timestamp: float = field(default_factory=time.time)
    
    def __str__(self) -> str:
        if self.is_positive:
            return f"[+] Service 0x{self.service_id:02X}: {self.data.hex()}"
        else:
            return f"[-] Service 0x{self.service_id:02X}: NRC=0x{self.negative_response_code:02X}"


@dataclass
class DiagnosticTroubleCode:
    """DTC (Diagnostic Trouble Code)"""
    code: int
    status: int
    
    @property
    def code_string(self) -> str:
        """Convert DTC to standard format (P0123, B0456, etc.)"""
        first_char = ['P', 'C', 'B', 'U'][(self.code >> 14) & 0x03]
        second = (self.code >> 12) & 0x03
        rest = self.code & 0x0FFF
        return f"{first_char}{second}{rest:03X}"
    
    @property
    def is_active(self) -> bool:
        return bool(self.status & 0x01)
    
    @property
    def is_pending(self) -> bool:
        return bool(self.status & 0x04)
    
    @property
    def is_confirmed(self) -> bool:
        return bool(self.status & 0x08)


class UDSClient:
    """
    UDS (Unified Diagnostic Services) Client
    
    Full implementation of ISO 14229-1 for vehicle ECU diagnostics:
    - Session management
    - Security access (seed-key authentication)
    - Memory read/write
    - DTC manipulation
    - Firmware upload/download
    - Routine control
    """
    
    def __init__(
        self,
        can_controller: CANBusController,
        tx_id: int = 0x7E0,
        rx_id: int = 0x7E8,
        timeout: float = 2.0
    ):
        self.can = can_controller
        self.tx_id = tx_id
        self.rx_id = rx_id
        self.timeout = timeout
        
        self._current_session = UDSSession.DEFAULT
        self._security_unlocked = False
        self._security_level = 0
        
        self._tester_present_thread = None
        self._keep_alive = False
        
        self._response_pending_timeout = 5.0
        self._p2_timeout = 0.050  # 50ms
        self._p2_star_timeout = 5.0  # 5s for response pending
        
        logger.info(f"UDS Client initialized: TX=0x{tx_id:03X} RX=0x{rx_id:03X}")
    
    def _send_request(self, service: int, data: bytes = b'') -> bytes:
        """Send UDS request and return response data"""
        # Build ISO-TP frame (simplified single frame)
        payload = bytes([service]) + data
        
        if len(payload) <= 7:
            # Single frame
            frame_data = bytes([len(payload)]) + payload
            frame_data = frame_data.ljust(8, b'\x00')
        else:
            # Multi-frame (ISO-TP) - simplified
            # First frame
            length = len(payload)
            frame_data = bytes([0x10 | ((length >> 8) & 0x0F), length & 0xFF]) + payload[:6]
            # Would need consecutive frames for full implementation
            logger.warning("Multi-frame messages not fully implemented")
        
        frame = CANFrame(arbitration_id=self.tx_id, data=frame_data)
        
        if not self.can.send(frame):
            raise UDSError("Failed to send CAN frame")
        
        return self._receive_response(service)
    
    def _receive_response(self, expected_service: int) -> bytes:
        """Receive and parse UDS response"""
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            frame = self.can.receive(timeout=0.1)
            
            if frame and frame.arbitration_id == self.rx_id:
                # Parse ISO-TP
                pci = frame.data[0]
                
                if pci <= 0x07:
                    # Single frame
                    length = pci
                    data = frame.data[1:1+length]
                elif (pci & 0xF0) == 0x10:
                    # First frame of multi-frame
                    length = ((pci & 0x0F) << 8) | frame.data[1]
                    data = frame.data[2:]
                    # Would need to receive consecutive frames
                    logger.warning("Received first frame of multi-frame response")
                else:
                    continue
                
                if len(data) < 1:
                    continue
                
                response_sid = data[0]
                
                # Positive response
                if response_sid == expected_service + 0x40:
                    return data[1:]
                
                # Negative response
                if response_sid == 0x7F:
                    rejected_service = data[1] if len(data) > 1 else 0
                    nrc = data[2] if len(data) > 2 else 0
                    
                    # Response pending
                    if nrc == UDSNegativeResponse.RESPONSE_PENDING:
                        logger.debug("Response pending, waiting...")
                        start_time = time.time()
                        self.timeout = self._p2_star_timeout
                        continue
                    
                    raise UDSError(
                        f"Negative response: Service=0x{rejected_service:02X} NRC=0x{nrc:02X}",
                        UDSNegativeResponse(nrc) if nrc in [e.value for e in UDSNegativeResponse] else None
                    )
        
        raise UDSError("Response timeout")
    
    # =========================================================================
    # Session Management
    # =========================================================================
    
    def diagnostic_session_control(self, session: UDSSession) -> bool:
        """
        Change diagnostic session
        
        Args:
            session: Target session type
            
        Returns:
            True if session changed successfully
        """
        try:
            response = self._send_request(
                UDSService.DIAGNOSTIC_SESSION_CONTROL,
                bytes([session])
            )
            
            self._current_session = session
            logger.info(f"Session changed to: {session.name}")
            return True
            
        except UDSError as e:
            logger.error(f"Session control failed: {e}")
            return False
    
    def tester_present(self, response_required: bool = False) -> bool:
        """Send tester present to keep session alive"""
        try:
            subfunction = 0x00 if response_required else 0x80
            self._send_request(UDSService.TESTER_PRESENT, bytes([subfunction]))
            return True
        except UDSError:
            return False
    
    def start_tester_present(self, interval: float = 2.0):
        """Start background tester present"""
        import threading
        
        self._keep_alive = True
        
        def _keep_alive_loop():
            while self._keep_alive:
                self.tester_present(response_required=False)
                time.sleep(interval)
        
        self._tester_present_thread = threading.Thread(
            target=_keep_alive_loop,
            daemon=True
        )
        self._tester_present_thread.start()
        logger.info("Tester present started")
    
    def stop_tester_present(self):
        """Stop background tester present"""
        self._keep_alive = False
        if self._tester_present_thread:
            self._tester_present_thread.join(timeout=3.0)
        logger.info("Tester present stopped")
    
    # =========================================================================
    # Security Access
    # =========================================================================
    
    def security_access_request_seed(self, level: int = 0x01) -> bytes:
        """
        Request security seed
        
        Args:
            level: Security level (odd numbers)
            
        Returns:
            Seed bytes
        """
        response = self._send_request(
            UDSService.SECURITY_ACCESS,
            bytes([level])
        )
        
        seed = response[1:] if len(response) > 1 else response
        logger.info(f"Received seed: {seed.hex()}")
        return seed
    
    def security_access_send_key(self, level: int, key: bytes) -> bool:
        """
        Send security key
        
        Args:
            level: Security level (even number = level + 1)
            key: Calculated key bytes
            
        Returns:
            True if access granted
        """
        try:
            self._send_request(
                UDSService.SECURITY_ACCESS,
                bytes([level + 1]) + key
            )
            
            self._security_unlocked = True
            self._security_level = level
            logger.info(f"Security access granted at level {level}")
            return True
            
        except UDSError as e:
            logger.error(f"Security access denied: {e}")
            return False
    
    def security_access(
        self,
        level: int = 0x01,
        key_algorithm: Callable[[bytes], bytes] = None
    ) -> bool:
        """
        Perform full security access sequence
        
        Args:
            level: Security level
            key_algorithm: Function to calculate key from seed
            
        Returns:
            True if access granted
        """
        seed = self.security_access_request_seed(level)
        
        if seed == b'\x00' * len(seed):
            logger.info("Zero seed - already unlocked")
            self._security_unlocked = True
            return True
        
        if key_algorithm:
            key = key_algorithm(seed)
        else:
            # Default: XOR with 0xFF (common weak implementation)
            key = bytes(b ^ 0xFF for b in seed)
            logger.warning("Using default key algorithm (XOR 0xFF)")
        
        return self.security_access_send_key(level, key)
    
    def brute_force_security(
        self,
        level: int = 0x01,
        key_length: int = 2,
        callback: Callable[[int, int], None] = None
    ) -> Optional[bytes]:
        """
        Brute force security access
        
        WARNING: May trigger lockout!
        
        Args:
            level: Security level
            key_length: Key length in bytes
            callback: Progress callback (current, total)
            
        Returns:
            Valid key if found, None otherwise
        """
        logger.warning("Starting security brute force - may cause lockout!")
        
        total = 256 ** key_length
        
        for i in range(total):
            key = i.to_bytes(key_length, 'big')
            
            try:
                # Get fresh seed
                seed = self.security_access_request_seed(level)
                
                if self.security_access_send_key(level, key):
                    logger.info(f"Key found: {key.hex()}")
                    return key
                    
            except UDSError as e:
                if e.nrc == UDSNegativeResponse.EXCEEDED_ATTEMPTS:
                    logger.error("Lockout triggered!")
                    return None
                elif e.nrc == UDSNegativeResponse.REQUIRED_TIME_DELAY:
                    logger.warning("Time delay required, waiting...")
                    time.sleep(10)
            
            if callback and i % 100 == 0:
                callback(i, total)
        
        return None
    
    # =========================================================================
    # Data Reading
    # =========================================================================
    
    def read_data_by_id(self, data_id: int) -> bytes:
        """
        Read data by identifier
        
        Args:
            data_id: 16-bit data identifier
            
        Returns:
            Data bytes
        """
        response = self._send_request(
            UDSService.READ_DATA_BY_ID,
            struct.pack('>H', data_id)
        )
        
        # Response contains DID echo + data
        if len(response) >= 2:
            return response[2:]
        return response
    
    def read_memory_by_address(
        self,
        address: int,
        size: int,
        address_format: int = 0x14  # 1 byte size, 4 bytes address
    ) -> bytes:
        """
        Read memory by address
        
        Args:
            address: Memory address
            size: Number of bytes to read
            address_format: Address and length format
            
        Returns:
            Memory contents
        """
        addr_len = (address_format >> 4) & 0x0F
        size_len = address_format & 0x0F
        
        addr_bytes = address.to_bytes(addr_len, 'big')
        size_bytes = size.to_bytes(size_len, 'big')
        
        response = self._send_request(
            UDSService.READ_MEMORY_BY_ADDRESS,
            bytes([address_format]) + addr_bytes + size_bytes
        )
        
        return response
    
    def dump_memory(
        self,
        start_address: int,
        length: int,
        chunk_size: int = 64,
        callback: Callable[[int, int, bytes], None] = None
    ) -> bytes:
        """
        Dump memory region
        
        Args:
            start_address: Start address
            length: Total bytes to read
            chunk_size: Bytes per request
            callback: Progress callback (offset, total, data)
            
        Returns:
            Complete memory dump
        """
        logger.info(f"Dumping memory: 0x{start_address:08X} - 0x{start_address+length:08X}")
        
        data = b''
        offset = 0
        
        while offset < length:
            remaining = length - offset
            read_size = min(chunk_size, remaining)
            
            try:
                chunk = self.read_memory_by_address(
                    start_address + offset,
                    read_size
                )
                data += chunk
                
                if callback:
                    callback(offset, length, chunk)
                    
            except UDSError as e:
                logger.error(f"Read failed at 0x{start_address + offset:08X}: {e}")
                data += b'\xFF' * read_size
            
            offset += read_size
        
        logger.info(f"Memory dump complete: {len(data)} bytes")
        return data
    
    # =========================================================================
    # Data Writing
    # =========================================================================
    
    def write_data_by_id(self, data_id: int, data: bytes) -> bool:
        """
        Write data by identifier
        
        Args:
            data_id: 16-bit data identifier
            data: Data to write
            
        Returns:
            True if successful
        """
        try:
            self._send_request(
                UDSService.WRITE_DATA_BY_ID,
                struct.pack('>H', data_id) + data
            )
            logger.info(f"Written DID 0x{data_id:04X}")
            return True
        except UDSError as e:
            logger.error(f"Write failed: {e}")
            return False
    
    def write_memory_by_address(
        self,
        address: int,
        data: bytes,
        address_format: int = 0x14
    ) -> bool:
        """
        Write memory by address
        
        Args:
            address: Memory address
            data: Data to write
            address_format: Address format
            
        Returns:
            True if successful
        """
        addr_len = (address_format >> 4) & 0x0F
        size_len = address_format & 0x0F
        
        addr_bytes = address.to_bytes(addr_len, 'big')
        size_bytes = len(data).to_bytes(size_len, 'big')
        
        try:
            self._send_request(
                UDSService.WRITE_MEMORY_BY_ADDRESS,
                bytes([address_format]) + addr_bytes + size_bytes + data
            )
            logger.info(f"Written {len(data)} bytes to 0x{address:08X}")
            return True
        except UDSError as e:
            logger.error(f"Write failed: {e}")
            return False
    
    # =========================================================================
    # DTC Operations
    # =========================================================================
    
    def read_dtc_info(
        self,
        subfunction: int = 0x01
    ) -> List[DiagnosticTroubleCode]:
        """
        Read diagnostic trouble codes
        
        Args:
            subfunction: Report type (0x01 = by status mask)
            
        Returns:
            List of DTCs
        """
        response = self._send_request(
            UDSService.READ_DTC_INFO,
            bytes([subfunction, 0xFF])  # All statuses
        )
        
        dtcs = []
        
        if len(response) >= 3:
            # Parse DTC records (3 bytes each: 2 DTC + 1 status)
            i = 1  # Skip subfunction echo
            while i + 2 < len(response):
                dtc_high = response[i]
                dtc_low = response[i + 1]
                status = response[i + 2]
                
                dtc = DiagnosticTroubleCode(
                    code=(dtc_high << 8) | dtc_low,
                    status=status
                )
                dtcs.append(dtc)
                i += 3
        
        logger.info(f"Read {len(dtcs)} DTCs")
        return dtcs
    
    def clear_dtc(self, group: int = 0xFFFFFF) -> bool:
        """
        Clear diagnostic trouble codes
        
        Args:
            group: DTC group (0xFFFFFF = all)
            
        Returns:
            True if successful
        """
        try:
            self._send_request(
                UDSService.CLEAR_DIAGNOSTIC_INFO,
                struct.pack('>I', group)[1:]  # 3 bytes
            )
            logger.info("DTCs cleared")
            return True
        except UDSError as e:
            logger.error(f"Clear DTC failed: {e}")
            return False
    
    # =========================================================================
    # ECU Reset
    # =========================================================================
    
    def ecu_reset(self, reset_type: int = 0x01) -> bool:
        """
        Reset ECU
        
        Args:
            reset_type: 0x01=hard, 0x02=key off/on, 0x03=soft
            
        Returns:
            True if successful
        """
        try:
            self._send_request(
                UDSService.ECU_RESET,
                bytes([reset_type])
            )
            logger.info(f"ECU reset type {reset_type} initiated")
            return True
        except UDSError as e:
            logger.error(f"ECU reset failed: {e}")
            return False
    
    # =========================================================================
    # Routine Control
    # =========================================================================
    
    def routine_control(
        self,
        routine_id: int,
        control_type: int = 0x01,  # Start
        data: bytes = b''
    ) -> bytes:
        """
        Control a routine
        
        Args:
            routine_id: Routine identifier
            control_type: 0x01=start, 0x02=stop, 0x03=request results
            data: Optional routine data
            
        Returns:
            Routine response data
        """
        response = self._send_request(
            UDSService.ROUTINE_CONTROL,
            bytes([control_type]) + struct.pack('>H', routine_id) + data
        )
        
        return response[3:] if len(response) > 3 else b''
    
    # =========================================================================
    # Upload/Download
    # =========================================================================
    
    def request_download(
        self,
        address: int,
        size: int,
        compression: int = 0x00,
        encryption: int = 0x00
    ) -> int:
        """
        Request download (ECU <- Tester)
        
        Args:
            address: Memory address
            size: Data size
            compression: Compression method
            encryption: Encryption method
            
        Returns:
            Maximum block size
        """
        data_format = (compression << 4) | encryption
        address_format = 0x44  # 4 bytes address, 4 bytes size
        
        response = self._send_request(
            UDSService.REQUEST_DOWNLOAD,
            bytes([data_format, address_format]) +
            struct.pack('>I', address) +
            struct.pack('>I', size)
        )
        
        # Parse max block length
        length_format = response[0]
        length_size = (length_format >> 4) & 0x0F
        max_block = int.from_bytes(response[1:1+length_size], 'big')
        
        logger.info(f"Download request accepted, max block: {max_block}")
        return max_block
    
    def request_upload(
        self,
        address: int,
        size: int
    ) -> int:
        """
        Request upload (ECU -> Tester)
        
        Args:
            address: Memory address
            size: Data size
            
        Returns:
            Maximum block size
        """
        response = self._send_request(
            UDSService.REQUEST_UPLOAD,
            bytes([0x00, 0x44]) +
            struct.pack('>I', address) +
            struct.pack('>I', size)
        )
        
        length_format = response[0]
        length_size = (length_format >> 4) & 0x0F
        max_block = int.from_bytes(response[1:1+length_size], 'big')
        
        logger.info(f"Upload request accepted, max block: {max_block}")
        return max_block
    
    def transfer_data(self, block_counter: int, data: bytes = b'') -> bytes:
        """
        Transfer data block
        
        Args:
            block_counter: Block sequence counter (1-255)
            data: Data to transfer (download) or empty (upload)
            
        Returns:
            Received data (upload) or empty (download)
        """
        response = self._send_request(
            UDSService.TRANSFER_DATA,
            bytes([block_counter]) + data
        )
        
        return response[1:] if len(response) > 1 else b''
    
    def transfer_exit(self) -> bool:
        """Exit transfer mode"""
        try:
            self._send_request(UDSService.REQUEST_TRANSFER_EXIT, b'')
            logger.info("Transfer complete")
            return True
        except UDSError as e:
            logger.error(f"Transfer exit failed: {e}")
            return False
    
    def flash_firmware(
        self,
        address: int,
        firmware: bytes,
        callback: Callable[[int, int], None] = None
    ) -> bool:
        """
        Flash firmware to ECU
        
        Args:
            address: Flash address
            firmware: Firmware binary
            callback: Progress callback
            
        Returns:
            True if successful
        """
        logger.warning(f"Flashing {len(firmware)} bytes to 0x{address:08X}")
        
        try:
            # Request download
            max_block = self.request_download(address, len(firmware))
            
            # Transfer data in blocks
            offset = 0
            block_counter = 1
            
            while offset < len(firmware):
                chunk_size = min(max_block - 2, len(firmware) - offset)
                chunk = firmware[offset:offset + chunk_size]
                
                self.transfer_data(block_counter, chunk)
                
                offset += chunk_size
                block_counter = (block_counter % 255) + 1
                
                if callback:
                    callback(offset, len(firmware))
            
            # Complete transfer
            self.transfer_exit()
            
            logger.info("Firmware flash complete")
            return True
            
        except UDSError as e:
            logger.error(f"Firmware flash failed: {e}")
            return False
    
    @property
    def current_session(self) -> UDSSession:
        return self._current_session
    
    @property
    def is_security_unlocked(self) -> bool:
        return self._security_unlocked
