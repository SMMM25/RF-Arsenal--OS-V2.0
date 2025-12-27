#!/usr/bin/env python3
"""
RF Arsenal OS - CAN Bus Controller

Professional CAN bus security testing module supporting:
- CAN frame capture and injection
- ECU fuzzing and discovery
- Protocol reverse engineering
- Real-time traffic analysis
- Attack detection and evasion

Supports hardware:
- CANable (slcan)
- ELM327 (OBD-II)
- SocketCAN (Linux native)
- Peak CAN
- Kvaser

Author: RF Arsenal Team
License: For authorized security testing only
"""

import struct
import time
import threading
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any, Tuple
from datetime import datetime
from collections import defaultdict
import queue

logger = logging.getLogger(__name__)


class CANSpeed(Enum):
    """Standard CAN bus speeds"""
    CAN_10KBPS = 10000
    CAN_20KBPS = 20000
    CAN_50KBPS = 50000
    CAN_100KBPS = 100000
    CAN_125KBPS = 125000
    CAN_250KBPS = 250000
    CAN_500KBPS = 500000
    CAN_800KBPS = 800000
    CAN_1MBPS = 1000000


class CANProtocol(Enum):
    """CAN protocol variants"""
    CAN_2_0A = "can_2.0a"  # 11-bit identifier
    CAN_2_0B = "can_2.0b"  # 29-bit identifier (extended)
    CAN_FD = "can_fd"       # CAN Flexible Data-rate
    ISO_TP = "iso_tp"       # ISO 15765-2 Transport Protocol
    J1939 = "j1939"         # SAE J1939 (heavy vehicles)
    OBD_II = "obd_ii"       # On-Board Diagnostics


class CANInterface(Enum):
    """Supported CAN interfaces"""
    SOCKETCAN = "socketcan"
    SLCAN = "slcan"
    ELM327 = "elm327"
    PEAK = "peak"
    KVASER = "kvaser"
    VIRTUAL = "virtual"


class CANError(Exception):
    """CAN bus error"""
    pass


@dataclass
class CANFilter:
    """CAN message filter"""
    can_id: int
    can_mask: int = 0x7FF
    extended: bool = False
    
    def matches(self, frame_id: int) -> bool:
        """Check if frame ID matches filter"""
        return (frame_id & self.can_mask) == (self.can_id & self.can_mask)


@dataclass
class CANFrame:
    """CAN frame representation"""
    arbitration_id: int
    data: bytes
    timestamp: float = field(default_factory=time.time)
    is_extended: bool = False
    is_remote: bool = False
    is_error: bool = False
    is_fd: bool = False
    bitrate_switch: bool = False
    error_state_indicator: bool = False
    dlc: int = 0
    
    def __post_init__(self):
        if self.dlc == 0:
            self.dlc = len(self.data)
    
    def to_bytes(self) -> bytes:
        """Serialize frame to bytes"""
        flags = 0
        if self.is_extended:
            flags |= 0x01
        if self.is_remote:
            flags |= 0x02
        if self.is_error:
            flags |= 0x04
        if self.is_fd:
            flags |= 0x08
        
        return struct.pack(
            '<IBB',
            self.arbitration_id,
            flags,
            self.dlc
        ) + self.data[:self.dlc].ljust(8, b'\x00')
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'CANFrame':
        """Deserialize frame from bytes"""
        arb_id, flags, dlc = struct.unpack('<IBB', data[:6])
        frame_data = data[6:6+dlc]
        
        return cls(
            arbitration_id=arb_id,
            data=frame_data,
            is_extended=bool(flags & 0x01),
            is_remote=bool(flags & 0x02),
            is_error=bool(flags & 0x04),
            is_fd=bool(flags & 0x08),
            dlc=dlc
        )
    
    def __str__(self) -> str:
        hex_data = ' '.join(f'{b:02X}' for b in self.data)
        ext = 'X' if self.is_extended else ' '
        return f"[{self.arbitration_id:08X}]{ext} ({self.dlc}) {hex_data}"


@dataclass
class CANStatistics:
    """CAN bus statistics"""
    frames_received: int = 0
    frames_transmitted: int = 0
    errors: int = 0
    bus_load: float = 0.0
    unique_ids: set = field(default_factory=set)
    start_time: float = field(default_factory=time.time)
    
    @property
    def duration(self) -> float:
        return time.time() - self.start_time
    
    @property
    def rx_rate(self) -> float:
        if self.duration > 0:
            return self.frames_received / self.duration
        return 0.0


class CANBusController:
    """
    Professional CAN Bus Controller
    
    Features:
    - Multi-interface support (SocketCAN, SLCAN, ELM327)
    - Real-time frame capture and injection
    - ECU discovery and fuzzing
    - Protocol analysis (UDS, OBD-II, J1939)
    - Attack detection evasion
    - Stealth mode operation
    """
    
    def __init__(
        self,
        interface: CANInterface = CANInterface.SOCKETCAN,
        channel: str = "can0",
        bitrate: CANSpeed = CANSpeed.CAN_500KBPS,
        stealth_mode: bool = True
    ):
        self.interface = interface
        self.channel = channel
        self.bitrate = bitrate
        self.stealth_mode = stealth_mode
        
        self._connected = False
        self._bus = None
        self._receive_thread: Optional[threading.Thread] = None
        self._running = False
        
        self._rx_queue: queue.Queue = queue.Queue(maxsize=10000)
        self._tx_queue: queue.Queue = queue.Queue(maxsize=1000)
        
        self._filters: List[CANFilter] = []
        self._callbacks: List[Callable[[CANFrame], None]] = []
        self._statistics = CANStatistics()
        
        self._frame_history: List[CANFrame] = []
        self._max_history = 100000
        
        self._ecu_map: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
        logger.info(f"CAN Controller initialized: {interface.value}:{channel} @ {bitrate.value}")
    
    def connect(self) -> bool:
        """Connect to CAN interface"""
        try:
            if self.interface == CANInterface.SOCKETCAN:
                return self._connect_socketcan()
            elif self.interface == CANInterface.SLCAN:
                return self._connect_slcan()
            elif self.interface == CANInterface.ELM327:
                return self._connect_elm327()
            elif self.interface == CANInterface.VIRTUAL:
                return self._connect_virtual()
            else:
                logger.error(f"Unsupported interface: {self.interface}")
                return False
                
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def _connect_socketcan(self) -> bool:
        """Connect via SocketCAN (Linux)"""
        try:
            import socket
            
            self._bus = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
            self._bus.bind((self.channel,))
            self._connected = True
            self._start_receive_thread()
            logger.info(f"Connected to SocketCAN: {self.channel}")
            return True
            
        except ImportError:
            logger.warning("SocketCAN not available, using virtual mode")
            return self._connect_virtual()
        except OSError as e:
            logger.error(f"SocketCAN error: {e}")
            return self._connect_virtual()
    
    def _connect_slcan(self) -> bool:
        """Connect via SLCAN (serial line CAN)"""
        try:
            import serial
            
            self._bus = serial.Serial(self.channel, 115200, timeout=0.1)
            
            # Initialize SLCAN
            self._bus.write(b'\r')
            time.sleep(0.1)
            self._bus.write(f'S{self._speed_to_slcan()}\r'.encode())
            time.sleep(0.1)
            self._bus.write(b'O\r')  # Open CAN channel
            
            self._connected = True
            self._start_receive_thread()
            logger.info(f"Connected to SLCAN: {self.channel}")
            return True
            
        except Exception as e:
            logger.error(f"SLCAN error: {e}")
            return self._connect_virtual()
    
    def _connect_elm327(self) -> bool:
        """Connect via ELM327 OBD-II adapter"""
        try:
            import serial
            
            self._bus = serial.Serial(self.channel, 38400, timeout=1)
            
            # Initialize ELM327
            self._elm327_command('ATZ')  # Reset
            time.sleep(1)
            self._elm327_command('ATE0')  # Echo off
            self._elm327_command('ATL0')  # Linefeeds off
            self._elm327_command('ATS0')  # Spaces off
            self._elm327_command('ATH1')  # Headers on
            self._elm327_command('ATSP0')  # Auto protocol
            
            self._connected = True
            self._start_receive_thread()
            logger.info(f"Connected to ELM327: {self.channel}")
            return True
            
        except Exception as e:
            logger.error(f"ELM327 error: {e}")
            return self._connect_virtual()
    
    def _connect_virtual(self) -> bool:
        """Connect to virtual CAN interface (for testing)"""
        self._connected = True
        self._bus = None
        logger.info("Connected to virtual CAN interface")
        return True
    
    def _elm327_command(self, cmd: str) -> str:
        """Send ELM327 AT command"""
        if self._bus:
            self._bus.write(f'{cmd}\r'.encode())
            time.sleep(0.1)
            return self._bus.read(1000).decode('utf-8', errors='ignore')
        return ""
    
    def _speed_to_slcan(self) -> int:
        """Convert CANSpeed to SLCAN speed code"""
        speed_map = {
            CANSpeed.CAN_10KBPS: 0,
            CANSpeed.CAN_20KBPS: 1,
            CANSpeed.CAN_50KBPS: 2,
            CANSpeed.CAN_100KBPS: 3,
            CANSpeed.CAN_125KBPS: 4,
            CANSpeed.CAN_250KBPS: 5,
            CANSpeed.CAN_500KBPS: 6,
            CANSpeed.CAN_800KBPS: 7,
            CANSpeed.CAN_1MBPS: 8,
        }
        return speed_map.get(self.bitrate, 6)
    
    def disconnect(self) -> bool:
        """Disconnect from CAN interface"""
        self._running = False
        
        if self._receive_thread and self._receive_thread.is_alive():
            self._receive_thread.join(timeout=2.0)
        
        if self._bus:
            try:
                if self.interface == CANInterface.SLCAN:
                    self._bus.write(b'C\r')  # Close CAN channel
                self._bus.close()
            except Exception:
                pass
        
        self._connected = False
        self._bus = None
        logger.info("Disconnected from CAN interface")
        return True
    
    def _start_receive_thread(self):
        """Start frame receive thread"""
        self._running = True
        self._receive_thread = threading.Thread(
            target=self._receive_loop,
            daemon=True
        )
        self._receive_thread.start()
    
    def _receive_loop(self):
        """Background frame receive loop"""
        while self._running:
            try:
                frame = self._receive_frame()
                if frame:
                    self._process_received_frame(frame)
            except Exception as e:
                if self._running:
                    logger.debug(f"Receive error: {e}")
            time.sleep(0.001)
    
    def _receive_frame(self) -> Optional[CANFrame]:
        """Receive a single CAN frame"""
        if not self._connected or not self._bus:
            return None
        
        try:
            if self.interface == CANInterface.SOCKETCAN:
                data = self._bus.recv(16)
                if len(data) >= 16:
                    arb_id, dlc = struct.unpack('<IB', data[:5])
                    frame_data = data[8:8+dlc]
                    return CANFrame(
                        arbitration_id=arb_id & 0x1FFFFFFF,
                        data=frame_data,
                        is_extended=(arb_id & 0x80000000) != 0,
                        dlc=dlc
                    )
            
            elif self.interface == CANInterface.SLCAN:
                line = self._bus.readline().decode('utf-8', errors='ignore').strip()
                if line and line[0] in 'tTrR':
                    return self._parse_slcan_frame(line)
            
        except Exception:
            pass
        
        return None
    
    def _parse_slcan_frame(self, line: str) -> Optional[CANFrame]:
        """Parse SLCAN format frame"""
        try:
            is_extended = line[0] in 'TR'
            is_remote = line[0] in 'rR'
            
            if is_extended:
                arb_id = int(line[1:9], 16)
                dlc = int(line[9])
                data_str = line[10:]
            else:
                arb_id = int(line[1:4], 16)
                dlc = int(line[4])
                data_str = line[5:]
            
            data = bytes.fromhex(data_str) if not is_remote else b''
            
            return CANFrame(
                arbitration_id=arb_id,
                data=data,
                is_extended=is_extended,
                is_remote=is_remote,
                dlc=dlc
            )
        except Exception:
            return None
    
    def _process_received_frame(self, frame: CANFrame):
        """Process a received frame"""
        # Apply filters
        if self._filters:
            if not any(f.matches(frame.arbitration_id) for f in self._filters):
                return
        
        # Update statistics
        with self._lock:
            self._statistics.frames_received += 1
            self._statistics.unique_ids.add(frame.arbitration_id)
            
            # Store in history
            if len(self._frame_history) < self._max_history:
                self._frame_history.append(frame)
        
        # Queue for processing
        try:
            self._rx_queue.put_nowait(frame)
        except queue.Full:
            pass
        
        # Call registered callbacks
        for callback in self._callbacks:
            try:
                callback(frame)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def send(self, frame: CANFrame) -> bool:
        """Send a CAN frame"""
        if not self._connected:
            logger.error("Not connected")
            return False
        
        try:
            if self.stealth_mode:
                self._apply_stealth_timing()
            
            if self.interface == CANInterface.SOCKETCAN:
                return self._send_socketcan(frame)
            elif self.interface == CANInterface.SLCAN:
                return self._send_slcan(frame)
            elif self.interface == CANInterface.ELM327:
                return self._send_elm327(frame)
            elif self.interface == CANInterface.VIRTUAL:
                logger.debug(f"Virtual TX: {frame}")
                self._statistics.frames_transmitted += 1
                return True
            
        except Exception as e:
            logger.error(f"Send error: {e}")
            self._statistics.errors += 1
            return False
        
        return False
    
    def _send_socketcan(self, frame: CANFrame) -> bool:
        """Send via SocketCAN"""
        arb_id = frame.arbitration_id
        if frame.is_extended:
            arb_id |= 0x80000000
        
        data = struct.pack('<IBxxx', arb_id, frame.dlc) + frame.data.ljust(8, b'\x00')
        self._bus.send(data)
        self._statistics.frames_transmitted += 1
        return True
    
    def _send_slcan(self, frame: CANFrame) -> bool:
        """Send via SLCAN"""
        if frame.is_extended:
            cmd = f'T{frame.arbitration_id:08X}{frame.dlc}'
        else:
            cmd = f't{frame.arbitration_id:03X}{frame.dlc}'
        
        cmd += frame.data.hex().upper()
        self._bus.write(f'{cmd}\r'.encode())
        self._statistics.frames_transmitted += 1
        return True
    
    def _send_elm327(self, frame: CANFrame) -> bool:
        """Send via ELM327"""
        # ELM327 uses different format
        header = f'ATSH{frame.arbitration_id:03X}'
        self._elm327_command(header)
        
        data_hex = frame.data.hex().upper()
        response = self._elm327_command(data_hex)
        self._statistics.frames_transmitted += 1
        return 'OK' in response or len(response) > 0
    
    def _apply_stealth_timing(self):
        """Apply randomized timing for stealth"""
        import random
        delay = random.uniform(0.001, 0.010)
        time.sleep(delay)
    
    def add_filter(self, filter: CANFilter):
        """Add a receive filter"""
        self._filters.append(filter)
        logger.debug(f"Added filter: ID={filter.can_id:03X} Mask={filter.can_mask:03X}")
    
    def clear_filters(self):
        """Clear all filters"""
        self._filters.clear()
    
    def register_callback(self, callback: Callable[[CANFrame], None]):
        """Register frame receive callback"""
        self._callbacks.append(callback)
    
    def receive(self, timeout: float = 1.0) -> Optional[CANFrame]:
        """Receive a frame from queue"""
        try:
            return self._rx_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_statistics(self) -> CANStatistics:
        """Get current statistics"""
        return self._statistics
    
    def get_frame_history(self) -> List[CANFrame]:
        """Get captured frame history"""
        with self._lock:
            return list(self._frame_history)
    
    def clear_history(self):
        """Clear frame history"""
        with self._lock:
            self._frame_history.clear()
    
    # =========================================================================
    # ECU Discovery and Analysis
    # =========================================================================
    
    def discover_ecus(self, timeout: float = 5.0) -> Dict[int, Dict[str, Any]]:
        """
        Discover ECUs on the bus
        
        Sends diagnostic requests and analyzes responses
        """
        logger.info("Starting ECU discovery...")
        self._ecu_map.clear()
        
        # Standard OBD-II broadcast
        obd_request = CANFrame(
            arbitration_id=0x7DF,
            data=bytes([0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        )
        
        # UDS tester present
        uds_request = CANFrame(
            arbitration_id=0x7DF,
            data=bytes([0x02, 0x3E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        )
        
        discovered = {}
        start_time = time.time()
        
        # Listen for existing traffic first
        while time.time() - start_time < timeout / 2:
            frame = self.receive(timeout=0.1)
            if frame:
                if frame.arbitration_id not in discovered:
                    discovered[frame.arbitration_id] = {
                        'first_seen': time.time(),
                        'frame_count': 1,
                        'is_response': False
                    }
                else:
                    discovered[frame.arbitration_id]['frame_count'] += 1
        
        # Send discovery requests
        self.send(obd_request)
        time.sleep(0.1)
        self.send(uds_request)
        
        # Collect responses
        while time.time() - start_time < timeout:
            frame = self.receive(timeout=0.1)
            if frame:
                # Check for diagnostic responses (0x7E8-0x7EF)
                if 0x7E8 <= frame.arbitration_id <= 0x7EF:
                    ecu_id = frame.arbitration_id - 0x7E8
                    discovered[frame.arbitration_id] = {
                        'ecu_id': ecu_id,
                        'response_data': frame.data.hex(),
                        'is_obd': True,
                        'is_response': True
                    }
        
        self._ecu_map = discovered
        logger.info(f"Discovered {len(discovered)} CAN IDs")
        return discovered
    
    def fuzz_ecu(
        self,
        target_id: int,
        data_pattern: bytes = None,
        iterations: int = 1000,
        delay: float = 0.01
    ) -> List[Tuple[bytes, Optional[CANFrame]]]:
        """
        Fuzz an ECU with random or patterned data
        
        Args:
            target_id: Target arbitration ID
            data_pattern: Base pattern (None for random)
            iterations: Number of iterations
            delay: Delay between frames
            
        Returns:
            List of (sent_data, response) tuples
        """
        import random
        
        logger.warning(f"Starting ECU fuzz on ID 0x{target_id:03X}")
        results = []
        response_id = target_id + 8  # Standard response offset
        
        for i in range(iterations):
            if data_pattern:
                # Mutate pattern
                data = bytearray(data_pattern)
                pos = random.randint(0, len(data) - 1)
                data[pos] = random.randint(0, 255)
            else:
                # Random data
                dlc = random.randint(1, 8)
                data = bytes(random.randint(0, 255) for _ in range(dlc))
            
            frame = CANFrame(arbitration_id=target_id, data=bytes(data))
            self.send(frame)
            
            # Check for response
            time.sleep(delay)
            response = None
            while True:
                rx = self.receive(timeout=0.01)
                if not rx:
                    break
                if rx.arbitration_id == response_id:
                    response = rx
                    break
            
            results.append((bytes(data), response))
            
            if i % 100 == 0:
                logger.debug(f"Fuzz progress: {i}/{iterations}")
        
        logger.info(f"Fuzz complete. {sum(1 for _, r in results if r)} responses received")
        return results
    
    def replay_capture(
        self,
        frames: List[CANFrame],
        speed_multiplier: float = 1.0
    ) -> int:
        """
        Replay captured frames
        
        Args:
            frames: List of frames to replay
            speed_multiplier: Playback speed (1.0 = original timing)
            
        Returns:
            Number of frames sent
        """
        if not frames:
            return 0
        
        logger.info(f"Replaying {len(frames)} frames at {speed_multiplier}x speed")
        
        sent = 0
        base_time = frames[0].timestamp
        start_time = time.time()
        
        for frame in frames:
            # Calculate timing
            target_time = (frame.timestamp - base_time) / speed_multiplier
            elapsed = time.time() - start_time
            
            if target_time > elapsed:
                time.sleep(target_time - elapsed)
            
            if self.send(frame):
                sent += 1
        
        logger.info(f"Replay complete. {sent}/{len(frames)} frames sent")
        return sent
    
    def monitor(
        self,
        duration: float = 10.0,
        callback: Callable[[CANFrame], None] = None
    ) -> List[CANFrame]:
        """
        Monitor bus traffic for a duration
        
        Args:
            duration: Monitoring duration in seconds
            callback: Optional callback for each frame
            
        Returns:
            List of captured frames
        """
        logger.info(f"Monitoring CAN bus for {duration}s...")
        captured = []
        start = time.time()
        
        while time.time() - start < duration:
            frame = self.receive(timeout=0.1)
            if frame:
                captured.append(frame)
                if callback:
                    callback(frame)
        
        logger.info(f"Captured {len(captured)} frames")
        return captured
    
    @property
    def is_connected(self) -> bool:
        return self._connected
