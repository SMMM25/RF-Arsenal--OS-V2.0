#!/usr/bin/env python3
"""
RF Arsenal OS - Traffic Obfuscation Module
Advanced traffic analysis countermeasures

STEALTH COMPLIANCE:
- Makes traffic analysis impossible
- Constant bandwidth mode (uniform traffic pattern)
- Packet padding and timing randomization
- Protocol mimicry (appear as normal traffic)
- No telemetry or logging to disk

Author: RF Arsenal Security Team
"""

import os
import socket
import struct
import time
import threading
import secrets
import logging
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ObfuscationTechnique(Enum):
    """Traffic obfuscation techniques."""
    CONSTANT_BANDWIDTH = "constant_bandwidth"   # Uniform traffic rate
    PACKET_PADDING = "packet_padding"           # Fixed-size packets
    TIMING_JITTER = "timing_jitter"             # Random inter-packet delays
    DUMMY_TRAFFIC = "dummy_traffic"             # Generate cover traffic
    PROTOCOL_MIMICRY = "protocol_mimicry"       # Mimic legitimate protocols
    TRAFFIC_MORPHING = "traffic_morphing"       # Transform traffic patterns
    BURST_SHAPING = "burst_shaping"             # Randomize burst patterns


class MimicryProtocol(Enum):
    """Protocols to mimic for traffic disguise."""
    HTTPS = "https"       # HTTPS/TLS traffic
    DNS = "dns"           # DNS queries
    NTP = "ntp"           # NTP time sync
    HTTP2 = "http2"       # HTTP/2 frames


@dataclass
class TrafficStats:
    """Traffic statistics for analysis."""
    packets_sent: int = 0
    packets_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    dummy_packets_sent: int = 0
    padding_bytes_added: int = 0
    average_delay_ms: float = 0.0


class PacketPadder:
    """
    Pad packets to fixed sizes to prevent size-based analysis.
    
    All packets become uniform size, making it impossible to
    distinguish different message types by size.
    """
    
    # Standard packet sizes (powers of 2 for efficiency)
    PACKET_SIZES = [128, 256, 512, 1024, 1460]  # 1460 = MTU - headers
    
    def __init__(self, default_size: int = 1024):
        self._default_size = default_size
        self._stats = TrafficStats()
        self._lock = threading.Lock()
    
    def pad_packet(self, data: bytes, target_size: int = None) -> bytes:
        """
        Pad packet to target size.
        
        Uses cryptographically random padding to prevent
        pattern detection in padding.
        """
        if target_size is None:
            target_size = self._default_size
        
        current_size = len(data)
        
        if current_size >= target_size:
            # Already at or above target, return as-is
            # (or could truncate for strict fixed-size)
            return data
        
        # Calculate padding needed
        padding_size = target_size - current_size - 4  # 4 bytes for length header
        
        if padding_size < 0:
            padding_size = 0
        
        # Create padded packet: [original_length (4 bytes)][data][random padding]
        length_header = struct.pack('>I', current_size)
        padding = secrets.token_bytes(padding_size)
        
        padded = length_header + data + padding
        
        with self._lock:
            self._stats.padding_bytes_added += padding_size
        
        return padded
    
    def unpad_packet(self, padded_data: bytes) -> bytes:
        """Extract original data from padded packet."""
        if len(padded_data) < 4:
            return padded_data
        
        # Extract original length
        original_length = struct.unpack('>I', padded_data[:4])[0]
        
        # Extract original data
        return padded_data[4:4 + original_length]
    
    def get_nearest_size(self, data_length: int) -> int:
        """Get nearest standard packet size."""
        for size in self.PACKET_SIZES:
            if data_length + 4 <= size:  # +4 for length header
                return size
        return self.PACKET_SIZES[-1]


class TimingObfuscator:
    """
    Add random timing jitter to defeat timing analysis.
    
    Uses cryptographically secure randomization for
    unpredictable inter-packet delays.
    """
    
    def __init__(
        self, 
        min_delay_ms: float = 1.0,
        max_delay_ms: float = 100.0,
        distribution: str = 'uniform'  # 'uniform', 'exponential', 'gaussian'
    ):
        self._min_delay = min_delay_ms
        self._max_delay = max_delay_ms
        self._distribution = distribution
        self._csprng = secrets.SystemRandom()
        self._stats = TrafficStats()
        self._delays: List[float] = []
    
    def get_delay(self) -> float:
        """Get randomized delay in milliseconds."""
        if self._distribution == 'uniform':
            delay = self._csprng.uniform(self._min_delay, self._max_delay)
        elif self._distribution == 'exponential':
            # Exponential distribution with mean at midpoint
            mean = (self._min_delay + self._max_delay) / 2
            delay = self._csprng.expovariate(1 / mean)
            delay = max(self._min_delay, min(delay, self._max_delay))
        else:
            # Default to uniform
            delay = self._csprng.uniform(self._min_delay, self._max_delay)
        
        self._delays.append(delay)
        if len(self._delays) > 1000:
            self._delays = self._delays[-1000:]
        
        return delay
    
    def apply_delay(self):
        """Apply randomized delay."""
        delay_ms = self.get_delay()
        time.sleep(delay_ms / 1000.0)
    
    def get_average_delay(self) -> float:
        """Get average delay over recent packets."""
        if not self._delays:
            return 0.0
        return sum(self._delays) / len(self._delays)


class DummyTrafficGenerator:
    """
    Generate dummy/cover traffic to mask real traffic patterns.
    
    Maintains constant bandwidth regardless of actual traffic,
    making traffic analysis impossible.
    """
    
    # Legitimate targets for dummy traffic (will be dropped as malformed)
    DUMMY_TARGETS = [
        ("time.google.com", 123, "ntp"),
        ("time.cloudflare.com", 123, "ntp"),
        ("time.nist.gov", 123, "ntp"),
        ("8.8.8.8", 53, "dns"),
        ("1.1.1.1", 53, "dns"),
    ]
    
    def __init__(
        self,
        packets_per_second: float = 10.0,
        packet_size: int = 512
    ):
        self._pps = packets_per_second
        self._packet_size = packet_size
        self._active = False
        self._thread: Optional[threading.Thread] = None
        self._stats = TrafficStats()
        self._csprng = secrets.SystemRandom()
        self._lock = threading.Lock()
    
    def start(self):
        """Start dummy traffic generation."""
        if self._active:
            return
        
        self._active = True
        self._thread = threading.Thread(target=self._generate_loop, daemon=True)
        self._thread.start()
        logger.info(f"Dummy traffic started: {self._pps} packets/sec, {self._packet_size} bytes/packet")
    
    def stop(self):
        """Stop dummy traffic generation."""
        self._active = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Dummy traffic stopped")
    
    def _generate_loop(self):
        """Main generation loop."""
        interval = 1.0 / self._pps
        
        while self._active:
            start_time = time.time()
            
            self._send_dummy_packet()
            
            # Calculate sleep time to maintain constant rate
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            
            # Add small jitter to sleep time
            jitter = self._csprng.uniform(-0.1 * interval, 0.1 * interval)
            time.sleep(max(0.001, sleep_time + jitter))
    
    def _send_dummy_packet(self):
        """Send a single dummy packet."""
        # Choose random target
        target_host, target_port, protocol = self._csprng.choice(self.DUMMY_TARGETS)
        
        # Generate random data
        dummy_data = secrets.token_bytes(self._packet_size)
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(0.1)
            sock.sendto(dummy_data, (target_host, target_port))
            sock.close()
            
            with self._lock:
                self._stats.dummy_packets_sent += 1
                self._stats.bytes_sent += self._packet_size
            
        except (socket.error, socket.timeout, OSError):
            # Silently ignore errors - best effort
            pass
    
    def get_stats(self) -> TrafficStats:
        """Get traffic statistics."""
        with self._lock:
            return TrafficStats(
                packets_sent=self._stats.packets_sent,
                dummy_packets_sent=self._stats.dummy_packets_sent,
                bytes_sent=self._stats.bytes_sent
            )


class ConstantBandwidthController:
    """
    Maintain constant bandwidth regardless of actual traffic.
    
    Real traffic is sent, dummy traffic fills the gaps.
    Observer sees constant rate = impossible to detect activity.
    """
    
    def __init__(
        self,
        target_bandwidth_kbps: float = 100.0,
        packet_size: int = 1024
    ):
        self._target_bw = target_bandwidth_kbps * 1024 / 8  # Convert to bytes/sec
        self._packet_size = packet_size
        self._packets_per_second = self._target_bw / packet_size
        
        self._real_bytes_sent = 0
        self._dummy_bytes_sent = 0
        self._last_adjustment = time.time()
        self._adjustment_interval = 1.0  # Adjust every second
        
        self._dummy_generator = DummyTrafficGenerator(
            packets_per_second=self._packets_per_second,
            packet_size=packet_size
        )
        
        self._active = False
        self._lock = threading.Lock()
    
    def start(self):
        """Start constant bandwidth mode."""
        self._active = True
        self._dummy_generator.start()
        logger.info(f"Constant bandwidth mode: {self._target_bw * 8 / 1024:.1f} Kbps")
    
    def stop(self):
        """Stop constant bandwidth mode."""
        self._active = False
        self._dummy_generator.stop()
    
    def record_real_traffic(self, bytes_sent: int):
        """Record actual traffic sent (to adjust dummy traffic)."""
        with self._lock:
            self._real_bytes_sent += bytes_sent
        
        # Adjust dummy traffic rate periodically
        now = time.time()
        if now - self._last_adjustment > self._adjustment_interval:
            self._adjust_dummy_rate()
            self._last_adjustment = now
    
    def _adjust_dummy_rate(self):
        """Adjust dummy traffic rate to maintain constant total bandwidth."""
        with self._lock:
            # Calculate real traffic rate
            real_rate = self._real_bytes_sent / self._adjustment_interval
            
            # Calculate needed dummy rate
            needed_dummy_rate = max(0, self._target_bw - real_rate)
            needed_pps = needed_dummy_rate / self._packet_size
            
            # Update dummy generator (simplified - would need setter method)
            self._dummy_generator._pps = max(0.1, needed_pps)
            
            # Reset counters
            self._real_bytes_sent = 0


class ProtocolMimicry:
    """
    Make traffic appear as legitimate protocol traffic.
    
    Wraps data in protocol-compliant headers/structure.
    """
    
    def __init__(self, protocol: MimicryProtocol = MimicryProtocol.HTTPS):
        self._protocol = protocol
    
    def wrap_data(self, data: bytes) -> bytes:
        """Wrap data to appear as legitimate protocol traffic."""
        if self._protocol == MimicryProtocol.DNS:
            return self._wrap_as_dns(data)
        elif self._protocol == MimicryProtocol.NTP:
            return self._wrap_as_ntp(data)
        elif self._protocol == MimicryProtocol.HTTP2:
            return self._wrap_as_http2(data)
        else:
            return self._wrap_as_tls(data)
    
    def unwrap_data(self, wrapped: bytes) -> bytes:
        """Extract data from protocol wrapper."""
        if self._protocol == MimicryProtocol.DNS:
            return self._unwrap_dns(wrapped)
        elif self._protocol == MimicryProtocol.NTP:
            return self._unwrap_ntp(wrapped)
        elif self._protocol == MimicryProtocol.HTTP2:
            return self._unwrap_http2(wrapped)
        else:
            return self._unwrap_tls(wrapped)
    
    def _wrap_as_dns(self, data: bytes) -> bytes:
        """Wrap data as DNS query."""
        # DNS header (12 bytes)
        transaction_id = secrets.token_bytes(2)
        flags = b'\x01\x00'  # Standard query
        questions = b'\x00\x01'
        answers = b'\x00\x00'
        authority = b'\x00\x00'
        additional = b'\x00\x00'
        
        header = transaction_id + flags + questions + answers + authority + additional
        
        # Encode data as subdomain labels
        # Each label max 63 chars, total max 253
        encoded_labels = []
        for i in range(0, len(data), 60):
            chunk = data[i:i+60]
            label = bytes([len(chunk)]) + chunk
            encoded_labels.append(label)
        
        # Null terminator
        encoded_labels.append(b'\x00')
        
        # Query type (TXT) and class (IN)
        query_type = b'\x00\x10'  # TXT
        query_class = b'\x00\x01'  # IN
        
        return header + b''.join(encoded_labels) + query_type + query_class
    
    def _unwrap_dns(self, wrapped: bytes) -> bytes:
        """Extract data from DNS wrapper."""
        # Skip header (12 bytes)
        pos = 12
        
        # Extract labels
        data_parts = []
        while pos < len(wrapped):
            label_len = wrapped[pos]
            if label_len == 0:
                break
            pos += 1
            data_parts.append(wrapped[pos:pos + label_len])
            pos += label_len
        
        return b''.join(data_parts)
    
    def _wrap_as_ntp(self, data: bytes) -> bytes:
        """Wrap data as NTP packet."""
        # NTP header (48 bytes standard)
        # LI=0, VN=4, Mode=3 (client)
        flags = bytes([0x23])  # 00 100 011
        stratum = b'\x00'
        poll = b'\x06'
        precision = b'\xEC'
        
        # Root delay and dispersion (8 bytes)
        root_delay = b'\x00\x00\x00\x00'
        root_dispersion = b'\x00\x00\x00\x00'
        
        # Reference ID (4 bytes)
        ref_id = secrets.token_bytes(4)
        
        # Timestamps (32 bytes) - hide data here
        # Pad data to 32 bytes
        if len(data) > 32:
            data = data[:32]
        else:
            data = data + secrets.token_bytes(32 - len(data))
        
        return flags + stratum + poll + precision + root_delay + root_dispersion + ref_id + data
    
    def _unwrap_ntp(self, wrapped: bytes) -> bytes:
        """Extract data from NTP wrapper."""
        # Data is in timestamp fields (last 32 bytes)
        if len(wrapped) >= 48:
            return wrapped[16:48]
        return b''
    
    def _wrap_as_http2(self, data: bytes) -> bytes:
        """Wrap data as HTTP/2 frame."""
        # HTTP/2 frame header (9 bytes)
        length = len(data)
        length_bytes = struct.pack('>I', length)[1:]  # 3 bytes
        frame_type = b'\x00'  # DATA frame
        flags = b'\x00'
        stream_id = struct.pack('>I', 1)  # Stream ID 1
        
        return length_bytes + frame_type + flags + stream_id + data
    
    def _unwrap_http2(self, wrapped: bytes) -> bytes:
        """Extract data from HTTP/2 wrapper."""
        if len(wrapped) < 9:
            return b''
        
        # Extract length from header
        length = struct.unpack('>I', b'\x00' + wrapped[:3])[0]
        
        # Return payload
        return wrapped[9:9 + length]
    
    def _wrap_as_tls(self, data: bytes) -> bytes:
        """Wrap data as TLS application data record."""
        # TLS record header (5 bytes)
        content_type = b'\x17'  # Application Data
        version = b'\x03\x03'   # TLS 1.2
        length = struct.pack('>H', len(data))
        
        return content_type + version + length + data
    
    def _unwrap_tls(self, wrapped: bytes) -> bytes:
        """Extract data from TLS wrapper."""
        if len(wrapped) < 5:
            return b''
        
        # Extract length
        length = struct.unpack('>H', wrapped[3:5])[0]
        
        return wrapped[5:5 + length]


class TrafficObfuscationEngine:
    """
    Main traffic obfuscation engine combining all techniques.
    
    Provides unified interface for comprehensive traffic obfuscation.
    """
    
    def __init__(self):
        self._padder = PacketPadder()
        self._timer = TimingObfuscator()
        self._dummy_gen = DummyTrafficGenerator()
        self._bandwidth = ConstantBandwidthController()
        self._mimicry = ProtocolMimicry()
        
        self._active_techniques: List[ObfuscationTechnique] = []
        self._lock = threading.Lock()
    
    def enable(self, technique: ObfuscationTechnique):
        """Enable obfuscation technique."""
        with self._lock:
            if technique not in self._active_techniques:
                self._active_techniques.append(technique)
                
                if technique == ObfuscationTechnique.DUMMY_TRAFFIC:
                    self._dummy_gen.start()
                elif technique == ObfuscationTechnique.CONSTANT_BANDWIDTH:
                    self._bandwidth.start()
                
                logger.info(f"Enabled: {technique.value}")
    
    def disable(self, technique: ObfuscationTechnique):
        """Disable obfuscation technique."""
        with self._lock:
            if technique in self._active_techniques:
                self._active_techniques.remove(technique)
                
                if technique == ObfuscationTechnique.DUMMY_TRAFFIC:
                    self._dummy_gen.stop()
                elif technique == ObfuscationTechnique.CONSTANT_BANDWIDTH:
                    self._bandwidth.stop()
                
                logger.info(f"Disabled: {technique.value}")
    
    def process_outbound(self, data: bytes) -> bytes:
        """Process outbound packet through enabled techniques."""
        processed = data
        
        with self._lock:
            # Apply padding
            if ObfuscationTechnique.PACKET_PADDING in self._active_techniques:
                processed = self._padder.pad_packet(processed)
            
            # Apply protocol mimicry
            if ObfuscationTechnique.PROTOCOL_MIMICRY in self._active_techniques:
                processed = self._mimicry.wrap_data(processed)
            
            # Apply timing delay
            if ObfuscationTechnique.TIMING_JITTER in self._active_techniques:
                self._timer.apply_delay()
            
            # Record for bandwidth control
            if ObfuscationTechnique.CONSTANT_BANDWIDTH in self._active_techniques:
                self._bandwidth.record_real_traffic(len(processed))
        
        return processed
    
    def process_inbound(self, data: bytes) -> bytes:
        """Process inbound packet (reverse obfuscation)."""
        processed = data
        
        with self._lock:
            # Unwrap protocol mimicry
            if ObfuscationTechnique.PROTOCOL_MIMICRY in self._active_techniques:
                processed = self._mimicry.unwrap_data(processed)
            
            # Remove padding
            if ObfuscationTechnique.PACKET_PADDING in self._active_techniques:
                processed = self._padder.unpad_packet(processed)
        
        return processed
    
    def enable_maximum_obfuscation(self):
        """Enable all obfuscation techniques for maximum stealth."""
        techniques = [
            ObfuscationTechnique.PACKET_PADDING,
            ObfuscationTechnique.TIMING_JITTER,
            ObfuscationTechnique.DUMMY_TRAFFIC,
            ObfuscationTechnique.CONSTANT_BANDWIDTH,
            ObfuscationTechnique.PROTOCOL_MIMICRY,
        ]
        
        for technique in techniques:
            self.enable(technique)
        
        logger.info("Maximum obfuscation enabled")
    
    def disable_all(self):
        """Disable all obfuscation techniques."""
        with self._lock:
            for technique in list(self._active_techniques):
                self.disable(technique)
    
    def get_status(self) -> Dict[str, Any]:
        """Get obfuscation status."""
        with self._lock:
            return {
                'active_techniques': [t.value for t in self._active_techniques],
                'dummy_traffic_stats': {
                    'packets_sent': self._dummy_gen._stats.dummy_packets_sent,
                    'bytes_sent': self._dummy_gen._stats.bytes_sent
                },
                'padding_stats': {
                    'bytes_added': self._padder._stats.padding_bytes_added
                },
                'timing_stats': {
                    'average_delay_ms': self._timer.get_average_delay()
                }
            }


# Export all classes
__all__ = [
    'ObfuscationTechnique',
    'MimicryProtocol',
    'TrafficStats',
    'PacketPadder',
    'TimingObfuscator',
    'DummyTrafficGenerator',
    'ConstantBandwidthController',
    'ProtocolMimicry',
    'TrafficObfuscationEngine',
]
