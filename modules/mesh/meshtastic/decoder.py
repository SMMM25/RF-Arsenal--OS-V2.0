#!/usr/bin/env python3
"""
RF Arsenal OS - Meshtastic Network Decoder
===========================================

Passive Meshtastic mesh network monitoring and analysis.

Capabilities:
- Real-time packet capture and decoding
- Node discovery and enumeration
- Mesh topology mapping
- Traffic pattern analysis
- Channel detection (encrypted vs. unencrypted)
- GPS location tracking (when available)

PASSIVE OPERATION - No transmission required.

LEGAL NOTICE:
AUTHORIZED USE ONLY. Passive monitoring may be legal in some jurisdictions
for security research, but active participation without authorization is illegal.

README COMPLIANCE:
✅ Stealth-First: Passive RX only mode available
✅ RAM-Only: All data stored in volatile memory
✅ No Telemetry: Zero external communications
✅ Real-World Functional: Actual protocol decoding
"""

import threading
import time
import logging
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import secrets

from .protocol import (
    MeshtasticProtocol, MeshtasticPacket, PortNum,
    NodeInfo, Position, node_id_to_str
)

# Import LoRa PHY if available
try:
    from ..lora.phy import LoRaPHY, LoRaConfig, SpreadingFactor, Bandwidth, LoRaRegion
    LORA_PHY_AVAILABLE = True
except ImportError:
    LORA_PHY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MeshNode:
    """Detailed mesh node information."""
    node_id: int
    first_seen: datetime
    last_seen: datetime
    packet_count: int = 0
    
    # User info
    long_name: str = ""
    short_name: str = ""
    hw_model: str = ""
    
    # Position
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[int] = None
    gps_accuracy: Optional[int] = None
    
    # Signal quality
    last_rssi: int = 0
    last_snr: float = 0.0
    avg_rssi: float = 0.0
    avg_snr: float = 0.0
    
    # Network role
    is_router: bool = False
    is_gateway: bool = False
    channels_seen: List[int] = field(default_factory=list)
    
    # Neighbors (nodes this node has relayed packets for)
    neighbors: Dict[int, float] = field(default_factory=dict)  # node_id -> last SNR
    
    # Telemetry
    battery_level: Optional[int] = None
    voltage: Optional[float] = None
    channel_utilization: Optional[float] = None
    air_util_tx: Optional[float] = None


@dataclass
class MeshLink:
    """Link between two mesh nodes."""
    from_node: int
    to_node: int
    first_seen: datetime
    last_seen: datetime
    packet_count: int = 0
    avg_snr: float = 0.0
    hop_count: int = 1
    bidirectional: bool = False


@dataclass
class ChannelInfo:
    """Discovered channel information."""
    index: int
    encrypted: bool = True
    packets_seen: int = 0
    nodes_seen: List[int] = field(default_factory=list)
    last_activity: Optional[datetime] = None
    psk_known: bool = False


class MeshtasticDecoder:
    """
    Meshtastic network passive decoder and analyzer.
    
    Monitors Meshtastic mesh networks to:
    - Discover and track nodes
    - Map network topology
    - Analyze traffic patterns
    - Track GPS-enabled nodes
    - Detect channel encryption status
    
    Operates in passive receive-only mode for stealth operation.
    """
    
    # Meshtastic regional frequencies
    REGIONAL_FREQUENCIES = {
        'US': [
            903_080_000, 905_240_000, 907_400_000, 909_560_000,
            911_720_000, 913_880_000, 906_160_000, 907_320_000,
            # ... 64 uplink + 8 downlink channels total
        ],
        'EU': [
            868_100_000, 868_300_000, 868_500_000,
            867_100_000, 867_300_000, 867_500_000, 867_700_000, 867_900_000,
        ],
        'AU': [
            915_400_000, 915_600_000, 915_800_000, 916_000_000,
            916_200_000, 916_400_000, 916_600_000, 916_800_000,
        ],
        'AS': [
            923_200_000, 923_400_000, 923_600_000, 923_800_000,
        ],
    }
    
    # Common Meshtastic presets
    PRESETS = {
        'LONG_FAST': {
            'spreading_factor': SpreadingFactor.SF11 if LORA_PHY_AVAILABLE else 11,
            'bandwidth': Bandwidth.BW_250K if LORA_PHY_AVAILABLE else 250_000,
            'coding_rate': '4/5',
        },
        'LONG_SLOW': {
            'spreading_factor': SpreadingFactor.SF12 if LORA_PHY_AVAILABLE else 12,
            'bandwidth': Bandwidth.BW_125K if LORA_PHY_AVAILABLE else 125_000,
            'coding_rate': '4/8',
        },
        'MEDIUM_FAST': {
            'spreading_factor': SpreadingFactor.SF9 if LORA_PHY_AVAILABLE else 9,
            'bandwidth': Bandwidth.BW_250K if LORA_PHY_AVAILABLE else 250_000,
            'coding_rate': '4/5',
        },
        'SHORT_FAST': {
            'spreading_factor': SpreadingFactor.SF7 if LORA_PHY_AVAILABLE else 7,
            'bandwidth': Bandwidth.BW_250K if LORA_PHY_AVAILABLE else 250_000,
            'coding_rate': '4/5',
        },
    }
    
    def __init__(self, hardware_controller=None, region: str = 'US'):
        """
        Initialize Meshtastic decoder.
        
        Args:
            hardware_controller: SDR hardware controller
            region: Geographic region (US, EU, AU, AS)
        """
        self.hw = hardware_controller
        self.region = region
        
        # Initialize LoRa PHY
        self.phy = LoRaPHY(hardware_controller) if LORA_PHY_AVAILABLE else None
        
        # Initialize protocol handler
        self.protocol = MeshtasticProtocol()
        
        # Discovered network data (RAM-only)
        self.nodes: Dict[int, MeshNode] = {}
        self.links: Dict[tuple, MeshLink] = {}
        self.channels: Dict[int, ChannelInfo] = {}
        self.packets: List[MeshtasticPacket] = []  # Limited buffer
        
        # Statistics
        self._stats = {
            'packets_received': 0,
            'packets_decoded': 0,
            'decode_errors': 0,
            'nodes_discovered': 0,
            'channels_detected': 0,
            'encrypted_packets': 0,
            'position_updates': 0,
        }
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Callbacks
        self._packet_callbacks: List[Callable[[MeshtasticPacket], None]] = []
        self._node_callbacks: List[Callable[[MeshNode], None]] = []
        
        # Configuration
        self.max_packet_buffer = 10000  # Keep last N packets in RAM
        self.node_timeout_seconds = 3600  # Consider node offline after this
        
        logger.info(f"MeshtasticDecoder initialized: region={region}")
    
    def configure(
        self,
        frequency_hz: int = 906_875_000,
        preset: str = 'LONG_FAST',
        channel_key: Optional[bytes] = None
    ) -> bool:
        """
        Configure decoder for specific Meshtastic parameters.
        
        Args:
            frequency_hz: Center frequency in Hz
            preset: Meshtastic preset name
            channel_key: Encryption key (None for default channel)
            
        Returns:
            True if configuration successful
        """
        with self._lock:
            # Get preset parameters
            params = self.PRESETS.get(preset, self.PRESETS['LONG_FAST'])
            
            # Configure LoRa PHY
            if self.phy and LORA_PHY_AVAILABLE:
                config = LoRaConfig(
                    frequency_hz=frequency_hz,
                    spreading_factor=params['spreading_factor'],
                    bandwidth=params['bandwidth'],
                    sync_word=0x2B,  # Meshtastic sync word
                    preamble_symbols=16,
                    crc_enabled=True,
                )
                
                if not self.phy.configure(config):
                    logger.error("Failed to configure LoRa PHY")
                    return False
            
            # Configure protocol encryption
            if channel_key:
                self.protocol.set_channel_key(channel_key)
            
            logger.info(f"Decoder configured: freq={frequency_hz/1e6:.3f} MHz, preset={preset}")
            return True
    
    def start_monitoring(self, duration_seconds: Optional[int] = None):
        """
        Start passive monitoring of Meshtastic traffic.
        
        Args:
            duration_seconds: Monitor for this duration (None = indefinite)
        """
        with self._lock:
            if self._monitoring:
                logger.warning("Already monitoring")
                return
            
            self._monitoring = True
            
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                args=(duration_seconds,),
                daemon=True
            )
            self._monitor_thread.start()
            
            logger.info("Started Meshtastic monitoring")
    
    def stop_monitoring(self):
        """Stop passive monitoring."""
        with self._lock:
            self._monitoring = False
            
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5.0)
                self._monitor_thread = None
            
            logger.info("Stopped Meshtastic monitoring")
    
    def _monitor_loop(self, duration_seconds: Optional[int]):
        """Main monitoring loop."""
        start_time = time.time()
        
        while self._monitoring:
            # Check duration
            if duration_seconds and (time.time() - start_time) >= duration_seconds:
                self._monitoring = False
                break
            
            try:
                # Receive samples from hardware
                if self.hw and self.phy:
                    # Get samples (1 second worth)
                    samples = self.hw.receive(self.phy.sample_rate)
                    
                    if samples is not None and len(samples) > 0:
                        # Demodulate LoRa packets
                        lora_packets = self.phy.demodulate(samples)
                        
                        for lora_pkt in lora_packets:
                            self._process_lora_packet(lora_pkt)
                else:
                    # No hardware - sleep to avoid busy loop
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(0.1)
    
    def _process_lora_packet(self, lora_packet):
        """Process a received LoRa packet."""
        try:
            self._stats['packets_received'] += 1
            
            # Decode as Meshtastic
            mesh_packet = self.protocol.decode_packet(
                lora_packet.payload,
                rssi=int(lora_packet.rssi_dbm),
                snr=lora_packet.snr_db
            )
            
            if mesh_packet:
                self._process_mesh_packet(mesh_packet)
                self._stats['packets_decoded'] += 1
            else:
                self._stats['decode_errors'] += 1
                
        except Exception as e:
            logger.debug(f"Packet processing error: {e}")
            self._stats['decode_errors'] += 1
    
    def _process_mesh_packet(self, packet: MeshtasticPacket):
        """Process a decoded Meshtastic packet."""
        with self._lock:
            # Store packet (limited buffer)
            self.packets.append(packet)
            if len(self.packets) > self.max_packet_buffer:
                self.packets.pop(0)
            
            # Update node information
            self._update_node(packet)
            
            # Update link information
            self._update_links(packet)
            
            # Update channel information
            self._update_channel(packet)
            
            # Track encrypted packets
            if packet.encrypted:
                self._stats['encrypted_packets'] += 1
            
            # Process specific message types
            if packet.port_num == PortNum.POSITION_APP:
                self._process_position(packet)
            elif packet.port_num == PortNum.NODEINFO_APP:
                self._process_nodeinfo(packet)
            elif packet.port_num == PortNum.TELEMETRY_APP:
                self._process_telemetry(packet)
            elif packet.port_num == PortNum.NEIGHBORINFO_APP:
                self._process_neighborinfo(packet)
            
            # Invoke callbacks
            for callback in self._packet_callbacks:
                try:
                    callback(packet)
                except Exception as e:
                    logger.error(f"Packet callback error: {e}")
    
    def _update_node(self, packet: MeshtasticPacket):
        """Update or create node from packet."""
        node_id = packet.header.from_node
        now = datetime.utcnow()
        
        if node_id not in self.nodes:
            self.nodes[node_id] = MeshNode(
                node_id=node_id,
                first_seen=now,
                last_seen=now,
            )
            self._stats['nodes_discovered'] += 1
            logger.info(f"Discovered new node: {node_id_to_str(node_id)}")
        
        node = self.nodes[node_id]
        node.last_seen = now
        node.packet_count += 1
        node.last_rssi = packet.header.rx_rssi
        node.last_snr = packet.header.rx_snr
        
        # Update running averages
        alpha = 0.1  # Exponential moving average factor
        node.avg_rssi = alpha * node.last_rssi + (1 - alpha) * node.avg_rssi
        node.avg_snr = alpha * node.last_snr + (1 - alpha) * node.avg_snr
        
        # Track channels
        if packet.header.channel not in node.channels_seen:
            node.channels_seen.append(packet.header.channel)
        
        # Invoke node callbacks
        for callback in self._node_callbacks:
            try:
                callback(node)
            except Exception as e:
                logger.error(f"Node callback error: {e}")
    
    def _update_links(self, packet: MeshtasticPacket):
        """Update mesh link information."""
        from_node = packet.header.from_node
        to_node = packet.header.to_node
        now = datetime.utcnow()
        
        # Create link key (ordered pair for bidirectional detection)
        link_key = (min(from_node, to_node), max(from_node, to_node))
        
        if link_key not in self.links:
            self.links[link_key] = MeshLink(
                from_node=from_node,
                to_node=to_node,
                first_seen=now,
                last_seen=now,
            )
        
        link = self.links[link_key]
        link.last_seen = now
        link.packet_count += 1
        link.avg_snr = 0.1 * packet.header.rx_snr + 0.9 * link.avg_snr
        
        # Check for bidirectional
        if link.from_node != from_node:
            link.bidirectional = True
    
    def _update_channel(self, packet: MeshtasticPacket):
        """Update channel information."""
        channel_idx = packet.header.channel
        
        if channel_idx not in self.channels:
            self.channels[channel_idx] = ChannelInfo(index=channel_idx)
            self._stats['channels_detected'] += 1
        
        channel = self.channels[channel_idx]
        channel.packets_seen += 1
        channel.last_activity = datetime.utcnow()
        channel.encrypted = packet.encrypted
        
        if packet.header.from_node not in channel.nodes_seen:
            channel.nodes_seen.append(packet.header.from_node)
    
    def _process_position(self, packet: MeshtasticPacket):
        """Process position update."""
        if not isinstance(packet.decoded_payload, Position):
            return
        
        pos = packet.decoded_payload
        node = self.nodes.get(packet.header.from_node)
        
        if node:
            node.latitude = pos.latitude
            node.longitude = pos.longitude
            node.altitude = pos.altitude
            self._stats['position_updates'] += 1
            
            logger.debug(f"Position update: {node_id_to_str(node.node_id)} "
                        f"-> ({pos.latitude:.6f}, {pos.longitude:.6f})")
    
    def _process_nodeinfo(self, packet: MeshtasticPacket):
        """Process node info update."""
        if not isinstance(packet.decoded_payload, NodeInfo):
            return
        
        info = packet.decoded_payload
        node = self.nodes.get(packet.header.from_node)
        
        if node and info.user:
            node.long_name = info.user.long_name
            node.short_name = info.user.short_name
            node.hw_model = str(info.user.hw_model)
    
    def _process_telemetry(self, packet: MeshtasticPacket):
        """Process telemetry update."""
        if not isinstance(packet.decoded_payload, dict):
            return
        
        telemetry = packet.decoded_payload
        node = self.nodes.get(packet.header.from_node)
        
        if node:
            node.battery_level = telemetry.get('battery_level')
            node.voltage = telemetry.get('voltage')
            node.channel_utilization = telemetry.get('channel_utilization')
    
    def _process_neighborinfo(self, packet: MeshtasticPacket):
        """Process neighbor info update."""
        if not isinstance(packet.decoded_payload, list):
            return
        
        neighbors = packet.decoded_payload
        node = self.nodes.get(packet.header.from_node)
        
        if node:
            for neighbor in neighbors:
                neighbor_id = neighbor.get('node_id')
                snr = neighbor.get('snr', 0.0)
                if neighbor_id:
                    node.neighbors[neighbor_id] = snr
    
    # Public query methods
    
    def get_nodes(self) -> Dict[int, MeshNode]:
        """Get all discovered nodes."""
        with self._lock:
            return self.nodes.copy()
    
    def get_active_nodes(self, timeout_seconds: int = 300) -> List[MeshNode]:
        """Get nodes active within timeout period."""
        with self._lock:
            cutoff = datetime.utcnow() - timedelta(seconds=timeout_seconds)
            return [n for n in self.nodes.values() if n.last_seen >= cutoff]
    
    def get_node_positions(self) -> List[Dict[str, Any]]:
        """Get all nodes with known positions."""
        with self._lock:
            positions = []
            for node in self.nodes.values():
                if node.latitude is not None and node.longitude is not None:
                    positions.append({
                        'node_id': node.node_id,
                        'node_str': node_id_to_str(node.node_id),
                        'name': node.long_name or node.short_name,
                        'latitude': node.latitude,
                        'longitude': node.longitude,
                        'altitude': node.altitude,
                        'last_seen': node.last_seen.isoformat(),
                    })
            return positions
    
    def get_topology(self) -> Dict[str, Any]:
        """Get mesh network topology."""
        with self._lock:
            return {
                'nodes': [
                    {
                        'id': node_id_to_str(n.node_id),
                        'name': n.long_name or n.short_name or node_id_to_str(n.node_id),
                        'rssi': n.avg_rssi,
                        'snr': n.avg_snr,
                        'packets': n.packet_count,
                        'has_gps': n.latitude is not None,
                    }
                    for n in self.nodes.values()
                ],
                'links': [
                    {
                        'source': node_id_to_str(link.from_node),
                        'target': node_id_to_str(link.to_node),
                        'snr': link.avg_snr,
                        'packets': link.packet_count,
                        'bidirectional': link.bidirectional,
                    }
                    for link in self.links.values()
                ],
            }
    
    def get_channels(self) -> Dict[int, ChannelInfo]:
        """Get all detected channels."""
        with self._lock:
            return self.channels.copy()
    
    def get_recent_packets(self, count: int = 100) -> List[MeshtasticPacket]:
        """Get most recent packets."""
        with self._lock:
            return self.packets[-count:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get decoder statistics."""
        with self._lock:
            stats = self._stats.copy()
            stats['nodes_online'] = len(self.get_active_nodes())
            stats['total_nodes'] = len(self.nodes)
            stats['total_links'] = len(self.links)
            stats['monitoring'] = self._monitoring
            return stats
    
    # Callback registration
    
    def on_packet(self, callback: Callable[[MeshtasticPacket], None]):
        """Register callback for new packets."""
        self._packet_callbacks.append(callback)
    
    def on_node_discovered(self, callback: Callable[[MeshNode], None]):
        """Register callback for node updates."""
        self._node_callbacks.append(callback)
    
    # Utility methods
    
    def clear_data(self):
        """Clear all collected data (RAM wipe)."""
        with self._lock:
            self.nodes.clear()
            self.links.clear()
            self.channels.clear()
            self.packets.clear()
            self._stats = {k: 0 for k in self._stats}
            logger.info("Cleared all Meshtastic data from RAM")
    
    def export_nodes_geojson(self) -> Dict[str, Any]:
        """Export nodes with positions as GeoJSON."""
        features = []
        
        for pos in self.get_node_positions():
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [pos['longitude'], pos['latitude']]
                },
                'properties': {
                    'node_id': pos['node_str'],
                    'name': pos['name'],
                    'altitude': pos['altitude'],
                    'last_seen': pos['last_seen'],
                }
            })
        
        return {
            'type': 'FeatureCollection',
            'features': features,
        }


# Factory function
def create_meshtastic_decoder(hardware_controller=None, region: str = 'US') -> MeshtasticDecoder:
    """Create and return a Meshtastic decoder instance."""
    return MeshtasticDecoder(hardware_controller, region)


# Example usage
if __name__ == "__main__":
    print("=== Meshtastic Decoder Test ===")
    
    decoder = MeshtasticDecoder(region='US')
    
    # Configure for LongFast preset
    decoder.configure(
        frequency_hz=906_875_000,
        preset='LONG_FAST',
    )
    
    print(f"\nDecoder configured for region: {decoder.region}")
    print(f"Initial stats: {decoder.get_stats()}")
    
    # In real usage, would start monitoring:
    # decoder.start_monitoring(duration_seconds=300)
    
    print("\n=== Meshtastic Decoder Test Complete ===")
