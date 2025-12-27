#!/usr/bin/env python3
"""
RF Arsenal OS - Meshtastic SIGINT (Signals Intelligence)
=========================================================

Advanced signals intelligence capabilities for Meshtastic networks.

Intelligence Categories:
- COMINT: Communications Intelligence (message content, patterns)
- ELINT: Electronic Intelligence (signal characteristics)
- GEOINT: Geospatial Intelligence (location tracking)
- NETINT: Network Intelligence (topology, relationships)

Capabilities:
- Traffic pattern analysis
- Social graph construction
- Location tracking and history
- Channel activity profiling
- Timing analysis
- Encryption detection
- Network vulnerability assessment

LEGAL NOTICE:
Intelligence gathering on mesh networks may be restricted by law.
Obtain proper authorization before conducting SIGINT operations.

README COMPLIANCE:
✅ Stealth-First: Passive collection only
✅ RAM-Only: All intelligence stored in volatile memory
✅ No Telemetry: No external data transmission
✅ Emergency Wipe: Clear all collected intel on command
"""

import threading
import time
import logging
import hashlib
from typing import Optional, Dict, List, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json

from .protocol import (
    MeshtasticPacket, PortNum, NodeInfo, Position,
    node_id_to_str
)
from .decoder import MeshtasticDecoder, MeshNode

logger = logging.getLogger(__name__)


@dataclass
class CommunicationPattern:
    """Analyzed communication pattern between nodes."""
    node_a: int
    node_b: int
    first_contact: datetime
    last_contact: datetime
    message_count: int = 0
    avg_interval_seconds: float = 0.0
    peak_hours: List[int] = field(default_factory=list)
    channels_used: Set[int] = field(default_factory=set)
    encrypted_ratio: float = 0.0


@dataclass
class LocationHistory:
    """Historical location data for a node."""
    node_id: int
    positions: List[Tuple[datetime, float, float, int]] = field(default_factory=list)
    
    def add_position(self, timestamp: datetime, lat: float, lon: float, alt: int):
        """Add position to history."""
        self.positions.append((timestamp, lat, lon, alt))
        # Keep last 1000 positions
        if len(self.positions) > 1000:
            self.positions.pop(0)
    
    def get_movement_distance_km(self) -> float:
        """Calculate total movement distance in km."""
        if len(self.positions) < 2:
            return 0.0
        
        total_km = 0.0
        for i in range(1, len(self.positions)):
            lat1, lon1 = self.positions[i-1][1], self.positions[i-1][2]
            lat2, lon2 = self.positions[i][1], self.positions[i][2]
            total_km += self._haversine(lat1, lon1, lat2, lon2)
        
        return total_km
    
    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in km."""
        import math
        R = 6371  # Earth radius in km
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c


@dataclass
class ChannelProfile:
    """Intelligence profile for a mesh channel."""
    index: int
    encrypted: bool
    first_seen: datetime
    last_activity: datetime
    total_packets: int = 0
    unique_nodes: Set[int] = field(default_factory=set)
    message_types: Dict[str, int] = field(default_factory=dict)
    avg_packets_per_hour: float = 0.0
    peak_activity_hour: Optional[int] = None
    
    # Encryption analysis
    key_known: bool = False
    key_hash: Optional[str] = None  # Hash of known key for tracking


@dataclass
class NetworkVulnerability:
    """Identified network vulnerability."""
    vuln_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    affected_nodes: List[int]
    recommendations: List[str]
    discovered: datetime = field(default_factory=datetime.utcnow)


class MeshtasticSIGINT:
    """
    Meshtastic Signals Intelligence System.
    
    Performs passive intelligence collection and analysis:
    
    1. COMINT (Communications Intelligence)
       - Message interception and analysis
       - Communication pattern detection
       - Social graph construction
    
    2. ELINT (Electronic Intelligence)
       - Signal characteristics analysis
       - Transmission pattern detection
       - Device fingerprinting
    
    3. GEOINT (Geospatial Intelligence)
       - Location tracking
       - Movement pattern analysis
       - Coverage mapping
    
    4. NETINT (Network Intelligence)
       - Topology mapping
       - Vulnerability assessment
       - Critical node identification
    
    All data stored in RAM only - emergency wipe available.
    """
    
    def __init__(self, decoder: Optional[MeshtasticDecoder] = None):
        """
        Initialize SIGINT system.
        
        Args:
            decoder: Meshtastic decoder for live data feed
        """
        self.decoder = decoder
        self._lock = threading.RLock()
        
        # Intelligence databases (RAM-only)
        self.communication_patterns: Dict[tuple, CommunicationPattern] = {}
        self.location_histories: Dict[int, LocationHistory] = {}
        self.channel_profiles: Dict[int, ChannelProfile] = {}
        self.vulnerabilities: List[NetworkVulnerability] = []
        
        # Social graph
        self.social_graph: Dict[int, Set[int]] = defaultdict(set)
        
        # Timing analysis
        self.packet_times: Dict[int, List[datetime]] = defaultdict(list)
        
        # Message content analysis (hashed for privacy)
        self.message_hashes: Dict[str, List[Tuple[int, datetime]]] = {}
        
        # Statistics
        self._stats = {
            'packets_analyzed': 0,
            'patterns_detected': 0,
            'locations_tracked': 0,
            'vulnerabilities_found': 0,
        }
        
        # Register with decoder if available
        if decoder:
            decoder.on_packet(self._analyze_packet)
        
        logger.info("MeshtasticSIGINT initialized")
    
    def _analyze_packet(self, packet: MeshtasticPacket):
        """Analyze incoming packet for intelligence."""
        with self._lock:
            self._stats['packets_analyzed'] += 1
            
            # Update communication patterns
            self._update_communication_pattern(packet)
            
            # Update timing analysis
            self._update_timing(packet)
            
            # Update channel profile
            self._update_channel_profile(packet)
            
            # Process specific message types
            if packet.port_num == PortNum.POSITION_APP:
                self._process_position_intel(packet)
            elif packet.port_num == PortNum.TEXT_MESSAGE_APP:
                self._process_message_intel(packet)
            elif packet.port_num == PortNum.NODEINFO_APP:
                self._process_nodeinfo_intel(packet)
            elif packet.port_num == PortNum.NEIGHBORINFO_APP:
                self._process_neighbor_intel(packet)
    
    def _update_communication_pattern(self, packet: MeshtasticPacket):
        """Update communication pattern between nodes."""
        from_node = packet.header.from_node
        to_node = packet.header.to_node
        now = datetime.utcnow()
        
        # Skip broadcasts for direct pattern analysis
        if to_node == 0xFFFFFFFF:
            return
        
        # Create ordered pair key
        key = (min(from_node, to_node), max(from_node, to_node))
        
        if key not in self.communication_patterns:
            self.communication_patterns[key] = CommunicationPattern(
                node_a=key[0],
                node_b=key[1],
                first_contact=now,
                last_contact=now,
            )
            self._stats['patterns_detected'] += 1
        
        pattern = self.communication_patterns[key]
        
        # Update pattern stats
        if pattern.message_count > 0:
            elapsed = (now - pattern.last_contact).total_seconds()
            pattern.avg_interval_seconds = (
                0.9 * pattern.avg_interval_seconds + 0.1 * elapsed
            )
        
        pattern.last_contact = now
        pattern.message_count += 1
        pattern.channels_used.add(packet.header.channel)
        
        # Track peak hours
        hour = now.hour
        if hour not in pattern.peak_hours:
            pattern.peak_hours.append(hour)
        
        # Update encrypted ratio
        if packet.encrypted:
            pattern.encrypted_ratio = (
                pattern.encrypted_ratio * 0.9 + 0.1
            )
        else:
            pattern.encrypted_ratio = pattern.encrypted_ratio * 0.9
        
        # Update social graph
        self.social_graph[from_node].add(to_node)
        self.social_graph[to_node].add(from_node)
    
    def _update_timing(self, packet: MeshtasticPacket):
        """Update timing analysis for node."""
        node_id = packet.header.from_node
        now = datetime.utcnow()
        
        self.packet_times[node_id].append(now)
        
        # Keep last 1000 timestamps per node
        if len(self.packet_times[node_id]) > 1000:
            self.packet_times[node_id].pop(0)
    
    def _update_channel_profile(self, packet: MeshtasticPacket):
        """Update channel profile."""
        channel_idx = packet.header.channel
        now = datetime.utcnow()
        
        if channel_idx not in self.channel_profiles:
            self.channel_profiles[channel_idx] = ChannelProfile(
                index=channel_idx,
                encrypted=packet.encrypted,
                first_seen=now,
                last_activity=now,
            )
        
        profile = self.channel_profiles[channel_idx]
        profile.last_activity = now
        profile.total_packets += 1
        profile.unique_nodes.add(packet.header.from_node)
        
        # Track message types
        port_name = packet.port_num.name
        profile.message_types[port_name] = profile.message_types.get(port_name, 0) + 1
        
        # Update encrypted status
        if packet.encrypted:
            profile.encrypted = True
    
    def _process_position_intel(self, packet: MeshtasticPacket):
        """Extract intelligence from position message."""
        if not isinstance(packet.decoded_payload, Position):
            return
        
        pos = packet.decoded_payload
        node_id = packet.header.from_node
        now = datetime.utcnow()
        
        if node_id not in self.location_histories:
            self.location_histories[node_id] = LocationHistory(node_id=node_id)
        
        self.location_histories[node_id].add_position(
            now, pos.latitude, pos.longitude, pos.altitude
        )
        
        self._stats['locations_tracked'] += 1
    
    def _process_message_intel(self, packet: MeshtasticPacket):
        """Extract intelligence from text message."""
        if not isinstance(packet.decoded_payload, str):
            return
        
        # Hash message for pattern detection (privacy preserving)
        msg_hash = hashlib.sha256(packet.decoded_payload.encode()).hexdigest()[:16]
        
        if msg_hash not in self.message_hashes:
            self.message_hashes[msg_hash] = []
        
        self.message_hashes[msg_hash].append(
            (packet.header.from_node, datetime.utcnow())
        )
    
    def _process_nodeinfo_intel(self, packet: MeshtasticPacket):
        """Extract intelligence from node info."""
        # Node info processed by decoder, just note it here
        pass
    
    def _process_neighbor_intel(self, packet: MeshtasticPacket):
        """Extract intelligence from neighbor info."""
        if not isinstance(packet.decoded_payload, list):
            return
        
        from_node = packet.header.from_node
        
        for neighbor in packet.decoded_payload:
            neighbor_id = neighbor.get('node_id')
            if neighbor_id:
                self.social_graph[from_node].add(neighbor_id)
                self.social_graph[neighbor_id].add(from_node)
    
    # ============== ANALYSIS METHODS ==============
    
    def analyze_network_topology(self) -> Dict[str, Any]:
        """
        Analyze complete network topology.
        
        Returns:
            Network topology analysis including:
            - Node count and classifications
            - Link analysis
            - Critical nodes
            - Potential bottlenecks
        """
        with self._lock:
            if not self.decoder:
                return {'error': 'No decoder attached'}
            
            nodes = self.decoder.get_nodes()
            
            # Calculate node centrality
            centrality = {}
            for node_id in self.social_graph:
                # Simple degree centrality
                centrality[node_id] = len(self.social_graph[node_id])
            
            # Find critical nodes (high centrality)
            sorted_centrality = sorted(
                centrality.items(), key=lambda x: x[1], reverse=True
            )
            critical_nodes = [n[0] for n in sorted_centrality[:5]]
            
            # Find isolated nodes
            isolated = [
                n for n in nodes if n not in self.social_graph or 
                len(self.social_graph[n]) == 0
            ]
            
            # Calculate network density
            total_possible = len(nodes) * (len(nodes) - 1) / 2
            actual_links = sum(len(neighbors) for neighbors in self.social_graph.values()) / 2
            density = actual_links / total_possible if total_possible > 0 else 0
            
            return {
                'total_nodes': len(nodes),
                'total_links': int(actual_links),
                'network_density': density,
                'critical_nodes': [node_id_to_str(n) for n in critical_nodes],
                'isolated_nodes': [node_id_to_str(n) for n in isolated],
                'centrality_scores': {
                    node_id_to_str(k): v for k, v in sorted_centrality[:20]
                },
            }
    
    def analyze_communication_patterns(self) -> List[Dict[str, Any]]:
        """
        Analyze communication patterns between nodes.
        
        Returns:
            List of communication pattern summaries
        """
        with self._lock:
            patterns = []
            
            for key, pattern in self.communication_patterns.items():
                patterns.append({
                    'node_a': node_id_to_str(pattern.node_a),
                    'node_b': node_id_to_str(pattern.node_b),
                    'message_count': pattern.message_count,
                    'avg_interval_seconds': pattern.avg_interval_seconds,
                    'first_contact': pattern.first_contact.isoformat(),
                    'last_contact': pattern.last_contact.isoformat(),
                    'channels': list(pattern.channels_used),
                    'encrypted_ratio': pattern.encrypted_ratio,
                    'peak_hours': pattern.peak_hours,
                })
            
            # Sort by message count
            patterns.sort(key=lambda x: x['message_count'], reverse=True)
            
            return patterns
    
    def get_location_tracks(self) -> Dict[str, Any]:
        """
        Get all location tracking data.
        
        Returns:
            Location tracks for all GPS-enabled nodes
        """
        with self._lock:
            tracks = {}
            
            for node_id, history in self.location_histories.items():
                if history.positions:
                    tracks[node_id_to_str(node_id)] = {
                        'position_count': len(history.positions),
                        'total_distance_km': history.get_movement_distance_km(),
                        'first_position': {
                            'time': history.positions[0][0].isoformat(),
                            'lat': history.positions[0][1],
                            'lon': history.positions[0][2],
                        },
                        'last_position': {
                            'time': history.positions[-1][0].isoformat(),
                            'lat': history.positions[-1][1],
                            'lon': history.positions[-1][2],
                        },
                        'positions': [
                            {
                                'time': p[0].isoformat(),
                                'lat': p[1],
                                'lon': p[2],
                                'alt': p[3],
                            }
                            for p in history.positions[-100:]  # Last 100
                        ],
                    }
            
            return tracks
    
    def assess_vulnerabilities(self) -> List[NetworkVulnerability]:
        """
        Assess network for security vulnerabilities.
        
        Returns:
            List of identified vulnerabilities
        """
        with self._lock:
            vulnerabilities = []
            
            if not self.decoder:
                return vulnerabilities
            
            nodes = self.decoder.get_nodes()
            channels = self.decoder.get_channels()
            
            # Check for unencrypted channels
            for idx, channel in channels.items():
                if not channel.encrypted:
                    vulnerabilities.append(NetworkVulnerability(
                        vuln_type="UNENCRYPTED_CHANNEL",
                        severity="HIGH",
                        description=f"Channel {idx} is not encrypted - traffic visible",
                        affected_nodes=list(channel.nodes_seen),
                        recommendations=[
                            "Enable channel encryption",
                            "Use strong PSK",
                            "Consider private channels",
                        ],
                    ))
            
            # Check for single points of failure
            topology = self.analyze_network_topology()
            if 'critical_nodes' in topology:
                for node_str in topology['critical_nodes'][:3]:
                    node_id = int(node_str[1:], 16)  # Remove '!' prefix
                    vulnerabilities.append(NetworkVulnerability(
                        vuln_type="SINGLE_POINT_OF_FAILURE",
                        severity="MEDIUM",
                        description=f"Node {node_str} is critical - network may partition if offline",
                        affected_nodes=[node_id],
                        recommendations=[
                            "Add redundant relay nodes",
                            "Improve mesh coverage",
                            "Consider multiple routing paths",
                        ],
                    ))
            
            # Check for default channel usage
            if 0 in channels and channels[0].total_packets > 10:
                vulnerabilities.append(NetworkVulnerability(
                    vuln_type="DEFAULT_CHANNEL",
                    severity="MEDIUM",
                    description="Using default channel 0 - susceptible to monitoring",
                    affected_nodes=list(channels[0].nodes_seen),
                    recommendations=[
                        "Use custom channel with PSK",
                        "Avoid default LongFast channel",
                    ],
                ))
            
            # Check for isolated nodes (no routing)
            if 'isolated_nodes' in topology:
                for node_str in topology['isolated_nodes']:
                    node_id = int(node_str[1:], 16)
                    vulnerabilities.append(NetworkVulnerability(
                        vuln_type="ISOLATED_NODE",
                        severity="LOW",
                        description=f"Node {node_str} has no detected neighbors",
                        affected_nodes=[node_id],
                        recommendations=[
                            "Verify node is in range",
                            "Check antenna configuration",
                            "Consider relay nodes",
                        ],
                    ))
            
            self._stats['vulnerabilities_found'] = len(vulnerabilities)
            self.vulnerabilities = vulnerabilities
            
            return vulnerabilities
    
    def generate_intelligence_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive intelligence report.
        
        Returns:
            Complete SIGINT report in structured format
        """
        with self._lock:
            topology = self.analyze_network_topology()
            patterns = self.analyze_communication_patterns()
            tracks = self.get_location_tracks()
            vulnerabilities = self.assess_vulnerabilities()
            
            return {
                'generated': datetime.utcnow().isoformat(),
                'summary': {
                    'packets_analyzed': self._stats['packets_analyzed'],
                    'nodes_tracked': len(self.decoder.get_nodes()) if self.decoder else 0,
                    'communication_pairs': len(self.communication_patterns),
                    'location_tracks': len(self.location_histories),
                    'vulnerabilities': len(vulnerabilities),
                },
                'network_topology': topology,
                'communication_patterns': patterns[:20],  # Top 20
                'location_tracking': tracks,
                'channel_profiles': {
                    str(k): {
                        'encrypted': v.encrypted,
                        'packets': v.total_packets,
                        'nodes': len(v.unique_nodes),
                        'message_types': dict(v.message_types),
                    }
                    for k, v in self.channel_profiles.items()
                },
                'vulnerabilities': [
                    {
                        'type': v.vuln_type,
                        'severity': v.severity,
                        'description': v.description,
                        'affected': [node_id_to_str(n) for n in v.affected_nodes[:5]],
                        'recommendations': v.recommendations,
                    }
                    for v in vulnerabilities
                ],
            }
    
    # ============== DATA MANAGEMENT ==============
    
    def emergency_wipe(self):
        """
        Emergency wipe of all collected intelligence.
        
        Clears all data from RAM immediately.
        """
        with self._lock:
            self.communication_patterns.clear()
            self.location_histories.clear()
            self.channel_profiles.clear()
            self.vulnerabilities.clear()
            self.social_graph.clear()
            self.packet_times.clear()
            self.message_hashes.clear()
            self._stats = {k: 0 for k in self._stats}
            
            logger.warning("⚠️ SIGINT data wiped from RAM")
    
    def get_stats(self) -> Dict[str, int]:
        """Get SIGINT statistics."""
        with self._lock:
            return self._stats.copy()


# Factory function
def create_sigint_system(decoder: Optional[MeshtasticDecoder] = None) -> MeshtasticSIGINT:
    """Create SIGINT system instance."""
    return MeshtasticSIGINT(decoder)


# Example usage
if __name__ == "__main__":
    print("=== Meshtastic SIGINT System ===")
    
    sigint = MeshtasticSIGINT()
    
    print(f"Initial stats: {sigint.get_stats()}")
    print("\nSIGINT capabilities:")
    print("  - Communication pattern analysis")
    print("  - Location tracking")
    print("  - Network topology mapping")
    print("  - Vulnerability assessment")
    print("  - Social graph construction")
    
    print("\n=== SIGINT Test Complete ===")
