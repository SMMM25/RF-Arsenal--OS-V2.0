#!/usr/bin/env python3
"""
Mesh Networking System
BLE and LoRaWAN fallback communications for offline operations
Supports: BLE mesh, LoRaWAN, WiFi Direct, peer-to-peer routing
"""

import os
import time
import json
import hashlib
import secrets
import threading
import subprocess
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from queue import Queue
from datetime import datetime


class NetworkType(Enum):
    """Mesh network types"""
    BLE_MESH = "ble_mesh"
    LORA_WAN = "lora_wan"
    WIFI_DIRECT = "wifi_direct"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class MeshNode:
    """Mesh network node information"""
    node_id: str
    node_name: str
    network_type: NetworkType
    address: str
    last_seen: float
    rssi: int  # Signal strength (dBm)
    hop_count: int
    is_gateway: bool
    capabilities: List[str]


@dataclass
class MeshMessage:
    """Message for mesh network transmission"""
    message_id: str
    source_id: str
    destination_id: str  # Use 'broadcast' for all nodes
    payload: bytes
    priority: MessagePriority
    timestamp: float
    ttl: int  # Time-to-live (max hops)
    encrypted: bool
    route: List[str]  # Path taken through mesh


class BLEMeshNetwork:
    """
    Bluetooth Low Energy mesh network
    Short-range (10-100m) peer-to-peer communication
    Low power consumption, suitable for mobile operations
    """
    
    def __init__(self, node_name: str = "rf-arsenal-node"):
        self.node_id = self._generate_node_id()
        self.node_name = node_name
        self.peers = {}
        self.message_queue = Queue()
        self.sent_messages = {}  # Message ID cache to prevent loops
        self.running = False
        self.callbacks = []
        
        # BLE parameters
        self.ble_tx_power = 4  # dBm (adjustable: -20 to +4)
        self.scan_interval = 1.0  # seconds
        self.advertisement_interval = 1.0  # seconds
        self.max_peers = 7  # BLE mesh limitation
        
        # Routing table
        self.routing_table = {}
        
    def start_mesh(self) -> bool:
        """
        Start BLE mesh network
        Begin scanning for peers and advertising presence
        """
        print(f"\n[BLE MESH] Starting mesh network")
        print("="*60)
        print(f"  Node ID: {self.node_id[:12]}...")
        print(f"  Node Name: {self.node_name}")
        print(f"  TX Power: {self.ble_tx_power} dBm")
        print(f"  Max Peers: {self.max_peers}")
        
        try:
            # Check if Bluetooth is available
            result = subprocess.run(['which', 'hciconfig'], 
                                  capture_output=True, timeout=5)
            
            if result.returncode != 0:
                print("[BLE MESH] ⚠ Warning: Bluetooth tools not found (hciconfig)")
                print("  Install: sudo apt install bluez")
                # Continue anyway for demonstration
                
            # Start BLE advertising (announce presence)
            self._start_ble_advertising()
            
            # Start BLE scanning (discover peers)
            self._start_ble_scanning()
            
            # Start message handler
            self.running = True
            threading.Thread(target=self._message_handler, daemon=True).start()
            
            # Start peer maintenance
            threading.Thread(target=self._peer_maintenance, daemon=True).start()
            
            print("="*60)
            print("[BLE MESH] ✓ Mesh network started")
            print("="*60 + "\n")
            return True
            
        except Exception as e:
            print(f"[BLE MESH] ✗ Error starting mesh: {e}")
            return False
            
    def _start_ble_advertising(self):
        """Start BLE advertising to announce node presence"""
        try:
            print("[BLE MESH] → Starting BLE advertisement...")
            
            def advertise_loop():
                while self.running:
                    # Create advertisement packet
                    # Contains: Node ID, Name, Capabilities, Peer count
                    adv_data = {
                        'node_id': self.node_id,
                        'name': self.node_name,
                        'peers': len(self.peers),
                        'capabilities': ['mesh_routing', 'encryption']
                    }
                    
                    # Broadcast presence (simplified - would use BLE advertising packets)
                    # print(f"[BLE MESH] Broadcasting presence...")
                    
                    time.sleep(self.advertisement_interval)
                    
            threading.Thread(target=advertise_loop, daemon=True).start()
            print("  ✓ Advertisement started")
            
        except Exception as e:
            print(f"[BLE MESH] Advertising error: {e}")
            
    def _start_ble_scanning(self):
        """Start BLE scanning to discover peer nodes"""
        try:
            print("[BLE MESH] → Starting BLE scan...")
            
            def scan_loop():
                while self.running:
                    # Scan for BLE devices
                    discovered = self._ble_scan()
                    
                    for device in discovered:
                        if self._is_mesh_node(device):
                            self._add_peer(device)
                            
                    time.sleep(self.scan_interval)
                    
            threading.Thread(target=scan_loop, daemon=True).start()
            print("  ✓ Scanning started")
            
        except Exception as e:
            print(f"[BLE MESH] Scanning error: {e}")
            
    def _ble_scan(self) -> List[Dict]:
        """
        Perform BLE scan for nearby devices
        Returns list of discovered devices
        """
        try:
            # Would use bluetoothctl, bluepy, or bleak library
            # For demonstration, simulate discovery
            
            # Real implementation:
            # result = subprocess.run(['bluetoothctl', 'scan', 'on'], ...)
            # Parse output for RF Arsenal mesh nodes
            
            return []  # Empty for demonstration
            
        except Exception as e:
            return []
            
    def _is_mesh_node(self, device: Dict) -> bool:
        """
        Check if BLE device is an RF Arsenal mesh node
        Identifies nodes by service UUID or advertisement data
        """
        # Check for RF Arsenal mesh UUID or name pattern
        name = device.get('name', '')
        
        return (name.startswith('rf-arsenal') or 
                device.get('service_uuid') == 'RF-ARSENAL-MESH')
        
    def _add_peer(self, device: Dict):
        """Add discovered peer to mesh network"""
        node_id = device.get('node_id', device.get('address', ''))
        
        if node_id == self.node_id:
            return  # Don't add self
            
        if len(self.peers) >= self.max_peers:
            print(f"[BLE MESH] Max peers reached ({self.max_peers}), ignoring new peer")
            return
            
        if node_id not in self.peers:
            peer = MeshNode(
                node_id=node_id,
                node_name=device.get('name', 'Unknown'),
                network_type=NetworkType.BLE_MESH,
                address=device.get('address', ''),
                last_seen=time.time(),
                rssi=device.get('rssi', -100),
                hop_count=1,
                is_gateway=device.get('is_gateway', False),
                capabilities=device.get('capabilities', [])
            )
            
            self.peers[node_id] = peer
            self._update_routing_table()
            
            print(f"[BLE MESH] ✓ New peer: {peer.node_name} ({node_id[:8]}...) RSSI={peer.rssi}dBm")
        else:
            # Update existing peer
            self.peers[node_id].last_seen = time.time()
            self.peers[node_id].rssi = device.get('rssi', -100)
            
    def _peer_maintenance(self):
        """
        Maintain peer list
        Remove stale peers, update routing
        """
        while self.running:
            try:
                current_time = time.time()
                timeout = 30.0  # 30 seconds
                
                stale_peers = []
                for node_id, peer in self.peers.items():
                    if current_time - peer.last_seen > timeout:
                        stale_peers.append(node_id)
                        
                for node_id in stale_peers:
                    peer = self.peers[node_id]
                    print(f"[BLE MESH] Removing stale peer: {peer.node_name} ({node_id[:8]}...)")
                    del self.peers[node_id]
                    
                if stale_peers:
                    self._update_routing_table()
                    
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"[BLE MESH] Peer maintenance error: {e}")
                time.sleep(10)
                
    def send_message(self, destination_id: str, payload: bytes, 
                    priority: MessagePriority = MessagePriority.NORMAL,
                    encrypt: bool = True) -> bool:
        """
        Send message to destination node
        Uses multi-hop routing if necessary
        
        Args:
            destination_id: Target node ID or 'broadcast'
            payload: Message data
            priority: Message priority level
            encrypt: Whether to encrypt payload
            
        Returns:
            True if message was sent successfully
        """
        message_id = self._generate_message_id()
        
        message = MeshMessage(
            message_id=message_id,
            source_id=self.node_id,
            destination_id=destination_id,
            payload=payload,
            priority=priority,
            timestamp=time.time(),
            ttl=10,  # Max 10 hops
            encrypted=encrypt,
            route=[self.node_id]
        )
        
        print(f"\n[BLE MESH] Sending message")
        print(f"  → To: {destination_id if destination_id == 'broadcast' else destination_id[:8]+'...'}")
        print(f"  → Size: {len(payload)} bytes")
        print(f"  → Priority: {priority.name}")
        
        # Encrypt payload if requested
        if message.encrypted:
            message.payload = self._encrypt_payload(message.payload)
            print(f"  → Encrypted: {len(message.payload)} bytes")
            
        # Cache message ID to prevent loops
        self.sent_messages[message_id] = time.time()
        
        # Find route to destination
        if destination_id == 'broadcast':
            # Broadcast to all direct peers
            success_count = 0
            for peer_id in self.peers.keys():
                if self._transmit_to_peer(peer_id, message):
                    success_count += 1
                    
            print(f"[BLE MESH] ✓ Broadcast to {success_count}/{len(self.peers)} peers")
            return success_count > 0
        else:
            # Unicast to specific destination
            route = self._find_route(destination_id)
            
            if route:
                next_hop = route[0]
                print(f"  → Route: {' → '.join([n[:8]+'...' for n in route[:3]])}")
                return self._transmit_to_peer(next_hop, message)
            else:
                print(f"[BLE MESH] ✗ No route to {destination_id[:8]}...")
                return False
            
    def broadcast_message(self, payload: bytes, 
                         priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """Broadcast message to all nodes in mesh"""
        return self.send_message('broadcast', payload, priority)
        
    def _find_route(self, destination_id: str) -> Optional[List[str]]:
        """
        Find route to destination using mesh routing
        Returns list of node IDs representing path to destination
        """
        if destination_id == 'broadcast':
            return list(self.peers.keys())
            
        # Check if destination is direct peer
        if destination_id in self.peers:
            return [destination_id]
            
        # Multi-hop routing using routing table
        if destination_id in self.routing_table:
            return self.routing_table[destination_id]
            
        # No known route
        return None
        
    def _update_routing_table(self):
        """
        Update routing table based on peer information
        Simple distance-vector routing
        """
        self.routing_table = {}
        
        # Direct routes (1 hop)
        for peer_id in self.peers.keys():
            self.routing_table[peer_id] = [peer_id]
            
        # Multi-hop routes would be learned through route discovery
        # In production, implement AODV, OLSR, or similar mesh routing protocol
        
    def _transmit_to_peer(self, peer_id: str, message: MeshMessage) -> bool:
        """
        Transmit message to specific peer via BLE
        
        Args:
            peer_id: Target peer node ID
            message: Message to transmit
            
        Returns:
            True if transmission successful
        """
        if peer_id not in self.peers:
            return False
            
        peer = self.peers[peer_id]
        
        try:
            # Serialize message
            message_data = self._serialize_message(message)
            
            # Update route
            message.route.append(peer_id)
            
            # Transmit via BLE GATT characteristics
            # In production: use bluepy or bleak library to write to GATT characteristic
            # characteristic.write(message_data)
            
            # Simulated transmission
            # print(f"  ✓ Transmitted to {peer.node_name} ({peer_id[:8]}...)")
            return True
            
        except Exception as e:
            print(f"  ✗ Transmission failed to {peer.node_name}: {e}")
            return False
            
    def _message_handler(self):
        """
        Handle received messages from peers
        Process, forward, or deliver messages
        """
        while self.running:
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get()
                    
                    # Check for message loops
                    if message.message_id in self.sent_messages:
                        continue  # Already processed
                        
                    # Cache message ID
                    self.sent_messages[message.message_id] = time.time()
                    
                    # Decrypt if encrypted
                    if message.encrypted:
                        message.payload = self._decrypt_payload(message.payload)
                        
                    # Check if message is for us or broadcast
                    if message.destination_id in [self.node_id, 'broadcast']:
                        # Deliver to application layer
                        self._deliver_message(message)
                    else:
                        # Forward to next hop (mesh routing)
                        self._forward_message(message)
                        
                time.sleep(0.1)
                
            except Exception as e:
                print(f"[BLE MESH] Message handler error: {e}")
                
    def _deliver_message(self, message: MeshMessage):
        """Deliver message to application layer callbacks"""
        print(f"\n[BLE MESH] ← Received message")
        print(f"  ← From: {message.source_id[:8]}...")
        print(f"  ← Size: {len(message.payload)} bytes")
        print(f"  ← Hops: {len(message.route)}")
        
        for callback in self.callbacks:
            try:
                callback(message)
            except Exception as e:
                print(f"[BLE MESH] Callback error: {e}")
                
    def _forward_message(self, message: MeshMessage):
        """Forward message to next hop in mesh"""
        if message.ttl <= 0:
            print(f"[BLE MESH] Message {message.message_id[:8]}... TTL exceeded, dropped")
            return
            
        message.ttl -= 1
        route = self._find_route(message.destination_id)
        
        if route:
            next_hop = route[0]
            print(f"[BLE MESH] Forwarding message to {next_hop[:8]}... (TTL={message.ttl})")
            self._transmit_to_peer(next_hop, message)
        else:
            print(f"[BLE MESH] Cannot forward: no route to {message.destination_id[:8]}...")
            
    def register_callback(self, callback: Callable[[MeshMessage], None]):
        """
        Register callback function for received messages
        
        Args:
            callback: Function that takes MeshMessage as parameter
        """
        self.callbacks.append(callback)
        print(f"[BLE MESH] Callback registered (total: {len(self.callbacks)})")
        
    def get_peer_count(self) -> int:
        """Get number of connected peers"""
        return len(self.peers)
        
    def get_peers(self) -> List[Dict]:
        """Get list of peer nodes with details"""
        peer_list = []
        
        for peer_id, peer in self.peers.items():
            age = time.time() - peer.last_seen
            peer_list.append({
                'id': peer_id[:12],
                'full_id': peer_id,
                'name': peer.node_name,
                'address': peer.address,
                'rssi': peer.rssi,
                'hop_count': peer.hop_count,
                'last_seen_seconds': age,
                'is_gateway': peer.is_gateway,
                'capabilities': peer.capabilities
            })
            
        return peer_list
        
    def _generate_node_id(self) -> str:
        """Generate unique cryptographic node ID"""
        # Use MAC address + random data for uniqueness
        try:
            mac = ':'.join(['{:02x}'.format(secrets.randbelow(256)) for _ in range(6)])
        except:
            mac = 'random'
            
        data = f"{mac}{time.time()}{secrets.token_hex(16)}".encode()
        return hashlib.sha256(data).hexdigest()
        
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        data = f"{self.node_id}{time.time()}{secrets.token_hex(8)}".encode()
        return hashlib.sha256(data).hexdigest()[:16]
        
    def _encrypt_payload(self, payload: bytes) -> bytes:
        """
        Encrypt message payload using AES-256
        In production: use mesh-wide shared key or per-link keys
        """
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend
            
            # Mesh shared key (in production: derive from secure key exchange)
            key = hashlib.sha256(b"RF-ARSENAL-MESH-KEY").digest()
            iv = secrets.token_bytes(16)
            
            # Pad to 16-byte boundary
            padding_length = 16 - (len(payload) % 16)
            padded_payload = payload + bytes([padding_length] * padding_length)
            
            # Encrypt
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(padded_payload) + encryptor.finalize()
            
            # Prepend IV
            return iv + ciphertext
            
        except ImportError:
            print("[BLE MESH] ⚠ Cryptography library not available, skipping encryption")
            return payload
            
    def _decrypt_payload(self, encrypted_payload: bytes) -> bytes:
        """Decrypt message payload"""
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend
            
            # Extract IV
            iv = encrypted_payload[:16]
            ciphertext = encrypted_payload[16:]
            
            # Mesh shared key
            key = hashlib.sha256(b"RF-ARSENAL-MESH-KEY").digest()
            
            # Decrypt
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            padded_payload = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Remove padding
            padding_length = padded_payload[-1]
            payload = padded_payload[:-padding_length]
            
            return payload
            
        except ImportError:
            return encrypted_payload
            
    def _serialize_message(self, message: MeshMessage) -> bytes:
        """Serialize message for transmission"""
        message_dict = {
            'message_id': message.message_id,
            'source_id': message.source_id,
            'destination_id': message.destination_id,
            'payload': message.payload.hex(),  # Hex encode bytes
            'priority': message.priority.value,
            'timestamp': message.timestamp,
            'ttl': message.ttl,
            'encrypted': message.encrypted,
            'route': message.route
        }
        
        return json.dumps(message_dict).encode()
        
    def stop_mesh(self):
        """Stop BLE mesh network"""
        print("\n[BLE MESH] Stopping mesh network...")
        self.running = False
        time.sleep(1)
        print("[BLE MESH] ✓ Mesh network stopped\n")


class LoRaWANNetwork:
    """
    LoRaWAN long-range mesh network
    Long-range (up to 10+ km urban, 40+ km rural)
    Low bandwidth (~300 bps to 50 kbps depending on SF)
    Low power consumption (years on battery)
    """
    
    def __init__(self, node_name: str = "rf-arsenal-lora"):
        self.node_id = self._generate_node_id()
        self.node_name = node_name
        self.running = False
        self.message_queue = Queue()
        self.callbacks = []
        
        # LoRa parameters (US ISM band)
        self.frequency = 915_000_000  # 915 MHz (US), use 868 MHz for EU
        self.spreading_factor = 7  # 7-12 (higher = longer range, lower speed)
        self.bandwidth = 125_000  # 125 kHz
        self.coding_rate = 5  # 4/5
        self.tx_power = 17  # dBm (max 20 dBm)
        self.preamble_length = 8
        
        # Statistics
        self.packets_sent = 0
        self.packets_received = 0
        self.last_rssi = -100
        self.last_snr = 0
        
    def start_network(self) -> bool:
        """
        Start LoRaWAN network
        Initialize LoRa radio hardware
        """
        print(f"\n[LORA] Starting LoRaWAN network")
        print("="*60)
        print(f"  Node ID: {self.node_id[:12]}...")
        print(f"  Node Name: {self.node_name}")
        print(f"  Frequency: {self.frequency / 1e6:.1f} MHz")
        print(f"  SF: {self.spreading_factor}")
        print(f"  Bandwidth: {self.bandwidth / 1000:.0f} kHz")
        print(f"  TX Power: {self.tx_power} dBm")
        
        try:
            # Initialize LoRa hardware (e.g., SX1276, SX1262, SX1268)
            # Would use SPI interface on GPIO pins (e.g., via RPi.GPIO or gpiod)
            
            # Check for LoRa hardware
            # In production: initialize SPI, configure LoRa chip registers
            
            print("[LORA] ⚠ Note: LoRa hardware initialization requires:")
            print("  - SX1276/SX1262 LoRa chip")
            print("  - SPI interface")
            print("  - Libraries: python-lora, sx127x, or similar")
            
            self.running = True
            
            # Start receiver thread
            threading.Thread(target=self._receive_loop, daemon=True).start()
            
            print("="*60)
            print("[LORA] ✓ LoRaWAN network started")
            print("="*60 + "\n")
            return True
            
        except Exception as e:
            print(f"[LORA] ✗ Error starting network: {e}")
            return False
            
    def send_message(self, payload: bytes, 
                    priority: MessagePriority = MessagePriority.NORMAL,
                    destination_id: str = 'broadcast') -> bool:
        """
        Send LoRa message
        All LoRa messages are broadcast (no native addressing)
        
        Args:
            payload: Message data (max ~250 bytes depending on SF)
            priority: Message priority
            destination_id: Logical destination (for application layer)
            
        Returns:
            True if transmission successful
        """
        # Check payload size
        max_payload = self._get_max_payload_size()
        if len(payload) > max_payload:
            print(f"[LORA] ✗ Payload too large ({len(payload)} > {max_payload} bytes)")
            return False
            
        print(f"\n[LORA] Transmitting message")
        print(f"  → Size: {len(payload)} bytes")
        print(f"  → Priority: {priority.name}")
        print(f"  → SF: {self.spreading_factor}")
        
        message = MeshMessage(
            message_id=self._generate_message_id(),
            source_id=self.node_id,
            destination_id=destination_id,
            payload=payload,
            priority=priority,
            timestamp=time.time(),
            ttl=10,
            encrypted=True,
            route=[self.node_id]
        )
        
        try:
            # Encrypt payload
            if message.encrypted:
                message.payload = self._encrypt_payload(message.payload)
                print(f"  → Encrypted: {len(message.payload)} bytes")
                
            # Serialize message
            message_data = self._serialize_message(message)
            
            # Calculate air time
            air_time = self._calculate_air_time(len(message_data))
            print(f"  → Air time: {air_time:.1f} ms")
            
            # Transmit via LoRa radio
            self._lora_transmit(message_data)
            
            self.packets_sent += 1
            print(f"[LORA] ✓ Message transmitted ({self.packets_sent} total)")
            return True
            
        except Exception as e:
            print(f"[LORA] ✗ Transmission error: {e}")
            return False
            
    def _receive_loop(self):
        """Continuously listen for LoRa messages"""
        print("[LORA] Receiver started, listening...")
        
        while self.running:
            try:
                # Receive from LoRa radio (blocking with timeout)
                message_data = self._lora_receive(timeout=1.0)
                
                if message_data:
                    self.packets_received += 1
                    
                    # Deserialize message
                    message = self._deserialize_message(message_data)
                    
                    # Decrypt if encrypted
                    if message.encrypted:
                        message.payload = self._decrypt_payload(message.payload)
                        
                    # Deliver to callbacks
                    self._deliver_message(message)
                    
            except Exception as e:
                # print(f"[LORA] Receive error: {e}")
                time.sleep(1)
                
    def _lora_transmit(self, data: bytes):
        """
        Transmit data via LoRa radio
        In production: interface with SX1276/SX1262 via SPI
        """
        # Would configure LoRa registers and transmit
        # Example (pseudocode):
        # self.radio.set_frequency(self.frequency)
        # self.radio.set_spreading_factor(self.spreading_factor)
        # self.radio.set_tx_power(self.tx_power)
        # self.radio.transmit(data)
        
        pass  # Simulated for demonstration
        
    def _lora_receive(self, timeout: float = 1.0) -> Optional[bytes]:
        """
        Receive data from LoRa radio
        Blocking call with timeout
        
        Returns:
            Received packet data or None
        """
        # Would read from LoRa radio
        # Example (pseudocode):
        # if self.radio.receive_ready():
        #     packet = self.radio.read_packet()
        #     self.last_rssi = self.radio.get_rssi()
        #     self.last_snr = self.radio.get_snr()
        #     return packet
        
        return None  # Simulated
        
    def _calculate_air_time(self, payload_size: int) -> float:
        """
        Calculate LoRa packet air time in milliseconds
        Based on spreading factor, bandwidth, and payload size
        """
        # Simplified calculation
        # Real formula is more complex (see LoRa calculator)
        
        symbol_duration = (2 ** self.spreading_factor) / self.bandwidth * 1000  # ms
        preamble_time = (self.preamble_length + 4.25) * symbol_duration
        
        # Payload symbols (simplified)
        payload_symbols = 8 + max(0, (8 * payload_size - 4 * self.spreading_factor + 28 + 16) / (4 * self.spreading_factor))
        payload_time = payload_symbols * symbol_duration
        
        return preamble_time + payload_time
        
    def _get_max_payload_size(self) -> int:
        """
        Get maximum payload size for current configuration
        Depends on spreading factor
        """
        # LoRa max payload
        if self.spreading_factor <= 9:
            return 250
        elif self.spreading_factor == 10:
            return 200
        elif self.spreading_factor == 11:
            return 150
        else:  # SF12
            return 100
            
    def _deliver_message(self, message: MeshMessage):
        """Deliver message to application callbacks"""
        print(f"\n[LORA] ← Received message")
        print(f"  ← From: {message.source_id[:8]}...")
        print(f"  ← Size: {len(message.payload)} bytes")
        print(f"  ← RSSI: {self.last_rssi} dBm, SNR: {self.last_snr} dB")
        
        for callback in self.callbacks:
            try:
                callback(message)
            except Exception as e:
                print(f"[LORA] Callback error: {e}")
                
    def register_callback(self, callback: Callable[[MeshMessage], None]):
        """Register message callback"""
        self.callbacks.append(callback)
        print(f"[LORA] Callback registered (total: {len(self.callbacks)})")
        
    def get_link_quality(self) -> Dict:
        """Get LoRa link quality metrics"""
        return {
            'rssi': self.last_rssi,  # dBm
            'snr': self.last_snr,    # dB
            'frequency': self.frequency,
            'spreading_factor': self.spreading_factor,
            'bandwidth': self.bandwidth,
            'packets_sent': self.packets_sent,
            'packets_received': self.packets_received,
            'packet_loss': 0.0 if self.packets_sent == 0 else 
                          max(0, 1 - self.packets_received / self.packets_sent)
        }
        
    def set_spreading_factor(self, sf: int):
        """
        Set spreading factor (7-12)
        Higher SF = longer range, slower speed
        """
        if 7 <= sf <= 12:
            self.spreading_factor = sf
            print(f"[LORA] Spreading factor set to {sf}")
        else:
            print(f"[LORA] Invalid SF: {sf} (must be 7-12)")
            
    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        data = secrets.token_bytes(32)
        return hashlib.sha256(data).hexdigest()
        
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        data = f"{self.node_id}{time.time()}{secrets.token_hex(8)}".encode()
        return hashlib.sha256(data).hexdigest()[:16]
        
    def _encrypt_payload(self, payload: bytes) -> bytes:
        """Encrypt payload using AES-256"""
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend
            
            # LoRa network key
            key = hashlib.sha256(b"RF-ARSENAL-LORA-KEY").digest()
            iv = secrets.token_bytes(16)
            
            # Pad
            padding_length = 16 - (len(payload) % 16)
            padded_payload = payload + bytes([padding_length] * padding_length)
            
            # Encrypt
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(padded_payload) + encryptor.finalize()
            
            return iv + ciphertext
            
        except ImportError:
            return payload
            
    def _decrypt_payload(self, encrypted_payload: bytes) -> bytes:
        """Decrypt payload"""
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend
            
            iv = encrypted_payload[:16]
            ciphertext = encrypted_payload[16:]
            
            key = hashlib.sha256(b"RF-ARSENAL-LORA-KEY").digest()
            
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            padded_payload = decryptor.update(ciphertext) + decryptor.finalize()
            
            padding_length = padded_payload[-1]
            return padded_payload[:-padding_length]
            
        except ImportError:
            return encrypted_payload
            
    def _serialize_message(self, message: MeshMessage) -> bytes:
        """Serialize message for transmission"""
        message_dict = {
            'mid': message.message_id,
            'src': message.source_id,
            'dst': message.destination_id,
            'pay': message.payload.hex(),
            'pri': message.priority.value,
            'ts': int(message.timestamp),
            'ttl': message.ttl
        }
        
        return json.dumps(message_dict).encode()
        
    def _deserialize_message(self, data: bytes) -> MeshMessage:
        """Deserialize received message"""
        message_dict = json.loads(data.decode())
        
        return MeshMessage(
            message_id=message_dict['mid'],
            source_id=message_dict['src'],
            destination_id=message_dict['dst'],
            payload=bytes.fromhex(message_dict['pay']),
            priority=MessagePriority(message_dict['pri']),
            timestamp=message_dict['ts'],
            ttl=message_dict['ttl'],
            encrypted=True,
            route=[]
        )
        
    def stop_network(self):
        """Stop LoRaWAN network"""
        print("\n[LORA] Stopping network...")
        self.running = False
        time.sleep(1)
        print("[LORA] ✓ Network stopped\n")


class MeshNetworkManager:
    """
    Unified mesh network manager
    Manages multiple network types with automatic failover
    Provides unified API for mesh communications
    """
    
    def __init__(self):
        self.networks = {}
        self.active_network = None
        self.auto_failover = True
        self.message_log = []
        
    def start_all_networks(self) -> Dict[str, bool]:
        """Start all available mesh networks"""
        print("\n" + "="*70)
        print("MESH NETWORK MANAGER - STARTING ALL NETWORKS")
        print("="*70)
        
        results = {}
        
        # Start BLE mesh
        print("\n[1/2] Initializing BLE Mesh...")
        ble_mesh = BLEMeshNetwork()
        results['ble_mesh'] = ble_mesh.start_mesh()
        if results['ble_mesh']:
            self.networks[NetworkType.BLE_MESH] = ble_mesh
            
        # Start LoRaWAN
        print("\n[2/2] Initializing LoRaWAN...")
        lora_network = LoRaWANNetwork()
        results['lora_wan'] = lora_network.start_network()
        if results['lora_wan']:
            self.networks[NetworkType.LORA_WAN] = lora_network
            
        # Set primary network (prefer BLE for speed, LoRa for range)
        if NetworkType.BLE_MESH in self.networks:
            self.active_network = NetworkType.BLE_MESH
        elif NetworkType.LORA_WAN in self.networks:
            self.active_network = NetworkType.LORA_WAN
            
        print("\n" + "="*70)
        print("NETWORK INITIALIZATION COMPLETE")
        print("="*70)
        print(f"  Active networks: {[nt.value for nt in self.networks.keys()]}")
        print(f"  Primary network: {self.active_network.value if self.active_network else 'None'}")
        print(f"  Auto-failover: {'Enabled' if self.auto_failover else 'Disabled'}")
        print("="*70 + "\n")
        
        return results
        
    def send_message(self, destination: str, payload: bytes,
                    priority: MessagePriority = MessagePriority.NORMAL,
                    network_type: Optional[NetworkType] = None) -> bool:
        """
        Send message using specified or active network
        Automatic failover if primary fails
        
        Args:
            destination: Destination node ID or 'broadcast'
            payload: Message data
            priority: Message priority level
            network_type: Specific network to use (None = use active)
            
        Returns:
            True if message sent successfully
        """
        # Use specified network or active network
        target_network = network_type if network_type else self.active_network
        
        if not target_network or target_network not in self.networks:
            print("[MESH] ✗ Error: No active network available")
            return False
            
        network = self.networks[target_network]
        
        # Try to send
        print(f"\n[MESH] Sending via {target_network.value}...")
        
        if hasattr(network, 'send_message'):
            success = network.send_message(destination, payload, priority)
            
            # Log message
            self.message_log.append({
                'timestamp': time.time(),
                'network': target_network.value,
                'destination': destination,
                'size': len(payload),
                'success': success
            })
            
            if not success and self.auto_failover and not network_type:
                print("[MESH] Primary network failed, attempting failover...")
                return self._failover_send(destination, payload, priority)
                
            return success
            
        return False
        
    def _failover_send(self, destination: str, payload: bytes,
                      priority: MessagePriority) -> bool:
        """Attempt to send using alternative networks"""
        for network_type, network in self.networks.items():
            if network_type == self.active_network:
                continue  # Skip primary (already failed)
                
            print(f"[MESH] Trying failover: {network_type.value}...")
            
            if hasattr(network, 'send_message'):
                success = network.send_message(destination, payload, priority)
                if success:
                    print(f"[MESH] ✓ Failover successful via {network_type.value}")
                    return True
                    
        print("[MESH] ✗ All networks failed")
        return False
        
    def broadcast_message(self, payload: bytes,
                         priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """Broadcast message to all nodes on all networks"""
        return self.send_message('broadcast', payload, priority)
        
    def get_network_status(self) -> Dict:
        """Get detailed status of all networks"""
        status = {
            'active_network': self.active_network.value if self.active_network else None,
            'auto_failover': self.auto_failover,
            'message_count': len(self.message_log),
            'networks': {}
        }
        
        for network_type, network in self.networks.items():
            network_status = {
                'type': network_type.value,
                'running': network.running if hasattr(network, 'running') else False
            }
            
            # Add type-specific information
            if network_type == NetworkType.BLE_MESH:
                network_status['peer_count'] = network.get_peer_count()
                network_status['peers'] = network.get_peers()
                network_status['max_peers'] = network.max_peers
                network_status['tx_power'] = network.ble_tx_power
                
            elif network_type == NetworkType.LORA_WAN:
                link_quality = network.get_link_quality()
                network_status.update(link_quality)
                network_status['max_payload'] = network._get_max_payload_size()
                
            status['networks'][network_type.value] = network_status
            
        return status
        
    def switch_network(self, network_type: NetworkType):
        """Manually switch primary network"""
        if network_type in self.networks:
            old_network = self.active_network
            self.active_network = network_type
            print(f"[MESH] Switched from {old_network.value if old_network else 'None'} to {network_type.value}")
        else:
            print(f"[MESH] ✗ Error: {network_type.value} not available")
            
    def register_global_callback(self, callback: Callable):
        """Register callback for all networks"""
        count = 0
        for network in self.networks.values():
            if hasattr(network, 'register_callback'):
                network.register_callback(callback)
                count += 1
                
        print(f"[MESH] Callback registered on {count} networks")
        
    def stop_all_networks(self):
        """Stop all mesh networks gracefully"""
        print("\n" + "="*70)
        print("MESH NETWORK MANAGER - STOPPING ALL NETWORKS")
        print("="*70)
        
        for network_type, network in self.networks.items():
            print(f"\nStopping {network_type.value}...")
            if hasattr(network, 'stop_mesh'):
                network.stop_mesh()
            elif hasattr(network, 'stop_network'):
                network.stop_network()
                
        self.networks = {}
        self.active_network = None
        
        print("="*70)
        print("ALL NETWORKS STOPPED")
        print("="*70 + "\n")


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*70)
    print("MESH NETWORKING SYSTEM TEST")
    print("RF Arsenal OS - Offline Communications")
    print("="*70)
    
    # Test BLE mesh
    print("\n" + "─"*70)
    print("TEST 1: BLE Mesh Network")
    print("─"*70)
    
    ble_mesh = BLEMeshNetwork(node_name="test-node-1")
    ble_mesh.start_mesh()
    
    # Register message callback
    def message_callback(message: MeshMessage):
        print(f"\n[CALLBACK] Message received!")
        print(f"  From: {message.source_id[:8]}...")
        print(f"  Size: {len(message.payload)} bytes")
        print(f"  Priority: {message.priority.name}")
        print(f"  Content: {message.payload.decode() if len(message.payload) < 100 else '(binary data)'}")
        
    ble_mesh.register_callback(message_callback)
    
    # Display peers
    print(f"\nPeer count: {ble_mesh.get_peer_count()}")
    peers = ble_mesh.get_peers()
    if peers:
        print("Connected peers:")
        for peer in peers:
            print(f"  - {peer['name']} ({peer['id']}) RSSI={peer['rssi']}dBm")
    else:
        print("  (No peers discovered yet)")
    
    # Test LoRaWAN
    print("\n" + "─"*70)
    print("TEST 2: LoRaWAN Network")
    print("─"*70)
    
    lora = LoRaWANNetwork(node_name="test-lora-1")
    lora.start_network()
    
    lora.register_callback(message_callback)
    
    link_quality = lora.get_link_quality()
    print("\nLink quality metrics:")
    print(f"  RSSI: {link_quality['rssi']} dBm")
    print(f"  SNR: {link_quality['snr']} dB")
    print(f"  Frequency: {link_quality['frequency']/1e6:.1f} MHz")
    print(f"  SF: {link_quality['spreading_factor']}")
    print(f"  Packets sent: {link_quality['packets_sent']}")
    print(f"  Packets received: {link_quality['packets_received']}")
    
    # Test unified manager
    print("\n" + "─"*70)
    print("TEST 3: Mesh Network Manager")
    print("─"*70)
    
    manager = MeshNetworkManager()
    results = manager.start_all_networks()
    
    print("\nStartup results:")
    for network, success in results.items():
        status = '✓' if success else '✗'
        print(f"  {status} {network}")
        
    # Get detailed status
    status = manager.get_network_status()
    print(f"\nActive network: {status['active_network']}")
    print(f"Available networks: {list(status['networks'].keys())}")
    print(f"Auto-failover: {status['auto_failover']}")
    
    # Test message sending
    print("\n" + "─"*70)
    print("TEST 4: Sending Test Messages")
    print("─"*70)
    
    test_payload = b"This is a test message via mesh network. Operational communications enabled."
    
    print("\nSending via primary network...")
    success = manager.send_message('broadcast', test_payload, MessagePriority.HIGH)
    print(f"Result: {'✓ Success' if success else '✗ Failed'}")
    
    print("\nSending via LoRa (if available)...")
    if NetworkType.LORA_WAN in manager.networks:
        success = manager.send_message('broadcast', b"LoRa test message", 
                                      MessagePriority.NORMAL,
                                      network_type=NetworkType.LORA_WAN)
        print(f"Result: {'✓ Success' if success else '✗ Failed'}")
    
    print("\n" + "="*70)
    print("MESH NETWORKING SYSTEM TEST COMPLETE")
    print("="*70 + "\n")
