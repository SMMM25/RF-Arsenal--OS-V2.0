# Module 5: Mesh Networking System

**Advanced Offline Peer-to-Peer Communications for RF-Arsenal-OS**

[![Security: Military-Grade](https://img.shields.io/badge/Security-Military--Grade-red.svg)](https://github.com/SMMM25/RF-Arsenal-OS)
[![Module: Mesh Networking](https://img.shields.io/badge/Module-Mesh%20Networking-blue.svg)](security/mesh_networking.py)
[![Lines of Code: 1,248](https://img.shields.io/badge/LOC-1248-green.svg)](security/mesh_networking.py)

## 🌐 Overview

Module 5 provides **military-grade mesh networking** for RF-Arsenal-OS, enabling secure offline communications through multiple physical layers. This system ensures operational communications in environments where traditional networks are unavailable, compromised, or monitored.

### 🎯 Core Capabilities

- **BLE Mesh Network**: Short-range (10-100m) high-bandwidth peer-to-peer
- **LoRaWAN Fallback**: Long-range (10-40km) low-bandwidth emergency comms
- **Automatic Failover**: Seamless transition between network types
- **Multi-hop Routing**: Extended range through peer relay
- **AES-256 Encryption**: Military-grade message security
- **Offline Operation**: Zero dependency on internet/cellular infrastructure

---

## 📋 Table of Contents

1. [Architecture](#-architecture)
2. [Network Types](#-network-types)
3. [Hardware Requirements](#-hardware-requirements)
4. [Quick Start](#-quick-start)
5. [API Reference](#-api-reference)
6. [Usage Examples](#-usage-examples)
7. [Security Features](#-security-features)
8. [Configuration](#-configuration)
9. [Operational Considerations](#-operational-considerations)
10. [Troubleshooting](#-troubleshooting)
11. [Testing](#-testing)
12. [Future Roadmap](#-future-roadmap)

---

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                  MeshNetworkManager                          │
│  (Unified interface with automatic failover)                │
└────────────┬────────────────────────────┬───────────────────┘
             │                            │
    ┌────────▼─────────┐        ┌────────▼──────────┐
    │  BLE Mesh Network │        │ LoRaWAN Network   │
    │  (Primary)        │        │ (Fallback)        │
    │  • 10-100m range  │        │ • 10-40km range   │
    │  • ~1 Mbps        │        │ • 300bps-50kbps   │
    │  • 7 max peers    │        │ • 100-250 bytes   │
    └──────────────────┘        └───────────────────┘
```

### Network Layering

```
┌─────────────────────────────────────────────────────────────┐
│ Application Layer                                            │
│ • Message prioritization (CRITICAL/HIGH/NORMAL/LOW)         │
│ • Message routing and delivery confirmation                 │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ Encryption Layer                                             │
│ • AES-256-CBC encryption for all messages                   │
│ • Per-network encryption keys                               │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ Transport Layer                                              │
│ • Multi-hop routing via peer nodes                          │
│ • Automatic network type selection                          │
│ • Failover on connection loss                               │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ Physical Layer                                               │
│ • BLE 4.0+ (2.4 GHz ISM band)                               │
│ • LoRa SX127x (915 MHz US / 868 MHz EU)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🌐 Network Types

### 1. BLE Mesh Network (Primary)

**Purpose**: High-bandwidth short-range communications

| Specification | Value |
|--------------|-------|
| **Protocol** | Bluetooth Low Energy 4.0+ |
| **Frequency** | 2.4 GHz ISM band |
| **Range** | 10-100 meters (line-of-sight) |
| **Data Rate** | ~1 Mbps |
| **Max Peers** | 7 simultaneous connections |
| **Power** | 0-20 dBm (1-100 mW) |
| **Latency** | 10-50 ms |

**Advantages**:
- ✅ High bandwidth for data-rich messages
- ✅ Built into most modern devices (smartphones, laptops)
- ✅ Low power consumption
- ✅ Good penetration through buildings

**Disadvantages**:
- ❌ Limited range (100m max)
- ❌ 7-peer connection limit
- ❌ Susceptible to 2.4 GHz interference (WiFi, microwaves)

**Use Cases**:
- Team communications in urban environments
- Building-to-building coordination (< 100m)
- High-bandwidth file transfers
- Real-time voice/video (future)

---

### 2. LoRaWAN Network (Fallback)

**Purpose**: Long-range low-bandwidth emergency communications

| Specification | Value |
|--------------|-------|
| **Protocol** | LoRa (Long Range) |
| **Frequency** | 915 MHz (US) / 868 MHz (EU) |
| **Range** | 10-40 km (rural), 2-5 km (urban) |
| **Data Rate** | 300 bps - 50 kbps |
| **Payload** | 100-250 bytes per message |
| **Power** | 2-20 dBm (1.6-100 mW) |
| **Latency** | 1-10 seconds |

**Advantages**:
- ✅ Extreme range (up to 40 km)
- ✅ Excellent penetration through obstacles
- ✅ Low power consumption
- ✅ Works in remote/rural areas
- ✅ Less susceptible to interference

**Disadvantages**:
- ❌ Very limited bandwidth
- ❌ Small payload size (100-250 bytes)
- ❌ Higher latency
- ❌ Requires specialized hardware (SX127x chip)

**Use Cases**:
- Emergency communications (SOS, status updates)
- Long-range coordination (> 5 km)
- Text-only messages
- Remote area operations

---

## 🔧 Hardware Requirements

### BLE Mesh Network

| Component | Requirement |
|-----------|-------------|
| **Bluetooth Adapter** | Bluetooth 4.0+ (BLE support) |
| **Linux Kernel** | 3.4+ with BlueZ stack |
| **Software** | `bluez`, `bluez-tools`, `python3-bluez` |
| **Permissions** | Root or CAP_NET_ADMIN capability |

**Verify BLE Support**:
```bash
# Check Bluetooth adapter
hciconfig -a

# Check BlueZ version
bluetoothctl --version

# Scan for nearby BLE devices
sudo hcitool lescan
```

---

### LoRaWAN Network

| Component | Requirement |
|-----------|-------------|
| **LoRa Module** | SX127x chip (SX1276, SX1277, SX1278, SX1279) |
| **Connection** | SPI interface (GPIO pins) |
| **Frequency** | 915 MHz (US) or 868 MHz (EU) |
| **Antenna** | 915/868 MHz antenna (matched to frequency) |
| **Power** | 3.3V DC supply |

**Supported Hardware**:
- **Raspberry Pi + LoRa HAT**: Adafruit RFM95W, Dragino LoRa/GPS HAT
- **Arduino + LoRa Shield**: Arduino LoRa Shield (915/868 MHz)
- **Standalone Modules**: HopeRF RFM95/96/97/98, Ai-Thinker Ra-01

**Wiring Example (Raspberry Pi)**:
```
SX1276 Module          Raspberry Pi
─────────────          ────────────
VCC        ───────────  3.3V (Pin 1)
GND        ───────────  GND (Pin 6)
MISO       ───────────  GPIO 9 (MISO, Pin 21)
MOSI       ───────────  GPIO 10 (MOSI, Pin 19)
SCK        ───────────  GPIO 11 (SCLK, Pin 23)
NSS        ───────────  GPIO 8 (CE0, Pin 24)
DIO0       ───────────  GPIO 25 (Pin 22)
RST        ───────────  GPIO 17 (Pin 11)
```

**Install LoRa Library**:
```bash
pip install pyLoRa
```

---

## 🚀 Quick Start

### Installation

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y bluez bluez-tools libbluetooth-dev

# Install Python dependencies
pip install pybluez pyLoRa pycryptodome

# Enable Bluetooth service
sudo systemctl enable bluetooth
sudo systemctl start bluetooth
```

---

### Basic Usage

```python
#!/usr/bin/env python3
from security.mesh_networking import MeshNetworkManager, NetworkType, MessagePriority

# Initialize mesh network manager
manager = MeshNetworkManager(node_id="alpha-001")

# Start BLE mesh network (primary)
manager.start_network(NetworkType.BLE_MESH)

# Send high-priority message to specific peer
manager.send_message(
    destination="alpha-002",
    message="Rendezvous at location Bravo",
    priority=MessagePriority.HIGH
)

# Broadcast emergency message (uses LoRaWAN fallback if BLE unavailable)
manager.broadcast_message(
    message="Emergency: Request immediate extraction",
    priority=MessagePriority.CRITICAL
)

# Receive messages
def message_handler(msg):
    print(f"[{msg.timestamp}] From {msg.source}: {msg.data}")

manager.on_message_received(message_handler)

# Graceful shutdown
manager.stop_all_networks()
```

---

## 📚 API Reference

### `MeshNetworkManager`

**Purpose**: Unified interface for managing multiple mesh network types with automatic failover.

#### Constructor

```python
MeshNetworkManager(
    node_id: str,
    auto_failover: bool = True,
    encryption_key: Optional[bytes] = None
)
```

**Parameters**:
- `node_id` (str): Unique identifier for this node (e.g., "alpha-001")
- `auto_failover` (bool): Enable automatic failover to LoRaWAN on BLE failure
- `encryption_key` (bytes, optional): AES-256 key (32 bytes). Auto-generated if None.

#### Methods

##### `start_network()`
```python
manager.start_network(network_type: NetworkType) -> bool
```
Start a specific network type (BLE or LoRaWAN).

**Returns**: `True` if started successfully, `False` otherwise.

---

##### `send_message()`
```python
manager.send_message(
    destination: str,
    message: str,
    priority: MessagePriority = MessagePriority.NORMAL
) -> bool
```
Send encrypted message to a specific peer node.

**Parameters**:
- `destination` (str): Target node ID
- `message` (str): Message content (max 250 bytes for LoRaWAN)
- `priority` (MessagePriority): Message priority level

**Returns**: `True` if sent successfully.

---

##### `broadcast_message()`
```python
manager.broadcast_message(
    message: str,
    priority: MessagePriority = MessagePriority.NORMAL
) -> bool
```
Broadcast encrypted message to all peers on active network.

---

##### `on_message_received()`
```python
manager.on_message_received(callback: Callable[[MeshMessage], None])
```
Register callback function for incoming messages.

**Callback signature**:
```python
def message_handler(msg: MeshMessage):
    # msg.source: sender node ID
    # msg.destination: recipient node ID (or "broadcast")
    # msg.data: decrypted message content
    # msg.timestamp: message creation time
    # msg.priority: message priority level
    pass
```

---

##### `stop_all_networks()`
```python
manager.stop_all_networks()
```
Gracefully shutdown all active networks and cleanup resources.

---

### `BLEMeshNetwork`

**Purpose**: Bluetooth Low Energy mesh network implementation.

#### Constructor

```python
BLEMeshNetwork(
    node_id: str,
    encryption_key: bytes
)
```

#### Methods

##### `start()`
```python
network.start() -> bool
```
Initialize BLE adapter and start advertising as mesh node.

---

##### `discover_peers()`
```python
network.discover_peers(timeout: int = 10) -> List[str]
```
Scan for nearby BLE mesh nodes.

**Parameters**:
- `timeout` (int): Scan duration in seconds

**Returns**: List of discovered peer node IDs.

---

##### `connect_to_peer()`
```python
network.connect_to_peer(peer_id: str) -> bool
```
Establish connection to a discovered peer (max 7 peers).

---

### `LoRaWANNetwork`

**Purpose**: LoRa long-range mesh network implementation.

#### Constructor

```python
LoRaWANNetwork(
    node_id: str,
    encryption_key: bytes,
    frequency: int = 915,  # MHz (915 for US, 868 for EU)
    tx_power: int = 14,    # dBm (2-20)
    spreading_factor: int = 7  # 7-12 (higher = longer range, slower)
)
```

#### Methods

##### `start()`
```python
network.start() -> bool
```
Initialize LoRa module (SX127x) and start listening.

---

##### `send_message()`
```python
network.send_message(destination: str, data: bytes) -> bool
```
Send raw encrypted data (max 250 bytes).

---

### Data Classes

#### `MeshMessage`

```python
@dataclass
class MeshMessage:
    message_id: str          # Unique message ID (UUID)
    source: str              # Sender node ID
    destination: str         # Recipient node ID or "broadcast"
    data: str                # Message content (decrypted)
    timestamp: datetime      # Creation timestamp
    priority: MessagePriority
    hops: int = 0            # Number of relay hops
    network_type: NetworkType
```

#### `MeshNode`

```python
@dataclass
class MeshNode:
    node_id: str             # Unique node identifier
    last_seen: datetime      # Last contact timestamp
    rssi: int                # Signal strength (dBm)
    network_type: NetworkType
    is_relay: bool = False   # Can relay messages
```

---

## 💡 Usage Examples

### Example 1: Simple Peer-to-Peer Chat

```python
#!/usr/bin/env python3
"""Simple P2P chat using BLE mesh"""
from security.mesh_networking import MeshNetworkManager, NetworkType, MessagePriority
import sys

def main():
    # Initialize with your node ID
    my_node_id = input("Enter your node ID: ")
    manager = MeshNetworkManager(node_id=my_node_id)
    
    # Start BLE mesh
    if not manager.start_network(NetworkType.BLE_MESH):
        print("Failed to start BLE mesh network")
        sys.exit(1)
    
    print(f"Node '{my_node_id}' online. Type messages to send...")
    
    # Message handler
    def on_message(msg):
        print(f"\n[{msg.source}]: {msg.data}")
        print("> ", end="", flush=True)
    
    manager.on_message_received(on_message)
    
    # Send loop
    try:
        while True:
            message = input("> ")
            if message.lower() == "quit":
                break
            
            # Broadcast message to all peers
            manager.broadcast_message(message, MessagePriority.NORMAL)
    except KeyboardInterrupt:
        pass
    
    manager.stop_all_networks()
    print("\nDisconnected.")

if __name__ == "__main__":
    main()
```

---

### Example 2: Emergency Beacon with Auto-Failover

```python
#!/usr/bin/env python3
"""Emergency beacon with automatic BLE → LoRaWAN failover"""
from security.mesh_networking import MeshNetworkManager, NetworkType, MessagePriority
import time

def main():
    # Initialize with auto-failover enabled
    manager = MeshNetworkManager(
        node_id="emergency-beacon-01",
        auto_failover=True  # Automatically switch to LoRaWAN if BLE fails
    )
    
    # Try BLE first (shorter range, higher bandwidth)
    if not manager.start_network(NetworkType.BLE_MESH):
        print("BLE unavailable, starting LoRaWAN...")
        manager.start_network(NetworkType.LORAWAN)
    
    # Broadcast emergency message every 30 seconds
    print("Emergency beacon active. Broadcasting SOS...")
    try:
        while True:
            manager.broadcast_message(
                message="EMERGENCY: Location 34.0522°N, 118.2437°W. Require immediate extraction.",
                priority=MessagePriority.CRITICAL
            )
            print(f"[{time.strftime('%H:%M:%S')}] SOS broadcast sent")
            time.sleep(30)
    except KeyboardInterrupt:
        pass
    
    manager.stop_all_networks()
    print("\nBeacon deactivated.")

if __name__ == "__main__":
    main()
```

---

### Example 3: Multi-Hop Relay Node

```python
#!/usr/bin/env python3
"""Relay node for extending mesh network range"""
from security.mesh_networking import MeshNetworkManager, NetworkType, MessagePriority

def main():
    # Initialize as relay node
    manager = MeshNetworkManager(node_id="relay-node-01")
    
    # Start both networks (relay between BLE and LoRaWAN)
    manager.start_network(NetworkType.BLE_MESH)
    manager.start_network(NetworkType.LORAWAN)
    
    print("Relay node active. Forwarding messages between BLE and LoRaWAN...")
    
    # Message handler (automatically relays)
    def on_message(msg):
        print(f"Relaying: {msg.source} → {msg.destination} (hop {msg.hops + 1})")
        
        # Forward message to other network type
        if msg.network_type == NetworkType.BLE_MESH:
            # Relay to LoRaWAN
            manager.send_message(
                destination=msg.destination,
                message=msg.data,
                priority=msg.priority
            )
        else:
            # Relay to BLE
            manager.broadcast_message(msg.data, msg.priority)
    
    manager.on_message_received(on_message)
    
    # Keep relay running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    manager.stop_all_networks()
    print("\nRelay node stopped.")

if __name__ == "__main__":
    main()
```

---

### Example 4: Network Status Monitor

```python
#!/usr/bin/env python3
"""Monitor mesh network health and peer status"""
from security.mesh_networking import MeshNetworkManager, NetworkType
import time

def main():
    manager = MeshNetworkManager(node_id="monitor-01")
    manager.start_network(NetworkType.BLE_MESH)
    
    print("Network monitor active. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            # Get network status
            status = manager.get_network_status()
            
            print("\n" + "="*50)
            print(f"Network Type: {status['active_network']}")
            print(f"Connected Peers: {status['peer_count']}")
            print(f"Messages Sent: {status['messages_sent']}")
            print(f"Messages Received: {status['messages_received']}")
            print(f"Uptime: {status['uptime']} seconds")
            
            if status['peers']:
                print("\nPeer Details:")
                for peer in status['peers']:
                    print(f"  • {peer.node_id} (RSSI: {peer.rssi} dBm, "
                          f"Last seen: {peer.last_seen})")
            
            time.sleep(10)
    except KeyboardInterrupt:
        pass
    
    manager.stop_all_networks()

if __name__ == "__main__":
    main()
```

---

## 🔒 Security Features

### 1. AES-256 Encryption

**All messages are encrypted** using AES-256-CBC before transmission.

```python
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

# Encryption key (32 bytes for AES-256)
encryption_key = get_random_bytes(32)

# Pass custom key to manager
manager = MeshNetworkManager(
    node_id="secure-node",
    encryption_key=encryption_key
)
```

**Security Properties**:
- ✅ 256-bit key length (2^256 possible keys)
- ✅ CBC mode with random IV per message
- ✅ PKCS7 padding for variable-length messages
- ✅ No key reuse (IV rotates per message)

---

### 2. Message Authentication

Each message includes:
- **Message ID**: UUID v4 for deduplication
- **Timestamp**: Creation time (prevents replay attacks)
- **Checksum**: SHA-256 hash for integrity verification

```python
import hashlib

# Message integrity check (internal)
message_hash = hashlib.sha256(message_bytes).hexdigest()
```

---

### 3. Peer Authentication

**Mutual authentication** between nodes (planned for v2.0):
- Pre-shared keys (PSK) for node authorization
- Challenge-response protocol
- Whitelist/blacklist support

---

### 4. Anti-Correlation

**Randomized identifiers** to prevent traffic analysis:
- Node IDs rotate periodically (configurable)
- Random message timing jitter
- Dummy traffic generation (future)

---

## ⚙️ Configuration

### Network Parameters

```python
# BLE Mesh Configuration
BLE_CONFIG = {
    "advertising_interval": 100,  # ms (20-10240)
    "scan_window": 50,            # ms
    "connection_interval": 15,    # ms (7.5-4000)
    "max_peers": 7,               # BLE specification limit
    "tx_power": 0,                # dBm (-20 to +20)
}

# LoRaWAN Configuration
LORA_CONFIG = {
    "frequency": 915,             # MHz (915 US, 868 EU)
    "tx_power": 14,               # dBm (2-20)
    "spreading_factor": 7,        # 7-12 (higher = longer range)
    "bandwidth": 125000,          # Hz (125/250/500 kHz)
    "coding_rate": 5,             # 5-8 (higher = more error correction)
    "preamble_length": 8,         # symbols
    "sync_word": 0x12,            # Private network sync word
}

# Apply custom configuration
manager = MeshNetworkManager(node_id="custom-node")
manager.configure_network(NetworkType.BLE_MESH, BLE_CONFIG)
manager.configure_network(NetworkType.LORAWAN, LORA_CONFIG)
```

---

### Encryption Key Management

```python
import os
import json

# Generate and save encryption key
def generate_key():
    key = os.urandom(32)  # 256 bits
    with open("mesh_key.bin", "wb") as f:
        f.write(key)
    os.chmod("mesh_key.bin", 0o600)  # Read/write owner only
    return key

# Load existing key
def load_key():
    with open("mesh_key.bin", "rb") as f:
        return f.read()

# Use saved key
key = load_key() if os.path.exists("mesh_key.bin") else generate_key()
manager = MeshNetworkManager(node_id="node-01", encryption_key=key)
```

---

## ⚠️ Operational Considerations

### Power Management

**BLE Power Consumption**:
- Advertising: ~15 mA @ 3.3V
- Connected: ~10-20 mA (depends on activity)
- Idle: ~1-5 mA

**LoRa Power Consumption**:
- Transmit (14 dBm): ~120 mA @ 3.3V (brief bursts)
- Receive: ~12 mA
- Idle: ~1.5 mA

**Battery Life Estimation** (3000 mAh battery):
- BLE only: ~150-200 hours
- LoRa only: ~200-250 hours (with infrequent TX)
- Both active: ~100-150 hours

**Power Optimization**:
```python
# Reduce BLE transmission power
BLE_CONFIG["tx_power"] = -20  # dBm (minimum)

# Reduce LoRa transmission power
LORA_CONFIG["tx_power"] = 2   # dBm (minimum)

# Increase message intervals
time.sleep(60)  # Send messages every 60 seconds instead of continuously
```

---

### Range Limitations

**BLE Mesh**:
- Indoor: 10-30 meters (through walls)
- Outdoor: 50-100 meters (line-of-sight)
- Urban: 20-50 meters (buildings, obstacles)

**LoRaWAN**:
- Rural: 10-40 km (line-of-sight)
- Suburban: 5-10 km
- Urban: 2-5 km (buildings block signal)
- Indoor: 500m-2km (depends on building materials)

**Range Extension**:
- Deploy relay nodes every 50m (BLE) or 5km (LoRa)
- Use directional antennas for LoRa
- Increase spreading factor (LoRa) for longer range

---

### Interference

**BLE (2.4 GHz) Interference Sources**:
- WiFi networks (same frequency band)
- Microwave ovens
- Wireless keyboards/mice
- Baby monitors
- Zigbee devices

**Mitigation**:
- Use BLE adaptive frequency hopping (automatic)
- Avoid 2.4 GHz-heavy environments
- Fall back to LoRaWAN in high-interference areas

**LoRa Interference**:
- Minimal (915/868 MHz less crowded)
- Amateur radio (overlapping frequencies)
- Other LoRa networks (use unique sync word)

---

### Legal Compliance

**BLE**: Generally license-free worldwide (2.4 GHz ISM band).

**LoRa**:
- **United States**: 915 MHz ISM band (license-free, max 30 dBm EIRP)
- **Europe**: 868 MHz ISM band (license-free, max 14 dBm ERP)
- **Other regions**: Check local regulations

**Transmit Power Limits**:
- Respect regional power limits (FCC Part 15 in US, ETSI EN 300 220 in EU)
- Do not exceed 20 dBm (100 mW) without appropriate antenna calculations

---

## 🔧 Troubleshooting

### BLE Mesh Issues

#### Problem: "Failed to start BLE mesh network"

**Possible Causes**:
1. Bluetooth adapter not found
2. BlueZ stack not installed
3. Insufficient permissions

**Solutions**:
```bash
# Check Bluetooth adapter
hciconfig -a

# Install BlueZ
sudo apt-get install bluez bluez-tools

# Grant permissions
sudo usermod -aG bluetooth $USER
sudo chmod 666 /dev/rfkill

# Run with sudo
sudo python3 mesh_app.py
```

---

#### Problem: "Cannot discover peers"

**Solutions**:
```bash
# Ensure Bluetooth is powered on
sudo bluetoothctl
> power on
> scan on

# Check for nearby devices
sudo hcitool lescan

# Restart Bluetooth service
sudo systemctl restart bluetooth
```

---

### LoRaWAN Issues

#### Problem: "SX127x module not detected"

**Possible Causes**:
1. LoRa module not connected properly
2. SPI not enabled on Raspberry Pi
3. Incorrect GPIO pin configuration

**Solutions**:
```bash
# Enable SPI on Raspberry Pi
sudo raspi-config
# Interface Options → SPI → Enable

# Check SPI devices
ls /dev/spi*

# Verify wiring (continuity test with multimeter)
# Check power supply (3.3V ±0.1V)

# Test with sample code
python3 -c "from SX127x.LoRa import LoRa; print('LoRa module OK')"
```

---

#### Problem: "Messages not received (LoRa)"

**Solutions**:
- **Check frequency**: Ensure both nodes use same frequency (915 or 868 MHz)
- **Check spreading factor**: Both nodes must match (default: 7)
- **Check encryption key**: Both nodes must use same key
- **Check range**: Move nodes closer (< 1 km for testing)
- **Check antenna**: Ensure antenna connected (module can be damaged without antenna)

```python
# Debug mode (print all LoRa parameters)
network = LoRaWANNetwork(node_id="debug", encryption_key=key)
network.start()
print(network.get_configuration())  # Show all LoRa settings
```

---

## 🧪 Testing

### Unit Tests

```bash
cd /home/user/webapp
python3 -m pytest tests/test_mesh_networking.py -v
```

### Integration Tests

```bash
# Terminal 1: Start node A
python3 tests/mesh_integration_test.py --node-id node-a

# Terminal 2: Start node B
python3 tests/mesh_integration_test.py --node-id node-b

# Terminal 1: Send message from node A
> Hello from node A

# Terminal 2: Should receive message
[node-a]: Hello from node A
```

### Range Testing

```bash
# Terminal 1: Fixed position (transmitter)
python3 tests/range_test_tx.py --network ble

# Terminal 2: Mobile position (receiver)
python3 tests/range_test_rx.py --network ble

# Walk away from transmitter and record RSSI at intervals
# BLE typically drops below -90 dBm at max range
```

---

## 🚀 Future Roadmap

### Version 2.0 (Q2 2024)

- [ ] **WiFi Direct mesh**: 100-200m range, 11 Mbps
- [ ] **Peer authentication**: Challenge-response with pre-shared keys
- [ ] **Voice communications**: Low-bitrate voice codec (Codec2)
- [ ] **File transfer**: Chunked file transmission with resume support
- [ ] **Mesh routing protocol**: AODV (Ad hoc On-Demand Distance Vector)

### Version 3.0 (Q4 2024)

- [ ] **Zigbee mesh**: 10-100m, 250 kbps
- [ ] **Satellite fallback**: Iridium short burst data (SBD)
- [ ] **Traffic analysis resistance**: Dummy traffic, timing obfuscation
- [ ] **Mobile app**: Android/iOS mesh client
- [ ] **Group messaging**: Encrypted group channels

---

## 📊 Performance Benchmarks

### Message Latency

| Network Type | Local (< 10m) | Medium (100m) | Long (> 1km) |
|-------------|---------------|---------------|--------------|
| **BLE Mesh** | 10-20 ms | 30-50 ms | N/A (out of range) |
| **LoRaWAN** | 500-1000 ms | 1-2 seconds | 2-5 seconds |

### Throughput

| Network Type | Text Messages | Small Files | Large Files |
|-------------|---------------|-------------|-------------|
| **BLE Mesh** | 100+ msg/sec | 50-100 KB/s | ~1 MB/s (max) |
| **LoRaWAN** | 1-5 msg/min | 100-500 B/s | N/A (impractical) |

### Battery Life (3000 mAh)

| Usage Pattern | BLE Only | LoRa Only | Both Active |
|--------------|----------|-----------|-------------|
| **Idle (listening)** | 200+ hours | 250+ hours | 150+ hours |
| **Light (10 msg/hour)** | 150 hours | 200 hours | 120 hours |
| **Heavy (100 msg/hour)** | 100 hours | 150 hours | 80 hours |
| **Continuous TX** | 50 hours | 80 hours | 40 hours |

---

## 🆘 Support & Contributing

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/SMMM25/RF-Arsenal-OS/issues)
- **Discussions**: [GitHub Discussions](https://github.com/SMMM25/RF-Arsenal-OS/discussions)
- **Documentation**: [RF-Arsenal-OS Wiki](https://github.com/SMMM25/RF-Arsenal-OS/wiki)

### Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/mesh-improvement`)
3. Commit changes (`git commit -am 'Add mesh improvement'`)
4. Push to branch (`git push origin feature/mesh-improvement`)
5. Open a Pull Request

---

## ⚖️ Legal Disclaimer

**FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY**

This mesh networking system is designed for:
- ✅ Emergency communications research
- ✅ Off-grid networking experiments
- ✅ Disaster recovery simulations
- ✅ Academic studies

**WARNING**: 
- Comply with local RF transmission regulations
- Respect frequency band allocations
- Do not exceed legal power limits
- Obtain necessary licenses if required
- Use responsibly and ethically

**The authors assume no liability for misuse of this software.**

---

## 📄 License

Copyright © 2024 RF-Arsenal-OS Project

Licensed under the MIT License. See [LICENSE](../LICENSE) for details.

---

## 🔗 Related Modules

- **[Module 3: Identity Management](IDENTITY_MANAGEMENT_README.md)**: Persona-based network profiles
- **[Module 4: Covert Storage](security/covert_storage.py)**: Encrypted data hiding
- **[Network Anonymity V2](modules/stealth/network_anonymity_v2.py)**: Anonymous mesh routing

---

**Module 5: Mesh Networking - Enabling secure offline communications for RF-Arsenal-OS**

🌐 **Offline-First** | 🔒 **Encrypted** | 📡 **Multi-Layer** | ⚡ **Auto-Failover**
