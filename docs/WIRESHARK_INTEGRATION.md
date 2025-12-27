# RF Arsenal OS - Wireshark Integration Guide

**Version**: 1.0.0  
**Integration Date**: 2024-12-20  
**Status**: Production Ready

---

## 📋 OVERVIEW

RF Arsenal OS now includes full Wireshark/TShark integration for comprehensive packet capture and network analysis, seamlessly controlled through the AI interface.

### Key Features

✅ **AI-Controlled Packet Capture**  
✅ **Live Network Monitoring**  
✅ **PCAP File Analysis**  
✅ **Protocol Filtering**  
✅ **DNS Leak Detection**  
✅ **Credential Extraction**  
✅ **Anti-Forensics Integration**  
✅ **Emergency Cleanup**

---

## 🚀 QUICK START

### 1. Verify Installation

```bash
# Check TShark
tshark --version

# Check PyShark
python3 -c "import pyshark; print('PyShark OK')"

# List interfaces
ip link show
```

### 2. Launch RF Arsenal OS

```bash
# GUI Mode
sudo python3 rf_arsenal_os.py

# CLI Mode (AI Interface)
sudo python3 rf_arsenal_os.py --cli
```

---

## 🤖 AI COMMANDS

The AI controller understands natural language commands for packet capture:

### Basic Capture Commands

```
# Start packet capture
capture packets on wlan0
sniff packets on eth0
start packet capture

# Stop capture
stop capture
end packet capture

# Analyze captured packets
analyze packets
check capture
examine pcap
```

### Advanced Commands

```
# Capture with filter
capture packets on wlan0 filter tcp port 80
sniff http traffic on eth0
capture dns queries

# DNS leak detection
check for dns leaks
detect dns leaks
scan for network leaks

# Cleanup
clean up captures
wipe packet captures
delete capture files
```

### Example Session

```
[RF-Arsenal]> capture packets on wlan0
📡 Packet capture started on wlan0

[RF-Arsenal]> check for dns leaks
✅ No DNS leaks detected

[RF-Arsenal]> stop capture
✅ Capture stopped - 1247 packets captured

[RF-Arsenal]> analyze packets
📊 Analysis: 1247 packets, Protocols: ['TCP', 'UDP', 'DNS', 'HTTP', 'TLS']

[RF-Arsenal]> clean up captures
🧹 Capture files cleaned up
```

---

## 💻 PYTHON API USAGE

### Basic Packet Capture

```python
from modules.network.packet_capture import WiresharkCapture

# Initialize
pcap = WiresharkCapture()

# Start capture
pcap.start_capture(
    duration=60,              # 60 seconds
    bpf_filter='tcp port 443' # HTTPS only
)

# Wait for capture to complete
import time
time.sleep(61)

# Stop capture
result = pcap.stop_capture()
print(f"Captured {result['packets_captured']} packets")
```

### PCAP Analysis

```python
# Analyze captured file
result = pcap.analyze_pcap('/tmp/rf_arsenal_captures/capture_20241220.pcap')

if result['success']:
    stats = result['statistics']
    print(f"Total packets: {stats['total_packets']}")
    print(f"Protocols: {stats['protocols']}")
```

### DNS Leak Detection

```python
# Check for DNS leaks
leaks = pcap.check_leaks()

if leaks['dns_leaks'] > 0:
    print(f"⚠️ Found {leaks['dns_leaks']} DNS leaks")
    for query in leaks['dns_queries'][:5]:
        print(f"  - {query}")
else:
    print("✅ No DNS leaks detected")
```

### Secure Cleanup

```python
# Secure cleanup with shred
result = pcap.cleanup(secure_delete=True)
print(f"Deleted: {len(result['deleted'])} files")
```

---

## 🔒 SECURITY FEATURES

### 1. Anti-Forensics Integration

Packet captures are automatically integrated with the anti-forensics module:

```python
# Captures are auto-registered for secure deletion
# Emergency shutdown triggers secure wipe

# Manual secure cleanup
from security.anti_forensics import EncryptedRAMOverlay
ram = EncryptedRAMOverlay()
ram.secure_wipe_captures()
```

### 2. Emergency Protocols

Packet capture integrates with emergency shutdown:

```python
from core.emergency import EmergencySystem

emergency = EmergencySystem()
emergency.emergency_wipe("user_trigger")
# Automatically wipes all packet captures
```

### 3. No Network Exposure

✅ Passive capture only (no packets sent)  
✅ Local operation (no remote connections)  
✅ Zero impact on Tor/VPN anonymity  
✅ No interference with stealth features

---

## 🛡️ STEALTH CONSIDERATIONS

### Wireshark Does NOT Compromise:

| Security Feature | Impact | Status |
|-----------------|--------|--------|
| Tor Anonymity | None - Local capture only | ✅ Safe |
| MAC Randomization | None - Passive observation | ✅ Safe |
| RF Emission Masking | None - Network layer only | ✅ Safe |
| Network Stealth | None - No packets sent | ✅ Safe |
| Anti-Forensics | Enhanced - Auto cleanup | ✅ Enhanced |

### Actually Enhances Security By:

🔍 Detecting DNS leaks in Tor traffic  
🔍 Verifying VPN/proxy connections  
🔍 Monitoring for unexpected traffic  
🔍 Identifying security issues

---

## 📊 USE CASES

### 1. Verify Tor Anonymity

```bash
# Start Tor
systemctl start tor

# Capture traffic
[RF-Arsenal]> capture packets on eth0

# Browse with Tor browser
# ... browse websites ...

# Check for leaks
[RF-Arsenal]> check for dns leaks
✅ No DNS leaks detected
```

### 2. Monitor WiFi Attack Results

```bash
# Start WiFi deauth attack
[RF-Arsenal]> deauth attack on target_ap

# Capture resulting traffic
[RF-Arsenal]> capture packets on wlan0

# Analyze captured handshakes
[RF-Arsenal]> analyze packets
```

### 3. Post-Operation Forensics

```python
# After operation, analyze all captures
pcap.analyze_pcap('/tmp/operation_capture.pcap')

# Extract useful data
credentials = pcap.check_leaks()

# Secure wipe
pcap.cleanup(secure_delete=True)
```

---

## 🔧 ADVANCED FEATURES

### Custom BPF Filters

```python
# HTTP only
pcap.start_capture(bpf_filter='tcp port 80')

# Specific IP
pcap.start_capture(bpf_filter='host 192.168.1.1')

# DNS queries
pcap.start_capture(bpf_filter='udp port 53')

# Complex filter
pcap.start_capture(bpf_filter='tcp port 443 or tcp port 80')
```

---

## 📈 PERFORMANCE CONSIDERATIONS

### Resource Usage

| Operation | CPU | RAM | Disk I/O |
|-----------|-----|-----|----------|
| Live Capture | 5-10% | 50-100 MB | Medium |
| PCAP Analysis | 10-20% | 100-200 MB | High |

### Optimization Tips

- Use BPF Filters - Filter at capture time, not display time
- Limit Duration - Set reasonable time limits for captures
- Clean Up Regularly - Delete old captures to save disk space
- Use Secure Delete - Integrated with anti-forensics

---

## 🐛 TROUBLESHOOTING

### TShark Not Found

```bash
# Install Wireshark
sudo apt-get install wireshark tshark

# Verify
tshark --version
```

### PyShark Import Error

```bash
# Install PyShark
pip3 install pyshark

# Verify
python3 -c "import pyshark; print('OK')"
```

### Permission Denied

```bash
# Add user to wireshark group
sudo usermod -aG wireshark $USER

# Reboot or re-login
sudo reboot
```

### Interface Not Found

```bash
# List interfaces
ip link show

# Or use tshark
tshark -D
```

---

## 📚 WIRESHARK FILTER REFERENCE

### Common BPF Filters (Capture Time)

```
# Protocol
tcp
udp
icmp

# Port
port 80
portrange 1-1024
not port 22

# Host
host 192.168.1.1
src host 10.0.0.1
dst host 8.8.8.8

# Network
net 192.168.1.0/24

# Combined
tcp port 443 and host 192.168.1.1
not (port 22 or port 3389)
```

---

## ⚖️ LEGAL CONSIDERATIONS

### Authorized Use Only

Packet capture must comply with:

✅ **Authorization Required**
- Written permission to capture network traffic
- Compliance with organizational policies
- Adherence to applicable laws

✅ **Privacy Concerns**
- Do not capture personal information without consent
- Follow data protection regulations (GDPR, etc.)
- Secure storage and disposal of captures

✅ **Ethical Use**
- Only capture on networks you own or have authorization for
- Do not intercept communications without legal authority
- Respect privacy and confidentiality

---

## 🔗 INTEGRATION WITH RF ARSENAL MODULES

### WiFi + Packet Capture

```python
# Deauth attack + capture
[RF-Arsenal]> deauth attack on target_ap
[RF-Arsenal]> capture packets on wlan0
```

### Cellular + Packet Capture

```python
# 4G base station + capture
[RF-Arsenal]> start 4g base station
[RF-Arsenal]> capture packets on usb0
```

---

## 📞 SUPPORT

- **Documentation**: `/docs/WIRESHARK_INTEGRATION.md`
- **Issues**: https://github.com/SMMM25/RF-Arsenal-OS/issues
- **Module**: `modules/network/packet_capture.py`

---

## 📝 CHANGELOG

### v1.0.0 - 2024-12-20

✅ Initial Wireshark integration  
✅ PyShark wrapper implementation  
✅ AI controller commands  
✅ Anti-forensics integration  
✅ DNS leak detection  
✅ Credential extraction  
✅ Emergency cleanup protocols

**Status**: ✅ Production Ready  
**Security Impact**: None (Enhanced)  
**AI Integration**: Seamless  
**Documentation**: Complete

---

Built by white hats, for white hats. 🛡️
