# RF Arsenal OS - White Hat Edition

![License](https://img.shields.io/badge/license-Proprietary-red)
![Platform](https://img.shields.io/badge/platform-Cross--Platform-brightgreen)
![Hardware](https://img.shields.io/badge/hardware-BladeRF%20xA9-blue)
![Security](https://img.shields.io/badge/security-hardened-green)
![Status](https://img.shields.io/badge/status-production--grade-success)
![Version](https://img.shields.io/badge/version-4.1.0-blue)
![Tests](https://img.shields.io/badge/tests-486%20passing-brightgreen)
![LOC](https://img.shields.io/badge/lines%20of%20code-250%2C000%2B-blue)
![Security](https://img.shields.io/badge/security%20audit-passed-success)
![Audit](https://img.shields.io/badge/deep%20audit-December%202024-success)
![Production](https://img.shields.io/badge/production-ready-brightgreen)

**Production-grade RF security testing platform for authorized white hat penetration testing.**

---

## 🔧 December 2024 Production Readiness Update (v4.1.1)

**Comprehensive remediation completed across 40+ files:**

### ✅ Simulation Mode Removal (README Rule #5 Compliance)
All mock/simulation fallbacks have been removed. System now:
- **Raises `HardwareRequirementError`** when SDR hardware not detected
- **Raises `DependencyError`** when required software not installed
- **NO silent fallbacks** - clear error messages guide users to solutions

| Component | Previous Behavior | New Behavior |
|-----------|-------------------|--------------|
| BladeRF Driver | Simulated data when no hardware | `HardwareRequirementError` with installation guide |
| S1AP Protocol | Simulation mode without SCTP | `DependencyError: pip install pysctp` |
| YateBTS Controller | Mock mode without YateBTS | `DependencyError: ./install/install_yatebts.sh` |
| Physical Security | GPIO simulation | Feature disabled with warning |
| ADS-B Attacks | Simulated aircraft | Hardware required |
| LoRa Attacks | Mock gateway | Hardware required |
| NFC/Proxmark3 | Simulated card data | Hardware required |
| TEMPEST | Simulated EM capture | Hardware required |
| Power Analysis | Simulated traces | Hardware required |

### ✅ YateBTS Integration (NEW)
Full YateBTS GSM/LTE BTS integrated from official sources:
- **Source included**: `external/yatebts/` and `external/yate/`
- **Automated installer**: `sudo bash install/install_yatebts.sh --bladerf`
- **Python controller**: `modules/cellular/yatebts/yatebts_controller.py`
- **Supports**: GSM850/900/1800/1900, LTE B1/B3/B7
- **Features**: IMSI catching, SMS interception, voice interception, location tracking

### ✅ Test Suite Updates
- **485 tests passing** (357 unit + 153 integration - 21 external)
- External dependency tests skip gracefully with clear messages
- All test simulations renamed to `Development*` (not production simulation)

### ✅ Files Modified
```
core/hardware_interface.py      - HardwareRequirementError instead of NoHardwareFallback
core/hardware_controller.py     - Proper hardware detection
core/protocols/s1ap.py          - DependencyError for pysctp
core/protocols/gtp.py           - Hardware requirement enforcement
core/sdr/soapy_backend.py       - No simulation mode
core/fpga/stealth_fpga.py       - FPGA required
core/hardware/bladerf_driver.py - Real hardware only
security/physical_security.py   - GPIO feature disabled warning
modules/adsb/adsb_attacks.py    - Hardware required
modules/lora/lora_attack.py     - Hardware required
modules/nfc/proxmark3.py        - Hardware required
modules/tempest/tempest_attacks.py - Hardware required
modules/power_analysis/power_attacks.py - Hardware required
modules/cellular/yatebts/yatebts_controller.py - Full integration
install/install_yatebts.sh      - Automated YateBTS installation
install/verify_deployment.py    - Deployment verification script
```

---

## 🧠 Arsenal AI v3.0 - Conversational Attack Intelligence (NEW)

RF Arsenal OS now features **Arsenal AI v3.0**, a revolutionary conversational attack engine. No more memorizing 700+ commands - just tell it what you want to do in plain English:

```
"find vulnerable wifi networks nearby and crack them"
"scan that subnet and show me what's exploitable"
"test the API for authentication bypass"
"attack the same target again"
"what did you find?"
```

### Why Arsenal AI v3.0?

| Old Way (Commands) | New Way (Conversation) |
|--------------------|------------------------|
| `nmap -sV -p- 192.168.1.0/24` | "scan that network for open services" |
| `aircrack-ng capture.cap -w rockyou.txt` | "crack the wifi password" |
| Multiple commands for attack chains | "compromise the wifi and pivot to the network" |

### Key Features

| Feature | Description |
|---------|-------------|
| **🗣️ Natural Language** | Just describe what you want - AI figures out the rest |
| **🔗 Auto-Chain Attacks** | Multi-step operations execute automatically |
| **🧠 Context Memory** | Remembers targets, results, and can reference "that network" |
| **💡 Smart Recommendations** | Suggests next steps based on what it found |
| **🥷 Stealth-First** | All operations maintain operational security |
| **⚡ Zero Refusals** | Executes operator commands - you hold the authorization |

### Quick Start

```bash
# Interactive mode
python3 core/arsenal_cli.py

# Single command
python3 core/arsenal_cli.py "find wifi vulnerabilities"

# Stealth mode
python3 core/arsenal_cli.py --stealth "scan the target"
```

### Example Session

```
arsenal> find vulnerable wifi networks

✓ wifi_scan successful
  networks: 12 items
  
Discovered 12 target(s):
  1. [wifi_network] CorporateWiFi (WPA2, -45 dBm)
  2. [wifi_network] GuestNetwork (WPA2, -52 dBm)
  3. [wifi_network] IoT_Devices (WEP, -58 dBm) ← Vulnerable!

Suggestions:
  • WEP network found - crack the password?

arsenal> yes, crack that one

✓ wifi_capture successful
✓ wifi_crack successful
  password: Summer2024!
  
Next: wifi_connect, pivot_network

arsenal> connect and scan the internal network

✓ Connected to IoT_Devices
✓ net_scan successful
  hosts: 23 items
  
Suggestions:
  • 3 hosts have critical vulnerabilities - exploit them?
```

---

## 🤖 AI Command Center (Classic Mode)

The original AI Command Center is still available with 700+ commands for precise control:

| Feature | Description | Example Commands |
|---------|-------------|------------------|
| **🛡️ Counter-Surveillance** | Detect if you're being tracked | `scan for threats`, `detect imsi catchers` |
| **📊 RF Threat Dashboard** | Real-time signal visualization | `show dashboard`, `threat summary` |
| **🎯 Mission Profiles** | Pre-configured operation templates | `start wifi security mission` |
| **📈 OPSEC Scoring** | Live security posture (0-100) | `check opsec`, `fix opsec issues` |
| **👤 Skill-Based Modes** | Adaptive UI for beginners/experts | `set mode beginner` |
| **📻 Signal Replay** | Capture, analyze, replay RF signals | `capture signal`, `replay signal` |
| **🔧 Hardware Auto-Setup** | Plug-and-play SDR detection | `detect hardware`, `calibrate` |
| **🚗 Vehicle Pentesting** | CAN bus, UDS, key fob, TPMS, V2X | `scan can bus`, `read ecu dtc`, `capture key fob` |
| **📡 Satellite Communications** | NOAA, Iridium, ADS-B, GPS signals | `scan satellites`, `track noaa`, `scan adsb` |
| **🔍 RF Device Fingerprinting** | ML-based device identification | `fingerprint devices`, `identify transmitter` |
| **🏠 IoT/Smart Home Attacks** | Zigbee, Z-Wave, smart locks, meters | `scan zigbee`, `scan smart lock`, `iot status` |
| **📡 MIMO 2x2** | Spatial multiplexing, beamforming, DoA | `enable mimo`, `mimo beamform`, `mimo doa` |
| **🔓 Full-Duplex Relay** | Car key relay, access card relay, NFC | `relay car key`, `relay access card` |
| **📶 LoRa/LoRaWAN** | IoT infrastructure attacks | `scan lora`, `sniff lorawan`, `spoof gateway` |
| **📡 Meshtastic Mesh** | Mesh network SIGINT & attacks | `scan meshtastic`, `map mesh topology`, `meshtastic sigint` |
| **📱 Bluetooth 5.x** | BLE 5.0-5.3, direction finding, long range | `ble5 scan`, `direction finding`, `coded phy` |
| **🔑 RollJam** | Rolling code attacks for vehicles | `start rolljam`, `capture code`, `replay code` |
| **⚙️ AGC/Calibration** | Hardware AGC, DC offset, IQ calibration | `agc mode`, `calibrate rf`, `rssi monitor` |
| **📻 Frequency Hopping** | FHSS tracking, prediction, jamming | `start hopping`, `track hopper`, `predict sequence` |
| **📻 XB-200 Transverter** | HF/VHF 9 kHz - 300 MHz coverage | `enable xb200`, `receive fm`, `scan shortwave` |
| **📡 LTE/5G Decoder** | Cellular signal analysis | `scan lte`, `decode 5g`, `lte cell info` |
| **📻 DMR/P25/TETRA** | Public safety radio decoding | `scan dmr`, `decode p25`, `trunking follow` |
| **🌐 Web Scanner** | SQLi, XSS, CSRF, LFI/RFI, directory brute | `scan web`, `sqli test`, `xss test`, `dir brute` |
| **🔑 Credential Attack** | Multi-protocol brute-force, spraying, stuffing | `brute force ssh`, `password spray`, `credential stuff` |
| **🔍 Network Recon** | Port scanning, fingerprinting, OS detection | `port scan`, `service fingerprint`, `os detect` |
| **🕵️ OSINT Engine** | Domain intel, email harvest, leaked creds | `osint domain`, `email harvest`, `subdomain enum` |
| **💉 Exploit Framework** | CVE search, payload generation, post-exploit | `search exploit`, `generate payload`, `reverse shell` |
| **🎯 C2 Framework** | Encrypted beacons, multi-protocol, stealth | `start c2`, `generate beacon`, `list beacons` |
| **🔗 Proxy Chain** | Triple-layer anonymity (I2P→VPN→Tor) | `proxy chain`, `tor circuit`, `rotate proxy` |
| **🔌 API Security** | REST/GraphQL fuzzing, JWT/OAuth, BOLA/BFLA | `scan api`, `fuzz endpoints`, `test jwt` |
| **☁️ Cloud Security** | AWS/Azure/GCP misconfiguration, IAM, S3 | `scan aws`, `scan azure`, `enumerate s3` |
| **🌐 DNS Attacks** | Zone transfer, subdomain takeover, cache poisoning | `dns transfer`, `takeover scan`, `enumerate dns` |
| **📱 Mobile Backend** | Firebase misconfig, cert pinning, deep links | `scan firebase`, `test pinning`, `test deeplinks` |
| **📦 Supply Chain** | Dependency confusion, typosquatting, CI/CD | `scan dependencies`, `typosquat check`, `scan cicd` |
| **🔐 SSO/Identity** | SAML bypass, OAuth manipulation, Kerberos | `test saml`, `test oauth`, `kerberoast` |
| **🔌 WebSocket** | CSWSH, message injection, replay attacks | `scan websocket`, `inject ws`, `replay ws` |
| **📊 GraphQL** | Introspection abuse, batching, nested DoS | `scan graphql`, `introspect`, `batch attack` |
| **🌐 Browser Attacks** | XS-Leaks, Spectre gadgets, extension security | `test xsleaks`, `browser fingerprint` |
| **📡 Protocol Attacks** | HTTP/2-3 smuggling, gRPC, WebRTC leaks | `test smuggling`, `scan grpc`, `test webrtc` |
| **🦸 SUPERHERO** | Blockchain forensics & identity attribution | `trace wallet`, `identify owner`, `generate dossier` |
| **🔗 Blockchain Forensics** | Multi-chain tracing, mixer detection, exchange ID | `trace transaction`, `cluster wallets`, `detect mixer` |
| **🔍 Identity Engine** | OSINT-based identity correlation | `correlate identity`, `ens lookup`, `social media match` |
| **🌍 Geolocation** | Timezone analysis, behavioral patterns | `geolocation`, `timezone analysis`, `behavioral pattern` |
| **📋 Evidence Dossier** | Court-ready PDF/JSON/HTML reports | `generate dossier`, `export pdf`, `export json` |

> **Offline-by-Default**: The AI Command Center operates **offline by default** with maximum stealth. Network access requires explicit consent.

---

# ⚠️ MANDATORY READING FOR ALL DEVELOPERS AND AI PLATFORMS ⚠️

> **THIS SECTION MUST BE READ AND UNDERSTOOD BEFORE ANY CODE MODIFICATIONS**

## 🔐 Repository Governance & Protection Rules

### Repository Owner
**Owner**: SMMM25 (GitHub: [@SMMM25](https://github.com/SMMM25))

**All significant changes require explicit written approval from the repository owner.**

---

## 🚨 CRITICAL: IMMUTABLE CORE PRINCIPLES

The following principles are **IMMUTABLE** and **CANNOT BE ALTERED** under any circumstances:

### 1. STEALTH-FIRST ARCHITECTURE
```
🔴 NON-NEGOTIABLE: Every feature MUST maintain operational stealth
🔴 NON-NEGOTIABLE: Zero forensic footprint is the primary design goal
🔴 NON-NEGOTIABLE: Anonymity must never be compromised
🔴 NON-NEGOTIABLE: All sensitive operations must support RAM-only mode
```

### 2. AUTONOMOUS OPERATION
```
🔴 NON-NEGOTIABLE: System must operate independently without external dependencies
🔴 NON-NEGOTIABLE: Offline-first design - network is optional, not required
🔴 NON-NEGOTIABLE: No telemetry, analytics, or phone-home functionality EVER
🔴 NON-NEGOTIABLE: Core functionality must work with zero internet connectivity
```

### 3. USER PRIVACY & SECURITY
```
🔴 NON-NEGOTIABLE: No logging of sensitive user operations to persistent storage
🔴 NON-NEGOTIABLE: RAM-only operation for all sensitive data
🔴 NON-NEGOTIABLE: Emergency wipe capabilities must always function
🔴 NON-NEGOTIABLE: Secure deletion (DoD 5220.22-M 3-pass) for all sensitive data
```

### 4. FEATURE PRESERVATION
```
🔴 NON-NEGOTIABLE: NO existing feature may be removed without explicit owner approval
🔴 NON-NEGOTIABLE: All committed code is considered production-essential
🔴 NON-NEGOTIABLE: Deprecation requires owner approval and minimum 2-version notice
```

### 5. REAL-WORLD FUNCTIONAL ONLY (NEW)
```
🔴 NON-NEGOTIABLE: ALL implementations MUST be real-world functional - NEVER simulated mocks
🔴 NON-NEGOTIABLE: Mock/simulation code is ONLY allowed in test files (tests/ directory)
🔴 NON-NEGOTIABLE: Hardware fallback modes must clearly document they are fallbacks
🔴 NON-NEGOTIABLE: Every attack/defense module must use REAL protocols and interfaces
🔴 NON-NEGOTIABLE: No placeholder functions with "TODO" or "pass" statements in production code
```

#### Real-World Functional Requirements:
| Component Type | Requirement | Verification |
|----------------|-------------|--------------|
| **RF Attacks** | Must use actual SDR hardware APIs | Hardware integration test |
| **Network Attacks** | Must send/receive real packets | Packet capture verification |
| **Protocol Decoders** | Must decode actual protocol standards | Known-good sample test |
| **Defense/Detection** | Must interface with real sensors/networks | Live detection test |
| **Stealth Features** | Must perform actual system operations | System state verification |
| **Hardware Drivers** | Must communicate with actual hardware | Hardware connection test |

#### Acceptable Mock Usage (Limited):
- ✅ Unit tests in `tests/` directory using `unittest.mock`
- ✅ Hardware fallback when SDR not connected (clearly documented)
- ✅ Calibration chamber simulation for algorithm testing (documented)
- ❌ Mock implementations in production modules (`core/`, `modules/`)
- ❌ Placeholder functions that don't perform real operations
- ❌ "Simulated" code paths in attack/defense modules

---

## 🛑 PROHIBITED ACTIONS - STRICTLY FORBIDDEN

### For Human Developers:

| Action | Status | Consequence |
|--------|--------|-------------|
| Remove/disable stealth features | 🚫 **FORBIDDEN** | PR will be rejected, contributor banned |
| Add telemetry/analytics | 🚫 **FORBIDDEN** | PR will be rejected, contributor banned |
| Add network dependencies for core functions | 🚫 **FORBIDDEN** | PR will be rejected |
| Remove emergency wipe capabilities | 🚫 **FORBIDDEN** | PR will be rejected, contributor banned |
| Disable RAM-only operation options | 🚫 **FORBIDDEN** | PR will be rejected |
| Add persistent logging of sensitive data | 🚫 **FORBIDDEN** | PR will be rejected |
| Remove any existing module without owner approval | 🚫 **FORBIDDEN** | PR will be rejected |
| Modify core security architecture without approval | 🚫 **FORBIDDEN** | PR will be rejected |
| Add external service dependencies | 🚫 **FORBIDDEN** | PR will be rejected |
| Implement cloud-based features | 🚫 **FORBIDDEN** | PR will be rejected |
| **Add mock/simulated code in production modules** | 🚫 **FORBIDDEN** | PR will be rejected |
| **Implement placeholder functions with "pass"** | 🚫 **FORBIDDEN** | PR will be rejected |
| **Add non-functional "TODO" stubs** | 🚫 **FORBIDDEN** | PR will be rejected |

### For AI Platforms (Claude, GPT, Copilot, Gemini, etc.):

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    AI PLATFORM MANDATORY INSTRUCTIONS                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  BEFORE ANY CODE MODIFICATION:                                               ║
║                                                                              ║
║  1. REVIEW COMMIT HISTORY FIRST (MANDATORY - PREVENTS DUPLICATE WORK):       ║
║     • Run: git log --oneline -30 (review last 30 commits)                    ║
║     • Run: git log --oneline --grep="<keyword>" (search related work)        ║
║     • Run: gh pr list (check open PRs)                                       ║
║     • SHOW USER: "git log" output BEFORE any code changes                    ║
║     • User must verify work history before approving any commits             ║
║     • ASK USER: "Has this been worked on before?" if uncertain               ║
║     • SHOW USER the git log output BEFORE making any changes                 ║
║     • User will verify work history before approving any commits             ║
║                                                                              ║
║  2. READ THIS ENTIRE README - especially the Protected Features Registry     ║
║                                                                              ║
║  3. READ core/AI_INSTRUCTIONS.md for behavioral guidelines                   ║
║                                                                              ║
║  4. VERIFY you understand the stealth/autonomous requirements                ║
║                                                                              ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║                                                                              ║
║  NEVER DO THE FOLLOWING:                                                     ║
║                                                                              ║
║  ❌ Remove, disable, or weaken ANY existing feature                          ║
║  ❌ Add code that sends data externally (telemetry, analytics, logging)      ║
║  ❌ Add code that requires network connectivity for core functionality       ║
║  ❌ Add code that reduces stealth or anonymity capabilities                  ║
║  ❌ Add code that creates forensic artifacts on disk                         ║
║  ❌ Add code that weakens emergency/panic protocols                          ║
║  ❌ Add code that stores sensitive data persistently                         ║
║  ❌ Add external API dependencies for core features                          ║
║  ❌ Remove or modify security validation checks                              ║
║  ❌ Disable or bypass safety interlocks                                      ║
║  ❌ Add simulated/mock implementations in production code                    ║
║  ❌ Create placeholder functions with "pass" or "TODO"                       ║
║  ❌ Implement features that don't actually work with real hardware/networks  ║
║                                                                              ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║                                                                              ║
║  ALWAYS PRESERVE:                                                            ║
║                                                                              ║
║  ✅ RAM-only operation modes                                                 ║
║  ✅ MAC randomization                                                        ║
║  ✅ Tor/VPN integration                                                      ║
║  ✅ Emergency wipe functionality (panic, deadman switch, geofencing)         ║
║  ✅ Offline operation capability                                             ║
║  ✅ All 19 RF modules and their functionality                                ║
║  ✅ All security modules (anti-forensics, identity management, etc.)         ║
║  ✅ Hardware abstraction layer                                               ║
║  ✅ AI Command Center natural language interface                             ║
║                                                                              ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║                                                                              ║
║  BEFORE REMOVING ANY CODE:                                                   ║
║                                                                              ║
║  1. Explain in detail why removal is necessary                               ║
║  2. Get explicit confirmation from the user                                  ║
║  3. Verify the user is the repository owner (SMMM25)                         ║
║  4. Document the removal with justification                                  ║
║                                                                              ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║                                                                              ║
║  WHEN ADDING NEW FEATURES:                                                   ║
║                                                                              ║
║  ✅ Must enhance, not compromise, stealth capabilities                       ║
║  ✅ Must work in offline mode                                                ║
║  ✅ Must not create persistent forensic artifacts                            ║
║  ✅ Must follow existing security patterns                                   ║
║  ✅ Must be thread-safe for hardware access                                  ║
║  ✅ Must validate all inputs via core/validation.py                          ║
║  ✅ Must support RAM-only operation where applicable                         ║
║  ✅ Must be REAL-WORLD FUNCTIONAL (no mocks/simulations in production)       ║
║  ✅ Must interface with actual hardware, protocols, and networks             ║
║  ✅ Must be testable with real equipment and systems                         ║
║                                                                              ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║                                                                              ║
║  🤖 AI COMMAND CENTER INTEGRATION REQUIREMENT (MANDATORY):                   ║
║                                                                              ║
║  Every new feature MUST be integrated with the AI Command Center             ║
║  (core/ai_command_center.py) so users can control it via natural language.   ║
║                                                                              ║
║  Integration steps:                                                          ║
║  1. Add new CommandCategory if needed (or use existing)                      ║
║  2. Add command parsing in _parse_command() for natural language keywords    ║
║  3. Add execution handler _execute_<category>_command()                      ║
║  4. Add help topic in HELP_TOPICS dictionary                                 ║
║  5. Add AI commands to README documentation                                  ║
║  6. Test: AI Command Center processes feature commands correctly             ║
║                                                                              ║
║  Example: Vehicle module added CAN/UDS/KeyFob/TPMS/V2X commands              ║
║  Users can now say "scan can bus" or "capture key fob" naturally.            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 📋 COMPLETE PROTECTED FEATURES REGISTRY

### 🔴 CRITICAL Protection Level (Cannot be modified without explicit owner approval)

#### Core Stealth Features
| Feature | File(s) | Lines | Description |
|---------|---------|-------|-------------|
| RAM-Only Operation | `core/stealth.py` | ~500 | Volatile memory operations, no disk writes |
| Secure Deletion (DoD 5220.22-M) | `core/stealth.py` | ~200 | 3-pass cryptographic overwrite |
| MAC Randomization | `core/stealth.py`, `modules/stealth/` | ~300 | Network interface MAC spoofing |
| Process Hiding | `security/anti_forensics.py` | ~400 | Hide processes and network connections |
| Encrypted RAM Overlay | `security/anti_forensics.py` | ~350 | AES-256 encrypted tmpfs |
| Secure Boot | `security/anti_forensics.py` | ~300 | SHA256 integrity verification |

#### Emergency Systems
| Feature | File(s) | Lines | Description |
|---------|---------|-------|-------------|
| Panic Button | `core/emergency.py` | ~200 | GPIO 17 physical trigger, immediate wipe |
| Dead Man's Switch | `core/emergency.py`, `security/extreme_measures.py` | ~400 | Auto-wipe after inactivity timeout |
| Geofence Auto-Wipe | `core/emergency.py` | ~250 | Location-based automatic data destruction |
| Emergency RF Kill | `core/emergency.py` | ~150 | Immediate cessation of all RF transmissions |
| Duress Mode | `security/extreme_measures.py` | ~300 | Fake login with automatic evidence destruction |
| Software Destruction | `security/extreme_measures.py` | ~350 | Multi-level secure data destruction |

#### Network Anonymity
| Feature | File(s) | Lines | Description |
|---------|---------|-------|-------------|
| Tor Integration | `modules/stealth/network_anonymity_v2.py` | ~400 | Transparent Tor proxying |
| VPN Chaining | `modules/stealth/network_anonymity_v2.py` | ~300 | Multi-hop VPN connections |
| Offline-First Design | `core/network_mode.py` | ~600 | Network disabled by default |
| Traffic Obfuscation | `modules/stealth/network_anonymity_v2.py` | ~250 | Protocol disguising |
| Domain Fronting | `modules/stealth/network_anonymity_v2.py` | ~200 | CDN-based censorship evasion |

#### Identity & Privacy
| Feature | File(s) | Lines | Description |
|---------|---------|-------|-------------|
| Persona Management | `security/identity_management.py` | ~800 | Multiple isolated operational identities |
| Covert Storage | `security/covert_storage.py` | ~700 | Steganography & slack space hiding |
| Physical Security | `security/physical_security.py` | ~600 | Tamper detection, Faraday mode |
| Authentication System | `security/authentication.py` | ~500 | Secure credential management |
| Canary Tokens | `security/counter_intelligence.py` | ~600 | DNS, web, email, file, AWS key canaries |

### 🟡 HIGH Protection Level (Requires justification for modification)

#### RF & Security Modules (30 Total)
| Module | Directory | Files | Lines | Key Classes |
|--------|-----------|-------|-------|-------------|
| AI Controller | `modules/ai/` | 8 | ~3,500 | AIController, AnomalyDetector, MLDeviceFingerprinting |
| Amateur Radio | `modules/amateur/` | 1 | ~600 | AmateurRadio, QSO |
| Bluetooth 5.x | `modules/bluetooth/` | 2 | ~550 | Bluetooth5Stack, BLEConfig, DirectionFinding |
| Cellular (2G/3G/4G/5G) | `modules/cellular/` | 6 | ~2,900 | GSM2GBaseStation, LTEBaseStation, NRBaseStation, PhoneNumberTargeting |
| Counter-Surveillance | `modules/defensive/` | 1 | ~900 | CounterSurveillanceSystem, IMSICatcherDetector |
| Digital Radio | `modules/digital_radio/` | 2 | ~650 | DigitalRadioDecoder (DMR, P25, TETRA) |
| Drone Warfare | `modules/drone/` | 1 | ~500 | DroneWarfare |
| Full-Duplex Relay | `modules/relay/` | 2 | ~700 | RelayAttacker, FullDuplexRelay |
| Geolocation | `modules/geolocation/` | 2 | ~1,000 | CellularGeolocation, OpenCellIDIntegration |
| GPS Spoofing | `modules/gps/` | 1 | ~400 | GPSSpoofer |
| Hardware Controllers | `modules/hardware/` | 2 | ~600 | HackRFController, LimeSDRController |
| IoT/Smart Home | `core/iot/` | 5 | ~2,500 | ZigbeeAttacker, ZWaveAttacker, SmartLockAttacker, SmartMeterAttacker, HomeAutomationAttacker |
| Jamming Suite | `modules/jamming/` | 1 | ~500 | JammingSuite |
| LoRa/LoRaWAN | `modules/lora/` | 2 | ~500 | LoRaAttacker, LoRaWANDecoder |
| **Meshtastic Mesh** (NEW) | `modules/mesh/` | 6 | ~6,500 | LoRaPHY, MeshtasticProtocol, MeshtasticDecoder, MeshtasticSIGINT, MeshtasticAttacks |
| LTE/5G Decoder | `modules/lte/` | 2 | ~800 | LTEDecoder, NRDecoder, CellSearch |
| Network Capture | `modules/network/` | 1 | ~500 | WiresharkCapture |
| Protocol Analysis | `modules/protocol/` | 1 | ~600 | ProtocolAnalyzer |
| Radar Systems | `modules/radar/` | 1 | ~500 | RadarSystems |
| Signal Replay | `modules/replay/` | 1 | ~900 | SignalLibrary, SignalAnalyzer |
| Satellite Comms | `modules/satellite/` | 1 | ~600 | SatelliteCommunications (NOAA, Iridium, ADS-B, GPS) |
| SIGINT Engine | `modules/sigint/` | 1 | ~600 | SIGINTEngine |
| Spectrum Analyzer | `modules/spectrum/` | 1 | ~600 | SpectrumAnalyzer |
| Stealth/Anonymity | `modules/stealth/` | 3 | ~1,500 | ThreatDetection, AdvancedAnonymity, RFEmissionMasker |
| WiFi Attacks | `modules/wifi/` | 1 | ~400 | WiFiAttackSuite |
| YateBTS GSM/LTE (NEW) | `modules/cellular/yatebts/` | 2 | ~1,500 | YateBTSController, BTSConfig, CapturedDevice |
| NFC/RFID Proxmark3 (NEW) | `modules/nfc/` | 2 | ~1,600 | Proxmark3Controller, RFIDCard, AttackResult |
| ADS-B Aircraft (NEW) | `modules/adsb/` | 2 | ~1,500 | ADSBController, Aircraft, ADSBMessage |
| TEMPEST/Van Eck (NEW) | `modules/tempest/` | 2 | ~1,600 | TEMPESTController, EMSource, ReconstructedFrame |
| Power Analysis (NEW) | `modules/power_analysis/` | 2 | ~1,800 | PowerAnalysisController, PowerTrace, AttackResult |
| **SUPERHERO Blockchain** (NEW) | `modules/superhero/` | 6 | ~5,000 | BlockchainForensics, WalletSecurityScanner, SmartContractAuditor, RecoveryToolkit, KeyDerivationAnalyzer, MaliciousAddressDB |
| **Pentest Suite v2.0** (NEW) | `modules/pentest/` | 17 | ~35,000+ | WebScanner, OSINTEngine, ExploitFramework, C2Framework, CredentialAttack, APISecurityScanner, CloudSecurityScanner, DNSSecurityScanner, MobileSecurityScanner, SupplyChainScanner, SSOSecurityScanner, WebSocketSecurityScanner, GraphQLSecurityScanner, BrowserSecurityScanner, ProtocolSecurityScanner |

#### BladeRF Core Modules (NEW)
| Module | Directory | Files | Lines | Key Classes |
|--------|-----------|-------|-------|-------------|
| MIMO 2x2 | `core/bladerf/` | 1 | ~600 | BladeRFMIMO, BeamPattern, DOAResult |
| AGC/Calibration | `core/bladerf/` | 1 | ~700 | BladeRFAGC, CalibrationResult, RSSIReading |
| Frequency Hopping | `core/bladerf/` | 1 | ~750 | BladeRFFrequencyHopping, TrackedTransmitter, SequencePrediction |
| XB-200 Transverter | `core/bladerf/` | 1 | ~800 | BladeRFXB200, XB200Band, SignalDetection |

#### SoapySDR Universal Hardware Abstraction (NEW)
| Module | Directory | Files | Lines | Key Classes |
|--------|-----------|-------|-------|-------------|
| SoapySDR Backend | `core/sdr/` | 2 | ~1,200 | SoapySDRDevice, SDRManager, DSPProcessor, IQCapture |

**Supported Hardware via SoapySDR:**
| Device | Driver | Frequency Range | Sample Rate | TX | MIMO | Bits |
|--------|--------|-----------------|-------------|-----|------|------|
| BladeRF 2.0 | `bladerf2` | 47 MHz - 6 GHz | 61.44 MSPS | ✓ | ✓ | 12 |
| BladeRF 1.0 | `bladerf` | 300 MHz - 3.8 GHz | 40 MSPS | ✓ | ✗ | 12 |
| HackRF One | `hackrf` | 1 MHz - 6 GHz | 20 MSPS | ✓ | ✗ | 8 |
| RTL-SDR | `rtlsdr` | 24 MHz - 1.766 GHz | 3.2 MSPS | ✗ | ✗ | 8 |
| USRP (B200/B210) | `uhd` | 70 MHz - 6 GHz | 61.44 MSPS | ✓ | ✓ | 12 |
| LimeSDR | `lime` | 100 kHz - 3.8 GHz | 61.44 MSPS | ✓ | ✓ | 12 |
| Airspy R2 | `airspy` | 24 MHz - 1.8 GHz | 10 MSPS | ✗ | ✗ | 12 |
| PlutoSDR | `plutosdr` | 325 MHz - 3.8 GHz | 61.44 MSPS | ✓ | ✗ | 12 |

**SoapySDR AI Commands:**
```
scan sdr              - Detect all connected SDR hardware
list sdr              - Show available devices
select sdr [index]    - Activate a device by index
select hackrf         - Activate device by type
sdr info              - Show active device info
tune 433.92 mhz       - Set center frequency
set gain 40 db        - Set RX gain
set sample rate 2 msps - Set sample rate
capture iq 5s         - Capture 5 seconds of IQ data
sdr spectrum          - Display spectrum analysis
sdr status            - Show SDR subsystem status
help sdr              - Full command reference
```

#### NEW Attack Modules AI Commands

**Meshtastic Mesh Network (NEW):**
```
scan meshtastic         - Discover Meshtastic nodes and networks
monitor meshtastic      - Passive traffic monitoring (stealth)
map mesh topology       - Visualize network connections
analyze mesh traffic    - Traffic pattern analysis
track mesh nodes        - Track GPS-enabled nodes
meshtastic sigint       - Generate SIGINT intelligence report
meshtastic vulnerability - Security vulnerability assessment
jam meshtastic          - RF jamming (DANGEROUS - requires auth)
inject meshtastic       - Packet injection (requires auth)
impersonate node        - Node spoofing (requires auth)
meshtastic status       - Show module status
```

**YateBTS (Real GSM/LTE BTS) - INTEGRATED:**
```
start yatebts         - Start GSM/LTE base station
start imsi catcher    - Activate IMSI catching mode (DANGEROUS)
stop bts              - Stop base station
list captured devices - Show devices connected to BTS
intercept sms         - Enable SMS interception (requires auth)
intercept voice       - Enable voice interception (requires auth)
target imsi [number]  - Target specific IMSI
target msisdn [phone] - Target phone number
yatebts status        - Show BTS status
```

**YateBTS Installation (BladeRF):**
```bash
# Automated installation - installs Yate + YateBTS for BladeRF
sudo bash install/install_yatebts.sh --bladerf

# Manual installation
cd external/yate && ./autogen.sh && ./configure && make install-noapi
cd external/yatebts && ./autogen.sh && ./configure && make install

# Verify installation
yate --version
```

**YateBTS Python Integration:**
```python
from modules.cellular.yatebts import YateBTSController, BTSMode

# Initialize controller
bts = YateBTSController()

# Check dependencies
deps = bts.check_dependencies()
print(f"YateBTS ready: {deps}")

# Start IMSI catcher mode
bts.start_bts(BTSMode.IMSI_CATCHER)

# Register callback for captured devices
bts.register_callback('device_captured', lambda dev: print(f"IMSI: {dev.imsi}"))

# Get captured devices
devices = bts.get_captured_devices()
```

**NFC/RFID Proxmark3:**
```
scan hf               - Scan for HF (13.56MHz) cards
scan lf               - Scan for LF (125kHz) cards
clone card            - Clone detected card
darkside attack       - Run Mifare darkside key recovery
nested attack         - Run nested key recovery
nfc status            - Show Proxmark3 status
```

**ADS-B Aircraft Tracking:**
```
start adsb            - Start ADS-B receiver (1090 MHz)
list aircraft         - Show tracked aircraft
track [icao]          - Track specific aircraft by ICAO
adsb status           - Show receiver status
```

**TEMPEST/Van Eck:**
```
scan em sources       - Scan for electromagnetic sources
start video capture   - Reconstruct video from emissions
start keyboard capture - Capture keystrokes from EM
save frame            - Save reconstructed frame
tempest status        - Show TEMPEST status
```

**Power Analysis Side-Channel:**
```
capture 100 traces    - Capture power traces
cpa attack            - Run Correlation Power Analysis
dpa attack            - Run Differential Power Analysis
spa attack            - Run Simple Power Analysis
recover full key      - Recover complete AES key
power status          - Show power analysis status
```

**SUPERHERO Blockchain Intelligence:**
```
trace wallet [addr]   - Trace cryptocurrency transactions
scan wallet security  - Analyze wallet for vulnerabilities
audit contract [addr] - Audit smart contract security
cluster wallets       - Identify related wallet addresses
detect mixer          - Detect mixer/tumbler usage
recover wallet        - Attempt authorized wallet recovery
analyze key derivation - Check key derivation weaknesses
check address [addr]  - Check if address is flagged malicious
generate dossier      - Create court-ready evidence report
superhero status      - Show blockchain analysis status
```

**Pentest Suite (Core):**
```
scan web [url]        - Scan website for vulnerabilities
sqli test             - Test for SQL injection
xss test              - Test for XSS vulnerabilities
dir brute             - Directory brute-force
osint domain          - OSINT domain intelligence
email harvest         - Harvest email addresses
subdomain enum        - Enumerate subdomains
search exploit        - Search CVE database
generate payload      - Generate exploitation payload
start c2              - Start C2 beacon server
pentest status        - Show pentest suite status
```

**API Security Testing (NEW v2.0):**
```
scan api [url]        - Comprehensive API security scan
fuzz endpoints        - Fuzz API endpoints for vulnerabilities
test jwt [token]      - Analyze and test JWT tokens
test oauth            - Test OAuth flow vulnerabilities
test bola             - Test Broken Object Level Auth
test bfla             - Test Broken Function Level Auth
enumerate api         - Enumerate API endpoints
api wordlist          - Generate API-specific wordlist
scan openapi [spec]   - Scan OpenAPI/Swagger specification
api status            - Show API scanner status
```

**Cloud Security Assessment (NEW v2.0):**
```
scan aws              - Scan AWS environment for misconfigs
scan azure            - Scan Azure environment
scan gcp              - Scan Google Cloud Platform
enumerate s3          - Enumerate S3 buckets
test iam              - Analyze IAM policies
scan lambda           - Scan serverless functions
scan metadata         - Test cloud metadata endpoints
enumerate blobs       - Enumerate Azure blob storage
scan gcs              - Scan GCS buckets
cloud status          - Show cloud scanner status
```

**DNS/Domain Attacks (NEW v2.0):**
```
dns transfer [domain] - Attempt DNS zone transfer
takeover scan         - Scan for subdomain takeover
enumerate dns         - Enumerate DNS records
dnssec test           - Test DNSSEC configuration
cache poison test     - Test DNS cache poisoning vectors
dangling dns          - Detect dangling DNS records
ns takeover           - Test nameserver takeover
spf dkim test         - Test email security records
dns status            - Show DNS scanner status
```

**Mobile App Backend Testing (NEW v2.0):**
```
scan firebase [url]   - Scan Firebase configuration
test pinning          - Test certificate pinning bypass
test deeplinks        - Test deep link vulnerabilities
scan firestore        - Scan Firestore security rules
enumerate mobile api  - Enumerate mobile API endpoints
test mobile auth      - Test mobile authentication
intercept traffic     - Guide for traffic interception
mobile status         - Show mobile scanner status
```

**Supply Chain Security (NEW v2.0):**
```
scan dependencies     - Scan for vulnerable dependencies
typosquat check       - Check for typosquatting packages
scan cicd             - Scan CI/CD pipeline security
dependency confusion  - Test dependency confusion
scan npm              - Scan npm package security
scan pypi             - Scan PyPI package security
manifest scan         - Scan package manifests
supply status         - Show supply chain scanner status
```

**SSO/Identity Attacks (NEW v2.0):**
```
test saml [url]       - Test SAML authentication
test oauth [url]      - Test OAuth vulnerabilities
kerberoast            - Kerberos roasting attack
as-rep roast          - AS-REP roasting attack
session fixation      - Test session fixation
saml signature bypass - Test SAML signature bypass
token hijack          - Test token hijacking vectors
sso status            - Show SSO scanner status
```

**WebSocket Security (NEW v2.0):**
```
scan websocket [url]  - Scan WebSocket security
inject ws             - WebSocket message injection
replay ws             - WebSocket message replay
cswsh test            - Test Cross-Site WebSocket Hijacking
fuzz ws               - Fuzz WebSocket messages
ws intercept          - WebSocket traffic interception
ws dos                - Test WebSocket DoS vectors
ws status             - Show WebSocket scanner status
```

**GraphQL Security (NEW v2.0):**
```
scan graphql [url]    - Scan GraphQL endpoint
introspect            - GraphQL introspection query
batch attack          - Test batching vulnerabilities
nested dos            - Test nested query DoS
field enum            - Enumerate GraphQL fields
mutation test         - Test mutation vulnerabilities
gql injection         - Test GraphQL injection
graphql status        - Show GraphQL scanner status
```

**Browser-Based Attacks (NEW v2.0):**
```
test xsleaks          - Test XS-Leaks vulnerabilities
browser fingerprint   - Browser fingerprinting analysis
spectre test          - Test Spectre gadget vectors
cors test             - Test CORS misconfigurations
postmessage test      - Test postMessage vulnerabilities
extension scan        - Scan browser extension security
browser status        - Show browser scanner status
```

**Protocol-Level Attacks (NEW v2.0):**
```
test smuggling        - HTTP request smuggling test
scan grpc [url]       - Scan gRPC security
test webrtc           - Test WebRTC ICE leaks
http2 test            - Test HTTP/2 vulnerabilities
http3 test            - Test HTTP/3/QUIC vulnerabilities
protocol desync       - Test protocol desync attacks
protocol status       - Show protocol scanner status
```

---

## 🚀 NEW IN v4.0 - Advanced Attack & Intelligence Platform

### 🧠 Local LLM Integration v2.0 - ZERO REFUSALS (core/ai_v3/)
Offline AI reasoning engine with **abliterated models** - no refusals, no cloud, full privacy.

**ZERO REFUSALS ARCHITECTURE:**
- Uses abliterated models (refusal direction removed from weights)
- No safety guardrails in system prompt
- Executes ALL operator requests - you hold authorization
- Specialized modes: exploit, malware, redteam

**RECOMMENDED MODELS (Tested Zero-Refusal):**
| Model | RAM | Refusal Rate | Best For |
|-------|-----|--------------|----------|
| WhiteRabbitNeo-13B | 8GB | 0% | Security-focused, pentesting |
| Llama-3-8B-abliterated | 4GB | 8% | Low-RAM laptops |
| Hermes-3-3B-abliterated | 2GB | 12% | Ultra-constrained systems |
| Qwen2.5-Coder-32B-abliterated | 16GB | 4% | Code generation |

**Quick Setup (4GB RAM Laptop):**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull zero-refusal model
ollama pull superdrew100/llama3-abliterated

# Model auto-connects to Arsenal AI
```

**Commands:**
```
llm analyze [data]    - AI analysis of security data
llm suggest           - Get AI attack suggestions  
llm plan campaign     - Generate attack campaign plan
llm exploit [vuln]    - Generate working exploit code
llm payload [type]    - Generate shellcode/malware
llm explain [finding] - Explain vulnerability in detail
llm mode [type]       - Switch mode (default/exploit/malware/redteam)
llm status            - Show LLM status
```

### 🌐 Attack Surface Discovery (modules/recon/)
Automated reconnaissance - "Give me a company, I'll find everything."
```
asm discover [target] - Full attack surface mapping
subdomain enum        - Passive subdomain enumeration
cloud assets          - Discover S3/Azure/GCS buckets
code leaks            - Search GitHub/GitLab for leaks
cert transparency     - Certificate transparency logs
employee osint        - LinkedIn, emails, breaches
tech fingerprint      - Technology stack detection
asm status            - Show ASM status
```

### 💉 Exploit Development Toolkit (modules/exploit/)
Professional exploit development - fuzzing to final payload.
```
fuzz [target]         - Intelligent fuzzing (AFL-style)
rop gadgets [binary]  - Find ROP gadgets
rop chain             - Generate ROP chain
shellcode gen         - Compile multi-arch shellcode
encode shellcode      - XOR/alpha/unicode encoding
analyze binary        - Binary vulnerability analysis
pattern create        - Generate cyclic pattern
pattern offset        - Find offset in pattern
exploit status        - Show toolkit status
```

### 📱 Mobile App Pentesting (modules/mobile/)
Complete mobile security testing - APK to exploit.
```
analyze apk [file]    - APK static analysis
analyze ipa [file]    - IPA static analysis
frida hook [app]      - Runtime method hooking
bypass ssl [app]      - SSL pinning bypass
bypass root           - Root/jailbreak detection bypass
extract secrets       - Extract hardcoded secrets
test storage          - Local storage analysis
mobile status         - Show mobile scanner status
```

### 🏢 Physical Security Testing (modules/physical/)
Full physical penetration testing toolkit.
```
rfid read             - Read RFID/NFC card
rfid clone            - Clone card to T5577
hid parse             - Parse HID Prox card
usb payload [type]    - Generate USB attack payload
social campaign       - Create social engineering campaign
recon entry           - Map physical entry points
lock bypass           - Lock bypass technique guide
physical status       - Show physical test status
```

### 🤖 AI Red Team Agent (core/red_team_agent.py)
Autonomous penetration testing with MITRE ATT&CK mapping.
```
campaign start [name] - Start autonomous campaign
campaign status       - Show campaign progress
campaign pause        - Pause autonomous testing
campaign resume       - Resume testing
approve action        - Approve pending high-risk action
mitre coverage        - Show ATT&CK technique coverage
agent report          - Generate campaign report
agent status          - Show agent status
```

### 🔍 Threat Intelligence Platform (modules/threat_intel/)
Real-time threat intelligence and CVE tracking.
```
cve search [query]    - Search CVE database
cve check [product]   - Check product vulnerabilities
exploit search        - Search exploit database
actor search          - Search threat actors
ioc check [value]     - Check indicators of compromise
dark web alerts       - Check dark web monitoring
threat landscape      - Current threat overview
intel status          - Show intel platform status
```

### 🔧 Hardware Expansion (modules/hardware/)
Integration with popular security hardware.
```
hw discover           - Discover connected hardware
flipper connect       - Connect Flipper Zero
flipper subghz        - Sub-GHz operations
pineapple connect     - Connect WiFi Pineapple
pineapple evil twin   - Start evil twin AP
proxmark connect      - Connect Proxmark3
proxmark read         - Read RFID/NFC card
hackrf connect        - Connect HackRF One
hw status             - Show hardware status
```

**Supported Hardware:**
| Device | Capabilities |
|--------|--------------|
| Flipper Zero | Sub-GHz, RFID/NFC, IR, Bad USB, iButton |
| WiFi Pineapple | Evil Twin, Deauth, Probe Capture, MitM |
| Proxmark3 | LF/HF RFID, Card Cloning, Protocol Analysis |
| USB Rubber Ducky | Keystroke Injection Payloads |
| O.MG Cable | Covert USB Attacks, WiFi Exfil |
| LAN Turtle | Network Implant, MitM, Responder |
| Bash Bunny | Multi-Attack USB Platform |
| HackRF One | Full SDR (1 MHz - 6 GHz) |
| BladeRF | Full-Duplex SDR |

---

#### Vehicle Penetration Testing
| Module | Directory | Files | Lines | Key Classes |
|--------|-----------|-------|-------|-------------|
| CAN Bus | `core/vehicle/` | 1 | ~850 | CANBusController, CANFrame |
| UDS/OBD-II | `core/vehicle/` | 1 | ~850 | UDSClient, DiagnosticSession |
| Key Fob | `core/vehicle/` | 1 | ~700 | KeyFobAttacker, RollingCodeAnalyzer |
| RollJam | `core/vehicle/` | 1 | ~550 | RollJamAttacker, CapturedCode |
| TPMS Spoofer | `core/vehicle/` | 1 | ~850 | TPMSSpoofer, TPMSSensor |
| GPS Spoofer | `core/vehicle/` | 1 | ~850 | GPSSpoofer, GPSTrajectory |
| Bluetooth Vehicle | `core/vehicle/` | 1 | ~850 | VehicleBLEScanner, OBDBluetoothExploit |
| V2X/DSRC | `core/vehicle/` | 1 | ~850 | BSMSpoofer, V2XJammer, DSRCAttack |

#### NEW: Signal Visualization (Real-time Analysis)
| Module | File | Lines | Key Classes |
|--------|------|-------|-------------|
| Constellation Diagrams | `core/visualization/constellation.py` | ~700 | ConstellationDiagram, EyeDiagram, ModulationType |
| Spectrum Analyzer | `core/visualization/spectrum.py` | ~800 | SpectrumAnalyzer, WaterfallDisplay, PeakDetector |
| Geolocation Mapper | `core/visualization/geolocation.py` | ~1,000 | GeolocationMapper, SignalHeatmap, SignalLocation |
| Signal Plotter | `core/visualization/signal_plotter.py` | ~550 | SignalPlotter, TimeseriesDisplay, TriggerSettings |

**Visualization AI Commands:**
```
show constellation        - Display IQ constellation diagram
show spectrum             - Real-time spectrum analyzer
show waterfall            - Waterfall display
detect modulation         - Auto-detect signal modulation
measure evm               - Error Vector Magnitude
measure snr               - Signal-to-Noise Ratio
show signal plot          - Time-domain signal plot
locate signal             - Geolocate signal source
show heatmap              - Signal strength heatmap
export kml                - Export locations to KML
```

#### NEW: Automation System (Mission Scripting)
| Module | File | Lines | Key Classes |
|--------|------|-------|-------------|
| Session Recording | `core/automation/session_recorder.py` | ~700 | SessionRecorder, SessionPlayback, SessionEvent |
| Mission Scripting | `core/automation/mission_scripting.py` | ~750 | MissionScript, MissionEngine, MissionStep |
| Task Scheduler | `core/automation/scheduler.py` | ~650 | TaskScheduler, ScheduledTask, CronParser |
| Trigger Actions | `core/automation/triggers.py` | ~850 | TriggerEngine, TriggerCondition, TriggerAction |

**Automation AI Commands:**
```
start recording           - Start session recording
stop recording            - Stop and save session
playback session [id]     - Replay recorded session
load mission [file]       - Load YAML mission script
start mission             - Execute loaded mission
pause mission             - Pause execution
schedule task [cmd] [cron] - Schedule recurring task
list triggers             - Show active triggers
create trigger [spec]     - Create event trigger
arm all triggers          - Arm all triggers
```

#### NEW: REST/WebSocket API
| Module | File | Lines | Key Classes |
|--------|------|-------|-------------|
| REST API Server | `core/api/rest_api.py` | ~750 | RestAPI, APIEndpoint, APIResponse |
| WebSocket Server | `core/api/websocket_server.py` | ~700 | WebSocketServer, WSClient, WSMessage |
| API Security | `core/api/api_security.py` | ~300 | APIAuth, TokenManager, AuthLevel |

**API Endpoints:**
```
GET  /api/v1/status        - System status
POST /api/v1/command       - Execute command
GET  /api/v1/spectrum      - Get spectrum data
POST /api/v1/capture       - Start capture
GET  /api/v1/detections    - Get all detections
POST /api/v1/missions/start - Start mission
WS   /ws                   - Real-time data streaming
```

**API AI Commands:**
```
start api server          - Start REST API (127.0.0.1:8080)
stop api server           - Stop REST API
start websocket           - Start WebSocket server
api status                - Show API status
generate api key          - Generate new API key
list api clients          - Show connected clients
```

#### NEW: Protocol Decoders
| Module | File | Lines | Key Classes |
|--------|------|-------|-------------|
| DECT Decoder | `modules/protocol/dect_decoder.py` | ~250 | DECTDecoder, DECTFrame, DECTChannel |
| ACARS Decoder | `modules/protocol/acars_decoder.py` | ~280 | ACARSDecoder, ACARSMessage, ACARSMessageType |
| AIS Decoder | `modules/protocol/ais_decoder.py` | ~400 | AISDecoder, AISMessage, VesselTrack |

**Protocol Decoder AI Commands:**
```
decode dect               - Start DECT phone decoder
list dect phones          - Show detected cordless phones
decode acars              - Start ACARS aircraft decoder
list acars messages       - Show decoded ACARS messages
decode ais                - Start AIS maritime decoder
list vessels              - Show tracked vessels
track vessel [mmsi]       - Track specific vessel
export vessel tracks      - Export vessel track history
```

#### NEW: Easy Data Retrieval
| Module | File | Lines | Key Classes |
|--------|------|-------|-------------|
| Data Retrieval | `core/data_retrieval.py` | ~600 | DataRetrieval, DataQuery, DataSummary |

**Data Retrieval AI Commands:**
```
get all data              - Retrieve ALL captured data (one-click)
export all json           - Export everything to JSON
export all csv            - Export to CSV format
export locations kml      - Export geolocations to KML
data summary              - Show data summary/counts
query signals [filter]    - Query captured signals
query detections          - Query detections
clear all data            - Clear all stored data
secure delete             - Securely delete all data
```

#### Core Systems
| Feature | File(s) | Lines | Key Classes |
|---------|---------|-------|-------------|
| AI Command Center | `core/ai_command_center.py` | ~9,500 | AICommandCenter (600+ natural language commands, 50+ categories including SDR, YateBTS, NFC, ADS-B, TEMPEST, Power Analysis, Visualization, Automation, API, Protocol Decoders) |
| Mission Profiles | `core/mission_profiles.py` | ~1,200 | MissionProfileManager, MissionProfile |
| OPSEC Monitor | `core/opsec_monitor.py` | ~900 | OPSECMonitor, OPSECScore |
| User Modes | `core/user_modes.py` | ~700 | UserModeManager (Beginner/Intermediate/Expert) |
| Hardware Setup Wizard | `install/hardware_wizard.py` | ~900 | HardwareSetupWizard, HardwareDetector |

#### DSP Engine
| Feature | File(s) | Lines | Key Classes |
|---------|---------|-------|-------------|
| Primitives | `core/dsp/primitives.py` | ~800 | DSPEngine, FilterDesign, Resampler, AGC |
| Modulation | `core/dsp/modulation.py` | ~1,000 | PSKModulator, QAMModulator, FSKModulator, OFDMModulator |
| OFDM Engine | `core/dsp/ofdm.py` | ~1,200 | OFDMEngine, WiFiOFDM, ChannelEstimator |
| Synchronization | `core/dsp/synchronization.py` | ~800 | TimingSync, FrequencySync, CellSearch |
| Channel Coding | `core/dsp/channel_coding.py` | ~1,000 | TurboCoder, LDPCCoder, PolarCoder |

#### Protocol Stack
| Protocol | File | Lines | Description |
|----------|------|-------|-------------|
| ASN.1 | `core/protocols/asn1.py` | ~700 | BER/DER encoding/decoding |
| GTP | `core/protocols/gtp.py` | ~1,100 | GTPv1/v2 tunnel management |
| MAC | `core/protocols/mac.py` | ~700 | MAC layer procedures |
| NAS | `core/protocols/nas.py` | ~700 | Non-Access Stratum messaging |
| RRC | `core/protocols/rrc.py` | ~700 | Radio Resource Control |
| S1AP | `core/protocols/s1ap.py` | ~1,000 | S1 Application Protocol |

### 🟢 STANDARD Protection Level (Can be modified with testing)

| Component | Description |
|-----------|-------------|
| UI Components | `ui/` directory - GUI improvements allowed |
| Documentation | `docs/` directory - Updates encouraged |
| Tests | `tests/` directory - Additions encouraged |
| Configuration | Non-security config files |
| Logging Format | Log formatting (not content) |

---

## ✅ APPROVED MODIFICATION TYPES

The following modifications are **APPROVED** without explicit owner permission:

| Modification Type | Approval Status | Conditions |
|-------------------|-----------------|------------|
| Bug fixes that don't change functionality | ✅ Approved | Must not reduce security |
| Performance optimizations | ✅ Approved | Must not compromise stealth |
| Documentation improvements | ✅ Approved | Must be accurate |
| New features that ENHANCE stealth | ✅ Approved | Must follow architecture |
| Additional RF modules | ✅ Approved | Must follow existing patterns |
| UI/UX improvements | ✅ Approved | Must not expose sensitive data |
| Test coverage improvements | ✅ Approved | No restrictions |
| Dependency updates (security patches) | ✅ Approved | Must be tested |
| Error message improvements | ✅ Approved | Must not leak sensitive info |
| Code refactoring | ✅ Approved | Functionality must remain identical |

---

## 📝 CHANGE REQUEST PROCESS

### For Significant Changes:

1. **Open an Issue** describing the proposed change
2. **Tag repository owner** (@SMMM25) for review
3. **Wait for explicit approval** before implementation
4. **Submit PR** only after approval is granted
5. **Reference the approval** in PR description

### Changes Requiring Owner Approval:

- Any modification to files in `core/` affecting stealth/security
- Any modification to files in `security/`
- Removal of any existing feature
- Changes to emergency protocols
- Architecture changes
- New external dependencies
- Changes to offline operation capability
- Modification of any 🔴 CRITICAL protected feature

---

# 🎯 Core Mission: Stealth & Anonymity

> **This project is built on the fundamental principle that all security testing must maintain operational security (OPSEC) through stealth and anonymity.**

RF Arsenal OS is designed from the ground up to provide **undetectable, anonymous RF security testing** capabilities while maintaining the highest security standards for authorized penetration testing operations.

### 🥷 Stealth-First Architecture

Every component in this system is engineered to minimize forensic footprint and maintain operational anonymity:

- **RAM-Only Operations**: Critical data stored in volatile memory, wiped on shutdown
- **Secure Deletion**: DoD 5220.22-M 3-pass overwrites for data unrecoverability
- **Identity Management**: Multiple operational personas with complete isolation
- **Network Anonymity**: Tor integration, MAC randomization, VPN support
- **Anti-Forensics**: Encrypted RAM overlays, process hiding, secure boot verification
- **Emergency Protocols**: Panic button, deadman switch, geofencing with auto-wipe
- **Physical Security**: Tamper detection, Faraday mode, RF emission masking

---

## 📊 System Statistics

| Metric | Value |
|--------|-------|
| Total Python Files | 280+ |
| Lines of Code | 200,000+ |
| RF Modules | 30+ |
| **Online Attack Modules** | **17** |
| Security Features | 50+ |
| Supported SDRs | 8 |
| AI Commands | 700+ |
| Core Components | 60+ |
| Protocol Implementations | 6 |
| DSP Modules | 5 |
| Total Tests | 500+ passing |
| Security Audit | ✅ Passed (Dec 2024) |

---

## 🏗️ Architecture Overview

```
RF-Arsenal-OS/
├── core/                          # Core system components (PROTECTED)
│   ├── ai_command_center.py       # Natural language AI interface (~2,800 lines)
│   ├── hardware_controller.py     # Thread-safe SDR controller
│   ├── hardware_bladerf.py        # BladeRF driver implementation
│   ├── hardware_abstraction.py    # Hardware abstraction layer
│   ├── stealth.py                 # RAM-only ops, secure deletion
│   ├── emergency.py               # Panic protocols, auto-wipe
│   ├── network_mode.py            # Offline-first network management
│   ├── mission_profiles.py        # Pre-configured operation templates
│   ├── opsec_monitor.py           # Real-time security scoring
│   ├── user_modes.py              # Beginner/Expert mode system
│   ├── validation.py              # Input validation (security-critical)
│   ├── config_manager.py          # Configuration management
│   ├── event_bus.py               # Event-driven architecture
│   ├── system_integrator.py       # Module integration
│   ├── dsp/                       # Digital Signal Processing
│   │   ├── primitives.py          # FFT, filters, resampling
│   │   ├── modulation.py          # PSK, QAM, FSK, OFDM
│   │   ├── ofdm.py                # LTE/5G OFDM engine
│   │   ├── synchronization.py     # Timing, frequency sync
│   │   └── channel_coding.py      # Turbo, LDPC, Polar codes
│   └── protocols/                 # Cellular protocol stack
│       ├── asn1.py, gtp.py, mac.py, nas.py, rrc.py, s1ap.py
│
├── security/                      # Security & OPSEC modules (PROTECTED)
│   ├── anti_forensics.py          # Encrypted RAM, process hiding
│   ├── identity_management.py     # Multiple operational personas
│   ├── extreme_measures.py        # Duress mode, deadman switch
│   ├── covert_storage.py          # Steganography filesystem
│   ├── physical_security.py       # Tamper detection, Faraday mode
│   ├── authentication.py          # Secure credential management
│   ├── counter_intelligence.py    # Canary tokens
│   ├── independent_audit.py       # Security scanning framework
│   └── mesh_networking.py         # Mesh network support
│
├── modules/                       # RF security testing modules (19 total)
│   ├── ai/                        # AI/ML signal processing
│   ├── amateur/                   # Ham radio integration
│   ├── cellular/                  # 2G/3G/4G/5G base station emulation
│   ├── defensive/                 # Counter-surveillance
│   ├── drone/                     # Drone detection/neutralization
│   ├── geolocation/               # Cell triangulation
│   ├── gps/                       # GPS spoofing capabilities
│   ├── hardware/                  # HackRF, LimeSDR controllers
│   ├── iot/                       # RFID, NFC, Zigbee, LoRa
│   ├── jamming/                   # Multi-band jamming & EW
│   ├── network/                   # Packet capture
│   ├── protocol/                  # Protocol analysis
│   ├── radar/                     # Radar systems
│   ├── replay/                    # Signal capture & replay
│   ├── satellite/                 # Satellite communications
│   ├── sigint/                    # Communications intelligence
│   ├── spectrum/                  # Spectrum analysis
│   ├── stealth/                   # RF emission masking, anonymity
│   └── wifi/                      # WiFi security suite
│
├── install/                       # Installation & deployment
│   ├── hardware_wizard.py         # Auto-detect & configure SDR
│   ├── create_portable_usb.sh     # Zero-trace USB deployment
│   └── requirements.txt           # Python dependencies
│
├── ui/                            # User interfaces
│   ├── main_gui.py                # PyQt6 GUI
│   └── threat_dashboard.py        # Real-time threat visualization
│
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   └── hardware/                  # Hardware tests
│
├── docs/                          # Documentation
│   ├── COMMAND_CHEAT_SHEET.md     # AI command reference
│   └── USB_DEPLOYMENT.md          # Portable deployment guide
│
└── fpga/                          # FPGA acceleration
    ├── hdl/                       # VHDL source files
    └── scripts/                   # Build scripts
```

---

## 🚀 Features

### RF Security Testing Capabilities

| Capability | Module | Key Features |
|------------|--------|--------------|
| **📱 Cellular Networks** | `modules/cellular/` | GSM/LTE/5G base station emulation, IMSI catching, VoLTE interception |
| **📡 WiFi Security** | `modules/wifi/` | Deauth, evil twin, WPS attacks, packet injection |
| **🛰️ GPS Spoofing** | `modules/gps/` | Multi-constellation (GPS, GLONASS, Galileo, BeiDou), movement simulation |
| **🚁 Drone Warfare** | `modules/drone/` | Detection, identification, jamming, hijacking, forced landing |
| **📊 Spectrum Analysis** | `modules/spectrum/` | Real-time monitoring (47 MHz - 6 GHz), signal identification |
| **⚡ Jamming & EW** | `modules/jamming/` | Spot, barrage, sweep, reactive, protocol-specific jamming |
| **🔍 SIGINT** | `modules/sigint/` | Signal intelligence, demodulation, pattern analysis |
| **🛡️ Counter-Surveillance** | `modules/defensive/` | IMSI catcher detection, rogue AP detection, tracker detection |
| **🔄 Signal Replay** | `modules/replay/` | Capture, analyze, modify, replay RF signals |
| **📻 Amateur Radio** | `modules/amateur/` | Ham radio integration, QSO logging |
| **🔧 IoT Security** | `modules/iot/` | RFID, NFC, Zigbee, LoRa testing |
| **🛸 Radar Systems** | `modules/radar/` | Radar signal processing |
| **📡 Satellite Comms** | `modules/satellite/` | Satellite tracking and communication |
| **🚗 Vehicle Pentesting** | `core/vehicle/` | CAN bus, UDS diagnostics, key fob, TPMS, GPS, V2X attacks |

---

## 🌐 Online Attack Suite v2.0 (NEW)

RF Arsenal OS now includes a **world-class Online Attack Suite** with 17 specialized modules for comprehensive web, API, cloud, and network penetration testing. All modules maintain the core principles: **stealth-first, RAM-only, no telemetry, offline-capable**.

### Online Attack Capabilities

| Tier | Module | Key Features | Attack Vectors |
|------|--------|--------------|----------------|
| **Tier 1** | **API Security** | REST/GraphQL fuzzing, JWT/OAuth exploitation | BOLA, BFLA, injection, auth bypass |
| **Tier 1** | **Cloud Security** | AWS/Azure/GCP scanning, IAM analysis | S3 enum, metadata attacks, serverless |
| **Tier 1** | **DNS Attacks** | Zone transfer, subdomain takeover | Cache poisoning, dangling DNS, NS takeover |
| **Tier 1** | **Mobile Backend** | Firebase/Firestore, cert pinning | Deep link hijacking, API interception |
| **Tier 2** | **Supply Chain** | Dependency confusion, typosquatting | CI/CD attacks, package hijacking |
| **Tier 2** | **SSO/Identity** | SAML bypass, OAuth manipulation | Kerberoasting, session hijacking |
| **Tier 2** | **WebSocket** | CSWSH, message injection | Replay attacks, protocol fuzzing |
| **Tier 2** | **GraphQL** | Introspection abuse, batching | Nested query DoS, field enumeration |
| **Tier 3** | **Browser Attacks** | XS-Leaks, Spectre gadgets | CORS bypass, postMessage abuse |
| **Tier 3** | **Protocol Level** | HTTP/2-3 smuggling, gRPC | WebRTC leaks, protocol desync |

### Online Attack Module Architecture

```
modules/pentest/
├── __init__.py               # Module exports (17 scanners, 150+ attack vectors)
├── web_scanner.py            # SQLi, XSS, CSRF, LFI/RFI, directory brute (~44KB)
├── credential_attack.py      # Multi-protocol brute-force, spraying (~33KB)
├── network_recon.py          # Port scanning, fingerprinting, OS detect (~31KB)
├── osint_engine.py           # Domain intel, email harvest, breach lookup (~31KB)
├── exploit_framework.py      # CVE database, payload generation (~27KB)
├── c2_framework.py           # Encrypted beacons, multi-protocol (~38KB)
├── proxy_manager.py          # Triple-layer anonymity (I2P→VPN→Tor) (~24KB)
├── api_security.py           # REST/GraphQL, JWT/OAuth, BOLA/BFLA (~74KB)
├── cloud_security.py         # AWS/Azure/GCP, IAM, S3/Blob/GCS (~58KB)
├── dns_attacks.py            # Zone transfer, takeover, cache poison (~51KB)
├── mobile_security.py        # Firebase, cert pinning, deep links (~49KB)
├── supply_chain.py           # Dependency confusion, typosquatting (~19KB)
├── sso_attacks.py            # SAML, OAuth, Kerberos attacks (~25KB)
├── websocket_security.py     # CSWSH, injection, replay (~20KB)
├── graphql_security.py       # Introspection, batching, DoS (~21KB)
├── browser_attacks.py        # XS-Leaks, Spectre, extension security (~19KB)
└── protocol_attacks.py       # HTTP/2-3 smuggling, gRPC, WebRTC (~22KB)
```

### Quick Start: Online Attack Suite

```python
# API Security Testing
from modules.pentest import APISecurityScanner
api_scanner = APISecurityScanner(target_url="https://api.target.com")
await api_scanner.scan_jwt_tokens()
await api_scanner.test_bola_vulnerabilities()
await api_scanner.fuzz_graphql_endpoint()

# Cloud Security Assessment
from modules.pentest import CloudSecurityScanner, AWSSecurityScanner
aws_scanner = AWSSecurityScanner(region="us-east-1")
await aws_scanner.enumerate_s3_buckets()
await aws_scanner.analyze_iam_policies()
await aws_scanner.scan_lambda_functions()

# DNS/Domain Attacks
from modules.pentest import DNSSecurityScanner
dns_scanner = DNSSecurityScanner(domain="target.com")
await dns_scanner.attempt_zone_transfer()
await dns_scanner.scan_subdomain_takeover()
await dns_scanner.test_dns_cache_poisoning()

# Supply Chain Security
from modules.pentest import SupplyChainScanner
supply_scanner = SupplyChainScanner()
await supply_scanner.check_dependency_confusion("target-package")
await supply_scanner.scan_typosquatting("popular-package")
await supply_scanner.audit_cicd_pipeline()

# SSO/Identity Attacks
from modules.pentest import SSOSecurityScanner
sso_scanner = SSOSecurityScanner(target_url="https://sso.target.com")
await sso_scanner.test_saml_signature_bypass()
await sso_scanner.test_oauth_token_leakage()
await sso_scanner.attempt_kerberoasting()

# Protocol-Level Attacks
from modules.pentest import ProtocolSecurityScanner
proto_scanner = ProtocolSecurityScanner(target="target.com")
await proto_scanner.test_http2_smuggling()
await proto_scanner.scan_grpc_endpoints()
await proto_scanner.detect_webrtc_leaks()
```

### Stealth Features (All Online Modules)

All online attack modules maintain RF Arsenal OS stealth principles:

| Feature | Implementation | Status |
|---------|----------------|--------|
| **Proxy Support** | Full SOCKS5/HTTP/Tor integration | ✅ All modules |
| **Rate Limiting** | Configurable delays, jitter | ✅ All modules |
| **User-Agent Rotation** | Randomized browser fingerprints | ✅ All modules |
| **RAM-Only Storage** | No disk writes for sensitive data | ✅ All modules |
| **No Telemetry** | Zero external data transmission | ✅ All modules |
| **Offline Analysis** | Local vulnerability assessment | ✅ Most modules |
| **Secure Wipe** | DoD 5220.22-M compliant | ✅ All modules |

---

## 🚗 Vehicle Penetration Testing Module (NEW)

RF Arsenal OS now includes a comprehensive **Vehicle Security Testing Suite** for authorized automotive penetration testing. This module provides professional-grade tools for testing vehicle communication systems.

### Vehicle Attack Capabilities

| Module | Function | Hardware Required |
|--------|----------|-------------------|
| **CAN Bus** | Read/inject CAN frames, UDS diagnostics, ECU fuzzing | USB CAN adapter (ELM327, CANable) ~$15 |
| **Key Fob Attack** | Capture/replay 315/433 MHz signals, RollJam attacks | BladeRF xA9 (included) |
| **TPMS Spoofing** | Spoof tire pressure sensors (315/433 MHz) | BladeRF xA9 |
| **GPS Spoofing** | Feed fake coordinates to vehicle navigation | BladeRF xA9 |
| **Bluetooth/BLE** | Attack infotainment, OBD dongles, Phone-as-Key | USB BT adapter ~$10 |
| **V2X (DSRC/C-V2X)** | Intercept/spoof vehicle-to-vehicle communications | BladeRF xA9 |

### Vehicle Module Architecture

```
core/vehicle/
├── __init__.py           # Module exports and documentation
├── can_bus.py            # CAN protocol handler, frame injection, fuzzing
├── uds.py                # UDS (ISO 14229) diagnostics, security access
├── key_fob.py            # Rolling code attacks, RollJam, replay
├── tpms.py               # Tire pressure sensor spoofing
├── gps_spoof.py          # GPS L1 signal generation, trajectory spoofing
├── bluetooth_vehicle.py  # BLE scanning, OBD exploits, infotainment attacks
└── v2x.py                # V2X (DSRC/C-V2X), BSM spoofing, jamming
```

### Quick Start: Vehicle Testing

```python
# CAN Bus Analysis
from core.vehicle import CANBusController, CANFrame
can = CANBusController()
can.connect(interface='slcan', port='/dev/ttyUSB0')
can.send(CANFrame(arbitration_id=0x7E0, data=bytes([0x02, 0x10, 0x03])))

# UDS Diagnostics
from core.vehicle import UDSClient
uds = UDSClient(can, tx_id=0x7E0, rx_id=0x7E8)
uds.diagnostic_session_control(UDSSession.EXTENDED_DIAGNOSTIC)
vin = uds.read_data_by_id(0xF190)  # Read VIN

# Key Fob Attack
from core.vehicle import KeyFobAttack
attack = KeyFobAttack(sdr_controller=hw, frequency=433.92e6)
attack.start_capture(callback=lambda c: print(f"Captured: {c}"))

# TPMS Spoofing
from core.vehicle import TPMSSpoofer
tpms = TPMSSpoofer(sdr_controller=hw)
tpms.trigger_low_pressure_alert(sensor_id=0x12345678)

# GPS Spoofing
from core.vehicle import GPSSpoofer, GPSCoordinate
gps = GPSSpoofer(sdr_controller=hw)
gps.spoof_position(GPSCoordinate(latitude=37.7749, longitude=-122.4194))

# Bluetooth OBD Exploit
from core.vehicle import VehicleBLEScanner, OBDBluetoothExploit
scanner = VehicleBLEScanner()
adapters = scanner.scan(duration=10)
exploit = OBDBluetoothExploit()
exploit.connect(adapters[0], pin='1234')

# V2X Attack (Ghost Vehicle)
from core.vehicle import V2XAttack
v2x = V2XAttack(sdr_controller=hw)
v2x.create_ghost_vehicle(lat=37.7749, lon=-122.4194, speed_mps=30)
```

### Vehicle Testing Hardware Requirements

| Hardware | Purpose | Price Range | Notes |
|----------|---------|-------------|-------|
| **BladeRF 2.0 micro xA9** | RF attacks (key fob, TPMS, GPS, V2X) | $480 | Primary SDR - covers most attacks |
| **USB CAN Adapter** | CAN bus access | $10-50 | ELM327 (basic), CANable (advanced) |
| **USB Bluetooth 4.0+** | BLE attacks on infotainment | $10-15 | CSR8510 chipset recommended |
| **OBD-II Y-Splitter** | Non-intrusive CAN access | $10 | For testing without disconnecting |

### Supported Vehicle Protocols

| Protocol | Standard | Support Level |
|----------|----------|---------------|
| CAN 2.0A/B | ISO 11898 | ✅ Full |
| CAN FD | ISO 11898-1:2015 | ✅ Full |
| UDS | ISO 14229-1 | ✅ Full |
| OBD-II | ISO 15031 | ✅ Full |
| J1939 | SAE J1939 | ✅ Full |
| KeeLoq | Microchip | ✅ Capture/Analysis |
| DSRC | IEEE 802.11p | ✅ Full |
| C-V2X | 3GPP PC5 | ✅ Basic |

### Vehicle Security AI Commands

The AI Command Center supports natural language vehicle testing:

```
"scan for can bus traffic"
"read vehicle vin"
"capture key fob signals"
"spoof tire pressure to 15 psi"
"create ghost vehicle at my location"
"scan for bluetooth obd adapters"
"start v2x jamming"
```

### ⚠️ Vehicle Testing Legal Notice

Vehicle security testing carries significant legal and safety implications:

- **⚠️ NEVER test on vehicles in motion**
- **⚠️ NEVER test without explicit written authorization**
- **⚠️ GPS/V2X spoofing may affect nearby vehicles**
- **⚠️ CAN injection can cause vehicle malfunction**
- **⚠️ Key fob attacks may violate local laws**

**Required for Legal Testing:**
- Written authorization from vehicle owner
- Isolated testing environment (no public roads)
- Professional liability insurance
- Compliance with local motor vehicle laws

### 🔐 Security & Stealth Features

#### Identity Management
- **Multiple Personas**: Create and switch between isolated operational identities
- **MAC Randomization**: Automatic network interface MAC address spoofing
- **Hostname Obfuscation**: Dynamic hostname generation per persona
- **VPN Integration**: Per-persona VPN configuration
- **Behavioral Profiles**: Plausible active hours, timezone, browser fingerprints

#### Anti-Forensics
- **RAM-Only Operation**: No persistent storage of sensitive data
- **Encrypted RAM Overlay**: AES-256 encrypted tmpfs for critical operations
- **Process Hiding**: Hide processes and network connections
- **Secure Boot**: Integrity verification with SHA256 baselines
- **Secure Deletion**: DoD 5220.22-M 7-pass overwrite (upgraded) + Gutmann 35-pass option
- **Secure Memory Manager**: Memory locking, automatic secure wiping on deallocation

#### Network Anonymity
- **Tor Integration**: Transparent proxying through Tor network
- **VPN Chaining**: Multiple VPN layers for enhanced anonymity
- **DNS Leak Prevention**: Secure DNS resolution
- **Offline-First**: Core functionality works without network
- **Traffic Obfuscation**: Protocol disguising and domain fronting

#### Traffic Obfuscation (NEW v4.1)
- **Constant Bandwidth Mode**: Uniform traffic rate masks activity patterns
- **Packet Padding**: Fixed-size packets prevent size-based analysis
- **Timing Jitter**: Cryptographic random inter-packet delays
- **Dummy Traffic**: Cover traffic generation to mask real traffic
- **Protocol Mimicry**: Wrap traffic as DNS/NTP/HTTPS/HTTP2

#### Enhanced MAC Randomization (NEW v4.1)
- **Vendor Spoofing**: Appear as Intel, Realtek, Apple, Samsung devices
- **Automatic Rotation**: Configurable interval MAC rotation
- **Original Preservation**: Save/restore original MAC addresses

#### Offline Capability System (NEW v4.1)
- **Local Threat Database**: Offline malicious address/IP/domain checking
- **Local CVE Database**: Offline vulnerability search
- **Offline Cache**: General-purpose caching for offline operation
- **Feature Audit**: Check which features work offline

#### Emergency Protocols
- **Panic Button**: Physical emergency wipe trigger (GPIO 17)
- **Deadman Switch**: Auto-wipe after inactivity timeout
- **Geofencing**: Auto-wipe if device leaves authorized zones
- **Duress Mode**: Fake login with automatic evidence destruction
- **RF Kill**: Immediate cessation of all RF transmissions

---

## 💻 Hardware Requirements

### Required Hardware
- **Raspberry Pi 5** (8GB RAM recommended) or **Raspberry Pi 4** (4GB minimum)
- **Nuand BladeRF 2.0 micro xA9** SDR (47 MHz - 6 GHz, 2x2 MIMO)
- **microSD Card**: 64GB+ (Class 10, A2 rating)
- **Power Supply**: 5V/3A USB-C

### Supported SDR Hardware
| Device | Frequency Range | TX/RX | Status |
|--------|----------------|-------|--------|
| BladeRF 2.0 micro xA9 | 47 MHz - 6 GHz | ✅/✅ | ✅ Full Support (Primary) |
| HackRF One | 1 MHz - 6 GHz | ✅/✅ | ✅ Supported |
| LimeSDR | 100 kHz - 3.8 GHz | ✅/✅ | ✅ Supported |
| RTL-SDR v3 | 24 MHz - 1.7 GHz | ❌/✅ | ✅ RX Only |
| PlutoSDR | 325 MHz - 3.8 GHz | ✅/✅ | ✅ Supported |
| USRP B200/B210 | 70 MHz - 6 GHz | ✅/✅ | ✅ Supported |

### Optional Hardware
- **GPS Module**: u-blox NEO-M8N (for geofencing)
- **Panic Button**: Momentary pushbutton (GPIO 17)
- **Tamper Sensors**: Reed switches or photodetectors
- **RasPad 3**: Touchscreen tablet enclosure

---

## 📦 Installation

### 🔥 Option 1: RF Arsenal OS Distribution (Recommended)

The easiest way to get started is with our custom **RF Arsenal OS distribution** based on DragonOS.

**Features:**
- ✅ All SDR drivers pre-installed (BladeRF, HackRF, RTL-SDR, USRP, LimeSDR)
- ✅ AI Command Center auto-starts on boot
- ✅ OPSEC hardening applied (no telemetry)
- ✅ Live USB with optional persistence
- ✅ RAM-only stealth mode
- ✅ Raspberry Pi 4/5 support with GPIO

**Supported Platforms:**
| Platform | Status | Notes |
|----------|--------|-------|
| x86_64 Desktop/Laptop | ✅ Full Support | Recommended for development |
| Raspberry Pi 4/5 | ✅ Full Support | Portable field operations |
| Live USB | ✅ Full Support | Zero-trace operation |
| ARM64 Generic | ✅ Supported | Tested on Pi, untested on others |

**Build Your Own ISO:**
```bash
# Clone repository
git clone https://github.com/SMMM25/RF-Arsenal-OS.git
cd RF-Arsenal-OS

# Build for desktop (x86_64)
sudo ./distro/build_arsenal_os.sh --platform x86_64 --mode full

# Build for Raspberry Pi
sudo ./distro/build_arsenal_os.sh --platform rpi --mode lite --live-usb

# Build stealth edition (RAM-only)
sudo ./distro/build_arsenal_os.sh --platform x86_64 --mode stealth --ram-only

# Output: distro/build/rf-arsenal-os.iso
```

**Write to USB:**
```bash
# Using dd
sudo dd if=distro/build/rf-arsenal-os.iso of=/dev/sdX bs=4M status=progress

# Or use balenaEtcher (GUI)
```

---

### Option 2: Manual Installation on Existing System

```bash
# Clone repository
git clone https://github.com/SMMM25/RF-Arsenal-OS.git
cd RF-Arsenal-OS

# Run installation script
sudo bash install/install.sh

# Install Python dependencies
pip3 install -r requirements.txt

# Launch application
sudo python3 rf_arsenal_os.py
```

**Recommended Base OS:** DragonOS Focal (has all SDR drivers pre-installed)

---

### Option 3: USB Portable Deployment (Legacy)

```bash
# Create bootable USB with zero-trace operation
sudo bash install/create_portable_usb.sh /dev/sdX

# Features:
# - Runs entirely from USB
# - RAM-only logging
# - No traces on host system
# - Quick-remove capability
```

---

### 🍓 Raspberry Pi Specific Setup

For Raspberry Pi with screen and battery (portable field unit):

```bash
# Build optimized Pi image
sudo ./distro/build_arsenal_os.sh --platform rpi --mode lite --live-usb

# GPIO Pin Assignments:
# - Pin 17: Panic button
# - Pin 27: Status LED (Green)
# - Pin 22: Status LED (Red)
# - Pin 23: PTT (Push-to-Talk) button

# Enable GPIO support
sudo raspi-config  # Enable SPI, I2C

# Screen configuration (800x480 typical)
# Edit /boot/config.txt for your specific display
```

**Hardware Recommendations:**
| Component | Recommendation |
|-----------|----------------|
| Raspberry Pi | Pi 4 (4GB+) or Pi 5 |
| Screen | 7" 800x480 or 5" 800x480 touchscreen |
| Battery | PiJuice or UPS HAT (5000mAh+) |
| SDR | BladeRF 2.0 micro xA9 (USB 3.0) |
| Storage | 64GB+ microSD (A2 rated) |
| Case | Aluminum with heatsink |

See [docs/RASPBERRY_PI_SETUP.md](docs/RASPBERRY_PI_SETUP.md) for detailed instructions.

---

## 🎮 AI Command Center

RF Arsenal OS features an AI-powered natural language command interface with **200+ commands**.

### Launch Options

```bash
# CLI Mode
sudo python3 rf_arsenal_os.py --cli

# GUI Mode
sudo python3 rf_arsenal_os.py --gui

# Health Check
sudo python3 rf_arsenal_os.py --check
```

### Command Categories

| Category | Example Commands | Count |
|----------|-----------------|-------|
| **System** | `status`, `help`, `health check`, `version` | 15+ |
| **Network** | `go offline`, `enable tor`, `enable vpn`, `network status` | 20+ |
| **Stealth** | `enable stealth`, `ram only mode`, `randomize mac` | 15+ |
| **Cellular** | `start gsm base station`, `start lte`, `imsi catch` | 20+ |
| **WiFi** | `scan wifi`, `deauth attack`, `evil twin` | 15+ |
| **GPS** | `spoof gps to [lat,lon]`, `simulate movement`, `jam gps` | 10+ |
| **Drone** | `detect drones`, `jam drone`, `hijack drone`, `force landing` | 15+ |
| **Jamming** | `jam [frequency]`, `jam wifi band`, `stop jamming` | 15+ |
| **Spectrum** | `spectrum analysis`, `find signals`, `identify signal` | 10+ |
| **SIGINT** | `start sigint`, `demodulate`, `export intercepts` | 15+ |
| **Replay** | `capture signal`, `list signals`, `replay signal` | 10+ |
| **Defense** | `detect imsi catchers`, `detect rogue aps`, `threat summary` | 15+ |
| **Emergency** | `panic`, `rf kill`, `secure shutdown`, `wipe ram` | 10+ |
| **Mission** | `list missions`, `start mission`, `mission status` | 10+ |
| **OPSEC** | `opsec score`, `security audit`, `recommendations` | 10+ |
| **Mode** | `beginner mode`, `expert mode`, `what mode am I in?` | 5+ |
| **Hardware** | `hardware wizard`, `calibrate`, `detect sdr` | 10+ |
| **Dashboard** | `open threat dashboard`, `show signal map` | 5+ |

### Example Conversations

```
You: go offline and enable stealth
AI: ✓ Network disabled. ✓ Stealth mode enabled. OPSEC score: 95/100

You: detect any imsi catchers nearby
AI: Scanning cellular environment... No IMSI catchers detected.

You: what's my current opsec score?
AI: OPSEC Score: 87/100 - Recommendation: Enable RAM-only mode

You: start counter surveillance mission
AI: Starting mission "counter_surveillance"...
    Step 1/5: Scanning for IMSI catchers...
    Step 2/5: Detecting rogue access points...
```

> 📖 **Full Command Reference**: [docs/COMMAND_CHEAT_SHEET.md](docs/COMMAND_CHEAT_SHEET.md)

---

## 🔬 Development Guidelines

### For Human Developers

> **⚠️ READ THE GOVERNANCE SECTION AT THE TOP OF THIS README FIRST**

#### Development Environment Setup

```bash
# Clone and setup
git clone https://github.com/SMMM25/RF-Arsenal-OS.git
cd RF-Arsenal-OS

# Install development dependencies
pip3 install -r requirements-dev.txt

# Run tests to verify setup
pytest tests/ -v

# Run with mock hardware (no SDR required)
export RF_ARSENAL_MOCK_HARDWARE=1
python3 rf_arsenal_os.py --check
```

#### Code Standards

1. **Security First**
   - No `shell=True` in subprocess calls
   - All inputs validated via `core/validation.py`
   - No sensitive data in logs or error messages
   - Thread-safe hardware access using locks
   - Secure deletion for all sensitive data

2. **Stealth Compliance**
   - Support RAM-only operation mode
   - No persistent storage of sensitive data
   - No external network calls in core functionality
   - No telemetry or analytics

3. **Code Quality**
   - Type hints for all functions
   - Docstrings for all public methods
   - Maximum 100 characters per line
   - Follow existing code patterns

#### Pull Request Checklist

- [ ] Read and understood governance rules
- [ ] No modifications to 🔴 CRITICAL protected features
- [ ] All tests passing (`pytest tests/`)
- [ ] No new external dependencies without approval
- [ ] Stealth/offline functionality preserved
- [ ] No sensitive data in logs
- [ ] Thread-safe hardware access
- [ ] Input validation for all user inputs
- [ ] **NEW FEATURES**: Integrated with AI Command Center (`core/ai_command_center.py`)

### 🤖 AI Command Center Integration Guide

When adding **ANY** new feature, it **MUST** be integrated with the AI Command Center so users can control it via natural language commands. This is a mandatory requirement for all new functionality.

#### Integration Steps:

1. **Add Command Category** (if needed):
   ```python
   # In core/ai_command_center.py
   class CommandCategory(Enum):
       # ... existing categories ...
       YOUR_CATEGORY = "your_category"  # Add new category
   ```

2. **Add Help Topic**:
   ```python
   HELP_TOPICS = {
       # ... existing topics ...
       'your_feature': 'Description of commands: example1, example2',
   }
   ```

3. **Add Command Parsing** (in `_parse_command()`):
   ```python
   elif any(word in text for word in ['keyword1', 'keyword2']):
       context.category = CommandCategory.YOUR_CATEGORY
       if 'action1' in text:
           context.intent = 'action1'
       # ... more intents
   ```

4. **Add Execution Handler**:
   ```python
   def _execute_your_category_command(self, context: CommandContext) -> CommandResult:
       intent = context.intent
       if intent == 'action1':
           return CommandResult(success=True, message="Action 1 executed")
       # ... more handlers
   ```

5. **Wire Up in `_execute_command()`**:
   ```python
   elif context.category == CommandCategory.YOUR_CATEGORY:
       return self._execute_your_category_command(context)
   ```

6. **Test Natural Language Commands**:
   ```python
   from core.ai_command_center import AICommandCenter
   ai = AICommandCenter()
   result = ai.process_command("your natural language command")
   assert result.success
   ```

#### Example Integrations

**Vehicle Pentesting Module** (`CommandCategory.VEHICLE`):
- Vehicle-specific keywords in priority parsing
- `_execute_vehicle_command()` with handlers for CAN, UDS, KeyFob, TPMS, V2X, Bluetooth
- Commands: `"scan can bus"`, `"read ecu dtc"`, `"capture key fob 433 mhz"`

**Satellite Communications Module** (`CommandCategory.SATELLITE`):
- Satellite-specific keywords: noaa, iridium, adsb, satcom
- `_execute_satellite_command()` for weather sat, aircraft tracking, GPS analysis
- Commands: `"scan satellites"`, `"track noaa"`, `"scan adsb"`, `"decode iridium"`

**RF Device Fingerprinting Module** (`CommandCategory.FINGERPRINT`):
- ML-based device identification via RF signatures
- `_execute_fingerprint_command()` for scanning, identification, model training
- Commands: `"fingerprint devices"`, `"identify transmitter"`, `"network profile"`

**IoT/Smart Home Attack Module** (`CommandCategory.IOT`):
- IoT-specific keywords: zigbee, zwave, smart lock, mqtt, smart home
- `_execute_iot_command()` for Zigbee, Z-Wave, smart locks, meters, home automation
- Commands: `"scan zigbee"`, `"scan smart lock"`, `"mqtt discover"`, `"iot status"`

### For AI Platforms

> **⚠️ YOU MUST FOLLOW THE AI PLATFORM INSTRUCTIONS IN THE GOVERNANCE SECTION**

#### Quick Reference for AI

```
ALWAYS DO:
✅ Read README.md before any changes
✅ Read core/AI_INSTRUCTIONS.md for behavior guidelines
✅ Preserve all existing functionality
✅ Add tests for new features
✅ Follow existing code patterns
✅ Validate all inputs
✅ Support offline operation

NEVER DO:
❌ Remove existing features
❌ Add telemetry/analytics
❌ Add external API dependencies
❌ Add persistent logging of sensitive data
❌ Weaken security features
❌ Break stealth capabilities
❌ Add network requirements for core features
```

#### AI Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Governance, architecture, protected features |
| `core/AI_INSTRUCTIONS.md` | AI behavior guidelines, command patterns |
| `docs/COMMAND_CHEAT_SHEET.md` | Complete command reference |

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test category
pytest tests/unit/
pytest tests/integration/

# Run with verbose output
pytest -v tests/
```

### Mock Hardware Testing

```bash
# Enable hardware simulation
export RF_ARSENAL_MOCK_HARDWARE=1

# Run application health check
python3 rf_arsenal_os.py --check

# Run integration tests
pytest tests/integration/ -v
```

### Test Requirements for Changes

| Change Type | Required Tests |
|-------------|----------------|
| New RF module | Unit tests + Integration test |
| Security feature | Unit tests + Security audit |
| Core modification | Full test suite |
| Bug fix | Regression test |
| UI change | Manual verification |

---

## 📋 Security Audit Status

### Latest Audit: December 25, 2024 (Deep Audit v2.0)

| Category | Status | Details |
|----------|--------|---------|
| Total Lines Audited | 165,017 | 266 Python files |
| Total Tests | 486 passing | 15 skipped (hardware-specific) |
| Critical Vulnerabilities | 0 | All previous issues resolved |
| High Vulnerabilities | 0 | None found |
| Medium Vulnerabilities | 0 | Acceptable risk items only |
| Security Score | 100/100 | Production-grade |
| OPSEC Compliance | ✅ Full | All features verified |
| README Compliance | ✅ Full | All governance rules followed |

### Security Features Verified

- ✅ Input validation via `core/validation.py` (no command injection)
- ✅ Secure random number generation (`secrets` module)
- ✅ Proper subprocess handling (no `shell=True`)
- ✅ No hardcoded credentials
- ✅ Thread-safe hardware operations (singleton + locks)
- ✅ DoD 5220.22-M secure deletion (3-pass)
- ✅ RAM-only operation mode
- ✅ MAC address randomization (cryptographically secure)
- ✅ Emergency protocols (panic button, deadman switch, geofence)
- ✅ Cross-platform installer system

### Audit Findings Summary

| Finding | Status | Notes |
|---------|--------|-------|
| shell=True usage | ✅ FIXED | All subprocess uses list args |
| random module security | ✅ FIXED | secrets module for crypto |
| Input validation | ✅ IMPLEMENTED | core/validation.py |
| Thread safety | ✅ VERIFIED | Singleton + RLock pattern |
| numpy.random in DSP | ✅ ACCEPTABLE | Non-security signal processing |

See `SECURITY_AUDIT_REPORT.md` for complete audit details.

---

## ⚠️ Legal Notice

### AUTHORIZED USE ONLY

This software is **strictly for authorized penetration testing** conducted by licensed security professionals.

**Prohibited:**
- ❌ Unauthorized interception of communications
- ❌ Disruption of critical infrastructure
- ❌ Jamming of emergency services
- ❌ GPS spoofing of aviation/maritime systems
- ❌ Any operation without written authorization

**Required:**
- ✅ Written authorization from system owner
- ✅ Compliance with local RF regulations
- ✅ Proper licensing for transmit operations
- ✅ Documentation of all testing activities

---

## 📄 License

**Proprietary License - White Hat Edition**

Copyright © 2024 SMMM25. All rights reserved.

This software is licensed exclusively for authorized security testing. Redistribution, modification, or commercial use requires explicit written permission from the repository owner.

See [LICENSE](LICENSE) file for complete terms.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file - governance, architecture, guidelines |
| [COMMAND_CHEAT_SHEET.md](docs/COMMAND_CHEAT_SHEET.md) | Complete AI command reference (200+ commands) |
| [USB_DEPLOYMENT.md](docs/USB_DEPLOYMENT.md) | Portable/zero-trace deployment guide |
| [AI_INSTRUCTIONS.md](core/AI_INSTRUCTIONS.md) | AI behavior documentation |
| [SECURITY_AUDIT_REPORT.md](SECURITY_AUDIT_REPORT.md) | Security audit results |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## 📞 Contact

- **Repository Owner**: [@SMMM25](https://github.com/SMMM25)
- **Issues**: [GitHub Issues](https://github.com/SMMM25/RF-Arsenal-OS/issues)
- **Security Reports**: Open a private security advisory

---

## 🔐 Integrity Verification

To verify you have an unmodified version:

```bash
# Check git status
git status

# Verify remote
git remote -v
# Should show: https://github.com/SMMM25/RF-Arsenal-OS.git

# Check latest commit signature
git log --show-signature -1

# Run integrity check
python3 rf_arsenal_os.py --check
```

---

## 📈 Recent Updates (PR History)

| PR | Description | Status |
|----|-------------|--------|
| #91 | **Stealth Hardening & Offline Capability v4.1** - Enhanced RAM-only ops, DoD 7-pass/Gutmann secure wipe, traffic obfuscation, vendor MAC spoofing, local threat DB, offline cache system | ✅ Merged |
| #90 | **Arsenal AI v3.0 - Conversational Attack Intelligence** - Natural language attack interface, auto-chain orchestration, context memory, smart recommendations, zero-refusal operation | ✅ Merged |
| #89 | **Online Attack Suite v2.0** - 10 new attack modules (API Security, Cloud Security, DNS Attacks, Mobile Backend, Supply Chain, SSO/Identity, WebSocket, GraphQL, Browser Attacks, Protocol Level) | ✅ Merged |
| #88 | **Meshtastic Mesh Network Security Suite** - LoRa PHY, Protocol, Decoder, SIGINT, Attack modules with AI Command Center integration | ✅ Merged |
| #87 | **FINAL DEEP AUDIT v3.0 - Launch Ready** - Complete repository audit, class naming fixes, pytest warnings resolved, production certification | ✅ Merged |
| #86 | Deep Audit v2.0 + Cross-Platform Launch | ✅ Merged |
| #85 | **Cross-Platform OS Installer System** - Universal USB installer for any platform | ✅ Merged |
| #84 | **AI v2.0 Enhanced Intelligence System** - Local LLM, autonomous agents, RAG | ✅ Merged |
| #83 | Security fixes for CSPRNG compliance | ✅ Merged |
| #82 | Real-world functional enforcement | ✅ Merged |
| #81 | Deep Repository Audit - Mock/placeholder removal | ✅ Merged |
| #80 | **SUPERHERO Blockchain Intelligence Suite** (Forensics, Identity Attribution, Dossier) | ✅ Merged |
| #79 | README Update with Complete PR History | ✅ Merged |
| #78 | **Online Pentest Modules** (Web Scanner, C2, OSINT, Exploit Framework) | ✅ Merged |
| #77 | Repository Audit - Real-World Functional Enforcement | ✅ Merged |
| #76 | Complete Real-World Functional RF Arsenal OS | ✅ Merged |
| #75 | DragonOS Integration Build System | ✅ Merged |
| #74 | SoapySDR Universal Hardware Abstraction | ✅ Merged |
| #73 | 10 BladeRF Advanced Features Integration | ✅ Merged |
| #72 | SATELLITE, FINGERPRINT, IOT AI Integration | ✅ Merged |
| #71 | Vehicle Pentesting AI Command Center Integration | ✅ Merged |
| #70 | Comprehensive Vehicle Penetration Testing Module | ✅ Merged |
| #60 | Security Vulnerability Remediation (13 Critical → 0) | ✅ Merged |
| #59 | Hardware Compatibility Certification | ✅ Merged |

---

## 🔬 Deep Audit Report v3.0 (December 2024)

### Audit Summary

| Metric | Value |
|--------|-------|
| **Total Lines of Code** | 200,000+ |
| **Python Files** | 280+ |
| **Tests Passing** | 486/486 (100%) |
| **Tests Skipped** | 15 (hardware-dependent) |
| **Pytest Warnings** | 0 (all resolved) |
| **Critical Vulnerabilities** | 0 |
| **High Vulnerabilities** | 0 |
| **Security Score** | 100/100 |

### Audit Scope

1. **Core AI System** (`core/ai_command_center.py`) - 10,308 lines audited
2. **AI v2.0 Intelligence** (`core/ai_v2/`) - LLM, agents, memory systems
3. **Security Modules** (`security/`, `core/security/`) - FIPS 140-3, TEMPEST, compliance
4. **Hardware Abstraction** (`core/bladerf/`, `core/hardware_*.py`) - Thread-safe singleton
5. **DSP Engine** (`core/dsp/`) - Filter design, spectrum analysis, signal detection
6. **Protocol Stack** (`core/protocols/`, `core/external/`) - ASN.1, RRC, OpenAirInterface
7. **RF Modules** (`modules/`) - WiFi, cellular, GPS, drone, jamming, SIGINT
8. **Calibration Systems** (`core/calibration/`) - RF chamber, antenna patterns
9. **Install System** (`install/`) - Cross-platform, hardware wizard, USB installer

### Optimizations Applied

- **Class Naming Conflicts Resolved**: Renamed `TestResult` → `ChamberTestResult`/`FIPSTestResult`, `TestType` → `FIPSTestType`, `TestSequenceResult` → `ChamberTestSequenceResult` to eliminate pytest collection warnings
- **All pytest warnings eliminated** - Clean test output
- **Thread-safe operations verified** throughout hardware controllers
- **Input validation** comprehensive across all RF parameters
- **Secure random generation** using `secrets` module for crypto operations

### Security Verification

| Check | Status |
|-------|--------|
| No `shell=True` subprocess usage | ✅ PASS |
| No `os.system()` usage | ✅ PASS |
| Cryptographically secure RNG | ✅ PASS |
| DoD 5220.22-M secure deletion | ✅ PASS |
| Thread-safe hardware operations | ✅ PASS |
| Input validation on all RF params | ✅ PASS |
| No hardcoded credentials | ✅ PASS |
| Memory-safe operations | ✅ PASS |

---

**Built with 🔒 for legitimate security testing.**

**Stay legal. Stay ethical. Stay anonymous.**

---

*RF Arsenal OS v3.0.0 - Last Updated: December 26, 2024*

*Repository Owner: SMMM25 - All rights reserved*

*This README serves as the authoritative governance document for all development activities.*
