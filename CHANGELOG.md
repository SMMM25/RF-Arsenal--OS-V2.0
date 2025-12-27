# Changelog

All notable changes to RF Arsenal OS are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive unit test coverage for core modules
- Test file fixes to match actual API implementations

### Fixed
- Fixed `test_dsp.py` to use correct constructor parameters for AGC, Resampler, PSKModulator, QAMModulator
- Fixed `test_protocols.py` to skip tests requiring `pycryptodome` when not installed
- Fixed `test_opsec_monitor.py` to use correct class names (OPSECIssue instead of OPSECViolation)
- Fixed `test_hardware_wizard.py` assertion for GPS antenna frequency range
- Added missing `Tuple` import to `user_modes.py`

## [1.3.0] - 2024-12-23

### Added
- **Signal Replay Library** (`modules/replay/signal_library.py`)
  - Capture, store, analyze, and replay RF signals
  - Auto-analysis for modulation (OOK, ASK, FSK, PSK)
  - Rolling code detection and warnings
  - Categories: keyfobs, garage doors, wireless sensors, TPMS
  - Export/import to IQ, WAV, JSON formats
  - AI Command Center integration with REPLAY commands

- **Hardware Auto-Setup Wizard** (`install/hardware_wizard.py`)
  - Plug-and-play SDR detection
  - Supports BladeRF, HackRF One, LimeSDR, RTL-SDR, Airspy, USRP
  - Automatic driver verification
  - Self-calibration routines (DC offset, IQ balance)
  - Antenna selection guide
  - AI Command Center integration with HARDWARE commands

## [1.2.0] - 2024-12-22

### Added
- **Counter-Surveillance System** (`modules/defensive/counter_surveillance.py`)
  - IMSI Catcher (Stingray) detection
  - Rogue Access Point detection
  - Bluetooth tracker detection
  - GPS spoofing detection
  - AI Command Center integration with DEFENSIVE commands

- **RF Threat Dashboard** (`ui/threat_dashboard.py`)
  - Real-time signal tracking and visualization
  - Threat classification (WiFi, Cellular, GPS, Bluetooth, Unknown)
  - ASCII threat map
  - Stealth footprint analysis
  - Alert management system
  - AI Command Center integration with DASHBOARD commands

## [1.1.0] - 2024-12-21

### Added
- **AI Command Center** (`core/ai_command_center.py`)
  - Natural language command interface
  - Supports 200+ commands across 16 categories
  - Dangerous command confirmation system
  - Context-aware command parsing
  - Offline-by-default operation

- **Network Mode Manager** (`core/network_mode.py`)
  - Offline-by-default architecture
  - On-demand network access with consent
  - Anonymity modes: Tor, VPN, Full, Direct
  - Auto-timeout for online sessions
  - Network kill switch

- **Mission Profiles** (`core/mission_profiles.py`)
  - Pre-configured operation templates
  - AI-guided execution
  - OPSEC validation before mission start
  - Skill-level requirements

- **OPSEC Monitor** (`core/opsec_monitor.py`)
  - Real-time security scoring (0-100)
  - Six categories: Network, Identity, Forensics, Hardware, Behavior, Location
  - Threat level assessment
  - Auto-fix capability for some issues
  - Detailed remediation recommendations

- **User Mode System** (`core/user_modes.py`)
  - Four skill levels: Beginner, Intermediate, Advanced, Expert
  - Adaptive UI based on skill level
  - Hidden dangerous options in beginner mode
  - Full control in expert mode

## [1.0.0] - 2024-12-20

### Added
- Initial release with core RF security testing capabilities
- BladeRF 2.0 micro xA9 support
- Cellular modules (2G/3G/4G/5G)
- WiFi security suite
- GPS spoofing capabilities
- Drone warfare detection/neutralization
- Spectrum analysis
- Multi-band jamming
- SIGINT capabilities
- Security features:
  - RAM-only operations
  - Secure deletion
  - Identity management
  - Network anonymity (Tor, VPN)
  - Anti-forensics
  - Emergency protocols
  - Physical security

### Security
- Production-grade security architecture
- Thread-safe hardware access
- Comprehensive input validation
- No `shell=True` in subprocess calls
- SHA-256 authentication
- DoD 5220.22-M secure deletion

---

## Legend

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security-related changes
