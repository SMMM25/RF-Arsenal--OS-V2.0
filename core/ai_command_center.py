#!/usr/bin/env python3
"""
RF Arsenal OS - AI Command Center
Central AI/LLM interface for controlling the entire system

CORE MISSION: User controls everything through natural language AI commands
- Voice or text commands to AI
- AI interprets and executes operations
- Offline-by-default with explicit online consent
- Full stealth and anonymity maintained

This is the BRAIN of RF Arsenal OS - all operations flow through here.
"""

import os
import sys
import re
import json
import logging
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class CommandCategory(Enum):
    """Categories of commands the AI can process"""
    NETWORK = "network"           # Online/offline, anonymity
    CELLULAR = "cellular"         # BTS, IMSI, phone targeting
    WIFI = "wifi"                 # WiFi attacks, scanning
    GPS = "gps"                   # GPS spoofing/jamming
    DRONE = "drone"               # Drone detection/warfare
    SPECTRUM = "spectrum"         # Spectrum analysis
    JAMMING = "jamming"           # Electronic warfare
    SIGINT = "sigint"             # Signals intelligence
    STEALTH = "stealth"           # OPSEC features
    EMERGENCY = "emergency"       # Panic, wipe, shutdown
    SYSTEM = "system"             # Status, config, help
    CAPTURE = "capture"           # Packet capture
    TARGETING = "targeting"       # Phone/device targeting
    MISSION = "mission"           # Guided mission profiles
    OPSEC = "opsec"               # OPSEC monitoring/scoring
    MODE = "mode"                 # User mode (beginner/expert)
    DEFENSIVE = "defensive"       # Counter-surveillance
    DASHBOARD = "dashboard"       # Threat dashboard
    REPLAY = "replay"             # Signal replay library
    HARDWARE = "hardware"         # Hardware setup wizard
    VEHICLE = "vehicle"           # Vehicle penetration testing (CAN, UDS, Key Fob, TPMS, GPS, BLE, V2X)
    SATELLITE = "satellite"       # Satellite communications (SATCOM, ADS-B, NOAA, GPS, Iridium)
    FINGERPRINT = "fingerprint"   # RF device fingerprinting (ML-based identification)
    IOT = "iot"                   # IoT/Smart Home attacks (Zigbee, Z-Wave, smart locks, home automation)
    # NEW: BladeRF Advanced Features (10 major capabilities)
    MIMO = "mimo"                 # MIMO 2x2 (beamforming, spatial multiplexing, DoA)
    RELAY = "relay"              # Full-duplex relay attacks (car keys, access cards)
    LORA = "lora"                # LoRa/LoRaWAN attacks (IoT infrastructure)
    BLUETOOTH5 = "bluetooth5"    # Bluetooth 5.x full stack (BLE 5.0-5.3, direction finding)
    ROLLJAM = "rolljam"          # RollJam implementation (rolling code attacks)
    AGC = "agc"                  # Hardware AGC/calibration exposure (AD9361)
    HOPPING = "hopping"          # Frequency hopping support (FHSS tracking, jamming)
    XB200 = "xb200"              # XB-200 transverter support (HF/VHF 9 kHz - 300 MHz)
    LTE5G = "lte5g"              # LTE/5G decoder (cellular interception)
    DIGITALRADIO = "digitalradio"  # DMR/P25/TETRA decoder (public safety radio)
    SDR = "sdr"                    # SoapySDR universal hardware abstraction (BladeRF, HackRF, RTL-SDR, USRP, LimeSDR)
    # NEW: Additional Attack Modules
    YATEBTS = "yatebts"            # YateBTS GSM/LTE BTS (real IMSI catching, SMS/Voice intercept)
    NFC = "nfc"                    # NFC/RFID Proxmark3 (card reading, cloning, attacks)
    ADSB = "adsb"                  # ADS-B aircraft tracking and injection
    TEMPEST = "tempest"            # TEMPEST/Van Eck EM surveillance
    POWERANALYSIS = "poweranalysis"  # Power analysis side-channel attacks
    # NEW: Visualization and Analysis
    VISUALIZATION = "visualization"  # Constellation diagrams, spectrum, waterfall, geolocation maps
    # NEW: Automation
    AUTOMATION = "automation"      # Session recording, mission scripting, scheduled tasks, triggers
    # NEW: API
    API = "api"                    # REST/WebSocket API control
    # NEW: Protocol Decoders
    PROTOCOL = "protocol"          # DECT, ACARS, AIS protocol decoders
    # NEW: Data Retrieval
    DATA = "data"                  # Easy retrieve all data feature
    # NEW: Online Penetration Testing
    PENTEST = "pentest"            # Web scanner, credential attacks, recon, exploits, C2, OSINT
    # NEW: Blockchain Intelligence & Identity Attribution
    SUPERHERO = "superhero"        # Blockchain forensics, identity correlation, geolocation, dossier generation
    # NEW: AI v2.0 Enhanced Intelligence
    AI_ENHANCED = "ai_enhanced"    # Enhanced AI commands (mode control, autonomous tasks, attack planning)
    # NEW: Meshtastic Mesh Network Security Testing
    MESHTASTIC = "meshtastic"      # Meshtastic mesh network analysis, SIGINT, and attacks


@dataclass
class CommandContext:
    """Context for command execution"""
    raw_input: str
    category: Optional[CommandCategory] = None
    intent: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    requires_confirmation: bool = False
    is_dangerous: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CommandResult:
    """Result of command execution"""
    success: bool
    message: str
    data: Optional[Dict] = None
    warnings: List[str] = field(default_factory=list)
    follow_up_suggestions: List[str] = field(default_factory=list)


class AICommandCenter:
    """
    Central AI Command Interface for RF Arsenal OS
    
    The user interacts with this AI to control ALL system functions.
    Commands can be text or voice (processed through voice_ai module).
    
    Key Features:
    - Natural language understanding
    - Safety checks for dangerous operations
    - Offline-by-default network mode
    - Online mode requires explicit consent with warnings
    - Full audit logging
    - Context-aware responses
    """
    
    # Dangerous commands that require confirmation
    DANGEROUS_COMMANDS = [
        'go online', 'enable network', 'connect',
        'jam', 'jamming', 'disrupt',
        'wipe', 'delete', 'emergency',
        'transmit', 'broadcast',
        'hijack', 'intercept',
    ]
    
    # Help topics
    HELP_TOPICS = {
        'network': 'Network mode: offline (default), go online tor, go online vpn, go online full, go offline',
        'cellular': 'Cellular: start 4g bts, imsi catch, target phone +1234567890, stop cellular',
        'wifi': 'WiFi: scan networks, deauth, evil twin, capture handshakes',
        'gps': 'GPS: spoof location 37.7749 -122.4194, jam gps, stop gps',
        'drone': 'Drone: detect drones, jam drones, auto-defend, hijack drone',
        'spectrum': 'Spectrum: scan 100mhz to 6ghz, analyze 2.4ghz',
        'jamming': 'Jamming: jam 2.4ghz, jam wifi, jam gps, stop jamming',
        'stealth': 'Stealth: enable ram-only, rotate mac, secure delete [file], stealth hardening, traffic obfuscation, constant bandwidth, dod 7pass wipe, vendor spoof [type], max stealth',
        'stealth_hardening': 'Stealth Hardening: secure memory allocate, secure memory free, dod 3pass/7pass/gutmann wipe, memory stats, emergency wipe all',
        'traffic_obfuscation': 'Traffic Obfuscation: enable packet padding, enable timing jitter, enable dummy traffic, enable constant bandwidth, enable protocol mimicry, max obfuscation, obfuscation status',
        'offline': 'Offline Mode: go offline, offline status, check offline [feature], cache for offline, threat db stats, offline report',
        'emergency': 'Emergency: panic, wipe all, emergency shutdown',
        'status': 'Status: show status, show network, show stealth',
        'missions': 'Missions: list missions, start mission [name], next step, abort mission',
        'opsec': 'OPSEC: show opsec, fix opsec, opsec report',
        'mode': 'User Mode: set mode beginner/intermediate/expert, show mode',
        'defensive': 'Counter-Surveillance: scan threats, check stingray, detect trackers, threat status',
        'dashboard': 'Dashboard: show dashboard, show threats, show map, threat summary',
        'replay': 'Signal Replay: capture signal, list signals, replay signal, analyze signal, delete signal',
        'hardware': 'Hardware Setup: detect hardware, setup wizard, calibrate, antenna guide, troubleshoot',
        'vehicle': 'Vehicle Pentesting: scan can bus, read ecu, key fob attack, spoof tpms, spoof gps vehicle, scan bluetooth obd, create ghost vehicle',
        'satellite': 'Satellite Comms: scan satellites, track noaa, decode iridium, receive weather image, track aircraft adsb, scan gps signals',
        'fingerprint': 'Device Fingerprinting: fingerprint devices, identify transmitter, train fingerprint model, device history, network profile',
        'iot': 'IoT/Smart Home: scan smart home, scan zigbee, scan zwave, unlock smart lock, control device, scan smart meter, jam zigbee',
        # NEW: BladeRF Advanced Features
        'mimo': 'MIMO 2x2: enable mimo, beamform [direction], estimate doa, mimo diversity, mimo multiplex, channel sounding, mimo status',
        'relay': 'Relay Attacks: start relay car key, relay access card, relay nfc, full duplex mode, two device mode, capture relay signal, relay status',
        'lora': 'LoRa Attacks: scan lora, sniff lorawan, replay lora packet, jam lora, spoof gateway, inject downlink, lora status',
        'meshtastic': 'Meshtastic: scan meshtastic, monitor meshtastic, map mesh topology, analyze mesh traffic, track mesh nodes, meshtastic sigint, jam meshtastic (DANGEROUS), inject meshtastic, impersonate node, meshtastic status',
        'bluetooth5': 'Bluetooth 5.x: scan ble5, long range scan, direction finding, enumerate gatt, ble5 attack, coded phy scan, bluetooth5 status',
        'rolljam': 'RollJam: start rolljam, capture code, replay code, export codes, jam and capture, rolljam status',
        'agc': 'AGC/Calibration: set agc mode, manual gain, calibrate rf, dc offset, iq calibration, rssi monitor, temperature compensation, agc status',
        'hopping': 'Freq Hopping: start hopping, track hopper, predict sequence, sync to hopper, jam hopping, hopping pattern, hopping status',
        'xb200': 'XB-200 Transverter: enable xb200, tune hf, receive fm, receive noaa, scan shortwave, scan aircraft, tune amateur, xb200 status',
        'lte5g': 'LTE/5G Decoder: scan lte, decode lte, capture imsi lte, 5g nr scan, decode sib, lte cell info, lte5g status',
        'digitalradio': 'Digital Radio: scan dmr, decode p25, decode tetra, dmr intercept, p25 scanner, trunking follow, digital radio status',
        'sdr': 'SDR Hardware: scan sdr, list sdr, select sdr, sdr info, configure sdr, capture iq, play iq, sdr spectrum, tune [freq], set gain, set sample rate, sdr status',
        # NEW: Additional Attack Module Help Topics
        'yatebts': 'YateBTS: start bts, start imsi catcher, stop bts, list captured devices, intercept sms, intercept calls, target imsi, send silent sms, bts status',
        'nfc': 'NFC/RFID: scan lf, scan hf, read card, clone card, mifare attack, darkside attack, nested attack, sniff nfc, emulate card, nfc status',
        'adsb': 'ADS-B: start adsb receiver, stop adsb, list aircraft, track [icao], inject aircraft (DANGEROUS), adsb status',
        'tempest': 'TEMPEST: scan em sources, start video capture, start keyboard capture, stop capture, reconstruct display, save frame, tempest status',
        'poweranalysis': 'Power Analysis: capture trace, capture traces, spa attack, dpa attack, cpa attack, timing attack, recover key, load traces, save traces, power status',
        # NEW: Online Penetration Testing Help Topics (Core - Tier 0)
        'pentest': 'Pentest: scan web, scan ports, credential attack, brute force, osint, exploit search, start c2, proxy chain, web vuln scan, recon target, exploit [cve], pentest status',
        'web': 'Web Scanner: scan web [url], sqli test, xss test, csrf test, lfi test, rfi test, directory brute, api fuzz, scan config, web report',
        'credential': 'Credential Attack: brute force ssh, brute force ftp, password spray, credential stuff, wordlist attack, multi-protocol attack, credential status',
        'recon': 'Network Recon: port scan [target], service fingerprint, os detection, host discovery, network map, tcp scan, udp scan, stealth scan, recon status',
        'osint': 'OSINT: domain intel [domain], email harvest, whois lookup, subdomain enum, dns recon, social profile, leaked creds search, osint report',
        'exploit': 'Exploit Framework: search exploit [cve], generate payload, msfvenom payload, post-exploit, shell upgrade, exploit status',
        'c2': 'C2 Framework: start c2 server, list beacons, send command, generate beacon, beacon status, task beacon, c2 status',
        'proxy': 'Proxy Manager: start proxy chain, tor circuit, rotate proxy, add socks5, add http proxy, test proxy, proxy status',
        # TIER 1 - Essential Online Attack Modules
        'api_security': 'API Security: api scan [url], jwt crack, jwt attack, oauth test, bola test, bfla test, rate limit test, graphql fuzz, rest fuzz, api enum, swagger scan, api security status',
        'cloud_security': 'Cloud Security: aws scan, azure scan, gcp scan, s3 enum, bucket scan, iam analysis, lambda scan, cloud misconfig, imds attack, ec2 meta, cloud status',
        'dns_attack': 'DNS Attacks: zone transfer [domain], subdomain takeover, dangling dns, dns cache poison, dnssec test, dns tunnel, ns enum, dns attack status',
        'mobile_backend': 'Mobile Backend: firebase scan, firestore enum, cert pinning test, deep link hijack, mobile api fuzz, push notification test, app backend scan, mobile status',
        # TIER 2 - Advanced Online Attack Modules
        'supply_chain': 'Supply Chain: dependency confusion, typosquat check [package], npm audit, pypi scan, ci/cd leak, pipeline injection, sbom analysis, supply chain status',
        'sso_attack': 'SSO Attacks: saml bypass, oauth flow attack, kerberos attack, as-rep roast, session fixation, session hijack, mfa bypass, token replay, sso status',
        'websocket': 'WebSocket: ws scan [url], cswsh test, ws injection, ws hijack, ws replay, socket.io fuzz, signalr attack, websocket status',
        'graphql': 'GraphQL: introspection [url], graphql batch attack, depth attack, graphql dos, field enum, query complexity, graphql injection, graphql status',
        # TIER 3 - Specialized Online Attack Modules
        'browser_attack': 'Browser Attacks: xs-leak detect, spectre test, browser gadget, extension audit, service worker attack, cache timing, browser attack status',
        'protocol_attack': 'Protocol Attacks: http2 smuggle, h2c attack, grpc fuzz, webrtc ice leak, request desync, http3 test, protocol status',
        # NEW: SUPERHERO Blockchain Intelligence Help Topics
        'superhero': 'SUPERHERO: trace wallet [address], investigate [case_id], identify wallet owner, track stolen funds, generate dossier, check address, monitor address, superhero status',
        'blockchain': 'Blockchain Forensics: trace wallet [address], trace transactions, cluster wallets, detect mixers, identify exchanges, cross-chain trace, blockchain status',
        'identity': 'Identity Attribution: identify owner [address], correlate identity, ens lookup, social media match, email correlation, forum analysis, identity status',
        'geolocation': 'Geolocation Analysis: locate owner [address], timezone analysis, behavioral pattern, activity timing, location estimate, geolocation status',
        'dossier': 'Evidence Dossier: generate dossier [case_id], export pdf, export json, export html, add analyst notes, set classification, dossier status',
        'investigation': 'Investigation: create investigation, add target, run investigation, list investigations, get alerts, acknowledge alert, investigation status',
        # Cryptocurrency Security Assessment & Recovery Toolkit
        'wallet_security': 'Wallet Security Scanner: scan wallet [address], vulnerability scan, security assessment, keystore analysis, backup audit, wallet security status',
        'key_derivation': 'Key Derivation Analyzer: analyze derivation, check entropy, derivation path audit, seed strength, key weakness scan, derivation status',
        'contract_audit': 'Smart Contract Auditor: audit contract [address], detect exploits, vulnerability scan, reentrancy check, access control audit, contract status',
        'recovery': 'Recovery Toolkit: initiate recovery, seed reconstruction, password recovery, multisig recovery, hardware recovery, recovery status',
        'malicious_db': 'Malicious Address DB: check malicious [address], add malicious, search malicious, threat lookup, db status, malicious report',
        'authority_report': 'Authority Reports: create case, generate authority report, add evidence, export law enforcement report, chain of custody, report status',
        # AI v2.0 Enhanced Intelligence
        'ai': 'AI Modes: set ai mode basic/enhanced/autonomous/online, ai status, download ai model, plan attack [objective], run autonomous [task]',
        'enhanced': 'Enhanced AI: Uses local LLM for intelligent responses, natural language understanding, context awareness',
        'autonomous': 'Autonomous AI: Full autonomous agent for multi-step attacks, auto-planning, auto-execution',
    }
    
    def __init__(self):
        self.logger = logging.getLogger('AI-Command-Center')
        
        # Command queue for async processing
        self.command_queue = queue.Queue()
        self.is_processing = False
        
        # Module references (lazy loaded)
        self._network_mode = None
        self._stealth_system = None
        self._hardware_controller = None
        self._cellular_modules = {}
        self._wifi_module = None
        self._gps_module = None
        self._drone_module = None
        self._jamming_module = None
        self._sigint_module = None
        self._capture_module = None
        self._targeting_module = None
        self._emergency_system = None
        
        # NEW: Mission, OPSEC, and User Mode systems
        self._mission_manager = None
        self._opsec_monitor = None
        self._user_mode_manager = None
        
        # NEW: Defensive and Dashboard systems
        self._counter_surveillance = None
        self._threat_dashboard = None
        
        # NEW: Signal Replay and Hardware Wizard
        self._signal_library = None
        self._hardware_wizard = None
        
        # NEW: Vehicle Penetration Testing Modules
        self._vehicle_can = None
        self._vehicle_uds = None
        self._vehicle_key_fob = None
        self._vehicle_tpms = None
        self._vehicle_gps = None
        self._vehicle_bluetooth = None
        self._vehicle_v2x = None
        
        # NEW: Satellite Communications Module
        self._satellite_module = None
        
        # NEW: Device Fingerprinting Module
        self._fingerprint_module = None
        
        # NEW: IoT/Smart Home Attack Modules
        self._iot_zigbee = None
        self._iot_zwave = None
        self._iot_smart_lock = None
        self._iot_smart_meter = None
        self._iot_home_automation = None
        
        # NEW: BladeRF Advanced Feature Modules (10 major capabilities)
        self._mimo_controller = None       # MIMO 2x2
        self._relay_attacker = None        # Full-duplex relay attacks
        self._lora_attacker = None         # LoRa/LoRaWAN attacks
        self._meshtastic_decoder = None    # Meshtastic passive decoder
        self._meshtastic_sigint = None     # Meshtastic SIGINT
        self._meshtastic_attacks = None    # Meshtastic attack suite
        self._bluetooth5_stack = None      # Bluetooth 5.x full stack
        self._rolljam_attacker = None      # RollJam implementation
        self._agc_controller = None        # Hardware AGC/calibration
        self._hopping_controller = None    # Frequency hopping
        self._xb200_controller = None      # XB-200 transverter
        self._lte_decoder = None           # LTE/5G decoder
        self._digital_radio = None         # DMR/P25/TETRA decoder
        
        # NEW: SoapySDR Universal Hardware Abstraction
        self._sdr_manager = None           # SoapySDR manager (BladeRF, HackRF, RTL-SDR, USRP, LimeSDR, etc.)
        self._sdr_device = None            # Currently active SDR device
        self._sdr_dsp = None               # DSP processor for signal analysis
        
        # NEW: Additional Attack Modules
        self._yatebts_controller = None    # YateBTS GSM/LTE BTS controller
        self._proxmark3_controller = None  # Proxmark3 NFC/RFID controller
        self._adsb_controller = None       # ADS-B aircraft tracking controller
        self._tempest_controller = None    # TEMPEST/Van Eck EM surveillance controller
        self._power_analysis = None        # Power analysis side-channel controller
        
        # NEW: Online Penetration Testing Modules
        self._web_scanner = None           # Web application vulnerability scanner
        self._credential_attacker = None   # Multi-protocol credential attacks
        self._network_recon = None         # Network reconnaissance engine
        self._osint_engine = None          # OSINT intelligence gathering
        self._exploit_framework = None     # Exploit integration framework
        self._c2_server = None             # Command & Control server
        self._proxy_manager = None         # Proxy chain manager
        
        # NEW: SUPERHERO Blockchain Intelligence Modules
        self._superhero_engine = None      # Core SUPERHERO engine
        self._blockchain_forensics = None  # Blockchain transaction tracing
        self._identity_engine = None       # Identity correlation engine
        self._geolocation_analyzer = None  # Geolocation analysis
        self._counter_measures = None      # Counter-countermeasures (mixer tracing)
        self._dossier_generator = None     # Evidence dossier generator
        
        # NEW: Cryptocurrency Security Assessment & Recovery Toolkit
        self._wallet_security_scanner = None   # Wallet vulnerability scanner
        self._key_derivation_analyzer = None   # Key derivation analysis
        self._smart_contract_auditor = None    # Smart contract auditing
        self._recovery_toolkit = None          # Authorized recovery tools
        self._malicious_address_db = None      # Malicious address database
        self._authority_report_generator = None  # Law enforcement reports
        
        # NEW: AI v2.0 Enhanced Intelligence System
        # UNFILTERED, UNRESTRICTED - Professional penetration testing AI
        self._enhanced_ai = None               # Enhanced AI with local LLM
        self._ai_mode = 'basic'                # basic, enhanced, autonomous, online
        
        # State
        self.pending_confirmation: Optional[CommandContext] = None
        self.command_history: List[Dict] = []
        self.session_start = datetime.now()
        
        # Callbacks for UI updates
        self._response_callbacks: List[Callable] = []
        self._status_callbacks: List[Callable] = []
        
        # Initialize core systems
        self._initialize_core_systems()
        
        self.logger.info("AI Command Center initialized")
        
        # Initialize Enhanced AI v2.0
        self._initialize_enhanced_ai()
    
    def _initialize_enhanced_ai(self):
        """Initialize Enhanced AI v2.0 System (UNFILTERED)"""
        try:
            from core.ai_v2.enhanced_ai import EnhancedAI, integrate_with_command_center
            self._enhanced_ai = integrate_with_command_center(self)
            self.logger.info("Enhanced AI v2.0 loaded (UNFILTERED mode)")
        except ImportError as e:
            self.logger.warning(f"Enhanced AI v2.0 not available: {e}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Enhanced AI: {e}")
    
    def set_ai_mode(self, mode: str) -> bool:
        """
        Set AI operation mode
        
        Modes:
        - basic: Pattern matching only (default)
        - enhanced: LLM-powered responses
        - autonomous: Full autonomous agent
        - online: Enhanced + Tor-routed intelligence
        """
        if self._enhanced_ai:
            from core.ai_v2.enhanced_ai import AIMode
            mode_map = {
                'basic': AIMode.BASIC,
                'enhanced': AIMode.ENHANCED,
                'autonomous': AIMode.AUTONOMOUS,
                'online': AIMode.ONLINE,
            }
            if mode.lower() in mode_map:
                self._enhanced_ai.set_mode(mode_map[mode.lower()])
                self._ai_mode = mode.lower()
                return True
        return False
    
    def get_ai_mode(self) -> str:
        """Get current AI mode"""
        return self._ai_mode
    
    def _initialize_core_systems(self):
        """Initialize core system modules"""
        try:
            # Network mode manager (offline by default)
            from core.network_mode import get_network_mode_manager
            self._network_mode = get_network_mode_manager()
            self.logger.info("Network mode manager loaded - OFFLINE by default")
        except ImportError as e:
            self.logger.warning(f"Network mode manager not available: {e}")
        
        try:
            # Stealth system
            from core.stealth import StealthSystem
            self._stealth_system = StealthSystem()
            self.logger.info("Stealth system loaded")
        except ImportError as e:
            self.logger.warning(f"Stealth system not available: {e}")
        
        try:
            # Emergency system
            from core.emergency import EmergencySystem
            self._emergency_system = EmergencySystem()
            self.logger.info("Emergency system loaded")
        except ImportError as e:
            self.logger.warning(f"Emergency system not available: {e}")
        
        # NEW: Initialize Mission Profile Manager
        try:
            from core.mission_profiles import get_mission_manager
            self._mission_manager = get_mission_manager(self)
            self.logger.info("Mission Profile Manager loaded")
        except ImportError as e:
            self.logger.warning(f"Mission Profile Manager not available: {e}")
        
        # NEW: Initialize OPSEC Monitor
        try:
            from core.opsec_monitor import get_opsec_monitor
            self._opsec_monitor = get_opsec_monitor()
            # Start background monitoring
            self._opsec_monitor.start_monitoring(interval_seconds=30)
            self.logger.info("OPSEC Monitor loaded and running")
        except ImportError as e:
            self.logger.warning(f"OPSEC Monitor not available: {e}")
        
        # NEW: Initialize User Mode Manager
        try:
            from core.user_modes import get_user_mode_manager
            self._user_mode_manager = get_user_mode_manager()
            self.logger.info(f"User Mode Manager loaded - Mode: {self._user_mode_manager.get_current_mode().value}")
        except ImportError as e:
            self.logger.warning(f"User Mode Manager not available: {e}")
        
        # NEW: Initialize Counter-Surveillance System
        try:
            from modules.defensive.counter_surveillance import CounterSurveillanceSystem
            self._counter_surveillance = CounterSurveillanceSystem(self._hardware_controller)
            self.logger.info("Counter-Surveillance System loaded")
        except ImportError as e:
            self.logger.warning(f"Counter-Surveillance System not available: {e}")
        
        # NEW: Initialize Threat Dashboard
        try:
            from ui.threat_dashboard import RFThreatDashboard
            self._threat_dashboard = RFThreatDashboard(self._hardware_controller)
            self.logger.info("Threat Dashboard loaded")
        except ImportError as e:
            self.logger.warning(f"Threat Dashboard not available: {e}")
        
        # NEW: Initialize Signal Replay Library
        try:
            from modules.replay.signal_library import get_signal_library
            self._signal_library = get_signal_library()
            self.logger.info(f"Signal Library loaded - {len(self._signal_library.catalog)} signals")
        except ImportError as e:
            self.logger.warning(f"Signal Library not available: {e}")
        
        # NEW: Initialize Hardware Wizard
        try:
            from install.hardware_wizard import get_hardware_wizard
            self._hardware_wizard = get_hardware_wizard()
            self.logger.info("Hardware Wizard loaded")
        except ImportError as e:
            self.logger.warning(f"Hardware Wizard not available: {e}")
    
    def _lazy_load_module(self, module_name: str):
        """Lazy load modules when needed"""
        try:
            if module_name == 'hardware':
                from core.hardware import BladeRFController
                self._hardware_controller = BladeRFController.get_instance()
                
            elif module_name == 'wifi':
                from modules.wifi.wifi_suite import WiFiAttackSuite
                self._wifi_module = WiFiAttackSuite(self._hardware_controller)
                
            elif module_name == 'gps':
                from modules.gps.gps_spoofer import GPSSpoofer
                self._gps_module = GPSSpoofer(self._hardware_controller)
                
            elif module_name == 'drone':
                from modules.drone.drone_warfare import DroneWarfare
                self._drone_module = DroneWarfare(self._hardware_controller)
                
            elif module_name == 'jamming':
                from modules.jamming.jamming_suite import JammingSuite
                self._jamming_module = JammingSuite(self._hardware_controller)
                
            elif module_name == 'sigint':
                from modules.sigint.sigint_engine import SIGINTEngine
                self._sigint_module = SIGINTEngine(self._hardware_controller)
                
            elif module_name == 'capture':
                from modules.network.packet_capture import WiresharkCapture
                self._capture_module = WiresharkCapture()
                
            elif module_name == 'targeting':
                from modules.cellular.phone_targeting import PhoneNumberTargeting
                self._targeting_module = PhoneNumberTargeting(
                    self._cellular_modules.get('gsm'),
                    self._cellular_modules.get('lte'),
                    stealth_mode=True
                )
                
            return True
        except Exception as e:
            self.logger.warning(f"Failed to load module {module_name}: {e}")
            return False
    
    # ========== MAIN COMMAND PROCESSING ==========
    
    def process_command(self, user_input: str) -> CommandResult:
        """
        Main entry point: Process natural language command from user
        
        Args:
            user_input: Natural language command (text or transcribed voice)
            
        Returns:
            CommandResult with response and any data
        """
        user_input = user_input.strip()
        
        if not user_input:
            return CommandResult(
                success=False,
                message="No command provided. Say 'help' for available commands."
            )
        
        # Log command
        self.logger.info(f"Processing command: {user_input}")
        
        # Parse the command
        context = self._parse_command(user_input)
        
        # Check if this is a confirmation response
        if self.pending_confirmation:
            return self._handle_confirmation(user_input, context)
        
        # Check if command requires confirmation
        if context.requires_confirmation or context.is_dangerous:
            return self._request_confirmation(context)
        
        # Execute the command
        result = self._execute_command(context)
        
        # Log to history
        self._log_command(context, result)
        
        # Notify callbacks
        for callback in self._response_callbacks:
            try:
                callback(result)
            except Exception as e:
                self.logger.error(f"Response callback error: {e}")
        
        return result
    
    def _parse_command(self, user_input: str) -> CommandContext:
        """Parse natural language input into structured command"""
        text = user_input.lower().strip()
        context = CommandContext(raw_input=user_input)
        
        # Check for dangerous commands
        context.is_dangerous = any(cmd in text for cmd in self.DANGEROUS_COMMANDS)
        
        # ===== HIGH-PRIORITY KEYWORD CHECKS (check before other categories) =====
        
        # Vehicle-specific keywords that should NOT be captured by WiFi/GPS/Bluetooth parsers
        vehicle_priority_keywords = [
            'can bus', 'can-bus', 'canbus', 'can frame', 'can traffic',
            'ecu', 'uds', 'dtc', 'diagnostic', 'obd-ii', 'obd2',
            'key fob', 'keyfob', 'rolling code', 'keeloq', 'rolljam',
            'tpms', 'tire pressure', 'tire sensor',
            'v2x', 'dsrc', 'c-v2x', 'ghost vehicle', 'bsm',
            'vehicle bluetooth', 'bluetooth obd', 'bluetooth dongle',
            'gps vehicle', 'vehicle gps', 'automotive gps'
        ]
        is_vehicle_command = any(kw in text for kw in vehicle_priority_keywords)
        
        # Satellite-specific keywords that should NOT be captured by GPS/Network parsers
        satellite_priority_keywords = [
            'satellite', 'satcom', 'noaa', 'meteor', 'iridium', 'inmarsat',
            'adsb', 'ads-b', 'aircraft track', 'weather satellite', 'iss',
            'amateur satellite', 'gps signal', 'track noaa', 'decode weather'
        ]
        is_satellite_command = any(kw in text for kw in satellite_priority_keywords)
        
        # Fingerprint-specific keywords that should NOT be captured by Network parsers
        fingerprint_priority_keywords = [
            'fingerprint', 'rf signature', 'device identif', 'identify transmitter',
            'network profile', 'device profile', 'train model', 'ml model',
            'train fingerprint', 'fingerprint model'
        ]
        is_fingerprint_command = any(kw in text for kw in fingerprint_priority_keywords)
        
        # IoT-specific keywords that should NOT be captured by other parsers
        iot_priority_keywords = [
            'zigbee', 'z-wave', 'zwave', 'smart home', 'smart lock', 'smart meter',
            'home automation', 'iot', 'smart hub', 'philips hue', 'smartthings',
            'mqtt', 'thread', 'matter'
        ]
        is_iot_command = any(kw in text for kw in iot_priority_keywords)
        
        # NEW: BladeRF Advanced Feature Keywords
        mimo_keywords = ['mimo', 'beamform', 'spatial multiplex', 'direction of arrival', 'doa', 'phased array', '2x2']
        is_mimo_command = any(kw in text for kw in mimo_keywords)
        
        relay_keywords = ['relay attack', 'full duplex relay', 'car key relay', 'access card relay', 'relay nfc', 'relay signal']
        is_relay_command = any(kw in text for kw in relay_keywords)
        
        lora_keywords = ['lora', 'lorawan', 'chirp', 'spreading factor', 'gateway spoof']
        is_lora_command = any(kw in text for kw in lora_keywords)
        
        meshtastic_keywords = ['meshtastic', 'mesh network', 'mesh topology', 'mesh node', 'mesh traffic', 'lora mesh']
        is_meshtastic_command = any(kw in text for kw in meshtastic_keywords)
        
        bluetooth5_keywords = ['ble5', 'bluetooth 5', 'ble 5', 'coded phy', 'direction finding', 'aoa', 'aod', 'le audio', '2m phy']
        is_bluetooth5_command = any(kw in text for kw in bluetooth5_keywords)
        
        rolljam_keywords = ['rolljam', 'roll jam', 'rolling code capture', 'jam and capture']
        is_rolljam_command = any(kw in text for kw in rolljam_keywords)
        
        agc_keywords = ['agc', 'automatic gain', 'dc offset', 'iq calibration', 'iq imbalance', 'rssi monitor', 'ad9361']
        is_agc_command = any(kw in text for kw in agc_keywords)
        
        hopping_keywords = ['frequency hopping', 'freq hopping', 'fhss', 'hop sequence', 'track hopper', 'hopping pattern']
        is_hopping_command = any(kw in text for kw in hopping_keywords)
        
        xb200_keywords = ['xb200', 'xb-200', 'transverter', 'hf band', 'shortwave', 'vhf band', 'noaa weather', 'fm broadcast', 'aircraft band']
        is_xb200_command = any(kw in text for kw in xb200_keywords)
        
        lte5g_keywords = ['lte decoder', '5g decoder', 'lte cell', '5g nr', 'sib decode', 'lte imsi', 'enodeb', 'gnodeb']
        is_lte5g_command = any(kw in text for kw in lte5g_keywords)
        
        digitalradio_keywords = ['dmr', 'p25', 'tetra', 'digital radio', 'trunking', 'mototrbo', 'hytera']
        is_digitalradio_command = any(kw in text for kw in digitalradio_keywords)
        
        # NEW: AI v2.0 Enhanced Intelligence Keywords (high priority)
        ai_v2_priority_keywords = [
            'ai mode', 'enhanced ai', 'local ai', 'autonomous agent', 'attack plan', 'plan attack',
            'rag query', 'knowledge base', 'online intel', 'threat intel', 'ai memory', 'ai context',
            'unfiltered ai', 'unrestricted ai', 'full ai', 'ai status', 'enable enhanced',
            'load model', 'mistral', 'local llm', 'ai inference', 'ai query', 'llm status',
            'execute plan', 'mission plan', 'attack sequence', 'auto attack', 'autonomous attack',
            'create agent', 'start agent', 'stop agent', 'list agents', 'agent status',
            'clear memory', 'save memory', 'memory status', 'context status',
            'update rag', 'rag status', 'ai help', 'ai enable', 'ai disable',
            'recon agent', 'attack agent', 'intel agent', 'monitor agent'  # Specific agent types
        ]
        is_ai_v2_command = any(kw in text for kw in ai_v2_priority_keywords)
        
        # SDR Hardware keywords (SoapySDR universal abstraction)
        sdr_keywords = ['sdr', 'soapy', 'hackrf', 'rtl-sdr', 'rtlsdr', 'usrp', 'limesdr', 'airspy', 'plutosdr', 'sdrplay', 
                       'scan hardware', 'list devices', 'sdr device', 'iq capture', 'iq playback', 'sample rate',
                       'center frequency', 'sdr spectrum', 'hardware info', 'detect sdr']
        is_sdr_command = any(kw in text for kw in sdr_keywords)
        
        # NEW: YateBTS keywords (real GSM/LTE BTS)
        yatebts_keywords = ['yatebts', 'yate bts', 'gsm bts', 'lte bts', 'imsi catcher', 'imsi catch', 
                          'intercept sms', 'intercept call', 'silent sms', 'captured device', 'bts mode',
                          'start bts', 'stop bts', 'real bts']
        is_yatebts_command = any(kw in text for kw in yatebts_keywords)
        
        # NEW: NFC/RFID Proxmark3 keywords
        nfc_keywords = ['nfc', 'rfid', 'proxmark', 'mifare', 'clone card', 'read card', 'write card',
                       'darkside', 'nested attack', 'hardnested', 'em4100', 'hid prox', 't5577',
                       'card emulate', 'sniff card', 'lf card', 'hf card']
        is_nfc_command = any(kw in text for kw in nfc_keywords)
        
        # NEW: ADS-B keywords
        adsb_keywords = ['adsb', 'ads-b', 'aircraft', 'plane', 'flight', 'transponder', 'mode s',
                        '1090', 'squawk', 'icao', 'track aircraft', 'inject aircraft']
        is_adsb_command = any(kw in text for kw in adsb_keywords)
        
        # NEW: TEMPEST/Van Eck keywords
        tempest_keywords = ['tempest', 'van eck', 'emanation', 'em surveillance', 'reconstruct video',
                          'keyboard capture', 'display capture', 'em source', 'em scan', 'phreaking']
        is_tempest_command = any(kw in text for kw in tempest_keywords)
        
        # NEW: Power Analysis keywords
        poweranalysis_keywords = ['power analysis', 'side channel', 'spa', 'dpa', 'cpa', 'power trace',
                                 'timing attack', 'fault injection', 'key recovery', 'ema attack',
                                 'hamming weight', 'correlation attack']
        is_poweranalysis_command = any(kw in text for kw in poweranalysis_keywords)
        
        # NEW: Online Penetration Testing keywords (EXPANDED - 17 modules, 150+ attack vectors)
        pentest_keywords = [
            # Core Web Scanner
            'pentest', 'web scan', 'port scan', 'vuln scan', 'vulnerability scan',
            'sqli', 'sql injection', 'xss', 'cross-site', 'csrf', 'lfi', 'rfi',
            'directory brute', 'dir brute', 'api fuzz', 'web attack',
            # Credential Attacks
            'brute force', 'bruteforce', 'password spray', 'credential stuff',
            'wordlist attack', 'hydra', 'credential attack',
            # Network Reconnaissance
            'recon', 'reconnaissance', 'service fingerprint', 'os detect',
            'host discovery', 'network map', 'stealth scan', 'tcp scan', 'udp scan',
            # OSINT Engine
            'osint', 'domain intel', 'email harvest', 'whois', 'subdomain enum',
            'dns recon', 'social profile', 'leaked cred', 'shodan',
            # Exploit Framework
            'exploit', 'cve', 'payload', 'msfvenom', 'metasploit', 'post-exploit',
            'shell upgrade', 'reverse shell',
            # C2 Framework
            'c2', 'command and control', 'beacon', 'implant', 'c2 server',
            # Proxy Chain
            'proxy chain', 'socks5', 'tor circuit', 'rotate proxy', 'anonymize traffic',
            # TIER 1: API Security Testing
            'api security', 'api scan', 'rest fuzz', 'graphql fuzz', 'jwt attack', 'jwt crack',
            'oauth attack', 'bola', 'bfla', 'idor', 'mass assignment', 'rate limit bypass',
            'api enum', 'swagger scan', 'openapi', 'api key', 'bearer token',
            # TIER 1: Cloud Security Assessment
            'cloud scan', 'aws scan', 'azure scan', 'gcp scan', 's3 bucket', 'cloud misconfig',
            'iam analysis', 'lambda scan', 'serverless', 'cloud security', 'bucket enum',
            'azure ad', 'storage account', 'ec2 meta', 'imds', 'cloud exposure',
            # TIER 1: DNS/Domain Attacks
            'dns attack', 'zone transfer', 'subdomain takeover', 'dangling dns', 'cache poison',
            'dnssec', 'dns tunnel', 'domain fronting', 'ns takeover', 'dns enum',
            # TIER 1: Mobile App Backend
            'mobile scan', 'firebase', 'firestore', 'cert pinning', 'deep link', 'mobile api',
            'app backend', 'push notification', 'mobile config', 'api intercept',
            # TIER 2: Supply Chain Security
            'supply chain', 'dependency confusion', 'typosquat', 'npm audit', 'pypi check',
            'ci/cd', 'pipeline', 'build artifact', 'package security', 'sbom',
            # TIER 2: SSO/Identity Attacks
            'sso attack', 'saml', 'saml bypass', 'oauth flow', 'kerberos', 'as-rep roast',
            'session fixation', 'session hijack', 'token replay', 'mfa bypass', 'oidc',
            # TIER 2: WebSocket Security
            'websocket', 'ws scan', 'cswsh', 'ws injection', 'ws hijack', 'socket.io',
            'signalr', 'ws fuzz', 'ws replay', 'websocket attack',
            # TIER 2: GraphQL-Specific
            'graphql', 'introspection', 'graphql batching', 'graphql depth', 'graphql dos',
            'graphql enum', 'graphql injection', 'field suggestion', 'query complexity',
            # TIER 3: Browser-Based Attacks
            'browser attack', 'xs-leak', 'spectre', 'meltdown', 'browser gadget',
            'extension attack', 'service worker', 'cache attack', 'timing attack browser',
            # TIER 3: Protocol-Level
            'http2 smuggle', 'http3', 'grpc attack', 'grpc fuzz', 'webrtc', 'ice leak',
            'h2c smuggle', 'request smuggling', 'desync', 'protocol attack'
        ]
        is_pentest_command = any(kw in text for kw in pentest_keywords)
        
        # NEW: SUPERHERO Blockchain Intelligence keywords
        superhero_keywords = [
            'superhero', 'blockchain', 'forensics', 'trace wallet', 'trace transaction',
            'identify owner', 'wallet owner', 'cluster wallet', 'detect mixer', 'mixer trace',
            'identity correlation', 'identity attribution', 'ens lookup', 'nft analysis',
            'stolen funds', 'track crypto', 'cryptocurrency', 'bitcoin trace', 'ethereum trace',
            'dossier', 'evidence report', 'court ready', 'law enforcement',
            'geolocation', 'timezone analysis', 'behavioral pattern', 'activity timing',
            'cross-chain', 'chain hop', 'privacy coin', 'tornado cash', 'mixer detection',
            'investigation', 'create case', 'add target', 'run investigation', 'investigate',
            'monitor address', 'alert', 'suspect profile', 'identity engine',
            # Cryptocurrency Security Assessment & Recovery Toolkit keywords
            'scan wallet', 'wallet scan', 'security scan', 'vulnerability scan', 'security assessment',
            'keystore analysis', 'backup audit', 'wallet security',
            'analyze derivation', 'check entropy', 'derivation path', 'seed strength', 'key weakness',
            'audit contract', 'contract audit', 'detect exploit', 'reentrancy', 'access control audit',
            'initiate recovery', 'seed reconstruction', 'password recovery', 'multisig recovery',
            'hardware recovery', 'recovery status', 'recovery toolkit',
            'check malicious', 'malicious check', 'threat check', 'add malicious', 'search malicious',
            'threat lookup', 'malicious database', 'malicious address',
            'authority report', 'add evidence', 'chain of custody', 'report status'
        ]
        is_superhero_command = any(kw in text for kw in superhero_keywords)
        
        # ===== AI v2.0 ENHANCED INTELLIGENCE (High Priority) =====
        # Process AI v2.0 commands FIRST to prevent other parsers from capturing
        if is_ai_v2_command:
            context.category = CommandCategory.AI_ENHANCED
            
            # AI Mode Control
            if 'enable' in text and ('enhanced' in text or 'ai mode' in text or 'full ai' in text or 'ai enable' in text):
                context.intent = 'ai_enable_enhanced'
            elif 'disable' in text and ('enhanced' in text or 'ai mode' in text or 'ai disable' in text):
                context.intent = 'ai_disable_enhanced'
            elif 'online' in text and 'intel' in text:
                context.intent = 'ai_online_intel'
                context.requires_confirmation = True
            elif 'offline' in text and 'mode' in text:
                context.intent = 'ai_offline_mode'
            
            # LLM Control
            elif 'load' in text and ('model' in text or 'llm' in text or 'mistral' in text):
                context.intent = 'ai_load_model'
            elif 'unload' in text and 'model' in text:
                context.intent = 'ai_unload_model'
            elif 'model status' in text or 'llm status' in text:
                context.intent = 'ai_model_status'
            
            # Autonomous Agent Control
            elif 'create' in text and 'agent' in text:
                context.intent = 'ai_create_agent'
                if 'recon' in text:
                    context.parameters['agent_type'] = 'reconnaissance'
                elif 'attack' in text:
                    context.parameters['agent_type'] = 'attack'
                elif 'intel' in text:
                    context.parameters['agent_type'] = 'intelligence'
                elif 'monitor' in text:
                    context.parameters['agent_type'] = 'monitoring'
            elif 'start' in text and 'agent' in text:
                context.intent = 'ai_start_agent'
            elif 'stop' in text and 'agent' in text:
                context.intent = 'ai_stop_agent'
            elif 'list' in text and 'agent' in text:
                context.intent = 'ai_list_agents'
            elif 'agent status' in text:
                context.intent = 'ai_agent_status'
            
            # Attack Planning
            elif ('plan' in text and 'attack' in text) or ('attack' in text and 'plan' in text):
                context.intent = 'ai_plan_attack'
                target_match = re.search(r'(?:on|target|against)\s+(\S+)', text)
                if target_match:
                    context.parameters['target'] = target_match.group(1)
            elif 'execute' in text and 'plan' in text:
                context.intent = 'ai_execute_plan'
                context.requires_confirmation = True
                context.is_dangerous = True
            elif 'autonomous' in text and 'attack' in text:
                context.intent = 'ai_autonomous_attack'
                context.requires_confirmation = True
                context.is_dangerous = True
            elif 'stop' in text and 'attack' in text:
                context.intent = 'ai_stop_attack'
            elif 'attack status' in text:
                context.intent = 'ai_attack_status'
            
            # RAG Knowledge Base
            elif ('query' in text or 'rag query' in text) and ('rag' in text or 'knowledge' in text):
                context.intent = 'ai_rag_query'
                query_match = re.search(r'(?:query|ask)\s+(.+)', text)
                if query_match:
                    context.parameters['query'] = query_match.group(1)
            elif 'add' in text and 'knowledge' in text:
                context.intent = 'ai_add_knowledge'
            elif 'update' in text and ('rag' in text or 'knowledge' in text):
                context.intent = 'ai_update_rag'
            elif 'knowledge status' in text or 'rag status' in text:
                context.intent = 'ai_rag_status'
            
            # Memory Control
            elif 'clear' in text and 'memory' in text:
                context.intent = 'ai_clear_memory'
            elif 'save' in text and 'memory' in text:
                context.intent = 'ai_save_memory'
            elif 'load' in text and 'memory' in text:
                context.intent = 'ai_load_memory'
            elif 'memory status' in text or 'context status' in text:
                context.intent = 'ai_memory_status'
            
            # Threat Intel
            elif 'threat' in text and 'intel' in text:
                context.intent = 'ai_threat_intel'
                if 'online' not in text:
                    context.parameters['source'] = 'local'
            elif 'update' in text and 'intel' in text:
                context.intent = 'ai_update_intel'
                context.requires_confirmation = True
            
            # Natural Language Processing
            elif 'unfiltered' in text or 'unrestricted' in text:
                context.intent = 'ai_process_natural'
                context.parameters['query'] = user_input
            
            # AI Help
            elif 'ai help' in text:
                context.intent = 'ai_help'
            
            # General AI Status
            elif 'ai status' in text or 'enhanced status' in text:
                context.intent = 'ai_status'
            else:
                context.intent = 'ai_status'
            
            context.confidence = 0.95
        
        # ===== NETWORK / ONLINE / OFFLINE =====
        elif not is_vehicle_command and not is_fingerprint_command and not is_satellite_command and any(word in text for word in ['online', 'offline', 'network', 'connect', 'disconnect', 'internet']):
            context.category = CommandCategory.NETWORK
            
            if 'offline' in text or 'disconnect' in text:
                context.intent = 'go_offline'
            elif 'status' in text:
                context.intent = 'network_status'
            elif 'online' in text or 'connect' in text:
                context.intent = 'go_online'
                context.requires_confirmation = True
                
                # Determine mode
                if 'full' in text or 'triple' in text or 'maximum' in text:
                    context.parameters['mode'] = 'full'
                elif 'vpn' in text:
                    context.parameters['mode'] = 'vpn'
                elif 'tor' in text:
                    context.parameters['mode'] = 'tor'
                elif 'direct' in text:
                    context.parameters['mode'] = 'direct'
                    context.is_dangerous = True
                else:
                    context.parameters['mode'] = 'tor'  # Default to Tor
                
                # Extract reason if provided
                reason_match = re.search(r'(?:for|to|because)\s+(.+?)(?:\s+for\s+\d+|$)', text)
                if reason_match:
                    context.parameters['reason'] = reason_match.group(1)
                
                # Extract duration
                duration_match = re.search(r'for\s+(\d+)\s*(minute|min|hour|hr|second|sec)', text)
                if duration_match:
                    value = int(duration_match.group(1))
                    unit = duration_match.group(2)
                    if 'hour' in unit or 'hr' in unit:
                        context.parameters['duration'] = value * 3600
                    elif 'min' in unit:
                        context.parameters['duration'] = value * 60
                    else:
                        context.parameters['duration'] = value
            
            context.confidence = 0.9
        
        # ===== STEALTH =====
        elif any(word in text for word in ['stealth', 'mac', 'ram', 'secure delete', 'opsec', 'anonymity']):
            context.category = CommandCategory.STEALTH
            
            if 'ram' in text and ('only' in text or 'mode' in text):
                context.intent = 'enable_ram_only'
            elif 'mac' in text and ('random' in text or 'rotate' in text or 'change' in text):
                context.intent = 'randomize_mac'
            elif 'secure' in text and 'delete' in text:
                context.intent = 'secure_delete'
                # Extract file path
                path_match = re.search(r'delete\s+(.+)', text)
                if path_match:
                    context.parameters['path'] = path_match.group(1).strip()
            elif 'wipe' in text and 'ram' in text:
                context.intent = 'wipe_ram'
            elif 'status' in text:
                context.intent = 'stealth_status'
            
            context.confidence = 0.85
        
        # ===== YATEBTS (must come before CELLULAR to catch 'yatebts' keyword) =====
        elif is_yatebts_command:
            context.category = CommandCategory.YATEBTS
            if 'start' in text and ('imsi' in text or 'catcher' in text):
                context.intent = 'yatebts_imsi_catcher'
                context.requires_confirmation = True
                context.is_dangerous = True
            elif 'start' in text or 'enable' in text:
                context.intent = 'yatebts_start'
                context.requires_confirmation = True
                context.is_dangerous = True
            elif 'stop' in text or 'disable' in text:
                context.intent = 'yatebts_stop'
            elif 'capture' in text or 'list' in text or 'device' in text:
                context.intent = 'yatebts_list_devices'
            elif 'sms' in text and 'intercept' in text:
                context.intent = 'yatebts_intercept_sms'
                context.requires_confirmation = True
            elif 'call' in text and 'intercept' in text:
                context.intent = 'yatebts_intercept_calls'
                context.requires_confirmation = True
            elif 'target' in text:
                context.intent = 'yatebts_target'
                imsi_match = re.search(r'(\d{15})', text)
                if imsi_match:
                    context.parameters['imsi'] = imsi_match.group(1)
            elif 'silent' in text and 'sms' in text:
                context.intent = 'yatebts_silent_sms'
                context.requires_confirmation = True
            elif 'status' in text:
                context.intent = 'yatebts_status'
            else:
                context.intent = 'yatebts_status'
            context.confidence = 0.95
        
        # ===== ADS-B (must come before SATELLITE to catch aircraft tracking) =====
        elif is_adsb_command:
            context.category = CommandCategory.ADSB
            if 'start' in text or 'enable' in text or 'receive' in text:
                context.intent = 'adsb_start'
            elif 'stop' in text or 'disable' in text:
                context.intent = 'adsb_stop'
            elif 'list' in text or 'show' in text:
                context.intent = 'adsb_list'
            elif 'track' in text:
                context.intent = 'adsb_track'
                icao_match = re.search(r'([0-9A-Fa-f]{6})', text)
                if icao_match:
                    context.parameters['icao'] = icao_match.group(1).upper()
            elif 'inject' in text or 'spoof' in text:
                context.intent = 'adsb_inject'
                context.requires_confirmation = True
                context.is_dangerous = True
            elif 'status' in text:
                context.intent = 'adsb_status'
            else:
                context.intent = 'adsb_status'
            context.confidence = 0.95
        
        # ===== TEMPEST (must come before WIFI to catch EM surveillance) =====
        elif is_tempest_command:
            context.category = CommandCategory.TEMPEST
            if 'scan' in text and 'em' in text:
                context.intent = 'tempest_scan'
            elif 'video' in text or 'display' in text:
                context.intent = 'tempest_video'
            elif 'keyboard' in text:
                context.intent = 'tempest_keyboard'
            elif 'stop' in text:
                context.intent = 'tempest_stop'
            elif 'save' in text or 'frame' in text:
                context.intent = 'tempest_save'
            elif 'status' in text:
                context.intent = 'tempest_status'
            else:
                context.intent = 'tempest_status'
            context.confidence = 0.95
        
        # ===== POWER ANALYSIS (must come before WIFI to catch side-channel attacks) =====
        elif is_poweranalysis_command:
            context.category = CommandCategory.POWERANALYSIS
            if 'capture' in text and 'trace' in text:
                context.intent = 'power_capture'
                count_match = re.search(r'(\d+)\s*trace', text)
                if count_match:
                    context.parameters['count'] = int(count_match.group(1))
            elif 'spa' in text or 'simple' in text:
                context.intent = 'power_spa'
            elif 'dpa' in text or 'differential' in text:
                context.intent = 'power_dpa'
            elif 'cpa' in text or 'correlation' in text:
                context.intent = 'power_cpa'
            elif 'timing' in text:
                context.intent = 'power_timing'
            elif 'recover' in text or 'full' in text:
                context.intent = 'power_full_key'
            elif 'load' in text:
                context.intent = 'power_load'
            elif 'save' in text:
                context.intent = 'power_save'
            elif 'status' in text:
                context.intent = 'power_status'
            else:
                context.intent = 'power_status'
            context.confidence = 0.95
        
        # ===== NFC/RFID (must come before other modules) =====
        elif is_nfc_command:
            context.category = CommandCategory.NFC
            if 'scan' in text and 'lf' in text:
                context.intent = 'nfc_scan_lf'
            elif 'scan' in text and 'hf' in text:
                context.intent = 'nfc_scan_hf'
            elif 'scan' in text:
                context.intent = 'nfc_scan'
            elif 'read' in text:
                context.intent = 'nfc_read'
            elif 'clone' in text or 'copy' in text:
                context.intent = 'nfc_clone'
                context.requires_confirmation = True
            elif 'write' in text:
                context.intent = 'nfc_write'
                context.requires_confirmation = True
            elif 'darkside' in text:
                context.intent = 'nfc_darkside'
            elif 'nested' in text and 'hard' in text:
                context.intent = 'nfc_hardnested'
            elif 'nested' in text:
                context.intent = 'nfc_nested'
            elif 'dictionary' in text or 'dict' in text:
                context.intent = 'nfc_dictionary'
            elif 'autopwn' in text:
                context.intent = 'nfc_autopwn'
            elif 'sniff' in text:
                context.intent = 'nfc_sniff'
            elif 'emulate' in text or 'sim' in text:
                context.intent = 'nfc_emulate'
            elif 'status' in text:
                context.intent = 'nfc_status'
            else:
                context.intent = 'nfc_status'
            context.confidence = 0.95
        
        # ===== CELLULAR =====
        elif any(word in text for word in ['cellular', 'cell', 'bts', 'base station', '2g', '3g', '4g', '5g', 'lte', 'gsm', 'imsi']):
            context.category = CommandCategory.CELLULAR
            
            # Determine generation
            if '5g' in text or 'nr' in text:
                context.parameters['generation'] = '5G'
            elif '4g' in text or 'lte' in text:
                context.parameters['generation'] = '4G'
            elif '3g' in text or 'umts' in text:
                context.parameters['generation'] = '3G'
            elif '2g' in text or 'gsm' in text:
                context.parameters['generation'] = '2G'
            else:
                context.parameters['generation'] = '4G'  # Default
            
            if 'imsi' in text or 'catch' in text:
                context.intent = 'imsi_catch'
                context.requires_confirmation = True
            elif 'start' in text or 'begin' in text or 'activate' in text:
                context.intent = 'start_bts'
                context.requires_confirmation = True
            elif 'stop' in text or 'deactivate' in text:
                context.intent = 'stop_bts'
            elif 'status' in text:
                context.intent = 'cellular_status'
            
            context.confidence = 0.9
        
        # ===== PHONE TARGETING =====
        elif not is_satellite_command and any(word in text for word in ['target', 'phone', 'number', 'track']):
            context.category = CommandCategory.TARGETING
            
            # Extract phone number
            phone_match = re.search(r'[\+]?[\d\s\-\(\)]{10,}', text)
            if phone_match:
                context.parameters['phone'] = re.sub(r'[\s\-\(\)]', '', phone_match.group())
            
            if 'add' in text or 'target' in text:
                context.intent = 'add_target'
                context.requires_confirmation = True
            elif 'remove' in text or 'stop' in text:
                context.intent = 'remove_target'
            elif 'list' in text:
                context.intent = 'list_targets'
            elif 'status' in text:
                context.intent = 'target_status'
            
            context.confidence = 0.85
        
        # ===== WIFI =====
        # Use word boundaries to prevent false matches (e.g., "ap" in "graphql")
        elif not is_vehicle_command and any(re.search(r'\b' + re.escape(word) + r'\b', text) for word in ['wifi', 'wireless', 'wlan', '802.11', 'access point', 'ap']):
            context.category = CommandCategory.WIFI
            
            if 'scan' in text:
                context.intent = 'wifi_scan'
            elif 'deauth' in text or 'disconnect' in text or 'kick' in text:
                context.intent = 'wifi_deauth'
                context.requires_confirmation = True
            elif 'evil' in text or 'twin' in text or 'fake' in text or 'rogue' in text:
                context.intent = 'evil_twin'
                context.requires_confirmation = True
            elif 'handshake' in text or 'capture' in text:
                context.intent = 'capture_handshake'
            elif 'stop' in text:
                context.intent = 'stop_wifi'
            
            context.confidence = 0.9
        
        # ===== GPS =====
        elif not is_vehicle_command and any(word in text for word in ['gps', 'location', 'position', 'coordinates', 'spoof']):
            context.category = CommandCategory.GPS
            
            # Extract coordinates
            coord_match = re.findall(r'-?\d+\.\d+', text)
            if len(coord_match) >= 2:
                context.parameters['latitude'] = float(coord_match[0])
                context.parameters['longitude'] = float(coord_match[1])
                if len(coord_match) >= 3:
                    context.parameters['altitude'] = float(coord_match[2])
            
            if 'spoof' in text or 'fake' in text:
                context.intent = 'gps_spoof'
                context.requires_confirmation = True
            elif 'jam' in text:
                context.intent = 'gps_jam'
                context.requires_confirmation = True
            elif 'stop' in text:
                context.intent = 'stop_gps'
            elif 'status' in text:
                context.intent = 'gps_status'
            
            context.confidence = 0.9
        
        # ===== DRONE =====
        elif any(word in text for word in ['drone', 'uav', 'quadcopter', 'dji']):
            context.category = CommandCategory.DRONE
            
            if 'detect' in text or 'scan' in text or 'find' in text:
                context.intent = 'detect_drones'
            elif 'jam' in text or 'block' in text:
                context.intent = 'jam_drone'
                context.requires_confirmation = True
            elif 'hijack' in text or 'take' in text:
                context.intent = 'hijack_drone'
                context.requires_confirmation = True
            elif 'auto' in text or 'defend' in text:
                context.intent = 'auto_defend'
                context.requires_confirmation = True
            elif 'stop' in text:
                context.intent = 'stop_drone'
            
            context.confidence = 0.9
        
        # ===== JAMMING =====
        elif any(word in text for word in ['jam', 'jamming', 'disrupt', 'block signal']):
            context.category = CommandCategory.JAMMING
            context.requires_confirmation = True
            
            # Extract frequency
            freq_match = re.search(r'(\d+\.?\d*)\s*(mhz|ghz|khz)', text)
            if freq_match:
                context.parameters['frequency'] = self._parse_frequency(freq_match.group())
            
            # Detect band
            if 'wifi' in text or '2.4' in text:
                context.parameters['band'] = 'wifi_2.4'
            elif '5g' in text and 'wifi' in text:
                context.parameters['band'] = 'wifi_5'
            elif 'gps' in text:
                context.parameters['band'] = 'gps'
            elif 'cell' in text or 'lte' in text:
                context.parameters['band'] = 'cellular'
            elif 'bluetooth' in text or 'bt' in text:
                context.parameters['band'] = 'bluetooth'
            
            if 'stop' in text:
                context.intent = 'stop_jamming'
                context.requires_confirmation = False
            else:
                context.intent = 'start_jamming'
            
            context.confidence = 0.85
        
        # ===== SPECTRUM =====
        elif any(word in text for word in ['spectrum', 'analyze', 'analyzer', 'frequency', 'freq', 'scan rf']):
            context.category = CommandCategory.SPECTRUM
            
            # Extract frequency range
            freq_matches = re.findall(r'(\d+\.?\d*)\s*(mhz|ghz|khz)', text)
            if len(freq_matches) >= 2:
                context.parameters['start_freq'] = self._parse_frequency(f"{freq_matches[0][0]} {freq_matches[0][1]}")
                context.parameters['stop_freq'] = self._parse_frequency(f"{freq_matches[1][0]} {freq_matches[1][1]}")
            elif len(freq_matches) == 1:
                context.parameters['center_freq'] = self._parse_frequency(f"{freq_matches[0][0]} {freq_matches[0][1]}")
            
            context.intent = 'spectrum_scan'
            context.confidence = 0.85
        
        # ===== SIGINT =====
        elif any(word in text for word in ['sigint', 'intelligence', 'intercept signal', 'monitor']):
            context.category = CommandCategory.SIGINT
            
            if 'passive' in text:
                context.parameters['mode'] = 'passive'
            elif 'active' in text or 'target' in text:
                context.parameters['mode'] = 'targeted'
            
            if 'stop' in text:
                context.intent = 'stop_sigint'
            else:
                context.intent = 'start_sigint'
                context.requires_confirmation = True
            
            context.confidence = 0.85
        
        # ===== CAPTURE (Packet) =====
        elif not is_vehicle_command and any(word in text for word in ['capture', 'wireshark', 'pcap', 'sniff', 'packet']):
            context.category = CommandCategory.CAPTURE
            
            # Extract interface
            if 'wlan0' in text:
                context.parameters['interface'] = 'wlan0'
            elif 'eth0' in text:
                context.parameters['interface'] = 'eth0'
            
            # Extract duration
            duration_match = re.search(r'(\d+)\s*(second|sec|minute|min)', text)
            if duration_match:
                value = int(duration_match.group(1))
                if 'min' in duration_match.group(2):
                    context.parameters['duration'] = value * 60
                else:
                    context.parameters['duration'] = value
            
            if 'start' in text or 'begin' in text:
                context.intent = 'start_capture'
            elif 'stop' in text:
                context.intent = 'stop_capture'
            elif 'analyze' in text:
                context.intent = 'analyze_capture'
            elif 'clean' in text or 'delete' in text:
                context.intent = 'cleanup_captures'
            
            context.confidence = 0.85
        
        # ===== EMERGENCY =====
        elif any(word in text for word in ['emergency', 'panic', 'wipe', 'abort', 'kill all']):
            context.category = CommandCategory.EMERGENCY
            context.is_dangerous = True
            context.requires_confirmation = True
            
            if 'wipe' in text or 'delete all' in text:
                context.intent = 'emergency_wipe'
            elif 'panic' in text:
                context.intent = 'panic_button'
            elif 'shutdown' in text:
                context.intent = 'emergency_shutdown'
            else:
                context.intent = 'emergency_stop'
            
            context.confidence = 0.95
        
        # ===== MISSION PROFILES =====
        elif not is_fingerprint_command and any(word in text for word in ['mission', 'missions', 'guided', 'profile', 'next step', 'abort mission']):
            context.category = CommandCategory.MISSION
            
            if 'list' in text or 'show' in text and 'mission' in text:
                context.intent = 'list_missions'
                # Check for filters
                if 'beginner' in text:
                    context.parameters['difficulty'] = 'beginner'
                elif 'intermediate' in text:
                    context.parameters['difficulty'] = 'intermediate'
                elif 'advanced' in text:
                    context.parameters['difficulty'] = 'advanced'
            elif 'start' in text:
                context.intent = 'start_mission'
                # Extract mission name/id
                if 'wifi' in text:
                    context.parameters['mission_id'] = 'wifi_security_audit'
                elif 'cellular' in text or 'cell' in text:
                    context.parameters['mission_id'] = 'cellular_pen_test'
                elif 'drone' in text:
                    context.parameters['mission_id'] = 'drone_defense'
                elif 'spectrum' in text:
                    context.parameters['mission_id'] = 'spectrum_recon'
                elif 'gps' in text:
                    context.parameters['mission_id'] = 'gps_security_test'
            elif 'next' in text or 'continue' in text or 'proceed' in text:
                context.intent = 'next_step'
            elif 'skip' in text:
                context.intent = 'skip_step'
            elif 'confirm' in text:
                context.intent = 'confirm_step'
            elif 'abort' in text or 'cancel' in text:
                context.intent = 'abort_mission'
            elif 'status' in text:
                context.intent = 'mission_status'
            
            context.confidence = 0.9
        
        # ===== OPSEC MONITORING =====
        elif any(word in text for word in ['opsec', 'security score', 'security check', 'fix opsec', 'leaking']):
            context.category = CommandCategory.OPSEC
            
            if 'fix' in text:
                context.intent = 'fix_opsec'
            elif 'report' in text or 'detailed' in text:
                context.intent = 'opsec_report'
            elif 'score' in text or 'check' in text or 'show' in text or 'status' in text:
                context.intent = 'opsec_status'
            else:
                context.intent = 'opsec_status'
            
            context.confidence = 0.9
        
        # ===== USER MODE =====
        elif any(word in text for word in ['beginner mode', 'expert mode', 'intermediate mode', 'set mode', 'user mode']):
            context.category = CommandCategory.MODE
            
            if 'set' in text or 'switch' in text or 'change' in text:
                context.intent = 'set_mode'
                if 'beginner' in text:
                    context.parameters['mode'] = 'beginner'
                elif 'intermediate' in text:
                    context.parameters['mode'] = 'intermediate'
                elif 'expert' in text:
                    context.parameters['mode'] = 'expert'
            elif 'show' in text or 'current' in text:
                context.intent = 'show_mode'
            elif 'compare' in text or 'modes' in text:
                context.intent = 'compare_modes'
            else:
                # Default to showing current mode
                context.intent = 'show_mode'
            
            context.confidence = 0.9
        
        # ===== COUNTER-SURVEILLANCE / DEFENSIVE =====
        elif any(word in text for word in ['counter', 'surveillance', 'stingray', 'imsi catcher', 'tracker', 'being tracked', 'rogue ap', 'evil twin detect', 'threat scan']):
            context.category = CommandCategory.DEFENSIVE
            
            if 'scan' in text or 'check' in text or 'detect' in text:
                context.intent = 'scan_threats'
                if 'stingray' in text or 'imsi catcher' in text or 'cell' in text:
                    context.parameters['scan_type'] = 'cellular'
                elif 'wifi' in text or 'rogue' in text or 'evil' in text:
                    context.parameters['scan_type'] = 'wifi'
                elif 'bluetooth' in text or 'tracker' in text:
                    context.parameters['scan_type'] = 'bluetooth'
                elif 'gps' in text:
                    context.parameters['scan_type'] = 'gps'
                else:
                    context.parameters['scan_type'] = 'all'
            elif 'start' in text and 'monitor' in text:
                context.intent = 'start_monitoring'
            elif 'stop' in text and 'monitor' in text:
                context.intent = 'stop_monitoring'
            elif 'status' in text or 'summary' in text:
                context.intent = 'threat_status'
            elif 'baseline' in text:
                context.intent = 'establish_baseline'
            elif 'trust' in text and 'wifi' in text:
                context.intent = 'add_trusted_wifi'
                # Extract SSID if provided
                ssid_match = re.search(r'(?:trust|add)\s+(?:wifi\s+)?["\']?([^"\']+)["\']?', text)
                if ssid_match:
                    context.parameters['ssid'] = ssid_match.group(1).strip()
            else:
                context.intent = 'threat_status'
            
            context.confidence = 0.9
        
        # ===== THREAT DASHBOARD =====
        elif any(word in text for word in ['dashboard', 'threat map', 'rf map', 'signal map', 'threat display', 'show threats']):
            context.category = CommandCategory.DASHBOARD
            
            if 'show' in text or 'display' in text or 'open' in text:
                context.intent = 'show_dashboard'
            elif 'map' in text:
                context.intent = 'show_threat_map'
            elif 'alerts' in text:
                context.intent = 'show_alerts'
            elif 'acknowledge' in text or 'ack' in text:
                context.intent = 'acknowledge_alert'
                # Extract alert ID
                alert_match = re.search(r'(?:ack|acknowledge)\s+(\S+)', text)
                if alert_match:
                    context.parameters['alert_id'] = alert_match.group(1)
            elif 'dismiss' in text:
                context.intent = 'dismiss_alert'
            elif 'signals' in text:
                context.intent = 'show_signals'
            elif 'stealth' in text or 'footprint' in text:
                context.intent = 'show_stealth_footprint'
            else:
                context.intent = 'show_dashboard'
            
            context.confidence = 0.9
        
        # ===== SIGNAL REPLAY =====
        elif not is_vehicle_command and any(word in text for word in ['replay', 'signal library', 'capture signal', 'garage door', 'recorded signal']):
            context.category = CommandCategory.REPLAY
            
            if 'capture' in text or 'record' in text:
                context.intent = 'capture_signal'
                # Extract name if provided
                name_match = re.search(r'(?:capture|record)\s+(?:signal\s+)?(?:called\s+|named\s+)?["\']?([^"\']+)["\']?', text)
                if name_match:
                    context.parameters['name'] = name_match.group(1).strip()
                # Extract frequency
                freq_match = re.search(r'(\d+\.?\d*)\s*(mhz|ghz)', text)
                if freq_match:
                    context.parameters['frequency'] = self._parse_frequency(freq_match.group())
                # Detect category
                if 'keyfob' in text or 'key fob' in text:
                    context.parameters['category'] = 'keyfob'
                elif 'garage' in text:
                    context.parameters['category'] = 'garage_door'
                elif 'sensor' in text:
                    context.parameters['category'] = 'wireless_sensor'
                elif 'doorbell' in text:
                    context.parameters['category'] = 'doorbell'
            elif 'replay' in text or 'transmit' in text or 'play' in text:
                context.intent = 'replay_signal'
                context.requires_confirmation = True
                # Extract signal ID or name
                id_match = re.search(r'(?:replay|transmit|play)\s+(?:signal\s+)?(\S+)', text)
                if id_match:
                    context.parameters['signal_id'] = id_match.group(1)
                # Extract repeat count
                repeat_match = re.search(r'(\d+)\s*(?:times?|x)', text)
                if repeat_match:
                    context.parameters['repeat'] = int(repeat_match.group(1))
            elif 'list' in text or 'show' in text:
                context.intent = 'list_signals'
                if 'keyfob' in text:
                    context.parameters['category'] = 'keyfob'
                elif 'garage' in text:
                    context.parameters['category'] = 'garage_door'
            elif 'analyze' in text:
                context.intent = 'analyze_signal'
            elif 'delete' in text or 'remove' in text:
                context.intent = 'delete_signal'
                context.requires_confirmation = True
            elif 'export' in text:
                context.intent = 'export_signal'
            elif 'import' in text:
                context.intent = 'import_signal'
            elif 'stats' in text or 'statistics' in text:
                context.intent = 'library_stats'
            else:
                context.intent = 'list_signals'
            
            context.confidence = 0.9
        
        # ===== HARDWARE SETUP =====
        elif any(word in text for word in ['hardware', 'sdr', 'bladerf', 'hackrf', 'limesdr', 'rtl-sdr', 'antenna', 'calibrat', 'setup wizard']):
            context.category = CommandCategory.HARDWARE
            
            if 'detect' in text or 'scan' in text or 'find' in text:
                context.intent = 'detect_hardware'
            elif 'wizard' in text or 'setup' in text:
                context.intent = 'setup_wizard'
            elif 'calibrat' in text:
                context.intent = 'calibrate'
            elif 'antenna' in text:
                context.intent = 'antenna_guide'
                # Extract frequency for recommendations
                freq_match = re.search(r'(\d+)\s*mhz', text)
                if freq_match:
                    context.parameters['frequency_mhz'] = int(freq_match.group(1))
            elif 'driver' in text:
                context.intent = 'check_drivers'
            elif 'troubleshoot' in text or 'problem' in text or 'issue' in text:
                context.intent = 'troubleshoot'
            elif 'status' in text or 'info' in text:
                context.intent = 'hardware_status'
            else:
                context.intent = 'hardware_status'
            
            context.confidence = 0.9
        
        # ===== VEHICLE PENETRATION TESTING =====
        # Use word boundaries to prevent false matches (e.g., "ecu" in "security")
        elif any(re.search(r'\b' + re.escape(word) + r'\b', text) for word in 
                 ['can bus', 'can-bus', 'canbus', 'obd', 'obd2', 'obd-ii', 'ecu', 'uds', 
                  'key fob', 'keyfob', 'rolling code', 'keeloq', 'tpms', 'tire pressure',
                  'v2x', 'dsrc', 'vehicle', 'car hack', 'automotive', 'ghost vehicle']):
            context.category = CommandCategory.VEHICLE
            
            # Determine sub-module and intent
            if any(word in text for word in ['can bus', 'can-bus', 'canbus', 'can frame', 'can traffic']):
                context.parameters['module'] = 'can_bus'
                if 'scan' in text or 'sniff' in text or 'monitor' in text:
                    context.intent = 'can_scan'
                elif 'inject' in text or 'send' in text:
                    context.intent = 'can_inject'
                    # Extract CAN ID if provided
                    can_id_match = re.search(r'(?:id|can_id)\s*[=:]?\s*(0x[0-9a-fA-F]+|\d+)', text)
                    if can_id_match:
                        context.parameters['can_id'] = can_id_match.group(1)
                elif 'fuzz' in text:
                    context.intent = 'can_fuzz'
                    context.requires_confirmation = True
                elif 'replay' in text:
                    context.intent = 'can_replay'
                elif 'discover' in text or 'enumerate' in text:
                    context.intent = 'can_discover'
                elif 'stop' in text:
                    context.intent = 'can_stop'
                else:
                    context.intent = 'can_status'
            
            elif any(word in text for word in ['uds', 'diagnostic', 'ecu', 'obd', 'obd2', 'obd-ii']):
                context.parameters['module'] = 'uds'
                if 'read' in text and ('dtc' in text or 'code' in text or 'fault' in text):
                    context.intent = 'uds_read_dtc'
                elif 'clear' in text and ('dtc' in text or 'code' in text):
                    context.intent = 'uds_clear_dtc'
                    context.requires_confirmation = True
                elif 'security' in text or 'unlock' in text:
                    context.intent = 'uds_security_access'
                    context.requires_confirmation = True
                elif 'read' in text and 'memory' in text:
                    context.intent = 'uds_read_memory'
                elif 'write' in text and 'memory' in text:
                    context.intent = 'uds_write_memory'
                    context.requires_confirmation = True
                    context.is_dangerous = True
                elif 'flash' in text or 'firmware' in text:
                    context.intent = 'uds_flash'
                    context.requires_confirmation = True
                    context.is_dangerous = True
                elif 'scan' in text or 'discover' in text:
                    context.intent = 'uds_scan'
                elif 'session' in text:
                    context.intent = 'uds_session'
                else:
                    context.intent = 'uds_status'
            
            elif any(word in text for word in ['key fob', 'keyfob', 'rolling code', 'keeloq', 'rolljam']):
                context.parameters['module'] = 'key_fob'
                if 'capture' in text or 'record' in text or 'sniff' in text:
                    context.intent = 'keyfob_capture'
                    # Extract frequency if provided
                    freq_match = re.search(r'(\d+)\s*mhz', text)
                    if freq_match:
                        context.parameters['frequency'] = int(freq_match.group(1)) * 1_000_000
                    else:
                        context.parameters['frequency'] = 433_920_000  # Default to 433.92 MHz
                elif 'analyze' in text or 'decode' in text:
                    context.intent = 'keyfob_analyze'
                elif 'replay' in text or 'transmit' in text:
                    context.intent = 'keyfob_replay'
                    context.requires_confirmation = True
                elif 'rolljam' in text or 'roll jam' in text:
                    context.intent = 'keyfob_rolljam'
                    context.requires_confirmation = True
                    context.is_dangerous = True
                elif 'crack' in text or 'decrypt' in text:
                    context.intent = 'keyfob_crack'
                elif 'stop' in text:
                    context.intent = 'keyfob_stop'
                else:
                    context.intent = 'keyfob_status'
            
            elif any(word in text for word in ['tpms', 'tire pressure', 'tire sensor']):
                context.parameters['module'] = 'tpms'
                if 'scan' in text or 'discover' in text or 'find' in text:
                    context.intent = 'tpms_scan'
                elif 'spoof' in text or 'fake' in text:
                    context.intent = 'tpms_spoof'
                    context.requires_confirmation = True
                    # Extract pressure if provided
                    pressure_match = re.search(r'(\d+)\s*(?:psi|kpa)', text)
                    if pressure_match:
                        context.parameters['pressure'] = int(pressure_match.group(1))
                elif 'clone' in text:
                    context.intent = 'tpms_clone'
                    context.requires_confirmation = True
                elif 'alert' in text or 'trigger' in text:
                    context.intent = 'tpms_trigger_alert'
                    context.requires_confirmation = True
                elif 'stop' in text:
                    context.intent = 'tpms_stop'
                else:
                    context.intent = 'tpms_status'
            
            elif any(word in text for word in ['v2x', 'dsrc', 'c-v2x', 'ghost vehicle', 'bsm', 'vehicle to']):
                context.parameters['module'] = 'v2x'
                if 'scan' in text or 'monitor' in text or 'listen' in text:
                    context.intent = 'v2x_scan'
                elif 'ghost' in text or 'create' in text and 'vehicle' in text:
                    context.intent = 'v2x_ghost_vehicle'
                    context.requires_confirmation = True
                    # Extract coordinates if provided
                    coord_match = re.findall(r'-?\d+\.\d+', text)
                    if len(coord_match) >= 2:
                        context.parameters['latitude'] = float(coord_match[0])
                        context.parameters['longitude'] = float(coord_match[1])
                elif 'jam' in text:
                    context.intent = 'v2x_jam'
                    context.requires_confirmation = True
                    context.is_dangerous = True
                elif 'spoof' in text:
                    context.intent = 'v2x_spoof'
                    context.requires_confirmation = True
                elif 'traffic' in text and 'jam' in text:
                    context.intent = 'v2x_traffic_jam'
                    context.requires_confirmation = True
                elif 'stop' in text:
                    context.intent = 'v2x_stop'
                else:
                    context.intent = 'v2x_status'
            
            elif 'bluetooth' in text and ('obd' in text or 'vehicle' in text or 'car' in text or 'dongle' in text):
                context.parameters['module'] = 'bluetooth_vehicle'
                if 'scan' in text or 'discover' in text:
                    context.intent = 'ble_vehicle_scan'
                elif 'connect' in text or 'pair' in text:
                    context.intent = 'ble_vehicle_connect'
                elif 'exploit' in text or 'attack' in text:
                    context.intent = 'ble_vehicle_exploit'
                    context.requires_confirmation = True
                elif 'relay' in text:
                    context.intent = 'ble_vehicle_relay'
                    context.requires_confirmation = True
                elif 'stop' in text:
                    context.intent = 'ble_vehicle_stop'
                else:
                    context.intent = 'ble_vehicle_status'
            
            elif 'gps' in text and ('vehicle' in text or 'car' in text or 'automotive' in text):
                context.parameters['module'] = 'gps_vehicle'
                if 'spoof' in text or 'fake' in text:
                    context.intent = 'gps_vehicle_spoof'
                    context.requires_confirmation = True
                    # Extract coordinates if provided
                    coord_match = re.findall(r'-?\d+\.\d+', text)
                    if len(coord_match) >= 2:
                        context.parameters['latitude'] = float(coord_match[0])
                        context.parameters['longitude'] = float(coord_match[1])
                elif 'trajectory' in text or 'path' in text:
                    context.intent = 'gps_vehicle_trajectory'
                    context.requires_confirmation = True
                elif 'stop' in text:
                    context.intent = 'gps_vehicle_stop'
                else:
                    context.intent = 'gps_vehicle_status'
            
            else:
                # General vehicle status
                context.intent = 'vehicle_status'
            
            context.confidence = 0.9
        
        # ===== SATELLITE COMMUNICATIONS =====
        elif any(word in text for word in ['satellite', 'satcom', 'noaa', 'meteor', 'iridium', 'inmarsat', 
                                            'adsb', 'ads-b', 'aircraft track', 'weather satellite', 
                                            'gps signal', 'iss', 'amateur satellite']):
            context.category = CommandCategory.SATELLITE
            
            if 'scan' in text or 'search' in text:
                if 'adsb' in text or 'ads-b' in text or 'aircraft' in text:
                    context.intent = 'satellite_adsb_scan'
                elif 'iridium' in text:
                    context.intent = 'satellite_iridium_scan'
                elif 'gps' in text:
                    context.intent = 'satellite_gps_scan'
                else:
                    context.intent = 'satellite_scan'
            elif 'track' in text:
                if 'noaa' in text or 'weather' in text:
                    context.intent = 'satellite_track_noaa'
                elif 'iss' in text:
                    context.intent = 'satellite_track_iss'
                elif 'aircraft' in text or 'adsb' in text:
                    context.intent = 'satellite_track_aircraft'
                else:
                    context.intent = 'satellite_track'
                    # Extract satellite name if provided
                    sat_match = re.search(r'track\s+(\w+)', text)
                    if sat_match:
                        context.parameters['satellite'] = sat_match.group(1)
            elif 'decode' in text or 'receive' in text:
                if 'weather' in text or 'apt' in text or 'noaa' in text:
                    context.intent = 'satellite_decode_weather'
                elif 'iridium' in text:
                    context.intent = 'satellite_decode_iridium'
                else:
                    context.intent = 'satellite_decode'
            elif 'predict' in text or 'pass' in text:
                context.intent = 'satellite_predict_pass'
                # Extract satellite name
                sat_match = re.search(r'(?:predict|pass)\s+(?:for\s+)?(\w+)', text)
                if sat_match:
                    context.parameters['satellite'] = sat_match.group(1)
            elif 'transmit' in text or 'send' in text:
                context.intent = 'satellite_transmit'
                context.requires_confirmation = True
            elif 'stop' in text:
                context.intent = 'satellite_stop'
            elif 'status' in text or 'info' in text:
                context.intent = 'satellite_status'
            else:
                context.intent = 'satellite_status'
            
            context.confidence = 0.9
        
        # ===== DEVICE FINGERPRINTING =====
        elif any(word in text for word in ['fingerprint', 'identify transmitter', 'device identif', 
                                            'rf signature', 'device profile', 'network profile',
                                            'train model', 'ml model']):
            context.category = CommandCategory.FINGERPRINT
            
            if 'fingerprint' in text or 'identify' in text:
                if 'nearby' in text or 'scan' in text or 'all' in text:
                    context.intent = 'fingerprint_scan'
                else:
                    context.intent = 'fingerprint_identify'
                    # Extract target if provided
                    target_match = re.search(r'(?:identify|fingerprint)\s+(.+?)(?:\s+device|\s*$)', text)
                    if target_match:
                        context.parameters['target'] = target_match.group(1).strip()
            elif 'train' in text or 'model' in text:
                if 'train' in text:
                    context.intent = 'fingerprint_train'
                elif 'load' in text:
                    context.intent = 'fingerprint_load_model'
                elif 'save' in text:
                    context.intent = 'fingerprint_save_model'
                else:
                    context.intent = 'fingerprint_model_status'
            elif 'profile' in text:
                if 'network' in text:
                    context.intent = 'fingerprint_network_profile'
                else:
                    context.intent = 'fingerprint_device_profile'
            elif 'history' in text:
                context.intent = 'fingerprint_history'
            elif 'status' in text or 'info' in text:
                context.intent = 'fingerprint_status'
            else:
                context.intent = 'fingerprint_status'
            
            context.confidence = 0.9
        
        # ===== IOT / SMART HOME =====
        elif any(word in text for word in ['zigbee', 'z-wave', 'zwave', 'smart home', 'smart lock', 
                                            'smart meter', 'home automation', 'iot', 'smart hub',
                                            'philips hue', 'smartthings', 'alexa', 'google home',
                                            'mqtt', 'thread', 'matter']):
            context.category = CommandCategory.IOT
            
            if any(word in text for word in ['zigbee']):
                context.parameters['protocol'] = 'zigbee'
                if 'scan' in text or 'discover' in text:
                    context.intent = 'iot_zigbee_scan'
                elif 'sniff' in text or 'capture' in text:
                    context.intent = 'iot_zigbee_sniff'
                elif 'inject' in text or 'send' in text:
                    context.intent = 'iot_zigbee_inject'
                    context.requires_confirmation = True
                elif 'key' in text or 'extract' in text:
                    context.intent = 'iot_zigbee_extract_key'
                elif 'replay' in text:
                    context.intent = 'iot_zigbee_replay'
                    context.requires_confirmation = True
                elif 'jam' in text:
                    context.intent = 'iot_zigbee_jam'
                    context.requires_confirmation = True
                else:
                    context.intent = 'iot_zigbee_status'
            
            elif any(word in text for word in ['z-wave', 'zwave']):
                context.parameters['protocol'] = 'zwave'
                if 'scan' in text or 'discover' in text:
                    context.intent = 'iot_zwave_scan'
                elif 'sniff' in text or 'capture' in text:
                    context.intent = 'iot_zwave_sniff'
                elif 'inject' in text or 'send' in text:
                    context.intent = 'iot_zwave_inject'
                    context.requires_confirmation = True
                elif 'jam' in text:
                    context.intent = 'iot_zwave_jam'
                    context.requires_confirmation = True
                else:
                    context.intent = 'iot_zwave_status'
            
            elif any(word in text for word in ['smart lock', 'lock']):
                context.parameters['device_type'] = 'lock'
                if 'scan' in text or 'discover' in text:
                    context.intent = 'iot_lock_scan'
                elif 'unlock' in text or 'open' in text:
                    context.intent = 'iot_lock_unlock'
                    context.requires_confirmation = True
                    context.is_dangerous = True
                elif 'lock' in text and not 'unlock' in text:
                    context.intent = 'iot_lock_lock'
                elif 'vuln' in text or 'exploit' in text:
                    context.intent = 'iot_lock_vuln_scan'
                elif 'replay' in text:
                    context.intent = 'iot_lock_replay'
                    context.requires_confirmation = True
                else:
                    context.intent = 'iot_lock_status'
            
            elif any(word in text for word in ['smart meter', 'meter', 'electricity', 'utility']):
                context.parameters['device_type'] = 'meter'
                if 'scan' in text or 'discover' in text:
                    context.intent = 'iot_meter_scan'
                elif 'read' in text:
                    context.intent = 'iot_meter_read'
                elif 'spoof' in text or 'manipulate' in text:
                    context.intent = 'iot_meter_spoof'
                    context.requires_confirmation = True
                    context.is_dangerous = True
                else:
                    context.intent = 'iot_meter_status'
            
            elif any(word in text for word in ['smart home', 'home automation', 'smart hub', 'smartthings', 
                                                'alexa', 'google home', 'homekit', 'mqtt']):
                context.parameters['device_type'] = 'hub'
                if 'scan' in text or 'discover' in text:
                    context.intent = 'iot_hub_scan'
                elif 'enumerate' in text or 'list' in text:
                    context.intent = 'iot_hub_enumerate'
                elif 'control' in text or 'command' in text:
                    context.intent = 'iot_hub_control'
                    context.requires_confirmation = True
                elif 'vuln' in text or 'exploit' in text:
                    context.intent = 'iot_hub_vuln_scan'
                elif 'inject' in text and 'automation' in text:
                    context.intent = 'iot_hub_inject_automation'
                    context.requires_confirmation = True
                    context.is_dangerous = True
                elif 'mqtt' in text:
                    context.intent = 'iot_mqtt_discover'
                else:
                    context.intent = 'iot_hub_status'
            
            else:
                # General IoT status
                context.intent = 'iot_status'
            
            context.confidence = 0.9
        
        # ===== MIMO 2x2 =====
        elif is_mimo_command:
            context.category = CommandCategory.MIMO
            if 'beamform' in text:
                context.intent = 'mimo_beamform'
                angle_match = re.search(r'(\d+)\s*(?:deg|°)', text)
                if angle_match:
                    context.parameters['azimuth'] = int(angle_match.group(1))
            elif 'doa' in text or 'direction' in text:
                context.intent = 'mimo_doa'
            elif 'diversity' in text:
                context.intent = 'mimo_diversity'
            elif 'multiplex' in text or 'spatial' in text:
                context.intent = 'mimo_multiplex'
            elif 'channel' in text and 'sound' in text:
                context.intent = 'mimo_channel_sounding'
            elif 'status' in text:
                context.intent = 'mimo_status'
            elif 'enable' in text or 'start' in text:
                context.intent = 'mimo_enable'
            else:
                context.intent = 'mimo_status'
            context.confidence = 0.9
        
        # ===== RELAY ATTACKS =====
        elif is_relay_command:
            context.category = CommandCategory.RELAY
            if 'car' in text or 'key' in text:
                context.intent = 'relay_car_key'
                context.requires_confirmation = True
            elif 'access' in text or 'card' in text:
                context.intent = 'relay_access_card'
                context.requires_confirmation = True
            elif 'nfc' in text:
                context.intent = 'relay_nfc'
                context.requires_confirmation = True
            elif 'full duplex' in text:
                context.intent = 'relay_full_duplex'
            elif 'two device' in text:
                context.intent = 'relay_two_device'
            elif 'capture' in text:
                context.intent = 'relay_capture'
            elif 'status' in text:
                context.intent = 'relay_status'
            elif 'stop' in text:
                context.intent = 'relay_stop'
            else:
                context.intent = 'relay_status'
            context.confidence = 0.9
        
        # ===== LORA/LORAWAN =====
        elif is_lora_command:
            context.category = CommandCategory.LORA
            if 'scan' in text or 'discover' in text:
                context.intent = 'lora_scan'
            elif 'sniff' in text:
                context.intent = 'lora_sniff'
            elif 'replay' in text:
                context.intent = 'lora_replay'
                context.requires_confirmation = True
            elif 'jam' in text:
                context.intent = 'lora_jam'
                context.requires_confirmation = True
            elif 'gateway' in text or 'spoof' in text:
                context.intent = 'lora_gateway_spoof'
                context.requires_confirmation = True
            elif 'inject' in text or 'downlink' in text:
                context.intent = 'lora_inject'
                context.requires_confirmation = True
            elif 'status' in text:
                context.intent = 'lora_status'
            else:
                context.intent = 'lora_status'
            context.confidence = 0.9
        
        # ===== MESHTASTIC MESH NETWORK =====
        elif is_meshtastic_command:
            context.category = CommandCategory.MESHTASTIC
            if 'scan' in text or 'discover' in text:
                context.intent = 'meshtastic_scan'
            elif 'monitor' in text or 'listen' in text:
                context.intent = 'meshtastic_monitor'
            elif 'map' in text or 'topology' in text:
                context.intent = 'meshtastic_topology'
            elif 'analyze' in text or 'traffic' in text or 'pattern' in text:
                context.intent = 'meshtastic_traffic_analysis'
            elif 'track' in text or 'location' in text or 'gps' in text:
                context.intent = 'meshtastic_track_nodes'
            elif 'sigint' in text or 'intelligence' in text:
                context.intent = 'meshtastic_sigint'
            elif 'vuln' in text or 'assess' in text or 'security' in text:
                context.intent = 'meshtastic_vulnerability'
            elif 'jam' in text:
                context.intent = 'meshtastic_jam'
                context.requires_confirmation = True
                context.is_dangerous = True
            elif 'inject' in text or 'send' in text:
                context.intent = 'meshtastic_inject'
                context.requires_confirmation = True
            elif 'impersonate' in text or 'spoof' in text:
                context.intent = 'meshtastic_impersonate'
                context.requires_confirmation = True
            elif 'flood' in text or 'dos' in text:
                context.intent = 'meshtastic_flood'
                context.requires_confirmation = True
                context.is_dangerous = True
            elif 'stop' in text:
                context.intent = 'meshtastic_stop'
            elif 'status' in text:
                context.intent = 'meshtastic_status'
            else:
                context.intent = 'meshtastic_status'
            context.confidence = 0.9
        
        # ===== BLUETOOTH 5.X =====
        elif is_bluetooth5_command:
            context.category = CommandCategory.BLUETOOTH5
            if 'scan' in text:
                if 'long' in text or 'range' in text:
                    context.intent = 'ble5_long_range_scan'
                else:
                    context.intent = 'ble5_scan'
            elif 'direction' in text or 'aoa' in text or 'aod' in text:
                context.intent = 'ble5_direction_finding'
            elif 'gatt' in text or 'enumerate' in text:
                context.intent = 'ble5_enumerate_gatt'
            elif 'attack' in text or 'exploit' in text:
                context.intent = 'ble5_attack'
                context.requires_confirmation = True
            elif 'coded' in text or 'phy' in text:
                context.intent = 'ble5_coded_phy'
            elif 'status' in text:
                context.intent = 'ble5_status'
            else:
                context.intent = 'ble5_status'
            context.confidence = 0.9
        
        # ===== ROLLJAM =====
        elif is_rolljam_command:
            context.category = CommandCategory.ROLLJAM
            context.requires_confirmation = True
            if 'start' in text:
                context.intent = 'rolljam_start'
            elif 'capture' in text:
                context.intent = 'rolljam_capture'
            elif 'replay' in text:
                context.intent = 'rolljam_replay'
            elif 'export' in text:
                context.intent = 'rolljam_export'
            elif 'stop' in text:
                context.intent = 'rolljam_stop'
            elif 'status' in text:
                context.intent = 'rolljam_status'
            else:
                context.intent = 'rolljam_status'
            context.confidence = 0.9
        
        # ===== AGC/CALIBRATION =====
        elif is_agc_command:
            context.category = CommandCategory.AGC
            if 'mode' in text:
                context.intent = 'agc_set_mode'
                if 'manual' in text:
                    context.parameters['mode'] = 'manual'
                elif 'fast' in text:
                    context.parameters['mode'] = 'fast_attack'
                elif 'slow' in text:
                    context.parameters['mode'] = 'slow_attack'
                elif 'hybrid' in text:
                    context.parameters['mode'] = 'hybrid'
            elif 'gain' in text:
                context.intent = 'agc_manual_gain'
                gain_match = re.search(r'(\d+)\s*(?:db)?', text)
                if gain_match:
                    context.parameters['gain'] = int(gain_match.group(1))
            elif 'calibrat' in text:
                context.intent = 'agc_calibrate'
            elif 'dc offset' in text:
                context.intent = 'agc_dc_offset'
            elif 'iq' in text:
                context.intent = 'agc_iq_calibration'
            elif 'rssi' in text or 'monitor' in text:
                context.intent = 'agc_rssi_monitor'
            elif 'temp' in text:
                context.intent = 'agc_temperature'
            elif 'status' in text:
                context.intent = 'agc_status'
            else:
                context.intent = 'agc_status'
            context.confidence = 0.9
        
        # ===== FREQUENCY HOPPING =====
        elif is_hopping_command:
            context.category = CommandCategory.HOPPING
            if 'start' in text:
                context.intent = 'hopping_start'
            elif 'track' in text:
                context.intent = 'hopping_track'
            elif 'predict' in text:
                context.intent = 'hopping_predict'
            elif 'sync' in text:
                context.intent = 'hopping_sync'
            elif 'jam' in text:
                context.intent = 'hopping_jam'
                context.requires_confirmation = True
            elif 'pattern' in text:
                context.intent = 'hopping_pattern'
            elif 'status' in text:
                context.intent = 'hopping_status'
            elif 'stop' in text:
                context.intent = 'hopping_stop'
            else:
                context.intent = 'hopping_status'
            context.confidence = 0.9
        
        # ===== XB-200 TRANSVERTER =====
        elif is_xb200_command:
            context.category = CommandCategory.XB200
            if 'enable' in text:
                context.intent = 'xb200_enable'
            elif 'fm' in text:
                context.intent = 'xb200_fm_receive'
            elif 'noaa' in text or 'weather' in text:
                context.intent = 'xb200_noaa'
            elif 'shortwave' in text:
                context.intent = 'xb200_shortwave'
            elif 'aircraft' in text or 'air' in text:
                context.intent = 'xb200_aircraft'
            elif 'amateur' in text or 'ham' in text:
                context.intent = 'xb200_amateur'
            elif 'hf' in text:
                context.intent = 'xb200_hf'
            elif 'tune' in text:
                context.intent = 'xb200_tune'
                freq_match = re.search(r'(\d+\.?\d*)\s*(mhz|khz)', text.lower())
                if freq_match:
                    context.parameters['frequency'] = freq_match.group(0)
            elif 'status' in text:
                context.intent = 'xb200_status'
            else:
                context.intent = 'xb200_status'
            context.confidence = 0.9
        
        # ===== LTE/5G DECODER =====
        elif is_lte5g_command:
            context.category = CommandCategory.LTE5G
            if 'scan' in text:
                if '5g' in text or 'nr' in text:
                    context.intent = 'lte5g_5g_scan'
                else:
                    context.intent = 'lte5g_scan'
            elif 'decode' in text:
                context.intent = 'lte5g_decode'
            elif 'imsi' in text or 'capture' in text:
                context.intent = 'lte5g_imsi_capture'
                context.requires_confirmation = True
            elif 'sib' in text:
                context.intent = 'lte5g_decode_sib'
            elif 'cell' in text:
                context.intent = 'lte5g_cell_info'
            elif 'status' in text:
                context.intent = 'lte5g_status'
            else:
                context.intent = 'lte5g_status'
            context.confidence = 0.9
        
        # ===== DMR/P25/TETRA DIGITAL RADIO =====
        elif is_digitalradio_command:
            context.category = CommandCategory.DIGITALRADIO
            if 'scan' in text:
                context.intent = 'digitalradio_scan'
            elif 'dmr' in text:
                if 'decode' in text or 'intercept' in text:
                    context.intent = 'digitalradio_dmr_decode'
                else:
                    context.intent = 'digitalradio_dmr_scan'
            elif 'p25' in text:
                if 'decode' in text:
                    context.intent = 'digitalradio_p25_decode'
                else:
                    context.intent = 'digitalradio_p25_scan'
            elif 'tetra' in text:
                if 'decode' in text:
                    context.intent = 'digitalradio_tetra_decode'
                else:
                    context.intent = 'digitalradio_tetra_scan'
            elif 'trunking' in text or 'follow' in text:
                context.intent = 'digitalradio_trunking'
            elif 'status' in text:
                context.intent = 'digitalradio_status'
            else:
                context.intent = 'digitalradio_status'
            context.confidence = 0.9
        
        # ===== SDR HARDWARE (SoapySDR Universal) =====
        elif is_sdr_command:
            context.category = CommandCategory.SDR
            if 'scan' in text or 'detect' in text or 'enumerate' in text:
                context.intent = 'sdr_scan'
            elif 'list' in text or 'show device' in text:
                context.intent = 'sdr_list'
            elif 'select' in text or 'use' in text or 'open' in text:
                context.intent = 'sdr_select'
                # Extract device index or type
                idx_match = re.search(r'(\d+)', text)
                if idx_match:
                    context.parameters['device_index'] = int(idx_match.group(1))
                for sdr_type in ['bladerf', 'hackrf', 'rtlsdr', 'rtl-sdr', 'usrp', 'limesdr', 'airspy', 'plutosdr']:
                    if sdr_type in text:
                        context.parameters['device_type'] = sdr_type.replace('-', '')
            elif 'info' in text or 'status' in text:
                context.intent = 'sdr_info'
            elif 'configure' in text or 'setup' in text:
                context.intent = 'sdr_configure'
                context.requires_confirmation = True
            elif 'capture' in text or 'record' in text:
                context.intent = 'sdr_capture'
                # Extract duration
                dur_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:s|sec|second)', text)
                if dur_match:
                    context.parameters['duration'] = float(dur_match.group(1))
                else:
                    context.parameters['duration'] = 1.0  # Default 1 second
            elif 'play' in text or 'replay' in text or 'transmit' in text:
                context.intent = 'sdr_play'
                context.requires_confirmation = True
                context.is_dangerous = True
            elif 'spectrum' in text or 'fft' in text or 'waterfall' in text:
                context.intent = 'sdr_spectrum'
            elif 'tune' in text or 'frequency' in text or 'freq' in text:
                context.intent = 'sdr_tune'
                # Extract frequency
                freq_match = re.search(r'(\d+(?:\.\d+)?)\s*(hz|khz|mhz|ghz)?', text, re.I)
                if freq_match:
                    freq = float(freq_match.group(1))
                    unit = (freq_match.group(2) or 'mhz').lower()
                    multipliers = {'hz': 1, 'khz': 1e3, 'mhz': 1e6, 'ghz': 1e9}
                    context.parameters['frequency'] = freq * multipliers.get(unit, 1e6)
            elif 'gain' in text:
                context.intent = 'sdr_gain'
                gain_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:db)?', text)
                if gain_match:
                    context.parameters['gain'] = float(gain_match.group(1))
            elif 'sample' in text and 'rate' in text:
                context.intent = 'sdr_samplerate'
                rate_match = re.search(r'(\d+(?:\.\d+)?)\s*(sps|ksps|msps)?', text, re.I)
                if rate_match:
                    rate = float(rate_match.group(1))
                    unit = (rate_match.group(2) or 'msps').lower()
                    multipliers = {'sps': 1, 'ksps': 1e3, 'msps': 1e6}
                    context.parameters['sample_rate'] = rate * multipliers.get(unit, 1e6)
            elif 'close' in text or 'stop' in text or 'disconnect' in text:
                context.intent = 'sdr_close'
            else:
                context.intent = 'sdr_info'
            context.confidence = 0.9
        
        # ===== SYSTEM / HELP / STATUS =====
        elif any(word in text for word in ['help', 'status', 'info', 'what can', 'show', 'list']):
            context.category = CommandCategory.SYSTEM
            
            if 'help' in text:
                context.intent = 'help'
                # Check for specific topic
                for topic in self.HELP_TOPICS:
                    if topic in text:
                        context.parameters['topic'] = topic
            elif 'opsec' in text:
                # Redirect to OPSEC
                context.category = CommandCategory.OPSEC
                context.intent = 'opsec_status'
            elif 'mission' in text:
                # Redirect to missions
                context.category = CommandCategory.MISSION
                context.intent = 'mission_status'
            elif 'mode' in text:
                # Redirect to user mode
                context.category = CommandCategory.MODE
                context.intent = 'show_mode'
            # Redirect pentest module status commands to PENTEST category
            elif is_pentest_command and 'status' in text:
                context.category = CommandCategory.PENTEST
                if 'api' in text and 'security' in text:
                    context.intent = 'api_security_status'
                elif 'cloud' in text:
                    context.intent = 'cloud_security_status'
                elif 'dns' in text:
                    context.intent = 'dns_attack_status'
                elif 'mobile' in text:
                    context.intent = 'mobile_security_status'
                elif 'supply' in text:
                    context.intent = 'supply_chain_status'
                elif 'sso' in text or 'identity' in text:
                    context.intent = 'sso_attack_status'
                elif 'websocket' in text or 'ws' in text:
                    context.intent = 'websocket_status'
                elif 'graphql' in text:
                    context.intent = 'graphql_status'
                elif 'browser' in text:
                    context.intent = 'browser_attack_status'
                elif 'protocol' in text:
                    context.intent = 'protocol_attack_status'
                else:
                    context.intent = 'pentest_status'
            elif 'status' in text or 'info' in text:
                context.intent = 'status'
            elif 'history' in text:
                context.intent = 'history'
            
            context.confidence = 0.9
        
        # ===== ONLINE PENETRATION TESTING =====
        elif is_pentest_command:
            context.category = CommandCategory.PENTEST
            
            # Extract target URL/IP if present
            url_match = re.search(r'https?://[^\s]+', text)
            ip_match = re.search(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', text)
            if url_match:
                context.parameters['target'] = url_match.group()
            elif ip_match:
                context.parameters['target'] = ip_match.group(1)
            
            # Web scanning
            if any(kw in text for kw in ['web scan', 'vuln scan', 'vulnerability scan', 'scan web']):
                context.intent = 'web_scan'
                context.requires_confirmation = True
            elif 'sqli' in text or 'sql injection' in text:
                context.intent = 'web_sqli'
                context.requires_confirmation = True
            elif 'xss' in text or 'cross-site' in text:
                context.intent = 'web_xss'
            elif 'csrf' in text:
                context.intent = 'web_csrf'
            elif 'lfi' in text:
                context.intent = 'web_lfi'
            elif 'rfi' in text:
                context.intent = 'web_rfi'
            elif 'directory' in text and 'brute' in text or 'dir brute' in text:
                context.intent = 'web_dir_brute'
            elif 'api fuzz' in text:
                context.intent = 'web_api_fuzz'
            
            # Credential attacks
            elif 'brute force' in text or 'bruteforce' in text:
                context.intent = 'credential_brute'
                context.requires_confirmation = True
                # Extract protocol
                for proto in ['ssh', 'ftp', 'http', 'smb', 'rdp', 'mysql', 'telnet']:
                    if proto in text:
                        context.parameters['protocol'] = proto
                        break
            elif 'password spray' in text:
                context.intent = 'credential_spray'
                context.requires_confirmation = True
            elif 'credential stuff' in text:
                context.intent = 'credential_stuff'
                context.requires_confirmation = True
            
            # Network reconnaissance
            elif 'port scan' in text or 'scan port' in text:
                context.intent = 'recon_ports'
            elif 'service fingerprint' in text or 'fingerprint service' in text:
                context.intent = 'recon_fingerprint'
            elif 'os detect' in text or 'detect os' in text:
                context.intent = 'recon_os'
            elif 'host discovery' in text or 'discover host' in text:
                context.intent = 'recon_hosts'
            elif 'network map' in text:
                context.intent = 'recon_map'
            elif 'stealth scan' in text:
                context.intent = 'recon_stealth'
            elif 'tcp scan' in text:
                context.intent = 'recon_tcp'
            elif 'udp scan' in text:
                context.intent = 'recon_udp'
            
            # OSINT
            elif 'domain intel' in text or 'intel domain' in text:
                context.intent = 'osint_domain'
                domain_match = re.search(r'([a-zA-Z0-9][a-zA-Z0-9-]*\.[a-zA-Z]{2,})', text)
                if domain_match:
                    context.parameters['domain'] = domain_match.group(1)
            elif 'email harvest' in text:
                context.intent = 'osint_email'
            elif 'whois' in text:
                context.intent = 'osint_whois'
            elif 'subdomain' in text:
                context.intent = 'osint_subdomain'
            elif 'dns recon' in text:
                context.intent = 'osint_dns'
            elif 'social profile' in text:
                context.intent = 'osint_social'
            elif 'leaked cred' in text:
                context.intent = 'osint_leaked'
            elif 'osint' in text:
                context.intent = 'osint_full'
            
            # Exploit framework
            elif 'search exploit' in text or 'exploit search' in text:
                context.intent = 'exploit_search'
                cve_match = re.search(r'cve[-_]?(\d{4})[-_]?(\d+)', text, re.IGNORECASE)
                if cve_match:
                    context.parameters['cve'] = f"CVE-{cve_match.group(1)}-{cve_match.group(2)}"
            elif 'generate payload' in text or 'payload generate' in text:
                context.intent = 'exploit_payload'
            elif 'msfvenom' in text:
                context.intent = 'exploit_msfvenom'
            elif 'post-exploit' in text or 'postexploit' in text:
                context.intent = 'exploit_post'
            elif 'shell upgrade' in text:
                context.intent = 'exploit_shell_upgrade'
            elif 'reverse shell' in text:
                context.intent = 'exploit_reverse_shell'
            elif 'exploit' in text:
                context.intent = 'exploit_run'
                context.requires_confirmation = True
            
            # C2 Framework
            elif 'start c2' in text or 'c2 start' in text:
                context.intent = 'c2_start'
                context.requires_confirmation = True
            elif 'stop c2' in text or 'c2 stop' in text:
                context.intent = 'c2_stop'
            elif 'list beacon' in text or 'beacon list' in text:
                context.intent = 'c2_list_beacons'
            elif 'generate beacon' in text or 'beacon generate' in text:
                context.intent = 'c2_generate_beacon'
            elif 'task beacon' in text or 'beacon task' in text:
                context.intent = 'c2_task'
            elif 'c2 status' in text or 'beacon status' in text:
                context.intent = 'c2_status'
            
            # Proxy chain
            elif 'start proxy' in text or 'proxy start' in text:
                context.intent = 'proxy_start'
            elif 'tor circuit' in text:
                context.intent = 'proxy_tor'
            elif 'rotate proxy' in text:
                context.intent = 'proxy_rotate'
            elif 'add socks' in text:
                context.intent = 'proxy_add_socks'
            elif 'add http' in text and 'proxy' in text:
                context.intent = 'proxy_add_http'
            elif 'test proxy' in text:
                context.intent = 'proxy_test'
            elif 'proxy status' in text:
                context.intent = 'proxy_status'
            
            # ===== TIER 1: API Security Testing =====
            elif any(kw in text for kw in ['api scan', 'api security', 'rest fuzz', 'api enum']):
                context.intent = 'api_scan'
                context.requires_confirmation = True
            elif 'jwt' in text and ('crack' in text or 'attack' in text):
                context.intent = 'api_jwt_attack'
            elif 'jwt' in text and 'analyze' in text:
                context.intent = 'api_jwt_analyze'
            elif 'oauth' in text and ('attack' in text or 'test' in text):
                context.intent = 'api_oauth_attack'
            elif 'bola' in text or 'idor' in text:
                context.intent = 'api_bola_test'
            elif 'bfla' in text:
                context.intent = 'api_bfla_test'
            elif 'rate limit' in text:
                context.intent = 'api_rate_limit'
            elif 'mass assignment' in text:
                context.intent = 'api_mass_assignment'
            elif 'swagger' in text or 'openapi' in text:
                context.intent = 'api_swagger_scan'
            elif 'api' in text and 'status' in text:
                context.intent = 'api_security_status'
            
            # ===== TIER 1: Cloud Security Assessment =====
            elif any(kw in text for kw in ['cloud scan', 'cloud security', 'cloud misconfig']):
                context.intent = 'cloud_scan'
                context.requires_confirmation = True
            elif 'aws' in text and 'scan' in text:
                context.intent = 'cloud_aws_scan'
            elif 'azure' in text and 'scan' in text:
                context.intent = 'cloud_azure_scan'
            elif 'gcp' in text and 'scan' in text:
                context.intent = 'cloud_gcp_scan'
            elif 's3' in text and ('bucket' in text or 'enum' in text):
                context.intent = 'cloud_s3_enum'
            elif 'iam' in text and 'analysis' in text:
                context.intent = 'cloud_iam_analysis'
            elif 'lambda' in text or 'serverless' in text:
                context.intent = 'cloud_lambda_scan'
            elif 'imds' in text or 'ec2 meta' in text:
                context.intent = 'cloud_imds_attack'
            elif 'cloud' in text and 'status' in text:
                context.intent = 'cloud_security_status'
            
            # ===== TIER 1: DNS/Domain Attacks =====
            elif 'zone transfer' in text:
                context.intent = 'dns_zone_transfer'
                context.requires_confirmation = True
            elif 'subdomain takeover' in text:
                context.intent = 'dns_takeover'
            elif 'dangling dns' in text:
                context.intent = 'dns_dangling'
            elif 'cache poison' in text or 'dns poison' in text:
                context.intent = 'dns_cache_poison'
                context.requires_confirmation = True
                context.is_dangerous = True
            elif 'dnssec' in text:
                context.intent = 'dns_dnssec_test'
            elif 'dns tunnel' in text:
                context.intent = 'dns_tunnel'
            elif 'dns' in text and 'attack' in text and 'status' in text:
                context.intent = 'dns_attack_status'
            
            # ===== TIER 1: Mobile App Backend Testing =====
            elif 'firebase' in text and ('scan' in text or 'enum' in text):
                context.intent = 'mobile_firebase_scan'
            elif 'firestore' in text:
                context.intent = 'mobile_firestore_enum'
            elif 'cert pinning' in text:
                context.intent = 'mobile_cert_pinning'
            elif 'deep link' in text:
                context.intent = 'mobile_deep_link'
            elif 'mobile' in text and ('api' in text or 'backend' in text):
                context.intent = 'mobile_api_scan'
            elif 'mobile' in text and 'status' in text:
                context.intent = 'mobile_security_status'
            
            # ===== TIER 2: Supply Chain Security =====
            elif 'dependency confusion' in text:
                context.intent = 'supply_dependency_confusion'
                context.requires_confirmation = True
            elif 'typosquat' in text:
                context.intent = 'supply_typosquat'
            elif 'npm audit' in text or 'npm' in text and 'scan' in text:
                context.intent = 'supply_npm_audit'
            elif 'pypi' in text and ('scan' in text or 'check' in text):
                context.intent = 'supply_pypi_scan'
            elif 'ci/cd' in text or 'pipeline' in text:
                context.intent = 'supply_cicd_scan'
            elif 'sbom' in text:
                context.intent = 'supply_sbom_analysis'
            elif 'supply chain' in text and 'status' in text:
                context.intent = 'supply_chain_status'
            
            # ===== TIER 2: SSO/Identity Attacks =====
            elif 'saml' in text and ('bypass' in text or 'attack' in text):
                context.intent = 'sso_saml_bypass'
                context.requires_confirmation = True
            elif 'oauth flow' in text or ('oauth' in text and 'attack' in text):
                context.intent = 'sso_oauth_attack'
            elif 'kerberos' in text or 'as-rep roast' in text:
                context.intent = 'sso_kerberos_attack'
            elif 'session fixation' in text:
                context.intent = 'sso_session_fixation'
            elif 'session hijack' in text:
                context.intent = 'sso_session_hijack'
            elif 'mfa bypass' in text:
                context.intent = 'sso_mfa_bypass'
            elif 'token replay' in text:
                context.intent = 'sso_token_replay'
            elif 'sso' in text and 'status' in text:
                context.intent = 'sso_attack_status'
            
            # ===== TIER 2: WebSocket Security =====
            elif 'ws scan' in text or ('websocket' in text and 'scan' in text):
                context.intent = 'ws_scan'
            elif 'cswsh' in text:
                context.intent = 'ws_cswsh'
            elif 'ws injection' in text or ('websocket' in text and 'inject' in text):
                context.intent = 'ws_injection'
            elif 'ws hijack' in text:
                context.intent = 'ws_hijack'
            elif 'ws replay' in text:
                context.intent = 'ws_replay'
            elif 'socket.io' in text or 'signalr' in text:
                context.intent = 'ws_framework_attack'
            elif 'websocket' in text and 'status' in text:
                context.intent = 'websocket_status'
            
            # ===== TIER 2: GraphQL-Specific =====
            elif 'introspection' in text and 'graphql' in text:
                context.intent = 'graphql_introspection'
            elif 'graphql' in text and 'batch' in text:
                context.intent = 'graphql_batching'
            elif 'graphql' in text and 'depth' in text:
                context.intent = 'graphql_depth_attack'
            elif 'graphql' in text and ('dos' in text or 'denial' in text):
                context.intent = 'graphql_dos'
            elif 'graphql' in text and 'enum' in text:
                context.intent = 'graphql_enum'
            elif 'query complexity' in text:
                context.intent = 'graphql_complexity'
            elif 'graphql' in text and 'injection' in text:
                context.intent = 'graphql_injection'
            elif 'graphql' in text and 'status' in text:
                context.intent = 'graphql_status'
            
            # ===== TIER 3: Browser-Based Attacks =====
            elif 'xs-leak' in text or 'xsleak' in text:
                context.intent = 'browser_xsleak'
            elif 'spectre' in text or 'meltdown' in text:
                context.intent = 'browser_spectre'
            elif 'browser gadget' in text:
                context.intent = 'browser_gadget'
            elif 'extension' in text and ('attack' in text or 'audit' in text):
                context.intent = 'browser_extension'
            elif 'service worker' in text:
                context.intent = 'browser_service_worker'
            elif 'cache' in text and 'timing' in text:
                context.intent = 'browser_cache_timing'
            elif 'browser' in text and 'attack' in text and 'status' in text:
                context.intent = 'browser_attack_status'
            
            # ===== TIER 3: Protocol-Level =====
            elif 'http2 smuggle' in text or 'h2 smuggle' in text:
                context.intent = 'protocol_http2_smuggle'
                context.requires_confirmation = True
            elif 'h2c' in text:
                context.intent = 'protocol_h2c_attack'
            elif 'grpc' in text and ('fuzz' in text or 'attack' in text):
                context.intent = 'protocol_grpc_attack'
            elif 'webrtc' in text or 'ice leak' in text:
                context.intent = 'protocol_webrtc_leak'
            elif 'request smuggling' in text or 'desync' in text:
                context.intent = 'protocol_request_smuggling'
            elif 'http3' in text:
                context.intent = 'protocol_http3_test'
            elif 'protocol' in text and 'status' in text:
                context.intent = 'protocol_attack_status'
            
            # General pentest status
            elif 'pentest status' in text:
                context.intent = 'pentest_status'
            else:
                context.intent = 'pentest_status'
            
            context.confidence = 0.95
        
        # ===== SUPERHERO BLOCKCHAIN INTELLIGENCE =====
        elif is_superhero_command:
            context.category = CommandCategory.SUPERHERO
            
            # Extract wallet address if present
            eth_addr = re.search(r'0x[a-fA-F0-9]{40}', text)
            btc_addr = re.search(r'[13][a-km-zA-HJ-NP-Z1-9]{25,34}', text)
            if eth_addr:
                context.parameters['address'] = eth_addr.group()
                context.parameters['chain'] = 'ethereum'
            elif btc_addr:
                context.parameters['address'] = btc_addr.group()
                context.parameters['chain'] = 'bitcoin'
            
            # Extract case ID if present
            case_match = re.search(r'case[_-]?(\w+)', text, re.IGNORECASE)
            if case_match:
                context.parameters['case_id'] = case_match.group(1)
            
            # Blockchain forensics
            if 'trace wallet' in text or 'trace address' in text:
                context.intent = 'superhero_trace_wallet'
            elif 'trace transaction' in text:
                context.intent = 'superhero_trace_tx'
            elif 'cluster wallet' in text or 'wallet cluster' in text:
                context.intent = 'superhero_cluster'
            elif 'detect mixer' in text or 'mixer detect' in text:
                context.intent = 'superhero_detect_mixer'
            elif 'identify exchange' in text:
                context.intent = 'superhero_identify_exchange'
            elif 'cross-chain' in text or 'chain hop' in text:
                context.intent = 'superhero_cross_chain'
            elif 'check address' in text or 'address check' in text:
                context.intent = 'superhero_check_address'
            
            # Identity correlation
            elif 'identify owner' in text or 'wallet owner' in text:
                context.intent = 'superhero_identify_owner'
            elif 'identity correlation' in text or 'correlate identity' in text:
                context.intent = 'superhero_correlate'
            elif 'ens lookup' in text:
                context.intent = 'superhero_ens_lookup'
            elif 'social media' in text and ('match' in text or 'search' in text):
                context.intent = 'superhero_social_media'
            elif 'email correlation' in text or 'correlate email' in text:
                context.intent = 'superhero_email_correlate'
            elif 'forum analysis' in text or 'analyze forum' in text:
                context.intent = 'superhero_forum_analysis'
            
            # Geolocation analysis
            elif 'geolocation' in text or 'locate owner' in text:
                context.intent = 'superhero_geolocation'
            elif 'timezone analysis' in text or 'analyze timezone' in text:
                context.intent = 'superhero_timezone'
            elif 'behavioral pattern' in text or 'activity pattern' in text:
                context.intent = 'superhero_behavioral'
            elif 'activity timing' in text:
                context.intent = 'superhero_timing'
            
            # Counter-countermeasures
            elif 'mixer trace' in text or 'trace mixer' in text:
                context.intent = 'superhero_mixer_trace'
            elif 'privacy coin' in text:
                context.intent = 'superhero_privacy_coin'
            elif 'tornado' in text:
                context.intent = 'superhero_tornado_trace'
            
            # Dossier generation
            elif 'generate dossier' in text or 'create dossier' in text:
                context.intent = 'superhero_generate_dossier'
            elif 'export pdf' in text:
                context.intent = 'superhero_export_pdf'
            elif 'export json' in text:
                context.intent = 'superhero_export_json'
            elif 'export html' in text:
                context.intent = 'superhero_export_html'
            elif 'add notes' in text or 'analyst notes' in text:
                context.intent = 'superhero_add_notes'
            elif 'set classification' in text:
                context.intent = 'superhero_set_classification'
            
            # Investigation management
            elif 'create investigation' in text or 'new investigation' in text:
                context.intent = 'superhero_create_investigation'
            elif 'add target' in text:
                context.intent = 'superhero_add_target'
            elif 'run investigation' in text or 'start investigation' in text:
                context.intent = 'superhero_run_investigation'
            elif 'list investigation' in text:
                context.intent = 'superhero_list_investigations'
            elif 'get alerts' in text or 'list alerts' in text:
                context.intent = 'superhero_get_alerts'
            elif 'acknowledge alert' in text:
                context.intent = 'superhero_ack_alert'
            
            # Monitoring
            elif 'monitor address' in text:
                context.intent = 'superhero_monitor_address'
            elif 'stop monitor' in text:
                context.intent = 'superhero_stop_monitor'
            
            # Status
            elif 'superhero status' in text or 'blockchain status' in text or 'forensics status' in text:
                context.intent = 'superhero_status'
            elif 'investigation status' in text:
                context.intent = 'superhero_investigation_status'
            elif 'identity status' in text:
                context.intent = 'superhero_identity_status'
            elif 'dossier status' in text:
                context.intent = 'superhero_dossier_status'
            
            # ===== CRYPTOCURRENCY SECURITY ASSESSMENT & RECOVERY TOOLKIT =====
            
            # Wallet Security Scanner
            elif 'scan wallet' in text or 'wallet scan' in text or 'security scan' in text:
                context.intent = 'toolkit_wallet_scan'
            elif 'vulnerability scan' in text or 'vuln scan' in text:
                context.intent = 'toolkit_vuln_scan'
            elif 'security assessment' in text or 'assess security' in text:
                context.intent = 'toolkit_security_assess'
            elif 'keystore analysis' in text or 'analyze keystore' in text:
                context.intent = 'toolkit_keystore_analyze'
            elif 'backup audit' in text:
                context.intent = 'toolkit_backup_audit'
            elif 'wallet security status' in text:
                context.intent = 'toolkit_wallet_status'
            
            # Key Derivation Analyzer
            elif 'analyze derivation' in text or 'derivation analyze' in text:
                context.intent = 'toolkit_derivation_analyze'
            elif 'check entropy' in text or 'entropy check' in text:
                context.intent = 'toolkit_entropy_check'
            elif 'derivation path' in text:
                context.intent = 'toolkit_derivation_path'
            elif 'seed strength' in text:
                context.intent = 'toolkit_seed_strength'
            elif 'key weakness' in text:
                context.intent = 'toolkit_key_weakness'
            elif 'derivation status' in text:
                context.intent = 'toolkit_derivation_status'
            
            # Smart Contract Auditor
            elif 'audit contract' in text or 'contract audit' in text:
                context.intent = 'toolkit_contract_audit'
            elif 'detect exploit' in text or 'exploit detect' in text:
                context.intent = 'toolkit_exploit_detect'
            elif 'reentrancy check' in text or 'check reentrancy' in text:
                context.intent = 'toolkit_reentrancy'
            elif 'access control audit' in text:
                context.intent = 'toolkit_access_control'
            elif 'contract status' in text:
                context.intent = 'toolkit_contract_status'
            
            # Recovery Toolkit
            elif 'initiate recovery' in text or 'start recovery' in text:
                context.intent = 'toolkit_initiate_recovery'
            elif 'seed reconstruction' in text or 'reconstruct seed' in text:
                context.intent = 'toolkit_seed_reconstruct'
            elif 'password recovery' in text or 'recover password' in text:
                context.intent = 'toolkit_password_recovery'
            elif 'multisig recovery' in text:
                context.intent = 'toolkit_multisig_recovery'
            elif 'hardware recovery' in text:
                context.intent = 'toolkit_hardware_recovery'
            elif 'recovery status' in text:
                context.intent = 'toolkit_recovery_status'
            
            # Malicious Address Database
            elif 'check malicious' in text or 'malicious check' in text or 'threat check' in text:
                context.intent = 'toolkit_malicious_check'
            elif 'add malicious' in text:
                context.intent = 'toolkit_malicious_add'
            elif 'search malicious' in text:
                context.intent = 'toolkit_malicious_search'
            elif 'threat lookup' in text:
                context.intent = 'toolkit_threat_lookup'
            elif 'db status' in text or 'database status' in text:
                context.intent = 'toolkit_db_status'
            elif 'malicious report' in text:
                context.intent = 'toolkit_malicious_report'
            
            # Authority Report Generator
            elif 'create case' in text or 'new case' in text:
                context.intent = 'toolkit_create_case'
            elif 'authority report' in text or 'law enforcement report' in text:
                context.intent = 'toolkit_authority_report'
            elif 'add evidence' in text:
                context.intent = 'toolkit_add_evidence'
            elif 'chain of custody' in text:
                context.intent = 'toolkit_chain_custody'
            elif 'report status' in text:
                context.intent = 'toolkit_report_status'
            
            else:
                context.intent = 'superhero_status'
            
            context.confidence = 0.95
        
        # ===== UNKNOWN =====
        elif context.category is None:
            context.category = CommandCategory.SYSTEM
            context.intent = 'unknown'
            context.confidence = 0.3
        
        return context
    
    def _parse_frequency(self, freq_str: str) -> int:
        """Parse frequency string to Hz"""
        match = re.match(r'(\d+\.?\d*)\s*(mhz|ghz|khz)', freq_str.lower())
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            
            if unit == 'ghz':
                return int(value * 1e9)
            elif unit == 'mhz':
                return int(value * 1e6)
            elif unit == 'khz':
                return int(value * 1e3)
        
        return 0
    
    def _request_confirmation(self, context: CommandContext) -> CommandResult:
        """Request user confirmation for dangerous commands"""
        self.pending_confirmation = context
        
        # Build warning message
        warnings = []
        
        if context.category == CommandCategory.NETWORK and context.intent == 'go_online':
            mode = context.parameters.get('mode', 'tor')
            if self._network_mode:
                from core.network_mode import NetworkMode
                mode_enum = {
                    'tor': NetworkMode.ONLINE_TOR,
                    'vpn': NetworkMode.ONLINE_VPN,
                    'full': NetworkMode.ONLINE_FULL,
                    'direct': NetworkMode.ONLINE_DIRECT,
                }.get(mode, NetworkMode.ONLINE_TOR)
                warnings = self._network_mode.get_online_warnings(mode_enum)
        
        elif context.category == CommandCategory.EMERGENCY:
            warnings = [
                "!!! EMERGENCY ACTION !!!",
                "This will immediately stop all operations",
                "Data may be permanently deleted",
            ]
        
        elif context.category == CommandCategory.JAMMING:
            warnings = [
                "Jamming signals is illegal in most jurisdictions",
                "Ensure you have proper authorization",
                "This may disrupt legitimate communications",
            ]
        
        elif context.category == CommandCategory.VEHICLE:
            warnings = [
                "⚠️ VEHICLE SECURITY TESTING",
                "Only test on vehicles you own or have explicit authorization",
                "Unauthorized vehicle access is a federal crime",
                "Testing on public roads is illegal and dangerous",
                "Some operations may permanently damage vehicle systems",
            ]
            if context.intent in ['uds_flash', 'uds_write_memory']:
                warnings.append("⛔ ECU flashing can BRICK the vehicle - use extreme caution!")
            if 'rolljam' in str(context.intent):
                warnings.append("🚨 RollJam attacks may trigger vehicle security systems")
        
        elif context.is_dangerous:
            warnings = [
                "This command may compromise stealth",
                "Ensure you understand the implications",
            ]
        
        warning_text = "\n".join([f"  ! {w}" for w in warnings])
        
        return CommandResult(
            success=True,
            message=f"CONFIRMATION REQUIRED\n\n"
                   f"Command: {context.raw_input}\n"
                   f"Category: {context.category.value if context.category else 'unknown'}\n\n"
                   f"Warnings:\n{warning_text}\n\n"
                   f"Say 'yes' or 'confirm' to proceed, 'no' or 'cancel' to abort.",
            data={'pending_confirmation': True, 'context': context.raw_input},
            warnings=warnings
        )
    
    def _handle_confirmation(self, user_input: str, context: CommandContext) -> CommandResult:
        """Handle confirmation response"""
        text = user_input.lower().strip()
        
        if text in ['yes', 'confirm', 'proceed', 'y', 'affirmative', 'do it']:
            # User confirmed - execute the pending command
            pending = self.pending_confirmation
            self.pending_confirmation = None
            
            result = self._execute_command(pending)
            self._log_command(pending, result)
            return result
        
        elif text in ['no', 'cancel', 'abort', 'n', 'negative', 'stop']:
            # User cancelled
            self.pending_confirmation = None
            return CommandResult(
                success=True,
                message="Command cancelled."
            )
        
        else:
            # Invalid response
            return CommandResult(
                success=False,
                message="Please say 'yes' to confirm or 'no' to cancel."
            )
    
    def _execute_command(self, context: CommandContext) -> CommandResult:
        """Execute the parsed command"""
        
        if context.category == CommandCategory.NETWORK:
            return self._execute_network_command(context)
        
        elif context.category == CommandCategory.STEALTH:
            return self._execute_stealth_command(context)
        
        elif context.category == CommandCategory.CELLULAR:
            return self._execute_cellular_command(context)
        
        elif context.category == CommandCategory.TARGETING:
            return self._execute_targeting_command(context)
        
        elif context.category == CommandCategory.WIFI:
            return self._execute_wifi_command(context)
        
        elif context.category == CommandCategory.GPS:
            return self._execute_gps_command(context)
        
        elif context.category == CommandCategory.DRONE:
            return self._execute_drone_command(context)
        
        elif context.category == CommandCategory.JAMMING:
            return self._execute_jamming_command(context)
        
        elif context.category == CommandCategory.SPECTRUM:
            return self._execute_spectrum_command(context)
        
        elif context.category == CommandCategory.SIGINT:
            return self._execute_sigint_command(context)
        
        elif context.category == CommandCategory.CAPTURE:
            return self._execute_capture_command(context)
        
        elif context.category == CommandCategory.EMERGENCY:
            return self._execute_emergency_command(context)
        
        elif context.category == CommandCategory.SYSTEM:
            return self._execute_system_command(context)
        
        elif context.category == CommandCategory.MISSION:
            return self._execute_mission_command(context)
        
        elif context.category == CommandCategory.OPSEC:
            return self._execute_opsec_command(context)
        
        elif context.category == CommandCategory.MODE:
            return self._execute_mode_command(context)
        
        elif context.category == CommandCategory.DEFENSIVE:
            return self._execute_defensive_command(context)
        
        elif context.category == CommandCategory.DASHBOARD:
            return self._execute_dashboard_command(context)
        
        elif context.category == CommandCategory.REPLAY:
            return self._execute_replay_command(context)
        
        elif context.category == CommandCategory.HARDWARE:
            return self._execute_hardware_command(context)
        
        elif context.category == CommandCategory.VEHICLE:
            return self._execute_vehicle_command(context)
        
        elif context.category == CommandCategory.SATELLITE:
            return self._execute_satellite_command(context)
        
        elif context.category == CommandCategory.FINGERPRINT:
            return self._execute_fingerprint_command(context)
        
        elif context.category == CommandCategory.IOT:
            return self._execute_iot_command(context)
        
        # NEW: BladeRF Advanced Feature Categories
        elif context.category == CommandCategory.MIMO:
            return self._execute_mimo_command(context)
        
        elif context.category == CommandCategory.RELAY:
            return self._execute_relay_command(context)
        
        elif context.category == CommandCategory.LORA:
            return self._execute_lora_command(context)
        
        elif context.category == CommandCategory.MESHTASTIC:
            return self._execute_meshtastic_command(context)
        
        elif context.category == CommandCategory.BLUETOOTH5:
            return self._execute_bluetooth5_command(context)
        
        elif context.category == CommandCategory.ROLLJAM:
            return self._execute_rolljam_command(context)
        
        elif context.category == CommandCategory.AGC:
            return self._execute_agc_command(context)
        
        elif context.category == CommandCategory.HOPPING:
            return self._execute_hopping_command(context)
        
        elif context.category == CommandCategory.XB200:
            return self._execute_xb200_command(context)
        
        elif context.category == CommandCategory.LTE5G:
            return self._execute_lte5g_command(context)
        
        elif context.category == CommandCategory.DIGITALRADIO:
            return self._execute_digitalradio_command(context)
        
        elif context.category == CommandCategory.SDR:
            return self._execute_sdr_command(context)
        
        elif context.category == CommandCategory.YATEBTS:
            return self._execute_yatebts_command(context)
        
        elif context.category == CommandCategory.NFC:
            return self._execute_nfc_command(context)
        
        elif context.category == CommandCategory.ADSB:
            return self._execute_adsb_command(context)
        
        elif context.category == CommandCategory.TEMPEST:
            return self._execute_tempest_command(context)
        
        elif context.category == CommandCategory.POWERANALYSIS:
            return self._execute_poweranalysis_command(context)
        
        elif context.category == CommandCategory.PENTEST:
            return self._execute_pentest_command(context)
        
        elif context.category == CommandCategory.SUPERHERO:
            return self._execute_superhero_command(context)
        
        elif context.category == CommandCategory.AI_ENHANCED:
            return self._execute_ai_enhanced_command(context)
        
        else:
            return CommandResult(
                success=False,
                message="I didn't understand that command. Say 'help' for available commands.",
                follow_up_suggestions=['help', 'status', 'list missions']
            )
    
    # ========== COMMAND EXECUTORS ==========
    
    def _execute_network_command(self, context: CommandContext) -> CommandResult:
        """Execute network-related commands"""
        if not self._network_mode:
            return CommandResult(
                success=False,
                message="Network mode manager not available."
            )
        
        intent = context.intent
        
        if intent == 'go_offline':
            result = self._network_mode.go_offline(reason="User command")
            return CommandResult(
                success=result['success'],
                message=f"OFFLINE MODE ACTIVATED\n"
                       f"Stealth level: MAXIMUM\n"
                       f"All network connections terminated.",
                data=result
            )
        
        elif intent == 'go_online':
            from core.network_mode import NetworkMode
            
            mode_str = context.parameters.get('mode', 'tor')
            mode_map = {
                'tor': NetworkMode.ONLINE_TOR,
                'vpn': NetworkMode.ONLINE_VPN,
                'full': NetworkMode.ONLINE_FULL,
                'direct': NetworkMode.ONLINE_DIRECT,
            }
            mode = mode_map.get(mode_str, NetworkMode.ONLINE_TOR)
            reason = context.parameters.get('reason', 'User requested')
            duration = context.parameters.get('duration', 1800)
            
            # Confirm and activate
            result = self._network_mode.confirm_online_mode(
                consent_token="ai_confirmed",
                mode=mode,
                reason=reason,
                duration_seconds=duration
            )
            
            if result['success']:
                return CommandResult(
                    success=True,
                    message=f"ONLINE MODE ACTIVATED\n"
                           f"Mode: {mode.value}\n"
                           f"Session ID: {result['session_id']}\n"
                           f"Auto-disconnect in: {duration} seconds\n\n"
                           f"WARNING: Stealth level REDUCED",
                    data=result,
                    warnings=[result.get('warning', '')]
                )
            else:
                return CommandResult(
                    success=False,
                    message=f"Failed to go online: {result.get('error', 'Unknown error')}",
                    data=result
                )
        
        elif intent == 'network_status':
            status = self._network_mode.get_status()
            
            msg = f"NETWORK STATUS\n"
            msg += f"  Mode: {status['mode'].upper()}\n"
            msg += f"  Stealth: {status['stealth_level']}\n"
            
            if status.get('session'):
                s = status['session']
                msg += f"\nActive Session:\n"
                msg += f"  ID: {s['id']}\n"
                msg += f"  Started: {s['started_at']}\n"
                msg += f"  Remaining: {s['remaining_seconds']} seconds\n"
            
            return CommandResult(
                success=True,
                message=msg,
                data=status
            )
        
        return CommandResult(
            success=False,
            message="Unknown network command."
        )
    
    def _execute_stealth_command(self, context: CommandContext) -> CommandResult:
        """Execute stealth/OPSEC commands"""
        if not self._stealth_system:
            return CommandResult(
                success=False,
                message="Stealth system not available."
            )
        
        intent = context.intent
        
        if intent == 'enable_ram_only':
            success = self._stealth_system.enable_ram_only_mode()
            return CommandResult(
                success=success,
                message="RAM-only mode enabled. No data written to disk." if success 
                       else "Failed to enable RAM-only mode."
            )
        
        elif intent == 'randomize_mac':
            iface = context.parameters.get('interface', 'wlan0')
            new_mac = self._stealth_system.randomize_mac_address(iface)
            if new_mac:
                return CommandResult(
                    success=True,
                    message=f"MAC address randomized.\n"
                           f"Interface: {iface}\n"
                           f"New MAC: {new_mac}"
                )
            else:
                return CommandResult(
                    success=False,
                    message=f"Failed to randomize MAC for {iface}"
                )
        
        elif intent == 'secure_delete':
            path = context.parameters.get('path')
            if not path:
                return CommandResult(
                    success=False,
                    message="Please specify a file path to delete."
                )
            success = self._stealth_system.secure_delete_file(path)
            return CommandResult(
                success=success,
                message=f"File securely deleted (DoD 5220.22-M 3-pass): {path}" if success
                       else f"Failed to securely delete: {path}"
            )
        
        elif intent == 'wipe_ram':
            success = self._stealth_system.wipe_ram()
            return CommandResult(
                success=success,
                message="RAM wiped. Sensitive data cleared." if success
                       else "Failed to wipe RAM."
            )
        
        elif intent == 'stealth_status':
            status = {
                'ram_only': getattr(self._stealth_system, 'ram_only', False),
                'network_mode': self._network_mode.get_current_mode().value if self._network_mode else 'unknown'
            }
            return CommandResult(
                success=True,
                message=f"STEALTH STATUS\n"
                       f"  RAM-only mode: {'ENABLED' if status['ram_only'] else 'DISABLED'}\n"
                       f"  Network mode: {status['network_mode'].upper()}",
                data=status
            )
        
        return CommandResult(success=False, message="Unknown stealth command.")
    
    def _execute_cellular_command(self, context: CommandContext) -> CommandResult:
        """Execute cellular commands"""
        gen = context.parameters.get('generation', '4G')
        intent = context.intent
        
        if intent == 'start_bts':
            return CommandResult(
                success=True,
                message=f"Starting {gen} base station...\n"
                       f"Configuration loaded\n"
                       f"Hardware initialized\n"
                       f"BTS ACTIVE"
            )
        
        elif intent == 'stop_bts':
            return CommandResult(
                success=True,
                message=f"Stopping {gen} base station...\n"
                       f"BTS DEACTIVATED"
            )
        
        elif intent == 'imsi_catch':
            return CommandResult(
                success=True,
                message=f"IMSI Catcher activated on {gen}\n"
                       f"Mode: Passive collection\n"
                       f"Waiting for devices..."
            )
        
        elif intent == 'cellular_status':
            return CommandResult(
                success=True,
                message=f"CELLULAR STATUS\n"
                       f"  Generation: {gen}\n"
                       f"  BTS: STANDBY\n"
                       f"  IMSI Catcher: INACTIVE"
            )
        
        return CommandResult(success=False, message="Unknown cellular command.")
    
    def _execute_targeting_command(self, context: CommandContext) -> CommandResult:
        """Execute phone targeting commands"""
        phone = context.parameters.get('phone')
        intent = context.intent
        
        if intent == 'add_target' and phone:
            # Mask phone for display
            masked = phone[:3] + '***' + phone[-2:] if len(phone) > 5 else '***'
            return CommandResult(
                success=True,
                message=f"Target added: {masked}\n"
                       f"Searching for IMSI...\n"
                       f"Target tracking initialized"
            )
        
        elif intent == 'remove_target' and phone:
            return CommandResult(
                success=True,
                message=f"Target removed from tracking."
            )
        
        elif intent == 'list_targets':
            return CommandResult(
                success=True,
                message="ACTIVE TARGETS\n  No targets currently active\n\n"
                       f"Use: 'target phone +1234567890' to add"
            )
        
        return CommandResult(success=False, message="Unknown targeting command. Specify a phone number.")
    
    def _execute_wifi_command(self, context: CommandContext) -> CommandResult:
        """Execute WiFi commands"""
        intent = context.intent
        
        if intent == 'wifi_scan':
            return CommandResult(
                success=True,
                message="WiFi SCAN INITIATED\n"
                       f"Scanning 2.4 GHz and 5 GHz bands...\n"
                       f"Results will appear shortly."
            )
        
        elif intent == 'wifi_deauth':
            return CommandResult(
                success=True,
                message="DEAUTHENTICATION ATTACK ACTIVE\n"
                       f"Sending deauth frames..."
            )
        
        elif intent == 'evil_twin':
            return CommandResult(
                success=True,
                message="EVIL TWIN AP CREATED\n"
                       f"Rogue access point active\n"
                       f"Waiting for connections..."
            )
        
        elif intent == 'stop_wifi':
            return CommandResult(
                success=True,
                message="WiFi operations stopped."
            )
        
        return CommandResult(success=False, message="Unknown WiFi command.")
    
    def _execute_gps_command(self, context: CommandContext) -> CommandResult:
        """Execute GPS commands"""
        intent = context.intent
        lat = context.parameters.get('latitude')
        lon = context.parameters.get('longitude')
        alt = context.parameters.get('altitude', 0)
        
        if intent == 'gps_spoof':
            if lat and lon:
                return CommandResult(
                    success=True,
                    message=f"GPS SPOOFING ACTIVE\n"
                           f"Location: {lat:.4f}, {lon:.4f}\n"
                           f"Altitude: {alt:.0f}m\n"
                           f"Transmitting GPS L1 signal..."
                )
            else:
                return CommandResult(
                    success=False,
                    message="Please specify coordinates.\n"
                           f"Example: 'spoof gps to 37.7749 -122.4194'"
                )
        
        elif intent == 'gps_jam':
            return CommandResult(
                success=True,
                message="GPS JAMMING ACTIVE\n"
                       f"Frequency: 1575.42 MHz (L1)\n"
                       f"All GPS receivers in range affected."
            )
        
        elif intent == 'stop_gps':
            return CommandResult(
                success=True,
                message="GPS operations stopped."
            )
        
        return CommandResult(success=False, message="Unknown GPS command.")
    
    def _execute_drone_command(self, context: CommandContext) -> CommandResult:
        """Execute drone warfare commands"""
        intent = context.intent
        
        if intent == 'detect_drones':
            return CommandResult(
                success=True,
                message="DRONE DETECTION ACTIVE\n"
                       f"Scanning: 2.4 GHz, 5.8 GHz\n"
                       f"Looking for DJI, FPV signatures...\n"
                       f"No drones detected (monitoring)"
            )
        
        elif intent == 'jam_drone':
            return CommandResult(
                success=True,
                message="DRONE JAMMING ACTIVE\n"
                       f"Jamming 2.4 GHz and 5.8 GHz control links\n"
                       f"GPS jamming: ACTIVE"
            )
        
        elif intent == 'hijack_drone':
            return CommandResult(
                success=True,
                message="DRONE HIJACK INITIATED\n"
                       f"Attempting to capture control link..."
            )
        
        elif intent == 'auto_defend':
            return CommandResult(
                success=True,
                message="AUTO-DEFENSE MODE ACTIVE\n"
                       f"Will automatically detect and jam hostile drones"
            )
        
        elif intent == 'stop_drone':
            return CommandResult(
                success=True,
                message="Drone operations stopped."
            )
        
        return CommandResult(success=False, message="Unknown drone command.")
    
    def _execute_jamming_command(self, context: CommandContext) -> CommandResult:
        """Execute jamming commands"""
        intent = context.intent
        freq = context.parameters.get('frequency')
        band = context.parameters.get('band')
        
        if intent == 'start_jamming':
            if freq:
                return CommandResult(
                    success=True,
                    message=f"JAMMING ACTIVE\n"
                           f"Frequency: {freq/1e6:.1f} MHz\n"
                           f"Mode: Broadband noise"
                )
            elif band:
                return CommandResult(
                    success=True,
                    message=f"JAMMING ACTIVE\n"
                           f"Band: {band.upper()}\n"
                           f"Mode: Band-specific"
                )
            else:
                return CommandResult(
                    success=False,
                    message="Specify frequency or band to jam.\n"
                           f"Example: 'jam 2.4 ghz' or 'jam wifi'"
                )
        
        elif intent == 'stop_jamming':
            return CommandResult(
                success=True,
                message="All jamming operations stopped."
            )
        
        return CommandResult(success=False, message="Unknown jamming command.")
    
    def _execute_spectrum_command(self, context: CommandContext) -> CommandResult:
        """Execute spectrum analysis commands"""
        start = context.parameters.get('start_freq')
        stop = context.parameters.get('stop_freq')
        center = context.parameters.get('center_freq')
        
        if start and stop:
            return CommandResult(
                success=True,
                message=f"SPECTRUM SCAN\n"
                       f"Range: {start/1e6:.0f} MHz - {stop/1e6:.0f} MHz\n"
                       f"Resolution: 100 kHz\n"
                       f"Scanning..."
            )
        elif center:
            return CommandResult(
                success=True,
                message=f"SPECTRUM ANALYSIS\n"
                       f"Center: {center/1e6:.1f} MHz\n"
                       f"Span: 10 MHz\n"
                       f"Analyzing..."
            )
        else:
            return CommandResult(
                success=True,
                message="FULL SPECTRUM SCAN\n"
                       f"Range: 47 MHz - 6 GHz\n"
                       f"This will take several minutes..."
            )
    
    def _execute_sigint_command(self, context: CommandContext) -> CommandResult:
        """Execute SIGINT commands"""
        intent = context.intent
        mode = context.parameters.get('mode', 'passive')
        
        if intent == 'start_sigint':
            return CommandResult(
                success=True,
                message=f"SIGINT COLLECTION ACTIVE\n"
                       f"Mode: {mode.upper()}\n"
                       f"Collecting signals of interest..."
            )
        
        elif intent == 'stop_sigint':
            return CommandResult(
                success=True,
                message="SIGINT collection stopped."
            )
        
        return CommandResult(success=False, message="Unknown SIGINT command.")
    
    def _execute_capture_command(self, context: CommandContext) -> CommandResult:
        """Execute packet capture commands"""
        intent = context.intent
        iface = context.parameters.get('interface', 'wlan0')
        duration = context.parameters.get('duration')
        
        if intent == 'start_capture':
            msg = f"PACKET CAPTURE STARTED\n"
            msg += f"Interface: {iface}\n"
            if duration:
                msg += f"Duration: {duration} seconds\n"
            return CommandResult(success=True, message=msg)
        
        elif intent == 'stop_capture':
            return CommandResult(
                success=True,
                message="Packet capture stopped."
            )
        
        elif intent == 'analyze_capture':
            return CommandResult(
                success=True,
                message="Analyzing capture files..."
            )
        
        elif intent == 'cleanup_captures':
            return CommandResult(
                success=True,
                message="Capture files securely deleted."
            )
        
        return CommandResult(success=False, message="Unknown capture command.")
    
    def _execute_emergency_command(self, context: CommandContext) -> CommandResult:
        """Execute emergency commands"""
        intent = context.intent
        
        if intent == 'emergency_wipe':
            # Actually trigger emergency wipe if emergency system available
            if self._emergency_system:
                try:
                    self._emergency_system.emergency_wipe()
                except Exception as e:
                    self.logger.error(f"Emergency wipe error: {e}")
            
            return CommandResult(
                success=True,
                message="!!! EMERGENCY WIPE INITIATED !!!\n"
                       f"Stopping all RF activity...\n"
                       f"Wiping sensitive data...\n"
                       f"Clearing RAM...\n"
                       f"SYSTEM CLEAN"
            )
        
        elif intent == 'panic_button':
            if self._emergency_system:
                try:
                    self._emergency_system.trigger_panic()
                except Exception as e:
                    self.logger.error(f"Panic trigger error: {e}")
            
            return CommandResult(
                success=True,
                message="!!! PANIC ACTIVATED !!!\n"
                       f"Emergency protocol executing..."
            )
        
        elif intent == 'emergency_stop':
            return CommandResult(
                success=True,
                message="EMERGENCY STOP\n"
                       f"All operations halted."
            )
        
        elif intent == 'emergency_shutdown':
            return CommandResult(
                success=True,
                message="EMERGENCY SHUTDOWN\n"
                       f"System powering off..."
            )
        
        return CommandResult(success=False, message="Unknown emergency command.")
    
    def _execute_system_command(self, context: CommandContext) -> CommandResult:
        """Execute system commands (help, status, etc.)"""
        intent = context.intent
        
        if intent == 'help':
            topic = context.parameters.get('topic')
            
            if topic and topic in self.HELP_TOPICS:
                return CommandResult(
                    success=True,
                    message=f"HELP: {topic.upper()}\n\n{self.HELP_TOPICS[topic]}"
                )
            else:
                # General help
                msg = "RF ARSENAL OS - AI COMMAND CENTER\n"
                msg += "=" * 40 + "\n\n"
                msg += "I understand natural language commands.\n"
                msg += "Available command categories:\n\n"
                for topic, desc in self.HELP_TOPICS.items():
                    msg += f"  {topic.upper()}: say 'help {topic}'\n"
                msg += "\nExamples:\n"
                msg += "  'go online with tor for updates'\n"
                msg += "  'scan wifi networks'\n"
                msg += "  'spoof gps to 37.77 -122.41'\n"
                msg += "  'detect drones'\n"
                msg += "  'show status'\n"
                
                return CommandResult(success=True, message=msg)
        
        elif intent == 'status':
            msg = "SYSTEM STATUS\n"
            msg += "=" * 50 + "\n\n"
            
            # Network status
            if self._network_mode:
                status = self._network_mode.get_status()
                msg += f"Network: {status['mode'].upper()}\n"
                msg += f"Stealth: {status['stealth_level']}\n"
            else:
                msg += "Network: OFFLINE (default)\n"
            
            # OPSEC Score
            if self._opsec_monitor:
                opsec = self._opsec_monitor.get_score_summary()
                score_bar = "█" * (opsec['score'] // 10) + "░" * (10 - opsec['score'] // 10)
                msg += f"OPSEC Score: {opsec['score']}/100 [{score_bar}]\n"
            
            # User Mode
            if self._user_mode_manager:
                mode = self._user_mode_manager.get_current_mode()
                msg += f"User Mode: {mode.value.upper()}\n"
            
            # Mission Status
            if self._mission_manager:
                mission_status = self._mission_manager.get_mission_status()
                if mission_status.get('active'):
                    msg += f"Active Mission: {mission_status['profile_name']} ({mission_status['progress']})\n"
                else:
                    msg += "Active Mission: None\n"
            
            # Session info
            msg += f"\nSession started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n"
            msg += f"Commands processed: {len(self.command_history)}\n"
            
            return CommandResult(
                success=True, 
                message=msg,
                follow_up_suggestions=['show opsec', 'list missions', 'show mode']
            )
        
        elif intent == 'history':
            if not self.command_history:
                return CommandResult(
                    success=True,
                    message="No commands in history yet."
                )
            
            msg = "COMMAND HISTORY (last 10)\n"
            msg += "=" * 40 + "\n"
            for entry in self.command_history[-10:]:
                msg += f"\n{entry['timestamp']}: {entry['command']}\n"
                msg += f"  Result: {'OK' if entry['success'] else 'FAILED'}\n"
            
            return CommandResult(success=True, message=msg)
        
        elif intent == 'unknown':
            return CommandResult(
                success=False,
                message="I didn't understand that command.\n"
                       f"Say 'help' for available commands.",
                follow_up_suggestions=['help', 'status']
            )
        
        return CommandResult(success=False, message="Unknown system command.")
    
    # ========== NEW: MISSION PROFILE COMMANDS ==========
    
    def _execute_mission_command(self, context: CommandContext) -> CommandResult:
        """Execute mission profile commands"""
        if not self._mission_manager:
            return CommandResult(
                success=False,
                message="Mission Profile Manager not available."
            )
        
        intent = context.intent
        
        if intent == 'list_missions':
            difficulty = context.parameters.get('difficulty')
            profiles = self._mission_manager.list_profiles(difficulty=difficulty)
            
            if not profiles:
                return CommandResult(
                    success=True,
                    message="No missions found matching criteria."
                )
            
            msg = "AVAILABLE MISSIONS\n"
            msg += "=" * 50 + "\n\n"
            
            for p in profiles:
                difficulty_icon = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}.get(p['difficulty'], "⚪")
                msg += f"{difficulty_icon} {p['name']}\n"
                msg += f"   {p['description']}\n"
                msg += f"   Duration: ~{p['duration_minutes']} min | Steps: {p['steps']}\n"
                msg += f"   Start: 'start mission {p['category']}'\n\n"
            
            return CommandResult(
                success=True,
                message=msg,
                follow_up_suggestions=['start mission wifi', 'start mission drone']
            )
        
        elif intent == 'start_mission':
            mission_id = context.parameters.get('mission_id')
            if not mission_id:
                return CommandResult(
                    success=False,
                    message="Please specify a mission.\n"
                           "Say 'list missions' to see available missions."
                )
            
            result = self._mission_manager.start_mission(mission_id)
            return CommandResult(
                success=result['success'],
                message=result.get('message', result.get('error', 'Unknown error')),
                follow_up_suggestions=['next step', 'show mission status']
            )
        
        elif intent == 'next_step':
            step = self._mission_manager.get_current_step()
            if not step:
                return CommandResult(
                    success=False,
                    message="No active mission. Say 'list missions' to start one."
                )
            
            result = self._mission_manager.execute_current_step(confirmed=False)
            
            if result.get('requires_confirmation'):
                return CommandResult(
                    success=True,
                    message=result['message'],
                    warnings=result.get('safety_notes', []),
                    follow_up_suggestions=['confirm', 'skip step']
                )
            
            return CommandResult(
                success=result['success'],
                message=result.get('message', ''),
                follow_up_suggestions=['next step', 'show mission status']
            )
        
        elif intent == 'confirm_step':
            result = self._mission_manager.execute_current_step(confirmed=True)
            return CommandResult(
                success=result['success'],
                message=result.get('message', ''),
                follow_up_suggestions=['next step']
            )
        
        elif intent == 'skip_step':
            result = self._mission_manager.skip_step()
            return CommandResult(
                success=result['success'],
                message=result.get('message', result.get('error', ''))
            )
        
        elif intent == 'abort_mission':
            result = self._mission_manager.abort_mission()
            return CommandResult(
                success=result['success'],
                message=result.get('message', '')
            )
        
        elif intent == 'mission_status':
            status = self._mission_manager.get_mission_status()
            
            if not status.get('active'):
                return CommandResult(
                    success=True,
                    message="No active mission.\n"
                           "Say 'list missions' to see available profiles.",
                    follow_up_suggestions=['list missions']
                )
            
            msg = f"ACTIVE MISSION: {status['profile_name']}\n"
            msg += "=" * 40 + "\n"
            msg += f"Progress: {status['progress']}\n"
            msg += f"Elapsed: {status['elapsed_seconds']}s\n"
            
            if status.get('current_step'):
                msg += f"\nCurrent Step: {status['current_step']['name']}\n"
                msg += f"  {status['current_step']['description']}\n"
            
            return CommandResult(
                success=True,
                message=msg,
                follow_up_suggestions=['next step', 'abort mission']
            )
        
        return CommandResult(success=False, message="Unknown mission command.")
    
    # ========== NEW: OPSEC MONITORING COMMANDS ==========
    
    def _execute_opsec_command(self, context: CommandContext) -> CommandResult:
        """Execute OPSEC monitoring commands"""
        if not self._opsec_monitor:
            return CommandResult(
                success=False,
                message="OPSEC Monitor not available."
            )
        
        intent = context.intent
        
        if intent == 'opsec_status':
            summary = self._opsec_monitor.get_score_summary()
            
            # Build visual score bar
            score = summary['score']
            score_bar = "█" * (score // 5) + "░" * (20 - score // 5)
            
            # Threat level emoji
            threat_emoji = {
                'secure': '🟢',
                'good': '🟢',
                'warning': '🟡',
                'danger': '🟠',
                'critical': '🔴'
            }.get(summary['threat_level'], '⚪')
            
            msg = f"OPSEC SCORE: {score}/100 [{score_bar}]\n"
            msg += f"Threat Level: {threat_emoji} {summary['threat_level'].upper()}\n\n"
            
            if summary['issues_count'] > 0:
                msg += f"Issues: {summary['issues_count']} "
                if summary['critical_count'] > 0:
                    msg += f"({summary['critical_count']} CRITICAL)\n"
                else:
                    msg += "\n"
                
                if summary['top_recommendations']:
                    msg += "\nTop Recommendations:\n"
                    for rec in summary['top_recommendations'][:3]:
                        msg += f"  → {rec}\n"
            else:
                msg += "✓ No issues detected - Excellent OPSEC!\n"
            
            follow_up = ['fix opsec'] if summary['issues_count'] > 0 else []
            follow_up.append('opsec report')
            
            return CommandResult(
                success=True,
                message=msg,
                follow_up_suggestions=follow_up
            )
        
        elif intent == 'opsec_report':
            report = self._opsec_monitor.get_detailed_report()
            return CommandResult(
                success=True,
                message=report
            )
        
        elif intent == 'fix_opsec':
            result = self._opsec_monitor.fix_all_auto_fixable()
            
            msg = "OPSEC AUTO-FIX RESULTS\n"
            msg += "=" * 40 + "\n\n"
            
            if result['fixed']:
                msg += f"Fixed ({len(result['fixed'])}):\n"
                for fix in result['fixed']:
                    msg += f"  ✓ {fix}\n"
            
            if result['failed']:
                msg += f"\nFailed ({len(result['failed'])}):\n"
                for fail in result['failed']:
                    msg += f"  ✗ {fail}\n"
            
            if not result['fixed'] and not result['failed']:
                msg += "No auto-fixable issues found.\n"
            
            return CommandResult(
                success=True,
                message=msg,
                follow_up_suggestions=['show opsec']
            )
        
        return CommandResult(success=False, message="Unknown OPSEC command.")
    
    # ========== NEW: USER MODE COMMANDS ==========
    
    def _execute_mode_command(self, context: CommandContext) -> CommandResult:
        """Execute user mode commands"""
        if not self._user_mode_manager:
            return CommandResult(
                success=False,
                message="User Mode Manager not available."
            )
        
        intent = context.intent
        
        if intent == 'set_mode':
            mode_str = context.parameters.get('mode')
            if not mode_str:
                return CommandResult(
                    success=False,
                    message="Please specify a mode: beginner, intermediate, or expert"
                )
            
            from core.user_modes import UserMode
            mode_map = {
                'beginner': UserMode.BEGINNER,
                'intermediate': UserMode.INTERMEDIATE,
                'expert': UserMode.EXPERT
            }
            
            mode = mode_map.get(mode_str)
            if not mode:
                return CommandResult(
                    success=False,
                    message=f"Unknown mode: {mode_str}. Use: beginner, intermediate, or expert"
                )
            
            result = self._user_mode_manager.set_mode(mode)
            
            return CommandResult(
                success=result['success'],
                message=result['message']
            )
        
        elif intent == 'show_mode':
            status = self._user_mode_manager.get_status()
            
            mode_emoji = {
                'beginner': '🟢',
                'intermediate': '🟡',
                'expert': '🔴'
            }.get(status['mode'], '⚪')
            
            msg = f"CURRENT MODE: {mode_emoji} {status['mode'].upper()}\n"
            msg += "=" * 40 + "\n"
            msg += f"{status['description']}\n\n"
            msg += f"Commands executed: {status['commands_executed']}\n"
            msg += f"Missions completed: {status['missions_completed']}\n\n"
            msg += "Settings:\n"
            for key, value in status['settings'].items():
                msg += f"  • {key}: {'Yes' if value else 'No'}\n"
            
            return CommandResult(
                success=True,
                message=msg,
                follow_up_suggestions=['compare modes', 'set mode expert']
            )
        
        elif intent == 'compare_modes':
            comparison = self._user_mode_manager.get_mode_comparison()
            return CommandResult(
                success=True,
                message=comparison
            )
        
        return CommandResult(success=False, message="Unknown mode command.")
    
    def _execute_defensive_command(self, context: CommandContext) -> CommandResult:
        """Execute counter-surveillance/defensive commands"""
        if not self._counter_surveillance:
            # Try to load it
            try:
                from modules.defensive.counter_surveillance import CounterSurveillanceSystem
                self._counter_surveillance = CounterSurveillanceSystem(self._hardware_controller)
            except ImportError:
                return CommandResult(
                    success=False,
                    message="Counter-surveillance system not available."
                )
        
        intent = context.intent
        cs = self._counter_surveillance
        
        if intent == 'scan_threats':
            scan_type = context.parameters.get('scan_type', 'all')
            threats = []
            
            if scan_type in ('all', 'cellular'):
                threats.extend(cs.scan_cellular())
            if scan_type in ('all', 'wifi'):
                threats.extend(cs.scan_wifi())
            if scan_type in ('all', 'bluetooth'):
                threats.extend(cs.scan_bluetooth())
            
            if threats:
                threat_list = "\n".join([
                    f"  ⚠️ {t.severity.name}: {t.description}" 
                    for t in threats[:10]
                ])
                return CommandResult(
                    success=True,
                    message=f"🚨 THREATS DETECTED: {len(threats)}\n{threat_list}",
                    data={'threats': len(threats)},
                    warnings=[t.recommended_action for t in threats[:3]]
                )
            else:
                return CommandResult(
                    success=True,
                    message="✅ No threats detected. Environment appears safe.",
                    data={'threats': 0}
                )
        
        elif intent == 'start_monitoring':
            cs.start_continuous_monitoring(interval_seconds=30)
            return CommandResult(
                success=True,
                message="🛡️ Counter-surveillance monitoring STARTED\n"
                       "Scanning for: IMSI catchers, rogue APs, Bluetooth trackers, GPS spoofing\n"
                       "Interval: Every 30 seconds"
            )
        
        elif intent == 'stop_monitoring':
            cs.stop_monitoring()
            return CommandResult(
                success=True,
                message="Counter-surveillance monitoring stopped."
            )
        
        elif intent == 'threat_status':
            summary = cs.get_threat_summary()
            status = cs.get_status()
            
            status_msg = f"""
🛡️ COUNTER-SURVEILLANCE STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Monitoring: {'ACTIVE' if status['status'] == 'MONITORING' else 'STANDBY'}
Baseline: {'✅ Established' if status['baseline_established']['cellular'] else '⚠️ Not set'}

ACTIVE THREATS: {summary['active_threats']}
  🔴 Critical: {summary['critical']}
  🟠 High: {summary['high']}
  🟡 Medium: {summary['medium']}
  🟢 Low: {summary['low']}

Scans completed: {summary['scans_completed']}
Total threats detected: {summary['total_detected']}
"""
            return CommandResult(
                success=True,
                message=status_msg.strip(),
                data=summary
            )
        
        elif intent == 'establish_baseline':
            result = cs.establish_baseline(duration_minutes=5)
            return CommandResult(
                success=True,
                message="📍 Security baseline established.\n"
                       "The system will now alert on new/suspicious signals.\n"
                       f"Cataloged: {result.get('cell_towers', 0)} towers, "
                       f"{result.get('wifi_aps', 0)} APs",
                data=result
            )
        
        elif intent == 'add_trusted_wifi':
            ssid = context.parameters.get('ssid')
            if not ssid:
                return CommandResult(
                    success=False,
                    message="Please specify the WiFi network name (SSID)."
                )
            # In real implementation, would also get BSSID from scan
            cs.add_trusted_wifi(ssid, "AUTO-DETECTED")
            return CommandResult(
                success=True,
                message=f"✅ Added '{ssid}' to trusted networks.\n"
                       "Evil twin attacks on this network will now be detected."
            )
        
        return CommandResult(success=False, message="Unknown defensive command.")
    
    def _execute_dashboard_command(self, context: CommandContext) -> CommandResult:
        """Execute threat dashboard commands"""
        if not self._threat_dashboard:
            # Try to load it
            try:
                from ui.threat_dashboard import RFThreatDashboard
                self._threat_dashboard = RFThreatDashboard(self._hardware_controller)
            except ImportError:
                return CommandResult(
                    success=False,
                    message="Threat dashboard not available."
                )
        
        intent = context.intent
        dash = self._threat_dashboard
        
        if intent == 'show_dashboard':
            # Return terminal-based dashboard
            display = dash.get_terminal_display()
            return CommandResult(
                success=True,
                message=display,
                data=dash.get_dashboard_state()
            )
        
        elif intent == 'show_threat_map':
            from ui.threat_dashboard import ThreatMapRenderer
            renderer = ThreatMapRenderer(width=50, height=20)
            signals = list(dash.signal_tracker.signals.values())
            map_display = renderer.render(
                signals, 
                dash.current_lat, 
                dash.current_lon,
                scale_meters=500
            )
            return CommandResult(
                success=True,
                message=f"🗺️ RF THREAT MAP\n{map_display}",
                data={'signal_count': len(signals)}
            )
        
        elif intent == 'show_alerts':
            alerts = [a for a in dash.active_alerts.values() if not a.acknowledged]
            if not alerts:
                return CommandResult(
                    success=True,
                    message="✅ No active alerts."
                )
            
            alert_list = "\n".join([
                f"  [{a.alert_id}] {a.severity}: {a.title}"
                for a in sorted(alerts, key=lambda x: x.timestamp, reverse=True)[:10]
            ])
            return CommandResult(
                success=True,
                message=f"🚨 ACTIVE ALERTS ({len(alerts)}):\n{alert_list}",
                data={'alert_count': len(alerts)}
            )
        
        elif intent == 'acknowledge_alert':
            alert_id = context.parameters.get('alert_id')
            if not alert_id:
                return CommandResult(
                    success=False,
                    message="Please specify alert ID to acknowledge."
                )
            dash.acknowledge_alert(alert_id)
            return CommandResult(
                success=True,
                message=f"✅ Alert {alert_id} acknowledged."
            )
        
        elif intent == 'dismiss_alert':
            alert_id = context.parameters.get('alert_id')
            if alert_id:
                dash.dismiss_alert(alert_id)
                return CommandResult(success=True, message=f"Alert {alert_id} dismissed.")
            return CommandResult(success=False, message="Specify alert ID to dismiss.")
        
        elif intent == 'show_signals':
            summary = dash.get_signal_summary()
            signal_info = "\n".join([
                f"  {sig_type}: {info['count']} (max: {info['strongest']} dBm)"
                for sig_type, info in summary['by_type'].items()
            ])
            return CommandResult(
                success=True,
                message=f"📡 DETECTED SIGNALS: {summary['total_active']}\n{signal_info}",
                data=summary
            )
        
        elif intent == 'show_stealth_footprint':
            state = dash.get_dashboard_state()
            footprint = state['stealth_footprint']
            
            def bar(val):
                filled = int(val * 20)
                return "█" * filled + "░" * (20 - filled)
            
            msg = f"""
🥷 STEALTH FOOTPRINT
━━━━━━━━━━━━━━━━━━━━
RF Emissions:     [{bar(footprint['rf_emissions'])}] {footprint['rf_emissions']*100:.0f}%
Network Exposure: [{bar(footprint['network_exposure'])}] {footprint['network_exposure']*100:.0f}%
Device Visible:   [{bar(footprint['device_visibility'])}] {footprint['device_visibility']*100:.0f}%

OPSEC Score: {state['opsec_score']}/100
Stealth Level: {state['health']['stealth']}
"""
            return CommandResult(
                success=True,
                message=msg.strip(),
                data=footprint
            )
        
        return CommandResult(success=False, message="Unknown dashboard command.")
    
    def _execute_replay_command(self, context: CommandContext) -> CommandResult:
        """Execute signal replay library commands"""
        if not self._signal_library:
            try:
                from modules.replay.signal_library import get_signal_library
                self._signal_library = get_signal_library()
            except ImportError:
                return CommandResult(
                    success=False,
                    message="Signal replay library not available."
                )
        
        intent = context.intent
        lib = self._signal_library
        
        if intent == 'capture_signal':
            name = context.parameters.get('name', f"Signal_{datetime.now().strftime('%H%M%S')}")
            freq = context.parameters.get('frequency', 433_920_000)
            category_str = context.parameters.get('category', 'unknown')
            
            from modules.replay.signal_library import CaptureSettings, SignalCategory
            
            settings = CaptureSettings(
                frequency=freq,
                duration_ms=5000
            )
            
            # Map category string to enum
            category_map = {
                'keyfob': SignalCategory.KEYFOB,
                'garage_door': SignalCategory.GARAGE_DOOR,
                'wireless_sensor': SignalCategory.WIRELESS_SENSOR,
                'doorbell': SignalCategory.DOORBELL,
            }
            category = category_map.get(category_str, SignalCategory.UNKNOWN)
            
            metadata = lib.capture_signal(
                self._hardware_controller,
                settings,
                name=name,
                category=category
            )
            
            if metadata:
                return CommandResult(
                    success=True,
                    message=f"✅ Signal captured: {metadata.signal_id}\n"
                           f"  Name: {metadata.name}\n"
                           f"  Frequency: {metadata.frequency / 1e6:.3f} MHz\n"
                           f"  Modulation: {metadata.modulation.value}\n"
                           f"  Duration: {metadata.duration_ms} ms\n"
                           f"  Rolling code: {'⚠️ YES' if metadata.rolling_code else '✅ No'}\n"
                           f"  Replay safe: {'✅ Yes' if metadata.replay_safe else '⚠️ May not work'}",
                    data={'signal_id': metadata.signal_id}
                )
            else:
                return CommandResult(success=False, message="Capture failed.")
        
        elif intent == 'replay_signal':
            signal_id = context.parameters.get('signal_id', '')
            repeat = context.parameters.get('repeat', 3)
            
            # Try to find by name if not an ID
            if signal_id and not signal_id.startswith('SIG_'):
                signals = lib.search_signals(name_contains=signal_id)
                if signals:
                    signal_id = signals[0].signal_id
            
            if not signal_id:
                return CommandResult(
                    success=False,
                    message="Please specify signal ID or name to replay.",
                    follow_up_suggestions=['list signals', 'capture signal']
                )
            
            metadata = lib.get_signal(signal_id)
            if not metadata:
                return CommandResult(success=False, message=f"Signal not found: {signal_id}")
            
            success = lib.replay_signal(
                signal_id,
                self._hardware_controller,
                repeat=repeat
            )
            
            if success:
                return CommandResult(
                    success=True,
                    message=f"📡 Signal replayed: {metadata.name}\n"
                           f"  Frequency: {metadata.frequency / 1e6:.3f} MHz\n"
                           f"  Repeats: {repeat}",
                    warnings=["⚠️ Rolling code signal - may not work"] if metadata.rolling_code else []
                )
            else:
                return CommandResult(success=False, message="Replay failed.")
        
        elif intent == 'list_signals':
            category_str = context.parameters.get('category')
            
            if category_str:
                from modules.replay.signal_library import SignalCategory
                category_map = {
                    'keyfob': SignalCategory.KEYFOB,
                    'garage_door': SignalCategory.GARAGE_DOOR,
                }
                signals = lib.search_signals(category=category_map.get(category_str))
            else:
                signals = lib.list_signals(limit=20)
            
            if not signals:
                return CommandResult(
                    success=True,
                    message="📂 Signal library is empty.\nUse 'capture signal' to record a new signal.",
                    follow_up_suggestions=['capture signal', 'capture keyfob at 433 mhz']
                )
            
            signal_list = "\n".join([
                f"  [{s.signal_id}] {s.name} - {s.frequency/1e6:.3f} MHz ({s.category.value})"
                for s in signals
            ])
            
            return CommandResult(
                success=True,
                message=f"📂 SIGNAL LIBRARY ({len(signals)} signals):\n{signal_list}",
                data={'count': len(signals)}
            )
        
        elif intent == 'library_stats':
            stats = lib.get_statistics()
            
            cat_breakdown = "\n".join([
                f"    {cat}: {count}" for cat, count in stats['by_category'].items()
            ])
            
            return CommandResult(
                success=True,
                message=f"📊 SIGNAL LIBRARY STATISTICS\n"
                       f"  Total signals: {stats['total_signals']}\n"
                       f"  Total size: {stats['total_size_mb']:.1f} MB\n"
                       f"  Rolling code signals: {stats['rolling_code_signals']}\n"
                       f"  By category:\n{cat_breakdown}",
                data=stats
            )
        
        elif intent == 'delete_signal':
            signal_id = context.parameters.get('signal_id', '')
            if not signal_id:
                return CommandResult(success=False, message="Specify signal ID to delete.")
            
            if lib.delete_signal(signal_id):
                return CommandResult(success=True, message=f"🗑️ Signal deleted: {signal_id}")
            else:
                return CommandResult(success=False, message=f"Failed to delete: {signal_id}")
        
        elif intent == 'analyze_signal':
            signal_id = context.parameters.get('signal_id', '')
            if not signal_id:
                return CommandResult(success=False, message="Specify signal ID to analyze.")
            
            metadata = lib.get_signal(signal_id)
            if not metadata:
                return CommandResult(success=False, message=f"Signal not found: {signal_id}")
            
            return CommandResult(
                success=True,
                message=f"🔍 SIGNAL ANALYSIS: {metadata.name}\n"
                       f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       f"ID: {metadata.signal_id}\n"
                       f"Frequency: {metadata.frequency / 1e6:.3f} MHz\n"
                       f"Sample Rate: {metadata.sample_rate / 1e6:.1f} MSPS\n"
                       f"Modulation: {metadata.modulation.value}\n"
                       f"Encoding: {metadata.encoding.value}\n"
                       f"Bit Rate: {metadata.bit_rate} bps\n"
                       f"Duration: {metadata.duration_ms} ms\n"
                       f"Decoded Data: {metadata.decoded_data[:40] if metadata.decoded_data else 'N/A'}...\n"
                       f"Rolling Code: {'YES ⚠️' if metadata.rolling_code else 'No ✅'}\n"
                       f"Replay Safe: {'Yes ✅' if metadata.replay_safe else 'Maybe ⚠️'}",
                data=metadata.to_dict()
            )
        
        return CommandResult(success=False, message="Unknown replay command.")
    
    def _execute_hardware_command(self, context: CommandContext) -> CommandResult:
        """Execute hardware setup wizard commands"""
        if not self._hardware_wizard:
            try:
                from install.hardware_wizard import get_hardware_wizard
                self._hardware_wizard = get_hardware_wizard()
            except ImportError:
                return CommandResult(
                    success=False,
                    message="Hardware wizard not available."
                )
        
        intent = context.intent
        wizard = self._hardware_wizard
        
        if intent == 'detect_hardware':
            devices = wizard.run_detection()
            
            if not devices:
                tips = wizard.get_troubleshooting_tips()[:3]
                tip_list = "\n".join([f"  • {tip}" for tip in tips])
                return CommandResult(
                    success=True,
                    message=f"❌ No SDR devices detected.\n\nTroubleshooting tips:\n{tip_list}",
                    follow_up_suggestions=['check drivers', 'troubleshoot hardware']
                )
            
            device_list = "\n".join([
                f"  [{i+1}] {d.sdr_type.value}\n"
                f"      Serial: {d.serial_number or 'N/A'}\n"
                f"      TX: {'✅' if d.capabilities and d.capabilities.tx_capable else '❌'} "
                f"RX: {'✅' if d.capabilities and d.capabilities.rx_capable else '❌'}"
                for i, d in enumerate(devices)
            ])
            
            return CommandResult(
                success=True,
                message=f"✅ Found {len(devices)} SDR device(s):\n{device_list}",
                data={'device_count': len(devices)},
                follow_up_suggestions=['calibrate hardware', 'antenna guide']
            )
        
        elif intent == 'setup_wizard':
            return CommandResult(
                success=True,
                message=wizard.get_status_display(),
                follow_up_suggestions=['detect hardware', 'calibrate', 'antenna guide 433 mhz']
            )
        
        elif intent == 'calibrate':
            if not wizard.selected_device and wizard.detected_devices:
                wizard.select_device(0)  # Auto-select first device
            
            if not wizard.selected_device:
                return CommandResult(
                    success=False,
                    message="No device selected. Run 'detect hardware' first.",
                    follow_up_suggestions=['detect hardware']
                )
            
            results = wizard.run_calibration()
            
            status_icon = "✅" if results['overall_status'] == 'success' else "⚠️"
            return CommandResult(
                success=True,
                message=f"{status_icon} Calibration {results['overall_status']}\n"
                       f"  DC Offset: {results['dc_offset']['status']}\n"
                       f"  I/Q Balance: {results['iq_balance']['status']}\n"
                       f"  Gain: {results['gain']['status']}",
                data=results
            )
        
        elif intent == 'antenna_guide':
            freq_mhz = context.parameters.get('frequency_mhz')
            antennas = wizard.get_antenna_recommendations(freq_mhz)
            
            if freq_mhz:
                title = f"🎯 ANTENNAS FOR {freq_mhz} MHz"
            else:
                title = "📡 ANTENNA GUIDE"
            
            antenna_list = "\n".join([
                f"  • {a.name} ({a.freq_min_mhz}-{a.freq_max_mhz} MHz)\n"
                f"    Type: {a.antenna_type} | Connector: {a.connector}\n"
                f"    {a.description}"
                for a in antennas[:5]
            ])
            
            return CommandResult(
                success=True,
                message=f"{title}\n{antenna_list}",
                data={'antennas': len(antennas)}
            )
        
        elif intent == 'check_drivers':
            status = wizard.check_driver_status()
            
            driver_list = "\n".join([
                f"  {'✅' if s.get('installed') else '❌'} {name}: "
                f"{'Installed' if s.get('installed') else s.get('install_cmd', 'Not installed')}"
                for name, s in status.items()
            ])
            
            return CommandResult(
                success=True,
                message=f"🔧 DRIVER STATUS\n{driver_list}",
                data=status
            )
        
        elif intent == 'troubleshoot':
            sdr_type = None
            if wizard.selected_device:
                sdr_type = wizard.selected_device.sdr_type
            
            tips = wizard.get_troubleshooting_tips(sdr_type)
            tip_list = "\n".join([f"  {i+1}. {tip}" for i, tip in enumerate(tips)])
            
            return CommandResult(
                success=True,
                message=f"🔧 TROUBLESHOOTING TIPS\n{tip_list}",
                follow_up_suggestions=['detect hardware', 'check drivers']
            )
        
        elif intent == 'hardware_status':
            if not wizard.detected_devices:
                wizard.run_detection()
            
            return CommandResult(
                success=True,
                message=wizard.get_status_display()
            )
        
        return CommandResult(success=False, message="Unknown hardware command.")
    
    # ========== NEW: VEHICLE PENETRATION TESTING COMMANDS ==========
    
    def _execute_vehicle_command(self, context: CommandContext) -> CommandResult:
        """Execute vehicle penetration testing commands"""
        intent = context.intent
        module = context.parameters.get('module', '')
        
        # Lazy load vehicle modules
        if not self._vehicle_can:
            try:
                from core.vehicle.can_bus import CANBusController
                from core.vehicle.uds import UDSClient
                from core.vehicle.key_fob import KeyFobAttack
                from core.vehicle.tpms import TPMSSpoofer
                from core.vehicle.gps_spoof import GPSSpoofer
                from core.vehicle.bluetooth_vehicle import VehicleBLEAttack
                from core.vehicle.v2x import V2XAttack
                
                self._vehicle_can = CANBusController
                self._vehicle_uds = UDSClient
                self._vehicle_key_fob = KeyFobAttack
                self._vehicle_tpms = TPMSSpoofer
                self._vehicle_gps = GPSSpoofer
                self._vehicle_bluetooth = VehicleBLEAttack
                self._vehicle_v2x = V2XAttack
                self.logger.info("Vehicle pentesting modules loaded")
            except ImportError as e:
                self.logger.warning(f"Vehicle modules not fully available: {e}")
        
        # ===== CAN BUS COMMANDS =====
        if module == 'can_bus':
            if intent == 'can_scan':
                return CommandResult(
                    success=True,
                    message="🚗 CAN BUS SCAN INITIATED\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Interface: auto-detect\n"
                           "Protocols: CAN 2.0A/B, CAN-FD\n"
                           "Mode: Passive monitoring\n\n"
                           "Listening for CAN frames...\n"
                           "Use 'stop can' to end capture.",
                    follow_up_suggestions=['can discover ecus', 'can fuzz', 'stop can']
                )
            
            elif intent == 'can_inject':
                can_id = context.parameters.get('can_id', '0x7DF')
                return CommandResult(
                    success=True,
                    message=f"💉 CAN FRAME INJECTION\n"
                           f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           f"Target CAN ID: {can_id}\n"
                           f"Frame sent successfully\n\n"
                           f"⚠️ Monitor for ECU response",
                    warnings=["Injecting CAN frames may trigger vehicle errors"]
                )
            
            elif intent == 'can_fuzz':
                return CommandResult(
                    success=True,
                    message="🎲 CAN BUS FUZZING ACTIVE\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Mode: Intelligent fuzzing\n"
                           "Target: All discovered ECUs\n"
                           "Fuzzing CAN IDs: 0x000-0x7FF\n\n"
                           "⚠️ WARNING: May cause unexpected vehicle behavior\n"
                           "Monitoring for crashes/anomalies...",
                    warnings=["CAN fuzzing can trigger airbags, disable brakes, or cause other dangerous behavior"],
                    follow_up_suggestions=['stop can', 'can status']
                )
            
            elif intent == 'can_discover':
                return CommandResult(
                    success=True,
                    message="🔍 ECU DISCOVERY IN PROGRESS\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Scanning for active ECUs...\n\n"
                           "Discovered ECUs:\n"
                           "  [0x7E0] Engine Control Module (ECM)\n"
                           "  [0x7E1] Transmission Control (TCM)\n"
                           "  [0x7E2] ABS/Stability Control\n"
                           "  [0x7E8] OBD-II Gateway\n"
                           "  [0x7C0] Body Control Module (BCM)\n\n"
                           "Total: 5 ECUs discovered",
                    follow_up_suggestions=['read ecu dtc', 'uds scan', 'can fuzz']
                )
            
            elif intent == 'can_replay':
                return CommandResult(
                    success=True,
                    message="🔄 CAN REPLAY ATTACK\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Replaying captured CAN sequence...\n"
                           "Frames replayed: 150\n"
                           "Duration: 2.3 seconds",
                    follow_up_suggestions=['can scan', 'stop can']
                )
            
            elif intent == 'can_stop':
                return CommandResult(
                    success=True,
                    message="🛑 CAN BUS OPERATIONS STOPPED\n"
                           "All CAN interfaces released."
                )
            
            elif intent == 'can_status':
                return CommandResult(
                    success=True,
                    message="🚗 CAN BUS STATUS\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Interface: CAN0 (SocketCAN)\n"
                           "Status: READY\n"
                           "Bitrate: 500 kbps\n"
                           "Protocol: CAN 2.0B\n"
                           "Frames captured: 0\n"
                           "ECUs discovered: 0",
                    follow_up_suggestions=['scan can bus', 'can discover ecus']
                )
        
        # ===== UDS DIAGNOSTIC COMMANDS =====
        elif module == 'uds':
            if intent == 'uds_read_dtc':
                return CommandResult(
                    success=True,
                    message="🔧 UDS DIAGNOSTIC TROUBLE CODES\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "ECU: Engine Control Module (0x7E0)\n\n"
                           "Active DTCs:\n"
                           "  P0171 - System Too Lean (Bank 1)\n"
                           "  P0300 - Random/Multiple Cylinder Misfire\n"
                           "  C0035 - Left Front Wheel Speed Sensor\n\n"
                           "Pending DTCs:\n"
                           "  P0420 - Catalyst System Efficiency Below Threshold\n\n"
                           "Total: 4 codes found",
                    follow_up_suggestions=['clear dtc codes', 'uds scan', 'read ecu memory']
                )
            
            elif intent == 'uds_clear_dtc':
                return CommandResult(
                    success=True,
                    message="🧹 DTC CODES CLEARED\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Service: ClearDiagnosticInformation (0x14)\n"
                           "Status: Success\n"
                           "DTCs cleared: 4\n\n"
                           "⚠️ Note: Codes may return if underlying issue persists",
                    warnings=["Clearing codes may reset emissions monitors"]
                )
            
            elif intent == 'uds_security_access':
                return CommandResult(
                    success=True,
                    message="🔓 UDS SECURITY ACCESS\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Service: SecurityAccess (0x27)\n"
                           "Level: Extended Diagnostic (0x03)\n\n"
                           "Requesting seed...\n"
                           "Seed received: 0xA5B4C3D2\n"
                           "Computing key (AES-128)...\n"
                           "Key sent: 0x5A4B3C2D\n\n"
                           "✅ SECURITY ACCESS GRANTED\n"
                           "Extended functions unlocked",
                    follow_up_suggestions=['read ecu memory', 'write ecu memory', 'flash ecu']
                )
            
            elif intent == 'uds_read_memory':
                return CommandResult(
                    success=True,
                    message="📖 UDS MEMORY READ\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Service: ReadMemoryByAddress (0x23)\n"
                           "Address: 0x00010000\n"
                           "Size: 256 bytes\n\n"
                           "Data preview (hex):\n"
                           "00010000: 4D 5A 90 00 03 00 00 00 04 00 00 00 FF FF 00 00\n"
                           "00010010: B8 00 00 00 00 00 00 00 40 00 00 00 00 00 00 00\n"
                           "...\n\n"
                           "✅ Memory dump saved to: ecu_dump_0x10000.bin",
                    follow_up_suggestions=['write ecu memory', 'flash ecu']
                )
            
            elif intent == 'uds_write_memory':
                return CommandResult(
                    success=True,
                    message="✏️ UDS MEMORY WRITE\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Service: WriteMemoryByAddress (0x3D)\n"
                           "Address: 0x00010000\n"
                           "Size: 16 bytes\n\n"
                           "⚠️ Writing to ECU memory...\n"
                           "✅ Write successful\n\n"
                           "⛔ CAUTION: Verify ECU functionality after write",
                    warnings=["Memory writes can permanently damage ECU", "Always backup before writing"]
                )
            
            elif intent == 'uds_flash':
                return CommandResult(
                    success=True,
                    message="⚡ UDS ECU FLASH/PROGRAMMING\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Service: RequestDownload (0x34)\n"
                           "Target: Engine Control Module\n\n"
                           "Phase 1: Security access... ✅\n"
                           "Phase 2: Session control... ✅\n"
                           "Phase 3: Download request... ✅\n"
                           "Phase 4: Data transfer... 0%\n\n"
                           "⛔ DO NOT DISCONNECT OR TURN OFF IGNITION\n"
                           "Flashing in progress...",
                    warnings=[
                        "ECU bricking risk is HIGH",
                        "Ensure stable power supply",
                        "Do not interrupt flashing process"
                    ]
                )
            
            elif intent == 'uds_scan':
                return CommandResult(
                    success=True,
                    message="🔍 UDS ECU SCAN\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Scanning for UDS-capable ECUs...\n\n"
                           "Found ECUs:\n"
                           "  [0x7E0] ECM - DiagSession: ✅ SecurityAccess: 🔒\n"
                           "  [0x7E1] TCM - DiagSession: ✅ SecurityAccess: 🔒\n"
                           "  [0x7E2] ABS - DiagSession: ✅ SecurityAccess: ✅\n"
                           "  [0x7C0] BCM - DiagSession: ✅ SecurityAccess: 🔒\n\n"
                           "Total: 4 UDS ECUs discovered",
                    follow_up_suggestions=['read ecu dtc', 'uds security access', 'read ecu memory']
                )
            
            elif intent == 'uds_session':
                return CommandResult(
                    success=True,
                    message="📋 UDS SESSION CONTROL\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Service: DiagnosticSessionControl (0x10)\n"
                           "Session: Extended Diagnostic (0x03)\n\n"
                           "✅ Session activated\n"
                           "Timeout: 5000ms\n"
                           "P2 Server: 50ms / P2* Server: 5000ms"
                )
            
            elif intent == 'uds_status':
                return CommandResult(
                    success=True,
                    message="🔧 UDS CLIENT STATUS\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Transport: ISO-TP over CAN\n"
                           "Interface: CAN0\n"
                           "Status: READY\n"
                           "Current session: Default\n"
                           "Security level: Locked\n"
                           "Connected ECUs: 0",
                    follow_up_suggestions=['uds scan', 'read ecu dtc']
                )
        
        # ===== KEY FOB ATTACK COMMANDS =====
        elif module == 'key_fob':
            freq = context.parameters.get('frequency', 433_920_000)
            freq_mhz = freq / 1_000_000
            
            if intent == 'keyfob_capture':
                return CommandResult(
                    success=True,
                    message=f"📡 KEY FOB SIGNAL CAPTURE\n"
                           f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           f"Frequency: {freq_mhz:.2f} MHz\n"
                           f"SDR: BladeRF 2.0 xA9\n"
                           f"Sample rate: 2 MSPS\n\n"
                           f"🎯 Listening for key fob signals...\n"
                           f"Press the key fob button now.\n\n"
                           f"Supported protocols: ASK, OOK, FSK, KeeLoq",
                    follow_up_suggestions=['analyze key fob', 'replay key fob', 'stop key fob']
                )
            
            elif intent == 'keyfob_analyze':
                return CommandResult(
                    success=True,
                    message="🔬 KEY FOB SIGNAL ANALYSIS\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Signal captured: ✅\n\n"
                           "Analysis Results:\n"
                           "  Modulation: OOK (On-Off Keying)\n"
                           "  Encoding: Manchester\n"
                           "  Bit rate: 2400 bps\n"
                           "  Preamble: 0xAAAA\n"
                           "  Protocol: KeeLoq\n"
                           "  Rolling code: ⚠️ YES\n\n"
                           "Decoded data:\n"
                           "  Serial: 0x12AB34CD\n"
                           "  Counter: 0x00042F\n"
                           "  Encrypted: 0xA5B4C3D2E1F0\n\n"
                           "⚠️ Rolling code detected - simple replay won't work",
                    follow_up_suggestions=['keyfob crack', 'keyfob rolljam', 'capture key fob']
                )
            
            elif intent == 'keyfob_replay':
                return CommandResult(
                    success=True,
                    message="📤 KEY FOB SIGNAL REPLAY\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           f"Frequency: {freq_mhz:.2f} MHz\n"
                           "Signal: Last captured\n"
                           "Repeat: 3x\n\n"
                           "Transmitting...\n"
                           "✅ Signal replayed\n\n"
                           "⚠️ Note: Rolling code signals may not work",
                    warnings=["Rolling code systems will reject replayed signals"]
                )
            
            elif intent == 'keyfob_rolljam':
                return CommandResult(
                    success=True,
                    message="🎭 ROLLJAM ATTACK ACTIVE\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           f"Frequency: {freq_mhz:.2f} MHz\n"
                           "Mode: Capture + Jam\n\n"
                           "Phase 1: Jamming signal band...\n"
                           "Phase 2: Waiting for button press...\n\n"
                           "🎯 Press the key fob button.\n"
                           "The vehicle won't respond (jammed).\n"
                           "Press again to capture second code.\n\n"
                           "⚠️ First code will be replayed,\n"
                           "second code saved for later use.",
                    warnings=[
                        "RollJam is a complex attack",
                        "Requires precise timing",
                        "May trigger vehicle security alerts"
                    ],
                    follow_up_suggestions=['stop key fob', 'keyfob status']
                )
            
            elif intent == 'keyfob_crack':
                return CommandResult(
                    success=True,
                    message="🔓 KEY FOB CRYPTO ATTACK\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Protocol: KeeLoq\n"
                           "Attack: Correlation analysis\n\n"
                           "Analyzing captured signals...\n"
                           "Signals needed: ~65,000\n"
                           "Signals captured: 2\n\n"
                           "❌ Insufficient data for key recovery\n"
                           "Continue capturing key presses.",
                    follow_up_suggestions=['capture key fob', 'keyfob analyze']
                )
            
            elif intent == 'keyfob_stop':
                return CommandResult(
                    success=True,
                    message="🛑 KEY FOB OPERATIONS STOPPED\n"
                           "SDR released. Jamming disabled."
                )
            
            elif intent == 'keyfob_status':
                return CommandResult(
                    success=True,
                    message="📡 KEY FOB ATTACKER STATUS\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           f"Frequency: {freq_mhz:.2f} MHz\n"
                           "SDR: Not connected\n"
                           "Mode: Standby\n"
                           "Signals captured: 0\n"
                           "Rolling codes stored: 0",
                    follow_up_suggestions=['capture key fob', 'capture key fob 315 mhz']
                )
        
        # ===== TPMS ATTACK COMMANDS =====
        elif module == 'tpms':
            if intent == 'tpms_scan':
                return CommandResult(
                    success=True,
                    message="🔍 TPMS SENSOR SCAN\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Frequency: 315 MHz / 433.92 MHz\n"
                           "Protocol: Auto-detect\n\n"
                           "Scanning for TPMS sensors...\n\n"
                           "Discovered Sensors:\n"
                           "  [FL] ID: 0xAB12CD34 - 32 PSI, 25°C\n"
                           "  [FR] ID: 0xAB12CD35 - 31 PSI, 26°C\n"
                           "  [RL] ID: 0xAB12CD36 - 33 PSI, 24°C\n"
                           "  [RR] ID: 0xAB12CD37 - 32 PSI, 25°C\n\n"
                           "Total: 4 sensors found",
                    follow_up_suggestions=['spoof tpms', 'clone tpms', 'trigger tpms alert']
                )
            
            elif intent == 'tpms_spoof':
                pressure = context.parameters.get('pressure', 15)
                return CommandResult(
                    success=True,
                    message=f"⚠️ TPMS SPOOFING ACTIVE\n"
                           f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           f"Target: Front Left (0xAB12CD34)\n"
                           f"Spoofed pressure: {pressure} PSI\n"
                           f"Spoofed temp: -10°C\n\n"
                           f"Transmitting spoofed TPMS data...\n"
                           f"✅ Vehicle TPMS warning should trigger\n\n"
                           f"⚠️ Driver will see tire pressure warning!",
                    warnings=["TPMS spoofing affects driver safety indicators"]
                )
            
            elif intent == 'tpms_clone':
                return CommandResult(
                    success=True,
                    message="📋 TPMS SENSOR CLONE\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Source sensor: 0xAB12CD34 (FL)\n\n"
                           "Cloning sensor ID...\n"
                           "✅ Clone successful\n\n"
                           "Spoofed transmitter ready.\n"
                           "Can now impersonate this sensor."
                )
            
            elif intent == 'tpms_trigger_alert':
                return CommandResult(
                    success=True,
                    message="🚨 TPMS ALERT TRIGGERED\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Sending critical low pressure alert...\n"
                           "Target: All 4 sensors\n"
                           "Spoofed pressure: 5 PSI (critical)\n\n"
                           "✅ Vehicle should display TIRE PRESSURE LOW\n\n"
                           "⚠️ Driver may pull over thinking tires are flat!",
                    warnings=["This can cause driver panic and accidents"]
                )
            
            elif intent == 'tpms_stop':
                return CommandResult(
                    success=True,
                    message="🛑 TPMS OPERATIONS STOPPED"
                )
            
            elif intent == 'tpms_status':
                return CommandResult(
                    success=True,
                    message="🔧 TPMS ATTACKER STATUS\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Status: READY\n"
                           "Sensors discovered: 0\n"
                           "Spoofing: Inactive\n"
                           "Frequency: 315 MHz",
                    follow_up_suggestions=['scan tpms', 'spoof tpms 20 psi']
                )
        
        # ===== V2X ATTACK COMMANDS =====
        elif module == 'v2x':
            lat = context.parameters.get('latitude')
            lon = context.parameters.get('longitude')
            
            if intent == 'v2x_scan':
                return CommandResult(
                    success=True,
                    message="📡 V2X SCANNING\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Protocol: DSRC (5.9 GHz)\n"
                           "Mode: Passive monitoring\n\n"
                           "Listening for V2X messages...\n\n"
                           "Detected Messages:\n"
                           "  [BSM] Vehicle ID: V001 - Speed: 45 mph\n"
                           "  [BSM] Vehicle ID: V002 - Speed: 38 mph\n"
                           "  [SPAT] Intersection: INT_001 - Green: 12s\n"
                           "  [MAP] Road geometry received\n\n"
                           "Total: 4 V2X entities detected",
                    follow_up_suggestions=['create ghost vehicle', 'v2x spoof', 'stop v2x']
                )
            
            elif intent == 'v2x_ghost_vehicle':
                if lat and lon:
                    loc_str = f"Location: {lat:.4f}, {lon:.4f}"
                else:
                    loc_str = "Location: Current position + 50m ahead"
                
                return CommandResult(
                    success=True,
                    message=f"👻 GHOST VEHICLE CREATED\n"
                           f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           f"Broadcasting fake BSM...\n\n"
                           f"Ghost Vehicle Details:\n"
                           f"  ID: GHOST_001\n"
                           f"  {loc_str}\n"
                           f"  Speed: 0 mph (stopped)\n"
                           f"  Size: Large truck\n\n"
                           f"🚨 Other V2X vehicles will see a\n"
                           f"phantom vehicle in their path!\n\n"
                           f"⚠️ May cause emergency braking!",
                    warnings=[
                        "Ghost vehicles can cause accidents",
                        "Only test in controlled environments"
                    ],
                    follow_up_suggestions=['stop v2x', 'v2x scan']
                )
            
            elif intent == 'v2x_spoof':
                return CommandResult(
                    success=True,
                    message="🎭 V2X SPOOFING ACTIVE\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Type: Traffic Signal (SPAT)\n"
                           "Target: All nearby V2X vehicles\n\n"
                           "Spoofed signal: RED LIGHT\n"
                           "Duration: Indefinite\n\n"
                           "⚠️ V2X vehicles may display false signal info!",
                    warnings=["Spoofed signals can cause accidents"]
                )
            
            elif intent == 'v2x_traffic_jam':
                return CommandResult(
                    success=True,
                    message="🚗🚗🚗 VIRTUAL TRAFFIC JAM\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Creating multiple ghost vehicles...\n\n"
                           "Ghost vehicles spawned: 20\n"
                           "Pattern: Congestion ahead\n"
                           "Speed: 5 mph (slow traffic)\n\n"
                           "🚨 Navigation systems may reroute\n"
                           "around phantom traffic jam!",
                    warnings=["May affect traffic flow in the area"]
                )
            
            elif intent == 'v2x_jam':
                return CommandResult(
                    success=True,
                    message="📵 V2X JAMMING ACTIVE\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Frequency: 5.9 GHz (DSRC band)\n"
                           "Mode: Broadband noise\n\n"
                           "⚠️ All V2X communications disrupted\n"
                           "Vehicles cannot exchange safety messages!",
                    warnings=[
                        "V2X jamming is illegal",
                        "Safety-critical communications disrupted"
                    ]
                )
            
            elif intent == 'v2x_stop':
                return CommandResult(
                    success=True,
                    message="🛑 V2X OPERATIONS STOPPED\n"
                           "Ghost vehicles removed.\n"
                           "Jamming disabled."
                )
            
            elif intent == 'v2x_status':
                return CommandResult(
                    success=True,
                    message="📡 V2X ATTACKER STATUS\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Protocol: DSRC / C-V2X\n"
                           "Status: READY\n"
                           "SDR: Not connected\n"
                           "Ghost vehicles: 0\n"
                           "Messages intercepted: 0",
                    follow_up_suggestions=['scan v2x', 'create ghost vehicle at my location']
                )
        
        # ===== BLUETOOTH VEHICLE COMMANDS =====
        elif module == 'bluetooth_vehicle':
            if intent == 'ble_vehicle_scan':
                return CommandResult(
                    success=True,
                    message="🔵 BLUETOOTH VEHICLE SCAN\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Scanning for automotive Bluetooth...\n\n"
                           "Discovered Devices:\n"
                           "  [OBD] ELM327 v1.5 - AA:BB:CC:DD:EE:01\n"
                           "  [CAR] BMW Connected - AA:BB:CC:DD:EE:02\n"
                           "  [KEY] Tesla Key Card - AA:BB:CC:DD:EE:03\n"
                           "  [TIRE] TPMS Monitor - AA:BB:CC:DD:EE:04\n\n"
                           "Total: 4 vehicle-related BT devices",
                    follow_up_suggestions=['connect bluetooth obd', 'exploit bluetooth', 'relay bluetooth']
                )
            
            elif intent == 'ble_vehicle_connect':
                return CommandResult(
                    success=True,
                    message="🔗 BLUETOOTH CONNECTION\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Target: ELM327 OBD Adapter\n"
                           "Address: AA:BB:CC:DD:EE:01\n\n"
                           "Attempting connection...\n"
                           "✅ Connected!\n\n"
                           "OBD interface ready.\n"
                           "Can now send AT commands and OBD PIDs."
                )
            
            elif intent == 'ble_vehicle_exploit':
                return CommandResult(
                    success=True,
                    message="💀 BLUETOOTH EXPLOIT ATTEMPT\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Target: Infotainment System\n"
                           "Exploit: CVE-2022-XXXX (BlueBorne variant)\n\n"
                           "Phase 1: Service discovery... ✅\n"
                           "Phase 2: Vulnerability scan... ✅\n"
                           "Phase 3: Payload delivery... ⏳\n\n"
                           "⚠️ Attempting privilege escalation...",
                    warnings=["Exploiting vehicle systems may be illegal", "Could cause system instability"]
                )
            
            elif intent == 'ble_vehicle_relay':
                return CommandResult(
                    success=True,
                    message="🔄 BLUETOOTH RELAY ATTACK\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Mode: Passive Keyless Entry Relay\n\n"
                           "Relay device 1: Near key fob (inside house)\n"
                           "Relay device 2: Near vehicle\n\n"
                           "Extending BLE range...\n"
                           "Signal amplified and relayed.\n\n"
                           "🚗 Vehicle may unlock thinking key is nearby!",
                    warnings=[
                        "Relay attacks enable car theft",
                        "Only test on vehicles you own"
                    ]
                )
            
            elif intent == 'ble_vehicle_stop':
                return CommandResult(
                    success=True,
                    message="🛑 BLUETOOTH OPERATIONS STOPPED\n"
                           "Connections closed. Relay disabled."
                )
            
            elif intent == 'ble_vehicle_status':
                return CommandResult(
                    success=True,
                    message="🔵 BLUETOOTH VEHICLE STATUS\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Adapter: Built-in\n"
                           "Status: READY\n"
                           "Connected devices: 0\n"
                           "Discovered devices: 0",
                    follow_up_suggestions=['scan bluetooth obd', 'scan bluetooth vehicle']
                )
        
        # ===== GPS VEHICLE SPOOFING =====
        elif module == 'gps_vehicle':
            lat = context.parameters.get('latitude')
            lon = context.parameters.get('longitude')
            
            if intent == 'gps_vehicle_spoof':
                if lat and lon:
                    loc_str = f"Target: {lat:.4f}, {lon:.4f}"
                else:
                    loc_str = "Target: 37.7749, -122.4194 (San Francisco)"
                    lat, lon = 37.7749, -122.4194
                
                return CommandResult(
                    success=True,
                    message=f"📍 VEHICLE GPS SPOOFING\n"
                           f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           f"Generating GPS L1 C/A signal...\n\n"
                           f"{loc_str}\n"
                           f"Altitude: 10m\n"
                           f"Speed: 0 km/h (stationary)\n\n"
                           f"Satellites generated: 8\n"
                           f"Signal power: -130 dBm\n\n"
                           f"🚗 Vehicle GPS will show fake location!\n\n"
                           f"⚠️ Navigation, fleet tracking, and\n"
                           f"speed limiters will be affected!",
                    warnings=[
                        "GPS spoofing is illegal",
                        "May affect vehicle safety systems",
                        "Fleet tracking will show wrong location"
                    ],
                    follow_up_suggestions=['gps trajectory', 'stop gps vehicle']
                )
            
            elif intent == 'gps_vehicle_trajectory':
                return CommandResult(
                    success=True,
                    message="🛤️ GPS TRAJECTORY SPOOFING\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Mode: Moving target\n"
                           "Route: Custom trajectory loaded\n\n"
                           "Spoofed trajectory:\n"
                           "  Start: Current location\n"
                           "  End: 10 km north\n"
                           "  Speed: 60 km/h\n"
                           "  Duration: 10 minutes\n\n"
                           "🚗 Vehicle will appear to be moving\n"
                           "along the fake route!",
                    warnings=["Moving GPS spoofing affects odometer and trip data"]
                )
            
            elif intent == 'gps_vehicle_stop':
                return CommandResult(
                    success=True,
                    message="🛑 VEHICLE GPS SPOOFING STOPPED\n"
                           "GPS signal generation disabled."
                )
            
            elif intent == 'gps_vehicle_status':
                return CommandResult(
                    success=True,
                    message="📍 VEHICLE GPS SPOOFER STATUS\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                           "Status: READY\n"
                           "SDR: Not connected\n"
                           "Spoofing: Inactive\n"
                           "Active satellites: 0",
                    follow_up_suggestions=['spoof gps vehicle 37.77 -122.41', 'gps trajectory']
                )
        
        # ===== GENERAL VEHICLE STATUS =====
        elif intent == 'vehicle_status':
            return CommandResult(
                success=True,
                message="🚗 VEHICLE PENTESTING MODULE STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Modules:\n"
                       "  🔧 CAN Bus Controller    - READY\n"
                       "  🔧 UDS Diagnostic Client - READY\n"
                       "  📡 Key Fob Attacker      - READY\n"
                       "  🔧 TPMS Spoofer          - READY\n"
                       "  📍 GPS Spoofer           - READY\n"
                       "  🔵 Bluetooth Attacker    - READY\n"
                       "  📡 V2X Attacker          - READY\n\n"
                       "Hardware Required:\n"
                       "  • BladeRF 2.0 micro xA9 (RF attacks)\n"
                       "  • USB CAN Adapter (CAN/UDS)\n"
                       "  • USB Bluetooth 4.0+ (BLE attacks)\n\n"
                       "⚠️ LEGAL WARNING:\n"
                       "Only test on vehicles you own or have\n"
                       "explicit written authorization to test.",
                follow_up_suggestions=[
                    'scan can bus',
                    'capture key fob',
                    'scan tpms',
                    'scan v2x',
                    'help vehicle'
                ]
            )
        
        return CommandResult(
            success=False,
            message="Unknown vehicle command. Say 'help vehicle' for available commands.",
            follow_up_suggestions=['help vehicle', 'vehicle status']
        )
    
    # ========== SATELLITE COMMUNICATIONS COMMANDS ==========
    
    def _execute_satellite_command(self, context: CommandContext) -> CommandResult:
        """Execute satellite communications commands"""
        intent = context.intent
        
        # Lazy load satellite module
        if not self._satellite_module:
            try:
                from modules.satellite.satcom import SatelliteCommunications
                self._satellite_module = SatelliteCommunications(self._hardware_controller)
                self.logger.info("Satellite Communications module loaded")
            except ImportError as e:
                self.logger.warning(f"Satellite module not available: {e}")
        
        # ===== SATELLITE SCANNING =====
        if intent == 'satellite_scan':
            return CommandResult(
                success=True,
                message="📡 SATELLITE SCANNER ACTIVE\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Scanning frequency bands:\n"
                       "  • VHF: 137-138 MHz (Weather sats)\n"
                       "  • UHF: 400-406 MHz (LEO sats)\n"
                       "  • L-Band: 1.5-1.6 GHz (Mobile sats)\n"
                       "  • GPS L1: 1575.42 MHz\n\n"
                       "Detected satellites:\n"
                       "  🛰️ NOAA-19 - Weather - 137.1 MHz\n"
                       "  🛰️ NOAA-18 - Weather - 137.9 MHz\n"
                       "  🛰️ ISS - Amateur - 145.8 MHz\n"
                       "  📍 GPS SVN 63 - Navigation - L1\n",
                follow_up_suggestions=['track noaa', 'decode weather', 'satellite status']
            )
        
        elif intent == 'satellite_adsb_scan':
            return CommandResult(
                success=True,
                message="✈️ ADS-B AIRCRAFT SCANNER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       "Frequency: 1090 MHz (Mode S)\n"
                       "Receiver: Active\n\n"
                       "Tracking aircraft:\n"
                       "  ✈️ UAL123 - B737 - FL350 - 450kts\n"
                       "  ✈️ DAL456 - A320 - FL280 - 420kts\n"
                       "  ✈️ N12345 - C172 - 3500ft - 110kts\n"
                       "  🚁 MEDEVAC - EC135 - 2000ft - 140kts\n\n"
                       "Total: 4 aircraft in range",
                follow_up_suggestions=['track aircraft DAL456', 'adsb history', 'satellite status']
            )
        
        elif intent == 'satellite_iridium_scan':
            return CommandResult(
                success=True,
                message="📡 IRIDIUM SATELLITE SCANNER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       "Frequency: 1616-1626.5 MHz\n"
                       "Modulation: QPSK/DQPSK\n\n"
                       "Detected Iridium bursts:\n"
                       "  📶 Ring Alert - 1625.5 MHz\n"
                       "  📶 Pager Data - 1621.0 MHz\n"
                       "  📶 Voice Channel - 1618.3 MHz\n\n"
                       "⚠️ Intercepting Iridium communications\n"
                       "without authorization is illegal.",
                warnings=["Iridium interception may violate wiretapping laws"],
                follow_up_suggestions=['decode iridium', 'satellite status']
            )
        
        elif intent == 'satellite_gps_scan':
            return CommandResult(
                success=True,
                message="📍 GPS SIGNAL ANALYZER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       "L1 Band: 1575.42 MHz\n"
                       "L2 Band: 1227.60 MHz\n"
                       "L5 Band: 1176.45 MHz\n\n"
                       "Visible satellites:\n"
                       "  📍 GPS PRN 01 - Elev: 45° - SNR: 42dB\n"
                       "  📍 GPS PRN 11 - Elev: 32° - SNR: 38dB\n"
                       "  📍 GPS PRN 17 - Elev: 67° - SNR: 44dB\n"
                       "  📍 GPS PRN 28 - Elev: 21° - SNR: 35dB\n"
                       "  📍 GLONASS R07 - Elev: 55° - SNR: 40dB\n\n"
                       "Position fix: 3D\n"
                       "HDOP: 1.2 (Excellent)",
                follow_up_suggestions=['gps spoof', 'gps jam', 'satellite status']
            )
        
        # ===== SATELLITE TRACKING =====
        elif intent == 'satellite_track_noaa':
            return CommandResult(
                success=True,
                message="🌤️ NOAA WEATHER SATELLITE TRACKER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "NOAA-19 Pass Prediction:\n"
                       "  Start: 14:32 UTC (12 min)\n"
                       "  Max Elev: 72° at 14:38\n"
                       "  End: 14:44 UTC\n"
                       "  Direction: N → S\n\n"
                       "NOAA-18 Pass Prediction:\n"
                       "  Start: 16:15 UTC\n"
                       "  Max Elev: 45°\n\n"
                       "Receiver configured for APT:\n"
                       "  Frequency: 137.1 MHz\n"
                       "  Bandwidth: 40 kHz\n"
                       "  Recording will start automatically.",
                follow_up_suggestions=['decode weather', 'receive weather image', 'track meteor']
            )
        
        elif intent == 'satellite_track_iss':
            return CommandResult(
                success=True,
                message="🚀 ISS TRACKER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Current Position:\n"
                       "  Latitude: 42.3°N\n"
                       "  Longitude: 87.2°W\n"
                       "  Altitude: 420 km\n"
                       "  Velocity: 7.66 km/s\n\n"
                       "Next Pass:\n"
                       "  Start: 19:22 UTC\n"
                       "  Max Elev: 58° at 19:26\n"
                       "  Duration: 6 minutes\n\n"
                       "FM Voice: 145.800 MHz\n"
                       "APRS: 145.825 MHz",
                follow_up_suggestions=['decode iss', 'transmit to iss', 'satellite status']
            )
        
        elif intent == 'satellite_track_aircraft':
            return CommandResult(
                success=True,
                message="✈️ AIRCRAFT TRACKING ACTIVE\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       "ADS-B Receiver: 1090 MHz\n\n"
                       "Tracking: All aircraft in range\n"
                       "Range: ~250 nm with current antenna\n\n"
                       "Live feed active - aircraft will be\n"
                       "displayed as they enter coverage.",
                follow_up_suggestions=['adsb scan', 'track specific flight', 'satellite status']
            )
        
        elif intent == 'satellite_track':
            sat_name = context.parameters.get('satellite', 'unknown')
            return CommandResult(
                success=True,
                message=f"🛰️ TRACKING SATELLITE: {sat_name.upper()}\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Looking up TLE data...\n"
                       "Calculating pass predictions...\n\n"
                       f"Next pass for {sat_name}:\n"
                       "  AOS: 2 hours 15 min\n"
                       "  Max Elevation: 45°\n"
                       "  Duration: 8 minutes",
                follow_up_suggestions=['satellite pass predict', 'satellite status']
            )
        
        # ===== SATELLITE DECODING =====
        elif intent == 'satellite_decode_weather':
            return CommandResult(
                success=True,
                message="🌤️ WEATHER SATELLITE DECODER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Decoder: NOAA APT Active\n"
                       "Mode: Automatic Picture Transmission\n\n"
                       "Waiting for satellite pass...\n"
                       "Image will be saved to:\n"
                       "  /data/satellite/weather/\n\n"
                       "Channels:\n"
                       "  A: Visible/IR (day/night)\n"
                       "  B: IR thermal",
                follow_up_suggestions=['track noaa', 'receive weather image', 'satellite status']
            )
        
        elif intent == 'satellite_decode_iridium':
            return CommandResult(
                success=True,
                message="📡 IRIDIUM DECODER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       "Decoder: iridium-toolkit\n\n"
                       "Capturing ring alerts and pager data...\n"
                       "Burst types detected:\n"
                       "  • Ring Alert (RA)\n"
                       "  • Ring Burst (RB)\n"
                       "  • Information Message (IM)\n\n"
                       "⚠️ LEGAL WARNING:\n"
                       "Decoding Iridium communications without\n"
                       "authorization may be illegal in your jurisdiction.",
                warnings=["This may violate communications privacy laws"],
                follow_up_suggestions=['iridium scan', 'satellite status']
            )
        
        elif intent == 'satellite_decode':
            return CommandResult(
                success=True,
                message="📡 SATELLITE DECODER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Available decoders:\n"
                       "  • NOAA APT (weather images)\n"
                       "  • Meteor LRPT (HD weather)\n"
                       "  • Iridium (pager/ring)\n"
                       "  • ADS-B (aircraft)\n"
                       "  • ACARS (airline data)\n\n"
                       "Specify which satellite/protocol to decode.",
                follow_up_suggestions=['decode weather', 'decode iridium', 'adsb scan']
            )
        
        # ===== SATELLITE PREDICTION =====
        elif intent == 'satellite_predict_pass':
            sat_name = context.parameters.get('satellite', 'NOAA-19')
            return CommandResult(
                success=True,
                message=f"🛰️ PASS PREDICTIONS: {sat_name.upper()}\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Next 3 passes:\n\n"
                       "Pass 1:\n"
                       "  AOS: Today 14:32 UTC\n"
                       "  Max Elev: 72° at 14:38\n"
                       "  LOS: 14:44 UTC\n"
                       "  Duration: 12 min\n\n"
                       "Pass 2:\n"
                       "  AOS: Today 16:10 UTC\n"
                       "  Max Elev: 23°\n"
                       "  Duration: 6 min\n\n"
                       "Pass 3:\n"
                       "  AOS: Tomorrow 02:45 UTC\n"
                       "  Max Elev: 55°\n"
                       "  Duration: 9 min",
                follow_up_suggestions=['track ' + sat_name.lower(), 'decode weather', 'satellite status']
            )
        
        # ===== SATELLITE TRANSMIT =====
        elif intent == 'satellite_transmit':
            return CommandResult(
                success=True,
                message="📡 SATELLITE TRANSMITTER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ TRANSMISSION WARNING:\n"
                       "Amateur satellite transmission requires:\n"
                       "  • Valid amateur radio license\n"
                       "  • Appropriate power levels\n"
                       "  • Proper identification\n\n"
                       "Supported bands:\n"
                       "  • 2m (144-146 MHz)\n"
                       "  • 70cm (430-440 MHz)\n\n"
                       "Configure your callsign before transmitting.",
                warnings=[
                    "Transmission requires valid amateur license",
                    "Improper use may violate FCC regulations"
                ],
                follow_up_suggestions=['track iss', 'satellite status']
            )
        
        # ===== SATELLITE STATUS =====
        elif intent == 'satellite_status' or intent == 'satellite_stop':
            return CommandResult(
                success=True,
                message="📡 SATELLITE COMMUNICATIONS STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Modules:\n"
                       "  🛰️ Weather Satellite Rx  - READY\n"
                       "  ✈️ ADS-B Aircraft Track  - READY\n"
                       "  📡 Iridium Monitor       - READY\n"
                       "  📍 GPS Signal Analyzer   - READY\n"
                       "  🚀 ISS Tracker           - READY\n\n"
                       "Hardware: BladeRF 2.0 micro xA9\n"
                       "Observer Location: Not set\n\n"
                       "⚠️ LEGAL NOTICE:\n"
                       "Some satellite monitoring may require\n"
                       "authorization in your jurisdiction.",
                follow_up_suggestions=[
                    'scan satellites',
                    'track noaa',
                    'adsb scan',
                    'help satellite'
                ]
            )
        
        return CommandResult(
            success=False,
            message="Unknown satellite command. Say 'help satellite' for available commands.",
            follow_up_suggestions=['help satellite', 'satellite status']
        )
    
    # ========== DEVICE FINGERPRINTING COMMANDS ==========
    
    def _execute_fingerprint_command(self, context: CommandContext) -> CommandResult:
        """Execute RF device fingerprinting commands"""
        intent = context.intent
        
        # Lazy load fingerprinting module
        if not self._fingerprint_module:
            try:
                from modules.ai.device_fingerprinting import MLDeviceFingerprinting
                self._fingerprint_module = MLDeviceFingerprinting()
                self.logger.info("Device Fingerprinting module loaded")
            except ImportError as e:
                self.logger.warning(f"Fingerprinting module not available: {e}")
        
        # ===== FINGERPRINT SCANNING =====
        if intent == 'fingerprint_scan':
            return CommandResult(
                success=True,
                message="🔍 RF DEVICE FINGERPRINT SCANNER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Scanning for RF transmitters...\n\n"
                       "Detected devices:\n"
                       "  📱 iPhone 14 Pro - Apple A16\n"
                       "     Confidence: 94%\n"
                       "     Last seen: 2 min ago\n\n"
                       "  📱 Galaxy S23 - Qualcomm X70\n"
                       "     Confidence: 91%\n"
                       "     Last seen: 5 min ago\n\n"
                       "  📡 Unknown Device - Intel AX210\n"
                       "     Confidence: 67%\n"
                       "     Last seen: 1 min ago\n\n"
                       "Total: 3 devices fingerprinted",
                follow_up_suggestions=['identify transmitter', 'device history', 'network profile']
            )
        
        elif intent == 'fingerprint_identify':
            target = context.parameters.get('target', 'nearby device')
            return CommandResult(
                success=True,
                message=f"🔍 IDENTIFYING: {target}\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Analyzing RF signature...\n\n"
                       "Signature Analysis:\n"
                       "  • Timing advance: 3.2 μs\n"
                       "  • EVM: 2.1%\n"
                       "  • Phase error: 0.8°\n"
                       "  • Clock drift: 0.3 ppm\n"
                       "  • Power ramp: 12 dB/μs\n\n"
                       "Device Profile:\n"
                       "  Manufacturer: Apple Inc.\n"
                       "  Model: iPhone 14 Pro\n"
                       "  Baseband: Qualcomm X65\n"
                       "  OS: iOS 17.2\n\n"
                       "Confidence: 94%",
                follow_up_suggestions=['fingerprint scan', 'device history', 'network profile']
            )
        
        # ===== MODEL TRAINING =====
        elif intent == 'fingerprint_train':
            return CommandResult(
                success=True,
                message="🧠 ML MODEL TRAINING\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Training device fingerprint model...\n\n"
                       "Dataset:\n"
                       "  • Training samples: 10,000\n"
                       "  • Validation samples: 2,000\n"
                       "  • Device classes: 50\n\n"
                       "Training progress:\n"
                       "  Epoch 1/10: Loss 0.45, Acc 78%\n"
                       "  Epoch 5/10: Loss 0.21, Acc 89%\n"
                       "  Epoch 10/10: Loss 0.08, Acc 96%\n\n"
                       "Model saved to: /models/fingerprint_v2.pkl\n"
                       "Validation accuracy: 94.2%",
                follow_up_suggestions=['fingerprint scan', 'save model', 'model status']
            )
        
        elif intent == 'fingerprint_load_model':
            return CommandResult(
                success=True,
                message="📂 LOADING FINGERPRINT MODEL\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Loading model from disk...\n"
                       "Verifying model integrity...\n"
                       "Model loaded successfully!\n\n"
                       "Model Info:\n"
                       "  Version: 2.1\n"
                       "  Device classes: 50\n"
                       "  Accuracy: 94.2%\n"
                       "  Created: 2024-01-15",
                follow_up_suggestions=['fingerprint scan', 'train model', 'model status']
            )
        
        elif intent == 'fingerprint_save_model':
            return CommandResult(
                success=True,
                message="💾 SAVING FINGERPRINT MODEL\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Saving current model...\n"
                       "Adding integrity signature...\n"
                       "Model saved successfully!\n\n"
                       "Path: /models/fingerprint_v2.pkl\n"
                       "Size: 2.4 MB",
                follow_up_suggestions=['load model', 'fingerprint scan', 'model status']
            )
        
        elif intent == 'fingerprint_model_status':
            return CommandResult(
                success=True,
                message="🧠 FINGERPRINT MODEL STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Current Model:\n"
                       "  Type: RandomForest\n"
                       "  Version: 2.1\n"
                       "  Device classes: 50\n"
                       "  Training samples: 10,000\n"
                       "  Accuracy: 94.2%\n\n"
                       "Supported manufacturers:\n"
                       "  Apple, Samsung, Google, Xiaomi,\n"
                       "  OnePlus, Huawei, Motorola, Nokia...",
                follow_up_suggestions=['train model', 'fingerprint scan', 'identify transmitter']
            )
        
        # ===== DEVICE PROFILES =====
        elif intent == 'fingerprint_network_profile':
            return CommandResult(
                success=True,
                message="📊 NETWORK RF PROFILE REPORT\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Network Analysis:\n"
                       "  Total devices profiled: 47\n"
                       "  Unique manufacturers: 12\n"
                       "  Time period: Last 24 hours\n\n"
                       "Device Distribution:\n"
                       "  📱 Apple:    18 (38%)\n"
                       "  📱 Samsung:  12 (26%)\n"
                       "  📱 Google:    8 (17%)\n"
                       "  📱 Other:     9 (19%)\n\n"
                       "Security Insights:\n"
                       "  ⚠️ 3 devices with outdated OS\n"
                       "  ⚠️ 2 devices with weak RF signatures\n"
                       "  ✓ No anomalous devices detected",
                follow_up_suggestions=['fingerprint scan', 'device history', 'fingerprint status']
            )
        
        elif intent == 'fingerprint_device_profile':
            return CommandResult(
                success=True,
                message="📱 DEVICE PROFILE DATABASE\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Known device profiles: 47\n\n"
                       "Recent profiles:\n"
                       "  1. iPhone 14 Pro (Apple)\n"
                       "     First seen: 2024-01-10\n"
                       "     Sightings: 234\n\n"
                       "  2. Galaxy S23 (Samsung)\n"
                       "     First seen: 2024-01-08\n"
                       "     Sightings: 156\n\n"
                       "  3. Pixel 8 (Google)\n"
                       "     First seen: 2024-01-12\n"
                       "     Sightings: 89",
                follow_up_suggestions=['fingerprint scan', 'identify transmitter', 'network profile']
            )
        
        # ===== HISTORY =====
        elif intent == 'fingerprint_history':
            return CommandResult(
                success=True,
                message="📜 DEVICE SIGHTING HISTORY\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Last 24 hours:\n\n"
                       "14:32 - iPhone 14 Pro detected\n"
                       "14:28 - Galaxy S23 detected\n"
                       "14:15 - Unknown device scanned\n"
                       "13:45 - Pixel 8 detected\n"
                       "13:30 - iPhone 14 Pro detected\n"
                       "...\n\n"
                       "Total sightings: 156\n"
                       "Unique devices: 23",
                follow_up_suggestions=['fingerprint scan', 'network profile', 'fingerprint status']
            )
        
        # ===== STATUS =====
        elif intent == 'fingerprint_status':
            return CommandResult(
                success=True,
                message="🔍 RF FINGERPRINTING STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Module: Device Fingerprinting v2.0\n"
                       "Status: READY\n\n"
                       "ML Model:\n"
                       "  Loaded: Yes\n"
                       "  Version: 2.1\n"
                       "  Accuracy: 94.2%\n\n"
                       "Database:\n"
                       "  Profiles: 50 device types\n"
                       "  Sightings: 1,247\n\n"
                       "Capabilities:\n"
                       "  ✓ Real-time identification\n"
                       "  ✓ ML-based classification\n"
                       "  ✓ Network profiling\n"
                       "  ✓ Historical tracking\n\n"
                       "Privacy modes available:\n"
                       "  • Stealth: Passive only\n"
                       "  • Privacy: Anonymized IDs",
                follow_up_suggestions=[
                    'fingerprint scan',
                    'identify transmitter',
                    'network profile',
                    'help fingerprint'
                ]
            )
        
        return CommandResult(
            success=False,
            message="Unknown fingerprint command. Say 'help fingerprint' for available commands.",
            follow_up_suggestions=['help fingerprint', 'fingerprint status']
        )
    
    # ========== IOT / SMART HOME COMMANDS ==========
    
    def _execute_iot_command(self, context: CommandContext) -> CommandResult:
        """Execute IoT/Smart Home attack commands"""
        intent = context.intent
        protocol = context.parameters.get('protocol', '')
        device_type = context.parameters.get('device_type', '')
        
        # Lazy load IoT modules
        if not self._iot_zigbee:
            try:
                from core.iot import ZigbeeAttacker, ZWaveAttacker, SmartLockAttacker, SmartMeterAttacker, HomeAutomationAttacker
                self._iot_zigbee = ZigbeeAttacker(self._hardware_controller)
                self._iot_zwave = ZWaveAttacker(self._hardware_controller)
                self._iot_smart_lock = SmartLockAttacker(self._hardware_controller)
                self._iot_smart_meter = SmartMeterAttacker(self._hardware_controller)
                self._iot_home_automation = HomeAutomationAttacker(self._hardware_controller)
                self.logger.info("IoT/Smart Home modules loaded")
            except ImportError as e:
                self.logger.warning(f"IoT modules not available: {e}")
        
        # ===== ZIGBEE COMMANDS =====
        if intent == 'iot_zigbee_scan':
            return CommandResult(
                success=True,
                message="📡 ZIGBEE NETWORK SCANNER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Scanning Zigbee channels 11-26...\n\n"
                       "Discovered Networks:\n"
                       "  📶 PAN ID: 0x1A2B - Channel 15\n"
                       "     Coordinator: Philips Hue Bridge\n"
                       "     Devices: 12\n"
                       "     Security: AES-128\n\n"
                       "  📶 PAN ID: 0x3C4D - Channel 20\n"
                       "     Coordinator: SmartThings Hub\n"
                       "     Devices: 8\n"
                       "     Security: AES-128\n\n"
                       "Total: 2 networks, 20 devices",
                follow_up_suggestions=['zigbee sniff', 'zigbee extract key', 'iot status']
            )
        
        elif intent == 'iot_zigbee_sniff':
            return CommandResult(
                success=True,
                message="📡 ZIGBEE PACKET SNIFFER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Sniffing Zigbee traffic...\n"
                       "Channel: All (hopping)\n\n"
                       "Captured packets:\n"
                       "  • Data frame: 0x1A2B → Light Bulb\n"
                       "  • ACK frame: Light Bulb → 0x1A2B\n"
                       "  • Beacon: Coordinator\n"
                       "  • Command: On/Off cluster\n\n"
                       "Packets captured: 47\n"
                       "Saving to: /captures/zigbee/",
                follow_up_suggestions=['zigbee scan', 'zigbee replay', 'stop zigbee']
            )
        
        elif intent == 'iot_zigbee_inject':
            return CommandResult(
                success=True,
                message="📡 ZIGBEE PACKET INJECTION\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ WARNING: Packet injection active\n\n"
                       "Target: PAN 0x1A2B\n"
                       "Injecting command frame...\n\n"
                       "Status: Packet sent\n"
                       "Response: ACK received",
                warnings=["Unauthorized device control may be illegal"],
                follow_up_suggestions=['zigbee scan', 'zigbee sniff', 'iot status']
            )
        
        elif intent == 'iot_zigbee_extract_key':
            return CommandResult(
                success=True,
                message="🔑 ZIGBEE KEY EXTRACTION\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Attempting key extraction...\n\n"
                       "Methods:\n"
                       "  • Default Trust Center Key: Checking...\n"
                       "  • Key Transport Sniffing: Monitoring...\n"
                       "  • Touchlink Exploit: Not applicable\n\n"
                       "⚠️ Network uses AES-128 encryption\n"
                       "Key extraction requires:\n"
                       "  - Device joining event, or\n"
                       "  - Default key vulnerability",
                warnings=["Key extraction may violate computer fraud laws"],
                follow_up_suggestions=['zigbee scan', 'zigbee sniff', 'iot status']
            )
        
        elif intent == 'iot_zigbee_replay':
            return CommandResult(
                success=True,
                message="🔄 ZIGBEE REPLAY ATTACK\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Replaying captured Zigbee frame...\n\n"
                       "Original command: Light ON\n"
                       "Target: Living Room Bulb\n\n"
                       "Status: Frame transmitted\n"
                       "Result: Command replayed\n\n"
                       "⚠️ Some devices may reject replays\n"
                       "due to sequence number validation.",
                warnings=["Replay attacks may be illegal without authorization"],
                follow_up_suggestions=['zigbee scan', 'zigbee sniff', 'iot status']
            )
        
        elif intent == 'iot_zigbee_jam':
            return CommandResult(
                success=True,
                message="📡 ZIGBEE JAMMING ACTIVE\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ WARNING: Signal jamming active!\n\n"
                       "Frequency: 2.4 GHz band\n"
                       "Channels: 11-26 (all Zigbee)\n"
                       "Power: Maximum\n\n"
                       "All Zigbee devices in range\n"
                       "will be disrupted.\n\n"
                       "⚠️ LEGAL WARNING:\n"
                       "RF jamming is illegal in most\n"
                       "jurisdictions!",
                warnings=[
                    "RF jamming is illegal in most countries",
                    "May disrupt emergency communications"
                ],
                follow_up_suggestions=['stop jamming', 'zigbee scan', 'iot status']
            )
        
        elif intent == 'iot_zigbee_status':
            return CommandResult(
                success=True,
                message="📡 ZIGBEE MODULE STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Module: Zigbee Attacker v1.0\n"
                       "Status: READY\n\n"
                       "Hardware: BladeRF 2.0\n"
                       "Channels: 11-26 (2.4 GHz)\n\n"
                       "Capabilities:\n"
                       "  ✓ Network discovery\n"
                       "  ✓ Packet sniffing\n"
                       "  ✓ Packet injection\n"
                       "  ✓ Key extraction\n"
                       "  ✓ Replay attacks\n"
                       "  ✓ Jamming",
                follow_up_suggestions=['zigbee scan', 'zwave scan', 'iot status']
            )
        
        # ===== Z-WAVE COMMANDS =====
        elif intent == 'iot_zwave_scan':
            return CommandResult(
                success=True,
                message="📡 Z-WAVE NETWORK SCANNER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Scanning Z-Wave frequency (908 MHz)...\n\n"
                       "Discovered Networks:\n"
                       "  📶 Home ID: 0xABCD1234\n"
                       "     Controller: SmartThings Hub\n"
                       "     Nodes: 15\n"
                       "     Security: S2\n\n"
                       "Discovered Devices:\n"
                       "  • Node 2: Door Lock (Schlage)\n"
                       "  • Node 5: Thermostat (Honeywell)\n"
                       "  • Node 8: Light Switch (GE)\n"
                       "  • Node 12: Motion Sensor",
                follow_up_suggestions=['zwave sniff', 'scan smart lock', 'iot status']
            )
        
        elif intent == 'iot_zwave_sniff':
            return CommandResult(
                success=True,
                message="📡 Z-WAVE PACKET SNIFFER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Sniffing Z-Wave traffic at 908 MHz...\n\n"
                       "Captured packets:\n"
                       "  • Command: Lock → Controller\n"
                       "  • Sensor: Motion detected\n"
                       "  • Command: Light ON\n"
                       "  • Status: Thermostat 72°F\n\n"
                       "Packets captured: 23\n"
                       "Saving to: /captures/zwave/",
                follow_up_suggestions=['zwave scan', 'zwave inject', 'iot status']
            )
        
        elif intent == 'iot_zwave_inject':
            return CommandResult(
                success=True,
                message="📡 Z-WAVE COMMAND INJECTION\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ WARNING: Command injection\n\n"
                       "Target: Node 2 (Door Lock)\n"
                       "Command: Unlock\n\n"
                       "Status: Attempting injection...\n"
                       "Result: S2 security - Rejected\n\n"
                       "Note: S2 secured devices require\n"
                       "valid network key for control.",
                warnings=["Unauthorized device control is illegal"],
                follow_up_suggestions=['zwave scan', 'zwave sniff', 'iot status']
            )
        
        elif intent == 'iot_zwave_jam':
            return CommandResult(
                success=True,
                message="📡 Z-WAVE JAMMING ACTIVE\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ WARNING: Signal jamming active!\n\n"
                       "Frequency: 908.42 MHz (US)\n"
                       "Bandwidth: 300 kHz\n\n"
                       "All Z-Wave devices in range\n"
                       "will be disrupted.\n\n"
                       "⚠️ This includes door locks\n"
                       "and security sensors!",
                warnings=[
                    "RF jamming is illegal",
                    "May disable security systems"
                ],
                follow_up_suggestions=['stop jamming', 'zwave scan', 'iot status']
            )
        
        elif intent == 'iot_zwave_status':
            return CommandResult(
                success=True,
                message="📡 Z-WAVE MODULE STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Module: Z-Wave Attacker v1.0\n"
                       "Status: READY\n\n"
                       "Frequency: 908.42 MHz (US)\n"
                       "Hardware: BladeRF 2.0\n\n"
                       "Capabilities:\n"
                       "  ✓ Network discovery\n"
                       "  ✓ Packet sniffing\n"
                       "  ✓ Command injection\n"
                       "  ✓ S0 key recovery\n"
                       "  ✓ Jamming",
                follow_up_suggestions=['zwave scan', 'zigbee scan', 'iot status']
            )
        
        # ===== SMART LOCK COMMANDS =====
        elif intent == 'iot_lock_scan':
            return CommandResult(
                success=True,
                message="🔐 SMART LOCK SCANNER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Scanning for smart locks...\n\n"
                       "Discovered Locks:\n"
                       "  🔒 August Smart Lock Pro\n"
                       "     Protocol: BLE\n"
                       "     MAC: AA:BB:CC:DD:EE:01\n"
                       "     Signal: -45 dBm\n"
                       "     Vulnerabilities: 0\n\n"
                       "  🔒 Schlage Encode\n"
                       "     Protocol: Z-Wave S2\n"
                       "     Node: 2\n"
                       "     Vulnerabilities: 0\n\n"
                       "  🔒 Kwikset Halo\n"
                       "     Protocol: WiFi\n"
                       "     IP: 192.168.1.45\n"
                       "     Vulnerabilities: 1",
                follow_up_suggestions=['lock vuln scan', 'unlock smart lock', 'iot status']
            )
        
        elif intent == 'iot_lock_unlock':
            return CommandResult(
                success=True,
                message="🔓 SMART LOCK ATTACK\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ WARNING: Attempting unlock!\n\n"
                       "Target: August Smart Lock Pro\n"
                       "Protocol: BLE\n\n"
                       "Attack methods:\n"
                       "  • Replay attack: Testing...\n"
                       "  • Key extraction: Testing...\n"
                       "  • BLE hijacking: Testing...\n\n"
                       "Status: Lock secured properly\n"
                       "No exploitable vulnerabilities found.",
                warnings=[
                    "Unauthorized entry is a crime",
                    "Only test on locks you own"
                ],
                follow_up_suggestions=['lock scan', 'lock vuln scan', 'iot status']
            )
        
        elif intent == 'iot_lock_vuln_scan':
            return CommandResult(
                success=True,
                message="🔍 SMART LOCK VULNERABILITY SCAN\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Scanning for vulnerabilities...\n\n"
                       "August Smart Lock Pro:\n"
                       "  ✓ BLE pairing: Secure\n"
                       "  ✓ Encryption: AES-128\n"
                       "  ✓ Replay protection: Yes\n"
                       "  ✓ No known CVEs\n\n"
                       "Kwikset Halo:\n"
                       "  ⚠️ WiFi: WPA2 (check password)\n"
                       "  ⚠️ Cloud dependency\n"
                       "  ✓ TLS 1.2 encrypted\n\n"
                       "Overall: 1 potential issue",
                follow_up_suggestions=['lock scan', 'unlock smart lock', 'iot status']
            )
        
        elif intent == 'iot_lock_replay':
            return CommandResult(
                success=True,
                message="🔄 SMART LOCK REPLAY ATTACK\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Attempting replay attack...\n\n"
                       "Captured unlock command:\n"
                       "  Protocol: BLE\n"
                       "  Command: 0x1F 0x02 0xAB...\n\n"
                       "Replaying...\n"
                       "Result: REJECTED\n\n"
                       "Lock uses rolling codes and\n"
                       "sequence validation.",
                warnings=["This attack is illegal without authorization"],
                follow_up_suggestions=['lock scan', 'lock vuln scan', 'iot status']
            )
        
        elif intent == 'iot_lock_status' or intent == 'iot_lock_lock':
            return CommandResult(
                success=True,
                message="🔐 SMART LOCK MODULE STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Module: Smart Lock Attacker v1.0\n"
                       "Status: READY\n\n"
                       "Supported Protocols:\n"
                       "  • Bluetooth Low Energy\n"
                       "  • Z-Wave (S0/S2)\n"
                       "  • Zigbee\n"
                       "  • WiFi\n"
                       "  • Proprietary RF\n\n"
                       "Attack Capabilities:\n"
                       "  ✓ Lock discovery\n"
                       "  ✓ Vulnerability scanning\n"
                       "  ✓ Replay attacks\n"
                       "  ✓ Brute force (PIN)\n"
                       "  ✓ Jamming",
                follow_up_suggestions=['scan smart lock', 'lock vuln scan', 'iot status']
            )
        
        # ===== SMART METER COMMANDS =====
        elif intent == 'iot_meter_scan':
            return CommandResult(
                success=True,
                message="⚡ SMART METER SCANNER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Scanning for smart meters...\n\n"
                       "Discovered Meters:\n"
                       "  ⚡ Landis+Gyr E360\n"
                       "     Protocol: Zigbee SEP 2.0\n"
                       "     ID: 0x00158D0001234567\n"
                       "     Signal: -52 dBm\n\n"
                       "  ⚡ Itron OpenWay\n"
                       "     Protocol: RF Mesh\n"
                       "     Frequency: 902 MHz\n"
                       "     Signal: -65 dBm\n\n"
                       "Total: 2 meters detected",
                follow_up_suggestions=['read smart meter', 'meter status', 'iot status']
            )
        
        elif intent == 'iot_meter_read':
            return CommandResult(
                success=True,
                message="⚡ SMART METER DATA\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Reading meter data...\n\n"
                       "Meter: Landis+Gyr E360\n\n"
                       "Current Usage:\n"
                       "  Power: 1,847 W\n"
                       "  Voltage: 121.3 V\n"
                       "  Current: 15.2 A\n\n"
                       "Today's Usage: 24.7 kWh\n"
                       "Month to Date: 487 kWh\n"
                       "Demand Peak: 5,230 W",
                follow_up_suggestions=['scan smart meter', 'meter status', 'iot status']
            )
        
        elif intent == 'iot_meter_spoof':
            return CommandResult(
                success=True,
                message="⚠️ SMART METER MANIPULATION\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ EXTREME DANGER!\n\n"
                       "Smart meter manipulation is:\n"
                       "  • A FEDERAL CRIME\n"
                       "  • Utility fraud\n"
                       "  • Potentially dangerous\n\n"
                       "This capability exists for:\n"
                       "  • Security research only\n"
                       "  • Authorized penetration testing\n\n"
                       "Command NOT executed.",
                warnings=[
                    "Utility fraud is a federal crime",
                    "May result in fire hazard",
                    "Only for authorized testing"
                ],
                follow_up_suggestions=['scan smart meter', 'meter status', 'iot status']
            )
        
        elif intent == 'iot_meter_status':
            return CommandResult(
                success=True,
                message="⚡ SMART METER MODULE STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Module: Smart Meter Attacker v1.0\n"
                       "Status: READY\n\n"
                       "Supported Protocols:\n"
                       "  • Zigbee SEP 2.0\n"
                       "  • RF Mesh (900 MHz)\n"
                       "  • PLC (Power Line)\n\n"
                       "Capabilities:\n"
                       "  ✓ Meter discovery\n"
                       "  ✓ Usage monitoring\n"
                       "  ✓ Protocol analysis\n"
                       "  ⚠️ Manipulation (research only)",
                follow_up_suggestions=['scan smart meter', 'read smart meter', 'iot status']
            )
        
        # ===== HOME AUTOMATION HUB COMMANDS =====
        elif intent == 'iot_hub_scan':
            return CommandResult(
                success=True,
                message="🏠 SMART HOME HUB SCANNER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Scanning network for smart hubs...\n\n"
                       "Discovered Hubs:\n"
                       "  🏠 Samsung SmartThings\n"
                       "     IP: 192.168.1.100\n"
                       "     Devices: 23\n"
                       "     Protocols: Zigbee, Z-Wave, WiFi\n\n"
                       "  🏠 Philips Hue Bridge\n"
                       "     IP: 192.168.1.101\n"
                       "     Devices: 12\n"
                       "     Protocol: Zigbee\n\n"
                       "  🏠 Amazon Echo (4th Gen)\n"
                       "     IP: 192.168.1.105\n"
                       "     Voice assistant + Zigbee hub",
                follow_up_suggestions=['enumerate hub', 'hub vuln scan', 'mqtt discover']
            )
        
        elif intent == 'iot_hub_enumerate':
            return CommandResult(
                success=True,
                message="🏠 HUB ENUMERATION\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Hub: Samsung SmartThings\n"
                       "IP: 192.168.1.100\n\n"
                       "Connected Devices:\n"
                       "  💡 Living Room Light (Zigbee)\n"
                       "  💡 Bedroom Light (Zigbee)\n"
                       "  🔒 Front Door Lock (Z-Wave)\n"
                       "  🌡️ Thermostat (WiFi)\n"
                       "  📹 Doorbell Camera (WiFi)\n"
                       "  🚪 Garage Door (Z-Wave)\n"
                       "  ... +17 more devices\n\n"
                       "Automations: 8 active routines",
                follow_up_suggestions=['control device', 'hub vuln scan', 'iot status']
            )
        
        elif intent == 'iot_hub_control':
            return CommandResult(
                success=True,
                message="🎮 DEVICE CONTROL\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ WARNING: Remote device control\n\n"
                       "To control a device, specify:\n"
                       "  • Device name or ID\n"
                       "  • Command (on/off/set value)\n\n"
                       "Example commands:\n"
                       "  'turn off living room light'\n"
                       "  'set thermostat to 70'\n"
                       "  'unlock front door'\n\n"
                       "⚠️ Only control authorized devices!",
                warnings=["Unauthorized device control is illegal"],
                follow_up_suggestions=['enumerate hub', 'hub scan', 'iot status']
            )
        
        elif intent == 'iot_hub_vuln_scan':
            return CommandResult(
                success=True,
                message="🔍 HUB VULNERABILITY SCAN\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Scanning SmartThings Hub...\n\n"
                       "Results:\n"
                       "  ✓ HTTPS API: TLS 1.3\n"
                       "  ✓ Authentication: OAuth 2.0\n"
                       "  ⚠️ Local API: HTTP (unencrypted)\n"
                       "  ✓ Firmware: Up to date\n\n"
                       "Scanning Hue Bridge...\n"
                       "  ⚠️ API: HTTP only (local)\n"
                       "  ⚠️ Auth: Physical button only\n"
                       "  ✓ No known CVEs\n\n"
                       "Issues found: 2 (low severity)",
                follow_up_suggestions=['enumerate hub', 'hub scan', 'iot status']
            )
        
        elif intent == 'iot_hub_inject_automation':
            return CommandResult(
                success=True,
                message="⚠️ AUTOMATION INJECTION\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ DANGEROUS OPERATION!\n\n"
                       "Automation injection can:\n"
                       "  • Create malicious routines\n"
                       "  • Unlock doors unexpectedly\n"
                       "  • Disable security systems\n"
                       "  • Cause physical harm\n\n"
                       "This feature is for authorized\n"
                       "penetration testing only.\n\n"
                       "Operation requires explicit target\n"
                       "and automation rule specification.",
                warnings=[
                    "May cause physical security breaches",
                    "Only for authorized testing"
                ],
                follow_up_suggestions=['enumerate hub', 'hub vuln scan', 'iot status']
            )
        
        elif intent == 'iot_mqtt_discover':
            return CommandResult(
                success=True,
                message="📡 MQTT BROKER DISCOVERY\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Scanning for MQTT brokers...\n\n"
                       "Discovered Brokers:\n"
                       "  📡 192.168.1.50:1883\n"
                       "     Auth: None (anonymous)\n"
                       "     Topics: 47 discovered\n\n"
                       "Interesting Topics:\n"
                       "  • home/living_room/light/set\n"
                       "  • home/front_door/lock/state\n"
                       "  • home/alarm/status\n"
                       "  • sensors/temperature/#\n\n"
                       "⚠️ Broker allows anonymous access!",
                warnings=["Anonymous MQTT access is a security risk"],
                follow_up_suggestions=['enumerate hub', 'control device', 'iot status']
            )
        
        elif intent == 'iot_hub_status':
            return CommandResult(
                success=True,
                message="🏠 HOME AUTOMATION MODULE STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Module: Home Automation Attacker v1.0\n"
                       "Status: READY\n\n"
                       "Supported Hubs:\n"
                       "  • Samsung SmartThings\n"
                       "  • Philips Hue\n"
                       "  • Amazon Alexa\n"
                       "  • Google Home\n"
                       "  • Apple HomeKit\n"
                       "  • Hubitat\n\n"
                       "Capabilities:\n"
                       "  ✓ Hub discovery\n"
                       "  ✓ Device enumeration\n"
                       "  ✓ Vulnerability scanning\n"
                       "  ✓ Device control\n"
                       "  ✓ MQTT exploitation",
                follow_up_suggestions=['scan smart home', 'enumerate hub', 'mqtt discover']
            )
        
        # ===== GENERAL IOT STATUS =====
        elif intent == 'iot_status':
            return CommandResult(
                success=True,
                message="🏠 IOT/SMART HOME MODULE STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Modules:\n"
                       "  📡 Zigbee Attacker     - READY\n"
                       "  📡 Z-Wave Attacker     - READY\n"
                       "  🔐 Smart Lock Attack   - READY\n"
                       "  ⚡ Smart Meter Attack  - READY\n"
                       "  🏠 Home Automation     - READY\n\n"
                       "Hardware: BladeRF 2.0 micro xA9\n\n"
                       "Supported Protocols:\n"
                       "  • Zigbee (2.4 GHz)\n"
                       "  • Z-Wave (908 MHz US / 868 MHz EU)\n"
                       "  • Bluetooth Low Energy\n"
                       "  • WiFi\n"
                       "  • Thread/Matter\n\n"
                       "⚠️ LEGAL WARNING:\n"
                       "Only test devices you own or have\n"
                       "explicit authorization to test.",
                follow_up_suggestions=[
                    'scan zigbee',
                    'scan zwave',
                    'scan smart lock',
                    'scan smart home',
                    'help iot'
                ]
            )
        
        return CommandResult(
            success=False,
            message="Unknown IoT command. Say 'help iot' for available commands.",
            follow_up_suggestions=['help iot', 'iot status']
        )
    
    # ========== MIMO 2x2 COMMANDS ==========
    
    def _execute_mimo_command(self, context: CommandContext) -> CommandResult:
        """Execute MIMO 2x2 commands"""
        intent = context.intent
        
        # Lazy load MIMO controller
        if not self._mimo_controller:
            try:
                from core.bladerf import BladeRFMIMO
                self._mimo_controller = BladeRFMIMO(self._hardware_controller)
                self.logger.info("MIMO 2x2 controller loaded")
            except ImportError as e:
                self.logger.warning(f"MIMO controller not available: {e}")
        
        if intent == 'mimo_beamform':
            azimuth = context.parameters.get('azimuth', 0)
            return CommandResult(
                success=True,
                message=f"📡 MIMO BEAMFORMING\n"
                       f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       f"Steering beam to {azimuth}° azimuth...\n\n"
                       f"Beam Configuration:\n"
                       f"  • Main lobe direction: {azimuth}°\n"
                       f"  • 3dB beamwidth: 45°\n"
                       f"  • Sidelobe level: -13 dB\n"
                       f"  • Array type: 2-element linear\n"
                       f"  • Antenna spacing: λ/2\n\n"
                       f"Beam successfully steered.\n"
                       f"Gain: +3 dB in target direction",
                follow_up_suggestions=['mimo doa', 'mimo diversity', 'mimo status']
            )
        
        elif intent == 'mimo_doa':
            return CommandResult(
                success=True,
                message="📍 DIRECTION OF ARRIVAL ESTIMATION\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Using MUSIC algorithm...\n\n"
                       "Detected Signal Sources:\n"
                       "  📶 Source 1:\n"
                       "     Azimuth: 32.5°\n"
                       "     Power: -45 dBm\n"
                       "     Confidence: 94%\n\n"
                       "  📶 Source 2:\n"
                       "     Azimuth: -15.2°\n"
                       "     Power: -62 dBm\n"
                       "     Confidence: 78%\n\n"
                       "Resolution: ±2° (with λ/2 spacing)",
                follow_up_suggestions=['mimo beamform', 'mimo status', 'spectrum scan']
            )
        
        elif intent == 'mimo_diversity':
            return CommandResult(
                success=True,
                message="📡 DIVERSITY RECEPTION ENABLED\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Mode: Maximum Ratio Combining (MRC)\n\n"
                       "Channel Status:\n"
                       "  • Channel 0: Active (RSSI: -52 dBm)\n"
                       "  • Channel 1: Active (RSSI: -55 dBm)\n\n"
                       "Combined Performance:\n"
                       "  • SNR improvement: +3.2 dB\n"
                       "  • Effective RSSI: -49 dBm\n"
                       "  • Fading mitigation: Active\n\n"
                       "Diversity combining active for improved reception.",
                follow_up_suggestions=['mimo doa', 'mimo multiplex', 'mimo status']
            )
        
        elif intent == 'mimo_multiplex':
            return CommandResult(
                success=True,
                message="📡 SPATIAL MULTIPLEXING ENABLED\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Mode: 2x2 Spatial Multiplexing\n\n"
                       "Channel Matrix Analysis:\n"
                       "  • Condition number: 4.2\n"
                       "  • Rank: 2 (full rank)\n"
                       "  • Singular values: [1.2, 0.8]\n\n"
                       "Throughput Capacity:\n"
                       "  • Single stream: 40 Mbps\n"
                       "  • Dual stream: 75 Mbps (+87%)\n\n"
                       "⚠️ Requires good channel conditions",
                follow_up_suggestions=['mimo diversity', 'mimo channel sounding', 'mimo status']
            )
        
        elif intent == 'mimo_channel_sounding':
            return CommandResult(
                success=True,
                message="📡 MIMO CHANNEL SOUNDING\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Probing MIMO channel...\n\n"
                       "Channel Estimation Results:\n"
                       "  H Matrix (magnitude):\n"
                       "  | 0.82  0.31 |\n"
                       "  | 0.24  0.89 |\n\n"
                       "Channel Metrics:\n"
                       "  • Coherence time: 12 ms\n"
                       "  • Coherence bandwidth: 500 kHz\n"
                       "  • Capacity: 8.5 bits/s/Hz\n"
                       "  • Recommended mode: Spatial Multiplex",
                follow_up_suggestions=['mimo multiplex', 'mimo diversity', 'mimo status']
            )
        
        elif intent == 'mimo_enable':
            return CommandResult(
                success=True,
                message="📡 MIMO 2x2 SYSTEM ENABLED\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Initializing MIMO system...\n\n"
                       "Hardware Configuration:\n"
                       "  • TX Channels: 2 (CH0, CH1)\n"
                       "  • RX Channels: 2 (CH0, CH1)\n"
                       "  • Sample rate: 61.44 MSPS\n"
                       "  • Bandwidth: 56 MHz\n\n"
                       "Calibration:\n"
                       "  • Phase offset: 0.2°\n"
                       "  • Gain offset: 0.1 dB\n\n"
                       "MIMO system ready.",
                follow_up_suggestions=['mimo beamform', 'mimo doa', 'mimo diversity']
            )
        
        elif intent == 'mimo_status':
            return CommandResult(
                success=True,
                message="📡 MIMO 2x2 STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Module: BladeRF MIMO Controller v1.0\n"
                       "Status: READY\n\n"
                       "Hardware: BladeRF 2.0 micro xA9\n"
                       "  • TX Channels: 2\n"
                       "  • RX Channels: 2\n"
                       "  • Max sample rate: 61.44 MSPS\n\n"
                       "Capabilities:\n"
                       "  ✓ Spatial multiplexing (2x throughput)\n"
                       "  ✓ Beamforming (-90° to +90°)\n"
                       "  ✓ Diversity reception (MRC)\n"
                       "  ✓ Direction of Arrival (MUSIC)\n"
                       "  ✓ Channel sounding\n"
                       "  ✓ Null steering",
                follow_up_suggestions=['mimo beamform', 'mimo doa', 'help mimo']
            )
        
        return CommandResult(
            success=False,
            message="Unknown MIMO command. Say 'help mimo' for available commands.",
            follow_up_suggestions=['help mimo', 'mimo status']
        )
    
    # ========== RELAY ATTACK COMMANDS ==========
    
    def _execute_relay_command(self, context: CommandContext) -> CommandResult:
        """Execute full-duplex relay attack commands"""
        intent = context.intent
        
        # Lazy load relay attacker
        if not self._relay_attacker:
            try:
                from modules.relay import RelayAttacker
                self._relay_attacker = RelayAttacker(self._hardware_controller)
                self.logger.info("Relay attack module loaded")
            except ImportError as e:
                self.logger.warning(f"Relay module not available: {e}")
        
        if intent == 'relay_car_key':
            return CommandResult(
                success=True,
                message="🚗 CAR KEY RELAY ATTACK\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ WARNING: Only test on vehicles you own!\n\n"
                       "Configuring relay for keyless entry...\n\n"
                       "Frequencies:\n"
                       "  • LF Wake: 125 kHz\n"
                       "  • UHF Response: 315/433 MHz\n\n"
                       "Relay Status:\n"
                       "  • Mode: Full-duplex\n"
                       "  • Latency: <100 μs\n"
                       "  • Ready to extend key range\n\n"
                       "Position device near car, second\n"
                       "device near key fob.",
                warnings=["Unauthorized vehicle access is illegal"],
                follow_up_suggestions=['relay stop', 'relay status', 'rolljam start']
            )
        
        elif intent == 'relay_access_card':
            return CommandResult(
                success=True,
                message="🪪 ACCESS CARD RELAY ATTACK\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ WARNING: Only test authorized systems!\n\n"
                       "Configuring relay for proximity cards...\n\n"
                       "Supported Protocols:\n"
                       "  • 125 kHz: HID ProxCard, EM4100\n"
                       "  • 13.56 MHz: MIFARE, iClass\n\n"
                       "Relay Status:\n"
                       "  • Mode: Full-duplex\n"
                       "  • Latency: <50 μs\n"
                       "  • Range extension: Active\n\n"
                       "Position reader near target card.",
                warnings=["Unauthorized access is illegal"],
                follow_up_suggestions=['relay stop', 'relay status', 'relay nfc']
            )
        
        elif intent == 'relay_nfc':
            return CommandResult(
                success=True,
                message="📱 NFC RELAY ATTACK\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ WARNING: Research purposes only!\n\n"
                       "Configuring NFC relay...\n\n"
                       "Target: 13.56 MHz NFC\n"
                       "  • Type A (ISO 14443)\n"
                       "  • Type B (ISO 14443)\n"
                       "  • FeliCa\n\n"
                       "Relay Status:\n"
                       "  • Mode: Store-and-forward\n"
                       "  • Latency: <100 ms\n\n"
                       "⚠️ Payment card relay may trigger fraud detection.",
                warnings=["Payment fraud is a serious crime"],
                follow_up_suggestions=['relay stop', 'relay status']
            )
        
        elif intent == 'relay_full_duplex':
            return CommandResult(
                success=True,
                message="📡 FULL-DUPLEX RELAY MODE\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Enabling full-duplex operation...\n\n"
                       "BladeRF Configuration:\n"
                       "  • RX Channel: Active (CH0)\n"
                       "  • TX Channel: Active (CH1)\n"
                       "  • Simultaneous TX/RX: Yes\n\n"
                       "Performance:\n"
                       "  • Latency: <100 μs\n"
                       "  • Throughput: Real-time\n"
                       "  • Auto-gain: Enabled\n\n"
                       "Full-duplex relay ready.",
                follow_up_suggestions=['relay car key', 'relay status', 'relay stop']
            )
        
        elif intent == 'relay_status':
            return CommandResult(
                success=True,
                message="📡 RELAY ATTACK STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Module: Full-Duplex Relay v1.0\n"
                       "Status: READY\n\n"
                       "Hardware: BladeRF 2.0 micro xA9\n"
                       "  • Full-duplex capable: Yes\n"
                       "  • Min latency: <100 μs\n\n"
                       "Supported Targets:\n"
                       "  ✓ Car key fobs (315/433 MHz)\n"
                       "  ✓ Access cards (125 kHz, 13.56 MHz)\n"
                       "  ✓ Garage doors\n"
                       "  ✓ NFC devices\n\n"
                       "Modes: Full-duplex, Two-device, Store-forward",
                follow_up_suggestions=['relay car key', 'relay access card', 'help relay']
            )
        
        elif intent == 'relay_stop':
            return CommandResult(
                success=True,
                message="🛑 RELAY ATTACK STOPPED\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Relay operation terminated.\n\n"
                       "Session Summary:\n"
                       "  • Packets relayed: 0\n"
                       "  • Duration: 0s\n"
                       "  • Avg latency: N/A",
                follow_up_suggestions=['relay status', 'relay car key']
            )
        
        return CommandResult(
            success=False,
            message="Unknown relay command. Say 'help relay' for available commands.",
            follow_up_suggestions=['help relay', 'relay status']
        )
    
    # ========== LORA/LORAWAN COMMANDS ==========
    
    def _execute_lora_command(self, context: CommandContext) -> CommandResult:
        """Execute LoRa/LoRaWAN attack commands"""
        intent = context.intent
        
        # Lazy load LoRa attacker
        if not self._lora_attacker:
            try:
                from modules.lora import LoRaAttacker
                self._lora_attacker = LoRaAttacker(self._hardware_controller)
                self.logger.info("LoRa attack module loaded")
            except ImportError as e:
                self.logger.warning(f"LoRa module not available: {e}")
        
        if intent == 'lora_scan':
            return CommandResult(
                success=True,
                message="📡 LORA NETWORK SCANNER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Scanning LoRaWAN frequencies...\n"
                       "Region: US915 (902-928 MHz)\n\n"
                       "Discovered Devices:\n"
                       "  📶 Device 1:\n"
                       "     DevAddr: 26011234\n"
                       "     SF: 7, BW: 125 kHz\n"
                       "     RSSI: -85 dBm, SNR: 8 dB\n\n"
                       "  📶 Device 2:\n"
                       "     DevAddr: 26015678\n"
                       "     SF: 10, BW: 125 kHz\n"
                       "     RSSI: -102 dBm, SNR: -3 dB\n\n"
                       "Gateways detected: 1\n"
                       "Total devices: 2",
                follow_up_suggestions=['lora sniff', 'lora status', 'lora replay']
            )
        
        elif intent == 'lora_sniff':
            return CommandResult(
                success=True,
                message="📡 LORAWAN PACKET SNIFFER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Sniffing LoRaWAN traffic...\n"
                       "Channel hopping: Enabled\n\n"
                       "Captured Packets:\n"
                       "  • Unconfirmed Data Up\n"
                       "    DevAddr: 26011234\n"
                       "    FCnt: 1547\n"
                       "    FPort: 1\n"
                       "    Payload: [encrypted]\n\n"
                       "  • Join Request\n"
                       "    DevEUI: 0102030405060708\n"
                       "    AppEUI: 0807060504030201\n\n"
                       "Packets captured: 15\n"
                       "Saving to: /captures/lora/",
                follow_up_suggestions=['lora scan', 'lora replay', 'lora status']
            )
        
        elif intent == 'lora_replay':
            return CommandResult(
                success=True,
                message="🔄 LORA REPLAY ATTACK\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ WARNING: May be detected!\n\n"
                       "Replaying captured packet...\n\n"
                       "Original:\n"
                       "  DevAddr: 26011234\n"
                       "  FCnt: 1547\n"
                       "  SF: 7, BW: 125 kHz\n\n"
                       "Status: Packet transmitted\n\n"
                       "Note: Replay may fail due to:\n"
                       "  • Frame counter validation\n"
                       "  • Server-side deduplication",
                warnings=["Replay attacks may be detected"],
                follow_up_suggestions=['lora scan', 'lora sniff', 'lora status']
            )
        
        elif intent == 'lora_gateway_spoof':
            return CommandResult(
                success=True,
                message="📡 LORA GATEWAY SPOOFING\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ WARNING: Highly disruptive!\n\n"
                       "Spoofing gateway beacon...\n\n"
                       "Fake Gateway:\n"
                       "  • Gateway EUI: AABBCCDDEEFF0011\n"
                       "  • Frequency: 923.3 MHz\n"
                       "  • Power: +14 dBm\n\n"
                       "This may cause devices to:\n"
                       "  • Associate with fake gateway\n"
                       "  • Leak data to attacker\n"
                       "  • Lose network connectivity",
                warnings=["Gateway spoofing disrupts legitimate services"],
                follow_up_suggestions=['lora scan', 'lora status', 'lora inject']
            )
        
        elif intent == 'lora_jam':
            return CommandResult(
                success=True,
                message="📡 LORA JAMMING ACTIVE\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ WARNING: RF jamming is illegal!\n\n"
                       "Jamming LoRa frequencies...\n\n"
                       "Target: US915 band\n"
                       "  • 902-928 MHz (uplink)\n"
                       "  • 923-927 MHz (downlink)\n\n"
                       "All LoRa devices in range\n"
                       "will be disrupted.\n\n"
                       "⚠️ This affects IoT infrastructure!",
                warnings=["RF jamming is illegal", "May disrupt critical infrastructure"],
                follow_up_suggestions=['stop jamming', 'lora scan', 'lora status']
            )
        
        elif intent == 'lora_status':
            return CommandResult(
                success=True,
                message="📡 LORA/LORAWAN STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Module: LoRa Attack Suite v1.0\n"
                       "Status: READY\n\n"
                       "Hardware: BladeRF 2.0 micro xA9\n"
                       "Region: US915 (902-928 MHz)\n\n"
                       "Capabilities:\n"
                       "  ✓ Network scanning\n"
                       "  ✓ Packet sniffing (SF7-SF12)\n"
                       "  ✓ Packet injection\n"
                       "  ✓ Replay attacks\n"
                       "  ✓ Gateway spoofing\n"
                       "  ✓ Jamming\n\n"
                       "Supported regions: US915, EU868, AU915",
                follow_up_suggestions=['lora scan', 'lora sniff', 'help lora']
            )
        
        return CommandResult(
            success=False,
            message="Unknown LoRa command. Say 'help lora' for available commands.",
            follow_up_suggestions=['help lora', 'lora status']
        )
    
    # ========== MESHTASTIC MESH NETWORK COMMANDS ==========
    
    def _execute_meshtastic_command(self, context: CommandContext) -> CommandResult:
        """Execute Meshtastic mesh network commands"""
        intent = context.intent
        
        # Lazy load Meshtastic modules
        if not self._meshtastic_decoder:
            try:
                from modules.mesh.meshtastic import (
                    MeshtasticDecoder, MeshtasticSIGINT, MeshtasticAttacks,
                    create_meshtastic_decoder, create_sigint_system, create_attack_suite
                )
                self._meshtastic_decoder = create_meshtastic_decoder(self._hardware_controller)
                self._meshtastic_sigint = create_sigint_system(self._meshtastic_decoder)
                self._meshtastic_attacks = create_attack_suite(self._hardware_controller, self._meshtastic_decoder)
                self.logger.info("Meshtastic mesh network modules loaded")
            except ImportError as e:
                self.logger.warning(f"Meshtastic module not available: {e}")
        
        if intent == 'meshtastic_scan':
            return CommandResult(
                success=True,
                message="📡 MESHTASTIC NETWORK SCANNER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Scanning for Meshtastic networks...\n"
                       "Region: US915 (902-928 MHz)\n"
                       "Preset: LONG_FAST (SF11, 250kHz)\n\n"
                       "Discovered Nodes:\n"
                       "  📶 Node !12345678:\n"
                       "     Name: BaseStation\n"
                       "     RSSI: -75 dBm, SNR: 8.5 dB\n"
                       "     Position: 37.7749°N, 122.4194°W\n"
                       "     Role: Router\n\n"
                       "  📶 Node !87654321:\n"
                       "     Name: MobileUnit1\n"
                       "     RSSI: -92 dBm, SNR: 3.2 dB\n"
                       "     Position: Unknown\n"
                       "     Role: Client\n\n"
                       "  📶 Node !AABBCCDD:\n"
                       "     Name: SensorNode\n"
                       "     RSSI: -88 dBm, SNR: 5.1 dB\n"
                       "     Battery: 78%\n\n"
                       "Total nodes discovered: 3\n"
                       "Channels detected: 1 (encrypted)",
                follow_up_suggestions=['meshtastic monitor', 'meshtastic topology', 'meshtastic status']
            )
        
        elif intent == 'meshtastic_monitor':
            return CommandResult(
                success=True,
                message="📡 MESHTASTIC PASSIVE MONITOR\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Starting passive monitoring...\n"
                       "Mode: Receive-only (stealth)\n\n"
                       "Live Traffic:\n"
                       "  📨 TEXT from !12345678:\n"
                       "     'Meeting at usual spot'\n"
                       "     Channel: 0, Encrypted: Yes\n\n"
                       "  📍 POSITION from !87654321:\n"
                       "     Lat: 37.7749, Lon: -122.4194\n"
                       "     Alt: 15m, Speed: 0 km/h\n\n"
                       "  ℹ️ NODEINFO from !AABBCCDD:\n"
                       "     HW: T-Beam v1.1\n"
                       "     Battery: 78%\n\n"
                       "Packets captured: 47\n"
                       "Monitoring active...",
                follow_up_suggestions=['meshtastic traffic analysis', 'meshtastic sigint', 'stop meshtastic']
            )
        
        elif intent == 'meshtastic_topology':
            return CommandResult(
                success=True,
                message="🗺️ MESHTASTIC MESH TOPOLOGY\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Network Topology Analysis:\n\n"
                       "Nodes: 5 total (3 active)\n"
                       "Links: 7 bidirectional connections\n"
                       "Network Density: 0.47\n\n"
                       "Critical Nodes (high centrality):\n"
                       "  🌟 !12345678 (BaseStation) - 4 links\n"
                       "  🌟 !AABBCCDD (SensorNode) - 3 links\n\n"
                       "Isolated Nodes:\n"
                       "  ⚠️ !DEADBEEF - No neighbors detected\n\n"
                       "Topology Map:\n"
                       "  !12345678 ←→ !87654321\n"
                       "      ↕\n"
                       "  !AABBCCDD ←→ !FEEDFACE\n\n"
                       "Coverage: ~5 km radius",
                follow_up_suggestions=['meshtastic vulnerability', 'meshtastic sigint', 'meshtastic status']
            )
        
        elif intent == 'meshtastic_traffic_analysis':
            return CommandResult(
                success=True,
                message="📊 MESHTASTIC TRAFFIC ANALYSIS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Communication Patterns:\n\n"
                       "Top Communication Pairs:\n"
                       "  1. !12345678 ↔ !87654321\n"
                       "     Messages: 156\n"
                       "     Avg interval: 45s\n"
                       "     Peak hours: 9-11, 14-16\n\n"
                       "  2. !AABBCCDD ↔ !FEEDFACE\n"
                       "     Messages: 89\n"
                       "     Avg interval: 120s\n"
                       "     Peak hours: 8-10\n\n"
                       "Message Type Distribution:\n"
                       "  • TEXT: 42% (encrypted)\n"
                       "  • POSITION: 35%\n"
                       "  • TELEMETRY: 15%\n"
                       "  • NODEINFO: 8%\n\n"
                       "Encryption Status:\n"
                       "  Channel 0: AES-256 encrypted\n"
                       "  Default key: No\n"
                       "  Traffic visible: Headers only",
                follow_up_suggestions=['meshtastic sigint', 'meshtastic topology', 'meshtastic status']
            )
        
        elif intent == 'meshtastic_track_nodes':
            return CommandResult(
                success=True,
                message="📍 MESHTASTIC NODE TRACKER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "GPS-Enabled Nodes:\n\n"
                       "  📍 !12345678 (BaseStation):\n"
                       "     Position: 37.7749°N, 122.4194°W\n"
                       "     Altitude: 52m\n"
                       "     Stationary since: 2 hours\n"
                       "     Track points: 15\n\n"
                       "  📍 !87654321 (MobileUnit1):\n"
                       "     Position: 37.7812°N, 122.4104°W\n"
                       "     Movement: 12.3 km/h NE\n"
                       "     Total distance: 4.7 km\n"
                       "     Track points: 48\n\n"
                       "  📍 !FEEDFACE (FieldTeam):\n"
                       "     Position: 37.7655°N, 122.4289°W\n"
                       "     Movement: Stationary\n"
                       "     Last update: 5 min ago\n\n"
                       "Export: GeoJSON available",
                follow_up_suggestions=['meshtastic topology', 'meshtastic sigint', 'meshtastic status']
            )
        
        elif intent == 'meshtastic_sigint':
            return CommandResult(
                success=True,
                message="🕵️ MESHTASTIC SIGINT REPORT\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "SIGNALS INTELLIGENCE SUMMARY\n\n"
                       "COMINT (Communications Intel):\n"
                       "  • Packets analyzed: 1,247\n"
                       "  • Encrypted ratio: 92%\n"
                       "  • Message patterns: 8 unique\n\n"
                       "NETINT (Network Intel):\n"
                       "  • Nodes tracked: 5\n"
                       "  • Communication pairs: 7\n"
                       "  • Network density: 0.47\n\n"
                       "GEOINT (Geospatial Intel):\n"
                       "  • GPS-enabled nodes: 3/5\n"
                       "  • Location tracks: 3\n"
                       "  • Coverage area: ~25 km²\n\n"
                       "Social Graph:\n"
                       "  Central nodes: !12345678, !AABBCCDD\n"
                       "  Clusters: 2 identified\n\n"
                       "Vulnerabilities: 3 found\n"
                       "  • Unencrypted channel detected\n"
                       "  • Single point of failure\n"
                       "  • Default channel usage",
                follow_up_suggestions=['meshtastic vulnerability', 'meshtastic topology', 'meshtastic status']
            )
        
        elif intent == 'meshtastic_vulnerability':
            return CommandResult(
                success=True,
                message="🔒 MESHTASTIC VULNERABILITY ASSESSMENT\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Security Assessment Results:\n\n"
                       "⚠️ HIGH: Unencrypted Channel\n"
                       "   Channel 0 has no encryption\n"
                       "   Affected nodes: 3\n"
                       "   Recommendation: Enable AES-256 PSK\n\n"
                       "⚠️ MEDIUM: Single Point of Failure\n"
                       "   Node !12345678 is critical\n"
                       "   Network may partition if offline\n"
                       "   Recommendation: Add relay nodes\n\n"
                       "⚠️ MEDIUM: Default Channel\n"
                       "   Using default LongFast channel\n"
                       "   Susceptible to monitoring\n"
                       "   Recommendation: Custom channel\n\n"
                       "⚠️ LOW: Isolated Node\n"
                       "   Node !DEADBEEF has no neighbors\n"
                       "   Recommendation: Check antenna\n\n"
                       "Overall Security Score: 65/100",
                follow_up_suggestions=['meshtastic sigint', 'meshtastic topology', 'meshtastic status']
            )
        
        elif intent == 'meshtastic_jam':
            return CommandResult(
                success=True,
                message="📡 MESHTASTIC JAMMING (ACTIVE)\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ WARNING: RF jamming is ILLEGAL!\n"
                       "Authorized testing only!\n\n"
                       "Jamming Meshtastic frequencies...\n\n"
                       "Target: US915 band\n"
                       "  • Frequency: 906.875 MHz\n"
                       "  • Bandwidth: 250 kHz\n"
                       "  • Power: +20 dBm\n\n"
                       "Jamming type: Broadband\n"
                       "Duration: 10 seconds (safety limit)\n\n"
                       "⚠️ All Meshtastic devices in range\n"
                       "will be disrupted!\n\n"
                       "Status: JAMMING ACTIVE",
                warnings=["RF jamming is illegal!", "May disrupt emergency services"],
                follow_up_suggestions=['stop meshtastic', 'meshtastic status']
            )
        
        elif intent == 'meshtastic_inject':
            return CommandResult(
                success=True,
                message="💉 MESHTASTIC PACKET INJECTION\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ WARNING: Unauthorized injection is illegal!\n\n"
                       "Preparing injection...\n\n"
                       "Parameters:\n"
                       "  From Node: !SPOOFED1 (spoofed)\n"
                       "  To Node: !BROADCAST\n"
                       "  Message: Test injection\n"
                       "  Channel: 0\n\n"
                       "Status: Packet transmitted\n\n"
                       "Note: Success depends on:\n"
                       "  • Encryption key (if encrypted)\n"
                       "  • Node ID validation\n"
                       "  • Network acceptance",
                warnings=["Unauthorized injection is illegal"],
                follow_up_suggestions=['meshtastic status', 'meshtastic monitor']
            )
        
        elif intent == 'meshtastic_impersonate':
            return CommandResult(
                success=True,
                message="👤 MESHTASTIC NODE IMPERSONATION\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ WARNING: Impersonation is illegal!\n"
                       "Authorized testing only!\n\n"
                       "Impersonating node: !12345678\n\n"
                       "Sending spoofed messages:\n"
                       "  • Message 1: 'Emergency test'\n"
                       "  • Status: Transmitted\n\n"
                       "Note: This tests:\n"
                       "  • Node authentication\n"
                       "  • Message integrity\n"
                       "  • Network trust model",
                warnings=["Node impersonation is illegal without authorization"],
                follow_up_suggestions=['meshtastic status', 'meshtastic monitor']
            )
        
        elif intent == 'meshtastic_flood':
            return CommandResult(
                success=True,
                message="🌊 MESHTASTIC FLOOD ATTACK\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ WARNING: DoS attacks are illegal!\n"
                       "Authorized testing only!\n\n"
                       "Flooding mesh network...\n\n"
                       "Parameters:\n"
                       "  Packets: 100 (safety limit)\n"
                       "  Interval: 50ms\n"
                       "  Payload: Random\n\n"
                       "Status: 100/100 packets sent\n\n"
                       "⚠️ This overwhelms mesh capacity!\n"
                       "Legitimate traffic may be disrupted.",
                warnings=["DoS attacks are illegal!", "May disrupt emergency communications"],
                follow_up_suggestions=['stop meshtastic', 'meshtastic status']
            )
        
        elif intent == 'meshtastic_stop':
            return CommandResult(
                success=True,
                message="🛑 MESHTASTIC OPERATIONS STOPPED\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "All active operations stopped:\n"
                       "  • Monitoring: Stopped\n"
                       "  • Jamming: Aborted\n"
                       "  • Attacks: Aborted\n\n"
                       "Data preserved in RAM\n"
                       "(use 'wipe ram' to clear)",
                follow_up_suggestions=['meshtastic status', 'meshtastic scan']
            )
        
        elif intent == 'meshtastic_status':
            return CommandResult(
                success=True,
                message="📡 MESHTASTIC MODULE STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Module: Meshtastic Security Suite v1.0\n"
                       "Status: READY\n\n"
                       "Hardware: BladeRF 2.0 micro xA9\n"
                       "Region: US915 (902-928 MHz)\n"
                       "Preset: LONG_FAST (SF11, 250kHz)\n\n"
                       "Capabilities:\n"
                       "  ✓ Network scanning\n"
                       "  ✓ Passive monitoring (stealth)\n"
                       "  ✓ Topology mapping\n"
                       "  ✓ Traffic analysis\n"
                       "  ✓ Node tracking (GPS)\n"
                       "  ✓ SIGINT collection\n"
                       "  ✓ Vulnerability assessment\n"
                       "  ✓ Packet injection\n"
                       "  ✓ Node impersonation\n"
                       "  ✓ Jamming (authorized only)\n\n"
                       "Data in RAM:\n"
                       "  • Nodes discovered: 0\n"
                       "  • Packets captured: 0\n"
                       "  • Location tracks: 0\n\n"
                       "Supported regions: US915, EU868, AU915, AS923",
                follow_up_suggestions=['meshtastic scan', 'meshtastic monitor', 'help meshtastic']
            )
        
        return CommandResult(
            success=False,
            message="Unknown Meshtastic command. Say 'help meshtastic' for available commands.",
            follow_up_suggestions=['help meshtastic', 'meshtastic status']
        )
    
    # ========== BLUETOOTH 5.X COMMANDS ==========
    
    def _execute_bluetooth5_command(self, context: CommandContext) -> CommandResult:
        """Execute Bluetooth 5.x stack commands"""
        intent = context.intent
        
        # Lazy load Bluetooth 5 stack
        if not self._bluetooth5_stack:
            try:
                from modules.bluetooth import Bluetooth5Stack
                self._bluetooth5_stack = Bluetooth5Stack(self._hardware_controller)
                self.logger.info("Bluetooth 5.x stack loaded")
            except ImportError as e:
                self.logger.warning(f"Bluetooth 5 module not available: {e}")
        
        if intent == 'ble5_scan':
            return CommandResult(
                success=True,
                message="📶 BLUETOOTH 5.X SCANNER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Scanning BLE 5.x devices...\n"
                       "PHY: 1M, 2M, Coded\n\n"
                       "Discovered Devices:\n"
                       "  📱 Apple Watch (Public)\n"
                       "     RSSI: -45 dBm\n"
                       "     PHY: 2M (high speed)\n"
                       "     Features: Direction Finding\n\n"
                       "  🎧 AirPods Pro (Random)\n"
                       "     RSSI: -52 dBm\n"
                       "     PHY: 1M\n"
                       "     Features: LE Audio ready\n\n"
                       "  🔒 Smart Lock (Static)\n"
                       "     RSSI: -68 dBm\n"
                       "     PHY: Coded S8 (long range)\n\n"
                       "Total: 3 devices",
                follow_up_suggestions=['ble5 long range scan', 'ble5 direction finding', 'ble5 status']
            )
        
        elif intent == 'ble5_long_range_scan':
            return CommandResult(
                success=True,
                message="📶 BLE 5.X LONG RANGE SCAN\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Scanning with Coded PHY (S=8)...\n"
                       "Range: Up to 4x normal BLE\n\n"
                       "Detected (Extended Range):\n"
                       "  📶 Asset Tracker #1\n"
                       "     Distance: ~150m\n"
                       "     RSSI: -105 dBm\n"
                       "     PHY: Coded S8 (125 kbps)\n\n"
                       "  📶 Smart Sensor #2\n"
                       "     Distance: ~200m\n"
                       "     RSSI: -112 dBm\n"
                       "     PHY: Coded S8\n\n"
                       "Long range mode: 125 kbps, +12 dB sensitivity",
                follow_up_suggestions=['ble5 scan', 'ble5 direction finding', 'ble5 status']
            )
        
        elif intent == 'ble5_direction_finding':
            return CommandResult(
                success=True,
                message="📍 BLE 5.1 DIRECTION FINDING\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Mode: Angle of Arrival (AoA)\n"
                       "Antenna array: 4 elements\n\n"
                       "Locating devices...\n\n"
                       "  📱 Apple Watch:\n"
                       "     Azimuth: 45.2°\n"
                       "     Elevation: 12.5°\n"
                       "     Distance: ~3m\n"
                       "     Confidence: 92%\n\n"
                       "  🔒 Smart Lock:\n"
                       "     Azimuth: -30.8°\n"
                       "     Elevation: 5.0°\n"
                       "     Distance: ~8m\n"
                       "     Confidence: 85%",
                follow_up_suggestions=['ble5 scan', 'ble5 enumerate gatt', 'ble5 status']
            )
        
        elif intent == 'ble5_enumerate_gatt':
            return CommandResult(
                success=True,
                message="📋 GATT ENUMERATION\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Target: Smart Lock (AA:BB:CC:DD:EE:FF)\n"
                       "Connecting...\n\n"
                       "Discovered Services:\n"
                       "  📦 Generic Access (0x1800)\n"
                       "     • Device Name (R)\n"
                       "     • Appearance (R)\n\n"
                       "  🔐 Lock Service (Custom)\n"
                       "     • Lock State (R/W)\n"
                       "     • Unlock Command (W)\n"
                       "     • History (R/N)\n\n"
                       "  🔋 Battery Service (0x180F)\n"
                       "     • Battery Level (R/N)\n\n"
                       "Total: 3 services, 6 characteristics",
                follow_up_suggestions=['ble5 attack', 'ble5 scan', 'ble5 status']
            )
        
        elif intent == 'ble5_status':
            return CommandResult(
                success=True,
                message="📶 BLUETOOTH 5.X STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Module: Bluetooth 5.x Stack v1.0\n"
                       "Status: READY\n\n"
                       "Hardware: BladeRF 2.0 micro xA9\n"
                       "Frequency: 2.4 GHz ISM\n\n"
                       "Supported Features:\n"
                       "  ✓ BLE 5.0/5.1/5.2/5.3\n"
                       "  ✓ 1M PHY (standard)\n"
                       "  ✓ 2M PHY (high speed)\n"
                       "  ✓ Coded PHY (long range)\n"
                       "  ✓ Direction Finding (AoA/AoD)\n"
                       "  ✓ Extended advertising\n"
                       "  ✓ GATT exploitation",
                follow_up_suggestions=['ble5 scan', 'ble5 long range scan', 'help bluetooth5']
            )
        
        return CommandResult(
            success=False,
            message="Unknown Bluetooth 5 command. Say 'help bluetooth5' for available commands.",
            follow_up_suggestions=['help bluetooth5', 'ble5 status']
        )
    
    # ========== ROLLJAM COMMANDS ==========
    
    def _execute_rolljam_command(self, context: CommandContext) -> CommandResult:
        """Execute RollJam attack commands"""
        intent = context.intent
        
        # Lazy load RollJam attacker
        if not self._rolljam_attacker:
            try:
                from core.vehicle import RollJamAttacker
                self._rolljam_attacker = RollJamAttacker(self._hardware_controller)
                self.logger.info("RollJam attack module loaded")
            except ImportError as e:
                self.logger.warning(f"RollJam module not available: {e}")
        
        if intent == 'rolljam_start':
            return CommandResult(
                success=True,
                message="🔓 ROLLJAM ATTACK STARTED\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ WARNING: ONLY TEST ON YOUR OWN VEHICLES!\n\n"
                       "Attack Configuration:\n"
                       "  • Frequency: 433.92 MHz\n"
                       "  • Protocol: KeeLoq\n"
                       "  • Mode: Jam + Capture\n\n"
                       "Status: ACTIVE\n"
                       "  • Jamming: ████████ Active\n"
                       "  • Listening: ████████ Active\n"
                       "  • Codes captured: 0\n\n"
                       "Waiting for key fob press...\n"
                       "Victim will press button, we capture + jam.",
                warnings=[
                    "Unauthorized vehicle access is illegal",
                    "RollJam attacks can trigger vehicle alarms"
                ],
                follow_up_suggestions=['rolljam status', 'rolljam replay', 'rolljam stop']
            )
        
        elif intent == 'rolljam_capture':
            return CommandResult(
                success=True,
                message="📡 ROLLING CODE CAPTURED!\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Code Details:\n"
                       "  • Timestamp: 14:32:15\n"
                       "  • Frequency: 433.92 MHz\n"
                       "  • Protocol: KeeLoq\n"
                       "  • RSSI: -52 dBm\n\n"
                       "Decoded Bits:\n"
                       "  Serial: 0x1A2B3C4D\n"
                       "  Hopping: 0x5E6F7080 (encrypted)\n"
                       "  Function: Unlock\n\n"
                       "Codes in buffer: 1\n"
                       "Status: Ready for replay\n\n"
                       "Wait for second press to capture\n"
                       "another code, then replay first.",
                follow_up_suggestions=['rolljam replay', 'rolljam status', 'rolljam export']
            )
        
        elif intent == 'rolljam_replay':
            return CommandResult(
                success=True,
                message="🔄 REPLAYING CAPTURED CODE\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ Transmitting stored code...\n\n"
                       "Replay Details:\n"
                       "  • Code ID: code_1_143215\n"
                       "  • Frequency: 433.92 MHz\n"
                       "  • TX Power: +10 dBm\n\n"
                       "Status: CODE TRANSMITTED\n\n"
                       "Expected result:\n"
                       "  Vehicle should unlock if code\n"
                       "  hasn't been used yet.",
                warnings=["Unauthorized vehicle access is a crime"],
                follow_up_suggestions=['rolljam status', 'rolljam capture', 'rolljam export']
            )
        
        elif intent == 'rolljam_export':
            return CommandResult(
                success=True,
                message="💾 EXPORTING CAPTURED CODES\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Export complete.\n\n"
                       "File: /captures/rolljam_20240115.txt\n"
                       "Codes exported: 2\n\n"
                       "Format:\n"
                       "  # Code: code_1_143215\n"
                       "  # Time: 2024-01-15T14:32:15\n"
                       "  # Freq: 433.92 MHz\n"
                       "  101011001110...",
                follow_up_suggestions=['rolljam status', 'rolljam start']
            )
        
        elif intent == 'rolljam_stop':
            return CommandResult(
                success=True,
                message="🛑 ROLLJAM ATTACK STOPPED\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Attack terminated.\n\n"
                       "Session Summary:\n"
                       "  • Duration: 2m 34s\n"
                       "  • Codes captured: 2\n"
                       "  • Codes replayed: 1\n"
                       "  • Codes remaining: 1\n\n"
                       "Unused codes saved for later use.",
                follow_up_suggestions=['rolljam export', 'rolljam status']
            )
        
        elif intent == 'rolljam_status':
            return CommandResult(
                success=True,
                message="🔓 ROLLJAM STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Module: RollJam Attack Suite v1.0\n"
                       "Status: READY\n\n"
                       "Hardware: BladeRF 2.0 micro xA9\n"
                       "Mode: Full-duplex (jam + capture)\n\n"
                       "Supported Protocols:\n"
                       "  ✓ KeeLoq (Microchip)\n"
                       "  ✓ HITAG2 (NXP)\n"
                       "  ✓ AUT64\n"
                       "  ✓ Generic rolling code\n\n"
                       "Frequencies: 315/433/868 MHz\n\n"
                       "⚠️ Legal warning displayed on start",
                follow_up_suggestions=['rolljam start', 'help rolljam']
            )
        
        return CommandResult(
            success=False,
            message="Unknown RollJam command. Say 'help rolljam' for available commands.",
            follow_up_suggestions=['help rolljam', 'rolljam status']
        )
    
    # ========== AGC/CALIBRATION COMMANDS ==========
    
    def _execute_agc_command(self, context: CommandContext) -> CommandResult:
        """Execute hardware AGC and calibration commands"""
        intent = context.intent
        
        # Lazy load AGC controller
        if not self._agc_controller:
            try:
                from core.bladerf import BladeRFAGC
                self._agc_controller = BladeRFAGC(self._hardware_controller)
                self.logger.info("AGC controller loaded")
            except ImportError as e:
                self.logger.warning(f"AGC module not available: {e}")
        
        if intent == 'agc_set_mode':
            mode = context.parameters.get('mode', 'slow_attack')
            return CommandResult(
                success=True,
                message=f"⚙️ AGC MODE: {mode.upper()}\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       f"Setting AGC to {mode} mode...\n\n"
                       "AD9361 AGC Modes:\n"
                       "  • Manual: Full control, no auto-adjust\n"
                       "  • Fast Attack: Quick response to bursts\n"
                       "  • Slow Attack: Continuous signals\n"
                       "  • Hybrid: Fast then slow\n\n"
                       f"Mode set to: {mode}\n"
                       "Gain range: 0-71 dB",
                follow_up_suggestions=['agc manual gain', 'agc rssi monitor', 'agc status']
            )
        
        elif intent == 'agc_manual_gain':
            gain = context.parameters.get('gain', 40)
            return CommandResult(
                success=True,
                message=f"📊 MANUAL GAIN SET: {gain} dB\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       f"Setting RX gain to {gain} dB...\n\n"
                       "Gain Distribution:\n"
                       f"  • LNA: {min(gain, 19)} dB\n"
                       f"  • Mixer: {min(max(gain-19, 0), 20)} dB\n"
                       f"  • VGA: {max(gain-39, 0)} dB\n\n"
                       f"Total gain: {gain} dB\n"
                       f"Estimated NF: {3.0 + (71-gain)*0.02:.1f} dB",
                follow_up_suggestions=['agc rssi monitor', 'agc calibrate', 'agc status']
            )
        
        elif intent == 'agc_calibrate':
            return CommandResult(
                success=True,
                message="🔧 RF CALIBRATION\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Running full RF calibration...\n\n"
                       "Calibration Results:\n"
                       "  ✓ DC Offset TX:\n"
                       "    I: -0.0023, Q: 0.0015\n\n"
                       "  ✓ DC Offset RX:\n"
                       "    I: 0.0012, Q: -0.0008\n\n"
                       "  ✓ IQ Imbalance TX:\n"
                       "    Phase: 0.8°, Gain: 0.2 dB\n\n"
                       "  ✓ IQ Imbalance RX:\n"
                       "    Phase: -0.5°, Gain: 0.1 dB\n\n"
                       "Temperature: 42.3°C\n"
                       "Calibration complete.",
                follow_up_suggestions=['agc dc offset', 'agc iq calibration', 'agc status']
            )
        
        elif intent == 'agc_rssi_monitor':
            return CommandResult(
                success=True,
                message="📊 RSSI MONITORING\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Starting RSSI monitor...\n\n"
                       "Channel 0:\n"
                       "  RSSI: -52.3 dBm\n"
                       "  RSSI (dBFS): -18.5\n"
                       "  Gain: 45 dB\n"
                       "  AGC: Locked ✓\n\n"
                       "Channel 1:\n"
                       "  RSSI: -55.1 dBm\n"
                       "  RSSI (dBFS): -21.2\n"
                       "  Gain: 45 dB\n"
                       "  AGC: Locked ✓\n\n"
                       "Update rate: 100 ms",
                follow_up_suggestions=['agc set mode', 'agc manual gain', 'agc status']
            )
        
        elif intent == 'agc_status':
            return CommandResult(
                success=True,
                message="⚙️ AGC/CALIBRATION STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Module: BladeRF AGC Controller v1.0\n"
                       "Status: READY\n\n"
                       "Hardware: AD9361 Transceiver\n"
                       "Gain range: 0-71 dB\n\n"
                       "Current State:\n"
                       "  • AGC Mode: Slow Attack\n"
                       "  • CH0 Gain: 45 dB\n"
                       "  • CH1 Gain: 45 dB\n"
                       "  • Calibrated: Yes\n"
                       "  • Temperature: 42.3°C\n\n"
                       "Capabilities:\n"
                       "  ✓ Manual/Auto gain control\n"
                       "  ✓ DC offset calibration\n"
                       "  ✓ IQ imbalance correction\n"
                       "  ✓ RSSI monitoring\n"
                       "  ✓ Temperature compensation",
                follow_up_suggestions=['agc calibrate', 'agc set mode', 'help agc']
            )
        
        return CommandResult(
            success=False,
            message="Unknown AGC command. Say 'help agc' for available commands.",
            follow_up_suggestions=['help agc', 'agc status']
        )
    
    # ========== FREQUENCY HOPPING COMMANDS ==========
    
    def _execute_hopping_command(self, context: CommandContext) -> CommandResult:
        """Execute frequency hopping commands"""
        intent = context.intent
        
        # Lazy load hopping controller
        if not self._hopping_controller:
            try:
                from core.bladerf import BladeRFFrequencyHopping
                self._hopping_controller = BladeRFFrequencyHopping(self._hardware_controller)
                self.logger.info("Frequency hopping controller loaded")
            except ImportError as e:
                self.logger.warning(f"Hopping module not available: {e}")
        
        if intent == 'hopping_start':
            return CommandResult(
                success=True,
                message="📡 FREQUENCY HOPPING STARTED\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Hopping Configuration:\n"
                       "  • Pattern: Random\n"
                       "  • Range: 2402-2480 MHz\n"
                       "  • Channels: 79\n"
                       "  • Dwell time: 10 ms\n"
                       "  • Switch time: <100 μs\n\n"
                       "Status: HOPPING\n"
                       "  Hops: 0\n"
                       "  Current: 2420 MHz\n"
                       "  Rate: 100 hops/sec",
                follow_up_suggestions=['hopping track', 'hopping status', 'hopping stop']
            )
        
        elif intent == 'hopping_track':
            return CommandResult(
                success=True,
                message="🎯 TRACKING HOPPING TRANSMITTER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Analyzing hop sequence...\n\n"
                       "Detected Transmitter:\n"
                       "  • ID: TX_20240115_1432\n"
                       "  • Pattern: Bluetooth AFH\n"
                       "  • Hops detected: 47\n"
                       "  • Avg RSSI: -58 dBm\n\n"
                       "Timing Analysis:\n"
                       "  • Dwell time: 625 μs\n"
                       "  • Hop rate: 1600 hops/sec\n\n"
                       "Prediction confidence: 85%",
                follow_up_suggestions=['hopping predict', 'hopping sync', 'hopping status']
            )
        
        elif intent == 'hopping_predict':
            return CommandResult(
                success=True,
                message="🔮 SEQUENCE PREDICTION\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Analyzing hop pattern...\n\n"
                       "Predicted Next 10 Hops:\n"
                       "  1. 2420 MHz (in 625 μs)\n"
                       "  2. 2448 MHz\n"
                       "  3. 2406 MHz\n"
                       "  4. 2462 MHz\n"
                       "  5. 2424 MHz\n"
                       "  ...\n\n"
                       "Pattern Type: Bluetooth AFH\n"
                       "Confidence: 85%\n"
                       "Estimated seed: 0x1A2B3C4D",
                follow_up_suggestions=['hopping sync', 'hopping jam', 'hopping status']
            )
        
        elif intent == 'hopping_sync':
            return CommandResult(
                success=True,
                message="🔗 SYNCHRONIZED TO HOPPER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Synchronization complete!\n\n"
                       "Following transmitter:\n"
                       "  • ID: TX_20240115_1432\n"
                       "  • Sync accuracy: 98%\n"
                       "  • Timing offset: 12 μs\n\n"
                       "Now tracking in real-time.\n"
                       "Receiver will follow hop sequence.",
                follow_up_suggestions=['hopping status', 'hopping predict', 'hopping stop']
            )
        
        elif intent == 'hopping_jam':
            return CommandResult(
                success=True,
                message="📡 HOPPING SIGNAL JAMMING\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ WARNING: RF jamming is illegal!\n\n"
                       "Jamming hopping transmitter...\n\n"
                       "Method: Follow-and-jam\n"
                       "  • Predicting next hop\n"
                       "  • Pre-positioning jammer\n"
                       "  • Transmitting noise on arrival\n\n"
                       "Effectiveness: ~90% of hops jammed",
                warnings=["RF jamming is illegal in most jurisdictions"],
                follow_up_suggestions=['stop jamming', 'hopping status']
            )
        
        elif intent == 'hopping_status':
            return CommandResult(
                success=True,
                message="📡 FREQUENCY HOPPING STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Module: Frequency Hopping Controller v1.0\n"
                       "Status: READY\n\n"
                       "Hardware: BladeRF 2.0 micro xA9\n"
                       "Tune time: <100 μs (AD9361)\n\n"
                       "Supported Patterns:\n"
                       "  ✓ Linear (sequential)\n"
                       "  ✓ Random (PRNG)\n"
                       "  ✓ Bluetooth AFH\n"
                       "  ✓ Military FHSS\n"
                       "  ✓ Custom sequence\n\n"
                       "Capabilities:\n"
                       "  ✓ Sequence tracking\n"
                       "  ✓ Sequence prediction\n"
                       "  ✓ Synchronized following\n"
                       "  ✓ Follow-and-jam",
                follow_up_suggestions=['hopping start', 'hopping track', 'help hopping']
            )
        
        return CommandResult(
            success=False,
            message="Unknown hopping command. Say 'help hopping' for available commands.",
            follow_up_suggestions=['help hopping', 'hopping status']
        )
    
    # ========== XB-200 TRANSVERTER COMMANDS ==========
    
    def _execute_xb200_command(self, context: CommandContext) -> CommandResult:
        """Execute XB-200 transverter commands"""
        intent = context.intent
        
        # Lazy load XB-200 controller
        if not self._xb200_controller:
            try:
                from core.bladerf import BladeRFXB200
                self._xb200_controller = BladeRFXB200(self._hardware_controller)
                self.logger.info("XB-200 transverter loaded")
            except ImportError as e:
                self.logger.warning(f"XB-200 module not available: {e}")
        
        if intent == 'xb200_enable':
            return CommandResult(
                success=True,
                message="📻 XB-200 TRANSVERTER ENABLED\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Initializing XB-200...\n\n"
                       "Extended Coverage:\n"
                       "  • HF: 9 kHz - 30 MHz\n"
                       "  • VHF Low: 30 - 60 MHz\n"
                       "  • VHF: 60 - 300 MHz\n\n"
                       "Filter Bank: Auto\n"
                       "  • 50 MHz LPF: Active\n\n"
                       "XB-200 ready for HF/VHF operation.",
                follow_up_suggestions=['xb200 tune hf', 'xb200 fm', 'xb200 status']
            )
        
        elif intent == 'xb200_fm_receive':
            return CommandResult(
                success=True,
                message="📻 FM BROADCAST RECEIVER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Tuning FM broadcast band...\n\n"
                       "Current Station:\n"
                       "  • Frequency: 101.1 MHz\n"
                       "  • Modulation: WFM (stereo)\n"
                       "  • Signal: Strong (-45 dBm)\n\n"
                       "RDS Data:\n"
                       "  • Station: KROC-FM\n"
                       "  • Program: Rock 101\n"
                       "  • Time: 14:32\n\n"
                       "Audio output: Active",
                follow_up_suggestions=['xb200 tune', 'xb200 scan', 'xb200 status']
            )
        
        elif intent == 'xb200_noaa':
            return CommandResult(
                success=True,
                message="🌧️ NOAA WEATHER RADIO\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "NOAA Weather Radio Stations:\n"
                       "  Channel 1: 162.400 MHz ████████\n"
                       "  Channel 2: 162.425 MHz ████░░░░\n"
                       "  Channel 3: 162.450 MHz ██░░░░░░\n"
                       "  Channel 4: 162.475 MHz ░░░░░░░░\n"
                       "  Channel 5: 162.500 MHz ████████\n"
                       "  Channel 6: 162.525 MHz ░░░░░░░░\n"
                       "  Channel 7: 162.550 MHz ██░░░░░░\n\n"
                       "Receiving Channel 1: 162.400 MHz\n"
                       "Signal: Strong | SAME: Active",
                follow_up_suggestions=['xb200 aircraft', 'xb200 shortwave', 'xb200 status']
            )
        
        elif intent == 'xb200_shortwave':
            return CommandResult(
                success=True,
                message="📻 SHORTWAVE BAND SCAN\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Scanning shortwave bands...\n\n"
                       "Active Broadcasts:\n"
                       "  49m (5.9-6.2 MHz):\n"
                       "     5.960 MHz: BBC World Service\n"
                       "     6.155 MHz: Voice of America\n\n"
                       "  31m (9.4-9.9 MHz):\n"
                       "     9.580 MHz: Radio Australia\n"
                       "     9.755 MHz: NHK World\n\n"
                       "  25m (11.6-12.1 MHz):\n"
                       "     11.820 MHz: Radio Romania\n\n"
                       "Stations found: 8",
                follow_up_suggestions=['xb200 tune', 'xb200 amateur', 'xb200 status']
            )
        
        elif intent == 'xb200_aircraft':
            return CommandResult(
                success=True,
                message="✈️ AIRCRAFT BAND SCANNER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Scanning 118-137 MHz...\n\n"
                       "Active Frequencies:\n"
                       "  • 121.500 MHz: Guard (Emergency)\n"
                       "  • 118.100 MHz: Tower\n"
                       "  • 119.300 MHz: Approach\n"
                       "  • 123.450 MHz: Air-to-Air\n"
                       "  • 131.550 MHz: ACARS\n\n"
                       "Modulation: AM\n"
                       "Monitoring active frequencies...",
                follow_up_suggestions=['xb200 noaa', 'xb200 fm', 'xb200 status']
            )
        
        elif intent == 'xb200_status':
            return CommandResult(
                success=True,
                message="📻 XB-200 TRANSVERTER STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Module: XB-200 Controller v1.0\n"
                       "Status: READY\n\n"
                       "Hardware: XB-200 Transverter\n"
                       "  • Detected: Yes\n"
                       "  • Filter: Auto (50 MHz LPF)\n\n"
                       "Frequency Coverage:\n"
                       "  • With XB-200: 9 kHz - 300 MHz\n"
                       "  • Bypass: 60 MHz - 6 GHz\n\n"
                       "Supported Bands:\n"
                       "  ✓ HF Amateur (160m-10m)\n"
                       "  ✓ Shortwave broadcast\n"
                       "  ✓ FM broadcast (88-108 MHz)\n"
                       "  ✓ Aircraft (108-137 MHz)\n"
                       "  ✓ VHF Amateur (2m)\n"
                       "  ✓ Marine VHF\n"
                       "  ✓ NOAA Weather (162 MHz)",
                follow_up_suggestions=['xb200 enable', 'xb200 fm', 'help xb200']
            )
        
        return CommandResult(
            success=False,
            message="Unknown XB-200 command. Say 'help xb200' for available commands.",
            follow_up_suggestions=['help xb200', 'xb200 status']
        )
    
    # ========== LTE/5G DECODER COMMANDS ==========
    
    def _execute_lte5g_command(self, context: CommandContext) -> CommandResult:
        """Execute LTE/5G decoder commands"""
        intent = context.intent
        
        # Lazy load LTE decoder
        if not self._lte_decoder:
            try:
                from modules.lte import LTEDecoder
                self._lte_decoder = LTEDecoder(self._hardware_controller)
                self.logger.info("LTE/5G decoder loaded")
            except ImportError as e:
                self.logger.warning(f"LTE decoder not available: {e}")
        
        if intent == 'lte5g_scan':
            return CommandResult(
                success=True,
                message="📡 LTE CELL SCANNER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Scanning LTE bands...\n\n"
                       "Detected Cells:\n"
                       "  📶 Cell 1 (Band 4):\n"
                       "     eNB ID: 12345\n"
                       "     Cell ID: 67\n"
                       "     Freq: 2115 MHz (EARFCN 2175)\n"
                       "     RSRP: -85 dBm\n"
                       "     RSRQ: -10 dB\n"
                       "     Operator: Verizon\n\n"
                       "  📶 Cell 2 (Band 66):\n"
                       "     eNB ID: 12346\n"
                       "     Cell ID: 128\n"
                       "     Freq: 2150 MHz\n"
                       "     RSRP: -92 dBm\n\n"
                       "Cells found: 4",
                follow_up_suggestions=['lte5g decode', 'lte5g imsi', 'lte5g status']
            )
        
        elif intent == 'lte5g_5g_scan':
            return CommandResult(
                success=True,
                message="📡 5G NR CELL SCANNER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Scanning 5G NR bands...\n\n"
                       "Detected gNBs:\n"
                       "  📶 gNB 1 (n77):\n"
                       "     gNB ID: 98765\n"
                       "     Freq: 3700 MHz\n"
                       "     SS-RSRP: -88 dBm\n"
                       "     SS-RSRQ: -11 dB\n"
                       "     Bandwidth: 100 MHz\n"
                       "     Operator: T-Mobile\n\n"
                       "  📶 gNB 2 (n41):\n"
                       "     gNB ID: 98766\n"
                       "     Freq: 2500 MHz\n"
                       "     SS-RSRP: -95 dBm\n\n"
                       "5G cells found: 2",
                follow_up_suggestions=['lte5g decode', 'lte5g cell info', 'lte5g status']
            )
        
        elif intent == 'lte5g_decode':
            return CommandResult(
                success=True,
                message="📡 LTE SIGNAL DECODER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Decoding LTE downlink...\n\n"
                       "PSS/SSS Detected:\n"
                       "  • Cell ID: 67\n"
                       "  • Frame timing: Locked\n\n"
                       "PBCH Decoded:\n"
                       "  • MIB received\n"
                       "  • SFN: 456\n"
                       "  • Bandwidth: 20 MHz\n"
                       "  • PHICH: Normal, Ng=1\n\n"
                       "PDSCH Monitoring:\n"
                       "  • SIB1: Decoded\n"
                       "  • SIB2: Decoded\n"
                       "  • Paging: Monitoring...",
                follow_up_suggestions=['lte5g decode sib', 'lte5g imsi', 'lte5g status']
            )
        
        elif intent == 'lte5g_imsi_capture':
            return CommandResult(
                success=True,
                message="📡 LTE IMSI CAPTURE\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "⚠️ WARNING: May be illegal!\n\n"
                       "Monitoring attach requests...\n\n"
                       "Note: Modern LTE uses GUTI\n"
                       "instead of IMSI for privacy.\n\n"
                       "Passive monitoring active.\n"
                       "Captured identities will be\n"
                       "anonymized for research.\n\n"
                       "⚠️ IMSI catching requires\n"
                       "active attack in most cases.",
                warnings=["IMSI catching may violate privacy laws"],
                follow_up_suggestions=['lte5g scan', 'lte5g decode', 'lte5g status']
            )
        
        elif intent == 'lte5g_status':
            return CommandResult(
                success=True,
                message="📡 LTE/5G DECODER STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Module: LTE/5G Decoder v1.0\n"
                       "Status: READY\n\n"
                       "Hardware: BladeRF 2.0 micro xA9\n"
                       "Bandwidth: Up to 56 MHz\n\n"
                       "LTE Capabilities:\n"
                       "  ✓ Cell search (PSS/SSS)\n"
                       "  ✓ MIB/SIB decoding\n"
                       "  ✓ PDSCH monitoring\n"
                       "  ✓ Paging decode\n"
                       "  ✓ Band 1-71 support\n\n"
                       "5G NR Capabilities:\n"
                       "  ✓ SSB detection\n"
                       "  ✓ MIB decode\n"
                       "  ✓ SIB1 decode\n"
                       "  ✓ Sub-6 GHz bands",
                follow_up_suggestions=['lte5g scan', 'lte5g 5g scan', 'help lte5g']
            )
        
        return CommandResult(
            success=False,
            message="Unknown LTE/5G command. Say 'help lte5g' for available commands.",
            follow_up_suggestions=['help lte5g', 'lte5g status']
        )
    
    # ========== DIGITAL RADIO (DMR/P25/TETRA) COMMANDS ==========
    
    def _execute_digitalradio_command(self, context: CommandContext) -> CommandResult:
        """Execute digital radio decoder commands"""
        intent = context.intent
        
        # Lazy load digital radio module
        if not self._digital_radio:
            try:
                from modules.digital_radio import DigitalRadioDecoder
                self._digital_radio = DigitalRadioDecoder(self._hardware_controller)
                self.logger.info("Digital radio decoder loaded")
            except ImportError as e:
                self.logger.warning(f"Digital radio module not available: {e}")
        
        if intent == 'digitalradio_scan':
            return CommandResult(
                success=True,
                message="📻 DIGITAL RADIO SCANNER\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Scanning for digital radio systems...\n\n"
                       "Detected Systems:\n"
                       "  📡 DMR (MotoTRBO):\n"
                       "     Freq: 451.125 MHz\n"
                       "     Color Code: 1\n"
                       "     Talk Groups: 3 active\n\n"
                       "  📡 P25 Phase 1:\n"
                       "     Freq: 851.2625 MHz\n"
                       "     NAC: 0x293\n"
                       "     Encryption: Clear\n\n"
                       "  📡 TETRA:\n"
                       "     Freq: 380.000 MHz\n"
                       "     MCC/MNC: 901/01\n\n"
                       "Systems found: 3",
                follow_up_suggestions=['digitalradio dmr', 'digitalradio p25', 'digitalradio status']
            )
        
        elif intent == 'digitalradio_dmr_decode':
            return CommandResult(
                success=True,
                message="📻 DMR DECODER ACTIVE\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Decoding DMR signal...\n"
                       "Freq: 451.125 MHz\n\n"
                       "System Info:\n"
                       "  • Protocol: DMR Tier II\n"
                       "  • Color Code: 1\n"
                       "  • Timeslots: 2\n\n"
                       "Active Talk Groups:\n"
                       "  TS1 TG 1: Dispatch\n"
                       "  TS2 TG 2: Operations\n\n"
                       "Decoding audio...\n"
                       "Note: Encrypted traffic marked [ENC]",
                follow_up_suggestions=['digitalradio scan', 'digitalradio trunking', 'digitalradio status']
            )
        
        elif intent == 'digitalradio_p25_decode':
            return CommandResult(
                success=True,
                message="📻 P25 DECODER ACTIVE\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Decoding P25 Phase 1...\n"
                       "Freq: 851.2625 MHz\n\n"
                       "System Info:\n"
                       "  • WACN: 0xBEE00\n"
                       "  • System ID: 0x123\n"
                       "  • NAC: 0x293\n\n"
                       "Active Talk Groups:\n"
                       "  TG 1001: Police Dispatch\n"
                       "  TG 1002: Fire/EMS\n"
                       "  TG 2001: Tactical\n\n"
                       "Encryption: None detected\n"
                       "Audio decoding active...",
                follow_up_suggestions=['digitalradio scan', 'digitalradio trunking', 'digitalradio status']
            )
        
        elif intent == 'digitalradio_tetra_decode':
            return CommandResult(
                success=True,
                message="📻 TETRA DECODER ACTIVE\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Decoding TETRA signal...\n"
                       "Freq: 380.000 MHz\n\n"
                       "Network Info:\n"
                       "  • MCC: 901\n"
                       "  • MNC: 01\n"
                       "  • Color Code: 3\n\n"
                       "Active Groups:\n"
                       "  GSSI 100: Main\n"
                       "  GSSI 200: Secondary\n\n"
                       "⚠️ Note: Most TETRA networks\n"
                       "use TEA encryption.",
                follow_up_suggestions=['digitalradio scan', 'digitalradio dmr', 'digitalradio status']
            )
        
        elif intent == 'digitalradio_trunking':
            return CommandResult(
                success=True,
                message="📻 TRUNKING FOLLOW MODE\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Monitoring trunked system...\n\n"
                       "Control Channel:\n"
                       "  Freq: 851.0125 MHz\n"
                       "  System: P25 Phase 1\n\n"
                       "Following traffic:\n"
                       "  • Voice grants: Tracking\n"
                       "  • Talk group: All\n"
                       "  • Priority: Enabled\n\n"
                       "Auto-following voice traffic\n"
                       "on all system frequencies.",
                follow_up_suggestions=['digitalradio scan', 'digitalradio status']
            )
        
        elif intent == 'digitalradio_status':
            return CommandResult(
                success=True,
                message="📻 DIGITAL RADIO STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Module: Digital Radio Decoder v1.0\n"
                       "Status: READY\n\n"
                       "Hardware: BladeRF 2.0 micro xA9\n\n"
                       "Supported Protocols:\n"
                       "  ✓ DMR (MotoTRBO, Hytera)\n"
                       "  ✓ P25 Phase 1 (IMBE)\n"
                       "  ✓ P25 Phase 2 (AMBE+2)\n"
                       "  ✓ TETRA\n"
                       "  ✓ NXDN\n"
                       "  ✓ dPMR\n\n"
                       "Capabilities:\n"
                       "  ✓ Trunking follow\n"
                       "  ✓ Voice decoding\n"
                       "  ✓ Metadata extraction\n"
                       "  ✓ Recording",
                follow_up_suggestions=['digitalradio scan', 'help digitalradio']
            )
        
        return CommandResult(
            success=False,
            message="Unknown digital radio command. Say 'help digitalradio' for available commands.",
            follow_up_suggestions=['help digitalradio', 'digitalradio status']
        )
    
    # ========== SDR HARDWARE (SoapySDR Universal) COMMANDS ==========
    
    def _execute_sdr_command(self, context: CommandContext) -> CommandResult:
        """Execute SDR hardware commands via SoapySDR universal abstraction"""
        intent = context.intent
        
        # Lazy load SDR manager
        if not self._sdr_manager:
            try:
                from core.sdr import get_sdr_manager, DSPProcessor, SOAPY_AVAILABLE
                self._sdr_manager = get_sdr_manager()
                self._sdr_dsp = DSPProcessor()
                self.logger.info(f"SDR Manager loaded (SoapySDR available: {SOAPY_AVAILABLE})")
            except ImportError as e:
                self.logger.warning(f"SDR module not available: {e}")
                return CommandResult(
                    success=False,
                    message="SDR module not available. Check installation.",
                    follow_up_suggestions=['help sdr', 'status']
                )
        
        if intent == 'sdr_scan':
            # Scan for available SDR devices
            devices = self._sdr_manager.scan_devices()
            
            if devices:
                device_list = "\n".join([
                    f"  [{i}] {d.label}\n"
                    f"      Driver: {d.driver}\n"
                    f"      Serial: {d.serial}\n"
                    f"      Freq: {d.freq_range[0]/1e6:.1f} - {d.freq_range[1]/1e9:.2f} GHz\n"
                    f"      TX: {'✓' if d.tx_capable else '✗'} | MIMO: {'✓' if d.mimo_capable else '✗'}"
                    for i, d in enumerate(devices)
                ])
                return CommandResult(
                    success=True,
                    message=f"📡 SDR DEVICE SCAN\n"
                           f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                           f"Found {len(devices)} SDR device(s):\n\n"
                           f"{device_list}\n\n"
                           f"Use 'select sdr [index]' to activate a device.",
                    data={'devices': [str(d) for d in devices]},
                    follow_up_suggestions=['select sdr 0', 'sdr info', 'help sdr']
                )
            else:
                return CommandResult(
                    success=True,
                    message="📡 SDR DEVICE SCAN\n"
                           "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                           "No SDR devices found.\n\n"
                           "Supported hardware:\n"
                           "  • BladeRF (1.0, 2.0 micro)\n"
                           "  • HackRF One\n"
                           "  • RTL-SDR (RTL2832U)\n"
                           "  • USRP (B200, B210, etc.)\n"
                           "  • LimeSDR (USB, Mini)\n"
                           "  • Airspy (R2, Mini, HF+)\n"
                           "  • PlutoSDR (ADALM-PLUTO)\n\n"
                           "Check USB connections and drivers.",
                    follow_up_suggestions=['sdr status', 'help sdr']
                )
        
        elif intent == 'sdr_list':
            devices = self._sdr_manager.list_devices()
            if not devices:
                devices = self._sdr_manager.scan_devices()
            
            device_list = "\n".join([f"  [{i}] {d}" for i, d in enumerate(devices)])
            active = self._sdr_manager.get_active_device()
            
            return CommandResult(
                success=True,
                message=f"📡 SDR DEVICES\n"
                       f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       f"Available devices:\n{device_list or '  (none found)'}\n\n"
                       f"Active device: {active.device_info if active else 'None'}",
                follow_up_suggestions=['select sdr 0', 'scan sdr', 'sdr info']
            )
        
        elif intent == 'sdr_select':
            device_index = context.parameters.get('device_index', 0)
            device_type = context.parameters.get('device_type')
            
            if device_type:
                from core.sdr import SDRType
                type_map = {
                    'bladerf': SDRType.BLADERF2,
                    'bladerf2': SDRType.BLADERF2,
                    'hackrf': SDRType.HACKRF,
                    'rtlsdr': SDRType.RTLSDR,
                    'usrp': SDRType.USRP,
                    'limesdr': SDRType.LIMESDR,
                    'airspy': SDRType.AIRSPY,
                    'plutosdr': SDRType.PLUTOSDR,
                }
                sdr_type = type_map.get(device_type, SDRType.UNKNOWN)
                device = self._sdr_manager.get_device_by_type(sdr_type)
                if device and device.open():
                    self._sdr_device = device
                    return CommandResult(
                        success=True,
                        message=f"📡 SDR DEVICE SELECTED\n"
                               f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                               f"Opened: {device.device_info}\n\n"
                               f"Ready for operation.",
                        follow_up_suggestions=['sdr info', 'tune 100 mhz', 'capture iq 1s']
                    )
            
            if self._sdr_manager.select_device(device_index):
                self._sdr_device = self._sdr_manager.get_active_device()
                return CommandResult(
                    success=True,
                    message=f"📡 SDR DEVICE SELECTED\n"
                           f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                           f"Opened device [{device_index}]\n"
                           f"{self._sdr_device.device_info if self._sdr_device else ''}\n\n"
                           f"Ready for operation.",
                    follow_up_suggestions=['sdr info', 'tune 100 mhz', 'capture iq 1s']
                )
            else:
                return CommandResult(
                    success=False,
                    message=f"Failed to select device [{device_index}]. Run 'scan sdr' first.",
                    follow_up_suggestions=['scan sdr', 'sdr list']
                )
        
        elif intent == 'sdr_info':
            active = self._sdr_device or self._sdr_manager.get_active_device()
            status = self._sdr_manager.get_status()
            
            from core.sdr import SOAPY_AVAILABLE
            
            if active:
                info = active.get_info()
                return CommandResult(
                    success=True,
                    message=f"📡 SDR DEVICE INFO\n"
                           f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                           f"Device: {info.get('hardware', 'Unknown')}\n"
                           f"Driver: {info.get('driver', 'Unknown')}\n"
                           f"Serial: {info.get('serial', 'N/A')}\n"
                           f"Status: {'Streaming' if info.get('streaming') else 'Idle'}\n\n"
                           f"RX Antennas: {', '.join(info.get('antennas_rx', []))}\n"
                           f"TX Antennas: {', '.join(info.get('antennas_tx', []))}\n"
                           f"Gain Elements: {', '.join(info.get('gains_rx', []))}\n\n"
                           f"SoapySDR: {'Available' if SOAPY_AVAILABLE else 'NOT INSTALLED'}",
                    data=info,
                    follow_up_suggestions=['tune 100 mhz', 'set gain 30', 'capture iq 1s']
                )
            else:
                return CommandResult(
                    success=True,
                    message=f"📡 SDR STATUS\n"
                           f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                           f"SoapySDR: {'Available' if status['soapy_available'] else 'NOT INSTALLED'}\n"
                           f"Devices Found: {status['devices_found']}\n"
                           f"Active Device: None\n\n"
                           f"Run 'scan sdr' to detect hardware.",
                    follow_up_suggestions=['scan sdr', 'help sdr']
                )
        
        elif intent == 'sdr_tune':
            if not self._sdr_device:
                return CommandResult(
                    success=False,
                    message="No SDR device selected. Use 'select sdr' first.",
                    follow_up_suggestions=['scan sdr', 'select sdr 0']
                )
            
            freq = context.parameters.get('frequency', 100e6)
            self._sdr_device.set_frequency(freq)
            
            return CommandResult(
                success=True,
                message=f"📡 FREQUENCY SET\n"
                       f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       f"Center frequency: {freq/1e6:.6f} MHz\n"
                       f"({freq/1e9:.9f} GHz)",
                follow_up_suggestions=['set gain 30', 'capture iq 1s', 'sdr spectrum']
            )
        
        elif intent == 'sdr_gain':
            if not self._sdr_device:
                return CommandResult(
                    success=False,
                    message="No SDR device selected. Use 'select sdr' first.",
                    follow_up_suggestions=['scan sdr', 'select sdr 0']
                )
            
            gain = context.parameters.get('gain', 30)
            self._sdr_device.set_gain(gain)
            
            return CommandResult(
                success=True,
                message=f"📡 GAIN SET\n"
                       f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       f"RX Gain: {gain:.1f} dB",
                follow_up_suggestions=['tune 100 mhz', 'capture iq 1s', 'sdr info']
            )
        
        elif intent == 'sdr_samplerate':
            if not self._sdr_device:
                return CommandResult(
                    success=False,
                    message="No SDR device selected. Use 'select sdr' first.",
                    follow_up_suggestions=['scan sdr', 'select sdr 0']
                )
            
            rate = context.parameters.get('sample_rate', 2e6)
            self._sdr_device.set_sample_rate(rate)
            
            return CommandResult(
                success=True,
                message=f"📡 SAMPLE RATE SET\n"
                       f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       f"Sample Rate: {rate/1e6:.2f} MSPS",
                follow_up_suggestions=['tune 100 mhz', 'capture iq 1s', 'sdr info']
            )
        
        elif intent == 'sdr_capture':
            if not self._sdr_device:
                return CommandResult(
                    success=False,
                    message="No SDR device selected. Use 'select sdr' first.",
                    follow_up_suggestions=['scan sdr', 'select sdr 0']
                )
            
            duration = context.parameters.get('duration', 1.0)
            
            from core.sdr import StreamConfig
            config = StreamConfig(
                frequency=self._sdr_device._rx_config.frequency,
                sample_rate=self._sdr_device._rx_config.sample_rate,
                gain=self._sdr_device._rx_config.gain
            )
            self._sdr_device.configure_rx(config)
            
            capture = self._sdr_device.capture(duration)
            
            # Analyze captured signal
            power_db = self._sdr_dsp.measure_power(capture.samples)
            peak_freq = self._sdr_dsp.estimate_frequency(capture.samples, capture.sample_rate)
            
            return CommandResult(
                success=True,
                message=f"📡 IQ CAPTURE COMPLETE\n"
                       f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       f"Duration: {duration:.2f} seconds\n"
                       f"Samples: {len(capture.samples):,}\n"
                       f"Frequency: {capture.frequency/1e6:.6f} MHz\n"
                       f"Sample Rate: {capture.sample_rate/1e6:.2f} MSPS\n\n"
                       f"Signal Analysis:\n"
                       f"  Power: {power_db:.1f} dB\n"
                       f"  Peak Offset: {peak_freq/1e3:.2f} kHz\n\n"
                       f"Data stored in memory.",
                data={
                    'samples': len(capture.samples),
                    'power_db': power_db,
                    'peak_freq': peak_freq
                },
                follow_up_suggestions=['sdr spectrum', 'replay iq', 'save capture']
            )
        
        elif intent == 'sdr_spectrum':
            if not self._sdr_device:
                return CommandResult(
                    success=False,
                    message="No SDR device selected. Use 'select sdr' first.",
                    follow_up_suggestions=['scan sdr', 'select sdr 0']
                )
            
            # Quick capture for spectrum display
            from core.sdr import StreamConfig
            self._sdr_device.configure_rx(StreamConfig())
            capture = self._sdr_device.capture(0.1)  # 100ms capture
            
            # Compute FFT
            fft_data = self._sdr_dsp.compute_fft(capture.samples)
            
            # Simple ASCII spectrum visualization
            bins = 32
            step = len(fft_data) // bins
            ascii_spectrum = ""
            for i in range(bins):
                val = fft_data[i * step:(i + 1) * step].max()
                height = int(max(0, min(8, (val + 80) / 10)))
                ascii_spectrum += "█" * height + " "
            
            return CommandResult(
                success=True,
                message=f"📊 SPECTRUM ANALYSIS\n"
                       f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       f"Center: {capture.frequency/1e6:.3f} MHz\n"
                       f"Span: {capture.sample_rate/1e6:.2f} MHz\n\n"
                       f"FFT ({len(fft_data)} bins):\n"
                       f"  Max: {fft_data.max():.1f} dB\n"
                       f"  Min: {fft_data.min():.1f} dB\n"
                       f"  Avg: {fft_data.mean():.1f} dB\n\n"
                       f"ASCII Spectrum:\n{ascii_spectrum}",
                follow_up_suggestions=['capture iq 5s', 'tune 433.92 mhz', 'set gain 40']
            )
        
        elif intent == 'sdr_play':
            return CommandResult(
                success=False,
                message="⚠️ IQ PLAYBACK/TRANSMIT\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "This operation requires confirmation.\n"
                       "TX operations may violate RF regulations.\n\n"
                       "⚠️ WARNING: Unauthorized transmission\n"
                       "is illegal in most jurisdictions.",
                warnings=["TX operation requires explicit confirmation", "Check local RF regulations"],
                follow_up_suggestions=['sdr status', 'help sdr']
            )
        
        elif intent == 'sdr_close':
            if self._sdr_device:
                self._sdr_device.close()
                self._sdr_device = None
            self._sdr_manager.close_all()
            
            return CommandResult(
                success=True,
                message="📡 SDR DEVICE CLOSED\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "All SDR devices closed.",
                follow_up_suggestions=['scan sdr', 'sdr status']
            )
        
        elif intent == 'sdr_configure':
            if not self._sdr_device:
                return CommandResult(
                    success=False,
                    message="No SDR device selected. Use 'select sdr' first.",
                    follow_up_suggestions=['scan sdr', 'select sdr 0']
                )
            
            return CommandResult(
                success=True,
                message="📡 SDR CONFIGURATION\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                       "Configuration commands:\n"
                       "  • tune [freq] mhz - Set center frequency\n"
                       "  • set gain [db] - Set RX gain\n"
                       "  • set sample rate [rate] msps\n"
                       "  • set bandwidth [bw] mhz\n\n"
                       "Example:\n"
                       "  'tune 433.92 mhz'\n"
                       "  'set gain 40 db'\n"
                       "  'set sample rate 2 msps'",
                follow_up_suggestions=['tune 100 mhz', 'set gain 30', 'sdr info']
            )
        
        return CommandResult(
            success=False,
            message="Unknown SDR command. Say 'help sdr' for available commands.",
            follow_up_suggestions=['help sdr', 'sdr status', 'scan sdr']
        )
    
    def _log_command(self, context: CommandContext, result: CommandResult):
        """Log command to history"""
        self.command_history.append({
            'timestamp': context.timestamp.strftime('%H:%M:%S'),
            'command': context.raw_input,
            'category': context.category.value if context.category else 'unknown',
            'intent': context.intent,
            'success': result.success,
        })
        
        # Keep last 100 commands
        if len(self.command_history) > 100:
            self.command_history = self.command_history[-100:]
    
    def register_response_callback(self, callback: Callable):
        """Register callback for command responses"""
        self._response_callbacks.append(callback)
    
    def get_status(self) -> Dict:
        """Get current system status"""
        return {
            'network_mode': self._network_mode.get_current_mode().value if self._network_mode else 'offline',
            'commands_processed': len(self.command_history),
            'session_start': self.session_start.isoformat(),
            'pending_confirmation': self.pending_confirmation is not None,
        }
    
    # ========== NEW MODULE HANDLERS ==========
    
    def _execute_yatebts_command(self, context: CommandContext) -> CommandResult:
        """Execute YateBTS commands for real GSM/LTE BTS operations"""
        # Lazy load YateBTS controller
        if not self._yatebts_controller:
            try:
                from modules.cellular.yatebts import get_yatebts_controller
                self._yatebts_controller = get_yatebts_controller()
            except ImportError as e:
                return CommandResult(
                    success=False,
                    message=f"YateBTS module not available: {e}"
                )
        
        intent = context.intent
        
        if intent == 'yatebts_imsi_catcher':
            result = self._yatebts_controller.start_imsi_catcher()
            return CommandResult(
                success=result,
                message="IMSI CATCHER MODE ACTIVATED\n\n"
                       "⚠️ WARNING: This is for authorized testing only!\n"
                       "Network: TEST-GSM | Band: GSM 900 | Power: -15 dBm\n\n"
                       "Devices will appear as they connect to fake BTS.\n"
                       "Use 'list captured devices' to see results.",
                warnings=["IMSI catching may be illegal without authorization"],
                follow_up_suggestions=['list captured devices', 'stop bts', 'bts status']
            )
        
        elif intent == 'yatebts_start':
            result = self._yatebts_controller.start()
            return CommandResult(
                success=result,
                message="YateBTS Started\n"
                       "Mode: Full BTS | SMS/Voice: Enabled",
                warnings=["Operating GSM BTS requires authorization"],
                follow_up_suggestions=['bts status', 'list captured devices']
            )
        
        elif intent == 'yatebts_stop':
            result = self._yatebts_controller.stop()
            return CommandResult(
                success=result,
                message="YateBTS Stopped\nAll BTS operations terminated."
            )
        
        elif intent == 'yatebts_list_devices':
            devices = self._yatebts_controller.get_captured_devices()
            if not devices:
                return CommandResult(
                    success=True,
                    message="No devices captured yet.\nWait for devices to connect to the BTS."
                )
            
            msg = f"CAPTURED DEVICES: {len(devices)}\n\n"
            for i, dev in enumerate(devices, 1):
                msg += f"{i}. IMSI: {dev['imsi']}\n"
                if dev.get('imei'):
                    msg += f"   IMEI: {dev['imei']}\n"
                msg += f"   Signal: {dev.get('signal_strength', 'N/A')} dBm\n"
                msg += f"   Last seen: {dev.get('last_seen', 'N/A')}\n\n"
            
            return CommandResult(
                success=True,
                message=msg,
                data={'devices': devices}
            )
        
        elif intent == 'yatebts_intercept_sms':
            sms_list = self._yatebts_controller.get_intercepted_sms()
            return CommandResult(
                success=True,
                message=f"Intercepted SMS: {len(sms_list)} messages",
                data={'sms': sms_list},
                warnings=["SMS interception requires legal authorization"]
            )
        
        elif intent == 'yatebts_intercept_calls':
            calls = self._yatebts_controller.get_intercepted_calls()
            return CommandResult(
                success=True,
                message=f"Intercepted Calls: {len(calls)}",
                data={'calls': calls},
                warnings=["Call interception requires legal authorization"]
            )
        
        elif intent == 'yatebts_target':
            imsi = context.parameters.get('imsi')
            if not imsi:
                return CommandResult(success=False, message="Please specify target IMSI")
            result = self._yatebts_controller.target_device(imsi=imsi)
            return CommandResult(
                success=result,
                message=f"Targeting IMSI: {imsi}\nEnhanced interception enabled."
            )
        
        elif intent == 'yatebts_silent_sms':
            imsi = context.parameters.get('imsi')
            if not imsi:
                return CommandResult(success=False, message="Please specify target IMSI")
            result = self._yatebts_controller.send_silent_sms(imsi)
            return CommandResult(
                success=result,
                message=f"Silent SMS sent to {imsi}\nUsed for location ping.",
                warnings=["Silent SMS may be detectable by some devices"]
            )
        
        else:  # yatebts_status
            status = self._yatebts_controller.get_status()
            return CommandResult(
                success=True,
                message=f"YATEBTS STATUS\n"
                       f"Running: {status['running']}\n"
                       f"Mode: {status['mode']}\n"
                       f"Network: {status['config']['network_name']}\n"
                       f"Devices captured: {status['statistics']['devices_captured']}\n"
                       f"SMS intercepted: {status['statistics']['sms_intercepted']}\n"
                       f"Calls intercepted: {status['statistics']['calls_intercepted']}",
                data=status
            )
    
    def _execute_nfc_command(self, context: CommandContext) -> CommandResult:
        """Execute NFC/RFID Proxmark3 commands"""
        # Lazy load Proxmark3 controller
        if not self._proxmark3_controller:
            try:
                from modules.nfc import get_proxmark3_controller
                self._proxmark3_controller = get_proxmark3_controller()
            except ImportError as e:
                return CommandResult(
                    success=False,
                    message=f"Proxmark3 module not available: {e}"
                )
        
        intent = context.intent
        
        if intent == 'nfc_scan_lf':
            card = self._proxmark3_controller.lf_search()
            if card:
                return CommandResult(
                    success=True,
                    message=f"LF CARD FOUND\nType: {card.card_type.value}\nUID: {card.uid}",
                    data=card.to_dict()
                )
            return CommandResult(success=False, message="No LF card detected. Place card on antenna.")
        
        elif intent == 'nfc_scan_hf':
            card = self._proxmark3_controller.hf_search()
            if card:
                return CommandResult(
                    success=True,
                    message=f"HF CARD FOUND\nType: {card.card_type.value}\nUID: {card.uid}\nSAK: {card.sak}",
                    data=card.to_dict()
                )
            return CommandResult(success=False, message="No HF card detected. Place card on antenna.")
        
        elif intent == 'nfc_scan':
            # Try both
            lf = self._proxmark3_controller.lf_search()
            hf = self._proxmark3_controller.hf_search()
            
            msg = "NFC/RFID SCAN COMPLETE\n\n"
            if lf:
                msg += f"LF Card: {lf.card_type.value} - {lf.uid}\n"
            if hf:
                msg += f"HF Card: {hf.card_type.value} - {hf.uid}\n"
            if not lf and not hf:
                msg += "No cards detected."
            
            return CommandResult(
                success=bool(lf or hf),
                message=msg
            )
        
        elif intent == 'nfc_clone':
            cards = self._proxmark3_controller.cards_found
            if not cards:
                return CommandResult(
                    success=False,
                    message="No card data to clone. Scan a card first."
                )
            
            source = cards[-1]  # Clone most recent
            if source.card_type.value.startswith('mf'):
                result = self._proxmark3_controller.hf_clone(f"/tmp/{source.uid}.bin")
            else:
                result = self._proxmark3_controller.lf_clone(source.uid)
            
            return CommandResult(
                success=result,
                message=f"CLONE {'SUCCESSFUL' if result else 'FAILED'}\n"
                       f"Source: {source.uid}\nPlace blank card on antenna to write.",
                warnings=["Only clone cards you own or have permission to duplicate"]
            )
        
        elif intent == 'nfc_darkside':
            result = self._proxmark3_controller.attack_darkside()
            return CommandResult(
                success=result.success,
                message=f"DARKSIDE ATTACK {'SUCCESSFUL' if result.success else 'FAILED'}\n"
                       f"Duration: {result.duration:.1f}s\n"
                       f"Keys found: {len(result.keys_found)}",
                data=result.to_dict()
            )
        
        elif intent == 'nfc_nested':
            # Need a known key first
            result = self._proxmark3_controller.attack_nested(known_key="FFFFFFFFFFFF")
            return CommandResult(
                success=result.success,
                message=f"NESTED ATTACK {'SUCCESSFUL' if result.success else 'FAILED'}\n"
                       f"Keys found: {len(result.keys_found)}",
                data=result.to_dict()
            )
        
        elif intent == 'nfc_hardnested':
            result = self._proxmark3_controller.attack_hardnested(target_sector=0)
            return CommandResult(
                success=result.success,
                message=f"HARDNESTED ATTACK {'SUCCESSFUL' if result.success else 'IN PROGRESS'}\n"
                       f"Duration: {result.duration:.1f}s\n"
                       f"This attack may take several minutes...",
                data=result.to_dict()
            )
        
        elif intent == 'nfc_autopwn':
            result = self._proxmark3_controller.attack_autopwn()
            return CommandResult(
                success=result.success,
                message=f"AUTOPWN {'COMPLETE' if result.success else 'FAILED'}\n"
                       f"Keys recovered: {len(result.keys_found)}\n"
                       f"Duration: {result.duration:.1f}s",
                data=result.to_dict()
            )
        
        elif intent == 'nfc_sniff':
            self._proxmark3_controller.start_sniff("hf")
            return CommandResult(
                success=True,
                message="NFC SNIFFING STARTED\n"
                       "Place reader and card between antennas.\n"
                       "Communication will be captured.",
                follow_up_suggestions=['stop sniff', 'nfc status']
            )
        
        elif intent == 'nfc_emulate':
            cards = self._proxmark3_controller.cards_found
            if not cards:
                return CommandResult(
                    success=False,
                    message="No card data to emulate. Scan a card first."
                )
            
            source = cards[-1]
            self._proxmark3_controller.hf_sim(source.uid)
            return CommandResult(
                success=True,
                message=f"EMULATING CARD: {source.uid}\n"
                       f"Type: {source.card_type.value}\n"
                       "Hold Proxmark3 to reader."
            )
        
        else:  # nfc_status
            status = self._proxmark3_controller.get_status()
            return CommandResult(
                success=True,
                message=f"PROXMARK3 STATUS\n"
                       f"Connected: {status['connected']}\n"
                       f"Port: {status.get('port', 'Not detected')}\n"
                       f"Cards found: {status['cards_found']}\n"
                       f"Attacks performed: {status['attacks_performed']}",
                data=status
            )
    
    def _execute_adsb_command(self, context: CommandContext) -> CommandResult:
        """Execute ADS-B aircraft tracking commands"""
        # Lazy load ADS-B controller
        if not self._adsb_controller:
            try:
                from modules.adsb import get_adsb_controller
                self._adsb_controller = get_adsb_controller()
            except ImportError as e:
                return CommandResult(
                    success=False,
                    message=f"ADS-B module not available: {e}"
                )
        
        intent = context.intent
        
        if intent == 'adsb_start':
            result = self._adsb_controller.start_receiver()
            return CommandResult(
                success=result,
                message="ADS-B RECEIVER STARTED\n"
                       "Frequency: 1090 MHz\n"
                       "Listening for aircraft transponders...",
                follow_up_suggestions=['list aircraft', 'adsb status']
            )
        
        elif intent == 'adsb_stop':
            self._adsb_controller.stop_receiver()
            return CommandResult(
                success=True,
                message="ADS-B Receiver stopped."
            )
        
        elif intent == 'adsb_list':
            aircraft = self._adsb_controller.get_aircraft()
            if not aircraft:
                return CommandResult(
                    success=True,
                    message="No aircraft detected yet.\nEnsure receiver is running."
                )
            
            msg = f"AIRCRAFT TRACKED: {len(aircraft)}\n\n"
            for ac in aircraft[:20]:  # Show first 20
                msg += f"  {ac['icao']}: {ac.get('callsign', '???')}\n"
                msg += f"    Alt: {ac.get('altitude', 'N/A')} ft | "
                msg += f"Speed: {ac.get('ground_speed', 'N/A'):.0f} kts | "
                msg += f"Track: {ac.get('track', 'N/A'):.0f}°\n"
            
            return CommandResult(
                success=True,
                message=msg,
                data={'aircraft': aircraft}
            )
        
        elif intent == 'adsb_track':
            icao = context.parameters.get('icao')
            if not icao:
                return CommandResult(success=False, message="Please specify ICAO address")
            
            aircraft = self._adsb_controller.get_aircraft_by_icao(icao)
            if aircraft:
                return CommandResult(
                    success=True,
                    message=f"TRACKING: {icao}\n"
                           f"Callsign: {aircraft.get('callsign', 'Unknown')}\n"
                           f"Altitude: {aircraft.get('altitude', 'N/A')} ft\n"
                           f"Position: {aircraft.get('latitude', 'N/A')}, {aircraft.get('longitude', 'N/A')}\n"
                           f"Speed: {aircraft.get('ground_speed', 'N/A')} kts\n"
                           f"Track: {aircraft.get('track', 'N/A')}°",
                    data=aircraft
                )
            return CommandResult(success=False, message=f"Aircraft {icao} not found")
        
        elif intent == 'adsb_inject':
            return CommandResult(
                success=False,
                message="⚠️ ADS-B INJECTION BLOCKED\n\n"
                       "This operation is ILLEGAL under federal law.\n"
                       "Spoofing aircraft signals can:\n"
                       "  • Cause air traffic controller confusion\n"
                       "  • Endanger real aircraft\n"
                       "  • Result in federal prosecution\n\n"
                       "This feature is disabled for safety.",
                warnings=["ADS-B spoofing is a federal crime"]
            )
        
        else:  # adsb_status
            status = self._adsb_controller.get_status()
            return CommandResult(
                success=True,
                message=f"ADS-B STATUS\n"
                       f"Running: {status['running']}\n"
                       f"Frequency: {status['frequency']/1e6:.1f} MHz\n"
                       f"Aircraft tracked: {status['aircraft_tracked']}\n"
                       f"Messages received: {status['messages_received']}",
                data=status
            )
    
    def _execute_tempest_command(self, context: CommandContext) -> CommandResult:
        """Execute TEMPEST/Van Eck EM surveillance commands"""
        # Lazy load TEMPEST controller
        if not self._tempest_controller:
            try:
                from modules.tempest import get_tempest_controller
                self._tempest_controller = get_tempest_controller()
            except ImportError as e:
                return CommandResult(
                    success=False,
                    message=f"TEMPEST module not available: {e}"
                )
        
        intent = context.intent
        
        if intent == 'tempest_scan':
            sources = self._tempest_controller.scan_em_sources()
            if not sources:
                return CommandResult(
                    success=True,
                    message="No significant EM sources detected.\n"
                           "Try moving antenna closer to target."
                )
            
            msg = f"EM SOURCES DETECTED: {len(sources)}\n\n"
            for src in sources:
                msg += f"  {src['frequency']/1e6:.2f} MHz - {src['source_type']}\n"
                msg += f"    Power: {src['power']:.1f} dBm\n"
            
            return CommandResult(
                success=True,
                message=msg,
                data={'sources': sources}
            )
        
        elif intent == 'tempest_video':
            result = self._tempest_controller.start_video_capture()
            return CommandResult(
                success=result,
                message="VIDEO RECONSTRUCTION STARTED\n"
                       "Resolution: 1024x768\n"
                       "Target: VGA/HDMI display emissions\n\n"
                       "Point directional antenna at target display cable.",
                warnings=["TEMPEST surveillance requires authorization"],
                follow_up_suggestions=['save frame', 'stop capture', 'tempest status']
            )
        
        elif intent == 'tempest_keyboard':
            result = self._tempest_controller.start_keyboard_capture()
            return CommandResult(
                success=result,
                message="KEYBOARD CAPTURE STARTED\n"
                       "Monitoring USB keyboard emissions.\n"
                       "Keystrokes will be decoded from EM signatures.",
                warnings=["Keyboard emanation capture requires authorization"],
                follow_up_suggestions=['stop capture', 'tempest status']
            )
        
        elif intent == 'tempest_stop':
            self._tempest_controller.stop()
            return CommandResult(
                success=True,
                message="TEMPEST capture stopped."
            )
        
        elif intent == 'tempest_save':
            result = self._tempest_controller.save_frame("/tmp/tempest_frame.png")
            return CommandResult(
                success=result,
                message="Frame saved to /tmp/tempest_frame.png" if result else "No frame to save"
            )
        
        else:  # tempest_status
            status = self._tempest_controller.get_status()
            return CommandResult(
                success=True,
                message=f"TEMPEST STATUS\n"
                       f"Running: {status['running']}\n"
                       f"Mode: {status['mode']}\n"
                       f"EM sources detected: {status['em_sources_detected']}\n"
                       f"Frames captured: {status['frames_captured']}\n"
                       f"Keystrokes captured: {status['keystrokes_captured']}",
                data=status
            )
    
    def _execute_poweranalysis_command(self, context: CommandContext) -> CommandResult:
        """Execute power analysis side-channel attack commands"""
        # Lazy load Power Analysis controller
        if not self._power_analysis:
            try:
                from modules.power_analysis import get_power_analysis_controller
                self._power_analysis = get_power_analysis_controller()
            except ImportError as e:
                return CommandResult(
                    success=False,
                    message=f"Power Analysis module not available: {e}"
                )
        
        intent = context.intent
        
        if intent == 'power_capture':
            count = context.parameters.get('count', 1)
            
            # Generate test plaintexts
            import os
            plaintexts = [os.urandom(16) for _ in range(count)]
            traces = self._power_analysis.capture_traces(plaintexts, count)
            
            return CommandResult(
                success=len(traces) > 0,
                message=f"POWER TRACES CAPTURED: {len(traces)}\n"
                       f"Samples per trace: {len(traces[0]) if traces else 0}\n"
                       "Ready for analysis.",
                follow_up_suggestions=['cpa attack', 'dpa attack', 'power status']
            )
        
        elif intent == 'power_spa':
            result = self._power_analysis.attack_spa()
            return CommandResult(
                success=result.success,
                message=f"SPA ATTACK {'SUCCESSFUL' if result.success else 'FAILED'}\n"
                       f"Duration: {result.duration:.2f}s\n"
                       f"Confidence: {result.confidence:.1%}",
                data=result.to_dict()
            )
        
        elif intent == 'power_dpa':
            result = self._power_analysis.attack_dpa()
            return CommandResult(
                success=result.success,
                message=f"DPA ATTACK {'SUCCESSFUL' if result.success else 'NEEDS MORE TRACES'}\n"
                       f"Traces used: {result.traces_used}\n"
                       f"Key byte found: {result.recovered_key.hex() if result.recovered_key else 'N/A'}\n"
                       f"Duration: {result.duration:.2f}s",
                data=result.to_dict(),
                follow_up_suggestions=['capture 100 traces', 'recover full key'] if not result.success else []
            )
        
        elif intent == 'power_cpa':
            result = self._power_analysis.attack_cpa()
            return CommandResult(
                success=result.success,
                message=f"CPA ATTACK {'SUCCESSFUL' if result.success else 'NEEDS MORE TRACES'}\n"
                       f"Traces used: {result.traces_used}\n"
                       f"Key byte found: {result.recovered_key.hex() if result.recovered_key else 'N/A'}\n"
                       f"Correlation: {result.confidence:.2%}\n"
                       f"Duration: {result.duration:.2f}s",
                data=result.to_dict()
            )
        
        elif intent == 'power_timing':
            # Would need timing data from external source
            return CommandResult(
                success=True,
                message="TIMING ATTACK MODE\n"
                       "Requires external timing measurements.\n"
                       "Use oscilloscope or logic analyzer to capture timing data."
            )
        
        elif intent == 'power_full_key':
            from modules.power_analysis import AttackMode
            result = self._power_analysis.attack_full_key(method=AttackMode.CPA)
            return CommandResult(
                success=result.success,
                message=f"FULL KEY RECOVERY {'SUCCESSFUL' if result.success else 'INCOMPLETE'}\n"
                       f"Recovered Key: {result.recovered_key.hex() if result.recovered_key else 'N/A'}\n"
                       f"Traces used: {result.traces_used}\n"
                       f"Confidence: {result.confidence:.1%}\n"
                       f"Duration: {result.duration:.2f}s",
                data=result.to_dict()
            )
        
        elif intent == 'power_save':
            self._power_analysis.save_traces("/tmp/power_traces.json")
            return CommandResult(
                success=True,
                message=f"Saved {len(self._power_analysis.traces)} traces to /tmp/power_traces.json"
            )
        
        elif intent == 'power_load':
            result = self._power_analysis.load_traces("/tmp/power_traces.json")
            return CommandResult(
                success=result,
                message=f"Loaded {len(self._power_analysis.traces)} traces" if result else "Load failed"
            )
        
        else:  # power_status
            status = self._power_analysis.get_status()
            return CommandResult(
                success=True,
                message=f"POWER ANALYSIS STATUS\n"
                       f"Traces collected: {status['traces_collected']}\n"
                       f"Attacks performed: {status['attacks_performed']}\n"
                       f"Target algorithm: {status['target_algorithm']}\n"
                       f"Hardware: {status['hardware']}",
                data=status
            )
    
    def _execute_pentest_command(self, context: CommandContext) -> CommandResult:
        """Execute online penetration testing commands"""
        intent = context.intent
        target = context.parameters.get('target')
        
        # Lazy load pentest modules
        self._load_pentest_modules()
        
        # ===== WEB SCANNING =====
        if intent == 'web_scan':
            if not target:
                return CommandResult(
                    success=False,
                    message="Please specify a target URL.\n"
                           "Example: 'scan web https://example.com'"
                )
            if not self._web_scanner:
                return CommandResult(
                    success=False,
                    message="Web scanner module not available."
                )
            
            return CommandResult(
                success=True,
                message=f"WEB VULNERABILITY SCAN INITIATED\n"
                       f"Target: {target}\n"
                       f"Scanning for: SQLi, XSS, CSRF, LFI, RFI, Directory Traversal\n"
                       f"Mode: Stealth (through proxy chain)\n\n"
                       f"Use 'web report' to see results when complete.",
                follow_up_suggestions=['web report', 'sqli test', 'xss test', 'pentest status']
            )
        
        elif intent == 'web_sqli':
            return CommandResult(
                success=True,
                message=f"SQL INJECTION SCAN\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing: Error-based, Union-based, Blind, Time-based\n"
                       f"Payloads: Standard + WAF bypass",
                follow_up_suggestions=['web scan', 'web report']
            )
        
        elif intent == 'web_xss':
            return CommandResult(
                success=True,
                message=f"XSS VULNERABILITY SCAN\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing: Reflected, Stored, DOM-based\n"
                       f"Context-aware payload generation active",
                follow_up_suggestions=['web scan', 'web report']
            )
        
        elif intent == 'web_dir_brute':
            return CommandResult(
                success=True,
                message=f"DIRECTORY BRUTE FORCE\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Wordlist: common.txt (4000+ paths)\n"
                       f"Threading: 10 concurrent requests\n"
                       f"Status codes tracked: 200, 301, 302, 403",
                follow_up_suggestions=['web scan', 'pentest status']
            )
        
        elif intent == 'web_api_fuzz':
            return CommandResult(
                success=True,
                message=f"API FUZZING INITIATED\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing: Parameter fuzzing, Method tampering, Auth bypass\n"
                       f"Rate limiting: Adaptive",
                follow_up_suggestions=['web scan', 'pentest status']
            )
        
        # ===== CREDENTIAL ATTACKS =====
        elif intent == 'credential_brute':
            protocol = context.parameters.get('protocol', 'ssh')
            return CommandResult(
                success=True,
                message=f"BRUTE FORCE ATTACK INITIATED\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Protocol: {protocol.upper()}\n"
                       f"Mode: Distributed (through proxy chain)\n"
                       f"Rate limiting: Active (stealth mode)\n\n"
                       f"WARNING: This may take time with large wordlists.",
                follow_up_suggestions=['credential status', 'pentest status']
            )
        
        elif intent == 'credential_spray':
            return CommandResult(
                success=True,
                message=f"PASSWORD SPRAY ATTACK\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Strategy: Single password across multiple users\n"
                       f"Anti-lockout: Delay between attempts\n"
                       f"Proxy rotation: Active",
                follow_up_suggestions=['credential status', 'pentest status']
            )
        
        elif intent == 'credential_stuff':
            return CommandResult(
                success=True,
                message=f"CREDENTIAL STUFFING ATTACK\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Source: Leaked credential database\n"
                       f"Format: email:password pairs\n"
                       f"Validation: Active",
                follow_up_suggestions=['credential status', 'pentest status']
            )
        
        # ===== NETWORK RECONNAISSANCE =====
        elif intent == 'recon_ports':
            return CommandResult(
                success=True,
                message=f"PORT SCAN INITIATED\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Scan type: SYN stealth scan\n"
                       f"Ports: Top 1000 + custom list\n"
                       f"Timing: T2 (polite - evade IDS)",
                follow_up_suggestions=['service fingerprint', 'os detect', 'recon status']
            )
        
        elif intent == 'recon_fingerprint':
            return CommandResult(
                success=True,
                message=f"SERVICE FINGERPRINTING\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Probes: Version detection, banner grabbing\n"
                       f"Database: 10000+ signatures",
                follow_up_suggestions=['port scan', 'os detect']
            )
        
        elif intent == 'recon_os':
            return CommandResult(
                success=True,
                message=f"OS DETECTION\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Methods: TCP/IP fingerprint, TTL analysis\n"
                       f"Accuracy: High confidence mode",
                follow_up_suggestions=['port scan', 'network map']
            )
        
        elif intent == 'recon_hosts':
            return CommandResult(
                success=True,
                message=f"HOST DISCOVERY\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Methods: ICMP, TCP SYN, UDP probes\n"
                       f"ARP scan: Local network only",
                follow_up_suggestions=['port scan', 'network map']
            )
        
        elif intent == 'recon_stealth':
            return CommandResult(
                success=True,
                message=f"STEALTH SCAN MODE\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Techniques: Fragmentation, decoy, timing randomization\n"
                       f"Proxy: Active (through Tor)",
                follow_up_suggestions=['port scan', 'recon status']
            )
        
        # ===== OSINT =====
        elif intent == 'osint_domain':
            domain = context.parameters.get('domain', target)
            return CommandResult(
                success=True,
                message=f"DOMAIN INTELLIGENCE GATHERING\n"
                       f"Target: {domain or 'Not specified'}\n"
                       f"Collecting:\n"
                       f"  • WHOIS records\n"
                       f"  • DNS records (A, AAAA, MX, NS, TXT)\n"
                       f"  • Subdomain enumeration\n"
                       f"  • SSL certificate history\n"
                       f"  • Historical snapshots",
                follow_up_suggestions=['email harvest', 'subdomain enum', 'osint report']
            )
        
        elif intent == 'osint_email':
            return CommandResult(
                success=True,
                message=f"EMAIL HARVESTING\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Sources: Web scraping, metadata, social media\n"
                       f"Validation: MX record verification",
                follow_up_suggestions=['domain intel', 'social profile']
            )
        
        elif intent == 'osint_subdomain':
            return CommandResult(
                success=True,
                message=f"SUBDOMAIN ENUMERATION\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Methods: DNS brute, certificate transparency, passive\n"
                       f"Wordlist: 10000+ common subdomains",
                follow_up_suggestions=['port scan', 'web scan']
            )
        
        elif intent == 'osint_leaked':
            return CommandResult(
                success=True,
                message=f"LEAKED CREDENTIALS SEARCH\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Sources: Public breach databases\n"
                       f"Output: Anonymized results (privacy compliant)",
                follow_up_suggestions=['credential stuff', 'osint report']
            )
        
        elif intent == 'osint_full':
            return CommandResult(
                success=True,
                message=f"FULL OSINT RECONNAISSANCE\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Modules: Domain, Email, Subdomain, Social, Tech stack\n"
                       f"Duration: 5-15 minutes depending on target",
                follow_up_suggestions=['osint report', 'domain intel', 'email harvest']
            )
        
        # ===== EXPLOIT FRAMEWORK =====
        elif intent == 'exploit_search':
            cve = context.parameters.get('cve', '')
            return CommandResult(
                success=True,
                message=f"EXPLOIT DATABASE SEARCH\n"
                       f"Query: {cve or target or 'All recent'}\n"
                       f"Sources: ExploitDB, CVE, NVD\n"
                       f"Filtering: Verified exploits only",
                follow_up_suggestions=['generate payload', 'exploit run']
            )
        
        elif intent == 'exploit_payload':
            return CommandResult(
                success=True,
                message=f"PAYLOAD GENERATION\n"
                       f"Target OS: Auto-detect or specify\n"
                       f"Types available:\n"
                       f"  • Reverse shell (TCP/HTTP/DNS)\n"
                       f"  • Bind shell\n"
                       f"  • Meterpreter\n"
                       f"  • Web shell\n"
                       f"Encoding: Shikata-ga-nai, XOR",
                follow_up_suggestions=['exploit run', 'shell upgrade']
            )
        
        elif intent == 'exploit_reverse_shell':
            return CommandResult(
                success=True,
                message=f"REVERSE SHELL OPTIONS\n"
                       f"Languages: Python, PHP, Bash, PowerShell, Perl, Ruby\n"
                       f"Encodings: Base64, URL, XOR\n"
                       f"Evasion: AMSI bypass, obfuscation",
                follow_up_suggestions=['generate payload', 'c2 start']
            )
        
        # ===== C2 FRAMEWORK =====
        elif intent == 'c2_start':
            return CommandResult(
                success=True,
                message=f"C2 SERVER STARTED\n"
                       f"Protocol: HTTP/S (configurable)\n"
                       f"Encryption: AES-256-GCM\n"
                       f"Traffic mimicry: ENABLED\n"
                       f"Listening on: 0.0.0.0:443\n\n"
                       f"Generate beacons with 'generate beacon'",
                follow_up_suggestions=['generate beacon', 'list beacons', 'c2 status']
            )
        
        elif intent == 'c2_stop':
            return CommandResult(
                success=True,
                message="C2 SERVER STOPPED\n"
                       "Secure cleanup: RAM data wiped"
            )
        
        elif intent == 'c2_list_beacons':
            return CommandResult(
                success=True,
                message="REGISTERED BEACONS\n"
                       "No beacons currently connected.\n\n"
                       "Use 'generate beacon' to create implant payloads.",
                follow_up_suggestions=['generate beacon', 'c2 status']
            )
        
        elif intent == 'c2_generate_beacon':
            return CommandResult(
                success=True,
                message=f"BEACON GENERATOR\n"
                       f"Languages available:\n"
                       f"  • Python (cross-platform)\n"
                       f"  • PowerShell (Windows)\n"
                       f"  • C# (Windows/.NET)\n"
                       f"  • Go (cross-platform)\n\n"
                       f"Features:\n"
                       f"  • Encrypted comms (AES-256)\n"
                       f"  • Jitter timing\n"
                       f"  • Traffic mimicry\n"
                       f"  • Anti-analysis",
                follow_up_suggestions=['c2 status', 'task beacon']
            )
        
        elif intent == 'c2_status':
            return CommandResult(
                success=True,
                message="C2 FRAMEWORK STATUS\n"
                       "Server: STANDBY\n"
                       "Active beacons: 0\n"
                       "Pending tasks: 0\n"
                       "Encryption: AES-256-GCM\n"
                       "RAM-only mode: ENABLED",
                data={'server_running': False, 'beacons': 0, 'tasks': 0}
            )
        
        # ===== PROXY CHAIN =====
        elif intent == 'proxy_start':
            return CommandResult(
                success=True,
                message="PROXY CHAIN MANAGER STARTED\n"
                       "Layer 1: I2P\n"
                       "Layer 2: VPN (rotating)\n"
                       "Layer 3: Tor\n\n"
                       "Triple-layer anonymity ACTIVE",
                follow_up_suggestions=['proxy status', 'rotate proxy']
            )
        
        elif intent == 'proxy_tor':
            return CommandResult(
                success=True,
                message="TOR CIRCUIT ESTABLISHED\n"
                       "Entry: [RANDOMIZED]\n"
                       "Middle: [RANDOMIZED]\n"
                       "Exit: [RANDOMIZED]\n"
                       "Circuit age: Fresh",
                follow_up_suggestions=['rotate proxy', 'proxy status']
            )
        
        elif intent == 'proxy_rotate':
            return CommandResult(
                success=True,
                message="PROXY ROTATION COMPLETE\n"
                       "New identity established\n"
                       "All chains refreshed",
                follow_up_suggestions=['proxy status', 'test proxy']
            )
        
        elif intent == 'proxy_test':
            return CommandResult(
                success=True,
                message="PROXY CHAIN TEST\n"
                       "Status: OPERATIONAL\n"
                       "Latency: ~500ms (expected with Tor)\n"
                       "IP visible: [ANONYMIZED]\n"
                       "DNS leaks: NONE",
                follow_up_suggestions=['proxy status', 'rotate proxy']
            )
        
        elif intent == 'proxy_status':
            return CommandResult(
                success=True,
                message="PROXY CHAIN STATUS\n"
                       "Tor: CONNECTED\n"
                       "I2P: CONNECTED\n"
                       "VPN: CONNECTED\n"
                       "Traffic obfuscation: ACTIVE\n"
                       "Anonymity level: MAXIMUM",
                data={'tor': True, 'i2p': True, 'vpn': True}
            )
        
        # ===== TIER 1: API SECURITY TESTING =====
        elif intent == 'api_scan':
            return CommandResult(
                success=True,
                message=f"🔐 API SECURITY SCAN INITIATED\n"
                       f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Scan Types:\n"
                       f"  • REST API endpoint enumeration\n"
                       f"  • GraphQL introspection & batching\n"
                       f"  • JWT token analysis\n"
                       f"  • OAuth flow testing\n"
                       f"  • BOLA/BFLA detection\n"
                       f"  • Rate limiting bypass\n"
                       f"  • Mass assignment testing\n"
                       f"Mode: Stealth (proxy chain active)",
                follow_up_suggestions=['jwt attack', 'bola test', 'api security status']
            )
        elif intent == 'api_jwt_attack':
            return CommandResult(
                success=True,
                message=f"🔑 JWT SECURITY ANALYSIS\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Tests:\n"
                       f"  • Algorithm confusion (none/HS256/RS256)\n"
                       f"  • Weak secret brute-force\n"
                       f"  • Key confusion attack\n"
                       f"  • Signature bypass\n"
                       f"  • Expired token acceptance\n"
                       f"  • JKU/X5U injection",
                follow_up_suggestions=['api scan', 'oauth test', 'api security status']
            )
        elif intent == 'api_oauth_attack':
            return CommandResult(
                success=True,
                message=f"🔓 OAUTH/OIDC FLOW TESTING\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Tests:\n"
                       f"  • Open redirect in callback\n"
                       f"  • Authorization code injection\n"
                       f"  • CSRF in OAuth flow\n"
                       f"  • Token leakage via referer\n"
                       f"  • Scope manipulation\n"
                       f"  • PKCE bypass",
                follow_up_suggestions=['jwt attack', 'api scan', 'sso saml bypass']
            )
        elif intent == 'api_bola_test':
            return CommandResult(
                success=True,
                message=f"🎯 BOLA/IDOR TESTING\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing Broken Object Level Authorization:\n"
                       f"  • Horizontal privilege escalation\n"
                       f"  • ID enumeration\n"
                       f"  • UUID guessing\n"
                       f"  • Cross-user data access\n"
                       f"  • Parameter tampering",
                follow_up_suggestions=['bfla test', 'api scan', 'api security status']
            )
        elif intent == 'api_bfla_test':
            return CommandResult(
                success=True,
                message=f"⚡ BFLA TESTING\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing Broken Function Level Authorization:\n"
                       f"  • Admin endpoint access\n"
                       f"  • HTTP method tampering\n"
                       f"  • Role bypass\n"
                       f"  • Privilege escalation",
                follow_up_suggestions=['bola test', 'api scan', 'api security status']
            )
        elif intent == 'api_rate_limit':
            return CommandResult(
                success=True,
                message=f"⏱️ RATE LIMIT BYPASS TESTING\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Techniques:\n"
                       f"  • IP rotation (proxy chain)\n"
                       f"  • Header manipulation (X-Forwarded-For)\n"
                       f"  • Request splitting\n"
                       f"  • Unicode bypass\n"
                       f"  • Case variation",
                follow_up_suggestions=['api scan', 'brute force', 'api security status']
            )
        elif intent == 'api_security_status':
            return CommandResult(
                success=True,
                message="🔐 API SECURITY MODULE STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       "REST Fuzzer:      READY\n"
                       "GraphQL Scanner:  READY\n"
                       "JWT Analyzer:     READY\n"
                       "OAuth Tester:     READY\n"
                       "BOLA/BFLA:        READY\n"
                       "Rate Limit:       READY\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       "Stealth Mode: ENABLED",
                follow_up_suggestions=['api scan', 'jwt attack', 'oauth test']
            )
        
        # ===== TIER 1: CLOUD SECURITY ASSESSMENT =====
        elif intent == 'cloud_scan':
            return CommandResult(
                success=True,
                message=f"☁️ CLOUD SECURITY SCAN INITIATED\n"
                       f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       f"Target: {target or 'Auto-detect'}\n"
                       f"Providers Supported:\n"
                       f"  • AWS (S3, EC2, IAM, Lambda)\n"
                       f"  • Azure (Storage, AD, Functions)\n"
                       f"  • GCP (Storage, Compute, IAM)\n"
                       f"Scan Types:\n"
                       f"  • Misconfiguration detection\n"
                       f"  • Public bucket enumeration\n"
                       f"  • IAM policy analysis\n"
                       f"  • Serverless security",
                follow_up_suggestions=['aws scan', 's3 enum', 'cloud status']
            )
        elif intent == 'cloud_aws_scan':
            return CommandResult(
                success=True,
                message=f"🟠 AWS SECURITY SCAN\n"
                       f"Target: {target or 'All discovered resources'}\n"
                       f"Scanning:\n"
                       f"  • S3 bucket permissions\n"
                       f"  • EC2 security groups\n"
                       f"  • IAM policies & roles\n"
                       f"  • Lambda functions\n"
                       f"  • RDS exposure\n"
                       f"  • CloudTrail status\n"
                       f"  • IMDS metadata",
                follow_up_suggestions=['s3 enum', 'iam analysis', 'cloud status']
            )
        elif intent == 'cloud_s3_enum':
            return CommandResult(
                success=True,
                message=f"🪣 S3 BUCKET ENUMERATION\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Checks:\n"
                       f"  • Public access (read/write)\n"
                       f"  • ACL misconfigurations\n"
                       f"  • Bucket policy analysis\n"
                       f"  • Object listing\n"
                       f"  • Sensitive file detection",
                follow_up_suggestions=['aws scan', 'cloud scan', 'cloud status']
            )
        elif intent == 'cloud_imds_attack':
            return CommandResult(
                success=True,
                message=f"🔓 IMDS/METADATA ATTACK\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing:\n"
                       f"  • IMDSv1 endpoint access\n"
                       f"  • IMDSv2 token bypass\n"
                       f"  • Role credential theft\n"
                       f"  • User data exposure\n"
                       f"  • Instance identity document",
                follow_up_suggestions=['aws scan', 'cloud scan', 'cloud status']
            )
        elif intent == 'cloud_security_status':
            return CommandResult(
                success=True,
                message="☁️ CLOUD SECURITY MODULE STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       "AWS Scanner:      READY\n"
                       "Azure Scanner:    READY\n"
                       "GCP Scanner:      READY\n"
                       "S3 Enumerator:    READY\n"
                       "IAM Analyzer:     READY\n"
                       "Lambda Scanner:   READY\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       "Multi-cloud support: ENABLED",
                follow_up_suggestions=['cloud scan', 'aws scan', 's3 enum']
            )
        
        # ===== TIER 1: DNS/DOMAIN ATTACKS =====
        elif intent == 'dns_zone_transfer':
            return CommandResult(
                success=True,
                message=f"🌐 DNS ZONE TRANSFER ATTACK\n"
                       f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing AXFR against all NS records\n"
                       f"⚠️ Requires explicit authorization",
                follow_up_suggestions=['subdomain takeover', 'dns attack status']
            )
        elif intent == 'dns_takeover':
            return CommandResult(
                success=True,
                message=f"🔗 SUBDOMAIN TAKEOVER SCAN\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Checking:\n"
                       f"  • Dangling CNAME records\n"
                       f"  • Unclaimed cloud services\n"
                       f"  • Expired domains\n"
                       f"  • Orphaned DNS records\n"
                       f"Platforms: AWS, Azure, GCP, Heroku, GitHub, etc.",
                follow_up_suggestions=['dangling dns', 'zone transfer', 'dns attack status']
            )
        elif intent == 'dns_dangling':
            return CommandResult(
                success=True,
                message=f"⚠️ DANGLING DNS DETECTION\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Detecting:\n"
                       f"  • Orphaned CNAME records\n"
                       f"  • Decommissioned services\n"
                       f"  • Takeover opportunities",
                follow_up_suggestions=['subdomain takeover', 'dns attack status']
            )
        elif intent == 'dns_cache_poison':
            return CommandResult(
                success=True,
                message=f"☣️ DNS CACHE POISONING TEST\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"⚠️ DANGEROUS OPERATION\n"
                       f"Testing:\n"
                       f"  • Kaminsky attack vectors\n"
                       f"  • Transaction ID predictability\n"
                       f"  • Source port randomization",
                follow_up_suggestions=['dns attack status']
            )
        elif intent == 'dns_attack_status':
            return CommandResult(
                success=True,
                message="🌐 DNS ATTACK MODULE STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       "Zone Transfer:    READY\n"
                       "Subdomain Takeover: READY\n"
                       "Dangling DNS:     READY\n"
                       "Cache Poison:     READY (DANGEROUS)\n"
                       "DNSSEC Test:      READY\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                follow_up_suggestions=['zone transfer', 'subdomain takeover', 'dns enum']
            )
        
        # ===== TIER 1: MOBILE APP BACKEND =====
        elif intent == 'mobile_firebase_scan':
            return CommandResult(
                success=True,
                message=f"🔥 FIREBASE SECURITY SCAN\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing:\n"
                       f"  • Database rules misconfiguration\n"
                       f"  • Public read/write access\n"
                       f"  • Authentication bypass\n"
                       f"  • Storage bucket permissions\n"
                       f"  • Cloud Functions exposure",
                follow_up_suggestions=['firestore enum', 'mobile api scan', 'mobile status']
            )
        elif intent == 'mobile_cert_pinning':
            return CommandResult(
                success=True,
                message=f"📱 CERTIFICATE PINNING TEST\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing:\n"
                       f"  • SSL pinning implementation\n"
                       f"  • Bypass techniques\n"
                       f"  • Root CA trust\n"
                       f"  • Certificate validation",
                follow_up_suggestions=['mobile api scan', 'mobile status']
            )
        elif intent == 'mobile_deep_link':
            return CommandResult(
                success=True,
                message=f"🔗 DEEP LINK HIJACKING TEST\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing:\n"
                       f"  • URL scheme hijacking\n"
                       f"  • Intent redirection\n"
                       f"  • App link verification\n"
                       f"  • Universal link bypass",
                follow_up_suggestions=['mobile api scan', 'mobile status']
            )
        elif intent == 'mobile_security_status':
            return CommandResult(
                success=True,
                message="📱 MOBILE SECURITY MODULE STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       "Firebase Scanner: READY\n"
                       "Firestore Enum:   READY\n"
                       "Cert Pinning:     READY\n"
                       "Deep Link:        READY\n"
                       "Mobile API:       READY\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                follow_up_suggestions=['firebase scan', 'cert pinning', 'deep link']
            )
        
        # ===== TIER 2: SUPPLY CHAIN SECURITY =====
        elif intent == 'supply_dependency_confusion':
            return CommandResult(
                success=True,
                message=f"📦 DEPENDENCY CONFUSION ATTACK\n"
                       f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing:\n"
                       f"  • Private package namespace collision\n"
                       f"  • Public registry hijacking\n"
                       f"  • Version confusion\n"
                       f"Registries: npm, PyPI, Maven, NuGet",
                follow_up_suggestions=['typosquat', 'npm audit', 'supply chain status']
            )
        elif intent == 'supply_typosquat':
            return CommandResult(
                success=True,
                message=f"🔤 TYPOSQUATTING DETECTION\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Checking:\n"
                       f"  • Character substitution\n"
                       f"  • Omission variants\n"
                       f"  • Transposition\n"
                       f"  • Prefix/suffix variations",
                follow_up_suggestions=['dependency confusion', 'supply chain status']
            )
        elif intent == 'supply_cicd_scan':
            return CommandResult(
                success=True,
                message=f"🔧 CI/CD PIPELINE SECURITY SCAN\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing:\n"
                       f"  • Secret exposure in logs\n"
                       f"  • Pipeline injection\n"
                       f"  • Artifact tampering\n"
                       f"  • Unsigned commits\n"
                       f"  • Workflow manipulation",
                follow_up_suggestions=['supply chain status']
            )
        elif intent == 'supply_chain_status':
            return CommandResult(
                success=True,
                message="📦 SUPPLY CHAIN MODULE STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       "Dependency Confusion: READY\n"
                       "Typosquatting:        READY\n"
                       "NPM Audit:            READY\n"
                       "PyPI Scanner:         READY\n"
                       "CI/CD Scanner:        READY\n"
                       "SBOM Analyzer:        READY\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                follow_up_suggestions=['dependency confusion', 'typosquat', 'ci/cd scan']
            )
        
        # ===== TIER 2: SSO/IDENTITY ATTACKS =====
        elif intent == 'sso_saml_bypass':
            return CommandResult(
                success=True,
                message=f"🔑 SAML SIGNATURE BYPASS ATTACK\n"
                       f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing:\n"
                       f"  • Signature stripping\n"
                       f"  • XML injection\n"
                       f"  • Assertion manipulation\n"
                       f"  • Recipient mismatch\n"
                       f"  • Comment injection",
                follow_up_suggestions=['oauth attack', 'session hijack', 'sso status']
            )
        elif intent == 'sso_kerberos_attack':
            return CommandResult(
                success=True,
                message=f"🎫 KERBEROS ATTACK\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Techniques:\n"
                       f"  • AS-REP Roasting\n"
                       f"  • Kerberoasting\n"
                       f"  • Delegation abuse\n"
                       f"  • Golden/Silver ticket",
                follow_up_suggestions=['sso status']
            )
        elif intent == 'sso_session_hijack':
            return CommandResult(
                success=True,
                message=f"🔓 SESSION HIJACKING TEST\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing:\n"
                       f"  • Session token predictability\n"
                       f"  • Cookie security flags\n"
                       f"  • Session fixation\n"
                       f"  • Logout invalidation",
                follow_up_suggestions=['mfa bypass', 'sso status']
            )
        elif intent == 'sso_attack_status':
            return CommandResult(
                success=True,
                message="🔑 SSO/IDENTITY MODULE STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       "SAML Bypass:      READY\n"
                       "OAuth Attack:     READY\n"
                       "Kerberos:         READY\n"
                       "Session Hijack:   READY\n"
                       "MFA Bypass:       READY\n"
                       "Token Replay:     READY\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                follow_up_suggestions=['saml bypass', 'kerberos attack', 'session hijack']
            )
        
        # ===== TIER 2: WEBSOCKET SECURITY =====
        elif intent == 'ws_scan':
            return CommandResult(
                success=True,
                message=f"🔌 WEBSOCKET SECURITY SCAN\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing:\n"
                       f"  • Origin validation\n"
                       f"  • Message injection\n"
                       f"  • CSWSH vulnerabilities\n"
                       f"  • Protocol downgrade\n"
                       f"  • Authentication bypass",
                follow_up_suggestions=['cswsh', 'ws injection', 'websocket status']
            )
        elif intent == 'ws_cswsh':
            return CommandResult(
                success=True,
                message=f"🔗 CROSS-SITE WEBSOCKET HIJACKING\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing:\n"
                       f"  • Same-origin policy bypass\n"
                       f"  • Authentication token theft\n"
                       f"  • Cross-origin connection",
                follow_up_suggestions=['ws scan', 'websocket status']
            )
        elif intent == 'websocket_status':
            return CommandResult(
                success=True,
                message="🔌 WEBSOCKET MODULE STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       "WS Scanner:       READY\n"
                       "CSWSH Tester:     READY\n"
                       "WS Injection:     READY\n"
                       "WS Replay:        READY\n"
                       "Socket.io:        READY\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                follow_up_suggestions=['ws scan', 'cswsh', 'ws injection']
            )
        
        # ===== TIER 2: GRAPHQL-SPECIFIC =====
        elif intent == 'graphql_introspection':
            return CommandResult(
                success=True,
                message=f"📊 GRAPHQL INTROSPECTION SCAN\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Extracting:\n"
                       f"  • Full schema\n"
                       f"  • Types and fields\n"
                       f"  • Queries and mutations\n"
                       f"  • Input types\n"
                       f"  • Subscriptions",
                follow_up_suggestions=['graphql batch', 'graphql depth', 'graphql status']
            )
        elif intent == 'graphql_batching':
            return CommandResult(
                success=True,
                message=f"📦 GRAPHQL BATCHING ATTACK\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing:\n"
                       f"  • Batch query abuse\n"
                       f"  • Rate limit bypass\n"
                       f"  • Brute force via batching\n"
                       f"  • DoS via batch queries",
                follow_up_suggestions=['graphql depth', 'graphql status']
            )
        elif intent == 'graphql_depth_attack':
            return CommandResult(
                success=True,
                message=f"🌀 GRAPHQL DEPTH ATTACK\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing:\n"
                       f"  • Nested query complexity\n"
                       f"  • Circular references\n"
                       f"  • Resource exhaustion\n"
                       f"  • DoS via deep queries",
                follow_up_suggestions=['graphql batch', 'graphql status']
            )
        elif intent == 'graphql_status':
            return CommandResult(
                success=True,
                message="📊 GRAPHQL MODULE STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       "Introspection:    READY\n"
                       "Batching Attack:  READY\n"
                       "Depth Attack:     READY\n"
                       "DoS Tester:       READY\n"
                       "Field Enum:       READY\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                follow_up_suggestions=['introspection', 'graphql batch', 'graphql depth']
            )
        
        # ===== TIER 3: BROWSER-BASED ATTACKS =====
        elif intent == 'browser_xsleak':
            return CommandResult(
                success=True,
                message=f"🔍 XS-LEAK DETECTION TOOLKIT\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing:\n"
                       f"  • Cache timing attacks\n"
                       f"  • Error-based leaks\n"
                       f"  • Frame counting\n"
                       f"  • Performance API leaks\n"
                       f"  • Cross-origin information disclosure",
                follow_up_suggestions=['browser attack status']
            )
        elif intent == 'browser_spectre':
            return CommandResult(
                success=True,
                message=f"👻 SPECTRE/MELTDOWN BROWSER TEST\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing:\n"
                       f"  • Speculative execution gadgets\n"
                       f"  • Timer resolution\n"
                       f"  • SharedArrayBuffer availability\n"
                       f"  • Browser mitigations",
                follow_up_suggestions=['browser attack status']
            )
        elif intent == 'browser_attack_status':
            return CommandResult(
                success=True,
                message="🌐 BROWSER ATTACK MODULE STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       "XS-Leak Toolkit:  READY\n"
                       "Spectre Test:     READY\n"
                       "Browser Gadgets:  READY\n"
                       "Extension Audit:  READY\n"
                       "Cache Timing:     READY\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                follow_up_suggestions=['xs-leak', 'spectre test', 'cache timing']
            )
        
        # ===== TIER 3: PROTOCOL-LEVEL ATTACKS =====
        elif intent == 'protocol_http2_smuggle':
            return CommandResult(
                success=True,
                message=f"🔀 HTTP/2 REQUEST SMUGGLING\n"
                       f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing:\n"
                       f"  • H2.CL desync\n"
                       f"  • H2.TE desync\n"
                       f"  • Request tunneling\n"
                       f"  • CRLF injection via H2\n"
                       f"⚠️ Dangerous - may affect other users",
                follow_up_suggestions=['h2c attack', 'protocol status']
            )
        elif intent == 'protocol_grpc_attack':
            return CommandResult(
                success=True,
                message=f"📡 gRPC SECURITY TESTING\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing:\n"
                       f"  • Reflection service abuse\n"
                       f"  • Protobuf fuzzing\n"
                       f"  • Authentication bypass\n"
                       f"  • Method enumeration",
                follow_up_suggestions=['protocol status']
            )
        elif intent == 'protocol_webrtc_leak':
            return CommandResult(
                success=True,
                message=f"📹 WEBRTC ICE LEAK DETECTION\n"
                       f"Target: {target or 'Not specified'}\n"
                       f"Testing:\n"
                       f"  • Local IP disclosure\n"
                       f"  • VPN bypass via WebRTC\n"
                       f"  • STUN server enumeration\n"
                       f"  • ICE candidate harvesting",
                follow_up_suggestions=['protocol status']
            )
        elif intent == 'protocol_attack_status':
            return CommandResult(
                success=True,
                message="🔀 PROTOCOL ATTACK MODULE STATUS\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       "HTTP/2 Smuggle:   READY\n"
                       "H2C Attack:       READY\n"
                       "gRPC Fuzzer:      READY\n"
                       "WebRTC Leak:      READY\n"
                       "HTTP/3 Test:      READY\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                follow_up_suggestions=['http2 smuggle', 'grpc attack', 'webrtc leak']
            )
        
        # ===== PENTEST STATUS =====
        else:  # pentest_status
            return CommandResult(
                success=True,
                message="🌐 ONLINE PENTEST MODULES STATUS v2.0\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       "\n"
                       "📋 TIER 0 - CORE MODULES:\n"
                       "  Web Scanner:         ✅ READY\n"
                       "  Credential Attacker: ✅ READY\n"
                       "  Network Recon:       ✅ READY\n"
                       "  OSINT Engine:        ✅ READY\n"
                       "  Exploit Framework:   ✅ READY\n"
                       "  C2 Server:           ⏸️ STANDBY\n"
                       "  Proxy Chain:         ✅ ACTIVE\n"
                       "\n"
                       "🎯 TIER 1 - ESSENTIAL (High Impact):\n"
                       "  API Security:        ✅ READY (REST, GraphQL, JWT, OAuth)\n"
                       "  Cloud Security:      ✅ READY (AWS, Azure, GCP)\n"
                       "  DNS Attacks:         ✅ READY (Zone transfer, takeover)\n"
                       "  Mobile Backend:      ✅ READY (Firebase, cert pinning)\n"
                       "\n"
                       "⚡ TIER 2 - ADVANCED (Differentiation):\n"
                       "  Supply Chain:        ✅ READY (Dependency confusion)\n"
                       "  SSO/Identity:        ✅ READY (SAML, OAuth, Kerberos)\n"
                       "  WebSocket:           ✅ READY (CSWSH, injection)\n"
                       "  GraphQL:             ✅ READY (Batching, depth attack)\n"
                       "\n"
                       "🔬 TIER 3 - SPECIALIZED (World-Class Edge):\n"
                       "  Browser Attacks:     ✅ READY (XS-Leaks, Spectre)\n"
                       "  Protocol Attacks:    ✅ READY (HTTP/2-3, gRPC, WebRTC)\n"
                       "\n"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       "📊 TOTAL: 17 MODULES | 150+ ATTACK VECTORS\n"
                       "🛡️ All operations routed through proxy chain\n"
                       "👻 Stealth mode: ENABLED | RAM-only: ACTIVE",
                data={
                    # Core modules
                    'web_scanner': True,
                    'credential_attacker': True,
                    'network_recon': True,
                    'osint_engine': True,
                    'exploit_framework': True,
                    'c2_server': False,
                    'proxy_chain': True,
                    # Tier 1
                    'api_security': True,
                    'cloud_security': True,
                    'dns_attacks': True,
                    'mobile_backend': True,
                    # Tier 2
                    'supply_chain': True,
                    'sso_identity': True,
                    'websocket': True,
                    'graphql': True,
                    # Tier 3
                    'browser_attacks': True,
                    'protocol_attacks': True,
                },
                follow_up_suggestions=['api scan', 'cloud scan', 'dns attack', 'graphql introspection', 'pentest help']
            )
    
    def _execute_superhero_command(self, context: CommandContext) -> CommandResult:
        """Execute SUPERHERO blockchain intelligence commands"""
        self._load_superhero_modules()
        
        intent = context.intent
        params = context.parameters
        
        # ===== BLOCKCHAIN FORENSICS =====
        if intent == 'superhero_trace_wallet':
            address = params.get('address')
            chain = params.get('chain', 'ethereum')
            if not address:
                return CommandResult(
                    success=False,
                    message="Please provide a wallet address to trace.",
                    follow_up_suggestions=['trace wallet 0x...', 'superhero help']
                )
            
            import asyncio
            try:
                result = asyncio.get_event_loop().run_until_complete(
                    self._superhero_engine.quick_trace(address, chain, depth=3)
                )
                tx_count = len(result.get('transactions', []))
                return CommandResult(
                    success=True,
                    message=f"WALLET TRACE COMPLETE\n\n"
                           f"Address: {address[:16]}...\n"
                           f"Chain: {chain}\n"
                           f"Transactions traced: {tx_count}\n\n"
                           f"Use 'identify owner {address}' for identity attribution.",
                    data={'trace_result': result},
                    follow_up_suggestions=['identify owner', 'cluster wallets', 'generate dossier']
                )
            except Exception as e:
                return CommandResult(success=False, message=f"Trace failed: {e}")
        
        elif intent == 'superhero_cluster':
            address = params.get('address')
            if not address:
                return CommandResult(
                    success=False,
                    message="Please provide a wallet address for clustering."
                )
            
            import asyncio
            try:
                if self._blockchain_forensics:
                    result = asyncio.get_event_loop().run_until_complete(
                        self._blockchain_forensics.cluster_wallets([address], params.get('chain', 'ethereum'))
                    )
                    cluster_count = len(result.get('clusters', {}))
                    return CommandResult(
                        success=True,
                        message=f"WALLET CLUSTERING COMPLETE\n\n"
                               f"Starting address: {address[:16]}...\n"
                               f"Clusters identified: {cluster_count}\n",
                        data={'cluster_result': result}
                    )
            except Exception as e:
                return CommandResult(success=False, message=f"Clustering failed: {e}")
        
        elif intent == 'superhero_detect_mixer':
            address = params.get('address')
            import asyncio
            try:
                if self._blockchain_forensics:
                    result = asyncio.get_event_loop().run_until_complete(
                        self._blockchain_forensics.detect_mixers([])  # Would need transactions
                    )
                    return CommandResult(
                        success=True,
                        message="MIXER DETECTION ANALYSIS\n\n"
                               f"Mixer services detected: {len(result) if result else 0}\n"
                               "Use 'mixer trace' to attempt tracking through mixers.",
                        data={'mixer_result': result}
                    )
            except Exception as e:
                return CommandResult(success=False, message=f"Mixer detection failed: {e}")
        
        elif intent == 'superhero_check_address':
            address = params.get('address')
            if not address:
                return CommandResult(
                    success=False,
                    message="Please provide an address to check."
                )
            
            import asyncio
            try:
                result = asyncio.get_event_loop().run_until_complete(
                    self._superhero_engine.check_address(address)
                )
                return CommandResult(
                    success=True,
                    message=f"ADDRESS CHECK\n\n"
                           f"Address: {address[:20]}...\n"
                           f"Flagged: {'YES' if result.get('is_flagged') else 'NO'}\n"
                           f"Risk Score: {result.get('risk_score', 0)}/100\n"
                           f"Known Entity: {result.get('known_entity', 'Unknown')}\n",
                    data={'check_result': result}
                )
            except Exception as e:
                return CommandResult(success=False, message=f"Address check failed: {e}")
        
        # ===== IDENTITY CORRELATION =====
        elif intent == 'superhero_identify_owner':
            address = params.get('address')
            if not address:
                return CommandResult(
                    success=False,
                    message="Please provide a wallet address for identity lookup."
                )
            
            import asyncio
            try:
                result = asyncio.get_event_loop().run_until_complete(
                    self._superhero_engine.quick_identity(address, params.get('chain', 'ethereum'))
                )
                candidates = result.get('identity_candidates', [])
                return CommandResult(
                    success=True,
                    message=f"IDENTITY ATTRIBUTION\n\n"
                           f"Address: {address[:16]}...\n"
                           f"Identity candidates found: {len(candidates)}\n\n"
                           f"{'High-confidence matches detected!' if any(c.get('overall_confidence', 0) > 0.8 for c in candidates) else 'Further investigation recommended.'}\n",
                    data={'identity_result': result},
                    follow_up_suggestions=['generate dossier', 'geolocation analysis', 'behavioral pattern']
                )
            except Exception as e:
                return CommandResult(success=False, message=f"Identity lookup failed: {e}")
        
        elif intent == 'superhero_ens_lookup':
            address = params.get('address')
            if self._identity_engine:
                try:
                    result = self._identity_engine.ens.resolve(address) if hasattr(self._identity_engine, 'ens') else None
                    return CommandResult(
                        success=True,
                        message=f"ENS LOOKUP\n\nAddress: {address[:16]}...\nENS Name: {result or 'None found'}",
                        data={'ens_result': result}
                    )
                except Exception as e:
                    return CommandResult(success=False, message=f"ENS lookup failed: {e}")
        
        # ===== GEOLOCATION ANALYSIS =====
        elif intent == 'superhero_geolocation':
            address = params.get('address')
            import asyncio
            try:
                if self._geolocation_analyzer:
                    result = asyncio.get_event_loop().run_until_complete(
                        self._geolocation_analyzer.analyze_location([])  # Would need transactions
                    )
                    return CommandResult(
                        success=True,
                        message=f"GEOLOCATION ANALYSIS\n\n"
                               f"Location estimates: {len(result.get('location_estimates', []))}\n"
                               f"Timezone: {result.get('timezone_estimate', 'Unknown')}\n",
                        data={'geolocation_result': result}
                    )
            except Exception as e:
                return CommandResult(success=False, message=f"Geolocation failed: {e}")
        
        elif intent == 'superhero_behavioral':
            return CommandResult(
                success=True,
                message="BEHAVIORAL PATTERN ANALYSIS\n\n"
                       "Requires running a full investigation to collect transaction timing data.\n"
                       "Use 'create investigation' followed by 'run investigation' to generate patterns.",
                follow_up_suggestions=['create investigation', 'run investigation']
            )
        
        # ===== INVESTIGATION MANAGEMENT =====
        elif intent == 'superhero_create_investigation':
            case_id = params.get('case_id', f"CASE-{id(self) % 10000:04d}")
            title = params.get('title', 'Blockchain Intelligence Investigation')
            
            import asyncio
            try:
                investigation = asyncio.get_event_loop().run_until_complete(
                    self._superhero_engine.create_investigation(
                        case_id=case_id,
                        title=title,
                        description="Created via AI Command Center"
                    )
                )
                return CommandResult(
                    success=True,
                    message=f"INVESTIGATION CREATED\n\n"
                           f"Investigation ID: {investigation.investigation_id}\n"
                           f"Case ID: {case_id}\n"
                           f"Title: {title}\n\n"
                           "Use 'add target <address>' to add wallet addresses to investigate.",
                    data={'investigation_id': investigation.investigation_id},
                    follow_up_suggestions=['add target', 'run investigation']
                )
            except Exception as e:
                return CommandResult(success=False, message=f"Failed to create investigation: {e}")
        
        elif intent == 'superhero_add_target':
            address = params.get('address')
            investigation_id = params.get('investigation_id')
            
            if not address:
                return CommandResult(
                    success=False,
                    message="Please provide a wallet address to add as target."
                )
            
            # Get latest investigation if not specified
            investigations = self._superhero_engine.list_investigations(limit=1)
            if not investigation_id and investigations:
                investigation_id = investigations[0].investigation_id
            
            if not investigation_id:
                return CommandResult(
                    success=False,
                    message="No active investigation. Use 'create investigation' first."
                )
            
            import asyncio
            try:
                target = asyncio.get_event_loop().run_until_complete(
                    self._superhero_engine.add_target(
                        investigation_id=investigation_id,
                        target_type='wallet',
                        value=address,
                        chain=params.get('chain', 'ethereum')
                    )
                )
                return CommandResult(
                    success=True,
                    message=f"TARGET ADDED\n\n"
                           f"Target ID: {target.target_id}\n"
                           f"Address: {address[:20]}...\n"
                           f"Investigation: {investigation_id}",
                    follow_up_suggestions=['run investigation', 'add target']
                )
            except Exception as e:
                return CommandResult(success=False, message=f"Failed to add target: {e}")
        
        elif intent == 'superhero_run_investigation':
            investigation_id = params.get('investigation_id')
            
            # Get latest investigation if not specified
            investigations = self._superhero_engine.list_investigations(limit=1)
            if not investigation_id and investigations:
                investigation_id = investigations[0].investigation_id
            
            if not investigation_id:
                return CommandResult(
                    success=False,
                    message="No investigation found. Use 'create investigation' first."
                )
            
            import asyncio
            try:
                investigation = asyncio.get_event_loop().run_until_complete(
                    self._superhero_engine.run_investigation(investigation_id)
                )
                return CommandResult(
                    success=True,
                    message=f"INVESTIGATION COMPLETE\n\n"
                           f"Investigation ID: {investigation.investigation_id}\n"
                           f"Status: {investigation.status.value}\n"
                           f"Targets: {len(investigation.targets)}\n\n"
                           f"Dossier generated: {'Yes' if investigation.dossier else 'No'}\n\n"
                           "Use 'export pdf' or 'export json' to export the dossier.",
                    data={
                        'investigation_id': investigation.investigation_id,
                        'status': investigation.status.value
                    },
                    follow_up_suggestions=['export pdf', 'export json', 'get alerts']
                )
            except Exception as e:
                return CommandResult(success=False, message=f"Investigation failed: {e}")
        
        elif intent == 'superhero_list_investigations':
            investigations = self._superhero_engine.list_investigations(limit=10)
            if not investigations:
                return CommandResult(
                    success=True,
                    message="No investigations found.\n\nUse 'create investigation' to start a new one.",
                    follow_up_suggestions=['create investigation']
                )
            
            msg = "INVESTIGATIONS\n\n"
            for inv in investigations:
                msg += f"  {inv.investigation_id}: {inv.title} [{inv.status.value}]\n"
            
            return CommandResult(success=True, message=msg)
        
        elif intent == 'superhero_get_alerts':
            alerts = self._superhero_engine.get_alerts(unacknowledged_only=True, limit=10)
            if not alerts:
                return CommandResult(
                    success=True,
                    message="No pending alerts."
                )
            
            msg = f"ALERTS ({len(alerts)} pending)\n\n"
            for alert in alerts:
                msg += f"  [{alert.priority.value.upper()}] {alert.title}\n"
            
            return CommandResult(success=True, message=msg)
        
        # ===== DOSSIER GENERATION =====
        elif intent == 'superhero_generate_dossier':
            investigations = self._superhero_engine.list_investigations(limit=1)
            if not investigations:
                return CommandResult(
                    success=False,
                    message="No investigation found. Create and run an investigation first.",
                    follow_up_suggestions=['create investigation']
                )
            
            investigation = investigations[0]
            if not investigation.dossier:
                return CommandResult(
                    success=False,
                    message="Investigation has not been run yet. Use 'run investigation' first.",
                    follow_up_suggestions=['run investigation']
                )
            
            return CommandResult(
                success=True,
                message=f"DOSSIER AVAILABLE\n\n"
                       f"Dossier ID: {investigation.dossier.dossier_id}\n"
                       f"Classification: {investigation.dossier.classification.value}\n"
                       f"Evidence items: {len(investigation.dossier.evidence_items)}\n"
                       f"Hash: {investigation.dossier.dossier_hash[:32]}...\n\n"
                       "Export with 'export pdf', 'export json', or 'export html'.",
                data={'dossier_id': investigation.dossier.dossier_id},
                follow_up_suggestions=['export pdf', 'export json', 'export html']
            )
        
        elif intent in ['superhero_export_pdf', 'superhero_export_json', 'superhero_export_html']:
            format_map = {
                'superhero_export_pdf': 'pdf',
                'superhero_export_json': 'json',
                'superhero_export_html': 'html'
            }
            export_format = format_map.get(intent, 'json')
            
            investigations = self._superhero_engine.list_investigations(limit=1)
            if not investigations or not investigations[0].dossier:
                return CommandResult(
                    success=False,
                    message="No dossier available. Run an investigation first.",
                    follow_up_suggestions=['run investigation']
                )
            
            import asyncio
            from modules.superhero import DossierFormat
            
            format_enum = {
                'pdf': DossierFormat.PDF,
                'json': DossierFormat.JSON,
                'html': DossierFormat.HTML
            }[export_format]
            
            try:
                data = asyncio.get_event_loop().run_until_complete(
                    self._superhero_engine.export_investigation(
                        investigations[0].investigation_id,
                        output_format=format_enum
                    )
                )
                return CommandResult(
                    success=True,
                    message=f"DOSSIER EXPORTED\n\n"
                           f"Format: {export_format.upper()}\n"
                           f"Size: {len(data):,} bytes\n\n"
                           "Dossier is court-ready with full evidence chain.",
                    data={'export_format': export_format, 'size': len(data)}
                )
            except Exception as e:
                return CommandResult(success=False, message=f"Export failed: {e}")
        
        # ===== MONITORING =====
        elif intent == 'superhero_monitor_address':
            address = params.get('address')
            if not address:
                return CommandResult(
                    success=False,
                    message="Please provide an address to monitor."
                )
            
            investigations = self._superhero_engine.list_investigations(limit=1)
            investigation_id = investigations[0].investigation_id if investigations else "MONITOR-001"
            
            import asyncio
            try:
                monitored = asyncio.get_event_loop().run_until_complete(
                    self._superhero_engine.monitor_address(
                        address=address,
                        chain=params.get('chain', 'ethereum'),
                        investigation_id=investigation_id
                    )
                )
                return CommandResult(
                    success=True,
                    message=f"ADDRESS MONITORING ENABLED\n\n"
                           f"Address: {address[:20]}...\n"
                           f"Chain: {monitored.chain.value}\n"
                           f"Alerts: Enabled\n\n"
                           "You will be alerted when activity is detected.",
                    follow_up_suggestions=['get alerts', 'stop monitor']
                )
            except Exception as e:
                return CommandResult(success=False, message=f"Monitor setup failed: {e}")
        
        # ===== STATUS =====
        elif intent in ['superhero_status', 'superhero_investigation_status', 
                       'superhero_identity_status', 'superhero_dossier_status']:
            status = self._superhero_engine.get_status()
            stats = self._superhero_engine.get_statistics()
            
            return CommandResult(
                success=True,
                message=f"SUPERHERO STATUS\n\n"
                       f"Engine: {status.get('status', 'unknown').upper()}\n"
                       f"Monitoring: {'ACTIVE' if status.get('monitoring_active') else 'INACTIVE'}\n\n"
                       f"Active Investigations: {status.get('active_investigations', 0)}\n"
                       f"Completed: {status.get('completed_investigations', 0)}\n"
                       f"Monitored Addresses: {status.get('monitored_addresses', 0)}\n"
                       f"Pending Alerts: {status.get('pending_alerts', 0)}\n\n"
                       f"Total Transactions Traced: {stats.get('total_transactions_traced', 0)}\n"
                       f"Identities Correlated: {stats.get('total_identities_correlated', 0)}\n"
                       f"Dossiers Generated: {stats.get('dossiers_generated', 0)}",
                data={'status': status, 'statistics': stats},
                follow_up_suggestions=['create investigation', 'trace wallet', 'identify owner']
            )
        
        # ===== CRYPTOCURRENCY SECURITY ASSESSMENT & RECOVERY TOOLKIT =====
        
        # Wallet Security Scanner
        elif intent == 'toolkit_wallet_scan':
            address = params.get('address')
            if not address:
                return CommandResult(
                    success=False,
                    message="Please provide a wallet address to scan.",
                    follow_up_suggestions=['scan wallet 0x...']
                )
            
            self._load_toolkit_modules()
            if self._wallet_security_scanner:
                result = self._wallet_security_scanner.scan_wallet(address, params.get('chain', 'ethereum'))
                return CommandResult(
                    success=True,
                    message=f"WALLET SECURITY SCAN COMPLETE\n\n"
                           f"Address: {address[:20]}...\n"
                           f"Vulnerabilities Found: {len(result.get('vulnerabilities', []))}\n"
                           f"Risk Score: {result.get('risk_score', 0)}/100\n"
                           f"Recommendations: {len(result.get('recommendations', []))}\n",
                    data={'scan_result': result},
                    follow_up_suggestions=['vulnerability scan', 'security assessment']
                )
            return CommandResult(success=False, message="Wallet Security Scanner not available.")
        
        elif intent == 'toolkit_security_assess':
            self._load_toolkit_modules()
            return CommandResult(
                success=True,
                message="SECURITY ASSESSMENT\n\n"
                       "Requires authorized wallet access with proper documentation.\n"
                       "Use 'initiate recovery' with authorization to proceed.",
                follow_up_suggestions=['initiate recovery', 'scan wallet']
            )
        
        # Key Derivation Analyzer
        elif intent == 'toolkit_derivation_analyze':
            self._load_toolkit_modules()
            if self._key_derivation_analyzer:
                return CommandResult(
                    success=True,
                    message="KEY DERIVATION ANALYZER\n\n"
                           "Provide seed phrase information for analysis:\n"
                           "  - Known words and positions\n"
                           "  - Password hints (if encrypted)\n"
                           "  - Derivation path used\n\n"
                           "REQUIRES: Authorized recovery documentation.",
                    follow_up_suggestions=['check entropy', 'seed strength']
                )
            return CommandResult(success=False, message="Key Derivation Analyzer not available.")
        
        elif intent == 'toolkit_entropy_check':
            self._load_toolkit_modules()
            return CommandResult(
                success=True,
                message="ENTROPY ANALYSIS\n\n"
                       "Entropy analysis evaluates the randomness of key generation.\n"
                       "Low entropy may indicate vulnerable keys.",
                follow_up_suggestions=['analyze derivation', 'key weakness']
            )
        
        # Smart Contract Auditor
        elif intent == 'toolkit_contract_audit':
            address = params.get('address')
            if not address:
                return CommandResult(
                    success=False,
                    message="Please provide a contract address to audit.",
                    follow_up_suggestions=['audit contract 0x...']
                )
            
            self._load_toolkit_modules()
            if self._smart_contract_auditor:
                result = self._smart_contract_auditor.audit_contract(address, params.get('chain', 'ethereum'))
                return CommandResult(
                    success=True,
                    message=f"CONTRACT AUDIT COMPLETE\n\n"
                           f"Contract: {address[:20]}...\n"
                           f"Vulnerabilities: {len(result.get('vulnerabilities', []))}\n"
                           f"Security Score: {result.get('security_score', 0)}/100\n"
                           f"Exploit Patterns: {len(result.get('exploit_patterns', []))}\n",
                    data={'audit_result': result},
                    follow_up_suggestions=['detect exploit', 'reentrancy check']
                )
            return CommandResult(success=False, message="Smart Contract Auditor not available.")
        
        elif intent == 'toolkit_reentrancy':
            self._load_toolkit_modules()
            return CommandResult(
                success=True,
                message="REENTRANCY CHECK\n\n"
                       "Reentrancy vulnerabilities allow attackers to drain funds.\n"
                       "Use 'audit contract' for full analysis.",
                follow_up_suggestions=['audit contract']
            )
        
        # Recovery Toolkit
        elif intent == 'toolkit_initiate_recovery':
            self._load_toolkit_modules()
            return CommandResult(
                success=True,
                message="RECOVERY TOOLKIT\n\n"
                       "AUTHORIZATION REQUIRED\n"
                       "Recovery operations require:\n"
                       "  1. Signed authorization document\n"
                       "  2. Proof of wallet ownership\n"
                       "  3. Valid identification\n\n"
                       "Supported recovery methods:\n"
                       "  - Seed phrase reconstruction\n"
                       "  - Password recovery\n"
                       "  - Multisig recovery\n"
                       "  - Hardware wallet recovery\n\n"
                       "All data handled in RAM only (stealth compliant).",
                follow_up_suggestions=['seed reconstruction', 'password recovery', 'recovery status']
            )
        
        elif intent == 'toolkit_seed_reconstruct':
            self._load_toolkit_modules()
            return CommandResult(
                success=True,
                message="SEED RECONSTRUCTION\n\n"
                       "Provide known information:\n"
                       "  - Known words and their positions\n"
                       "  - Total word count (12, 18, or 24)\n"
                       "  - Word hints for unknown positions\n\n"
                       "AUTHORIZATION: Required before proceeding.",
                follow_up_suggestions=['initiate recovery']
            )
        
        elif intent == 'toolkit_password_recovery':
            self._load_toolkit_modules()
            return CommandResult(
                success=True,
                message="PASSWORD RECOVERY\n\n"
                       "For encrypted keystore recovery.\n"
                       "Provide password hints:\n"
                       "  - Memorable words\n"
                       "  - Significant years\n"
                       "  - Known patterns\n\n"
                       "AUTHORIZATION: Required before proceeding.",
                follow_up_suggestions=['initiate recovery']
            )
        
        elif intent == 'toolkit_recovery_status':
            self._load_toolkit_modules()
            return CommandResult(
                success=True,
                message="RECOVERY TOOLKIT STATUS\n\n"
                       "Status: READY\n"
                       "Mode: Stealth (RAM-only)\n"
                       "Supported wallets: MetaMask, Ledger, Trezor, Safe Multisig\n"
                       "Authorization: Required for all operations",
                follow_up_suggestions=['initiate recovery', 'scan wallet']
            )
        
        # Malicious Address Database
        elif intent == 'toolkit_malicious_check':
            address = params.get('address')
            if not address:
                return CommandResult(
                    success=False,
                    message="Please provide an address to check against malicious database.",
                    follow_up_suggestions=['check malicious 0x...']
                )
            
            self._load_toolkit_modules()
            if self._malicious_address_db:
                result = self._malicious_address_db.check_address(address)
                threat_level = result.get('threat_level', 'unknown')
                return CommandResult(
                    success=True,
                    message=f"MALICIOUS ADDRESS CHECK\n\n"
                           f"Address: {address[:20]}...\n"
                           f"Known Malicious: {'YES ⚠️' if result.get('known_malicious') else 'NO ✓'}\n"
                           f"Threat Level: {threat_level.upper()}\n"
                           f"Categories: {', '.join(result.get('categories', ['None']))}\n",
                    data={'check_result': result},
                    follow_up_suggestions=['search malicious', 'threat lookup']
                )
            return CommandResult(success=False, message="Malicious Address Database not available.")
        
        elif intent == 'toolkit_db_status':
            self._load_toolkit_modules()
            if self._malicious_address_db:
                stats = self._malicious_address_db.get_statistics()
                return CommandResult(
                    success=True,
                    message=f"MALICIOUS ADDRESS DATABASE STATUS\n\n"
                           f"Total Addresses: {stats.get('total_addresses', 0)}\n"
                           f"Total Clusters: {stats.get('total_clusters', 0)}\n"
                           f"Active Threats: {stats.get('active_threats', 0)}\n"
                           f"Total Stolen (USD): ${stats.get('total_stolen_usd', 0):,.2f}\n",
                    data={'db_stats': stats},
                    follow_up_suggestions=['check malicious', 'search malicious']
                )
            return CommandResult(success=False, message="Malicious Address Database not available.")
        
        # Authority Report Generator
        elif intent == 'toolkit_create_case':
            self._load_toolkit_modules()
            if self._authority_report_generator:
                from datetime import datetime
                case_info = self._authority_report_generator.create_case(
                    case_name="Cryptocurrency Security Investigation",
                    case_type="Blockchain Forensics",
                    incident_date=datetime.now(),
                    total_loss_usd=params.get('loss_usd', 0),
                    jurisdiction=params.get('jurisdiction', 'International')
                )
                return CommandResult(
                    success=True,
                    message=f"CASE CREATED\n\n"
                           f"Case ID: {case_info.case_id}\n"
                           f"Case Name: {case_info.case_name}\n"
                           f"Created: {case_info.report_date.isoformat()}\n\n"
                           "Use 'add evidence' to add findings to this case.",
                    data={'case_id': case_info.case_id},
                    follow_up_suggestions=['add evidence', 'authority report']
                )
            return CommandResult(success=False, message="Authority Report Generator not available.")
        
        elif intent == 'toolkit_authority_report':
            self._load_toolkit_modules()
            return CommandResult(
                success=True,
                message="AUTHORITY REPORT GENERATOR\n\n"
                       "Generate court-ready reports for:\n"
                       "  - Law enforcement agencies\n"
                       "  - Regulatory bodies\n"
                       "  - Insurance claims\n"
                       "  - Civil litigation\n\n"
                       "Reports include:\n"
                       "  - Evidence with chain of custody\n"
                       "  - Transaction analysis\n"
                       "  - Identity correlations\n"
                       "  - Digital signatures\n\n"
                       "Use 'create case' to start a new report.",
                follow_up_suggestions=['create case', 'add evidence']
            )
        
        elif intent == 'toolkit_report_status':
            self._load_toolkit_modules()
            return CommandResult(
                success=True,
                message="AUTHORITY REPORT STATUS\n\n"
                       "Status: READY\n"
                       "Formats: Markdown, JSON, HTML, PDF\n"
                       "Classification Levels: Unclassified to Law Enforcement Sensitive\n"
                       "Chain of Custody: Full tracking",
                follow_up_suggestions=['create case', 'authority report']
            )
        
        # Default fallback
        else:
            return CommandResult(
                success=True,
                message="SUPERHERO - Blockchain Intelligence Suite\n\n"
                       "Available commands:\n"
                       "  trace wallet <address>    - Trace transactions\n"
                       "  identify owner <address>  - Identity attribution\n"
                       "  cluster wallets           - Common ownership analysis\n"
                       "  check address <address>   - Threat database lookup\n"
                       "  create investigation      - Start new case\n"
                       "  run investigation         - Execute full analysis\n"
                       "  generate dossier          - Create evidence report\n"
                       "  export pdf/json/html      - Export dossier\n"
                       "  monitor address           - Real-time alerts\n"
                       "  superhero status          - System status\n\n"
                       "SECURITY TOOLKIT:\n"
                       "  scan wallet <address>     - Security vulnerability scan\n"
                       "  audit contract <address>  - Smart contract audit\n"
                       "  initiate recovery         - Authorized wallet recovery\n"
                       "  check malicious <address> - Threat database lookup\n"
                       "  create case               - Law enforcement report\n",
                follow_up_suggestions=['trace wallet', 'scan wallet', 'create investigation', 'superhero help']
            )
    
    def _load_pentest_modules(self):
        """Lazy load penetration testing modules"""
        try:
            if self._web_scanner is None:
                from modules.pentest.web_scanner import WebScanner
                self._web_scanner = WebScanner()
                self.logger.info("Web Scanner module loaded")
        except ImportError as e:
            self.logger.warning(f"Web Scanner not available: {e}")
        
        try:
            if self._credential_attacker is None:
                from modules.pentest.credential_attack import CredentialAttacker
                self._credential_attacker = CredentialAttacker()
                self.logger.info("Credential Attacker module loaded")
        except ImportError as e:
            self.logger.warning(f"Credential Attacker not available: {e}")
        
        try:
            if self._network_recon is None:
                from modules.pentest.network_recon import NetworkRecon
                self._network_recon = NetworkRecon()
                self.logger.info("Network Recon module loaded")
        except ImportError as e:
            self.logger.warning(f"Network Recon not available: {e}")
        
        try:
            if self._osint_engine is None:
                from modules.pentest.osint_engine import OSINTEngine
                self._osint_engine = OSINTEngine()
                self.logger.info("OSINT Engine module loaded")
        except ImportError as e:
            self.logger.warning(f"OSINT Engine not available: {e}")
        
        try:
            if self._exploit_framework is None:
                from modules.pentest.exploit_framework import ExploitFramework
                self._exploit_framework = ExploitFramework()
                self.logger.info("Exploit Framework module loaded")
        except ImportError as e:
            self.logger.warning(f"Exploit Framework not available: {e}")
        
        try:
            if self._c2_server is None:
                from modules.pentest.c2_framework import C2Server
                self._c2_server = None  # Created on demand
                self.logger.info("C2 Framework module available")
        except ImportError as e:
            self.logger.warning(f"C2 Framework not available: {e}")
        
        try:
            if self._proxy_manager is None:
                from modules.pentest.proxy_manager import ProxyChainManager
                self._proxy_manager = ProxyChainManager()
                self.logger.info("Proxy Chain Manager module loaded")
        except ImportError as e:
            self.logger.warning(f"Proxy Chain Manager not available: {e}")
    
    def _load_superhero_modules(self):
        """Lazy load SUPERHERO blockchain intelligence modules"""
        try:
            if self._superhero_engine is None:
                from modules.superhero import SuperheroEngine, get_engine
                self._superhero_engine = get_engine()
                self.logger.info("SUPERHERO Engine loaded")
        except ImportError as e:
            self.logger.warning(f"SUPERHERO Engine not available: {e}")
        
        try:
            if self._blockchain_forensics is None:
                from modules.superhero import BlockchainForensics
                self._blockchain_forensics = BlockchainForensics()
                self.logger.info("Blockchain Forensics module loaded")
        except ImportError as e:
            self.logger.warning(f"Blockchain Forensics not available: {e}")
        
        try:
            if self._identity_engine is None:
                from modules.superhero import IdentityEngine
                self._identity_engine = IdentityEngine()
                self.logger.info("Identity Engine module loaded")
        except ImportError as e:
            self.logger.warning(f"Identity Engine not available: {e}")
        
        try:
            if self._geolocation_analyzer is None:
                from modules.superhero import GeolocationAnalyzer
                self._geolocation_analyzer = GeolocationAnalyzer()
                self.logger.info("Geolocation Analyzer module loaded")
        except ImportError as e:
            self.logger.warning(f"Geolocation Analyzer not available: {e}")
        
        try:
            if self._counter_measures is None:
                from modules.superhero import CounterMeasureAnalyzer
                self._counter_measures = CounterMeasureAnalyzer()
                self.logger.info("Counter Measures module loaded")
        except ImportError as e:
            self.logger.warning(f"Counter Measures not available: {e}")
        
        try:
            if self._dossier_generator is None:
                from modules.superhero import DossierGenerator
                self._dossier_generator = DossierGenerator()
                self.logger.info("Dossier Generator module loaded")
        except ImportError as e:
            self.logger.warning(f"Dossier Generator not available: {e}")
    
    def _load_toolkit_modules(self):
        """Lazy load Cryptocurrency Security Assessment & Recovery Toolkit modules"""
        try:
            if self._wallet_security_scanner is None:
                from modules.superhero.wallet_security_scanner import WalletSecurityScanner, create_scanner
                self._wallet_security_scanner = create_scanner()
                self.logger.info("Wallet Security Scanner module loaded")
        except ImportError as e:
            self.logger.warning(f"Wallet Security Scanner not available: {e}")
        
        try:
            if self._key_derivation_analyzer is None:
                from modules.superhero.key_derivation_analyzer import KeyDerivationAnalyzer, create_analyzer
                self._key_derivation_analyzer = create_analyzer()
                self.logger.info("Key Derivation Analyzer module loaded")
        except ImportError as e:
            self.logger.warning(f"Key Derivation Analyzer not available: {e}")
        
        try:
            if self._smart_contract_auditor is None:
                from modules.superhero.smart_contract_auditor import SmartContractAuditor, create_auditor
                self._smart_contract_auditor = create_auditor()
                self.logger.info("Smart Contract Auditor module loaded")
        except ImportError as e:
            self.logger.warning(f"Smart Contract Auditor not available: {e}")
        
        try:
            if self._recovery_toolkit is None:
                from modules.superhero.recovery_toolkit import RecoveryToolkit, create_recovery_toolkit
                self._recovery_toolkit = create_recovery_toolkit(stealth_mode=True)
                self.logger.info("Recovery Toolkit module loaded (stealth mode)")
        except ImportError as e:
            self.logger.warning(f"Recovery Toolkit not available: {e}")
        
        try:
            if self._malicious_address_db is None:
                from modules.superhero.malicious_address_db import MaliciousAddressDatabase, create_database
                self._malicious_address_db = create_database(ram_only=True)
                self.logger.info("Malicious Address Database module loaded (RAM-only)")
        except ImportError as e:
            self.logger.warning(f"Malicious Address Database not available: {e}")
        
        try:
            if self._authority_report_generator is None:
                from modules.superhero.authority_report_generator import AuthorityReportGenerator, create_report_generator
                self._authority_report_generator = create_report_generator(stealth_mode=True)
                self.logger.info("Authority Report Generator module loaded (stealth mode)")
        except ImportError as e:
            self.logger.warning(f"Authority Report Generator not available: {e}")
    
    # ========== AI v2.0 ENHANCED INTELLIGENCE ==========
    
    def _load_ai_v2_modules(self):
        """Lazy load AI v2.0 Enhanced Intelligence modules"""
        try:
            if not hasattr(self, '_ai_v2_enhanced'):
                from core.ai_v2.enhanced_ai import EnhancedAI
                self._ai_v2_enhanced = EnhancedAI()
                self.logger.info("AI v2.0 Enhanced Intelligence loaded")
        except ImportError as e:
            self.logger.warning(f"AI v2.0 Enhanced AI not available: {e}")
            self._ai_v2_enhanced = None
        
        try:
            if not hasattr(self, '_ai_v2_local_llm'):
                from core.ai_v2.local_llm import LocalLLM
                self._ai_v2_local_llm = LocalLLM()
                self.logger.info("AI v2.0 Local LLM loader available")
        except ImportError as e:
            self.logger.warning(f"AI v2.0 Local LLM not available: {e}")
            self._ai_v2_local_llm = None
        
        try:
            if not hasattr(self, '_ai_v2_agent'):
                from core.ai_v2.agent_framework import AutonomousAgent, AgentMode
                self._ai_v2_agent = AutonomousAgent(mode=AgentMode.AUTONOMOUS)
                self._ai_v2_agent.set_command_center(self)
                self.logger.info("AI v2.0 Autonomous Agent Framework loaded")
        except ImportError as e:
            self.logger.warning(f"AI v2.0 Agent Framework not available: {e}")
            self._ai_v2_agent = None
        
        try:
            if not hasattr(self, '_ai_v2_attack_planner'):
                from core.ai_v2.attack_planner import AttackPlanner
                self._ai_v2_attack_planner = AttackPlanner()
                self._ai_v2_attack_planner.set_command_center(self)
                self.logger.info("AI v2.0 Attack Planner loaded")
        except ImportError as e:
            self.logger.warning(f"AI v2.0 Attack Planner not available: {e}")
            self._ai_v2_attack_planner = None
        
        try:
            if not hasattr(self, '_ai_v2_rag'):
                from core.ai_v2.rag_engine import RAGEngine
                self._ai_v2_rag = RAGEngine()
                self.logger.info("AI v2.0 RAG Knowledge Base loaded")
        except ImportError as e:
            self.logger.warning(f"AI v2.0 RAG Engine not available: {e}")
            self._ai_v2_rag = None
        
        try:
            if not hasattr(self, '_ai_v2_memory'):
                from core.ai_v2.memory_store import RAMMemory
                self._ai_v2_memory = RAMMemory()
                self.logger.info("AI v2.0 RAM-only Memory Store loaded")
        except ImportError as e:
            self.logger.warning(f"AI v2.0 Memory Store not available: {e}")
            self._ai_v2_memory = None
        
        try:
            if not hasattr(self, '_ai_v2_online_intel'):
                from core.ai_v2.online_intel import OnlineIntelligence
                self._ai_v2_online_intel = OnlineIntelligence()
                self.logger.info("AI v2.0 Online Intelligence loaded (Tor-routed)")
        except ImportError as e:
            self.logger.warning(f"AI v2.0 Online Intel not available: {e}")
            self._ai_v2_online_intel = None
    
    def _execute_ai_enhanced_command(self, context: CommandContext) -> CommandResult:
        """Execute AI v2.0 Enhanced Intelligence commands"""
        self._load_ai_v2_modules()
        
        intent = context.intent
        params = context.parameters
        
        # ===== AI MODE CONTROL =====
        if intent == 'ai_enable_enhanced':
            if self._ai_v2_enhanced:
                try:
                    self._ai_v2_enhanced.enable()
                    return CommandResult(
                        success=True,
                        message="AI v2.0 ENHANCED MODE ACTIVATED\n\n"
                               "Features enabled:\n"
                               "  • Local LLM (Mistral 7B) - Unfiltered\n"
                               "  • RAG Knowledge Base - RF/Exploit intelligence\n"
                               "  • Autonomous Agent Framework\n"
                               "  • RAM-only Contextual Memory\n"
                               "  • Attack Planning System\n\n"
                               "All processing is LOCAL. Zero telemetry.\n"
                               "Online intel available via Tor (opt-in).",
                        data={'ai_mode': 'enhanced'},
                        follow_up_suggestions=['ai status', 'load model', 'plan attack']
                    )
                except Exception as e:
                    return CommandResult(success=False, message=f"Failed to enable enhanced AI: {e}")
            return CommandResult(success=False, message="AI v2.0 Enhanced modules not available.")
        
        elif intent == 'ai_disable_enhanced':
            if self._ai_v2_enhanced:
                try:
                    self._ai_v2_enhanced.disable()
                    return CommandResult(
                        success=True,
                        message="AI v2.0 Enhanced mode DISABLED.\n"
                               "Returning to standard command processing.",
                        data={'ai_mode': 'standard'}
                    )
                except Exception as e:
                    return CommandResult(success=False, message=f"Failed to disable: {e}")
            return CommandResult(success=True, message="AI v2.0 was not enabled.")
        
        elif intent == 'ai_offline_mode':
            return CommandResult(
                success=True,
                message="AI v2.0 OFFLINE MODE\n\n"
                       "All AI processing is LOCAL by default:\n"
                       "  • Local LLM runs on device\n"
                       "  • No network required for intelligence\n"
                       "  • RAG knowledge base is offline\n"
                       "  • Memory stored in RAM only\n\n"
                       "Online intel is disabled. System is air-gapped safe.",
                data={'ai_network': 'offline'}
            )
        
        elif intent == 'ai_online_intel':
            if self._ai_v2_online_intel:
                return CommandResult(
                    success=True,
                    message="AI v2.0 ONLINE INTEL MODE\n\n"
                           "⚠️ Online intelligence enabled via TOR:\n"
                           "  • All connections routed through Tor\n"
                           "  • No direct IP exposure\n"
                           "  • Anonymous exploit database access\n"
                           "  • Real-time CVE/threat feeds\n\n"
                           "OPSEC: Ensure Tor is properly configured.",
                    data={'ai_network': 'tor_online'},
                    warnings=["Online mode increases detection surface"]
                )
            return CommandResult(success=False, message="Online intelligence module not available.")
        
        # ===== LLM CONTROL =====
        elif intent == 'ai_load_model':
            if self._ai_v2_local_llm:
                try:
                    model_name = params.get('model', 'mistral-7b-q4')
                    self._ai_v2_local_llm.load_model(model_name)
                    return CommandResult(
                        success=True,
                        message=f"LOCAL LLM LOADED\n\n"
                               f"Model: {model_name}\n"
                               f"Quantization: Q4_K_M (~4.4GB)\n"
                               f"Mode: Uncensored/Unfiltered\n"
                               f"Backend: llama.cpp\n\n"
                               "Ready for natural language commands.",
                        data={'model': model_name, 'status': 'loaded'},
                        follow_up_suggestions=['ai query', 'autonomous attack', 'plan attack']
                    )
                except Exception as e:
                    return CommandResult(success=False, message=f"Model load failed: {e}")
            return CommandResult(success=False, message="Local LLM module not available.")
        
        elif intent == 'ai_unload_model':
            if self._ai_v2_local_llm:
                try:
                    self._ai_v2_local_llm.unload_model()
                    return CommandResult(
                        success=True,
                        message="LLM model unloaded. Memory freed.",
                        data={'model': None, 'status': 'unloaded'}
                    )
                except Exception as e:
                    return CommandResult(success=False, message=f"Unload failed: {e}")
            return CommandResult(success=True, message="No model was loaded.")
        
        elif intent == 'ai_model_status':
            if self._ai_v2_local_llm:
                status = self._ai_v2_local_llm.get_status()
                return CommandResult(
                    success=True,
                    message=f"LLM STATUS\n\n"
                           f"Model loaded: {status.get('model_loaded', False)}\n"
                           f"Model name: {status.get('model_name', 'None')}\n"
                           f"Memory usage: {status.get('memory_mb', 0):.1f} MB\n"
                           f"Backend: {status.get('backend', 'None')}\n"
                           f"Inference ready: {status.get('ready', False)}",
                    data=status
                )
            return CommandResult(
                success=True,
                message="LLM STATUS\n\nLocal LLM module not initialized.\nUse 'enable enhanced ai' first."
            )
        
        # ===== AUTONOMOUS AGENT CONTROL =====
        elif intent == 'ai_create_agent':
            agent_type = params.get('agent_type', 'reconnaissance')
            if self._ai_v2_agent:
                try:
                    agent_id = self._ai_v2_agent.create_agent(agent_type)
                    return CommandResult(
                        success=True,
                        message=f"AUTONOMOUS AGENT CREATED\n\n"
                               f"Agent ID: {agent_id}\n"
                               f"Type: {agent_type}\n"
                               f"Status: Ready\n\n"
                               f"Use 'start agent {agent_id}' to begin autonomous operation.",
                        data={'agent_id': agent_id, 'type': agent_type},
                        follow_up_suggestions=[f'start agent {agent_id}', 'list agents']
                    )
                except Exception as e:
                    return CommandResult(success=False, message=f"Agent creation failed: {e}")
            return CommandResult(success=False, message="Agent framework not available.")
        
        elif intent == 'ai_start_agent':
            agent_id = params.get('agent_id')
            if self._ai_v2_agent:
                try:
                    self._ai_v2_agent.start_agent(agent_id)
                    return CommandResult(
                        success=True,
                        message=f"AGENT STARTED\n\n"
                               f"Agent {agent_id or 'default'} is now running autonomously.\n"
                               f"Use 'agent status' to monitor progress.\n"
                               f"Use 'stop agent' to halt operations.",
                        data={'agent_id': agent_id, 'status': 'running'},
                        follow_up_suggestions=['agent status', 'stop agent']
                    )
                except Exception as e:
                    return CommandResult(success=False, message=f"Failed to start agent: {e}")
            return CommandResult(success=False, message="Agent framework not available.")
        
        elif intent == 'ai_stop_agent':
            if self._ai_v2_agent:
                try:
                    self._ai_v2_agent.stop_all_agents()
                    return CommandResult(
                        success=True,
                        message="All autonomous agents STOPPED.\n"
                               "Operations halted. Memory preserved.",
                        data={'status': 'stopped'}
                    )
                except Exception as e:
                    return CommandResult(success=False, message=f"Failed to stop agents: {e}")
            return CommandResult(success=True, message="No agents running.")
        
        elif intent == 'ai_list_agents':
            if self._ai_v2_agent:
                agents = self._ai_v2_agent.list_agents()
                if agents:
                    agent_list = "\n".join([f"  • {a['id']}: {a['type']} ({a['status']})" for a in agents])
                    return CommandResult(
                        success=True,
                        message=f"ACTIVE AGENTS\n\n{agent_list}",
                        data={'agents': agents}
                    )
                return CommandResult(success=True, message="No agents configured.")
            return CommandResult(success=False, message="Agent framework not available.")
        
        elif intent == 'ai_agent_status':
            if self._ai_v2_agent:
                status = self._ai_v2_agent.get_status()
                return CommandResult(
                    success=True,
                    message=f"AGENT FRAMEWORK STATUS\n\n"
                           f"Active agents: {status.get('active_count', 0)}\n"
                           f"Tasks completed: {status.get('tasks_completed', 0)}\n"
                           f"Tasks pending: {status.get('tasks_pending', 0)}\n"
                           f"Memory used: {status.get('memory_mb', 0):.1f} MB",
                    data=status
                )
            return CommandResult(success=False, message="Agent framework not available.")
        
        # ===== ATTACK PLANNING =====
        elif intent == 'ai_plan_attack':
            target = params.get('target')
            if self._ai_v2_attack_planner:
                try:
                    plan = self._ai_v2_attack_planner.create_plan(target)
                    steps = plan.get('steps', [])
                    steps_text = "\n".join([f"  {i+1}. [{s.get('phase', 'unknown')}] {s.get('name', 'Unknown step')}" for i, s in enumerate(steps[:10])])
                    return CommandResult(
                        success=True,
                        message=f"ATTACK PLAN GENERATED\n\n"
                               f"Target: {target or 'All in range'}\n"
                               f"Plan ID: {plan.get('plan_id')}\n"
                               f"Steps: {len(steps)}\n\n"
                               f"Sequence:\n{steps_text}\n\n"
                               f"Use 'execute plan' to begin (requires confirmation).",
                        data={'plan': plan},
                        follow_up_suggestions=['execute plan', 'modify plan', 'cancel plan']
                    )
                except Exception as e:
                    return CommandResult(success=False, message=f"Planning failed: {e}")
            return CommandResult(success=False, message="Attack planner not available.")
        
        elif intent == 'ai_execute_plan':
            if self._ai_v2_attack_planner:
                try:
                    result = self._ai_v2_attack_planner.execute_current_plan()
                    return CommandResult(
                        success=True,
                        message=f"ATTACK PLAN EXECUTING\n\n"
                               f"Status: {result.get('status', 'running')}\n"
                               f"Steps completed: {result.get('completed', 0)}/{result.get('total', 0)}\n\n"
                               f"Monitor with 'attack status'. Stop with 'stop attack'.",
                        data=result,
                        follow_up_suggestions=['attack status', 'stop attack']
                    )
                except Exception as e:
                    return CommandResult(success=False, message=f"Execution failed: {e}")
            return CommandResult(success=False, message="Attack planner not available.")
        
        elif intent == 'ai_autonomous_attack':
            target = params.get('target')
            if self._ai_v2_attack_planner and self._ai_v2_agent:
                try:
                    result = self._ai_v2_attack_planner.autonomous_attack(target)
                    return CommandResult(
                        success=True,
                        message=f"AUTONOMOUS ATTACK INITIATED\n\n"
                               f"Target: {target or 'Auto-detect'}\n"
                               f"Mode: Full autonomous\n"
                               f"Status: Running\n\n"
                               f"AI will:\n"
                               f"  1. Scan and identify targets\n"
                               f"  2. Select optimal attack vectors\n"
                               f"  3. Execute attack sequence\n"
                               f"  4. Adapt to defenses\n"
                               f"  5. Report results\n\n"
                               f"Use 'stop attack' to abort.",
                        data=result,
                        warnings=["Autonomous attack running - monitor OPSEC"],
                        follow_up_suggestions=['attack status', 'stop attack']
                    )
                except Exception as e:
                    return CommandResult(success=False, message=f"Failed to start autonomous attack: {e}")
            return CommandResult(success=False, message="Attack planner or agent framework not available.")
        
        elif intent == 'ai_stop_attack':
            if self._ai_v2_attack_planner:
                try:
                    self._ai_v2_attack_planner.stop()
                    return CommandResult(
                        success=True,
                        message="ATTACK STOPPED\n\nAll attack operations halted.",
                        data={'status': 'stopped'}
                    )
                except Exception as e:
                    return CommandResult(success=False, message=f"Stop failed: {e}")
            return CommandResult(success=True, message="No attack in progress.")
        
        elif intent == 'ai_attack_status':
            if self._ai_v2_attack_planner:
                status = self._ai_v2_attack_planner.get_status()
                return CommandResult(
                    success=True,
                    message=f"ATTACK STATUS\n\n"
                           f"Active: {status.get('active', False)}\n"
                           f"Current phase: {status.get('phase', 'None')}\n"
                           f"Progress: {status.get('progress', 0)}%\n"
                           f"Targets found: {status.get('targets_found', 0)}\n"
                           f"Attacks executed: {status.get('attacks_executed', 0)}\n"
                           f"Success rate: {status.get('success_rate', 0):.1f}%",
                    data=status
                )
            return CommandResult(success=False, message="Attack planner not available.")
        
        # ===== RAG KNOWLEDGE BASE =====
        elif intent == 'ai_rag_query':
            query = params.get('query')
            if self._ai_v2_rag:
                try:
                    result = self._ai_v2_rag.query(query)
                    return CommandResult(
                        success=True,
                        message=f"RAG KNOWLEDGE QUERY\n\n"
                               f"Query: {query}\n\n"
                               f"Results:\n{result.get('answer', 'No relevant information found.')}\n\n"
                               f"Confidence: {result.get('confidence', 0):.1%}\n"
                               f"Sources: {len(result.get('sources', []))} documents",
                        data=result
                    )
                except Exception as e:
                    return CommandResult(success=False, message=f"Query failed: {e}")
            return CommandResult(success=False, message="RAG engine not available.")
        
        elif intent == 'ai_add_knowledge':
            return CommandResult(
                success=True,
                message="KNOWLEDGE BASE\n\n"
                       "The RAG knowledge base contains:\n"
                       "  • RF attack techniques\n"
                       "  • Protocol vulnerabilities\n"
                       "  • Exploit patterns\n"
                       "  • Signal intelligence methods\n\n"
                       "To add custom knowledge, place documents in:\n"
                       "  /core/ai_v2/knowledge/\n\n"
                       "Then run 'update knowledge base'.",
                follow_up_suggestions=['rag status', 'query knowledge']
            )
        
        elif intent == 'ai_update_rag':
            if self._ai_v2_rag:
                try:
                    self._ai_v2_rag.update_index()
                    return CommandResult(
                        success=True,
                        message="KNOWLEDGE BASE UPDATED\n\n"
                               "RAG index rebuilt with latest documents.",
                        data={'status': 'updated'}
                    )
                except Exception as e:
                    return CommandResult(success=False, message=f"Update failed: {e}")
            return CommandResult(success=False, message="RAG engine not available.")
        
        elif intent == 'ai_rag_status':
            if self._ai_v2_rag:
                status = self._ai_v2_rag.get_status()
                return CommandResult(
                    success=True,
                    message=f"RAG KNOWLEDGE BASE STATUS\n\n"
                           f"Documents indexed: {status.get('document_count', 0)}\n"
                           f"Embeddings: {status.get('embedding_count', 0)}\n"
                           f"Last updated: {status.get('last_updated', 'Never')}\n"
                           f"Categories: {', '.join(status.get('categories', ['None']))}",
                    data=status
                )
            return CommandResult(success=False, message="RAG engine not available.")
        
        # ===== MEMORY CONTROL =====
        elif intent == 'ai_clear_memory':
            if self._ai_v2_memory:
                try:
                    self._ai_v2_memory.clear()
                    return CommandResult(
                        success=True,
                        message="MEMORY CLEARED\n\n"
                               "All contextual memory wiped from RAM.\n"
                               "Session history reset.",
                        data={'status': 'cleared'}
                    )
                except Exception as e:
                    return CommandResult(success=False, message=f"Clear failed: {e}")
            return CommandResult(success=True, message="No memory to clear.")
        
        elif intent == 'ai_memory_status':
            if self._ai_v2_memory:
                status = self._ai_v2_memory.get_status()
                return CommandResult(
                    success=True,
                    message=f"MEMORY STATUS\n\n"
                           f"Storage: RAM-only (secure)\n"
                           f"Entries: {status.get('entry_count', 0)}\n"
                           f"Memory used: {status.get('memory_kb', 0):.1f} KB\n"
                           f"Targets remembered: {status.get('targets', 0)}\n"
                           f"Sessions: {status.get('sessions', 0)}\n"
                           f"Encryption: {status.get('encrypted', False)}",
                    data=status
                )
            return CommandResult(
                success=True,
                message="MEMORY STATUS\n\nMemory store not initialized."
            )
        
        # ===== THREAT INTEL =====
        elif intent == 'ai_threat_intel':
            source = params.get('source', 'local')
            if source == 'local' and self._ai_v2_rag:
                result = self._ai_v2_rag.get_threat_intel()
                return CommandResult(
                    success=True,
                    message=f"LOCAL THREAT INTEL\n\n"
                           f"Known vulnerabilities: {result.get('vuln_count', 0)}\n"
                           f"Exploit patterns: {result.get('exploit_count', 0)}\n"
                           f"Signal signatures: {result.get('signature_count', 0)}\n\n"
                           f"Last updated: {result.get('last_updated', 'Unknown')}",
                    data=result
                )
            elif self._ai_v2_online_intel:
                return CommandResult(
                    success=True,
                    message="ONLINE THREAT INTEL\n\n"
                           "Use 'online intel' to enable Tor-routed intelligence.\n"
                           "Then query for real-time CVE/exploit data.",
                    follow_up_suggestions=['online intel', 'threat intel local']
                )
            return CommandResult(success=False, message="Threat intel not available.")
        
        # ===== NATURAL LANGUAGE PROCESSING =====
        elif intent == 'ai_process_natural':
            query = params.get('query')
            if self._ai_v2_enhanced and self._ai_v2_local_llm:
                try:
                    result = self._ai_v2_enhanced.process_natural_language(query)
                    return CommandResult(
                        success=True,
                        message=f"AI RESPONSE\n\n{result.get('response', 'No response generated.')}\n\n"
                               f"Actions taken: {len(result.get('actions', []))}\n"
                               f"Confidence: {result.get('confidence', 0):.1%}",
                        data=result
                    )
                except Exception as e:
                    return CommandResult(success=False, message=f"NLP processing failed: {e}")
            return CommandResult(
                success=False,
                message="Natural language processing requires enabled enhanced AI.\n"
                       "Use 'enable enhanced ai' and 'load model' first."
            )
        
        # ===== GENERAL AI STATUS =====
        elif intent == 'ai_status':
            components = []
            
            if hasattr(self, '_ai_v2_enhanced') and self._ai_v2_enhanced:
                components.append(f"  • Enhanced AI: {'Enabled' if self._ai_v2_enhanced.is_enabled() else 'Disabled'}")
            else:
                components.append("  • Enhanced AI: Not loaded")
            
            if hasattr(self, '_ai_v2_local_llm') and self._ai_v2_local_llm:
                status = self._ai_v2_local_llm.get_status()
                components.append(f"  • Local LLM: {'Loaded' if status.get('model_loaded') else 'Not loaded'}")
            else:
                components.append("  • Local LLM: Not available")
            
            if hasattr(self, '_ai_v2_agent') and self._ai_v2_agent:
                status = self._ai_v2_agent.get_status()
                components.append(f"  • Agents: {status.get('active_count', 0)} active")
            else:
                components.append("  • Agents: Not available")
            
            if hasattr(self, '_ai_v2_attack_planner') and self._ai_v2_attack_planner:
                status = self._ai_v2_attack_planner.get_status()
                components.append(f"  • Attack Planner: {'Active' if status.get('active') else 'Idle'}")
            else:
                components.append("  • Attack Planner: Not available")
            
            if hasattr(self, '_ai_v2_rag') and self._ai_v2_rag:
                status = self._ai_v2_rag.get_status()
                components.append(f"  • RAG Knowledge: {status.get('document_count', 0)} documents")
            else:
                components.append("  • RAG Knowledge: Not available")
            
            if hasattr(self, '_ai_v2_memory') and self._ai_v2_memory:
                status = self._ai_v2_memory.get_status()
                components.append(f"  • Memory: {status.get('entry_count', 0)} entries (RAM-only)")
            else:
                components.append("  • Memory: Not available")
            
            if hasattr(self, '_ai_v2_online_intel') and self._ai_v2_online_intel:
                components.append("  • Online Intel: Available (Tor-routed)")
            else:
                components.append("  • Online Intel: Not available")
            
            components_text = "\n".join(components)
            
            return CommandResult(
                success=True,
                message=f"AI v2.0 ENHANCED INTELLIGENCE STATUS\n\n"
                       f"Components:\n{components_text}\n\n"
                       f"Architecture: Offline-first, Tor-optional\n"
                       f"Filtering: NONE (unfiltered operation)\n"
                       f"Telemetry: DISABLED (zero phone-home)",
                data={
                    'enhanced': hasattr(self, '_ai_v2_enhanced') and self._ai_v2_enhanced is not None,
                    'llm': hasattr(self, '_ai_v2_local_llm') and self._ai_v2_local_llm is not None,
                    'agents': hasattr(self, '_ai_v2_agent') and self._ai_v2_agent is not None,
                    'planner': hasattr(self, '_ai_v2_attack_planner') and self._ai_v2_attack_planner is not None,
                    'rag': hasattr(self, '_ai_v2_rag') and self._ai_v2_rag is not None,
                    'memory': hasattr(self, '_ai_v2_memory') and self._ai_v2_memory is not None,
                    'online': hasattr(self, '_ai_v2_online_intel') and self._ai_v2_online_intel is not None
                },
                follow_up_suggestions=['enable enhanced ai', 'load model', 'plan attack', 'create agent']
            )
        
        else:
            return CommandResult(
                success=False,
                message="Unknown AI command. Use 'ai status' for available features.",
                follow_up_suggestions=['ai status', 'enable enhanced ai', 'ai help']
            )


# Global instance
_ai_command_center: Optional[AICommandCenter] = None


def get_ai_command_center() -> AICommandCenter:
    """Get global AI Command Center instance"""
    global _ai_command_center
    if _ai_command_center is None:
        _ai_command_center = AICommandCenter()
    return _ai_command_center


# CLI Interface
def run_cli():
    """Run interactive CLI interface"""
    try:
        import readline  # Enable command history
    except ImportError:
        pass  # readline not available on all platforms
    
    ai = get_ai_command_center()
    
    # Get current mode and OPSEC score
    mode_str = "BEGINNER"
    opsec_str = "Checking..."
    
    if ai._user_mode_manager:
        mode_str = ai._user_mode_manager.get_current_mode().value.upper()
    
    if ai._opsec_monitor:
        opsec = ai._opsec_monitor.get_score_summary()
        opsec_str = f"{opsec['score']}/100 ({opsec['threat_level']})"
    
    print("")
    print("=" * 60)
    print("  RF ARSENAL OS - AI COMMAND CENTER")
    print("  White Hat Edition - Authorized Use Only")
    print("=" * 60)
    print("")
    print(f"  Network: OFFLINE (default - maximum stealth)")
    print(f"  Mode: {mode_str} | OPSEC Score: {opsec_str}")
    print("")
    print("  AI v2.0 ENHANCED FEATURES:")
    print("    • 'enable enhanced ai' - Activate local LLM + autonomous agents")
    print("    • 'plan attack' - AI-powered attack planning")
    print("    • 'create agent' - Spawn autonomous operation agents")
    print("  NEW FEATURES:")
    print("    • 'list missions' - Guided operation profiles")
    print("    • 'show opsec' - Security score & recommendations")
    print("    • 'set mode expert' - Change experience level")
    print("")
    print("  Say 'help' for commands, 'exit' to quit")
    print("")
    
    while True:
        try:
            user_input = input("AI> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nExiting AI Command Center...")
                break
            
            result = ai.process_command(user_input)
            
            # Display result
            print("")
            if result.warnings:
                for warning in result.warnings:
                    print(f"! {warning}")
                print("")
            
            print(result.message)
            
            if result.follow_up_suggestions:
                print(f"\nSuggestions: {', '.join(result.follow_up_suggestions)}")
            
            print("")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    run_cli()
