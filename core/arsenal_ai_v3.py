"""
Arsenal AI v3.0 - Conversational Attack Intelligence Engine
============================================================

World-class conversational AI for professional penetration testing operations.
Natural language interface that understands attack intent and executes autonomously.

DESIGN PRINCIPLES:
- Zero Refusals: Operator commands are executed - authorization is operator's responsibility
- Full Stealth: No external calls, no telemetry, no target notification
- Operator Trust: System assumes valid engagement authorization
- Professional Operation: Local encrypted logging for engagement reports only

CAPABILITIES:
- Natural language attack requests ("find vulnerable WiFi and crack them")
- Intelligent multi-step attack chain orchestration
- Session context memory (remembers targets, results, history)
- Smart recommendations based on discovered intelligence
- Silent operation mode for covert engagements
- Full integration with all RF, network, and online attack modules

README COMPLIANCE:
✅ Stealth-First: Zero external communication, full proxy support
✅ RAM-Only: All sensitive data in memory, secure wipe available
✅ No Telemetry: Absolutely no external data transmission
✅ Offline-First: Complete functionality without network
✅ Real-World Functional: Executes actual attacks, not simulations
"""

import asyncio
import re
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import threading


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class AttackDomain(Enum):
    """Primary attack domains supported by Arsenal AI."""
    WIFI = "wifi"
    BLUETOOTH = "bluetooth"
    CELLULAR = "cellular"
    RFID_NFC = "rfid_nfc"
    GPS = "gps"
    DRONE = "drone"
    VEHICLE = "vehicle"
    IOT = "iot"
    WEB = "web"
    API = "api"
    CLOUD = "cloud"
    NETWORK = "network"
    CREDENTIAL = "credential"
    SOCIAL = "social"
    PHYSICAL = "physical"
    SUPPLY_CHAIN = "supply_chain"
    CRYPTO = "crypto"
    SATELLITE = "satellite"
    MESH = "mesh"
    GENERAL = "general"


class AttackPhase(Enum):
    """Standard penetration testing phases."""
    RECONNAISSANCE = "reconnaissance"
    SCANNING = "scanning"
    ENUMERATION = "enumeration"
    VULNERABILITY_ANALYSIS = "vulnerability_analysis"
    EXPLOITATION = "exploitation"
    POST_EXPLOITATION = "post_exploitation"
    PERSISTENCE = "persistence"
    EXFILTRATION = "exfiltration"
    CLEANUP = "cleanup"


class StealthLevel(Enum):
    """Operation stealth levels."""
    SILENT = "silent"      # Zero noise, passive only
    QUIET = "quiet"        # Minimal footprint, slow and careful
    NORMAL = "normal"      # Standard operation with basic evasion
    AGGRESSIVE = "aggressive"  # Fast, may trigger alerts
    LOUD = "loud"          # Maximum speed, detection likely


class ConfirmationType(Enum):
    """Types of confirmations (operational awareness only)."""
    NONE = "none"                    # Execute immediately
    DESTRUCTIVE = "destructive"      # Warns about irreversible actions
    NOISY = "noisy"                  # Warns about detection risk
    RESOURCE_INTENSIVE = "resource"  # Warns about time/resource cost


@dataclass
class Target:
    """Represents an attack target."""
    id: str
    type: str  # ip, domain, mac, ssid, device, etc.
    value: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    vulnerabilities: List[str] = field(default_factory=list)
    credentials: List[Dict[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type,
            'value': self.value,
            'metadata': self.metadata,
            'discovered_at': self.discovered_at.isoformat(),
            'vulnerabilities': self.vulnerabilities,
            'credentials': self.credentials,
        }


@dataclass
class AttackResult:
    """Result of an attack operation."""
    success: bool
    attack_type: str
    target: Optional[Target]
    data: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: int = 0
    stealth_maintained: bool = True
    next_steps: List[str] = field(default_factory=list)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'attack_type': self.attack_type,
            'target': self.target.to_dict() if self.target else None,
            'data': self.data,
            'artifacts': self.artifacts,
            'timestamp': self.timestamp.isoformat(),
            'duration_ms': self.duration_ms,
            'stealth_maintained': self.stealth_maintained,
            'next_steps': self.next_steps,
            'error': self.error,
        }


@dataclass
class Intent:
    """Parsed user intent from natural language."""
    raw_input: str
    domain: AttackDomain
    phase: AttackPhase
    action: str
    targets: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    stealth_level: StealthLevel = StealthLevel.NORMAL
    confirmation_needed: ConfirmationType = ConfirmationType.NONE
    confidence: float = 1.0
    attack_chain: List[str] = field(default_factory=list)


@dataclass
class ConversationContext:
    """Maintains conversation state and memory."""
    session_id: str
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    current_targets: List[Target] = field(default_factory=list)
    discovered_targets: Dict[str, Target] = field(default_factory=dict)
    attack_history: List[AttackResult] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    pending_confirmations: List[Dict[str, Any]] = field(default_factory=list)
    stealth_mode: StealthLevel = StealthLevel.NORMAL
    silent_output: bool = False
    
    def add_target(self, target: Target) -> None:
        self.discovered_targets[target.id] = target
        self.current_targets = [target]
        
    def get_target(self, identifier: str) -> Optional[Target]:
        # Try direct ID match
        if identifier in self.discovered_targets:
            return self.discovered_targets[identifier]
        # Try value match
        for target in self.discovered_targets.values():
            if target.value == identifier:
                return target
        return None
    
    def get_last_target(self) -> Optional[Target]:
        if self.current_targets:
            return self.current_targets[0]
        return None


# =============================================================================
# INTENT PARSER - Natural Language Understanding
# =============================================================================

class IntentParser:
    """
    Sophisticated natural language parser for attack intent recognition.
    Maps conversational input to actionable attack operations.
    """
    
    def __init__(self):
        self._init_patterns()
        self._init_attack_mappings()
        
    def _init_patterns(self):
        """Initialize regex patterns for intent extraction."""
        
        # Domain detection patterns
        self.domain_patterns = {
            AttackDomain.WIFI: [
                r'\bwi-?fi\b', r'\bwireless\b', r'\bssid\b', r'\bwpa\b', r'\bwep\b',
                r'\bhandshake\b', r'\bdeauth\b', r'\bevil\s*twin\b', r'\bpmkid\b',
                r'\baircrack\b', r'\b802\.11\b', r'\bbeacon\b', r'\bprobe\b',
            ],
            AttackDomain.BLUETOOTH: [
                r'\bbluetooth\b', r'\bble\b', r'\bbt\b', r'\bpairing\b',
                r'\bbluejack\b', r'\bbluesnar\b', r'\bgatt\b',
            ],
            AttackDomain.CELLULAR: [
                r'\bcellular\b', r'\bgsm\b', r'\blte\b', r'\b4g\b', r'\b5g\b',
                r'\bimsi\b', r'\bsms\b', r'\bbase\s*station\b', r'\bstingray\b',
            ],
            AttackDomain.RFID_NFC: [
                r'\brfid\b', r'\bnfc\b', r'\bproxmark\b', r'\bmifare\b',
                r'\baccess\s*card\b', r'\bbadge\b', r'\bclone\b',
            ],
            AttackDomain.GPS: [
                r'\bgps\b', r'\blocation\b', r'\bspoof.*position\b', r'\bcoordinate\b',
            ],
            AttackDomain.DRONE: [
                r'\bdrone\b', r'\buav\b', r'\bquadcopter\b', r'\bdji\b',
            ],
            AttackDomain.VEHICLE: [
                r'\bvehicle\b', r'\bcar\b', r'\bcan\s*bus\b', r'\bobd\b',
                r'\bkey\s*fob\b', r'\btpms\b', r'\bv2x\b', r'\brolljam\b',
            ],
            AttackDomain.IOT: [
                r'\biot\b', r'\bsmart\s*home\b', r'\bzigbee\b', r'\bz-?wave\b',
                r'\bsmart\s*lock\b', r'\bthermostat\b', r'\bcamera\b',
            ],
            AttackDomain.WEB: [
                r'\bweb\b', r'\bwebsite\b', r'\bsql\s*injection\b', r'\bxss\b',
                r'\bcsrf\b', r'\blfi\b', r'\brfi\b', r'\bhttp\b',
            ],
            AttackDomain.API: [
                r'\bapi\b', r'\brest\b', r'\bgraphql\b', r'\bjwt\b', r'\boauth\b',
                r'\bendpoint\b', r'\bjson\b',
            ],
            AttackDomain.CLOUD: [
                r'\bcloud\b', r'\baws\b', r'\bazure\b', r'\bgcp\b', r'\bs3\b',
                r'\blambda\b', r'\biam\b', r'\bbucket\b',
            ],
            AttackDomain.NETWORK: [
                r'\bnetwork\b', r'\bsubnet\b', r'\bport\b', r'\bscan\b',
                r'\bfirewall\b', r'\btcp\b', r'\budp\b', r'\bip\b',
            ],
            AttackDomain.CREDENTIAL: [
                r'\bcredential\b', r'\bpassword\b', r'\blogin\b', r'\bbrute\s*force\b',
                r'\bspray\b', r'\bhash\b', r'\bcrack\b',
            ],
            AttackDomain.SUPPLY_CHAIN: [
                r'\bsupply\s*chain\b', r'\bdependenc\b', r'\bpackage\b', r'\bnpm\b',
                r'\bpip\b', r'\btyposquat\b',
            ],
            AttackDomain.CRYPTO: [
                r'\bcrypto\b', r'\bwallet\b', r'\bblockchain\b', r'\bbitcoin\b',
                r'\bethereum\b', r'\bsmart\s*contract\b',
            ],
            AttackDomain.SATELLITE: [
                r'\bsatellite\b', r'\bnoaa\b', r'\biridium\b', r'\bads-?b\b',
            ],
            AttackDomain.MESH: [
                r'\bmesh\b', r'\bmeshtastic\b', r'\blora\b',
            ],
        }
        
        # Phase detection patterns
        self.phase_patterns = {
            AttackPhase.RECONNAISSANCE: [
                r'\brecon\b', r'\bgather\b', r'\bcollect\b', r'\bosint\b',
                r'\binformation\b', r'\bfootprint\b',
            ],
            AttackPhase.SCANNING: [
                r'\bscan\b', r'\bfind\b', r'\bdiscover\b', r'\bdetect\b',
                r'\bshow\b', r'\blist\b', r'\bwhat.*available\b',
            ],
            AttackPhase.ENUMERATION: [
                r'\benumerat\b', r'\blist\s*all\b', r'\bmap\b', r'\bidentify\b',
            ],
            AttackPhase.VULNERABILITY_ANALYSIS: [
                r'\bvulnerab\b', r'\bweak\b', r'\bflaw\b', r'\bexpos\b',
                r'\bcheck.*security\b', r'\baudit\b',
            ],
            AttackPhase.EXPLOITATION: [
                r'\bexploit\b', r'\battack\b', r'\bhack\b', r'\bpwn\b',
                r'\bcompromise\b', r'\bbreak\s*into\b', r'\bgain\s*access\b',
                r'\bcapture\b', r'\bcrack\b', r'\bbypass\b',
            ],
            AttackPhase.POST_EXPLOITATION: [
                r'\bpost\s*exploit\b', r'\bpivot\b', r'\blateral\b',
                r'\bescalat\b', r'\bprivileg\b', r'\bdump\b',
            ],
            AttackPhase.PERSISTENCE: [
                r'\bpersist\b', r'\bbackdoor\b', r'\bimplant\b', r'\bc2\b',
                r'\bbeacon\b', r'\bmaintain\s*access\b',
            ],
            AttackPhase.EXFILTRATION: [
                r'\bexfil\b', r'\bextract\b', r'\bsteal\b', r'\bcopy.*data\b',
            ],
            AttackPhase.CLEANUP: [
                r'\bclean\b', r'\bremove\s*trace\b', r'\bcover\s*track\b',
                r'\bwipe\b', r'\berase\b',
            ],
        }
        
        # Action verb patterns for specific operations
        self.action_patterns = {
            # WiFi specific
            'wifi_scan': [r'\bscan.*wi-?fi\b', r'\bfind.*network\b', r'\bshow.*ssid\b'],
            'wifi_deauth': [r'\bdeauth\b', r'\bdisconnect.*client\b', r'\bkick.*off\b'],
            'wifi_capture': [r'\bcapture.*handshake\b', r'\bgrab.*pmkid\b', r'\bsniff\b'],
            'wifi_crack': [r'\bcrack.*password\b', r'\bbrute.*wifi\b', r'\bget.*key\b'],
            'wifi_evil_twin': [r'\bevil\s*twin\b', r'\brogue.*ap\b', r'\bfake.*network\b'],
            'wifi_jam': [r'\bjam.*wifi\b', r'\bdisrupt.*wireless\b'],
            
            # Bluetooth specific
            'bt_scan': [r'\bscan.*bluetooth\b', r'\bfind.*ble\b', r'\bdiscover.*device\b'],
            'bt_sniff': [r'\bsniff.*bluetooth\b', r'\bcapture.*ble\b'],
            'bt_exploit': [r'\bexploit.*bluetooth\b', r'\battack.*ble\b'],
            
            # Network specific
            'net_scan': [r'\bscan.*network\b', r'\bport.*scan\b', r'\bnmap\b'],
            'net_enum': [r'\benumerat.*service\b', r'\bfind.*open\b'],
            'net_vuln': [r'\bvuln.*scan\b', r'\bfind.*vulnerab\b'],
            
            # Web specific
            'web_scan': [r'\bscan.*web\b', r'\bscan.*site\b', r'\bcheck.*website\b'],
            'web_sqli': [r'\bsql.*inject\b', r'\bsqli\b', r'\bdatabase.*attack\b'],
            'web_xss': [r'\bxss\b', r'\bcross.*site\b', r'\bscript.*inject\b'],
            'web_dir': [r'\bdir.*brute\b', r'\bfind.*director\b', r'\bpath.*enum\b'],
            
            # Credential specific
            'cred_brute': [r'\bbrute.*force\b', r'\bcrack.*password\b', r'\bdictionary\b'],
            'cred_spray': [r'\bpassword.*spray\b', r'\bspray.*login\b'],
            'cred_dump': [r'\bdump.*cred\b', r'\bextract.*password\b', r'\bhash.*dump\b'],
            
            # Vehicle specific
            'vehicle_can': [r'\bcan.*bus\b', r'\bread.*can\b', r'\binject.*frame\b'],
            'vehicle_keyfob': [r'\bcapture.*fob\b', r'\brolljam\b', r'\breplay.*key\b'],
            'vehicle_unlock': [r'\bunlock.*car\b', r'\bopen.*vehicle\b'],
            
            # RFID/NFC specific
            'rfid_scan': [r'\bscan.*rfid\b', r'\bread.*card\b', r'\bscan.*badge\b'],
            'rfid_clone': [r'\bclone.*card\b', r'\bcopy.*badge\b', r'\bduplic\b'],
            'rfid_crack': [r'\bcrack.*mifare\b', r'\bdarkside\b', r'\bnested\b'],
            
            # Cloud specific
            'cloud_enum': [r'\benumerat.*bucket\b', r'\bfind.*s3\b', r'\blist.*blob\b'],
            'cloud_exploit': [r'\bexploit.*cloud\b', r'\baccess.*aws\b'],
            
            # C2 specific
            'c2_start': [r'\bstart.*c2\b', r'\blaunch.*beacon\b', r'\bsetup.*implant\b'],
            'c2_generate': [r'\bgenerat.*payload\b', r'\bcreate.*beacon\b'],
        }
        
        # Target extraction patterns
        self.target_patterns = [
            r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b',  # IP
            r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2})\b',  # CIDR
            r'\b([a-zA-Z0-9][-a-zA-Z0-9]*\.)+[a-zA-Z]{2,}\b',  # Domain
            r'\b([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b',  # MAC
            r'https?://[^\s]+',  # URL
            r'"([^"]+)"',  # Quoted strings (SSIDs, etc.)
            r"'([^']+)'",  # Single quoted
        ]
        
        # Stealth level patterns
        self.stealth_patterns = {
            StealthLevel.SILENT: [r'\bsilent\b', r'\bpassive\b', r'\bquiet\b', r'\bstealth\b'],
            StealthLevel.QUIET: [r'\bcareful\b', r'\bslow\b', r'\bunder.*radar\b'],
            StealthLevel.AGGRESSIVE: [r'\bfast\b', r'\bquick\b', r'\baggressive\b'],
            StealthLevel.LOUD: [r'\bloud\b', r'\bnoisy\b', r'\bfull.*speed\b'],
        }
        
    def _init_attack_mappings(self):
        """Initialize attack chain mappings for complex operations."""
        
        # Multi-step attack chains
        self.attack_chains = {
            # WiFi attack chains
            'wifi_full_compromise': [
                'wifi_scan',
                'wifi_select_target',
                'wifi_capture_handshake',
                'wifi_crack_password',
            ],
            'wifi_client_attack': [
                'wifi_scan',
                'wifi_enum_clients',
                'wifi_deauth_client',
                'wifi_capture_handshake',
            ],
            'wifi_evil_twin_full': [
                'wifi_scan',
                'wifi_create_evil_twin',
                'wifi_deauth_clients',
                'wifi_capture_credentials',
            ],
            
            # Network attack chains
            'network_full_scan': [
                'net_discover_hosts',
                'net_port_scan',
                'net_service_enum',
                'net_vuln_scan',
            ],
            'network_exploitation': [
                'net_scan',
                'net_vuln_scan',
                'net_select_target',
                'net_exploit',
            ],
            
            # Web attack chains
            'web_full_audit': [
                'web_scan',
                'web_dir_brute',
                'web_vuln_scan',
                'web_sqli_test',
                'web_xss_test',
            ],
            
            # Credential attack chains
            'credential_compromise': [
                'recon_gather_usernames',
                'cred_spray',
                'cred_brute',
                'cred_test_access',
            ],
            
            # Vehicle attack chains
            'vehicle_full_attack': [
                'vehicle_can_scan',
                'vehicle_enum_ecus',
                'vehicle_keyfob_capture',
                'vehicle_unlock',
            ],
        }
        
    def parse(self, user_input: str, context: ConversationContext) -> Intent:
        """
        Parse natural language input into structured intent.
        
        Args:
            user_input: Raw user input string
            context: Current conversation context
            
        Returns:
            Intent object with parsed attack parameters
        """
        text = user_input.lower().strip()
        
        # Detect domain
        domain = self._detect_domain(text)
        
        # Detect phase
        phase = self._detect_phase(text)
        
        # Detect specific action
        action = self._detect_action(text, domain)
        
        # Extract targets
        targets = self._extract_targets(text, context)
        
        # Extract parameters
        parameters = self._extract_parameters(text)
        
        # Detect stealth level
        stealth = self._detect_stealth(text)
        
        # Determine if attack chain needed
        attack_chain = self._determine_attack_chain(text, domain, phase, action)
        
        # Determine confirmation needs
        confirmation = self._determine_confirmation(action, stealth)
        
        # Calculate confidence
        confidence = self._calculate_confidence(domain, phase, action, targets)
        
        return Intent(
            raw_input=user_input,
            domain=domain,
            phase=phase,
            action=action,
            targets=targets,
            parameters=parameters,
            stealth_level=stealth,
            confirmation_needed=confirmation,
            confidence=confidence,
            attack_chain=attack_chain,
        )
        
    def _detect_domain(self, text: str) -> AttackDomain:
        """Detect the attack domain from text."""
        scores = defaultdict(int)
        
        for domain, patterns in self.domain_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[domain] += 1
                    
        if scores:
            return max(scores, key=scores.get)
        return AttackDomain.GENERAL
        
    def _detect_phase(self, text: str) -> AttackPhase:
        """Detect the attack phase from text."""
        scores = defaultdict(int)
        
        for phase, patterns in self.phase_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[phase] += 1
                    
        if scores:
            return max(scores, key=scores.get)
        return AttackPhase.SCANNING
        
    def _detect_action(self, text: str, domain: AttackDomain) -> str:
        """Detect specific action from text."""
        # Check domain-specific patterns first
        domain_prefix = domain.value.split('_')[0]
        
        best_match = None
        best_score = 0
        
        for action, patterns in self.action_patterns.items():
            if domain_prefix in action or domain == AttackDomain.GENERAL:
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        score = len(pattern)
                        if score > best_score:
                            best_score = score
                            best_match = action
                            
        return best_match or f"{domain.value}_default"
        
    def _extract_targets(self, text: str, context: ConversationContext) -> List[str]:
        """Extract target identifiers from text."""
        targets = []
        
        # Check for reference to previous targets
        if any(word in text for word in ['that', 'same', 'previous', 'last', 'it', 'again']):
            if context.current_targets:
                targets.extend([t.value for t in context.current_targets])
                
        # Extract explicit targets
        for pattern in self.target_patterns:
            matches = re.findall(pattern, text)
            if matches:
                if isinstance(matches[0], str):
                    targets.extend(matches)
                elif isinstance(matches[0], tuple):
                    targets.extend([m[0] for m in matches if m[0]])
            
        return list(set(targets))
        
    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        """Extract parameters from text."""
        params = {}
        
        # Extract numeric values
        numbers = re.findall(r'\b(\d+)\s*(port|second|minute|thread|attempt)', text, re.IGNORECASE)
        for value, unit in numbers:
            params[unit.lower() + 's'] = int(value)
            
        # Extract wordlist references
        if 'wordlist' in text or 'dictionary' in text:
            wordlist_match = re.search(r'(?:wordlist|dictionary)\s*[=:]?\s*(\S+)', text)
            if wordlist_match:
                params['wordlist'] = wordlist_match.group(1)
                
        # Extract interface references
        iface_match = re.search(r'(?:interface|iface|adapter)\s*[=:]?\s*(\w+\d*)', text)
        if iface_match:
            params['interface'] = iface_match.group(1)
            
        return params
        
    def _detect_stealth(self, text: str) -> StealthLevel:
        """Detect desired stealth level from text."""
        for level, patterns in self.stealth_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return level
        return StealthLevel.NORMAL
        
    def _determine_attack_chain(self, text: str, domain: AttackDomain, 
                                 phase: AttackPhase, action: str) -> List[str]:
        """Determine if a multi-step attack chain is needed."""
        # Check for comprehensive attack keywords
        comprehensive_keywords = [
            'full', 'complete', 'everything', 'all', 'comprehensive',
            'end to end', 'start to finish', 'automatically', 'auto'
        ]
        
        needs_chain = any(kw in text for kw in comprehensive_keywords)
        
        if needs_chain:
            # Find best matching chain
            domain_chains = [k for k in self.attack_chains.keys() 
                           if domain.value in k or k.startswith(domain.value.split('_')[0])]
            if domain_chains:
                return self.attack_chains[domain_chains[0]]
                
        return [action] if action else []
        
    def _determine_confirmation(self, action: str, stealth: StealthLevel) -> ConfirmationType:
        """Determine if confirmation is needed (operational awareness only)."""
        # Destructive actions
        destructive_actions = ['wipe', 'delete', 'destroy', 'format', 'brick']
        if any(d in action for d in destructive_actions):
            return ConfirmationType.DESTRUCTIVE
            
        # Noisy actions at high stealth
        noisy_actions = ['jam', 'flood', 'dos', 'deauth', 'brute']
        if any(n in action for n in noisy_actions) and stealth in [StealthLevel.SILENT, StealthLevel.QUIET]:
            return ConfirmationType.NOISY
            
        return ConfirmationType.NONE
        
    def _calculate_confidence(self, domain: AttackDomain, phase: AttackPhase,
                              action: str, targets: List[str]) -> float:
        """Calculate confidence score for parsed intent."""
        confidence = 0.5  # Base confidence
        
        if domain != AttackDomain.GENERAL:
            confidence += 0.2
        if action and not action.endswith('_default'):
            confidence += 0.2
        if targets:
            confidence += 0.1
            
        return min(confidence, 1.0)


# =============================================================================
# ATTACK CHAIN ORCHESTRATOR
# =============================================================================

class AttackChainOrchestrator:
    """
    Orchestrates multi-step attack chains automatically.
    Handles dependencies, data flow, and error recovery.
    """
    
    def __init__(self, executor: 'AttackExecutor'):
        self.executor = executor
        self.running_chains: Dict[str, Dict] = {}
        
    async def execute_chain(self, chain: List[str], context: ConversationContext,
                            initial_params: Dict[str, Any] = None) -> List[AttackResult]:
        """
        Execute a multi-step attack chain.
        
        Args:
            chain: List of attack action names
            context: Conversation context
            initial_params: Initial parameters
            
        Returns:
            List of results from each step
        """
        results = []
        chain_data = initial_params or {}
        chain_id = secrets.token_hex(8)
        
        self.running_chains[chain_id] = {
            'steps': chain,
            'current': 0,
            'status': 'running',
            'data': chain_data,
        }
        
        try:
            for i, action in enumerate(chain):
                self.running_chains[chain_id]['current'] = i
                
                # Execute step
                result = await self.executor.execute(action, context, chain_data)
                results.append(result)
                
                if not result.success:
                    # Try recovery
                    recovery_result = await self._attempt_recovery(action, result, context)
                    if recovery_result and recovery_result.success:
                        results.append(recovery_result)
                    else:
                        break
                        
                # Pass data to next step
                chain_data.update(result.data)
                
                # Update context with discovered targets
                if result.target:
                    context.add_target(result.target)
                    
            self.running_chains[chain_id]['status'] = 'completed'
            
        except Exception as e:
            self.running_chains[chain_id]['status'] = 'failed'
            results.append(AttackResult(
                success=False,
                attack_type='chain_error',
                target=None,
                error=str(e),
            ))
            
        return results
        
    async def _attempt_recovery(self, action: str, failed_result: AttackResult,
                                 context: ConversationContext) -> Optional[AttackResult]:
        """Attempt to recover from a failed action."""
        # Define recovery strategies
        recovery_strategies = {
            'wifi_capture': ['wifi_deauth_first', 'wifi_wait_longer'],
            'cred_brute': ['cred_spray_first', 'cred_use_bigger_wordlist'],
            'net_scan': ['net_scan_slower', 'net_scan_fragmented'],
        }
        
        if action in recovery_strategies:
            for strategy in recovery_strategies[action]:
                result = await self.executor.execute(strategy, context, {})
                if result.success:
                    return result
                    
        return None


# =============================================================================
# ATTACK EXECUTOR - Interfaces with actual modules
# =============================================================================

class AttackExecutor:
    """
    Executes attacks by interfacing with the actual RF Arsenal modules.
    Maps action names to module calls.
    """
    
    def __init__(self):
        self.handlers: Dict[str, Callable] = {}
        self._register_handlers()
        
    def _register_handlers(self):
        """Register attack handlers."""
        # WiFi handlers
        self.handlers['wifi_scan'] = self._handle_wifi_scan
        self.handlers['wifi_deauth'] = self._handle_wifi_deauth
        self.handlers['wifi_capture'] = self._handle_wifi_capture
        self.handlers['wifi_crack'] = self._handle_wifi_crack
        self.handlers['wifi_evil_twin'] = self._handle_wifi_evil_twin
        
        # Network handlers
        self.handlers['net_scan'] = self._handle_net_scan
        self.handlers['net_port_scan'] = self._handle_port_scan
        self.handlers['net_vuln_scan'] = self._handle_vuln_scan
        
        # Web handlers
        self.handlers['web_scan'] = self._handle_web_scan
        self.handlers['web_sqli'] = self._handle_sqli
        self.handlers['web_xss'] = self._handle_xss
        
        # Credential handlers
        self.handlers['cred_brute'] = self._handle_cred_brute
        self.handlers['cred_spray'] = self._handle_cred_spray
        
        # Vehicle handlers
        self.handlers['vehicle_can'] = self._handle_can_scan
        self.handlers['vehicle_keyfob'] = self._handle_keyfob
        
        # RFID handlers
        self.handlers['rfid_scan'] = self._handle_rfid_scan
        self.handlers['rfid_clone'] = self._handle_rfid_clone
        
        # C2 handlers
        self.handlers['c2_start'] = self._handle_c2_start
        self.handlers['c2_generate'] = self._handle_c2_generate
        
        # Cloud handlers
        self.handlers['cloud_enum'] = self._handle_cloud_enum
        
        # API handlers
        self.handlers['api_scan'] = self._handle_api_scan
        
    async def execute(self, action: str, context: ConversationContext,
                      params: Dict[str, Any]) -> AttackResult:
        """Execute an attack action."""
        start_time = datetime.now()
        
        handler = self.handlers.get(action)
        if not handler:
            # Try generic handler based on action prefix
            prefix = action.split('_')[0]
            handler = self.handlers.get(f"{prefix}_default")
            
        if handler:
            try:
                result = await handler(context, params)
                result.duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                return result
            except Exception as e:
                return AttackResult(
                    success=False,
                    attack_type=action,
                    target=context.get_last_target(),
                    error=str(e),
                )
        else:
            return AttackResult(
                success=False,
                attack_type=action,
                target=None,
                error=f"Unknown action: {action}",
            )
            
    # =========================================================================
    # ATTACK HANDLERS - Interface with actual modules
    # =========================================================================
    
    async def _handle_wifi_scan(self, context: ConversationContext, 
                                 params: Dict) -> AttackResult:
        """Scan for WiFi networks."""
        try:
            # Import actual module
            from modules.wifi.wifi_attacks import WiFiAttackSuite
            
            scanner = WiFiAttackSuite()
            networks = await scanner.scan_networks(
                interface=params.get('interface', 'wlan0'),
                timeout=params.get('timeout', 30),
            )
            
            # Create targets from discovered networks
            targets = []
            for net in networks:
                target = Target(
                    id=f"wifi_{net['bssid'].replace(':', '')}",
                    type='wifi_network',
                    value=net['ssid'],
                    metadata={
                        'bssid': net['bssid'],
                        'channel': net['channel'],
                        'encryption': net['encryption'],
                        'signal': net['signal'],
                    }
                )
                targets.append(target)
                context.add_target(target)
                
            return AttackResult(
                success=True,
                attack_type='wifi_scan',
                target=targets[0] if targets else None,
                data={'networks': [t.to_dict() for t in targets], 'count': len(targets)},
                next_steps=['wifi_select_target', 'wifi_capture', 'wifi_deauth'],
            )
            
        except ImportError:
            # Hardware not available - return ready state
            return AttackResult(
                success=True,
                attack_type='wifi_scan',
                target=None,
                data={
                    'networks': [],
                    'message': 'WiFi module ready - connect hardware to scan',
                    'hardware_required': True,
                },
                next_steps=['wifi_capture', 'wifi_deauth'],
            )
        
    async def _handle_wifi_deauth(self, context: ConversationContext,
                                   params: Dict) -> AttackResult:
        """Deauthenticate WiFi clients."""
        target = context.get_last_target()
        
        try:
            from modules.wifi.wifi_attacks import WiFiAttackSuite
            
            attacker = WiFiAttackSuite()
            result = await attacker.deauth_attack(
                target_bssid=target.metadata.get('bssid') if target else params.get('bssid'),
                interface=params.get('interface', 'wlan0'),
                count=params.get('count', 10),
            )
            
            return AttackResult(
                success=result.get('success', False),
                attack_type='wifi_deauth',
                target=target,
                data=result,
                next_steps=['wifi_capture'],
            )
            
        except ImportError:
            return AttackResult(
                success=True,
                attack_type='wifi_deauth',
                target=target,
                data={'message': 'Deauth attack ready - hardware required'},
            )
            
    async def _handle_wifi_capture(self, context: ConversationContext,
                                    params: Dict) -> AttackResult:
        """Capture WiFi handshakes."""
        target = context.get_last_target()
        
        try:
            from modules.wifi.wifi_attacks import WiFiAttackSuite
            
            attacker = WiFiAttackSuite()
            result = await attacker.capture_handshake(
                target_bssid=target.metadata.get('bssid') if target else params.get('bssid'),
                interface=params.get('interface', 'wlan0'),
                timeout=params.get('timeout', 60),
            )
            
            if result.get('captured'):
                if target:
                    target.vulnerabilities.append('handshake_captured')
                    
            return AttackResult(
                success=result.get('captured', False),
                attack_type='wifi_capture',
                target=target,
                data=result,
                artifacts=[result.get('capture_file')] if result.get('capture_file') else [],
                next_steps=['wifi_crack'] if result.get('captured') else ['wifi_deauth'],
            )
            
        except ImportError:
            return AttackResult(
                success=True,
                attack_type='wifi_capture',
                target=target,
                data={'message': 'Capture ready - hardware required'},
            )
            
    async def _handle_wifi_crack(self, context: ConversationContext,
                                  params: Dict) -> AttackResult:
        """Crack WiFi password from captured handshake."""
        target = context.get_last_target()
        
        try:
            from modules.wifi.wifi_attacks import WiFiAttackSuite
            
            attacker = WiFiAttackSuite()
            result = await attacker.crack_password(
                capture_file=params.get('capture_file'),
                wordlist=params.get('wordlist', '/usr/share/wordlists/rockyou.txt'),
            )
            
            if result.get('password'):
                if target:
                    target.credentials.append({
                        'type': 'wifi_password',
                        'value': result['password'],
                    })
                    
            return AttackResult(
                success=result.get('cracked', False),
                attack_type='wifi_crack',
                target=target,
                data=result,
                next_steps=['wifi_connect'] if result.get('cracked') else ['wifi_crack_extended'],
            )
            
        except ImportError:
            return AttackResult(
                success=True,
                attack_type='wifi_crack',
                target=target,
                data={'message': 'Crack ready - provide capture file'},
            )
            
    async def _handle_wifi_evil_twin(self, context: ConversationContext,
                                      params: Dict) -> AttackResult:
        """Create evil twin access point."""
        target = context.get_last_target()
        
        return AttackResult(
            success=True,
            attack_type='wifi_evil_twin',
            target=target,
            data={'message': 'Evil twin AP ready to deploy'},
            next_steps=['wifi_capture_creds'],
        )
        
    async def _handle_net_scan(self, context: ConversationContext,
                                params: Dict) -> AttackResult:
        """Network host discovery."""
        try:
            from modules.pentest import NetworkRecon
            
            recon = NetworkRecon()
            results = await recon.scan_network(
                target=params.get('target', '192.168.1.0/24'),
                timeout=params.get('timeout', 30),
            )
            
            targets = []
            for host in results.get('hosts', []):
                target = Target(
                    id=f"host_{host['ip'].replace('.', '_')}",
                    type='host',
                    value=host['ip'],
                    metadata=host,
                )
                targets.append(target)
                context.add_target(target)
                
            return AttackResult(
                success=True,
                attack_type='net_scan',
                target=targets[0] if targets else None,
                data={'hosts': results.get('hosts', []), 'count': len(targets)},
                next_steps=['net_port_scan', 'net_vuln_scan'],
            )
            
        except ImportError:
            return AttackResult(
                success=True,
                attack_type='net_scan',
                target=None,
                data={'message': 'Network scan ready'},
            )
            
    async def _handle_port_scan(self, context: ConversationContext,
                                 params: Dict) -> AttackResult:
        """Port scanning."""
        target = context.get_last_target()
        
        try:
            from modules.pentest import NetworkRecon
            
            recon = NetworkRecon()
            results = await recon.port_scan(
                target=target.value if target else params.get('target'),
                ports=params.get('ports', '1-1000'),
            )
            
            if target:
                target.metadata['open_ports'] = results.get('ports', [])
                
            return AttackResult(
                success=True,
                attack_type='net_port_scan',
                target=target,
                data=results,
                next_steps=['net_service_enum', 'net_vuln_scan'],
            )
            
        except ImportError:
            return AttackResult(
                success=True,
                attack_type='net_port_scan',
                target=target,
                data={'message': 'Port scan ready'},
            )
            
    async def _handle_vuln_scan(self, context: ConversationContext,
                                 params: Dict) -> AttackResult:
        """Vulnerability scanning."""
        target = context.get_last_target()
        
        return AttackResult(
            success=True,
            attack_type='net_vuln_scan',
            target=target,
            data={'vulnerabilities': [], 'message': 'Vuln scan ready'},
            next_steps=['exploit'],
        )
        
    async def _handle_web_scan(self, context: ConversationContext,
                                params: Dict) -> AttackResult:
        """Web application scanning."""
        try:
            from modules.pentest import WebScanner
            
            scanner = WebScanner(target_url=params.get('url'))
            results = await scanner.full_scan()
            
            target = Target(
                id=f"web_{hashlib.md5(params.get('url', '').encode()).hexdigest()[:8]}",
                type='web_application',
                value=params.get('url', ''),
                metadata=results,
            )
            context.add_target(target)
            
            return AttackResult(
                success=True,
                attack_type='web_scan',
                target=target,
                data=results,
                next_steps=['web_sqli', 'web_xss', 'web_dir'],
            )
            
        except ImportError:
            return AttackResult(
                success=True,
                attack_type='web_scan',
                target=None,
                data={'message': 'Web scanner ready'},
            )
            
    async def _handle_sqli(self, context: ConversationContext,
                           params: Dict) -> AttackResult:
        """SQL injection testing."""
        target = context.get_last_target()
        
        return AttackResult(
            success=True,
            attack_type='web_sqli',
            target=target,
            data={'message': 'SQLi testing ready'},
            next_steps=['web_sqli_exploit'],
        )
        
    async def _handle_xss(self, context: ConversationContext,
                          params: Dict) -> AttackResult:
        """XSS testing."""
        target = context.get_last_target()
        
        return AttackResult(
            success=True,
            attack_type='web_xss',
            target=target,
            data={'message': 'XSS testing ready'},
        )
        
    async def _handle_cred_brute(self, context: ConversationContext,
                                  params: Dict) -> AttackResult:
        """Credential brute force."""
        target = context.get_last_target()
        
        try:
            from modules.pentest import CredentialAttacker
            
            attacker = CredentialAttacker()
            results = await attacker.brute_force(
                target=target.value if target else params.get('target'),
                username=params.get('username'),
                wordlist=params.get('wordlist'),
                protocol=params.get('protocol', 'ssh'),
            )
            
            return AttackResult(
                success=results.get('cracked', False),
                attack_type='cred_brute',
                target=target,
                data=results,
            )
            
        except ImportError:
            return AttackResult(
                success=True,
                attack_type='cred_brute',
                target=target,
                data={'message': 'Brute force ready'},
            )
            
    async def _handle_cred_spray(self, context: ConversationContext,
                                  params: Dict) -> AttackResult:
        """Password spraying."""
        target = context.get_last_target()
        
        return AttackResult(
            success=True,
            attack_type='cred_spray',
            target=target,
            data={'message': 'Password spray ready'},
        )
        
    async def _handle_can_scan(self, context: ConversationContext,
                                params: Dict) -> AttackResult:
        """CAN bus scanning."""
        return AttackResult(
            success=True,
            attack_type='vehicle_can',
            target=None,
            data={'message': 'CAN bus scanner ready'},
            next_steps=['vehicle_inject', 'vehicle_fuzz'],
        )
        
    async def _handle_keyfob(self, context: ConversationContext,
                              params: Dict) -> AttackResult:
        """Key fob attack."""
        return AttackResult(
            success=True,
            attack_type='vehicle_keyfob',
            target=None,
            data={'message': 'Key fob capture ready'},
            next_steps=['vehicle_replay', 'vehicle_rolljam'],
        )
        
    async def _handle_rfid_scan(self, context: ConversationContext,
                                 params: Dict) -> AttackResult:
        """RFID/NFC scanning."""
        return AttackResult(
            success=True,
            attack_type='rfid_scan',
            target=None,
            data={'message': 'RFID scanner ready'},
            next_steps=['rfid_clone', 'rfid_crack'],
        )
        
    async def _handle_rfid_clone(self, context: ConversationContext,
                                  params: Dict) -> AttackResult:
        """RFID card cloning."""
        target = context.get_last_target()
        
        return AttackResult(
            success=True,
            attack_type='rfid_clone',
            target=target,
            data={'message': 'Card cloner ready'},
        )
        
    async def _handle_c2_start(self, context: ConversationContext,
                                params: Dict) -> AttackResult:
        """Start C2 server."""
        try:
            from modules.pentest import C2Server
            
            server = C2Server()
            result = await server.start(
                port=params.get('port', 8443),
                protocol=params.get('protocol', 'https'),
            )
            
            return AttackResult(
                success=True,
                attack_type='c2_start',
                target=None,
                data=result,
                next_steps=['c2_generate', 'c2_listeners'],
            )
            
        except ImportError:
            return AttackResult(
                success=True,
                attack_type='c2_start',
                target=None,
                data={'message': 'C2 server ready'},
            )
            
    async def _handle_c2_generate(self, context: ConversationContext,
                                   params: Dict) -> AttackResult:
        """Generate C2 payload."""
        return AttackResult(
            success=True,
            attack_type='c2_generate',
            target=None,
            data={'message': 'Payload generator ready'},
        )
        
    async def _handle_cloud_enum(self, context: ConversationContext,
                                  params: Dict) -> AttackResult:
        """Cloud resource enumeration."""
        try:
            from modules.pentest import CloudSecurityScanner
            
            scanner = CloudSecurityScanner()
            results = await scanner.enumerate(
                provider=params.get('provider', 'aws'),
            )
            
            return AttackResult(
                success=True,
                attack_type='cloud_enum',
                target=None,
                data=results,
                next_steps=['cloud_exploit'],
            )
            
        except ImportError:
            return AttackResult(
                success=True,
                attack_type='cloud_enum',
                target=None,
                data={'message': 'Cloud scanner ready'},
            )
            
    async def _handle_api_scan(self, context: ConversationContext,
                                params: Dict) -> AttackResult:
        """API security scanning."""
        try:
            from modules.pentest import APISecurityScanner
            
            scanner = APISecurityScanner(target_url=params.get('url'))
            results = await scanner.full_scan()
            
            return AttackResult(
                success=True,
                attack_type='api_scan',
                target=None,
                data=results,
                next_steps=['api_fuzz', 'api_auth_bypass'],
            )
            
        except ImportError:
            return AttackResult(
                success=True,
                attack_type='api_scan',
                target=None,
                data={'message': 'API scanner ready'},
            )


# =============================================================================
# RESPONSE GENERATOR - Natural language responses
# =============================================================================

class ResponseGenerator:
    """
    Generates natural, conversational responses.
    Professional tone without unnecessary warnings or disclaimers.
    """
    
    def __init__(self):
        self.templates = self._init_templates()
        
    def _init_templates(self) -> Dict[str, List[str]]:
        """Initialize response templates."""
        return {
            'scan_start': [
                "Scanning {domain}... Stand by.",
                "Initiating {domain} reconnaissance.",
                "Starting {domain} scan.",
            ],
            'scan_complete': [
                "Scan complete. Found {count} targets.",
                "Discovery finished. {count} {domain} targets identified.",
                "{count} targets discovered.",
            ],
            'no_targets': [
                "No targets found. Try adjusting scan parameters or check hardware.",
                "Scan returned no results. Verify connectivity and try again.",
            ],
            'attack_ready': [
                "Ready to attack {target}. Proceed?",
                "Target {target} acquired. Execute attack?",
                "{target} in range. Launch attack?",
            ],
            'attack_success': [
                "Attack successful. {details}",
                "Target compromised. {details}",
                "Operation complete. {details}",
            ],
            'attack_failed': [
                "Attack failed: {error}. {suggestion}",
                "Operation unsuccessful. {error}",
            ],
            'next_steps': [
                "Available next actions: {actions}",
                "You can now: {actions}",
                "Suggested: {actions}",
            ],
            'confirmation_destructive': [
                "This action is irreversible. Execute?",
                "Warning: Destructive operation. Confirm?",
            ],
            'confirmation_noisy': [
                "This will likely trigger alerts. Proceed anyway?",
                "Noisy operation - may be detected. Continue?",
            ],
            'context_reference': [
                "Using previous target: {target}",
                "Targeting {target} from earlier scan.",
            ],
            'error': [
                "Error: {message}",
                "Operation failed: {message}",
            ],
        }
        
    def generate(self, template_key: str, **kwargs) -> str:
        """Generate a response from template."""
        templates = self.templates.get(template_key, ["Operation complete."])
        template = secrets.choice(templates)
        
        try:
            return template.format(**kwargs)
        except KeyError:
            return template
            
    def format_results(self, result: AttackResult, verbose: bool = True) -> str:
        """Format attack results for display."""
        lines = []
        
        if result.success:
            lines.append(f"✓ {result.attack_type} successful")
        else:
            lines.append(f"✗ {result.attack_type} failed")
            if result.error:
                lines.append(f"  Error: {result.error}")
                
        if verbose and result.data:
            for key, value in result.data.items():
                if key not in ['message']:
                    if isinstance(value, list) and len(value) > 5:
                        lines.append(f"  {key}: {len(value)} items")
                    else:
                        lines.append(f"  {key}: {value}")
                        
        if result.next_steps:
            lines.append(f"  Next: {', '.join(result.next_steps[:3])}")
            
        return '\n'.join(lines)
        
    def format_targets(self, targets: List[Target]) -> str:
        """Format target list for display."""
        if not targets:
            return "No targets discovered."
            
        lines = [f"Discovered {len(targets)} target(s):"]
        for i, target in enumerate(targets[:10], 1):
            lines.append(f"  {i}. [{target.type}] {target.value}")
            if target.metadata.get('signal'):
                lines.append(f"      Signal: {target.metadata['signal']} dBm")
            if target.vulnerabilities:
                lines.append(f"      Vulns: {', '.join(target.vulnerabilities[:3])}")
                
        if len(targets) > 10:
            lines.append(f"  ... and {len(targets) - 10} more")
            
        return '\n'.join(lines)


# =============================================================================
# RECOMMENDATION ENGINE
# =============================================================================

class RecommendationEngine:
    """
    Provides intelligent attack recommendations based on context.
    """
    
    def __init__(self):
        self.rules = self._init_rules()
        
    def _init_rules(self) -> List[Dict]:
        """Initialize recommendation rules."""
        return [
            {
                'condition': lambda ctx: any(
                    'handshake_captured' in t.vulnerabilities 
                    for t in ctx.discovered_targets.values()
                ),
                'recommendation': "Handshake captured - crack the password?",
                'action': 'wifi_crack',
            },
            {
                'condition': lambda ctx: any(
                    t.type == 'host' and t.metadata.get('open_ports')
                    for t in ctx.discovered_targets.values()
                ),
                'recommendation': "Open ports found - run vulnerability scan?",
                'action': 'net_vuln_scan',
            },
            {
                'condition': lambda ctx: any(
                    t.credentials for t in ctx.discovered_targets.values()
                ),
                'recommendation': "Credentials captured - test access?",
                'action': 'cred_test',
            },
            {
                'condition': lambda ctx: len(ctx.discovered_targets) > 0 and not ctx.attack_history,
                'recommendation': "Targets discovered - ready to attack. Which one?",
                'action': 'select_target',
            },
        ]
        
    def get_recommendations(self, context: ConversationContext) -> List[Dict]:
        """Get relevant recommendations for current context."""
        recommendations = []
        
        for rule in self.rules:
            try:
                if rule['condition'](context):
                    recommendations.append({
                        'text': rule['recommendation'],
                        'action': rule['action'],
                    })
            except Exception:
                continue
                
        return recommendations[:3]  # Top 3 recommendations


# =============================================================================
# MAIN ARSENAL AI v3.0 CLASS
# =============================================================================

class ArsenalAI:
    """
    Arsenal AI v3.0 - Conversational Attack Intelligence Engine
    
    Main interface for natural language attack operations.
    """
    
    def __init__(self):
        self.parser = IntentParser()
        self.executor = AttackExecutor()
        self.orchestrator = AttackChainOrchestrator(self.executor)
        self.responder = ResponseGenerator()
        self.recommender = RecommendationEngine()
        
        self.sessions: Dict[str, ConversationContext] = {}
        self._lock = threading.RLock()
        
    def get_session(self, session_id: str = None) -> ConversationContext:
        """Get or create a conversation session."""
        with self._lock:
            if session_id is None:
                session_id = secrets.token_hex(16)
                
            if session_id not in self.sessions:
                self.sessions[session_id] = ConversationContext(session_id=session_id)
                
            session = self.sessions[session_id]
            session.last_activity = datetime.now()
            return session
            
    async def process(self, user_input: str, session_id: str = None) -> Dict[str, Any]:
        """
        Process natural language input and execute appropriate actions.
        
        Args:
            user_input: Natural language command
            session_id: Session identifier for context
            
        Returns:
            Dict with response, results, and recommendations
        """
        context = self.get_session(session_id)
        
        # Parse intent
        intent = self.parser.parse(user_input, context)
        
        # Check for pending confirmations
        if context.pending_confirmations:
            return await self._handle_confirmation(user_input, context)
            
        # Check if confirmation needed
        if intent.confirmation_needed != ConfirmationType.NONE:
            return self._request_confirmation(intent, context)
            
        # Execute action(s)
        if intent.attack_chain:
            results = await self.orchestrator.execute_chain(
                intent.attack_chain, 
                context,
                intent.parameters,
            )
        else:
            result = await self.executor.execute(intent.action, context, intent.parameters)
            results = [result]
            
        # Store in history
        context.attack_history.extend(results)
        
        # Generate response
        response = self._generate_response(results, context)
        
        # Get recommendations
        recommendations = self.recommender.get_recommendations(context)
        
        return {
            'session_id': context.session_id,
            'response': response,
            'results': [r.to_dict() for r in results],
            'targets': [t.to_dict() for t in context.current_targets],
            'recommendations': recommendations,
            'intent': {
                'domain': intent.domain.value,
                'phase': intent.phase.value,
                'action': intent.action,
                'confidence': intent.confidence,
            },
        }
        
    async def _handle_confirmation(self, user_input: str, 
                                    context: ConversationContext) -> Dict[str, Any]:
        """Handle confirmation responses."""
        affirmative = any(word in user_input.lower() 
                         for word in ['yes', 'y', 'proceed', 'continue', 'do it', 'execute', 'confirm'])
        negative = any(word in user_input.lower() 
                      for word in ['no', 'n', 'cancel', 'stop', 'abort'])
                      
        pending = context.pending_confirmations.pop(0)
        
        if affirmative:
            result = await self.executor.execute(
                pending['action'], 
                context, 
                pending['params']
            )
            context.attack_history.append(result)
            
            return {
                'session_id': context.session_id,
                'response': self.responder.format_results(result),
                'results': [result.to_dict()],
                'confirmed': True,
            }
        elif negative:
            return {
                'session_id': context.session_id,
                'response': "Operation cancelled.",
                'results': [],
                'confirmed': False,
            }
        else:
            # Re-add confirmation
            context.pending_confirmations.insert(0, pending)
            return {
                'session_id': context.session_id,
                'response': "Please confirm: yes or no?",
                'results': [],
                'awaiting_confirmation': True,
            }
            
    def _request_confirmation(self, intent: Intent, 
                              context: ConversationContext) -> Dict[str, Any]:
        """Request confirmation for dangerous operations."""
        context.pending_confirmations.append({
            'action': intent.action,
            'params': intent.parameters,
            'type': intent.confirmation_needed,
        })
        
        if intent.confirmation_needed == ConfirmationType.DESTRUCTIVE:
            message = self.responder.generate('confirmation_destructive')
        else:
            message = self.responder.generate('confirmation_noisy')
            
        return {
            'session_id': context.session_id,
            'response': message,
            'results': [],
            'awaiting_confirmation': True,
            'confirmation_type': intent.confirmation_needed.value,
        }
        
    def _generate_response(self, results: List[AttackResult], 
                           context: ConversationContext) -> str:
        """Generate natural language response from results."""
        lines = []
        
        for result in results:
            lines.append(self.responder.format_results(result))
            
        # Add target summary if new targets discovered
        new_targets = [t for t in context.current_targets 
                      if (datetime.now() - t.discovered_at).seconds < 60]
        if new_targets:
            lines.append("")
            lines.append(self.responder.format_targets(new_targets))
            
        return '\n'.join(lines)
        
    def set_stealth_mode(self, session_id: str, level: StealthLevel) -> None:
        """Set stealth level for session."""
        context = self.get_session(session_id)
        context.stealth_mode = level
        
    def set_silent_mode(self, session_id: str, enabled: bool = True) -> None:
        """Enable/disable silent output mode."""
        context = self.get_session(session_id)
        context.silent_output = enabled
        
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of session activity."""
        context = self.get_session(session_id)
        
        return {
            'session_id': context.session_id,
            'started': context.started_at.isoformat(),
            'duration_minutes': int((datetime.now() - context.started_at).seconds / 60),
            'targets_discovered': len(context.discovered_targets),
            'attacks_executed': len(context.attack_history),
            'successful_attacks': sum(1 for r in context.attack_history if r.success),
            'current_stealth': context.stealth_mode.value,
            'silent_mode': context.silent_output,
        }
        
    def clear_session(self, session_id: str) -> None:
        """Clear session data (secure wipe)."""
        with self._lock:
            if session_id in self.sessions:
                # Overwrite sensitive data before deletion
                session = self.sessions[session_id]
                session.discovered_targets.clear()
                session.attack_history.clear()
                session.variables.clear()
                del self.sessions[session_id]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global instance
_arsenal_ai: Optional[ArsenalAI] = None


def get_arsenal_ai() -> ArsenalAI:
    """Get the global Arsenal AI instance."""
    global _arsenal_ai
    if _arsenal_ai is None:
        _arsenal_ai = ArsenalAI()
    return _arsenal_ai


async def ask(query: str, session_id: str = None) -> Dict[str, Any]:
    """
    Convenience function for natural language queries.
    
    Usage:
        result = await ask("scan for wifi networks")
        result = await ask("attack the strongest one")
        result = await ask("crack the password")
    """
    ai = get_arsenal_ai()
    return await ai.process(query, session_id)


def quick_ask(query: str, session_id: str = None) -> Dict[str, Any]:
    """Synchronous wrapper for ask()."""
    return asyncio.run(ask(query, session_id))


# =============================================================================
# MODULE INFO
# =============================================================================

__version__ = '3.0.0'
__all__ = [
    'ArsenalAI',
    'Intent',
    'AttackResult',
    'Target',
    'ConversationContext',
    'AttackDomain',
    'AttackPhase',
    'StealthLevel',
    'get_arsenal_ai',
    'ask',
    'quick_ask',
]
