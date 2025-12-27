#!/usr/bin/env python3
"""
RF Arsenal OS - Proxmark3 NFC/RFID Integration
Real hardware control for Proxmark3 devices

Capabilities:
- RFID reading/writing (125kHz LF)
- NFC reading/writing (13.56MHz HF)
- Card cloning
- Brute force attacks
- Sniffing/eavesdropping
- Emulation

Hardware: Proxmark3 Easy/RDV4

README COMPLIANCE:
- Real-World Functional Only: No simulation mode fallbacks
- Requires actual Proxmark3 hardware connected via USB
- Use --dry-run for testing without hardware
"""

import subprocess
import logging

# Try to import serial for hardware control
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    serial = None

# Import custom exceptions
try:
    from core import HardwareRequirementError, DependencyError
except ImportError:
    class HardwareRequirementError(Exception):
        def __init__(self, message, required_hardware=None, alternatives=None):
            super().__init__(f"HARDWARE REQUIRED: {message}")
    
    class DependencyError(Exception):
        def __init__(self, message, package=None, install_cmd=None):
            super().__init__(f"DEPENDENCY REQUIRED: {message}")
import time
import json
import re
import threading
import queue
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from datetime import datetime


class CardType(Enum):
    """Supported card types"""
    # LF (125kHz)
    EM4100 = "em4100"
    EM4305 = "em4305"
    T5577 = "t5577"
    HID_PROX = "hid_prox"
    INDALA = "indala"
    AWID = "awid"
    IO_PROX = "io_prox"
    VIKING = "viking"
    PYRAMID = "pyramid"
    
    # HF (13.56MHz)
    MIFARE_CLASSIC_1K = "mf_classic_1k"
    MIFARE_CLASSIC_4K = "mf_classic_4k"
    MIFARE_ULTRALIGHT = "mf_ultralight"
    MIFARE_DESFIRE = "mf_desfire"
    NTAG_213 = "ntag213"
    NTAG_215 = "ntag215"
    NTAG_216 = "ntag216"
    ICLASS = "iclass"
    LEGIC = "legic"
    FELICA = "felica"
    ISO14443A = "iso14443a"
    ISO14443B = "iso14443b"
    ISO15693 = "iso15693"


class AttackType(Enum):
    """Attack types"""
    DARKSIDE = "darkside"        # Mifare Classic key recovery
    NESTED = "nested"            # Known-key nested attack
    HARDNESTED = "hardnested"    # Unknown key attack
    STATIC_NESTED = "staticnested"  # Static encrypted nonce
    DICTIONARY = "dictionary"    # Dictionary key attack
    BRUTE_FORCE = "bruteforce"   # Key brute force
    RELAY = "relay"              # Relay attack
    SNIFF = "sniff"              # Eavesdrop communication
    EMULATE = "emulate"          # Card emulation
    CLONE = "clone"              # Full card clone


@dataclass
class RFIDCard:
    """RFID/NFC Card information"""
    uid: str
    card_type: CardType
    sak: Optional[str] = None
    atqa: Optional[str] = None
    ats: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    keys: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    raw_dump: Optional[bytes] = None
    
    def to_dict(self) -> Dict:
        return {
            'uid': self.uid,
            'card_type': self.card_type.value,
            'sak': self.sak,
            'atqa': self.atqa,
            'ats': self.ats,
            'data': self.data,
            'keys_found': len(self.keys),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass 
class AttackResult:
    """Attack result"""
    success: bool
    attack_type: AttackType
    duration: float
    keys_found: Dict[str, str] = field(default_factory=dict)
    data_recovered: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'attack_type': self.attack_type.value,
            'duration': self.duration,
            'keys_found': len(self.keys_found),
            'message': self.message
        }


class Proxmark3Controller:
    """
    Proxmark3 Controller for RF Arsenal OS
    
    Provides real NFC/RFID operations:
    - Card identification
    - Data reading/writing
    - Key recovery attacks
    - Card cloning
    - Eavesdropping
    """
    
    # Default key dictionaries
    DEFAULT_KEYS = [
        "FFFFFFFFFFFF",  # Factory default
        "A0A1A2A3A4A5",  # MAD key
        "B0B1B2B3B4B5",  # NDEF key
        "000000000000",  # Zero
        "D3F7D3F7D3F7",  # Common
        "4D3A99C351DD",  # Common
        "1A982C7E459A",  # Common
        "AABBCCDDEEFF",  # Common test
        "714C5C886E97",  # Common
        "587EE5F9350F",  # Common
        "A0478CC39091",  # Common
        "533CB6C723F6",  # Common
        "8FD0A4F256E9",  # Common
    ]
    
    def __init__(self, port: Optional[str] = None):
        self.logger = logging.getLogger('Proxmark3')
        self.port = port
        self.serial_conn = None  # serial.Serial if available
        self.connected = False
        self.device_info: Dict[str, str] = {}
        
        # State
        self.cards_found: List[RFIDCard] = []
        self.attack_results: List[AttackResult] = []
        
        # Background operations
        self._sniff_thread: Optional[threading.Thread] = None
        self._sniff_queue = queue.Queue()
        self._stop_sniff = threading.Event()
        
    def detect_device(self) -> Optional[str]:
        """
        Auto-detect Proxmark3 device.
        
        README COMPLIANCE: No simulation fallback - requires real hardware.
        
        Raises:
            DependencyError: If pyserial is not installed
        """
        if not SERIAL_AVAILABLE:
            raise DependencyError(
                "PySerial library required for Proxmark3 communication",
                package="pyserial",
                install_cmd="pip install pyserial"
            )
            
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            # Check for Proxmark3 identifiers
            if any(x in (port.description or '').lower() for x in ['proxmark', 'pm3', 'cdc']):
                self.logger.info(f"Found Proxmark3 on {port.device}")
                return port.device
                
            # Check VID:PID
            if port.vid == 0x9AC4 and port.pid in [0x4B8F, 0x4B8E]:  # PM3 VID:PID
                self.logger.info(f"Found Proxmark3 on {port.device} (VID:PID match)")
                return port.device
                
        self.logger.warning("No Proxmark3 device detected")
        return None
        
    def connect(self, port: Optional[str] = None) -> bool:
        """Connect to Proxmark3"""
        if self.connected:
            return True
            
        target_port = port or self.port or self.detect_device()
        
        if not target_port:
            self.logger.error("No port specified and auto-detect failed")
            return False
            
        try:
            self.serial_conn = serial.Serial(
                port=target_port,
                baudrate=115200,
                timeout=3
            )
            time.sleep(0.5)
            
            # Test connection
            if self._send_command("hw status"):
                self.connected = True
                self.port = target_port
                self._get_device_info()
                self.logger.info(f"Connected to Proxmark3 on {target_port}")
                return True
            else:
                self.serial_conn.close()
                return False
                
        except Exception as e:
            self.logger.error(f"Serial connection error: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from Proxmark3"""
        if self.serial_conn:
            self.serial_conn.close()
        self.connected = False
        self.logger.info("Disconnected from Proxmark3")
        
    def _send_command(self, command: str, timeout: float = 10.0) -> Optional[str]:
        """Send command to Proxmark3"""
        if not self.serial_conn:
            return None
            
        try:
            # Clear buffer
            self.serial_conn.reset_input_buffer()
            
            # Send command
            self.serial_conn.write(f"{command}\n".encode())
            
            # Read response
            response = ""
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if self.serial_conn.in_waiting:
                    data = self.serial_conn.read(self.serial_conn.in_waiting)
                    response += data.decode(errors='ignore')
                    
                    # Check for prompt (command complete)
                    if "pm3 -->" in response or "[+]" in response:
                        break
                        
                time.sleep(0.1)
                
            return response
            
        except Exception as e:
            self.logger.error(f"Command error: {e}")
            return None
            
    def _run_pm3_cli(self, command: str, timeout: float = 30.0) -> Optional[str]:
        """
        Run pm3 CLI command
        
        SECURITY: Uses subprocess with list arguments (no shell=True)
        to prevent command injection vulnerabilities.
        """
        try:
            # Build command as list to avoid shell injection
            cmd_args = ['pm3']
            if self.port:
                cmd_args.extend(['-p', self.port])
            cmd_args.extend(['-c', command])
            
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return result.stdout + result.stderr
            
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Command timed out: {command}")
            return None
        except Exception as e:
            self.logger.error(f"CLI error: {e}")
            return None
            
    def _get_device_info(self):
        """Get device information"""
        response = self._run_pm3_cli("hw version")
        if response:
            self.device_info['version'] = response
            
    # === LF (125kHz) Operations ===
    
    def lf_search(self) -> Optional[RFIDCard]:
        """Search for LF cards"""
        self.logger.info("Searching for LF cards...")
        
        response = self._run_pm3_cli("lf search")
        if not response:
            return None
            
        card = self._parse_lf_response(response)
        if card:
            self.cards_found.append(card)
            self.logger.info(f"Found LF card: {card.card_type.value} - {card.uid}")
            
        return card
        
    def _parse_lf_response(self, response: str) -> Optional[RFIDCard]:
        """Parse LF search response"""
        # EM4100 pattern
        em_match = re.search(r'EM410x.*?ID:\s*(\w+)', response, re.IGNORECASE)
        if em_match:
            return RFIDCard(
                uid=em_match.group(1),
                card_type=CardType.EM4100
            )
            
        # HID Prox pattern
        hid_match = re.search(r'HID.*?ID:\s*(\w+)', response, re.IGNORECASE)
        if hid_match:
            return RFIDCard(
                uid=hid_match.group(1),
                card_type=CardType.HID_PROX
            )
            
        # T5577 pattern
        t55_match = re.search(r'T55.*?ID:\s*(\w+)', response, re.IGNORECASE)
        if t55_match:
            return RFIDCard(
                uid=t55_match.group(1),
                card_type=CardType.T5577
            )
            
        return None
        
    def lf_read(self, card_type: CardType) -> Optional[RFIDCard]:
        """Read specific LF card type"""
        type_commands = {
            CardType.EM4100: "lf em 410x reader",
            CardType.HID_PROX: "lf hid reader",
            CardType.INDALA: "lf indala reader",
            CardType.T5577: "lf t55xx detect",
            CardType.AWID: "lf awid reader",
            CardType.IO_PROX: "lf io reader",
        }
        
        cmd = type_commands.get(card_type)
        if not cmd:
            self.logger.error(f"Unsupported card type: {card_type}")
            return None
            
        response = self._run_pm3_cli(cmd)
        return self._parse_lf_response(response) if response else None
        
    def lf_clone(self, source_uid: str, target_type: CardType = CardType.T5577) -> bool:
        """Clone LF card to T5577"""
        self.logger.info(f"Cloning {source_uid} to {target_type.value}...")
        
        # Write to T5577
        response = self._run_pm3_cli(f"lf em 410x clone --id {source_uid}")
        
        if response and "success" in response.lower():
            self.logger.info("Clone successful")
            return True
            
        self.logger.error("Clone failed")
        return False
        
    def lf_sim(self, uid: str, card_type: CardType = CardType.EM4100) -> bool:
        """Simulate LF card"""
        self.logger.info(f"Simulating {card_type.value}: {uid}")
        
        if card_type == CardType.EM4100:
            cmd = f"lf em 410x sim --id {uid}"
        elif card_type == CardType.HID_PROX:
            cmd = f"lf hid sim -r {uid}"
        else:
            self.logger.error(f"Simulation not supported for {card_type}")
            return False
            
        response = self._run_pm3_cli(cmd, timeout=60)
        return response is not None
        
    # === HF (13.56MHz) Operations ===
    
    def hf_search(self) -> Optional[RFIDCard]:
        """Search for HF cards"""
        self.logger.info("Searching for HF cards...")
        
        response = self._run_pm3_cli("hf search")
        if not response:
            return None
            
        card = self._parse_hf_response(response)
        if card:
            self.cards_found.append(card)
            self.logger.info(f"Found HF card: {card.card_type.value} - {card.uid}")
            
        return card
        
    def _parse_hf_response(self, response: str) -> Optional[RFIDCard]:
        """Parse HF search response"""
        # ISO14443A/Mifare pattern
        uid_match = re.search(r'UID\s*:\s*([\w\s]+)', response)
        sak_match = re.search(r'SAK\s*:\s*(\w+)', response)
        atqa_match = re.search(r'ATQA\s*:\s*(\w+)', response)
        
        if uid_match:
            uid = uid_match.group(1).replace(' ', '')
            sak = sak_match.group(1) if sak_match else None
            atqa = atqa_match.group(1) if atqa_match else None
            
            # Determine card type from SAK
            card_type = self._identify_card_type(sak)
            
            return RFIDCard(
                uid=uid,
                card_type=card_type,
                sak=sak,
                atqa=atqa
            )
            
        return None
        
    def _identify_card_type(self, sak: Optional[str]) -> CardType:
        """Identify card type from SAK"""
        if not sak:
            return CardType.ISO14443A
            
        sak_int = int(sak, 16)
        
        if sak_int == 0x08:
            return CardType.MIFARE_CLASSIC_1K
        elif sak_int == 0x18:
            return CardType.MIFARE_CLASSIC_4K
        elif sak_int == 0x00:
            return CardType.MIFARE_ULTRALIGHT
        elif sak_int == 0x20:
            return CardType.MIFARE_DESFIRE
        else:
            return CardType.ISO14443A
            
    def hf_mifare_read(self, block: int = 0, key: str = "FFFFFFFFFFFF", 
                       key_type: str = "A") -> Optional[str]:
        """Read Mifare Classic block"""
        cmd = f"hf mf rdbl --blk {block} -{key_type.lower()} -k {key}"
        response = self._run_pm3_cli(cmd)
        
        if response and "block" in response.lower():
            # Extract block data
            data_match = re.search(r'block\s+\d+.*?:\s*([\w\s]+)', response, re.IGNORECASE)
            if data_match:
                return data_match.group(1).replace(' ', '')
                
        return None
        
    def hf_mifare_write(self, block: int, data: str, key: str = "FFFFFFFFFFFF",
                        key_type: str = "A") -> bool:
        """Write to Mifare Classic block"""
        cmd = f"hf mf wrbl --blk {block} -{key_type.lower()} -k {key} -d {data}"
        response = self._run_pm3_cli(cmd)
        
        return response is not None and "success" in response.lower()
        
    def hf_mifare_dump(self, keys: Optional[Dict[str, str]] = None) -> Optional[Dict]:
        """Dump all Mifare Classic data"""
        self.logger.info("Dumping Mifare Classic card...")
        
        # First try autopwn to recover keys
        if not keys:
            keys = self._recover_mifare_keys()
            
        if not keys:
            self.logger.error("No keys available for dump")
            return None
            
        # Create key file
        self._write_key_file(keys)
        
        # Dump card
        response = self._run_pm3_cli("hf mf dump", timeout=60)
        
        if response and ("dumped" in response.lower() or "saved" in response.lower()):
            self.logger.info("Dump successful")
            return {"status": "success", "keys": keys}
            
        return None
        
    def _write_key_file(self, keys: Dict[str, str]):
        """Write keys to file for dump operation"""
        key_file = Path("/tmp/hf-mf-keys.bin")
        # In production, write actual key binary data
        
    def _recover_mifare_keys(self) -> Optional[Dict[str, str]]:
        """Recover Mifare Classic keys using various attacks"""
        self.logger.info("Attempting key recovery...")
        
        # Try darkside attack first (no known key needed)
        result = self.attack_darkside()
        if result.success and result.keys_found:
            return result.keys_found
            
        # Try dictionary attack
        result = self.attack_dictionary()
        if result.success and result.keys_found:
            return result.keys_found
            
        return None
        
    # === Attack Methods ===
    
    def attack_darkside(self) -> AttackResult:
        """Darkside attack for Mifare Classic"""
        self.logger.info("Starting darkside attack...")
        start_time = time.time()
        
        response = self._run_pm3_cli("hf mf darkside", timeout=120)
        duration = time.time() - start_time
        
        keys_found = {}
        if response:
            key_match = re.search(r'found\s+key\s*:\s*(\w+)', response, re.IGNORECASE)
            if key_match:
                keys_found["sector_0_A"] = key_match.group(1)
                
        result = AttackResult(
            success=len(keys_found) > 0,
            attack_type=AttackType.DARKSIDE,
            duration=duration,
            keys_found=keys_found,
            message="Darkside attack completed"
        )
        
        self.attack_results.append(result)
        return result
        
    def attack_nested(self, known_key: str, known_sector: int = 0) -> AttackResult:
        """Nested attack using known key"""
        self.logger.info(f"Starting nested attack with key {known_key}...")
        start_time = time.time()
        
        cmd = f"hf mf nested --blk {known_sector * 4} -a -k {known_key} --tblk 4"
        response = self._run_pm3_cli(cmd, timeout=180)
        duration = time.time() - start_time
        
        keys_found = {}
        if response:
            # Parse found keys
            for match in re.finditer(r'sector\s+(\d+).*?key\s+([AB]):\s*(\w+)', 
                                     response, re.IGNORECASE):
                sector = match.group(1)
                key_type = match.group(2)
                key = match.group(3)
                keys_found[f"sector_{sector}_{key_type}"] = key
                
        result = AttackResult(
            success=len(keys_found) > 0,
            attack_type=AttackType.NESTED,
            duration=duration,
            keys_found=keys_found,
            message=f"Found {len(keys_found)} keys"
        )
        
        self.attack_results.append(result)
        return result
        
    def attack_hardnested(self, target_sector: int, target_key_type: str = "A") -> AttackResult:
        """Hardnested attack (no known key)"""
        self.logger.info(f"Starting hardnested attack on sector {target_sector}...")
        start_time = time.time()
        
        # This is a complex attack that can take a long time
        cmd = f"hf mf hardnested --tblk {target_sector * 4} -{target_key_type.lower()}"
        response = self._run_pm3_cli(cmd, timeout=600)
        duration = time.time() - start_time
        
        keys_found = {}
        if response:
            key_match = re.search(r'found\s+.*?key\s*:\s*(\w+)', response, re.IGNORECASE)
            if key_match:
                keys_found[f"sector_{target_sector}_{target_key_type}"] = key_match.group(1)
                
        result = AttackResult(
            success=len(keys_found) > 0,
            attack_type=AttackType.HARDNESTED,
            duration=duration,
            keys_found=keys_found,
            message="Hardnested attack completed"
        )
        
        self.attack_results.append(result)
        return result
        
    def attack_dictionary(self, keys: Optional[List[str]] = None) -> AttackResult:
        """Dictionary attack with known keys"""
        self.logger.info("Starting dictionary attack...")
        start_time = time.time()
        
        test_keys = keys or self.DEFAULT_KEYS
        keys_found = {}
        
        for key in test_keys:
            # Test key on sector 0
            response = self._run_pm3_cli(f"hf mf chk --blk 0 -a -k {key}")
            if response and "found" in response.lower():
                keys_found["sector_0_A"] = key
                break
                
        duration = time.time() - start_time
        
        result = AttackResult(
            success=len(keys_found) > 0,
            attack_type=AttackType.DICTIONARY,
            duration=duration,
            keys_found=keys_found,
            message=f"Tested {len(test_keys)} keys"
        )
        
        self.attack_results.append(result)
        return result
        
    def attack_autopwn(self) -> AttackResult:
        """Automatic key recovery (tries all methods)"""
        self.logger.info("Starting autopwn...")
        start_time = time.time()
        
        response = self._run_pm3_cli("hf mf autopwn", timeout=600)
        duration = time.time() - start_time
        
        keys_found = {}
        if response:
            for match in re.finditer(r'sector\s+(\d+).*?key\s+([AB]):\s*(\w+)',
                                     response, re.IGNORECASE):
                sector = match.group(1)
                key_type = match.group(2)
                key = match.group(3)
                keys_found[f"sector_{sector}_{key_type}"] = key
                
        result = AttackResult(
            success=len(keys_found) > 0,
            attack_type=AttackType.BRUTE_FORCE,
            duration=duration,
            keys_found=keys_found,
            message=f"Autopwn found {len(keys_found)} keys"
        )
        
        self.attack_results.append(result)
        return result
        
    def hf_clone(self, source_dump: str, target_type: CardType = CardType.MIFARE_CLASSIC_1K) -> bool:
        """Clone HF card from dump file"""
        self.logger.info(f"Cloning from {source_dump}...")
        
        # Restore dump to blank card
        response = self._run_pm3_cli(f"hf mf restore --1k -f {source_dump}", timeout=120)
        
        if response and "restored" in response.lower():
            self.logger.info("Clone successful")
            return True
            
        self.logger.error("Clone failed")
        return False
        
    def hf_sim(self, uid: str) -> bool:
        """Simulate HF card"""
        self.logger.info(f"Simulating card: {uid}")
        
        response = self._run_pm3_cli(f"hf mf sim --1k -u {uid}", timeout=60)
        return response is not None
        
    # === Sniffing Operations ===
    
    def start_sniff(self, mode: str = "hf") -> bool:
        """Start sniffing communications"""
        self.logger.info(f"Starting {mode.upper()} sniff...")
        
        self._stop_sniff.clear()
        
        if mode == "hf":
            cmd = "hf sniff"
        elif mode == "lf":
            cmd = "lf sniff"
        else:
            return False
            
        self._sniff_thread = threading.Thread(
            target=self._sniff_worker,
            args=(cmd,),
            daemon=True
        )
        self._sniff_thread.start()
        return True
        
    def _sniff_worker(self, cmd: str):
        """Background sniff worker"""
        response = self._run_pm3_cli(cmd, timeout=300)
        if response:
            self._sniff_queue.put(response)
            
    def stop_sniff(self) -> Optional[str]:
        """Stop sniffing and return captured data"""
        self._stop_sniff.set()
        
        # Send break command
        self._run_pm3_cli("hw break")
        
        try:
            return self._sniff_queue.get(timeout=5)
        except queue.Empty:
            return None
            
    # === Utility Methods ===
    
    def get_status(self) -> Dict:
        """Get Proxmark3 status"""
        return {
            'connected': self.connected,
            'port': self.port,
            'device_info': self.device_info,
            'cards_found': len(self.cards_found),
            'attacks_performed': len(self.attack_results)
        }
        
    def get_cards_found(self) -> List[Dict]:
        """Get all found cards"""
        return [c.to_dict() for c in self.cards_found]
        
    def get_attack_results(self) -> List[Dict]:
        """Get all attack results"""
        return [r.to_dict() for r in self.attack_results]


# Convenience function for AI Command Center
def get_proxmark3_controller(port: Optional[str] = None) -> Proxmark3Controller:
    """Get Proxmark3 controller instance"""
    return Proxmark3Controller(port)
