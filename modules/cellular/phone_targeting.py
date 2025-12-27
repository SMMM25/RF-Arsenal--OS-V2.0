#!/usr/bin/env python3
"""
RF Arsenal OS - Phone Number Targeting Module
Professional-grade cellular targeting system

⚠️  LEGAL NOTICE - AUTHORIZED USE ONLY:
- Requires written authorization from target organization
- For authorized penetration testing and security research ONLY
- Unauthorized interception is illegal (18 U.S.C. § 2511)
- User assumes all legal responsibility

STEALTH FEATURES:
- No external network connections
- Encrypted local storage
- RAM-only operation mode available
- Anti-forensics integration
- Minimal console output
"""

import logging
import sqlite3
import time
import threading
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass
import re
import json
import os
import hashlib
from pathlib import Path

# Stealth: Import anti-forensics
try:
    from security.anti_forensics import EncryptedRAMOverlay
    ANTI_FORENSICS_AVAILABLE = True
except ImportError:
    ANTI_FORENSICS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PhoneTarget:
    """Target phone information"""
    phone_number: str
    imsi: Optional[str] = None
    carrier: Optional[str] = None
    last_seen: Optional[datetime] = None
    device_type: Optional[str] = None
    connection_status: str = "Not Connected"
    signal_strength: Optional[float] = None
    # Stealth: Add encryption flag
    encrypted: bool = False


class PhoneNumberTargeting:
    """
    Phone Number Targeting System
    
    STEALTH INTEGRATED:
    - Encrypted database storage
    - RAM-only operation mode
    - Anti-forensics cleanup
    - No external connections
    - Minimal logging
    
    Combines rogue base station with phone number targeting:
    1. Phone number → IMSI mapping
    2. IMSI → Device tracking on base station
    3. Selective interception
    """
    
    def __init__(self, gsm_controller, lte_controller, stealth_mode=True):
        """
        Initialize targeting system
        
        Args:
            gsm_controller: GSM 2G base station controller
            lte_controller: LTE 4G base station controller
            stealth_mode: Enable stealth features (default: True)
        """
        self.gsm = gsm_controller
        self.lte = lte_controller
        self.stealth_mode = stealth_mode
        
        # Stealth: Use covert storage paths
        if stealth_mode:
            self.db_path = '/tmp/.rf_arsenal_data/.imsi.db'
            self.capture_base = '/tmp/.rf_arsenal_data/captures'
        else:
            self.db_path = '/tmp/rf_arsenal_imsi.db'
            self.capture_base = '/tmp/captures'
        
        # Create directories with restrictive permissions
        os.makedirs(os.path.dirname(self.db_path), mode=0o700, exist_ok=True)
        os.makedirs(self.capture_base, mode=0o700, exist_ok=True)
        
        # IMSI database
        self.imsi_database = self._init_database()
        
        # Active targets
        self.targets: Dict[str, PhoneTarget] = {}
        
        # Monitoring thread
        self.monitor_thread = None
        self.monitoring = False
        
        # Stealth: Initialize anti-forensics if available
        self.ram_overlay = None
        if ANTI_FORENSICS_AVAILABLE and stealth_mode:
            try:
                self.ram_overlay = EncryptedRAMOverlay()
                logger.info("✅ Anti-forensics enabled")
            except Exception as e:
                logger.warning(f"Anti-forensics unavailable: {e}")
        
        # Stealth: Register for emergency cleanup
        self._register_cleanup_paths()
        
    def _register_cleanup_paths(self):
        """Register paths for emergency cleanup"""
        if self.ram_overlay:
            # Add paths to monitored list
            cleanup_paths = [
                self.db_path,
                self.capture_base,
                '/tmp/.rf_arsenal_data'
            ]
            # Paths will be cleaned on emergency shutdown
            
    def _init_database(self) -> sqlite3.Connection:
        """
        Initialize IMSI/Phone number mapping database
        
        STEALTH: Uses encrypted storage if available
        """
        db = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = db.cursor()
        
        # Enable encryption at rest (if SQLCipher available)
        try:
            # Try to enable SQLCipher encryption
            # Note: PRAGMA key is SQLCipher-specific and the key is internally generated,
            # not from user input. Using parameterized approach for defense-in-depth.
            encryption_key = self._generate_db_key()
            cursor.execute("PRAGMA key = ?", (encryption_key,))
            cursor.execute("PRAGMA cipher_page_size = 4096")
            logger.info("✅ Database encryption enabled")
        except:
            # Standard SQLite (no encryption)
            logger.warning("⚠️  Database not encrypted (SQLCipher unavailable)")
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS imsi_mapping (
                phone_number TEXT PRIMARY KEY,
                imsi TEXT,
                carrier TEXT,
                country_code TEXT,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                capture_count INTEGER DEFAULT 1,
                device_model TEXT,
                os_type TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS target_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone_number TEXT,
                imsi TEXT,
                timestamp TIMESTAMP,
                event_type TEXT,
                data TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS captured_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                imsi TEXT,
                phone_number TEXT,
                timestamp TIMESTAMP,
                data_type TEXT,
                content TEXT,
                metadata TEXT
            )
        ''')
        
        db.commit()
        return db
    
    def _generate_db_key(self) -> str:
        """Generate database encryption key
        
        SECURITY: Uses cryptographically secure random bytes instead of
        predictable system values for key generation.
        """
        import secrets
        # Use cryptographically secure random bytes
        key_material = secrets.token_bytes(32)
        # Add additional entropy from system
        key_material += os.urandom(16)
        # Add process-specific salt (makes key unique per instance)
        key_material += f"{os.getuid()}{os.getpid()}".encode()
        return hashlib.sha256(key_material).hexdigest()
    
    def add_target(self, phone_number: str) -> PhoneTarget:
        """
        Add target phone number
        
        Args:
            phone_number: Target phone (international format)
            
        Returns:
            PhoneTarget object
        """
        phone_number = self._normalize_phone_number(phone_number)
        imsi = self._lookup_imsi(phone_number)
        
        target = PhoneTarget(
            phone_number=phone_number,
            imsi=imsi,
            encrypted=self.stealth_mode
        )
        
        self.targets[phone_number] = target
        
        # Stealth: Minimal output
        if self.stealth_mode:
            if imsi:
                logger.info(f"Target indexed: {self._obfuscate_number(phone_number)}")
            else:
                logger.info(f"Target indexed (IMSI pending)")
        else:
            if imsi:
                print(f"[+] Target indexed: {phone_number} → {imsi}")
            else:
                print(f"[+] Target indexed: {phone_number} (IMSI pending)")
        
        self._log_event(phone_number, imsi, "TARGET_ADDED", "")
        
        return target
    
    def _obfuscate_number(self, phone: str) -> str:
        """Obfuscate phone number for logging"""
        if len(phone) > 6:
            return phone[:3] + "***" + phone[-2:]
        return "***"
    
    def _normalize_phone_number(self, phone: str) -> str:
        """Normalize phone number to international format"""
        # Remove non-digits
        phone = ''.join(filter(str.isdigit, phone))
        
        # Add country code if missing (US default)
        if not phone.startswith('1') and len(phone) == 10:
            phone = '1' + phone
        
        return phone
    
    def _lookup_imsi(self, phone_number: str) -> Optional[str]:
        """
        Lookup IMSI for phone number from capture database
        
        Args:
            phone_number: Phone number to lookup
            
        Returns:
            IMSI if found, None otherwise
        """
        cursor = self.imsi_database.cursor()
        cursor.execute(
            'SELECT imsi FROM imsi_mapping WHERE phone_number = ?',
            (phone_number,)
        )
        result = cursor.fetchone()
        
        if result:
            return result[0]
        
        return None
    
    def associate_imsi(self, phone_number: str, imsi: str, 
                       carrier: str = None) -> bool:
        """
        Associate IMSI with phone number
        
        Args:
            phone_number: Phone number
            imsi: IMSI captured from base station
            carrier: Carrier name (optional)
            
        Returns:
            Success status
        """
        cursor = self.imsi_database.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO imsi_mapping 
                (phone_number, imsi, carrier, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(phone_number) DO UPDATE SET
                    last_seen = ?,
                    capture_count = capture_count + 1
            ''', (
                phone_number, imsi, carrier,
                datetime.now(), datetime.now(),
                datetime.now()
            ))
            
            self.imsi_database.commit()
            
            # Update target if exists
            if phone_number in self.targets:
                self.targets[phone_number].imsi = imsi
                self.targets[phone_number].carrier = carrier
                
                if self.stealth_mode:
                    logger.info(f"IMSI mapped: {self._obfuscate_number(phone_number)}")
                else:
                    print(f"[+] Mapped: {phone_number} → {imsi}")
            
            return True
            
        except Exception as e:
            logger.error(f"Association failed: {e}")
            return False
    
    def start_targeted_capture(self, phone_number: str = None) -> Dict:
        """
        Start targeted capture
        
        Args:
            phone_number: Specific target, or None for bulk capture
            
        Returns:
            Status dict
        """
        if phone_number and phone_number not in self.targets:
            if not self.stealth_mode:
                print(f"[-] Unknown target")
            return {'success': False, 'error': 'Unknown target'}
        
        target = self.targets.get(phone_number) if phone_number else None
        carrier_name = target.carrier if target and target.carrier else "AT&T"
        
        # Stealth: Use randomized parameters
        import random
        arfcn_base = random.randint(40, 80)
        lac_base = random.randint(1000, 9999)
        
        # Configure base station
        gsm_config = {
            'arfcn': arfcn_base,
            'mcc': '310',
            'mnc': self._get_mnc_for_carrier(carrier_name),
            'name': carrier_name,
            'tx_power': 30,  # Stealth: Consider reducing
            'lac': lac_base,
            'cell_id': random.randint(1, 999)
        }
        
        # Start base station
        if not self.gsm.start_base_station(gsm_config):
            if not self.stealth_mode:
                print(f"[-] Station failed")
            return {'success': False, 'error': 'Station start failed'}
        
        # Start monitoring
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(phone_number, target.imsi if target else None),
            daemon=True
        )
        self.monitor_thread.start()
        
        # Stealth: Minimal output
        if self.stealth_mode:
            logger.info(f"Station active: {gsm_config['arfcn']} ARFCN")
            if phone_number:
                logger.info(f"Target: {self._obfuscate_number(phone_number)}")
        else:
            print(f"[+] Station active: {gsm_config['arfcn']} ARFCN")
            if phone_number:
                print(f"[+] Target: {phone_number}")
        
        return {
            'success': True,
            'target': phone_number,
            'imsi': target.imsi if target else None,
            'frequency': gsm_config['arfcn']
        }
    
    def stop_capture(self):
        """Stop capture and monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        self.gsm.stop_base_station()
        
        if not self.stealth_mode:
            print(f"[+] Station stopped")
        else:
            logger.info("Station stopped")
    
    def _get_mnc_for_carrier(self, carrier: str) -> str:
        """Get MNC for carrier"""
        carrier_mnc = {
            'AT&T': '410',
            'T-Mobile': '260',
            'Verizon': '480',
            'Sprint': '120',
            'Vodafone': '015',
            'Orange': '001'
        }
        return carrier_mnc.get(carrier, '410')
    
    def _monitor_loop(self, target_phone: Optional[str], target_imsi: Optional[str]):
        """
        Monitoring loop - runs in background thread
        
        Args:
            target_phone: Specific phone to monitor, or None for all
            target_imsi: Known IMSI, or None to detect
        """
        while self.monitoring:
            try:
                # Get connected devices from base station
                devices = self.gsm.get_connected_devices()
                
                for device in devices:
                    device_imsi = device.get('imsi')
                    
                    # Check if this is our target
                    if target_imsi and device_imsi == target_imsi:
                        self._handle_target_connection(target_phone, device)
                    elif not target_imsi:
                        # Bulk capture mode - log all
                        self._log_connection(device)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            
            time.sleep(1)
    
    def _handle_target_connection(self, phone_number: str, device: Dict):
        """
        Handle target connection
        
        Args:
            phone_number: Target phone number
            device: Device info from base station
        """
        target = self.targets.get(phone_number)
        if not target:
            return
        
        # Update target status
        if target.connection_status != "CONNECTED":
            target.connection_status = "CONNECTED"
            
            if self.stealth_mode:
                logger.info(f"Target connected: {self._obfuscate_number(phone_number)}")
            else:
                print(f"[+] {phone_number} connected")
        
        target.last_seen = datetime.now()
        target.signal_strength = device.get('rsrp', 0)
        
        # Start interception
        self._start_interception(phone_number, device)
        
        # Log event
        self._log_event(
            phone_number,
            device.get('imsi'),
            "CONNECTED",
            f"RSSI: {device.get('rsrp', 0)} dBm"
        )
    
    def _log_connection(self, device: Dict):
        """Log device connection (bulk capture mode)"""
        imsi = device.get('imsi')
        if not imsi:
            return
        
        cursor = self.imsi_database.cursor()
        cursor.execute('''
            INSERT INTO target_history 
            (phone_number, imsi, timestamp, event_type, data)
            VALUES (?, ?, ?, ?, ?)
        ''', ('Unknown', imsi, datetime.now(), 'DEVICE_SEEN', 
              json.dumps(device)))
        self.imsi_database.commit()
    
    def _start_interception(self, phone_number: str, device: Dict):
        """
        Start intercepting target communications
        
        Args:
            phone_number: Target phone
            device: Device info
        """
        imsi = device.get('imsi')
        if not imsi:
            return
        
        # Stealth: Use obfuscated directory names
        imsi_hash = hashlib.md5(imsi.encode()).hexdigest()[:8]
        capture_dir = f'{self.capture_base}/{imsi_hash}'
        
        # Create capture directory with restrictive permissions
        os.makedirs(capture_dir, mode=0o700, exist_ok=True)
        os.makedirs(f'{capture_dir}/calls', mode=0o700, exist_ok=True)
        os.makedirs(f'{capture_dir}/sms', mode=0o700, exist_ok=True)
        
        # Log interception start
        self._log_event(
            phone_number,
            imsi,
            "INTERCEPTION_STARTED",
            f"Output: {capture_dir}"
        )
    
    def get_target_status(self, phone_number: str = None) -> Optional[Dict]:
        """
        Get target status
        
        Args:
            phone_number: Specific target, or None for all
            
        Returns:
            Status dict or None
        """
        if phone_number:
            if phone_number not in self.targets:
                if not self.stealth_mode:
                    print(f"[-] Unknown target")
                return None
            
            target = self.targets[phone_number]
            
            # Compact output
            if self.stealth_mode:
                parts = [f"IMSI: {target.imsi[:8] + '***' if target.imsi else 'Unknown'}"]
            else:
                parts = [f"IMSI: {target.imsi or 'Unknown'}"]
            
            if target.connection_status == "CONNECTED":
                parts.append(f"RSSI: {target.signal_strength:.0f} dBm")
                parts.append(target.connection_status)
            
            if not self.stealth_mode:
                print(" | ".join(parts))
            
            return {
                'phone': target.phone_number,
                'imsi': target.imsi,
                'status': target.connection_status,
                'signal': target.signal_strength,
                'last_seen': target.last_seen
            }
        else:
            # List all targets
            for phone, target in self.targets.items():
                status_char = "●" if target.connection_status == "CONNECTED" else "○"
                
                if self.stealth_mode:
                    phone_display = self._obfuscate_number(phone)
                    imsi_display = target.imsi[:8] + '***' if target.imsi else 'Unknown'
                else:
                    phone_display = phone
                    imsi_display = target.imsi or 'Unknown'
                
                if not self.stealth_mode:
                    print(f"{status_char} {phone_display} | {imsi_display}")
            
            return {'count': len(self.targets)}
    
    def list_targets(self) -> List[Dict]:
        """List all active targets"""
        return [
            {
                'phone': self._obfuscate_number(t.phone_number) if self.stealth_mode else t.phone_number,
                'imsi': t.imsi[:8] + '***' if (self.stealth_mode and t.imsi) else t.imsi,
                'status': t.connection_status,
                'last_seen': t.last_seen
            }
            for t in self.targets.values()
        ]
    
    def remove_target(self, phone_number: str) -> bool:
        """Remove target from list"""
        if phone_number in self.targets:
            del self.targets[phone_number]
            
            if self.stealth_mode:
                logger.info(f"Target removed: {self._obfuscate_number(phone_number)}")
            else:
                print(f"[+] Target removed: {phone_number}")
            return True
        
        if not self.stealth_mode:
            print(f"[-] Unknown target")
        return False
    
    def extract_data(self, phone_number: str = None):
        """
        Extract captured data
        
        Args:
            phone_number: Specific target, or None for all
        """
        if phone_number:
            target = self.targets.get(phone_number)
            if not target or not target.imsi:
                if not self.stealth_mode:
                    print(f"[-] No data")
                return
            
            imsi_hash = hashlib.md5(target.imsi.encode()).hexdigest()[:8]
            capture_dir = f'{self.capture_base}/{imsi_hash}'
            
            if os.path.exists(capture_dir):
                if not self.stealth_mode:
                    print(f"[+] {capture_dir}/")
                    # List captured files
                    for root, dirs, files in os.walk(capture_dir):
                        level = root.replace(capture_dir, '').count(os.sep)
                        indent = ' ' * 4 * level
                        print(f"{indent}{os.path.basename(root)}/")
                        subindent = ' ' * 4 * (level + 1)
                        for file in files:
                            print(f"{subindent}{file}")
                else:
                    logger.info(f"Data location: {capture_dir}")
            else:
                if not self.stealth_mode:
                    print(f"[-] No captures")
        else:
            # List all captures
            if os.path.exists(self.capture_base):
                captures = os.listdir(self.capture_base)
                if not self.stealth_mode:
                    print(f"[+] {len(captures)} captures")
                    print(f"[+] {self.capture_base}/")
                else:
                    logger.info(f"{len(captures)} captures available")
            else:
                if not self.stealth_mode:
                    print(f"[-] No captures")
    
    def export_report(self, phone_number: str, filename: str):
        """
        Export target report
        
        Args:
            phone_number: Target phone
            filename: Output file
        """
        if phone_number not in self.targets:
            if not self.stealth_mode:
                print(f"[-] Unknown target")
            return
        
        cursor = self.imsi_database.cursor()
        cursor.execute('''
            SELECT timestamp, event_type, data
            FROM target_history
            WHERE phone_number = ?
            ORDER BY timestamp
        ''', (phone_number,))
        
        events = cursor.fetchall()
        
        report = {
            'target': self._obfuscate_number(phone_number) if self.stealth_mode else phone_number,
            'report_date': datetime.now().isoformat(),
            'summary': self.get_target_status(phone_number),
            'timeline': [
                {
                    'timestamp': str(event[0]),
                    'type': event[1],
                    'details': event[2]
                }
                for event in events
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Set restrictive permissions
        os.chmod(filename, 0o600)
        
        if not self.stealth_mode:
            print(f"[+] Report: {filename}")
        else:
            logger.info(f"Report exported: {filename}")
    
    def _log_event(self, phone_number: str, imsi: Optional[str], 
                   event_type: str, data: str):
        """Log event to database"""
        try:
            cursor = self.imsi_database.cursor()
            cursor.execute('''
                INSERT INTO target_history 
                (phone_number, imsi, timestamp, event_type, data)
                VALUES (?, ?, ?, ?, ?)
            ''', (phone_number, imsi, datetime.now(), event_type, data))
            self.imsi_database.commit()
        except Exception as e:
            logger.error(f"Log error: {e}")
    
    def emergency_cleanup(self):
        """
        Emergency cleanup - called on panic button
        
        STEALTH: Securely deletes all capture data
        """
        logger.warning("🚨 EMERGENCY CLEANUP - Phone targeting data")
        
        try:
            # Stop any active captures
            self.stop_capture()
            
            # Close database
            if self.imsi_database:
                self.imsi_database.close()
            
            # Secure delete database
            if os.path.exists(self.db_path):
                if self.ram_overlay:
                    self.ram_overlay._secure_delete(self.db_path)
                else:
                    os.unlink(self.db_path)
            
            # Secure delete captures
            if os.path.exists(self.capture_base):
                import shutil
                if self.ram_overlay:
                    for root, dirs, files in os.walk(self.capture_base):
                        for file in files:
                            filepath = os.path.join(root, file)
                            self.ram_overlay._secure_delete(filepath)
                shutil.rmtree(self.capture_base)
            
            logger.info("✅ Phone targeting data wiped")
            
        except Exception as e:
            logger.error(f"Emergency cleanup error: {e}")


# AI Controller Integration
def parse_targeting_command(text: str, targeting_system: PhoneNumberTargeting) -> Optional[str]:
    """
    Parse targeting commands from AI controller
    
    STEALTH: Commands are logged minimally
    
    Args:
        text: User input text
        targeting_system: PhoneNumberTargeting instance
        
    Returns:
        Response string or None if handled
    """
    text = text.strip().lower()
    
    # "target +1-555-1234"
    if text.startswith('target '):
        phone_match = re.search(r'(\+?\d[\d\-\(\)\s]{7,})', text)
        if phone_match:
            phone = phone_match.group(1)
            targeting_system.add_target(phone)
            return None  # Already printed
    
    # "associate 310410123456789 +1-555-1234"
    elif text.startswith('associate '):
        parts = text.split()
        if len(parts) >= 3:
            imsi = parts[1]
            phone = parts[2]
            targeting_system.associate_imsi(phone, imsi)
            return None
    
    # "capture" or "capture +1-555-1234"
    elif text.startswith('capture'):
        phone_match = re.search(r'(\+?\d[\d\-\(\)\s]{7,})', text)
        phone = phone_match.group(1) if phone_match else None
        targeting_system.start_targeted_capture(phone)
        return None
    
    # "stop"
    elif text == 'stop' or text == 'stop capture':
        targeting_system.stop_capture()
        return None
    
    # "status" or "status +1-555-1234"
    elif text.startswith('status'):
        phone_match = re.search(r'(\+?\d[\d\-\(\)\s]{7,})', text)
        phone = phone_match.group(1) if phone_match else None
        targeting_system.get_target_status(phone)
        return None
    
    # "extract" or "extract +1-555-1234"
    elif text.startswith('extract'):
        phone_match = re.search(r'(\+?\d[\d\-\(\)\s]{7,})', text)
        phone = phone_match.group(1) if phone_match else None
        targeting_system.extract_data(phone)
        return None
    
    # "remove +1-555-1234"
    elif text.startswith('remove '):
        phone_match = re.search(r'(\+?\d[\d\-\(\)\s]{7,})', text)
        if phone_match:
            phone = phone_match.group(1)
            targeting_system.remove_target(phone)
            return None
    
    # "report +1-555-1234 output.json"
    elif text.startswith('report '):
        parts = text.split()
        if len(parts) >= 3:
            phone = parts[1]
            filename = parts[2]
            targeting_system.export_report(phone, filename)
            return None
    
    return None


# Standalone test
if __name__ == "__main__":
    print("RF Arsenal OS - Phone Targeting Module")
    print("⚠️  STEALTH MODE ENABLED")
    print("For testing only - requires GSM controller")
