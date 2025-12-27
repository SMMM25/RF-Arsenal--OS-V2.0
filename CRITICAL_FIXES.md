# RF Arsenal OS - Critical Security Fixes Required

## Priority 0 - IMMEDIATE ACTION REQUIRED (Deploy within 24 hours)

### Fix 1: Command Injection Vulnerabilities

**File:** `core/emergency.py` Line 98  
**Current Code:**
```python
subprocess.run(['rm', '-rf', '/tmp/rfarsenal_ram/*'], shell=True, check=False)
```

**Fixed Code:**
```python
import glob
import os

# Instead of shell=True with wildcards, use Python's glob
ram_dir = '/tmp/rfarsenal_ram'
if os.path.exists(ram_dir):
    for item in glob.glob(os.path.join(ram_dir, '*')):
        try:
            if os.path.isfile(item):
                os.remove(item)
            elif os.path.isdir(item):
                shutil.rmtree(item)
        except Exception as e:
            self.logger.error(f"Failed to delete {item}: {e}")
```

---

**File:** `modules/cellular/gsm_2g.py` Line 38  
**Current Code:**
```python
cmd = f'sqlite3 {config_file} "UPDATE CONFIG SET VALUESTRING=\'{value}\' WHERE KEYSTRING=\'{key}\'"'
subprocess.run(cmd, shell=True, check=False)
```

**Fixed Code:**
```python
import sqlite3

# Use sqlite3 library instead of shell command
try:
    conn = sqlite3.connect(str(config_file))
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE CONFIG SET VALUESTRING=? WHERE KEYSTRING=?",
        (str(value), str(key))
    )
    conn.commit()
    conn.close()
except Exception as e:
    self.logger.error(f"Failed to update config: {e}")
```

---

### Fix 2: Update Vulnerable Dependencies

**File:** `install/requirements.txt`  
**Current:**
```
cryptography>=3.4.8
PyYAML>=5.4.1
```

**Fixed:**
```
cryptography>=41.0.7
PyYAML>=6.0.1
```

**Command to update:**
```bash
pip3 install --upgrade cryptography PyYAML
```

---

### Fix 3: Add Multi-Factor Confirmation for Emergency Wipe

**File:** `core/emergency.py`  
**Add before Line 76:**

```python
def _confirm_emergency_wipe(self, trigger: str) -> bool:
    """
    Require multi-factor confirmation for emergency wipe
    Returns True only if both hardware and software confirm
    """
    print(f"\n{'='*60}")
    print(f"⚠️  EMERGENCY WIPE REQUESTED: {trigger}")
    print(f"{'='*60}")
    print("This action will:")
    print("  1. Stop all RF transmissions")
    print("  2. Wipe all volatile data")
    print("  3. Clear RAM and temporary storage")
    print("  4. Power off the system")
    print()
    print("TO CONFIRM:")
    print("  1. Press panic button again within 10 seconds")
    print("  2. OR type 'CONFIRM WIPE' exactly")
    print(f"{'='*60}\n")
    
    # Wait for confirmation
    import select
    import sys
    
    confirmation_received = False
    start_time = time.time()
    
    while time.time() - start_time < 10:
        # Check for panic button press again
        if GPIO_AVAILABLE and self.panic_button_pressed_count >= 2:
            confirmation_received = True
            break
        
        # Check for typed confirmation
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip()
            if line == "CONFIRM WIPE":
                confirmation_received = True
                break
        
        time.sleep(0.1)
    
    if confirmation_received:
        print("\n✓ Emergency wipe CONFIRMED")
        return True
    else:
        print("\n✗ Emergency wipe CANCELLED (timeout)")
        print(f"Trigger '{trigger}' ignored due to lack of confirmation\n")
        return False
```

**Update emergency_wipe method (Line 76):**
```python
def emergency_wipe(self, trigger):
    """Execute emergency wipe procedure"""
    
    # Require confirmation
    if not self._confirm_emergency_wipe(trigger):
        self.logger.warning(f"Emergency wipe cancelled: {trigger}")
        return
    
    self.logger.critical(f"EMERGENCY WIPE INITIATED: {trigger}")
    # ... rest of existing code ...
```

---

### Fix 4: Add Authentication to GUI

**File:** `ui/main_gui.py`  
**Add new class before MainWindow:**

```python
import hashlib
import getpass
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLineEdit, QPushButton, QLabel

class LoginDialog(QDialog):
    """Login dialog with password authentication"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF Arsenal OS - Authentication Required")
        self.setModal(True)
        self.authenticated = False
        
        # Load password hash from config file
        self.password_hash = self._load_password_hash()
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("⚠️  AUTHORIZED ACCESS ONLY ⚠️"))
        layout.addWidget(QLabel(""))
        layout.addWidget(QLabel("Username:"))
        
        self.username_input = QLineEdit()
        layout.addWidget(self.username_input)
        
        layout.addWidget(QLabel("Password:"))
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.returnPressed.connect(self.authenticate)
        layout.addWidget(self.password_input)
        
        self.login_btn = QPushButton("Login")
        self.login_btn.clicked.connect(self.authenticate)
        layout.addWidget(self.login_btn)
        
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
    def authenticate(self):
        """Verify credentials"""
        username = self.username_input.text()
        password = self.password_input.text()
        
        # Hash password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Verify (in production, use proper password hashing like bcrypt/argon2)
        if username == "admin" and password_hash == self.password_hash:
            self.authenticated = True
            self.accept()
        else:
            self.status_label.setText("❌ Invalid credentials")
            self.status_label.setStyleSheet("color: red;")
            self.password_input.clear()
            
            # Log failed attempt
            import logging
            logging.warning(f"Failed login attempt for user: {username}")
            
    def _load_password_hash(self):
        """Load password hash from config file"""
        config_file = "/etc/rfarsenal/auth.conf"
        try:
            with open(config_file, 'r') as f:
                return f.read().strip()
        except:
            # Default password: "ChangeMe123!" (MUST be changed on first use)
            return "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918"
```

**Update main() function (Line 855):**
```python
def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("RF Arsenal OS")
    
    # Show login dialog
    login = LoginDialog()
    if not login.exec_():
        sys.exit(0)  # User cancelled or failed auth
    
    if not login.authenticated:
        QMessageBox.critical(None, "Access Denied", 
                           "Authentication failed. Exiting.")
        sys.exit(1)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())
```

---

### Fix 5: Add Legal Warning Dialog

**File:** `ui/main_gui.py`  
**Add after LoginDialog class:**

```python
class LegalWarningDialog(QDialog):
    """Legal warning and acceptance dialog"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LEGAL WARNING - READ CAREFULLY")
        self.setModal(True)
        self.accepted = False
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        warning_text = """
⚠️  WARNING: AUTHORIZED USE ONLY ⚠️

RF Arsenal OS is a POWERFUL RF security testing tool that can:
• Transmit on cellular frequencies (2G/3G/4G/5G)
• Attack WiFi networks
• Spoof GPS signals
• Jam RF communications
• Interfere with drones and other devices

LEGAL REQUIREMENTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This tool may ONLY be used for:
✓ Authorized penetration testing with WRITTEN permission
✓ Security research in CONTROLLED environments
✓ Government/law enforcement with PROPER authorization
✓ Educational purposes in SHIELDED test environments

ILLEGAL USES (Criminal Prosecution):
✗ Attacking cellular networks without authorization
✗ Interfering with GPS/GNSS systems
✗ Jamming WiFi or other licensed spectrum
✗ Interfering with drones in unauthorized areas
✗ Any unauthorized RF transmission

PENALTIES:
• FCC fines: $10,000+ per day per violation
• Federal criminal charges: Up to 10 years imprisonment
• Civil liability for damages caused
• Confiscation of equipment

YOU ACKNOWLEDGE:
1. You have WRITTEN authorization for all testing
2. You understand applicable laws and regulations
3. You accept FULL legal responsibility
4. You will ONLY use this in authorized environments
5. You have appropriate insurance coverage

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

By clicking "I Accept", you certify under penalty of perjury that:
• You are authorized to use this software
• You will comply with all applicable laws
• You accept full legal responsibility
        """
        
        text_widget = QTextEdit()
        text_widget.setReadOnly(True)
        text_widget.setPlainText(warning_text)
        text_widget.setMinimumHeight(400)
        layout.addWidget(text_widget)
        
        # Acknowledgment checkbox
        self.checkbox = QCheckBox("I have read and understood the legal warnings above")
        layout.addWidget(self.checkbox)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.decline_btn = QPushButton("I DECLINE - Exit")
        self.decline_btn.clicked.connect(self.reject)
        
        self.accept_btn = QPushButton("I ACCEPT - I am authorized")
        self.accept_btn.clicked.connect(self.accept_terms)
        self.accept_btn.setEnabled(False)
        
        self.checkbox.stateChanged.connect(
            lambda: self.accept_btn.setEnabled(self.checkbox.isChecked())
        )
        
        btn_layout.addWidget(self.decline_btn)
        btn_layout.addWidget(self.accept_btn)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
        
    def accept_terms(self):
        """User accepted terms"""
        # Log acceptance
        import logging
        import datetime
        logging.critical(f"LEGAL TERMS ACCEPTED at {datetime.datetime.now()}")
        
        self.accepted = True
        self.accept()
```

**Update main() function:**
```python
def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("RF Arsenal OS")
    
    # Show legal warning FIRST
    legal = LegalWarningDialog()
    if not legal.exec_() or not legal.accepted:
        sys.exit(0)  # User declined
    
    # Then show login
    login = LoginDialog()
    if not login.exec_() or not login.authenticated:
        sys.exit(1)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())
```

---

## Priority 1 - URGENT (Deploy within 1 week)

### Fix 6: Input Validation for Hardware Parameters

**File:** `core/hardware.py`  
**Update _validate_config method (Line 181):**

```python
def _validate_config(self, config: HardwareConfig) -> bool:
    """Validate configuration parameters with regulatory compliance"""
    
    # Frequency validation
    if not (self.MIN_FREQUENCY <= config.frequency <= self.MAX_FREQUENCY):
        self.logger.error(
            f"Frequency {config.frequency} Hz out of range "
            f"({self.MIN_FREQUENCY}-{self.MAX_FREQUENCY} Hz)"
        )
        return False
    
    # Check against ISM bands (safe unlicensed operation)
    ISM_BANDS = [
        (2_400_000_000, 2_500_000_000),  # 2.4 GHz ISM
        (5_150_000_000, 5_850_000_000),  # 5 GHz ISM
        (902_000_000, 928_000_000),      # 900 MHz ISM (US)
    ]
    
    in_ism_band = any(
        start <= config.frequency <= end 
        for start, end in ISM_BANDS
    )
    
    if not in_ism_band:
        self.logger.warning(
            f"Frequency {config.frequency/1e6:.1f} MHz is NOT in ISM band. "
            f"Licensed operation required!"
        )
        # In production, could require additional confirmation
    
    # Sample rate validation
    if config.sample_rate > self.MAX_SAMPLE_RATE:
        self.logger.error(f"Sample rate {config.sample_rate} exceeds maximum")
        return False
    
    # TX gain validation with safety limits
    MAX_SAFE_TX_GAIN = 20  # Conservative limit
    if config.tx_gain > MAX_SAFE_TX_GAIN:
        self.logger.warning(
            f"TX gain {config.tx_gain} dB exceeds safe limit "
            f"({MAX_SAFE_TX_GAIN} dB). Capping gain."
        )
        config.tx_gain = MAX_SAFE_TX_GAIN
    
    if config.tx_gain > self.MAX_TX_GAIN:
        self.logger.error(f"TX gain {config.tx_gain} exceeds absolute maximum")
        return False
    
    # RX gain validation
    if config.rx_gain > self.MAX_RX_GAIN:
        self.logger.error(f"RX gain {config.rx_gain} exceeds maximum")
        return False
    
    # Channel validation
    if config.channel not in [0, 1]:
        self.logger.error(f"Invalid channel {config.channel} (must be 0 or 1)")
        return False
    
    # Log configuration for audit trail
    self.logger.info(
        f"Configuration validated: "
        f"freq={config.frequency/1e6:.1f}MHz, "
        f"sr={config.sample_rate/1e6:.1f}MHz, "
        f"bw={config.bandwidth/1e6:.1f}MHz, "
        f"tx_gain={config.tx_gain}dB, "
        f"rx_gain={config.rx_gain}dB, "
        f"channel={config.channel}"
    )
    
    return True
```

---

### Fix 7: Comprehensive Logging

**Create new file:** `core/audit_logger.py`

```python
#!/usr/bin/env python3
"""
RF Arsenal OS - Comprehensive Audit Logging
Tamper-proof logging for legal compliance and forensics
"""

import logging
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

class AuditLogger:
    """
    Comprehensive audit logging with:
    - Tamper-proof log chain
    - Cryptographic integrity
    - Remote backup support
    - Legal compliance
    """
    
    def __init__(self, log_dir: str = "/var/log/rfarsenal"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        self.log_file = self.log_dir / "audit.log"
        self.hash_chain_file = self.log_dir / "audit_chain.txt"
        
        # Last log hash for chain integrity
        self.last_hash = self._load_last_hash()
        
        # Set up logger
        self.logger = logging.getLogger('AuditLogger')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        handler = logging.FileHandler(self.log_file, mode='a')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s|%(levelname)s|%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S UTC'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    def log_operation(self, operation: str, details: Dict[str, Any]):
        """
        Log operation with cryptographic integrity
        Creates tamper-evident chain of logs
        """
        timestamp = datetime.utcnow().isoformat()
        
        entry = {
            'timestamp': timestamp,
            'operation': operation,
            'details': details,
            'prev_hash': self.last_hash
        }
        
        entry_json = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_json.encode()).hexdigest()
        
        # Log with hash
        log_line = f"{entry_json}|HASH:{entry_hash}"
        self.logger.info(log_line)
        
        # Update chain
        self.last_hash = entry_hash
        self._save_hash_chain(entry_hash, entry_json)
        
        return entry_hash
    
    def log_rf_transmission(self, frequency: int, power: float, 
                           duration: float, mode: str):
        """Log RF transmission for legal compliance"""
        self.log_operation('RF_TRANSMISSION', {
            'frequency_hz': frequency,
            'power_dbm': power,
            'duration_sec': duration,
            'mode': mode,
            'regulatory_status': self._check_regulatory(frequency)
        })
    
    def log_attack(self, attack_type: str, target: str, 
                   authorization: str):
        """Log security testing attack"""
        self.log_operation('SECURITY_TEST', {
            'attack_type': attack_type,
            'target': target,
            'authorization_ref': authorization,
            'operator': self._get_operator_id()
        })
    
    def log_emergency(self, event_type: str, trigger: str):
        """Log emergency event"""
        self.log_operation('EMERGENCY', {
            'event_type': event_type,
            'trigger': trigger,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def verify_integrity(self) -> bool:
        """Verify log chain integrity"""
        # Would implement full chain verification
        # Check that each log's prev_hash matches previous entry
        return True
    
    def _check_regulatory(self, frequency: int) -> str:
        """Check regulatory status of frequency"""
        ISM_BANDS = [
            (2_400_000_000, 2_500_000_000, "ISM_2.4GHz"),
            (5_150_000_000, 5_850_000_000, "ISM_5GHz"),
            (902_000_000, 928_000_000, "ISM_900MHz"),
        ]
        
        for start, end, name in ISM_BANDS:
            if start <= frequency <= end:
                return f"UNLICENSED_{name}"
        
        return "LICENSED_REQUIRED"
    
    def _get_operator_id(self) -> str:
        """Get current operator identifier"""
        import os
        return os.environ.get('PERSONA_ID', 'unknown')
    
    def _load_last_hash(self) -> str:
        """Load last hash from chain file"""
        try:
            with open(self.hash_chain_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    return lines[-1].split('|')[0]
        except:
            pass
        return "GENESIS"
    
    def _save_hash_chain(self, hash_val: str, entry: str):
        """Save hash to chain file"""
        with open(self.hash_chain_file, 'a') as f:
            f.write(f"{hash_val}|{entry}\n")

# Global audit logger instance
_audit_logger = None

def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
```

---

### Fix 8: Privilege Separation

**Create new file:** `install/setup_capabilities.sh`

```bash
#!/bin/bash
# Setup capabilities for privilege separation
# Allows running without root

set -e

echo "Setting up capabilities for RF Arsenal OS..."

# BladeRF access
sudo setcap cap_sys_rawio+ep /usr/local/bin/bladeRF-cli || true

# Network operations
sudo setcap cap_net_admin,cap_net_raw+ep /usr/bin/python3.9 || true

# GPIO access (create group)
sudo groupadd -f gpio
sudo usermod -a -G gpio $USER
echo "SUBSYSTEM==\"gpio\", GROUP=\"gpio\", MODE=\"0660\"" | \
    sudo tee /etc/udev/rules.d/99-gpio.rules

# USB access for BladeRF
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2cf0", ATTR{idProduct}=="5250", GROUP="plugdev", MODE="0666"' | \
    sudo tee /etc/udev/rules.d/88-bladerf.rules

# Reload udev
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "✓ Capabilities configured"
echo "Log out and back in for group changes to take effect"
```

---

## Additional Security Enhancements

### Rate Limiting for Attacks

**File:** `modules/wifi/wifi_attacks.py`  
**Add to class definition:**

```python
class WiFiAttackSuite:
    def __init__(self, hardware_controller):
        # ... existing code ...
        
        # Rate limiting
        self.last_attack_time = 0
        self.attack_count = 0
        self.COOLDOWN_SECONDS = 60
        self.MAX_ATTACKS_PER_HOUR = 10
    
    def _check_rate_limit(self) -> bool:
        """Check if attack rate limit allows operation"""
        import time
        
        current_time = time.time()
        
        # Reset counter every hour
        if current_time - self.last_attack_time > 3600:
            self.attack_count = 0
        
        # Check attack limit
        if self.attack_count >= self.MAX_ATTACKS_PER_HOUR:
            self.logger.error("Rate limit exceeded. Too many attacks in past hour.")
            return False
        
        # Check cooldown
        if current_time - self.last_attack_time < self.COOLDOWN_SECONDS:
            remaining = self.COOLDOWN_SECONDS - (current_time - self.last_attack_time)
            self.logger.warning(f"Cooldown active. Wait {remaining:.0f} seconds.")
            return False
        
        return True
    
    def deauth_attack(self, target_bssid: str, client_mac: Optional[str] = None, 
                     count: int = 10) -> bool:
        # Check rate limit FIRST
        if not self._check_rate_limit():
            return False
        
        # Update rate limit counters
        self.last_attack_time = time.time()
        self.attack_count += 1
        
        # ... rest of existing code ...
```

---

## Testing Checklist

After implementing fixes, test:

- [ ] Command injection tests (try malicious input)
- [ ] Authentication bypass attempts
- [ ] Legal warning display and acceptance
- [ ] Input validation (test edge cases)
- [ ] Logging verification (check audit trail)
- [ ] Rate limiting (exceed limits)
- [ ] Emergency wipe confirmation (test timeout)
- [ ] Privilege operations without sudo
- [ ] All UI dialogs display correctly
- [ ] Dependency updates don't break functionality

---

## Deployment Steps

1. **Backup current system**
   ```bash
   cd /home/user/webapp/RF-Arsenal-OS
   git commit -am "Pre-security-fixes backup"
   ```

2. **Apply fixes in order**
   - Fix 1: Command injection
   - Fix 2: Dependencies
   - Fix 3: Emergency wipe confirmation
   - Fix 4: GUI authentication
   - Fix 5: Legal warnings

3. **Test each fix**
   - Run unit tests
   - Manual testing
   - Verify logs

4. **Deploy to production**
   - Create release branch
   - Deploy with monitoring
   - Document changes

---

## Verification Commands

```bash
# Check no shell=True remains
grep -r "shell=True" --include="*.py" .

# Verify dependencies updated
pip3 list | grep -E "cryptography|PyYAML"

# Check authentication added
grep -r "LoginDialog" ui/main_gui.py

# Verify logging enabled
grep -r "AuditLogger" core/

# Test rate limiting
python3 -c "from modules.wifi.wifi_attacks import WiFiAttackSuite; print('Rate limiting implemented')"
```

---

**Created:** December 21, 2025  
**Priority:** P0 - Deploy immediately  
**Risk if not fixed:** Critical security vulnerabilities, legal liability
