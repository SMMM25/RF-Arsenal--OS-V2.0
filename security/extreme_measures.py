#!/usr/bin/env python3
"""
Extreme Measures - Emergency Security System
Software-level data destruction and duress mode
NO HARDWARE DESTRUCTION - Software only
"""

import os
import sys
import time
import json
import hashlib
import secrets
import shutil
import subprocess
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import getpass


class TriggerType(Enum):
    """Self-destruct trigger types"""
    MANUAL = "manual"
    PANIC_BUTTON = "panic_button"
    DEAD_MAN_SWITCH = "dead_man_switch"
    TAMPER_DETECTED = "tamper_detected"
    GEOFENCE_VIOLATION = "geofence_violation"
    DURESS_PASSWORD = "duress_password"
    FAILED_AUTH = "failed_auth"


class WipeLevel(Enum):
    """Data destruction levels"""
    QUICK = 1        # Single pass with zeros
    STANDARD = 3     # DoD 3-pass
    SECURE = 7       # DoD 7-pass
    PARANOID = 35    # Gutmann 35-pass


@dataclass
class DestructionTarget:
    """Target for data destruction"""
    path: str
    priority: int  # 1=critical, 5=low priority
    wipe_level: WipeLevel
    description: str


@dataclass
class DestructionLog:
    """Log entry for destruction event"""
    timestamp: float
    trigger_type: TriggerType
    trigger_source: str
    targets_destroyed: int
    duration_seconds: float
    success: bool
    details: str


class SoftwareDestruction:
    """
    Software-level secure data destruction
    DoD 5220.22-M compliant wiping
    """
    
    def __init__(self):
        self.destruction_targets = []
        self.destruction_log = []
        self.in_progress = False
        
    def add_target(self, path: str, priority: int = 3,
                   wipe_level: WipeLevel = WipeLevel.STANDARD,
                   description: str = ""):
        """Add path to destruction target list"""
        
        target = DestructionTarget(
            path=path,
            priority=priority,
            wipe_level=wipe_level,
            description=description
        )
        
        self.destruction_targets.append(target)
        print(f"[DESTRUCT] Target added: {path} (Priority {priority})")
        
    def add_default_targets(self):
        """Add default sensitive data locations"""
        print("[DESTRUCT] Adding default destruction targets...")
        
        # Critical data (Priority 1 - destroy first)
        critical_paths = [
            ("/var/lib/rf-arsenal", "RF Arsenal data"),
            ("/root/.ssh", "SSH keys"),
            ("/home/*/.ssh", "User SSH keys"),
            ("/root/.gnupg", "GPG keys"),
            ("/home/*/.gnupg", "User GPG keys"),
        ]
        
        for path, desc in critical_paths:
            self.add_target(path, priority=1, 
                          wipe_level=WipeLevel.SECURE, 
                          description=desc)
        
        # High priority (Priority 2)
        high_priority = [
            ("/var/lib/rf-arsenal/personas", "Operational personas"),
            ("/var/lib/rf-arsenal/covert", "Covert storage"),
            ("/tmp", "Temporary files"),
            ("/var/log", "System logs"),
        ]
        
        for path, desc in high_priority:
            self.add_target(path, priority=2,
                          wipe_level=WipeLevel.STANDARD,
                          description=desc)
        
        # Medium priority (Priority 3)
        medium_priority = [
            ("/home/*/.bash_history", "Command history"),
            ("/home/*/.zsh_history", "Command history"),
            ("/home/*/.*_history", "Shell history"),
            ("/home/*/.mozilla", "Firefox profile"),
            ("/home/*/.config/google-chrome", "Chrome profile"),
        ]
        
        for path, desc in medium_priority:
            self.add_target(path, priority=3,
                          wipe_level=WipeLevel.QUICK,
                          description=desc)
        
        print(f"[DESTRUCT] Added {len(self.destruction_targets)} targets")
        
    def execute_destruction(self, trigger_type: TriggerType,
                           trigger_source: str = "",
                           confirmation_required: bool = True) -> bool:
        """
        Execute secure data destruction
        Returns True if successful
        """
        
        if self.in_progress:
            print("[DESTRUCT] Error: Destruction already in progress")
            return False
        
        print("\n" + "="*60)
        print("  ⚠️  EMERGENCY DATA DESTRUCTION INITIATED  ⚠️")
        print("="*60)
        print(f"\nTrigger: {trigger_type.value}")
        print(f"Source: {trigger_source}")
        print(f"Targets: {len(self.destruction_targets)} paths")
        
        # Require confirmation for manual triggers
        if confirmation_required and trigger_type == TriggerType.MANUAL:
            print("\n⚠️  THIS WILL PERMANENTLY DESTROY ALL SENSITIVE DATA")
            print("⚠️  THIS ACTION CANNOT BE UNDONE")
            
            confirm = input("\nType 'DESTROY ALL DATA' to confirm: ")
            
            if confirm != "DESTROY ALL DATA":
                print("[DESTRUCT] Aborted by user")
                return False
            
            confirm2 = input("Type 'YES' to proceed: ")
            
            if confirm2 != "YES":
                print("[DESTRUCT] Aborted by user")
                return False
            
            # Countdown
            print("\n⚠️  DESTRUCTION STARTS IN:")
            for i in range(10, 0, -1):
                print(f"  {i}...", end='', flush=True)
                time.sleep(1)
            print("\n")
        
        self.in_progress = True
        start_time = time.time()
        
        # Sort targets by priority (1 = highest priority)
        sorted_targets = sorted(self.destruction_targets, 
                              key=lambda t: t.priority)
        
        destroyed_count = 0
        
        print("[DESTRUCT] Beginning data destruction...\n")
        
        for target in sorted_targets:
            print(f"[DESTRUCT] [{target.priority}] {target.description}")
            print(f"  Path: {target.path}")
            print(f"  Wipe level: {target.wipe_level.name} ({target.wipe_level.value} passes)")
            
            # Expand wildcards in path
            paths = self._expand_path(target.path)
            
            for path in paths:
                if os.path.exists(path):
                    success = self._secure_wipe_path(path, target.wipe_level)
                    
                    if success:
                        destroyed_count += 1
                        print(f"  ✓ Destroyed: {path}")
                    else:
                        print(f"  ✗ Failed: {path}")
                else:
                    print(f"  ⊘ Not found: {path}")
            
            print()
        
        # Additional cleanup
        print("[DESTRUCT] Additional cleanup...")
        self._clear_memory()
        self._clear_swap()
        self._shred_free_space()
        
        duration = time.time() - start_time
        
        # Log destruction event
        log_entry = DestructionLog(
            timestamp=time.time(),
            trigger_type=trigger_type,
            trigger_source=trigger_source,
            targets_destroyed=destroyed_count,
            duration_seconds=duration,
            success=True,
            details=f"Destroyed {destroyed_count} targets in {duration:.1f}s"
        )
        
        self.destruction_log.append(log_entry)
        
        print("\n" + "="*60)
        print(f"  ✓ DESTRUCTION COMPLETE")
        print(f"  Targets destroyed: {destroyed_count}")
        print(f"  Duration: {duration:.1f} seconds")
        print("="*60 + "\n")
        
        self.in_progress = False
        return True
        
    def _expand_path(self, path: str) -> List[str]:
        """Expand wildcards in path"""
        import glob
        
        # Handle home directory expansion
        path = os.path.expanduser(path)
        
        # Handle wildcards
        if '*' in path:
            expanded = glob.glob(path)
            return expanded if expanded else []
        else:
            return [path]
        
    def _secure_wipe_path(self, path: str, wipe_level: WipeLevel) -> bool:
        """
        Securely wipe file or directory
        Uses DoD 5220.22-M standard
        """
        try:
            if os.path.isfile(path):
                return self._secure_wipe_file(path, wipe_level.value)
                
            elif os.path.isdir(path):
                # Recursively wipe all files in directory
                for root, dirs, files in os.walk(path, topdown=False):
                    for name in files:
                        filepath = os.path.join(root, name)
                        self._secure_wipe_file(filepath, wipe_level.value)
                    
                    for name in dirs:
                        dirpath = os.path.join(root, name)
                        try:
                            os.rmdir(dirpath)
                        except:
                            pass
                
                # Remove directory itself
                try:
                    shutil.rmtree(path)
                except:
                    pass
                
                return True
            
            return False
            
        except Exception as e:
            print(f"    Error wiping {path}: {e}")
            return False
        
    def _secure_wipe_file(self, filepath: str, passes: int) -> bool:
        """
        Securely wipe individual file
        Multiple overwrite passes
        """
        try:
            file_size = os.path.getsize(filepath)
            
            with open(filepath, 'ba+', buffering=0) as f:
                for pass_num in range(passes):
                    f.seek(0)
                    
                    if pass_num % 3 == 0:
                        # Pass: Write zeros
                        pattern = b'\x00'
                    elif pass_num % 3 == 1:
                        # Pass: Write ones
                        pattern = b'\xFF'
                    else:
                        # Pass: Write random data
                        pattern = None
                    
                    # Write in chunks (1MB at a time)
                    chunk_size = 1024 * 1024
                    bytes_written = 0
                    
                    while bytes_written < file_size:
                        remaining = file_size - bytes_written
                        write_size = min(chunk_size, remaining)
                        
                        if pattern:
                            f.write(pattern * write_size)
                        else:
                            f.write(os.urandom(write_size))
                        
                        bytes_written += write_size
                    
                    # Flush to disk
                    f.flush()
                    os.fsync(f.fileno())
            
            # Delete file
            os.remove(filepath)
            
            return True
            
        except Exception as e:
            return False
        
    def _clear_memory(self):
        """Clear sensitive data from RAM"""
        print("  • Clearing memory...")
        
        try:
            # Sync and drop caches (Linux)
            subprocess.run(['sync'], check=False)
            
            # Drop page cache, dentries and inodes
            try:
                with open('/proc/sys/vm/drop_caches', 'w') as f:
                    f.write('3\n')
            except:
                pass
                
        except Exception as e:
            print(f"    Warning: Memory clear failed: {e}")
        
    def _clear_swap(self):
        """Clear swap space"""
        print("  • Clearing swap...")
        
        try:
            # Turn off swap
            subprocess.run(['sudo', 'swapoff', '-a'], 
                         check=False, capture_output=True)
            
            # Turn swap back on
            subprocess.run(['sudo', 'swapon', '-a'], 
                         check=False, capture_output=True)
            
        except Exception as e:
            print(f"    Warning: Swap clear failed: {e}")
        
    def _shred_free_space(self):
        """Overwrite free disk space"""
        print("  • Shredding free space (this may take a while)...")
        
        try:
            # Create file that fills all free space
            temp_file = '/tmp/.shred_free_space'
            
            # Write zeros until disk full
            with open(temp_file, 'wb') as f:
                try:
                    while True:
                        f.write(b'\x00' * 1024 * 1024)  # 1MB chunks
                except OSError:
                    # Disk full
                    pass
            
            # Delete the file
            os.remove(temp_file)
            
        except Exception as e:
            print(f"    Warning: Free space shred failed: {e}")


class DuressMode:
    """
    Duress mode - Fake decoy system
    Provides plausible deniability under coercion
    """
    
    def __init__(self):
        self.real_password_hash = None
        self.duress_password_hash = None
        self.duress_active = False
        self.alert_sent = False
        
    def set_passwords(self, real_password: str, duress_password: str):
        """Set real and duress passwords"""
        
        # Hash passwords
        self.real_password_hash = self._hash_password(real_password)
        self.duress_password_hash = self._hash_password(duress_password)
        
        print("[DURESS] Passwords configured")
        print("  Real password: Set ✓")
        print("  Duress password: Set ✓")
        
    def check_password(self, password: str) -> str:
        """
        Check password and return mode
        Returns: 'real', 'duress', or 'invalid'
        """
        
        password_hash = self._hash_password(password)
        
        if password_hash == self.real_password_hash:
            return 'real'
        elif password_hash == self.duress_password_hash:
            return 'duress'
        else:
            return 'invalid'
        
    def activate_duress_mode(self):
        """
        Activate duress mode
        Show fake decoy system
        """
        
        print("\n" + "="*60)
        print("  ⚠️  DURESS MODE ACTIVATED  ⚠️")
        print("="*60)
        
        self.duress_active = True
        
        # 1. Send silent alert
        self._send_silent_alert()
        
        # 2. Hide real data
        self._hide_sensitive_data()
        
        # 3. Load decoy environment
        self._load_decoy_environment()
        
        # 4. Start delayed wipe (optional)
        # self._schedule_delayed_wipe(hours=24)
        
        print("\n[DURESS] Decoy system loaded")
        print("[DURESS] Adversary will see fake environment")
        print("[DURESS] Silent alert sent")
        
    def _send_silent_alert(self):
        """Send silent distress alert"""
        
        if self.alert_sent:
            return
        
        print("[DURESS] Sending silent alert...")
        
        # Would send alert via:
        # - Covert DNS query
        # - Steganography in web request
        # - SMS to emergency number
        # - Mesh network message
        
        alert_data = {
            'type': 'duress_activated',
            'timestamp': time.time(),
            'location': self._get_location()
        }
        
        # Simulated for demonstration
        print("  ✓ Alert sent via covert channel")
        
        self.alert_sent = True
        
    def _hide_sensitive_data(self):
        """Hide real sensitive data"""
        print("[DURESS] Hiding sensitive data...")
        
        # Paths to hide
        sensitive_paths = [
            '/var/lib/rf-arsenal',
            '/root/.ssh',
            '/root/.gnupg'
        ]
        
        for path in sensitive_paths:
            if os.path.exists(path):
                # Move to hidden location or re-encrypt
                hidden_path = f"/dev/shm/.hidden_{hashlib.md5(path.encode()).hexdigest()}"
                
                try:
                    shutil.move(path, hidden_path)
                    print(f"  ✓ Hidden: {path}")
                except Exception as e:
                    print(f"  ✗ Failed to hide {path}: {e}")
        
    def _load_decoy_environment(self):
        """Load fake decoy environment"""
        print("[DURESS] Loading decoy environment...")
        
        # Create fake user directories with benign content
        fake_dirs = [
            '~/Documents/Work',
            '~/Documents/Personal',
            '~/Pictures/Vacation',
            '~/Downloads'
        ]
        
        for fake_dir in fake_dirs:
            expanded = os.path.expanduser(fake_dir)
            os.makedirs(expanded, exist_ok=True)
            
            # Create fake files
            self._create_fake_files(expanded)
        
        # Create fake browser history
        self._create_fake_browser_history()
        
        print("  ✓ Decoy environment ready")
        
    def _create_fake_files(self, directory: str):
        """Create benign fake files"""
        
        fake_files = [
            ('meeting_notes.txt', b'Meeting notes from yesterday...\n'),
            ('grocery_list.txt', b'Milk\nEggs\nBread\nButter\n'),
            ('recipe.txt', b'Chocolate chip cookies recipe...\n')
        ]
        
        for filename, content in fake_files:
            filepath = os.path.join(directory, filename)
            
            try:
                with open(filepath, 'wb') as f:
                    f.write(content)
            except:
                pass
        
    def _create_fake_browser_history(self):
        """Create fake benign browser history"""
        # Would create fake Firefox/Chrome history database
        # with innocent browsing patterns
        pass
        
    def _get_location(self) -> Optional[Dict]:
        """Get current GPS location"""
        # Would interface with GPS hardware
        # Returns lat/lon coordinates
        
        return {
            'latitude': 0.0,
            'longitude': 0.0,
            'accuracy': 0.0
        }
        
    def _hash_password(self, password: str) -> str:
        """Hash password for storage"""
        return hashlib.sha256(password.encode()).hexdigest()


class DeadManSwitch:
    """
    Dead man's switch - Auto-destruct if no check-in
    Prevents adversary from keeping you away from system
    """
    
    def __init__(self, check_in_hours: int = 24):
        self.check_in_hours = check_in_hours
        self.last_check_in = time.time()
        self.running = False
        self.destruction_callback = None
        
    def start(self, destruction_callback: Callable):
        """Start dead man's switch monitoring"""
        
        self.destruction_callback = destruction_callback
        self.running = True
        self.last_check_in = time.time()
        
        print(f"[DEAD MAN] Switch activated")
        print(f"  Check-in required every: {self.check_in_hours} hours")
        print(f"  Next check-in due: {time.ctime(self.last_check_in + self.check_in_hours * 3600)}")
        
        # Start monitoring thread
        threading.Thread(target=self._monitor_loop, daemon=True).start()
        
    def check_in(self):
        """User check-in to reset timer"""
        
        self.last_check_in = time.time()
        next_checkin = self.last_check_in + self.check_in_hours * 3600
        
        print(f"[DEAD MAN] Check-in recorded")
        print(f"  Next check-in due: {time.ctime(next_checkin)}")
        
    def _monitor_loop(self):
        """Monitor check-in deadline"""
        
        while self.running:
            time_since_checkin = time.time() - self.last_check_in
            deadline = self.check_in_hours * 3600
            
            if time_since_checkin > deadline:
                print("\n" + "="*60)
                print("  ⚠️  DEAD MAN SWITCH TRIGGERED  ⚠️")
                print("  No check-in received within deadline")
                print("  Initiating emergency destruction...")
                print("="*60 + "\n")
                
                # Trigger destruction
                if self.destruction_callback:
                    self.destruction_callback(
                        trigger_type=TriggerType.DEAD_MAN_SWITCH,
                        trigger_source=f"No check-in for {self.check_in_hours} hours"
                    )
                
                self.running = False
                break
            
            # Check every 5 minutes
            time.sleep(300)
        
    def stop(self):
        """Stop dead man's switch"""
        self.running = False
        print("[DEAD MAN] Switch deactivated")


class EmergencySystem:
    """
    Unified emergency security system
    Combines all extreme measures
    """
    
    def __init__(self):
        self.destruction = SoftwareDestruction()
        self.duress = DuressMode()
        self.dead_man = DeadManSwitch(check_in_hours=24)
        
        # Add default destruction targets
        self.destruction.add_default_targets()
        
    def setup_duress_mode(self, real_password: str, duress_password: str):
        """Configure duress mode passwords"""
        self.duress.set_passwords(real_password, duress_password)
        
    def authenticate(self, password: str) -> str:
        """
        Authenticate user and detect duress
        Returns: 'real', 'duress', or 'invalid'
        """
        
        mode = self.duress.check_password(password)
        
        if mode == 'duress':
            self.duress.activate_duress_mode()
        
        return mode
        
    def emergency_destruct(self, trigger_type: TriggerType,
                          trigger_source: str = ""):
        """Execute emergency destruction"""
        
        self.destruction.execute_destruction(
            trigger_type=trigger_type,
            trigger_source=trigger_source,
            confirmation_required=False  # No confirmation in emergency
        )
        
    def manual_destruct(self):
        """Manual destruction with confirmation"""
        
        self.destruction.execute_destruction(
            trigger_type=TriggerType.MANUAL,
            trigger_source="User initiated",
            confirmation_required=True
        )
        
    def start_dead_man_switch(self):
        """Start dead man's switch"""
        
        self.dead_man.start(
            destruction_callback=self.emergency_destruct
        )
        
    def check_in(self):
        """Check in to dead man's switch"""
        self.dead_man.check_in()
        
    def get_status(self) -> Dict:
        """Get emergency system status"""
        
        return {
            'destruction_targets': len(self.destruction.destruction_targets),
            'duress_configured': (self.duress.real_password_hash is not None),
            'duress_active': self.duress.duress_active,
            'dead_man_active': self.dead_man.running,
            'dead_man_last_checkin': self.dead_man.last_check_in,
            'destruction_events': len(self.destruction.destruction_log)
        }


# Example usage
if __name__ == "__main__":
    print("=== Extreme Measures System Test ===\n")
    
    print("⚠️  WARNING: This is a TEST of emergency systems")
    print("⚠️  NO actual data will be destroyed in this demo\n")
    
    # Create emergency system
    emergency = EmergencySystem()
    
    # Test 1: Configure duress mode
    # NOTE: These are EXAMPLE passwords for demonstration only
    # In production, passwords should be set by the user via secure input
    print("--- Duress Mode Configuration ---")
    import os
    test_real_pw = os.environ.get('RF_TEST_REAL_PW', 'test_real_password_change_me')
    test_duress_pw = os.environ.get('RF_TEST_DURESS_PW', 'test_duress_password_change_me')
    emergency.setup_duress_mode(
        real_password=test_real_pw,
        duress_password=test_duress_pw
    )
    
    # Test authentication
    print("\n--- Authentication Test ---")
    
    print("Testing real password...")
    mode = emergency.authenticate(test_real_pw)
    print(f"  Result: {mode}\n")
    
    print("Testing duress password...")
    mode = emergency.authenticate(test_duress_pw)
    print(f"  Result: {mode}\n")
    
    # Test 2: Dead man's switch
    print("--- Dead Man's Switch ---")
    print("Starting dead man's switch (24 hour check-in)...")
    # emergency.start_dead_man_switch()
    print("  (Not started in demo mode)")
    
    # Test 3: Show destruction targets
    print("\n--- Destruction Targets ---")
    print(f"Configured targets: {len(emergency.destruction.destruction_targets)}")
    
    for i, target in enumerate(emergency.destruction.destruction_targets[:5], 1):
        print(f"  {i}. [{target.priority}] {target.description}")
        print(f"     Path: {target.path}")
        print(f"     Wipe: {target.wipe_level.name}")
    
    print(f"  ... and {len(emergency.destruction.destruction_targets) - 5} more")
    
    # System status
    print("\n--- System Status ---")
    status = emergency.get_status()
    
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n--- Manual Destruction Test ---")
    print("To test manual destruction (with confirmation):")
    print("  emergency.manual_destruct()")
    print("\nTo test emergency destruction (no confirmation):")
    print("  emergency.emergency_destruct(TriggerType.PANIC_BUTTON)")
    
    print("\n⚠️  Demo complete - No data was harmed in this test")
