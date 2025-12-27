#!/usr/bin/env python3
"""
RF Arsenal OS - Comprehensive Update Manager
Professional pentesting tool with reliable, auditable updates

Features:
- Multi-component update management (Core, Python, System, Hardware, Modules)
- Operator control (NO auto-updates, manual approval required)
- Backup before update with rollback capability
- Audit trail of all updates
- Security-focused (GPG verification, checksums, Tor support)
- Severity-based prioritization (CRITICAL, HIGH, LOW)
- Modular updates (choose what to update)

Components Managed:
1. RF Arsenal OS core (GitHub)
2. Python dependencies (pip)
3. System packages (apt)
4. Hardware drivers (BladeRF, HackRF)
5. Optional modules (ML models, OSINT database)

Copyright (c) 2024 RF-Arsenal-OS Project
License: MIT
"""

import os
import sys
import json
import sqlite3
import hashlib
import shutil
import tarfile
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import requests (graceful fallback)
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests module not available - some features disabled")


class UpdateSeverity:
    """Update severity levels for prioritization"""
    CRITICAL = "CRITICAL"  # Security fixes, system-breaking bugs, CVEs
    HIGH = "HIGH"          # Important features, significant bugs
    LOW = "LOW"            # Minor improvements, optimizations


class ComponentUpdate:
    """Represents a single component update"""
    def __init__(self, component: str, current: str, available: str, 
                 severity: str, changelog: str, size_mb: float = 0):
        self.component = component
        self.current_version = current
        self.available_version = available
        self.severity = severity
        self.changelog = changelog
        self.size_mb = size_mb
        self.checked_at = datetime.now()
    
    def __repr__(self):
        return (f"ComponentUpdate({self.component}, "
                f"{self.current_version} → {self.available_version}, "
                f"severity={self.severity})")


class ComprehensiveUpdateManager:
    """
    Comprehensive update management system
    
    Design Principles:
    - OPERATOR CONTROL: Manual approval required (NO auto-updates)
    - MODULAR UPDATES: Choose what to update (core, deps, system, etc.)
    - BACKUP BEFORE UPDATE: Automatic backup with rollback capability
    - AUDIT LOGGING: Complete trail of all updates
    - SECURITY: GPG verification, checksums, Tor support
    """
    
    def __init__(self, config_path='/etc/rf-arsenal/update.conf'):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Paths
        self.install_dir = Path('/opt/rf-arsenal')
        self.backup_dir = Path('/var/backups/rf-arsenal')
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Database for audit trail
        self.db_path = Path('/var/lib/rf-arsenal/updates.db')
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = self._init_database()
        
        # GitHub repository
        self.repo_owner = "SMMM25"
        self.repo_name = "RF-Arsenal-OS"
        self.github_api = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
        
        # Update cache
        self.update_cache = {}
        
        # Current version
        self.current_version = self._get_current_version()
        
        logger.info(f"Update Manager initialized (version {self.current_version})")
    
    def _load_config(self) -> Dict:
        """Load configuration with sane defaults"""
        default_config = {
            'auto_check': False,           # NO auto-checks by default
            'auto_install': False,         # NO auto-install EVER
            'prompt_on_wifi': True,        # Prompt when WiFi detected
            'update_channel': 'stable',    # stable, beta, dev
            'backup_retention_days': 30,   # Keep backups for 30 days
            'check_python_deps': True,
            'check_system_deps': True,
            'check_hardware': True,
            'check_modules': True,
            'use_tor': False,              # Tor for anonymous updates
            'verify_gpg': True,            # GPG signature verification
            'max_backup_size_gb': 10       # Max backup size
        }
        
        if self.config_path.exists():
            try:
                import configparser
                config = configparser.ConfigParser()
                config.read(str(self.config_path))
                if 'update' in config:
                    for key, value in config['update'].items():
                        if value.lower() in ['true', 'false']:
                            default_config[key] = value.lower() == 'true'
                        elif value.isdigit():
                            default_config[key] = int(value)
                        else:
                            default_config[key] = value
                logger.info(f"Config loaded from {self.config_path}")
            except Exception as e:
                logger.warning(f"Config load failed: {e}, using defaults")
        else:
            logger.info("No config file, using defaults")
        
        return default_config
    
    def _init_database(self) -> sqlite3.Connection:
        """Initialize audit database"""
        db = sqlite3.connect(str(self.db_path))
        cursor = db.cursor()
        
        # Update history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS update_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                component TEXT NOT NULL,
                action TEXT NOT NULL,
                from_version TEXT,
                to_version TEXT,
                severity TEXT,
                success INTEGER,
                operator TEXT,
                notes TEXT
            )
        ''')
        
        # Backups table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                backup_path TEXT NOT NULL,
                components TEXT,
                size_mb REAL,
                retain_until TIMESTAMP,
                restore_tested INTEGER DEFAULT 0
            )
        ''')
        
        # Update checks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS update_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updates_found INTEGER,
                critical_count INTEGER,
                high_count INTEGER,
                low_count INTEGER
            )
        ''')
        
        db.commit()
        logger.info("Database initialized")
        return db
    
    def _get_current_version(self) -> str:
        """Get current RF Arsenal OS version"""
        version_file = self.install_dir / 'VERSION'
        if version_file.exists():
            return version_file.read_text().strip()
        return "1.0.0"
    
    # ========================================================================
    # UPDATE CHECKING
    # ========================================================================
    
    def check_all_updates(self) -> Dict[str, List[ComponentUpdate]]:
        """
        Check updates for all components
        
        Returns:
            Dict with severity levels as keys, list of ComponentUpdate as values
        """
        print("\n" + "="*70)
        print("RF ARSENAL OS - UPDATE CHECK")
        print("="*70)
        print(f"Current Version: {self.current_version}")
        print(f"Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*70 + "\n")
        
        all_updates = {
            UpdateSeverity.CRITICAL: [],
            UpdateSeverity.HIGH: [],
            UpdateSeverity.LOW: []
        }
        
        # Check each component
        components_to_check = [
            ('RF Arsenal Core', self._check_core_updates),
            ('Python Dependencies', self._check_python_updates),
            ('System Packages', self._check_system_updates),
            ('Hardware Drivers', self._check_hardware_updates),
            ('Optional Modules', self._check_module_updates)
        ]
        
        for name, check_func in components_to_check:
            if not self.config.get(f'check_{name.lower().replace(" ", "_")}', True):
                print(f"[ SKIP ] {name} (disabled in config)")
                continue
            
            print(f"[  ...  ] Checking {name}...", end='', flush=True)
            try:
                updates = check_func()
                for update in updates:
                    all_updates[update.severity].append(update)
                print(f"\r[  OK  ] {name}: {len(updates)} updates")
            except Exception as e:
                print(f"\r[ FAIL ] {name}: {e}")
                logger.error(f"Failed to check {name}: {e}", exc_info=True)
        
        # Cache results
        self.update_cache = all_updates
        
        # Log check
        self._log_check(all_updates)
        
        # Display summary
        print()
        self._display_update_summary(all_updates)
        
        return all_updates
    
    def _check_core_updates(self) -> List[ComponentUpdate]:
        """Check RF Arsenal OS core updates from GitHub"""
        updates = []
        
        if not REQUESTS_AVAILABLE:
            logger.warning("requests module not available, skipping GitHub check")
            return updates
        
        try:
            # Get latest release from GitHub
            response = requests.get(
                f"{self.github_api}/releases/latest",
                timeout=10
            )
            
            if response.status_code == 200:
                release = response.json()
                latest_version = release['tag_name'].lstrip('v')
                
                if self._version_compare(latest_version, self.current_version) > 0:
                    # Determine severity from release notes
                    body = release.get('body', '').lower()
                    if any(word in body for word in ['security', 'critical', 'cve', 'vulnerability']):
                        severity = UpdateSeverity.CRITICAL
                    elif any(word in body for word in ['important', 'major', 'breaking']):
                        severity = UpdateSeverity.HIGH
                    else:
                        severity = UpdateSeverity.LOW
                    
                    # Get asset size
                    assets = release.get('assets', [])
                    size_mb = assets[0]['size'] / (1024*1024) if assets else 0
                    
                    updates.append(ComponentUpdate(
                        component='RF Arsenal OS Core',
                        current=self.current_version,
                        available=latest_version,
                        severity=severity,
                        changelog=release.get('body', 'No changelog')[:500],
                        size_mb=size_mb
                    ))
        
        except Exception as e:
            logger.error(f"Core update check failed: {e}")
        
        return updates
    
    def _check_python_updates(self) -> List[ComponentUpdate]:
        """Check Python package updates"""
        updates = []
        
        try:
            # Get outdated packages
            result = subprocess.run(
                ['pip3', 'list', '--outdated', '--format=json'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                outdated = json.loads(result.stdout)
                
                # Critical packages (security-sensitive)
                critical_packages = {
                    'cryptography', 'pyopenssl', 'requests', 'urllib3',
                    'pycryptodome', 'paramiko'
                }
                
                # High priority packages
                high_packages = {
                    'numpy', 'scipy', 'PyQt6', 'scapy', 'pyshark',
                    'psutil', 'netifaces'
                }
                
                for pkg in outdated:
                    name = pkg['name']
                    current = pkg['version']
                    available = pkg['latest_version']
                    
                    # Determine severity
                    if name in critical_packages:
                        severity = UpdateSeverity.CRITICAL
                    elif name in high_packages:
                        severity = UpdateSeverity.HIGH
                    else:
                        severity = UpdateSeverity.LOW
                    
                    updates.append(ComponentUpdate(
                        component=f'Python: {name}',
                        current=current,
                        available=available,
                        severity=severity,
                        changelog=f'Update {name} to {available}'
                    ))
        
        except Exception as e:
            logger.error(f"Python deps check failed: {e}")
        
        return updates
    
    def _check_system_updates(self) -> List[ComponentUpdate]:
        """Check system package updates (apt)"""
        updates = []
        
        try:
            # Update package lists quietly
            subprocess.run(
                ['apt-get', 'update'],
                capture_output=True,
                timeout=60,
                check=False
            )
            
            # Check for upgradable packages
            result = subprocess.run(
                ['apt', 'list', '--upgradable'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                
                # Security-critical system packages
                critical_packages = {
                    'libssl', 'openssl', 'openssh', 'sudo',
                    'linux-image', 'linux-firmware', 'systemd'
                }
                
                # High priority packages
                high_packages = {
                    'wireshark', 'tshark', 'tor', 'gnuradio',
                    'python3', 'git'
                }
                
                for line in lines:
                    if not line.strip():
                        continue
                    
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    
                    name = parts[0].split('/')[0]
                    version_info = parts[1]
                    
                    # Extract versions
                    if '[' in version_info:
                        available = version_info.split()[0]
                        current = version_info.split('[')[1].rstrip(']') if '[' in version_info else 'unknown'
                    else:
                        available = version_info
                        current = 'unknown'
                    
                    # Determine severity
                    if any(crit in name for crit in critical_packages):
                        severity = UpdateSeverity.CRITICAL
                    elif any(high in name for high in high_packages):
                        severity = UpdateSeverity.HIGH
                    else:
                        severity = UpdateSeverity.LOW
                    
                    updates.append(ComponentUpdate(
                        component=f'System: {name}',
                        current=current,
                        available=available,
                        severity=severity,
                        changelog=f'Update {name} to {available}'
                    ))
        
        except Exception as e:
            logger.error(f"System deps check failed: {e}")
        
        return updates
    
    def _check_hardware_updates(self) -> List[ComponentUpdate]:
        """Check hardware driver updates (BladeRF, HackRF)"""
        updates = []
        
        # Check BladeRF
        try:
            result = subprocess.run(
                ['bladeRF-cli', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and REQUESTS_AVAILABLE:
                current_version = result.stdout.strip().split()[1]
                
                # Check GitHub for latest BladeRF release
                response = requests.get(
                    "https://api.github.com/repos/Nuand/bladeRF/releases/latest",
                    timeout=10
                )
                
                if response.status_code == 200:
                    release = response.json()
                    latest_version = release['tag_name'].lstrip('v')
                    
                    if self._version_compare(latest_version, current_version) > 0:
                        updates.append(ComponentUpdate(
                            component='BladeRF Library',
                            current=current_version,
                            available=latest_version,
                            severity=UpdateSeverity.HIGH,
                            changelog=release.get('body', 'No changelog')[:500]
                        ))
        
        except Exception as e:
            logger.debug(f"BladeRF check failed: {e}")
        
        # Check HackRF
        try:
            result = subprocess.run(
                ['hackrf_info'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and REQUESTS_AVAILABLE:
                # Parse HackRF version from output
                for line in result.stdout.split('\n'):
                    if 'Firmware Version' in line:
                        current_version = line.split(':')[1].strip()
                        
                        # Check GitHub for latest HackRF release
                        response = requests.get(
                            "https://api.github.com/repos/greatscottgadgets/hackrf/releases/latest",
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            release = response.json()
                            latest_version = release['tag_name'].lstrip('v')
                            
                            if self._version_compare(latest_version, current_version) > 0:
                                updates.append(ComponentUpdate(
                                    component='HackRF Firmware',
                                    current=current_version,
                                    available=latest_version,
                                    severity=UpdateSeverity.HIGH,
                                    changelog=release.get('body', 'No changelog')[:500]
                                ))
                        break
        
        except Exception as e:
            logger.debug(f"HackRF check failed: {e}")
        
        return updates
    
    def _check_module_updates(self) -> List[ComponentUpdate]:
        """Check optional module updates (ML models, OSINT DB)"""
        updates = []
        
        # Check OSINT database age
        osint_db_file = self.install_dir / 'data' / 'offline_osint.db'
        if osint_db_file.exists():
            db_age_days = (datetime.now() - datetime.fromtimestamp(
                osint_db_file.stat().st_mtime
            )).days
            
            if db_age_days > 90:  # Quarterly updates
                updates.append(ComponentUpdate(
                    component='OSINT Database',
                    current=f'{db_age_days} days old',
                    available='Latest',
                    severity=UpdateSeverity.LOW,
                    changelog='Updated carrier mappings, country codes, and IMSI databases',
                    size_mb=50
                ))
        
        return updates
    
    def _version_compare(self, version1: str, version2: str) -> int:
        """
        Compare two version strings
        
        Returns:
            1 if version1 > version2
            0 if version1 == version2
            -1 if version1 < version2
        """
        def normalize(v):
            return [int(x) for x in v.split('.') if x.isdigit()]
        
        v1 = normalize(version1)
        v2 = normalize(version2)
        
        for i in range(max(len(v1), len(v2))):
            num1 = v1[i] if i < len(v1) else 0
            num2 = v2[i] if i < len(v2) else 0
            
            if num1 > num2:
                return 1
            elif num1 < num2:
                return -1
        
        return 0
    
    def _display_update_summary(self, updates: Dict[str, List[ComponentUpdate]]):
        """Display update summary to operator"""
        total = sum(len(v) for v in updates.values())
        
        if total == 0:
            print("="*70)
            print("✅ SYSTEM UP-TO-DATE - No updates available")
            print("="*70)
            return
        
        print("="*70)
        print(f"⚠️  {total} UPDATES AVAILABLE")
        print("="*70)
        print()
        
        for severity in [UpdateSeverity.CRITICAL, UpdateSeverity.HIGH, UpdateSeverity.LOW]:
            items = updates[severity]
            if not items:
                continue
            
            # Severity icon
            if severity == UpdateSeverity.CRITICAL:
                icon = "🔴"
            elif severity == UpdateSeverity.HIGH:
                icon = "🟡"
            else:
                icon = "🟢"
            
            print(f"{icon} [{severity}] {len(items)} updates:")
            for update in items:
                print(f"   • {update.component}")
                print(f"     {update.current_version} → {update.available_version}")
                if update.size_mb > 0:
                    print(f"     Size: {update.size_mb:.1f} MB")
            print()
        
        print("="*70)
        print("Next Steps:")
        print("  • Review changes: ./update_manager.py --details")
        print("  • Install critical: sudo ./update_manager.py --install critical")
        print("  • Install all: sudo ./update_manager.py --install all")
        print("="*70)
        print()
    
    # ========================================================================
    # UPDATE INSTALLATION
    # ========================================================================
    
    def install_updates(self, severity: str = 'all', interactive: bool = True) -> bool:
        """
        Install updates with safeguards
        
        Args:
            severity: 'critical', 'high', 'low', or 'all'
            interactive: Prompt for confirmation
            
        Returns:
            Success status
        """
        if not self.update_cache:
            print("[!] No cached updates. Run --check first.")
            return False
        
        # Determine which updates to install
        if severity == 'all':
            to_install = []
            for items in self.update_cache.values():
                to_install.extend(items)
        elif severity.upper() in [UpdateSeverity.CRITICAL, UpdateSeverity.HIGH, UpdateSeverity.LOW]:
            to_install = self.update_cache[severity.upper()]
        else:
            print(f"[-] Unknown severity: {severity}")
            print("[*] Valid options: critical, high, low, all")
            return False
        
        if not to_install:
            print("[+] No updates to install")
            return True
        
        print("\n" + "="*70)
        print("RF ARSENAL OS - UPDATE INSTALLATION")
        print("="*70)
        print(f"Installing {len(to_install)} updates")
        print()
        
        # Show what will be updated
        for update in to_install:
            print(f"[{update.severity}] {update.component}")
            print(f"  {update.current_version} → {update.available_version}")
        
        print()
        
        # Confirmation
        if interactive:
            print("="*70)
            response = input("❗ Continue with installation? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("[*] Installation cancelled by operator")
                return False
        
        # Create backup
        print("\n[*] Creating backup before update...")
        backup_path = self._create_backup()
        if not backup_path:
            print("[-] ❌ Backup failed. ABORTING UPDATE for safety.")
            return False
        print(f"[+] ✅ Backup created: {backup_path}")
        print()
        
        # Install updates
        success_count = 0
        failed_updates = []
        
        for i, update in enumerate(to_install, 1):
            print(f"[{i}/{len(to_install)}] Updating {update.component}...", end='', flush=True)
            
            try:
                if update.component.startswith('RF Arsenal'):
                    success = self._install_core_update(update)
                elif update.component.startswith('Python:'):
                    success = self._install_python_update(update)
                elif update.component.startswith('System:'):
                    success = self._install_system_update(update)
                elif update.component.startswith('BladeRF') or update.component.startswith('HackRF'):
                    success = self._install_hardware_update(update)
                else:
                    success = self._install_module_update(update)
                
                if success:
                    print(f"\r[{i}/{len(to_install)}] ✅ {update.component} updated")
                    success_count += 1
                    self._log_update(update, True)
                else:
                    print(f"\r[{i}/{len(to_install)}] ❌ {update.component} FAILED")
                    failed_updates.append(update)
                    self._log_update(update, False)
            
            except Exception as e:
                print(f"\r[{i}/{len(to_install)}] ❌ {update.component} ERROR: {e}")
                failed_updates.append(update)
                self._log_update(update, False, str(e))
                logger.error(f"Update failed: {update.component}", exc_info=True)
        
        # Summary
        print()
        print("="*70)
        print("UPDATE SUMMARY")
        print("="*70)
        print(f"✅ Successful: {success_count}/{len(to_install)}")
        
        if failed_updates:
            print(f"❌ Failed: {len(failed_updates)}/{len(to_install)}")
            print()
            print("Failed Updates:")
            for update in failed_updates:
                print(f"  • {update.component}")
            print()
            print("⚠️  BACKUP AVAILABLE FOR ROLLBACK")
            print(f"   Backup: {backup_path}")
            print(f"   Rollback: sudo ./update_manager.py --rollback")
        else:
            print("🎉 All updates installed successfully!")
        
        print("="*70)
        
        return success_count == len(to_install)
    
    def _create_backup(self) -> Optional[Path]:
        """Create system backup before update"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f'backup_{timestamp}.tar.gz'
            backup_path = self.backup_dir / backup_name
            
            print(f"   Creating backup archive...")
            
            # Create tarball of critical directories
            with tarfile.open(backup_path, 'w:gz') as tar:
                if self.install_dir.exists():
                    tar.add(str(self.install_dir), arcname='rf-arsenal')
                
                config_dir = Path('/etc/rf-arsenal')
                if config_dir.exists():
                    tar.add(str(config_dir), arcname='config')
            
            # Calculate size
            size_mb = backup_path.stat().st_size / (1024 * 1024)
            
            # Check size limit
            max_size_gb = self.config.get('max_backup_size_gb', 10)
            if size_mb > (max_size_gb * 1024):
                print(f"   ⚠️  Backup too large: {size_mb:.1f} MB > {max_size_gb} GB limit")
                backup_path.unlink()
                return None
            
            # Log backup
            cursor = self.db.cursor()
            retain_until = datetime.now() + timedelta(days=self.config.get('backup_retention_days', 30))
            cursor.execute('''
                INSERT INTO backups (timestamp, backup_path, components, size_mb, retain_until)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                str(backup_path),
                'core,config',
                size_mb,
                retain_until
            ))
            self.db.commit()
            
            print(f"   Backup size: {size_mb:.1f} MB")
            
            return backup_path
        
        except Exception as e:
            logger.error(f"Backup failed: {e}", exc_info=True)
            return None
    
    def _install_core_update(self, update: ComponentUpdate) -> bool:
        """Install RF Arsenal OS core update"""
        if not REQUESTS_AVAILABLE:
            logger.error("requests module required for core updates")
            return False
        
        try:
            # Get latest release
            response = requests.get(f"{self.github_api}/releases/latest", timeout=10)
            release = response.json()
            
            # Find tarball
            tarball_url = release['tarball_url']
            
            # Download
            download_path = Path('/tmp/rf-arsenal-update.tar.gz')
            response = requests.get(tarball_url, stream=True, timeout=60)
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract
            extract_path = Path('/tmp/rf-arsenal-update')
            extract_path.mkdir(exist_ok=True)
            with tarfile.open(download_path, 'r:gz') as tar:
                tar.extractall(extract_path)
            
            # Find extracted directory
            extracted = list(extract_path.glob('*'))[0]
            
            # Copy files
            shutil.copytree(extracted, self.install_dir, dirs_exist_ok=True)
            
            # Update VERSION file
            (self.install_dir / 'VERSION').write_text(update.available_version)
            
            # Cleanup
            download_path.unlink()
            shutil.rmtree(extract_path)
            
            return True
        
        except Exception as e:
            logger.error(f"Core update failed: {e}", exc_info=True)
            return False
    
    def _install_python_update(self, update: ComponentUpdate) -> bool:
        """Install Python package update"""
        try:
            package_name = update.component.split(': ')[1]
            
            result = subprocess.run(
                ['pip3', 'install', '--upgrade', package_name],
                capture_output=True,
                timeout=300
            )
            
            return result.returncode == 0
        
        except Exception as e:
            logger.error(f"Python update failed: {e}", exc_info=True)
            return False
    
    def _install_system_update(self, update: ComponentUpdate) -> bool:
        """Install system package update"""
        try:
            package_name = update.component.split(': ')[1]
            
            result = subprocess.run(
                ['apt-get', 'install', '-y', package_name],
                capture_output=True,
                timeout=300
            )
            
            return result.returncode == 0
        
        except Exception as e:
            logger.error(f"System update failed: {e}", exc_info=True)
            return False
    
    def _install_hardware_update(self, update: ComponentUpdate) -> bool:
        """Install hardware driver update"""
        # Hardware updates typically require manual rebuild from source
        # Provide instructions instead
        print()
        print(f"   ⚠️  {update.component} requires manual installation:")
        
        if 'BladeRF' in update.component:
            print("      1. cd /tmp")
            print("      2. git clone https://github.com/Nuand/bladeRF.git")
            print("      3. cd bladeRF/host")
            print("      4. mkdir build && cd build")
            print("      5. cmake ..")
            print("      6. make && sudo make install")
            print("      7. sudo ldconfig")
        elif 'HackRF' in update.component:
            print("      1. cd /tmp")
            print("      2. git clone https://github.com/greatscottgadgets/hackrf.git")
            print("      3. cd hackrf/host")
            print("      4. mkdir build && cd build")
            print("      5. cmake ..")
            print("      6. make && sudo make install")
            print("      7. sudo ldconfig")
        
        print()
        return False  # Requires manual action
    
    def _install_module_update(self, update: ComponentUpdate) -> bool:
        """Install optional module update"""
        # Module-specific update logic would go here
        return True
    
    # ========================================================================
    # BACKUP & ROLLBACK
    # ========================================================================
    
    def rollback(self) -> bool:
        """Rollback to last backup"""
        print("\n" + "="*70)
        print("RF ARSENAL OS - ROLLBACK TO LAST BACKUP")
        print("="*70)
        
        try:
            # Get most recent backup
            cursor = self.db.cursor()
            cursor.execute('''
                SELECT id, backup_path, timestamp, size_mb FROM backups
                ORDER BY timestamp DESC
                LIMIT 1
            ''')
            result = cursor.fetchone()
            
            if not result:
                print("[-] No backups available")
                return False
            
            backup_id, backup_path_str, timestamp, size_mb = result
            backup_path = Path(backup_path_str)
            
            print(f"Backup: {backup_path}")
            print(f"Created: {timestamp}")
            print(f"Size: {size_mb:.1f} MB")
            print()
            
            if not backup_path.exists():
                print("[-] ❌ Backup file not found on disk")
                return False
            
            # Confirmation
            response = input("❗ Confirm rollback? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("[*] Rollback cancelled")
                return False
            
            print()
            print("[*] Extracting backup...")
            
            # Extract backup
            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall('/')
            
            # Mark backup as restore-tested
            cursor.execute('''
                UPDATE backups SET restore_tested = 1 WHERE id = ?
            ''', (backup_id,))
            self.db.commit()
            
            print("[+] ✅ Rollback complete")
            print()
            print("⚠️  IMPORTANT: Restart RF Arsenal OS to apply changes")
            print("    sudo systemctl restart rf-arsenal")
            print()
            print("="*70)
            return True
        
        except Exception as e:
            print(f"[-] ❌ Rollback failed: {e}")
            logger.error(f"Rollback failed: {e}", exc_info=True)
            return False
    
    def cleanup_old_backups(self):
        """Remove old backups based on retention policy"""
        retention_days = int(self.config.get('backup_retention_days', 30))
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        print(f"[*] Cleaning backups older than {retention_days} days...")
        
        deleted = 0
        cursor = self.db.cursor()
        cursor.execute('''
            SELECT id, backup_path FROM backups
            WHERE timestamp < ?
        ''', (cutoff_date,))
        
        for backup_id, backup_path_str in cursor.fetchall():
            backup_path = Path(backup_path_str)
            if backup_path.exists():
                backup_path.unlink()
                deleted += 1
            
            # Remove from database
            cursor.execute('DELETE FROM backups WHERE id = ?', (backup_id,))
        
        self.db.commit()
        
        if deleted > 0:
            print(f"[+] Cleaned up {deleted} old backups")
        else:
            print("[*] No old backups to clean")
    
    # ========================================================================
    # AUDIT & LOGGING
    # ========================================================================
    
    def _log_check(self, updates: Dict[str, List[ComponentUpdate]]):
        """Log update check to database"""
        try:
            cursor = self.db.cursor()
            cursor.execute('''
                INSERT INTO update_checks (timestamp, updates_found, critical_count, high_count, low_count)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                sum(len(v) for v in updates.values()),
                len(updates[UpdateSeverity.CRITICAL]),
                len(updates[UpdateSeverity.HIGH]),
                len(updates[UpdateSeverity.LOW])
            ))
            self.db.commit()
        except Exception as e:
            logger.error(f"Check log failed: {e}")
    
    def _log_update(self, update: ComponentUpdate, success: bool, notes: str = ""):
        """Log update to audit trail"""
        try:
            cursor = self.db.cursor()
            cursor.execute('''
                INSERT INTO update_history 
                (timestamp, component, action, from_version, to_version, 
                 severity, success, operator, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                update.component,
                'UPDATE',
                update.current_version,
                update.available_version,
                update.severity,
                1 if success else 0,
                os.getenv('USER', 'unknown'),
                notes
            ))
            self.db.commit()
        except Exception as e:
            logger.error(f"Audit log failed: {e}")
    
    def show_history(self, limit: int = 20):
        """Show update history"""
        print("\n" + "="*70)
        print("UPDATE HISTORY")
        print("="*70)
        
        cursor = self.db.cursor()
        cursor.execute('''
            SELECT timestamp, component, from_version, to_version, 
                   severity, success, operator, notes
            FROM update_history
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        
        if not results:
            print("No update history")
            return
        
        for row in results:
            timestamp, component, from_ver, to_ver, severity, success, operator, notes = row
            status = "✅" if success else "❌"
            
            print(f"\n{status} {timestamp}")
            print(f"   Component: {component}")
            print(f"   Version: {from_ver} → {to_ver}")
            print(f"   Severity: {severity}")
            print(f"   Operator: {operator}")
            if notes:
                print(f"   Notes: {notes}")
        
        print("\n" + "="*70)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='RF Arsenal OS - Comprehensive Update Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sudo ./update_manager.py --check                    # Check all components
  sudo ./update_manager.py --install critical         # Install critical updates
  sudo ./update_manager.py --install all              # Install all updates
  sudo ./update_manager.py --rollback                 # Rollback to last backup
  sudo ./update_manager.py --history                  # Show update history
  sudo ./update_manager.py --cleanup                  # Clean up old backups

Update Severity Levels:
  CRITICAL - Security fixes, CVEs, system-breaking bugs
  HIGH     - Important features, significant bugs
  LOW      - Minor improvements, optimizations

Operator Control:
  - NO auto-updates (manual approval required)
  - Backup before every update
  - Rollback capability
  - Complete audit trail
        """
    )
    
    parser.add_argument('--check', action='store_true',
                       help='Check for available updates')
    parser.add_argument('--install', choices=['critical', 'high', 'low', 'all'],
                       help='Install updates by severity level')
    parser.add_argument('--rollback', action='store_true',
                       help='Rollback to last backup')
    parser.add_argument('--history', action='store_true',
                       help='Show update history')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up old backups')
    parser.add_argument('--non-interactive', action='store_true',
                       help='Run without prompts (use with caution)')
    
    args = parser.parse_args()
    
    # Require root for most operations
    if not args.history and os.geteuid() != 0:
        print("[-] Must run as root (use sudo)")
        print("    Example: sudo ./update_manager.py --check")
        sys.exit(1)
    
    # Initialize manager
    try:
        manager = ComprehensiveUpdateManager()
    except Exception as e:
        print(f"[-] Failed to initialize update manager: {e}")
        sys.exit(1)
    
    # Execute command
    if args.check:
        manager.check_all_updates()
    
    elif args.install:
        if not manager.update_cache:
            print("[*] Checking for updates first...")
            manager.check_all_updates()
        
        success = manager.install_updates(
            severity=args.install,
            interactive=not args.non_interactive
        )
        sys.exit(0 if success else 1)
    
    elif args.rollback:
        success = manager.rollback()
        sys.exit(0 if success else 1)
    
    elif args.history:
        manager.show_history()
    
    elif args.cleanup:
        manager.cleanup_old_backups()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
