#!/usr/bin/env python3
"""
RF Arsenal OS - OPSEC Monitor & Scoring System
Real-time operational security assessment and leak detection

DESIGN PHILOSOPHY:
- Continuous monitoring of user's security posture
- Live scoring (0-100) with clear feedback
- Automatic detection of OPSEC violations
- Warnings before user compromises themselves
- Beginner-friendly explanations of risks
"""

import os
import subprocess
import socket
import threading
import time
import logging
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """OPSEC threat levels"""
    SECURE = "secure"           # 90-100: Excellent OPSEC
    GOOD = "good"               # 70-89: Good, minor issues
    WARNING = "warning"         # 50-69: Concerns present
    DANGER = "danger"           # 25-49: Significant risks
    CRITICAL = "critical"       # 0-24: Severely compromised


class OPSECCategory(Enum):
    """Categories of OPSEC checks"""
    NETWORK = "network"         # Network exposure
    IDENTITY = "identity"       # MAC, hostname, user agent
    FORENSICS = "forensics"     # Disk writes, logs, traces
    HARDWARE = "hardware"       # RF emissions, USB devices
    BEHAVIOR = "behavior"       # Timing, patterns
    LOCATION = "location"       # GPS, IP geolocation


@dataclass
class OPSECIssue:
    """Represents a detected OPSEC issue"""
    id: str
    category: OPSECCategory
    severity: int  # 1-10, 10 being most severe
    title: str
    description: str
    recommendation: str
    detected_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    auto_fixable: bool = False
    fix_command: Optional[str] = None


@dataclass
class OPSECScore:
    """Current OPSEC score and breakdown"""
    total_score: int  # 0-100
    threat_level: ThreatLevel
    category_scores: Dict[str, int]
    active_issues: List[OPSECIssue]
    recommendations: List[str]
    last_updated: datetime = field(default_factory=datetime.now)


class OPSECMonitor:
    """
    Real-time OPSEC monitoring and scoring system
    
    Features:
    - Continuous background monitoring
    - Live score calculation (0-100)
    - Automatic issue detection
    - Clear recommendations for fixes
    - Auto-fix capability for some issues
    - Beginner-friendly explanations
    """
    
    # Score weights for each category (must sum to 100)
    CATEGORY_WEIGHTS = {
        OPSECCategory.NETWORK: 30,      # Network is most critical
        OPSECCategory.IDENTITY: 20,
        OPSECCategory.FORENSICS: 20,
        OPSECCategory.HARDWARE: 10,
        OPSECCategory.BEHAVIOR: 10,
        OPSECCategory.LOCATION: 10,
    }
    
    def __init__(self):
        self.issues: Dict[str, OPSECIssue] = {}
        self.current_score: Optional[OPSECScore] = None
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
        self._check_interval = 10  # seconds
        
        # Baseline values (set during initialization)
        self._original_mac: Dict[str, str] = {}
        self._original_hostname: Optional[str] = None
        
        # Track state
        self._network_mode = "offline"
        self._stealth_mode = False
        self._ram_only_mode = False
        
        # Initialize baseline
        self._capture_baseline()
        
        logger.info("OPSECMonitor initialized")
    
    def _capture_baseline(self):
        """Capture baseline system state"""
        try:
            # Capture original hostname
            self._original_hostname = socket.gethostname()
            
            # Capture original MAC addresses
            interfaces = ['wlan0', 'wlan1', 'eth0', 'eth1']
            for iface in interfaces:
                try:
                    with open(f'/sys/class/net/{iface}/address', 'r') as f:
                        self._original_mac[iface] = f.read().strip()
                except:
                    pass
            
            logger.debug(f"Baseline captured: hostname={self._original_hostname}, MACs={len(self._original_mac)}")
            
        except Exception as e:
            logger.warning(f"Failed to capture baseline: {e}")
    
    def start_monitoring(self, interval_seconds: int = 10):
        """Start continuous OPSEC monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._check_interval = interval_seconds
        
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info(f"OPSEC monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("OPSEC monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                self.run_all_checks()
                self._calculate_score()
                self._notify_callbacks()
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
            
            time.sleep(self._check_interval)
    
    def run_all_checks(self) -> List[OPSECIssue]:
        """Run all OPSEC checks and return issues found"""
        new_issues = []
        
        # Network checks
        new_issues.extend(self._check_network())
        
        # Identity checks
        new_issues.extend(self._check_identity())
        
        # Forensics checks
        new_issues.extend(self._check_forensics())
        
        # Hardware checks
        new_issues.extend(self._check_hardware())
        
        # Behavior checks
        new_issues.extend(self._check_behavior())
        
        # Location checks
        new_issues.extend(self._check_location())
        
        return new_issues
    
    # ========== NETWORK CHECKS ==========
    
    def _check_network(self) -> List[OPSECIssue]:
        """Check for network-related OPSEC issues"""
        issues = []
        
        # Check if network interfaces are up when should be offline
        if self._network_mode == "offline":
            try:
                result = subprocess.run(['ip', 'link', 'show'], capture_output=True, text=True, timeout=5)
                
                for iface in ['wlan0', 'eth0']:
                    if f'{iface}:' in result.stdout and 'state UP' in result.stdout.split(f'{iface}:')[1].split('\n')[0]:
                        issue = OPSECIssue(
                            id=f'net_iface_up_{iface}',
                            category=OPSECCategory.NETWORK,
                            severity=8,
                            title=f'Network Interface Active: {iface}',
                            description=f'Network interface {iface} is UP while system should be offline. This could leak data.',
                            recommendation=f'Disable interface: "sudo ip link set {iface} down"',
                            auto_fixable=True,
                            fix_command=f'ip link set {iface} down'
                        )
                        self._add_issue(issue)
                        issues.append(issue)
            except:
                pass
        
        # Check for active network connections
        try:
            result = subprocess.run(['ss', '-tuln'], capture_output=True, text=True, timeout=5)
            
            # Check for listening services
            if 'LISTEN' in result.stdout:
                listening_ports = []
                for line in result.stdout.split('\n'):
                    if 'LISTEN' in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            listening_ports.append(parts[4])
                
                if listening_ports and self._network_mode == "offline":
                    issue = OPSECIssue(
                        id='net_listening_services',
                        category=OPSECCategory.NETWORK,
                        severity=6,
                        title='Listening Network Services Detected',
                        description=f'Services listening on: {", ".join(listening_ports[:5])}. Could expose system.',
                        recommendation='Stop unnecessary services or ensure firewall blocks inbound.',
                        auto_fixable=False
                    )
                    self._add_issue(issue)
                    issues.append(issue)
        except:
            pass
        
        # Check for DNS leaks (if online via Tor/VPN)
        if self._network_mode in ["online_tor", "online_vpn", "online_full"]:
            try:
                # Check if DNS is going through anonymity layer
                result = subprocess.run(['cat', '/etc/resolv.conf'], capture_output=True, text=True, timeout=5)
                
                # If using public DNS directly, that's a leak
                public_dns = ['8.8.8.8', '8.8.4.4', '1.1.1.1', '9.9.9.9']
                for dns in public_dns:
                    if dns in result.stdout:
                        issue = OPSECIssue(
                            id='net_dns_leak',
                            category=OPSECCategory.NETWORK,
                            severity=9,
                            title='DNS Leak Detected',
                            description=f'DNS queries going to {dns} instead of anonymity network. Your DNS queries reveal your activity.',
                            recommendation='Configure DNS to use Tor DNS (port 5353) or VPN DNS.',
                            auto_fixable=False
                        )
                        self._add_issue(issue)
                        issues.append(issue)
                        break
            except:
                pass
        
        # Check for WebRTC leak indicators (browser processes)
        try:
            result = subprocess.run(['pgrep', '-l', 'chrome|firefox|chromium'], 
                                  capture_output=True, text=True, timeout=5)
            if result.stdout.strip() and self._network_mode in ["online_tor", "online_vpn"]:
                issue = OPSECIssue(
                    id='net_browser_webrtc',
                    category=OPSECCategory.NETWORK,
                    severity=7,
                    title='Browser Running - WebRTC Leak Risk',
                    description='A browser is running which may leak real IP via WebRTC.',
                    recommendation='Disable WebRTC in browser or use Tor Browser.',
                    auto_fixable=False
                )
                self._add_issue(issue)
                issues.append(issue)
        except:
            pass
        
        return issues
    
    # ========== IDENTITY CHECKS ==========
    
    def _check_identity(self) -> List[OPSECIssue]:
        """Check for identity-related OPSEC issues"""
        issues = []
        
        # Check if MAC address has been randomized
        for iface, original_mac in self._original_mac.items():
            try:
                with open(f'/sys/class/net/{iface}/address', 'r') as f:
                    current_mac = f.read().strip()
                
                if current_mac == original_mac:
                    issue = OPSECIssue(
                        id=f'id_mac_not_random_{iface}',
                        category=OPSECCategory.IDENTITY,
                        severity=6,
                        title=f'Original MAC Address: {iface}',
                        description=f'Interface {iface} is using original MAC ({original_mac}). This is a unique hardware identifier.',
                        recommendation='Randomize MAC: "randomize mac address"',
                        auto_fixable=True,
                        fix_command=f'randomize mac {iface}'
                    )
                    self._add_issue(issue)
                    issues.append(issue)
                else:
                    # MAC was changed, remove any existing issue
                    self._remove_issue(f'id_mac_not_random_{iface}')
            except:
                pass
        
        # Check hostname
        try:
            current_hostname = socket.gethostname()
            
            # Check for revealing hostnames
            revealing_patterns = ['laptop', 'desktop', 'macbook', 'pc', 'workstation']
            for pattern in revealing_patterns:
                if pattern.lower() in current_hostname.lower():
                    issue = OPSECIssue(
                        id='id_revealing_hostname',
                        category=OPSECCategory.IDENTITY,
                        severity=4,
                        title='Revealing Hostname',
                        description=f'Hostname "{current_hostname}" may reveal device type or owner.',
                        recommendation='Change to generic hostname: "sudo hostnamectl set-hostname localhost"',
                        auto_fixable=True,
                        fix_command='hostnamectl set-hostname localhost'
                    )
                    self._add_issue(issue)
                    issues.append(issue)
                    break
            
            # Check for username in hostname
            username = os.environ.get('USER', '')
            if username and username.lower() in current_hostname.lower():
                issue = OPSECIssue(
                    id='id_username_in_hostname',
                    category=OPSECCategory.IDENTITY,
                    severity=5,
                    title='Username in Hostname',
                    description=f'Hostname contains username. This links network traffic to your identity.',
                    recommendation='Change hostname to something generic.',
                    auto_fixable=True,
                    fix_command='hostnamectl set-hostname localhost'
                )
                self._add_issue(issue)
                issues.append(issue)
        except:
            pass
        
        return issues
    
    # ========== FORENSICS CHECKS ==========
    
    def _check_forensics(self) -> List[OPSECIssue]:
        """Check for forensic trace issues"""
        issues = []
        
        # Check if swap is enabled (can contain sensitive data)
        try:
            result = subprocess.run(['swapon', '--show'], capture_output=True, text=True, timeout=5)
            if result.stdout.strip():
                issue = OPSECIssue(
                    id='forensics_swap_enabled',
                    category=OPSECCategory.FORENSICS,
                    severity=7,
                    title='Swap Enabled',
                    description='Swap is enabled. Sensitive data in RAM may be written to disk and recoverable.',
                    recommendation='Disable swap: "sudo swapoff -a"',
                    auto_fixable=True,
                    fix_command='swapoff -a'
                )
                self._add_issue(issue)
                issues.append(issue)
            else:
                self._remove_issue('forensics_swap_enabled')
        except:
            pass
        
        # Check for bash history
        history_files = [
            Path.home() / '.bash_history',
            Path.home() / '.zsh_history',
            Path('/root/.bash_history')
        ]
        
        for hist_file in history_files:
            try:
                if hist_file.exists() and hist_file.stat().st_size > 0:
                    issue = OPSECIssue(
                        id=f'forensics_history_{hist_file.name}',
                        category=OPSECCategory.FORENSICS,
                        severity=5,
                        title=f'Command History Exists: {hist_file.name}',
                        description=f'Shell history at {hist_file} contains record of commands. Forensic evidence.',
                        recommendation=f'Clear and disable: "cat /dev/null > {hist_file} && export HISTSIZE=0"',
                        auto_fixable=True,
                        fix_command=f'cat /dev/null > {hist_file}'
                    )
                    self._add_issue(issue)
                    issues.append(issue)
            except:
                pass
        
        # Check for recent files in common locations
        recent_dirs = [
            Path.home() / '.local/share/recently-used.xbel',
            Path.home() / '.recently-used'
        ]
        
        for recent in recent_dirs:
            try:
                if recent.exists():
                    issue = OPSECIssue(
                        id=f'forensics_recent_files',
                        category=OPSECCategory.FORENSICS,
                        severity=3,
                        title='Recent Files History',
                        description='System maintains record of recently accessed files.',
                        recommendation='Clear recent files history and disable tracking.',
                        auto_fixable=True,
                        fix_command=f'rm -f {recent}'
                    )
                    self._add_issue(issue)
                    issues.append(issue)
                    break
            except:
                pass
        
        # Check if RAM-only mode is enabled
        if not self._ram_only_mode:
            # Check if tmpfs is mounted for sensitive operations
            try:
                result = subprocess.run(['mount'], capture_output=True, text=True, timeout=5)
                if 'tmpfs' not in result.stdout or '/tmp/rfarsenal' not in result.stdout:
                    issue = OPSECIssue(
                        id='forensics_ram_only_disabled',
                        category=OPSECCategory.FORENSICS,
                        severity=6,
                        title='RAM-Only Mode Not Active',
                        description='Operations may be writing to disk instead of RAM. Creates forensic evidence.',
                        recommendation='Enable RAM-only mode: "enable ram-only mode"',
                        auto_fixable=True,
                        fix_command='enable_ram_only'
                    )
                    self._add_issue(issue)
                    issues.append(issue)
            except:
                pass
        else:
            self._remove_issue('forensics_ram_only_disabled')
        
        return issues
    
    # ========== HARDWARE CHECKS ==========
    
    def _check_hardware(self) -> List[OPSECIssue]:
        """Check for hardware-related OPSEC issues"""
        issues = []
        
        # Check for unexpected USB devices
        try:
            result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
            
            # Known risky USB devices
            risky_devices = ['webcam', 'camera', 'microphone', 'hub']
            
            for line in result.stdout.lower().split('\n'):
                for risky in risky_devices:
                    if risky in line and 'bladerf' not in line:
                        issue = OPSECIssue(
                            id=f'hw_usb_{risky}',
                            category=OPSECCategory.HARDWARE,
                            severity=4,
                            title=f'Potentially Risky USB Device: {risky}',
                            description=f'USB device detected that could be used for surveillance: {risky}',
                            recommendation='Disconnect unnecessary USB devices.',
                            auto_fixable=False
                        )
                        self._add_issue(issue)
                        issues.append(issue)
        except:
            pass
        
        # Check Bluetooth status
        try:
            result = subprocess.run(['hciconfig'], capture_output=True, text=True, timeout=5)
            if 'UP RUNNING' in result.stdout:
                issue = OPSECIssue(
                    id='hw_bluetooth_enabled',
                    category=OPSECCategory.HARDWARE,
                    severity=5,
                    title='Bluetooth Enabled',
                    description='Bluetooth is enabled. Device is discoverable and trackable via Bluetooth.',
                    recommendation='Disable Bluetooth: "sudo hciconfig hci0 down"',
                    auto_fixable=True,
                    fix_command='hciconfig hci0 down'
                )
                self._add_issue(issue)
                issues.append(issue)
            else:
                self._remove_issue('hw_bluetooth_enabled')
        except:
            pass
        
        return issues
    
    # ========== BEHAVIOR CHECKS ==========
    
    def _check_behavior(self) -> List[OPSECIssue]:
        """Check for behavioral OPSEC issues"""
        issues = []
        
        # Check for predictable timing patterns (basic check)
        # In a real implementation, this would track operation timing
        
        # Check if system time is synchronized (could correlate activities)
        try:
            result = subprocess.run(['timedatectl', 'show', '-p', 'NTPSynchronized'], 
                                  capture_output=True, text=True, timeout=5)
            if 'yes' in result.stdout.lower():
                issue = OPSECIssue(
                    id='behavior_ntp_sync',
                    category=OPSECCategory.BEHAVIOR,
                    severity=2,
                    title='NTP Time Sync Enabled',
                    description='System clock synced via NTP. Creates network traffic and could correlate activities.',
                    recommendation='Consider disabling NTP for sensitive operations.',
                    auto_fixable=False
                )
                self._add_issue(issue)
                issues.append(issue)
        except:
            pass
        
        return issues
    
    # ========== LOCATION CHECKS ==========
    
    def _check_location(self) -> List[OPSECIssue]:
        """Check for location-related OPSEC issues"""
        issues = []
        
        # Check if WiFi is enabled (can be used for location tracking)
        try:
            result = subprocess.run(['nmcli', 'radio', 'wifi'], capture_output=True, text=True, timeout=5)
            if 'enabled' in result.stdout.lower() and self._network_mode == "offline":
                issue = OPSECIssue(
                    id='loc_wifi_enabled_offline',
                    category=OPSECCategory.LOCATION,
                    severity=5,
                    title='WiFi Radio Enabled (Offline Mode)',
                    description='WiFi radio is enabled while in offline mode. Device sends probe requests that reveal location.',
                    recommendation='Disable WiFi radio: "sudo nmcli radio wifi off"',
                    auto_fixable=True,
                    fix_command='nmcli radio wifi off'
                )
                self._add_issue(issue)
                issues.append(issue)
            else:
                self._remove_issue('loc_wifi_enabled_offline')
        except:
            pass
        
        # Check for location services
        try:
            result = subprocess.run(['systemctl', 'is-active', 'geoclue'], 
                                  capture_output=True, text=True, timeout=5)
            if 'active' in result.stdout.lower():
                issue = OPSECIssue(
                    id='loc_geoclue_active',
                    category=OPSECCategory.LOCATION,
                    severity=6,
                    title='Location Service Active',
                    description='Geolocation service (geoclue) is running. Applications can query your location.',
                    recommendation='Stop geoclue: "sudo systemctl stop geoclue"',
                    auto_fixable=True,
                    fix_command='systemctl stop geoclue'
                )
                self._add_issue(issue)
                issues.append(issue)
        except:
            pass
        
        return issues
    
    # ========== ISSUE MANAGEMENT ==========
    
    def _add_issue(self, issue: OPSECIssue):
        """Add or update an issue"""
        self.issues[issue.id] = issue
    
    def _remove_issue(self, issue_id: str):
        """Remove an issue (resolved)"""
        if issue_id in self.issues:
            del self.issues[issue_id]
    
    def get_active_issues(self) -> List[OPSECIssue]:
        """Get all active issues sorted by severity"""
        return sorted(
            [i for i in self.issues.values() if i.is_active],
            key=lambda x: x.severity,
            reverse=True
        )
    
    def fix_issue(self, issue_id: str) -> Dict:
        """Attempt to auto-fix an issue"""
        if issue_id not in self.issues:
            return {'success': False, 'error': 'Issue not found'}
        
        issue = self.issues[issue_id]
        
        if not issue.auto_fixable:
            return {
                'success': False, 
                'error': 'This issue cannot be auto-fixed',
                'recommendation': issue.recommendation
            }
        
        try:
            # Execute fix command
            if issue.fix_command:
                subprocess.run(
                    issue.fix_command.split(),
                    capture_output=True,
                    timeout=30,
                    check=True
                )
            
            # Remove issue
            self._remove_issue(issue_id)
            
            return {
                'success': True,
                'message': f'Fixed: {issue.title}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Fix failed: {e}'
            }
    
    def fix_all_auto_fixable(self) -> Dict:
        """Attempt to fix all auto-fixable issues"""
        fixed = []
        failed = []
        
        for issue_id, issue in list(self.issues.items()):
            if issue.auto_fixable:
                result = self.fix_issue(issue_id)
                if result['success']:
                    fixed.append(issue.title)
                else:
                    failed.append(f"{issue.title}: {result.get('error', 'Unknown error')}")
        
        return {
            'fixed': fixed,
            'failed': failed,
            'total_fixed': len(fixed)
        }
    
    # ========== SCORING ==========
    
    def _calculate_score(self) -> OPSECScore:
        """Calculate current OPSEC score"""
        
        # Start with perfect scores
        category_scores = {cat.value: 100 for cat in OPSECCategory}
        
        # Deduct points for each issue
        for issue in self.issues.values():
            if issue.is_active:
                cat = issue.category.value
                # Deduct based on severity (max 10 points per issue)
                deduction = issue.severity * 3
                category_scores[cat] = max(0, category_scores[cat] - deduction)
        
        # Calculate weighted total
        total = 0
        for cat, weight in self.CATEGORY_WEIGHTS.items():
            total += category_scores[cat.value] * (weight / 100)
        
        total = int(total)
        
        # Determine threat level
        if total >= 90:
            threat_level = ThreatLevel.SECURE
        elif total >= 70:
            threat_level = ThreatLevel.GOOD
        elif total >= 50:
            threat_level = ThreatLevel.WARNING
        elif total >= 25:
            threat_level = ThreatLevel.DANGER
        else:
            threat_level = ThreatLevel.CRITICAL
        
        # Generate recommendations
        recommendations = []
        critical_issues = [i for i in self.issues.values() if i.severity >= 7]
        for issue in critical_issues[:3]:  # Top 3 critical
            recommendations.append(issue.recommendation)
        
        self.current_score = OPSECScore(
            total_score=total,
            threat_level=threat_level,
            category_scores=category_scores,
            active_issues=self.get_active_issues(),
            recommendations=recommendations
        )
        
        return self.current_score
    
    def get_score(self) -> OPSECScore:
        """Get current OPSEC score (recalculates if needed)"""
        if not self.current_score:
            self.run_all_checks()
            self._calculate_score()
        return self.current_score
    
    def get_score_summary(self) -> Dict:
        """Get a summary of current OPSEC status"""
        score = self.get_score()
        
        return {
            'score': score.total_score,
            'threat_level': score.threat_level.value,
            'issues_count': len(score.active_issues),
            'critical_count': len([i for i in score.active_issues if i.severity >= 7]),
            'categories': score.category_scores,
            'top_recommendations': score.recommendations[:3]
        }
    
    def get_detailed_report(self) -> str:
        """Generate detailed OPSEC report"""
        score = self.get_score()
        
        # Header
        report = []
        report.append("=" * 60)
        report.append("       OPSEC STATUS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall score
        score_bar = "█" * (score.total_score // 5) + "░" * (20 - score.total_score // 5)
        report.append(f"OVERALL SCORE: {score.total_score}/100 [{score_bar}]")
        report.append(f"THREAT LEVEL:  {score.threat_level.value.upper()}")
        report.append("")
        
        # Category breakdown
        report.append("CATEGORY BREAKDOWN:")
        report.append("-" * 40)
        for cat, cat_score in score.category_scores.items():
            bar = "█" * (cat_score // 10) + "░" * (10 - cat_score // 10)
            status = "✓" if cat_score >= 70 else "!" if cat_score >= 50 else "✗"
            report.append(f"  {status} {cat.upper():12} {cat_score:3}/100 [{bar}]")
        report.append("")
        
        # Active issues
        if score.active_issues:
            report.append(f"ACTIVE ISSUES ({len(score.active_issues)}):")
            report.append("-" * 40)
            for issue in score.active_issues[:10]:
                severity_indicator = "🔴" if issue.severity >= 7 else "🟡" if issue.severity >= 4 else "🟢"
                auto_fix = "[AUTO-FIX]" if issue.auto_fixable else ""
                report.append(f"  {severity_indicator} [{issue.severity}/10] {issue.title} {auto_fix}")
                report.append(f"      {issue.description[:60]}...")
            report.append("")
        else:
            report.append("NO ACTIVE ISSUES - Excellent OPSEC!")
            report.append("")
        
        # Recommendations
        if score.recommendations:
            report.append("TOP RECOMMENDATIONS:")
            report.append("-" * 40)
            for rec in score.recommendations:
                report.append(f"  → {rec}")
            report.append("")
        
        report.append("=" * 60)
        report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("Say 'fix opsec' to auto-fix resolvable issues")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    # ========== STATE UPDATES ==========
    
    def set_network_mode(self, mode: str):
        """Update current network mode (called by NetworkModeManager)"""
        self._network_mode = mode
    
    def set_stealth_mode(self, enabled: bool):
        """Update stealth mode status"""
        self._stealth_mode = enabled
    
    def set_ram_only_mode(self, enabled: bool):
        """Update RAM-only mode status"""
        self._ram_only_mode = enabled
    
    # ========== CALLBACKS ==========
    
    def register_callback(self, callback: Callable):
        """Register callback for score updates"""
        self._callbacks.append(callback)
    
    def _notify_callbacks(self):
        """Notify all registered callbacks"""
        for callback in self._callbacks:
            try:
                callback(self.current_score)
            except Exception as e:
                logger.error(f"Callback error: {e}")


# Global instance
_opsec_monitor: Optional[OPSECMonitor] = None


def get_opsec_monitor() -> OPSECMonitor:
    """Get global OPSECMonitor instance"""
    global _opsec_monitor
    if _opsec_monitor is None:
        _opsec_monitor = OPSECMonitor()
    return _opsec_monitor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    monitor = get_opsec_monitor()
    
    print("Running OPSEC checks...")
    monitor.run_all_checks()
    
    print("\n" + monitor.get_detailed_report())
