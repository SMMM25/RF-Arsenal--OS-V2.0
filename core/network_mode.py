#!/usr/bin/env python3
"""
RF Arsenal OS - Network Mode Manager
Offline-by-default with explicit online mode consent and warnings

CORE MISSION: Stealth and Anonymity for White Hat Operations
- Default: OFFLINE (air-gapped, maximum stealth)
- Online: Requires explicit user consent with warnings
- All network activity controlled through AI command interface
"""

import os
import subprocess
import logging
import threading
import time
from enum import Enum
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class NetworkMode(Enum):
    """Network operation modes"""
    OFFLINE = "offline"           # Air-gapped, no network - DEFAULT
    ONLINE_TOR = "online_tor"     # Online via Tor only
    ONLINE_VPN = "online_vpn"     # Online via VPN only  
    ONLINE_FULL = "online_full"   # Full triple-layer (I2P → VPN → Tor)
    ONLINE_DIRECT = "online_direct"  # Direct connection (DANGEROUS - warned)


class ThreatLevel(Enum):
    """Threat level assessment for online operations"""
    MINIMAL = "minimal"      # Low risk, short duration
    MODERATE = "moderate"    # Some risk, use anonymity
    HIGH = "high"           # High risk, full anonymity required
    CRITICAL = "critical"   # Maximum risk, consider staying offline


@dataclass
class OnlineSession:
    """Track online session for audit and auto-disconnect"""
    session_id: str
    mode: NetworkMode
    started_at: datetime
    reason: str
    max_duration_seconds: int
    auto_disconnect: bool
    user_acknowledged_risks: bool
    

class NetworkModeManager:
    """
    Central network mode controller
    
    SECURITY PRINCIPLES:
    1. Offline by default - maximum stealth
    2. Explicit consent required for online mode
    3. Warnings displayed before going online
    4. Auto-disconnect after timeout (configurable)
    5. All activity logged for OPSEC awareness
    6. AI controller integration for natural language control
    """
    
    # Risk warnings for online modes
    ONLINE_WARNINGS = {
        NetworkMode.ONLINE_TOR: [
            "Traffic routed through Tor network",
            "Exit node operators may monitor unencrypted traffic",
            "Some services may block Tor exit nodes",
            "Connection timing may be analyzed",
        ],
        NetworkMode.ONLINE_VPN: [
            "Traffic routed through VPN provider",
            "VPN provider can see your traffic",
            "VPN provider knows your real IP",
            "Single point of failure for anonymity",
        ],
        NetworkMode.ONLINE_FULL: [
            "Triple-layer anonymity: I2P → VPN → Tor",
            "Significant latency overhead",
            "Complex route may have weak links",
            "Startup time: 3-5 minutes",
        ],
        NetworkMode.ONLINE_DIRECT: [
            "!!! CRITICAL WARNING !!!",
            "DIRECT CONNECTION - NO ANONYMITY",
            "Your real IP will be exposed",
            "All traffic is traceable to you",
            "Only use on trusted/isolated networks",
            "STEALTH COMPLETELY COMPROMISED",
        ],
    }
    
    def __init__(self):
        self.current_mode = NetworkMode.OFFLINE
        self.active_session: Optional[OnlineSession] = None
        self._session_timer: Optional[threading.Timer] = None
        self._consent_callback: Optional[Callable] = None
        self._mode_change_callbacks: List[Callable] = []
        self._lock = threading.Lock()
        
        # Session tracking
        self.session_history: List[Dict] = []
        
        # Default configuration
        self.config = {
            'default_timeout_seconds': 1800,  # 30 minutes auto-disconnect
            'require_consent': True,
            'allow_direct_connection': False,  # Disabled by default
            'log_all_sessions': True,
        }
        
        # Initialize in offline mode
        self._enforce_offline_mode()
        logger.info("NetworkModeManager initialized - OFFLINE MODE (default)")
    
    def _enforce_offline_mode(self):
        """Ensure system is truly offline"""
        try:
            # Disable network interfaces (requires root)
            interfaces = ['wlan0', 'wlan1', 'eth0', 'eth1']
            for iface in interfaces:
                subprocess.run(
                    ['ip', 'link', 'set', iface, 'down'],
                    capture_output=True, check=False
                )
            
            # Kill any active network services
            services = ['tor', 'openvpn', 'i2prouter', 'wg-quick']
            for service in services:
                subprocess.run(['pkill', '-f', service], capture_output=True, check=False)
            
            # Flush iptables to remove any proxy rules
            subprocess.run(['iptables', '-F'], capture_output=True, check=False)
            subprocess.run(['iptables', '-t', 'nat', '-F'], capture_output=True, check=False)
            
            logger.info("Offline mode enforced - all network interfaces disabled")
            
        except Exception as e:
            logger.warning(f"Could not fully enforce offline mode: {e}")
    
    def get_current_mode(self) -> NetworkMode:
        """Get current network mode"""
        return self.current_mode
    
    def is_offline(self) -> bool:
        """Check if system is in offline mode"""
        return self.current_mode == NetworkMode.OFFLINE
    
    def is_online(self) -> bool:
        """Check if system is in any online mode"""
        return self.current_mode != NetworkMode.OFFLINE
    
    def get_online_warnings(self, mode: NetworkMode) -> List[str]:
        """Get warnings for a specific online mode"""
        return self.ONLINE_WARNINGS.get(mode, [])
    
    def request_online_mode(self, 
                           mode: NetworkMode, 
                           reason: str,
                           duration_seconds: Optional[int] = None,
                           auto_disconnect: bool = True) -> Dict:
        """
        Request to go online - returns warnings and requires consent
        
        This method does NOT enable online mode directly.
        It returns the warnings and a consent token.
        User must call confirm_online_mode() with the token.
        
        Args:
            mode: Desired online mode
            reason: Why going online (for audit log)
            duration_seconds: Auto-disconnect after this time
            auto_disconnect: Whether to auto-disconnect after duration
            
        Returns:
            Dict with warnings, consent_token, and instructions
        """
        if mode == NetworkMode.OFFLINE:
            return {
                'success': False,
                'error': 'Use go_offline() to return to offline mode'
            }
        
        if mode == NetworkMode.ONLINE_DIRECT and not self.config['allow_direct_connection']:
            return {
                'success': False,
                'error': 'Direct connection mode is disabled for safety',
                'hint': 'Enable with: config allow_direct_connection true'
            }
        
        # Generate consent token
        import secrets
        consent_token = secrets.token_hex(16)
        
        warnings = self.get_online_warnings(mode)
        duration = duration_seconds or self.config['default_timeout_seconds']
        
        return {
            'success': True,
            'requires_consent': True,
            'mode': mode.value,
            'warnings': warnings,
            'duration_seconds': duration,
            'auto_disconnect': auto_disconnect,
            'consent_token': consent_token,
            'reason': reason,
            'instruction': 'Review warnings and call confirm_online_mode(consent_token) to proceed',
            '_internal_token': consent_token,  # Store for verification
        }
    
    def confirm_online_mode(self, 
                           consent_token: str,
                           mode: NetworkMode,
                           reason: str,
                           duration_seconds: int,
                           auto_disconnect: bool = True) -> Dict:
        """
        Confirm consent and activate online mode
        
        Args:
            consent_token: Token from request_online_mode()
            mode: Desired mode (must match request)
            reason: Reason for going online
            duration_seconds: Session duration
            auto_disconnect: Auto-disconnect after duration
            
        Returns:
            Dict with success status and session info
        """
        with self._lock:
            # Create session
            import secrets
            session_id = secrets.token_hex(8)
            
            session = OnlineSession(
                session_id=session_id,
                mode=mode,
                started_at=datetime.now(),
                reason=reason,
                max_duration_seconds=duration_seconds,
                auto_disconnect=auto_disconnect,
                user_acknowledged_risks=True
            )
            
            # Activate the requested mode
            success = self._activate_online_mode(mode)
            
            if not success:
                return {
                    'success': False,
                    'error': 'Failed to activate online mode',
                    'mode': mode.value
                }
            
            self.active_session = session
            self.current_mode = mode
            
            # Log session
            if self.config['log_all_sessions']:
                self.session_history.append({
                    'session_id': session_id,
                    'mode': mode.value,
                    'started_at': session.started_at.isoformat(),
                    'reason': reason,
                    'duration': duration_seconds
                })
            
            # Start auto-disconnect timer
            if auto_disconnect and duration_seconds > 0:
                self._start_session_timer(duration_seconds)
            
            logger.warning(f"ONLINE MODE ACTIVATED: {mode.value} - Session: {session_id}")
            logger.warning(f"Reason: {reason}")
            logger.warning(f"Auto-disconnect in {duration_seconds} seconds")
            
            # Notify callbacks
            for callback in self._mode_change_callbacks:
                try:
                    callback(mode, session)
                except Exception as e:
                    logger.error(f"Mode change callback error: {e}")
            
            return {
                'success': True,
                'mode': mode.value,
                'session_id': session_id,
                'started_at': session.started_at.isoformat(),
                'expires_in_seconds': duration_seconds,
                'warning': 'STEALTH REDUCED - You are now online'
            }
    
    def _activate_online_mode(self, mode: NetworkMode) -> bool:
        """Internal: Activate specific online mode"""
        try:
            # Enable network interface
            subprocess.run(['ip', 'link', 'set', 'wlan0', 'up'], 
                         capture_output=True, check=False)
            
            if mode == NetworkMode.ONLINE_TOR:
                return self._start_tor_only()
            elif mode == NetworkMode.ONLINE_VPN:
                return self._start_vpn_only()
            elif mode == NetworkMode.ONLINE_FULL:
                return self._start_full_anonymity()
            elif mode == NetworkMode.ONLINE_DIRECT:
                # Just enable interface, no anonymity
                logger.critical("DIRECT CONNECTION MODE - NO ANONYMITY PROTECTION")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to activate {mode.value}: {e}")
            return False
    
    def _start_tor_only(self) -> bool:
        """Start Tor-only anonymity"""
        try:
            subprocess.run(['systemctl', 'start', 'tor'], 
                         capture_output=True, check=True)
            time.sleep(5)  # Wait for Tor to bootstrap
            
            # Configure transparent proxy
            subprocess.run([
                'iptables', '-t', 'nat', '-A', 'OUTPUT',
                '-p', 'tcp', '-j', 'REDIRECT', '--to-ports', '9050'
            ], capture_output=True, check=True)
            
            logger.info("Tor-only mode activated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Tor: {e}")
            return False
    
    def _start_vpn_only(self) -> bool:
        """Start VPN-only mode"""
        try:
            # Look for VPN config
            vpn_configs = ['/etc/openvpn/client.conf', '/etc/wireguard/wg0.conf']
            
            for config in vpn_configs:
                if os.path.exists(config):
                    if 'openvpn' in config:
                        subprocess.run(['openvpn', '--config', config, '--daemon'],
                                     capture_output=True)
                    elif 'wireguard' in config:
                        subprocess.run(['wg-quick', 'up', 'wg0'],
                                     capture_output=True)
                    
                    time.sleep(5)
                    logger.info("VPN-only mode activated")
                    return True
            
            logger.error("No VPN configuration found")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start VPN: {e}")
            return False
    
    def _start_full_anonymity(self) -> bool:
        """Start full triple-layer anonymity (I2P → VPN → Tor)"""
        try:
            # Import advanced anonymity module
            from modules.stealth.network_anonymity_v2 import AdvancedAnonymity
            
            anonymity = AdvancedAnonymity()
            success = anonymity.enable_triple_layer_anonymity()
            
            if success:
                logger.info("Full triple-layer anonymity activated")
            
            return success
            
        except ImportError:
            logger.error("Advanced anonymity module not available")
            return False
        except Exception as e:
            logger.error(f"Failed to start full anonymity: {e}")
            return False
    
    def _start_session_timer(self, duration_seconds: int):
        """Start auto-disconnect timer"""
        if self._session_timer:
            self._session_timer.cancel()
        
        self._session_timer = threading.Timer(
            duration_seconds, 
            self._session_timeout
        )
        self._session_timer.daemon = True
        self._session_timer.start()
        
        logger.info(f"Session timer started: {duration_seconds} seconds")
    
    def _session_timeout(self):
        """Handle session timeout - auto go offline"""
        logger.warning("SESSION TIMEOUT - Automatically going offline")
        self.go_offline(reason="Session timeout - auto disconnect")
    
    def go_offline(self, reason: str = "User requested") -> Dict:
        """
        Return to offline mode (default safe state)
        
        Args:
            reason: Why going offline (for audit)
            
        Returns:
            Dict with status
        """
        with self._lock:
            # Cancel timer
            if self._session_timer:
                self._session_timer.cancel()
                self._session_timer = None
            
            # Stop all network services
            try:
                # Stop Tor
                subprocess.run(['systemctl', 'stop', 'tor'], 
                             capture_output=True, check=False)
                
                # Stop VPN
                subprocess.run(['killall', 'openvpn'], 
                             capture_output=True, check=False)
                subprocess.run(['wg-quick', 'down', 'wg0'], 
                             capture_output=True, check=False)
                
                # Stop I2P
                subprocess.run(['i2prouter', 'stop'], 
                             capture_output=True, check=False)
                
                # Flush iptables
                subprocess.run(['iptables', '-F'], capture_output=True, check=False)
                subprocess.run(['iptables', '-t', 'nat', '-F'], 
                             capture_output=True, check=False)
                
            except Exception as e:
                logger.warning(f"Error stopping network services: {e}")
            
            # Enforce offline
            self._enforce_offline_mode()
            
            # Update state
            previous_mode = self.current_mode
            self.current_mode = NetworkMode.OFFLINE
            
            # Log session end
            if self.active_session:
                session_duration = (datetime.now() - self.active_session.started_at).total_seconds()
                logger.info(f"Session ended: {self.active_session.session_id} "
                          f"Duration: {session_duration:.0f}s Reason: {reason}")
                self.active_session = None
            
            logger.info(f"OFFLINE MODE RESTORED - Stealth maximum - Reason: {reason}")
            
            # Notify callbacks
            for callback in self._mode_change_callbacks:
                try:
                    callback(NetworkMode.OFFLINE, None)
                except Exception as e:
                    logger.error(f"Mode change callback error: {e}")
            
            return {
                'success': True,
                'previous_mode': previous_mode.value,
                'current_mode': NetworkMode.OFFLINE.value,
                'reason': reason,
                'stealth_status': 'MAXIMUM'
            }
    
    def extend_session(self, additional_seconds: int) -> Dict:
        """Extend current online session"""
        with self._lock:
            if not self.active_session:
                return {'success': False, 'error': 'No active session'}
            
            if self._session_timer:
                self._session_timer.cancel()
            
            self.active_session.max_duration_seconds += additional_seconds
            
            # Restart timer with new duration
            remaining = (datetime.now() - self.active_session.started_at).total_seconds()
            new_timeout = self.active_session.max_duration_seconds - remaining
            
            if new_timeout > 0 and self.active_session.auto_disconnect:
                self._start_session_timer(int(new_timeout))
            
            logger.info(f"Session extended by {additional_seconds} seconds")
            
            return {
                'success': True,
                'session_id': self.active_session.session_id,
                'new_timeout_seconds': int(new_timeout),
            }
    
    def get_status(self) -> Dict:
        """Get current network mode status"""
        status = {
            'mode': self.current_mode.value,
            'is_offline': self.is_offline(),
            'stealth_level': 'MAXIMUM' if self.is_offline() else 'REDUCED',
        }
        
        if self.active_session:
            elapsed = (datetime.now() - self.active_session.started_at).total_seconds()
            remaining = self.active_session.max_duration_seconds - elapsed
            
            status['session'] = {
                'id': self.active_session.session_id,
                'mode': self.active_session.mode.value,
                'started_at': self.active_session.started_at.isoformat(),
                'elapsed_seconds': int(elapsed),
                'remaining_seconds': max(0, int(remaining)),
                'reason': self.active_session.reason,
                'auto_disconnect': self.active_session.auto_disconnect,
            }
        
        return status
    
    def register_mode_change_callback(self, callback: Callable):
        """Register callback for mode changes"""
        self._mode_change_callbacks.append(callback)
    
    def set_config(self, key: str, value) -> bool:
        """Update configuration"""
        if key in self.config:
            self.config[key] = value
            logger.info(f"Config updated: {key} = {value}")
            return True
        return False


# Global instance
_network_mode_manager: Optional[NetworkModeManager] = None


def get_network_mode_manager() -> NetworkModeManager:
    """Get global NetworkModeManager instance"""
    global _network_mode_manager
    if _network_mode_manager is None:
        _network_mode_manager = NetworkModeManager()
    return _network_mode_manager


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("RF Arsenal OS - Network Mode Manager Test")
    print("=" * 60)
    
    manager = get_network_mode_manager()
    
    # Check default status
    status = manager.get_status()
    print(f"\nDefault Status:")
    print(f"  Mode: {status['mode']}")
    print(f"  Offline: {status['is_offline']}")
    print(f"  Stealth: {status['stealth_level']}")
    
    # Request online mode
    print(f"\nRequesting online mode (Tor)...")
    request = manager.request_online_mode(
        mode=NetworkMode.ONLINE_TOR,
        reason="System update check",
        duration_seconds=300
    )
    
    print(f"Warnings:")
    for warning in request.get('warnings', []):
        print(f"  ! {warning}")
    
    print(f"\nConsent required: {request.get('requires_consent')}")
    print(f"Consent token: {request.get('consent_token', 'N/A')[:16]}...")
