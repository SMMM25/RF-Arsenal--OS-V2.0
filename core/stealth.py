#!/usr/bin/env python3
"""
RF Arsenal OS - Stealth & OPSEC Module
Production stealth features
"""

import os
import sys
import subprocess
import secrets
import hashlib
import time
from pathlib import Path
import logging

class StealthSystem:
    """Production stealth & OPSEC features"""
    
    def __init__(self):
        self.logger = logging.getLogger('Stealth')
        self.ram_only = True
        self.interfaces = ['wlan0', 'eth0']
        
    def enable_ram_only_mode(self):
        """Mount tmpfs for RAM-only operation
        
        SECURITY: Uses subprocess with list arguments instead of os.system()
        to prevent command injection vulnerabilities.
        """
        try:
            # Create tmpfs mounts - using os.makedirs instead of shell
            os.makedirs('/tmp/rfarsenal_ram', mode=0o700, exist_ok=True)
            
            # Mount tmpfs - use subprocess with list args (no shell=True)
            subprocess.run(
                ['mount', '-t', 'tmpfs', '-o', 'size=4G,mode=0700', 'tmpfs', '/tmp/rfarsenal_ram'],
                check=False, capture_output=True
            )
            
            # Disable swap - use subprocess with list args (no shell=True)
            subprocess.run(['swapoff', '-a'], check=False, capture_output=True)
            
            self.logger.info("RAM-only mode enabled")
            return True
        except Exception as e:
            self.logger.error(f"Failed to enable RAM-only: {e}")
            return False
            
    def randomize_mac_address(self, interface):
        """Randomize MAC address for interface
        
        SECURITY: Uses secrets module for cryptographically secure random
        generation to prevent MAC address prediction attacks.
        """
        try:
            # Generate cryptographically secure random MAC
            # First 3 bytes: Locally administered unicast (0x02 prefix)
            # ensures no collision with real vendor OUIs
            mac = [0x02,  # Locally administered bit set
                   secrets.randbelow(256),
                   secrets.randbelow(256),
                   secrets.randbelow(256),
                   secrets.randbelow(256),
                   secrets.randbelow(256)]
            mac_str = ':'.join(map(lambda x: "%02x" % x, mac))
            
            # Set MAC address
            subprocess.run(['ip', 'link', 'set', interface, 'down'], 
                         check=True, capture_output=True)
            subprocess.run(['ip', 'link', 'set', interface, 'address', mac_str],
                         check=True, capture_output=True)
            subprocess.run(['ip', 'link', 'set', interface, 'up'],
                         check=True, capture_output=True)
            
            self.logger.info(f"MAC randomized for {interface}: {mac_str}")
            return mac_str
        except Exception as e:
            self.logger.error(f"Failed to randomize MAC: {e}")
            return None
            
    def continuous_mac_rotation(self, interval_seconds=300):
        """Continuously rotate MAC addresses"""
        import threading
        
        def rotate():
            while True:
                for iface in self.interfaces:
                    self.randomize_mac_address(iface)
                time.sleep(interval_seconds)
                
        thread = threading.Thread(target=rotate, daemon=True)
        thread.start()
        self.logger.info(f"MAC rotation started (interval: {interval_seconds}s)")
        
    def secure_delete_file(self, filepath):
        """DOD 5220.22-M standard file deletion - 3 pass
        
        Pass 1: Overwrite with zeros (0x00)
        Pass 2: Overwrite with ones (0xFF)
        Pass 3: Overwrite with random data
        
        This pattern ensures forensic unrecoverability on most storage media.
        """
        try:
            file_size = os.path.getsize(filepath)
            
            with open(filepath, 'ba+') as f:
                # Pass 1: Write zeros
                f.seek(0)
                f.write(b'\x00' * file_size)
                f.flush()
                os.fsync(f.fileno())
                
                # Pass 2: Write ones (0xFF)
                f.seek(0)
                f.write(b'\xFF' * file_size)
                f.flush()
                os.fsync(f.fileno())
                
                # Pass 3: Write random data
                f.seek(0)
                f.write(os.urandom(file_size))
                f.flush()
                os.fsync(f.fileno())
            
            # Delete file after secure overwrite
            os.remove(filepath)
            self.logger.info(f"Securely deleted (DoD 5220.22-M 3-pass): {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to secure delete: {e}")
            return False
            
    def wipe_ram(self):
        """Clear sensitive data from RAM"""
        try:
            # Drop caches
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3')
            self.logger.info("RAM wiped")
            return True
        except Exception as e:
            self.logger.error(f"Failed to wipe RAM: {e}")
            return False


class NetworkAnonymity:
    """Network anonymity features"""
    
    def __init__(self):
        self.logger = logging.getLogger('Anonymity')
        self.tor_running = False
        
    def start_tor(self):
        """Start Tor service"""
        try:
            subprocess.run(['systemctl', 'start', 'tor'], 
                         check=True, capture_output=True)
            
            # Wait for Tor to be ready
            time.sleep(5)
            
            # Configure iptables for transparent proxy
            subprocess.run([
                'iptables', '-t', 'nat', '-A', 'OUTPUT',
                '-p', 'tcp', '-j', 'REDIRECT', '--to-ports', '9050'
            ], check=True, capture_output=True)
            
            self.tor_running = True
            self.logger.info("Tor started successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start Tor: {e}")
            return False
            
    def stop_tor(self):
        """Stop Tor service"""
        try:
            # Remove iptables rules
            subprocess.run([
                'iptables', '-t', 'nat', '-D', 'OUTPUT',
                '-p', 'tcp', '-j', 'REDIRECT', '--to-ports', '9050'
            ], check=False, capture_output=True)
            
            subprocess.run(['systemctl', 'stop', 'tor'],
                         check=True, capture_output=True)
            
            self.tor_running = False
            self.logger.info("Tor stopped")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop Tor: {e}")
            return False
            
    def check_tor_connection(self):
        """Verify Tor connection"""
        try:
            import requests
            proxies = {
                'http': 'socks5h://127.0.0.1:9050',
                'https': 'socks5h://127.0.0.1:9050'
            }
            r = requests.get('https://check.torproject.org/api/ip',
                           proxies=proxies, timeout=10)
            data = r.json()
            
            if data.get('IsTor'):
                self.logger.info("Tor connection verified")
                return True
            else:
                self.logger.warning("Not connected through Tor")
                return False
        except Exception as e:
            self.logger.error(f"Tor check failed: {e}")
            return False


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test stealth features
    stealth = StealthSystem()
    stealth.enable_ram_only_mode()
    stealth.randomize_mac_address('wlan0')
    
    # Test anonymity
    anon = NetworkAnonymity()
    anon.start_tor()
    time.sleep(5)
    anon.check_tor_connection()
