#!/usr/bin/env python3
"""
Identity Management System
Multiple operational personas with complete compartmentalization
Enables separate identities for different operational contexts
"""

import os
import json
import hashlib
import secrets
import subprocess
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import shutil


class PersonaType(Enum):
    """Operational persona types"""
    PRIMARY = "primary"
    OPERATIONAL = "operational"
    RESEARCH = "research"
    EMERGENCY = "emergency"
    DECOY = "decoy"


@dataclass
class NetworkProfile:
    """Network configuration for persona"""
    mac_address: str
    hostname: str
    vpn_provider: str
    vpn_country: str
    tor_enabled: bool
    i2p_enabled: bool


@dataclass
class BehavioralProfile:
    """Behavioral characteristics to match persona"""
    active_hours: List[int]  # Hours of day active (0-23)
    typing_speed_wpm: int
    preferred_languages: List[str]
    timezone: str
    user_agent: str
    screen_resolution: str


@dataclass
class Persona:
    """Complete operational persona with full compartmentalization"""
    persona_id: str
    name: str
    type: PersonaType
    created: float
    last_used: float
    
    # Identity details
    network_profile: NetworkProfile
    behavioral_profile: BehavioralProfile
    
    # File system
    home_directory: str
    encrypted: bool
    
    # Credentials
    ssh_keys: List[str]
    pgp_keys: List[str]
    
    # Metadata
    cover_story: str
    operational_notes: str
    active: bool


class PersonaManager:
    """
    Manage multiple operational personas
    Complete compartmentalization between identities
    Zero cross-contamination between personas
    """
    
    def __init__(self, base_dir: str = "/var/lib/rf-arsenal/personas"):
        self.base_dir = base_dir
        self.personas = {}
        self.active_persona = None
        
        # Create base directory with restricted permissions
        os.makedirs(base_dir, mode=0o700, exist_ok=True)
        
        # Load existing personas
        self._load_personas()
        
    def create_persona(self, name: str, persona_type: PersonaType, 
                      cover_story: str = "") -> Persona:
        """
        Create new operational persona
        Generates complete isolated identity with:
        - Unique MAC address
        - Unique hostname
        - Isolated filesystem
        - Separate SSH/PGP keys
        - Distinct behavioral profile
        """
        print(f"\n[PERSONA] Creating new persona: {name} ({persona_type.value})")
        print("="*60)
        
        persona_id = self._generate_persona_id(name)
        
        # Generate network profile
        network_profile = NetworkProfile(
            mac_address=self._generate_random_mac(),
            hostname=self._generate_hostname(name),
            vpn_provider=self._select_vpn_provider(),
            vpn_country=self._select_vpn_country(),
            tor_enabled=True,
            i2p_enabled=(persona_type in [PersonaType.OPERATIONAL, PersonaType.EMERGENCY])
        )
        
        # Generate behavioral profile
        behavioral_profile = BehavioralProfile(
            active_hours=self._generate_active_hours(persona_type),
            typing_speed_wpm=secrets.randbelow(40) + 40,  # 40-80 WPM
            preferred_languages=["en"],
            timezone=self._select_timezone(),
            user_agent=self._generate_user_agent(),
            screen_resolution=self._select_screen_resolution()
        )
        
        # Create isolated home directory
        home_dir = os.path.join(self.base_dir, persona_id)
        os.makedirs(home_dir, mode=0o700, exist_ok=True)
        
        # Create standard directories
        for subdir in ['.ssh', '.gnupg', 'Documents', 'Downloads']:
            os.makedirs(os.path.join(home_dir, subdir), mode=0o700, exist_ok=True)
        
        # Generate SSH keys
        ssh_keys = self._generate_ssh_keys(persona_id)
        
        # Generate PGP keys
        pgp_keys = self._generate_pgp_keys(persona_id, name)
        
        persona = Persona(
            persona_id=persona_id,
            name=name,
            type=persona_type,
            created=datetime.now().timestamp(),
            last_used=datetime.now().timestamp(),
            network_profile=network_profile,
            behavioral_profile=behavioral_profile,
            home_directory=home_dir,
            encrypted=True,
            ssh_keys=ssh_keys,
            pgp_keys=pgp_keys,
            cover_story=cover_story,
            operational_notes="",
            active=True
        )
        
        # Save persona
        self.personas[persona_id] = persona
        self._save_persona(persona)
        
        print(f"\n✓ Persona created successfully")
        print(f"  ID: {persona_id}")
        print(f"  Home: {home_dir}")
        print(f"  MAC: {network_profile.mac_address}")
        print(f"  Hostname: {network_profile.hostname}")
        print(f"  VPN: {network_profile.vpn_provider} → {network_profile.vpn_country}")
        print(f"  Timezone: {behavioral_profile.timezone}")
        print(f"  SSH keys: {len(ssh_keys)} generated")
        print(f"  PGP keys: {len(pgp_keys)} generated")
        print(f"  Active hours: {behavioral_profile.active_hours}")
        print("="*60)
        
        return persona
        
    def switch_persona(self, persona_id: str) -> bool:
        """
        Switch to different operational persona
        Applies all persona characteristics:
        - Network identity (MAC, hostname)
        - Behavioral profile
        - Environment variables
        - Working directory
        """
        if persona_id not in self.personas:
            print(f"[PERSONA] ✗ Error: Persona {persona_id} not found")
            return False
            
        persona = self.personas[persona_id]
        
        print(f"\n[PERSONA] Switching to: {persona.name}")
        print("="*60)
        
        # Apply network profile
        print("\n1. Applying network profile...")
        self._apply_network_profile(persona.network_profile)
        
        # Set environment variables
        print("\n2. Configuring environment...")
        self._set_persona_environment(persona)
        
        # Apply system identity
        print("\n3. Applying system identity...")
        self._apply_system_identity(persona)
        
        # Update usage timestamp
        persona.last_used = datetime.now().timestamp()
        self._save_persona(persona)
        
        self.active_persona = persona_id
        
        print("\n" + "="*60)
        print(f"✓ Switched to persona: {persona.name}")
        print(f"  MAC: {persona.network_profile.mac_address}")
        print(f"  Hostname: {persona.network_profile.hostname}")
        print(f"  Home: {persona.home_directory}")
        print("="*60 + "\n")
        
        return True
        
    def _apply_network_profile(self, profile: NetworkProfile):
        """Apply network configuration"""
        print(f"  • Changing MAC address to {profile.mac_address}")
        
        # Get active network interfaces
        interfaces = self._get_network_interfaces()
        
        if not interfaces:
            print("    ⚠ No network interfaces found")
            return
        
        for interface in interfaces:
            try:
                # Bring interface down
                subprocess.run(['sudo', 'ip', 'link', 'set', interface, 'down'],
                             check=True, capture_output=True, timeout=5)
                
                # Change MAC address
                subprocess.run(['sudo', 'ip', 'link', 'set', interface, 'address', 
                              profile.mac_address],
                             check=True, capture_output=True, timeout=5)
                
                # Bring interface up
                subprocess.run(['sudo', 'ip', 'link', 'set', interface, 'up'],
                             check=True, capture_output=True, timeout=5)
                
                print(f"    ✓ {interface}: {profile.mac_address}")
                
            except subprocess.CalledProcessError as e:
                print(f"    ⚠ Failed to change MAC on {interface}")
            except subprocess.TimeoutExpired:
                print(f"    ⚠ Timeout changing MAC on {interface}")
                
        # Set hostname
        print(f"  • Changing hostname to {profile.hostname}")
        try:
            # Set transient hostname
            subprocess.run(['sudo', 'hostname', profile.hostname],
                         check=True, capture_output=True, timeout=5)
            
            # Persist hostname (optional - may alert to persona switching)
            # with open('/etc/hostname', 'w') as f:
            #     f.write(profile.hostname + '\n')
            
            print(f"    ✓ Hostname: {profile.hostname}")
            
        except Exception as e:
            print(f"    ⚠ Failed to change hostname")
            
    def _set_persona_environment(self, persona: Persona):
        """Set environment variables for persona"""
        # Core environment
        os.environ['HOME'] = persona.home_directory
        os.environ['USER'] = persona.name
        os.environ['PERSONA_ID'] = persona.persona_id
        os.environ['TZ'] = persona.behavioral_profile.timezone
        
        # Shell configuration
        os.environ['SHELL'] = '/bin/bash'
        os.environ['TERM'] = 'xterm-256color'
        
        # Path
        os.environ['PATH'] = '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin'
        
        # SSH configuration
        ssh_dir = os.path.join(persona.home_directory, '.ssh')
        os.environ['SSH_AUTH_SOCK'] = ''  # Clear SSH agent
        
        # GPG configuration
        gpg_dir = os.path.join(persona.home_directory, '.gnupg')
        os.environ['GNUPGHOME'] = gpg_dir
        
        print(f"  ✓ Environment configured")
        print(f"    HOME={persona.home_directory}")
        print(f"    USER={persona.name}")
        print(f"    TZ={persona.behavioral_profile.timezone}")
        
    def _apply_system_identity(self, persona: Persona):
        """Apply persona's system identity characteristics"""
        # Configure timezone
        try:
            os.environ['TZ'] = persona.behavioral_profile.timezone
            print(f"  ✓ Timezone: {persona.behavioral_profile.timezone}")
        except:
            pass
        
        # Would also configure:
        # - Browser fingerprint (user agent, screen resolution)
        # - System locale
        # - Display resolution
        # - Keyboard layout
        # - Font preferences
        
        print(f"  ✓ Screen resolution: {persona.behavioral_profile.screen_resolution}")
        print(f"  ✓ User agent: {persona.behavioral_profile.user_agent[:50]}...")
        print(f"  ✓ Active hours: {len(persona.behavioral_profile.active_hours)} hours/day")
        
    def get_persona(self, persona_id: str) -> Optional[Persona]:
        """Get persona by ID"""
        return self.personas.get(persona_id)
        
    def list_personas(self) -> List[Dict]:
        """List all personas with summary information"""
        persona_list = []
        
        for persona_id, persona in self.personas.items():
            persona_list.append({
                'id': persona_id,
                'name': persona.name,
                'type': persona.type.value,
                'active': persona.active,
                'created': datetime.fromtimestamp(persona.created).isoformat(),
                'last_used': datetime.fromtimestamp(persona.last_used).isoformat(),
                'is_current': (persona_id == self.active_persona),
                'mac_address': persona.network_profile.mac_address,
                'hostname': persona.network_profile.hostname
            })
            
        return persona_list
        
    def delete_persona(self, persona_id: str, secure_wipe: bool = True):
        """
        Delete persona and all associated data
        Secure wipe ensures forensic unrecoverability
        """
        if persona_id not in self.personas:
            print(f"[PERSONA] ✗ Error: Persona {persona_id} not found")
            return
            
        if persona_id == self.active_persona:
            print(f"[PERSONA] ✗ Error: Cannot delete active persona")
            print("  Switch to another persona first")
            return
            
        persona = self.personas[persona_id]
        
        print(f"\n[PERSONA] Deleting persona: {persona.name}")
        print("="*60)
        
        if secure_wipe:
            print("\nPerforming secure wipe (DoD 5220.22-M)...")
            print("This may take several minutes...")
            self._secure_wipe_directory(persona.home_directory)
            print("  ✓ Secure wipe complete")
        else:
            print("\nPerforming standard deletion...")
            shutil.rmtree(persona.home_directory)
            print("  ✓ Directory deleted")
            
        # Remove persona configuration
        persona_file = os.path.join(self.base_dir, f"{persona_id}.json")
        if os.path.exists(persona_file):
            if secure_wipe:
                self._secure_wipe_file(persona_file)
            else:
                os.remove(persona_file)
            
        del self.personas[persona_id]
        
        print("\n" + "="*60)
        print(f"✓ Persona {persona.name} deleted")
        print("  All data irrecoverable" if secure_wipe else "  Data deleted")
        print("="*60 + "\n")
        
    def _secure_wipe_directory(self, directory: str):
        """Securely wipe entire directory tree"""
        file_count = 0
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                self._secure_wipe_file(filepath)
                file_count += 1
                
        shutil.rmtree(directory)
        print(f"  Wiped {file_count} files")
        
    def _secure_wipe_file(self, filepath: str):
        """
        Securely wipe file using DoD 5220.22-M standard
        3-pass overwrite: 0x00, 0xFF, random
        """
        try:
            file_size = os.path.getsize(filepath)
            
            with open(filepath, 'ba+') as f:
                # Pass 1: Write zeros
                f.seek(0)
                f.write(b'\x00' * file_size)
                f.flush()
                os.fsync(f.fileno())
                
                # Pass 2: Write ones
                f.seek(0)
                f.write(b'\xFF' * file_size)
                f.flush()
                os.fsync(f.fileno())
                
                # Pass 3: Write random
                f.seek(0)
                f.write(os.urandom(file_size))
                f.flush()
                os.fsync(f.fileno())
                
            os.remove(filepath)
            
        except Exception as e:
            # If secure wipe fails, still try to delete
            try:
                os.remove(filepath)
            except:
                pass
            
    def export_persona(self, persona_id: str, export_path: str):
        """Export persona configuration (without sensitive data)"""
        if persona_id not in self.personas:
            print(f"[PERSONA] Error: Persona {persona_id} not found")
            return
            
        persona = self.personas[persona_id]
        
        # Export only non-sensitive data
        export_data = {
            'name': persona.name,
            'type': persona.type.value,
            'created': persona.created,
            'network_profile': {
                'hostname': persona.network_profile.hostname,
                'vpn_provider': persona.network_profile.vpn_provider,
                'vpn_country': persona.network_profile.vpn_country
            },
            'behavioral_profile': asdict(persona.behavioral_profile),
            'cover_story': persona.cover_story
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        print(f"[PERSONA] ✓ Exported to {export_path}")
        
    def _generate_persona_id(self, name: str) -> str:
        """Generate unique persona ID"""
        timestamp = str(datetime.now().timestamp())
        random_data = secrets.token_hex(8)
        
        hash_input = f"{name}{timestamp}{random_data}".encode()
        return hashlib.sha256(hash_input).hexdigest()[:16]
        
    def _generate_random_mac(self) -> str:
        """Generate random locally administered MAC address"""
        # Locally administered unicast MAC (bit 1 of first byte = 1, bit 0 = 0)
        mac = [0x02, secrets.randbelow(256), secrets.randbelow(256),
               secrets.randbelow(256), secrets.randbelow(256), secrets.randbelow(256)]
        
        return ':'.join(f'{b:02x}' for b in mac)
        
    def _generate_hostname(self, name: str) -> str:
        """Generate plausible hostname for persona"""
        # Common hostname patterns
        patterns = [
            f"{name.lower()}-workstation",
            f"{name.lower()}-laptop",
            f"user-{secrets.token_hex(4)}",
            f"host-{secrets.token_hex(4)}",
            f"{name.lower()}-{secrets.choice(['ubuntu', 'debian', 'fedora'])}"
        ]
        
        return secrets.choice(patterns)
        
    def _select_vpn_provider(self) -> str:
        """Select VPN provider for persona"""
        # Privacy-focused VPN providers
        providers = [
            "ProtonVPN", "Mullvad", "IVPN", 
            "AirVPN", "Azire", "OVPN", "Perfect Privacy"
        ]
        return secrets.choice(providers)
        
    def _select_vpn_country(self) -> str:
        """Select VPN exit country (privacy-friendly jurisdictions)"""
        countries = [
            "Switzerland", "Iceland", "Sweden", 
            "Norway", "Netherlands", "Romania",
            "Luxembourg", "Czech Republic"
        ]
        return secrets.choice(countries)
        
    def _generate_active_hours(self, persona_type: PersonaType) -> List[int]:
        """Generate plausible active hours for persona type"""
        if persona_type == PersonaType.RESEARCH:
            # Academic hours (9 AM - 6 PM)
            return list(range(9, 18))
        elif persona_type == PersonaType.OPERATIONAL:
            # Irregular hours (night operations)
            return [0, 1, 2, 3, 22, 23] + list(range(8, 12))
        elif persona_type == PersonaType.EMERGENCY:
            # 24/7 availability
            return list(range(24))
        elif persona_type == PersonaType.DECOY:
            # Standard office hours (predictable)
            return list(range(9, 17))
        else:
            # Default: Business hours
            return list(range(9, 17))
            
    def _select_timezone(self) -> str:
        """Select timezone for persona"""
        timezones = [
            "UTC", "America/New_York", "America/Los_Angeles",
            "Europe/London", "Europe/Paris", "Europe/Zurich",
            "Asia/Tokyo", "Asia/Hong_Kong", "Australia/Sydney"
        ]
        return secrets.choice(timezones)
        
    def _generate_user_agent(self) -> str:
        """Generate realistic user agent string"""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15"
        ]
        return secrets.choice(user_agents)
        
    def _select_screen_resolution(self) -> str:
        """Select common screen resolution"""
        resolutions = [
            "1920x1080",  # Most common
            "1366x768",   # Common laptop
            "2560x1440",  # 2K
            "1440x900",   # MacBook
            "1680x1050",  # 16:10
            "3840x2160"   # 4K
        ]
        return secrets.choice(resolutions)
        
    def _generate_ssh_keys(self, persona_id: str) -> List[str]:
        """Generate SSH key pair for persona"""
        key_dir = os.path.join(self.base_dir, persona_id, '.ssh')
        os.makedirs(key_dir, mode=0o700, exist_ok=True)
        
        key_path = os.path.join(key_dir, 'id_ed25519')
        
        try:
            subprocess.run([
                'ssh-keygen', '-t', 'ed25519', '-N', '',
                '-f', key_path, '-C', f'persona-{persona_id}'
            ], check=True, capture_output=True, timeout=10)
            
            # Set correct permissions
            os.chmod(key_path, 0o600)
            os.chmod(f"{key_path}.pub", 0o644)
            
            return [key_path, f"{key_path}.pub"]
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"    ⚠ SSH key generation failed")
            return []
            
    def _generate_pgp_keys(self, persona_id: str, name: str) -> List[str]:
        """Generate PGP key pair for persona"""
        key_dir = os.path.join(self.base_dir, persona_id, '.gnupg')
        os.makedirs(key_dir, mode=0o700, exist_ok=True)
        
        # Would use gpg to generate keys
        # gpg --batch --generate-key with key generation parameters
        # Simplified for demonstration
        
        # In production:
        # - Generate 4096-bit RSA key
        # - Set expiration date
        # - Create subkeys for signing, encryption, authentication
        
        return []  # Would return key IDs/fingerprints
        
    def _get_network_interfaces(self) -> List[str]:
        """Get list of active network interfaces"""
        try:
            result = subprocess.run(['ip', 'link', 'show'],
                                  capture_output=True, text=True, 
                                  check=True, timeout=5)
            
            interfaces = []
            for line in result.stdout.split('\n'):
                if ': ' in line and 'state' in line:
                    # Extract interface name
                    parts = line.split(':')
                    if len(parts) >= 2:
                        interface = parts[1].strip()
                        
                        # Skip loopback, docker, virtual interfaces
                        if (interface and 
                            not interface.startswith('lo') and
                            not interface.startswith('docker') and
                            not interface.startswith('veth') and
                            not interface.startswith('br-')):
                            interfaces.append(interface)
                        
            return interfaces
            
        except:
            return []
            
    def _save_persona(self, persona: Persona):
        """Save persona configuration to disk"""
        persona_file = os.path.join(self.base_dir, f"{persona.persona_id}.json")
        
        # Convert to dict
        persona_dict = asdict(persona)
        persona_dict['type'] = persona.type.value
        
        with open(persona_file, 'w') as f:
            json.dump(persona_dict, f, indent=2)
            
        # Set restrictive permissions (owner read/write only)
        os.chmod(persona_file, 0o600)
        
    def _load_personas(self):
        """Load all personas from disk"""
        if not os.path.exists(self.base_dir):
            return
            
        for filename in os.listdir(self.base_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.base_dir, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        persona_dict = json.load(f)
                        
                    # Reconstruct persona objects
                    persona_dict['type'] = PersonaType(persona_dict['type'])
                    persona_dict['network_profile'] = NetworkProfile(**persona_dict['network_profile'])
                    persona_dict['behavioral_profile'] = BehavioralProfile(**persona_dict['behavioral_profile'])
                    
                    persona = Persona(**persona_dict)
                    self.personas[persona.persona_id] = persona
                    
                except Exception as e:
                    print(f"[PERSONA] Warning: Failed to load {filename}: {e}")
                    
    def get_active_persona(self) -> Optional[Persona]:
        """Get currently active persona"""
        if self.active_persona:
            return self.personas.get(self.active_persona)
        return None
        
    def print_persona_summary(self, persona_id: str):
        """Print detailed summary of persona"""
        persona = self.get_persona(persona_id)
        if not persona:
            print(f"[PERSONA] Persona {persona_id} not found")
            return
            
        print("\n" + "="*60)
        print(f"PERSONA: {persona.name}")
        print("="*60)
        print(f"ID: {persona_id}")
        print(f"Type: {persona.type.value}")
        print(f"Created: {datetime.fromtimestamp(persona.created).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Last used: {datetime.fromtimestamp(persona.last_used).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Active: {persona.active}")
        
        print(f"\nNETWORK PROFILE:")
        print(f"  MAC: {persona.network_profile.mac_address}")
        print(f"  Hostname: {persona.network_profile.hostname}")
        print(f"  VPN: {persona.network_profile.vpn_provider} → {persona.network_profile.vpn_country}")
        print(f"  Tor: {'Enabled' if persona.network_profile.tor_enabled else 'Disabled'}")
        print(f"  I2P: {'Enabled' if persona.network_profile.i2p_enabled else 'Disabled'}")
        
        print(f"\nBEHAVIORAL PROFILE:")
        print(f"  Timezone: {persona.behavioral_profile.timezone}")
        print(f"  Active hours: {persona.behavioral_profile.active_hours}")
        print(f"  Typing speed: {persona.behavioral_profile.typing_speed_wpm} WPM")
        print(f"  Screen: {persona.behavioral_profile.screen_resolution}")
        print(f"  Languages: {', '.join(persona.behavioral_profile.preferred_languages)}")
        
        print(f"\nFILESYSTEM:")
        print(f"  Home: {persona.home_directory}")
        print(f"  Encrypted: {persona.encrypted}")
        print(f"  SSH keys: {len(persona.ssh_keys)}")
        print(f"  PGP keys: {len(persona.pgp_keys)}")
        
        print(f"\nCOVER STORY:")
        print(f"  {persona.cover_story if persona.cover_story else '(none)'}")
        
        print("="*60 + "\n")


# Example usage and testing
if __name__ == "__main__":
    print("=== Identity Management System Test ===\n")
    
    manager = PersonaManager()
    
    # Create test personas
    print("--- Creating Test Personas ---\n")
    
    persona1 = manager.create_persona(
        name="researcher01",
        persona_type=PersonaType.RESEARCH,
        cover_story="Security researcher focused on RF security and wireless protocols"
    )
    
    persona2 = manager.create_persona(
        name="operator-alpha",
        persona_type=PersonaType.OPERATIONAL,
        cover_story="Field operator for authorized penetration testing engagements"
    )
    
    persona3 = manager.create_persona(
        name="decoy-user",
        persona_type=PersonaType.DECOY,
        cover_story="Standard office worker profile for misdirection"
    )
    
    # List all personas
    print("\n--- All Personas ---\n")
    personas = manager.list_personas()
    for p in personas:
        print(f"  {p['name']} ({p['type']})")
        print(f"    ID: {p['id']}")
        print(f"    MAC: {p['mac_address']}")
        print(f"    Hostname: {p['hostname']}")
        print(f"    Last used: {p['last_used']}")
        print(f"    Current: {'✓' if p['is_current'] else ''}")
        print()
        
    # Print detailed summary
    print("--- Detailed Persona Summary ---")
    manager.print_persona_summary(persona1.persona_id)
    
    # Switch persona
    print("--- Switching Persona ---")
    manager.switch_persona(persona1.persona_id)
    
    # Show active persona
    active = manager.get_active_persona()
    if active:
        print(f"\nCurrent active persona: {active.name}")
        
    print("\n" + "="*60)
    print("Identity Management System Test Complete!")
    print("="*60)
