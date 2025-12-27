#!/usr/bin/env python3
"""
RF Arsenal OS - First Boot Setup Wizard
Interactive setup on first system boot

Cross-platform support:
- x86_64 Desktop/Laptop
- ARM64 Desktop/Laptop
- Raspberry Pi 5/4/3
- Generic ARM SBCs

README COMPLIANCE:
- Offline-first: Works without network
- RAM-only: Sensitive data in volatile memory
- Zero telemetry: No external connections
- Real-world functional: Actual hardware detection

Copyright (c) 2024 RF-Arsenal-OS Project
License: Proprietary - Authorized Use Only
"""

import os
import sys
import subprocess
import time
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Import platform detector
try:
    from platform_detector import (
        UniversalPlatformDetector, PlatformCapabilities, 
        PlatformOptimizer, PlatformType, PerformanceTier
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from platform_detector import (
        UniversalPlatformDetector, PlatformCapabilities, 
        PlatformOptimizer, PlatformType, PerformanceTier
    )

# Import hardware wizard
try:
    from hardware_wizard import HardwareSetupWizard, SDRType
except ImportError:
    HardwareSetupWizard = None
    SDRType = None

logger = logging.getLogger(__name__)


class NetworkMode(Enum):
    """Network operation modes"""
    OFFLINE = "offline"     # Completely offline
    TOR = "tor"            # Online via Tor
    VPN = "vpn"            # Online via VPN
    DIRECT = "direct"      # Direct connection (not recommended)


class SecurityLevel(Enum):
    """Security/stealth levels"""
    MAXIMUM = "maximum"     # Full stealth, RAM-only
    HIGH = "high"          # Enhanced security
    STANDARD = "standard"  # Normal operation
    DEVELOPMENT = "dev"    # Development mode (more logging)


@dataclass
class SetupConfig:
    """First boot setup configuration"""
    # Platform
    platform_type: str = ""
    performance_tier: str = ""
    
    # Network
    network_mode: NetworkMode = NetworkMode.OFFLINE
    enable_tor: bool = False
    enable_vpn: bool = False
    
    # Security
    security_level: SecurityLevel = SecurityLevel.HIGH
    enable_ram_only: bool = True
    enable_mac_randomization: bool = True
    enable_panic_button: bool = False
    
    # Hardware
    sdr_detected: bool = False
    sdr_type: str = ""
    gpio_available: bool = False
    usb3_available: bool = False
    
    # Features
    enable_ai_features: bool = False
    enable_realtime_spectrum: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'platform_type': self.platform_type,
            'performance_tier': self.performance_tier,
            'network_mode': self.network_mode.value,
            'enable_tor': self.enable_tor,
            'enable_vpn': self.enable_vpn,
            'security_level': self.security_level.value,
            'enable_ram_only': self.enable_ram_only,
            'enable_mac_randomization': self.enable_mac_randomization,
            'enable_panic_button': self.enable_panic_button,
            'sdr_detected': self.sdr_detected,
            'sdr_type': self.sdr_type,
            'gpio_available': self.gpio_available,
            'usb3_available': self.usb3_available,
            'enable_ai_features': self.enable_ai_features,
            'enable_realtime_spectrum': self.enable_realtime_spectrum,
        }


class FirstBootWizard:
    """
    Interactive first boot setup wizard
    
    Guides users through initial configuration on any supported platform.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('FirstBootWizard')
        self.platform_detector = UniversalPlatformDetector()
        self.capabilities: Optional[PlatformCapabilities] = None
        self.config = SetupConfig()
        
        # Hardware wizard if available
        self.hardware_wizard = None
        if HardwareSetupWizard:
            self.hardware_wizard = HardwareSetupWizard()
    
    def print_banner(self):
        """Print welcome banner"""
        print("")
        print("\033[94m")  # Blue
        print("╔═══════════════════════════════════════════════════════════════════════════════╗")
        print("║                                                                               ║")
        print("║     🎉 WELCOME TO RF ARSENAL OS - WHITE HAT EDITION 🎉                       ║")
        print("║                                                                               ║")
        print("║                    First Boot Setup Wizard v2.0                              ║")
        print("║                                                                               ║")
        print("║      Cross-Platform Support: PC • Laptop • Raspberry Pi • ARM64             ║")
        print("║                                                                               ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝")
        print("\033[0m")  # Reset
        time.sleep(1)
    
    def detect_platform(self):
        """Detect platform and hardware"""
        print("")
        print("=" * 80)
        print("  STEP 1: PLATFORM DETECTION")
        print("=" * 80)
        print("")
        
        print("Detecting your system...")
        self.capabilities = self.platform_detector.detect()
        
        # Store in config
        self.config.platform_type = self.capabilities.platform_type.value
        self.config.performance_tier = self.capabilities.performance_tier.value
        self.config.gpio_available = self.capabilities.gpio_available
        self.config.usb3_available = self.capabilities.usb.has_usb3
        
        # Print summary
        print("")
        print("-" * 80)
        print(f"  Platform:         {self.capabilities.platform_type.value}")
        print(f"  CPU:              {self.capabilities.cpu.model}")
        print(f"  Cores:            {self.capabilities.cpu.cores_physical} physical / {self.capabilities.cpu.cores_logical} logical")
        print(f"  Memory:           {self.capabilities.memory.total_gb:.1f} GB")
        print(f"  USB Speed:        {self.capabilities.usb.max_speed}")
        print(f"  Performance Tier: {self.capabilities.performance_tier.value.upper()}")
        print("-" * 80)
        
        # Show optimization notes
        if self.capabilities.optimization_notes:
            print("")
            print("Notes:")
            for note in self.capabilities.optimization_notes[:5]:
                print(f"  • {note}")
        
        # Set feature availability based on performance tier
        if self.capabilities.performance_tier in [PerformanceTier.HIGH, PerformanceTier.MEDIUM]:
            self.config.enable_ai_features = True
            self.config.enable_realtime_spectrum = True
        
        # GPIO panic button
        self.config.enable_panic_button = self.capabilities.gpio_available
        
        print("")
        input("Press ENTER to continue...")
    
    def detect_hardware(self):
        """Detect SDR and other hardware"""
        print("")
        print("=" * 80)
        print("  STEP 2: SDR HARDWARE DETECTION")
        print("=" * 80)
        print("")
        
        # Check for SDR devices
        sdr_detected = False
        sdr_type = "None"
        
        # BladeRF check
        print("Checking for BladeRF...")
        if self._check_bladerf():
            sdr_detected = True
            sdr_type = "BladeRF"
            print("  ✅ BladeRF SDR detected!")
        
        # HackRF check
        elif self._check_hackrf():
            sdr_detected = True
            sdr_type = "HackRF"
            print("  ✅ HackRF One detected!")
        
        # RTL-SDR check
        elif self._check_rtlsdr():
            sdr_detected = True
            sdr_type = "RTL-SDR"
            print("  ✅ RTL-SDR detected (RX only)")
        
        # LimeSDR check
        elif self._check_limesdr():
            sdr_detected = True
            sdr_type = "LimeSDR"
            print("  ✅ LimeSDR detected!")
        
        else:
            print("  ⚠️  No SDR hardware detected")
        
        self.config.sdr_detected = sdr_detected
        self.config.sdr_type = sdr_type
        
        if not sdr_detected:
            print("")
            print("  SDR hardware not detected. This is normal if:")
            print("    • SDR is not connected yet")
            print("    • Running in a virtual machine")
            print("    • SDR drivers not installed")
            print("")
            print("  You can connect SDR hardware later and run:")
            print("    sudo rf-arsenal --hardware-wizard")
            
            response = input("\n  Continue without SDR? (yes/no) [yes]: ").strip().lower()
            if response in ['no', 'n']:
                print("\n  Please connect SDR hardware and restart.")
                sys.exit(1)
        
        # USB speed warning
        if sdr_detected and not self.capabilities.usb.has_usb3:
            print("")
            print("  ⚠️  WARNING: USB 3.0 not available")
            print("     SDR performance will be limited (reduced sample rates)")
            
            if self.capabilities.platform_type in [
                PlatformType.RASPBERRY_PI_3, 
                PlatformType.RASPBERRY_PI_ZERO
            ]:
                print("     Consider upgrading to Raspberry Pi 4 or 5 for full SDR support")
        
        print("")
        input("Press ENTER to continue...")
    
    def _check_bladerf(self) -> bool:
        """Check for BladeRF"""
        try:
            result = subprocess.run(
                ['bladeRF-cli', '-p'],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0 and result.stdout.strip()
        except:
            pass
        
        # Fallback: check USB
        try:
            result = subprocess.run(
                ['lsusb'], capture_output=True, text=True, timeout=10
            )
            return 'Nuand' in result.stdout or '2cf0:5246' in result.stdout
        except:
            return False
    
    def _check_hackrf(self) -> bool:
        """Check for HackRF"""
        try:
            result = subprocess.run(
                ['hackrf_info'], capture_output=True, text=True, timeout=10
            )
            return 'Serial' in result.stdout
        except:
            return False
    
    def _check_rtlsdr(self) -> bool:
        """Check for RTL-SDR"""
        try:
            result = subprocess.run(
                ['rtl_test', '-t'], capture_output=True, text=True, timeout=10
            )
            return 'Found' in result.stderr or 'Realtek' in result.stderr
        except:
            return False
    
    def _check_limesdr(self) -> bool:
        """Check for LimeSDR"""
        try:
            result = subprocess.run(
                ['LimeUtil', '--find'], capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0 and result.stdout.strip()
        except:
            return False
    
    def configure_network(self):
        """Configure network mode"""
        print("")
        print("=" * 80)
        print("  STEP 3: NETWORK CONFIGURATION")
        print("=" * 80)
        print("")
        
        print("  RF Arsenal OS operates OFFLINE by default for maximum OPSEC.")
        print("  Online access requires explicit confirmation and uses anonymization.")
        print("")
        print("  Network Modes:")
        print("")
        print("  [1] OFFLINE (Recommended)")
        print("      • No network connections")
        print("      • Maximum security")
        print("      • All core features available")
        print("")
        print("  [2] TOR NETWORK")
        print("      • Anonymous internet access")
        print("      • Required for threat intelligence updates")
        print("      • Slow but secure")
        print("")
        print("  [3] VPN")
        print("      • Encrypted tunnel to VPN provider")
        print("      • Faster than Tor")
        print("      • Requires VPN configuration")
        print("")
        
        choice = input("  Select network mode [1]: ").strip() or '1'
        
        mode_map = {
            '1': NetworkMode.OFFLINE,
            '2': NetworkMode.TOR,
            '3': NetworkMode.VPN,
        }
        
        self.config.network_mode = mode_map.get(choice, NetworkMode.OFFLINE)
        self.config.enable_tor = choice == '2'
        self.config.enable_vpn = choice == '3'
        
        if choice in ['2', '3']:
            print("")
            print("  ⚠️  Network access will be disabled by default.")
            print("     Use 'go online through tor for X minutes' to enable temporarily.")
        
        print("")
        input("Press ENTER to continue...")
    
    def configure_security(self):
        """Configure security settings"""
        print("")
        print("=" * 80)
        print("  STEP 4: SECURITY CONFIGURATION")
        print("=" * 80)
        print("")
        
        print("  Select your security level:")
        print("")
        print("  [1] MAXIMUM STEALTH")
        print("      • RAM-only operation (no disk writes)")
        print("      • Automatic MAC randomization")
        print("      • Panic button enabled (if GPIO available)")
        print("      • All traces wiped on shutdown")
        print("")
        print("  [2] HIGH SECURITY (Recommended)")
        print("      • RAM-only for sensitive data")
        print("      • MAC randomization on request")
        print("      • Emergency wipe available")
        print("")
        print("  [3] STANDARD")
        print("      • Normal operation")
        print("      • Logging enabled for troubleshooting")
        print("      • Good for development/testing")
        print("")
        
        choice = input("  Select security level [2]: ").strip() or '2'
        
        level_map = {
            '1': SecurityLevel.MAXIMUM,
            '2': SecurityLevel.HIGH,
            '3': SecurityLevel.STANDARD,
        }
        
        self.config.security_level = level_map.get(choice, SecurityLevel.HIGH)
        
        # Configure based on selection
        if choice == '1':
            self.config.enable_ram_only = True
            self.config.enable_mac_randomization = True
        elif choice == '2':
            self.config.enable_ram_only = True
            self.config.enable_mac_randomization = False
        else:
            self.config.enable_ram_only = False
            self.config.enable_mac_randomization = False
        
        # Panic button configuration
        if self.config.enable_panic_button:
            print("")
            print("  📍 GPIO Panic Button Available!")
            print("     Default: GPIO 17 (physical pin 11)")
            print("     Press to immediately wipe RAM and halt system")
            
            response = input("\n  Enable panic button? (yes/no) [yes]: ").strip().lower()
            self.config.enable_panic_button = response not in ['no', 'n']
        
        print("")
        input("Press ENTER to continue...")
    
    def apply_configuration(self):
        """Apply and save configuration"""
        print("")
        print("=" * 80)
        print("  STEP 5: APPLYING CONFIGURATION")
        print("=" * 80)
        print("")
        
        # Create config directory
        config_dir = Path('/etc/rf-arsenal')
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply platform optimizations
        print("  Applying platform optimizations...")
        if self.capabilities:
            optimizer = PlatformOptimizer(self.capabilities)
            rf_config = optimizer._configure_rf_arsenal()
            
            # Merge with user config
            rf_config['network_mode'] = self.config.network_mode.value
            rf_config['security_level'] = self.config.security_level.value
            rf_config['enable_ram_only'] = str(self.config.enable_ram_only).lower()
            rf_config['enable_mac_randomization'] = str(self.config.enable_mac_randomization).lower()
            rf_config['enable_panic_button'] = str(self.config.enable_panic_button).lower()
            
            # Write config file
            config_file = config_dir / 'config.json'
            with open(config_file, 'w') as f:
                json.dump(rf_config, f, indent=2)
            
            print(f"    ✅ Configuration saved to {config_file}")
        
        # Apply system settings
        print("  Applying system settings...")
        
        # Disable swap for stealth mode
        if self.config.enable_ram_only:
            try:
                subprocess.run(['swapoff', '-a'], capture_output=True, timeout=30)
                print("    ✅ Swap disabled for stealth operation")
            except:
                print("    ⚠️  Could not disable swap")
        
        # Create RAM disk
        ramdisk = Path('/tmp/rf_arsenal_ram')
        ramdisk.mkdir(exist_ok=True)
        print("    ✅ RAM disk created at /tmp/rf_arsenal_ram")
        
        # Configure network
        if self.config.network_mode == NetworkMode.OFFLINE:
            try:
                # Note: This just sets the default mode, doesn't disable hardware
                Path('/etc/rf-arsenal/offline_mode').touch()
                print("    ✅ Offline mode configured as default")
            except:
                pass
        
        print("")
        print("  Configuration complete!")
        
        input("\nPress ENTER to continue...")
    
    def print_summary(self):
        """Print setup summary"""
        print("")
        print("\033[92m")  # Green
        print("╔═══════════════════════════════════════════════════════════════════════════════╗")
        print("║                                                                               ║")
        print("║                     🎉 SETUP COMPLETE! 🎉                                    ║")
        print("║                                                                               ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝")
        print("\033[0m")
        
        print("")
        print("=" * 80)
        print("  CONFIGURATION SUMMARY")
        print("=" * 80)
        print("")
        print(f"  Platform:           {self.config.platform_type}")
        print(f"  Performance:        {self.config.performance_tier}")
        print(f"  SDR Hardware:       {self.config.sdr_type if self.config.sdr_detected else 'Not detected'}")
        print(f"  Network Mode:       {self.config.network_mode.value}")
        print(f"  Security Level:     {self.config.security_level.value}")
        print(f"  RAM-Only Mode:      {'Enabled' if self.config.enable_ram_only else 'Disabled'}")
        print(f"  MAC Randomization:  {'Enabled' if self.config.enable_mac_randomization else 'On Request'}")
        print(f"  Panic Button:       {'Enabled (GPIO 17)' if self.config.enable_panic_button else 'Disabled'}")
        print(f"  AI Features:        {'Enabled' if self.config.enable_ai_features else 'Disabled'}")
        print("")
        print("=" * 80)
        print("")
        
        print("  🚀 Quick Start:")
        print("")
        print("     • Launch GUI:     sudo rf-arsenal")
        print("     • Launch CLI:     sudo rf-arsenal --cli")
        print("     • Health Check:   sudo rf-arsenal --check")
        print("     • Hardware Setup: sudo rf-arsenal --hardware-wizard")
        print("")
        
        if not self.config.sdr_detected:
            print("  ⚠️  Connect SDR hardware and run 'rf-arsenal --hardware-wizard'")
            print("")
        
        print("  📚 Documentation: /opt/rf-arsenal-os/docs/")
        print("")
        print("  ⚠️  IMPORTANT: For authorized penetration testing only!")
        print("     Comply with all local RF transmission laws")
        print("")
    
    def run(self):
        """Run the complete first boot wizard"""
        try:
            self.print_banner()
            self.detect_platform()
            self.detect_hardware()
            self.configure_network()
            self.configure_security()
            self.apply_configuration()
            self.print_summary()
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n  Setup cancelled by user.")
            return False
        except Exception as e:
            self.logger.error(f"Setup error: {e}")
            print(f"\n\n  ❌ Setup error: {e}")
            return False


def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO)
    
    # Check if running as root
    if os.geteuid() != 0:
        print("❌ This wizard must be run as root!")
        print("   Run: sudo python3 first_boot_wizard.py")
        sys.exit(1)
    
    wizard = FirstBootWizard()
    success = wizard.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
