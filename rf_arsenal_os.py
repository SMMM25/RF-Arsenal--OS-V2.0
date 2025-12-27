#!/usr/bin/env python3
"""
RF Arsenal OS - Main Launcher
Unified entry point for GUI, CLI, and AI Command modes

CORE MISSION: Stealth and Anonymity for White Hat Operations
- OFFLINE BY DEFAULT - Maximum stealth
- AI Command Center for natural language control
- Online mode requires explicit consent with warnings

Copyright (c) 2024 RF-Arsenal-OS Project
License: MIT
"""

import sys
import os
import argparse
import warnings
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


class RFArsenalLauncher:
    """Main launcher for RF Arsenal OS"""
    
    def __init__(self):
        self.version = "1.0.0"
        self.hardware_available = {}
        
    def check_dependencies(self):
        """Check if required dependencies are available"""
        print("🔍 Checking dependencies...")
        
        # Check core dependencies
        dependencies = {
            'numpy': 'NumPy',
            'scipy': 'SciPy',
            'PyQt6': 'PyQt6',
            'cryptography': 'Cryptography',
            'scapy': 'Scapy',
            'psutil': 'psutil'
        }
        
        missing = []
        for module, name in dependencies.items():
            try:
                __import__(module)
                print(f"  ✅ {name}")
            except ImportError:
                print(f"  ❌ {name} (missing)")
                missing.append(name)
        
        if missing:
            print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
            print("Run: pip install -r install/requirements.txt")
            return False
        
        return True
    
    def check_hardware(self):
        """Check available hardware (graceful handling)"""
        print("\n🔌 Checking hardware...")
        
        # Check BladeRF
        try:
            import bladerf
            self.hardware_available['bladerf'] = True
            print("  ✅ BladeRF library available")
        except ImportError:
            self.hardware_available['bladerf'] = False
            print("  ⚠️  BladeRF library not found (install: sudo apt install libbladerf-dev)")
        
        # Check GPIO (Raspberry Pi)
        try:
            import RPi.GPIO as GPIO
            self.hardware_available['gpio'] = True
            print("  ✅ GPIO available (Raspberry Pi detected)")
        except (ImportError, RuntimeError):
            self.hardware_available['gpio'] = False
            print("  ℹ️  GPIO not available (not running on Raspberry Pi)")
        
        # Check Bluetooth
        try:
            import bluetooth
            self.hardware_available['bluetooth'] = True
            print("  ✅ Bluetooth available")
        except ImportError:
            self.hardware_available['bluetooth'] = False
            print("  ℹ️  Bluetooth not available (optional)")
        
        return True
    
    def system_health_check(self):
        """Perform system health check"""
        print("\n🏥 System Health Check")
        print("=" * 50)
        
        # Check if running as root (recommended for hardware access)
        if os.geteuid() != 0:
            print("  ⚠️  Not running as root (some features may be limited)")
            print("     Tip: sudo python3 rf_arsenal_os.py")
        else:
            print("  ✅ Running with root privileges")
        
        # Check Raspberry Pi model
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip()
                if 'Raspberry Pi 5' in model:
                    print(f"  🚀 Hardware: {model} (OPTIMAL)")
                elif 'Raspberry Pi 4' in model:
                    print(f"  ✅ Hardware: {model} (GOOD)")
                elif 'Raspberry Pi 3' in model:
                    print(f"  ⚠️  Hardware: {model} (MINIMUM)")
                else:
                    print(f"  ℹ️  Hardware: {model}")
        except FileNotFoundError:
            print("  ℹ️  Not running on Raspberry Pi")
        
        # Check USB devices (look for BladeRF)
        try:
            import subprocess
            lsusb_output = subprocess.check_output(['lsusb'], text=True)
            if 'Nuand' in lsusb_output or 'bladeRF' in lsusb_output:
                print("  ✅ BladeRF SDR detected (USB)")
            else:
                print("  ⚠️  BladeRF SDR not detected")
        except:
            pass
        
        # Check available disk space
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (2**30)
        print(f"  {'✅' if free_gb > 5 else '⚠️ '} Free disk space: {free_gb} GB")
        
        # Check memory
        import psutil
        mem = psutil.virtual_memory()
        mem_gb = mem.total // (2**30)
        print(f"  {'✅' if mem_gb >= 4 else '⚠️ '} Total RAM: {mem_gb} GB")
        
        print("=" * 50)
    
    def launch_gui(self):
        """Launch GUI mode"""
        print("\n🖥️  Launching RF Arsenal OS GUI...")
        
        try:
            from ui.main_gui import RFArsenalGUI
            from PyQt6.QtWidgets import QApplication
            
            app = QApplication(sys.argv)
            window = RFArsenalGUI(hardware_available=self.hardware_available)
            window.show()
            
            print("✅ GUI launched successfully")
            print("   Press Ctrl+C in terminal to exit\n")
            
            sys.exit(app.exec())
            
        except ImportError as e:
            print(f"❌ Failed to import GUI: {e}")
            print("   Make sure PyQt6 is installed: pip install PyQt6")
            return False
        except Exception as e:
            print(f"❌ GUI launch failed: {e}")
            return False
    
    def launch_cli(self):
        """Launch CLI mode with AI Command Center"""
        print("\n⌨️  Launching RF Arsenal OS AI Command Center...")
        print("=" * 60)
        
        # Import AI Command Center
        try:
            from core.ai_command_center import get_ai_command_center, run_cli
            
            print("✅ AI Command Center initialized")
            print("")
            print("  NETWORK MODE: OFFLINE (default - maximum stealth)")
            print("")
            print("  The AI understands natural language commands.")
            print("  Type 'help' for available commands.")
            print("  Type 'exit' to quit.")
            print("")
            print("  Examples:")
            print("    'go online with tor for updates'")
            print("    'scan wifi networks'")
            print("    'show status'")
            print("    'spoof gps to 37.77 -122.41'")
            print("=" * 60)
            
            # Run the AI Command Center CLI
            run_cli()
        
        except ImportError as e:
            print(f"❌ Failed to import AI Command Center: {e}")
            print("   Falling back to basic CLI...")
            self._launch_basic_cli()
    
    def _launch_basic_cli(self):
        """Fallback basic CLI mode"""
        print("\n⌨️  Basic CLI Mode (AI Command Center not available)")
        
        try:
            from modules.ai.ai_controller import AIController
            ai_controller = AIController(main_controller=None)
            
            while True:
                try:
                    command = input("\nrf-arsenal> ").strip()
                    
                    if command.lower() in ['exit', 'quit', 'q']:
                        print("👋 Goodbye!")
                        break
                    
                    elif command.lower() == 'help':
                        self.show_cli_help()
                    
                    elif command.lower() == 'status':
                        self.show_system_status()
                    
                    elif command:
                        result = ai_controller.execute_command(command)
                        if result:
                            print(f"{result}")
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Interrupted. Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        except ImportError as e:
            print(f"❌ Failed to import CLI modules: {e}")
            return False
    
    def show_cli_help(self):
        """Show CLI help"""
        help_text = """
📖 RF Arsenal OS - AI Command Center Help

═══════════════════════════════════════════════════════════
  NETWORK MODE (OFFLINE by default for maximum stealth)
═══════════════════════════════════════════════════════════

  go offline             Return to offline mode (default)
  go online tor          Enable Tor for anonymity
  go online vpn          Enable VPN
  go online full         Enable I2P → VPN → Tor (max anonymity)
  show network status    Show current network mode

═══════════════════════════════════════════════════════════
  RF OPERATIONS (Natural Language Commands)
═══════════════════════════════════════════════════════════

  WIFI:
    scan wifi networks
    deauth wifi clients
    create evil twin

  GPS:
    spoof gps to 37.7749 -122.4194
    jam gps

  CELLULAR:
    start 4g base station
    imsi catch
    target phone +1234567890

  DRONE:
    detect drones
    jam drones
    auto defend

  SPECTRUM:
    scan spectrum 100mhz to 6ghz
    analyze 2.4 ghz

  JAMMING:
    jam 2.4 ghz
    jam wifi
    stop jamming

  STEALTH:
    enable ram-only mode
    rotate mac address
    secure delete [file]

  EMERGENCY:
    emergency stop
    panic
    wipe all

═══════════════════════════════════════════════════════════
  SYSTEM
═══════════════════════════════════════════════════════════

  help [topic]      Show help (topics: network, wifi, gps, etc.)
  status            Show system status
  exit              Exit CLI

Documentation: https://github.com/SMMM25/RF-Arsenal-OS
        """
        print(help_text)
    
    def show_system_status(self):
        """Show current system status"""
        print("\n📊 System Status")
        print("=" * 50)
        print(f"Version: {self.version}")
        print(f"Hardware:")
        for hw, available in self.hardware_available.items():
            status = "✅ Available" if available else "❌ Not available"
            print(f"  • {hw.upper()}: {status}")
        print("=" * 50)
    
    def check_for_updates(self):
        """Check for system updates (if online)"""
        print("\n🔄 Checking for updates...")
        
        try:
            from update_manager import UpdateManager
            
            updater = UpdateManager()
            if updater.check_for_updates():
                print("  ✅ Updates available!")
                print("     Run: sudo python3 update_manager.py --install")
            else:
                print("  ✅ System is up to date")
        
        except ImportError:
            print("  ℹ️  Update manager not available")
        except Exception as e:
            print(f"  ℹ️  Update check skipped (offline mode or error: {e})")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='RF Arsenal OS - Complete RF Security Testing Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 rf_arsenal_os.py              # Launch GUI (default)
  python3 rf_arsenal_os.py --cli        # Launch CLI mode
  python3 rf_arsenal_os.py --check      # System health check only
  sudo python3 rf_arsenal_os.py         # Run with root privileges (recommended)

For documentation: https://github.com/SMMM25/RF-Arsenal-OS
        """
    )
    
    parser.add_argument(
        '--cli',
        action='store_true',
        help='Launch AI Command Center CLI mode instead of GUI'
    )
    
    parser.add_argument(
        '--ai',
        action='store_true',
        help='Alias for --cli (AI Command Center mode)'
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Run system health check and exit'
    )
    
    parser.add_argument(
        '--no-update-check',
        action='store_true',
        help='Skip update check on startup'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='RF Arsenal OS v1.0.0'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║        🛡️  RF ARSENAL OS - WHITE HAT EDITION  🛡️         ║
║                                                           ║
║     Complete RF Security Testing Platform v1.0.0         ║
║     Optimized for Raspberry Pi 5                         ║
║                                                           ║
║     FOR AUTHORIZED PENETRATION TESTING ONLY              ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Initialize launcher
    launcher = RFArsenalLauncher()
    
    # Check dependencies
    if not launcher.check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        sys.exit(1)
    
    # Check hardware
    launcher.check_hardware()
    
    # System health check
    launcher.system_health_check()
    
    # Check for updates (unless disabled)
    if not args.no_update_check:
        launcher.check_for_updates()
    
    # Exit if only health check requested
    if args.check:
        print("\n✅ Health check complete.")
        sys.exit(0)
    
    # Launch appropriate mode
    if args.cli or args.ai:
        launcher.launch_cli()
    else:
        launcher.launch_gui()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
