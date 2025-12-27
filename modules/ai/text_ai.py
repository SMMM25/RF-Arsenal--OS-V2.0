#!/usr/bin/env python3
"""
RF Arsenal OS - Text-Only AI Interface
Lightweight command interface without speech recognition
"""

import logging
from modules.ai.ai_controller import AIController

logger = logging.getLogger(__name__)

class TextAIInterface:
    """Simple text command interface"""
    
    def __init__(self, main_controller=None):
        """
        Initialize text AI interface
        
        Args:
            main_controller: Main system controller (optional)
        """
        self.logger = logging.getLogger('Text-AI')
        self.controller = AIController(main_controller)
        
    def start_cli(self):
        """Start command-line interface"""
        print("╔═══════════════════════════════════════════════════════╗")
        print("║       RF Arsenal OS - AI Command Interface           ║")
        print("╚═══════════════════════════════════════════════════════╝")
        print("")
        print("Natural Language Commands:")
        print("")
        print("📡 Cellular:")
        print("  • Start 4G base station")
        print("  • Start 5G base station with IMSI catching")
        print("  • Stop cellular operations")
        print("")
        print("📶 WiFi:")
        print("  • Scan WiFi networks")
        print("  • Deauth WiFi clients")
        print("  • Create evil twin access point")
        print("")
        print("🛰️ GPS:")
        print("  • Spoof GPS to 37.7749, -122.4194")
        print("  • Jam GPS signals")
        print("")
        print("🚁 Drones:")
        print("  • Detect drones")
        print("  • Jam drone frequencies")
        print("  • Auto-defend against drones")
        print("")
        print("📊 Spectrum:")
        print("  • Scan spectrum from 100 MHz to 6 GHz")
        print("  • Analyze frequency 2.4 GHz")
        print("")
        print("⚡ Jamming:")
        print("  • Jam 2.4 GHz")
        print("  • Jam WiFi band")
        print("  • Stop jamming")
        print("")
        print("🔍 Other:")
        print("  • Start SIGINT collection")
        print("  • Scan for IoT devices")
        print("  • Track NOAA satellite")
        print("  • Listen on amateur radio")
        print("  • Show system status")
        print("  • Emergency stop")
        print("")
        print("Type 'help' for more examples, 'exit' to quit")
        print("")
        
        while True:
            try:
                command = input("RF Arsenal> ").strip()
                
                if not command:
                    continue
                    
                if command.lower() in ['exit', 'quit', 'q']:
                    print("✓ Exiting RF Arsenal AI...")
                    break
                    
                if command.lower() in ['help', '?']:
                    self.show_help()
                    continue
                    
                if command.lower() == 'examples':
                    self.show_examples()
                    continue
                    
                # Execute command
                response = self.controller.execute_command(command)
                print(f"{response}\n")
                
            except KeyboardInterrupt:
                print("\n✓ Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")
                self.logger.error(f"Command error: {e}")
    
    def show_help(self):
        """Show help information"""
        print("")
        print("╔═══════════════════════════════════════════════════════╗")
        print("║                  COMMAND HELP                         ║")
        print("╚═══════════════════════════════════════════════════════╝")
        print("")
        print("RF Arsenal understands natural language!")
        print("")
        print("Command Structure:")
        print("  [Action] [Target] [Parameters]")
        print("")
        print("Actions:")
        print("  start, stop, scan, detect, jam, spoof, monitor, track")
        print("")
        print("Targets:")
        print("  cellular, wifi, gps, drone, spectrum, signals, etc.")
        print("")
        print("Examples:")
        print("  'start 5g base station'")
        print("  'scan wifi networks'")
        print("  'detect drones'")
        print("  'jam 2.4 ghz'")
        print("")
        print("Special Commands:")
        print("  status  - Show system status")
        print("  help    - Show this help")
        print("  examples - Show more examples")
        print("  exit    - Exit interface")
        print("")
    
    def show_examples(self):
        """Show example commands"""
        print("")
        print("╔═══════════════════════════════════════════════════════╗")
        print("║              EXAMPLE COMMANDS                         ║")
        print("╚═══════════════════════════════════════════════════════╝")
        print("")
        print("Cellular Operations:")
        print("  > Start 2G base station")
        print("  > Start 4G LTE base station with IMSI catcher")
        print("  > Start 5G base station")
        print("")
        print("WiFi Attacks:")
        print("  > Scan all WiFi networks")
        print("  > Deauthenticate WiFi clients")
        print("  > Create evil twin access point")
        print("  > Capture WPA handshakes")
        print("")
        print("GPS Spoofing:")
        print("  > Spoof GPS to 51.5074, -0.1278 (London)")
        print("  > Spoof GPS to 35.6762, 139.6503 altitude 50 (Tokyo)")
        print("  > Jam GPS signals")
        print("")
        print("Drone Warfare:")
        print("  > Detect drones in area")
        print("  > Jam all drone frequencies")
        print("  > Auto-defend against drones")
        print("  > Hijack drone control")
        print("")
        print("Spectrum Analysis:")
        print("  > Scan spectrum from 2.4 GHz to 2.5 GHz")
        print("  > Analyze frequency 433 MHz")
        print("  > Find quiet frequency")
        print("")
        print("Electronic Warfare:")
        print("  > Jam 2.4 GHz band")
        print("  > Jam GPS L1 frequency")
        print("  > Jam cellular 5G band")
        print("  > Stop all jamming")
        print("")
        print("Intelligence:")
        print("  > Start passive SIGINT collection")
        print("  > Monitor military VHF frequencies")
        print("  > Intercept cellular traffic")
        print("")
        print("Other Systems:")
        print("  > Start FMCW radar")
        print("  > Scan for IoT devices")
        print("  > Scan RFID tags")
        print("  > Track NOAA-19 satellite")
        print("  > Listen on ham radio 20m band")
        print("")
        print("System:")
        print("  > Show system status")
        print("  > Stop all operations")
        print("  > Emergency shutdown")
        print("")


# Standalone launcher
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    interface = TextAIInterface()
    interface.start_cli()
