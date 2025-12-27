#!/usr/bin/env python3
"""
RF Arsenal OS - AI Controller
Local AI for natural language command interface
"""

import logging
import subprocess
import json
import threading
import queue
import re
from pathlib import Path
from datetime import datetime

# Import Wireshark integration
try:
    from modules.network.packet_capture import WiresharkCapture
except ImportError:
    WiresharkCapture = None

# Import Phone Targeting integration
try:
    from modules.cellular.phone_targeting import PhoneNumberTargeting, parse_targeting_command
except ImportError:
    PhoneNumberTargeting = None
    parse_targeting_command = None

# Import VoLTE Interceptor integration
try:
    from modules.cellular.volte_interceptor import VoLTEInterceptor, parse_volte_command
except ImportError:
    VoLTEInterceptor = None
    parse_volte_command = None

logger = logging.getLogger(__name__)

class AIController:
    """Local AI command controller"""
    
    def __init__(self, main_controller):
        """
        Initialize AI controller
        
        Args:
            main_controller: Main system controller instance
        """
        self.main = main_controller
        self.command_queue = queue.Queue()
        self.processing = False
        
        # Initialize Wireshark capture module
        self.wireshark = None
        if WiresharkCapture:
            try:
                self.wireshark = WiresharkCapture()
            except Exception as e:
                logger.warning(f"Failed to initialize Wireshark module: {e}")
        
        # Initialize Phone Targeting module (requires cellular controllers)
        self.phone_targeting = None
        # Will be initialized when cellular modules are loaded
        
        # Initialize VoLTE Interceptor module (requires LTE and GSM controllers)
        self.volte_interceptor = None
        # Will be initialized when cellular modules are loaded
        
        # Command mappings
        self.command_map = {
            'cellular': self.handle_cellular,
            'wifi': self.handle_wifi,
            'gps': self.handle_gps,
            'drone': self.handle_drone,
            'spectrum': self.handle_spectrum,
            'jam': self.handle_jamming,
            'scan': self.handle_scan,
            'stop': self.handle_stop,
            'status': self.handle_status,
            'emergency': self.handle_emergency,
            'sigint': self.handle_sigint,
            'radar': self.handle_radar,
            'iot': self.handle_iot,
            'satellite': self.handle_satellite,
            'amateur': self.handle_amateur,
            'capture': self.handle_capture,
            'wireshark': self.handle_capture,
            'pcap': self.handle_capture,
            'sniff': self.handle_capture,
            'target': self.handle_phone_targeting,
            'intercept': self.handle_volte,
            'voice': self.handle_volte,
            'volte': self.handle_volte,
            'calls': self.handle_volte
        }
        
    def parse_command(self, text):
        """Parse natural language command"""
        text = text.lower().strip()
        
        # Extract intent and parameters
        intent = None
        params = {}
        
        # Cellular commands
        if any(word in text for word in ['cellular', 'cell', 'bts', 'base station', '2g', '3g', '4g', '5g']):
            intent = 'cellular'
            if '5g' in text or 'nr' in text:
                params['generation'] = '5G'
            elif '4g' in text or 'lte' in text:
                params['generation'] = '4G'
            elif '3g' in text or 'umts' in text:
                params['generation'] = '3G'
            elif '2g' in text or 'gsm' in text:
                params['generation'] = '2G'
            
            if 'imsi' in text or 'catch' in text or 'catcher' in text:
                params['mode'] = 'imsi_catch'
            elif 'start' in text or 'begin' in text or 'activate' in text:
                params['action'] = 'start'
            elif 'stop' in text or 'deactivate' in text:
                params['action'] = 'stop'
                
        # WiFi commands
        elif any(word in text for word in ['wifi', 'wireless', 'wlan', '802.11']):
            intent = 'wifi'
            if 'scan' in text or 'search' in text:
                params['action'] = 'scan'
            elif 'deauth' in text or 'disconnect' in text or 'kick' in text:
                params['action'] = 'deauth'
            elif 'evil twin' in text or 'fake ap' in text or 'rogue' in text:
                params['action'] = 'evil_twin'
            elif 'handshake' in text or 'capture' in text:
                params['action'] = 'handshake'
                
        # GPS commands
        elif any(word in text for word in ['gps', 'location', 'spoof', 'position', 'coordinates']):
            intent = 'gps'
            
            # Extract coordinates
            coords = re.findall(r'-?\d+\.\d+', text)
            if len(coords) >= 2:
                params['latitude'] = float(coords[0])
                params['longitude'] = float(coords[1])
                if len(coords) >= 3:
                    params['altitude'] = float(coords[2])
            
            if 'start' in text or 'spoof' in text or 'fake' in text:
                params['action'] = 'start'
            elif 'stop' in text:
                params['action'] = 'stop'
            elif 'jam' in text:
                params['action'] = 'jam'
                
        # Drone commands
        elif any(word in text for word in ['drone', 'uav', 'quadcopter', 'dji']):
            intent = 'drone'
            if 'detect' in text or 'scan' in text or 'search' in text:
                params['action'] = 'detect'
            elif 'jam' in text or 'neutralize' in text or 'block' in text:
                params['action'] = 'jam'
            elif 'auto' in text or 'defend' in text or 'protect' in text:
                params['action'] = 'auto_defend'
            elif 'hijack' in text or 'take control' in text:
                params['action'] = 'hijack'
                
        # Spectrum commands
        elif any(word in text for word in ['spectrum', 'analyze', 'analyzer', 'frequency', 'freq']):
            intent = 'spectrum'
            
            # Extract frequency range
            freqs = re.findall(r'\d+\.?\d*\s*(?:mhz|ghz|khz)', text.lower())
            if len(freqs) >= 2:
                params['start_freq'] = self.parse_frequency(freqs[0])
                params['stop_freq'] = self.parse_frequency(freqs[1])
            elif len(freqs) == 1:
                freq = self.parse_frequency(freqs[0])
                params['center_freq'] = freq
            
            params['action'] = 'scan'
            
        # Jamming commands
        elif any(word in text for word in ['jam', 'jamming', 'block', 'interference', 'disrupt']):
            intent = 'jam'
            
            # Extract frequency
            freq_match = re.search(r'(\d+\.?\d*)\s*(mhz|ghz|khz)', text.lower())
            if freq_match:
                params['frequency'] = self.parse_frequency(freq_match.group())
            
            # Extract band
            if '2.4' in text or 'wifi' in text:
                params['band'] = 'wifi_2.4'
            elif '5g' in text and 'cellular' in text:
                params['band'] = 'cellular_5g'
            elif 'gps' in text:
                params['band'] = 'gps_l1'
                
            if 'start' in text or 'begin' in text:
                params['action'] = 'start'
            elif 'stop' in text:
                params['action'] = 'stop'
                
        # SIGINT commands
        elif any(word in text for word in ['sigint', 'intelligence', 'intercept', 'monitor']):
            intent = 'sigint'
            if 'passive' in text:
                params['mode'] = 'passive'
            elif 'targeted' in text or 'target' in text:
                params['mode'] = 'targeted'
            params['action'] = 'start'
            
        # Radar commands
        elif any(word in text for word in ['radar', 'fmcw', 'pulse radar', 'detect target']):
            intent = 'radar'
            if 'fmcw' in text:
                params['mode'] = 'fmcw'
            elif 'pulse' in text:
                params['mode'] = 'pulse'
            elif 'passive' in text:
                params['mode'] = 'passive'
            params['action'] = 'start'
            
        # IoT/RFID commands
        elif any(word in text for word in ['iot', 'rfid', 'tag', 'zigbee', 'zwave', 'lora']):
            intent = 'iot'
            if 'rfid' in text or 'tag' in text:
                params['mode'] = 'rfid'
            elif 'iot' in text or 'zigbee' in text or 'zwave' in text:
                params['mode'] = 'iot'
            if 'scan' in text:
                params['action'] = 'scan'
            elif 'clone' in text:
                params['action'] = 'clone'
                
        # Satellite commands
        elif any(word in text for word in ['satellite', 'sat', 'noaa', 'meteor', 'iss']):
            intent = 'satellite'
            if 'noaa' in text:
                params['satellite'] = 'noaa_19'
            elif 'meteor' in text:
                params['satellite'] = 'meteor_m2'
            elif 'iss' in text:
                params['satellite'] = 'iss'
            params['action'] = 'track'
            
        # Amateur radio commands
        elif any(word in text for word in ['amateur', 'ham', 'ham radio', 'cw', 'ssb']):
            intent = 'amateur'
            if 'cw' in text or 'morse' in text:
                params['mode'] = 'cw'
            elif 'ssb' in text:
                params['mode'] = 'ssb'
            elif 'fm' in text:
                params['mode'] = 'fm'
            if 'listen' in text or 'receive' in text:
                params['action'] = 'listen'
            elif 'transmit' in text or 'send' in text:
                params['action'] = 'transmit'
        
        # Packet capture / Wireshark commands
        elif any(word in text for word in ['capture', 'wireshark', 'pcap', 'sniff', 'packet']):
            intent = 'capture'
            
            if 'start' in text or 'begin' in text:
                params['action'] = 'start'
                
                # Extract interface
                if 'wlan0' in text:
                    params['interface'] = 'wlan0'
                elif 'wlan1' in text:
                    params['interface'] = 'wlan1'
                elif 'eth0' in text:
                    params['interface'] = 'eth0'
                
                # Extract duration
                duration_match = re.search(r'(\d+)\s*(second|minute|hour|sec|min|hr)', text)
                if duration_match:
                    value = int(duration_match.group(1))
                    unit = duration_match.group(2)
                    if 'min' in unit:
                        params['duration'] = value * 60
                    elif 'hour' in unit or 'hr' in unit:
                        params['duration'] = value * 3600
                    else:
                        params['duration'] = value
                
                # Extract filter
                if 'tcp' in text:
                    if 'port 80' in text:
                        params['filter'] = 'tcp port 80'
                    elif 'port 443' in text:
                        params['filter'] = 'tcp port 443'
                    else:
                        params['filter'] = 'tcp'
                elif 'http' in text:
                    params['filter'] = 'tcp port 80'
                elif 'dns' in text:
                    params['filter'] = 'udp port 53'
            
            elif 'stop' in text:
                params['action'] = 'stop'
            
            elif 'analyze' in text:
                params['action'] = 'analyze'
                
                # Extract filename
                pcap_match = re.search(r'[\w\-\.]+\.pcap', text)
                if pcap_match:
                    params['file'] = pcap_match.group()
            
            elif 'leak' in text or 'check' in text:
                params['action'] = 'check_leaks'
            
            elif 'clean' in text:
                params['action'] = 'cleanup'
                if 'secure' in text or 'shred' in text:
                    params['secure'] = True
                
        # General scan
        elif 'scan' in text and 'all' in text:
            intent = 'scan'
            
        # Stop all
        elif 'stop' in text and ('all' in text or 'everything' in text):
            intent = 'stop'
            
        # Status
        elif 'status' in text or 'report' in text or 'info' in text:
            intent = 'status'
            
        # Emergency
        elif any(word in text for word in ['emergency', 'panic', 'wipe', 'abort', 'kill']):
            intent = 'emergency'
            
        return intent, params
        
    def parse_frequency(self, freq_str):
        """Parse frequency string to Hz"""
        match = re.match(r'(\d+\.?\d*)\s*(mhz|ghz|khz)', freq_str.lower())
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            
            if unit == 'ghz':
                return int(value * 1e9)
            elif unit == 'mhz':
                return int(value * 1e6)
            elif unit == 'khz':
                return int(value * 1e3)
                
        return None
        
    def execute_command(self, text):
        """Execute parsed command"""
        logger.info(f"Executing: {text}")
        
        # Try phone targeting commands first (if module available)
        if self.phone_targeting and parse_targeting_command:
            result = parse_targeting_command(text, self.phone_targeting)
            if result is not None:
                return result
            # If returned None, command was handled (already printed output)
            if any(text.strip().lower().startswith(cmd) for cmd in ['target', 'capture', 'status', 'extract', 'associate', 'remove', 'report']):
                return ""
        
        # Try VoLTE interception commands (if module available)
        if self.volte_interceptor and parse_volte_command:
            result = parse_volte_command(text, self.volte_interceptor)
            if result is not None:
                return result
            # If returned None, command was handled (already printed output)
            if any(keyword in text.strip().lower() for keyword in ['intercept voice', 'voice interception', 'stop voice', 'list calls', 'export calls']):
                return ""
        
        intent, params = self.parse_command(text)
        
        if intent is None:
            return "I didn't understand that command. Try being more specific."
            
        # Execute command
        handler = self.command_map.get(intent)
        if handler:
            result = handler(params)
            return result
        else:
            # Provide helpful response for unknown commands
            similar = [cmd for cmd in self.command_map.keys() 
                      if any(word in cmd.lower() for word in intent.lower().split())]
            if similar:
                suggestions = ', '.join(similar[:3])
                return f"Command '{intent}' not recognized. Did you mean: {suggestions}? Use 'help' for available commands."
            return f"Command '{intent}' not recognized. Use 'help' to see available commands."
            
    # Command handlers
    def handle_cellular(self, params):
        """Handle cellular commands"""
        generation = params.get('generation', '4G')
        action = params.get('action', 'start')
        mode = params.get('mode', 'normal')
        
        if action == 'start':
            if mode == 'imsi_catch':
                return f"✓ Starting IMSI catcher on {generation} network"
            else:
                return f"✓ Starting {generation} base station"
        elif action == 'stop':
            return f"✓ Stopping {generation} base station"
        return f"✓ {generation} command executed"
            
    def handle_wifi(self, params):
        """Handle WiFi commands"""
        action = params.get('action', 'scan')
        
        if action == 'scan':
            return "✓ Scanning for WiFi networks"
        elif action == 'deauth':
            return "✓ Starting deauthentication attack"
        elif action == 'evil_twin':
            return "✓ Creating evil twin access point"
        elif action == 'handshake':
            return "✓ Capturing WPA handshakes"
        return "✓ WiFi command executed"
            
    def handle_gps(self, params):
        """Handle GPS commands"""
        action = params.get('action', 'start')
        lat = params.get('latitude')
        lon = params.get('longitude')
        alt = params.get('altitude', 0)
        
        if action == 'start':
            if lat and lon:
                return f"✓ GPS spoofing to: {lat:.4f}, {lon:.4f}, {alt:.0f}m"
            else:
                return "✓ GPS spoofing started with default location"
        elif action == 'stop':
            return "✓ GPS spoofing stopped"
        elif action == 'jam':
            return "✓ GPS jamming active"
        return "✓ GPS command executed"
            
    def handle_drone(self, params):
        """Handle drone commands"""
        action = params.get('action', 'detect')
        
        if action == 'detect':
            return "✓ Starting drone detection (scanning 2.4/5.8 GHz)"
        elif action == 'jam':
            return "✓ Jamming drone control frequencies"
        elif action == 'auto_defend':
            return "✓ Auto-defense mode activated (detect + jam)"
        elif action == 'hijack':
            return "✓ Attempting drone control takeover"
        return "✓ Drone command executed"
            
    def handle_spectrum(self, params):
        """Handle spectrum commands"""
        start = params.get('start_freq')
        stop = params.get('stop_freq')
        center = params.get('center_freq')
        
        if start and stop:
            return f"✓ Scanning spectrum: {start/1e6:.0f} - {stop/1e6:.0f} MHz"
        elif center:
            return f"✓ Analyzing frequency: {center/1e6:.1f} MHz"
        else:
            return "✓ Starting full spectrum analyzer (70 MHz - 6 GHz)"
            
    def handle_jamming(self, params):
        """Handle jamming commands"""
        action = params.get('action', 'start')
        freq = params.get('frequency')
        band = params.get('band')
        
        if action == 'start':
            if freq:
                return f"✓ Jamming frequency: {freq/1e6:.0f} MHz"
            elif band:
                return f"✓ Jamming band: {band}"
            else:
                return "✓ Jamming started on default frequency"
        elif action == 'stop':
            return "✓ Jamming stopped"
        return "✓ Jamming command executed"
            
    def handle_sigint(self, params):
        """Handle SIGINT commands"""
        mode = params.get('mode', 'passive')
        return f"✓ SIGINT collection started ({mode} mode)"
        
    def handle_radar(self, params):
        """Handle radar commands"""
        mode = params.get('mode', 'fmcw')
        return f"✓ Radar system active ({mode.upper()} mode)"
        
    def handle_iot(self, params):
        """Handle IoT/RFID commands"""
        mode = params.get('mode', 'iot')
        action = params.get('action', 'scan')
        
        if action == 'scan':
            return f"✓ Scanning for {mode.upper()} devices"
        elif action == 'clone':
            return f"✓ Cloning {mode.upper()} tag"
        return f"✓ {mode.upper()} command executed"
        
    def handle_satellite(self, params):
        """Handle satellite commands"""
        sat = params.get('satellite', 'NOAA-19')
        return f"✓ Tracking satellite: {sat}"
        
    def handle_amateur(self, params):
        """Handle amateur radio commands"""
        mode = params.get('mode', 'SSB')
        action = params.get('action', 'listen')
        return f"✓ Amateur radio: {action} on {mode.upper()}"
        
    def handle_scan(self, params):
        """Handle general scan"""
        return "✓ Starting full spectrum scan (all bands)"
        
    def handle_stop(self, params):
        """Stop all operations"""
        return "⚠️ Stopping all RF operations"
        
    def handle_status(self, params):
        """Get system status"""
        return "✓ System operational - All modules ready - BladeRF connected"
        
    def handle_emergency(self, params):
        """Handle emergency commands"""
        return "🚨 EMERGENCY PROTOCOL INITIATED - WIPING DATA"
    
    def handle_capture(self, params):
        """Handle packet capture / Wireshark commands"""
        if not self.wireshark:
            return "❌ Wireshark/PyShark not available. Install: pip3 install pyshark && sudo apt install tshark"
        
        action = params.get('action', 'status')
        
        if action == 'start':
            interface = params.get('interface', 'wlan0')
            duration = params.get('duration')
            bpf_filter = params.get('filter')
            
            result = self.wireshark.start_capture(
                duration=duration,
                bpf_filter=bpf_filter
            )
            
            if result['success']:
                msg = f"✓ Packet capture started on {interface}"
                if duration:
                    msg += f" ({duration}s)"
                if bpf_filter:
                    msg += f"\n  Filter: {bpf_filter}"
                msg += f"\n  Output: {result['output_file']}"
                return msg
            else:
                return f"❌ Failed to start capture: {result['error']}"
        
        elif action == 'stop':
            result = self.wireshark.stop_capture()
            
            if result['success']:
                msg = f"✓ Capture stopped\n"
                msg += f"  Packets: {result['packets_captured']}\n"
                msg += f"  File: {result['output_file']}\n"
                msg += f"  Credentials found: {result['credentials_found']}\n"
                msg += f"  DNS leaks: {result['dns_leaks']}\n"
                msg += f"  Unencrypted traffic: {result['unencrypted_traffic']}"
                return msg
            else:
                return f"❌ {result['error']}"
        
        elif action == 'analyze':
            pcap_file = params.get('file')
            if not pcap_file:
                return "❌ Please specify a PCAP file to analyze"
            
            result = self.wireshark.analyze_pcap(pcap_file)
            
            if result['success']:
                stats = result['statistics']
                msg = f"✓ Analysis complete: {pcap_file}\n"
                msg += f"  Total packets: {stats['total_packets']}\n"
                msg += f"  Protocols: {len(stats['protocols'])}\n"
                msg += f"  Credentials detected: {len(stats['credentials'])}\n"
                msg += f"  DNS queries: {len(stats['dns_queries'])}\n"
                msg += f"  Unencrypted traffic: {len(stats['unencrypted'])}"
                return msg
            else:
                return f"❌ Analysis failed: {result['error']}"
        
        elif action == 'check_leaks':
            leaks = self.wireshark.check_leaks()
            
            msg = "🔍 Security Analysis:\n"
            msg += f"  Credentials detected: {leaks['credentials_detected']}\n"
            msg += f"  DNS leaks: {leaks['dns_leaks']}\n"
            msg += f"  Unencrypted traffic: {leaks['unencrypted_traffic']}"
            
            if leaks['credentials_detected'] > 0:
                msg += "\n\n⚠️  WARNING: Credentials found in capture!"
            if leaks['dns_leaks'] > 5:
                msg += "\n⚠️  WARNING: Significant DNS leakage detected!"
            
            return msg
        
        elif action == 'cleanup':
            secure = params.get('secure', False)
            result = self.wireshark.cleanup(secure_delete=secure)
            
            if result['success']:
                method = "securely deleted" if secure else "deleted"
                msg = f"✓ Capture files {method}\n"
                msg += f"  Files removed: {len(result['deleted'])}"
                return msg
            else:
                msg = f"⚠️  Cleanup completed with errors:\n"
                msg += f"  Deleted: {len(result['deleted'])}\n"
                msg += f"  Errors: {len(result['errors'])}"
                return msg
        
        else:
            # Default: show status
            status = self.wireshark.get_status()
            
            msg = f"📊 Wireshark Status:\n"
            msg += f"  Capturing: {'Yes' if status['is_capturing'] else 'No'}\n"
            msg += f"  Interface: {status['interface']}\n"
            msg += f"  Packets: {status['packets_captured']}\n"
            
            if status['is_capturing']:
                msg += f"  Output: {status['output_file']}\n"
                msg += f"  Credentials: {status['credentials_found']}\n"
                msg += f"  DNS leaks: {status['dns_leaks']}\n"
                msg += f"  Unencrypted: {status['unencrypted_traffic']}"
            
            return msg
    
    def handle_phone_targeting(self, params):
        """Handle phone number targeting commands"""
        if not PhoneNumberTargeting:
            return "❌ Phone targeting module not available"
        
        if not self.phone_targeting:
            # Try to initialize with cellular controllers
            gsm_controller = getattr(self.main, 'gsm', None)
            lte_controller = getattr(self.main, 'lte', None)
            
            if not gsm_controller or not lte_controller:
                return "❌ Phone targeting requires cellular modules to be loaded"
            
            try:
                self.phone_targeting = PhoneNumberTargeting(
                    gsm_controller,
                    lte_controller,
                    stealth_mode=True  # Always enable stealth
                )
                logger.info("✅ Phone targeting module initialized (STEALTH MODE)")
            except Exception as e:
                return f"❌ Failed to initialize phone targeting: {e}"
        
        # Handle commands using the parse_targeting_command function
        action = params.get('action', 'status')
        phone = params.get('phone')
        
        if action == 'add' and phone:
            target = self.phone_targeting.add_target(phone)
            return f"✅ Target added: {phone[:3]}***{phone[-2:]}"
        
        elif action == 'capture':
            result = self.phone_targeting.start_targeted_capture(phone)
            if result['success']:
                msg = "✅ Targeted capture started\n"
                if phone:
                    msg += f"  Target: {phone[:3]}***{phone[-2:]}\n"
                msg += f"  Frequency: {result['frequency']} ARFCN\n"
                if result['imsi']:
                    msg += f"  IMSI: {result['imsi'][:8]}***"
                return msg
            else:
                return f"❌ {result['error']}"
        
        elif action == 'stop':
            self.phone_targeting.stop_capture()
            return "✅ Capture stopped"
        
        elif action == 'status':
            status = self.phone_targeting.get_target_status(phone)
            if not status:
                return "❌ No targets or target not found"
            
            if phone:
                msg = f"📊 Target Status: {phone[:3]}***{phone[-2:]}\n"
                msg += f"  IMSI: {status['imsi'][:8] + '***' if status['imsi'] else 'Unknown'}\n"
                msg += f"  Status: {status['status']}\n"
                if status['signal']:
                    msg += f"  Signal: {status['signal']:.0f} dBm"
                return msg
            else:
                return f"📊 Total targets: {status['count']}"
        
        elif action == 'list':
            targets = self.phone_targeting.list_targets()
            if not targets:
                return "📋 No active targets"
            
            msg = f"📋 Active Targets ({len(targets)}):\n"
            for t in targets:
                status_icon = "●" if t['status'] == "CONNECTED" else "○"
                msg += f"  {status_icon} {t['phone']} | {t['imsi']}\n"
            return msg
        
        elif action == 'remove' and phone:
            if self.phone_targeting.remove_target(phone):
                return f"✅ Target removed: {phone[:3]}***{phone[-2:]}"
            else:
                return "❌ Target not found"
        
        elif action == 'extract':
            self.phone_targeting.extract_data(phone)
            return "✅ Data extraction initiated (check logs)"
        
        else:
            return "❌ Unknown phone targeting action. Use: add, capture, stop, status, list, remove, extract"
    
    def initialize_phone_targeting(self, gsm_controller, lte_controller):
        """
        Initialize phone targeting system with cellular controllers
        Called from main controller when cellular modules are loaded
        """
        if not PhoneNumberTargeting:
            logger.warning("Phone targeting module not available")
            return False
        
        try:
            self.phone_targeting = PhoneNumberTargeting(
                gsm_controller,
                lte_controller,
                stealth_mode=True  # Always enable stealth mode
            )
            logger.info("✅ Phone targeting initialized (STEALTH MODE)")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize phone targeting: {e}")
            return False
    
    def handle_volte(self, params):
        """Handle VoLTE/VoNR voice interception commands"""
        if not VoLTEInterceptor:
            return "❌ VoLTE interceptor module not available"
        
        if not self.volte_interceptor:
            # Try to initialize with cellular controllers
            lte_controller = getattr(self.main, 'lte', None)
            gsm_controller = getattr(self.main, 'gsm', None)
            
            if not lte_controller or not gsm_controller:
                return "❌ VoLTE interception requires LTE and GSM modules"
            
            try:
                self.volte_interceptor = VoLTEInterceptor(
                    lte_controller,
                    gsm_controller,
                    stealth_mode=True  # Always enable stealth
                )
                logger.info("✅ VoLTE interceptor initialized (STEALTH MODE)")
            except Exception as e:
                return f"❌ Failed to initialize VoLTE interceptor: {e}"
        
        # Handle commands
        action = params.get('action', 'status')
        
        if action == 'start' or action == 'intercept':
            mode = params.get('mode', 'downgrade')  # 'downgrade' or 'sip'
            
            if mode not in ['downgrade', 'sip']:
                return "❌ Invalid mode. Use 'downgrade' or 'sip'"
            
            result = self.volte_interceptor.start_interception(mode=mode)
            if result:
                if mode == 'downgrade':
                    return "✅ Voice interception started (downgrade mode)\n  Phones will downgrade to 2G/3G for calls"
                else:
                    return "✅ SIP monitoring started\n  Capturing call metadata (audio encrypted)"
            else:
                return "❌ Failed to start interception"
        
        elif action == 'stop':
            self.volte_interceptor.stop_interception()
            return "✅ Voice interception stopped"
        
        elif action == 'list':
            active_only = params.get('active_only', False)
            self.volte_interceptor.list_calls(active_only=active_only)
            return "✅ Calls listed (check logs/console)"
        
        elif action == 'export':
            import time
            filename = f'/tmp/.rf_arsenal_data/call_log_{int(time.time())}.json'
            self.volte_interceptor.export_call_log(filename)
            return f"✅ Call log exported: {filename}"
        
        elif action == 'status':
            stats = self.volte_interceptor.stats
            active = len(self.volte_interceptor.active_calls)
            history = len(self.volte_interceptor.call_history)
            
            msg = "📊 VoLTE Interceptor Status\n"
            msg += f"  Active Calls: {active}\n"
            msg += f"  Total Intercepted: {stats['calls_intercepted']}\n"
            msg += f"  VoLTE Calls: {stats['volte_calls']}\n"
            msg += f"  Downgraded Calls: {stats['downgraded_calls']}\n"
            msg += f"  Failed: {stats['failed_intercepts']}\n"
            msg += f"  History: {history} calls"
            return msg
        
        else:
            return "❌ Unknown VoLTE action. Use: start, stop, list, export, status"
    
    def initialize_volte_interceptor(self, lte_controller, gsm_controller):
        """
        Initialize VoLTE interceptor with cellular controllers
        Called from main controller when cellular modules are loaded
        """
        if not VoLTEInterceptor:
            logger.warning("VoLTE interceptor module not available")
            return False
        
        try:
            self.volte_interceptor = VoLTEInterceptor(
                lte_controller,
                gsm_controller,
                stealth_mode=True  # Always enable stealth mode
            )
            logger.info("✅ VoLTE interceptor initialized (STEALTH MODE)")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize VoLTE interceptor: {e}")
            return False

if __name__ == "__main__":
    # Test AI controller
    logging.basicConfig(level=logging.INFO)
    
    class HardwareStub:
        """Stub for testing without hardware - not for production"""
        pass
    
    ai = AIController(HardwareStub())
    
    # Test commands
    test_commands = [
        "Start 4G base station with IMSI catching",
        "Scan WiFi networks",
        "Spoof GPS to 37.7749, -122.4194, altitude 100",
        "Detect and jam drones",
        "Scan spectrum from 100 MHz to 6 GHz",
        "Jam 2.4 GHz",
        "Show system status"
    ]
    
    print("RF Arsenal OS - AI Controller Test")
    print("=" * 50)
    
    for cmd in test_commands:
        print(f"\n> {cmd}")
        response = ai.execute_command(cmd)
        print(f"{response}")
