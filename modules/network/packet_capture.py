#!/usr/bin/env python3
"""
RF Arsenal OS - Wireshark/PyShark Packet Capture Module
Deep packet inspection, credential extraction, DNS leak detection
"""

import os
import sys
import time
import threading
import subprocess
from datetime import datetime
from pathlib import Path

try:
    import pyshark
except ImportError:
    pyshark = None

class WiresharkCapture:
    """
    Wireshark/PyShark integration for RF Arsenal OS
    Provides packet capture, analysis, and security auditing
    """
    
    def __init__(self, interface='wlan0', output_dir='/tmp/rf_arsenal_captures'):
        self.interface = interface
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.capture = None
        self.capture_thread = None
        self.is_capturing = False
        self.packet_count = 0
        self.current_file = None
        
        # Security configurations
        self.stealth_mode = False
        self.auto_cleanup = True
        self.max_capture_size = 100  # MB
        
        # Detection flags
        self.credentials_found = []
        self.dns_leaks = []
        self.unencrypted_traffic = []
    
    def check_dependencies(self):
        """Verify Wireshark/TShark installation"""
        try:
            result = subprocess.run(['tshark', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return True, result.stdout.split('\n')[0]
        except:
            pass
        
        return False, "TShark not found. Install: sudo apt install tshark"
    
    def start_capture(self, duration=None, packet_limit=None, bpf_filter=None):
        """
        Start packet capture session
        
        Args:
            duration: Capture duration in seconds (None = continuous)
            packet_limit: Stop after N packets
            bpf_filter: Berkeley Packet Filter (e.g., 'tcp port 80')
        """
        if not pyshark:
            return {
                'success': False,
                'error': 'PyShark not installed. Run: pip3 install pyshark'
            }
        
        if self.is_capturing:
            return {
                'success': False,
                'error': 'Capture already in progress'
            }
        
        # Generate output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_file = self.output_dir / f'capture_{timestamp}.pcap'
        
        try:
            # Create capture object
            capture_params = {
                'interface': self.interface,
                'output_file': str(self.current_file)
            }
            
            if bpf_filter:
                capture_params['bpf_filter'] = bpf_filter
            
            if packet_limit:
                capture_params['packet_count'] = packet_limit
            
            self.capture = pyshark.LiveCapture(**capture_params)
            
            # Start capture in background thread
            self.is_capturing = True
            self.packet_count = 0
            
            if duration:
                self.capture_thread = threading.Thread(
                    target=self._capture_with_timeout,
                    args=(duration,)
                )
            else:
                self.capture_thread = threading.Thread(
                    target=self._capture_continuous
                )
            
            self.capture_thread.start()
            
            return {
                'success': True,
                'message': f'Capture started on {self.interface}',
                'output_file': str(self.current_file),
                'filter': bpf_filter or 'None',
                'duration': duration or 'Continuous'
            }
            
        except PermissionError:
            return {
                'success': False,
                'error': 'Permission denied. Run with sudo or configure capabilities'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to start capture: {str(e)}'
            }
    
    def _capture_continuous(self):
        """Continuous capture (background thread)"""
        try:
            for packet in self.capture.sniff_continuously():
                if not self.is_capturing:
                    break
                self.packet_count += 1
                self._analyze_packet(packet)
        except Exception as e:
            print(f"[ERROR] Capture thread error: {e}", file=sys.stderr)
        finally:
            self.is_capturing = False
    
    def _capture_with_timeout(self, duration):
        """Timed capture (background thread)"""
        try:
            start_time = time.time()
            for packet in self.capture.sniff_continuously():
                if time.time() - start_time >= duration:
                    break
                self.packet_count += 1
                self._analyze_packet(packet)
        except Exception as e:
            print(f"[ERROR] Capture thread error: {e}", file=sys.stderr)
        finally:
            self.is_capturing = False
            self.capture.close()
    
    def stop_capture(self):
        """Stop active packet capture"""
        if not self.is_capturing:
            return {
                'success': False,
                'error': 'No capture in progress'
            }
        
        self.is_capturing = False
        
        # Wait for capture thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5)
        
        if self.capture:
            try:
                self.capture.close()
            except:
                pass
        
        return {
            'success': True,
            'message': 'Capture stopped',
            'packets_captured': self.packet_count,
            'output_file': str(self.current_file),
            'credentials_found': len(self.credentials_found),
            'dns_leaks': len(self.dns_leaks),
            'unencrypted_traffic': len(self.unencrypted_traffic)
        }
    
    def _analyze_packet(self, packet):
        """Real-time packet analysis"""
        try:
            # Check for unencrypted credentials
            if hasattr(packet, 'http'):
                self._check_http_credentials(packet)
            
            # Check for DNS leaks
            if hasattr(packet, 'dns'):
                self._check_dns_leak(packet)
            
            # Check for unencrypted traffic
            if hasattr(packet, 'tcp'):
                self._check_unencrypted_tcp(packet)
                
        except Exception as e:
            # Silent failures in analysis shouldn't stop capture
            pass
    
    def _check_http_credentials(self, packet):
        """Detect HTTP authentication attempts"""
        try:
            if hasattr(packet.http, 'authorization'):
                self.credentials_found.append({
                    'timestamp': packet.sniff_time,
                    'type': 'HTTP Auth',
                    'data': packet.http.authorization
                })
            
            # Check POST data for common credential fields
            if hasattr(packet.http, 'file_data'):
                data = packet.http.file_data.lower()
                if any(term in data for term in ['password', 'passwd', 'pwd', 'login']):
                    self.credentials_found.append({
                        'timestamp': packet.sniff_time,
                        'type': 'HTTP POST',
                        'data': '[Credentials detected in POST data]'
                    })
        except:
            pass
    
    def _check_dns_leak(self, packet):
        """Detect DNS queries (potential privacy leaks)"""
        try:
            if hasattr(packet.dns, 'qry_name'):
                query = packet.dns.qry_name
                
                # Flag suspicious or privacy-sensitive queries
                sensitive_domains = ['google', 'facebook', 'amazon', 'microsoft']
                if any(domain in query.lower() for domain in sensitive_domains):
                    self.dns_leaks.append({
                        'timestamp': packet.sniff_time,
                        'query': query,
                        'type': 'Potentially sensitive'
                    })
        except:
            pass
    
    def _check_unencrypted_tcp(self, packet):
        """Flag unencrypted TCP traffic on common ports"""
        try:
            if hasattr(packet.tcp, 'dstport'):
                port = int(packet.tcp.dstport)
                
                # Unencrypted ports
                if port in [21, 23, 80, 110, 143]:  # FTP, Telnet, HTTP, POP3, IMAP
                    self.unencrypted_traffic.append({
                        'timestamp': packet.sniff_time,
                        'port': port,
                        'dst': packet.ip.dst if hasattr(packet, 'ip') else 'Unknown'
                    })
        except:
            pass
    
    def analyze_pcap(self, pcap_file):
        """
        Analyze existing PCAP file
        
        Returns comprehensive security analysis
        """
        if not pyshark:
            return {'success': False, 'error': 'PyShark not installed'}
        
        try:
            cap = pyshark.FileCapture(pcap_file)
            
            stats = {
                'total_packets': 0,
                'protocols': {},
                'credentials': [],
                'dns_queries': [],
                'unencrypted': [],
                'suspicious': []
            }
            
            for packet in cap:
                stats['total_packets'] += 1
                
                # Protocol statistics
                if hasattr(packet, 'highest_layer'):
                    protocol = packet.highest_layer
                    stats['protocols'][protocol] = stats['protocols'].get(protocol, 0) + 1
                
                # Security checks
                if hasattr(packet, 'http'):
                    if hasattr(packet.http, 'authorization'):
                        stats['credentials'].append({
                            'type': 'HTTP Auth',
                            'time': packet.sniff_time
                        })
                
                if hasattr(packet, 'dns') and hasattr(packet.dns, 'qry_name'):
                    stats['dns_queries'].append(packet.dns.qry_name)
                
                if hasattr(packet, 'tcp') and hasattr(packet.tcp, 'dstport'):
                    port = int(packet.tcp.dstport)
                    if port in [21, 23, 80]:
                        stats['unencrypted'].append({
                            'port': port,
                            'time': packet.sniff_time
                        })
            
            cap.close()
            
            return {
                'success': True,
                'file': pcap_file,
                'statistics': stats
            }
            
        except FileNotFoundError:
            return {'success': False, 'error': f'File not found: {pcap_file}'}
        except Exception as e:
            return {'success': False, 'error': f'Analysis failed: {str(e)}'}
    
    def check_leaks(self):
        """
        Return current session security findings
        """
        return {
            'credentials_detected': len(self.credentials_found),
            'credentials': self.credentials_found[-10:],  # Last 10
            'dns_leaks': len(self.dns_leaks),
            'dns_queries': self.dns_leaks[-20:],  # Last 20
            'unencrypted_traffic': len(self.unencrypted_traffic),
            'unencrypted': self.unencrypted_traffic[-10:]
        }
    
    def cleanup(self, secure_delete=True):
        """
        Clean up capture files
        
        Args:
            secure_delete: Use shred for secure deletion
        """
        if self.is_capturing:
            self.stop_capture()
        
        deleted = []
        errors = []
        
        for pcap_file in self.output_dir.glob('*.pcap'):
            try:
                if secure_delete and os.path.exists('/usr/bin/shred'):
                    subprocess.run(['shred', '-vfz', '-n', '3', str(pcap_file)],
                                 check=True, capture_output=True)
                else:
                    pcap_file.unlink()
                
                deleted.append(str(pcap_file))
            except Exception as e:
                errors.append(f"{pcap_file}: {str(e)}")
        
        return {
            'success': len(errors) == 0,
            'deleted': deleted,
            'errors': errors
        }
    
    def get_status(self):
        """Return current capture status"""
        return {
            'is_capturing': self.is_capturing,
            'interface': self.interface,
            'packets_captured': self.packet_count,
            'output_file': str(self.current_file) if self.current_file else None,
            'credentials_found': len(self.credentials_found),
            'dns_leaks': len(self.dns_leaks),
            'unencrypted_traffic': len(self.unencrypted_traffic)
        }


# CLI Interface
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='RF Arsenal OS - Packet Capture Module')
    parser.add_argument('--interface', '-i', default='wlan0', help='Network interface')
    parser.add_argument('--duration', '-d', type=int, help='Capture duration (seconds)')
    parser.add_argument('--filter', '-f', help='BPF filter (e.g., "tcp port 80")')
    parser.add_argument('--analyze', '-a', help='Analyze existing PCAP file')
    parser.add_argument('--cleanup', action='store_true', help='Clean up capture files')
    
    args = parser.parse_args()
    
    capture = WiresharkCapture(interface=args.interface)
    
    # Check dependencies
    installed, version = capture.check_dependencies()
    print(f"[*] TShark: {'✓' if installed else '✗'} {version if installed else ''}")
    
    if args.analyze:
        print(f"\n[*] Analyzing {args.analyze}...")
        result = capture.analyze_pcap(args.analyze)
        if result['success']:
            stats = result['statistics']
            print(f"\n[+] Total packets: {stats['total_packets']}")
            print(f"[+] Protocols: {stats['protocols']}")
            print(f"[!] Credentials found: {len(stats['credentials'])}")
            print(f"[!] DNS queries: {len(stats['dns_queries'])}")
            print(f"[!] Unencrypted traffic: {len(stats['unencrypted'])}")
        else:
            print(f"[-] Error: {result['error']}")
    
    elif args.cleanup:
        print("[*] Cleaning up capture files...")
        result = capture.cleanup(secure_delete=True)
        print(f"[+] Deleted: {len(result['deleted'])} files")
        if result['errors']:
            print(f"[-] Errors: {result['errors']}")
    
    else:
        print(f"\n[*] Starting capture on {args.interface}")
        result = capture.start_capture(duration=args.duration, bpf_filter=args.filter)
        
        if result['success']:
            print(f"[+] {result['message']}")
            print(f"[+] Output: {result['output_file']}")
            print(f"[+] Filter: {result['filter']}")
            
            if args.duration:
                print(f"\n[*] Capturing for {args.duration} seconds...")
                time.sleep(args.duration + 1)
                result = capture.stop_capture()
                print(f"\n[+] Capture complete: {result['packets_captured']} packets")
            else:
                print("\n[*] Press Ctrl+C to stop capture...")
                try:
                    while capture.is_capturing:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n[*] Stopping capture...")
                    result = capture.stop_capture()
                    print(f"[+] Captured {result['packets_captured']} packets")
            
            # Show security findings
            leaks = capture.check_leaks()
            print(f"\n[!] Security Findings:")
            print(f"    Credentials: {leaks['credentials_detected']}")
            print(f"    DNS Leaks: {leaks['dns_leaks']}")
            print(f"    Unencrypted: {leaks['unencrypted_traffic']}")
        else:
            print(f"[-] Error: {result['error']}")
