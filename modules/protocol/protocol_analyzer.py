#!/usr/bin/env python3
"""
RF Arsenal OS - Wireless Protocol Analyzer
Hardware: BladeRF 2.0 micro xA9
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ProtocolConfig:
    """Protocol Analyzer Configuration"""
    frequency: int = 2_400_000_000  # 2.4 GHz
    sample_rate: int = 20_000_000   # 20 MSPS
    bandwidth: int = 20_000_000     # 20 MHz
    protocol: str = "auto"          # auto, bluetooth, zigbee, wifi, lora, etc.

@dataclass
class Packet:
    """Decoded Packet"""
    timestamp: datetime
    protocol: str
    packet_type: str
    source: str
    destination: str
    length: int
    data: bytes
    rssi: float
    metadata: Dict

class ProtocolAnalyzer:
    """Multi-Protocol Wireless Analyzer"""
    
    # Protocol signatures and parameters
    PROTOCOLS = {
        'bluetooth': {
            'frequency': (2_402_000_000, 2_480_000_000),
            'channel_width': 1_000_000,
            'num_channels': 79,
            'modulation': 'GFSK',
            'preamble': b'\xaa\xaa',
        },
        'ble': {
            'frequency': (2_402_000_000, 2_480_000_000),
            'channel_width': 2_000_000,
            'num_channels': 40,
            'modulation': 'GFSK',
            'preamble': b'\xaa',
        },
        'zigbee': {
            'frequency': (2_405_000_000, 2_480_000_000),
            'channel_width': 2_000_000,
            'num_channels': 16,
            'modulation': 'O-QPSK',
            'preamble': b'\x00\x00\x00\x00',
        },
        'wifi': {
            'frequency': (2_412_000_000, 2_484_000_000),
            'channel_width': 20_000_000,
            'num_channels': 14,
            'modulation': 'OFDM',
            'preamble': b'\x00\x00',
        },
        'zwave': {
            'frequency': (908_420_000, 916_000_000),
            'channel_width': 40_000,
            'num_channels': 3,
            'modulation': 'FSK',
            'preamble': b'\xf0\xf0',
        },
        'lora': {
            'frequency': (902_000_000, 928_000_000),
            'channel_width': 125_000,
            'num_channels': 64,
            'modulation': 'CSS',
            'preamble': None,
        },
    }
    
    def __init__(self, hardware_controller):
        """
        Initialize protocol analyzer
        
        Args:
            hardware_controller: BladeRF hardware controller instance
        """
        self.hw = hardware_controller
        self.config = ProtocolConfig()
        self.is_running = False
        self.captured_packets: List[Packet] = []
        self.protocol_stats: Dict[str, int] = defaultdict(int)
        
    def configure(self, config: ProtocolConfig) -> bool:
        """Configure protocol analyzer"""
        try:
            self.config = config
            
            # Configure BladeRF
            if not self.hw.configure_hardware({
                'frequency': config.frequency,
                'sample_rate': config.sample_rate,
                'bandwidth': config.bandwidth,
                'rx_gain': 40,
                'tx_gain': 0
            }):
                logger.error("Failed to configure hardware")
                return False
                
            logger.info(f"Protocol analyzer configured: {config.frequency/1e6:.1f} MHz, "
                       f"Protocol: {config.protocol}")
            return True
            
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            return False
    
    def capture(self, duration: float = 10.0) -> List[Packet]:
        """
        Capture and analyze wireless packets
        
        Args:
            duration: Capture duration in seconds
            
        Returns:
            List of captured packets
        """
        try:
            logger.info(f"Capturing packets for {duration}s...")
            
            packets = []
            start_time = datetime.now()
            
            while (datetime.now() - start_time).total_seconds() < duration:
                # Receive samples
                samples = self.hw.receive_samples(
                    int(self.config.sample_rate * 0.1)  # 100ms
                )
                
                if samples is None:
                    continue
                
                # Detect and decode packets
                detected_packets = self._analyze_samples(samples)
                packets.extend(detected_packets)
                
                # Update statistics
                for packet in detected_packets:
                    self.protocol_stats[packet.protocol] += 1
            
            self.captured_packets.extend(packets)
            logger.info(f"Captured {len(packets)} packet(s)")
            return packets
            
        except Exception as e:
            logger.error(f"Capture error: {e}")
            return []
    
    def _analyze_samples(self, samples: np.ndarray) -> List[Packet]:
        """Analyze samples and decode packets"""
        packets = []
        
        # Detect protocol if auto mode
        if self.config.protocol == "auto":
            protocol = self._detect_protocol(samples)
        else:
            protocol = self.config.protocol
        
        if protocol:
            # Decode based on protocol
            decoded = self._decode_protocol(samples, protocol)
            packets.extend(decoded)
        
        return packets
    
    def _detect_protocol(self, samples: np.ndarray) -> Optional[str]:
        """Auto-detect protocol from signal characteristics"""
        # Analyze signal characteristics
        fft = np.fft.fftshift(np.fft.fft(samples))
        power = np.abs(fft) ** 2
        
        # Estimate bandwidth
        power_db = 10 * np.log10(power + 1e-12)
        threshold = np.max(power_db) - 20
        occupied = np.sum(power_db > threshold)
        bandwidth = occupied * self.config.sample_rate / len(power)
        
        # Check frequency
        freq = self.config.frequency
        
        # Match against known protocols
        for protocol, params in self.PROTOCOLS.items():
            freq_range = params['frequency']
            if freq_range[0] <= freq <= freq_range[1]:
                # Check bandwidth match
                expected_bw = params['channel_width']
                if abs(bandwidth - expected_bw) < expected_bw * 0.5:
                    logger.debug(f"Detected protocol: {protocol}")
                    return protocol
        
        return None
    
    def _decode_protocol(self, samples: np.ndarray, protocol: str) -> List[Packet]:
        """Decode packets for specific protocol"""
        if protocol == 'bluetooth':
            return self._decode_bluetooth(samples)
        elif protocol == 'ble':
            return self._decode_ble(samples)
        elif protocol == 'zigbee':
            return self._decode_zigbee(samples)
        elif protocol == 'wifi':
            return self._decode_wifi(samples)
        elif protocol == 'zwave':
            return self._decode_zwave(samples)
        elif protocol == 'lora':
            return self._decode_lora(samples)
        else:
            return []
    
    def _decode_bluetooth(self, samples: np.ndarray) -> List[Packet]:
        """Decode Bluetooth Classic packets"""
        packets = []
        
        # GFSK demodulation
        freq = np.diff(np.angle(samples))
        
        # Find preamble
        preamble = self.PROTOCOLS['bluetooth']['preamble']
        preamble_bits = self._bytes_to_bits(preamble)
        
        # Threshold to bits
        bits = (freq > 0).astype(int)
        
        # Search for preamble
        for i in range(len(bits) - len(preamble_bits)):
            if np.array_equal(bits[i:i+len(preamble_bits)], preamble_bits):
                # Found preamble, extract packet
                packet_bits = bits[i:i+1000]  # Extract up to 1000 bits
                
                # Decode packet
                packet_bytes = self._bits_to_bytes(packet_bits)
                
                if len(packet_bytes) >= 10:
                    # Parse Bluetooth packet header
                    access_code = packet_bytes[0:4]
                    header = packet_bytes[4:5]
                    payload = packet_bytes[5:]
                    
                    # Calculate RSSI
                    rssi = 10 * np.log10(np.mean(np.abs(samples) ** 2))
                    
                    packet = Packet(
                        timestamp=datetime.now(),
                        protocol='Bluetooth',
                        packet_type='DATA',
                        source=access_code.hex(),
                        destination='unknown',
                        length=len(payload),
                        data=payload,
                        rssi=rssi,
                        metadata={'header': header.hex()}
                    )
                    
                    packets.append(packet)
        
        return packets
    
    def _decode_ble(self, samples: np.ndarray) -> List[Packet]:
        """Decode Bluetooth Low Energy packets"""
        packets = []
        
        # GFSK demodulation
        freq = np.diff(np.angle(samples))
        bits = (freq > 0).astype(int)
        
        # BLE advertising channel packet structure
        # Preamble (1 byte) + Access Address (4 bytes) + PDU + CRC (3 bytes)
        
        # Find BLE advertising access address: 0x8E89BED6
        access_address = [1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 
                         1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
        
        for i in range(len(bits) - len(access_address)):
            # Check for access address match (with tolerance)
            match = np.sum(bits[i:i+len(access_address)] == access_address)
            if match > 28:  # Allow some bit errors
                # Extract PDU
                pdu_start = i + len(access_address)
                pdu = bits[pdu_start:pdu_start+320]  # Max BLE payload
                
                pdu_bytes = self._bits_to_bytes(pdu)
                
                if len(pdu_bytes) >= 2:
                    pdu_header = pdu_bytes[0]
                    pdu_length = pdu_bytes[1]
                    payload = pdu_bytes[2:2+pdu_length]
                    
                    rssi = 10 * np.log10(np.mean(np.abs(samples) ** 2))
                    
                    packet = Packet(
                        timestamp=datetime.now(),
                        protocol='BLE',
                        packet_type='ADV' if (pdu_header & 0x0F) < 7 else 'DATA',
                        source='advertising',
                        destination='broadcast',
                        length=pdu_length,
                        data=payload,
                        rssi=rssi,
                        metadata={'pdu_type': pdu_header & 0x0F}
                    )
                    
                    packets.append(packet)
        
        return packets
    
    def _decode_zigbee(self, samples: np.ndarray) -> List[Packet]:
        """Decode ZigBee packets (IEEE 802.15.4)"""
        packets = []
        
        # O-QPSK demodulation (simplified)
        phase = np.angle(samples)
        symbols = np.diff(phase)
        
        # ZigBee uses chip sequences
        # Simplified: threshold to bits
        bits = (symbols > 0).astype(int)
        
        # Find preamble (4 zero bytes)
        preamble_bits = [0] * 32
        
        for i in range(len(bits) - 32):
            if np.sum(bits[i:i+32]) == 0:  # All zeros
                # Found preamble
                # SFD (Start of Frame Delimiter): 0xA7
                sfd_start = i + 32
                packet_start = sfd_start + 8
                
                # Extract frame
                frame_bits = bits[packet_start:packet_start+1024]
                frame_bytes = self._bits_to_bytes(frame_bits)
                
                if len(frame_bytes) >= 5:
                    # Parse 802.15.4 frame
                    frame_length = frame_bytes[0]
                    fcf = frame_bytes[1:3]  # Frame Control Field
                    seq_num = frame_bytes[3]
                    
                    rssi = 10 * np.log10(np.mean(np.abs(samples) ** 2))
                    
                    packet = Packet(
                        timestamp=datetime.now(),
                        protocol='ZigBee',
                        packet_type='DATA',
                        source='unknown',
                        destination='unknown',
                        length=frame_length,
                        data=frame_bytes[4:],
                        rssi=rssi,
                        metadata={'fcf': fcf.hex(), 'seq': seq_num}
                    )
                    
                    packets.append(packet)
        
        return packets
    
    def _decode_wifi(self, samples: np.ndarray) -> List[Packet]:
        """Decode WiFi (802.11) packets"""
        packets = []
        
        # Simplified WiFi detection
        # Look for high power bursts (packets)
        power = np.abs(samples) ** 2
        threshold = np.mean(power) + 5 * np.std(power)
        
        # Find packet starts
        packet_starts = []
        in_packet = False
        for i in range(len(power)):
            if power[i] > threshold and not in_packet:
                packet_starts.append(i)
                in_packet = True
            elif power[i] < threshold:
                in_packet = False
        
        for start in packet_starts[:10]:  # Limit to 10 packets
            # Extract packet region
            end = min(start + 10000, len(samples))
            packet_samples = samples[start:end]
            
            # Simplified: just extract magnitude
            packet_data = (np.abs(packet_samples) * 255).astype(np.uint8)
            
            rssi = 10 * np.log10(np.mean(power[start:end]))
            
            packet = Packet(
                timestamp=datetime.now(),
                protocol='WiFi',
                packet_type='802.11',
                source='unknown',
                destination='unknown',
                length=len(packet_data),
                data=packet_data.tobytes()[:100],  # First 100 bytes
                rssi=rssi,
                metadata={}
            )
            
            packets.append(packet)
        
        return packets
    
    def _decode_zwave(self, samples: np.ndarray) -> List[Packet]:
        """Decode Z-Wave packets"""
        packets = []
        
        # FSK demodulation
        freq = np.diff(np.angle(samples))
        bits = (freq > np.median(freq)).astype(int)
        
        # Z-Wave preamble: 0xF0F0
        preamble_bits = self._bytes_to_bits(b'\xf0\xf0')
        
        for i in range(len(bits) - len(preamble_bits)):
            if np.array_equal(bits[i:i+len(preamble_bits)], preamble_bits):
                # Extract packet
                packet_bits = bits[i:i+500]
                packet_bytes = self._bits_to_bytes(packet_bits)
                
                if len(packet_bytes) >= 5:
                    home_id = packet_bytes[2:6]
                    
                    rssi = 10 * np.log10(np.mean(np.abs(samples) ** 2))
                    
                    packet = Packet(
                        timestamp=datetime.now(),
                        protocol='Z-Wave',
                        packet_type='DATA',
                        source=home_id.hex(),
                        destination='unknown',
                        length=len(packet_bytes),
                        data=packet_bytes[6:],
                        rssi=rssi,
                        metadata={'home_id': home_id.hex()}
                    )
                    
                    packets.append(packet)
        
        return packets
    
    def _decode_lora(self, samples: np.ndarray) -> List[Packet]:
        """Decode LoRa packets"""
        packets = []
        
        # LoRa uses chirp spread spectrum
        # Simplified detection: look for chirp patterns
        fft = np.fft.fft(samples)
        magnitude = np.abs(fft)
        
        # Detect chirp (increasing frequency over time)
        chirp_detected = False
        for i in range(len(magnitude) - 100):
            if np.mean(magnitude[i:i+100]) > np.mean(magnitude) * 2:
                chirp_detected = True
                break
        
        if chirp_detected:
            # Extract payload (simplified)
            packet_data = (magnitude[:100] / np.max(magnitude) * 255).astype(np.uint8)
            
            rssi = 10 * np.log10(np.mean(np.abs(samples) ** 2))
            
            packet = Packet(
                timestamp=datetime.now(),
                protocol='LoRa',
                packet_type='DATA',
                source='unknown',
                destination='unknown',
                length=len(packet_data),
                data=packet_data.tobytes(),
                rssi=rssi,
                metadata={}
            )
            
            packets.append(packet)
        
        return packets
    
    def _bytes_to_bits(self, data: bytes) -> List[int]:
        """Convert bytes to bits"""
        bits = []
        for byte in data:
            bits.extend([int(b) for b in format(byte, '08b')])
        return bits
    
    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        """Convert bits to bytes"""
        num_bytes = len(bits) // 8
        result = bytearray()
        for i in range(num_bytes):
            byte_bits = bits[i*8:(i+1)*8]
            byte_val = sum([bit << (7-j) for j, bit in enumerate(byte_bits)])
            result.append(byte_val)
        return bytes(result)
    
    def get_statistics(self) -> Dict:
        """Get capture statistics"""
        stats = {
            'total_packets': len(self.captured_packets),
            'by_protocol': dict(self.protocol_stats),
            'by_type': defaultdict(int)
        }
        
        for packet in self.captured_packets:
            stats['by_type'][packet.packet_type] += 1
        
        stats['by_type'] = dict(stats['by_type'])
        
        return stats
    
    def filter_packets(self, protocol: Optional[str] = None,
                      source: Optional[str] = None) -> List[Packet]:
        """Filter captured packets"""
        filtered = self.captured_packets
        
        if protocol:
            filtered = [p for p in filtered if p.protocol.lower() == protocol.lower()]
        
        if source:
            filtered = [p for p in filtered if p.source == source]
        
        return filtered
    
    def export_pcap(self, filename: str = "capture.pcap") -> bool:
        """Export packets to PCAP format (simplified)"""
        try:
            # Simplified PCAP export
            # In production, use proper PCAP library
            
            with open(filename, 'wb') as f:
                # PCAP global header
                f.write(b'\xd4\xc3\xb2\xa1')  # Magic number
                f.write(b'\x02\x00\x04\x00')  # Version
                f.write(b'\x00\x00\x00\x00')  # Timezone
                f.write(b'\x00\x00\x00\x00')  # Sigfigs
                f.write(b'\xff\xff\x00\x00')  # Snaplen
                f.write(b'\x01\x00\x00\x00')  # Network (Ethernet)
                
                # Packet records
                for packet in self.captured_packets:
                    ts = int(packet.timestamp.timestamp())
                    ts_usec = int((packet.timestamp.timestamp() % 1) * 1e6)
                    
                    # Packet header
                    f.write(ts.to_bytes(4, 'little'))
                    f.write(ts_usec.to_bytes(4, 'little'))
                    f.write(len(packet.data).to_bytes(4, 'little'))
                    f.write(len(packet.data).to_bytes(4, 'little'))
                    
                    # Packet data
                    f.write(packet.data)
            
            logger.info(f"Exported {len(self.captured_packets)} packets to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Export error: {e}")
            return False
    
    def get_captured_packets(self) -> List[Packet]:
        """Get all captured packets"""
        return self.captured_packets
    
    def stop(self):
        """Stop protocol analyzer"""
        self.is_running = False
        logger.info("Protocol analyzer stopped")

def main():
    """Test protocol analyzer"""
    from core.hardware import HardwareController
    
    # Initialize hardware
    hw = HardwareController()
    if not hw.connect():
        print("Failed to connect to BladeRF")
        return
    
    # Create protocol analyzer
    analyzer = ProtocolAnalyzer(hw)
    
    # Configure for 2.4 GHz (Bluetooth/WiFi/ZigBee)
    config = ProtocolConfig(
        frequency=2_440_000_000,  # 2.44 GHz
        protocol="auto"
    )
    
    if not analyzer.configure(config):
        print("Configuration failed")
        return
    
    print("RF Arsenal OS - Protocol Analyzer")
    print("=" * 50)
    
    # Capture packets
    print("\nCapturing packets for 10 seconds...")
    packets = analyzer.capture(duration=10.0)
    
    print(f"\nCaptured {len(packets)} packet(s):")
    for i, packet in enumerate(packets[:10], 1):
        print(f"{i}. {packet.protocol} - {packet.packet_type} - "
              f"{packet.length} bytes - RSSI: {packet.rssi:.1f} dBm")
    
    # Statistics
    stats = analyzer.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total packets: {stats['total_packets']}")
    print(f"  By protocol: {stats['by_protocol']}")
    
    # Export
    analyzer.export_pcap("rf_capture.pcap")
    
    analyzer.stop()
    hw.disconnect()

if __name__ == "__main__":
    main()
