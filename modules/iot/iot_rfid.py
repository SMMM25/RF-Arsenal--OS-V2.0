#!/usr/bin/env python3
"""
RF Arsenal OS - IoT and RFID Security Testing Module
Hardware: BladeRF 2.0 micro xA9
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class RFIDConfig:
    """RFID Configuration"""
    frequency: int = 13_560_000  # 13.56 MHz (HF RFID)
    sample_rate: int = 2_000_000  # 2 MSPS
    protocol: str = "iso14443a"   # iso14443a, iso14443b, iso15693, mifare

@dataclass
class IoTConfig:
    """IoT Configuration"""
    frequency: int = 433_000_000  # 433 MHz ISM band
    sample_rate: int = 2_000_000  # 2 MSPS
    protocol: str = "auto"        # auto, zigbee, zwave, lora, ble

@dataclass
class RFIDTag:
    """RFID Tag"""
    uid: str
    tag_type: str
    protocol: str
    data: bytes
    rssi: float
    timestamp: datetime

@dataclass
class IoTDevice:
    """IoT Device"""
    device_id: str
    device_type: str
    protocol: str
    frequency: int
    data: bytes
    rssi: float
    timestamp: datetime

class IoTRFIDSuite:
    """IoT and RFID Security Testing Suite"""
    
    # RFID frequencies
    RFID_FREQUENCIES = {
        'lf': 125_000,           # Low Frequency (125 kHz)
        'hf': 13_560_000,        # High Frequency (13.56 MHz)
        'uhf': 915_000_000,      # Ultra High Frequency (915 MHz)
    }
    
    # IoT protocols and frequencies
    IOT_PROTOCOLS = {
        'zigbee': (2_400_000_000, 2_483_500_000),
        'zwave': (908_420_000, 916_000_000),
        'lora': (433_000_000, 915_000_000),
        'ble': (2_400_000_000, 2_483_500_000),
        'thread': (2_400_000_000, 2_483_500_000),
        'matter': (2_400_000_000, 2_483_500_000),
    }
    
    def __init__(self, hardware_controller):
        """
        Initialize IoT/RFID suite
        
        Args:
            hardware_controller: BladeRF hardware controller instance
        """
        self.hw = hardware_controller
        self.rfid_config = RFIDConfig()
        self.iot_config = IoTConfig()
        self.is_running = False
        self.detected_tags: Dict[str, RFIDTag] = {}
        self.detected_devices: Dict[str, IoTDevice] = {}
        
    def configure_rfid(self, config: RFIDConfig) -> bool:
        """Configure RFID reader"""
        try:
            self.rfid_config = config
            
            # Configure BladeRF for RFID
            if not self.hw.configure_hardware({
                'frequency': config.frequency,
                'sample_rate': config.sample_rate,
                'bandwidth': config.sample_rate,
                'rx_gain': 40,
                'tx_gain': 20
            }):
                logger.error("Failed to configure hardware for RFID")
                return False
                
            logger.info(f"RFID configured: {config.frequency/1e6:.3f} MHz, "
                       f"Protocol: {config.protocol}")
            return True
            
        except Exception as e:
            logger.error(f"RFID configuration error: {e}")
            return False
    
    def configure_iot(self, config: IoTConfig) -> bool:
        """Configure IoT scanner"""
        try:
            self.iot_config = config
            
            # Configure BladeRF for IoT
            if not self.hw.configure_hardware({
                'frequency': config.frequency,
                'sample_rate': config.sample_rate,
                'bandwidth': config.sample_rate,
                'rx_gain': 40,
                'tx_gain': 20
            }):
                logger.error("Failed to configure hardware for IoT")
                return False
                
            logger.info(f"IoT configured: {config.frequency/1e6:.1f} MHz, "
                       f"Protocol: {config.protocol}")
            return True
            
        except Exception as e:
            logger.error(f"IoT configuration error: {e}")
            return False
    
    def scan_rfid_tags(self, duration: float = 5.0) -> List[RFIDTag]:
        """
        Scan for RFID tags
        
        Args:
            duration: Scan duration in seconds
            
        Returns:
            List of detected RFID tags
        """
        try:
            logger.info(f"Scanning for RFID tags ({duration}s)...")
            
            tags = []
            start_time = datetime.now()
            
            while (datetime.now() - start_time).total_seconds() < duration:
                # Send reader query (poll command)
                query = self._generate_rfid_query()
                self.hw.transmit_burst(query)
                
                # Listen for tag responses
                response = self.hw.receive_samples(
                    int(self.rfid_config.sample_rate * 0.1)  # 100ms
                )
                
                if response is None:
                    continue
                
                # Decode tag responses
                detected_tags = self._decode_rfid_response(response)
                tags.extend(detected_tags)
            
            # Remove duplicates
            unique_tags = {}
            for tag in tags:
                if tag.uid not in unique_tags:
                    unique_tags[tag.uid] = tag
                    self.detected_tags[tag.uid] = tag
            
            logger.info(f"Found {len(unique_tags)} RFID tag(s)")
            return list(unique_tags.values())
            
        except Exception as e:
            logger.error(f"RFID scan error: {e}")
            return []
    
    def _generate_rfid_query(self) -> np.ndarray:
        """Generate RFID reader query signal"""
        # ISO 14443A REQA (Request Type A) command
        # Simplified implementation
        
        if self.rfid_config.protocol == "iso14443a":
            # REQA: 0x26 (short frame, 7 bits)
            command = 0x26
            bits = [int(b) for b in format(command, '07b')]
        else:
            # Generic query
            bits = [1, 0, 1, 0, 1, 0, 1, 0]
        
        # Modulate to carrier (ASK modulation)
        samples_per_bit = 128
        query = np.zeros(len(bits) * samples_per_bit, dtype=np.complex64)
        
        for i, bit in enumerate(bits):
            start = i * samples_per_bit
            end = start + samples_per_bit
            
            if bit == 1:
                # 100% modulation
                query[start:end] = 1.0 + 0j
            else:
                # 10% modulation (simplified)
                query[start:end] = 0.1 + 0j
        
        query *= 0.3  # Power
        return query
    
    def _decode_rfid_response(self, samples: np.ndarray) -> List[RFIDTag]:
        """Decode RFID tag responses"""
        tags = []
        
        # Demodulate ASK signal
        envelope = np.abs(samples)
        
        # Detect response (simplified)
        threshold = np.mean(envelope) + 2 * np.std(envelope)
        peaks = np.where(envelope > threshold)[0]
        
        if len(peaks) > 0:
            # Extract UID (simplified)
            # In production, properly decode Manchester encoding
            uid_bytes = self._extract_uid(samples, peaks)
            
            if uid_bytes:
                uid = uid_bytes.hex().upper()
                rssi = 10 * np.log10(np.max(envelope) ** 2)
                
                tag = RFIDTag(
                    uid=uid,
                    tag_type="ISO14443A",
                    protocol=self.rfid_config.protocol,
                    data=uid_bytes,
                    rssi=rssi,
                    timestamp=datetime.now()
                )
                
                tags.append(tag)
        
        return tags
    
    def _extract_uid(self, samples: np.ndarray, peaks: List[int]) -> Optional[bytes]:
        """Extract UID from RFID response"""
        # Simplified UID extraction
        # In production, properly decode the protocol
        
        if len(peaks) < 32:  # Need at least 4 bytes (32 bits)
            return None
        
        # Convert peaks to bits (simplified)
        bits = []
        for i in range(min(32, len(peaks))):
            bits.append(1 if i % 2 == 0 else 0)
        
        # Convert bits to bytes
        uid_bytes = bytearray()
        for i in range(0, len(bits), 8):
            if i + 8 <= len(bits):
                byte_bits = bits[i:i+8]
                byte_val = sum([bit << (7-j) for j, bit in enumerate(byte_bits)])
                uid_bytes.append(byte_val)
        
        return bytes(uid_bytes) if uid_bytes else None
    
    def clone_rfid_tag(self, target_uid: str) -> bool:
        """
        Clone RFID tag
        
        Args:
            target_uid: UID to clone (hex string)
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Cloning RFID tag: {target_uid}")
            
            # Convert UID to bytes
            uid_bytes = bytes.fromhex(target_uid.replace(':', ''))
            
            # Generate tag emulation signal
            emulation = self._generate_tag_emulation(uid_bytes)
            
            # Transmit emulation
            if self.hw.transmit_continuous(emulation):
                logger.info("Tag cloning active")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Tag cloning error: {e}")
            return False
    
    def _generate_tag_emulation(self, uid: bytes) -> np.ndarray:
        """Generate tag emulation signal"""
        # Convert UID to bits
        bits = []
        for byte in uid:
            bits.extend([int(b) for b in format(byte, '08b')])
        
        # Modulate (ASK)
        samples_per_bit = 128
        signal = np.zeros(len(bits) * samples_per_bit, dtype=np.complex64)
        
        for i, bit in enumerate(bits):
            start = i * samples_per_bit
            end = start + samples_per_bit
            signal[start:end] = (1.0 if bit else 0.1) + 0j
        
        # Repeat signal
        signal = np.tile(signal, 10)
        signal *= 0.3
        
        return signal
    
    def scan_iot_devices(self, duration: float = 10.0) -> List[IoTDevice]:
        """
        Scan for IoT devices
        
        Args:
            duration: Scan duration in seconds
            
        Returns:
            List of detected IoT devices
        """
        try:
            logger.info(f"Scanning for IoT devices ({duration}s)...")
            
            devices = []
            
            # Scan common IoT frequencies
            for protocol, freq_range in self.IOT_PROTOCOLS.items():
                if isinstance(freq_range, tuple):
                    scan_freq = freq_range[0]
                else:
                    scan_freq = freq_range
                
                self.iot_config.frequency = scan_freq
                self.configure_iot(self.iot_config)
                
                # Receive samples
                samples = self.hw.receive_samples(
                    int(self.iot_config.sample_rate * 1.0)  # 1 second per frequency
                )
                
                if samples is None:
                    continue
                
                # Detect IoT traffic
                detected = self._detect_iot_traffic(samples, protocol, scan_freq)
                devices.extend(detected)
            
            # Remove duplicates
            unique_devices = {}
            for device in devices:
                if device.device_id not in unique_devices:
                    unique_devices[device.device_id] = device
                    self.detected_devices[device.device_id] = device
            
            logger.info(f"Found {len(unique_devices)} IoT device(s)")
            return list(unique_devices.values())
            
        except Exception as e:
            logger.error(f"IoT scan error: {e}")
            return []
    
    def _detect_iot_traffic(self, samples: np.ndarray, protocol: str,
                           frequency: int) -> List[IoTDevice]:
        """Detect IoT device traffic"""
        devices = []
        
        # Analyze power spectrum
        fft = np.fft.fftshift(np.fft.fft(samples))
        power = np.abs(fft) ** 2
        power_db = 10 * np.log10(power + 1e-12)
        
        # Detect activity
        threshold = np.mean(power_db) + 10
        
        if np.any(power_db > threshold):
            # Device detected
            rssi = np.max(power_db)
            
            # Extract device info (simplified)
            device_id = f"{protocol}_{int(frequency/1e6)}MHz_{len(devices)}"
            
            # Attempt to decode packets
            data = self._decode_iot_packet(samples, protocol)
            
            device = IoTDevice(
                device_id=device_id,
                device_type=protocol.upper(),
                protocol=protocol,
                frequency=frequency,
                data=data,
                rssi=rssi,
                timestamp=datetime.now()
            )
            
            devices.append(device)
        
        return devices
    
    def _decode_iot_packet(self, samples: np.ndarray, protocol: str) -> bytes:
        """Decode IoT packet (simplified)"""
        # Simplified packet decoding
        # In production, implement full protocol decoders
        
        if protocol in ['zigbee', 'thread']:
            return self._decode_zigbee(samples)
        elif protocol == 'zwave':
            return self._decode_zwave(samples)
        elif protocol == 'lora':
            return self._decode_lora(samples)
        else:
            # Generic decode
            magnitude = np.abs(samples)
            data = (magnitude / np.max(magnitude) * 255).astype(np.uint8)
            return data[:64].tobytes()
    
    def _decode_zigbee(self, samples: np.ndarray) -> bytes:
        """Decode ZigBee packet (simplified)"""
        # ZigBee uses O-QPSK modulation
        # Simplified demodulation
        phase = np.angle(samples)
        symbols = np.diff(phase)
        
        # Convert to bytes
        data = ((symbols + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        return data[:32].tobytes()
    
    def _decode_zwave(self, samples: np.ndarray) -> bytes:
        """Decode Z-Wave packet (simplified)"""
        # Z-Wave uses FSK modulation
        freq = np.diff(np.angle(samples))
        
        # Threshold to bits
        threshold = np.median(freq)
        bits = (freq > threshold).astype(np.uint8)
        
        # Convert to bytes
        num_bytes = min(32, len(bits) // 8)
        data = np.packbits(bits[:num_bytes*8])
        return data.tobytes()
    
    def _decode_lora(self, samples: np.ndarray) -> bytes:
        """Decode LoRa packet (simplified)"""
        # LoRa uses chirp spread spectrum
        # Simplified detection
        fft = np.fft.fft(samples)
        magnitude = np.abs(fft)
        
        data = (magnitude / np.max(magnitude) * 255).astype(np.uint8)
        return data[:32].tobytes()
    
    def replay_iot_packet(self, device_id: str) -> bool:
        """
        Replay captured IoT packet
        
        Args:
            device_id: Device ID to replay
            
        Returns:
            True if successful
        """
        try:
            if device_id not in self.detected_devices:
                logger.error(f"Device {device_id} not found")
                return False
            
            device = self.detected_devices[device_id]
            logger.info(f"Replaying packet from {device_id}")
            
            # Configure for device frequency
            self.iot_config.frequency = device.frequency
            self.configure_iot(self.iot_config)
            
            # Modulate packet data
            replay_signal = self._modulate_iot_packet(device.data, device.protocol)
            
            # Transmit
            if self.hw.transmit_burst(replay_signal):
                logger.info("Packet replayed")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Replay error: {e}")
            return False
    
    def _modulate_iot_packet(self, data: bytes, protocol: str) -> np.ndarray:
        """Modulate IoT packet for transmission"""
        # Simplified modulation
        samples_per_byte = 128
        signal = np.zeros(len(data) * samples_per_byte, dtype=np.complex64)
        
        for i, byte in enumerate(data):
            start = i * samples_per_byte
            end = start + samples_per_byte
            
            # Simple FSK modulation
            freq = 50000 if byte > 127 else -50000
            t = np.linspace(0, samples_per_byte / self.iot_config.sample_rate,
                          samples_per_byte, endpoint=False)
            signal[start:end] = np.exp(2j * np.pi * freq * t)
        
        signal *= 0.3
        return signal
    
    def jam_iot_device(self, device_id: str) -> bool:
        """
        Jam specific IoT device
        
        Args:
            device_id: Device ID to jam
            
        Returns:
            True if successful
        """
        try:
            if device_id not in self.detected_devices:
                logger.error(f"Device {device_id} not found")
                return False
            
            device = self.detected_devices[device_id]
            logger.info(f"Jamming {device_id}")
            
            # Configure for device frequency
            self.iot_config.frequency = device.frequency
            self.configure_iot(self.iot_config)
            
            # Generate jamming signal
            num_samples = int(self.iot_config.sample_rate * 0.01)
            jamming = (np.random.randn(num_samples) + 
                      1j * np.random.randn(num_samples)) / np.sqrt(2)
            jamming *= 0.5
            
            # Transmit
            if self.hw.transmit_continuous(jamming):
                self.is_running = True
                logger.info("Jamming active")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Jamming error: {e}")
            return False
    
    def get_detected_tags(self) -> Dict[str, RFIDTag]:
        """Get all detected RFID tags"""
        return self.detected_tags
    
    def get_detected_devices(self) -> Dict[str, IoTDevice]:
        """Get all detected IoT devices"""
        return self.detected_devices
    
    def stop(self):
        """Stop IoT/RFID operations"""
        self.is_running = False
        self.hw.stop_transmission()
        logger.info("IoT/RFID operations stopped")

def main():
    """Test IoT/RFID suite"""
    from core.hardware import HardwareController
    
    # Initialize hardware
    hw = HardwareController()
    if not hw.connect():
        print("Failed to connect to BladeRF")
        return
    
    # Create IoT/RFID suite
    iot_rfid = IoTRFIDSuite(hw)
    
    print("RF Arsenal OS - IoT/RFID Security Suite")
    print("=" * 50)
    
    # RFID scanning
    print("\n1. RFID Tag Scanning")
    rfid_config = RFIDConfig(
        frequency=13_560_000,  # 13.56 MHz
        protocol="iso14443a"
    )
    
    if iot_rfid.configure_rfid(rfid_config):
        print("Scanning for RFID tags (5s)...")
        tags = iot_rfid.scan_rfid_tags(duration=5.0)
        
        print(f"\nFound {len(tags)} tag(s):")
        for tag in tags:
            print(f"  UID: {tag.uid}, Type: {tag.tag_type}, "
                  f"RSSI: {tag.rssi:.1f} dBm")
    
    # IoT scanning
    print("\n2. IoT Device Scanning")
    iot_config = IoTConfig(
        frequency=2_400_000_000,  # 2.4 GHz
        protocol="auto"
    )
    
    if iot_rfid.configure_iot(iot_config):
        print("Scanning for IoT devices (10s)...")
        devices = iot_rfid.scan_iot_devices(duration=10.0)
        
        print(f"\nFound {len(devices)} device(s):")
        for device in devices:
            print(f"  ID: {device.device_id}, Type: {device.device_type}, "
                  f"Freq: {device.frequency/1e6:.1f} MHz, "
                  f"RSSI: {device.rssi:.1f} dBm")
    
    iot_rfid.stop()
    hw.disconnect()

if __name__ == "__main__":
    main()
