"""
RF Arsenal OS - DECT Protocol Decoder
Digital Enhanced Cordless Telecommunications decoder for cordless phones.
Frequency range: 1880-1900 MHz (Europe), 1920-1930 MHz (US)
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque


class DECTBand(Enum):
    EUROPE = "europe"
    US = "us"
    JAPAN = "japan"


class DECTSlotType(Enum):
    TRAFFIC = "traffic"
    CONTROL = "control"
    IDLE = "idle"


@dataclass
class DECTChannel:
    carrier: int
    frequency_hz: float
    slot: int
    slot_type: DECTSlotType
    rssi_dbm: float
    active: bool = False


@dataclass
class DECTFrame:
    frame_id: str
    timestamp: float
    channel: DECTChannel
    header: int
    tail: bytes
    crc: int
    payload: bytes
    rfpi: Optional[str] = None
    ipui: Optional[str] = None
    frame_type: str = "unknown"
    
    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "rfpi": self.rfpi,
            "ipui": self.ipui,
            "frame_type": self.frame_type,
            "channel": self.channel.carrier,
            "rssi_dbm": self.channel.rssi_dbm
        }


class DECTDecoder:
    """DECT protocol decoder for cordless phone analysis."""
    
    BANDS = {
        DECTBand.EUROPE: {"start": 1880e6, "end": 1900e6, "carriers": 10},
        DECTBand.US: {"start": 1920e6, "end": 1930e6, "carriers": 5},
    }
    CARRIER_SPACING = 1.728e6
    SYNC_WORD = 0xE98A
    
    def __init__(self, band: DECTBand = DECTBand.EUROPE, sample_rate: float = 2e6):
        self.band = band
        self.sample_rate = sample_rate
        band_config = self.BANDS.get(band, self.BANDS[DECTBand.EUROPE])
        self.freq_start = band_config["start"]
        self.num_carriers = band_config["carriers"]
        self._frames: deque = deque(maxlen=10000)
        self._base_stations: Dict[str, Dict] = {}
        self._handsets: Dict[str, Dict] = {}
        self._stats = {"frames_decoded": 0, "crc_errors": 0, "active_channels": 0}
        
    def get_carrier_frequency(self, carrier: int) -> float:
        return self.freq_start + (carrier * self.CARRIER_SPACING)
        
    def scan_channels(self, iq_samples: np.ndarray, center_freq: float) -> List[DECTChannel]:
        active = []
        fft_size = min(4096, len(iq_samples))
        spectrum = np.fft.fftshift(np.abs(np.fft.fft(iq_samples[:fft_size])))
        freq_bins = np.fft.fftshift(np.fft.fftfreq(fft_size, 1/self.sample_rate)) + center_freq
        
        for carrier in range(self.num_carriers):
            carrier_freq = self.get_carrier_frequency(carrier)
            if freq_bins[0] <= carrier_freq <= freq_bins[-1]:
                idx = np.argmin(np.abs(freq_bins - carrier_freq))
                power = np.mean(spectrum[max(0,idx-10):min(len(spectrum),idx+10)]**2)
                power_dbm = 10 * np.log10(power + 1e-20) - 30
                if power_dbm > -80:
                    active.append(DECTChannel(carrier, carrier_freq, 0, DECTSlotType.TRAFFIC, power_dbm, True))
        
        self._stats["active_channels"] = len(active)
        return active
        
    def decode_frame(self, iq_samples: np.ndarray, channel: DECTChannel) -> Optional[DECTFrame]:
        bits = self._demodulate_gfsk(iq_samples)
        if len(bits) < 480:
            return None
            
        sync_pos = self._find_sync(bits)
        if sync_pos < 0:
            return None
            
        frame_bits = bits[sync_pos:]
        if len(frame_bits) < 384:
            return None
            
        header = self._bits_to_int(frame_bits[:8])
        tail = bytes([self._bits_to_int(frame_bits[8+i*8:16+i*8]) for i in range(5)])
        crc = self._bits_to_int(frame_bits[48:64])
        payload = bytes([self._bits_to_int(frame_bits[64+i*8:72+i*8]) for i in range(40)])
        
        rfpi = tail.hex().upper() if len(tail) >= 5 else None
        ipui = payload[:8].hex().upper() if len(payload) >= 8 else None
        frame_type = self._classify_frame(header)
        
        frame = DECTFrame(f"dect_{int(time.time()*1000)}", time.time(), channel, header, tail, crc, payload, rfpi, ipui, frame_type)
        self._frames.append(frame)
        self._stats["frames_decoded"] += 1
        
        if rfpi:
            self._update_base_station(rfpi, frame)
        if ipui:
            self._update_handset(ipui, frame)
        return frame
        
    def _demodulate_gfsk(self, iq: np.ndarray) -> np.ndarray:
        phase = np.angle(iq)
        freq = np.diff(np.unwrap(phase))
        sps = int(self.sample_rate / 1.152e6)
        symbols = freq[sps//2::sps] if sps > 0 else freq
        return (symbols > 0).astype(int)
        
    def _find_sync(self, bits: np.ndarray) -> int:
        sync_bits = np.array([(self.SYNC_WORD >> (15-i)) & 1 for i in range(16)])
        for i in range(len(bits) - 16):
            if np.array_equal(bits[i:i+16], sync_bits):
                return i + 16
        return -1
        
    def _bits_to_int(self, bits: np.ndarray) -> int:
        result = 0
        for bit in bits:
            result = (result << 1) | int(bit)
        return result
        
    def _classify_frame(self, header: int) -> str:
        ta = (header >> 4) & 0x07
        types = {0: "ct", 1: "nt", 2: "nt", 3: "qt", 4: "pt_escape", 5: "mt_first", 6: "mt_last", 7: "mt_single"}
        return types.get(ta, "unknown")
        
    def _update_base_station(self, rfpi: str, frame: DECTFrame) -> None:
        if rfpi not in self._base_stations:
            self._base_stations[rfpi] = {"rfpi": rfpi, "first_seen": time.time(), "frame_count": 0}
        self._base_stations[rfpi]["last_seen"] = time.time()
        self._base_stations[rfpi]["frame_count"] += 1
        
    def _update_handset(self, ipui: str, frame: DECTFrame) -> None:
        if ipui not in self._handsets:
            self._handsets[ipui] = {"ipui": ipui, "first_seen": time.time(), "frame_count": 0}
        self._handsets[ipui]["last_seen"] = time.time()
        self._handsets[ipui]["frame_count"] += 1
        
    def get_base_stations(self) -> List[Dict]:
        return list(self._base_stations.values())
        
    def get_handsets(self) -> List[Dict]:
        return list(self._handsets.values())
        
    def get_frames(self, limit: int = 100) -> List[Dict]:
        return [f.to_dict() for f in list(self._frames)[-limit:]]
        
    def get_statistics(self) -> Dict:
        return {**self._stats, "base_stations": len(self._base_stations), "handsets": len(self._handsets)}
