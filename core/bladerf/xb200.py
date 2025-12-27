#!/usr/bin/env python3
"""
RF Arsenal OS - BladeRF XB-200 Transverter Module
Hardware: BladeRF 2.0 micro xA9 + XB-200 Transverter

XB-200 extends BladeRF coverage to:
- 9 kHz - 300 MHz (HF/VHF bands)
- 60 MHz - 6 GHz (standard BladeRF)

Supported bands/applications:
- HF amateur radio (1.8 - 30 MHz)
- Shortwave broadcast (3 - 30 MHz)  
- VHF FM radio (88 - 108 MHz)
- VHF aircraft band (108 - 137 MHz)
- VHF amateur (144 - 148 MHz)
- Marine VHF (156 - 162 MHz)
- NOAA weather radio (162.4 - 162.55 MHz)
- VHF public safety (138 - 174 MHz)
- Military VHF (225 - 400 MHz)
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
from enum import Enum
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


class XB200FilterBank(Enum):
    """XB-200 filter bank selections"""
    AUTO = "auto"                          # Automatic selection
    BYPASS = "bypass"                      # No filtering (60 MHz - 6 GHz)
    CUSTOM = "custom"                      # Custom filter
    # Low-pass filters
    LP_50MHZ = "50mhz_lpf"                 # 50 MHz LPF
    LP_144MHZ = "144mhz_lpf"               # 144 MHz LPF
    LP_222MHZ = "222mhz_lpf"               # 222 MHz LPF


class XB200Band(Enum):
    """XB-200 frequency bands"""
    HF_LOW = "hf_low"                      # 9 kHz - 1.8 MHz
    HF_160M = "hf_160m"                    # 1.8 - 2.0 MHz (160m amateur)
    HF_80M = "hf_80m"                      # 3.5 - 4.0 MHz (80m amateur)
    HF_40M = "hf_40m"                      # 7.0 - 7.3 MHz (40m amateur)
    HF_20M = "hf_20m"                      # 14.0 - 14.35 MHz (20m amateur)
    HF_15M = "hf_15m"                      # 21.0 - 21.45 MHz (15m amateur)
    HF_10M = "hf_10m"                      # 28.0 - 29.7 MHz (10m amateur)
    VHF_LOW = "vhf_low"                    # 30 - 50 MHz
    VHF_FM = "vhf_fm"                      # 88 - 108 MHz (FM broadcast)
    VHF_AIR = "vhf_air"                    # 108 - 137 MHz (aircraft)
    VHF_2M = "vhf_2m"                      # 144 - 148 MHz (2m amateur)
    VHF_MARINE = "vhf_marine"              # 156 - 162 MHz (marine)
    VHF_WEATHER = "vhf_weather"            # 162 - 163 MHz (NOAA)
    VHF_HIGH = "vhf_high"                  # 174 - 300 MHz


@dataclass
class XB200Config:
    """XB-200 configuration"""
    enabled: bool = True
    filter_bank: XB200FilterBank = XB200FilterBank.AUTO
    band: XB200Band = XB200Band.VHF_FM
    frequency_hz: int = 100_000_000        # 100 MHz default
    
    # RX settings
    rx_enabled: bool = True
    rx_gain_db: int = 30
    rx_lna_enabled: bool = True
    
    # TX settings (if licensed)
    tx_enabled: bool = False
    tx_power_dbm: int = 0
    
    # Preamp/attenuator
    preamp_enabled: bool = False
    attenuator_db: int = 0


@dataclass 
class XB200BandInfo:
    """Information about a frequency band"""
    name: str
    freq_min_hz: int
    freq_max_hz: int
    modulation: str
    bandwidth_hz: int
    common_uses: List[str]
    filter_recommended: XB200FilterBank
    requires_license: bool


@dataclass
class SignalDetection:
    """Detected signal in XB-200 range"""
    frequency_hz: int
    bandwidth_hz: int
    power_dbm: float
    modulation: str
    classification: str
    timestamp: str
    snr_db: float


class BladeRFXB200:
    """
    BladeRF XB-200 Transverter Controller
    
    Extends BladeRF frequency coverage:
    - 9 kHz - 300 MHz (with XB-200)
    - HF amateur radio bands
    - Shortwave broadcast
    - VHF radio services
    """
    
    # XB-200 frequency range
    FREQ_MIN = 9_000                       # 9 kHz
    FREQ_MAX = 300_000_000                 # 300 MHz
    FREQ_BYPASS_MIN = 60_000_000           # 60 MHz bypass mode
    
    # Band definitions
    BANDS: Dict[XB200Band, XB200BandInfo] = {
        XB200Band.HF_LOW: XB200BandInfo(
            name="LF/MF",
            freq_min_hz=9_000,
            freq_max_hz=1_800_000,
            modulation="AM/CW",
            bandwidth_hz=10_000,
            common_uses=["NDB", "AM broadcast", "Maritime"],
            filter_recommended=XB200FilterBank.LP_50MHZ,
            requires_license=False
        ),
        XB200Band.HF_160M: XB200BandInfo(
            name="160m Amateur",
            freq_min_hz=1_800_000,
            freq_max_hz=2_000_000,
            modulation="SSB/CW/Digital",
            bandwidth_hz=3_000,
            common_uses=["Amateur radio", "DX", "Contests"],
            filter_recommended=XB200FilterBank.LP_50MHZ,
            requires_license=True
        ),
        XB200Band.HF_80M: XB200BandInfo(
            name="80m Amateur",
            freq_min_hz=3_500_000,
            freq_max_hz=4_000_000,
            modulation="SSB/CW/Digital",
            bandwidth_hz=3_000,
            common_uses=["Amateur radio", "NVIS", "Regional"],
            filter_recommended=XB200FilterBank.LP_50MHZ,
            requires_license=True
        ),
        XB200Band.HF_40M: XB200BandInfo(
            name="40m Amateur",
            freq_min_hz=7_000_000,
            freq_max_hz=7_300_000,
            modulation="SSB/CW/Digital",
            bandwidth_hz=3_000,
            common_uses=["Amateur radio", "DX", "Digital modes"],
            filter_recommended=XB200FilterBank.LP_50MHZ,
            requires_license=True
        ),
        XB200Band.HF_20M: XB200BandInfo(
            name="20m Amateur",
            freq_min_hz=14_000_000,
            freq_max_hz=14_350_000,
            modulation="SSB/CW/Digital",
            bandwidth_hz=3_000,
            common_uses=["Amateur radio", "DX", "Contests"],
            filter_recommended=XB200FilterBank.LP_50MHZ,
            requires_license=True
        ),
        XB200Band.HF_15M: XB200BandInfo(
            name="15m Amateur",
            freq_min_hz=21_000_000,
            freq_max_hz=21_450_000,
            modulation="SSB/CW/Digital",
            bandwidth_hz=3_000,
            common_uses=["Amateur radio", "DX"],
            filter_recommended=XB200FilterBank.LP_50MHZ,
            requires_license=True
        ),
        XB200Band.HF_10M: XB200BandInfo(
            name="10m Amateur",
            freq_min_hz=28_000_000,
            freq_max_hz=29_700_000,
            modulation="SSB/FM/CW",
            bandwidth_hz=10_000,
            common_uses=["Amateur radio", "Sporadic E"],
            filter_recommended=XB200FilterBank.LP_50MHZ,
            requires_license=True
        ),
        XB200Band.VHF_LOW: XB200BandInfo(
            name="VHF Low Band",
            freq_min_hz=30_000_000,
            freq_max_hz=50_000_000,
            modulation="FM/AM",
            bandwidth_hz=25_000,
            common_uses=["Military", "Public safety", "CB (27 MHz)"],
            filter_recommended=XB200FilterBank.LP_50MHZ,
            requires_license=False
        ),
        XB200Band.VHF_FM: XB200BandInfo(
            name="FM Broadcast",
            freq_min_hz=88_000_000,
            freq_max_hz=108_000_000,
            modulation="WFM",
            bandwidth_hz=200_000,
            common_uses=["FM radio stations", "RDS data"],
            filter_recommended=XB200FilterBank.LP_144MHZ,
            requires_license=False
        ),
        XB200Band.VHF_AIR: XB200BandInfo(
            name="Aircraft Band",
            freq_min_hz=108_000_000,
            freq_max_hz=137_000_000,
            modulation="AM",
            bandwidth_hz=25_000,
            common_uses=["Aviation comm", "ACARS", "VOR/ILS"],
            filter_recommended=XB200FilterBank.LP_144MHZ,
            requires_license=False
        ),
        XB200Band.VHF_2M: XB200BandInfo(
            name="2m Amateur",
            freq_min_hz=144_000_000,
            freq_max_hz=148_000_000,
            modulation="FM/SSB/Digital",
            bandwidth_hz=25_000,
            common_uses=["Amateur radio", "Repeaters", "APRS"],
            filter_recommended=XB200FilterBank.LP_144MHZ,
            requires_license=True
        ),
        XB200Band.VHF_MARINE: XB200BandInfo(
            name="Marine VHF",
            freq_min_hz=156_000_000,
            freq_max_hz=162_000_000,
            modulation="FM",
            bandwidth_hz=25_000,
            common_uses=["Ship-to-ship", "Ship-to-shore", "DSC"],
            filter_recommended=XB200FilterBank.LP_222MHZ,
            requires_license=False
        ),
        XB200Band.VHF_WEATHER: XB200BandInfo(
            name="NOAA Weather",
            freq_min_hz=162_400_000,
            freq_max_hz=162_550_000,
            modulation="NFM",
            bandwidth_hz=5_000,
            common_uses=["Weather broadcasts", "SAME alerts"],
            filter_recommended=XB200FilterBank.LP_222MHZ,
            requires_license=False
        ),
        XB200Band.VHF_HIGH: XB200BandInfo(
            name="VHF High Band",
            freq_min_hz=174_000_000,
            freq_max_hz=300_000_000,
            modulation="Various",
            bandwidth_hz=25_000,
            common_uses=["TV broadcast", "DAB radio", "Military"],
            filter_recommended=XB200FilterBank.LP_222MHZ,
            requires_license=False
        ),
    }
    
    # NOAA Weather Radio frequencies
    NOAA_FREQUENCIES = [
        162_400_000, 162_425_000, 162_450_000,
        162_475_000, 162_500_000, 162_525_000, 162_550_000
    ]
    
    # Common shortwave broadcast bands
    SHORTWAVE_BANDS = [
        (2_300_000, 2_495_000, "120m tropical"),
        (3_200_000, 3_400_000, "90m tropical"),
        (3_900_000, 4_000_000, "75m"),
        (4_750_000, 5_060_000, "60m tropical"),
        (5_900_000, 6_200_000, "49m"),
        (7_200_000, 7_450_000, "41m"),
        (9_400_000, 9_900_000, "31m"),
        (11_600_000, 12_100_000, "25m"),
        (13_570_000, 13_870_000, "22m"),
        (15_100_000, 15_800_000, "19m"),
        (17_480_000, 17_900_000, "16m"),
        (18_900_000, 19_020_000, "15m"),
        (21_450_000, 21_850_000, "13m"),
        (25_600_000, 26_100_000, "11m"),
    ]
    
    def __init__(self, hardware_controller=None):
        """
        Initialize XB-200 controller
        
        Args:
            hardware_controller: BladeRF hardware controller
        """
        self.hw = hardware_controller
        self.config = XB200Config()
        self.is_receiving = False
        self._rx_thread = None
        
        # State
        self._xb200_detected = False
        self._current_filter = XB200FilterBank.AUTO
        self._signal_detections: List[SignalDetection] = []
        
        # Callbacks
        self._rx_callback: Optional[Callable] = None
        self._signal_callback: Optional[Callable] = None
        
        # Try to detect XB-200
        self._detect_xb200()
        
        logger.info("BladeRF XB-200 controller initialized")
    
    def _detect_xb200(self) -> bool:
        """Detect if XB-200 is attached"""
        if self.hw:
            try:
                # In real implementation: self._xb200_detected = self.hw.xb200_attached()
                pass
            except Exception:
                pass
        
        # Assume present for simulation
        self._xb200_detected = True
        logger.info(f"XB-200 detected: {self._xb200_detected}")
        return self._xb200_detected
    
    def is_present(self) -> bool:
        """Check if XB-200 is present"""
        return self._xb200_detected
    
    def configure(self, config: XB200Config) -> bool:
        """
        Configure XB-200 transverter
        
        Args:
            config: XB200Config with desired settings
            
        Returns:
            True if configuration successful
        """
        if not self._xb200_detected:
            logger.error("XB-200 not detected")
            return False
        
        # Validate frequency
        if not self.FREQ_MIN <= config.frequency_hz <= self.FREQ_MAX:
            logger.error(f"Frequency {config.frequency_hz/1e6:.3f} MHz out of XB-200 range")
            return False
        
        self.config = config
        
        # Select appropriate filter
        if config.filter_bank == XB200FilterBank.AUTO:
            self._current_filter = self._select_filter(config.frequency_hz)
        else:
            self._current_filter = config.filter_bank
        
        # Apply to hardware
        if self.hw:
            try:
                # self.hw.xb200_set_filterbank(self._current_filter.value)
                # self.hw.set_frequency(config.frequency_hz)
                pass
            except Exception as e:
                logger.error(f"XB-200 configuration failed: {e}")
                return False
        
        logger.info(f"XB-200 configured: {config.frequency_hz/1e6:.3f} MHz, "
                   f"filter={self._current_filter.value}")
        return True
    
    def _select_filter(self, freq_hz: int) -> XB200FilterBank:
        """Auto-select appropriate filter for frequency"""
        if freq_hz < 50_000_000:
            return XB200FilterBank.LP_50MHZ
        elif freq_hz < 144_000_000:
            return XB200FilterBank.LP_144MHZ
        elif freq_hz < 222_000_000:
            return XB200FilterBank.LP_222MHZ
        else:
            return XB200FilterBank.BYPASS
    
    def tune(self, frequency_hz: int) -> bool:
        """
        Tune to frequency
        
        Args:
            frequency_hz: Target frequency in Hz
            
        Returns:
            True if tuned successfully
        """
        if not self.FREQ_MIN <= frequency_hz <= self.FREQ_MAX:
            logger.error(f"Frequency out of XB-200 range")
            return False
        
        self.config.frequency_hz = frequency_hz
        
        # Update filter if in auto mode
        if self.config.filter_bank == XB200FilterBank.AUTO:
            new_filter = self._select_filter(frequency_hz)
            if new_filter != self._current_filter:
                self._current_filter = new_filter
                logger.info(f"Filter changed to {new_filter.value}")
        
        if self.hw:
            # self.hw.set_frequency(frequency_hz)
            pass
        
        logger.debug(f"Tuned to {frequency_hz/1e6:.6f} MHz")
        return True
    
    def tune_band(self, band: XB200Band) -> bool:
        """
        Tune to center of a band
        
        Args:
            band: XB200Band to tune to
            
        Returns:
            True if tuned successfully
        """
        if band not in self.BANDS:
            logger.error(f"Unknown band: {band}")
            return False
        
        band_info = self.BANDS[band]
        center_freq = (band_info.freq_min_hz + band_info.freq_max_hz) // 2
        
        self.config.band = band
        return self.tune(center_freq)
    
    def start_receive(self, callback: Optional[Callable] = None) -> bool:
        """
        Start receiving on XB-200
        
        Args:
            callback: Function to call with received samples
            
        Returns:
            True if started successfully
        """
        if self.is_receiving:
            logger.warning("Already receiving")
            return False
        
        self._rx_callback = callback
        self.is_receiving = True
        
        self._rx_thread = threading.Thread(target=self._rx_worker, daemon=True)
        self._rx_thread.start()
        
        logger.info("XB-200 receive started")
        return True
    
    def stop_receive(self):
        """Stop receiving"""
        self.is_receiving = False
        if self._rx_thread:
            self._rx_thread.join(timeout=1.0)
        logger.info("XB-200 receive stopped")
    
    def _rx_worker(self):
        """Receive worker thread"""
        while self.is_receiving:
            try:
                # Receive samples
                if self.hw:
                    # samples = self.hw.receive(4096)
                    pass
                else:
                    # Simulate
                    samples = np.random.randn(4096) + 1j * np.random.randn(4096)
                    samples *= 0.01
                
                if self._rx_callback:
                    self._rx_callback(samples)
                
                # Small delay
                import time
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"RX worker error: {e}")
    
    def scan_band(self, band: XB200Band, step_hz: int = 10_000,
                  dwell_ms: int = 100) -> List[SignalDetection]:
        """
        Scan a frequency band for signals
        
        Args:
            band: XB200Band to scan
            step_hz: Frequency step size
            dwell_ms: Time on each frequency
            
        Returns:
            List of detected signals
        """
        if band not in self.BANDS:
            logger.error(f"Unknown band: {band}")
            return []
        
        band_info = self.BANDS[band]
        detections = []
        
        logger.info(f"Scanning {band_info.name} ({band_info.freq_min_hz/1e6:.3f} - "
                   f"{band_info.freq_max_hz/1e6:.3f} MHz)")
        
        import time
        freq = band_info.freq_min_hz
        while freq <= band_info.freq_max_hz:
            self.tune(freq)
            time.sleep(dwell_ms / 1000)
            
            # Measure signal
            power = self._measure_power()
            
            # Detection threshold
            if power > -80:
                detection = SignalDetection(
                    frequency_hz=freq,
                    bandwidth_hz=band_info.bandwidth_hz,
                    power_dbm=power,
                    modulation=band_info.modulation,
                    classification=self._classify_signal(freq, band_info),
                    timestamp=datetime.now().isoformat(),
                    snr_db=power + 100  # Rough SNR estimate
                )
                detections.append(detection)
                
                if self._signal_callback:
                    self._signal_callback(detection)
            
            freq += step_hz
        
        self._signal_detections.extend(detections)
        logger.info(f"Scan complete: {len(detections)} signals detected")
        return detections
    
    def _measure_power(self) -> float:
        """Measure signal power at current frequency"""
        if self.hw:
            try:
                # return self.hw.get_rssi()
                pass
            except Exception:
                pass
        
        # Simulate with some random signals
        return np.random.uniform(-100, -60)
    
    def _classify_signal(self, freq_hz: int, band_info: XB200BandInfo) -> str:
        """Classify detected signal based on frequency"""
        # NOAA Weather Radio
        for noaa_freq in self.NOAA_FREQUENCIES:
            if abs(freq_hz - noaa_freq) < 5000:
                return "NOAA Weather Radio"
        
        # Aircraft
        if 118_000_000 <= freq_hz <= 137_000_000:
            return "Aviation Communication"
        
        # Marine
        if 156_025_000 <= freq_hz <= 157_425_000:
            return "Marine VHF"
        
        # FM Broadcast
        if 88_000_000 <= freq_hz <= 108_000_000:
            return "FM Broadcast Station"
        
        return band_info.name
    
    def receive_fm_broadcast(self, frequency_hz: int = 100_000_000) -> Dict:
        """
        Receive FM broadcast radio
        
        Args:
            frequency_hz: FM station frequency
            
        Returns:
            Dict with demodulated audio info
        """
        if not 88_000_000 <= frequency_hz <= 108_000_000:
            logger.error("Frequency not in FM broadcast band")
            return {}
        
        self.tune(frequency_hz)
        
        # Would demodulate WFM here
        return {
            'frequency_mhz': frequency_hz / 1e6,
            'modulation': 'WFM',
            'status': 'receiving',
            'rds': None,  # Would extract RDS data
        }
    
    def receive_noaa_weather(self, channel: int = 1) -> Dict:
        """
        Receive NOAA Weather Radio
        
        Args:
            channel: NOAA channel 1-7
            
        Returns:
            Dict with weather radio info
        """
        if not 1 <= channel <= 7:
            logger.error("NOAA channel must be 1-7")
            return {}
        
        freq = self.NOAA_FREQUENCIES[channel - 1]
        self.tune(freq)
        
        return {
            'channel': channel,
            'frequency_mhz': freq / 1e6,
            'modulation': 'NFM',
            'status': 'receiving',
        }
    
    def receive_aircraft(self, frequency_hz: int = 121_500_000) -> Dict:
        """
        Receive aircraft band communications
        
        Args:
            frequency_hz: Aircraft frequency (121.5 MHz = emergency)
            
        Returns:
            Dict with aircraft comm info
        """
        if not 108_000_000 <= frequency_hz <= 137_000_000:
            logger.error("Frequency not in aircraft band")
            return {}
        
        self.tune(frequency_hz)
        
        return {
            'frequency_mhz': frequency_hz / 1e6,
            'modulation': 'AM',
            'status': 'receiving',
            'special': '121.5 MHz = Guard/Emergency' if frequency_hz == 121_500_000 else None,
        }
    
    def scan_shortwave(self) -> List[SignalDetection]:
        """
        Scan shortwave broadcast bands
        
        Returns:
            List of detected broadcasts
        """
        logger.info("Scanning shortwave broadcast bands...")
        
        detections = []
        import time
        
        for start, end, name in self.SHORTWAVE_BANDS:
            logger.info(f"Scanning {name} band ({start/1e6:.1f} - {end/1e6:.1f} MHz)")
            
            freq = start
            while freq <= end:
                self.tune(freq)
                time.sleep(0.05)
                
                power = self._measure_power()
                if power > -85:
                    detection = SignalDetection(
                        frequency_hz=freq,
                        bandwidth_hz=5000,
                        power_dbm=power,
                        modulation="AM/SSB",
                        classification=f"Shortwave {name}",
                        timestamp=datetime.now().isoformat(),
                        snr_db=power + 100
                    )
                    detections.append(detection)
                
                freq += 5000  # 5 kHz step
        
        logger.info(f"Shortwave scan complete: {len(detections)} signals")
        return detections
    
    def get_band_info(self, band: XB200Band) -> Optional[XB200BandInfo]:
        """Get information about a band"""
        return self.BANDS.get(band)
    
    def list_bands(self) -> List[Dict]:
        """List all available bands"""
        return [
            {
                'band': band.value,
                'name': info.name,
                'freq_min_mhz': info.freq_min_hz / 1e6,
                'freq_max_mhz': info.freq_max_hz / 1e6,
                'modulation': info.modulation,
                'uses': info.common_uses,
                'license_required': info.requires_license,
            }
            for band, info in self.BANDS.items()
        ]
    
    def get_status(self) -> Dict:
        """Get XB-200 status"""
        return {
            'xb200_present': self._xb200_detected,
            'enabled': self.config.enabled,
            'frequency_mhz': self.config.frequency_hz / 1e6,
            'band': self.config.band.value,
            'filter': self._current_filter.value,
            'rx_enabled': self.config.rx_enabled,
            'tx_enabled': self.config.tx_enabled,
            'receiving': self.is_receiving,
            'signals_detected': len(self._signal_detections),
        }


# Convenience function
def get_xb200_controller(hardware_controller=None) -> BladeRFXB200:
    """Get BladeRF XB-200 controller instance"""
    return BladeRFXB200(hardware_controller)
