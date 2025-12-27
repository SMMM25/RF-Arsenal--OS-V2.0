#!/usr/bin/env python3
"""
RF Arsenal OS - Amateur Radio (Ham Radio) Module
Hardware: BladeRF 2.0 micro xA9
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class HamRadioConfig:
    """Ham Radio Configuration"""
    frequency: int = 14_200_000  # 20m band (14.2 MHz)
    sample_rate: int = 2_000_000  # 2 MSPS
    bandwidth: int = 3_000        # 3 kHz SSB
    mode: str = "usb"             # usb, lsb, cw, fm, am, digital
    tx_power: int = 5             # dBm (low power for testing)
    callsign: str = "N0CALL"      # Amateur radio callsign

@dataclass
class QSO:
    """QSO (Contact) Record"""
    timestamp: datetime
    frequency: int
    mode: str
    callsign: str
    rst_sent: str
    rst_received: str
    name: str
    qth: str  # Location
    notes: str

class AmateurRadio:
    """Amateur Radio Transceiver System"""
    
    # Amateur radio bands (meters: frequency range in Hz)
    HAM_BANDS = {
        '160m': (1_800_000, 2_000_000),
        '80m': (3_500_000, 4_000_000),
        '60m': (5_330_500, 5_403_500),
        '40m': (7_000_000, 7_300_000),
        '30m': (10_100_000, 10_150_000),
        '20m': (14_000_000, 14_350_000),
        '17m': (18_068_000, 18_168_000),
        '15m': (21_000_000, 21_450_000),
        '12m': (24_890_000, 24_990_000),
        '10m': (28_000_000, 29_700_000),
        '6m': (50_000_000, 54_000_000),
        '2m': (144_000_000, 148_000_000),
        '70cm': (420_000_000, 450_000_000),
    }
    
    # Common calling frequencies
    CALLING_FREQUENCIES = {
        'cw': {
            '20m': 14_060_000,
            '40m': 7_030_000,
            '80m': 3_560_000,
        },
        'ssb': {
            '20m': 14_200_000,
            '40m': 7_200_000,
            '80m': 3_750_000,
        },
        'fm': {
            '2m': 146_520_000,  # National simplex
            '70cm': 446_000_000,
        },
        'digital': {
            '20m': 14_070_000,  # FT8
            '40m': 7_074_000,
        }
    }
    
    def __init__(self, hardware_controller):
        """
        Initialize amateur radio system
        
        Args:
            hardware_controller: BladeRF hardware controller instance
        """
        self.hw = hardware_controller
        self.config = HamRadioConfig()
        self.is_running = False
        self.qso_log: List[QSO] = []
        self.received_messages: List[str] = []
        
    def configure(self, config: HamRadioConfig) -> bool:
        """Configure amateur radio transceiver"""
        try:
            self.config = config
            
            # Validate frequency is in ham band
            if not self._is_valid_ham_frequency(config.frequency):
                logger.warning(f"Frequency {config.frequency/1e6:.3f} MHz "
                             "may not be in amateur radio band")
            
            # Configure BladeRF
            if not self.hw.configure_hardware({
                'frequency': config.frequency,
                'sample_rate': config.sample_rate,
                'bandwidth': config.bandwidth,
                'rx_gain': 40,
                'tx_gain': config.tx_power
            }):
                logger.error("Failed to configure hardware")
                return False
                
            logger.info(f"Ham radio configured: {config.frequency/1e6:.3f} MHz, "
                       f"Mode: {config.mode.upper()}, Callsign: {config.callsign}")
            return True
            
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            return False
    
    def _is_valid_ham_frequency(self, frequency: int) -> bool:
        """Check if frequency is in amateur radio band"""
        for band_range in self.HAM_BANDS.values():
            if band_range[0] <= frequency <= band_range[1]:
                return True
        return False
    
    def listen(self, duration: float = 60.0) -> List[str]:
        """
        Listen for transmissions
        
        Args:
            duration: Listen duration in seconds
            
        Returns:
            List of decoded messages
        """
        try:
            logger.info(f"Listening on {self.config.frequency/1e6:.3f} MHz "
                       f"({self.config.mode.upper()}) for {duration}s...")
            
            messages = []
            start_time = datetime.now()
            
            while (datetime.now() - start_time).total_seconds() < duration:
                # Receive samples
                samples = self.hw.receive_samples(
                    int(self.config.sample_rate * 1.0)  # 1 second
                )
                
                if samples is None:
                    continue
                
                # Demodulate based on mode
                audio = self._demodulate(samples)
                
                # Detect signal presence
                if audio is not None and np.max(np.abs(audio)) > 0.1:
                    # Decode message
                    message = self._decode_message(audio)
                    if message:
                        messages.append(message)
                        self.received_messages.append(message)
                        logger.info(f"Received: {message}")
            
            logger.info(f"Listening complete: {len(messages)} message(s)")
            return messages
            
        except Exception as e:
            logger.error(f"Listen error: {e}")
            return []
    
    def _demodulate(self, samples: np.ndarray) -> Optional[np.ndarray]:
        """Demodulate based on mode"""
        if self.config.mode in ['usb', 'lsb']:
            return self._demodulate_ssb(samples)
        elif self.config.mode == 'am':
            return self._demodulate_am(samples)
        elif self.config.mode == 'fm':
            return self._demodulate_fm(samples)
        elif self.config.mode == 'cw':
            return self._demodulate_cw(samples)
        else:
            return None
    
    def _demodulate_ssb(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate SSB (Single Sideband)"""
        # SSB demodulation: shift to baseband and take real part
        if self.config.mode == 'usb':
            # USB: use positive frequencies
            audio = np.real(samples)
        else:  # lsb
            # LSB: use negative frequencies
            audio = np.real(samples * np.exp(-2j * np.pi * np.arange(len(samples))))
        
        return audio
    
    def _demodulate_am(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate AM (Amplitude Modulation)"""
        # AM demodulation: envelope detection
        envelope = np.abs(samples)
        return envelope
    
    def _demodulate_fm(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate FM (Frequency Modulation)"""
        # FM demodulation: phase differentiation
        phase = np.angle(samples)
        audio = np.diff(phase)
        audio = np.unwrap(audio)
        return audio
    
    def _demodulate_cw(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate CW (Morse Code)"""
        # CW demodulation: envelope detection at audio tone
        envelope = np.abs(samples)
        return envelope
    
    def _decode_message(self, audio: np.ndarray) -> Optional[str]:
        """Decode audio to text (simplified)"""
        # Simplified message decoding
        # In production, implement proper audio processing and speech recognition
        
        # Calculate signal strength
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms > 0.1:
            return f"Signal detected (RMS: {rms:.3f})"
        
        return None
    
    def transmit_voice(self, audio_data: np.ndarray) -> bool:
        """
        Transmit voice (requires microphone input)
        
        Args:
            audio_data: Audio samples to transmit
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Transmitting voice on {self.config.frequency/1e6:.3f} MHz")
            
            # Send ID (callsign) first
            self._send_id()
            
            # Modulate audio
            modulated = self._modulate_voice(audio_data)
            
            # Transmit
            if self.hw.transmit_burst(modulated):
                logger.info("Voice transmission complete")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Transmission error: {e}")
            return False
    
    def _send_id(self):
        """Send station identification (required by FCC)"""
        logger.info(f"ID: {self.config.callsign}")
        # In production, generate CW or voice ID
    
    def _modulate_voice(self, audio: np.ndarray) -> np.ndarray:
        """Modulate audio for transmission"""
        if self.config.mode in ['usb', 'lsb']:
            return self._modulate_ssb(audio)
        elif self.config.mode == 'am':
            return self._modulate_am(audio)
        elif self.config.mode == 'fm':
            return self._modulate_fm(audio)
        else:
            return audio.astype(np.complex64)
    
    def _modulate_ssb(self, audio: np.ndarray) -> np.ndarray:
        """Modulate SSB"""
        # SSB modulation: Hilbert transform for single sideband
        signal = audio + 1j * np.imag(np.fft.hilbert(audio))
        
        if self.config.mode == 'lsb':
            signal = np.conj(signal)
        
        signal *= 0.5  # Power
        return signal.astype(np.complex64)
    
    def _modulate_am(self, audio: np.ndarray) -> np.ndarray:
        """Modulate AM"""
        # AM modulation
        carrier = 1.0 + 0.5 * audio  # 50% modulation
        signal = carrier.astype(np.complex64)
        return signal
    
    def _modulate_fm(self, audio: np.ndarray) -> np.ndarray:
        """Modulate FM"""
        # FM modulation
        deviation = 5000  # 5 kHz deviation
        phase = 2 * np.pi * np.cumsum(audio) * deviation / self.config.sample_rate
        signal = np.exp(1j * phase)
        signal *= 0.5
        return signal.astype(np.complex64)
    
    def transmit_cw(self, message: str, wpm: int = 20) -> bool:
        """
        Transmit CW (Morse Code)
        
        Args:
            message: Message to send
            wpm: Words per minute
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Transmitting CW: {message}")
            
            # Convert to Morse code
            morse = self._text_to_morse(message)
            
            # Generate CW signal
            cw_signal = self._generate_cw_signal(morse, wpm)
            
            # Transmit
            if self.hw.transmit_burst(cw_signal):
                logger.info("CW transmission complete")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"CW transmission error: {e}")
            return False
    
    def _text_to_morse(self, text: str) -> str:
        """Convert text to Morse code"""
        morse_code = {
            'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
            'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
            'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
            'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
            'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
            'Z': '--..', '0': '-----', '1': '.----', '2': '..---',
            '3': '...--', '4': '....-', '5': '.....', '6': '-....',
            '7': '--...', '8': '---..', '9': '----.', ' ': '/'
        }
        
        morse = ' '.join([morse_code.get(c.upper(), '') for c in text])
        return morse
    
    def _generate_cw_signal(self, morse: str, wpm: int) -> np.ndarray:
        """Generate CW signal from Morse code"""
        # Timing: 1 dit = 1.2 / wpm seconds
        dit_duration = 1.2 / wpm
        samples_per_dit = int(self.config.sample_rate * dit_duration)
        
        # CW tone (typically 700 Hz audio tone)
        tone_freq = 700
        
        signal_parts = []
        
        for symbol in morse:
            if symbol == '.':
                # Dit
                duration = samples_per_dit
            elif symbol == '-':
                # Dah (3x dit)
                duration = samples_per_dit * 3
            elif symbol == ' ':
                # Inter-character space
                duration = samples_per_dit * 3
                signal_parts.append(np.zeros(duration, dtype=np.complex64))
                continue
            elif symbol == '/':
                # Word space
                duration = samples_per_dit * 7
                signal_parts.append(np.zeros(duration, dtype=np.complex64))
                continue
            else:
                continue
            
            # Generate tone
            t = np.linspace(0, duration / self.config.sample_rate, 
                          duration, endpoint=False)
            tone = np.exp(2j * np.pi * tone_freq * t)
            signal_parts.append(tone * 0.5)
            
            # Inter-element space (1 dit)
            signal_parts.append(np.zeros(samples_per_dit, dtype=np.complex64))
        
        signal = np.concatenate(signal_parts).astype(np.complex64)
        return signal
    
    def transmit_digital(self, message: str, mode: str = "psk31") -> bool:
        """
        Transmit digital mode (PSK31, FT8, etc.)
        
        Args:
            message: Message to send
            mode: Digital mode (psk31, rtty, ft8)
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Transmitting {mode.upper()}: {message}")
            
            # Modulate for digital mode
            if mode == "psk31":
                modulated = self._modulate_psk31(message)
            elif mode == "rtty":
                modulated = self._modulate_rtty(message)
            else:
                logger.error(f"Unsupported mode: {mode}")
                return False
            
            # Transmit
            if self.hw.transmit_burst(modulated):
                logger.info("Digital transmission complete")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Digital transmission error: {e}")
            return False
    
    def _modulate_psk31(self, message: str) -> np.ndarray:
        """Modulate PSK31"""
        # PSK31: 31.25 baud BPSK
        baud_rate = 31.25
        samples_per_bit = int(self.config.sample_rate / baud_rate)
        
        # Convert to bits (simplified)
        bits = []
        for char in message:
            bits.extend([int(b) for b in format(ord(char), '08b')])
        
        # BPSK modulation
        signal = np.zeros(len(bits) * samples_per_bit, dtype=np.complex64)
        carrier_freq = 1000  # 1 kHz audio
        
        for i, bit in enumerate(bits):
            start = i * samples_per_bit
            end = start + samples_per_bit
            t = np.linspace(0, samples_per_bit / self.config.sample_rate,
                          samples_per_bit, endpoint=False)
            
            # Phase: 0 for bit 1, π for bit 0
            phase = 0 if bit else np.pi
            signal[start:end] = np.exp(1j * (2 * np.pi * carrier_freq * t + phase))
        
        signal *= 0.3
        return signal
    
    def _modulate_rtty(self, message: str) -> np.ndarray:
        """Modulate RTTY (Radioteletype)"""
        # RTTY: FSK modulation (mark/space)
        baud_rate = 45.45  # 45.45 baud
        samples_per_bit = int(self.config.sample_rate / baud_rate)
        
        mark_freq = 2125  # Hz
        space_freq = 2295  # Hz (170 Hz shift)
        
        # Convert to Baudot code (simplified: use ASCII)
        bits = []
        for char in message:
            bits.extend([int(b) for b in format(ord(char), '08b')])
        
        # FSK modulation
        signal = np.zeros(len(bits) * samples_per_bit, dtype=np.complex64)
        
        for i, bit in enumerate(bits):
            start = i * samples_per_bit
            end = start + samples_per_bit
            t = np.linspace(0, samples_per_bit / self.config.sample_rate,
                          samples_per_bit, endpoint=False)
            
            freq = mark_freq if bit else space_freq
            signal[start:end] = np.exp(2j * np.pi * freq * t)
        
        signal *= 0.3
        return signal
    
    def log_qso(self, callsign: str, rst_sent: str = "59", 
               rst_received: str = "59", name: str = "", 
               qth: str = "", notes: str = ""):
        """
        Log a QSO (contact)
        
        Args:
            callsign: Other station's callsign
            rst_sent: RST report sent
            rst_received: RST report received
            name: Operator name
            qth: Location
            notes: Additional notes
        """
        qso = QSO(
            timestamp=datetime.now(),
            frequency=self.config.frequency,
            mode=self.config.mode.upper(),
            callsign=callsign,
            rst_sent=rst_sent,
            rst_received=rst_received,
            name=name,
            qth=qth,
            notes=notes
        )
        
        self.qso_log.append(qso)
        logger.info(f"QSO logged: {callsign}")
    
    def export_adif(self, filename: str = "logbook.adi") -> bool:
        """Export QSO log to ADIF format"""
        try:
            with open(filename, 'w') as f:
                f.write("ADIF Export from RF Arsenal OS\n")
                f.write("<ADIF_VER:5>3.1.0\n")
                f.write("<EOH>\n\n")
                
                for qso in self.qso_log:
                    f.write(f"<CALL:{len(qso.callsign)}>{qso.callsign}\n")
                    f.write(f"<QSO_DATE:8>{qso.timestamp.strftime('%Y%m%d')}\n")
                    f.write(f"<TIME_ON:6>{qso.timestamp.strftime('%H%M%S')}\n")
                    f.write(f"<FREQ:{len(str(qso.frequency/1e6))}>{qso.frequency/1e6}\n")
                    f.write(f"<MODE:{len(qso.mode)}>{qso.mode}\n")
                    f.write(f"<RST_SENT:{len(qso.rst_sent)}>{qso.rst_sent}\n")
                    f.write(f"<RST_RCVD:{len(qso.rst_received)}>{qso.rst_received}\n")
                    if qso.name:
                        f.write(f"<NAME:{len(qso.name)}>{qso.name}\n")
                    if qso.qth:
                        f.write(f"<QTH:{len(qso.qth)}>{qso.qth}\n")
                    f.write("<EOR>\n\n")
            
            logger.info(f"Exported {len(self.qso_log)} QSOs to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Export error: {e}")
            return False
    
    def get_qso_log(self) -> List[QSO]:
        """Get QSO log"""
        return self.qso_log
    
    def stop(self):
        """Stop amateur radio operations"""
        self.is_running = False
        self.hw.stop_transmission()
        logger.info("Amateur radio operations stopped")

def main():
    """Test amateur radio module"""
    from core.hardware import HardwareController
    
    # Initialize hardware
    hw = HardwareController()
    if not hw.connect():
        print("Failed to connect to BladeRF")
        return
    
    # Create ham radio system
    ham = AmateurRadio(hw)
    
    # Configure for 20m SSB
    config = HamRadioConfig(
        frequency=14_200_000,  # 14.2 MHz
        mode="usb",
        callsign="N0CALL"  # Use your actual callsign
    )
    
    if not ham.configure(config):
        print("Configuration failed")
        return
    
    print("RF Arsenal OS - Amateur Radio Transceiver")
    print("=" * 50)
    print(f"Callsign: {config.callsign}")
    print(f"Frequency: {config.frequency/1e6:.3f} MHz")
    print(f"Mode: {config.mode.upper()}")
    
    # Listen for transmissions
    print("\nListening for 10 seconds...")
    messages = ham.listen(duration=10.0)
    
    print(f"\nReceived {len(messages)} transmission(s)")
    
    # Demo: Send CW (commented for safety)
    # print("\nSending CW identification...")
    # ham.transmit_cw("CQ CQ CQ DE N0CALL K")
    
    # Log a QSO (example)
    ham.log_qso("W1AW", rst_sent="59", rst_received="59", 
                name="Test", qth="Newington, CT")
    
    # Export log
    ham.export_adif("ham_logbook.adi")
    
    ham.stop()
    hw.disconnect()

if __name__ == "__main__":
    main()
