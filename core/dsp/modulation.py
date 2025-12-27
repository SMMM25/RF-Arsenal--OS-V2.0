#!/usr/bin/env python3
"""
RF Arsenal OS - Modulation Engine

Production-grade digital modulation/demodulation.
Supports PSK, QAM, FSK, GMSK, and OFDM modulation schemes.
"""

import numpy as np
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import signal as scipy_signal
from scipy.fft import fft, ifft

logger = logging.getLogger(__name__)


class ModulationType(Enum):
    """Supported modulation types"""
    BPSK = "bpsk"
    QPSK = "qpsk"
    PSK8 = "8psk"
    QAM16 = "16qam"
    QAM64 = "64qam"
    QAM256 = "256qam"
    FSK2 = "2fsk"
    FSK4 = "4fsk"
    GMSK = "gmsk"
    OFDM = "ofdm"
    PI4_DQPSK = "pi4dqpsk"  # Used in TETRA, NADC


@dataclass
class ModulationConfig:
    """Modulation configuration"""
    mod_type: ModulationType
    samples_per_symbol: int = 4
    pulse_shaping: bool = True
    rolloff: float = 0.35  # Root raised cosine roll-off
    filter_span: int = 10  # Filter span in symbols
    
    # FSK specific
    fsk_deviation: float = 1000.0  # Hz
    
    # GMSK specific
    gmsk_bt: float = 0.3  # Bandwidth-time product


class ModulationEngine:
    """
    Central modulation engine supporting multiple schemes.
    
    Features:
    - Gray-coded constellation mapping
    - Root raised cosine pulse shaping
    - Efficient vectorized processing
    - Stealth-aware (randomized preambles)
    """
    
    def __init__(self, config: ModulationConfig, sample_rate: float = 1e6):
        self.config = config
        self.sample_rate = sample_rate
        self.symbol_rate = sample_rate / config.samples_per_symbol
        
        # Initialize modulator based on type
        self._init_modulator()
    
    def _init_modulator(self):
        """Initialize appropriate modulator"""
        if self.config.mod_type in [ModulationType.BPSK, ModulationType.QPSK,
                                     ModulationType.PSK8]:
            self.modulator = PSKModulator(self.config, self.sample_rate)
        elif self.config.mod_type in [ModulationType.QAM16, ModulationType.QAM64,
                                       ModulationType.QAM256]:
            self.modulator = QAMModulator(self.config, self.sample_rate)
        elif self.config.mod_type in [ModulationType.FSK2, ModulationType.FSK4]:
            self.modulator = FSKModulator(self.config, self.sample_rate)
        elif self.config.mod_type == ModulationType.GMSK:
            self.modulator = GMSKModulator(self.config, self.sample_rate)
        elif self.config.mod_type == ModulationType.PI4_DQPSK:
            self.modulator = Pi4DQPSKModulator(self.config, self.sample_rate)
        else:
            raise ValueError(f"Unsupported modulation: {self.config.mod_type}")
    
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Modulate bit stream to IQ samples"""
        return self.modulator.modulate(bits)
    
    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate IQ samples to bits"""
        return self.modulator.demodulate(samples)
    
    def get_bits_per_symbol(self) -> int:
        """Get number of bits per symbol"""
        return self.modulator.bits_per_symbol


class PSKModulator:
    """
    Phase Shift Keying modulator/demodulator
    
    Supports BPSK, QPSK, 8-PSK with Gray coding.
    """
    
    # Gray-coded constellation points
    CONSTELLATIONS = {
        ModulationType.BPSK: np.array([1, -1]),
        ModulationType.QPSK: np.array([
            1+1j, -1+1j, 1-1j, -1-1j  # Gray coded
        ]) / np.sqrt(2),
        ModulationType.PSK8: np.array([
            np.exp(1j * k * np.pi / 4) for k in [0, 1, 3, 2, 7, 6, 4, 5]  # Gray
        ]),
    }
    
    BITS_PER_SYMBOL = {
        ModulationType.BPSK: 1,
        ModulationType.QPSK: 2,
        ModulationType.PSK8: 3,
    }
    
    def __init__(self, config: ModulationConfig, sample_rate: float):
        self.config = config
        self.sample_rate = sample_rate
        self.constellation = self.CONSTELLATIONS[config.mod_type]
        self.bits_per_symbol = self.BITS_PER_SYMBOL[config.mod_type]
        
        # Create pulse shaping filter
        if config.pulse_shaping:
            self.pulse_filter = self._create_rrc_filter()
        else:
            self.pulse_filter = None
    
    def _create_rrc_filter(self) -> np.ndarray:
        """Create root raised cosine pulse shaping filter"""
        num_taps = self.config.filter_span * self.config.samples_per_symbol + 1
        
        # Time vector centered at 0
        t = (np.arange(num_taps) - (num_taps - 1) / 2) / self.config.samples_per_symbol
        
        h = np.zeros(num_taps)
        beta = self.config.rolloff
        
        for i, ti in enumerate(t):
            if ti == 0:
                h[i] = 1 + beta * (4 / np.pi - 1)
            elif beta > 0 and abs(abs(ti) - 1 / (4 * beta)) < 1e-10:
                h[i] = (beta / np.sqrt(2)) * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                    (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                )
            else:
                denom = np.pi * ti * (1 - (4 * beta * ti) ** 2)
                if abs(denom) > 1e-10:
                    h[i] = (
                        np.sin(np.pi * ti * (1 - beta)) +
                        4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
                    ) / denom
        
        # Normalize for unity gain
        h /= np.sqrt(np.sum(h ** 2))
        
        return h
    
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Modulate bits to complex baseband"""
        # Ensure bits array is proper length
        num_bits = len(bits)
        padding = (self.bits_per_symbol - num_bits % self.bits_per_symbol) % self.bits_per_symbol
        if padding > 0:
            bits = np.concatenate([bits, np.zeros(padding, dtype=int)])
        
        # Convert bits to symbols
        num_symbols = len(bits) // self.bits_per_symbol
        bits_reshaped = bits.reshape(num_symbols, self.bits_per_symbol)
        
        # Convert to decimal indices
        indices = np.sum(bits_reshaped * (2 ** np.arange(self.bits_per_symbol)[::-1]), axis=1)
        indices = indices.astype(int)
        
        # Map to constellation
        symbols = self.constellation[indices]
        
        # Upsample
        samples = np.zeros(num_symbols * self.config.samples_per_symbol, dtype=complex)
        samples[::self.config.samples_per_symbol] = symbols
        
        # Apply pulse shaping
        if self.pulse_filter is not None:
            samples = np.convolve(samples, self.pulse_filter, mode='same')
        
        return samples
    
    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate complex baseband to bits"""
        # Apply matched filter
        if self.pulse_filter is not None:
            samples = np.convolve(samples, self.pulse_filter[::-1], mode='same')
        
        # Downsample to symbol rate
        symbols = samples[::self.config.samples_per_symbol]
        
        # Make hard decisions (minimum distance)
        indices = np.zeros(len(symbols), dtype=int)
        for i, sym in enumerate(symbols):
            distances = np.abs(self.constellation - sym) ** 2
            indices[i] = np.argmin(distances)
        
        # Convert indices to bits
        bits = np.zeros(len(indices) * self.bits_per_symbol, dtype=int)
        for i, idx in enumerate(indices):
            for j in range(self.bits_per_symbol):
                bits[i * self.bits_per_symbol + j] = (idx >> (self.bits_per_symbol - 1 - j)) & 1
        
        return bits


class QAMModulator:
    """
    Quadrature Amplitude Modulation modulator/demodulator
    
    Supports 16-QAM, 64-QAM, 256-QAM with Gray coding.
    """
    
    def __init__(self, config: ModulationConfig, sample_rate: float):
        self.config = config
        self.sample_rate = sample_rate
        
        # Set up constellation
        if config.mod_type == ModulationType.QAM16:
            self.bits_per_symbol = 4
            self.m = 4  # Constellation size per dimension
        elif config.mod_type == ModulationType.QAM64:
            self.bits_per_symbol = 6
            self.m = 8
        elif config.mod_type == ModulationType.QAM256:
            self.bits_per_symbol = 8
            self.m = 16
        else:
            raise ValueError(f"Invalid QAM type: {config.mod_type}")
        
        self.constellation = self._create_constellation()
        
        # Pulse shaping filter
        if config.pulse_shaping:
            self.pulse_filter = self._create_rrc_filter()
        else:
            self.pulse_filter = None
    
    def _create_constellation(self) -> np.ndarray:
        """Create Gray-coded QAM constellation"""
        # Generate Gray code
        def gray_code(n):
            if n == 0:
                return [0]
            smaller = gray_code(n - 1)
            return smaller + [x | (1 << (n - 1)) for x in reversed(smaller)]
        
        gray_i = gray_code(int(np.log2(self.m)))
        gray_q = gray_code(int(np.log2(self.m)))
        
        # Create constellation points
        levels = np.arange(self.m) - (self.m - 1) / 2
        levels = levels * 2  # Standard spacing
        
        constellation = np.zeros(self.m ** 2, dtype=complex)
        for i in range(self.m):
            for q in range(self.m):
                idx = gray_i[i] * self.m + gray_q[q]
                constellation[idx] = levels[i] + 1j * levels[q]
        
        # Normalize average power to 1
        avg_power = np.mean(np.abs(constellation) ** 2)
        constellation /= np.sqrt(avg_power)
        
        return constellation
    
    def _create_rrc_filter(self) -> np.ndarray:
        """Create root raised cosine filter"""
        num_taps = self.config.filter_span * self.config.samples_per_symbol + 1
        t = (np.arange(num_taps) - (num_taps - 1) / 2) / self.config.samples_per_symbol
        beta = self.config.rolloff
        
        h = np.zeros(num_taps)
        for i, ti in enumerate(t):
            if ti == 0:
                h[i] = 1 + beta * (4 / np.pi - 1)
            elif beta > 0 and abs(abs(ti) - 1 / (4 * beta)) < 1e-10:
                h[i] = (beta / np.sqrt(2)) * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                    (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                )
            else:
                denom = np.pi * ti * (1 - (4 * beta * ti) ** 2)
                if abs(denom) > 1e-10:
                    h[i] = (
                        np.sin(np.pi * ti * (1 - beta)) +
                        4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
                    ) / denom
        
        h /= np.sqrt(np.sum(h ** 2))
        return h
    
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Modulate bits to QAM signal"""
        # Pad bits if needed
        padding = (self.bits_per_symbol - len(bits) % self.bits_per_symbol) % self.bits_per_symbol
        if padding > 0:
            bits = np.concatenate([bits, np.zeros(padding, dtype=int)])
        
        # Convert to symbols
        num_symbols = len(bits) // self.bits_per_symbol
        bits_reshaped = bits.reshape(num_symbols, self.bits_per_symbol)
        indices = np.sum(bits_reshaped * (2 ** np.arange(self.bits_per_symbol)[::-1]), axis=1)
        symbols = self.constellation[indices.astype(int)]
        
        # Upsample
        samples = np.zeros(num_symbols * self.config.samples_per_symbol, dtype=complex)
        samples[::self.config.samples_per_symbol] = symbols
        
        # Pulse shaping
        if self.pulse_filter is not None:
            samples = np.convolve(samples, self.pulse_filter, mode='same')
        
        return samples
    
    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate QAM signal to bits"""
        # Matched filter
        if self.pulse_filter is not None:
            samples = np.convolve(samples, self.pulse_filter[::-1], mode='same')
        
        # Downsample
        symbols = samples[::self.config.samples_per_symbol]
        
        # Hard decision
        indices = np.zeros(len(symbols), dtype=int)
        for i, sym in enumerate(symbols):
            distances = np.abs(self.constellation - sym) ** 2
            indices[i] = np.argmin(distances)
        
        # Convert to bits
        bits = np.zeros(len(indices) * self.bits_per_symbol, dtype=int)
        for i, idx in enumerate(indices):
            for j in range(self.bits_per_symbol):
                bits[i * self.bits_per_symbol + j] = (idx >> (self.bits_per_symbol - 1 - j)) & 1
        
        return bits


class FSKModulator:
    """
    Frequency Shift Keying modulator/demodulator
    
    Supports 2-FSK and 4-FSK.
    """
    
    def __init__(self, config: ModulationConfig, sample_rate: float):
        self.config = config
        self.sample_rate = sample_rate
        self.symbol_rate = sample_rate / config.samples_per_symbol
        
        if config.mod_type == ModulationType.FSK2:
            self.bits_per_symbol = 1
            self.m = 2
            self.freq_offsets = np.array([-1, 1]) * config.fsk_deviation
        elif config.mod_type == ModulationType.FSK4:
            self.bits_per_symbol = 2
            self.m = 4
            self.freq_offsets = np.array([-3, -1, 1, 3]) * config.fsk_deviation / 3
        
        # Phase accumulator for continuous phase
        self._phase = 0.0
    
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Modulate bits to FSK signal"""
        # Pad bits
        padding = (self.bits_per_symbol - len(bits) % self.bits_per_symbol) % self.bits_per_symbol
        if padding > 0:
            bits = np.concatenate([bits, np.zeros(padding, dtype=int)])
        
        num_symbols = len(bits) // self.bits_per_symbol
        samples_per_sym = self.config.samples_per_symbol
        total_samples = num_symbols * samples_per_sym
        
        samples = np.zeros(total_samples, dtype=complex)
        
        for i in range(num_symbols):
            # Get symbol index
            bit_slice = bits[i * self.bits_per_symbol:(i + 1) * self.bits_per_symbol]
            idx = int(np.sum(bit_slice * (2 ** np.arange(self.bits_per_symbol)[::-1])))
            
            # Get frequency for this symbol
            freq = self.freq_offsets[idx]
            
            # Generate samples with continuous phase
            for j in range(samples_per_sym):
                sample_idx = i * samples_per_sym + j
                samples[sample_idx] = np.exp(1j * self._phase)
                self._phase += 2 * np.pi * freq / self.sample_rate
            
            # Wrap phase
            self._phase = self._phase % (2 * np.pi)
        
        return samples
    
    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate FSK signal using frequency discriminator"""
        # Frequency discriminator (differentiate phase)
        phase = np.unwrap(np.angle(samples))
        freq = np.diff(phase) * self.sample_rate / (2 * np.pi)
        
        # Downsample to symbol rate
        samples_per_sym = self.config.samples_per_symbol
        num_symbols = len(freq) // samples_per_sym
        
        bits = []
        for i in range(num_symbols):
            # Average frequency over symbol
            sym_freq = np.mean(freq[i * samples_per_sym:(i + 1) * samples_per_sym])
            
            # Find closest frequency
            distances = np.abs(self.freq_offsets - sym_freq)
            idx = np.argmin(distances)
            
            # Convert to bits
            for j in range(self.bits_per_symbol):
                bits.append((idx >> (self.bits_per_symbol - 1 - j)) & 1)
        
        return np.array(bits, dtype=int)
    
    def reset(self):
        """Reset phase accumulator"""
        self._phase = 0.0


class GMSKModulator:
    """
    Gaussian Minimum Shift Keying modulator/demodulator
    
    Used in GSM, DECT, Bluetooth.
    """
    
    def __init__(self, config: ModulationConfig, sample_rate: float):
        self.config = config
        self.sample_rate = sample_rate
        self.bits_per_symbol = 1
        self.bt = config.gmsk_bt  # Bandwidth-time product
        
        # Create Gaussian filter
        self.gaussian_filter = self._create_gaussian_filter()
        
        # Phase accumulator
        self._phase = 0.0
    
    def _create_gaussian_filter(self) -> np.ndarray:
        """Create Gaussian filter for MSK"""
        # Filter length in symbol periods
        span = self.config.filter_span
        samples_per_sym = self.config.samples_per_symbol
        
        # Time vector
        num_taps = span * samples_per_sym + 1
        t = (np.arange(num_taps) - (num_taps - 1) / 2) / samples_per_sym
        
        # Gaussian pulse
        alpha = np.sqrt(np.log(2) / 2) / self.bt
        h = (np.sqrt(np.pi) / alpha) * np.exp(-(np.pi * t / alpha) ** 2)
        
        # Normalize
        h /= np.sum(h)
        
        return h
    
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Modulate bits to GMSK signal"""
        samples_per_sym = self.config.samples_per_symbol
        
        # NRZ encoding (-1, +1)
        nrz = 2 * bits.astype(float) - 1
        
        # Upsample
        upsampled = np.zeros(len(nrz) * samples_per_sym)
        upsampled[::samples_per_sym] = nrz
        
        # Apply Gaussian filter
        filtered = np.convolve(upsampled, self.gaussian_filter, mode='same')
        
        # Integrate to get phase
        # MSK modulation index h = 0.5
        phase_increment = filtered * (np.pi / 2) / samples_per_sym
        phase = np.cumsum(phase_increment) + self._phase
        
        # Store final phase for continuity
        self._phase = phase[-1] if len(phase) > 0 else self._phase
        
        # Generate complex signal
        samples = np.exp(1j * phase)
        
        return samples
    
    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate GMSK signal"""
        samples_per_sym = self.config.samples_per_symbol
        
        # Frequency discriminator
        phase = np.unwrap(np.angle(samples))
        freq = np.diff(phase)
        
        # Integrate and dump
        num_symbols = len(freq) // samples_per_sym
        bits = np.zeros(num_symbols, dtype=int)
        
        for i in range(num_symbols):
            sym_phase = np.sum(freq[i * samples_per_sym:(i + 1) * samples_per_sym])
            bits[i] = 1 if sym_phase > 0 else 0
        
        return bits
    
    def reset(self):
        """Reset modulator state"""
        self._phase = 0.0


class Pi4DQPSKModulator:
    """
    π/4 Differential QPSK modulator
    
    Used in TETRA, NADC, PDC.
    """
    
    def __init__(self, config: ModulationConfig, sample_rate: float):
        self.config = config
        self.sample_rate = sample_rate
        self.bits_per_symbol = 2
        
        # π/4 DQPSK phase transitions
        # Maps dibit to phase change
        self.phase_map = {
            (0, 0): np.pi / 4,
            (0, 1): 3 * np.pi / 4,
            (1, 0): -np.pi / 4,
            (1, 1): -3 * np.pi / 4,
        }
        
        # Current phase
        self._phase = 0.0
        
        # Pulse shaping
        if config.pulse_shaping:
            self.pulse_filter = self._create_rrc_filter()
        else:
            self.pulse_filter = None
    
    def _create_rrc_filter(self) -> np.ndarray:
        """Create root raised cosine filter"""
        num_taps = self.config.filter_span * self.config.samples_per_symbol + 1
        t = (np.arange(num_taps) - (num_taps - 1) / 2) / self.config.samples_per_symbol
        beta = self.config.rolloff
        
        h = np.zeros(num_taps)
        for i, ti in enumerate(t):
            if ti == 0:
                h[i] = 1 + beta * (4 / np.pi - 1)
            elif beta > 0 and abs(abs(ti) - 1 / (4 * beta)) < 1e-10:
                h[i] = (beta / np.sqrt(2)) * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                    (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                )
            else:
                denom = np.pi * ti * (1 - (4 * beta * ti) ** 2)
                if abs(denom) > 1e-10:
                    h[i] = (
                        np.sin(np.pi * ti * (1 - beta)) +
                        4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
                    ) / denom
        
        h /= np.sqrt(np.sum(h ** 2))
        return h
    
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Modulate bits to π/4 DQPSK"""
        # Pad bits
        padding = (2 - len(bits) % 2) % 2
        if padding > 0:
            bits = np.concatenate([bits, np.zeros(padding, dtype=int)])
        
        num_symbols = len(bits) // 2
        symbols = np.zeros(num_symbols, dtype=complex)
        
        for i in range(num_symbols):
            dibit = (bits[2 * i], bits[2 * i + 1])
            phase_change = self.phase_map[dibit]
            self._phase += phase_change
            symbols[i] = np.exp(1j * self._phase)
        
        # Upsample
        samples = np.zeros(num_symbols * self.config.samples_per_symbol, dtype=complex)
        samples[::self.config.samples_per_symbol] = symbols
        
        # Pulse shaping
        if self.pulse_filter is not None:
            samples = np.convolve(samples, self.pulse_filter, mode='same')
        
        return samples
    
    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate π/4 DQPSK signal"""
        # Matched filter
        if self.pulse_filter is not None:
            samples = np.convolve(samples, self.pulse_filter[::-1], mode='same')
        
        # Downsample
        symbols = samples[::self.config.samples_per_symbol]
        
        # Differential decoding
        bits = []
        prev_phase = 0.0
        
        for sym in symbols:
            phase = np.angle(sym)
            phase_diff = phase - prev_phase
            
            # Wrap to [-π, π]
            while phase_diff > np.pi:
                phase_diff -= 2 * np.pi
            while phase_diff < -np.pi:
                phase_diff += 2 * np.pi
            
            # Map phase difference to dibit
            if -np.pi / 2 < phase_diff <= 0:
                bits.extend([1, 0])
            elif 0 < phase_diff <= np.pi / 2:
                bits.extend([0, 0])
            elif np.pi / 2 < phase_diff <= np.pi:
                bits.extend([0, 1])
            else:
                bits.extend([1, 1])
            
            prev_phase = phase
        
        return np.array(bits, dtype=int)
    
    def reset(self):
        """Reset modulator state"""
        self._phase = 0.0


class OFDMModulator:
    """
    Basic OFDM modulator/demodulator
    
    For full LTE/5G OFDM, see ofdm.py
    """
    
    def __init__(self, num_subcarriers: int = 64,
                 cp_length: int = 16,
                 subcarrier_modulation: ModulationType = ModulationType.QPSK):
        self.num_subcarriers = num_subcarriers
        self.cp_length = cp_length
        self.subcarrier_mod = subcarrier_modulation
        
        # Create subcarrier modulator
        config = ModulationConfig(
            mod_type=subcarrier_modulation,
            samples_per_symbol=1,
            pulse_shaping=False
        )
        
        if subcarrier_modulation in [ModulationType.BPSK, ModulationType.QPSK]:
            self.subcarrier_modulator = PSKModulator(config, 1.0)
        else:
            self.subcarrier_modulator = QAMModulator(config, 1.0)
        
        self.bits_per_symbol = self.subcarrier_modulator.bits_per_symbol * num_subcarriers
    
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Modulate bits to OFDM symbols"""
        bits_per_ofdm = self.subcarrier_modulator.bits_per_symbol * self.num_subcarriers
        
        # Pad bits
        padding = (bits_per_ofdm - len(bits) % bits_per_ofdm) % bits_per_ofdm
        if padding > 0:
            bits = np.concatenate([bits, np.zeros(padding, dtype=int)])
        
        num_ofdm_symbols = len(bits) // bits_per_ofdm
        ofdm_symbol_length = self.num_subcarriers + self.cp_length
        samples = np.zeros(num_ofdm_symbols * ofdm_symbol_length, dtype=complex)
        
        for i in range(num_ofdm_symbols):
            # Get bits for this OFDM symbol
            sym_bits = bits[i * bits_per_ofdm:(i + 1) * bits_per_ofdm]
            
            # Modulate subcarriers
            subcarrier_symbols = self.subcarrier_modulator.modulate(sym_bits)[:self.num_subcarriers]
            
            # IFFT
            time_domain = ifft(subcarrier_symbols) * np.sqrt(self.num_subcarriers)
            
            # Add cyclic prefix
            cp = time_domain[-self.cp_length:]
            ofdm_symbol = np.concatenate([cp, time_domain])
            
            # Store
            samples[i * ofdm_symbol_length:(i + 1) * ofdm_symbol_length] = ofdm_symbol
        
        return samples
    
    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate OFDM symbols to bits"""
        ofdm_symbol_length = self.num_subcarriers + self.cp_length
        num_ofdm_symbols = len(samples) // ofdm_symbol_length
        
        all_bits = []
        
        for i in range(num_ofdm_symbols):
            # Extract OFDM symbol
            ofdm_symbol = samples[i * ofdm_symbol_length:(i + 1) * ofdm_symbol_length]
            
            # Remove cyclic prefix
            time_domain = ofdm_symbol[self.cp_length:]
            
            # FFT
            subcarrier_symbols = fft(time_domain) / np.sqrt(self.num_subcarriers)
            
            # Demodulate subcarriers
            bits = self.subcarrier_modulator.demodulate(subcarrier_symbols)
            all_bits.extend(bits)
        
        return np.array(all_bits, dtype=int)


class Demodulator:
    """
    Universal demodulator with automatic modulation detection
    """
    
    def __init__(self, sample_rate: float = 1e6):
        self.sample_rate = sample_rate
    
    def detect_modulation(self, samples: np.ndarray) -> ModulationType:
        """
        Automatically detect modulation type
        
        Uses statistical analysis of constellation and spectrum.
        """
        # Normalize
        samples = samples / np.sqrt(np.mean(np.abs(samples) ** 2))
        
        # Analyze constellation
        # Check for constant envelope (FSK, GMSK)
        envelope_variance = np.var(np.abs(samples))
        
        if envelope_variance < 0.1:
            # Constant envelope - likely FSK or GMSK
            # Check frequency distribution
            phase_diff = np.diff(np.unwrap(np.angle(samples)))
            unique_freqs = len(np.unique(np.round(phase_diff, 2)))
            
            if unique_freqs <= 2:
                return ModulationType.GMSK
            elif unique_freqs <= 4:
                return ModulationType.FSK2
            else:
                return ModulationType.FSK4
        
        # Variable envelope - PSK or QAM
        # Count unique amplitude levels
        amplitudes = np.abs(samples)
        amplitude_levels = len(np.unique(np.round(amplitudes, 1)))
        
        if amplitude_levels <= 2:
            # Likely PSK
            phases = np.angle(samples)
            phase_levels = len(np.unique(np.round(phases, 1)))
            
            if phase_levels <= 2:
                return ModulationType.BPSK
            elif phase_levels <= 4:
                return ModulationType.QPSK
            else:
                return ModulationType.PSK8
        else:
            # Likely QAM
            if amplitude_levels <= 4:
                return ModulationType.QAM16
            elif amplitude_levels <= 8:
                return ModulationType.QAM64
            else:
                return ModulationType.QAM256
    
    def demodulate(self, samples: np.ndarray, 
                   mod_type: Optional[ModulationType] = None,
                   samples_per_symbol: int = 4) -> np.ndarray:
        """
        Demodulate samples
        
        Args:
            samples: IQ samples
            mod_type: Modulation type (auto-detect if None)
            samples_per_symbol: Oversampling factor
        """
        if mod_type is None:
            mod_type = self.detect_modulation(samples)
        
        config = ModulationConfig(
            mod_type=mod_type,
            samples_per_symbol=samples_per_symbol
        )
        
        engine = ModulationEngine(config, self.sample_rate)
        return engine.demodulate(samples)
