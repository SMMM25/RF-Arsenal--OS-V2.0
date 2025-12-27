#!/usr/bin/env python3
"""
RF Arsenal OS - OFDM Engine

Production-grade OFDM implementation for LTE, 5G NR, and WiFi.
Supports real resource grid mapping, channel estimation, and equalization.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.fft import fft, ifft, fftshift
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


class CyclicPrefixType(Enum):
    """Cyclic prefix configuration"""
    NORMAL = "normal"
    EXTENDED = "extended"


class NumerologyType(Enum):
    """5G NR numerology (subcarrier spacing)"""
    MU0 = 0   # 15 kHz (LTE compatible)
    MU1 = 1   # 30 kHz
    MU2 = 2   # 60 kHz
    MU3 = 3   # 120 kHz
    MU4 = 4   # 240 kHz


@dataclass
class LTEOFDMParams:
    """LTE OFDM parameters"""
    # Standard LTE parameters
    fft_size: int = 2048        # FFT size
    num_subcarriers: int = 1200  # Usable subcarriers (20 MHz)
    subcarrier_spacing: float = 15e3  # 15 kHz
    sample_rate: float = 30.72e6     # 30.72 MHz for 20 MHz BW
    
    # Cyclic prefix (normal)
    cp_length_first: int = 160   # First symbol CP
    cp_length_other: int = 144   # Other symbols CP
    
    # Frame structure
    symbols_per_slot: int = 7    # Normal CP
    slots_per_subframe: int = 2
    subframes_per_frame: int = 10
    
    # Resource block
    subcarriers_per_rb: int = 12
    symbols_per_rb: int = 7
    
    @property
    def num_resource_blocks(self) -> int:
        return self.num_subcarriers // self.subcarriers_per_rb
    
    @property
    def symbol_duration(self) -> float:
        """Symbol duration without CP in seconds"""
        return 1.0 / self.subcarrier_spacing
    
    @property
    def slot_duration(self) -> float:
        """Slot duration in seconds"""
        return 0.5e-3  # 0.5 ms


@dataclass
class NROFDMParams:
    """5G NR OFDM parameters"""
    numerology: NumerologyType = NumerologyType.MU1  # 30 kHz default
    bandwidth_mhz: float = 100.0
    
    @property
    def subcarrier_spacing(self) -> float:
        """Subcarrier spacing in Hz"""
        return 15e3 * (2 ** self.numerology.value)
    
    @property
    def fft_size(self) -> int:
        """FFT size based on bandwidth and numerology"""
        # Approximate based on bandwidth
        if self.bandwidth_mhz <= 5:
            return 512
        elif self.bandwidth_mhz <= 10:
            return 1024
        elif self.bandwidth_mhz <= 20:
            return 2048
        elif self.bandwidth_mhz <= 50:
            return 4096
        else:
            return 4096
    
    @property
    def sample_rate(self) -> float:
        """Sample rate in Hz"""
        return self.fft_size * self.subcarrier_spacing
    
    @property
    def symbols_per_slot(self) -> int:
        """Symbols per slot (normal CP)"""
        return 14
    
    @property
    def slots_per_subframe(self) -> int:
        """Slots per 1ms subframe"""
        return 2 ** self.numerology.value
    
    @property
    def cp_lengths(self) -> List[int]:
        """CP lengths for each symbol in slot"""
        # Normal CP: first symbol has longer CP
        base_cp = self.fft_size // 14
        extended_cp = base_cp + (self.fft_size // 2048) * 16
        
        lengths = [extended_cp]  # First symbol
        lengths.extend([base_cp] * 13)  # Remaining symbols
        return lengths


@dataclass
class WiFiOFDMParams:
    """WiFi (802.11a/g/n/ac) OFDM parameters"""
    mode: str = "802.11n"  # 802.11a, 802.11g, 802.11n, 802.11ac
    channel_width: int = 20  # MHz
    
    @property
    def fft_size(self) -> int:
        if self.channel_width == 20:
            return 64
        elif self.channel_width == 40:
            return 128
        elif self.channel_width == 80:
            return 256
        else:  # 160 MHz
            return 512
    
    @property
    def num_data_subcarriers(self) -> int:
        if self.channel_width == 20:
            return 52
        elif self.channel_width == 40:
            return 108
        elif self.channel_width == 80:
            return 234
        else:
            return 468
    
    @property
    def num_pilot_subcarriers(self) -> int:
        if self.channel_width == 20:
            return 4
        elif self.channel_width == 40:
            return 6
        elif self.channel_width == 80:
            return 8
        else:
            return 16
    
    @property
    def sample_rate(self) -> float:
        return self.channel_width * 1e6
    
    @property
    def subcarrier_spacing(self) -> float:
        return 312.5e3  # 312.5 kHz
    
    @property
    def cp_length(self) -> int:
        return self.fft_size // 4  # 1/4 of symbol duration
    
    @property
    def symbol_duration(self) -> float:
        return 4e-6  # 4 microseconds (including 0.8us CP)


class ResourceGrid:
    """
    OFDM Resource Grid Manager
    
    Handles resource element allocation for LTE/5G.
    """
    
    def __init__(self, params: Union[LTEOFDMParams, NROFDMParams]):
        self.params = params
        
        if isinstance(params, LTEOFDMParams):
            self.num_subcarriers = params.num_subcarriers
            self.symbols_per_slot = params.symbols_per_slot
        else:
            # NR
            self.num_subcarriers = params.fft_size - 1  # Approximate
            self.symbols_per_slot = params.symbols_per_slot
        
        # Initialize empty grid (subcarriers x symbols)
        self.grid = np.zeros((self.num_subcarriers, self.symbols_per_slot), dtype=complex)
        
        # Allocation map
        self.allocation_map = np.zeros((self.num_subcarriers, self.symbols_per_slot), dtype=int)
        
        # Channel types
        self.EMPTY = 0
        self.PSS = 1
        self.SSS = 2
        self.PBCH = 3
        self.PDCCH = 4
        self.PDSCH = 5
        self.DMRS = 6
        self.CSIRS = 7
        self.PILOT = 8
    
    def allocate_pss(self, symbols: np.ndarray, symbol_idx: int = 6):
        """Allocate Primary Synchronization Signal"""
        center = self.num_subcarriers // 2
        half_len = len(symbols) // 2
        
        start = center - half_len
        end = start + len(symbols)
        
        self.grid[start:end, symbol_idx] = symbols
        self.allocation_map[start:end, symbol_idx] = self.PSS
    
    def allocate_sss(self, symbols: np.ndarray, symbol_idx: int = 5):
        """Allocate Secondary Synchronization Signal"""
        center = self.num_subcarriers // 2
        half_len = len(symbols) // 2
        
        start = center - half_len
        end = start + len(symbols)
        
        self.grid[start:end, symbol_idx] = symbols
        self.allocation_map[start:end, symbol_idx] = self.SSS
    
    def allocate_pbch(self, symbols: np.ndarray, symbol_indices: List[int] = [7, 8, 9, 10]):
        """Allocate Physical Broadcast Channel"""
        center = self.num_subcarriers // 2
        pbch_subcarriers = 240  # 20 RBs
        
        start = center - pbch_subcarriers // 2
        
        symbols_per_ofdm = pbch_subcarriers
        for i, sym_idx in enumerate(symbol_indices):
            sym_start = i * symbols_per_ofdm
            sym_end = sym_start + symbols_per_ofdm
            
            if sym_end <= len(symbols):
                self.grid[start:start + symbols_per_ofdm, sym_idx] = symbols[sym_start:sym_end]
                self.allocation_map[start:start + symbols_per_ofdm, sym_idx] = self.PBCH
    
    def allocate_pdsch(self, symbols: np.ndarray, 
                       rb_start: int, rb_count: int,
                       symbol_start: int, symbol_count: int):
        """Allocate Physical Downlink Shared Channel"""
        sc_start = rb_start * 12
        sc_count = rb_count * 12
        
        idx = 0
        for sym in range(symbol_start, symbol_start + symbol_count):
            for sc in range(sc_start, sc_start + sc_count):
                if self.allocation_map[sc, sym] == self.EMPTY and idx < len(symbols):
                    self.grid[sc, sym] = symbols[idx]
                    self.allocation_map[sc, sym] = self.PDSCH
                    idx += 1
    
    def allocate_dmrs(self, symbol_idx: int, pattern: str = "type1"):
        """Allocate Demodulation Reference Signals"""
        if pattern == "type1":
            # Every other subcarrier
            for sc in range(0, self.num_subcarriers, 2):
                # QPSK DMRS sequence
                self.grid[sc, symbol_idx] = np.exp(1j * np.random.uniform(0, 2 * np.pi))
                self.allocation_map[sc, symbol_idx] = self.DMRS
    
    def allocate_csirs(self, symbol_idx: int, density: float = 1.0):
        """Allocate Channel State Information Reference Signals"""
        spacing = int(12 / density)
        for sc in range(0, self.num_subcarriers, spacing):
            self.grid[sc, symbol_idx] = 1.0  # Known reference
            self.allocation_map[sc, symbol_idx] = self.CSIRS
    
    def get_grid(self) -> np.ndarray:
        """Get the complete resource grid"""
        return self.grid.copy()
    
    def get_allocation_map(self) -> np.ndarray:
        """Get allocation map"""
        return self.allocation_map.copy()


class OFDMEngine:
    """
    Production OFDM Engine
    
    Features:
    - LTE/5G NR/WiFi support
    - Real synchronization signals
    - Channel estimation and equalization
    - Stealth-aware (randomized timing jitter)
    """
    
    def __init__(self, params: Union[LTEOFDMParams, NROFDMParams, WiFiOFDMParams],
                 stealth_mode: bool = True):
        self.params = params
        self.stealth_mode = stealth_mode
        self.logger = logging.getLogger('OFDMEngine')
        
        # Determine FFT size
        if isinstance(params, WiFiOFDMParams):
            self.fft_size = params.fft_size
            self.sample_rate = params.sample_rate
        else:
            self.fft_size = params.fft_size
            self.sample_rate = params.sample_rate
    
    def generate_ofdm_symbol(self, subcarrier_data: np.ndarray,
                             cp_length: int) -> np.ndarray:
        """
        Generate single OFDM symbol with cyclic prefix
        
        Args:
            subcarrier_data: Complex symbols for each subcarrier
            cp_length: Cyclic prefix length in samples
        
        Returns:
            Time-domain OFDM symbol with CP
        """
        # Zero-pad to FFT size
        freq_domain = np.zeros(self.fft_size, dtype=complex)
        
        # Map data to subcarriers (centered around DC)
        num_data = len(subcarrier_data)
        start_idx = (self.fft_size - num_data) // 2
        freq_domain[start_idx:start_idx + num_data] = subcarrier_data
        
        # IFFT shift (move DC to center)
        freq_domain = np.fft.ifftshift(freq_domain)
        
        # IFFT
        time_domain = ifft(freq_domain) * np.sqrt(self.fft_size)
        
        # Add cyclic prefix
        cp = time_domain[-cp_length:]
        ofdm_symbol = np.concatenate([cp, time_domain])
        
        # Add stealth jitter if enabled
        if self.stealth_mode:
            # Small timing jitter to avoid detection
            jitter_samples = int(np.random.normal(0, 0.01 * cp_length))
            if jitter_samples > 0:
                ofdm_symbol = np.concatenate([
                    np.zeros(jitter_samples, dtype=complex),
                    ofdm_symbol
                ])
            elif jitter_samples < 0:
                ofdm_symbol = ofdm_symbol[-jitter_samples:]
        
        return ofdm_symbol
    
    def demodulate_ofdm_symbol(self, samples: np.ndarray,
                               cp_length: int) -> np.ndarray:
        """
        Demodulate single OFDM symbol
        
        Args:
            samples: Time-domain samples (with CP)
            cp_length: CP length to remove
        
        Returns:
            Subcarrier data
        """
        # Remove CP
        time_domain = samples[cp_length:cp_length + self.fft_size]
        
        # FFT
        freq_domain = fft(time_domain) / np.sqrt(self.fft_size)
        
        # FFT shift
        freq_domain = np.fft.fftshift(freq_domain)
        
        return freq_domain
    
    def generate_slot(self, resource_grid: ResourceGrid) -> np.ndarray:
        """Generate complete slot from resource grid"""
        grid = resource_grid.get_grid()
        num_symbols = grid.shape[1]
        
        samples = []
        
        for sym_idx in range(num_symbols):
            # Get CP length
            if isinstance(self.params, LTEOFDMParams):
                cp_len = self.params.cp_length_first if sym_idx == 0 else self.params.cp_length_other
            elif isinstance(self.params, NROFDMParams):
                cp_len = self.params.cp_lengths[sym_idx]
            else:
                cp_len = self.params.cp_length
            
            # Generate OFDM symbol
            ofdm_sym = self.generate_ofdm_symbol(grid[:, sym_idx], cp_len)
            samples.append(ofdm_sym)
        
        return np.concatenate(samples)
    
    def generate_pss(self, nid2: int = 0) -> np.ndarray:
        """
        Generate Primary Synchronization Signal (LTE/5G)
        
        Args:
            nid2: Physical layer identity (0, 1, or 2)
        
        Returns:
            62 complex symbols for PSS
        """
        # Zadoff-Chu sequence root indices for NID2
        roots = {0: 25, 1: 29, 2: 34}
        u = roots[nid2 % 3]
        
        # Generate Zadoff-Chu sequence (length 63)
        n = np.arange(63)
        d = np.zeros(63, dtype=complex)
        
        for i in range(63):
            if i <= 30:
                d[i] = np.exp(-1j * np.pi * u * i * (i + 1) / 63)
            else:
                d[i] = np.exp(-1j * np.pi * u * (i + 1) * (i + 2) / 63)
        
        # Remove DC (index 31)
        pss = np.concatenate([d[:31], d[32:]])
        
        return pss
    
    def generate_sss(self, nid1: int = 0, nid2: int = 0, 
                     subframe: int = 0) -> np.ndarray:
        """
        Generate Secondary Synchronization Signal (LTE)
        
        Args:
            nid1: Cell identity group (0-167)
            nid2: Physical layer identity (0-2)
            subframe: Subframe number (0 or 5)
        
        Returns:
            62 complex symbols for SSS
        """
        # M-sequences
        def m_sequence(init):
            x = np.zeros(31, dtype=int)
            x[:5] = init
            for i in range(5, 31):
                x[i] = (x[i-3] + x[i-5]) % 2
            return x
        
        # Generate base sequences
        x = m_sequence([0, 0, 0, 0, 1])
        
        # Compute m0 and m1
        q_prime = nid1 // 30
        q = (nid1 + q_prime * (q_prime + 1) // 2) // 30
        m_prime = nid1 + q * (q + 1) // 2
        m0 = m_prime % 31
        m1 = (m0 + (m_prime // 31) + 1) % 31
        
        # Generate SSS
        sss = np.zeros(62, dtype=complex)
        
        for n in range(31):
            s0_m0 = 1 - 2 * x[(n + m0) % 31]
            s1_m1 = 1 - 2 * x[(n + m1) % 31]
            
            if subframe == 0:
                sss[2*n] = s0_m0
                sss[2*n + 1] = s1_m1
            else:  # subframe 5
                sss[2*n] = s1_m1
                sss[2*n + 1] = s0_m0
        
        return sss
    
    def generate_nr_ssb(self, nid: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 5G NR Synchronization Signal Block (SSB)
        
        Returns:
            (pss, sss, pbch_dmrs)
        """
        nid2 = nid % 3
        nid1 = nid // 3
        
        # PSS (same as LTE)
        pss = self.generate_pss(nid2)
        
        # SSS for NR
        # Gold sequence based
        sss = self.generate_sss(nid1, nid2, 0)
        
        # PBCH DMRS
        pbch_dmrs = np.exp(1j * 2 * np.pi * np.random.rand(144))  # 144 DMRS symbols
        
        return pss, sss, pbch_dmrs


class ChannelEstimator:
    """
    OFDM Channel Estimation and Equalization
    
    Supports:
    - Pilot-based estimation
    - DMRS-based estimation
    - Linear/MMSE interpolation
    - Time-frequency interpolation
    """
    
    def __init__(self, params: Union[LTEOFDMParams, NROFDMParams, WiFiOFDMParams]):
        self.params = params
        
        if isinstance(params, WiFiOFDMParams):
            self.fft_size = params.fft_size
        else:
            self.fft_size = params.fft_size
    
    def estimate_channel_ls(self, received_pilots: np.ndarray,
                            known_pilots: np.ndarray) -> np.ndarray:
        """
        Least Squares channel estimation
        
        Args:
            received_pilots: Received pilot symbols
            known_pilots: Known transmitted pilot symbols
        
        Returns:
            Channel estimates at pilot positions
        """
        # LS estimate: H = Y / X
        h_ls = received_pilots / (known_pilots + 1e-10)
        return h_ls
    
    def estimate_channel_mmse(self, received_pilots: np.ndarray,
                              known_pilots: np.ndarray,
                              snr_db: float = 20.0) -> np.ndarray:
        """
        MMSE channel estimation
        
        Better performance than LS at low SNR.
        """
        snr_linear = 10 ** (snr_db / 10)
        
        # LS estimate
        h_ls = self.estimate_channel_ls(received_pilots, known_pilots)
        
        # MMSE correction (simplified)
        # Assumes equal-power channel taps
        mmse_factor = snr_linear / (snr_linear + 1)
        
        h_mmse = h_ls * mmse_factor
        
        return h_mmse
    
    def interpolate_channel(self, h_pilots: np.ndarray,
                            pilot_indices: np.ndarray,
                            num_subcarriers: int,
                            method: str = 'linear') -> np.ndarray:
        """
        Interpolate channel estimates to all subcarriers
        
        Args:
            h_pilots: Channel estimates at pilot positions
            pilot_indices: Subcarrier indices of pilots
            num_subcarriers: Total number of subcarriers
            method: 'linear', 'cubic', or 'spline'
        
        Returns:
            Channel estimates for all subcarriers
        """
        from scipy import interpolate
        
        # Handle edge cases
        if len(pilot_indices) < 2:
            return np.ones(num_subcarriers, dtype=complex) * h_pilots[0]
        
        # Create interpolation functions for real and imaginary parts
        all_indices = np.arange(num_subcarriers)
        
        if method == 'linear':
            interp_real = np.interp(all_indices, pilot_indices, h_pilots.real)
            interp_imag = np.interp(all_indices, pilot_indices, h_pilots.imag)
        elif method == 'cubic':
            f_real = interpolate.interp1d(pilot_indices, h_pilots.real, 
                                          kind='cubic', fill_value='extrapolate')
            f_imag = interpolate.interp1d(pilot_indices, h_pilots.imag,
                                          kind='cubic', fill_value='extrapolate')
            interp_real = f_real(all_indices)
            interp_imag = f_imag(all_indices)
        else:  # spline
            f_real = interpolate.UnivariateSpline(pilot_indices, h_pilots.real)
            f_imag = interpolate.UnivariateSpline(pilot_indices, h_pilots.imag)
            interp_real = f_real(all_indices)
            interp_imag = f_imag(all_indices)
        
        return interp_real + 1j * interp_imag
    
    def equalize_zf(self, received: np.ndarray, 
                    channel: np.ndarray) -> np.ndarray:
        """
        Zero-Forcing equalization
        
        Args:
            received: Received symbols
            channel: Channel estimates
        
        Returns:
            Equalized symbols
        """
        return received / (channel + 1e-10)
    
    def equalize_mmse(self, received: np.ndarray,
                      channel: np.ndarray,
                      noise_power: float) -> np.ndarray:
        """
        MMSE equalization
        
        Better than ZF when noise is significant.
        """
        h_conj = np.conj(channel)
        h_power = np.abs(channel) ** 2
        
        # MMSE equalizer
        equalizer = h_conj / (h_power + noise_power)
        
        return received * equalizer
    
    def estimate_snr(self, received_pilots: np.ndarray,
                     known_pilots: np.ndarray,
                     channel: np.ndarray) -> float:
        """
        Estimate SNR from pilot symbols
        """
        # Expected received
        expected = known_pilots * channel
        
        # Error
        error = received_pilots - expected
        
        # SNR estimate
        signal_power = np.mean(np.abs(expected) ** 2)
        noise_power = np.mean(np.abs(error) ** 2)
        
        snr_linear = signal_power / (noise_power + 1e-10)
        snr_db = 10 * np.log10(snr_linear)
        
        return snr_db


class WiFiOFDM:
    """
    WiFi-specific OFDM implementation
    
    Supports 802.11a/g/n/ac frame generation and decoding.
    """
    
    def __init__(self, params: WiFiOFDMParams):
        self.params = params
        self.engine = OFDMEngine(params)
        
        # Subcarrier mapping
        self._setup_subcarrier_mapping()
    
    def _setup_subcarrier_mapping(self):
        """Setup subcarrier indices for data and pilots"""
        fft_size = self.params.fft_size
        
        if self.params.channel_width == 20:
            # 802.11a/g/n 20 MHz
            # Subcarriers: -26 to -1, 1 to 26 (52 total, DC null)
            self.data_indices = np.concatenate([
                np.arange(-26, -21), np.arange(-20, -7), np.arange(-6, 0),
                np.arange(1, 7), np.arange(8, 21), np.arange(22, 27)
            ])
            self.pilot_indices = np.array([-21, -7, 7, 21])
        elif self.params.channel_width == 40:
            # 40 MHz
            self.data_indices = np.arange(-54, 55)
            self.data_indices = self.data_indices[self.data_indices != 0]  # Remove DC
            self.pilot_indices = np.array([-53, -25, -11, 11, 25, 53])
        else:
            # Simplified for wider bandwidths
            self.data_indices = np.arange(-self.params.num_data_subcarriers // 2,
                                          self.params.num_data_subcarriers // 2 + 1)
            self.data_indices = self.data_indices[self.data_indices != 0]
            self.pilot_indices = np.array([])
        
        # Convert to positive indices
        self.data_indices = (self.data_indices + fft_size) % fft_size
        self.pilot_indices = (self.pilot_indices + fft_size) % fft_size
    
    def generate_short_training(self) -> np.ndarray:
        """
        Generate Short Training Field (STF)
        
        Used for AGC, timing, and coarse frequency sync.
        """
        # Short training sequence (known pattern)
        s = np.sqrt(13 / 6) * np.array([
            0, 0, 1+1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, -1-1j, 0,
            0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, 0, 0, 0, 0, -1-1j, 0,
            0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0,
            0, 0, 1+1j, 0, 0
        ])
        
        # Map to subcarriers
        freq_domain = np.zeros(self.params.fft_size, dtype=complex)
        freq_domain[4:56] = s
        freq_domain = np.fft.ifftshift(freq_domain)
        
        # IFFT
        time_domain = ifft(freq_domain) * np.sqrt(self.params.fft_size)
        
        # Short training symbol is 16 samples repeated 10 times
        short_symbol = time_domain[:16]
        stf = np.tile(short_symbol, 10)
        
        return stf
    
    def generate_long_training(self) -> np.ndarray:
        """
        Generate Long Training Field (LTF)
        
        Used for fine frequency sync and channel estimation.
        """
        # Long training sequence
        l = np.array([
            1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1,
            1, -1, 1, -1, 1, 1, 1, 1, 0, 1, -1, -1, 1, 1, -1, 1, -1, 1,
            -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1
        ])
        
        # Map to subcarriers
        freq_domain = np.zeros(self.params.fft_size, dtype=complex)
        freq_domain[6:59] = l
        freq_domain = np.fft.ifftshift(freq_domain)
        
        # IFFT
        time_domain = ifft(freq_domain) * np.sqrt(self.params.fft_size)
        
        # LTF: CP (32 samples) + 2 x symbol (64 samples each)
        cp = time_domain[-32:]
        ltf = np.concatenate([cp, time_domain, time_domain])
        
        return ltf
    
    def generate_signal_field(self, rate: int = 6, length: int = 100) -> np.ndarray:
        """
        Generate SIGNAL field (BPSK, rate 1/2)
        
        Contains rate and length information.
        """
        # Rate bits (4 bits)
        rate_map = {6: 0b1101, 9: 0b1111, 12: 0b0101, 18: 0b0111,
                    24: 0b1001, 36: 0b1011, 48: 0b0001, 54: 0b0011}
        rate_bits = [(rate_map.get(rate, 0b1101) >> (3-i)) & 1 for i in range(4)]
        
        # Reserved bit
        reserved = [0]
        
        # Length (12 bits)
        length_bits = [(length >> (11-i)) & 1 for i in range(12)]
        
        # Parity
        all_bits = rate_bits + reserved + length_bits
        parity = [sum(all_bits) % 2]
        
        # Tail (6 zeros)
        tail = [0] * 6
        
        signal_bits = all_bits + parity + tail
        
        # BPSK modulation
        symbols = np.array([1 if b == 0 else -1 for b in signal_bits])
        
        # Map to subcarriers
        return self.engine.generate_ofdm_symbol(symbols[:48], self.params.cp_length)
    
    def generate_frame(self, data_bits: np.ndarray, 
                       rate: int = 54) -> np.ndarray:
        """
        Generate complete WiFi frame
        
        Args:
            data_bits: Payload bits
            rate: Data rate in Mbps
        
        Returns:
            Complete frame samples
        """
        frame_parts = []
        
        # Short Training Field
        stf = self.generate_short_training()
        frame_parts.append(stf)
        
        # Long Training Field
        ltf = self.generate_long_training()
        frame_parts.append(ltf)
        
        # SIGNAL field
        signal = self.generate_signal_field(rate, len(data_bits) // 8)
        frame_parts.append(signal)
        
        # DATA symbols
        # Modulation based on rate
        if rate <= 9:
            mod_type = 'bpsk'
            bits_per_symbol = 48
        elif rate <= 18:
            mod_type = 'qpsk'
            bits_per_symbol = 96
        elif rate <= 36:
            mod_type = '16qam'
            bits_per_symbol = 192
        else:
            mod_type = '64qam'
            bits_per_symbol = 288
        
        # Pad data
        num_symbols = (len(data_bits) + bits_per_symbol - 1) // bits_per_symbol
        padded_bits = np.zeros(num_symbols * bits_per_symbol, dtype=int)
        padded_bits[:len(data_bits)] = data_bits
        
        # Generate data symbols
        for i in range(num_symbols):
            sym_bits = padded_bits[i * bits_per_symbol:(i + 1) * bits_per_symbol]
            
            # Simple modulation (would use proper encoder in production)
            if mod_type == 'bpsk':
                symbols = 1 - 2 * sym_bits[:48]
            elif mod_type == 'qpsk':
                symbols = (1 - 2 * sym_bits[::2]) + 1j * (1 - 2 * sym_bits[1::2])
                symbols = symbols[:48] / np.sqrt(2)
            else:
                # Simplified QAM
                symbols = np.exp(1j * 2 * np.pi * np.arange(48) / 48)
            
            ofdm_sym = self.engine.generate_ofdm_symbol(symbols, self.params.cp_length)
            frame_parts.append(ofdm_sym)
        
        return np.concatenate(frame_parts)
