#!/usr/bin/env python3
"""
RF Arsenal OS - LTE/5G FPGA Accelerator
Hardware-accelerated LTE/5G signal processing

Implements:
- OFDM modulation/demodulation with hardware IFFT/FFT
- LTE/5G resource element mapping
- Channel estimation and equalization
- Physical channel processing
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class Numerology(IntEnum):
    """5G NR numerology (subcarrier spacing)"""
    MU_0 = 0  # 15 kHz (LTE compatible)
    MU_1 = 1  # 30 kHz
    MU_2 = 2  # 60 kHz
    MU_3 = 3  # 120 kHz
    MU_4 = 4  # 240 kHz


class CyclicPrefixType(Enum):
    """Cyclic prefix types"""
    NORMAL = "normal"
    EXTENDED = "extended"


class ChannelType(Enum):
    """Physical channel types"""
    # Downlink
    PDSCH = "pdsch"  # Physical Downlink Shared Channel
    PDCCH = "pdcch"  # Physical Downlink Control Channel
    PBCH = "pbch"  # Physical Broadcast Channel
    # Uplink
    PUSCH = "pusch"  # Physical Uplink Shared Channel
    PUCCH = "pucch"  # Physical Uplink Control Channel
    PRACH = "prach"  # Physical Random Access Channel
    # Signals
    PSS = "pss"  # Primary Synchronization Signal
    SSS = "sss"  # Secondary Synchronization Signal
    DMRS = "dmrs"  # Demodulation Reference Signal
    PTRS = "ptrs"  # Phase Tracking Reference Signal
    CSI_RS = "csi_rs"  # CSI Reference Signal


@dataclass
class OFDMConfig:
    """OFDM configuration for LTE/5G"""
    # FFT size based on bandwidth
    nfft: int = 2048
    
    # Cyclic prefix
    cp_type: CyclicPrefixType = CyclicPrefixType.NORMAL
    
    # Numerology (5G NR)
    numerology: Numerology = Numerology.MU_0
    
    # Subcarrier spacing (derived from numerology)
    @property
    def subcarrier_spacing_hz(self) -> int:
        return 15000 * (2 ** self.numerology)
    
    # Sample rate (derived)
    @property
    def sample_rate(self) -> float:
        return self.nfft * self.subcarrier_spacing_hz
    
    # CP lengths (in samples)
    def get_cp_length(self, symbol_index: int) -> int:
        """Get CP length for symbol index"""
        if self.cp_type == CyclicPrefixType.EXTENDED:
            return self.nfft // 4
        else:
            # Normal CP: first symbol in slot has longer CP
            if symbol_index % 7 == 0:
                return self.nfft // 8 + self.nfft // 128
            else:
                return self.nfft // 8
    
    # Symbols per slot
    @property
    def symbols_per_slot(self) -> int:
        return 14 if self.cp_type == CyclicPrefixType.NORMAL else 12
    
    # Slots per subframe
    @property
    def slots_per_subframe(self) -> int:
        return 2 ** self.numerology


@dataclass
class ChannelConfig:
    """Physical channel configuration"""
    channel_type: ChannelType = ChannelType.PDSCH
    
    # Resource allocation
    start_prb: int = 0
    num_prb: int = 50
    start_symbol: int = 0
    num_symbols: int = 14
    
    # Modulation
    modulation: str = "qpsk"  # qpsk, 16qam, 64qam, 256qam
    
    # Layer mapping
    num_layers: int = 1
    
    # DMRS configuration
    dmrs_positions: List[int] = field(default_factory=lambda: [2, 11])
    dmrs_additional_positions: int = 0
    
    def get_re_count(self) -> int:
        """Get total resource elements"""
        return self.num_prb * 12 * self.num_symbols


@dataclass
class LTEFrameConfig:
    """LTE/5G frame configuration"""
    # System bandwidth
    bandwidth_mhz: float = 20.0
    
    # Number of resource blocks
    num_prb: int = 100
    
    # Cell ID
    cell_id: int = 0
    
    # Antenna configuration
    num_tx_antennas: int = 1
    num_rx_antennas: int = 1
    
    # Duplex mode
    duplex_mode: str = "fdd"  # fdd or tdd
    
    # TDD configuration (if TDD)
    tdd_config: int = 0
    
    # Frame structure
    frame_type: str = "lte"  # lte or nr
    
    @property
    def nfft(self) -> int:
        """Get FFT size based on bandwidth"""
        bandwidth_to_nfft = {
            1.4: 128,
            3: 256,
            5: 512,
            10: 1024,
            15: 1536,
            20: 2048,
            40: 4096,
            50: 4096,
            100: 8192,
        }
        return bandwidth_to_nfft.get(self.bandwidth_mhz, 2048)
    
    @property
    def subcarriers_per_prb(self) -> int:
        return 12


class LTEAccelerator:
    """
    Hardware-accelerated LTE/5G signal processing
    
    Provides:
    - OFDM modulation/demodulation
    - Resource element mapping
    - Reference signal generation
    - Channel estimation
    - Physical channel processing
    """
    
    # LTE/5G parameters
    SUBCARRIERS_PER_PRB = 12
    MAX_PRB = 275  # Maximum for 100 MHz in NR
    
    # Bandwidth to PRB mapping
    BANDWIDTH_TO_PRB = {
        1.4: 6, 3: 15, 5: 25, 10: 50, 15: 75, 20: 100,
        40: 106, 50: 133, 100: 275,  # NR FR1
    }
    
    def __init__(
        self,
        fpga_controller: Any,
        dsp_accelerator: Any = None
    ):
        """
        Initialize LTE Accelerator
        
        Args:
            fpga_controller: FPGAController instance
            dsp_accelerator: Optional DSPAccelerator for FFT
        """
        self._fpga = fpga_controller
        self._dsp = dsp_accelerator
        
        # Current configuration
        self._ofdm_config = OFDMConfig()
        self._frame_config = LTEFrameConfig()
        
        # Pre-computed tables
        self._pss_sequences: Dict[int, np.ndarray] = {}
        self._sss_sequences: Dict[Tuple[int, int], np.ndarray] = {}
        self._dmrs_sequences: Dict[int, np.ndarray] = {}
        
        # Resource grid buffer
        self._resource_grid: Optional[np.ndarray] = None
        
        # Statistics
        self._stats = {
            'symbols_generated': 0,
            'symbols_received': 0,
            'channel_estimates': 0,
        }
        
        logger.info("LTEAccelerator initialized")
    
    @property
    def ofdm_config(self) -> OFDMConfig:
        return self._ofdm_config
    
    @property
    def frame_config(self) -> LTEFrameConfig:
        return self._frame_config
    
    # =========================================================================
    # Configuration
    # =========================================================================
    
    def configure_ofdm(self, config: OFDMConfig) -> bool:
        """Configure OFDM parameters"""
        self._ofdm_config = config
        
        # Configure FPGA if available
        if self._fpga and self._fpga.is_configured:
            return self._fpga.configure_lte_ofdm(
                nfft=config.nfft,
                cp_type=config.cp_type.value,
                num_prb=self._frame_config.num_prb,
                subcarrier_spacing_khz=config.subcarrier_spacing_hz // 1000
            )
        
        return True
    
    def configure_frame(self, config: LTEFrameConfig) -> bool:
        """Configure frame parameters"""
        self._frame_config = config
        
        # Update OFDM config
        self._ofdm_config.nfft = config.nfft
        
        # Generate reference signals
        self._generate_pss_sequences()
        self._generate_sss_sequences()
        
        # Allocate resource grid
        self._allocate_resource_grid()
        
        return True
    
    def _allocate_resource_grid(self) -> None:
        """Allocate resource grid buffer"""
        num_subcarriers = self._frame_config.num_prb * self.SUBCARRIERS_PER_PRB
        num_symbols = self._ofdm_config.symbols_per_slot
        num_layers = self._frame_config.num_tx_antennas
        
        self._resource_grid = np.zeros(
            (num_layers, num_symbols, num_subcarriers),
            dtype=complex
        )
    
    # =========================================================================
    # OFDM Modulation
    # =========================================================================
    
    def generate_ofdm_symbol(
        self,
        frequency_data: np.ndarray,
        symbol_index: int = 0
    ) -> np.ndarray:
        """
        Generate OFDM symbol with cyclic prefix
        
        Args:
            frequency_data: Frequency domain data
            symbol_index: Symbol index for CP length
            
        Returns:
            Time domain OFDM symbol
        """
        nfft = self._ofdm_config.nfft
        
        # Create frequency domain frame
        freq_frame = np.zeros(nfft, dtype=complex)
        
        # Map data to subcarriers (center around DC)
        num_data = len(frequency_data)
        half_data = num_data // 2
        
        # Lower subcarriers (negative frequencies)
        freq_frame[nfft - half_data:] = frequency_data[:half_data]
        # Upper subcarriers (positive frequencies)
        freq_frame[1:half_data + 1] = frequency_data[half_data:]
        
        # IFFT (use hardware if available)
        if self._dsp:
            time_data = self._dsp.ifft(freq_frame)
        else:
            time_data = np.fft.ifft(freq_frame)
        
        # Add cyclic prefix
        cp_length = self._ofdm_config.get_cp_length(symbol_index)
        cp = time_data[-cp_length:]
        
        ofdm_symbol = np.concatenate([cp, time_data])
        
        self._stats['symbols_generated'] += 1
        return ofdm_symbol
    
    def demodulate_ofdm_symbol(
        self,
        time_data: np.ndarray,
        symbol_index: int = 0
    ) -> np.ndarray:
        """
        Demodulate OFDM symbol
        
        Args:
            time_data: Time domain signal
            symbol_index: Symbol index for CP removal
            
        Returns:
            Frequency domain data
        """
        nfft = self._ofdm_config.nfft
        cp_length = self._ofdm_config.get_cp_length(symbol_index)
        
        # Remove cyclic prefix
        data_start = cp_length
        time_symbol = time_data[data_start:data_start + nfft]
        
        # FFT (use hardware if available)
        if self._dsp:
            freq_data = self._dsp.fft(time_symbol)
        else:
            freq_data = np.fft.fft(time_symbol)
        
        # Extract data subcarriers
        num_prb = self._frame_config.num_prb
        num_data = num_prb * self.SUBCARRIERS_PER_PRB
        half_data = num_data // 2
        
        # Extract lower and upper subcarriers
        lower = freq_data[nfft - half_data:]
        upper = freq_data[1:half_data + 1]
        
        output = np.concatenate([lower, upper])
        
        self._stats['symbols_received'] += 1
        return output
    
    def generate_slot(
        self,
        resource_grid: np.ndarray,
        add_cp: bool = True
    ) -> np.ndarray:
        """
        Generate complete OFDM slot
        
        Args:
            resource_grid: Resource grid (symbols x subcarriers)
            add_cp: Whether to add cyclic prefix
            
        Returns:
            Time domain slot signal
        """
        num_symbols = resource_grid.shape[0]
        slot_signal = []
        
        for symbol_idx in range(num_symbols):
            freq_data = resource_grid[symbol_idx, :]
            ofdm_symbol = self.generate_ofdm_symbol(freq_data, symbol_idx)
            slot_signal.append(ofdm_symbol)
        
        return np.concatenate(slot_signal)
    
    def demodulate_slot(
        self,
        time_signal: np.ndarray
    ) -> np.ndarray:
        """
        Demodulate complete OFDM slot
        
        Args:
            time_signal: Time domain slot signal
            
        Returns:
            Resource grid (symbols x subcarriers)
        """
        num_symbols = self._ofdm_config.symbols_per_slot
        nfft = self._ofdm_config.nfft
        num_subcarriers = self._frame_config.num_prb * self.SUBCARRIERS_PER_PRB
        
        resource_grid = np.zeros((num_symbols, num_subcarriers), dtype=complex)
        
        sample_offset = 0
        for symbol_idx in range(num_symbols):
            cp_length = self._ofdm_config.get_cp_length(symbol_idx)
            symbol_length = cp_length + nfft
            
            symbol_data = time_signal[sample_offset:sample_offset + symbol_length]
            resource_grid[symbol_idx, :] = self.demodulate_ofdm_symbol(
                symbol_data, symbol_idx
            )
            
            sample_offset += symbol_length
        
        return resource_grid
    
    # =========================================================================
    # Resource Element Mapping
    # =========================================================================
    
    def map_to_resource_elements(
        self,
        data: np.ndarray,
        config: ChannelConfig
    ) -> np.ndarray:
        """
        Map data to resource elements
        
        Args:
            data: Data symbols to map
            config: Channel configuration
            
        Returns:
            Resource grid with mapped data
        """
        if self._resource_grid is None:
            self._allocate_resource_grid()
        
        grid = self._resource_grid[0].copy()  # First layer
        
        # Calculate RE positions
        start_sc = config.start_prb * self.SUBCARRIERS_PER_PRB
        end_sc = start_sc + config.num_prb * self.SUBCARRIERS_PER_PRB
        
        # Map data, avoiding DMRS positions
        data_idx = 0
        for symbol_idx in range(config.start_symbol, 
                                config.start_symbol + config.num_symbols):
            if symbol_idx in config.dmrs_positions:
                continue  # Skip DMRS symbols
            
            for sc_idx in range(start_sc, end_sc):
                if data_idx < len(data):
                    grid[symbol_idx, sc_idx] = data[data_idx]
                    data_idx += 1
        
        return grid
    
    def extract_from_resource_elements(
        self,
        grid: np.ndarray,
        config: ChannelConfig
    ) -> np.ndarray:
        """
        Extract data from resource elements
        
        Args:
            grid: Resource grid
            config: Channel configuration
            
        Returns:
            Extracted data symbols
        """
        start_sc = config.start_prb * self.SUBCARRIERS_PER_PRB
        end_sc = start_sc + config.num_prb * self.SUBCARRIERS_PER_PRB
        
        data = []
        for symbol_idx in range(config.start_symbol,
                                config.start_symbol + config.num_symbols):
            if symbol_idx in config.dmrs_positions:
                continue
            
            for sc_idx in range(start_sc, end_sc):
                data.append(grid[symbol_idx, sc_idx])
        
        return np.array(data)
    
    # =========================================================================
    # Reference Signals
    # =========================================================================
    
    def _generate_pss_sequences(self) -> None:
        """Generate Primary Synchronization Signals"""
        for n_id_2 in range(3):
            # ZC sequence for PSS
            root = [25, 29, 34][n_id_2]
            length = 127
            
            n = np.arange(length)
            seq = np.exp(-1j * np.pi * root * n * (n + 1) / length)
            
            self._pss_sequences[n_id_2] = seq
    
    def _generate_sss_sequences(self) -> None:
        """Generate Secondary Synchronization Signals"""
        # Simplified SSS generation
        for n_id_1 in range(168):
            for n_id_2 in range(3):
                cell_id = 3 * n_id_1 + n_id_2
                
                # Generate m-sequences
                m0 = (15 * (n_id_1 // 112) + 5 * n_id_2) % 127
                m1 = (n_id_1 % 112 + n_id_2 * 2 + 1) % 127
                
                # Create SSS sequence
                length = 127
                seq = np.zeros(length, dtype=complex)
                for n in range(length):
                    d0 = 1 - 2 * self._m_sequence((n + m0) % 127)
                    d1 = 1 - 2 * self._m_sequence((n + m1) % 127)
                    seq[n] = d0 + 1j * d1
                
                self._sss_sequences[(n_id_1, n_id_2)] = seq / np.sqrt(2)
    
    def _m_sequence(self, n: int) -> int:
        """Generate m-sequence value"""
        # Simplified m-sequence generator
        x = [1, 0, 0, 0, 0, 0, 0]
        for _ in range(n):
            new_bit = x[0] ^ x[6]
            x = x[1:] + [new_bit]
        return x[0]
    
    def generate_dmrs(
        self,
        config: ChannelConfig,
        slot_number: int = 0
    ) -> np.ndarray:
        """
        Generate Demodulation Reference Signal
        
        Args:
            config: Channel configuration
            slot_number: Slot number for scrambling
            
        Returns:
            DMRS symbols
        """
        # DMRS sequence generation
        cell_id = self._frame_config.cell_id
        num_subcarriers = config.num_prb * self.SUBCARRIERS_PER_PRB
        
        # Generate gold sequence
        c_init = ((2**17 * (14 * slot_number + 0 + 1) * (2 * cell_id + 1) + 
                   2 * cell_id) % (2**31))
        
        # Create QPSK symbols
        dmrs = np.zeros(num_subcarriers // 2, dtype=complex)
        
        for i in range(len(dmrs)):
            # Pseudo-random bits
            c1 = (c_init >> (2 * i)) & 1
            c2 = (c_init >> (2 * i + 1)) & 1
            
            dmrs[i] = ((1 - 2 * c1) + 1j * (1 - 2 * c2)) / np.sqrt(2)
        
        return dmrs
    
    def generate_pss(self, n_id_2: int = 0) -> np.ndarray:
        """Get PSS for given N_ID_2"""
        if n_id_2 not in self._pss_sequences:
            self._generate_pss_sequences()
        return self._pss_sequences[n_id_2]
    
    def generate_sss(
        self,
        n_id_1: int = 0,
        n_id_2: int = 0
    ) -> np.ndarray:
        """Get SSS for given N_ID_1 and N_ID_2"""
        if (n_id_1, n_id_2) not in self._sss_sequences:
            self._generate_sss_sequences()
        return self._sss_sequences[(n_id_1, n_id_2)]
    
    # =========================================================================
    # Channel Estimation
    # =========================================================================
    
    def estimate_channel(
        self,
        rx_grid: np.ndarray,
        config: ChannelConfig
    ) -> np.ndarray:
        """
        Estimate channel using DMRS
        
        Args:
            rx_grid: Received resource grid
            config: Channel configuration
            
        Returns:
            Channel estimates for data symbols
        """
        self._stats['channel_estimates'] += 1
        
        # Extract DMRS positions
        start_sc = config.start_prb * self.SUBCARRIERS_PER_PRB
        end_sc = start_sc + config.num_prb * self.SUBCARRIERS_PER_PRB
        
        # Get expected DMRS
        expected_dmrs = self.generate_dmrs(config)
        
        # Extract received DMRS and estimate channel
        channel_estimates = []
        
        for dmrs_symbol in config.dmrs_positions:
            rx_dmrs = rx_grid[dmrs_symbol, start_sc:end_sc:2]  # DMRS on every other SC
            
            # LS channel estimation
            h_estimate = rx_dmrs / expected_dmrs[:len(rx_dmrs)]
            
            # Interpolate to all subcarriers
            h_interp = self._interpolate_channel(h_estimate, 2)
            channel_estimates.append(h_interp)
        
        # Average estimates from different DMRS symbols
        if channel_estimates:
            h_avg = np.mean(channel_estimates, axis=0)
        else:
            h_avg = np.ones(end_sc - start_sc, dtype=complex)
        
        return h_avg
    
    def equalize(
        self,
        rx_data: np.ndarray,
        channel_estimate: np.ndarray
    ) -> np.ndarray:
        """
        Equalize received data using channel estimate
        
        Args:
            rx_data: Received data symbols
            channel_estimate: Channel estimate
            
        Returns:
            Equalized data
        """
        # Zero-forcing equalization
        h = channel_estimate[:len(rx_data)]
        h = np.where(np.abs(h) < 1e-10, 1e-10, h)  # Avoid division by zero
        
        return rx_data / h
    
    def _interpolate_channel(
        self,
        h_estimate: np.ndarray,
        factor: int
    ) -> np.ndarray:
        """Interpolate channel estimate"""
        # Linear interpolation
        output = np.zeros(len(h_estimate) * factor, dtype=complex)
        
        for i in range(len(h_estimate)):
            output[i * factor] = h_estimate[i]
            
            if i < len(h_estimate) - 1:
                # Linear interpolation
                for j in range(1, factor):
                    alpha = j / factor
                    output[i * factor + j] = (1 - alpha) * h_estimate[i] + \
                                             alpha * h_estimate[i + 1]
        
        return output
    
    # =========================================================================
    # Physical Channel Processing
    # =========================================================================
    
    def process_pdsch(
        self,
        data_bits: np.ndarray,
        config: ChannelConfig
    ) -> np.ndarray:
        """
        Process Physical Downlink Shared Channel
        
        Args:
            data_bits: Input data bits
            config: Channel configuration
            
        Returns:
            Resource grid with PDSCH
        """
        # Modulate data
        symbols = self._modulate(data_bits, config.modulation)
        
        # Layer mapping (simplified)
        if config.num_layers > 1:
            symbols = self._layer_map(symbols, config.num_layers)
        
        # Map to resource elements
        grid = self.map_to_resource_elements(symbols, config)
        
        # Add DMRS
        dmrs = self.generate_dmrs(config)
        for dmrs_pos in config.dmrs_positions:
            start_sc = config.start_prb * self.SUBCARRIERS_PER_PRB
            grid[dmrs_pos, start_sc::2] = dmrs
        
        return grid
    
    def process_pusch(
        self,
        data_bits: np.ndarray,
        config: ChannelConfig
    ) -> np.ndarray:
        """Process Physical Uplink Shared Channel"""
        # Similar to PDSCH but with different DMRS pattern
        return self.process_pdsch(data_bits, config)
    
    def receive_pdsch(
        self,
        rx_grid: np.ndarray,
        config: ChannelConfig
    ) -> np.ndarray:
        """
        Receive and decode PDSCH
        
        Args:
            rx_grid: Received resource grid
            config: Channel configuration
            
        Returns:
            Decoded data bits
        """
        # Channel estimation
        h_estimate = self.estimate_channel(rx_grid, config)
        
        # Extract data
        rx_data = self.extract_from_resource_elements(rx_grid, config)
        
        # Equalize
        eq_data = self.equalize(rx_data, h_estimate)
        
        # Demodulate
        bits = self._demodulate(eq_data, config.modulation)
        
        return bits
    
    def _modulate(
        self,
        bits: np.ndarray,
        modulation: str
    ) -> np.ndarray:
        """Modulate bits to symbols"""
        if self._dsp:
            return self._dsp.modulate_qam(bits)
        
        # Fallback implementation
        if modulation == 'qpsk':
            bits = bits[:len(bits) // 2 * 2].reshape(-1, 2)
            symbols = np.zeros(len(bits), dtype=complex)
            for i, (b0, b1) in enumerate(bits):
                symbols[i] = ((1 - 2*b0) + 1j*(1 - 2*b1)) / np.sqrt(2)
            return symbols
        
        # Add more modulation schemes as needed
        return bits.astype(complex)
    
    def _demodulate(
        self,
        symbols: np.ndarray,
        modulation: str
    ) -> np.ndarray:
        """Demodulate symbols to bits"""
        if self._dsp:
            return self._dsp.demodulate_qam(symbols)
        
        # Fallback QPSK demodulation
        bits = np.zeros(len(symbols) * 2, dtype=np.uint8)
        bits[0::2] = (np.real(symbols) < 0).astype(np.uint8)
        bits[1::2] = (np.imag(symbols) < 0).astype(np.uint8)
        return bits
    
    def _layer_map(
        self,
        symbols: np.ndarray,
        num_layers: int
    ) -> np.ndarray:
        """Layer mapping for MIMO"""
        # Simple layer mapping
        layer_symbols = np.zeros((num_layers, len(symbols) // num_layers), dtype=complex)
        for layer in range(num_layers):
            layer_symbols[layer, :] = symbols[layer::num_layers]
        return layer_symbols
    
    # =========================================================================
    # Synchronization
    # =========================================================================
    
    def detect_pss(
        self,
        signal: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[Optional[int], int, float]:
        """
        Detect PSS and determine timing/cell ID
        
        Args:
            signal: Received signal
            threshold: Detection threshold
            
        Returns:
            (N_ID_2, timing offset, correlation peak)
        """
        best_n_id_2 = None
        best_timing = 0
        best_corr = 0.0
        
        for n_id_2 in range(3):
            pss = self.generate_pss(n_id_2)
            
            # Cross-correlation
            corr = np.correlate(signal, pss, mode='valid')
            corr_mag = np.abs(corr)
            
            peak_idx = np.argmax(corr_mag)
            peak_val = corr_mag[peak_idx] / np.max(np.abs(pss))**2
            
            if peak_val > best_corr:
                best_corr = peak_val
                best_timing = peak_idx
                best_n_id_2 = n_id_2
        
        if best_corr >= threshold:
            return best_n_id_2, best_timing, best_corr
        
        return None, 0, 0.0
    
    def detect_sss(
        self,
        signal: np.ndarray,
        n_id_2: int,
        pss_timing: int,
        threshold: float = 0.5
    ) -> Tuple[Optional[int], float]:
        """
        Detect SSS and determine N_ID_1
        
        Args:
            signal: Received signal
            n_id_2: N_ID_2 from PSS detection
            pss_timing: Timing from PSS detection
            threshold: Detection threshold
            
        Returns:
            (N_ID_1, correlation peak)
        """
        best_n_id_1 = None
        best_corr = 0.0
        
        # SSS is in symbol before PSS
        sss_timing = pss_timing - (self._ofdm_config.nfft + 
                                   self._ofdm_config.get_cp_length(0))
        
        if sss_timing < 0:
            return None, 0.0
        
        # Extract SSS region
        sss_signal = signal[sss_timing:sss_timing + 127]
        
        for n_id_1 in range(168):
            sss = self.generate_sss(n_id_1, n_id_2)
            
            corr = np.abs(np.sum(sss_signal * np.conj(sss)))
            corr /= len(sss)
            
            if corr > best_corr:
                best_corr = corr
                best_n_id_1 = n_id_1
        
        if best_corr >= threshold:
            return best_n_id_1, best_corr
        
        return None, 0.0
    
    # =========================================================================
    # Status and Statistics
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get accelerator status"""
        return {
            'ofdm_config': {
                'nfft': self._ofdm_config.nfft,
                'numerology': self._ofdm_config.numerology.value,
                'cp_type': self._ofdm_config.cp_type.value,
                'sample_rate': self._ofdm_config.sample_rate,
            },
            'frame_config': {
                'bandwidth_mhz': self._frame_config.bandwidth_mhz,
                'num_prb': self._frame_config.num_prb,
                'cell_id': self._frame_config.cell_id,
            },
            'stats': self._stats.copy(),
        }
    
    def reset_stats(self) -> None:
        """Reset statistics"""
        for key in self._stats:
            self._stats[key] = 0
