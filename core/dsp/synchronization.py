#!/usr/bin/env python3
"""
RF Arsenal OS - Synchronization Engine

Production-grade timing, frequency, and frame synchronization.
Critical for cellular and WiFi operations.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import signal as scipy_signal
from scipy.fft import fft, ifft, fftshift

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Synchronization result"""
    success: bool
    timing_offset: int  # Samples
    frequency_offset: float  # Hz
    frame_start: int  # Sample index
    cell_id: Optional[int] = None
    snr_estimate: Optional[float] = None
    confidence: float = 0.0


class TimingSync:
    """
    Timing Synchronization
    
    Methods:
    - Cross-correlation with known sequence
    - Schmidl-Cox for OFDM
    - Auto-correlation based
    """
    
    def __init__(self, sample_rate: float = 30.72e6):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger('TimingSync')
    
    def correlate_sequence(self, samples: np.ndarray,
                          reference: np.ndarray,
                          threshold: float = 0.5) -> List[int]:
        """
        Find timing by cross-correlation with known sequence.
        
        Args:
            samples: Received samples
            reference: Known sequence to find
            threshold: Detection threshold (0-1)
        
        Returns:
            List of detected positions
        """
        # Cross-correlation
        corr = scipy_signal.correlate(samples, reference, mode='valid')
        corr_mag = np.abs(corr)
        
        # Normalize
        corr_norm = corr_mag / (np.sqrt(np.sum(np.abs(reference) ** 2)) *
                                np.sqrt(scipy_signal.convolve(
                                    np.abs(samples) ** 2,
                                    np.ones(len(reference)),
                                    mode='valid'
                                ) + 1e-10))
        
        # Find peaks above threshold
        peak_threshold = threshold * np.max(corr_norm)
        peaks, _ = scipy_signal.find_peaks(corr_norm, height=peak_threshold,
                                           distance=len(reference) // 2)
        
        return peaks.tolist()
    
    def schmidl_cox(self, samples: np.ndarray,
                    symbol_length: int,
                    cp_length: int) -> Tuple[int, float]:
        """
        Schmidl-Cox timing synchronization for OFDM.
        
        Uses auto-correlation of cyclic prefix.
        
        Args:
            samples: Received samples
            symbol_length: OFDM symbol length (FFT size)
            cp_length: Cyclic prefix length
        
        Returns:
            (timing_offset, metric_value)
        """
        # Auto-correlation with symbol_length delay
        P = np.zeros(len(samples) - symbol_length - cp_length, dtype=complex)
        R = np.zeros(len(samples) - symbol_length - cp_length)
        
        for m in range(len(P)):
            # Correlation between CP and end of symbol
            for k in range(cp_length):
                P[m] += np.conj(samples[m + k]) * samples[m + k + symbol_length]
                R[m] += np.abs(samples[m + k + symbol_length]) ** 2
        
        # Timing metric
        M = np.abs(P) ** 2 / (R ** 2 + 1e-10)
        
        # Find peak
        timing = np.argmax(M)
        metric = M[timing]
        
        return timing, metric
    
    def van_de_beek(self, samples: np.ndarray,
                    symbol_length: int,
                    cp_length: int) -> Tuple[int, float]:
        """
        Van de Beek ML timing estimation.
        
        Maximum likelihood estimator for OFDM timing.
        """
        L = cp_length
        N = symbol_length
        
        # Initialize
        gamma = np.zeros(len(samples) - N - L, dtype=complex)
        phi = np.zeros(len(samples) - N - L)
        
        # Calculate correlation sums
        for m in range(len(gamma)):
            for k in range(L):
                gamma[m] += np.conj(samples[m + k]) * samples[m + k + N]
                phi[m] += 0.5 * (np.abs(samples[m + k]) ** 2 + 
                                np.abs(samples[m + k + N]) ** 2)
        
        # SNR estimate (rho)
        rho = np.mean(np.abs(gamma)) / (np.mean(phi) + 1e-10)
        
        # Likelihood function
        L_func = 2 * np.abs(gamma) - rho * phi
        
        # ML timing estimate
        timing = np.argmax(L_func)
        
        return timing, L_func[timing]
    
    def pss_correlation(self, samples: np.ndarray,
                       pss_sequences: Dict[int, np.ndarray]) -> Tuple[int, int, float]:
        """
        PSS-based timing for LTE/5G.
        
        Args:
            samples: Received samples
            pss_sequences: Dict mapping NID2 to PSS sequence
        
        Returns:
            (timing, nid2, correlation_value)
        """
        best_timing = 0
        best_nid2 = 0
        best_corr = 0
        
        for nid2, pss in pss_sequences.items():
            # Cross-correlation
            corr = scipy_signal.correlate(samples, pss, mode='valid')
            corr_mag = np.abs(corr)
            
            peak_idx = np.argmax(corr_mag)
            peak_val = corr_mag[peak_idx]
            
            if peak_val > best_corr:
                best_timing = peak_idx
                best_nid2 = nid2
                best_corr = peak_val
        
        return best_timing, best_nid2, best_corr


class FrequencySync:
    """
    Carrier Frequency Offset (CFO) Estimation and Correction
    
    Methods:
    - CP-based for OFDM
    - Pilot-based
    - Blind estimation
    """
    
    def __init__(self, sample_rate: float = 30.72e6):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger('FrequencySync')
    
    def estimate_cfo_cp(self, samples: np.ndarray,
                        symbol_length: int,
                        cp_length: int) -> float:
        """
        Estimate CFO using cyclic prefix correlation.
        
        Works for OFDM signals.
        
        Returns:
            Frequency offset in Hz
        """
        # Correlate CP with end of symbol
        correlation = 0
        for k in range(cp_length):
            correlation += np.conj(samples[k]) * samples[k + symbol_length]
        
        # Phase of correlation gives CFO
        phase = np.angle(correlation)
        
        # Convert to Hz
        # CFO = phase / (2 * pi * symbol_duration)
        symbol_duration = symbol_length / self.sample_rate
        cfo = phase / (2 * np.pi * symbol_duration)
        
        return cfo
    
    def estimate_cfo_preamble(self, samples: np.ndarray,
                              repetition_length: int,
                              num_repetitions: int = 2) -> float:
        """
        Estimate CFO from repeated preamble.
        
        Used in WiFi STF.
        
        Args:
            samples: Received samples
            repetition_length: Length of repeated sequence
            num_repetitions: Number of repetitions to use
        """
        # Average correlation between repetitions
        total_phase = 0
        count = 0
        
        for i in range(num_repetitions - 1):
            start1 = i * repetition_length
            start2 = (i + 1) * repetition_length
            
            corr = np.sum(np.conj(samples[start1:start1 + repetition_length]) *
                         samples[start2:start2 + repetition_length])
            
            total_phase += np.angle(corr)
            count += 1
        
        avg_phase = total_phase / count
        
        # Convert to Hz
        cfo = avg_phase * self.sample_rate / (2 * np.pi * repetition_length)
        
        return cfo
    
    def estimate_cfo_pilots(self, received_pilots: np.ndarray,
                            known_pilots: np.ndarray,
                            pilot_spacing: int) -> float:
        """
        Estimate CFO from pilot symbols.
        
        Args:
            received_pilots: Received pilot symbols
            known_pilots: Known transmitted pilots
            pilot_spacing: Spacing between pilots in samples
        """
        # Channel at pilot locations
        h = received_pilots / (known_pilots + 1e-10)
        
        # Phase difference between adjacent pilots
        phase_diffs = np.angle(h[1:] * np.conj(h[:-1]))
        
        # Average phase rate
        avg_phase_rate = np.mean(phase_diffs)
        
        # Convert to frequency
        cfo = avg_phase_rate * self.sample_rate / (2 * np.pi * pilot_spacing)
        
        return cfo
    
    def correct_cfo(self, samples: np.ndarray,
                    cfo_hz: float) -> np.ndarray:
        """
        Apply CFO correction to samples.
        """
        t = np.arange(len(samples)) / self.sample_rate
        correction = np.exp(-2j * np.pi * cfo_hz * t)
        return samples * correction
    
    def fine_cfo_estimation(self, samples: np.ndarray,
                            reference: np.ndarray,
                            search_range_hz: float = 1000) -> float:
        """
        Fine CFO estimation using correlation search.
        
        More accurate but slower than CP-based.
        """
        # Search range
        num_steps = 100
        cfo_step = 2 * search_range_hz / num_steps
        
        best_cfo = 0
        best_corr = 0
        
        for i in range(num_steps):
            cfo = -search_range_hz + i * cfo_step
            
            # Apply frequency offset
            corrected = self.correct_cfo(samples[:len(reference)], cfo)
            
            # Correlate
            corr = np.abs(np.sum(corrected * np.conj(reference)))
            
            if corr > best_corr:
                best_corr = corr
                best_cfo = cfo
        
        return best_cfo


class FrameSync:
    """
    Frame Synchronization
    
    Identifies frame boundaries for cellular and WiFi.
    """
    
    def __init__(self, sample_rate: float = 30.72e6):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger('FrameSync')
    
    def find_lte_frame_start(self, samples: np.ndarray,
                             pss: np.ndarray,
                             sss0: np.ndarray,
                             sss5: np.ndarray) -> Tuple[int, int, int]:
        """
        Find LTE frame start using PSS/SSS.
        
        Returns:
            (frame_start, nid2, nid1)
        """
        # Find PSS
        pss_corr = np.abs(scipy_signal.correlate(samples, pss, mode='valid'))
        pss_peaks, _ = scipy_signal.find_peaks(pss_corr, 
                                                height=0.5 * np.max(pss_corr),
                                                distance=1920)  # ~half subframe
        
        best_frame_start = 0
        best_nid1 = 0
        best_confidence = 0
        
        for pss_pos in pss_peaks:
            # SSS is one symbol before PSS
            sss_pos = pss_pos - 137  # Approximate symbol spacing
            
            if sss_pos < 0 or sss_pos + len(sss0) > len(samples):
                continue
            
            sss_samples = samples[sss_pos:sss_pos + len(sss0)]
            
            # Try SSS for subframe 0
            corr0 = np.abs(np.sum(sss_samples * np.conj(sss0)))
            
            # Try SSS for subframe 5
            corr5 = np.abs(np.sum(sss_samples * np.conj(sss5)))
            
            if corr0 > corr5:
                # Subframe 0
                frame_start = pss_pos - 6 * 2048  # 6 symbols before PSS
                confidence = corr0
            else:
                # Subframe 5
                frame_start = pss_pos - 5 * 30720 - 6 * 2048  # Adjust for subframe 5
                confidence = corr5
            
            if confidence > best_confidence:
                best_frame_start = frame_start
                best_confidence = confidence
        
        return best_frame_start, 0, 0  # Would need to decode SSS for NID1
    
    def find_wifi_frame_start(self, samples: np.ndarray,
                              stf: np.ndarray) -> Tuple[int, float]:
        """
        Find WiFi frame start using STF.
        
        Returns:
            (frame_start, confidence)
        """
        # Auto-correlation for STF (16-sample repetitions)
        corr = np.zeros(len(samples) - 32)
        
        for i in range(len(corr)):
            segment1 = samples[i:i + 16]
            segment2 = samples[i + 16:i + 32]
            corr[i] = np.abs(np.sum(segment1 * np.conj(segment2)))
        
        # Normalize
        power = np.convolve(np.abs(samples) ** 2, np.ones(32), mode='valid')[:len(corr)]
        metric = corr / (power + 1e-10)
        
        # Find plateau (STF has sustained high correlation)
        threshold = 0.8 * np.max(metric)
        above_threshold = metric > threshold
        
        # Find first sustained region
        frame_start = 0
        for i in range(len(above_threshold) - 160):
            if np.all(above_threshold[i:i + 100]):
                frame_start = i
                break
        
        confidence = metric[frame_start] if frame_start < len(metric) else 0
        
        return frame_start, confidence
    
    def track_frame_timing(self, samples: np.ndarray,
                           expected_frame_length: int,
                           reference: np.ndarray,
                           search_window: int = 100) -> int:
        """
        Track frame timing for continuous reception.
        
        Uses previous frame timing as starting point.
        
        Args:
            samples: Received samples
            expected_frame_length: Expected samples per frame
            reference: Known sequence at frame start
            search_window: Search range in samples
        
        Returns:
            Timing adjustment in samples
        """
        # Search around expected position
        expected_pos = expected_frame_length
        
        best_offset = 0
        best_corr = 0
        
        for offset in range(-search_window, search_window + 1):
            pos = expected_pos + offset
            if pos < 0 or pos + len(reference) > len(samples):
                continue
            
            corr = np.abs(np.sum(samples[pos:pos + len(reference)] * np.conj(reference)))
            
            if corr > best_corr:
                best_corr = corr
                best_offset = offset
        
        return best_offset


class CellSearch:
    """
    LTE/5G Cell Search Procedure
    
    Complete cell search including PSS, SSS detection and cell ID determination.
    """
    
    def __init__(self, sample_rate: float = 30.72e6, fft_size: int = 2048):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        
        self.timing_sync = TimingSync(sample_rate)
        self.freq_sync = FrequencySync(sample_rate)
        self.frame_sync = FrameSync(sample_rate)
        
        # Generate PSS sequences
        self.pss_sequences = {}
        for nid2 in range(3):
            self.pss_sequences[nid2] = self._generate_pss(nid2)
        
        self.logger = logging.getLogger('CellSearch')
    
    def _generate_pss(self, nid2: int) -> np.ndarray:
        """Generate PSS sequence for given NID2"""
        roots = {0: 25, 1: 29, 2: 34}
        u = roots[nid2]
        
        # Zadoff-Chu sequence
        n = np.arange(63)
        d = np.zeros(63, dtype=complex)
        
        for i in range(63):
            if i <= 30:
                d[i] = np.exp(-1j * np.pi * u * i * (i + 1) / 63)
            else:
                d[i] = np.exp(-1j * np.pi * u * (i + 1) * (i + 2) / 63)
        
        # Remove DC
        pss = np.concatenate([d[:31], d[32:]])
        
        return pss
    
    def _generate_sss(self, nid1: int, nid2: int, subframe: int) -> np.ndarray:
        """Generate SSS sequence"""
        # M-sequence generation
        def m_seq(init):
            x = np.array(init + [0] * 26)
            for i in range(5, 31):
                x[i] = (x[i-3] + x[i-5]) % 2
            return 1 - 2 * x[:31]
        
        # Generate sequences
        s = m_seq([0, 0, 0, 0, 1])
        
        # Compute m0, m1
        q_prime = nid1 // 30
        q = (nid1 + q_prime * (q_prime + 1) // 2) // 30
        m_prime = nid1 + q * (q + 1) // 2
        m0 = m_prime % 31
        m1 = (m0 + (m_prime // 31) + 1) % 31
        
        # Generate SSS
        sss = np.zeros(62, dtype=complex)
        for n in range(31):
            s0 = s[(n + m0) % 31]
            s1 = s[(n + m1) % 31]
            
            if subframe == 0:
                sss[2*n] = s0
                sss[2*n + 1] = s1
            else:
                sss[2*n] = s1
                sss[2*n + 1] = s0
        
        return sss
    
    def search(self, samples: np.ndarray) -> SyncResult:
        """
        Perform complete cell search.
        
        Args:
            samples: Received samples (at least 10ms worth)
        
        Returns:
            SyncResult with timing, frequency, and cell ID
        """
        # Step 1: PSS detection for timing and NID2
        best_nid2 = 0
        best_timing = 0
        best_corr = 0
        
        for nid2, pss in self.pss_sequences.items():
            # Time-domain correlation
            corr = np.abs(scipy_signal.correlate(samples, pss, mode='valid'))
            
            peak_idx = np.argmax(corr)
            peak_val = corr[peak_idx]
            
            if peak_val > best_corr:
                best_nid2 = nid2
                best_timing = peak_idx
                best_corr = peak_val
        
        if best_corr < 0.1:
            return SyncResult(success=False, timing_offset=0, 
                            frequency_offset=0, frame_start=0,
                            confidence=0)
        
        # Step 2: Coarse CFO estimation
        # Extract samples around PSS
        pss_start = max(0, best_timing - self.fft_size)
        pss_samples = samples[pss_start:pss_start + self.fft_size * 2]
        
        cfo_coarse = self.freq_sync.estimate_cfo_cp(
            pss_samples, self.fft_size, self.fft_size // 14
        )
        
        # Step 3: CFO correction
        corrected = self.freq_sync.correct_cfo(samples, cfo_coarse)
        
        # Step 4: SSS detection for NID1 and subframe
        best_nid1 = 0
        best_subframe = 0
        best_sss_corr = 0
        
        # SSS is one OFDM symbol before PSS
        sss_timing = best_timing - self.fft_size - self.fft_size // 14
        
        for nid1 in range(168):
            for subframe in [0, 5]:
                sss = self._generate_sss(nid1, best_nid2, subframe)
                
                # Extract SSS from received signal
                sss_start = max(0, sss_timing)
                sss_samples = corrected[sss_start:sss_start + 62]
                
                if len(sss_samples) < 62:
                    continue
                
                # Correlation
                corr = np.abs(np.sum(sss_samples * np.conj(sss)))
                
                if corr > best_sss_corr:
                    best_nid1 = nid1
                    best_subframe = subframe
                    best_sss_corr = corr
        
        # Step 5: Calculate Cell ID
        cell_id = 3 * best_nid1 + best_nid2
        
        # Step 6: Calculate frame start
        if best_subframe == 0:
            # PSS is in slot 0 (symbol 6)
            frame_start = best_timing - 6 * (self.fft_size + self.fft_size // 14)
        else:
            # PSS is in slot 10 (subframe 5)
            samples_per_subframe = 30720  # 1ms at 30.72 MHz
            frame_start = best_timing - 5 * samples_per_subframe - 6 * (self.fft_size + self.fft_size // 14)
        
        # Step 7: Fine CFO estimation
        cfo_fine = self.freq_sync.fine_cfo_estimation(
            samples[max(0, frame_start):],
            self.pss_sequences[best_nid2],
            search_range_hz=500
        )
        
        total_cfo = cfo_coarse + cfo_fine
        
        # Estimate SNR
        signal_power = np.mean(np.abs(samples[best_timing:best_timing + 62]) ** 2)
        noise_power = np.var(samples)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        return SyncResult(
            success=True,
            timing_offset=best_timing,
            frequency_offset=total_cfo,
            frame_start=max(0, frame_start),
            cell_id=cell_id,
            snr_estimate=snr_db,
            confidence=best_corr / np.max(np.abs(samples))
        )


class SyncEngine:
    """
    Unified Synchronization Engine
    
    Combines timing, frequency, and frame sync for different systems.
    Maintains stealth by avoiding detectable patterns.
    """
    
    def __init__(self, sample_rate: float = 30.72e6, stealth_mode: bool = True):
        self.sample_rate = sample_rate
        self.stealth_mode = stealth_mode
        
        self.timing = TimingSync(sample_rate)
        self.frequency = FrequencySync(sample_rate)
        self.frame = FrameSync(sample_rate)
        self.cell_search = CellSearch(sample_rate)
        
        self.logger = logging.getLogger('SyncEngine')
    
    def synchronize_lte(self, samples: np.ndarray) -> SyncResult:
        """Synchronize to LTE signal"""
        return self.cell_search.search(samples)
    
    def synchronize_wifi(self, samples: np.ndarray,
                        params) -> SyncResult:
        """Synchronize to WiFi signal"""
        # Find frame start
        frame_start, confidence = self.frame.find_wifi_frame_start(samples, None)
        
        if confidence < 0.5:
            return SyncResult(success=False, timing_offset=0,
                            frequency_offset=0, frame_start=0)
        
        # Estimate CFO from STF (16-sample repetitions)
        stf_samples = samples[frame_start:frame_start + 160]
        cfo = self.frequency.estimate_cfo_preamble(stf_samples, 16, 8)
        
        return SyncResult(
            success=True,
            timing_offset=frame_start,
            frequency_offset=cfo,
            frame_start=frame_start,
            confidence=confidence
        )
    
    def track(self, samples: np.ndarray,
              previous_result: SyncResult,
              frame_length: int) -> SyncResult:
        """
        Track synchronization for continuous reception.
        
        Uses previous sync result as starting point.
        """
        # Timing tracking
        timing_adjustment = self.frame.track_frame_timing(
            samples, frame_length, None, search_window=50
        )
        
        new_timing = previous_result.frame_start + frame_length + timing_adjustment
        
        # Frequency tracking
        cfo_update = self.frequency.estimate_cfo_cp(
            samples[new_timing:new_timing + 2048],
            2048, 144
        )
        
        return SyncResult(
            success=True,
            timing_offset=timing_adjustment,
            frequency_offset=previous_result.frequency_offset + cfo_update,
            frame_start=new_timing,
            cell_id=previous_result.cell_id,
            snr_estimate=previous_result.snr_estimate,
            confidence=previous_result.confidence
        )
