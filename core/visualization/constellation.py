"""
RF Arsenal OS - Constellation Diagram Visualization
Real-time IQ constellation plotting for signal analysis and demodulation verification.
Supports BPSK, QPSK, 8PSK, 16QAM, 64QAM, 256QAM and custom modulation schemes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from collections import deque


class ModulationType(Enum):
    """Supported modulation types for constellation analysis"""
    BPSK = "bpsk"
    QPSK = "qpsk"
    PSK8 = "8psk"
    QAM16 = "16qam"
    QAM64 = "64qam"
    QAM256 = "256qam"
    APSK = "apsk"
    OFDM = "ofdm"
    FSK = "fsk"
    GMSK = "gmsk"
    UNKNOWN = "unknown"


@dataclass
class ConstellationPoint:
    """Single point in constellation diagram"""
    i_value: float  # In-phase component
    q_value: float  # Quadrature component
    timestamp: float = field(default_factory=time.time)
    symbol_index: int = 0
    error_magnitude: float = 0.0
    snr_estimate: float = 0.0


@dataclass
class ConstellationMetrics:
    """Metrics computed from constellation analysis"""
    evm_percent: float = 0.0  # Error Vector Magnitude
    mer_db: float = 0.0  # Modulation Error Ratio
    snr_db: float = 0.0  # Signal-to-Noise Ratio
    ber_estimate: float = 0.0  # Bit Error Rate estimate
    phase_error_deg: float = 0.0  # Average phase error
    amplitude_error_percent: float = 0.0  # Average amplitude error
    iq_offset_i: float = 0.0  # DC offset on I channel
    iq_offset_q: float = 0.0  # DC offset on Q channel
    iq_imbalance_db: float = 0.0  # IQ gain imbalance
    carrier_frequency_offset: float = 0.0  # Hz
    symbol_rate_offset: float = 0.0  # ppm


class ConstellationDiagram:
    """
    Production-grade constellation diagram analyzer for RF signal analysis.
    
    Features:
    - Real-time IQ plotting with configurable update rates
    - Automatic modulation detection
    - Signal quality metrics (EVM, MER, SNR)
    - Symbol timing recovery
    - Carrier frequency offset estimation
    - Support for all common modulation schemes
    - Memory-efficient circular buffer operation
    - Thread-safe for real-time streaming
    """
    
    def __init__(self,
                 max_points: int = 10000,
                 modulation: ModulationType = ModulationType.UNKNOWN,
                 sample_rate: float = 1e6,
                 symbol_rate: float = 0,
                 enable_metrics: bool = True):
        """
        Initialize constellation diagram analyzer.
        
        Args:
            max_points: Maximum points to store (circular buffer)
            modulation: Expected modulation type
            sample_rate: Sample rate in Hz
            symbol_rate: Symbol rate in Hz (0 for auto-detect)
            enable_metrics: Enable real-time metrics calculation
        """
        self.max_points = max_points
        self.modulation = modulation
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.enable_metrics = enable_metrics
        
        # Circular buffer for constellation points
        self._points_buffer = deque(maxlen=max_points)
        self._raw_iq_buffer = deque(maxlen=max_points * 10)
        
        # Reference constellation for error calculation
        self._reference_constellation = self._generate_reference_constellation()
        
        # Metrics
        self._metrics = ConstellationMetrics()
        self._metrics_lock = threading.Lock()
        
        # Symbol timing recovery
        self._timing_offset = 0.0
        self._timing_error_history = deque(maxlen=100)
        
        # Carrier recovery
        self._phase_accumulator = 0.0
        self._frequency_offset = 0.0
        
        # Callbacks for real-time updates
        self._update_callbacks: List[Callable] = []
        
        # State
        self._running = False
        self._symbol_count = 0
        
    def _generate_reference_constellation(self) -> np.ndarray:
        """Generate ideal reference constellation for current modulation"""
        if self.modulation == ModulationType.BPSK:
            return np.array([1+0j, -1+0j])
        elif self.modulation == ModulationType.QPSK:
            return np.array([
                1+1j, 1-1j, -1+1j, -1-1j
            ]) / np.sqrt(2)
        elif self.modulation == ModulationType.PSK8:
            angles = np.arange(8) * 2 * np.pi / 8
            return np.exp(1j * angles)
        elif self.modulation == ModulationType.QAM16:
            coords = [-3, -1, 1, 3]
            const = np.array([i + 1j*q for i in coords for q in coords])
            return const / np.sqrt(np.mean(np.abs(const)**2))
        elif self.modulation == ModulationType.QAM64:
            coords = [-7, -5, -3, -1, 1, 3, 5, 7]
            const = np.array([i + 1j*q for i in coords for q in coords])
            return const / np.sqrt(np.mean(np.abs(const)**2))
        elif self.modulation == ModulationType.QAM256:
            coords = np.arange(-15, 16, 2)
            const = np.array([i + 1j*q for i in coords for q in coords])
            return const / np.sqrt(np.mean(np.abs(const)**2))
        else:
            return np.array([])
            
    def add_samples(self, iq_samples: np.ndarray) -> None:
        """
        Add raw IQ samples for constellation analysis.
        
        Args:
            iq_samples: Complex IQ samples array
        """
        if len(iq_samples) == 0:
            return
            
        # Store raw samples
        for sample in iq_samples:
            self._raw_iq_buffer.append(sample)
            
        # Perform symbol timing recovery if symbol rate is known
        if self.symbol_rate > 0:
            symbols = self._recover_symbols(iq_samples)
        else:
            # Use samples directly (downsample if needed)
            symbols = iq_samples[::int(self.sample_rate / 1e5 + 1)]
            
        # Apply carrier recovery
        symbols = self._carrier_recovery(symbols)
        
        # Add to constellation buffer
        timestamp = time.time()
        for i, sym in enumerate(symbols):
            point = ConstellationPoint(
                i_value=float(np.real(sym)),
                q_value=float(np.imag(sym)),
                timestamp=timestamp,
                symbol_index=self._symbol_count
            )
            
            # Calculate error if reference constellation available
            if len(self._reference_constellation) > 0:
                closest_idx, error = self._find_closest_symbol(sym)
                point.error_magnitude = error
                
            self._points_buffer.append(point)
            self._symbol_count += 1
            
        # Update metrics
        if self.enable_metrics:
            self._update_metrics()
            
        # Notify callbacks
        for callback in self._update_callbacks:
            callback(self.get_constellation_data())
            
    def _recover_symbols(self, samples: np.ndarray) -> np.ndarray:
        """
        Gardner timing error detector for symbol recovery.
        """
        samples_per_symbol = int(self.sample_rate / self.symbol_rate)
        if samples_per_symbol < 2:
            return samples
            
        symbols = []
        idx = int(samples_per_symbol / 2)  # Start at midpoint
        
        while idx + samples_per_symbol < len(samples):
            # Current symbol
            symbol = samples[idx]
            symbols.append(symbol)
            
            # Gardner timing error detector
            if len(symbols) >= 2:
                # Mid-sample
                mid_idx = idx - samples_per_symbol // 2
                if mid_idx >= 0:
                    y_mid = samples[mid_idx]
                    y_prev = symbols[-2] if len(symbols) >= 2 else symbol
                    y_curr = symbol
                    
                    # Timing error
                    ted = np.real((y_curr - y_prev) * np.conj(y_mid))
                    self._timing_error_history.append(ted)
                    
                    # Adjust timing
                    timing_adjustment = 0.1 * ted
                    idx += int(timing_adjustment)
                    
            idx += samples_per_symbol
            
        return np.array(symbols) if symbols else samples
        
    def _carrier_recovery(self, symbols: np.ndarray) -> np.ndarray:
        """
        Costas loop carrier recovery for phase/frequency offset correction.
        """
        if len(symbols) == 0:
            return symbols
            
        # Loop filter constants
        bw = 0.01  # Loop bandwidth
        damping = 0.707
        k0 = 1.0  # NCO gain
        kd = 1.0  # Phase detector gain
        
        theta1 = bw / (damping + 1/(4*damping))
        theta2 = theta1 * theta1 / (4 * damping * damping)
        
        recovered = []
        phase = self._phase_accumulator
        freq = self._frequency_offset
        
        for sym in symbols:
            # Apply current phase correction
            corrected = sym * np.exp(-1j * phase)
            recovered.append(corrected)
            
            # Phase detector (decision-directed)
            if self.modulation in [ModulationType.BPSK, ModulationType.QPSK]:
                # Quadrature phase detector
                error = np.sign(np.real(corrected)) * np.imag(corrected) - \
                        np.sign(np.imag(corrected)) * np.real(corrected)
            else:
                # Generic phase detector
                error = np.angle(corrected * np.conj(self._quantize(corrected)))
                
            # Loop filter
            freq += theta2 * error
            phase += theta1 * error + freq
            
            # Wrap phase
            while phase > np.pi:
                phase -= 2 * np.pi
            while phase < -np.pi:
                phase += 2 * np.pi
                
        self._phase_accumulator = phase
        self._frequency_offset = freq
        
        return np.array(recovered)
        
    def _quantize(self, symbol: complex) -> complex:
        """Quantize symbol to nearest constellation point"""
        if len(self._reference_constellation) == 0:
            return symbol
            
        idx, _ = self._find_closest_symbol(symbol)
        return self._reference_constellation[idx]
        
    def _find_closest_symbol(self, symbol: complex) -> Tuple[int, float]:
        """Find closest reference symbol and return index and error"""
        distances = np.abs(self._reference_constellation - symbol)
        closest_idx = np.argmin(distances)
        return closest_idx, float(distances[closest_idx])
        
    def _update_metrics(self) -> None:
        """Calculate constellation metrics"""
        if len(self._points_buffer) < 10:
            return
            
        points = list(self._points_buffer)[-1000:]  # Use recent points
        iq_data = np.array([p.i_value + 1j * p.q_value for p in points])
        
        with self._metrics_lock:
            # DC offset
            self._metrics.iq_offset_i = float(np.mean(np.real(iq_data)))
            self._metrics.iq_offset_q = float(np.mean(np.imag(iq_data)))
            
            # IQ imbalance
            i_power = np.mean(np.real(iq_data)**2)
            q_power = np.mean(np.imag(iq_data)**2)
            if q_power > 0:
                self._metrics.iq_imbalance_db = float(10 * np.log10(i_power / q_power))
            
            # EVM calculation
            if len(self._reference_constellation) > 0:
                errors = np.array([p.error_magnitude for p in points])
                avg_power = np.mean(np.abs(iq_data)**2)
                if avg_power > 0:
                    self._metrics.evm_percent = float(
                        100 * np.sqrt(np.mean(errors**2)) / np.sqrt(avg_power)
                    )
                    
                # MER in dB
                if self._metrics.evm_percent > 0:
                    self._metrics.mer_db = float(
                        -20 * np.log10(self._metrics.evm_percent / 100)
                    )
                    
            # SNR estimation (M2M4 method)
            m2 = np.mean(np.abs(iq_data)**2)
            m4 = np.mean(np.abs(iq_data)**4)
            if m2 > 0:
                # For PSK modulation
                kurtosis = m4 / (m2**2) - 2
                if kurtosis > 0:
                    self._metrics.snr_db = float(10 * np.log10(1 / kurtosis))
                    
            # Phase error
            phases = np.angle(iq_data)
            self._metrics.phase_error_deg = float(np.std(phases) * 180 / np.pi)
            
            # Carrier frequency offset
            self._metrics.carrier_frequency_offset = float(
                self._frequency_offset * self.sample_rate / (2 * np.pi)
            )
            
    def detect_modulation(self) -> ModulationType:
        """
        Automatic modulation recognition using constellation clustering.
        
        Returns:
            Detected modulation type
        """
        if len(self._points_buffer) < 100:
            return ModulationType.UNKNOWN
            
        points = list(self._points_buffer)
        iq_data = np.array([p.i_value + 1j * p.q_value for p in points])
        
        # Normalize
        iq_data = iq_data / np.sqrt(np.mean(np.abs(iq_data)**2))
        
        # Count unique amplitude levels
        amplitudes = np.abs(iq_data)
        amp_hist, _ = np.histogram(amplitudes, bins=20)
        amp_peaks = np.sum(amp_hist > len(iq_data) * 0.02)
        
        # Count unique phase levels
        phases = np.angle(iq_data)
        phase_hist, _ = np.histogram(phases, bins=32)
        phase_peaks = np.sum(phase_hist > len(iq_data) * 0.02)
        
        # Decision tree for modulation detection
        if amp_peaks <= 2 and phase_peaks <= 2:
            return ModulationType.BPSK
        elif amp_peaks <= 2 and phase_peaks <= 4:
            return ModulationType.QPSK
        elif amp_peaks <= 2 and phase_peaks <= 8:
            return ModulationType.PSK8
        elif amp_peaks >= 3 and phase_peaks >= 4:
            # QAM detection based on clustering
            unique_count = self._estimate_symbol_count(iq_data)
            if unique_count <= 16:
                return ModulationType.QAM16
            elif unique_count <= 64:
                return ModulationType.QAM64
            else:
                return ModulationType.QAM256
        else:
            return ModulationType.UNKNOWN
            
    def _estimate_symbol_count(self, iq_data: np.ndarray) -> int:
        """Estimate number of unique symbols using k-means-like clustering"""
        # Simple histogram-based counting
        i_values = np.real(iq_data)
        q_values = np.imag(iq_data)
        
        # Count unique I and Q levels
        i_hist, _ = np.histogram(i_values, bins=20)
        q_hist, _ = np.histogram(q_values, bins=20)
        
        i_levels = np.sum(i_hist > len(iq_data) * 0.02)
        q_levels = np.sum(q_hist > len(iq_data) * 0.02)
        
        return i_levels * q_levels
        
    def get_constellation_data(self) -> Dict[str, Any]:
        """
        Get current constellation data for visualization.
        
        Returns:
            Dictionary with I/Q arrays, metrics, and metadata
        """
        points = list(self._points_buffer)
        
        return {
            "i_values": [p.i_value for p in points],
            "q_values": [p.q_value for p in points],
            "timestamps": [p.timestamp for p in points],
            "errors": [p.error_magnitude for p in points],
            "modulation": self.modulation.value,
            "metrics": self.get_metrics(),
            "symbol_count": self._symbol_count,
            "reference_constellation": {
                "i": list(np.real(self._reference_constellation)),
                "q": list(np.imag(self._reference_constellation))
            } if len(self._reference_constellation) > 0 else None
        }
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current constellation metrics"""
        with self._metrics_lock:
            return {
                "evm_percent": self._metrics.evm_percent,
                "mer_db": self._metrics.mer_db,
                "snr_db": self._metrics.snr_db,
                "ber_estimate": self._metrics.ber_estimate,
                "phase_error_deg": self._metrics.phase_error_deg,
                "amplitude_error_percent": self._metrics.amplitude_error_percent,
                "iq_offset_i": self._metrics.iq_offset_i,
                "iq_offset_q": self._metrics.iq_offset_q,
                "iq_imbalance_db": self._metrics.iq_imbalance_db,
                "carrier_frequency_offset_hz": self._metrics.carrier_frequency_offset,
                "symbol_rate_offset_ppm": self._metrics.symbol_rate_offset
            }
            
    def set_modulation(self, modulation: ModulationType) -> None:
        """Set expected modulation type"""
        self.modulation = modulation
        self._reference_constellation = self._generate_reference_constellation()
        
    def clear(self) -> None:
        """Clear constellation buffer"""
        self._points_buffer.clear()
        self._raw_iq_buffer.clear()
        self._symbol_count = 0
        self._metrics = ConstellationMetrics()
        
    def register_callback(self, callback: Callable) -> None:
        """Register callback for real-time updates"""
        self._update_callbacks.append(callback)
        
    def export_data(self, filepath: str) -> bool:
        """Export constellation data to file (RAM-only safe)"""
        try:
            import json
            data = self.get_constellation_data()
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception:
            return False


class EyeDiagram:
    """
    Eye diagram analyzer for timing jitter and ISI analysis.
    """
    
    def __init__(self, samples_per_symbol: int = 16, persistence: int = 100):
        self.samples_per_symbol = samples_per_symbol
        self.persistence = persistence
        self._traces = deque(maxlen=persistence)
        
    def add_trace(self, trace: np.ndarray) -> None:
        """Add one eye trace (2 symbol periods)"""
        if len(trace) >= 2 * self.samples_per_symbol:
            self._traces.append(trace[:2 * self.samples_per_symbol])
            
    def get_eye_data(self) -> Dict[str, Any]:
        """Get eye diagram data for visualization"""
        if not self._traces:
            return {"traces": [], "metrics": {}}
            
        traces = list(self._traces)
        
        # Calculate eye opening
        traces_array = np.array(traces)
        midpoint = self.samples_per_symbol
        
        # Eye height at midpoint
        eye_height = np.min(np.max(traces_array[:, midpoint], axis=0)) - \
                     np.max(np.min(traces_array[:, midpoint], axis=0))
                     
        # Eye width estimation
        threshold = np.mean(traces_array)
        
        return {
            "traces": [t.tolist() for t in traces],
            "metrics": {
                "eye_height": float(eye_height),
                "jitter_rms": float(np.std(traces_array[:, midpoint])),
                "samples_per_symbol": self.samples_per_symbol
            }
        }
