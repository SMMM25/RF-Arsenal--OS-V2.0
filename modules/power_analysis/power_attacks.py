#!/usr/bin/env python3
"""
RF Arsenal OS - Power Analysis Side-Channel Attack Module

Capabilities:
- Simple Power Analysis (SPA)
- Differential Power Analysis (DPA)
- Correlation Power Analysis (CPA)
- Electromagnetic Analysis (EMA)
- Timing attacks
- Fault injection

Hardware: Oscilloscope/Logic Analyzer + BladeRF for EM capture

WARNING: These techniques may be restricted. Use only on devices you own
or have explicit authorization to test.
"""

import logging
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from enum import Enum
from datetime import datetime
from pathlib import Path

# Try imports
try:
    import SoapySDR
    SOAPY_AVAILABLE = True
except ImportError:
    SOAPY_AVAILABLE = False


class AttackMode(Enum):
    """Power analysis attack modes"""
    SPA = "spa"           # Simple Power Analysis
    DPA = "dpa"           # Differential Power Analysis
    CPA = "cpa"           # Correlation Power Analysis
    EMA = "ema"           # Electromagnetic Analysis
    TIMING = "timing"     # Timing attack
    FAULT = "fault"       # Fault injection


class TargetAlgorithm(Enum):
    """Target cryptographic algorithms"""
    AES_128 = "aes128"
    AES_256 = "aes256"
    DES = "des"
    RSA = "rsa"
    ECC = "ecc"
    SHA = "sha"
    CUSTOM = "custom"


@dataclass
class PowerTrace:
    """Single power/EM trace"""
    samples: np.ndarray
    plaintext: Optional[bytes] = None
    ciphertext: Optional[bytes] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.samples)


@dataclass
class KeyHypothesis:
    """Key byte hypothesis"""
    byte_index: int
    value: int
    correlation: float
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            'byte_index': self.byte_index,
            'value': self.value,
            'value_hex': f"{self.value:02x}",
            'correlation': self.correlation,
            'confidence': self.confidence
        }


@dataclass
class AttackResult:
    """Power analysis attack result"""
    success: bool
    attack_mode: AttackMode
    algorithm: TargetAlgorithm
    recovered_key: Optional[bytes] = None
    key_hypotheses: List[KeyHypothesis] = field(default_factory=list)
    traces_used: int = 0
    duration: float = 0.0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'attack_mode': self.attack_mode.value,
            'algorithm': self.algorithm.value,
            'recovered_key': self.recovered_key.hex() if self.recovered_key else None,
            'traces_used': self.traces_used,
            'duration': self.duration,
            'confidence': self.confidence,
            'key_bytes': [h.to_dict() for h in self.key_hypotheses]
        }


class PowerAnalysisController:
    """
    Power Analysis Attack Controller
    
    Performs side-channel attacks using power consumption
    or electromagnetic emanation measurements.
    """
    
    # AES S-box for CPA
    AES_SBOX = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
    ]
    
    def __init__(self):
        self.logger = logging.getLogger('PowerAnalysis')
        
        # State
        self.traces: List[PowerTrace] = []
        self.results: List[AttackResult] = []
        
        # Hardware
        self._sdr = None  # For EM capture
        self._sample_rate = 100_000_000  # 100 MSPS for high resolution
        
        # Configuration
        self.target_algorithm = TargetAlgorithm.AES_128
        
    def init_em_capture(self) -> bool:
        """Initialize EM capture hardware"""
        if not SOAPY_AVAILABLE:
            self.logger.warning("SoapySDR not available - simulation mode")
            return True
            
        try:
            devices = SoapySDR.Device.enumerate()
            if devices:
                self._sdr = SoapySDR.Device(devices[0])
                self._sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, self._sample_rate)
                self._sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 60)
                self.logger.info("EM capture initialized")
                return True
                
        except Exception as e:
            self.logger.error(f"EM init error: {e}")
            
        return False
        
    def capture_trace(self, plaintext: bytes, trigger_callback: Callable = None,
                     num_samples: int = 10000) -> Optional[PowerTrace]:
        """
        Capture single power/EM trace
        
        Args:
            plaintext: Input data for encryption
            trigger_callback: Optional callback to trigger encryption
            num_samples: Number of samples to capture
        """
        try:
            if not self._sdr:
                self.logger.error("No SDR connected - power analysis requires EM capture hardware")
                return None
                
            # Real EM capture
            samples = self._em_capture(num_samples)
                
            trace = PowerTrace(
                samples=samples,
                plaintext=plaintext
            )
            
            self.traces.append(trace)
            return trace
            
        except Exception as e:
            self.logger.error(f"Capture error: {e}")
            return None
            
    def _em_capture(self, num_samples: int) -> np.ndarray:
        """Capture EM samples"""
        if not self._sdr:
            return np.zeros(num_samples, dtype=np.float32)
            
        try:
            stream = self._sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
            self._sdr.activateStream(stream)
            
            buff = np.zeros(num_samples, dtype=np.complex64)
            self._sdr.readStream(stream, [buff], num_samples)
            
            self._sdr.deactivateStream(stream)
            self._sdr.closeStream(stream)
            
            # Return magnitude (power)
            return np.abs(buff).astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"EM capture error: {e}")
            return np.zeros(num_samples, dtype=np.float32)
            
    def get_leakage_models(self) -> Dict[str, str]:
        """Return supported leakage models for analysis"""
        return {
            'hamming_weight': 'Count of 1-bits in intermediate value',
            'hamming_distance': 'Bit transitions between consecutive values',
            'zero_value': 'Detect when value equals zero',
            'identity': 'Direct value correlation',
            'lsb': 'Least significant bit leakage',
        }
    
    def get_attack_requirements(self) -> Dict[str, Any]:
        """Return requirements for successful power analysis"""
        return {
            'min_traces_dpa': 1000,
            'min_traces_cpa': 500,
            'recommended_sample_rate': 500e6,
            'bandwidth_mhz': 200,
            'trigger_requirements': 'Precise timing sync with crypto operations',
            'probe_placement': 'Near VCC or GND of target chip',
        }
        
    def capture_traces(self, plaintexts: List[bytes], num_traces: int = None) -> List[PowerTrace]:
        """Capture multiple traces"""
        if num_traces:
            plaintexts = plaintexts[:num_traces]
            
        traces = []
        self.logger.info(f"Capturing {len(plaintexts)} traces...")
        
        for i, pt in enumerate(plaintexts):
            trace = self.capture_trace(pt)
            if trace:
                traces.append(trace)
                
            if (i + 1) % 100 == 0:
                self.logger.info(f"Captured {i + 1}/{len(plaintexts)} traces")
                
        return traces
        
    # === Simple Power Analysis (SPA) ===
    
    def attack_spa(self, trace: PowerTrace = None) -> AttackResult:
        """
        Simple Power Analysis
        
        Analyzes single trace for visible key-dependent patterns
        """
        self.logger.info("Starting SPA attack...")
        start_time = time.time()
        
        target_trace = trace or (self.traces[-1] if self.traces else None)
        
        if not target_trace:
            return AttackResult(
                success=False,
                attack_mode=AttackMode.SPA,
                algorithm=self.target_algorithm,
                duration=0
            )
            
        # Find peaks and patterns
        samples = target_trace.samples
        
        # Detect operations by power spikes
        threshold = np.mean(samples) + 2 * np.std(samples)
        peaks = self._find_peaks(samples, threshold)
        
        # Try to identify operations
        operations = self._identify_operations(samples, peaks)
        
        # For RSA, look for square vs multiply pattern
        if self.target_algorithm == TargetAlgorithm.RSA:
            key_bits = self._spa_rsa(samples, peaks)
            if key_bits:
                recovered_key = self._bits_to_bytes(key_bits)
                
                result = AttackResult(
                    success=True,
                    attack_mode=AttackMode.SPA,
                    algorithm=self.target_algorithm,
                    recovered_key=recovered_key,
                    traces_used=1,
                    duration=time.time() - start_time,
                    confidence=0.6
                )
                self.results.append(result)
                return result
                
        return AttackResult(
            success=False,
            attack_mode=AttackMode.SPA,
            algorithm=self.target_algorithm,
            traces_used=1,
            duration=time.time() - start_time
        )
        
    def _find_peaks(self, samples: np.ndarray, threshold: float) -> List[int]:
        """Find peaks above threshold"""
        peaks = []
        above = samples > threshold
        
        in_peak = False
        peak_start = 0
        
        for i, val in enumerate(above):
            if val and not in_peak:
                in_peak = True
                peak_start = i
            elif not val and in_peak:
                in_peak = False
                peak_center = peak_start + np.argmax(samples[peak_start:i])
                peaks.append(peak_center)
                
        return peaks
        
    def _identify_operations(self, samples: np.ndarray, peaks: List[int]) -> List[str]:
        """Identify operations from power patterns"""
        operations = []
        
        for i, peak in enumerate(peaks):
            # Analyze local pattern
            start = max(0, peak - 100)
            end = min(len(samples), peak + 100)
            
            local = samples[start:end]
            duration = end - start
            power = np.max(local) - np.min(local)
            
            # Classify (simplified)
            if power > 0.1:
                if duration > 150:
                    operations.append('multiply')
                else:
                    operations.append('square')
            else:
                operations.append('unknown')
                
        return operations
        
    def _spa_rsa(self, samples: np.ndarray, peaks: List[int]) -> Optional[List[int]]:
        """SPA on RSA square-and-multiply"""
        # Simplified - real implementation would be more sophisticated
        operations = self._identify_operations(samples, peaks)
        
        # Extract bits from square-multiply pattern
        bits = []
        for op in operations:
            if op == 'square':
                bits.append(0)
            elif op == 'multiply':
                bits.append(1)
                
        return bits if bits else None
        
    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        """Convert bit list to bytes"""
        # Pad to multiple of 8
        while len(bits) % 8 != 0:
            bits.append(0)
            
        result = bytearray()
        for i in range(0, len(bits), 8):
            byte = sum(bits[i+j] << (7-j) for j in range(8))
            result.append(byte)
            
        return bytes(result)
        
    # === Differential Power Analysis (DPA) ===
    
    def attack_dpa(self, traces: List[PowerTrace] = None, 
                  target_byte: int = 0) -> AttackResult:
        """
        Differential Power Analysis
        
        Uses difference of means to recover key
        """
        self.logger.info(f"Starting DPA attack on byte {target_byte}...")
        start_time = time.time()
        
        target_traces = traces or self.traces
        
        if len(target_traces) < 100:
            self.logger.warning("DPA typically needs 100+ traces")
            
        if not target_traces:
            return AttackResult(
                success=False,
                attack_mode=AttackMode.DPA,
                algorithm=self.target_algorithm
            )
            
        # Build trace matrix
        trace_matrix = np.array([t.samples for t in target_traces])
        plaintexts = [t.plaintext for t in target_traces]
        
        best_key = 0
        best_diff = 0
        key_hypotheses = []
        
        # Try all possible key byte values
        for key_guess in range(256):
            # Compute selection function (Hamming weight of S-box output)
            selections = []
            for pt in plaintexts:
                if pt and len(pt) > target_byte:
                    sbox_in = pt[target_byte] ^ key_guess
                    sbox_out = self.AES_SBOX[sbox_in]
                    hw = bin(sbox_out).count('1')
                    selections.append(hw >= 4)  # Binary selection
                else:
                    selections.append(False)
                    
            selections = np.array(selections)
            
            # Compute difference of means
            if np.sum(selections) > 0 and np.sum(~selections) > 0:
                mean_1 = np.mean(trace_matrix[selections], axis=0)
                mean_0 = np.mean(trace_matrix[~selections], axis=0)
                diff = np.max(np.abs(mean_1 - mean_0))
                
                key_hypotheses.append(KeyHypothesis(
                    byte_index=target_byte,
                    value=key_guess,
                    correlation=diff,
                    confidence=diff / (best_diff if best_diff > 0 else 1)
                ))
                
                if diff > best_diff:
                    best_diff = diff
                    best_key = key_guess
                    
        # Sort hypotheses by correlation
        key_hypotheses.sort(key=lambda x: x.correlation, reverse=True)
        
        # Update confidences
        if key_hypotheses:
            max_corr = key_hypotheses[0].correlation
            for h in key_hypotheses:
                h.confidence = h.correlation / max_corr if max_corr > 0 else 0
                
        result = AttackResult(
            success=best_diff > 0.01,  # Threshold for success
            attack_mode=AttackMode.DPA,
            algorithm=self.target_algorithm,
            recovered_key=bytes([best_key]) if best_diff > 0.01 else None,
            key_hypotheses=key_hypotheses[:10],  # Top 10
            traces_used=len(target_traces),
            duration=time.time() - start_time,
            confidence=key_hypotheses[0].confidence if key_hypotheses else 0
        )
        
        self.results.append(result)
        return result
        
    # === Correlation Power Analysis (CPA) ===
    
    def attack_cpa(self, traces: List[PowerTrace] = None,
                  target_byte: int = 0) -> AttackResult:
        """
        Correlation Power Analysis
        
        Uses Pearson correlation for higher accuracy
        """
        self.logger.info(f"Starting CPA attack on byte {target_byte}...")
        start_time = time.time()
        
        target_traces = traces or self.traces
        
        if len(target_traces) < 50:
            self.logger.warning("CPA works best with 50+ traces")
            
        if not target_traces:
            return AttackResult(
                success=False,
                attack_mode=AttackMode.CPA,
                algorithm=self.target_algorithm
            )
            
        # Build trace matrix
        trace_matrix = np.array([t.samples for t in target_traces])
        plaintexts = [t.plaintext for t in target_traces]
        
        num_traces = len(target_traces)
        num_samples = trace_matrix.shape[1]
        
        best_key = 0
        best_corr = 0
        key_hypotheses = []
        
        # Try all possible key byte values
        for key_guess in range(256):
            # Compute power model (Hamming weight of S-box output)
            power_model = np.zeros(num_traces)
            
            for i, pt in enumerate(plaintexts):
                if pt and len(pt) > target_byte:
                    sbox_in = pt[target_byte] ^ key_guess
                    sbox_out = self.AES_SBOX[sbox_in]
                    power_model[i] = bin(sbox_out).count('1')
                    
            # Compute Pearson correlation for each sample point
            correlations = np.zeros(num_samples)
            
            # Vectorized correlation computation
            pm_centered = power_model - np.mean(power_model)
            pm_std = np.std(power_model)
            
            if pm_std > 0:
                traces_centered = trace_matrix - np.mean(trace_matrix, axis=0)
                traces_std = np.std(trace_matrix, axis=0)
                traces_std[traces_std == 0] = 1  # Avoid division by zero
                
                correlations = np.abs(np.dot(pm_centered, traces_centered) / 
                                      (num_traces * pm_std * traces_std))
                                      
            max_corr = np.max(correlations)
            
            key_hypotheses.append(KeyHypothesis(
                byte_index=target_byte,
                value=key_guess,
                correlation=max_corr,
                confidence=0  # Will be updated
            ))
            
            if max_corr > best_corr:
                best_corr = max_corr
                best_key = key_guess
                
        # Sort and update confidences
        key_hypotheses.sort(key=lambda x: x.correlation, reverse=True)
        
        if key_hypotheses:
            max_corr = key_hypotheses[0].correlation
            for h in key_hypotheses:
                h.confidence = h.correlation / max_corr if max_corr > 0 else 0
                
        success = best_corr > 0.3  # Correlation threshold
        
        result = AttackResult(
            success=success,
            attack_mode=AttackMode.CPA,
            algorithm=self.target_algorithm,
            recovered_key=bytes([best_key]) if success else None,
            key_hypotheses=key_hypotheses[:10],
            traces_used=len(target_traces),
            duration=time.time() - start_time,
            confidence=key_hypotheses[0].confidence if key_hypotheses else 0
        )
        
        self.results.append(result)
        return result
        
    def attack_full_key(self, traces: List[PowerTrace] = None,
                       method: AttackMode = AttackMode.CPA) -> AttackResult:
        """
        Recover full key (all bytes)
        """
        self.logger.info(f"Starting full key recovery with {method.value}...")
        start_time = time.time()
        
        target_traces = traces or self.traces
        
        key_bytes = []
        all_hypotheses = []
        
        # Attack each byte
        for byte_idx in range(16):  # AES-128 = 16 bytes
            if method == AttackMode.CPA:
                result = self.attack_cpa(target_traces, byte_idx)
            else:
                result = self.attack_dpa(target_traces, byte_idx)
                
            if result.recovered_key:
                key_bytes.append(result.recovered_key[0])
                all_hypotheses.extend(result.key_hypotheses[:3])
            else:
                key_bytes.append(0)  # Unknown
                
            self.logger.info(f"Byte {byte_idx}: 0x{key_bytes[-1]:02x}")
            
        full_key = bytes(key_bytes)
        
        # Calculate overall confidence
        avg_confidence = np.mean([h.confidence for h in all_hypotheses]) if all_hypotheses else 0
        
        result = AttackResult(
            success=avg_confidence > 0.5,
            attack_mode=method,
            algorithm=self.target_algorithm,
            recovered_key=full_key,
            key_hypotheses=all_hypotheses,
            traces_used=len(target_traces),
            duration=time.time() - start_time,
            confidence=avg_confidence
        )
        
        self.results.append(result)
        self.logger.info(f"Recovered key: {full_key.hex()}")
        
        return result
        
    # === Timing Attack ===
    
    def attack_timing(self, timing_data: List[Tuple[bytes, float]]) -> AttackResult:
        """
        Timing attack on comparison operations
        
        Args:
            timing_data: List of (input, timing) tuples
        """
        self.logger.info("Starting timing attack...")
        start_time = time.time()
        
        if len(timing_data) < 16:
            return AttackResult(
                success=False,
                attack_mode=AttackMode.TIMING,
                algorithm=self.target_algorithm
            )
            
        # Analyze timing variations by byte
        recovered = bytearray()
        
        for byte_pos in range(16):  # Assume 16-byte comparison
            byte_timings = {}
            
            for input_data, timing in timing_data:
                if len(input_data) > byte_pos:
                    byte_val = input_data[byte_pos]
                    if byte_val not in byte_timings:
                        byte_timings[byte_val] = []
                    byte_timings[byte_val].append(timing)
                    
            # Find byte value with longest average time
            # (indicates successful comparison)
            best_byte = 0
            best_time = 0
            
            for byte_val, times in byte_timings.items():
                avg_time = np.mean(times)
                if avg_time > best_time:
                    best_time = avg_time
                    best_byte = byte_val
                    
            recovered.append(best_byte)
            
        result = AttackResult(
            success=True,
            attack_mode=AttackMode.TIMING,
            algorithm=TargetAlgorithm.CUSTOM,
            recovered_key=bytes(recovered),
            traces_used=len(timing_data),
            duration=time.time() - start_time,
            confidence=0.7
        )
        
        self.results.append(result)
        return result
        
    # === Utility Methods ===
    
    def get_status(self) -> Dict:
        """Get controller status"""
        return {
            'traces_collected': len(self.traces),
            'attacks_performed': len(self.results),
            'target_algorithm': self.target_algorithm.value,
            'hardware': 'BladeRF EM' if self._sdr else 'Simulation'
        }
        
    def get_traces(self, limit: int = None) -> List[PowerTrace]:
        """Get collected traces"""
        if limit:
            return self.traces[-limit:]
        return self.traces
        
    def get_results(self) -> List[Dict]:
        """Get attack results"""
        return [r.to_dict() for r in self.results]
        
    def clear_traces(self):
        """Clear collected traces"""
        self.traces = []
        self.logger.info("Traces cleared")
        
    def save_traces(self, path: str):
        """Save traces to file"""
        try:
            data = {
                'traces': [
                    {
                        'samples': t.samples.tolist(),
                        'plaintext': t.plaintext.hex() if t.plaintext else None,
                        'timestamp': t.timestamp.isoformat()
                    }
                    for t in self.traces
                ]
            }
            
            import json
            with open(path, 'w') as f:
                json.dump(data, f)
                
            self.logger.info(f"Saved {len(self.traces)} traces to {path}")
            
        except Exception as e:
            self.logger.error(f"Save error: {e}")
            
    def load_traces(self, path: str) -> bool:
        """Load traces from file"""
        try:
            import json
            with open(path, 'r') as f:
                data = json.load(f)
                
            for t_data in data['traces']:
                trace = PowerTrace(
                    samples=np.array(t_data['samples'], dtype=np.float32),
                    plaintext=bytes.fromhex(t_data['plaintext']) if t_data['plaintext'] else None,
                    timestamp=datetime.fromisoformat(t_data['timestamp'])
                )
                self.traces.append(trace)
                
            self.logger.info(f"Loaded {len(self.traces)} traces from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Load error: {e}")
            return False


# Convenience function
def get_power_analysis_controller() -> PowerAnalysisController:
    """Get Power Analysis controller instance"""
    return PowerAnalysisController()
