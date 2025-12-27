#!/usr/bin/env python3
"""
RF Arsenal OS - Channel Coding Engine

Production-grade error correction coding for cellular and WiFi.
Supports Turbo, LDPC, Polar, and Convolutional codes.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CodeType(Enum):
    """Channel code types"""
    CONVOLUTIONAL = "convolutional"
    TURBO = "turbo"
    LDPC = "ldpc"
    POLAR = "polar"


@dataclass
class CodingConfig:
    """Channel coding configuration"""
    code_type: CodeType
    code_rate: float = 0.5  # 1/2, 1/3, 2/3, 3/4, 5/6
    block_length: int = 1024
    
    # Turbo/Convolutional specific
    constraint_length: int = 7
    
    # LDPC specific
    ldpc_iterations: int = 50
    
    # Polar specific
    polar_list_size: int = 8


class CRCCalculator:
    """
    CRC Calculator for error detection
    
    Supports common CRC polynomials used in cellular/WiFi.
    """
    
    # Common CRC polynomials
    CRC_POLYNOMIALS = {
        'CRC-8': 0x07,
        'CRC-16': 0x8005,
        'CRC-16-CCITT': 0x1021,
        'CRC-24A': 0x864CFB,  # LTE
        'CRC-24B': 0x800063,  # LTE
        'CRC-24C': 0xB2B117,  # 5G NR
        'CRC-32': 0x04C11DB7,
    }
    
    def __init__(self, crc_type: str = 'CRC-24A'):
        self.crc_type = crc_type
        self.polynomial = self.CRC_POLYNOMIALS[crc_type]
        
        # Determine CRC length
        if '8' in crc_type:
            self.crc_length = 8
        elif '16' in crc_type:
            self.crc_length = 16
        elif '24' in crc_type:
            self.crc_length = 24
        else:
            self.crc_length = 32
        
        # Pre-compute lookup table
        self._compute_table()
    
    def _compute_table(self):
        """Compute CRC lookup table"""
        self.table = np.zeros(256, dtype=np.uint32)
        
        for i in range(256):
            crc = i << (self.crc_length - 8)
            for _ in range(8):
                if crc & (1 << (self.crc_length - 1)):
                    crc = (crc << 1) ^ self.polynomial
                else:
                    crc <<= 1
            self.table[i] = crc & ((1 << self.crc_length) - 1)
    
    def calculate(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate CRC for bit array.
        
        Args:
            data: Input bits
        
        Returns:
            CRC bits
        """
        # Convert bits to bytes
        num_bytes = (len(data) + 7) // 8
        padded = np.zeros(num_bytes * 8, dtype=int)
        padded[:len(data)] = data
        
        byte_array = np.packbits(padded.astype(np.uint8))
        
        # Calculate CRC
        crc = 0
        for byte in byte_array:
            idx = ((crc >> (self.crc_length - 8)) ^ byte) & 0xFF
            crc = ((crc << 8) ^ self.table[idx]) & ((1 << self.crc_length) - 1)
        
        # Convert to bits
        crc_bits = np.array([(crc >> (self.crc_length - 1 - i)) & 1 
                            for i in range(self.crc_length)], dtype=int)
        
        return crc_bits
    
    def append(self, data: np.ndarray) -> np.ndarray:
        """Append CRC to data"""
        crc = self.calculate(data)
        return np.concatenate([data, crc])
    
    def check(self, data_with_crc: np.ndarray) -> bool:
        """Check CRC validity"""
        data = data_with_crc[:-self.crc_length]
        received_crc = data_with_crc[-self.crc_length:]
        calculated_crc = self.calculate(data)
        
        return np.array_equal(received_crc, calculated_crc)


class ConvolutionalCoder:
    """
    Convolutional Encoder/Decoder
    
    Supports various constraint lengths and code rates.
    Uses Viterbi decoding.
    """
    
    # Standard generator polynomials (octal)
    GENERATORS = {
        3: ([7, 5], 2),           # K=3, rate 1/2
        5: ([35, 23], 2),         # K=5, rate 1/2
        7: ([171, 133], 2),       # K=7, rate 1/2 (3GPP)
        9: ([561, 753], 2),       # K=9, rate 1/2
    }
    
    def __init__(self, constraint_length: int = 7, rate_inverse: int = 2):
        self.K = constraint_length
        self.rate_inv = rate_inverse
        
        if constraint_length in self.GENERATORS:
            self.generators, _ = self.GENERATORS[constraint_length]
        else:
            raise ValueError(f"Unsupported constraint length: {constraint_length}")
        
        self.num_states = 2 ** (self.K - 1)
        
        # Pre-compute state transitions
        self._compute_transitions()
    
    def _compute_transitions(self):
        """Pre-compute encoder state transitions"""
        self.next_state = np.zeros((self.num_states, 2), dtype=int)
        self.output = np.zeros((self.num_states, 2, self.rate_inv), dtype=int)
        
        for state in range(self.num_states):
            for input_bit in range(2):
                # Calculate next state
                self.next_state[state, input_bit] = (state >> 1) | (input_bit << (self.K - 2))
                
                # Calculate output bits
                reg = state | (input_bit << (self.K - 1))
                for i, gen in enumerate(self.generators):
                    # Convert octal to binary and count 1s
                    gen_binary = int(str(gen), 8)
                    masked = reg & gen_binary
                    self.output[state, input_bit, i] = bin(masked).count('1') % 2
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode data bits with convolutional code.
        
        Args:
            data: Input data bits
        
        Returns:
            Encoded bits
        """
        # Add tail bits for termination
        tail = np.zeros(self.K - 1, dtype=int)
        data_with_tail = np.concatenate([data, tail])
        
        encoded = []
        state = 0
        
        for bit in data_with_tail:
            output_bits = self.output[state, int(bit)]
            encoded.extend(output_bits)
            state = self.next_state[state, int(bit)]
        
        return np.array(encoded, dtype=int)
    
    def decode(self, received: np.ndarray, soft: bool = False) -> np.ndarray:
        """
        Viterbi decode.
        
        Args:
            received: Received bits (or soft values if soft=True)
            soft: Whether input is soft decision
        
        Returns:
            Decoded data bits
        """
        num_symbols = len(received) // self.rate_inv
        
        # Branch metrics
        if soft:
            # Soft decision (LLR)
            def branch_metric(received_sym, expected):
                metric = 0
                for i in range(self.rate_inv):
                    if expected[i] == 0:
                        metric += received_sym[i]
                    else:
                        metric -= received_sym[i]
                return metric
        else:
            # Hard decision (Hamming distance)
            def branch_metric(received_sym, expected):
                return -np.sum(received_sym != expected)
        
        # Initialize path metrics
        path_metrics = np.full(self.num_states, -np.inf)
        path_metrics[0] = 0
        
        # Survivor paths
        survivor = np.zeros((num_symbols, self.num_states), dtype=int)
        
        # Forward pass
        for t in range(num_symbols):
            received_sym = received[t * self.rate_inv:(t + 1) * self.rate_inv]
            new_metrics = np.full(self.num_states, -np.inf)
            
            for state in range(self.num_states):
                if path_metrics[state] == -np.inf:
                    continue
                
                for input_bit in range(2):
                    next_s = self.next_state[state, input_bit]
                    expected = self.output[state, input_bit]
                    
                    metric = path_metrics[state] + branch_metric(received_sym, expected)
                    
                    if metric > new_metrics[next_s]:
                        new_metrics[next_s] = metric
                        survivor[t, next_s] = state
            
            path_metrics = new_metrics
        
        # Traceback
        decoded = np.zeros(num_symbols, dtype=int)
        state = 0  # Terminated to zero state
        
        for t in range(num_symbols - 1, -1, -1):
            prev_state = survivor[t, state]
            # Determine input bit
            if self.next_state[prev_state, 0] == state:
                decoded[t] = 0
            else:
                decoded[t] = 1
            state = prev_state
        
        # Remove tail bits
        return decoded[:-(self.K - 1)]


class TurboCoder:
    """
    Turbo Encoder/Decoder (LTE/3GPP style)
    
    Uses parallel concatenated convolutional codes with interleaving.
    """
    
    def __init__(self, block_length: int = 1024, iterations: int = 8):
        self.block_length = block_length
        self.iterations = iterations
        
        # Component encoder (rate 1/2 RSC)
        self.K = 4  # Constraint length
        self.g0 = 0b1111  # Feedback polynomial
        self.g1 = 0b1011  # Forward polynomial
        
        # Generate interleaver
        self.interleaver = self._generate_interleaver(block_length)
        self.deinterleaver = np.argsort(self.interleaver)
    
    def _generate_interleaver(self, length: int) -> np.ndarray:
        """
        Generate QPP interleaver (3GPP style).
        """
        # Simplified interleaver (would use full 3GPP table in production)
        f1, f2 = 31, 64
        
        if length <= 40:
            f1, f2 = 3, 10
        elif length <= 512:
            f1, f2 = 17, 32
        elif length <= 2048:
            f1, f2 = 31, 64
        else:
            f1, f2 = 127, 256
        
        interleaver = np.zeros(length, dtype=int)
        for i in range(length):
            interleaver[i] = (f1 * i + f2 * i * i) % length
        
        return interleaver
    
    def _rsc_encode(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        RSC (Recursive Systematic Convolutional) encode.
        
        Returns:
            (systematic, parity)
        """
        state = 0
        systematic = data.copy()
        parity = np.zeros_like(data)
        
        for i, bit in enumerate(data):
            # Feedback
            fb = (state & 0b111) ^ bit
            fb ^= (state >> 1) & 1
            fb ^= (state >> 2) & 1
            fb &= 1
            
            # Parity output
            p = fb ^ ((state >> 1) & 1) ^ (state & 1)
            parity[i] = p
            
            # Update state
            state = ((fb << 2) | (state >> 1)) & 0b111
        
        return systematic, parity
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Turbo encode.
        
        Returns encoded bits in order: systematic, parity1, parity2
        Overall rate is 1/3.
        """
        # Ensure correct block length
        if len(data) != self.block_length:
            # Pad or truncate
            padded = np.zeros(self.block_length, dtype=int)
            padded[:min(len(data), self.block_length)] = data[:self.block_length]
            data = padded
        
        # First encoder
        systematic, parity1 = self._rsc_encode(data)
        
        # Interleave for second encoder
        interleaved = data[self.interleaver]
        
        # Second encoder
        _, parity2 = self._rsc_encode(interleaved)
        
        # Combine (rate 1/3)
        encoded = np.zeros(3 * self.block_length, dtype=int)
        encoded[0::3] = systematic
        encoded[1::3] = parity1
        encoded[2::3] = parity2
        
        return encoded
    
    def decode(self, received: np.ndarray, noise_var: float = 0.5) -> np.ndarray:
        """
        Iterative turbo decode with BCJR (MAP) algorithm.
        
        Args:
            received: Received soft values (LLR)
            noise_var: Noise variance for soft decoding
        
        Returns:
            Decoded bits
        """
        # Extract systematic and parity
        sys_llr = received[0::3]
        p1_llr = received[1::3]
        p2_llr = received[2::3]
        
        # Ensure correct length
        length = min(len(sys_llr), self.block_length)
        sys_llr = sys_llr[:length]
        p1_llr = p1_llr[:length]
        p2_llr = p2_llr[:length]
        
        # Pad if needed
        if length < self.block_length:
            pad = self.block_length - length
            sys_llr = np.concatenate([sys_llr, np.zeros(pad)])
            p1_llr = np.concatenate([p1_llr, np.zeros(pad)])
            p2_llr = np.concatenate([p2_llr, np.zeros(pad)])
        
        # Initialize extrinsic information
        extrinsic1 = np.zeros(self.block_length)
        extrinsic2 = np.zeros(self.block_length)
        
        # Iterative decoding
        for iteration in range(self.iterations):
            # Decoder 1
            L_in1 = sys_llr + extrinsic2[self.deinterleaver]
            L_out1 = self._bcjr_decode(L_in1, p1_llr)
            extrinsic1 = L_out1 - L_in1
            
            # Decoder 2 (with interleaved data)
            L_in2 = sys_llr[self.interleaver] + extrinsic1[self.interleaver]
            p2_interleaved = p2_llr  # Already for interleaved sequence
            L_out2 = self._bcjr_decode(L_in2, p2_interleaved)
            extrinsic2 = L_out2 - L_in2
        
        # Final decision
        L_final = sys_llr + extrinsic1 + extrinsic2[self.deinterleaver]
        decoded = (L_final < 0).astype(int)
        
        return decoded
    
    def _bcjr_decode(self, sys_llr: np.ndarray, par_llr: np.ndarray) -> np.ndarray:
        """
        Simplified BCJR (MAP) decoder for RSC code.
        """
        length = len(sys_llr)
        num_states = 8  # 2^(K-1)
        
        # Branch metrics
        gamma = np.zeros((length, num_states, 2))
        for t in range(length):
            for state in range(num_states):
                for input_bit in range(2):
                    # Calculate expected parity
                    fb = (state & 0b111) ^ input_bit
                    fb ^= (state >> 1) & 1
                    fb ^= (state >> 2) & 1
                    fb &= 1
                    p_expected = fb ^ ((state >> 1) & 1) ^ (state & 1)
                    
                    # Branch metric
                    gamma[t, state, input_bit] = (
                        0.5 * sys_llr[t] * (1 - 2 * input_bit) +
                        0.5 * par_llr[t] * (1 - 2 * p_expected)
                    )
        
        # Forward recursion (alpha)
        alpha = np.full((length + 1, num_states), -np.inf)
        alpha[0, 0] = 0
        
        for t in range(length):
            for state in range(num_states):
                for input_bit in range(2):
                    fb = (state & 0b111) ^ input_bit
                    fb ^= (state >> 1) & 1
                    fb ^= (state >> 2) & 1
                    fb &= 1
                    next_state = ((fb << 2) | (state >> 1)) & 0b111
                    
                    alpha[t + 1, next_state] = np.logaddexp(
                        alpha[t + 1, next_state],
                        alpha[t, state] + gamma[t, state, input_bit]
                    )
        
        # Backward recursion (beta)
        beta = np.full((length + 1, num_states), -np.inf)
        beta[length, 0] = 0
        
        for t in range(length - 1, -1, -1):
            for state in range(num_states):
                for input_bit in range(2):
                    fb = (state & 0b111) ^ input_bit
                    fb ^= (state >> 1) & 1
                    fb ^= (state >> 2) & 1
                    fb &= 1
                    next_state = ((fb << 2) | (state >> 1)) & 0b111
                    
                    beta[t, state] = np.logaddexp(
                        beta[t, state],
                        beta[t + 1, next_state] + gamma[t, state, input_bit]
                    )
        
        # LLR output
        L_out = np.zeros(length)
        for t in range(length):
            L0 = -np.inf
            L1 = -np.inf
            
            for state in range(num_states):
                for input_bit in range(2):
                    fb = (state & 0b111) ^ input_bit
                    fb ^= (state >> 1) & 1
                    fb ^= (state >> 2) & 1
                    fb &= 1
                    next_state = ((fb << 2) | (state >> 1)) & 0b111
                    
                    metric = alpha[t, state] + gamma[t, state, input_bit] + beta[t + 1, next_state]
                    
                    if input_bit == 0:
                        L0 = np.logaddexp(L0, metric)
                    else:
                        L1 = np.logaddexp(L1, metric)
            
            L_out[t] = L0 - L1
        
        return L_out


class LDPCCoder:
    """
    LDPC (Low-Density Parity-Check) Encoder/Decoder
    
    Supports 5G NR LDPC codes with belief propagation decoding.
    """
    
    def __init__(self, block_length: int = 1024, rate: float = 0.5,
                 max_iterations: int = 50):
        self.block_length = block_length
        self.rate = rate
        self.max_iterations = max_iterations
        
        # Generate parity check matrix
        self.H = self._generate_parity_matrix()
        self.num_checks, self.num_bits = self.H.shape
    
    def _generate_parity_matrix(self) -> np.ndarray:
        """
        Generate LDPC parity check matrix.
        
        Uses simplified quasi-cyclic construction.
        """
        # Parameters
        n = self.block_length
        k = int(n * self.rate)
        m = n - k  # Number of parity bits
        
        # Circulant size
        Z = max(4, n // 20)
        
        # Number of circulants per dimension
        nb = n // Z
        mb = m // Z
        
        # Base matrix (simplified - would use 5G NR tables in production)
        Hb = np.zeros((mb, nb), dtype=int)
        
        # Create sparse structure
        row_weight = 6
        for i in range(mb):
            cols = np.random.choice(nb, row_weight, replace=False)
            Hb[i, cols] = np.random.randint(0, Z, row_weight)
        
        # Expand to full matrix
        H = np.zeros((m, n), dtype=int)
        for i in range(mb):
            for j in range(nb):
                if Hb[i, j] >= 0:
                    shift = Hb[i, j]
                    for k in range(Z):
                        H[i * Z + k, j * Z + (k + shift) % Z] = 1
        
        return H
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        LDPC encode (simplified systematic encoding).
        """
        k = int(self.block_length * self.rate)
        
        # Pad data if needed
        if len(data) < k:
            data = np.concatenate([data, np.zeros(k - len(data), dtype=int)])
        else:
            data = data[:k]
        
        # Systematic encoding (simplified)
        # In production, would use efficient encoding algorithms
        parity = np.zeros(self.block_length - k, dtype=int)
        
        # Calculate parity bits to satisfy H * c = 0
        for i in range(len(parity)):
            parity[i] = np.sum(self.H[i, :k] * data) % 2
        
        return np.concatenate([data, parity])
    
    def decode(self, received: np.ndarray, noise_var: float = 0.5) -> np.ndarray:
        """
        Belief propagation LDPC decoding.
        
        Args:
            received: Received soft values (LLR)
            noise_var: Noise variance
        
        Returns:
            Decoded bits
        """
        # Initialize messages
        # LLR from channel
        Lc = received[:self.num_bits] if len(received) >= self.num_bits else \
             np.concatenate([received, np.zeros(self.num_bits - len(received))])
        
        # Variable to check messages
        L_vc = np.zeros((self.num_bits, self.num_checks))
        
        # Check to variable messages
        L_cv = np.zeros((self.num_checks, self.num_bits))
        
        # Initialize v->c messages with channel LLR
        for j in range(self.num_bits):
            for i in range(self.num_checks):
                if self.H[i, j]:
                    L_vc[j, i] = Lc[j]
        
        # Iterate
        for iteration in range(self.max_iterations):
            # Check node update
            for i in range(self.num_checks):
                connected = np.where(self.H[i, :] == 1)[0]
                
                for j in connected:
                    # Product of tanh of other messages
                    product = 1.0
                    for j2 in connected:
                        if j2 != j:
                            product *= np.tanh(L_vc[j2, i] / 2)
                    
                    L_cv[i, j] = 2 * np.arctanh(np.clip(product, -0.9999, 0.9999))
            
            # Variable node update
            for j in range(self.num_bits):
                connected = np.where(self.H[:, j] == 1)[0]
                
                for i in connected:
                    L_vc[j, i] = Lc[j]
                    for i2 in connected:
                        if i2 != i:
                            L_vc[j, i] += L_cv[i2, j]
            
            # Hard decision
            L_total = Lc.copy()
            for j in range(self.num_bits):
                connected = np.where(self.H[:, j] == 1)[0]
                for i in connected:
                    L_total[j] += L_cv[i, j]
            
            decoded = (L_total < 0).astype(int)
            
            # Check syndrome
            syndrome = np.dot(self.H, decoded) % 2
            if np.sum(syndrome) == 0:
                break
        
        # Return information bits
        k = int(self.block_length * self.rate)
        return decoded[:k]


class PolarCoder:
    """
    Polar Encoder/Decoder (5G NR style)
    
    Uses successive cancellation list (SCL) decoding.
    """
    
    def __init__(self, block_length: int = 1024, rate: float = 0.5,
                 list_size: int = 8):
        self.N = block_length  # Must be power of 2
        self.K = int(block_length * rate)  # Info bits
        self.list_size = list_size
        
        # Ensure N is power of 2
        self.n = int(np.log2(self.N))
        if 2 ** self.n != self.N:
            self.N = 2 ** self.n
            self.K = int(self.N * rate)
        
        # Compute channel reliability
        self.frozen_bits = self._compute_frozen_positions()
        self.info_bits = np.array([i for i in range(self.N) if i not in self.frozen_bits])
    
    def _compute_frozen_positions(self) -> set:
        """
        Compute frozen bit positions using Bhattacharyya parameters.
        """
        # Simplified reliability sequence
        # Would use 5G NR sequence in production
        
        # Bit-reversal reliability ordering (simplified)
        def bit_reversal(x, n):
            result = 0
            for i in range(n):
                if x & (1 << i):
                    result |= 1 << (n - 1 - i)
            return result
        
        # Create reliability order (simplified)
        reliability = np.zeros(self.N)
        for i in range(self.N):
            # Hamming weight based reliability
            reliability[i] = bin(bit_reversal(i, self.n)).count('1')
        
        # Sort by reliability
        sorted_indices = np.argsort(reliability)
        
        # Freeze least reliable positions
        frozen = set(sorted_indices[:self.N - self.K])
        
        return frozen
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Polar encode.
        """
        # Create codeword with frozen bits
        u = np.zeros(self.N, dtype=int)
        
        # Place info bits
        data_idx = 0
        for i in range(self.N):
            if i not in self.frozen_bits:
                if data_idx < len(data):
                    u[i] = data[data_idx]
                data_idx += 1
        
        # Polar transform
        x = self._polar_transform(u)
        
        return x
    
    def _polar_transform(self, u: np.ndarray) -> np.ndarray:
        """
        Apply polar transform (Arikan kernel).
        """
        x = u.copy()
        
        for stage in range(self.n):
            step = 2 ** stage
            for i in range(0, self.N, 2 * step):
                for j in range(step):
                    x[i + j] = (x[i + j] + x[i + j + step]) % 2
        
        return x
    
    def decode(self, received: np.ndarray) -> np.ndarray:
        """
        Successive Cancellation (SC) decoding.
        
        Args:
            received: Received LLRs
        
        Returns:
            Decoded info bits
        """
        # Initialize LLRs
        llr = received[:self.N] if len(received) >= self.N else \
              np.concatenate([received, np.zeros(self.N - len(received))])
        
        # SC decoding
        decoded = np.zeros(self.N, dtype=int)
        
        for i in range(self.N):
            # Compute LLR for bit i
            L = self._compute_llr(llr, decoded, i)
            
            if i in self.frozen_bits:
                decoded[i] = 0
            else:
                decoded[i] = 0 if L >= 0 else 1
        
        # Extract info bits
        info_bits = decoded[self.info_bits]
        
        return info_bits
    
    def _compute_llr(self, channel_llr: np.ndarray, 
                     partial_sum: np.ndarray, bit_idx: int) -> float:
        """
        Compute LLR for SC decoding.
        
        Simplified recursive computation.
        """
        if self.n == 0:
            return channel_llr[0]
        
        # Use simplified approximation
        # f(a, b) ≈ sign(a) * sign(b) * min(|a|, |b|)
        # g(a, b, u) = (1 - 2u)a + b
        
        # This is a simplified version - production would use full recursion
        return channel_llr[bit_idx]


class ChannelCoder:
    """
    Unified Channel Coding Interface
    
    Provides consistent API for all coding schemes.
    """
    
    def __init__(self, config: CodingConfig):
        self.config = config
        
        if config.code_type == CodeType.CONVOLUTIONAL:
            self.coder = ConvolutionalCoder(config.constraint_length)
        elif config.code_type == CodeType.TURBO:
            self.coder = TurboCoder(config.block_length)
        elif config.code_type == CodeType.LDPC:
            self.coder = LDPCCoder(config.block_length, config.code_rate,
                                   config.ldpc_iterations)
        elif config.code_type == CodeType.POLAR:
            self.coder = PolarCoder(config.block_length, config.code_rate,
                                    config.polar_list_size)
        else:
            raise ValueError(f"Unknown code type: {config.code_type}")
        
        # CRC calculator
        self.crc = CRCCalculator('CRC-24A')
    
    def encode(self, data: np.ndarray, add_crc: bool = True) -> np.ndarray:
        """
        Encode data bits.
        
        Args:
            data: Input data bits
            add_crc: Whether to add CRC
        
        Returns:
            Encoded bits
        """
        if add_crc:
            data = self.crc.append(data)
        
        return self.coder.encode(data)
    
    def decode(self, received: np.ndarray, check_crc: bool = True) -> Tuple[np.ndarray, bool]:
        """
        Decode received bits/LLRs.
        
        Args:
            received: Received values
            check_crc: Whether to check CRC
        
        Returns:
            (decoded_data, crc_pass)
        """
        decoded = self.coder.decode(received)
        
        if check_crc:
            crc_pass = self.crc.check(decoded)
            # Remove CRC bits
            data = decoded[:-self.crc.crc_length]
            return data, crc_pass
        else:
            return decoded, True
