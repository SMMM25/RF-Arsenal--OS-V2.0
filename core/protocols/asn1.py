#!/usr/bin/env python3
"""
RF Arsenal OS - ASN.1 Encoder/Decoder

Production-grade ASN.1 encoding for cellular protocols.
Supports PER (Packed Encoding Rules) as used in LTE/5G RRC.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import struct

logger = logging.getLogger(__name__)


class ASN1Types(Enum):
    """ASN.1 type identifiers"""
    BOOLEAN = 0x01
    INTEGER = 0x02
    BIT_STRING = 0x03
    OCTET_STRING = 0x04
    NULL = 0x05
    OBJECT_IDENTIFIER = 0x06
    ENUMERATED = 0x0A
    UTF8_STRING = 0x0C
    SEQUENCE = 0x10
    SET = 0x11
    PRINTABLE_STRING = 0x13
    IA5_STRING = 0x16
    UTC_TIME = 0x17
    CHOICE = 0x80  # Context-specific
    

@dataclass
class ASN1Constraint:
    """ASN.1 constraint specification"""
    min_value: Optional[int] = None
    max_value: Optional[int] = None
    size_min: Optional[int] = None
    size_max: Optional[int] = None
    extensible: bool = False


class BitBuffer:
    """Bit-level buffer for PER encoding"""
    
    def __init__(self, data: Optional[bytes] = None):
        if data:
            self.buffer = bytearray(data)
            self.bit_offset = 0
            self.byte_offset = 0
        else:
            self.buffer = bytearray()
            self.bit_offset = 0
            self.byte_offset = 0
    
    def write_bit(self, bit: int):
        """Write single bit"""
        if self.bit_offset == 0:
            self.buffer.append(0)
        
        if bit:
            self.buffer[-1] |= (1 << (7 - self.bit_offset))
        
        self.bit_offset += 1
        if self.bit_offset == 8:
            self.bit_offset = 0
    
    def write_bits(self, value: int, num_bits: int):
        """Write multiple bits"""
        for i in range(num_bits - 1, -1, -1):
            self.write_bit((value >> i) & 1)
    
    def read_bit(self) -> int:
        """Read single bit"""
        if self.byte_offset >= len(self.buffer):
            return 0
        
        bit = (self.buffer[self.byte_offset] >> (7 - self.bit_offset)) & 1
        
        self.bit_offset += 1
        if self.bit_offset == 8:
            self.bit_offset = 0
            self.byte_offset += 1
        
        return bit
    
    def read_bits(self, num_bits: int) -> int:
        """Read multiple bits"""
        value = 0
        for _ in range(num_bits):
            value = (value << 1) | self.read_bit()
        return value
    
    def align(self):
        """Align to byte boundary"""
        if self.bit_offset != 0:
            self.bit_offset = 0
            if len(self.buffer) > 0:
                self.byte_offset += 1
    
    def get_bytes(self) -> bytes:
        """Get encoded bytes"""
        return bytes(self.buffer)
    
    def remaining_bits(self) -> int:
        """Get remaining bits to read"""
        return (len(self.buffer) - self.byte_offset) * 8 - self.bit_offset


class ASN1Encoder:
    """
    ASN.1 BER/DER Encoder
    
    Basic Encoding Rules for general ASN.1 encoding.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('ASN1Encoder')
    
    def encode_length(self, length: int) -> bytes:
        """Encode length field"""
        if length < 128:
            return bytes([length])
        else:
            # Long form
            length_bytes = []
            temp = length
            while temp > 0:
                length_bytes.insert(0, temp & 0xFF)
                temp >>= 8
            
            return bytes([0x80 | len(length_bytes)] + length_bytes)
    
    def encode_boolean(self, value: bool) -> bytes:
        """Encode BOOLEAN"""
        return bytes([ASN1Types.BOOLEAN.value, 1, 0xFF if value else 0x00])
    
    def encode_integer(self, value: int) -> bytes:
        """Encode INTEGER"""
        # Determine minimum bytes needed
        if value == 0:
            content = bytes([0])
        elif value > 0:
            content = value.to_bytes((value.bit_length() + 8) // 8, 'big')
            # Remove leading zeros but keep one if high bit is set
            while len(content) > 1 and content[0] == 0 and content[1] < 128:
                content = content[1:]
        else:
            # Negative (two's complement)
            byte_len = (value.bit_length() + 9) // 8
            content = (value + (1 << (byte_len * 8))).to_bytes(byte_len, 'big')
        
        return bytes([ASN1Types.INTEGER.value]) + self.encode_length(len(content)) + content
    
    def encode_bit_string(self, bits: Union[bytes, np.ndarray], 
                         num_bits: Optional[int] = None) -> bytes:
        """Encode BIT STRING"""
        if isinstance(bits, np.ndarray):
            # Convert bit array to bytes
            num_bits = len(bits) if num_bits is None else num_bits
            num_bytes = (num_bits + 7) // 8
            padded = np.zeros(num_bytes * 8, dtype=int)
            padded[:num_bits] = bits[:num_bits]
            bits = bytes(np.packbits(padded))
        
        if num_bits is None:
            num_bits = len(bits) * 8
        
        # Calculate unused bits in last byte
        unused = (8 - (num_bits % 8)) % 8
        
        content = bytes([unused]) + bits
        return bytes([ASN1Types.BIT_STRING.value]) + self.encode_length(len(content)) + content
    
    def encode_octet_string(self, data: bytes) -> bytes:
        """Encode OCTET STRING"""
        return bytes([ASN1Types.OCTET_STRING.value]) + self.encode_length(len(data)) + data
    
    def encode_null(self) -> bytes:
        """Encode NULL"""
        return bytes([ASN1Types.NULL.value, 0])
    
    def encode_enumerated(self, value: int) -> bytes:
        """Encode ENUMERATED"""
        content = value.to_bytes((value.bit_length() + 8) // 8, 'big') if value else bytes([0])
        return bytes([ASN1Types.ENUMERATED.value]) + self.encode_length(len(content)) + content
    
    def encode_sequence(self, *elements: bytes) -> bytes:
        """Encode SEQUENCE"""
        content = b''.join(elements)
        return bytes([ASN1Types.SEQUENCE.value | 0x20]) + self.encode_length(len(content)) + content
    
    def encode_set(self, *elements: bytes) -> bytes:
        """Encode SET"""
        content = b''.join(elements)
        return bytes([ASN1Types.SET.value | 0x20]) + self.encode_length(len(content)) + content
    
    def encode_utf8_string(self, value: str) -> bytes:
        """Encode UTF8String"""
        content = value.encode('utf-8')
        return bytes([ASN1Types.UTF8_STRING.value]) + self.encode_length(len(content)) + content
    
    def encode_context_specific(self, tag: int, content: bytes, 
                                constructed: bool = False) -> bytes:
        """Encode context-specific tagged value"""
        tag_byte = 0x80 | tag
        if constructed:
            tag_byte |= 0x20
        return bytes([tag_byte]) + self.encode_length(len(content)) + content


class ASN1Decoder:
    """
    ASN.1 BER/DER Decoder
    """
    
    def __init__(self):
        self.logger = logging.getLogger('ASN1Decoder')
    
    def decode_length(self, data: bytes, offset: int) -> Tuple[int, int]:
        """
        Decode length field.
        
        Returns:
            (length, new_offset)
        """
        if data[offset] < 128:
            return data[offset], offset + 1
        else:
            num_bytes = data[offset] & 0x7F
            length = 0
            for i in range(num_bytes):
                length = (length << 8) | data[offset + 1 + i]
            return length, offset + 1 + num_bytes
    
    def decode_tlv(self, data: bytes, offset: int = 0) -> Tuple[int, int, bytes, int]:
        """
        Decode Tag-Length-Value.
        
        Returns:
            (tag, constructed, value, new_offset)
        """
        tag = data[offset]
        constructed = bool(tag & 0x20)
        
        length, offset = self.decode_length(data, offset + 1)
        value = data[offset:offset + length]
        
        return tag & 0x1F, constructed, value, offset + length
    
    def decode_boolean(self, data: bytes) -> bool:
        """Decode BOOLEAN"""
        return data[0] != 0
    
    def decode_integer(self, data: bytes) -> int:
        """Decode INTEGER"""
        value = int.from_bytes(data, 'big', signed=True)
        return value
    
    def decode_bit_string(self, data: bytes) -> Tuple[bytes, int]:
        """
        Decode BIT STRING.
        
        Returns:
            (bits, num_bits)
        """
        unused = data[0]
        bits = data[1:]
        num_bits = len(bits) * 8 - unused
        return bits, num_bits
    
    def decode_octet_string(self, data: bytes) -> bytes:
        """Decode OCTET STRING"""
        return data
    
    def decode_enumerated(self, data: bytes) -> int:
        """Decode ENUMERATED"""
        return int.from_bytes(data, 'big')
    
    def decode_utf8_string(self, data: bytes) -> str:
        """Decode UTF8String"""
        return data.decode('utf-8')
    
    def decode(self, data: bytes) -> Dict[str, Any]:
        """
        Decode ASN.1 structure recursively.
        
        Returns dict with type and value.
        """
        tag, constructed, value, _ = self.decode_tlv(data)
        
        result = {'tag': tag, 'constructed': constructed}
        
        if tag == ASN1Types.BOOLEAN.value:
            result['type'] = 'BOOLEAN'
            result['value'] = self.decode_boolean(value)
        elif tag == ASN1Types.INTEGER.value:
            result['type'] = 'INTEGER'
            result['value'] = self.decode_integer(value)
        elif tag == ASN1Types.BIT_STRING.value:
            result['type'] = 'BIT STRING'
            bits, num_bits = self.decode_bit_string(value)
            result['value'] = bits
            result['num_bits'] = num_bits
        elif tag == ASN1Types.OCTET_STRING.value:
            result['type'] = 'OCTET STRING'
            result['value'] = value
        elif tag == ASN1Types.NULL.value:
            result['type'] = 'NULL'
            result['value'] = None
        elif tag == ASN1Types.ENUMERATED.value:
            result['type'] = 'ENUMERATED'
            result['value'] = self.decode_enumerated(value)
        elif tag == ASN1Types.SEQUENCE.value:
            result['type'] = 'SEQUENCE'
            result['value'] = self._decode_constructed(value)
        elif tag == ASN1Types.SET.value:
            result['type'] = 'SET'
            result['value'] = self._decode_constructed(value)
        elif tag == ASN1Types.UTF8_STRING.value:
            result['type'] = 'UTF8String'
            result['value'] = self.decode_utf8_string(value)
        else:
            result['type'] = 'UNKNOWN'
            result['value'] = value
        
        return result
    
    def _decode_constructed(self, data: bytes) -> List[Dict]:
        """Decode constructed type (SEQUENCE/SET)"""
        elements = []
        offset = 0
        
        while offset < len(data):
            tag, constructed, value, offset = self.decode_tlv(data, offset)
            elements.append(self.decode(bytes([tag | (0x20 if constructed else 0)]) + 
                                       self._encode_length(len(value)) + value))
        
        return elements
    
    def _encode_length(self, length: int) -> bytes:
        """Helper to encode length for reconstruction"""
        if length < 128:
            return bytes([length])
        else:
            length_bytes = []
            temp = length
            while temp > 0:
                length_bytes.insert(0, temp & 0xFF)
                temp >>= 8
            return bytes([0x80 | len(length_bytes)] + length_bytes)


class PEREncoder:
    """
    ASN.1 PER (Packed Encoding Rules) Encoder
    
    Used for LTE/5G RRC messages.
    Supports both aligned and unaligned variants.
    """
    
    def __init__(self, aligned: bool = True):
        self.aligned = aligned
        self.logger = logging.getLogger('PEREncoder')
    
    def encode_constrained_integer(self, value: int, 
                                   constraint: ASN1Constraint) -> BitBuffer:
        """
        Encode constrained whole number.
        
        Args:
            value: Integer value
            constraint: Value constraints (min, max)
        """
        buf = BitBuffer()
        
        lb = constraint.min_value if constraint.min_value is not None else 0
        ub = constraint.max_value
        
        if ub is None:
            # Semi-constrained or unconstrained
            return self.encode_unconstrained_integer(value - lb)
        
        range_val = ub - lb + 1
        offset_value = value - lb
        
        if range_val == 1:
            # No encoding needed
            pass
        elif range_val <= 256:
            # Single octet or less
            bits_needed = (range_val - 1).bit_length()
            buf.write_bits(offset_value, bits_needed)
        elif range_val <= 65536:
            # Two octets
            if self.aligned:
                buf.align()
            buf.write_bits(offset_value, 16)
        else:
            # Multiple octets
            if self.aligned:
                buf.align()
            # Length-determinant encoding
            num_bytes = (offset_value.bit_length() + 7) // 8
            buf.write_bits(num_bytes - 1, 8)  # Length
            buf.write_bits(offset_value, num_bytes * 8)
        
        return buf
    
    def encode_unconstrained_integer(self, value: int) -> BitBuffer:
        """Encode unconstrained integer"""
        buf = BitBuffer()
        
        if self.aligned:
            buf.align()
        
        if value == 0:
            buf.write_bits(1, 8)  # Length = 1
            buf.write_bits(0, 8)  # Value = 0
        else:
            # Determine number of octets
            if value > 0:
                num_bytes = (value.bit_length() + 7) // 8
                if value.bit_length() % 8 == 0:
                    num_bytes += 1  # Need extra byte for sign
            else:
                num_bytes = (value.bit_length() + 8) // 8
            
            buf.write_bits(num_bytes, 8)
            
            # Encode value
            if value >= 0:
                buf.write_bits(value, num_bytes * 8)
            else:
                # Two's complement
                buf.write_bits(value + (1 << (num_bytes * 8)), num_bytes * 8)
        
        return buf
    
    def encode_boolean(self, value: bool) -> BitBuffer:
        """Encode BOOLEAN"""
        buf = BitBuffer()
        buf.write_bit(1 if value else 0)
        return buf
    
    def encode_enumerated(self, value: int, num_values: int,
                         extensible: bool = False) -> BitBuffer:
        """Encode ENUMERATED"""
        buf = BitBuffer()
        
        if extensible:
            buf.write_bit(0)  # Not extended
        
        bits_needed = (num_values - 1).bit_length()
        buf.write_bits(value, bits_needed)
        
        return buf
    
    def encode_bit_string(self, bits: np.ndarray, 
                         constraint: ASN1Constraint) -> BitBuffer:
        """Encode BIT STRING"""
        buf = BitBuffer()
        
        length = len(bits)
        
        if constraint.size_min == constraint.size_max:
            # Fixed size
            if length <= 16:
                for bit in bits:
                    buf.write_bit(int(bit))
            else:
                if self.aligned:
                    buf.align()
                for bit in bits:
                    buf.write_bit(int(bit))
        else:
            # Variable size
            lb = constraint.size_min or 0
            ub = constraint.size_max
            
            if ub is not None and ub < 65536:
                # Constrained length
                range_val = ub - lb + 1
                bits_needed = (range_val - 1).bit_length()
                buf.write_bits(length - lb, bits_needed)
            else:
                # Unconstrained
                buf = self._encode_length_determinant(buf, length)
            
            if self.aligned and length > 16:
                buf.align()
            
            for bit in bits:
                buf.write_bit(int(bit))
        
        return buf
    
    def encode_octet_string(self, data: bytes,
                           constraint: ASN1Constraint) -> BitBuffer:
        """Encode OCTET STRING"""
        buf = BitBuffer()
        
        length = len(data)
        
        if constraint.size_min == constraint.size_max:
            # Fixed size
            if self.aligned and length > 2:
                buf.align()
            for byte in data:
                buf.write_bits(byte, 8)
        else:
            # Variable size
            lb = constraint.size_min or 0
            ub = constraint.size_max
            
            if ub is not None and ub < 65536:
                range_val = ub - lb + 1
                bits_needed = (range_val - 1).bit_length()
                buf.write_bits(length - lb, bits_needed)
            else:
                buf = self._encode_length_determinant(buf, length)
            
            if self.aligned:
                buf.align()
            
            for byte in data:
                buf.write_bits(byte, 8)
        
        return buf
    
    def encode_sequence_header(self, num_optional: int,
                               optional_bitmap: List[bool],
                               extensible: bool = False) -> BitBuffer:
        """Encode SEQUENCE preamble (extension bit + optional bitmap)"""
        buf = BitBuffer()
        
        if extensible:
            buf.write_bit(0)  # Not extended
        
        for present in optional_bitmap:
            buf.write_bit(1 if present else 0)
        
        return buf
    
    def encode_choice(self, index: int, num_choices: int,
                     extensible: bool = False) -> BitBuffer:
        """Encode CHOICE index"""
        buf = BitBuffer()
        
        if extensible:
            buf.write_bit(0)  # Not extended
        
        bits_needed = (num_choices - 1).bit_length()
        buf.write_bits(index, bits_needed)
        
        return buf
    
    def _encode_length_determinant(self, buf: BitBuffer, 
                                   length: int) -> BitBuffer:
        """Encode length determinant"""
        if length < 128:
            buf.write_bits(length, 8)
        elif length < 16384:
            buf.write_bit(1)
            buf.write_bit(0)
            buf.write_bits(length, 14)
        else:
            # Fragmented (not fully implemented)
            buf.write_bit(1)
            buf.write_bit(1)
            buf.write_bits(length, 6)  # Simplified
        
        return buf
    
    def combine(self, *buffers: BitBuffer) -> bytes:
        """Combine multiple bit buffers"""
        result = BitBuffer()
        
        for buf in buffers:
            data = buf.get_bytes()
            for byte in data:
                for i in range(8):
                    result.write_bit((byte >> (7 - i)) & 1)
        
        return result.get_bytes()


class PERDecoder:
    """
    ASN.1 PER Decoder
    """
    
    def __init__(self, aligned: bool = True):
        self.aligned = aligned
        self.logger = logging.getLogger('PERDecoder')
    
    def decode_constrained_integer(self, buf: BitBuffer,
                                   constraint: ASN1Constraint) -> int:
        """Decode constrained whole number"""
        lb = constraint.min_value if constraint.min_value is not None else 0
        ub = constraint.max_value
        
        if ub is None:
            return lb + self.decode_unconstrained_integer(buf)
        
        range_val = ub - lb + 1
        
        if range_val == 1:
            return lb
        elif range_val <= 256:
            bits_needed = (range_val - 1).bit_length()
            return lb + buf.read_bits(bits_needed)
        elif range_val <= 65536:
            if self.aligned:
                buf.align()
            return lb + buf.read_bits(16)
        else:
            if self.aligned:
                buf.align()
            num_bytes = buf.read_bits(8) + 1
            return lb + buf.read_bits(num_bytes * 8)
    
    def decode_unconstrained_integer(self, buf: BitBuffer) -> int:
        """Decode unconstrained integer"""
        if self.aligned:
            buf.align()
        
        num_bytes = buf.read_bits(8)
        value = buf.read_bits(num_bytes * 8)
        
        # Check sign bit
        if value >= (1 << (num_bytes * 8 - 1)):
            value -= (1 << (num_bytes * 8))
        
        return value
    
    def decode_boolean(self, buf: BitBuffer) -> bool:
        """Decode BOOLEAN"""
        return buf.read_bit() == 1
    
    def decode_enumerated(self, buf: BitBuffer, num_values: int,
                         extensible: bool = False) -> int:
        """Decode ENUMERATED"""
        if extensible:
            extended = buf.read_bit()
            if extended:
                # Handle extension (not implemented)
                return 0
        
        bits_needed = (num_values - 1).bit_length()
        return buf.read_bits(bits_needed)
    
    def decode_bit_string(self, buf: BitBuffer,
                         constraint: ASN1Constraint) -> np.ndarray:
        """Decode BIT STRING"""
        if constraint.size_min == constraint.size_max:
            length = constraint.size_min
        else:
            lb = constraint.size_min or 0
            ub = constraint.size_max
            
            if ub is not None and ub < 65536:
                range_val = ub - lb + 1
                bits_needed = (range_val - 1).bit_length()
                length = lb + buf.read_bits(bits_needed)
            else:
                length = self._decode_length_determinant(buf)
        
        if self.aligned and length > 16:
            buf.align()
        
        bits = np.zeros(length, dtype=int)
        for i in range(length):
            bits[i] = buf.read_bit()
        
        return bits
    
    def decode_octet_string(self, buf: BitBuffer,
                           constraint: ASN1Constraint) -> bytes:
        """Decode OCTET STRING"""
        if constraint.size_min == constraint.size_max:
            length = constraint.size_min
        else:
            lb = constraint.size_min or 0
            ub = constraint.size_max
            
            if ub is not None and ub < 65536:
                range_val = ub - lb + 1
                bits_needed = (range_val - 1).bit_length()
                length = lb + buf.read_bits(bits_needed)
            else:
                length = self._decode_length_determinant(buf)
        
        if self.aligned:
            buf.align()
        
        data = bytearray()
        for _ in range(length):
            data.append(buf.read_bits(8))
        
        return bytes(data)
    
    def decode_sequence_header(self, buf: BitBuffer,
                               num_optional: int,
                               extensible: bool = False) -> Tuple[bool, List[bool]]:
        """
        Decode SEQUENCE preamble.
        
        Returns:
            (extended, optional_bitmap)
        """
        extended = False
        if extensible:
            extended = buf.read_bit() == 1
        
        optional_bitmap = []
        for _ in range(num_optional):
            optional_bitmap.append(buf.read_bit() == 1)
        
        return extended, optional_bitmap
    
    def decode_choice(self, buf: BitBuffer, num_choices: int,
                     extensible: bool = False) -> int:
        """Decode CHOICE index"""
        if extensible:
            extended = buf.read_bit()
            if extended:
                # Handle extension (not implemented)
                return 0
        
        bits_needed = (num_choices - 1).bit_length()
        return buf.read_bits(bits_needed)
    
    def _decode_length_determinant(self, buf: BitBuffer) -> int:
        """Decode length determinant"""
        first_bit = buf.read_bit()
        
        if first_bit == 0:
            return buf.read_bits(7)
        else:
            second_bit = buf.read_bit()
            if second_bit == 0:
                return buf.read_bits(14)
            else:
                # Fragmented
                return buf.read_bits(6)  # Simplified
