#!/usr/bin/env python3
"""
RF Arsenal OS - Input Validation Utilities
Provides secure input validation and sanitization to prevent injection attacks
"""

import re
import logging
from typing import Any, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation failures"""
    pass


class InputValidator:
    """
    Comprehensive input validation for security-critical operations
    Prevents command injection, path traversal, and other attacks
    """
    
    # Regular expressions for various input types
    ALPHANUMERIC = re.compile(r'^[a-zA-Z0-9]+$')
    ALPHANUMERIC_EXTENDED = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
    HOSTNAME = re.compile(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$')
    IPV4 = re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$')
    MAC_ADDRESS = re.compile(r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$')
    
    # Dangerous characters and patterns
    SHELL_DANGEROUS_CHARS = set(';&|`$()<>{}[]!*?\'"\\')
    DANGEROUS_PATTERNS = [
        '..',  # Path traversal
        '//',  # Double slash
        '\\x',  # Hex escape
        '%00',  # Null byte
        '\x00',  # Null byte
    ]
    
    @staticmethod
    def validate_string(value: str, 
                       min_length: int = 1, 
                       max_length: int = 255,
                       allow_special: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Validate string input
        
        Args:
            value: String to validate
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            allow_special: Allow special characters
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(value, str):
            return False, f"Expected string, got {type(value).__name__}"
        
        if len(value) < min_length:
            return False, f"String too short (minimum {min_length} characters)"
        
        if len(value) > max_length:
            return False, f"String too long (maximum {max_length} characters)"
        
        if not allow_special:
            if not InputValidator.ALPHANUMERIC_EXTENDED.match(value):
                return False, "String contains invalid characters"
        
        # Check for dangerous patterns
        for pattern in InputValidator.DANGEROUS_PATTERNS:
            if pattern in value:
                return False, f"String contains dangerous pattern: {pattern}"
        
        return True, None
    
    @staticmethod
    def validate_integer(value: Any, 
                        min_value: Optional[int] = None,
                        max_value: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate integer input
        
        Args:
            value: Value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            int_value = int(value)
        except (ValueError, TypeError):
            return False, f"Invalid integer: {value}"
        
        if min_value is not None and int_value < min_value:
            return False, f"Value {int_value} below minimum {min_value}"
        
        if max_value is not None and int_value > max_value:
            return False, f"Value {int_value} above maximum {max_value}"
        
        return True, None
    
    @staticmethod
    def validate_frequency(frequency: int) -> Tuple[bool, Optional[str]]:
        """
        Validate RF frequency (BladeRF range: 47 MHz - 6 GHz)
        
        Args:
            frequency: Frequency in Hz
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        MIN_FREQ = 47_000_000  # 47 MHz
        MAX_FREQ = 6_000_000_000  # 6 GHz
        
        return InputValidator.validate_integer(frequency, MIN_FREQ, MAX_FREQ)
    
    @staticmethod
    def validate_gain(gain: int, gain_type: str = "tx") -> Tuple[bool, Optional[str]]:
        """
        Validate RF gain value
        
        Args:
            gain: Gain in dB
            gain_type: "tx" or "rx"
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if gain_type == "tx":
            return InputValidator.validate_integer(gain, -89, 60)
        elif gain_type == "rx":
            return InputValidator.validate_integer(gain, 0, 60)
        else:
            return False, f"Invalid gain type: {gain_type}"
    
    @staticmethod
    def validate_mac_address(mac: str) -> Tuple[bool, Optional[str]]:
        """
        Validate MAC address format
        
        Args:
            mac: MAC address string
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(mac, str):
            return False, "MAC address must be a string"
        
        if not InputValidator.MAC_ADDRESS.match(mac):
            return False, "Invalid MAC address format (expected: XX:XX:XX:XX:XX:XX)"
        
        return True, None
    
    @staticmethod
    def validate_ipv4(ip: str) -> Tuple[bool, Optional[str]]:
        """
        Validate IPv4 address
        
        Args:
            ip: IP address string
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(ip, str):
            return False, "IP address must be a string"
        
        if not InputValidator.IPV4.match(ip):
            return False, "Invalid IPv4 address format"
        
        return True, None
    
    @staticmethod
    def validate_hostname(hostname: str) -> Tuple[bool, Optional[str]]:
        """
        Validate hostname format
        
        Args:
            hostname: Hostname string
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(hostname, str):
            return False, "Hostname must be a string"
        
        if len(hostname) > 253:
            return False, "Hostname too long (maximum 253 characters)"
        
        if not InputValidator.HOSTNAME.match(hostname):
            return False, "Invalid hostname format"
        
        return True, None
    
    @staticmethod
    def validate_path(path: str, must_exist: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Validate file system path (prevents path traversal)
        
        Args:
            path: File path string
            must_exist: Whether path must exist
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(path, str):
            return False, "Path must be a string"
        
        # Check for path traversal attempts
        for pattern in InputValidator.DANGEROUS_PATTERNS:
            if pattern in path:
                return False, f"Path contains dangerous pattern: {pattern}"
        
        # Convert to Path object for validation
        try:
            path_obj = Path(path).resolve()
        except Exception as e:
            return False, f"Invalid path: {e}"
        
        # Check if path must exist
        if must_exist and not path_obj.exists():
            return False, f"Path does not exist: {path}"
        
        return True, None
    
    @staticmethod
    def sanitize_for_shell(input_str: str) -> Optional[str]:
        """
        Sanitize input for use in shell commands (USE SPARINGLY - prefer list-based subprocess)
        
        Args:
            input_str: Input string
        
        Returns:
            Sanitized string or None if input contains dangerous characters
        """
        if not isinstance(input_str, str):
            logger.error(f"Expected string, got {type(input_str).__name__}")
            return None
        
        # Check for dangerous characters
        if any(char in input_str for char in InputValidator.SHELL_DANGEROUS_CHARS):
            logger.error(f"Input contains dangerous shell characters: {input_str}")
            return None
        
        # Check for dangerous patterns
        for pattern in InputValidator.DANGEROUS_PATTERNS:
            if pattern in input_str:
                logger.error(f"Input contains dangerous pattern: {pattern}")
                return None
        
        return input_str
    
    @staticmethod
    def validate_command_args(args: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate command arguments for subprocess execution
        
        Args:
            args: List of command arguments
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(args, list):
            return False, "Arguments must be a list"
        
        if not args:
            return False, "Arguments list cannot be empty"
        
        for i, arg in enumerate(args):
            if not isinstance(arg, str):
                return False, f"Argument {i} is not a string: {type(arg).__name__}"
            
            # Check for null bytes
            if '\x00' in arg:
                return False, f"Argument {i} contains null byte"
        
        return True, None


# Convenience functions for common validations

def require_valid_string(value: str, **kwargs) -> str:
    """Validate string or raise ValidationError"""
    is_valid, error = InputValidator.validate_string(value, **kwargs)
    if not is_valid:
        raise ValidationError(error)
    return value


def require_valid_integer(value: Any, **kwargs) -> int:
    """Validate integer or raise ValidationError"""
    is_valid, error = InputValidator.validate_integer(value, **kwargs)
    if not is_valid:
        raise ValidationError(error)
    return int(value)


def require_valid_frequency(frequency: int) -> int:
    """Validate frequency or raise ValidationError"""
    is_valid, error = InputValidator.validate_frequency(frequency)
    if not is_valid:
        raise ValidationError(error)
    return frequency


def require_valid_mac(mac: str) -> str:
    """Validate MAC address or raise ValidationError"""
    is_valid, error = InputValidator.validate_mac_address(mac)
    if not is_valid:
        raise ValidationError(error)
    return mac


def require_valid_path(path: str, **kwargs) -> str:
    """Validate path or raise ValidationError"""
    is_valid, error = InputValidator.validate_path(path, **kwargs)
    if not is_valid:
        raise ValidationError(error)
    return path
