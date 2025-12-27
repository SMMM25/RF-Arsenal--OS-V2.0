#!/usr/bin/env python3
"""
RF Arsenal OS - LoRa Physical Layer Module
===========================================

Production-grade LoRa Chirp Spread Spectrum implementation.

Features:
- Complete Semtech LoRa PHY specification
- Signal generation (TX) and demodulation (RX)
- All spreading factors (SF5-SF12)
- All bandwidth options (7.8 kHz - 500 kHz)
- Forward Error Correction (CR 4/5 - 4/8)
- Regional frequency plan support
- Gray coding and interleaving
- CRC-16 CCITT

Supported Regions:
- US915: 902-928 MHz
- EU868: 863-870 MHz  
- AU915: 915-928 MHz
- AS923: 920-925 MHz
- IN865: 865-867 MHz
- KR920: 920-923 MHz
- CN470: 470-510 MHz

README COMPLIANCE:
✅ Real-World Functional: Actual LoRa modulation mathematics
✅ Thread-Safe: Proper locking for hardware access
✅ Stealth: No external communications
✅ Validated: All RF parameters validated per regional regulations
"""

from .phy import (
    LoRaPHY,
    LoRaConfig,
    LoRaPacket,
    LoRaSymbol,
    SpreadingFactor,
    Bandwidth,
    CodingRate,
    LoRaRegion,
    create_lora_phy,
)

__all__ = [
    'LoRaPHY',
    'LoRaConfig',
    'LoRaPacket',
    'LoRaSymbol',
    'SpreadingFactor',
    'Bandwidth',
    'CodingRate',
    'LoRaRegion',
    'create_lora_phy',
]

__version__ = '1.0.0'
