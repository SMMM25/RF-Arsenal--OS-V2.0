"""
RF Arsenal OS - NFC/RFID Module

Real Proxmark3 integration for:
- LF (125kHz) cards: EM4100, HID Prox, T5577, Indala
- HF (13.56MHz) cards: Mifare Classic, Ultralight, DESFire, NTAG
- Key recovery attacks: Darkside, Nested, Hardnested, Dictionary
- Card cloning and emulation
- Communication sniffing

Hardware: Proxmark3 Easy/RDV4

WARNING: These capabilities require proper legal authorization.
Use only on cards you own or have explicit permission to test.
"""

from .proxmark3 import (
    Proxmark3Controller,
    RFIDCard,
    AttackResult,
    CardType,
    AttackType,
    get_proxmark3_controller
)

__all__ = [
    'Proxmark3Controller',
    'RFIDCard',
    'AttackResult',
    'CardType',
    'AttackType',
    'get_proxmark3_controller'
]

__version__ = '1.0.0'
