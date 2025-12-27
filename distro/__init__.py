#!/usr/bin/env python3
"""
RF Arsenal OS - Distribution Build System

Tools for creating custom RF Arsenal OS distributions based on DragonOS.

Supported Platforms:
- x86_64 (Desktop/Laptop)
- ARM64 (Generic ARM)
- Raspberry Pi 4/5

Features:
- Automated DragonOS integration
- Live USB with persistence
- RAM-only stealth mode
- Raspberry Pi GPIO support
- Mobile/small screen optimization
- OPSEC hardening

Usage:
    ./build_arsenal_os.sh --platform x86_64 --mode full
    ./build_arsenal_os.sh --platform rpi --mode lite --live-usb
"""

__version__ = "1.0.0"
__author__ = "RF Arsenal Team"
