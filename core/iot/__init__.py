#!/usr/bin/env python3
"""
RF Arsenal OS - IoT/Smart Home Attack Module

Comprehensive IoT security testing toolkit supporting:
- Zigbee (802.15.4) exploitation
- Z-Wave protocol attacks
- Smart lock vulnerabilities
- Smart meter manipulation
- Home automation hijacking
- BLE IoT device attacks

Hardware Requirements:
- BladeRF 2.0 micro xA9 (primary SDR)
- CC2531 USB dongle (Zigbee sniffer - optional)
- Z-Wave USB stick (optional)

⚠️ LEGAL WARNING: Only use on devices you own or have explicit authorization to test.
"""

from .zigbee import ZigbeeAttacker, ZigbeeDevice, ZigbeeNetwork
from .zwave import ZWaveAttacker, ZWaveDevice, ZWaveNetwork
from .smart_lock import SmartLockAttacker, SmartLock, LockVulnerability
from .smart_meter import SmartMeterAttacker, SmartMeter, MeterProtocol
from .home_automation import HomeAutomationAttacker, SmartHub, AutomationProtocol

__all__ = [
    # Zigbee
    'ZigbeeAttacker',
    'ZigbeeDevice', 
    'ZigbeeNetwork',
    # Z-Wave
    'ZWaveAttacker',
    'ZWaveDevice',
    'ZWaveNetwork',
    # Smart Locks
    'SmartLockAttacker',
    'SmartLock',
    'LockVulnerability',
    # Smart Meters
    'SmartMeterAttacker',
    'SmartMeter',
    'MeterProtocol',
    # Home Automation
    'HomeAutomationAttacker',
    'SmartHub',
    'AutomationProtocol',
]

__version__ = "1.0.0"
__author__ = "RF Arsenal OS Team"
