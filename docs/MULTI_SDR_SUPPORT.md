# RF Arsenal OS - Multi-SDR Support Guide

## Overview

RF Arsenal OS supports **6 different SDR platforms** for maximum flexibility and budget options.

---

## Supported Hardware

| SDR | Price | TX | RX | Full-Duplex | Best Use |
|-----|-------|----|----|-------------|----------|
| **BladeRF 2.0** | $450 | ✅ | ✅ | ✅ | Professional pentesting |
| **HackRF One** | $300 | ✅ | ✅ | ❌ | Budget all-around |
| **LimeSDR Mini** | $159 | ✅ | ✅ | ✅ | Budget full-duplex |
| **PlutoSDR** | $150 | ✅ | ✅ | ✅ | Portable operations |
| **USRP B200/B210** | $700+ | ✅ | ✅ | ✅ | Research/professional |
| **RTL-SDR** | $25 | ❌ | ✅ | ❌ | Passive monitoring |

---

## Quick Start

### Auto-Detection
```python
from core.hardware_abstraction import SDRHardwareAbstraction

hal = SDRHardwareAbstraction()
detected = hal.auto_detect_sdr()
print(f"Found {len(detected)} SDR(s)")
```

### Select Best SDR
```python
requirements = {
    'frequency': 2_450_000_000,  # 2.45 GHz
    'bandwidth': 20_000_000,      # 20 MHz
    'tx_required': True,
    'full_duplex': False
}

best_sdr = hal.select_best_sdr(requirements)
```

---

## Installation

### BladeRF
```bash
sudo apt-get install bladerf libbladerf-dev
python3 -m pip install bladerf
```

### HackRF
```bash
sudo apt-get install hackrf libhackrf-dev
sudo usermod -a -G plugdev $USER
```

### LimeSDR
```bash
sudo add-apt-repository -y ppa:myriadrf/drivers
sudo apt-get update
sudo apt-get install limesuite liblimesuite-dev soapysdr-module-lms7
```

### PlutoSDR
```bash
sudo apt-get install libiio-utils libiio-dev python3-iio
python3 -m pip install pyadi-iio
```

### USRP
```bash
sudo apt-get install libuhd-dev uhd-host
python3 -m pip install uhd
sudo uhd_images_downloader
```

### RTL-SDR
```bash
sudo apt-get install rtl-sdr librtlsdr-dev
python3 -m pip install pyrtlsdr
```

---

## Feature Compatibility

### ✅ Fully Compatible (All Operations)
**BladeRF, USRP, PlutoSDR, LimeSDR**
- Cellular (2G/3G/4G/5G)
- WiFi attacks
- GPS spoofing
- Drone jamming
- Spectrum analysis

### ⚠️ Half-Duplex (TX or RX)
**HackRF One**
- WiFi attacks
- GPS spoofing
- Spectrum analysis
- Cellular (TX or RX only)

### 🔍 RX-Only (Passive Monitoring)
**RTL-SDR**
- Spectrum monitoring
- SIGINT (passive)
- WiFi scanning
- Drone detection

---

## Budget Recommendations

### $25 Budget: RTL-SDR
- ✅ Spectrum monitoring
- ✅ Learn SDR basics
- ❌ No transmission

### $150-200 Budget: LimeSDR or PlutoSDR
- ✅ Full-duplex TX/RX
- ✅ WiFi attacks
- ✅ Cellular 2G/3G/4G

### $300 Budget: HackRF One
- ✅ Wide frequency range (1 MHz - 6 GHz)
- ✅ WiFi, GPS, cellular
- ⚠️ Half-duplex only

### $450+ Budget: BladeRF 2.0
- ✅ Full system capabilities
- ✅ Professional 2G/3G/4G/5G
- ✅ 2x2 MIMO

---

## Hardware Abstraction Layer

The HAL automatically:
- Detects all available SDRs
- Selects best SDR for each operation
- Provides unified TX/RX interface
- Handles fallbacks gracefully

```python
# Unified interface for any SDR
hal.unified_configure(
    frequency=2_450_000_000,
    sample_rate=20_000_000,
    bandwidth=20_000_000,
    tx_gain=40,
    rx_gain=40
)

# Transmit (works with any TX-capable SDR)
hal.unified_transmit(samples)

# Receive (works with any SDR)
samples = hal.unified_receive(num_samples)
```

---

## Troubleshooting

### SDR Not Detected
1. Check USB connection: `lsusb`
2. Check permissions: `sudo usermod -a -G plugdev $USER`
3. Install drivers (see installation above)
4. Logout/login for permissions

### Permission Denied
```bash
sudo usermod -a -G plugdev $USER
sudo usermod -a -G dialout $USER
# Logout/login required
```

### HackRF "Device Busy"
```bash
sudo killall hackrf_transfer
```

---

## Performance Tips

### HackRF
- Use ≤20 MSPS sample rate
- Enable RF amplifier (+14 dB)
- Half-duplex: switch between TX/RX

### LimeSDR
- USB 3.0 required for >10 MSPS
- BAND1: <1.5 GHz, BAND2: >1.5 GHz
- Calibrate: `LimeQuickTest`

### RTL-SDR
- Use RTL-SDR Blog V3
- Enable bias-tee for active antennas
- High-quality USB cables

---

## Module Compatibility Matrix

| Operation | BladeRF | HackRF | LimeSDR | PlutoSDR | USRP | RTL-SDR |
|-----------|---------|--------|---------|----------|------|---------|
| 2G/3G/4G | ✅ Full | ⚠️ Half | ✅ Full | ✅ Full | ✅ Full | 🔍 Passive |
| 5G | ✅ Full | ⚠️ Half | ❌ | ❌ | ✅ Full | ❌ |
| WiFi | ✅ | ✅ | ✅ | ✅ | ✅ | 🔍 Scan |
| GPS Spoofing | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| Jamming | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Spectrum Analysis | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

Legend:
- ✅ Fully supported
- ⚠️ Half-duplex limitations
- 🔍 RX-only (passive)
- ❌ Not supported

---

**Version**: v1.0.7  
**Status**: PRODUCTION READY  
**Multi-SDR Support**: 6 platforms

For detailed installation: `install/install_sdr_support.sh`
