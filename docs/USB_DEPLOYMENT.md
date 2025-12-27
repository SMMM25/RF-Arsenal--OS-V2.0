# RF Arsenal OS - USB Deployment Guide

## Overview

RF Arsenal OS is designed for **mobile, portable operation** on Raspberry Pi with RasPad touchscreen setups. This guide covers creating bootable USB drives for easy deployment.

## Two Deployment Options

| Option | Use Case | Leaves Traces | Internet Required |
|--------|----------|---------------|-------------------|
| **USB Installer** | Permanent install to Pi | Yes (on SD/USB) | Yes (first boot) |
| **Portable USB** | Live boot, no install | No (RAM-only) | No (pre-configured) |

---

## Option 1: USB Installer (Permanent Installation)

Creates a USB that **installs** RF Arsenal OS onto your Pi's storage.

### Create the Installer

```bash
# On any Linux machine with USB drive inserted
cd /path/to/RF-Arsenal-OS
sudo bash install/create_usb_installer.sh /dev/sdX
```

### Installation Process

1. Insert USB into Raspberry Pi
2. Power on - boots from USB automatically (Pi 4/5)
3. First boot setup runs automatically (~10 minutes)
4. System reboots when complete
5. Login: `pi` / `raspberry` (change immediately!)
6. Run: `rf-arsenal`

### What Gets Installed

- Full RF Arsenal OS to `/opt/rf-arsenal-os`
- All dependencies (BladeRF, Python packages)
- SDR udev rules
- System services
- Desktop shortcut (if GUI available)

---

## Option 2: Portable USB (Live Boot - Recommended for Field Ops)

Creates a USB that **runs RF Arsenal OS directly** without installing anything.

### Create Portable USB

```bash
cd /path/to/RF-Arsenal-OS
sudo bash install/create_portable_usb.sh /dev/sdX
```

### Features

- **Zero installation** - runs entirely from USB
- **RAM-only logging** - no forensic traces on disk
- **Stealth boot** - minimal boot messages
- **Works on ANY Pi** - just plug in and boot
- **Pre-configured** - ready to use immediately

### Stealth Features Enabled by Default

| Feature | Description |
|---------|-------------|
| RAM disk | All logs written to RAM, not disk |
| No swap | Memory never written to storage |
| Quiet boot | Minimal console output |
| Auto-wipe | RAM cleared on shutdown |

### Boot Instructions

**Raspberry Pi 5/4:**
1. Insert USB
2. Power on - boots automatically from USB

**Raspberry Pi 3:**
1. First, enable USB boot: `sudo raspi-config` → Boot Options → USB Boot
2. Then insert USB and power on

---

## Hardware Requirements

### Minimum
- Raspberry Pi 3 B+ (2GB RAM)
- 16GB USB drive (USB 3.0 recommended)
- BladeRF x40 or compatible SDR

### Recommended
- Raspberry Pi 5 (8GB RAM)
- 32GB+ USB 3.0 drive
- BladeRF 2.0 micro xA9
- RasPad 3 touchscreen

### Supported SDR Devices
- BladeRF (x40, x115, 2.0 micro xA4/xA9)
- HackRF One / PortaPack
- RTL-SDR v3/v4
- LimeSDR USB/Mini
- PlutoSDR
- USRP B200/B210
- Airspy R2/Mini/HF+

---

## RasPad 3 Touchscreen Setup

After booting, run:

```bash
sudo /opt/rf-arsenal-os/install/setup_raspad.sh
sudo reboot
```

This enables:
- 1024x600 display resolution
- Touch calibration
- Proper orientation

---

## Operational Profiles

### Standard Mode
```bash
rf-arsenal
```
Normal operation with standard logging.

### GUI Mode
```bash
rf-arsenal --gui
```
Launches PyQt5 graphical interface.

### Stealth Mode
```bash
rf-arsenal --stealth
```
Activates:
- WiFi/Bluetooth disabled
- MAC randomization
- Logging to RAM only
- Command history disabled
- Minimal RF emissions

### Power Profiles

```bash
# Low power (battery saving)
sudo /opt/rf-arsenal-os/profiles/low_power.sh

# High performance (demanding operations)
sudo /opt/rf-arsenal-os/profiles/high_performance.sh

# Stealth (minimal emissions)
sudo /opt/rf-arsenal-os/profiles/stealth.sh
```

---

## Quick Reference

### Creating USB Drives

```bash
# Installer USB (permanent install)
sudo bash install/create_usb_installer.sh /dev/sdX

# Portable USB (live boot)
sudo bash install/create_portable_usb.sh /dev/sdX
```

### First Commands After Boot

```bash
# Change default password!
passwd

# Check SDR connection
bladeRF-cli -p

# Launch RF Arsenal
rf-arsenal

# Launch GUI
rf-arsenal --gui
```

### File Locations

| Path | Description |
|------|-------------|
| `/opt/rf-arsenal-os` | Main installation |
| `/tmp/rf_arsenal_ram` | RAM disk for sensitive ops |
| `/var/log/rf-arsenal` | Log files (RAM on portable) |
| `/etc/rf-arsenal` | Configuration files |

---

## Troubleshooting

### USB Won't Boot

**Pi 4/5:** Check USB port (use USB 3.0 blue ports)

**Pi 3:** Enable USB boot first:
```bash
echo program_usb_boot_mode=1 | sudo tee -a /boot/config.txt
sudo reboot
```

### SDR Not Detected

```bash
# Check USB connection
lsusb | grep -i blade

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Check permissions
groups  # Should include 'bladerf' or 'plugdev'
```

### Out of RAM (Portable Mode)

Portable mode uses RAM disk. If running low:
```bash
# Check RAM usage
free -h

# Clear RAM disk cache
rm -rf /tmp/rf_arsenal_ram/cache/*
```

---

## Security Notes

### Portable USB Security

- **No traces left** on the Pi itself
- All data stored on USB is encrypted (optional)
- RAM cleared on power off
- USB can be quickly removed in emergency

### Physical Security

- Use encrypted USB drives for sensitive operations
- Enable the deadman switch for automatic wipe
- Configure geofence for operational area

### Operational Security

```bash
# Before field operation
rf-arsenal --stealth

# After operation - wipe RAM
sudo /opt/rf-arsenal-os/security/emergency_wipe.sh
```

---

## Support

- GitHub: https://github.com/SMMM25/RF-Arsenal-OS
- Documentation: `/opt/rf-arsenal-os/docs/`

---

*RF Arsenal OS - Mobile RF Security Platform*
*For authorized security testing only*
