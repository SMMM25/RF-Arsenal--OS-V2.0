# RF Arsenal OS - Installation Guide

**Version**: 1.0.0  
**Last Updated**: 2024-12-20  
**Status**: Production Ready

---

## 📋 QUICK START

RF Arsenal OS supports **3 installation methods**:

1. **Pre-built USB Image** (Easiest - Flash & Boot) - *Recommended for beginners*
2. **Quick Install Script** (One Command) - *For existing Raspberry Pi OS*
3. **Manual Installation** (Full Control) - *For advanced users*

---

## 🔧 HARDWARE REQUIREMENTS

### Minimum Requirements
- **Raspberry Pi**: 3 B+ or newer
- **RAM**: 2GB minimum (4GB+ recommended)
- **Storage**: 16GB microSD or USB drive
- **SDR**: BladeRF x40, xA9, or 2.0 micro
- **Power**: Official Raspberry Pi power adapter

### Hardware Compatibility

| Raspberry Pi | Performance | USB Speed | Status |
|--------------|-------------|-----------|--------|
| **Pi 5 (8GB)** | ⭐⭐⭐⭐⭐ Optimal | USB 3.0 (5 Gbps) | ✅ Fully Optimized |
| **Pi 4 (4GB+)** | ⭐⭐⭐⭐ Good | USB 3.0 (5 Gbps) | ✅ Full Support |
| **Pi 3 B+** | ⭐⭐⭐ Minimum | USB 2.0 (480 Mbps) | ✅ Basic Features |

---

## 🚀 METHOD 1: PRE-BUILT USB IMAGE (Recommended)

**Time**: 15-20 minutes  
**Difficulty**: ⭐ Easy

### Step 1: Download Image
```bash
# Visit releases page
https://github.com/SMMM25/RF-Arsenal-OS/releases

# Download latest image
RF-Arsenal-OS-v1.0-RaspberryPi5.img.xz
Step 2: Flash to USB/SD Card
Using balenaEtcher (Easiest):

Download Etcher: https://etcher.balena.io/
Select downloaded .img.xz file
Select your USB/SD drive (16GB+)
Click "Flash!"
Wait for completion (~10-15 minutes)
Using dd (Linux/Mac):

Copy# Extract and flash
xz -d RF-Arsenal-OS-v1.0-RaspberryPi5.img.xz
sudo dd if=RF-Arsenal-OS-v1.0-RaspberryPi5.img of=/dev/sdX bs=4M status=progress
sync
Step 3: Boot Raspberry Pi
Insert flashed SD card
Connect BladeRF SDR (if available)
Connect network (Ethernet recommended)
Connect monitor, keyboard, mouse
Power on
The First Boot Setup Wizard will launch automatically!

🔧 METHOD 2: QUICK INSTALL SCRIPT
Time: 30-45 minutes
Difficulty: ⭐⭐ Easy

For existing Raspberry Pi OS installations:

Copy# Clone repository
git clone https://github.com/SMMM25/RF-Arsenal-OS.git
cd RF-Arsenal-OS

# Run quick installer
sudo bash install/quick_install.sh

# Reboot
sudo reboot

# Launch
python3 rf_arsenal_os.py
What it installs:

✅ System dependencies (BladeRF, Tor, Python)
✅ Python packages from requirements.txt
✅ RF Arsenal OS from GitHub
✅ Hardware optimizations
✅ Desktop shortcuts
🛠️ METHOD 3: MANUAL INSTALLATION
Time: 45-60 minutes
Difficulty: ⭐⭐⭐ Advanced

Step 1: Install System Dependencies
Copy# Update system
sudo apt update && sudo apt upgrade -y

# Install build tools
sudo apt install -y build-essential git cmake python3 python3-pip python3-dev

# Install BladeRF
sudo apt install -y libbladerf-dev bladerf

# Install anonymity tools (optional)
sudo apt install -y tor i2p

# Install Bluetooth (for mesh networking)
sudo apt install -y bluez bluez-tools libbluetooth-dev
Step 2: Clone RF Arsenal OS
Copygit clone https://github.com/SMMM25/RF-Arsenal-OS.git
cd RF-Arsenal-OS
Step 3: Install Python Dependencies
Copypip3 install -r install/requirements.txt
Or manually:

Copypip3 install numpy scipy PyQt6 pyqtgraph requests PySocks cryptography PyYAML psutil scapy skyfield RPi.GPIO
Step 4: Hardware Optimization
Copy# Run hardware detection and optimization
sudo python3 install/pi_detect.py
This automatically:

Detects Raspberry Pi model (5/4/3)
Applies model-specific optimizations
Configures CPU governor
Sets GPU memory allocation
Optimizes USB 3.0 (Pi 5/4)
Step 5: Verify Installation
Copy# Run system check
python3 rf_arsenal_os.py --check
Expected output:

✅ Checking dependencies... (all present)
🔌 Checking hardware... (BladeRF detected)
🏥 System Health Check (all optimal)
🎛️ FIRST BOOT SETUP WIZARD
The First Boot Setup Wizard runs automatically (pre-built image) or manually:

Copysudo python3 install/first_boot_wizard.py
Wizard Steps:
1. Welcome Screen
Displays welcome banner
Press ENTER to continue
2. Hardware Detection
Detects Raspberry Pi model
Scans for BladeRF SDR
Checks Bluetooth adapter
Detects GPIO availability
3. Network Mode Selection
Choose operating mode:

Online Mode - Full features (Tor, VPN, updates)
Offline Mode - Air-gapped security (no network)
Hybrid Mode - Selective connectivity (recommended)
4. Security Configuration
Enable/disable features:

Encryption for stored data
Stealth mode (RF emission masking)
Anti-forensics features
Mesh networking (offline P2P)
5. User Preferences
Preferred interface (GUI/CLI/Both)
Auto-start on boot
Verbose logging
6. Finalization
Displays configuration summary
Applies settings
Reboots system
✅ VERIFICATION & TESTING
System Health Check
Copypython3 rf_arsenal_os.py --check
BladeRF Connection Test
Copy# Check if detected
bladeRF-cli -p

# Test basic functionality
bladeRF-cli -i
bladeRF> info
bladeRF> quit
Launch RF Arsenal OS
Copy# GUI mode (default)
python3 rf_arsenal_os.py

# CLI mode
python3 rf_arsenal_os.py --cli

# With sudo for hardware access
sudo python3 rf_arsenal_os.py
🔧 TROUBLESHOOTING
Issue: BladeRF Not Detected
Solutions:

Copy# Check USB connection
lsusb | grep Nuand

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Check permissions
sudo usermod -aG plugdev $USER
# Log out and back in
Issue: Import Errors (Missing Dependencies)
Solutions:

Copy# Reinstall dependencies
pip3 install -r install/requirements.txt

# Check Python version (must be 3.7+)
python3 --version
Issue: GUI Won't Launch
Solutions:

Copy# Install PyQt6
pip3 install PyQt6

# Or use CLI mode
python3 rf_arsenal_os.py --cli
Issue: Permission Denied
Solutions:

Copy# Run with sudo
sudo python3 rf_arsenal_os.py

# Add user to groups
sudo usermod -aG dialout,plugdev,gpio $USER
# Log out and back in
📚 NEXT STEPS
After successful installation:

Read Documentation:

docs/UPDATE_GUIDE.md
docs/FISSURE_INTEGRATION.md
Module-specific READMEs
Test Basic Functions:

Spectrum analyzer
WiFi scanning (authorized networks only)
System health checks
Configure Security:

Review stealth settings
Configure mesh networking
Set up VPN/Tor (if online mode)
Join Community:

GitHub Issues: Report bugs
GitHub Discussions: Ask questions
Stay Updated:

Copy# Check for updates
python3 update_manager.py --check

# Install updates
python3 update_manager.py --install
⚖️ LEGAL NOTICE
FOR AUTHORIZED PENETRATION TESTING ONLY

⚠️ This software requires:

Proper authorization for RF transmission
FCC/regulatory compliance
Licensed operator for amateur radio
Written permission for security testing
❌ Illegal to:

Interfere with licensed services
Unauthorized cellular/GPS spoofing (federal crime)
Intercept communications without warrant
Malicious activities
📞 SUPPORT
Repository: https://github.com/SMMM25/RF-Arsenal-OS
Issues: https://github.com/SMMM25/RF-Arsenal-OS/issues
Discussions: https://github.com/SMMM25/RF-Arsenal-OS/discussions
Releases: https://github.com/SMMM25/RF-Arsenal-OS/releases
Version: 1.0.0
Last Updated: 2024-12-20
Status: Production Ready
License: MIT

🎉 Installation complete! Welcome to RF Arsenal OS! 🎉
