#!/bin/bash
# RF Arsenal OS - Multi-SDR Support Installer
# Installs drivers and libraries for multiple SDR platforms

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "================================================"
echo "  RF Arsenal OS - Multi-SDR Support Installer"
echo "================================================"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
   echo -e "${RED}[-] Please do not run as root${NC}"
   echo "    Run as normal user, will use sudo when needed"
   exit 1
fi

echo "This installer will help you set up support for multiple SDR platforms."
echo ""
echo "Supported SDRs:"
echo "  1. BladeRF 2.0 - Professional full-duplex"
echo "  2. HackRF One - Budget all-around"
echo "  3. LimeSDR Mini - Budget full-duplex"
echo "  4. PlutoSDR - Portable"
echo "  5. USRP B200/B210 - High-end"
echo "  6. RTL-SDR - RX-only monitoring"
echo ""

# Detect connected SDRs
echo "[+] Detecting connected SDRs..."
DETECTED_BLADERF=$(lsusb | grep -i "2cf0:5250" || true)
DETECTED_HACKRF=$(lsusb | grep -i "1d50:6089" || true)
DETECTED_LIMESDR=$(lsusb | grep -i "0403:601f\|1d50:6108" || true)
DETECTED_PLUTO=$(lsusb | grep -i "0456:b673" || true)
DETECTED_USRP=$(lsusb | grep -i "2500:0020\|2500:0021" || true)
DETECTED_RTLSDR=$(lsusb | grep -i "0bda:2838" || true)

DETECTED_COUNT=0
if [ -n "$DETECTED_BLADERF" ]; then
    echo -e "  ${GREEN}✓${NC} BladeRF detected"
    DETECTED_COUNT=$((DETECTED_COUNT + 1))
fi
if [ -n "$DETECTED_HACKRF" ]; then
    echo -e "  ${GREEN}✓${NC} HackRF detected"
    DETECTED_COUNT=$((DETECTED_COUNT + 1))
fi
if [ -n "$DETECTED_LIMESDR" ]; then
    echo -e "  ${GREEN}✓${NC} LimeSDR detected"
    DETECTED_COUNT=$((DETECTED_COUNT + 1))
fi
if [ -n "$DETECTED_PLUTO" ]; then
    echo -e "  ${GREEN}✓${NC} PlutoSDR detected"
    DETECTED_COUNT=$((DETECTED_COUNT + 1))
fi
if [ -n "$DETECTED_USRP" ]; then
    echo -e "  ${GREEN}✓${NC} USRP detected"
    DETECTED_COUNT=$((DETECTED_COUNT + 1))
fi
if [ -n "$DETECTED_RTLSDR" ]; then
    echo -e "  ${GREEN}✓${NC} RTL-SDR detected"
    DETECTED_COUNT=$((DETECTED_COUNT + 1))
fi

if [ $DETECTED_COUNT -eq 0 ]; then
    echo -e "  ${YELLOW}⚠${NC} No SDRs detected (may not be connected yet)"
fi

echo ""
echo "Select which SDR support to install:"
echo ""

# Ask for each SDR
read -p "Install BladeRF support? (y/n) [default: n]: " INSTALL_BLADERF
INSTALL_BLADERF=${INSTALL_BLADERF:-n}

read -p "Install HackRF support? (y/n) [default: n]: " INSTALL_HACKRF
INSTALL_HACKRF=${INSTALL_HACKRF:-n}

read -p "Install LimeSDR support? (y/n) [default: n]: " INSTALL_LIMESDR
INSTALL_LIMESDR=${INSTALL_LIMESDR:-n}

read -p "Install PlutoSDR support? (y/n) [default: n]: " INSTALL_PLUTO
INSTALL_PLUTO=${INSTALL_PLUTO:-n}

read -p "Install USRP support? (y/n) [default: n]: " INSTALL_USRP
INSTALL_USRP=${INSTALL_USRP:-n}

read -p "Install RTL-SDR support? (y/n) [default: n]: " INSTALL_RTLSDR
INSTALL_RTLSDR=${INSTALL_RTLSDR:-n}

echo ""

# Update package lists
echo "[+] Updating package lists..."
sudo apt-get update -qq

# Install BladeRF
if [ "${INSTALL_BLADERF,,}" = "y" ]; then
    echo ""
    echo "[+] Installing BladeRF support..."
    
    # Add PPA
    sudo add-apt-repository -y ppa:bladerf/bladerf > /dev/null 2>&1
    sudo apt-get update -qq
    
    # Install packages
    sudo apt-get install -y bladerf libbladerf-dev bladerf-firmware-fx3 bladerf-fpga-hostedx115
    
    # Install Python bindings
    pip3 install --user bladerf
    
    echo -e "${GREEN}✓${NC} BladeRF support installed"
fi

# Install HackRF
if [ "${INSTALL_HACKRF,,}" = "y" ]; then
    echo ""
    echo "[+] Installing HackRF support..."
    
    # Install packages
    sudo apt-get install -y hackrf libhackrf-dev libhackrf0
    
    # Add user to plugdev group
    sudo usermod -a -G plugdev $USER
    
    echo -e "${GREEN}✓${NC} HackRF support installed"
    echo -e "${YELLOW}⚠${NC} You must logout/login for permission changes to take effect"
fi

# Install LimeSDR
if [ "${INSTALL_LIMESDR,,}" = "y" ]; then
    echo ""
    echo "[+] Installing LimeSDR support..."
    
    # Add PPA
    sudo add-apt-repository -y ppa:myriadrf/drivers > /dev/null 2>&1
    sudo apt-get update -qq
    
    # Install packages
    sudo apt-get install -y limesuite liblimesuite-dev limesuite-udev
    sudo apt-get install -y soapysdr-tools soapysdr-module-lms7
    
    # Install Python bindings
    pip3 install --user SoapySDR
    
    # Add user to plugdev group
    sudo usermod -a -G plugdev $USER
    
    echo -e "${GREEN}✓${NC} LimeSDR support installed"
    echo -e "${YELLOW}⚠${NC} You must logout/login for permission changes to take effect"
fi

# Install PlutoSDR
if [ "${INSTALL_PLUTO,,}" = "y" ]; then
    echo ""
    echo "[+] Installing PlutoSDR support..."
    
    # Install packages
    sudo apt-get install -y libiio-utils libiio-dev python3-iio
    
    # Install Python bindings
    pip3 install --user pyadi-iio
    
    echo -e "${GREEN}✓${NC} PlutoSDR support installed"
fi

# Install USRP
if [ "${INSTALL_USRP,,}" = "y" ]; then
    echo ""
    echo "[+] Installing USRP support..."
    
    # Add PPA
    sudo add-apt-repository -y ppa:ettusresearch/uhd > /dev/null 2>&1
    sudo apt-get update -qq
    
    # Install packages
    sudo apt-get install -y libuhd-dev uhd-host
    
    # Install Python bindings
    pip3 install --user uhd
    
    # Download firmware images (may take a while)
    echo "  [+] Downloading USRP firmware images (this may take a few minutes)..."
    sudo uhd_images_downloader > /dev/null 2>&1
    
    echo -e "${GREEN}✓${NC} USRP support installed"
fi

# Install RTL-SDR
if [ "${INSTALL_RTLSDR,,}" = "y" ]; then
    echo ""
    echo "[+] Installing RTL-SDR support..."
    
    # Install packages
    sudo apt-get install -y rtl-sdr librtlsdr-dev librtlsdr0
    
    # Install Python bindings
    pip3 install --user pyrtlsdr
    
    # Blacklist kernel drivers
    echo "  [+] Blacklisting DVB-T kernel drivers..."
    echo 'blacklist dvb_usb_rtl28xxu' | sudo tee /etc/modprobe.d/blacklist-rtl.conf > /dev/null
    
    # Unload if currently loaded
    sudo rmmod dvb_usb_rtl28xxu 2>/dev/null || true
    
    # Add user to plugdev group
    sudo usermod -a -G plugdev $USER
    
    echo -e "${GREEN}✓${NC} RTL-SDR support installed"
    echo -e "${YELLOW}⚠${NC} Kernel driver blacklisted - may need reboot for RTL-SDR to work"
fi

echo ""
echo "================================================"
echo "  Installation Complete!"
echo "================================================"
echo ""

# Test detection
echo "[+] Testing SDR detection..."
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/' + __import__('os').environ['USER'] + '/RF-Arsenal-OS')

try:
    from core.hardware_abstraction import SDRHardwareAbstraction
    
    hal = SDRHardwareAbstraction()
    detected = hal.auto_detect_sdr()
    
    if detected:
        print(f"\n✓ Detected {len(detected)} SDR device(s):")
        for sdr_type, device in detected:
            caps = hal.get_capabilities(sdr_type)
            if caps:
                print(f"  - {caps.model}")
    else:
        print("\n⚠ No SDRs detected (connect hardware and re-run)")
except Exception as e:
    print(f"\n⚠ Detection test failed: {e}")
    print("  (This is normal if SDRs are not connected yet)")
EOF

echo ""
echo "Next steps:"
echo "  1. Logout and login again (for permission changes)"
echo "  2. Connect your SDR hardware"
echo "  3. Test detection: sudo python3 rf_arsenal_os.py --detect-hardware"
echo "  4. Run RF Arsenal OS: sudo python3 rf_arsenal_os.py --cli"
echo ""
echo "For troubleshooting, see: docs/MULTI_SDR_SUPPORT.md"
echo ""
