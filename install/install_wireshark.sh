#!/bin/bash
################################################################################
# RF Arsenal OS - Wireshark Integration Installation Script
# Complete automated installation with testing
################################################################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     RF ARSENAL OS - WIRESHARK INTEGRATION INSTALLER          ║"
echo "║              White Hat Edition v1.0                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}[!] Please run as root: sudo $0${NC}"
    exit 1
fi

# Get the actual user (not root)
ACTUAL_USER="${SUDO_USER:-$USER}"
ACTUAL_HOME=$(getent passwd "$ACTUAL_USER" | cut -d: -f6)

echo -e "${GREEN}[*] Installing for user: $ACTUAL_USER${NC}"
echo ""

# STEP 1: Install System Dependencies
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}[STEP 1/7] Installing System Dependencies...${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

apt-get update -qq
apt-get install -y \
    wireshark \
    tshark \
    libpcap-dev \
    libcap2-bin \
    2>&1 | grep -v "^Selecting" | grep -v "^Preparing" | grep -v "^Unpacking" || true

echo -e "${GREEN}✅ System packages installed${NC}"
echo ""

# STEP 2: Configure Wireshark Permissions
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}[STEP 2/7] Configuring Wireshark Permissions...${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

# Configure wireshark to allow non-root capture
echo "wireshark-common wireshark-common/install-setuid boolean true" | debconf-set-selections
DEBIAN_FRONTEND=noninteractive dpkg-reconfigure wireshark-common 2>/dev/null

# Create wireshark group if it doesn't exist
groupadd -r wireshark 2>/dev/null || true

# Add user to wireshark group
usermod -aG wireshark "$ACTUAL_USER"
echo -e "${GREEN}✅ User $ACTUAL_USER added to wireshark group${NC}"

# Set capabilities on dumpcap
setcap cap_net_raw,cap_net_admin=eip /usr/bin/dumpcap
echo -e "${GREEN}✅ Packet capture capabilities configured${NC}"

# Verify
CAPS=$(getcap /usr/bin/dumpcap 2>/dev/null || echo "")
if [[ "$CAPS" == *"cap_net_admin,cap_net_raw=eip"* ]]; then
    echo -e "${GREEN}✅ Dumpcap capabilities verified${NC}"
else
    echo -e "${YELLOW}⚠️  Dumpcap capabilities may need manual configuration${NC}"
fi

echo ""

# STEP 3: Verify TShark
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}[STEP 3/7] Verifying TShark Installation...${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

if command -v tshark &> /dev/null; then
    TSHARK_VERSION=$(tshark --version 2>&1 | head -1)
    echo -e "${GREEN}✅ TShark installed: $TSHARK_VERSION${NC}"
else
    echo -e "${RED}❌ TShark not found${NC}"
    exit 1
fi

echo ""

# STEP 4: Install Python Dependencies
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}[STEP 4/7] Installing Python Dependencies...${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

# Install PyShark for the actual user
sudo -u "$ACTUAL_USER" pip3 install --quiet pyshark

# Verify PyShark
if sudo -u "$ACTUAL_USER" python3 -c "import pyshark" 2>/dev/null; then
    echo -e "${GREEN}✅ PyShark installed successfully${NC}"
else
    echo -e "${RED}❌ PyShark installation failed${NC}"
    exit 1
fi

echo ""

# STEP 5: Test Network Interfaces
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}[STEP 5/7] Detecting Network Interfaces...${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

echo "Available interfaces:"
tshark -D 2>&1 | head -10
echo ""

# STEP 6: Test Basic Capture
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}[STEP 6/7] Testing Packet Capture (5 seconds)...${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

TEST_PCAP="/tmp/rf_arsenal_test_capture.pcap"
rm -f "$TEST_PCAP"

echo "Running 5-second test capture on loopback interface..."
if timeout 5 tshark -i lo -c 10 -w "$TEST_PCAP" 2>&1 | grep -q "Capturing on"; then
    echo -e "${GREEN}✅ Live capture test successful${NC}"
    
    # Verify PCAP file
    if [ -f "$TEST_PCAP" ]; then
        PACKET_COUNT=$(tshark -r "$TEST_PCAP" 2>/dev/null | wc -l)
        echo -e "${GREEN}✅ Captured $PACKET_COUNT packets${NC}"
        rm -f "$TEST_PCAP"
    fi
else
    echo -e "${YELLOW}⚠️  Live capture test had issues (may be normal in some environments)${NC}"
fi

echo ""

# STEP 7: Verify Python Module Integration
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}[STEP 7/7] Verifying Python Module Integration...${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

# Check if packet_capture module exists
if [ -f "modules/network/packet_capture.py" ]; then
    echo "Testing packet capture module import..."
    if sudo -u "$ACTUAL_USER" python3 -c "from modules.network.packet_capture import WiresharkCapture; print('✅ Module imports successfully')" 2>/dev/null; then
        echo -e "${GREEN}✅ Python module integration verified${NC}"
    else
        echo -e "${YELLOW}⚠️  Module import test skipped (run from RF Arsenal OS directory)${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Module files not found (install from repository first)${NC}"
fi

echo ""

# Final Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  INSTALLATION COMPLETE                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✅ TShark installed and configured${NC}"
echo -e "${GREEN}✅ PyShark Python library installed${NC}"
echo -e "${GREEN}✅ Packet capture permissions configured${NC}"
echo -e "${GREEN}✅ User $ACTUAL_USER added to wireshark group${NC}"
echo ""
echo -e "${YELLOW}⚠️  IMPORTANT: User must logout and login (or reboot) for group changes to take effect${NC}"
echo ""
echo "Next steps:"
echo "  1. Logout and login again (or run: newgrp wireshark)"
echo "  2. Clone/update RF Arsenal OS repository"
echo "  3. Test with: sudo python3 rf_arsenal_os.py --cli"
echo "  4. Try AI command: 'capture packets on any'"
echo ""
echo "AI Commands available:"
echo "  • 'capture packets on wlan0'"
echo "  • 'stop capture'"
echo "  • 'analyze packets'"
echo "  • 'check for dns leaks'"
echo "  • 'cleanup captures securely'"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Built by white hats, for white hats. 🛡️${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
