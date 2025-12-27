#!/bin/bash
#
# RF Arsenal OS - Production Installation Script
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════╗"
echo "║     RF ARSENAL OS - PRODUCTION INSTALLER          ║"
echo "║              White Hat Edition v1.0                ║"
echo "╚════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Error: Must run as root${NC}"
    exit 1
fi

# Update system
echo -e "${GREEN}[*] Updating system...${NC}"
apt-get update
apt-get upgrade -y

# Install base dependencies
echo -e "${GREEN}[*] Installing base dependencies...${NC}"
apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-numpy \
    python3-scipy \
    libusb-1.0-0-dev \
    pkg-config \
    libtecla-dev \
    libncurses5-dev \
    wget \
    curl

# Install BladeRF library
echo -e "${GREEN}[*] Installing BladeRF library...${NC}"
cd /tmp
if [ -d "bladeRF" ]; then
    rm -rf bladeRF
fi
git clone https://github.com/Nuand/bladeRF.git
cd bladeRF/host
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DINSTALL_UDEV_RULES=ON \
      ..
make -j$(nproc)
make install
ldconfig

# Install Python dependencies
echo -e "${GREEN}[*] Installing Python dependencies...${NC}"
pip3 install --upgrade pip
pip3 install -r /home/user/webapp/install/requirements.txt

# Install security tools
echo -e "${GREEN}[*] Installing security tools...${NC}"
apt-get install -y \
    tor \
    macchanger \
    secure-delete \
    cryptsetup \
    ufw \
    fail2ban

# Configure Tor
echo -e "${GREEN}[*] Configuring Tor...${NC}"
cat > /etc/tor/torrc <<TOREOF
SocksPort 9050
ControlPort 9051
CookieAuthentication 1
TOREOF
systemctl enable tor
systemctl start tor

# Configure firewall
echo -e "${GREEN}[*] Configuring firewall...${NC}"
ufw default deny incoming
ufw default allow outgoing
ufw --force enable

# Disable swap
echo -e "${GREEN}[*] Disabling swap...${NC}"
swapoff -a
sed -i '/swap/d' /etc/fstab

echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════╗"
echo "║           INSTALLATION COMPLETE                    ║"
echo "╚════════════════════════════════════════════════════╝"
echo -e "${NC}"
