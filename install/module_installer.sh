#!/bin/bash
#
# RF Arsenal OS - Module Installer
# Selectively install individual modules and dependencies
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Installation directory
INSTALL_DIR="/opt/rfarsenal"
if [ ! -d "$INSTALL_DIR" ]; then
    INSTALL_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
fi

# Banner
echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         RF ARSENAL OS - MODULE INSTALLER                      ║"
echo "║              Selective Component Installation                 ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Error: Must run as root${NC}"
    exit 1
fi

# Module installation functions

install_core() {
    echo -e "${GREEN}[*] Installing Core Module...${NC}"
    pip3 install numpy scipy
    echo -e "${GREEN}[✓] Core module ready${NC}"
}

install_bladerf() {
    echo -e "${GREEN}[*] Installing BladeRF Support...${NC}"
    
    apt-get install -y \
        build-essential \
        cmake \
        libusb-1.0-0-dev \
        pkg-config \
        libtecla-dev \
        libncurses5-dev
    
    # Build BladeRF
    cd /tmp
    if [ -d "bladeRF" ]; then
        rm -rf bladeRF
    fi
    git clone https://github.com/Nuand/bladeRF.git
    cd bladeRF/host
    mkdir -p build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DINSTALL_UDEV_RULES=ON \
          ..
    make -j$(nproc)
    make install
    ldconfig
    
    echo -e "${GREEN}[✓] BladeRF support installed${NC}"
}

install_cellular() {
    echo -e "${GREEN}[*] Installing Cellular Module...${NC}"
    
    # Core dependencies
    pip3 install numpy scipy
    
    # Optional: srsRAN dependencies
    apt-get install -y \
        libfftw3-dev \
        libmbedtls-dev \
        libsctp-dev \
        libconfig-dev \
        libzmq3-dev || true
    
    echo -e "${GREEN}[✓] Cellular module ready${NC}"
    echo -e "${YELLOW}Note: Full cellular requires srsRAN or OpenBTS${NC}"
}

install_wifi() {
    echo -e "${GREEN}[*] Installing WiFi Module...${NC}"
    
    apt-get install -y \
        aircrack-ng \
        mdk4 \
        hostapd \
        dnsmasq \
        macchanger || true
    
    pip3 install scapy
    
    echo -e "${GREEN}[✓] WiFi module ready${NC}"
}

install_gps() {
    echo -e "${GREEN}[*] Installing GPS Module...${NC}"
    
    pip3 install numpy scipy
    
    # Optional: GPS tools
    apt-get install -y \
        gpsd \
        gpsd-clients || true
    
    echo -e "${GREEN}[✓] GPS module ready${NC}"
}

install_drone() {
    echo -e "${GREEN}[*] Installing Drone Module...${NC}"
    
    pip3 install numpy scipy
    
    # Optional: MAVLink
    pip3 install pymavlink || true
    
    echo -e "${GREEN}[✓] Drone module ready${NC}"
}

install_jamming() {
    echo -e "${GREEN}[*] Installing Jamming Module...${NC}"
    
    pip3 install numpy scipy
    
    echo -e "${GREEN}[✓] Jamming module ready${NC}"
}

install_spectrum() {
    echo -e "${GREEN}[*] Installing Spectrum Module...${NC}"
    
    pip3 install numpy scipy matplotlib
    
    echo -e "${GREEN}[✓] Spectrum module ready${NC}"
}

install_sigint() {
    echo -e "${GREEN}[*] Installing SIGINT Module...${NC}"
    
    pip3 install numpy scipy
    
    echo -e "${GREEN}[✓] SIGINT module ready${NC}"
}

install_radar() {
    echo -e "${GREEN}[*] Installing Radar Module...${NC}"
    
    pip3 install numpy scipy
    
    echo -e "${GREEN}[✓] Radar module ready${NC}"
}

install_iot() {
    echo -e "${GREEN}[*] Installing IoT/RFID Module...${NC}"
    
    pip3 install numpy scipy
    
    # Optional: RFID tools
    apt-get install -y \
        libnfc-bin \
        mfoc \
        mfcuk || true
    
    echo -e "${GREEN}[✓] IoT/RFID module ready${NC}"
}

install_satellite() {
    echo -e "${GREEN}[*] Installing Satellite Module...${NC}"
    
    pip3 install numpy scipy
    
    # Optional: Satellite tracking
    pip3 install pyephem || true
    
    echo -e "${GREEN}[✓] Satellite module ready${NC}"
}

install_amateur() {
    echo -e "${GREEN}[*] Installing Amateur Radio Module...${NC}"
    
    pip3 install numpy scipy
    
    # Optional: Ham radio tools
    apt-get install -y \
        fldigi \
        wsjtx || true
    
    echo -e "${GREEN}[✓] Amateur radio module ready${NC}"
}

install_protocol() {
    echo -e "${GREEN}[*] Installing Protocol Analyzer Module...${NC}"
    
    pip3 install numpy scipy scapy
    
    echo -e "${GREEN}[✓] Protocol analyzer ready${NC}"
}

install_stealth() {
    echo -e "${GREEN}[*] Installing Stealth Module...${NC}"
    
    apt-get install -y \
        tor \
        macchanger \
        secure-delete \
        cryptsetup || true
    
    pip3 install stem pysocks requests
    
    # Configure Tor
    cat > /etc/tor/torrc <<TOREOF
SocksPort 9050
ControlPort 9051
CookieAuthentication 1
TOREOF
    
    systemctl enable tor || true
    systemctl start tor || true
    
    echo -e "${GREEN}[✓] Stealth module ready${NC}"
}

install_security() {
    echo -e "${GREEN}[*] Installing Security Module...${NC}"
    
    apt-get install -y \
        cryptsetup \
        secure-delete \
        steghide || true
    
    pip3 install cryptography pillow
    
    echo -e "${GREEN}[✓] Security module ready${NC}"
}

install_ai() {
    echo -e "${GREEN}[*] Installing AI Module...${NC}"
    
    # Core AI dependencies
    pip3 install numpy
    
    read -p "Install voice control (Whisper ~1GB)? (y/N): " voice
    if [ "$voice" = "y" ] || [ "$voice" = "Y" ]; then
        pip3 install openai-whisper sounddevice pyttsx3
        echo -e "${GREEN}[✓] Voice control installed${NC}"
    fi
    
    echo -e "${GREEN}[✓] AI module ready${NC}"
}

install_gui() {
    echo -e "${GREEN}[*] Installing GUI Module...${NC}"
    
    apt-get install -y \
        python3-pyqt5 \
        python3-pyqt5.qtchart || true
    
    pip3 install PyQt5 matplotlib
    
    echo -e "${GREEN}[✓] GUI module ready${NC}"
}

install_all() {
    echo -e "${GREEN}[*] Installing ALL Modules...${NC}"
    
    install_core
    install_bladerf
    install_cellular
    install_wifi
    install_gps
    install_drone
    install_jamming
    install_spectrum
    install_sigint
    install_radar
    install_iot
    install_satellite
    install_amateur
    install_protocol
    install_stealth
    install_security
    install_ai
    install_gui
    
    echo -e "${GREEN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║           ALL MODULES INSTALLED SUCCESSFULLY                  ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Menu
show_menu() {
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}                   SELECT MODULES TO INSTALL                   ${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "  CORE"
    echo "  ────"
    echo "  1)  Core Dependencies (numpy, scipy)"
    echo "  2)  BladeRF Hardware Support"
    echo ""
    echo "  RF MODULES"
    echo "  ──────────"
    echo "  3)  Cellular (2G/3G/4G/5G)"
    echo "  4)  WiFi Attack Suite"
    echo "  5)  GPS Spoofing"
    echo "  6)  Drone Warfare"
    echo "  7)  Jamming / EW"
    echo "  8)  Spectrum Analysis"
    echo "  9)  SIGINT Engine"
    echo "  10) Radar Systems"
    echo "  11) IoT / RFID"
    echo "  12) Satellite Communications"
    echo "  13) Amateur Radio"
    echo "  14) Protocol Analyzer"
    echo ""
    echo "  SECURITY & INTERFACE"
    echo "  ─────────────────────"
    echo "  15) Stealth / OPSEC"
    echo "  16) Security / Anti-Forensics"
    echo "  17) AI Control"
    echo "  18) GUI Interface"
    echo ""
    echo "  ─────────────────────"
    echo "  A)  Install ALL Modules"
    echo "  Q)  Quit"
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Main
main() {
    while true; do
        show_menu
        read -p "Select module(s) to install (comma-separated): " choices
        
        # Handle quit
        if [[ "$choices" =~ ^[Qq]$ ]]; then
            echo -e "${GREEN}[*] Exiting installer...${NC}"
            exit 0
        fi
        
        # Handle all
        if [[ "$choices" =~ ^[Aa]$ ]]; then
            install_all
            continue
        fi
        
        # Process comma-separated choices
        IFS=',' read -ra MODULES <<< "$choices"
        
        for module in "${MODULES[@]}"; do
            module=$(echo "$module" | tr -d ' ')
            
            case $module in
                1)  install_core ;;
                2)  install_bladerf ;;
                3)  install_cellular ;;
                4)  install_wifi ;;
                5)  install_gps ;;
                6)  install_drone ;;
                7)  install_jamming ;;
                8)  install_spectrum ;;
                9)  install_sigint ;;
                10) install_radar ;;
                11) install_iot ;;
                12) install_satellite ;;
                13) install_amateur ;;
                14) install_protocol ;;
                15) install_stealth ;;
                16) install_security ;;
                17) install_ai ;;
                18) install_gui ;;
                *)  echo -e "${RED}Invalid option: $module${NC}" ;;
            esac
        done
        
        echo ""
        read -p "Install more modules? (y/N): " more
        if [ "$more" != "y" ] && [ "$more" != "Y" ]; then
            break
        fi
    done
    
    echo -e "${GREEN}[✓] Module installation complete${NC}"
}

# Run
main "$@"
