#!/bin/bash
#
# RF Arsenal OS - System Launcher
# Main entry point for RF Arsenal OS
#

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
show_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                                                               ║"
    echo "║    ██████╗ ███████╗     █████╗ ██████╗ ███████╗███████╗███╗   ║"
    echo "║    ██╔══██╗██╔════╝    ██╔══██╗██╔══██╗██╔════╝██╔════╝████╗  ║"
    echo "║    ██████╔╝█████╗      ███████║██████╔╝███████╗█████╗  ██╔██╗ ║"
    echo "║    ██╔══██╗██╔══╝      ██╔══██║██╔══██╗╚════██║██╔══╝  ██║╚██╗║"
    echo "║    ██║  ██║██║         ██║  ██║██║  ██║███████║███████╗██║ ╚██║"
    echo "║    ╚═╝  ╚═╝╚═╝         ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═║"
    echo "║                                                               ║"
    echo "║              RF ARSENAL OS - WHITE HAT EDITION                ║"
    echo "║                   Professional RF Security                    ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then 
        echo -e "${YELLOW}[!] Warning: Not running as root. Some features may be limited.${NC}"
        return 1
    fi
    return 0
}

# Check BladeRF
check_bladerf() {
    if command -v bladeRF-cli &> /dev/null; then
        if bladeRF-cli -p &> /dev/null; then
            echo -e "${GREEN}[✓] BladeRF detected${NC}"
            return 0
        else
            echo -e "${YELLOW}[!] BladeRF not connected${NC}"
            return 1
        fi
    else
        echo -e "${RED}[✗] BladeRF tools not installed${NC}"
        return 1
    fi
}

# Check Python dependencies
check_python() {
    if python3 -c "import numpy, scipy" 2>/dev/null; then
        echo -e "${GREEN}[✓] Python dependencies OK${NC}"
        return 0
    else
        echo -e "${RED}[✗] Missing Python dependencies${NC}"
        return 1
    fi
}

# Show menu
show_menu() {
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}                       MAIN MENU                               ${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "  1)  Launch GUI Interface"
    echo "  2)  Launch Text AI Interface"
    echo "  3)  Launch Voice AI Interface"
    echo ""
    echo "  4)  Cellular Operations"
    echo "  5)  WiFi Operations"
    echo "  6)  GPS Operations"
    echo "  7)  Drone Operations"
    echo "  8)  Spectrum Analysis"
    echo "  9)  Electronic Warfare"
    echo "  10) SIGINT Operations"
    echo ""
    echo "  11) System Status"
    echo "  12) Hardware Check"
    echo "  13) Emergency Shutdown"
    echo ""
    echo "  0)  Exit"
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Launch GUI
launch_gui() {
    echo -e "${GREEN}[*] Launching GUI interface...${NC}"
    cd "$INSTALL_DIR"
    python3 ui/main_gui.py
}

# Launch Text AI
launch_text_ai() {
    echo -e "${GREEN}[*] Launching Text AI interface...${NC}"
    cd "$INSTALL_DIR"
    python3 modules/ai/text_ai.py
}

# Launch Voice AI
launch_voice_ai() {
    echo -e "${GREEN}[*] Launching Voice AI interface...${NC}"
    cd "$INSTALL_DIR"
    python3 modules/ai/voice_ai.py --simulate
}

# Cellular submenu
cellular_menu() {
    echo ""
    echo -e "${BLUE}═══════════ CELLULAR OPERATIONS ═══════════${NC}"
    echo ""
    echo "  1) Start 2G/GSM Base Station"
    echo "  2) Start 3G/UMTS Base Station"
    echo "  3) Start 4G/LTE Base Station"
    echo "  4) Start 5G/NR Base Station"
    echo "  5) IMSI Catcher Mode"
    echo "  0) Back"
    echo ""
    read -p "Select: " choice
    
    case $choice in
        1) python3 -c "from modules.cellular.gsm_2g import GSM2GBaseStation; print('2G module loaded')" ;;
        2) python3 -c "from modules.cellular.umts_3g import UMTSBaseStation; print('3G module loaded')" ;;
        3) python3 -c "from modules.cellular.lte_4g import LTEBaseStation; print('4G module loaded')" ;;
        4) python3 -c "from modules.cellular.nr_5g import NRBaseStation; print('5G module loaded')" ;;
        5) echo "IMSI Catcher requires GUI or AI interface" ;;
        0) return ;;
    esac
}

# WiFi submenu
wifi_menu() {
    echo ""
    echo -e "${BLUE}═══════════ WIFI OPERATIONS ═══════════${NC}"
    echo ""
    echo "  1) Scan Networks"
    echo "  2) Deauthentication Attack"
    echo "  3) Evil Twin Attack"
    echo "  4) WPS Attack"
    echo "  0) Back"
    echo ""
    read -p "Select: " choice
    
    case $choice in
        1) python3 -c "from modules.wifi.wifi_attacks import WiFiAttackSuite; print('WiFi scanner loaded')" ;;
        2) echo "Use GUI or AI interface for attacks" ;;
        3) echo "Use GUI or AI interface for attacks" ;;
        4) echo "Use GUI or AI interface for attacks" ;;
        0) return ;;
    esac
}

# GPS submenu
gps_menu() {
    echo ""
    echo -e "${BLUE}═══════════ GPS OPERATIONS ═══════════${NC}"
    echo ""
    echo "  1) Spoof GPS Location"
    echo "  2) GPS Path Simulation"
    echo "  3) GPS Jamming"
    echo "  0) Back"
    echo ""
    read -p "Select: " choice
    
    case $choice in
        1|2|3) echo "Use GUI or AI interface for GPS operations" ;;
        0) return ;;
    esac
}

# Drone submenu
drone_menu() {
    echo ""
    echo -e "${BLUE}═══════════ DRONE OPERATIONS ═══════════${NC}"
    echo ""
    echo "  1) Detect Drones"
    echo "  2) Jam Drone Frequencies"
    echo "  3) GPS Spoof Drone"
    echo "  4) Hijack Control"
    echo "  5) Force Landing"
    echo "  0) Back"
    echo ""
    read -p "Select: " choice
    
    case $choice in
        1) python3 -c "from modules.drone.drone_warfare import DroneWarfare; print('Drone detection ready')" ;;
        2|3|4|5) echo "Use GUI or AI interface for drone warfare" ;;
        0) return ;;
    esac
}

# Spectrum submenu
spectrum_menu() {
    echo ""
    echo -e "${BLUE}═══════════ SPECTRUM ANALYSIS ═══════════${NC}"
    echo ""
    echo "  1) Full Spectrum Sweep"
    echo "  2) Signal Detection"
    echo "  3) Waterfall Display"
    echo "  4) Band Monitor"
    echo "  0) Back"
    echo ""
    read -p "Select: " choice
    
    case $choice in
        1|2|3|4) echo "Use GUI for spectrum analysis" ;;
        0) return ;;
    esac
}

# EW submenu
ew_menu() {
    echo ""
    echo -e "${BLUE}═══════════ ELECTRONIC WARFARE ═══════════${NC}"
    echo ""
    echo "  1) Band Jamming"
    echo "  2) Frequency Jamming"
    echo "  3) Adaptive Jamming"
    echo "  4) Reactive Jamming"
    echo "  5) Stop All Jamming"
    echo "  0) Back"
    echo ""
    read -p "Select: " choice
    
    case $choice in
        1|2|3|4) echo "Use GUI or AI interface for jamming operations" ;;
        5) echo "Stopping all jamming operations..." ;;
        0) return ;;
    esac
}

# SIGINT submenu
sigint_menu() {
    echo ""
    echo -e "${BLUE}═══════════ SIGINT OPERATIONS ═══════════${NC}"
    echo ""
    echo "  1) Passive Collection"
    echo "  2) Targeted Collection"
    echo "  3) Signal Classification"
    echo "  4) Pattern Analysis"
    echo "  5) Export Intelligence"
    echo "  0) Back"
    echo ""
    read -p "Select: " choice
    
    case $choice in
        1|2|3|4|5) echo "Use GUI or AI interface for SIGINT" ;;
        0) return ;;
    esac
}

# System status
system_status() {
    echo ""
    echo -e "${GREEN}═══════════ SYSTEM STATUS ═══════════${NC}"
    echo ""
    
    # Check BladeRF
    check_bladerf
    
    # Check Python
    check_python
    
    # Check Tor
    if systemctl is-active --quiet tor; then
        echo -e "${GREEN}[✓] Tor service running${NC}"
    else
        echo -e "${YELLOW}[!] Tor service not running${NC}"
    fi
    
    # Memory usage
    echo ""
    echo "Memory Usage:"
    free -h | head -2
    
    # Disk usage
    echo ""
    echo "Disk Usage:"
    df -h / | tail -1
    
    echo ""
}

# Hardware check
hardware_check() {
    echo ""
    echo -e "${GREEN}═══════════ HARDWARE CHECK ═══════════${NC}"
    echo ""
    
    # BladeRF
    if command -v bladeRF-cli &> /dev/null; then
        echo "BladeRF Status:"
        bladeRF-cli -p 2>/dev/null || echo "  No device connected"
        echo ""
    fi
    
    # USB devices
    echo "RF USB Devices:"
    lsusb 2>/dev/null | grep -iE "(blade|rtl|hackrf|sdr|nuand)" || echo "  No RF devices detected"
    echo ""
    
    # Network interfaces
    echo "Network Interfaces:"
    ip link show | grep -E "^[0-9]" | awk '{print "  " $2}'
    echo ""
}

# Emergency shutdown
emergency_shutdown() {
    echo -e "${RED}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                   EMERGENCY SHUTDOWN                          ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    read -p "Are you sure? This will stop all RF operations. (y/N): " confirm
    
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        echo -e "${RED}[!] Initiating emergency shutdown...${NC}"
        
        # Kill all RF Arsenal processes
        pkill -f "python.*rfarsenal" 2>/dev/null
        pkill -f "python.*ui/main_gui" 2>/dev/null
        pkill -f "python.*modules" 2>/dev/null
        
        # Reset BladeRF
        if command -v bladeRF-cli &> /dev/null; then
            bladeRF-cli -e "set frequency rx1 1000000000" 2>/dev/null
            bladeRF-cli -e "set txvga1 -35" 2>/dev/null
        fi
        
        echo -e "${GREEN}[✓] Emergency shutdown complete${NC}"
    else
        echo "Cancelled"
    fi
}

# Main loop
main() {
    show_banner
    
    # Initial checks
    check_root
    
    while true; do
        show_menu
        read -p "Select option: " choice
        
        case $choice in
            1)  launch_gui ;;
            2)  launch_text_ai ;;
            3)  launch_voice_ai ;;
            4)  cellular_menu ;;
            5)  wifi_menu ;;
            6)  gps_menu ;;
            7)  drone_menu ;;
            8)  spectrum_menu ;;
            9)  ew_menu ;;
            10) sigint_menu ;;
            11) system_status ;;
            12) hardware_check ;;
            13) emergency_shutdown ;;
            0)  
                echo -e "${GREEN}[*] Exiting RF Arsenal OS...${NC}"
                exit 0
                ;;
            *)  
                echo -e "${RED}Invalid option${NC}"
                ;;
        esac
    done
}

# Run
main "$@"
