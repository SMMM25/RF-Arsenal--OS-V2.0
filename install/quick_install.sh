#!/bin/bash
# RF Arsenal OS - Quick Installation Script
# For installing on existing Raspberry Pi OS
#
# Usage: 
#   curl -fsSL https://raw.githubusercontent.com/SMMM25/RF-Arsenal-OS/main/install/quick_install.sh | sudo bash
#   OR
#   sudo bash quick_install.sh
#
# Copyright (c) 2024 RF-Arsenal-OS Project
# License: MIT

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
INSTALL_DIR="/opt/rf-arsenal-os"
REPO_URL="https://github.com/SMMM25/RF-Arsenal-OS.git"
BRANCH="main"

print_banner() {
    echo -e "${BLUE}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     🚀 RF ARSENAL OS - QUICK INSTALLER 🚀                ║
║                                                           ║
║          Installing on Raspberry Pi OS...                ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo -e "${RED}❌ This script must be run as root${NC}"
        echo "   Run: sudo bash quick_install.sh"
        exit 1
    fi
}

detect_system() {
    echo -e "${GREEN}==>${NC} Detecting system..."
    
    if [[ -f /proc/device-tree/model ]]; then
        MODEL=$(cat /proc/device-tree/model)
        echo "   Detected: $MODEL"
        
        if [[ $MODEL == *"Raspberry Pi 5"* ]]; then
            echo "   ✅ Raspberry Pi 5 - OPTIMAL"
        elif [[ $MODEL == *"Raspberry Pi 4"* ]]; then
            echo "   ✅ Raspberry Pi 4 - GOOD"
        elif [[ $MODEL == *"Raspberry Pi 3"* ]]; then
            echo "   ⚠️  Raspberry Pi 3 - MINIMUM"
        else
            echo "   ⚠️  Unknown Raspberry Pi model"
        fi
    else
        echo "   ⚠️  Not running on Raspberry Pi"
    fi
}

install_system_dependencies() {
    echo -e "${GREEN}==>${NC} Installing system dependencies..."
    
    apt-get update
    
    # Core tools
    apt-get install -y git python3 python3-pip python3-dev
    
    # BladeRF
    echo "   Installing BladeRF..."
    apt-get install -y libbladerf-dev bladerf
    
    # Network tools
    apt-get install -y tor i2p openvpn wireguard
    
    # Build tools
    apt-get install -y build-essential cmake
    
    # Bluetooth (for mesh networking)
    apt-get install -y bluez bluez-tools libbluetooth-dev
    
    # GPIO (Raspberry Pi)
    apt-get install -y python3-rpi.gpio || true
    
    echo "   ✅ System dependencies installed"
}

clone_repository() {
    echo -e "${GREEN}==>${NC} Cloning RF Arsenal OS repository..."
    
    if [[ -d "$INSTALL_DIR" ]]; then
        echo "   Removing existing installation..."
        rm -rf "$INSTALL_DIR"
    fi
    
    git clone -b "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
    
    echo "   ✅ Repository cloned to $INSTALL_DIR"
}

install_python_dependencies() {
    echo -e "${GREEN}==>${NC} Installing Python dependencies..."
    
    cd "$INSTALL_DIR"
    
    # Install requirements
    pip3 install -r install/requirements.txt --break-system-packages
    
    echo "   ✅ Python dependencies installed"
}

configure_system() {
    echo -e "${GREEN}==>${NC} Configuring system..."
    
    # Create config directories
    mkdir -p /etc/rf-arsenal
    mkdir -p /var/backups/rf-arsenal
    
    # Set permissions
    chown -R pi:pi "$INSTALL_DIR"
    
    # Create symlink
    ln -sf "$INSTALL_DIR/rf_arsenal_os.py" /usr/local/bin/rf-arsenal
    chmod +x /usr/local/bin/rf-arsenal
    
    echo "   ✅ System configured"
}

run_hardware_detection() {
    echo -e "${GREEN}==>${NC} Running hardware detection..."
    
    cd "$INSTALL_DIR"
    python3 install/pi_detect.py
    
    echo "   ✅ Hardware detection complete"
}

create_shortcuts() {
    echo -e "${GREEN}==>${NC} Creating desktop shortcuts..."
    
    # Create desktop directory
    mkdir -p /home/pi/Desktop
    
    # GUI shortcut
    cat > /home/pi/Desktop/RF-Arsenal-GUI.desktop <<EOF
[Desktop Entry]
Type=Application
Name=RF Arsenal OS (GUI)
Comment=Launch RF Arsenal OS GUI
Exec=sudo python3 $INSTALL_DIR/rf_arsenal_os.py
Terminal=true
Icon=$INSTALL_DIR/docs/icon.png
Categories=Development;System;
EOF
    
    # CLI shortcut
    cat > /home/pi/Desktop/RF-Arsenal-CLI.desktop <<EOF
[Desktop Entry]
Type=Application
Name=RF Arsenal OS (CLI)
Comment=Launch RF Arsenal OS CLI
Exec=sudo python3 $INSTALL_DIR/rf_arsenal_os.py --cli
Terminal=true
Icon=$INSTALL_DIR/docs/icon.png
Categories=Development;System;
EOF
    
    # Set permissions
    chmod +x /home/pi/Desktop/*.desktop
    chown pi:pi /home/pi/Desktop/*.desktop
    
    echo "   ✅ Desktop shortcuts created"
}

print_completion() {
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                           ║${NC}"
    echo -e "${GREEN}║         🎉 INSTALLATION COMPLETE! 🎉                     ║${NC}"
    echo -e "${GREEN}║                                                           ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "📍 Installation location: $INSTALL_DIR"
    echo ""
    echo "🚀 Quick Start:"
    echo "   • Launch GUI: sudo rf-arsenal"
    echo "   • Launch CLI: sudo rf-arsenal --cli"
    echo "   • Or use desktop shortcuts"
    echo ""
    echo "📚 Documentation: $INSTALL_DIR/docs/"
    echo ""
    echo "⚠️  IMPORTANT:"
    echo "   • For authorized penetration testing only"
    echo "   • BladeRF requires USB 3.0 for best performance"
    echo "   • Comply with local RF transmission laws"
    echo ""
    echo "🔄 To check for updates:"
    echo "   sudo python3 $INSTALL_DIR/update_manager.py --check"
    echo ""
}

main() {
    print_banner
    check_root
    detect_system
    install_system_dependencies
    clone_repository
    install_python_dependencies
    configure_system
    run_hardware_detection
    create_shortcuts
    print_completion
}

# Run installation
main "$@"
