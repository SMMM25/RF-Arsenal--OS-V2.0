#!/bin/bash
#===============================================================================
# RF Arsenal OS - Cross-Platform Quick Installer
#===============================================================================
#
# Universal installation script that works on any supported platform:
# - x86_64 Desktop/Laptop (Intel/AMD)
# - ARM64 Desktop/Laptop (Apple Silicon via compatible distros)
# - Raspberry Pi 5/4/3
# - Generic ARM64 SBCs
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/SMMM25/RF-Arsenal-OS/main/install/cross_platform_install.sh | sudo bash
#   OR
#   sudo bash cross_platform_install.sh
#
# README COMPLIANCE:
# - Offline-first: Works without network after initial download
# - RAM-only: Sensitive data operations in volatile memory
# - Zero telemetry: No external tracking or analytics
# - Real-world functional: Actual hardware detection and configuration
#
# Copyright (c) 2024 RF-Arsenal-OS Project
# License: Proprietary - Authorized Use Only
#===============================================================================

set -e

#===============================================================================
# Configuration
#===============================================================================

INSTALL_DIR="/opt/rf-arsenal-os"
CONFIG_DIR="/etc/rf-arsenal"
REPO_URL="https://github.com/SMMM25/RF-Arsenal-OS.git"
BRANCH="main"
VERSION="1.2.0"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Platform detection results
PLATFORM_TYPE=""
PLATFORM_ARCH=""
PLATFORM_VENDOR=""
PERFORMANCE_TIER=""
HAS_USB3=false
HAS_GPIO=false
MEMORY_GB=0
CPU_CORES=0

#===============================================================================
# Functions
#===============================================================================

print_banner() {
    echo -e "${BLUE}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║     ██████╗ ███████╗     █████╗ ██████╗ ███████╗███████╗███╗   ██╗ █████╗ ██╗ ║
║     ██╔══██╗██╔════╝    ██╔══██╗██╔══██╗██╔════╝██╔════╝████╗  ██║██╔══██╗██║ ║
║     ██████╔╝█████╗      ███████║██████╔╝███████╗█████╗  ██╔██╗ ██║███████║██║ ║
║     ██╔══██╗██╔══╝      ██╔══██║██╔══██╗╚════██║██╔══╝  ██║╚██╗██║██╔══██║██║ ║
║     ██║  ██║██║         ██║  ██║██║  ██║███████║███████╗██║ ╚████║██║  ██║███████╗ ║
║     ╚═╝  ╚═╝╚═╝         ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝ ║
║                                                                               ║
║                  🚀 CROSS-PLATFORM INSTALLER v1.2.0 🚀                       ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${CYAN}==> ${NC}$1"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        echo "   Run: sudo bash $0"
        exit 1
    fi
}

#===============================================================================
# Platform Detection
#===============================================================================

detect_platform() {
    log_step "Detecting platform..."
    
    # Get architecture
    PLATFORM_ARCH=$(uname -m)
    log_info "Architecture: $PLATFORM_ARCH"
    
    # Detect operating system
    if [[ -f /etc/os-release ]]; then
        source /etc/os-release
        log_info "OS: $PRETTY_NAME"
    fi
    
    # Check for Raspberry Pi
    if [[ -f /proc/device-tree/model ]]; then
        local model=$(cat /proc/device-tree/model 2>/dev/null || echo "")
        
        if [[ "$model" == *"Raspberry Pi 5"* ]]; then
            PLATFORM_TYPE="raspberry_pi_5"
            PLATFORM_VENDOR="Raspberry Pi Foundation"
            echo -e "   ${GREEN}✅ Raspberry Pi 5 detected - OPTIMAL${NC}"
        elif [[ "$model" == *"Raspberry Pi 4"* ]]; then
            PLATFORM_TYPE="raspberry_pi_4"
            PLATFORM_VENDOR="Raspberry Pi Foundation"
            echo -e "   ${GREEN}✅ Raspberry Pi 4 detected - GOOD${NC}"
        elif [[ "$model" == *"Raspberry Pi 3"* ]]; then
            PLATFORM_TYPE="raspberry_pi_3"
            PLATFORM_VENDOR="Raspberry Pi Foundation"
            echo -e "   ${YELLOW}⚠️  Raspberry Pi 3 detected - MINIMUM${NC}"
        elif [[ "$model" == *"Raspberry Pi Zero 2"* ]]; then
            PLATFORM_TYPE="raspberry_pi_zero2"
            PLATFORM_VENDOR="Raspberry Pi Foundation"
            echo -e "   ${YELLOW}⚠️  Raspberry Pi Zero 2 detected - LIMITED${NC}"
        else
            PLATFORM_TYPE="arm_sbc"
            PLATFORM_VENDOR="Generic ARM SBC"
            echo -e "   ${CYAN}ℹ️  ARM SBC detected: $model${NC}"
        fi
    # Check for x86_64
    elif [[ "$PLATFORM_ARCH" == "x86_64" ]] || [[ "$PLATFORM_ARCH" == "amd64" ]]; then
        # Check if laptop (has battery)
        if [[ -d /sys/class/power_supply/BAT0 ]] || [[ -d /sys/class/power_supply/BAT1 ]]; then
            PLATFORM_TYPE="x86_64_laptop"
            PLATFORM_VENDOR="PC Laptop"
            echo -e "   ${GREEN}✅ x86_64 Laptop detected${NC}"
        else
            PLATFORM_TYPE="x86_64_desktop"
            PLATFORM_VENDOR="PC Desktop"
            echo -e "   ${GREEN}✅ x86_64 Desktop detected${NC}"
        fi
    # Check for ARM64 (non-Pi)
    elif [[ "$PLATFORM_ARCH" == "aarch64" ]] || [[ "$PLATFORM_ARCH" == "arm64" ]]; then
        if [[ -d /sys/class/power_supply/BAT0 ]]; then
            PLATFORM_TYPE="arm64_laptop"
            PLATFORM_VENDOR="ARM64 Laptop"
        else
            PLATFORM_TYPE="arm64_desktop"
            PLATFORM_VENDOR="ARM64 Desktop"
        fi
        echo -e "   ${CYAN}ℹ️  ARM64 system detected${NC}"
    else
        PLATFORM_TYPE="unknown"
        PLATFORM_VENDOR="Unknown"
        echo -e "   ${YELLOW}⚠️  Unknown platform: $PLATFORM_ARCH${NC}"
    fi
    
    # Detect virtual machine
    local vm_type="none"
    if [[ -f /sys/class/dmi/id/product_name ]]; then
        local product=$(cat /sys/class/dmi/id/product_name 2>/dev/null | tr '[:upper:]' '[:lower:]')
        if [[ "$product" == *"vmware"* ]]; then
            vm_type="vmware"
        elif [[ "$product" == *"virtualbox"* ]]; then
            vm_type="virtualbox"
        elif [[ "$product" == *"kvm"* ]] || [[ "$product" == *"qemu"* ]]; then
            vm_type="kvm"
        elif [[ "$product" == *"hyper-v"* ]] || [[ "$product" == *"microsoft"* ]]; then
            vm_type="hyperv"
        fi
    fi
    
    if [[ "$vm_type" != "none" ]]; then
        echo -e "   ${YELLOW}⚠️  Virtual Machine detected: $vm_type${NC}"
        echo -e "   ${YELLOW}   USB passthrough may be required for SDR${NC}"
    fi
}

detect_hardware() {
    log_step "Detecting hardware capabilities..."
    
    # CPU cores
    CPU_CORES=$(nproc 2>/dev/null || echo 1)
    log_info "CPU Cores: $CPU_CORES"
    
    # Memory
    if [[ -f /proc/meminfo ]]; then
        local mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        MEMORY_GB=$((mem_kb / 1024 / 1024))
        log_info "Memory: ${MEMORY_GB} GB"
    fi
    
    # USB 3.0 detection
    if command -v lspci &> /dev/null; then
        if lspci 2>/dev/null | grep -qi "xhci\|usb.*3"; then
            HAS_USB3=true
            echo -e "   ${GREEN}✅ USB 3.0 available${NC}"
        else
            echo -e "   ${YELLOW}⚠️  USB 3.0 not detected - SDR bandwidth limited${NC}"
        fi
    elif [[ -d /sys/bus/usb ]]; then
        # Check USB device speeds
        for speed_file in /sys/bus/usb/devices/*/speed; do
            if [[ -f "$speed_file" ]]; then
                local speed=$(cat "$speed_file" 2>/dev/null)
                if [[ "$speed" == "5000" ]] || [[ "$speed" == "10000" ]]; then
                    HAS_USB3=true
                    echo -e "   ${GREEN}✅ USB 3.0 available${NC}"
                    break
                fi
            fi
        done
    fi
    
    # GPIO detection (Raspberry Pi)
    if [[ -d /sys/class/gpio ]]; then
        HAS_GPIO=true
        echo -e "   ${GREEN}✅ GPIO available (panic button supported)${NC}"
    fi
    
    # Determine performance tier
    if [[ $CPU_CORES -ge 8 ]] && [[ $MEMORY_GB -ge 16 ]]; then
        PERFORMANCE_TIER="high"
        echo -e "   ${GREEN}✅ Performance Tier: HIGH${NC}"
    elif [[ $CPU_CORES -ge 4 ]] && [[ $MEMORY_GB -ge 8 ]]; then
        PERFORMANCE_TIER="medium"
        echo -e "   ${CYAN}ℹ️  Performance Tier: MEDIUM${NC}"
    elif [[ $CPU_CORES -ge 2 ]] && [[ $MEMORY_GB -ge 4 ]]; then
        PERFORMANCE_TIER="low"
        echo -e "   ${YELLOW}⚠️  Performance Tier: LOW${NC}"
    else
        PERFORMANCE_TIER="minimal"
        echo -e "   ${YELLOW}⚠️  Performance Tier: MINIMAL${NC}"
    fi
}

#===============================================================================
# Package Manager Detection
#===============================================================================

detect_package_manager() {
    if command -v apt-get &> /dev/null; then
        PKG_MGR="apt"
        PKG_UPDATE="apt-get update"
        PKG_INSTALL="apt-get install -y"
    elif command -v dnf &> /dev/null; then
        PKG_MGR="dnf"
        PKG_UPDATE="dnf check-update || true"
        PKG_INSTALL="dnf install -y"
    elif command -v pacman &> /dev/null; then
        PKG_MGR="pacman"
        PKG_UPDATE="pacman -Sy"
        PKG_INSTALL="pacman -S --noconfirm"
    elif command -v apk &> /dev/null; then
        PKG_MGR="apk"
        PKG_UPDATE="apk update"
        PKG_INSTALL="apk add"
    else
        log_error "No supported package manager found"
        exit 1
    fi
    
    log_info "Package manager: $PKG_MGR"
}

#===============================================================================
# Installation Functions
#===============================================================================

install_base_dependencies() {
    log_step "Installing base dependencies..."
    
    $PKG_UPDATE
    
    # Core tools
    $PKG_INSTALL git python3 python3-pip python3-dev python3-venv
    $PKG_INSTALL build-essential cmake pkg-config
    $PKG_INSTALL curl wget
    
    log_info "Base dependencies installed"
}

install_sdr_dependencies() {
    log_step "Installing SDR dependencies..."
    
    case "$PKG_MGR" in
        apt)
            # BladeRF
            $PKG_INSTALL libbladerf-dev bladerf || log_warn "BladeRF packages not available"
            
            # HackRF
            $PKG_INSTALL hackrf libhackrf-dev || log_warn "HackRF packages not available"
            
            # RTL-SDR
            $PKG_INSTALL rtl-sdr librtlsdr-dev || log_warn "RTL-SDR packages not available"
            
            # LimeSDR
            $PKG_INSTALL limesuite || log_warn "LimeSDR packages not available"
            
            # SoapySDR
            $PKG_INSTALL soapysdr-tools soapysdr-module-all || log_warn "SoapySDR packages not available"
            ;;
        dnf)
            $PKG_INSTALL bladeRF hackrf rtl-sdr || log_warn "SDR packages may need manual installation"
            ;;
        pacman)
            $PKG_INSTALL bladerf hackrf rtl-sdr limesuite-git || log_warn "SDR packages may need AUR"
            ;;
        *)
            log_warn "SDR packages may need manual installation for this package manager"
            ;;
    esac
    
    log_info "SDR dependencies installed (where available)"
}

install_security_tools() {
    log_step "Installing security tools..."
    
    case "$PKG_MGR" in
        apt)
            $PKG_INSTALL tor || log_warn "Tor not available"
            $PKG_INSTALL macchanger || log_warn "macchanger not available"
            $PKG_INSTALL secure-delete || log_warn "secure-delete not available"
            $PKG_INSTALL cryptsetup || log_warn "cryptsetup not available"
            ;;
        dnf)
            $PKG_INSTALL tor macchanger || log_warn "Some security tools not available"
            ;;
        pacman)
            $PKG_INSTALL tor macchanger || log_warn "Some security tools not available"
            ;;
    esac
    
    log_info "Security tools installed (where available)"
}

install_platform_specific() {
    log_step "Installing platform-specific packages..."
    
    case "$PLATFORM_TYPE" in
        raspberry_pi_5|raspberry_pi_4|raspberry_pi_3|raspberry_pi_zero2)
            log_info "Installing Raspberry Pi specific packages..."
            $PKG_INSTALL python3-rpi.gpio raspi-gpio i2c-tools || true
            
            # Enable interfaces
            if command -v raspi-config &> /dev/null; then
                log_info "Enabling SPI and I2C..."
                raspi-config nonint do_spi 0 || true
                raspi-config nonint do_i2c 0 || true
            fi
            ;;
        x86_64_desktop|x86_64_laptop)
            log_info "Installing x86_64 specific packages..."
            # No specific packages needed
            ;;
        arm64_desktop|arm64_laptop)
            log_info "Installing ARM64 specific packages..."
            # No specific packages needed
            ;;
    esac
}

clone_repository() {
    log_step "Installing RF Arsenal OS..."
    
    if [[ -d "$INSTALL_DIR" ]]; then
        log_info "Updating existing installation..."
        cd "$INSTALL_DIR"
        git pull origin "$BRANCH" || {
            log_warn "Git pull failed, reinstalling..."
            rm -rf "$INSTALL_DIR"
            git clone -b "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
        }
    else
        git clone -b "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
    fi
    
    log_info "Repository cloned to $INSTALL_DIR"
}

install_python_dependencies() {
    log_step "Installing Python dependencies..."
    
    cd "$INSTALL_DIR"
    
    # Try to install with --break-system-packages for newer pip versions
    pip3 install -r install/requirements.txt --break-system-packages 2>/dev/null || \
    pip3 install -r install/requirements.txt || {
        log_warn "pip install failed, trying with venv..."
        python3 -m venv "$INSTALL_DIR/venv"
        "$INSTALL_DIR/venv/bin/pip" install -r install/requirements.txt
    }
    
    log_info "Python dependencies installed"
}

configure_system() {
    log_step "Configuring system..."
    
    # Create config directories
    mkdir -p "$CONFIG_DIR"
    mkdir -p /var/backups/rf-arsenal
    
    # Create RAM disk mount point
    mkdir -p /tmp/rf_arsenal_ram
    
    # Add to fstab if not already there
    if ! grep -q "rf_arsenal_ram" /etc/fstab; then
        echo "tmpfs /tmp/rf_arsenal_ram tmpfs rw,nodev,nosuid,size=512M 0 0" >> /etc/fstab
    fi
    
    # Create launcher symlink
    ln -sf "$INSTALL_DIR/rf_arsenal_os.py" /usr/local/bin/rf-arsenal
    chmod +x /usr/local/bin/rf-arsenal
    
    # Set permissions
    if [[ -n "$SUDO_USER" ]]; then
        chown -R "$SUDO_USER:$SUDO_USER" "$INSTALL_DIR"
    fi
    
    log_info "System configured"
}

apply_platform_optimizations() {
    log_step "Applying platform optimizations..."
    
    cd "$INSTALL_DIR"
    
    # Run Python platform optimizer
    python3 -c "
import sys
sys.path.insert(0, 'install')
from platform_detector import detect_platform, PlatformOptimizer, print_platform_summary

caps = detect_platform()
print_platform_summary(caps)

optimizer = PlatformOptimizer(caps)
result = optimizer.apply_optimizations()
print('Optimizations applied:', result)
" 2>/dev/null || log_warn "Platform optimization script not available"
    
    # Platform-specific configurations
    case "$PLATFORM_TYPE" in
        raspberry_pi_5)
            log_info "Applying Raspberry Pi 5 optimizations..."
            # Maximum performance
            echo "arm_freq=1800" >> /boot/config.txt 2>/dev/null || true
            echo "gpu_mem=256" >> /boot/config.txt 2>/dev/null || true
            ;;
        raspberry_pi_4)
            log_info "Applying Raspberry Pi 4 optimizations..."
            echo "gpu_mem=128" >> /boot/config.txt 2>/dev/null || true
            ;;
        raspberry_pi_3)
            log_info "Applying Raspberry Pi 3 optimizations (limited features)..."
            echo "gpu_mem=64" >> /boot/config.txt 2>/dev/null || true
            ;;
    esac
    
    log_info "Platform optimizations applied"
}

create_desktop_shortcuts() {
    log_step "Creating shortcuts..."
    
    # Determine user's home directory
    if [[ -n "$SUDO_USER" ]]; then
        USER_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
    else
        USER_HOME="$HOME"
    fi
    
    DESKTOP_DIR="$USER_HOME/Desktop"
    
    if [[ -d "$DESKTOP_DIR" ]]; then
        # GUI shortcut
        cat > "$DESKTOP_DIR/RF-Arsenal-OS.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=RF Arsenal OS
Comment=RF Security Testing Platform
Exec=sudo python3 $INSTALL_DIR/rf_arsenal_os.py
Terminal=true
Icon=$INSTALL_DIR/docs/icon.png
Categories=Development;System;Security;
EOF
        
        # CLI shortcut
        cat > "$DESKTOP_DIR/RF-Arsenal-CLI.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=RF Arsenal OS (CLI)
Comment=RF Arsenal OS Command Line Interface
Exec=sudo python3 $INSTALL_DIR/rf_arsenal_os.py --cli
Terminal=true
Icon=$INSTALL_DIR/docs/icon.png
Categories=Development;System;Security;
EOF
        
        chmod +x "$DESKTOP_DIR"/*.desktop
        
        if [[ -n "$SUDO_USER" ]]; then
            chown "$SUDO_USER:$SUDO_USER" "$DESKTOP_DIR"/*.desktop
        fi
        
        log_info "Desktop shortcuts created"
    fi
}

print_completion() {
    echo ""
    echo -e "${GREEN}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                   🎉 INSTALLATION COMPLETE! 🎉                               ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    
    echo ""
    echo "📍 Installation Details:"
    echo "   Location:    $INSTALL_DIR"
    echo "   Platform:    $PLATFORM_TYPE"
    echo "   Performance: $PERFORMANCE_TIER"
    echo ""
    echo "🚀 Quick Start:"
    echo "   • Launch GUI: sudo rf-arsenal"
    echo "   • Launch CLI: sudo rf-arsenal --cli"
    echo "   • Health check: sudo rf-arsenal --check"
    echo ""
    
    if [[ "$HAS_USB3" == "false" ]]; then
        echo -e "${YELLOW}⚠️  USB 3.0 not detected:${NC}"
        echo "   BladeRF performance will be limited on USB 2.0"
        echo ""
    fi
    
    if [[ "$HAS_GPIO" == "true" ]]; then
        echo "📍 GPIO Features Available:"
        echo "   • Panic button (GPIO 17)"
        echo "   • Status LEDs (GPIO 22, 27)"
        echo ""
    fi
    
    case "$PLATFORM_TYPE" in
        raspberry_pi_3)
            echo -e "${YELLOW}⚠️  Raspberry Pi 3 Limitations:${NC}"
            echo "   • AI features disabled (insufficient memory)"
            echo "   • Real-time spectrum limited"
            echo "   • Consider upgrading to Pi 4/5"
            echo ""
            ;;
    esac
    
    echo "📚 Documentation: $INSTALL_DIR/docs/"
    echo ""
    echo "⚠️  IMPORTANT: For authorized penetration testing only"
    echo "   Comply with all local RF transmission laws"
    echo ""
}

#===============================================================================
# Main
#===============================================================================

main() {
    print_banner
    check_root
    
    echo ""
    log_step "Starting RF Arsenal OS Installation..."
    echo ""
    
    # Platform detection
    detect_platform
    detect_hardware
    detect_package_manager
    
    echo ""
    echo "Press Enter to continue with installation, or Ctrl+C to cancel..."
    read -r
    
    # Installation steps
    install_base_dependencies
    install_sdr_dependencies
    install_security_tools
    install_platform_specific
    clone_repository
    install_python_dependencies
    configure_system
    apply_platform_optimizations
    create_desktop_shortcuts
    
    print_completion
}

# Run main function
main "$@"
