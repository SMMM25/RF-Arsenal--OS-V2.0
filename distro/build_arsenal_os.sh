#!/bin/bash
#===============================================================================
# RF Arsenal OS - DragonOS Integration Build Script
#===============================================================================
# 
# This script creates a custom RF Arsenal OS distribution based on DragonOS.
# Supports: x86_64 (Desktop/Laptop), ARM64 (Raspberry Pi 4/5), Live USB
#
# Usage:
#   ./build_arsenal_os.sh [OPTIONS]
#
# Options:
#   --platform <x86_64|arm64|rpi>  Target platform (default: x86_64)
#   --mode <full|lite|stealth>     Build mode (default: full)
#   --output <path>                Output ISO/image path
#   --live-usb                     Create Live USB compatible image
#   --ram-only                     Enable RAM-only mode (no disk writes)
#   --help                         Show this help
#
# Author: RF Arsenal Team
# License: Proprietary - For authorized use only
#===============================================================================

set -e

#===============================================================================
# Configuration
#===============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$SCRIPT_DIR/build"
CONFIG_DIR="$SCRIPT_DIR/config"
OVERLAYS_DIR="$SCRIPT_DIR/overlays"

# Defaults
PLATFORM="x86_64"
BUILD_MODE="full"
OUTPUT_PATH="$BUILD_DIR/rf-arsenal-os.iso"
LIVE_USB=false
RAM_ONLY=false
DRAGONOS_ISO=""
DRAGONOS_URL_X86="https://sourceforge.net/projects/dragonos-focal/files/latest/download"
DRAGONOS_URL_RPI="https://sourceforge.net/projects/dragonos-pi64/files/latest/download"

# Version
VERSION="1.0.0"
BUILD_DATE=$(date +%Y%m%d)
DISTRO_NAME="RF Arsenal OS"
DISTRO_CODENAME="Phantom"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#===============================================================================
# Functions
#===============================================================================

print_banner() {
    echo -e "${BLUE}"
    cat << "EOF"
    ____  ______   ___                                __   ____  _____
   / __ \/ ____/  /   |  _____________  ____  ____ _/ /  / __ \/ ___/
  / /_/ / /_     / /| | / ___/ ___/ _ \/ __ \/ __ `/ /  / / / /\__ \ 
 / _, _/ __/    / ___ |/ /  (__  )  __/ / / / /_/ / /  / /_/ /___/ / 
/_/ |_/_/      /_/  |_/_/  /____/\___/_/ /_/\__,_/_/   \____//____/  
                                                                     
    DragonOS Integration Build System v${VERSION}
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

show_help() {
    cat << EOF
RF Arsenal OS Build Script v${VERSION}

Usage: $0 [OPTIONS]

Options:
  --platform <x86_64|arm64|rpi>  Target platform (default: x86_64)
  --mode <full|lite|stealth>     Build mode (default: full)
  --output <path>                Output ISO/image path
  --iso <path>                   Path to DragonOS ISO (optional, will download if not provided)
  --live-usb                     Create Live USB compatible image
  --ram-only                     Enable RAM-only mode (no disk writes)
  --help                         Show this help

Build Modes:
  full     - All features, GUI, development tools
  lite     - Minimal GUI, essential tools only (for Raspberry Pi)
  stealth  - Maximum OPSEC, no logging, RAM-only default

Platforms:
  x86_64   - Desktop/Laptop (Intel/AMD 64-bit)
  arm64    - Generic ARM 64-bit
  rpi      - Raspberry Pi 4/5 optimized

Examples:
  $0 --platform x86_64 --mode full
  $0 --platform rpi --mode lite --live-usb
  $0 --platform x86_64 --mode stealth --ram-only

EOF
}

check_dependencies() {
    log_info "Checking build dependencies..."
    
    local deps=(
        "squashfs-tools"
        "genisoimage"
        "xorriso"
        "wget"
        "curl"
        "git"
        "python3"
    )
    
    local missing=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null && ! dpkg -l | grep -q "^ii.*$dep"; then
            missing+=("$dep")
        fi
    done
    
    if [ ${#missing[@]} -gt 0 ]; then
        log_warn "Missing dependencies: ${missing[*]}"
        log_info "Install with: sudo apt install ${missing[*]}"
        
        read -p "Install now? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo apt update
            sudo apt install -y "${missing[@]}"
        else
            log_error "Cannot continue without dependencies"
            exit 1
        fi
    fi
    
    log_info "All dependencies satisfied"
}

download_dragonos() {
    local url=""
    local filename=""
    
    case "$PLATFORM" in
        x86_64|arm64)
            url="$DRAGONOS_URL_X86"
            filename="dragonos-focal-x86_64.iso"
            ;;
        rpi)
            url="$DRAGONOS_URL_RPI"
            filename="dragonos-pi64.img.xz"
            ;;
    esac
    
    local iso_path="$BUILD_DIR/$filename"
    
    if [ -f "$iso_path" ]; then
        log_info "DragonOS already downloaded: $iso_path"
        DRAGONOS_ISO="$iso_path"
        return
    fi
    
    log_info "Downloading DragonOS from SourceForge..."
    log_warn "This may take a while (4-8 GB download)"
    
    mkdir -p "$BUILD_DIR"
    wget -O "$iso_path" "$url" --progress=bar:force 2>&1
    
    if [ $? -eq 0 ]; then
        log_info "Download complete: $iso_path"
        DRAGONOS_ISO="$iso_path"
    else
        log_error "Download failed"
        exit 1
    fi
}

extract_iso() {
    log_info "Extracting DragonOS ISO..."
    
    local extract_dir="$BUILD_DIR/dragonos_extracted"
    local squashfs_dir="$BUILD_DIR/squashfs"
    
    # Clean previous extraction
    rm -rf "$extract_dir" "$squashfs_dir"
    mkdir -p "$extract_dir" "$squashfs_dir"
    
    # Mount and extract ISO
    local mount_point="/tmp/dragonos_mount_$$"
    mkdir -p "$mount_point"
    
    sudo mount -o loop "$DRAGONOS_ISO" "$mount_point"
    cp -a "$mount_point"/* "$extract_dir/"
    sudo umount "$mount_point"
    rmdir "$mount_point"
    
    # Extract squashfs filesystem
    local squashfs_file=$(find "$extract_dir" -name "*.squashfs" -o -name "filesystem.squashfs" 2>/dev/null | head -1)
    
    if [ -z "$squashfs_file" ]; then
        squashfs_file=$(find "$extract_dir" -name "*.squashfs" 2>/dev/null | head -1)
    fi
    
    if [ -n "$squashfs_file" ]; then
        log_info "Extracting squashfs: $squashfs_file"
        sudo unsquashfs -d "$squashfs_dir" "$squashfs_file"
    else
        log_warn "No squashfs found, using ISO directly"
        cp -a "$extract_dir"/* "$squashfs_dir/"
    fi
    
    log_info "Extraction complete"
}

install_rf_arsenal() {
    log_info "Installing RF Arsenal OS..."
    
    local squashfs_dir="$BUILD_DIR/squashfs"
    local arsenal_dest="$squashfs_dir/opt/rf-arsenal-os"
    
    # Create installation directory
    sudo mkdir -p "$arsenal_dest"
    
    # Copy RF Arsenal OS
    sudo cp -a "$PROJECT_ROOT"/* "$arsenal_dest/"
    
    # Remove unnecessary files
    sudo rm -rf "$arsenal_dest/.git"
    sudo rm -rf "$arsenal_dest/distro/build"
    sudo rm -rf "$arsenal_dest/__pycache__"
    sudo find "$arsenal_dest" -name "*.pyc" -delete
    sudo find "$arsenal_dest" -name "__pycache__" -type d -delete
    
    # Set permissions
    sudo chmod -R 755 "$arsenal_dest"
    sudo chown -R root:root "$arsenal_dest"
    
    log_info "RF Arsenal OS installed to /opt/rf-arsenal-os"
}

install_dependencies() {
    log_info "Installing Python dependencies..."
    
    local squashfs_dir="$BUILD_DIR/squashfs"
    
    # Create requirements install script
    cat << 'DEPS_EOF' | sudo tee "$squashfs_dir/tmp/install_deps.sh" > /dev/null
#!/bin/bash
pip3 install --upgrade pip
pip3 install numpy scipy
pip3 install pyserial
pip3 install cryptography
pip3 install scapy
pip3 install netifaces
pip3 install psutil
pip3 install pyyaml
pip3 install requests
pip3 install Flask  # For API endpoint
DEPS_EOF
    
    sudo chmod +x "$squashfs_dir/tmp/install_deps.sh"
    
    # Run in chroot
    sudo chroot "$squashfs_dir" /tmp/install_deps.sh 2>/dev/null || true
    
    # Cleanup
    sudo rm -f "$squashfs_dir/tmp/install_deps.sh"
    
    log_info "Dependencies installed"
}

configure_autostart() {
    log_info "Configuring auto-start..."
    
    local squashfs_dir="$BUILD_DIR/squashfs"
    
    # Create desktop entry
    sudo mkdir -p "$squashfs_dir/usr/share/applications"
    cat << 'DESKTOP_EOF' | sudo tee "$squashfs_dir/usr/share/applications/rf-arsenal.desktop" > /dev/null
[Desktop Entry]
Version=1.0
Type=Application
Name=RF Arsenal OS
Comment=AI-Powered RF Security Platform
Exec=/opt/rf-arsenal-os/distro/scripts/launch_arsenal.sh
Icon=/opt/rf-arsenal-os/distro/overlays/icons/rf-arsenal.png
Terminal=true
Categories=Security;System;
Keywords=RF;SDR;Security;Hacking;
StartupNotify=true
DESKTOP_EOF

    # Create autostart entry
    sudo mkdir -p "$squashfs_dir/etc/xdg/autostart"
    cat << 'AUTOSTART_EOF' | sudo tee "$squashfs_dir/etc/xdg/autostart/rf-arsenal.desktop" > /dev/null
[Desktop Entry]
Type=Application
Name=RF Arsenal OS
Exec=/opt/rf-arsenal-os/distro/scripts/launch_arsenal.sh --startup
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
AUTOSTART_EOF

    # Create systemd service for headless operation
    cat << 'SERVICE_EOF' | sudo tee "$squashfs_dir/etc/systemd/system/rf-arsenal.service" > /dev/null
[Unit]
Description=RF Arsenal OS AI Command Center
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/rf-arsenal-os
ExecStart=/usr/bin/python3 -m core.ai_command_center --headless
Restart=on-failure
RestartSec=5
Environment=PYTHONPATH=/opt/rf-arsenal-os

[Install]
WantedBy=multi-user.target
SERVICE_EOF

    log_info "Auto-start configured"
}

apply_opsec_hardening() {
    log_info "Applying OPSEC hardening..."
    
    local squashfs_dir="$BUILD_DIR/squashfs"
    
    # Remove telemetry packages
    cat << 'OPSEC_EOF' | sudo tee "$squashfs_dir/tmp/opsec_harden.sh" > /dev/null
#!/bin/bash
# Remove Ubuntu telemetry
apt-get remove -y popularity-contest apport whoopsie ubuntu-report 2>/dev/null || true
apt-get autoremove -y 2>/dev/null || true

# Disable telemetry services
systemctl disable apport.service 2>/dev/null || true
systemctl disable whoopsie.service 2>/dev/null || true

# Disable crash reporting
echo 'enabled=0' > /etc/default/apport 2>/dev/null || true

# Disable apt news
pro config set apt_news=false 2>/dev/null || true

# Configure privacy settings
mkdir -p /etc/gdm3
cat > /etc/gdm3/greeter.dconf-defaults << 'GREETER'
[org/gnome/login-screen]
disable-user-list=true
GREETER

# Disable recent files tracking
mkdir -p /etc/skel/.config
cat > /etc/skel/.config/user-dirs.conf << 'USERDIRS'
enabled=False
USERDIRS

echo "OPSEC hardening complete"
OPSEC_EOF
    
    sudo chmod +x "$squashfs_dir/tmp/opsec_harden.sh"
    sudo chroot "$squashfs_dir" /tmp/opsec_harden.sh 2>/dev/null || true
    sudo rm -f "$squashfs_dir/tmp/opsec_harden.sh"
    
    # Create OPSEC config
    sudo mkdir -p "$squashfs_dir/opt/rf-arsenal-os/config"
    cat << 'OPSEC_CONF' | sudo tee "$squashfs_dir/opt/rf-arsenal-os/config/opsec_defaults.yaml" > /dev/null
# RF Arsenal OS - OPSEC Default Configuration
# These settings are applied on first boot

network:
  mode: offline  # offline | tor | vpn | direct
  auto_connect: false
  block_telemetry: true
  dns_over_https: true

stealth:
  mac_randomization: true
  hostname_randomization: true
  disable_ipv6: true
  ram_only_mode: false  # Set true for maximum stealth
  secure_delete: true
  
logging:
  enabled: false  # Disable by default for OPSEC
  level: WARNING
  to_disk: false
  max_history: 0

forensics:
  anti_forensics: true
  clear_bash_history: true
  clear_logs_on_shutdown: true
  tmpfs_for_temp: true
OPSEC_CONF

    log_info "OPSEC hardening applied"
}

optimize_raspberry_pi() {
    log_info "Applying Raspberry Pi optimizations..."
    
    local squashfs_dir="$BUILD_DIR/squashfs"
    
    # Create Pi-specific config
    cat << 'PI_CONFIG' | sudo tee "$squashfs_dir/opt/rf-arsenal-os/config/raspberry_pi.yaml" > /dev/null
# RF Arsenal OS - Raspberry Pi Configuration

hardware:
  platform: raspberry_pi
  gpio_enabled: true
  spi_enabled: true
  i2c_enabled: true

performance:
  low_memory_mode: true
  max_ram_usage_mb: 2048
  reduce_animations: true
  lightweight_ui: true
  
display:
  small_screen_mode: true
  touch_enabled: true
  default_resolution: "800x480"
  font_scale: 1.2

power:
  battery_monitoring: true
  low_power_warnings: true
  auto_shutdown_percent: 5
  
gpio_assignments:
  panic_button: 17
  status_led_green: 27
  status_led_red: 22
  ptt_button: 23
PI_CONFIG

    # Create GPIO control script
    cat << 'GPIO_SCRIPT' | sudo tee "$squashfs_dir/opt/rf-arsenal-os/distro/scripts/gpio_control.py" > /dev/null
#!/usr/bin/env python3
"""
RF Arsenal OS - Raspberry Pi GPIO Control
Handles physical buttons, LEDs, and hardware interfaces
"""

import os
import sys
import time
import threading

# Check if running on Raspberry Pi
IS_PI = os.path.exists('/sys/firmware/devicetree/base/model')

if IS_PI:
    try:
        import RPi.GPIO as GPIO
        GPIO_AVAILABLE = True
    except ImportError:
        GPIO_AVAILABLE = False
else:
    GPIO_AVAILABLE = False

# GPIO Pin Assignments
PANIC_BUTTON = 17
STATUS_LED_GREEN = 27
STATUS_LED_RED = 22
PTT_BUTTON = 23

class GPIOController:
    def __init__(self):
        self.enabled = GPIO_AVAILABLE and IS_PI
        self._panic_callback = None
        self._ptt_callback = None
        
        if self.enabled:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            self._setup_pins()
    
    def _setup_pins(self):
        # Inputs with pull-up
        GPIO.setup(PANIC_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(PTT_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # Outputs
        GPIO.setup(STATUS_LED_GREEN, GPIO.OUT)
        GPIO.setup(STATUS_LED_RED, GPIO.OUT)
        
        # Initial state
        GPIO.output(STATUS_LED_GREEN, GPIO.LOW)
        GPIO.output(STATUS_LED_RED, GPIO.LOW)
        
        # Add interrupt handlers
        GPIO.add_event_detect(PANIC_BUTTON, GPIO.FALLING, 
                             callback=self._panic_pressed, bouncetime=300)
        GPIO.add_event_detect(PTT_BUTTON, GPIO.BOTH,
                             callback=self._ptt_changed, bouncetime=50)
    
    def _panic_pressed(self, channel):
        print("[GPIO] PANIC BUTTON PRESSED!")
        if self._panic_callback:
            self._panic_callback()
        self.blink_red(5)
    
    def _ptt_changed(self, channel):
        state = not GPIO.input(PTT_BUTTON)  # Inverted due to pull-up
        if self._ptt_callback:
            self._ptt_callback(state)
    
    def set_panic_callback(self, callback):
        self._panic_callback = callback
    
    def set_ptt_callback(self, callback):
        self._ptt_callback = callback
    
    def set_status(self, status):
        """Set status LED: 'ready', 'busy', 'error', 'stealth'"""
        if not self.enabled:
            return
        
        if status == 'ready':
            GPIO.output(STATUS_LED_GREEN, GPIO.HIGH)
            GPIO.output(STATUS_LED_RED, GPIO.LOW)
        elif status == 'busy':
            GPIO.output(STATUS_LED_GREEN, GPIO.HIGH)
            GPIO.output(STATUS_LED_RED, GPIO.HIGH)
        elif status == 'error':
            GPIO.output(STATUS_LED_GREEN, GPIO.LOW)
            GPIO.output(STATUS_LED_RED, GPIO.HIGH)
        elif status == 'stealth':
            GPIO.output(STATUS_LED_GREEN, GPIO.LOW)
            GPIO.output(STATUS_LED_RED, GPIO.LOW)
    
    def blink_red(self, times=3):
        if not self.enabled:
            return
        for _ in range(times):
            GPIO.output(STATUS_LED_RED, GPIO.HIGH)
            time.sleep(0.2)
            GPIO.output(STATUS_LED_RED, GPIO.LOW)
            time.sleep(0.2)
    
    def cleanup(self):
        if self.enabled:
            GPIO.cleanup()


# Singleton instance
_gpio_controller = None

def get_gpio_controller():
    global _gpio_controller
    if _gpio_controller is None:
        _gpio_controller = GPIOController()
    return _gpio_controller


if __name__ == "__main__":
    ctrl = get_gpio_controller()
    if ctrl.enabled:
        print("GPIO Controller initialized on Raspberry Pi")
        ctrl.set_status('ready')
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            ctrl.cleanup()
    else:
        print("GPIO not available (not running on Raspberry Pi)")
GPIO_SCRIPT
    
    sudo chmod +x "$squashfs_dir/opt/rf-arsenal-os/distro/scripts/gpio_control.py"
    
    # Enable SPI and I2C in boot config (for Pi)
    if [ -f "$squashfs_dir/boot/config.txt" ]; then
        sudo tee -a "$squashfs_dir/boot/config.txt" > /dev/null << 'BOOT_CONFIG'

# RF Arsenal OS Configuration
dtparam=spi=on
dtparam=i2c_arm=on
gpu_mem=128
# Overclock for better SDR performance (optional)
# arm_freq=2000
# over_voltage=6
BOOT_CONFIG
    fi
    
    log_info "Raspberry Pi optimizations applied"
}

configure_live_usb() {
    log_info "Configuring Live USB support..."
    
    local squashfs_dir="$BUILD_DIR/squashfs"
    
    # Create persistence setup script
    cat << 'PERSIST_SCRIPT' | sudo tee "$squashfs_dir/opt/rf-arsenal-os/distro/scripts/setup_persistence.sh" > /dev/null
#!/bin/bash
# RF Arsenal OS - Persistence Setup
# Run this after booting from USB to enable persistence

PERSISTENCE_FILE="/run/live/medium/persistence.img"
PERSISTENCE_SIZE="4G"

echo "RF Arsenal OS - Persistence Setup"
echo "=================================="

# Check if running from live USB
if [ ! -d "/run/live" ]; then
    echo "Not running from Live USB. Persistence not needed."
    exit 0
fi

# Check if persistence already exists
if [ -f "$PERSISTENCE_FILE" ]; then
    echo "Persistence file already exists."
    read -p "Recreate? This will DELETE all saved data. [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
    rm -f "$PERSISTENCE_FILE"
fi

# Create persistence file
echo "Creating ${PERSISTENCE_SIZE} persistence file..."
dd if=/dev/zero of="$PERSISTENCE_FILE" bs=1M count=4096 status=progress

# Format as ext4
echo "Formatting persistence volume..."
mkfs.ext4 -L persistence "$PERSISTENCE_FILE"

# Create persistence.conf
MOUNT_POINT="/tmp/persistence_mount"
mkdir -p "$MOUNT_POINT"
mount "$PERSISTENCE_FILE" "$MOUNT_POINT"
echo "/ union" > "$MOUNT_POINT/persistence.conf"
umount "$MOUNT_POINT"
rmdir "$MOUNT_POINT"

echo ""
echo "Persistence configured! Reboot to activate."
echo "Your data will be saved to the USB drive."
PERSIST_SCRIPT
    
    sudo chmod +x "$squashfs_dir/opt/rf-arsenal-os/distro/scripts/setup_persistence.sh"
    
    # Create RAM-only mode script
    cat << 'RAM_SCRIPT' | sudo tee "$squashfs_dir/opt/rf-arsenal-os/distro/scripts/enable_ram_only.sh" > /dev/null
#!/bin/bash
# RF Arsenal OS - RAM-Only Mode
# Ensures NO data is written to any disk

echo "RF Arsenal OS - RAM-Only Mode"
echo "=============================="
echo ""
echo "WARNING: All data will be lost on shutdown!"
echo "This mode provides maximum operational security."
echo ""

# Remount all filesystems as read-only
mount -o remount,ro /

# Create tmpfs for writable directories
mount -t tmpfs -o size=512M tmpfs /tmp
mount -t tmpfs -o size=256M tmpfs /var/log
mount -t tmpfs -o size=256M tmpfs /var/tmp
mount -t tmpfs -o size=128M tmpfs /home

# Disable swap
swapoff -a

# Set environment variable
export RF_ARSENAL_RAM_ONLY=1

echo ""
echo "RAM-Only mode enabled."
echo "All writes go to RAM. Nothing touches disk."
RAM_SCRIPT
    
    sudo chmod +x "$squashfs_dir/opt/rf-arsenal-os/distro/scripts/enable_ram_only.sh"
    
    log_info "Live USB support configured"
}

create_launcher_script() {
    log_info "Creating launcher script..."
    
    local squashfs_dir="$BUILD_DIR/squashfs"
    
    cat << 'LAUNCHER' | sudo tee "$squashfs_dir/opt/rf-arsenal-os/distro/scripts/launch_arsenal.sh" > /dev/null
#!/bin/bash
#===============================================================================
# RF Arsenal OS - Main Launcher
#===============================================================================

ARSENAL_DIR="/opt/rf-arsenal-os"
CONFIG_DIR="$ARSENAL_DIR/config"
LOG_DIR="/tmp/rf-arsenal-logs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Detect platform
detect_platform() {
    if [ -f /sys/firmware/devicetree/base/model ]; then
        if grep -q "Raspberry Pi" /sys/firmware/devicetree/base/model 2>/dev/null; then
            echo "rpi"
            return
        fi
    fi
    
    if [ "$(uname -m)" = "aarch64" ]; then
        echo "arm64"
    else
        echo "x86_64"
    fi
}

# Detect display
detect_display() {
    if [ -n "$DISPLAY" ]; then
        # Get screen resolution
        if command -v xrandr &> /dev/null; then
            RESOLUTION=$(xrandr | grep '\*' | awk '{print $1}' | head -1)
            WIDTH=$(echo $RESOLUTION | cut -d'x' -f1)
            if [ "$WIDTH" -lt 1024 ] 2>/dev/null; then
                echo "small"
                return
            fi
        fi
        echo "desktop"
    else
        echo "headless"
    fi
}

# Print banner
print_banner() {
    clear
    echo -e "${CYAN}"
    cat << "EOF"
 ██████╗ ███████╗     █████╗ ██████╗ ███████╗███████╗███╗   ██╗ █████╗ ██╗     
 ██╔══██╗██╔════╝    ██╔══██╗██╔══██╗██╔════╝██╔════╝████╗  ██║██╔══██╗██║     
 ██████╔╝█████╗      ███████║██████╔╝███████╗█████╗  ██╔██╗ ██║███████║██║     
 ██╔══██╗██╔══╝      ██╔══██║██╔══██╗╚════██║██╔══╝  ██║╚██╗██║██╔══██║██║     
 ██║  ██║██║         ██║  ██║██║  ██║███████║███████╗██║ ╚████║██║  ██║███████╗
 ╚═╝  ╚═╝╚═╝         ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝
                        AI-Powered RF Security Platform
EOF
    echo -e "${NC}"
    echo ""
}

# Check hardware
check_hardware() {
    echo -e "${BLUE}[*]${NC} Detecting SDR hardware..."
    
    # Check for BladeRF
    if command -v bladeRF-cli &> /dev/null; then
        if bladeRF-cli -p 2>/dev/null | grep -q "bladerf"; then
            echo -e "${GREEN}[✓]${NC} BladeRF detected"
        fi
    fi
    
    # Check for HackRF
    if command -v hackrf_info &> /dev/null; then
        if hackrf_info 2>/dev/null | grep -q "Found HackRF"; then
            echo -e "${GREEN}[✓]${NC} HackRF detected"
        fi
    fi
    
    # Check for RTL-SDR
    if command -v rtl_test &> /dev/null; then
        if rtl_test -t 2>&1 | grep -q "Found"; then
            echo -e "${GREEN}[✓]${NC} RTL-SDR detected"
        fi
    fi
    
    # SoapySDR enumeration
    if command -v SoapySDRUtil &> /dev/null; then
        SOAPY_DEVICES=$(SoapySDRUtil --find 2>/dev/null | grep -c "driver=")
        if [ "$SOAPY_DEVICES" -gt 0 ]; then
            echo -e "${GREEN}[✓]${NC} SoapySDR: $SOAPY_DEVICES device(s) found"
        fi
    fi
    
    echo ""
}

# Apply platform-specific settings
apply_platform_settings() {
    local platform=$1
    local display=$2
    
    export PYTHONPATH="$ARSENAL_DIR:$PYTHONPATH"
    export RF_ARSENAL_PLATFORM="$platform"
    export RF_ARSENAL_DISPLAY="$display"
    
    case "$platform" in
        rpi)
            export RF_ARSENAL_LOW_MEMORY=1
            export RF_ARSENAL_GPIO_ENABLED=1
            # Start GPIO controller
            python3 "$ARSENAL_DIR/distro/scripts/gpio_control.py" &
            ;;
    esac
    
    case "$display" in
        small)
            export RF_ARSENAL_SMALL_SCREEN=1
            ;;
        headless)
            export RF_ARSENAL_HEADLESS=1
            ;;
    esac
}

# First boot setup
first_boot_setup() {
    local marker_file="$CONFIG_DIR/.first_boot_complete"
    
    if [ -f "$marker_file" ]; then
        return
    fi
    
    echo -e "${YELLOW}[!]${NC} First boot detected - running setup..."
    echo ""
    
    # Hardware detection wizard
    python3 << 'WIZARD'
import sys
sys.path.insert(0, '/opt/rf-arsenal-os')

try:
    from install.hardware_wizard import get_hardware_wizard
    wizard = get_hardware_wizard()
    wizard.run_detection()
except Exception as e:
    print(f"Hardware wizard: {e}")
WIZARD
    
    # Create marker
    mkdir -p "$CONFIG_DIR"
    touch "$marker_file"
    
    echo ""
    echo -e "${GREEN}[✓]${NC} First boot setup complete"
    echo ""
    sleep 2
}

# Main entry point
main() {
    # Parse arguments
    STARTUP_MODE=false
    HEADLESS_MODE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --startup)
                STARTUP_MODE=true
                shift
                ;;
            --headless)
                HEADLESS_MODE=true
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    # Detect environment
    PLATFORM=$(detect_platform)
    DISPLAY_TYPE=$(detect_display)
    
    if [ "$HEADLESS_MODE" = true ]; then
        DISPLAY_TYPE="headless"
    fi
    
    # Print banner (unless headless)
    if [ "$DISPLAY_TYPE" != "headless" ]; then
        print_banner
        echo -e "${BLUE}Platform:${NC} $PLATFORM | ${BLUE}Display:${NC} $DISPLAY_TYPE"
        echo ""
    fi
    
    # Apply settings
    apply_platform_settings "$PLATFORM" "$DISPLAY_TYPE"
    
    # Check hardware
    check_hardware
    
    # First boot setup
    first_boot_setup
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Launch AI Command Center
    echo -e "${GREEN}[*]${NC} Starting AI Command Center..."
    echo ""
    
    cd "$ARSENAL_DIR"
    
    if [ "$DISPLAY_TYPE" = "headless" ]; then
        # Headless mode - just run the command center
        exec python3 -m core.ai_command_center --headless
    else
        # Interactive mode
        exec python3 -m core.ai_command_center
    fi
}

# Run
main "$@"
LAUNCHER
    
    sudo chmod +x "$squashfs_dir/opt/rf-arsenal-os/distro/scripts/launch_arsenal.sh"
    
    log_info "Launcher script created"
}

create_mobile_ui_config() {
    log_info "Creating mobile UI configuration..."
    
    local squashfs_dir="$BUILD_DIR/squashfs"
    
    # Create mobile UI config
    cat << 'MOBILE_UI' | sudo tee "$squashfs_dir/opt/rf-arsenal-os/config/mobile_ui.yaml" > /dev/null
# RF Arsenal OS - Mobile/Small Screen UI Configuration

display:
  # Optimized for 800x480 and similar small displays
  min_width: 480
  min_height: 320
  default_resolution: "800x480"
  
  # Font scaling for readability
  font_scale: 1.2
  header_font_size: 14
  body_font_size: 12
  mono_font_size: 11
  
  # Touch optimization
  touch_enabled: true
  button_min_size: 48  # px - minimum touch target
  scroll_sensitivity: 1.5

theme:
  # High contrast for outdoor visibility
  mode: dark
  contrast: high
  colors:
    background: "#0a0a0a"
    foreground: "#00ff00"
    accent: "#00ffff"
    warning: "#ffff00"
    error: "#ff0000"
    success: "#00ff00"

layout:
  # Compact layout for small screens
  compact_mode: true
  hide_menu_bar: false
  collapsible_panels: true
  
  # Status bar
  status_bar:
    position: top
    height: 24
    show_battery: true
    show_network: true
    show_gps: true
    show_clock: true
  
  # Quick action buttons
  quick_actions:
    enabled: true
    position: bottom
    buttons:
      - name: "Scan"
        icon: "radar"
        command: "scan spectrum"
      - name: "Capture"
        icon: "record"
        command: "capture iq 10s"
      - name: "Stealth"
        icon: "shield"
        command: "enable stealth"
      - name: "Panic"
        icon: "alert"
        command: "panic"

keyboard:
  # Virtual keyboard settings
  virtual_keyboard: auto  # auto | always | never
  prediction: false
  autocomplete: true

gestures:
  enabled: true
  swipe_left: "history back"
  swipe_right: "history forward"
  swipe_down: "show status"
  swipe_up: "hide status"
  long_press: "context menu"
  pinch_zoom: false
MOBILE_UI

    log_info "Mobile UI configuration created"
}

apply_branding() {
    log_info "Applying RF Arsenal OS branding..."
    
    local squashfs_dir="$BUILD_DIR/squashfs"
    
    # Create branding directory
    sudo mkdir -p "$squashfs_dir/opt/rf-arsenal-os/distro/overlays/icons"
    sudo mkdir -p "$squashfs_dir/opt/rf-arsenal-os/distro/overlays/wallpapers"
    
    # Update OS release info
    cat << RELEASE_EOF | sudo tee "$squashfs_dir/etc/os-release" > /dev/null
NAME="RF Arsenal OS"
VERSION="${VERSION}"
ID=rf-arsenal
ID_LIKE=ubuntu
PRETTY_NAME="RF Arsenal OS ${VERSION} (${DISTRO_CODENAME})"
VERSION_ID="${VERSION}"
HOME_URL="https://github.com/SMMM25/RF-Arsenal-OS"
DOCUMENTATION_URL="https://github.com/SMMM25/RF-Arsenal-OS/wiki"
BUG_REPORT_URL="https://github.com/SMMM25/RF-Arsenal-OS/issues"
VERSION_CODENAME=${DISTRO_CODENAME}
RELEASE_EOF

    # Update lsb-release
    cat << LSB_EOF | sudo tee "$squashfs_dir/etc/lsb-release" > /dev/null
DISTRIB_ID=RFArsenalOS
DISTRIB_RELEASE=${VERSION}
DISTRIB_CODENAME=${DISTRO_CODENAME}
DISTRIB_DESCRIPTION="RF Arsenal OS ${VERSION}"
LSB_EOF

    # Update issue
    cat << 'ISSUE_EOF' | sudo tee "$squashfs_dir/etc/issue" > /dev/null

 ██████╗ ███████╗     █████╗ ██████╗ ███████╗███████╗███╗   ██╗ █████╗ ██╗     
 ██╔══██╗██╔════╝    ██╔══██╗██╔══██╗██╔════╝██╔════╝████╗  ██║██╔══██╗██║     
 ██████╔╝█████╗      ███████║██████╔╝███████╗█████╗  ██╔██╗ ██║███████║██║     
 ██╔══██╗██╔══╝      ██╔══██║██╔══██╗╚════██║██╔══╝  ██║╚██╗██║██╔══██║██║     
 ██║  ██║██║         ██║  ██║██║  ██║███████║███████╗██║ ╚████║██║  ██║███████╗
 ╚═╝  ╚═╝╚═╝         ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝

        AI-Powered RF Security Platform | \l

ISSUE_EOF

    log_info "Branding applied"
}

rebuild_squashfs() {
    log_info "Rebuilding squashfs filesystem..."
    
    local squashfs_dir="$BUILD_DIR/squashfs"
    local extract_dir="$BUILD_DIR/dragonos_extracted"
    local new_squashfs="$BUILD_DIR/filesystem.squashfs"
    
    # Remove old squashfs
    rm -f "$new_squashfs"
    
    # Build new squashfs
    sudo mksquashfs "$squashfs_dir" "$new_squashfs" \
        -comp xz \
        -b 1M \
        -Xdict-size 100% \
        -no-recovery
    
    # Replace in extracted ISO
    local old_squashfs=$(find "$extract_dir" -name "*.squashfs" -o -name "filesystem.squashfs" 2>/dev/null | head -1)
    if [ -n "$old_squashfs" ]; then
        sudo cp "$new_squashfs" "$old_squashfs"
    fi
    
    log_info "Squashfs rebuilt"
}

create_iso() {
    log_info "Creating final ISO..."
    
    local extract_dir="$BUILD_DIR/dragonos_extracted"
    
    # Update ISO label
    local iso_label="RF_ARSENAL_OS_${VERSION}"
    
    # Create ISO
    xorriso -as mkisofs \
        -r -V "$iso_label" \
        -o "$OUTPUT_PATH" \
        -J -joliet-long \
        -b isolinux/isolinux.bin \
        -c isolinux/boot.cat \
        -no-emul-boot \
        -boot-load-size 4 \
        -boot-info-table \
        -isohybrid-mbr /usr/lib/ISOLINUX/isohdpfx.bin \
        -eltorito-alt-boot \
        -e boot/grub/efi.img \
        -no-emul-boot \
        -isohybrid-gpt-basdat \
        "$extract_dir" 2>/dev/null || \
    genisoimage -r -V "$iso_label" \
        -o "$OUTPUT_PATH" \
        -J -joliet-long \
        -b isolinux/isolinux.bin \
        -c isolinux/boot.cat \
        -no-emul-boot \
        -boot-load-size 4 \
        -boot-info-table \
        "$extract_dir"
    
    if [ -f "$OUTPUT_PATH" ]; then
        local size=$(du -h "$OUTPUT_PATH" | cut -f1)
        log_info "ISO created: $OUTPUT_PATH ($size)"
    else
        log_error "Failed to create ISO"
        exit 1
    fi
}

cleanup() {
    log_info "Cleaning up build files..."
    
    # Unmount any leftover mounts
    sudo umount /tmp/dragonos_mount_* 2>/dev/null || true
    
    # Optionally clean build directory
    read -p "Remove extracted files to save space? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$BUILD_DIR/squashfs"
        rm -rf "$BUILD_DIR/dragonos_extracted"
    fi
    
    log_info "Cleanup complete"
}

#===============================================================================
# Main
#===============================================================================

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --platform)
                PLATFORM="$2"
                shift 2
                ;;
            --mode)
                BUILD_MODE="$2"
                shift 2
                ;;
            --output)
                OUTPUT_PATH="$2"
                shift 2
                ;;
            --iso)
                DRAGONOS_ISO="$2"
                shift 2
                ;;
            --live-usb)
                LIVE_USB=true
                shift
                ;;
            --ram-only)
                RAM_ONLY=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validate platform
    case "$PLATFORM" in
        x86_64|arm64|rpi) ;;
        *)
            log_error "Invalid platform: $PLATFORM"
            exit 1
            ;;
    esac
    
    # Print banner
    print_banner
    
    log_info "Build Configuration:"
    log_info "  Platform: $PLATFORM"
    log_info "  Mode: $BUILD_MODE"
    log_info "  Output: $OUTPUT_PATH"
    log_info "  Live USB: $LIVE_USB"
    log_info "  RAM-Only: $RAM_ONLY"
    echo ""
    
    # Check if running as root
    if [ "$EUID" -ne 0 ]; then
        log_warn "This script requires root privileges for some operations."
        log_info "You may be prompted for sudo password."
    fi
    
    # Build steps
    check_dependencies
    
    if [ -z "$DRAGONOS_ISO" ]; then
        download_dragonos
    fi
    
    extract_iso
    install_rf_arsenal
    install_dependencies
    configure_autostart
    apply_opsec_hardening
    create_launcher_script
    create_mobile_ui_config
    
    if [ "$PLATFORM" = "rpi" ]; then
        optimize_raspberry_pi
    fi
    
    if [ "$LIVE_USB" = true ]; then
        configure_live_usb
    fi
    
    apply_branding
    rebuild_squashfs
    create_iso
    cleanup
    
    echo ""
    log_info "=========================================="
    log_info "RF Arsenal OS build complete!"
    log_info "=========================================="
    log_info ""
    log_info "Output: $OUTPUT_PATH"
    log_info ""
    log_info "To write to USB:"
    log_info "  sudo dd if=$OUTPUT_PATH of=/dev/sdX bs=4M status=progress"
    log_info ""
    log_info "Or use balenaEtcher for a GUI option."
}

main "$@"
