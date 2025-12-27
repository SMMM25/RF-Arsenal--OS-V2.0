#!/bin/bash
# ============================================================================
# RF Arsenal OS - USB Installer Creator
# ============================================================================
# Creates a bootable USB drive that can install RF Arsenal OS onto
# a Raspberry Pi (SD card or USB boot) with a single command.
#
# USAGE:
#   1. On any Linux machine: sudo bash create_usb_installer.sh /dev/sdX
#   2. Plug USB into Raspberry Pi and boot
#   3. System auto-installs and configures
#
# SUPPORTS:
#   - Raspberry Pi 5 (8GB recommended) - OPTIMAL
#   - Raspberry Pi 4 (4GB+) - FULL SUPPORT
#   - Raspberry Pi 3 B+ (2GB) - BASIC FEATURES
#   - RasPad 3 touchscreen integration
#
# Copyright (c) 2024 RF-Arsenal-OS Project
# License: MIT
# ============================================================================

set -e

# Configuration
VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORK_DIR="/tmp/rf-arsenal-usb-build"
RASPIOS_URL="https://downloads.raspberrypi.org/raspios_lite_arm64/images/raspios_lite_arm64-2024-11-19/2024-11-19-raspios-bookworm-arm64-lite.img.xz"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================================
# FUNCTIONS
# ============================================================================

print_banner() {
    clear
    echo -e "${CYAN}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ██████╗ ███████╗     █████╗ ██████╗ ███████╗███████╗███╗   ██╗ █████╗ ██╗║
║   ██╔══██╗██╔════╝    ██╔══██╗██╔══██╗██╔════╝██╔════╝████╗  ██║██╔══██╗██║║
║   ██████╔╝█████╗█████╗███████║██████╔╝███████╗█████╗  ██╔██╗ ██║███████║██║║
║   ██╔══██╗██╔══╝╚════╝██╔══██║██╔══██╗╚════██║██╔══╝  ██║╚██╗██║██╔══██║██║║
║   ██║  ██║██║         ██║  ██║██║  ██║███████║███████╗██║ ╚████║██║  ██║███║
║   ╚═╝  ╚═╝╚═╝         ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══╝
║                                                                           ║
║                    USB INSTALLER CREATOR v1.0                             ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}==>${NC} $1"; }

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        echo "  Usage: sudo bash $0 /dev/sdX"
        exit 1
    fi
}

check_dependencies() {
    log_step "Checking dependencies..."
    
    local deps=(wget xz parted mkfs.ext4 mkfs.vfat losetup rsync)
    local missing=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing+=("$dep")
        fi
    done
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing[*]}"
        echo "  Install: sudo apt-get install ${missing[*]}"
        exit 1
    fi
    
    log_info "All dependencies available"
}

select_usb_device() {
    if [[ -n "$1" ]]; then
        USB_DEVICE="$1"
    else
        echo ""
        log_step "Available USB devices:"
        echo ""
        lsblk -d -o NAME,SIZE,MODEL,TRAN | grep -E "usb|NAME"
        echo ""
        read -p "Enter USB device (e.g., /dev/sdb): " USB_DEVICE
    fi
    
    # Validate device
    if [[ ! -b "$USB_DEVICE" ]]; then
        log_error "Device not found: $USB_DEVICE"
        exit 1
    fi
    
    # Safety check - don't write to system disk
    if [[ "$USB_DEVICE" == "/dev/sda" ]] || [[ "$USB_DEVICE" == "/dev/nvme0n1" ]]; then
        log_error "Refusing to write to system disk: $USB_DEVICE"
        exit 1
    fi
    
    # Confirm
    echo ""
    log_warn "WARNING: All data on $USB_DEVICE will be DESTROYED!"
    echo ""
    lsblk "$USB_DEVICE"
    echo ""
    read -p "Are you sure? Type 'YES' to confirm: " confirm
    
    if [[ "$confirm" != "YES" ]]; then
        log_info "Aborted by user"
        exit 0
    fi
}

prepare_workspace() {
    log_step "Preparing workspace..."
    
    mkdir -p "$WORK_DIR"/{boot,root,image}
    cd "$WORK_DIR"
    
    # Check disk space (need ~15GB)
    local available=$(df "$WORK_DIR" | tail -1 | awk '{print $4}')
    if [[ $available -lt 15000000 ]]; then
        log_error "Insufficient disk space. Need 15GB, have $((available/1024/1024))GB"
        exit 1
    fi
    
    log_info "Workspace ready: $WORK_DIR"
}

download_raspios() {
    log_step "Downloading Raspberry Pi OS..."
    
    local img_file="$WORK_DIR/raspios.img"
    local xz_file="$WORK_DIR/raspios.img.xz"
    
    if [[ -f "$img_file" ]]; then
        log_info "Using cached image"
        return
    fi
    
    wget -O "$xz_file" "$RASPIOS_URL"
    
    log_step "Extracting image..."
    xz -d "$xz_file"
    mv "${xz_file%.xz}" "$img_file"
    
    log_info "Image ready"
}

create_installer_image() {
    log_step "Creating installer image..."
    
    local img_file="$WORK_DIR/raspios.img"
    
    # Expand image to 12GB
    truncate -s 12G "$img_file"
    
    # Setup loop device
    LOOP_DEV=$(losetup -f --show -P "$img_file")
    
    # Resize partition
    parted -s "$LOOP_DEV" resizepart 2 100%
    e2fsck -f "${LOOP_DEV}p2" || true
    resize2fs "${LOOP_DEV}p2"
    
    # Mount
    mount "${LOOP_DEV}p2" "$WORK_DIR/root"
    mount "${LOOP_DEV}p1" "$WORK_DIR/boot"
    
    log_info "Image mounted"
}

install_rf_arsenal() {
    log_step "Installing RF Arsenal OS..."
    
    local root="$WORK_DIR/root"
    local install_dir="$root/opt/rf-arsenal-os"
    
    # Create directory structure
    mkdir -p "$install_dir"
    mkdir -p "$root/etc/rf-arsenal"
    mkdir -p "$root/var/log/rf-arsenal"
    
    # Copy entire project
    log_info "Copying RF Arsenal OS files..."
    rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        "$PROJECT_ROOT/" "$install_dir/"
    
    # Set permissions
    chmod -R 755 "$install_dir/install"
    chmod +x "$install_dir/rf_arsenal.py"
    chmod +x "$install_dir/rf_arsenal_os.py"
    
    log_info "RF Arsenal OS installed to /opt/rf-arsenal-os"
}

configure_auto_install() {
    log_step "Configuring auto-install on first boot..."
    
    local root="$WORK_DIR/root"
    local boot="$WORK_DIR/boot"
    
    # Create first-boot service
    cat > "$root/etc/systemd/system/rf-arsenal-setup.service" << 'EOF'
[Unit]
Description=RF Arsenal OS First Boot Setup
After=network-online.target
Wants=network-online.target
ConditionPathExists=!/opt/rf-arsenal-os/.installed

[Service]
Type=oneshot
ExecStart=/opt/rf-arsenal-os/install/first_boot_setup.sh
RemainAfterExit=yes
StandardOutput=journal+console
StandardError=journal+console

[Install]
WantedBy=multi-user.target
EOF

    # Create first boot setup script
    cat > "$root/opt/rf-arsenal-os/install/first_boot_setup.sh" << 'SETUP_EOF'
#!/bin/bash
# RF Arsenal OS - First Boot Setup
# Runs automatically on first boot to complete installation

set -e

LOG_FILE="/var/log/rf-arsenal/first_boot.log"
INSTALL_DIR="/opt/rf-arsenal-os"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================="
echo "  RF Arsenal OS - First Boot Setup"
echo "  $(date)"
echo "=============================================="

# Detect hardware
echo "[1/8] Detecting hardware..."
PI_MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "Unknown")
echo "  Model: $PI_MODEL"

TOTAL_RAM=$(free -m | awk '/^Mem:/{print $2}')
echo "  RAM: ${TOTAL_RAM}MB"

# Expand filesystem
echo "[2/8] Expanding filesystem..."
raspi-config --expand-rootfs nonint || true

# Update system
echo "[3/8] Updating system packages..."
apt-get update
apt-get upgrade -y

# Install system dependencies
echo "[4/8] Installing system dependencies..."
apt-get install -y \
    python3 python3-pip python3-dev python3-venv \
    git wget curl \
    libusb-1.0-0-dev libbladerf-dev bladerf \
    libfftw3-dev libboost-all-dev \
    cmake build-essential \
    hostapd dnsmasq \
    network-manager \
    i2c-tools

# Install Python dependencies
echo "[5/8] Installing Python dependencies..."
cd "$INSTALL_DIR"
pip3 install --break-system-packages -r install/requirements.txt || \
    pip3 install -r install/requirements.txt

# Configure BladeRF
echo "[6/8] Configuring BladeRF..."
if ! getent group bladerf > /dev/null; then
    groupadd bladerf
fi
usermod -aG bladerf pi 2>/dev/null || usermod -aG bladerf $SUDO_USER 2>/dev/null || true

# Setup udev rules for SDR devices
cat > /etc/udev/rules.d/88-rf-arsenal-sdr.rules << 'UDEV_EOF'
# BladeRF
ATTR{idVendor}=="2cf0", ATTR{idProduct}=="5246", MODE="0666", GROUP="bladerf"
ATTR{idVendor}=="2cf0", ATTR{idProduct}=="5250", MODE="0666", GROUP="bladerf"
# HackRF
ATTR{idVendor}=="1d50", ATTR{idProduct}=="6089", MODE="0666", GROUP="plugdev"
# RTL-SDR
ATTR{idVendor}=="0bda", ATTR{idProduct}=="2838", MODE="0666", GROUP="plugdev"
# LimeSDR
ATTR{idVendor}=="1d50", ATTR{idProduct}=="6108", MODE="0666", GROUP="plugdev"
# PlutoSDR
ATTR{idVendor}=="0456", ATTR{idProduct}=="b673", MODE="0666", GROUP="plugdev"
UDEV_EOF

udevadm control --reload-rules
udevadm trigger

# Create RAM disk for stealth operations
echo "[7/8] Configuring stealth features..."
mkdir -p /tmp/rf_arsenal_ram

# Add to fstab for persistent RAM disk
if ! grep -q "rf_arsenal_ram" /etc/fstab; then
    echo "tmpfs /tmp/rf_arsenal_ram tmpfs nodev,nosuid,size=512M 0 0" >> /etc/fstab
fi

# Create launcher script
echo "[8/8] Creating launcher..."
cat > /usr/local/bin/rf-arsenal << 'LAUNCHER_EOF'
#!/bin/bash
cd /opt/rf-arsenal-os
python3 rf_arsenal_os.py "$@"
LAUNCHER_EOF
chmod +x /usr/local/bin/rf-arsenal

# Create desktop shortcut (if desktop environment exists)
if [[ -d /usr/share/applications ]]; then
    cat > /usr/share/applications/rf-arsenal.desktop << 'DESKTOP_EOF'
[Desktop Entry]
Name=RF Arsenal OS
Comment=RF Security Testing Platform
Exec=/usr/local/bin/rf-arsenal --gui
Icon=/opt/rf-arsenal-os/ui/icon.png
Terminal=false
Type=Application
Categories=Security;Network;
DESKTOP_EOF
fi

# Mark installation complete
touch "$INSTALL_DIR/.installed"
echo "$(date)" > "$INSTALL_DIR/.installed"

echo ""
echo "=============================================="
echo "  ✅ RF Arsenal OS Installation Complete!"
echo "=============================================="
echo ""
echo "  Launch with: rf-arsenal"
echo "  Or GUI mode: rf-arsenal --gui"
echo ""
echo "  Rebooting in 10 seconds..."
echo ""

sleep 10
reboot
SETUP_EOF

    chmod +x "$root/opt/rf-arsenal-os/install/first_boot_setup.sh"
    
    # Enable the service
    ln -sf /etc/systemd/system/rf-arsenal-setup.service \
        "$root/etc/systemd/system/multi-user.target.wants/rf-arsenal-setup.service"
    
    # Enable SSH
    touch "$boot/ssh"
    
    # Configure WiFi (optional - user can set)
    cat > "$boot/wpa_supplicant.conf.example" << 'EOF'
# Rename this file to wpa_supplicant.conf and edit with your WiFi details
# This will auto-connect on boot

country=US
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
    ssid="YOUR_WIFI_SSID"
    psk="YOUR_WIFI_PASSWORD"
    key_mgmt=WPA-PSK
}
EOF

    # Set default password reminder
    cat > "$root/etc/motd" << 'EOF'

╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🛡️  RF ARSENAL OS - Mobile RF Security Platform  🛡️   ║
║                                                           ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║   Launch:     rf-arsenal                                  ║
║   GUI Mode:   rf-arsenal --gui                            ║
║   Help:       rf-arsenal --help                           ║
║                                                           ║
║   ⚠️  CHANGE DEFAULT PASSWORD: passwd                     ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

EOF

    log_info "Auto-install configured"
}

configure_raspad() {
    log_step "Configuring RasPad touchscreen support..."
    
    local root="$WORK_DIR/root"
    local boot="$WORK_DIR/boot"
    
    # RasPad 3 display configuration
    cat >> "$boot/config.txt" << 'EOF'

# RasPad 3 Display Configuration
# Uncomment if using RasPad 3 touchscreen
#dtoverlay=vc4-kms-v3d
#max_framebuffers=2
#hdmi_force_hotplug=1
#hdmi_group=2
#hdmi_mode=87
#hdmi_cvt=1024 600 60 6 0 0 0
#hdmi_drive=1

# Touch calibration (RasPad 3)
#dtoverlay=ads7846,cs=1,penirq=25,penirq_pull=2,speed=50000,keep_vref_on=0,swapxy=0,pmax=255,xohms=150,xmin=200,xmax=3900,ymin=200,ymax=3900
EOF

    # Create RasPad setup script
    cat > "$root/opt/rf-arsenal-os/install/setup_raspad.sh" << 'RASPAD_EOF'
#!/bin/bash
# RasPad 3 Touchscreen Setup

echo "Configuring RasPad 3 display..."

# Install touchscreen drivers
apt-get install -y xserver-xorg-input-evdev

# Enable display settings in config.txt
sed -i 's/^#dtoverlay=vc4-kms-v3d/dtoverlay=vc4-kms-v3d/' /boot/config.txt
sed -i 's/^#max_framebuffers=2/max_framebuffers=2/' /boot/config.txt
sed -i 's/^#hdmi_force_hotplug=1/hdmi_force_hotplug=1/' /boot/config.txt
sed -i 's/^#hdmi_group=2/hdmi_group=2/' /boot/config.txt
sed -i 's/^#hdmi_mode=87/hdmi_mode=87/' /boot/config.txt
sed -i 's/^#hdmi_cvt=1024 600/hdmi_cvt=1024 600/' /boot/config.txt

# Create touch calibration
mkdir -p /etc/X11/xorg.conf.d
cat > /etc/X11/xorg.conf.d/99-calibration.conf << 'CALIB_EOF'
Section "InputClass"
    Identifier "calibration"
    MatchProduct "ADS7846 Touchscreen"
    Option "Calibration" "200 3900 200 3900"
    Option "SwapAxes" "0"
EndSection
CALIB_EOF

echo "RasPad configuration complete. Please reboot."
RASPAD_EOF

    chmod +x "$root/opt/rf-arsenal-os/install/setup_raspad.sh"
    
    log_info "RasPad support configured"
}

optimize_for_mobile() {
    log_step "Optimizing for mobile/portable operation..."
    
    local root="$WORK_DIR/root"
    local boot="$WORK_DIR/boot"
    
    # Power management optimizations
    cat >> "$boot/config.txt" << 'EOF'

# Power Management for Mobile Operation
# Reduce power consumption on battery
#arm_freq=1500
#over_voltage=0
#gpu_mem=128

# Disable unused interfaces to save power (uncomment as needed)
#dtoverlay=disable-bt
#dtoverlay=disable-wifi
EOF

    # Create power profile scripts
    mkdir -p "$root/opt/rf-arsenal-os/profiles"
    
    # Low power profile
    cat > "$root/opt/rf-arsenal-os/profiles/low_power.sh" << 'LOWPOW_EOF'
#!/bin/bash
# Low Power Profile - Extends battery life

echo "Activating low power profile..."

# Reduce CPU frequency
echo "powersave" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Disable HDMI (if headless)
# tvservice -o

# Reduce GPU memory
# Requires reboot: gpu_mem=64 in config.txt

# Disable USB power on unused ports
# echo '1-1' > /sys/bus/usb/drivers/usb/unbind

echo "Low power profile active"
LOWPOW_EOF

    # High performance profile
    cat > "$root/opt/rf-arsenal-os/profiles/high_performance.sh" << 'HIGHPERF_EOF'
#!/bin/bash
# High Performance Profile - Maximum speed for demanding operations

echo "Activating high performance profile..."

# Max CPU frequency
echo "performance" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Increase GPU memory (requires reboot)
# gpu_mem=256 in config.txt

echo "High performance profile active"
HIGHPERF_EOF

    # Stealth profile (minimal emissions)
    cat > "$root/opt/rf-arsenal-os/profiles/stealth.sh" << 'STEALTH_EOF'
#!/bin/bash
# Stealth Profile - Minimal RF emissions and forensic footprint

echo "Activating stealth profile..."

# Disable WiFi
rfkill block wifi
ip link set wlan0 down 2>/dev/null

# Disable Bluetooth
rfkill block bluetooth
systemctl stop bluetooth

# Disable logging to disk
systemctl stop rsyslog
systemctl stop systemd-journald

# Mount /var/log as tmpfs
mount -t tmpfs -o size=64M tmpfs /var/log

# Clear command history
export HISTFILE=/dev/null
history -c

# Randomize MAC on any remaining interfaces
for iface in $(ip link show | grep -E "^[0-9]+:" | cut -d: -f2 | tr -d ' '); do
    if [[ "$iface" != "lo" ]]; then
        ip link set "$iface" down
        macchanger -r "$iface" 2>/dev/null || true
        ip link set "$iface" up
    fi
done

echo "Stealth profile active - minimal RF/forensic footprint"
STEALTH_EOF

    chmod +x "$root/opt/rf-arsenal-os/profiles/"*.sh
    
    log_info "Mobile optimizations configured"
}

write_to_usb() {
    log_step "Writing image to USB drive..."
    
    # Unmount image
    umount "$WORK_DIR/boot" || true
    umount "$WORK_DIR/root" || true
    losetup -d "$LOOP_DEV" || true
    
    local img_file="$WORK_DIR/raspios.img"
    
    # Write to USB
    log_warn "Writing to $USB_DEVICE - this will take several minutes..."
    dd if="$img_file" of="$USB_DEVICE" bs=4M status=progress conv=fsync
    
    # Sync
    sync
    
    log_info "Image written to USB"
}

cleanup() {
    log_step "Cleaning up..."
    
    umount "$WORK_DIR/boot" 2>/dev/null || true
    umount "$WORK_DIR/root" 2>/dev/null || true
    
    if [[ -n "$LOOP_DEV" ]]; then
        losetup -d "$LOOP_DEV" 2>/dev/null || true
    fi
    
    # Optionally keep workspace for debugging
    # rm -rf "$WORK_DIR"
    
    log_info "Cleanup complete"
}

# ============================================================================
# MAIN
# ============================================================================

trap cleanup EXIT

print_banner
check_root
check_dependencies
select_usb_device "$1"
prepare_workspace
download_raspios
create_installer_image
install_rf_arsenal
configure_auto_install
configure_raspad
optimize_for_mobile
write_to_usb

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                           ║${NC}"
echo -e "${GREEN}║   ✅  USB INSTALLER CREATED SUCCESSFULLY!                 ║${NC}"
echo -e "${GREEN}║                                                           ║${NC}"
echo -e "${GREEN}╠═══════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                                           ║${NC}"
echo -e "${GREEN}║   Next Steps:                                             ║${NC}"
echo -e "${GREEN}║                                                           ║${NC}"
echo -e "${GREEN}║   1. Remove USB drive safely                              ║${NC}"
echo -e "${GREEN}║   2. Insert into Raspberry Pi                             ║${NC}"
echo -e "${GREEN}║   3. Power on - installation is automatic                 ║${NC}"
echo -e "${GREEN}║   4. Wait ~10 minutes for setup to complete               ║${NC}"
echo -e "${GREEN}║   5. Login: pi / raspberry (CHANGE THIS!)                 ║${NC}"
echo -e "${GREEN}║   6. Run: rf-arsenal                                      ║${NC}"
echo -e "${GREEN}║                                                           ║${NC}"
echo -e "${GREEN}║   For RasPad touchscreen:                                 ║${NC}"
echo -e "${GREEN}║   Run: sudo /opt/rf-arsenal-os/install/setup_raspad.sh    ║${NC}"
echo -e "${GREEN}║                                                           ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
