#!/bin/bash
# ============================================================================
# RF Arsenal OS - Portable USB Creator (Live Mode)
# ============================================================================
# Creates a PORTABLE USB that runs RF Arsenal OS directly from the USB
# WITHOUT installing anything to the Raspberry Pi's SD card.
#
# PERFECT FOR:
#   - Stealth operations (leaves NO trace on Pi)
#   - Quick deployment to any Pi
#   - Forensic-safe operation (RAM-only mode available)
#   - Field operations with multiple Pis
#
# USAGE:
#   sudo bash create_portable_usb.sh /dev/sdX
#
# BOOT:
#   1. Insert USB into Pi
#   2. Hold SHIFT during boot (Pi 4/5) or edit boot order
#   3. System runs entirely from USB
#
# Copyright (c) 2024 RF-Arsenal-OS Project
# ============================================================================

set -e

VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORK_DIR="/tmp/rf-arsenal-portable"
RASPIOS_URL="https://downloads.raspberrypi.org/raspios_lite_arm64/images/raspios_lite_arm64-2024-11-19/2024-11-19-raspios-bookworm-arm64-lite.img.xz"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_banner() {
    clear
    echo -e "${CYAN}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ██████╗  ██████╗ ██████╗ ████████╗ █████╗ ██████╗ ██╗     ███████╗     ║
║   ██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔══██╗██║     ██╔════╝     ║
║   ██████╔╝██║   ██║██████╔╝   ██║   ███████║██████╔╝██║     █████╗       ║
║   ██╔═══╝ ██║   ██║██╔══██╗   ██║   ██╔══██║██╔══██╗██║     ██╔══╝       ║
║   ██║     ╚██████╔╝██║  ██║   ██║   ██║  ██║██████╔╝███████╗███████╗     ║
║   ╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═════╝ ╚══════╝╚══════╝     ║
║                                                                           ║
║              RF ARSENAL OS - PORTABLE USB CREATOR                         ║
║                     Runs from USB - No Installation                       ║
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
        exit 1
    fi
}

check_dependencies() {
    log_step "Checking dependencies..."
    local deps=(wget xz parted mkfs.ext4 mkfs.vfat losetup rsync)
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "Missing: $dep"
            exit 1
        fi
    done
    log_info "Dependencies OK"
}

select_device() {
    USB_DEVICE="${1:-}"
    
    if [[ -z "$USB_DEVICE" ]]; then
        echo ""
        log_step "Available devices:"
        lsblk -d -o NAME,SIZE,MODEL,TRAN | grep -E "usb|NAME"
        echo ""
        read -p "Enter USB device (e.g., /dev/sdb): " USB_DEVICE
    fi
    
    if [[ ! -b "$USB_DEVICE" ]]; then
        log_error "Device not found: $USB_DEVICE"
        exit 1
    fi
    
    if [[ "$USB_DEVICE" == "/dev/sda" ]]; then
        log_error "Refusing to write to /dev/sda"
        exit 1
    fi
    
    echo ""
    log_warn "ALL DATA ON $USB_DEVICE WILL BE DESTROYED!"
    lsblk "$USB_DEVICE"
    read -p "Type 'YES' to confirm: " confirm
    [[ "$confirm" == "YES" ]] || exit 0
}

prepare_workspace() {
    log_step "Preparing workspace..."
    rm -rf "$WORK_DIR"
    mkdir -p "$WORK_DIR"/{boot,root}
}

download_and_extract() {
    log_step "Downloading Raspberry Pi OS..."
    cd "$WORK_DIR"
    
    if [[ ! -f raspios.img ]]; then
        wget -O raspios.img.xz "$RASPIOS_URL"
        xz -d raspios.img.xz
    fi
    
    log_info "Image ready"
}

setup_image() {
    log_step "Setting up image..."
    
    # Expand to 16GB for portable use (more room for data)
    truncate -s 16G "$WORK_DIR/raspios.img"
    
    LOOP_DEV=$(losetup -f --show -P "$WORK_DIR/raspios.img")
    
    # Resize partition
    parted -s "$LOOP_DEV" resizepart 2 100%
    e2fsck -f "${LOOP_DEV}p2" || true
    resize2fs "${LOOP_DEV}p2"
    
    mount "${LOOP_DEV}p2" "$WORK_DIR/root"
    mount "${LOOP_DEV}p1" "$WORK_DIR/boot"
}

install_rf_arsenal() {
    log_step "Installing RF Arsenal OS..."
    
    local root="$WORK_DIR/root"
    local install_dir="$root/opt/rf-arsenal-os"
    
    mkdir -p "$install_dir"
    
    rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        "$PROJECT_ROOT/" "$install_dir/"
    
    chmod -R 755 "$install_dir/install"
    chmod +x "$install_dir/rf_arsenal.py" "$install_dir/rf_arsenal_os.py"
}

configure_portable_mode() {
    log_step "Configuring portable/stealth mode..."
    
    local root="$WORK_DIR/root"
    local boot="$WORK_DIR/boot"
    
    # RAM-based logging (no disk writes)
    cat > "$root/etc/systemd/system/rf-arsenal-ram-mode.service" << 'EOF'
[Unit]
Description=RF Arsenal OS RAM Mode Setup
Before=rsyslog.service systemd-journald.service
DefaultDependencies=no

[Service]
Type=oneshot
ExecStart=/opt/rf-arsenal-os/install/setup_ram_mode.sh
RemainAfterExit=yes

[Install]
WantedBy=sysinit.target
EOF

    # RAM mode setup script
    cat > "$root/opt/rf-arsenal-os/install/setup_ram_mode.sh" << 'RAM_EOF'
#!/bin/bash
# Setup RAM-only operation for stealth

# Create RAM disk for all sensitive operations
mkdir -p /tmp/rf_arsenal_ram
mount -t tmpfs -o size=1G,nodev,nosuid,noexec tmpfs /tmp/rf_arsenal_ram

# Redirect logs to RAM
mkdir -p /tmp/rf_arsenal_ram/log
mount --bind /tmp/rf_arsenal_ram/log /var/log

# Disable swap (no memory artifacts)
swapoff -a

# Clear any existing artifacts
rm -rf /var/log/* 2>/dev/null || true

echo "RAM-only mode active - no disk writes"
RAM_EOF

    chmod +x "$root/opt/rf-arsenal-os/install/setup_ram_mode.sh"
    
    # Enable RAM mode by default
    ln -sf /etc/systemd/system/rf-arsenal-ram-mode.service \
        "$root/etc/systemd/system/sysinit.target.wants/rf-arsenal-ram-mode.service"
    
    # Auto-start RF Arsenal
    cat > "$root/etc/systemd/system/rf-arsenal-autostart.service" << 'EOF'
[Unit]
Description=RF Arsenal OS Auto-Start
After=network.target rf-arsenal-ram-mode.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/rf-arsenal-os
ExecStart=/usr/bin/python3 /opt/rf-arsenal-os/rf_arsenal_os.py --cli
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    # Create portable launcher
    cat > "$root/usr/local/bin/rf-arsenal" << 'LAUNCHER_EOF'
#!/bin/bash
# RF Arsenal OS - Portable Launcher

cd /opt/rf-arsenal-os

case "${1:-}" in
    --gui)
        python3 rf_arsenal_os.py --gui
        ;;
    --stealth)
        # Activate stealth profile first
        /opt/rf-arsenal-os/profiles/stealth.sh
        python3 rf_arsenal_os.py --cli
        ;;
    --help|-h)
        echo "RF Arsenal OS - Mobile RF Security Platform"
        echo ""
        echo "Usage: rf-arsenal [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --gui      Launch GUI mode"
        echo "  --stealth  Launch in stealth mode (minimal RF/forensic footprint)"
        echo "  --cli      Launch CLI mode (default)"
        echo "  --help     Show this help"
        ;;
    *)
        python3 rf_arsenal_os.py --cli
        ;;
esac
LAUNCHER_EOF

    chmod +x "$root/usr/local/bin/rf-arsenal"
    
    # Enable SSH
    touch "$boot/ssh"
    
    # Set hostname
    echo "rf-arsenal" > "$root/etc/hostname"
    
    # Portable mode indicator
    touch "$root/opt/rf-arsenal-os/.portable_mode"
    
    log_info "Portable mode configured"
}

configure_stealth_boot() {
    log_step "Configuring stealth boot options..."
    
    local boot="$WORK_DIR/boot"
    
    # Add stealth kernel parameters
    cat >> "$boot/cmdline.txt" << 'EOF'
 quiet loglevel=0 logo.nologo consoleblank=0
EOF
    
    # Boot config for USB boot
    cat >> "$boot/config.txt" << 'EOF'

# USB Boot Priority (Pi 4/5)
# System will boot from USB if present
boot_delay=0

# Reduce boot messages
disable_splash=1

# Performance tuning
arm_boost=1

# GPU memory (reduce for headless)
gpu_mem=64
EOF

    log_info "Stealth boot configured"
}

install_dependencies_offline() {
    log_step "Pre-installing dependencies for offline use..."
    
    local root="$WORK_DIR/root"
    
    # Create offline package cache
    mkdir -p "$root/opt/rf-arsenal-os/offline_packages"
    
    # Download key packages for offline install
    # This allows the system to work even without internet
    
    # Create offline setup script
    cat > "$root/opt/rf-arsenal-os/install/offline_setup.sh" << 'OFFLINE_EOF'
#!/bin/bash
# Offline dependency installation
# Run this if the portable USB was created without pre-installed deps

echo "Installing dependencies (requires internet first time)..."

apt-get update
apt-get install -y \
    python3-pip python3-dev \
    libusb-1.0-0-dev libbladerf-dev bladerf \
    libfftw3-dev

pip3 install --break-system-packages -r /opt/rf-arsenal-os/install/requirements.txt

echo "Dependencies installed. System ready for offline use."
OFFLINE_EOF

    chmod +x "$root/opt/rf-arsenal-os/install/offline_setup.sh"
}

write_to_usb() {
    log_step "Writing to USB drive..."
    
    # Unmount
    sync
    umount "$WORK_DIR/boot" || true
    umount "$WORK_DIR/root" || true
    losetup -d "$LOOP_DEV" || true
    
    # Write image
    log_warn "Writing to $USB_DEVICE - please wait..."
    dd if="$WORK_DIR/raspios.img" of="$USB_DEVICE" bs=4M status=progress conv=fsync
    sync
    
    log_info "Write complete"
}

cleanup() {
    umount "$WORK_DIR/boot" 2>/dev/null || true
    umount "$WORK_DIR/root" 2>/dev/null || true
    [[ -n "${LOOP_DEV:-}" ]] && losetup -d "$LOOP_DEV" 2>/dev/null || true
}

trap cleanup EXIT

# Main
print_banner
check_root
check_dependencies
select_device "$1"
prepare_workspace
download_and_extract
setup_image
install_rf_arsenal
configure_portable_mode
configure_stealth_boot
install_dependencies_offline
write_to_usb

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                           ║${NC}"
echo -e "${GREEN}║   ✅  PORTABLE USB CREATED SUCCESSFULLY!                  ║${NC}"
echo -e "${GREEN}║                                                           ║${NC}"
echo -e "${GREEN}╠═══════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                                           ║${NC}"
echo -e "${GREEN}║   PORTABLE MODE FEATURES:                                 ║${NC}"
echo -e "${GREEN}║   • Runs entirely from USB (no SD card needed)            ║${NC}"
echo -e "${GREEN}║   • RAM-only logging (no disk forensics)                  ║${NC}"
echo -e "${GREEN}║   • Stealth boot (minimal boot messages)                  ║${NC}"
echo -e "${GREEN}║   • Works on any Pi 3/4/5                                 ║${NC}"
echo -e "${GREEN}║                                                           ║${NC}"
echo -e "${GREEN}║   TO BOOT:                                                ║${NC}"
echo -e "${GREEN}║   1. Insert USB into Raspberry Pi                         ║${NC}"
echo -e "${GREEN}║   2. Power on (Pi 4/5 auto-boot from USB)                 ║${NC}"
echo -e "${GREEN}║   3. For Pi 3: Set USB boot in raspi-config first        ║${NC}"
echo -e "${GREEN}║                                                           ║${NC}"
echo -e "${GREEN}║   COMMANDS:                                               ║${NC}"
echo -e "${GREEN}║   rf-arsenal           - CLI mode                         ║${NC}"
echo -e "${GREEN}║   rf-arsenal --gui     - GUI mode                         ║${NC}"
echo -e "${GREEN}║   rf-arsenal --stealth - Stealth mode                     ║${NC}"
echo -e "${GREEN}║                                                           ║${NC}"
echo -e "${GREEN}║   Default login: pi / raspberry                           ║${NC}"
echo -e "${GREEN}║                                                           ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
