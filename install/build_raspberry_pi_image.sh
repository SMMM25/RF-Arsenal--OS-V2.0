#!/bin/bash
# RF Arsenal OS - Raspberry Pi Image Builder
# Creates bootable USB/SD card image for Raspberry Pi 5/4/3
#
# Usage: sudo bash build_raspberry_pi_image.sh
#
# Copyright (c) 2024 RF-Arsenal-OS Project
# License: MIT

set -e  # Exit on error

# Configuration
IMAGE_NAME="RF-Arsenal-OS-v1.0-RaspberryPi5.img"
IMAGE_SIZE="12G"  # 12 GB uncompressed
BASE_IMAGE_URL="https://downloads.raspberrypi.org/raspios_lite_arm64/images/raspios_lite_arm64-2024-11-19/2024-11-19-raspios-bookworm-arm64-lite.img.xz"
BASE_IMAGE_FILE="raspios-base.img.xz"
WORK_DIR="/tmp/rf-arsenal-build"
MOUNT_BOOT="${WORK_DIR}/boot"
MOUNT_ROOT="${WORK_DIR}/root"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                                                           ║${NC}"
    echo -e "${BLUE}║   🛠️  RF ARSENAL OS - RASPBERRY PI IMAGE BUILDER  🛠️    ║${NC}"
    echo -e "${BLUE}║                                                           ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}==>${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️  ${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

check_requirements() {
    print_step "Checking requirements..."
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root"
        echo "  Run: sudo bash build_raspberry_pi_image.sh"
        exit 1
    fi
    
    # Check required tools
    local required_tools=("wget" "xz" "parted" "mkfs.ext4" "mkfs.vfat" "losetup" "git")
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            print_error "Required tool not found: $tool"
            echo "  Install: sudo apt-get install $tool"
            exit 1
        fi
    done
    
    # Check available disk space
    local available_space=$(df /tmp | tail -1 | awk '{print $4}')
    local required_space=$((15 * 1024 * 1024))  # 15 GB in KB
    
    if [[ $available_space -lt $required_space ]]; then
        print_error "Insufficient disk space in /tmp"
        echo "  Required: 15 GB, Available: $((available_space / 1024 / 1024)) GB"
        exit 1
    fi
    
    echo "  ✅ All requirements met"
}

download_base_image() {
    print_step "Downloading Raspberry Pi OS base image..."
    
    cd "$WORK_DIR"
    
    if [[ -f "$BASE_IMAGE_FILE" ]]; then
        echo "  ℹ️  Base image already downloaded"
    else
        wget -O "$BASE_IMAGE_FILE" "$BASE_IMAGE_URL"
        echo "  ✅ Downloaded base image"
    fi
    
    # Extract base image
    print_step "Extracting base image..."
    xz -d -k "$BASE_IMAGE_FILE"
    
    # Rename to our image name
    mv *.img "$IMAGE_NAME"
    
    echo "  ✅ Base image ready"
}

resize_image() {
    print_step "Resizing image to ${IMAGE_SIZE}..."
    
    # Add space to image
    truncate -s "$IMAGE_SIZE" "$WORK_DIR/$IMAGE_NAME"
    
    # Resize partition
    parted "$WORK_DIR/$IMAGE_NAME" resizepart 2 100%
    
    echo "  ✅ Image resized"
}

mount_image() {
    print_step "Mounting image partitions..."
    
    # Find loop device
    local loop_device=$(losetup -f)
    losetup -P "$loop_device" "$WORK_DIR/$IMAGE_NAME"
    
    # Mount boot partition
    mkdir -p "$MOUNT_BOOT"
    mount "${loop_device}p1" "$MOUNT_BOOT"
    
    # Resize root filesystem and mount
    e2fsck -f "${loop_device}p2"
    resize2fs "${loop_device}p2"
    
    mkdir -p "$MOUNT_ROOT"
    mount "${loop_device}p2" "$MOUNT_ROOT"
    
    echo "  ✅ Image mounted at $MOUNT_ROOT"
}

install_rf_arsenal() {
    print_step "Installing RF Arsenal OS..."
    
    # Clone RF Arsenal OS repository
    cd "$MOUNT_ROOT/home/pi"
    
    if [[ -d "RF-Arsenal-OS" ]]; then
        rm -rf RF-Arsenal-OS
    fi
    
    git clone https://github.com/SMMM25/RF-Arsenal-OS.git
    chown -R 1000:1000 RF-Arsenal-OS
    
    echo "  ✅ RF Arsenal OS cloned"
}

install_dependencies() {
    print_step "Installing system dependencies..."
    
    # Copy qemu-arm-static for chroot
    cp /usr/bin/qemu-arm-static "$MOUNT_ROOT/usr/bin/" 2>/dev/null || true
    
    # Chroot and install packages
    chroot "$MOUNT_ROOT" /bin/bash <<'EOF'
# Update package list
apt-get update

# Install BladeRF
apt-get install -y libbladerf-dev bladerf

# Install Python dependencies
apt-get install -y python3-pip python3-dev

# Install RF Arsenal OS Python packages
cd /home/pi/RF-Arsenal-OS
pip3 install -r install/requirements.txt --break-system-packages

# Install Tor (for anonymous updates)
apt-get install -y tor

# Install I2P (optional)
# apt-get install -y i2p

# Install additional tools
apt-get install -y git vim htop

# Clean up
apt-get clean
rm -rf /var/lib/apt/lists/*
EOF
    
    echo "  ✅ Dependencies installed"
}

configure_first_boot() {
    print_step "Configuring first boot setup..."
    
    # Create first boot service
    cat > "$MOUNT_ROOT/etc/systemd/system/rf-arsenal-first-boot.service" <<EOF
[Unit]
Description=RF Arsenal OS First Boot Setup
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/bin/python3 /home/pi/RF-Arsenal-OS/install/first_boot_wizard.py
RemainAfterExit=yes
StandardOutput=journal

[Install]
WantedBy=multi-user.target
EOF
    
    # Enable first boot service
    chroot "$MOUNT_ROOT" systemctl enable rf-arsenal-first-boot.service
    
    echo "  ✅ First boot configured"
}

configure_boot() {
    print_step "Configuring boot parameters..."
    
    # Add boot config for USB 3.0 optimization
    cat >> "$MOUNT_BOOT/config.txt" <<EOF

# RF Arsenal OS Optimizations
# USB 3.0 settings for BladeRF
dwc2.usb_max_packet_size=1024
dwc2.usb_burst_size=1024
max_usb_current=1
usb_max_current_enable=1

# GPU memory (will be adjusted by first boot wizard)
gpu_mem=128

# Enable UART for debugging
enable_uart=1
EOF
    
    echo "  ✅ Boot configuration updated"
}

cleanup_and_unmount() {
    print_step "Cleaning up..."
    
    # Sync filesystems
    sync
    
    # Unmount partitions
    umount "$MOUNT_BOOT" 2>/dev/null || true
    umount "$MOUNT_ROOT" 2>/dev/null || true
    
    # Detach loop device
    losetup -d /dev/loop* 2>/dev/null || true
    
    # Remove mount points
    rm -rf "$MOUNT_BOOT" "$MOUNT_ROOT"
    
    echo "  ✅ Cleanup complete"
}

compress_image() {
    print_step "Compressing image (this may take 10-20 minutes)..."
    
    cd "$WORK_DIR"
    
    # Compress with xz (high compression)
    xz -9 -T 0 -v "$IMAGE_NAME"
    
    local compressed_size=$(du -h "${IMAGE_NAME}.xz" | cut -f1)
    
    echo "  ✅ Image compressed"
    echo "     Compressed size: $compressed_size"
}

generate_checksums() {
    print_step "Generating checksums..."
    
    cd "$WORK_DIR"
    
    # Generate SHA-256 checksum
    sha256sum "${IMAGE_NAME}.xz" > "${IMAGE_NAME}.xz.sha256"
    
    echo "  ✅ Checksum generated"
}

print_completion() {
    local final_image="$WORK_DIR/${IMAGE_NAME}.xz"
    local image_size=$(du -h "$final_image" | cut -f1)
    
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                           ║${NC}"
    echo -e "${GREEN}║              🎉 IMAGE BUILD COMPLETE! 🎉                 ║${NC}"
    echo -e "${GREEN}║                                                           ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "📦 Image location: $final_image"
    echo "📏 Image size: $image_size"
    echo ""
    echo "🚀 Next steps:"
    echo "  1. Flash image to USB/SD card:"
    echo "     • Download Etcher: https://etcher.balena.io/"
    echo "     • Select: $final_image"
    echo "     • Flash to 32GB+ USB drive or SD card"
    echo ""
    echo "  2. Boot Raspberry Pi 5:"
    echo "     • Insert USB/SD card"
    echo "     • Connect BladeRF to USB 3.0 port"
    echo "     • Power on"
    echo "     • First boot wizard will start automatically"
    echo ""
    echo "  3. Distribute image:"
    echo "     • Upload to GitHub Releases"
    echo "     • Include .sha256 checksum file"
    echo ""
}

# Main execution
main() {
    print_header
    
    # Create work directory
    mkdir -p "$WORK_DIR"
    
    # Build process
    check_requirements
    download_base_image
    resize_image
    mount_image
    install_rf_arsenal
    install_dependencies
    configure_first_boot
    configure_boot
    cleanup_and_unmount
    compress_image
    generate_checksums
    print_completion
}

# Run main function
main "$@"
