#!/bin/bash
#
# RF Arsenal OS - YateBTS Complete Installation Script
# Installs Yate base engine and YateBTS GSM BTS module
# Supports BladeRF 2.0 micro xA9 with automatic configuration
#
# Usage: sudo bash install_yatebts.sh [--bladerf|--hackrf|--usrp]
#
# REAL-WORLD FUNCTIONAL ONLY - No simulation modes
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EXTERNAL_DIR="$PROJECT_ROOT/external"
YATE_DIR="$EXTERNAL_DIR/yate"
YATEBTS_DIR="$EXTERNAL_DIR/yatebts"

# Default SDR - BladeRF for RF Arsenal OS
SDR_TYPE="${1:-bladerf}"

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run as root (sudo)"
        exit 1
    fi
}

# Check for SDR hardware
check_sdr_hardware() {
    log_header "Checking SDR Hardware"
    
    case "$SDR_TYPE" in
        bladerf|--bladerf)
            SDR_TYPE="bladerf"
            log_info "Looking for BladeRF device..."
            if lsusb | grep -i "2cf0:5246\|Nuand\|bladeRF" > /dev/null 2>&1; then
                log_info "✓ BladeRF detected via USB"
            else
                log_warn "BladeRF not detected - installation will proceed but operation requires hardware"
            fi
            ;;
        hackrf|--hackrf)
            SDR_TYPE="hackrf"
            log_info "Looking for HackRF device..."
            if lsusb | grep -i "1d50:6089\|Great Scott\|HackRF" > /dev/null 2>&1; then
                log_info "✓ HackRF detected via USB"
            else
                log_warn "HackRF not detected - installation will proceed but operation requires hardware"
            fi
            ;;
        usrp|--usrp)
            SDR_TYPE="usrp"
            log_info "Looking for USRP device..."
            if lsusb | grep -i "Ettus\|USRP\|2500:0020" > /dev/null 2>&1; then
                log_info "✓ USRP detected via USB"
            else
                log_warn "USRP not detected - installation will proceed but operation requires hardware"
            fi
            ;;
        *)
            log_error "Unknown SDR type: $SDR_TYPE"
            log_info "Supported: bladerf, hackrf, usrp"
            exit 1
            ;;
    esac
}

# Install system dependencies
install_dependencies() {
    log_header "Installing System Dependencies"
    
    # Update package list
    apt-get update
    
    # Build tools
    apt-get install -y \
        build-essential \
        autoconf \
        automake \
        libtool \
        pkg-config \
        git \
        subversion \
        cmake
    
    # Required libraries
    apt-get install -y \
        libusb-1.0-0-dev \
        libgsm1-dev \
        libspeex-dev \
        libssl-dev \
        libpq-dev \
        libmysqlclient-dev \
        libasound2-dev \
        libspandsp-dev \
        libsqlite3-dev
    
    # SDR-specific dependencies
    case "$SDR_TYPE" in
        bladerf)
            apt-get install -y libbladerf-dev bladerf || true
            ;;
        hackrf)
            apt-get install -y libhackrf-dev hackrf || true
            ;;
        usrp)
            apt-get install -y libuhd-dev uhd-host || true
            ;;
    esac
    
    log_info "✓ Dependencies installed"
}

# Build and install Yate base engine
install_yate() {
    log_header "Building Yate Telephony Engine"
    
    if [ ! -d "$YATE_DIR" ]; then
        log_error "Yate source not found at $YATE_DIR"
        log_info "Cloning from official repository..."
        cd "$EXTERNAL_DIR"
        git clone --depth 1 https://github.com/yatevoip/yate.git yate
    fi
    
    cd "$YATE_DIR"
    
    log_info "Running autogen.sh..."
    ./autogen.sh
    
    log_info "Configuring Yate..."
    ./configure --prefix=/usr/local
    
    log_info "Building Yate (this may take several minutes)..."
    make -j$(nproc)
    
    log_info "Installing Yate..."
    make install-noapi
    
    # Update library path
    echo "/usr/local/lib" > /etc/ld.so.conf.d/yate.conf
    ldconfig
    
    log_info "✓ Yate installed successfully"
}

# Build and install YateBTS
install_yatebts() {
    log_header "Building YateBTS GSM Module"
    
    if [ ! -d "$YATEBTS_DIR" ]; then
        log_error "YateBTS source not found at $YATEBTS_DIR"
        log_info "Cloning from official repository..."
        cd "$EXTERNAL_DIR"
        git clone --depth 1 https://github.com/yatevoip/yatebts.git yatebts
    fi
    
    cd "$YATEBTS_DIR"
    
    log_info "Running autogen.sh..."
    ./autogen.sh
    
    log_info "Configuring YateBTS..."
    ./configure --prefix=/usr/local
    
    log_info "Building YateBTS..."
    make -j$(nproc)
    
    log_info "Installing YateBTS..."
    make install
    
    log_info "✓ YateBTS installed successfully"
}

# Configure YateBTS for RF Arsenal OS
configure_yatebts() {
    log_header "Configuring YateBTS for RF Arsenal OS"
    
    YATE_CONFIG_DIR="/usr/local/etc/yate"
    mkdir -p "$YATE_CONFIG_DIR"
    
    # Determine transceiver path based on SDR
    case "$SDR_TYPE" in
        bladerf)
            TRANSCEIVER_PATH="/usr/local/lib/yate/bts/transceiver-bladerf"
            ;;
        hackrf)
            TRANSCEIVER_PATH="/usr/local/lib/yate/bts/transceiver"
            ;;
        usrp)
            TRANSCEIVER_PATH="/usr/local/lib/yate/bts/transceiver-uhd"
            ;;
    esac
    
    # Create YateBTS configuration
    cat > "$YATE_CONFIG_DIR/ybts.conf" << EOF
; RF Arsenal OS - YateBTS Configuration
; SDR Type: $SDR_TYPE
; Generated: $(date)
;
; WARNING: GSM transmission requires proper licensing and authorization
; Use only in authorized testing environments

[ybts]
; Radio band and ARFCN - Configure for your region
; GSM850: ARFCN 128-251
; GSM900: ARFCN 1-124, 975-1023
; DCS1800: ARFCN 512-885
; PCS1900: ARFCN 512-810

Radio.Band=GSM850
Radio.C0=128

; Network identity - TEST VALUES ONLY
Identity.MCC=001
Identity.MNC=01
Identity.ShortName=RF-Arsenal

; Transmission power (dBm) - Adjust based on testing requirements
Radio.PowerManager.MaxAttenDB=30
Radio.PowerManager.MinAttenDB=0

[transceiver]
; Path to SDR transceiver module
Path=$TRANSCEIVER_PATH

; Sample rate and timing
SampleRate=4

[gprs]
; GPRS data services
Enable=yes
RAC=0

[control]
; Control interface
VEA=yes
LUR.OpenRegistration=.*

[tapping]
; Call and SMS interception (RF Arsenal feature)
TargetPresentRecording=yes
RecordLocal=/var/spool/yate/recordings/
EOF

    # Create Yate main configuration
    cat > "$YATE_CONFIG_DIR/yate.conf" << EOF
; RF Arsenal OS - Yate Main Configuration

[general]
modload=yes
modpath=/usr/local/lib/yate/:/usr/local/lib/yate/server/:/usr/local/lib/yate/client/
EOF

    # Create javascript configuration for NiPC mode
    cat > "$YATE_CONFIG_DIR/javascript.conf" << EOF
; JavaScript routing configuration

[general]
routing=welcome.js
scripts_dir=/usr/local/share/yate/scripts/

[scripts]
nipc=nipc.js
EOF

    # Create recordings directory
    mkdir -p /var/spool/yate/recordings
    chmod 755 /var/spool/yate/recordings
    
    log_info "✓ YateBTS configured for $SDR_TYPE"
    log_info "Configuration files in: $YATE_CONFIG_DIR"
}

# Create systemd service
create_service() {
    log_header "Creating Systemd Service"
    
    cat > /etc/systemd/system/yatebts.service << EOF
[Unit]
Description=YateBTS GSM Base Station
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/yate -vvv -Df
ExecStop=/usr/bin/kill -TERM \$MAINPID
Restart=on-failure
RestartSec=5
Environment=LD_LIBRARY_PATH=/usr/local/lib

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    
    log_info "✓ Systemd service created"
    log_info "Start with: sudo systemctl start yatebts"
    log_info "Enable at boot: sudo systemctl enable yatebts"
}

# Verify installation
verify_installation() {
    log_header "Verifying Installation"
    
    local PASS=0
    local FAIL=0
    
    # Check Yate binary
    if [ -x /usr/local/bin/yate ]; then
        log_info "✓ Yate binary installed"
        ((PASS++))
    else
        log_error "✗ Yate binary not found"
        ((FAIL++))
    fi
    
    # Check YateBTS module
    if [ -d /usr/local/lib/yate/bts ]; then
        log_info "✓ YateBTS module installed"
        ((PASS++))
    else
        log_error "✗ YateBTS module not found"
        ((FAIL++))
    fi
    
    # Check configuration
    if [ -f /usr/local/etc/yate/ybts.conf ]; then
        log_info "✓ YateBTS configuration created"
        ((PASS++))
    else
        log_error "✗ YateBTS configuration not found"
        ((FAIL++))
    fi
    
    # Check library
    if ldconfig -p | grep libyate > /dev/null; then
        log_info "✓ Yate library in system path"
        ((PASS++))
    else
        log_error "✗ Yate library not in system path"
        ((FAIL++))
    fi
    
    # Check Yate version
    if /usr/local/bin/yate --version 2>/dev/null; then
        ((PASS++))
    fi
    
    echo ""
    log_info "Verification: $PASS passed, $FAIL failed"
    
    if [ $FAIL -gt 0 ]; then
        log_error "Installation verification failed"
        return 1
    fi
    
    return 0
}

# Print usage instructions
print_usage() {
    log_header "RF Arsenal OS - YateBTS Ready"
    
    echo "YateBTS has been installed and configured for $SDR_TYPE."
    echo ""
    echo "Quick Start:"
    echo "  1. Connect your $SDR_TYPE device"
    echo "  2. Edit /usr/local/etc/yate/ybts.conf for your region"
    echo "  3. Start YateBTS: sudo systemctl start yatebts"
    echo "  4. Or run directly: sudo yate -vvv -Df"
    echo ""
    echo "RF Arsenal Integration:"
    echo "  from modules.cellular.yatebts import YateBTSController"
    echo "  bts = YateBTSController()"
    echo "  bts.start_bts(BTSMode.IMSI_CATCHER)"
    echo ""
    echo "WARNING: GSM transmission requires proper licensing."
    echo "Use only in authorized testing environments."
    echo ""
    echo "Configuration: /usr/local/etc/yate/ybts.conf"
    echo "Logs: journalctl -u yatebts -f"
    echo ""
}

# Main installation flow
main() {
    log_header "RF Arsenal OS - YateBTS Installation"
    
    echo "SDR Type: $SDR_TYPE"
    echo "Yate Source: $YATE_DIR"
    echo "YateBTS Source: $YATEBTS_DIR"
    echo ""
    
    check_root
    check_sdr_hardware
    install_dependencies
    install_yate
    install_yatebts
    configure_yatebts
    create_service
    
    if verify_installation; then
        print_usage
        log_info "Installation completed successfully!"
    else
        log_error "Installation completed with errors"
        exit 1
    fi
}

# Handle command line arguments
case "${1:-}" in
    -h|--help)
        echo "RF Arsenal OS - YateBTS Installation Script"
        echo ""
        echo "Usage: sudo bash $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --bladerf    Configure for BladeRF SDR (default)"
        echo "  --hackrf     Configure for HackRF SDR"
        echo "  --usrp       Configure for USRP SDR"
        echo "  -h, --help   Show this help message"
        echo ""
        exit 0
        ;;
    *)
        main
        ;;
esac
