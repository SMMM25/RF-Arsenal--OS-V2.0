#!/bin/bash
###############################################################################
# RF Arsenal OS - srsRAN Installation Script
# 
# Installs srsRAN 4G/5G software suite with BladeRF support
# Tested on: Ubuntu 22.04 LTS, Raspberry Pi OS (64-bit)
#
# Usage:
#   sudo ./install_srsran.sh [options]
#
# Options:
#   --with-5g       Include 5G NR support (srsGNB)
#   --with-zmq      Enable ZeroMQ for virtual RF
#   --prefix PATH   Installation prefix (default: /opt/srsran)
#   --skip-deps     Skip dependency installation
#   --clean         Clean build before compiling
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SRSRAN_VERSION="release_23_11"
SRSRAN_REPO="https://github.com/srsran/srsRAN_4G.git"
SRSRAN_5G_REPO="https://github.com/srsran/srsRAN_Project.git"
INSTALL_PREFIX="/opt/srsran"
BUILD_DIR="/tmp/srsran_build"
WITH_5G=false
WITH_ZMQ=false
SKIP_DEPS=false
CLEAN_BUILD=false
NUM_CORES=$(nproc)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --with-5g)
            WITH_5G=true
            shift
            ;;
        --with-zmq)
            WITH_ZMQ=true
            shift
            ;;
        --prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--with-5g] [--with-zmq] [--prefix PATH] [--skip-deps] [--clean]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_header() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║          RF Arsenal OS - srsRAN Installation                 ║"
    echo "║                                                              ║"
    echo "║  Installing: srsRAN 4G $([ "$WITH_5G" = true ] && echo "+ 5G NR")                           ║"
    echo "║  Prefix: $INSTALL_PREFIX                                     ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
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

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (sudo)"
        exit 1
    fi
}

detect_platform() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$ID
        VER=$VERSION_ID
    else
        log_error "Cannot detect OS"
        exit 1
    fi
    
    log_info "Detected: $OS $VER"
    
    # Check architecture
    ARCH=$(uname -m)
    if [[ "$ARCH" == "aarch64" ]]; then
        log_info "ARM64 architecture detected (Raspberry Pi)"
        IS_ARM=true
    else
        IS_ARM=false
    fi
}

install_dependencies() {
    if [[ "$SKIP_DEPS" = true ]]; then
        log_warn "Skipping dependency installation"
        return
    fi
    
    log_info "Installing dependencies..."
    
    apt-get update
    
    # Core build tools
    apt-get install -y \
        build-essential \
        cmake \
        git \
        pkg-config \
        ninja-build
    
    # srsRAN dependencies
    apt-get install -y \
        libfftw3-dev \
        libmbedtls-dev \
        libboost-program-options-dev \
        libconfig++-dev \
        libsctp-dev \
        libyaml-cpp-dev \
        libgtest-dev
    
    # RF drivers
    apt-get install -y \
        libbladerf-dev \
        libuhd-dev \
        libsoapysdr-dev
    
    # ZeroMQ (optional)
    if [[ "$WITH_ZMQ" = true ]]; then
        apt-get install -y \
            libzmq3-dev \
            libczmq-dev
    fi
    
    # Additional tools
    apt-get install -y \
        libpcsclite-dev \
        pcscd \
        pcsc-tools
    
    # 5G specific dependencies
    if [[ "$WITH_5G" = true ]]; then
        apt-get install -y \
            nlohmann-json3-dev \
            googletest
        
        # Install additional deps for srsRAN Project (5G)
        apt-get install -y \
            libnuma-dev \
            libdpdk-dev 2>/dev/null || true
    fi
    
    log_info "Dependencies installed"
}

build_srsran_4g() {
    log_info "Building srsRAN 4G..."
    
    # Clone repository
    if [[ ! -d "$BUILD_DIR/srsRAN_4G" ]]; then
        git clone --depth 1 --branch $SRSRAN_VERSION $SRSRAN_REPO "$BUILD_DIR/srsRAN_4G"
    fi
    
    cd "$BUILD_DIR/srsRAN_4G"
    
    # Clean if requested
    if [[ "$CLEAN_BUILD" = true ]] && [[ -d build ]]; then
        rm -rf build
    fi
    
    mkdir -p build
    cd build
    
    # Configure
    CMAKE_ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release"
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_BLADERF=ON"
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_SOAPYSDR=ON"
    
    if [[ "$WITH_ZMQ" = true ]]; then
        CMAKE_ARGS="$CMAKE_ARGS -DENABLE_ZEROMQ=ON"
    fi
    
    cmake $CMAKE_ARGS ..
    
    # Build
    make -j$NUM_CORES
    
    # Install
    make install
    
    log_info "srsRAN 4G installed to $INSTALL_PREFIX"
}

build_srsran_5g() {
    if [[ "$WITH_5G" = false ]]; then
        return
    fi
    
    log_info "Building srsRAN Project (5G NR)..."
    
    # Clone repository
    if [[ ! -d "$BUILD_DIR/srsRAN_Project" ]]; then
        git clone --depth 1 $SRSRAN_5G_REPO "$BUILD_DIR/srsRAN_Project"
    fi
    
    cd "$BUILD_DIR/srsRAN_Project"
    
    # Clean if requested
    if [[ "$CLEAN_BUILD" = true ]] && [[ -d build ]]; then
        rm -rf build
    fi
    
    mkdir -p build
    cd build
    
    # Configure
    CMAKE_ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release"
    CMAKE_ARGS="$CMAKE_ARGS -GNinja"
    
    cmake $CMAKE_ARGS ..
    
    # Build
    ninja -j$NUM_CORES
    
    # Install
    ninja install
    
    log_info "srsRAN 5G installed to $INSTALL_PREFIX"
}

configure_system() {
    log_info "Configuring system..."
    
    # Create config directory
    mkdir -p /etc/srsran
    mkdir -p /var/log/srsran
    
    # Copy default configs
    if [[ -d "$INSTALL_PREFIX/share/srsran" ]]; then
        cp -r "$INSTALL_PREFIX/share/srsran/"* /etc/srsran/ 2>/dev/null || true
    fi
    
    # Set up PATH
    echo "export PATH=\$PATH:$INSTALL_PREFIX/bin" > /etc/profile.d/srsran.sh
    chmod +x /etc/profile.d/srsran.sh
    
    # Create symlinks
    for bin in srsenb srsue srsepc; do
        if [[ -f "$INSTALL_PREFIX/bin/$bin" ]]; then
            ln -sf "$INSTALL_PREFIX/bin/$bin" /usr/local/bin/
        fi
    done
    
    # Configure kernel parameters for real-time
    cat > /etc/sysctl.d/99-srsran.conf << 'EOF'
# srsRAN real-time optimizations
net.core.rmem_max=50000000
net.core.wmem_max=50000000
kernel.sched_rt_runtime_us=-1
EOF
    
    sysctl -p /etc/sysctl.d/99-srsran.conf 2>/dev/null || true
    
    # Set up real-time limits
    cat >> /etc/security/limits.conf << 'EOF'
# srsRAN real-time limits
*               soft    rtprio          99
*               hard    rtprio          99
*               soft    memlock         unlimited
*               hard    memlock         unlimited
EOF
    
    log_info "System configured"
}

create_default_configs() {
    log_info "Creating default configurations..."
    
    # Default eNB config
    cat > /etc/srsran/enb.conf << 'EOF'
#####################################################################
# RF Arsenal OS - srsENB Configuration
#####################################################################

[enb]
enb_id = 0x19B
mcc = 001
mnc = 01
mme_addr = 127.0.1.100
gtp_bind_addr = 127.0.1.1
s1c_bind_addr = 127.0.1.1
s1c_bind_port = 0
n_prb = 50
tm = 1
nof_ports = 1

[enb_files]
sib_config = /etc/srsran/sib.conf
rr_config = /etc/srsran/rr.conf
rb_config = /etc/srsran/rb.conf

[rf]
dl_earfcn = 3350
tx_gain = 50
rx_gain = 40
device_name = bladeRF
device_args = auto
time_adv_nsamples = auto

[pcap]
enable = false
filename = /tmp/enb.pcap
s1ap_enable = false
s1ap_filename = /tmp/enb_s1ap.pcap
mac_net_enable = false

[log]
all_level = info
all_hex_limit = 32
filename = /var/log/srsran/enb.log
file_max_size = -1

[scheduler]
policy = time_rr
policy_args = 2
max_aggr_level = -1

[expert]
metrics_period_secs = 1
metrics_csv_enable = false
pregenerate_signals = false
rrc_inactivity_timer = 30000
print_buffer_state = false
EOF

    # Default EPC config
    cat > /etc/srsran/epc.conf << 'EOF'
#####################################################################
# RF Arsenal OS - srsEPC Configuration
#####################################################################

[mme]
mme_code = 0x01
mme_group = 0x0001
tac = 0x0001
mcc = 001
mnc = 01
mme_bind_addr = 127.0.1.100
apn = internet
dns_addr = 8.8.8.8
encryption_algo = EEA0
integrity_algo = EIA2
paging_timer = 2

[hss]
db_file = /etc/srsran/user_db.csv

[spgw]
gtpu_bind_addr = 127.0.1.100
sgi_if_addr = 172.16.0.1
sgi_if_name = srs_spgw_sgi
max_paging_queue = 100

[pcap]
enable = false
filename = /tmp/epc.pcap

[log]
all_level = info
all_hex_limit = 32
filename = /var/log/srsran/epc.log
file_max_size = -1
EOF

    # Default UE config
    cat > /etc/srsran/ue.conf << 'EOF'
#####################################################################
# RF Arsenal OS - srsUE Configuration
#####################################################################

[rf]
freq_offset = 0
tx_gain = 40
rx_gain = 40
device_name = bladeRF
device_args = auto
nof_antennas = 1
time_adv_nsamples = auto

[rat.eutra]
dl_earfcn = 3350
nof_carriers = 1

[pcap]
enable = false
filename = /tmp/ue.pcap
nas_enable = false

[log]
all_level = info
all_hex_limit = 32
filename = /var/log/srsran/ue.log
file_max_size = -1

[usim]
mode = soft
algo = milenage
opc = 63bfa50ee6523365ff14c1f45f88737d
k = 00112233445566778899aabbccddeeff
imsi = 001010123456789
imei = 353490069873319

[nas]
apn = internet
apn_protocol = ipv4

[gw]
netns =
ip_devname = tun_srsue
ip_netmask = 255.255.255.0
EOF

    # Default user database
    cat > /etc/srsran/user_db.csv << 'EOF'
# user_db.csv - srsEPC subscriber database
# Format: Name,Auth,IMSI,Key,OP_Type,OP/OPc,AMF,SQN,QCI,IP_alloc
#
ue1,mil,001010123456789,00112233445566778899aabbccddeeff,opc,63bfa50ee6523365ff14c1f45f88737d,8000,000000000000,9,dynamic
EOF

    chmod 600 /etc/srsran/user_db.csv
    
    log_info "Default configurations created in /etc/srsran/"
}

verify_installation() {
    log_info "Verifying installation..."
    
    local errors=0
    
    # Check binaries
    for bin in srsenb srsue srsepc; do
        if command -v $bin &> /dev/null; then
            log_info "  ✓ $bin found"
        else
            log_error "  ✗ $bin not found"
            ((errors++))
        fi
    done
    
    # Check 5G binaries
    if [[ "$WITH_5G" = true ]]; then
        for bin in gnb srsue; do
            if [[ -f "$INSTALL_PREFIX/bin/$bin" ]]; then
                log_info "  ✓ $bin (5G) found"
            fi
        done
    fi
    
    # Check configs
    if [[ -f /etc/srsran/enb.conf ]]; then
        log_info "  ✓ Configuration files present"
    else
        log_warn "  ⚠ Configuration files may need setup"
    fi
    
    if [[ $errors -eq 0 ]]; then
        log_info "Installation verified successfully"
        return 0
    else
        log_error "Installation verification failed"
        return 1
    fi
}

print_summary() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           srsRAN Installation Complete!                      ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Installation Summary:"
    echo "  - srsRAN 4G: $INSTALL_PREFIX"
    [[ "$WITH_5G" = true ]] && echo "  - srsRAN 5G: $INSTALL_PREFIX"
    echo "  - Configs:   /etc/srsran/"
    echo "  - Logs:      /var/log/srsran/"
    echo ""
    echo "Quick Start:"
    echo "  1. Start EPC:  sudo srsepc /etc/srsran/epc.conf"
    echo "  2. Start eNB:  sudo srsenb /etc/srsran/enb.conf"
    echo "  3. Start UE:   sudo srsue /etc/srsran/ue.conf"
    echo ""
    echo "For RF Arsenal integration:"
    echo "  from core.external.srsran import create_srsran_controller"
    echo "  ctrl = create_srsran_controller()"
    echo "  ctrl.start_full_network()"
    echo ""
}

cleanup() {
    log_info "Cleaning up build files..."
    rm -rf "$BUILD_DIR"
}

# Main
main() {
    print_header
    check_root
    detect_platform
    
    mkdir -p "$BUILD_DIR"
    
    install_dependencies
    build_srsran_4g
    build_srsran_5g
    configure_system
    create_default_configs
    verify_installation
    
    # Cleanup unless there was an error
    if [[ $? -eq 0 ]]; then
        cleanup
    fi
    
    print_summary
}

main "$@"
