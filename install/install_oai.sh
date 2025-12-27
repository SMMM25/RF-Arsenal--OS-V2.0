#!/bin/bash
###############################################################################
# RF Arsenal OS - OpenAirInterface Installation Script
# 
# Installs OpenAirInterface 5G RAN and Core Network
# Tested on: Ubuntu 22.04 LTS
#
# Usage:
#   sudo ./install_oai.sh [options]
#
# Options:
#   --ran-only      Install only RAN components (gNB/UE)
#   --core-only     Install only Core Network (Docker)
#   --with-bladerf  Enable BladeRF support
#   --prefix PATH   Installation prefix (default: /opt/openairinterface5g)
#   --skip-deps     Skip dependency installation
#   --clean         Clean build before compiling
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
OAI_RAN_REPO="https://gitlab.eurecom.fr/oai/openairinterface5g.git"
OAI_CN_REPO="https://gitlab.eurecom.fr/oai/cn5g/oai-cn5g-fed.git"
OAI_RAN_BRANCH="develop"
INSTALL_PREFIX="/opt/openairinterface5g"
INSTALL_CN_PREFIX="/opt/oai-cn5g"
BUILD_DIR="/tmp/oai_build"
INSTALL_RAN=true
INSTALL_CORE=true
WITH_BLADERF=true
SKIP_DEPS=false
CLEAN_BUILD=false
NUM_CORES=$(nproc)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ran-only)
            INSTALL_CORE=false
            shift
            ;;
        --core-only)
            INSTALL_RAN=false
            shift
            ;;
        --with-bladerf)
            WITH_BLADERF=true
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
            echo "Usage: $0 [--ran-only] [--core-only] [--with-bladerf] [--prefix PATH] [--skip-deps] [--clean]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_header() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║      RF Arsenal OS - OpenAirInterface Installation           ║"
    echo "║                                                              ║"
    echo "║  Components: $([ "$INSTALL_RAN" = true ] && echo "RAN ")$([ "$INSTALL_CORE" = true ] && echo "Core")                                        ║"
    echo "║  Prefix: $INSTALL_PREFIX                              ║"
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
    
    if [[ "$OS" != "ubuntu" ]] || [[ "$VER" != "22.04" && "$VER" != "20.04" ]]; then
        log_warn "OAI is best supported on Ubuntu 20.04/22.04"
        log_warn "Proceeding anyway..."
    fi
    
    # Check architecture
    ARCH=$(uname -m)
    if [[ "$ARCH" != "x86_64" ]]; then
        log_warn "OAI is primarily tested on x86_64"
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
        git \
        build-essential \
        cmake \
        pkg-config \
        ninja-build \
        ccache
    
    # OAI RAN dependencies
    apt-get install -y \
        libfftw3-dev \
        liblapacke-dev \
        libatlas-base-dev \
        libblas-dev \
        libconfig-dev \
        libyaml-cpp-dev \
        libgtest-dev \
        libsctp-dev
    
    # Additional libraries
    apt-get install -y \
        libboost-all-dev \
        libtool \
        automake \
        autoconf \
        flex \
        bison
    
    # RF drivers
    if [[ "$WITH_BLADERF" = true ]]; then
        apt-get install -y libbladerf-dev
    fi
    
    apt-get install -y \
        libuhd-dev \
        uhd-host
    
    # Docker for Core Network
    if [[ "$INSTALL_CORE" = true ]]; then
        install_docker
    fi
    
    log_info "Dependencies installed"
}

install_docker() {
    if command -v docker &> /dev/null; then
        log_info "Docker already installed"
        return
    fi
    
    log_info "Installing Docker..."
    
    # Install Docker
    apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
        tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    apt-get update
    apt-get install -y \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-compose-plugin
    
    # Add current user to docker group
    usermod -aG docker $SUDO_USER 2>/dev/null || true
    
    # Install docker-compose standalone
    curl -SL "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-linux-x86_64" \
        -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    
    systemctl enable docker
    systemctl start docker
    
    log_info "Docker installed"
}

build_oai_ran() {
    if [[ "$INSTALL_RAN" = false ]]; then
        return
    fi
    
    log_info "Building OpenAirInterface RAN..."
    
    # Clone repository
    if [[ ! -d "$BUILD_DIR/openairinterface5g" ]]; then
        git clone --depth 1 --branch $OAI_RAN_BRANCH $OAI_RAN_REPO "$BUILD_DIR/openairinterface5g"
    fi
    
    cd "$BUILD_DIR/openairinterface5g"
    
    # Source OAI environment
    source oaienv
    
    # Clean if requested
    if [[ "$CLEAN_BUILD" = true ]]; then
        cd cmake_targets
        ./build_oai -c
        cd ..
    fi
    
    # Build gNB (5G base station)
    log_info "Building gNB..."
    cd cmake_targets
    
    BUILD_ARGS="-w USRP --ninja"
    
    if [[ "$WITH_BLADERF" = true ]]; then
        BUILD_ARGS="$BUILD_ARGS --bladerf"
    fi
    
    ./build_oai $BUILD_ARGS --gNB -c
    
    # Build NR UE
    log_info "Building NR UE..."
    ./build_oai $BUILD_ARGS --nrUE -c
    
    # Install
    mkdir -p "$INSTALL_PREFIX/bin"
    mkdir -p "$INSTALL_PREFIX/lib"
    
    # Copy binaries
    cp ran_build/build/nr-softmodem "$INSTALL_PREFIX/bin/" 2>/dev/null || true
    cp ran_build/build/nr-uesoftmodem "$INSTALL_PREFIX/bin/" 2>/dev/null || true
    cp ran_build/build/lte-softmodem "$INSTALL_PREFIX/bin/" 2>/dev/null || true
    cp ran_build/build/lte-uesoftmodem "$INSTALL_PREFIX/bin/" 2>/dev/null || true
    
    # Copy libraries
    cp -r ran_build/build/*.so* "$INSTALL_PREFIX/lib/" 2>/dev/null || true
    
    # Copy configs
    mkdir -p "$INSTALL_PREFIX/etc"
    cp -r ../targets/PROJECTS/GENERIC-NR-5GC/CONF/* "$INSTALL_PREFIX/etc/" 2>/dev/null || true
    cp -r ../ci-scripts/conf_files/* "$INSTALL_PREFIX/etc/" 2>/dev/null || true
    
    log_info "OAI RAN installed to $INSTALL_PREFIX"
}

setup_oai_core() {
    if [[ "$INSTALL_CORE" = false ]]; then
        return
    fi
    
    log_info "Setting up OAI 5G Core Network..."
    
    # Clone CN5G repository
    if [[ ! -d "$INSTALL_CN_PREFIX" ]]; then
        git clone --depth 1 $OAI_CN_REPO "$INSTALL_CN_PREFIX"
    fi
    
    cd "$INSTALL_CN_PREFIX"
    
    # Pull Docker images
    log_info "Pulling OAI Core Network Docker images..."
    
    # Pull latest images
    docker pull oaisoftwarealliance/oai-amf:v1.5.1 || true
    docker pull oaisoftwarealliance/oai-smf:v1.5.1 || true
    docker pull oaisoftwarealliance/oai-spgwu-tiny:v1.5.1 || true
    docker pull oaisoftwarealliance/oai-nrf:v1.5.1 || true
    docker pull oaisoftwarealliance/oai-ausf:v1.5.1 || true
    docker pull oaisoftwarealliance/oai-udm:v1.5.1 || true
    docker pull oaisoftwarealliance/oai-udr:v1.5.1 || true
    docker pull mysql:8.0 || true
    
    log_info "OAI Core Network images pulled"
    
    # Create default docker-compose file
    create_core_compose
    
    log_info "OAI Core Network setup complete at $INSTALL_CN_PREFIX"
}

create_core_compose() {
    log_info "Creating default Core Network configuration..."
    
    mkdir -p "$INSTALL_CN_PREFIX/deploy"
    
    cat > "$INSTALL_CN_PREFIX/deploy/docker-compose.yaml" << 'EOF'
version: '3.8'

services:
  mysql:
    container_name: mysql
    image: mysql:8.0
    environment:
      MYSQL_DATABASE: oai_db
      MYSQL_USER: oai
      MYSQL_PASSWORD: oai_password
      MYSQL_ROOT_PASSWORD: root_password
    volumes:
      - ./mysql:/var/lib/mysql
      - ./oai_db.sql:/docker-entrypoint-initdb.d/oai_db.sql
    healthcheck:
      test: /usr/bin/mysql --user=oai --password=oai_password --execute "SHOW DATABASES;"
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      oai_network:
        ipv4_address: 192.168.70.131

  oai-nrf:
    container_name: oai-nrf
    image: oaisoftwarealliance/oai-nrf:v1.5.1
    environment:
      - NRF_INTERFACE_NAME_FOR_SBI=eth0
      - NRF_INTERFACE_PORT_FOR_SBI=80
      - NRF_INTERFACE_HTTP2_PORT_FOR_SBI=8080
      - NRF_API_VERSION=v1
    networks:
      oai_network:
        ipv4_address: 192.168.70.130
    depends_on:
      - mysql

  oai-amf:
    container_name: oai-amf
    image: oaisoftwarealliance/oai-amf:v1.5.1
    environment:
      - TZ=UTC
      - INSTANCE=0
      - PID_DIRECTORY=/var/run
      - MCC=001
      - MNC=01
      - REGION_ID=128
      - AMF_SET_ID=1
      - SERVED_GUAMI_MCC_0=001
      - SERVED_GUAMI_MNC_0=01
      - SERVED_GUAMI_REGION_ID_0=128
      - SERVED_GUAMI_AMF_SET_ID_0=1
      - PLMN_SUPPORT_MCC=001
      - PLMN_SUPPORT_MNC=01
      - PLMN_SUPPORT_TAC=1
      - SST_0=1
      - SD_0=0xFFFFFF
      - AMF_INTERFACE_NAME_FOR_NGAP=eth0
      - AMF_INTERFACE_NAME_FOR_N11=eth0
      - SMF_INSTANCE_ID_0=1
      - SMF_FQDN_0=oai-smf
      - SMF_IPV4_ADDR_0=192.168.70.133
      - SMF_HTTP_VERSION_0=v1
      - NRF_IPV4_ADDRESS=192.168.70.130
      - NRF_PORT=8080
      - NRF_API_VERSION=v1
      - NF_REGISTRATION=yes
      - SMF_SELECTION=yes
      - USE_FQDN_DNS=no
      - MYSQL_SERVER=192.168.70.131
      - MYSQL_USER=oai
      - MYSQL_PASS=oai_password
      - MYSQL_DB=oai_db
      - OPERATOR_KEY=c42449363bbad02b66d16bc975d77cc1
    depends_on:
      - mysql
      - oai-nrf
    networks:
      oai_network:
        ipv4_address: 192.168.70.132

  oai-smf:
    container_name: oai-smf
    image: oaisoftwarealliance/oai-smf:v1.5.1
    environment:
      - TZ=UTC
      - INSTANCE=0
      - PID_DIRECTORY=/var/run
      - SMF_INTERFACE_NAME_FOR_N4=eth0
      - SMF_INTERFACE_NAME_FOR_SBI=eth0
      - SMF_INTERFACE_PORT_FOR_SBI=80
      - SMF_INTERFACE_HTTP2_PORT_FOR_SBI=8080
      - SMF_API_VERSION=v1
      - DEFAULT_DNS_IPV4_ADDRESS=8.8.8.8
      - DEFAULT_DNS_SEC_IPV4_ADDRESS=8.8.4.4
      - AMF_IPV4_ADDRESS=192.168.70.132
      - AMF_PORT=8080
      - AMF_API_VERSION=v1
      - UPF_IPV4_ADDRESS=192.168.70.134
      - UPF_FQDN_0=oai-spgwu
      - NRF_IPV4_ADDRESS=192.168.70.130
      - NRF_PORT=8080
      - NRF_API_VERSION=v1
      - REGISTER_NRF=yes
      - DISCOVER_UPF=yes
      - USE_FQDN_DNS=no
      - DNN_NI0=oai
      - TYPE0=IPv4
      - DNN_RANGE0=12.1.1.2 - 12.1.1.254
      - NSSAI_SST0=1
      - SESSION_AMBR_UL0=200Mbps
      - SESSION_AMBR_DL0=400Mbps
    depends_on:
      - oai-nrf
      - oai-amf
    networks:
      oai_network:
        ipv4_address: 192.168.70.133

  oai-spgwu:
    container_name: oai-spgwu
    image: oaisoftwarealliance/oai-spgwu-tiny:v1.5.1
    cap_add:
      - NET_ADMIN
      - SYS_ADMIN
    cap_drop:
      - ALL
    privileged: true
    environment:
      - TZ=UTC
      - INSTANCE=0
      - PID_DIRECTORY=/var/run
      - SGW_INTERFACE_NAME_FOR_S1U_S12_S4_UP=eth0
      - SGW_INTERFACE_NAME_FOR_SX=eth0
      - PGW_INTERFACE_NAME_FOR_SGI=eth0
      - NETWORK_UE_NAT_OPTION=yes
      - NETWORK_UE_IP=12.1.1.0/24
      - SPGWC0_IP_ADDRESS=192.168.70.133
      - BYPASS_UL_PFCP_RULES=no
      - MCC=001
      - MNC=01
      - MNC03=001
      - TAC=1
      - GW_ID=1
      - REALM=3gppnetwork.org
      - ENABLE_5G_FEATURES=yes
      - REGISTER_NRF=yes
      - USE_FQDN_NRF=no
      - UPF_FQDN_5G=oai-spgwu
      - NRF_IPV4_ADDRESS=192.168.70.130
      - NRF_PORT=8080
      - NRF_API_VERSION=v1
      - NSSAI_SST_0=1
      - NSSAI_SD_0=0xFFFFFF
      - DNN_0=oai
    depends_on:
      - oai-nrf
      - oai-smf
    networks:
      oai_network:
        ipv4_address: 192.168.70.134

networks:
  oai_network:
    driver: bridge
    ipam:
      config:
        - subnet: 192.168.70.0/24
          gateway: 192.168.70.1
EOF

    # Create MySQL init script
    cat > "$INSTALL_CN_PREFIX/deploy/oai_db.sql" << 'EOF'
-- OAI 5G Core Database Initialization

CREATE DATABASE IF NOT EXISTS oai_db;
USE oai_db;

-- AuthenticationSubscription table
CREATE TABLE IF NOT EXISTS AuthenticationSubscription (
    ueid VARCHAR(20) PRIMARY KEY,
    authenticationMethod VARCHAR(25),
    encPermanentKey VARCHAR(50),
    protectionParameterId VARCHAR(50),
    sequenceNumber TEXT,
    authenticationManagementField VARCHAR(10),
    algorithmId VARCHAR(10),
    encOpcKey VARCHAR(50),
    encTopcKey VARCHAR(50),
    vectorGenerationInHss BOOLEAN,
    n5gcAuthMethod VARCHAR(20),
    rgAuthenticationInd BOOLEAN,
    supiOrSuci VARCHAR(50)
);

-- Default test subscriber
INSERT INTO AuthenticationSubscription 
    (ueid, authenticationMethod, encPermanentKey, protectionParameterId, 
     sequenceNumber, authenticationManagementField, algorithmId, encOpcKey, 
     encTopcKey, vectorGenerationInHss, n5gcAuthMethod, rgAuthenticationInd, supiOrSuci)
VALUES 
    ('001010000000001', '5G_AKA', 'fec86ba6eb707ed08905757b1bb44b8f', 
     'fec86ba6eb707ed08905757b1bb44b8f', 
     '{"sqn": "000000000020", "sqnScheme": "NON_TIME_BASED", "lastIndexes": {"ausf": 0}}',
     '8000', 'milenage', 'C42449363BBAD02B66D16BC975D77CC1', NULL, 
     FALSE, NULL, TRUE, '001010000000001');

-- SessionManagementSubscriptionData
CREATE TABLE IF NOT EXISTS SessionManagementSubscriptionData (
    ueid VARCHAR(20),
    servingPlmnid VARCHAR(10),
    singleNssai TEXT,
    dnnConfigurations TEXT,
    PRIMARY KEY (ueid, servingPlmnid)
);

INSERT INTO SessionManagementSubscriptionData 
    (ueid, servingPlmnid, singleNssai, dnnConfigurations)
VALUES 
    ('001010000000001', '00101', 
     '{"sst": 1, "sd": "FFFFFF"}',
     '{"oai": {"pduSessionTypes": {"defaultSessionType": "IPV4"}, "sscModes": {"defaultSscMode": "SSC_MODE_1"}, "5gQosProfile": {"5qi": 9, "arp": {"priorityLevel": 8, "preemptCap": "NOT_PREEMPT", "preemptVuln": "NOT_PREEMPTABLE"}}, "sessionAmbr": {"uplink": "100Mbps", "downlink": "200Mbps"}}}');

-- AccessAndMobilitySubscriptionData
CREATE TABLE IF NOT EXISTS AccessAndMobilitySubscriptionData (
    ueid VARCHAR(20) PRIMARY KEY,
    servingPlmnid VARCHAR(10),
    supportedFeatures VARCHAR(50),
    gpsis TEXT,
    internalGroupIds TEXT,
    subscribedUeAmbr TEXT,
    nssai TEXT,
    ratRestrictions TEXT,
    forbiddenAreas TEXT,
    serviceAreaRestriction TEXT,
    coreNetworkTypeRestrictions TEXT,
    rfspIndex INT,
    subsRegTimer INT,
    ueUsageType INT,
    mpsPriority BOOLEAN,
    mcsPriority BOOLEAN,
    activeTime INT,
    sorInfo TEXT,
    sorInfoExpectInd BOOLEAN,
    sorafRetrieval BOOLEAN,
    sorUpdateIndicatorList TEXT,
    upuInfo TEXT,
    micoAllowed BOOLEAN,
    sharedAmDataIds TEXT,
    odbPacketServices TEXT,
    serviceGapTime INT,
    mdtUserConsent VARCHAR(20),
    mdtConfiguration TEXT,
    traceData TEXT,
    cagData TEXT,
    stnSr VARCHAR(50),
    cMsisdn VARCHAR(20),
    nbIoTUePriority INT,
    nssaiInclusionAllowed BOOLEAN,
    rgWirelineCharacteristics VARCHAR(10),
    ecRestrictionDataWb TEXT,
    ecRestrictionDataNb BOOLEAN,
    expectedUeBehaviourList TEXT,
    primaryRatRestrictions TEXT,
    secondaryRatRestrictions TEXT,
    edrxParametersList TEXT,
    ptwParametersList TEXT,
    iabOperationAllowed BOOLEAN,
    wirelineServiceAreaRestriction TEXT
);

INSERT INTO AccessAndMobilitySubscriptionData (ueid, servingPlmnid, nssai, subscribedUeAmbr)
VALUES ('001010000000001', '00101', 
        '{"defaultSingleNssais": [{"sst": 1, "sd": "FFFFFF"}]}',
        '{"uplink": "100Mbps", "downlink": "200Mbps"}');
EOF

    log_info "Core Network configuration created"
}

configure_system() {
    log_info "Configuring system..."
    
    # Create directories
    mkdir -p /etc/oai
    mkdir -p /var/log/oai
    
    # Set up PATH
    echo "export PATH=\$PATH:$INSTALL_PREFIX/bin" > /etc/profile.d/oai.sh
    chmod +x /etc/profile.d/oai.sh
    
    # Create symlinks
    for bin in nr-softmodem nr-uesoftmodem lte-softmodem; do
        if [[ -f "$INSTALL_PREFIX/bin/$bin" ]]; then
            ln -sf "$INSTALL_PREFIX/bin/$bin" /usr/local/bin/
        fi
    done
    
    # Configure kernel parameters
    cat > /etc/sysctl.d/99-oai.conf << 'EOF'
# OAI real-time optimizations
net.core.rmem_max=50000000
net.core.wmem_max=50000000
net.core.rmem_default=50000000
net.core.wmem_default=50000000
net.core.netdev_max_backlog=250000
kernel.sched_rt_runtime_us=-1
EOF
    
    sysctl -p /etc/sysctl.d/99-oai.conf 2>/dev/null || true
    
    # Real-time limits
    cat >> /etc/security/limits.conf << 'EOF'
# OAI real-time limits
*               soft    rtprio          99
*               hard    rtprio          99
*               soft    memlock         unlimited
*               hard    memlock         unlimited
EOF
    
    log_info "System configured"
}

verify_installation() {
    log_info "Verifying installation..."
    
    local errors=0
    
    # Check RAN binaries
    if [[ "$INSTALL_RAN" = true ]]; then
        for bin in nr-softmodem nr-uesoftmodem; do
            if [[ -f "$INSTALL_PREFIX/bin/$bin" ]]; then
                log_info "  ✓ $bin found"
            else
                log_warn "  ⚠ $bin not found"
                ((errors++))
            fi
        done
    fi
    
    # Check Core Network
    if [[ "$INSTALL_CORE" = true ]]; then
        if command -v docker &> /dev/null; then
            log_info "  ✓ Docker available"
            
            # Check images
            for img in oai-amf oai-smf oai-spgwu-tiny oai-nrf; do
                if docker images | grep -q "oaisoftwarealliance/$img"; then
                    log_info "  ✓ $img image present"
                else
                    log_warn "  ⚠ $img image not found"
                fi
            done
        else
            log_error "  ✗ Docker not available"
            ((errors++))
        fi
    fi
    
    if [[ $errors -eq 0 ]]; then
        log_info "Installation verified successfully"
        return 0
    else
        log_warn "Installation completed with warnings"
        return 0
    fi
}

print_summary() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║      OpenAirInterface Installation Complete!                 ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Installation Summary:"
    [[ "$INSTALL_RAN" = true ]] && echo "  - OAI RAN: $INSTALL_PREFIX"
    [[ "$INSTALL_CORE" = true ]] && echo "  - OAI Core: $INSTALL_CN_PREFIX"
    echo "  - Configs: /etc/oai/"
    echo "  - Logs:    /var/log/oai/"
    echo ""
    
    if [[ "$INSTALL_CORE" = true ]]; then
        echo "Start 5G Core Network:"
        echo "  cd $INSTALL_CN_PREFIX/deploy"
        echo "  docker-compose up -d"
        echo ""
    fi
    
    if [[ "$INSTALL_RAN" = true ]]; then
        echo "Start 5G gNB:"
        echo "  sudo $INSTALL_PREFIX/bin/nr-softmodem -O /etc/oai/gnb.conf --sa -E"
        echo ""
        echo "Start NR UE:"
        echo "  sudo $INSTALL_PREFIX/bin/nr-uesoftmodem -O /etc/oai/ue.conf --sa -E"
        echo ""
    fi
    
    echo "For RF Arsenal integration:"
    echo "  from core.external.openairinterface import create_oai_controller"
    echo "  ctrl = create_oai_controller()"
    echo "  ctrl.start_full_network()"
    echo ""
}

cleanup() {
    log_info "Cleaning up build files..."
    # Keep source for potential rebuilds
    # rm -rf "$BUILD_DIR"
}

# Main
main() {
    print_header
    check_root
    detect_platform
    
    mkdir -p "$BUILD_DIR"
    
    install_dependencies
    build_oai_ran
    setup_oai_core
    configure_system
    verify_installation
    
    print_summary
}

main "$@"
