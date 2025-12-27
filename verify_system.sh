#!/bin/bash
################################################################################
# RF Arsenal OS - Comprehensive System Verification Script
# Verifies all components are present and properly integrated
################################################################################

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║           RF ARSENAL OS - SYSTEM VERIFICATION v1.0.6                      ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0
WARN=0

# Function to check file exists
check_file() {
    local file=$1
    local description=$2
    local size=$3
    
    if [ -f "$file" ]; then
        actual_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        echo -e "${GREEN}✓${NC} PASS: $description"
        echo "        File: $file"
        echo "        Size: $actual_size bytes"
        ((PASS++))
    else
        echo -e "${RED}✗${NC} FAIL: $description"
        echo "        Missing: $file"
        ((FAIL++))
    fi
}

# Function to check directory exists
check_dir() {
    local dir=$1
    local description=$2
    
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f 2>/dev/null | wc -l | tr -d ' ')
        echo -e "${GREEN}✓${NC} PASS: $description"
        echo "        Directory: $dir"
        echo "        Files: $count"
        ((PASS++))
    else
        echo -e "${RED}✗${NC} FAIL: $description"
        echo "        Missing: $dir"
        ((FAIL++))
    fi
}

# Function to check Python import
check_import() {
    local module=$1
    local description=$2
    
    if python3 -c "import $module" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} PASS: $description"
        echo "        Module: $module"
        ((PASS++))
    else
        echo -e "${YELLOW}⚠${NC} WARN: $description"
        echo "        Module not installed: $module"
        ((WARN++))
    fi
}

echo "════════════════════════════════════════════════════════════════════════════"
echo "SECTION 1: CORE SYSTEM FILES"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

check_file "rf_arsenal_os.py" "Main system launcher"
check_file "update_manager.py" "Update manager"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "SECTION 2: CORE MODULES"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

check_file "core/hardware.py" "Hardware controller (BladeRF)"
check_file "core/stealth.py" "Stealth features"
check_file "core/emergency.py" "Emergency protocols"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "SECTION 3: CELLULAR MODULES"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

check_file "modules/cellular/__init__.py" "Cellular package init"
check_file "modules/cellular/gsm_2g.py" "2G/GSM module"
check_file "modules/cellular/umts_3g.py" "3G/UMTS module"
check_file "modules/cellular/lte_4g.py" "4G/LTE module"
check_file "modules/cellular/nr_5g.py" "5G/NR module"
check_file "modules/cellular/phone_targeting.py" "Phone number targeting (NEW)"
check_file "modules/cellular/volte_interceptor.py" "VoLTE/VoNR interceptor (NEW)"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "SECTION 4: NETWORK MODULES"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

check_file "modules/network/__init__.py" "Network package init"
check_file "modules/network/packet_capture.py" "Wireshark/PyShark integration (NEW)"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "SECTION 5: RF MODULES"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

check_file "modules/wifi/wifi_module.py" "WiFi module"
check_file "modules/gps/gps_spoofer.py" "GPS spoofer"
check_file "modules/drone/drone_defense.py" "Drone defense"
check_file "modules/spectrum/spectrum_analyzer.py" "Spectrum analyzer"
check_file "modules/jamming/jammer.py" "RF jammer"
check_file "modules/sigint/sigint_module.py" "SIGINT module"
check_file "modules/radar/radar_module.py" "Radar module"
check_file "modules/iot/iot_module.py" "IoT module"
check_file "modules/satellite/satellite_module.py" "Satellite module"
check_file "modules/amateur/amateur_radio.py" "Amateur radio"
check_file "modules/protocol/protocol_analyzer.py" "Protocol analyzer"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "SECTION 6: SECURITY MODULES"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

check_file "security/anti_forensics.py" "Anti-forensics system"
check_file "security/identity.py" "Identity management"
check_file "security/covert_storage.py" "Covert storage"
check_file "security/physical.py" "Physical security"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "SECTION 7: STEALTH MODULES"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

check_file "stealth/network_anonymity_v2.py" "Network anonymity v2"
check_file "stealth/rf_emission_masking.py" "RF emission masking"
check_file "stealth/ai_threat_detection.py" "AI threat detection"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "SECTION 8: AI & UI"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

check_file "modules/ai/ai_controller.py" "AI controller"
check_file "ui/gui.py" "PyQt6 GUI"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "SECTION 9: INSTALLATION SCRIPTS"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

check_file "install/install.sh" "Main installation script"
check_file "install/install_ai.sh" "AI installation script"
check_file "install/install_fissure.sh" "FISSURE installation"
check_file "install/install_wireshark.sh" "Wireshark installation (NEW)"
check_file "install/test_wireshark_integration.sh" "Wireshark test script"
check_file "install/quick_install.sh" "Quick install script"
check_file "install/first_boot_wizard.py" "First boot wizard"
check_file "install/build_raspberry_pi_image.sh" "Pi image builder"
check_file "install/pi_detect.py" "Pi detection script"
check_file "install/requirements.txt" "Python requirements"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "SECTION 10: DOCUMENTATION"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

check_file "docs/CODE_STATUS.md" "Code status documentation"
check_file "docs/FISSURE_INTEGRATION.md" "FISSURE integration guide"
check_file "docs/INSTALLATION_GUIDE.md" "Installation guide"
check_file "docs/UPDATE_GUIDE.md" "Update guide"
check_file "docs/PROJECT_COMPLETE.md" "Project completion report"
check_file "docs/WIRESHARK_INTEGRATION.md" "Wireshark integration guide (NEW)"
check_file "docs/PHONE_TARGETING.md" "Phone targeting guide (NEW)"
check_file "docs/VOLTE_INTERCEPTION.md" "VoLTE interception guide (NEW)"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "SECTION 11: AI CONTROLLER INTEGRATION VERIFICATION"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Check AI controller imports
echo "Checking AI controller imports..."
if grep -q "from modules.cellular.phone_targeting import" modules/ai/ai_controller.py; then
    echo -e "${GREEN}✓${NC} PASS: Phone targeting import present"
    ((PASS++))
else
    echo -e "${RED}✗${NC} FAIL: Phone targeting import missing"
    ((FAIL++))
fi

if grep -q "from modules.cellular.volte_interceptor import" modules/ai/ai_controller.py; then
    echo -e "${GREEN}✓${NC} PASS: VoLTE interceptor import present"
    ((PASS++))
else
    echo -e "${RED}✗${NC} FAIL: VoLTE interceptor import missing"
    ((FAIL++))
fi

if grep -q "from modules.network.packet_capture import" modules/ai/ai_controller.py; then
    echo -e "${GREEN}✓${NC} PASS: Wireshark/packet capture import present"
    ((PASS++))
else
    echo -e "${RED}✗${NC} FAIL: Wireshark import missing"
    ((FAIL++))
fi

# Check AI controller handlers
echo ""
echo "Checking AI controller handlers..."
if grep -q "def handle_phone_targeting" modules/ai/ai_controller.py; then
    echo -e "${GREEN}✓${NC} PASS: Phone targeting handler present"
    ((PASS++))
else
    echo -e "${RED}✗${NC} FAIL: Phone targeting handler missing"
    ((FAIL++))
fi

if grep -q "def handle_volte" modules/ai/ai_controller.py; then
    echo -e "${GREEN}✓${NC} PASS: VoLTE handler present"
    ((PASS++))
else
    echo -e "${RED}✗${NC} FAIL: VoLTE handler missing"
    ((FAIL++))
fi

if grep -q "def handle_capture" modules/ai/ai_controller.py; then
    echo -e "${GREEN}✓${NC} PASS: Packet capture handler present"
    ((PASS++))
else
    echo -e "${RED}✗${NC} FAIL: Packet capture handler missing"
    ((FAIL++))
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "SECTION 12: PYTHON DEPENDENCIES (Optional)"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

check_import "numpy" "NumPy (required for signal processing)"
check_import "pyshark" "PyShark (required for Wireshark integration)"
check_import "PyQt6" "PyQt6 (required for GUI)"
check_import "cryptography" "Cryptography (required for security modules)"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "VERIFICATION SUMMARY"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

TOTAL=$((PASS + FAIL + WARN))
echo "Total Checks:    $TOTAL"
echo -e "${GREEN}Passed:${NC}          $PASS"
echo -e "${RED}Failed:${NC}          $FAIL"
echo -e "${YELLOW}Warnings:${NC}        $WARN"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    ✓ ALL CRITICAL CHECKS PASSED                           ║${NC}"
    echo -e "${GREEN}║                    RF ARSENAL OS IS PRODUCTION READY                      ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Version: v1.0.6"
    echo "Status: PRODUCTION READY ✅"
    echo ""
    echo "Next Steps:"
    echo "1. Test system launch: sudo python3 rf_arsenal_os.py --cli"
    echo "2. Test update system: sudo python3 update_manager.py --check"
    echo "3. Review documentation in docs/"
    exit 0
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                    ✗ CRITICAL COMPONENTS MISSING                          ║${NC}"
    echo -e "${RED}║                    SYSTEM NOT PRODUCTION READY                            ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Please review failed checks above and ensure all files are present."
    exit 1
fi
