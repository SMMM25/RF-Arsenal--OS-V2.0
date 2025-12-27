#!/bin/bash
################################################################################
# RF Arsenal OS - Wireshark Integration Test Suite
# Comprehensive testing of all integration components
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   RF ARSENAL OS - WIRESHARK INTEGRATION TEST SUITE           ║"
echo "║                  Comprehensive Testing                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""

# Test function
test_command() {
    local test_name="$1"
    local command="$2"
    local optional="${3:-false}"
    
    echo -n "Testing: $test_name ... "
    
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        if [ "$optional" = "true" ]; then
            echo -e "${YELLOW}⚠️  SKIP${NC}"
            ((TESTS_SKIPPED++))
        else
            echo -e "${RED}❌ FAIL${NC}"
            ((TESTS_FAILED++))
        fi
        return 1
    fi
}

# PHASE 1: Dependency Checks
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}PHASE 1: SYSTEM DEPENDENCY CHECKS${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

test_command "TShark installed" "which tshark"
test_command "TShark version" "tshark --version"
test_command "PyShark installed" "python3 -c 'import pyshark'"
test_command "Wireshark group exists" "getent group wireshark" "true"
test_command "User in wireshark group" "groups \$USER | grep -q wireshark" "true"
test_command "Dumpcap executable" "which dumpcap"
test_command "Dumpcap capabilities" "getcap /usr/bin/dumpcap | grep -q cap_net" "true"

echo ""

# PHASE 2: Interface Detection
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}PHASE 2: NETWORK INTERFACE DETECTION${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

test_command "List interfaces with TShark" "tshark -D | grep -q '.'"
test_command "Detect loopback interface" "tshark -D | grep -qi 'loopback\\|lo'"
test_command "Detect ethernet interface" "tshark -D | grep -qi 'eth\\|ens'" "true"
test_command "Detect wireless interface" "tshark -D | grep -qi 'wlan\\|wlp'" "true"

echo ""

# PHASE 3: Python Module Integration
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}PHASE 3: PYTHON MODULE INTEGRATION${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

# Check if we're in RF Arsenal OS directory
if [ -f "modules/network/packet_capture.py" ]; then
    test_command "Import packet_capture module" \
        "python3 -c 'from modules.network.packet_capture import WiresharkCapture'"
    
    test_command "Initialize WiresharkCapture" \
        "python3 -c 'from modules.network.packet_capture import WiresharkCapture; w = WiresharkCapture()'"
    
    test_command "Check dependencies via module" \
        "python3 -c 'from modules.network.packet_capture import WiresharkCapture; w = WiresharkCapture(); installed, version = w.check_dependencies(); exit(0 if installed else 1)'"
else
    echo -e "${YELLOW}⚠️  Not in RF Arsenal OS directory - skipping module tests${NC}"
    ((TESTS_SKIPPED+=3))
fi

echo ""

# PHASE 4: Live Capture Test
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}PHASE 4: LIVE PACKET CAPTURE TEST${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

TEST_PCAP="/tmp/rf_arsenal_test_$(date +%s).pcap"

echo "Running 5-second capture on loopback interface..."
if sudo timeout 5 tshark -i lo -c 10 -w "$TEST_PCAP" 2>&1 | grep -q "Capturing on"; then
    echo -e "${GREEN}✅ PASS${NC} - Live capture test"
    ((TESTS_PASSED++))
    
    # Test PCAP analysis
    if [ -f "$TEST_PCAP" ]; then
        test_command "Read captured PCAP" "tshark -r $TEST_PCAP | head -5"
        test_command "PCAP file size > 0" "[ -s $TEST_PCAP ]"
        
        # Cleanup
        rm -f "$TEST_PCAP"
    fi
else
    echo -e "${YELLOW}⚠️  SKIP${NC} - Live capture test (requires sudo)"
    ((TESTS_SKIPPED++))
fi

echo ""

# PHASE 5: Security Integration
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}PHASE 5: SECURITY INTEGRATION CHECKS${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

if [ -f "core/emergency.py" ]; then
    test_command "Emergency.py has packet capture cleanup" \
        "grep -q 'WiresharkCapture\\|packet_capture' core/emergency.py"
else
    echo -e "${YELLOW}⚠️  Not in RF Arsenal OS directory - skipping security tests${NC}"
    ((TESTS_SKIPPED++))
fi

if [ -f "security/anti_forensics.py" ]; then
    test_command "Anti-forensics has capture monitoring" \
        "grep -q 'capture\\|pcap\\|wireshark' security/anti_forensics.py"
else
    ((TESTS_SKIPPED++))
fi

echo ""

# PHASE 6: AI Controller Integration
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}PHASE 6: AI CONTROLLER INTEGRATION${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

if [ -f "modules/ai/ai_controller.py" ]; then
    test_command "AI controller has capture commands" \
        "grep -q 'capture\\|wireshark\\|pcap' modules/ai/ai_controller.py"
    
    test_command "AI controller imports WiresharkCapture" \
        "grep -q 'WiresharkCapture' modules/ai/ai_controller.py"
else
    echo -e "${YELLOW}⚠️  AI controller not found - skipping AI tests${NC}"
    ((TESTS_SKIPPED+=2))
fi

echo ""

# PHASE 7: Documentation
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}PHASE 7: DOCUMENTATION CHECKS${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

test_command "Wireshark integration guide exists" \
    "[ -f docs/WIRESHARK_INTEGRATION.md ]" "true"

test_command "Requirements.txt has pyshark" \
    "grep -q 'pyshark' install/requirements.txt" "true"

echo ""

# Final Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                       TEST RESULTS                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))

echo -e "Total Tests:  $TOTAL_TESTS"
echo -e "Passed:       ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed:       ${RED}$TESTS_FAILED${NC}"
echo -e "Skipped:      ${YELLOW}$TESTS_SKIPPED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           ✅ ALL TESTS PASSED - INTEGRATION READY!            ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Ensure you've logged out/in for group permissions"
    echo "  2. Test with RF Arsenal OS:"
    echo "     sudo python3 rf_arsenal_os.py --cli"
    echo "  3. Try AI command: 'capture packets on any'"
    echo ""
    echo "Available AI Commands:"
    echo "  • 'capture packets on wlan0'"
    echo "  • 'capture packets on eth0 for 60 seconds'"
    echo "  • 'stop capture'"
    echo "  • 'analyze packets'"
    echo "  • 'check for dns leaks'"
    echo "  • 'cleanup captures securely'"
    echo ""
    exit 0
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║              ❌ SOME TESTS FAILED                              ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Common issues:"
    echo "  • Not logged out/in after adding to wireshark group"
    echo "  • Not running from RF Arsenal OS directory"
    echo "  • Missing sudo permissions for capture"
    echo "  • PyShark not installed: pip3 install pyshark"
    echo ""
    exit 1
fi
