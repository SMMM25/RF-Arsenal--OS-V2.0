#!/bin/bash
#
# RF Arsenal OS - FISSURE Integration Installer
# Integrates RF Arsenal modules with FISSURE GUI
#

set -e

echo "╔════════════════════════════════════════════════════════╗"
echo "║   RF Arsenal OS - FISSURE GUI Integration             ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "[!] Please run as root (use sudo)"
    exit 1
fi

INSTALL_DIR="/opt/rfarsenal"
FISSURE_DIR="/opt/fissure"

echo "[*] Installing FISSURE RF Assessment Framework..."
echo ""

# Install dependencies
echo "[*] Installing system dependencies..."
apt-get update -qq
apt-get install -y \
    git \
    python3-pip \
    python3-pyqt5 \
    python3-pyqt5.qtsvg \
    python3-pyqt5.qtwebkit \
    python3-serial \
    python3-zmq \
    python3-yaml \
    python3-lxml \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    gnuradio \
    gr-osmosdr \
    sox \
    libsox-fmt-all

echo "    ✓ System dependencies installed"

# Clone FISSURE
if [ ! -d "$FISSURE_DIR" ]; then
    echo ""
    echo "[*] Cloning FISSURE repository..."
    git clone https://github.com/ainfosec/FISSURE.git $FISSURE_DIR
    echo "    ✓ FISSURE cloned"
else
    echo ""
    echo "[*] FISSURE already exists, updating..."
    cd $FISSURE_DIR
    git pull
    echo "    ✓ FISSURE updated"
fi

# Install FISSURE
echo ""
echo "[*] Installing FISSURE components..."
cd $FISSURE_DIR
./install

echo "    ✓ FISSURE installed"

# Create RF Arsenal integration directory
echo ""
echo "[*] Integrating RF Arsenal modules..."
mkdir -p $FISSURE_DIR/Custom_Attacks/RF_Arsenal

# Copy RF Arsenal modules to FISSURE
cp -r $INSTALL_DIR/modules/* $FISSURE_DIR/Custom_Attacks/RF_Arsenal/
cp -r $INSTALL_DIR/core $FISSURE_DIR/Custom_Attacks/RF_Arsenal/

echo "    ✓ RF Arsenal modules copied"

# Create FISSURE attack definitions
echo ""
echo "[*] Creating FISSURE attack definitions..."

cat > $FISSURE_DIR/Custom_Attacks/RF_Arsenal/fissure_attacks.yaml << 'EOF'
# RF Arsenal OS - FISSURE Attack Definitions

attacks:
  - name: "2G GSM Base Station"
    category: "Cellular"
    module: "modules.cellular.gsm_2g"
    class: "GSMBaseStation"
    hardware: ["BladeRF"]
    parameters:
      - name: "frequency"
        type: "frequency"
        default: 900000000
      - name: "arfcn"
        type: "integer"
        default: 20
    description: "Deploy 2G GSM base station with IMSI catching"

  - name: "3G UMTS Base Station"
    category: "Cellular"
    module: "modules.cellular.umts_3g"
    class: "UMTSBaseStation"
    hardware: ["BladeRF"]
    parameters:
      - name: "frequency"
        type: "frequency"
        default: 1950000000
    description: "Deploy 3G UMTS base station"

  - name: "4G LTE eNodeB"
    category: "Cellular"
    module: "modules.cellular.lte_4g"
    class: "LTEBaseStation"
    hardware: ["BladeRF"]
    parameters:
      - name: "frequency"
        type: "frequency"
        default: 2140000000
      - name: "bandwidth"
        type: "integer"
        default: 20000000
    description: "Deploy 4G LTE base station"

  - name: "5G NR gNodeB"
    category: "Cellular"
    module: "modules.cellular.nr_5g"
    class: "NRBaseStation"
    hardware: ["BladeRF"]
    parameters:
      - name: "frequency"
        type: "frequency"
        default: 3500000000
      - name: "bandwidth"
        type: "integer"
        default: 100000000
    description: "Deploy 5G NR base station"

  - name: "WiFi Deauthentication"
    category: "WiFi"
    module: "modules.wifi.wifi_attacks"
    class: "WiFiAttackSuite"
    hardware: ["BladeRF"]
    parameters:
      - name: "channel"
        type: "integer"
        default: 6
      - name: "target_bssid"
        type: "string"
        default: "FF:FF:FF:FF:FF:FF"
    description: "Deauthenticate WiFi clients"

  - name: "WiFi Evil Twin"
    category: "WiFi"
    module: "modules.wifi.wifi_attacks"
    class: "WiFiAttackSuite"
    hardware: ["BladeRF"]
    parameters:
      - name: "ssid"
        type: "string"
        default: "FreeWiFi"
      - name: "channel"
        type: "integer"
        default: 6
    description: "Create rogue access point"

  - name: "GPS Spoofing"
    category: "GPS"
    module: "modules.gps.gps_spoofer"
    class: "GPSSpoofer"
    hardware: ["BladeRF"]
    parameters:
      - name: "latitude"
        type: "float"
        default: 37.7749
      - name: "longitude"
        type: "float"
        default: -122.4194
      - name: "altitude"
        type: "float"
        default: 100.0
    description: "Spoof GPS location"

  - name: "GPS Jamming"
    category: "GPS"
    module: "modules.gps.gps_spoofer"
    class: "GPSSpoofer"
    hardware: ["BladeRF"]
    parameters:
      - name: "frequency"
        type: "frequency"
        default: 1575420000
    description: "Jam GPS L1 signals"

  - name: "Drone Detection"
    category: "Drone"
    module: "modules.drone.drone_warfare"
    class: "DroneWarfare"
    hardware: ["BladeRF"]
    parameters:
      - name: "scan_duration"
        type: "float"
        default: 5.0
    description: "Detect drones in area"

  - name: "Drone Jamming"
    category: "Drone"
    module: "modules.drone.drone_warfare"
    class: "DroneWarfare"
    hardware: ["BladeRF"]
    parameters:
      - name: "frequency"
        type: "frequency"
        default: 2400000000
    description: "Jam drone control signals"

  - name: "Multi-Band Jamming"
    category: "Jamming"
    module: "modules.jamming.jamming_suite"
    class: "JammingSuite"
    hardware: ["BladeRF"]
    parameters:
      - name: "band"
        type: "string"
        default: "wifi_2.4"
      - name: "mode"
        type: "string"
        default: "noise"
    description: "Jam specific frequency band"

  - name: "Protocol-Specific Jamming"
    category: "Jamming"
    module: "modules.jamming.jamming_suite"
    class: "JammingSuite"
    hardware: ["BladeRF"]
    parameters:
      - name: "protocol"
        type: "string"
        default: "wifi"
    description: "Protocol-optimized jamming"

  - name: "Spectrum Analyzer"
    category: "Analysis"
    module: "modules.spectrum.spectrum_analyzer"
    class: "SpectrumAnalyzer"
    hardware: ["BladeRF"]
    parameters:
      - name: "start_freq"
        type: "frequency"
        default: 70000000
      - name: "stop_freq"
        type: "frequency"
        default: 6000000000
    description: "Real-time spectrum analysis"

  - name: "SIGINT Collection"
    category: "Intelligence"
    module: "modules.sigint.sigint_engine"
    class: "SIGINTEngine"
    hardware: ["BladeRF"]
    parameters:
      - name: "mode"
        type: "string"
        default: "passive"
      - name: "duration"
        type: "float"
        default: 60.0
    description: "Passive signals intelligence"

  - name: "FMCW Radar"
    category: "Radar"
    module: "modules.radar.radar_systems"
    class: "RadarSystems"
    hardware: ["BladeRF"]
    parameters:
      - name: "frequency"
        type: "frequency"
        default: 2400000000
    description: "FMCW radar system"

  - name: "IoT Device Scanning"
    category: "IoT"
    module: "modules.iot.iot_rfid"
    class: "IoTRFIDSuite"
    hardware: ["BladeRF"]
    parameters:
      - name: "protocol"
        type: "string"
        default: "auto"
    description: "Scan for IoT devices"

  - name: "RFID Tag Cloning"
    category: "IoT"
    module: "modules.iot.iot_rfid"
    class: "IoTRFIDSuite"
    hardware: ["BladeRF"]
    parameters:
      - name: "frequency"
        type: "frequency"
        default: 13560000
    description: "Clone RFID tags"

  - name: "Satellite Tracking"
    category: "Satellite"
    module: "modules.satellite.satcom"
    class: "SatelliteCommunications"
    hardware: ["BladeRF"]
    parameters:
      - name: "satellite"
        type: "string"
        default: "noaa_19"
    description: "Track and receive satellite"

  - name: "Amateur Radio"
    category: "HamRadio"
    module: "modules.amateur.ham_radio"
    class: "AmateurRadio"
    hardware: ["BladeRF"]
    parameters:
      - name: "frequency"
        type: "frequency"
        default: 14200000
      - name: "mode"
        type: "string"
        default: "usb"
    description: "Amateur radio operation"

  - name: "Protocol Analysis"
    category: "Analysis"
    module: "modules.protocol.protocol_analyzer"
    class: "ProtocolAnalyzer"
    hardware: ["BladeRF"]
    parameters:
      - name: "protocol"
        type: "string"
        default: "auto"
    description: "Wireless protocol analyzer"
EOF

echo "    ✓ Attack definitions created"

# Create FISSURE integration script
cat > $FISSURE_DIR/Custom_Attacks/RF_Arsenal/integration.py << 'PYEOF'
#!/usr/bin/env python3
"""
RF Arsenal OS - FISSURE Integration Layer
"""

import sys
import yaml
import importlib
from pathlib import Path

class RFArsenalIntegration:
    """FISSURE integration for RF Arsenal modules"""
    
    def __init__(self):
        self.attacks = self.load_attacks()
        
    def load_attacks(self):
        """Load attack definitions"""
        config_path = Path(__file__).parent / "fissure_attacks.yaml"
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        return data['attacks']
    
    def get_attack_list(self):
        """Get list of available attacks"""
        return [(a['name'], a['category']) for a in self.attacks]
    
    def execute_attack(self, attack_name, parameters):
        """Execute specific attack"""
        attack = next((a for a in self.attacks if a['name'] == attack_name), None)
        
        if not attack:
            raise ValueError(f"Attack {attack_name} not found")
        
        # Import module
        module_path = attack['module']
        class_name = attack['class']
        
        module = importlib.import_module(module_path)
        attack_class = getattr(module, class_name)
        
        # Initialize with hardware controller
        from core.hardware import HardwareController
        hw = HardwareController()
        
        if not hw.connect():
            raise RuntimeError("Failed to connect to BladeRF")
        
        # Create attack instance
        instance = attack_class(hw)
        
        # Configure with parameters
        # (implementation depends on specific module)
        
        return instance

# Export for FISSURE
integration = RFArsenalIntegration()
PYEOF

chmod +x $FISSURE_DIR/Custom_Attacks/RF_Arsenal/integration.py

echo "    ✓ Integration script created"

# Create launcher script
cat > /usr/local/bin/rfarsenal-gui << 'EOF'
#!/bin/bash
# RF Arsenal OS - FISSURE GUI Launcher

cd /opt/fissure
python3 fissure_dashboard.py --custom-attacks /opt/fissure/Custom_Attacks/RF_Arsenal
EOF

chmod +x /usr/local/bin/rfarsenal-gui

echo "    ✓ Launcher created"

# Create desktop entry
cat > /usr/share/applications/rfarsenal-gui.desktop << 'EOF'
[Desktop Entry]
Version=1.0
Type=Application
Name=RF Arsenal OS GUI
Comment=Professional RF Security Testing Platform
Exec=/usr/local/bin/rfarsenal-gui
Icon=/opt/rfarsenal/docs/icon.png
Terminal=false
Categories=Development;Security;
EOF

echo "    ✓ Desktop entry created"

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║       FISSURE Integration Complete!                    ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "✓ FISSURE installed at: $FISSURE_DIR"
echo "✓ RF Arsenal modules integrated"
echo "✓ Custom attack library created"
echo "✓ 20+ attacks available in FISSURE GUI"
echo ""
echo "Launch GUI:"
echo "  rfarsenal-gui"
echo ""
echo "or manually:"
echo "  cd $FISSURE_DIR"
echo "  python3 fissure_dashboard.py"
echo ""
echo "Features available in FISSURE:"
echo "  • Target Signal Identification (TSI)"
echo "  • Protocol Discovery (PD)"
echo "  • Flow Graph Editor (FGE)"
echo "  • Attack Library (your 20+ modules)"
echo "  • Real-time spectrum display"
echo "  • BladeRF hardware control"
echo ""
