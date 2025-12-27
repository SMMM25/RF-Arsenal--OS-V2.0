# RF Arsenal OS - FISSURE Integration

## Professional RF Security GUI

RF Arsenal OS integrates with **FISSURE** (Framework for Integrated SDR-based Signal Understanding and Reverse Engineering) to provide a world-class graphical interface for RF security testing.

### What is FISSURE?

FISSURE is a professional RF assessment framework developed by AIS (Applied Information Sciences) and used by military/government organizations for:

- Signal identification and classification
- Protocol discovery and reverse engineering
- RF attack execution and automation
- Multi-SDR hardware support
- Real-time spectrum analysis

**Repository**: https://github.com/ainfosec/FISSURE

### Why FISSURE + RF Arsenal?

| Feature | Description |
|---------|-------------|
| **Professional GUI** | PyQt5-based dashboard with real-time displays |
| **Attack Library** | Visual attack selection and execution |
| **Hardware Control** | Native BladeRF 2.0 support |
| **Spectrum Display** | Waterfall, FFT, time-domain plots |
| **Flow Graph Editor** | GNU Radio integration for custom signals |
| **TSI Module** | Automatic signal identification |
| **Protocol Discovery** | Reverse engineer unknown protocols |
| **Automation** | Script complex attack sequences |

### Installation

```bash
# Install FISSURE with RF Arsenal integration
cd /opt/rfarsenal
sudo bash install/install_fissure.sh
```

Installation includes:
- ✅ FISSURE framework
- ✅ GNU Radio with gr-osmosdr
- ✅ All RF Arsenal modules integrated
- ✅ Custom attack library (20+ attacks)
- ✅ Desktop launcher

### Launch

```bash
# Launch FISSURE GUI with RF Arsenal
rfarsenal-gui
```

Or manually:
```bash
cd /opt/fissure
python3 fissure_dashboard.py
```

### Available Attacks in FISSURE

#### Cellular
- **2G GSM Base Station** - IMSI catching
- **3G UMTS Base Station** - 3G network deployment
- **4G LTE eNodeB** - LTE base station
- **5G NR gNodeB** - 5G NR deployment

#### WiFi
- **WiFi Deauthentication** - Disconnect clients
- **WiFi Evil Twin** - Rogue access point

#### GPS
- **GPS Spoofing** - Location manipulation
- **GPS Jamming** - L1 signal denial

#### Drone
- **Drone Detection** - Multi-frequency scanning
- **Drone Jamming** - Control signal disruption

#### Jamming
- **Multi-Band Jamming** - Cellular/WiFi/GPS
- **Protocol-Specific Jamming** - Optimized jamming

#### Analysis
- **Spectrum Analyzer** - 70 MHz - 6 GHz
- **SIGINT Collection** - Passive intelligence
- **Protocol Analysis** - Multi-protocol decoding

#### Radar & IoT
- **FMCW Radar** - Target detection
- **IoT Device Scanning** - ZigBee/Z-Wave/LoRa
- **RFID Tag Cloning** - Tag emulation

#### Satellite & Amateur
- **Satellite Tracking** - NOAA/ISS reception
- **Amateur Radio** - Ham radio transceiver

### FISSURE Dashboard Features

#### 1. **TSI (Target Signal Identification)**
Automatically identifies and classifies unknown signals:
```
Signal Detected → Analyze Parameters → Classify Protocol → Display Info
```

#### 2. **PD (Protocol Discovery)**
Reverse engineer unknown wireless protocols:
```
Capture Samples → Extract Features → Decode Structure → Generate Attack
```

#### 3. **FGE (Flow Graph Editor)**
Visual GNU Radio flowgraph editor for custom attacks:
```
Drag & Drop Blocks → Configure Parameters → Execute Attack
```

#### 4. **Attack Library**
Visual attack selection with RF Arsenal modules:
- Category-based organization
- Parameter configuration forms
- Real-time execution status
- Results visualization

#### 5. **Hardware Control**
Direct BladeRF control panel:
- Frequency tuning
- Gain adjustment
- Sample rate selection
- TX/RX switching

#### 6. **Spectrum Display**
Real-time visualization:
- Waterfall plot
- FFT plot
- Time-domain plot
- Signal strength indicators

### Integration Architecture

```
┌─────────────────────────────────────────────────────┐
│              FISSURE Dashboard (PyQt5)              │
│  ┌──────────┬──────────┬──────────┬──────────────┐ │
│  │   TSI    │    PD    │   FGE    │ Attack Lib   │ │
│  └──────────┴──────────┴──────────┴──────────────┘ │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│         RF Arsenal Attack Definitions                │
│        (fissure_attacks.yaml)                       │
│                                                      │
│  • Cellular (2G/3G/4G/5G)                          │
│  • WiFi (Deauth, Evil Twin)                        │
│  • GPS (Spoofing, Jamming)                         │
│  • Drone (Detection, Jamming)                      │
│  • Jamming (Multi-band)                            │
│  • Analysis (Spectrum, SIGINT, Protocol)           │
│  • Radar, IoT, Satellite, Amateur                  │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│           RF Arsenal Modules (Python)                │
│                                                      │
│  modules/                                           │
│    ├── cellular/    (2G/3G/4G/5G)                  │
│    ├── wifi/        (Attack suite)                 │
│    ├── gps/         (Spoofing)                     │
│    ├── drone/       (Warfare)                      │
│    ├── jamming/     (EW suite)                     │
│    ├── spectrum/    (Analyzer)                     │
│    ├── sigint/      (Intelligence)                 │
│    ├── radar/       (Systems)                      │
│    ├── iot/         (RFID/IoT)                     │
│    ├── satellite/   (Comms)                        │
│    ├── amateur/     (Ham radio)                    │
│    └── protocol/    (Analyzer)                     │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│        Hardware Controller (BladeRF)                 │
│                                                      │
│  core/hardware.py                                   │
│    • Device initialization                          │
│    • Frequency control                             │
│    • Gain adjustment                               │
│    • TX/RX switching                               │
│    • Sample streaming                              │
└─────────────────────────────────────────────────────┘
```

### Usage Examples

#### Example 1: WiFi Deauthentication
1. Launch FISSURE GUI
2. Navigate to **Attack Library**
3. Select **WiFi → Deauthentication**
4. Configure parameters:
   - Channel: 6
   - Target BSSID: `AA:BB:CC:DD:EE:FF`
   - Count: 100
5. Click **Execute**
6. Monitor results in real-time

#### Example 2: GPS Spoofing
1. Launch FISSURE GUI
2. Select **GPS → GPS Spoofing**
3. Enter coordinates:
   - Latitude: 51.5074 (London)
   - Longitude: -0.1278
   - Altitude: 100m
4. Click **Start**
5. View spectrum display for GPS L1 transmission

#### Example 3: Spectrum Analysis
1. Launch FISSURE GUI
2. Open **Spectrum Analyzer**
3. Set frequency range: 2.4 GHz - 2.5 GHz
4. Adjust FFT settings
5. View waterfall and FFT plots
6. Identify signals automatically with TSI

#### Example 4: Drone Detection
1. Select **Drone → Drone Detection**
2. Configure scan parameters
3. Click **Detect**
4. FISSURE displays:
   - Detected drones
   - Frequencies
   - Protocol types
   - Signal strength

### Alternative GUI Options

If FISSURE doesn't meet your needs, consider:

#### 1. **GNU Radio Companion (GRC)**
- Visual flowgraph editor
- Perfect for custom signal generation
- Native BladeRF support

```bash
# Install
sudo apt install gnuradio
# Launch
gnuradio-companion
```

#### 2. **Universal Radio Hacker (URH)**
- Protocol analysis focus
- Record/analyze/replay attacks
- Built-in demodulation

```bash
# Install
pip3 install urh
# Launch
urh
```

#### 3. **GQRX**
- Simple spectrum analyzer
- Receive-only
- Great for monitoring

```bash
# Install
sudo apt install gqrx-sdr
# Launch
gqrx
```

#### 4. **SDR++**
- Modern, beautiful UI
- Cross-platform
- Plugin architecture

```bash
# Download from GitHub
# https://github.com/AlexandreRouma/SDRPlusPlus
```

### Custom GUI Development

Want to build your own GUI? Use our modules:

```python
from PyQt5.QtWidgets import QMainWindow
from modules.jamming.jamming_suite import JammingSuite
from core.hardware import HardwareController

class CustomGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.hw = HardwareController()
        self.jammer = JammingSuite(self.hw)
        
    def start_jamming(self):
        self.jammer.jam_band('wifi_2.4', mode='sweep')
```

### Performance Notes

- **RAM Usage**: FISSURE requires ~2-3GB RAM
- **CPU**: Multi-core recommended for real-time processing
- **Disk**: ~5GB for FISSURE + dependencies
- **Display**: 1920x1080 minimum recommended

### Troubleshooting

#### FISSURE won't start
```bash
# Check dependencies
cd /opt/fissure
./install

# Check BladeRF connection
bladeRF-cli -i
```

#### Attacks not appearing
```bash
# Verify RF Arsenal integration
ls /opt/fissure/Custom_Attacks/RF_Arsenal/
# Should show all modules + fissure_attacks.yaml
```

#### BladeRF not detected
```bash
# Check hardware
bladeRF-cli -p

# Update firmware if needed
cd /opt/fissure
python3 fissure_dashboard.py --hardware-check
```

### Screenshots

See the FISSURE GitHub repository for screenshots:
https://github.com/ainfosec/FISSURE#screenshots

### Contributing

To add new attacks to FISSURE integration:

1. Edit `install/fissure_attacks.yaml`
2. Add new attack definition
3. Ensure module path is correct
4. Restart FISSURE

### License Compatibility

- **FISSURE**: GPL-3.0
- **RF Arsenal OS**: MIT
- **Integration**: GPL-3.0 (when distributed with FISSURE)

### Support

- FISSURE Issues: https://github.com/ainfosec/FISSURE/issues
- RF Arsenal Issues: https://github.com/SMMM25/RF-Arsenal-OS/issues

### Credits

- **FISSURE**: AIS (Applied Information Sciences)
- **RF Arsenal OS**: RF security modules and integration
- **GNU Radio**: Signal processing framework
- **BladeRF**: Nuand LLC
