# RF Arsenal OS - Geolocation & Tracking

## Overview

Real-time cellular geolocation system for precise target tracking using cellular signals.

## Features

### 🎯 Geolocation Methods

1. **Timing Advance (TA)** - GSM only
   - Distance from tower: TA × 550 meters
   - Accuracy: ±550m
   - Best for GSM 2G/3G networks

2. **RSSI Triangulation**
   - Uses signal strength from multiple capture points
   - Requires 3+ measurements
   - Accuracy: 100m-1km depending on measurements

3. **Cell ID Lookup**
   - Maps Cell ID to tower GPS coordinates
   - Uses OpenCellID/Mozilla Location Services
   - Accuracy: Cell tower range (typically 5km)

### 📡 Cell Tower Database

#### Data Sources

- **OpenCellID.org** - Free API with token (250M+ cells worldwide)
- **Mozilla Location Services** - Free geolocation API
- **Local SQLite cache** - Offline operation support

#### Database Format

```sql
CREATE TABLE cells (
    cell_key TEXT PRIMARY KEY,      -- "MCC-MNC-LAC-CellID"
    mcc INTEGER,                     -- Mobile Country Code
    mnc INTEGER,                     -- Mobile Network Code
    lac INTEGER,                     -- Location Area Code
    cell_id TEXT,                    -- Cell ID
    latitude REAL,                   -- Tower GPS latitude
    longitude REAL,                  -- Tower GPS longitude
    range INTEGER,                   -- Coverage range (meters)
    samples INTEGER,                 -- Measurement count
    updated TIMESTAMP                -- Last update time
);
```

### 🗺️ Real-Time Mapping

#### Interactive HTML Maps

- **OpenStreetMap integration** via Leaflet.js
- **Live target tracking** with colored trails
- **Accuracy circles** showing position uncertainty
- **Heatmaps** of activity patterns
- **Multi-target support** with 10+ simultaneous tracks

#### Export Formats

- **KML** - Google Earth compatible
- **GeoJSON** - Standard geospatial format
- **HTML** - Standalone interactive maps

## Installation

### Dependencies

```bash
# Python packages
pip3 install numpy requests

# Optional: For advanced features
pip3 install folium  # Alternative mapping library
```

### Cell Database Setup

```bash
# Option 1: Use online APIs (no setup required)
# - OpenCellID API (register for free token)
# - Mozilla Location Services (no token needed)

# Option 2: Import OpenCellID CSV database
# Download from: https://opencellid.org/downloads.php
# Import: ~50M cells, ~5GB database size

python3 << EOF
from modules.geolocation.opencellid_integration import OpenCellIDIntegration

opencell = OpenCellIDIntegration(api_token="YOUR_TOKEN_HERE")
opencell.import_csv("cell_towers.csv", limit=1000000)  # Import 1M cells
EOF
```

## Usage

### Basic Geolocation

```python
from modules.geolocation.opencellid_integration import OpenCellIDIntegration

# Initialize
opencell = OpenCellIDIntegration(api_token="YOUR_TOKEN")

# Look up cell tower
result = opencell.lookup_cell(
    mcc=310,      # USA
    mnc=410,      # AT&T
    lac=7033,
    cell_id="20033"
)

if result:
    print(f"Tower location: {result['lat']:.6f}, {result['lon']:.6f}")
    print(f"Coverage range: {result['range']}m")
```

### Real-Time Tracking

```python
from modules.geolocation.cell_triangulation import CellularGeolocation
from modules.geolocation.opencellid_integration import OpenCellIDIntegration

# Initialize geolocation engine
geo = CellularGeolocation()

# Load cell database
geo.cell_database = OpenCellIDIntegration().cell_database

# Track target for 5 minutes
track = geo.track_movement(
    imsi="001010000000001",
    duration=300  # seconds
)

# Export to Google Earth
geo.export_kml("001010000000001", "target_trail.kml")
```

### Interactive Map

```python
from ui.geolocation_map_panel import GeolocationMapPanel

# Create map
map_panel = GeolocationMapPanel()

# Add target
map_panel.add_target("001010000000001", "Target Alpha")

# Update position
map_panel.update_position("001010000000001", {
    'latitude': 37.7749,
    'longitude': -122.4194,
    'accuracy': 100,
    'method': 'timing_advance',
    'timestamp': 1703106000
})

# Generate interactive HTML map
map_panel.generate_html_map("tracking_map.html")
# Open tracking_map.html in browser
```

## Command-Line Interface

### Via AI Controller

```bash
# Start RF Arsenal OS
sudo python3 rf_arsenal_os.py --cli

# Geolocation commands
rf-arsenal> geolocate 001010000000001
rf-arsenal> track 001010000000001 300
rf-arsenal> show map
rf-arsenal> export kml target_trail.kml
```

## Technical Details

### Accuracy Comparison

| Method | Accuracy | Requirements | Speed |
|--------|----------|--------------|-------|
| **Timing Advance** | ±550m | GSM network, TA value | Instant |
| **RSSI Triangulation** | 100m-1km | 3+ measurements | 30s-5min |
| **Cell ID Lookup** | 1-5km | Cell database | Instant |

### Position Calculation

#### Timing Advance Formula

```
Distance = TA × (Speed of Light × Bit Period)
         = TA × (299,792,458 m/s × 3.69 μs)
         ≈ TA × 1,113 meters
         
Simplified: Distance ≈ TA × 550 meters
```

#### RSSI Path Loss Model

```
RSSI = TxPower - 20×log₁₀(distance) - 20×log₁₀(frequency) + constant

Rearranged:
distance = 10^((TxPower - RSSI) / 20)
```

#### Haversine Distance

```python
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    
    φ1 = radians(lat1)
    φ2 = radians(lat2)
    Δφ = radians(lat2 - lat1)
    Δλ = radians(lon2 - lon1)
    
    a = sin²(Δφ/2) + cos(φ1) × cos(φ2) × sin²(Δλ/2)
    c = 2 × atan2(√a, √(1-a))
    
    return R × c
```

## API Reference

### OpenCellIDIntegration

```python
class OpenCellIDIntegration:
    def __init__(api_token: str = None, db_path: str = "data/cells.db")
    def lookup_cell(mcc, mnc, lac, cell_id) -> Dict
    def import_csv(csv_path, limit=None) -> int
    def get_database_stats() -> Dict
    def search_nearby_cells(lat, lon, radius_km=10) -> List[Dict]
```

### CellularGeolocation

```python
class CellularGeolocation:
    def __init__()
    def add_measurement(imsi, measurement: CellMeasurement)
    def calculate_position(imsi) -> Position
    def track_movement(imsi, duration) -> MovementTrack
    def export_kml(imsi, output_path) -> bool
```

### GeolocationMapPanel

```python
class GeolocationMapPanel:
    def __init__(parent=None)
    def add_target(imsi, name=None)
    def update_position(imsi, position: Dict)
    def generate_html_map(output_path) -> bool
    def export_geojson(output_path) -> bool
```

## Performance

### Database Queries

- **Cache lookup**: <1ms (SQLite index)
- **Online API**: 100-500ms (network dependent)
- **CSV import**: ~1M cells/minute

### Tracking Overhead

- **Position updates**: Every 5 seconds
- **Memory usage**: ~1KB per position
- **CPU usage**: <1% on Raspberry Pi 4

## Privacy & Legal

⚠️ **WARNING**: Geolocation tracking is subject to legal restrictions.

- **Authorized testing only** - Requires explicit written permission
- **Comply with local laws** - Different jurisdictions have different rules
- **Data protection** - GDPR, CCPA, and other privacy regulations apply
- **No commercial use** - This is a security research tool

### Relevant Laws

- **USA**: 18 U.S.C. § 2511 (Electronic Communications Privacy Act)
- **EU**: GDPR Article 6 (Lawful basis for processing)
- **UK**: Investigatory Powers Act 2016

## Troubleshooting

### Cell Not Found

```
Issue: Cell ID not in database
Solution:
1. Register for OpenCellID API token
2. Try Mozilla Location Services (no token required)
3. Import local CSV database for offline operation
```

### Low Accuracy

```
Issue: Position accuracy >5km
Solution:
1. Use Timing Advance if available (GSM only)
2. Collect more measurements for triangulation
3. Verify cell database is up-to-date
```

### API Rate Limits

```
Issue: OpenCellID API throttling
Solution:
1. Import CSV database for offline operation
2. Use local cache (automatic)
3. Implement request queuing with delays
```

## Future Enhancements

- [ ] 4G/5G cell ID formats
- [ ] Machine learning position prediction
- [ ] Crowdsourced cell database updates
- [ ] WebSocket real-time streaming
- [ ] 3D terrain-aware calculations
- [ ] Multi-path propagation modeling

## References

- OpenCellID: https://opencellid.org/
- Mozilla Location Services: https://location.services.mozilla.com/
- Leaflet.js: https://leafletjs.com/
- GSM Timing Advance: 3GPP TS 45.010
- LTE Reference Signal Received Power (RSRP): 3GPP TS 36.214

---

**Version**: 1.0.0  
**Last Updated**: 2024-12-21  
**Module**: `modules/geolocation/`
