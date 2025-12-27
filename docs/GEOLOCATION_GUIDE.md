# RF Arsenal OS - Geolocation & Tracking Guide

## Overview

Real-time geolocation and movement tracking for cellular targets with interactive map visualization.

---

## Quick Start

### 1. Initialize
```python
from modules.geolocation.cell_triangulation import CellularGeolocation, CellMeasurement
from modules.geolocation.opencellid_integration import OpenCellIDIntegration

geo = CellularGeolocation()
opencell = OpenCellIDIntegration(api_token="YOUR_TOKEN")
```

### 2. Add Measurements
```python
import time

measurement = CellMeasurement(
    timestamp=time.time(),
    cell_id="310-410-12345",
    lac=1000, mcc=310, mnc=410,
    rssi=-75.0, ta=15,
    latitude=37.7749, longitude=-122.4194
)

geo.add_measurement("310410123456789", measurement)
```

### 3. Calculate Position
```python
position = geo.calculate_position("310410123456789")
print(f"Position: {position.latitude:.6f}, {position.longitude:.6f}")
print(f"Accuracy: ±{position.accuracy}m")
print(f"Method: {position.method}")
```

### 4. Track Movement
```python
track = geo.track_movement("310410123456789", duration=300)
print(f"Distance: {track.total_distance:.0f}m")
print(f"Avg Speed: {track.average_speed * 3.6:.1f} km/h")
```

### 5. Visualize
```python
from ui.geolocation_map_panel import GeolocationMapPanel

map_panel = GeolocationMapPanel()
map_panel.add_target("310410123456789", "Target Alpha")

for pos in track.positions:
    map_panel.update_position("310410123456789", {
        'latitude': pos.latitude,
        'longitude': pos.longitude,
        'accuracy': pos.accuracy,
        'method': pos.method,
        'timestamp': pos.timestamp
    })

map_panel.generate_html_map("target_map.html")
```

### 6. Export
```python
geo.export_kml("310410123456789", "track.kml")
map_panel.export_geojson("track.geojson")
```

---

## Positioning Methods

### 1. Timing Advance (TA) - Best Accuracy
- **Accuracy**: ±550m
- **Method**: Distance = TA × 550 meters
- **Network**: GSM 2G/3G only
- **Best for**: Fixed installations

### 2. RSSI Triangulation - Good Accuracy
- **Accuracy**: ±100-500m
- **Requirements**: 3+ measurement points with GPS
- **Method**: Weighted average by signal strength
- **Best for**: Mobile operations

### 3. Cell ID Lookup - Basic Accuracy
- **Accuracy**: ±500-5000m
- **Method**: Tower location from database
- **Requirements**: OpenCellID or Mozilla MLS
- **Best for**: Quick estimates

---

## Cell Database Setup

### Option 1: OpenCellID API (Recommended)
```python
# Get free token: https://opencellid.org/
opencell = OpenCellIDIntegration(api_token="YOUR_TOKEN")

cell_info = opencell.lookup_cell(mcc=310, mnc=410, lac=1000, cell_id="12345")
print(f"Tower: {cell_info['lat']}, {cell_info['lon']}")
```

### Option 2: Import Database (Offline)
```bash
# Download from https://opencellid.org/downloads.php
gunzip cell_towers.csv.gz

# Import
python3 -c "
from modules.geolocation.opencellid_integration import OpenCellIDIntegration
opencell = OpenCellIDIntegration()
opencell.import_csv('cell_towers.csv', limit=100000)
"
```

### Option 3: Mozilla MLS (No Token)
```python
# Automatic fallback, no setup required
opencell = OpenCellIDIntegration()
```

---

## Advanced Usage

### Multi-Target Tracking
```python
targets = ["310410111111111", "310410222222222"]

for imsi in targets:
    geo.start_tracking(imsi)
    map_panel.add_target(imsi, f"Target-{imsi[-4:]}")
```

### Geofencing
```python
def check_geofence(pos, center_lat, center_lon, radius_m):
    distance = geo._haversine_distance(
        pos.latitude, pos.longitude, center_lat, center_lon
    )
    return distance <= radius_m

# Alert on geofence breach
geofence = {'lat': 37.7749, 'lon': -122.4194, 'radius': 500}
if check_geofence(position, geofence['lat'], geofence['lon'], geofence['radius']):
    print("⚠️ Target entered geofence!")
```

### Speed Analysis
```python
positions = geo.tracking_sessions[imsi].positions
if len(positions) >= 2:
    p1, p2 = positions[-2], positions[-1]
    distance = geo._haversine_distance(p1.latitude, p1.longitude, 
                                       p2.latitude, p2.longitude)
    time_diff = p2.timestamp - p1.timestamp
    speed_kmh = (distance / time_diff) * 3.6 if time_diff > 0 else 0
    print(f"Speed: {speed_kmh:.1f} km/h")
```

---

## Map Features

### Interactive HTML Map
- **OpenStreetMap** base layer (CDN)
- **Colored trails** for each target
- **Accuracy circles** showing uncertainty
- **Heatmap** of activity
- **Auto-refresh** every 5 seconds
- **Click targets** to zoom

### Controls
- Toggle trails on/off
- Toggle accuracy circles
- Toggle heatmap
- Target list with zoom
- Real-time statistics

---

## Export Formats

### KML (Google Earth)
```python
geo.export_kml(imsi, "track.kml")
# Open in Google Earth: File → Open → track.kml
```

Features:
- Start/end markers (green/red)
- Movement trail (blue line)
- Timestamps in descriptions

### GeoJSON (GIS Tools)
```python
map_panel.export_geojson("track.geojson")
# Compatible with QGIS, ArcGIS, Mapbox
```

Structure:
```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "properties": {"imsi": "...", "name": "...", "positions_count": 25},
    "geometry": {"type": "LineString", "coordinates": [...]}
  }]
}
```

---

## Accuracy Factors

### Positive ✅
- Urban areas (more towers)
- Multiple measurements
- Strong signal strength
- Line-of-sight to tower

### Negative ❌
- Rural areas (sparse towers)
- Indoor locations
- Weak signals
- Obstacles (buildings, terrain)

---

## Real-World Scenarios

### Executive Tracking
```python
ceo_imsi = "310410987654321"
geo.start_tracking(ceo_imsi)

while True:
    pos = geo.calculate_position(ceo_imsi)
    if pos:
        print(f"CEO: {pos.latitude:.6f}, {pos.longitude:.6f} (±{pos.accuracy}m)")
    time.sleep(300)  # Check every 5 minutes
```

### Asset Recovery
```python
stolen_imsi = "310410555555555"
track = geo.track_movement(stolen_imsi, duration=3600)

stats = geo.get_statistics(stolen_imsi)
print(f"Moved {stats['total_distance_km']:.1f} km")
print(f"Avg speed: {stats['average_speed_kmh']:.1f} km/h")

geo.export_kml(stolen_imsi, "evidence/track.kml")
```

### Competitor Analysis
```python
employee_imsi = "310410777777777"
competitor_hq = {'lat': 37.7749, 'lon': -122.4194, 'radius': 200}

for day in range(7):
    positions = geo.measurements_cache.get(employee_imsi, [])
    for pos in positions:
        if check_geofence(pos, competitor_hq['lat'], 
                         competitor_hq['lon'], competitor_hq['radius']):
            print(f"⚠️ Day {day}: Employee at competitor!")
```

---

## Performance

### Optimization
```python
# Cache reduces API calls
opencell = OpenCellIDIntegration(db_path="data/cells.db")
opencell.lookup_cell(...)  # First: ~500ms (API + cache)
opencell.lookup_cell(...)  # Next: ~5ms (cache only)
```

### Batch Processing
```python
measurements = [CellMeasurement(...), CellMeasurement(...)]
for m in measurements:
    geo.add_measurement(imsi, m)
position = geo.calculate_position(imsi, measurements)
```

---

## Troubleshooting

### No Position Calculated
```python
# Check measurements
measurements = geo.measurements_cache.get(imsi, [])
print(f"Measurements: {len(measurements)}")

# Check cell database
cell_info = opencell.lookup_cell(mcc, mnc, lac, cell_id)
if not cell_info:
    print("Cell not in database")
```

### Low Accuracy
```python
# Use TA if available
if measurement.ta:
    pos = geo.calculate_position_timing_advance(cell_id, ta)

# Collect more measurements for triangulation
# Import more cell towers to database
opencell.import_csv('cell_towers.csv')
```

### Map Not Displaying
```python
# Check targets
print(f"Targets: {len(map_panel.targets)}")

# Verify positions
for imsi, target in map_panel.targets.items():
    print(f"{imsi}: {len(target['positions'])} positions")

# Check browser console (F12) for JavaScript errors
```

---

## Legal & Compliance

⚠️ **CRITICAL: Authorization Required**

### Legal Use Cases ✅
- Authorized security assessments (written permission)
- Law enforcement with warrants
- Corporate asset tracking (company-owned devices)
- Research with informed consent

### Illegal Use Cases ❌
- Unauthorized surveillance
- Stalking or harassment
- Commercial espionage
- Privacy invasion

### Authorization Check
```python
def verify_authorization(imsi):
    """
    Verify legal authorization to track target
    
    Returns True only if:
    - Written authorization exists
    - Target is within scope of engagement
    - Tracking is legally permitted in jurisdiction
    """
    # Implement authorization check
    pass

if not verify_authorization(imsi):
    raise PermissionError("No authorization to track")
```

---

## API Reference

### CellularGeolocation
```python
class CellularGeolocation:
    def add_measurement(imsi: str, measurement: CellMeasurement)
    def calculate_position(imsi: str) -> Position
    def start_tracking(imsi: str)
    def track_movement(imsi: str, duration: float) -> MovementTrack
    def export_kml(imsi: str, path: str) -> bool
    def get_statistics(imsi: str) -> Dict
```

### GeolocationMapPanel
```python
class GeolocationMapPanel:
    def add_target(imsi: str, name: str = None)
    def update_position(imsi: str, position: Dict)
    def generate_html_map(path: str) -> bool
    def export_geojson(path: str) -> bool
    def clear_target(imsi: str)
```

---

## Business Value

### For Executives
- **Visual proof**: "Show me where the CEO went"
- **Metrics**: Distance, speed, time at locations
- **Geofence alerts**: Entered/exited sensitive areas
- **Export**: Google Earth presentations

### For Security Teams
- **Incident response**: Track stolen devices
- **Investigation**: Locate lost assets
- **Compliance**: Audit trails (KML/GeoJSON exports)
- **Evidence**: Timestamped, accuracy-rated data

---

## Next Steps

1. Set up cell database (OpenCellID)
2. Test with authorized devices
3. Integrate with cellular modules
4. Deploy map server for real-time tracking
5. Configure geofencing for alerts

---

**Support**: See `docs/INSTALLATION_GUIDE.md`  
**Security**: See `docs/SECURITY_OPERATIONAL_GUIDELINES.md`  
**Version**: 1.0.8  
**Status**: Production Ready
