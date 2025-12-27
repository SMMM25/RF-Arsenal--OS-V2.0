#!/usr/bin/env python3
"""
RF Arsenal OS - Geolocation Map Panel
Real-time map display with OpenStreetMap
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
import json

from core.anonymization import get_anonymizer

logger = logging.getLogger(__name__)


class GeolocationMapPanel:
    """
    Real-time geolocation map display
    
    Features:
    - OpenStreetMap integration
    - Live target tracking
    - Historical trails
    - Heatmap of activity
    - Export KML for Google Earth
    - Multiple target tracking
    """
    
    def __init__(self, parent=None, anonymize_identifiers: bool = True):
        """Initialize geolocation map panel
        
        Args:
            parent: Parent widget (optional)
            anonymize_identifiers: Automatically anonymize IMSI (default: True)
        """
        self.parent = parent
        self.targets = {}  # IMSI_hash → target data
        self.map_center = (37.7749, -122.4194)  # Default: San Francisco
        self.zoom_level = 12
        
        # 🔐 Centralized anonymization
        self.anonymize_identifiers = anonymize_identifiers
        self.anonymizer = get_anonymizer()
        
        # Map layers
        self.show_cell_towers = True
        self.show_trails = True
        self.show_heatmap = False
        self.show_accuracy_circles = True
        
        logger.info(f"Geolocation map panel initialized (Anonymization: {anonymize_identifiers})")
    
    def add_target(self, imsi: str, name: str = None):
        """
        Add target to tracking
        
        Args:
            imsi: Target IMSI (will be anonymized automatically)
            name: Optional friendly name
        """
        # 🔐 SECURITY FIX: Anonymize IMSI before storage
        imsi_hash = self.anonymizer.anonymize_imsi(imsi) if self.anonymize_identifiers else imsi
        
        if imsi_hash not in self.targets:
            self.targets[imsi_hash] = {
                'imsi': imsi_hash,  # 🔐 Store anonymized version
                'name': name or f"Target-{imsi_hash[-4:]}",
                'positions': [],
                'color': self._get_color(len(self.targets)),
                'visible': True,
                'tracking': False
            }
            logger.info(f"Added target {imsi_hash} to map")
    
    def update_position(self, imsi: str, position: Dict):
        """
        Update target position on map
        
        Args:
            imsi: Target IMSI (will be anonymized automatically)
            position: Position dict with lat, lon, accuracy, etc.
        """
        # 🔐 SECURITY FIX: Anonymize IMSI
        imsi_hash = self.anonymizer.anonymize_imsi(imsi) if self.anonymize_identifiers else imsi
        
        if imsi_hash not in self.targets:
            self.add_target(imsi)  # Will anonymize internally
        
        self.targets[imsi_hash]['positions'].append(position)
        
        # Auto-center map on latest position
        self.map_center = (position['latitude'], position['longitude'])
        
        logger.debug(f"Updated position for {imsi_hash}: "
                    f"{position['latitude']:.6f}, {position['longitude']:.6f}")
    
    def generate_html_map(self, output_path: str) -> bool:
        """
        Generate interactive HTML map using Leaflet.js
        
        Args:
            output_path: Output HTML file path
            
        Returns:
            True if generated successfully
        """
        try:
            html = self._generate_leaflet_html()
            
            with open(output_path, 'w') as f:
                f.write(html)
            
            logger.info(f"Generated HTML map: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"HTML map generation failed: {e}")
            return False
    
    def _generate_leaflet_html(self) -> str:
        """Generate Leaflet.js HTML map"""
        # Prepare targets data as JSON
        targets_json = json.dumps(self.targets, default=str)
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>RF Arsenal OS - Geolocation Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
    
    <style>
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
        #map {{ width: 100%; height: 100vh; }}
        #controls {{
            position: absolute; top: 10px; right: 10px;
            background: white; padding: 15px; border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3); z-index: 1000;
            max-width: 300px;
        }}
        .target-item {{
            margin: 5px 0; padding: 8px; border-left: 4px solid;
            background: #f5f5f5; cursor: pointer;
        }}
        .target-item:hover {{ background: #e0e0e0; }}
        .stat {{ font-size: 12px; color: #666; }}
        h3 {{ margin-top: 0; color: #333; }}
        label {{ display: block; margin: 8px 0; cursor: pointer; }}
        input[type="checkbox"] {{ margin-right: 5px; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div id="controls">
        <h3>🎯 RF Arsenal OS</h3>
        <div id="targets-list"></div>
        <hr>
        <label><input type="checkbox" id="show-trails" checked> Show Trails</label>
        <label><input type="checkbox" id="show-accuracy" checked> Accuracy Circles</label>
        <label><input type="checkbox" id="show-heatmap"> Heatmap</label>
        <hr>
        <div id="stats"></div>
    </div>
    
    <script>
        var map = L.map('map').setView([{self.map_center[0]}, {self.map_center[1]}], {self.zoom_level});
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap', maxZoom: 19
        }}).addTo(map);
        
        var targets = {targets_json};
        var trailLayers = {{}}, markerLayers = {{}}, accuracyLayers = {{}}, heatmapLayer = null;
        
        function drawTargets() {{
            Object.keys(targets).forEach(function(imsi) {{
                var target = targets[imsi];
                if (!target.visible || target.positions.length === 0) return;
                
                var positions = target.positions;
                var latest = positions[positions.length - 1];
                
                if (document.getElementById('show-trails').checked) {{
                    var trailPoints = positions.map(p => [p.latitude, p.longitude]);
                    if (trailLayers[imsi]) {{
                        trailLayers[imsi].setLatLngs(trailPoints);
                    }} else {{
                        trailLayers[imsi] = L.polyline(trailPoints, {{
                            color: target.color, weight: 3, opacity: 0.7
                        }}).addTo(map);
                    }}
                }}
                
                if (markerLayers[imsi]) {{
                    markerLayers[imsi].setLatLng([latest.latitude, latest.longitude]);
                }} else {{
                    var marker = L.circleMarker([latest.latitude, latest.longitude], {{
                        radius: 8, fillColor: target.color, color: '#fff',
                        weight: 2, opacity: 1, fillOpacity: 0.8
                    }}).addTo(map);
                    marker.bindPopup(`<b>${{target.name}}</b><br>IMSI: ${{imsi}}<br>
                        Position: ${{latest.latitude.toFixed(6)}}, ${{latest.longitude.toFixed(6)}}<br>
                        Accuracy: ±${{latest.accuracy}}m<br>Method: ${{latest.method}}<br>
                        Time: ${{new Date(latest.timestamp * 1000).toLocaleString()}}`);
                    markerLayers[imsi] = marker;
                }}
                
                if (document.getElementById('show-accuracy').checked) {{
                    if (accuracyLayers[imsi]) {{
                        accuracyLayers[imsi].setLatLng([latest.latitude, latest.longitude]);
                        accuracyLayers[imsi].setRadius(latest.accuracy);
                    }} else {{
                        accuracyLayers[imsi] = L.circle([latest.latitude, latest.longitude], {{
                            radius: latest.accuracy, color: target.color,
                            fillColor: target.color, fillOpacity: 0.1, weight: 1
                        }}).addTo(map);
                    }}
                }}
            }});
        }}
        
        function drawHeatmap() {{
            if (heatmapLayer) {{ map.removeLayer(heatmapLayer); heatmapLayer = null; }}
            if (!document.getElementById('show-heatmap').checked) return;
            
            var heatPoints = [];
            Object.keys(targets).forEach(function(imsi) {{
                targets[imsi].positions.forEach(function(pos) {{
                    heatPoints.push([pos.latitude, pos.longitude, 0.5]);
                }});
            }});
            
            if (heatPoints.length > 0) {{
                heatmapLayer = L.heatLayer(heatPoints, {{
                    radius: 25, blur: 15, maxZoom: 17
                }}).addTo(map);
            }}
        }}
        
        function updateTargetsList() {{
            var list = document.getElementById('targets-list');
            list.innerHTML = '';
            Object.keys(targets).forEach(function(imsi) {{
                var target = targets[imsi];
                var div = document.createElement('div');
                div.className = 'target-item';
                div.style.borderLeftColor = target.color;
                div.innerHTML = `<strong>${{target.name}}</strong><br>
                    <span class="stat">Positions: ${{target.positions.length}}</span>`;
                div.onclick = function() {{
                    if (target.positions.length > 0) {{
                        var latest = target.positions[target.positions.length - 1];
                        map.setView([latest.latitude, latest.longitude], 15);
                    }}
                }};
                list.appendChild(div);
            }});
        }}
        
        function updateStats() {{
            var totalTargets = Object.keys(targets).length;
            var totalPositions = 0;
            Object.keys(targets).forEach(function(imsi) {{
                totalPositions += targets[imsi].positions.length;
            }});
            document.getElementById('stats').innerHTML = 
                `<strong>Statistics:</strong><br>Targets: ${{totalTargets}}<br>Total Positions: ${{totalPositions}}`;
        }}
        
        document.getElementById('show-trails').addEventListener('change', drawTargets);
        document.getElementById('show-accuracy').addEventListener('change', drawTargets);
        document.getElementById('show-heatmap').addEventListener('change', drawHeatmap);
        
        drawTargets();
        drawHeatmap();
        updateTargetsList();
        updateStats();
        
        setInterval(drawTargets, 5000);
    </script>
</body>
</html>''';
        
        return html
    
    def _get_color(self, index: int) -> str:
        """Get color for target by index"""
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', 
                 '#00FFFF', '#FFA500', '#800080', '#FFC0CB', '#A52A2A']
        return colors[index % len(colors)]
    
    def export_geojson(self, output_path: str) -> bool:
        """Export tracking data to GeoJSON"""
        try:
            features = []
            for imsi, target in self.targets.items():
                if not target['positions']:
                    continue
                coordinates = [[p['longitude'], p['latitude']] for p in target['positions']]
                feature = {
                    'type': 'Feature',
                    'properties': {
                        'imsi': imsi, 'name': target['name'],
                        'color': target['color'],
                        'positions_count': len(target['positions'])
                    },
                    'geometry': {'type': 'LineString', 'coordinates': coordinates}
                }
                features.append(feature)
            
            geojson = {'type': 'FeatureCollection', 'features': features}
            with open(output_path, 'w') as f:
                json.dump(geojson, f, indent=2)
            
            logger.info(f"Exported GeoJSON to {output_path}")
            return True
        except Exception as e:
            logger.error(f"GeoJSON export failed: {e}")
            return False
    
    def clear_target(self, imsi: str):
        """Clear target from map"""
        if imsi in self.targets:
            del self.targets[imsi]
            logger.info(f"Cleared target {imsi}")
    
    def clear_all(self):
        """Clear all targets"""
        self.targets.clear()
        logger.info("Cleared all targets")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("RF Arsenal OS - Geolocation Map Panel Test")
    map_panel = GeolocationMapPanel()
    test_imsi = "001010000000001"
    map_panel.add_target(test_imsi, "Test Target")
    
    import time
    positions = [
        {'latitude': 37.7749, 'longitude': -122.4194, 'accuracy': 100, 'method': 'cell_id', 'timestamp': time.time()},
        {'latitude': 37.7759, 'longitude': -122.4184, 'accuracy': 150, 'method': 'rssi_triangulation', 'timestamp': time.time() + 10}
    ]
    for pos in positions:
        map_panel.update_position(test_imsi, pos)
    
    map_panel.generate_html_map("test_map.html")
    print("[+] Map saved to: test_map.html")
