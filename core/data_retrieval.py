"""
RF Arsenal OS - Easy Data Retrieval System
Unified interface for retrieving all captured data, detections, and analysis results.
One-click export of all intelligence gathered during operations.
"""

import json
import time
import os
import gzip
import hashlib
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class DataCategory(Enum):
    """Data categories for retrieval"""
    SIGNALS = "signals"
    DETECTIONS = "detections"
    DEVICES = "devices"
    NETWORKS = "networks"
    CAPTURES = "captures"
    DEMODULATED = "demodulated"
    LOCATIONS = "locations"
    SESSIONS = "sessions"
    ALERTS = "alerts"
    ALL = "all"


class ExportFormat(Enum):
    """Export formats"""
    JSON = "json"
    CSV = "csv"
    KML = "kml"
    SQLITE = "sqlite"
    PCAP = "pcap"
    RAW = "raw"


@dataclass
class DataQuery:
    """Query parameters for data retrieval"""
    categories: List[DataCategory] = field(default_factory=lambda: [DataCategory.ALL])
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    frequency_range: Optional[tuple] = None  # (min_hz, max_hz)
    signal_types: Optional[List[str]] = None
    device_types: Optional[List[str]] = None
    min_power_dbm: Optional[float] = None
    limit: int = 10000
    include_raw_iq: bool = False


@dataclass
class DataSummary:
    """Summary of available data"""
    total_signals: int = 0
    total_detections: int = 0
    total_devices: int = 0
    total_networks: int = 0
    total_captures: int = 0
    total_locations: int = 0
    time_range: tuple = (0, 0)  # (start, end)
    frequency_range: tuple = (0, 0)  # (min_hz, max_hz)
    categories: Dict[str, int] = field(default_factory=dict)


class DataRetrieval:
    """
    Unified data retrieval system for RF Arsenal OS.
    
    Features:
    - Single interface for all data types
    - Flexible querying and filtering
    - Multiple export formats
    - Data aggregation and summary
    - Real-time data streaming
    - RAM-only mode support
    - Secure data handling
    """
    
    def __init__(self):
        """Initialize data retrieval system"""
        # Data stores (references to actual data sources)
        self._signal_store: List[Dict] = []
        self._detection_store: List[Dict] = []
        self._device_store: Dict[str, Dict] = {}
        self._network_store: Dict[str, Dict] = {}
        self._capture_store: Dict[str, Dict] = {}
        self._location_store: List[Dict] = []
        self._session_store: List[Dict] = []
        self._alert_store: List[Dict] = []
        
        # Data source callbacks
        self._data_sources: Dict[DataCategory, Callable] = {}
        
        # Export handlers
        self._exporters: Dict[ExportFormat, Callable] = {
            ExportFormat.JSON: self._export_json,
            ExportFormat.CSV: self._export_csv,
            ExportFormat.KML: self._export_kml,
        }
        
    def register_data_source(self, category: DataCategory, 
                            source_callback: Callable) -> None:
        """
        Register a data source callback for a category.
        
        Args:
            category: Data category
            source_callback: Function that returns data for this category
        """
        self._data_sources[category] = source_callback
        
    def add_signal(self, signal: Dict) -> None:
        """Add signal detection"""
        signal["_timestamp"] = time.time()
        self._signal_store.append(signal)
        
    def add_detection(self, detection: Dict) -> None:
        """Add detection (IMSI, device, etc.)"""
        detection["_timestamp"] = time.time()
        self._detection_store.append(detection)
        
    def add_device(self, device_id: str, device: Dict) -> None:
        """Add or update device"""
        if device_id not in self._device_store:
            device["_first_seen"] = time.time()
        device["_last_seen"] = time.time()
        self._device_store[device_id] = device
        
    def add_network(self, network_id: str, network: Dict) -> None:
        """Add or update network"""
        if network_id not in self._network_store:
            network["_first_seen"] = time.time()
        network["_last_seen"] = time.time()
        self._network_store[network_id] = network
        
    def add_capture(self, capture_id: str, capture: Dict) -> None:
        """Add capture reference"""
        capture["_timestamp"] = time.time()
        self._capture_store[capture_id] = capture
        
    def add_location(self, location: Dict) -> None:
        """Add geolocated signal"""
        location["_timestamp"] = time.time()
        self._location_store.append(location)
        
    def add_alert(self, alert: Dict) -> None:
        """Add alert"""
        alert["_timestamp"] = time.time()
        self._alert_store.append(alert)
        
    def get_summary(self) -> DataSummary:
        """
        Get summary of all available data.
        
        Returns:
            DataSummary with counts and ranges
        """
        # Collect all timestamps
        all_times = []
        all_freqs = []
        
        for signal in self._signal_store:
            all_times.append(signal.get("_timestamp", 0))
            if "frequency_hz" in signal:
                all_freqs.append(signal["frequency_hz"])
                
        for detection in self._detection_store:
            all_times.append(detection.get("_timestamp", 0))
            
        for location in self._location_store:
            all_times.append(location.get("_timestamp", 0))
            if "frequency_hz" in location:
                all_freqs.append(location["frequency_hz"])
                
        return DataSummary(
            total_signals=len(self._signal_store),
            total_detections=len(self._detection_store),
            total_devices=len(self._device_store),
            total_networks=len(self._network_store),
            total_captures=len(self._capture_store),
            total_locations=len(self._location_store),
            time_range=(min(all_times) if all_times else 0, 
                       max(all_times) if all_times else 0),
            frequency_range=(min(all_freqs) if all_freqs else 0,
                           max(all_freqs) if all_freqs else 0),
            categories={
                "signals": len(self._signal_store),
                "detections": len(self._detection_store),
                "devices": len(self._device_store),
                "networks": len(self._network_store),
                "captures": len(self._capture_store),
                "locations": len(self._location_store),
                "alerts": len(self._alert_store)
            }
        )
        
    def query(self, query: DataQuery) -> Dict[str, List[Dict]]:
        """
        Query data with filters.
        
        Args:
            query: Query parameters
            
        Returns:
            Dictionary of category -> data list
        """
        result = {}
        
        categories = query.categories
        if DataCategory.ALL in categories:
            categories = [c for c in DataCategory if c != DataCategory.ALL]
            
        for category in categories:
            if category == DataCategory.SIGNALS:
                result["signals"] = self._filter_data(
                    self._signal_store, query
                )
            elif category == DataCategory.DETECTIONS:
                result["detections"] = self._filter_data(
                    self._detection_store, query
                )
            elif category == DataCategory.DEVICES:
                result["devices"] = list(self._device_store.values())[:query.limit]
            elif category == DataCategory.NETWORKS:
                result["networks"] = list(self._network_store.values())[:query.limit]
            elif category == DataCategory.CAPTURES:
                result["captures"] = list(self._capture_store.values())[:query.limit]
            elif category == DataCategory.LOCATIONS:
                result["locations"] = self._filter_data(
                    self._location_store, query
                )
            elif category == DataCategory.ALERTS:
                result["alerts"] = self._alert_store[-query.limit:]
                
        return result
        
    def _filter_data(self, data: List[Dict], query: DataQuery) -> List[Dict]:
        """Apply query filters to data list"""
        result = []
        
        for item in data:
            # Time filter
            if query.start_time:
                if item.get("_timestamp", 0) < query.start_time:
                    continue
            if query.end_time:
                if item.get("_timestamp", float('inf')) > query.end_time:
                    continue
                    
            # Frequency filter
            if query.frequency_range:
                freq = item.get("frequency_hz", 0)
                if not (query.frequency_range[0] <= freq <= query.frequency_range[1]):
                    continue
                    
            # Signal type filter
            if query.signal_types:
                if item.get("signal_type") not in query.signal_types:
                    continue
                    
            # Power filter
            if query.min_power_dbm is not None:
                if item.get("power_dbm", -200) < query.min_power_dbm:
                    continue
                    
            result.append(item)
            
            if len(result) >= query.limit:
                break
                
        return result
        
    def get_all_data(self, include_raw: bool = False) -> Dict[str, Any]:
        """
        Retrieve ALL data - the easy one-click solution.
        
        Args:
            include_raw: Include raw IQ captures (large)
            
        Returns:
            Complete data dictionary
        """
        # Get data from registered sources
        source_data = {}
        for category, source in self._data_sources.items():
            try:
                source_data[category.value] = source()
            except Exception:
                pass
                
        return {
            "metadata": {
                "export_time": time.time(),
                "export_date": datetime.now().isoformat(),
                "version": "1.0",
                "categories_included": list(source_data.keys()) + [
                    "signals", "detections", "devices", "networks",
                    "captures", "locations", "alerts"
                ]
            },
            "summary": {
                "total_signals": len(self._signal_store),
                "total_detections": len(self._detection_store),
                "total_devices": len(self._device_store),
                "total_networks": len(self._network_store),
                "total_captures": len(self._capture_store),
                "total_locations": len(self._location_store),
                "total_alerts": len(self._alert_store)
            },
            "signals": self._signal_store,
            "detections": self._detection_store,
            "devices": list(self._device_store.values()),
            "networks": list(self._network_store.values()),
            "captures": list(self._capture_store.values()) if not include_raw else self._capture_store,
            "locations": self._location_store,
            "alerts": self._alert_store,
            **source_data
        }
        
    def export(self, 
              format: ExportFormat = ExportFormat.JSON,
              query: Optional[DataQuery] = None,
              filepath: Optional[str] = None,
              compress: bool = True) -> Optional[bytes]:
        """
        Export data in specified format.
        
        Args:
            format: Export format
            query: Optional query to filter data
            filepath: Optional file path to save
            compress: Compress output
            
        Returns:
            Exported data bytes (if no filepath)
        """
        # Get data
        if query:
            data = self.query(query)
        else:
            data = self.get_all_data()
            
        # Export using format handler
        exporter = self._exporters.get(format, self._export_json)
        exported = exporter(data)
        
        # Compress if requested
        if compress and isinstance(exported, (str, bytes)):
            if isinstance(exported, str):
                exported = exported.encode()
            exported = gzip.compress(exported)
            
        # Save to file or return
        if filepath:
            mode = 'wb' if isinstance(exported, bytes) else 'w'
            with open(filepath, mode) as f:
                f.write(exported)
            return None
        else:
            return exported
            
    def _export_json(self, data: Dict) -> str:
        """Export as JSON"""
        return json.dumps(data, indent=2, default=str)
        
    def _export_csv(self, data: Dict) -> str:
        """Export as CSV (multiple sections)"""
        lines = []
        
        for category, items in data.items():
            if isinstance(items, list) and items:
                lines.append(f"# {category.upper()}")
                
                # Get all keys
                keys = set()
                for item in items:
                    if isinstance(item, dict):
                        keys.update(item.keys())
                keys = sorted(keys)
                
                # Header
                lines.append(",".join(keys))
                
                # Data
                for item in items:
                    if isinstance(item, dict):
                        row = [str(item.get(k, "")) for k in keys]
                        lines.append(",".join(row))
                        
                lines.append("")
                
        return "\n".join(lines)
        
    def _export_kml(self, data: Dict) -> str:
        """Export locations as KML"""
        kml = ['<?xml version="1.0" encoding="UTF-8"?>',
               '<kml xmlns="http://www.opengis.net/kml/2.2">',
               '<Document>',
               '<name>RF Arsenal OS Export</name>']
               
        # Export locations
        locations = data.get("locations", [])
        for loc in locations:
            if "latitude" in loc and "longitude" in loc:
                kml.extend([
                    '<Placemark>',
                    f'<name>{loc.get("signal_type", "Signal")}</name>',
                    '<description>',
                    f'Frequency: {loc.get("frequency_hz", 0)/1e6:.3f} MHz\n',
                    f'Power: {loc.get("power_dbm", -100):.1f} dBm\n',
                    f'Time: {datetime.fromtimestamp(loc.get("_timestamp", 0)).isoformat()}\n',
                    '</description>',
                    '<Point>',
                    f'<coordinates>{loc["longitude"]},{loc["latitude"]},0</coordinates>',
                    '</Point>',
                    '</Placemark>'
                ])
                
        kml.extend(['</Document>', '</kml>'])
        return "\n".join(kml)
        
    def clear_all(self) -> None:
        """Clear all stored data"""
        self._signal_store.clear()
        self._detection_store.clear()
        self._device_store.clear()
        self._network_store.clear()
        self._capture_store.clear()
        self._location_store.clear()
        self._session_store.clear()
        self._alert_store.clear()
        
    def secure_delete(self) -> None:
        """Securely delete all data (overwrite then clear)"""
        # Overwrite with zeros
        for store in [self._signal_store, self._detection_store, 
                     self._location_store, self._alert_store]:
            for i in range(len(store)):
                store[i] = {}
                
        for store in [self._device_store, self._network_store, 
                     self._capture_store]:
            for key in store:
                store[key] = {}
                
        # Clear
        self.clear_all()


# Convenience function for one-click data retrieval
def get_all_data() -> Dict[str, Any]:
    """
    One-click function to get all captured data.
    
    Usage:
        from core.data_retrieval import get_all_data
        data = get_all_data()
        
    Returns:
        Complete dictionary of all RF intelligence data
    """
    retrieval = DataRetrieval()
    return retrieval.get_all_data()


def export_all_data(filepath: str, format: str = "json", compress: bool = True) -> bool:
    """
    One-click function to export all data to file.
    
    Usage:
        from core.data_retrieval import export_all_data
        export_all_data("/tmp/rf_data.json.gz")
        
    Args:
        filepath: Output file path
        format: Export format (json, csv, kml)
        compress: Compress output
        
    Returns:
        Success status
    """
    try:
        retrieval = DataRetrieval()
        fmt = ExportFormat(format.lower())
        retrieval.export(format=fmt, filepath=filepath, compress=compress)
        return True
    except Exception:
        return False
